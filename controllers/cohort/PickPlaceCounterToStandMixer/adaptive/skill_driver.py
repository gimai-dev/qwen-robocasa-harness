"""Development pilot: Qwen semantic target + public-RGB grounding + pose skill.

This is a hybrid controller, not the former Qwen-authored motor-proposal policy.
The first pilot ends after grasp/lift, with the full-task environment outcome
reported independently from any grasp diagnostic.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,math
from itertools import combinations
import numpy as np
from .panda_embodiment import panda_fk
from .pose_skill import pregrasp_targets,plan_pose_segment,tracking_step,choose_parallel_jaw_path,plan_joint3_reconfiguration,outboard_side_offset
from scipy.spatial.transform import Rotation
from .rgb_grounding import ground_dark_object
from .world_anchor import base_pose,world_pose
from .joint_protocol import validate_waypoint_progress
from .grasp_tracking import lost_during_carry


def object_schema():
    point={'anyOf':[{'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':2,'maxItems':2},{'type':'null'}]}
    view={'type':'object','properties':{'visible':{'type':'boolean'},'description':{'type':'string'},'point_2d':point},'required':['visible','description','point_2d'],'additionalProperties':False}
    return {'type':'object','properties':{'target':{'type':'string'},**{v:view for v in ['left','right','wrist']}},'required':['target','left','right','wrist'],'additionalProperties':False}


def make_client(run,grasp_candidate='bowl',yaw_override=None,depth_offset_m=0.,place=False,perception='classical',approach='direct',release_clearance=.10,side_clearance=.245,side_offset=.12,base_backoff=0.,operation='pick_place',joint3_posture=None,approach_height=None,grasp_down_angle=0.,transfer_clearance=None):
    from robocasa_inspect.model_client import QwenClient
    class SkillClient:
        def __init__(self,**kwargs):
            self.held_gap_m=None
            self.grasp_retry_count=0;self.grasp_cost_actions=None;self.carry_start_index=None
            self.action_budget=None
            self.current_depth_offset_m=depth_offset_m;self.attempt_start_actions=0;self.attempt_run=run
            self.side_offset=side_offset
            self.mixer_floor_cache={}
            self.mixer_grounding=None
            self.faucet_frames=[]
            self.pending_path_start=None;self.pending_path_length=0;self.path_queue_prepared=False
            self.joint3_posture_active=None
            self.qwen=QwenClient(**kwargs);self.stage='ground';self.calls=[];self.events=[];self.queue=[];self.index=0;self.ready_calls=0;self.previous=None;self.pending=None;self.executed=[];self.gripper=1.;self.stage_attempts=0;self.baseline=None;self.abort_reason=None;self.target=None;self.last_unloaded=0;self.goal_position=None;self.goal_rotation=None;self.no_progress_count=0;self.stage_path_start=0;self.target_name=None;self.view_attempts=0;self.place_views=[];self.base_world_goal=None;self.base_align_calls=0;self.base_positioned=base_backoff==0
        def verify(self): self.qwen.verify()
        def close(self):
            self.save();self.qwen.close()
        def save(self):
            (run/'skill-evidence.json').write_text(json.dumps({'policy_class':'Qwen semantic planner + RGB tools + numerical pose skills','qwen_calls':len(self.calls),'model_requests':self.calls,'events':self.events,'stage':self.stage,'abort_reason':self.abort_reason,'target':self.target,'pilot_scope':'faucet temperature actuation attempt' if operation=='faucet_turn' else 'faucet closer-view diagnostic; no temperature actuation' if operation=='faucet_view' else 'full pick and place attempt' if place else 'grasp and lift diagnostic; official full task is separately evaluated'},indent=2)+'\n')
        def event(self,kind,**detail):
            self.events.append({'kind':kind,**detail});self.save()
        def query(self,kwargs,instruction,schema,image_roles=None):
            record={'observation_id':kwargs['observation_id'],'instruction':instruction,'response_schema':schema,'status':'started','image_roles':image_roles}
            self.calls.append(record);self.save()
            try:
                response=self.qwen.complete(observation_id=kwargs['observation_id'],system_prompt='Locate objects in the supplied images. Return the requested JSON.',instruction=instruction,public_state={},images=kwargs['images'],response_schema=schema,max_tokens=700,image_roles=image_roles)
                record.update(status='completed',response=response.command,evidence=response.evidence)
            except Exception as error:
                record.update(status='failed',error=type(error).__name__+': '+str(error)[:500]);self.save();raise
            self.save();return response.command
        def queue_path_allowed(self):
            if self.stage in {'retreat','grasp_retry_retreat'}:return False
            if self.stage in {'ready','side_clear','side_lower'} and self.is_mixer:
                return sum(v.get('phase')==self.stage for v in self.mixer_views)>=4
            return True
        def queue_move(self,kwargs):
            if not self.queue_path_allowed():
                return self.move(kwargs,self.queue[self.index],self.stage,intermediate=self.index<len(self.queue)-1)
            if not self.path_queue_prepared:
                # Retain the original first point and select the same short
                # joint spans, using each selected endpoint for future spans.
                selected=[self.queue[self.index]];cursor=self.index+1
                while cursor<len(self.queue):
                    end=cursor
                    while end+1<len(self.queue) and np.max(np.abs(np.asarray(self.queue[end+1])-selected[-1]))<.16:end+=1
                    selected.append(self.queue[end]);cursor=end+1
                self.queue=self.queue[:self.index]+selected
                self.path_queue_prepared=True
            path=self.queue[self.index:self.index+32]
            self.pending_path_start=self.index;self.pending_path_length=len(path)
            return self.move(kwargs,path[-1],self.stage,intermediate=self.index+len(path)<len(self.queue),waypoints=path)
        def resume_path(self,receipt):
            if self.pending_path_start is None:return
            progress=validate_waypoint_progress(receipt.get('waypoint_progress'))
            if progress['total']!=self.pending_path_length:raise ValueError('waypoint progress does not match pending path')
            self.index=self.pending_path_start+progress['completed']
            self.pending_path_start=None;self.pending_path_length=0
        def move(self,kwargs,q,stage,gripper=None,*,intermediate=False,waypoints=None):
            context=kwargs['proposal_audit_context']
            if gripper is not None and abs(gripper-context['current_gripper'])>1e-8 and context['remaining_actions']<16:
                return self.finish(kwargs,'insufficient actions for gripper transition')
            if gripper is not None:self.gripper=gripper
            self.previous=np.array(kwargs['public_state']['state.arm_joint_position']);self.pending=np.array(q)
            tolerance=.025 if intermediate and self.queue and not np.array_equal(q,self.queue[-1]) else .002
            # The recorded mixer views depend on settled camera poses. Keep the
            # existing phase history anchors precise before streaming onward.
            if stage in {'ready','side_clear','side_lower'} and self.is_mixer:
                if sum(v.get('phase')==stage for v in self.mixer_views)<4:tolerance=.002
            path_fields={'waypoints':[[float(x) for x in point] for point in waypoints]} if waypoints is not None else {}
            self.event('command',observation_id=kwargs['observation_id'],stage=stage,q=list(q),gripper=self.gripper,endpoint_tolerance=tolerance,**path_fields)
            return SimpleNamespace(command={'kind':'move_joints','observation_id':kwargs['observation_id'],'endpoint_tolerance':tolerance,**path_fields,'targets':{**{f'joint{i+1}':float(x) for i,x in enumerate(q)},'gripper':self.gripper},'note':f'hybrid pose skill stage={stage}; numerical IK target; Qwen authored semantic target'},evidence={'origin':'numerical_pose_skill','stage':stage,'qwen_calls_so_far':len(self.calls)})
        def finish(self,kwargs,reason):
            self.abort_reason=reason;self.event('finish',reason=reason)
            return SimpleNamespace(command={'kind':'finish','observation_id':kwargs['observation_id'],'note':reason[:320]},evidence={'origin':'hybrid_skill_executor','qwen_calls_so_far':len(self.calls)})
        def retry_grasp(self,kwargs,q):
            remaining=kwargs['proposal_audit_context']['remaining_actions']
            if (not self.is_mixer or self.grasp_retry_count>=1 or self.grasp_cost_actions is None
                    or remaining<self.grasp_cost_actions+16):
                return self.finish(kwargs,'object slipped during carry')
            self.grasp_retry_count+=1
            self.attempt_run=run/f'grasp-retry-{self.grasp_retry_count}'
            self.attempt_run.mkdir(exist_ok=False)
            # One alternate grasp: reobserve the fallen object, then insert10mm
            # deeper. The observed shallow cake grip slipped during withdrawal.
            self.current_depth_offset_m-=.010
            self.queue=[np.asarray(x) for x in reversed(self.executed[self.carry_start_index:])]
            self.index=0;self.path_queue_prepared=False;self.pending_path_start=None
            self.held_gap_m=None;self.stage='grasp_retry_open'
            self.event('grasp_retry',attempt=self.grasp_retry_count,
                       prior_grasp_cost_actions=self.grasp_cost_actions,remaining_actions=remaining,
                       depth_offset_m=self.current_depth_offset_m,return_waypoints=[x.tolist() for x in self.queue])
            return self.move(kwargs,q,self.stage,1.)
        def base_move(self,kwargs,velocity):
            self.pending=None;self.previous=None
            self.event('base_backoff',normalized_velocity=float(velocity),goal_world_m=self.base_world_goal.tolist(),measured_base_world_m=kwargs['public_state']['state.base_position'])
            return SimpleNamespace(command={'kind':'base_action','observation_id':kwargs['observation_id'],'axis':'x','normalized_velocity':float(velocity),'gripper':'hold','note':'Back away from counter to improve arm workspace; measured public base-pose feedback'},evidence={'origin':'numerical_base_position_skill','qwen_calls_so_far':len(self.calls)})
        def refresh_source(self,state):
            target,self.down=base_pose(self.source_world_position,self.source_world_rotation,state)
            self.target=target.tolist()
            self.pre,_=base_pose(self.source_world_position+np.array([0,0,.12]),self.source_world_rotation,state)
            center,_=base_pose(self.source_world_center,self.source_world_rotation,state)
            self.object_center_offset=center-self.target;self.object_center_offset[2]=0.
            self.event('source_reanchored',target_base_m=self.target,source_world_position_m=self.source_world_position.tolist())
        def begin_faucet_turn(self,kwargs,q,state):
            from .faucet_geometry import ground_faucet_hub,temperature_arc
            geometry=ground_faucet_hub(self.faucet_frames,state,self.faucet_initial_ground,run/'faucet-hub',sink_calibration=self.faucet_initial_calibration)
            self.event('faucet_hub_geometry',result=geometry)
            if geometry['status']!='estimated':return self.finish(kwargs,'faucet hub unresolved from recorded public RGB')
            grip=np.asarray(geometry['grip_base_m']);axis=np.asarray(geometry['temperature_axis_base'])
            self.down=np.asarray(geometry['grip_rotation_base'])
            approach_point=np.asarray(geometry['pregrasp_base_m'])
            self.faucet_grip_world=world_pose(grip,self.down,state)
            arc=temperature_arc(geometry['pivot_base_m'],axis,grip,self.down,-30.,5.)
            self.faucet_arc_world=[world_pose(a['target_base_m'],a['target_rotation'],state) for a in arc]
            self.faucet_arc_index=0
            self.event('faucet_contact_plan',approach_base_m=approach_point.tolist(),grip_base_m=grip.tolist(),arc=arc)
            if not self.segment(q,approach_point,self.down,'faucet_pregrasp'):return self.finish(kwargs,self.abort_reason)
            return self.queue_move(kwargs)
        def next_faucet_arc(self,kwargs,q,state):
            if self.faucet_arc_index==len(self.faucet_arc_world):
                self.stage='faucet_release';return self.move(kwargs,q,self.stage,1.)
            point,rotation=base_pose(*self.faucet_arc_world[self.faucet_arc_index],state)
            self.faucet_arc_index+=1
            self.event('faucet_temperature_increment',cumulative_degrees=-5*self.faucet_arc_index)
            if not self.segment(q,point,rotation,'faucet_arc'):return self.finish(kwargs,self.abort_reason)
            return self.queue_move(kwargs)
        def segment(self,q,pos,rot,stage,world_goal=None):
            posture=self.joint3_posture_active if stage in {'posture_clear','side_enter','pregrasp','descend'} else None
            plan=(choose_parallel_jaw_path(q.tolist(),np.asarray(pos).tolist(),np.asarray(rot).tolist()) if posture is None and world_goal is None and stage in {'pregrasp','overhead','side_lower','side_rotate'} and yaw_override is None else plan_pose_segment(q.tolist(),np.asarray(pos).tolist(),np.asarray(rot).tolist(),joint3_posture_rad=posture))
            if plan['status']=='kinematically_reachable' and 'target_rotation' in plan:
                rot=plan['target_rotation'];self.down=np.asarray(rot)
                if stage in {'pregrasp','overhead','side_lower','side_rotate'}:
                    _,self.source_world_rotation=world_pose(pos,rot,self.current_state)
            self.event('pose_plan',stage=stage,plan=plan)
            if plan['status']!='kinematically_reachable':self.abort_reason='unreachable_'+stage;return False
            self.world_goal=world_goal if world_goal is not None else world_pose(pos,rot,self.current_state)
            self.queue=[np.array(w['q']) for w in plan['waypoints']];self.index=0;self.path_queue_prepared=False;self.stage=stage;self.no_progress_count=0
            if world_goal is None:
                self.stage_attempts=0;self.stage_path_start=max(0,len(self.executed)-1);self.base_reanchors=0
            else:self.base_reanchors+=1
            self.goal_position=np.asarray(pos);self.goal_rotation=np.asarray(rot);return True
        def ground_mixer(self,kwargs,state):
            from .sam_circle_destination import ground_circle_destination_views
            if self.mixer_grounding is not None:
                return self.mixer_grounding
            def select_circle(instruction,images,response_schema,image_roles):
                return self.query({**kwargs,'images':images},instruction,response_schema,image_roles=image_roles)
            result=ground_circle_destination_views(self.mixer_views,state,run/'mixer-destination',select_circle,selection_cache=self.mixer_floor_cache)
            self.event('circle_destination',result=result)
            if result['status']=='grounded':self.mixer_grounding=result
            return result
        def start_side_descent(self,kwargs,q,state,rotation):
            self.refresh_source(state)
            side_point=np.array(self.target)+[0,-self.side_offset,side_clearance]
            if approach in {'side_split','side_horizontal'}:
                if not self.segment(q,side_point,rotation,'side_drop'):return self.finish(kwargs,self.abort_reason)
            elif not self.segment(q,side_point,self.down,'side_lower'):return self.finish(kwargs,self.abort_reason)
            return self.queue_move(kwargs)
        def start_recovered_pick(self,kwargs,q,p,rotation,force):
            # The dropped object can move farther out on the counter. Reuse
            # the reached observation line instead of adding another25cm.
            self.side_offset=float(self.target[1]-p[1])
            self.baseline=force.copy();self.executed=[q.tolist()];self.last_unloaded=0
            self.pending=None;self.previous=None
            self.event('reuse_observed_staging',position_base_m=p.tolist(),
                       source_base_m=self.target,side_offset_m=self.side_offset)
            return self.start_side_descent(kwargs,q,kwargs['public_state'],rotation)
        def preflight_mixer(self,kwargs,q,state,rotation):
            from .mixer_active_view import approximate_floor_for_viewing,camera_view_plan
            if self.grasp_retry_count==0:
                self.grasp_observe_world=world_pose(panda_fk(q.tolist()).position_m,rotation,state)
            grounded=self.ground_mixer(kwargs,state)
            if grounded['status']=='grounded':return self.start_side_descent(kwargs,q,state,rotation)
            hint=approximate_floor_for_viewing(grounded,self.mixer_views,state)
            if hint is None:return self.finish(kwargs,'no RGB bowl hint for an active view')
            camera=kwargs['proposal_audit_context']['camera_calibration']['wrist']
            plan=camera_view_plan(q.tolist(),state,camera,hint['target_world_m'])
            self.event('mixer_active_view_plan',hint=hint,plan=plan)
            if plan['status']!='planned':return self.finish(kwargs,plan['reason'])
            self.mixer_view_restore_world=world_pose(plan['restore_position_base_m'],rotation,state)
            if not self.segment(q,plan['position_base_m'],plan['rotation_base'],'mixer_view_turn'):return self.finish(kwargs,self.abort_reason)
            return self.queue_move(kwargs)
        def destination(self,kwargs,q,p,state,context):
            if self.is_mixer:
                grounded=self.ground_mixer(kwargs,state)
                if grounded['status']!='grounded':return self.finish(kwargs,'mixer bowl unresolved from public RGB circle contours')
                floor,_=base_pose(grounded['target_world_m'],np.eye(3),state)
                held=np.asarray(panda_fk(q.tolist()).rotation_matrix)
                jaw=held[:,0].copy();jaw[2]=0.;jaw/=np.linalg.norm(jaw)
                vertical=np.column_stack([jaw,np.cross([0.,0.,-1.],jaw),[0.,0.,-1.]])
                outward=p+np.array([0.,-.20*np.sign(self.side_offset),0.])
                orient_point=outward+np.array([0.,0.,.05])
                orientation_plan=choose_parallel_jaw_path(q.tolist(),orient_point.tolist(),vertical.tolist())
                if orientation_plan['status']!='kinematically_reachable':return self.finish(kwargs,'held-object vertical reorientation unreachable')
                self.down=np.asarray(orientation_plan['target_rotation'])
                self.release_point=floor+np.array([0,0,release_clearance])
                self.release_world_goal=world_pose(self.release_point,self.down,state)
                transit=self.release_point.copy();transit[2]=max(p[2],self.release_point[2],floor[2]+transfer_clearance if transfer_clearance is not None else self.release_point[2])
                self.event('placement_geometry',floor_center_base_m=floor.tolist(),release_point_base_m=self.release_point.tolist(),transit_point_base_m=transit.tolist(),route='vertical rise then transfer above bowl rim' if transfer_clearance is not None else 'low transfer below mixer upper assembly')
                if transfer_clearance is not None:
                    self.mixer_transit_world=world_pose(transit,self.down,state)
                    raised=orient_point.copy();raised[2]=transit[2]
                    self.mixer_raise_world=world_pose(raised,self.down,state)
                    self.mixer_orient_world=world_pose(orient_point,self.down,state)
                    if not self.segment(q,outward,held,'mixer_outward'):return self.finish(kwargs,self.abort_reason)
                elif not self.segment(q,transit,self.down,'transfer'):return self.finish(kwargs,self.abort_reason)
                return self.queue_move(kwargs)
            from .sam_destination import ground_destination_mask
            from .temporal_grounding import temporal_floor_point
            import cv2
            folder=run/'destination';folder.mkdir(exist_ok=True)
            if self.view_attempts>0:
                view={'image':kwargs['images']['wrist'],'calibration':context['camera_calibration']['wrist'],'observation_id':kwargs['observation_id']}
                self.place_views.append(view)
                (folder/f'view-{self.view_attempts}.png').write_bytes(view['image'])
                (folder/f'view-{self.view_attempts}.json').write_text(json.dumps({k:v for k,v in view.items() if k!='image'},indent=2))
            if len(self.place_views)>=2:
                def select_callback(instruction,images,response_schema):
                    roles={'left':'current_wrist_original','right':'current_wrist_mask_overview','wrist':'current_wrist_candidate_gallery'}
                    return self.query({**kwargs,'images':images},instruction,response_schema,image_roles=roles)
                selection=ground_destination_mask(kwargs['images']['wrist'],folder/f'selection-{self.view_attempts}',select_callback)
                self.event('destination_mask',result={k:v for k,v in selection.items() if k!='mask_bytes'})
                if selection.get('mask_path'):
                    mask=cv2.imread(selection['mask_path'],cv2.IMREAD_GRAYSCALE)>0
                    previous=self.place_views[-2];current=self.place_views[-1]
                    grounded=temporal_floor_point(previous['image'],current['image'],previous['calibration'],current['calibration'],mask,state)
                    self.event('temporal_destination',result=grounded)
                    if grounded['status']=='grounded':
                        dest_point=np.array(grounded['target_base_m'])
                        _,release_rotation=pregrasp_targets(dest_point,yaw=math.pi)
                        offset=release_rotation@self.grasp_rotation.T@self.object_center_offset
                        self.release_point=dest_point-offset
                        self.release_point[2]=dest_point[2]+release_clearance
                        self.down=release_rotation
                        self.release_world_goal=world_pose(self.release_point,self.down,state)
                        self.event('placement_geometry',floor_center_base_m=dest_point.tolist(),object_center_offset_source_m=self.object_center_offset.tolist(),object_center_offset_release_m=offset.tolist(),release_point_base_m=self.release_point.tolist())
                        transit=self.release_point.copy();transit[2]=max(p[2],self.release_point[2]+.12)
                        if not self.segment(q,transit,self.down,'transfer'):return self.finish(kwargs,self.abort_reason)
                        return self.queue_move(kwargs)
            if self.view_attempts>=5:return self.finish(kwargs,'destination unresolved after changed views and RGB mask matching')
            self.view_attempts+=1
            view=p.copy()
            if self.view_attempts==1:
                view[0]-=.18;view[1]*=.75;view[2]+=.03
            elif self.view_attempts==2:
                view[0]-=.06;view[1]*=.75;view[2]+=.03
            elif self.view_attempts==3:view+=np.array([.03,.07,.02])
            elif self.view_attempts==4:view[0]-=.10
            else:view[1]+=.15
            self.event('active_view',reason='obtain drawer floor from two known wrist camera poses',viewpoint_base_m=view.tolist())
            if not self.segment(q,view,self.down,'inspect_place'):return self.finish(kwargs,self.abort_reason)
            return self.queue_move(kwargs)
        def complete(self,**kwargs):
            state=kwargs['public_state'];q=np.array(state['state.arm_joint_position']);pose=panda_fk(q.tolist());p=np.array(pose.position_m);R=np.array(pose.rotation_matrix)
            self.current_state=state
            self.resume_path(kwargs['proposal_audit_context'].get('last_execution'))
            if operation=='faucet_turn' and self.stage in {'pregrasp','faucet_shift'}:
                self.faucet_frames.append({'image_bytes':kwargs['images']['wrist'],'calibration':kwargs['proposal_audit_context']['camera_calibration']['wrist']})
            if self.stage in {'ready','side_clear','side_lower'} and self.is_mixer:
                if sum(v.get('phase')==self.stage for v in self.mixer_views)<4:
                    self.mixer_views.append({'image':kwargs['images']['wrist'],'calibration':kwargs['proposal_audit_context']['camera_calibration']['wrist'],'phase':self.stage})
            force=np.array(state['state.end_effector_wrench']['force_n']);context=kwargs['proposal_audit_context'];remaining=context['remaining_actions'] if 'remaining_actions' in context else 900-context['consumed_actions']
            if self.action_budget is None:self.action_budget=remaining
            consumed=self.action_budget-remaining
            self.event('observation',observation_id=kwargs['observation_id'],stage=self.stage,position=p.tolist(),force=force.tolist(),gripper_qpos=state['state.gripper_qpos'],remaining_actions=remaining)
            if remaining<1:return self.finish(kwargs,'episode action budget reached')
            if lost_during_carry(self.stage,self.held_gap_m,state['state.gripper_qpos']):
                self.event('grasp_lost',stage=self.stage,confirmed_gap_m=self.held_gap_m,
                           current_gap_m=abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1]))
                return self.retry_grasp(kwargs,q)
            if self.stage=='grasp_retry_open':
                self.stage='grasp_retry_retreat';self.pending=None;self.stage_attempts=0
                return self.queue_move(kwargs)
            if self.stage=='ground':
                task=json.loads(kwargs['instruction'])['task']
                self.is_mixer='stand mixer' in str(task).lower()
                self.mixer_views=[{'image':kwargs['images']['wrist'],'calibration':context['camera_calibration']['wrist']}]
                if operation in {'faucet_view','faucet_turn'}:
                    from .faucet_grounding import ground_faucet_lever
                    def select_faucet(instruction,images,schema):
                        return self.query({**kwargs,'images':images},instruction,schema)
                    ground=ground_faucet_lever(kwargs['images'],context['camera_calibration'],state,run/'faucet-source',select_faucet)
                    self.faucet_initial_ground=ground
                    self.faucet_initial_calibration=context['camera_calibration']['wrist']
                    coarse={'target':'faucet control lever'}
                    if ground['status']=='grounded':
                        view_target=(np.asarray(ground['target_base_m'])+[-.10,0,.03]).tolist()
                        ground['candidates']=[{'feature':'lever_view','target_base_m':view_target,'grasp_candidate':False}]
                else:
                    coarse=self.query(kwargs,f'Task: {task}\nIdentify the movable source object to pick up and return its name in target. Images are ordered left camera, right camera, wrist camera. For each view, describe its appearance and give its visible center. All points must refer to that same source object. If it is not visible, set visible false and point_2d null. Use relative coordinates 0 through 1000 on each image; top-left [0,0], bottom-right [1000,1000].',object_schema())
                if operation in {'faucet_view','faucet_turn'}:pass
                elif perception=='sam2':
                    if 'ladle' in coarse['target'].lower():
                        from .sam_rgb_grounding import ground_sam_ladle
                        ground=ground_sam_ladle(kwargs['images'],coarse,context['camera_calibration'],state,self.attempt_run/'sam2-source')
                    else:
                        from .sam_source_grounding import ground_sam_source
                        def select_source(instruction,images,response_schema):
                            roles={'left':'single_camera_original','right':'single_camera_original_color_source_crops','wrist':'single_camera_binary_source_masks'}
                            return self.query({**kwargs,'images':images},instruction,response_schema,image_roles=roles)
                        ground=ground_sam_source(kwargs['images'],coarse,context['camera_calibration'],state,self.attempt_run/'sam2-source',select_source)
                else:ground=ground_dark_object(kwargs['images'],coarse,context['camera_calibration'],state)
                self.event('grounding',result=ground)
                if ground['status']!='grounded':return self.finish(kwargs,'RGB grounding unresolved')
                feature='lever_view' if operation in {'faucet_view','faucet_turn'} else ('bowl_center' if any(c['feature']=='bowl_center' for c in ground['candidates']) else 'object_center') if grasp_candidate=='auto' else {'handle':'handle_center','bowl':'bowl_center','object':'object_center'}[grasp_candidate]
                selected=next((c for c in ground['candidates'] if c['feature']==feature),None)
                if selected is None:return self.finish(kwargs,'requested RGB feature unresolved: '+feature)
                if feature in {'handle_center','object_center'} and not selected.get('grasp_candidate'):return self.finish(kwargs,'handle has no supported grasp axis')
                self.target=list(selected['target_base_m']);self.target[2]+=self.current_depth_offset_m;self.target_name=coarse['target']
                if self.is_mixer:self.side_offset=outboard_side_offset(self.target,side_offset)
                self.object_center_offset=np.array(ground.get('object_center_base_m') or self.target)-self.target
                self.object_center_offset[2]=0.
                axis=selected.get('jaw_closing_axis_base',[1.,0.,0.]);self.yaw=math.atan2(axis[1],axis[0]) if yaw_override is None else yaw_override
                _,initial_down=pregrasp_targets(self.target,yaw=self.yaw)
                self.source_world_position,self.source_world_rotation=world_pose(self.target,initial_down,state)
                self.source_world_center,_=world_pose(ground.get('object_center_base_m') or self.target,initial_down,state)
                self.event('grasp_hypothesis',feature=feature,target_base_m=self.target,yaw_rad=self.yaw,grasp_axis_supported=selected.get('grasp_candidate',False),depth_offset_m=self.current_depth_offset_m,scope='bowl-center grasp is an unvalidated development hypothesis' if feature=='bowl_center' else 'RGB mask-supported grasp')
                if self.grasp_retry_count:
                    return self.start_recovered_pick(kwargs,q,p,R,force)
                self.stage='ready'
            if self.stage=='ready':
                ready=q.copy();ready[1]=-.5;ready[3]=-1.5;ready[5]=1.5
                if self.ready_calls<2 or np.max(np.abs(q[[1,3,5]]-ready[[1,3,5]]))>.04:
                    self.ready_calls+=1
                    if self.ready_calls>5:return self.finish(kwargs,'ready posture did not track')
                    return self.move(kwargs,ready,'ready',1.)
                self.baseline=force.copy();self.executed=[q.tolist()]
                if not self.base_positioned:
                    if self.base_world_goal is None:
                        self.base_world_goal,_=world_pose([-base_backoff,0,0],np.eye(3),state)
                    error,_=base_pose(self.base_world_goal,np.eye(3),state)
                    if abs(error[0])>.01:
                        self.base_align_calls+=1
                        if self.base_align_calls>80:return self.finish(kwargs,'base backoff did not reach its measured target')
                        return self.base_move(kwargs,np.sign(error[0])*.25)
                    self.base_positioned=True
                self.refresh_source(state)
                # Begin in a bent, unloaded pose before imposing the grasp orientation.
                approach_point=self.pre.copy()
                if self.is_mixer:
                    # Stage the camera relative to the observed object center;
                    # a jaw-centering correction belongs to the later grasp.
                    source_center,_=base_pose(self.source_world_center,np.eye(3),state)
                    approach_point[:2]=source_center[:2]
                if approach=='overhead':approach_point[2]=max(p[2],self.pre[2])
                if approach in {'side','side_split','side_horizontal'}:
                    approach_point[1]-=self.side_offset;approach_point[2]=max(p[2],self.pre[2]) if approach_height is None else approach_height
                    if not self.segment(q,approach_point,R,'side_clear'):return self.finish(kwargs,self.abort_reason)
                elif not self.segment(q,approach_point,self.down,'overhead' if approach=='overhead' else 'pregrasp'):return self.finish(kwargs,self.abort_reason)
            if self.pending is not None and self.stage not in {'ready','retreat','verify'}:
                tracking=tracking_step(actual_q=q,target_q=self.pending,previous_q=self.previous,force=force,unloaded_force=self.baseline,executed_q=self.executed[self.last_unloaded:])
                self.event('tracking',result=tracking)
                slow=tracking['progress_fraction']<.03 and (tracking['position_error_m']>.005 or tracking['orientation_error_rad']>math.radians(4))
                self.no_progress_count=self.no_progress_count+1 if slow else 0
                if self.no_progress_count>=3:
                    tracking['status']='blocked';tracking['retreat_waypoints']=list(reversed(self.executed[self.stage_path_start:]));tracking['reason']='no measured pose progress for three execution chunks'
                if tracking['status']=='blocked':
                    self.event('blocked',**tracking)
                    self.queue=[np.array(x) for x in tracking['retreat_waypoints']];self.index=0;self.stage='retreat';self.stage_attempts=0;self.abort_reason='contact blocked; reversing executed waypoints';self.pending=None
                else:
                    self.executed.append(q.tolist())
                    if tracking['force_increase_n']<15:self.last_unloaded=len(self.executed)-1
            if self.stage in {'mixer_view_turn','mixer_view_restore','orient','overhead','side_clear','side_lower','side_drop','side_rotate','posture_clear','side_enter','pregrasp','descend','lift','retreat','grasp_retry_retreat','grasp_retry_view','mixer_outward','mixer_orient','mixer_raise','mixer_side','mixer_front','transfer','lower_place','leave','inspect_place','faucet_shift','faucet_pregrasp','faucet_descend','faucet_arc','faucet_leave'}:
                self.stage_attempts+=1
                if self.stage_attempts>65:return self.finish(kwargs,'pose stage stalled')
                # The observed shaft approach settles 3.1 mm from its RGB target,
                # predominantly along the shaft; this remains within its grip region.
                position_tolerance=.005 if self.stage=='faucet_descend' else .003
                use_path=self.queue_path_allowed()
                while self.index<len(self.queue) and (not use_path or self.index==len(self.queue)-1):
                    if self.index==len(self.queue)-1 and self.stage not in {'retreat','grasp_retry_retreat'}:
                        reached=np.linalg.norm(p-self.goal_position)<=position_tolerance and np.linalg.norm(Rotation.from_matrix(self.goal_rotation@R.T).as_rotvec())<=math.radians(3)
                        if self.stage=='posture_clear':reached=reached and abs(q[2]-self.joint3_posture_active)<.03
                    else:reached=np.max(np.abs(q-self.queue[self.index]))<.025
                    if not reached:break
                    self.index+=1
                if self.index<len(self.queue):
                    if use_path:return self.queue_move(kwargs)
                    # Follow local IK waypoints, batching only a short joint span.
                    end=self.index
                    while end+1<len(self.queue) and np.max(np.abs(self.queue[end+1]-q))<.16:end+=1
                    self.index=end
                    return self.move(kwargs,self.queue[end],self.stage,intermediate=end<len(self.queue)-1)
                old=self.stage;self.pending=None
                if old=='retreat':return self.finish(kwargs,self.abort_reason)
                if old=='grasp_retry_retreat':
                    point,rotation=base_pose(*self.grasp_observe_world,state)
                    if not self.segment(q,point,rotation,'grasp_retry_view'):return self.finish(kwargs,self.abort_reason)
                    return self.queue_move(kwargs)
                if old=='grasp_retry_view':
                    self.stage='ground';self.queue=[];self.index=0;self.ready_calls=0
                    self.previous=None;self.pending_path_start=None;self.no_progress_count=0
                    self.stage_attempts=0;self.view_attempts=0;self.place_views=[]
                    self.attempt_start_actions=consumed
                    self.event('grasp_retry_reobserve',attempt=self.grasp_retry_count)
                    return self.complete(**kwargs)
                anchor_position,anchor_rotation=base_pose(*self.world_goal,state)
                if np.linalg.norm(anchor_position-p)>position_tolerance or np.linalg.norm(Rotation.from_matrix(anchor_rotation@R.T).as_rotvec())>math.radians(3):
                    self.event('pose_reanchored',stage=old,old_base_target_m=self.goal_position.tolist(),new_base_target_m=anchor_position.tolist())
                    if not self.segment(q,anchor_position,anchor_rotation,old,world_goal=self.world_goal):return self.finish(kwargs,self.abort_reason)
                    return self.queue_move(kwargs)
                if old!='side_clear':self.down=anchor_rotation
                if old=='faucet_pregrasp':
                    point,rotation=base_pose(*self.faucet_grip_world,state)
                    if not self.segment(q,point,rotation,'faucet_descend'):return self.finish(kwargs,self.abort_reason)
                elif old=='faucet_descend':
                    self.stage='faucet_close';return self.move(kwargs,q,self.stage,0.)
                elif old=='faucet_arc':return self.next_faucet_arc(kwargs,q,state)
                elif old=='faucet_leave':return self.finish(kwargs,'temperature arc executed; evaluate official outcome')
                elif old=='faucet_shift':
                    if operation=='faucet_turn':return self.begin_faucet_turn(kwargs,q,state)
                    evidence=self.query(kwargs,'Inspect the short faucet control lever and the red/blue marks on its hub. Locate the hinge pivot where that short lever rotates. Describe the actual visible hub and return its point in each visible camera, in normalized coordinates 0 through 1000. Mark an unresolved pivot invisible.',object_schema())
                    self.event('faucet_transverse_view',result=evidence)
                    return self.finish(kwargs,'faucet transverse-view diagnostic complete; no temperature actuation attempted')
                elif old=='side_clear':
                    if self.is_mixer and place:return self.preflight_mixer(kwargs,q,state,R)
                    return self.start_side_descent(kwargs,q,state,R)
                elif old=='mixer_view_turn':
                    self.mixer_views.append({'image':kwargs['images']['wrist'],'calibration':context['camera_calibration']['wrist'],'phase':'active_view'})
                    grounded=self.ground_mixer(kwargs,state)
                    if grounded['status']!='grounded':return self.finish(kwargs,'mixer floor unresolved after active camera view')
                    point,rotation=base_pose(*self.mixer_view_restore_world,state)
                    if not self.segment(q,point,rotation,'mixer_view_restore'):return self.finish(kwargs,self.abort_reason)
                elif old=='mixer_view_restore':
                    return self.start_side_descent(kwargs,q,state,R)
                elif old=='side_drop':
                    self.refresh_source(state)
                    if approach=='side_horizontal':
                        jaw=self.down[:,0].copy();jaw[2]=0.;jaw/=np.linalg.norm(jaw)
                        approach_axis=np.cross(jaw,[0.,0.,-1.])
                        if approach_axis[1]*self.side_offset<0:jaw=-jaw;approach_axis=-approach_axis
                        angle=math.radians(grasp_down_angle)
                        approach_axis=math.cos(angle)*approach_axis+np.array([0.,0.,-math.sin(angle)])
                        self.down=np.column_stack([jaw,np.cross(approach_axis,jaw),approach_axis])
                        _,self.source_world_rotation=world_pose(p,self.down,state)
                        self.event('side_grasp_orientation',rotation_base=self.down.tolist(),down_angle_deg=grasp_down_angle,source='observed RGB short jaw axis; approach from outside')
                    if not self.segment(q,p,self.down,'side_rotate'):return self.finish(kwargs,self.abort_reason)
                elif old=='side_lower' and joint3_posture is not None:
                    plan=plan_joint3_reconfiguration(q.tolist(),joint3_posture)
                    self.event('posture_reconfiguration',plan=plan)
                    if plan['status']!='kinematically_reachable':return self.finish(kwargs,'low side posture reconfiguration unreachable')
                    self.joint3_posture_active=joint3_posture
                    self.queue=[np.asarray(w['q']) for w in plan['waypoints']];self.index=0;self.path_queue_prepared=False;self.stage='posture_clear'
                    self.world_goal=world_pose(p,R,state);self.goal_position=p.copy();self.goal_rotation=R.copy()
                    self.stage_attempts=0;self.stage_path_start=max(0,len(self.executed)-1);self.base_reanchors=0
                elif old=='side_rotate' and approach=='side_horizontal':
                    self.refresh_source(state)
                    pre=np.asarray(self.target)-.12*self.down[:,2]
                    if not self.segment(q,pre,self.down,'pregrasp'):return self.finish(kwargs,self.abort_reason)
                elif old in {'side_lower','posture_clear','side_rotate'}:
                    self.refresh_source(state)
                    side_point=np.array(self.target)+[0,0,side_clearance]
                    if not self.segment(q,side_point,self.down,'side_enter'):return self.finish(kwargs,self.abort_reason)
                elif old in {'orient','overhead','side_enter'}:
                    self.refresh_source(state)
                    if old=='side_enter' and side_clearance<.12:
                        self.pre,_=base_pose(self.source_world_position+np.array([0,0,side_clearance]),self.source_world_rotation,state)
                    if not self.segment(q,self.pre,self.down,'pregrasp'):return self.finish(kwargs,self.abort_reason)
                elif old=='pregrasp':
                    if operation in {'faucet_view','faucet_turn'}:
                        evidence=self.query(kwargs,'Inspect the faucet control lever from these closer left, right and wrist views. Locate the hinge pivot where the short lever rotates, not the long water spout. Describe the lever and pivot and return pivot pixel coordinates in each visible view. Coordinates are 0 through 1000. Mark occluded or unresolved pivots invisible.',object_schema())
                        self.event('faucet_close_view',result=evidence)
                        if not self.segment(q,p+np.array([.06,0,0]),R,'faucet_shift'):return self.finish(kwargs,self.abort_reason)
                        return self.queue_move(kwargs)
                    self.refresh_source(state)
                    if not self.segment(q,self.target,self.down,'descend'):return self.finish(kwargs,self.abort_reason)
                elif old=='descend':
                    self.grasp_rotation=R.copy()
                    self.stage='close';self.lift_start=p.tolist();return self.move(kwargs,q,'close',0.)
                elif old=='lift':self.stage='verify'
                elif old=='mixer_outward':
                    orient_point,self.down=base_pose(*self.mixer_orient_world,state)
                    if not self.segment(q,orient_point,self.down,'mixer_orient'):return self.finish(kwargs,self.abort_reason)
                elif old=='mixer_orient':
                    raised,self.down=base_pose(*self.mixer_raise_world,state)
                    if not self.segment(q,raised,self.down,'mixer_raise'):return self.finish(kwargs,self.abort_reason)
                elif old=='mixer_raise':
                    transit,self.down=base_pose(*self.mixer_transit_world,state)
                    side=p.copy();side[0]=min(p[0],transit[0])-.12
                    front=side.copy();front[1]=transit[1]
                    self.mixer_front_world=world_pose(front,self.down,state)
                    self.event('mixer_front_route',side_base_m=side.tolist(),front_base_m=front.tolist(),over_bowl_base_m=transit.tolist())
                    if not self.segment(q,side,self.down,'mixer_side'):return self.finish(kwargs,self.abort_reason)
                elif old=='mixer_side':
                    front,self.down=base_pose(*self.mixer_front_world,state)
                    if not self.segment(q,front,self.down,'mixer_front'):return self.finish(kwargs,self.abort_reason)
                elif old=='mixer_front':
                    transit,self.down=base_pose(*self.mixer_transit_world,state)
                    if not self.segment(q,transit,self.down,'transfer'):return self.finish(kwargs,self.abort_reason)
                elif old=='transfer':
                    self.release_point,self.down=base_pose(*self.release_world_goal,state)
                    if not self.segment(q,self.release_point,self.down,'lower_place'):return self.finish(kwargs,self.abort_reason)
                elif old=='lower_place':
                    self.stage='release';return self.move(kwargs,q,'release',1.)
                elif old=='inspect_place':self.stage='locate_place'
                elif old=='leave':return self.finish(kwargs,'pick and place sequence finished; evaluate official outcome')
                if self.stage not in {'verify','locate_place'}:return self.queue_move(kwargs)
            if self.stage=='faucet_close':
                self.event('faucet_closed',gripper_qpos=state['state.gripper_qpos'])
                return self.next_faucet_arc(kwargs,q,state)
            if self.stage=='faucet_release':
                if not self.segment(q,p-.10*R[:,2],R,'faucet_leave'):return self.finish(kwargs,self.abort_reason)
                return self.queue_move(kwargs)
            if self.stage=='close':
                self.event('closed',gripper_qpos=state['state.gripper_qpos'])
                if not self.segment(q,p+np.array([0,0,.15]),self.down,'lift'):return self.finish(kwargs,self.abort_reason)
                return self.queue_move(kwargs)
            if self.stage=='verify':
                schema={'type':'object','properties':{'held':{'type':'boolean'},'description':{'type':'string'}},'required':['held','description'],'additionalProperties':False}
                check=self.query(kwargs,'Inspect the current three RGB views. Is the target '+str(self.target_name)+' held between the gripper fingers and lifted clear of the surface? Measured finger separation is '+str(abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1]))+' metres; the close action was executed, followed by a 15cm lift. Describe object evidence in the images.',schema)
                self.event('qwen_grasp_visual_check',result=check,end_effector_lift_m=float(p[2]-self.lift_start[2]),gripper_qpos=state['state.gripper_qpos'])
                if check['held']:
                    self.held_gap_m=abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1])
                    self.grasp_cost_actions=consumed-self.attempt_start_actions
                    self.carry_start_index=len(self.executed)-1
                if not place:return self.finish(kwargs,'grasp/lift development pilot complete; full task judged by environment')
                return self.destination(kwargs,q,p,state,context)
            if self.stage=='locate_place':return self.destination(kwargs,q,p,state,context)
            if self.stage=='release':
                if not self.segment(q,p+np.array([0,0,.05 if self.is_mixer else .20]),self.down,'leave'):return self.finish(kwargs,self.abort_reason)
                return self.queue_move(kwargs)
            return self.finish(kwargs,'unexpected skill state '+self.stage)
    return SkillClient


def main():
    from .joint_runner import run_episode
    parser=argparse.ArgumentParser();parser.add_argument('--task',default='PickPlaceCounterToDrawer');parser.add_argument('--run',type=Path,required=True);parser.add_argument('--actions',type=int,default=900);parser.add_argument('--candidate',choices=['bowl','handle','object','auto'],default='bowl');parser.add_argument('--yaw',type=float);parser.add_argument('--depth-offset',type=float,default=0.);parser.add_argument('--place',action='store_true');parser.add_argument('--perception',choices=['classical','sam2'],default='classical')
    parser.add_argument('--transfer-clearance',type=float);parser.add_argument('--grasp-down-angle',type=float,default=0.);parser.add_argument('--approach-height',type=float);parser.add_argument('--joint3-posture',type=float);parser.add_argument('--base-backoff',type=float,default=0.);parser.add_argument('--side-offset',type=float,default=.12);parser.add_argument('--operation',choices=['pick_place','faucet_view','faucet_turn'],default='pick_place');parser.add_argument('--side-clearance',type=float,default=.245);parser.add_argument('--release-clearance',type=float,default=.10);parser.add_argument('--seed',type=int,default=7);parser.add_argument('--approach',choices=['direct','overhead','side','side_split','side_horizontal'],default='direct')
    args=parser.parse_args();result=run_episode(task=args.task,seed=args.seed,run=args.run,max_decisions=180,client_class=make_client(args.run,args.candidate,args.yaw,args.depth_offset,args.place,args.perception,args.approach,args.release_clearance,args.side_clearance,args.side_offset,args.base_backoff,args.operation,args.joint3_posture,args.approach_height,args.grasp_down_angle,args.transfer_clearance),protocol='skills',action_budget=args.actions,prompt_variant='rig')
    evidence=json.loads((args.run/'skill-evidence.json').read_text());result['policy_class']=evidence['policy_class'];result['executor_decisions']=result.pop('model_decisions');result['qwen_model_calls']=evidence['qwen_calls'];result['pilot_scope']=evidence['pilot_scope'];result['action_budget']=args.actions;(args.run/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'requests','receipts'}}),flush=True)
if __name__=='__main__':main()
