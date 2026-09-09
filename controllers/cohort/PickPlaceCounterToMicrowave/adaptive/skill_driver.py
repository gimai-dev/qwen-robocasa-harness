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


def direct_grasp_target(candidate):
    target=np.asarray(candidate['target_base_m'],dtype=float).copy()
    if 'hull_height_bounds_base_m' in candidate:
        # The grip site is 97 mm below the palm. Grip the upper body of a
        # tall object so the palm does not descend into its top surface.
        target[2]=max(target[2],candidate['hull_height_bounds_base_m'][1]-.04)
    return target


def make_client(run,grasp_candidate='bowl',yaw_override=None,depth_offset_m=0.,place=False,perception='classical',approach='direct',release_clearance=.10,side_clearance=.245,side_offset=.12,base_backoff=0.,operation='pick_place',joint3_posture=None,approach_height=None,grasp_down_angle=0.,transfer_clearance=None,*,skill_memory):
    from robocasa_inspect.model_client import QwenClient
    class SkillClient:
        def __init__(self,**kwargs):
            self.pregrasp_distance=.12;self.nominal_pregrasp_distance=.12;self.lift_distance=.15;self.withdraw_distance=.20
            self.selected_memory_id=None;self.sink_front_offset=.12;self.sink_withdraw_distance=.10
            self.source_view_attempts=0
            self.source_history=[];self.targeted_view_done=False;self.navigation_stowed=False;self.stow_calls=0;self.source_support_anchor=None;self.source_support_queries=0;self.source_base_samples=[];self.source_clear_used=False;self.source_reference=None;self.contact_repair_used=False;self.visual_repair_steps=0;self.visual_repair_moves=[];self.arm_view_calls=0;self.arm_view_done=False
            self.receiver_attempt=0;self.receiver_base_goal=None;self.receiver_base_moves=0
            self.destination_anchor=None;self.destination_transit_world=None;self.grasp_views=[]
            (run/"memory-before.json").write_text(json.dumps(skill_memory,indent=2)+"\n")
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
        def select_memory(self,kwargs,task,coarse):
            nonlocal approach,release_clearance,side_clearance,grasp_down_angle,approach_height,transfer_clearance
            sink_destination=place and operation=='pick_place' and 'place it in the sink' in task.lower()
            matched=[c for c in skill_memory['skills'] if c.get('destination_route')=='sink_front_release'] if sink_destination else []
            cards=matched or [c for c in skill_memory['skills'] if 'destination_route' not in c]
            schema={'type':'object','properties':{'skill_id':{'type':'string','enum':[c['id'] for c in cards]},
                    'reason':{'type':'string'}},'required':['skill_id','reason'],'additionalProperties':False}
            choice=self.query(kwargs,'Task: '+task+'\nCurrent source localization: '+json.dumps(coarse)+
                '\nChoose the successful approach most suitable for CURRENT RGB. Direct enters from above; '
                'side_horizontal enters from the side at the recorded angle in degrees. '
                'Memory scene claims are historical hypotheses. The executor re-localizes and replans.\n'+
                json.dumps({'skills':cards,'lessons':skill_memory['lessons']},separators=(',',':')),schema)
            card=next(c for c in cards if c['id']==choice['skill_id'] and c['official_success'] is True)
            params=card['parameters'];self.selected_memory_id=card['id']
            approach=params['approach'];release_clearance=params['release_clearance']
            self.current_depth_offset_m=params['depth_offset_m']
            self.pregrasp_distance=params['pregrasp_distance'];self.nominal_pregrasp_distance=self.pregrasp_distance;self.lift_distance=params['lift_distance']
            self.withdraw_distance=params['withdraw_distance']
            if card.get('destination_route')=='sink_front_release':
                self.sink_front_offset=params['sink_front_offset_m'];self.sink_withdraw_distance=params['sink_withdraw_distance_m']
            if approach=='side_horizontal':
                self.side_offset=params['side_offset_m'];side_clearance=params['side_clearance']
                grasp_down_angle=params['grasp_down_angle']
                approach_height=params.get('approach_height',approach_height)
                transfer_clearance=params.get('transfer_clearance',transfer_clearance)
            self.event('skill_selected',skill_id=card['id'],reason=choice['reason'],parameters=params,
                       parameter_units='lengths in metres; grasp_down_angle in degrees',
                       consumed_memory_runs=[r['source_run'] for r in skill_memory['lessons']])
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
            self.pre,_=base_pose(self.source_world_position+np.array([0,0,self.pregrasp_distance]),self.source_world_rotation,state)
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
        def visual_repair(self,kwargs,q,p,R):
            if self.visual_repair_steps>=4:return self.finish(kwargs,'visual grasp repair exhausted four corrections')
            self.visual_repair_steps+=1
            schema={'type':'object','properties':{'action':{'type':'string','enum':['move','close','abort']},'translation_gripper_m':{'type':'array','items':{'type':'number','minimum':-.05,'maximum':.05},'minItems':3,'maxItems':3},'reason':{'type':'string'}},'required':['action','translation_gripper_m','reason'],'additionalProperties':False}
            choice=self.query(kwargs,'Task: '+self.task_instruction+'. The initial 3D source estimate caused contact before reaching its target. The gripper has withdrawn 8 cm along its approach axis. Inspect the source relative to the two fingers in CURRENT left, right, and wrist images. Choose a small correction or close only if the SOURCE lies between the fingers. Coordinates are in the GRIPPER frame: +X along jaw separation, +Y sideways, +Z forward toward the fingertips (downward for this top grasp). A negative Z withdraws; a positive Z inserts. Translation at most 5 cm per axis. Do not keep pursuing the old 3D estimate. Previous repairs: '+json.dumps(self.visual_repair_moves[-3:]),schema)
            self.event('qwen_visual_grasp_repair',result=choice);self.visual_repair_moves.append(choice)
            if choice['action']=='abort':return self.finish(kwargs,'Qwen could not identify a visual grasp repair')
            if choice['action']=='close':
                self.grasp_views=[{'state':dict(kwargs['public_state']),'images':kwargs['images']}]
                self.grasp_rotation=R.copy();self.lift_start=p.tolist();self.stage='close'
                return self.move(kwargs,q,'close',0.)
            delta=R@np.asarray(choice['translation_gripper_m'])
            if np.linalg.norm(delta)<.001:return self.finish(kwargs,'visual repair proposed no movement')
            if not self.segment(q,p+delta,R,'visual_repair_move'):return self.finish(kwargs,self.abort_reason)
            return self.queue_move(kwargs)
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
            if self.destination_anchor is not None:
                floor,_=base_pose(self.destination_anchor['target_world_m'],np.eye(3),state)
                self.down=np.asarray(panda_fk(q.tolist()).rotation_matrix)
                self.release_point=floor-self.object_center_offset
                self.release_point[2]=floor[2]+release_clearance
                self.release_world_goal=world_pose(self.release_point,self.down,state)
                transit=self.release_point.copy();transit[2]=max(p[2],self.release_point[2])
                # Approach from the robot-facing side before descending inward,
                # keeping the raised palm away from the faucet over the drain.
                front=-floor[:2]/np.linalg.norm(floor[:2])
                transit[:2]+=self.sink_front_offset*front
                self.release_point=transit.copy()
                self.release_world_goal=world_pose(self.release_point,self.down,state)
                withdraw=transit.copy();withdraw[:2]+=self.sink_withdraw_distance*front
                self.destination_transit_world=world_pose(withdraw,self.down,state)
                self.event('placement_geometry',floor_center_base_m=floor.tolist(),release_point_base_m=self.release_point.tolist(),transit_point_base_m=transit.tolist(),withdraw_point_base_m=withdraw.tolist(),sink_front_offset_m=self.sink_front_offset,sink_withdraw_distance_m=self.sink_withdraw_distance,route='release from robot-facing waypoint at reached lift height, then withdraw outward')
                if not self.segment(q,transit,self.down,'transfer'):return self.finish(kwargs,self.abort_reason)
                return self.queue_move(kwargs)
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
            from .mobile_destination import ground_receiver
            coarse,ground=ground_receiver(kwargs,self.task_instruction,run/('receiver-'+str(self.receiver_attempt)),self.query)
            self.event('receiver_grounding',coarse=coarse,result=ground)
            if ground['status']!='grounded':
                if self.receiver_attempt>=3:return self.finish(kwargs,'receiver unresolved after lateral views')
                from .image_servo import _camera_geometry,_pixel_ray_world
                visible=[v for v in ('left','right','wrist') if coarse[v]['visible'] and coarse[v]['point_2d'] is not None]
                if not visible:return self.finish(kwargs,'receiver not visible for directed base search')
                v=visible[0];pixel=np.asarray(coarse[v]['point_2d'])*255/1000
                origin,direction=_pixel_ray_world(pixel,_camera_geometry(context['camera_calibration'][v]))
                bearing,_=base_pose(np.asarray(origin)+np.asarray(direction),np.eye(3),state)
                offset=[0,float(np.sign(bearing[1]) or 1.)*.25,0]
            else:
                floor=np.asarray(ground['target_base_m'])
                if np.linalg.norm(floor[:2])>.65 or abs(floor[1])>.35:
                    offset=[float(np.clip(floor[0]-.45,-.25,.25)),float(np.clip(floor[1],-.30,.30)),0]
                    if self.receiver_attempt>=3:return self.finish(kwargs,'receiver remains outside arm workspace')
                else:
                    self.down=R=np.asarray(panda_fk(q.tolist()).rotation_matrix)
                    self.release_point=floor.copy();self.release_point[2]+=.10
                    self.release_world_goal=world_pose(self.release_point,R,state)
                    transit=self.release_point.copy();transit[2]=max(p[2],self.release_point[2])
                    self.event('placement_geometry',floor_center_base_m=floor.tolist(),release_point_base_m=self.release_point.tolist(),transit_point_base_m=transit.tolist(),route='mobile receiver localization; retain held orientation')
                    if not self.segment(q,transit,R,'transfer'):return self.finish(kwargs,self.abort_reason)
                    return self.queue_move(kwargs)
            self.receiver_base_goal,_=world_pose(offset,np.eye(3),state)
            self.receiver_base_moves=0;self.receiver_attempt+=1;self.stage='receiver_base';self.pending=None
            self.event('receiver_base_plan',offset_base_m=offset,goal_world_m=self.receiver_base_goal.tolist())
            return self.complete(**kwargs)
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
                if self.stage=='lower_place' and self.destination_anchor is not None:
                    release,_=base_pose(*self.release_world_goal,state)
                    if np.linalg.norm(p-release)<.03:
                        self.event('destination_unloaded',reason='grip emptied within 3cm of release point; open and withdraw before official evaluation')
                        self.held_gap_m=None;self.pending=None;self.stage='release'
                        return self.move(kwargs,q,'release',1.)
                return self.retry_grasp(kwargs,q)
            if self.stage=='grasp_retry_open':
                self.stage='grasp_retry_retreat';self.pending=None;self.stage_attempts=0
                return self.queue_move(kwargs)
            if self.stage=='receiver_base':
                error,_=base_pose(self.receiver_base_goal,np.eye(3),state)
                axis=int(np.argmax(np.abs(error[:2])))
                if abs(error[axis])>.02:
                    self.receiver_base_moves+=1
                    if self.receiver_base_moves>40:return self.finish(kwargs,'receiver base motion stalled')
                    speed=.5 if abs(error[axis])>.05 else .35
                    return SimpleNamespace(command={'kind':'base_action','observation_id':kwargs['observation_id'],'axis':['x','y'][axis],'normalized_velocity':float(np.sign(error[axis])*speed),'gripper':'hold','note':'Bring receiving container into view and arm reach while holding source'},evidence={'origin':'mobile_place','qwen_calls_so_far':len(self.calls)})
                self.stage='locate_place';self.pending=None
                return self.destination(kwargs,q,p,state,context)
            if self.stage in {'source_clear','source_base_align'}:
                error,_=base_pose(self.source_navigation_goal,np.eye(3),state)
                axis=int(np.argmax(np.abs(error[:2])))
                if abs(error[axis])>.02:
                    self.source_navigation_moves+=1
                    if self.source_navigation_moves>40:return self.finish(kwargs,'source navigation remained blocked')
                    return SimpleNamespace(command={'kind':'base_action','observation_id':kwargs['observation_id'],'axis':['x','y'][axis],'normalized_velocity':float(np.sign(error[axis])*(.5 if abs(error[axis])>.05 else .35)),'gripper':'hold','note':'Clear blocked lateral path or bring visually grounded source into reach'},evidence={'origin':'source_navigation','qwen_calls_so_far':len(self.calls)})
                self.pending=None;self.previous=None;self.source_base_samples=[]
                self.stage='source_view' if self.stage=='source_clear' else 'ready'
                return self.complete(**kwargs)
            if self.stage=='source_arm_view':
                ready=q.copy();ready[[1,3,5]]=[-.5,-1.5,1.5]
                if self.arm_view_calls<2 or np.max(np.abs(q[[1,3,5]]-ready[[1,3,5]]))>.04:
                    self.arm_view_calls+=1
                    if self.arm_view_calls>5:return self.finish(kwargs,'source observation posture did not track')
                    return self.move(kwargs,ready,'source_arm_view',1.)
                self.stage='ground';self.pending=None;self.previous=None;self.arm_view_done=True
                self.attempt_run=run/'source-arm-view'
                return self.complete(**kwargs)
            if self.stage=='source_view':
                error,_=base_pose(self.source_view_goal,np.eye(3),state)
                if abs(error[1])>.015:
                    self.source_view_moves+=1
                    self.source_base_samples.append(np.asarray(state['state.base_position'])[:2]);self.source_base_samples=self.source_base_samples[-5:]
                    if len(self.source_base_samples)==5 and np.linalg.norm(self.source_base_samples[-1]-self.source_base_samples[0])<.005 and not self.source_clear_used:
                        self.source_clear_used=True;self.source_navigation_moves=0
                        self.source_navigation_goal,_=world_pose([-.15,0,0],np.eye(3),state)
                        self.stage='source_clear';self.source_view_moves=0
                        self.event('source_navigation_recovery',reason='lateral motion made less than 5mm progress across five observations; back away before resuming')
                        return self.complete(**kwargs)
                    if self.source_view_moves>30:return self.finish(kwargs,'source-view base did not reach lateral goal')
                    self.event('source_base_view',goal_world_m=self.source_view_goal.tolist(),remaining_lateral_m=float(error[1]))
                    return SimpleNamespace(command={'kind':'base_action','observation_id':kwargs['observation_id'],'axis':'y','normalized_velocity':float(np.sign(error[1])*(.5 if abs(error[1])>.05 else .35)),'gripper':'hold','note':'Move laterally toward the source-bearing ray to obtain a second source view'},evidence={'origin':'source_active_view','qwen_calls_so_far':len(self.calls)})
                self.stage='ground';self.pending=None;self.previous=None;self.ready_calls=0
                self.attempt_run=run/('source-view-'+str(self.source_view_attempts))
                return self.complete(**kwargs)
            if self.stage=='ground' and not self.base_positioned:
                if self.base_world_goal is None:
                    self.base_world_goal,_=world_pose([-base_backoff,0,0],np.eye(3),state)
                error,_=base_pose(self.base_world_goal,np.eye(3),state)
                if abs(error[0])>.01:
                    self.base_align_calls+=1
                    if self.base_align_calls>40:return self.finish(kwargs,'initial backoff blocked before source search')
                    return self.base_move(kwargs,np.sign(error[0])*(.5 if abs(error[0])>.05 else .35))
                self.base_positioned=True
            if self.stage=='ground' and not self.navigation_stowed:
                stow=q.copy();stow[[1,3,5]]=[-.5,-2.3,1.5]
                if self.stow_calls<2 or np.max(np.abs(q[[1,3,5]]-stow[[1,3,5]]))>.04:
                    self.stow_calls+=1
                    if self.stow_calls>5:return self.finish(kwargs,'navigation stow posture did not track')
                    return self.move(kwargs,stow,'navigation_stow',1.)
                self.navigation_stowed=True;self.pending=None;self.previous=None
                self.event('navigation_stowed',position_base_m=p.tolist(),reason='lower and fold arm before lateral source-view motion near fixtures')
            if self.stage=='ground':
                task=json.loads(kwargs['instruction'])['task']
                self.task_instruction=task
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
                    coarse=self.query(kwargs,(' Garlic may be a pink or purple bulb or a single clove, with an irregular organic outline and pointed stalk. Gray circular metal discs with radial openings are sink drains, not garlic. Use the actual food morphology rather than assuming a white circular bulb. ' if 'garlic' in task.lower() else '')+f'Task: {task}\nIdentify the movable source object to pick up and return its name in target. Images are ordered left camera, right camera, wrist camera. For each view, describe its appearance and give its visible center. All points must refer to that same source object. If it is not visible, set visible false and point_2d null. Use relative coordinates 0 through 1000 on each image; top-left [0,0], bottom-right [1000,1000].',object_schema())
                if operation in {'faucet_view','faucet_turn'}:pass
                elif perception=='sam2':
                    if 'ladle' in coarse['target'].lower():
                        from .sam_rgb_grounding import ground_sam_ladle
                        ground=ground_sam_ladle(kwargs['images'],coarse,context['camera_calibration'],state,self.attempt_run/'sam2-source')
                    else:
                        from .sam_source_grounding import ground_sam_source
                        def select_source(instruction,images,response_schema):
                            roles={'left':'single_camera_original','right':'single_camera_original_color_source_crops','wrist':'single_camera_binary_source_masks'}
                            return self.query({**kwargs,'images':images},instruction,response_schema)
                        ground=ground_sam_source(kwargs['images'],coarse,context['camera_calibration'],state,self.attempt_run/'sam2-source',select_source,task=task,source_reference=self.source_reference)
                        self.source_reference=ground.get('evidence',{}).get('source_reference',self.source_reference)
                else:ground=ground_dark_object(kwargs['images'],coarse,context['camera_calibration'],state)
                if ground['status']!='grounded' and 'garlic' in task.lower() and 'from the sink' in task.lower() and self.source_support_queries<2:
                    from .source_support import ground_on_sink
                    def support_query(instruction,images,schema,roles):return self.query({**kwargs,'images':images},instruction,schema,image_roles=roles)
                    self.source_support_queries+=1
                    ground,self.source_support_anchor=ground_on_sink(ground,coarse,kwargs['images'],context['camera_calibration'],state,self.attempt_run/'source-support',support_query,self.source_support_anchor)
                    self.event('source_support_grounding',result=ground,anchor=self.source_support_anchor)
                from .temporal_source import update_source_history
                ground=update_source_history(self.source_history,ground,coarse,context['camera_calibration'],state)
                self.event('grounding',result=ground)
                if ground['status']!='grounded':
                    locations=ground.get('evidence',{}).get('localization_candidates',[])
                    if locations and not self.targeted_view_done:
                        candidate=min(locations,key=lambda x:x['one_pixel_sensitivity_m'])
                        point=np.asarray(candidate['target_base_m'])
                        if 'from the microwave' in task.lower() or 'from the cabinet' in task.lower():
                            inspect=point+[-.18,0,.06];axis=np.array([math.cos(math.radians(20)),0.,-math.sin(math.radians(20))]);jaw=np.array([0.,1.,0.]);look=np.column_stack([jaw,np.cross(axis,jaw),axis])
                        else:
                            inspect=point+[0,0,.18];_,look=pregrasp_targets(point,yaw=0.)
                        self.targeted_view_done=True;self.pending=None;self.previous=None;self.baseline=force.copy();self.executed=[q.tolist()]
                        if self.segment(q,inspect,look,'source_inspect'):
                            self.event('targeted_source_view',localization=candidate,reason='location supports a closer observation although grasp width is unresolved')
                            return self.queue_move(kwargs)
                        self.abort_reason=None
                    if self.source_view_attempts<3:
                        from .image_servo import _camera_geometry,_pixel_ray_world
                        visible=ground.get('evidence',{}).get('per_view',{})
                        if visible:
                            view=next(v for v in ('left','right','wrist') if v in visible);row=visible[view]
                            cal=context['camera_calibration'][view]
                            origin,direction=_pixel_ray_world(row['pixel'],_camera_geometry(cal))
                            base_y_world,_=world_pose([0,.01,0],np.eye(3),state)
                            delta=base_y_world-np.asarray(state['state.base_position'])
                            local=np.asarray(cal['camera_xmat_world']).T@(np.asarray(direction)-delta)
                            shifted_u=cal['cx_px']+cal['fx_px']*local[0]/-local[2]
                            derivative=shifted_u-row['pixel'][0]
                            side=float(np.sign((cal['cx_px']-row['pixel'][0])*derivative) or 1.)
                            self.event('pixel_directed_source_view',camera=view,pixel=row['pixel'],predicted_u_shift_per_cm=shifted_u-row['pixel'][0],direction=side)
                        else:
                            side=1. if self.source_view_attempts==0 else -1.
                        distance=.15
                        self.source_view_goal,_=world_pose([0,side*distance,0],np.eye(3),state)
                        self.source_view_attempts+=1;self.source_view_moves=0;self.stage='source_view'
                        self.event('source_active_view',reason='lateral parallax toward observed source bearing',side=side,distance_m=distance)
                        return self.complete(**kwargs)
                    if not self.arm_view_done:
                        self.stage='source_arm_view';self.pending=None
                        self.event('source_arm_view',reason='wrist viewpoint after lateral camera views remain unresolved')
                        return self.complete(**kwargs)
                    return self.finish(kwargs,'RGB grounding unresolved after base and wrist views')
                if self.selected_memory_id is None:self.select_memory(kwargs,task,coarse)
                if place and operation=='pick_place' and 'place it in the sink' in task.lower() and self.destination_anchor is None:
                    from .destination_anchor import ground_sink_destination
                    def select_destination(instruction,images,schema,roles):
                        return self.query({**kwargs,'images':images},instruction,schema,image_roles=roles)
                    anchor=ground_sink_destination(kwargs['images'],context['camera_calibration'],state,run/'initial-destination',select_destination,task)
                    self.event('initial_destination',result=anchor)
                    if anchor['status']!='grounded':return self.finish(kwargs,'destination unresolved before pickup: '+anchor['reason'])
                    self.destination_anchor=anchor
                feature='lever_view' if operation in {'faucet_view','faucet_turn'} else ('bowl_center' if any(c['feature']=='bowl_center' for c in ground['candidates']) else 'object_center') if grasp_candidate=='auto' else {'handle':'handle_center','bowl':'bowl_center','object':'object_center'}[grasp_candidate]
                selected=next((c for c in ground['candidates'] if c['feature']==feature),None)
                if selected is None:return self.finish(kwargs,'requested RGB feature unresolved: '+feature)
                if feature in {'handle_center','object_center'} and not selected.get('grasp_candidate'):return self.finish(kwargs,'handle has no supported grasp axis')
                self.target=(direct_grasp_target(selected).tolist() if approach=='direct' else list(selected['target_base_m']))
                if approach=='direct':
                    height_adjustment=self.target[2]-selected['target_base_m'][2]
                    self.pregrasp_distance=max(.04,self.nominal_pregrasp_distance-height_adjustment)
                    self.lift_distance=min(self.lift_distance,self.pregrasp_distance)
                self.event('grasp_height',effective_lift_distance_m=self.lift_distance,nominal_pregrasp_distance_m=self.nominal_pregrasp_distance,effective_pregrasp_distance_m=self.pregrasp_distance,approach=approach,stereo_target_base_m=selected['target_base_m'],adjusted_target_base_m=self.target.copy(),hull_height_bounds_base_m=selected.get('hull_height_bounds_base_m'))
                self.target[2]+=self.current_depth_offset_m;self.target_name=coarse['target']
                if self.is_mixer:self.side_offset=outboard_side_offset(self.target,self.side_offset)
                self.object_center_offset=np.array(ground.get('object_center_base_m') or self.target)-self.target
                self.object_center_offset[2]=0.
                axis=selected.get('jaw_closing_axis_base',[1.,0.,0.]);self.yaw=math.atan2(axis[1],axis[0]) if yaw_override is None else yaw_override
                _,initial_down=pregrasp_targets(self.target,yaw=self.yaw)
                self.source_world_position,self.source_world_rotation=world_pose(self.target,initial_down,state)
                self.source_world_center,_=world_pose(ground.get('object_center_base_m') or self.target,initial_down,state)
                self.event('grasp_hypothesis',feature=feature,target_base_m=self.target,yaw_rad=self.yaw,grasp_axis_supported=selected.get('grasp_candidate',False),depth_offset_m=self.current_depth_offset_m,scope='bowl-center grasp is an unvalidated development hypothesis' if feature=='bowl_center' else 'RGB mask-supported grasp')
                if self.target[0]>.65 or abs(self.target[1])>.30:
                    offset=[float(np.clip(self.target[0]-.50,-.20,.20)),float(np.clip(self.target[1],-.30,.30)),0]
                    self.source_navigation_goal,_=world_pose(offset,np.eye(3),state);self.source_navigation_moves=0
                    self.stage='source_base_align';self.pending=None
                    self.event('source_reach_alignment',offset_base_m=offset,source_base_m=self.target)
                    return self.complete(**kwargs)
                if self.grasp_retry_count:
                    return self.start_recovered_pick(kwargs,q,p,R,force)
                if approach=='direct':
                    self.refresh_source(state);self.baseline=force.copy();self.executed=[q.tolist()]
                    if self.segment(q,self.pre,self.down,'pregrasp'):
                        self.event('direct_entry',reason='reachable source pregrasp from measured current configuration')
                        return self.queue_move(kwargs)
                    self.abort_reason=None
                self.stage='ready'
            if self.stage=='ready':
                if not self.base_positioned:
                    if self.base_world_goal is None:
                        self.base_world_goal,_=world_pose([-base_backoff,0,0],np.eye(3),state)
                    error,_=base_pose(self.base_world_goal,np.eye(3),state)
                    if abs(error[0])>.01:
                        self.base_align_calls+=1
                        if self.base_align_calls>80:return self.finish(kwargs,'base backoff did not reach its measured target')
                        return self.base_move(kwargs,np.sign(error[0])*(.5 if abs(error[0])>.05 else .35))
                    self.base_positioned=True
                ready=q.copy();ready[1]=-.5;ready[3]=-1.5;ready[5]=1.5
                if self.ready_calls<2 or np.max(np.abs(q[[1,3,5]]-ready[[1,3,5]]))>.04:
                    self.ready_calls+=1
                    if self.ready_calls>5:return self.finish(kwargs,'ready posture did not track')
                    return self.move(kwargs,ready,'ready',1.)
                self.baseline=force.copy();self.executed=[q.tolist()]
                self.refresh_source(state)
                # Begin in a bent, unloaded pose before imposing the grasp orientation.
                approach_point=self.pre.copy()
                if self.is_mixer:
                    # Stage the camera relative to the observed object center;
                    # a jaw-centering correction belongs to the later grasp.
                    source_center,_=base_pose(self.source_world_center,np.eye(3),state)
                    approach_point[:2]=source_center[:2]
                if approach=='direct' and 'from the stove' in self.task_instruction.lower() and p[2]>self.pre[2]+.15:
                    low=p.copy();low[2]=self.pre[2]+.10
                    if not self.segment(q,low,R,'ready_lower'):return self.finish(kwargs,self.abort_reason)
                    return self.queue_move(kwargs)
                if approach=='overhead':approach_point[2]=max(p[2],self.pre[2])
                if approach in {'side','side_split','side_horizontal'}:
                    approach_point[1]-=self.side_offset;approach_point[2]=max(p[2],self.pre[2]) if approach_height is None else approach_height
                    if not self.segment(q,approach_point,R,'side_clear'):return self.finish(kwargs,self.abort_reason)
                elif not self.segment(q,approach_point,self.down,'overhead' if approach=='overhead' else 'pregrasp'):return self.finish(kwargs,self.abort_reason)
            if self.stage=='descend':
                goal,rotation=base_pose(*self.world_goal,state)
                distance=float(np.linalg.norm(goal-p))
                angle=float(np.linalg.norm(Rotation.from_matrix(rotation@R.T).as_rotvec()))
                if distance<.02 and angle<math.radians(5) and np.linalg.norm(force-self.baseline)>3:
                    self.event('grasp_contact_arrival',position_error_m=distance,orientation_error_rad=angle,force_delta_n=float(np.linalg.norm(force-self.baseline)))
                    self.pending_path_start=None;self.pending_path_length=0;self.pending=None
                    self.grasp_rotation=R.copy();self.lift_start=p.tolist();self.stage='close'
                    if place:self.grasp_views=[{'state':dict(state),'images':kwargs['images']}]
                    return self.move(kwargs,q,'close',0.)
            if self.pending is not None and self.stage not in {'ready','retreat','verify'}:
                tracking=tracking_step(actual_q=q,target_q=self.pending,previous_q=self.previous,force=force,unloaded_force=self.baseline,executed_q=self.executed[self.last_unloaded:])
                self.event('tracking',result=tracking)
                slow=tracking['progress_fraction']<.03 and (tracking['position_error_m']>.005 or tracking['orientation_error_rad']>math.radians(4))
                self.no_progress_count=self.no_progress_count+1 if slow else 0
                if self.no_progress_count>=3:
                    tracking['status']='blocked';tracking['retreat_waypoints']=list(reversed(self.executed[self.stage_path_start:]));tracking['reason']='no measured pose progress for three execution chunks'
                if tracking['status']=='blocked':
                    self.event('blocked',**tracking)
                    if self.stage=='descend' and not self.contact_repair_used and remaining>=250:
                        self.contact_repair_used=True;self.pending=None;self.pending_path_start=None
                        if not self.segment(q,p-.08*R[:,2],R,'contact_clear'):return self.finish(kwargs,self.abort_reason)
                        return self.queue_move(kwargs)
                    self.queue=[np.array(x) for x in tracking['retreat_waypoints']];self.index=0;self.stage='retreat';self.stage_attempts=0;self.abort_reason='contact blocked; reversing executed waypoints';self.pending=None
                else:
                    self.executed.append(q.tolist())
                    if tracking['force_increase_n']<15:self.last_unloaded=len(self.executed)-1
            if self.stage in {'source_inspect','contact_clear','visual_repair_move','ready_lower','mixer_view_turn','mixer_view_restore','orient','overhead','side_clear','side_lower','side_drop','side_rotate','posture_clear','side_enter','pregrasp','descend','lift','retreat','grasp_retry_retreat','grasp_retry_view','mixer_outward','mixer_orient','mixer_raise','mixer_side','mixer_front','transfer','lower_place','leave','inspect_place','faucet_shift','faucet_pregrasp','faucet_descend','faucet_arc','faucet_leave'}:
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
                if old=='source_inspect':
                    self.stage='ground';self.pending=None;self.previous=None;self.ready_calls=0
                    self.attempt_run=run/'targeted-source-view'
                    return self.complete(**kwargs)
                if old in {'contact_clear','visual_repair_move'}:
                    self.baseline=force.copy()
                    return self.visual_repair(kwargs,q,p,R)
                if old=='ready_lower':
                    self.refresh_source(state)
                    if not self.segment(q,p,self.down,'orient'):return self.finish(kwargs,self.abort_reason)
                elif old=='faucet_pregrasp':
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
                    pre=np.asarray(self.target)-self.pregrasp_distance*self.down[:,2]
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
                    if place:
                        self.grasp_views=[{'state':dict(state),'images':kwargs['images']}]
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
                    if self.destination_anchor is not None:
                        self.event('intentional_release',position_base_m=p.tolist(),gripper_qpos=state['state.gripper_qpos'],reason='arrived over robot-facing basin region; release before inward descent')
                        self.stage='release';return self.move(kwargs,q,'release',1.)
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
                if place:
                    self.grasp_views.append({'state':dict(state),'images':kwargs['images']})
                self.event('closed',gripper_qpos=state['state.gripper_qpos'])
                if not self.segment(q,p+np.array([0,0,self.lift_distance]),self.down,'lift'):return self.finish(kwargs,self.abort_reason)
                return self.queue_move(kwargs)
            if self.stage=='verify':
                schema={'type':'object','properties':{'held':{'type':'boolean'},'description':{'type':'string'}},'required':['held','description'],'additionalProperties':False}
                if place:
                    from .grasp_visual import temporal_grasp_images,grasp_check_instruction
                    self.grasp_views.append({'state':dict(state),'images':kwargs['images']})
                    images,visual_context=temporal_grasp_images(self.grasp_views,run/'grasp-temporal')
                    instruction=grasp_check_instruction(self.task_instruction,self.target_name,visual_context['camera'],self.lift_distance,abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1]))
                    check=self.query({**kwargs,'images':images},instruction,schema,image_roles={'left':'before_close','right':'after_close','wrist':'after_lift'})
                    self.event('grasp_temporal_context',**visual_context)
                else:
                    check=self.query(kwargs,'Task: '+self.task_instruction+' Inspect the current three RGB views. Verify the identity of the TASK SOURCE, not merely any held object. Is the target '+str(self.target_name)+' held between the gripper fingers and lifted clear of the surface? Measured finger separation is '+str(abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1]))+' metres; the close action was executed, followed by a commanded '+str(self.lift_distance)+'m lift. Describe object evidence in the images.',schema)
                self.event('qwen_grasp_visual_check',result=check,end_effector_lift_m=float(p[2]-self.lift_start[2]),gripper_qpos=state['state.gripper_qpos'])
                if abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1])<.003:return self.finish(kwargs,'measured empty gripper after lift')
                if not check['held']:return self.finish(kwargs,'task source not held after lift')
                if check['held']:
                    self.held_gap_m=abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1])
                    self.grasp_cost_actions=consumed-self.attempt_start_actions
                    self.carry_start_index=len(self.executed)-1
                if not place:return self.finish(kwargs,'grasp/lift development pilot complete; full task judged by environment')
                return self.destination(kwargs,q,p,state,context)
            if self.stage=='locate_place':return self.destination(kwargs,q,p,state,context)
            if self.stage=='release':
                leave=p+np.array([0,0,self.withdraw_distance])
                if self.destination_transit_world is not None:
                    leave,_=base_pose(*self.destination_transit_world,state)
                if not self.segment(q,leave,self.down,'leave'):return self.finish(kwargs,self.abort_reason)
                return self.queue_move(kwargs)
            return self.finish(kwargs,'unexpected skill state '+self.stage)
    return SkillClient


def main():
    from .joint_runner import run_episode
    parser=argparse.ArgumentParser();parser.add_argument('--task',default='PickPlaceCounterToDrawer');parser.add_argument('--run',type=Path,required=True);parser.add_argument('--actions',type=int,default=900);parser.add_argument('--candidate',choices=['bowl','handle','object','auto'],default='bowl');parser.add_argument('--yaw',type=float);parser.add_argument('--depth-offset',type=float,default=0.);parser.add_argument('--place',action='store_true');parser.add_argument('--perception',choices=['classical','sam2'],default='classical')
    parser.add_argument('--transfer-clearance',type=float);parser.add_argument('--grasp-down-angle',type=float,default=0.);parser.add_argument('--approach-height',type=float);parser.add_argument('--joint3-posture',type=float);parser.add_argument('--base-backoff',type=float,default=0.);parser.add_argument('--side-offset',type=float,default=.12);parser.add_argument('--operation',choices=['pick_place','faucet_view','faucet_turn'],default='pick_place');parser.add_argument('--side-clearance',type=float,default=.245);parser.add_argument('--release-clearance',type=float,default=.10);parser.add_argument('--seed',type=int,default=7);parser.add_argument('--approach',choices=['direct','overhead','side','side_split','side_horizontal'],default='direct')
    parser.add_argument('--memory',type=Path,required=True)
    args=parser.parse_args();result=run_episode(task=args.task,seed=args.seed,run=args.run,max_decisions=180,client_class=make_client(args.run,args.candidate,args.yaw,args.depth_offset,args.place,args.perception,args.approach,args.release_clearance,args.side_clearance,args.side_offset,args.base_backoff,args.operation,args.joint3_posture,args.approach_height,args.grasp_down_angle,args.transfer_clearance,skill_memory=json.loads(args.memory.read_text())),protocol='skills',action_budget=args.actions,prompt_variant='rig')
    evidence=json.loads((args.run/'skill-evidence.json').read_text());result['policy_class']=evidence['policy_class'];result['executor_decisions']=result.pop('model_decisions');result['qwen_model_calls']=evidence['qwen_calls'];result['pilot_scope']=evidence['pilot_scope'];result['action_budget']=args.actions;(args.run/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'requests','receipts'}}),flush=True)
if __name__=='__main__':main()
