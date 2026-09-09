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
from .pose_skill import pregrasp_targets,plan_pose_segment,tracking_step,choose_parallel_jaw_path
from scipy.spatial.transform import Rotation
from .rgb_grounding import ground_dark_object


def object_schema():
    point={'anyOf':[{'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':2,'maxItems':2},{'type':'null'}]}
    view={'type':'object','properties':{'visible':{'type':'boolean'},'description':{'type':'string'},'point_2d':point},'required':['visible','description','point_2d'],'additionalProperties':False}
    return {'type':'object','properties':{'target':{'type':'string'},**{v:view for v in ['left','right','wrist']}},'required':['target','left','right','wrist'],'additionalProperties':False}


def make_client(run,grasp_candidate='bowl',yaw_override=None,depth_offset_m=0.,place=False,perception='classical',approach='direct',release_clearance=.10):
    from robocasa_inspect.model_client import QwenClient
    class SkillClient:
        def __init__(self,**kwargs):
            self.qwen=QwenClient(**kwargs);self.stage='ground';self.calls=[];self.events=[];self.queue=[];self.index=0;self.ready_calls=0;self.previous=None;self.pending=None;self.executed=[];self.gripper=1.;self.stage_attempts=0;self.baseline=None;self.abort_reason=None;self.target=None;self.last_unloaded=0;self.goal_position=None;self.goal_rotation=None;self.no_progress_count=0;self.stage_path_start=0;self.target_name=None;self.view_attempts=0;self.place_views=[]
        def verify(self): self.qwen.verify()
        def close(self):
            self.save();self.qwen.close()
        def save(self):
            (run/'skill-evidence.json').write_text(json.dumps({'policy_class':'Qwen semantic planner + RGB tools + numerical pose skills','qwen_calls':len(self.calls),'model_requests':self.calls,'events':self.events,'stage':self.stage,'abort_reason':self.abort_reason,'target':self.target,'pilot_scope':'full pick and place attempt' if place else 'grasp and lift diagnostic; official full task is separately evaluated'},indent=2)+'\n')
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
        def move(self,kwargs,q,stage,gripper=None):
            context=kwargs['proposal_audit_context']
            if gripper is not None and abs(gripper-context['current_gripper'])>1e-8 and context['remaining_actions']<16:
                return self.finish(kwargs,'insufficient actions for gripper transition')
            if gripper is not None:self.gripper=gripper
            self.previous=np.array(kwargs['public_state']['state.arm_joint_position']);self.pending=np.array(q)
            self.event('command',observation_id=kwargs['observation_id'],stage=stage,q=list(q),gripper=self.gripper)
            return SimpleNamespace(command={'kind':'move_joints','observation_id':kwargs['observation_id'],'targets':{**{f'joint{i+1}':float(x) for i,x in enumerate(q)},'gripper':self.gripper},'note':f'hybrid pose skill stage={stage}; numerical IK target; Qwen authored semantic target'},evidence={'origin':'numerical_pose_skill','stage':stage,'qwen_calls_so_far':len(self.calls)})
        def finish(self,kwargs,reason):
            self.abort_reason=reason;self.event('finish',reason=reason)
            return SimpleNamespace(command={'kind':'finish','observation_id':kwargs['observation_id'],'note':reason[:320]},evidence={'origin':'hybrid_skill_executor','qwen_calls_so_far':len(self.calls)})
        def segment(self,q,pos,rot,stage):
            plan=(choose_parallel_jaw_path(q.tolist(),np.asarray(pos).tolist(),np.asarray(rot).tolist()) if stage in {'pregrasp','overhead'} and yaw_override is None else plan_pose_segment(q.tolist(),np.asarray(pos).tolist(),np.asarray(rot).tolist()))
            if plan['status']=='kinematically_reachable' and 'target_rotation' in plan:
                rot=plan['target_rotation'];self.down=np.asarray(rot)
            self.event('pose_plan',stage=stage,plan=plan)
            if plan['status']!='kinematically_reachable':self.abort_reason='unreachable_'+stage;return False
            self.queue=[np.array(w['q']) for w in plan['waypoints']];self.index=0;self.stage=stage;self.stage_attempts=0;self.no_progress_count=0;self.stage_path_start=max(0,len(self.executed)-1);self.goal_position=np.asarray(pos);self.goal_rotation=np.asarray(rot);return True
        def destination(self,kwargs,q,p,state,context):
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
                        self.event('placement_geometry',floor_center_base_m=dest_point.tolist(),object_center_offset_source_m=self.object_center_offset.tolist(),object_center_offset_release_m=offset.tolist(),release_point_base_m=self.release_point.tolist())
                        transit=self.release_point.copy();transit[2]=max(p[2],self.release_point[2]+.12)
                        if not self.segment(q,transit,self.down,'transfer'):return self.finish(kwargs,self.abort_reason)
                        return self.move(kwargs,self.queue[0],self.stage)
            if self.view_attempts>=3:return self.finish(kwargs,'destination unresolved after changed views and RGB mask matching')
            self.view_attempts+=1
            view=p.copy()
            if self.view_attempts==1:
                view[0]-=.18;view[1]*=.75;view[2]+=.03
            elif self.view_attempts==2:
                view[0]-=.06;view[1]*=.75;view[2]+=.03
            else:view+=np.array([.03,.07,.02])
            self.event('active_view',reason='obtain drawer floor from two known wrist camera poses',viewpoint_base_m=view.tolist())
            if not self.segment(q,view,self.down,'inspect_place'):return self.finish(kwargs,self.abort_reason)
            return self.move(kwargs,self.queue[0],self.stage)
        def complete(self,**kwargs):
            state=kwargs['public_state'];q=np.array(state['state.arm_joint_position']);pose=panda_fk(q.tolist());p=np.array(pose.position_m);R=np.array(pose.rotation_matrix)
            force=np.array(state['state.end_effector_wrench']['force_n']);context=kwargs['proposal_audit_context'];remaining=context['remaining_actions'] if 'remaining_actions' in context else 900-context['consumed_actions']
            self.event('observation',observation_id=kwargs['observation_id'],stage=self.stage,position=p.tolist(),force=force.tolist(),gripper_qpos=state['state.gripper_qpos'],remaining_actions=remaining)
            if remaining<1:return self.finish(kwargs,'episode action budget reached')
            if self.stage=='ground':
                task=json.loads(kwargs['instruction'])['task']
                coarse=self.query(kwargs,f'Task: {task}\nIdentify the movable source object to pick up and return its name in target. Images are ordered left camera, right camera, wrist camera. For each view, describe its appearance and give its visible center. All points must refer to that same source object. If it is not visible, set visible false and point_2d null. Use relative coordinates 0 through 1000 on each image; top-left [0,0], bottom-right [1000,1000].',object_schema())
                if perception=='sam2':
                    if 'ladle' in coarse['target'].lower():
                        from .sam_rgb_grounding import ground_sam_ladle
                        ground=ground_sam_ladle(kwargs['images'],coarse,context['camera_calibration'],state,run/'sam2-source')
                    else:
                        from .sam_source_grounding import ground_sam_source
                        def select_source(instruction,images,response_schema):
                            roles={'left':'single_camera_original','right':'single_camera_original_color_source_crops','wrist':'single_camera_binary_source_masks'}
                            return self.query({**kwargs,'images':images},instruction,response_schema,image_roles=roles)
                        ground=ground_sam_source(kwargs['images'],coarse,context['camera_calibration'],state,run/'sam2-source',select_source)
                else:ground=ground_dark_object(kwargs['images'],coarse,context['camera_calibration'],state)
                self.event('grounding',result=ground)
                if ground['status']!='grounded':return self.finish(kwargs,'RGB grounding unresolved')
                feature={'handle':'handle_center','bowl':'bowl_center','object':'object_center'}[grasp_candidate]
                selected=next((c for c in ground['candidates'] if c['feature']==feature),None)
                if selected is None:return self.finish(kwargs,'requested RGB feature unresolved: '+feature)
                if grasp_candidate in {'handle','object'} and not selected.get('grasp_candidate'):return self.finish(kwargs,'handle has no supported grasp axis')
                self.target=list(selected['target_base_m']);self.target[2]+=depth_offset_m;self.target_name=coarse['target']
                self.object_center_offset=np.array(ground.get('object_center_base_m') or self.target)-self.target
                self.object_center_offset[2]=0.
                axis=selected.get('jaw_closing_axis_base',[1.,0.,0.]);self.yaw=math.atan2(axis[1],axis[0]) if yaw_override is None else yaw_override
                self.event('grasp_hypothesis',feature=feature,target_base_m=self.target,yaw_rad=self.yaw,grasp_axis_supported=selected.get('grasp_candidate',False),depth_offset_m=depth_offset_m,scope='bowl-center grasp is an unvalidated development hypothesis' if grasp_candidate=='bowl' else 'RGB mask-supported grasp')
                self.stage='ready'
            if self.stage=='ready':
                ready=q.copy();ready[1]=-.5;ready[3]=-1.5;ready[5]=1.5
                if self.ready_calls<2 or np.max(np.abs(q[[1,3,5]]-ready[[1,3,5]]))>.04:
                    self.ready_calls+=1
                    if self.ready_calls>5:return self.finish(kwargs,'ready posture did not track')
                    return self.move(kwargs,ready,'ready',1.)
                self.baseline=force.copy();self.executed=[q.tolist()]
                self.pre,self.down=pregrasp_targets(self.target,yaw=self.yaw,clearance=.12)
                # Begin in a bent, unloaded pose before imposing the grasp orientation.
                approach_point=self.pre.copy()
                if approach=='overhead':approach_point[2]=max(p[2],self.pre[2])
                if not self.segment(q,approach_point,self.down,'overhead' if approach=='overhead' else 'pregrasp'):return self.finish(kwargs,self.abort_reason)
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
            if self.stage in {'orient','overhead','pregrasp','descend','lift','retreat','transfer','lower_place','leave','inspect_place'}:
                self.stage_attempts+=1
                if self.stage_attempts>65:return self.finish(kwargs,'pose stage stalled')
                while self.index<len(self.queue):
                    if self.index==len(self.queue)-1 and self.stage!='retreat':
                        reached=np.linalg.norm(p-self.goal_position)<=.003 and np.linalg.norm(Rotation.from_matrix(self.goal_rotation@R.T).as_rotvec())<=math.radians(3)
                    else:reached=np.max(np.abs(q-self.queue[self.index]))<.025
                    if not reached:break
                    self.index+=1
                if self.index<len(self.queue):
                    # Follow local IK waypoints, batching only a short joint span.
                    end=self.index
                    while end+1<len(self.queue) and np.max(np.abs(self.queue[end+1]-q))<.16:end+=1
                    self.index=end
                    return self.move(kwargs,self.queue[end],self.stage)
                old=self.stage;self.pending=None
                if old=='retreat':return self.finish(kwargs,self.abort_reason)
                if old in {'orient','overhead'}:
                    if not self.segment(q,self.pre,self.down,'pregrasp'):return self.finish(kwargs,self.abort_reason)
                elif old=='pregrasp':
                    if not self.segment(q,self.target,self.down,'descend'):return self.finish(kwargs,self.abort_reason)
                elif old=='descend':
                    self.grasp_rotation=R.copy()
                    self.stage='close';self.lift_start=p.tolist();return self.move(kwargs,q,'close',0.)
                elif old=='lift':self.stage='verify'
                elif old=='transfer':
                    if not self.segment(q,self.release_point,self.down,'lower_place'):return self.finish(kwargs,self.abort_reason)
                elif old=='lower_place':
                    self.stage='release';return self.move(kwargs,q,'release',1.)
                elif old=='inspect_place':self.stage='locate_place'
                elif old=='leave':return self.finish(kwargs,'pick and place sequence finished; evaluate official outcome')
                if self.stage not in {'verify','locate_place'}:return self.move(kwargs,self.queue[0],self.stage)
            if self.stage=='close':
                self.event('closed',gripper_qpos=state['state.gripper_qpos'])
                if not self.segment(q,p+np.array([0,0,.15]),self.down,'lift'):return self.finish(kwargs,self.abort_reason)
                return self.move(kwargs,self.queue[0],self.stage)
            if self.stage=='verify':
                schema={'type':'object','properties':{'held':{'type':'boolean'},'description':{'type':'string'}},'required':['held','description'],'additionalProperties':False}
                check=self.query(kwargs,'Inspect the current three RGB views. Is the target '+str(self.target_name)+' held between the gripper fingers and lifted clear of the surface? Measured finger separation is '+str(abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1]))+' metres; the close action was executed, followed by a 15cm lift. Describe object evidence in the images.',schema)
                self.event('qwen_grasp_visual_check',result=check,end_effector_lift_m=float(p[2]-self.lift_start[2]),gripper_qpos=state['state.gripper_qpos'])
                if not place:return self.finish(kwargs,'grasp/lift development pilot complete; full task judged by environment')
                return self.destination(kwargs,q,p,state,context)
            if self.stage=='locate_place':return self.destination(kwargs,q,p,state,context)
            if self.stage=='release':
                if not self.segment(q,p+np.array([0,0,.20]),self.down,'leave'):return self.finish(kwargs,self.abort_reason)
                return self.move(kwargs,self.queue[0],self.stage)
            return self.finish(kwargs,'unexpected skill state '+self.stage)
    return SkillClient


def main():
    from .joint_runner import run_episode
    parser=argparse.ArgumentParser();parser.add_argument('--task',default='PickPlaceCounterToDrawer');parser.add_argument('--run',type=Path,required=True);parser.add_argument('--actions',type=int,default=900);parser.add_argument('--candidate',choices=['bowl','handle','object'],default='bowl');parser.add_argument('--yaw',type=float);parser.add_argument('--depth-offset',type=float,default=0.);parser.add_argument('--place',action='store_true');parser.add_argument('--perception',choices=['classical','sam2'],default='classical')
    parser.add_argument('--release-clearance',type=float,default=.10);parser.add_argument('--seed',type=int,default=7);parser.add_argument('--approach',choices=['direct','overhead'],default='direct')
    args=parser.parse_args();result=run_episode(task=args.task,seed=args.seed,run=args.run,max_decisions=180,client_class=make_client(args.run,args.candidate,args.yaw,args.depth_offset,args.place,args.perception,args.approach,args.release_clearance),protocol='skills',action_budget=args.actions,prompt_variant='rig')
    evidence=json.loads((args.run/'skill-evidence.json').read_text());result['policy_class']=evidence['policy_class'];result['executor_decisions']=result.pop('model_decisions');result['qwen_model_calls']=evidence['qwen_calls'];result['pilot_scope']=evidence['pilot_scope'];result['action_budget']=args.actions;(args.run/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'requests','receipts'}}),flush=True)
if __name__=='__main__':main()
