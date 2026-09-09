import json
import numpy as np
from adaptive.skill_driver import make_client
from adaptive import sam_destination

def test_selected_memory_changes_regrounded_pregrasp_geometry(tmp_path,monkeypatch):
 import robocasa_inspect.model_client as model
 monkeypatch.setattr(model,'QwenClient',lambda **kw:object())
 card={'id':'successful-donor','official_success':True,'parameters':{
  'approach':'direct','depth_offset_m':.01,'pregrasp_distance':.18,'lift_distance':.23,
  'release_clearance':.22,'withdraw_distance':.11,'grasp_down_angle':90.}}
 client=make_client(tmp_path,skill_memory={'skills':[card],'lessons':[]})()
 client.query=lambda *args,**kwargs:{'skill_id':'successful-donor','reason':'clear top access'}
 client.select_memory({},'pick tomato into bowl',{'target':'tomato'})
 client.source_world_position=np.array([.2,.3,.1]);client.source_world_rotation=np.eye(3)
 client.source_world_center=client.source_world_position.copy()
 state={'state.base_position':[0.,0.,0.],'state.base_rotation':[0.,0.,0.,1.]}
 client.refresh_source(state)
 assert np.allclose(client.pre,np.array(client.target)+[0,0,.18])
 assert client.lift_distance==.23 and client.withdraw_distance==.11
 assert client.current_depth_offset_m==.01
 assert client.events[-2]['kind']=='skill_selected'

def test_destination_prompt_uses_actual_task_without_drawer_assumption(tmp_path,monkeypatch):
 def proposal(image,folder,*,task):
  return {'candidates':[],'instruction':sam_destination.destination_instruction(task),
          'images':{},'image_roles':{},'response_schema':{},'evidence':{}}
 monkeypatch.setattr(sam_destination,'segment_floor_candidates',proposal)
 result=sam_destination.ground_destination_mask(b'',tmp_path,lambda *args:None,
     task='Pick the tomato from the pan and place it in the bowl on the counter.')
 instruction=result['evidence']['selection']['instruction']
 assert 'tomato from the pan' in instruction and 'bowl on the counter' in instruction
 assert 'open drawer' not in instruction

def test_unreachable_first_view_tries_next_reachable_view_from_actual_sink_pose(tmp_path,monkeypatch):
 import robocasa_inspect.model_client as model
 from pathlib import Path
 from adaptive.panda_embodiment import panda_fk
 monkeypatch.setattr(model,'QwenClient',lambda **kw:object())
 result=json.loads(Path('/home/jli/state/skill-transfer-20260908/PickPlaceCounterToSink/result.json').read_text())
 q=np.asarray(result['receipts'][-1]['realized_arm_qpos']);pose=panda_fk(q.tolist())
 client=make_client(tmp_path,skill_memory={'skills':[],'lessons':[]})()
 client.is_mixer=False;client.down=np.asarray(pose.rotation_matrix)
 client.current_state={'state.base_position':[0.,0.,0.],'state.base_rotation':[0.,0.,0.,1.]}
 client.queue_move=lambda kwargs:'reachable_view_move'
 answer=client.destination({'observation_id':'replay'},q,np.asarray(pose.position_m),client.current_state,{})
 assert answer=='reachable_view_move'
 plans=[e['plan'] for e in client.events if e['kind']=='pose_plan']
 assert plans[0]['status']=='kinematically_unresolved'
 assert plans[1]['status']=='kinematically_reachable'
 assert client.view_attempts==2

def test_unresolved_source_changes_view_before_retrying_grounding(tmp_path,monkeypatch):
 import robocasa_inspect.model_client as model
 from pathlib import Path
 from adaptive import sam_source_grounding
 monkeypatch.setattr(model,'QwenClient',lambda **kw:object())
 data=json.loads(Path('/home/jli/state/skill-transfer-20260908/PickPlaceCounterToSink/sam2-source/public-input.json').read_text())
 monkeypatch.setattr(sam_source_grounding,'ground_sam_source',lambda *args,**kw:{'status':'unresolved'})
 client=make_client(tmp_path,perception='sam2',skill_memory={'skills':[],'lessons':[]})()
 client.query=lambda *args,**kw:data['coarse']
 answer=client.complete(observation_id='source-replay',public_state=data['public_state'],
   images={v:b'' for v in ['left','right','wrist']},instruction=json.dumps({'task':'Pick the boxed drink from the counter and place it in the sink.'}),
   proposal_audit_context={'remaining_actions':900,'current_gripper':1.,'last_execution':None,'camera_calibration':data['calibration']})
 assert answer.command['kind']=='move_joints'
 assert client.stage=='source_view'
 assert answer.command['targets']['gripper']==1.


def test_tall_carton_grasp_clears_palm_using_recorded_rgb_hull(tmp_path,monkeypatch):
 from pathlib import Path
 import cv2
 from adaptive.sam_source_grounding import ground_source_masks
 from adaptive.skill_driver import direct_grasp_target
 p=Path('/home/jli/state/grasp-retry-20260908/PickPlaceCounterToSink-baseline/sam2-source')
 data=json.loads((p/'public-input.json').read_text());sam=json.loads((p/'sam-result.json').read_text())
 masks={}
 for v,row in sam['views'].items():
  pick=json.loads((p/(v+'-selection.json')).read_text())['response']
  if pick['visible']:
   selected=next(c for c in row['candidates'] if c['id']==pick['mask_id'])
   masks[v]=cv2.imread(selected['mask_path'],cv2.IMREAD_GRAYSCALE)>0
 g=ground_source_masks(masks,data['coarse'],data['calibration'],data['public_state'])
 assert g['status']=='grounded'
 target=direct_grasp_target(g)
 assert target[2] >= g['hull_height_bounds_base_m'][1]-.04-1e-9
 assert target[2] > g['target_base_m'][2]+.01
 assert np.allclose(target[:2],g['target_base_m'][:2])
 assert np.allclose(direct_grasp_target({'target_base_m':[.1,.2,.3]}),[.1,.2,.3])

 import robocasa_inspect.model_client as model
 from adaptive import sam_source_grounding
 monkeypatch.setattr(model,'QwenClient',lambda **kw:object())
 monkeypatch.setattr(sam_source_grounding,'ground_sam_source',lambda *args,**kw:g)
 bank=json.loads((p.parent/'memory-before.json').read_text())
 donor=next(c for c in bank['skills'] if c['parameters']['approach']=='direct')
 client=make_client(tmp_path,grasp_candidate='auto',perception='sam2',skill_memory=bank)()
 client.query=lambda kwargs,instruction,schema,**extra: data['coarse'] if 'target' in schema['properties'] else {'skill_id':donor['id'],'reason':'direct carton grasp'}
 client.complete(observation_id='height-replay',public_state=data['public_state'],images={'wrist':b''},
  instruction=json.dumps({'task':'Pick the boxed drink from the counter and place it in the sink.'}),
  proposal_audit_context={'remaining_actions':900,'current_gripper':1.,'last_execution':None,'camera_calibration':data['calibration']})
 assert np.allclose(client.target,target)
 client.refresh_source(data['public_state'])
 assert np.allclose(client.target,target)

 # Raising the close target must not raise the already-clear transit corridor.
 expected_pre=np.asarray(g['target_base_m'])+[0,0,donor['parameters']['pregrasp_distance']]
 assert np.allclose(client.pre,expected_pre)

 # Under a shelf, initial extraction returns to the reached pregrasp height.
 assert client.lift_distance <= client.pregrasp_distance


def test_destination_observation_detects_recorded_carton_drop():
 from pathlib import Path
 from adaptive.grasp_tracking import lost_during_carry
 e=json.loads(Path('/home/jli/state/grasp-retry-20260908/PickPlaceCounterToSink-lowlift/skill-evidence.json').read_text())
 check=next(x for x in e['events'] if x['kind']=='qwen_grasp_visual_check')
 held_gap=abs(check['gripper_qpos'][0]-check['gripper_qpos'][1])
 last=[x for x in e['events'] if x['kind']=='observation'][-1]
 assert last['stage']=='inspect_place'
 assert lost_during_carry(last['stage'],held_gap,last['gripper_qpos'])
 assert lost_during_carry('locate_place',held_gap,last['gripper_qpos'])
 assert not lost_during_carry('inspect_place',held_gap,check['gripper_qpos'])
