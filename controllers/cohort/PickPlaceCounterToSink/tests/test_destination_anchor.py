import json
from pathlib import Path
import numpy as np
from adaptive.skill_driver import make_client
from adaptive.panda_embodiment import panda_fk
from adaptive.world_anchor import base_pose,world_pose

def test_cached_destination_uses_held_rotation_and_no_extra_rise(tmp_path,monkeypatch):
 import robocasa_inspect.model_client as model
 monkeypatch.setattr(model,'QwenClient',lambda **kw:object())
 source=Path('/home/jli/state/grasp-retry-20260908/PickPlaceCounterToSink-lowlift')
 r=json.loads((source/'result.json').read_text())
 state=json.loads((source/'sam2-source/public-input.json').read_text())['public_state']
 q=np.asarray(r['receipts'][9]['realized_arm_qpos']);pose=panda_fk(q.tolist());p=np.asarray(pose.position_m);rotation=np.asarray(pose.rotation_matrix)
 anchor=json.loads(Path('/home/jli/state/destination-anchor-20260908/drain-roi-probe/geometry.json').read_text())
 client=make_client(tmp_path,release_clearance=.22,skill_memory={'skills':[],'lessons':[]})()
 client.is_mixer=False;client.destination_anchor=anchor;client.down=rotation;client.object_center_offset=np.zeros(3);client.current_state=state
 client.queue_move=lambda kwargs:'move'
 out=client.destination({'observation_id':'anchor-test'},q,p,state,{})
 assert out=='move'
 assert client.stage=='transfer'
 assert client.view_attempts==0
 assert np.allclose(client.goal_rotation,rotation)
 floor,_=base_pose(anchor['target_world_m'],np.eye(3),state)
 assert np.isclose(client.goal_position[2],max(p[2],floor[2]+.22))
 assert np.allclose(client.goal_position[:2],floor[:2]-.12*floor[:2]/np.linalg.norm(floor[:2]))
 assert np.allclose(client.release_point,client.goal_position)
 # Withdrawal stays at the reached height and travels toward the robot.
 withdraw,_=base_pose(*client.destination_transit_world,state)
 assert np.isclose(withdraw[2],client.release_point[2])
 assert np.isclose(np.linalg.norm(withdraw[:2]-client.release_point[:2]),.10)
 next_q=client.queue[-1].copy()
 assert client.segment(next_q,withdraw,client.down,'leave')
 # World anchor is re-expressed from the current public base pose.
 shifted=dict(state);shifted['state.base_position']=(np.asarray(state['state.base_position'])+[.08,-.04,0]).tolist()
 client.current_state=shifted
 client.destination({'observation_id':'anchor-shift'},q,p,shifted,{})
 world,_=world_pose(client.release_point,client.down,shifted)
 assert np.isclose(world[2],p[2]+state['state.base_position'][2])
 assert np.isclose(np.linalg.norm(world[:2]-np.asarray(anchor['target_world_m'])[:2]),.12)

def test_destination_gallery_excludes_background_using_recorded_basin(tmp_path):
 from adaptive.destination_anchor import focused_landmark_gallery
 root=Path('/home/jli/state/destination-anchor-20260908/initial-probe')
 floors=json.loads((root/'selected.json').read_text())
 for view,expected in [('left',17),('right',6)]:
  p=root/view;proposals=json.loads((p/'sam-result.json').read_text())['candidates']
  candidates,_=focused_landmark_gallery(proposals,floors[view],(p/'original.png').read_bytes(),tmp_path)
  assert expected in [c['id'] for c in candidates]
  assert len(candidates)<len(proposals)
