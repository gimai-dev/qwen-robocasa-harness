import io,json
from pathlib import Path
from PIL import Image,ImageChops
from adaptive.grasp_visual import temporal_grasp_images

def test_recorded_grasp_keeps_same_camera_region_and_actual_time_order(tmp_path):
    root=Path('/home/jli/state/destination-anchor-20260908/PickPlaceCounterToSink-anchor')
    samples=[]
    for seq in (9,10,11):
        obs=json.loads((root/f'sim/mailbox/observation-{seq:06d}.json').read_text())
        samples.append({'state':obs['public_state'],'images':{v:(root/f'sim/frames/{seq:06d}/{v}.png').read_bytes() for v in ('left','right','wrist')}})
    images,meta=temporal_grasp_images(samples,tmp_path)
    assert meta['camera']=='left' # The right view clips the source at this real grasp.
    assert len(set(images.values()))==3
    for slot,sample in zip(('left','right','wrist'),samples):
        expected=Image.open(io.BytesIO(sample['images']['left'])).crop(meta['crop_box'])
        actual=Image.open(io.BytesIO(images[slot]))
        expected=expected.resize(actual.size)
        assert ImageChops.difference(expected,actual).getbbox() is None


def test_observed_near_destination_unload_finishes_with_open_and_withdraw(tmp_path,monkeypatch):
    import numpy as np
    import robocasa_inspect.model_client as model
    from adaptive.skill_driver import make_client
    from adaptive.panda_embodiment import panda_fk
    from adaptive.world_anchor import world_pose
    monkeypatch.setattr(model,'QwenClient',lambda **kwargs:object())
    root=Path('/home/jli/state/destination-anchor-20260908/PickPlaceCounterToSink-temporal')
    obs=json.loads((root/'sim/mailbox/observation-000018.json').read_text())
    state=obs['public_state'];pose=panda_fk(state['state.arm_joint_position'])
    p=np.array(pose.position_m);rotation=np.array(pose.rotation_matrix)
    client=make_client(tmp_path,skill_memory={'skills':[],'lessons':[]})()
    client.stage='lower_place';client.destination_anchor={};client.held_gap_m=.0438
    client.release_world_goal=world_pose(p,rotation,state)
    client.is_mixer=False;client.resume_path=lambda execution:None
    client.move=lambda kwargs,q,phase,grip: {'phase':phase,'gripper':grip}
    request={'observation_id':'actual-final-slip','public_state':state,'proposal_audit_context':{'remaining_actions':434}}
    assert client.complete(**request)=={'phase':'release','gripper':1.}
    assert client.stage=='release'
    assert any(e['kind']=='destination_unloaded' for e in client.events)
    # The same observed empty grip away from release must still stop/retry.
    client.stage='lower_place';client.held_gap_m=.0438
    client.release_world_goal=world_pose(p+np.array([.10,0,0]),rotation,state)
    client.retry_grasp=lambda kwargs,q:'retry_or_stop'
    assert client.complete(**request)=='retry_or_stop'
