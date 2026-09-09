import copy,json
from pathlib import Path
from adaptive.episode_memory import record_episode

def test_real_official_success_with_slip_is_saved_without_fabricating_release_skill():
    root=Path('/home/jli/state/destination-anchor-20260908/PickPlaceCounterToSink-front')
    read=lambda name:json.loads((root/name).read_text())
    result=read('result.json');evidence=read('skill-evidence.json')
    assert result['terminal_outcome']['success'] is True
    bank=read('memory-before.json');original=copy.deepcopy(bank)
    out=record_episode(bank,result,evidence,read('reflection.json')['reflection'],root.name)
    assert len(out['lessons'])==len(original['lessons'])+1
    assert out['lessons'][-1]['official_success'] is True
    assert out['lessons'][-1]['skill_status'].startswith('not_promoted:')
    assert out['skills']==original['skills']


def test_actual_release_skill_preserves_nominal_grasp_and_horizontal_route(tmp_path,monkeypatch):
    import numpy as np
    import robocasa_inspect.model_client as model
    from adaptive.skill_driver import make_client
    root=Path('/home/jli/state/controlled-release-20260908/PickPlaceCounterToSink-release-seed7-a')
    read=lambda name:json.loads((root/name).read_text())
    bank=record_episode(read('memory-before.json'),read('result.json'),read('skill-evidence.json'),read('reflection.json')['reflection'],root.name)
    card=bank['skills'][-1];params=card['parameters']
    assert card['destination_route']=='sink_front_release'
    assert params['pregrasp_distance']==.12
    assert params['sink_front_offset_m']==.12
    assert params['sink_withdraw_distance_m']==.10
    assert params['release_clearance']==.22
    monkeypatch.setattr(model,'QwenClient',lambda **kwargs:object())
    client=make_client(tmp_path,place=True,skill_memory=bank)()
    choices=[]
    def query(kwargs,instruction,schema):
        ids=schema['properties']['skill_id']['enum'];choices.append(ids)
        return {'skill_id':ids[0],'reason':'matching observed destination route'}
    client.query=query
    client.select_memory({},'Pick the boxed drink from the counter and place it in the sink.',{})
    assert choices[-1]==[card['id']]
    assert client.nominal_pregrasp_distance==.12
    assert np.isclose(client.sink_front_offset,params['sink_front_offset_m'])
    assert client.sink_withdraw_distance==.10
    client.select_memory({},'Pick the object from the sink and place it on the counter.',{})
    assert card['id'] not in choices[-1]


def test_successful_same_scene_reuse_updates_skill_instead_of_duplicate_id():
    root=Path('/home/jli/state/controlled-release-20260908/PickPlaceCounterToSink-memory-seed7')
    read=lambda name:json.loads((root/name).read_text())
    before=read('memory-before.json')
    bank=record_episode(before,read('result.json'),read('skill-evidence.json'),read('reflection.json')['reflection'],root.name)
    cards=[c for c in bank['skills'] if c['id']=='PickPlaceCounterToSink-seed7']
    assert len(cards)==1
    assert cards[0]['source_run']==root.name
    assert cards[0]['parameters']['sink_front_offset_m']==.120172
