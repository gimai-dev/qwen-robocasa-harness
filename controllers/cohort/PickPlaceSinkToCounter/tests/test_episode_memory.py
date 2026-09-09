import copy
import json
from pathlib import Path
import pytest
from adaptive import episode_memory as memory

DONORS=Path('/home/jli/state/pickplace-ten-20260908')

def donor(name):
 p=DONORS/name
 return json.loads((p/'result.json').read_text()),json.loads((p/'skill-evidence.json').read_text())

def test_real_donors_have_distinct_executable_approaches():
 cards=[memory.extract_card(*donor(n)) for n in ['PickPlaceCounterToDrawer','PickPlaceCounterToStandMixer']]
 assert cards[0]['parameters']['approach']=='direct'
 assert cards[1]['parameters']['approach']=='side_horizontal'
 assert cards[1]['parameters']['grasp_down_angle']==pytest.approx(30)
 assert cards[0]['parameters']['depth_offset_m']==0
 assert cards[1]['parameters']['depth_offset_m']==pytest.approx(.01)
 for c in cards:
  assert c['parameters']['pregrasp_distance']==pytest.approx(.12,abs=.002)
  assert c['parameters']['lift_distance']==pytest.approx(.15,abs=.002)

def test_failed_episode_cannot_enter_success_memory():
 result,evidence=donor('PickPlaceCounterToDrawer')
 result['terminal_outcome']['success']=False
 with pytest.raises(ValueError,match='official success'):
  memory.extract_card(result,evidence)

def test_translating_donor_scene_does_not_change_relative_skill():
 result,evidence=donor('PickPlaceCounterToStandMixer')
 shifted=copy.deepcopy(evidence)
 vector=[1.,-2.,.4]
 def change(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in {'target_base_m','target_position_m','position','release_point_base_m','floor_center_base_m'} and isinstance(v,list) and len(v)==3:
     x[k]=[a+b for a,b in zip(v,vector)]
    else:change(v)
  elif isinstance(x,list):
   for v in x:change(v)
 change(shifted)
 a=memory.extract_card(result,evidence)['parameters'];b=memory.extract_card(result,shifted)['parameters']
 assert a.keys()==b.keys()
 for k in a:
  if isinstance(a[k],str):assert a[k]==b[k]
  else:assert a[k]==pytest.approx(b[k],abs=1e-8)

def test_failure_lesson_is_consumed_without_promoting_failed_skill():
 result,evidence=donor('PickPlaceCounterToDrawer');result['terminal_outcome']['success']=False
 bank={'skills':[],'lessons':[]}
 memory.record_episode(bank,result,evidence,{'lesson':'Approach stalled; change direction','next_skill_id':'side','applicability':'occluded side approach'},'failed-run')
 assert bank['skills']==[]
 context=memory.retrieval_context(bank)
 assert 'Approach stalled; change direction' in context
 assert 'failed-run' in context

def test_success_writeback_yields_retrievable_executable_card():
 result,evidence=donor('PickPlaceCounterToDrawer')
 bank={'skills':[],'lessons':[]}
 memory.record_episode(bank,result,evidence,{'lesson':'Top grasp lifted the ladle','next_skill_id':None,'applicability':'ladle'},'successful-run')
 assert len(bank['skills'])==1
 assert bank['skills'][0]['parameters']['approach']=='direct'
 assert 'successful-run' in memory.retrieval_context(bank)
