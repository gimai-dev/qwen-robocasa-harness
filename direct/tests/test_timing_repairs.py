"""Distinguish when the scored state existed from later model/cleanup cost."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from direct.actions import Action
from direct.episode import Episode
from direct.executor import Receipt
from direct.recovery import qualify_states


class TimingRepairs(unittest.TestCase):
    def test_qualification_uses_final_state_time_and_keeps_strict_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'
            state={'task':'task','seed':0,'snapshot':'/snapshot.json'}
            result={'task':'task','seed':0,'restore_from':'/snapshot.json','official_success':True,
                    'steps_budget':400,'simulator_steps':200,'max_decisions':80,'decisions':30,
                    'wall_budget_s':600,'wall_s':710,'control_wall_s':650,'final_state_wall_s':590}
            path.write_text(json.dumps(result))
            self.assertTrue(qualify_states([state],[path])[0]['qualified_for_rsr'])
            result['final_state_wall_s']=600.01
            path.write_text(json.dumps(result))
            self.assertFalse(qualify_states([state],[path])[0]['qualified_for_rsr'])

    def test_episode_separates_motion_model_wait_and_postprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=argparse.Namespace(task='task',seed=0,interface='ee',mode='short',method='clean',
                out=str(root/'run'),scenes=str(root/'scenes'),steps_budget=400,wall_budget_s=600,
                max_decisions=80,method_config='{}',restore_from=None,ready_pose=False,
                history_from=None,history_decision=None)
            clock=[1000.]
            sim=Mock()
            sim.observation={'gripper_command':1}
            client=Mock();client.totals.return_value={}
            sam=Mock(counter=0,total_elapsed_s=0)
            with patch('direct.episode.git_revision',return_value='test'):
                ep=Episode(args)

            def move(sim, action, slot_steps):
                clock[0]=1500.
                return Receipt('completed',action,20,{})
            def policy():
                ep.act(Action('hold'),1)
                clock[0]=1650.  # The model returns later without another physics step.
                ep.termination='stop'
            def finish():
                clock[0]=1700.
                return {'official_success':True,'simulator_steps':20}
            def render(*args):
                clock[0]=1800.
                return 0
            sim.finish.side_effect=finish
            with patch('direct.episode.time.monotonic',side_effect=lambda:clock[0]), \
                 patch('direct.episode.QwenDirectClient',return_value=client), \
                 patch('direct.episode.SamClient',return_value=sam), \
                 patch('direct.episode.Simulator',return_value=sim), \
                 patch('direct.episode.execute',side_effect=move), \
                 patch('direct.episode.render_video',side_effect=render), \
                 patch.object(ep,'run_short',side_effect=policy):
                result=ep.run_episode()
            self.assertEqual(result['final_state_wall_s'],500)
            self.assertEqual(result['control_wall_s'],650)
            self.assertEqual(result['wall_s'],800)


if __name__=='__main__':
    unittest.main()
