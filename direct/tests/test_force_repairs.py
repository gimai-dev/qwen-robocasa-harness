"""Reset transients must not become the wrist's unloaded force baseline."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from direct.sim_child import Child


class ForceRepairs(unittest.TestCase):
    def test_reset_transient_does_not_hide_later_loads(self):
        model=SimpleNamespace(sensor_name2id=Mock(return_value=0),sensor_objid=[0],site_bodyid=[0],
                              body_subtreemass=[.52],opt=SimpleNamespace(gravity=[0,0,-9.81]))
        robot=SimpleNamespace(gripper={'right':SimpleNamespace(important_sensors={'force_ee':'wrist_force'})})
        env=SimpleNamespace(unwrapped=SimpleNamespace(sim=SimpleNamespace(model=model),robots=[robot]))
        with tempfile.TemporaryDirectory() as tmp:
            child=Child('task',0,Path(tmp),Path(tmp),action_budget=100,wall_budget_s=100)
            state={'wrench':{'force_n':[113,-78,-169]},'camera_calibration':{}}
            def measured():
                return {**state,'camera_calibration':{}}
            with patch('direct.sim_child._public_state',side_effect=lambda *a:measured()), \
                 patch('direct.sim_child._save_images',return_value={}), \
                 patch('direct.sim_child._atomic_json'), patch.object(child,'snapshot'):
                first=child.publish({},env,0,None)['public_state']
                self.assertIsNone(first['contact_force_delta_n'])
                self.assertAlmostEqual(first['unloaded_wrist_weight_n'],5.1012)
                child.total_steps=20
                state['wrench']['force_n']=[0,0,15]
                loaded=child.publish({},env,1,None)['public_state']
                self.assertAlmostEqual(loaded['contact_force_delta_n'],9.90)
                state['wrench']['force_n']=[-15,0,0]
                rotated=child.publish({},env,2,None)['public_state']
                self.assertEqual(rotated['contact_force_delta_n'],loaded['contact_force_delta_n'])
                state['wrench']['force_n']=[0,0,5.1012]
                empty=child.publish({},env,3,None)['public_state']
                self.assertEqual(empty['contact_force_delta_n'],0)


if __name__=='__main__':
    unittest.main()
