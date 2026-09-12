"""Count the invalid actions and failed request observed during the reruns."""
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import httpx
from direct.episode import Episode
from direct.policy import InfrastructureError,QwenDirectClient


class AccountingRepairs(unittest.TestCase):
    def test_terminal_counts_include_invalid_actions_but_not_full_plan_header(self):
        cases={
            'short':([{'decision':1,'status':'invalid_action'},
                      {'decision':2,'status':'unreachable','steps':0},
                      {'decision':3,'status':'completed','steps':20},
                      {'decision':4,'status':'stop'}],2),
            'full':([{'decision':1,'sequence_length':2},
                     {'decision':1,'slot':1,'status':'completed','steps':20},
                     {'decision':1,'slot':2,'status':'invalid_action'}],1),
        }
        for mode,(events,expected) in cases.items():
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                args=argparse.Namespace(task='task',seed=0,interface='joint',mode=mode,method='clean',
                    out=str(Path(tmp)/'run'),scenes=str(Path(tmp)/'scenes'),steps_budget=900,wall_budget_s=1200,
                    max_decisions=180,method_config='{}',restore_from=None,ready_pose=False,
                    history_from=None,history_decision=None)
                with patch('direct.episode.git_revision',return_value='test'):
                    ep=Episode(args)
                sim=Mock();sim.observation={'gripper_command':1}
                sim.finish.return_value={'official_success':False,'simulator_steps':20}
                client=Mock();client.totals.return_value={}
                def policy():
                    ep.decisions=events
                    ep.termination='stop' if mode=='short' else 'invalid_action_at_slot_2'
                with patch('direct.episode.QwenDirectClient',return_value=client), \
                     patch('direct.episode.SamClient',return_value=Mock(counter=0,total_elapsed_s=0)), \
                     patch('direct.episode.Simulator',return_value=sim), \
                     patch('direct.episode.render_video',return_value=0), \
                     patch.object(ep,'run_'+mode,side_effect=policy):
                    result=ep.run_episode()
                self.assertEqual(result['rejected_actions'],expected)

    def make_client(self):
        client=object.__new__(QwenDirectClient)
        client.http=Mock()
        client.attestation={};client.model='test';client._token='test-secret';client.log=io.StringIO()
        client.calls=0;client.latency_s=0.;client.prompt_tokens=0;client.completion_tokens=0;client.by_category={}
        return client

    def test_connection_reset_is_a_logged_counted_call_with_unknown_usage(self):
        client=self.make_client()
        client.http.post.side_effect=httpx.ReadError('connection reset by peer')
        with patch('direct.policy.verify_process'),patch('direct.policy.time.monotonic',side_effect=[10.,12.]), \
             self.assertRaises(InfrastructureError):
            client.complete(system_prompt='test',user_text='{}',images=[],response_schema={},
                            max_tokens=20,category='control',decision=6)
        self.assertEqual(client.calls,1)
        self.assertEqual(client.latency_s,2.)
        self.assertEqual(client.by_category['control']['calls'],1)
        self.assertEqual(client.by_category['control']['latency_s'],2.)
        record=json.loads(client.log.getvalue())
        self.assertEqual(record['decision'],6)
        self.assertEqual(record['transport_error'],'ReadError')
        self.assertIsNone(record['http_status'])
        self.assertIsNone(record['usage'])

    def test_successful_call_and_usage_are_counted_once(self):
        client=self.make_client()
        client.http.post.return_value=httpx.Response(200,json={'choices':[{'message':{'content':'{"action":{}}'},
            'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':3}})
        with patch('direct.policy.verify_process'),patch('direct.policy.time.monotonic',side_effect=[10.,12.]):
            record=client.complete(system_prompt='test',user_text='{}',images=[],response_schema={},
                                   max_tokens=20,category='control',decision=1)
        self.assertEqual(record['parsed'],{'action':{}})
        self.assertEqual(client.calls,1)
        self.assertEqual(client.latency_s,2.)
        self.assertEqual(client.by_category['control'],{'calls':1,'latency_s':2.,'prompt_tokens':10,'completion_tokens':3})
        self.assertEqual((client.prompt_tokens,client.completion_tokens),(10,3))


if __name__=='__main__':unittest.main()
