"""Advertised joint endpoints must pass the actual action decoder."""
from pathlib import Path
import re
import unittest

from direct.actions import decode_action


class JointPromptContractTests(unittest.TestCase):
    def test_advertised_joint_endpoints_are_executable(self):
        prompts=Path(__file__).resolve().parents[1]/'prompts'
        for mode in ('short','full'):
            text=(prompts/f'system_joint_{mode}.txt').read_text()
            ranges=re.findall(r'joint([1-7]) \[([^\]]+)\]',text)
            self.assertEqual(len(ranges),7)
            for number,bounds in ranges:
                for endpoint in map(float,bounds.split(',')):
                    with self.subTest(mode=mode,joint=number,endpoint=endpoint):
                        q=[0.,0.,0.,-1.5,0.,1.5,0.]
                        q[int(number)-1]=endpoint
                        decode_action({'k':'joint','q':q,'g':1},interface='joint')


if __name__=='__main__':unittest.main()
