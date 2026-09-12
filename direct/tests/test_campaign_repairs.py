"""Campaign completion and retained-inspection regressions from code review."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from direct.revision_campaign import inspect_spec, summarize


class CampaignRepairs(unittest.TestCase):
    def test_unscored_and_infrastructure_attempts_are_not_completed_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            specs = []
            for index, (success, termination) in enumerate(((True, 'stop'), (False, 'step_budget'),
                                                          (None, 'stop'), (False, 'infrastructure_error'))):
                run = out / str(index)
                run.mkdir()
                (run / 'result.json').write_text(json.dumps({'official_success': success, 'termination': termination,
                    'simulator_steps': 100 if index < 2 else 900, 'qwen_calls': 5, 'wall_s': 20}))
                specs.append({'condition': 'clean', 'run': str(run)})
            summary = summarize(specs, out)
            row = next(line for line in summary.splitlines() if line.startswith('| clean |'))
            fields = [x.strip() for x in row.split('|')[1:-1]]
            self.assertEqual(fields[:4], ['clean', '2', '2', '1'])
            self.assertEqual(fields[-3], '100.0')

    def test_interrupted_inspection_is_retained_and_retried_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = out / 'episodes' / 'scene'
            spec = {'condition': 'clean', 'run': str(run)}
            work = out / 'inspection' / 'clean' / 'inspect' / run.name
            work.mkdir(parents=True)
            (work / 'stale-observation.json').write_text('old')

            def fresh_inspection(source, directory):
                target = directory / 'inspect' / source.name
                self.assertFalse(target.exists(), 'stale mailbox must not be reused')
                target.mkdir(parents=True)
                return [{'sequence': 0}]

            with patch('direct.recovery.inspect_run', side_effect=fresh_inspection) as inspect:
                inspect_spec(spec, out)
                inspect_spec(spec, out)
            self.assertEqual(inspect.call_count, 1, 'completed inspection should be cached')
            retained = list((out / 'inspection' / 'clean' / 'interrupted').rglob('stale-observation.json'))
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0].read_text(), 'old')


if __name__ == '__main__':
    unittest.main()
