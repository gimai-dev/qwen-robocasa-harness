# Results

Runs: 18 from /home/jli/state/qwen-direct/matrix/phaseC-ee-full

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| clean | ee | full | PickPlaceCounterToDrawer | 0/3 | {'unreachable_at_slot_4': 1, 'sequence_complete': 2} |
| clean | ee | full | PickPlaceCounterToSink | 0/3 | {'sequence_complete': 2, 'unreachable_at_slot_3': 1} |
| clean | ee | full | PickPlaceStoveToCounter | 0/3 | {'sequence_complete': 2, 'unreachable_at_slot_3': 1} |
| clean | ee | full | **all** | 0/9 | |
| h3 | ee | full | PickPlaceCounterToDrawer | 0/3 | {'sequence_complete': 2, 'truncated_output': 1} |
| h3 | ee | full | PickPlaceCounterToSink | 0/3 | {'sequence_complete': 3} |
| h3 | ee | full | PickPlaceStoveToCounter | 0/3 | {'sequence_complete': 3} |
| h3 | ee | full | **all** | 0/9 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|
| h3 | ee | full | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | full | 9 | 49 | 1.0 | 0.3 | 1.0 | 3480 | 378 | 8 | 4 | 16 |
| h3 | ee | full | 9 | 18 | 1.0 | 0.0 | 1.9 | 5564 | 958 | 20 | 5 | 28 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | full | PickPlaceCounterToDrawer | 0 | False | unreachable_at_slot_4 | 60 | 1 | 23 | PickPlaceCounterToDrawer-s0-ee-full-clean |
| clean | ee | full | PickPlaceCounterToDrawer | 1 | False | sequence_complete | 20 | 1 | 9 | PickPlaceCounterToDrawer-s1-ee-full-clean |
| clean | ee | full | PickPlaceCounterToDrawer | 2 | False | sequence_complete | 20 | 1 | 13 | PickPlaceCounterToDrawer-s2-ee-full-clean |
| clean | ee | full | PickPlaceCounterToSink | 0 | False | sequence_complete | 20 | 1 | 9 | PickPlaceCounterToSink-s0-ee-full-clean |
| clean | ee | full | PickPlaceCounterToSink | 1 | False | unreachable_at_slot_3 | 40 | 1 | 21 | PickPlaceCounterToSink-s1-ee-full-clean |
| clean | ee | full | PickPlaceCounterToSink | 2 | False | sequence_complete | 20 | 1 | 10 | PickPlaceCounterToSink-s2-ee-full-clean |
| clean | ee | full | PickPlaceStoveToCounter | 0 | False | sequence_complete | 20 | 1 | 9 | PickPlaceStoveToCounter-s0-ee-full-clean |
| clean | ee | full | PickPlaceStoveToCounter | 1 | False | sequence_complete | 200 | 1 | 29 | PickPlaceStoveToCounter-s1-ee-full-clean |
| clean | ee | full | PickPlaceStoveToCounter | 2 | False | unreachable_at_slot_3 | 40 | 1 | 23 | PickPlaceStoveToCounter-s2-ee-full-clean |
| h3 | ee | full | PickPlaceCounterToDrawer | 0 | False | sequence_complete | 20 | 1 | 22 | PickPlaceCounterToDrawer-s0-ee-full-h3 |
| h3 | ee | full | PickPlaceCounterToDrawer | 1 | False | truncated_output | 0 | 1 | 89 | PickPlaceCounterToDrawer-s1-ee-full-h3 |
| h3 | ee | full | PickPlaceCounterToDrawer | 2 | False | sequence_complete | 20 | 1 | 21 | PickPlaceCounterToDrawer-s2-ee-full-h3 |
| h3 | ee | full | PickPlaceCounterToSink | 0 | False | sequence_complete | 20 | 1 | 16 | PickPlaceCounterToSink-s0-ee-full-h3 |
| h3 | ee | full | PickPlaceCounterToSink | 1 | False | sequence_complete | 20 | 1 | 26 | PickPlaceCounterToSink-s1-ee-full-h3 |
| h3 | ee | full | PickPlaceCounterToSink | 2 | False | sequence_complete | 20 | 1 | 19 | PickPlaceCounterToSink-s2-ee-full-h3 |
| h3 | ee | full | PickPlaceStoveToCounter | 0 | False | sequence_complete | 20 | 1 | 20 | PickPlaceStoveToCounter-s0-ee-full-h3 |
| h3 | ee | full | PickPlaceStoveToCounter | 1 | False | sequence_complete | 20 | 1 | 20 | PickPlaceStoveToCounter-s1-ee-full-h3 |
| h3 | ee | full | PickPlaceStoveToCounter | 2 | False | sequence_complete | 20 | 1 | 19 | PickPlaceStoveToCounter-s2-ee-full-h3 |
