# Results

Runs: 36 from /home/jli/state/qwen-direct/matrix/phaseB-clean

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| clean | ee | full | PickPlaceCounterToDrawer | 0/3 | {'sequence_complete': 1, 'unreachable_at_slot_1': 1, 'unreachable_at_slot_4': 1} |
| clean | ee | full | PickPlaceCounterToSink | 0/3 | {'sequence_complete': 1, 'unreachable_at_slot_3': 1, 'unreachable_at_slot_2': 1} |
| clean | ee | full | PickPlaceStoveToCounter | 0/3 | {'unreachable_at_slot_3': 2, 'sequence_complete': 1} |
| clean | ee | full | **all** | 0/9 | |
| clean | ee | short | PickPlaceCounterToDrawer | 0/3 | {'no_progress': 2, 'step_budget': 1} |
| clean | ee | short | PickPlaceCounterToSink | 1/3 | {'stop': 2, 'no_progress': 1} |
| clean | ee | short | PickPlaceStoveToCounter | 0/3 | {'no_progress': 1, 'stop': 2} |
| clean | ee | short | **all** | 1/9 | |
| clean | joint | full | PickPlaceCounterToDrawer | 0/3 | {'sequence_complete': 3} |
| clean | joint | full | PickPlaceCounterToSink | 0/3 | {'sequence_complete': 3} |
| clean | joint | full | PickPlaceStoveToCounter | 0/3 | {'sequence_complete': 3} |
| clean | joint | full | **all** | 0/9 | |
| clean | joint | short | PickPlaceCounterToDrawer | 0/3 | {'stop': 2, 'step_budget': 1} |
| clean | joint | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| clean | joint | short | PickPlaceStoveToCounter | 0/3 | {'stop': 2, 'step_budget': 1} |
| clean | joint | short | **all** | 0/9 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | full | 9 | 31 | 1.0 | 0.7 | 1.0 | 3611 | 467 | 10 | 6 | 20 |
| clean | ee | short | 9 | 576 | 53.7 | 24.4 | 53.7 | 213994 | 7950 | 206 | 372 | 620 |
| clean | joint | full | 9 | 149 | 1.0 | 0.0 | 1.0 | 3739 | 474 | 9 | 3 | 22 |
| clean | joint | short | 9 | 884 | 46.1 | 0.0 | 46.1 | 198019 | 7087 | 169 | 288 | 515 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | full | PickPlaceCounterToDrawer | 0 | False | sequence_complete | 20 | 1 | 16 | PickPlaceCounterToDrawer-s0-ee-full-clean |
| clean | ee | full | PickPlaceCounterToDrawer | 1 | False | unreachable_at_slot_4 | 60 | 1 | 27 | PickPlaceCounterToDrawer-s1-ee-full-clean |
| clean | ee | full | PickPlaceCounterToDrawer | 2 | False | unreachable_at_slot_1 | 0 | 1 | 13 | PickPlaceCounterToDrawer-s2-ee-full-clean |
| clean | ee | full | PickPlaceCounterToSink | 0 | False | sequence_complete | 20 | 1 | 12 | PickPlaceCounterToSink-s0-ee-full-clean |
| clean | ee | full | PickPlaceCounterToSink | 1 | False | unreachable_at_slot_2 | 20 | 1 | 23 | PickPlaceCounterToSink-s1-ee-full-clean |
| clean | ee | full | PickPlaceCounterToSink | 2 | False | unreachable_at_slot_3 | 40 | 1 | 27 | PickPlaceCounterToSink-s2-ee-full-clean |
| clean | ee | full | PickPlaceStoveToCounter | 0 | False | sequence_complete | 40 | 1 | 16 | PickPlaceStoveToCounter-s0-ee-full-clean |
| clean | ee | full | PickPlaceStoveToCounter | 1 | False | unreachable_at_slot_3 | 40 | 1 | 25 | PickPlaceStoveToCounter-s1-ee-full-clean |
| clean | ee | full | PickPlaceStoveToCounter | 2 | False | unreachable_at_slot_3 | 40 | 1 | 20 | PickPlaceStoveToCounter-s2-ee-full-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 0 | False | no_progress | 80 | 20 | 281 | PickPlaceCounterToDrawer-s0-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 1 | False | no_progress | 200 | 43 | 519 | PickPlaceCounterToDrawer-s1-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 64 | 735 | PickPlaceCounterToDrawer-s2-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 0 | True | stop | 760 | 48 | 519 | PickPlaceCounterToSink-s0-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 1 | False | stop | 820 | 80 | 1051 | PickPlaceCounterToSink-s1-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 2 | False | no_progress | 220 | 24 | 330 | PickPlaceCounterToSink-s2-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 0 | False | stop | 880 | 47 | 570 | PickPlaceStoveToCounter-s0-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 1 | False | stop | 880 | 101 | 999 | PickPlaceStoveToCounter-s1-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 2 | False | no_progress | 440 | 56 | 581 | PickPlaceStoveToCounter-s2-ee-short-clean |
| clean | joint | full | PickPlaceCounterToDrawer | 0 | False | sequence_complete | 20 | 1 | 8 | PickPlaceCounterToDrawer-s0-joint-full-clean |
| clean | joint | full | PickPlaceCounterToDrawer | 1 | False | sequence_complete | 20 | 1 | 15 | PickPlaceCounterToDrawer-s1-joint-full-clean |
| clean | joint | full | PickPlaceCounterToDrawer | 2 | False | sequence_complete | 20 | 1 | 7 | PickPlaceCounterToDrawer-s2-joint-full-clean |
| clean | joint | full | PickPlaceCounterToSink | 0 | False | sequence_complete | 60 | 1 | 12 | PickPlaceCounterToSink-s0-joint-full-clean |
| clean | joint | full | PickPlaceCounterToSink | 1 | False | sequence_complete | 20 | 1 | 6 | PickPlaceCounterToSink-s1-joint-full-clean |
| clean | joint | full | PickPlaceCounterToSink | 2 | False | sequence_complete | 400 | 1 | 48 | PickPlaceCounterToSink-s2-joint-full-clean |
| clean | joint | full | PickPlaceStoveToCounter | 0 | False | sequence_complete | 200 | 1 | 30 | PickPlaceStoveToCounter-s0-joint-full-clean |
| clean | joint | full | PickPlaceStoveToCounter | 1 | False | sequence_complete | 200 | 1 | 35 | PickPlaceStoveToCounter-s1-joint-full-clean |
| clean | joint | full | PickPlaceStoveToCounter | 2 | False | sequence_complete | 400 | 1 | 40 | PickPlaceStoveToCounter-s2-joint-full-clean |
| clean | joint | short | PickPlaceCounterToDrawer | 0 | False | stop | 880 | 45 | 538 | PickPlaceCounterToDrawer-s0-joint-short-clean |
| clean | joint | short | PickPlaceCounterToDrawer | 1 | False | stop | 860 | 53 | 610 | PickPlaceCounterToDrawer-s1-joint-short-clean |
| clean | joint | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 45 | 551 | PickPlaceCounterToDrawer-s2-joint-short-clean |
| clean | joint | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 45 | 453 | PickPlaceCounterToSink-s0-joint-short-clean |
| clean | joint | short | PickPlaceCounterToSink | 1 | False | stop | 880 | 45 | 527 | PickPlaceCounterToSink-s1-joint-short-clean |
| clean | joint | short | PickPlaceCounterToSink | 2 | False | stop | 880 | 47 | 536 | PickPlaceCounterToSink-s2-joint-short-clean |
| clean | joint | short | PickPlaceStoveToCounter | 0 | False | stop | 880 | 45 | 539 | PickPlaceStoveToCounter-s0-joint-short-clean |
| clean | joint | short | PickPlaceStoveToCounter | 1 | False | stop | 880 | 45 | 437 | PickPlaceStoveToCounter-s1-joint-short-clean |
| clean | joint | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 45 | 443 | PickPlaceStoveToCounter-s2-joint-short-clean |
