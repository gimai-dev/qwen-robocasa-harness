# Results

Runs: 18 from /home/jli/state/qwen-direct/matrix/phaseC-joint-short

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| clean | joint | short | PickPlaceCounterToDrawer | 0/3 | {'stop': 3} |
| clean | joint | short | PickPlaceCounterToSink | 0/3 | {'step_budget': 1, 'stop': 2} |
| clean | joint | short | PickPlaceStoveToCounter | 0/3 | {'stop': 3} |
| clean | joint | short | **all** | 0/9 | |
| h2 | joint | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 1, 'stop': 2} |
| h2 | joint | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| h2 | joint | short | PickPlaceStoveToCounter | 0/3 | {'stop': 3} |
| h2 | joint | short | **all** | 0/9 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|
| h2 | joint | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | joint | short | 9 | 829 | 43.3 | 0.0 | 43.3 | 179742 | 6727 | 186 | 186 | 427 |
| h2 | joint | short | 9 | 771 | 43.6 | 0.0 | 43.6 | 185575 | 6445 | 176 | 183 | 410 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | joint | short | PickPlaceCounterToDrawer | 0 | False | stop | 880 | 45 | 490 | PickPlaceCounterToDrawer-s0-joint-short-clean |
| clean | joint | short | PickPlaceCounterToDrawer | 1 | False | stop | 860 | 47 | 456 | PickPlaceCounterToDrawer-s1-joint-short-clean |
| clean | joint | short | PickPlaceCounterToDrawer | 2 | False | stop | 860 | 44 | 461 | PickPlaceCounterToDrawer-s2-joint-short-clean |
| clean | joint | short | PickPlaceCounterToSink | 0 | False | stop | 880 | 45 | 375 | PickPlaceCounterToSink-s0-joint-short-clean |
| clean | joint | short | PickPlaceCounterToSink | 1 | False | stop | 440 | 23 | 197 | PickPlaceCounterToSink-s1-joint-short-clean |
| clean | joint | short | PickPlaceCounterToSink | 2 | False | step_budget | 900 | 47 | 405 | PickPlaceCounterToSink-s2-joint-short-clean |
| clean | joint | short | PickPlaceStoveToCounter | 0 | False | stop | 880 | 46 | 473 | PickPlaceStoveToCounter-s0-joint-short-clean |
| clean | joint | short | PickPlaceStoveToCounter | 1 | False | stop | 880 | 48 | 514 | PickPlaceStoveToCounter-s1-joint-short-clean |
| clean | joint | short | PickPlaceStoveToCounter | 2 | False | stop | 880 | 45 | 474 | PickPlaceStoveToCounter-s2-joint-short-clean |
| h2 | joint | short | PickPlaceCounterToDrawer | 0 | False | stop | 880 | 47 | 445 | PickPlaceCounterToDrawer-s0-joint-short-h2 |
| h2 | joint | short | PickPlaceCounterToDrawer | 1 | False | stop | 860 | 45 | 450 | PickPlaceCounterToDrawer-s1-joint-short-h2 |
| h2 | joint | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 51 | 496 | PickPlaceCounterToDrawer-s2-joint-short-h2 |
| h2 | joint | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 48 | 401 | PickPlaceCounterToSink-s0-joint-short-h2 |
| h2 | joint | short | PickPlaceCounterToSink | 1 | False | stop | 620 | 32 | 319 | PickPlaceCounterToSink-s1-joint-short-h2 |
| h2 | joint | short | PickPlaceCounterToSink | 2 | False | stop | 180 | 10 | 101 | PickPlaceCounterToSink-s2-joint-short-h2 |
| h2 | joint | short | PickPlaceStoveToCounter | 0 | False | stop | 860 | 48 | 483 | PickPlaceStoveToCounter-s0-joint-short-h2 |
| h2 | joint | short | PickPlaceStoveToCounter | 1 | False | stop | 880 | 64 | 569 | PickPlaceStoveToCounter-s1-joint-short-h2 |
| h2 | joint | short | PickPlaceStoveToCounter | 2 | False | stop | 860 | 47 | 430 | PickPlaceStoveToCounter-s2-joint-short-h2 |
