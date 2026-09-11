# Results

Runs: 9 from /home/jli/state/qwen-direct/matrix/phaseC-h6

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| h6 | ee | short | PickPlaceCounterToDrawer | 0/3 | {'stop': 2, 'step_budget': 1} |
| h6 | ee | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| h6 | ee | short | PickPlaceStoveToCounter | 0/3 | {'stop': 3} |
| h6 | ee | short | **all** | 0/9 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h6 | ee | short | 9 | 747 | 49.1 | 11.0 | 49.1 | 204967 | 7600 | 176 | 145 | 373 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| h6 | ee | short | PickPlaceCounterToDrawer | 0 | False | stop | 880 | 50 | 523 | PickPlaceCounterToDrawer-s0-ee-short-h6 |
| h6 | ee | short | PickPlaceCounterToDrawer | 1 | False | stop | 880 | 45 | 453 | PickPlaceCounterToDrawer-s1-ee-short-h6 |
| h6 | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 56 | 532 | PickPlaceCounterToDrawer-s2-ee-short-h6 |
| h6 | ee | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 57 | 502 | PickPlaceCounterToSink-s0-ee-short-h6 |
| h6 | ee | short | PickPlaceCounterToSink | 1 | False | stop | 0 | 1 | 10 | PickPlaceCounterToSink-s1-ee-short-h6 |
| h6 | ee | short | PickPlaceCounterToSink | 2 | False | stop | 580 | 35 | 359 | PickPlaceCounterToSink-s2-ee-short-h6 |
| h6 | ee | short | PickPlaceStoveToCounter | 0 | False | stop | 880 | 67 | 422 | PickPlaceStoveToCounter-s0-ee-short-h6 |
| h6 | ee | short | PickPlaceStoveToCounter | 1 | False | stop | 880 | 50 | 229 | PickPlaceStoveToCounter-s1-ee-short-h6 |
| h6 | ee | short | PickPlaceStoveToCounter | 2 | False | stop | 820 | 81 | 330 | PickPlaceStoveToCounter-s2-ee-short-h6 |
