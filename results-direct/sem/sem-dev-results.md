# Results

Runs: 36 from /home/jli/state/qwen-direct/matrix/sem-dev

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| sem | semantic | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 3} |
| sem | semantic | short | PickPlaceCounterToSink | 0/3 | {'decision_budget': 1, 'step_budget': 1, 'stop': 1} |
| sem | semantic | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| sem | semantic | short | **all** | 0/9 | |
| sem+plan | semantic | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 3} |
| sem+plan | semantic | short | PickPlaceCounterToSink | 0/3 | {'step_budget': 2, 'no_progress': 1} |
| sem+plan | semantic | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| sem+plan | semantic | short | **all** | 0/9 | |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 2, 'decision_budget': 1} |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 0/3 | {'no_progress': 1, 'step_budget': 2} |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| sem+plan+rec | semantic | short | **all** | 0/9 | |
| sem-full | semantic | short | PickPlaceCounterToDrawer | 0/3 | {'decision_budget': 1, 'step_budget': 2} |
| sem-full | semantic | short | PickPlaceCounterToSink | 0/3 | {'no_progress': 1, 'step_budget': 2} |
| sem-full | semantic | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| sem-full | semantic | short | **all** | 0/9 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sem | semantic | short | 9 | 806 | 116.4 | 7.0 | 116.4 | 80789 | 353 | 26 | 0 | 80 |
| sem+plan | semantic | short | 9 | 819 | 134.7 | 13.8 | 135.7 | 99069 | 3382 | 99 | 0 | 158 |
| sem+plan+rec | semantic | short | 9 | 806 | 132.0 | 13.0 | 132.6 | 96483 | 3302 | 97 | 0 | 155 |
| sem-full | semantic | short | 9 | 809 | 137.2 | 15.1 | 137.9 | 101038 | 3444 | 84 | 0 | 142 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| sem | semantic | short | PickPlaceCounterToDrawer | 0 | False | step_budget | 900 | 149 | 95 | PickPlaceCounterToDrawer-s0-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 900 | 103 | 76 | PickPlaceCounterToDrawer-s1-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 133 | 94 | PickPlaceCounterToDrawer-s2-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 0 | False | decision_budget | 816 | 180 | 98 | PickPlaceCounterToSink-s0-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 1 | False | step_budget | 900 | 84 | 68 | PickPlaceCounterToSink-s1-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 2 | False | stop | 142 | 10 | 12 | PickPlaceCounterToSink-s2-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 129 | 90 | PickPlaceStoveToCounter-s0-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 131 | 89 | PickPlaceStoveToCounter-s1-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 129 | 96 | PickPlaceStoveToCounter-s2-ee-short-sem |
| sem+plan | semantic | short | PickPlaceCounterToDrawer | 0 | False | step_budget | 900 | 174 | 204 | PickPlaceCounterToDrawer-s0-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 900 | 161 | 174 | PickPlaceCounterToDrawer-s1-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 136 | 168 | PickPlaceCounterToDrawer-s2-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceCounterToSink | 0 | False | no_progress | 168 | 26 | 37 | PickPlaceCounterToSink-s0-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceCounterToSink | 1 | False | step_budget | 900 | 133 | 160 | PickPlaceCounterToSink-s1-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceCounterToSink | 2 | False | step_budget | 900 | 159 | 198 | PickPlaceCounterToSink-s2-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 140 | 168 | PickPlaceStoveToCounter-s0-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 136 | 154 | PickPlaceStoveToCounter-s1-ee-short-sem+plan |
| sem+plan | semantic | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 147 | 162 | PickPlaceStoveToCounter-s2-ee-short-sem+plan |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 0 | False | decision_budget | 786 | 180 | 200 | PickPlaceCounterToDrawer-s0-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 900 | 149 | 167 | PickPlaceCounterToDrawer-s1-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 156 | 183 | PickPlaceCounterToDrawer-s2-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 0 | False | no_progress | 168 | 26 | 32 | PickPlaceCounterToSink-s0-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 1 | False | step_budget | 900 | 133 | 158 | PickPlaceCounterToSink-s1-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 2 | False | step_budget | 900 | 140 | 174 | PickPlaceCounterToSink-s2-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 140 | 167 | PickPlaceStoveToCounter-s0-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 119 | 144 | PickPlaceStoveToCounter-s1-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 145 | 171 | PickPlaceStoveToCounter-s2-ee-short-sem+plan+rec |
| sem-full | semantic | short | PickPlaceCounterToDrawer | 0 | False | decision_budget | 816 | 180 | 182 | PickPlaceCounterToDrawer-s0-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 900 | 178 | 160 | PickPlaceCounterToDrawer-s1-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 145 | 148 | PickPlaceCounterToDrawer-s2-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceCounterToSink | 0 | False | no_progress | 168 | 26 | 37 | PickPlaceCounterToSink-s0-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceCounterToSink | 1 | False | step_budget | 900 | 133 | 160 | PickPlaceCounterToSink-s1-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceCounterToSink | 2 | False | step_budget | 900 | 161 | 186 | PickPlaceCounterToSink-s2-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 140 | 143 | PickPlaceStoveToCounter-s0-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 126 | 125 | PickPlaceStoveToCounter-s1-ee-short-sem-full |
| sem-full | semantic | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 146 | 133 | PickPlaceStoveToCounter-s2-ee-short-sem-full |
