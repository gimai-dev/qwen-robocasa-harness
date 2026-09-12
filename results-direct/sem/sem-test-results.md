# Results

Runs: 120 from /home/jli/state/qwen-direct/matrix/sem-test

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| sem | semantic | short | PickPlaceCounterToDrawer | 0/20 | {'step_budget': 20} |
| sem | semantic | short | PickPlaceCounterToSink | 0/20 | {'step_budget': 15, 'stop': 3, 'decision_budget': 2} |
| sem | semantic | short | PickPlaceStoveToCounter | 0/20 | {'step_budget': 13, 'decision_budget': 6, 'no_progress': 1} |
| sem | semantic | short | **all** | 0/60 | |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 0/20 | {'step_budget': 13, 'no_progress': 4, 'decision_budget': 3} |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 0/20 | {'step_budget': 15, 'decision_budget': 2, 'no_progress': 3} |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 0/20 | {'step_budget': 18, 'decision_budget': 1, 'no_progress': 1} |
| sem+plan+rec | semantic | short | **all** | 0/60 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sem | semantic | short | 60 | 852 | 126.0 | 11.1 | 126.0 | 87406 | 377 | 24 | 0 | 78 |
| sem+plan+rec | semantic | short | 60 | 838 | 136.3 | 13.3 | 136.6 | 99869 | 3398 | 79 | 0 | 135 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| sem | semantic | short | PickPlaceCounterToDrawer | 100 | False | step_budget | 900 | 140 | 68 | PickPlaceCounterToDrawer-s100-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 101 | False | step_budget | 900 | 133 | 81 | PickPlaceCounterToDrawer-s101-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 102 | False | step_budget | 900 | 105 | 92 | PickPlaceCounterToDrawer-s102-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 103 | False | step_budget | 900 | 126 | 88 | PickPlaceCounterToDrawer-s103-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 104 | False | step_budget | 900 | 128 | 83 | PickPlaceCounterToDrawer-s104-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 105 | False | step_budget | 900 | 152 | 96 | PickPlaceCounterToDrawer-s105-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 106 | False | step_budget | 900 | 122 | 72 | PickPlaceCounterToDrawer-s106-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 107 | False | step_budget | 900 | 141 | 81 | PickPlaceCounterToDrawer-s107-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 108 | False | step_budget | 900 | 115 | 75 | PickPlaceCounterToDrawer-s108-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 109 | False | step_budget | 900 | 151 | 90 | PickPlaceCounterToDrawer-s109-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 110 | False | step_budget | 900 | 162 | 95 | PickPlaceCounterToDrawer-s110-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 111 | False | step_budget | 900 | 140 | 87 | PickPlaceCounterToDrawer-s111-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 112 | False | step_budget | 900 | 66 | 62 | PickPlaceCounterToDrawer-s112-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 113 | False | step_budget | 900 | 150 | 80 | PickPlaceCounterToDrawer-s113-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 114 | False | step_budget | 900 | 144 | 93 | PickPlaceCounterToDrawer-s114-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 115 | False | step_budget | 900 | 172 | 99 | PickPlaceCounterToDrawer-s115-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 116 | False | step_budget | 900 | 116 | 76 | PickPlaceCounterToDrawer-s116-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 117 | False | step_budget | 900 | 96 | 74 | PickPlaceCounterToDrawer-s117-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 118 | False | step_budget | 900 | 140 | 89 | PickPlaceCounterToDrawer-s118-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToDrawer | 119 | False | step_budget | 900 | 108 | 80 | PickPlaceCounterToDrawer-s119-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 100 | False | step_budget | 900 | 112 | 72 | PickPlaceCounterToSink-s100-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 101 | False | stop | 130 | 8 | 9 | PickPlaceCounterToSink-s101-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 102 | False | decision_budget | 792 | 180 | 87 | PickPlaceCounterToSink-s102-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 103 | False | stop | 590 | 66 | 53 | PickPlaceCounterToSink-s103-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 104 | False | step_budget | 900 | 91 | 64 | PickPlaceCounterToSink-s104-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 105 | False | step_budget | 900 | 105 | 68 | PickPlaceCounterToSink-s105-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 106 | False | step_budget | 900 | 126 | 72 | PickPlaceCounterToSink-s106-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 107 | False | stop | 148 | 11 | 12 | PickPlaceCounterToSink-s107-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 108 | False | step_budget | 900 | 82 | 63 | PickPlaceCounterToSink-s108-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 109 | False | step_budget | 900 | 142 | 82 | PickPlaceCounterToSink-s109-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 110 | False | step_budget | 900 | 129 | 78 | PickPlaceCounterToSink-s110-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 111 | False | step_budget | 900 | 101 | 69 | PickPlaceCounterToSink-s111-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 112 | False | step_budget | 900 | 108 | 73 | PickPlaceCounterToSink-s112-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 113 | False | step_budget | 900 | 115 | 70 | PickPlaceCounterToSink-s113-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 114 | False | step_budget | 900 | 134 | 77 | PickPlaceCounterToSink-s114-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 115 | False | step_budget | 900 | 105 | 72 | PickPlaceCounterToSink-s115-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 116 | False | step_budget | 900 | 148 | 82 | PickPlaceCounterToSink-s116-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 117 | False | decision_budget | 822 | 180 | 112 | PickPlaceCounterToSink-s117-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 118 | False | step_budget | 900 | 115 | 78 | PickPlaceCounterToSink-s118-ee-short-sem |
| sem | semantic | short | PickPlaceCounterToSink | 119 | False | step_budget | 900 | 123 | 74 | PickPlaceCounterToSink-s119-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 100 | False | step_budget | 900 | 133 | 87 | PickPlaceStoveToCounter-s100-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 101 | False | decision_budget | 788 | 180 | 85 | PickPlaceStoveToCounter-s101-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 102 | False | step_budget | 900 | 129 | 78 | PickPlaceStoveToCounter-s102-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 103 | False | step_budget | 900 | 137 | 114 | PickPlaceStoveToCounter-s103-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 104 | False | step_budget | 900 | 178 | 94 | PickPlaceStoveToCounter-s104-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 105 | False | step_budget | 900 | 132 | 84 | PickPlaceStoveToCounter-s105-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 106 | False | step_budget | 900 | 115 | 90 | PickPlaceStoveToCounter-s106-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 107 | False | decision_budget | 804 | 180 | 80 | PickPlaceStoveToCounter-s107-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 108 | False | step_budget | 900 | 136 | 84 | PickPlaceStoveToCounter-s108-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 109 | False | decision_budget | 900 | 180 | 93 | PickPlaceStoveToCounter-s109-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 110 | False | no_progress | 420 | 54 | 36 | PickPlaceStoveToCounter-s110-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 111 | False | step_budget | 900 | 157 | 81 | PickPlaceStoveToCounter-s111-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 112 | False | step_budget | 900 | 126 | 87 | PickPlaceStoveToCounter-s112-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 113 | False | step_budget | 900 | 89 | 77 | PickPlaceStoveToCounter-s113-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 114 | False | decision_budget | 748 | 180 | 82 | PickPlaceStoveToCounter-s114-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 115 | False | decision_budget | 888 | 180 | 92 | PickPlaceStoveToCounter-s115-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 116 | False | step_budget | 900 | 138 | 82 | PickPlaceStoveToCounter-s116-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 117 | False | step_budget | 900 | 82 | 70 | PickPlaceStoveToCounter-s117-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 118 | False | step_budget | 900 | 87 | 65 | PickPlaceStoveToCounter-s118-ee-short-sem |
| sem | semantic | short | PickPlaceStoveToCounter | 119 | False | decision_budget | 882 | 180 | 87 | PickPlaceStoveToCounter-s119-ee-short-sem |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 100 | False | step_budget | 900 | 140 | 130 | PickPlaceCounterToDrawer-s100-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 101 | False | step_budget | 900 | 140 | 140 | PickPlaceCounterToDrawer-s101-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 102 | False | decision_budget | 874 | 180 | 183 | PickPlaceCounterToDrawer-s102-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 103 | False | no_progress | 670 | 114 | 114 | PickPlaceCounterToDrawer-s103-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 104 | False | step_budget | 900 | 153 | 150 | PickPlaceCounterToDrawer-s104-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 105 | False | step_budget | 900 | 149 | 151 | PickPlaceCounterToDrawer-s105-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 106 | False | step_budget | 900 | 117 | 118 | PickPlaceCounterToDrawer-s106-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 107 | False | step_budget | 900 | 135 | 135 | PickPlaceCounterToDrawer-s107-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 108 | False | decision_budget | 900 | 180 | 166 | PickPlaceCounterToDrawer-s108-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 109 | False | no_progress | 530 | 97 | 97 | PickPlaceCounterToDrawer-s109-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 110 | False | step_budget | 900 | 147 | 144 | PickPlaceCounterToDrawer-s110-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 111 | False | step_budget | 900 | 148 | 139 | PickPlaceCounterToDrawer-s111-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 112 | False | step_budget | 900 | 46 | 68 | PickPlaceCounterToDrawer-s112-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 113 | False | no_progress | 780 | 134 | 128 | PickPlaceCounterToDrawer-s113-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 114 | False | step_budget | 900 | 142 | 149 | PickPlaceCounterToDrawer-s114-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 115 | False | step_budget | 900 | 136 | 144 | PickPlaceCounterToDrawer-s115-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 116 | False | no_progress | 750 | 162 | 147 | PickPlaceCounterToDrawer-s116-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 117 | False | step_budget | 900 | 152 | 152 | PickPlaceCounterToDrawer-s117-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 118 | False | step_budget | 900 | 175 | 168 | PickPlaceCounterToDrawer-s118-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToDrawer | 119 | False | decision_budget | 882 | 180 | 172 | PickPlaceCounterToDrawer-s119-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 100 | False | no_progress | 174 | 28 | 33 | PickPlaceCounterToSink-s100-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 101 | False | decision_budget | 792 | 180 | 158 | PickPlaceCounterToSink-s101-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 102 | False | step_budget | 900 | 99 | 113 | PickPlaceCounterToSink-s102-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 103 | False | step_budget | 900 | 134 | 146 | PickPlaceCounterToSink-s103-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 104 | False | step_budget | 900 | 140 | 132 | PickPlaceCounterToSink-s104-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 105 | False | step_budget | 900 | 112 | 113 | PickPlaceCounterToSink-s105-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 106 | False | step_budget | 900 | 150 | 143 | PickPlaceCounterToSink-s106-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 107 | False | step_budget | 900 | 125 | 137 | PickPlaceCounterToSink-s107-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 108 | False | step_budget | 900 | 136 | 135 | PickPlaceCounterToSink-s108-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 109 | False | decision_budget | 748 | 180 | 158 | PickPlaceCounterToSink-s109-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 110 | False | step_budget | 900 | 145 | 142 | PickPlaceCounterToSink-s110-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 111 | False | step_budget | 900 | 141 | 139 | PickPlaceCounterToSink-s111-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 112 | False | step_budget | 900 | 168 | 160 | PickPlaceCounterToSink-s112-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 113 | False | step_budget | 900 | 140 | 132 | PickPlaceCounterToSink-s113-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 114 | False | step_budget | 900 | 168 | 156 | PickPlaceCounterToSink-s114-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 115 | False | no_progress | 174 | 31 | 34 | PickPlaceCounterToSink-s115-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 116 | False | step_budget | 900 | 144 | 138 | PickPlaceCounterToSink-s116-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 117 | False | no_progress | 174 | 31 | 39 | PickPlaceCounterToSink-s117-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 118 | False | step_budget | 900 | 140 | 139 | PickPlaceCounterToSink-s118-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceCounterToSink | 119 | False | step_budget | 900 | 139 | 132 | PickPlaceCounterToSink-s119-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 100 | False | step_budget | 900 | 140 | 142 | PickPlaceStoveToCounter-s100-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 101 | False | step_budget | 900 | 142 | 135 | PickPlaceStoveToCounter-s101-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 102 | False | step_budget | 900 | 140 | 138 | PickPlaceStoveToCounter-s102-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 103 | False | step_budget | 900 | 140 | 172 | PickPlaceStoveToCounter-s103-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 104 | False | step_budget | 900 | 147 | 146 | PickPlaceStoveToCounter-s104-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 105 | False | step_budget | 900 | 140 | 137 | PickPlaceStoveToCounter-s105-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 106 | False | step_budget | 900 | 140 | 153 | PickPlaceStoveToCounter-s106-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 107 | False | step_budget | 900 | 145 | 137 | PickPlaceStoveToCounter-s107-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 108 | False | decision_budget | 892 | 180 | 169 | PickPlaceStoveToCounter-s108-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 109 | False | step_budget | 900 | 147 | 143 | PickPlaceStoveToCounter-s109-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 110 | False | no_progress | 534 | 88 | 90 | PickPlaceStoveToCounter-s110-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 111 | False | step_budget | 900 | 136 | 136 | PickPlaceStoveToCounter-s111-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 112 | False | step_budget | 900 | 140 | 145 | PickPlaceStoveToCounter-s112-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 113 | False | step_budget | 900 | 138 | 143 | PickPlaceStoveToCounter-s113-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 114 | False | step_budget | 900 | 141 | 139 | PickPlaceStoveToCounter-s114-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 115 | False | step_budget | 900 | 157 | 149 | PickPlaceStoveToCounter-s115-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 116 | False | step_budget | 900 | 140 | 134 | PickPlaceStoveToCounter-s116-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 117 | False | step_budget | 900 | 136 | 133 | PickPlaceStoveToCounter-s117-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 118 | False | step_budget | 900 | 145 | 138 | PickPlaceStoveToCounter-s118-ee-short-sem+plan+rec |
| sem+plan+rec | semantic | short | PickPlaceStoveToCounter | 119 | False | step_budget | 900 | 140 | 134 | PickPlaceStoveToCounter-s119-ee-short-sem+plan+rec |
