# Local-progress milestones (evaluator-side, EE-short)

| method | n | approach<=0.10m | contact | hold | lift>=3cm | displaced>=10cm | success | median closest dist m |
|---|---|---|---|---|---|---|---|---|
| sem | 9 | 4 | 2 | 1 | 0 | 1 | 0 | 0.116 |
| sem+plan | 9 | 4 | 3 | 1 | 0 | 2 | 0 | 0.183 |
| sem+plan+rec | 9 | 4 | 3 | 2 | 2 | 2 | 0 | 0.173 |
| sem-full | 9 | 4 | 4 | 0 | 0 | 0 | 0 | 0.198 |

## Per run

| method | task | seed | closest m | contact | hold | max lift m | displaced | success |
|---|---|---|---|---|---|---|---|---|
| sem | PickPlaceCounterToDrawer | 0 | 0.466 | False | False | 0.000 | False | False |
| sem | PickPlaceCounterToDrawer | 1 | 0.076 | False | False | 0.000 | False | False |
| sem | PickPlaceCounterToDrawer | 2 | 0.116 | False | False | 0.000 | False | False |
| sem | PickPlaceCounterToSink | 0 | 0.078 | False | False | 0.000 | False | False |
| sem | PickPlaceCounterToSink | 1 | 0.698 | False | False | 0.000 | False | False |
| sem | PickPlaceCounterToSink | 2 | 0.553 | False | False | 0.000 | False | False |
| sem | PickPlaceStoveToCounter | 0 | 0.187 | False | False | 0.000 | False | False |
| sem | PickPlaceStoveToCounter | 1 | 0.048 | True | False | 0.000 | True | False |
| sem | PickPlaceStoveToCounter | 2 | 0.052 | True | True | 0.000 | False | False |
| sem+plan | PickPlaceCounterToDrawer | 0 | 0.351 | False | False | 0.000 | True | False |
| sem+plan | PickPlaceCounterToDrawer | 1 | 0.017 | True | False | 0.000 | True | False |
| sem+plan | PickPlaceCounterToDrawer | 2 | 0.084 | False | False | 0.000 | False | False |
| sem+plan | PickPlaceCounterToSink | 0 | 0.292 | False | False | 0.000 | False | False |
| sem+plan | PickPlaceCounterToSink | 1 | 0.698 | False | False | 0.000 | False | False |
| sem+plan | PickPlaceCounterToSink | 2 | 0.553 | False | False | 0.000 | False | False |
| sem+plan | PickPlaceStoveToCounter | 0 | 0.183 | False | False | 0.000 | False | False |
| sem+plan | PickPlaceStoveToCounter | 1 | 0.045 | True | True | 0.004 | False | False |
| sem+plan | PickPlaceStoveToCounter | 2 | 0.087 | True | False | 0.000 | False | False |
| sem+plan+rec | PickPlaceCounterToDrawer | 0 | 0.471 | False | False | 0.000 | False | False |
| sem+plan+rec | PickPlaceCounterToDrawer | 1 | 0.024 | False | False | 0.000 | False | False |
| sem+plan+rec | PickPlaceCounterToDrawer | 2 | 0.091 | True | True | 0.033 | True | False |
| sem+plan+rec | PickPlaceCounterToSink | 0 | 0.292 | False | False | 0.000 | False | False |
| sem+plan+rec | PickPlaceCounterToSink | 1 | 0.698 | False | False | 0.000 | False | False |
| sem+plan+rec | PickPlaceCounterToSink | 2 | 0.553 | False | False | 0.000 | False | False |
| sem+plan+rec | PickPlaceStoveToCounter | 0 | 0.173 | False | False | 0.000 | False | False |
| sem+plan+rec | PickPlaceStoveToCounter | 1 | 0.027 | True | True | 0.070 | True | False |
| sem+plan+rec | PickPlaceStoveToCounter | 2 | 0.088 | True | False | 0.000 | False | False |
| sem-full | PickPlaceCounterToDrawer | 0 | 0.471 | False | False | 0.000 | False | False |
| sem-full | PickPlaceCounterToDrawer | 1 | 0.037 | True | False | 0.000 | False | False |
| sem-full | PickPlaceCounterToDrawer | 2 | 0.073 | True | False | 0.000 | False | False |
| sem-full | PickPlaceCounterToSink | 0 | 0.292 | False | False | 0.000 | False | False |
| sem-full | PickPlaceCounterToSink | 1 | 0.698 | False | False | 0.000 | False | False |
| sem-full | PickPlaceCounterToSink | 2 | 0.553 | False | False | 0.000 | False | False |
| sem-full | PickPlaceStoveToCounter | 0 | 0.198 | False | False | 0.000 | False | False |
| sem-full | PickPlaceStoveToCounter | 1 | 0.049 | True | False | 0.000 | False | False |
| sem-full | PickPlaceStoveToCounter | 2 | 0.080 | True | False | 0.000 | False | False |
