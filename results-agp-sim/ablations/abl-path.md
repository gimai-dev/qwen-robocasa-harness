| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | SUCCESS | success (agent) | 0.118 | 9 | 5 | 4 | 44 | 9.8 | loop nudges x2 |
| PickPlaceCounterToDrawer-seed1 | never-approached | no RESULT.md | 0.287 | 1 | 0 | 1 | 60 | 46.5 | never closed gripper; loop nudges x1; wall_clock |
| PickPlaceCounterToDrawer-seed2 | lifted-not-placed | no RESULT.md | 0.02 | 46 | 27 | 19 | 149 | 45.6 | loop nudges x6; wall_clock |
| PickPlaceCounterToSink-seed0 | lifted-not-placed | success (agent) | 0.021 | 11 | 6 | 5 | 44 | 10.4 | loop nudges x2 |
| PickPlaceCounterToSink-seed1 | SUCCESS | success (agent) | 0.008 | 11 | 4 | 4 | 50 | 9.5 | loop nudges x3 |
| PickPlaceCounterToSink-seed2 | never-approached | no RESULT.md | 0.438 | 116 | 68 | 8 | 160 | 33.6 | loop nudges x15; max_turns; move_base x40 |
| PickPlaceStoveToCounter-seed0 | never-approached | success (agent) | 0.196 | 35 | 26 | 9 | 63 | 19.6 | loop nudges x1 |
| PickPlaceStoveToCounter-seed1 | lifted-not-placed | no RESULT.md | 0.021 | 99 | 80 | 10 | 160 | 45.9 | SETTLE_MISS x51; loop nudges x22; max_turns |
| PickPlaceStoveToCounter-seed2 | never-approached | no RESULT.md | 0.186 | 96 | 94 | 2 | 110 | 46.1 | loop nudges x30; wall_clock |

stages: {"SUCCESS": 2, "never-approached": 4, "lifted-not-placed": 3}
