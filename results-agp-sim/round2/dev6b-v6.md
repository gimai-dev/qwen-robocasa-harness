| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | SUCCESS | success (agent) | 0.032 | 28 | 12 | 8 | 50 | 10.3 | loop nudges x1 |
| PickPlaceCounterToDrawer-seed1 | SUCCESS | no RESULT.md | 0.043 | 86 | 43 | 20 | 138 | 45.7 | loop nudges x6; wall_clock |
| PickPlaceCounterToDrawer-seed2 | never-approached | no RESULT.md | 0.257 | 56 | 35 | 8 | 114 | 45.5 | SETTLE_MISS x37; never closed gripper; loop nudges x21; wall_clock; move_base x10 |
| PickPlaceCounterToSink-seed0 | lifted-not-placed | no RESULT.md | 0.018 | 119 | 67 | 37 | 138 | 45.7 | SETTLE_MISS x12; CLAMP x3; loop nudges x2; wall_clock |
| PickPlaceCounterToSink-seed1 | lifted-not-placed | no RESULT.md | 0.013 | 94 | 64 | 16 | 154 | 45.5 | SETTLE_MISS x16; loop nudges x7; wall_clock; move_base x5 |
| PickPlaceCounterToSink-seed2 | never-approached | no RESULT.md | 0.222 | 85 | 32 | 18 | 145 | 46.0 | CLAMP x4; loop nudges x9; wall_clock |
| PickPlaceStoveToCounter-seed0 | touched-no-lift | no RESULT.md | 0.026 | 94 | 41 | 20 | 160 | 41.6 | CLAMP x3; loop nudges x9; max_turns; move_base x3 |
| PickPlaceStoveToCounter-seed1 | touched-no-lift | no RESULT.md | 0.036 | 60 | 41 | 6 | 160 | 46.0 | SETTLE_MISS x11; loop nudges x35; max_turns |
| PickPlaceStoveToCounter-seed2 | SUCCESS | success (agent) | 0.015 | 32 | 18 | 8 | 49 | 13.5 | SETTLE_MISS x4; loop nudges x2 |

stages: {"SUCCESS": 3, "never-approached": 2, "lifted-not-placed": 2, "touched-no-lift": 2}
