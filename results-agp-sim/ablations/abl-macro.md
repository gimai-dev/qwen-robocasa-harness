| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | never-approached | no RESULT.md | 0.117 | 75 | 60 | 10 | 160 | 26.6 | IK_FAILED x11; CLAMP x12; loop nudges x17; max_turns; move_base x4 |
| PickPlaceCounterToDrawer-seed1 | never-approached | no RESULT.md | 0.116 | 39 | 26 | 10 | 160 | 40.3 | SETTLE_MISS x21; loop nudges x22; max_turns |
| PickPlaceCounterToDrawer-seed2 | never-approached | no RESULT.md | 0.373 | 3 | 0 | 2 | 160 | 27.8 | never closed gripper; loop nudges x46; max_turns |
| PickPlaceCounterToSink-seed0 | SUCCESS | success (agent) | 0.024 | 21 | 11 | 8 | 41 | 9.3 | SETTLE_MISS x2; loop nudges x1 |
| PickPlaceCounterToSink-seed1 | SUCCESS | success (agent) | 0.025 | 42 | 30 | 9 | 78 | 15.4 | IK_FAILED x3; SETTLE_MISS x2; CLAMP x4; loop nudges x3 |
| PickPlaceCounterToSink-seed2 | never-approached | no RESULT.md | 0.516 | 101 | 59 | 42 | 160 | 33.1 | IK_FAILED x4; SETTLE_MISS x2; never closed gripper; loop nudges x16; max_turns |
| PickPlaceStoveToCounter-seed0 | never-approached | no RESULT.md | 0.133 | 47 | 30 | 14 | 160 | 30.1 | IK_FAILED x3; CLAMP x13; loop nudges x9; max_turns |
| PickPlaceStoveToCounter-seed1 | never-approached | no RESULT.md | 0.137 | 129 | 110 | 15 | 154 | 46.0 | IK_FAILED x24; SETTLE_MISS x46; loop nudges x28; wall_clock; move_base x3 |
| PickPlaceStoveToCounter-seed2 | SUCCESS | success (agent) | 0.009 | 14 | 8 | 5 | 40 | 12.2 | SETTLE_MISS x2; loop nudges x1 |

stages: {"never-approached": 6, "SUCCESS": 3}
