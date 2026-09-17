| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | SUCCESS | unclear (agent) | 0.019 | 76 | 51 | 13 | 102 | 23.3 | SETTLE_MISS x5; CLAMP x2; loop nudges x2 |
| PickPlaceCounterToDrawer-seed1 | lifted-not-placed | no RESULT.md | 0.004 | 108 | 87 | 13 | 160 | 36.2 | IK_FAILED x3; SETTLE_MISS x5; loop nudges x15; max_turns |
| PickPlaceCounterToDrawer-seed2 | lifted-not-placed | no RESULT.md | 0.011 | 107 | 44 | 23 | 160 | 33.8 | loop nudges x6; max_turns |
| PickPlaceCounterToSink-seed0 | SUCCESS | no RESULT.md | 0.022 | 83 | 64 | 13 | 160 | 42.4 | SETTLE_MISS x16; loop nudges x10; max_turns |
| PickPlaceCounterToSink-seed1 | lifted-not-placed | no RESULT.md | 0.028 | 69 | 45 | 17 | 160 | 41.9 | SETTLE_MISS x13; loop nudges x4; max_turns |
| PickPlaceCounterToSink-seed2 | never-approached | no RESULT.md | 0.362 | 264 | 0 | 132 | 160 | 29.4 | never closed gripper; loop nudges x14; max_turns; move_base x132 |
| PickPlaceStoveToCounter-seed0 | lifted-not-placed | no RESULT.md | 0.021 | 113 | 54 | 19 | 160 | 37.4 | IK_FAILED x3; loop nudges x7; max_turns; move_base x6 |
| PickPlaceStoveToCounter-seed1 | never-approached | no RESULT.md | 0.412 | 140 | 0 | 3 | 160 | 36.1 | never closed gripper; loop nudges x20; max_turns; move_base x137 |
| PickPlaceStoveToCounter-seed2 | lifted-not-placed | success (agent) | 0.01 | 27 | 15 | 8 | 84 | 16.8 | SETTLE_MISS x9; loop nudges x11 |

stages: {"SUCCESS": 2, "lifted-not-placed": 5, "never-approached": 2}
