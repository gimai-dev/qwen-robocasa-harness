| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | touched-no-lift | no RESULT.md | 0.023 | 84 | 77 | 6 | 160 | 32.5 | IK_FAILED x6; loop nudges x23; max_turns |
| PickPlaceCounterToDrawer-seed1 | touched-no-lift | no RESULT.md | 0.067 | 68 | 31 | 20 | 123 | 45.7 | loop nudges x2; wall_clock |
| PickPlaceCounterToDrawer-seed2 | lifted-not-placed | success (agent) | 0.007 | 19 | 12 | 5 | 44 | 13.1 | loop nudges x3 |
| PickPlaceCounterToSink-seed0 | touched-no-lift | no RESULT.md | 0.043 | 69 | 33 | 18 | 160 | 28.8 | IK_FAILED x8; loop nudges x14; max_turns; move_base x4 |
| PickPlaceCounterToSink-seed1 | never-approached | no RESULT.md | 0.412 | 293 | 0 | 2 | 160 | 37.7 | never closed gripper; loop nudges x14; max_turns; move_base x290 |
| PickPlaceCounterToSink-seed2 | never-approached | no RESULT.md | 0.231 | 82 | 20 | 21 | 160 | 39.8 | IK_FAILED x3; loop nudges x8; max_turns; move_base x13 |
| PickPlaceStoveToCounter-seed0 | SUCCESS | success (agent) | 0.03 | 20 | 13 | 4 | 33 | 10.7 | SETTLE_MISS x2; loop nudges x1 |
| PickPlaceStoveToCounter-seed1 | lifted-not-placed | no RESULT.md | 0.009 | 114 | 71 | 26 | 153 | 46.0 | SETTLE_MISS x42; loop nudges x35; wall_clock |
| PickPlaceStoveToCounter-seed2 | SUCCESS | success (agent) | 0.016 | 18 | 6 | 5 | 53 | 21.4 | loop nudges x1 |

stages: {"touched-no-lift": 3, "lifted-not-placed": 2, "never-approached": 2, "SUCCESS": 2}
