| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | never-approached | no RESULT.md | 0.54 | 2 | 0 | 1 | 46 | 47.1 | never closed gripper; loop nudges x2; wall_clock |
| PickPlaceCounterToDrawer-seed1 | never-approached | no RESULT.md | 0.248 | 87 | 81 | 6 | 160 | 40.5 | never closed gripper; loop nudges x17; max_turns |
| PickPlaceCounterToDrawer-seed2 | never-approached | no RESULT.md | 0.373 | 59 | 6 | 23 | 160 | 23.0 | never closed gripper; loop nudges x10; max_turns; move_base x30 |
| PickPlaceCounterToSink-seed0 | lifted-not-placed | no RESULT.md | 0.005 | 89 | 80 | 6 | 144 | 45.7 | SETTLE_MISS x2; loop nudges x27; wall_clock |
| PickPlaceCounterToSink-seed1 | touched-no-lift | no RESULT.md | 0.033 | 52 | 26 | 16 | 94 | 45.6 | SETTLE_MISS x2; loop nudges x3; wall_clock |
| PickPlaceCounterToSink-seed2 | never-approached | no RESULT.md | 0.309 | 94 | 87 | 4 | 160 | 39.0 | SETTLE_MISS x4; never closed gripper; loop nudges x20; max_turns |
| PickPlaceStoveToCounter-seed0 | never-approached | no RESULT.md | 0.318 | 47 | 0 | 24 | 160 | 17.3 | never closed gripper; loop nudges x11; max_turns; move_base x23 |
| PickPlaceStoveToCounter-seed1 | never-approached | no RESULT.md | 0.412 | 3 | 0 | 2 | 37 | 46.4 | never closed gripper; wall_clock |
| PickPlaceStoveToCounter-seed2 | never-approached | no RESULT.md | 0.327 | 1 | 0 | 1 | 26 | 47.3 | never closed gripper; wall_clock |

stages: {"never-approached": 7, "lifted-not-placed": 1, "touched-no-lift": 1}
