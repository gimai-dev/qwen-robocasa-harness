| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | lifted-not-placed | no RESULT.md | 0.106 | 112 | 92 | 9 | 160 | 45.4 | IK_FAILED x9; SETTLE_MISS x4; loop nudges x19; max_turns; move_base x11 |
| PickPlaceCounterToDrawer-seed1 | approached-no-contact | no RESULT.md | 0.042 | 123 | 58 | 24 | 160 | 43.2 | SETTLE_MISS x11; loop nudges x8; max_turns; move_base x3 |
| PickPlaceCounterToDrawer-seed2 | approached-no-contact | no RESULT.md | 0.069 | 35 | 16 | 7 | 70 | 46.8 | IK_FAILED x4; SETTLE_MISS x5; loop nudges x3; wall_clock |
| PickPlaceCounterToSink-seed0 | lifted-not-placed | no RESULT.md | 0.005 | 99 | 80 | 11 | 135 | 45.6 | SETTLE_MISS x14; loop nudges x10; wall_clock |
| PickPlaceCounterToSink-seed1 | never-approached | no RESULT.md | 0.633 | 163 | 78 | 81 | 144 | 45.5 | IK_FAILED x69; CLAMP x2; never closed gripper; loop nudges x13; wall_clock; move_base x3 |
| PickPlaceCounterToSink-seed2 | lifted-not-placed | no RESULT.md | 0.055 | 125 | 39 | 30 | 145 | 45.8 | IK_FAILED x6; CLAMP x2; loop nudges x7; wall_clock; move_base x10 |
| PickPlaceStoveToCounter-seed0 | touched-no-lift | no RESULT.md | 0.028 | 172 | 71 | 42 | 158 | 45.6 | SETTLE_MISS x10; CLAMP x2; loop nudges x9; wall_clock; move_base x4 |
| PickPlaceStoveToCounter-seed1 | never-approached | no RESULT.md | 0.397 | 80 | 41 | 15 | 160 | 39.6 | SETTLE_MISS x18; CLAMP x3; loop nudges x13; max_turns |
| PickPlaceStoveToCounter-seed2 | SUCCESS | success (agent) | 0.018 | 11 | 6 | 3 | 25 | 5.7 | loop nudges x1 |

stages: {"lifted-not-placed": 3, "approached-no-contact": 2, "never-approached": 2, "touched-no-lift": 1, "SUCCESS": 1}
