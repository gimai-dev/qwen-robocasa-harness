| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |
|---|---|---|---|---|---|---|---|---|---|
| PickPlaceCounterToDrawer-seed0 | lifted-not-placed | no RESULT.md | 0.026 | 79 | 42 | 19 | 139 | 45.8 | SETTLE_MISS x7; loop nudges x1; wall_clock |
| PickPlaceCounterToDrawer-seed1 | touched-no-lift | no RESULT.md | 0.037 | 78 | 30 | 17 | 160 | 33.8 | loop nudges x7; max_turns |
| PickPlaceCounterToDrawer-seed2 | touched-no-lift | no RESULT.md | 0.042 | 102 | 56 | 18 | 160 | 39.5 | SETTLE_MISS x15; loop nudges x19; max_turns |
| PickPlaceCounterToSink-seed0 | touched-no-lift | no RESULT.md | 0.041 | 118 | 48 | 24 | 160 | 44.3 | loop nudges x10; max_turns |
| PickPlaceCounterToSink-seed1 | lifted-not-placed | no RESULT.md | 0.021 | 49 | 21 | 9 | 160 | 45.7 | SETTLE_MISS x2; CLAMP x2; loop nudges x9; max_turns; move_base x17 |
| PickPlaceCounterToSink-seed2 | touched-no-lift | no RESULT.md | 0.057 | 95 | 58 | 13 | 143 | 46.0 | SETTLE_MISS x4; loop nudges x7; wall_clock |
| PickPlaceStoveToCounter-seed0 | touched-no-lift | no RESULT.md | 0.026 | 143 | 72 | 19 | 160 | 33.9 | loop nudges x6; max_turns |
| PickPlaceStoveToCounter-seed1 | lifted-not-placed | no RESULT.md | 0.022 | 124 | 57 | 31 | 160 | 32.5 | SETTLE_MISS x19; loop nudges x23; max_turns; move_base x5 |
| PickPlaceStoveToCounter-seed2 | touched-no-lift | no RESULT.md | 0.028 | 142 | 71 | 36 | 160 | 28.4 | SETTLE_MISS x3; loop nudges x12; max_turns |

stages: {"lifted-not-placed": 3, "touched-no-lift": 6}
