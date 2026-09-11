# Results

Runs: 72 from /home/jli/state/qwen-direct/matrix/phaseC-ee-short

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| clean | ee | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 2, 'stop': 1} |
| clean | ee | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| clean | ee | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| clean | ee | short | **all** | 0/9 | |
| h1 | ee | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 2, 'stop': 1} |
| h1 | ee | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| h1 | ee | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| h1 | ee | short | **all** | 0/9 | |
| h2 | ee | short | PickPlaceCounterToDrawer | 0/3 | {'stop': 2, 'step_budget': 1} |
| h2 | ee | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| h2 | ee | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| h2 | ee | short | **all** | 0/9 | |
| h3 | ee | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 3} |
| h3 | ee | short | PickPlaceCounterToSink | 0/3 | {'step_budget': 3} |
| h3 | ee | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 2, 'stop': 1} |
| h3 | ee | short | **all** | 0/9 | |
| h4 | ee | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 1, 'stop': 2} |
| h4 | ee | short | PickPlaceCounterToSink | 0/3 | {'stop': 3} |
| h4 | ee | short | PickPlaceStoveToCounter | 0/3 | {'stop': 2, 'step_budget': 1} |
| h4 | ee | short | **all** | 0/9 | |
| h4c | ee | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 3} |
| h4c | ee | short | PickPlaceCounterToSink | 0/3 | {'step_budget': 1, 'stop': 2} |
| h4c | ee | short | PickPlaceStoveToCounter | 0/3 | {'wall_budget': 1, 'step_budget': 2} |
| h4c | ee | short | **all** | 0/9 | |
| h5 | ee | short | PickPlaceCounterToDrawer | 0/3 | {'step_budget': 2, 'stop': 1} |
| h5 | ee | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| h5 | ee | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 2, 'stop': 1} |
| h5 | ee | short | **all** | 0/9 | |
| h8 | ee | short | PickPlaceCounterToDrawer | 0/3 | {'stop': 2, 'step_budget': 1} |
| h8 | ee | short | PickPlaceCounterToSink | 0/3 | {'step_budget': 1, 'stop': 2} |
| h8 | ee | short | PickPlaceStoveToCounter | 0/3 | {'step_budget': 3} |
| h8 | ee | short | **all** | 0/9 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|
| h1 | ee | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |
| h2 | ee | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |
| h3 | ee | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |
| h4 | ee | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |
| h4c | ee | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |
| h5 | ee | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |
| h8 | ee | short | 9 | 0/9 | 0/9 | +0.000 | [+0.000, +0.000] | 0/0 |

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | 9 | 893 | 75.4 | 30.4 | 75.4 | 294102 | 11201 | 270 | 261 | 594 |
| h1 | ee | short | 9 | 893 | 71.1 | 26.1 | 71.1 | 301040 | 10343 | 251 | 229 | 549 |
| h2 | ee | short | 9 | 804 | 65.0 | 24.3 | 65.0 | 260623 | 9661 | 232 | 237 | 530 |
| h3 | ee | short | 9 | 898 | 45.4 | 0.4 | 90.9 | 291855 | 19764 | 418 | 199 | 689 |
| h4 | ee | short | 9 | 799 | 67.3 | 26.4 | 67.3 | 271245 | 10091 | 243 | 240 | 538 |
| h4c | ee | short | 9 | 501 | 137.1 | 51.3 | 137.1 | 551001 | 19762 | 474 | 497 | 1018 |
| h5 | ee | short | 9 | 889 | 56.1 | 11.2 | 56.1 | 235024 | 21385 | 442 | 240 | 743 |
| h8 | ee | short | 9 | 818 | 64.2 | 22.9 | 64.2 | 264421 | 14223 | 315 | 256 | 627 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | PickPlaceCounterToDrawer | 0 | False | stop | 880 | 58 | 491 | PickPlaceCounterToDrawer-s0-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 900 | 64 | 518 | PickPlaceCounterToDrawer-s1-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 77 | 592 | PickPlaceCounterToDrawer-s2-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 0 | False | stop | 880 | 52 | 364 | PickPlaceCounterToSink-s0-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 1 | False | stop | 880 | 78 | 620 | PickPlaceCounterToSink-s1-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 2 | False | step_budget | 900 | 83 | 642 | PickPlaceCounterToSink-s2-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 77 | 634 | PickPlaceStoveToCounter-s0-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 99 | 767 | PickPlaceStoveToCounter-s1-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 91 | 720 | PickPlaceStoveToCounter-s2-ee-short-clean |
| h1 | ee | short | PickPlaceCounterToDrawer | 0 | False | step_budget | 900 | 62 | 521 | PickPlaceCounterToDrawer-s0-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToDrawer | 1 | False | stop | 880 | 96 | 699 | PickPlaceCounterToDrawer-s1-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 47 | 390 | PickPlaceCounterToDrawer-s2-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 48 | 349 | PickPlaceCounterToSink-s0-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 1 | False | stop | 880 | 49 | 396 | PickPlaceCounterToSink-s1-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 2 | False | stop | 880 | 60 | 473 | PickPlaceCounterToSink-s2-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 89 | 688 | PickPlaceStoveToCounter-s0-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 102 | 753 | PickPlaceStoveToCounter-s1-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 87 | 671 | PickPlaceStoveToCounter-s2-ee-short-h1 |
| h2 | ee | short | PickPlaceCounterToDrawer | 0 | False | stop | 880 | 56 | 485 | PickPlaceCounterToDrawer-s0-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 1 | False | stop | 880 | 74 | 660 | PickPlaceCounterToDrawer-s1-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 78 | 619 | PickPlaceCounterToDrawer-s2-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 55 | 394 | PickPlaceCounterToSink-s0-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 1 | False | stop | 740 | 67 | 516 | PickPlaceCounterToSink-s1-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 2 | False | stop | 240 | 20 | 180 | PickPlaceCounterToSink-s2-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 71 | 585 | PickPlaceStoveToCounter-s0-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 72 | 593 | PickPlaceStoveToCounter-s1-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 92 | 739 | PickPlaceStoveToCounter-s2-ee-short-h2 |
| h3 | ee | short | PickPlaceCounterToDrawer | 0 | False | step_budget | 900 | 46 | 729 | PickPlaceCounterToDrawer-s0-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 900 | 45 | 689 | PickPlaceCounterToDrawer-s1-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 45 | 671 | PickPlaceCounterToDrawer-s2-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 47 | 664 | PickPlaceCounterToSink-s0-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 1 | False | step_budget | 900 | 45 | 660 | PickPlaceCounterToSink-s1-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 2 | False | step_budget | 900 | 45 | 678 | PickPlaceCounterToSink-s2-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 45 | 703 | PickPlaceStoveToCounter-s0-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 45 | 681 | PickPlaceStoveToCounter-s1-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 2 | False | stop | 880 | 46 | 730 | PickPlaceStoveToCounter-s2-ee-short-h3 |
| h4 | ee | short | PickPlaceCounterToDrawer | 0 | False | step_budget | 900 | 62 | 524 | PickPlaceCounterToDrawer-s0-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToDrawer | 1 | False | stop | 880 | 103 | 807 | PickPlaceCounterToDrawer-s1-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToDrawer | 2 | False | stop | 880 | 78 | 620 | PickPlaceCounterToDrawer-s2-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 0 | False | stop | 560 | 30 | 243 | PickPlaceCounterToSink-s0-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 1 | False | stop | 820 | 65 | 528 | PickPlaceCounterToSink-s1-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 2 | False | stop | 490 | 29 | 259 | PickPlaceCounterToSink-s2-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 0 | False | stop | 880 | 70 | 575 | PickPlaceStoveToCounter-s0-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 83 | 624 | PickPlaceStoveToCounter-s1-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 2 | False | stop | 880 | 86 | 664 | PickPlaceStoveToCounter-s2-ee-short-h4 |
| h4c | ee | short | PickPlaceCounterToDrawer | 0 | False | step_budget | 695 | 152 | 1202 | PickPlaceCounterToDrawer-s0-ee-short-h4c |
| h4c | ee | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 395 | 164 | 1202 | PickPlaceCounterToDrawer-s1-ee-short-h4c |
| h4c | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 460 | 164 | 1202 | PickPlaceCounterToDrawer-s2-ee-short-h4c |
| h4c | ee | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 120 | 859 | PickPlaceCounterToSink-s0-ee-short-h4c |
| h4c | ee | short | PickPlaceCounterToSink | 1 | False | stop | 0 | 1 | 9 | PickPlaceCounterToSink-s1-ee-short-h4c |
| h4c | ee | short | PickPlaceCounterToSink | 2 | False | stop | 695 | 152 | 1074 | PickPlaceCounterToSink-s2-ee-short-h4c |
| h4c | ee | short | PickPlaceStoveToCounter | 0 | False | step_budget | 505 | 158 | 1202 | PickPlaceStoveToCounter-s0-ee-short-h4c |
| h4c | ee | short | PickPlaceStoveToCounter | 1 | False | wall_budget | 415 | 159 | 1206 | PickPlaceStoveToCounter-s1-ee-short-h4c |
| h4c | ee | short | PickPlaceStoveToCounter | 2 | False | step_budget | 445 | 164 | 1203 | PickPlaceStoveToCounter-s2-ee-short-h4c |
| h5 | ee | short | PickPlaceCounterToDrawer | 0 | False | step_budget | 900 | 53 | 653 | PickPlaceCounterToDrawer-s0-ee-short-h5 |
| h5 | ee | short | PickPlaceCounterToDrawer | 1 | False | stop | 860 | 63 | 860 | PickPlaceCounterToDrawer-s1-ee-short-h5 |
| h5 | ee | short | PickPlaceCounterToDrawer | 2 | False | step_budget | 900 | 74 | 963 | PickPlaceCounterToDrawer-s2-ee-short-h5 |
| h5 | ee | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 46 | 572 | PickPlaceCounterToSink-s0-ee-short-h5 |
| h5 | ee | short | PickPlaceCounterToSink | 1 | False | stop | 880 | 51 | 696 | PickPlaceCounterToSink-s1-ee-short-h5 |
| h5 | ee | short | PickPlaceCounterToSink | 2 | False | stop | 880 | 57 | 803 | PickPlaceCounterToSink-s2-ee-short-h5 |
| h5 | ee | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 49 | 672 | PickPlaceStoveToCounter-s0-ee-short-h5 |
| h5 | ee | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 58 | 741 | PickPlaceStoveToCounter-s1-ee-short-h5 |
| h5 | ee | short | PickPlaceStoveToCounter | 2 | False | stop | 880 | 54 | 723 | PickPlaceStoveToCounter-s2-ee-short-h5 |
| h8 | ee | short | PickPlaceCounterToDrawer | 0 | False | stop | 880 | 67 | 710 | PickPlaceCounterToDrawer-s0-ee-short-h8 |
| h8 | ee | short | PickPlaceCounterToDrawer | 1 | False | step_budget | 900 | 77 | 804 | PickPlaceCounterToDrawer-s1-ee-short-h8 |
| h8 | ee | short | PickPlaceCounterToDrawer | 2 | False | stop | 880 | 75 | 777 | PickPlaceCounterToDrawer-s2-ee-short-h8 |
| h8 | ee | short | PickPlaceCounterToSink | 0 | False | step_budget | 900 | 47 | 392 | PickPlaceCounterToSink-s0-ee-short-h8 |
| h8 | ee | short | PickPlaceCounterToSink | 1 | False | stop | 820 | 63 | 616 | PickPlaceCounterToSink-s1-ee-short-h8 |
| h8 | ee | short | PickPlaceCounterToSink | 2 | False | stop | 280 | 24 | 231 | PickPlaceCounterToSink-s2-ee-short-h8 |
| h8 | ee | short | PickPlaceStoveToCounter | 0 | False | step_budget | 900 | 63 | 658 | PickPlaceStoveToCounter-s0-ee-short-h8 |
| h8 | ee | short | PickPlaceStoveToCounter | 1 | False | step_budget | 900 | 80 | 767 | PickPlaceStoveToCounter-s1-ee-short-h8 |
| h8 | ee | short | PickPlaceStoveToCounter | 2 | False | step_budget | 900 | 82 | 688 | PickPlaceStoveToCounter-s2-ee-short-h8 |
