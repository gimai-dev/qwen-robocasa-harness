# Results

Runs: 75 from /home/jli/state/qwen-direct/matrix/phaseD-singles

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| clean | ee | short | PickPlaceCounterToDrawer | 0/5 | {'stop': 3, 'step_budget': 2} |
| clean | ee | short | PickPlaceCounterToSink | 0/5 | {'step_budget': 2, 'no_progress': 1, 'stop': 2} |
| clean | ee | short | PickPlaceStoveToCounter | 0/5 | {'step_budget': 5} |
| clean | ee | short | **all** | 0/15 | |
| h1 | ee | short | PickPlaceCounterToDrawer | 0/5 | {'stop': 1, 'step_budget': 4} |
| h1 | ee | short | PickPlaceCounterToSink | 0/5 | {'step_budget': 2, 'stop': 2, 'no_progress': 1} |
| h1 | ee | short | PickPlaceStoveToCounter | 0/5 | {'step_budget': 5} |
| h1 | ee | short | **all** | 0/15 | |
| h2 | ee | short | PickPlaceCounterToDrawer | 0/5 | {'step_budget': 4, 'stop': 1} |
| h2 | ee | short | PickPlaceCounterToSink | 0/5 | {'stop': 4, 'step_budget': 1} |
| h2 | ee | short | PickPlaceStoveToCounter | 0/5 | {'step_budget': 4, 'no_progress': 1} |
| h2 | ee | short | **all** | 0/15 | |
| h3 | ee | short | PickPlaceCounterToDrawer | 0/5 | {'step_budget': 4, 'stop': 1} |
| h3 | ee | short | PickPlaceCounterToSink | 0/5 | {'step_budget': 5} |
| h3 | ee | short | PickPlaceStoveToCounter | 0/5 | {'step_budget': 5} |
| h3 | ee | short | **all** | 0/15 | |
| h4 | ee | short | PickPlaceCounterToDrawer | 0/5 | {'step_budget': 3, 'stop': 2} |
| h4 | ee | short | PickPlaceCounterToSink | 0/5 | {'step_budget': 2, 'stop': 3} |
| h4 | ee | short | PickPlaceStoveToCounter | 0/5 | {'stop': 2, 'step_budget': 3} |
| h4 | ee | short | **all** | 0/15 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|
| h1 | ee | short | 15 | 0/15 | 0/15 | +0.000 | [+0.000, +0.000] | 0/0 |
| h2 | ee | short | 15 | 0/15 | 0/15 | +0.000 | [+0.000, +0.000] | 0/0 |
| h3 | ee | short | 15 | 0/15 | 0/15 | +0.000 | [+0.000, +0.000] | 0/0 |
| h4 | ee | short | 15 | 0/15 | 0/15 | +0.000 | [+0.000, +0.000] | 0/0 |

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | 15 | 815 | 68.5 | 27.5 | 68.5 | 260516 | 10121 | 257 | 275 | 591 |
| h1 | ee | short | 15 | 856 | 81.4 | 38.4 | 81.4 | 332331 | 11918 | 286 | 245 | 598 |
| h2 | ee | short | 15 | 819 | 68.1 | 26.9 | 68.1 | 264860 | 10473 | 246 | 222 | 532 |
| h3 | ee | short | 15 | 899 | 45.5 | 0.5 | 91.1 | 289518 | 19346 | 409 | 183 | 664 |
| h4 | ee | short | 15 | 861 | 68.6 | 24.9 | 68.6 | 268941 | 10445 | 247 | 235 | 543 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | PickPlaceCounterToDrawer | 10 | False | step_budget | 900 | 106 | 1067 | PickPlaceCounterToDrawer-s10-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 11 | False | stop | 880 | 58 | 530 | PickPlaceCounterToDrawer-s11-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 12 | False | stop | 860 | 70 | 676 | PickPlaceCounterToDrawer-s12-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 13 | False | stop | 880 | 52 | 520 | PickPlaceCounterToDrawer-s13-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 14 | False | step_budget | 900 | 87 | 669 | PickPlaceCounterToDrawer-s14-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 10 | False | no_progress | 100 | 19 | 131 | PickPlaceCounterToSink-s10-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 11 | False | step_budget | 900 | 80 | 704 | PickPlaceCounterToSink-s11-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 12 | False | step_budget | 900 | 65 | 574 | PickPlaceCounterToSink-s12-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 13 | False | stop | 520 | 35 | 317 | PickPlaceCounterToSink-s13-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 14 | False | stop | 880 | 61 | 535 | PickPlaceCounterToSink-s14-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 10 | False | step_budget | 900 | 79 | 747 | PickPlaceStoveToCounter-s10-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 11 | False | step_budget | 900 | 71 | 558 | PickPlaceStoveToCounter-s11-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 12 | False | step_budget | 900 | 79 | 629 | PickPlaceStoveToCounter-s12-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 13 | False | step_budget | 900 | 75 | 619 | PickPlaceStoveToCounter-s13-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 14 | False | step_budget | 900 | 91 | 585 | PickPlaceStoveToCounter-s14-ee-short-clean |
| h1 | ee | short | PickPlaceCounterToDrawer | 10 | False | step_budget | 900 | 96 | 774 | PickPlaceCounterToDrawer-s10-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToDrawer | 11 | False | step_budget | 900 | 54 | 426 | PickPlaceCounterToDrawer-s11-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToDrawer | 12 | False | stop | 880 | 69 | 542 | PickPlaceCounterToDrawer-s12-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToDrawer | 13 | False | step_budget | 900 | 72 | 524 | PickPlaceCounterToDrawer-s13-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToDrawer | 14 | False | step_budget | 900 | 101 | 703 | PickPlaceCounterToDrawer-s14-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 10 | False | stop | 880 | 93 | 552 | PickPlaceCounterToSink-s10-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 11 | False | step_budget | 900 | 74 | 534 | PickPlaceCounterToSink-s11-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 12 | False | no_progress | 500 | 54 | 406 | PickPlaceCounterToSink-s12-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 13 | False | stop | 680 | 68 | 510 | PickPlaceCounterToSink-s13-ee-short-h1 |
| h1 | ee | short | PickPlaceCounterToSink | 14 | False | step_budget | 900 | 94 | 650 | PickPlaceCounterToSink-s14-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 10 | False | step_budget | 900 | 89 | 718 | PickPlaceStoveToCounter-s10-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 11 | False | step_budget | 900 | 70 | 539 | PickPlaceStoveToCounter-s11-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 12 | False | step_budget | 900 | 104 | 790 | PickPlaceStoveToCounter-s12-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 13 | False | step_budget | 900 | 96 | 733 | PickPlaceStoveToCounter-s13-ee-short-h1 |
| h1 | ee | short | PickPlaceStoveToCounter | 14 | False | step_budget | 900 | 87 | 570 | PickPlaceStoveToCounter-s14-ee-short-h1 |
| h2 | ee | short | PickPlaceCounterToDrawer | 10 | False | step_budget | 900 | 63 | 543 | PickPlaceCounterToDrawer-s10-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 11 | False | stop | 880 | 58 | 478 | PickPlaceCounterToDrawer-s11-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 12 | False | step_budget | 900 | 78 | 669 | PickPlaceCounterToDrawer-s12-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 13 | False | step_budget | 900 | 56 | 488 | PickPlaceCounterToDrawer-s13-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 14 | False | step_budget | 900 | 78 | 591 | PickPlaceCounterToDrawer-s14-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 10 | False | stop | 860 | 90 | 538 | PickPlaceCounterToSink-s10-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 11 | False | stop | 880 | 68 | 551 | PickPlaceCounterToSink-s11-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 12 | False | stop | 500 | 47 | 355 | PickPlaceCounterToSink-s12-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 13 | False | step_budget | 900 | 72 | 540 | PickPlaceCounterToSink-s13-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 14 | False | stop | 860 | 73 | 533 | PickPlaceCounterToSink-s14-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 10 | False | step_budget | 900 | 66 | 615 | PickPlaceStoveToCounter-s10-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 11 | False | step_budget | 900 | 78 | 598 | PickPlaceStoveToCounter-s11-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 12 | False | step_budget | 900 | 71 | 600 | PickPlaceStoveToCounter-s12-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 13 | False | no_progress | 200 | 33 | 238 | PickPlaceStoveToCounter-s13-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 14 | False | step_budget | 900 | 91 | 645 | PickPlaceStoveToCounter-s14-ee-short-h2 |
| h3 | ee | short | PickPlaceCounterToDrawer | 10 | False | step_budget | 900 | 45 | 702 | PickPlaceCounterToDrawer-s10-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToDrawer | 11 | False | stop | 880 | 45 | 635 | PickPlaceCounterToDrawer-s11-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToDrawer | 12 | False | step_budget | 900 | 47 | 740 | PickPlaceCounterToDrawer-s12-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToDrawer | 13 | False | step_budget | 900 | 45 | 643 | PickPlaceCounterToDrawer-s13-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToDrawer | 14 | False | step_budget | 900 | 45 | 618 | PickPlaceCounterToDrawer-s14-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 10 | False | step_budget | 900 | 49 | 655 | PickPlaceCounterToSink-s10-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 11 | False | step_budget | 900 | 46 | 664 | PickPlaceCounterToSink-s11-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 12 | False | step_budget | 900 | 45 | 639 | PickPlaceCounterToSink-s12-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 13 | False | step_budget | 900 | 45 | 710 | PickPlaceCounterToSink-s13-ee-short-h3 |
| h3 | ee | short | PickPlaceCounterToSink | 14 | False | step_budget | 900 | 45 | 613 | PickPlaceCounterToSink-s14-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 10 | False | step_budget | 900 | 46 | 707 | PickPlaceStoveToCounter-s10-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 11 | False | step_budget | 900 | 45 | 612 | PickPlaceStoveToCounter-s11-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 12 | False | step_budget | 900 | 45 | 625 | PickPlaceStoveToCounter-s12-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 13 | False | step_budget | 900 | 45 | 715 | PickPlaceStoveToCounter-s13-ee-short-h3 |
| h3 | ee | short | PickPlaceStoveToCounter | 14 | False | step_budget | 900 | 45 | 685 | PickPlaceStoveToCounter-s14-ee-short-h3 |
| h4 | ee | short | PickPlaceCounterToDrawer | 10 | False | step_budget | 900 | 96 | 856 | PickPlaceCounterToDrawer-s10-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToDrawer | 11 | False | stop | 880 | 51 | 430 | PickPlaceCounterToDrawer-s11-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToDrawer | 12 | False | step_budget | 900 | 78 | 647 | PickPlaceCounterToDrawer-s12-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToDrawer | 13 | False | step_budget | 900 | 49 | 415 | PickPlaceCounterToDrawer-s13-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToDrawer | 14 | False | stop | 880 | 77 | 533 | PickPlaceCounterToDrawer-s14-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 10 | False | step_budget | 900 | 54 | 377 | PickPlaceCounterToSink-s10-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 11 | False | stop | 890 | 55 | 446 | PickPlaceCounterToSink-s11-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 12 | False | stop | 440 | 25 | 217 | PickPlaceCounterToSink-s12-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 13 | False | stop | 860 | 72 | 563 | PickPlaceCounterToSink-s13-ee-short-h4 |
| h4 | ee | short | PickPlaceCounterToSink | 14 | False | step_budget | 900 | 71 | 538 | PickPlaceCounterToSink-s14-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 10 | False | step_budget | 900 | 68 | 598 | PickPlaceStoveToCounter-s10-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 11 | False | stop | 890 | 67 | 531 | PickPlaceStoveToCounter-s11-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 12 | False | step_budget | 900 | 106 | 819 | PickPlaceStoveToCounter-s12-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 13 | False | stop | 880 | 91 | 671 | PickPlaceStoveToCounter-s13-ee-short-h4 |
| h4 | ee | short | PickPlaceStoveToCounter | 14 | False | step_budget | 900 | 69 | 505 | PickPlaceStoveToCounter-s14-ee-short-h4 |
