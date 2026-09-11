# Results

Runs: 120 from /home/jli/state/qwen-direct/matrix/phaseE-test

## Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| clean | ee | short | PickPlaceCounterToDrawer | 0/20 | {'stop': 9, 'step_budget': 9, 'infrastructure_error': 2} |
| clean | ee | short | PickPlaceCounterToSink | 1/20 | {'step_budget': 15, 'stop': 5} |
| clean | ee | short | PickPlaceStoveToCounter | 0/20 | {'stop': 2, 'step_budget': 18} |
| clean | ee | short | **all** | 1/60 | |
| h2 | ee | short | PickPlaceCounterToDrawer | 0/20 | {'step_budget': 10, 'stop': 10} |
| h2 | ee | short | PickPlaceCounterToSink | 1/20 | {'step_budget': 11, 'stop': 9} |
| h2 | ee | short | PickPlaceStoveToCounter | 0/20 | {'step_budget': 16, 'stop': 4} |
| h2 | ee | short | **all** | 1/60 | |

## Paired differences vs baseline (same task+seed starts)

| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |
|---|---|---|---|---|---|---|---|---|
| h2 | ee | short | 60 | 1/60 | 1/60 | +0.000 | [+0.000, +0.000] | 0/0 |

## Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | 60 | 861 | 66.1 | 22.8 | 66.1 | 254210 | 9649 | 230 | 230 | 517 |
| h2 | ee | short | 60 | 866 | 65.1 | 21.4 | 65.1 | 256333 | 9803 | 233 | 242 | 540 |

## Per-run outcomes

| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | PickPlaceCounterToDrawer | 100 | False | step_budget | 900 | 88 | 575 | PickPlaceCounterToDrawer-s100-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 101 | False | step_budget | 900 | 84 | 609 | PickPlaceCounterToDrawer-s101-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 102 | False | stop | 880 | 51 | 445 | PickPlaceCounterToDrawer-s102-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 103 | False | stop | 880 | 56 | 440 | PickPlaceCounterToDrawer-s103-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 104 | False | stop | 880 | 65 | 549 | PickPlaceCounterToDrawer-s104-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 105 | False | stop | 840 | 61 | 524 | PickPlaceCounterToDrawer-s105-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 106 | False | step_budget | 900 | 84 | 543 | PickPlaceCounterToDrawer-s106-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 107 | False | stop | 880 | 90 | 600 | PickPlaceCounterToDrawer-s107-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 108 | None | infrastructure_error | 560 | 44 | 342 | PickPlaceCounterToDrawer-s108-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 109 | False | stop | 880 | 84 | 628 | PickPlaceCounterToDrawer-s109-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 110 | False | step_budget | 900 | 73 | 607 | PickPlaceCounterToDrawer-s110-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 111 | False | stop | 880 | 79 | 606 | PickPlaceCounterToDrawer-s111-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 112 | False | step_budget | 900 | 97 | 765 | PickPlaceCounterToDrawer-s112-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 113 | False | step_budget | 900 | 93 | 571 | PickPlaceCounterToDrawer-s113-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 114 | False | step_budget | 900 | 63 | 528 | PickPlaceCounterToDrawer-s114-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 115 | False | stop | 880 | 56 | 453 | PickPlaceCounterToDrawer-s115-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 116 | False | step_budget | 900 | 66 | 535 | PickPlaceCounterToDrawer-s116-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 117 | None | infrastructure_error | 280 | 23 | 190 | PickPlaceCounterToDrawer-s117-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 118 | False | step_budget | 900 | 65 | 552 | PickPlaceCounterToDrawer-s118-ee-short-clean |
| clean | ee | short | PickPlaceCounterToDrawer | 119 | False | stop | 880 | 64 | 561 | PickPlaceCounterToDrawer-s119-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 100 | True | stop | 500 | 30 | 263 | PickPlaceCounterToSink-s100-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 101 | False | step_budget | 900 | 66 | 558 | PickPlaceCounterToSink-s101-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 102 | False | step_budget | 900 | 50 | 433 | PickPlaceCounterToSink-s102-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 103 | False | step_budget | 900 | 60 | 536 | PickPlaceCounterToSink-s103-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 104 | False | step_budget | 900 | 56 | 439 | PickPlaceCounterToSink-s104-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 105 | False | step_budget | 900 | 49 | 424 | PickPlaceCounterToSink-s105-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 106 | False | step_budget | 900 | 61 | 472 | PickPlaceCounterToSink-s106-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 107 | False | step_budget | 900 | 60 | 480 | PickPlaceCounterToSink-s107-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 108 | False | step_budget | 900 | 55 | 436 | PickPlaceCounterToSink-s108-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 109 | False | stop | 860 | 62 | 467 | PickPlaceCounterToSink-s109-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 110 | False | step_budget | 900 | 71 | 451 | PickPlaceCounterToSink-s110-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 111 | False | stop | 880 | 61 | 537 | PickPlaceCounterToSink-s111-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 112 | False | step_budget | 900 | 64 | 495 | PickPlaceCounterToSink-s112-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 113 | False | stop | 500 | 38 | 321 | PickPlaceCounterToSink-s113-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 114 | False | step_budget | 900 | 66 | 520 | PickPlaceCounterToSink-s114-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 115 | False | step_budget | 900 | 76 | 575 | PickPlaceCounterToSink-s115-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 116 | False | step_budget | 900 | 67 | 537 | PickPlaceCounterToSink-s116-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 117 | False | stop | 660 | 40 | 337 | PickPlaceCounterToSink-s117-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 118 | False | step_budget | 900 | 85 | 729 | PickPlaceCounterToSink-s118-ee-short-clean |
| clean | ee | short | PickPlaceCounterToSink | 119 | False | step_budget | 900 | 54 | 446 | PickPlaceCounterToSink-s119-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 100 | False | step_budget | 900 | 55 | 477 | PickPlaceStoveToCounter-s100-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 101 | False | step_budget | 900 | 56 | 464 | PickPlaceStoveToCounter-s101-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 102 | False | step_budget | 900 | 56 | 466 | PickPlaceStoveToCounter-s102-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 103 | False | step_budget | 900 | 49 | 432 | PickPlaceStoveToCounter-s103-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 104 | False | step_budget | 900 | 82 | 640 | PickPlaceStoveToCounter-s104-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 105 | False | step_budget | 900 | 96 | 662 | PickPlaceStoveToCounter-s105-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 106 | False | step_budget | 900 | 49 | 448 | PickPlaceStoveToCounter-s106-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 107 | False | step_budget | 900 | 72 | 550 | PickPlaceStoveToCounter-s107-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 108 | False | step_budget | 900 | 82 | 622 | PickPlaceStoveToCounter-s108-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 109 | False | stop | 880 | 67 | 541 | PickPlaceStoveToCounter-s109-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 110 | False | step_budget | 900 | 71 | 561 | PickPlaceStoveToCounter-s110-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 111 | False | step_budget | 900 | 92 | 653 | PickPlaceStoveToCounter-s111-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 112 | False | stop | 880 | 52 | 432 | PickPlaceStoveToCounter-s112-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 113 | False | step_budget | 900 | 77 | 615 | PickPlaceStoveToCounter-s113-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 114 | False | step_budget | 900 | 76 | 575 | PickPlaceStoveToCounter-s114-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 115 | False | step_budget | 900 | 82 | 654 | PickPlaceStoveToCounter-s115-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 116 | False | step_budget | 900 | 79 | 580 | PickPlaceStoveToCounter-s116-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 117 | False | step_budget | 900 | 46 | 377 | PickPlaceStoveToCounter-s117-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 118 | False | step_budget | 900 | 82 | 654 | PickPlaceStoveToCounter-s118-ee-short-clean |
| clean | ee | short | PickPlaceStoveToCounter | 119 | False | step_budget | 900 | 89 | 551 | PickPlaceStoveToCounter-s119-ee-short-clean |
| h2 | ee | short | PickPlaceCounterToDrawer | 100 | False | stop | 880 | 70 | 540 | PickPlaceCounterToDrawer-s100-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 101 | False | step_budget | 900 | 67 | 579 | PickPlaceCounterToDrawer-s101-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 102 | False | step_budget | 900 | 59 | 543 | PickPlaceCounterToDrawer-s102-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 103 | False | stop | 880 | 55 | 504 | PickPlaceCounterToDrawer-s103-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 104 | False | step_budget | 900 | 60 | 544 | PickPlaceCounterToDrawer-s104-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 105 | False | step_budget | 900 | 58 | 526 | PickPlaceCounterToDrawer-s105-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 106 | False | step_budget | 900 | 63 | 431 | PickPlaceCounterToDrawer-s106-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 107 | False | stop | 880 | 89 | 610 | PickPlaceCounterToDrawer-s107-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 108 | False | step_budget | 900 | 62 | 473 | PickPlaceCounterToDrawer-s108-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 109 | False | stop | 880 | 81 | 642 | PickPlaceCounterToDrawer-s109-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 110 | False | step_budget | 900 | 64 | 573 | PickPlaceCounterToDrawer-s110-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 111 | False | step_budget | 900 | 79 | 660 | PickPlaceCounterToDrawer-s111-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 112 | False | stop | 880 | 93 | 780 | PickPlaceCounterToDrawer-s112-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 113 | False | step_budget | 900 | 78 | 567 | PickPlaceCounterToDrawer-s113-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 114 | False | stop | 880 | 55 | 518 | PickPlaceCounterToDrawer-s114-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 115 | False | stop | 880 | 58 | 488 | PickPlaceCounterToDrawer-s115-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 116 | False | step_budget | 900 | 62 | 560 | PickPlaceCounterToDrawer-s116-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 117 | False | stop | 880 | 61 | 549 | PickPlaceCounterToDrawer-s117-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 118 | False | stop | 880 | 64 | 580 | PickPlaceCounterToDrawer-s118-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToDrawer | 119 | False | stop | 880 | 50 | 423 | PickPlaceCounterToDrawer-s119-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 100 | True | step_budget | 900 | 56 | 471 | PickPlaceCounterToSink-s100-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 101 | False | step_budget | 900 | 70 | 584 | PickPlaceCounterToSink-s101-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 102 | False | step_budget | 900 | 56 | 485 | PickPlaceCounterToSink-s102-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 103 | False | step_budget | 900 | 62 | 556 | PickPlaceCounterToSink-s103-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 104 | False | stop | 880 | 60 | 489 | PickPlaceCounterToSink-s104-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 105 | False | stop | 500 | 36 | 293 | PickPlaceCounterToSink-s105-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 106 | False | stop | 120 | 10 | 80 | PickPlaceCounterToSink-s106-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 107 | False | stop | 880 | 60 | 496 | PickPlaceCounterToSink-s107-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 108 | False | stop | 880 | 76 | 622 | PickPlaceCounterToSink-s108-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 109 | False | step_budget | 900 | 68 | 542 | PickPlaceCounterToSink-s109-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 110 | False | stop | 460 | 29 | 208 | PickPlaceCounterToSink-s110-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 111 | False | step_budget | 900 | 65 | 584 | PickPlaceCounterToSink-s111-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 112 | False | step_budget | 900 | 64 | 533 | PickPlaceCounterToSink-s112-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 113 | False | stop | 880 | 80 | 660 | PickPlaceCounterToSink-s113-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 114 | False | stop | 880 | 55 | 460 | PickPlaceCounterToSink-s114-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 115 | False | step_budget | 900 | 85 | 650 | PickPlaceCounterToSink-s115-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 116 | False | stop | 880 | 63 | 554 | PickPlaceCounterToSink-s116-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 117 | False | step_budget | 900 | 64 | 566 | PickPlaceCounterToSink-s117-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 118 | False | step_budget | 900 | 77 | 686 | PickPlaceCounterToSink-s118-ee-short-h2 |
| h2 | ee | short | PickPlaceCounterToSink | 119 | False | step_budget | 900 | 52 | 426 | PickPlaceCounterToSink-s119-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 100 | False | stop | 880 | 75 | 617 | PickPlaceStoveToCounter-s100-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 101 | False | step_budget | 900 | 80 | 631 | PickPlaceStoveToCounter-s101-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 102 | False | step_budget | 900 | 55 | 468 | PickPlaceStoveToCounter-s102-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 103 | False | step_budget | 900 | 57 | 500 | PickPlaceStoveToCounter-s103-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 104 | False | step_budget | 900 | 93 | 704 | PickPlaceStoveToCounter-s104-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 105 | False | step_budget | 900 | 78 | 594 | PickPlaceStoveToCounter-s105-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 106 | False | step_budget | 900 | 52 | 474 | PickPlaceStoveToCounter-s106-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 107 | False | stop | 880 | 66 | 580 | PickPlaceStoveToCounter-s107-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 108 | False | step_budget | 900 | 69 | 606 | PickPlaceStoveToCounter-s108-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 109 | False | step_budget | 900 | 66 | 589 | PickPlaceStoveToCounter-s109-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 110 | False | step_budget | 900 | 67 | 544 | PickPlaceStoveToCounter-s110-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 111 | False | stop | 880 | 80 | 601 | PickPlaceStoveToCounter-s111-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 112 | False | step_budget | 900 | 63 | 570 | PickPlaceStoveToCounter-s112-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 113 | False | step_budget | 900 | 70 | 589 | PickPlaceStoveToCounter-s113-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 114 | False | step_budget | 900 | 66 | 555 | PickPlaceStoveToCounter-s114-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 115 | False | step_budget | 900 | 84 | 694 | PickPlaceStoveToCounter-s115-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 116 | False | step_budget | 900 | 63 | 512 | PickPlaceStoveToCounter-s116-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 117 | False | stop | 860 | 51 | 432 | PickPlaceStoveToCounter-s117-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 118 | False | step_budget | 900 | 92 | 671 | PickPlaceStoveToCounter-s118-ee-short-h2 |
| h2 | ee | short | PickPlaceStoveToCounter | 119 | False | step_budget | 900 | 65 | 421 | PickPlaceStoveToCounter-s119-ee-short-h2 |
