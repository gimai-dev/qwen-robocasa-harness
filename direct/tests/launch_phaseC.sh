#!/bin/bash
# Phase C independent harness screen on the 9 development starts (3 tasks x seeds 0-2).
# Run from the FROZEN checkout after copying the dev checkout into it.
set -e
cd /home/jli/work/qwen-direct-control
export PYTHONPATH=/home/jli/work/qwen-direct-control:/home/jli/work/qwen-direct-control/runtime
PY=/home/jli/work/robocasa-inspect-official/.venv/bin/python
OUT=/home/jli/state/qwen-direct/matrix
TASKS="PickPlaceCounterToSink PickPlaceCounterToDrawer PickPlaceStoveToCounter"
# main screen: EE-short, clean + H1 H2 H3 H4 H4c H5 H8 (72 episodes)
$PY -m direct.matrix --tasks $TASKS --seeds 0 1 2 --interfaces ee --modes short --methods clean h1 h2 h4 h4c h5 h3 h8 --out $OUT/phaseC-ee-short --parallel 3
# additional lanes counted separately: H2 in joint-short (+ clean), H3 transferred to full (+ clean)
$PY -m direct.matrix --tasks $TASKS --seeds 0 1 2 --interfaces joint --modes short --methods clean h2 --out $OUT/phaseC-joint-short --parallel 3
$PY -m direct.matrix --tasks $TASKS --seeds 0 1 2 --interfaces ee --modes full --methods clean h3 --out $OUT/phaseC-ee-full --parallel 3
echo PHASE_C_DONE
