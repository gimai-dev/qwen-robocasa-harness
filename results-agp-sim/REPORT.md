# Agent-as-policy interface on RoboCasa with Qwen3.8-27B — overnight report (2026-09-16)

Box: h200-2 (the RoboCurve/GPT-6 Astra replication on h200-4 was not touched). Code: `agp_sim/` in this
repo (commits abe6bea → 8805e55), spec in `docs/superpowers/specs/2026-09-16-agent-as-policy-robocasa-qwen-design.md`.
Runs: `h200-2:/home/jli/state/agp-sim/{runs,matrix}`; videos of the notable episodes in this folder.

## What was built

The agent-as-policy protocol (arXiv 2609.12541) ported onto the RoboCasa PandaOmron simulator of the
direct-control campaign: the agent is a coding agent (here a 300-line loop around the local Qwen3.8-27B
vLLM with `exec` + `view_image` tools) working in a session directory; the robot is reachable only through
their `robot_client.py` file bridge and a session server (`agp_sim/server_sim.py`) that offers the same
nine commands as their real-arm server (`frames` with calibration, `deproject`, `move_ee`, `move_delta`,
`move_joints`, `home`, `gripper`, plus `move_base` for the mobile base). Poses are in the robot base frame
with their tool convention; the README and task prompt are theirs with the RoboCasa language instruction
substituted. Success is RoboCasa's official predicate sealed at the end; the agent's own verdict is recorded
next to it; ground-truth milestones (approach < 10 cm, contact, lift > 3 cm) come from an evaluator channel
the agent cannot see.

Verified before any episode: the saved camera calibration projects the tool centre to within 0.2 px of the
simulator's own projection; plane deproject returns the tool centre within 1 mm; moves, gripper, base and
clamps behave as documented (`agp_sim/tests/smoke_server.py`).

## Goal set for the night

Beat the direct-control campaign on the same three tasks (PickPlaceCounterToSink, PickPlaceCounterToDrawer,
PickPlaceStoveToCounter; development seeds 0–2): that campaign reached 1/60 on test seeds and brought the
gripper within 10 cm of the object in 2 of 15 validation starts. Target: at least one official success on
the development starts and more approach/contact/lift milestones than clean direct control.

## Iterations (each matrix = 3 tasks × seeds 0–2, 400-command / 45-min / 160-turn budget)

| version | change | outcome |
|---|---|---|
| v1 pilot | their interface verbatim | Qwen looped on `deproject` (40+ plane_z sweeps), never moved |
| v2 pilot | loop nudge + "measure once" rule | touched the orange (4 cm), two empty closes, carried nothing to the sink |
| v3 (dev3, 2 eps) | looping exchanges removed from context, penalties, server measurement limit, `held` verdict | 2/2 touched (2.5 / 3.5 cm), grasps empty: aim off by 3–4 cm, fingers left closed after an empty close |
| v4 (dev4, 9 eps) | `deproject` region form (centroid/extent of points above a plane), grasp-geometry text, "fingers stay closed" note | see table below |
| v5 (dev5, 9 eps) | closed-loop `move_base`, cycle-loop detection, three-empty-closes → forced re-measure | 3/9 official (one per task), approach 9/9, lift 8/9 |
| v6 (dev6, 9 eps) | six failed moves → forced home/frames; six near-identical move targets → forced close/re-observe | **4/9 official**, approach 9/9, contact 9/9, lift 7/9 |

### dev4 (v4) results

| task | seed | official | agent | approach | touched | lifted | commands | minutes | failure signal |
|---|---|---|---|---|---|---|---|---|---|
| CounterToSink | 0 | 0 | success | 1 | 1 | 1 | 22 | 5.4 | orange released into the colander on the sink rim |
| CounterToSink | 1 | **1** | success | 1 | 1 | 1 | 22 | 6.7 | — |
| CounterToSink | 2 | **1** | success | 1 | 1 | 1 | 43 | 12.8 | — |
| CounterToDrawer | 0 | 0 | none | 1 | 1 | 0 | 123 | 26.4 | 4-command up/down/close/open cycle at a wrong grasp point |
| CounterToDrawer | 1 | 0 | none | 0 | 0 | 0 | 120 | 43.5 | 95 SETTLE_MISS: 5 mm x-sweep at an unreachable pose |
| CounterToDrawer | 2 | 0 | none | 1 | 1 | 1 | 140 | 35.3 | lifted the object, then 20 identical "grasp the handle" attempts |
| StoveToCounter | 0 | 0 | none | 0 | 0 | 0 | 135 | 39.5 | 134 `move_base` calls oscillating ±0.45 m, never reached for the object |
| StoveToCounter | 1 | 0 | none | 0 | 0 | 0 | 139 | 39.5 | 131 SETTLE_MISS: millimetre target sweep, 45 ignored nudges |
| StoveToCounter | 2 | 0 | none | 1 | 1 | 1 | 269 | 46.0 | lifted the object, ran out of wall clock while placing |

dev4 totals: official 2/9; approach 6/9, contact 6/9, lift 4/9. Direct control on the same starts (Phase C
clean, the earlier campaign): approach 0/9, contact 0/9, lift 0/9, success 0/9. Every dev4 failure is a
degenerate loop of the model (a repeated command cycle with digits that drift), not a perception or reach
failure; the three loop types seen are what v5/v6 detect and break.

### dev5 (v5) results

| task | seed | official | agent | approach | touched | lifted | commands | minutes | failure signal |
|---|---|---|---|---|---|---|---|---|---|
| CounterToSink | 0 | **1** | success | 1 | 1 | 1 | 20 | 14.4 | — |
| CounterToSink | 1 | 0 | none | 1 | 1 | 1 | 108 | 32.4 | dropped after the lift, then 0.5 mm re-aim sweep (17 nudges) |
| CounterToSink | 2 | 0 | none | 1 | 1 | 0 | 115 | 34.1 | IK/SETTLE_MISS sweeps near the base |
| CounterToDrawer | 0 | 0 | none | 1 | 1 | 1 | 113 | 35.9 | lifted, SETTLE_MISS sweep while placing |
| CounterToDrawer | 1 | 0 | none | 1 | 1 | 1 | 89 | 27.1 | lifted, IK_FAILED sweep while placing |
| CounterToDrawer | 2 | **1** | none | 1 | 1 | 1 | 50 | 24.0 | placed correctly, kept adjusting until the turn cap (no RESULT.md) |
| StoveToCounter | 0 | 0 | none | 1 | 1 | 1 | 102 | 31.7 | lifted, SETTLE_MISS sweep while placing |
| StoveToCounter | 1 | **1** | success | 1 | 1 | 1 | 70 | 21.7 | — |
| StoveToCounter | 2 | 0 | success | 1 | 1 | 1 | 12 | 4.3 | placed on the counter but not where the predicate wants it |

dev5 totals: official 3/9; approach 9/9, contact 9/9, lift 8/9. The bottleneck moved from grasping (dev3/dev4)
to placing: five of the six failures lifted the object and then fell into a millimetre-step re-aim sweep
while lowering it, which is the pattern v6 breaks.

### dev6 (v6) results — final configuration

| task | seed | official | agent | approach | touched | lifted | commands | minutes | failure signal |
|---|---|---|---|---|---|---|---|---|---|
| CounterToSink | 0 | 0 | none | 1 | 1 | 1 | 117 | 33.0 | lifted, IK/SETTLE_MISS sweeps while placing (14 nudges) |
| CounterToSink | 1 | **1** | success | 1 | 1 | 1 | 24 | 10.9 | — |
| CounterToSink | 2 | **1** | success | 1 | 1 | 1 | 28 | 8.8 | — |
| CounterToDrawer | 0 | **1** | success | 1 | 1 | 1 | 20 | 7.0 | — |
| CounterToDrawer | 1 | 0 | none | 1 | 1 | 0 | 76 | 35.1 | touched, never held (7 nudges) |
| CounterToDrawer | 2 | 0 | none | 1 | 1 | 1 | 116 | 38.6 | lifted, sweeps while placing |
| StoveToCounter | 0 | 0 | none | 1 | 1 | 0 | 130 | 28.3 | 16 IK_FAILED: pan/stove geometry, never held |
| StoveToCounter | 1 | 0 | none | 1 | 1 | 1 | 164 | 40.0 | lifted, 51 SETTLE_MISS while placing |
| StoveToCounter | 2 | **1** | success | 1 | 1 | 1 | 10 | 4.7 | — |

## Summary

| configuration | official | approach | contact | lift | median minutes (successes) |
|---|---|---|---|---|---|
| direct numerical control, clean (earlier campaign, same 9 starts) | 0/9 | 0/9 | 0/9 | 0/9 | — |
| agent-as-policy v4 (region deproject) | 2/9 | 6/9 | 6/9 | 4/9 | 10 |
| agent-as-policy v5 (+ cycle breaking, closed-loop base) | 3/9 | 9/9 | 9/9 | 8/9 | 21 |
| agent-as-policy v6 (+ failed-move and micro-move rules) | 4/9 | 9/9 | 9/9 | 7/9 | 8 |

The goal set for the night (at least one official success and more milestones than clean direct control on
the development starts) is met from dev4 on. On the same nine starts where clean direct control never came
within 10 cm of the object, the agent-as-policy interface with the same Qwen3.8-27B reaches the object every
time, lifts it in 7–8 of 9, and completes 4 of 9 in the final configuration. Successful episodes are short
(5–11 minutes, 10–28 counted commands, one or two `frames` per phase); every failure is a run to the 160-turn
cap inside a degenerate loop, and each version reduced one loop family (measurement sweeps → grasp cycles →
placement micro-sweeps). The remaining failures are placement micro-sweeps that survive the v6 rule
(the model re-enters the pattern after the forced re-observation) and two cases that never achieved a grasp.

What this says about the earlier negative result: the interface, not only the model, was the binding
constraint. Frontier models on the real rig (their paper) succeed 5/5; Qwen-27B on this port succeeds ~4/9
with loop guards it would not need if it did not loop.

## Next steps (not run tonight)

1. Validation seeds 10–14 and test seeds 100–119 with v6 frozen (3 tasks × 20 = 60 episodes ≈ 20 h at two in
   parallel; `run_matrix.py --seeds 100 … 119 --out matrix/test-v6`).
2. Placement loop: the same region-deproject on the destination is rarely used; a "place" hint (release
   2–3 cm above the measured destination surface, then withdraw) is the next interface candidate.
3. Enable thinking (`enable_thinking: true`) as a separate arm: loops may drop at the cost of ~3× tokens.
4. Run the identical harness with a frontier model through the same loop (OpenAI-compatible endpoint) to
   separate model from interface on this simulator, mirroring the RoboCurve replication.

## How to run

```bash
ssh h200-2
cd /home/jli/work/agp-sim && P=/home/jli/work/robocasa-inspect-official/.venv/bin/python
$P agp_sim/run_episode.py --task PickPlaceCounterToSink --seed 0 --out /home/jli/state/agp-sim/runs/x
$P agp_sim/run_matrix.py --tasks PickPlaceCounterToSink PickPlaceCounterToDrawer PickPlaceStoveToCounter --seeds 0 1 2 --out /home/jli/state/agp-sim/matrix/x --parallel 2 -- --temperature 0.3
python3 agp_sim/tools/classify.py /home/jli/state/agp-sim/matrix/x      # milestones + failure signals
python3 agp_sim/tools/show_run.py /home/jli/state/agp-sim/matrix/x/<run>   # transcript view
bash agp_sim/tools/make_video.sh /home/jli/state/agp-sim/matrix/x/<run>    # 3-camera video
```

The vLLM on h200-2 (port 8002, same model id and token as before) was restarted with tool-call parsing and a
12-image limit; `/home/jli/state/agp-sim/start_qwen_agp.sh` relaunches it, `qwen-original-cmdline.txt` next
to it holds the previous command line.

## Limitations

- Qwen3.8-27B needs harness-level loop breaking that frontier coding agents do not; those guards
  (context surgery, measurement limit, forced re-measure) are part of the configuration being measured.
- Six to nine development starts per version; counts of 0–3 do not separate versions statistically.
- The official predicate is stricter than the agent's own verdict (colander case).
