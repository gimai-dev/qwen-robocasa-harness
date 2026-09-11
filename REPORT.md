# Direct numerical Qwen control on RoboCasa: clean baselines and independent harness screen

Status: living report, updated per phase. Numbers are machine-written from `result.json` / `summary.json` files listed in each section.

## 1. Setting (fixed)

RoboCasa 1.0.1 / robosuite 1.5.2 / MuJoCo 3.3.1, PandaOmron (7-DoF arm, parallel-jaw gripper, mobile base, torso held at zero), frozen Qwen3.8-27B BF16 (`qwen3.8-27b-bf16`, vLLM, loopback on h200-4), temperature 0 / top_p 1 / thinking off / strict JSON schema, SAM2.1 Hiera Small on the three official 256x256 RGB views with the official camera calibration. Policy-visible information: public joint/TCP/gripper/base state, wrist wrench relative to baseline, execution receipts, SAM regions with multi-view triangulated positions, current three views and the three views from before the previous action. Budget 900 simulator steps / 1200 s / 180 decisions; control slot 20 steps. Official success is the RoboCasa task predicate evaluated by the simulator at `stop` or budget end. Full protocol constants: RUNBOOK.md.

Shared adjustments relative to the released harness (recorded, applied to every condition): (1) the vLLM server was restarted with `--limit-mm-per-prompt {"image":8} --max-num-seqs 4` (same model snapshot, new attestation) because the clean window needs six images; (2) the Qwen client accepts N labelled images; (3) policy-facing orientation is the FK grip-site frame (the public eef quaternion is that frame rotated 90 degrees about z); (4) EE targets are tracked as straight lines at 0.01 m/step, with a direct-IK joint-space fallback when the straight line has no local solution (the straight-arm home posture).

## 2. Phase A: shared loop (done 2026-09-10)

Implemented: `direct/` package (kinematics, actions, sim_child, executor, perception + sam_server, observation, policy, methods, harnesses, episode, matrix, banks, recovery). Interface checks and their results are in RUNBOOK.md ("Phase A record"). One complete EE-short clean episode on PickPlaceCounterToSink seed 0 ("Pick the orange from the counter and place it in the sink."):

| Field | Value |
|---|---|
| Official success | false (termination `stop` by the model at 880 steps) |
| Decisions / slots / rejected | 60 / 44 / 15 (all rejections: IK-unreachable EE targets 0.74 m from the shoulder) |
| Qwen calls / prompt tokens / completion tokens | 60 / 234,399 / 7,883 |
| Wall time | 245 s (Qwen 141 s, SAM 52 s, simulation + IK the rest) |
| Video | `/home/jli/state/qwen-direct/runs/phaseA-ee-short-s0/episode.mp4` (220 frames, 22 s) |

What happened: the model's first EE target (0.1 m above the orange, identity orientation) was unreachable; it then drove the base forward 40 times although the base was blocked by the counter after the first push (base x stayed at 4.957 m), never moving laterally toward the orange. Interface changes made after this run (before freezing Phase B): receipts now carry `tcp_moved_m` / `base_moved_m` deltas with a blocked-base note, unreachable receipts report the target's distance from the shoulder, and the prompt states the measured base displacement per slot (0.5 -> 0.16 m in free space, dead zone below 0.25).

Development episodes used for integration before the freeze: 3 (EE-short, joint-short, EE-full on CounterToSink seed 0) plus 6 four-decision harness smoke runs. None counts toward Phase B.

## 3. Phase B: four clean baselines (done 2026-09-10)

Code frozen at commit 7275128; matrix `/home/jli/state/qwen-direct/matrix/phaseB-clean` (36 episodes, 3 tasks x development seeds 0-2 x {EE, joint} x {short, full}, 3 episodes in parallel on h200-4). Machine-generated tables: `results-direct/phaseB-clean/phaseB-clean-results.md`.

| Condition | Success / attempts | Notes |
|---|---|---|
| EE-short | 1/9 | CounterToSink seed 0 succeeded (48 decisions, 760 steps); 3 runs ended by the no-progress rule (8 consecutive rejected targets), 5 ran to a model `stop` or the step budget |
| Joint-short | 0/9 | no rejected actions (joint targets are always executable) but no grasp; 6 model stops, 3 step-budget ends |
| EE-full | 0/9 | one-shot sequences of 1 to 4 actions (mean 31 simulator steps); 6 of 9 ended at an unreachable slot, 3 completed their short sequence |
| Joint-full | 0/9 | one-shot sequences of 1 to 14 joint actions, all executed, none reaching the object |

### Success per condition and task

| method | interface | mode | task | success/attempts | terminations |
|---|---|---|---|---|---|
| clean | ee | full | PickPlaceCounterToDrawer | 0/3 | {'sequence_complete': 1, 'unreachable_at_slot_1': 1, 'unreachable_at_slot_4': 1} |
| clean | ee | full | PickPlaceCounterToSink | 0/3 | {'sequence_complete': 1, 'unreachable_at_slot_3': 1, 'unreachable_at_slot_2': 1} |
| clean | ee | full | PickPlaceStoveToCounter | 0/3 | {'unreachable_at_slot_3': 2, 'sequence_complete': 1} |
| clean | ee | full | **all** | 0/9 | |
| clean | ee | short | PickPlaceCounterToDrawer | 0/3 | {'no_progress': 2, 'step_budget': 1} |
| clean | ee | short | PickPlaceCounterToSink | 1/3 | {'stop': 2, 'no_progress': 1} |
| clean | ee | short | PickPlaceStoveToCounter | 0/3 | {'no_progress': 1, 'stop': 2} |
| clean | ee | short | **all** | 1/9 | |
| clean | joint | full | PickPlaceCounterToDrawer | 0/3 | {'sequence_complete': 3} |
| clean | joint | full | PickPlaceCounterToSink | 0/3 | {'sequence_complete': 3} |
| clean | joint | full | PickPlaceStoveToCounter | 0/3 | {'sequence_complete': 3} |
| clean | joint | full | **all** | 0/9 | |
| clean | joint | short | PickPlaceCounterToDrawer | 0/3 | {'stop': 2, 'step_budget': 1} |
| clean | joint | short | PickPlaceCounterToSink | 0/3 | {'stop': 2, 'step_budget': 1} |
| clean | joint | short | PickPlaceStoveToCounter | 0/3 | {'stop': 2, 'step_budget': 1} |
| clean | joint | short | **all** | 0/9 | |


### Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | full | 9 | 31 | 1.0 | 0.7 | 1.0 | 3611 | 467 | 10 | 6 | 20 |
| clean | ee | short | 9 | 576 | 53.7 | 24.4 | 53.7 | 213994 | 7950 | 206 | 372 | 620 |
| clean | joint | full | 9 | 149 | 1.0 | 0.0 | 1.0 | 3739 | 474 | 9 | 3 | 22 |
| clean | joint | short | 9 | 884 | 46.1 | 0.0 | 46.1 | 198019 | 7087 | 169 | 288 | 515 |


What the traces show (EE-short): the dominant failure is target selection, not execution. Rejected targets were either beyond reach (0.7 m or more from the shoulder) or directly above the base at 1.6-1.8 m height; the base was blocked by furniture in the forward direction in every start and Qwen kept pushing it; when the arm did reach the object's neighbourhood it pressed on furniture (47-62 N) and the model did not interpret the force field (it read "holding the pizza cutter" with the gripper open). Full modes produce very short one-shot plans (1-4 EE actions) that stop at the first unreachable target; the 4,096-token limit was never reached (max 474 completion tokens), so truncation is not the cause.

Interface changes after Phase B, applied uniformly to every Phase C condition including its own clean control: contact force reported as norm growth over the free-hanging baseline (the vector form read a constant 214 N); partial receipts name contact blocking when the commanded path was exhausted under force; the prompt states the minimum horizontal reach; SAM prompt grid 16x16 (SAM was 60% of wall time under 3-way GPU contention).

## 4. Phase C: independent harness screen

### Prerequisite status

- H7 (successful skills): **not run, prerequisite missing.** The bank builder found exactly one local grasp-and-lift fragment in the nine clean EE-short development runs (CounterToDrawer seed 1, wooden spoon, grasp width 0.010 m, no complete-task success) and no second development start on which to validate it. A skill that was never validated on another initial state would be a single-episode recipe, which the isolation rule for H7 excludes.
- H6 (failure experience): run (0/9) with the bank built from the nine Phase C clean EE-short runs (`/home/jli/state/qwen-direct/banks/phaseC/h6-failures.json`); records are unreachable targets, blocked base motions and empty closes with a correction marked verified only when a later action fixed the same failure type.
- H8 recovery-state evaluation: pending selection of recoverable failure states from the evaluator-side inspection of the Phase C clean runs.

### EE-short screen (72 episodes, done 2026-09-11)

Code frozen at commit 067e5fb; matrix `/home/jli/state/qwen-direct/matrix/phaseC-ee-short`; tables in `results-direct/phaseC/`. Same nine development starts for every condition (3 tasks x seeds 0-2), clean rerun on the same code as its control.

| Condition | Success / attempts | Paired diff vs clean (n=9) |
|---|---|---|
| clean | 0/9 | - |
| H1 visual markers | 0/9 | 0, interval [0, 0] |
| H2 relative actions | 0/9 | 0 |
| H3 propose and preview | 0/9 | 0 |
| H4 execution timing (5/10/20) | 0/9 | 0 |
| H4c fixed 5-step control | 0/9 | 0 |
| H5 working memory | 0/9 | 0 |
| H8 explicit recovery | 0/9 | 0 |
| H6 failure experience | 0/9 | bank of 569 clean failure records (17 with a verified correction), 2 retrieved per call under a 700-token cap; one episode stopped at decision 1 with no motion |
| H7 successful skills | not run (prerequisite missing, see above) | |

Complete-task success does not separate any condition from clean at this scale: every paired difference is exactly zero. Local progress, measured evaluator-side from the simulator snapshots the policy never sees, does separate them:

| method | n | approach<=0.10m | contact | hold | lift>=3cm | displaced>=10cm | success | median closest dist m |
|---|---|---|---|---|---|---|---|---|
| clean | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0.647 |
| h1 | 9 | 2 | 2 | 0 | 0 | 1 | 0 | 0.410 |
| h2 | 9 | 2 | 2 | 0 | 0 | 1 | 0 | 0.335 |
| h3 | 9 | 1 | 2 | 1 | 0 | 1 | 0 | 0.620 |
| h4 | 9 | 1 | 1 | 1 | 1 | 1 | 0 | 0.611 |
| h4c | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0.656 |
| h5 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0.649 |
| h8 | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 0.659 |


Reading: clean never brought the gripper within 10 cm of the target object in any of its nine runs (median closest approach 0.65 m); H1 and H2 reached and touched the object in two runs each, H3 and H4 achieved a held grasp once each, and H4 (CounterToSink seed 0) lifted the orange 17.5 cm, carried it into the sink and released it, then stopped without withdrawing 0.25 m, failing the official predicate. H4c, H5 and H8 show no local progress. With n=9 and counts of 0-2, none of this is statistically separable; it identifies H1, H2, H3, H4 as the candidates with a mechanism worth validating and H4c, H5, H8 as unsupported.

Note on Phase B vs Phase C clean: the Phase B clean run that succeeded (CounterToSink seed 0) did not repeat under the Phase C code; the Phase C clean runs also approached objects less often than Phase B's traces suggest. The interface changes between the phases (norm-growth force, minimum-reach text, contact-blocking note, SAM 16-point grid) changed the deterministic trajectories; whether they helped or hurt clean is not resolved by these data, but every Phase C comparison is internally consistent.

#### Costs (means per episode)

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



### Additional lanes (counted separately, same nine starts, same code)

| Lane | Condition | Success | Notes |
|---|---|---|---|
| Joint-short | clean | 0/9 | no rejected actions; mean 43 decisions, 829 steps, 427 s |
| Joint-short | H2 delta-q | 0/9 | paired diff 0; mean 44 decisions, 771 steps |
| EE-full | clean | 0/9 | one-shot plans of 1 to 10 actions (143-724 completion tokens); 3 ended at an unreachable slot, 6 completed their short plan |
| EE-full | H3 candidates + preview | 0/9 | 2 counted calls per episode; chosen plans of 1-2 actions (mean 18 steps); one call hit the 4,096-token limit (`truncated_output`, no motion executed) |

Joint-short milestones (evaluator-side, `results-direct/phaseC/phaseC-joint-short-milestones.md`): clean 0/9 approaches within 10 cm (median closest 0.50 m); H2 delta-q 2/9 approaches, 2 contacts, no hold (median closest 0.20 m). The relative representation helps the model get near the object in both interfaces but does not produce a grasp.

The 4,096-token bound is not what limits full mode: the longest clean plan (10 actions) used 724 tokens. Qwen writes short one-shot plans and stops.

### H8 separate recovery evaluation (done 2026-09-11)

State bank: `results-direct/recovery/states.json` (5 states, selected evaluator-side from Phase C snapshots; one per source episode, split by source episode). The clean runs produced no candidate because clean never came within 10 cm of an object, so states were taken from the H1, H2, H3 and H4 runs that did: 2 empty closes near the object (H1 CounterToDrawer seed 2 at 260 steps, H4 CounterToSink seed 0 at 320 steps) and 3 contacts lost without a grasp (H1, H2, H3 on CounterToSink seed 2 at 460, 220 and 160 steps). Each continuation restores the exact simulator state, carries the preceding action/receipt and pre-action images, and runs with the recovery budget (400 steps, 600 s, 80 decisions).

| Continuation | Recovered / states | RSR | Notes |
|---|---|---|---|
| clean | 0/5 | 0.00 | 4 ran the full 400 steps, 1 stopped after 40 steps |
| H8 explicit recovery | 0/5 | 0.00 | 4 ran the full 400 steps (9-12 of 20 expectation checks were mismatches), 1 stopped at decision 1 with no motion |

Recoverability of these five states is **unconfirmed**: the plan requires at least one independent continuation to complete the task within the budget, and none did. The table is therefore a paired comparison on states of unknown recoverability, not a recovery success rate; no L1-L4 breakdown is reported. Evaluator-side milestones of the continuations (`results-direct/recovery/phaseC-states-milestones.md`) show the states are physically recoverable at least to a grasp: from the same 5 states, clean re-approached in 4, re-grasped (held) in 3, lifted in 2 and displaced the object by more than 10 cm in 3; H8 re-approached in 4, held in 3, lifted in 2 and displaced in 0. Neither completed the task within the 400-step recovery budget. Raw data: `/home/jli/state/qwen-direct/recovery/phaseC-states/`.

## 5. Phase D: validation and combinations

### Singles on the 15 validation starts (done 2026-09-11)

Matrix `/home/jli/state/qwen-direct/matrix/phaseD-singles` (3 tasks x seeds 10-14, EE-short, same code as Phase C); tables in `results-direct/phaseD-singles-results.md`. Candidates were the four conditions with local progress in Phase C.

| Condition | Success / attempts | Paired diff vs clean (n=15) |
|---|---|---|
| clean | 0/15 | - |
| H1 visual markers | 0/15 | 0, interval [0, 0] |
| H2 relative actions | 0/15 | 0 |
| H3 propose and preview | 0/15 | 0 |
| H4 execution timing | 0/15 | 0 |

No condition reaches any complete-task success on unseen development seeds. H3 removes almost all rejected actions (0.5 per episode vs 27.5 for clean, because it previews reachability before committing) at the price of two calls per decision (91 calls, 19k completion tokens, 409 s of Qwen time per episode). H1 makes more decisions and more rejections than clean (81 and 38 vs 69 and 28).

#### Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | 15 | 815 | 68.5 | 27.5 | 68.5 | 260516 | 10121 | 257 | 275 | 591 |
| h1 | ee | short | 15 | 856 | 81.4 | 38.4 | 81.4 | 332331 | 11918 | 286 | 245 | 598 |
| h2 | ee | short | 15 | 819 | 68.1 | 26.9 | 68.1 | 264860 | 10473 | 246 | 222 | 532 |
| h3 | ee | short | 15 | 899 | 45.5 | 0.5 | 91.1 | 289518 | 19346 | 409 | 183 | 664 |
| h4 | ee | short | 15 | 861 | 68.6 | 24.9 | 68.6 | 268941 | 10445 | 247 | 235 | 543 |


#### Local-progress milestones on validation (evaluator-side; `results-direct/phaseD-singles-milestones.md`)

| method | n | approach<=0.10m | contact | hold | lift>=3cm | displaced>=10cm | success | median closest dist m |
|---|---|---|---|---|---|---|---|---|
| clean | 15 | 2 | 2 | 2 | 2 | 1 | 0 | 0.664 |
| h1 | 15 | 1 | 1 | 1 | 1 | 1 | 0 | 0.694 |
| h2 | 15 | 5 | 5 | 3 | 1 | 1 | 0 | 0.453 |
| h3 | 15 | 2 | 2 | 1 | 1 | 1 | 0 | 0.624 |
| h4 | 15 | 1 | 1 | 0 | 0 | 0 | 0 | 0.599 |


On validation the only local-progress signal that replicates from development is H2 (relative actions): it brings the gripper within 10 cm of the object in 5 of 15 starts (clean 2 of 15; in Phase C 2 of 9 vs 0 of 9) and holds it in 3 (clean 2). H1, H3 and H4 do not exceed clean on any milestone on validation; H4's single Phase C lift did not recur. Clean itself lifted an object twice on validation (CounterToDrawer and StoveToCounter seed 13) and carried it once, all without completing the task. Counts of 0-5 out of 15 do not separate any condition statistically.

(combinations pending)

## 6. Costs

(pending)

## 7. Limitations and known issues

- The home posture is a straight-arm singularity; the first descent from it uses joint-space interpolation and usually ends one slot late (`partial`).
- Base velocities below 0.25 do not move the base (installed controller dead zone); the base is blocked by furniture in most initial poses in the forward direction and, once pressed against the counter, also sideways.
- SAM region ids are not stable across steps; positions are surface centroids, 1-3 cm above/outside the geometric centre.
