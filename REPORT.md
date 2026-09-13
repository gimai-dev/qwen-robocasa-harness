# Direct numerical Qwen control on RoboCasa: clean baselines and independent harness screen

Status: historical campaign report, corrected after the September 12 audit. Original raw runs remain intact. The repaired campaign is stored separately under `/home/jli/state/qwen-direct/revision-2026-09/`; historical results below are not results of the repaired executor.

The repaired campaign is complete; see the [revision handoff](HANDOFF-REVISION-2026-09-12.md) and [final report](results-direct/revision-2026-09/revision-campaign-report.md).

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


What the traces show (EE-short): many rejected targets were far from the shoulder or above the base at 1.6-1.8 m height. The later audit found that the executor also rejected solvable targets, so these traces do not isolate target selection from execution; the base was blocked by furniture in the forward direction in every start and Qwen kept pushing it; when the arm did reach the object's neighbourhood it pressed on furniture (47-62 N) and the model did not interpret the force field (it read "holding the pizza cutter" with the gripper open). Full modes produce very short one-shot plans (1-4 EE actions) that stop at the first unreachable target; the 4,096-token limit was never reached (max 474 completion tokens), so truncation is not the cause.

Interface changes after Phase B, applied uniformly to every Phase C condition including its own clean control: contact force reported as norm growth over the free-hanging baseline (the vector form read a constant 214 N); partial receipts name contact blocking when the commanded path was exhausted under force; the prompt states the minimum horizontal reach; SAM prompt grid 16x16 (SAM was 60% of wall time under 3-way GPU contention).

## 4. Phase C: independent harness screen

### Prerequisite status

- H7 (successful skills): **not run.** The old builder emitted one candidate from CounterToDrawer seed 1, but it was a false grasp-and-lift match: pre/post-state alignment and object-following evidence were missing. It is not a verified skill. Rebuild using evaluator-confirmed retained contact and object/TCP lift, then validate on a second development start.
- H6 (failure experience): run (0/9), but its builder associated some failures with the wrong pre-action state and could label an unrelated later action as a verified correction. The stored bank must be rebuilt before evaluating the repaired method. H6 needs valid failure records; complete-task success is not a prerequisite.
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

State bank: `results-direct/recovery/states.json` (5 states, selected evaluator-side from Phase C snapshots; one per source episode, split by source episode). The clean runs produced no candidate because clean never came within 10 cm of an object, so states were taken from the H1, H2, H3 and H4 runs that did: 2 empty closes near the object (H1 CounterToDrawer seed 2 at 260 steps, H4 CounterToSink seed 0 at 320 steps) and 3 contacts lost without a grasp (H1, H2, H3 on CounterToSink seed 2 at 460, 220 and 160 steps). Each continuation restored the simulator state and carried the preceding action/receipt, but the audit found that all ten runs loaded incorrect historical images. Runs used the recovery budget (400 steps, 600 s, 80 decisions).

| Continuation | Task completions / candidate states | Qualified RSR | Notes |
|---|---|---|---|
| clean | 0/5 | N/A | 4 ran the full 400 steps, 1 stopped after 40 steps |
| H8 explicit recovery | 0/5 | N/A | 4 ran the full 400 steps (9-12 of 20 expectation checks were mismatches), 1 stopped at decision 1 with no motion |

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

### Combinations on the same 15 validation starts (done 2026-09-11)

Chosen from the Phase C milestones (success gave no signal): H1+H2 (both improved approach) and H3+H4 (both produced a held grasp). Matrix `/home/jli/state/qwen-direct/matrix/phaseD-combos`; tables in `results-direct/phaseD-all-results.md`.

| Condition | Success / attempts | vs components | Cost notes |
|---|---|---|---|
| H1+H2 | 0/15 | H1 0/15, H2 0/15, clean 0/15 | 72 decisions, 27 rejected, 300k prompt tokens |
| H3+H4 | 0/15 | H3 0/15, H4 0/15, clean 0/15 | 44 decisions, 0.3 rejected, 88 calls, 21k completion tokens |

Neither combination changes complete-task success; the paired differences against clean and against each component are zero. Combination milestones on the same 15 starts (`results-direct/phaseD-combos-milestones.md`): H1+H2 approached in 4, held in 3, lifted in 3, displaced in 1; H3+H4 approached in 2, held in 2, lifted in 2, displaced in 2; clean on these starts approached in 2, held in 2, lifted in 2, displaced in 1; H2 alone approached in 5 and held in 3. Neither combination is separable from its components or from clean at these counts.

### Phase E decision

No candidate reached a single complete-task success on the 15 validation starts, so no candidate qualifies by the plan's criterion. Following the rule to adjust the count and explain, the confirmatory test runs clean and one candidate: H2, the only condition whose local-progress advantage replicated from development (approach 2/9 vs 0/9) to validation (5/15 vs 2/15). H1, H3, H4 and both combinations are reported as unsupported. Code, prompts and configuration are frozen at the Phase C/D checkout (no memory conditions are involved).

## 6. Phase E: confirmatory test (frozen; done 2026-09-11)

Conditions: clean and H2 (relative actions), EE-short, 3 tasks x 20 test seeds (100-119) = 60 starts each, 120 episodes. Code, prompts and configuration frozen at the Phase C/D checkout (commit 067e5fb plus the bank/combination modules, which the tested conditions do not use). Matrix `/home/jli/state/qwen-direct/matrix/phaseE-test`; tables in `results-direct/phaseE-test-results.md`. Two clean episodes (CounterToDrawer seeds 108 and 117) ended in `infrastructure_error` (the vLLM server dropped a connection mid-request); they are labelled as such, kept in `matrix/phaseE-test-infra`, and were rerun on the same initial scenes (both reruns failed at the step budget); the table below uses the reruns, so all 120 test episodes are complete.

| Condition | CounterToSink | CounterToDrawer | StoveToCounter | All | Paired diff vs clean (n=60) |
|---|---|---|---|---|---|
| clean | 1/20 | 0/20 | 0/20 | 1/60 | - |
| H2 relative actions | 1/20 | 0/20 | 0/20 | 1/60 | 0.000, bootstrap 95% interval [0, 0], 0 wins / 0 losses |

Both successes are on the same start (CounterToSink seed 100: a glass cup; clean in 500 steps, H2 at the step budget). The paired difference is exactly zero on all 60 starts; the evidence is insufficient to distinguish H2 from clean on complete-task success, and the estimated success rate of the direct-control clean controller on unseen seeds is 1/60 (about 1.7%, exact binomial 95% interval 0.04% to 8.9%).

#### Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | ee | short | 60 | 877 | 67.7 | 23.6 | 67.7 | 259849 | 9882 | 235 | 232 | 525 |
| h2 | ee | short | 60 | 866 | 65.1 | 21.4 | 65.1 | 256333 | 9803 | 233 | 242 | 540 |


#### Test-set local-progress milestones (evaluator-side, n=60 paired starts; `results-direct/phaseE-test-milestones.md`)

| Milestone | clean | H2 | paired diff | bootstrap 95% interval | wins / losses |
|---|---|---|---|---|---|
| approach within 0.10 m | 7/60 | 15/60 | +0.133 | [+0.017, +0.250] | 11 / 3 |
| gripper touched object | 5/60 | 12/60 | +0.117 | [+0.033, +0.217] | 8 / 1 |
| held grasp | 3/60 | 5/60 | +0.033 | [-0.033, +0.100] | 3 / 1 |
| lifted >= 0.03 m | 2/60 | 4/60 | +0.033 | [0.000, +0.083] | 2 / 0 |
| object displaced >= 0.10 m | 2/60 | 6/60 | +0.067 | [0.000, +0.150] | 5 / 1 |
| official success | 1/60 | 1/60 | 0.000 | [0, 0] | 0 / 0 |

Median closest approach: clean 0.66 m, H2 0.32 m. On the frozen test set the relative action representation reliably improves the pre-grasp stages (approach and contact intervals exclude zero) but not the grasp-and-beyond stages, and not complete-task success. This is the one harness effect in the campaign that is supported by data; it is an effect on local progress, not on the task objective.

## 7. Phase F: transfer

No approach was effective on complete-task success, so there is nothing to transfer. What was measured in other interfaces: H2 was already run in joint-short on the development starts (0/9, clean 0/9) and H3 in EE-full (0/9, clean 0/9); the remaining pick-and-place tasks were not run. Unseen-seed generalization is reported by Phase E (1/60 for both tested conditions); unseen-task generalization was not measured.

## 8. Costs

All episodes actually run on h200-4 (one H200, vLLM Qwen3.8-27B BF16 shared by up to 4 concurrent episodes, SAM2.1 small per episode). Wall time is per-episode wall summed over episodes, with 3 episodes usually in parallel; the calendar time for the whole campaign was about 30 hours.

| Block | episodes | sim steps | Qwen calls | prompt tokens | completion tokens | Qwen s | SAM s | wall h |
|---|---|---|---|---|---|---|---|---|
| Phase B clean | 36 | 14760 | 916 | 3774266 | 143805 | 3546 | 6030 | 2.9 |
| Phase C EE-short screen | 72 | 58460 | 5645 | 22223799 | 1047863 | 23804 | 19447 | 13.2 |
| Phase C joint-short lane | 18 | 14400 | 782 | 3287858 | 118544 | 3256 | 3319 | 2.1 |
| Phase C EE-full lane | 18 | 600 | 26 | 81390 | 12029 | 260 | 77 | 0.1 |
| Phase C H6 | 9 | 6720 | 442 | 1844704 | 68398 | 1580 | 1304 | 0.9 |
| H8 recovery continuations | 10 | 3240 | 179 | 741618 | 32217 | 857 | 777 | 0.5 |
| Phase D singles | 75 | 63740 | 5666 | 21242485 | 934548 | 21688 | 17407 | 12.2 |
| Phase D combinations | 30 | 26100 | 2396 | 8855647 | 473747 | 10377 | 6148 | 5.2 |
| Phase E test | 120 | 104600 | 7969 | 30970905 | 1181097 | 28032 | 28475 | 17.8 |
| Phase E infra-failed attempts | 2 | 840 | 67 | 255327 | 9539 | 228 | 252 | 0.1 |
| development runs (Phase A, smoke) | 13 | 3140 | 143 | 566091 | 25221 | 588 | 389 | 0.3 |
| **total** | 403 | 296600 | 24231 | 93844090 | 4047008 | 94216 | 83625 | 55.4 |

Per-episode means are in each block's results table; a short-mode episode costs about 65 Qwen calls, 250k prompt tokens, 10k completion tokens and 9 minutes of wall time under 3-way contention (about 4 minutes alone). H3 doubles the calls and completion tokens. Full mode costs one or two calls and under 30 s.

## 9. Recommendations for further work

- The binding failure is target selection from 256x256 images plus region positions: the model rarely reaches the object (clean approach within 10 cm in 0/9 development and 2/15 validation starts). The one supported effect is the relative action representation (H2): on the 60-start test set it raises approach from 7 to 15 and contact from 5 to 12 (paired intervals exclude zero) without moving holds, lifts or success beyond noise. Kinematic previews (H3) remove rejections (0.5 per episode instead of 27) at double the call cost, with no downstream gain. Neither converts into grasps at a useful rate. A next step with a plausible mechanism is a grasp-stage representation change rather than more history: e.g. wrist-camera-anchored displacement commands, or an explicit "descend until contact" primitive that Qwen parameterizes numerically.
- The recovery continuations recorded 3/5 holds per condition and no task completions, but no state was independently qualified as recoverable to original-task completion. The audit also found incorrect historical images in all ten continuations. Replace this comparison after fixing history and qualifying states; it does not isolate recovery detection from transport or release.
- H6 can use valid failure experience even when task success is zero; rebuild its records with actual pre-action context. H7 needs verified successful skills. The existing candidate fails object-lift qualification and H7 remains untested.
- Do not compare Phase B clean numbers with Phase C-E numbers: the interface changed between them (documented in section 3).

## 10. Deliverables

1. Shared runner and commands: `direct/` package, RUNBOOK.md (verified commands, constants, artefact layout).
2. Clean trajectories and the four-condition screen: `results-direct/phaseB-clean/`, videos in each run directory (`episode.mp4`); representative success `matrix/phaseB-clean/PickPlaceCounterToSink-s0-ee-short-clean/episode.mp4`.
3. Independent harness screening table: section 4, `results-direct/phaseC/`.
4. Validation and combinations: section 5, `results-direct/phaseD-*`.
5. Frozen test results, costs, videos: sections 6 and 8, `results-direct/phaseE-test-results.md`, success videos `matrix/phaseE-test/PickPlaceCounterToSink-s100-ee-short-clean/episode.mp4` and `.../-h2/episode.mp4`.
6. H8 recovery table: section 4, `results-direct/recovery/`.
7. RUNBOOK, configurations (`config.json` and `system-prompt.txt` in every run directory), raw data under `/home/jli/state/qwen-direct/` (25 GB).

## 11. Limitations and known issues

- The home posture is a straight-arm singularity; the first descent from it uses joint-space interpolation and usually ends one slot late (`partial`).
- Base velocities below 0.25 do not move the base (installed controller dead zone); the base is blocked by furniture in most initial poses in the forward direction and, once pressed against the counter, also sideways.
- SAM region ids are not stable across steps; positions are surface centroids, 1-3 cm above/outside the geometric centre.
- Every condition was run once per start with greedy decoding; the deterministic trajectory changes with any prompt or perception change, so a single success (Phase B seed 0, Phase E seed 100) is not evidence of a stable rate.
- Wall times include GPU contention from running three episodes plus SAM servers concurrently; timing comparisons between conditions within one block are fair, between blocks approximately so.
- Milestones are computed from simulator state the policy never sees; they are diagnostic, not part of any objective.
- The no-progress rule (8 consecutive decisions without motion) ended 6 Phase B episodes early; from Phase C on, after the reach and contact notes, it rarely triggered.
- The Qwen server dropped two requests during Phase E (infrastructure errors, rerun on the same scenes).


## 12. Show-Harness-style interface with zero-shot Qwen-27B (2026-09-12)

Plan: `docs/superpowers/plans/2026-09-11-show-harness-style-qwen27b.md`. Two lanes: their code with our model, and their interface inside our loop. Both use the frozen `qwen3.8-27b-bf16`, temperature 0, thinking off, no demonstrations.

### Lane A: Show-Harness on its own ManiSkill benchmark, our model

Setup (RUNBOOK "Lane A"): Show-Harness at 137d571, ManiSkill 3.0.1, stock `PickCube-v1` with `panda_wristcam` at 256 px (their default BlockPAP-v1 real2sim rig needs an RLinf module the public repo does not ship). Their runner unchanged except that the controller uses their zero-shot `complete_token` call (vLLM `guided_choice` over the nine tokens) instead of the bare-token call meant for fine-tuned adapters. 60 decisions per episode, 30 episodes over consecutive seeds, `auto_release` plugin on (their default). Raw logs: `results-direct/laneA/`, rollouts under `/home/jli/state/show-harness/rollouts/MS-PickCube-v1/`.

| Prompt | Success | Token histogram over 1,742 decisions |
|---|---|---|
| their `v3` lite prompt (task + recent moves) | 1/30 | GRASP 1699, MV_DOWN 39, MV_LEFT 2, MV_RIGHT 2 |
| `v3q`: same plus one `Gripper: {gripper_state}` line and "RELEASE after an empty close" (the proprioception plugin's content) | 1/30 | GRASP 1493, MV_DOWN 203, MV_UP 28, MV_RIGHT 10, MV_LEFT 8, RELEASE 0 |

The one success in each batch is the same seed (8): the cube happened to sit under the open gripper, and MV_DOWN then GRASP finished the task in 2 steps. In every other episode the model answers GRASP on almost every step, the empty gripper is reopened by auto-release, and it grasps again; it never emits RELEASE, MV_FWD or MV_BACK. The scene is trivially legible (a red cube centred on a wooden table, visible in both views). On the same interface the paper reports 86-96% for Gemini 3.1 Pro / GPT-5.6 / Opus 5 and 86% for a 2B Qwen fine-tuned on a few GPU-hours of demonstrations. The zero-shot 27B model does not use the interface at all: it does not steer with the wrist view before grasping.

Caveats: their ManiSkill runner carries only the `auto_release` plugin (subtask planning, recovery, multi-view guidance live in the real-robot/RoboLab runners); PickCube is not their headline BlockPAP rig; 30 episodes per condition.

### Lane B: their interface inside our RoboCasa loop (development screen, 36 episodes)

Setup (RUNBOOK "Lane B"): token vocabulary + deterministic interpreter (2 cm / 4 cm base-frame steps, 15 deg rotations, gripper, base), one guided-choice token per decision, RGB only (agent view + wrist), proprioception text, recent moves, ready pose before decision 1; the same nine development starts and budgets as the numerical campaign. Matrix `/home/jli/state/qwen-direct/matrix/sem-dev`; tables in `results-direct/sem/`.

| Method | Success | Token use (9 episodes) |
|---|---|---|
| `sem` (base interface) | 0/9 | MV_DOWN 608, MV_LEFT 179, MV_UP 164, GRASP 44, ROTATE 23, RELEASE 15 |
| `sem+plan` (+ subtask planner with completion checks) | 0/9 | MV_DOWN 439, MV_LEFT 381, MV_RIGHT 242, MV_FWD 82, GRASP 5 |
| `sem+plan+rec` (+ empty-close recovery) | 0/9 | as above; 4 recoveries fired |
| `sem-full` (+ agent-view selection) | 0/9 | as above; 3 recoveries |

Local-progress milestones (evaluator-side), with the numerical clean from Phase C on the same nine starts as reference:

| method | n | approach<=0.10m | contact | hold | lift>=3cm | displaced>=10cm | success | median closest dist m |
|---|---|---|---|---|---|---|---|---|
| sem | 9 | 4 | 2 | 1 | 0 | 1 | 0 | 0.116 |
| sem+plan | 9 | 4 | 3 | 1 | 0 | 2 | 0 | 0.183 |
| sem+plan+rec | 9 | 4 | 3 | 2 | 2 | 2 | 0 | 0.173 |
| sem-full | 9 | 4 | 4 | 0 | 0 | 0 | 0 | 0.198 |


| numerical clean (Phase C, same starts) | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0.647 |

Reading: unlike on ManiSkill, the 27B model uses this interface in our loop: varied moves, few GRASP tokens, median closest approach 0.12-0.20 m against 0.65 m for the numerical clean on the same starts, 4 of 9 approaches within 10 cm for every variant (numerical clean 0 of 9), and `sem+plan+rec` held and lifted the object twice. No variant completed a task, and the plan's bar for advancing (3 of 9 holds or a success) was not met; because the interface effect on approach is the largest local-progress effect measured in the campaign and a test-set run costs under an hour, `sem` and `sem+plan+rec` were nevertheless run on the 60 frozen test starts for a paired milestone comparison with Phase E clean and H2 (results below).

#### Costs (means per episode)

| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sem | semantic | short | 9 | 806 | 116.4 | 7.0 | 116.4 | 80789 | 353 | 26 | 0 | 80 |
| sem+plan | semantic | short | 9 | 819 | 134.7 | 13.8 | 135.7 | 99069 | 3382 | 99 | 0 | 158 |
| sem+plan+rec | semantic | short | 9 | 806 | 132.0 | 13.0 | 132.6 | 96483 | 3302 | 97 | 0 | 155 |
| sem-full | semantic | short | 9 | 809 | 137.2 | 15.1 | 137.9 | 101038 | 3444 | 84 | 0 | 142 |



### Lane B: frozen test set (sem and sem+plan+rec, 60 starts each; done 2026-09-12)

Matrix `/home/jli/state/qwen-direct/matrix/sem-test`; tables `results-direct/sem/sem-test-results.md`, milestones `results-direct/sem/sem-test-milestones.md`. Same 60 starts as Phase E, so the comparison with numerical clean and H2 is paired.

| Condition | Success | approach<=0.10 m | contact | hold | lift | displaced | median closest m | decisions | calls | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| numerical clean (Phase E) | 1/60 | 7 | 5 | 3 | 2 | 2 | 0.662 | 68 | 68 | 525 |
| numerical H2 relative (Phase E) | 1/60 | 15 | 12 | 5 | 4 | 6 | 0.323 | 65 | 65 | 540 |
| `sem` (Show-Harness-style tokens) | 0/60 | 14 | 14 | 4 | 1 | 3 | 0.221 | 126 | 126 | 78 |
| `sem+plan+rec` (+ subtask plan + recovery) | 0/60 | 14 | 10 | 3 | 3 | 6 | 0.260 | 136 | 137 | 135 |

Paired bootstrap differences (n=60, 95% intervals):
- `sem` vs clean: approach +0.117 [0.000, +0.233], contact +0.150 [+0.033, +0.283]; hold, lift, displaced within noise.
- `sem+plan+rec` vs clean: approach +0.117 [0.000, +0.233]; contact +0.083 [-0.033, +0.200]; the rest within noise.
- `sem` vs H2 and `sem+plan+rec` vs H2: every milestone difference is within noise (e.g. approach -0.017 [-0.167, +0.133]); success is 0/60 vs 1/60.

Reading: these semantic controller packages improved approach relative to the original absolute numerical controller and did not demonstrate improved complete-task success. Their difference from H2 remains uncertain, not an equivalence result. The packages are cheaper per episode (78-135 s, 0.4-3.4k completion tokens), but also change posture, remove SAM, reduce image inputs, and change action timing; cost and approach differences cannot be attributed to guided-token output alone.

### What the two lanes say together

- Lane A: the tested ManiSkill configuration gets 1/30 with a repeated GRASP loop. This is a negative result for that configuration; the comparison does not isolate model capability from benchmark and implementation differences.
- Lane B: the semantic packages approach in 14 of 60 starts and never complete a task. Their differences from H2 remain inconclusive, and the missing common-posture control prevents action-representation attribution.
- Conclusion: these tested configurations did not demonstrate improved complete-task success. H2 improved approach and contact. The limiting cause is not isolated: the audit found execution and harness defects, H7 was untested, and the recovery comparison needs replacement. Additional harness designs remain an open empirical question.
