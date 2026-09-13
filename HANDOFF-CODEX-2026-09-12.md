# Handoff for Codex: direct numerical Qwen control on RoboCasa (2026-09-10 to 2026-09-12)

Historical handoff. The subsequent repairs and completed selective reruns are documented in [the revision handoff](HANDOFF-REVISION-2026-09-12.md), which is the current entry point.

Read in this order: this file, `RUNBOOK.md` (commands, constants, artefact layout), `REPORT.md` (all results, sections 1-12), `docs/superpowers/plans/2026-09-11-show-harness-style-qwen27b.md` (the follow-up plan). Everything below is machine-checked against `result.json` files; nothing is estimated.

## 1. Goal and setting

Original plan: `/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/claude-execution-plan-en.md`. Reuse the released `qwen-robocasa-harness` environment (RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1, PandaOmron mobile manipulator, SAM2.1 small, frozen Qwen3.8-27B BF16 on vLLM), replace its task-specific recipes with a controller where **frozen Qwen directly emits numerical robot targets**, establish four clean baselines, then test eight harness mechanisms H1-H8 independently against clean. Zero-shot only. Official RoboCasa task predicate is the only success metric.

Fixed decoding: seed 3074294, temperature 0, top_p 1, thinking off, strict JSON schema. Budget per episode: 900 sim steps, 1200 s, 180 decisions; control slot 20 sim steps. Tasks: PickPlaceCounterToSink, PickPlaceCounterToDrawer, PickPlaceStoveToCounter. Seeds: dev 0-2, validation 10-14, test 100-119.

## 2. What was built (all committed on branch `direct-control`, 45 commits)

Repo: `~/Desktop/first try/qwen-direct-control` (clone of the published harness, history preserved). Box mirror: `h200-4:/home/jli/work/qwen-direct-control` (frozen, used by matrices) and `.../qwen-direct-control-dev` (development). Package `direct/`:

| Module | Responsibility |
|---|---|
| `kinematics.py` | Panda FK (verified 1.8e-8 m vs public state), bounded IK with regularization toward the current q, straight-line segments (1 cm / 2.5 deg), multi-start fallback |
| `actions.py` | Compact action protocol `k/p/o/q/g/a/v/n`, strict JSON schemas (short and full), relative decoding for H2 |
| `sim_child.py` | Sandboxed RoboCasa child (sudo/unshare/setpriv), pinned scenes, fixed-length slots with joint tracking (0.08 rad/step cap, lag pause 0.10), base slots, per-observation snapshots, official predicate, video frames |
| `executor.py` | Launches the child, mailbox protocol, action -> command (IK in EE mode), receipts with deltas, blocked-base / contact-blocked / unreachable reasons, `move_to_ready` |
| `perception.py` + `sam_server.py` | Persistent CUDA SAM2 server, multi-view centroid triangulation into region records with world positions |
| `policy.py` | N-image Qwen client reusing the released identity/attestation checks; per-category call accounting |
| `observation.py` | Clean window: current 3 views + pre-previous-action 3 views + state + regions + previous action/receipt |
| `episode.py`, `matrix.py`, `analyze.py`, `milestones.py`, `banks.py`, `recovery.py` | Episode CLI, matrix runner (resumable), tables + paired bootstrap, evaluator-side local-progress milestones, H6/H7 bank builder, H8 recovery-state tooling |
| `harnesses.py`, `harnesses_extra.py`, `combos.py`, `methods.py` | H1 visual markers, H2 relative actions, H3 propose-and-preview (2 calls), H4 execution timing + H4c fixed-5 control, H5 working memory, H6 failure bank, H7 skill bank, H8 explicit recovery, `h1+h2` / `h3+h4` combinations |
| `semantic.py`, `semantic_policy.py`, `semantic_plugins.py`, `semantic_episode.py` | Show-Harness-style follow-up: 15-token vocabulary, interpreter (2/4 cm base-frame steps, 15 deg rotations, base tokens), guided-choice single-token call, subtask planner + empty-close recovery + agent-view selection |
| `direct/tests/` | pure tests (`test_semantic.py`, 9 pass), on-box smoke/probe scripts |

Shared adjustments recorded in REPORT section 1 and 3: vLLM restarted with `--limit-mm-per-prompt {"image":8} --max-num-seqs 4` (`/home/jli/state/panda-qwen38/start_direct_qwen.py`); FK grip-site orientation frame (public eef quat is that frame rotated 90 deg about z); joint-space IK fallback at the straight-arm home singularity; contact force as norm growth over baseline; SAM 16x16 grid from Phase C on; minimum-reach and contact-blocking notes in prompt/receipts.

## 3. What was run and what came out

Corrected count (Codex audit): 563 RoboCasa attempt records on h200-4 (403 numerical and 160 semantic, including smoke runs and retained infrastructure failures). The RoboCasa records total 60.3 summed wall hours and 108.5M prompt tokens. Lane A's 60 ManiSkill episodes are reported separately. Source aggregation: `results-direct/revision-2026-09/historical-summary.json`. Raw data: `/home/jli/state/qwen-direct/` and `/home/jli/state/show-harness/`.

| Block | Episodes | Result |
|---|---|---|
| Phase A shared loop | smoke + 1 | interface checks pass (5 cm EE move = 4.98 cm in 20 steps; gripper 0.079 -> 0.002 -> 0.078 m; base 0.16 m per slot at v=0.5, dead zone below 0.25); first episode fails (base blocked by counter, target 0.74 m from shoulder) |
| Phase B clean baselines (dev seeds 0-2) | 36 | EE-short 1/9, joint-short 0/9, EE-full 0/9, joint-full 0/9 |
| Phase C screen (same 9 starts, EE-short) | 72 + 9 (H6) | clean, H1, H2, H3, H4, H4c, H5, H6, H8 all 0/9; H7 not run (only one unvalidated skill in the bank); paired diffs exactly 0 |
| Phase C extra lanes | 18 + 18 | joint-short clean 0/9, H2 0/9; EE-full clean 0/9, H3 0/9 (plans of 1-14 actions, token limit never binding) |
| H8 recovery evaluation | 10 | 5 real failure states (from H1-H4 runs; clean never reached an object); clean 0/5, H8 0/5; states recoverable to a grasp (3/5 each) but not to task completion |
| Phase D validation (seeds 10-14) | 75 + 30 | clean, H1, H2, H3, H4, H1+H2, H3+H4 all 0/15 |
| Phase E frozen test (seeds 100-119) | 120 | clean 1/60, H2 1/60 (same start, CounterToSink s100), paired diff 0; 2 infra errors (vLLM disconnect) rerun on the same scenes |
| Lane A: Show-Harness on ManiSkill PickCube-v1, our Qwen | 30 + 30 | 1/30 with their prompt, 1/30 with a gripper-state line added; GRASP on 86-98% of tokens (degenerate loop) on a trivially legible scene |
| Lane B: Show-Harness interface in our loop | 36 dev + 120 test | dev: sem / sem+plan / sem+plan+rec / sem-full all 0/9; test: sem 0/60, sem+plan+rec 0/60 |

Evaluator-side milestones (simulator truth the policy never sees) are the only thing that separates conditions:

| Test set, n=60 paired starts | success | approach<=10 cm | contact | hold | lift | median closest |
|---|---|---|---|---|---|---|
| numerical clean | 1 | 7 | 5 | 3 | 2 | 0.66 m |
| numerical H2 (relative displacement) | 1 | 15 | 12 | 5 | 4 | 0.32 m |
| semantic tokens (sem) | 0 | 14 | 14 | 4 | 1 | 0.22 m |
| semantic + plan + recovery | 0 | 14 | 10 | 3 | 3 | 0.26 m |

H2 vs clean: approach +0.133 [+0.017, +0.250], contact +0.117 [+0.033, +0.217] (paired bootstrap 95%); holds/lifts/success within noise. Semantic vs H2: every milestone within noise.

## 4. Why it fails (REPORT section 12 and memory `qwen-direct-control-failure-diagnosis`)

Per-decision diagnosis over 7,969 Phase E decisions:
- Perception: the SAM+triangulation region list has a region within 5 cm of the true object in only 49% of decisions (median nearest-region error 6 cm, p75 34 cm).
- Selection: when the region exists, Qwen's target is within 3 cm horizontally 54% of the time (median 2.6 cm); when it is absent, 1% (median 42 cm). The model uses correct numbers; it cannot estimate 3D from 256 px images.
- Grasp: grasp-intent targets sit a median 12 cm above the object; it never descends to grasp height.
- Execution: 34% of decisions IK-rejected (55% of those in the 0.3-0.6 m band that should be reachable); base blocked by furniture in every start.
- Lane A shows the same model cannot steer with a wrist camera even on a red cube on an empty table, where Show-Harness reports 86-96% for frontier VLMs and 86% for a demo-fine-tuned 2B Qwen.

Corrected conclusion: the tested configurations did not demonstrate improved complete-task success; H2 improved approach and contact. These experiments do not isolate the limiting model capability or establish that another harness cannot help. The audit found valid IK targets rejected by a slot-length restriction, missing H3 selection context, incorrect recovery history, and unverified skill extraction. H7 was not evaluated. The revised campaign will preserve these historical results and rerun affected comparisons separately.

## 5. Options that remain (not chosen; Jiachen decides)

1. Same interfaces, frontier VLM (Gemini / GPT / Claude), 30 episodes each on Lane A and Lane B: separates model from interface. Needs API keys. Half a day.
2. Show-Harness's small-model route: generate 2 cm-token demonstrations in sim (`scripts/trajectory/real2sim/`), LoRA-fine-tune Qwen (their `train/`), evaluate. 1-2 days, a few GPU-hours. No longer zero-shot.
3. Keep 27B zero-shot but hand localization to a dedicated module (depth/point cloud + grasp detector); Qwen only selects. Conflicts with the "Qwen authors the numbers" goal.

## 6. Infrastructure state on h200-4 (as of 2026-09-12)

- vLLM `qwen3.8-27b-bf16` still running on port 8002, about 82 GB GPU memory, launched by `/home/jli/state/panda-qwen38/start_direct_qwen.py` (attestation + token regenerated; the released harness's own scripts still work against it). Stop it if the GPU is needed.
- Show-Harness checkout `/home/jli/work/show-harness` (commit 137d571), venv `.venv` (Python 3.11, mani_skill 3.0.1). Vulkan works after `libnvidia-gl-580=580.173.02-1ubuntu1` was installed with `dpkg -i` (dpkg reports it "unconfigured"; rendering works). Always `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`. BlockPAP-v1 is unavailable (RLinf public repo lacks `real_franka.real2sim_env`); PickCube-v1 was used.
- No experiment processes are running. Both checkouts are in sync with the Mac repo at commit 24f82aa+ (this handoff adds one commit).
- Pitfalls: `pkill -f` / `pgrep -f` patterns match your own ssh/monitor command line (e.g. a log path containing `matrix/phaseC-h6`); use bracket-escaped patterns and never grep a log path inside a wait loop's own command. Never rsync into the frozen checkout while a matrix runs there.

## 7. Where things are

| What | Path |
|---|---|
| Report | `REPORT.md` (sections 1-12), tables in `results-direct/` |
| Runbook | `RUNBOOK.md` |
| Follow-up plan | `docs/superpowers/plans/2026-09-11-show-harness-style-qwen27b.md` |
| Matrices | `h200-4:/home/jli/state/qwen-direct/matrix/{phaseB-clean, phaseC-ee-short, phaseC-joint-short, phaseC-ee-full, phaseC-h6, phaseD-singles, phaseD-combos, phaseE-test, sem-dev, sem-test}` |
| Recovery states + continuations | `.../banks/recovery-all/states.json`, `.../recovery/phaseC-states/` |
| Milestone inspections | `.../banks/*-inspect/inspection*.json`, `.../matrix/*-milestones.md` |
| Lane A rollouts | `/home/jli/state/show-harness/rollouts/MS-PickCube-v1/`, logs in `results-direct/laneA/` |
| Videos | every run dir has `episode.mp4`; representative: Phase B success `phaseB-clean/PickPlaceCounterToSink-s0-ee-short-clean`, Phase E success `phaseE-test/PickPlaceCounterToSink-s100-ee-short-clean`, H4 near-miss `phaseC-ee-short/PickPlaceCounterToSink-s0-ee-short-h4` |
