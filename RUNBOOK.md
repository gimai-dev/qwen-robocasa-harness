# RUNBOOK: direct numerical Qwen control (branch `direct-control`)

## Checkouts and locations

| What | Where |
|---|---|
| Working repository (Mac) | `/Users/jiachen/Desktop/first try/qwen-direct-control`, branch `direct-control` (cloned from the published harness at `/Users/jiachen/Documents/Codex/2026-09-06/ban/publish/qwen-robocasa-harness`, history preserved) |
| Runtime checkout (h200-4) | `/home/jli/work/qwen-direct-control` (frozen copy used by a running matrix) and `/home/jli/work/qwen-direct-control-dev` (development copy); each episode subprocess imports code from its own checkout, so never rsync into a checkout while its matrix runs |
| New code | `direct/` package (kinematics, actions, sim_child, executor, perception, sam_server, observation, policy, methods, harnesses, episode, matrix) |
| Shared runtime modules | `runtime/robocasa_inspect/` (identical to `/home/jli/work/robocasa-inspect-official/robocasa_inspect`, verified by diff 2026-09-10) |
| Episode outputs | `/home/jli/state/qwen-direct/runs/<name>` (single episodes), `/home/jli/state/qwen-direct/matrix/<name>` (matrices) |
| Pinned initial scenes | `/home/jli/state/qwen-direct/scenes/<Task>-seed<N>.json` (model XML + flattened state + ep_meta; created on first use, restored exactly afterwards) |
| Simulator | `/home/jli/work/robocasa-inspect-official` (RoboCasa 1.0.1, robosuite 1.5.2, MuJoCo 3.3.1), Python `/home/jli/work/robocasa-inspect-official/.venv/bin/python` |
| SAM | `/home/jli/state/qwen-rgb-sam2/.venv/bin/python`, checkpoint `sam2.1_hiera_small.pt`, run as a persistent CUDA server per episode (`direct/sam_server.py`) |
| Qwen | vLLM `qwen3.8-27b-bf16` on `http://127.0.0.1:8002/v1` (h200-4), identity/attestation/token files under `/home/jli/state/panda-qwen38/`. Restarted 2026-09-10 with `/home/jli/serving/venv-llm/bin/python /home/jli/state/panda-qwen38/start_direct_qwen.py` (copy of the released start script with `--limit-mm-per-prompt {"image":8} --max-num-seqs 4`); rerun that script after a box restart, it regenerates the attestation and token |

Sync Mac -> box:

```bash
rsync -a --exclude .git --exclude videos --exclude results --exclude __pycache__ "/Users/jiachen/Desktop/first try/qwen-direct-control/" h200-4:/home/jli/work/qwen-direct-control/
```

## Commands (verified on h200-4)

All commands run on the box with:

```bash
cd /home/jli/work/qwen-direct-control && export PYTHONPATH=/home/jli/work/qwen-direct-control:/home/jli/work/qwen-direct-control/runtime
PY=/home/jli/work/robocasa-inspect-official/.venv/bin/python
```

Interface smoke test (FK, EE displacement, gripper polarity, 20-step receipts, base motion, unreachable rejection):

```bash
$PY direct/tests/smoke_child.py --run /home/jli/state/qwen-direct/smoke-N --scenes /home/jli/state/qwen-direct/scenes
```

Perception smoke test on a saved run:

```bash
$PY direct/tests/smoke_perception.py /home/jli/state/qwen-direct/smoke-N
```

One episode:

```bash
$PY -m direct.episode --task PickPlaceCounterToSink --seed 0 --interface ee --mode short --method clean --out /home/jli/state/qwen-direct/runs/<name>
```

Options: `--interface ee|joint`, `--mode short|full`, `--method clean|h1..h8`, `--method-config '{...}'`, `--steps-budget 900`, `--wall-budget-s 1200`, `--max-decisions 180`, `--scenes DIR`, `--restore-from SNAPSHOT` (H8).

Matrix:

```bash
$PY -m direct.matrix --tasks PickPlaceCounterToSink PickPlaceCounterToDrawer PickPlaceStoveToCounter --seeds 0 1 2 --interfaces ee joint --modes short full --methods clean --out /home/jli/state/qwen-direct/matrix/<name> --parallel 2
```

`summary.json` / `summary.md` are rewritten after every finished episode; a run directory with an existing `result.json` is reused, so a matrix can be resumed.

Analysis (tables, paired differences vs clean with bootstrap intervals, costs):

```bash
$PY -m direct.analyze --matrices /home/jli/state/qwen-direct/matrix/<a> /home/jli/state/qwen-direct/matrix/<b> --out /home/jli/state/qwen-direct/matrix/<name>-results.md
```

H6/H7 banks from clean development runs, then use them with `--method h6 --method-config '{"bank": ".../h6-failures.json"}'` (H7: `h7-skills.json`):

```bash
$PY -m direct.banks --runs /home/jli/state/qwen-direct/matrix/<clean matrix> --out /home/jli/state/qwen-direct/banks/<name>
```

H8 recovery evaluation (inspect clean-run snapshots evaluator-side, select recoverable failure states, continue clean vs h8 from the same state with the 400-step / 600 s / 80-decision recovery budget):

```bash
$PY -m direct.recovery inspect --runs /home/jli/state/qwen-direct/matrix/<clean matrix> --out /home/jli/state/qwen-direct/banks/recovery
$PY -m direct.recovery select --bank /home/jli/state/qwen-direct/banks/recovery --max-states 10
$PY -m direct.recovery continue --bank /home/jli/state/qwen-direct/banks/recovery --methods clean h8 --out /home/jli/state/qwen-direct/recovery/<name> --parallel 2
```

## Matrices actually run (2026-09-10/11)

| Block | Command (from the frozen checkout) | Output |
|---|---|---|
| Phase B clean | `direct.matrix --tasks <3 tasks> --seeds 0 1 2 --interfaces ee joint --modes short full --methods clean --parallel 3` | `matrix/phaseB-clean` |
| Phase C screen | `direct/tests/launch_phaseC.sh` (EE-short clean h1 h2 h4 h4c h5 h3 h8; joint-short clean h2; EE-full clean h3) | `matrix/phaseC-{ee-short,joint-short,ee-full}` |
| Phase C H6 | `direct.matrix ... --methods h6 --method-config '{"bank": ".../banks/phaseC/h6-failures.json"}' --parallel 1` | `matrix/phaseC-h6` |
| H8 recovery | `direct.recovery inspect/select/continue` (states from `banks/recovery-all`) | `recovery/phaseC-states` |
| Phase D singles | `direct.matrix ... --seeds 10 11 12 13 14 --methods clean h1 h2 h3 h4 --parallel 3` | `matrix/phaseD-singles` |
| Phase D combos | `direct.matrix ... --seeds 10 11 12 13 14 --methods h1+h2 h3+h4 --parallel 3` | `matrix/phaseD-combos` |
| Phase E test | `direct.matrix ... --seeds 100..119 --methods clean h2 --parallel 3` | `matrix/phaseE-test` (two infrastructure-failed attempts kept in `matrix/phaseE-test-infra`, rerun on the same scenes) |
| Milestones | `direct.recovery.inspect_run` over each matrix, then `direct.milestones` | `banks/*-inspect`, `matrix/*-milestones.md` |

Milestone inspection restores every snapshot of a run in a sandboxed child and reads object pose, gripper contact and the official predicate; it is evaluator-side only.

## Protocol constants (executor level)

| Constant | Value | Where |
|---|---|---|
| Control slot | 20 simulator steps (1.0 s at 20 Hz); H4/H4c arm/hold actions may request 5 or 10 when the gripper command is unchanged; base and gripper changes retain 20 | `actions.SLOT_STEPS` |
| Full-mode sequence | at most 45 slots | `actions.MAX_FULL_SLOTS` |
| EE path | straight line in position, slerp in orientation, IK every 0.01 m / 2.5 deg, one waypoint per step (0.2 m/s) | `kinematics.plan_pose_segment` |
| EE fallback | if the straight line has no local IK solution (straight-arm home posture), direct multi-start IK + bounded joint-space interpolation; execute the available prefix and return `partial` when it does not finish; receipt `path` says which | `executor.execute` |
| Joint rate cap | 0.08 rad per step; tracking pauses when the measured lag exceeds 0.10 rad | `kinematics.MAX_JOINT_STEP`, `sim_child.TRACKING_LAG_PAUSE` |
| Joint controller | JOINT_POSITION kp=150, damping ratio 1 (released harness configuration), torso absolute zero | `sim_child.joint_controller_config` |
| Gripper | 0 closed / 1 open; drive +1 closed / -1 open inside robosuite; measured width `finger1 - finger2` (0.079 open, 0.002 closed empty) | `sim_child` |
| Base | axis x/y/yaw, absolute v <= 0.5, velocity for 16 steps then brake for 4; 0.5 -> about 0.16 m or 0.464 rad per slot in free space; translation friction compensation preserves direction after turning; arm joints held, chassis hold during arm motion | `actions`, `sim_child.execute_base`, `chassis_hold` |
| Orientation frame | FK `grip_site` frame (+z approach out of the gripper, fingers close along local x); the public eef quaternion is this frame rotated 90 deg about z and is only logged | `sim_child._public_state` |
| Budgets | 900 steps, 1200 s, 180 decisions (episode); rejected actions cost a decision, not steps | `episode` |
| Contact force | scalar: wrist force magnitude minus the free-hanging baseline magnitude (the sensor bias rotates with the wrist, so vector deltas are meaningless); Phase B ran with the earlier vector form | `sim_child.publish` |
| SAM prompt grid | 16x16 points per view from Phase C on (Phase B used 24x24: 2.6x slower under GPU contention, 13 vs 8 regions on the CounterToSink seed-0 start, same target object found) | `sam_server` |
| Snapshots | full simulator state saved at every observation under `sim/snapshots/` (evaluator-side; used by `direct.recovery`) | `sim_child.publish` |
| Qwen decoding | seed 3074294, temperature 0, top_p 1, thinking off, strict JSON schema; 1024 max tokens short, 4096 full | `policy` |
| No-progress stop | 8 consecutive decisions without motion end the episode (`no_progress`) | `episode.MAX_CONSECUTIVE_NO_MOTION` |

## Per-run artefacts

`config.json`, `system-prompt.txt`, `qwen-calls.jsonl` (full user text, raw output, usage, finish reason, latency), `decisions.jsonl` (action, receipt, status), `perception/<decision>/regions.json` + masks, `sim/frames/<seq>/{left,right,wrist}.png`, `sim/mailbox/*.json` (every command and observation with per-step tracking trace), `sim/scene.json`, `sim/snapshots/*.json`, `sim/terminal-outcome.json` (official predicate), `episode.mp4` (10 fps mosaic, one frame per 4 steps), `result.json`.

## Phase A record (2026-09-10)

Interface checks, run `smoke-7` on CounterToSink seed 0 ("Pick the orange from the counter and place it in the sink."):

| Check | Detects | Result |
|---|---|---|
| FK chain vs public relative TCP | joint order / frame error | 1.8e-8 m |
| EE +0.05 m world x | transform, unit, sign | moved (+0.0498, +0.0001, -0.0001) m in 20 steps, straight line, 5 IK waypoints |
| EE -0.05 m z from home | singular posture handling | straight line unsolvable; joint-space fallback used; slot ended partial (TCP first rises as the elbow bends) - documented limitation |
| Gripper close / open | polarity, duration | width 0.0794 -> 0.0019 -> 0.0781 m, 20 steps each |
| Joint1 +0.2 rad | joint order | error 0.0007 rad after 20 steps |
| Base x 0.5 | base coordinate update | base +0.0471 m, TCP +0.0473 m, arm error 0.0006 rad |
| Base yaw 0.5 | yaw sign | +0.2157 rad |
| Target 3 m away | rejection | `unreachable`, 0 steps |
| finish | official predicate | evaluated, `finished_false`, terminal outcome sealed |

Timings: child launch about 20 s; one 20-step slot about 1.1 s; SAM server start about 6 s; SAM three views about 0.5 s.


## Show-Harness-style semantic family (2026-09-12)

Plan: `docs/superpowers/plans/2026-09-11-show-harness-style-qwen27b.md`.

### Lane B: their interface in our loop (`direct/semantic*.py`)

```bash
$PY -m direct.semantic_episode --task PickPlaceCounterToSink --seed 0 --method sem-full --out /home/jli/state/qwen-direct/runs/<name>
$PY -m direct.matrix --tasks <3 tasks> --seeds 0 1 2 --interfaces ee --modes short --methods sem sem+plan sem+plan+rec sem-full --out /home/jli/state/qwen-direct/matrix/sem-dev --parallel 3
```

| Item | Value |
|---|---|
| Vocabulary | MV_FWD/BACK/LEFT/RIGHT/UP/DOWN, ROTATE_CW/CCW, GRASP, RELEASE, DONE, BASE_FWD/BACK/LEFT/RIGHT (base tokens are our mobile-base extension) |
| Interpreter | moves are fixed steps in the robot base frame (+x forward, +y left, +z up) rotated by the measured base yaw: 0.02 m fine / 0.04 m coarse; ROTATE 15 deg about the tool axis; GRASP/RELEASE = hold slot with gripper 0/1; BASE_* = one base slot at |v| = 0.5 |
| Slots | moves 6 sim steps (`MOVE_SLOT_STEPS`), gripper and base 20 |
| Policy call | one token per decision via vLLM `guided_choice`; with the planner, a strict JSON `{subtask_done, token}` instead |
| Images | Image A = agent view (left camera; `sem-full` switches to right when the TCP leaves the left view), Image B = wrist. No SAM regions |
| Text | task, current subtask + done_when, proprioception (gripper height, gripper state, contact force, last action effect), last 8 moves |
| Methods | `sem`: base interface (fine step below 1.15 m TCP height); `sem+plan`: + one planning call (3-6 subtasks with completion criteria, phase approach/align sets fine/coarse); `sem+plan+rec`: + empty-close -> forced RELEASE + rollback to the grasp subtask; `sem-full`: + agent-view selection |
| Ready pose | `READY_Q = (0, -0.35, 0, -2.0, 0, 1.65, 0.785)` reached in up to 3 slots before decision 1 (logged as decision 0); TCP 0.45 m ahead, 0.51 m above the base, tool pointing down |
| Budgets | as the numerical family: 900 steps, 1200 s, 180 decisions |

### Lane A: Show-Harness itself on ManiSkill with our Qwen

Checkout `/home/jli/work/show-harness` (showlab/Show-Harness at 137d571), venv `.venv` (uv, Python 3.11, mani_skill 3.0.1). Vulkan: `libnvidia-gl-580=580.173.02-1ubuntu1` installed with `dpkg -i` (left "unconfigured" by dpkg but the ICD and libraries work); always `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

BlockPAP-v1 (their default, RLinf real2sim rig) is NOT available: RLinf's public repo does not ship `real_franka.real2sim_env`. Lane A therefore uses the stock `PickCube-v1` with `panda_wristcam` (the docs list it as supported "with a much larger domain gap").

Config `configs/robot_maniskill_qwen27b.yaml` (copy of `robot_maniskill.yaml`: env PickCube-v1, agentview base_camera, camera_resolution 256, vlm base_url 127.0.0.1:8002, backend `qwen27b` = provider vllm, model = our served id, max_tokens 24, thinking off; api_key is our token, file mode 600). Runner `scripts/run_maniskill_qwen27b.py` = stock `run_maniskill_mvtoken.py` with the controller switched from the fine-tuned bare-token call to their zero-shot `complete_token` (guided_choice + strict retry). Plugins in this runner: `auto_release` only (their ManiSkill path has no subgoal/recovery plugins; those live in the real-robot/RoboLab runners).

```bash
cd /home/jli/work/show-harness && export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
.venv/bin/python -u scripts/run_maniskill_qwen27b.py --robot-config configs/robot_maniskill_qwen27b.yaml --version v3 --episode-index 0 --max-steps 60
```


## September 12 revision campaign

The accepted repair plan is executed in a separate checkout and result root. Preserve all earlier raw runs. The serving model, scenes, perception setting, and budgets remain fixed. Code revision is recorded in every episode config.

Run from the frozen revision checkout on h200-4 with the existing RoboCasa Python and `PYTHONPATH=.:runtime`:

```bash
python -m direct.revision_campaign --batch stage2 --parallel 3
python -m direct.revision_campaign --batch stage3 --parallel 3
python -m direct.revision_campaign --batch stage4 --parallel 3
```

Review each completed batch before launching the next. Stage2 has 27 EE-short episodes (clean, H2, clean+ready); Stage3 has 27 interface/sequence episodes (joint-short, EE-full, joint-full); Stage4 has 18 repaired H3/H4 episodes. Each condition uses the same three tasks and seeds 0-2. Results, fixed manifests, videos, and evaluator-only milestone inspection are under `/home/jli/state/qwen-direct/revision-2026-09/`.

Ready initialization may take up to three 20-step slots. All initialization steps and elapsed time count toward episode budgets; decision0 records initialization and is excluded from VLM decision count. Each action records its actual observation sequence, requested/selected duration, and executed duration. `control_wall_s` ends before teardown; `wall_s` includes teardown and video rendering. `final_state_wall_s` records when the last executed action state was observed and provides conservative timing evidence for recovery qualification without charging later model waiting or video rendering to task completion. H3 previews replace current image slots, retaining prior images within the server's eight-image limit.

H6 banks require actual pre-action state and same-failure correction evidence. H7 requires independently inspected object-following lift and transfer validation. Recovery candidates are not a qualified denominator until an independent continuation completes the original task from that exact snapshot within 400 steps, 600 seconds, and 80 decisions. An empty qualified set has RSR N/A.

The restored-state chassis target is reset to the restored pose while retaining the inner controller frame from the scene reset. H3 yaw preview uses the measured nonlinear curve, including small nonzero motion at velocity0.1. The common prompt distinguishes straight Cartesian tracking from the bounded joint-space fallback; finger gap alone is described as contact evidence, not proof of holding.
