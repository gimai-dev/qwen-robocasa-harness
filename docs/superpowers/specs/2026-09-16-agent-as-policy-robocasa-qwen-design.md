# Agent-as-policy harness on RoboCasa with Qwen3.8-27B — design

Date: 2026-09-16. Box: h200-2 (the RoboCurve/GPT-6 Astra replication stays on h200-4 and is not touched).

## Goal

Run the agent-as-policy protocol (arXiv 2609.12541: a coding agent drives a robot through a
file-bridge CLI with `frames` / `deproject` / `move_ee` / `gripper` and judges its own completion)
with a local Qwen3.8-27B as the agent, on the three RoboCasa pick-and-place tasks of the earlier
direct-control campaign, and measure whether the interface moves the needle where direct numerical
control did not (direct control: 1/60 on the test seeds; approach within 10 cm in 2/15).

Goal for the overnight run: on the development starts (3 tasks × seeds 0–2) reach at least one
official success and clear the direct-control milestones (approach, contact, lift) on more starts
than clean direct control did; then freeze the best configuration and run the unseen seeds.

## What is reused verbatim

- The agent-side contract of agent-as-policy: `robot_client.py` (their file, unchanged), the command
  set and wording of `README_interface_real.md`, the rules and deliverable of `PROMPT_stack_blocks.md`
  (task text swapped for RoboCasa's language instruction), the budget model (free vs counted commands),
  the `RESULT.md` self-verdict regexes of `run_paper_pyramid.sh`.
- The validated simulator side of the direct-control campaign: `direct.executor.Simulator` (sandboxed
  RoboCasa child with the mailbox protocol), `direct.kinematics` (FK/IK, straight-line planning,
  multistart fallback), `READY_Q`, and the official success predicate sealed at finish.

## Components (all under `agp_sim/`)

| unit | role | depends on |
|---|---|---|
| `server_sim.py` | the session server: watches `<session>/bridge/req_*.json`, executes one command at a time on the child, writes `resp_*.json`; physics guards only (reach radius, z range, step); logs ground truth to `evaluator.jsonl` outside the session | `direct.executor`, `direct.kinematics` |
| `direct/sim_child.py` (+`render`, +`evaluator`) | new mailbox command `render` = RGB + metric depth for named cameras at up to the model's offscreen size, with world-frame pinhole calibration; every observation now carries an `evaluator` block (object pose, gripper-object distance, contact, official success) | robosuite `sim.render` |
| `agent_loop.py` | the coding agent: OpenAI chat completions to vLLM with two tools (`exec` in the session dir, `view_image`), inline `<tool_call>` parsing for Qwen3.8's XML-ish format, image window (last 8) and context compaction (44k soft cap on a 64k model), events + usage files | vLLM at 127.0.0.1:8002 |
| `run_episode.py` | one episode: session layout → server boot (instruction, ready pose, image size) → PROMPT/README rendering → agent → SERVER_STOP → sealed outcome → `result.json` with milestones | the three above |
| `run_matrix.py` | sequential/parallel (task, seed) runs, resumable, `summary.md` | `run_episode.py` |

Frames: every pose the agent sees or sends is in the robot base frame; the tool frame is the AgP
convention (+z out of the fingers, fingers close along tool y) = the FK grip_site frame rotated
−90° about z. Camera poses are converted from MuJoCo's (x right, y up, −z forward) to the CV
convention the AgP README describes, so `Xc = Rᵀ(X − t)`, `pixel = K·Xc/Xc[2]` holds verbatim.
Verified by `tests/smoke_server.py`: projecting the TCP with the saved calibration lands within
~1 px of the child's own projection, plane-deproject of that pixel returns the TCP within 1 mm.

Differences from the real rig, deliberate: a 7-DOF arm on a mobile base (`move_base` added, joints
are 7), three RGB-D cameras instead of RGB-D wrist + RGB top, depth is exact, the gripper is
binary open/close (fraction reported from the measured width), the official RoboCasa predicate
replaces the human verdict, and the agent's verdict is recorded next to it.

## Protocol

- Tasks: PickPlaceCounterToSink, PickPlaceCounterToDrawer, PickPlaceStoveToCounter (pinned scenes
  from `/home/jli/state/agp-sim/scenes`). Development seeds 0–2, validation 10–14, test 100–119 —
  the same split as the direct-control campaign.
- Budget: 400 counted commands, 45 min wall clock, 160 model turns; temperature 0.2; images 512×480.
- Success: official predicate at SERVER_STOP. Milestones from `evaluator.jsonl`: approach (<10 cm),
  touched, lifted (>3 cm while in contact), displaced.
- Improvement loop (autonomous, overnight): after each dev matrix read every transcript, classify
  the failure (perception / reach / grasp / placement / loop), change ONE thing in prompt, README or
  server (never the success predicate, never the task set), rerun the dev matrix. Freeze when a
  configuration is ahead on both success and milestones, then run validation seeds.

## Infrastructure decisions

- vLLM on h200-2 restarted (2026-09-16 05:49) with `--enable-auto-tool-choice --tool-call-parser hermes
  --limit-mm-per-prompt {"image":12} --max-num-seqs 2 --gpu-memory-utilization 0.75`; same model id,
  port and token, so the qwen10 project's clients still work. Original command line saved at
  `/home/jli/state/agp-sim/qwen-original-cmdline.txt`; restart script `start_qwen_agp.sh` next to it.
- Checkout `h200-2:/home/jli/work/agp-sim` (rsync of this repo), runs under `/home/jli/state/agp-sim/`.
- The EGL rootfs path in `direct/executor.py` is now discovered by glob (h200-2 ships 580.159.04).
