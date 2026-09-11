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

## 3. Phase B: four clean baselines

(pending: `/home/jli/state/qwen-direct/matrix/phaseB-clean/summary.md`)

## 4. Phase C: independent harness screen

(pending)

## 5. Costs

(pending)

## 6. Limitations and known issues

- The home posture is a straight-arm singularity; the first descent from it uses joint-space interpolation and usually ends one slot late (`partial`).
- Base velocities below 0.25 do not move the base (installed controller dead zone); the base is blocked by furniture in most initial poses in the forward direction and, once pressed against the counter, also sideways.
- SAM region ids are not stable across steps; positions are surface centroids, 1-3 cm above/outside the geometric centre.
