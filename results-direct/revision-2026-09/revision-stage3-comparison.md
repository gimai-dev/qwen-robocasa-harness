# Stage3 comparison: joint targets and one-shot trajectories

Stage3 is complete:27 valid scored episodes after selective repairs. Every condition scored0/9 official task successes. Joint-short and both one-shot conditions recorded no target contact or retained lift. The repaired EE-short baseline remains the only clean interface with a verified local lift in these nine development starts. Proceed to the planned18 H3/H4 episodes; there is no remaining demonstrated execution-contract defect blocking that comparison.

| Clean condition | n | Task success | Sampled approach≤10cm | Target contact | Closed-gripper contact | Lift≥3cm | Mean steps | Mean Qwen calls | Mean wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EE-short (Stage2) |9|0|2|2|1|1|808.9|44.2|363.1|
| Joint-short |9|0|0|0|0|0|822.2|42.3|318.2|
| EE-full |9|0|0|0|0|0|82.2|1.0|19.5|
| Joint-full |9|0|0|0|0|0|157.8|1.0|28.9|

## Joint-short

The nine runs executed370 slots:266 joint targets and104 base motions.331 slots completed,39 ended partial, and four invalid outputs were rejected. Seven runs stopped and two exhausted the step budget. The remaining invalid outputs were outside the correctly advertised limits; the misleading error-bound feedback was repaired before the affected starts were rerun.

The closest recorded target distances were:

| Task | Seed0 | Seed1 | Seed2 |
|---|---:|---:|---:|
| CounterToSink | 0.878m | 0.793m | 0.520m |
| CounterToDrawer | 0.239m | 0.103m | 0.506m |
| StoveToCounter | 0.170m | 0.569m | 0.353m |

Joint-short consumed7400steps,381Qwen calls,1640878prompt tokens,57884completion tokens,1302.2Qwen seconds,1075.7SAM seconds, and2864.1aggregate wall seconds. Correctly executed joint targets frequently failed to move the TCP toward the intended object. Low rejection counts therefore do not establish useful control.

## Full trajectories

Both one-shot conditions scored0/9 task successes and recorded no target approach within10cm, contact, retained grasp, or lift. They used the same nine starts as the EE-short baseline. Each made exactly one Qwen control call and one initial SAM call per episode; there was no intermediate replanning. All18 generation finish reasons were `stop`, so output truncation does not explain the short sequences.

| Metric | EE-full | Joint-full |
|---|---:|---:|
| Emitted slots, total |48|71|
| Executed slots, total |37|71|
| Median emitted sequence length |7|9|
| Range of emitted sequence lengths |1–10|1–15|
| Executed simulator steps |740|1420|
| Qwen calls |9|9|
| Prompt tokens |32712|34593|
| Completion tokens |3710|5597|
| Qwen seconds |83.7|122.7|
| SAM seconds |30.1|31.8|
| Aggregate wall seconds |175.5|259.8|
| Partial execution slots |12|29|
| First execution rejections |3|0|
| Executed closed-gripper-command slots |8|27|
| Those ending with finger gap below5mm |8|27|

EE-full emitted lengths in task-within-seed order (Sink, Drawer, Stove) are `[1,8,10,7,10,9,1,1,1]`. Five sequences contain only base actions: Sink0, Stove0, and all three seed2 starts. Drawer0 ends after lift/raise-clear, without placement/release. Drawer1 and Stove1 include the named task stages, but their physical outcomes fail. Sink1 misidentifies the target as already in the sink and attempts a pickup there. Three sequences stop at an unresolved target: Drawer0 slot4, Sink1 slot6, Drawer1 slot7. The Drawer1 receipt reports a straight-path branch discontinuity, and its direct-IK fallback is `kinematically_unresolved` with0.2076m residual at a requested horizontal base distance of1.061m. It does not reproduce the historical rejection of an already valid fallback.

Joint-full lengths are `[15,9,12,1,1,10,4,10,9]`. Drawer1 emits a single navigation action, Sink1 a single hold after falsely declaring the object already placed, and Sink2 ends before closing the gripper. The other six contain named manipulation stages, but none records target contact.29 of71 slots end partial: the following specified slot starts from the actual achieved state, not from the intended previous target. No task stages or extra settle actions are inserted by the executor.

Interpretation: the low one-shot cost reflects short or unsuccessful execution. Joint-full's longer, syntactically complete sequences do not demonstrate successful physical planning. These are negative results for this Qwen prompt, numerical representation, slot budget, and observation setting; they do not establish a general limitation of VLM controllers.


## Provenance and retained attempts

The final data use the repaired common executor and original initial postures. Four unexposed joint-short runs and16 full runs remain at`ef3e956`; the first two unchanged EE-full runs remain at`88243a1`. Five joint-short starts that received outward-rounded error bounds were retained with their logs and inspections and replaced at`e34dbf7`. The intervening accounting and H7-text changes do not alter these retained trajectories. Rejection totals come from decision events, preserving raw earlier counters.

Five earlier joint-guide attempts at`88243a1` and five later error-feedback attempts at`ef3e956` are excluded. The latter consumed4440steps,237calls,1058793prompt tokens,36117completion tokens, and1780.1wall seconds; all failed. The new error-bound regression failed four advertised endpoints before repair and passes all14 after repair. All48 repair regressions pass locally and remotely.

## Limits and next action

These nine matched development starts diagnose this implementation and prompt configuration; they do not establish a generalization rate or general model incapacity. Intermediate milestones use saved observation states, so transient events between observations may be missed. The one-shot prompts still inherit common control guidance alongside their explicit full-sequence instruction; the result describes that tested prompt. Aggregate wall time is summed episode cost, not elapsed campaign time.

Run H3 and H4 on the same original-posture EE-short starts, with the Stage2 clean baseline unchanged. Record H3 candidate/selection behavior and extra calls, and H4 requested versus executed duration and observations near the target. Then perform the independent H1/H5/H6/H8 screen and the now-qualified H7 screen.

Evidence: [results](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage3-data/results.json), [event analysis](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage3-data/analysis.json), [manifest](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage3-data/manifest.json). Remote raw data: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/stage3/`.
