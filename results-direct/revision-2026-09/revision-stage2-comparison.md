# Stage2: repaired EE control, relative actions, and ready initialization

**The27-episode development comparison is complete. All three conditions scored0/9 full-task successes. The execution repairs held; clean produced one verified grasp and lift. Proceed to the joint and full-trajectory baselines and the independent harness experiments.**

All scored runs used revision`88243a1`, the same Qwen/SAM setting, the original three tasks, and seeds0–2. Each condition retained900 simulator steps,1200seconds, and180 control decisions. The ready condition charged its60 initialization steps per episode within that budget.

## Outcomes

Counts below are episodes satisfying each milestone.

| condition | n | task success | approach≤10cm | target contact | closed-gripper contact | lift≥3cm |
|---|---:|---:|---:|---:|---:|---:|
| EE-short clean | 9 | 0 | 2 | 2 | 1 | 1 |
| EE-short H2 relative | 9 | 0 | 1 | 2 | 0 | 0 |
| EE-short clean + ready | 9 | 0 | 2 | 1 | 1 | 0 |

The earlier H2 approach advantage did not appear on these nine starts. H2 incurred136 rejected actions, compared with32 for clean and26 for ready. Its mean model-call count was also higher.

Ready initialization had mixed effects. It produced closed-gripper contact on the seed1 spoon task, where clean stopped about10.8cm away. On sink seed2, clean contacted and displaced the target while ready did not contact it. These results do not support adopting ready initialization globally; subsequent baseline comparisons retain the original posture.

Clean StoveToCounter seed2 grasped and lifted the lemon by approximately14.1cm, transported it, and released it. Qwen then stopped without executing a retreat. The final gripper–target distance was8.1cm, below the prompt's required25cm clearance. The full task was false; the local grasp/lift is verified.

## Execution and remaining failure mechanisms

The scored batch executed1007 policy action slots, each20steps, plus27 initialization slots in the ready condition. Valid joint-space fallback prefixes executed59times across the three conditions. The pre-retry audit of22 branch-discontinuity rejections found an unresolved final-pose solve in every case; it did not find a solved target rejected by the removed whole-slot cutoff.

The rollout evidence separates several policy failures:

- **Object identification:** Qwen identified robot parts and a brown distractor as the whisk. Initial sink segmentation correctly located the orange, but later estimates shifted substantially.
- **Numerical representation:** Qwen sometimes called identity quaternion`[0,0,0,1]` downward-facing, and H2 sometimes supplied absolute positions in its displacement field. One request became a3.455m segment and was rejected.
- **Use of feedback:** repeated base commands often produced negligible measured displacement near furniture.
- **Completion judgment:** two seed1 sink conditions declared completion while the actual target remained near its original counter position. The verified lemon grasp also ended with no post-release retreat.

These are useful targets for different harnesses: visual grounding, memory, candidate previews, action timing, and explicit effect/recovery reasoning.

## Resource use

| condition | mean steps | mean Qwen calls | mean rejected actions | mean episode wall seconds | total prompt tokens | total completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| clean | 808.9 | 44.2 | 3.6 | 363.1 | 1,606,055 | 58,119 |
| H2 | 740.0 | 52.6 | 15.1 | 405.1 | 1,973,911 | 73,430 |
| clean + ready | 748.9 | 37.6 | 2.9 | 313.4 | 1,365,270 | 49,873 |

The27 scored episodes used20,680 simulator steps,1209 Qwen calls,4,945,236 prompt tokens, and181,422 completion tokens. Summed episode wall time was9734.6seconds; this is aggregate episode time, not elapsed campaign time under concurrent execution. Ready initialization contributed540steps and33.15seconds.

One H2 transport failure was retained separately and successfully rerun to a scored outcome. It used80steps and5 logged model calls; one additional transport-failed request has unknown token usage. Five earlier attempts at`0ce14c7`, before the force-baseline repair, also remain separately retained. Neither group enters the27-episode outcome table.

## Banks and next work

The nine repaired EE-short clean episodes produced87 failure records:32 unresolved IK cases,45 base-motion failures, and10 empty-close cases. Five base-motion corrections have supporting later motion; other corrections remain unverified. All45 base-failure records used translation speeds of magnitude0.3–0.5, above the documented dead zone.

H7 has two overlapping verified grasp/lift records from the single lemon episode. It needs an independent applicability check before its main screen. Recovery candidates have not yet been qualified by successful independent original-task continuations, so no recovery-success rate is claimed.

Next, execute27 clean-interface episodes: joint-short, EE-full, and joint-full, nine each. Then execute18 H3/H4 episodes. Use the original starting posture and the same frozen shared implementation. Follow with the planned independent H1/H5/H6/H8 screen; qualify H7 and recovery separately.

## Limitations

These nine matched development starts do not estimate generalization or establish equivalence. Contact, distance, and lift metrics use saved observation states and can miss transient events. H4's later observation density will affect intermediate-event detection. Closed-gripper contact is a progress metric; verified skills additionally require the object to follow a lift. H2 uses an explicit suffix overriding an earlier absolute-position definition; a future dedicated prompt can test whether removing that mixed presentation reduces the observed confusion. None of these conditions demonstrated reliable full-task completion.

## Evidence

- [Per-run results and milestones](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage2-data/results.json).
- [Action, rejection, timing, and cost analysis](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage2-data/analysis.json).
- [Image-and-trace diagnosis](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/revision-first-start-diagnostics.md).
- Raw data: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/stage2/`.
- Source banks: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/banks/`.
