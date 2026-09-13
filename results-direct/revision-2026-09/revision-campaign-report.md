# Direct numerical Qwen control: repaired campaign

**Completed: 117 development episodes and 45 fresh validation episodes, all scored and inspected. Clean, H7, and H8 each finish the fresh comparison at 0/15 terminal task successes. H7 improves approach/contact on the new starts, but neither selected harness improves completion.**

The repairs were necessary: the old implementation rejected valid bounded motions, supplied incomplete or mismatched harness context, and mislabeled some evidence. The repaired controller now executes those motions and supports the intended harness comparisons. Development outcomes show better local manipulation in several conditions, but no reliable complete-task control.

## What changed and why

Seven local commits follow the audited `df164fe` revision. The final policy revision is `1223102` on branch `direct-control` in `/Users/jiachen/Desktop/first try/qwen-direct-control`.

| Area | Demonstrated defect | Repaired behavior |
|---|---|---|
| EE execution | A valid fallback IK solution was rejected when the entire joint move would exceed one slot; posture regularization could also prevent a valid pose solution. | Execute a bounded prefix and report partial progress; refine pose accuracy within the existing joint limits. |
| Mobile base | Per-axis friction distorted turned-heading translation, and chassis hold lacked usable correction authority. Restored snapshots also needed the original controller frame. | Calibrated friction compensation and correctly restored controller coordinates. |
| Wrist feedback | A 218.219 N reset transient was treated as the unloaded baseline, suppressing subsequent load feedback. | Robot-derived unloaded weight of 5.1012 N; reset/restore feedback remains unavailable until physics advances. |
| Joint interface | Prompt and error-message endpoints rounded beyond accepted bounds; the downward-orientation formula omitted its zero-roll condition. | Advertise executable inward-rounded bounds and state the formula's actual assumptions. |
| H1 grounding | Text supplied as many as 14 regions while the gallery displayed only the first 12. | Display every supplied region. Three exposed runs were retained and replaced on the same starts. |
| H3 preview | Selection lost the task/state/history and used stale base calibration and inconsistent sequence transforms. | Preserve full context and proposals; preview bounded motion with the actual base calibration and advancing transforms. |
| H4 timing | An unchanged explicit gripper command incorrectly forced a long slot. | Short arm/hold slots work when the command is unchanged; actual gripper changes keep their required duration. |
| History and banks | Recovery used the wrong prior images; some failures lacked pre-state; an open-to-regrasp action was labeled a successful lift. | Align saved observations with decisions, retain real pre-state, and require target-following motion with retained grasp for local skill qualification. |
| H8 and recovery metrics | Numeric TCP/gap agreement could be confused with object retention; recovery states lacked independent task-completing qualification. | Holding remains explicitly unknown without evidence. Only independently qualified states enter the recovery denominator. |
| Accounting | Invalid outputs and failed request attempts were omitted from some counters. | Count rejection events and request attempts, retaining unknown token usage as unknown. |

The actual robot's FK, joint limits, quaternion order, TCP conversion, and gripper polarity were correct. The initial audit's rotation-sign allegation was withdrawn after inspecting and probing RoboCasa's selected legacy base controller. The supported base repairs concern friction, hold authority, and restoration of its coordinate frame.

The final 49 focused repair tests passed locally and remotely. Physical probes turned a saved zero-motion IK rejection into a 20-step bounded motion of approximately 0.121 m, reduced a restored-heading sideways drift from 0.111 m to 0.001 m, and verified the force proxy on clear, contact, and retreat states. Historical-image and false-skill reproductions also passed. These are verified execution and evidence repairs; task-level effects are reported separately below.

Detailed receipts are in `/Users/jiachen/Desktop/first try/qwen-direct-control/results-direct/revision-2026-09/`. The original raw campaign and superseded revision attempts remain available. The original remote checkouts were not overwritten.

## Completed development: 117 episodes

The campaign covers all four requested clean interfaces, the ready-pose control, and H1–H8 individually on the same nine development starts. Every final scored condition has 0/9 terminal task successes. The complete [development table](revision-development-summary.md) includes paired local-progress outcomes and costs.

![Matched development milestones](/Users/jiachen/Desktop/first try/qwen-direct-control/results-direct/revision-2026-09/revision-development-milestones.png)

EE-short clean reached target contact on 2/9 starts and a retained lift on 1/9. Joint-short, EE-full, and joint-full recorded no target contact. Full outputs ended normally and often failed to supply or execute a complete task sequence; their low call count is not successful task efficiency. H2 and the ready-pose condition did not improve completion on these starts.

The strongest development contact signal came from H7: 7/9 contact episodes, 6/9 closed-contact episodes, and 2/9 retained lifts, versus clean's 2/1/1. H8 achieved 3/9 retained lift episodes. H1, H3, and H6 each achieved two; H5 achieved one; H4 achieved none. H3 used 826 model calls across nine episodes, including its selection calls, without a terminal success. H4 exercised the repaired shorter slots but rarely obtained extra observations near the target.

H7 and H8 were selected for fresh validation before creating or running the validation scenes. H7 received two overlapping local grasp/lift examples from one clean lemon episode. Its independent applicability episode demonstrated an apple lift, but did not complete the task. Every development H7 call received the same two examples; this tests a small fixed skill context, not a demonstrated adaptive retrieval library.

## Fresh validation

Clean, H7, and H8 completed the same 15 previously unused starts: three tasks × seeds 20–24. All use `1223102`, the original posture, the frozen banks, 900 simulator steps, 1,200 seconds, and at most 180 control decisions. There were no validation infrastructure failures or retries. All45 offline inspections are complete.

| Condition | Terminal success | Approach | Target contact | Closed contact | Lift milestone | Verified retained lift |
|---|---:|---:|---:|---:|---:|---:|
| Clean |0/15|3|4|3|2|2|
| H7 skill context |0/15|8|6|5|2|2|
| H8 expected effects |0/15|4|3|2|2|1|

H7 adds five approaches with no paired losses and three contacts with one loss. It gains a Drawer23 tongs lift while losing clean's Sink20 eggplant lift, leaving no net lift gain. H8's development lift advantage does not carry over: it has one verified retained apple-lift episode and a brief tongs elevation during closing. The latter satisfies the original 3 cm closed-contact milestone but does not contain a retained object-following lift fragment. Both measurements are preserved.

No validation observation satisfies an intermediate official goal predicate. Each observed0/15 success rate has a two-sided 95% exact binomial interval of 0–21.8%; the matched terminal outcomes are all failures. This does not establish equivalence.

Fresh failures repeat the development diagnosis. H7 deliberately opens after retained apple and tongs lifts because Qwen believes the objects are back on their source surfaces. Clean also repeatedly opens a real apple grasp after misreading its finger gap. H8 deliberately releases the apple at an incorrect destination. Six stops across the 45 runs reflect false completion beliefs; eleven correctly acknowledge an unfinished task near the budget limit.

Mean episode times are412.9 s for clean,413.9 s for H7, and 424.6 s for H8. H7 uses 22.4% more prompt tokens than clean. The [fresh validation report](revision-validation-comparison.md) contains the paired effects, failure transitions, and complete costs.

## Why local progress still fails to become completion

**Qwen sometimes undoes a retained grasp.** H5 Stove1 decision 36 and H6 Stove2 decision 30 deliberately open the gripper after verified lifts because Qwen believes the object is still in the pan. H8 Stove1 decision 32 similarly mistakes the closed-on-object gap for a failed close. Public observations already contain the gripper command and measured gap; those values alone do not solve the interpretation problem.

**Completion recognition can fail after the task has actually been achieved.** H7 Stove2 has one saved state satisfying the official predicate at observation 40, after decision 41. The lemon is in the bowl and the empty gripper is 0.279 m away. The next action returns toward the bowl, reducing that distance to 0.199 m; the predicate becomes false because it requires distance greater than 0.25 m. The episode ends unsuccessful. This is the only terminal-versus-sampled-goal discrepancy in the 117 development episodes. It is a development case overlapping the skill bank's source scene.

The common controller prompt already states the destination, release, and greater-than-0.25 m clearance requirements (`direct/prompts/system_common.txt:23`). The observed return toward the placed object occurs despite that explicit task definition.

**Incorrect placement is a separate failure.** H7 Sink0 lifts and transports the orange, then releases and claims completion. The final gripper is already 0.658 m away, but the object is not successfully placed in the sink. A retreat-only explanation would be wrong for this episode.

**Perception can fail before numerical control has a useful target.** The visible Drawer0 whisk has no dedicated mask among 21 retained right-view SAM proposals. One supplied region pairs a large counter mask with a wrist-view sliver. Enlarging the gallery fixes missing displays; it does not establish object identity or correct 3-D correspondence. The old logs do not retain rejected raw SAM proposals, so generation versus filtering cannot yet be separated.

**Memory and examples can preserve mistaken beliefs.** H5 sometimes replaces a correct target location with a wrong one. H6 often presents the same failure records across changing action phases. H7's bank has only grasp/lift examples, including source-scene absolute coordinates, and continues to show them during transport and placement. These mechanisms have measurable effects on behavior but no established completion benefit from development alone.

The repaired H8 mechanism checks expected TCP and finger-gap measurements. All 345 holding checks in its development screen remain unknown; it has no independent visual target-retention verifier. Its outcome therefore evaluates this particular expected-effect feedback, not the effectiveness of a visual grasp verifier that has yet to be built.

The [H7 report](revision-h7-comparison.md) includes the temporary-goal recording and inspected states. The [screen mechanism notes](revision-screen-mechanism-notes.md) document the retained-grasp failures. These observations motivate [two targeted follow-up experiments](revision-next-experiments.md): action-outcome evidence and target-directed perception.

## Recovery qualification

Three independent clean continuations attempted to qualify candidate failure states within 400 steps, 600 seconds, and 80 decisions. None completed its original task. The qualified bank is empty, so recovery success rate is **N/A** and no qualified-state clean/H8 comparison was launched. These failed qualification attempts do not establish physical unrecoverability. H8's ordinary-start episodes remain ordinary task evaluations.

## Cost and retained attempts

The campaign contains **162 final scored episodes** (117 development + 45 fresh validation), 18 superseded attempts, two infrastructure failures, one skill applicability episode, and three recovery qualification episodes: **186 episode attempts** in total. The sole success in the all-attempt ledger belongs to the superseded H1 gallery run; it is not a success of the final scored comparison.

| Category | Episodes | Qwen request attempts | Simulator steps | Summed episode wall hours |
|---|---:|---:|---:|---:|
| Final scored comparisons |162|6,769|116,335|16.74|
| Retained superseded attempts |18|764|13,480|1.64|
| Retained infrastructure failures |2|63|780|0.16|
| H7 applicability |1|50|880|0.09|
| Recovery qualification |3|41|720|0.10|
| **Total** |**186**|**7,687**|**132,195**|**18.72**|

Known usage totals are 34,098,025 prompt tokens and 1,381,315 completion tokens. Two retained failed requests have unknown token usage; one of them also predates request logging and is added explicitly to the attempt count. The scored comparisons account for 6,769 calls, 30,114,897 prompt tokens, and 1,244,223 completion tokens.

Wall time is summed across episodes, including concurrent runs; it is not elapsed campaign duration. Fifteen zero-action scene setups took another 322.34 seconds. Offline inspection and physical verification time are outside these episode totals. The [full ledger](revision-cost-ledger.json) retains each category and run.

## Recommendation and handoff

Keep the verified repairs and H7's measured contact result. Prioritize evidence about grasp retention and placement, then target-directed perception. The completed fresh comparison does not justify a larger confirmation of the unchanged policies. The optional 120–180-episode confirmation and semantic-control branch were not launched.

The detailed [next-experiment plan](revision-next-experiments.md) first tests saved observations for grasp/completion interpretation and target correspondence, then proposes at most two nine-run development conditions. Qwen remains the numerical controller. Recovery qualification stays confined to H8. Joint/full prompt studies and skill-bank refinement are separate later questions.

The [reproduction guide](revision-reproduction.md) records code revisions, model settings, launch commands, and archived scripts. Raw revised runs are at `h200-4:/home/jli/state/qwen-direct/revision-2026-09/`; the frozen runtime is `/home/jli/work/qwen-direct-control-revision-r7`.

## Limitations

The development starts informed repairs, banks, and candidate selection. The 117 episodes are a collection of conditions, not 117 independent samples of one policy's generalization. Milestones and intermediate goal attainment use saved observation states, so events between observations may be missed. Closed contact is a progress indicator; a secure lift requires object-following evidence. Terminal scoring remains fixed, and simulator object truth is evaluator-only.

The supplemental intermediate-goal diagnostic was added after candidate selection, when the H7 development discrepancy was noticed. It is exploratory and did not change the selected candidates, primary terminal metric, or rollout behavior.

The final fresh comparison has only 15 starts per condition across three tasks. Per-rate exact intervals describe its limited precision; matched gains and losses describe the paired comparison. Zero observed success or a zero difference cannot establish universal incapability or equivalence. The old 1/60 historical success rates and the revised development counts use different starts and should not be treated as a before/after success-rate experiment.
