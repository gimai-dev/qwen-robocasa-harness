# H7 verified skill context

H7 completed nine matched development starts:0/9 terminal task successes,7/9 approaches and contacts,6/9 closed-gripper contacts, and2/9 retained lift episodes. One intermediate saved state satisfied the official task predicate before later actions made the terminal result false. The117-episode development campaign is complete:0 terminal successes, with one episode containing a sampled successful intermediate state.

| Condition | Terminal success | Any sampled official goal state | Approach | Target contact | Closed contact | Lift ≥3cm |
|---|---:|---:|---:|---:|---:|---:|
| EE-clean |0/9|0/9|2|2|1|1|
| H7 skill context |0/9|1/9|7|7|6|2|

Compared with clean on the same starts, H7 adds five approaches, five target-contact episodes, five closed-contact episodes, and one lift episode, with no paired losses in those metrics. Closed contact is a progress indicator, not proof of a secure grasp. Only the orange and lemon episodes qualify as retained object-following lifts.

## What the policy received

The frozen bank contains two overlapping grasp/lift fragments from one clean StoveToCounter seed2 lemon episode. An independent Stove seed3 apple episode had already passed the local applicability check. Every H7 control call received the same two source records:341 calls,682 record exposures. The examples contain source-scene absolute poses and gripper timing; the prompt explicitly tells Qwen to generate fresh current-scene targets and distinguishes a local skill from complete-task success. There is no automatic replay or task-specific motion routine.

The examples also remain present during transport and placement because this bank has only grasp/lift records. The screen tests this limited bank and retrieval design. The seed2 lemon source overlaps this development comparison; new-scene validation is therefore especially important. Neither the applicability episode nor these screen outcomes add records to the frozen bank.

## Progress and failure transitions

Sink0 grasps and lifts the orange0.130m, carries it, releases at decision16, and stops at17 claiming completion. The target ends at `[4.997,-1.810,0.414]`, with TCP distance0.658m. The official sink predicate still fails. Its clearance requirement is met; the failed placement predicate, rather than an insufficient final distance, explains this result.

Stove2 lifts the lemon0.160m and retains target contact through observation36. Decision38 releases it in the bowl. Qwen then changes its belief, says the lemon is still in the pan, and attempts another grasp. After decision41, observation40 satisfies the official predicate: the object is in the receptacle and the gripper is0.279m away. Decision42 returns the empty gripper toward the bowl, reducing distance to0.199m. The official predicate becomes false. The object stays near the same location; the final distance is0.136m, and Qwen stops at decision46 claiming it cannot finish.

The installed `PickPlaceStoveToCounter._check_success` requires object-in-container and gripper distance greater than0.25m. Thus re-approaching the placed object can undo success even without removing it. This is a demonstrated failure to recognize and preserve completion. The terminal metric remains0/9 under the fixed campaign protocol; the sampled intermediate-goal metric is1/9. A scan of all117 completed development result/inspection summaries found this to be the only terminal-versus-sampled-success discrepancy.

The spoon, pizza cutter, tomato, and juice-bottle starts reached closed contact but did not record a qualifying lift. Sweet-potato contact did not become a hold. The whisk and pepper starts never approached within10cm. Sink1 and Sink2 also stop with false completion claims. Drawer0 and Stove2's stops explicitly acknowledge an unfinished task near the budget limit.

## Cost and selection

H7 used6520steps,341Qwen calls,1683595prompt tokens,51206completion tokens,1210.5Qwen seconds,1020.6SAM seconds, and2674.7aggregate episode wall seconds. It executed326 twenty-step slots, with133 completed and193 partial receipts; ten unresolved targets were rejected. All341 model responses finished normally. Mean cost is37.9calls and297.2wall seconds per episode. Shorter unsuccessful Sink episodes contribute to that lower runtime, so it is not a demonstrated cost per successful task.

H7 and H8 were selected for the fixed45-episode fresh validation alongside clean: three tasks × seeds20–24 × three conditions. The selection was recorded after development inspection and before validation scene setup or policy rollout. H7 supplies the broadest contact/grasp progress; H8 supplies the strongest lift signal. H1/H3/H6 have narrower local gains, and the other conditions do not justify expanding this validation.

The intermediate-goal finding was diagnosed after candidate selection. It does not change the primary terminal metric, selected methods, or rollout behavior. The final validation report also includes sampled intermediate-goal counts as a supplemental diagnostic. No evaluator success signal is added to the policy and no automatic stop is introduced during validation.

A [recording of the lemon episode](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/videos/h7-lemon-transient-goal.mp4) is retained with the report. The source video samples one frame every four simulator steps and encodes at10fps.

Evidence: [results](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/h7-screen-data/results.json), [analysis](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/h7-screen-data/analysis.json), [lemon inspection states](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/h7-screen-data/stove-s2-inspection.json), [selection record](/Users/jiachen/Desktop/first try/qwen-direct-control/results-direct/revision-2026-09/revision-validation-selection.json), [development chart](/Users/jiachen/Desktop/first try/qwen-direct-control/results-direct/revision-2026-09/revision-development-milestones.png).

## Limits

These nine development starts and the small bank do not establish generalization. Milestones and intermediate-goal counts use saved simulator observations, so events between observations may be missed. The intermediate predicate is evaluated offline from a saved state. H7's ordinary runs are separate from recovery qualification; the qualified clean-source recovery bank remains empty and RSR remainsN/A.
