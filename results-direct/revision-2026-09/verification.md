# September revision: execution and evidence checks

The historical code baseline was `df164fe`. The accepted plan is the September 12 revision and selective rerun plan in the Codex collaboration workspace. This record covers the repairs before robot-task evaluation.

## Confirmed repairs

- A valid IK fallback executes a rate-bounded prefix and returns `partial` instead of being rejected because the complete joint move exceeds one slot.
- Pose-only refinement removes posture-regularization bias when the first solve misses pose tolerance; joint limits remain unchanged.
- Base translation compensates the installed per-axis friction, preserving the public current-heading direction. Chassis hold uses the actual legacy controller transform and enough authority to overcome the translation dead zone.
- H3 receives the original task, state, action history, proposals, and previews. Previews model the bounded slot prefix, propagate the offset chassis yaw pivot, and replace current image slots to preserve history within the eight-image limit.
- H4/H4c shorten arm/hold slots only when the gripper command is unchanged. Base motion retains 16 drive plus 4 brake steps; gripper changes retain 20 steps.
- Action logs record pre-state, observation sequence, and requested/selected/executed duration. Ready initialization counts all its slots and elapsed time within the episode budgets.
- Recovery history loads the selected pre-action frame. H6 retains failure pre-state and requires evidence for a correction. H7 requires an inspected object-following lift. H8 distinguishes measured numerical agreement from unknown holding effects.
- Recovery candidates enter RSR only after an independently supplied original-task success within the recovery budget; an empty qualified denominator is N/A.

## Verification

The 42 focused unittest regressions passed locally and on h200-4 using the existing RoboCasa Python. Nine existing semantic tests passed locally. No lint or type-check command is configured for this repository. Three bounded simplification reviews (reuse, quality, efficiency) found no useful additional simplifications.

Physical simulator probes, without Qwen/SAM calls:

| Probe | Measured result |
|---|---|
| Recorded CounterToDrawer s100 decision21 | Executed20steps, partial; TCP moved0.121m; maximum commanded joint increment0.08004rad (trace rounding) |
| Recorded decision26 | Executed20steps, partial; TCP moved0.027m; refined IK meets the3mm pose tolerance |
| Base translation at original heading | About0.160m along intended direction; about1.2mm cross-direction drift |
| Base translation after26.5deg turn | About0.157m along intended direction; at most2.9mm cross-direction drift |
| Base translation after53.1deg turn | About0.156m along intended direction; about1.0mm cross-direction drift |
| Chassis during the saved arm probes | Final XY hold error below3mm per axis |

Raw successful probes: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/probes/motion-02/`. `motion-01` retains the earlier failed sign hypothesis. The original audit's rotation-sign allegation was withdrawn: RoboCasa installs `JOINT_VELOCITY_LEGACY`, which already uses the correct public rotation sign. The supported repair is friction compensation and corrected hold mapping.

Actual historical-data checks also passed: the five recovery starts resolve preceding sequences12,15,22,10,7 with images loaded; the old spoon “open to re-grasp” candidate produces no H7 skill under object inspection; zero-step H6 failures retain their original pre-action TCP.

The campaign runner produces27/27/18 unique core runs, routes ready initialization correctly, and preserves900steps/1200seconds/180decisions for every condition. Independent review raised three additional defects, all repaired with fail-before/pass-after regressions: unscored terminal results are excluded from completed metrics; interrupted inspections retain their old work directory and restart fresh; verified H6 corrections retain the enabling action sequence within the existing retrieval budget. The independent Claude pass also identified the fallback prompt mismatch, restored controller frame, completion timing, and nonlinear yaw preview. The fallback remains bounded by joint limits and per-step rates; no observed oscillation claim is made. Restore-frame drift was reproduced at0.111m sideways before repair. The clock repair records final-state time separately from later model/cleanup cost and adds no budget tolerance. The yaw curve uses actual measurements, including nonzero motion at velocity0.1. The final physical follow-up and review receipt are recorded separately.

## Limits of this evidence

A reporting follow-up counts invalid actions whose event rows omit `steps` and logs/counts transport-failed request attempts with unknown token usage. Three new regressions reproduce those failures and preserve successful-call accounting; all47 repair tests pass locally and remotely. Focused review found no actionable issue. See `accounting-repair.md`. Prompts, request payloads, and physics are unchanged; earlier comparisons derive rejection totals from event logs.

Stage3 subsequently exposed a joint-guide contract mismatch: advertised joint endpoints could be rejected by the decoder. The two joint guides now advertise accepted endpoints and qualify their downward-orientation formula with zero shoulder/forearm roll. The endpoint regression failed18 subcases before the repair and passes all28 afterward. All44 repair tests pass locally and remotely. See `joint-guide-repair.md`. EE prompts and the shared executor are unchanged, so Stage2 remains valid.

The first Stage 2 attempts subsequently exposed a reset-transient force baseline. The focused follow-up replaces it with robot-derived unloaded weight and marks initial sensor feedback unavailable until stepping. All43 repair regressions pass locally and remotely. See `force-repair.md` and `force-probes.json` for the physical measurements and separately retained invalid attempts.

These are execution-contract and evidence-alignment checks. Partial motion does not establish successful approach, grasp, or task completion. Two recorded yaw turns also agree with the repaired H3 preview: base-position errors were 0.05 and 0.34 mm; TCP errors were 0.51 and 0.68 mm. The H3 model is an ideal kinematic prefix with measured free-space base calibration; contacts, tracking pauses, and perception errors remain for the real rollout to reveal.

Final follow-up physics passed: restored-heading translation drift fell from0.111m to0.000995m; the restored arm hold remained within1cm per axis. Across signed yaw velocities0.05,0.1,0.15,0.2,0.3,0.4, maximum preview error was0.000124rad. The0.05/0.2 points were interpolation checks, not calibration inputs. Raw follow-up: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/probes/review-after/`. All42 merged repair tests passed locally and remotely.
