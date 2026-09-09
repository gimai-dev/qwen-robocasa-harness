# Evaluation report: Qwen RoboCasa recovery portfolio

## Result

The completed, frozen ten-task cohort achieved **5/10 official successes (50%)**. The earlier complete ten-task baseline achieved **2/10 (20%)**, a difference of **30 percentage points**. Total evaluation time was **1,202.5 seconds (20.0 minutes)**, with **6,196 simulator actions and 132 Qwen calls**.

The cohort uses Qwen3.8-27B BF16, seed 7, one fresh episode per original task, and a maximum of 900 simulator actions per task. Controller code, arguments, and input memory were selected before the cohort started. This is a task-routed portfolio developed on these scenes.

All ten native result files contain Boolean official outcomes. All ten launches exited with code 0. An ordinary task failure is distinct from a process or infrastructure failure. The native results and summary agree on all ten outcomes.

## Per-task outcomes and recordings

The common `PickPlace` prefix is omitted from the display names. Follow **Evidence** for the corresponding native result and **Video** for the original MP4. The [public video index](https://github.com/gimai-dev/qwen-robocasa-harness/blob/main/videos/README.md) provides individual task anchors and direct MP4 playback links.

| Task | Outcome | Actions | Qwen calls | Final stage | Recording and evidence |
|---|---|---:|---:|---|---|
| CounterToDrawer | Success | 745 | 4 | leave | [Video](../videos/PickPlaceCounterToDrawer.mp4) · [Evidence](../results/fullten-r1/PickPlaceCounterToDrawer/result.json) |
| CounterToStandMixer | Success | 873 | 10 | leave | [Video](../videos/PickPlaceCounterToStandMixer.mp4) · [Evidence](../results/fullten-r1/PickPlaceCounterToStandMixer/result.json) |
| CounterToSink | Success | 398 | 12 | leave | [Video](../videos/PickPlaceCounterToSink.mp4) · [Evidence](../results/fullten-r1/PickPlaceCounterToSink/result.json) |
| StoveToCounter | Success | 775 | 12 | leave | [Video](../videos/PickPlaceStoveToCounter.mp4) · [Evidence](../results/fullten-r1/PickPlaceStoveToCounter/result.json) |
| SinkToCounter | Success | 642 | 12 | leave | [Video](../videos/PickPlaceSinkToCounter.mp4) · [Evidence](../results/fullten-r1/PickPlaceSinkToCounter/result.json) |
| DrawerToCounter | Failure | 458 | 5 | verify | [Video](../videos/PickPlaceDrawerToCounter.mp4) · [Evidence](../results/fullten-r1/PickPlaceDrawerToCounter/result.json) |
| CabinetToCounter | Failure | 336 | 9 | retreat | [Video](../videos/PickPlaceCabinetToCounter.mp4) · [Evidence](../results/fullten-r1/PickPlaceCabinetToCounter/result.json) |
| CounterToCabinet | Failure | 698 | 30 | locate_place | [Video](../videos/PickPlaceCounterToCabinet.mp4) · [Evidence](../results/fullten-r1/PickPlaceCounterToCabinet/result.json) |
| CounterToMicrowave | Failure | 423 | 18 | ground | [Video](../videos/PickPlaceCounterToMicrowave.mp4) · [Evidence](../results/fullten-r1/PickPlaceCounterToMicrowave/result.json) |
| MicrowaveToCounter | Failure | 848 | 20 | close | [Video](../videos/PickPlaceMicrowaveToCounter.mp4) · [Evidence](../results/fullten-r1/PickPlaceMicrowaveToCounter/result.json) |

## Successful routes

### Counter to Drawer

Restoring the original complete controller recipe recovered this task. The route uses the bowl candidate, SAM2, direct approach, and a 0.22 m release-clearance parameter. The baseline reproduction and the frozen cohort both finished in 745 actions. The official predicate confirmed success after placement and withdrawal.

### Counter to Stand Mixer

This route depends on a full set of side-entry and transport parameters. Restoring only part of that configuration had caused a regression during development. The retained recipe includes a 30-degree downward grasp, 0.74 m approach height, 0.25 m side offset, 0.07 m side clearance, and 0.3 m transfer/release-clearance parameters. Reproduction and the cohort both succeeded in 873 actions, leaving 27 actions below the limit.

### Counter to Sink

The retained controller uses the visible drain geometry to estimate the sink receiver and releases from the side nearer the robot. This addressed the destination-localization and placement behavior encountered during development. Reproduction and the cohort both succeeded in 398 actions.

### Stove to Counter

The retained approach combines 0.15 m base backoff with a lower-then-turn entry and transport that preserves the gripping orientation. Reproduction and the cohort both succeeded in 775 actions.

### Sink to Counter

The retained route addresses palm contact, clearance, and receiver re-localization together. It applies a +0.02 m grasp-height offset, raises vertically above the observed receiver before lateral/base movement, and preserves the stationary plate's world-coordinate anchor across measured base motion. Development, independent reproduction, and the cohort each succeeded in 642 actions.

A subsequent review added empty-gripper checks to the receiver-raise and receiver-base stages. That successor independently reran successfully in 642 actions. It is included separately under `controllers/reviewed/SinkToCounter/`; it does not replace the code associated with the published cohort or video.

## Failed routes

### Drawer to Counter: source not held after lifting

The controller stopped at `verify` after 458 actions with `task source not held after lift`. The whisk grasp did not produce a verified held object. Development explored handle selection, offsets, depth, vertical entry, front entry, base backoff, and alternative grasp orientation. Some attempts progressed farther through the sequence, but they remained official failures.

**Next experiment:** test an image-grounded grasp on the thick handle section with measured closure and lift displacement, before spending actions on transport. This is a proposed experiment, not a validated improvement.

### Cabinet to Counter: contact blocks the approach

The controller stopped at `retreat` after 336 actions with `contact blocked; reversing executed waypoints`. The attempted front-entry route did not complete a usable approach.

**Next experiment:** compare a small set of collision-aware pregrasp orientations and approach corridors for the cabinet opening, retaining the same grasp-verification requirement.

### Counter to Cabinet: receiver remains unresolved

The controller stopped at `locate_place` after 698 actions with `receiver unresolved after lateral views`. It used 30 Qwen calls, the highest count in this cohort. Additional lateral observations did not produce an accepted receiver location.

**Next experiment:** estimate the visible receiving shelf before pickup and retain it through later occlusion; measure receiver-localization success separately from grasp success.

### Counter to Microwave: RGB grounding remains unresolved

The controller stopped at `ground` after 423 actions with `RGB grounding unresolved after base and wrist views`. The source-grounding pipeline did not produce an accepted target for subsequent execution.

**Next experiment:** separate semantic source identification from finding a locally graspable portion of the object, and evaluate those two stages on the saved multi-view observations.

### Microwave to Counter: withdrawal is unreachable

The controller stopped at `close` after 848 actions with `unreachable_front_withdraw`. The attempted front-grasp route could not resolve the required withdrawal plan. This stopping stage does not prove that the object was securely held.

**Next experiment:** require a feasible approach-and-withdrawal pair before committing to closure, and compare grasp orientations compatible with the opening.

## Development beyond the fifth success

The saved development ledger contains **87 native episodes and 4 probes**. Probes are marked separately and are not counted as task-success episodes. These development attempts are also separate from the final ten-task cohort.

After the fifth successful route was found, nine further development attempts completed without adding a sixth officially successful task:

| Attempt | Main variation | Outcome |
|---|---|---|
| `whisk-vertical` | Vertical whisk entry | Failure |
| `whisk-inset` | Grasp moved inward along the handle | Failure |
| `whisk-inset-depth` | Inset grasp with changed depth | Failure |
| `bun-vertical-pinch` | Vertical-jaw side pinch | Failure |
| `whisk-surface` | Surface localization from calibrated visual correspondences | Failure |
| `whisk-front60-backoff30` | Steeper front entry with larger base backoff | Failure |
| `bun-vertical-pinch30` | Angled variant of the vertical-jaw pinch | Failure |
| `whisk-wood-center` | Target the thicker handle region | Failure |
| `microwave-front-source` | Front-source route for microwave extraction | Failure |

Offline IK, contact, and geometry probes helped diagnose these failures. They are not official task successes. The [development ledger](../results/experiments.json) preserves the measured run-level outcomes; the complete source and videos in this release are centered on the final cohort and reviewed successor rather than every development branch.

## Memory and regression control

The final bank contains five success skills and 111 lessons. The lesson ledger combines 14 earlier entries, 87 development episodes, and ten cohort episodes. Complete recipe restoration preserves controller code, all arguments, and the original input memory. Fallback retrieval selects at most four recent same-task lessons using actual completion timestamps.

Review during development found and fixed two concrete issues: lesson retrieval ordered by run name instead of completion time, and missing empty-gripper checks in newly added carrying stages. The Mixer recipe-persistence test, lesson-ordering test, and carrying-stage checks passed during development. The carrying change was rerun natively on a separate successor snapshot.

## How to interpret the result

The 50% figure is the fraction of the original ten tasks whose **fresh cohort episode** satisfied the native terminal predicate. It is not the best fraction among arbitrarily selected individual development runs. All five unsuccessful task episodes remain in the denominator.

The result supports the practical value of preserving complete successful recipes and fixing task-specific perception/control failures. It does not isolate the causal contribution of memory, Qwen prompting, SAM2, or individual geometry changes; that would require matched ablations.

## Limitations

The recipes were developed on the same seed-7 scenes used for this cohort. There are no reported unseen-seed or unseen-task results, repeated-cohort uncertainty estimates, or unified-controller comparisons. Development included offline simulator contact and object-geometry diagnosis, while runtime uses RGB, calibration, and public robot state. The base-action cap increased from 0.25 to 0.5 within the official range. The five successes therefore establish this portfolio's observed fixed-scene performance rather than general robotic competence or a prompt-only improvement.

The videos are 4 fps montages of saved left/right/wrist observations. They cover all saved observations, not every simulator frame, and do not preserve episode wall-clock timing. Raw result and controller-evidence JSON provide the action counts and stopping reasons.

## Evidence index

- [Completed-cohort summary](../results/fullten-r1/summary.json)
- [Frozen cohort manifest](../results/fullten-r1/manifest.json)
- [All per-task native results and controller evidence](../results/fullten-r1/)
- [Earlier complete ten-task baseline](../results/baseline-ten/cohort.json)
- [Two successful-route restoration checks](../results/baseline-summary.json)
- [Development ledger](../results/experiments.json)
- [Reviewed Sink-to-Counter native result](../results/garlic-reviewed-result.json)
- [Final learned memory](../memory/learned-memory-final.json)
- [Registered reproduced recipes](../workflow/validated-routes.json)
- [Recording metadata](../videos/metadata.json)
