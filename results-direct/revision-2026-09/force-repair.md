# Wrist-feedback follow-up

The first five Stage 2 attempts at revision `0ce14c7` exposed a shared feedback defect. Reset stored a force magnitude of 218.219 N as the baseline; the settled unloaded sensor reads about 5.1012 N. Subtracting the reset transient suppressed subsequent wrist-load feedback. These attempts are retained separately under `h200-4:/home/jli/state/qwen-direct/revision-2026-09/stage2-force-baseline-attempts/` and are excluded from the restarted comparison. All five completed without official task success.

The repair derives the unloaded weight from the robot bodies distal to its wrist force sensor: 0.52 kg × 9.81 m/s². Both published load and slot peaks use `max(0, force magnitude - unloaded weight)`. The initial reset/restore value is null until physics advances. This adds no settling steps and preserves the pinned physical start. No object information enters calibration or the policy.

The simulator probe records reset, an initial hold, a clear ready pose, downward fixture contact, and retreat. The initial hold still carries 11.15 N of excess load; it is not a valid unloaded reference. The first probe incorrectly assumed that hold was unloaded and failed that assertion. Its raw observations remain in `probes/force-after/`. The corrected probe checks the clear ready pose and post-retreat hold instead; the calibration itself was unchanged.

Observed clear-pose excess load was 0.02 N, contact peaks reached 229.93 N, and the final post-retreat load was 0.00 N. These values verify the response of the load proxy. Contact, acceleration, and carried weight can all affect it; the prompt makes that interpretation explicit.

A fresh focused reviewer found no actionable critical defect in the calibration, initial-null consumers, or associated unit test. The complete 43-test repair suite passed locally and on h200-4. The physical confirmation is retained in `probes/force-confirmation/`; its compact measurements accompany this report.
