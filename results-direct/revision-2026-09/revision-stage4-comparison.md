# Stage4: candidate previews and action timing

**The 72-episode core campaign is complete, with zero full-task successes. H3 produced two verified lifts on the nine development starts, compared with one for clean, and used about twice the model calls. H4 executed short slots correctly but produced no grasp or lift. Continue the independent harness screens; H3 is a possible local-progress validation candidate.**

All 18 Stage4 episodes used frozen revision `e34dbf7`, original posture, the same Qwen/SAM setting, tasks and seeds 0–2, and budgets of 900 simulator steps, 1,200 seconds, and 180 decisions. The Stage2 clean baseline remains eligible: subsequent changes repaired joint-only instructions, accounting, and unused H7 text without changing its EE execution or policy inputs. H3/H4 were not combined with other harnesses.

## Outcomes

Counts are episodes satisfying each evaluator-side milestone.

| condition | n | task success | approach ≤10 cm | target contact | closed-gripper contact | lift ≥3 cm |
|---|---:|---:|---:|---:|---:|---:|
| EE-short clean | 9 | 0 | 2 | 2 | 1 | 1 |
| H3 propose/preview/select | 9 | 0 | 4 | 2 | 2 | 2 |
| H4 variable timing | 9 | 0 | 1 | 1 | 0 | 0 |

H3's lifts occurred on different starts from clean's lift: it gained lifts on Stove seed 1 and Drawer seed 2, but lost the clean lemon lift on Stove seed 2. This is two paired local gains and one loss, with no task-completion gain.

- **H3 Stove seed 1:** the tomato rose 16.9 cm while retained. The final target remained near the gripper, but transport and placement were unfinished.
- **H3 Drawer seed 2:** the pizza cutter followed a 12.0 cm lift. It was grasped and lifted again later; at decision 42 Qwen opened the gripper before establishing drawer placement. The next decisions returned to searching and re-grasping, then stopped.
- **H3 Drawer seed 1:** the TCP approached within 4.4 cm without target contact. H3 Stove seed 0 approached within 8.8 cm without contact.
- **H4:** Sink seed 2 contacted and displaced the target without grasping it. Stove seed 1 approached within 9.8 cm without contact. All other starts remained farther away.

False grasp beliefs persist. H3 Sink seed 1 described holding the pepper while the inspected target remained away from the gripper. H4 Drawer seed 1 described holding the spoon with a 1.9 mm finger gap; the episode recorded no target contact. Numerical reachability and a closed command do not resolve object identity or retention.

## Did the harness mechanisms work?

H3 made 413 proposal calls and 413 selection calls. Its previews evaluated 1,230 candidates: 1,156 were kinematically reachable and 74 unresolved. The 405 parsed selections chose A 314 times, B 40, C 27, and a revision 24. Thus 91 selections departed from candidate A. Executed actions included 33 bounded joint-space fallback prefixes; no executed decision was rejected as unreachable. The previews provide useful kinematic feedback, while their documented contact-free approximation cannot predict collisions with pans or furniture.

Eight selection calls hit the fixed 1,024-token limit and produced malformed output; every proposal call finished normally. A reviewed Drawer seed 1 response completed its reasoning string, then generated roughly 900 spaces until truncation. These eight zero-motion decisions remain scored model-output failures. The decoder, budget, and prompts stayed fixed throughout the comparison. A later constrained-output experiment can investigate this specific generation failure separately.

H4 executed 416 action slots: 294 at 20 steps, 117 at 10, and five at 5. All 12 actual gripper-command changes requested and executed 20 steps. Short slots therefore worked, including actions with an unchanged gripper command. However, the run set exposed only one distinct controller observation within 10 cm of the target, versus 14 for clean and 43 for H3. Its sole executed slot starting within that distance lasted 20 steps. The shorter slots did not create the intended extra feedback near the target in this screen.

## Cost

| condition | mean simulator steps | mean Qwen calls | rejected decisions | mean episode wall seconds | prompt tokens | completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| clean | 808.9 | 44.2 | 32 | 363.1 | 1,606,055 | 58,119 |
| H3 | 884.4 | 91.8 | 8 | 734.1 | 3,878,564 | 191,686 |
| H4 | 786.1 | 51.7 | 43 | 460.7 | 1,956,935 | 72,074 |

Rejection counts are totals from decision events, including malformed output. H3 used 4,309.1 seconds of Qwen time and 1,569.5 seconds of SAM time; H4 used 1,644.6 and 1,975.9 seconds respectively. These are aggregate costs under concurrent execution.

The full 72 scored core episodes used 45,275 simulator steps, 2,899 Qwen calls, 12,488,918 prompt tokens, 512,373 completion tokens, and 23,787.8 aggregate episode wall seconds (6.61 hours). Initialization is included. These totals exclude retained attempts and separately reported H7/recovery prerequisites.

One H4 Drawer seed 1 transport failure was retained unchanged and replaced by a scored same-scene retry. The retained attempt consumed 700 steps, 57 request attempts, 236,851 known prompt tokens, 9,077 known completion tokens, and 530.3 seconds. One request's token usage is unknown. An overlapping manual/batch offline inspection was also retained; its single affected measurement file was regenerated serially from unchanged snapshots. No additional policy rollout was needed for that inspection repair.

## Next decision

Proceed with H1 visual grounding, H5 working memory, H6 failure experience, and H8 explicit effects/recovery, nine matched development starts each. H7's independently verified applicability result also qualifies its separate nine-start screen. Keep the clean-derived banks frozen. After these screens, select at most two useful candidates for validation on reserved fresh seeds 20–24, alongside clean. H4 currently has no measured local benefit supporting validation; H3 has a concrete local-progress signal that must be weighed against its cost and the remaining methods.

Recovery qualification remains 0/3 task-completing continuations. The qualified bank is empty and RSR remains N/A. Ordinary H8 development episodes can proceed independently.

## Limitations and evidence

Nine development starts support diagnosis and candidate selection, not a general success-rate claim. Milestones use sampled observation states; contact can occur even when the TCP center is more than 10 cm from an object's center. H4 changes observation density. H3's larger near-target observation count primarily reflects its trajectories because its slots remain 20 steps. Skill qualification additionally checks object-following motion, beyond the closed-gripper-contact milestone.

- [Per-run outcomes](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage4-data/results.json).
- [Action, timing, selection, and cost analysis](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage4-data/analysis.json).
- [Fixed manifest](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/stage4-data/manifest.json).
- Raw runs and inspection states: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/stage4/`.
- Retained transport attempt: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/infrastructure-attempts/stage4/ee-h4/PickPlaceCounterToDrawer-s1-ee-short-h4-attempt1/`.
