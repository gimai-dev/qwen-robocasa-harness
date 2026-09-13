# H7 and recovery prerequisites

H7 passed the planned independent local-skill check. On StoveToCounter seed3, Qwen used the clean seed2 skill context and produced a verified object-following apple lift. Offline inspection measured a maximum height gain of0.247m, retained object contact, and three qualifying local fragments. Complete-task success was false. This supports proceeding with the nine-start H7 screen; it is not yet a controlled improvement over clean on seed3.

The source bank stays frozen: two overlapping fragments from one clean lemon episode, with absolute source-scene poses and gripper timing. The H7 prompt was corrected to describe those coordinates accurately. Qwen generated fresh apple-scene targets. The applicability episode is development evidence, outside the matched nine-start comparison, and contributes no new entries to the clean source bank.

The H7 episode consumed880steps,50Qwen calls,260571prompt tokens,7343completion tokens, and318.2wall seconds. It ended with the apple still touching the closed gripper, about1.96cm from the TCP. Qwen's last response nevertheless said the apple was still in the pan and ungrasped, then stopped with20steps left. The local skill transferred, while the task-state belief and remaining transport/placement did not stay reliable.

Recovery qualification produced0/3 original-task completions. All attempts used corrected pre-action visual history and the separate400-step/600-second/80-decision budget. They were independent clean continuations, excluded from any clean/H8 recovery comparison.

| Candidate source | Failure state | Qualification result | Steps | Qwen calls | Wall s |
|---|---|---|---:|---:|---:|
| Drawer seed1, sequence22 | Empty close near target | Step budget; task false |400|22|180.1|
| Sink seed2, sequence9 | Contact lost without grasp | False completion claim; stop |300|17|155.2|
| Stove seed2, sequence28 | Lost target contact during placement after verified lift | Release then stop; task false |20|2|10.9|

The first two candidates came from the existing selector. The late placement failure at source step560 was included explicitly because recovery has its own budget; the inherited automatic selector caps source steps at500. Only three distinct clean-source candidates were used. Total qualification cost was720steps,41calls,170097prompt tokens,6119completion tokens, and346.2wall seconds.

The qualified bank is empty, so RSR isN/A and the ten-continuation recovery comparison is deferred. Failed clean qualification does not prove these physical states are unrecoverable. H8's ordinary nine-start development screen still proceeds independently.

Evidence: [H7 applicability](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/prerequisite-data/applicability.json), [all prerequisite results](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/prerequisite-data/attempts.json). Remote snapshots and inspections are under `h200-4:/home/jli/state/qwen-direct/revision-2026-09/h7-applicability/` and `recovery-prerequisites/`.
