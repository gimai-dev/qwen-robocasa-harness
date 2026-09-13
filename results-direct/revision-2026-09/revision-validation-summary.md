# Fresh validation: observed outcomes

Candidates were selected before evaluation on the reserved seeds20–24. All conditions use the same15 pinned starts and the frozen implementation and banks.

| Condition | Terminal success | 95% exact interval | Any sampled goal state | Approach | Contact | Closed contact | Lift milestone | Verified retained lift | Mean calls | Mean wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ee-clean |0/15|0.0–21.8%|0|3|4|3|2|2|45.6|412.9|
| ee-h7 |0/15|0.0–21.8%|0|8|6|5|2|2|46.2|413.9|
| ee-h8 |0/15|0.0–21.8%|0|4|3|2|2|1|43.4|424.6|

The lift milestone requires the target to rise at least3cm while closed-gripper contact is observed. Verified retained lift additionally requires at least one object-following fragment under the pre-existing skill extractor. H8 Drawer23 records a brief elevation during closing but no qualifying retained-lift fragment; both measurements are preserved.

Two-sided 95% Clopper–Pearson exact binomial intervals for each observed task-success rate; descriptive under a binomial model across these sampled starts. The paired comparison is reported as matched gains/losses, not inferred from overlapping rate intervals.

| Candidate | Task-success gains/losses vs clean | Lift gains/losses vs clean |
|---|---:|---:|
|ee-h7|0/0|1/1|
|ee-h8|0/0|1/1|

A zero observed difference is not evidence of equivalence. These15 starts per condition cover three tasks and provide limited precision; full-task success remains the primary outcome.
