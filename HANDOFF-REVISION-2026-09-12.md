# Handoff: repaired direct numerical Qwen campaign

The accepted repair and selective-rerun campaign is complete. **117 development episodes and 45 fresh validation episodes have been scored and inspected.** All final scored episodes fail the terminal task predicate. One development H7 episode briefly reaches an official goal state, then undoes it. None of the fresh validation observations reaches the goal.

Read the [campaign report](results-direct/revision-2026-09/revision-campaign-report.md), [fresh comparison](results-direct/revision-2026-09/revision-validation-comparison.md), and [proposed next experiments](results-direct/revision-2026-09/revision-next-experiments.md). The next plan is a proposal; no additional harness or confirmation batch is running.

## Results and decision

| Fresh condition | Terminal task success | Approach | Target contact | Closed contact | Lift milestone | Verified retained lift |
|---|---:|---:|---:|---:|---:|---:|
| Clean |0/15|3|4|3|2|2|
| H7 skill context |0/15|8|6|5|2|2|
| H8 expected effects |0/15|4|3|2|2|1|

The three conditions use the same 15 starts: CounterToSink, CounterToDrawer, and StoveToCounter, seeds 20–24. Candidate selection preceded scene setup and outcomes. Validation had no infrastructure failures or retries. Each observed0/15 terminal rate has a two-sided 95% exact binomial interval of 0–21.8%; equal observed counts do not establish equivalence.

H7 adds five approaches with no paired losses, and three contacts with one loss. Its verified lift count stays equal to clean, with one gained and one lost start. H8's development lift advantage does not reproduce. Its additional Drawer23 lift milestone is brief elevation during closing and has no verified object-following lift fragment.

Keep the repairs and H7's contact result. Prioritize visual evidence about retained grasps and placement, then target-directed SAM perception. The repeated failures are concrete: Qwen opens retained grasps because it thinks they failed, sometimes transports toward the wrong destination, and sometimes stops with a false completion belief. The shared prompt already states release and greater-than 25 cm clearance requirements. The proposed next work keeps Qwen as the numeric action author and begins with saved-image diagnostics, followed by at most two nine-run development conditions. The optional 120–180-episode confirmation and semantic comparison were not launched.

## Code and verification

Policy revision: **`1223102`**, after seven local repair commits from the audited `df164fe` revision. A later documentation commit records this handoff and the final results; it does not change the frozen policy.

- `0ce14c7`: bounded valid-IK motion, pose refinement, base friction/hold/restoration, H3/H4 contracts, correct history, evidence qualification, and campaign accounting boundaries.
- `88243a1`: unloaded wrist-force calibration, replacing the reset transient.
- `ef3e956`: accepted joint-guide endpoints and the orientation formula's assumptions.
- `936a5e5`: invalid-action and failed-request accounting.
- `c8c7288`: correct description of H7 source-scene absolute poses.
- `e34dbf7`: inward-rounded joint error-message bounds.
- `1223102`: display all regions supplied to the H1 gallery.

All49 focused repair tests passed locally and remotely. Physical IK, base-heading, restored-frame, force, yaw-preview, historical-image, and false-skill probes passed. Receipts and numerical results are in `results-direct/revision-2026-09/`. Original FK, quaternion order, TCP conversion, and gripper polarity were correct; the initial rotation-sign allegation was withdrawn after a live probe of the selected legacy base controller.

## Frozen setting and provenance

The declared service remains `qwen3.8-27b-bf16-51f2ba0db1760338e6790c90ac1feab9` on the existing port8002 endpoint: frozen weights, seed 3074294, temperature0, top_p1, thinking disabled, strict JSON. Short/full output limits remain 1024/4096 tokens. RoboCasa, PandaOmron, SAM2.1 small, cameras, and public robot feedback remain the numerical-control setting. Main episode limits are 900 steps,1,200 seconds,180decisions. Ordinary short actions execute20-step slots; H4 is the separate timing condition.

Policy inputs exclude simulator object truth. Offline evaluator states label milestones and qualify bank records. H7 receives two overlapping local grasp/lift examples from one clean development lemon episode, supplies them as context, and generates fresh numeric targets. It does not replay actions automatically or execute task recipes. An independent apple episode demonstrated local applicability but failed the complete task. No later campaign data were added to the bank.

Three independent clean continuations attempted recovery qualification. None completed its original task within 400 steps/600 seconds/80decisions. The qualified bank is empty, RSR is N/A, and no unqualified clean/H8 recovery comparison was run. A failed clean qualification does not establish physical unrecoverability; the next plan permits separately identified expert qualification while keeping the evaluated policies fixed.

Development results retain earlier code revisions only when later changes did not affect their executed paths. The [reproduction guide](results-direct/revision-2026-09/revision-reproduction.md) gives the per-batch revision table, commands, and source paths. The intermediate-goal diagnostic was added after candidate selection and remains exploratory; it did not change terminal scoring or policy actions.

## Episode accounting

There are 162 final scored evaluations,18 retained superseded episodes,2 retained infrastructure failures,1 H7 applicability episode, and 3 recovery qualification episodes:186 attempts. The only success in that all-attempt ledger is the superseded H1 Sink0 run with the incomplete gallery; its raw evidence remains preserved.

Scored comparisons used 6,769 Qwen calls and 60,273.5 summed episode-wall seconds. Across all attempts there are 7,687 request attempts,34,098,025 known prompt tokens,1,381,315 known completion tokens, and 67,407.2 summed episode-wall seconds. Two retained failed requests have unknown token usage; one also predates request logging and is explicitly added to the count. Fifteen zero-action scene setups took322.34 seconds separately. Offline inspection and verification time are outside the episode totals. See the [full cost ledger](results-direct/revision-2026-09/revision-cost-ledger.json).

## Locations and current state

- Local source: `/Users/jiachen/Desktop/first try/qwen-direct-control`, branch `direct-control`.
- Frozen policy runtime: `h200-4:/home/jli/work/qwen-direct-control-revision-r7`, at `1223102`.
- Revised raw runs, banks, inspections, and protocols: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/`.
- Exact launch and analysis scripts: the revised root's `protocols/` directory.
- Final report archive: the revised root's `reports/` directory.
- Detailed local data and presentation scripts: `/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/` and `work/`.
- Historical source checkouts remain at `/home/jli/work/qwen-direct-control` and `/home/jli/work/qwen-direct-control-dev`; original raw results remain under `/home/jli/state/qwen-direct/` outside the revision root.

The final validation runner exited successfully after all 45 inspections. No required campaign batch remains pending. Develop a proposed new treatment from the frozen policy in a separate checkout, preserve the recordings, and reserve unused scenes before its future validation. Do not resume the completed runner expecting it to create new evidence: its commands reuse the completed manifest and cached records.
