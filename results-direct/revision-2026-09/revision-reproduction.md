# Revision campaign locations and commands

Source checkout: `/Users/jiachen/Desktop/first try/qwen-direct-control`, branch`direct-control`. The final policy revision is`1223102`. All seven repair commits follow the audited`df164fe` revision. Reports and presentation scripts are in `/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/` and`work/`.

The active frozen runtime is `h200-4:/home/jli/work/qwen-direct-control-revision-r7`. Original remote checkouts at `/home/jli/work/qwen-direct-control` and `/home/jli/work/qwen-direct-control-dev` remain unchanged. The revised raw result root is `/home/jli/state/qwen-direct/revision-2026-09/`; the original campaign remains under `/home/jli/state/qwen-direct/` outside that root.

| Scored batch | Episodes | Recorded code revisions |
|---|---:|---|
| Stage2 |27|`88243a1`|
| Stage3 |27|2 EE-full runs at`88243a1`;20 unaffected runs at`ef3e956`;5 affected joint-short replacements at`e34dbf7`|
| Stage4 |18|`e34dbf7`|
| H1/H5/H6/H8 screen |36|33 unaffected runs at`e34dbf7`;3 H1 gallery replacements at`1223102`|
| H7 screen |9|`1223102`|
| Fresh validation |45|`1223102`|

Later revisions only change paths that the retained earlier runs did not execute. Shared physical execution after the force calibration remains common. Individual `config.json` and `result.json` files retain their recorded revisions. The staged reports explain each selective replacement.

The declared service model is `qwen3.8-27b-bf16-51f2ba0db1760338e6790c90ac1feab9`, at the existing local port8002 service. Decoding remains seed3074294, temperature0, top_p1, thinking disabled, and strict JSON output. Short calls allow1024 output tokens; full calls allow4096. The simulator environment is `/home/jli/work/robocasa-inspect-official/.venv/bin/python`. SAM and camera settings remain those of the original numerical-control pipeline.

The exact H7/validation launchers and analysis helpers are permanently archived at `/home/jli/state/qwen-direct/revision-2026-09/protocols/`. From an SSH session on h200-4, use the frozen checkout as the working directory:

```bash
cd /home/jli/work/qwen-direct-control-revision-r7
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:runtime /home/jli/work/robocasa-inspect-official/.venv/bin/python /home/jli/state/qwen-direct/revision-2026-09/protocols/run_revision_validation.py h7 h8
```

The validation runner completed all45 episodes and inspections with exit code0. The command above reuses the completed records. For an interrupted replication, the launcher reuses completed results with the identical manifest; failed infrastructure attempts must be retained separately before their same starts are retried. It creates each fresh scene once, serially, before the three policy workers. All15 validation scenes were created with zero simulator actions and zero Qwen calls, taking322.34seconds. The batch runner owns the offline inspections after policy episodes finish.

For the completed H7 screen, the corresponding archived entry point is `protocols/run_revision_h7_screen.py`. The core and independent screen use `python -m direct.revision_campaign --batch stage2|stage3|stage4|screen --parallel 3`. These commands reuse their existing manifests and cached completed results.

For final analysis, use the archived `analyze_revision_batch.py` with a batch directory and `--out` destination, then `diagnose_revision_lifts.py` with that batch directory. Both read existing logs and inspection files; they do not execute policy actions. `collect_revision_costs.py` regenerates the root `cost-ledger.json` from episode records, including superseded and infrastructure attempts as separate categories.

Local presentation scripts are `work/render_revision_development.py` and `work/summarize_revision_validation.py`. Run them with `work/revision-env/bin/python` after the batch data have been copied into their corresponding `outputs/*-data/` directories. The local environment includes Matplotlib3.11.2 for the PNG/SVG development chart; this installation does not alter the robot runtime.

Terminal success comes from the episode's official terminal result. The supplemental sampled-goal metric is `milestones.success`, which tests whether any saved inspected state meets the official predicate. The H7 lemon episode differs on these two metrics; the report preserves that distinction. Simulator object truth appears only in offline evaluation and bank qualification, not in current policy observations.

The clean-derived banks remain frozen at `banks/h6-failures.json` and `banks/h7-skills.json`. H7 contains two overlapping local lift examples from one clean lemon episode. The recovery qualification bank is empty after three unsuccessful independent clean continuations, so RSR isN/A and no qualified-state clean/H8 recovery comparison is run.

There are18 retained superseded episodes and2 retained infrastructure attempts. Fresh validation required no retries. Their raw files and embedded original paths are preserved, with directory mappings in adjacent retention notes. One old transport failure has one additional unlogged request attempt; its usage is unknown. A later transport failure is logged and counted, also with unknown usage. Cost totals report known tokens and summed episode wall time; offline inspection/probe time and the separately reported scene setup are outside those episode totals.
