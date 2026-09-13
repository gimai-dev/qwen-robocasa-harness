# Independent H1/H5/H6/H8 screen

All36 repaired screen episodes and their simulator inspections are complete. Every harness scored0/9 full-task success. H8 has the clearest local-progress signal: three lift episodes versus clean's one, while aggregate episode runtime is about9% higher. H1 and H6 each have two lift episodes. The108 scored episodes completed so far, including the72-episode core, have no full-task success. H7's eligible nine-start screen follows before fresh validation selection.

| Condition | Episodes | Task success | Approach ≤10cm | Contact | Closed contact | Lift ≥3cm | Mean steps | Mean Qwen calls | Mean wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EE-clean, Stage2 |9|0|2|2|1|1|808.9|44.2|363.1|
| H1 visual markers/crops |9|0|3|3|2|2|826.7|48.6|423.4|
| H5 working memory |9|0|2|3|2|1|751.1|38.3|500.2|
| H6 failure experience |9|0|4|4|2|2|720.0|39.2|351.1|
| H8 expected effects/recovery |9|0|5|3|3|3|766.7|40.7|396.3|

Each uses the same three tasks, seeds0–2, original posture, frozen Qwen/SAM setting,900-step/1200-second/180-decision limits, and repaired shared executor. No task-specific movement routine was added. All eight lift episodes across these four harnesses also have retained object-following lift evidence under the stricter skill criterion. Their fragments were not added to the frozen clean-derived banks.

## H1 repair and outcome

The original H1 screen scored1/9 full-task success, on Sink0. Code inspection then found a real omission: the gallery promised a crop of every supplied region but truncated at12, while the perception layer supplied up to14. It occurred on44 decisions across Sink0(5), Stove1(26), and Stove2(13). The renderer now includes the whole supplied list. A regression with an object only in region14 failed before the repair and passes afterward; all49 repair tests passed locally and remotely. Real14-region observations were rendered and checked for legibility.

Those three attempts, including the successful one, their logs, cached inspections, and the original36-run aggregate files are preserved under `screen-h1-gallery-attempts`. The three identical pinned starts were rerun at`1223102`. The other six H1 runs and all27 other-harness runs are reused at`e34dbf7`, because they did not execute a changed path. The repaired H1 comparison is0/9. This is not evidence that the additional crops caused success to fall: each start was run once per affected implementation, and observations and subsequent decisions diverged.

Repaired Sink0 never approached within10cm; its minimum recorded target distance was0.575m. Stove1 lifted the tomato up to0.291m and retained it from observation15 through the terminal observation45, but ran out of budget during transport. Stove2 lifted the lemon0.139m. Decision12 explicitly opened the gripper during the movement toward the bowl, losing the object; the remaining episode failed to finish. Sink2 contacted the target and stopped with a false completion claim. None of the Drawer starts recorded target contact.

The initial Drawer0 whisk is visibly present in RGB, but no dedicated whisk mask exists among the21 retained right-view SAM proposals. The six-entry unpaired cap did not remove a dedicated whisk proposal in that observation. A large counter mask is also paired with a small wrist fragment, which the crop gallery prefers. Raw rejected SAM proposals were not saved, so generation versus filtering is unresolved. The gallery omission is repaired; this separate perception limitation remains and was not changed midway through the campaign.

## What the other harnesses actually did

H5 wrote339 working-memory updates. Its sole lift episode, Stove1, repeatedly treated a closed-on-tomato gap as a failed grasp, reopened after a retained lift, and eventually ended holding the tomato with placement unfinished. Sink1's memory treated the pepper as already in the sink. Of its six stops, two falsely claimed completion and four correctly acknowledged an unfinished task near the budget limit. Persistent memory alone did not make the stored beliefs reliable.

H6 delivered two records on all353 calls. Its706 exposures included103 verified corrections, all on Drawer0/2; those starts still failed. Six starts received one unchanged retrieval set throughout the episode. Sink0 received the same two base-blocked records even during later empty-grasp attempts. The current task/phase/word-overlap retrieval is functioning, but does not adapt well to those different failures. Stove0 ended holding a lifted sweet potato; Stove2 reopened after a0.099m retained lemon lift because Qwen believed it had never lifted it. Its bank contains no task-level false-completion examples.

H8 made345 measured effect checks:233 TCP/gap mismatches and112 matches. All345 holding checks correctly remained unknown; no simulator object truth was exposed. It sometimes diagnosed empty closes and chose a retry, but also abandoned retained grasps. Sink0 lifted an orange0.157m, lost it during lateral transport, then failed to recover. Stove1 lifted a tomato0.163m and ended holding it. Stove2 lifted a lemon0.094m and later reopened after a false failed-grasp diagnosis. H8 retains clean's Stove2 lift and adds Sink0/Stove1; it establishes local progress on these starts, not task completion or recovery success.

The observation already supplies `gripper_command` and `gripper_width_m`. Several failures reflect Qwen's interpretation of those fields and temporal images. The feedback is not missing simply because Qwen's prose contradicts it. Detailed event evidence is in [mechanism notes](/Users/jiachen/Desktop/first try/qwen-direct-control/results-direct/revision-2026-09/revision-screen-mechanism-notes.md).

## Cost and provenance

| Condition | Steps | Calls | Rejections | Prompt tokens | Completion tokens | Qwen s | SAM s | Aggregate wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 |7440|437|64|1869692|64726|1519.7|1707.1|3810.2|
| H5 |6760|345|1|1531678|130028|2906.6|1139.3|4502.1|
| H6 |6480|353|27|1916345|52946|1250.7|1439.5|3160.3|
| H8 |6900|366|19|1630888|81946|1791.2|1283.5|3566.7|

The36 scored episodes used27580steps,1501Qwen calls,6948603prompt tokens,329646completion tokens, and15039.3aggregate episode wall seconds. All1501 control responses finished normally. The three superseded H1 attempts are additional campaign cost and are excluded from this table. There were no screen infrastructure failures. The original H1 success remains in its retained raw result, rather than being rewritten as a failure.

## Limits and next action

These are nine matched development starts. H6/H7 banks derive from the clean development starts; this screen is not an independent generalization test. Intermediate milestones use sampled simulator observations and can miss transient contact. A lift is not complete placement, and a model stop is not a success predicate. Failed clean recovery qualification left zero qualified states, so RSR isN/A; H8's ordinary episodes cannot fill that denominator.

Complete the nine-start H7 screen with its frozen bank, then select at most two useful candidates alongside clean on the reserved fresh seeds20–24. H8 currently warrants that comparison. H1 has a second local-progress signal with a different mechanism, while H3's two lifts cost roughly twice clean's runtime and H6's retrieval remains coarse. The final choice should include the H7 result. Do not launch the120–180-episode confirmation sweep at this stage.

Evidence: [results](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/screen-data/results.json), [analysis](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/screen-data/analysis.json), [lift transitions](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/screen-data/lift-diagnostics.json), [original screen](/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/screen-pre-gallery-data/results.json). Raw data remain at `h200-4:/home/jli/state/qwen-direct/revision-2026-09/screen/` and `screen-h1-gallery-attempts/`.
