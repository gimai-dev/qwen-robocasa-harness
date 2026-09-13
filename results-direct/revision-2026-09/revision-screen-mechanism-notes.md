# Independent harness screen: completed mechanism analysis

H5, H6, and H8 each completed nine matched development starts with no full-task success. Their local progress differs. The final H1 comparison and its gallery repair are described in `revision-screen-comparison.md`. These notes use the completed, unaffected H5/H6/H8 results from the original36-run screen.

| Condition | Episodes | Task success | Approach ≤10cm | Target contact | Closed contact | Lift ≥3cm | Calls | Aggregate episode wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EE-clean |9|0|2|2|1|1|398|3267.8|
| H5 working memory |9|0|2|3|2|1|345|4502.1|
| H6 failure experience |9|0|4|4|2|2|353|3160.3|
| H8 expected effects/recovery |9|0|5|3|3|3|366|3566.7|

Approach and grasp milestones come from saved simulator observation states. Closed contact requires a closed command, a gap above5mm, and gripper contact with the target. A lift requires the target to rise3cm while that condition holds. All six lift episodes above also contain at least one retained object-following lift fragment under the stricter skill extractor. This diagnosis does not add their fragments to the frozen clean-derived bank.

H5 updated memory339 times. All345 control responses finished normally. Its six stops comprise two false completion claims on Sink seeds1/2 and four explicit acknowledgments of an unfinished task near the budget limit. The latter are correct descriptions of failure, not hallucinated successes. Sink1 treats the pepper as already inside the sink and eventually says placement and retreat are complete, despite no target contact. Sink0 remembers the orange's location early, then replaces it with an incorrect region position and is still moving the base at the budget limit.

H5 Stove1 has a real tomato lift up to0.163m. The target remains held from observations21–35, but decision36 opens the gripper because Qwen says the tomato is still in the pan. It later grasps/lifts again and ends holding the tomato. Transport and placement are unfinished. Working memory preserves both correct observations and mistaken beliefs; its tested form does not establish a task-success benefit. It used1,531,678 prompt tokens and130,028 completion tokens; Qwen time2906.6s and SAM time1139.3s.

H6 delivered two failure records on every one of353 calls. The706 record exposures comprise437 base-blocked,129 empty-close, and140 unreachable examples. There are103 verified-correction exposures, all on Drawer0(59) and Drawer2(44). Other runs received unverified failure descriptions. Six of nine runs receive one unchanged retrieval set; the three Drawer starts receive two sets each. Sink0 receives the same two base-blocked records through navigation and later empty-grasp attempts. The retrieval mechanism works as implemented, but its task/phase/word-overlap ranking is too coarse to respond to those changing failures. The bank has no task-level false-completion examples.

H6 Stove0 achieves a retained sweet-potato lift up to0.210m and ends holding it, still attempting transport. Stove2 lifts the lemon0.099m, then decision30 opens to retry because Qwen believes it was never lifted; no complete placement follows. Sink1 stops after its first call with zero simulator actions, incorrectly declaring the pepper already in the sink. H6 has local gains but0/9 task success. It used1,916,345 prompt tokens and52,946 completion tokens; Qwen time1250.7s and SAM time1439.5s.

H8 records345 post-action checks:233 mismatches and112 matches of predicted TCP/gap. All345 object-holding checks remain explicitly unknown. That is the intended repaired behavior: matching finger gap and pose does not verify an object. All366 control responses finished normally. H8 recognizes some empty closes and contact blockages, yet also repeats blocked directions and misinterprets actual grasps.

H8 Sink0 lifts the orange0.157m at observation32 and loses it during the next lateral transport action. Decision34 recognizes an empty grasp and retries, but the episode ends without replacing it in the sink. Stove1 lifts the tomato0.163m and ends holding it; earlier, decision32 deliberately reopens after a retained lift because it mistakes the large closed-on-object gap for an unclosed gripper. Stove2 lifts the lemon0.094m; decision46 opens after another false failed-grasp diagnosis. H8 retains clean's Stove2 lift and gains Sink0/Stove1, with no full-task success. It used1,630,888 prompt tokens and81,946 completion tokens; Qwen time1791.2s and SAM time1283.5s.

The public observation already includes both `gripper_command` and `gripper_width_m`. The next research question is how to use temporal object motion and observed contact/gap evidence to distinguish a retained grasp, a dropped object, and an empty close. No new grasp script or state estimator was inserted during this comparison.

H8's ordinary starts are not recovery-success-rate trials. All three independent clean qualification attempts failed to complete their original task, leaving no qualified recovery states. RSR remainsN/A.

Evidence: `screen-pre-gallery-data/results.json`, `analysis.json`, and `lift-diagnostics.json` in this output directory; full raw runs remain on h200-4 under `/home/jli/state/qwen-direct/revision-2026-09/screen/`. The original H1 aggregates are preserved before replacement. These nine-start comparisons guide candidate selection; fresh validation remains separate.
