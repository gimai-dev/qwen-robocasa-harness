# Fresh validation: clean, skill context, and expected-effect feedback

All 45 episodes and offline inspections are complete. Clean, H7, and H8 each scored **0/15 terminal task successes** on the same three tasks and reserved seeds 20–24. No saved validation observation satisfied an intermediate official goal predicate either. There were no infrastructure failures or retries in this batch.

H7 and H8 were selected after the development screen and before validation scene setup or outcomes. All runs use policy revision `1223102`, the original posture, the same frozen Qwen/SAM setting, 900 simulator steps, 1,200 seconds, and at most 180 decisions. H7's bank remains the same two overlapping clean-development lemon fragments. No validation examples enter it.

| Condition | Terminal success | Approach ≤10 cm | Target contact | Closed contact | Lift milestone | Verified retained lift |
|---|---:|---:|---:|---:|---:|---:|
| Clean |0/15|3|4|3|2|2|
| H7 skill context |0/15|8|6|5|2|2|
| H8 expected effects |0/15|4|3|2|2|1|

The lift milestone records a target rise of at least3 cm while closed-gripper contact is observed. Verified retained lift additionally requires an object-following fragment under the existing skill extractor. H8 Drawer23 briefly elevates the tongs during closing, then immediately reopens; it has no qualifying retained-lift fragment. Its original milestone remains recorded. Closed contact alone does not establish a secure grasp.

## Paired effects

| Candidate | Approach gains/losses | Contact gains/losses | Closed-contact gains/losses | Lift-milestone gains/losses |
|---|---:|---:|---:|---:|
| H7 versus clean |5/0|3/1|3/1|1/1|
| H8 versus clean |2/1|1/2|1/2|1/1|

H7's approach gain carries over to fresh starts: five new approaches, with no lost clean approaches. It adds contact on Drawer20 and Stove20/24, but loses clean's Drawer21 contact. Both H7 and clean retain the Stove22 apple-lift case. H7 gains a verified Drawer23 tongs lift while losing clean's Sink20 eggplant lift. There is no net lift increase and no completion gain.

H8's stronger development lift count does not reproduce as a fresh retained-lift advantage. Its only verified retained-lift episode is Stove22. It loses clean's Sink20 lift; the additional Drawer23 milestone is brief closed-contact elevation. H8 records fewer contact and closed-contact episodes than clean on these starts.

Each observed 0/15 terminal success rate has a two-sided 95% Clopper–Pearson interval of **0–21.8%**, under a binomial model. The matched terminal outcomes are all joint failures. These observations do not establish equivalence or universal inability to succeed.

## What the failures show

**Clean Stove22 repeatedly opens after real apple lifts.** It has eleven overlapping verified local fragments, with maximum held elevation0.430 m. At decisions10 and 24, Qwen interprets the closed-on-apple gap of about6.3 cm as a failed close and explicitly opens to re-grasp. A later transport action at decision43 loses the object while the gripper remains commanded closed; subsequent reasoning still claims to be carrying it despite a near-empty gap. The apple ends outside the destination.

**H7 repeats the mistaken re-grasp pattern on fresh scenes.** Stove22 retains the apple from observations10–22 and lifts it0.272 m. Decision27 opens because Qwen now believes the apple is back in the pan. Drawer23 lifts the tongs0.417 m and retains contact through observation32. Decision34 opens and approaches a supposed counter location, undoing the retained grasp. Both episodes finish without placement. These cases support an action-outcome interpretation experiment independently of the development lemon example.

**H8 can recognize a grasp yet target the wrong destination.** On Stove22 it retains and lifts the apple0.320 m, then deliberately releases at decision38 believing it is over the bowl. The object ends outside the bowl near its release location. On Drawer23, the sole closed-contact observation rises4.3 cm during the close; decision18 opens because Qwen interprets the gap as a failed grasp. There is no observed retained arm-lift fragment in that episode.

**False completion is also present in fresh validation.** Six of the seventeen model stops are based on a mistaken belief that placement is complete: clean Sink21/22/23, H7 Sink23, and H8 Sink21/23. The other eleven stops acknowledge an unfinished task near the step limit. Those acknowledgments are correct failure reports. The remaining 28 episodes end at the simulator-step budget.

H7 Sink23 is a concrete localization failure: it stops after120 steps claiming the reamer is in the sink. The actual target stays within 1 mm of its initial counter position, while the final gripper is0.518 m away. The clearance condition is met; placement fails. This case cannot be repaired merely by adding another retreat instruction.

## Mechanism and cost

H7 supplies the same two examples on all 693 control calls:1,386 example exposures. It obtains96 distinct control observations within 10 cm of the target, versus clean47. Its contact benefit is therefore accompanied by more observed near-target interaction, but it still lacks reliable grasp retention and placement.

H8 records 598 TCP/gap checks:403 mismatches and 195 matches. All598 holding checks remain unknown. This is the repaired mechanism behaving as specified; it does not supply independent visual proof of target retention.

All2,028 Qwen responses finish normally. The 163 rejected actions are unresolved numeric targets (clean60, H7 56, H8 47), not truncated control responses. All executed slots retain the fixed20-step duration.

| Condition | Calls | Prompt tokens | Completion tokens | Qwen s | SAM s | Summed episode wall s |
|---|---:|---:|---:|---:|---:|---:|
| Clean |684|2,761,241|102,858|2,446.9|2,905.7|6,194.0|
| H7 |693|3,379,448|103,584|2,518.9|2,836.5|6,208.5|
| H8 |651|2,853,092|144,556|3,346.4|2,234.7|6,369.2|

H7 uses about22.4% more prompt tokens than clean, with similar mean episode runtime (413.9 versus 412.9 seconds). H8 averages424.6 seconds. With no successful tasks, these are episode costs, not costs per successful task.

## Decision

Keep the repaired executor and H7's measured contact result. The next useful intervention is evidence about whether the target actually moved with the gripper and where it was released, alongside target-directed perception. The fresh failures support that diagnosis; they do not justify another large confirmation of the unchanged H7/H8 policies.

The bounded [next-experiment plan](revision-next-experiments.md) keeps these mechanisms separate and retains Qwen as the numeric action author. No further policy episodes are launched as part of this campaign.

## Evidence and limits

The [generated summary](revision-validation-summary.md) and [paired records](revision-validation-summary.json) contain the full comparison. Local batch data are in `validation-data/`; full raw trajectories, videos, and inspections remain at `h200-4:/home/jli/state/qwen-direct/revision-2026-09/validation/`. The H7 reamer inspection is also copied to `validation-data/h7-sink23-inspection.json`.

Fifteen starts per condition across three tasks provide limited precision. Milestones and intermediate goals use saved observation states; events between observations may be missed. The stricter skill-fragment check is the same extractor used before validation. Fragment counts can overlap within one episode and must not be treated as independent successful trials. The supplemental intermediate-goal analysis is exploratory; terminal scoring and the frozen policies were unchanged.
