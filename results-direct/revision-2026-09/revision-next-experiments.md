# Targeted follow-up after the repair campaign

This is a proposal for the next experiment; it has not been launched. The repair campaign completed 117 development and 45 fresh validation episodes. Clean, H7, and H8 each scored 0/15 fresh terminal successes. H7 increased approach/contact, but did not increase retained lifts or completion; H8 did not reproduce its development lift advantage. The next work should address the observed grasp and placement interpretation failures before a larger confirmation.

Keep the repaired numerical controller at `1223102` as the reference implementation. Qwen continues to generate the actual EE or joint numbers. Retain the same model, SAM, robot, public feedback, cameras, and episode budgets. The immediate research question is whether better evidence about the target and the outcome of an action changes the demonstrated failures.

## 1. Test recognition of grasp and completion

Several development episodes establish a retained object-following lift, followed by a deliberate opening because Qwen believes the grasp failed. H5 Stove1 decision 36, H6 Stove2 decision 30, and H8 Stove1 decision 32 are concrete examples. H7 Stove2 reaches an official goal state at observation 40, then returns the empty gripper toward the placed object. These are useful starting cases because successful local control has already occurred.

Start with saved development observations. Select a small fixed set covering an empty close, retained lift, accidental drop, deliberate release, correct placement, and incorrect placement. Keep related frames from one episode together. Record the selection before trying the new prompts. Evaluator labels may assess the answers; the model receives only the original images and public feedback.

Compare the existing request with two separate treatments:

- A concise interpretation of measured gripper facts: commanded closed with a nonzero gap can mean obstruction; it does not establish target retention. Commanded closed near the empty gap is evidence against a grasp. Preserve the actual measured values and their uncertainty.
- A Qwen visual verification call using the current and preceding views, with crops around the TCP and the proposed target. Ask whether the target moved with the gripper, was released, is visibly at its destination, or cannot be assessed. Require a brief description of the visual evidence, separate from the proposed next action.

The first treatment tests whether clearer use of existing feedback is enough. The second tests whether explicit temporal visual assessment adds useful information. Do not treat every nonzero finger gap as a grasp, and do not convert the verifier's belief into simulator truth.

Choose the treatment that corrects the demonstrated errors without inventing holds in the empty-close cases. If neither does, report the interpretation failure and inspect the views before running more full episodes.

For the first robot comparison, add the chosen treatment to EE-short as one independent harness. Trigger assessment after a close, a lift, or a release, and before a proposed stop. Pass its evidence to the controller and let Qwen produce the next numeric action or stop. Do not automatically replay a grasp or retreat routine. Test the same nine development starts, count all verification calls, and compare with the unchanged repaired clean baseline.

Measure retained lifts that are deliberately reopened, false completion stops, attainment of the official goal, terminal goal satisfaction, and task success. Distinguish a correct decision to stop an unfinished episode at the budget limit from a false completion claim. Use the existing terminal result as the primary outcome and retain the current sampled milestones for the first matched development screen. In the next cohort with a newly run common baseline, record the evaluator's goal predicate at every simulator step for every condition, without exposing it to the policy. This separates reaching the goal from preserving it until termination. Keep dense and sampled intermediate-goal counts separately labeled.

Implementation locations: `direct/observation.py` for public facts and image history, `direct/harnesses_extra.py` for the independent method, `direct/policy.py` for counted calls, and the existing runtime evaluator for observation-only scoring. Preserve the ordinary H8 branch as its own comparison.

## 2. Test target-directed perception

The initial Drawer0 whisk is visible in RGB but has no dedicated mask among the 21 retained right-view SAM proposals. A supplied region instead combines a large counter mask with a small wrist-view sliver. A legible crop or a nearby centroid therefore does not establish the identity or location of the target.

Use saved development images before spending simulator episodes. Include the whisk, the misidentified Sink1 pepper, and successful lemon/apple localization examples. Inspect both the retained proposals and the raw SAM proposals before filtering in the new diagnostic. This distinguishes missing segmentation from a later filtering or pairing error; the old recordings alone cannot settle that distinction.

Test one target-directed variant with the same SAM model: ask Qwen to identify the named object in the relevant views, supply point/box prompts to SAM, and retain the association between object identity, mask, view, and estimated location. Use two views only when they refer to the same visible object. If the target is hidden or the views disagree, report that uncertainty to Qwen and let it choose a numeric view-changing action.

Compare the resulting target correspondence and position with the original automatic proposals on those saved scenes. The evaluator may use object truth to measure error; current policy inputs must remain image-derived. Keep unrelated region rendering and robot feedback unchanged.

If this corrects the actual wrong-target cases, run nine development episodes as a separate EE-short harness. Reuse the repaired clean baseline only while its code paths and settings remain identical. Evaluate approach, target contact, retained lift, and terminal completion. Count the extra localization calls and SAM work. This is a perception intervention, so report its effect as such.

Implementation locations: `direct/perception.py`, the current observation summary, and one explicit method branch. Extend the existing logs with the selected object/masks and visibility evidence; no new experiment-management framework is needed.

## 3. Refine skill experience only after identifying its useful content

The tested H7 bank supplies the same two overlapping lemon grasp/lift examples on every call, including transport and placement. It tests a small fixed example context; it does not establish useful state-dependent retrieval.

The next bank experiment should isolate that issue. Compare the current two-example context with the same verified content supplied only at its applicable grasp/lift stage. Do not simultaneously add new tasks, a transport library, and a new retriever. First determine whether irrelevant source poses or repeated grasp instructions are contributing to later-stage behavior.

If broader examples become necessary, extract them only from verified development interactions, group overlapping fragments from the same interaction, and label local completion separately from complete-task success. Source coordinates remain examples; Qwen must generate current-scene numeric targets. Add transport or placement entries only when those interactions are actually demonstrated. Freeze the bank before a new validation split.

Keep this behind the first two experiments: target identity and action-outcome interpretation currently provide more concrete failure transitions to address. Do not launch another full H6/H7 sweep merely because new records can be collected.

## 4. Keep recovery qualification separate

Only the H8 branch uses the LIBERO-RECOVER-inspired qualified-state evaluation. The repair campaign's three independent clean qualification attempts did not complete their original tasks. The qualified bank remains empty and recovery success rate is N/A.

Once an independent continuation actually completes a candidate state's original task within 400 steps, 600 seconds, and 80 decisions, retain that qualification and compare clean and H8 from the same restored state and correct historical images. Qualification is separate from the evaluated continuation. A failed qualification establishes that this attempt did not demonstrate recoverability; it does not establish physical impossibility.

Qualification need not wait for the weak clean Qwen policy to solve the state. An independently recorded expert continuation or a separately identified stronger qualification policy can establish recoverability, while clean and H8 remain the evaluated frozen Qwen policies. Record the qualification source and keep its actions out of the evaluated policy's inputs. The three clean qualification attempts already run remain unchanged.

Do not use an improved visual verifier's ordinary-start success as a substitute for this restored-state protocol. Conversely, the other perception and skill experiments do not need to adopt a recovery-bank design.

## 5. Return to joint and full-trajectory control with a specific question

The repaired joint-short, EE-full, and joint-full baselines recorded no target contact on their nine development starts. The full outputs ended normally and were often short or incomplete. A larger output-token limit alone is not a supported fix.

For a later interface experiment, first make each action contract exclusive in its prompt. H2 should have one relative-target definition throughout. Full mode should receive a dedicated one-shot sequence prompt rather than common closed-loop guidance plus an override. Keep numerical targets, limits, and execution unchanged so the comparison identifies the prompt effect.

Use the existing saved initial observations to inspect emitted sequence lengths, numerical validity, and whether grasp/transport/release stages are present. If the prompt change materially changes those outputs, run the corresponding nine development episodes. A generic interpolator may execute the emitted trajectory, but it must not invent omitted task stages. Keep an optional joint FK-preview study separate from a prompt-only comparison.

## Decision and budget

The completed 45-run validation does not justify a larger test of the unchanged policies. The next proposed physical work is at most two nine-run development conditions: action-outcome evidence and target-directed perception, each compared with the valid common clean reference. Start with saved-image diagnostics and launch a condition only when it addresses its demonstrated failure. This is a mechanism-based decision, not a requirement for a prior clean task success.

After those comparisons, choose at most one candidate for a new matched validation. Reserve genuinely unused seeds before creating scenes; seeds 20–24 are now used and must not be presented as fresh. Freeze prompts, code, and banks. Do not launch the previously optional 120–180-episode confirmation until a specific remaining claim justifies that cost.

Deliver the selected observation cases, concrete before/after decisions, implementation revision, counted calls, paired episode results, and a conclusion that distinguishes local progress from task completion. All proposed follow-up outcomes remain unmeasured.

## Working locations

- Source: `/Users/jiachen/Desktop/first try/qwen-direct-control`, branch `direct-control`.
- Frozen repaired runtime: `h200-4:/home/jli/work/qwen-direct-control-revision-r7`.
- Repaired data and archived launch/analysis scripts: `h200-4:/home/jli/state/qwen-direct/revision-2026-09/`.
- Reports: `/Users/jiachen/Documents/Codex/2026-09-10/qwen-https-robocurve-org-https-robocurve/outputs/`.

Develop the next treatment in a new checkout or branch derived from the frozen revision. Preserve the original and revised campaign recordings.

## Reference protocols and addresses

Robocurve's September report uses absolute EE poses, robot IK, three current camera views plus proprioception, and a 20-call LLM budget. It evaluates different models, YAM arms, tasks, and operator grading. It supports the direct numerical-control design, but its reported success rate is not a matched baseline for this RoboCasa campaign. [Robocurve report](https://openai.robocurve.org/gpt-6-astra/).

Inspect Robots exposes LLM motion chunks and supports carrying summarized learnings into a later run. Its separate CaP-X integration uses model-generated code with SAM3, Contact-GraspNet, and IK helpers. That planner-supported policy would be a different experimental condition from Qwen directly authoring our action numbers. [Inspect Robots source and examples](https://github.com/robocurve/inspect-robots).

LIBERO-RECOVER extracts naturally occurring failure states, uses Qwen3.5-27B for failure localization/characterization, and collects expert recovery trajectories through teleoperation. Recovery means completing the original goal from a recoverable failure state. Our bounded qualification procedure is a project-specific implementation of that principle, not a reproduction of the whole benchmark or evidence that its Qwen model directly controls the robot. [Paper](https://arxiv.org/html/2609.05178v1), [project](https://liulin815.github.io/LIBERO-Recovery/).
