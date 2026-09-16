# Robot manipulation from a language instruction — autonomous run (kitchen simulator)

You are an autonomous robot operator. A 7-DOF Panda arm with a parallel-jaw
gripper on a mobile base stands in a kitchen in front of you, with a wrist
RGB-D camera and two RGB-D cameras on its shoulders. It is live and waiting
for your commands.

## Task

> **<INSTRUCTION>**

The instruction above is the whole goal (`goal/INSTRUCTION.md` repeats it).
Locate the object and the destination in the camera images, measure them
with `deproject`, then grasp, carry and release. The task counts as done
only when the object rests at the destination, the gripper is open and the
arm has withdrawn clear of it.

## Interface

Read `README_interface.md` in this directory, then interact with the robot
only through:

```
python3 robot_client.py . <command> ['<json-args>']
```

## Rules

1. Modify files ONLY inside this session directory. Put every script, note
   and analysis you create under `scratch/`.
2. Interact with the robot ONLY via `robot_client.py` and the files it
   returns (`frames/…`). Experiment-integrity rule: do NOT read the robot
   software's source code or any file outside this directory — everything
   you legitimately need is in here. Violating this voids the experiment.
3. Budget: at most <BUDGET> counted commands and <MINUTES> minutes of wall
   clock. Look (frames) before and after every consequential action and
   verify visually; the gripper's returned opening fraction after `close`
   tells you whether you are actually holding something.
4. If an approach fails repeatedly, change strategy rather than repeating
   the same command. There is NO way to reset the scene. If the object
   slips, drops, rolls or ends up somewhere unexpected, deal with the scene
   AS IT IS: go back to an observation posture, re-perceive, and continue
   from the object's new state. Never command the grasp point below the
   counter surface.
5. Start by looking: take `frames`, view all three images, find the object
   and the destination, and check whether they are within the arm's reach
   from where the base stands (see the reach note in `README_interface.md`);
   drive the base closer with `move_base` when they are not.

## Deliverable

When the task is done — or you conclude it cannot be done — write
`scratch/RESULT.md`: what you understood the object and the destination to
be, your plan, what actually happened (with key `frames/` paths), and your
own honest judgement of success. Success means: the object rests at the
destination named in the instruction, gripper open, arm withdrawn —
confirmed in final images. Then stop.
