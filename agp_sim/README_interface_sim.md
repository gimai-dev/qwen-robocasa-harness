# Robot interface (7-DOF Panda arm on a mobile base, kitchen simulator)

You control a 7-DOF Franka Panda arm with a two-finger parallel gripper (max
inner opening **<MAX_OPEN> m**), mounted on a mobile base, standing in a kitchen.
The ONLY way to interact with it is the CLI below. There is no reset and no
undo: whatever you move stays moved. The server holds state between commands
(persistent scene, persistent robot).

## Calling convention

```
python3 robot_client.py . <command> ['<json-args>']
```

Run from the session directory (`.` = session dir). One command at a time —
each call blocks until the robot finishes it (a long move takes 5–40 s).
Client timeout 240 s. The response is a single JSON object on stdout, always
with an `"ok"` field (and an `"error_code"` when `ok` is false). If a call
times out, check server liveness with `tail -3 server.log` and retry once.

## Frames & units

- All poses are in the **robot base frame**: origin at the arm mount on the
  mobile base, x forward (the direction the robot faces), y left, z up,
  metres. The base origin sits about 0.70 m above the floor, so kitchen
  counter tops are at roughly z ≈ 0.20–0.30 in this frame and the
  sink basin is lower; measure the exact heights from the cameras
  (`deproject` on a counter pixel) before descending.
- Quaternions are scalar-first `{"w","x","y","z"}`.
- Tool frame: **+z points out of the fingers** toward the workspace; the
  fingers close along tool y. Poses refer to the **grasp point** between the
  finger pads. Rotation `{"w":0,"x":1,"y":0,"z":0}` points the tool straight
  DOWN with a canonical wrist roll; rotate about tool z for other rolls.
  The ready posture (`home`) has the tool at `<READY_POSE>`.
- Workspace clamp (checked before anything moves): √(x²+y²) ≤ 0.95 from the
  base origin, -0.3 ≤ z ≤ 1.5, and a single move must not travel more than
  0.40 m from the current pose (split long moves). The arm additionally
  enforces joint limits and reachability: an unreachable target returns
  `ok:false` with `IK_FAILED: ...` or `JOINT_LIMIT: ...` and the arm stays
  put. Practical reach of the arm is about 0.75 m from the shoulder (which
  sits 0.33 m above the base origin); targets farther than ~0.7 m
  horizontally from the base usually need the base to drive closer
  (`move_base`).
- Depth gives the VISIBLE surface, not the centre: a single pixel on a
  rounded object returns a point on its near/top surface, several
  centimetres away from the centre. Measure a whole region (see
  `deproject` region form) or reason with the object's size.
- Grasping: the fingers open at most <MAX_OPEN> m, so an object must be
  grasped along a dimension smaller than that, with the grasp point at
  the object's centre height. Rotate the tool about its own z axis (a
  `move_delta` with `drot_deg` about base z while pointing down) to align
  the finger closing direction (tool y) with the object's narrow axis, and
  approach from straight above so the fingers straddle it before closing.
- If a move's path completes but the arm cannot settle onto the target within
  tolerance (typically the fingers or the held object are pressing against
  something, or the target is inside a surface), the response is `ok:false`
  with `error_code: "SETTLE_MISS"` plus the achieved `ee_pose` and
  `target_error_mm`. The arm holds where it stopped and the gripper keeps its
  grip. Decide from fresh frames: lift/retreat, open the gripper there, or
  retry with a corrected target — do not repeat the identical command.

## Commands

| command | args (JSON) | effect / returns |
|---|---|---|
| `status` | — | server and budget info (free) |
| `help` | — | this command list (free) |
| `state` | — | joints (7, rad), ee_pose, gripper_fraction, gripper_width_m, base_world, contact_force_n (free) |
| `frames` | `{"cams":["left","right","wrist"],"depth":true}` (both optional) | captures the cameras from the same instant; saves `frames/NNNN_<cam>.png`, `frames/NNNN_<cam>_depth.npy` (float32 metres), `frames/NNNN_calib.json`; returns the paths |
| `deproject` | `{"capture":N,"cam":"wrist","u":<px>,"v":<px>}` | 3D point (base frame) of that pixel using the saved depth (5×5 median) (free). Add `"plane_z":<m>` to intersect the pixel ray with the horizontal plane z = plane_z instead of using depth |
| `deproject` (region) | `{"capture":N,"cam":"wrist","region":[u0,v0,u1,v1],"above_z":<m>}` | 3D statistics of ALL surface points inside that pixel rectangle that lie above the plane z = above_z (use the counter height to drop the counter): `centroid_base`, `min_base`/`max_base`, `extent_m`, `top_point_base`, `n_points` (free). This is how to measure an object's centre and size: draw the rectangle around the object in the image, give the counter height as `above_z`, grasp at the middle of min/max in x,y and at `top_point_base.z − extent_m.z/2` |
| `move_ee` | `{"position":{"x","y","z"},"rotation":{"w","x","y","z"},"mode":"linear"\|"plan"}` | `linear` = straight Cartesian line holding orientation (default); `plan` = joint-space move to the IK solution. Returns achieved `ee_pose`, `target_error_mm`, and `ok:false` + error if the move was refused or failed (arm stays where it stopped) |
| `move_delta` | `{"dpos":[dx,dy,dz],"drot_deg":[rx,ry,rz]}` (either optional) | relative straight-line move from the current pose; rotation deltas about the BASE axes, applied before the current rotation |
| `move_joints` | `{"joints":[7 floats]}` | joint-space move (radians) |
| `check_pose` | `{"position":{"x","y","z"},"rotation":{"w","x","y","z"}}` | tests a target WITHOUT moving: reachable or not, whether a straight line works, IK residual, horizontal distance (free). Use it before a `move_ee` you are unsure about instead of finding out by a failed move |
| `home` | — | move to the READY posture (arm raised, wrist camera looking forward and down over the workspace) |
| `gripper` | `{"action":"open"\|"close"}` | returns `fraction` (0 closed … 1 open), `width_m` and, after `close`, `held` (true only when the pads stopped on an object at least 12 mm wide). `held: false` means the grasp is EMPTY, whatever the images look like: re-observe and re-aim. The fingers keep their last state through every move: after an empty close, `open` before descending again. While an object is held, every move response carries `carrying: true`; if the width collapses during a move the response says `dropped: true` — the object is no longer in the gripper |
| `move_base` | `{"axis":"x"\|"y"\|"yaw","distance":<m or rad>}` | drives the mobile base: `x` forward along its heading, `y` to its left, `yaw` counter-clockwise; at most 0.5 m / 1.0 rad per call; the arm keeps its joint angles (so the tool moves with the base). Returns `base_world` before/after and the distance actually moved; a note tells you when furniture blocked it |

## Cameras

- `wrist` — <IMGSIZE> RGB-D mounted on the hand, moves with the arm. Depth is
  float32 metres along the optical axis, 0/inf = invalid. Its pose comes from the
  arm's kinematics at capture time.
- `left`, `right` — <IMGSIZE> RGB-D cameras fixed on the robot's base, looking
  over the workspace from the robot's left and right shoulder. They move only
  when the base moves.

`frames/NNNN_calib.json` per capture: for each camera `intrinsics` (3×3 K),
`pose` (camera position + orientation **in the robot base frame** at capture
time), `image_size` [w,h], plus a `_meta` entry with the robot joints,
`ee_pose`, gripper fraction and base pose at that instant. Projection
convention: for a base-frame point X, `Xc = R^T (X - t)` (R = rotation matrix
of `pose.rotation`, t = position), pixel = `K @ (Xc / Xc[2])`; image row 0 is
the top of the picture. `deproject` is the exact inverse of this — you
rarely need to do the math yourself.

## Budget

`status`/`help`/`state`/`deproject` are free. Everything else counts (the
cap is shown by `status`). The session also has a wall-clock limit set in
your task prompt. Be deliberate: look (frames), think, then act.

## Safety

- Collisions are real in this scene: a move that presses into a counter, an
  appliance or the held object ends with `SETTLE_MISS`; do not keep pushing.
- Never command the grasp point below the counter surface you are working on.
- If a move fails part-way the response still reports the achieved `ee_pose`;
  re-observe before the next action.

## Local tooling

- `python3` in this session has numpy, PIL, scipy and cv2 (for your own
  analysis scripts in `scratch/`); `robot_client.py` needs only the stdlib.
- View any local image with your image-viewing tool.
