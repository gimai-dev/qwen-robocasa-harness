"""Network-isolated RoboCasa child for bounded absolute Panda joint control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import socket
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .chassis_hold import ChassisHold, validate_chassis_receipt

from .joint_protocol import (
    ACTUAL_RELATIVE_TRACKING,
    JOINT_LIMIT_MARGIN,
    JOINT_LIMITS,
    JOINT_STEP_TOLERANCE,
    JOINT_TRACKING_MODES,
    LAG_PAUSE_TRACKING,
    MAX_COMMAND_ACTIONS,
    MAX_JOINT_STEP,
    MIN_GRIPPER_ACTIONS,
    SETTLE_ACTIONS,
    next_joint_waypoint,
    validate_endpoint_tolerance,
    validate_joint_waypoints,
    validate_waypoint_progress,
)
from .panda_embodiment import (
    OFFICIAL_PUBLIC_STATE_KEYS,
    PUBLIC_TELEMETRY_KEYS,
    read_public_telemetry,
    select_named_arm_vector,
    summarize_telemetry_samples,
    validate_public_state,
    validate_telemetry_summary,
)

EPISODE_ACTION_BUDGET = 450
MAX_DEVELOPMENT_ACTION_BUDGET = 3000
EPISODE_WALL_BUDGET_S = 1_200
BASE_STEPS = 5
BASE_VELOCITY_LIMIT = 0.25
_PUBLIC_EXECUTION_FIELDS = {
    "move_joints": frozenset(
        {
            "kind",
            "accepted",
            "bounded_endpoint",
            "gripper_intent",
            "step_count",
            "maximum_commanded_step",
            "realized_arm_qpos",
            "endpoint_error",
            "minimum_hard_limit_margin",
            "tracking_pause_count",
            "telemetry_summary",
            "end_effector_external_pixels_before",
            "end_effector_external_pixels_after",
        }
    ),
    "base_action": frozenset(
        {
            "kind",
            "accepted",
            "axis",
            "normalized_velocity",
            "gripper_intent",
            "step_count",
            "base_motion_step_count",
            "realized_arm_qpos",
            "tracking_pause_count",
            "remaining_endpoint_error",
            "telemetry_summary",
            "end_effector_external_pixels_before",
            "end_effector_external_pixels_after",
        }
    ),
}


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label} must be a finite number")
    return output


def _seven(value: object, *, label: str) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must contain seven finite values")
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{label} must contain seven finite values") from error
    if len(items) != 7:
        raise ValueError(f"{label} must contain seven finite values")
    return [_finite(item, label=label) for item in items]


def _hard_limit_qpos(value: object, *, label: str) -> list[float]:
    output = _seven(value, label=label)
    for index, (item, (lower, upper)) in enumerate(
        zip(output, JOINT_LIMITS, strict=True)
    ):
        if not lower <= item <= upper:
            raise ValueError(f"{label} joint{index + 1} is outside its hard limit")
    return output


def _endpoint_and_mask(
    endpoint: object,
    explicit_mask: object,
    *,
    current_qpos: object,
) -> tuple[list[float], list[bool], list[float]]:
    current = _hard_limit_qpos(current_qpos, label="current qpos")
    target = _seven(endpoint, label="endpoint")
    if (
        not isinstance(explicit_mask, Sequence)
        or isinstance(explicit_mask, (str, bytes))
        or len(explicit_mask) != 7
        or any(type(item) is not bool for item in explicit_mask)
    ):
        raise ValueError("explicit mask must contain seven booleans")
    mask = list(explicit_mask)
    for index, (item, is_explicit, limits) in enumerate(
        zip(target, mask, JOINT_LIMITS, strict=True)
    ):
        lower, upper = limits
        if is_explicit:
            if not lower + JOINT_LIMIT_MARGIN <= item <= upper - JOINT_LIMIT_MARGIN:
                raise ValueError(
                    f"explicit endpoint joint{index + 1} is outside the inset"
                )
        else:
            if not lower <= item <= upper:
                raise ValueError(
                    f"held endpoint joint{index + 1} is outside its hard limit"
                )
            if abs(item - current[index]) > JOINT_STEP_TOLERANCE:
                raise ValueError(
                    f"held endpoint joint{index + 1} differs from execution qpos"
                )
            target[index] = current[index]
    return target, mask, current


def joint_controller_config(config: Mapping[str, object]) -> dict[str, object]:
    """Use absolute arm joints and hold the kitchen's zero torso extension."""
    output = copy.deepcopy(dict(config))
    body_parts = output.get("body_parts")
    if not isinstance(body_parts, dict) or not isinstance(body_parts.get("right"), dict):
        raise ValueError("Panda composite controller is missing the right arm")
    old_right = body_parts["right"]
    gripper = copy.deepcopy(old_right.get("gripper"))
    if not isinstance(gripper, dict):
        raise ValueError("Panda right-arm gripper controller is missing")
    lower = [limits[0] for limits in JOINT_LIMITS]
    upper = [limits[1] for limits in JOINT_LIMITS]
    body_parts["right"] = {
        "type": "JOINT_POSITION",
        "input_min": lower,
        "input_max": upper,
        "output_min": lower,
        "output_max": upper,
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": [lower, upper],
        "interpolation": None,
        "ramp_ratio": 0.2,
        "input_type": "absolute",
        "gripper": gripper,
    }
    # Kitchen._setup_model resets the torso to zero. Delta zero would reset
    # its goal to each newly achieved height, accumulating contact displacement.
    body_parts["torso"]["input_type"] = "absolute"
    return output


def _validated_child_action(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != {"joint_position", "gripper_open", "base_motion", "torso"}:
        raise ValueError("joint child action fields drifted")
    joint_position = _hard_limit_qpos(
        value.get("joint_position"), label="joint position"
    )
    gripper_open = _finite(value.get("gripper_open"), label="gripper")
    if not 0.0 <= gripper_open <= 1.0:
        raise ValueError("gripper must be between 0 and 1")
    base = value.get("base_motion")
    if (
        not isinstance(base, Sequence)
        or isinstance(base, (str, bytes))
        or len(base) != 3
    ):
        raise ValueError("base motion must contain three finite values")
    base_motion = [_finite(item, label="base motion") for item in base]
    if any(abs(item) > BASE_VELOCITY_LIMIT + 1e-12 for item in base_motion):
        raise ValueError("base motion exceeds its bound")
    if sum(abs(item) > 1e-12 for item in base_motion) > 1:
        raise ValueError("base motion must use one axis")
    torso = _finite(value.get("torso"), label="torso")
    if abs(torso) > 1e-12:
        raise ValueError("torso must remain zero")
    return {
        "joint_position": joint_position,
        "gripper_open": gripper_open,
        "base_motion": base_motion,
        "torso": torso,
    }


def joint_unmap_action(value: Mapping[str, object]) -> dict[str, object]:
    """Map candidate joint fields to the existing Panda composite parts."""
    action = _validated_child_action(value)
    # RoboSuite's PandaGripper declares drive +1 as closed and -1 as open.
    # The direction-sensitive H200 smoke guards the action ramp and endpoints.
    gripper_drive = 1.0 - 2.0 * float(action["gripper_open"])
    # HybridMobileBase applies the base velocity goal in either mode. Mode
    # selects the arm goal-update convention; arm actions still brake the base
    # with a zero velocity goal.
    base_motion = list(action["base_motion"])
    base_mode = 1.0 if any(abs(item) > 1e-12 for item in base_motion) else -1.0
    return {
        "robot0_right": list(action["joint_position"]),
        "robot0_right_gripper": gripper_drive,
        "robot0_base": base_motion,
        "robot0_torso": [0.0],
        "robot0_base_mode": base_mode,
    }


def validate_joint_action_sequence(
    actions: Sequence[Mapping[str, object]],
    *,
    current_qpos: object,
    endpoint: object,
    explicit_mask: object,
) -> tuple[dict[str, object], ...]:
    """Independently validate every action immediately before simulator use."""
    if not 1 <= len(actions) <= MAX_COMMAND_ACTIONS:
        raise ValueError("joint action sequence must contain one to 32 actions")
    target, _mask, current = _endpoint_and_mask(
        endpoint, explicit_mask, current_qpos=current_qpos
    )
    del target
    output: list[dict[str, object]] = []
    previous = current
    for index, raw in enumerate(actions):
        action = _validated_child_action(raw)
        position = list(action["joint_position"])
        maximum_step = max(
            abs(item - prior)
            for item, prior in zip(position, previous, strict=True)
        )
        limit = MAX_JOINT_STEP + (JOINT_STEP_TOLERANCE if index == 0 else 0.0)
        if maximum_step > limit + 1e-12:
            label = "first waypoint" if index == 0 else "later waypoint"
            raise ValueError(f"{label} exceeds the joint step bound")
        previous = position
        output.append(action)
    return tuple(output)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _network_denied() -> str:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.2):
            pass
    except OSError as error:
        return type(error).__name__
    raise RuntimeError("simulator network namespace is not isolated")


def _select_arm_qpos(
    all_qpos: object, robot_joint_names: object, arm_joint_names: object
) -> list[float]:
    return select_named_arm_vector(
        all_qpos,
        robot_joint_names,
        arm_joint_names,
        "mobile robot arm qpos",
    )


def _arm_qpos(environment: object) -> list[float]:
    robot = environment.unwrapped.robots[0]
    return _hard_limit_qpos(
        _select_arm_qpos(
            robot.get_robot_joint_positions(),
            robot.robot_joints,
            robot.robot_arm_joints,
        ),
        label="measured arm qpos",
    )


def _minimum_hard_limit_margin(qpos_history: Sequence[Sequence[float]]) -> float:
    return min(
        min(value - lower, upper - value)
        for qpos in qpos_history
        for value, (lower, upper) in zip(qpos, JOINT_LIMITS, strict=True)
    )


def _coordinate_grid_image(image: object) -> object:
    """Overlay a sparse 32 px coordinate grid without hiding scene content."""
    output = image.convert("RGB").copy()
    width, height = output.size
    for x in range(0, width, 32):
        for y in range(0, height, 4):
            output.putpixel((x, y), (255, 255, 0))
    for y in range(0, height, 32):
        for x in range(0, width, 4):
            output.putpixel((x, y), (0, 255, 255))
    return output


def _publish_observation(
    raw: Mapping[str, object],
    *,
    environment: object,
    run: Path,
    episode: str,
    sequence: int,
    execution: Mapping[str, object] | None,
) -> dict[str, object]:
    import numpy as np
    from PIL import Image
    from robocasa_inspect.camera_geometry import official_camera_calibration
    from robocasa_inspect.contracts import CAMERAS, project_observation

    public = project_observation(raw, episode=episode, sequence=sequence)
    camera_calibration = official_camera_calibration(environment)
    telemetry = read_public_telemetry(environment, camera_calibration)
    if set(public.state_groups) != OFFICIAL_PUBLIC_STATE_KEYS:
        raise ValueError("official public observation schema drifted")
    if set(telemetry) != PUBLIC_TELEMETRY_KEYS:
        raise ValueError("public observation schema drifted")
    public_execution: dict[str, object] | None = None
    if execution is not None:
        expected_execution_fields = _PUBLIC_EXECUTION_FIELDS.get(
            str(execution.get("kind"))
        )
        if (
            expected_execution_fields is None
            or set(execution) - {"chassis_hold", "endpoint_tolerance", "waypoint_progress"} != expected_execution_fields
        ):
            raise ValueError("public observation schema drifted")
        public_execution = dict(execution)
        if "endpoint_tolerance" in execution:
            if execution.get("kind") != "move_joints":
                raise ValueError("endpoint tolerance requires move_joints")
            public_execution["endpoint_tolerance"] = validate_endpoint_tolerance(execution["endpoint_tolerance"])
        if "waypoint_progress" in execution:
            if execution.get("kind") != "move_joints":
                raise ValueError("waypoint progress requires move_joints")
            public_execution["waypoint_progress"] = validate_waypoint_progress(execution["waypoint_progress"])
        if "chassis_hold" in execution:
            public_execution["chassis_hold"] = validate_chassis_receipt(execution["chassis_hold"], execution["step_count"])
        public_execution["telemetry_summary"] = validate_telemetry_summary(
            execution.get("telemetry_summary")
        )
    frame_dir = run / "frames" / f"{sequence:06d}"
    frame_dir.mkdir(parents=True, mode=0o700)
    images: dict[str, dict[str, str]] = {}
    for label, key in zip(("left", "right", "wrist"), CAMERAS, strict=True):
        target = frame_dir / f"{label}.png"
        public_image = Image.fromarray(
            np.asarray(public.images[key], dtype=np.uint8)
        )
        _coordinate_grid_image(public_image).save(target, format="PNG")
        target.chmod(0o600)
        images[label] = {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    state = dict(public.state_groups)
    state.update(telemetry)
    state = validate_public_state(state, require_torque_available=False)
    telemetry = {key: state[key] for key in PUBLIC_TELEMETRY_KEYS}
    observation_id = hashlib.sha256(
        json.dumps(
            {
                "official_observation_id": public.observation_id,
                "telemetry": telemetry,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    record: dict[str, object] = {
        "schema": "robocasa-inspect-joint-observation/v1",
        "episode": episode,
        "sequence": sequence,
        "observation_id": observation_id,
        "instruction": public.instruction,
        "images": images,
        "public_state": state,
        "camera_calibration": camera_calibration,
    }
    if public_execution is not None:
        record["execution"] = public_execution
    _atomic_json(run / "mailbox" / f"observation-{sequence:06d}.json", record)
    return record


def _read_command(run: Path, sequence: int, timeout_s: float) -> dict[str, object]:
    target = run / "mailbox" / f"command-{sequence:06d}.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            value = json.loads(target.read_text())
        except FileNotFoundError:
            time.sleep(0.01)
            continue
        if value.get("sequence") != sequence:
            raise RuntimeError("mailbox command sequence mismatch")
        return value
    raise TimeoutError("simulator command mailbox timed out")


def _install_joint_controller() -> None:
    import numpy as np
    from robocasa.utils import env_utils
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter

    original_loader = env_utils.load_composite_controller_config
    joint_config = joint_controller_config(
        original_loader(controller=None, robot="PandaOmron")
    )

    def load_config(controller: object = None, robot: object = None) -> object:
        if controller is None and robot == "PandaOmron":
            return copy.deepcopy(joint_config)
        return original_loader(controller=controller, robot=robot)

    def unmap(_cls: object, input_action: Mapping[str, object]) -> dict[str, object]:
        if set(input_action) != {
            "action.joint_position", "action.gripper_open",
            "action.base_motion", "action.torso",
        }:
            raise ValueError("joint wrapper action fields drifted")
        gripper = np.asarray(
            input_action["action.gripper_open"], dtype=np.float64
        ).reshape(-1)
        torso = np.asarray(input_action["action.torso"], dtype=np.float64).reshape(-1)
        if gripper.shape != (1,) or torso.shape != (1,):
            raise ValueError("joint wrapper scalar fields drifted")
        candidate = {
            "joint_position": np.asarray(
                input_action["action.joint_position"], dtype=np.float64
            ).reshape(-1).tolist(),
            "gripper_open": float(gripper[0]),
            "base_motion": np.asarray(
                input_action["action.base_motion"], dtype=np.float64
            ).reshape(-1).tolist(),
            "torso": float(torso[0]),
        }
        return {
            key: np.asarray(value, dtype=np.float64)
            for key, value in joint_unmap_action(candidate).items()
        }

    env_utils.load_composite_controller_config = load_config
    PandaOmronKeyConverter.unmap_action = classmethod(unmap)


def _official_terminal_success(environment: object) -> bool:
    import numpy as np

    success = environment.unwrapped._check_success()
    if not isinstance(success, (bool, np.bool_)):
        raise TypeError("official success predicate drift")
    return bool(success)


def _terminal_snapshot_sha256(environment: object) -> str:
    import numpy as np

    simulator = environment.unwrapped.sim
    parts = [
        np.asarray(simulator.data.qpos, dtype="<f8").reshape(-1),
        np.asarray(simulator.data.qvel, dtype="<f8").reshape(-1),
        np.asarray(simulator.data.act, dtype="<f8").reshape(-1),
        np.asarray(simulator.data.ctrl, dtype="<f8").reshape(-1),
        np.asarray([simulator.data.time], dtype="<f8"),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(np.asarray([len(part)], dtype="<u8").tobytes())
        digest.update(part.tobytes(order="C"))
    return digest.hexdigest()


def _step(environment: object, action: Mapping[str, object]) -> Mapping[str, object]:
    import numpy as np

    payload = {
        "action.joint_position": np.asarray(action["joint_position"], dtype=np.float64),
        "action.gripper_open": np.asarray([action["gripper_open"]], dtype=np.float64),
        "action.base_motion": np.asarray(action["base_motion"], dtype=np.float64),
        "action.torso": np.asarray([action["torso"]], dtype=np.float64),
    }
    raw, _, _, _, _ = environment.step(payload)
    return raw


def _telemetry_evidence(
    before: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    after: Mapping[str, object],
) -> dict[str, object]:
    return {
        "telemetry_summary": summarize_telemetry_samples(before, samples, after),
        "end_effector_external_pixels_before": before[
            "state.end_effector_external_pixels"
        ],
        "end_effector_external_pixels_after": after[
            "state.end_effector_external_pixels"
        ],
    }


def _read_fresh_public_telemetry(environment: object) -> dict[str, object]:
    """Project public telemetry with the cameras' current world calibration."""

    from robocasa_inspect.camera_geometry import official_camera_calibration

    return read_public_telemetry(
        environment, official_camera_calibration(environment)
    )


def _execute_move(
    environment: object,
    command: Mapping[str, object],
    *,
    current_gripper: float,
    total_actions: int,
    started: float,
    action_budget: int = EPISODE_ACTION_BUDGET,
    chassis_hold: ChassisHold | None = None,
    public_raw: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, object], dict[str, object], float, int]:
    required = {
        "schema",
        "sequence",
        "kind",
        "observation_id",
        "endpoint",
        "explicit_mask",
        "gripper_open",
        "previous_gripper_open",
        "gripper_transition",
        "max_actions",
    }
    if set(command) - {"tracking_mode", "endpoint_tolerance", "waypoints"} != required:
        raise ValueError("move_joints mailbox fields drifted")
    tracking_mode = command.get("tracking_mode", LAG_PAUSE_TRACKING)
    endpoint_tolerance = validate_endpoint_tolerance(command.get("endpoint_tolerance", JOINT_STEP_TOLERANCE))
    if tracking_mode not in JOINT_TRACKING_MODES:
        raise ValueError("move_joints tracking mode is invalid")
    max_actions = command.get("max_actions")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int):
        raise ValueError("max_actions must be an integer")
    if not 1 <= max_actions <= MAX_COMMAND_ACTIONS:
        raise ValueError("max_actions exceeds the 32-action command budget")
    target, mask, initial = _endpoint_and_mask(
        command.get("endpoint"),
        command.get("explicit_mask"),
        current_qpos=_arm_qpos(environment),
    )
    path = None
    path_index = 0
    if "waypoints" in command:
        if not all(mask):
            raise ValueError("joint paths require all seven explicit dimensions")
        path = validate_joint_waypoints(command["waypoints"], target)
    gripper_open = _finite(command.get("gripper_open"), label="gripper")
    if not 0.0 <= gripper_open <= 1.0:
        raise ValueError("gripper must be between 0 and 1")
    gripper_changed = abs(gripper_open - current_gripper) > 1e-12
    previous_gripper_open = _finite(
        command.get("previous_gripper_open"), label="previous gripper intent"
    )
    if (
        not 0.0 <= previous_gripper_open <= 1.0
        or abs(previous_gripper_open - current_gripper) > 1e-12
        or command.get("gripper_transition") is not gripper_changed
    ):
        raise ValueError("move_joints gripper transition metadata drifted")
    if gripper_changed and (
        max_actions < MIN_GRIPPER_ACTIONS
        or total_actions + MIN_GRIPPER_ACTIONS > action_budget
    ):
        raise RuntimeError(
            f"gripper transition requires {MIN_GRIPPER_ACTIONS} available actions"
        )
    actions: list[dict[str, object]] = []
    realized_history: list[list[float]] = [initial]
    prior: tuple[float, ...] | None = None
    reached = False
    settle_count = 0
    tracking_pauses = 0
    maximum_step = 0.0
    raw: Mapping[str, object] | None = None
    telemetry_before = _read_fresh_public_telemetry(environment)
    telemetry_samples: list[dict[str, object]] = []
    hold_corrections: list[list[float]] = []
    if chassis_hold is not None and public_raw is None:
        raise ValueError("chassis hold requires the preceding public raw observation")
    for _ in range(max_actions):
        if total_actions >= action_budget:
            break
        if time.monotonic() - started > EPISODE_WALL_BUDGET_S:
            break
        actual = _arm_qpos(environment)
        if path is not None:
            while path_index < len(path)-1 and max(abs(a-b) for a,b in zip(actual,path[path_index],strict=True)) < MAX_JOINT_STEP:
                path_index += 1
                reached = False
                settle_count = 0
        active_target = target if path is None else path[path_index]
        if reached and tracking_mode == LAG_PAUSE_TRACKING:
            waypoint = tuple(active_target)
        else:
            waypoint = next_joint_waypoint(
                actual,
                prior,
                active_target,
                tracking_mode=str(tracking_mode),
            )
            if (
                tracking_mode == LAG_PAUSE_TRACKING
                and prior is not None
                and waypoint == prior
                and waypoint != tuple(active_target)
            ):
                tracking_pauses += 1
        reference = (
            actual
            if prior is None or tracking_mode == ACTUAL_RELATIVE_TRACKING
            else list(prior)
        )
        maximum_step = max(
            maximum_step,
            max(abs(a - b) for a, b in zip(waypoint, reference, strict=True)),
        )
        action = {
            "joint_position": list(waypoint),
            "gripper_open": gripper_open,
            "base_motion": chassis_hold.correction(public_raw) if chassis_hold is not None else [0.0, 0.0, 0.0],
            "torso": 0.0,
        }
        actions.append(action)
        if tracking_mode == ACTUAL_RELATIVE_TRACKING:
            validate_joint_action_sequence(
                [action],
                current_qpos=actual,
                endpoint=target,
                explicit_mask=mask,
            )
        else:
            validate_joint_action_sequence(
                actions,
                current_qpos=initial,
                endpoint=target,
                explicit_mask=mask,
            )
        raw = _step(environment, action)
        public_raw = raw
        hold_corrections.append(list(action["base_motion"]))
        telemetry_samples.append(_read_fresh_public_telemetry(environment))
        total_actions += 1
        realized_history.append(_arm_qpos(environment))
        prior = waypoint
        if path is not None and path_index < len(path)-1:
            if max(abs(a-b) for a,b in zip(active_target,realized_history[-1],strict=True)) < MAX_JOINT_STEP:
                path_index += 1
                reached = False
                settle_count = 0
            continue
        # Intermediate IK waypoints use the driver's measured arrival bound.
        # Stage endpoints and every gripper transition retain precise settling.
        if endpoint_tolerance > JOINT_STEP_TOLERANCE and not gripper_changed:
            if max(abs(a-b) for a,b in zip(target, realized_history[-1], strict=True)) < endpoint_tolerance:
                if path is not None:path_index = len(path)
                break
        if waypoint == tuple(target):
            if reached:
                endpoint_error = max(
                    abs(expected - realized)
                    for expected, realized in zip(
                        target, realized_history[-1], strict=True
                    )
                )
                settle_count = (
                    settle_count + 1
                    if endpoint_error <= JOINT_STEP_TOLERANCE
                    else 0
                )
            else:
                reached = True
            if settle_count >= SETTLE_ACTIONS and (
                    not gripper_changed or len(actions) >= MIN_GRIPPER_ACTIONS
            ):
                if path is not None:path_index = len(path)
                break
    if gripper_changed and len(actions) < MIN_GRIPPER_ACTIONS:
        raise RuntimeError(
            f"gripper transition settling truncated after {len(actions)} actions"
        )
    if raw is None:
        raise RuntimeError("joint command had no remaining simulator action budget")
    telemetry_after = _read_fresh_public_telemetry(environment)
    realized = realized_history[-1]
    receipt = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": target,
        "gripper_intent": gripper_open,
        "step_count": len(actions),
        "maximum_commanded_step": maximum_step,
        "realized_arm_qpos": realized,
        "endpoint_error": max(
            abs(a - b) for a, b in zip(target, realized, strict=True)
        ),
        "minimum_hard_limit_margin": _minimum_hard_limit_margin(realized_history),
        "tracking_pause_count": tracking_pauses,
        **({"endpoint_tolerance": endpoint_tolerance} if "endpoint_tolerance" in command else {}),
        **({"waypoint_progress": {"completed": path_index, "total": len(path)}} if path is not None else {}),
        **_telemetry_evidence(telemetry_before, telemetry_samples, telemetry_after),
    }
    if chassis_hold is not None:
        receipt["chassis_hold"] = chassis_hold.receipt(raw, hold_corrections)
    return raw, receipt, gripper_open, total_actions


def _execute_base(
    environment: object,
    command: Mapping[str, object],
    *,
    current_gripper: float,
    total_actions: int,
    started: float,
    action_budget: int = EPISODE_ACTION_BUDGET,
) -> tuple[Mapping[str, object], dict[str, object], float, int]:
    required = {
        "schema",
        "sequence",
        "kind",
        "observation_id",
        "axis",
        "normalized_velocity",
        "gripper_open",
        "previous_gripper_open",
        "gripper_transition",
    }
    if set(command) != required:
        raise ValueError("base_action mailbox fields drifted")
    axis = command.get("axis")
    if axis not in {"x", "y", "yaw"}:
        raise ValueError("base axis is invalid")
    velocity = _finite(command.get("normalized_velocity"), label="base velocity")
    if not 0.0 < abs(velocity) <= BASE_VELOCITY_LIMIT:
        raise ValueError("base velocity exceeds its bound")
    gripper_open = _finite(command.get("gripper_open"), label="gripper")
    if not 0.0 <= gripper_open <= 1.0:
        raise ValueError("gripper must be between 0 and 1")
    gripper_changed = abs(gripper_open - current_gripper) > 1e-12
    previous_gripper_open = _finite(
        command.get("previous_gripper_open"), label="previous gripper intent"
    )
    if (
        not 0.0 <= previous_gripper_open <= 1.0
        or abs(previous_gripper_open - current_gripper) > 1e-12
        or command.get("gripper_transition") is not gripper_changed
    ):
        raise ValueError("base_action gripper transition metadata drifted")
    action_count = MIN_GRIPPER_ACTIONS if gripper_changed else BASE_STEPS
    if total_actions + action_count > action_budget:
        if gripper_changed:
            raise RuntimeError(
                f"gripper transition requires {MIN_GRIPPER_ACTIONS} available actions"
            )
        raise RuntimeError("base action exceeds the episode action budget")
    initial = _arm_qpos(environment)
    index = {"x": 0, "y": 1, "yaw": 2}[str(axis)]
    base = [0.0, 0.0, 0.0]
    base[index] = velocity
    actions = [
        {
            "joint_position": initial,
            "gripper_open": gripper_open,
            "base_motion": list(base),
            "torso": 0.0,
        }
        for _ in range(BASE_STEPS)
    ] + [
        {
            "joint_position": initial,
            "gripper_open": gripper_open,
            "base_motion": [0.0, 0.0, 0.0],
            "torso": 0.0,
        }
        for _ in range(action_count - BASE_STEPS)
    ]
    validate_joint_action_sequence(
        actions,
        current_qpos=initial,
        endpoint=initial,
        explicit_mask=[False] * 7,
    )
    telemetry_before = _read_fresh_public_telemetry(environment)
    telemetry_samples: list[dict[str, object]] = []
    raw: Mapping[str, object] | None = None
    for action in actions:
        if time.monotonic() - started > EPISODE_WALL_BUDGET_S:
            raise RuntimeError("base action exceeded the episode wall budget")
        raw = _step(environment, action)
        telemetry_samples.append(_read_fresh_public_telemetry(environment))
        total_actions += 1
    assert raw is not None
    telemetry_after = _read_fresh_public_telemetry(environment)
    realized = _arm_qpos(environment)
    receipt = {
        "kind": "base_action",
        "accepted": True,
        "axis": axis,
        "normalized_velocity": velocity,
        "gripper_intent": gripper_open,
        "step_count": len(actions),
        "base_motion_step_count": BASE_STEPS,
        "realized_arm_qpos": realized,
        "tracking_pause_count": 0,
        "remaining_endpoint_error": max(
            abs(start - end) for start, end in zip(initial, realized, strict=True)
        ),
        **_telemetry_evidence(telemetry_before, telemetry_samples, telemetry_after),
    }
    return raw, receipt, gripper_open, total_actions


def run(
    task: str,
    seed: int,
    run: Path,
    *,
    action_budget: int = EPISODE_ACTION_BUDGET,
) -> None:
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa_inspect.active_skills import TerminalOutcomePersister

    if (
        type(action_budget) is not int
        or not 1 <= action_budget <= MAX_DEVELOPMENT_ACTION_BUDGET
    ):
        raise ValueError("joint simulator action budget is invalid")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    (run / "mailbox").mkdir(mode=0o700)
    (run / "frames").mkdir(mode=0o700)
    episode_tmp = run / "episode-tmp"
    episode_tmp.mkdir(mode=0o700)
    os.environ["ROBOCASA_EPISODE_TMPDIR"] = str(episode_tmp)
    episode = hashlib.sha256(f"{task}:{seed}:{run.name}".encode()).hexdigest()[:32]
    _atomic_json(run / "simulator.json", {
        "schema": "robocasa-inspect-joint-simulator/v1",
        "task": task,
        "seed": seed,
        "episode": episode,
        "network_probe": _network_denied(),
        "action_budget": action_budget,
        "wall_budget_s": EPISODE_WALL_BUDGET_S,
    })
    _install_joint_controller()
    # This renderer frees the old offscreen context before a hard reset.
    environment = gym.make(
        f"robocasa/{task}", split="pretrain", seed=seed, renderer="mujoco"
    )
    terminal: dict[str, object] = {"status": "incomplete"}
    accepted_commands: list[dict[str, object]] = []
    started = time.monotonic()
    total_actions = 0
    gripper_open = 1.0
    try:
        raw, _ = environment.reset()
        chassis_hold = ChassisHold.from_public_state(raw)
        sequence = 0
        observation = _publish_observation(
            raw,
            environment=environment,
            run=run,
            episode=episode,
            sequence=sequence,
            execution=None,
        )
        while True:
            command = _read_command(run, sequence, timeout_s=1_300)
            kind = command.get("kind")
            if kind == "close":
                if set(command) != {"schema", "sequence", "kind"}:
                    raise RuntimeError("close command violates the closed schema")
                terminal = {"status": "closed", "sequence": sequence}
                break
            if kind == "finish":
                if set(command) != {"schema", "sequence", "kind"}:
                    raise RuntimeError("finish command violates the closed schema")
                snapshot_sha256 = _terminal_snapshot_sha256(environment)
                outcome = TerminalOutcomePersister(run).seal(
                    trace=accepted_commands,
                    receipts=[{"sequence": sequence, "accepted": True}],
                    terminal_snapshot_sha256=snapshot_sha256,
                    evaluate=lambda: _official_terminal_success(environment),
                )
                terminal = {
                    "status": "success" if outcome["success"] else "finished_false",
                    "sequence": sequence,
                    "terminal_outcome_sha256": hashlib.sha256(
                        (run / "terminal-outcome.json").read_bytes()
                    ).hexdigest(),
                    "terminal_snapshot_sha256": snapshot_sha256,
                }
                break
            if command.get("observation_id") != observation["observation_id"]:
                raise RuntimeError("joint child received a stale observation command")
            if kind == "move_joints":
                raw, receipt, gripper_open, total_actions = _execute_move(
                    environment,
                    command,
                    current_gripper=gripper_open,
                    total_actions=total_actions,
                    started=started,
                    action_budget=action_budget,
                    chassis_hold=chassis_hold,
                    public_raw=raw,
                )
            elif kind == "base_action":
                raw, receipt, gripper_open, total_actions = _execute_base(
                    environment,
                    command,
                    current_gripper=gripper_open,
                    total_actions=total_actions,
                    started=started,
                    action_budget=action_budget,
                )
                chassis_hold.reset_after_base_action(raw)
                receipt["chassis_hold"] = chassis_hold.receipt(raw, [], reset=True)
            else:
                raise RuntimeError("invalid joint simulator mailbox command")
            accepted_commands.append({
                "sequence": sequence,
                "kind": kind,
                "command_sha256": hashlib.sha256(
                    json.dumps(command, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "step_count": receipt["step_count"],
            })
            sequence += 1
            observation = _publish_observation(
                raw,
                environment=environment,
                run=run,
                episode=episode,
                sequence=sequence,
                execution=receipt,
            )
    except Exception as error:
        terminal = {"status": "infrastructure_blocked", "error": repr(error)}
        raise
    finally:
        environment.close()
        terminal["episode_tmp_empty"] = not any(episode_tmp.iterdir())
        terminal["simulator_actions"] = total_actions
        terminal["wall_s"] = time.monotonic() - started
        _atomic_json(run / "simulator-terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--action-budget", type=int, default=EPISODE_ACTION_BUDGET
    )
    args = parser.parse_args()
    run(
        args.task,
        args.seed,
        args.run_dir.resolve(),
        action_budget=args.action_budget,
    )


if __name__ == "__main__":
    main()
