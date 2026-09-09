"""H200 no-contact audit of target-only grounding and RGB-derived gripper servo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .active_skills import ActiveSkillSupervisor, decode_skill_command
from .authority import current_execution_authority, validate_execution_authority
from .base_active import base_pulse_chunk, torso_pulse_chunk
from .button_skill import (
    PRESS_TASKS,
    VERTICAL_PRESS_TASKS,
    annotate_microwave_button_pair,
    button_candidate_system_prompt,
    button_retreat_goal,
    decode_button_candidate,
    detect_microwave_button_pair,
    preserve_tracked_button_identity,
    refine_task_microwave_button,
    task_press_direction_world,
)
from .camera_geometry import (
    best_strict_triangulation,
    project_world_point,
    triangulate_two_view_point,
)
from .collision_planner import plan_collision_free_cartesian_path
from .contact_skill import confirm_public_motion_contact
from .continuation_skill import (
    continuation_base_request,
    continuation_finish_gate,
    continuation_retreat_goal,
)
from .contracts import Command, official_action_chunk
from .geometric_servo import (
    PANDA_GRIPPER_PLUS_Z_SUPPORT_M,
    PUBLIC_RGB_STANDOFF_MARGIN_M,
    base_reposition_request,
    bounded_base_frame_delta,
    bounded_orientation_delta,
    downward_push_site_orientation_world_xyzw,
    final_push_direction_after_base_reposition,
    grasp_site_orientation_for_direction_world_xyzw,
    orient_surface_normal_from_base,
    planned_waypoint_requires_abort,
    public_eef_world,
    push_site_orientation_for_direction_world_xyzw,
    waypoint_increment_budget,
    world_quaternion_to_base_xyzw,
)
from .gripper_articulation import locate_gripper_anchor
from .handle_refinement import handle_axis_endpoints, refine_horizontal_metal_handle
from .harness import ServoExecution, ServoObservation
from .harness_evidence import seal_harness_evidence
from .model_client import QwenClient, load_authority, verify_process
from .pull_skill import (
    grasp_search_entry_depths,
    public_grasp_aperture_m,
    public_grasp_occupied,
    pull_finish_gate,
    pull_grasp_entry_goal,
    pull_target,
)
from .runner import (
    _atomic_json,
    _launch_simulator,
    _read_images,
    _render_video,
    _serialize_actions,
    _wait_process_json,
)
from .single_view_harness import run_single_view_center
from .surface_skill import refine_trackable_surface_anchor
from .task_recipes import recipe_system_prompt, task_recipe
from .visual_completion import (
    close_drawer_verifier_prompt,
    decode_visual_completion,
    decode_visual_open_completion,
    open_drawer_verifier_prompt,
)
from .visual_servo import (
    normalized_to_pixel,
    parallax_direction_from_probe,
    track_points,
)


def _servo_observation(observation: dict[str, object]) -> ServoObservation:
    state = observation["public_state"]
    return ServoObservation(
        observation_id=str(observation["observation_id"]),
        left=np.asarray(Image.open(observation["images"]["left"]["path"]).convert("RGB")),
        right=np.asarray(
            Image.open(observation["images"]["right"]["path"]).convert("RGB")
        ),
        eef_position_m=np.asarray(
            state["state.end_effector_position_relative"], dtype=np.float64
        ),
        finger_qpos=np.asarray(state["state.gripper_qpos"], dtype=np.float64),
    )


def run_audit(
    *,
    run: Path,
    task: str = "CloseDrawer",
    seed: int = 7,
    center: bool = False,
    base_precenter: bool = False,
    geometry_center: bool = False,
    contact_trial: bool = False,
) -> dict[str, object]:
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    project_root = Path(__file__).resolve().parents[1]
    authority_paths = {
        "project_root": project_root,
        "robocasa_root": project_root / "robocasa",
        "robosuite_root": project_root / "robosuite",
        "asset_manifest_path": Path(
            os.environ.get(
                "ROBOCASA_ASSET_MANIFEST",
                "/home/jli/state/robocasa-inspect-official/assets/content-manifest.json",
            )
        ),
        "identity_path": Path(os.environ["QWEN_IDENTITY_MANIFEST"]),
        "attestation_path": Path(os.environ["QWEN_SERVER_ATTESTATION"]),
    }
    execution_authority = current_execution_authority(**authority_paths)
    authority_path = run / "execution-authority.json"
    _atomic_json(authority_path, execution_authority)
    log_path = run / "simulator.log"
    log = log_path.open("w", encoding="utf-8")
    os.chmod(log_path, 0o600)
    model_identity, attestation = load_authority(
        Path(os.environ["QWEN_IDENTITY_MANIFEST"]),
        Path(os.environ["QWEN_SERVER_ATTESTATION"]),
    )
    client = QwenClient(
        base_url="http://127.0.0.1:8002/v1",
        api_key=Path(os.environ["QWEN_API_TOKEN_FILE"]).read_text().strip(),
        identity=model_identity,
        attestation=attestation,
    )
    client.verify()
    process = _launch_simulator(task=task, seed=seed, run=run, log_handle=log)
    sequence = 0
    action_chunks = 0
    receipts: list[dict[str, object]] = []
    supervisor = ActiveSkillSupervisor(max_model_calls=40)
    recipe = task_recipe(task)
    terminal_issued = False
    official_success = False

    def send_action(
        delta: tuple[float, float, float],
        gripper: str,
        label: str,
        *,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        nonlocal sequence, action_chunks
        command = Command(
            kind="action",
            observation_id="harness-owned",
            translation_m=delta,
            rotation_axis_angle_rad=rotation,
            gripper=gripper,
        )
        actions, _ = official_action_chunk(command, previous_gripper="open")
        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-command/v1",
                "sequence": sequence,
                "kind": "action",
                "actions": _serialize_actions(actions),
            },
        )
        receipts.append(
            {
                "sequence": sequence,
                "label": label,
                "translation_m": list(delta),
                "gripper": gripper,
                "accepted": True,
            }
        )
        sequence += 1
        action_chunks += 1
        return _wait_process_json(
            run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
            process=process,
            log_path=log_path,
            timeout_s=180,
        )

    def send_base(
        axis: str, direction: int, label: str, *, velocity: float = 1.0
    ):
        nonlocal sequence, action_chunks
        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-command/v1",
                "sequence": sequence,
                "kind": "action",
                "actions": _serialize_actions(
                    base_pulse_chunk(
                        axis=axis,
                        normalized_velocity=float(direction) * velocity,
                    )
                ),
            },
        )
        receipts.append(
            {
                "sequence": sequence,
                "label": label,
                "base_axis": axis,
                "direction": direction,
                "normalized_velocity": velocity,
                "accepted": True,
            }
        )
        sequence += 1
        action_chunks += 1
        return _wait_process_json(
            run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
            process=process,
            log_path=log_path,
            timeout_s=180,
        )

    def send_torso(direction: int, label: str):
        nonlocal sequence, action_chunks
        velocity = float(direction) * 0.25
        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-command/v1",
                "sequence": sequence,
                "kind": "action",
                "actions": _serialize_actions(
                    torso_pulse_chunk(normalized_velocity=velocity)
                ),
            },
        )
        receipts.append(
            {
                "sequence": sequence,
                "label": label,
                "torso_direction": direction,
                "normalized_velocity": abs(velocity),
                "accepted": True,
            }
        )
        sequence += 1
        action_chunks += 1
        return _wait_process_json(
            run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
            process=process,
            log_path=log_path,
            timeout_s=180,
        )

    try:
        reset = _wait_process_json(
            run / "sim" / "mailbox" / "observation-000000.json",
            process=process,
            log_path=log_path,
            timeout_s=180,
        )
        opened = _servo_observation(reset)
        observation = reset
        for index in range(5):
            observation = send_action(
                (0.0, 0.0, 0.0), "close", f"gripper_close_{index + 1}"
            )
        closed = _servo_observation(observation)
        for index in range(5):
            observation = send_action(
                (0.0, 0.0, 0.0), "open", f"gripper_open_{index + 1}"
            )
        returned = _servo_observation(observation)
        supervisor.record_model_call()
        response = client.complete(
            observation_id=str(observation["observation_id"]),
            system_prompt=recipe_system_prompt(task),
            instruction=json.dumps(
                {
                    "task": observation["instruction"],
                    "required_view": "choose_the_visible_official_external_view",
                    "harness_stage": "no_contact_calibration_audit",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            public_state=observation["public_state"],
            images=_read_images(observation),
        )
        command = decode_skill_command(
            response.command,
            observation_id=str(observation["observation_id"]),
        )
        if (
            command.kind != "center_feature"
            or command.view
            not in {"robot0_agentview_left", "robot0_agentview_right"}
            or command.feature_kind != recipe.feature_kind
            or command.feature_uv is None
        ):
            raise RuntimeError("Qwen target-only citation drifted")

        selected_view = (
            "left" if command.view == "robot0_agentview_left" else "right"
        )
        anchor = locate_gripper_anchor(
            getattr(opened, selected_view),
            getattr(closed, selected_view),
            getattr(returned, selected_view),
            static_noise_px=0.5,
        )

        def execute(delta, label):
            nonlocal observation, sequence
            if delta is None:
                _atomic_json(
                    run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                    {
                        "schema": "robocasa-inspect-command/v1",
                        "sequence": sequence,
                        "kind": "observe",
                    },
                )
                sequence += 1
                observation = _wait_process_json(
                    run
                    / "sim"
                    / "mailbox"
                    / f"observation-{sequence:06d}.json",
                    process=process,
                    log_path=log_path,
                    timeout_s=180,
                )
            else:
                start_eef = np.asarray(
                    observation["public_state"][
                        "state.end_effector_position_relative"
                    ],
                    dtype=np.float64,
                )
                repeats = 5 if label.startswith("center_") else 1
                for repeat in range(repeats):
                    suffix = f"_{repeat + 1}" if repeats > 1 else ""
                    observation = send_action(delta, "open", f"{label}{suffix}")
                    realized = np.linalg.norm(
                        np.asarray(
                            observation["public_state"][
                                "state.end_effector_position_relative"
                            ],
                            dtype=np.float64,
                        )
                        - start_eef
                    )
                    if realized >= 0.5 * np.linalg.norm(delta):
                        break
            return _servo_observation(observation)

        target = normalized_to_pixel(command.feature_uv)
        visual_refinement: dict[str, object] | None = None
        initial_handle_endpoints: np.ndarray | None = None
        button_pair_locked = False
        visual_target_authority = (
            "qwen_semantic_citation_plus_public_rgb_refinement"
        )
        if command.feature_kind == "handle":
            refined = refine_horizontal_metal_handle(
                getattr(returned, selected_view), target
            )
            target = np.asarray(refined.anchor_uv_px, dtype=np.float64)
            initial_handle_endpoints = handle_axis_endpoints(refined)
            visual_refinement = {
                "kind": "horizontal_metal_handle",
                "anchor_uv_px": list(refined.anchor_uv_px),
                "bounding_box_xywh": list(refined.bounding_box_xywh),
                "area_px": refined.area_px,
            }
        elif command.feature_kind == "button":
            selected_image = getattr(returned, selected_view)
            button_pair = detect_microwave_button_pair(selected_image, target)
            if button_pair is None:
                refined_button = refine_task_microwave_button(
                    selected_image, target, task=task
                )
                target = np.asarray(
                    refined_button.anchor_uv_px, dtype=np.float64
                )
                visual_refinement = {
                    "kind": "neutral_microwave_button",
                    "anchor_uv_px": list(refined_button.anchor_uv_px),
                    "area_px": refined_button.area_px,
                    "semantic_disagreement_px": (
                        refined_button.semantic_disagreement_px
                    ),
                }
            else:
                supervisor.record_model_call()
                detail_images = _read_images(observation)
                detail_images["wrist"] = annotate_microwave_button_pair(
                    selected_image, button_pair
                )
                detail_state = dict(observation["public_state"])
                detail_state["derived_panel_zoom"] = {
                    "source_view": selected_view,
                    "crop_xywh": list(button_pair.crop_xywh),
                    "candidates": [
                        {
                            "label": candidate.label,
                            "center_uv_px": list(candidate.center_uv_px),
                            "area_px": candidate.area_px,
                        }
                        for candidate in button_pair.candidates
                    ],
                }
                detail_response = client.complete(
                    observation_id=str(observation["observation_id"]),
                    system_prompt=button_candidate_system_prompt(task),
                    instruction=(
                        "Select the visible control that stops the microwave."
                        if task == "TurnOffMicrowave"
                        else "Select the visible control that starts the microwave."
                    ),
                    public_state=detail_state,
                    images=detail_images,
                    image_roles={
                        "left": "official_left",
                        "right": "official_right",
                        "wrist": "derived_annotated_panel_zoom",
                    },
                )
                selected_candidate = decode_button_candidate(
                    detail_response.command,
                    observation_id=str(observation["observation_id"]),
                )
                candidate = next(
                    candidate
                    for candidate in button_pair.candidates
                    if candidate.label == selected_candidate
                )
                target = np.asarray(candidate.center_uv_px, dtype=np.float64)
                button_pair_locked = True
                visual_target_authority = (
                    "qwen_closed_pair_plus_public_rgb_candidates"
                )
                visual_refinement = {
                    "kind": "dense_keypad_closed_button_pair",
                    "layout": button_pair.layout,
                    "crop_xywh": list(button_pair.crop_xywh),
                    "selected_candidate": selected_candidate,
                    "anchor_uv_px": list(candidate.center_uv_px),
                    "candidates": [
                        {
                            "label": item.label,
                            "center_uv_px": list(item.center_uv_px),
                            "area_px": item.area_px,
                        }
                        for item in button_pair.candidates
                    ],
                    "qwen_evidence": detail_response.evidence,
                }
        elif command.feature_kind == "front_surface":
            refined_surface = refine_trackable_surface_anchor(
                getattr(returned, selected_view), target
            )
            target = np.asarray(refined_surface.anchor_uv_px, dtype=np.float64)
            visual_refinement = {
                "kind": "trackable_surface_corner",
                "anchor_uv_px": list(refined_surface.anchor_uv_px),
                "semantic_disagreement_px": (
                    refined_surface.semantic_disagreement_px
                ),
                "candidate_count": refined_surface.candidate_count,
            }
        base_report: list[dict[str, object]] = []
        geometry_report: dict[str, object] = {}
        gripper_point = np.asarray(anchor.anchor_uv_px)
        parallax_start = observation
        parallax_start_handle_endpoints = (
            None
            if initial_handle_endpoints is None
            else initial_handle_endpoints.copy()
        )
        tracked_handle_endpoints = parallax_start_handle_endpoints
        if base_precenter:
            supervisor.accept_motion("scan_view")
            points = np.asarray(
                [
                    target,
                    gripper_point,
                    *(
                        []
                        if initial_handle_endpoints is None
                        else list(initial_handle_endpoints)
                    ),
                ],
                dtype=np.float64,
            )
            current_servo = returned
            previous_error = float(np.linalg.norm(points[0] - points[1]))
            base_limit = 180
            base_velocity = 0.25
            for index in range(base_limit):
                next_observation = send_base(
                    "y",
                    1,
                    f"base_precenter_y_{index + 1}",
                    velocity=base_velocity,
                )
                next_servo = _servo_observation(next_observation)
                try:
                    tracked = track_points(
                        getattr(current_servo, selected_view),
                        getattr(next_servo, selected_view),
                        points,
                    )
                except ValueError as error:
                    base_report.append(
                        {
                            "index": index + 1,
                            "failure": str(error),
                            "accepted_motion": True,
                        }
                    )
                    break
                next_points = tracked.points
                if len(next_points) == 4:
                    reacquired = refine_horizontal_metal_handle(
                        getattr(next_servo, selected_view), next_points[0]
                    )
                    reacquired_target = np.asarray(
                        reacquired.anchor_uv_px, dtype=np.float64
                    )
                    if np.linalg.norm(reacquired_target - next_points[0]) > 8.0:
                        raise RuntimeError("horizontal handle identity drifted")
                    next_points = np.asarray(
                        [
                            reacquired_target,
                            next_points[1],
                            *list(handle_axis_endpoints(reacquired)),
                        ],
                        dtype=np.float64,
                    )
                elif command.feature_kind == "button" and not button_pair_locked:
                    refined_button = refine_task_microwave_button(
                        getattr(next_servo, selected_view), next_points[0], task=task
                    )
                    refined_target = np.asarray(
                        refined_button.anchor_uv_px, dtype=np.float64
                    )
                    if np.linalg.norm(refined_target - next_points[0]) > 8.0:
                        raise RuntimeError("microwave button identity drifted")
                    next_points[0] = refined_target
                feature_step_px = float(
                    np.linalg.norm(next_points[0] - points[0])
                )
                if geometry_center and feature_step_px > 5.0:
                    raise RuntimeError("parallax feature step exceeds five pixels")
                error = float(np.linalg.norm(next_points[0] - next_points[1]))
                base_report.append(
                    {
                        "index": index + 1,
                        "error_before_px": previous_error,
                        "error_after_px": error,
                        "forward_backward_error_px": tracked.max_forward_backward_error_px,
                        "feature_step_px": feature_step_px,
                    }
                )
                observation = next_observation
                current_servo = next_servo
                points = next_points
                if error >= previous_error - 0.05:
                    break
                previous_error = error
                if error <= 70.0:
                    break
            returned = current_servo
            target = points[0]
            gripper_point = points[1]
            if len(points) == 2:
                tracked_handle_endpoints = None
                if command.feature_kind == "button" and not button_pair_locked:
                    refined_button = refine_task_microwave_button(
                        getattr(current_servo, selected_view), target, task=task
                    )
                    target = np.asarray(
                        refined_button.anchor_uv_px, dtype=np.float64
                    )
            else:
                reacquired = refine_horizontal_metal_handle(
                    getattr(current_servo, selected_view), target
                )
                reacquired_target = np.asarray(
                    reacquired.anchor_uv_px, dtype=np.float64
                )
                if np.linalg.norm(reacquired_target - target) > 8.0:
                    raise RuntimeError("horizontal handle identity drifted")
                target = reacquired_target
                tracked_handle_endpoints = handle_axis_endpoints(reacquired)
        geometry_parallax_report: list[dict[str, object]] = []
        parallax_samples: list[dict[str, object]] = []
        semantic_reacquisition: dict[str, object] | None = None
        if geometry_center:
            supervisor.record_model_call()
            response = client.complete(
                observation_id=str(observation["observation_id"]),
                system_prompt=recipe_system_prompt(task),
                instruction=json.dumps(
                    {
                        "task": observation["instruction"],
                        "required_view": "choose_the_clearest_official_external_view",
                        "harness_stage": "post_scan_semantic_reacquisition",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                public_state=observation["public_state"],
                images=_read_images(observation),
            )
            reacquisition = decode_skill_command(
                response.command,
                observation_id=str(observation["observation_id"]),
            )
            if (
                reacquisition.kind != "center_feature"
                or reacquisition.view
                not in {"robot0_agentview_left", "robot0_agentview_right"}
                or reacquisition.feature_kind != recipe.feature_kind
                or reacquisition.feature_uv is None
            ):
                raise RuntimeError("post-scan Qwen citation drifted")
            advisory_view = (
                "left"
                if reacquisition.view == "robot0_agentview_left"
                else "right"
            )
            semantic_target = normalized_to_pixel(reacquisition.feature_uv)
            if recipe.feature_kind == "handle":
                selected_view = advisory_view
                reacquired = refine_horizontal_metal_handle(
                    getattr(_servo_observation(observation), selected_view),
                    semantic_target,
                )
                target = np.asarray(reacquired.anchor_uv_px, dtype=np.float64)
                tracked_handle_endpoints = handle_axis_endpoints(reacquired)
                refined_box: list[int] | None = list(
                    reacquired.bounding_box_xywh
                )
                identity_authority = "post_scan_qwen_plus_public_rgb_refinement"
                advisory_consistent = True
                advisory_disagreement_px = 0.0
            elif recipe.feature_kind == "button":
                button_identity = preserve_tracked_button_identity(
                    view=selected_view,
                    tracked_uv_px=target,
                    advisory_view=advisory_view,
                    advisory_uv_px=semantic_target,
                    authority=(
                        "initial_qwen_closed_pair_plus_public_optical_flow"
                        if button_pair_locked
                        else "initial_qwen_plus_public_optical_flow"
                    ),
                )
                selected_view = button_identity.view
                if button_pair_locked:
                    target = np.asarray(
                        button_identity.target_uv_px, dtype=np.float64
                    )
                else:
                    refined_button = refine_task_microwave_button(
                        getattr(_servo_observation(observation), selected_view),
                        button_identity.target_uv_px,
                        task=task,
                    )
                    target = np.asarray(
                        refined_button.anchor_uv_px, dtype=np.float64
                    )
                tracked_handle_endpoints = None
                refined_box = None
                identity_authority = button_identity.authority
                advisory_consistent = button_identity.advisory_consistent
                advisory_disagreement_px = button_identity.advisory_disagreement_px
            else:
                selected_view = advisory_view
                if recipe.feature_kind == "front_surface":
                    refined_surface = refine_trackable_surface_anchor(
                        getattr(_servo_observation(observation), selected_view),
                        semantic_target,
                    )
                    target = np.asarray(
                        refined_surface.anchor_uv_px, dtype=np.float64
                    )
                    refined_box = None
                    identity_authority = (
                        "post_scan_qwen_plus_public_rgb_surface_corner"
                    )
                else:
                    target = np.asarray(semantic_target, dtype=np.float64)
                    refined_box = None
                    identity_authority = "post_scan_qwen"
                tracked_handle_endpoints = None
                advisory_consistent = True
                advisory_disagreement_px = 0.0
            semantic_reacquisition = {
                "view": selected_view,
                "advisory_view": advisory_view,
                "qwen_uv": list(reacquisition.feature_uv),
                "refined_anchor_uv_px": target.tolist(),
                "refined_bounding_box_xywh": refined_box,
                "identity_authority": identity_authority,
                "advisory_consistent": advisory_consistent,
                "advisory_disagreement_px": advisory_disagreement_px,
                "qwen_evidence": response.evidence,
            }
            parallax_start = observation
            parallax_samples = [
                {
                    "calibration": observation["camera_calibration"][selected_view],
                    "uv_px": target.tolist(),
                }
            ]
            parallax_start_handle_endpoints = (
                None
                if tracked_handle_endpoints is None
                else tracked_handle_endpoints.copy()
            )
            points = np.asarray(
                [
                    target,
                    *(
                        []
                        if tracked_handle_endpoints is None
                        else list(tracked_handle_endpoints)
                    ),
                ],
                dtype=np.float64,
            )
            current_servo = _servo_observation(observation)
            start_camera = np.asarray(
                parallax_start["camera_calibration"][selected_view][
                    "camera_position_world_m"
                ],
                dtype=np.float64,
            )
            parallax_direction = 1
            for index in range(100):
                executed_direction = parallax_direction
                next_observation = send_base(
                    "y",
                    executed_direction,
                    f"geometry_parallax_y_{index + 1}",
                    velocity=0.25,
                )
                next_servo = _servo_observation(next_observation)
                tracked = track_points(
                    getattr(current_servo, selected_view),
                    getattr(next_servo, selected_view),
                    points,
                )
                if index == 0:
                    parallax_direction = parallax_direction_from_probe(
                        points[0],
                        tracked.points[0],
                        probe_direction=executed_direction,
                    )
                if tracked.max_forward_backward_error_px > 1.5:
                    raise RuntimeError("geometry parallax tracking drifted")
                step_px = float(np.max(np.linalg.norm(tracked.points - points, axis=1)))
                if step_px > 5.0:
                    raise RuntimeError("geometry parallax step exceeds five pixels")
                if len(points) == 3:
                    reacquired = refine_horizontal_metal_handle(
                        getattr(next_servo, selected_view), tracked.points[0]
                    )
                    reacquired_target = np.asarray(
                        reacquired.anchor_uv_px, dtype=np.float64
                    )
                    if np.linalg.norm(reacquired_target - tracked.points[0]) > 8.0:
                        raise RuntimeError("horizontal handle identity drifted")
                    next_points = np.asarray(
                        [reacquired_target, *list(handle_axis_endpoints(reacquired))],
                        dtype=np.float64,
                    )
                elif recipe.feature_kind == "button" and not button_pair_locked:
                    refined_button = refine_task_microwave_button(
                        getattr(next_servo, selected_view), tracked.points[0], task=task
                    )
                    refined_target = np.asarray(
                        refined_button.anchor_uv_px, dtype=np.float64
                    )
                    if np.linalg.norm(refined_target - tracked.points[0]) > 8.0:
                        raise RuntimeError("microwave button identity drifted")
                    next_points = np.asarray([refined_target], dtype=np.float64)
                else:
                    next_points = tracked.points
                observation = next_observation
                current_servo = next_servo
                points = next_points
                parallax_samples.append(
                    {
                        "calibration": observation["camera_calibration"][selected_view],
                        "uv_px": points[0].tolist(),
                    }
                )
                current_camera = np.asarray(
                    observation["camera_calibration"][selected_view][
                        "camera_position_world_m"
                    ],
                    dtype=np.float64,
                )
                baseline_m = float(np.linalg.norm(current_camera - start_camera))
                geometry_parallax_report.append(
                    {
                        "index": index + 1,
                        "executed_direction": executed_direction,
                        "selected_direction": parallax_direction,
                        "baseline_m": baseline_m,
                        "max_forward_backward_error_px": (
                            tracked.max_forward_backward_error_px
                        ),
                        "max_feature_step_px": step_px,
                    }
                )
                if baseline_m >= 0.12:
                    break
            else:
                raise RuntimeError("geometry parallax baseline exhausted its budget")
            target = points[0]
            tracked_handle_endpoints = (
                None if len(points) == 1 else points[1:3].copy()
            )
        if geometry_center:
            if not base_precenter:
                raise ValueError("geometry centering requires a parallax base scan")
            triangulated, triangulation_pair = best_strict_triangulation(
                parallax_samples,
                base_position_world_m=observation["public_state"][
                    "state.base_position"
                ],
            )
            geometry_report = {
                "schema": "robocasa-public-geometric-servo/v1",
                "target_world_m": list(triangulated.point_world_m),
                "baseline_m": triangulated.baseline_m,
                "ray_angle_deg": triangulated.ray_angle_deg,
                "reprojection_error_px": list(
                    triangulated.reprojection_error_px
                ),
                "triangulation_sample_pair": list(triangulation_pair),
                "target_authority": visual_target_authority,
                "visual_refinement": visual_refinement,
                "semantic_reacquisition": semantic_reacquisition,
                "dedicated_parallax": geometry_parallax_report,
                "waypoints": [],
            }
            target_world = np.asarray(triangulated.point_world_m, dtype=np.float64)
            base_world = np.asarray(
                observation["public_state"]["state.base_position"], dtype=np.float64
            )
            if (
                parallax_start_handle_endpoints is not None
                and tracked_handle_endpoints is not None
            ):
                handle_world = [
                    triangulate_two_view_point(
                        parallax_start["camera_calibration"][selected_view],
                        parallax_start_handle_endpoints[index],
                        observation["camera_calibration"][selected_view],
                        tracked_handle_endpoints[index],
                        base_position_world_m=observation["public_state"][
                            "state.base_position"
                        ],
                    )
                    for index in range(2)
                ]
                tangent = (
                    np.asarray(handle_world[1].point_world_m, dtype=np.float64)
                    - np.asarray(handle_world[0].point_world_m, dtype=np.float64)
                )
                height_delta_m = abs(
                    float(handle_world[1].point_world_m[2])
                    - float(handle_world[0].point_world_m[2])
                )
                if height_delta_m > 0.03:
                    raise RuntimeError("triangulated horizontal handle is inconsistent")
                tangent[2] = 0.0
                horizontal_separation_m = float(np.linalg.norm(tangent))
                if not 0.02 <= horizontal_separation_m <= 0.15:
                    raise RuntimeError("triangulated handle axis length is inconsistent")
                tangent /= np.linalg.norm(tangent)
                push_direction = orient_surface_normal_from_base(
                    np.cross(tangent, np.asarray([0.0, 0.0, 1.0])),
                    target_world,
                    base_world,
                )
                geometry_report["handle_axis"] = {
                    "start_pixels": parallax_start_handle_endpoints.tolist(),
                    "end_pixels": tracked_handle_endpoints.tolist(),
                    "world_endpoints_m": [
                        list(point.point_world_m) for point in handle_world
                    ],
                    "world_tangent": tangent.tolist(),
                    "endpoint_height_delta_m": height_delta_m,
                    "horizontal_separation_m": horizontal_separation_m,
                    "rgb_derived_push_direction": push_direction.tolist(),
                }
            else:
                push_direction = target_world - base_world
                push_direction[2] = 0.0
                push_direction /= np.linalg.norm(push_direction)
                geometry_report["handle_axis"] = None
            base_reposition: list[dict[str, object]] = []
            for index in range(550 if recipe.workspace_reposition else 0):
                request = base_reposition_request(
                    observation["public_state"],
                    triangulated.point_world_m,
                    desired_forward_m=recipe.workspace_forward_m,
                )
                if request is None:
                    break
                axis, direction = request
                before = np.asarray(
                    observation["public_state"]["state.base_position"],
                    dtype=np.float64,
                )
                observation = send_base(
                    axis,
                    direction,
                    f"base_workspace_{axis}_{index + 1}",
                    velocity=0.25,
                )
                after = np.asarray(
                    observation["public_state"]["state.base_position"],
                    dtype=np.float64,
                )
                realized = float(np.linalg.norm(after - before))
                if realized <= 1e-5:
                    raise RuntimeError("base workspace reposition made no progress")
                base_reposition.append(
                    {
                        "index": index + 1,
                        "axis": axis,
                        "direction": direction,
                        "base_before_m": before.tolist(),
                        "base_after_m": after.tolist(),
                        "realized_m": realized,
                    }
                )
            else:
                if recipe.workspace_reposition:
                    raise RuntimeError("base workspace reposition exhausted its budget")
            geometry_report["base_reposition"] = base_reposition
            geometry_report["base_reposition_enabled"] = recipe.workspace_reposition
            geometry_report["torso_reposition"] = {
                "enabled": False,
                "reason": "official torso starts at its measured lower limit",
            }
            final_base_world = np.asarray(
                observation["public_state"]["state.base_position"], dtype=np.float64
            )
            if task in VERTICAL_PRESS_TASKS:
                push_direction = task_press_direction_world(task, push_direction)
            else:
                push_direction = final_push_direction_after_base_reposition(
                    initial_direction=push_direction,
                    target_world_m=target_world,
                    final_base_world_m=final_base_world,
                    preserve_rgb_surface_normal=(
                        geometry_report["handle_axis"] is not None
                    ),
                )
            if geometry_report["handle_axis"] is not None:
                geometry_report["handle_axis"][
                    "rgb_derived_push_direction"
                ] = push_direction.tolist()
            standoff_distance_m = (
                PANDA_GRIPPER_PLUS_Z_SUPPORT_M + PUBLIC_RGB_STANDOFF_MARGIN_M
            )
            standoff = target_world - standoff_distance_m * push_direction
            if task in VERTICAL_PRESS_TASKS:
                push_orientation = downward_push_site_orientation_world_xyzw(
                    roll_sign=1
                )
            elif recipe.manipulation == "pull":
                push_orientation = grasp_site_orientation_for_direction_world_xyzw(
                    push_direction, vertical_sign=-1
                )
            else:
                push_orientation = push_site_orientation_for_direction_world_xyzw(
                    push_direction,
                    vertical_sign=-1,
                )
            plan = plan_collision_free_cartesian_path(
                task=task,
                seed=seed,
                target_world_m=standoff,
                base_world_m=observation["public_state"]["state.base_position"],
                orientation_world_xyzw=push_orientation,
                run=run,
            )
            measured_support = float(plan["gripper_axis_support_m"]["z_plus"])
            if not np.isclose(
                measured_support,
                PANDA_GRIPPER_PLUS_Z_SUPPORT_M,
                atol=1e-6,
                rtol=0,
            ):
                raise RuntimeError("official gripper push support drifted")
            geometry_report["collision_plan"] = {
                "planned_joint_path_nodes": plan["planned_joint_path_nodes"],
                "cartesian_waypoint_count": len(plan["cartesian_path"]),
                "result_sha256": hashlib.sha256(
                    (run / "collision-plan/result.json").read_bytes()
                ).hexdigest(),
                "requested_push_site_orientation_world_xyzw": push_orientation.tolist(),
                "measured_gripper_plus_z_support_m": measured_support,
                "standoff_margin_m": PUBLIC_RGB_STANDOFF_MARGIN_M,
                "steps": [],
            }
            geometry_status = "geometric_standoff_reached"
            final_target_position: np.ndarray | None = None
            final_target_orientation: np.ndarray | None = None
            for waypoint_index, waypoint in enumerate(plan["cartesian_path"][1:], 1):
                target_position = np.asarray(
                    waypoint["eef_position_world_m"], dtype=np.float64
                )
                desired_relative = world_quaternion_to_base_xyzw(
                    observation["public_state"],
                    waypoint["eef_orientation_world_xyzw"],
                )
                final_target_position = target_position
                final_target_orientation = np.asarray(
                    waypoint["eef_orientation_world_xyzw"], dtype=np.float64
                )
                waypoint_report: dict[str, object] = {
                    "waypoint_index": waypoint_index,
                    "target_world_m": target_position.tolist(),
                    "target_relative_xyzw": desired_relative.tolist(),
                    "increments": [],
                }
                initial_position_error = float(
                    np.linalg.norm(
                        public_eef_world(observation["public_state"])
                        - target_position
                    )
                )
                increment_budget = waypoint_increment_budget(
                    initial_position_error
                )
                waypoint_report["initial_position_error_m"] = (
                    initial_position_error
                )
                waypoint_report["increment_budget"] = increment_budget
                for increment in range(increment_budget):
                    delta, position_before = bounded_base_frame_delta(
                        observation["public_state"], target_position
                    )
                    rotation, orientation_before = bounded_orientation_delta(
                        observation["public_state"], desired_relative
                    )
                    if position_before <= 0.025 and orientation_before <= 0.04:
                        break
                    observation = send_action(
                        tuple(float(value) for value in delta),
                        "open",
                        f"planned_waypoint_{waypoint_index}_{increment + 1}",
                        rotation=tuple(float(value) for value in rotation),
                    )
                    _, orientation_after = bounded_orientation_delta(
                        observation["public_state"], desired_relative
                    )
                    position_after = float(
                        np.linalg.norm(
                            public_eef_world(observation["public_state"])
                            - target_position
                        )
                    )
                    waypoint_report["increments"].append(
                        {
                            "position_before_m": position_before,
                            "position_after_m": position_after,
                            "orientation_before_rad": orientation_before,
                            "orientation_after_rad": orientation_after,
                        }
                    )
                final_position = float(
                    np.linalg.norm(
                        public_eef_world(observation["public_state"])
                        - target_position
                    )
                )
                _, final_orientation = bounded_orientation_delta(
                    observation["public_state"], desired_relative
                )
                waypoint_report["final_position_error_m"] = final_position
                waypoint_report["final_orientation_error_rad"] = final_orientation
                geometry_report["collision_plan"]["steps"].append(waypoint_report)
                if planned_waypoint_requires_abort(
                    is_final=(
                        waypoint_index == len(plan["cartesian_path"]) - 1
                    ),
                    position_error_m=final_position,
                    orientation_error_rad=final_orientation,
                ):
                    geometry_status = "geometric_planned_waypoint_failed"
                    break
            endpoint_settle: list[dict[str, float]] = []
            if (
                geometry_status == "geometric_standoff_reached"
                and final_target_position is not None
                and final_target_orientation is not None
            ):
                for increment in range(24):
                    desired_relative = world_quaternion_to_base_xyzw(
                        observation["public_state"], final_target_orientation
                    )
                    delta, position_before = bounded_base_frame_delta(
                        observation["public_state"], final_target_position
                    )
                    rotation, orientation_before = bounded_orientation_delta(
                        observation["public_state"], desired_relative
                    )
                    if position_before <= 0.012 and orientation_before <= 0.06:
                        break
                    observation = send_action(
                        tuple(float(value) for value in delta),
                        "open",
                        f"standoff_settle_{increment + 1}",
                        rotation=tuple(float(value) for value in rotation),
                    )
                    desired_relative = world_quaternion_to_base_xyzw(
                        observation["public_state"], final_target_orientation
                    )
                    _, orientation_after = bounded_orientation_delta(
                        observation["public_state"], desired_relative
                    )
                    position_after = float(
                        np.linalg.norm(
                            public_eef_world(observation["public_state"])
                            - final_target_position
                        )
                    )
                    endpoint_settle.append(
                        {
                            "position_before_m": position_before,
                            "position_after_m": position_after,
                            "orientation_before_rad": orientation_before,
                            "orientation_after_rad": orientation_after,
                        }
                    )
                final_position = float(
                    np.linalg.norm(
                        public_eef_world(observation["public_state"])
                        - final_target_position
                    )
                )
                desired_relative = world_quaternion_to_base_xyzw(
                    observation["public_state"], final_target_orientation
                )
                _, final_orientation = bounded_orientation_delta(
                    observation["public_state"], desired_relative
                )
                if final_position > 0.012 or final_orientation > 0.06:
                    geometry_status = "geometric_standoff_settle_failed"
            geometry_report["collision_plan"]["endpoint_settle"] = endpoint_settle
            contact_report: dict[str, object] = {
                "enabled": contact_trial,
                "increments": [],
                "visual_verifications": [],
                "official_success_queried": False,
            }
            if contact_trial and geometry_status == "geometric_standoff_reached":
                if (
                    recipe.manipulation not in {"push", "pull"}
                    and task not in PRESS_TASKS
                ):
                    raise ValueError("task has no certified bounded contact pilot")
                if final_target_position is None or final_target_orientation is None:
                    raise AssertionError("contact trial has no certified standoff")
                supervisor.accept_motion("manipulate_feature")
                push_axis = (
                    np.asarray(triangulated.point_world_m, dtype=np.float64)
                    - final_target_position
                )
                contact_plane_distance_m = float(np.linalg.norm(push_axis))
                push_axis /= contact_plane_distance_m
                contact_report["contact_plane_distance_m"] = contact_plane_distance_m
                if recipe.manipulation == "pull":
                    grasp_entry_goal = pull_grasp_entry_goal(
                        final_target_position, push_axis
                    )
                    entry_steps: list[dict[str, object]] = []
                    entry_start_eef = public_eef_world(
                        observation["public_state"]
                    )
                    entry_contact_confirmations = 0
                    entry_contact_confirmed = False
                    # Near-handle OSC progress is slower than free-space progress;
                    # reuse the reviewed bounded long-approach budget.
                    for entry_index in range(waypoint_increment_budget(0.20)):
                        desired_relative = world_quaternion_to_base_xyzw(
                            observation["public_state"], final_target_orientation
                        )
                        delta, entry_error_before = bounded_base_frame_delta(
                            observation["public_state"],
                            grasp_entry_goal,
                            max_step_m=0.01,
                        )
                        rotation, entry_orientation_before = (
                            bounded_orientation_delta(
                                observation["public_state"], desired_relative
                            )
                        )
                        before = public_eef_world(observation["public_state"])
                        observation = send_action(
                            tuple(float(value) for value in delta),
                            "open",
                            f"pull_grasp_entry_{entry_index + 1}",
                            rotation=tuple(float(value) for value in rotation),
                        )
                        after = public_eef_world(observation["public_state"])
                        entry_contact = confirm_public_motion_contact(
                            realized_m=float(np.linalg.norm(after - before)),
                            requested_m=float(np.linalg.norm(delta)),
                            realized_forward_m=max(
                                0.0,
                                float(np.dot(after - entry_start_eef, push_axis)),
                            ),
                            contact_entry_m=0.05,
                            previous_confirmations=entry_contact_confirmations,
                        )
                        entry_contact_confirmations = (
                            entry_contact.confirmations
                        )
                        entry_contact_confirmed = entry_contact.confirmed
                        entry_steps.append(
                            {
                                "index": entry_index + 1,
                                "position_error_before_m": entry_error_before,
                                "orientation_error_before_rad": (
                                    entry_orientation_before
                                ),
                                "realized_m": float(np.linalg.norm(after - before)),
                                "public_contact_eligible": entry_contact.eligible,
                                "public_contact_stalled": entry_contact.stalled,
                                "public_contact_confirmations": (
                                    entry_contact.confirmations
                                ),
                            }
                        )
                        if float(np.linalg.norm(after - grasp_entry_goal)) <= 0.012:
                            break
                    contact_report["pull_grasp_entry"] = {
                        "goal_world_m": grasp_entry_goal.tolist(),
                        "steps": entry_steps,
                        "final_position_error_m": float(
                            np.linalg.norm(
                                public_eef_world(observation["public_state"])
                                - grasp_entry_goal
                            )
                        ),
                        "public_contact_confirmed": entry_contact_confirmed,
                    }
                    if (
                        contact_report["pull_grasp_entry"][
                            "final_position_error_m"
                        ]
                        > 0.012
                        and not entry_contact_confirmed
                    ):
                        geometry_status = "pull_grasp_entry_failed"
                if geometry_status == "geometric_standoff_reached":
                    for index in range(8 if recipe.manipulation == "pull" else 5):
                        observation = send_action(
                            (0.0, 0.0, 0.0),
                            "close",
                            f"contact_gripper_close_{index + 1}",
                        )
                if (
                    recipe.manipulation == "pull"
                    and geometry_status == "geometric_standoff_reached"
                ):
                    grasp_attempts: list[dict[str, object]] = []
                    search_depths = grasp_search_entry_depths()
                    occupied = public_grasp_occupied(
                        observation["public_state"]["state.gripper_qpos"]
                    )
                    grasp_attempts.append(
                        {
                            "entry_depth_m": search_depths[0],
                            "aperture_m": public_grasp_aperture_m(
                                observation["public_state"]["state.gripper_qpos"]
                            ),
                            "occupied": occupied,
                        }
                    )
                    for attempt_index, entry_depth_m in enumerate(
                        search_depths[1:], start=2
                    ):
                        if occupied:
                            break
                        for open_index in range(5):
                            observation = send_action(
                                (0.0, 0.0, 0.0),
                                "open",
                                f"grasp_search_{attempt_index}_open_{open_index + 1}",
                            )
                        attempt_goal = pull_grasp_entry_goal(
                            final_target_position,
                            push_axis,
                            entry_depth_m=entry_depth_m,
                        )
                        attempt_steps = 0
                        for attempt_steps in range(1, waypoint_increment_budget(0.02) + 1):
                            desired_relative = world_quaternion_to_base_xyzw(
                                observation["public_state"], final_target_orientation
                            )
                            delta, _ = bounded_base_frame_delta(
                                observation["public_state"],
                                attempt_goal,
                                max_step_m=0.01,
                            )
                            rotation, _ = bounded_orientation_delta(
                                observation["public_state"], desired_relative
                            )
                            observation = send_action(
                                tuple(float(value) for value in delta),
                                "open",
                                f"grasp_search_{attempt_index}_advance_{attempt_steps}",
                                rotation=tuple(float(value) for value in rotation),
                            )
                            if (
                                float(
                                    np.linalg.norm(
                                        public_eef_world(observation["public_state"])
                                        - attempt_goal
                                    )
                                )
                                <= 0.012
                            ):
                                break
                        position_error_m = float(
                            np.linalg.norm(
                                public_eef_world(observation["public_state"])
                                - attempt_goal
                            )
                        )
                        if position_error_m > 0.012:
                            grasp_attempts.append(
                                {
                                    "entry_depth_m": entry_depth_m,
                                    "advance_steps": attempt_steps,
                                    "position_error_m": position_error_m,
                                    "occupied": False,
                                }
                            )
                            geometry_status = "grasp_search_settle_failed"
                            break
                        for close_index in range(8):
                            observation = send_action(
                                (0.0, 0.0, 0.0),
                                "close",
                                f"grasp_search_{attempt_index}_close_{close_index + 1}",
                            )
                        occupied = public_grasp_occupied(
                            observation["public_state"]["state.gripper_qpos"]
                        )
                        grasp_attempts.append(
                            {
                                "entry_depth_m": entry_depth_m,
                                "advance_steps": attempt_steps,
                                "position_error_m": position_error_m,
                                "aperture_m": public_grasp_aperture_m(
                                    observation["public_state"]["state.gripper_qpos"]
                                ),
                                "occupied": occupied,
                            }
                        )
                    contact_report["grasp_attempts"] = grasp_attempts
                    if not occupied and geometry_status == "geometric_standoff_reached":
                        geometry_status = "empty_grasp_exhausted"
                contact_start = observation
                contact_start_servo = _servo_observation(contact_start)
                contact_start_eef = public_eef_world(contact_start["public_state"])
                button_contact_entry_m = (
                    contact_plane_distance_m - measured_support
                )
                contact_report["button_contact_entry_m"] = button_contact_entry_m
                shortfalls = 0
                visual_finish_streak = 0
                button_contact_confirmations = 0
                if geometry_status != "geometric_standoff_reached":
                    increment_count = 0
                elif recipe.manipulation == "pull":
                    increment_count = 40
                else:
                    increment_count = 60
                pull_finish_streak = 0
                for increment in range(increment_count):
                    distance_m = 0.02 * (increment + 1)
                    if recipe.manipulation == "pull":
                        distance_m = 0.01 * (increment + 1)
                        target_position = pull_target(
                            contact_start_eef,
                            push_axis,
                            distance_m=distance_m,
                        )
                    else:
                        target_position = (
                            final_target_position + distance_m * push_axis
                        )
                    desired_relative = world_quaternion_to_base_xyzw(
                        observation["public_state"], final_target_orientation
                    )
                    delta, position_before = bounded_base_frame_delta(
                        observation["public_state"],
                        target_position,
                        max_step_m=0.02,
                    )
                    rotation, orientation_before = bounded_orientation_delta(
                        observation["public_state"], desired_relative
                    )
                    eef_before = public_eef_world(observation["public_state"])
                    observation = send_action(
                        tuple(float(value) for value in delta),
                        "close",
                        f"contact_push_{increment + 1}",
                        rotation=tuple(float(value) for value in rotation),
                    )
                    eef_after = public_eef_world(observation["public_state"])
                    realized_m = float(np.linalg.norm(eef_after - eef_before))
                    realized_forward_m = float(
                        np.dot(eef_after - contact_start_eef, push_axis)
                    )
                    realized_pull_m = max(
                        0.0,
                        float(np.dot(contact_start_eef - eef_after, push_axis)),
                    )
                    requested_m = float(np.linalg.norm(delta))
                    shortfalls = shortfalls + 1 if realized_m < 0.2 * requested_m else 0
                    contact_report["increments"].append(
                        {
                            "index": increment + 1,
                            "commanded_distance_from_standoff_m": distance_m,
                            "position_error_before_m": position_before,
                            "orientation_error_before_rad": orientation_before,
                            "requested_m": requested_m,
                            "realized_m": realized_m,
                            "realized_forward_from_standoff_m": realized_forward_m,
                            "realized_pull_from_grasp_m": realized_pull_m,
                            "consecutive_motion_shortfalls": shortfalls,
                        }
                    )
                    if recipe.manipulation == "pull":
                        if realized_pull_m < 0.20 or (increment + 1) % 2:
                            continue
                        supervisor.record_model_call()
                        response = client.complete(
                            observation_id=str(observation["observation_id"]),
                            system_prompt=open_drawer_verifier_prompt(task),
                            instruction=json.dumps(
                                {
                                    "task": observation["instruction"],
                                    "harness_stage": "bounded_pull_completion_check",
                                    "realized_public_pull_m": realized_pull_m,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            public_state=observation["public_state"],
                            images=_read_images(observation),
                        )
                        try:
                            open_verification = decode_visual_open_completion(
                                response.command,
                                observation_id=str(observation["observation_id"]),
                            )
                        except ValueError as error:
                            contact_report["visual_verifications"].append(
                                {
                                    "index": increment + 1,
                                    "kind": "rejected",
                                    "evidence": str(error),
                                    "observation_id": str(
                                        observation["observation_id"]
                                    ),
                                    "realized_public_pull_m": realized_pull_m,
                                }
                            )
                            pull_finish_streak = 0
                            continue
                        verification_record = {
                            "index": increment + 1,
                            "kind": open_verification.kind,
                            "left_extension_visible": (
                                open_verification.left_extension_visible
                            ),
                            "right_extension_visible": (
                                open_verification.right_extension_visible
                            ),
                            "evidence": open_verification.evidence,
                            "observation_id": open_verification.observation_id,
                            "realized_public_pull_m": realized_pull_m,
                        }
                        contact_report["visual_verifications"].append(
                            verification_record
                        )
                        if open_verification.kind == "give_up":
                            geometry_status = "pull_visual_verifier_gave_up"
                            break
                        pull_finish_streak, finish_ready = pull_finish_gate(
                            realized_pull_m=realized_pull_m,
                            visual_kind=open_verification.kind,
                            previous_streak=pull_finish_streak,
                        )
                        verification_record["consecutive_finish_count"] = (
                            pull_finish_streak
                        )
                        if not finish_ready:
                            continue
                        _atomic_json(
                            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                            {
                                "schema": "robocasa-inspect-command/v1",
                                "sequence": sequence,
                                "kind": "finish",
                            },
                        )
                        terminal_issued = True
                        contact_report["official_success_queried"] = True
                        process.wait(timeout=180)
                        outcome = json.loads(
                            (run / "sim" / "terminal-outcome.json").read_text()
                        )
                        official_success = outcome.get("success") is True
                        geometry_status = (
                            "official_success"
                            if official_success
                            else "official_finish_false"
                        )
                        contact_report["terminal_outcome"] = outcome
                        break
                    if task in PRESS_TASKS:
                        contact_evidence = confirm_public_motion_contact(
                            realized_m=realized_m,
                            requested_m=requested_m,
                            realized_forward_m=realized_forward_m,
                            contact_entry_m=button_contact_entry_m,
                            previous_confirmations=button_contact_confirmations,
                        )
                        button_contact_confirmations = (
                            contact_evidence.confirmations
                        )
                        contact_report["increments"][-1][
                            "public_motion_progress_ratio"
                        ] = contact_evidence.progress_ratio
                        contact_report["increments"][-1][
                            "button_contact_eligible"
                        ] = contact_evidence.eligible
                        contact_report["increments"][-1][
                            "button_contact_stalled"
                        ] = contact_evidence.stalled
                        contact_report["increments"][-1][
                            "button_contact_confirmations"
                        ] = contact_evidence.confirmations
                        if not contact_evidence.confirmed:
                            continue
                        retreat_goal = button_retreat_goal(
                            triangulated.point_world_m, push_axis
                        )
                        retreat_steps: list[dict[str, object]] = []
                        retreat_clearance_m = 0.0
                        for retreat_index in range(60):
                            desired_relative = world_quaternion_to_base_xyzw(
                                observation["public_state"],
                                final_target_orientation,
                            )
                            delta, retreat_error_before = bounded_base_frame_delta(
                                observation["public_state"],
                                retreat_goal,
                                max_step_m=0.02,
                            )
                            rotation, retreat_orientation_before = (
                                bounded_orientation_delta(
                                    observation["public_state"], desired_relative
                                )
                            )
                            retreat_before = public_eef_world(
                                observation["public_state"]
                            )
                            observation = send_action(
                                tuple(float(value) for value in delta),
                                "close",
                                f"button_retreat_{retreat_index + 1}",
                                rotation=tuple(float(value) for value in rotation),
                            )
                            retreat_after = public_eef_world(
                                observation["public_state"]
                            )
                            retreat_clearance_m = float(
                                np.linalg.norm(
                                    retreat_after
                                    - np.asarray(
                                        triangulated.point_world_m,
                                        dtype=np.float64,
                                    )
                                )
                            )
                            retreat_steps.append(
                                {
                                    "index": retreat_index + 1,
                                    "position_error_before_m": retreat_error_before,
                                    "orientation_error_before_rad": (
                                        retreat_orientation_before
                                    ),
                                    "realized_m": float(
                                        np.linalg.norm(retreat_after - retreat_before)
                                    ),
                                    "public_button_clearance_m": (
                                        retreat_clearance_m
                                    ),
                                }
                            )
                            if retreat_clearance_m >= 0.17:
                                break
                        contact_report["retreat"] = {
                            "goal_world_m": retreat_goal.tolist(),
                            "required_public_clearance_m": 0.17,
                            "steps": retreat_steps,
                            "final_public_clearance_m": retreat_clearance_m,
                        }
                        if retreat_clearance_m < 0.17:
                            geometry_status = "button_retreat_incomplete"
                            break
                        _atomic_json(
                            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                            {
                                "schema": "robocasa-inspect-command/v1",
                                "sequence": sequence,
                                "kind": "finish",
                            },
                        )
                        terminal_issued = True
                        contact_report["official_success_queried"] = True
                        process.wait(timeout=180)
                        outcome = json.loads(
                            (run / "sim" / "terminal-outcome.json").read_text()
                        )
                        official_success = outcome.get("success") is True
                        geometry_status = (
                            "official_success"
                            if official_success
                            else "official_finish_false"
                        )
                        contact_report["terminal_outcome"] = outcome
                        break
                    reached_contact_plane = (
                        distance_m >= contact_plane_distance_m - 1e-9
                    )
                    should_verify = (
                        reached_contact_plane
                        and realized_forward_m
                        >= contact_plane_distance_m + 0.12
                        and (increment + 1) % 2 == 0
                    )
                    if not should_verify:
                        continue
                    current_servo = _servo_observation(observation)
                    visual_displacement_px: dict[str, float] = {}
                    for view in ("left", "right"):
                        start_point = np.asarray(
                            project_world_point(
                                contact_start["camera_calibration"][view],
                                triangulated.point_world_m,
                            )[:2],
                            dtype=np.float64,
                        )
                        try:
                            tracked = track_points(
                                getattr(contact_start_servo, view),
                                getattr(current_servo, view),
                                np.asarray([start_point]),
                            )
                        except ValueError:
                            continue
                        if tracked.max_forward_backward_error_px <= 1.5:
                            visual_displacement_px[view] = float(
                                np.linalg.norm(tracked.points[0] - start_point)
                            )
                    contact_report["increments"][-1][
                        "target_visual_displacement_px"
                    ] = visual_displacement_px
                    if (
                        visual_displacement_px.get("left", 0.0) < 10.0
                        or visual_displacement_px.get("right", 0.0) < 2.0
                    ):
                        continue
                    supervisor.record_model_call()
                    response = client.complete(
                        observation_id=str(observation["observation_id"]),
                        system_prompt=close_drawer_verifier_prompt(task),
                        instruction=json.dumps(
                            {
                                "task": observation["instruction"],
                                "harness_stage": "bounded_visual_completion_check",
                                "commanded_push_distance_m": distance_m,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        public_state=observation["public_state"],
                        images=_read_images(observation),
                    )
                    try:
                        verification = decode_visual_completion(
                            response.command,
                            observation_id=str(observation["observation_id"]),
                        )
                    except ValueError as error:
                        contact_report["visual_verifications"].append(
                            {
                                "index": increment + 1,
                                "kind": "rejected",
                                "evidence": str(error),
                                "observation_id": str(observation["observation_id"]),
                            }
                        )
                        continue
                    contact_report["visual_verifications"].append(
                        {
                            "index": increment + 1,
                            "kind": verification.kind,
                            "left_open_gap_visible": (
                                verification.left_open_gap_visible
                            ),
                            "right_open_gap_visible": (
                                verification.right_open_gap_visible
                            ),
                            "evidence": verification.evidence,
                            "observation_id": verification.observation_id,
                        }
                    )
                    if verification.kind == "give_up":
                        geometry_status = "visual_verifier_gave_up"
                        break
                    if verification.kind == "continue_push":
                        visual_finish_streak = 0
                        continue
                    if verification.kind == "finish":
                        visual_finish_streak += 1
                        contact_report["visual_verifications"][-1][
                            "consecutive_finish_count"
                        ] = visual_finish_streak
                        if visual_finish_streak < 3:
                            continue
                        _atomic_json(
                            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                            {
                                "schema": "robocasa-inspect-command/v1",
                                "sequence": sequence,
                                "kind": "finish",
                            },
                        )
                        terminal_issued = True
                        contact_report["official_success_queried"] = True
                        process.wait(timeout=180)
                        outcome = json.loads(
                            (run / "sim" / "terminal-outcome.json").read_text()
                        )
                        official_success = outcome.get("success") is True
                        geometry_status = (
                            "official_success"
                            if official_success
                            else "official_finish_false"
                        )
                        contact_report["terminal_outcome"] = outcome
                        break
                else:
                    if geometry_status != "geometric_standoff_reached":
                        pass
                    elif recipe.manipulation == "pull":
                        geometry_status = "bounded_pull_exhausted"
                    elif (
                        recipe.manipulation != "push"
                        or recipe.continuation_base_m <= 0
                    ):
                        geometry_status = "bounded_push_exhausted"
                    else:
                        continuation: dict[str, object] = {
                            "schema": "robocasa-public-long-push-continuation/v1",
                            "base_shift_m": recipe.continuation_base_m,
                            "retreat": [],
                            "base_shift": [],
                            "reapproach": [],
                            "push": [],
                            "visual_verifications": [],
                        }
                        contact_report["continuation"] = continuation
                        last_contact_world = public_eef_world(
                            observation["public_state"]
                        )
                        retreat_goal = continuation_retreat_goal(
                            last_contact_world,
                            push_axis,
                            clearance_m=0.20,
                        )
                        for continuation_index in range(
                            waypoint_increment_budget(0.20)
                        ):
                            desired_relative = world_quaternion_to_base_xyzw(
                                observation["public_state"],
                                final_target_orientation,
                            )
                            delta, error_before = bounded_base_frame_delta(
                                observation["public_state"],
                                retreat_goal,
                                max_step_m=0.02,
                            )
                            rotation, orientation_before = (
                                bounded_orientation_delta(
                                    observation["public_state"], desired_relative
                                )
                            )
                            before = public_eef_world(
                                observation["public_state"]
                            )
                            observation = send_action(
                                tuple(float(value) for value in delta),
                                "close",
                                f"continuation_retreat_{continuation_index + 1}",
                                rotation=tuple(float(value) for value in rotation),
                            )
                            after = public_eef_world(observation["public_state"])
                            continuation["retreat"].append(
                                {
                                    "index": continuation_index + 1,
                                    "position_error_before_m": error_before,
                                    "orientation_error_before_rad": orientation_before,
                                    "realized_m": float(np.linalg.norm(after - before)),
                                }
                            )
                            if float(np.linalg.norm(after - retreat_goal)) <= 0.012:
                                break
                        else:
                            geometry_status = "continuation_retreat_failed"
                        if geometry_status != "continuation_retreat_failed":
                            base_start = np.asarray(
                                observation["public_state"]["state.base_position"],
                                dtype=np.float64,
                            )
                            for continuation_index in range(120):
                                request = continuation_base_request(
                                    observation["public_state"],
                                    start_base_world_m=base_start,
                                    push_direction_world=push_axis,
                                    distance_m=recipe.continuation_base_m,
                                )
                                if request is None:
                                    break
                                axis, direction = request
                                before = np.asarray(
                                    observation["public_state"][
                                        "state.base_position"
                                    ],
                                    dtype=np.float64,
                                )
                                observation = send_base(
                                    axis,
                                    direction,
                                    f"continuation_base_{axis}_{continuation_index + 1}",
                                    velocity=0.25,
                                )
                                after = np.asarray(
                                    observation["public_state"][
                                        "state.base_position"
                                    ],
                                    dtype=np.float64,
                                )
                                realized = float(np.linalg.norm(after - before))
                                continuation["base_shift"].append(
                                    {
                                        "index": continuation_index + 1,
                                        "axis": axis,
                                        "direction": direction,
                                        "realized_m": realized,
                                    }
                                )
                                if realized <= 1e-5:
                                    geometry_status = (
                                        "continuation_base_shift_failed"
                                    )
                                    break
                            else:
                                geometry_status = "continuation_base_shift_failed"
                        if geometry_status not in {
                            "continuation_retreat_failed",
                            "continuation_base_shift_failed",
                        }:
                            for continuation_index in range(
                                waypoint_increment_budget(0.20)
                            ):
                                desired_relative = world_quaternion_to_base_xyzw(
                                    observation["public_state"],
                                    final_target_orientation,
                                )
                                delta, error_before = bounded_base_frame_delta(
                                    observation["public_state"],
                                    last_contact_world,
                                    max_step_m=0.02,
                                )
                                rotation, orientation_before = (
                                    bounded_orientation_delta(
                                        observation["public_state"], desired_relative
                                    )
                                )
                                before = public_eef_world(
                                    observation["public_state"]
                                )
                                observation = send_action(
                                    tuple(float(value) for value in delta),
                                    "close",
                                    f"continuation_reapproach_{continuation_index + 1}",
                                    rotation=tuple(float(value) for value in rotation),
                                )
                                after = public_eef_world(
                                    observation["public_state"]
                                )
                                continuation["reapproach"].append(
                                    {
                                        "index": continuation_index + 1,
                                        "position_error_before_m": error_before,
                                        "orientation_error_before_rad": (
                                            orientation_before
                                        ),
                                        "realized_m": float(
                                            np.linalg.norm(after - before)
                                        ),
                                    }
                                )
                                if (
                                    float(
                                        np.linalg.norm(after - last_contact_world)
                                    )
                                    <= 0.012
                                ):
                                    break
                            else:
                                geometry_status = "continuation_reapproach_failed"
                        if geometry_status not in {
                            "continuation_retreat_failed",
                            "continuation_base_shift_failed",
                            "continuation_reapproach_failed",
                        }:
                            continuation_finish_streak = 0
                            for continuation_index in range(44):
                                target_position = (
                                    last_contact_world
                                    + 0.02
                                    * (continuation_index + 1)
                                    * push_axis
                                )
                                desired_relative = world_quaternion_to_base_xyzw(
                                    observation["public_state"],
                                    final_target_orientation,
                                )
                                delta, error_before = bounded_base_frame_delta(
                                    observation["public_state"],
                                    target_position,
                                    max_step_m=0.02,
                                )
                                rotation, orientation_before = (
                                    bounded_orientation_delta(
                                        observation["public_state"], desired_relative
                                    )
                                )
                                before = public_eef_world(
                                    observation["public_state"]
                                )
                                observation = send_action(
                                    tuple(float(value) for value in delta),
                                    "close",
                                    f"continuation_push_{continuation_index + 1}",
                                    rotation=tuple(float(value) for value in rotation),
                                )
                                after = public_eef_world(
                                    observation["public_state"]
                                )
                                realized_forward_m = max(
                                    0.0,
                                    float(
                                        np.dot(
                                            after - last_contact_world,
                                            push_axis,
                                        )
                                    ),
                                )
                                continuation["push"].append(
                                    {
                                        "index": continuation_index + 1,
                                        "position_error_before_m": error_before,
                                        "orientation_error_before_rad": (
                                            orientation_before
                                        ),
                                        "realized_m": float(
                                            np.linalg.norm(after - before)
                                        ),
                                        "realized_forward_from_recontact_m": (
                                            realized_forward_m
                                        ),
                                    }
                                )
                                if (continuation_index + 1) % 2:
                                    continue
                                supervisor.record_model_call()
                                response = client.complete(
                                    observation_id=str(
                                        observation["observation_id"]
                                    ),
                                    system_prompt=close_drawer_verifier_prompt(task),
                                    instruction=json.dumps(
                                        {
                                            "task": observation["instruction"],
                                            "harness_stage": (
                                                "long_push_continuation_check"
                                            ),
                                            "continuation_increment": (
                                                continuation_index + 1
                                            ),
                                        },
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    ),
                                    public_state=observation["public_state"],
                                    images=_read_images(observation),
                                )
                                try:
                                    verification = decode_visual_completion(
                                        response.command,
                                        observation_id=str(
                                            observation["observation_id"]
                                        ),
                                    )
                                except ValueError as error:
                                    continuation["visual_verifications"].append(
                                        {
                                            "index": continuation_index + 1,
                                            "kind": "rejected",
                                            "evidence": str(error),
                                        }
                                    )
                                    continuation_finish_streak = 0
                                    continue
                                verification_record = {
                                    "index": continuation_index + 1,
                                    "kind": verification.kind,
                                    "left_open_gap_visible": (
                                        verification.left_open_gap_visible
                                    ),
                                    "right_open_gap_visible": (
                                        verification.right_open_gap_visible
                                    ),
                                    "evidence": verification.evidence,
                                    "observation_id": verification.observation_id,
                                    "realized_forward_from_recontact_m": (
                                        realized_forward_m
                                    ),
                                }
                                continuation["visual_verifications"].append(
                                    verification_record
                                )
                                if verification.kind == "give_up":
                                    geometry_status = (
                                        "continuation_visual_verifier_gave_up"
                                    )
                                    break
                                continuation_finish_streak, finish_ready = (
                                    continuation_finish_gate(
                                        realized_forward_m=realized_forward_m,
                                        visual_kind=verification.kind,
                                        previous_streak=continuation_finish_streak,
                                    )
                                )
                                verification_record[
                                    "consecutive_finish_count"
                                ] = continuation_finish_streak
                                if (
                                    verification.kind == "finish"
                                    and continuation_finish_streak == 0
                                ):
                                    verification_record[
                                        "finish_deferred_by_public_travel"
                                    ] = True
                                if not finish_ready:
                                    continue
                                _atomic_json(
                                    run
                                    / "sim"
                                    / "mailbox"
                                    / f"command-{sequence:06d}.json",
                                    {
                                        "schema": (
                                            "robocasa-inspect-command/v1"
                                        ),
                                        "sequence": sequence,
                                        "kind": "finish",
                                    },
                                )
                                terminal_issued = True
                                contact_report[
                                    "official_success_queried"
                                ] = True
                                process.wait(timeout=180)
                                outcome = json.loads(
                                    (
                                        run
                                        / "sim"
                                        / "terminal-outcome.json"
                                    ).read_text()
                                )
                                official_success = outcome.get("success") is True
                                geometry_status = (
                                    "official_success"
                                    if official_success
                                    else "official_finish_false"
                                )
                                contact_report["terminal_outcome"] = outcome
                                break
                            else:
                                geometry_status = "continuation_push_exhausted"
            geometry_report["contact_trial"] = contact_report
            execution = ServoExecution(
                geometry_status,
                _servo_observation(observation),
                geometry_report,
            )
        else:
            execution = run_single_view_center(
                initial=returned,
                view=selected_view,
                target_px=target,
                gripper_px=gripper_point,
                reset_eef_position_m=opened.eef_position_m,
                execute=execute,
                audit_only=not center,
            )
        if center and not base_precenter:
            supervisor.accept_motion("center_feature")
        supervisor.terminate(execution.status)
        overlay = Image.fromarray(getattr(returned, selected_view))
        draw = ImageDraw.Draw(overlay)
        for point, color, label in (
            (target, "lime", "target"),
            (np.asarray(anchor.anchor_uv_px), "cyan", "gripper"),
        ):
            x, y = (float(value) for value in point)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
            draw.text((x + 9, y - 9), label, fill=color)
        overlay_path = run / "single-view-anchor-overlay.png"
        overlay.save(overlay_path)
        overlay_path.chmod(0o600)
        if not terminal_issued:
            _atomic_json(
                run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                {
                    "schema": "robocasa-inspect-command/v1",
                    "sequence": sequence,
                    "kind": "close",
                },
            )
            process.wait(timeout=180)
        verify_process(attestation)
        validate_execution_authority(execution_authority, **authority_paths)
        video = run / "single-view-calibration-audit.mp4"
        frames = _render_video(run / "sim", video)
        sealed_evidence = seal_harness_evidence(
            run=run,
            receipts=receipts,
            authority_path=authority_path,
            video_path=video,
        )
        result = {
            "schema": "robocasa-inspect-single-view-audit/v1",
            "task": task,
            "seed": seed,
            "status": execution.status,
            "success": official_success,
            "success_was_queried": terminal_issued,
            "qwen_target_uv": list(command.feature_uv),
            "selected_view": selected_view,
            "feature_kind": recipe.feature_kind,
            "manipulation_recipe": recipe.manipulation,
            "rgb_gripper_anchor_px": list(anchor.anchor_uv_px),
            "qwen_evidence": response.evidence,
            "harness_report": execution.report,
            "base_precenter_report": base_report,
            "receipts": receipts,
            "action_chunks": action_chunks,
            "revision_consumed": supervisor.revision_consumed,
            "consumption_trigger": supervisor.consumption_trigger,
            "frame_sets": frames,
            "overlay": str(overlay_path),
            "video": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "snapshot_digest": model_identity.get("snapshot_digest"),
            "served_model_id": attestation.get("served_model_id"),
            "system_prompt_sha256": hashlib.sha256(
                recipe_system_prompt(task).encode("utf-8")
            ).hexdigest(),
            "execution_authority": str(authority_path),
            "execution_authority_sha256": execution_authority["sha256"],
            "execution_authority_file_sha256": hashlib.sha256(
                authority_path.read_bytes()
            ).hexdigest(),
            "harness_evidence": str(run / "harness-evidence.json"),
            "harness_evidence_sha256": hashlib.sha256(
                (run / "harness-evidence.json").read_bytes()
            ).hexdigest(),
            "receipts_sha256": sealed_evidence["receipts_sha256"],
            "frame_manifest_sha256": sealed_evidence[
                "frame_manifest_sha256"
            ],
        }
        _atomic_json(run / "single-view-audit.json", result)
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        log.close()
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task", default="CloseDrawer")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--base-precenter", action="store_true")
    parser.add_argument("--geometry-center", action="store_true")
    parser.add_argument("--contact-trial", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
                run=args.run_dir.resolve(),
                task=args.task,
                seed=args.seed,
                center=args.center,
                base_precenter=args.base_precenter,
                geometry_center=args.geometry_center,
                contact_trial=args.contact_trial,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
