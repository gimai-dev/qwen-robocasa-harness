"""Content authority for the failed run that precedes harness development."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path

RUNTIME_DISTRIBUTIONS = (
    "av",
    "gymnasium",
    "httpx",
    "imageio",
    "imageio-ffmpeg",
    "mujoco",
    "numpy",
    "opencv-python-headless",
    "Pillow",
    "robosuite",
    "scipy",
)
MODEL_SUFFIXES = frozenset({".dae", ".mtl", ".obj", ".stl", ".xml"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _bundle(files: Mapping[str, Path]) -> dict[str, object]:
    if not files:
        raise ValueError("execution authority group is empty")
    digests = {label: _sha256(path) for label, path in sorted(files.items())}
    return {
        "files": digests,
        "sha256": hashlib.sha256(_canonical_json(digests)).hexdigest(),
    }


def _python_files(root: Path, *, prefix: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        f"{prefix}/{path.relative_to(root).as_posix()}": path
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    }


def _robot_model_files(robosuite_root: Path) -> dict[str, Path]:
    assets = robosuite_root / "robosuite" / "models" / "assets"
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    selected: dict[str, Path] = {}
    for path in assets.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
            continue
        label = path.relative_to(assets).as_posix()
        lowered = label.lower()
        if (
            "/panda/" in f"/{lowered}"
            or "panda_gripper" in lowered
            or "omron_mobile_base" in lowered
        ):
            selected[f"robosuite-model/{label}"] = path
    return selected


def _runtime_fingerprint() -> dict[str, object]:
    versions: dict[str, str] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required execution distribution is missing: {distribution}"
            ) from error
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "platform": platform.platform(),
            "executable_implementation": sys.implementation.name,
        },
        "distributions": versions,
    }


def _protocol_fingerprint() -> dict[str, object]:
    from .active_skills import skill_protocol_manifest
    from .button_skill import button_candidate_system_prompt
    from .contracts import CAMERAS, CHUNK_STEPS, STATE_KEYS
    from .inspect_prompt import SYSTEM_PROMPT
    from .task_recipes import TASK_RECIPES, recipe_system_prompt

    return {
        "schema": "robocasa-inspect-protocol-authority/v1",
        "cameras": list(CAMERAS),
        "public_state": [
            {"name": name, "width": width} for name, width in STATE_KEYS
        ],
        "action_chunk_steps": CHUNK_STEPS,
        "skills": skill_protocol_manifest(),
        "prompts_sha256": {
            "inspect_dossier": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "button_candidate": {
                task: hashlib.sha256(
                    button_candidate_system_prompt(task).encode()
                ).hexdigest()
                for task in ("TurnOnMicrowave", "TurnOffMicrowave")
            },
            "task_recipes": {
                task: hashlib.sha256(recipe_system_prompt(task).encode()).hexdigest()
                for task in sorted(TASK_RECIPES)
            },
        },
        "qwen_decoding": {
            "seed": 3074294,
            "temperature": 0,
            "max_tokens": 512,
            "response_format": "json_object",
            "responses": 1,
        },
        "image_roles": {
            "ordinary": ["official_left", "official_right", "official_wrist"],
            "button_disambiguation": [
                "official_left",
                "official_right",
                "derived_annotated_panel_zoom",
            ],
        },
        "terminal": {
            "predicate": "official_robocasa_check_success",
            "query": "exactly_once_after_explicit_finish",
            "model_visible": False,
        },
    }


def current_execution_authority(
    *,
    project_root: Path,
    robocasa_root: Path,
    robosuite_root: Path,
    asset_manifest_path: Path,
    identity_path: Path,
    attestation_path: Path,
) -> dict[str, object]:
    """Bind the complete public-policy, simulator, model, and runtime surface."""
    roots = {
        "project_root": project_root.resolve(),
        "robocasa_root": robocasa_root.resolve(),
        "robosuite_root": robosuite_root.resolve(),
        "asset_manifest_path": asset_manifest_path.resolve(),
        "identity_path": identity_path.resolve(),
        "attestation_path": attestation_path.resolve(),
    }
    harness_files = _python_files(
        roots["project_root"] / "robocasa_inspect", prefix="harness"
    )
    for filename in (
        "freeze_goal30.py",
        "install_assets.py",
        "probe_paired_grounding.py",
        "replay_demonstration.py",
        "run_100_task_eval.py",
    ):
        path = roots["project_root"] / filename
        if path.is_file():
            harness_files[f"harness/{filename}"] = path
    groups: dict[str, object] = {
        "harness": _bundle(harness_files),
        "official_robocasa": _bundle(
            _python_files(
                roots["robocasa_root"] / "robocasa",
                prefix="official-robocasa",
            )
        ),
        "official_robosuite": _bundle(
            _python_files(
                roots["robosuite_root"] / "robosuite",
                prefix="official-robosuite",
            )
        ),
        "robot_model": _bundle(_robot_model_files(roots["robosuite_root"])),
        "robocasa_asset_manifest": _bundle(
            {"robocasa-assets/content-manifest.json": roots["asset_manifest_path"]}
        ),
        "qwen_identity": _bundle(
            {"qwen/identity.json": roots["identity_path"]}
        ),
        "qwen_server_attestation": _bundle(
            {"qwen/server-attestation.json": roots["attestation_path"]}
        ),
        "runtime": _runtime_fingerprint(),
        "protocol": _protocol_fingerprint(),
    }
    group_digests = {
        label: (
            value["sha256"]
            if isinstance(value, Mapping) and "sha256" in value
            else hashlib.sha256(_canonical_json(value)).hexdigest()
        )
        for label, value in sorted(groups.items())
    }
    authority = {
        "schema": "robocasa-inspect-execution-authority/v1",
        "groups": groups,
        "group_sha256": group_digests,
    }
    authority["sha256"] = hashlib.sha256(
        _canonical_json(authority)
    ).hexdigest()
    return authority


def validate_execution_authority(
    expected: Mapping[str, object],
    **paths: Path,
) -> None:
    current = current_execution_authority(**paths)
    if dict(expected) != current:
        raise RuntimeError("execution authority drift")


def freeze_predecessor(
    *,
    sources: Mapping[str, Path],
    result_path: Path,
    video_path: Path,
    official_contract: Mapping[str, object],
) -> dict[str, object]:
    """Build a closed manifest without copying model text into later prompts."""
    if not sources or set(official_contract) != {"task", "seed", "chunk_steps"}:
        raise ValueError("predecessor authority is incomplete")
    terminal = json.loads(result_path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "task",
        "seed",
        "success",
        "snapshot_digest",
        "served_model_id",
        "video_sha256",
    }
    if not isinstance(terminal, dict) or not required.issubset(terminal):
        raise ValueError("predecessor result is incomplete")
    video_sha256 = _sha256(video_path)
    if terminal["video_sha256"] != video_sha256:
        raise ValueError("predecessor video digest mismatch")
    digests = {label: _sha256(path) for label, path in sorted(sources.items())}
    digests["result.json"] = _sha256(result_path)
    digests["demo.mp4"] = video_sha256
    return {
        "schema": "robocasa-inspect-predecessor/v1",
        "official_contract": dict(official_contract),
        "model": {
            "snapshot_digest": terminal["snapshot_digest"],
            "served_model_id": terminal["served_model_id"],
        },
        "terminal": {
            "schema": terminal["schema"],
            "task": terminal["task"],
            "seed": terminal["seed"],
            "success": terminal["success"],
        },
        "sha256": digests,
    }
