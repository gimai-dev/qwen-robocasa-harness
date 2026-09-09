"""Closed active-skill protocol and evidence lifecycle.

This module deliberately contains no RoboCasa task or geometry knowledge.  It accepts
only fresh official observation provenance and public robot state.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import STATE_KEYS

EXTERNAL_VIEWS = {"robot0_agentview_left", "robot0_agentview_right"}
FEATURE_KINDS = {
    "handle",
    "front_surface",
    "button",
    "control",
    "object",
    "placement_region",
}
SCAN_AXES = {"x", "y", "yaw"}
MANIPULATIONS = {"push", "push_down", "pull", "grip_then_move"}
TERMINAL_ONLY_SKILLS = {"retreat", "finish", "give_up"}

_SCHEMAS = {
    "observe": {"kind", "observation_id"},
    "scan_view": {"kind", "observation_id", "axis", "magnitude"},
    "center_feature": {
        "kind",
        "observation_id",
        "view",
        "feature_uv",
        "feature_kind",
    },
    "approach_feature": {"kind", "observation_id", "feature_id"},
    "manipulate_feature": {
        "kind",
        "observation_id",
        "feature_id",
        "action",
    },
    "retreat": {"kind", "observation_id"},
    "finish": {"kind", "observation_id"},
    "give_up": {"kind", "observation_id"},
}


def skill_protocol_manifest() -> dict[str, object]:
    """Return the immutable, public-evidence-only tool contract."""
    return {
        "schema": "robocasa-inspect-skill-protocol/v1",
        "commands": {
            kind: sorted(fields) for kind, fields in sorted(_SCHEMAS.items())
        },
        "external_views": sorted(EXTERNAL_VIEWS),
        "feature_kinds": sorted(FEATURE_KINDS),
        "scan_axes": sorted(SCAN_AXES),
        "manipulations": sorted(MANIPULATIONS),
        "bounds": {
            "feature_border_px": 12,
            "scan_translation_m": 0.05,
            "scan_yaw_rad": 0.15,
            "model_calls": 40,
        },
        "terminal": {
            "official_success_query": "exactly_once_after_finish",
            "intermediate_success": False,
        },
    }


@dataclass(frozen=True)
class ActiveSkillCommand:
    kind: str
    observation_id: str
    axis: str = ""
    magnitude: float = 0.0
    view: str = ""
    feature_uv: tuple[float, float] | None = None
    feature_kind: str = ""
    feature_id: str = ""
    action: str = ""


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"invalid {name}")
    return value


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"invalid {name}")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"invalid {name}")
    return scalar


def decode_skill_command(
    value: Mapping[str, object], *, observation_id: str
) -> ActiveSkillCommand:
    """Decode one exact closed command without coercing an invalid motion."""
    kind = value.get("kind")
    if kind not in _SCHEMAS:
        raise ValueError("unsupported skill kind")
    if set(value) != _SCHEMAS[str(kind)]:
        raise ValueError("skill fields do not match the closed schema")
    if value.get("observation_id") != observation_id:
        raise ValueError("stale observation citation")
    command = ActiveSkillCommand(kind=str(kind), observation_id=observation_id)
    if kind == "scan_view":
        axis = value.get("axis")
        if axis not in SCAN_AXES:
            raise ValueError("unsupported scan axis")
        magnitude = _finite_scalar(value.get("magnitude"), name="scan magnitude")
        limit = 0.15 if axis == "yaw" else 0.05
        if abs(magnitude) <= 1e-12 or abs(magnitude) > limit:
            raise ValueError("scan magnitude exceeds its bound")
        return ActiveSkillCommand(
            kind=str(kind), observation_id=observation_id, axis=str(axis), magnitude=magnitude
        )
    if kind == "center_feature":
        view = value.get("view")
        if view not in {*EXTERNAL_VIEWS, "robot0_eye_in_hand"}:
            raise ValueError("unsupported official view")
        feature_kind = value.get("feature_kind")
        if feature_kind not in FEATURE_KINDS:
            raise ValueError("unsupported feature kind")
        uv = np.asarray(value.get("feature_uv"), dtype=np.float64)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError("feature_uv must contain two finite values")
        border = 12.0 / 255.0
        if np.any(uv < border) or np.any(uv > 1.0 - border):
            raise ValueError("feature citation violates the 12-pixel border")
        return ActiveSkillCommand(
            kind=str(kind),
            observation_id=observation_id,
            view=str(view),
            feature_uv=(float(uv[0]), float(uv[1])),
            feature_kind=str(feature_kind),
        )
    if kind == "approach_feature":
        return ActiveSkillCommand(
            kind=str(kind),
            observation_id=observation_id,
            feature_id=_identifier(value.get("feature_id"), name="feature_id"),
        )
    if kind == "manipulate_feature":
        action = value.get("action")
        if action not in MANIPULATIONS:
            raise ValueError("unsupported manipulation")
        return ActiveSkillCommand(
            kind=str(kind),
            observation_id=observation_id,
            feature_id=_identifier(value.get("feature_id"), name="feature_id"),
            action=str(action),
        )
    return command


class ActiveSkillSupervisor:
    """Own bounded call, revision, and motion-shortfall state for one episode."""

    def __init__(self, *, max_model_calls: int = 40) -> None:
        if max_model_calls <= 0:
            raise ValueError("max_model_calls must be positive")
        self.max_model_calls = max_model_calls
        self.model_calls = 0
        self.revision_consumed = False
        self.consumption_trigger: str | None = None
        self._motion_trigger: str | None = None
        self._shortfalls = 0
        self._terminated = False
        self.allowed_after_session_end: set[str] | None = None

    def record_model_call(self) -> None:
        if self.model_calls >= self.max_model_calls:
            raise RuntimeError("budget_exhausted")
        self.model_calls += 1

    def record_observe(self) -> None:
        if self._terminated:
            raise RuntimeError("post-terminal observe")

    def accept_motion(self, skill: str) -> None:
        if self._terminated:
            raise RuntimeError("post-terminal motion")
        if skill == "observe":
            raise ValueError("observe is not motion-producing")
        if self._motion_trigger is None:
            self._motion_trigger = f"accepted_{skill}_motion"

    def terminate(self, outcome: str) -> None:
        if self._terminated:
            raise RuntimeError("terminal already recorded")
        self._terminated = True
        if self._motion_trigger is not None:
            self.revision_consumed = True
            self.consumption_trigger = self._motion_trigger

    def record_increment(
        self,
        *,
        phase: str,
        requested_m: float,
        realized_m: float,
        endpoint_error_m: float,
    ) -> dict[str, object] | None:
        values = np.asarray(
            [requested_m, realized_m, endpoint_error_m], dtype=np.float64
        )
        if not np.isfinite(values).all() or requested_m <= 0 or realized_m < 0:
            raise ValueError("invalid increment evidence")
        short = realized_m < 0.5 * requested_m or endpoint_error_m > 0.0025
        self._shortfalls = self._shortfalls + 1 if short else 0
        if self._shortfalls < 2:
            return None
        self._shortfalls = 0
        if phase == "manipulate":
            self.allowed_after_session_end = set(TERMINAL_ONLY_SKILLS)
            return {"kind": "session_ended_shortfall", "accepted": True}
        raise RuntimeError("safety_gate_failed")

    def require_skill_allowed(self, kind: str) -> None:
        if (
            self.allowed_after_session_end is not None
            and kind not in self.allowed_after_session_end
        ):
            raise RuntimeError("session_ended_shortfall")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def public_state_sha256(state: Mapping[str, object]) -> str:
    """Hash the exact public 16D state in the pinned wrapper order."""
    flattened: list[float] = []
    for key, width in STATE_KEYS:
        array = np.asarray(state.get(key), dtype=np.float64).reshape(-1)
        if array.shape != (width,) or not np.isfinite(array).all():
            raise ValueError(f"state contract drift: {key}")
        flattened.extend(float(item) for item in array)
    if len(flattened) != 16:
        raise AssertionError("public state is not 16D")
    return hashlib.sha256(_canonical_json(flattened)).hexdigest()


def validate_exact_replay(expected: Sequence[str], realized: Sequence[str]) -> None:
    if list(expected) != list(realized):
        raise RuntimeError("replay_diverged")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


class TerminalOutcomePersister:
    """Seal terminal digests before invoking the one-shot official evaluator."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run_dir.chmod(0o700)

    def seal(
        self,
        *,
        trace: object,
        receipts: object,
        evaluate: Callable[[], bool],
        terminal_snapshot_sha256: str | None = None,
        fail_after_digests: bool = False,
    ) -> dict[str, object]:
        if any(
            (self.run_dir / name).exists()
            for name in (
                "terminal-digests.json",
                "terminal-outcome.json",
                "terminal-unsealed.json",
            )
        ):
            raise RuntimeError("terminal evidence already sealed")
        digests = {
            "schema": "robocasa-inspect-terminal-digests/v1",
            "trace_sha256": hashlib.sha256(_canonical_json(trace)).hexdigest(),
            "receipts_sha256": hashlib.sha256(_canonical_json(receipts)).hexdigest(),
        }
        if terminal_snapshot_sha256 is not None:
            if (
                len(terminal_snapshot_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in terminal_snapshot_sha256
                )
            ):
                raise ValueError("terminal snapshot digest is invalid")
            digests["terminal_snapshot_sha256"] = terminal_snapshot_sha256
        _atomic_json(self.run_dir / "terminal-digests.json", digests)
        if fail_after_digests:
            unsealed = {
                "outcome": "terminal_evidence_unsealed",
                "revision_consumed": True,
                "retry_allowed": False,
                "success": False,
            }
            _atomic_json(self.run_dir / "terminal-unsealed.json", unsealed)
            raise RuntimeError("terminal_evidence_unsealed")
        success = evaluate()
        if type(success) is not bool:
            raise RuntimeError("official success predicate drift")
        outcome = {
            "schema": "robocasa-inspect-terminal-outcome/v1",
            **digests,
            "success": success,
        }
        _atomic_json(self.run_dir / "terminal-outcome.json", outcome)
        return outcome
