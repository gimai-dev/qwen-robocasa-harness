"""Owner-only, content-addressed evidence for official-camera harness runs."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path

CAMERA_NAMES = ("left", "right", "wrist")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _frame_manifest(run: Path) -> dict[str, object]:
    root = run / "sim" / "frames"
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not directories:
        raise RuntimeError("frame evidence is empty")
    expected_names = [f"{index:06d}" for index in range(len(directories))]
    if [path.name for path in directories] != expected_names:
        raise RuntimeError("frame evidence sequence is not contiguous")
    files: dict[str, str] = {}
    for directory in directories:
        actual = sorted(path.name for path in directory.glob("*.png"))
        expected = [f"{camera}.png" for camera in CAMERA_NAMES]
        if actual != expected:
            raise RuntimeError("frame evidence camera inventory drift")
        for camera in CAMERA_NAMES:
            path = directory / f"{camera}.png"
            label = path.relative_to(run).as_posix()
            files[label] = _sha256(path)
    return {
        "schema": "robocasa-inspect-frame-manifest/v1",
        "camera_names": list(CAMERA_NAMES),
        "frame_sets": len(directories),
        "files": files,
    }


def seal_harness_evidence(
    *,
    run: Path,
    receipts: Sequence[Mapping[str, object]],
    authority_path: Path,
    video_path: Path,
) -> dict[str, object]:
    """Seal exact public frames, accepted receipts, terminal, model, and video."""
    manifest = _frame_manifest(run)
    manifest_path = run / "frame-manifest.json"
    _write_once(manifest_path, manifest)
    terminal_path = run / "sim" / "terminal-outcome.json"
    if terminal_path.is_file():
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if (
            terminal.get("schema") != "robocasa-inspect-terminal-digests/v1"
            or not isinstance(terminal.get("success"), bool)
        ):
            raise RuntimeError("terminal evidence is incomplete")
    else:
        terminal_path = run / "sim" / "simulator-terminal.json"
        simulator_terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if simulator_terminal.get("status") != "closed":
            raise RuntimeError("terminal evidence is incomplete")
        terminal = {
            "schema": "robocasa-inspect-terminal-not-queried/v1",
            "simulator_status": "closed",
            "success": False,
            "success_queried": False,
        }
    receipts_sha256 = hashlib.sha256(
        _canonical_json(list(receipts))
    ).hexdigest()
    evidence: dict[str, object] = {
        "schema": "robocasa-inspect-harness-evidence/v1",
        "camera_names": list(CAMERA_NAMES),
        "frame_sets": manifest["frame_sets"],
        "frames": len(manifest["files"]),
        "frame_manifest": manifest_path.name,
        "frame_manifest_sha256": _sha256(manifest_path),
        "receipts_sha256": receipts_sha256,
        "terminal": terminal,
        "terminal_artifact": terminal_path.relative_to(run).as_posix(),
        "terminal_outcome_sha256": _sha256(terminal_path),
        "execution_authority": authority_path.name,
        "execution_authority_file_sha256": _sha256(authority_path),
        "video": video_path.name,
        "video_sha256": _sha256(video_path),
    }
    evidence["integrity_sha256"] = hashlib.sha256(
        _canonical_json(evidence)
    ).hexdigest()
    _write_once(run / "harness-evidence.json", evidence)
    return evidence


def validate_harness_evidence(
    evidence_path: Path, *, receipts: Sequence[Mapping[str, object]]
) -> None:
    run = evidence_path.parent
    expected = json.loads(evidence_path.read_text(encoding="utf-8"))
    integrity = expected.pop("integrity_sha256", None)
    if integrity != hashlib.sha256(_canonical_json(expected)).hexdigest():
        raise RuntimeError("harness evidence integrity drift")
    manifest_path = run / str(expected["frame_manifest"])
    if _sha256(manifest_path) != expected["frame_manifest_sha256"]:
        raise RuntimeError("frame evidence drift")
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if persisted_manifest != _frame_manifest(run):
        raise RuntimeError("frame evidence drift")
    receipt_digest = hashlib.sha256(_canonical_json(list(receipts))).hexdigest()
    if receipt_digest != expected["receipts_sha256"]:
        raise RuntimeError("receipt evidence drift")
    for name, digest_key in (
        ("execution_authority", "execution_authority_file_sha256"),
        ("video", "video_sha256"),
    ):
        if _sha256(run / str(expected[name])) != expected[digest_key]:
            raise RuntimeError(f"{name} evidence drift")
    terminal_path = run / str(expected["terminal_artifact"])
    if _sha256(terminal_path) != expected["terminal_outcome_sha256"]:
        raise RuntimeError("terminal evidence drift")
