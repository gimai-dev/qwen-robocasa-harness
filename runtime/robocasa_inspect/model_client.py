"""Task-owned identity-bound client for the existing loopback Qwen service."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

AUTHORITY = {
    "repo_id": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "served_model_name": "qwen3.8-27b-bf16",
    "dtype": "bfloat16",
}


@dataclass(frozen=True)
class Response:
    command: dict[str, object]
    evidence: dict[str, object]


class MalformedResponse(RuntimeError):
    """Fail-closed model output with sanitized request evidence."""

    def __init__(self, message: str, *, evidence: Mapping[str, object]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


class TransportFailure(RuntimeError):
    """Exhausted same-frame model transport without a simulator action."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        finish_classes: list[str],
        evidence: list[Mapping[str, object]],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.finish_classes = list(finish_classes)
        self.evidence = [dict(item) for item in evidence]
        self.taxonomy = "transport_failure"


_ATTEMPT_FIELDS = {
    "request_sha256",
    "observation_id",
    "attempt_index",
    "raw_body_sha256",
    "sanitized_raw_command",
    "http_status",
    "finish_reason",
    "completion_tokens",
    "latency_s",
    "response_schema_sha256",
    "served_model_id",
}


class AttemptEvidenceLog:
    """Owner-only, append-and-fsync evidence that survives abrupt termination."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise RuntimeError("attempt evidence parent must be a regular directory")
        stat = parent.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise RuntimeError("attempt evidence parent is not owner-only")
        if path.exists():
            file_stat = path.stat()
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("attempt evidence must be a regular file")
            if file_stat.st_uid != os.getuid() or file_stat.st_mode & 0o077:
                raise RuntimeError("attempt evidence is not owner-only")

    def append(self, record: Mapping[str, object]) -> None:
        if set(record) != _ATTEMPT_FIELDS:
            raise ValueError("attempt evidence fields drifted")
        payload = (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        with self._lock:
            fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
                0o600,
            )
            try:
                stat = os.fstat(fd)
                if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                    raise RuntimeError("attempt evidence descriptor is not owner-only")
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)


def _owner_json(path: Path, *, schema: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("authority artifact must be a regular file")
    stat = path.stat()
    parent = path.parent.stat()
    if stat.st_uid != os.getuid() or parent.st_uid != os.getuid():
        raise RuntimeError("authority artifact must be owner-controlled")
    if stat.st_mode & 0o077 or parent.st_mode & 0o077:
        raise RuntimeError("authority artifact permissions are too broad")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise RuntimeError("authority artifact schema mismatch")
    return value


def load_authority(
    identity_path: Path, attestation_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    identity = _owner_json(identity_path, schema="panda-qwen-model-identity/v2")
    if identity.get("authority") != AUTHORITY:
        raise RuntimeError("Qwen snapshot authority mismatch")
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    attestation = _owner_json(
        attestation_path, schema="panda-qwen-server-attestation/v1"
    )
    if (
        attestation.get("identity_manifest_sha256")
        != hashlib.sha256(encoded).hexdigest()
    ):
        raise RuntimeError("server attestation does not bind the identity manifest")
    if attestation.get("snapshot_digest") != identity.get("snapshot_digest"):
        raise RuntimeError("server snapshot digest mismatch")
    if attestation.get("snapshot_path") != identity.get("snapshot_path"):
        raise RuntimeError("server snapshot path mismatch")
    return identity, attestation


def _start_ticks(pid: int, proc_root: Path) -> str:
    fields = (proc_root / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
    return fields[19]


def verify_process(
    attestation: Mapping[str, object], *, proc_root: Path = Path("/proc")
) -> None:
    pid = attestation.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("attested process ID is invalid")
    process = proc_root / str(pid)
    try:
        if process.stat().st_uid != os.getuid():
            raise RuntimeError("attested process is not owner-controlled")
        if _start_ticks(pid, proc_root) != attestation.get("process_start_ticks"):
            raise RuntimeError("server attestation is stale")
        argv = [
            item.decode()
            for item in (process / "cmdline").read_bytes().split(b"\0")
            if item
        ]
    except FileNotFoundError as error:
        raise RuntimeError("attested Qwen process is unavailable") from error
    expected = {
        "--host": "127.0.0.1",
        "--port": str(attestation.get("port")),
        "--served-model-name": str(attestation.get("served_model_id")),
        "--dtype": AUTHORITY["dtype"],
    }
    for flag, value in expected.items():
        if (
            flag not in argv
            or argv.index(flag) + 1 >= len(argv)
            or argv[argv.index(flag) + 1] != value
        ):
            raise RuntimeError(f"attested Qwen process has wrong {flag}")
    if attestation.get("snapshot_path") not in argv:
        raise RuntimeError("attested Qwen process has the wrong snapshot")


class QwenClient:
    """Serial, greedy, three-image client with per-request live identity checks."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        identity: Mapping[str, object],
        attestation: Mapping[str, object],
        transport: httpx.BaseTransport | None = None,
        proc_root: Path = Path("/proc"),
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Qwen endpoint must be loopback HTTP")
        if (parsed.port or 80) != attestation.get("port"):
            raise RuntimeError("Qwen endpoint does not match the attestation")
        if identity.get("authority") != AUTHORITY or not api_key:
            raise RuntimeError("Qwen client authority is incomplete")
        self.identity = dict(identity)
        self.attestation = dict(attestation)
        self.proc_root = proc_root
        self._http = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
            transport=transport,
        )
        self._lock = threading.Lock()
        self._verified = False
        self._api_key = api_key

    def verify(self) -> None:
        verify_process(self.attestation, proc_root=self.proc_root)
        response = self._http.get("models")
        response.raise_for_status()
        identifiers = {row.get("id") for row in response.json().get("data", [])}
        if identifiers != {self.attestation.get("served_model_id")}:
            raise RuntimeError("live Qwen model identity mismatch")
        verify_process(self.attestation, proc_root=self.proc_root)
        self._verified = True

    def complete(
        self,
        *,
        observation_id: str,
        system_prompt: str,
        instruction: str,
        public_state: Mapping[str, object],
        images: Mapping[str, bytes],
        image_roles: Mapping[str, str] | None = None,
        response_schema: Mapping[str, object] | None = None,
        attempt_index: int = 0,
        attempt_log: AttemptEvidenceLog | None = None,
        max_tokens: int = 1024,
    ) -> Response:
        if set(images) != {"left", "right", "wrist"}:
            raise ValueError("exactly left, right, and wrist image slots are required")
        roles = (
            {"left": "official_left", "right": "official_right", "wrist": "official_wrist"}
            if image_roles is None
            else dict(image_roles)
        )
        if set(roles) != set(images) or not all(
            isinstance(role, str) and role for role in roles.values()
        ):
            raise ValueError("image roles must describe exactly the three image slots")
        if not self._verified:
            raise RuntimeError("Qwen identity must be verified before inference")
        if not isinstance(attempt_index, int) or attempt_index < 0:
            raise ValueError("attempt index is invalid")
        if not isinstance(max_tokens, int) or not 32 <= max_tokens <= 4096:
            raise ValueError("max tokens is outside the reviewed bound")
        verify_process(self.attestation, proc_root=self.proc_root)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("only one Qwen request may be in flight")
        try:
            hashes = {name: hashlib.sha256(images[name]).hexdigest() for name in images}
            content: list[dict[str, object]] = [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "observation_id": observation_id,
                            "instruction": instruction,
                            "public_state": public_state,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            ]
            for name in ("left", "right", "wrist"):
                data = base64.b64encode(images[name]).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{data}",
                            "detail": "high",
                        },
                    }
                )
            response_format: dict[str, object]
            if response_schema is None:
                response_format = {"type": "json_object"}
            else:
                schema = dict(response_schema)
                if schema.get("type") != "object":
                    raise ValueError("response schema must describe one JSON object")
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "robocasa_inspect_command",
                        "schema": schema,
                        "strict": True,
                    },
                }
            payload = {
                "model": self.attestation["served_model_id"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "seed": 3074294,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": response_format,
            }
            request_sha256 = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            response_schema_sha256 = (
                None
                if response_schema is None
                else hashlib.sha256(
                    json.dumps(
                        response_schema,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
            started = time.monotonic()
            response = self._http.post("chat/completions", json=payload)
            latency = time.monotonic() - started
            if response.is_error and attempt_log is not None:
                attempt_log.append(
                    {
                        "request_sha256": request_sha256,
                        "observation_id": observation_id,
                        "attempt_index": attempt_index,
                        "raw_body_sha256": hashlib.sha256(
                            response.content
                        ).hexdigest(),
                        "sanitized_raw_command": response.text.replace(
                            self._api_key, "[REDACTED]"
                        ),
                        "http_status": response.status_code,
                        "finish_reason": None,
                        "completion_tokens": None,
                        "latency_s": latency,
                        "response_schema_sha256": response_schema_sha256,
                        "served_model_id": self.attestation.get("served_model_id"),
                    }
                )
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError as error:
                if attempt_log is not None:
                    attempt_log.append(
                        {
                            "request_sha256": request_sha256,
                            "observation_id": observation_id,
                            "attempt_index": attempt_index,
                            "raw_body_sha256": hashlib.sha256(
                                response.content
                            ).hexdigest(),
                            "sanitized_raw_command": response.text.replace(
                                self._api_key, "[REDACTED]"
                            ),
                            "http_status": response.status_code,
                            "finish_reason": "invalid_http_json",
                            "completion_tokens": None,
                            "latency_s": latency,
                            "response_schema_sha256": response_schema_sha256,
                            "served_model_id": self.attestation.get(
                                "served_model_id"
                            ),
                        }
                    )
                raise MalformedResponse(
                    "Qwen returned a malformed HTTP body",
                    evidence={
                        "latency_s": latency,
                        "finish_reason": "invalid_http_json",
                        "raw_sha256": hashlib.sha256(response.content).hexdigest(),
                    },
                ) from error
            choices = body.get("choices")
            choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
            raw_candidate = (
                choice.get("message", {}).get("content")
                if isinstance(choice, Mapping)
                and isinstance(choice.get("message"), Mapping)
                else None
            )
            raw_for_evidence = (
                raw_candidate if isinstance(raw_candidate, str) else response.text
            )
            finish_reason = (
                choice.get("finish_reason") if isinstance(choice, Mapping) else None
            )
            usage = body.get("usage", {})
            if attempt_log is not None:
                attempt_log.append(
                    {
                        "request_sha256": request_sha256,
                        "observation_id": observation_id,
                        "attempt_index": attempt_index,
                        "raw_body_sha256": hashlib.sha256(
                            response.content
                        ).hexdigest(),
                        "sanitized_raw_command": raw_for_evidence.replace(
                            self._api_key, "[REDACTED]"
                        ),
                        "http_status": response.status_code,
                        "finish_reason": finish_reason,
                        "completion_tokens": (
                            usage.get("completion_tokens")
                            if isinstance(usage, Mapping)
                            else None
                        ),
                        "latency_s": latency,
                        "response_schema_sha256": response_schema_sha256,
                        "served_model_id": self.attestation.get("served_model_id"),
                    }
                )
            preliminary_evidence = {
                "latency_s": latency,
                "finish_reason": finish_reason,
                "raw_sha256": hashlib.sha256(raw_for_evidence.encode()).hexdigest(),
                "raw_chars": len(raw_for_evidence),
            }
            if choice is None:
                raise MalformedResponse(
                    "Qwen response must contain one choice",
                    evidence=preliminary_evidence,
                )
            raw = raw_candidate
            if not isinstance(raw, str):
                raise MalformedResponse(
                    "Qwen response content is not text",
                    evidence=preliminary_evidence,
                )
            evidence = {
                "latency_s": latency,
                "system_prompt_sha256": hashlib.sha256(
                    system_prompt.encode()
                ).hexdigest(),
                "instruction_sha256": hashlib.sha256(
                    instruction.encode()
                ).hexdigest(),
                "image_sha256": hashes,
                "image_roles": roles,
                "usage": usage,
                "finish_reason": finish_reason,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "raw_chars": len(raw),
                "snapshot_digest": self.identity.get("snapshot_digest"),
                "served_model_id": self.attestation.get("served_model_id"),
                "decoding": {
                    "seed": 3074294,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": max_tokens,
                    "enable_thinking": False,
                },
                "response_schema_sha256": response_schema_sha256,
            }
            try:
                command = json.loads(raw.rsplit("</think>", 1)[-1].strip())
            except json.JSONDecodeError as error:
                raise MalformedResponse(
                    "Qwen returned malformed JSON", evidence=evidence
                ) from error
            if not isinstance(command, dict):
                raise MalformedResponse(
                    "Qwen returned a non-object command", evidence=evidence
                )
            verify_process(self.attestation, proc_root=self.proc_root)
            return Response(command=command, evidence=evidence)
        finally:
            self._lock.release()

    def close(self) -> None:
        self._http.close()


def complete_with_repair(
    client: QwenClient,
    *,
    observation_id: str,
    system_prompt: str,
    instruction: str,
    public_state: Mapping[str, object],
    images: Mapping[str, bytes],
    attempt_log: AttemptEvidenceLog,
    attempts_available: int,
    wall_deadline_s: float,
    max_tokens: int,
    image_roles: Mapping[str, str] | None = None,
    response_schema: Mapping[str, object] | None = None,
) -> Response:
    """Retry malformed output on one immutable observation without taking action."""
    if not 1 <= attempts_available <= 3:
        raise ValueError("repair attempts must be between one and three")
    retry_suffix = "previous response was not schema-valid; re-emit"
    finish_classes: list[str] = []
    failures: list[Mapping[str, object]] = []
    for attempt_index in range(attempts_available):
        if time.monotonic() >= wall_deadline_s:
            raise TransportFailure(
                "Qwen repair exceeded the wall budget",
                attempts=attempt_index,
                finish_classes=[*finish_classes, "wall_budget_exhausted"],
                evidence=failures,
            )
        attempt_instruction = (
            instruction
            if attempt_index == 0
            else f"{instruction}\n{retry_suffix}"
        )
        try:
            response = client.complete(
                observation_id=observation_id,
                system_prompt=system_prompt,
                instruction=attempt_instruction,
                public_state=public_state,
                images=images,
                image_roles=image_roles,
                response_schema=response_schema,
                attempt_index=attempt_index,
                attempt_log=attempt_log,
                max_tokens=max_tokens,
            )
        except MalformedResponse as error:
            finish_reason = error.evidence.get("finish_reason")
            finish_classes.append(
                "length" if finish_reason == "length" else "malformed_json"
            )
            failures.append(error.evidence)
            continue
        except (httpx.HTTPError, TimeoutError) as error:
            finish_classes.append("transport_error")
            failures.append({"error_type": type(error).__name__})
            continue
        if time.monotonic() >= wall_deadline_s:
            raise TransportFailure(
                "Qwen repair exceeded the wall budget",
                attempts=attempt_index + 1,
                finish_classes=[*finish_classes, "wall_budget_exhausted"],
                evidence=[*failures, response.evidence],
            )
        evidence = dict(response.evidence)
        evidence.update(
            {
                "transport_attempts": attempt_index + 1,
                "transport_repaired": attempt_index > 0,
                "transport_finish_classes": finish_classes,
            }
        )
        return Response(command=response.command, evidence=evidence)
    raise TransportFailure(
        "Qwen transport exhausted same-frame schema repair",
        attempts=attempts_available,
        finish_classes=finish_classes,
        evidence=failures,
    )
