"""Fixed Qwen client for direct numerical control.

Reuses the released identity/attestation checks and decoding settings
(seed 3074294, temperature 0, top_p 1, thinking disabled, strict JSON schema).
Shared adjustment relative to the released three-image client: any number of
labelled images may be attached, so the clean window can carry the current
and previous three views. Raw responses, usage and finish reasons are kept.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from robocasa_inspect.model_client import load_authority, verify_process

DECODING = {"seed": 3074294, "temperature": 0, "top_p": 1}
MAX_TOKENS_SHORT = 1024
MAX_TOKENS_FULL = 4096


class MalformedOutput(RuntimeError):
    def __init__(self, message: str, record: dict) -> None:
        super().__init__(message)
        self.record = record


class QwenDirectClient:
    def __init__(self, *, log_path: Path) -> None:
        identity_path = Path(os.environ.get("QWEN_IDENTITY_MANIFEST", "/home/jli/state/panda-qwen38/identity.json"))
        attestation_path = Path(os.environ.get("QWEN_SERVER_ATTESTATION", "/home/jli/state/panda-qwen38/server-attestation.json"))
        token_path = Path(os.environ.get("QWEN_API_TOKEN_FILE", "/home/jli/state/panda-qwen38/api-token"))
        self.identity, self.attestation = load_authority(identity_path, attestation_path)
        verify_process(self.attestation)
        token = token_path.read_text().strip()
        self._token = token
        self.model = str(self.attestation["served_model_id"])
        self.http = httpx.Client(base_url="http://127.0.0.1:8002/v1/", headers={"Authorization": f"Bearer {token}"}, timeout=300)
        response = self.http.get("models")
        response.raise_for_status()
        if {row.get("id") for row in response.json().get("data", [])} != {self.model}:
            raise RuntimeError("live Qwen model identity mismatch")
        self.log = log_path.open("a")
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_s = 0.0

    def complete(self, *, system_prompt: str, user_text: str, images: Sequence[tuple[str, bytes]],
                 response_schema: Mapping[str, object], max_tokens: int, category: str,
                 decision: int) -> dict:
        """One counted call. Returns {"parsed", "raw", "usage", "finish_reason", ...}."""
        content: list[dict[str, object]] = [{"type": "text", "text": user_text}]
        for label, data in images:
            content.append({"type": "text", "text": f"[image: {label}]"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"), "detail": "high"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            **DECODING, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "direct_control", "schema": dict(response_schema), "strict": True}},
        }
        verify_process(self.attestation)
        started = time.monotonic()
        response = self.http.post("chat/completions", json=payload)
        latency = time.monotonic() - started
        self.calls += 1
        self.latency_s += latency
        record: dict[str, object] = {
            "decision": decision, "category": category, "latency_s": round(latency, 2),
            "http_status": response.status_code, "max_tokens": max_tokens,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest()[:16],
            "user_text": user_text, "image_labels": [label for label, _ in images],
            "image_sha256": [hashlib.sha256(data).hexdigest()[:16] for _, data in images],
        }
        if response.is_error:
            record.update({"error": response.text.replace(self._token, "[REDACTED]")[:2000]})
            self._log(record)
            raise MalformedOutput(f"Qwen HTTP {response.status_code}", record)
        body = response.json()
        choice = body["choices"][0]
        raw = choice["message"]["content"] or ""
        usage = body.get("usage", {})
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        record.update({"raw": raw, "finish_reason": choice.get("finish_reason"), "usage": usage})
        parsed = None
        try:
            parsed = json.loads(raw.rsplit("</think>", 1)[-1].strip())
        except json.JSONDecodeError:
            record["parse_error"] = True
        record["parsed"] = parsed
        record["truncated"] = choice.get("finish_reason") == "length"
        self._log(record)
        if parsed is None or not isinstance(parsed, dict):
            raise MalformedOutput("Qwen output is not a JSON object" + (" (truncated)" if record["truncated"] else ""), record)
        return record

    def _log(self, record: Mapping[str, object]) -> None:
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()

    def totals(self) -> dict:
        return {"qwen_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "qwen_latency_s": round(self.latency_s, 1)}

    def close(self) -> None:
        self.http.close()
        self.log.close()
