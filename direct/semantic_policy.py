"""Single-token guided-choice control call (Show-Harness style) over our Qwen client."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Mapping, Sequence

from .policy import DECODING, InfrastructureError, MalformedOutput, QwenDirectClient
from .semantic import TOKENS, parse_token


class SemanticPolicy:
    def __init__(self, client: QwenDirectClient) -> None:
        self.client = client

    def choose(self, *, system_prompt: str, user_text: str, images: Sequence[tuple[str, bytes]],
               decision: int, category: str = "control", choices: Sequence[str] = TOKENS,
               schema: Mapping[str, object] | None = None) -> dict:
        """One counted call. Without ``schema``: vLLM guided_choice over the vocabulary.
        With ``schema``: strict JSON (must contain a "token" field) so a plugin can add fields."""
        content: list[dict] = [{"type": "text", "text": user_text}]
        for label, data in images:
            content.append({"type": "text", "text": f"[{label}]"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"), "detail": "high"}})
        payload: dict = {"model": self.client.model,
                         "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
                         **DECODING, "chat_template_kwargs": {"enable_thinking": False}}
        if schema is None:
            payload.update({"max_tokens": 24, "guided_choice": list(choices)})
        else:
            payload.update({"max_tokens": 96, "response_format": {"type": "json_schema", "json_schema": {
                "name": "semantic_step", "schema": dict(schema), "strict": True}}})
        started = time.monotonic()
        response = self.client.http.post("chat/completions", json=payload)
        latency = time.monotonic() - started
        self.client.calls += 1
        self.client.latency_s += latency
        record: dict = {"decision": decision, "category": category, "latency_s": round(latency, 2),
                        "http_status": response.status_code, "user_text": user_text,
                        "image_labels": [label for label, _ in images],
                        "image_sha256": [hashlib.sha256(data).hexdigest()[:16] for _, data in images]}
        if response.is_error:
            record["error"] = response.text[:500]
            self.client._log(record)
            raise InfrastructureError(f"Qwen HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        raw = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage", {})
        self.client.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.client.completion_tokens += int(usage.get("completion_tokens", 0))
        bucket = self.client.by_category.setdefault(category, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0})
        bucket["calls"] += 1; bucket["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        bucket["completion_tokens"] += int(usage.get("completion_tokens", 0)); bucket["latency_s"] += latency
        record.update({"raw": raw, "usage": usage})
        extra: dict = {}
        try:
            if schema is None:
                token = parse_token(raw)
            else:
                parsed = json.loads(raw.rsplit("</think>", 1)[-1].strip())
                token = parse_token(str(parsed.get("token", "")))
                extra = {k: v for k, v in parsed.items() if k != "token"}
        except (ValueError, json.JSONDecodeError, AttributeError):
            record["parse_error"] = True
            self.client._log(record)
            raise MalformedOutput(f"not a token: {raw!r}", record)
        record["token"] = token
        record.update(extra)
        self.client._log(record)
        return {"token": token, "raw": raw, "latency_s": latency, "usage": usage, **extra}


def build_controller_prompt(template: str, *, task: str, subtask: str, proprio: Mapping[str, object],
                            recent_moves: Sequence[str], fine: bool) -> str:
    return template.format(task=task, subtask=subtask or "(none)",
                           proprio=json.dumps(proprio, separators=(",", ":")),
                           recent_moves=" ".join(list(recent_moves)[-8:]) or "(none)",
                           step_note=("2 cm per step (target visible in wrist view)" if fine
                                      else "4 cm per step (target not yet in wrist view)"))


def image_pair(images: Mapping[str, bytes], agent_view: str) -> list[tuple[str, bytes]]:
    return [("Image A: agent view", images[agent_view]), ("Image B: wrist view", images["wrist"])]
