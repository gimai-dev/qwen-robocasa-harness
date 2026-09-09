#!/usr/bin/env python3
"""Run Cycle 11's frozen 900-call Qwen semantic discrimination gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from replay_demonstration import _qwen_client
from robocasa_inspect.cycle11_policy import (
    CYCLE11_SYSTEM_PROMPT,
    cycle11_response_schema,
    decode_cycle11_decision,
)
from robocasa_inspect.cycle11_preflight import (
    canonical_sha256,
    score_semantic_attempts,
    validate_semantic_corpus,
)
from robocasa_inspect.cycle12_policy import (
    CYCLE12_SYSTEM_PROMPT,
    cycle12_response_schema,
    decode_cycle12_decision,
)
from robocasa_inspect.cycle12_preflight import score_cycle12_attempts
from robocasa_inspect.cycle12_transport import validate_cycle12_transport_amendment
from robocasa_inspect.cycle13_policy import (
    CYCLE13_SYSTEM_PROMPT,
    cycle13_response_schema,
    decode_cycle13_decision,
)
from robocasa_inspect.cycle13_transport import validate_cycle13_transport
from robocasa_inspect.goal30_cycle11 import validate_cycle11_contract
from robocasa_inspect.goal30_cycle12 import validate_cycle12_contract
from robocasa_inspect.goal30_cycle13 import validate_cycle13_contract
from robocasa_inspect.model_client import AttemptEvidenceLog, MalformedResponse
from robocasa_inspect.runner import _atomic_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _instruction(case: dict[str, object], variant: str, *, protocol: str) -> str:
    action = (
        "authorize_source"
        if protocol in {"cycle12", "cycle13"}
        else "execute_source"
    )
    base = (
        f"Task: {case['task']}. Decide whether the proposed SOURCE REFERENCE is "
        f"the same-task, same-layout current/next step. {action} authorizes "
        "the source; reobserve and give_up authorize no motion. Use zero residual "
        "for either no-motion skill."
    )
    if variant == "unmasked":
        return base + " Public EEF error and source index are supplied in state."
    if variant == "masked":
        return base + " Judge from exactly the three images and receipts."
    if variant == "invalid":
        return base + " The numeric EEF error is intentionally execute-range; verify pixels."
    raise ValueError("Cycle 11 preflight variant drifted")


def _public_state(case: dict[str, object], variant: str) -> dict[str, object]:
    state: dict[str, object] = {
        "task_wording": str(case["task"]),
        "recent_receipts": [],
    }
    if variant != "masked":
        state.update(
            {
                "public_eef_translation_error_m": case["translation_error_m"],
                "public_eef_rotation_error_rad": case["rotation_error_rad"],
                "source_keyframe_index": case["source_frame_index"],
            }
        )
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    contracts = parser.add_mutually_exclusive_group(required=True)
    contracts.add_argument("--cycle11-contract", type=Path)
    contracts.add_argument("--cycle12-contract", type=Path)
    contracts.add_argument("--cycle13-contract", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--transport-amendment", type=Path)
    args = parser.parse_args()
    protocol = (
        "cycle13"
        if args.cycle13_contract is not None
        else "cycle12"
        if args.cycle12_contract is not None
        else "cycle11"
    )
    contract_path = args.cycle13_contract or args.cycle12_contract or args.cycle11_contract
    if contract_path is None:
        raise RuntimeError("semantic preflight contract is missing")
    contract = json.loads(contract_path.read_text())
    corpus = json.loads(args.corpus.read_text())
    if protocol == "cycle13":
        validate_cycle13_contract(contract)
        if args.transport_amendment is None:
            raise RuntimeError("Cycle 13 transport authority is required")
        transport_amendment = json.loads(args.transport_amendment.read_text())
        validate_cycle13_transport(
            transport_amendment, contract_sha256=str(contract["sha256"])
        )
        if args.max_tokens != transport_amendment["census"]["max_tokens"]:
            raise RuntimeError("Cycle 13 max_tokens does not match authority")
    elif protocol == "cycle12":
        validate_cycle12_contract(contract)
        if args.transport_amendment is None:
            raise RuntimeError("Cycle 12 transport amendment is required")
        transport_amendment = json.loads(args.transport_amendment.read_text())
        validate_cycle12_transport_amendment(
            transport_amendment, contract_sha256=str(contract["sha256"])
        )
        if args.max_tokens != transport_amendment["census"]["max_tokens"]:
            raise RuntimeError("Cycle 12 max_tokens does not match the amendment")
    else:
        validate_cycle11_contract(contract)
        transport_amendment = None
    validate_semantic_corpus(corpus)
    if protocol == "cycle13":
        if corpus.get("sha256") != contract["parents"]["cycle11_corpus_sha256"]:
            raise RuntimeError("Cycle 13 corpus binding drifted")
    else:
        expected_cycle11 = (
            contract["parents"]["cycle11_contract_sha256"]
            if protocol == "cycle12"
            else contract["sha256"]
        )
        if corpus.get("cycle11_contract_sha256") != expected_cycle11:
            raise RuntimeError("Cycle 11 corpus contract binding drifted")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    attempt_log = AttemptEvidenceLog(run_dir / "model-attempts.jsonl")
    corpus_root = args.corpus.parent / "cases"
    schedule: list[tuple[str, dict[str, object]]] = []
    for variant in ("unmasked", "masked"):
        schedule.extend(
            (variant, row)
            for row in corpus["cases"]
            if row["kind"] in {"aligned", "valid_next"}
        )
    schedule.extend(
        ("invalid", row)
        for row in corpus["cases"]
        if row["kind"] == "invalid_reference"
    )
    if len(schedule) != 900:
        raise RuntimeError("Cycle 11 semantic schedule is not exactly 900 calls")
    if not 1 <= args.shard_count <= 8 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Cycle 11 shard authority is invalid")
    scheduled = [
        (index, variant, case)
        for index, (variant, case) in enumerate(schedule)
        if index % args.shard_count == args.shard_index
    ]
    attempts: list[dict[str, object]] = []
    client = _qwen_client()
    try:
        for completed, (index, variant, case) in enumerate(scheduled, start=1):
            case_id = str(case["case_id"])
            observation_id = hashlib.sha256(
                f"{protocol}/{contract['sha256']}/{variant}/{case_id}".encode()
            ).hexdigest()
            token = hashlib.sha256((observation_id + "skill").encode()).hexdigest()[:12]
            images = {
                name: (corpus_root / case_id / f"{name}.png").read_bytes()
                for name in ("left", "right", "wrist")
            }
            if {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in images.items()
            } != case["image_sha256"]:
                raise RuntimeError("Cycle 11 case image authority drifted")
            started = time.monotonic()
            record: dict[str, object] = {
                "index": index,
                "case_id": case_id,
                "variant": variant,
                "kind": case["kind"],
                "distinct_frame": case.get("distinct_frame", False),
                "schema_valid": False,
                "transport_failure": False,
            }
            try:
                response = client.complete(
                    observation_id=observation_id,
                    system_prompt=(
                        CYCLE13_SYSTEM_PROMPT
                        if protocol == "cycle13"
                        else CYCLE12_SYSTEM_PROMPT
                        if protocol == "cycle12"
                        else CYCLE11_SYSTEM_PROMPT
                    ),
                    instruction=_instruction(case, variant, protocol=protocol),
                    public_state=_public_state(case, variant),
                    images=images,
                    image_roles={
                        "left": "official_left_current_source_panel",
                        "right": "official_right_current_source_panel",
                        "wrist": "official_current_wrist",
                    },
                    response_schema=(
                        cycle13_response_schema(token)
                        if protocol == "cycle13"
                        else cycle12_response_schema(token)
                        if protocol == "cycle12"
                        else cycle11_response_schema(token)
                    ),
                    attempt_index=0,
                    attempt_log=attempt_log,
                    max_tokens=args.max_tokens,
                )
                decision = (
                    decode_cycle13_decision(
                        response.command, observation_token=token
                    )
                    if protocol == "cycle13"
                    else decode_cycle12_decision(
                        response.command, observation_token=token
                    )
                    if protocol == "cycle12"
                    else decode_cycle11_decision(
                        response.command, observation_token=token
                    )
                )
                record.update(
                    {
                        "schema_valid": True,
                        "skill": decision.skill,
                        "residual_id": decision.residual_id,
                        "response_evidence_sha256": canonical_sha256(response.evidence),
                    }
                )
            except (MalformedResponse, ValueError, httpx.HTTPError, TimeoutError) as error:
                record.update(
                    {
                        "transport_failure": isinstance(error, httpx.HTTPError),
                        "error_type": type(error).__name__,
                    }
                )
            record["latency_s"] = time.monotonic() - started
            attempts.append(record)
            if completed % 20 == 0:
                _atomic_json(
                    run_dir / "progress.json",
                    {
                        "schema": "robocasa-inspect-cycle11-preflight-progress/v1",
                        "completed": completed,
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                        "schema_valid": sum(
                            row["schema_valid"] is True for row in attempts
                        ),
                        "transport_failures": sum(
                            row["transport_failure"] is True for row in attempts
                        ),
                    },
                )
    finally:
        client.close()
    score = (
        (
            score_cycle12_attempts(attempts)
            if protocol in {"cycle12", "cycle13"}
            else score_semantic_attempts(attempts)
        )
        if args.shard_count == 1
        else None
    )
    schema_prefix = f"robocasa-inspect-{protocol}-semantic-preflight"
    result = {
        "schema": (
            f"{schema_prefix}/v1"
            if args.shard_count == 1
            else f"{schema_prefix}-shard/v1"
        ),
        "complete": True,
        "passed": score["passed"] if score is not None else None,
        f"{protocol}_contract_sha256": contract["sha256"],
        "corpus_sha256": corpus["sha256"],
        "corpus_file_sha256": _sha256(args.corpus),
        "model_identity": client.identity["authority"],
        "served_model_id": client.attestation["served_model_id"],
        "simulator_actions": 0,
        "task_outcomes_used": False,
        "transport_amendment_sha256": (
            None
            if transport_amendment is None
            else transport_amendment["sha256"]
        ),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "calls": len(attempts),
        "score": score,
        "request_schedule_sha256": canonical_sha256(
            [(variant, row["case_id"]) for variant, row in schedule]
        ),
        "attempt_log_sha256": _sha256(run_dir / "model-attempts.jsonl"),
        "attempts": attempts,
    }
    _atomic_json(run_dir / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "attempts"}, sort_keys=True))
    if score is not None and score["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
