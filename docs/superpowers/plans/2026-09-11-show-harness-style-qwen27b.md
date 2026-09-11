# Show-Harness-style zero-shot control with Qwen3.8-27B: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the Show-Harness interface (discrete semantic action units + interpreter, wrist-view closed loop, subtask plan, grasp recovery) rescues zero-shot Qwen3.8-27B, on two testbeds: Show-Harness's own ManiSkill block pick-and-place (their code, our model) and our RoboCasa loop (our code, their interface), compared against our numerical-control clean baseline on the same starts.

**Architecture:** Lane A installs the Show-Harness repo on h200-4 and points its vLLM provider at the existing Qwen server; nothing of theirs is modified except config. Lane B adds one new method family to our `direct/` package: a token vocabulary, a deterministic interpreter that maps tokens to our existing slot executor (straight-line EE moves in the robot base frame, gripper, base), a single-token guided-choice policy call, and three plugins (proprioception text, subtask planner with completion checks, grasp-failure recovery). Perception is RGB only: SAM regions are not sent to the model in this family.

**Tech Stack:** Python 3.11 sim venv on h200-4, vLLM `qwen3.8-27b-bf16` on `127.0.0.1:8002` (guided_choice), ManiSkill 3 (Lane A), RoboCasa 1.0.1 via `direct.executor` (Lane B), pytest.

## Global Constraints

- Model: frozen `qwen3.8-27b-bf16`, temperature 0, top_p 1, `enable_thinking: False`, seed 3074294; identity/attestation checks as in `direct/policy.py`. No fine-tuning.
- Zero-shot only: no demonstrations, no bank, no task-specific recipes; the prompt may describe the embodiment and the interface, not any task's solution.
- Every Lane B episode uses the existing budgets (900 sim steps, 1200 s, 180 decisions) and the official RoboCasa predicate; every rejected/invalid token costs a decision.
- All Lane B comparisons use the same starts as the finished campaign: dev seeds 0-2, test seeds 100-119, tasks PickPlaceCounterToSink, PickPlaceCounterToDrawer, PickPlaceStoveToCounter.
- Never rsync into `/home/jli/work/qwen-direct-control` while a matrix runs there; develop in `/home/jli/work/qwen-direct-control-dev`.
- pkill/pgrep patterns must be bracket-escaped (`"[d]irect"`), never a bare log path.
- Lane A must not touch `/home/jli/state/panda-qwen38/` except reading the token; use a separate venv under `/home/jli/work/show-harness/`.

---

## File map

Lane A (no code, config only):
- Create: `/home/jli/work/show-harness/` (git clone of showlab/Show-Harness), `configs/site/qwen27b.yaml` (endpoint overlay), `scripts/run_laneA.sh`.
- Output: `/home/jli/state/show-harness/laneA-<tag>/`.

Lane B (in our repo, `direct/`):
- Create `direct/semantic.py`: vocabulary, token parsing, `Interpreter` (token -> `Action` for `direct.executor.execute`), proprioception text.
- Create `direct/semantic_policy.py`: one guided-choice call (`guided_choice` over the vocabulary) using `QwenDirectClient`'s connection.
- Create `direct/semantic_plugins.py`: `SubgoalPlanner` (one JSON planning call, completion check inside the control call), `Recovery` (empty-close detection -> RELEASE + subtask rollback), `RecentMoves`.
- Create `direct/prompts/semantic_controller.txt`, `direct/prompts/semantic_planner.txt`.
- Create `direct/semantic_episode.py`: episode loop for this family (`--method sem`, `sem+plan`, `sem+plan+rec`, `sem-full` = all plugins), reusing `Simulator`, `execute`, `render_video`, `result.json` schema.
- Modify `direct/matrix.py:19-30`: route methods starting with `sem` to `direct.semantic_episode`.
- Modify `direct/executor.py`: add `ready_pose` action support (task-agnostic pre-episode move out of the singular home posture).
- Tests: `direct/tests/test_semantic.py` (pure), `direct/tests/smoke_semantic.py` (on-box).

---

### Task 1: Lane A install and endpoint wiring (Show-Harness on ManiSkill, our Qwen)

**Files:**
- Create: `/home/jli/work/show-harness/` (clone), `/home/jli/work/show-harness/configs/site/qwen27b.yaml`, `/home/jli/work/show-harness/scripts/run_laneA.sh`

**Interfaces:**
- Consumes: Qwen endpoint `http://127.0.0.1:8002/v1`, token at `/home/jli/state/panda-qwen38/api-token`, served model id from `/home/jli/state/panda-qwen38/server-attestation.json` (`served_model_id`).
- Produces: a runnable batch command that writes per-episode results and a closed-loop success rate under `/home/jli/state/show-harness/laneA-<tag>/`.

- [ ] **Step 1: Clone and install into an isolated venv (on h200-4)**

```bash
ssh h200-4 'mkdir -p /home/jli/work /home/jli/state/show-harness && cd /home/jli/work && git clone https://github.com/showlab/Show-Harness.git show-harness && cd show-harness && git log --oneline -1 && cat docs/simulators.md | head -80 && cat configs/robot_maniskill.yaml'
```
Expected: clone succeeds; `robot_maniskill.yaml` shows `vlm:` / `provider:` / `model:` keys and the plugin block. Record the commit hash in the RUNBOOK.

- [ ] **Step 2: Create the ManiSkill environment their way**

```bash
ssh h200-4 'cd /home/jli/work/show-harness && python3.11 -m venv .venv && . .venv/bin/activate && pip install -U pip && bash scripts/setup.sh base 2>&1 | tail -20 && pip install "mani_skill" 2>&1 | tail -3 && python -c "import mani_skill, gymnasium; print(mani_skill.__version__)"'
```
Expected: prints a ManiSkill version. If `setup.sh` pins a different ManiSkill or asset download, follow `docs/simulators.md` exactly and note the deviation. If the box's EGL setup is needed, export the same variables our launcher uses (`MUJOCO_GL` is irrelevant here; ManiSkill uses SAPIEN/Vulkan: check `vulkaninfo --summary | head -5`; the Vulkan ICD fix from memory `nebius-vulkan-icd-fix` applies if it fails).

- [ ] **Step 3: Write the endpoint overlay**

`/home/jli/work/show-harness/configs/site/qwen27b.yaml` (keys per `core/vlm/vlm_client.py`; adjust names to the loaded `robot_maniskill.yaml` if they differ):
```yaml
vlm:
  provider: vllm
  api_dialect: vllm
  base_url: http://127.0.0.1:8002/v1
  api_key_env: QWEN_API_KEY
  model: __FILL_FROM_ATTESTATION__
  temperature: 0.0
  max_tokens: 24
  chat_template_kwargs:
    enable_thinking: false
```
Fill `model` with:
```bash
ssh h200-4 'python3 -c "import json;print(json.load(open(\"/home/jli/state/panda-qwen38/server-attestation.json\"))[\"served_model_id\"])"'
```

- [ ] **Step 4: Verify one call goes through their client with guided choice**

```bash
ssh h200-4 'cd /home/jli/work/show-harness && . .venv/bin/activate && export QWEN_API_KEY=$(cat /home/jli/state/panda-qwen38/api-token) && python - <<EOF
from core.vlm.vlm_client import VLMClient
import yaml
cfg = yaml.safe_load(open("configs/site/qwen27b.yaml"))["vlm"]
c = VLMClient(**{k: v for k, v in cfg.items() if k in VLMClient.__init__.__code__.co_varnames})
print(c.complete_text("Reply with the single token MV_UP", choices=["MV_UP","MV_DOWN"]) if hasattr(c, "complete_text") else "adapt: inspect VLMClient public methods")
EOF'
```
Expected: `MV_UP`. If the constructor/method names differ, read `core/vlm/vlm_client.py` and `core/runners/mvtoken.py` for the exact call used by the runner and use that; the requirement is one successful guided-choice completion against our server.

- [ ] **Step 5: Smoke one episode**

```bash
ssh h200-4 'cd /home/jli/work/show-harness && . .venv/bin/activate && export QWEN_API_KEY=$(cat /home/jli/state/panda-qwen38/api-token) && python scripts/run_maniskill_mvtoken.py --version v4 --config configs/robot_maniskill.yaml --site configs/site/qwen27b.yaml --max-steps 60 --out /home/jli/state/show-harness/smoke 2>&1 | tail -30'
```
Expected: an episode runs to DONE or 60 steps; a results JSON and a video/frames folder appear under `--out`. If the runner has no `--site` flag, follow `configs/README.md` for overlay loading (likely `SHOW_HARNESS_SITE=...` or `--overlay`). Record the exact working command.

- [ ] **Step 6: Write the batch script and commit (in our repo, as documentation)**

`/home/jli/work/show-harness/scripts/run_laneA.sh`:
```bash
#!/bin/bash
# Lane A: Show-Harness zero-shot on ManiSkill BlockPAP-v1 with our Qwen3.8-27B.
set -e
cd /home/jli/work/show-harness && . .venv/bin/activate
export QWEN_API_KEY=$(cat /home/jli/state/panda-qwen38/api-token)
TAG=${1:-qwen27b-default}; N=${2:-30}
bash scripts/maniskill/eval_batch.sh configs/robot_maniskill.yaml qwen27b "$N" 60 "$TAG" 2>&1 | tee /home/jli/state/show-harness/laneA-$TAG.log
```
Copy the working commands into `RUNBOOK.md` (new section "Lane A: Show-Harness on ManiSkill") and commit:
```bash
cd "/Users/jiachen/Desktop/first try/qwen-direct-control" && git add RUNBOOK.md && git commit -m "RUNBOOK: Lane A Show-Harness install and run commands

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Lane A experiments (their method, our model)

**Files:** outputs only under `/home/jli/state/show-harness/`.

**Interfaces:**
- Consumes: Task 1's `run_laneA.sh`.
- Produces: `results-direct/laneA/*.md` tables in our repo.

- [ ] **Step 1: Default plugin set, 30 episodes, BlockPAP-v1**

```bash
ssh h200-4 'nohup bash /home/jli/work/show-harness/scripts/run_laneA.sh qwen27b-default 30 > /dev/null 2>&1 &'
```
Expected: about 30 × 3 min. Success rate printed by `eval_batch.sh`.

- [ ] **Step 2: Ablation pair that the paper says matters most**

Run two more 30-episode batches editing the `plugins:` block of a copied config: (a) all plugins off (their "byte-identical" baseline), (b) default plus `variable_step` and `recovery` on if not already default. Tags `qwen27b-noplugins`, `qwen27b-full`.

- [ ] **Step 3: Frontier reference (optional, needs an API key from Jiachen)**

If `GEMINI_API_KEY` or an OpenAI key is provided, run 30 episodes with `provider: gemini` (or `openai`) on the default config, tag `frontier-default`. This separates "model" from "interface". Skip and record "not run: no key" otherwise.

- [ ] **Step 4: Collect and record**

```bash
ssh h200-4 'for t in qwen27b-default qwen27b-noplugins qwen27b-full frontier-default; do echo "== $t"; grep -i "success" /home/jli/state/show-harness/laneA-$t.log | tail -2; done'
```
Write `results-direct/laneA/summary.md` with success/attempts per tag, mean steps, mean VLM calls; commit.

Acceptance: a table with Qwen-27B zero-shot success on their benchmark under their default harness, with the paper's 96% (Gemini) as the reference row.

---

### Task 3: Vocabulary and interpreter (Lane B core, pure code, TDD)

**Files:**
- Create: `direct/semantic.py`
- Test: `direct/tests/test_semantic.py`

**Interfaces:**
- Produces: `TOKENS: tuple[str,...]`, `parse_token(text) -> str` (raises `ValueError`), `Interpreter(step_fine_m=0.02, step_coarse_m=0.04, yaw_step_rad=0.26, base_step_v=0.5)` with `to_action(token, state, *, fine: bool) -> Action | None` (None for DONE) and `proprio_text(observation, receipt) -> dict`.
- Consumes: `direct.actions.Action`, `direct.kinematics.quat_xyzw_to_matrix`, `matrix_to_quat_xyzw`.

- [ ] **Step 1: Write the failing tests**

`direct/tests/test_semantic.py`:
```python
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from direct.semantic import TOKENS, parse_token, Interpreter

STATE = {"tcp_world_position_m": [1.0, 2.0, 1.2], "tcp_world_quat_xyzw": [1.0, 0.0, 0.0, 0.0],
         "base_world_yaw_rad": 0.0, "gripper_width_m": 0.078}

def test_vocabulary():
    assert TOKENS == ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
                      "ROTATE_CW", "ROTATE_CCW", "GRASP", "RELEASE", "DONE",
                      "BASE_FWD", "BASE_BACK", "BASE_LEFT", "BASE_RIGHT")

def test_parse_token_strips_and_validates():
    assert parse_token(" MV_UP\n") == "MV_UP"
    try:
        parse_token("MOVE UP"); assert False
    except ValueError:
        pass

def test_move_fine_is_two_cm_in_base_frame():
    a = Interpreter().to_action("MV_FWD", STATE, fine=True)
    assert a.kind == "ee" and abs(a.position_m[0] - 1.02) < 1e-9 and abs(a.position_m[1] - 2.0) < 1e-9

def test_move_uses_base_heading():
    s = dict(STATE, base_world_yaw_rad=math.pi / 2)
    a = Interpreter().to_action("MV_FWD", s, fine=False)          # coarse 4 cm along +y world
    assert abs(a.position_m[0] - 1.0) < 1e-9 and abs(a.position_m[1] - 2.04) < 1e-9

def test_rotate_changes_only_yaw_about_tool_axis():
    a = Interpreter().to_action("ROTATE_CW", STATE, fine=True)
    assert a.kind == "ee" and a.position_m == (1.0, 2.0, 1.2) and a.quat_xyzw != (1.0, 0.0, 0.0, 0.0)

def test_gripper_tokens_and_done():
    i = Interpreter()
    assert i.to_action("GRASP", STATE, fine=True).kind == "hold" and i.to_action("GRASP", STATE, fine=True).gripper == 0
    assert i.to_action("RELEASE", STATE, fine=True).gripper == 1
    assert i.to_action("DONE", STATE, fine=True) is None

def test_base_tokens():
    a = Interpreter().to_action("BASE_LEFT", STATE, fine=True)
    assert a.kind == "base" and a.axis == "y" and a.velocity == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run (on the Mac, no numpy needed for the import? `direct.actions` imports numpy; run on the box):
```bash
ssh h200-4 'cd /home/jli/work/qwen-direct-control-dev && PYTHONPATH=.:runtime /home/jli/work/robocasa-inspect-official/.venv/bin/python -m pytest direct/tests/test_semantic.py -q 2>&1 | tail -3'
```
Expected: `ModuleNotFoundError: direct.semantic`.

- [ ] **Step 3: Implement**

`direct/semantic.py`:
```python
"""Show-Harness-style semantic action units for the RoboCasa PandaOmron.

The VLM emits ONE token per decision; this interpreter supplies the metric
magnitude and turns it into an Action for direct.executor.execute. Moves are
fixed steps in the robot BASE frame (+x forward, +y left, +z up), rotated
into the world frame with the measured base yaw. Base tokens drive the
mobile base one slot (our embodiment extension; Show-Harness arms are fixed).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .actions import Action
from .kinematics import matrix_to_quat_xyzw, quat_xyzw_to_matrix

TOKENS = ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
          "ROTATE_CW", "ROTATE_CCW", "GRASP", "RELEASE", "DONE",
          "BASE_FWD", "BASE_BACK", "BASE_LEFT", "BASE_RIGHT")
_MOVE = {"MV_FWD": (1, 0, 0), "MV_BACK": (-1, 0, 0), "MV_LEFT": (0, 1, 0),
         "MV_RIGHT": (0, -1, 0), "MV_UP": (0, 0, 1), "MV_DOWN": (0, 0, -1)}
_BASE = {"BASE_FWD": ("x", 1), "BASE_BACK": ("x", -1), "BASE_LEFT": ("y", 1), "BASE_RIGHT": ("y", -1)}


def parse_token(text: str) -> str:
    token = (text or "").strip().split()[0].strip(".,;:\"'`") if (text or "").strip() else ""
    if token not in TOKENS:
        raise ValueError(f"not an action token: {text!r}")
    return token


class Interpreter:
    def __init__(self, step_fine_m: float = 0.02, step_coarse_m: float = 0.04,
                 yaw_step_rad: float = math.radians(15), base_step_v: float = 0.5) -> None:
        self.step_fine_m, self.step_coarse_m = step_fine_m, step_coarse_m
        self.yaw_step_rad, self.base_step_v = yaw_step_rad, base_step_v

    def to_action(self, token: str, state: Mapping[str, object], *, fine: bool) -> Action | None:
        tcp = tuple(float(v) for v in state["tcp_world_position_m"])
        quat = tuple(float(v) for v in state["tcp_world_quat_xyzw"])
        if token == "DONE":
            return None
        if token in _MOVE:
            step = self.step_fine_m if fine else self.step_coarse_m
            yaw = float(state["base_world_yaw_rad"])
            dx, dy, dz = _MOVE[token]
            world = (tcp[0] + step * (dx * math.cos(yaw) - dy * math.sin(yaw)),
                     tcp[1] + step * (dx * math.sin(yaw) + dy * math.cos(yaw)),
                     tcp[2] + step * dz)
            return Action("ee", position_m=world, quat_xyzw=quat, note=token)
        if token in ("ROTATE_CW", "ROTATE_CCW"):
            sign = -1.0 if token == "ROTATE_CW" else 1.0
            rot = quat_xyzw_to_matrix(quat) @ Rotation.from_euler("z", sign * self.yaw_step_rad).as_matrix()
            return Action("ee", position_m=tcp, quat_xyzw=tuple(matrix_to_quat_xyzw(rot)), note=token)
        if token == "GRASP":
            return Action("hold", gripper=0, note=token)
        if token == "RELEASE":
            return Action("hold", gripper=1, note=token)
        axis, sign = _BASE[token]
        return Action("base", axis=axis, velocity=sign * self.base_step_v, note=token)


def proprio_text(observation: Mapping[str, object], receipt: Mapping[str, object] | None) -> dict:
    """The proprioception plugin's text: gripper height, gripper state, contact, last step effect."""
    s = observation["public_state"]
    width = float(s["gripper_width_m"])
    command = int(round(float(observation["gripper_command"])))
    gripper = ("closed on nothing" if command == 0 and width < 0.005 else
               f"closed, holding something (gap {width:.3f} m)" if command == 0 else "open")
    out = {"gripper_height_m": round(float(s["tcp_world_position_m"][2]), 3), "gripper": gripper,
           "contact_force_n": s.get("contact_force_delta_n"),
           "base_heading_rad": round(float(s["base_world_yaw_rad"]), 2)}
    if receipt:
        out["last_action_effect"] = {"status": receipt.get("status"), "tcp_moved_m": receipt.get("tcp_moved_m"),
                                     "base_moved_m": receipt.get("base_moved_m"), "reason": receipt.get("reason")}
    return out
```

- [ ] **Step 4: Run tests**

Same command as Step 2. Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
cd "/Users/jiachen/Desktop/first try/qwen-direct-control" && git add direct/semantic.py direct/tests/test_semantic.py && git commit -m "Semantic action units and interpreter for the Show-Harness-style family

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Guided single-token policy call

**Files:**
- Create: `direct/semantic_policy.py`, `direct/prompts/semantic_controller.txt`
- Test: on-box smoke in Step 4 (network call; no unit test)

**Interfaces:**
- Consumes: `direct.policy.QwenDirectClient` (its `http`, `model`, `attestation`, `_log`, counters).
- Produces: `SemanticPolicy(client).choose(system_prompt, user_text, images, decision, category="control") -> dict` with keys `token`, `raw`, `latency_s`, `usage`; raises `MalformedOutput` when the guided completion is not a token.

- [ ] **Step 1: Write the controller prompt** (`direct/prompts/semantic_controller.txt`; mirrors the Show-Harness lite prompt plus our embodiment notes)

```
You are controlling a Franka Panda arm with a parallel-jaw gripper on a mobile base in a simulated kitchen. You see two camera images: Image A is the fixed agent view (scene overview), Image B is the wrist camera (close-up from the gripper; the fingers are at the bottom of the image).
Output exactly one action token and nothing else:
MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN  (move the gripper one step: {step_note})
ROTATE_CW, ROTATE_CCW  (turn the gripper 15 degrees about its own axis to align the fingers with an object)
GRASP  (close the gripper), RELEASE  (open the gripper), DONE  (task complete)
BASE_FWD, BASE_BACK, BASE_LEFT, BASE_RIGHT  (drive the mobile base about 0.16 m; use these only when the target is far from the arm)
Directions are relative to the robot's heading: FWD is away from the robot body, LEFT is the robot's left, UP is vertical.
Use Image A to locate the target when it is not visible in Image B; use Image B for fine alignment when the target is visible close-up. Descend with MV_DOWN until the object is between the fingers in Image B, then GRASP. After GRASP check the gripper state: "closed on nothing" means the grasp missed; RELEASE, move up and re-align. RELEASE only when the object is above its destination. Mark DONE only when the object rests at its destination and the gripper has moved up and away from it.
Avoid repeating a direction that recent moves show did not change the view.
Task: {task}
Subtask: {subtask}
Proprioception: {proprio}
Recent moves: {recent_moves}
Return the single token only.
```

- [ ] **Step 2: Implement the policy**

`direct/semantic_policy.py`:
```python
"""Single-token guided-choice control call (Show-Harness style) over our Qwen client."""
from __future__ import annotations

import base64, hashlib, json, time
from collections.abc import Sequence

from .policy import DECODING, MalformedOutput, QwenDirectClient
from .semantic import TOKENS, parse_token


class SemanticPolicy:
    def __init__(self, client: QwenDirectClient) -> None:
        self.client = client

    def choose(self, *, system_prompt: str, user_text: str, images: Sequence[tuple[str, bytes]],
               decision: int, category: str = "control", choices: Sequence[str] = TOKENS) -> dict:
        content = [{"type": "text", "text": user_text}]
        for label, data in images:
            content.append({"type": "text", "text": f"[{label}]"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode(), "detail": "high"}})
        payload = {"model": self.client.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
                   **DECODING, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False},
                   "guided_choice": list(choices)}
        started = time.monotonic()
        response = self.client.http.post("chat/completions", json=payload)
        latency = time.monotonic() - started
        self.client.calls += 1; self.client.latency_s += latency
        record = {"decision": decision, "category": category, "latency_s": round(latency, 2), "http_status": response.status_code,
                  "user_text": user_text, "image_labels": [l for l, _ in images],
                  "image_sha256": [hashlib.sha256(d).hexdigest()[:16] for _, d in images]}
        if response.is_error:
            record["error"] = response.text[:500]; self.client._log(record)
            from .policy import InfrastructureError
            raise InfrastructureError(f"Qwen HTTP {response.status_code}")
        body = response.json(); raw = body["choices"][0]["message"]["content"] or ""
        usage = body.get("usage", {})
        self.client.prompt_tokens += int(usage.get("prompt_tokens", 0)); self.client.completion_tokens += int(usage.get("completion_tokens", 0))
        record.update({"raw": raw, "usage": usage})
        try:
            token = parse_token(raw)
        except ValueError:
            record["parse_error"] = True; self.client._log(record)
            raise MalformedOutput(f"not a token: {raw!r}", record)
        record["token"] = token; self.client._log(record)
        return {"token": token, "raw": raw, "latency_s": latency, "usage": usage}
```

- [ ] **Step 3: Prompt-building helper and image pair**

Append to `direct/semantic_policy.py`:
```python
def build_controller_prompt(template: str, *, task: str, subtask: str, proprio: dict, recent_moves: list[str], fine: bool) -> str:
    return template.format(task=task, subtask=subtask or "(none)", proprio=json.dumps(proprio, separators=(",", ":")),
                           recent_moves=" ".join(recent_moves[-8:]) or "(none)",
                           step_note="2 cm per step (target visible in wrist view)" if fine else "4 cm per step (target not yet in wrist view)")


def image_pair(images: dict[str, bytes], agent_view: str) -> list[tuple[str, bytes]]:
    return [("Image A: agent view", images[agent_view]), ("Image B: wrist view", images["wrist"])]
```

- [ ] **Step 4: On-box smoke of a single guided call**

```bash
ssh h200-4 'cd /home/jli/work/qwen-direct-control-dev && PYTHONPATH=.:runtime /home/jli/work/robocasa-inspect-official/.venv/bin/python - <<EOF
from pathlib import Path
from direct.policy import QwenDirectClient
from direct.semantic_policy import SemanticPolicy, build_controller_prompt, image_pair
run = Path("/home/jli/state/qwen-direct/matrix/phaseE-test/PickPlaceCounterToSink-s100-ee-short-clean/sim/frames/000000")
images = {l: (run / f"{l}.png").read_bytes() for l in ("left", "right", "wrist")}
c = QwenDirectClient(log_path=Path("/tmp/sem-smoke.jsonl"))
tpl = Path("direct/prompts/semantic_controller.txt").read_text()
text = build_controller_prompt(tpl, task="Pick the cup from the counter and place it in the sink.", subtask="approach the cup", proprio={"gripper": "open", "gripper_height_m": 1.55}, recent_moves=[], fine=False)
print(SemanticPolicy(c).choose(system_prompt="You are a robot controller.", user_text=text, images=image_pair(images, "left"), decision=1))
EOF'
```
Expected: a dict with `token` in the vocabulary, latency about 1-2 s, `completion_tokens` <= 6. If vLLM rejects `guided_choice`, use `response_format` with a JSON schema `{"type":"object","properties":{"token":{"enum":[...]}},"required":["token"]}` and parse `token`; record which path is used.

- [ ] **Step 5: Commit**

```bash
cd "/Users/jiachen/Desktop/first try/qwen-direct-control" && git add direct/semantic_policy.py direct/prompts/semantic_controller.txt && git commit -m "Guided single-token semantic policy call

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Plugins: recent moves, subtask planner with completion checks, grasp recovery

**Files:**
- Create: `direct/semantic_plugins.py`, `direct/prompts/semantic_planner.txt`
- Test: `direct/tests/test_semantic.py` (extend)

**Interfaces:**
- Produces: `RecentMoves(n=8)` with `.push(token, status)` / `.text() -> list[str]`; `SubgoalPlanner(client)` with `.plan(task, images) -> list[dict(name, done_when)]` (one counted JSON call) and `.current() -> dict`, `.advance()`; `Recovery` with `.check(token, receipt_summary, proprio) -> str | None` returning a forced token (`RELEASE`) and a flag to roll the planner back to the grasp subtask.
- Consumes: `QwenDirectClient.complete(...)` for the planner (JSON schema), `Interpreter`.

- [ ] **Step 1: Planner prompt** (`direct/prompts/semantic_planner.txt`)

```
You are planning for a robot arm with a parallel-jaw gripper. Given the task and the agent-view image, write an ordered list of 3 to 6 subtasks that a step-by-step controller (moves of 2-4 cm, grasp, release) can execute. Each subtask has a short name and a "done_when" completion criterion that can be checked from the camera images and gripper state (e.g. "object visible between the fingers in the wrist view", "gripper closed and holding something", "object above the sink basin", "gripper open and raised above the destination"). Use "phase": "approach" for coarse travel and "align" for fine positioning, grasping and placing. Do not include robot coordinates.
Task: {task}
```
Schema: `{"subtasks": [{"name": str, "done_when": str, "phase": "approach"|"align"}]}` (minItems 3, maxItems 6).

- [ ] **Step 2: Failing tests** (append to `direct/tests/test_semantic.py`)

```python
from direct.semantic_plugins import RecentMoves, Recovery

def test_recent_moves_window_and_annotation():
    r = RecentMoves(n=3)
    for t, s in (("MV_FWD", "completed"), ("MV_FWD", "blocked"), ("GRASP", "completed"), ("MV_UP", "completed")):
        r.push(t, s)
    assert r.text() == ["MV_FWD(blocked)", "GRASP", "MV_UP"]

def test_recovery_forces_release_after_empty_close():
    rec = Recovery()
    forced, rollback = rec.check("GRASP", {"status": "completed", "gripper_width_after_m": 0.002}, {"gripper": "closed on nothing"})
    assert forced == "RELEASE" and rollback is True
    forced, rollback = rec.check("GRASP", {"status": "completed", "gripper_width_after_m": 0.05}, {"gripper": "closed, holding something (gap 0.050 m)"})
    assert forced is None and rollback is False
```

- [ ] **Step 3: Run to verify failure**

Expected: `ModuleNotFoundError: direct.semantic_plugins`.

- [ ] **Step 4: Implement**

`direct/semantic_plugins.py`:
```python
"""Show-Harness-style plugins: action history, subtask planning with completion checks, grasp recovery."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .policy import MAX_TOKENS_SHORT, QwenDirectClient

PROMPTS = Path(__file__).resolve().with_name("prompts")
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["subtasks"], "properties": {"subtasks": {
    "type": "array", "minItems": 3, "maxItems": 6, "items": {"type": "object", "additionalProperties": False,
    "required": ["name", "done_when", "phase"], "properties": {"name": {"type": "string"}, "done_when": {"type": "string"},
    "phase": {"type": "string", "enum": ["approach", "align"]}}}}}}
CHECK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["subtask_done", "token"],
                "properties": {"subtask_done": {"type": "boolean"}, "token": {"type": "string"}}}


class RecentMoves:
    def __init__(self, n: int = 8) -> None:
        self.n, self.items = n, []

    def push(self, token: str, status: str) -> None:
        self.items.append(token if status in ("completed", "partial") else f"{token}({status})")
        self.items = self.items[-self.n:]

    def text(self) -> list[str]:
        return list(self.items)


class SubgoalPlanner:
    def __init__(self, client: QwenDirectClient, run: Path) -> None:
        self.client, self.run, self.subtasks, self.index = client, run, [], 0

    def plan(self, task: str, images: Sequence[tuple[str, bytes]], system_prompt: str) -> list[dict]:
        text = (PROMPTS / "semantic_planner.txt").read_text().format(task=task)
        call = self.client.complete(system_prompt=system_prompt, user_text=text, images=list(images), response_schema=PLAN_SCHEMA,
                                    max_tokens=MAX_TOKENS_SHORT, category="plan", decision=0)
        self.subtasks = call["parsed"]["subtasks"]; self.index = 0
        (self.run / "subtasks.json").write_text(json.dumps(self.subtasks, indent=1))
        return self.subtasks

    def current(self) -> dict | None:
        return self.subtasks[self.index] if self.index < len(self.subtasks) else None

    def advance(self) -> None:
        self.index = min(self.index + 1, len(self.subtasks))

    def rollback_to_grasp(self) -> None:
        for i, s in enumerate(self.subtasks):
            if "grasp" in s["name"].lower() or "align" in s["name"].lower():
                self.index = i; return
        self.index = 0

    def fine(self) -> bool:
        cur = self.current()
        return bool(cur and cur["phase"] == "align")


class Recovery:
    """Empty close -> force RELEASE and roll the plan back to the grasp subtask (Show-Harness `recovery`)."""
    def __init__(self) -> None:
        self.events = 0

    def check(self, token: str, receipt: Mapping[str, object], proprio: Mapping[str, object]) -> tuple[str | None, bool]:
        if token == "GRASP" and (receipt.get("gripper_width_after_m") or 1.0) < 0.005:
            self.events += 1
            return "RELEASE", True
        return None, False
```
The completion check runs inside the control call: when the planner is active, the control call uses `CHECK_SCHEMA` (`{"subtask_done": bool, "token": <one of TOKENS>}`) instead of `guided_choice`, so one call both checks the criterion and picks the token (this is how Show-Harness's `subgoal` plugin keeps the call count at one per step). Add to `SemanticPolicy.choose` an optional `schema` argument: when given, post `response_format=json_schema` (as in `QwenDirectClient.complete`) with `max_tokens=64`, parse `token` with `parse_token`, and return `subtask_done` too.

- [ ] **Step 5: Run tests** (expected `9 passed`) and commit

```bash
cd "/Users/jiachen/Desktop/first try/qwen-direct-control" && git add direct/semantic_plugins.py direct/prompts/semantic_planner.txt direct/semantic_policy.py direct/tests/test_semantic.py && git commit -m "Semantic plugins: recent moves, subtask planner, grasp recovery

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Ready pose (remove the singular home posture, task-agnostic)

**Files:**
- Modify: `direct/executor.py` (add `READY_Q` and `move_to_ready(sim)`), `direct/tests/smoke_child.py` (add a ready-pose check)

**Interfaces:**
- Produces: `move_to_ready(sim: Simulator) -> Receipt` executing one joint slot to `READY_Q = (0.0, -0.35, 0.0, -2.0, 0.0, 1.65, 0.785)` (elbow bent, tool pointing down, TCP about 0.45 m ahead and 0.35 m above the base) before the first decision; counted as 20 sim steps, not as a decision. Applies to the semantic family only (flag `--ready-pose`), so numerical baselines stay as run.

- [ ] **Step 1: Add to `direct/executor.py`** (after `execute`)

```python
READY_Q = (0.0, -0.35, 0.0, -2.0, 0.0, 1.65, 0.785)


def move_to_ready(sim: Simulator) -> Receipt:
    """Task-agnostic pre-episode move out of the straight-arm home singularity."""
    return execute(sim, Action("joint", q_rad=READY_Q, gripper=1, note="ready pose"), slot_steps=SLOT_STEPS)
```
Also add `--ready-pose` to `direct/episode.py`'s parser (default off) and call `move_to_ready(self.sim)` right after `self.sim.launch()` when set; log it in `decisions.jsonl` as `{"decision": 0, "status": "ready_pose", ...}`.

- [ ] **Step 2: Verify on the box** (append to `direct/tests/smoke_child.py` before the base tests)

```python
from direct.executor import move_to_ready
r = move_to_ready(sim); s_ready = sim.observation["public_state"]
print(f"ready pose: status={r.status} err={r.child['final_joint_error_rad']:.3f} tcp_base={[round(v,3) for v in s_ready['tcp_base_position_m']]} tool_down={round(float(__import__('direct.kinematics',fromlist=['quat_xyzw_to_matrix']).quat_xyzw_to_matrix(s_ready['tcp_base_quat_xyzw'])[2][2]),3)}")
```
Run the smoke; expected: `status=completed`, joint error < 0.03, tcp_base x in 0.35-0.55, z in 0.25-0.45, tool_down about -0.99 (third column z of the rotation is -1 when pointing down). If unreached, adjust `READY_Q` and rerun; record the final values in the RUNBOOK.

- [ ] **Step 3: Commit**

```bash
cd "/Users/jiachen/Desktop/first try/qwen-direct-control" && git add direct/executor.py direct/episode.py direct/tests/smoke_child.py && git commit -m "Ready pose before semantic episodes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Semantic episode loop and matrix routing

**Files:**
- Create: `direct/semantic_episode.py`
- Modify: `direct/matrix.py:19-30` (route `sem*` methods), `RUNBOOK.md`

**Interfaces:**
- Consumes: `Simulator`, `execute`, `move_to_ready`, `render_video`, `git_revision` from `direct.episode`; `Interpreter`, `proprio_text`; `SemanticPolicy`, `build_controller_prompt`, `image_pair`; `RecentMoves`, `SubgoalPlanner`, `Recovery`.
- Produces: CLI `python -m direct.semantic_episode --task T --seed S --method sem|sem+plan|sem+plan+rec|sem-full --out DIR [--agent-view left|right] [--steps-budget 900 --wall-budget-s 1200 --max-decisions 180]`; writes the same `result.json` fields as `direct.episode` plus `semantic: {tokens: Counter, plan_calls, recoveries, subtasks_completed}`.

Method semantics:
- `sem`: tokens + interpreter + proprioception + recent moves; coarse step until the gripper is below 1.15 m world height, fine below (a simple stand-in for "target visible in wrist view"); no planner, no recovery.
- `sem+plan`: adds `SubgoalPlanner` (one plan call at start; completion check in each control call; `fine()` from the subtask phase).
- `sem+plan+rec`: adds `Recovery`.
- `sem-full`: `sem+plan+rec` with the agent view chosen per step by a `view_select` rule: use `right` when the TCP's left-view pixel is not visible, else `left` (our stand-in for multi-view guidance; both external views are also always mentioned in proprioception as "target side").

- [ ] **Step 1: Write the loop** (`direct/semantic_episode.py`; structure mirrors `direct/episode.py` and reuses its helpers)

```python
"""Show-Harness-style episode: one token per decision, interpreter supplies magnitudes."""
from __future__ import annotations

import argparse, collections, json, sys, time, traceback
from pathlib import Path

from .actions import SLOT_STEPS
from .episode import DEFAULT_SCENES, MAX_CONSECUTIVE_NO_MOTION, git_revision, render_video
from .executor import Receipt, Simulator, execute, move_to_ready
from .observation import read_images
from .perception import SamClient  # unused by the policy; kept out of the prompt entirely
from .policy import InfrastructureError, MalformedOutput, QwenDirectClient
from .semantic import Interpreter, proprio_text
from .semantic_plugins import CHECK_SCHEMA, RecentMoves, Recovery, SubgoalPlanner
from .semantic_policy import SemanticPolicy, build_controller_prompt, image_pair

PROMPTS = Path(__file__).resolve().with_name("prompts")
MOVE_SLOT_STEPS = 6          # 2-4 cm at 1 cm/step plus settle; gripper/base keep the full 20-step slot


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", default="sem-full", choices=["sem", "sem+plan", "sem+plan+rec", "sem-full"])
    p.add_argument("--out", required=True); p.add_argument("--scenes", default=str(DEFAULT_SCENES))
    p.add_argument("--agent-view", default="left", choices=["left", "right"])
    p.add_argument("--steps-budget", type=int, default=900); p.add_argument("--wall-budget-s", type=float, default=1200.0)
    p.add_argument("--max-decisions", type=int, default=180)
    a = p.parse_args(argv)
    run = Path(a.out).resolve(); run.mkdir(parents=True, exist_ok=False)
    use_plan = "plan" in a.method or a.method == "sem-full"; use_rec = "rec" in a.method or a.method == "sem-full"; view_select = a.method == "sem-full"
    system_prompt = "You are a careful robot controller. Follow the interface exactly."
    template = (PROMPTS / "semantic_controller.txt").read_text()
    config = {"task": a.task, "seed": a.seed, "interface": "semantic", "mode": "short", "method": a.method, "steps_budget": a.steps_budget,
              "wall_budget_s": a.wall_budget_s, "max_decisions": a.max_decisions, "move_slot_steps": MOVE_SLOT_STEPS,
              "code_revision": git_revision(), "ready_pose": True, "agent_view": a.agent_view}
    (run / "config.json").write_text(json.dumps(config, indent=1)); (run / "system-prompt.txt").write_text(system_prompt + "\n\n" + template)
    decisions, termination, error_text, outcome = [], "incomplete", None, {}
    tokens = collections.Counter(); previous_receipt = None; started = time.monotonic()
    interp, recent, recovery = Interpreter(), RecentMoves(), Recovery() if use_rec else None
    try:
        client = QwenDirectClient(log_path=run / "qwen-calls.jsonl"); policy = SemanticPolicy(client)
        sim = Simulator(task=a.task, seed=a.seed, run=run, scenes=Path(a.scenes), action_budget=a.steps_budget, wall_budget_s=a.wall_budget_s)
        sim.launch(); started = time.monotonic()
        ready = move_to_ready(sim); decisions.append({"decision": 0, "status": "ready_pose", "receipt": ready.summary()})
        planner = SubgoalPlanner(client, run) if use_plan else None
        images = read_images(sim.observation)
        if planner:
            planner.plan(sim.observation["instruction"], image_pair(images, a.agent_view), system_prompt)
        decision, no_motion = 0, 0; forced = None
        while True:
            if decision >= a.max_decisions: termination = "decision_budget"; break
            if sim.steps_left() <= 0: termination = "step_budget"; break
            if time.monotonic() - started > a.wall_budget_s: termination = "wall_budget"; break
            decision += 1
            obs = sim.observation; state = obs["public_state"]; images = read_images(obs)
            agent_view = a.agent_view
            if view_select and not state["tcp_pixels"]["left"]["visible"]: agent_view = "right"
            proprio = proprio_text(obs, previous_receipt)
            fine = planner.fine() if planner else state["tcp_world_position_m"][2] < 1.15
            subtask = planner.current() if planner else None
            text = build_controller_prompt(template, task=obs["instruction"], subtask=(f"{subtask['name']} (done when: {subtask['done_when']})" if subtask else ""),
                                           proprio=proprio, recent_moves=recent.text(), fine=fine)
            record = {"decision": decision, "steps_used": obs["steps_used"], "subtask": subtask["name"] if subtask else None, "fine": fine, "agent_view": agent_view}
            if forced:
                token, record["forced"] = forced, True; forced = None
            else:
                try:
                    if planner:
                        out = policy.choose(system_prompt=system_prompt, user_text=text + "\nAlso report whether the current subtask's done_when criterion is satisfied.",
                                            images=image_pair(images, agent_view), decision=decision, schema=CHECK_SCHEMA)
                        if out.get("subtask_done"): planner.advance(); record["subtask_done"] = True
                    else:
                        out = policy.choose(system_prompt=system_prompt, user_text=text, images=image_pair(images, agent_view), decision=decision)
                    token = out["token"]
                except MalformedOutput as e:
                    record.update({"status": "malformed_output", "error": str(e)}); decisions.append(record); no_motion += 1
                    if no_motion >= MAX_CONSECUTIVE_NO_MOTION: termination = "no_progress"; break
                    continue
            tokens[token] += 1; record["token"] = token
            action = interp.to_action(token, state, fine=fine)
            if action is None:
                record["status"] = "done"; decisions.append(record); termination = "stop"; break
            steps = MOVE_SLOT_STEPS if action.kind == "ee" else SLOT_STEPS
            receipt = execute(sim, action, slot_steps=steps)
            record.update({"status": receipt.status, "steps": receipt.steps, "receipt": receipt.summary()}); decisions.append(record)
            recent.push(token, receipt.status); previous_receipt = receipt.summary()
            no_motion = 0 if receipt.steps > 0 else no_motion + 1
            if recovery and receipt.child is not None:
                forced, rollback = recovery.check(token, receipt.child, proprio_text(sim.observation, None))
                if rollback and planner: planner.rollback_to_grasp()
            if no_motion >= MAX_CONSECUTIVE_NO_MOTION: termination = "no_progress"; break
        outcome = sim.finish()
    except Exception:
        error_text = traceback.format_exc(); termination = "infrastructure_error"
        try: outcome = sim.close()
        except Exception: pass
    finally:
        try: client.close()
        except Exception: pass
    (run / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n")
    frames = 0
    try: frames = render_video(run / "sim", run / "episode.mp4")
    except Exception as e: error_text = (error_text or "") + f"\nvideo: {e!r}"
    result = {**config, "official_success": outcome.get("official_success"), "termination": termination, "simulator_steps": outcome.get("simulator_steps"),
              "decisions": len([d for d in decisions if d["decision"] > 0]), "slots_executed": sum(1 for d in decisions if d.get("steps", 0) > 0),
              "rejected_actions": sum(1 for d in decisions if d.get("steps") == 0 and d.get("status") not in ("done", "ready_pose")),
              "wall_s": round(time.monotonic() - started, 1), **(client.totals() if "client" in dir() else {}),
              "video": str(run / "episode.mp4") if frames else None, "video_frames": frames, "simulator_terminal": outcome,
              "semantic": {"tokens": dict(tokens), "recoveries": recovery.events if recovery else 0,
                           "subtasks": planner.subtasks if planner else None, "subtask_index": planner.index if planner else None}, "error": error_text}
    (run / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("task", "seed", "method", "official_success", "termination", "simulator_steps", "decisions", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Route the matrix** (`direct/matrix.py`, in `run_one`, replace the `command = [...]` construction)

```python
    module = "direct.semantic_episode" if spec["method"].startswith("sem") else "direct.episode"
    command = [PYTHON, "-m", module, "--task", spec["task"], "--seed", str(spec["seed"]), "--method", spec["method"], "--out", str(run)]
    if module == "direct.episode":
        command += ["--interface", spec["interface"], "--mode", spec["mode"], "--method-config", json.dumps(spec.get("method_config", {}))]
```
Keep the budget flags loop unchanged. `direct.analyze` and `direct.milestones` work unchanged because `result.json` keeps the same keys and `sim/snapshots` are produced by the same child.

- [ ] **Step 3: Smoke one 12-decision episode per method on the dev checkout**

```bash
ssh h200-4 'cd /home/jli/work/qwen-direct-control-dev && export PYTHONPATH=/home/jli/work/qwen-direct-control-dev:/home/jli/work/qwen-direct-control-dev/runtime; PY=/home/jli/work/robocasa-inspect-official/.venv/bin/python; for m in sem sem+plan sem+plan+rec sem-full; do rm -rf "/home/jli/state/qwen-direct/runs/dev-$m"; $PY -m direct.semantic_episode --task PickPlaceCounterToSink --seed 0 --method "$m" --max-decisions 12 --out "/home/jli/state/qwen-direct/runs/dev-$m" 2>&1 | grep -v WARNING | tail -1; python3 -c "
import json; run=\"/home/jli/state/qwen-direct/runs/dev-$m\"; r=json.load(open(run+\"/result.json\")); print(\"  \", r[\"termination\"], (r[\"error\"] or \"\")[-200:], r[\"semantic\"][\"tokens\"], r[\"qwen_calls\"], \"calls\")"; done'
```
Expected: each method completes 12 decisions without `infrastructure_error`; `sem+plan` writes `subtasks.json` with 3-6 entries; `sem-full` shows `agent_view` switching at least once if the TCP leaves the left view; token histogram is not a single repeated token. Interface checks: one `MV_FWD` moves the TCP +0.02/0.04 m along the base heading in `receipt.tcp_moved_m`; `ROTATE_CW` leaves `tcp_moved_m` near zero; `GRASP` changes `gripper_width_after_m`.

- [ ] **Step 4: RUNBOOK and commit**

Add a "Semantic family (Show-Harness style)" section with the CLI, the method definitions above, `MOVE_SLOT_STEPS`, step sizes, the ready pose, and the fact that SAM is not used by this family. Commit:
```bash
cd "/Users/jiachen/Desktop/first try/qwen-direct-control" && git add direct/semantic_episode.py direct/matrix.py RUNBOOK.md && git commit -m "Semantic episode loop and matrix routing

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Lane B experiments

**Files:** outputs under `/home/jli/state/qwen-direct/matrix/sem-*`; tables in `results-direct/sem/`; `REPORT.md` section 12.

- [ ] **Step 1: Freeze and screen on the 9 development starts**

```bash
ssh h200-4 'rsync -a --delete --exclude __pycache__ /home/jli/work/qwen-direct-control-dev/ /home/jli/work/qwen-direct-control/ && cd /home/jli/work/qwen-direct-control && export PYTHONPATH=/home/jli/work/qwen-direct-control:/home/jli/work/qwen-direct-control/runtime && nohup /home/jli/work/robocasa-inspect-official/.venv/bin/python -m direct.matrix --tasks PickPlaceCounterToSink PickPlaceCounterToDrawer PickPlaceStoveToCounter --seeds 0 1 2 --interfaces ee --modes short --methods sem sem+plan sem+plan+rec sem-full --out /home/jli/state/qwen-direct/matrix/sem-dev --parallel 3 > /home/jli/state/qwen-direct/matrix/sem-dev.log 2>&1 &'
```
36 episodes; each is up to 180 calls at about 1.5 s (single token) so about 5 min: about 1 h at parallel 3. Comparators: Phase C clean EE-short on the same starts (0/9, milestones: 0 approaches).

- [ ] **Step 2: Milestones**

Run the inspection as in RUNBOOK ("Milestones") over `matrix/sem-dev` and write `results-direct/sem/sem-dev-milestones.md`. Decision rule for the next step: the family advances to the test set if any method reaches >= 3/9 holds or >= 1/9 successes; otherwise stop, report, and diagnose the token histograms (which tokens dominate, whether GRASP is ever emitted with the object in the wrist view).

- [ ] **Step 3: Test set for the best method**

```bash
ssh h200-4 'cd /home/jli/work/qwen-direct-control && export PYTHONPATH=/home/jli/work/qwen-direct-control:/home/jli/work/qwen-direct-control/runtime && nohup /home/jli/work/robocasa-inspect-official/.venv/bin/python -m direct.matrix --tasks PickPlaceCounterToSink PickPlaceCounterToDrawer PickPlaceStoveToCounter --seeds 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 --interfaces ee --modes short --methods <best> --out /home/jli/state/qwen-direct/matrix/sem-test --parallel 3 > /home/jli/state/qwen-direct/matrix/sem-test.log 2>&1 &'
```
60 episodes (about 2 h). Paired comparison against Phase E clean (1/60) with `direct.analyze --matrices matrix/sem-test matrix/phaseE-test --baseline clean` and milestone paired bootstrap as in REPORT.md section 6.

- [ ] **Step 4: Report**

Add `REPORT.md` section 12 "Show-Harness-style interface with zero-shot Qwen-27B": Lane A table (their benchmark), Lane B dev screen, Lane B test paired table, token histograms, cost per episode, and the reading: whether the interface (Lane B vs our clean) or the model (Lane A vs the paper's frontier numbers) is the binding constraint. Commit.

---

## Cost estimate

| Block | Episodes | Wall (3 parallel) |
|---|---|---|
| Lane A install + smoke | - | 1-3 h (ManiSkill/Vulkan risk) |
| Lane A default + noplugins + full | 90 | about 1.5 h |
| Lane A frontier reference (optional) | 30 | 30 min + API cost |
| Lane B implementation (Tasks 3-7) | - | about 1 day |
| Lane B dev screen | 36 | about 1 h |
| Lane B test | 60 | about 2 h |

## Risks and how the plan handles them

- Their runner's config keys/flags may differ from the fetched summaries: Task 1 Steps 4-5 instruct reading `core/vlm/vlm_client.py` and `core/runners/mvtoken.py` and recording the exact working command rather than guessing.
- vLLM `guided_choice` may be unavailable on the installed version: Task 4 Step 4 gives the JSON-schema fallback.
- ManiSkill needs Vulkan on the H200 box: memory note `nebius-vulkan-icd-fix` records the working ICD fix.
- The ready pose changes the start state relative to the numerical campaign: it is applied only to the semantic family and logged as decision 0; if Lane B looks promising, rerun clean EE-short with `--ready-pose` on the 9 dev starts (9 episodes) to separate the pose effect from the interface effect.
