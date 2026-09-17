#!/usr/bin/env python3
"""Minimal coding-agent loop for a local OpenAI-compatible VLM (Qwen3.8-27B on vLLM).

Plays the role the codex / claude CLIs play in agent-as-policy: the model gets a
session directory, a shell tool and an image-viewing tool, the task prompt as the
first user message, and works until it stops calling tools. Nothing here knows
about robots: the robot is reached only through robot_client.py inside the session.

Tools
  exec        {"command": str, "timeout_s": int}   bash -lc in the session dir; stdout+stderr, truncated
  view_image  {"path": str}                         returns the PNG/JPEG as an image content part

Context management: at most MAX_IMAGES images stay in the conversation (older ones
become a one-line text placeholder); when the prompt exceeds CTX_SOFT tokens the
oldest tool exchanges after the task prompt are dropped with a note.
Events: <session>/agent_events.jsonl (one JSON line per request / tool call);
        <session>/agent_last_message.txt; <session>/agent_usage.json.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM = """You are an autonomous agent operating a robot from a Linux shell. Your working directory is the
session directory; every relative path is relative to it. You have two tools:
- exec: run a bash command (the robot is driven only through `python3 robot_client.py . <command> ['<json>']`).
- view_image: look at an image file (PNG/JPEG). You cannot see an image until you view it.
Work step by step: run a command, read its output, look at the images you captured, decide, act.
Keep your notes and scripts under scratch/. Do not ask the user questions; there is no user.
When the task is finished (or you conclude it cannot be finished), write scratch/RESULT.md as the
task prompt asks and then reply with a short final message and no tool call."""

TOOLS = [
    {"type": "function", "function": {
        "name": "exec",
        "description": "Run a bash command in the session directory and return its stdout and stderr (truncated). "
                       "Blocking robot commands take up to a minute.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "the bash command line"},
            "timeout_s": {"type": "integer", "description": "seconds before the command is killed (default 300)"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "view_image",
        "description": "Show an image file (PNG or JPEG, path relative to the session directory) so you can see it.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]

MAX_OUTPUT = 6000
MAX_IMAGES = 8
CTX_SOFT = 44000
MAX_TOKENS = 2048


def _now():
    return time.time()


_TC_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TC_FUNC = re.compile(r"<function=([\w.-]+)>\s*(.*?)\s*</function>", re.S)
_TC_PARAM = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_inline_tool_calls(text: str) -> list[dict]:
    """Qwen3.8 emits tool calls the vLLM hermes parser cannot read:
    <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
    (also accepts the JSON hermes body). Returns OpenAI-shaped tool_calls."""
    calls = []
    for i, body in enumerate(_TC_BLOCK.findall(text)):
        m = _TC_FUNC.search(body)
        if m:
            name = m.group(1)
            args = {}
            for k, v in _TC_PARAM.findall(m.group(2)):
                v = v.strip()
                if k == "timeout_s":
                    try:
                        v = int(float(v))
                    except ValueError:
                        pass
                args[k] = v
        else:
            try:
                obj = json.loads(body)
                name, args = obj["name"], obj.get("arguments") or obj.get("parameters") or {}
            except Exception:  # noqa: BLE001
                continue
        calls.append({"id": f"call_inline_{int(_now() * 1000) % 100000}_{i}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
    return calls


class Agent:
    def __init__(self, session: Path, base_url: str, token: str, model: str, *, max_turns: int, wall_s: float,
                 temperature: float = 0.2, max_images: int = MAX_IMAGES, ctx_soft: int = CTX_SOFT, python_bin: str | None = None,
                 log_dir: Path | None = None, thinking: bool = False):
        self.session = session.resolve()
        # harness files live OUTSIDE the session so the agent cannot read its own event log
        self.log_dir = (log_dir or self.session.parent).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.token, self.model = token, model
        self.max_turns, self.wall_s, self.temperature = max_turns, wall_s, temperature
        self.max_images, self.ctx_soft = max_images, ctx_soft
        self.events = open(self.log_dir / "agent_events.jsonl", "a")
        self.thinking = bool(thinking)
        self.consecutive_nudges = 0
        self.require_action = False
        self.required_cmds: set | None = None      # when set, the next exec must contain one of these robot commands
        self.required_msg = ""
        self.empty_closes = 0                      # consecutive empty closes without a fresh measurement in between
        self.failed_moves = 0                      # consecutive motion commands that returned ok:false
        self.recent_targets: list = []             # move_ee target positions of consecutive move commands
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "max_prompt_tokens": 0,
                      "tool_calls": 0, "exec_calls": 0, "view_image_calls": 0, "compactions": 0, "errors": 0}
        env = dict(os.environ)
        env["HOME"] = str(self.session)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("PYTHONPATH", None)
        if python_bin:
            bindir = self.session / ".bin"
            bindir.mkdir(exist_ok=True)
            link = bindir / "python3"
            if not link.exists():
                link.symlink_to(python_bin)
            env["PATH"] = f"{bindir}:{env.get('PATH', '/usr/bin:/bin')}"
        self.env = env
        self.messages: list[dict] = []
        self.n_images = 0
        self.recent_cmds: list[str] = []      # exec command lines, for loop detection
        self.nudges_sent = 0

    # ---------- events ----------
    def _event(self, kind, **payload):
        self.events.write(json.dumps({"t": _now(), "type": kind, **payload}, default=str) + "\n")
        self.events.flush()

    # ---------- tools ----------
    _ROBOT_CMD = re.compile(r"robot_client\.py\s+\.\s+(?:--arm\s+\w+\s+)?(\w+)")

    def _loop_note(self):
        """Two loop signatures a coding-agent harness would also flag: the identical command line issued
        three times in a row, or eight consecutive exec calls that only measure (deproject/state/status/help)
        without any motion or capture. A note alone does not break Qwen's loops (the repeated exchanges in
        context reinforce them), so the looping exchanges are also removed from the context, keeping one."""
        rc = self.recent_cmds
        note, n_loop = None, 0
        if self.failed_moves >= 2:
            self.failed_moves = 0
            self.required_cmds = {"frames", "home", "check_pose"}
            self.required_msg = ("Two motion commands in a row failed (SETTLE_MISS / IK_FAILED / CLAMP). Do not nudge the target again: "
                                 "take `frames` and look, or test candidates with the free `check_pose`, or go `home`; then choose a "
                                 "clearly different target (a different height, farther from the base, or move the base first).")
            self.consecutive_nudges += 1
            self._event("nudge", note="failed-moves rule")
            return "[harness note] " + self.required_msg
        if len(self.recent_targets) >= 6:
            pts = self.recent_targets[-6:]
            spread = max(max(abs(a[i] - b[i]) for i in range(3)) for a in pts for b in pts)
            if spread < 0.015:
                self.recent_targets = []
                self.required_cmds = {"gripper", "frames", "home"}
                self.required_msg = ("Six move_ee commands in a row targeted the same spot within 1.5 cm: millimetre adjustments do not "
                                     "change anything. Either close the gripper now, or take `frames` and re-measure, or go `home`.")
                self.consecutive_nudges += 1
                self._event("nudge", note="micro-move rule")
                return "[harness note] " + self.required_msg
        if self.empty_closes >= 3:
            self.empty_closes = 0
            self.required_cmds = {"deproject", "frames"}
            self.required_msg = ("Three grasps in a row closed on nothing at about the same spot: the object is NOT where you think. "
                                 "Take `frames`, then measure the object again with a `deproject` region call (wrist camera, "
                                 "above_z = counter height) and aim at the middle of min/max in x,y and at its centre height.")
            self.consecutive_nudges += 1
            self._event("nudge", note="empty-close rule")
            return "[harness note] " + self.required_msg
        cyc = self._cycle_length(rc)
        if cyc:
            n_loop = cyc * 2
            note = (f"[harness note] You are repeating the same {cyc}-command cycle; the repeats were removed from your context "
                    "because the outcome never changes. Do something different: re-observe (frames), re-measure the object with a "
                    "deproject region call, change the grasp height/orientation, or drive the base.")
        elif len(rc) >= 3 and rc[-1] == rc[-2] == rc[-3]:
            n_loop = 3
            while n_loop < len(rc) and rc[-n_loop - 1] == rc[-1]:
                n_loop += 1
            note = (f"[harness note] You ran the exact same command {n_loop} times in a row; the repeats were removed from your "
                    "context because their answer never changes. Decide from what you already know and take a DIFFERENT action "
                    "(a move toward the object, gripper, base, or frames).")
        elif len(rc) >= 8:
            kinds = [set(self._ROBOT_CMD.findall(c)) for c in rc]
            n_loop = 0
            for k in reversed(kinds):
                if k and k <= {"deproject", "state", "status", "help"}:
                    n_loop += 1
                else:
                    break
            if n_loop >= 8:
                note = (f"[harness note] Your last {n_loop} tool calls only measured (deproject/state) and were removed from your "
                        "context except the first. You already have the coordinates; measuring again will not change them. "
                        "Write the object position, the destination and your move sequence to scratch/plan.md, then execute "
                        "the first move now.")
            else:
                n_loop = 0
        if note:
            summary = self._drop_last_exchanges(n_loop - 1)
            if summary:
                note += "\nResults of the removed calls (for reference):\n" + summary
            self.recent_cmds = []
            self.consecutive_nudges += 1
            if self.consecutive_nudges >= 2:
                self.require_action = True
                note += ("\n[harness rule now in force] Your next tool call MUST contain a robot action: move_ee, move_delta, "
                         "move_base, home or gripper (or `frames` to look again). Other commands will not be executed until you act.")
        return note

    _POINT = re.compile(r'"point_base": (\{[^}]*\})')

    @staticmethod
    def _cycle_length(rc):
        """k if the last 2k commands are the same k-command cycle twice (k in 2..6, ignoring digits), else 0."""
        norm = [re.sub(r"[-\d.]+", "#", c) for c in rc]
        for k in range(2, 7):
            if len(norm) >= 3 * k and norm[-k:] == norm[-2 * k:-k] == norm[-3 * k:-2 * k]:
                return k
        return 0

    def _drop_last_exchanges(self, n):
        """Remove the last n assistant tool-call exchanges (assistant + tool results + image messages) and
        return a compact summary of the measurements they contained, so no number is lost."""
        removed, lines = 0, []
        while removed < n:
            idx = None
            for i in range(len(self.messages) - 1, 1, -1):
                if self.messages[i].get("role") == "assistant" and self.messages[i].get("tool_calls"):
                    idx = i
                    break
            if idx is None:
                break
            j = idx + 1
            cmd = ""
            try:
                cmd = json.loads(self.messages[idx]["tool_calls"][0]["function"]["arguments"]).get("command", "")
            except Exception:  # noqa: BLE001
                pass
            while j < len(self.messages) and (self.messages[j].get("role") == "tool" or self.messages[j].get("_image")):
                if self.messages[j].get("role") == "tool":
                    for m_ in self._POINT.finditer(str(self.messages[j].get("content", ""))):
                        arg = re.search(r"deproject\s+'([^']*)'", cmd)
                        lines.append(f"- deproject {arg.group(1) if arg else ''} -> point_base {m_.group(1)}")
                j += 1
            del self.messages[idx:j]
            removed += 1
        self._event("drop_exchanges", n=removed, kept_results=len(lines))
        lines = list(dict.fromkeys(lines))[-12:]
        return "\n".join(reversed(lines))

    _ACTION_CMDS = {"move_ee", "move_delta", "move_joints", "home", "gripper", "move_base", "frames", "move_path", "grasp_at", "place_at", "approach_base"}

    def tool_exec(self, args):
        cmd = str(args.get("command", ""))
        self.recent_cmds.append(cmd.strip())
        self.recent_cmds = self.recent_cmds[-20:]
        names = set(self._ROBOT_CMD.findall(cmd))
        acts = names & self._ACTION_CMDS
        if self.required_cmds is not None:
            if names & self.required_cmds:
                self.required_cmds = None
                self.consecutive_nudges = 0
            else:
                self._event("exec_refused", command=cmd)
                return "[harness] This command was NOT executed. " + self.required_msg
        if acts:
            self.require_action = False
            self.consecutive_nudges = 0
        elif self.require_action:
            self._event("exec_refused", command=cmd)
            return ("[harness] This command was NOT executed. After repeated loops the next tool call must contain a robot action "
                    "(a move, gripper or base command from README_interface.md) or `frames`. Use the coordinates you already have and act.")
        if "deproject" in names or "frames" in names:
            self.empty_closes = 0
        timeout = int(args.get("timeout_s") or 300)
        timeout = max(5, min(timeout, 900))
        t0 = _now()
        try:
            p = subprocess.run(["bash", "-lc", cmd], cwd=self.session, env=self.env, capture_output=True, text=True,
                               timeout=timeout, errors="replace")
            out = p.stdout
            if p.stderr:
                out += ("\n[stderr]\n" + p.stderr)
            out += f"\n[exit {p.returncode}]"
        except subprocess.TimeoutExpired as e:
            partial = e.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            out = partial + f"\n[killed after {timeout}s]"
        if len(out) > MAX_OUTPUT:
            out = out[: MAX_OUTPUT // 2] + f"\n...[{len(out) - MAX_OUTPUT} chars omitted]...\n" + out[-MAX_OUTPUT // 2:]
        self.usage["exec_calls"] += 1
        self._event("exec", command=cmd, seconds=round(_now() - t0, 1), output=out[:2000])
        if '"held": false' in out:
            self.empty_closes += 1
        elif '"held": true' in out:
            self.empty_closes = 0
        # micro-adjustment sweeps: consecutive move_ee targets within 1.5 cm of each other
        tgt = re.search(r'"position"\s*:\s*\{\s*"x"\s*:\s*([-\d.]+)\s*,\s*"y"\s*:\s*([-\d.]+)\s*,\s*"z"\s*:\s*([-\d.]+)', cmd)
        if "move_ee" in names and tgt and "gripper" not in names and "frames" not in names:
            try:
                self.recent_targets.append(tuple(float(v) for v in tgt.groups()))
            except ValueError:
                pass
            self.recent_targets = self.recent_targets[-8:]
        elif acts:
            self.recent_targets = []
        # consecutive failed motion commands (SETTLE_MISS / IK_FAILED / CLAMP sweeps)
        if acts and acts <= {"move_ee", "move_delta", "move_joints", "move_base"}:
            self.failed_moves = self.failed_moves + 1 if ('"ok": false' in out and '"ok": true' not in out) else 0
        elif acts:
            self.failed_moves = 0
        return out

    def tool_view_image(self, args):
        rel = str(args.get("path", ""))
        path = (self.session / rel).resolve()
        if not str(path).startswith(str(self.session)):
            return None, f"error: {rel} is outside the session directory"
        if not path.exists():
            return None, f"error: no such file {rel}"
        suffix = path.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg"):
            return None, f"error: {rel} is not a PNG/JPEG image"
        data = path.read_bytes()
        if len(data) > 6_000_000:
            return None, f"error: {rel} is too large ({len(data)} bytes)"
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        self.usage["view_image_calls"] += 1
        self._event("view_image", path=rel, bytes=len(data))
        return f"data:{mime};base64,{base64.b64encode(data).decode()}", f"[image {rel} is shown below]"

    # ---------- context ----------
    def _prune_images(self):
        seen = 0
        for m in reversed(self.messages):
            if m.get("role") != "user" or not isinstance(m.get("content"), list):
                continue
            for part in m["content"]:
                if part.get("type") == "image_url":
                    seen += 1
                    if seen > self.max_images:
                        part.clear()
                        part.update({"type": "text", "text": "[an earlier image, no longer in context; view it again if needed]"})

    def _compact(self):
        """Drop the oldest tool exchanges after the task prompt (index 1) until the estimate is small."""
        keep_head = 2   # system + task prompt
        # find message boundaries: assistant(tool_calls) -> tool -> [user image] groups
        while self._estimate() > self.ctx_soft * 0.6 and len(self.messages) > keep_head + 6:
            # remove one exchange: assistant + its tool results (+ image user msgs)
            i = keep_head
            if self.messages[i].get("role") == "user" and self.messages[i].get("_note"):
                i += 1
            if i >= len(self.messages):
                break
            j = i + 1
            while j < len(self.messages) and self.messages[j].get("role") in ("tool",) or (
                    j < len(self.messages) and self.messages[j].get("role") == "user" and self.messages[j].get("_image")):
                j += 1
            del self.messages[i:j]
            self.usage["compactions"] += 1
        note = {"role": "user", "_note": True,
                "content": "[context compacted: earlier tool exchanges were removed to fit the context window. "
                           "Your files under scratch/ and frames/ persist; re-read scratch/ notes if you need them.]"}
        if not (len(self.messages) > keep_head and self.messages[keep_head].get("_note")):
            self.messages.insert(keep_head, note)
        self._event("compact", messages=len(self.messages))

    def _estimate(self):
        n = 0
        for m in self.messages:
            c = m.get("content")
            if isinstance(c, str):
                n += len(c) / 3.5
            elif isinstance(c, list):
                for part in c:
                    n += len(part.get("text", "")) / 3.5 if part.get("type") == "text" else 400
            for tc in m.get("tool_calls") or []:
                n += len(json.dumps(tc)) / 3.5
        return n

    def _wire(self):
        out = []
        for m in self.messages:
            mm = {k: v for k, v in m.items() if not k.startswith("_")}
            out.append(mm)
        return out

    # ---------- model ----------
    def _request(self):
        body = {"model": self.model, "messages": self._wire(), "tools": TOOLS, "tool_choice": "auto",
                "max_tokens": (MAX_TOKENS * 3 if self.thinking else MAX_TOKENS), "temperature": self.temperature, "top_p": 0.95,
                "presence_penalty": 0.3, "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=data,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        t0 = _now()
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:800]
                last = f"HTTP {e.code}: {text}"
                self.usage["errors"] += 1
                self._event("http_error", attempt=attempt, error=last)
                if e.code == 400 and ("context" in text.lower() or "maximum" in text.lower() or "too long" in text.lower()):
                    self._compact()
                    body["messages"] = self._wire()
                    req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(), headers=req.headers)
                    continue
                time.sleep(5 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                last = repr(e)
                self.usage["errors"] += 1
                self._event("request_error", attempt=attempt, error=last)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"model request failed: {last}")
        dt = _now() - t0
        u = resp.get("usage") or {}
        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        self.usage["completion_tokens"] += int(u.get("completion_tokens", 0))
        self.usage["max_prompt_tokens"] = max(self.usage["max_prompt_tokens"], int(u.get("prompt_tokens", 0)))
        self.usage["latency_s"] += dt
        choice = resp["choices"][0]
        msg = choice["message"]
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if msg.get("content") and "</think>" in msg["content"]:
            # no reasoning parser on the server: the template opens <think>, the model closes it
            head, _, tail = msg["content"].partition("</think>")
            reasoning = (reasoning + head.replace("<think>", "")).strip()
            msg["content"] = tail.strip()
        if reasoning:
            self._event("reasoning", text=reasoning[:1500], chars=len(reasoning))
        if not msg.get("tool_calls") and msg.get("content") and "<tool_call>" in msg["content"]:
            parsed = parse_inline_tool_calls(msg["content"])
            if parsed:
                msg["tool_calls"] = parsed
                msg["content"] = msg["content"].split("<tool_call>", 1)[0].strip()
        self._event("response", seconds=round(dt, 1), prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                    finish_reason=choice.get("finish_reason"), content=(msg.get("content") or "")[:3000],
                    tool_calls=[{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"][:1500]} for tc in (msg.get("tool_calls") or [])])
        return msg, int(u.get("prompt_tokens", 0))

    # ---------- main loop ----------
    def run(self, prompt: str):
        self.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        t_start = _now()
        nudges = 0
        outcome = "max_turns"
        for turn in range(self.max_turns):
            if _now() - t_start > self.wall_s:
                outcome = "wall_clock"
                break
            msg, ptok = self._request()
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            calls = msg.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = [{"id": tc.get("id") or f"call_{turn}_{i}", "type": "function",
                                            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                                           for i, tc in enumerate(calls)]
            self.messages.append(assistant)
            if not calls:
                (self.log_dir / "agent_last_message.txt").write_text(assistant["content"])
                if (self.session / "scratch" / "RESULT.md").exists() or nudges >= 2:
                    outcome = "finished" if (self.session / "scratch" / "RESULT.md").exists() else "stopped_without_result"
                    break
                nudges += 1
                self.messages.append({"role": "user", "content": "Continue working on the task. When it is finished (or impossible), "
                                                                 "write scratch/RESULT.md as instructed, then send a final message."})
                continue
            for tc in assistant["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc["function"]["arguments"]}
                self.usage["tool_calls"] += 1
                if name == "exec":
                    text = self.tool_exec(args) if "_raw" not in args else "error: arguments were not valid JSON"
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
                elif name == "view_image":
                    url, text = self.tool_view_image(args)
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
                    if url:
                        self.messages.append({"role": "user", "_image": True, "content": [
                            {"type": "text", "text": f"[image: {args.get('path')}]"},
                            {"type": "image_url", "image_url": {"url": url}}]})
                else:
                    self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": f"error: unknown tool {name}"})
            self._prune_images()
            note = self._loop_note()
            if note:
                self.messages.append({"role": "user", "content": note})
                self.nudges_sent += 1
                self._event("nudge", note=note)
            if ptok > self.ctx_soft:
                self._compact()
        self.usage["wall_s"] = round(_now() - t_start, 1)
        self.usage["turns"] = self.usage["requests"]
        self.usage["outcome"] = outcome
        (self.log_dir / "agent_usage.json").write_text(json.dumps(self.usage, indent=1))
        stripped = []
        for m in self.messages:
            mm = {k: v for k, v in m.items() if not k.startswith("_")}
            if isinstance(mm.get("content"), list):
                mm["content"] = [p if p.get("type") == "text" else {"type": "image_url", "image_url": {"url": "<stripped>"}} for p in mm["content"]]
            stripped.append(mm)
        (self.log_dir / "agent_messages.json").write_text(json.dumps(stripped, indent=1))
        self._event("done", outcome=outcome, **{k: v for k, v in self.usage.items() if k != "outcome"})
        return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--prompt", default="PROMPT.md")
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--token-file", default="/home/jli/state/panda-qwen38/api-token")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=160)
    ap.add_argument("--wall-min", type=float, default=45)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-images", type=int, default=MAX_IMAGES)
    ap.add_argument("--python-bin", default=None, help="interpreter exposed as python3 inside the session (numpy/PIL/scipy/cv2)")
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking (reasoning is logged, not shown to the tools)")
    a = ap.parse_args()
    session = Path(a.session)
    token = Path(a.token_file).read_text().strip()
    model = a.model
    if not model:
        req = urllib.request.Request(a.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            model = json.loads(r.read())["data"][0]["id"]
    agent = Agent(session, a.base_url, token, model, max_turns=a.max_turns, wall_s=a.wall_min * 60,
                  temperature=a.temperature, max_images=a.max_images, python_bin=a.python_bin, thinking=a.thinking)
    prompt = (session / a.prompt).read_text()
    outcome = agent.run(prompt)
    print(json.dumps({"outcome": outcome, **agent.usage}))


if __name__ == "__main__":
    main()
