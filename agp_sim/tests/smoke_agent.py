#!/usr/bin/env python3
"""Agent loop against the live vLLM without a robot: exec + view_image tool calls must round-trip."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_loop import Agent
import numpy as np
from PIL import Image

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "scratch").mkdir(exist_ok=True)
img = np.zeros((256, 256, 3), np.uint8); img[60:200, 40:120] = (220, 30, 30); img[20:80, 160:240] = (30, 200, 60)
Image.fromarray(img).save(out / "scene.png")
token = Path("/home/jli/state/panda-qwen38/api-token").read_text().strip()
import urllib.request
req = urllib.request.Request("http://127.0.0.1:8002/v1/models", headers={"Authorization": f"Bearer {token}"})
model = json.loads(urllib.request.urlopen(req, timeout=30).read())["data"][0]["id"]
agent = Agent(out, "http://127.0.0.1:8002/v1", token, model, max_turns=12, wall_s=600, python_bin="/home/jli/work/robocasa-inspect-official/.venv/bin/python")
prompt = """Smoke test. Steps: (1) run `ls` with exec; (2) view the image scene.png and tell me the colours and rough positions of the
two rectangles; (3) with exec, run a python3 one-liner that imports numpy and PIL, loads scene.png and prints the mean colour of the
left rectangle region rows 60:200 cols 40:120; (4) write scratch/RESULT.md containing one line 'smoke ok' plus the colours you saw;
(5) reply with a final message."""
t = time.time()
outcome = agent.run(prompt)
print("outcome", outcome, f"{time.time()-t:.0f}s", json.dumps(agent.usage))
print((out / "scratch" / "RESULT.md").read_text() if (out / "scratch" / "RESULT.md").exists() else "NO RESULT.md")
