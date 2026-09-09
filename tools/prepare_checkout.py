"""Resolve archived recipe paths for a checkout on the original Linux environment.

Controller snapshots and archived evidence remain untouched. The simulator,
SAM2, Qwen service, and their original host paths must already be available.
"""
import argparse
import json
from pathlib import Path


def prepare(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    restored = output / "restored-memory.json"
    restored.write_bytes((root / "memory/restored-memory.json").read_bytes())

    manifests = {}
    for name in ("fullten-r1.json", "validated-routes.json"):
        manifest = json.loads((root / "workflow" / name).read_text())
        manifest["memory"] = str(restored)
        for route in manifest["routes"]:
            code = root / "controllers/cohort" / route["task"]
            if name == "validated-routes.json" and route["task"] == "PickPlaceSinkToCounter":
                code = root / "controllers/reviewed/SinkToCounter"
            route["code"] = str(code)
            if "memory" in route:
                route["memory"] = str(restored)
        (output / name).write_text(json.dumps(manifest, indent=2) + "\n")
        manifests[name] = manifest

    bank = json.loads((root / "memory/learned-memory-final.json").read_text())
    registered = {r["task"]: r for r in manifests["validated-routes.json"]["routes"]}
    for card in bank["skills"]:
        if card.get("execution_recipe"):
            route = registered[card["source_task"]]
            card["execution_recipe"] = {
                key: route[key] for key in ("code", "entry", "args", "memory") if key in route
            }
    (output / "learned-memory-final.json").write_text(json.dumps(bank, indent=2) + "\n")

    for source in (root / "workflow").glob("*.py"):
        text = source.read_text()
        if source.name == "run_workflow.py":
            text = text.replace(
                "/home/jli/work/recovery-validated-20260908/garlic-reviewed",
                str(root / "controllers/reviewed/SinkToCounter"),
            )
        (output / source.name).write_text(text)
    print(f"Prepared workflow: {output}")
    print("Requires the existing RoboCasa, SAM2, EGL, and Qwen host environment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
