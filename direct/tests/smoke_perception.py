"""Run the SAM server + region pairing on saved frames; report timing and regions."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from direct.perception import SamClient, perceive
run = Path(sys.argv[1])
obs = json.loads((run / "sim/mailbox/observation-000000.json").read_text())
t = time.monotonic(); sam = SamClient(run / "sam.log"); print(f"sam start {time.monotonic()-t:.1f}s")
for i in range(2):
    t = time.monotonic(); regions = perceive(sam, obs, run / f"perception-{i}"); print(f"perceive {time.monotonic()-t:.1f}s sam={regions['sam_elapsed_s']:.2f}s")
print(json.dumps({k: v for k, v in regions.items() if k != "unpaired"}, indent=1)[:3500])
print("unpaired counts", {k: len(v) for k, v in regions["unpaired"].items()})
sam.close()
