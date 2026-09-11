"""Persistent SAM2.1 automatic-mask server (runs in the SAM environment on CUDA).

Reads one JSON request per stdin line: {"id", "views": {label: image_path},
"out_dir"}; writes one JSON response line per request with per-view mask
candidates (id, bbox, area, centroid, predicted_iou, stability_score,
mask_path). The model is loaded once per episode instead of per call.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

ROOT = Path("/home/jli/state/qwen-rgb-sam2")
MAX_CANDIDATES = 24
MIN_AREA_PX = 25


def main() -> None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(ROOT / "sam2.1_hiera_small.pt"),
                       device="cuda", apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(model, points_per_side=24, points_per_batch=64,
                                           pred_iou_thresh=0.8, stability_score_thresh=0.9, crop_n_layers=0)
    sys.stdout.write(json.dumps({"ready": True, "model": "sam2.1_hiera_small", "device": "cuda"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        started = time.monotonic()
        out_dir = Path(request["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        response = {"id": request["id"], "views": {}}
        for label, path in request["views"].items():
            image = np.array(Image.open(path).convert("RGB"))
            with torch.inference_mode():
                annotations = generator.generate(image)
            annotations = [a for a in annotations if a["area"] >= MIN_AREA_PX]
            annotations.sort(key=lambda a: -a["predicted_iou"])
            candidates = []
            for index, annotation in enumerate(annotations[:MAX_CANDIDATES]):
                mask = annotation["segmentation"]
                ys, xs = np.where(mask)
                mask_path = out_dir / f"{label}-mask-{index}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                candidates.append({
                    "id": index, "mask_path": str(mask_path), "area": int(annotation["area"]),
                    "bbox_xywh": [int(v) for v in annotation["bbox"]],
                    "centroid": [float(xs.mean()), float(ys.mean())],
                    "predicted_iou": float(annotation["predicted_iou"]),
                    "stability_score": float(annotation["stability_score"]),
                    "mean_rgb": [int(v) for v in image[mask].mean(axis=0)],
                })
            response["views"][label] = {"candidates": candidates, "raw_count": len(annotations)}
        response["elapsed_s"] = time.monotonic() - started
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
