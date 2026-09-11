"""SAM2 regions with calibrated multi-view geometry (no action selection).

``SamClient`` drives the persistent CUDA SAM server. ``build_regions`` pairs
mask centroids across the three calibrated views by ray triangulation and
colour agreement, producing region records with world/base positions. It is
the only object information the policy receives beyond the raw images.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .kinematics import quat_xyzw_to_matrix

SAM_PYTHON = "/home/jli/state/qwen-rgb-sam2/.venv/bin/python"
SAM_SERVER = Path(__file__).resolve().with_name("sam_server.py")
MAX_RAY_GAP_M = 0.03
MAX_COLOR_DISTANCE = 70.0
MAX_REGIONS = 14
MAX_SINGLE_VIEW = 6


class SamClient:
    def __init__(self, log_path: Path) -> None:
        self.log = log_path.open("w")
        self.process = subprocess.Popen([SAM_PYTHON, str(SAM_SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        ready = json.loads(self.process.stdout.readline())
        if not ready.get("ready"):
            raise RuntimeError("SAM server failed to start")
        self.counter = 0
        self.total_elapsed_s = 0.0

    def segment(self, views: Mapping[str, str], out_dir: Path) -> dict:
        self.counter += 1
        request = {"id": self.counter, "views": dict(views), "out_dir": str(out_dir)}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("SAM server exited")
        response = json.loads(line)
        self.total_elapsed_s += float(response.get("elapsed_s", 0.0))
        return response

    def close(self) -> None:
        try:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()
        self.log.close()


def _ray(calibration: Mapping[str, object], uv: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(calibration["camera_position_world_m"], dtype=float)
    xmat = np.asarray(calibration["camera_xmat_world"], dtype=float)
    local = np.array([(uv[0] - calibration["cx_px"]) / calibration["fx_px"],
                      -(uv[1] - calibration["cy_px"]) / calibration["fy_px"], -1.0])
    direction = xmat @ local
    return position, direction / np.linalg.norm(direction)


def _closest_points(oa, da, ob, db):
    w0 = oa - ob
    a, b, c = da @ da, da @ db, db @ db
    d, e = da @ w0, db @ w0
    denominator = a * c - b * b
    if denominator < 1e-9:
        return None
    sa = (b * e - c * d) / denominator
    sb = (a * e - b * d) / denominator
    if sa <= 0.05 or sb <= 0.05:
        return None
    pa, pb = oa + sa * da, ob + sb * db
    return (pa + pb) / 2, float(np.linalg.norm(pa - pb))


def triangulate(cal_a, uv_a, cal_b, uv_b):
    oa, da = _ray(cal_a, uv_a)
    ob, db = _ray(cal_b, uv_b)
    return _closest_points(oa, da, ob, db)


def build_regions(sam: Mapping[str, object], calibration: Mapping[str, Mapping[str, object]],
                  state: Mapping[str, object]) -> dict:
    """Pair SAM candidates across views into region records with 3D estimates."""
    base_p = np.asarray(state["base_world_position_m"], dtype=float)
    base_r = quat_xyzw_to_matrix(state["base_world_quat_xyzw"])
    views = {label: list(sam["views"][label]["candidates"]) for label in sam["views"]}
    pairs = []
    for a, b in (("left", "right"), ("left", "wrist"), ("right", "wrist")):
        if a not in views or b not in views:
            continue
        for ca in views[a]:
            for cb in views[b]:
                color = float(np.linalg.norm(np.asarray(ca["mean_rgb"], float) - np.asarray(cb["mean_rgb"], float)))
                if color > MAX_COLOR_DISTANCE:
                    continue
                if "wrist" not in (a, b):
                    ratio = max(ca["area"], cb["area"]) / max(1, min(ca["area"], cb["area"]))
                    if ratio > 4.0:
                        continue
                result = triangulate(calibration[a], ca["centroid"], calibration[b], cb["centroid"])
                if result is None:
                    continue
                point, gap = result
                if gap > MAX_RAY_GAP_M or not (0.0 <= point[2] <= 2.5) or np.linalg.norm(point[:2] - base_p[:2]) > 2.0:
                    continue
                pairs.append((gap + 0.0002 * color, a, ca["id"], b, cb["id"], point, gap))
    pairs.sort(key=lambda item: item[0])
    used: set[tuple[str, int]] = set()
    regions: list[dict] = []
    for score, a, ia, b, ib, point, gap in pairs:
        if (a, ia) in used or (b, ib) in used:
            continue
        used.update({(a, ia), (b, ib)})
        ca = next(c for c in views[a] if c["id"] == ia)
        cb = next(c for c in views[b] if c["id"] == ib)
        depth_a = float(np.linalg.norm(point - np.asarray(calibration[a]["camera_position_world_m"])))
        width_m = ca["bbox_xywh"][2] * depth_a / float(calibration[a]["fx_px"])
        height_m = ca["bbox_xywh"][3] * depth_a / float(calibration[a]["fy_px"])
        base_point = base_r.T @ (point - base_p)
        regions.append({
            "id": f"r{len(regions) + 1}", "world_m": [round(float(v), 3) for v in point],
            "base_m": [round(float(v), 3) for v in base_point], "ray_gap_m": round(gap, 3),
            "approx_size_m": [round(width_m, 2), round(height_m, 2)],
            "mean_rgb": ca["mean_rgb"],
            "views": {a: {"uv": [round(v) for v in ca["centroid"]], "bbox_xywh": ca["bbox_xywh"], "area_px": ca["area"]},
                      b: {"uv": [round(v) for v in cb["centroid"]], "bbox_xywh": cb["bbox_xywh"], "area_px": cb["area"]}},
            "sam_ids": {a: ia, b: ib},
        })
        if len(regions) >= MAX_REGIONS:
            break
    # a third view supporting an existing region
    for region in regions:
        for label in ("left", "right", "wrist"):
            if label in region["views"] or label not in views or label not in calibration:
                continue
            best = None
            for c in views[label]:
                if (label, c["id"]) in used:
                    continue
                color = float(np.linalg.norm(np.asarray(c["mean_rgb"], float) - np.asarray(region["mean_rgb"], float)))
                if color > MAX_COLOR_DISTANCE:
                    continue
                for other, view in region["views"].items():
                    result = triangulate(calibration[label], c["centroid"], calibration[other], view["uv"])
                    if result is None:
                        continue
                    point, gap = result
                    if gap <= MAX_RAY_GAP_M and np.linalg.norm(point - np.asarray(region["world_m"])) < 0.05:
                        if best is None or gap < best[0]:
                            best = (gap, c)
            if best is not None:
                c = best[1]
                used.add((label, c["id"]))
                region["views"][label] = {"uv": [round(v) for v in c["centroid"]], "bbox_xywh": c["bbox_xywh"], "area_px": c["area"]}
                region["sam_ids"][label] = c["id"]
    single = {}
    for label, candidates in views.items():
        rest = [c for c in candidates if (label, c["id"]) not in used][:MAX_SINGLE_VIEW]
        single[label] = [{"uv": [round(v) for v in c["centroid"]], "bbox_xywh": c["bbox_xywh"], "area_px": c["area"],
                          "mean_rgb": c["mean_rgb"], "sam_id": c["id"]} for c in rest]
    return {"regions": regions, "unpaired": single, "sam_elapsed_s": sam.get("elapsed_s")}


def perceive(sam_client: SamClient, observation: Mapping[str, object], out_dir: Path) -> dict:
    images = {label: record["path"] for label, record in observation["images"].items()}
    started = time.monotonic()
    sam = sam_client.segment(images, out_dir)
    regions = build_regions(sam, observation["camera_calibration"], observation["public_state"])
    regions["perception_wall_s"] = round(time.monotonic() - started, 2)
    (out_dir / "regions.json").write_text(json.dumps(regions, indent=1))
    return regions
