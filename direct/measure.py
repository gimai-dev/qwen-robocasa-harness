"""Depth-region measurement (ablation of the agent-as-policy ``deproject`` region mode).

The ``measure`` action asks the simulator child for a 256x256 RGB+depth render of
the three cameras (no motion, no simulator steps) and returns 3D statistics of
every valid depth pixel inside a pixel rectangle of one camera, optionally keeping
only points above a horizontal world plane. Depth samples are the VISIBLE surface,
so an object's centre sits about half its z-extent below ``top_point``.

Pixel (u, v) with depth d (metres along the optical axis) maps to the world as
    X = cam_pos + xmat @ [(u - cx) / fx * d, -(v - cy) / fy * d, -d]
using the MuJoCo camera convention (x right, y up, -z forward). ``u`` grows to
the right and ``v`` downward, matching the 256x256 images the model sees.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .actions import short_response_schema
from .methods import Method

RENDER_SIZE = 256
MIN_POINTS = 5
PROMPTS = Path(__file__).resolve().with_name("prompts")


def deproject_region(depth: np.ndarray, calibration: Mapping[str, object], region: tuple[int, int, int, int],
                     above_z: float | None) -> dict:
    """World-frame statistics of the valid depth points in ``region`` = (u0, v0, u1, v1), inclusive."""
    h, w = depth.shape
    u0, v0, u1, v1 = region
    u0, u1 = max(0, u0), min(w - 1, u1)
    v0, v1 = max(0, v0), min(h - 1, v1)
    if u1 <= u0 or v1 <= v0:
        return {"ok": False, "error": f"region lies outside the {w}x{h} image"}
    fx, fy, cx, cy = (float(calibration[k]) for k in ("fx", "fy", "cx", "cy"))
    cam_pos = np.asarray(calibration["camera_position_world_m"], dtype=float)
    xmat = np.asarray(calibration["camera_xmat_world"], dtype=float).reshape(3, 3)
    vs, us = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    d = depth[v0:v1 + 1, u0:u1 + 1].astype(float).ravel()
    ok = np.isfinite(d) & (d > 0)
    us, vs, d = us.ravel()[ok], vs.ravel()[ok], d[ok]
    local = np.stack([(us - cx) / fx * d, -(vs - cy) / fy * d, -d], axis=1)
    points = local @ xmat.T + cam_pos
    if above_z is not None:
        points = points[points[:, 2] > float(above_z)]
    if len(points) < MIN_POINTS:
        return {"ok": False, "n_points": int(len(points)),
                "error": f"fewer than {MIN_POINTS} valid depth points in the region"
                         + (" above above_z" if above_z is not None else "")}
    lo, hi = points.min(axis=0), points.max(axis=0)
    centroid = points.mean(axis=0)
    top = points[int(np.argmax(points[:, 2]))]
    r = lambda v: [round(float(x), 4) for x in v]
    return {"ok": True, "n_points": int(len(points)), "centroid_world_m": r(centroid), "min_world_m": r(lo),
            "max_world_m": r(hi), "extent_m": r(hi - lo), "top_point_world_m": r(top)}


def measure_from_render(cameras: Mapping[str, Mapping[str, object]], cam: str, region: tuple[int, int, int, int],
                        above_z: float | None) -> dict:
    """Apply ``deproject_region`` to one camera entry of a child ``render`` receipt."""
    entry = cameras[cam]
    depth = np.load(entry["depth_npy"])
    if depth.shape != (RENDER_SIZE, RENDER_SIZE):
        return {"ok": False, "error": f"unexpected depth shape {depth.shape}"}
    return deproject_region(depth, entry, region, above_z)


class MeasureMethod(Method):
    """clean + the ``measure`` action. Nothing else changes: same images, regions,
    slot, IK, budgets and success predicate."""
    name = "measure"
    allow_measure = True

    def __init__(self, *, run: Path, config: Mapping[str, object]) -> None:
        super().__init__(run=run, config=config)
        self.max_consecutive_measures = int(self.config.get("max_consecutive_measures", 6))
        self.consecutive_measures = 0
        self.measures = 0
        self.measure_failures = 0

    def prompt_suffix(self) -> str:
        return (PROMPTS / "measure_ee_short.txt").read_text()

    def short_schema(self, interface: str) -> dict:
        return short_response_schema(interface=interface, measure=True)

    def after_receipt(self, ctx: Mapping[str, object], action, receipt) -> None:
        if action.kind == "measure":
            self.consecutive_measures += 1
            self.measures += 1
            if not receipt.detail.get("ok"):
                self.measure_failures += 1
        elif receipt.steps > 0:
            self.consecutive_measures = 0

    def finalize(self) -> dict:
        return {"measures": self.measures, "measure_failures": self.measure_failures,
                "max_consecutive_measures": self.max_consecutive_measures}

