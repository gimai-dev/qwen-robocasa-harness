"""Pinned source-image DINOv2 comparator; never exposed to the policy."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = "dinov2_vits14"
PREPROCESSING = {
    "input": "rgb_uint8_256x256",
    "resize": [224, 224],
    "interpolation": "bicubic",
    "antialias": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "embedding": "normalized_cls_token",
}
PATCH_GRID_SIZE = 16
PATCH_SIZE_INPUT_PX = 16.0


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class DinoAuthority:
    model_name: str
    repo_commit: str
    repo_tree: str
    weights_sha256: str
    torch_version: str
    numpy_version: str
    pillow_version: str
    preprocessing: dict[str, object]

    @classmethod
    def from_paths(cls, *, repo: Path, weights: Path) -> DinoAuthority:
        if repo.is_symlink() or not (repo / ".git").is_dir():
            raise RuntimeError("DINO repository authority is invalid")
        if weights.is_symlink() or not weights.is_file():
            raise RuntimeError("DINO weights authority is invalid")
        if _git(repo, "status", "--porcelain"):
            raise RuntimeError("DINO repository must be clean")
        return cls(
            model_name=MODEL_NAME,
            repo_commit=_git(repo, "rev-parse", "HEAD"),
            repo_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
            weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            torch_version=importlib.metadata.version("torch"),
            numpy_version=importlib.metadata.version("numpy"),
            pillow_version=importlib.metadata.version("Pillow"),
            preprocessing=dict(PREPROCESSING),
        )

    def verify(self, *, repo: Path, weights: Path) -> None:
        current = self.from_paths(repo=repo, weights=weights)
        if current != self:
            if current.weights_sha256 != self.weights_sha256:
                raise RuntimeError("DINO weights authority drift")
            raise RuntimeError("DINO repository or runtime authority drift")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DinoEncoder:
    """Small, deterministic frozen DINOv2 image encoder."""

    def __init__(self, *, repo: Path, weights: Path, authority: DinoAuthority) -> None:
        authority.verify(repo=repo, weights=weights)
        import torch

        self._torch = torch
        model = torch.hub.load(
            str(repo), authority.model_name, source="local", pretrained=False
        )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.eval().to(self._device)

    def _tensor(self, image: object) -> object:
        torch = self._torch
        pixels = np.asarray(image, dtype=np.uint8)
        if pixels.shape != (256, 256, 3):
            raise ValueError("DINO input must be official 256x256 RGB")
        resized = Image.fromarray(pixels, mode="RGB").resize(
            (224, 224), resample=Image.Resampling.BICUBIC, reducing_gap=None
        )
        tensor = torch.from_numpy(
            np.asarray(resized, dtype=np.float32).copy()
        ).permute(2, 0, 1)[None] / 255.0
        mean = torch.tensor(PREPROCESSING["mean"], dtype=tensor.dtype)[:, None, None]
        std = torch.tensor(PREPROCESSING["std"], dtype=tensor.dtype)[:, None, None]
        return ((tensor - mean) / std).to(self._device)

    def encode(self, image: object) -> np.ndarray:
        torch = self._torch
        tensor = self._tensor(image)
        with torch.inference_mode():
            embedding = self._model(tensor)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].detach().cpu().numpy().astype(np.float64)

    def encode_patches(self, image: object) -> np.ndarray:
        """Return normalized 16x16 public-image patch descriptors."""
        torch = self._torch
        tensor = self._tensor(image)
        with torch.inference_mode():
            features = self._model.forward_features(tensor)["x_norm_patchtokens"]
            features = torch.nn.functional.normalize(features, dim=-1)
        value = features[0].detach().cpu().numpy().astype(np.float64)
        if value.shape[0] != PATCH_GRID_SIZE**2:
            raise RuntimeError("DINO patch-grid authority drifted")
        return value.reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE, value.shape[-1])


def cosine_similarity(first: object, second: object) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("DINO embeddings are invalid")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise ValueError("DINO embedding is degenerate")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
