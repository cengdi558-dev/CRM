"""Load the SSA semantic cache and attach it to an ``ImageFolder``."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SemanticCache:
    """Lookup from image path (or file name) to ``[levels, dim]`` embeddings."""

    def __init__(self, path: str, fallback_dim: int | None = None) -> None:
        data = np.load(path, allow_pickle=False)
        self.paths = [str(p) for p in data["paths"]]
        self.emb = torch.from_numpy(np.asarray(data["emb"], dtype=np.float32))
        self.dim = self.emb.shape[-1]
        self.levels = self.emb.shape[1]
        self._by_path = {p: i for i, p in enumerate(self.paths)}
        self._by_name = {os.path.basename(p): i for i, p in enumerate(self.paths)}
        # any image missing from the cache (e.g. a split that was not described)
        # falls back to a neutral vector instead of aborting a training run
        self._fallback = torch.zeros(self.levels, fallback_dim or self.dim)

    def __len__(self) -> int:
        return self.emb.shape[0]

    def get(self, path: str) -> torch.Tensor:
        idx = self._by_path.get(path)
        if idx is None:
            idx = self._by_name.get(os.path.basename(path))
        if idx is None:
            return self._fallback.clone()
        return self.emb[idx]


class SemanticDataset(Dataset):
    """Wrap a dataset whose items are ``(image, label)`` or ``(image, label, path)``."""

    def __init__(self, base: Dataset, cache: SemanticCache) -> None:
        self.base = base
        self.cache = cache

    def __len__(self) -> int:
        return len(self.base)

    @property
    def classes(self):
        return getattr(self.base, "classes", None)

    @property
    def samples(self):
        return getattr(self.base, "samples", None)

    def __getitem__(self, index: int):
        item = self.base[index]
        image, label = item[0], item[1]
        path = item[2] if len(item) > 2 else self.base.samples[index][0]  # type: ignore[attr-defined]
        return image, label, self.cache.get(path)

