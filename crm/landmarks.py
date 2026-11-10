"""Landmark acquisition for COC (Sect. III-B).

The paper uses "a landmark detector"; this module keeps the detector pluggable
and dependency-optional, and degrades to the fixed template when nothing is
available. Three sources are supported, in order:

1. a precomputed JSON file (``{ "<image path>": [[x, y], ...], ... }``) — the
   practical route for benchmarks that ship landmarks;
2. MediaPipe FaceMesh, if ``mediapipe`` is installed;
3. the normalised template in :mod:`crm.regions`.

Everything returns points in ``[0, 1]`` relative to the image box, which is what
``region_masks`` expects.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np

from .regions import boxes_from_points

PointCloud = np.ndarray


class NullDetector:
    """Always returns ``None``; the caller then uses the template."""

    name = "null"

    def detect(self, image_path: str) -> PointCloud | None:  # noqa: ARG002
        return None


class JsonLandmarkProvider:
    """Look up precomputed landmarks by path or by file name."""

    name = "json"

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            self.table: dict[str, Any] = json.load(fh)

    def detect(self, image_path: str) -> PointCloud | None:
        for key in (image_path, os.path.abspath(image_path), os.path.basename(image_path)):
            if key in self.table:
                return np.asarray(self.table[key], dtype=np.float32)
        return None


class MediaPipeDetector:
    """FaceMesh adapter; requires ``mediapipe`` and yields ~468 points."""

    name = "mediapipe"

    def __init__(self, refine: bool = True) -> None:
        import mediapipe as mp  # local import: optional dependency

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=refine
        )

    def detect(self, image_path: str) -> PointCloud | None:
        from PIL import Image

        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        pts = res.multi_face_landmarks[0].landmark
        return np.asarray([[p.x, p.y] for p in pts], dtype=np.float32)


def build_detector(prefer: Iterable[str] = ("mediapipe",), json_path: str | None = None):
    """Return the first detector that can be constructed, else ``NullDetector``."""
    if json_path:
        if not os.path.exists(json_path):
            raise FileNotFoundError(json_path)
        return JsonLandmarkProvider(json_path)

    for name in prefer:
        if name == "mediapipe":
            try:
                return MediaPipeDetector()
            except Exception:
                continue
    return NullDetector()


def boxes_for_image(detector, image_path: str) -> np.ndarray:
    """``[8, 4]`` normalised boxes for one image (template when undetected)."""
    points = detector.detect(image_path)
    return np.asarray(boxes_from_points(points), dtype=np.float32)


def batch_boxes(detector, paths: list[str]) -> np.ndarray:
    """``[B, 8, 4]`` boxes for a batch of paths."""
    return np.stack([boxes_for_image(detector, p) for p in paths], axis=0)
