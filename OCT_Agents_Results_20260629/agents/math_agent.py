"""Math Agent: empirical FPR estimation from SVM segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


class MathDecision(str, Enum):
    NOISE = "NOISE"
    UNCERTAIN = "UNCERTAIN"
    TUMOR = "TUMOR"


MATH_OPINION_LABELS: dict[str, str] = {
    MathDecision.NOISE.value: "Low Risk (Шум)",
    MathDecision.UNCERTAIN.value: "Uncertainty",
    MathDecision.TUMOR.value: "High Risk",
}


def math_opinion_label(math_decision: str) -> str:
    return MATH_OPINION_LABELS.get(math_decision, math_decision)


@dataclass
class MathStats:
    red_percentage: float
    yellow_percentage: float
    green_percentage: float
    clustering_ratio: float
    math_decision: str
    red_pixel_count: int
    tissue_pixel_count: int
    largest_component_pixels: int
    connected_components: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MathAgent:
    """Analyzes SVM segmentation images using area and spatial clustering metrics."""

    RED_AREA_NOISE_THRESHOLD = 5.0
    RED_AREA_TUMOR_THRESHOLD = 30.0
    CLUSTERING_NOISE_THRESHOLD = 20.0
    CLUSTERING_TUMOR_THRESHOLD = 50.0
    LEGEND_WIDTH_RATIO = 0.15

    def analyze(self, image_path: str | Path) -> MathStats:
        rgb = self._load_rgb(image_path)
        tissue_region = self._exclude_legend(rgb)

        red_mask, yellow_mask, green_mask, tissue_mask = self._classify_pixels(tissue_region)
        tissue_count = int(tissue_mask.sum())

        if tissue_count == 0:
            return MathStats(
                red_percentage=0.0,
                yellow_percentage=0.0,
                green_percentage=0.0,
                clustering_ratio=0.0,
                math_decision=MathDecision.NOISE.value,
                red_pixel_count=0,
                tissue_pixel_count=0,
                largest_component_pixels=0,
                connected_components=0,
            )

        red_count = int(red_mask.sum())
        yellow_count = int(yellow_mask.sum())
        green_count = int(green_mask.sum())

        red_pct = 100.0 * red_count / tissue_count
        yellow_pct = 100.0 * yellow_count / tissue_count
        green_pct = 100.0 * green_count / tissue_count

        largest_component, n_components = self._largest_connected_component(red_mask)
        clustering_ratio = 100.0 * largest_component / red_count if red_count > 0 else 0.0

        decision = self._make_decision(red_pct, clustering_ratio)

        return MathStats(
            red_percentage=round(red_pct, 2),
            yellow_percentage=round(yellow_pct, 2),
            green_percentage=round(green_pct, 2),
            clustering_ratio=round(clustering_ratio, 2),
            math_decision=decision.value,
            red_pixel_count=red_count,
            tissue_pixel_count=tissue_count,
            largest_component_pixels=largest_component,
            connected_components=n_components,
        )

    def _load_rgb(self, image_path: str | Path) -> np.ndarray:
        return np.array(Image.open(image_path).convert("RGB"))

    def _exclude_legend(self, rgb: np.ndarray) -> np.ndarray:
        legend_start = int(rgb.shape[1] * (1.0 - self.LEGEND_WIDTH_RATIO))
        return rgb[:, :legend_start]

    def _classify_pixels(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        red_ch, green_ch, blue_ch = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        white_mask = (red_ch > 240) & (green_ch > 240) & (blue_ch > 240)
        red_mask = (red_ch > 200) & (green_ch < 80) & (blue_ch < 80)
        yellow_mask = (red_ch > 200) & (green_ch > 200) & (blue_ch < 80)
        green_mask = (
            (green_ch > 200)
            & (red_ch < 150)
            & (blue_ch < 200)
            & ~red_mask
            & ~yellow_mask
        )
        tissue_mask = ~white_mask
        return red_mask, yellow_mask, green_mask, tissue_mask

    def _largest_connected_component(self, red_mask: np.ndarray) -> tuple[int, int]:
        if red_mask.sum() == 0:
            return 0, 0

        labeled, n_components = ndimage.label(red_mask)
        if n_components == 0:
            return 0, 0

        component_sizes = ndimage.sum(red_mask, labeled, range(1, n_components + 1))
        return int(max(component_sizes)), int(n_components)

    def _make_decision(self, red_percentage: float, clustering_ratio: float) -> MathDecision:
        if red_percentage < self.RED_AREA_NOISE_THRESHOLD:
            if clustering_ratio >= self.CLUSTERING_TUMOR_THRESHOLD and red_percentage >= 2.0:
                return MathDecision.UNCERTAIN
            return MathDecision.NOISE

        if red_percentage > self.RED_AREA_TUMOR_THRESHOLD:
            if clustering_ratio < self.CLUSTERING_NOISE_THRESHOLD:
                return MathDecision.UNCERTAIN
            return MathDecision.TUMOR

        if clustering_ratio >= self.CLUSTERING_TUMOR_THRESHOLD:
            return MathDecision.UNCERTAIN

        if clustering_ratio < self.CLUSTERING_NOISE_THRESHOLD:
            return MathDecision.NOISE

        return MathDecision.UNCERTAIN
