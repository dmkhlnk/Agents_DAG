"""Few-shot 5x5 reference panel for VLM decision-making."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .math_agent import MathAgent, MathStats, math_opinion_label

GRID_SIZE = 5
EXAMPLES_PER_CLASS = 12
CELL_WIDTH = 280
CELL_HEIGHT = 210
LABEL_HEIGHT = 28

GROUND_TRUTH_LABELS = {
    "Class1": "Normal",
    "Class2": "Tumor",
}


@dataclass
class FewShotExample:
    image_path: str
    ground_truth: str
    grid_label: str
    red_percentage: float
    clustering_ratio: float
    math_opinion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FewShotPanel:
    panel_image: Image.Image
    examples: list[FewShotExample]
    test_label: str
    layout_description: str


class FewShotGallery:
    """Builds a 5x5 panel: 12 Normal + 12 Tumor exemplars + 1 TEST cell."""

    def __init__(
        self,
        dataset_dir: str | Path,
        math_agent: MathAgent | None = None,
        seed: int = 42,
        manifest_path: str | Path | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.math_agent = math_agent or MathAgent()
        self.seed = seed
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._normal_pool: list[FewShotExample] = []
        self._tumor_pool: list[FewShotExample] = []
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        if self.manifest_path and self.manifest_path.exists():
            self._load_manifest()
        else:
            self._normal_pool = self._select_exemplars("Class1", "N", "Normal")
            self._tumor_pool = self._select_exemplars("Class2", "T", "Tumor")
            self._save_manifest()

        self._initialized = True
        print(
            f"Few-shot gallery: {len(self._normal_pool)} Normal + "
            f"{len(self._tumor_pool)} Tumor exemplars"
        )

    def build_panel(
        self,
        test_image_path: str | Path,
        test_math_stats: MathStats,
    ) -> FewShotPanel:
        self.initialize()
        test_path = Path(test_image_path).resolve()

        normal_examples = self._resolve_pool(self._normal_pool, test_path)
        tumor_examples = self._resolve_pool(self._tumor_pool, test_path)

        test_example = FewShotExample(
            image_path=str(test_path),
            ground_truth="TEST",
            grid_label="TEST",
            red_percentage=test_math_stats.red_percentage,
            clustering_ratio=test_math_stats.clustering_ratio,
            math_opinion=math_opinion_label(test_math_stats.math_decision),
        )

        grid_items = self._layout_grid(normal_examples, tumor_examples, test_example)
        panel_image = self._render_panel(grid_items)
        layout_description = self._describe_layout(normal_examples, tumor_examples, test_example)

        return FewShotPanel(
            panel_image=panel_image,
            examples=normal_examples + tumor_examples,
            test_label="TEST",
            layout_description=layout_description,
        )

    def _select_exemplars(self, class_name: str, prefix: str, ground_truth: str) -> list[FewShotExample]:
        class_dir = self.dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        candidates: list[tuple[float, Path, MathStats]] = []
        for image_path in sorted(class_dir.glob("*.jpg")):
            stats = self.math_agent.analyze(image_path)
            candidates.append((stats.red_percentage, image_path, stats))

        candidates.sort(key=lambda item: item[0])
        selected_paths = self._stratified_pick(candidates, EXAMPLES_PER_CLASS)

        examples: list[FewShotExample] = []
        for index, image_path in enumerate(selected_paths, start=1):
            stats = self.math_agent.analyze(image_path)
            examples.append(
                FewShotExample(
                    image_path=str(image_path.resolve()),
                    ground_truth=ground_truth,
                    grid_label=f"{prefix}{index}",
                    red_percentage=stats.red_percentage,
                    clustering_ratio=stats.clustering_ratio,
                    math_opinion=math_opinion_label(stats.math_decision),
                )
            )
        return examples

    def _stratified_pick(
        self,
        candidates: list[tuple[float, Path, MathStats]],
        count: int,
    ) -> list[Path]:
        if len(candidates) <= count:
            return [path for _, path, _ in candidates]

        rng = random.Random(self.seed)
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda item: item[0])

        step = len(shuffled) / count
        picked: list[Path] = []
        used: set[str] = set()

        for i in range(count):
            idx = min(int(i * step), len(shuffled) - 1)
            path = shuffled[idx][1]
            if str(path.resolve()) in used:
                for alt_idx in range(len(shuffled)):
                    alt_path = shuffled[alt_idx][1]
                    if str(alt_path.resolve()) not in used:
                        path = alt_path
                        break
            used.add(str(path.resolve()))
            picked.append(path)

        return picked

    def _resolve_pool(
        self,
        pool: list[FewShotExample],
        test_path: Path,
    ) -> list[FewShotExample]:
        resolved: list[FewShotExample] = []
        test_str = str(test_path)

        for example in pool:
            if example.image_path == test_str:
                replacement = self._find_replacement(example.ground_truth, test_str, resolved)
                resolved.append(replacement)
            else:
                resolved.append(example)

        return resolved

    def _find_replacement(
        self,
        ground_truth: str,
        exclude_paths: str,
        already_used: list[FewShotExample],
    ) -> FewShotExample:
        class_name = "Class1" if ground_truth == "Normal" else "Class2"
        prefix = "N" if ground_truth == "Normal" else "T"
        used_paths = {exclude_paths} | {item.image_path for item in already_used}

        candidates = sorted((self.dataset_dir / class_name).glob("*.jpg"), key=lambda p: p.name)
        for image_path in candidates:
            resolved = str(image_path.resolve())
            if resolved in used_paths:
                continue
            stats = self.math_agent.analyze(image_path)
            return FewShotExample(
                image_path=resolved,
                ground_truth=ground_truth,
                grid_label=f"{prefix}*",
                red_percentage=stats.red_percentage,
                clustering_ratio=stats.clustering_ratio,
                math_opinion=math_opinion_label(stats.math_decision),
            )

        raise RuntimeError(f"No replacement exemplar found for {ground_truth}")

    def _layout_grid(
        self,
        normal_examples: list[FewShotExample],
        tumor_examples: list[FewShotExample],
        test_example: FewShotExample,
    ) -> list[list[FewShotExample | None]]:
        cells: list[FewShotExample | None] = (
            list(normal_examples)
            + [test_example]
            + list(tumor_examples)
        )

        if len(cells) != GRID_SIZE * GRID_SIZE:
            raise ValueError(f"Expected {GRID_SIZE * GRID_SIZE} cells, got {len(cells)}")

        grid: list[list[FewShotExample | None]] = []
        for row in range(GRID_SIZE):
            start = row * GRID_SIZE
            grid.append(cells[start : start + GRID_SIZE])
        return grid

    def _render_panel(self, grid: list[list[FewShotExample | None]]) -> Image.Image:
        width = GRID_SIZE * CELL_WIDTH
        height = GRID_SIZE * (CELL_HEIGHT + LABEL_HEIGHT)
        panel = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()

        for row_idx, row in enumerate(grid):
            for col_idx, example in enumerate(row):
                if example is None:
                    continue

                x0 = col_idx * CELL_WIDTH
                y0 = row_idx * (CELL_HEIGHT + LABEL_HEIGHT)

                image = Image.open(example.image_path).convert("RGB")
                image = image.resize((CELL_WIDTH - 4, CELL_HEIGHT - 4), Image.Resampling.LANCZOS)
                panel.paste(image, (x0 + 2, y0 + 2))

                if example.grid_label == "TEST":
                    border_color = (220, 20, 60)
                    text = (
                        f"TEST | red {example.red_percentage:.1f}% | "
                        f"clust {example.clustering_ratio:.0f}% | {example.math_opinion}"
                    )
                elif example.ground_truth == "Normal":
                    border_color = (34, 139, 34)
                    text = (
                        f"{example.grid_label} Normal | red {example.red_percentage:.1f}% | "
                        f"clust {example.clustering_ratio:.0f}%"
                    )
                else:
                    border_color = (178, 34, 34)
                    text = (
                        f"{example.grid_label} Tumor | red {example.red_percentage:.1f}% | "
                        f"clust {example.clustering_ratio:.0f}%"
                    )

                draw.rectangle(
                    [x0, y0, x0 + CELL_WIDTH - 1, y0 + CELL_HEIGHT - 1],
                    outline=border_color,
                    width=3,
                )
                draw.rectangle(
                    [x0, y0 + CELL_HEIGHT, x0 + CELL_WIDTH - 1, y0 + CELL_HEIGHT + LABEL_HEIGHT - 1],
                    fill=(245, 245, 245),
                )
                draw.text((x0 + 4, y0 + CELL_HEIGHT + 6), text, fill=(0, 0, 0), font=font)

        return panel

    def _describe_layout(
        self,
        normal_examples: list[FewShotExample],
        tumor_examples: list[FewShotExample],
        test_example: FewShotExample,
    ) -> str:
        lines = [
            "Сетка 5x5 (25 ячеек, читай слева направо, сверху вниз):",
            "- Ячейки N1–N12: эталонные сканы с Ground Truth = Normal (шум/FPR, нет опухоли).",
            "- Ячейка TEST (центр сетки, строка 3, столбец 3): скан для оценки.",
            "- Ячейки T1–T12: эталонные сканы с Ground Truth = Tumor (истинная опухоль).",
            "",
            "Эталоны Normal:",
        ]

        for example in normal_examples:
            lines.append(
                f"  {example.grid_label}: red={example.red_percentage:.1f}%, "
                f"clustering={example.clustering_ratio:.1f}%, math={example.math_opinion}"
            )

        lines.append("")
        lines.append("Эталоны Tumor:")
        for example in tumor_examples:
            lines.append(
                f"  {example.grid_label}: red={example.red_percentage:.1f}%, "
                f"clustering={example.clustering_ratio:.1f}%, math={example.math_opinion}"
            )

        lines.append("")
        lines.append(
            f"TEST-скан: red={test_example.red_percentage:.1f}%, "
            f"clustering={test_example.clustering_ratio:.1f}%, math={test_example.math_opinion}"
        )
        return "\n".join(lines)

    def _save_manifest(self) -> None:
        if self.manifest_path is None:
            return

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": self.seed,
            "normal": [item.to_dict() for item in self._normal_pool],
            "tumor": [item.to_dict() for item in self._tumor_pool],
        }
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_manifest(self) -> None:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._normal_pool = [FewShotExample(**item) for item in payload["normal"]]
        self._tumor_pool = [FewShotExample(**item) for item in payload["tumor"]]
