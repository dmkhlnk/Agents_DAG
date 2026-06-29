#!/usr/bin/env python3
"""Quick evaluation of Math Agent decisions against dataset class labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from agents.math_agent import MathAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Math Agent on DATASETS_MDPI")
    parser.add_argument("dataset", nargs="?", default="DATASETS_MDPI")
    args = parser.parse_args()

    agent = MathAgent()
    rows: list[tuple[str, str, float, float]] = []

    for class_dir in sorted(Path(args.dataset).iterdir()):
        if not class_dir.is_dir() or not class_dir.name.startswith("Class"):
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            stats = agent.analyze(image_path)
            rows.append(
                (class_dir.name, stats.math_decision, stats.red_percentage, stats.clustering_ratio)
            )

    print(f"Total images: {len(rows)}\n")

    for cls in sorted({r[0] for r in rows}):
        subset = [r for r in rows if r[0] == cls]
        decisions = Counter(r[1] for r in subset)
        avg_red = sum(r[2] for r in subset) / len(subset)
        avg_cluster = sum(r[3] for r in subset) / len(subset)
        print(f"{cls} (n={len(subset)})")
        print(f"  decisions: {dict(decisions)}")
        print(f"  avg red%: {avg_red:.1f}, avg clustering: {avg_cluster:.1f}%")
        print()


if __name__ == "__main__":
    main()
