#!/usr/bin/env python3
"""Build Level 2/3 CSV reports from saved pipeline JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.models import PipelineResult
from agents.reporting import save_reports


def load_results(path: Path) -> list[PipelineResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results: list[PipelineResult] = []
    for item in payload:
        if "final_decision" not in item:
            vlm = item.get("vlm_result") or {}
            decision = (vlm.get("final_decision") or "").strip().upper()
            item["final_decision"] = decision if decision in {"TUMOR", "NOISE"} else None
            item["decision_source"] = "VLM" if vlm else "Math"
        results.append(PipelineResult(**item))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV reports from pipeline JSON")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="results/pipeline_results.json",
        help="Path to pipeline_results.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/scans_summary.csv",
    )
    parser.add_argument(
        "--metrics-csv",
        default="results/metrics_comparison.csv",
    )
    args = parser.parse_args()

    results = load_results(Path(args.json_path))
    save_reports(results, summary_csv=args.summary_csv, metrics_csv=args.metrics_csv)


if __name__ == "__main__":
    main()
