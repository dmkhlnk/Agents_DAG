#!/usr/bin/env python3
"""CLI entry point for the OCT tumor FPR agent pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agents.pipeline import AgentPipeline
from agents.reporting import save_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-agent pipeline for dynamic FPR management on OCT SVM segmentations.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="DATASETS_MDPI",
        help="Path to a single image or dataset directory (default: DATASETS_MDPI)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/pipeline_results.json",
        help="Output JSON path for raw pipeline results",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/scans_summary.csv",
        help="Level 2 summary table (CSV) for all scans",
    )
    parser.add_argument(
        "--metrics-csv",
        default="results/metrics_comparison.csv",
        help="Level 3 metrics comparison table (CSV)",
    )
    parser.add_argument(
        "--math-only",
        action="store_true",
        help="Run only Math Agent (skip Gemini VLM call)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N images in dataset mode",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model name (default: gemini-3.1-flash-lite)",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=10,
        help="Max Gemini API requests per minute (default: 10)",
    )
    parser.add_argument(
        "--always-vlm",
        action="store_true",
        help="Disable cascade: call Gemini for every scan (old behaviour)",
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable 5x5 few-shot panel (12 Normal + 12 Tumor + 1 TEST)",
    )
    parser.add_argument(
        "--few-shot-dataset",
        default="DATASETS_MDPI",
        help="Dataset used to pick few-shot exemplars (default: DATASETS_MDPI)",
    )
    parser.add_argument(
        "--preview-panel",
        default=None,
        metavar="PATH",
        help="Save 5x5 few-shot panel image for a single test scan and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)

    if args.preview_panel:
        from agents.few_shot import FewShotGallery
        from agents.math_agent import MathAgent

        math_agent = MathAgent()
        gallery = FewShotGallery(
            dataset_dir=args.few_shot_dataset,
            math_agent=math_agent,
            manifest_path="results/few_shot_exemplars.json",
        )
        stats = math_agent.analyze(input_path)
        panel = gallery.build_panel(input_path, stats)
        out = Path(args.preview_panel)
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.panel_image.save(out)
        print(f"Saved 5x5 panel to {out}")
        print(panel.layout_description)
        return

    vlm_agent = None
    if not args.math_only:
        from agents.few_shot import FewShotGallery
        from agents.vlm_agent import VLMAgent

        few_shot_gallery = None
        if not args.no_few_shot:
            few_shot_gallery = FewShotGallery(
                dataset_dir=args.few_shot_dataset,
                manifest_path="results/few_shot_exemplars.json",
            )

        vlm_agent = VLMAgent(
            model=args.model,
            max_requests_per_minute=args.max_rpm,
            few_shot_gallery=few_shot_gallery,
            use_few_shot=not args.no_few_shot,
        )

    pipeline = AgentPipeline(
        vlm_agent=vlm_agent,
        skip_vlm=args.math_only,
        cascade=not args.always_vlm,
    )

    if input_path.is_file():
        ground_truth = input_path.parent.name if input_path.parent.name.startswith("Class") else None
        result = pipeline.process_image(input_path, ground_truth)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        save_reports([result], summary_csv=args.summary_csv, metrics_csv=None)
        return

    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    results = pipeline.process_dataset(
        input_path,
        output_path=args.output,
        summary_csv=args.summary_csv,
        metrics_csv=None if args.math_only else args.metrics_csv,
        limit=args.limit,
    )

    tumor_count = sum(1 for r in results if r.final_decision == "TUMOR")
    print(f"\nDone. Processed {len(results)} images. Final TUMOR verdicts: {tumor_count}")


if __name__ == "__main__":
    main()
