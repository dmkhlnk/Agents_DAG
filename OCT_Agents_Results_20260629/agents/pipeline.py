"""Two-agent pipeline: Math Agent -> VLM Agent (strict cascade)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .math_agent import MathAgent, MathDecision, MathStats
from .models import PipelineResult
from .vlm_agent import VLMAgent, VLMResult


class AgentPipeline:
    def __init__(
        self,
        math_agent: MathAgent | None = None,
        vlm_agent: VLMAgent | None = None,
        skip_vlm: bool = False,
        cascade: bool = True,
    ):
        self.math_agent = math_agent or MathAgent()
        self.skip_vlm = skip_vlm
        self.cascade = cascade
        self.vlm_agent = None if skip_vlm else (vlm_agent or VLMAgent())

    def process_image(
        self,
        image_path: str | Path,
        ground_truth_class: str | None = None,
    ) -> PipelineResult:
        image_path = Path(image_path)
        math_stats = self.math_agent.analyze(image_path)

        vlm_result: VLMResult | None = None
        final_decision: str | None = None
        decision_source: str | None = None

        if self.skip_vlm or self.vlm_agent is None:
            final_decision, decision_source = self._resolve_math_only(math_stats)
        elif self.cascade and math_stats.math_decision != MathDecision.UNCERTAIN.value:
            final_decision = math_stats.math_decision
            decision_source = "Math"
        else:
            vlm_result = self.vlm_agent.analyze(image_path, math_stats)
            final_decision = self._normalize_binary_decision(vlm_result.final_decision)
            decision_source = "VLM"

        agreement = self._check_agreement(math_stats, final_decision)

        return PipelineResult(
            image_path=str(image_path),
            ground_truth_class=ground_truth_class,
            math_stats=math_stats.to_dict(),
            vlm_result=vlm_result.to_dict() if vlm_result else None,
            final_decision=final_decision,
            decision_source=decision_source,
            agreement=agreement,
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

    def process_dataset(
        self,
        dataset_dir: str | Path,
        output_path: str | Path | None = None,
        summary_csv: str | Path | None = None,
        metrics_csv: str | Path | None = None,
        limit: int | None = None,
    ) -> list[PipelineResult]:
        dataset_dir = Path(dataset_dir)
        image_paths: list[tuple[Path, str | None]] = []

        for class_dir in sorted(dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.glob("*.jpg")):
                image_paths.append((image_path, class_dir.name))

        if limit is not None:
            image_paths = image_paths[:limit]

        results: list[PipelineResult] = []
        for image_path, ground_truth in image_paths:
            print(f"Processing: {image_path.name} ({ground_truth})")
            results.append(self.process_image(image_path, ground_truth))

        if output_path is not None:
            self._save_results(results, output_path)

        if summary_csv is not None or metrics_csv is not None:
            from .reporting import save_reports

            save_reports(
                results,
                summary_csv=summary_csv or "results/scans_summary.csv",
                metrics_csv=metrics_csv,
            )

        self._print_cascade_stats(results)
        return results

    def _resolve_math_only(self, math_stats: MathStats) -> tuple[str | None, str]:
        if math_stats.math_decision == MathDecision.UNCERTAIN.value:
            return "UNCERTAIN", "Math"
        return math_stats.math_decision, "Math"

    def _normalize_binary_decision(self, decision: str) -> str:
        normalized = decision.strip().upper()
        if normalized not in {"TUMOR", "NOISE"}:
            return "NOISE"
        return normalized

    def _check_agreement(self, math_stats: MathStats, final_decision: str | None) -> bool | None:
        if final_decision is None or final_decision == "UNCERTAIN":
            return None

        math_label = self._math_to_binary(math_stats.math_decision)
        if math_label == "UNCERTAIN":
            return None
        return math_label == final_decision

    def _math_to_binary(self, math_decision: str) -> str:
        if math_decision == MathDecision.TUMOR.value:
            return "TUMOR"
        if math_decision == MathDecision.NOISE.value:
            return "NOISE"
        return "UNCERTAIN"

    def _print_cascade_stats(self, results: list[PipelineResult]) -> None:
        if not results:
            return

        math_count = sum(1 for r in results if r.decision_source == "Math" and r.final_decision in {"TUMOR", "NOISE"})
        vlm_count = sum(1 for r in results if r.decision_source == "VLM")
        uncertain = sum(1 for r in results if r.final_decision == "UNCERTAIN")

        if self.skip_vlm:
            return

        mode = "cascade" if self.cascade else "always-VLM"
        print(
            f"\nCascade stats ({mode}): "
            f"Math-only decisions={math_count}, VLM calls={vlm_count}, UNCERTAIN={uncertain}"
        )

    def _save_results(self, results: list[PipelineResult], output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.to_dict() for result in results]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(results)} results to {output_path}")
