"""Level 2 summary table and Level 3 metrics comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .math_agent import math_opinion_label
from .models import PipelineResult

GROUND_TRUTH_LABELS: dict[str, str] = {
    "Class1": "Normal",
    "Class2": "Tumor",
}

SUMMARY_COLUMNS = [
    "Имя файла",
    "% Красного (SVM)",
    "Мнение Math Agent",
    "Мнение VLM (Gemini)",
    "Финальное решение",
    "Источник",
    "Ground Truth",
    "Ошибка?",
    "Тип ошибки (если есть)",
]

METRICS_COLUMNS = [
    "Метод оценки",
    "Чувствительность (Sensitivity), %",
    "Специфичность (Specificity), %",
    "Общая точность (Global Accuracy), %",
]

BASELINE_METRICS: list[dict[str, Any]] = [
    {
        "Метод оценки": "SVM (baseline) (из статьи) [11]",
        "Чувствительность (Sensitivity), %": 83,
        "Специфичность (Specificity), %": 85,
        "Общая точность (Global Accuracy), %": 84,
    },
    {
        "Метод оценки": "Gemini 3.1 Pro (baseline) (из статьи) [11]",
        "Чувствительность (Sensitivity), %": 86,
        "Специфичность (Specificity), %": 93,
        "Общая точность (Global Accuracy), %": 90,
    },
    {
        "Метод оценки": "Врачи-нейрохирурги (HITL) (из статьи) [11]",
        "Чувствительность (Sensitivity), %": 98,
        "Специфичность (Specificity), %": 90,
        "Общая точность (Global Accuracy), %": 94,
    },
]


def ground_truth_label(class_name: str | None) -> str:
    if class_name is None:
        return ""
    return GROUND_TRUTH_LABELS.get(class_name, class_name)


def classify_error(vlm_decision: str | None, ground_truth: str) -> tuple[str, str]:
    if not vlm_decision or not ground_truth:
        return "", ""

    predicted_tumor = vlm_decision.strip().upper() == "TUMOR"
    actual_tumor = ground_truth == "Tumor"

    if predicted_tumor == actual_tumor:
        return "Нет", "—"

    if predicted_tumor and not actual_tumor:
        return "Да", "False Positive (Ложное срабатывание)"

    return "Да", "False Negative (Пропуск опухоли)"


def resolve_final_decision(result: PipelineResult) -> str:
    if result.final_decision:
        return result.final_decision.strip().upper()

    vlm_result = result.vlm_result or {}
    vlm_decision = (vlm_result.get("final_decision") or "").strip().upper()
    if vlm_decision in {"TUMOR", "NOISE"}:
        return vlm_decision

    math_decision = result.math_stats.get("math_decision", "")
    if math_decision in {"TUMOR", "NOISE"}:
        return math_decision
    return ""


def results_to_dataframe(results: list[PipelineResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for result in results:
        math_stats = result.math_stats
        vlm_result = result.vlm_result or {}
        ground_truth = ground_truth_label(result.ground_truth_class)

        vlm_decision = (vlm_result.get("final_decision") or "").strip().upper()
        if vlm_decision not in {"TUMOR", "NOISE"}:
            vlm_decision = "—" if result.decision_source == "Math" else ""

        final_decision = resolve_final_decision(result)
        if final_decision not in {"TUMOR", "NOISE", "UNCERTAIN"}:
            final_decision = ""

        is_error, error_type = classify_error(
            final_decision if final_decision in {"TUMOR", "NOISE"} else None,
            ground_truth,
        )

        rows.append(
            {
                "Имя файла": Path(result.image_path).name,
                "% Красного (SVM)": f"{math_stats['red_percentage']:.1f}%",
                "Мнение Math Agent": math_opinion_label(math_stats["math_decision"]),
                "Мнение VLM (Gemini)": vlm_decision,
                "Финальное решение": final_decision,
                "Источник": result.decision_source or "",
                "Ground Truth": ground_truth,
                "Ошибка?": is_error,
                "Тип ошибки (если есть)": error_type,
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def calculate_pipeline_metrics(df: pd.DataFrame) -> dict[str, float]:
    evaluated = df[df["Финальное решение"].isin(["TUMOR", "NOISE"])].copy()
    if evaluated.empty:
        return {"sensitivity": 0.0, "specificity": 0.0, "accuracy": 0.0}

    predicted_tumor = evaluated["Финальное решение"] == "TUMOR"
    actual_tumor = evaluated["Ground Truth"] == "Tumor"

    tp = int((predicted_tumor & actual_tumor).sum())
    tn = int((~predicted_tumor & ~actual_tumor).sum())
    fp = int((predicted_tumor & ~actual_tumor).sum())
    fn = int((~predicted_tumor & actual_tumor).sum())

    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    specificity = 100.0 * tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = 100.0 * (tp + tn) / len(evaluated) if len(evaluated) else 0.0

    return {
        "sensitivity": round(sensitivity, 1),
        "specificity": round(specificity, 1),
        "accuracy": round(accuracy, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": len(evaluated),
    }


def build_metrics_comparison(df: pd.DataFrame) -> pd.DataFrame:
    pipeline_metrics = calculate_pipeline_metrics(df)

    rows = list(BASELINE_METRICS)
    rows.append(
        {
            "Метод оценки": "Math Agent (cascade) + Gemini 3.1 Flash-Lite",
            "Чувствительность (Sensitivity), %": pipeline_metrics["sensitivity"],
            "Специфичность (Specificity), %": pipeline_metrics["specificity"],
            "Общая точность (Global Accuracy), %": pipeline_metrics["accuracy"],
        }
    )

    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def save_reports(
    results: list[PipelineResult],
    summary_csv: str | Path,
    metrics_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    summary_df = results_to_dataframe(results)
    summary_path = Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary table ({len(summary_df)} scans) to {summary_path}")

    metrics_df: pd.DataFrame | None = None
    if metrics_csv is not None:
        metrics_df = build_metrics_comparison(summary_df)
        metrics_path = Path(metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"Saved metrics comparison to {metrics_path}")

        pipeline_row = metrics_df.iloc[-1]
        print(
            "\n=== Math Agent (cascade) + Gemini 3.1 Flash-Lite ===\n"
            f"Sensitivity: {pipeline_row['Чувствительность (Sensitivity), %']}%\n"
            f"Specificity: {pipeline_row['Специфичность (Specificity), %']}%\n"
            f"Accuracy:    {pipeline_row['Общая точность (Global Accuracy), %']}%"
        )

    return summary_df, metrics_df
