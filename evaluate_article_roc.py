#!/usr/bin/env python3
"""Build ROC figure for the paper: Geo/Math (nested-LOOCV Platt) vs Full DAG.

Geo/Math ROC = empirical ROC of nested-LOOCV Platt probabilities over red%.
Its operating point (marker + Table metrics) = Youden point of that ROC.
Full DAG ROC = empirical ROC over the cascade tumor score; marker = its Youden.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from evaluate_roc import (
    load_samples,
    metrics_from_predictions,
    find_youden_threshold,
    roc_curve_from_arrays,
)
from evaluate_geostat_platt_roc import (
    nested_loocv_platt_probabilities,
    youden_on_roc,
)


def load_full_dag_scores(results_json: Path) -> pd.DataFrame:
    data = json.loads(results_json.read_text(encoding="utf-8"))
    rows: list[dict[str, float | int | str]] = []

    for item in data:
        gt = item.get("ground_truth_class")
        if gt not in {"Class1", "Class2"}:
            continue

        y_true = 1 if gt == "Class2" else 0
        final_decision = (item.get("final_decision") or "").strip().upper()
        source = item.get("decision_source")
        vlm_result = item.get("vlm_result") or {}

        # For VLM-routed scans use model confidence as tumor probability.
        # For deterministic Math exits use fixed extreme probabilities.
        if source == "VLM" and "confidence" in vlm_result:
            confidence = float(vlm_result["confidence"]) / 100.0
            score = confidence if final_decision == "TUMOR" else (1.0 - confidence)
        else:
            score = 0.99 if final_decision == "TUMOR" else 0.01

        rows.append(
            {
                "filename": Path(item.get("image_path", "")).name,
                "ground_truth": y_true,
                "tumor_score": float(np.clip(score, 0.0, 1.0)),
                "source": source or "",
                "final_decision": final_decision,
            }
        )

    return pd.DataFrame(rows)


def dag_metrics_at_youden(df: pd.DataFrame) -> tuple[dict[str, float | int], float]:
    y_true = df["ground_truth"].to_numpy()
    scores = df["tumor_score"].to_numpy()
    threshold = find_youden_threshold(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    return metrics_from_predictions(y_true, y_pred), threshold


def plot_combined_roc(
    math_fpr: np.ndarray,
    math_tpr: np.ndarray,
    math_auc: float,
    math_youden: dict[str, float],
    dag_fpr: np.ndarray,
    dag_tpr: np.ndarray,
    dag_auc: float,
    dag_metrics: dict[str, float | int],
    dag_threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    ax.plot(
        math_fpr,
        math_tpr,
        linewidth=2.5,
        color="#2563eb",
        label=(
            f"Geo/Math Agent (nested-LOOCV Platt): AUROC={math_auc:.3f}, "
            f"cutoff={math_youden['probability_threshold']:.2f}"
        ),
    )

    ax.plot(
        dag_fpr,
        dag_tpr,
        linewidth=2.5,
        color="#dc2626",
        label=f"Full DAG (Nodes 1-3): AUROC={dag_auc:.3f}, cutoff={dag_threshold:.2f}",
    )

    math_point = (float(math_youden["fpr"]), float(math_youden["tpr"]))
    dag_point = (
        1.0 - float(dag_metrics["specificity"]) / 100.0,
        float(dag_metrics["sensitivity"]) / 100.0,
    )

    ax.scatter(
        [math_point[0]],
        [math_point[1]],
        s=90,
        color="#1d4ed8",
        marker="o",
        zorder=5,
        label="Youden point (Geo/Math)",
    )
    ax.scatter(
        [dag_point[0]],
        [dag_point[1]],
        s=105,
        color="#b91c1c",
        marker="D",
        zorder=5,
        label="Youden point (Full DAG)",
    )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random (AUROC=0.5)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC curves: Geo/Math agent vs Full DAG cascade")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    roc_dir = project_root / "results" / "roc"
    roc_dir.mkdir(parents=True, exist_ok=True)

    # 1) Geo/Math: nested-LOOCV Platt probabilities -> empirical ROC + Youden point.
    math_df = load_samples(project_root / "DATASETS_MDPI", project_root / "results" / "pipeline_results_math.json")
    y_math = math_df["ground_truth"].to_numpy().astype(int)
    red = math_df["red_percentage"].to_numpy().astype(float)
    probs, _ = nested_loocv_platt_probabilities(red, y_math)
    math_fpr, math_tpr, math_thr = roc_curve(y_math, probs)
    math_auc = float(auc(math_fpr, math_tpr))
    math_youden = youden_on_roc(math_fpr, math_tpr, math_thr)
    math_metrics = metrics_from_predictions(
        y_math, (probs >= math_youden["probability_threshold"]).astype(int)
    )
    math_youden["accuracy_%"] = float(math_metrics["accuracy"])

    # 2) Full DAG ROC from cascade outputs + confidence.
    dag_df = load_full_dag_scores(project_root / "results" / "pipeline_results.json")
    dag_fpr, dag_tpr, dag_auc = roc_curve_from_arrays(
        dag_df["ground_truth"].to_numpy(),
        dag_df["tumor_score"].to_numpy(),
    )
    dag_metrics, dag_threshold = dag_metrics_at_youden(dag_df)

    # 3) Combined figure + summary.
    fig_path = roc_dir / "roc_curves_article.png"
    plot_combined_roc(
        math_fpr=math_fpr,
        math_tpr=math_tpr,
        math_auc=math_auc,
        math_youden=math_youden,
        dag_fpr=dag_fpr,
        dag_tpr=dag_tpr,
        dag_auc=dag_auc,
        dag_metrics=dag_metrics,
        dag_threshold=dag_threshold,
        output_path=fig_path,
    )

    paper_fig_path = project_root / "AIST_ML_BOOSTING_PAPER_V1" / "roc_curves_article.png"
    shutil.copy2(fig_path, paper_fig_path)

    summary = pd.DataFrame(
        [
            {
                "Method": "Geo/Math Agent (nested-LOOCV Platt, Youden)",
                "AUROC": round(float(math_auc), 3),
                "Youden J": round(float(math_youden["youden_j"]), 3),
                "Threshold (probability)": round(float(math_youden["probability_threshold"]), 3),
                "Sensitivity, %": round(float(math_youden["sensitivity_%"]), 1),
                "Specificity, %": round(float(math_youden["specificity_%"]), 1),
                "Accuracy, %": round(float(math_youden["accuracy_%"]), 1),
                "FPR": round(float(math_youden["fpr"]), 4),
                "TPR": round(float(math_youden["tpr"]), 4),
            },
            {
                "Method": "Full DAG (Nodes 1-3)",
                "AUROC": round(float(dag_auc), 3),
                "Youden J": round(float(dag_metrics["youden_j"]), 3),
                "Threshold (tumor score)": round(float(dag_threshold), 3),
                "Sensitivity, %": round(float(dag_metrics["sensitivity"]), 1),
                "Specificity, %": round(float(dag_metrics["specificity"]), 1),
                "Accuracy, %": round(float(dag_metrics["accuracy"]), 1),
            },
        ]
    )
    summary_path = roc_dir / "roc_article_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"\nSaved figure: {fig_path}")
    print(f"Copied to paper: {paper_fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
