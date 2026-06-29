#!/usr/bin/env python3
"""LOOCV-based ROC curve and Youden evaluation for Math Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.math_agent import MathAgent


def load_samples(dataset_dir: Path, math_json: Path | None) -> pd.DataFrame:
    if math_json and math_json.exists():
        rows = []
        for item in json.loads(math_json.read_text(encoding="utf-8")):
            class_name = item.get("ground_truth_class")
            if class_name not in {"Class1", "Class2"}:
                continue
            stats = item["math_stats"]
            rows.append(
                {
                    "filename": Path(item["image_path"]).name,
                    "ground_truth": 1 if class_name == "Class2" else 0,
                    "red_percentage": stats["red_percentage"],
                    "clustering_ratio": stats["clustering_ratio"],
                }
            )
        return pd.DataFrame(rows)

    agent = MathAgent()
    rows = []
    for image_path in sorted(dataset_dir.rglob("*.jpg")):
        class_name = image_path.parent.name
        if class_name not in {"Class1", "Class2"}:
            continue
        stats = agent.analyze(image_path)
        rows.append(
            {
                "filename": image_path.name,
                "ground_truth": 1 if class_name == "Class2" else 0,
                "red_percentage": stats.red_percentage,
                "clustering_ratio": stats.clustering_ratio,
            }
        )
    return pd.DataFrame(rows)


def roc_curve_from_arrays(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y_sorted = y_true[order]

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tp / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / n_neg, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def find_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_j = -1.0
    best_threshold = float(scores.min())

    for threshold in np.unique(scores):
        predicted = scores >= threshold
        tp = (predicted & (y_true == 1)).sum()
        fn = ((~predicted) & (y_true == 1)).sum()
        fp = (predicted & (y_true == 0)).sum()
        tn = ((~predicted) & (y_true == 0)).sum()
        if tp + fn == 0 or tn + fp == 0:
            continue
        j = tp / (tp + fn) + tn / (tn + fp) - 1.0
        if j > best_j:
            best_j = j
            best_threshold = float(threshold)

    return best_threshold


def loocv_roc_curve(df: pd.DataFrame, n_points: int = 101) -> dict:
    """
    For each LOOCV fold, build ROC on the training set (n-1 scans),
    then average TPR across folds at fixed FPR grid.
    """
    y_all = df["ground_truth"].to_numpy()
    red_all = df["red_percentage"].to_numpy()
    grid_fpr = np.linspace(0.0, 1.0, n_points)

    fold_tprs: list[np.ndarray] = []
    fold_aucs: list[float] = []

    for i in range(len(df)):
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[i] = False

        fpr, tpr, auc = roc_curve_from_arrays(y_all[train_mask], red_all[train_mask])
        interp_tpr = np.interp(grid_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        fold_tprs.append(interp_tpr)
        fold_aucs.append(auc)

    mean_tpr = np.mean(fold_tprs, axis=0)
    std_tpr = np.std(fold_tprs, axis=0)
    mean_auc = float(np.mean(fold_aucs))

    return {
        "fpr": grid_fpr,
        "tpr_mean": mean_tpr,
        "tpr_std": std_tpr,
        "auc_mean": mean_auc,
        "auc_std": float(np.std(fold_aucs)),
        "fold_aucs": fold_aucs,
    }


def loocv_youden_predictions(df: pd.DataFrame) -> pd.DataFrame:
    y_all = df["ground_truth"].to_numpy()
    red_all = df["red_percentage"].to_numpy()
    rows: list[dict] = []

    for i in range(len(df)):
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[i] = False

        threshold = find_youden_threshold(y_all[train_mask], red_all[train_mask])
        prediction = int(red_all[i] >= threshold)

        rows.append(
            {
                "filename": df.iloc[i]["filename"],
                "ground_truth": int(y_all[i]),
                "prediction": prediction,
                "red_percentage": float(red_all[i]),
                "fold_threshold": threshold,
                "correct": int(prediction == y_all[i]),
            }
        )

    return pd.DataFrame(rows)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int((y_pred.astype(bool) & (y_true == 1)).sum())
    fn = int((~y_pred.astype(bool) & (y_true == 1)).sum())
    fp = int((y_pred.astype(bool) & (y_true == 0)).sum())
    tn = int((~y_pred.astype(bool) & (y_true == 0)).sum())

    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    specificity = 100.0 * tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = 100.0 * (tp + tn) / len(y_true)
    youden_j = (sensitivity + specificity) / 100.0 - 1.0

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "youden_j": youden_j,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def plot_loocv_roc(
    roc: dict,
    youden_metrics: dict,
    output_path: Path,
    n_samples: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    fpr = roc["fpr"]
    tpr = roc["tpr_mean"]
    std = roc["tpr_std"]

    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        color="#2563eb",
        label=f"LOOCV mean ROC (AUC={roc['auc_mean']:.3f} ± {roc['auc_std']:.3f})",
    )
    ax.fill_between(
        fpr,
        np.clip(tpr - std, 0, 1),
        np.clip(tpr + std, 0, 1),
        color="#2563eb",
        alpha=0.15,
        label="±1 std across folds",
    )

    op_fpr = 1.0 - youden_metrics["specificity"] / 100.0
    op_tpr = youden_metrics["sensitivity"] / 100.0
    ax.scatter(
        [op_fpr],
        [op_tpr],
        s=100,
        color="#16a34a",
        zorder=5,
        label=(
            f"LOOCV Youden: Sens={youden_metrics['sensitivity']:.1f}%, "
            f"Spec={youden_metrics['specificity']:.1f}%, "
            f"Acc={youden_metrics['accuracy']:.1f}%"
        ),
    )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(
        f"ROC — Math Agent (LOOCV, n={n_samples})\n"
        "Each fold: ROC on 85 train scans; curve = mean over 86 folds"
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LOOCV ROC for Math Agent")
    parser.add_argument("dataset", nargs="?", default="DATASETS_MDPI")
    parser.add_argument("--math-json", default="results/pipeline_results_math.json")
    parser.add_argument("--output-dir", default="results/roc")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_samples(Path(args.dataset), Path(args.math_json) if args.math_json else None)
    y = df["ground_truth"].to_numpy()

    roc = loocv_roc_curve(df)
    predictions = loocv_youden_predictions(df)
    youden_metrics = metrics_from_predictions(y, predictions["prediction"].to_numpy())

    plot_loocv_roc(roc, youden_metrics, output_dir / "math_agent_roc.png", n_samples=len(df))

    predictions.to_csv(output_dir / "loocv_predictions.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "Метод": "Math Agent — red% (LOOCV)",
                "Порог": f"median fold threshold = {predictions['fold_threshold'].median():.2f}%",
                "AUC (mean ± std)": f"{roc['auc_mean']:.3f} ± {roc['auc_std']:.3f}",
                "Youden J": round(youden_metrics["youden_j"], 3),
                "Sensitivity, %": round(youden_metrics["sensitivity"], 1),
                "Specificity, %": round(youden_metrics["specificity"], 1),
                "Accuracy, %": round(youden_metrics["accuracy"], 1),
                "TP": youden_metrics["tp"],
                "TN": youden_metrics["tn"],
                "FP": youden_metrics["fp"],
                "FN": youden_metrics["fn"],
            }
        ]
    )
    summary.to_csv(output_dir / "loocv_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Samples: {len(df)} (Tumor={int(y.sum())}, Normal={int((y==0).sum())})")
    print(f"LOOCV mean AUC: {roc['auc_mean']:.3f} ± {roc['auc_std']:.3f}")
    print(f"LOOCV Youden threshold (median): {predictions['fold_threshold'].median():.2f}%")
    print()
    print(summary.to_string(index=False))
    print(f"\nSaved:")
    print(f"  {output_dir / 'math_agent_roc.png'}")
    print(f"  {output_dir / 'loocv_summary.csv'}")
    print(f"  {output_dir / 'loocv_predictions.csv'}")


if __name__ == "__main__":
    main()
