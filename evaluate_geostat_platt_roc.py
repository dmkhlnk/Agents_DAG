#!/usr/bin/env python3
"""Nested LOOCV + Platt scaling ROC for Geo/Math (red%).

Mirrors the SVM probability=True idea from Lev's Combined4Rocs script:
- Outer LOOCV: 85 train + 1 hold-out (hold-out never used for fitting A,B).
- Inner CV on the 85: build signed distances to a Youden threshold, fit Platt
  sigmoid P(y=1|d) = 1 / (1 + exp(A*d + B)).
- Apply to hold-out distance -> probability.
- ROC from the 86 out-of-fold probabilities.

Also exports:
- Case C table operating point (hard LOOCV Youden thresholds on red%),
- Binary-LOOCV ROC (degenerate curve that passes exactly through case C).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

from evaluate_roc import (
    find_youden_threshold,
    load_samples,
    loocv_youden_predictions,
    metrics_from_predictions,
)


def signed_distance(red: np.ndarray, threshold: float) -> np.ndarray:
    return red - threshold


def fit_platt_from_oof_distances(
    distances: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """Fit Platt sigmoid on (distance, label) pairs via logistic regression."""
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(distances.reshape(-1, 1), y)
    return lr


def nested_loocv_platt_probabilities(
    red: np.ndarray,
    y: np.ndarray,
    n_inner_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    For each outer LOOCV fold:
      1) On 85 train scans, run inner StratifiedKFold.
      2) On each inner-train, choose Youden threshold on red%.
      3) Inner-val distances = red - thr; collect OOF distances on the 85.
      4) Fit Platt (logistic) on those OOF distances.
      5) On full outer-train 85, choose Youden thr; distance_test = red_holdout - thr.
      6) prob_holdout = Platt(distance_test).
    """
    red = np.asarray(red, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    probs = np.zeros(n, dtype=float)
    rows: list[dict] = []

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        red_tr = red[train_mask]
        y_tr = y[train_mask]

        # Inner CV OOF distances for Platt fit (hold-out i never participates).
        oof_dist = np.full(len(y_tr), np.nan)
        skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
        for inner_tr, inner_va in skf.split(red_tr.reshape(-1, 1), y_tr):
            thr_inner = find_youden_threshold(y_tr[inner_tr], red_tr[inner_tr])
            oof_dist[inner_va] = signed_distance(red_tr[inner_va], thr_inner)

        if np.isnan(oof_dist).any():
            # Fallback: use distances to Youden thr on all train 85.
            thr_fallback = find_youden_threshold(y_tr, red_tr)
            oof_dist = signed_distance(red_tr, thr_fallback)

        platt = fit_platt_from_oof_distances(oof_dist, y_tr)

        thr_outer = find_youden_threshold(y_tr, red_tr)
        d_test = float(red[i] - thr_outer)
        prob = float(platt.predict_proba(np.array([[d_test]]))[0, 1])
        probs[i] = prob

        # sklearn stores P = 1/(1+exp(-(coef*x+intercept))); Platt form uses A=-coef, B=-intercept
        a = -float(platt.coef_[0, 0])
        b = -float(platt.intercept_[0])

        rows.append(
            {
                "filename": None,  # filled later if available
                "index": i,
                "ground_truth": int(y[i]),
                "red_percentage": float(red[i]),
                "fold_youden_threshold_red": float(thr_outer),
                "signed_distance": d_test,
                "platt_A": a,
                "platt_B": b,
                "tumor_probability": prob,
                "hard_pred_youden_red": int(red[i] >= thr_outer),
            }
        )

    return probs, pd.DataFrame(rows)


def youden_on_roc(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> dict:
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {
        "fpr": float(fpr[idx]),
        "tpr": float(tpr[idx]),
        "youden_j": float(j[idx]),
        "sensitivity_%": 100.0 * float(tpr[idx]),
        "specificity_%": 100.0 * (1.0 - float(fpr[idx])),
        "probability_threshold": float(thresholds[idx]),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / "results" / "roc" / "geostat_platt"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_samples(
        project_root / "DATASETS_MDPI",
        project_root / "results" / "pipeline_results_math.json",
    )
    y = df["ground_truth"].to_numpy().astype(int)
    red = df["red_percentage"].to_numpy().astype(float)

    # --- Platt nested LOOCV probabilities ---
    probs, detail = nested_loocv_platt_probabilities(red, y)
    detail["filename"] = df["filename"].to_numpy()
    detail.to_csv(out_dir / "geostat_platt_loocv_scores.csv", index=False, encoding="utf-8-sig")

    fpr, tpr, thr = roc_curve(y, probs)
    roc_auc = float(auc(fpr, tpr))
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        out_dir / "geostat_platt_roc_points.csv", index=False, encoding="utf-8-sig"
    )

    youden_platt = youden_on_roc(fpr, tpr, thr)
    youden_platt.update({"auc": roc_auc, "n": int(len(df)), "variant": "platt_nested_loocv"})
    # Metrics at that probability threshold
    m_platt = metrics_from_predictions(y, (probs >= youden_platt["probability_threshold"]).astype(int))
    youden_platt["accuracy_%"] = float(m_platt["accuracy"])
    pd.DataFrame([youden_platt]).to_csv(
        out_dir / "geostat_platt_youden_point.csv", index=False, encoding="utf-8-sig"
    )

    # --- Case C: hard LOOCV operating point (table) ---
    pred_c = loocv_youden_predictions(df)
    m_c = metrics_from_predictions(y, pred_c["prediction"].to_numpy())
    case_c = {
        "variant": "C_hard_loocv_table_operating_point",
        "fpr": 1.0 - m_c["specificity"] / 100.0,
        "tpr": m_c["sensitivity"] / 100.0,
        "youden_j": float(m_c["youden_j"]),
        "sensitivity_%": float(m_c["sensitivity"]),
        "specificity_%": float(m_c["specificity"]),
        "accuracy_%": float(m_c["accuracy"]),
        "cutoff_red%_median": float(pred_c["fold_threshold"].median()),
        "n": int(len(df)),
        "note": "Hard per-fold Youden on red%; not argmax(J) on Platt ROC",
    }
    pd.DataFrame([case_c]).to_csv(
        out_dir / "geostat_caseC_table_operating_point.csv", index=False, encoding="utf-8-sig"
    )

    # Degenerate ROC from binary LOOCV predictions (exactly through case C)
    fpr_b, tpr_b, thr_b = roc_curve(y, pred_c["prediction"].to_numpy().astype(float))
    pd.DataFrame({"fpr": fpr_b, "tpr": tpr_b, "threshold": thr_b}).to_csv(
        out_dir / "geostat_caseC_binary_roc_points.csv", index=False, encoding="utf-8-sig"
    )

    # Copy main Platt CSVs into names Lev's plot script expects (optional drop-in)
    shutil.copy2(out_dir / "geostat_platt_roc_points.csv", project_root / "results" / "roc" / "roc_math_loocv_points.csv")
    # Also write tpr_mean alias for Lev's Combined script compatibility
    pd.DataFrame({"fpr": fpr, "tpr_mean": tpr, "tpr_std": np.zeros_like(tpr)}).to_csv(
        project_root / "results" / "roc" / "roc_math_loocv_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "fpr": youden_platt["fpr"],
                "tpr": youden_platt["tpr"],
                "youden_j": youden_platt["youden_j"],
                "sensitivity_%": youden_platt["sensitivity_%"],
                "specificity_%": youden_platt["specificity_%"],
                "accuracy_%": youden_platt["accuracy_%"],
                "auc_mean": roc_auc,
                "n": len(df),
                "note": "Youden on nested-LOOCV Platt ROC",
            }
        ]
    ).to_csv(project_root / "results" / "roc" / "roc_math_youden_point.csv", index=False, encoding="utf-8-sig")

    # --- Figure: Platt ROC + case C marker + Platt Youden marker ---
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Geo/Math Platt LOOCV ROC (AUROC={roc_auc:.3f})")
    ax.plot(fpr_b, tpr_b, color="#93c5fd", lw=1.8, ls="--", label="Binary LOOCV ROC (passes through table OP C)")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)

    ax.scatter(
        [youden_platt["fpr"]],
        [youden_platt["tpr"]],
        s=90,
        c="#1d4ed8",
        zorder=5,
        label=(
            f"Youden on Platt ROC "
            f"(Sens={youden_platt['sensitivity_%']:.1f}, Spec={youden_platt['specificity_%']:.1f})"
        ),
    )
    ax.scatter(
        [case_c["fpr"]],
        [case_c["tpr"]],
        s=120,
        c="#16a34a",
        marker="X",
        zorder=6,
        label=(
            f"Table OP C / hard LOOCV "
            f"(Sens={case_c['sensitivity_%']:.1f}, Spec={case_c['specificity_%']:.1f})"
        ),
    )

    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Geo/Math: nested LOOCV Platt ROC vs table operating point C")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig_path = out_dir / "geostat_platt_roc.png"
    fig.savefig(fig_path, dpi=180)
    shutil.copy2(fig_path, project_root / "AIST_ML_BOOSTING_PAPER_V1" / "geostat_platt_roc.png")
    plt.close(fig)

    readme = out_dir / "README.txt"
    readme.write_text(
        f"""Geo/Math nested LOOCV + Platt ROC
=================================

AUROC (Platt probabilities): {roc_auc:.3f}
Youden on Platt ROC: Sens={youden_platt['sensitivity_%']:.1f}%, Spec={youden_platt['specificity_%']:.1f}%, J={youden_platt['youden_j']:.3f}
Table OP C (hard LOOCV): Sens={case_c['sensitivity_%']:.1f}%, Spec={case_c['specificity_%']:.1f}%, Acc={case_c['accuracy_%']:.1f}%

Important:
- Hold-out scan is never used to fit Platt A,B (outer LOOCV).
- Inner 5-fold CV on the 85 trains builds OOF signed distances for sigmoid fit.
- Case C is the paper-table operating point; it lies exactly on the binary-LOOCV ROC,
  and near (but not necessarily equal to) a non-Youden point of the Platt ROC.
""",
        encoding="utf-8",
    )

    print(readme.read_text(encoding="utf-8"))
    print(f"Saved under {out_dir}")


if __name__ == "__main__":
    main()
