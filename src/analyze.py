"""Teach a complete, leakage-safe binary classification workflow.

Run from the project directory with `python src/analyze.py`. The dataset is
loaded from scikit-learn, so no private credentials or manual downloads are
needed. Comments explain why each step is present, not only what it does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Use a file-rendering backend so plots work in headless terminals and hosted
# notebooks. Keep Matplotlib's cache inside the project rather than requiring
# write access to a user's home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Make the random split reproducible. Reproducible does not mean universally
# representative: it only means this exact split can be regenerated.
RANDOM_SEED = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50


def main() -> None:
    """Load data, fit the model, evaluate it, and save teaching artifacts."""
    # Anchor paths to this file, not the shell's current directory. This lets
    # a student run the script from the project root or another directory.
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # `as_frame=True` gives us named columns, which makes the later coefficient
    # table much easier to read than a matrix of anonymous column numbers.
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data

    # In the scikit-learn copy, target 0 means malignant and target 1 means
    # benign. We recode explicitly so that "positive" consistently means
    # malignant. This decision controls what sensitivity and recall mean.
    malignant_code = list(dataset.target_names).index("malignant")
    y = (dataset.target == malignant_code).astype(int)

    # Keep a record of the starting data shape and labels. These checks catch
    # common mistakes such as accidentally swapping the positive class.
    assert X.shape == (569, 30), f"Unexpected data shape: {X.shape}"
    assert int(y.sum()) == 212, "Positive label should represent 212 malignant cases."
    assert not X.isna().any().any(), "This tutorial assumes no missing feature values."

    # Split before learning any preprocessing parameters. Stratification keeps
    # roughly the same benign/malignant proportions in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # StandardScaler learns means and standard deviations. Placing it inside
    # the Pipeline ensures it learns from training rows only; scaling the full
    # dataset before the split would leak information from the test set.
    # Logistic regression is a useful teaching model: it is fast, and its
    # standardized coefficients can be inspected (with careful caveats).
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5_000,
                    class_weight=None,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    # Fit once on training rows. The held-out test rows remain untouched until
    # the final evaluation below.
    model.fit(X_train, y_train)

    # Score each test sample with a probability, then apply a clearly stated
    # threshold to turn that probability into a binary prediction.
    malignant_probability = model.predict_proba(X_test)[:, 1]
    y_pred = (malignant_probability >= DECISION_THRESHOLD).astype(int)

    # In this tutorial malignant is positive: recall is sensitivity, and
    # specificity measures the fraction of benign samples predicted benign.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = tn / (tn + fp)
    metrics = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn copy)",
        "positive_class": "malignant",
        "samples_total": int(len(y)),
        "features": int(X.shape[1]),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_malignant": int(y_test.sum()),
        "test_benign": int((y_test == 0).sum()),
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "decision_threshold": DECISION_THRESHOLD,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision_positive_predictive_value": float(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, malignant_probability)),
        "confusion_matrix_rows_true_0_benign_1_malignant_cols_predicted_0_1": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }

    # Save the scores as well as the thresholded predictions. Scores let a
    # learner explore other thresholds without fitting the model again.
    predictions = X_test.copy()
    predictions.insert(0, "true_malignant", y_test)
    predictions["malignant_probability"] = malignant_probability
    predictions["predicted_malignant_at_0_50"] = y_pred
    predictions.to_csv(output_dir / "predictions.csv", index_label="sample_index")

    # Logistic-regression coefficients are expressed on standardized feature
    # units because the scaler is part of the fitted pipeline. They are useful
    # for teaching model interpretation, but are not causal effects.
    coefficient_values = model.named_steps["classifier"].coef_[0]
    coefficients = pd.DataFrame(
        {"feature": X.columns, "standardized_coefficient": coefficient_values}
    ).assign(absolute_coefficient=lambda frame: frame.standardized_coefficient.abs())
    coefficients.sort_values(
        "absolute_coefficient", ascending=False
    ).to_csv(output_dir / "coefficients.csv", index=False)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    # Produce a labeled confusion matrix. Rows are truth and columns are model
    # predictions, so each error type is visible rather than hidden in accuracy.
    matrix_display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["Benign", "Malignant"],
        cmap="Blues",
        colorbar=False,
    )
    matrix_display.ax_.set_title("Held-out test set: confusion matrix")
    matrix_display.figure_.tight_layout()
    matrix_display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(matrix_display.figure_)

    # ROC curves show the trade-off between sensitivity and false-positive
    # rate across thresholds. AUC summarizes ranking performance, not clinical
    # usefulness, calibration, or an appropriate operating threshold.
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test, malignant_probability, pos_label=1
    )
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#245B78",
        linewidth=2.5,
        label=f"Logistic regression (AUC = {metrics['roc_auc']:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance ranking")
    ax.set(
        xlabel="False-positive rate (1 − specificity)",
        ylabel="Sensitivity (recall)",
        title="ROC curve on the held-out test set",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)

    # Print a concise summary so a learner can check the run without opening
    # the JSON file. More complete artifacts remain available in outputs/.
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
