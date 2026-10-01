# Replication evaluation

## Result

The complete analysis was run twice from the same code in the same Python environment. Both runs produced byte-identical `outputs/metrics.json` files. The train/test rows, confusion matrix, and reported metrics matched exactly.

The audited environment was Python from the workspace runtime with scikit-learn 1.9.1, NumPy 1.26.4, pandas 3.0.6, and Matplotlib 3.11.2. The code fixes the split seed (`42`), stratifies by outcome, places scaling inside the training pipeline, sets the decision threshold (`0.50`), and records the class meaning. This supports reproducibility for this run; it does not establish that the estimate is stable across other samples.

## Observed held-out results

The 80/20 stratified split held out 114 samples (42 malignant and 72 benign). At the teaching threshold of 0.50, the confusion matrix was:

| Actual class | Predicted benign | Predicted malignant |
| --- | ---: | ---: |
| Benign | 71 | 1 |
| Malignant | 3 | 39 |

That corresponds to 92.9% sensitivity, 98.6% specificity, 97.5% precision, 96.5% accuracy, and 0.996 ROC AUC on this single test split. These values describe the tutorial's held-out sample only.

## What makes this tutorial easier to reproduce

- It uses a dataset packaged with scikit-learn, so learners do not need gated MIMIC access, credentials, SQL, or a manual download.
- Installation commands and broad dependency version ranges are provided.
- The seed, target encoding, split ratio, threshold, and metric definitions are visible in the code.
- Assertions check the expected data shape, malignant count, and missingness assumption.
- The script writes machine-readable metrics and predictions as well as labeled plots.
- The notebook explains the sequence and calls the same canonical script, which reduces duplicate code that could drift.

## What still limits replication

- Broad version ranges improve install flexibility, but they do not lock every transitive dependency. Package versions can affect numerical results or plotting details.
- A fixed random seed reproduces one partition; another seed can produce different estimates.
- The dataset is small and historical. Reproducing its benchmark result is not the same as validating a model in another hospital, population, or workflow.
- The notebook expects the whole repository to be available in the current environment. Uploading only the notebook to Colab will not include `src/analyze.py` or `requirements.txt`.

## Classroom replication checklist

1. Clone or download the complete repository.
2. Install `requirements.txt` in a clean environment.
3. Run `python src/analyze.py` twice and compare `outputs/metrics.json`.
4. Change the seed and observe that held-out metrics can change.
5. Move `StandardScaler` outside the pipeline as a deliberate demonstration of leakage, then restore the safe version.
6. Explain why a benchmark split and high AUC do not prove clinical usefulness.

This audit is an educational reproducibility check, not an independent scientific replication or clinical validation.
