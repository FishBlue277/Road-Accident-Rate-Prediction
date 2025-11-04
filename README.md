diff --git a/README.md b/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..5944adc9181d7dafd10e24e0787e82d90fe620ac
--- /dev/null
+++ b/README.md
@@ -0,0 +1,42 @@
+# Road Accident Rate Prediction
+
+This repository contains a collection of experiments for forecasting road accident rates from tabular traffic and environmental data. The work started from a Kaggle-style competition dataset (see `train.csv` and `test.csv`, tracked with Git LFS because of their size) and explores several modelling strategies ranging from gradient boosting to AutoML.
+
+## Repository structure
+
+- `GBM_RF_Road_Accident_Rate.ipynb` – Feature engineering, exploratory analysis, and tree-based baselines (LightGBM and Random Forest).
+- `model3_tabm_residual.ipynb` – Residual modelling pipeline that refines the baseline with additional tabular models and out-of-fold evaluation.
+- `autogluon.ipynb` – AutoGluon AutoML sweep for rapid benchmarking.
+- `submission_*.csv` – Generated competition submissions.
+- `oof_*.csv`, `test_*.csv` – Out-of-fold predictions and test-set forecasts for ensembling.
+
+> **Note**: The raw data files are referenced via [Git Large File Storage](https://git-lfs.github.com/). Clone with `git lfs install` followed by `git lfs pull` to obtain them.
+
+## Getting started
+
+1. Create a Python environment (e.g. with `conda` or `venv`).
+2. Install the notebook dependencies. The notebooks rely on common data-science libraries such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`, and `autogluon.tabular`. Install them as needed:
+   ```bash
+   pip install -r requirements.txt  # if you maintain one
+   # or install packages manually
+   pip install pandas numpy scikit-learn matplotlib seaborn lightgbm autogluon.tabular
+   ```
+3. Launch Jupyter Lab/Notebook to reproduce the experiments:
+   ```bash
+   jupyter lab
+   ```
+4. Open any of the notebooks to inspect the workflow, adjust feature engineering, or rerun training.
+
+## Reproducing results
+
+Each notebook is self-contained:
+
+- Run all cells in `GBM_RF_Road_Accident_Rate.ipynb` to generate baseline models and feature transformations (including target encoding and residual learning variants).
+- Use `model3_tabm_residual.ipynb` to train the residual tabular model and export predictions used in the ensemble submissions.
+- Evaluate automated baselines with `autogluon.ipynb` to compare leaderboard-ready performance with minimal tuning.
+
+Intermediate predictions are saved to the repository (`oof_*.csv`, `test_*.csv`, `submission_*.csv`) so you can blend or audit the models without re-running everything from scratch.
+
+## Contributing
+
+Feel free to open issues or pull requests to share improvements, add new modelling notebooks, or document findings. When adding large artefacts (e.g. new datasets or model weights) remember to track them with Git LFS.
