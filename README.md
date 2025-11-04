# 🚗 Road Accident Rate Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-lightgreen?logo=lightgbm)](https://github.com/microsoft/LightGBM)
[![AutoGluon](https://img.shields.io/badge/AutoML-AutoGluon-00BFFF?logo=amazonaws)](https://auto.gluon.ai)

> Forecasting **road accident risk rates** from traffic and environmental data using advanced tabular ML techniques (LightGBM, Random Forest, TabM, and AutoGluon).

---

## 🧭 Overview

This repository contains a collection of experiments designed to predict **road accident rates** based on tabular datasets of environmental and traffic-related variables.  
It began as a **Kaggle-style competition** and evolved into a study of ensemble and residual modelling pipelines.

📊 The workflow explores:
- **Feature engineering & EDA**
- **Tree-based baselines (LightGBM, Random Forest)**
- **Residual tabular stacking**
- **AutoML benchmarking with AutoGluon**

---

## 📁 Repository Structure

| File / Folder | Description |
|:--------------|:------------|
| 📘 `GBM_RF_Road_Accident_Rate.ipynb` | Baseline notebook – EDA, feature engineering, and tree-based models. |
| 📗 `model3_tabm_residual.ipynb` | Residual modelling pipeline using TabM and OOF evaluation. |
| 📙 `autogluon.ipynb` | AutoML benchmark via AutoGluon for rapid leaderboard comparison. |
| 📄 `submission_*.csv` | Kaggle-style final predictions. |
| 📄 `oof_*.csv`, `test_*.csv` | Out-of-fold and test predictions for stacking and ensemble evaluation. |
| ⚙️ `requirements.txt` | (Optional) Python dependencies list. |

> **Note:** Raw data (`train.csv`, `test.csv`) are managed with **[Git LFS](https://git-lfs.github.com/)** due to size.  
> Use `git lfs install && git lfs pull` after cloning.
