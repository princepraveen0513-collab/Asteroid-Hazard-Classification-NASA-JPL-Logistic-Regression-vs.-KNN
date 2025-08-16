# Asteroid Hazard Classification (NASA JPL) — Logistic Regression vs. KNN

This project builds and compares two binary classifiers — **Logistic Regression** and **K-Nearest Neighbors (KNN)** — to predict whether an asteroid is **Potentially Hazardous (PHA)** using the **NASA JPL Asteroid dataset** (sourced from Kaggle). The notebook performs basic EDA, data cleaning, model training with cross-validated hyperparameter search, and evaluation using confusion matrices and standard classification metrics.

> **Why this matters:** In planetary defense, *false negatives* (predicting “safe” when it’s hazardous) are far more costly than *false positives*. The work here emphasizes recall-oriented evaluation and sets up a baseline for safety-first modeling.

---

## 📁 Repository Structure

```
.
├── AsteroidHazardClassification.ipynb   # Main analysis notebook
├── dataset.csv                       # Local copy of the dataset (not committed by default)
└── README.md                         # This file
```



---

## 🧠 Problem Statement

Given orbital and photometric features for near-earth objects, predict whether an object is **PHA** (`pha ∈ {0,1}`). Because hazardous objects are rare compared to non-hazardous ones, the dataset is **highly imbalanced**. As such, accuracy alone is misleading; **recall** and **F1** are prioritized.

---

## 📊 Dataset

- **Source:** *NASA JPL Asteroid Dataset – Kaggle* (add the exact Kaggle link you used here).
- **Target:** `pha` (1 = potentially hazardous, 0 = not hazardous)
- **Selected Features (from the notebook):**
  - `H` (absolute magnitude)
  - `moid` (minimum orbital intersection distance)
  - `a` (semi-major axis)
  - `q` (perihelion distance)
  - `i` (inclination)
  - `sigma_e` (uncertainty in eccentricity)
- **Cleaning & Imputation (as done in the notebook):**
  - Dropped rows with null `pha`.
  - For columns with missing values (e.g., `H`, `sigma_*`), mean imputation was applied.
  - Train/test split with `random_state=42` (80/20).


---

## 🛠️ Environment & Setup

**Python:** 3.9+ recommended

**Install dependencies:**

```bash
pip install -U numpy pandas scikit-learn matplotlib seaborn
```

**Run the analysis:**

```bash
jupyter notebook "Assignment W05_Prince (3).ipynb"
```

Place the dataset as `./dataset.csv` relative to the notebook or update the path in the loading cell.

---

## 🔧 Modeling Approach

- **Models:** Logistic Regression, KNN Classifier
- **Hyperparameter Tuning:** `GridSearchCV` (5-fold CV) optimizing **F1 score**
  - Logistic Regression: `C ∈ {0.01, 0.1, 1, 10, 100}`
  - KNN: `n_neighbors ∈ {1…20}`
- **Metrics Reported:** Accuracy, Precision, Recall, F1
- **Visuals:** Confusion matrices plotted for both models

---

## ✅ Results (from the executed notebook)

| Model                 | Accuracy | Precision | Recall | F1   |
|-----------------------|---------:|----------:|-------:|-----:|
| Logistic Regression   | **0.9990** | **0.8248**  | **0.7183** | **0.7679** |
| KNN Classifier        | 0.9975 | 0.4504   | 0.4155 | 0.4322 |

**Confusion Matrix Snapshot (Logistic Regression):**
- **TN:** 187,230
- **TP:** 306
- **FP:** 65
- **FN:** 120

> With such class imbalance, high accuracy is expected. The more useful signals here are **Recall** (to limit false negatives) and **Precision** (to limit false alarms). LR outperforms KNN on all metrics in this baseline.

---

## 🧭 Interpretation & Takeaways

- **Safety-first objective:** Prioritize **Recall** to reduce **False Negatives** (missing a hazardous object).
- **Baseline insight:** Logistic Regression provides a stronger baseline than unscaled KNN on this feature set.
- **Imbalance-aware training:** Standard accuracy is inflated by the overwhelming majority class (non-PHA). Prefer **Recall**, **F1**, **PR-AUC**, and calibrated thresholds.

---

## 🔄 Reproducibility

- Deterministic split via `random_state=42`.
- Hyperparameters selected via 5-fold cross-validation optimizing **F1**.

---

## 🚀 Suggested Next Steps

1. **Use Scaled Features** in modeling cells (especially for KNN and distance-based models).  
2. **Class Imbalance Strategies:** try `class_weight='balanced'` for Logistic Regression, or re-sampling (SMOTE, undersampling) and **threshold tuning** based on precision–recall trade-offs.  
3. **Model Extensions:** Evaluate tree-based models (Random Forests, XGBoost/LightGBM) and linear SVM with class weights.  
4. **Metrics:** Add **Precision–Recall curves** and **PR-AUC**; these are more informative than ROC curves under heavy imbalance.  
5. **Feature Expansion:** Consider more orbital features and engineered interactions; assess multicollinearity and feature importance.  
6. **Calibration:** Plot calibration curves and adjust decision thresholds to maximize **Recall** at acceptable **Precision**.  
7. **Validation:** Use **stratified** splits and possibly time-based validation if there’s temporal drift in discovery dates.

---

## 📦 How to Use This Repo

1. Clone the repo and set up the environment.
2. Put `dataset.csv` in the project root (or update the path in the notebook).
3. Open and run `AsteroidHazardClassification.ipynb` cell-by-cell.
4. Inspect the confusion matrices and metric printouts; experiment with the “Suggested Next Steps.”

---

## 📝 Notes & Limitations

- The dataset is **heavily imbalanced**, making accuracy less meaningful.
- The current notebook scales features but does **not** feed scaled arrays to the model cells (KNN especially benefits from scaling).
- No external validation (e.g., hold-out timeframe) is used; consider adding if temporal drift matters.

---

## 📚 References

- NASA/JPL Small-Body Database (via Kaggle). *(Insert the exact Kaggle dataset URL you used.)*
- Scikit-learn documentation on:
  - [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
  - [KNN](https://scikit-learn.org/stable/modules/neighbors.html#classification)
  - [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)


---

## 🙌 Acknowledgements

Thanks to NASA/JPL for the underlying data and the open-source community behind NumPy, pandas, scikit-learn, Matplotlib, and Seaborn.
