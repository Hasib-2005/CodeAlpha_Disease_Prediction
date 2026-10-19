# Disease Prediction from Medical Data (CodeAlpha Machine Learning Internship - Task 4)

## Overview
This repository contains the implementation of a **Disease Prediction System** developed as part of the Machine Learning Internship at **CodeAlpha**[cite: 5, 6]. The primary objective is to accurately predict the presence of disease (specifically Breast Cancer) using diagnostic measurements and clinical attributes through supervised classification algorithms[cite: 1, 6].

---

## Dataset
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (from Scikit-Learn / UCI ML Repository)[cite: 1, 6]
- **Target Variable:** `target` (Malignant vs. Benign)
- **Features (30 continuous numerical features):**
  - Radius, Texture, Perimeter, Area, Smoothness
  - Compactness, Concavity, Concave points, Symmetry, Fractal dimension
  - Computed mean, standard error, and "worst" values for each cell nucleus characteristic

---

## Workflow & Implementation

1. **Data Preprocessing:**
   - Loaded structured diagnostic data directly via `sklearn.datasets.load_breast_cancer`[cite: 1, 6].
   - Inspected feature distributions and verified there are no missing values.
   - Performed train-test split (80% training, 20% testing) with stratification to preserve class balance.
   - Applied `StandardScaler` to normalize feature magnitudes across all continuous clinical variables.

2. **Model Training:**
   - **Logistic Regression:** Serves as a fast, highly interpretable baseline classifier[cite: 1, 6].
   - **Random Forest Classifier:** An ensemble bagging model capturing non-linear relationships and feature interactions[cite: 1, 6].
   - **Support Vector Machine (SVM):** Utilized with probability estimation to find optimal decision boundaries in high-dimensional feature space[cite: 1, 6].

3. **Evaluation Metrics:**
   - Evaluated models using **Precision, Recall, F1-Score, and Support** to carefully observe diagnostic sensitivity[cite: 1, 6].
   - Plotted **ROC-AUC Curves** to compare trade-offs between true positive and false positive rates[cite: 1, 6].
   - Visualized the **Confusion Matrix** to ensure minimized false negative predictions in clinical screening.

---

## Tech Stack & Libraries
- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## How to Run
1. Open the `.ipynb` notebook file directly in Google Colab or locally in Jupyter Notebook:
   ```bash
   jupyter notebook Task4_Disease_Prediction.ipynb
