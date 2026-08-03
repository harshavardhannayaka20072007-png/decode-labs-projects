```markdown
# Task 1: Data Processing, Feature Engineering, and Validation Pipeline
## 📌 Overview
This project represents **Task 1** of the Data Science Internship at **Decode Labs**. The objective of this task was to build a robust, production-grade data processing and feature engineering pipeline using Python. The script automates data cleaning, outlier treatment, feature extraction, multicollinearity eradication, and rigorous schema validation.

---

## 📂 Folder Structure
```text
decodelab-stuffs/
│
└── task1/
    ├── dataset.csv                 # Raw dataset input
    ├── task1_pipeline.py           # Main automation script
    ├── cleaned_dataset.csv         # Processed dataset (CSV format)
    ├── cleaned_dataset.parquet     # Processed dataset (Parquet format)
    └── task1_visualization.png     # Generated statistical graphs

```
## 🛠️ Pipeline Architecture & Steps

1. **Smart Data Cleaning & Fidelity (Module 1):**
* Drops rows with minor missingness ($<5\%$).
* Imputes medium-missingness columns ($5\%-20\%$) using global medians (for numerical features) and modes (for categorical features).
* Applies multi-dimensional **KNN Imputation** ($n=5$) for heavy missingness ($>20\%$).
* Winsorizes extreme outliers using Interquartile Range (IQR) boundaries.
  
2. **Process Engine & Feature Transformation (Module 2):**
* Engineers powerful derived features: `Effective_UnitPrice`, `Is_Bulk_Order`, and `Log_TotalPrice`.
* Performs One-Hot Encoding on categorical features.
* Programmatically eradicates multicollinearity by dropping redundant features exceeding an absolute correlation threshold of $0.80$.

3. **Validation & Serving Framework (Module 3):**
* Enforces strict schema contracts using **Pandera** with lazy evaluation.
* Timestamps the dataset with event metadata and exports clean outputs in both **CSV** and **Parquet** formats.
---
## 🚀 How to Run the Script

1. Open your terminal inside the **`task1`** folder in VS Code.
2. Execute the pipeline script:
```bash
python task1_pipeline.py
```



```

```
