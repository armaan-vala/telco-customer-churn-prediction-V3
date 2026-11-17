# Telco Customer Churn Prediction — V3


## 🔍 Evolution from V1 & V2

- **V1 (Foundations):** Built a baseline ML pipeline comparing Logistic Regression vs. Random Forest, identifying data cleaning, encoding, and class imbalance handling.
- **V2 (Boosters & Explainability):** Integrated gradient boosters (XGBoost, LightGBM, CatBoost), SMOTE oversampling, hyperparameter optimization, and SHAP analysis.
- **V1 & V2 Limitations / What was Fixed:** V1 and V2 suffered from potential data leakage due to global preprocessing/SMOTE before splitting, lacked robust cross-validation discipline, relied on a lazy 0.5 decision threshold, and lacked probability calibration. understanding my mistake i find solution, understand those concept on paper and then developed v3.

---

## What's New in version 3

- **ColumnTransformer:** Separate preprocessing for numerical and categorical features.
- **SMOTE:** Handles class imbalance inside the training pipeline to prevent data leakage 
- **StratifiedKFold:** Maintains class distribution across cross-validation folds.
- **Threshold Tuning:** Optimized classification threshold to 0.1732 instead of using the default 0.5.
- **Probability Calibration:** Improved prediction calibration using calibration techniques.

## Project Performance

| Metric | Result |
|---|---:|
| Mean CV ROC-AUC | 0.8239 ± 0.0104 |
| Test PR-AUC | 0.6071 |
| Classification Threshold | 0.1732 |
| Churn Recall | 0.85 |
| Churn Precision | 0.47 |
| Churn F1-Score | 0.60 |
| Brier Score Before Calibration | 0.1598 |
| Brier Score After Calibration | 0.1493 |

The selected threshold prioritizes identifying customers who are likely to churn. This helps reduce missed churn cases.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
telco-customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── churn_analysis_v3.ipynb
├── requirements.txt
└── README.md
```

## Getting Started

Follow these steps to run the project on your local machine.

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/armaan-vala/telco-customer-churn-prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd telco-customer-churn-prediction
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

If Jupyter Notebook is not installed, install it:

```bash
pip install notebook
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter Notebook in your browser.

### 6. Run the Project

1. Open `churn_analysis_v3.ipynb`.
2. Make sure the dataset is available in the expected `data/` directory.
3. Run the notebook cells sequentially.
4. Review the data analysis, preprocessing, model training, evaluation and results.

## Machine Learning Workflow

1. Data loading and inspection
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Numerical and categorical feature transformation
5. Train-test splitting
6. Handling class imbalance using SMOTE
7. Model training and cross-validation
8. Hyperparameter tuning
9. Classification threshold optimization
10. Probability calibration
11. Final model evaluation

## Key Learning Outcomes

- Building an end-to-end machine learning pipeline.
- Handling imbalanced classification datasets.
- Preventing data leakage during preprocessing and resampling.
- Using cross-validation for reliable model evaluation.
- Understanding the difference between ROC-AUC and PR-AUC.
- Optimizing classification thresholds based on business requirements.
- Evaluating probability calibration using the Brier score.

## Author

**Armaan Vala**

GitHub: [armaan-vala](https://github.com/armaan-vala)

## License

This project is available for educational and learning purposes.