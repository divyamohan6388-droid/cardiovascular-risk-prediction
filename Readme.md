# Reliable and Explainable Machine Learning for Cardiovascular Disease Classification

An undergraduate research-oriented machine learning project exploring cardiovascular disease classification using statistical analysis, Logistic Regression, model evaluation, and model interpretability.

## Overview

Machine learning models can identify patterns in medical datasets, but reliable modelling requires more than achieving high predictive performance. Understanding data quality, preventing data leakage, evaluating model stability, and interpreting model behaviour are equally important.

This project investigates the application of interpretable machine learning to cardiovascular disease classification using the UCI Heart Disease dataset.

The primary objective is to develop a reproducible machine learning pipeline, evaluate its predictive performance, and examine how different patient characteristics are associated with the model's predictions.

**Project status:** Core modelling, evaluation, and interpretation experiments completed. Further validation and analysis remain future work.

## Research Objectives

- Explore the dataset and examine relationships between clinical attributes and recorded disease status.
- Investigate missing values, feature distributions, and class balance.
- Develop a leakage-aware preprocessing and classification pipeline.
- Evaluate Logistic Regression using stratified cross-validation.
- Compare Logistic Regression with Random Forest as a baseline alternative.
- Investigate model selection through hyperparameter tuning.
- Interpret model coefficients using odds ratios.
- Examine coefficient uncertainty through bootstrap resampling.
- Assess the sensitivity of model coefficients to regularisation strength.

## Dataset

**Source:** [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

The project uses the Cleveland subset of the UCI Heart Disease dataset.

| Property | Description |
|---|---|
| Observations | 303 |
| Input features | 13 |
| Original target | `num` |
| Binary target | 0 or 1 |
| Missing values | 6 cells |
| Classification task | Binary classification |

The original target variable contains values from 0 to 4. For this project, it is converted into a binary target:

- `0`: No recorded disease
- `1`: Recorded disease (`num > 0`)

**Important:** This target represents recorded disease status in the dataset. The model is not designed to predict future cardiovascular risk, provide a clinical diagnosis, or replace medical assessment.

## Methodology

### 1. Exploratory Data Analysis

The initial analysis examined:

- Dataset structure and descriptive statistics.
- Missing-value distribution.
- Numerical and categorical feature distributions.
- Binary target distribution.
- Associations between selected features and the target.

A chi-square test was conducted to investigate the association between chest pain type (`cp`) and the binary target.

The analysis found a statistically significant association:

- Chi-square statistic: 81.816
- Degrees of freedom: 3
- p-value: 1.25171 × 10⁻¹⁷

This indicates an association within the dataset, not a causal relationship.

### 2. Data Preprocessing

The dataset was divided using an 80:20 stratified train-test split with `random_state=42`.

| Dataset | Observations |
|---|---:|
| Training set | 242 |
| Testing set | 61 |

A Scikit-learn preprocessing pipeline was implemented.

**Numerical features**
- Median imputation
- Standard scaling

**Categorical features**
- Most-frequent-value imputation
- One-hot encoding with unknown-category handling

Preprocessing was incorporated into the modelling pipeline to prevent information leakage during cross-validation.

### 3. Model Development and Evaluation

Two classification algorithms were evaluated:

- Logistic Regression
- Random Forest

Both models were evaluated using five-fold stratified cross-validation.

#### Cross-validation results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.847 ± 0.010 | 0.814 ± 0.023 |
| Precision | 0.877 ± 0.054 | 0.797 ± 0.027 |
| Recall | 0.783 ± 0.046 | 0.801 ± 0.080 |
| F1-score | 0.825 ± 0.007 | 0.796 ± 0.036 |
| ROC-AUC | 0.902 ± 0.014 | 0.894 ± 0.033 |

Logistic Regression achieved stronger mean accuracy, precision, F1-score, and ROC-AUC, while Random Forest achieved slightly higher mean recall.

Logistic Regression also showed lower fold-to-fold variability across the reported metrics.

Based on these results, Logistic Regression was selected for further analysis.

### 4. Hyperparameter Tuning

GridSearchCV was used to investigate the regularisation parameter `C` for Logistic Regression.

The search evaluated:

`C = [0.001, 0.01, 0.1, 1, 10, 100]`

The selection criterion was ROC-AUC using five-fold stratified cross-validation.

**Best parameter:** `C = 1`

**Best mean cross-validation ROC-AUC:** 0.9025

The selected model was subsequently evaluated on the held-out test set.

#### Final held-out test performance

| Metric | Score |
|---|---:|
| Accuracy | 0.885 |
| Precision | 0.839 |
| Recall | 0.929 |
| F1-score | 0.881 |
| ROC-AUC | 0.966 |

**Confusion matrix**

| | Predicted: 0 | Predicted: 1 |
|---|---:|---:|
| Actual: 0 | 28 | 5 |
| Actual: 1 | 2 | 26 |

The held-out test set was not used for hyperparameter selection.

These results are internal estimates from a relatively small dataset. Independent external validation is required before drawing conclusions about performance on other populations.

### 5. Model Interpretability

Logistic Regression was selected partly because its coefficients provide a relatively interpretable representation of how input features relate to the model's predicted class.

Two coefficient analyses were conducted using separately fitted models.

**Numerical feature analysis**

Numerical predictors were standardised before fitting. Their odds ratios therefore represent the change in modelled odds associated with a one-standard-deviation increase in the corresponding feature.

| Feature | Odds Ratio |
|---|---:|
| trestbps | 1.406 |
| oldpeak | 1.296 |
| chol | 1.196 |
| age | 0.941 |
| thalach | 0.689 |

**Categorical feature analysis**

A separate reference-category model was fitted using one-hot encoding with the first category dropped.

Some of the resulting odds ratios were:

| Feature | Odds Ratio |
|---|---:|
| ca_1.0 | 4.339 |
| ca_2.0 | 4.274 |
| cp_4.0 | 3.840 |
| thal_7.0 | 3.531 |
| sex_1.0 | 3.392 |
| ca_3.0 | 3.046 |
| slope_2.0 | 2.229 |
| exang_1.0 | 1.587 |

Categorical odds ratios are relative to the corresponding omitted reference categories.

These numerical and categorical results come from two separately fitted models and should not be interpreted as coefficients from one shared model.

Odds ratios describe model associations, not causal effects or clinical importance.

### 6. Bootstrap Uncertainty Analysis

Bootstrap resampling was used to examine the variability of Logistic Regression coefficients.

The analysis involved:

- 300 bootstrap resamples.
- Sampling with replacement from the training set.
- A fixed sample size of 242 observations per resample.
- Consistent feature ordering across fitted models.
- Estimation of coefficient means, standard deviations, percentile intervals, and positive-coefficient frequencies.

Seven categorical features had bootstrap percentile intervals entirely above zero:

- `ca_1.0`
- `ca_2.0`
- `cp_4.0`
- `thal_7.0`
- `sex_1.0`
- `ca_3.0`
- `slope_2.0`

The intervals provide an indication of coefficient sampling variability under the chosen resampling procedure. They should not be treated as definitive statistical significance tests or evidence of causality.

The 300-resample analysis is exploratory; a larger number of resamples could provide more stable interval estimates.

### 7. Regularisation Sensitivity Analysis

The final analysis investigated how Logistic Regression coefficients changed across different regularisation strengths.

The following values were evaluated:

`C = [0.01, 0.1, 1, 10, 100]`

The analysis examined:

- Changes in coefficient magnitude.
- Stability of coefficient direction.
- Features whose coefficients changed sign.
- The effect of stronger and weaker regularisation.

Some feature coefficients maintained their direction across the tested values, while others changed direction.

This analysis highlights the sensitivity of coefficient interpretation to regularisation and the possible influence of relationships between predictors.

It is a coefficient sensitivity analysis, not an additional model-validation experiment.

## Key Findings

1. Logistic Regression provided stronger overall cross-validation performance than the Random Forest baseline on this dataset.
2. The selected Logistic Regression model achieved a held-out ROC-AUC of approximately 0.966.
3. The preprocessing pipeline handled missing values and scaling within the cross-validation workflow.
4. Coefficient-based analysis provided an interpretable view of modelled feature associations.
5. Bootstrap resampling revealed variation in coefficient estimates across training-set resamples.
6. Regularisation sensitivity analysis demonstrated that coefficient direction and magnitude are not equally stable across all predictors.

## Repository Structure

```text
cardiovascular-risk-prediction/
│
├── data/
│   └── raw/
│       └── heart_disease_data.csv
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   └── 04_model_interpretation.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

The raw dataset is excluded from version control through `.gitignore`.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- ucimlrepo

## Reproducibility

Clone the repository:

```bash
git clone https://github.com/divyamohan6388-droid/cardiovascular-risk-prediction.git
cd cardiovascular-risk-prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in numerical order, beginning with dataset exploration.

The raw dataset must be obtained from the UCI Machine Learning Repository and placed at:

```text
data/raw/heart_disease_data.csv
```

## Limitations

This project has several limitations:

- The dataset contains only 303 observations.
- The analysis uses a single dataset subset.
- The held-out test set is relatively small.
- Internal validation does not establish generalisation to other hospitals or populations.
- Model coefficients can be sensitive to regularisation and correlated predictors.
- Bootstrap intervals are based on a limited number of resamples.
- The binary target represents recorded disease status, not future disease development.
- The model has not undergone clinical validation or deployment.

Consequently, the reported results should be understood as an exploratory machine learning analysis rather than evidence of clinical readiness.

## Future Work

Potential extensions include:

- External validation using independent cardiovascular datasets.
- Calibration analysis to assess the reliability of predicted probabilities.
- Increasing bootstrap repetitions for more stable uncertainty estimates.
- Further investigation of coefficient stability and predictor relationships.
- Evaluation of model uncertainty and generalisation across different data distributions.
- Improving reproducibility and documenting the complete experimental environment.

## Author

**Undergraduate Student – Computer Science and Data Analytics**

Indian Institute of Technology Patna

Research interests: Data Science, Statistics, Machine Learning, and Artificial Intelligence.

GitHub: [divyamohan6388-droid](https://github.com/divyamohan6388-droid)

---

*This is an independent undergraduate research-oriented project developed for educational and exploratory purposes. It is not a clinically validated diagnostic system.*