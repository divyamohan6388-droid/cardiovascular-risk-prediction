# Reliable and Explainable Machine Learning for Cardiovascular Disease Classification

**An ongoing research-oriented machine learning project exploring predictive performance, model reliability, and interpretability using the UCI Heart Disease dataset.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Ongoing%20Research-success)
![Research](https://img.shields.io/badge/Focus-Explainable%20ML-purple)

## Overview

This project investigates how machine learning models can classify recorded cardiovascular disease status while maintaining a focus on predictive performance, model reliability, and interpretability.

Using the Cleveland subset of the UCI Heart Disease dataset, the project explores data preprocessing, statistical analysis, model comparison, hyperparameter optimisation, feature interpretation, and coefficient stability.

Rather than focusing exclusively on predictive accuracy, the research also investigates how preprocessing decisions, model selection, and regularisation influence the behaviour and reliability of machine learning predictions.

The primary model explored in this project is **Logistic Regression**, with Random Forest used as a comparative model.

## Research Objectives

The main objectives are:

- Investigate the structure, quality, and statistical characteristics of cardiovascular health-related data.
- Examine missing-data handling and feature preprocessing strategies.
- Develop reproducible machine learning pipelines with appropriate leakage prevention.
- Compare Logistic Regression and Random Forest using stratified cross-validation.
- Investigate the impact of regularisation through hyperparameter optimisation.
- Interpret model coefficients and odds ratios.
- Examine coefficient stability through bootstrap resampling.
- Investigate how modelling choices affect feature-level interpretations.

## Research Questions

1. How do preprocessing and missing-value handling strategies influence model performance?
2. How does Logistic Regression compare with Random Forest on a small cardiovascular dataset?
3. How consistent are model evaluation metrics across cross-validation folds?
4. Does hyperparameter tuning meaningfully improve predictive performance?
5. Which input features exhibit stronger associations with the model's predictions?
6. How stable are Logistic Regression coefficients under bootstrap resampling?
7. How sensitive are feature interpretations to regularisation strength?

## Dataset

**Source:** UCI Machine Learning Repository — Heart Disease Dataset

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

The project uses the Cleveland subset of the dataset.

| Property | Description |
|---|---|
| Total observations | 303 |
| Input features | 13 |
| Original target | `num` |
| Missing values | 6 |
| Classification type | Binary classification |
| Positive class | Recorded disease status (`num > 0`) |

### Target Definition

The original target variable contains values from 0 to 4.

For binary classification, the target is transformed as follows:

- `0`: No recorded disease (`num = 0`)
- `1`: Recorded disease (`num > 0`)

The resulting target distribution is:

| Class | Observations | Percentage |
|---|---:|---:|
| 0 | 164 | 54.13% |
| 1 | 139 | 45.87% |

### Input Features

The dataset contains 13 predictors:

| Feature | Description |
|---|---|
| age | Age |
| sex | Sex |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia-related category |

## Methodology

The project follows a structured machine learning workflow.

### 1. Exploratory Data Analysis

- Dataset structure and statistical summaries.
- Missing-value investigation.
- Target distribution analysis.
- Feature distribution analysis.
- Outlier investigation.
- Categorical feature analysis.
- Statistical association testing using the Chi-square test.

A significant association was observed between chest pain type (`cp`) and binary disease status:

- Chi-square statistic: 81.816
- Degrees of freedom: 3
- p-value: 1.25171 × 10⁻¹⁷

This indicates an association in the analysed dataset, not causation.

### 2. Data Preprocessing

Missing values are handled within Scikit-learn pipelines.

**Numerical features:**
- Median imputation.
- StandardScaler normalisation.

**Categorical features:**
- Most-frequent imputation.
- One-hot encoding.

An alternative missing-category imputation strategy was also investigated.

The alternative strategy produced similar cross-validation results, with no meaningful improvement over the baseline approach.

### 3. Train-Test Split

The dataset is divided using stratified sampling:

| Split | Observations |
|---|---:|
| Training | 242 |
| Testing | 61 |

Configuration:
- Test size: 20%
- Random state: 42
- Stratification: Enabled

The held-out test set is reserved for final evaluation.

### 4. Model Development

Two classification algorithms were investigated:

**Logistic Regression**

Used as the primary model because of its interpretability and suitability as a classification baseline.

**Random Forest**

Used as a comparative model to examine whether a tree-based ensemble provides improved predictive performance.

Both models were evaluated using the same preprocessing strategy and five-fold stratified cross-validation.

### 5. Model Evaluation

The following metrics were considered:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Cross-validation results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.847 ± 0.010 | 0.814 ± 0.023 |
| Precision | 0.877 ± 0.054 | 0.797 ± 0.027 |
| Recall | 0.783 ± 0.046 | 0.801 ± 0.080 |
| F1-score | 0.825 ± 0.007 | 0.796 ± 0.036 |
| ROC-AUC | 0.902 ± 0.014 | 0.894 ± 0.033 |

Logistic Regression achieved higher mean accuracy, precision, F1-score, and ROC-AUC. Random Forest achieved slightly higher mean recall.

Logistic Regression also demonstrated lower fold-to-fold standard deviations across the reported metrics.

### 6. Hyperparameter Optimisation

GridSearchCV was used to investigate the Logistic Regression regularisation parameter `C`.

Values tested:

`[0.001, 0.01, 0.1, 1, 10, 100]`

Best configuration:

- Best C: 1
- Cross-validation ROC-AUC: 0.90249

The default regularisation strength was already the best among the tested configurations. Hyperparameter tuning did not produce a meaningful improvement over the original baseline.

### 7. Final Test Evaluation

The selected Logistic Regression model was evaluated on the untouched test set.

| Metric | Test Performance |
|---|---:|
| Accuracy | 88.52% |
| Precision | 83.87% |
| Recall | 92.86% |
| F1-score | 88.14% |
| ROC-AUC | 96.65% |

**Confusion Matrix**

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 28 | 5 |
| Actual 1 | 2 | 26 |

The model correctly classified 54 out of 61 test observations.

These results represent performance on one held-out split and should not be interpreted as evidence of equivalent performance on external populations.

## Model Interpretability and Explainability

The project investigates how Logistic Regression produces its predictions through coefficient analysis and odds ratios.

### 1. Numerical Feature Interpretation

Numerical feature coefficients were extracted after standardisation.

The corresponding odds ratios were calculated using the exponential transformation of the coefficients.

Selected results:

| Feature | Odds Ratio |
|---|---:|
| trestbps | 1.406 |
| oldpeak | 1.296 |
| chol | 1.196 |
| age | 0.941 |
| thalach | 0.689 |

Since numerical features are standardised, these odds ratios represent a one-standard-deviation increase in the corresponding predictor, holding other model inputs constant.

### 2. Categorical Feature Interpretation

A separate interpretation pipeline using reference-category one-hot encoding was developed.

Selected odds ratios:

| Feature | Odds Ratio |
|---|---:|
| ca_1.0 | 4.339 |
| ca_2.0 | 4.274 |
| cp_4.0 | 3.840 |
| thal_7.0 | 3.531 |
| sex_1.0 | 3.392 |
| ca_3.0 | 3.046 |

These values compare each category with its respective reference category, holding other model inputs constant.

### 3. Feature Coefficient Visualisation

Coefficient visualisations were developed to examine positive and negative model associations across numerical and categorical predictors.

These visualisations help investigate the contribution of individual transformed features to the model's linear predictions.

### 4. Bootstrap Coefficient Stability

A bootstrap experiment was conducted to investigate the stability of Logistic Regression coefficients.

Configuration:

- Training observations: 242
- Bootstrap repetitions: 300
- Sampling method: Sampling with replacement
- Model: Logistic Regression
- Regularisation: C = 1

The experiment produced the following summary:

| Stability classification | Number of features |
|---|---:|
| Consistently positive | 7 |
| Intervals crossing or touching zero | 13 |
| Consistently negative | 0 |

The following features maintained positive coefficient directions across all 300 bootstrap samples:

- ca_1.0
- ca_2.0
- cp_4.0
- thal_7.0
- sex_1.0

The analysis suggests that coefficient stability varies across predictors.

Bootstrap percentile intervals were used to examine uncertainty. These intervals are not definitive statistical significance tests.

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

Note: The raw dataset is excluded from version control through `.gitignore`. Obtain it from the UCI source and place it in `data/raw/` before running the notebooks.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/divyamohan6388-droid/cardiovascular-risk-prediction.git
```

### 2. Navigate to the project

```bash
cd cardiovascular-risk-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the dataset

Obtain the Cleveland Heart Disease dataset from the UCI Machine Learning Repository and place the prepared CSV file at:

```text
data/raw/heart_disease_data.csv
```

### 6. Run the notebooks

Execute the notebooks in numerical order:

1. `01_dataset_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_training_and_evaluation.ipynb`
4. `04_model_interpretation.ipynb`

Run each notebook from top to bottom.

## Current Project Progress

**Project status: Ongoing Research**

| Component | Status |
|---|---|
| Dataset exploration and statistical analysis | Completed |
| Missing-value investigation | Completed |
| Data preprocessing pipelines | Completed |
| Logistic Regression baseline | Completed |
| Random Forest comparison | Completed |
| Five-fold cross-validation | Completed |
| Hyperparameter optimisation | Completed |
| Final held-out test evaluation | Completed |
| Numerical coefficient and odds ratio analysis | Completed |
| Categorical coefficient and odds ratio analysis | Completed |
| Feature coefficient visualisation | Completed |
| Bootstrap coefficient stability analysis | Completed |
| Regularisation sensitivity analysis | In progress |
| Final research conclusions and documentation | Pending |

## Future Work

Planned research directions include:

- Complete regularisation sensitivity analysis.
- Examine the robustness of feature-level interpretations.
- Investigate limitations associated with small datasets.
- Consolidate findings across preprocessing, model comparison, and interpretability experiments.
- Improve research documentation and visualisations.
- Explore further validation strategies where appropriate.

## Limitations

This project has several limitations:

- The dataset contains only 303 observations.
- The analysis uses the Cleveland subset rather than a large, diverse external population.
- The evaluation relies on one stratified train-test split and five-fold cross-validation.
- Bootstrap analysis uses 300 resamples.
- Model coefficients can be affected by correlated predictors, regularisation, and encoding choices.
- Observational associations do not establish causality.

**This project is intended for educational and research purposes. It is not a clinical diagnostic system and must not be used for medical decision-making.**

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git and GitHub

## Research Focus

**Machine Learning | Data Science | Statistical Analysis | Model Evaluation | Explainable AI | Model Reliability**

---

## Author

**Divya Mohan**

Undergraduate Student — BS Computer Science and Data Analytics, IIT Patna

GitHub: https://github.com/divyamohan6388-droid

LinkedIn: https://www.linkedin.com/in/divya-mohan-a42b12389/

---

*This repository documents an ongoing research-oriented machine learning project. Results, experiments, and documentation will be updated as the investigation progresses.*