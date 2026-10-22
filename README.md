
# House Price Prediction Using Machine Learning

A supervised machine learning project focused on predicting residential property prices using regression algorithms. This project explores data preprocessing, feature engineering, model implementation, hyperparameter tuning, and systematic model evaluation.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [Project Workflow](#4-project-workflow)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering and Selection](#6-feature-engineering-and-selection)
7. [Machine Learning Models](#7-machine-learning-models)
8. [Model Evaluation](#8-model-evaluation)
9. [Experiment Tracking](#9-experiment-tracking)
10. [Project Structure](#10-project-structure)
11. [Technologies Used](#11-technologies-used)
12. [Installation and Setup](#12-installation-and-setup)
13. [Running the Project](#13-running-the-project)
14. [House Price Prediction Plan](#14-house-price-prediction-plan)
15. [Challenges and Lessons Learned](#15-challenges-and-lessons-learned)
16. [Future Improvements](#16-future-improvements)
17. [Project Status](#17-project-status)
18. [Learning Outcomes](#18-learning-outcomes)
19. [Dataset Reference](#19-dataset-reference)
20. [License](#20-license)

---

## 1. Project Overview

House Price Prediction is a supervised machine learning project designed to estimate residential property prices using property characteristics, location information, and other relevant attributes.

The project uses the Indian Urban House Price Prediction dataset available on Kaggle.

Rather than relying on a single algorithm, the project follows an experimental approach. Multiple regression algorithms are implemented and evaluated to understand their mathematical foundations, predictive behavior, limitations, and ability to generalize to unseen properties.

The primary objective is not merely to train a model that produces predictions. It is to understand how different algorithms learn, how feature selection affects predictive performance, how hyperparameters influence model behavior, and how to determine whether a model generalizes effectively.

### Key Highlights

- Supervised learning for a real-estate price prediction problem.
- Implementation of multiple regression algorithms.
- Exploratory data analysis and data preprocessing.
- Feature selection and feature engineering.
- Investigation of correlated and redundant features.
- Hyperparameter tuning and model optimization.
- Evaluation using MAE, RMSE, and R².
- Prediction error analysis across different price ranges.
- Investigation of underfitting, overfitting, bias, and variance.
- A structured roadmap for advanced regression models.
- Development of a reproducible machine learning workflow.

## 2. Objectives

### Primary Objective

Develop a regression model capable of predicting residential property prices while maintaining reliable performance on unseen observations.

### Technical Objectives

1. Understand the structure and characteristics of the dataset.
2. Perform exploratory data analysis.
3. Identify missing values, outliers, and potential data-quality issues.
4. Investigate relationships between input features and property prices.
5. Identify redundant or uninformative features.
6. Implement multiple regression algorithms.
7. Understand the mathematical foundations of each algorithm.
8. Experiment with different feature configurations and transformations.
9. Optimize model hyperparameters.
10. Evaluate models using appropriate regression metrics.
11. Analyze prediction errors and systematic biases.
12. Compare algorithms under consistent evaluation conditions.
13. Develop a reproducible training and prediction pipeline.

### Learning Objectives

The project also explores the following machine learning concepts:

- Linear regression and optimization.
- Decision trees and recursive feature partitioning.
- Ensemble learning.
- Bootstrap sampling and bagging.
- Boosting and sequential learning.
- Distance-based regression.
- Bias-variance trade-off.
- Regularization.
- Feature scaling and categorical encoding.
- Cross-validation.
- Hyperparameter optimization.
- Generalization and data leakage prevention.

---

## 3. Dataset

### Dataset Source

**Dataset:** Indian Urban House Price Prediction

**Platform:** Kaggle

**Dataset URL:**

https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data

The dataset contains synthetic residential property data designed to simulate housing markets across Indian cities.

It is intended for machine learning experimentation, regression modeling, feature engineering, and model evaluation.

**Important:** The dataset is synthetic. Its observations should not be interpreted as actual property transactions or as a definitive representation of real-world Indian property prices.

### Dataset Features

The dataset includes property characteristics, location information, amenities, and a target price.

The following table summarizes the features described for this dataset. The exact column names and availability should be checked against the downloaded CSV files.

| Feature | Description | Type |
|---|---|---|
| `Property_ID` | Property identifier | Identifier |
| `City` | City where the property is located | Categorical |
| `Locality_Type` | Classification of the locality | Categorical |
| `Distance_to_Metro_km` | Distance to the nearest metro station | Numerical |
| `Distance_to_City_Center_km` | Distance to the city center | Numerical |
| `Property_Type` | Type of residential property | Categorical |
| `BHK` | Number of bedrooms | Numerical |
| `Bathrooms` | Number of bathrooms | Numerical |
| `Super_Area_SqFt` | Super built-up area in square feet | Numerical |
| `Carpet_Area_SqFt` | Carpet area in square feet | Numerical |
| `Floor_Number` | Floor on which the property is located | Numerical |
| `Total_Floors` | Total number of floors in the building | Numerical |
| `Age_of_Property` | Age of the property in years | Numerical |
| `Furnishing_Status` | Furnishing condition | Categorical |
| `Parking` | Parking availability | Binary or categorical |
| `Lift_Available` | Lift availability | Binary or categorical |
| `Gated_Community` | Gated-community indicator | Binary or categorical |
| `Price_INR_Lakhs` | Property price in lakhs of Indian rupees | Target |

The table is intended as a dataset reference. Before using it as a definitive schema, compare every column against the actual downloaded dataset.

### Target Variable

The target variable documented for the competition is:

`Price_INR_Lakhs`

The objective is to predict the property price in lakhs of Indian rupees.

Mathematically, the regression problem is represented as:

\[
\hat{y}=f(X)
\]

Where:

- \(X\) represents the input features.
- \(y\) represents the actual property price.
- \(\hat{y}\) represents the predicted property price.
- \(f\) represents the learned regression function.

### Dataset Files

The competition may provide files such as:

- `train.csv`: Training observations and target values.
- `test.csv`: Observations for which predictions are required.
- `sample_submission.csv`: Example submission format.
- `metaData.csv`: Supplemental dataset information, if provided.

The exact filenames must be verified against the downloaded competition files.

The Kaggle competition test set and a locally held-out test set serve different purposes. A local test set should be kept separate from model tuning, while the competition test set is used to generate predictions in the required submission format.

---

## 4. Project Workflow

The project follows a structured machine learning workflow.

```text
                    DATASET
                       |
                       v
              DATA UNDERSTANDING
                       |
                       v
          EXPLORATORY DATA ANALYSIS
                       |
                       v
                DATA CLEANING
                       |
                       v
            FEATURE ENGINEERING
                       |
                       v
            TRAIN-TEST SEPARATION
                       |
                       v
             DATA PREPROCESSING
                       |
                       v
           BASELINE MODEL TRAINING
                       |
                       v
             MODEL IMPLEMENTATION
                       |
                       v
           HYPERPARAMETER TUNING
                       |
                       v
              MODEL EVALUATION
                       |
                       v
             ERROR DISTRIBUTION
                  ANALYSIS
                       |
                       v
              MODEL COMPARISON
                       |
                       v
             FINAL MODEL SELECTION
                       |
                       v
           PREDICTION ON NEW DATA
```

### Step 1: Data Understanding

Inspect the dataset structure, data types, feature definitions, target variable, and data quality.

### Step 2: Exploratory Data Analysis

Investigate feature distributions, correlations, outliers, and relationships between property characteristics and prices.

### Step 3: Data Cleaning

Identify missing values, invalid observations, duplicate records, inconsistent categories, and other data-quality issues.

### Step 4: Feature Engineering

Evaluate relevant features, investigate redundancy, and experiment with transformations.

### Step 5: Train-Test Separation

Separate the available training data into appropriate subsets for model development and final evaluation.

### Step 6: Model Implementation

Implement regression algorithms in separate modules and train them using controlled experiments.

### Step 7: Hyperparameter Tuning

Investigate how model parameters influence predictive performance and generalization.

### Step 8: Evaluation

Calculate regression metrics and analyze prediction errors.

### Step 9: Model Comparison

Compare models using consistent evaluation conditions and select a promising candidate.

### Step 10: Prediction Pipeline

Develop a reusable workflow that applies the correct preprocessing and generates predictions for unseen properties.

---

## 5. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is used to understand the dataset before model training.

The project investigates relationships between property prices and the available features.

### Areas of Investigation

- Distribution of property prices.
- Distribution of property areas.
- Relationships between area and price.
- Correlations between area-related features.
- Price differences across cities.
- Price differences across locality types.
- Relationships between property age and price.
- Relationships between structural characteristics and price.
- Distribution of extreme observations.
- Prediction error distributions.

### Visualization Techniques

Potential visualizations include:

- Histograms.
- Box plots.
- Scatter plots.
- Correlation heatmaps.
- Actual-versus-predicted plots.
- Residual plots.
- Price-group error comparisons.

EDA should help formulate and test hypotheses. Correlation alone does not establish that a feature will improve predictive performance.

---

## 6. Feature Engineering and Selection

Feature selection is an important part of this project.

The objective is not to use every available feature. It is to identify a feature configuration that provides useful predictive information for the algorithm being evaluated.

### 6.1 Feature Selection Experiments

The project has investigated different combinations of property features.

Based on the experiments conducted so far:

- `BHK` was evaluated but was not retained in the selected feature configuration because it did not provide sufficient additional predictive value in the tested experiments.
- `Bathrooms` was evaluated but was not retained in the selected feature configuration for the same reason.
- `Carpet_Area` was investigated because of its high correlation with `Super_Area`.
- Location, property area, age, and distance-related features have been considered in the selected configurations.

These findings apply to the tested feature configurations. They do not imply that BHK or Bathrooms are universally useless for house-price prediction.

A feature's contribution depends on the dataset, model, other selected features, and evaluation procedure.

### 6.2 Multicollinearity

Multicollinearity occurs when input features contain strongly overlapping information.

For example, `Super_Area` and `Carpet_Area` both describe aspects of property size.

Including both may introduce redundant information.

In Linear Regression, strong multicollinearity can make coefficient estimates unstable.

Tree-based models are affected differently, although redundant features can still influence split selection.

The project has investigated configurations that exclude `Carpet_Area` because of its high correlation with `Super_Area`.

Feature removal should be justified through data analysis and controlled experiments rather than correlation alone.

### 6.3 Categorical Encoding

Features such as `City`, `Locality_Type`, `Property_Type`, and `Furnishing_Status` require appropriate preprocessing when used by algorithms that expect numerical inputs.

Possible approaches include:

- One-hot encoding.
- Ordinal encoding when a meaningful order exists.
- Other encoding strategies appropriate to the model and dataset.

Encoding must be fitted on training data and applied consistently to validation and test data.

### 6.4 Feature Scaling

Feature scaling is particularly important for distance-based algorithms such as K-Nearest Neighbors.

Without scaling, features with larger numerical ranges can dominate distance calculations.

Standardization is defined as:

\[
z=\frac{x-\mu}{\sigma}
\]

Where:

- \(x\) is the original value.
- \(\mu\) is the training-set mean.
- \(\sigma\) is the training-set standard deviation.

Scaling parameters must be calculated using training data to prevent data leakage.

Decision Trees, Random Forests, and conventional tree-based Gradient Boosting generally do not require standardization.

### 6.5 Target Transformation

A logarithmic target transformation can be investigated if the price distribution is strongly right-skewed.

For nonnegative prices, one possible transformation is:

\[
y'=\log(1+y)
\]

Predictions must be converted back to the original scale before interpreting errors in price units:

\[
\hat{y}=\exp(\hat{y}')-1
\]

Target transformation should be evaluated experimentally rather than assumed to improve every model.

---

## 7. Machine Learning Models

The project implements regression algorithms as separate modules.

The emphasis is on understanding the mathematical foundations, implementation details, and behavior of each algorithm.

### 7.1 Linear Regression

**Status: Implemented**

Linear Regression models the relationship between input features and the target using a linear combination of the features.

\[
\hat{y}=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p
\]

Ordinary least squares estimates coefficients by minimizing the sum of squared residuals.

#### Objectives

- Establish a baseline model.
- Understand the regression objective.
- Investigate feature selection.
- Experiment with target transformations.
- Evaluate the limitations of linear relationships.

#### Limitations

- May underfit nonlinear relationships.
- Can be sensitive to outliers.
- Can produce unstable coefficient estimates under strong multicollinearity.
- Requires appropriate categorical encoding.

### 7.2 Decision Tree Regression

**Status: Implemented**

Decision Tree Regression partitions the feature space into regions using feature-based splits.

Each leaf produces a prediction based on the training observations assigned to that region.

Regression trees commonly select splits that reduce squared prediction error.

#### Objectives

- Understand recursive feature partitioning.
- Model nonlinear relationships.
- Investigate feature interactions.
- Study tree depth and minimum leaf size.
- Understand overfitting and the bias-variance trade-off.

#### Important Hyperparameters

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

### 7.3 Random Forest Regression

**Status: Implemented**

Random Forest combines predictions from multiple decision trees.

It commonly uses bootstrap samples and randomized feature selection to construct an ensemble.

For regression, the ensemble prediction is typically the average of individual tree predictions:

\[
\hat{y}=\frac{1}{B}\sum_{b=1}^{B}\hat{y}_b
\]

Where \(B\) is the number of trees.

#### Objectives

- Understand ensemble learning.
- Investigate bootstrap sampling.
- Understand bagging.
- Study variance reduction.
- Compare ensemble predictions with a single Decision Tree.
- Investigate tree complexity and feature selection.

#### Important Hyperparameters

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `bootstrap`

### 7.4 K-Nearest Neighbors Regression

**Status: Implemented**

KNN Regression predicts a property's price using nearby observations in feature space.

For an unweighted KNN regressor:

\[
\hat{y}(x)=\frac{1}{k}\sum_{i\in N_k(x)}y_i
\]

Where \(N_k(x)\) contains the \(k\) nearest training observations.

#### Objectives

- Understand distance-based prediction.
- Investigate feature scaling.
- Study the number of neighbors.
- Evaluate different feature configurations.
- Analyze the limitations of distance-based learning.

#### Important Hyperparameters

- `n_neighbors`
- `weights`
- `metric`
- `p`

KNN performance depends on the selected features, scaling method, distance metric, and distribution of training observations.

### 7.5 Gradient Boosting Regression

**Status: In Progress**

Gradient Boosting Regression is the next algorithm being developed.

Unlike Random Forest, which generally builds trees independently and aggregates their predictions, Gradient Boosting builds an additive model sequentially.

Each new learner attempts to improve the current ensemble by following information derived from the selected loss function.

A simplified additive model is:

\[
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta h_m(x)
\]

Where:

- \(F_0(x)\) is the initial prediction.
- \(h_m(x)\) is the contribution of the \(m\)-th learner.
- \(M\) is the number of boosting stages.
- \(\eta\) is the learning rate.

For squared-error regression, boosting can be understood through the residuals of the current model.

The algorithm repeatedly fits a new learner to information about current prediction errors and adds its contribution to the ensemble.

#### Objectives

- Understand boosting mathematically.
- Distinguish bagging from boosting.
- Implement sequential model improvement.
- Understand residuals and gradients.
- Study learning rate and the number of trees.
- Investigate overfitting and generalization.

#### Important Hyperparameters

- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `subsample`
- `loss`

The model will be implemented in a dedicated source module and evaluated against the existing models.

---

## 8. Model Evaluation

The project uses multiple regression metrics to evaluate predictive performance.

### 8.1 Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Lower MAE indicates smaller average absolute prediction errors.

MAE is expressed in the same units as the target.

### 8.2 Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE penalizes large prediction errors more heavily than MAE.

The Kaggle competition uses RMSE as its evaluation metric.

### 8.3 Coefficient of Determination (R²)

R² measures predictive performance relative to predicting the evaluation-set target mean.

\[
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

Interpretation:

- \(R^2=1\): Perfect predictions.
- \(R^2=0\): Equivalent squared-error performance to predicting the evaluation-set mean.
- \(R^2<0\): Worse squared-error performance than the mean-prediction baseline.

A high R² does not guarantee small errors for every property or price range.

### 8.4 Signed Prediction Error

Define prediction error as:

\[
e_i=y_i-\hat{y}_i
\]

Under this convention:

- Positive error indicates underprediction.
- Negative error indicates overprediction.

The sign convention should remain consistent across experiments.

### 8.5 Price-Group Error Analysis

Overall metrics can conceal weaknesses in particular price ranges.

To investigate this, prediction errors can be grouped by actual property price.

Useful statistics include:

- MAE.
- Mean Error.
- Median Error.
- RMSE.
- Number of observations.

This analysis helps identify whether the model systematically underpredicts expensive properties or performs poorly in particular price ranges.

### 8.6 Fair Model Comparison

For meaningful comparisons:

- Use consistent evaluation data.
- Fit preprocessing only on training data.
- Use validation data or cross-validation for model selection.
- Avoid repeatedly tuning against the final test set.
- Evaluate all models on the same target scale.
- Record experiment configurations.
- Preserve a final held-out test set.

Training performance alone should not determine the selected model.

---

## 9. Experiment Tracking

Every experiment should record its configuration and results.

A suggested record structure is:

```python
{
    "Model": "Model Name",
    "Target": "Raw Price or Log Target",
    "Feature_Configuration": "Selected Features",
    "Hyperparameters": {},
    "Metrics": {
        "MAE": None,
        "RMSE": None,
        "R2": None
    }
}
```

This is a template rather than a record of an actual experiment.

### Questions Investigated

- Does a nonlinear model outperform Linear Regression?
- How does tree depth affect generalization?
- Does Random Forest improve on a single Decision Tree?
- How does feature scaling affect KNN?
- Does removing redundant features improve performance?
- Does target transformation improve predictive performance?
- Does Gradient Boosting improve on existing models?
- Which model performs consistently across price ranges?

A model should be selected based on measured performance, generalization, and the intended use of its predictions.

---

## 10. Project Structure

The following is a proposed repository structure. Update filenames and directories to match the actual implementation.

```text
House-Price-Prediction/
│
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── Linear_Regression.py
│   ├── Tree_Regression.py
│   ├── Random_Forest.py
│   ├── Knn_Regression.py
│   └── Gradient_Boosting_Regression.py
│
├── results/
│   ├── experiments/
│   ├── metrics/
│   └── plots/
│
├── models/
│
├── requirements.txt
│
└── .gitignore
```

### Directory Responsibilities

| Directory | Purpose |
|---|---|
| `data/raw/` | Original dataset files |
| `data/processed/` | Cleaned or transformed data |
| `notebooks/` | EDA, preprocessing, and experiments |
| `src/` | Machine learning implementations |
| `results/experiments/` | Experiment configurations and observations |
| `results/metrics/` | Evaluation results |
| `results/plots/` | Visualizations and error analysis |
| `models/` | Saved trained models, if required |

This structure is a proposal. The actual repository should reflect the files that exist.

Large datasets and generated model files should generally be excluded from Git unless there is a specific reason to include them.

---

## 11. Technologies Used

The project uses Python and common data science libraries.

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn |Preprocessing, and evaluation utilities |

The exact dependencies should match the imports used in the repository.

### Example Requirements

```text
numpy
pandas
matplotlib
scikit-learn
```

For reproducibility, package versions should be recorded after verifying the working environment.

---

## 12. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Git.
- A code editor such as VS Code.
- The required Python libraries.

### Step 1: Clone the Repository

Replace the placeholder with the actual GitHub repository URL.

```bash
git clone git@github.com:ashishkrsingh21/House-Price-Prediction.git
```

### Step 2: Enter the Project Directory

```bash
cd House-Price-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### Step 4: Install Dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

### Step 5: Download the Dataset

Download the dataset from:

https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data

Place the relevant files in the appropriate project directory and update the file paths in the notebooks or scripts if necessary.

---

## 13. Running the Project

Execution commands depend on the actual filenames and how each module is implemented.

If the source files support direct execution, the commands may follow this pattern:

```bash
python src/Linear_Regression.py
```

```bash
python src/Tree_Regression.py
```

```bash
python src/Random_Forest.py
```

```bash
python src/knn_Regression.py
```

After Gradient Boosting has been implemented:

```bash
python src/Gradient_Boosting_Regression.py
```

These commands assume that the corresponding files exist and support direct execution.

The notebooks can also be used for exploratory analysis, training, and evaluation.

---

## 14. House Price Prediction Plan

The roadmap describes the intended development of the project beyond the currently implemented models.

### Phase 1: Foundational Regression Models

**Status: Completed**

#### 1. Linear Regression

- Establish a baseline.
- Understand the regression objective.
- Investigate feature selection.
- Experiment with target transformations.

#### 2. Decision Tree Regression

- Understand recursive feature splitting.
- Investigate nonlinear relationships.
- Study tree complexity.
- Analyze overfitting and underfitting.

#### 3. Random Forest Regression

- Understand ensemble learning.
- Investigate bootstrap sampling.
- Study bagging.
- Investigate variance reduction.
- Tune tree parameters.

### Phase 2: Distance-Based Regression

**Status: Implemented**

#### 4. K-Nearest Neighbors Regression

- Understand distance-based prediction.
- Investigate feature scaling.
- Tune the number of neighbors.
- Compare feature configurations.
- Analyze prediction errors.

### Phase 3: Boosting

**Status: In Progress**

#### 5. Gradient Boosting Regression

**Planned source module:** `src/Gradient_Boosting_Regression.py`

Development will proceed module by module.

**Module 1: Mathematical Foundations**

- Boosting versus bagging.
- Additive models.
- Loss functions.
- Residuals and negative gradients.
- Weak learners.
- Learning rate.
- Number of boosting stages.

**Module 2: Algorithm Design**

- Initial prediction.
- Calculation of the negative gradient.
- Training a regression tree.
- Updating the ensemble.
- Understanding the mathematical objective.

**Module 3: Implementation**

- Create the source module.
- Implement training and prediction.
- Establish a baseline configuration.
- Evaluate the model.
- Record the experiment.

**Module 4: Hyperparameter Experiments**

Investigate:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `subsample`

**Module 5: Evaluation**

Compare Gradient Boosting with existing models using MAE, RMSE, R², and price-group error analysis.

**Module 6: Interpretation**

Investigate generalization, overfitting, prediction errors, and whether the model provides a meaningful improvement.

### Phase 4: Advanced Boosting Algorithms

**Status: Planned**

#### 6. XGBoost Regression

Topics:

- Regularized boosting.
- First- and second-order optimization.
- Tree construction.
- Feature and row subsampling.
- Hyperparameter tuning.

#### 7. LightGBM Regression

Topics:

- Histogram-based learning.
- Leaf-wise tree growth.
- Computational efficiency.
- Model optimization.

#### 8. CatBoost Regression

Topics:

- Categorical feature handling.
- Ordered boosting.
- Regularization.
- Performance on mixed numerical and categorical data.

These algorithms will be evaluated against existing models rather than assumed to be better.

### Phase 5: Additional Regression Models

**Status: Optional Future Work**

Potential models include:

- Support Vector Regression.
- Ridge Regression.
- Lasso Regression.
- Elastic Net.
- Extra Trees Regression.

These models can provide additional insight into regularization, nonlinear regression, and randomized ensembles.

### Phase 6: Robust Model Evaluation

**Status: Planned**

- Establish consistent evaluation procedures.
- Introduce cross-validation.
- Tune hyperparameters using validation data.
- Compare models using MAE, RMSE, and R².
- Analyze residuals.
- Evaluate price-group performance.
- Investigate model stability.
- Select a final candidate.
- Evaluate it on the held-out test set.

### Phase 7: Final Prediction Pipeline

**Status: Planned**

Develop a reusable prediction pipeline that includes:

- Input validation.
- Numerical preprocessing.
- Categorical encoding.
- Feature transformations.
- Model loading.
- Price prediction.
- Output formatting.

The preprocessing steps used during training must be reproduced during prediction.

### Phase 8: Deployment

**Status: Optional Future Work**

Potential deployment approaches include:

- Command-line prediction.
- Streamlit application.
- Flask API.
- FastAPI service.
- Web interface for entering property characteristics.

Deployment should follow model evaluation and pipeline development.

### Roadmap Summary

| Component | Status |
|---|---|
| Linear Regression | Completed |
| Decision Tree Regression | Completed |
| Random Forest Regression | Completed |
| KNN Regression | Completed |
| Gradient Boosting Regression | In Progress |
| XGBoost Regression | Planned |
| LightGBM Regression | Planned |
| CatBoost Regression | Planned |
| Additional Regression Models | Planned |
| Cross-Validation and Model Comparison | Planned |
| Final Prediction Pipeline | Planned |
| Deployment | Optional |

---

## 15. Challenges and Lessons Learned

### Feature Selection

More features do not necessarily improve predictive performance.

A feature may contain redundant or weak information when considered alongside other variables.

The project evaluates different feature configurations rather than assuming that every original feature must be retained.

### Nonlinear Relationships

Linear Regression may not represent complex nonlinear relationships adequately.

Decision Trees and ensemble methods can capture nonlinear patterns through feature-based partitions and combinations of learners.

### Overfitting

A complex model may fit training observations closely but perform poorly on unseen observations.

Controlling model complexity and using validation procedures help evaluate generalization.

### Feature Scaling

Distance-based algorithms can be sensitive to feature scale.

Scaling is important for KNN, while tree-based algorithms generally do not require standardization.

### Error Analysis

Overall metrics may hide systematic weaknesses.

Analyzing signed errors and price groups helps identify underprediction, overprediction, and segments where model performance deteriorates.

### Reproducibility

Experiment configurations and results should be recorded consistently.

This makes it easier to identify why a model improved or deteriorated and prevents conclusions based on uncontrolled comparisons.

---

## 16. Future Improvements

### Modeling

- Complete Gradient Boosting Regression.
- Investigate advanced boosting algorithms.
- Explore regularized regression.
- Evaluate alternative feature configurations.
- Investigate useful nonlinear transformations.

### Evaluation

- Introduce cross-validation.
- Preserve an untouched test set.
- Compare MAE, RMSE, and R².
- Analyze residual distributions.
- Evaluate errors across price groups.
- Investigate performance stability.

### Data Quality

- Validate feature ranges.
- Check missing values.
- Investigate duplicate observations.
- Verify categorical encoding.
- Prevent target leakage.
- Examine the effects of extreme observations.

### Engineering

- Improve code modularity.
- Standardize experiment records.
- Save model configurations and metrics.
- Build a reusable preprocessing pipeline.
- Record dependency versions.
- Document execution instructions.

### Deployment

- Save the selected model.
- Build a prediction interface.
- Validate user inputs.
- Return predictions in clearly defined units.
- Document model limitations.

---

## 17. Project Status

The project has progressed through foundational regression algorithms and is moving toward boosting methods.

| Component | Status |
|---|---|
| Exploratory Data Analysis | Completed |
| Data Cleaning and Preprocessing | Completed |
| Feature Selection Experiments | Ongoing |
| Linear Regression | Completed |
| Decision Tree Regression | Completed |
| Random Forest Regression | Completed |
| KNN Regression | Completed |
| Gradient Boosting Regression | In Progress |
| Advanced Boosting Algorithms | Planned |
| Robust Model Comparison | Planned |
| Final Prediction Pipeline | Planned |
| Deployment | Optional Future Work |

---

## 18. Learning Outcomes

The project aims to develop practical and mathematical understanding of:

- Supervised learning.
- Regression objectives.
- Linear and nonlinear models.
- Decision tree construction.
- Bootstrap sampling.
- Bagging and boosting.
- Bias-variance trade-off.
- Feature selection.
- Feature scaling.
- Categorical encoding.
- Hyperparameter optimization.
- Model evaluation.
- Residual analysis.
- Cross-validation.
- Generalization.
- Data leakage prevention.
- Reproducible machine learning workflows.

The emphasis is on understanding why algorithms work, how they behave, and when they are appropriate.

---

## 19. Dataset Reference

**Indian Urban House Price Prediction — Kaggle**

Competition and dataset page:

https://www.kaggle.com/competitions/indian-urban-house-price-prediction

Dataset files:

https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data

The dataset is synthetic and intended for machine learning experimentation.

The competition uses Root Mean Squared Error (RMSE) as its evaluation metric.

For local experiments, evaluation should follow the project's documented train-validation-test procedure.

---

## 20. License

The repository's code license should be specified separately in a `LICENSE` file.

Before redistributing the dataset, verify its current license and usage conditions on Kaggle.

The code license and dataset license may differ. Do not assume that the code license automatically applies to the dataset.

---

## Conclusion

This project explores house-price prediction through systematic implementation and evaluation of supervised regression algorithms.

It progresses from foundational models such as Linear Regression and Decision Tree Regression to ensemble approaches such as Random Forest and Gradient Boosting.

The immediate next step is completing Gradient Boosting Regression and comparing its performance against the existing models.

The broader objective is to develop a reliable prediction pipeline through mathematical understanding, controlled experimentation, feature engineering, and rigorous evaluation.

**The guiding principle is to understand the algorithm, implement it, evaluate it, analyze its errors, and use evidence to determine whether it improves the project.**
