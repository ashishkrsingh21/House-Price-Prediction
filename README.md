# House Price Prediction — India

**A Machine Learning Regression Project | From-Scratch Implementations | Model Evaluation & Optimization**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-In_Progress-orange)](#project-status)
[![Project Type](https://img.shields.io/badge/Project-Supervised_Machine_Learning-purple)](#project-overview)

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Repository Structure](#repository-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [Error Analysis and Findings](#error-analysis-and-findings)
- [Implementation Details](#implementation-details)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Results and Experiment Tracking](#results-and-experiment-tracking)
- [Challenges and Lessons Learned](#challenges-and-lessons-learned)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

House Price Prediction is an end-to-end supervised machine learning project that investigates how residential property characteristics influence housing prices across Indian cities.

The project uses a dataset of Indian urban properties to build, evaluate, and compare regression models. Its primary objective is not merely to obtain accurate predictions, but also to understand the mathematical foundations, implementation details, optimization procedures, and practical limitations of different regression algorithms.

A central aspect of this project is the **manual implementation of multiple machine learning algorithms**. Rather than treating machine learning libraries as black boxes, the project explores the underlying mechanics of regression models, including parameter estimation, loss minimization, tree construction, bootstrap sampling, nearest-neighbor prediction, boosting, and support vector optimization.

The project follows an iterative workflow:

1. Understand the dataset through exploratory data analysis.
2. Clean and transform the data.
3. Establish a baseline for comparison.
4. Implement regression algorithms.
5. Evaluate models using multiple metrics.
6. Investigate prediction errors and model behavior.
7. Experiment with hyperparameters and feature configurations.
8. Select and document promising models.
9. Develop a reproducible prediction and evaluation pipeline.

The project is being developed incrementally, with additional experiments, software engineering improvements, and model comparisons planned.

## Objectives

The main objectives are:

- Develop a complete regression workflow using a real-world housing dataset.
- Understand the statistical and mathematical foundations of supervised learning.
- Implement regression algorithms from scratch where practical.
- Study the effects of feature selection, categorical encoding, scaling, and target transformations.
- Compare linear, tree-based, instance-based, ensemble, and margin-based regression approaches.
- Investigate underfitting, overfitting, outliers, and generalization performance.
- Analyze prediction errors across price ranges, localities, and cities.
- Build reusable Python modules instead of maintaining all logic in a single notebook.
- Record experimental configurations and evaluation results.
- Establish a foundation for more advanced machine learning projects.

## Dataset

### Source

The project uses the **Indian Urban House Price Prediction** dataset obtained from Kaggle.

- **Platform:** Kaggle
- **Dataset:** Indian Urban House Price Prediction
- **Observations:** Approximately 80,000
- **Original columns:** 18
- **Dataset size:** Approximately 11 MB
- **Target variable:** `Price_INR_Lakhs`
- **Prediction task:** Supervised regression

The exact dataset URL should be added here before publication.

### Dataset split

| Subset | Observations | Proportion |
|---|---:|---:|
| Training set | 64,000 | 80% |
| Test set | 16,000 | 20% |
| Total | 80,000 | 100% |

The training set is used to fit model parameters. The test set is intended for final evaluation on unseen observations. Any hyperparameter tuning should use a separate validation set or an appropriate cross-validation procedure rather than repeatedly optimizing against the final test set.

### Feature categories

The dataset contains numerical and categorical property information.

| Category | Examples |
|---|---|
| Property size | Super area, carpet area |
| Property characteristics | BHK, bathrooms, floor information, parking |
| Property age | Age of property |
| Location | City, locality type, distance to city centre |
| Property classification | Property type |
| Furnishing | Furnishing status |
| Target | Price in Indian rupee lakhs |

The exact set of usable predictors varies between experiments because different feature configurations were tested.

### Target variable

The target is `Price_INR_Lakhs`, representing the property price in lakhs of Indian rupees.

For example, a target value of `100` represents ₹100 lakh, or ₹1 crore.

The project investigates prediction in the original price scale as well as experiments involving target transformations. Results obtained using different target scales must be converted consistently before they are compared.

### Important data considerations

- Property identifiers such as `Property_ID` are excluded from the predictive feature matrix.
- Features derived directly from the target must not be used as predictors.
- Price-per-square-foot information requires particular care: if it is calculated using the target price, including it as a feature would introduce target leakage.
- Extreme property prices can substantially affect squared-error metrics.
- Different cities and locality categories may have different price distributions and prediction difficulty.

---

## Project Workflow

The project is organized into stages so that data preparation, model development, and evaluation can be investigated separately.

### 1. Exploratory Data Analysis

The initial analysis examines:

- Dataset dimensions and column types.
- Numerical and categorical feature distributions.
- Summary statistics.
- Relationships between property characteristics and prices.
- Differences between cities and locality categories.
- Potential outliers and unusually expensive properties.
- Feature correlations and possible multicollinearity.
- The distribution and spread of the target variable.

### 2. Data cleaning

The cleaning stage prepares the original dataset for modelling.

Work includes examining missing values, inspecting feature validity, handling unsuitable columns, and preparing numerical and categorical features for subsequent transformations.

The final cleaning rules should be documented alongside the cleaning notebook so that the training and inference pipelines apply consistent transformations.

### 3. Feature engineering and preprocessing

The preprocessing workflow includes:

- Separating features from the target.
- Removing identifier columns.
- Splitting data into training and test subsets.
- Scaling numerical features where appropriate.
- Encoding categorical variables.
- Constructing the final feature matrix.
- Managing feature names and column ordering.
- Investigating alternative feature configurations.

Transformers should be fitted using training data only and then applied to validation and test data. This prevents information from evaluation data from leaking into the training process.

### 4. Baseline modelling

A baseline establishes a reference point for determining whether more complex algorithms provide meaningful improvements.

A simple baseline, such as predicting the training-target mean, provides a useful minimum benchmark for MAE, RMSE, and R². The baseline's final numerical results should be recorded in the experiment log.

### 5. Model implementation

The project investigates multiple regression algorithms, with emphasis on understanding their mathematical foundations and implementing their core operations manually.

### 6. Hyperparameter tuning

Experiments investigate the effects of parameters such as tree depth, number of estimators, learning rate, number of neighbors, regularization strength, and optimization settings.

### 7. Evaluation and error analysis

Models are evaluated using numerical metrics and group-level error analysis. The goal is to understand not only how accurate a model is overall, but also where and why it fails.

---

## Repository Structure

The following is the **working structure reconstructed from the project history**, not a verified listing of the current filesystem. Keep the actual filenames and locations synchronized with the repository.

```text
House-Price-Prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── train.csv
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── preprocessing.py
│   ├── linear_regression.py
│   ├── Tree_Regression.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   ├── knn_regression.py
│   ├── svr.py
│   ├── transformer.py
│   ├── feature_names.py
│   ├── evaluation.py
│   └── experiment_logger.py
│
├── report/
│   └── model_results.csv
│
└── tests/
    └── ...
```

**Structure notes:**

- The three notebooks listed above are the notebooks explicitly discussed during the project.
- Some model filenames shown here describe the logical modules and may differ from the names in the current repository.
- The project has used both a `preprocessing.py` module and a `src/preprocessing/` package in its development discussions. Avoid maintaining duplicate implementations of the same responsibility without a clear reason.
- Add a file to the tree only when it exists in the repository. Planned tests, modules, and reports should not be represented as completed files prematurely.

### Responsibilities of the main components

| Component | Responsibility |
|---|---|
| `01_data_exploration.ipynb` | Explore the dataset, distributions, relationships, and anomalies. |
| `02_data_cleaning.ipynb` | Investigate data quality and prepare clean modelling data. |
| `03_model_training.ipynb` | Run model experiments and compare configurations. |
| `data_loader.py` | Load data, separate predictors and target, and provide split datasets. |
| `transformer.py` | Scale numerical variables and encode categorical variables. |
| `feature_names.py` | Maintain interpretable names for transformed features. |
| `linear_regression.py` | Implement linear regression and parameter estimation. |
| `Tree_Regression.py` | Implement decision tree regression. |
| `random_forest.py` | Implement bootstrap-based tree ensembles. |
| `gradient_boosting.py` | Implement sequential boosting for regression. |
| `knn_regression.py` | Implement distance-based regression. |
| `svr.py` | Implement support vector regression experiments. |
| `evaluation.py` | Calculate regression metrics and organize evaluation results. |
| `experiment_logger.py` | Record model configurations and experimental outcomes. |
| `report/model_results.csv` | Store model comparison results, when generated by the logging workflow. |
| `tests/` | Validate preprocessing, model behavior, and evaluation functions. |

---

## Exploratory Data Analysis

Exploratory analysis is important because housing prices are influenced by several interacting variables and often contain a long upper tail of expensive properties.

### Target distribution

The recorded test-target summary from one experiment is:

| Statistic | Value (₹ lakhs) |
|---|---:|
| Count | 16,000 |
| Mean | 142.787 |
| Standard deviation | 148.800 |
| Minimum | 5.000 |
| 25th percentile | 54.385 |
| Median | 96.460 |
| 75th percentile | 175.790 |
| Maximum | 3,216.464 |

The mean is substantially higher than the median, indicating a right-skewed target distribution in this test split. A small number of expensive properties can exert considerable influence on the mean and squared-error-based metrics.

### Implications for modelling

1. A single global error metric may conceal substantial differences in performance across price ranges.
2. Linear models may struggle to represent nonlinear relationships and interactions.
3. Tree-based models can represent nonlinear decision boundaries, but they may still struggle with prices beyond the range represented adequately in training data.
4. Large errors on expensive properties can dominate MSE and RMSE.
5. A model's performance should be examined across cities, locality types, and price groups rather than assessed using R² alone.

The reported summary is an experimental snapshot and should be regenerated from the current dataset if the split, preprocessing, or target definition changes.

---

## Data Preprocessing and Feature Engineering

### Numerical and categorical features

Numerical features describe quantities such as property area, age, and distance. Categorical features describe properties such as city, locality type, property type, and furnishing status.

These groups require different preprocessing operations.

**Numerical features**

- Inspect distributions and extreme values.
- Investigate missing or invalid observations.
- Scale features when required by the algorithm.
- Examine the effect of feature removal and transformation.

**Categorical features**

- Identify categorical columns explicitly.
- Encode categories without introducing unintended ordinal relationships.
- Keep the transformed feature order consistent between training and inference.
- Handle unseen categories if the final pipeline must support new properties.

### Feature scaling

Standardization was used in the preprocessing workflow and is particularly relevant for algorithms that depend on distances or feature magnitudes.

For a feature \(x\), standardization is defined as

\[
z = \frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}
\]

where the mean and standard deviation are estimated from the training data.

Tree-based models generally do not require standardization, although using a consistent pipeline can simplify comparisons when its effects are controlled.

### One-hot encoding

Categorical features are converted into numerical indicator columns.

For example, a city feature may produce columns such as `City_Mumbai` and `City_Delhi`. A category is represented by its corresponding indicator rather than an arbitrary numerical ranking.

The transformed feature matrix depends on the categories present and the encoder configuration.

### Recorded 19-feature configuration

One processed feature configuration contained the following 19 base features:

```text
Super_Area_SqFt
Age_of_Property
Distance_to_City_Center_km
City_Bangalore
City_Chandigarh
City_Chennai
City_Delhi
City_Hyderabad
City_Jaipur
City_Kolkata
City_Mumbai
City_Pune
Locality_Type_Mid-Range
Locality_Type_Premium
Property_Type_Builder Floor
Property_Type_Independent House
Property_Type_Villa
Furnishing_Status_Semi-Furnished
Furnishing_Status_Unfurnished
```

This is one experimental configuration, not a universal definition of the final feature set. Other experiments used additional features, including carpet area, or different combinations of encoded columns.

A bias column was also added in some manual implementations. The number of model parameters therefore depends on the feature configuration and whether the intercept is represented explicitly.

### Feature selection experiments

Several experiments investigated the effect of removing or retaining variables, including:

- Bathrooms
- BHK
- Floor number
- Total floors
- Parking
- Gated community
- Carpet area

The purpose was to assess whether individual variables improved generalization or introduced redundancy, noise, or instability.

A feature should not be removed solely because it correlates with another feature. Its effect should be assessed through controlled validation experiments, with preprocessing and other model settings held constant.

### Target transformations

Experiments also investigated the possibility of transforming the target distribution.

A log transformation can reduce the influence of very large prices, but it changes the optimization objective and the scale on which errors are measured. Predictions must be transformed back carefully before being evaluated in the original currency units.

Results from raw-price and transformed-target experiments should be reported separately.

---

## Machine Learning Models

The project covers several families of regression algorithms. The implementations below are based on the algorithms and experiments documented during development.

### 1. Linear Regression

**Implementation:** Manual implementation using the Normal Equation.

Linear regression models the target as a linear combination of input features:

\[
\hat{y}=X\theta
\]

The Normal Equation estimates the parameters by minimizing the sum of squared residuals. In a full-rank setting, its familiar form is

\[
\theta=(X^\top X)^{-1}X^\top y
\]

Direct matrix inversion is not numerically ideal in all circumstances. A practical implementation should consider solving the linear system directly or using a numerically stable least-squares method.

**What this model provides**

- A simple reference for nonlinear models.
- An interpretable relationship between input features and predictions.
- An opportunity to study least squares, matrix operations, multicollinearity, and numerical stability.

A historical experiment reported R² of approximately 0.78. This is an earlier result, not a verified final score for the current pipeline.

### 2. Decision Tree Regression

**Implementation:** Manual tree construction.

The tree recursively partitions the feature space using feature thresholds. Each terminal node returns a prediction based on the training observations assigned to that region.

Parameters investigated include:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

**Key observations**

- Increasing depth initially improved predictive performance.
- Performance peaked around depth 8 in one recorded experiment.
- Increasing complexity beyond the useful range can lead to overfitting.
- Leaf-size and split-size constraints influence the bias-variance trade-off.

The implementation was developed in `src/Tree_Regression.py`.

### 3. Random Forest Regression

**Implementation:** Manual ensemble built using decision tree regressors.

Random Forest combines predictions from multiple trees trained on bootstrap samples. Feature subsampling can reduce correlation between individual trees.

The project implemented bootstrap sampling and investigated the effects of ensemble size, tree depth, minimum split size, minimum leaf size, and feature selection.

**Recorded configuration**

| Hyperparameter | Value |
|---|---:|
| Number of estimators | 150 |
| Maximum depth | 15 |
| Minimum samples to split | 50 |
| Minimum samples per leaf | 10 |
| Maximum features | 20 |

The corresponding recorded test results were MAE 33.4593, RMSE 69.1857, and R² 0.783801.

This is one of the stronger recorded experiments. It should be treated as the leading historical result only until all models are evaluated on a common, unchanged test set.

### 4. Gradient Boosting Regression

**Implementation:** Manual sequential boosting using regression trees.

Gradient Boosting builds an ensemble sequentially, with later trees learning to reduce the residual errors or loss gradients left by earlier predictions.

The experiments investigated:

- Number of estimators.
- Learning rate.
- Tree depth.
- Minimum leaf size.
- Minimum split size.

One documented experiment with 310 estimators produced:

| Metric | Value |
|---|---:|
| MAE | 35.3533 |
| MSE | 4,772.2360 |
| RMSE | 69.0814 |
| R² | 0.778705 |

A later tuning experiment identified a promising configuration with 690 estimators, learning rate 0.2, maximum depth 2, and minimum leaf size 6. The complete metrics for that configuration were not established in the available project record.

The 310-estimator result must not be presented as the result of the 690-estimator configuration.

### 5. K-Nearest Neighbors Regression

**Implementation:** Manual KNN regression.

KNN predicts a target using the target values of nearby training observations. The selected neighbors are determined by a distance function, and their target values are aggregated to produce a prediction.

The implementation required particular attention to NumPy and pandas indexing when mapping nearest-neighbor indices back to target values.

A later experiment using a reduced feature configuration recorded:

| Metric | Value |
|---|---:|
| MAE | 34.7310 |
| MSE | 5,088.3239 |
| RMSE | 71.3325 |
| R² | 0.770175 |

An earlier experiment using 10 neighbors performed less effectively, with MAE approximately 50.161 and R² approximately 0.579. The results illustrate the importance of feature selection and distance calculations for KNN.

### 6. Support Vector Regression (SVR)

**Implementation:** Manual optimization experiments.

SVR seeks a function that balances model complexity with prediction errors outside an epsilon-insensitive tolerance. Unlike ordinary least squares, errors within the epsilon tube do not contribute to the standard epsilon-insensitive loss.

The experiments investigated:

- Learning rate.
- Epsilon.
- Regularization parameter \(C\).
- Number of iterations.
- Optimization convergence behavior.
- Support-vector counts.

The project explored subgradient-based optimization and iterative parameter updates rather than treating a library implementation as a black box.

A recent recorded experiment produced:

| Metric | Value |
|---|---:|
| MAE | 43.3410 |
| MSE | 8,596.4366 |
| RMSE | 92.7170 |
| R² | 0.611724 |
| Iterations used | 84,142 |
| Maximum iterations | 85,000 |
| Support vectors | 56,357 |

These figures describe the recorded SVR experiment. The iteration count alone does not establish that the optimization converged successfully; convergence criteria and termination status should also be checked.

SVR has so far been less competitive than the strongest recorded tree ensemble, although comparisons must use identical data splits, feature configurations, and target scales to support a definitive ranking.

---

## Model Performance

The following table summarizes the recorded experiments available in the project history.

### Historical results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Decision Tree Regression — selected depth-8 experiment | 38.5983 | — | 75.3679 | 0.743436 |
| Random Forest Regression — recorded configuration | 33.4593 | — | 69.1857 | 0.783801 |
| Gradient Boosting — 310 estimators | 35.3533 | 4,772.2360 | 69.0814 | 0.778705 |
| KNN Regression — later feature configuration | 34.7310 | 5,088.3239 | 71.3325 | 0.770175 |
| SVR — recent optimization experiment | 43.3410 | 8,596.4366 | 92.7170 | 0.611724 |

A dash indicates that the corresponding metric was not available in the recorded result used for this table.

**Interpretation:** Random Forest has the strongest recorded combination of MAE and R² among the listed experiments. Gradient Boosting has a slightly lower RMSE in its recorded run, but its MAE and R² are weaker. These results are useful for identifying promising candidates, not for declaring a definitive winner across all experiments.

### Why these results are provisional

The project has used different feature sets and model configurations across experiments. Some models were also evaluated during separate stages of development.

A rigorous final comparison requires:

1. The same training, validation, and test observations.
2. A consistent target definition and target scale.
3. Consistent metric implementations.
4. Training-only fitting of all preprocessing transformations.
5. A documented feature configuration for every model.
6. A clear distinction between validation results used for tuning and final test results.
7. Recorded hyperparameters, random seeds where applicable, and experiment identifiers.

The final README should be updated after a common evaluation run establishes the comparable results.

---

## Error Analysis and Findings

Overall model scores do not fully describe prediction quality. Error analysis has been used to examine systematic differences across price ranges, locality categories, and cities.

### Error by locality type

A recorded error-analysis experiment produced the following mean absolute errors:

| Locality type | MAE (₹ lakhs) |
|---|---:|
| Affordable | 21.388 |
| Mid-Range | 37.071 |
| Premium | 104.043 |

The much higher error for premium properties indicates that the model has greater difficulty predicting prices in this segment.

Possible contributing factors include greater price variability, high-value outliers, nonlinear price relationships, and insufficient representation of expensive properties. These are hypotheses to investigate rather than established causal explanations.

### Error by price group

Another recorded experiment produced:

| Price group (₹ lakhs) | MAE (₹ lakhs) |
|---|---:|
| 0–50 | 23.739 |
| 50–100 | 24.524 |
| 100–200 | 43.073 |
| 200–500 | 83.979 |
| 500+ | 335.810 |

Absolute prediction error rises substantially in the more expensive price groups.

This does not necessarily mean that relative accuracy deteriorates at the same rate. A model can have a large absolute error on a high-priced property but a smaller percentage error than on a cheaper property. Both absolute and relative error measures should therefore be considered, with appropriate care around zero or near-zero target values.

### City-level performance

Mumbai was identified as a particularly challenging segment in the recorded analysis. Some actual property prices exceeded ₹2,000 lakh, contributing to large prediction errors.

This finding supports further investigation of:

- City-specific target distributions.
- Extreme-value observations.
- Sample counts within high-price segments.
- Model behavior outside densely represented price ranges.
- Whether a single global model is adequate for all cities.

It does not, by itself, establish that separate city-specific models would outperform a shared model.

### Prediction distribution

One historical prediction run had these summary statistics:

| Statistic | Actual target | Predicted target |
|---|---:|---:|
| Mean | 142.787 | 135.355 |
| Standard deviation | 148.800 | 94.588 |
| Minimum | 5.000 | 18.610 |
| Maximum | 3,216.464 | 772.190 |

The predicted distribution was narrower than the actual target distribution, and its maximum was substantially lower.

This is consistent with underprediction of some high-priced properties and limited coverage of the extreme upper tail. The summary is from a historical run, however, and should not be attributed to every model.

### Key analytical findings

- Housing prices have a broad and right-skewed distribution.
- Expensive properties are more difficult to predict accurately in absolute terms.
- Model performance depends on feature selection and preprocessing.
- Tree ensembles are promising candidates for this dataset.
- High overall R² does not eliminate significant errors for individual properties.
- Price-range and locality-level evaluation adds information that aggregate metrics cannot provide.

---

## Implementation Details

The project emphasizes understanding the mechanics of the algorithms and building a modular codebase.

### Manual algorithm development

The manual implementations are intended to make the underlying operations explicit, including:

- Matrix-based parameter estimation for linear regression.
- Recursive splitting and leaf prediction in decision trees.
- Bootstrap sampling and ensemble aggregation in Random Forest.
- Sequential residual correction in Gradient Boosting.
- Distance calculation and neighbor aggregation in KNN.
- Iterative optimization for SVR.

The scope of manual implementation can differ by component. Using NumPy for numerical operations or established utilities for data transformation does not negate a from-scratch model implementation, provided the core learning algorithm itself is implemented by the project.

### Reusable preprocessing

The preprocessing modules separate data preparation from model-specific code.

A reusable preprocessing workflow should consistently:

1. Load the dataset.
2. Remove unsuitable identifier columns.
3. Separate features and target.
4. Create data splits.
5. Fit transformations on training data.
6. Transform validation and test data using the fitted transformations.
7. Preserve the order and names of processed features.
8. Return appropriately aligned arrays or DataFrames.

### Evaluation utilities

Regression evaluation uses the following metrics.

**Mean Absolute Error (MAE)**

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

MAE expresses the average absolute prediction error in the target's units.

**Mean Squared Error (MSE)**

\[
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

MSE penalizes larger errors more heavily because the residuals are squared.

**Root Mean Squared Error (RMSE)**

\[
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
\]

RMSE is expressed in the same units as the target and is more sensitive to large errors than MAE.

**Coefficient of determination (R²)**

\[
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat y_i)^2}
{\sum_{i=1}^{n}(y_i-\bar y)^2}
\]

R² measures performance relative to predicting the evaluation-set mean. It can be negative when a model performs worse than that reference predictor.

No single metric is sufficient for every use case. The project uses multiple metrics to understand average error, sensitivity to outliers, and overall fit.

---

## Installation and Setup

### Prerequisites

- Python 3.x
- pip
- Jupyter Notebook or JupyterLab
- Git, if cloning the repository

The exact minimum Python version and package versions should be documented once the environment has been finalized.

### 1. Clone the repository

Replace the placeholder below with the actual repository URL.

```bash
git clone <YOUR_REPOSITORY_URL>
cd House-Price-Prediction
```

### 2. Create a virtual environment

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

### 3. Install dependencies

If the repository contains a maintained `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency list should reflect the actual imports in the project, including the numerical, data-processing, visualization, notebook, and evaluation packages used by the implementation.

### 4. Prepare the dataset

Place the downloaded training dataset at the path expected by the data loader:

```text
data/train.csv
```

The dataset is not assumed to be bundled with the repository. Follow the dataset's licensing and redistribution terms.

### 5. Verify the environment

```bash
python --version
python -m pip list
```

If imports fail, check that the virtual environment is active and that the installed package versions are compatible with the project.

---

## Running the Project

The exact execution commands depend on the current repository structure and the entry points implemented in the source modules.

### Explore the dataset

```bash
jupyter notebook
```

Open `01_data_exploration.ipynb` and run the cells in order.

### Review data cleaning

Open `02_data_cleaning.ipynb` and execute the cleaning workflow after confirming the expected input and output paths.

### Run model experiments

Open `03_model_training.ipynb` and execute the model-training and evaluation cells.

If the repository contains additional model-specific notebooks or scripts, use their documented entry points.

**Important:** The commands above describe the notebook-based workflow. They do not imply that a unified command-line training pipeline or a deployed prediction application has already been implemented.

---

## Results and Experiment Tracking

The project records model configurations and metrics so that different experiments can be compared and reproduced.

A useful experiment record should contain:

| Field | Purpose |
|---|---|
| Model | Algorithm name |
| Target | Raw or transformed target |
| Feature configuration | Predictors used by the experiment |
| Hyperparameters | Model settings |
| Train metrics | Training-set performance |
| Validation metrics | Performance used for model selection |
| Test metrics | Final evaluation on held-out data |
| Random seed | Reproducibility where relevant |
| Experiment notes | Observations, warnings, or changes |

The project has discussed recording results in `report/model_results.csv`. Its presence and current schema should be verified in the repository.

For reliable comparisons, each experiment should have a distinct record rather than overwriting previous configurations. The README's performance table should be updated only when the corresponding metric values and evaluation setup are known.

---

## Challenges and Lessons Learned

Developing the project has exposed practical issues that are not apparent from studying algorithm equations alone.

### 1. Data indexing and alignment

Combining NumPy index arrays with pandas Series can lead to unexpected indexing behavior. Positional indices and pandas index labels are not interchangeable.

The implementation must ensure that sampled rows and target values remain correctly aligned, particularly when creating bootstrap datasets and selecting nearest neighbors.

### 2. Feature consistency

One-hot encoding and feature selection can produce matrices with different numbers of columns. A model trained with one feature configuration cannot safely be evaluated using an incompatible matrix.

Feature ordering and transformation state must be preserved throughout training and inference.

### 3. Overfitting in decision trees

Increasing tree depth can improve training performance while harming generalization. Depth, leaf size, and split constraints must be tuned together.

### 4. Computational cost

Manual implementations can be expensive on datasets containing tens of thousands of observations. The Gradient Boosting experiments demonstrated that extensive hyperparameter searches can require substantial runtime.

Future tuning should use validation-based selection, bounded search spaces, sensible stopping criteria, and efficient numerical operations.

### 5. Outliers and skewed targets

A small number of expensive properties can dominate squared-error metrics. Overall model performance should be supplemented with price-segment and locality-level error analysis.

### 6. Numerical stability

Matrix operations, poorly scaled inputs, redundant features, and optimization settings can affect stability and convergence. A historical condition-number observation of approximately 231.14 motivated further attention to feature scaling and redundancy, but this value should not be interpreted as a universal property of every processed dataset.

### 7. Evaluation discipline

Results from different experiments are not automatically comparable. Changes in the feature set, target transformation, data split, and hyperparameters can all affect reported metrics.

A controlled experiment log is essential for selecting a model based on evidence rather than isolated scores.

---

## Limitations

The project has several limitations that should be considered when interpreting the results.

1. **Dataset representativeness:** The dataset may not reflect every Indian housing market, property category, or current market condition.
2. **Extreme-value prediction:** The most expensive properties remain challenging, particularly when their characteristics are poorly represented in training data.
3. **Feature limitations:** Important pricing variables may be unavailable, incomplete, or insufficiently detailed.
4. **Model comparability:** Historical results were generated across different experimental configurations and require a common evaluation run before final ranking.
5. **Generalization:** Good performance on a random held-out split does not guarantee equivalent performance on future listings or unseen cities.
6. **Manual implementation trade-offs:** From-scratch implementations are valuable for learning, but may lack the optimizations, numerical safeguards, and edge-case handling of mature production libraries.
7. **Deployment status:** A production API, web interface, automated retraining system, and monitoring pipeline should not be considered complete unless they are explicitly implemented and tested.

The outputs are intended for machine learning experimentation and educational purposes. They should not be treated as certified property valuations or financial advice.

---

## Future Work

The roadmap is designed to extend the existing work without treating planned experiments as completed implementations.

### Phase 1 — Consolidate the current project

- [ ] Verify the final repository structure and remove duplicated preprocessing logic.
- [ ] Standardize the model interfaces.
- [ ] Establish a reproducible train/validation/test workflow.
- [ ] Ensure every model uses a documented feature configuration.
- [ ] Finalize a baseline and record its metrics.
- [ ] Run the current models under a common evaluation protocol.
- [ ] Consolidate the results into a consistent experiment log.
- [ ] Add automated tests for data loading, transformations, and model predictions.

### Phase 2 — Improve the current models

- [ ] Investigate the high-price prediction errors.
- [ ] Evaluate alternative feature configurations systematically.
- [ ] Compare raw-target and log-target experiments.
- [ ] Tune model hyperparameters using validation data or cross-validation.
- [ ] Investigate regularization for linear regression.
- [ ] Examine feature importance and model interpretability.
- [ ] Evaluate relative errors alongside absolute errors.
- [ ] Analyze residuals for systematic patterns.

### Phase 3 — Extend the algorithm portfolio

Potential future models include:

- [ ] Ridge Regression — study L2 regularization and coefficient stability.
- [ ] Lasso Regression — investigate L1 regularization and feature selection.
- [ ] AdaBoost Regression — study sequential reweighting and ensemble learning.
- [ ] XGBoost — compare an optimized gradient-boosting implementation against the manual version.
- [ ] HistGradientBoosting or another efficient boosting implementation — evaluate computational efficiency and predictive performance.
- [ ] Kernel Ridge Regression — investigate nonlinear kernel-based regression.
- [ ] Improved SVR optimization — investigate convergence, computational cost, and regularization.
- [ ] Robust regression — assess methods less sensitive to extreme observations.

These are proposed extensions, not claims that the algorithms have already been completed.

### Phase 4 — Improve software engineering

- [ ] Create a consistent configuration system for experiments.
- [ ] Separate data preparation, model training, evaluation, and reporting.
- [ ] Add unit tests and integration tests.
- [ ] Record random seeds and environment information.
- [ ] Add structured logging and error handling.
- [ ] Create a repeatable training entry point.
- [ ] Save model artifacts and the preprocessing objects required for inference.
- [ ] Document the model-selection procedure.

### Phase 5 — Deployment and practical use

- [ ] Build a prediction interface for property characteristics.
- [ ] Add input validation and consistent feature transformations.
- [ ] Expose predictions through a simple API or web application.
- [ ] Document model limitations and prediction uncertainty.
- [ ] Evaluate performance on new data when available.
- [ ] Consider monitoring data drift and model degradation.

Deployment is a future objective unless the corresponding components are implemented and validated.

---

## Contributing

Contributions and suggestions are welcome.

A useful contribution should:

1. Describe the problem or improvement.
2. Explain the relevant algorithm or design decision.
3. Include reproducible steps where possible.
4. Preserve the distinction between training, validation, and test data.
5. Include appropriate tests for changes to reusable modules.
6. Report experimental results with the associated feature configuration and hyperparameters.

For major algorithm changes, comparisons should use the same evaluation protocol as the existing implementation.

---

## License

A license has not been specified in the project information available for this README.

Before publishing the repository, add a `LICENSE` file that reflects the intended reuse terms. Also verify the dataset's license and redistribution conditions separately; a code license does not automatically grant rights to redistribute third-party data.

---

## Acknowledgements

- **Kaggle** — for hosting the Indian Urban House Price Prediction dataset.
- The open-source Python ecosystem — for numerical computation, data processing, visualization, and notebook-based development.
- Educational resources on statistics, regression, optimization, and machine learning that support the theoretical and practical study of these algorithms.

---

## Project Status

**Current stage: Model development, hyperparameter experimentation, and evaluation consolidation.**

The project has progressed through data exploration, preprocessing, and multiple regression implementations, including Linear Regression, Decision Tree Regression, Random Forest Regression, Gradient Boosting Regression, KNN Regression, and SVR experiments.

The next priority is to establish a reproducible and comparable evaluation of the existing models, investigate errors on expensive properties, and finalize the project structure before extending the model portfolio or developing a deployment interface.

This README will evolve alongside the repository. Model results, implementation status, and the directory structure should be updated when the corresponding changes have been completed and verified.
