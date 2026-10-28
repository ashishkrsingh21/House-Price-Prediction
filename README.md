
# House Price Prediction

A supervised machine learning regression project for predicting residential property prices across urban India. The project covers exploratory data analysis, data preprocessing, feature engineering, regression modelling, hyperparameter tuning, experiment tracking, and error analysis, with an emphasis on implementing core regression algorithms from scratch.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Repository Structure](#repository-structure)
- [Implemented Models](#implemented-models)
- [Model Evaluation and Results](#model-evaluation-and-results)
- [Setup and Usage](#setup-and-usage)
- [Experiment Tracking](#experiment-tracking)
- [Limitations and Future Work](#limitations-and-future-work)
- [License and Dataset Attribution](#license-and-dataset-attribution)

## Project Overview

House prices depend on multiple interacting factors, including property size, location, property type, age, and available amenities. This project investigates how these characteristics can be used to estimate property prices using supervised regression.

The primary objective is to understand and implement regression algorithms, compare their predictive performance, and examine where their errors occur. Rather than relying exclusively on prebuilt estimators, the project includes manual implementations of several core learning algorithms.

The work is organized around three notebooks:

1. **Data exploration:** Understand the dataset, feature distributions, relationships, and outliers.
2. **Data cleaning:** Prepare the data and construct a consistent feature representation.
3. **Model training:** Train regression models, experiment with hyperparameters and feature configurations, and evaluate predictions.

The project also includes reusable preprocessing, evaluation, and experiment-logging modules.

## Dataset

**Source:** [Indian Urban House Price Prediction — Kaggle](https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data)

The Kaggle competition describes a synthetic dataset representing residential properties across major Indian cities. It was generated for educational machine learning and regression experiments rather than collected as a direct record of actual property transactions.

### Dataset characteristics

| Property | Description |
|---|---|
| Dataset source | Kaggle competition |
| Total records | 100,000, according to the competition description |
| Training file | `train.csv` |
| Competition test file | `test.csv` |
| Target variable | `Price_INR_Lakhs` |
| Target unit | Indian rupees, expressed in lakhs |
| Competition evaluation metric | Root Mean Squared Error (RMSE) |
| Data type | Synthetic, structured tabular data |

The project experiments previously used a separate 80/20 modelling split containing 64,000 training observations and 16,000 test observations. These figures describe that experimental split, not the original competition's training and test files.

### Original dataset features

The original competition schema includes the following feature groups.

| Feature group | Representative columns |
|---|---|
| Location | `City`, `Locality_Type` |
| Geographic accessibility | `Distance_to_Metro_km`, `Distance_to_City_Center_km` |
| Property characteristics | `Property_Type`, `BHK`, `Bathrooms` |
| Area | `Super_Area_SqFt`, `Carpet_Area_SqFt` |
| Building information | `Floor_Number`, `Total_Floors`, `Age_of_Property` |
| Amenities | `Furnishing_Status`, `Parking`, `Lift_Available`, `Gated_Community` |
| Identifier | `Property_ID` |
| Target | `Price_INR_Lakhs` |

`Property_ID` is an identifier rather than a predictive property characteristic. The target, `Price_INR_Lakhs`, must not be used as an input feature.

### Processed modelling features

The project has experimented with numerical features and encoded categorical variables. One recorded feature configuration contained 28 base features:

- `BHK`
- `Bathrooms`
- `Floor_Number`
- `Total_Floors`
- `Super_Area_SqFt`
- `Carpet_Area_SqFt`
- `Parking`
- `Age_of_Property`
- `Lift_Available`
- `Gated_Community`
- `Distance_to_Metro_km`
- `Distance_to_City_Center_km`
- Encoded city categories
- Encoded locality categories
- Encoded property types
- Encoded furnishing categories

The categorical features were represented using indicator columns. For example, `City_Bangalore` and `Locality_Type_Premium` are processed feature names rather than original dataset columns.

The project also experimented with additional features and feature-removal strategies. Consequently, the 19-feature configuration should not be interpreted as the input configuration for every experiment.

**Data considerations:** The target distribution is right-skewed, with high-priced properties contributing disproportionately to squared prediction errors. Outliers, differences between cities, and variations across locality types are important considerations when interpreting evaluation metrics.

## Project Workflow

### 1. Exploratory Data Analysis

The exploration stage examines the dataset's structure, feature distributions, missing values, relationships between variables, and price variation across property characteristics.

Particular attention is given to the target distribution, location-related differences, and unusually expensive properties that may affect model performance.

### 2. Data Cleaning and Preprocessing

The preprocessing workflow prepares numerical and categorical variables for modelling. The project separates feature preparation from model implementation through reusable modules.

Depending on the experiment, preprocessing includes:

- Handling missing or unsuitable data.
- Separating numerical and categorical features.
- Encoding categorical variables.
- Scaling numerical inputs where appropriate.
- Constructing train and test feature matrices.
- Maintaining a consistent feature representation between datasets.

Scaling and encoding requirements differ between algorithms. Any comparison should account for the preprocessing used by each experiment.

### 3. Feature Engineering and Selection

Alternative feature configurations were evaluated to understand how input selection affects regression performance. Experiments included removing features such as bathroom count, bedroom count, floor information, parking, and gated-community indicators.

Polynomial transformations were also explored, but the recorded experiments did not consistently show improvements.

Features derived directly from the target, such as price per square foot calculated using the property's actual price, must not be supplied as predictors when evaluating price prediction.

### 4. Model Training and Hyperparameter Tuning

The modelling stage includes linear regression, tree-based methods, nearest-neighbour regression, and support vector regression. Hyperparameters were explored to investigate the trade-offs between predictive accuracy, model complexity, and generalization.

### 5. Evaluation and Error Analysis

Regression performance is assessed using numerical metrics and prediction-error analysis. Beyond aggregate scores, the project examines errors across price ranges, locality types, and other property categories.

This helps identify whether a model performs consistently or struggles with expensive properties and particular segments of the dataset.

## Repository Structure

The following is the intended structure based on the project layout supplied for this documentation. The exact contents of the repository have not been independently inspected.

```text
House-Price-Prediction/
├── data/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_model_training.ipynb
├── report/
│   └── model_results.csv
├── src/
│   ├── evaluation/
│   │   ├── error_analysis.py
│   │   └── metrics.py
│   ├── models/
│   │   ├── decision_tree_regression.py
│   │   ├── gradient_boosting_regression.py
│   │   ├── knn_regression.py
│   │   ├── linear_regression.py
│   │   ├── random_forest_regression.py
│   │   └── svr.py
│   ├── preprocessing/
│   │   ├── data_loader.py
│   │   ├── feature_names.py
│   │   └── transformer.py
│   └── utils/
│       └── experiment_logger.py
├── test/
├── .gitignore
├── LICENSE
└── README.md
```

### Module responsibilities

| Location | Responsibility |
|---|---|
| `notebooks/` | Exploratory analysis, data cleaning, and model experiments |
| `src/preprocessing/` | Dataset loading, feature transformation, and encoded feature names |
| `src/models/` | Regression algorithm implementations |
| `src/evaluation/` | Regression metrics and prediction-error analysis |
| `src/utils/` | Experiment logging utilities |
| `report/model_results.csv` | Recorded model experiment results |
| `data/` | Dataset files used by the notebooks and preprocessing workflow |
| `test/` | Intended location for automated tests |

The directory tree documents the supplied layout; it does not establish that every listed file currently exists or that every module is fully tested.

## Implemented Models

The project history documents implementation work on the following algorithms. The descriptions below summarize the recorded approaches; the exact current implementation status should be confirmed against the source code.

### Linear Regression

A manual Normal Equation implementation was used to estimate model parameters by solving the least-squares problem. This provides a direct baseline for understanding linear relationships between features and the target.

The approach is useful for comparison but cannot, by itself, capture arbitrary nonlinear relationships.

### Decision Tree Regression

The project implements a regression tree using feature-based splits and numerical prediction at the leaves. Important hyperparameters include maximum depth and minimum sample requirements for splitting and leaf nodes.

Tree depth experiments demonstrated the trade-off between fitting the training data and generalizing to unseen observations.

### Random Forest Regression

The manual Random Forest implementation builds an ensemble of decision trees using bootstrap sampling. Aggregating predictions from multiple trees can reduce the variance of an individual tree.

The project history includes experiments with tree count, depth, minimum sample constraints, and the number of candidate features.

### Gradient Boosting Regression

The manual Gradient Boosting implementation builds an additive sequence of regression trees. Each successive tree is trained to improve the current ensemble's predictions.

Experiments investigated the number of estimators, learning rate, tree depth, and minimum sample settings. Tuning was computationally expensive for larger estimator counts.

### K-Nearest Neighbors Regression

The manual KNN regressor predicts a property's price using nearby training observations in feature space. Its performance depends on the selected neighborhood size, feature representation, and scaling.

Because distance calculations are sensitive to feature scales, preprocessing is particularly important for this method.

### Support Vector Regression (SVR)

The project includes manual SVR optimization experiments involving the regularization parameter, epsilon-insensitive loss, learning rate, and iterative optimization.

SVR is particularly sensitive to feature scaling and optimization settings. The recorded experiments required substantial iteration counts and did not establish that the implementation outperformed the other models.

These implementations are intended to support algorithmic understanding and controlled experimentation. Their presence should not be interpreted as evidence that every model has identical convergence guarantees, numerical stability, or production-level performance.

## Model Evaluation and Results

The project evaluates regression models using several complementary metrics.

| Metric | Interpretation |
|---|---|
| MAE | Mean absolute prediction error, in the target's units |
| RMSE | Square root of MSE, expressed in the target's units |
| R² | Proportion of target variance explained relative to a mean-prediction baseline |

For this dataset, prices are expressed in lakhs of Indian rupees. Therefore, an MAE of 35 corresponds to an average absolute error of approximately ₹35 lakh, assuming the target has not been transformed for evaluation.

### Historical experimental results

The following values are available in the recorded project history. They are included as historical reference points, not as a verified current leaderboard.

| Model / experiment | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression - recorded feature configuration | 32.1584 | 68.3593 | 0.7889
| Decision Tree — depth 8 | 38.5983 | 75.3679 | 0.7434 |
| Random Forest — recorded configuration | 33.4593 | 69.1857 | 0.7838 |
| Gradient Boosting — 310 estimators | 35.3533 | 69.0814 | 0.7787 |
| KNN — recorded feature configuration | 34.7310 | 71.3325 | 0.7702 |
| SVR — ongoing

**Comparability limitation:** These results come from historical experiments. The available history does not establish that all rows use the same feature set, preprocessing pipeline, target representation, split, or evaluation protocol. The table must not be used as a definitive model ranking.

The Gradient Boosting experiment has a slightly lower RMSE than the recorded Random Forest experiment, while the Random Forest experiment has a lower MAE and higher R². This illustrates why model selection should consider the evaluation objective and the comparability of the underlying experiments.

### Error analysis

Recorded analyses identified several important patterns:

- **Locality differences:** Historical mean absolute error was approximately 21.39 for Affordable, 37.07 for Mid-Range, and 104.04 for Premium properties in one experiment.
- **Price-range differences:** Errors increased substantially for higher-priced properties. One recorded analysis reported an MAE of approximately 335.81 for properties in the 500+ price group.
- **Extreme values:** Very expensive properties, including observations associated with Mumbai, contributed to large prediction errors.
- **Underprediction:** Earlier experiments showed that some models did not adequately capture the upper end of the target distribution.

These findings suggest that a single aggregate metric is insufficient to characterize performance. Segment-level evaluation and inspection of large residuals are necessary to understand model weaknesses.

The current `report/model_results.csv` should be used to verify the latest metrics and establish a final comparison using a common evaluation protocol.

## Setup and Usage

The repository's dependency files and execution commands have not been verified. The following instructions describe a conventional local notebook workflow; they are not a claim that the repository has been tested from a clean environment.

### 1. Obtain the repository and dataset

Clone the repository using its actual GitHub URL:

```bash
git clone git@github.com:ashishkrsingh21/House-Price-Prediction.git
cd House-Price-Prediction
```

Download the dataset from the [official Kaggle competition data page](https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data).

Place the required CSV files in the locations expected by the data-loading code. Confirm the actual file paths before running the notebooks; the loader's configured path must match the local directory structure.

### 2. Prepare a Python environment

Create and activate a virtual environment appropriate to your operating system. Install the dependencies specified by the repository's dependency file, if available.

For a repository without a dependency file, inspect the imports before installing the required packages. The project history indicates the use of common Python data-science tools, but the complete dependency list and compatible versions have not been verified.

### 3. Run the notebooks

Open the repository in Jupyter or another notebook-compatible environment and execute the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_model_training.ipynb`

Before executing the training notebook, verify that the dataset paths, preprocessing configuration, feature matrices, and target arrays are consistent.

### 4. Evaluate and record experiments

Use the evaluation functions and model implementations referenced by the training notebook. Check the existing experiment-logging interface before invoking it; a standalone command-line training interface has not been established by the available project information.

## Experiment Tracking

The project includes `report/model_results.csv` as a record of model experiments and `src/utils/experiment_logger.py` as the associated logging utility.

The intended purpose of this arrangement is to preserve results across experiments and make model comparisons easier without relying exclusively on notebook output.

The exact CSV schema, logger arguments, and recorded metadata have not been independently verified. Consult the implementation before assuming that it records particular metrics, hyperparameters, feature configurations, or target transformations.

For reliable comparisons, each recorded experiment should identify its model, feature configuration, preprocessing, target representation, evaluation split, hyperparameters, and evaluation metrics. These fields are recommendations for improving experiment reproducibility, not a claim that all are currently logged.

## Limitations and Future Work

### Current limitations

- **Synthetic data:** The Kaggle competition dataset is synthetic and should not be treated as evidence of performance on actual real-estate transactions.
- **Outlier sensitivity:** High-priced properties can dominate MSE and RMSE and produce substantial differences between aggregate and segment-level performance.
- **Experiment comparability:** Historical metrics may reflect different feature configurations and evaluation settings.
- **Optimization cost:** Manual tree ensembles and iterative SVR experiments can require substantial computation.
- **Reproducibility:** A verified dependency specification, consistent experiment protocol, and clean-environment execution check are needed before the workflow can be considered fully reproducible.
- **Deployment:** A production prediction API or user-facing application is not documented as part of the current project.

### Proposed next steps

1. Verify and consolidate the latest results in `report/model_results.csv`.
2. Establish one consistent train/validation/test protocol and compare models on the same held-out data.
3. Audit preprocessing for data leakage and ensure that all transformations are fitted using training data only.
4. Add automated tests for preprocessing, model predictions, evaluation metrics, and experiment logging.
5. Profile the computational cost and numerical stability of the manual implementations.
6. Evaluate the selected model on genuinely independent, real-world housing data before making claims about practical applicability.

These are proposed improvements rather than completed features.

## License and Dataset Attribution

### Project license

The repository contains a `LICENSE` file. Confirm its contents before specifying the project's license in this section. The repository license governs the project's own code and does not automatically determine the terms under which the dataset may be redistributed.

### Dataset attribution

The dataset is provided through the [Indian Urban House Price Prediction Kaggle competition](https://www.kaggle.com/competitions/indian-urban-house-price-prediction).

The competition describes the dataset as synthetic and intended for educational machine learning experimentation. Its data page lists an MIT license; verify the current dataset terms and applicable competition rules before redistributing dataset files.

Credit the dataset source when using or adapting the data, and follow the terms displayed on Kaggle. Keep dataset attribution separate from the license governing this repository's code.
