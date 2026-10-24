import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def fit_and_transform_features(X_train, X_test):
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    scaler = StandardScaler()
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    X_train_numeric = scaler.fit_transform(
        X_train[numeric_features]
    )
    X_test_numeric = scaler.transform(
        X_test[numeric_features]
    )

    X_train_categorical = encoder.fit_transform(
        X_train[categorical_features]
    ).toarray()

    X_test_categorical = encoder.transform(
        X_test[categorical_features]
    ).toarray()

    X_train_processed = np.hstack([
        X_train_numeric,
        X_train_categorical
    ])

    X_test_processed = np.hstack([
        X_test_numeric,
        X_test_categorical
    ])

    return X_train_processed, X_test_processed, encoder