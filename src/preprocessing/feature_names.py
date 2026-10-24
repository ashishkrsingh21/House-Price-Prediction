def get_feature_names(X_train, encoder):
    
    numerical_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    encoded_feature_names = encoder.get_feature_names_out(
        categorical_features
    ).tolist()

    processed_feature_names = (
        numerical_features + encoded_feature_names
    )

    return (
        numerical_features,
        categorical_features,
        processed_feature_names
    )