import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


def get_feature_names():

    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "data" / "train.csv"

    house_data = pd.read_csv(train_path)
    house_data = house_data.drop(columns=["Property_ID", "Price_INR_Lakhs"])

    numerical_features = house_data.select_dtypes(include="number").columns.tolist()
    categorical_features = house_data.select_dtypes(include="object").columns.tolist()

    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    encoder.fit(house_data[categorical_features])

    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    processed_features_name = list(numerical_features).copy() + list(encoded_feature_names).copy()

    return numerical_features, categorical_features, processed_features_name
