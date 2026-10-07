import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def prepare_data():

    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "data" / "train.csv"

    house_data = pd.read_csv(train_path)

    X = house_data.drop(columns=["Price_INR_Lakhs"])
    y = house_data["Price_INR_Lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
    )

    X_train = X_train.drop(columns=["Property_ID"])
    X_test = X_test.drop(columns=["Property_ID"])

    numeric_features = [
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km"
    ]

    categorical_features = [
    "City",
    "Locality_Type",
    "Property_Type",
    "Furnishing_Status"
    ]

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_features])

    X_train_numeric_scaled = scaler.transform(X_train[numeric_features])
    X_test_numeric_scaled = scaler.transform(X_test[numeric_features])

    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    encoder.fit(X_train[categorical_features])

    X_train_categorical_encoded = encoder.transform(X_train[categorical_features])
    X_test_categorical_encoded = encoder.transform(X_test[categorical_features])

    X_train_preprocessed = np.hstack([X_train_numeric_scaled, 
                                      X_train_categorical_encoded.toarray()])
    
    X_test_preprocessed = np.hstack([X_test_numeric_scaled, 
                                     X_test_categorical_encoded.toarray()])
    
    encoded_feature_names = encoder.get_feature_names_out(
    categorical_features
    )



    return (X_train_preprocessed, X_test_preprocessed, y_train, y_test, X_train, X_test, house_data)

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

    return numerical_features, categorical_features, encoded_feature_names, processed_features_name

def error_analysis(y_true, y_pred):
    error = y_true - y_pred

    mae_error = np.mean(np.abs(error))
    mse_error = np.mean(error **2)
    rmse_error = np.sqrt(mse_error)

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print("Mean Absolute Error (MAE):", mae_error)
    print("Mean Squared Error (MSE):", mse_error)
    print("Root Mean Squared Error (RMSE):", rmse_error)
    print("R-squared:", r_squared)


def add_bias(X_train, X_test):
    X_train_with_bias = np.c_[
        np.ones(X_train.shape[0]), 
        X_train]

    X_test_with_bias = np.c_[
        np.ones(X_test.shape[0]), 
        X_test]

    return X_train_with_bias, X_test_with_bias