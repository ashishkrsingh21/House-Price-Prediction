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



    return (X_train_preprocessed, X_test_preprocessed, y_train, y_test, X_train, X_test)