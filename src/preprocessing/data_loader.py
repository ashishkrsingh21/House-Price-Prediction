import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def data_loader():

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


    return X_train, y_train, X_test, y_test, house_data