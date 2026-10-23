import numpy as np
from data_loader import data_loader
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def transformer():

        X_train, y_train, X_test, y_test, house_data = data_loader()
        
        numeric_features = house_data.select_dtypes(include=["number"]).columns.to_list()
    
        categorical_features = house_data.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    
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
    
        return X_train_preprocessed, X_test_preprocessed