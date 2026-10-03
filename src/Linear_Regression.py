import numpy as np

class LinearRegression():

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        # Calculate theta using the Normal Equation
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.theta