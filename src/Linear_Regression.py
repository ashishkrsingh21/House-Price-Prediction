import numpy as np

class LinearRegression():
    """ A simple implementation of Linear Regression using the Normal Equation.
    
        args: None
        returns: Predicted values for the input features using the learned parameters.
    """

    def __init__(self):
        # Initialize the model parameters (theta) to None
        self.theta = None

    def fit(self, X, y):
        # Calculate theta using the Normal Equation
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Return the predicted values for the input features using the learned parameters
        return X @ self.theta