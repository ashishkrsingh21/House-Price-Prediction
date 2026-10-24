import numpy as np
from src.models.decision_tree_regression import DecisionTreeRegressor

class GradientBootingRegressor:

    def __init__(self,
                 n_estimators=None,
                 learning_rate=None,
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=None,
                 max_features=None
                 ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.initial_prediction = None
        self.trees = []

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.initial_prediction = np.mean(y)
        
        # Initialize predictions using the training target mean
        predictions = np.full(len(y), self.initial_prediction)

        self.trees = []
        
        # Train trees sequentially
        for _ in range(self.n_estimators):

            # Calculate residuals from the current ensemble
            residuals = y - predictions

            # Create a new regression tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )

            # Train the tree to predict residuals
            tree.fit(X, residuals)

            # Store the trained tree
            self.trees.append(tree)

            # Predict corrections
            corrections = tree.predict(X)

            # Update the current ensemble predictions
            predictions += self.learning_rate * corrections

    def predict(self, X):

        X = np.asarray(X, dtype=float)
    
        if self.initial_prediction is None:
            raise ValueError("Model must be trained before making predictions")

        # Start with the initial prediction
        predictions = np.full(
            len(X),
            self.initial_prediction
        )

        # Add the contribution of every trained tree
        for tree in self.trees:

            corrections = tree.predict(X)

            predictions += self.learning_rate * corrections

        return predictions