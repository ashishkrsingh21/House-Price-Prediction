import numpy as np
from src.Tree_Regression import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self,
                 n_estimators = 100,
                 max_depth = None,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 max_features = None
                ):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.tree = []

    def fit(self, X, y):

        self.tree = []

        n_samples = X.shape[0]

        for _ in range(self.n_estimators):

            sample_indices = np.random.choice(
                n_samples,
                size = n_samples,
                replace = True
            )

            X_bootstrap = X[sample_indices]
            y_bootstrap = y[sample_indices]

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features
                                        )

            tree.fit(X_bootstrap, y_bootstrap)

            self.tree.append(tree)

    def predict(self, X):

        tree_predictions = []

        for tree in self.tree:
            predictions = tree.predict(X)
            tree_predictions.append(predictions)

        tree_predictions = np.array(tree_predictions)

        return np.mean(tree_predictions, axis = 0)

