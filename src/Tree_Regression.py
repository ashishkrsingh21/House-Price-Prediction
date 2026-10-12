import numpy as np

class Node:

    def __init__(self,
                 feature_index = None,
                 threshold=None,
                 prediction=None
                 ):

        self.feature_index = feature_index
        self.threshold = threshold
        self.prediction = prediction

        self.left = None
        self.right = None


class DecisionTreeRegression:

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1
                 ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self,X,y):
        def calculate_sse(y):
            mean_y = np.mean(y)
            sse = np.sum((y-mean_y)**2)
            return sse

        def calculate_split_sse(X,y,threshold):

            left_mask = X < threshold
            right_mask = X >= threshold

            left_y = y[left_mask]
            right_y = y[right_mask]

            left_count = np.sum(left_mask)
            right_count = np.sum(right_mask)

            if (left_count < self.min_samples_leaf or right_count < self.min_samples_leaf):
                return np.inf
            
            left_sse = calculate_sse(left_y)
            right_sse = calculate_sse(right_y)

            return left_sse + right_sse

        def find_best_split(X, y):
            min_sse = np.inf
            best_feature_index = None
            best_threshold = None

            n_features = X.shape[1]

            for feature_index in range(n_features):

                X_feature = X[:, feature_index]

                sorted_indices = np.argsort(X_feature)
                X_sorted = X_feature[sorted_indices]
                y_sorted = y[sorted_indices]

                for i in range(len(X_sorted) - 1):

                    if X_sorted[i] != X_sorted[i + 1]:

                        threshold = (X_sorted[i] + X_sorted[i + 1]) / 2

                        sse = calculate_split_sse(
                            X_sorted,
                            y_sorted,
                            threshold
                        )

                        if sse < min_sse:
                            min_sse = sse
                            best_feature_index = feature_index
                            best_threshold = threshold

            return best_feature_index, best_threshold, min_sse

        def split_data(X, y, feature_index, threshold):

            left_mask = X[:, feature_index] < threshold
            right_mask = X[:, feature_index] >= threshold

            X_left = X[left_mask]
            y_left = y[left_mask]

            X_right = X[right_mask]
            y_right = y[right_mask]

            return X_left, y_left, X_right, y_right

        def create_leaf(y):
            return np.mean(y)

        def build_tree(X, y, depth):

            if len(y) < self.min_samples_split:
                prediction = create_leaf(y)
                return Node(prediction=prediction)

            if self.max_depth is not None and depth >= self.max_depth:
                prediction = create_leaf(y)
                return Node(prediction=prediction)

            if self.min_samples_split is not None and len(y) < self.min_samples_split:
                prediction = create_leaf(y)
                return Node(prediction=prediction)

            best_feature_index, best_threshold, min_sse = find_best_split(X, y)

            if best_feature_index is None:
                prediction = create_leaf(y)
                return Node(prediction=prediction)

            X_left, y_left, X_right, y_right = split_data(X,
                                                          y,
                                                          best_feature_index,
                                                          best_threshold
                                                            )

            node = Node(feature_index=best_feature_index,
                        threshold=best_threshold
                        )

            node.left = build_tree(X_left, y_left, depth + 1)
            node.right = build_tree(X_right, y_right, depth + 1)

            return node

        self.root = build_tree(X,y,depth = 0)


    def predict_one(self, X, node=None):

        if node is None:
            node = self.root

        if node.prediction is not None:
            return node.prediction
        
        if X[node.feature_index] < node.threshold:
            return self.predict_one(X, node.left)
        else:
            return self.predict_one(X, node.right)


    def predict(self, X):
        result = []

        for x in X:
            result.append(self.predict_one(x))

        return np.array(result)
