import numpy as np

class Node:

    def __init__(self,
                 feature_index = None,
                 threshold=None,
                 prediction=None,
                 sse_reduction=None,
                 n_samples=None
                 ):

        self.feature_index = feature_index
        self.threshold = threshold
        self.prediction = prediction
        self.sse_reduction = sse_reduction
        self.n_samples = n_samples

        self.left = None
        self.right = None


class DecisionTreeRegressor:

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None
                 ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self,X,y):
        def calculate_sse(y):
            sum_y_2 = np.sum(y**2)
            sum_y = np.sum(y)
            sse = sum_y_2 - ((sum_y ** 2)/len(y))
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

            if self.max_features is None:
                feature_indices = np.arange(n_features)
            else:
                if self.max_features < 1 or self.max_features > n_features:
                    raise ValueError("max_features must be between 1 and n_features")

                feature_indices = np.random.choice(
                    n_features,
                    size=self.max_features,
                    replace=False
                )

            for feature_index in feature_indices:

                X_feature = X[:, feature_index]

                sorted_indices = np.argsort(X_feature)
                X_sorted = X_feature[sorted_indices]
                y_sorted = y[sorted_indices]

                # Calculate cumulative statistics once for this feature
                cumulative_sum = np.cumsum(y_sorted)
                cumulative_sum_squared = np.cumsum(y_sorted ** 2)

                # Calculate total statistics
                total_sum = cumulative_sum[-1]
                total_sum_squared = cumulative_sum_squared[-1]
                n_samples = len(y_sorted)

                # Evaluate candidate splits
                for i in range(n_samples - 1):

                    if X_sorted[i] != X_sorted[i + 1]:

                        threshold = (X_sorted[i] + X_sorted[i + 1]) / 2

                        # Number of observations in each group
                        n_left = i + 1
                        n_right = n_samples - n_left

                        # Enforce minimum leaf size
                        if (n_left < self.min_samples_leaf or
                                n_right < self.min_samples_leaf):
                            continue

                        # Left group statistics
                        sum_left = cumulative_sum[i]
                        sum_squared_left = cumulative_sum_squared[i]

                        # Right group statistics
                        sum_right = total_sum - sum_left
                        sum_squared_right = total_sum_squared - sum_squared_left

                        # Calculate SSE for each group
                        left_sse = (
                            sum_squared_left - (sum_left ** 2 / n_left)
                        )

                        right_sse = (
                            sum_squared_right - (sum_right ** 2 / n_right)
                        )

                        sse = left_sse + right_sse

                        # Keep the best split
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

            node = Node(n_samples=len(y),
                        sse_reduction=calculate_sse(y) - min_sse,
                        feature_index=best_feature_index,
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

    def calculate_feature_importance(self, n_features, n_samples):
        importance = np.zeros(n_features)

        def traverse(node):
            if node is None or node.prediction is not None:
                return

            weighted_reduced_sse = (node.n_samples/
                                    n_samples) * node.sse_reduction

            importance[node.feature_index] += weighted_reduced_sse

            traverse(node.left)
            traverse(node.right)

        traverse(self.root)

        total_importance = np.sum(importance)

        if total_importance > 0:
            importance = importance / total_importance

        return importance