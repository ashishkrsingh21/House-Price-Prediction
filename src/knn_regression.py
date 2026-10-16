import numpy as np


class KnnRegressor:

    def __init__(self, n_neighbour=5):
        self.n_neighbour = n_neighbour
        self.is_fitted = False

    def fit(self, X_train, y_train):

        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64)

        if self.X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")

        if self.y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array.")

        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                "X_train and y_train must have the same number of observations."
            )

        if len(self.X_train) == 0:
            raise ValueError("Training data cannot be empty.")

        if not isinstance(self.n_neighbour, (int, np.integer)) or isinstance(
            self.n_neighbour, (bool, np.bool_)
        ):
            raise ValueError("k must be an integer.")

        if not 1 <= self.n_neighbour <= len(self.X_train):
            raise ValueError(
                "k must be between 1 and the number of training observations."
            )

        # Precompute squared norms of training observations.
        self.X_train_squared_norm = np.sum(
            self.X_train ** 2, axis=1
        )

        self.is_fitted = True

    def predict(self, X_test, batch_size=128):

        if not self.is_fitted:
            raise ValueError("Call fit() before predict().")

        X_test = np.asarray(X_test, dtype=np.float64)

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        if X_test.ndim != 2:
            raise ValueError("X_test must be a 1D or 2D array.")

        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "X_test and X_train must have the same number of features."
            )

        if (
            not isinstance(batch_size, (int, np.integer))
            or isinstance(batch_size, (bool, np.bool_))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")

        predictions = np.empty(len(X_test), dtype=np.float64)

        for start in range(0, len(X_test), batch_size):

            end = min(start + batch_size, len(X_test))
            X_batch = X_test[start:end]

            # Step 1: Calculate squared Euclidean distances
            X_batch_squared_norm = np.sum(
                X_batch ** 2,
                axis=1,
                keepdims=True
            )

            distances = (
                X_batch_squared_norm
                + self.X_train_squared_norm[None, :]
                - 2 * (X_batch @ self.X_train.T)
            )

            # Prevent tiny negative values caused by floating-point errors
            np.maximum(distances, 0, out=distances)

            # Step 2: Find the indices of the k nearest neighbors
            nearest_indices = np.argpartition(
                distances,
                kth=self.n_neighbour - 1,
                axis=1
            )[:, :self.n_neighbour]

            # Step 3: Retrieve their squared distances
            nearest_squared_distances = np.take_along_axis(
                distances,
                nearest_indices,
                axis=1
            )

            # Convert squared distances to actual Euclidean distances
            nearest_distances = np.sqrt(nearest_squared_distances)

            # Step 4: Retrieve the target prices of the neighbors
            nearest_targets = self.y_train[nearest_indices]

            # Step 5: Calculate inverse-distance weights
            zero_mask = nearest_distances == 0

            weights = np.zeros_like(nearest_distances)

            np.divide(
                1.0,
                nearest_distances,
                out=weights,
                where=~zero_mask
            )

            # Step 6: Calculate weighted predictions
            weighted_numerator = np.sum(
                weights * nearest_targets,
                axis=1
            )

            weighted_denominator = np.sum(
                weights,
                axis=1
            )

            # Prevent division by zero when all selected distances are zero
            safe_denominator = np.where(
                weighted_denominator == 0,
                1.0,
                weighted_denominator
            )

            weighted_predictions = (
                weighted_numerator / safe_denominator
            )

            # Step 7: Handle exact matches
            # If a test observation exactly matches training observations,
            # average the targets of those exact matches only.
            zero_count = np.sum(zero_mask, axis=1)

            zero_target_sum = np.sum(
                np.where(zero_mask, nearest_targets, 0.0),
                axis=1
            )

            safe_zero_count = np.where(
                zero_count == 0,
                1,
                zero_count
            )

            exact_match_predictions = zero_target_sum / safe_zero_count

            # Use exact-match predictions whenever a zero distance exists
            batch_predictions = np.where(
                zero_count > 0,
                exact_match_predictions,
                weighted_predictions
            )

            predictions[start:end] = batch_predictions

        return predictions