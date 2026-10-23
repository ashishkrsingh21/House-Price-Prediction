
import numpy as np


class SVR:

    def __init__(self,
                 learning_rate=None,
                 epsilon=None,
                 C=None,
                 epochs=None,
                 eps=2e-5):

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.C = C
        self.epochs = epochs
        self.eps = eps

        self.theta = None
        self.theta_0 = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.theta = np.zeros(n_features)
        self.theta_0 = 0.0
        self.loss_history = []

        def h(theta, theta_0, X):
            return X @ theta + theta_0

        def subgradient(residuals, epsilon):
            return np.where(
                residuals > epsilon,
                1,
                np.where(residuals < -epsilon, -1, 0)
            )

        def gradient_w(theta, X, s):
            return theta + self.C * (X.T @ s)

        def gradient_b(s):
            return self.C * np.sum(s)

        def objective(theta, residuals):
            regularization = 0.5 * np.dot(theta, theta)

            loss = np.sum(
                np.maximum(np.abs(residuals) - self.epsilon, 0)
            )

            return regularization + self.C * loss

        old_theta = self.theta.copy()
        old_theta_0 = self.theta_0

        for epoch in range(self.epochs):

            eta = (self.learning_rate) / np.sqrt(epoch + 1)

            predictions = h(old_theta, old_theta_0, X)
            residuals = predictions - y

            s = subgradient(residuals, self.epsilon)

            grad_w = gradient_w(old_theta, X, s)
            grad_b = gradient_b(s)

            new_theta = (
                old_theta - eta * grad_w
            )

            new_theta_0 = (
                old_theta_0 - eta * grad_b
            )

            new_predictions = h(new_theta, new_theta_0, X)
            new_residuals = new_predictions - y

            current_loss = objective(new_theta, new_residuals)
            self.loss_history.append(current_loss)

            theta_change = np.linalg.norm(
                new_theta - old_theta
            )

            theta_0_change = abs(
                new_theta_0 - old_theta_0
            )

            old_theta = new_theta
            old_theta_0 = new_theta_0

            if (
                theta_change < self.eps
                and theta_0_change < self.eps
            ):
                break

        self.theta = old_theta
        self.theta_0 = old_theta_0
        self.theta_change = theta_change
        self.theta_0_change = theta_0_change

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        return X @ self.theta + self.theta_0
