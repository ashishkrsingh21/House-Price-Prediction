
import numpy as np


class SVR:
    def __init__(
        self,
        learning_rate=0.001,
        epsilon=1.0,
        C=1.0,
        epochs=1000,
        eps=2e-5,
        optimization="primal",
        kernel="linear",
        gamma=None,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.C = C
        self.epochs = epochs
        self.eps = eps
        self.optimization = optimization.lower()
        self.kernel = kernel.lower()
        self.gamma = gamma

        self.theta = None
        self.theta_0 = None
        self.beta = None
        self.support_vectors_ = None
        self.support_indices_ = None
        self.dual_coef_ = None

        self.loss_history = []
        self.n_iter_ = 0
        self.iterations = 0
        self.successful_updates_ = 0
        self.converged_ = False
        self.kkt_violation_ = None
        self.final_objective_ = None

        self.intercept_bounds_ = None
        self.intercept_bounds_consistent_ = None
        self.intercept_candidates_spread_ = None

        self.X_train_ = None
        self.y_train_ = None

    # ==================================================
    # FIT
    # ==================================================
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must not be empty.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain finite values.")

        if self.C <= 0:
            raise ValueError("C must be greater than zero.")

        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative.")

        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")

        if self.eps <= 0:
            raise ValueError("eps must be greater than zero.")

        if self.optimization not in ("primal", "dual"):
            raise ValueError(
                "optimization must be 'primal' or 'dual'."
            )

        if self.kernel not in ("linear", "rbf"):
            raise ValueError(
                "kernel must be 'linear' or 'rbf'."
            )

        if self.optimization == "primal" and self.kernel != "linear":
            raise ValueError(
                "Primal SVR supports only the linear kernel here."
            )

        if self.gamma is not None and self.gamma <= 0:
            raise ValueError("gamma must be greater than zero.")

        self.X_train_ = X
        self.y_train_ = y
        self.loss_history = []

        if self.optimization == "primal":
            self._fit_primal(X, y)
        else:
            self._fit_dual(X, y)

        return self

    # ==================================================
    # PRIMAL SVR: SUBGRADIENT DESCENT
    # ==================================================
    def _fit_primal(self, X, y):
        n_samples, n_features = X.shape

        theta = np.zeros(n_features)
        theta_0 = 0.0

        self.converged_ = False
        self.successful_updates_ = 0

        for epoch in range(self.epochs):
            eta = self.learning_rate / np.sqrt(epoch + 1)

            predictions = X @ theta + theta_0
            residuals = predictions - y

            s = np.where(
                residuals > self.epsilon,
                1.0,
                np.where(residuals < -self.epsilon, -1.0, 0.0),
            )

            grad_w = theta + self.C * (X.T @ s)
            grad_b = self.C * np.sum(s)

            new_theta = theta - eta * grad_w
            new_theta_0 = theta_0 - eta * grad_b

            new_residuals = X @ new_theta + new_theta_0 - y

            regularization = 0.5 * np.dot(new_theta, new_theta)
            epsilon_loss = np.sum(
                np.maximum(
                    np.abs(new_residuals) - self.epsilon,
                    0.0,
                )
            )

            self.loss_history.append(
                regularization + self.C * epsilon_loss
            )

            theta_change = np.linalg.norm(new_theta - theta)
            intercept_change = abs(new_theta_0 - theta_0)

            theta = new_theta
            theta_0 = new_theta_0
            self.successful_updates_ += 1

            if (
                theta_change < self.eps
                and intercept_change < self.eps
            ):
                self.converged_ = True
                break

        self.theta = theta
        self.theta_0 = float(theta_0)

        self.theta_change = theta_change
        self.theta_0_change = intercept_change

        self.n_iter_ = epoch + 1
        self.iterations = self.n_iter_
        self.final_objective_ = self.loss_history[-1]

        self.beta = None
        self.support_vectors_ = None
        self.support_indices_ = None
        self.dual_coef_ = None

    # ==================================================
    # KERNEL
    # ==================================================
    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T

        gamma = self.gamma
        if gamma is None:
            gamma = 1.0 / X1.shape[1]

        X1_sq = np.sum(X1 * X1, axis=1)[:, None]
        X2_sq = np.sum(X2 * X2, axis=1)[None, :]

        distances = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
        distances = np.maximum(distances, 0.0)

        return np.exp(-gamma * distances)

    # ==================================================
    # DUAL OBJECTIVE
    # ==================================================
    def _dual_objective(self, beta, g, y):
        return (
            0.5 * np.dot(beta, g)
            - np.dot(y, beta)
            + self.epsilon * np.sum(np.abs(beta))
        )

    # ==================================================
    # PAIRWISE OBJECTIVE CHANGE
    # ==================================================
    def _pair_change(
        self,
        delta,
        curvature,
        q_pair,
        beta_i,
        beta_j,
    ):
        return (
            0.5 * curvature * delta**2
            + q_pair * delta
            + self.epsilon * (
                abs(beta_i + delta)
                - abs(beta_i)
                + abs(beta_j - delta)
                - abs(beta_j)
            )
        )

    # ==================================================
    # FIND THE BEST FEASIBLE UPDATE FOR A PAIR
    # ==================================================
    def _best_pair_update(
        self,
        beta_i,
        beta_j,
        q_pair,
        curvature,
    ):
        lower = max(
            -self.C - beta_i,
            beta_j - self.C,
        )
        upper = min(
            self.C - beta_i,
            beta_j + self.C,
        )

        if upper - lower <= 1e-14:
            return 0.0, 0.0

        breakpoints = [lower, upper]

        # The absolute-value terms change slope at these points.
        for point in (-beta_i, beta_j):
            if lower < point < upper:
                breakpoints.append(point)

        breakpoints = sorted(set(breakpoints))
        candidates = list(breakpoints)

        if curvature > 1e-14:
            for left, right in zip(
                breakpoints[:-1],
                breakpoints[1:],
            ):
                midpoint = (left + right) / 2.0

                sign_i = np.sign(beta_i + midpoint)
                sign_j = np.sign(beta_j - midpoint)

                stationary = -(
                    q_pair
                    + self.epsilon * (sign_i - sign_j)
                ) / curvature

                if left <= stationary <= right:
                    candidates.append(stationary)

        best_delta = 0.0
        best_change = 0.0

        for delta in candidates:
            change = self._pair_change(
                delta,
                curvature,
                q_pair,
                beta_i,
                beta_j,
            )

            if change < best_change:
                best_delta = delta
                best_change = change

        return best_delta, best_change

    # ==================================================
    # SELECT FEASIBLE PAIRS
    # ==================================================
    def _candidate_pairs(self, beta, q):
        # Directional derivative for increasing beta_i.
        inc_scores = q + self.epsilon * np.where(
            beta >= 0.0, 1.0, -1.0
        )

        # Directional derivative for decreasing beta_j.
        dec_scores = q + self.epsilon * np.where(
            beta > 0.0, 1.0, -1.0
        )

        inc_scores = inc_scores.copy()
        dec_scores = dec_scores.copy()

        inc_scores[beta >= self.C - 1e-10] = np.inf
        dec_scores[beta <= -self.C + 1e-10] = -np.inf

        valid_inc = np.flatnonzero(np.isfinite(inc_scores))
        valid_dec = np.flatnonzero(np.isfinite(dec_scores))

        if len(valid_inc) == 0 or len(valid_dec) == 0:
            return [], 0.0

        # Keep a small set of the best candidates in each direction.
        # This lets us try alternatives if the best pair stalls.
        k_inc = min(5, len(valid_inc))
        k_dec = min(5, len(valid_dec))

        inc_local = np.argpartition(
            inc_scores[valid_inc], k_inc - 1
        )[:k_inc]

        dec_local = np.argpartition(
            -dec_scores[valid_dec], k_dec - 1
        )[:k_dec]

        inc_indices = valid_inc[inc_local]
        dec_indices = valid_dec[dec_local]

        pairs = []

        for i in inc_indices:
            for j in dec_indices:
                if i == j:
                    continue

                violation = dec_scores[j] - inc_scores[i]

                if violation > 0.0:
                    pairs.append((float(violation), int(i), int(j)))

        pairs.sort(reverse=True)

        best_violation = pairs[0][0] if pairs else 0.0

        return pairs, best_violation

    # ==================================================
    # DUAL SVR: PAIRWISE COORDINATE OPTIMIZATION
    # ==================================================
    def _fit_dual(self, X, y):
        n_samples = X.shape[0]

        beta = np.zeros(n_samples)
        g = np.zeros(n_samples)

        self.loss_history = [0.0]
        self.converged_ = False
        self.successful_updates_ = 0
        self.kkt_violation_ = np.inf

        update_tol = 1e-12
        kkt_tol = self.eps

        def kernel_column(index):
            return self._kernel(
                X,
                X[index:index + 1],
            ).ravel()

        for iteration in range(self.epochs):
            q = g - y

            pairs, violation = self._candidate_pairs(beta, q)
            self.kkt_violation_ = violation

            if violation <= kkt_tol:
                self.converged_ = True
                break

            accepted = False

            # Try the highest-violation pair first. If it cannot produce
            # a meaningful decrease, try the next candidate pair.
            for _, i, j in pairs:
                K_ii = float(
                    self._kernel(X[i:i + 1], X[i:i + 1])[0, 0]
                )
                K_jj = float(
                    self._kernel(X[j:j + 1], X[j:j + 1])[0, 0]
                )
                K_ij = float(
                    self._kernel(X[i:i + 1], X[j:j + 1])[0, 0]
                )

                curvature = max(
                    K_ii + K_jj - 2.0 * K_ij,
                    0.0,
                )

                q_pair = q[i] - q[j]

                delta, change = self._best_pair_update(
                    beta[i],
                    beta[j],
                    q_pair,
                    curvature,
                )

                if abs(delta) <= update_tol or change >= -update_tol:
                    continue

                column_i = kernel_column(i)
                column_j = kernel_column(j)

                beta[i] += delta
                beta[j] -= delta

                # Update g = K @ beta without constructing the full Gram matrix.
                g += delta * (column_i - column_j)

                self.successful_updates_ += 1
                accepted = True
                break

            if not accepted:
                # No candidate pair produced a numerically meaningful
                # decrease. Do not incorrectly report convergence.
                self.converged_ = False
                break

            if (iteration + 1) % 100 == 0:
                self.loss_history.append(
                    self._dual_objective(beta, g, y)
                )

        self.n_iter_ = iteration + 1
        self.iterations = self.n_iter_

        final_objective = self._dual_objective(beta, g, y)

        if (
            not self.loss_history
            or self.loss_history[-1] != final_objective
        ):
            self.loss_history.append(final_objective)

        self.final_objective_ = final_objective
        self.beta = beta

        support_tol = 1e-10
        self.support_indices_ = np.flatnonzero(
            np.abs(beta) > support_tol
        )

        self.support_vectors_ = X[self.support_indices_]
        self.dual_coef_ = beta[self.support_indices_]

        self.theta = None

        if self.kernel == "linear":
            self.theta = X.T @ beta

        self.theta_0 = self._estimate_intercept(y, g, beta)

    # ==================================================
    # INTERCEPT ESTIMATION
    # ==================================================
    def _estimate_intercept(self, y, g, beta):
        tol = 1e-8

        # Nonzero coefficients strictly inside the box constraints
        # provide direct intercept estimates.
        free = (
            (np.abs(beta) > tol)
            & (beta > -self.C + tol)
            & (beta < self.C - tol)
        )

        if np.any(free):
            values = (
                y[free]
                - g[free]
                - self.epsilon * np.sign(beta[free])
            )

            self.intercept_candidates_spread_ = float(
                np.max(values) - np.min(values)
            )
            self.intercept_bounds_ = None
            self.intercept_bounds_consistent_ = None

            return float(np.median(values))

        lower_bounds = []
        upper_bounds = []

        for i in range(len(beta)):
            value = y[i] - g[i]

            if beta[i] <= -self.C + tol:
                lower_bounds.append(value + self.epsilon)

            elif beta[i] >= self.C - tol:
                upper_bounds.append(value - self.epsilon)

            else:
                lower_bounds.append(value - self.epsilon)
                upper_bounds.append(value + self.epsilon)

        lower = max(lower_bounds) if lower_bounds else -np.inf
        upper = min(upper_bounds) if upper_bounds else np.inf

        self.intercept_bounds_ = (lower, upper)
        self.intercept_bounds_consistent_ = lower <= upper
        self.intercept_candidates_spread_ = None

        if lower <= upper:
            if np.isfinite(lower) and np.isfinite(upper):
                return float((lower + upper) / 2.0)

            if np.isfinite(lower):
                return float(lower)

            if np.isfinite(upper):
                return float(upper)

            return 0.0

        # Keep the diagnostic showing inconsistency rather than claiming
        # that the intercept satisfies all KKT conditions.
        return float((lower + upper) / 2.0)

    # ==================================================
    # PREDICT
    # ==================================================
    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if self.theta is not None:
            return X @ self.theta + self.theta_0

        if self.support_vectors_ is None or len(self.support_vectors_) == 0:
            return np.full(X.shape[0], self.theta_0)

        predictions = np.empty(X.shape[0])
        batch_size = 256

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            K = self._kernel(
                X[start:end],
                self.support_vectors_,
            )

            predictions[start:end] = (
                K @ self.dual_coef_ + self.theta_0
            )

        return predictions
