"""Pre-defined saddle-point equations for common problems."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import erf


class ReplicaEquations(ABC):
    """
    Abstract base class for replica saddle-point equations.

    Subclasses implement specific equations for different learning problems.
    """

    def __init__(self, **params: Any):
        """
        Initialize with problem-specific parameters.

        Args:
            **params: Parameters like rho, eta, lambda, etc.

        """
        self.params = params

    @abstractmethod
    def __call__(
        self,
        *order_params: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, ...]:
        """
        Compute updated order parameters.

        Args:
            *order_params: Current order parameter values.
            alpha: Sample ratio n/d.
            **kwargs: Additional parameters.

        Returns:
            Tuple of updated order parameter values.

        """
        pass

    @abstractmethod
    def generalization_error(
        self,
        *order_params: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error from order parameters.

        Args:
            *order_params: Order parameter values.
            **kwargs: Additional parameters.

        Returns:
            Generalization error.

        """
        pass

    def get_order_param_names(self) -> list[str]:
        """Return names of order parameters."""
        return ["m", "q"]


class RidgeRegressionEquations(ReplicaEquations):
    """
    Saddle-point equations for ridge regression.

    Teacher-student setup with linear teacher:
        y = (1/sqrt(d)) * W0^T @ x + noise

    Ridge regression minimizes:
        L = (1/n) * ||y - Xw/sqrt(d)||^2 + λ||w||^2

    Order parameters:
        - m: overlap with teacher (w^T W0 / d)
        - q: self-overlap (||w||^2 / d)
    """

    def __init__(
        self,
        rho: float = 1.0,
        eta: float = 0.0,
        reg_param: float = 0.01,
        eps: float = 1e-6,
        **params: Any,
    ):
        """
        Initialize RidgeRegressionEquations.

        Args:
            rho: Teacher norm (||W0||^2 / d).
            eta: Noise variance.
            reg_param: Ridge parameter λ.
            eps: Small constant for numerical stability.

        """
        super().__init__(rho=rho, eta=eta, reg_param=reg_param, eps=eps, **params)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param
        self.eps = eps

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q.

        These are the standard replica equations for ridge regression.
        """
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)
        eps = kwargs.get("eps", self.eps)

        # Effective regularization (including noise contribution)
        # V = variance of effective noise in the problem
        V = rho - 2 * m + q + eta

        # Conjugate variables (from replica calculation)
        # For ridge: closed-form solutions
        hat_m = alpha * m / (1 + alpha * q / (lam + eps))
        hat_q = alpha * (V + m**2) / ((1 + alpha * q / (lam + eps)) ** 2)

        # Update equations (proximal interpretation)
        new_m = rho * hat_m / (lam + hat_q + eps)
        new_q = (rho * hat_m**2 + hat_q * (rho + eta)) / ((lam + hat_q + eps) ** 2)

        return new_m, new_q

    def generalization_error(
        self,
        m: float,
        q: float,
        alpha: float = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error.

        E_g = 0.5 * (rho - 2*m + q) for MSE loss.
        """
        rho = kwargs.get("rho", self.rho)
        return 0.5 * (rho - 2 * m + q)


class LassoEquations(ReplicaEquations):
    """
    Saddle-point equations for LASSO regression.

    Uses soft-thresholding proximal operator.
    """

    def __init__(
        self,
        rho: float = 1.0,
        eta: float = 0.0,
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize LassoEquations.

        Args:
            rho: Teacher norm.
            eta: Noise variance.
            reg_param: LASSO parameter λ.

        """
        super().__init__(rho=rho, eta=eta, reg_param=reg_param, **params)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param

    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator."""
        return np.sign(x) * max(abs(x) - threshold, 0)

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q using LASSO equations.

        Uses Gaussian integration over effective field.
        """
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)

        # Variance of effective noise
        V = rho - 2 * m + q + eta

        # Conjugate variable (approximate for LASSO)
        hat_q = alpha * V / max(1 - alpha, 0.01)

        # Gaussian integration for LASSO update
        # This is a simplified version; full version uses numerical integration

        def integrand_m(z):
            """Integrand for m update."""
            effective_signal = np.sqrt(rho) * m / np.sqrt(q + 0.001) + np.sqrt(hat_q) * z
            proximal = self._soft_threshold(effective_signal, lam / np.sqrt(hat_q + 0.001))
            return (
                proximal
                * np.sqrt(rho)
                / np.sqrt(q + 0.001)
                * np.exp(-(z**2) / 2)
                / np.sqrt(2 * np.pi)
            )

        def integrand_q(z):
            """Integrand for q update."""
            effective_signal = np.sqrt(rho) * m / np.sqrt(q + 0.001) + np.sqrt(hat_q) * z
            proximal = self._soft_threshold(effective_signal, lam / np.sqrt(hat_q + 0.001))
            return proximal**2 * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)

        # Numerical integration
        new_m, _ = quad(integrand_m, -10, 10)
        new_q, _ = quad(integrand_q, -10, 10)

        return max(new_m, 0.001), max(new_q, 0.001)

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """Compute generalization error."""
        rho = kwargs.get("rho", self.rho)
        return 0.5 * (rho - 2 * m + q)


class LogisticRegressionEquations(ReplicaEquations):
    """
    Saddle-point equations for logistic regression.

    Binary classification with logistic loss.
    """

    def __init__(
        self,
        rho: float = 1.0,
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize LogisticRegressionEquations.

        Args:
            rho: Teacher norm.
            reg_param: L2 regularization parameter.

        """
        super().__init__(rho=rho, reg_param=reg_param, **params)
        self.rho = rho
        self.reg_param = reg_param

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Compute updated m and q for logistic regression."""
        rho = kwargs.get("rho", self.rho)
        lam = kwargs.get("reg_param", self.reg_param)

        # Simplified logistic equations
        # Full version requires more complex integration

        # Effective field variance
        rho - 2 * m + q + 0.001

        # Update equations (approximate)
        scale = alpha / (1 + lam)

        def integrand_m(z):
            """Integrand for m."""
            y_teacher = np.sign(np.sqrt(rho) + 0.1 * z)
            field = m / np.sqrt(q + 0.001) * np.sqrt(rho) + np.sqrt(q) * z
            grad = y_teacher * self._sigmoid(-y_teacher * field)
            return grad * np.sqrt(rho) * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)

        def integrand_q(z):
            """Integrand for q."""
            y_teacher = np.sign(np.sqrt(rho) + 0.1 * z)
            field = m / np.sqrt(q + 0.001) * np.sqrt(rho) + np.sqrt(q) * z
            grad = y_teacher * self._sigmoid(-y_teacher * field)
            return grad**2 * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)

        dm, _ = quad(integrand_m, -5, 5)
        dq, _ = quad(integrand_q, -5, 5)

        new_m = m + 0.1 * (scale * dm - lam * m)
        new_q = q + 0.1 * (scale * dq - lam * q)

        return max(new_m, 0.001), max(new_q, 0.001)

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) * arccos(m / sqrt(q * rho))
        """
        rho = kwargs.get("rho", self.rho)
        if q > 0 and rho > 0:
            cos_angle = m / np.sqrt(q * rho)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5


class PerceptronEquations(ReplicaEquations):
    """
    Saddle-point equations for perceptron learning.

    Teacher-student setup with sign teacher:
        y = sign((1/sqrt(d)) * W0^T @ x)

    Various loss functions can be used (hinge, perceptron, logistic).
    """

    def __init__(
        self,
        rho: float = 1.0,
        margin: float = 0.0,
        reg_param: float = 0.0,
        **params: Any,
    ):
        """
        Initialize PerceptronEquations.

        Args:
            rho: Teacher norm (||W0||^2 / d).
            margin: Margin parameter (kappa for Gardner volume).
            reg_param: Regularization parameter.

        """
        super().__init__(rho=rho, margin=margin, reg_param=reg_param, **params)
        self.rho = rho
        self.margin = margin
        self.reg_param = reg_param

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q for perceptron.

        Uses the classic perceptron/SVM equations.
        """
        rho = kwargs.get("rho", self.rho)
        kappa = kwargs.get("margin", self.margin)
        lam = kwargs.get("reg_param", self.reg_param)

        # Correlation between student and teacher fields
        # rho_sq = m^2 / (rho * q) when normalized
        rho_corr = m / np.sqrt(rho * q) if q > 0 and rho > 0 else 0.0

        # Effective variance of student field given teacher label
        Delta = np.sqrt(q * (1 - rho_corr**2) + 1e-10)

        def H(x):
            """Gaussian tail function."""
            return 0.5 * (1 - erf(x / np.sqrt(2)))

        def G(x):
            """Gaussian PDF."""
            return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

        # Integrands for m and q updates (hinge/perceptron)
        def integrand_m(z):
            """Integrand for m update."""
            # Effective field
            h = m / np.sqrt(q + 1e-10) + Delta * z
            # Contribution from misclassified samples
            if kappa > 0:
                # Margin constraint
                indicator = 1.0 if h < kappa else 0.0
            else:
                indicator = 1.0 if h < 0 else 0.0
            return indicator * (kappa - h) * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)

        def integrand_q(z):
            """Integrand for q update."""
            h = m / np.sqrt(q + 1e-10) + Delta * z
            indicator = (1.0 if h < kappa else 0.0) if kappa > 0 else 1.0 if h < 0 else 0.0
            return indicator * (kappa - h) ** 2 * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)

        # Numerical integration
        dm_contrib, _ = quad(integrand_m, -10, 10)
        dq_contrib, _ = quad(integrand_q, -10, 10)

        # Update equations
        lr = 0.1
        new_m = m + lr * (alpha * np.sqrt(rho) * dm_contrib - lam * m)
        new_q = q + lr * (alpha * dq_contrib - lam * q)

        return max(new_m, 1e-10), max(new_q, 1e-10)

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) * arccos(m / sqrt(q * rho))
        """
        rho = kwargs.get("rho", self.rho)
        if q > 0 and rho > 0:
            cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5


class ProbitEquations(ReplicaEquations):
    """
    Saddle-point equations for probit regression.

    Teacher generates labels via:
        P(y=1|u) = Phi(u) where Phi is Gaussian CDF

    The probit model is analytically convenient because
    Gaussian integrals have closed-form solutions.
    """

    def __init__(
        self,
        rho: float = 1.0,
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize ProbitEquations.

        Args:
            rho: Teacher norm.
            reg_param: L2 regularization parameter.

        """
        super().__init__(rho=rho, reg_param=reg_param, **params)
        self.rho = rho
        self.reg_param = reg_param

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Compute updated m and q for probit model."""
        rho = kwargs.get("rho", self.rho)
        lam = kwargs.get("reg_param", self.reg_param)

        # For probit, the analysis simplifies due to Gaussian structure
        # The effective noise is sqrt(rho + 1) when teacher is probit

        # Variance components
        V_teacher = rho  # Teacher signal variance
        V_student = q  # Student self-overlap

        # Correlation
        corr = m / np.sqrt(V_teacher * V_student) if V_teacher > 0 and V_student > 0 else 0.0

        def G(x):
            """Gaussian PDF."""
            return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

        def Phi(x):
            """Gaussian CDF."""
            return 0.5 * (1 + erf(x / np.sqrt(2)))

        # Integrands for the saddle-point equations
        def integrand_m(z, u):
            """Integrand for m update."""
            # Joint Gaussian structure
            prob_y1 = Phi(u * np.sqrt(rho))
            student_field = (
                m * u / np.sqrt(rho + 1e-10) + np.sqrt(q - m**2 / (rho + 1e-10) + 1e-10) * z
            )
            # Gradient contribution
            y_eff = 2 * prob_y1 - 1
            grad = y_eff * G(student_field) / (Phi(y_eff * student_field) + 1e-10)
            return grad * u * np.sqrt(rho) * G(u) * G(z)

        # Simplified update using approximations
        scale = alpha / (1 + lam)

        # Approximate updates
        new_m = m + 0.1 * scale * (np.sqrt(rho) * corr * (1 - corr**2) - lam * m)
        new_q = q + 0.1 * scale * ((1 - corr**2) - lam * q)

        return max(new_m, 1e-10), max(new_q, 1e-10)

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """Compute classification error for probit."""
        rho = kwargs.get("rho", self.rho)
        if q > 0 and rho > 0:
            cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5


class CommitteeMachineEquations(ReplicaEquations):
    """
    Saddle-point equations for committee machine.

    Student: f(x) = (1/sqrt(K)) * sum_k phi(v_k^T x / sqrt(d))
    Teacher: y = (1/sqrt(M)) * sum_m phi(v*_m^T x / sqrt(d))

    Order parameters are matrices:
        Q_{kk'} = (1/d) * v_k^T v_k'
        R_{km} = (1/d) * v_k^T v*_m
        T_{mm'} = (1/d) * v*_m^T v*_m'
    """

    def __init__(
        self,
        K: int = 2,
        M: int = 2,
        rho: float = 1.0,
        eta: float = 0.0,
        activation: str = "erf",
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize CommitteeMachineEquations.

        Args:
            K: Number of student hidden units.
            M: Number of teacher hidden units.
            rho: Teacher weight norm per unit.
            eta: Noise variance.
            activation: Activation function ('erf', 'tanh', 'sign').
            reg_param: L2 regularization parameter.

        """
        super().__init__(
            K=K, M=M, rho=rho, eta=eta, activation=activation, reg_param=reg_param, **params
        )
        self.K = K
        self.M = M
        self.rho = rho
        self.eta = eta
        self.activation = activation
        self.reg_param = reg_param

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Activation function."""
        if self.activation == "erf":
            return erf(x / np.sqrt(2))
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sign":
            return np.sign(x)
        elif self.activation == "relu":
            return np.maximum(0, x)
        else:
            return x

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q.

        For committee machine, m and q are scalars assuming symmetric ansatz
        (all student units have same statistics).
        """
        rho = kwargs.get("rho", self.rho)
        kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)
        K = kwargs.get("K", self.K)

        # For symmetric ansatz:
        # Q_kk' = q * delta_{kk'} + c * (1 - delta_{kk'}) where c is off-diagonal
        # R_km = r for all k, m

        # Simplified symmetric update
        c = 0.0  # Off-diagonal (could be another order parameter)

        # Generalization error contribution
        eg = rho - 2 * m + q + (K - 1) * c

        # Update equations (simplified for K=M symmetric case)
        lr = 0.1

        # Correlation functions for erf activation (I_1, I_2, etc.)
        # These are standard integrals in committee machine analysis

        # Approximate updates
        new_m = m + lr * (alpha * np.sqrt(rho / K) * (rho - m) / (1 + eg + lam) - lam * m)
        new_q = q + lr * (alpha * (eg + m**2) / ((1 + eg + lam) ** 2) - lam * q)

        return max(new_m, 1e-10), max(new_q, 1e-10)

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error for committee machine.

        For erf activation with symmetric ansatz:
        E_g = (1/pi) * arccos(...) + noise contributions
        """
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)
        kwargs.get("K", self.K)

        # MSE-like error for regression
        eg = 0.5 * (rho - 2 * m + q)
        if eta > 0:
            eg += 0.5 * eta

        return eg

    def get_order_param_names(self) -> list[str]:
        """Return names of order parameters."""
        return ["m", "q"]
