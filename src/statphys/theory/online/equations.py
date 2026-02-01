"""Pre-defined ODE equations for online learning dynamics."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.special import erf


class OnlineEquations(ABC):
    """
    Abstract base class for online learning ODE equations.

    Subclasses implement specific dynamics for different
    learning algorithms and architectures.
    """

    def __init__(self, **params: Any):
        """
        Initialize with problem-specific parameters.

        Args:
            **params: Parameters like rho, eta, lr, reg_param, etc.

        """
        self.params = params

    @abstractmethod
    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute ODE right-hand side dy/dt.

        Args:
            t: Current time (= number of samples / d).
            y: Current order parameter values.
            params: Additional parameters.

        Returns:
            Array of dy/dt values.

        """
        pass

    @abstractmethod
    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error from order parameters.

        Args:
            y: Order parameter values.
            **kwargs: Additional parameters.

        Returns:
            Generalization error.

        """
        pass

    def get_order_param_names(self) -> list[str]:
        """Return names of order parameters."""
        return ["m", "q"]


class OnlineSGDEquations(OnlineEquations):
    """
    ODE equations for online SGD on linear regression.

    Teacher-student setup with linear teacher:
        y = (1/sqrt(d)) * W0^T @ x + noise

    SGD update: w <- w - η * ∇L(w)

    In the d→∞ limit, order parameter dynamics:
        dm/dt = η * (ρ - m) - η * λ * m
        dq/dt = η² * (ρ - 2m + q + η²*σ²) + 2*η*(m - q) - 2*η*λ*q

    where m = w^T W0 / d, q = ||w||^2 / d.
    """

    def __init__(
        self,
        rho: float = 1.0,
        eta_noise: float = 0.0,
        lr: float = 0.1,
        reg_param: float = 0.0,
        **params: Any,
    ):
        """
        Initialize OnlineSGDEquations.

        Args:
            rho: Teacher norm (||W0||^2 / d).
            eta_noise: Output noise variance.
            lr: Learning rate η.
            reg_param: L2 regularization λ.

        """
        super().__init__(rho=rho, eta_noise=eta_noise, lr=lr, reg_param=reg_param, **params)
        self.rho = rho
        self.eta_noise = eta_noise
        self.lr = lr
        self.reg_param = reg_param

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Compute dm/dt and dq/dt for online SGD."""
        m, q = y

        # Get parameters (allow override from params dict)
        rho = params.get("rho", self.rho)
        eta_noise = params.get("eta_noise", self.eta_noise)
        lr = params.get("lr", self.lr)
        lam = params.get("reg_param", self.reg_param)

        # Residual variance
        residual_var = rho - 2 * m + q + eta_noise

        # ODE for m: teacher-student overlap
        dm_dt = lr * (rho - m) - lr * lam * m

        # ODE for q: self-overlap
        # Second moment term from gradient noise
        grad_noise = lr**2 * residual_var

        dq_dt = grad_noise + 2 * lr * (m - q) - 2 * lr * lam * q

        return np.array([dm_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error.

        E_g = 0.5 * (ρ - 2m + q)
        """
        m, q = y
        rho = kwargs.get("rho", self.rho)
        return 0.5 * (rho - 2 * m + q)


class OnlinePerceptronEquations(OnlineEquations):
    """
    ODE equations for online perceptron learning.

    Binary classification with perceptron update rule.
    """

    def __init__(
        self,
        rho: float = 1.0,
        lr: float = 1.0,
        **params: Any,
    ):
        """
        Initialize OnlinePerceptronEquations.

        Args:
            rho: Teacher norm.
            lr: Learning rate.

        """
        super().__init__(rho=rho, lr=lr, **params)
        self.rho = rho
        self.lr = lr

    def _H(self, x: float) -> float:
        """Complementary Gaussian CDF."""
        return 0.5 * (1 - erf(x / np.sqrt(2)))

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Compute dm/dt and dq/dt for online perceptron."""
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)

        # Stability parameter
        kappa = m / np.sqrt(q * rho + 1e-10)
        kappa = np.clip(kappa, -10, 10)

        # Error rate
        epsilon = self._H(kappa)

        # Gaussian density at stability
        phi_kappa = np.exp(-(kappa**2) / 2) / np.sqrt(2 * np.pi)

        # ODE equations (Saad & Solla style)
        dm_dt = lr * np.sqrt(rho) * phi_kappa / np.sqrt(q + 1e-10)
        dq_dt = lr**2 * 2 * epsilon

        return np.array([dm_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) * arccos(m / sqrt(q * ρ))
        """
        m, q = y
        rho = kwargs.get("rho", self.rho)

        if q > 0 and rho > 0:
            cos_angle = m / np.sqrt(q * rho)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5


class OnlineRidgeEquations(OnlineSGDEquations):
    """
    ODE equations for online ridge regression.

    Same as OnlineSGDEquations but with explicit ridge naming.
    """

    pass


class OnlineLogisticEquations(OnlineEquations):
    """
    ODE equations for online logistic regression.

    Binary classification with logistic loss and SGD update.

    The order parameter dynamics involve Gaussian integrals
    over the joint distribution of teacher and student fields.
    """

    def __init__(
        self,
        rho: float = 1.0,
        lr: float = 0.1,
        reg_param: float = 0.0,
        **params: Any,
    ):
        """
        Initialize OnlineLogisticEquations.

        Args:
            rho: Teacher norm (||W0||^2 / d).
            lr: Learning rate η.
            reg_param: L2 regularization λ.

        """
        super().__init__(rho=rho, lr=lr, reg_param=reg_param, **params)
        self.rho = rho
        self.lr = lr
        self.reg_param = reg_param

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute dm/dt and dq/dt for online logistic regression.

        The dynamics depend on expectations over the joint Gaussian
        distribution of teacher (u) and student (z) fields.
        """
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)
        lam = params.get("reg_param", self.reg_param)

        # Correlation and variance
        rho_corr = m / np.sqrt(rho * q) if q > 0 and rho > 0 else 0.0
        rho_corr = np.clip(rho_corr, -0.999, 0.999)

        # Conditional variance of z given u
        var_z_given_u = q * (1 - rho_corr**2)

        # Monte Carlo estimation of the gradient expectations
        n_samples = 1000
        np.random.seed(int(t * 1000) % 2**31)

        # Sample teacher field u ~ N(0, rho)
        u_samples = np.random.randn(n_samples) * np.sqrt(rho)
        y_teacher = np.sign(u_samples)

        # Sample student field z | u ~ N(rho_corr * sqrt(q/rho) * u, var_z_given_u)
        z_mean = rho_corr * np.sqrt(q / (rho + 1e-10)) * u_samples
        z_samples = z_mean + np.sqrt(var_z_given_u + 1e-10) * np.random.randn(n_samples)

        # Logistic gradient: g(y, z) = y * sigmoid(-y * z)
        sigmoid_arg = -y_teacher * z_samples
        g = y_teacher * self._sigmoid(sigmoid_arg)

        # Expectations
        E_g_u = np.mean(g * u_samples / np.sqrt(rho + 1e-10))
        E_g_z = np.mean(g * z_samples / np.sqrt(q + 1e-10))
        E_g2 = np.mean(g**2)

        # ODE equations
        dm_dt = lr * np.sqrt(rho) * E_g_u - lr * lam * m
        dq_dt = 2 * lr * np.sqrt(q + 1e-10) * E_g_z + lr**2 * E_g2 - 2 * lr * lam * q

        return np.array([dm_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) * arccos(m / sqrt(q * ρ))
        """
        m, q = y
        rho = kwargs.get("rho", self.rho)

        if q > 0 and rho > 0:
            cos_angle = m / np.sqrt(q * rho)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5


class OnlineHingeEquations(OnlineEquations):
    """
    ODE equations for online SVM/hinge loss learning.

    Uses hinge loss: L(y, z) = max(0, 1 - y*z)
    """

    def __init__(
        self,
        rho: float = 1.0,
        lr: float = 0.1,
        margin: float = 1.0,
        reg_param: float = 0.0,
        **params: Any,
    ):
        """
        Initialize OnlineHingeEquations.

        Args:
            rho: Teacher norm.
            lr: Learning rate.
            margin: Hinge loss margin.
            reg_param: L2 regularization.

        """
        super().__init__(rho=rho, lr=lr, margin=margin, reg_param=reg_param, **params)
        self.rho = rho
        self.lr = lr
        self.margin = margin
        self.reg_param = reg_param

    def _H(self, x: float) -> float:
        """Complementary Gaussian CDF."""
        return 0.5 * (1 - erf(x / np.sqrt(2)))

    def _phi(self, x: float) -> float:
        """Gaussian PDF."""
        return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Compute dm/dt and dq/dt for online hinge loss."""
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)
        kappa = params.get("margin", self.margin)
        lam = params.get("reg_param", self.reg_param)

        # Stability parameter
        stability = m / np.sqrt(q * rho + 1e-10)
        stability = np.clip(stability, -10, 10)

        # Effective threshold for hinge
        Delta = np.sqrt(q * (1 - stability**2) + 1e-10)
        threshold = (kappa - m * stability / np.sqrt(q + 1e-10)) / Delta

        # Probability of margin violation
        prob_violation = self._H(-threshold)

        # Gradient contributions (from violated samples)
        phi_threshold = self._phi(threshold)

        # ODE equations
        dm_dt = lr * np.sqrt(rho) * phi_threshold / Delta - lr * lam * m
        dq_dt = lr**2 * 2 * prob_violation - 2 * lr * lam * q

        return np.array([dm_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute classification error."""
        m, q = y
        rho = kwargs.get("rho", self.rho)

        if q > 0 and rho > 0:
            cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5


class OnlineCommitteeEquations(OnlineEquations):
    """
    ODE equations for online learning of soft committee machine.

    Tracks overlap matrices Q (student-student) and M (student-teacher).
    More complex than linear case due to hidden layer.
    """

    def __init__(
        self,
        k_student: int = 2,
        k_teacher: int = 2,
        rho: float = 1.0,
        lr: float = 0.1,
        activation: str = "erf",
        **params: Any,
    ):
        """
        Initialize OnlineCommitteeEquations.

        Args:
            k_student: Number of student hidden units.
            k_teacher: Number of teacher hidden units.
            rho: Teacher norm per unit.
            lr: Learning rate.
            activation: Activation function ('erf', 'relu').

        """
        super().__init__(
            k_student=k_student,
            k_teacher=k_teacher,
            rho=rho,
            lr=lr,
            activation=activation,
            **params,
        )
        self.k_student = k_student
        self.k_teacher = k_teacher
        self.rho = rho
        self.lr = lr
        self.activation = activation

        # Total number of order parameters:
        # Q: k_student x k_student (symmetric) -> k_student * (k_student + 1) / 2
        # M: k_student x k_teacher
        self.n_Q = k_student * (k_student + 1) // 2
        self.n_M = k_student * k_teacher

    def _I2(self, a: float, b: float) -> float:
        """Integral I2 for erf activation."""
        return np.arcsin(a * b / np.sqrt((1 + a) * (1 + b))) / np.pi

    def _I3(self, a: float, b: float, c: float) -> float:
        """Integral I3 for erf activation (simplified)."""
        return self._I2(a, c) * self._I2(b, c)

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute dynamics for committee machine.

        WARNING: This is a placeholder implementation.
        Full committee machine dynamics require careful handling of
        the overlap matrices Q (student-student) and M (student-teacher).

        For proper implementation, see:
        - Saad & Solla (1995) "On-line learning in soft committee machines"
        - Biehl & Schwarze (1995) "Learning by on-line gradient descent"

        Raises:
            NotImplementedError: This method is not yet fully implemented.

        """
        raise NotImplementedError(
            "OnlineCommitteeEquations is not fully implemented. "
            "For committee machine online learning dynamics, please implement "
            "the proper ODE system following Saad & Solla (1995) or provide "
            "a custom equations function to ODESolver."
        )

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute generalization error for committee machine."""
        # Placeholder
        return np.sum(y**2) * 0.1

    def get_order_param_names(self) -> list[str]:
        """Return names of order parameters."""
        names = []
        for i in range(self.k_student):
            for j in range(i, self.k_student):
                names.append(f"Q_{i}{j}")
        for i in range(self.k_student):
            for j in range(self.k_teacher):
                names.append(f"M_{i}{j}")
        return names
