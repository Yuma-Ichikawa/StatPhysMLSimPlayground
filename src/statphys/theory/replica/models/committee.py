"""Saddle-point equations for committee machine."""

from typing import Any

import numpy as np
from scipy.special import erf

from statphys.theory.replica.models.base import ReplicaEquations


class CommitteeMachineEquations(ReplicaEquations):
    """
    Saddle-point equations for committee machine.

    Architecture:
        Student: f(x) = (1/√K) Σ_k φ(v_k^T x / √d)
        Teacher: y = (1/√M) Σ_m φ(v*_m^T x / √d)

    where φ is the activation function (erf, tanh, sign, relu).

    Order parameters are matrices:
        Q_{kk'} = (1/d) v_k^T v_{k'} : Student-student overlaps
        R_{km} = (1/d) v_k^T v*_m    : Student-teacher overlaps
        T_{mm'} = (1/d) v*_m^T v*_{m'} : Teacher-teacher overlaps (fixed)

    For symmetric ansatz (all students equivalent):
        Q_kk = q (self-overlap)
        Q_{kk'} = c (off-diagonal)
        R_km = r (all equal)

    The saddle-point equations involve correlation integrals I_1, I_2, I_3
    that depend on the activation function.

    For erf activation:
        I_2(a, b) = (2/π) arcsin(ab / √((1+a)(1+b)))

    Generalization error depends on these integrals.

    References:
        - Saad, Solla (1995). "Exact solution for on-line learning
          in multilayer neural networks." Phys. Rev. Lett. 74, 4337
        - Goldt et al. (2020). "Modeling the influence of data structure
          on learning in neural networks." Phys. Rev. X 10, 041044

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
            K: Number of student hidden units. Default 2.
            M: Number of teacher hidden units. Default 2.
            rho: Teacher weight norm per unit (T_mm = rho). Default 1.0.
            eta: Noise variance. Default 0.0.
            activation: Activation function ('erf', 'tanh', 'sign', 'relu').
            reg_param: L2 regularization λ. Default 0.01.

        """
        super().__init__(
            K=K,
            M=M,
            rho=rho,
            eta=eta,
            activation=activation,
            reg_param=reg_param,
            **params,
        )
        self.K = K
        self.M = M
        self.rho = rho
        self.eta = eta
        self.activation = activation
        self.reg_param = reg_param

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
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

    def _I2(self, a: float, b: float) -> float:
        """
        Correlation integral I_2 for erf activation.

        I_2(a, b) = (2/π) arcsin(ab / √((1+a)(1+b)))

        This appears in the generalization error and update equations.
        """
        if (1 + a) <= 0 or (1 + b) <= 0:
            return 0.0
        arg = a * b / np.sqrt((1 + a) * (1 + b))
        arg = np.clip(arg, -1, 1)
        return 2 * np.arcsin(arg) / np.pi

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q (symmetric ansatz).

        For committee machine with symmetric ansatz, m and q are scalars
        representing the average overlaps.

        Args:
            m: Average student-teacher overlap (mean of R_km)
            q: Average self-overlap (Q_kk)
            alpha: Sample ratio n/d
            **kwargs: Can override K, M, rho, eta, reg_param

        Returns:
            (m_new, q_new) updated order parameters

        """
        rho = kwargs.get("rho", self.rho)
        kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)
        K = kwargs.get("K", self.K)

        # Ensure numerical stability
        q = max(q, 1e-10)

        # For symmetric ansatz: simplified equations
        c = 0.0  # Off-diagonal (could be another order parameter)

        # Generalization error contribution
        eg = rho - 2 * m + q + (K - 1) * c

        # Damped update (simplified for K=M symmetric case)
        lr = 0.1

        # Approximate updates based on correlation integrals
        new_m = m + lr * (alpha * np.sqrt(rho / K) * (rho - m) / (1 + eg + lam) - lam * m)
        new_q = q + lr * (alpha * (eg + m**2) / ((1 + eg + lam) ** 2) - lam * q)

        # Physical constraints
        new_m = max(new_m, 1e-10)
        new_q = max(new_q, 1e-10)

        return new_m, new_q

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error for committee machine.

        For erf activation with symmetric ansatz:
            E_g = (M/π)I_2(T,T) + (K/π)I_2(Q,Q) - (2KM/π)I_2(R/√(Q*T), R/√(Q*T))

        Simplified version:
            E_g ≈ (1/2)(ρ - 2m + q) + noise

        Args:
            m: Average student-teacher overlap
            q: Average self-overlap
            **kwargs: Can override rho, eta, K

        Returns:
            Generalization error

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
        """Return names of order parameters for symmetric ansatz."""
        return ["m", "q"]

    def full_order_param_names(self) -> list[str]:
        """
        Return names for full (non-symmetric) parameterization.

        This would include all Q_{kk'} and R_{km} elements.
        """
        names = []
        for k in range(self.K):
            for kp in range(k, self.K):
                names.append(f"Q_{k}{kp}")
        for k in range(self.K):
            for m in range(self.M):
                names.append(f"R_{k}{m}")
        return names
