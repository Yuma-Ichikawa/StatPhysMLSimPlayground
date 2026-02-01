"""
Scenario: Gaussian Data × Committee Machine × MSE Loss

Saddle-point equations for committee machine.

Data: x ~ N(0, I_d), y = (1/√M) Σ_m φ(v*_m^T x / √d)
Model: Committee machine f(x) = (1/√K) Σ_k φ(v_k^T x / √d)
Loss: MSE (Mean Squared Error)

References:
    - Schwarze, Hertz (1993). "Generalization in multilayer
      neural networks." Europhys. Lett.
    - Saad, Solla (1995). "Exact asymptotic analysis
      of committee machine learning." Phys. Rev. E

"""

from typing import Any

import numpy as np
from scipy.special import erf

from statphys.theory.replica.scenario.base import ReplicaEquations


class GaussianCommitteeMseEquations(ReplicaEquations):
    """
    Saddle-point equations for committee machine.

    Student: f(x) = (1/√K) Σ_k φ(v_k^T x / √d)
    Teacher: y = (1/√M) Σ_m φ(v*_m^T x / √d)

    Order parameters are matrices:
        Q_{kk'} = (1/d) v_k^T v_{k'}
        R_{km} = (1/d) v_k^T v*_m
        T_{mm'} = (1/d) v*_m^T v*_{m'}

    For the symmetric ansatz (all students equivalent):
        - Q_kk = q (self-overlap)
        - Q_{kk'} = c (off-diagonal overlap)
        - R_km = r (student-teacher overlap)

    This implementation uses the symmetric ansatz where
    m and q are scalar order parameters.

    Correlation functions for erf activation:
        I_2(a, b) = (2/π) arcsin(ab / √((1+a)(1+b)))
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
        Initialize GaussianCommitteeMseEquations.

        Args:
            K: Number of student hidden units. Default 2.
            M: Number of teacher hidden units. Default 2.
            rho: Teacher weight norm per unit. Default 1.0.
            eta: Noise variance. Default 0.0.
            activation: Activation function ('erf', 'tanh', 'sign'). Default 'erf'.
            reg_param: L2 regularization parameter. Default 0.01.

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

    def _I2(self, a: float, b: float) -> float:
        """
        Correlation integral I_2 for erf activation.

        I_2(a, b) = (2/π) arcsin(ab / √((1+a)(1+b)))
        """
        if (1 + a) <= 0 or (1 + b) <= 0:
            return 0.0
        arg = a * b / np.sqrt((1 + a) * (1 + b) + 1e-10)
        arg = np.clip(arg, -1, 1)
        return (2 / np.pi) * np.arcsin(arg)

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

        Args:
            m: Average student-teacher overlap
            q: Average self-overlap
            alpha: Sample ratio n/d
            **kwargs: Can override parameters

        Returns:
            (m_new, q_new) updated order parameters

        """
        rho = kwargs.get("rho", self.rho)
        kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)
        K = kwargs.get("K", self.K)

        # For symmetric ansatz:
        # Q_kk' = q * δ_{kk'} + c * (1 - δ_{kk'})
        # R_km = r for all k, m
        c = 0.0  # Off-diagonal (could be another order parameter)

        # Generalization error contribution
        eg = rho - 2 * m + q + (K - 1) * c

        # Update equations (simplified for K=M symmetric case)
        lr = 0.1

        # Approximate updates using symmetric ansatz
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
        E_g = (1/π) arccos(...) + noise contributions

        For MSE-style loss:
        E_g = (1/2)(ρ - 2m + q)

        Args:
            m: Average student-teacher overlap
            q: Average self-overlap
            **kwargs: Can override parameters

        Returns:
            Generalization error

        """
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)

        # MSE-like error for regression
        eg = 0.5 * (rho - 2 * m + q)
        if eta > 0:
            eg += 0.5 * eta

        return eg

    def get_order_param_names(self) -> list[str]:
        """Return names of order parameters."""
        return ["m", "q"]
