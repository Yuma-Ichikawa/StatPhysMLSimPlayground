"""
Scenario: Gaussian Data × Linear Model × Hinge Loss (Online SVM)

ODE equations for online SVM/hinge loss learning.

Data: x ~ N(0, I_d), y = sign((1/√d) W₀ᵀ x)
Model: Linear classifier f(x) = wᵀx/√d
Loss: Hinge loss ℓ(y, z) = max(0, κ - yz)

References:
    - Dietrich, Opper, Sompolinsky (1999). Phys. Rev. Lett.
    - Gardner (1988). J. Phys. A

"""

from typing import Any

import numpy as np
from scipy.special import roots_hermite

from statphys.theory.online.scenario.base import OnlineEquations
from statphys.utils.constants import (
    DEFAULT_GH_POINTS,
    DEFAULT_ONLINE_LR,
    EPS_DIV,
    EPS_NORM,
)
from statphys.utils.special_functions import classification_error_linear, gaussian_cdf


class GaussianLinearHingeEquations(OnlineEquations):
    """
    ODE equations for online SVM/hinge loss learning.

    Teacher-student setup for binary classification:
        y = sign((1/√d) W₀ᵀ x)

    Hinge loss: ℓ(y, z) = max(0, κ - yz)
    where κ is the margin parameter. The SGD gradient is
        g(y, z) = y · Θ(κ - yz)
    (update whenever the margin is violated).

    In the d → ∞ limit with t = τ/d, order parameter dynamics:

        dm/dt = η E[g(y,z) u] - ηλm
        dq/dt = 2η E[g(y,z) z] + η² E[g²] - 2ηλq

    where expectations are over the joint Gaussian fields
    u = W₀ᵀx/√d ~ N(0, ρ), z = wᵀx/√d ~ N(0, q) with Cov(u, z) = m,
    and y = sign(u). They are evaluated by numerical quadrature.

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))
    """

    def __init__(
        self,
        rho: float = 1.0,
        lr: float = DEFAULT_ONLINE_LR,
        margin: float = 1.0,
        reg_param: float = 0.0,
        n_quad: int = DEFAULT_GH_POINTS,
        **params: Any,
    ):
        """
        Initialize GaussianLinearHingeEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            lr: Learning rate η. Default 0.1.
            margin: Hinge loss margin κ. Default 1.0 (standard SVM).
            reg_param: L2 regularization λ. Default 0.0.
            n_quad: Number of Gauss-Hermite quadrature points.

        """
        super().__init__(rho=rho, lr=lr, margin=margin, reg_param=reg_param, **params)
        self.rho = rho
        self.lr = lr
        self.margin = margin
        self.reg_param = reg_param
        self.n_quad = n_quad

    def _compute_expectations(
        self,
        m: float,
        q: float,
        rho: float,
        kappa: float,
    ) -> tuple[float, float, float]:
        """
        Compute (E[g u], E[g z], E[g²]) for g = y Θ(κ - y z), y = sign(u).

        Uses the decomposition u = √ρ s, z = √q (c s + √(1-c²) ζ) with
        s, ζ iid N(0,1) and c = m/√(qρ). Conditioning on s and integrating
        the (linear-in-ζ) integrands analytically over the margin-violation
        region leaves 1D Gauss-Hermite quadrature over s.
        """
        q = max(q, EPS_DIV)
        rho = max(rho, EPS_DIV)
        max_m = np.sqrt(q * rho) * 0.9999
        m = float(np.clip(m, -max_m, max_m))
        c = m / np.sqrt(q * rho)
        s_perp = np.sqrt(max(1 - c**2, EPS_NORM))

        x_nodes, w_nodes = roots_hermite(self.n_quad)
        s = np.sqrt(2.0) * x_nodes  # s ~ N(0,1) nodes
        w = w_nodes / np.sqrt(np.pi)

        y_sign = np.where(s >= 0, 1.0, -1.0)

        # Margin violation: y z < κ  <=>  y(c s + s_perp ζ) < κ/√q
        # For y=+1: ζ < a with a = (κ/√q - c s)/s_perp
        # For y=-1: ζ > -a' with a' = (κ/√q + c s)/s_perp (by symmetry handled via y)
        a = (kappa / np.sqrt(q) - y_sign * c * s) / s_perp

        # P(violation | s): Φ(a) for y=+1; for y=-1 the event is ζ > -a, also Φ(a)
        prob = gaussian_cdf(a)
        phi_a = np.exp(-0.5 * a**2) / np.sqrt(2 * np.pi)

        # E[ζ 1{ζ < a}] = -φ(a); for y=-1, E[ζ 1{ζ > -a}] = φ(-a) = φ(a)
        e_zeta = np.where(y_sign >= 0, -phi_a, phi_a)

        # g = y on the violation event
        # E[g u] = Σ w · y √ρ s · P(violation|s)
        E_gu = float(np.sum(w * y_sign * np.sqrt(rho) * s * prob))
        # E[g z] = Σ w · y √q (c s P + s_perp E[ζ 1])
        E_gz = float(np.sum(w * y_sign * np.sqrt(q) * (c * s * prob + s_perp * e_zeta)))
        # E[g²] = P(violation)
        E_g2 = float(np.sum(w * prob))

        return E_gu, E_gz, E_g2

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute dm/dt and dq/dt for online hinge loss.

        ODE system:
            dm/dt = η E[g u] - ηλm
            dq/dt = 2η E[g z] + η² E[g²] - 2ηλq

        with g = y Θ(κ - yz), evaluated by quadrature.

        Args:
            t: Normalized time t = τ/d
            y: [m, q] order parameters
            params: Can override rho, lr, margin, reg_param

        Returns:
            [dm/dt, dq/dt]

        """
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)
        kappa = params.get("margin", self.margin)
        lam = params.get("reg_param", self.reg_param)

        E_gu, E_gz, E_g2 = self._compute_expectations(m, q, rho, kappa)

        dm_dt = lr * E_gu - lr * lam * m
        dq_dt = 2 * lr * E_gz + lr**2 * E_g2 - 2 * lr * lam * q

        return np.array([dm_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute classification error.

        P(error) = (1/π) arccos(m/√(qρ))

        Args:
            y: [m, q] order parameters
            **kwargs: Can override rho

        Returns:
            Classification error probability

        """
        m, q = y
        rho = kwargs.get("rho", self.rho)
        return classification_error_linear(m, q, rho)
