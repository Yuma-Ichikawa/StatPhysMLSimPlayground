"""Saddle-point equations for LASSO regression."""

from typing import Any

import numpy as np
from scipy.integrate import quad

from statphys.theory.replica.models.base import ReplicaEquations


class LassoEquations(ReplicaEquations):
    """
    Saddle-point equations for LASSO regression.

    Teacher-student setup with linear teacher:
        y = (1/√d) W₀ᵀ x + ε

    LASSO minimizes:
        L = (1/n) Σᵢ (yᵢ - wᵀxᵢ/√d)² + λ||w||₁

    The key difference from ridge is the L1 regularization,
    which leads to soft-thresholding proximal operator:

        prox_λ(x) = sign(x) · max(|x| - λ, 0)

    Saddle-point equations involve Gaussian integrals over
    the effective field distribution:

        m_new = ∫ Dz · prox_{λ/√q̂}(ω + √q̂ z) · (√ρ m/√q)
        q_new = ∫ Dz · [prox_{λ/√q̂}(ω + √q̂ z)]²

    where Dz = (1/√(2π)) exp(-z²/2) dz is the Gaussian measure.

    References:
        - Bayati, Montanari (2011). "The LASSO risk for Gaussian matrices."
          IEEE Trans. Inf. Theory
        - Thrampoulidis, Oymak, Hassibi (2018). "Precise error analysis
          of regularized M-estimators." IEEE Trans. Inf. Theory

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
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            eta: Noise variance. Default 0.0.
            reg_param: LASSO parameter λ. Default 0.01.

        """
        super().__init__(rho=rho, eta=eta, reg_param=reg_param, **params)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param

    def _soft_threshold(self, x: float, threshold: float) -> float:
        """
        Soft thresholding operator (proximal of L1 norm).

        S_λ(x) = sign(x) · max(|x| - λ, 0)

        This is the proximal operator of f(x) = λ|x|.
        """
        return np.sign(x) * max(abs(x) - threshold, 0)

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q using LASSO saddle-point equations.

        Uses Gaussian integration over the effective field.

        Args:
            m: Current teacher-student overlap
            q: Current self-overlap
            alpha: Sample ratio n/d
            **kwargs: Can override rho, eta, reg_param

        Returns:
            (m_new, q_new) updated order parameters

        """
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)

        # Ensure positive q for stability
        q = max(q, 0.001)

        # Residual variance
        V = rho - 2 * m + q + eta

        # Conjugate variable (approximate for LASSO)
        # In underparameterized regime: use this approximation
        hat_q = alpha * V / max(1 - alpha, 0.01)
        hat_q = max(hat_q, 0.001)

        # Effective threshold for proximal operator
        threshold = lam / np.sqrt(hat_q + 0.001)

        # Gaussian integrals for m and q updates
        def integrand_m(z):
            """Integrand for m update."""
            effective_signal = np.sqrt(rho) * m / np.sqrt(q + 0.001) + np.sqrt(hat_q) * z
            proximal = self._soft_threshold(effective_signal, threshold)
            gaussian = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            return proximal * np.sqrt(rho) / np.sqrt(q + 0.001) * gaussian

        def integrand_q(z):
            """Integrand for q update."""
            effective_signal = np.sqrt(rho) * m / np.sqrt(q + 0.001) + np.sqrt(hat_q) * z
            proximal = self._soft_threshold(effective_signal, threshold)
            gaussian = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            return proximal**2 * gaussian

        # Numerical integration
        new_m, _ = quad(integrand_m, -10, 10)
        new_q, _ = quad(integrand_q, -10, 10)

        # Ensure physical constraints
        new_m = max(new_m, 0.001)
        new_q = max(new_q, 0.001)

        return new_m, new_q

    def generalization_error(
        self,
        m: float,
        q: float,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error.

        E_g = (1/2)(ρ - 2m + q)

        Args:
            m: Teacher-student overlap
            q: Self-overlap
            **kwargs: Can override rho

        Returns:
            Generalization error

        """
        rho = kwargs.get("rho", self.rho)
        return 0.5 * (rho - 2 * m + q)

    def sparsity(self, m: float, q: float, alpha: float, **kwargs: Any) -> float:
        """
        Estimate sparsity level (fraction of non-zero weights).

        For LASSO, the sparsity depends on the threshold relative
        to the typical signal magnitude.

        Args:
            m, q: Order parameters
            alpha: Sample ratio
            **kwargs: Model parameters

        Returns:
            Estimated fraction of non-zero weights

        """
        lam = kwargs.get("reg_param", self.reg_param)
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)

        # Rough estimate based on threshold
        V = rho - 2 * m + q + eta
        hat_q = alpha * V / max(1 - alpha, 0.01)
        threshold = lam / np.sqrt(hat_q + 0.001)

        # Fraction above threshold
        from scipy.special import erf

        sparsity = 1 - erf(threshold / np.sqrt(2))
        return sparsity
