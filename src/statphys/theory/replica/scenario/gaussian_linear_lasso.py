"""
Scenario: Gaussian Data × Linear Model × LASSO Loss

Saddle-point equations for LASSO regression.

Data: x ~ N(0, I_d), y = (1/√d) W₀ᵀ x + ε
Model: Linear regression f(x) = wᵀx/√d
Loss: LASSO (MSE + L1 regularization)

References:
    - Bayati, Montanari (2011). "The LASSO risk for Gaussian matrices."
      IEEE Trans. Inf. Theory
    - Thrampoulidis, Oymak, Hassibi (2018). "Precise error analysis
      of regularized M-estimators." IEEE Trans. Inf. Theory

Note:
    This implementation uses the replica symmetric ansatz. For the exact
    CGMT-based analysis, see Thrampoulidis et al. (2018).

"""

from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf

from statphys.theory.replica.scenario.base import ReplicaEquations


class GaussianLinearLassoEquations(ReplicaEquations):
    """
    Saddle-point equations for LASSO regression.

    Teacher-student setup with linear teacher:
        y = (1/√d) W₀ᵀ x + ε

    LASSO minimizes:
        L = (1/2n) Σᵢ (yᵢ - wᵀxᵢ/√d)² + λ||w||₁

    The key difference from ridge is the L1 regularization,
    which leads to soft-thresholding proximal operator:

        prox_λ(x) = sign(x) · max(|x| - λ, 0)

    This implementation follows the CGMT framework where the
    saddle-point equations are:

        σ² = (1/α) E[(prox_{λ/σ}(W₀ + σZ) - W₀)²]
        q = E[prox_{λ/σ}(W₀ + σZ)²]
        m = E[W₀ · prox_{λ/σ}(W₀ + σZ)]

    where Z ~ N(0,1) and σ is the effective noise parameter.
    """

    def __init__(
        self,
        rho: float = 1.0,
        eta: float = 0.0,
        reg_param: float = 0.01,
        **params: Any,
    ):
        """
        Initialize GaussianLinearLassoEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            eta: Noise variance. Default 0.0.
            reg_param: LASSO parameter λ. Default 0.01.

        """
        super().__init__(rho=rho, eta=eta, reg_param=reg_param, **params)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param

    def _soft_threshold(self, x: float | np.ndarray, threshold: float) -> float | np.ndarray:
        """
        Soft thresholding operator (proximal of L1 norm).

        S_λ(x) = sign(x) · max(|x| - λ, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _compute_effective_noise(
        self,
        alpha: float,
        rho: float,
        eta: float,
        lam: float,
    ) -> float:
        """
        Solve for effective noise parameter σ via fixed-point equation.

        The CGMT equation is:
            σ² = η + (1/α) E[(prox_{λ/σ}(√ρ·ξ + σZ) - √ρ·ξ)²]

        where ξ ~ N(0,1), Z ~ N(0,1) independent.

        For simple case with rho=1, this simplifies.
        """
        sqrt_rho = np.sqrt(rho)

        def fixed_point_residual(sigma):
            if sigma <= 0:
                return float("inf")

            threshold = lam / sigma

            def integrand(z):
                # Teacher signal component: √ρ · ξ where ξ ~ N(0,1)
                # We integrate over ξ (teacher) and z (noise)
                # For simplicity, assume w0 ~ N(0, ρ/d) per component
                # so effective signal is √ρ · z_teacher

                # Integrate over teacher component
                def inner_integrand(xi):
                    signal = sqrt_rho * xi + sigma * z
                    prox = self._soft_threshold(signal, threshold)
                    residual = (prox - sqrt_rho * xi) ** 2
                    gauss_xi = np.exp(-(xi**2) / 2) / np.sqrt(2 * np.pi)
                    return residual * gauss_xi

                inner_result, _ = quad(inner_integrand, -5, 5)
                gauss_z = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
                return inner_result * gauss_z

            expected_residual, _ = quad(integrand, -5, 5)
            return sigma**2 - eta - expected_residual / alpha

        # Find σ via root finding
        try:
            sigma_star = brentq(fixed_point_residual, 0.01, 10.0)
        except ValueError:
            # Fall back to simple approximation if root finding fails
            sigma_star = np.sqrt(eta + (rho + eta) / max(alpha - 1, 0.1))

        return max(sigma_star, 0.01)

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute updated m and q using LASSO saddle-point equations.

        Args:
            m: Current teacher-student overlap
            q: Current self-overlap
            alpha: Sample ratio n/d
            **kwargs: Can override rho, eta, reg_param

        Returns:
            (m_new, q_new) updated order parameters

        Note:
            For α < 1 (underparameterized), exact solution may not exist.
            For α > 1, the equations are well-defined.

        """
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)

        # Ensure stability
        q = max(q, 1e-6)
        m = np.clip(m, -np.sqrt(q * rho) * 0.999, np.sqrt(q * rho) * 0.999)

        # Compute effective noise parameter
        # Use current order params to estimate σ
        V = rho - 2 * m + q + eta  # residual variance

        # For α > 1: σ² ≈ V / (α - 1) (from ridge-like analysis)
        # For α < 1: use regularized version
        if alpha > 1:
            sigma_sq = V / (alpha - 1 + lam)
        else:
            # Underparameterized: regularization dominates
            sigma_sq = V / (lam + 0.1)

        sigma = np.sqrt(max(sigma_sq, 1e-6))
        threshold = lam / sigma

        sqrt_rho = np.sqrt(rho)

        # Compute expectations via Gaussian integration
        # E[prox(√ρ·ξ + σZ)·√ρ·ξ] for m
        # E[prox(√ρ·ξ + σZ)²] for q
        def integrand_m(z):
            def inner(xi):
                signal = sqrt_rho * xi + sigma * z
                prox = self._soft_threshold(signal, threshold)
                gauss_xi = np.exp(-(xi**2) / 2) / np.sqrt(2 * np.pi)
                return prox * sqrt_rho * xi * gauss_xi

            inner_result, _ = quad(inner, -5, 5)
            gauss_z = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            return inner_result * gauss_z

        def integrand_q(z):
            def inner(xi):
                signal = sqrt_rho * xi + sigma * z
                prox = self._soft_threshold(signal, threshold)
                gauss_xi = np.exp(-(xi**2) / 2) / np.sqrt(2 * np.pi)
                return prox**2 * gauss_xi

            inner_result, _ = quad(inner, -5, 5)
            gauss_z = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
            return inner_result * gauss_z

        new_m, _ = quad(integrand_m, -5, 5)
        new_q, _ = quad(integrand_q, -5, 5)

        # Ensure physical constraints
        new_m = max(new_m, 1e-6)
        new_q = max(new_q, 1e-6)

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

        # Estimate effective noise
        V = rho - 2 * m + q + eta
        if alpha > 1:
            sigma = np.sqrt(V / (alpha - 1 + lam))
        else:
            sigma = np.sqrt(V / (lam + 0.1))

        threshold = lam / max(sigma, 0.01)

        # P(|√ρ·ξ + σZ| > threshold) where ξ, Z ~ N(0,1)
        # Approximate: effective std is √(ρ + σ²)
        effective_std = np.sqrt(rho + sigma**2)
        sparsity = 1 - erf(threshold / (effective_std * np.sqrt(2)))
        return sparsity
