"""Saddle-point equations for ridge regression."""

from typing import Any

import numpy as np

from statphys.theory.replica.models.base import ReplicaEquations


class RidgeRegressionEquations(ReplicaEquations):
    """
    Saddle-point equations for ridge regression.

    Teacher-student setup with linear teacher:
        y = (1/√d) W₀ᵀ x + ε

    Ridge regression minimizes:
        L = (1/n) Σᵢ (yᵢ - wᵀxᵢ/√d)² + λ||w||²

    Order parameters:
        m = wᵀW₀/d : teacher-student overlap
        q = ||w||²/d : self-overlap

    Saddle-point equations (fixed-point form):

        Residual variance:
            V = ρ - 2m + q + η

        Conjugate variables:
            m̂ = αm / (1 + αq/(λ+ε))
            q̂ = α(V + m²) / (1 + αq/(λ+ε))²

        Update equations:
            m_new = ρm̂ / (λ + q̂)
            q_new = (ρm̂² + q̂(ρ+η)) / (λ + q̂)²

    Saddle-point equations (residual form, 0 = RHS):
        0 = m - ρm̂/(λ + q̂)
        0 = q - (ρm̂² + q̂(ρ+η))/(λ + q̂)²

    Generalization error:
        E_g = (1/2)(ρ - 2m + q)

    References:
        - Advani, Saxe (2017). "High-dimensional dynamics of generalization
          error in neural networks." arXiv:1710.03667
        - Hastie et al. (2022). "Surprises in high-dimensional ridgeless
          least squares interpolation." Ann. Statist.
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
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            eta: Noise variance σ². Default 0.0.
            reg_param: Ridge parameter λ. Default 0.01.
            eps: Small constant for numerical stability. Default 1e-6.
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
        Compute updated m and q (fixed-point iteration).

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
        eps = kwargs.get("eps", self.eps)

        # Residual variance (effective noise)
        V = rho - 2 * m + q + eta

        # Conjugate variables (from replica calculation)
        denominator = 1 + alpha * q / (lam + eps)
        hat_m = alpha * m / denominator
        hat_q = alpha * (V + m**2) / (denominator**2)

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

        E_g = (1/2)(ρ - 2m + q)

        This measures E[(f_teacher(x) - f_student(x))²].

        Args:
            m: Teacher-student overlap
            q: Self-overlap
            alpha: Not used (for interface compatibility)
            **kwargs: Can override rho

        Returns:
            Generalization error
        """
        rho = kwargs.get("rho", self.rho)
        return 0.5 * (rho - 2 * m + q)

    def analytical_solution(self, alpha: float, **kwargs: Any) -> tuple[float, float]:
        """
        Compute analytical solution for ridge regression.

        For ridge regression, the fixed point can be computed analytically
        in some regimes.

        Args:
            alpha: Sample ratio
            **kwargs: Model parameters

        Returns:
            (m*, q*) fixed point solution
        """
        rho = kwargs.get("rho", self.rho)
        eta = kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)

        # High regularization limit
        if lam > 1.0:
            m_star = alpha * rho / (lam + alpha)
            q_star = alpha * (rho + eta) / (lam + alpha) ** 2
            return m_star, q_star

        # Iterative solution for general case
        m, q = 0.5, 0.5
        for _ in range(1000):
            m_new, q_new = self(m, q, alpha, **kwargs)
            if abs(m_new - m) < 1e-10 and abs(q_new - q) < 1e-10:
                break
            m, q = m_new, q_new

        return m, q
