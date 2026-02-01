"""
Scenario: Gaussian Data × Linear Model × Ridge Loss

Analytical saddle-point equations for ridge regression.

Data: x ~ N(0, (1/d)I), y = w*ᵀx + ε
Model: Linear regression f(x) = wᵀx
Loss: Ridge L = (1/2)Σ(y - wᵀx)² + (λ/2)||w||²

Order parameters:
    m = w · w* / d  : teacher-student overlap
    q = ||w||² / d  : self-overlap
    ρ = ||w*||² / d : teacher norm (typically = 1)

Saddle-point equations (β→∞, RS):
    (1) m = α(ρ - m) / (λ + ρ - m)
        ⟺ m² - (ρ + λ + α)m + αρ = 0

    (2) q = 2m - ρ + σ²(ρ-m)/λ + (λ-σ²)(ρ-m)A²/(λ(A² + αλ))
        where A = λ + ρ - m

Generalization error:
    E_g = (1/2)(q + ρ - 2m)

References:
    - Engel, Van den Broeck (2001). Statistical Mechanics of Learning
    - Hastie et al. (2022). Ann. Statist.

"""

from typing import Any

import numpy as np

from statphys.theory.replica.scenario.base import ReplicaEquations


class GaussianLinearRidgeEquations(ReplicaEquations):
    """
    Saddle-point equations for ridge regression.

    Teacher-student setup with linear teacher:
        y = w*ᵀx + ε, where x ~ N(0, (1/d)I)

    Ridge regression minimizes:
        L = (1/2) Σ (yᵢ - wᵀxᵢ)² + (λ/2)||w||²

    Order parameters:
        m = w · w* / d : teacher-student overlap
        q = ||w||² / d : self-overlap

    Saddle-point equations for thermodynamic limit:
        m² - (ρ + λ + α)m + αρ = 0
        q = 2m - ρ + σ²(ρ-m)/λ + (λ-σ²)(ρ-m)A²/(λ(A² + αλ))

    Generalization error:
        E_g = (1/2)(ρ - 2m + q)
    """

    def __init__(
        self,
        rho: float = 1.0,
        eta: float = 0.0,
        reg_param: float = 0.01,
        eps: float = 1e-10,
        **params: Any,
    ):
        """
        Initialize GaussianLinearRidgeEquations.

        Args:
            rho: Teacher norm (||w*||²/d). Default 1.0.
            eta: Noise variance σ². Default 0.0.
            reg_param: Ridge parameter λ. Default 0.01.
            eps: Small constant for numerical stability.

        """
        super().__init__(rho=rho, eta=eta, reg_param=reg_param, eps=eps, **params)
        self.rho = rho
        self.eta = eta
        self.reg_param = reg_param
        self.eps = eps

    def solve_m(self, alpha: float, **kwargs: Any) -> float:
        """
        Solve saddle-point equation for m.

        m² - (ρ + λ + α)m + αρ = 0

        The physical solution is the smaller root.

        Args:
            alpha: Sample ratio n/d
            **kwargs: Can override rho, reg_param

        Returns:
            m* : teacher-student overlap

        """
        rho = kwargs.get("rho", self.rho)
        lam = kwargs.get("reg_param", self.reg_param)

        # Quadratic formula: m² - (ρ + λ + α)m + αρ = 0
        a = 1.0
        b = -(rho + lam + alpha)
        c = alpha * rho

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            # Shouldn't happen for physical parameters
            return rho / 2

        # Physical solution is the smaller root
        m = (-b - np.sqrt(discriminant)) / (2 * a)
        return m

    def solve_q(self, alpha: float, m: float, **kwargs: Any) -> float:
        """
        Solve saddle-point equation for q given m.

        q = 2m - ρ + σ²(ρ-m)/λ + (λ-σ²)(ρ-m)A²/(λ(A² + αλ))
        where A = λ + ρ - m

        Args:
            alpha: Sample ratio n/d
            m: Teacher-student overlap
            **kwargs: Can override rho, eta, reg_param

        Returns:
            q* : self-overlap

        """
        rho = kwargs.get("rho", self.rho)
        sigma_sq = kwargs.get("eta", self.eta)
        lam = kwargs.get("reg_param", self.reg_param)

        A = lam + rho - m

        # q = 2m - ρ + σ²(ρ-m)/λ + (λ-σ²)(ρ-m)A²/(λ(A² + αλ))
        term1 = 2 * m - rho
        term2 = sigma_sq * (rho - m) / (lam + self.eps)
        term3 = (lam - sigma_sq) * (rho - m) * A**2 / (lam * (A**2 + alpha * lam) + self.eps)

        q = term1 + term2 + term3
        return q

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Compute analytical m* and q* (ignores input m, q).

        For Ridge regression, the solution is analytical and doesn't require
        iterative fixed-point methods. This method returns the analytical
        solution directly.

        Args:
            m: Ignored (for interface compatibility)
            q: Ignored (for interface compatibility)
            alpha: Sample ratio n/d
            **kwargs: Can override rho, eta, reg_param

        Returns:
            (m*, q*) analytical solution

        """
        return self.analytical_solution(alpha, **kwargs)

    def generalization_error(
        self,
        m: float,
        q: float,
        alpha: float = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error.

        E_g = (1/2)(ρ - 2m + q) = (1/2d)||w - w*||²

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

        Solves the saddle-point equations:
            m² - (ρ + λ + α)m + αρ = 0
            q = 2m - ρ + σ²(ρ-m)/λ + (λ-σ²)(ρ-m)A²/(λ(A² + αλ))

        Args:
            alpha: Sample ratio n/d
            **kwargs: Can override rho, eta, reg_param

        Returns:
            (m*, q*) analytical fixed point solution

        """
        m_star = self.solve_m(alpha, **kwargs)
        q_star = self.solve_q(alpha, m_star, **kwargs)
        return m_star, q_star

    def solve_all(
        self,
        alpha_values: np.ndarray | list[float],
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """
        Solve for all alpha values and return m, q, E_g.

        Args:
            alpha_values: Array of sample ratio values
            **kwargs: Can override rho, eta, reg_param

        Returns:
            Dictionary with 'm', 'q', 'eg' arrays

        """
        alpha_values = np.asarray(alpha_values)
        m_values = np.zeros_like(alpha_values)
        q_values = np.zeros_like(alpha_values)
        eg_values = np.zeros_like(alpha_values)

        for i, alpha in enumerate(alpha_values):
            m, q = self.analytical_solution(alpha, **kwargs)
            m_values[i] = m
            q_values[i] = q
            eg_values[i] = self.generalization_error(m, q, **kwargs)

        return {
            "m": m_values,
            "q": q_values,
            "eg": eg_values,
        }
