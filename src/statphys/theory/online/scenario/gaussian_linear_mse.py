"""
Scenario: Gaussian Data × Linear Model × MSE Loss (Online SGD)

ODE equations for online SGD on linear regression.

Data: x ~ N(0, I_d), y = (1/√d) W₀ᵀ x + ε
Model: Linear regression f(x) = wᵀx/√d
Loss: MSE + optional L2 regularization

References:
    - Werfel, Xie, Seung (2005). "Learning curves for stochastic gradient
      descent in linear feedforward networks." Neural Computation
    - Engel, Van den Broeck (2001). Statistical Mechanics of Learning. Ch. 5
"""

from typing import Any

import numpy as np

from statphys.theory.online.scenario.base import OnlineEquations


class GaussianLinearMseEquations(OnlineEquations):
    """
    ODE equations for online SGD on linear regression.

    Teacher-student setup with linear teacher:
        y = (1/√d) W₀ᵀ x + ε

    where ε ~ N(0, σ²) is output noise.

    SGD update at each step τ:
        w^{τ+1} = w^τ - η ∇ℓ(w^τ)

    In the d → ∞ limit with t = τ/d, order parameter dynamics:

        dm/dt = η(ρ - m) - ηλm
        dq/dt = η²V + 2η(m - q) - 2ηλq

    where:
        - m = wᵀW₀/d : teacher-student overlap
        - q = ||w||²/d : self-overlap
        - V = ρ - 2m + q + σ² : residual variance (training loss)
        - η : learning rate
        - λ : L2 regularization parameter
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
        Initialize GaussianLinearMseEquations.

        Args:
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            eta_noise: Output noise variance σ². Default 0.0.
            lr: Learning rate η. Default 0.1.
            reg_param: L2 regularization λ. Default 0.0.
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
        """
        Compute dm/dt and dq/dt for online SGD.

        ODE system:
            dm/dt = η(ρ - m) - ηλm
            dq/dt = η²(ρ - 2m + q + σ²) + 2η(m - q) - 2ηλq

        Args:
            t: Normalized time t = τ/d
            y: [m, q] order parameters
            params: Can override rho, eta_noise, lr, reg_param

        Returns:
            [dm/dt, dq/dt]
        """
        m, q = y

        # Get parameters (allow override from params dict)
        rho = params.get("rho", self.rho)
        eta_noise = params.get("eta_noise", self.eta_noise)
        lr = params.get("lr", self.lr)
        lam = params.get("reg_param", self.reg_param)

        # Residual variance (= training loss)
        residual_var = rho - 2 * m + q + eta_noise

        # ODE for m: teacher-student overlap
        # dm/dt = η(ρ - m) - ηλm = η(ρ - m(1+λ))
        dm_dt = lr * (rho - m) - lr * lam * m

        # ODE for q: self-overlap
        # dq/dt = η²V + 2η(m - q) - 2ηλq
        # Second moment term from gradient noise: η²V
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

        E_g = (1/2)(ρ - 2m + q)

        This measures E[(f_teacher(x) - f_student(x))²].

        Args:
            y: [m, q] order parameters
            **kwargs: Can override rho

        Returns:
            Generalization error value
        """
        m, q = y
        rho = kwargs.get("rho", self.rho)
        return 0.5 * (rho - 2 * m + q)

    def steady_state(self, **kwargs: Any) -> tuple[float, float]:
        """
        Compute steady state (m*, q*) analytically.

        For η < η_c = 2(1+λ), the steady state exists:
            m* = ρ/(1+λ)
            q* = ρ/(1+λ) + η²σ²/(2(1+λ) - η(1+λ)²)

        Args:
            **kwargs: Can override rho, eta_noise, lr, reg_param

        Returns:
            (m*, q*) steady state values
        """
        rho = kwargs.get("rho", self.rho)
        sigma_sq = kwargs.get("eta_noise", self.eta_noise)
        lr = kwargs.get("lr", self.lr)
        lam = kwargs.get("reg_param", self.reg_param)

        m_star = rho / (1 + lam)

        # Solve for q* from dq/dt = 0
        # η²(ρ - 2m* + q* + σ²) + 2η(m* - q*) - 2ηλq* = 0
        # This gives a linear equation in q*
        a = lr**2 + 2 * lr * (1 + lam)
        b = lr**2 * (rho - 2 * m_star + sigma_sq) + 2 * lr * m_star
        q_star = b / a

        return m_star, q_star
