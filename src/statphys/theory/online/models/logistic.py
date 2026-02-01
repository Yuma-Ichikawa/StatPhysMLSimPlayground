"""ODE equations for online logistic regression."""

from typing import Any

import numpy as np

from statphys.theory.online.models.base import OnlineEquations


class OnlineLogisticEquations(OnlineEquations):
    """
    ODE equations for online logistic regression.

    Teacher-student setup for binary classification:
        y = sign((1/√d) W₀ᵀ x)  (deterministic teacher)

    Logistic loss: ℓ(y, z) = log(1 + exp(-yz))
    Logistic gradient: g(y, z) = y · σ(-yz) where σ is sigmoid

    SGD update:
        w^{τ+1} = w^τ - η ∇ℓ = w^τ + η g(y, z) x

    In the d → ∞ limit with t = τ/d, order parameter dynamics:

        dm/dt = η √ρ · E[g(y,z) · u/√ρ] - ηλm
        dq/dt = 2η √q · E[g(y,z) · z/√q] + η² E[g²] - 2ηλq

    where expectations are over the joint Gaussian (u, z) with:
        - u = W₀ᵀx/√d ~ N(0, ρ)  (teacher field)
        - z = wᵀx/√d ~ N(0, q)   (student field)
        - Cov(u, z) = m

    Classification error:
        P(error) = (1/π) arccos(m/√(qρ))

    References:
        - Dietrich, Opper, Sompolinsky (1999). Phys. Rev. Lett.
        - Engel, Van den Broeck (2001). Statistical Mechanics of Learning.
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
            rho: Teacher norm (||W₀||²/d). Default 1.0.
            lr: Learning rate η. Default 0.1.
            reg_param: L2 regularization λ. Default 0.0.
        """
        super().__init__(rho=rho, lr=lr, reg_param=reg_param, **params)
        self.rho = rho
        self.lr = lr
        self.reg_param = reg_param

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute dm/dt and dq/dt for online logistic regression.

        Uses Monte Carlo estimation for the Gaussian expectations.

        Args:
            t: Normalized time t = τ/d
            y: [m, q] order parameters
            params: Can override rho, lr, reg_param

        Returns:
            [dm/dt, dq/dt]
        """
        m, q = y

        rho = params.get("rho", self.rho)
        lr = params.get("lr", self.lr)
        lam = params.get("reg_param", self.reg_param)

        # Correlation coefficient
        rho_corr = m / np.sqrt(rho * q) if q > 0 and rho > 0 else 0.0
        rho_corr = np.clip(rho_corr, -0.999, 0.999)

        # Conditional variance of z given u
        var_z_given_u = q * (1 - rho_corr**2)

        # Monte Carlo estimation
        n_samples = 1000
        np.random.seed(int(t * 1000) % 2**31)

        # Sample teacher field u ~ N(0, rho)
        u_samples = np.random.randn(n_samples) * np.sqrt(rho)
        y_teacher = np.sign(u_samples)

        # Sample student field z | u ~ N(correlation, var_z_given_u)
        z_mean = rho_corr * np.sqrt(q / (rho + 1e-10)) * u_samples
        z_samples = z_mean + np.sqrt(var_z_given_u + 1e-10) * np.random.randn(n_samples)

        # Logistic gradient: g(y, z) = y * sigmoid(-y * z)
        sigmoid_arg = -y_teacher * z_samples
        g = y_teacher * self._sigmoid(sigmoid_arg)

        # Compute expectations
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

        P(error) = (1/π) arccos(m/√(qρ))

        Args:
            y: [m, q] order parameters
            **kwargs: Can override rho

        Returns:
            Classification error probability
        """
        m, q = y
        rho = kwargs.get("rho", self.rho)

        if q > 0 and rho > 0:
            cos_angle = m / np.sqrt(q * rho)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle) / np.pi
        return 0.5
