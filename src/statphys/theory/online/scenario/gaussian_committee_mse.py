"""
Scenario: Gaussian Data × Committee Machine × MSE Loss (Online)

ODE equations for online learning of soft committee machine.

Data: x ~ N(0, I_d), y = (1/√M) Σ_m φ(v*_m^T x / √d)
Model: Committee machine f(x) = (1/√K) Σ_k φ(v_k^T x / √d)
Loss: MSE (Mean Squared Error)

References:
    - Saad, Solla (1995). "On-line learning in soft committee machines."
      Phys. Rev. E 52, 4225
    - Biehl, Schwarze (1995). "Learning by on-line gradient descent."
      J. Phys. A 28, 5033
    - Goldt et al. (2020). Phys. Rev. X 10, 041044

"""

from typing import Any

import numpy as np

from statphys.theory.online.scenario.base import OnlineEquations


class GaussianCommitteeMseEquations(OnlineEquations):
    """
    ODE equations for online learning of soft committee machine.

    Committee machine architecture:
        Student: f(x) = (1/√K) Σ_k φ(v_k^T x / √d)
        Teacher: y = (1/√M) Σ_m φ(v*_m^T x / √d)

    where φ is the activation function (erf recommended for closed-form).

    Order parameters (symmetric ansatz with K=M):
        Q = (1/d) v_k^T v_k : Student self-overlap (diagonal)
        C = (1/d) v_k^T v_l : Student cross-overlap (off-diagonal, k≠l)
        R = (1/d) v_k^T v*_k : Student-teacher overlap (matched units)
        S = (1/d) v_k^T v*_l : Student-teacher cross-overlap (k≠l)
        T = (1/d) v*_m^T v*_m : Teacher self-overlap (given)

    For the simplified symmetric ansatz, we track:
        - r: Average student-teacher overlap
        - q: Average student self-overlap
        - c: Average student cross-overlap (can be set to 0 initially)

    The ODE system (Saad & Solla 1995, erf activation):
        dr/dt = η [(T-r)/√(1+Q) - (r-S)r/(1+Q)^(3/2)]
        dQ/dt = 2η [(r-S)/√(1+Q) - (Q-C)/(1+Q)^(3/2)] + 2η²/(π(1+Q))
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
        Initialize GaussianCommitteeMseEquations.

        Args:
            k_student: Number of student hidden units K. Default 2.
            k_teacher: Number of teacher hidden units M. Default 2.
            rho: Teacher norm per unit (T_mm = rho). Default 1.0.
            lr: Learning rate η. Default 0.1.
            activation: Activation function ('erf' only for now). Default 'erf'.

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
        self.rho = rho  # T (teacher self-overlap)
        self.lr = lr
        self.activation = activation

        if activation != "erf":
            raise NotImplementedError(
                f"Activation '{activation}' not implemented. Use 'erf' for closed-form equations."
            )

    def _I2(self, a: float, b: float) -> float:
        """
        Correlation integral I_2 for erf activation.

        I_2(a, b) = (2/π) arcsin(ab / √((1+a²)(1+b²)))

        This is E[g(u)g(v)] where g = erf(x/√2) and Cov(u,v) = ab.
        """
        denom = np.sqrt((1 + a**2) * (1 + b**2))
        if denom < 1e-10:
            return 0.0
        arg = a * b / denom
        arg = np.clip(arg, -1, 1)
        return (2 / np.pi) * np.arcsin(arg)

    def _I3(self, a: float, b: float, c: float) -> float:
        """
        Correlation integral I_3 for erf activation (derivative term).

        I_3(a,b,c) = E[g'(u)g(v)g(w)] for correlated Gaussians.

        For symmetric case where a=b=c=Q (diagonal self-overlap):
            I_3 ≈ (2/π) / √(1 + Q)
        """
        # Simplified symmetric approximation
        Q_avg = (a + b + c) / 3
        return (2 / np.pi) / np.sqrt(1 + Q_avg + 1e-10)

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute ODE dynamics for committee machine (symmetric ansatz).

        This implements a simplified 2-parameter model:
            y = [r, q] where:
            - r: student-teacher overlap (average R_kk)
            - q: student self-overlap (average Q_kk)

        The symmetric ansatz assumes:
            - All K student units are equivalent
            - K = M (matched architecture)
            - R_kl = s for k ≠ l (cross-overlap, assumed small)
            - Q_kl = c for k ≠ l (student cross-overlap, assumed 0)

        Args:
            t: Normalized time t = τ/d
            y: [r, q] order parameters
            params: Can override rho, lr

        Returns:
            [dr/dt, dq/dt]

        """
        r, q = y

        T = params.get("rho", self.rho)  # Teacher self-overlap
        lr = params.get("lr", self.lr)
        params.get("k_student", self.k_student)

        # Ensure numerical stability
        q = max(q, 1e-6)
        r = np.clip(r, -np.sqrt(q * T) * 0.999, np.sqrt(q * T) * 0.999)

        # Symmetric ansatz: c = 0 (no student cross-correlation initially)
        c = 0.0
        # s = 0 (no cross student-teacher overlap)
        s = 0.0

        # Denominators
        sqrt_1_q = np.sqrt(1 + q)

        # ODE for r (student-teacher overlap)
        # dr/dt = η * [(T - r) / √(1+q) - (r - s) * r / (1+q)^(3/2)]
        # Simplified for symmetric ansatz:
        dr_dt = lr * ((T - r) / sqrt_1_q - (r - s) * r / (sqrt_1_q**3))

        # ODE for q (student self-overlap)
        # dq/dt = 2η * [(r - s) / √(1+q) - (q - c) / (1+q)^(3/2)] + noise term
        # The noise term from SGD: η² * E[gradient²]
        # For erf: noise ≈ 2η² / (π(1+q))
        gradient_noise = 2 * lr**2 / (np.pi * (1 + q))
        dq_dt = 2 * lr * ((r - s) / sqrt_1_q - (q - c) / (sqrt_1_q**3)) + gradient_noise

        return np.array([dr_dt, dq_dt])

    def generalization_error(
        self,
        y: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Compute generalization error for committee machine.

        For erf activation with symmetric ansatz (K=M, matched):
            E_g = (M/π)[arcsin(T/(1+T)) + (K-1)arcsin(C/(1+Q))]
                - (2/π) Σ arcsin(R/√((1+T)(1+Q)))
                + (K/π)[arcsin(Q/(1+Q)) + (K-1)arcsin(C/(1+Q))]

        For simplified 2-param model (r, q):
            E_g ≈ (1/π)[arcsin(T/(1+T)) - 2*arcsin(r/√((1+T)(1+q))) + arcsin(q/(1+q))]

        Args:
            y: [r, q] order parameters
            **kwargs: Can override rho (T)

        Returns:
            Generalization error

        """
        r, q = y
        T = kwargs.get("rho", self.rho)
        kwargs.get("k_student", self.k_student)

        q = max(q, 1e-6)
        T = max(T, 1e-6)

        # Teacher contribution
        teacher_term = np.arcsin(T / (1 + T))

        # Student contribution
        student_term = np.arcsin(q / (1 + q))

        # Cross term (student-teacher)
        cross_denom = np.sqrt((1 + T) * (1 + q))
        cross_arg = np.clip(r / cross_denom, -1, 1)
        cross_term = np.arcsin(cross_arg)

        # E_g = (1/π) * [teacher + student - 2*cross]
        eg = (1 / np.pi) * (teacher_term + student_term - 2 * cross_term)

        return max(eg, 0.0)

    def get_order_param_names(self) -> list[str]:
        """Return names of order parameters for symmetric ansatz."""
        return ["r", "q"]

    def get_default_init(self) -> tuple[float, ...]:
        """Return default initial values."""
        # Small initial overlap, moderate self-overlap
        return (0.01, 0.5)

    def steady_state_estimate(self, **kwargs: Any) -> tuple[float, float]:
        """
        Estimate steady state (r*, q*).

        At perfect learning: r* → T, q* → T (for K=M, matched).

        Args:
            **kwargs: Can override rho

        Returns:
            (r*, q*) estimated steady state

        """
        T = kwargs.get("rho", self.rho)
        # For matched architecture, optimal is r = q = T
        return T, T
