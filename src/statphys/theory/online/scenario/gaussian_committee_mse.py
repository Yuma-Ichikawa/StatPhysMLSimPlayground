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

    The ODE system uses the exact Saad-Solla closed forms for erf
    activation (single-unit reduction, exact for K = M = 1):

        dr/dt = η ( I3(λ; ν; ν) - I3(λ; ν; λ) )
        dq/dt = 2η ( I3(λ; λ; ν) - I3(λ; λ; λ) )
              + η² ( I4(ν,ν) - 2 I4(ν,λ) + I4(λ,λ) )

    where λ (variance q) is the student field, ν (variance T) the teacher
    field with covariance r, and I3 = E[g'(u) v g(w)],
    I4 = E[g'(λ)² g(w) g(z)] are Gaussian correlation integrals with
    closed forms for g(x) = erf(x/√2) (Saad & Solla 1995, Appendix).

    For K = M > 1 the same per-unit equations are used as a symmetric
    ansatz approximation with negligible cross-overlaps (C = S = 0);
    they are then only approximate.
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

    @staticmethod
    def _i3(c11: float, c12: float, c13: float, c23: float, c33: float) -> float:
        """
        Saad-Solla I3 = E[g'(x1) x2 g(x3)] for g(x) = erf(x/√2).

        Closed form (Saad & Solla 1995, Appendix):
            Λ3 = (1+c11)(1+c33) - c13²
            I3 = (2/π) [c23(1+c11) - c12 c13] / ((1+c11) √Λ3)
        """
        lam3 = (1 + c11) * (1 + c33) - c13**2
        if lam3 <= 0:
            return 0.0
        return (2 / np.pi) * (c23 * (1 + c11) - c12 * c13) / ((1 + c11) * np.sqrt(lam3))

    @staticmethod
    def _i4_same_deriv(
        Q: float,
        c13: float,
        c14: float,
        c33: float,
        c34: float,
        c44: float,
    ) -> float:
        """
        Saad-Solla I4 = E[g'(x1) g'(x2) g(x3) g(x4)] with x1 = x2 (both the
        student field, variance Q), for g(x) = erf(x/√2).

        With c11 = c22 = c12 = Q the general closed form reduces to:
            Λ4 = 1 + 2Q
            Λ0 = Λ4 c34 - 2(1+Q)(c13 c14) + 2Q c13 c14   [x1=x2 → c23=c13 etc.]
            Λ1 = Λ4 (1+c33) - 2 c13² (1+Q) + 2Q c13²
            Λ2 = Λ4 (1+c44) - 2 c14² (1+Q) + 2Q c14²
            I4 = (4/π²) arcsin(Λ0/√(Λ1 Λ2)) / √Λ4
        """
        lam4 = 1 + 2 * Q
        lam0 = lam4 * c34 - 2 * c13 * c14 * (1 + Q) + 2 * Q * c13 * c14
        lam1 = lam4 * (1 + c33) - 2 * c13**2 * (1 + Q) + 2 * Q * c13**2
        lam2 = lam4 * (1 + c44) - 2 * c14**2 * (1 + Q) + 2 * Q * c14**2
        denom = np.sqrt(max(lam1 * lam2, 1e-30))
        arg = np.clip(lam0 / denom, -1, 1)
        return (4 / np.pi**2) * np.arcsin(arg) / np.sqrt(lam4)

    def __call__(
        self,
        t: float,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute ODE dynamics (Saad-Solla, erf activation).

        Tracks the 2-parameter reduction:
            y = [r, q] where:
            - r: student-teacher overlap R
            - q: student self-overlap Q

        Exact for K = M = 1; for K = M > 1 this is the symmetric ansatz
        with negligible cross-overlaps.

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

        # Ensure numerical stability (Cauchy-Schwarz: r² ≤ qT; the closed
        # forms remain finite at the boundary, so no shrink factor is needed)
        q = max(q, 1e-6)
        r = np.clip(r, -np.sqrt(q * T), np.sqrt(q * T))

        # Drift terms: E[g'(λ) x δ] with δ = g(ν) - g(λ)
        # dr/dt = η ( I3(λ; ν; ν) - I3(λ; ν; λ) )
        i3_r_teacher = self._i3(q, r, r, T, T)
        i3_r_student = self._i3(q, r, q, r, q)
        dr_dt = lr * (i3_r_teacher - i3_r_student)

        # dq/dt drift: 2η ( I3(λ; λ; ν) - I3(λ; λ; λ) )
        i3_q_teacher = self._i3(q, q, r, r, T)
        i3_q_student = self._i3(q, q, q, q, q)

        # dq/dt noise: η² E[g'(λ)² δ²]
        #            = η² ( I4(ν,ν) - 2 I4(ν,λ) + I4(λ,λ) )
        i4_tt = self._i4_same_deriv(q, r, r, T, T, T)
        i4_ts = self._i4_same_deriv(q, r, q, T, r, q)
        i4_ss = self._i4_same_deriv(q, q, q, q, q, q)

        dq_dt = 2 * lr * (i3_q_teacher - i3_q_student) + lr**2 * (i4_tt - 2 * i4_ts + i4_ss)

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
