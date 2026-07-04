r"""
Online SGD dynamics of soft committee machines with exact order parameters.

The classic setting of Saad & Solla (1995) and the modern DMFT program
(Goldt et al. 2019; Mignacco et al. 2020): a student committee machine

    f(x) = (1/sqrt(K)) sum_k g(w_k . x / sqrt(d)),   g(x) = erf(x/sqrt(2))

learns online (one fresh sample per step) from a matched teacher with M
hidden units. All macroscopic observables are *exact* functions of the
order-parameter matrices

    Q_kl = w_k . w_l / d,   R_km = w_k . w*_m / d,   T_mn = w*_m . w*_n / d,

including the generalization error (arcsin formulas for erf).

The physics this module makes measurable:

- **Specialization plateau**: with (near-)symmetric initialization the
  dynamics is trapped at an unstable fixed point where every student
  unit has the *same* overlap with every teacher unit; eps_g stays at a
  plateau for a long time.
- **Symmetry breaking / escape**: fluctuations (finite d) grow
  exponentially and eventually break the permutation symmetry: each
  student unit specializes to one teacher unit and eps_g drops.
- **Escape-time scaling**: the initial asymmetry is O(1/sqrt(d)), so the
  escape time grows as t_esc ~ (ln d) / lambda where lambda is the
  instability eigenvalue -- a finite-size effect *invisible* in the
  d -> infinity ODE theory, which predicts an infinite plateau for
  perfectly symmetric initial conditions.

Order parameters designed for this transition (see
docs/order_parameters.md, "Online dynamics"):

- `specialization_gap(R)`: mean over student units of the gap between
  the largest and second-largest (absolute) teacher overlap -- zero on
  the plateau, positive after symmetry breaking.
- `escape_time(t, spec_gap)`: first time the specialization gap crosses
  an O(1) threshold (symmetry officially broken).

Example:
    >>> from statphys.experiment.online_committee import simulate_online_committee
    >>> traj = simulate_online_committee(d=128, k=2, lr=0.5, t_max=300, seed=0)
    >>> traj["t"], traj["eps_g"], traj["R"][-1]   # doctest: +SKIP

"""

from __future__ import annotations

import numpy as np

__all__ = [
    "committee_generalization_error",
    "simulate_online_committee",
    "specialization_gap",
    "escape_time",
]


def _erf_corr(c12: float, c11: float, c22: float) -> float:
    """E[g(u)g(v)] for centered Gaussians with Cov=[[c11,c12],[c12,c22]], g=erf(x/sqrt(2))."""
    denom = np.sqrt((1.0 + c11) * (1.0 + c22))
    return (2.0 / np.pi) * np.arcsin(np.clip(c12 / denom, -1.0, 1.0))


def committee_generalization_error(Q: np.ndarray, R: np.ndarray, T: np.ndarray) -> float:
    r"""
    Exact generalization error of an erf committee from order parameters.

    For f_s = (1/sqrt(K)) sum_k g(h_k), f_t = (1/sqrt(M)) sum_m g(h*_m),

        eps_g = (1/2) E[(f_s - f_t)^2]
              = (1/2) [ (1/K) sum_{kl} G(Q_kl) + (1/M) sum_{mn} G(T_mn)
                        - (2/sqrt(KM)) sum_{km} G(R_km) ]

    with G the erf-Gaussian correlation (arcsin formula). Exact at any
    finite K, M, d (Saad & Solla 1995, Eq. 5).

    Args:
        Q: Student self-overlap matrix, shape (K, K).
        R: Student-teacher overlap matrix, shape (K, M).
        T: Teacher self-overlap matrix, shape (M, M).

    Returns:
        Generalization error (non-negative scalar).

    """
    K, M = R.shape
    s_term = sum(_erf_corr(Q[k, l], Q[k, k], Q[l, l]) for k in range(K) for l in range(K))
    t_term = sum(_erf_corr(T[m, n], T[m, m], T[n, n]) for m in range(M) for n in range(M))
    cross = sum(_erf_corr(R[k, m], Q[k, k], T[m, m]) for k in range(K) for m in range(M))
    eg = 0.5 * (s_term / K + t_term / M - 2.0 * cross / np.sqrt(K * M))
    return float(max(eg, 0.0))


def specialization_gap(R: np.ndarray) -> float:
    r"""
    Symmetry-breaking order parameter for committee specialization.

    For each student unit k, sort |R_km| over teacher units and take the
    gap between the largest and second-largest; average over k:

        Delta_spec = (1/K) sum_k ( |R|_{k,(1)} - |R|_{k,(2)} )

    Zero (up to O(1/sqrt(d)) fluctuations) in the unspecialized/plateau
    phase where every student unit overlaps all teacher units equally;
    grows to O(1) once each unit commits to one teacher direction. This
    is the natural scalar "magnetization" of the permutation-symmetry
    breaking.

    Args:
        R: Student-teacher overlap matrix, shape (K, M) with M >= 2.

    Returns:
        Mean sorted-overlap gap (scalar >= 0).

    """
    a = np.sort(np.abs(np.asarray(R, dtype=float)), axis=1)
    return float(np.mean(a[:, -1] - a[:, -2]))


def escape_time(
    t: np.ndarray,
    spec_gap: np.ndarray,
    threshold: float = 0.3,
) -> float:
    r"""
    Plateau escape time from the symmetry-breaking order parameter.

    Defined as the first time the specialization gap
    :math:`\Delta_{\rm spec}(t)` exceeds `threshold`: on the
    permutation-symmetric plateau :math:`\Delta_{\rm spec} = O(1/\sqrt d)`,
    and it grows to :math:`O(1)` at specialization, so any O(1)
    threshold gives the same scaling of the escape time.

    Args:
        t: Time grid, shape (T,).
        spec_gap: Specialization-gap trajectory Delta_spec(t), shape (T,).
        threshold: O(1) crossing level declaring escape.

    Returns:
        Escape time, or NaN if the trajectory never escapes.

    """
    t = np.asarray(t, dtype=float)
    gap = np.asarray(spec_gap, dtype=float)
    above = np.where(gap > threshold)[0]
    return float(t[above[0]]) if above.size else float("nan")


def simulate_online_committee(
    d: int = 128,
    k: int = 2,
    m: int | None = None,
    lr: float = 0.5,
    t_max: float = 300.0,
    n_snapshots: int = 300,
    init_scale: float = 1e-3,
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    r"""
    Online SGD for an erf soft committee machine, with exact observables.

    One fresh Gaussian sample per step (online/one-pass limit), teacher
    with orthonormal hidden directions (T = I). Records order-parameter
    snapshots and the exact eps_g at ~n_snapshots log-friendly times.

    Update rule (MSE loss, per-sample gradient step):

        w_k += (lr / sqrt(K)) * (y - f(x)) * g'(h_k) * x / sqrt(d)

    Time is measured as t = (number of samples) / d, matching the
    thermodynamic-limit ODE convention.

    Args:
        d: Input dimension.
        k: Number of student hidden units K.
        m: Number of teacher hidden units M (default: K, matched).
        lr: Learning rate eta.
        t_max: Total normalized time (samples / d).
        n_snapshots: Number of recorded measurement points.
        init_scale: Student init scale; small values give a long
            symmetric plateau (near-unstable-fixed-point start).
        noise_std: Additive label noise std.
        seed: RNG seed (controls teacher, init, and the sample stream).

    Returns:
        dict with keys:
        - "t": (S,) snapshot times
        - "eps_g": (S,) exact generalization error
        - "spec_gap": (S,) specialization gap Delta_spec
        - "R": (S, K, M) student-teacher overlap snapshots
        - "Q": (S, K, K) student self-overlap snapshots
        - "escape_time": scalar (NaN if no escape within t_max)
        - "params": run configuration

    """
    m = k if m is None else m
    rng = np.random.default_rng(seed)

    # Teacher with orthonormal rows scaled to |w*_m|^2 = d (T = I)
    a = rng.standard_normal((d, m))
    q_mat, _ = np.linalg.qr(a)
    w_teacher = q_mat.T * np.sqrt(d)  # (m, d)

    w = init_scale * rng.standard_normal((k, d))

    sqrt_d = np.sqrt(d)
    n_steps = int(t_max * d)
    snap_every = max(1, n_steps // n_snapshots)

    def g(h: np.ndarray) -> np.ndarray:
        from scipy.special import erf

        return erf(h / np.sqrt(2.0))

    def g_prime(h: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / np.pi) * np.exp(-0.5 * h**2)

    T = w_teacher @ w_teacher.T / d
    ts, egs, gaps, r_snaps, q_snaps = [], [], [], [], []

    def measure(step: int) -> None:
        R = w @ w_teacher.T / d
        Q = w @ w.T / d
        ts.append(step / d)
        egs.append(committee_generalization_error(Q, R, T))
        gaps.append(specialization_gap(R))
        r_snaps.append(R.copy())
        q_snaps.append(Q.copy())

    for step in range(n_steps):
        if step % snap_every == 0:
            measure(step)
        x = rng.standard_normal(d)
        h_s = w @ x / sqrt_d
        h_t = w_teacher @ x / sqrt_d
        y = g(h_t).sum() / np.sqrt(m)
        if noise_std > 0:
            y += noise_std * rng.standard_normal()
        delta = y - g(h_s).sum() / np.sqrt(k)
        w += (lr / np.sqrt(k)) * delta * np.outer(g_prime(h_s), x) / sqrt_d
    measure(n_steps)

    t_arr = np.asarray(ts)
    eg_arr = np.asarray(egs)
    gap_arr = np.asarray(gaps)
    return {
        "t": t_arr,
        "eps_g": eg_arr,
        "spec_gap": gap_arr,
        "R": np.asarray(r_snaps),
        "Q": np.asarray(q_snaps),
        "escape_time": escape_time(t_arr, gap_arr),
        "params": {
            "d": d,
            "k": k,
            "m": m,
            "lr": lr,
            "t_max": t_max,
            "init_scale": init_scale,
            "noise_std": noise_std,
            "seed": seed,
        },
    }
