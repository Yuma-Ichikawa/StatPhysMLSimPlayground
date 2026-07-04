"""
Shared machinery for heuristic gradient-flow replica scenarios.

Several classification scenarios (logistic, probit, hinge) do not have
simple closed-form RS saddle-point equations in this package; instead they
relax the regularized ERM stationarity conditions by a damped gradient
flow on the order parameters:

    m <- m + damping * (alpha * sqrt(rho) * E[g * u/sqrt(rho)] - lam * m)
    q <- q + damping * (alpha * E[g^2] - lam * q)

where g is the loss gradient evaluated on the joint Gaussian
teacher/student fields (u, z) ~ N(0, [[rho, m], [m, q]]).

This module centralizes:
    - the joint-field 2D Gaussian expectation (with correlation clipping)
    - the damped relaxation step and physical constraints

so that each scenario only specifies its per-field gradient moments.

Warning:
    These relaxations are qualitatively correct but not quantitatively
    exact; see the individual scenario docstrings.

"""

from typing import Any

import numpy as np
from scipy.integrate import dblquad

from statphys.theory.replica.scenario.base import ReplicaEquations
from statphys.utils.constants import (
    CORR_CLIP,
    DEFAULT_GRADFLOW_DAMPING,
    EPS_DIV,
    EPS_ORDER_PARAM,
    GAUSS_INT_BOUND,
)


class GradientFlowEquations(ReplicaEquations):
    """
    Base class for gradient-flow relaxation scenarios.

    Subclasses implement `_gradient_moments(u, z)` returning the pair
    (E_y[g | u, z], E_y[g^2 | u, z]) where the expectation is over the
    label distribution given the teacher field u.

    Args:
        rho: Teacher norm.
        reg_param: L2 regularization lambda.
        damping: Step size of the gradient-flow relaxation.
        int_bound: Half-width of the 2D Gaussian integration domain
            (in units of std).
        **params: Extra parameters stored on the instance.

    """

    def __init__(
        self,
        rho: float = 1.0,
        reg_param: float = 0.01,
        damping: float = DEFAULT_GRADFLOW_DAMPING,
        int_bound: float = GAUSS_INT_BOUND,
        **params: Any,
    ):
        super().__init__(rho=rho, reg_param=reg_param, **params)
        self.rho = rho
        self.reg_param = reg_param
        self.damping = damping
        self.int_bound = int_bound

    def _gradient_moments(self, u: float, z: float) -> tuple[float, float]:
        """
        Return (E_y[g | u, z], E_y[g^2 | u, z]) for teacher field u and
        student field z, averaged over the label distribution.
        """
        raise NotImplementedError

    def _joint_expectations(self, m: float, q: float, rho: float) -> tuple[float, float]:
        """
        Compute E[g * u/sqrt(rho)] and E[g^2] over the joint field Gaussian.

        The fields are parameterized by independent standard normals:
            u = sqrt(rho) * xi1
            z = sqrt(q) * (corr * xi1 + sqrt(1 - corr^2) * xi2)
        with corr = m / sqrt(q * rho), clipped inside (-1, 1).

        Returns:
            (E[g * xi1], E[g^2]).

        """
        q = max(q, EPS_ORDER_PARAM)
        rho = max(rho, EPS_ORDER_PARAM)
        max_m = np.sqrt(q * rho) * CORR_CLIP
        m = np.clip(m, -max_m, max_m)

        corr = m / np.sqrt(q * rho)
        std_perp = np.sqrt(max(1 - corr**2, EPS_DIV))
        b = self.int_bound

        def integrand_gu(xi2: float, xi1: float) -> float:
            u = np.sqrt(rho) * xi1
            z = np.sqrt(q) * (corr * xi1 + std_perp * xi2)
            e_g, _ = self._gradient_moments(u, z)
            gauss = np.exp(-0.5 * (xi1**2 + xi2**2)) / (2 * np.pi)
            return e_g * xi1 * gauss

        def integrand_g2(xi2: float, xi1: float) -> float:
            u = np.sqrt(rho) * xi1
            z = np.sqrt(q) * (corr * xi1 + std_perp * xi2)
            _, e_g2 = self._gradient_moments(u, z)
            gauss = np.exp(-0.5 * (xi1**2 + xi2**2)) / (2 * np.pi)
            return e_g2 * gauss

        E_gu, _ = dblquad(integrand_gu, -b, b, -b, b)
        E_g2, _ = dblquad(integrand_g2, -b, b, -b, b)
        return E_gu, E_g2

    def _relax(
        self,
        m: float,
        q: float,
        dm: float,
        dq: float,
    ) -> tuple[float, float]:
        """
        Apply one damped relaxation step and enforce physical constraints.

        Args:
            m, q: Current order parameters.
            dm, dq: Flow directions (already including -lam * m / -lam * q).

        Returns:
            Updated (m, q), floored at EPS_ORDER_PARAM.

        """
        new_m = m + self.damping * dm
        new_q = q + self.damping * dq
        return max(new_m, EPS_ORDER_PARAM), max(new_q, EPS_ORDER_PARAM)

    def __call__(
        self,
        m: float,
        q: float,
        alpha: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        One fixed-point iteration: joint expectations + damped relaxation.

        Args:
            m: Current teacher-student overlap.
            q: Current self-overlap.
            alpha: Sample ratio n/d.
            **kwargs: Can override rho, reg_param.

        Returns:
            Updated (m, q).

        """
        rho = kwargs.get("rho", self.rho)
        lam = kwargs.get("reg_param", self.reg_param)
        q = max(q, EPS_ORDER_PARAM)

        E_gu, E_g2 = self._joint_expectations(m, q, rho)
        scale = alpha / (1 + lam)
        dm = scale * np.sqrt(rho) * E_gu - lam * m
        dq = scale * E_g2 - lam * q
        return self._relax(m, q, dm, dq)
