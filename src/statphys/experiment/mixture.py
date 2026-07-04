"""
Gaussian-mixture classification: a generative (not discriminative)
teacher-student setting.

Standard high-dimensional statistics benchmark for classification
(Mignacco, Krzakala, Mezard, Urbani, Zdeborova 2020, "The role of
regularization in classification of high-dimensional noisy Gaussian
mixture"; Deng, Kammoun, Thrampoulidis 2019; Mai & Liao 2019):

    y ~ Uniform({-1, +1}),   x = y * mu * v + z,   z ~ N(0, I_d)

with a fixed unit "cluster axis" v in R^d. Unlike the rest of the
`experiment` package, the label y here determines x (not the other way
around) -- this is the natural setting for studying classification of
*structured, clustered* data rather than a function of an unstructured
input, and complements the function-approximation teacher-student
picture used elsewhere.

Because the model is exactly solvable, it gives a clean way to verify
that the generalization-error bookkeeping in this package is correct:
for a linear classifier x -> sign(w . x), the decision variable
z = (w . x) / ||w|| is Gaussian with mean y * mu * cos(w, v) and unit
variance (since ||v|| = 1), so the exact test error is

    eps_g(w) = Phi(-mu * cos_angle(w, v))

where Phi is the standard normal CDF. In particular the Bayes-optimal
classifier w = v attains eps_g = Phi(-mu). `bayes_error` computes this
closed form so it can be checked against the numerically measured
0-1 test error (see tests/test_mixture.py).
"""

from typing import Any

import torch
from scipy.stats import norm

__all__ = ["GaussianMixtureDataset", "bayes_error"]


def bayes_error(mu: float, cos_angle: float = 1.0) -> float:
    """
    Exact 0-1 test error of a linear classifier on the mixture model.

    Args:
        mu: Cluster separation ("signal-to-noise ratio").
        cos_angle: Cosine similarity between the classifier direction
            and the true cluster axis v (1.0 = Bayes-optimal).

    Returns:
        eps_g = Phi(-mu * cos_angle).

    """
    return float(norm.cdf(-mu * cos_angle))


class GaussianMixtureDataset:
    """
    Two-cluster Gaussian-mixture classification data.

    x = y * mu * v + z, y in {-1, +1} uniform, z ~ N(0, I_d), v a fixed
    random unit vector (the "cluster axis" / planted signal).

    Compatible with `TeacherStudentExperiment(dataset=...)`: implements
    `.sample(n)`, `.sample_inputs(n)`, and `.get_config()`. Pair with
    `GaussianMixtureDataset.oracle_teacher()` for a Teacher whose
    `.clean(x) = sign(v . x)` matches the Bayes-consistent decision rule,
    so `function_order_params` / `m_hat` behave as in the rest of the
    package.

    Args:
        d: Input dimension.
        mu: Cluster separation (Bayes error = Phi(-mu)).
        v: Optional fixed (d,) unit direction; a random one is drawn
            if omitted.
        device: Torch device.

    """

    def __init__(
        self,
        d: int,
        mu: float = 1.5,
        v: torch.Tensor | None = None,
        device: str = "cpu",
    ):
        self.d = d
        self.mu = mu
        self.device = torch.device(device)
        if v is None:
            v = torch.randn(d)
        v = v.to(self.device).float()
        self.v = v / v.norm().clamp_min(1e-12)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate n labelled samples (X, y) with y in {-1, +1}."""
        y = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            -torch.ones(1, device=self.device),
            torch.ones(1, device=self.device),
        )
        z = torch.randn(n, self.d, device=self.device)
        X = y.unsqueeze(1) * self.mu * self.v.unsqueeze(0) + z
        return X, y

    def sample_inputs(self, n: int) -> torch.Tensor:
        """Generate n unlabelled inputs (marginal mixture distribution)."""
        return self.sample(n)[0]

    def oracle_teacher(self):
        """Teacher whose clean(x) = sign(v . x), the Bayes-consistent rule."""
        import torch.nn as nn

        from statphys.experiment.teacher import Teacher

        linear = nn.Linear(self.d, 1, bias=False)
        with torch.no_grad():
            linear.weight.copy_(self.v.unsqueeze(0))
        return Teacher(linear, readout="sign", device=str(self.device))

    def get_config(self) -> dict[str, Any]:
        """Return a summary config for logging."""
        return {"d": self.d, "mu": self.mu, "type": "gaussian_mixture"}

    def __repr__(self) -> str:
        return f"GaussianMixtureDataset(d={self.d}, mu={self.mu})"
