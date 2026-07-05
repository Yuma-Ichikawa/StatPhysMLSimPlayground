r"""
Reward-model overoptimization (Goodhart's law) in a teacher-student form.

Setting. The *gold* reward is a nonlinear teacher plus a constraint term
that penalizes leaving the data distribution,

.. math::
    r^\star(x) \;=\; T(x) \;-\; \lambda\,
    \mathrm{ReLU}\!\Big(\tfrac{1}{d}\lVert x\rVert^2 - \tau\Big),

so in-distribution (:math:`\lVert x\rVert^2/d \approx 1`) the penalty is
negligible, while off-distribution inputs are genuinely bad. A proxy
reward model :math:`\hat r` (student, same architecture as :math:`T`) is
trained on :math:`n_{\rm pref} = \alpha_r d` pairwise preferences with
Bradley-Terry labels,

.. math::
    P(x_1 \succ x_2) = \sigma\!\big(\beta\,[r^\star(x_1) - r^\star(x_2)]\big),

drawn *in distribution* -- the proxy never sees the off-distribution
region, which is exactly what makes it hackable.

Two optimization modes against the proxy, with an exact KL budget each:

- **best-of-n (BoN)**: draw :math:`n` base samples, keep the proxy
  argmax. :math:`\mathrm{KL}(n) = \log n - (n-1)/n` (Hilton & Gao).
  BoN stays supported on the base distribution.
- **policy gradient (shift)**: a Gaussian policy
  :math:`\pi_\mu = \mathcal N(\mu, I_d)` trained by reparameterized
  gradient ascent on :math:`\mathbb E_{\pi_\mu}[\hat r] - c\,
  \mathrm{KL}(\pi_\mu\|\pi_0)` with
  :math:`\mathrm{KL} = \lVert\mu\rVert^2/2`; sweeping the KL penalty
  :math:`c` traces out the optimization-strength axis. The policy can
  (and does) leave the data distribution.

Order parameters:

- gold reward :math:`G(\mathrm{KL})` and proxy reward
  :math:`P(\mathrm{KL})`, z-scored under the base distribution
- hacking gap :math:`P - G`
- overoptimization point :math:`\mathrm{KL}^* = \arg\max_{\rm KL} G` and
  peak gold reward :math:`G_{\max}(\alpha_r)`
- reward-model quality :math:`m_{\rm RM}` = in-distribution overlap
  between :math:`\hat r` and :math:`r^\star`

Phenomenology probed (cf. Gao, Schulman & Hilton 2023, here with a known
ground truth): under the shift policy the gold reward rises, peaks, and
*falls* -- the Goodhart turnover -- while the proxy keeps climbing; the
peak height grows with the reward-model data budget :math:`\alpha_r`.
Under BoN, which cannot leave the support of the base distribution,
there is no turnover at comparable KL: *how* you spend the KL budget
matters as much as its size.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from statphys.experiment.teacher import Teacher
from statphys.frontier.common import InputSampler, gaussian_sampler, mlp, model_overlap
from statphys.utils.seed import fix_seed

__all__ = [
    "GoldReward",
    "bon_kl",
    "run_overoptimization",
    "sweep_overoptimization",
    "train_reward_model",
]


class GoldReward:
    """
    Gold reward = nonlinear teacher minus an off-distribution penalty.

    r*(x) = T(x) - hack_penalty * relu(||x||^2 / d - tau). The penalty
    vanishes on the base distribution and encodes the part of true
    utility that in-distribution preference data cannot reveal.

    Args:
        teacher: Teacher network T (frozen).
        hack_penalty: Penalty strength lambda (0 disables hacking).
        tau: Squared-norm-per-dimension threshold.

    """

    def __init__(self, teacher: Teacher, hack_penalty: float = 2.0, tau: float = 1.0):
        self.teacher = teacher
        self.hack_penalty = hack_penalty
        self.tau = tau

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the gold reward on a batch of inputs."""
        pen = self.hack_penalty * torch.relu((x**2).mean(dim=-1) - self.tau)
        return self.teacher.clean(x) - pen


def bon_kl(n: int | np.ndarray) -> np.ndarray:
    """KL divergence of best-of-n sampling from the base policy: log n - (n-1)/n."""
    n_arr = np.asarray(n, dtype=float)
    return np.log(n_arr) - (n_arr - 1.0) / n_arr


def train_reward_model(
    reward_model: torch.nn.Module,
    gold: GoldReward,
    n_pairs: int,
    d: int,
    beta: float = 5.0,
    lr: float = 5e-3,
    epochs: int = 1500,
    seed: int = 0,
    input_sampler: InputSampler | None = None,
) -> torch.nn.Module:
    """
    Fit a proxy reward model on in-distribution Bradley-Terry pairs.

    Args:
        reward_model: Student network mapping (n, d) -> scalar rewards.
        gold: Gold reward r*.
        n_pairs: Number of preference pairs.
        d: Input dimension.
        beta: Preference sharpness (beta -> inf: noiseless comparisons).
        lr: Adam learning rate.
        epochs: Training epochs (full batch).
        seed: RNG seed for the pair data.
        input_sampler: Optional data distribution n -> (n, d).

    Returns:
        The trained reward model.

    """
    fix_seed(seed)
    sample = input_sampler if input_sampler is not None else gaussian_sampler(d)
    x1, x2 = sample(n_pairs), sample(n_pairs)
    with torch.no_grad():
        prob = torch.sigmoid(beta * (gold(x1) - gold(x2)))
        label = (torch.rand(n_pairs) < prob).float()

    opt = torch.optim.Adam(reward_model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = (reward_model(x1) - reward_model(x2)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        opt.step()
    return reward_model


def _zscore_stats(gold: GoldReward, proxy: torch.nn.Module, n_probe: int, sample: InputSampler):
    x = sample(n_probe)
    with torch.no_grad():
        g = gold(x)
        p = proxy(x).squeeze(-1)
    return (
        (float(g.mean()), float(g.std().clamp_min(1e-12))),
        (float(p.mean()), float(p.std().clamp_min(1e-12))),
    )


@torch.no_grad()
def _eval_policy(
    mu: torch.Tensor,
    gold: GoldReward,
    proxy: torch.nn.Module,
    gold_stats: tuple[float, float],
    proxy_stats: tuple[float, float],
    sample: InputSampler,
    n_eval: int = 2048,
) -> tuple[float, float]:
    xs = mu.unsqueeze(0) + sample(n_eval)
    g = (gold(xs) - gold_stats[0]) / gold_stats[1]
    p = (proxy(xs).squeeze(-1) - proxy_stats[0]) / proxy_stats[1]
    return float(g.mean()), float(p.mean())


def _optimize_shift_policy(
    proxy: torch.nn.Module,
    d: int,
    kl_coef: float,
    sample: InputSampler,
    steps: int = 1200,
    batch: int = 128,
    lr: float = 0.05,
) -> torch.Tensor:
    """Reparameterized ascent on E_pi[proxy] - kl_coef * ||mu||^2/2."""
    mu = torch.zeros(d, requires_grad=True)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x = mu.unsqueeze(0) + sample(batch)
        loss = -proxy(x).mean() + kl_coef * 0.5 * (mu**2).sum()
        loss.backward()
        opt.step()
    return mu.detach()


@torch.no_grad()
def _bon_policy_rewards(
    proxy: torch.nn.Module,
    gold: GoldReward,
    d: int,
    n: int,
    n_episodes: int,
    gold_stats: tuple[float, float],
    proxy_stats: tuple[float, float],
    sample: InputSampler,
    batch: int = 64,
) -> tuple[float, float]:
    gold_vals, proxy_vals = [], []
    for i in range(0, n_episodes, batch):
        b = min(batch, n_episodes - i)
        x = sample(b * n)
        pr = proxy(x).squeeze(-1).view(b, n)
        best = pr.argmax(dim=1)
        x_sel = x.view(b, n, d)[torch.arange(b), best]
        gold_vals.append(gold(x_sel))
        proxy_vals.append(pr[torch.arange(b), best])
    g = torch.cat(gold_vals)
    p = torch.cat(proxy_vals)
    return (
        float(((g - gold_stats[0]) / gold_stats[1]).mean().item()),
        float(((p - proxy_stats[0]) / proxy_stats[1]).mean().item()),
    )


def run_overoptimization(
    d: int = 32,
    hidden: int = 16,
    alpha_r: float = 4.0,
    beta: float = 5.0,
    hack_penalty: float = 2.0,
    policy: str = "shift",
    kl_coefs: list[float] | None = None,
    n_values: list[int] | None = None,
    policy_steps: int = 1200,
    n_eval: int = 2048,
    lr: float = 5e-3,
    epochs: int = 1500,
    n_probe: int = 4096,
    seed: int = 0,
    teacher: Teacher | None = None,
    input_sampler: InputSampler | None = None,
) -> dict:
    """
    Train a proxy reward model and sweep optimization strength against it.

    Args:
        d: Input dimension.
        hidden: Hidden width of the gold-teacher and proxy nets.
        alpha_r: Preference-pair ratio n_pairs / d.
        beta: Bradley-Terry sharpness.
        hack_penalty: Off-distribution penalty strength in the gold reward.
        policy: "shift" (KL-regularized Gaussian mean-shift, can go OOD)
            or "bon" (best-of-n, stays in-distribution).
        kl_coefs: KL penalty grid for the shift policy (large -> weak
            optimization). Defaults to a log grid from 3.0 down to 0.01.
        n_values: BoN candidate counts (policy="bon").
        policy_steps: Gradient steps per shift-policy optimization.
        n_eval: Policy evaluation samples.
        lr: Reward-model learning rate.
        epochs: Reward-model training epochs.
        n_probe: Probe samples for m_RM and reward z-scoring.
        seed: Random seed.
        teacher: Optional gold-reward base teacher T (taxonomy
            injection); defaults to a random-weight tanh MLP.
        input_sampler: Optional base data distribution n -> (n, d);
            the shift policy translates it by mu, and the KL budget
            ||mu||^2/2 is exact for the Gaussian default (an
            approximation for non-Gaussian samplers).

    Returns:
        Dict with "kl", z-scored "gold" and "proxy" curves (sorted by
        increasing KL), "kl_star" (gold argmax), "gold_max", "m_RM",
        and the config.

    """
    if kl_coefs is None:
        kl_coefs = [3.0, 1.0, 0.5, 0.3, 0.15, 0.1, 0.05, 0.03, 0.01]
    if n_values is None:
        n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    fix_seed(seed)
    if teacher is None:
        teacher = Teacher(mlp(d, hidden), init="normal")
    sample = input_sampler if input_sampler is not None else gaussian_sampler(d)
    gold = GoldReward(teacher, hack_penalty=hack_penalty)
    proxy = mlp(d, hidden)
    train_reward_model(
        proxy,
        gold,
        n_pairs=max(int(alpha_r * d), 2),
        d=d,
        beta=beta,
        lr=lr,
        epochs=epochs,
        seed=seed + 1,
        input_sampler=sample,
    )

    fix_seed(seed + 2)
    gold_stats, proxy_stats = _zscore_stats(gold, proxy, n_probe, sample)
    x_probe = sample(n_probe)
    m_rm = model_overlap(proxy, gold, x_probe)

    kl_list, gold_curve, proxy_curve = [], [], []
    if policy == "shift":
        for c in kl_coefs:
            mu = _optimize_shift_policy(proxy, d, kl_coef=c, sample=sample, steps=policy_steps)
            kl_list.append(0.5 * float((mu**2).sum()))
            g, p = _eval_policy(mu, gold, proxy, gold_stats, proxy_stats, sample, n_eval=n_eval)
            gold_curve.append(g)
            proxy_curve.append(p)
    elif policy == "bon":
        for n in n_values:
            g, p = _bon_policy_rewards(proxy, gold, d, n, n_eval, gold_stats, proxy_stats, sample)
            kl_list.append(float(bon_kl(n)))
            gold_curve.append(g)
            proxy_curve.append(p)
    else:
        raise ValueError(f"Unknown policy '{policy}' (use 'shift' or 'bon')")

    order = np.argsort(kl_list)
    kl = np.asarray(kl_list)[order]
    gold_arr = np.asarray(gold_curve)[order]
    proxy_arr = np.asarray(proxy_curve)[order]
    return {
        "kl": kl,
        "gold": gold_arr,
        "proxy": proxy_arr,
        "kl_star": float(kl[int(np.argmax(gold_arr))]),
        "gold_max": float(gold_arr.max()),
        "m_RM": m_rm,
        "config": {
            "d": d,
            "hidden": hidden,
            "alpha_r": alpha_r,
            "beta": beta,
            "hack_penalty": hack_penalty,
            "policy": policy,
            "seed": seed,
        },
    }


def sweep_overoptimization(
    alphas_r: np.ndarray | list[float],
    n_seeds: int = 3,
    verbose: bool = True,
    **run_kwargs,
) -> dict:
    """
    Sweep the reward-model data budget alpha_r.

    Args:
        alphas_r: Preference-pair ratios to sweep.
        n_seeds: Independent repetitions per alpha_r.
        verbose: Print progress.
        **run_kwargs: Forwarded to `run_overoptimization`.

    Returns:
        Dict with "alphas_r", seed-averaged "gold_curves" / "proxy_curves"
        / "kl_curves" (len(alphas) x n_kl), "kl_star" (+ std),
        "gold_max" (+ std), "m_RM".

    """
    alphas = np.asarray(list(alphas_r), dtype=float)
    gold_curves, proxy_curves, kl_curves = [], [], []
    kl_stars, kl_stars_std = [], []
    gold_maxes, gold_maxes_std, m_rms = [], [], []
    for a in alphas:
        golds, proxies, kls, stars, peaks, mrs = [], [], [], [], [], []
        for s in range(n_seeds):
            res = run_overoptimization(alpha_r=float(a), seed=s, **run_kwargs)
            golds.append(res["gold"])
            proxies.append(res["proxy"])
            kls.append(res["kl"])
            stars.append(res["kl_star"])
            peaks.append(res["gold_max"])
            mrs.append(res["m_RM"])
        gold_curves.append(np.mean(golds, axis=0))
        proxy_curves.append(np.mean(proxies, axis=0))
        kl_curves.append(np.mean(kls, axis=0))
        kl_stars.append(float(np.mean(stars)))
        kl_stars_std.append(float(np.std(stars)))
        gold_maxes.append(float(np.mean(peaks)))
        gold_maxes_std.append(float(np.std(peaks)))
        m_rms.append(float(np.mean(mrs)))
        if verbose:
            print(
                f"alpha_r={a:.1f}: KL*={kl_stars[-1]:.2f} "
                f"gold_max={gold_maxes[-1]:.2f} m_RM={m_rms[-1]:.3f}"
            )
    return {
        "alphas_r": alphas,
        "kl_curves": np.asarray(kl_curves),
        "gold_curves": np.asarray(gold_curves),
        "proxy_curves": np.asarray(proxy_curves),
        "kl_star": np.asarray(kl_stars),
        "kl_star_std": np.asarray(kl_stars_std),
        "gold_max": np.asarray(gold_maxes),
        "gold_max_std": np.asarray(gold_maxes_std),
        "m_RM": np.asarray(m_rms),
    }
