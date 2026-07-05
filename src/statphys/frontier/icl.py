r"""
Emergence of in-context learning under a task-diversity dial.

Setting (Garg et al. 2022; Raventos et al. 2023, reduced). A small
causal transformer is pretrained on sequences

.. math::
    (x_1, y_1, \dots, x_k, y_k, x_{\rm query}) \mapsto y_{\rm query},
    \qquad y = w^\top x / \sqrt d,

where each sequence's task vector :math:`w` is drawn from a *finite
pretraining pool* of :math:`N_{\rm tasks}` fixed teachers. After
pretraining, the model is evaluated on (i) tasks from the pool (seen)
and (ii) fresh tasks :math:`w \sim \mathcal N(0, I_d)` (unseen). Solving
unseen tasks requires genuine *in-context learning* -- regression on the
prompt at inference time -- rather than retrieving a memorized task.

Order parameters:

- ICL score :math:`S_{\rm ICL} = 1 - \varepsilon_{\rm unseen} /
  \varepsilon_{\rm null}` with :math:`\varepsilon_{\rm null} =
  \mathbb E[y^2]` (0 = no in-context learning, 1 = perfect)
- memorization score, same with seen (pool) tasks
- ridge alignment: overlap between the transformer's unseen-task
  predictions and the Bayes-optimal ridge predictor computed from the
  same prompt -- distinguishes the "discrete memorization" solution
  from the "in-context regression" solution

Phenomenology probed. A sharp *algorithmic transition* as
:math:`N_{\rm tasks}` grows: below a task-diversity threshold the
network memorizes the pool (seen error small, unseen error at chance);
above it a qualitatively different in-context-regression solution takes
over and unseen error collapses onto the ridge predictor. This is an
emergence transition in the space of algorithms, far outside current
exact theory.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from statphys.utils.seed import fix_seed

__all__ = ["ICLTransformer", "make_icl_batch", "ridge_predictor", "run_icl", "sweep_icl"]


class ICLTransformer(nn.Module):
    """
    Minimal causal transformer for in-context regression.

    Tokens are (d+1)-dimensional: [x_i ; y_i] for context pairs and
    [x_query ; 0] for the query. The prediction is read out from the
    final position.

    Args:
        d: Input dimension of x.
        d_model: Transformer width.
        n_layers: Number of encoder layers.
        n_heads: Attention heads.
        max_len: Maximum sequence length (2k + 1 tokens are not used;
            we pack pairs, so length = k + 1).

    """

    def __init__(
        self,
        d: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        max_len: int = 64,
    ):
        super().__init__()
        self.embed = nn.Linear(d + 1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            batch_first=True,
            dropout=0.0,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.readout = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map (batch, seq, d+1) token sequences to scalar predictions."""
        seq = tokens.shape[1]
        h = self.embed(tokens) + self.pos[:, :seq]
        mask = nn.Transformer.generate_square_subsequent_mask(seq, device=tokens.device)
        h = self.encoder(h, mask=mask)
        return self.readout(h[:, -1]).squeeze(-1)


def make_icl_batch(
    task_pool: torch.Tensor | None,
    batch: int,
    d: int,
    k: int,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch of in-context linear-regression sequences.

    Args:
        task_pool: (N_tasks, d) fixed pool, or None to draw fresh
            Gaussian tasks per sequence (the "unseen" distribution).
        batch: Number of sequences.
        d: Input dimension.
        k: Context length (number of (x, y) pairs before the query).
        noise_std: Label noise on context and query labels.
        generator: Optional RNG.

    Returns:
        (tokens, y_query, W): tokens of shape (batch, k+1, d+1), query
        targets (batch,), and per-sequence task vectors (batch, d).

    """
    g = generator
    if task_pool is None:
        W = torch.randn(batch, d, generator=g)
    else:
        idx = torch.randint(0, task_pool.shape[0], (batch,), generator=g)
        W = task_pool[idx]
    X = torch.randn(batch, k + 1, d, generator=g)
    y = torch.einsum("bkd,bd->bk", X, W) / d**0.5
    if noise_std > 0:
        y = y + noise_std * torch.randn(y.shape, generator=g)
    y_in = y.clone()
    y_in[:, -1] = 0.0  # query label is hidden
    tokens = torch.cat([X, y_in.unsqueeze(-1)], dim=-1)
    return tokens, y[:, -1], W


@torch.no_grad()
def ridge_predictor(tokens: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """
    Bayes-optimal (ridge) in-context predictor from the same prompt.

    Solves w_hat = argmin ||X_ctx w / sqrt(d) - y_ctx||^2 + ridge ||w||^2
    per sequence and predicts w_hat . x_query / sqrt(d).

    Args:
        tokens: (batch, k+1, d+1) token array as built by make_icl_batch.
        ridge: Ridge regularization.

    Returns:
        Predictions of shape (batch,).

    """
    d = tokens.shape[-1] - 1
    X_ctx, y_ctx = tokens[:, :-1, :d], tokens[:, :-1, d]
    x_q = tokens[:, -1, :d]
    A = X_ctx.transpose(1, 2) @ X_ctx / d + ridge * torch.eye(d).unsqueeze(0)
    b = X_ctx.transpose(1, 2) @ y_ctx.unsqueeze(-1) / d**0.5
    w_hat = torch.linalg.solve(A, b).squeeze(-1)
    return (x_q * w_hat).sum(-1) / d**0.5


def run_icl(
    n_tasks: int,
    d: int = 8,
    k: int = 16,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    steps: int = 3000,
    batch: int = 64,
    lr: float = 1e-3,
    noise_std: float = 0.0,
    n_eval: int = 2048,
    seed: int = 0,
) -> dict:
    """
    Pretrain a transformer on a finite task pool and measure ICL.

    Args:
        n_tasks: Pool size N_tasks (the diversity dial).
        d: Input dimension.
        k: Context pairs per sequence.
        d_model: Transformer width.
        n_layers: Transformer depth.
        n_heads: Attention heads.
        steps: Pretraining steps.
        batch: Sequences per step.
        lr: Adam learning rate.
        noise_std: Label noise during pretraining.
        n_eval: Evaluation sequences per split.
        seed: Random seed.

    Returns:
        Dict with "icl_score", "memo_score", "eps_unseen", "eps_seen",
        "eps_null", "ridge_alignment", "eps_ridge", and the config.

    """
    fix_seed(seed)
    pool = torch.randn(n_tasks, d)
    model = ICLTransformer(d, d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=k + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        tokens, y_q, _ = make_icl_batch(pool, batch, d, k, noise_std=noise_std)
        opt.zero_grad()
        loss = ((model(tokens) - y_q) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    gen = torch.Generator().manual_seed(seed + 1)
    tok_seen, y_seen, _ = make_icl_batch(pool, n_eval, d, k, generator=gen)
    tok_new, y_new, _ = make_icl_batch(None, n_eval, d, k, generator=gen)
    with torch.no_grad():
        pred_seen = model(tok_seen)
        pred_new = model(tok_new)
    pred_ridge = ridge_predictor(tok_new)

    eps_seen = float(((pred_seen - y_seen) ** 2).mean())
    eps_new = float(((pred_new - y_new) ** 2).mean())
    eps_ridge = float(((pred_ridge - y_new) ** 2).mean())
    eps_null = float((y_new**2).mean())
    align = float(
        (
            ((pred_new - pred_new.mean()) * (pred_ridge - pred_ridge.mean())).mean()
            / (pred_new.std() * pred_ridge.std()).clamp_min(1e-12)
        ).item()
    )
    return {
        "icl_score": 1.0 - eps_new / eps_null,
        "memo_score": 1.0 - eps_seen / eps_null,
        "eps_unseen": eps_new,
        "eps_seen": eps_seen,
        "eps_ridge": eps_ridge,
        "eps_null": eps_null,
        "ridge_alignment": align,
        "config": {
            "n_tasks": n_tasks,
            "d": d,
            "k": k,
            "d_model": d_model,
            "n_layers": n_layers,
            "steps": steps,
            "noise_std": noise_std,
            "seed": seed,
        },
    }


def sweep_icl(
    n_tasks_values: list[int],
    n_seeds: int = 2,
    verbose: bool = True,
    **run_kwargs,
) -> dict:
    """
    Sweep the task-diversity dial N_tasks.

    Args:
        n_tasks_values: Pool sizes to sweep (e.g. powers of 2).
        n_seeds: Repetitions per pool size.
        verbose: Print progress.
        **run_kwargs: Forwarded to `run_icl`.

    Returns:
        Dict with "n_tasks", seed-averaged "icl_score", "memo_score",
        "ridge_alignment", "eps_unseen", "eps_ridge" (+ "_std" arrays).

    """
    keys = ("icl_score", "memo_score", "ridge_alignment", "eps_unseen", "eps_ridge")
    out: dict[str, list[float]] = {k: [] for k in keys}
    out_std: dict[str, list[float]] = {k: [] for k in keys}
    for n_tasks in n_tasks_values:
        vals: dict[str, list[float]] = {k: [] for k in keys}
        for s in range(n_seeds):
            res = run_icl(n_tasks=n_tasks, seed=s, **run_kwargs)
            for k in keys:
                vals[k].append(res[k])
        for k in keys:
            out[k].append(float(np.mean(vals[k])))
            out_std[k].append(float(np.std(vals[k])))
        if verbose:
            print(
                f"N_tasks={n_tasks}: ICL={out['icl_score'][-1]:.3f} "
                f"memo={out['memo_score'][-1]:.3f} align={out['ridge_alignment'][-1]:.3f}"
            )
    result = {"n_tasks": np.asarray(n_tasks_values)}
    result.update({k: np.asarray(v) for k, v in out.items()})
    result.update({f"{k}_std": np.asarray(v) for k, v in out_std.items()})
    return result
