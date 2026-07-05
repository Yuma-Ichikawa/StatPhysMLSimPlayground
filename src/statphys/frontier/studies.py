"""
Ready-made frontier studies: modern learning paradigms as physics experiments.

Each study runs a full sweep of one frontier setting, produces a
publication-style figure, and saves the raw records as JSON. All are
available through the standard CLI:

    statphys study sft --quick
    statphys study rlhf
    statphys study weak_to_strong
    statphys study collapse
    statphys study icl

See docs/frontier.md for the physics narrative behind each study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "FRONTIER_STUDIES",
    "study_sft",
    "study_rlhf",
    "study_weak_to_strong",
    "study_collapse",
    "study_icl",
]


def _jsonable(obj):
    """Recursively convert numpy containers to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _save(result: dict, fig, out_dir: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(_jsonable(result), indent=2))
    fig.savefig(out_dir / f"{name}.png", dpi=140)
    plt.close(fig)
    print(f"saved -> {out_dir}/{name}.json, .png")


def study_sft(out_dir: Path, quick: bool) -> None:
    """
    SFT phase diagram: catastrophic forgetting and the transfer sign boundary.

    Left: m_A(t), m_B(t) fine-tuning trajectories at three task
    similarities. Right: forgetting F in the (similarity, alpha_ft)
    plane with the transfer-gain sign boundary overlaid.
    """
    import matplotlib.pyplot as plt

    from statphys.frontier.sft import run_finetune, sweep_sft_phase_diagram

    d = 48 if quick else 64
    common = {"d": d, "hidden": 8 if quick else 16, "alpha_pre": 8.0, "noise_std": 0.1}
    sims = [0.0, 0.5, 0.9] if quick else [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    alphas_ft = [1.0, 4.0] if quick else [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    n_seeds = 1 if quick else 3
    ckpt = {
        "n_checkpoints": 8 if quick else 20,
        "epochs_per_checkpoint": 50 if quick else 100,
        "pretrain_epochs": 500 if quick else 2000,
    }

    # example trajectories
    traj_sims = [0.0, 0.5, 0.9]
    trajs = {
        rho: run_finetune(similarity=rho, alpha_ft=4.0, seed=0, **common, **ckpt)
        for rho in traj_sims
    }
    # phase diagram
    grid = sweep_sft_phase_diagram(sims, alphas_ft, n_seeds=n_seeds, **common, **ckpt)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(traj_sims)))
    for color, (rho, res) in zip(colors, trajs.items(), strict=True):
        ax.plot(res["epochs"], res["m_A"], "-", color=color, label=rf"$m_A$ ($\rho={rho}$)")
        ax.plot(res["epochs"], res["m_B"], "--", color=color, label=rf"$m_B$ ($\rho={rho}$)")
    ax.set_xlabel("fine-tuning epochs")
    ax.set_ylabel(r"overlap with teacher $\hat m$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("fine-tuning trajectories: forgetting A while learning B")

    ax = axes[1]
    im = ax.pcolormesh(
        grid["alphas_ft"], grid["similarities"], grid["forgetting"], cmap="magma", shading="auto"
    )
    fig.colorbar(im, ax=ax, label=r"forgetting $F = m_A^{\rm pre} - m_A^{\rm post}$")
    try:
        cs = ax.contour(
            grid["alphas_ft"],
            grid["similarities"],
            grid["transfer_gain"],
            levels=[0.0],
            colors="cyan",
            linewidths=2,
        )
        ax.clabel(cs, fmt={0.0: "transfer = 0"}, fontsize=8)
    except ValueError:
        pass  # contour may not exist if the gain does not change sign
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha_{\rm ft} = n_{\rm ft} / d$")
    ax.set_ylabel(r"task similarity $\rho$")
    ax.set_title("forgetting phase diagram + transfer sign boundary")

    fig.suptitle(f"SFT as a two-teacher problem (tanh MLP, d={d})")
    fig.tight_layout()
    _save(
        {
            "grid": {k: grid[k] for k in grid},
            "trajectories": {
                str(r): {k: t[k] for k in ("epochs", "m_A", "m_B")} for r, t in trajs.items()
            },
        },
        fig,
        out_dir,
        "sft",
    )


def study_rlhf(out_dir: Path, quick: bool) -> None:
    """
    Reward overoptimization (Goodhart) turnover and phase boundary.

    Left: gold vs proxy reward as a function of the policy KL budget
    (KL-regularized shift policy) for several reward-model data budgets
    alpha_r; the gold curves peak and turn over while the proxy keeps
    rising. Right: the overoptimization point KL* and achievable peak
    gold reward vs alpha_r.
    """
    import matplotlib.pyplot as plt

    from statphys.frontier.rlhf import sweep_overoptimization

    alphas_r = [2.0, 16.0] if quick else [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    res = sweep_overoptimization(
        alphas_r,
        n_seeds=1 if quick else 3,
        d=32,
        hidden=8 if quick else 16,
        policy="shift",
        kl_coefs=[3.0, 0.3, 0.03] if quick else None,
        policy_steps=200 if quick else 1200,
        n_eval=512 if quick else 2048,
        epochs=300 if quick else 1500,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(res["alphas_r"])))
    for i, a in enumerate(res["alphas_r"]):
        ax.plot(
            res["kl_curves"][i],
            res["gold_curves"][i],
            "-o",
            color=colors[i],
            markersize=3,
            label=rf"gold, $\alpha_r={a:g}$",
        )
        ax.plot(res["kl_curves"][i], res["proxy_curves"][i], ":", color=colors[i], alpha=0.6)
    ax.set_xlabel(r"optimization strength KL$(\pi \,\|\, \pi_0)$")
    ax.set_ylabel("reward (z-scored; solid gold, dotted proxy)")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Goodhart: proxy keeps rising, gold peaks then falls")

    ax = axes[1]
    ax.errorbar(
        res["alphas_r"],
        res["kl_star"],
        yerr=res["kl_star_std"],
        fmt="o-",
        capsize=3,
        color="crimson",
        label=r"KL$^*$ (gold-reward peak)",
    )
    ax.errorbar(
        res["alphas_r"],
        res["gold_max"],
        yerr=res["gold_max_std"],
        fmt="s--",
        capsize=3,
        color="darkorange",
        label=r"peak gold reward $G_{\max}$",
    )
    ax2 = ax.twinx()
    ax2.plot(res["alphas_r"], res["m_RM"], "d:", color="steelblue", alpha=0.7)
    ax2.set_ylabel(r"reward-model quality $m_{\rm RM}$", color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha_r = n_{\rm pref} / d$")
    ax.set_ylabel(r"KL$^*$ and $G_{\max}$")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("hacking onset and peak utility vs reward-model data")

    fig.suptitle("reward-model overoptimization (KL-regularized policy vs a hackable gold reward)")
    fig.tight_layout()
    _save({k: res[k] for k in res}, fig, out_dir, "rlhf")


def study_weak_to_strong(out_dir: Path, quick: bool) -> None:
    """
    Weak-to-strong generalization surface.

    Left: m_strong vs m_weak across the sweep (points above the
    diagonal = student surpasses its supervisor). Right: PGR heatmap in
    the (alpha_weak, alpha_strong) plane.
    """
    import matplotlib.pyplot as plt

    from statphys.frontier.weak_to_strong import sweep_weak_to_strong

    a_w = [2.0, 8.0] if quick else [1.0, 2.0, 4.0, 8.0, 16.0]
    a_s = [4.0, 16.0] if quick else [2.0, 4.0, 8.0, 16.0, 32.0]
    res = sweep_weak_to_strong(
        a_w,
        a_s,
        n_seeds=1 if quick else 3,
        d=48 if quick else 64,
        epochs=500 if quick else 3000,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    mw, ms = res["m_weak"].ravel(), res["m_strong"].ravel()
    sc = ax.scatter(
        mw,
        ms,
        c=np.repeat(np.log2(res["alphas_strong"])[None, :], len(a_w), axis=0).ravel(),
        cmap="viridis",
        s=45,
        edgecolor="k",
        linewidth=0.4,
    )
    fig.colorbar(sc, ax=ax, label=r"$\log_2 \alpha_{\rm strong}$")
    lims = [min(mw.min(), ms.min()) - 0.03, 1.0]
    ax.plot(lims, lims, "--", color="gray")
    ax.fill_between(lims, lims, [1.0, 1.0], alpha=0.08, color="green")
    ax.text(0.05, 0.9, "student > supervisor", transform=ax.transAxes, color="green", fontsize=9)
    ax.set_xlabel(r"weak supervisor $m_{\rm weak}$")
    ax.set_ylabel(r"strong student $m_{\rm strong}$ (vs truth)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("weak-to-strong gain")

    ax = axes[1]
    im = ax.pcolormesh(
        res["alphas_strong"],
        res["alphas_weak"],
        res["pgr"],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="performance gap recovered (PGR)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha_{\rm strong}$ (weak-label data)")
    ax.set_ylabel(r"$\alpha_{\rm weak}$ (true-label data)")
    ax.set_title("PGR surface: when imitation turns into generalization")

    fig.suptitle("weak-to-strong generalization (truth -> weak MLP -> strong MLP)")
    fig.tight_layout()
    _save({k: res[k] for k in res}, fig, out_dir, "weak_to_strong")


def study_collapse(out_dir: Path, quick: bool) -> None:
    """
    Model collapse under recursive synthetic data.

    Left: overlap-with-truth m(g) across generations for several real-
    data fractions p. Middle: output-variance ratio q(g)/q(0) (tail
    loss). Right: terminal overlap vs p -- the collapse boundary.
    """
    import matplotlib.pyplot as plt

    from statphys.frontier.collapse import sweep_collapse

    p_reals = [0.0, 0.3, 1.0] if quick else [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    res = sweep_collapse(
        p_reals,
        n_seeds=1 if quick else 3,
        d=48 if quick else 64,
        n_generations=4 if quick else 12,
        epochs=500 if quick else 2000,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.85, len(res["p_reals"])))
    for i, p in enumerate(res["p_reals"]):
        axes[0].plot(
            res["generations"],
            res["m_curves"][i],
            "o-",
            color=colors[i],
            markersize=3.5,
            label=rf"$p_{{\rm real}}={p:g}$",
        )
        axes[1].plot(res["generations"], res["q_curves"][i], "s-", color=colors[i], markersize=3.5)
    axes[0].set_ylabel(r"overlap with truth $\hat m(g)$")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel(r"output variance ratio $q(g)/q(0)$")
    for ax in axes[:2]:
        ax.set_xlabel("generation $g$")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_title("signal decay over generations")
    axes[1].set_title("variance / tail collapse")

    axes[2].errorbar(
        res["p_reals"],
        res["m_final"],
        yerr=res["m_final_std"],
        fmt="o-",
        capsize=3,
        color="crimson",
    )
    axes[2].set_xlabel(r"real-data fraction $p_{\rm real}$")
    axes[2].set_ylabel(r"terminal overlap $\hat m(g_{\max})$")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[2].set_title("real-data anchoring boundary")

    fig.suptitle("model collapse: recursive training on synthetic labels")
    fig.tight_layout()
    _save({k: res[k] for k in res}, fig, out_dir, "collapse")


def study_icl(out_dir: Path, quick: bool) -> None:
    """
    Emergence of in-context learning vs pretraining task diversity.

    Left: ICL score (unseen tasks), memorization score (pool tasks), and
    ridge alignment vs N_tasks. Right: unseen-task error against the
    Bayes-optimal ridge error -- the algorithmic transition from task
    retrieval to in-context regression.
    """
    import matplotlib.pyplot as plt

    from statphys.frontier.icl import sweep_icl

    n_tasks_values = [2, 32, 256] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    res = sweep_icl(
        n_tasks_values,
        n_seeds=1 if quick else 2,
        d=8,
        k=16,
        steps=200 if quick else 3000,
        n_eval=512 if quick else 2048,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.errorbar(
        res["n_tasks"],
        res["icl_score"],
        yerr=res["icl_score_std"],
        fmt="o-",
        capsize=3,
        color="crimson",
        label="ICL score (unseen tasks)",
    )
    ax.errorbar(
        res["n_tasks"],
        res["memo_score"],
        yerr=res["memo_score_std"],
        fmt="s--",
        capsize=3,
        color="steelblue",
        label="memorization score (pool tasks)",
    )
    ax.plot(
        res["n_tasks"],
        res["ridge_alignment"],
        "d:",
        color="darkorange",
        label="alignment with ridge predictor",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"pretraining task diversity $N_{\rm tasks}$")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("emergence of in-context learning")

    ax = axes[1]
    ax.errorbar(
        res["n_tasks"],
        res["eps_unseen"],
        yerr=res["eps_unseen_std"],
        fmt="o-",
        capsize=3,
        color="crimson",
        label=r"$\epsilon_{\rm unseen}$ (transformer)",
    )
    ax.plot(
        res["n_tasks"],
        res["eps_ridge"],
        "--",
        color="gray",
        label=r"$\epsilon_{\rm ridge}$ (Bayes-optimal in-context)",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\rm tasks}$")
    ax.set_ylabel("unseen-task test error")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("transition to the in-context regression solution")

    fig.suptitle("in-context learning emergence (causal transformer, linear tasks)")
    fig.tight_layout()
    _save({k: res[k] for k in res}, fig, out_dir, "icl")


def study_taxonomy(out_dir: Path, quick: bool) -> None:
    """
    Teacher x paradigm cross table: does teacher structure matter?

    Runs every paradigm probe (SFT forgetting, RLHF overoptimization,
    weak-to-strong PGR, collapse drop) on every teacher of the taxonomy
    (random / structured weights on Gaussian inputs, random and
    genuinely *trained* networks on real digit images) and renders one
    bar panel per paradigm plus a markdown table.
    """
    import matplotlib.pyplot as plt

    from statphys.frontier.taxonomy import KEY_METRICS, run_taxonomy, taxonomy_markdown
    from statphys.frontier.teachers import TEACHER_TAXONOMY

    teachers = ["random_mlp", "low_rank_mlp", "trained_digits"] if quick else None
    res = run_taxonomy(teachers=teachers, n_seeds=1 if quick else 2, quick=quick)

    paradigms = res["paradigms"]
    names = res["teachers"]
    family_color = {"random": "steelblue", "structured": "darkorange", "trained": "crimson"}

    fig, axes = plt.subplots(1, len(paradigms), figsize=(4.2 * len(paradigms), 4.8), sharey=False)
    y = np.arange(len(names))
    for ax, p in zip(np.atleast_1d(axes), paradigms, strict=True):
        key = KEY_METRICS[p]
        means = [res["cells"][t][p][key][0] for t in names]
        stds = [res["cells"][t][p][key][1] for t in names]
        colors = [
            family_color.get(TEACHER_TAXONOMY[t].family if t in TEACHER_TAXONOMY else "", "gray")
            for t in names
        ]
        ax.barh(y, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.axvline(0.0, color="k", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is np.atleast_1d(axes)[0] else [""] * len(names), fontsize=8)
        ax.set_title(f"{p}: {key}", fontsize=10)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in family_color.values()]
    fig.legend(handles, family_color.keys(), loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("frontier taxonomy: teacher structure x learning paradigm")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    md = taxonomy_markdown(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "taxonomy.md").write_text(md + "\n")
    print(md)
    _save({"result": res}, fig, out_dir, "taxonomy")


FRONTIER_STUDIES = {
    "sft": study_sft,
    "rlhf": study_rlhf,
    "weak_to_strong": study_weak_to_strong,
    "collapse": study_collapse,
    "icl": study_icl,
    "taxonomy": study_taxonomy,
}
