r"""
Teacher x paradigm cross experiments: the frontier taxonomy.

The frontier settings (SFT, RLHF, weak-to-strong, collapse) are
protocols; the teachers of `statphys.frontier.teachers` are systems.
This module crosses them: every teacher in the taxonomy is pushed
through every paradigm probe, and one scalar order parameter per
(teacher, paradigm) cell is collected into a table -- the numerical
analogue of a phase-diagram row per universality-class candidate.

Probes and their order parameters:

- ``sft``:      forgetting :math:`F = m_A^{\rm pre} - m_A^{\rm post}`
                and transfer gain :math:`\Delta_B` at fixed
                :math:`(\rho, \alpha_{\rm ft})`
- ``rlhf``:     overoptimization onset :math:`\mathrm{KL}^*`, peak gold
                reward :math:`G_{\max}`, reward-model quality
                :math:`m_{\rm RM}`
- ``w2s``:      performance gap recovered (PGR) and raw weak-to-strong
                gain :math:`m_{\rm strong} - m_{\rm weak}`
- ``collapse``: signal drop :math:`m(0) - m(g_{\max})` of the fully
                synthetic loop and the terminal overlap

(The ICL setting is excluded: its "teacher" is a task distribution, not
a single network, so the weight-structure axis does not apply.)

Example:
    >>> from statphys.frontier.taxonomy import run_taxonomy, taxonomy_markdown
    >>> res = run_taxonomy(teachers=["random_mlp", "trained_digits"], n_seeds=1)
    >>> print(taxonomy_markdown(res))

"""

from __future__ import annotations

import numpy as np

from statphys.frontier.teachers import TEACHER_TAXONOMY, make_teacher

__all__ = ["PARADIGM_PROBES", "run_taxonomy", "taxonomy_markdown"]


def _budget(quick: bool) -> dict:
    """Shared training budgets for the probes."""
    return {
        "epochs": 400 if quick else 2000,
        "sft_ckpts": 4 if quick else 10,
        "sft_epc": 50 if quick else 100,
        "rlhf_epochs": 300 if quick else 1000,
        "rlhf_kl_coefs": [1.0, 0.1] if quick else [3.0, 1.0, 0.3, 0.1, 0.03],
        "rlhf_steps": 200 if quick else 800,
        "gens": 3 if quick else 8,
        "n_eval": 512 if quick else 2048,
    }


def _probe_sft(name: str, seed: int, quick: bool) -> dict[str, float]:
    from statphys.frontier.sft import run_finetune

    b = _budget(quick)
    teacher, sampler, d = make_teacher(name, d=64, hidden=16, seed=seed)
    res = run_finetune(
        d=d,
        hidden=16,
        similarity=0.5,
        alpha_pre=8.0,
        alpha_ft=4.0,
        n_checkpoints=b["sft_ckpts"],
        epochs_per_checkpoint=b["sft_epc"],
        pretrain_epochs=b["epochs"],
        n_probe=b["n_eval"],
        seed=seed,
        teacher=teacher,
        input_sampler=sampler,
    )
    return {"forgetting": res["forgetting"], "transfer_gain": res["transfer_gain"]}


def _probe_rlhf(name: str, seed: int, quick: bool) -> dict[str, float]:
    from statphys.frontier.rlhf import run_overoptimization

    b = _budget(quick)
    teacher, sampler, d = make_teacher(name, d=32, hidden=16, seed=seed, noise_std=0.0)
    res = run_overoptimization(
        d=d,
        hidden=16,
        alpha_r=8.0,
        policy="shift",
        kl_coefs=b["rlhf_kl_coefs"],
        policy_steps=b["rlhf_steps"],
        n_eval=b["n_eval"],
        epochs=b["rlhf_epochs"],
        n_probe=b["n_eval"],
        seed=seed,
        teacher=teacher,
        input_sampler=sampler,
    )
    return {"kl_star": res["kl_star"], "gold_max": res["gold_max"], "m_RM": res["m_RM"]}


def _probe_w2s(name: str, seed: int, quick: bool) -> dict[str, float]:
    from statphys.frontier.weak_to_strong import run_weak_to_strong

    b = _budget(quick)
    teacher, sampler, d = make_teacher(name, d=64, hidden=16, seed=seed)
    res = run_weak_to_strong(
        d=d,
        hidden_weak=2,
        hidden_strong=32,
        alpha_weak=4.0,
        alpha_strong=16.0,
        epochs=b["epochs"],
        n_probe=b["n_eval"],
        seed=seed,
        teacher=teacher,
        input_sampler=sampler,
    )
    return {"pgr": res["pgr"], "w2s_gain": res["m_strong"] - res["m_weak"]}


def _probe_collapse(name: str, seed: int, quick: bool) -> dict[str, float]:
    from statphys.frontier.collapse import run_collapse

    b = _budget(quick)
    teacher, sampler, d = make_teacher(name, d=64, hidden=16, seed=seed)
    res = run_collapse(
        d=d,
        hidden=16,
        alpha=8.0,
        n_generations=b["gens"],
        p_real=0.0,
        epochs=b["epochs"],
        n_probe=b["n_eval"],
        seed=seed,
        teacher=teacher,
        input_sampler=sampler,
    )
    return {"collapse_drop": float(res["m"][0] - res["m"][-1]), "m_final": float(res["m"][-1])}


PARADIGM_PROBES = {
    "sft": _probe_sft,
    "rlhf": _probe_rlhf,
    "w2s": _probe_w2s,
    "collapse": _probe_collapse,
}

# Headline metric shown per paradigm in the summary figure/table.
KEY_METRICS = {
    "sft": "forgetting",
    "rlhf": "kl_star",
    "w2s": "pgr",
    "collapse": "collapse_drop",
}


def run_taxonomy(
    teachers: list[str] | None = None,
    paradigms: list[str] | None = None,
    n_seeds: int = 2,
    quick: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run every paradigm probe on every teacher of the taxonomy.

    Args:
        teachers: Teacher names (default: the full TEACHER_TAXONOMY).
        paradigms: Probe names (default: all of PARADIGM_PROBES).
        n_seeds: Repetitions per (teacher, paradigm) cell.
        quick: Small smoke-test budgets.
        verbose: Print progress.

    Returns:
        Dict with "teachers", "paradigms", and "cells": a nested dict
        cells[teacher][paradigm] = {metric: (mean, std)}.

    """
    teachers = list(teachers) if teachers is not None else list(TEACHER_TAXONOMY)
    paradigms = list(paradigms) if paradigms is not None else list(PARADIGM_PROBES)

    cells: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for t_name in teachers:
        cells[t_name] = {}
        for p_name in paradigms:
            probe = PARADIGM_PROBES[p_name]
            metric_vals: dict[str, list[float]] = {}
            for s in range(n_seeds):
                out = probe(t_name, seed=s, quick=quick)
                for k, v in out.items():
                    metric_vals.setdefault(k, []).append(float(v))
            cells[t_name][p_name] = {
                k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in metric_vals.items()
            }
            if verbose:
                key = KEY_METRICS[p_name]
                mean, std = cells[t_name][p_name][key]
                print(f"[{t_name} x {p_name}] {key} = {mean:+.3f} +- {std:.3f}")
    return {"teachers": teachers, "paradigms": paradigms, "cells": cells}


def taxonomy_markdown(result: dict) -> str:
    """
    Render a taxonomy result as a markdown table (teachers x paradigms).

    Each cell shows the paradigm's headline metric as mean +- std.

    Args:
        result: Output of `run_taxonomy`.

    Returns:
        Markdown table string.

    """
    paradigms = result["paradigms"]
    header = (
        "| teacher | family | " + " | ".join(f"{p} ({KEY_METRICS[p]})" for p in paradigms) + " |"
    )
    sep = "|---" * (len(paradigms) + 2) + "|"
    lines = [header, sep]
    for t_name in result["teachers"]:
        family = TEACHER_TAXONOMY[t_name].family if t_name in TEACHER_TAXONOMY else "?"
        row = [f"`{t_name}`", family]
        for p in paradigms:
            mean, std = result["cells"][t_name][p][KEY_METRICS[p]]
            row.append(f"{mean:+.3f} ± {std:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
