"""
Experiment protocols for general teacher-student setups.

TeacherStudentExperiment runs purely numerical experiments for arbitrary
architectures — no analytic theory required:

- run_sample_complexity: sweep alpha = n/d, train to convergence, and
  measure test error / overlaps. Phase transitions appear as sharp drops
  in the error-vs-alpha curve.
- run_online: single-pass online SGD; records learning dynamics in
  normalized time t = #samples / d.

Custom observables can be attached via the `metrics` dict; each metric is
a callable (student, dataset) -> float and is recorded alongside the
built-in test error.

Example:
    >>> exp = TeacherStudentExperiment(teacher=teacher,
    ...                                student_factory=make_student)
    >>> res = exp.run_sample_complexity(alphas=np.linspace(0.5, 8, 12))
    >>> res.plot()            # error bars vs alpha
    >>> res.to_dict()         # raw records for custom analysis

"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from statphys.experiment.dataset import TeacherStudentDataset
from statphys.experiment.metrics import test_error, weight_overlap
from statphys.experiment.observables import (
    binder_cumulant,
    function_order_params,
    replica_overlaps,
    specialization_index,
    susceptibility,
)
from statphys.experiment.teacher import Teacher
from statphys.utils.seed import fix_seed

MetricFn = Callable[[nn.Module, TeacherStudentDataset], float]


@dataclass
class ExperimentResult:
    """
    Container for general teacher-student experiment results.

    Attributes:
        x_name: Name of the sweep variable ("alpha" or "t").
        x_values: Sweep values.
        records: records[metric][seed_idx] is a list aligned with x_values.
        config: Experiment configuration snapshot.
        metadata: Extra info (wall time, etc.).

    """

    x_name: str
    x_values: list[float]
    records: dict[str, list[list[float]]]
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mean(self, metric: str) -> np.ndarray:
        """Seed-averaged values of a metric."""
        return np.array(self.records[metric]).mean(axis=0)

    def std(self, metric: str) -> np.ndarray:
        """Seed std of a metric."""
        return np.array(self.records[metric]).std(axis=0)

    def metrics(self) -> list[str]:
        """Names of recorded metrics."""
        return list(self.records.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary (JSON-serializable)."""
        return {
            "x_name": self.x_name,
            "x_values": self.x_values,
            "records": self.records,
            "config": self.config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        """Restore from a dictionary."""
        return cls(**data)

    def save(self, path: str) -> None:
        """Save the result as a JSON file."""
        import json
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        """Load a result saved with save()."""
        import json
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text()))

    def plot(
        self,
        metrics: list[str] | None = None,
        logx: bool = False,
        logy: bool = False,
        ax=None,
        show: bool = False,
    ):
        """
        Plot seed-averaged metrics with error bands.

        Args:
            metrics: Which metrics to draw (default: all).
            logx, logy: Log axes (useful for locating transitions).
            ax: Existing matplotlib Axes.
            show: Call plt.show().

        Returns:
            Tuple of (Figure, Axes).

        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.get_figure()

        x = np.array(self.x_values)
        for name in metrics or self.metrics():
            mean, std = self.mean(name), self.std(name)
            (line,) = ax.plot(x, mean, marker="o", markersize=4, label=name)
            ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=line.get_color())

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        xlabel = r"$\alpha = n/d$" if self.x_name == "alpha" else r"$t = \tau/d$"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("observable")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def __repr__(self) -> str:
        return (
            f"ExperimentResult({self.x_name}: {len(self.x_values)} points, "
            f"metrics={self.metrics()})"
        )


class TeacherStudentExperiment:
    """
    Numerical teacher-student experiment for arbitrary models.

    Args:
        teacher: Teacher instance, or a raw nn.Module (wrapped automatically).
        student_factory: Zero-argument callable returning a fresh student
            nn.Module. Called once per (sweep point, seed).
        d: Input dimension. Inferred from the teacher's first Linear layer
            if omitted.
        input_dist: Input distribution for TeacherStudentDataset.
        input_kwargs: Options for the input distribution.
        loss_fn: Training loss; defaults to MSE on raw outputs
            (works for sign teachers too, as ±1 regression).
        metrics: Extra observables {name: fn(student, dataset) -> float}.
            Built-in: "test_error" always; "overlap_avg" when student and
            teacher share parameter shapes.
        dataset: Optional pre-built dataset object exposing `.sample(n)`,
            `.sample_inputs(n)`, and `.get_config()`, used instead of the
            default TeacherStudentDataset. Enables generative data models
            that are not "sample x, then apply teacher" (e.g. Gaussian
            mixture classification, where the label determines x).
            `teacher` is still required and must expose a `.clean(x)`
            oracle consistent with the dataset for function_order_params.
        device: Torch device.

    """

    def __init__(
        self,
        teacher: Teacher | nn.Module,
        student_factory: Callable[[], nn.Module],
        d: int | None = None,
        input_dist: str | Callable[[int], torch.Tensor] = "gaussian",
        input_kwargs: dict[str, Any] | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        metrics: dict[str, MetricFn] | None = None,
        dataset: Any | None = None,
        device: str = "cpu",
    ):
        if not isinstance(teacher, Teacher):
            teacher = Teacher(teacher, device=device)
        self.teacher = teacher
        self.student_factory = student_factory
        self.device = torch.device(device)

        if d is None:
            d = self._infer_dim(teacher)
        self.d = d

        self.dataset = dataset or TeacherStudentDataset(
            teacher, d=d, input_dist=input_dist, input_kwargs=input_kwargs, device=device
        )
        self.loss_fn = loss_fn or (lambda pred, y: 0.5 * ((pred - y) ** 2).mean())
        self.custom_metrics = metrics or {}

    @staticmethod
    def _infer_dim(teacher: Teacher) -> int:
        model = teacher.model
        if isinstance(model, nn.Module):
            for mod in model.modules():
                if isinstance(mod, nn.Linear):
                    return mod.in_features
            for p in model.parameters():
                if p.dim() >= 2:
                    return p.shape[-1]
        raise ValueError("Could not infer input dimension d; pass d explicitly.")

    def _forward(self, student: nn.Module, X: torch.Tensor) -> torch.Tensor:
        pred = student(X)
        if pred.dim() > 1 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return pred

    def _measure(self, student: nn.Module, n_test: int) -> dict[str, float]:
        out: dict[str, float] = {"test_error": test_error(student, self.dataset, n_test=n_test)}
        tw = self.teacher.named_weights()
        if tw:
            ov = weight_overlap(student, tw)
            if "avg" in ov:
                out["overlap_avg"] = ov["avg"]
        for name, fn in self.custom_metrics.items():
            out[name] = float(fn(student, self.dataset))
        return out

    def _train_offline(
        self,
        student: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float,
        max_epochs: int,
        batch_size: int | None,
        tol: float,
        patience: int,
        weight_decay: float,
        l1_penalty: float = 0.0,
    ) -> None:
        opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
        n = X.shape[0]
        batch_size = batch_size or n
        prev, stall = float("inf"), 0

        for _ in range(max_epochs):
            perm = torch.randperm(n, device=X.device)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = self.loss_fn(self._forward(student, X[idx]), y[idx])
                if l1_penalty > 0:
                    loss = loss + l1_penalty * sum(p.abs().sum() for p in student.parameters())
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * idx.numel()
            epoch_loss /= n

            if abs(prev - epoch_loss) < tol:
                stall += 1
                if stall >= patience:
                    break
            else:
                stall = 0
            prev = epoch_loss

    def run_sample_complexity(
        self,
        alphas: list[float] | np.ndarray,
        n_seeds: int = 3,
        base_seed: int = 0,
        lr: float = 1e-2,
        max_epochs: int = 2000,
        batch_size: int | None = None,
        tol: float = 1e-7,
        patience: int = 20,
        weight_decay: float = 0.0,
        n_test: int = 2048,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Sweep the sample ratio alpha = n/d and train to convergence.

        Args:
            alphas: Sample ratios to sweep.
            n_seeds: Seeds per alpha.
            base_seed: First seed value.
            lr: Adam learning rate.
            max_epochs: Max training epochs per point.
            batch_size: Minibatch size (None = full batch).
            tol: Convergence tolerance on epoch loss.
            patience: Epochs of stagnation before early stop.
            weight_decay: L2 regularization for the optimizer.
            n_test: Fresh test samples per measurement.
            verbose: Print progress.

        Returns:
            ExperimentResult with per-seed records of all metrics vs alpha.

        """
        alphas = [float(a) for a in alphas]
        records: dict[str, list[list[float]]] = {}
        t_start = time.time()

        for s in range(n_seeds):
            seed = base_seed + s
            fix_seed(seed)
            seed_vals: dict[str, list[float]] = {}

            for a in alphas:
                n = max(1, int(a * self.d))
                X, y = self.dataset.sample(n)
                student = self.student_factory().to(self.device)
                self._train_offline(
                    student, X, y, lr, max_epochs, batch_size, tol, patience, weight_decay
                )
                for k, v in self._measure(student, n_test).items():
                    seed_vals.setdefault(k, []).append(v)
                if verbose:
                    print(
                        f"[seed {seed}] alpha={a:.3g}: " f"E_test={seed_vals['test_error'][-1]:.4f}"
                    )

            for k, v in seed_vals.items():
                records.setdefault(k, []).append(v)

        return ExperimentResult(
            x_name="alpha",
            x_values=alphas,
            records=records,
            config={
                "d": self.d,
                "n_seeds": n_seeds,
                "lr": lr,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "dataset": self.dataset.get_config(),
            },
            metadata={"wall_time_sec": time.time() - t_start},
        )

    def run_order_parameters(
        self,
        alphas: list[float] | np.ndarray,
        n_replicas: int = 4,
        share_data: bool = True,
        base_seed: int = 0,
        lr: float = 1e-2,
        max_epochs: int = 2000,
        batch_size: int | None = None,
        tol: float = 1e-7,
        patience: int = 20,
        weight_decay: float = 0.0,
        l1_penalty: float = 0.0,
        init_scale: float = 1.0,
        n_probe: int = 4096,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Sweep alpha and measure statistical-physics order parameters.

        At each alpha, `n_replicas` students are trained independently and
        function-space observables are evaluated on a shared probe set:

        per replica:
            m_hat  -- normalized teacher-student overlap (magnetization)
            q_f    -- student self-overlap E[f_s^2]
            test_error, and specialization_index when layer shapes match
        across replicas:
            q_ab_mean/std -- replica-replica overlap (RS order parameter)
            chi_m         -- susceptibility d * Var[m_hat]
            binder_m      -- Binder cumulant of m_hat (finite-size scaling)

        Args:
            alphas: Sample ratios to sweep.
            n_replicas: Independently trained students per alpha.
            share_data: If True, replicas share the same training set
                (same disorder, different initialization/dynamics);
                if False, each replica draws fresh data.
            base_seed: Base RNG seed.
            lr, max_epochs, batch_size, tol, patience, weight_decay:
                Training options (see run_sample_complexity).
            l1_penalty: Optional L1 penalty on all student parameters
                (LASSO-style; useful for sparse-recovery studies).
            init_scale: Multiply all initial student weights by this
                factor before training. Large values (>>1) shrink the
                *relative* weight movement during training and push the
                dynamics toward the lazy/NTK (kernel) regime (Chizat &
                Bach 2019); small values (~1) allow feature learning
                ("rich" regime). Tracked via the "weight_movement" metric.
            n_probe: Probe-set size for function-space observables.
            verbose: Print progress.

        Returns:
            ExperimentResult; per-replica metrics have n_replicas rows,
            cross-replica aggregates have a single row. Always includes
            "weight_movement" = ||theta_final - theta_init|| / ||theta_init||,
            a parametrization-agnostic feature-learning diagnostic.

        """
        alphas = [float(a) for a in alphas]
        per_replica: dict[str, list[list[float]]] = {}
        aggregates: dict[str, list[float]] = {}
        t_start = time.time()

        fix_seed(base_seed)
        X_probe = self.dataset.sample_inputs(n_probe)

        for ia, a in enumerate(alphas):
            n = max(1, int(a * self.d))

            fix_seed(base_seed + 1000 * ia)
            X_shared, y_shared = self.dataset.sample(n) if share_data else (None, None)

            students: list[nn.Module] = []
            rep_vals: dict[str, list[float]] = {}
            for r in range(n_replicas):
                fix_seed(base_seed + 1000 * ia + r + 1)
                if share_data:
                    X, y = X_shared, y_shared
                else:
                    X, y = self.dataset.sample(n)
                student = self.student_factory().to(self.device)
                if init_scale != 1.0:
                    with torch.no_grad():
                        for p in student.parameters():
                            p.mul_(init_scale)
                theta0 = torch.cat([p.detach().flatten() for p in student.parameters()])
                theta0_norm = theta0.norm().clamp_min(1e-12)

                self._train_offline(
                    student,
                    X,
                    y,
                    lr,
                    max_epochs,
                    batch_size,
                    tol,
                    patience,
                    weight_decay,
                    l1_penalty=l1_penalty,
                )
                students.append(student)

                theta1 = torch.cat([p.detach().flatten() for p in student.parameters()])
                rep_vals.setdefault("weight_movement", []).append(
                    ((theta1 - theta0).norm() / theta0_norm).item()
                )

                fop = function_order_params(student, self.teacher, X_probe)
                rep_vals.setdefault("m_hat", []).append(fop["m_hat"])
                rep_vals.setdefault("q_f", []).append(fop["q_f"])
                rep_vals.setdefault("test_error", []).append(
                    test_error(student, self.dataset, n_test=n_probe)
                )
                spec = specialization_index(student, self.teacher)
                if not np.isnan(spec):
                    rep_vals.setdefault("specialization", []).append(spec)
                for name, fn in self.custom_metrics.items():
                    rep_vals.setdefault(name, []).append(float(fn(student, self.dataset)))

            for k, vals in rep_vals.items():
                per_replica.setdefault(k, [[] for _ in range(n_replicas)])
                for r, v in enumerate(vals):
                    per_replica[k][r].append(v)

            q_ab = replica_overlaps(students, X_probe)
            aggregates.setdefault("q_ab_mean", []).append(float(q_ab["q_ab_mean"]))
            aggregates.setdefault("q_ab_std", []).append(float(q_ab["q_ab_std"]))
            aggregates.setdefault("chi_m", []).append(
                susceptibility(rep_vals["m_hat"], scale=self.d)
            )
            aggregates.setdefault("binder_m", []).append(binder_cumulant(rep_vals["m_hat"]))

            if verbose:
                m_bar = float(np.mean(rep_vals["m_hat"]))
                print(
                    f"alpha={a:.3g}: m_hat={m_bar:.3f} "
                    f"q_ab={q_ab['q_ab_mean']:.3f} chi={aggregates['chi_m'][-1]:.3g}"
                )

        records: dict[str, list[list[float]]] = dict(per_replica)
        for k, v in aggregates.items():
            records[k] = [v]

        return ExperimentResult(
            x_name="alpha",
            x_values=alphas,
            records=records,
            config={
                "d": self.d,
                "n_replicas": n_replicas,
                "share_data": share_data,
                "lr": lr,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "l1_penalty": l1_penalty,
                "init_scale": init_scale,
                "n_probe": n_probe,
                "dataset": self.dataset.get_config(),
            },
            metadata={"wall_time_sec": time.time() - t_start},
        )

    def run_training_dynamics(
        self,
        alpha: float = 1.0,
        n_replicas: int = 3,
        epochs: int = 10000,
        n_evals: int = 50,
        base_seed: int = 0,
        lr: float = 1e-3,
        batch_size: int | None = None,
        weight_decay: float = 0.0,
        init_scale: float = 1.0,
        share_data: bool = True,
        n_probe: int = 4096,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Epoch-resolved training dynamics at fixed alpha (grokking-style).

        Trains `n_replicas` students in lockstep on the same (or fresh)
        training set and records train error, test error, and the
        teacher overlap m_hat at log-spaced epochs, plus the cross-replica
        overlap q_ab. This is the protocol for *temporal* phase
        transitions: delayed generalization (grokking), plateaus, and
        stagewise learning appear as structure in these trajectories.

        Args:
            alpha: Sample ratio n/d of the (fixed) training set.
            n_replicas: Students trained in parallel.
            epochs: Total training epochs.
            n_evals: Number of measurement points (log-spaced in epoch).
            base_seed: Base RNG seed.
            lr: Adam learning rate.
            batch_size: Minibatch size (None = full batch).
            weight_decay: L2 regularization (essential for grokking).
            init_scale: Multiply all initial student weights by this
                factor; large values (4-10) induce delayed
                generalization (Liu et al. 2022, "Omnigrok").
            share_data: Replicas share the training set if True.
            n_probe: Probe-set size for function-space observables.
            verbose: Print progress.

        Returns:
            ExperimentResult with x_name="epoch". Per-replica records:
            train_error, test_error, m_hat; aggregates: q_ab_mean/std.

        """
        n = max(1, int(alpha * self.d))
        eval_epochs = np.unique(
            np.round(np.logspace(0, np.log10(max(epochs, 1)), n_evals)).astype(int)
        )
        eval_epochs = eval_epochs[eval_epochs <= epochs]
        eval_set = set(eval_epochs.tolist())

        fix_seed(base_seed)
        X_probe = self.dataset.sample_inputs(n_probe)
        X_shared, y_shared = self.dataset.sample(n) if share_data else (None, None)

        students: list[nn.Module] = []
        optimizers: list[torch.optim.Optimizer] = []
        data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for r in range(n_replicas):
            fix_seed(base_seed + r + 1)
            student = self.student_factory().to(self.device)
            if init_scale != 1.0:
                with torch.no_grad():
                    for p in student.parameters():
                        p.mul_(init_scale)
            students.append(student)
            optimizers.append(
                torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
            )
            data.append((X_shared, y_shared) if share_data else self.dataset.sample(n))

        per_replica: dict[str, list[list[float]]] = {
            k: [[] for _ in range(n_replicas)] for k in ("train_error", "test_error", "m_hat")
        }
        aggregates: dict[str, list[float]] = {"q_ab_mean": [], "q_ab_std": []}
        t_start = time.time()
        bs = batch_size or n

        def measure() -> None:
            for r, student in enumerate(students):
                X_tr, y_tr = data[r]
                with torch.no_grad():
                    pred = self._forward(student, X_tr)
                    per_replica["train_error"][r].append(0.5 * ((pred - y_tr) ** 2).mean().item())
                per_replica["test_error"][r].append(
                    test_error(student, self.dataset, n_test=n_probe)
                )
                fop = function_order_params(student, self.teacher, X_probe)
                per_replica["m_hat"][r].append(fop["m_hat"])
            q_ab = replica_overlaps(students, X_probe)
            aggregates["q_ab_mean"].append(float(q_ab["q_ab_mean"]))
            aggregates["q_ab_std"].append(float(q_ab["q_ab_std"]))

        for epoch in range(1, int(epochs) + 1):
            for r, student in enumerate(students):
                X_tr, y_tr = data[r]
                perm = torch.randperm(n, device=X_tr.device)
                for start in range(0, n, bs):
                    idx = perm[start : start + bs]
                    optimizers[r].zero_grad()
                    loss = self.loss_fn(self._forward(student, X_tr[idx]), y_tr[idx])
                    loss.backward()
                    optimizers[r].step()

            if epoch in eval_set:
                measure()
                if verbose and (epoch == eval_epochs[-1] or epoch == eval_epochs[0]):
                    print(
                        f"epoch={epoch}: train={per_replica['train_error'][0][-1]:.3g} "
                        f"test={per_replica['test_error'][0][-1]:.3g}"
                    )

        records: dict[str, list[list[float]]] = dict(per_replica)
        for k, v in aggregates.items():
            records[k] = [v]

        return ExperimentResult(
            x_name="epoch",
            x_values=[float(e) for e in eval_epochs],
            records=records,
            config={
                "d": self.d,
                "alpha": alpha,
                "n_replicas": n_replicas,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "init_scale": init_scale,
                "share_data": share_data,
                "n_probe": n_probe,
                "dataset": self.dataset.get_config(),
            },
            metadata={"wall_time_sec": time.time() - t_start},
        )

    def run_online(
        self,
        t_max: float = 20.0,
        t_steps: int = 50,
        n_seeds: int = 3,
        base_seed: int = 0,
        lr: float = 0.1,
        batch_size: int = 1,
        n_test: int = 2048,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Single-pass online SGD in normalized time t = #samples / d.

        Args:
            t_max: Maximum normalized time.
            t_steps: Number of measurement points.
            n_seeds: Seeds.
            base_seed: First seed.
            lr: SGD learning rate.
            batch_size: Samples per SGD step (1 = classic online).
            n_test: Fresh test samples per measurement.
            verbose: Print progress.

        Returns:
            ExperimentResult with per-seed records of all metrics vs t.

        """
        t_values = np.linspace(0.0, t_max, t_steps)
        eval_steps = np.unique((t_values * self.d / batch_size).astype(int))
        # Re-derive actual t grid from integer step counts
        t_values = eval_steps * batch_size / self.d
        n_steps = int(eval_steps.max()) if len(eval_steps) else 0

        records: dict[str, list[list[float]]] = {}
        t_start = time.time()

        for s in range(n_seeds):
            seed = base_seed + s
            fix_seed(seed)
            student = self.student_factory().to(self.device)
            opt = torch.optim.SGD(student.parameters(), lr=lr)
            seed_vals: dict[str, list[float]] = {}

            eval_set = set(eval_steps.tolist())
            if 0 in eval_set:
                for k, v in self._measure(student, n_test).items():
                    seed_vals.setdefault(k, []).append(v)

            for step in range(1, n_steps + 1):
                X, y = self.dataset.sample(batch_size)
                opt.zero_grad()
                loss = self.loss_fn(self._forward(student, X), y)
                loss.backward()
                opt.step()

                if step in eval_set:
                    for k, v in self._measure(student, n_test).items():
                        seed_vals.setdefault(k, []).append(v)

            if verbose:
                print(f"[seed {seed}] final E_test={seed_vals['test_error'][-1]:.4f}")
            for k, v in seed_vals.items():
                records.setdefault(k, []).append(v)

        return ExperimentResult(
            x_name="t",
            x_values=t_values.tolist(),
            records=records,
            config={
                "d": self.d,
                "n_seeds": n_seeds,
                "lr": lr,
                "batch_size": batch_size,
                "dataset": self.dataset.get_config(),
            },
            metadata={"wall_time_sec": time.time() - t_start},
        )
