"""
Example: Committee Machine Analysis

This example demonstrates how to:
1. Create a dataset with multi-output teacher
2. Train soft committee machines
3. Compare different hidden layer sizes
4. Analyze order parameter matrices

Committee machines are important models for studying
feature learning in neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import from statphys
from statphys.dataset import GaussianMultiOutputDataset
from statphys.model import SoftCommitteeMachine
from statphys.loss import MSELoss
from statphys.simulation import ReplicaSimulation, SimulationConfig, SimulationRunner
from statphys.vis import ComparisonPlotter
from statphys.utils import fix_seed


def main():
    """Run committee machine analysis."""

    # =========================================
    # 1. Configuration
    # =========================================
    d = 300  # Input dimension
    k_teacher = 2  # Teacher hidden units
    k_student = 2  # Student hidden units
    rho = 1.0  # Weight norm per unit
    eta = 0.0  # Noiseless
    reg_param = 0.01

    fix_seed(42)

    print("=" * 60)
    print("Soft Committee Machine Analysis")
    print("=" * 60)
    print(f"Dimension d = {d}")
    print(f"Teacher hidden units K0 = {k_teacher}")
    print(f"Student hidden units K = {k_student}")
    print(f"Weight norm ρ = {rho}")
    print("=" * 60)

    # =========================================
    # 2. Create Dataset with Committee Teacher
    # =========================================
    print("\n[Step 1] Creating multi-output dataset...")

    dataset = GaussianMultiOutputDataset(
        d=d,
        k=k_teacher,
        rho=rho,
        eta=eta,
        activation="erf",  # Soft committee uses erf
        aggregation="mean",
        device="cpu",
    )
    print(f"  Dataset created: {dataset}")

    # =========================================
    # 3. Configure Simulation
    # =========================================
    print("\n[Step 2] Configuring simulation...")

    config = SimulationConfig.for_replica(
        alpha_range=(0.5, 10.0),
        alpha_steps=12,
        n_seeds=3,
        lr=0.01,
        max_iter=20000,
        tol=1e-5,
        patience=100,
        reg_param=reg_param,
        verbose=True,
        verbose_interval=5000,
    )

    # =========================================
    # 4. Define Custom Order Parameter Calculator
    # =========================================
    def calc_committee_order_params(dataset, model):
        """
        Calculate order parameters for committee machine.

        Returns Q (student self-overlap) and M (student-teacher overlap).
        """
        teacher_params = dataset.get_teacher_params()
        W0 = teacher_params.get("W0")  # (K0, d)
        d = dataset.d

        W = model.W  # (K, d)

        # Student self-overlap: Q_ij = (1/d) * W_i^T @ W_j
        Q = (W @ W.T / d).detach().cpu().numpy()
        q_diag_mean = np.diag(Q).mean()

        # Student-teacher overlap
        if W0 is not None:
            M = (W @ W0.T / d).detach().cpu().numpy()
            m_mean = M.mean()
        else:
            m_mean = 0.0

        # Simplified generalization error estimate
        eg = q_diag_mean - 2 * m_mean + rho

        return [m_mean, q_diag_mean, eg]

    # =========================================
    # 5. Run Simulation
    # =========================================
    print("\n[Step 3] Running simulation...")

    # Create model class with correct k
    def create_committee_model(d):
        return SoftCommitteeMachine(
            d=d,
            k=k_student,
            activation="erf",
            init_scale=0.1,
        )

    sim = ReplicaSimulation(config)
    results = sim.run(
        dataset=dataset,
        model_class=lambda d: SoftCommitteeMachine(d=d, k=k_student, activation="erf"),
        loss_fn=MSELoss(reg_param=reg_param),
        calc_order_params=calc_committee_order_params,
    )

    print("\n  Simulation complete!")

    # =========================================
    # 6. Visualize Results
    # =========================================
    print("\n[Step 4] Visualizing results...")

    plotter = ComparisonPlotter()

    fig, ax = plotter.plot_generalization_error(
        results,
        eg_index=2,
    )
    ax.set_title(f"Soft Committee Machine: K0={k_teacher}, K={k_student}")
    plt.tight_layout()

    fig.savefig("committee_machine_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: committee_machine_results.png")

    # Plot all order parameters
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))

    alpha_values = np.array(results.experiment_results["alpha_values"])
    op_mean = np.array(results.experiment_results["order_params_mean"])
    op_std = np.array(results.experiment_results["order_params_std"])

    param_names = ["m (overlap)", "q (self)", "E_g (gen. error)"]
    for i, (ax, name) in enumerate(zip(axes2, param_names)):
        ax.errorbar(
            alpha_values,
            op_mean[:, i],
            yerr=op_std[:, i],
            marker="o",
            capsize=3,
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(name)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig2.savefig("committee_machine_order_params.png", dpi=150, bbox_inches="tight")
    print("  Saved: committee_machine_order_params.png")

    plt.show()

    # =========================================
    # 7. Print Summary
    # =========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nOrder Parameters at different α:")
    for i in range(0, len(alpha_values), 3):
        print(f"  α = {alpha_values[i]:.1f}:")
        print(f"    m = {op_mean[i, 0]:.4f} ± {op_std[i, 0]:.4f}")
        print(f"    q = {op_mean[i, 1]:.4f} ± {op_std[i, 1]:.4f}")
        print(f"    E_g = {op_mean[i, 2]:.4f} ± {op_std[i, 2]:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
