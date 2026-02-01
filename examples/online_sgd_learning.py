"""
Example: Online SGD Learning Dynamics

This example demonstrates how to:
1. Simulate online SGD learning
2. Track order parameters over time
3. Compare with theoretical ODE predictions
4. Visualize learning curves

The online learning theory predicts how order parameters evolve
as a function of time t = n/d (normalized sample count).
"""

import numpy as np
import matplotlib.pyplot as plt

# Import from statphys
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import RidgeLoss
from statphys.simulation import OnlineSimulation, SimulationConfig
from statphys.theory.online import ODESolver, OnlineSGDEquations
from statphys.vis import ComparisonPlotter, OrderParamPlotter
from statphys.utils import fix_seed


def main():
    """Run online SGD learning simulation."""

    # =========================================
    # 1. Configuration
    # =========================================
    d = 200  # Input dimension (smaller for online)
    rho = 1.0  # Teacher norm
    eta = 0.0  # Noiseless for clearer dynamics
    lr = 0.5  # Learning rate (scaled)
    reg_param = 0.1  # Regularization
    t_max = 5.0  # Maximum time t = n/d

    fix_seed(42)

    print("=" * 60)
    print("Online SGD Learning Dynamics")
    print("=" * 60)
    print(f"Dimension d = {d}")
    print(f"Teacher norm ρ = {rho}")
    print(f"Noise variance η = {eta}")
    print(f"Learning rate η_lr = {lr}")
    print(f"Regularization λ = {reg_param}")
    print(f"Max time t_max = {t_max}")
    print("=" * 60)

    # =========================================
    # 2. Create Dataset
    # =========================================
    print("\n[Step 1] Creating Gaussian dataset...")
    dataset = GaussianDataset(
        d=d,
        rho=rho,
        eta=eta,
        device="cpu",
    )
    print(f"  Dataset created: {dataset}")

    # =========================================
    # 3. Setup Theory Solver
    # =========================================
    print("\n[Step 2] Setting up ODE theory solver...")

    # Define ODE equations for online SGD
    sgd_equations = OnlineSGDEquations(
        rho=rho,
        eta_noise=eta,
        lr=lr / d,  # Scaled learning rate
        reg_param=reg_param,
    )

    theory_solver = ODESolver(
        equations=sgd_equations,
        order_params=["m", "q"],
        method="RK45",
        tol=1e-8,
        verbose=False,
    )
    print("  Theory solver ready.")

    # =========================================
    # 4. Configure and Run Simulation
    # =========================================
    print("\n[Step 3] Configuring simulation...")

    config = SimulationConfig.for_online(
        t_max=t_max,
        t_steps=50,
        n_seeds=5,
        lr=lr / d,  # Scaled learning rate for online
        reg_param=reg_param,
        verbose=True,
        verbose_interval=1000,
    )

    print(f"  Time range: [0, {t_max}]")
    print(f"  Number of seeds: {config.n_seeds}")

    print("\n[Step 4] Running online simulation...")
    sim = OnlineSimulation(config)
    results = sim.run(
        dataset=dataset,
        model_class=LinearRegression,
        loss_fn=RidgeLoss(reg_param=reg_param),
        theory_solver=theory_solver,
    )

    print("\n  Simulation complete!")

    # =========================================
    # 5. Visualize Results
    # =========================================
    print("\n[Step 5] Visualizing results...")

    # Create comparison plot
    plotter = ComparisonPlotter()

    fig, ax = plotter.plot_theory_vs_experiment(
        results,
        params_to_plot=["m", "q"],
        param_indices={"m": 0, "q": 1},
    )
    ax.set_title(f"Online SGD: lr={lr}, λ={reg_param}")
    ax.set_xlabel(r"Time $t = n/d$")
    plt.tight_layout()

    fig.savefig("online_sgd_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: online_sgd_results.png")

    # Plot order parameters with error bands
    op_plotter = OrderParamPlotter()
    fig2, ax2 = op_plotter.plot_from_result(
        results,
        param_indices={"m": 0, "q": 1, "eg": 2},
    )
    ax2.set_title("Order Parameter Trajectories")
    fig2.savefig("online_sgd_trajectories.png", dpi=150, bbox_inches="tight")
    print("  Saved: online_sgd_trajectories.png")

    plt.show()

    # =========================================
    # 6. Print Summary
    # =========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    t_values = np.array(results.experiment_results["t_values"])
    trajectories_mean = np.array(results.experiment_results["trajectories_mean"])

    print("\nFinal Order Parameters (Experiment):")
    print(f"  m(t={t_max:.1f}) = {trajectories_mean[-1, 0]:.4f}")
    print(f"  q(t={t_max:.1f}) = {trajectories_mean[-1, 1]:.4f}")
    print(f"  E_g(t={t_max:.1f}) = {trajectories_mean[-1, 2]:.4f}")

    if results.theory_results is not None:
        m_theory = np.array(results.theory_results.order_params["m"])
        q_theory = np.array(results.theory_results.order_params["q"])
        print("\nFinal Order Parameters (Theory):")
        print(f"  m(t={t_max:.1f}) = {m_theory[-1]:.4f}")
        print(f"  q(t={t_max:.1f}) = {q_theory[-1]:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
