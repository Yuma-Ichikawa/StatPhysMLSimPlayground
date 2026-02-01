"""
Example: Ridge Regression with Replica Theory

This example demonstrates how to:
1. Create a Gaussian dataset with linear teacher
2. Run replica simulation (batch gradient descent)
3. Compare experimental results with replica theory predictions
4. Visualize the results

The replica theory predicts the generalization error and order parameters
(m, q) as a function of the sample ratio α = n/d.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import from statphys
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import RidgeLoss
from statphys.simulation import ReplicaSimulation, SimulationConfig
from statphys.theory.replica import SaddlePointSolver, GaussianLinearRidgeEquations
from statphys.vis import ComparisonPlotter
from statphys.utils import fix_seed


def main():
    """Run ridge regression replica simulation."""

    # =========================================
    # 1. Configuration
    # =========================================
    d = 500  # Input dimension
    rho = 1.0  # Teacher norm ||W0||^2 / d
    eta = 0.1  # Noise variance
    reg_param = 0.1  # Ridge regularization λ

    # Fix random seed for reproducibility
    fix_seed(42)

    print("=" * 60)
    print("Ridge Regression with Replica Theory")
    print("=" * 60)
    print(f"Dimension d = {d}")
    print(f"Teacher norm ρ = {rho}")
    print(f"Noise variance η = {eta}")
    print(f"Regularization λ = {reg_param}")
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
    # 3. Setup Theory Solver (Optional)
    # =========================================
    print("\n[Step 2] Setting up replica theory solver...")

    # Define saddle-point equations for ridge regression
    ridge_equations = GaussianLinearRidgeEquations(
        rho=rho,
        eta=eta,
        reg_param=reg_param,
    )

    theory_solver = SaddlePointSolver(
        equations=ridge_equations,
        order_params=["m", "q"],
        damping=0.5,
        tol=1e-8,
        max_iter=10000,
        verbose=False,
    )
    print("  Theory solver ready.")

    # =========================================
    # 4. Configure and Run Simulation
    # =========================================
    print("\n[Step 3] Configuring simulation...")

    config = SimulationConfig.for_replica(
        alpha_range=(0.2, 5.0),
        alpha_steps=15,
        n_seeds=5,
        lr=0.01,
        max_iter=30000,
        tol=1e-6,
        patience=100,
        reg_param=reg_param,
        verbose=True,
        verbose_interval=10000,
    )

    print(f"  Alpha range: {config.alpha_range}")
    print(f"  Number of seeds: {config.n_seeds}")

    print("\n[Step 4] Running simulation...")
    sim = ReplicaSimulation(config)
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

    plotter = ComparisonPlotter()

    # Plot theory vs experiment comparison
    fig, ax = plotter.plot_theory_vs_experiment(
        results,
        params_to_plot=["m", "q", "eg"],
        param_indices={"m": 0, "q": 1, "eg": 2},
    )
    ax.set_title(f"Ridge Regression: ρ={rho}, η={eta}, λ={reg_param}")
    plt.tight_layout()

    # Save figure
    fig.savefig("replica_ridge_regression_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: replica_ridge_regression_results.png")

    # Plot generalization error only
    fig2, ax2 = plotter.plot_generalization_error(results, eg_index=2)
    ax2.set_title("Generalization Error vs Sample Ratio")
    fig2.savefig("replica_ridge_eg.png", dpi=150, bbox_inches="tight")
    print("  Saved: replica_ridge_eg.png")

    plt.show()

    # =========================================
    # 6. Print Summary Statistics
    # =========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    alpha_values = np.array(results.experiment_results["alpha_values"])
    eg_mean = np.array(results.experiment_results["order_params_mean"])[:, 2]
    eg_std = np.array(results.experiment_results["order_params_std"])[:, 2]

    print("\nGeneralization Error (Experiment):")
    for i in range(0, len(alpha_values), 3):
        print(f"  α = {alpha_values[i]:.2f}: E_g = {eg_mean[i]:.4f} ± {eg_std[i]:.4f}")

    if results.theory_results is not None:
        eg_theory = np.array(results.theory_results.order_params.get("eg", []))
        if len(eg_theory) > 0:
            print("\nGeneralization Error (Theory):")
            theory_alpha = np.array(results.theory_results.param_values)
            for i in range(0, len(theory_alpha), 3):
                print(f"  α = {theory_alpha[i]:.2f}: E_g = {eg_theory[i]:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
