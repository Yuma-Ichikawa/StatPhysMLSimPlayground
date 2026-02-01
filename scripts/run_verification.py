"""
動作確認スクリプト: statphysパッケージの基本機能テスト
結果はtests/figディレクトリに保存されます
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# パッケージのインポート
import statphys
from statphys.dataset import GaussianDataset
from statphys.model import LinearRegression
from statphys.loss import RidgeLoss
from statphys.simulation import ReplicaSimulation, OnlineSimulation, SimulationConfig
from statphys.vis import ComparisonPlotter, OrderParamPlotter

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_replica_simulation():
    """Replica simulation (Ridge Regression) の動作確認"""
    print("=" * 60)
    print("Test 1: Replica Simulation (Ridge Regression)")
    print("=" * 60)
    
    statphys.fix_seed(42)
    
    # パラメータ
    d = 200
    rho = 1.0
    eta = 0.1
    reg_param = 0.1
    
    # データセット作成
    dataset = GaussianDataset(d=d, rho=rho, eta=eta)
    print(f"Dataset: d={d}, rho={rho}, eta={eta}")
    
    # シミュレーション設定
    config = SimulationConfig.for_replica(
        alpha_range=(0.5, 4.0),
        alpha_steps=8,
        n_seeds=3,
        lr=0.05,
        max_iter=10000,
        tol=1e-5,
        patience=50,
        reg_param=reg_param,
        verbose=False,
    )
    
    # 実行
    print("Running simulation...")
    sim = ReplicaSimulation(config)
    results = sim.run(
        dataset=dataset,
        model_class=LinearRegression,
        loss_fn=RidgeLoss(reg_param=reg_param),
    )
    print("Simulation complete!")
    
    # 結果の可視化
    alpha_values = np.array(results.experiment_results["alpha_values"])
    op_mean = np.array(results.experiment_results["order_params_mean"])
    op_std = np.array(results.experiment_results["order_params_std"])
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    param_names = [r"$m$ (overlap)", r"$q$ (self)", r"$E_g$ (gen. error)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, (ax, name, color) in enumerate(zip(axes, param_names, colors)):
        ax.errorbar(
            alpha_values,
            op_mean[:, i],
            yerr=op_std[:, i],
            marker="o",
            capsize=3,
            color=color,
            label="Experiment"
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(name)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    
    plt.suptitle(f"Ridge Regression: d={d}, ρ={rho}, η={eta}, λ={reg_param}")
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, "replica_ridge_regression.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()
    
    return results


def test_online_simulation():
    """Online SGD simulation の動作確認"""
    print("\n" + "=" * 60)
    print("Test 2: Online SGD Simulation")
    print("=" * 60)
    
    statphys.fix_seed(42)
    
    # パラメータ
    d = 100
    rho = 1.0
    eta = 0.0
    lr = 0.3 / d
    reg_param = 0.05
    
    # データセット作成
    dataset = GaussianDataset(d=d, rho=rho, eta=eta)
    print(f"Dataset: d={d}, rho={rho}, eta={eta}")
    
    # シミュレーション設定
    config = SimulationConfig.for_online(
        t_max=3.0,
        t_steps=30,
        n_seeds=3,
        lr=lr,
        reg_param=reg_param,
        verbose=False,
    )
    
    # 実行
    print("Running simulation...")
    sim = OnlineSimulation(config)
    results = sim.run(
        dataset=dataset,
        model_class=LinearRegression,
        loss_fn=RidgeLoss(reg_param=reg_param),
    )
    print("Simulation complete!")
    
    # 結果の可視化
    t_values = np.array(results.experiment_results["t_values"])
    traj_mean = np.array(results.experiment_results["trajectories_mean"])
    traj_std = np.array(results.experiment_results["trajectories_std"])
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    param_names = [r"$m$ (overlap)", r"$q$ (self)", r"$E_g$ (gen. error)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, (ax, name, color) in enumerate(zip(axes, param_names, colors)):
        ax.plot(t_values, traj_mean[:, i], color=color, linewidth=2, label="Mean")
        ax.fill_between(
            t_values,
            traj_mean[:, i] - traj_std[:, i],
            traj_mean[:, i] + traj_std[:, i],
            color=color,
            alpha=0.3,
        )
        ax.set_xlabel(r"$t = n/d$")
        ax.set_ylabel(name)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    
    plt.suptitle(f"Online SGD: d={d}, lr={lr*d:.2f}/d, λ={reg_param}")
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, "online_sgd_learning.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()
    
    return results


def test_models():
    """各種モデルの動作確認"""
    print("\n" + "=" * 60)
    print("Test 3: Model Comparison")
    print("=" * 60)
    
    from statphys.model import (
        LinearRegression,
        CommitteeMachine,
        SoftCommitteeMachine,
        TwoLayerNetwork,
    )
    
    statphys.fix_seed(42)
    d = 100
    batch_size = 32
    
    # 入力データ
    x = np.random.randn(batch_size, d).astype(np.float32)
    import torch
    x_tensor = torch.tensor(x)
    
    models = {
        "LinearRegression": LinearRegression(d=d),
        "CommitteeMachine (K=3)": CommitteeMachine(d=d, k=3),
        "SoftCommitteeMachine (K=3)": SoftCommitteeMachine(d=d, k=3, activation="erf"),
        "TwoLayerNetwork (K=50)": TwoLayerNetwork(d=d, k=50, activation="relu"),
    }
    
    print("\nModel outputs:")
    for name, model in models.items():
        with torch.no_grad():
            output = model(x_tensor)
        print(f"  {name}: output shape = {output.shape}, mean = {output.mean():.4f}")
    
    # モデルパラメータ数を可視化
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = list(models.keys())
    params = [sum(p.numel() for p in m.parameters()) for m in models.values()]
    
    ax.barh(names, params, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_xlabel("Number of Parameters")
    ax.set_title(f"Model Parameter Counts (d={d})")
    ax.grid(True, axis='x', linestyle="--", alpha=0.3)
    
    for i, (name, p) in enumerate(zip(names, params)):
        ax.text(p + 100, i, f"{p:,}", va='center')
    
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def main():
    """メイン実行関数"""
    print("\n" + "=" * 60)
    print("StatPhys-ML Package Verification")
    print(f"Version: {statphys.__version__}")
    print("=" * 60)
    
    # Test 1: Replica simulation
    test_replica_simulation()
    
    # Test 2: Online simulation
    test_online_simulation()
    
    # Test 3: Model comparison
    test_models()
    
    print("\n" + "=" * 60)
    print("All verification tests completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
