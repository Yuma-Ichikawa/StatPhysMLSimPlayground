# Examples

Example scripts and notebooks demonstrating `statphys-ml` usage.

## Contents

| File | Description |
|------|-------------|
| `basic_usage.ipynb` | Comprehensive tutorial covering all features |
| `theory_vs_simulation_verification_en.ipynb` | **Theory vs Simulation verification** (Online ODE & Replica) |
| `theory_vs_simulation_verification_ja.ipynb` | **理論 vs シミュレーション検証**（日本語版） |
| `custom_components_tutorial.ipynb` | Create custom datasets, models, and losses |
| `custom_components_tutorial_ja.ipynb` | カスタムコンポーネントチュートリアル（日本語版） |
| `dataset_gallery.ipynb` | Visualization of all 22 supported datasets |
| `model_gallery.ipynb` | Visualization of all 19 supported models |
| `replica_ridge_regression.py` | Ridge regression with replica theory |
| `online_sgd_learning.py` | Online SGD learning dynamics |
| `committee_machine.py` | Soft committee machine analysis |

## Quick Start

```bash
# Install the package
pip install -e .

# Run a script
python examples/replica_ridge_regression.py

# Or open the notebook
jupyter notebook examples/basic_usage.ipynb
```

## Notebooks

### `theory_vs_simulation_verification_en.ipynb` / `theory_vs_simulation_verification_ja.ipynb`
**Verifies that thermodynamic limit theory matches finite-dimensional simulations:**
- **Online Learning (ODE)**: Order parameter trajectories as a function of normalized time $t = \tau/d$
- **Replica Method**: Equilibrium order parameters as a function of sample ratio $\alpha = n/d$

Uses Ridge Regression as a simple example to demonstrate:
- ODE solver for online learning dynamics
- Closed-form Ridge regression solution vs analytical theory
- Comparison plots showing theory vs simulation agreement

**Key features:**
- Standard Gaussian initialization ($m(0) \approx 0$, $q(0) \approx 1$)
- Error bars computed from multiple seeds
- Initial conditions matched between simulation and theory

### `basic_usage.ipynb`
Comprehensive tutorial covering:
- Package import and setup
- Dataset generation (Gaussian, Classification)
- Model creation (Linear, Committee)
- **Automatic Order Parameter Calculation** - Using `OrderParameterCalculator` and `auto_calc_order_params`
- Loss functions (Ridge, MSE)
- Running simulations (Replica, Online)
- Visualization tools

### `custom_components_tutorial.ipynb`
Step-by-step guide to creating custom components:

| Component | Base Class | Required Methods |
|-----------|------------|------------------|
| **Custom Dataset** | `BaseDataset` | `generate_sample()`, `get_teacher_params()` |
| **Custom Model** | `BaseModel` | `forward()`, `get_weight_vector()` |
| **Custom Loss** | `BaseLoss` | `_compute_loss()` |

Examples include:
- `PolynomialTeacherDataset`: Nonlinear teacher with quadratic terms
- `QuadraticModel`: Model with linear + quadratic features
- `CustomRobustLoss`: Huber-like robust loss function
- Custom order parameter functions

### `dataset_gallery.ipynb`
Visual gallery of all supported datasets:

| Category | Datasets |
|----------|----------|
| **Gaussian** | `GaussianDataset`, `GaussianClassificationDataset`, `GaussianMultiOutputDataset` |
| **Sparse** | `SparseDataset`, `BernoulliGaussianDataset` |
| **Structured** | `StructuredDataset`, `CorrelatedGaussianDataset`, `SpikedCovarianceDataset` |
| **GLM Teachers** | `LogisticTeacherDataset`, `ProbitTeacherDataset` |
| **Gaussian Mixture** | `GaussianMixtureDataset`, `MulticlassGaussianMixtureDataset` |
| **ICL Tasks** | `ICLLinearRegressionDataset`, `ICLNonlinearRegressionDataset` |
| **Sequence/Token** | `MarkovChainDataset`, `CopyTaskDataset`, `GeneralizedPottsDataset`, `TiedLowRankAttentionDataset`, `MixedGaussianSequenceDataset` |
| **Attention** | `AttentionIndexedModelDataset` |
| **Fairness** | `TeacherMixtureFairnessDataset` |
| **Noisy Labels** | `NoisyGMMSelfDistillationDataset` |

![Dataset Gallery](dataset_gallery_summary.png)

### `model_gallery.ipynb`
Visual gallery of all 19 supported models with architecture diagrams and I/O relationships:

| Category | Models |
|----------|--------|
| **Linear** | `LinearRegression`, `LinearClassifier`, `RidgeRegression` |
| **Committee** | `CommitteeMachine`, `SoftCommitteeMachine` |
| **MLP** | `TwoLayerNetwork`, `TwoLayerNetworkReLU`, `DeepNetwork` |
| **Deep Linear** | `DeepLinearNetwork` |
| **Random Features** | `RandomFeaturesModel`, `KernelRidgeModel` |
| **Softmax** | `SoftmaxRegression`, `SoftmaxRegressionWithBias` |
| **Transformer** | `SingleLayerAttention`, `SingleLayerTransformer` |
| **Sequence** | `LinearSelfAttention`, `StateSpaceModel`, `LinearRNN` |
| **Energy-Based** | `ModernHopfieldNetwork` |

![Model Gallery](model_gallery_summary.png)
