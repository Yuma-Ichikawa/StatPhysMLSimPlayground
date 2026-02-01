# Examples

Example scripts and notebooks demonstrating `statphys-ml` usage.

## Contents

| File | Description |
|------|-------------|
| `basic_usage.ipynb` | Comprehensive tutorial covering all features |
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

### `basic_usage.ipynb`
Comprehensive tutorial covering:
- Package import and setup
- Dataset generation (Gaussian, Classification)
- Model creation (Linear, Committee)
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
