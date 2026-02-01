"""
Comprehensive order parameter calculation for various model and dataset types.

This module provides automatic extraction of ALL relevant order parameters:
1. Student-Teacher overlaps: All overlaps between student and teacher parameters
2. Student self-overlaps: All self-overlaps of student parameters
3. Teacher self-overlaps: From dataset parameters
4. O(1) quantities: Bias terms, second-layer weights, scalars (not normalized by dimension)

For parameters scaling with dimension d:
    - Overlap = (1/d) * param1^T @ param2

For O(1) parameters (bias, scalars, second-layer weights):
    - Recorded directly without normalization
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ModelType(Enum):
    """Enum for different model architectures."""

    LINEAR = "linear"
    COMMITTEE = "committee"
    TWO_LAYER = "two_layer"
    DEEP = "deep"
    TRANSFORMER = "transformer"
    UNKNOWN = "unknown"


class TaskType(Enum):
    """Enum for different task types."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    UNKNOWN = "unknown"


@dataclass
class OrderParameters:
    """
    Container for all order parameters.

    Attributes:
        student_teacher_overlaps: Dict of M_{i,j} = (1/d) * W_i^T @ W0_j overlaps
        student_self_overlaps: Dict of Q_{i,j} = (1/d) * W_i^T @ W_j overlaps
        teacher_self_overlaps: Dict of R_{i,j} = (1/d) * W0_i^T @ W0_j overlaps
        scalars: Dict of O(1) quantities (bias, second-layer weights, etc.)
        generalization_error: Computed generalization error
        metadata: Additional information (model type, task type, etc.)

    """

    student_teacher_overlaps: dict[str, Any] = field(default_factory=dict)
    student_self_overlaps: dict[str, Any] = field(default_factory=dict)
    teacher_self_overlaps: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    generalization_error: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary."""
        result = {}

        # Add overlaps with prefixes
        for key, value in self.student_teacher_overlaps.items():
            result[f"M_{key}"] = value

        for key, value in self.student_self_overlaps.items():
            result[f"Q_{key}"] = value

        for key, value in self.teacher_self_overlaps.items():
            result[f"R_{key}"] = value

        for key, value in self.scalars.items():
            result[key] = value

        if self.generalization_error is not None:
            result["eg"] = self.generalization_error

        return result

    def to_list(self, model_type: ModelType) -> list[float]:
        """
        Convert to list format for simulation compatibility.

        Format depends on model type:
        - LINEAR: [m, q, eg]
        - COMMITTEE: [m_avg, q_diag_avg, q_offdiag_avg, eg]
        - TWO_LAYER: [m_avg, q_diag_avg, q_offdiag_avg, a_norm, eg]
        """
        eg = self.generalization_error or 0.0

        if model_type == ModelType.LINEAR:
            m = self.student_teacher_overlaps.get("w_W0", 0.0)
            q = self.student_self_overlaps.get("w_w", 0.0)
            return [m, q, eg]

        elif model_type in (ModelType.COMMITTEE, ModelType.TWO_LAYER):
            m_avg = self.student_teacher_overlaps.get("avg", 0.0)
            q_diag = self.student_self_overlaps.get("diag_avg", 0.0)
            q_offdiag = self.student_self_overlaps.get("offdiag_avg", 0.0)

            values = [m_avg, q_diag, q_offdiag]

            if model_type == ModelType.TWO_LAYER:
                a_norm = self.scalars.get("a_norm", 0.0)
                values.append(a_norm)

            values.append(eg)
            return values

        else:
            # Default fallback
            m = self.student_teacher_overlaps.get("avg", 0.0)
            if m == 0.0:
                m = next(iter(self.student_teacher_overlaps.values()), 0.0)
            q = self.student_self_overlaps.get("diag_avg", 0.0)
            if q == 0.0:
                q = next(iter(self.student_self_overlaps.values()), 0.0)
            return [m, q, eg]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=== Order Parameters ==="]

        if self.student_teacher_overlaps:
            lines.append("\n[Student-Teacher Overlaps (M)]")
            for k, v in self.student_teacher_overlaps.items():
                if isinstance(v, (list, np.ndarray)):
                    lines.append(f"  {k}: matrix of shape {np.array(v).shape}")
                else:
                    lines.append(f"  {k}: {v:.6f}")

        if self.student_self_overlaps:
            lines.append("\n[Student Self-Overlaps (Q)]")
            for k, v in self.student_self_overlaps.items():
                if isinstance(v, (list, np.ndarray)):
                    lines.append(f"  {k}: matrix of shape {np.array(v).shape}")
                else:
                    lines.append(f"  {k}: {v:.6f}")

        if self.teacher_self_overlaps:
            lines.append("\n[Teacher Self-Overlaps (R)]")
            for k, v in self.teacher_self_overlaps.items():
                if isinstance(v, (list, np.ndarray)):
                    lines.append(f"  {k}: matrix of shape {np.array(v).shape}")
                else:
                    lines.append(f"  {k}: {v:.6f}")

        if self.scalars:
            lines.append("\n[Scalars (O(1) quantities)]")
            for k, v in self.scalars.items():
                if isinstance(v, (list, np.ndarray)):
                    arr = np.array(v)
                    lines.append(f"  {k}: array of shape {arr.shape}")
                else:
                    lines.append(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        if self.generalization_error is not None:
            lines.append(f"\n[Generalization Error]: {self.generalization_error:.6f}")

        return "\n".join(lines)


class OrderParameterCalculator:
    """
    Comprehensive order parameter calculator.

    Automatically extracts ALL relevant order parameters from model and dataset:
    - All student-teacher overlaps
    - All student self-overlaps
    - All teacher self-overlaps (from dataset params)
    - All O(1) quantities (bias, second-layer weights, etc.)

    Example:
        >>> calculator = OrderParameterCalculator()
        >>> params = calculator(dataset, model)
        >>> print(params.summary())

        >>> # Use in simulation
        >>> results = sim.run(
        ...     dataset=dataset,
        ...     model_class=LinearRegression,
        ...     loss_fn=loss_fn,
        ...     calc_order_params=calculator,
        ... )

    """

    def __init__(
        self,
        return_format: str = "list",
        include_matrices: bool = True,
        include_teacher_overlaps: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize OrderParameterCalculator.

        Args:
            return_format: 'list' for simulation compatibility, 'dict' for flat dict,
                          'object' for OrderParameters dataclass.
            include_matrices: If True, include full overlap matrices.
            include_teacher_overlaps: If True, compute teacher self-overlaps.
            verbose: If True, print detection info.

        """
        self.return_format = return_format
        self.include_matrices = include_matrices
        self.include_teacher_overlaps = include_teacher_overlaps
        self.verbose = verbose

    def __call__(
        self,
        dataset: Any,
        model: nn.Module,
    ) -> list[float] | dict[str, Any] | OrderParameters:
        """
        Compute all order parameters.

        Args:
            dataset: Dataset instance with get_teacher_params() method.
            model: Trained model instance.

        Returns:
            Order parameters in requested format.

        """
        # Detect types
        model_type = self._detect_model_type(model)
        task_type = self._detect_task_type(dataset)

        if self.verbose:
            print(f"Detected model type: {model_type.value}")
            print(f"Detected task type: {task_type.value}")

        # Get teacher parameters
        teacher_params = dataset.get_teacher_params()

        # Create order parameters container
        params = OrderParameters()
        params.metadata = {
            "model_type": model_type.value,
            "task_type": task_type.value,
            "d": getattr(model, "d", None),
        }

        # Extract student parameters
        student_params = self._extract_student_params(model)

        # Extract teacher parameters
        teacher_weight_params = self._extract_teacher_weight_params(teacher_params)

        # Compute all student-teacher overlaps
        self._compute_student_teacher_overlaps(
            student_params, teacher_weight_params, model.d, params
        )

        # Compute all student self-overlaps
        self._compute_student_self_overlaps(student_params, model.d, params)

        # Compute teacher self-overlaps if requested
        if self.include_teacher_overlaps:
            self._compute_teacher_self_overlaps(teacher_weight_params, model.d, params)

        # Extract O(1) scalars
        self._extract_scalars(model, teacher_params, params)

        # Compute generalization error
        params.generalization_error = self._compute_generalization_error(
            params, task_type, teacher_params
        )

        # Return in requested format
        if self.return_format == "list":
            return params.to_list(model_type)
        elif self.return_format == "dict":
            return params.to_dict()
        else:
            return params

    def _detect_model_type(self, model: nn.Module) -> ModelType:
        """Detect model architecture type."""
        class_name = model.__class__.__name__.lower()

        if "committee" in class_name:
            return ModelType.COMMITTEE
        if "twolayer" in class_name or "two_layer" in class_name:
            return ModelType.TWO_LAYER
        if "deep" in class_name or "mlp" in class_name:
            return ModelType.DEEP
        if "transformer" in class_name or "attention" in class_name:
            return ModelType.TRANSFORMER
        if "linear" in class_name or "ridge" in class_name:
            return ModelType.LINEAR

        # Check by structure
        if (
            hasattr(model, "k")
            and hasattr(model, "W")
            and model.W.dim() == 2
            and model.W.shape[0] > 1
        ):
            if hasattr(model, "a"):
                return ModelType.TWO_LAYER
            return ModelType.COMMITTEE

        if hasattr(model, "layers") and len(model.layers) > 2:
            return ModelType.DEEP

        return ModelType.LINEAR

    def _detect_task_type(self, dataset: Any) -> TaskType:
        """Detect task type from dataset."""
        class_name = dataset.__class__.__name__.lower()

        if "classification" in class_name:
            if "multiclass" in class_name:
                return TaskType.MULTICLASS_CLASSIFICATION
            return TaskType.BINARY_CLASSIFICATION
        if "logistic" in class_name or "probit" in class_name:
            return TaskType.BINARY_CLASSIFICATION

        teacher_params = dataset.get_teacher_params()
        if "flip_prob" in teacher_params:
            return TaskType.BINARY_CLASSIFICATION
        if "n_classes" in teacher_params:
            n = teacher_params["n_classes"]
            return (
                TaskType.BINARY_CLASSIFICATION if n == 2 else TaskType.MULTICLASS_CLASSIFICATION
            )

        return TaskType.REGRESSION

    def _extract_student_params(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Extract all weight parameters from student model.

        Returns dict with parameter name -> tensor.
        Parameters are categorized by their scaling:
        - Keys starting with 'W' or 'w': scale with d, need normalization
        - Other keys: O(1), recorded in scalars
        """
        params = {}

        # Linear models
        if hasattr(model, "W"):
            W = model.W.detach()
            if W.dim() == 2:
                if W.shape[0] == 1 or W.shape[1] == 1:
                    # Linear: (d,1) or (1,d)
                    params["w"] = W.flatten()
                else:
                    # Committee/TwoLayer: (K, d)
                    K = W.shape[0]
                    for i in range(K):
                        params[f"W_{i}"] = W[i]
                    params["W_matrix"] = W
            else:
                params["w"] = W.flatten()

        # Deep networks
        if hasattr(model, "layers"):
            for layer_idx, layer in enumerate(model.layers):
                if hasattr(layer, "weight"):
                    weight = layer.weight.detach()
                    if weight.shape[0] > 1 and weight.shape[1] > 1:
                        # Hidden layer weight matrix
                        for i in range(weight.shape[0]):
                            params[f"W{layer_idx}_{i}"] = weight[i]
                        params[f"W{layer_idx}_matrix"] = weight
                    else:
                        params[f"W{layer_idx}"] = weight.flatten()

        return params

    def _extract_teacher_weight_params(
        self, teacher_params: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Extract weight parameters from teacher."""
        params = {}

        W0 = teacher_params.get("W0")
        if W0 is not None:
            if not isinstance(W0, torch.Tensor):
                W0 = torch.tensor(W0, dtype=torch.float32)
            W0 = W0.detach()

            if W0.dim() == 2:
                if W0.shape[0] == 1 or W0.shape[1] == 1:
                    params["W0"] = W0.flatten()
                else:
                    K0 = W0.shape[0]
                    for i in range(K0):
                        params[f"W0_{i}"] = W0[i]
                    params["W0_matrix"] = W0
            else:
                params["W0"] = W0.flatten()

        # Check for additional teacher weights
        for key in ["W0_1", "W0_2", "teacher_W"]:
            if key in teacher_params:
                val = teacher_params[key]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                params[key] = val.detach().flatten()

        return params

    def _compute_student_teacher_overlaps(
        self,
        student_params: dict[str, torch.Tensor],
        teacher_params: dict[str, torch.Tensor],
        d: int,
        result: OrderParameters,
    ) -> None:
        """Compute all student-teacher overlaps M = (1/d) * student^T @ teacher."""
        overlaps = {}
        total = 0.0
        count = 0

        for s_key, s_param in student_params.items():
            if s_key.endswith("_matrix"):
                continue  # Skip matrices, process individual vectors

            for t_key, t_param in teacher_params.items():
                if t_key.endswith("_matrix"):
                    continue

                # Only compute if dimensions match
                if s_param.numel() == t_param.numel():
                    overlap = (s_param.flatten() @ t_param.flatten()).item() / d
                    key = f"{s_key}_{t_key}"
                    overlaps[key] = overlap
                    total += overlap
                    count += 1

        # Compute matrix overlaps if available
        if "W_matrix" in student_params and "W0_matrix" in teacher_params:
            W = student_params["W_matrix"]
            W0 = teacher_params["W0_matrix"]
            if W.shape[1] == W0.shape[1]:  # Same d
                M = (W @ W0.T / d).cpu().numpy()
                if self.include_matrices:
                    overlaps["matrix"] = M.tolist()
                overlaps["matrix_mean"] = M.mean()
                overlaps["matrix_diag"] = np.diag(M).mean() if min(M.shape) > 0 else 0.0

        if count > 0:
            overlaps["avg"] = total / count

        result.student_teacher_overlaps = overlaps

    def _compute_student_self_overlaps(
        self,
        student_params: dict[str, torch.Tensor],
        d: int,
        result: OrderParameters,
    ) -> None:
        """Compute all student self-overlaps Q = (1/d) * student_i^T @ student_j."""
        overlaps = {}

        # Get individual weight vectors (exclude matrices)
        weight_keys = [k for k in student_params if not k.endswith("_matrix")]

        # Compute pairwise overlaps
        diag_sum = 0.0
        diag_count = 0
        offdiag_sum = 0.0
        offdiag_count = 0

        for i, key_i in enumerate(weight_keys):
            param_i = student_params[key_i]
            for j, key_j in enumerate(weight_keys[i:], start=i):
                param_j = student_params[key_j]

                if param_i.numel() == param_j.numel():
                    overlap = (param_i.flatten() @ param_j.flatten()).item() / d
                    overlap_key = f"{key_i}_{key_j}"
                    overlaps[overlap_key] = overlap

                    if i == j:
                        diag_sum += overlap
                        diag_count += 1
                    else:
                        offdiag_sum += overlap
                        offdiag_count += 1

        # Compute matrix overlap if available
        if "W_matrix" in student_params:
            W = student_params["W_matrix"]
            Q = (W @ W.T / d).cpu().numpy()
            K = Q.shape[0]

            if self.include_matrices:
                overlaps["matrix"] = Q.tolist()

            overlaps["matrix_diag"] = np.diag(Q).tolist()
            overlaps["matrix_offdiag"] = (Q - np.diag(np.diag(Q))).tolist() if K > 1 else []

        # Summary statistics
        if diag_count > 0:
            overlaps["diag_avg"] = diag_sum / diag_count
        if offdiag_count > 0:
            overlaps["offdiag_avg"] = offdiag_sum / offdiag_count

        result.student_self_overlaps = overlaps

    def _compute_teacher_self_overlaps(
        self,
        teacher_params: dict[str, torch.Tensor],
        d: int,
        result: OrderParameters,
    ) -> None:
        """Compute teacher self-overlaps R = (1/d) * teacher_i^T @ teacher_j."""
        overlaps = {}

        weight_keys = [k for k in teacher_params if not k.endswith("_matrix")]

        for i, key_i in enumerate(weight_keys):
            param_i = teacher_params[key_i]
            for _j, key_j in enumerate(weight_keys[i:], start=i):
                param_j = teacher_params[key_j]

                if param_i.numel() == param_j.numel():
                    overlap = (param_i.flatten() @ param_j.flatten()).item() / d
                    overlap_key = f"{key_i}_{key_j}"
                    overlaps[overlap_key] = overlap

        # Matrix overlap
        if "W0_matrix" in teacher_params:
            W0 = teacher_params["W0_matrix"]
            R = (W0 @ W0.T / d).cpu().numpy()
            if self.include_matrices:
                overlaps["matrix"] = R.tolist()
            overlaps["diag_avg"] = np.diag(R).mean()

        result.teacher_self_overlaps = overlaps

    def _extract_scalars(
        self,
        model: nn.Module,
        teacher_params: dict[str, Any],
        result: OrderParameters,
    ) -> None:
        """Extract O(1) scalar quantities (bias, second-layer weights, etc.)."""
        scalars = {}

        # Second layer weights (O(1) in high-dimensional limit)
        if hasattr(model, "a"):
            a = model.a.detach().cpu().numpy()
            scalars["a"] = a.tolist()
            scalars["a_norm"] = (a**2).mean()
            scalars["a_mean"] = a.mean()
            scalars["a_std"] = a.std()

        # Bias terms
        if hasattr(model, "b"):
            b = model.b.detach().cpu().numpy()
            if b.size == 1:
                scalars["b"] = float(b)
            else:
                scalars["b"] = b.tolist()

        if hasattr(model, "bias"):
            bias = model.bias
            if bias is not None:
                bias = bias.detach().cpu().numpy()
                if bias.size == 1:
                    scalars["bias"] = float(bias)
                else:
                    scalars["bias"] = bias.tolist()

        # Check layers for bias
        if hasattr(model, "layers"):
            for layer_idx, layer in enumerate(model.layers):
                if hasattr(layer, "bias") and layer.bias is not None:
                    b = layer.bias.detach().cpu().numpy()
                    if b.size == 1:
                        scalars[f"b{layer_idx}"] = float(b)
                    else:
                        scalars[f"b{layer_idx}"] = b.tolist()

        # Teacher scalars
        for key in ["rho", "eta", "sigma", "flip_prob", "lambda", "reg_param"]:
            if key in teacher_params:
                scalars[f"teacher_{key}"] = teacher_params[key]

        # Teacher second layer weights
        if "a0" in teacher_params:
            a0 = teacher_params["a0"]
            if isinstance(a0, torch.Tensor):
                a0 = a0.detach().cpu().numpy()
            if isinstance(a0, np.ndarray):
                scalars["a0"] = a0.tolist()
                scalars["a0_norm"] = (a0**2).mean()
            else:
                scalars["a0"] = a0

        # Model-specific scalars
        if hasattr(model, "k"):
            scalars["K"] = model.k

        result.scalars = scalars

    def _compute_generalization_error(
        self,
        params: OrderParameters,
        task_type: TaskType,
        teacher_params: dict[str, Any],
    ) -> float:
        """Compute generalization error based on order parameters."""
        rho = teacher_params.get("rho", 1.0)
        eta = teacher_params.get("eta", 0.0)

        # Get main overlaps
        m = params.student_teacher_overlaps.get("avg", 0.0)
        if m == 0.0:
            m = params.student_teacher_overlaps.get("w_W0", 0.0)
        if m == 0.0:
            # Try first value
            for v in params.student_teacher_overlaps.values():
                if isinstance(v, (int, float)):
                    m = v
                    break

        q = params.student_self_overlaps.get("diag_avg", 0.0)
        if q == 0.0:
            q = params.student_self_overlaps.get("w_w", 0.0)
        if q == 0.0:
            for k, v in params.student_self_overlaps.items():
                if isinstance(v, (int, float)) and k.split("_")[0] == k.split("_")[-1]:
                    q = v
                    break

        if task_type == TaskType.REGRESSION:
            # E_g = 0.5 * (rho - 2m + q) + noise
            eg = 0.5 * (rho - 2 * m + q)
            if eta > 0:
                eg += 0.5 * eta
            return eg
        else:
            # Classification error: P(error) = (1/pi) * arccos(m / sqrt(q * rho))
            if q > 0 and rho > 0:
                cos_angle = np.clip(m / np.sqrt(q * rho), -1, 1)
                return np.arccos(cos_angle) / np.pi
            return 0.5

    @staticmethod
    def get_param_names(model_type: ModelType | str) -> list[str]:
        """Get order parameter names for a given model type."""
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        if model_type == ModelType.LINEAR:
            return ["m", "q", "eg"]
        elif model_type == ModelType.COMMITTEE:
            return ["m_avg", "q_diag_avg", "q_offdiag_avg", "eg"]
        elif model_type == ModelType.TWO_LAYER:
            return ["m_avg", "q_diag_avg", "q_offdiag_avg", "a_norm", "eg"]
        else:
            return ["m", "q", "eg"]


def auto_calc_order_params(
    dataset: Any,
    model: nn.Module,
    return_format: str = "list",
) -> list[float] | dict[str, Any] | OrderParameters:
    """
    Convenience function to compute all order parameters automatically.

    Args:
        dataset: Dataset instance.
        model: Model instance.
        return_format: 'list', 'dict', or 'object'.

    Returns:
        Order parameters in requested format.

    Example:
        >>> params = auto_calc_order_params(dataset, model)
        >>> print(params)  # [m, q, eg]

        >>> params_dict = auto_calc_order_params(dataset, model, return_format="dict")
        >>> print(params_dict)  # {'M_w_W0': 0.8, 'Q_w_w': 0.9, ...}

        >>> params_obj = auto_calc_order_params(dataset, model, return_format="object")
        >>> print(params_obj.summary())

    """
    calculator = OrderParameterCalculator(return_format=return_format)
    return calculator(dataset, model)


# Pre-configured calculators
default_calculator = OrderParameterCalculator(return_format="list")
detailed_calculator = OrderParameterCalculator(
    return_format="object", include_matrices=True, include_teacher_overlaps=True
)
