"""Print the numerical environment used by a Slurm atlas job."""

from __future__ import annotations

import importlib.util
import json
import platform

import numpy
import torch


def main() -> None:
    available = {
        name: importlib.util.find_spec(name) is not None
        for name in ("scipy", "pandas", "matplotlib", "pytest")
    }
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "optional_packages": available,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
