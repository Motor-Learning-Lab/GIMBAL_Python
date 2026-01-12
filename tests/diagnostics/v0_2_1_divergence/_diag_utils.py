"""Diagnostic utilities for v0.2.1 divergence debugging.

Provides common functions for path management, environment collection,
safe logp evaluation, and gradient compilation.
"""

import json
import platform
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
import scipy


def make_paths(group_desc: str, run_id: str) -> Dict[str, Path]:
    """Create run-specific and 'latest' paths for diagnostics.

    Parameters
    ----------
    group_desc : str
        Group description (e.g., "group_1_build_only_sanity")
    run_id : str
        Run identifier (e.g., ISO timestamp or "local_run")

    Returns
    -------
    paths : dict
        Dictionary with keys:
        - results_dir: base results directory
        - run_dir: run-specific directory
        - latest_dir: symlink/copy to latest run
        - results_json: JSON file path
        - report_md: Markdown report path
        - plots_dir: Directory for plots
        - debug_dir: Directory for debug artifacts
    """
    base_dir = Path(__file__).parent / "results" / group_desc
    run_dir = base_dir / f"run_{run_id}"
    latest_dir = base_dir / "latest"

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "results_dir": base_dir,
        "run_dir": run_dir,
        "latest_dir": latest_dir,
        "results_json": run_dir / f"results_{group_desc}.json",
        "report_md": run_dir / f"report_{group_desc}.md",
        "plots_dir": run_dir / "plots",
        "debug_dir": run_dir / "debug",
    }

    # Create subdirectories
    paths["plots_dir"].mkdir(exist_ok=True)
    paths["debug_dir"].mkdir(exist_ok=True)

    return paths


def collect_environment() -> Dict[str, Any]:
    """Collect environment information.

    Returns
    -------
    env : dict
        Environment details including Python, PyMC, PyTensor, ArviZ versions,
        NumPy, SciPy, and platform information.
    """
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pymc_version": pm.__version__,
        "pytensor_version": pytensor.__version__,
        "arviz_version": az.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "timestamp": datetime.now().isoformat(),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON with stable UTF-8 encoding.

    Parameters
    ----------
    path : Path
        Output file path
    payload : dict
        Data to serialize
    """

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert(obj)

    clean_payload = recursive_convert(payload)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_payload, f, indent=2, ensure_ascii=False)


def save_text(path: Path, text: str) -> None:
    """Save text file with UTF-8 encoding.

    Parameters
    ----------
    path : Path
        Output file path
    text : str
        Text content to write
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_point_logps(model: pm.Model, point: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Safely evaluate model log-probabilities at a point.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    point : dict
        Parameter point to evaluate

    Returns
    -------
    result : dict
        Dictionary with:
        - total_logp: Total log probability (float or None if error)
        - point_logps: Dictionary of {var_name: logp_value}
        - has_nan: Boolean flag
        - has_inf: Boolean flag
        - error: Error message if any
    """
    result = {
        "total_logp": None,
        "point_logps": {},
        "has_nan": False,
        "has_inf": False,
        "error": None,
    }

    try:
        # Compute total logp
        result["total_logp"] = float(model.compile_logp()(point))

        # Compute individual term logps
        point_logps = model.point_logps(point)
        result["point_logps"] = {k: float(v) for k, v in point_logps.items()}

        # Check for NaN/Inf
        all_values = [result["total_logp"]] + list(result["point_logps"].values())
        result["has_nan"] = any(np.isnan(v) for v in all_values if v is not None)
        result["has_inf"] = any(np.isinf(v) for v in all_values if v is not None)

    except Exception as e:
        result["error"] = str(e)
        result["has_nan"] = True  # Treat errors as problematic

    return result


def compile_logp_and_grad(model: pm.Model) -> Tuple[callable, callable]:
    """Compile log-probability and gradient functions.

    Parameters
    ----------
    model : pm.Model
        PyMC model

    Returns
    -------
    logp_fn : callable
        Function that takes a point dict and returns total logp
    grad_fn : callable
        Function that takes a point dict and returns gradient dict
    """
    logp_fn = model.compile_logp()
    dlogp_fn = model.compile_dlogp()

    return logp_fn, dlogp_fn


def summarize_worst_terms(
    point_logps: Dict[str, float], n: int = 10
) -> List[Dict[str, Any]]:
    """Summarize the worst (most negative) log-probability terms.

    Parameters
    ----------
    point_logps : dict
        Dictionary of {var_name: logp_value}
    n : int
        Number of worst terms to return

    Returns
    -------
    worst_terms : list
        List of dicts with 'name' and 'value' keys, sorted ascending by value
    """
    sorted_terms = sorted(point_logps.items(), key=lambda x: x[1])
    return [{"name": name, "value": float(value)} for name, value in sorted_terms[:n]]


def format_worst_terms_table(worst_terms: List[Dict[str, Any]]) -> str:
    """Format worst terms as a markdown table.

    Parameters
    ----------
    worst_terms : list
        List of dicts with 'name' and 'value' keys

    Returns
    -------
    table : str
        Markdown table string
    """
    lines = [
        "| Rank | Variable Name | Log-Probability |",
        "|------|---------------|-----------------|",
    ]

    for i, term in enumerate(worst_terms, 1):
        name = term["name"]
        value = term["value"]
        if np.isnan(value):
            value_str = "NaN"
        elif np.isinf(value):
            value_str = "-∞" if value < 0 else "∞"
        else:
            value_str = f"{value:.4f}"
        lines.append(f"| {i} | `{name}` | {value_str} |")

    return "\n".join(lines)


def format_grad_components_table(components: List[Dict[str, Any]], n: int = 20) -> str:
    """Format gradient components as a markdown table.

    Parameters
    ----------
    components : list
        List of dicts with 'name', 'index' (optional), and 'value' keys
    n : int
        Number of components to include

    Returns
    -------
    table : str
        Markdown table string
    """
    lines = [
        "| Rank | Variable | Index | Gradient Value |",
        "|------|----------|-------|----------------|",
    ]

    for i, comp in enumerate(components[:n], 1):
        name = comp["name"]
        idx = comp.get("index", "")
        value = comp["value"]

        if np.isnan(value):
            value_str = "NaN"
        elif np.isinf(value):
            value_str = "±∞"
        else:
            value_str = f"{value:.6e}"

        lines.append(f"| {i} | `{name}` | {idx} | {value_str} |")

    return "\n".join(lines)
