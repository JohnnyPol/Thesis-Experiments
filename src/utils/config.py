from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def resolve_path(path_str: str | None, base_dir: str | Path) -> str | None:
    """
    Resolve a possibly relative path against a base directory.
    """
    if path_str is None:
        return None

    path = Path(path_str)
    if path.is_absolute():
        return str(path)

    return str((Path(base_dir) / path).resolve())


def load_experiment_bundle(experiment_config_path: str | Path) -> dict[str, Any]:
    """
    Load experiment config and the referenced dataset/model/system configs.

    Returns a dictionary with:
      - experiment_config
      - dataset_config
      - model_config
      - system_config
      - repo_root
    """
    experiment_config_path = Path(experiment_config_path).resolve()
    repo_root = experiment_config_path.parents[2]

    experiment_cfg = load_yaml(experiment_config_path)

    refs = experiment_cfg.get("config_refs", {})
    dataset_cfg_path = refs.get("dataset")
    model_cfg_path = refs.get("model")
    system_cfg_path = refs.get("system")

    if not dataset_cfg_path or not model_cfg_path or not system_cfg_path:
        raise ValueError(
            "Experiment config must contain config_refs.dataset, "
            "config_refs.model, and config_refs.system"
        )

    dataset_cfg_resolved = resolve_path(dataset_cfg_path, repo_root)
    model_cfg_resolved = resolve_path(model_cfg_path, repo_root)
    system_cfg_resolved = resolve_path(system_cfg_path, repo_root)

    if not dataset_cfg_resolved or not model_cfg_resolved or not system_cfg_resolved:
        raise ValueError("Failed to resolve config paths")

    dataset_cfg = load_yaml(dataset_cfg_resolved)
    model_cfg = load_yaml(model_cfg_resolved)
    system_cfg = load_yaml(system_cfg_resolved)

    return {
        "experiment_config": experiment_cfg,
        "dataset_config": dataset_cfg,
        "model_config": model_cfg,
        "system_config": system_cfg,
        "repo_root": str(repo_root),
    }