from pathlib import Path

import yaml


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise IOError("YAML config file not found: {0}".format(path))

    with open(str(path), "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return data or {}


def resolve_path(path_str, base_dir):
    if path_str is None:
        return None

    path = Path(path_str)
    if path.is_absolute():
        return str(path)

    return str((Path(base_dir) / path).resolve())


def load_experiment_bundle(experiment_config_path):
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

    dataset_cfg = load_yaml(resolve_path(dataset_cfg_path, repo_root))
    model_cfg = load_yaml(resolve_path(model_cfg_path, repo_root))
    system_cfg = load_yaml(resolve_path(system_cfg_path, repo_root))

    return {
        "experiment_config": experiment_cfg,
        "dataset_config": dataset_cfg,
        "model_config": model_cfg,
        "system_config": system_cfg,
        "repo_root": str(repo_root),
    }
