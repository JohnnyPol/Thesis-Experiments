from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_state_dict_file(
    weights_path: str | Path,
    map_location: torch.device | str,
) -> dict[str, Any]:
    path = Path(weights_path)
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the real checkpoint file. "
            "Download the checkpoint with: "
            f'git lfs pull --include="{path.as_posix()}" --exclude=""'
        )

    state_dict = torch.load(path, map_location=map_location)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict checkpoint at {path}")
    return state_dict


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(128)
    except FileNotFoundError:
        return False

    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
