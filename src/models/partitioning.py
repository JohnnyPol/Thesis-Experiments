from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ResidualBlock
from src.models.resnet_ee import ResNetEE18, ResNetEE34
from src.utils.config import resolve_path


FullEarlyExitResNet = ResNetEE18 | ResNetEE34


EE_MODEL_BUILDERS = {
    "resnet18": lambda block, num_classes, confidence_threshold: ResNetEE18(
        block,
        [2, 2, 2, 2],
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    ),
    "resnet34": lambda block, num_classes, confidence_threshold: ResNetEE34(
        block,
        [3, 4, 6, 3],
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    ),
}


def extract_architecture(model_cfg: dict[str, Any]) -> str:
    architecture = str(model_cfg.get("architecture", "resnet18")).lower()
    if architecture not in EE_MODEL_BUILDERS:
        raise ValueError(
            f"Unsupported ResNet architecture '{architecture}'. "
            f"Expected one of: {', '.join(sorted(EE_MODEL_BUILDERS))}."
        )
    return architecture


def _entropy_confident(logits: torch.Tensor, threshold: float) -> bool:
    """
    Entropy-based early-exit criterion.

    Current distributed pipeline assumes batch_size=1.
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return bool(entropy.item() <= threshold)


@dataclass
class PartitionOutput:
    status: str
    exit_id: int | None
    logits: torch.Tensor | None
    activation: torch.Tensor | None
    compute_time_sec: float


class ResNetEE3WayPartition0(nn.Module):
    """
    Stage 0:
      conv1 -> maxpool -> layer0 -> exit0 -> layer1 -> exit1

    If neither exit fires, forward activation after layer1.
    """

    def __init__(self, full_model: FullEarlyExitResNet):
        super().__init__()
        self.conv1 = full_model.conv1
        self.maxpool = full_model.maxpool
        self.layer0 = full_model.layer0
        self.layer1 = full_model.layer1
        self.exit0 = full_model.exit0
        self.exit1 = full_model.exit1
        self.confidence_threshold = float(full_model.confidence_threshold)

    def forward(self, x: torch.Tensor) -> PartitionOutput:
        start = time.time()

        x = self.conv1(x)
        x = self.maxpool(x)

        x0 = self.layer0(x)
        out0 = self.exit0(x0)
        if _entropy_confident(out0, self.confidence_threshold):
            return PartitionOutput(
                status="exited",
                exit_id=0,
                logits=out0,
                activation=None,
                compute_time_sec=time.time() - start,
            )

        x1 = self.layer1(x0)
        out1 = self.exit1(x1)
        if _entropy_confident(out1, self.confidence_threshold):
            return PartitionOutput(
                status="exited",
                exit_id=1,
                logits=out1,
                activation=None,
                compute_time_sec=time.time() - start,
            )

        return PartitionOutput(
            status="forward",
            exit_id=None,
            logits=None,
            activation=x1,
            compute_time_sec=time.time() - start,
        )


class ResNetEE3WayPartition1(nn.Module):
    """
    Stage 1:
      input activation(after layer1) -> layer2 -> exit2

    If exit2 does not fire, forward activation after layer2.
    """

    def __init__(self, full_model: FullEarlyExitResNet):
        super().__init__()
        self.layer2 = full_model.layer2
        self.exit2 = full_model.exit2
        self.confidence_threshold = float(full_model.confidence_threshold)

    def forward(self, x: torch.Tensor) -> PartitionOutput:
        start = time.time()

        x2 = self.layer2(x)
        out2 = self.exit2(x2)
        if _entropy_confident(out2, self.confidence_threshold):
            return PartitionOutput(
                status="exited",
                exit_id=2,
                logits=out2,
                activation=None,
                compute_time_sec=time.time() - start,
            )

        return PartitionOutput(
            status="forward",
            exit_id=None,
            logits=None,
            activation=x2,
            compute_time_sec=time.time() - start,
        )


class ResNetEE3WayPartition2(nn.Module):
    """
    Stage 2:
      input activation(after layer2) -> layer3 -> final classifier
    """

    def __init__(self, full_model: FullEarlyExitResNet):
        super().__init__()
        self.layer3 = full_model.layer3
        self.exit3 = getattr(full_model, "exit3", None)
        self.avgpool = full_model.avgpool if self.exit3 is None else None
        self.fc = full_model.fc if self.exit3 is None else None

    def forward(self, x: torch.Tensor) -> PartitionOutput:
        start = time.time()

        x3 = self.layer3(x)
        if self.exit3 is not None:
            out_final = self.exit3(x3)
        else:
            if self.avgpool is None or self.fc is None:
                raise RuntimeError("Missing final classifier for partition 2")
            xf = self.avgpool(x3)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)

        return PartitionOutput(
            status="completed",
            exit_id=3,
            logits=out_final,
            activation=None,
            compute_time_sec=time.time() - start,
        )


def extract_num_classes(dataset_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> int:
    return int(dataset_cfg.get("num_classes", model_cfg.get("num_classes", 10)))


def extract_entropy_threshold(model_cfg: dict[str, Any]) -> float:
    early_exit_cfg = model_cfg.get("early_exit", {})
    if isinstance(early_exit_cfg, dict):
        if "confidence_threshold" in early_exit_cfg:
            return float(early_exit_cfg["confidence_threshold"])
        if "entropy_threshold" in early_exit_cfg:
            return float(early_exit_cfg["entropy_threshold"])

    if "confidence_threshold" in model_cfg:
        return float(model_cfg["confidence_threshold"])

    return 0.9


def build_full_ee_resnet(
    model_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    repo_root: str,
    device: torch.device | str,
) -> FullEarlyExitResNet:
    """
    Build the full EE-ResNet and load weights.
    Workers then slice the full model into their local partition module.
    """
    device = torch.device(device)
    num_classes = extract_num_classes(dataset_cfg, model_cfg)
    confidence_threshold = extract_entropy_threshold(model_cfg)
    architecture = extract_architecture(model_cfg)

    model = EE_MODEL_BUILDERS[architecture](
        ResidualBlock,
        num_classes,
        confidence_threshold,
    ).to(device)

    weights_path = None
    weights_cfg = model_cfg.get("weights", {})
    if isinstance(weights_cfg, dict):
        weights_path = resolve_path(weights_cfg.get("path"), repo_root)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()
    return model


def build_full_ee_resnet18(
    model_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    repo_root: str,
    device: torch.device | str,
) -> FullEarlyExitResNet:
    return build_full_ee_resnet(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        repo_root=repo_root,
        device=device,
    )


def build_partition_module(
    partition_id: int,
    num_partitions: int,
    model_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    repo_root: str,
    device: torch.device | str,
) -> nn.Module:
    """
    Construct the local partition module for a 3-worker topology.
    """
    if num_partitions != 3:
        raise ValueError(f"Unsupported num_partitions={num_partitions}. Expected 3.")

    full_model = build_full_ee_resnet(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        repo_root=repo_root,
        device=device,
    )

    if partition_id == 0:
        return ResNetEE3WayPartition0(full_model).to(device).eval()
    if partition_id == 1:
        return ResNetEE3WayPartition1(full_model).to(device).eval()
    if partition_id == 2:
        return ResNetEE3WayPartition2(full_model).to(device).eval()

    raise ValueError(f"Unsupported partition_id={partition_id} for 3-worker topology.")
