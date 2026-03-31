from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ResidualBlock
from src.models.resnet_ee import ResNetEE18
from src.utils.config import resolve_path


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


class ResNetEE2WayPartition0(nn.Module):
    """
    2-worker topology, stage 0:
      conv1 -> maxpool -> layer0 -> exit0
                        -> layer1 -> exit1

    If neither exit fires, forward activation after layer1.
    """

    def __init__(self, full_model: ResNetEE18):
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


class ResNetEE2WayPartition1(nn.Module):
    """
    2-worker topology, stage 1:
      input activation(after layer1) -> layer2 -> exit2
                                     -> layer3 -> avgpool -> fc(final)
    """

    def __init__(self, full_model: ResNetEE18):
        super().__init__()
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.exit2 = full_model.exit2
        self.avgpool = full_model.avgpool
        self.fc = full_model.fc
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

        x3 = self.layer3(x2)
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


class ResNetEE3WayPartition0(nn.Module):
    """
    3-worker topology, stage 0:
      conv1 -> maxpool -> layer0 -> exit0

    If exit0 does not fire, forward activation after layer0.
    """

    def __init__(self, full_model: ResNetEE18):
        super().__init__()
        self.conv1 = full_model.conv1
        self.maxpool = full_model.maxpool
        self.layer0 = full_model.layer0
        self.exit0 = full_model.exit0
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

        return PartitionOutput(
            status="forward",
            exit_id=None,
            logits=None,
            activation=x0,
            compute_time_sec=time.time() - start,
        )


class ResNetEE3WayPartition1(nn.Module):
    """
    3-worker topology, stage 1:
      input activation(after layer0) -> layer1 -> exit1

    If exit1 does not fire, forward activation after layer1.
    """

    def __init__(self, full_model: ResNetEE18):
        super().__init__()
        self.layer1 = full_model.layer1
        self.exit1 = full_model.exit1
        self.confidence_threshold = float(full_model.confidence_threshold)

    def forward(self, x: torch.Tensor) -> PartitionOutput:
        start = time.time()

        x1 = self.layer1(x)
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


class ResNetEE3WayPartition2(nn.Module):
    """
    3-worker topology, stage 2:
      input activation(after layer1) -> layer2 -> exit2
                                     -> layer3 -> avgpool -> fc(final)
    """

    def __init__(self, full_model: ResNetEE18):
        super().__init__()
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.exit2 = full_model.exit2
        self.avgpool = full_model.avgpool
        self.fc = full_model.fc
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

        x3 = self.layer3(x2)
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


def build_full_ee_resnet18(
    model_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    repo_root: str,
    device: torch.device | str,
) -> ResNetEE18:
    """
    Build the full EE-ResNet18 and load weights.
    Workers then slice the full model into their local partition module.
    """
    device = torch.device(device)
    num_classes = extract_num_classes(dataset_cfg, model_cfg)
    confidence_threshold = extract_entropy_threshold(model_cfg)

    model = ResNetEE18(
        ResidualBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
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


def build_partition_module(
    partition_id: int,
    num_partitions: int,
    model_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    repo_root: str,
    device: torch.device | str,
) -> nn.Module:
    """
    Construct the local partition module for a worker.

    Supported topologies:
      - 2 workers
      - 3 workers
    """
    full_model = build_full_ee_resnet18(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        repo_root=repo_root,
        device=device,
    )

    if num_partitions == 2:
        if partition_id == 0:
            return ResNetEE2WayPartition0(full_model).to(device).eval()
        if partition_id == 1:
            return ResNetEE2WayPartition1(full_model).to(device).eval()
        raise ValueError(f"Unsupported partition_id={partition_id} for 2-worker topology.")

    if num_partitions == 3:
        if partition_id == 0:
            return ResNetEE3WayPartition0(full_model).to(device).eval()
        if partition_id == 1:
            return ResNetEE3WayPartition1(full_model).to(device).eval()
        if partition_id == 2:
            return ResNetEE3WayPartition2(full_model).to(device).eval()
        raise ValueError(f"Unsupported partition_id={partition_id} for 3-worker topology.")

    raise ValueError(f"Unsupported num_partitions={num_partitions}. Expected 2 or 3.")