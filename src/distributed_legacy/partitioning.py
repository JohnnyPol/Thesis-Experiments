import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ResidualBlock
from src.models.resnet_ee import ResNetEE18
from src.distributed_legacy.config import resolve_path


def _entropy_confident(logits, threshold):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return bool(entropy.item() <= threshold)


class PartitionOutput(object):
    def __init__(self, status, exit_id, logits, activation, compute_time_sec):
        self.status = status
        self.exit_id = exit_id
        self.logits = logits
        self.activation = activation
        self.compute_time_sec = compute_time_sec


class ResNetEE2WayPartition1(nn.Module):
    def __init__(self, full_model):
        super(ResNetEE2WayPartition1, self).__init__()
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.exit2 = full_model.exit2
        self.avgpool = full_model.avgpool
        self.fc = full_model.fc
        self.confidence_threshold = float(full_model.confidence_threshold)

    def forward(self, x):
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


class ResNetEE3WayPartition2(nn.Module):
    def __init__(self, full_model):
        super(ResNetEE3WayPartition2, self).__init__()
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.exit2 = full_model.exit2
        self.avgpool = full_model.avgpool
        self.fc = full_model.fc
        self.confidence_threshold = float(full_model.confidence_threshold)

    def forward(self, x):
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


def extract_num_classes(dataset_cfg, model_cfg):
    return int(dataset_cfg.get("num_classes", model_cfg.get("num_classes", 10)))


def extract_entropy_threshold(model_cfg):
    early_exit_cfg = model_cfg.get("early_exit", {})
    if isinstance(early_exit_cfg, dict):
        if "confidence_threshold" in early_exit_cfg:
            return float(early_exit_cfg["confidence_threshold"])
        if "entropy_threshold" in early_exit_cfg:
            return float(early_exit_cfg["entropy_threshold"])

    if "confidence_threshold" in model_cfg:
        return float(model_cfg["confidence_threshold"])

    return 0.9


def build_full_ee_resnet18(model_cfg, dataset_cfg, repo_root, device):
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


def build_partition_module(partition_id, num_partitions, model_cfg, dataset_cfg, repo_root, device):
    full_model = build_full_ee_resnet18(
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        repo_root=repo_root,
        device=device,
    )

    if num_partitions == 2:
        if partition_id == 1:
            return ResNetEE2WayPartition1(full_model).to(device).eval()
        raise ValueError(
            "Legacy Jetson runtime only supports the final partition for 2-worker topology."
        )

    if num_partitions == 3:
        if partition_id == 2:
            return ResNetEE3WayPartition2(full_model).to(device).eval()
        raise ValueError(
            "Legacy Jetson runtime only supports the final partition for 3-worker topology."
        )

    raise ValueError(
        "Unsupported num_partitions={0}. Expected 2 or 3.".format(num_partitions)
    )
