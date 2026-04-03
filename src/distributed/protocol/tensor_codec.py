from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


_DTYPE_TO_NUMPY: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "float16": np.dtype(np.float16),
    "int64": np.dtype(np.int64),
    "int32": np.dtype(np.int32),
    "int16": np.dtype(np.int16),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
}

_TORCH_TO_DTYPE_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def is_supported_torch_dtype(dtype: torch.dtype) -> bool:
    return dtype in _TORCH_TO_DTYPE_STR


def is_supported_dtype_str(dtype_str: str) -> bool:
    return dtype_str in _DTYPE_TO_NUMPY


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype not in _TORCH_TO_DTYPE_STR:
        raise ValueError(f"Unsupported torch dtype for transport: {dtype}")
    return _TORCH_TO_DTYPE_STR[dtype]


def dtype_str_to_numpy(dtype_str: str) -> np.dtype:
    if dtype_str not in _DTYPE_TO_NUMPY:
        raise ValueError(f"Unsupported dtype string for transport: {dtype_str}")
    return _DTYPE_TO_NUMPY[dtype_str]


def infer_tensor_metadata(tensor: torch.Tensor) -> tuple[list[int], str]:
    """
    Return shape and dtype string for a tensor that will be sent on the wire.
    """
    if not is_supported_torch_dtype(tensor.dtype):
        raise ValueError(f"Unsupported tensor dtype for transport: {tensor.dtype}")

    shape = list(tensor.shape)
    dtype_str = torch_dtype_to_str(tensor.dtype)
    return shape, dtype_str


def tensor_to_bytes(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    """
    Convert a tensor into raw bytes plus explicit metadata.

    The tensor is always detached, moved to CPU, and made contiguous before
    serialization so the receiver can reconstruct it deterministically.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_to_bytes expected a torch.Tensor")

    tensor_cpu = tensor.detach().cpu().contiguous()

    shape, dtype_str = infer_tensor_metadata(tensor_cpu)
    array = tensor_cpu.numpy()
    payload = array.tobytes(order="C")

    return payload, shape, dtype_str


def bytes_to_tensor(
    payload: bytes,
    shape: Sequence[int],
    dtype_str: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Reconstruct a tensor from raw bytes and explicit metadata.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")

    np_dtype = dtype_str_to_numpy(dtype_str)
    expected_numel = int(np.prod(shape)) if len(shape) > 0 else 1
    expected_nbytes = expected_numel * np_dtype.itemsize

    actual_nbytes = len(payload)
    if actual_nbytes != expected_nbytes:
        raise ValueError(
            "Tensor payload size mismatch: "
            f"expected {expected_nbytes} bytes for shape={list(shape)} dtype={dtype_str}, "
            f"got {actual_nbytes}"
        )

    array = np.frombuffer(payload, dtype=np_dtype).reshape(tuple(shape))
    tensor = torch.from_numpy(array.copy())
    return tensor.to(device)


def tensor_nbytes(shape: Sequence[int], dtype_str: str) -> int:
    """
    Compute expected payload size in bytes from shape and dtype.
    """
    np_dtype = dtype_str_to_numpy(dtype_str)
    numel = int(np.prod(shape)) if len(shape) > 0 else 1
    return numel * np_dtype.itemsize