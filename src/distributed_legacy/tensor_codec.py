from typing import Sequence, Tuple

import numpy as np
import torch


_DTYPE_TO_NUMPY = {
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

_TORCH_TO_DTYPE_STR = {
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


def torch_dtype_to_str(dtype):
    if dtype not in _TORCH_TO_DTYPE_STR:
        raise ValueError("Unsupported torch dtype for transport: {0}".format(dtype))
    return _TORCH_TO_DTYPE_STR[dtype]


def dtype_str_to_numpy(dtype_str):
    if dtype_str not in _DTYPE_TO_NUMPY:
        raise ValueError("Unsupported dtype string for transport: {0}".format(dtype_str))
    return _DTYPE_TO_NUMPY[dtype_str]


def infer_tensor_metadata(tensor):
    shape = list(tensor.shape)
    dtype_str = torch_dtype_to_str(tensor.dtype)
    return shape, dtype_str


def tensor_to_bytes(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_to_bytes expected a torch.Tensor")

    tensor_cpu = tensor.detach().cpu().contiguous()
    shape, dtype_str = infer_tensor_metadata(tensor_cpu)
    payload = tensor_cpu.numpy().tobytes(order="C")
    return payload, shape, dtype_str


def bytes_to_tensor(payload, shape, dtype_str, device="cpu"):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")

    np_dtype = dtype_str_to_numpy(dtype_str)
    expected_numel = int(np.prod(shape)) if len(shape) > 0 else 1
    expected_nbytes = expected_numel * np_dtype.itemsize
    actual_nbytes = len(payload)

    if actual_nbytes != expected_nbytes:
        raise ValueError(
            "Tensor payload size mismatch: expected {0} bytes for shape={1} "
            "dtype={2}, got {3}".format(
                expected_nbytes, list(shape), dtype_str, actual_nbytes
            )
        )

    array = np.frombuffer(payload, dtype=np_dtype).reshape(tuple(shape))
    tensor = torch.from_numpy(array.copy())
    return tensor.to(device)
