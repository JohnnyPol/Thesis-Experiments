from __future__ import annotations

import io
import pickle
import socket
import struct
from typing import Any

import torch


_LENGTH_STRUCT = struct.Struct("!Q")


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Serialize a tensor to bytes using torch.save for simplicity and fidelity.
    """
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def bytes_to_tensor(blob: bytes, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Deserialize a tensor from bytes.
    """
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location=device)


def object_to_bytes(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def bytes_to_object(blob: bytes) -> Any:
    return pickle.loads(blob)


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes

    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def send_message(sock: socket.socket, message: dict[str, Any]) -> int:
    """
    Send a length-prefixed pickled message.

    Returns:
        total bytes sent on the wire (header + payload)
    """
    payload = object_to_bytes(message)
    header = _LENGTH_STRUCT.pack(len(payload))
    sock.sendall(header)
    sock.sendall(payload)
    return len(header) + len(payload)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], int]:
    """
    Receive a length-prefixed pickled message.

    Returns:
        (message_dict, total_bytes_received)
    """
    header = _recv_exact(sock, _LENGTH_STRUCT.size)
    (payload_size,) = _LENGTH_STRUCT.unpack(header)
    payload = _recv_exact(sock, payload_size)
    message = bytes_to_object(payload)
    if not isinstance(message, dict):
        raise ValueError("Received payload is not a dict message")
    return message, len(header) + len(payload)


def roundtrip_request(
    host: str,
    port: int,
    request: dict[str, Any],
    timeout_sec: float = 30.0,
) -> tuple[dict[str, Any], int, int]:
    """
    Open a TCP connection, send one request, receive one response, and close.

    Returns:
        response, request_wire_bytes, response_wire_bytes
    """
    with socket.create_connection((host, port), timeout=timeout_sec) as sock:
        request_wire_bytes = send_message(sock, request)
        response, response_wire_bytes = recv_message(sock)

    return response, request_wire_bytes, response_wire_bytes