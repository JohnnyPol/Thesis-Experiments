from __future__ import annotations

from typing import Optional


def read_network_bytes(interface: Optional[str] = None) -> dict:
    """
    Read transmitted/received bytes from /proc/net/dev.

    Args:
        interface: Specific interface name (e.g. 'eth0', 'wlan0').
                   If None, aggregate over all non-loopback interfaces.

    Returns:
        Dict with rx_bytes and tx_bytes.
    """
    rx_total = 0
    tx_total = 0

    with open("/proc/net/dev", "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]  # skip headers

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 17:
            continue

        iface = parts[0].replace(":", "")
        if iface == "lo":
            continue

        if interface is not None and iface != interface:
            continue

        rx_bytes = int(parts[1])
        tx_bytes = int(parts[9])

        rx_total += rx_bytes
        tx_total += tx_bytes

    return {
        "rx_bytes": rx_total,
        "tx_bytes": tx_total,
    }


def compute_network_delta(start_snapshot: dict, end_snapshot: dict) -> dict:
    """
    Compute network byte deltas.

    Args:
        start_snapshot: Dict returned by read_network_bytes().
        end_snapshot: Dict returned by read_network_bytes().

    Returns:
        Dict with rx/tx deltas.
    """
    rx_delta = end_snapshot["rx_bytes"] - start_snapshot["rx_bytes"]
    tx_delta = end_snapshot["tx_bytes"] - start_snapshot["tx_bytes"]

    return {
        "rx_bytes": int(rx_delta),
        "tx_bytes": int(tx_delta),
        "total_bytes": int(rx_delta + tx_delta),
    }