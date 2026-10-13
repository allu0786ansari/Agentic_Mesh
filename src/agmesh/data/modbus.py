"""Parse CIC Modbus 2023 captures into compact tables (DVC stage ``prepare``).

Captures are memory-mapped and walked with :mod:`agmesh.feasibility.captures`; IPv4,
TCP and Modbus/TCP (MBAP) headers are decoded here directly, which is several
times faster than building per-packet objects and keeps memory flat on the
50-400 MB files. Three tables come out of every capture:

``frames``
    One row per Modbus/TCP frame on port 502 (a TCP payload can hold several
    stacked frames): time, endpoints, transaction ID, function code, address,
    quantity, exception code and frame-level anomalies. Requests carry the
    latency to their matching response.
``tcp``
    Connection openings (SYN without ACK) and resets.
``watched``
    Times of IPv4 packets to or from the watched address (the external attacker),
    which is the Site B label source (G0 decision 4).

Addresses are stored as unsigned 32-bit integers; :func:`ip_to_int` converts.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.feasibility.captures import _ipv4_offset, _open_packets

MODBUS_PORT = 502
TCP = 6
SYN, RST, ACK = 0x02, 0x04, 0x10
READ_FCS = (1, 2, 3, 4)
WRITE_FCS = (5, 6, 15, 16)
ADDRESS_FCS = (*READ_FCS, *WRITE_FCS)
MAX_LATENCY_S = 120.0

FRAME_COLUMNS = (
    "ts",
    "src",
    "dst",
    "sport",
    "dport",
    "is_request",
    "tid",
    "unit",
    "fc",
    "addr",
    "qty",
    "exception",
    "frame_index",
    "frames_in_packet",
    "len_mismatch",
    "truncated",
)
_unpack_ip = struct.Struct(">BBHHHBBH4s4s").unpack_from
_unpack_tcp = struct.Struct(">HHIIBB").unpack_from
_unpack_mbap = struct.Struct(">HHHBB").unpack_from
_unpack_hh = struct.Struct(">HH").unpack_from


def ip_to_int(address: str) -> int:
    return int.from_bytes(socket.inet_aton(address), "big")


def int_to_ip(value: int) -> str:
    return socket.inet_ntoa(int(value).to_bytes(4, "big"))


@dataclass
class CaptureTables:
    frames: pd.DataFrame
    tcp: pd.DataFrame
    watched: pd.DataFrame
    packets: int
    first_ts: float | None
    last_ts: float | None

    def span(self) -> dict[str, Any]:
        return {"packets": self.packets, "first_ts": self.first_ts, "last_ts": self.last_ts}


def _walk_frames(payload: bytes, rows: list[tuple[Any, ...]], head: tuple[Any, ...]) -> None:
    """Append one row per MBAP frame in ``payload``; ``head`` holds the packet fields."""
    is_request = head[5]
    n = len(payload)
    offset = 0
    parsed: list[tuple[Any, ...]] = []
    while offset + 8 <= n:
        tid, pid, length, unit, fc = _unpack_mbap(payload, offset)
        available = n - (offset + 6)
        pdu = payload[offset + 8 : offset + 6 + length] if length >= 2 else b""
        addr = qty = exception = -1
        if fc >= 0x80:
            exception = pdu[0] if pdu else -1
        elif fc in ADDRESS_FCS and len(pdu) >= 4:
            first, second = _unpack_hh(pdu, 0)
            if fc in (5, 6):
                addr, qty = first, 1
            elif is_request:
                addr, qty = first, second
        parsed.append(
            (tid, unit, fc, addr, qty, exception, length != available, length > available)
        )
        if pid != 0 or length < 2:
            break  # not Modbus, or a degenerate header: stop walking this payload
        offset += 6 + length
    for index, (tid, unit, fc, addr, qty, exc, mismatch, truncated) in enumerate(parsed):
        rows.append(
            (
                *head,
                tid,
                unit,
                fc,
                addr,
                qty,
                exc,
                index,
                len(parsed),
                mismatch if index == 0 else False,
                truncated,
            )
        )


def parse_capture(path: Path, watch_ip: str | None = None) -> CaptureTables:
    """Decode one pcap or pcapng capture (see module docstring)."""
    watch = socket.inet_aton(watch_ip) if watch_ip else None
    frame_rows: list[tuple[Any, ...]] = []
    tcp_rows: list[tuple[float, int, int, int]] = []
    watched: list[float] = []
    packets = 0
    first: float | None = None
    last: float | None = None
    with _open_packets(path) as (_fmt, walker, _state, buf):
        for ts, offset, length, linktype in walker:
            packets += 1
            if first is None or ts < first:
                first = ts
            if last is None or ts > last:
                last = ts
            ip = _ipv4_offset(buf, offset, length, linktype)
            if ip is None:
                continue
            end = offset + length
            ver_ihl, _tos, total, _id, _frag, _ttl, proto, _sum, src_b, dst_b = _unpack_ip(buf, ip)
            if watch is not None and (src_b == watch or dst_b == watch):
                watched.append(ts)
            if proto != TCP:
                continue
            tcp = ip + (ver_ihl & 0x0F) * 4
            if tcp + 20 > end:
                continue
            sport, dport, _seq, _ack, data_offset, flags = _unpack_tcp(buf, tcp)
            src = int.from_bytes(src_b, "big")
            dst = int.from_bytes(dst_b, "big")
            if flags & RST:
                tcp_rows.append((ts, src, dst, 1))
            elif flags & SYN and not flags & ACK:
                tcp_rows.append((ts, src, dst, 0))
            if MODBUS_PORT not in (sport, dport):
                continue
            payload_start = tcp + (data_offset >> 4) * 4
            payload_end = min(end, ip + total) if total else end
            if payload_end - payload_start < 8:
                continue
            payload = bytes(buf[payload_start:payload_end])
            head = (ts, src, dst, sport, dport, dport == MODBUS_PORT)
            _walk_frames(payload, frame_rows, head)
    frames = pd.DataFrame(frame_rows, columns=list(FRAME_COLUMNS))
    frames = frames.astype(
        {
            "ts": "float64",
            "src": "uint32",
            "dst": "uint32",
            "sport": "uint16",
            "dport": "uint16",
            "is_request": "bool",
            "tid": "uint16",
            "unit": "uint8",
            "fc": "uint8",
            "addr": "int32",
            "qty": "int32",
            "exception": "int16",
            "frame_index": "uint8",
            "frames_in_packet": "uint8",
            "len_mismatch": "bool",
            "truncated": "bool",
        }
    )
    frames = add_latency(frames)
    tcp_events = pd.DataFrame(tcp_rows, columns=["ts", "src", "dst", "rst"]).astype(
        {"ts": "float64", "src": "uint32", "dst": "uint32", "rst": "bool"}
    )
    watched_frame = pd.DataFrame({"ts": np.asarray(watched, dtype=np.float64)})
    return CaptureTables(frames, tcp_events, watched_frame, packets, first, last)


def add_latency(frames: pd.DataFrame) -> pd.DataFrame:
    """Pair each request with its response and store the latency on the request row.

    The key is (client, client port, server, transaction ID); when a key repeats
    (an attacker reusing transaction ID 0 on one connection), the n-th request is
    paired with the n-th response. Pairs slower than 120 s are not trusted.
    """
    out = frames.copy()
    out["latency_s"] = np.nan
    if out.empty:
        return out
    req = out.loc[out["is_request"], ["ts", "src", "sport", "dst", "tid"]]
    rsp = out.loc[~out["is_request"], ["ts", "dst", "dport", "src", "tid"]]
    rsp = rsp.rename(columns={"dst": "client", "dport": "cport", "src": "server"})
    req = req.rename(columns={"src": "client", "sport": "cport", "dst": "server"})
    keys = ["client", "cport", "server", "tid"]
    req = req.sort_values("ts", kind="stable")
    rsp = rsp.sort_values("ts", kind="stable")
    req["n"] = req.groupby(keys).cumcount()
    rsp["n"] = rsp.groupby(keys).cumcount()
    paired = req.reset_index().merge(rsp, on=[*keys, "n"], suffixes=("", "_rsp"))
    latency = paired["ts_rsp"] - paired["ts"]
    good = (latency >= 0) & (latency <= MAX_LATENCY_S)
    out.loc[paired.loc[good, "index"].to_numpy(), "latency_s"] = latency[good].to_numpy()
    return out


def capture_files(directory: Path) -> list[Path]:
    """Capture files in a device folder, ordered by their first packet time."""
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in (".pcap", ".pcapng", ".cap")
    ]
    return sorted(files, key=lambda p: (_first_ts(p), p.name))


def _first_ts(path: Path) -> float:
    with _open_packets(path) as (_fmt, walker, _state, _buf):
        for ts, *_rest in walker:
            return float(ts)
    return float("inf")
