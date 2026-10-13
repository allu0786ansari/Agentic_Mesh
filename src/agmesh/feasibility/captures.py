"""Read pcap and pcapng captures without external tools.

Two jobs: the time span and packet count of a capture, and which IPv4 hosts it
contains (used to find the external attacker's traffic, the label source for the
CIC Modbus attack day).

Files are memory-mapped and record headers are walked directly. Only link-layer
and IPv4 headers are read; payloads are never parsed. Modbus decoding for the
feature pipeline arrives in Week 2.
"""

from __future__ import annotations

import mmap
import socket
import struct
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),  # little-endian, microseconds
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),  # big-endian, microseconds
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),  # little-endian, nanoseconds
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),  # big-endian, nanoseconds
}
PCAPNG_SHB = 0x0A0D0D0A
PCAPNG_BYTE_ORDER_MAGIC = 0x1A2B3C4D
PCAPNG_IDB = 0x00000001
PCAPNG_EPB = 0x00000006
PCAPNG_OPT_TSRESOL = 9


class CaptureFormatError(ValueError):
    """Raised for files that are not readable pcap or pcapng captures."""


@dataclass(frozen=True)
class CaptureSpan:
    fmt: str
    n_packets: int
    first_ts: float | None
    last_ts: float | None
    truncated: bool = False

    @property
    def duration_s(self) -> float | None:
        if self.first_ts is None or self.last_ts is None:
            return None
        return self.last_ts - self.first_ts


LINKTYPE_ETHERNET = 1
LINKTYPE_LINUX_SLL = 113
ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_VLAN = 0x8100


@dataclass
class _WalkState:
    truncated: bool = False


Packet = tuple[float, int, int, int]  # timestamp, data offset, captured length, link type


def _walk_pcap(
    buf: mmap.mmap | bytes, endian: str, resolution: float, state: _WalkState
) -> Iterator[Packet]:
    header = struct.Struct(endian + "IIII")
    linktype = struct.unpack_from(endian + "I", buf, 20)[0] & 0xFFFF
    offset, size = 24, len(buf)
    while offset + 16 <= size:
        ts_sec, ts_frac, incl_len, _orig = header.unpack_from(buf, offset)
        if offset + 16 + incl_len > size:
            state.truncated = True
        captured = min(incl_len, size - offset - 16)
        yield ts_sec + ts_frac * resolution, offset + 16, captured, linktype
        offset += 16 + incl_len
    if offset != size:
        state.truncated = True


def _tsresol(options: bytes, endian: str) -> float:
    pos = 0
    while pos + 4 <= len(options):
        code, length = struct.unpack_from(endian + "HH", options, pos)
        if code == 0:
            break
        value = options[pos + 4 : pos + 4 + length]
        if code == PCAPNG_OPT_TSRESOL and length >= 1:
            raw = value[0]
            return 2.0 ** -(raw & 0x7F) if raw & 0x80 else 10.0**-raw
        pos += 4 + length + ((4 - length % 4) % 4)
    return 1e-6


def _walk_pcapng(buf: mmap.mmap | bytes, state: _WalkState) -> Iterator[Packet]:
    size = len(buf)
    offset = 0
    endian = "<"
    interfaces: list[tuple[int, float]] = []  # (link type, timestamp resolution)
    while offset + 12 <= size:
        block_type = struct.unpack_from(endian + "I", buf, offset)[0]
        if block_type == PCAPNG_SHB:
            magic_le = struct.unpack_from("<I", buf, offset + 8)[0]
            endian = "<" if magic_le == PCAPNG_BYTE_ORDER_MAGIC else ">"
            interfaces = []  # interface IDs restart in each section
        block_len = struct.unpack_from(endian + "I", buf, offset + 4)[0]
        if block_len < 12 or offset + block_len > size:
            state.truncated = True
            return
        if block_type == PCAPNG_IDB:
            linktype = struct.unpack_from(endian + "H", buf, offset + 8)[0]
            options = bytes(buf[offset + 16 : offset + block_len - 4])
            interfaces.append((linktype, _tsresol(options, endian)))
        elif block_type == PCAPNG_EPB:
            iface, ts_high, ts_low, cap_len = struct.unpack_from(endian + "IIII", buf, offset + 8)
            linktype, resolution = interfaces[iface] if iface < len(interfaces) else (0, 1e-6)
            ts = ((ts_high << 32) | ts_low) * resolution
            yield ts, offset + 28, min(cap_len, block_len - 32), linktype
        offset += block_len


@contextmanager
def _open_packets(path: Path) -> Iterator[tuple[str, Iterator[Packet], _WalkState, mmap.mmap]]:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if len(magic) < 4:
            raise CaptureFormatError(f"{path} is too small to be a capture")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            state = _WalkState()
            if magic in PCAP_MAGICS:
                endian, resolution = PCAP_MAGICS[magic]
                yield "pcap", _walk_pcap(buf, endian, resolution, state), state, buf
                return
            if struct.unpack("<I", magic)[0] == PCAPNG_SHB:
                yield "pcapng", _walk_pcapng(buf, state), state, buf
                return
    raise CaptureFormatError(f"{path} is not a pcap or pcapng capture")


def capture_span(path: Path) -> CaptureSpan:
    """Return the format, packet count and first/last timestamp of a capture file."""
    with _open_packets(path) as (fmt, packets, state, _buf):
        n, first, last = 0, None, None
        for ts, _off, _len, _link in packets:
            first = ts if first is None else min(first, ts)
            last = ts if last is None else max(last, ts)
            n += 1
        return CaptureSpan(fmt, n, first, last, state.truncated)


def _ipv4_offset(buf: mmap.mmap, offset: int, length: int, linktype: int) -> int | None:
    """Offset of the IPv4 header inside a captured frame, or None if it is not IPv4."""
    if linktype == LINKTYPE_ETHERNET:
        if length < 14:
            return None
        ethertype = int.from_bytes(buf[offset + 12 : offset + 14], "big")
        start = offset + 14
        if ethertype == ETHERTYPE_VLAN and length >= 18:
            ethertype = int.from_bytes(buf[offset + 16 : offset + 18], "big")
            start += 4
    elif linktype == LINKTYPE_LINUX_SLL:
        if length < 16:
            return None
        ethertype = int.from_bytes(buf[offset + 14 : offset + 16], "big")
        start = offset + 16
    else:
        return None
    if ethertype != ETHERTYPE_IPV4 or start + 20 > offset + length:
        return None
    return start


@dataclass
class HostScan:
    """IPv4 hosts seen in a capture, and when a watched host was active."""

    hosts: Counter[str] = field(default_factory=Counter)
    watched_packets: int = 0
    watched_timestamps: list[float] = field(default_factory=list)
    first_ts: float | None = None

    def watched_windows(self, window_s: float) -> int:
        """Number of distinct windows (from the capture's first packet) with watched traffic."""
        if self.first_ts is None or not self.watched_timestamps:
            return 0
        origin = self.first_ts
        return len({int((ts - origin) // window_s) for ts in self.watched_timestamps})


def scan_hosts(path: Path, watch_ip: str | None = None) -> HostScan:
    """Count IPv4 packets per host and record when ``watch_ip`` sends or receives traffic."""
    watch = socket.inet_aton(watch_ip) if watch_ip else None
    scan = HostScan()
    with _open_packets(path) as (_fmt, packets, _state, buf):
        for ts, offset, length, linktype in packets:
            scan.first_ts = ts if scan.first_ts is None else min(scan.first_ts, ts)
            ip = _ipv4_offset(buf, offset, length, linktype)
            if ip is None:
                continue
            src = buf[ip + 12 : ip + 16]
            dst = buf[ip + 16 : ip + 20]
            scan.hosts[socket.inet_ntoa(src)] += 1
            scan.hosts[socket.inet_ntoa(dst)] += 1
            if watch is not None and (src == watch or dst == watch):
                scan.watched_packets += 1
                scan.watched_timestamps.append(ts)
    return scan


Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Intersection of two lists of merged intervals."""
    result: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def total_length(intervals: list[Interval]) -> float:
    return sum(end - start for start, end in intervals)
