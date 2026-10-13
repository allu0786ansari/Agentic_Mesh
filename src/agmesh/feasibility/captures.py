"""Read the time span and packet count of pcap and pcapng captures without external tools.

Gate G0 needs to know whether the CIC Modbus device captures overlap in time
(Proposal §4.1, flaw R1). Only per-packet timestamps are needed, so the files are
memory-mapped and record headers are walked directly; packet payloads are never
parsed.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
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


def _pcap_span(buf: mmap.mmap | bytes, endian: str, resolution: float) -> CaptureSpan:
    header = struct.Struct(endian + "IIII")
    offset, size = 24, len(buf)
    n, first, last, truncated = 0, None, None, False
    while offset + 16 <= size:
        ts_sec, ts_frac, incl_len, _orig = header.unpack_from(buf, offset)
        ts = ts_sec + ts_frac * resolution
        first = ts if first is None else min(first, ts)
        last = ts if last is None else max(last, ts)
        n += 1
        offset += 16 + incl_len
    if offset != size:
        truncated = True
    return CaptureSpan("pcap", n, first, last, truncated)


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


def _pcapng_span(buf: mmap.mmap | bytes) -> CaptureSpan:
    size = len(buf)
    offset = 0
    endian = "<"
    resolutions: list[float] = []
    n, first, last, truncated = 0, None, None, False
    while offset + 12 <= size:
        block_type = struct.unpack_from(endian + "I", buf, offset)[0]
        if block_type == PCAPNG_SHB:
            magic_le = struct.unpack_from("<I", buf, offset + 8)[0]
            endian = "<" if magic_le == PCAPNG_BYTE_ORDER_MAGIC else ">"
            resolutions = []  # interface IDs restart in each section
        block_len = struct.unpack_from(endian + "I", buf, offset + 4)[0]
        if block_len < 12 or offset + block_len > size:
            truncated = True
            break
        if block_type == PCAPNG_IDB:
            options = bytes(buf[offset + 16 : offset + block_len - 4])
            resolutions.append(_tsresol(options, endian))
        elif block_type == PCAPNG_EPB:
            iface, ts_high, ts_low = struct.unpack_from(endian + "III", buf, offset + 8)
            resolution = resolutions[iface] if iface < len(resolutions) else 1e-6
            ts = ((ts_high << 32) | ts_low) * resolution
            first = ts if first is None else min(first, ts)
            last = ts if last is None else max(last, ts)
            n += 1
        offset += block_len
    return CaptureSpan("pcapng", n, first, last, truncated)


def capture_span(path: Path) -> CaptureSpan:
    """Return the format, packet count and first/last timestamp of a capture file."""
    with path.open("rb") as handle:
        magic = handle.read(4)
        if len(magic) < 4:
            raise CaptureFormatError(f"{path} is too small to be a capture")
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            if magic in PCAP_MAGICS:
                endian, resolution = PCAP_MAGICS[magic]
                return _pcap_span(buf, endian, resolution)
            if struct.unpack("<I", magic)[0] == PCAPNG_SHB:
                return _pcapng_span(buf)
    raise CaptureFormatError(f"{path} is not a pcap or pcapng capture")


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
