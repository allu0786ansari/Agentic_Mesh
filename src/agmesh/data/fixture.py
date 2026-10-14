"""A small, fully synthetic public fixture with the same layout as the real data.

CI runs the whole Week 2 pipeline on it, so no raw industrial data is ever needed
there (Implementation Plan, Week 2 task 7). Everything here is generated; numbers
computed from it are marked ``data_source = fixture`` and can never pass a gate.

    python -m agmesh.data.fixture --out fixtures/data

HAI part
    ``train1``-``train3`` and ``test1``-``test2`` at 1 Hz with a handful of
    features, including a training-constant one and a duplicate of ``P4_ST_PS``,
    and fourteen labelled attack events in the test files.
Modbus part
    Benign and attack captures for ``ied1a``, ``ied1b`` and ``scada-hmi``: the HMI
    polls each IED once per second over a new TCP connection; on the attack day an
    external attacker sends six episodes of requests to IED1A only.
"""

from __future__ import annotations

import argparse
import gzip
import socket
import struct
from pathlib import Path

import numpy as np
import pandas as pd

HMI, IED1A, IED1B, IED4C = "185.175.0.3", "185.175.0.4", "185.175.0.5", "185.175.0.8"
ATTACKER = "185.175.0.7"
DEVICE_IPS = {"ied1a": IED1A, "ied1b": IED1B, "scada-hmi": HMI}
SYN, ACK, PSH, RST = 0x02, 0x10, 0x08, 0x04


# ------------------------------------------------------------------------ packet bytes


def tcp_packet(
    src: str, dst: str, sport: int, dport: int, flags: int, payload: bytes = b""
) -> bytes:
    """Ethernet + IPv4 + TCP (no options) + payload, with correct length fields."""
    eth = b"\x02\x00\x00\x00\x00\x01\x02\x00\x00\x00\x00\x02\x08\x00"
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 0, 0, 5 << 4, flags, 65535, 0, 0)
    total = 20 + len(tcp) + len(payload)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total,
        0,
        0,
        64,
        6,
        0,
        socket.inet_aton(src),
        socket.inet_aton(dst),
    )
    return eth + ip + tcp + payload


def mbap(tid: int, fc: int, body: bytes, *, length: int | None = None, unit: int = 1) -> bytes:
    """One Modbus/TCP frame; ``length`` overrides the MBAP length field."""
    pdu = bytes([fc]) + body
    return struct.pack(">HHHB", tid, 0, len(pdu) + 1 if length is None else length, unit) + pdu


def read_request(tid: int, fc: int, addr: int, qty: int) -> bytes:
    return mbap(tid, fc, struct.pack(">HH", addr, qty))


def write_single(tid: int, fc: int, addr: int, value: int) -> bytes:
    return mbap(tid, fc, struct.pack(">HH", addr, value))


def read_response(tid: int, fc: int, n_bytes: int = 2) -> bytes:
    return mbap(tid, fc, bytes([n_bytes]) + b"\x00" * n_bytes)


def write_pcap(path: Path, packets: list[tuple[float, bytes]]) -> Path:
    """Little-endian microsecond pcap with Ethernet link type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bytearray(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    for ts, data in sorted(packets, key=lambda p: p[0]):
        seconds = int(ts)
        micros = round((ts - seconds) * 1e6)
        if micros == 1_000_000:
            seconds, micros = seconds + 1, 0
        out += struct.pack("<IIII", seconds, micros, len(data), len(data)) + data
    path.write_bytes(bytes(out))
    return path


# ------------------------------------------------------------------------------ Modbus


def _exchange(
    packets: dict[str, list[tuple[float, bytes]]],
    ts: float,
    client: str,
    server: str,
    port: int,
    request: bytes,
    response: bytes | None,
    latency: float = 0.002,
) -> None:
    """Append one connection (SYN, request, response, RST) to every capture that sees it."""
    seen = [d for d, ip in DEVICE_IPS.items() if ip in (client, server)]
    exchange = [(ts, tcp_packet(client, server, port, 502, SYN))]
    exchange.append((ts + 0.0005, tcp_packet(client, server, port, 502, PSH | ACK, request)))
    if response is not None:
        exchange.append(
            (ts + 0.0005 + latency, tcp_packet(server, client, 502, port, PSH | ACK, response))
        )
    exchange.append((ts + 0.01, tcp_packet(client, server, port, 502, RST)))
    for device in seen:
        packets[device].extend(exchange)


def _poll(
    packets: dict[str, list[tuple[float, bytes]]],
    start: float,
    seconds: int,
    rng: np.random.Generator,
    *,
    skip: tuple[float, float] | None = None,
) -> None:
    """The HMI polls each IED once per second; occasional operator writes."""
    counter = 0
    for second in range(seconds):
        t = start + second
        if skip and skip[0] <= t < skip[1]:
            continue
        for k, ied in enumerate((IED1A, IED1B, IED4C)):
            counter += 1
            port = 40000 + counter % 20000
            tid = (second * 3 + k) % 65536
            jitter = float(rng.uniform(0.0, 0.05))
            _exchange(
                packets,
                t + 0.1 * k + jitter,
                HMI,
                ied,
                port,
                read_request(tid, 3, 8, 1),
                read_response(tid, 3),
                latency=float(rng.uniform(0.0005, 0.003)),
            )
            if rng.random() < 0.01:
                counter += 1
                port = 40000 + counter % 20000
                _exchange(
                    packets,
                    t + 0.1 * k + 0.5,
                    HMI,
                    ied,
                    port,
                    write_single(tid, 6, 8, int(rng.integers(0, 100))),
                    write_single(tid, 6, 8, 0),
                )


def make_modbus(root: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    day = pd.Timestamp("2023-01-25T10:00:00Z").timestamp()
    for file_index in range(2):
        packets: dict[str, list[tuple[float, bytes]]] = {d: [] for d in DEVICE_IPS}
        start = day + file_index * 1500
        skip = (start + 600, start + 900) if file_index == 0 else None  # a 5-minute idle stretch
        _poll(packets, start, 1200, rng, skip=skip)
        for device, items in packets.items():
            write_pcap(root / "benign" / device / f"{device}-normal-{file_index}.pcap", items)

    attack_day = pd.Timestamp("2023-02-01T04:00:00Z").timestamp()
    packets = {d: [] for d in DEVICE_IPS}
    _poll(packets, attack_day, 1800, rng)
    for episode in range(6):
        begin = attack_day + 180 + episode * 240
        for i in range(5 + 5 * episode):
            port = 50000 + episode * 100 + i
            ts = begin + 2.0 * i
            if episode % 2 == 0:
                request = read_request(i, 3, 100 + i, 125)  # unseen address, oversized quantity
                response: bytes | None = read_response(i, 3, 250)
            else:
                request = write_single(0, 5, 200, 0xFF00)  # coil writes to an unseen address
                response = None
            if episode == 3 and i == 0:
                request = request + read_request(1, 4, 1, 1)  # two stacked frames
            _exchange(packets, ts, ATTACKER, IED1A, port, request, response)
    for device, items in packets.items():
        write_pcap(root / "attack" / device / f"{device}-0.pcap", items)


# --------------------------------------------------------------------------------- HAI


def _hai_frame(
    start: str, seconds: int, rng: np.random.Generator, offset: float = 1.2
) -> pd.DataFrame:
    t = np.arange(seconds)
    turbine = np.where((t // 1800) % 3 == 0, 50.0, 0.0)
    frame = pd.DataFrame(
        {
            "time": pd.date_range(start, periods=seconds, freq="s").strftime("%Y-%m-%d %H:%M:%S"),
            "P1_FT01": 10 + np.sin(t / 300) + rng.normal(0, 0.1, seconds),
            "P1_LIT01": 400 + 5 * np.cos(t / 900) + rng.normal(0, 0.5, seconds),
            "P1_PCV02D": np.full(seconds, 12.0),
            "P1_PCV02Z": offset + rng.normal(0, 0.002, seconds),  # calibration offset (HAI-8)
            "P2_SIT01": 750 + rng.normal(0, 2, seconds),
            "P3_LCP01D": 3000 + 10 * np.sin(t / 600) + rng.normal(0, 3, seconds),
            "P4_ST_PS": turbine,
            "P4_HT_PS": turbine,
        }
    )
    for col in ("attack", "attack_P1", "attack_P2", "attack_P3"):
        frame[col] = 0
    return frame


def make_hai(root: Path, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "train1": ("2020-07-11 00:00:00", 7200),
        "train2": ("2020-07-31 22:00:00", 7200),
        "train3": ("2020-08-04 22:00:00", 14400),
        "test1": ("2020-07-07 15:00:00", 5400),
        "test2": ("2020-07-09 15:00:00", 5400),
    }
    targets = [("P1",), ("P2",), ("P1", "P2"), ("P3",), ("P1",), ("P2",), ("P1", "P3")]
    events = {
        name: [
            (400 + 700 * i + 50 * k, 520 + 700 * i + 50 * k + 20 * (i % 3), targets[(i + k) % 7])
            for i in range(7)
        ]
        for k, name in enumerate(("test1", "test2"))
    }
    for name, (start, seconds) in files.items():
        frame = _hai_frame(start, seconds, rng, offset=1.2 if name in ("train1", "test1") else 0.4)
        for begin, stop, procs in events.get(name, []):
            frame.loc[begin : stop - 1, "attack"] = 1
            for proc in procs:
                frame.loc[begin : stop - 1, f"attack_{proc}"] = 1
                column = {"P1": "P1_FT01", "P2": "P2_SIT01", "P3": "P3_LCP01D"}[proc]
                frame.loc[begin : stop - 1, column] *= 1.3
            if "P1" in procs:
                frame.loc[begin : stop - 1, "P1_PCV02D"] = (
                    20.0  # training-constant variable changes
                )
        with gzip.open(root / f"{name}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)


def make_fixture(out: Path) -> dict[str, Path]:
    hai_dir = out / "hai" / "hai-fixture"
    modbus_dir = out / "modbus2023"
    make_hai(hai_dir)
    make_modbus(modbus_dir)
    return {"hai_dir": hai_dir, "modbus_dir": modbus_dir}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic Week 2 fixture")
    parser.add_argument("--out", type=Path, default=Path("fixtures/data"))
    args = parser.parse_args(argv)
    paths = make_fixture(args.out)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
