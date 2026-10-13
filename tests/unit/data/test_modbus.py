"""Modbus/TCP decoding, request/response pairing and attacker-address capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agmesh.data.fixture import (
    ACK,
    ATTACKER,
    HMI,
    IED1A,
    PSH,
    RST,
    SYN,
    mbap,
    read_request,
    read_response,
    tcp_packet,
    write_pcap,
    write_single,
)
from agmesh.data.modbus import add_latency, int_to_ip, ip_to_int, parse_capture


def test_ip_round_trip() -> None:
    assert int_to_ip(ip_to_int("185.175.0.7")) == "185.175.0.7"


def test_parse_capture_decodes_frames_and_flags(tmp_path: Path) -> None:
    stacked = read_request(7, 4, 1, 1) + read_request(8, 4, 2, 1)
    bad_length = mbap(9, 3, b"\x00\x01\x00\x01", length=40)  # MBAP length beyond the payload
    packets = [
        (100.0, tcp_packet(HMI, IED1A, 40001, 502, SYN)),
        (100.001, tcp_packet(HMI, IED1A, 40001, 502, PSH | ACK, read_request(1, 3, 8, 1))),
        (100.004, tcp_packet(IED1A, HMI, 502, 40001, PSH | ACK, read_response(1, 3))),
        (100.010, tcp_packet(HMI, IED1A, 40001, 502, RST)),
        (101.0, tcp_packet(ATTACKER, IED1A, 50000, 502, PSH | ACK, stacked)),
        (
            102.0,
            tcp_packet(ATTACKER, IED1A, 50001, 502, PSH | ACK, write_single(0, 5, 200, 0xFF00)),
        ),
        (103.0, tcp_packet(ATTACKER, IED1A, 50002, 502, PSH | ACK, bad_length)),
        (104.0, tcp_packet(IED1A, ATTACKER, 502, 50001, PSH | ACK, mbap(0, 0x85, b"\x02"))),
    ]
    tables = parse_capture(write_pcap(tmp_path / "a.pcap", packets), ATTACKER)
    f = tables.frames
    assert tables.packets == 8
    assert len(tables.watched) == 4
    assert len(tables.tcp) == 2 and tables.tcp["rst"].tolist() == [False, True]
    assert len(f) == 7
    first = f.iloc[0]
    assert first["is_request"] and first["fc"] == 3 and first["addr"] == 8 and first["qty"] == 1
    assert abs(first["latency_s"] - 0.003) < 1e-6
    stacked_rows = f[f["frames_in_packet"] == 2]
    assert stacked_rows["addr"].tolist() == [1, 2]
    coil = f[(f["fc"] == 5)].iloc[0]
    assert coil["addr"] == 200 and coil["qty"] == 1
    assert f[f["tid"] == 9].iloc[0]["truncated"]
    exception = f[f["fc"] == 0x85].iloc[0]
    assert not exception["is_request"] and exception["exception"] == 2
    # the exception reply pairs with the coil write on the same connection
    assert abs(coil["latency_s"] - 2.0) < 1e-6


def test_latency_pairs_repeated_transaction_ids_in_order() -> None:
    rows = []
    for i in range(3):
        rows.append((10.0 + i, 7, 4, 50000, 502, True, 0))
        rows.append((10.5 + i, 4, 7, 502, 50000, False, 0))
    frames = pd.DataFrame(rows, columns=["ts", "src", "dst", "sport", "dport", "is_request", "tid"])
    out = add_latency(frames)
    assert np.allclose(out.loc[out["is_request"], "latency_s"].to_numpy(), [0.5, 0.5, 0.5])
    assert out.loc[~out["is_request"], "latency_s"].isna().all()
