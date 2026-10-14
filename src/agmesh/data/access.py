"""Read one node's partition, with labels kept apart from features (CC1, WR-2).

:func:`load_partition` is the only supported way for models and experiments to read
partitions. It returns the features as a :class:`LabelFreeFrame` and the labels in a
separate :class:`Labels` object, and it refuses the test split unless given a
:class:`~agmesh.evaluation.sealed.FinalAccess`.

Site A rows are 1 Hz samples (windowing happens in the model input pipeline); Site B
rows are already 10-second windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from agmesh.common.config import Params
from agmesh.common.frames import LabelFreeFrame
from agmesh.data.manifest import PARTITIONS, read_manifest
from agmesh.data.site_b import FEATURES as SITE_B_FEATURES
from agmesh.evaluation.sealed import FinalAccess, authorise

Site = Literal["site_a", "site_b"]


@dataclass(frozen=True)
class Labels:
    target: npt.NDArray[np.int8]  # 1 = attack row or window
    time_s: npt.NDArray[np.float64]  # sample time (A) or window start (B), epoch seconds
    group: npt.NDArray[np.str_]  # run_id (A) or episode as text, "-1" = none (B)


@dataclass(frozen=True)
class Partition:
    site: Site
    client: str
    part: str
    features: LabelFreeFrame
    labels: Labels
    manifest_sha256: str


def _manifest_file_sha(root: Path, site: Site, client: str, part: str) -> str:
    manifest = read_manifest(root / site)
    sha: str = manifest["nodes"][client]["partitions"][part]["sha256"]
    return sha


def load_partition(
    params: Params,
    site: Site,
    client: str,
    part: str,
    *,
    root: Path | None = None,
    final: FinalAccess | None = None,
) -> Partition:
    if part not in PARTITIONS:
        raise ValueError(f"unknown partition {part!r}; expected one of {PARTITIONS}")
    if part == "test":
        authorise(final, site=site, client=client)
    base = root or params.paths.partitions_dir
    manifest = read_manifest(base / site)
    if client not in manifest["nodes"]:
        raise KeyError(f"{site} has no client {client!r}")
    frame = pd.read_parquet(base / site / client / f"{part}.parquet")
    sha = _manifest_file_sha(base, site, client, part)
    if site == "site_a":
        cfg = params.data.hai
        features = manifest["features"]["features"]
        forbidden = [cfg.label_column, *cfg.process_label_columns]
        time_s = frame["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
        labels = Labels(
            target=frame[cfg.label_column].to_numpy(dtype=np.int8),
            time_s=time_s.astype(np.float64),
            group=frame["run_id"].to_numpy(dtype=str),
        )
    else:
        features = list(SITE_B_FEATURES)
        forbidden = ["label", "episode", "attacker_packets"]
        labels = Labels(
            target=frame["label"].to_numpy(dtype=np.int8),
            time_s=frame["window_start"].to_numpy(dtype=np.float64),
            group=frame["episode"].astype(str).to_numpy(dtype=str),
        )
    return Partition(
        site=site,
        client=client,
        part=part,
        features=LabelFreeFrame.from_frame(frame, features, forbidden),
        labels=labels,
        manifest_sha256=sha,
    )
