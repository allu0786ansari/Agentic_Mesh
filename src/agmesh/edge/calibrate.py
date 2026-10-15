"""Fit a node's local calibration for the frozen global model (ADR 0009 §2).

This runs on the node, with the node's own partitions, and uses the same code path
as the detection results:

* ``x1.Fitted.fit_reference`` and ``x1.Fitted.shifted``;
* ``fpr_threshold`` on scored normal validation windows;
* the isotonic calibrator on all scored validation windows.

It returns the batch window scores too, so the runtime can check that its streaming
scores agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agmesh.common.config import Params
from agmesh.common.provenance import utc_now_iso
from agmesh.data.access import Site
from agmesh.edge.bundle import GlobalModel, NodeCalibration, buckets_from
from agmesh.evaluation.calibration import IsotonicCalibrator
from agmesh.evaluation.thresholds import fpr_threshold
from agmesh.experiments import x1_baselines as x1
from agmesh.models import inputs as mi
from agmesh.models.shift import Shifted


@dataclass
class Calibrated:
    calibration: NodeCalibration
    scaler: mi.Scaler
    fitted: x1.Fitted
    batch: dict[str, Shifted]  # val_normal / val_attack scores from the batch path
    inputs: dict[str, mi.ModelInput]


def scaler_for(site: Site, model: GlobalModel, params: Params) -> mi.Scaler:
    return mi.Scaler(
        np.asarray(model.norm.centre), np.asarray(model.norm.scale), x1._clip(site, params)
    )


def calibrate_node(
    site: Site,
    node: str,
    model: GlobalModel,
    data: x1.NodeData,
    params: Params,
    constants: dict[str, float],
    net: Any | None = None,
) -> Calibrated:
    from agmesh.models import nets

    network = net if net is not None else model.network()
    scaler = scaler_for(site, model, params)
    fitted = x1.Fitted(lambda x: nets.reconstruction_errors(network, x), None, {})
    scoring = x1.site_scoring(site, params)
    shift = x1.site_shift(site, params)
    window = params.models.scoring.shift_window
    train = x1._inputs(site, data.parts["train"], scaler, params, evaluation=False)
    fitted.fit_reference([train], scoring)
    inputs = {
        p: x1._inputs(site, data.parts[p], scaler, params, evaluation=True)
        for p in ("val_normal", "val_attack")
    }
    batch = {p: fitted.shifted(inp, scoring, shift, window) for p, inp in inputs.items()}
    normal = batch["val_normal"]
    threshold = fpr_threshold(normal.scores[normal.scored], params.evaluation.target_window_fpr)
    attack = batch["val_attack"]
    cal_scores = np.concatenate([normal.scores[normal.scored], attack.scores[attack.scored]])
    cal_labels = np.concatenate(
        [
            np.zeros(int(normal.scored.sum())),
            inputs["val_attack"].label[attack.scored].astype(float),
        ]
    )
    calibrator = IsotonicCalibrator().fit(cal_scores, cal_labels)
    vn = inputs["val_normal"]
    feature_errors = mi.per_feature(fitted.errors(vn.x), vn)  # type: ignore[misc]
    names = list(vn.base_features)
    request_rate = None
    if site == "site_b":
        cols = list(data.parts["val_normal"].features.columns)
        recv = data.parts["val_normal"].features.values[:, cols.index("req_recv")].astype(float)
        request_rate = [float(np.quantile(recv, 0.01)), float(np.quantile(recv, 0.99))]
    ref = fitted.reference
    sref = fitted.shift_reference
    calibration = NodeCalibration(
        node=node,
        model_version=model.version,
        features=names,
        threshold=float(threshold.value),
        threshold_val_fpr=float(threshold.achieved_fpr),
        calibrator={k: list(v) for k, v in calibrator.to_dict().items()},
        error_median=ref.median.tolist() if ref is not None else None,
        error_iqr=ref.iqr.tolist() if ref is not None else None,
        score_median=float(sref.score_median) if sref is not None else 0.0,
        buckets=buckets_from(feature_errors, names),
        request_rate=request_rate,
        constants=dict(constants) if site == "site_a" else {},
        shift=shift,
        fitted_at=utc_now_iso(),
    )
    return Calibrated(calibration, scaler, fitted, batch, inputs)
