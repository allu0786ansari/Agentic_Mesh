"""A federated client: one node, its own partition, its own privacy budget.

The client loads **only its own partition** (``train``, ``val_normal`` and
``val_attack`` of one node) and nothing else. Every release goes through its budget
guard, and the guard's ledger lives on disk. A client object can therefore be
recreated every round, as Flower does, without resetting anything. With ε = ∞ there
is no budget: exact histograms and non-private training with the same steps.

Methods map one-to-one to the protocol (``protocol.py``):

* ``hello``      model spec and noisy normalisation histograms (round 0);
* ``fit``        local training for one round from the global weights;
* ``evaluate``   validation loss (each round), or every metric (at the end);
* ``threshold``  the noisy normal-score histogram for H1c.
"""

from __future__ import annotations

import functools
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.config import Params
from agmesh.data.access import Site, load_partition
from agmesh.evaluation.detection import DETECTION_KEYS, node_arrays_npz
from agmesh.experiments import x1_baselines as x1
from agmesh.federated import fed_eval
from agmesh.federated.protocol import (
    EvalInstruction,
    EvalReply,
    FitInstruction,
    FitReply,
    HelloReply,
    ModelSpec,
    Normalisation,
    ThresholdReply,
    Weights,
)
from agmesh.models import inputs as mi
from agmesh.privacy import bounds, client_budget, noisy_stats
from agmesh.privacy.accountant_store import ROOT
from agmesh.privacy.budget_guard import BudgetExhausted
from agmesh.privacy.units import n_training_units, training_units

PARTS = ("train", "val_normal", "val_attack")


@functools.lru_cache(maxsize=16)
def _node_data(params_json: str, site: Site, node: str, root: str) -> x1.NodeData:
    params = Params.model_validate_json(params_json)
    return x1.NodeData({p: load_partition(params, site, node, p, root=Path(root)) for p in PARTS})


def get_weights(model: Any) -> Weights:
    return [v.detach().cpu().numpy().astype(np.float32).copy() for v in model.state_dict().values()]


def set_weights(model: Any, weights: Weights) -> None:
    import torch

    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError("weight list does not match the model")
    model.load_state_dict(
        {k: torch.from_numpy(np.asarray(w)) for k, w in zip(keys, weights, strict=True)}
    )


#: L2 sensitivity of the H1c histogram (ADR 0007). The histogram counts R0 scores of
#: non-overlapping normal-validation windows. Each window is one unit, and its R0 score
#: depends only on its own rows and the (DP) model, so adding or removing a unit changes
#: one count by one.
H1C_SENSITIVITY = 1.0


def check_h1c(params: Params, site: str) -> None:
    """H1c needs a score that uses no training statistic; R0 with ``mean`` qualifies."""
    cfg = params.models.scoring.site_a if site == "site_a" else params.models.scoring.site_b
    if cfg.aggregation != "mean" or cfg.smooth != 1:
        raise ValueError(
            "H1c needs the 'mean' aggregation without smoothing: a score normalised by "
            "training errors would make the histogram depend on every training window"
        )


class FederatedClient:
    def __init__(
        self,
        params: Params,
        site: Site,
        node: str,
        *,
        partitions: Path,
        run_id: str,
        epsilon: float,
        seed: int,
        node_index: int,
        h1c_epsilon: float | None = None,
        epochs: int | None = None,
        privacy_root: Path = ROOT,
    ) -> None:
        self.params, self.site, self.node = params, site, node
        self.seed, self.node_index = seed, node_index
        self.data = _node_data(params.model_dump_json(), site, node, str(partitions))
        train = self.data.parts["train"]
        self.n_windows = n_training_units(site, train, params)
        cols = (
            mi.site_a_columns(train, params)
            if site == "site_a"
            else mi.site_b_columns(train, params)
        )
        self.names = [train.features.columns[i] for i in cols]
        self.private = math.isfinite(epsilon)
        self.budget = (
            client_budget.prepare(
                params,
                site=site,
                node=node,
                n_windows=self.n_windows,
                run_id=run_id,
                target_epsilon=epsilon,
                threshold_epsilon=h1c_epsilon,
                epochs=epochs,
                root=privacy_root,
            )
            if self.private
            else None
        )
        self.sample_rate, _ = client_budget.calibrate_steps(params, site, self.n_windows)
        self.per_epoch = max(1, round(1.0 / self.sample_rate))

    # ------------------------------------------------------------------ helpers
    @property
    def kind(self) -> str:
        return "vae" if self.site == "site_a" else "ae"

    def _cfg(self) -> Any:
        return self.params.models.vae if self.site == "site_a" else self.params.models.ae

    def _scaler(self, norm: Normalisation) -> mi.Scaler:
        return mi.Scaler(
            np.asarray(norm.centre), np.asarray(norm.scale), x1._clip(self.site, self.params)
        )

    def _train_inputs(self, norm: Normalisation) -> mi.ModelInput:
        return x1._inputs(
            self.site, self.data.parts["train"], self._scaler(norm), self.params, evaluation=False
        )

    def _model(self, spec: ModelSpec, weights: Weights | None) -> Any:
        from agmesh.models import nets

        model = (
            nets.build("vae", spec.n_inputs, list(spec.hidden), spec.latent, seed=self.seed)
            if spec.kind == "vae"
            else nets.build("ae", spec.n_inputs, list(spec.hidden), seed=self.seed)
        )
        if weights is not None:
            set_weights(model, weights)
        return model

    def spec(self) -> ModelSpec:
        cfg = self._cfg()
        dims = len(self.names)
        segments = self.params.models.vae.segments_per_window if self.site == "site_a" else 1
        if self.site == "site_a" and self.params.models.vae.input_mode == "flat":
            segments = self.params.data.windows.site_a.length_s
        latent = self.params.models.vae.latent if self.site == "site_a" else 0
        return ModelSpec(self.kind, dims * segments, tuple(cfg.hidden), latent)

    # ------------------------------------------------------------------ protocol
    def hello(self) -> HelloReply:
        lo, hi = bounds.bounds_for(self.site, self.names)
        rng = np.random.default_rng([self.seed, self.node_index, 0])
        z = 0.0
        if self.budget is not None:
            z = self.budget.stats_noise_multiplier
            self.budget.guard.spend(noisy_stats.mechanism(z))  # accounted before release
        units, _names = training_units(self.site, self.data.parts["train"], self.params)
        hist = noisy_stats.release(
            units,
            lo,
            hi,
            bins=getattr(self.params.privacy.normalisation_bins, self.site),
            noise_multiplier=z,
            rng=rng,
            threshold_sd=self.params.privacy.normalisation_threshold_sd,
        )
        return HelloReply(self.node, self.spec(), lo, hi, hist.counts, z, hist.threshold_sd)

    def fit(self, ins: FitInstruction) -> FitReply:
        from agmesh.models import dp_train

        started = time.perf_counter()
        model = self._model(self.spec(), ins.weights)
        x = self._train_inputs(ins.norm).x
        steps = ins.local_epochs * self.per_epoch
        cfg = self._cfg()
        kl = self.params.models.vae.kl_weight
        try:
            if self.budget is not None:
                result = dp_train.fit_private(
                    model,
                    x,
                    plan=self.budget.plan,
                    guard=self.budget.guard,
                    lr=self.params.privacy.dp_lr.get(self.site, cfg.lr),
                    seed=ins.seed,
                    kl_weight=kl,
                    steps=steps,
                    proximal_mu=ins.proximal_mu,
                )
            else:
                result = dp_train.fit_reference(
                    model,
                    x,
                    steps=steps,
                    sample_rate=self.sample_rate,
                    lr=cfg.lr,
                    seed=ins.seed,
                    kl_weight=kl,
                    proximal_mu=ins.proximal_mu,
                )
        except BudgetExhausted as exc:
            return FitReply(
                self.node,
                ins.round,
                None,
                f"budget exhausted: {exc}",
                None,
                self.budget.store.epsilon() if self.budget else None,
                0,
                time.perf_counter() - started,
            )
        loss = result.history[-1]["train_loss"] if result.history else None
        return FitReply(
            self.node,
            ins.round,
            get_weights(model),
            None,
            loss,
            self.budget.store.epsilon() if self.budget else math.inf,
            steps,
            time.perf_counter() - started,
        )

    def _fitted(self, model: Any, norm: Normalisation) -> x1.Fitted:
        from agmesh.models import nets

        fitted = x1.Fitted(lambda x: nets.reconstruction_errors(model, x), None, {})
        fitted.fit_reference([self._train_inputs(norm)], x1.site_scoring(self.site, self.params))
        return fitted

    def evaluate(self, ins: EvalInstruction) -> EvalReply:
        from agmesh.models import nets

        model = self._model(self.spec(), ins.weights)
        scaler = self._scaler(ins.norm)
        vn = x1._inputs(
            self.site, self.data.parts["val_normal"], scaler, self.params, evaluation=False
        )
        val_loss = (
            float(nets.reconstruction_errors(model, vn.x).mean()) if len(vn) else float("nan")
        )
        if not ins.full:
            return EvalReply(self.node, val_loss)
        fitted = self._fitted(model, ins.norm)
        fit = x1.NodeFit(self.node, self.data, scaler, fitted)
        metrics, arrays = x1._evaluate(
            self.site, fit, self.params, shift=ins.shift, threshold_value=ins.threshold
        )
        normal = arrays.scores[arrays.scored & ~arrays.labels]
        # abstained attack windows count as missed: they rank lowest (bottom bin)
        attack = np.where(arrays.scored, arrays.scores, -1e30)[arrays.labels]
        return EvalReply(
            self.node,
            val_loss,
            {k: metrics.get(k) for k in (*DETECTION_KEYS, "threshold", "threshold_val_fpr")},
            fed_eval.bin_scores(normal),
            fed_eval.bin_scores(attack),
            node_arrays_npz({self.node: arrays}),
        )

    def threshold(self, ins: EvalInstruction) -> ThresholdReply:
        """H1c: Gaussian-noised histogram of R0 scores of non-overlapping normal windows."""
        check_h1c(self.params, self.site)
        if self.budget is not None and self.budget.threshold is None:
            raise ValueError("this run did not reserve budget for the H1c histogram")
        model = self._model(self.spec(), ins.weights)
        scaler = self._scaler(ins.norm)
        fitted = self._fitted(model, ins.norm)
        vn = x1._inputs(
            self.site, self.data.parts["val_normal"], scaler, self.params, evaluation=False
        )
        scores = fitted.shifted(
            vn,
            x1.site_scoring(self.site, self.params),
            "r0",
            self.params.models.scoring.shift_window,
        ).scores
        counts = fed_eval.coarsen(fed_eval.bin_scores(scores), self.params.privacy.threshold_bins)
        noise_sd = 0.0
        if self.budget is not None and self.budget.threshold is not None:
            try:
                self.budget.guard.release_reserved(client_budget.THRESHOLD)
            except (BudgetExhausted, KeyError) as exc:
                return ThresholdReply(self.node, None, 0.0, f"declined: {exc}")
            noise_sd = self.budget.threshold.noise_multiplier * H1C_SENSITIVITY
            rng = np.random.default_rng([self.seed, self.node_index, 1])
            counts = counts + rng.normal(0.0, noise_sd, counts.shape)
        return ThresholdReply(self.node, counts, noise_sd)

    def privacy(self) -> dict[str, Any]:
        if self.budget is None:
            return {"epsilon": math.inf, "private": False}
        return {
            **self.budget.store.summary(),
            "stats_noise_multiplier": self.budget.stats_noise_multiplier,
        }
