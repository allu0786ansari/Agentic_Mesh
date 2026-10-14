"""Typed loader for ``configs/params.yaml`` (the single source of truth, WR-8).

Every section is validated on load so that inconsistent settings (for example a
minimum client count larger than the federation, or overlapping training windows)
fail immediately instead of surfacing weeks later as a wrong result.
"""

from __future__ import annotations

import ipaddress
import math
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_PARAMS_PATH = Path("configs/params.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Section):
    seeds: list[int] = Field(min_length=1)


class PathsConfig(_Section):
    raw_root: Path
    hai_dir: Path
    modbus_dir: Path
    results_root: Path
    checksums_file: Path
    interim_dir: Path
    partitions_dir: Path


class ScalingConfig(_Section):
    method: Literal["standard"]
    clip_sd: float = Field(gt=0)


class HaiConfig(_Section):
    release: str
    sample_rate_hz: int = Field(gt=0)
    label_column: str
    process_label_columns: list[str]
    drop_train_constant_features: bool
    drop_features: list[str]
    max_ffill_s: int = Field(ge=0)
    scaling: ScalingConfig


def _check_ipv4(value: str, what: str) -> str:
    try:
        ipaddress.IPv4Address(value)
    except ValueError as exc:
        raise ValueError(f"{what} is not an IPv4 address: {value!r}") from exc
    return value


class ModbusConfig(_Section):
    devices: dict[str, str] = Field(min_length=1)
    other_hosts: dict[str, str]
    attacker_ip: str
    label_source: Literal["attacker_address", "attack_logs"]
    idle_gap_s: float = Field(gt=0)
    episode_gap_s: float = Field(gt=0)
    attack_val_episodes: int = Field(gt=0)

    @model_validator(mode="after")
    def _addresses_are_valid(self) -> ModbusConfig:
        for name, ip in {**self.devices, **self.other_hosts}.items():
            _check_ipv4(ip, name)
        _check_ipv4(self.attacker_ip, "attacker_ip")
        known = set(self.devices.values()) | set(self.other_hosts.values())
        if self.attacker_ip in known:
            raise ValueError("attacker_ip must not be one of the testbed's own devices")
        return self


class WindowConfig(_Section):
    length_s: int = Field(gt=0)
    train_stride_s: int = Field(gt=0)
    infer_stride_s: int = Field(gt=0)

    @model_validator(mode="after")
    def _privacy_unit_is_non_overlapping(self) -> WindowConfig:
        # The DP privacy unit is one training window, so training windows must not
        # overlap (Proposal §7.3). Inference windows may overlap freely.
        if self.train_stride_s != self.length_s:
            raise ValueError(
                "train_stride_s must equal length_s: training windows define the "
                "privacy unit and must not overlap"
            )
        if self.infer_stride_s > self.length_s:
            raise ValueError("infer_stride_s must not exceed length_s")
        return self


class WindowsConfig(_Section):
    site_a: WindowConfig
    site_b: WindowConfig


class SplitsConfig(_Section):
    normal_val_fraction: float = Field(gt=0, lt=1)
    gap_windows: int = Field(ge=0)
    attack_val_fraction: float = Field(gt=0, lt=1)
    seed: int


_PERIOD = re.compile(r"^train\d+(:first_half|:second_half)?$")


class SiteAPartition(_Section):
    clients: list[str] = Field(min_length=2)
    block_hours: int = Field(gt=0)
    scheme: Literal["regime", "iid", "time"]
    period_assignment: dict[str, str] | None = None

    @model_validator(mode="after")
    def _time_scheme_needs_assignment(self) -> SiteAPartition:
        if self.scheme != "time":
            return self
        if not self.period_assignment:
            raise ValueError("scheme 'time' needs period_assignment")
        if set(self.period_assignment) != set(self.clients):
            raise ValueError("period_assignment must name exactly the Site A clients")
        for client, period in self.period_assignment.items():
            if not _PERIOD.match(period):
                raise ValueError(f"{client}: period must look like 'train1' or 'train3:first_half'")
        if len(set(self.period_assignment.values())) != len(self.clients):
            raise ValueError("each Site A client must get a different period")
        return self


class SiteBPartition(_Section):
    clients: dict[str, str] = Field(min_length=2)


class PartitionConfig(_Section):
    site_a: SiteAPartition
    site_b: SiteBPartition


class DataConfig(_Section):
    hai: HaiConfig
    modbus: ModbusConfig
    windows: WindowsConfig
    splits: SplitsConfig
    partition: PartitionConfig


class VaeConfig(_Section):
    hidden: list[int] = Field(min_length=1)
    latent: int = Field(gt=0)
    lr: float = Field(gt=0)
    batch: int = Field(gt=0)
    input_mode: Literal["flat", "segment_means"]
    segments_per_window: int = Field(gt=0)
    kl_weight: float = Field(ge=0)
    exclude_features: list[str] = Field(default_factory=list)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)


class AeConfig(_Section):
    hidden: list[int] = Field(min_length=1)
    lr: float = Field(gt=0)
    batch: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    transform: Literal["log1p", "none"]
    clip_sd: float = Field(gt=0)
    exclude_features: list[str]


class ScoringSite(_Section):
    aggregation: Literal["mean", "norm_mean", "norm_top3"]
    smooth: int = Field(ge=1)
    top_k: int = Field(ge=1)
    shift: Literal["r0", "r1", "r2", "r3"] = "r0"  # ADR 0005


class ShiftWindow(_Section):
    """Causal history used by the ADR 0005 shift candidates R2 and R3."""

    history_s: float = Field(default=1800.0, gt=0)
    gap_s: float = Field(default=120.0, ge=0)
    min_history_s: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def _fits(self) -> ShiftWindow:
        if self.gap_s + self.min_history_s > self.history_s:
            raise ValueError("gap_s + min_history_s must not exceed history_s")
        return self


class ScoringConfig(_Section):
    site_a: ScoringSite
    site_b: ScoringSite
    shift_window: ShiftWindow = ShiftWindow()


class BaselinesConfig(_Section):
    pca_variance: float = Field(gt=0, lt=1)
    iforest_trees: int = Field(gt=0)
    iforest_max_samples: int = Field(gt=0)


class ModelsConfig(_Section):
    vae: VaeConfig
    ae: AeConfig
    scoring: ScoringConfig
    baselines: BaselinesConfig


class StoppingConfig(_Section):
    patience: int = Field(gt=0)
    min_rel_delta: float = Field(gt=0)


class MinFitClients(_Section):
    site_a: int = Field(gt=0)
    site_b: int = Field(gt=0)


class ClientResources(_Section):
    num_cpus: float = Field(gt=0)


class FederatedConfig(_Section):
    strategy: Literal["fedavg", "fedprox"]
    proximal_mu: float = Field(ge=0)
    max_rounds: int = Field(gt=0)
    local_epochs: int = Field(gt=0)
    dropout_prob: float = Field(ge=0, lt=1)
    min_fit_clients: MinFitClients
    round_timeout_s: float = Field(gt=0)
    stopping: StoppingConfig
    rounds_to_target_tolerance: float = Field(gt=0)
    client_resources: ClientResources


class SiteBins(_Section):
    site_a: int = Field(gt=1)
    site_b: int = Field(gt=1)


class PrivacyConfig(_Section):
    target_epsilon: float = Field(gt=0)  # math.inf means non-private reference runs
    delta_cap: float = Field(gt=0, lt=1)
    delta_divisor: float = Field(gt=1)
    max_grad_norm: float = Field(gt=0)
    accountant: Literal["rdp", "prv", "gdp"]
    # ε of the normalisation release on its own; it is composed with DP-SGD in RDP
    normalisation_stats_epsilon: float = Field(gt=0)
    normalisation_bins: SiteBins = SiteBins(site_a=50, site_b=200)
    normalisation_threshold_sd: float = Field(default=3.0, ge=0)
    # H1c: ε of the noisy normal-score histogram on its own; composed in RDP (ADR 0007)
    threshold_epsilon: float = Field(default=0.5, gt=0)
    threshold_bins: int = Field(default=100, gt=1)

    @property
    def is_private(self) -> bool:
        return math.isfinite(self.target_epsilon)

    def resolve_delta(self, n_train_windows: int) -> float:
        """Return delta = min(delta_cap, 1 / (delta_divisor * N)) for a client with N windows.

        This keeps delta strictly below 1/N, the usual requirement for a meaningful
        (epsilon, delta) guarantee (Proposal §7.3, flaw P5).
        """
        if n_train_windows <= 0:
            raise ValueError("n_train_windows must be positive")
        return min(self.delta_cap, 1.0 / (self.delta_divisor * n_train_windows))


class EvaluationConfig(_Section):
    target_window_fpr: float = Field(gt=0, lt=1)
    report_window_fprs: list[float] = Field(min_length=1)
    event_tolerance_s: int = Field(ge=0)
    calibration: Literal["isotonic", "platt"]
    pooled_auroc_bins: int = Field(gt=10)
    bootstrap_resamples: int = Field(gt=0)


class SiteBFusion(_Section):
    delta_t_s: int = Field(gt=0)
    method: Literal["noisy_or", "k_of_n"]
    k_of_n: int = Field(gt=0)
    exclude_latency: bool


class FusionConfig(_Section):
    site_b: SiteBFusion


class BreakerConfig(_Section):
    failure_threshold: int = Field(gt=0)
    open_seconds: float = Field(gt=0)


class AttributionConfig(_Section):
    mode: Literal["full", "retrieval_only", "llm_no_retrieval", "rule_only"]
    top_k: int = Field(gt=0)
    min_confidence: float = Field(ge=0, le=1)
    llm_timeout_s: float = Field(gt=0)
    l_llm_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _llm_fits_target(self) -> AttributionConfig:
        if self.llm_timeout_s > self.l_llm_s:
            raise ValueError("llm_timeout_s must not exceed the full-mode target l_llm_s")
        return self


class MeshConfig(_Section):
    max_hops: int = Field(gt=0)
    connect_timeout_s: float = Field(gt=0)
    read_timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    investigation_deadline_s: float = Field(gt=0)
    breaker: BreakerConfig
    attribution: AttributionConfig


class LlmConfig(_Section):
    base_url: str
    model: str
    alternative_model: str
    num_ctx: int = Field(gt=0)
    temperature: float = Field(ge=0)


class JwtConfig(_Section):
    issuer: str
    ttl_s: int = Field(gt=0)


class SecurityConfig(_Section):
    jwt: JwtConfig
    egress_allowlist: list[str] = Field(min_length=1)


class Params(_Section):
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    models: ModelsConfig
    federated: FederatedConfig
    privacy: PrivacyConfig
    evaluation: EvaluationConfig
    fusion: FusionConfig
    mesh: MeshConfig
    llm: LlmConfig
    security: SecurityConfig

    @model_validator(mode="after")
    def _cross_section_checks(self) -> Params:
        n_a = len(self.data.partition.site_a.clients)
        n_b = len(self.data.partition.site_b.clients)
        if self.federated.min_fit_clients.site_a > n_a:
            raise ValueError(f"min_fit_clients.site_a exceeds the {n_a} Site A clients")
        if self.federated.min_fit_clients.site_b > n_b:
            raise ValueError(f"min_fit_clients.site_b exceeds the {n_b} Site B clients")
        if self.fusion.site_b.k_of_n > n_b:
            raise ValueError(f"fusion k_of_n exceeds the {n_b} Site B clients")
        if self.evaluation.target_window_fpr not in self.evaluation.report_window_fprs:
            raise ValueError("target_window_fpr must be one of report_window_fprs")
        window_a = self.data.windows.site_a.length_s * self.data.hai.sample_rate_hz
        if window_a % self.models.vae.segments_per_window:
            raise ValueError("segments_per_window must divide the Site A window length")
        site_b_devices = set(self.data.partition.site_b.clients.values())
        if site_b_devices != set(self.data.modbus.devices):
            raise ValueError("partition.site_b.clients must map to exactly the modbus devices")
        return self


def load_params(path: Path | str = DEFAULT_PARAMS_PATH) -> Params:
    """Load and validate the parameter file."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return Params.model_validate(raw)
