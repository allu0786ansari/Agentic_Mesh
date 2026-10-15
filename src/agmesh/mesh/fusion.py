"""Site B evidence fusion: incidents, noisy-OR and k-of-n (Proposal §8.2; ADR 0009 §7).

Pure functions over alerts, with no I/O, so Triage (Week 10) and the X3 harness
(Week 13) share one definition.

* **Incidents.** Sort a site's alerts by ``window_end``. An alert joins the current
  incident when it ends within ``delta_t_s`` of that incident's previous alert;
  otherwise it opens a new incident.
* **Counting alerts.** With ``exclude_latency``, a latency-only alert counts for
  neither score. It is kept as single-device context.
* **Noisy-OR score:** ``1 - prod(1 - p_n)`` over the counting nodes, where ``p_n`` is
  node n's largest calibrated probability in the incident.
* **k-of-n:** the number of distinct counting nodes. The incident is confirmed when
  it is at least ``k``.

Δt and k come from ``configs/params.yaml`` (frozen at ``g1-frozen``). They are not
tuned here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class FusionAlert:
    alert_id: str
    node_id: str
    t_end_s: float
    prob: float
    latency_only: bool = False


@dataclass(frozen=True)
class Incident:
    alerts: tuple[FusionAlert, ...]
    counting: tuple[FusionAlert, ...]  # alerts that count for fusion
    context: tuple[FusionAlert, ...]  # latency-only alerts kept as context
    noisy_or: float
    nodes: tuple[str, ...]  # distinct counting nodes
    start_s: float
    end_s: float

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def confirmed(self, k: int) -> bool:
        return self.n_nodes >= k

    @property
    def alert_ids(self) -> list[str]:
        return [a.alert_id for a in self.alerts]


def parse_iso(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def from_alert(alert: dict[str, object]) -> FusionAlert:
    """A fusion view of an :class:`~agmesh.common.schemas.Alert` payload."""
    return FusionAlert(
        str(alert["alert_id"]),
        str(alert["node_id"]),
        parse_iso(str(alert["window_end"])),
        float(alert["calibrated_prob"]),  # type: ignore[arg-type]
        bool(alert["latency_only"]),
    )


def noisy_or(probs: Iterable[float]) -> float:
    out = 1.0
    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability {p} outside [0, 1]")
        out *= 1.0 - p
    return 1.0 - out


def fuse(alerts: Sequence[FusionAlert], *, exclude_latency: bool = True) -> Incident:
    """Fused view of one group of alerts."""
    if not alerts:
        raise ValueError("an incident needs at least one alert")
    ordered = tuple(sorted(alerts, key=lambda a: (a.t_end_s, a.node_id, a.alert_id)))
    counting = tuple(a for a in ordered if not (exclude_latency and a.latency_only))
    context = tuple(a for a in ordered if exclude_latency and a.latency_only)
    best: dict[str, float] = {}
    for a in counting:
        best[a.node_id] = max(best.get(a.node_id, 0.0), a.prob)
    return Incident(
        ordered,
        counting,
        context,
        noisy_or(best.values()) if best else 0.0,
        tuple(sorted(best)),
        ordered[0].t_end_s,
        ordered[-1].t_end_s,
    )


def incidents(
    alerts: Iterable[FusionAlert], delta_t_s: float, *, exclude_latency: bool = True
) -> list[Incident]:
    """Group one site's alerts into incidents by the Δt chaining rule."""
    if delta_t_s <= 0:
        raise ValueError("delta_t_s must be positive")
    ordered = sorted(alerts, key=lambda a: (a.t_end_s, a.node_id, a.alert_id))
    groups: list[list[FusionAlert]] = []
    for a in ordered:
        if groups and a.t_end_s - groups[-1][-1].t_end_s <= delta_t_s:
            groups[-1].append(a)
        else:
            groups.append([a])
    return [fuse(g, exclude_latency=exclude_latency) for g in groups]


@dataclass
class Sensitivity:
    """Descriptive only (ADR 0009 §7): incident counts over a Δt x k grid."""

    rows: list[dict[str, float]] = field(default_factory=list)


def sensitivity(
    alerts: Sequence[FusionAlert],
    deltas: Sequence[float] = (30.0, 60.0, 120.0),
    ks: Sequence[int] = (1, 2, 3),
    *,
    exclude_latency: bool = True,
) -> Sensitivity:
    out = Sensitivity()
    for d in deltas:
        found = incidents(alerts, d, exclude_latency=exclude_latency)
        for k in ks:
            out.rows.append(
                {
                    "delta_t_s": d,
                    "k": k,
                    "incidents": len(found),
                    "confirmed": sum(i.confirmed(k) for i in found),
                }
            )
    return out
