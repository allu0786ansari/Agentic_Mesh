"""Triage: alerts → incidents with a fused score and severity (ADR 0012 §2).

Alerts of one site are grouped by the Δt chaining rule, in data time (``window_end``).

* **Closing.** An incident closes when an alert of the same site ends more than Δt
  after the incident's last alert, or when the stream is drained.
* **Scores.** The fused score and k-of-n come from ``mesh.fusion``. Severity bands
  are fixed: high ≥ 0.9, medium ≥ 0.5, low otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agmesh.mesh import fusion


def severity(score: float) -> str:
    return "high" if score >= 0.9 else "medium" if score >= 0.5 else "low"


@dataclass
class ClosedIncident:
    site: str
    alerts: list[dict[str, Any]]
    msg_ids: list[str]
    fused: fusion.Incident
    k: int

    @property
    def alert_ids(self) -> list[str]:
        return sorted(a["alert_id"] for a in self.alerts)

    def summary(self) -> dict[str, Any]:
        f = self.fused
        return {
            "site": self.site,
            "alert_ids": self.alert_ids,
            "nodes": sorted({a["node_id"] for a in self.alerts}),
            "counting_nodes": list(f.nodes),
            "start_s": f.start_s,
            "end_s": f.end_s,
            "fused_score": f.noisy_or,
            "severity": severity(f.noisy_or),
            "k_of_n_confirmed": f.confirmed(self.k),
            "context_alerts": [a.alert_id for a in f.context],
        }


@dataclass
class Triage:
    delta_t_s: float
    exclude_latency: bool = True
    k: int = 2
    _open: dict[str, list[tuple[fusion.FusionAlert, dict[str, Any], str]]] = field(
        default_factory=dict
    )

    def feed(self, alert: dict[str, Any], msg_id: str) -> list[ClosedIncident]:
        """Add one alert; return the incidents it closes."""
        fa = fusion.from_alert(alert)
        site = str(alert["site_id"])
        group = self._open.setdefault(site, [])
        closed = []
        if group and fa.t_end_s - max(g[0].t_end_s for g in group) > self.delta_t_s:
            closed.append(self._close(site))
            group = self._open.setdefault(site, [])
        group.append((fa, alert, msg_id))
        return closed

    def drain(self) -> list[ClosedIncident]:
        return [self._close(site) for site in sorted(self._open) if self._open[site]]

    def _close(self, site: str) -> ClosedIncident:
        group = self._open.pop(site)
        incident = fusion.fuse([g[0] for g in group], exclude_latency=self.exclude_latency)
        return ClosedIncident(site, [g[1] for g in group], [g[2] for g in group], incident, self.k)
