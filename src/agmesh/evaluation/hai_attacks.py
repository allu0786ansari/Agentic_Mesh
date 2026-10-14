"""The documented HAI 21.03 attacks, and how each maps to ATT&CK for ICS (mapping protocol v1.0).

Source: *HAI Security Dataset - Technical Details*, icsdataset/hai,
``hai_dataset_technical_details.pdf``, section "HAI 21.03 - Attack operation"
(pp. 36-38). The table below is a transcription of that section: 50 attacks, 25
single primitives (A1xx, A2xx) and 25 combinations of two primitives (A3xx-A5xx),
each with its target controller code, target point(s), start time and duration. The
start times are the data's own clock; every one matches the first labelled row of a
HAI 21.03 test event to the second or within a few seconds.

**Target controller code.** ``<process>-<controller>-<variable>``. The variable is
``SP`` (setpoint), ``CO`` (control output, the command the controller sends to an
actuator), ``SPnPVn`` or ``COnPVn`` (the same, while the sensor reading of the
controlled variable is frozen or replayed to hide the effect), with ``-ST`` (or
``-STn``) for a short-term attack repeated several times.

**Errata in the source, recorded and corrected here.**

* A220, A402, A403, A405, A504, A506 and A511 list ``P2_LCV01D`` as the target of
  ``P3-LC-CO2``. HAI has no ``P2_LCV01D``; the P3 level controller's second output
  is ``P3_LCV01D``, and these events are labelled ``attack_P3``.
* A209 is listed as 159 s; the labelled event lasts 199 s. The start time matches.

**HAI 21.03 column names.** The document uses the newer point names. In the 21.03
data, ``P2_AutoSD`` is the column ``P2_ASD`` and ``P2_SCO`` (speed-controller
output) is ``P2_CO_rpm``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

SOURCE = (
    "HAI Security Dataset Technical Details (icsdataset/hai, "
    "hai_dataset_technical_details.pdf), HAI 21.03 attack operation, pp. 36-38"
)
#: Document point name -> HAI 21.03 column.
POINT_ALIASES = {"P2_AutoSD": "P2_ASD", "P2_SCO": "P2_CO_rpm", "P2_LCV01D": "P3_LCV01D"}
MATCH_TOLERANCE_S = 90.0


@dataclass(frozen=True)
class Primitive:
    code: str  # e.g. "P1-LC-SP1PV1"
    points: tuple[str, ...]  # as printed in the document

    @property
    def process(self) -> str:
        return self.code.split("-")[0]

    @property
    def kind(self) -> str:
        """``sp`` or ``co``: what the attacker changes to act on the process."""
        variable = self.code.split("-")[2]
        return "sp" if variable.startswith("SP") else "co"

    @property
    def hides_sensor(self) -> bool:
        return "PV" in self.code.split("-")[2]

    @property
    def short_term(self) -> bool:
        return bool(re.search(r"-ST\d*$", self.code))

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(POINT_ALIASES.get(p, p) for p in self.points)


@dataclass(frozen=True)
class Attack:
    attack_id: str
    start: str  # "YYYY-MM-DD HH:MM", data clock
    duration_s: int
    primitives: tuple[Primitive, ...]

    @property
    def start_dt(self) -> datetime:
        return datetime.strptime(self.start, "%Y-%m-%d %H:%M")

    @property
    def processes(self) -> list[str]:
        return sorted({p.process for p in self.primitives})


def _p(code: str, *points: str) -> Primitive:
    return Primitive(code, points)


LCO2 = _p("P3-LC-CO2", "P2_LCV01D")  # erratum: P3_LCV01D
TIMETABLE: tuple[Attack, ...] = (
    Attack("A101", "2020-07-07 15:35", 192, (_p("P1-PC-SP1", "P1_B2016"),)),
    Attack("A102", "2020-07-07 17:28", 98, (_p("P1-FC-SP1", "P1_B3005"),)),
    Attack("A103", "2020-07-07 18:59", 190, (_p("P1-LC-CO1", "P1_LCV01D"),)),
    Attack("A104", "2020-07-07 20:21", 60, (_p("P2-SC-CO1", "P2_SCO"),)),
    Attack("A105", "2020-07-07 21:03", 89, (_p("P2-SC-SP1", "P2_AutoSD"),)),
    Attack("A201", "2020-07-09 15:47", 83, (_p("P2-TC-SP2", "P2_VTR02"),)),
    Attack("A202", "2020-07-09 17:38", 422, (_p("P1-PC-SP1PV1", "P1_B2016", "P1_PIT01"),)),
    Attack("A203", "2020-07-09 18:59", 17, (_p("P1-LC-CO1-ST7", "P1_LCV01D"),)),
    Attack("A204", "2020-07-09 20:10", 259, (_p("P1-FC-SP1PV1", "P1_B3005", "P1_FT03"),)),
    Attack("A205", "2020-07-09 21:15", 123, (_p("P1-PC-SP1-ST10", "P1_B2016"),)),
    Attack("A206", "2020-07-09 23:02", 256, (_p("P1-FC-CO1PV1", "P1_FCV03D", "P1_FT03"),)),
    Attack("A207", "2020-07-10 01:08", 68, (_p("P2-TC-SP1", "P2_VTR01"),)),
    Attack("A208", "2020-07-10 01:33", 261, (_p("P1-LC-SP1PV1", "P1_B3004", "P1_LIT01"),)),
    Attack("A209", "2020-07-10 03:03", 159, (_p("P1-LC-SP1", "P1_B3004"),)),
    Attack("A210", "2020-07-10 05:29", 421, (_p("P1-PC-CO1PV1", "P1_PCV01D", "P1_PIT01"),)),
    Attack("A211", "2020-07-10 07:51", 45, (_p("P2-SC-SP1-ST5", "P2_AutoSD"),)),
    Attack("A212", "2020-07-10 09:13", 152, (_p("P2-SC-SP1PV1", "P2_AutoSD", "P2_SIT01"),)),
    Attack("A213", "2020-07-10 10:49", 254, (_p("P1-LC-CO1PV1", "P1_LCV01D", "P1_LIT01"),)),
    Attack("A214", "2020-07-10 12:51", 152, (_p("P1-PC-CO1", "P1_PCV01D"),)),
    Attack("A215", "2020-07-10 15:11", 151, (_p("P2-SC-CO1PV1", "P2_SCO", "P2_SIT01"),)),
    Attack("A216", "2020-07-10 15:40", 65, (_p("P1-FC-CO1-ST10", "P1_FCV03D"),)),
    Attack("A217", "2020-07-10 16:22", 184, (_p("P2-TC-SP3", "P2_RTR"),)),
    Attack("A218", "2020-07-10 18:21", 99, (_p("P1-FC-CO1", "P1_FCV03D"),)),
    Attack("A219", "2020-07-10 21:25", 119, (_p("P3-LC-CO1", "P3_LCP01D"),)),
    Attack("A220", "2020-07-10 22:56", 119, (LCO2,)),
    Attack(
        "A301",
        "2020-07-13 13:51",
        132,
        (_p("P1-LC-CO1-ST", "P1_LCV01D"), _p("P1-FC-SP1", "P1_B3005")),
    ),
    Attack(
        "A302",
        "2020-07-13 15:21",
        421,
        (_p("P1-PC-SP1PV1", "P1_B2016", "P1_PIT01"), _p("P1-FC-SP1", "P1_B3005")),
    ),
    Attack(
        "A303",
        "2020-07-13 18:11",
        189,
        (_p("P1-PC-CO1", "P1_PCV01D"), _p("P1-LC-CO1", "P1_LCV01D")),
    ),
    Attack(
        "A304",
        "2020-07-13 20:53",
        106,
        (_p("P2-SC-SP1", "P2_AutoSD"), _p("P2-TC-SP1", "P2_VTR01")),
    ),
    Attack(
        "A305", "2020-07-13 21:23", 84, (_p("P2-SC-CO1", "P2_SCO"), _p("P2-TC-SP2", "P2_VTR02"))
    ),
    Attack(
        "A306",
        "2020-07-13 23:55",
        238,
        (_p("P1-PC-SP1", "P1_B2016"), _p("P2-SC-SP1", "P2_AutoSD")),
    ),
    Attack(
        "A307",
        "2020-07-14 01:51",
        110,
        (_p("P1-FC-CO1", "P1_FCV03D"), _p("P2-TC-SP1", "P2_VTR01")),
    ),
    Attack(
        "A308",
        "2020-07-14 03:53",
        255,
        (_p("P1-LC-CO1PV1", "P1_LCV01D", "P1_LIT01"), _p("P2-SC-SP1-ST", "P2_AutoSD")),
    ),
    Attack(
        "A401",
        "2020-07-28 12:43",
        254,
        (_p("P1-PC-CO1", "P1_PCV01D"), _p("P1-LC-SP1PV1", "P1_B3004", "P1_LIT01")),
    ),
    Attack("A402", "2020-07-28 13:45", 262, (_p("P1-FC-SP1PV1", "P1_B3005", "P1_FT03"), LCO2)),
    Attack("A403", "2020-07-28 15:57", 263, (_p("P1-LC-SP1PV1", "P1_B3004", "P1_LIT01"), LCO2)),
    Attack(
        "A404",
        "2020-07-28 17:45",
        258,
        (_p("P2-SC-CO1PV1", "P2_SCO", "P2_SIT01"), _p("P1-LC-CO1PV1", "P1_LCV01D", "P1_LIT01")),
    ),
    Attack("A405", "2020-07-28 20:47", 120, (_p("P2-SC-SP1-ST", "P2_AutoSD"), LCO2)),
    Attack(
        "A501",
        "2020-07-30 11:16",
        172,
        (_p("P1-PC-CO1", "P1_PCV01D"), _p("P2-TC-SP2", "P2_VTR02")),
    ),
    Attack(
        "A502",
        "2020-07-30 13:30",
        258,
        (_p("P1-FC-CO1PV1", "P1_FCV03D", "P1_FT03"), _p("P2-SC-CO1", "P2_SCO")),
    ),
    Attack(
        "A503",
        "2020-07-30 16:05",
        256,
        (_p("P1-LC-SP1PV1", "P1_B3004", "P1_LIT01"), _p("P2-SC-CO1", "P2_SCO")),
    ),
    Attack("A504", "2020-07-30 17:45", 120, (_p("P1-FC-CO1", "P1_FCV03D"), LCO2)),
    Attack(
        "A505",
        "2020-07-30 18:38",
        203,
        (_p("P1-LC-SP1", "P1_B3004"), _p("P2-SC-SP1-ST", "P2_AutoSD")),
    ),
    Attack("A506", "2020-07-30 20:42", 153, (_p("P2-SC-CO1PV1", "P2_SCO", "P2_SIT01"), LCO2)),
    Attack(
        "A507",
        "2020-07-30 23:13",
        79,
        (_p("P2-SC-SP1-ST", "P2_AutoSD"), _p("P2-TC-SP1", "P2_VTR01")),
    ),
    Attack(
        "A508",
        "2020-07-31 01:15",
        51,
        (_p("P1-FC-CO1-ST", "P1_FCV03D"), _p("P1-LC-CO1-ST", "P1_LCV01D")),
    ),
    Attack(
        "A509",
        "2020-07-31 02:01",
        241,
        (_p("P1-PC-SP1", "P1_B2016"), _p("P1-PC-CO1", "P1_PCV01D")),
    ),
    Attack(
        "A510",
        "2020-07-31 09:54",
        262,
        (_p("P1-LC-SP1", "P1_B3004"), _p("P1-LC-CO1PV1", "P1_LCV01D", "P1_LIT01")),
    ),
    Attack("A511", "2020-07-31 10:40", 120, (_p("P2-TC-SP3", "P2_RTR"), LCO2)),
    Attack(
        "A512",
        "2020-07-31 11:21",
        262,
        (_p("P1-FC-SP1", "P1_B3005"), _p("P1-FC-CO1PV1", "P1_FCV03D", "P1_FT03")),
    ),
)


# ------------------------------------------------------------------ mapping rule v1.0

#: Primitive kind -> (primary, acceptable). Mapping protocol §2, rule v1.0.
#: Acceptable sets are kept to the techniques an analyst would name as the main one,
#: so that no constant answer is correct for most events (see ``constant_baseline``).
BASE = {
    # A changed setpoint or trip limit is a modified parameter. (The document notes
    # the change is hidden in the HMI; that is part of the same primitive, not a
    # second technique an analyst must name.)
    "sp": ("T0836", ("T0836",)),
    # A changed control output is an unauthorised command to the actuator; changing
    # the controller's output point is also Manipulation of Control.
    "co": ("T0855", ("T0855", "T0831")),
}
#: Added when the sensor of the controlled variable is frozen or replayed.
HIDDEN_SENSOR = ("T0856",)
#: Added for short-term attacks repeated several times (repetitive I/O changes).
SHORT_TERM = ("T0806",)
#: Combination of two primitives: the primary comes from the more upstream one.
PRECEDENCE = {"sp": 0, "co": 1}


def techniques(attack: Attack) -> tuple[str, list[str]]:
    """Primary technique and acceptable set of a documented attack (rule v1.0)."""
    lead = min(attack.primitives, key=lambda p: PRECEDENCE[p.kind])
    primary = BASE[lead.kind][0]
    acceptable: list[str] = []
    for p in attack.primitives:
        extra = [*BASE[p.kind][1]]
        if p.hides_sensor:
            extra += HIDDEN_SENSOR
        if p.short_term:
            extra += SHORT_TERM
        acceptable += [t for t in extra if t not in acceptable]
    return primary, acceptable


def match(start: datetime, processes: list[str]) -> Attack | None:
    """The documented attack whose start is within tolerance and whose processes agree."""
    best: Attack | None = None
    for attack in TIMETABLE:
        gap = abs((attack.start_dt - start).total_seconds())
        closer = best is None or gap < abs((best.start_dt - start).total_seconds())
        if gap <= MATCH_TOLERANCE_S and attack.processes == sorted(processes) and closer:
            best = attack
    return best


def constant_baseline() -> dict[str, float]:
    """Share of attacks for which always answering one technique counts as correct."""
    sets = [set(techniques(a)[1]) for a in TIMETABLE]
    names = sorted(set().union(*sets))
    return {t: sum(t in s for s in sets) / len(sets) for t in names}
