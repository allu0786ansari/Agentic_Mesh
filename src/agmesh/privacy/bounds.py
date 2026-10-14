"""Public value ranges for every modelled feature (no data involved).

DP normalisation needs bounds on each unit's contribution that do not depend on the
private data. The bounds here come from public sources only.

* **Site A (HAI 21.03).** The documented range ("Min", "Max") of each point in the
  HAI point list (*HAI Security Dataset Technical Details*, pp. 12-14). In 21.03 some
  points still carry their older names: ``P2_VYT02/VXT02/VYT03/VXT03`` are
  ``P2_VIBTR01-04``, ``P2_CO_rpm`` is ``P2_SCO`` and ``P2_ASD`` is ``P2_AutoSD``.
  Two modelled points are not in the list. ``P2_SIT02`` is the second speed probe,
  given the range of ``P2_SIT01`` (0-3,200 RPM). ``P2_HILout`` has no public range
  and gets a deliberately wide declared range; a wide range only costs resolution.
  Two public conventions are applied on top of the document, both fixed in advance
  and never fitted to data. Every documented range is widened by half its span on
  each side, and points that HAI 21.03 stores as raw S7 analog counts get the S7 raw
  range (``S7_RAW``).
* **Site B (Modbus).** All features are per-10-second counts, or a latency in ms,
  passed through ``log1p``. They get the declared range [0, log1p(10,000)]. A larger
  count is clipped for the statistics only.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

SITE_A: dict[str, tuple[float, float]] = {
    "P1_B2004": (0.0, 10.0),
    "P1_B2016": (0.0, 10.0),
    "P1_B3004": (0.0, 720.0),
    "P1_B3005": (0.0, 2500.0),
    "P1_B4002": (0.0, 100.0),
    "P1_B4005": (0.0, 100.0),
    "P1_B400B": (0.0, 2500.0),
    "P1_B4022": (0.0, 40.0),
    "P1_FCV01D": (0.0, 100.0),
    "P1_FCV01Z": (0.0, 100.0),
    "P1_FCV02D": (0.0, 100.0),
    "P1_FCV02Z": (0.0, 100.0),
    "P1_FCV03D": (0.0, 100.0),
    "P1_FCV03Z": (0.0, 100.0),
    "P1_FT01": (0.0, 2500.0),
    "P1_FT01Z": (0.0, 3190.0),
    "P1_FT02": (0.0, 2500.0),
    "P1_FT02Z": (0.0, 3190.0),
    "P1_FT03": (0.0, 2500.0),
    "P1_FT03Z": (0.0, 3190.0),
    "P1_LCV01D": (0.0, 100.0),
    "P1_LCV01Z": (0.0, 100.0),
    "P1_LIT01": (0.0, 720.0),
    "P1_PCV01D": (0.0, 100.0),
    "P1_PCV01Z": (0.0, 100.0),
    "P1_PCV02Z": (0.0, 100.0),
    "P1_PIT01": (0.0, 10.0),
    "P1_PIT02": (0.0, 10.0),
    "P1_TIT01": (-50.0, 150.0),
    "P1_TIT02": (-50.0, 150.0),
    "P2_24Vdc": (0.0, 30.0),
    "P2_CO_rpm": (0.0, 100000.0),
    "P2_HILout": (-100000.0, 100000.0),  # not documented: wide declared range
    "P2_SIT01": (0.0, 3200.0),
    "P2_SIT02": (0.0, 3200.0),  # not listed: second speed probe, range of P2_SIT01
    "P2_VT01": (11.0, 12.0),
    "P2_VXT02": (-10.0, 10.0),
    "P2_VXT03": (-10.0, 10.0),
    "P2_VYT02": (-10.0, 10.0),
    "P2_VYT03": (-10.0, 10.0),
    "P3_FIT01": (0.0, 27648.0),
    "P3_LCP01D": (0.0, 27648.0),
    "P3_LCV01D": (0.0, 27648.0),
    "P3_LIT01": (0.0, 90.0),
    "P3_PIT01": (0.0, 27648.0),
    "P4_HT_FD": (-0.02, 0.02),
    "P4_HT_LD": (0.0, 100.0),
    "P4_HT_PO": (0.0, 100.0),
    "P4_LD": (0.0, 500.0),
    "P4_ST_FD": (-0.02, 0.02),
    "P4_ST_GOV": (0.0, 27648.0),
    "P4_ST_LD": (0.0, 500.0),
    "P4_ST_PO": (0.0, 500.0),
    "P4_ST_PS": (0.0, 500.0),
    "P4_ST_PT01": (0.0, 27648.0),
    "P4_ST_TT01": (0.0, 27648.0),
}
SITE_B_RANGE = (0.0, math.log1p(10_000.0))


#: Siemens S7 analog input convention (public): nominal 0..27,648, with under-range
#: down to -4,864 and over-range up to 32,511. HAI 21.03 stores its P3 analog points
#: and the HIL simulator's digital values in these raw units.
S7_RAW = (-4864.0, 32511.0)
S7_POINTS = frozenset(
    {
        "P3_FIT01",
        "P3_LCP01D",
        "P3_LCV01D",
        "P3_LIT01",  # documented in %, stored in raw counts in 21.03
        "P3_PIT01",
        "P4_ST_GOV",
        "P4_ST_PT01",
        "P4_ST_TT01",
    }
)
#: Documented ranges are widened by this share of their span on each side. Sensor
#: noise, calibration offsets and setpoints just past the nominal range are common.
#: The margin is fixed in advance; it is never fitted to data.
MARGIN = 0.5


def bounds_for(site: str, features: tuple[str, ...] | list[str]) -> tuple[F64, F64]:
    """Lower and upper public bound of each feature, in order."""
    if site == "site_a":
        missing = [f for f in features if f not in SITE_A]
        if missing:
            raise KeyError(f"no public range for {missing}")
        lo, hi = [], []
        for f in features:
            a, b = S7_RAW if f in S7_POINTS else SITE_A[f]
            pad = 0.0 if f in S7_POINTS else MARGIN * (b - a)
            lo.append(a - pad)
            hi.append(b + pad)
        return np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)
    n = len(features)
    return np.full(n, SITE_B_RANGE[0]), np.full(n, SITE_B_RANGE[1])
