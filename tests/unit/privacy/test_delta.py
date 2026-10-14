"""δ is tied to the client's dataset size (flaw P5)."""

from __future__ import annotations

import pytest

from agmesh.common.config import Params


@pytest.mark.parametrize("n", [1, 50, 3_060, 10_000, 64_792, 10**7])
def test_delta_below_one_over_n(params: Params, n: int) -> None:
    cfg = params.privacy
    delta = cfg.resolve_delta(n)
    assert delta < 1.0 / n
    assert delta == min(cfg.delta_cap, 1.0 / (cfg.delta_divisor * n))


def test_delta_rejects_empty_client(params: Params) -> None:
    with pytest.raises(ValueError):
        params.privacy.resolve_delta(0)
