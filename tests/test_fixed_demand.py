from __future__ import annotations

import pytest

from fixed_demand import count_scheduled_vehicles, sha256_file


def test_fixed_demand_count_and_digest(tmp_path):
    route = tmp_path / "paired.rou.xml"
    route.write_text(
        '<routes><vehicle id="a" depart="0"/><trip id="b" depart="1"/></routes>'
    )
    assert count_scheduled_vehicles(route) == 2
    assert len(sha256_file(route)) == 64


def test_fixed_demand_rejects_runtime_flows(tmp_path):
    route = tmp_path / "flow.rou.xml"
    route.write_text('<routes><flow id="dynamic" begin="0" end="10"/></routes>')
    with pytest.raises(ValueError, match="cannot contain <flow>"):
        count_scheduled_vehicles(route)
