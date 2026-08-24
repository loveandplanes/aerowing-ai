"""
Configuration loader unit tests.
"""

import pytest

from aerowing.config import load_config, wing_from_config, flight_condition_from_config


def test_load_config_default():
    config = load_config("default_3d")
    assert config["geometry"]["span"] == 30.0
    assert config["flight_condition"]["mach"] == 0.82


def test_load_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_wing_from_config_onera_m6():
    config = load_config("onera_m6")
    wing = wing_from_config(config)
    assert wing.name == "ONERA_M6_Benchmark"
    assert abs(wing.span - 2.392) < 1e-9
    assert abs(wing.aspect_ratio - 3.80) < 1e-9
    assert wing.root_airfoil is not None and wing.tip_airfoil is not None


def test_flight_condition_from_config():
    fc = flight_condition_from_config(load_config("nasa_crm"))
    assert fc is not None
    assert fc["mach"] == 0.85
    assert fc["alpha_deg"] == 2.2