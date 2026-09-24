from datetime import date, datetime

import pytest
import yaml

from fraud.cli import main
from fraud.config import DEFAULT_CONFIG, load_settings


def test_default_config_loads():
    s = load_settings()
    assert s.anchor == datetime(2017, 11, 30)
    assert s.label_delay_days == 30
    assert s.split.train.start == date(2017, 12, 1)


def test_split_is_contiguous_and_ordered():
    bounds = load_settings().split_bounds_seconds()
    assert bounds["train"][1] == bounds["valid"][0]
    assert bounds["valid"][1] == bounds["test"][0]
    # The first raw TransactionDT in the data is 86400 (one day after the anchor).
    assert bounds["train"][0] == 86_400


def test_anchor_round_trip():
    s = load_settings()
    assert s.to_seconds(s.to_datetime(123_456)) == 123_456


@pytest.mark.parametrize(
    "split",
    [
        # overlapping train and valid
        {
            "train": ["2017-12-01", "2018-04-15"],
            "valid": ["2018-04-01", "2018-05-01"],
            "test": ["2018-05-01", "2018-06-01"],
        },
        # gap between valid and test
        {
            "train": ["2017-12-01", "2018-04-01"],
            "valid": ["2018-04-01", "2018-04-20"],
            "test": ["2018-05-01", "2018-06-01"],
        },
        # empty interval
        {
            "train": ["2017-12-01", "2017-12-01"],
            "valid": ["2017-12-01", "2018-05-01"],
            "test": ["2018-05-01", "2018-06-01"],
        },
    ],
)
def test_bad_split_is_rejected(tmp_path, split):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["split"] = split
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_settings(path)


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_DATA_DIR", str(tmp_path))
    assert load_settings().data_dir == tmp_path


def test_cli_config(capsys):
    assert main(["config"]) == 0
    assert '"label_delay_days": 30' in capsys.readouterr().out
