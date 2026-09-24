"""The whole stream on the fixtures: Redpanda, the processor, the Spark sink, parity.

Needs the broker: ``make up`` locally; a Redpanda service container in CI.
"""

import pytest

pytestmark = [pytest.mark.kafka, pytest.mark.spark, pytest.mark.slow]


def test_stream_end_to_end_parity(fixture_models, tmp_path, monkeypatch):
    from fraud.stream.run import run_all

    lake, policy_path = fixture_models
    monkeypatch.setenv("FRAUD_POLICY", str(policy_path))
    result = run_all(
        lake, lake.split.test.start, lake.split.test.end, speedup=0, report_dir=tmp_path
    )
    par = result["parity"]
    assert par["rows_compared"] == par["window_rows"] > 0
    assert par["feature_mismatches"] == 0, par["feature_mismatches_by_name"]
    assert par["actions_agree"] == par["actions_compared"]
    assert par["decisions_without_reasons"] == 0
    assert result["bronze_rows"]["transactions"] == par["window_rows"]
    assert result["bronze_rows"]["decisions"] == par["window_rows"]
    assert result["bronze_rows"]["labels"] == result["produced"]["labels"]
    assert (tmp_path / "stream.md").exists()


def test_a_second_replay_starts_clean(fixture_models, tmp_path, monkeypatch):
    """Topics are recreated per replay; the sink must not skip or double-count messages."""
    from fraud.stream.run import run_all

    lake, policy_path = fixture_models
    monkeypatch.setenv("FRAUD_POLICY", str(policy_path))
    first = run_all(lake, lake.split.test.start, lake.split.test.end, 0, report_dir=tmp_path)
    second = run_all(lake, lake.split.test.start, lake.split.test.end, 0, report_dir=tmp_path)
    assert second["bronze_rows"] == first["bronze_rows"]
    assert second["parity"]["feature_mismatches"] == 0
