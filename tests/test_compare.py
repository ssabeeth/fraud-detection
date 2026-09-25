"""The rolling-origin model comparison: folds only look back, and May is never read."""

import json

import pandas as pd
import pytest

from fraud.model import compare
from fraud.model.data import list_test_touches, load_frame


@pytest.mark.spark
def test_folds_only_look_back_and_never_touch_the_test_month(spark, feature_lake):
    df = load_frame(spark, feature_lake, ["train", "valid"])
    got = compare.folds(df)
    assert [f.scored for f in got] == list(compare.SCORED_MONTHS)
    for f in got:
        fit, inner, score = (pd.to_datetime(x["event_date"]) for x in (f.fit, f.inner, f.score))
        assert fit.max() < inner.min() and inner.max() < score.min()
        assert (inner.max() - inner.min()).days == compare.INNER_DAYS - 1
        assert set(score.dt.strftime("%Y-%m")) == {f.scored}
        assert "test" not in set(f.fit["split"]) | set(f.inner["split"]) | set(f.score["split"])


@pytest.mark.spark
@pytest.mark.slow
def test_run_compares_all_libraries_without_reading_may(spark, feature_lake, tmp_path, monkeypatch):
    for lib in compare.LIBRARIES:
        monkeypatch.setitem(compare.GRIDS, lib, compare.GRIDS[lib][:1])
    monkeypatch.setattr(compare, "MAX_ROUNDS", 30)
    monkeypatch.setattr(compare, "PATIENCE", 10)
    touches = len(list_test_touches())
    result = compare.run(spark, feature_lake, out_dir=tmp_path)
    assert len(list_test_touches()) == touches
    assert set(result["summary"]) == set(compare.LIBRARIES)
    for lib, s in result["summary"].items():
        assert len(s["pr_auc"]) == len(compare.SCORED_MONTHS), lib
        assert all(c > 0 for c in s["total_cost"]), lib
        assert result["latency_ms"][lib] > 0
    assert result["summary"]["lightgbm"]["cost_vs_lightgbm"] == [0.0, 0.0, 0.0]
    saved = json.loads((tmp_path / "model_comparison.json").read_text())
    assert saved["best"] == result["best"]
    report = (tmp_path / "model_comparison.md").read_text()
    assert "the test month (May) is not read" in report
    assert "CatBoost" in report and "XGBoost" in report
