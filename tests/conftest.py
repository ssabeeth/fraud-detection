from pathlib import Path

import pytest

from fraud.config import load_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _reports_elsewhere(tmp_path_factory):
    """Tests never write to the committed reports/ or exports/ directories."""
    import os

    root = tmp_path_factory.mktemp("reports")
    old = {k: os.environ.get(k) for k in ("FRAUD_REPORTS_DIR", "FRAUD_EXPORTS_DIR")}
    os.environ["FRAUD_REPORTS_DIR"] = str(root)
    os.environ["FRAUD_EXPORTS_DIR"] = str(root / "exports")
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def spark():
    from fraud.spark import get_spark

    session = get_spark("tests", shuffle_partitions=2, kafka=True)
    yield session
    session.stop()


@pytest.fixture(scope="session")
def lake_settings(tmp_path_factory):
    """Settings whose data directory is a fresh temporary lake."""
    return load_settings().model_copy(update={"data_dir": tmp_path_factory.mktemp("data")})


@pytest.fixture(scope="session")
def lake(spark, lake_settings):
    """Bronze, silver and gold built once from the synthetic fixtures."""
    from fraud.lakehouse.bronze import build_bronze
    from fraud.lakehouse.gold import build_gold
    from fraud.lakehouse.silver import build_silver

    build_bronze(spark, lake_settings, FIXTURES, delete_source=False)
    build_silver(spark, lake_settings)
    build_gold(spark, lake_settings)
    return lake_settings


@pytest.fixture(scope="session")
def feature_lake(spark, lake):
    """The fixture lake with the gold feature table built."""
    from fraud.features.offline import build_features

    build_features(spark, lake)
    return lake


@pytest.fixture(scope="session")
def fixture_models(spark, feature_lake, tmp_path_factory):
    """Rules, logistic regression and LightGBM trained on the fixtures (small grids),
    saved under the fixture lake, plus a frozen expected-loss policy file."""
    import json
    from dataclasses import asdict

    from fraud.model import train as t
    from fraud.model.data import load_frame
    from fraud.model.pipeline import models_dir
    from fraud.policy.costs import CostModel

    train = load_frame(spark, feature_lake, ["train"])
    valid = load_frame(spark, feature_lake, ["valid"])
    t.setup_mlflow(feature_lake)
    grid, lr_grid = t.LGBM_GRID, t.LOGREG_GRID
    t.LGBM_GRID, t.LOGREG_GRID = [t.LGBM_GRID[0]], [t.LOGREG_GRID[1]]
    try:
        for fit in (t.fit_rules, t.fit_logreg):
            fit(train, valid)[0].save(models_dir(feature_lake))
        t.fit_lightgbm(train, valid, register=False)[0].save(models_dir(feature_lake))
    finally:
        t.LGBM_GRID, t.LOGREG_GRID = grid, lr_grid
    policy = tmp_path_factory.mktemp("policy") / "policy_frozen.json"
    policy.write_text(
        json.dumps(
            {
                "chosen": "lightgbm_expected_loss",
                "policy": {
                    "type": "ExpectedLossPolicy",
                    "score": "score_lightgbm",
                    "review_threshold": 5.0,
                    "name": "Expected loss",
                },
                "costs": asdict(CostModel.load().with_(review_capacity_per_day=3)),
            }
        )
    )
    return feature_lake, policy
