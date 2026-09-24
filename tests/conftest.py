from pathlib import Path

import pytest

from fraud.config import load_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def spark():
    from fraud.spark import get_spark

    session = get_spark("tests", shuffle_partitions=2)
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
