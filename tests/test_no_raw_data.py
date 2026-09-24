import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_raw_data.py"
spec = importlib.util.spec_from_file_location("check_no_raw_data", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def _write_csv(path: Path, ids: list[int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("TransactionID,TransactionAmt\n" + "".join(f"{i},10.0\n" for i in ids))
    return str(path)


def test_aggregate_csv_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("daily.csv").write_text("day,transactions,fraud_usd\n2018-05-01,100,20.5\n")
    assert check.main(["daily.csv"]) == 0


def test_row_level_csv_outside_fixtures_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert check.main([_write_csv(Path("reports/rows.csv"), [1, 2])]) == 1


def test_synthetic_fixture_passes_and_real_ids_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok = _write_csv(Path("tests/fixtures/tx.csv"), [1, 2, 3])
    bad = _write_csv(Path("tests/fixtures/real.csv"), [2_987_000])
    assert check.main([ok]) == 0
    assert check.main([bad]) == 1


def test_parquet_and_zip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    pq.write_table(pa.table({"TransactionID": [3_000_000]}), "reports/x.parquet")
    Path("a.zip").write_bytes(b"PK")
    assert check.main(["reports/x.parquet"]) == 1
    assert check.main(["a.zip"]) == 1
