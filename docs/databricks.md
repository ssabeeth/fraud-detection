# Running phases 2-5 on Databricks Free Edition

The same code that runs locally runs as a Databricks Job, defined in an Asset Bundle
(`databricks.yml`, `databricks/resources.yml`). The job's tasks are the CLI commands of
the local pipeline, run from the project's wheel on serverless compute:

`bronze → silver → gold → (profile) → features → check_pit → train → evaluate → policy`

Free Edition blocks most outbound internet, so the data is never downloaded inside the
workspace: you download it on your machine and upload the two CSVs to a Unity Catalog
volume. Bronze deletes them from the volume once it has converted them, so the lake holds
the only copy, as it does locally.

## One-time setup (owner)

1. Create a Free Edition workspace (free, no card): <https://www.databricks.com/learn/free-edition>.
2. Install the CLI (already on this machine): `brew install databricks/tap/databricks`.
3. Log in, which stores a token in `~/.databrickscfg`, outside the repository:

   ```bash
   databricks auth login --host https://<your-workspace>.cloud.databricks.com
   ```

## Deploy and run

```bash
make databricks-deploy     # validate and deploy: schema `fraud`, volumes `raw` and `lake`, the job
make databricks-upload     # download the CSVs from Kaggle, upload them to the raw volume, delete local copies
make databricks-run        # run the job and wait for it
```

The job writes the lake, models and reports under `/Volumes/workspace/fraud/lake/`.
MLflow uses the workspace's tracking server (experiment `/Shared/fraud-models`) and the
Unity Catalog model registry. Download the reports to compare with the local run:

```bash
databricks fs cp -r dbfs:/Volumes/workspace/fraud/lake/reports ./data/databricks-reports
```

Check that bronze removed the uploaded CSVs (the listing should be empty):

```bash
databricks fs ls dbfs:/Volumes/workspace/fraud/raw
```

If a later task fails, repair the run from that task rather than starting again: bronze
has already deleted the CSVs, so a full rerun would need them uploaded again.

```bash
databricks jobs repair-run --json '{"run_id": <run>, "rerun_tasks": ["train", "evaluate", "policy"]}'
```

## Differences from the local run

- Delta tables are written by path into the `lake` volume rather than registered in the
  catalog, so the jobs stay identical to the local ones.
- Serverless compute does not cache DataFrames; the profile job skips `.cache()` there.
- The job uses serverless environment version 6. Version 3 ships pandas 1.5 and numpy
  1.26, and serverless refuses a wheel that upgrades those core packages.
- Unity Catalog volumes cannot append to a file, so every file the pipeline writes is
  written in one go (the LightGBM model, the test-read log).
- Models are registered in Unity Catalog as `workspace.fraud.fraud_<model>`.
- The target has no `mode: development`, which would rename the schema to
  `dev_<user>_fraud` and leave the job's volume paths pointing at nothing.
- The modelling tasks pull the training months into pandas on the driver (1.2 GB with
  all columns); this fitted in the serverless memory.

## Result (2026-09-25)

Deployed to the owner's Free Edition workspace and run end to end on serverless compute:
bronze 1.6 min, silver 1.2, gold 0.6, profile 0.8, features 0.8, check_pit 13.2,
train 55.5, evaluate 1.5 and policy 3.8. It reproduces the local run:

| | Local | Databricks |
|---|---|---|
| Rows, fraud and dollars in each split; card keys | 590,540 rows; 217,850 keys | identical |
| Point-in-time check | 76,415 values, 0 differences | 76,347 values, 0 differences |
| LightGBM PR-AUC, validation / test | 0.6156 / 0.5474 | 0.6156 / 0.5474 |
| Rules baseline PR-AUC, validation / test | 0.0519 / 0.0469 | identical |
| Logistic regression PR-AUC, test | 0.1128 | 0.1126 |
| Chosen policy | LightGBM, expected loss | the same |
| **Chosen policy on May: cost, fraud value caught** | **$278,535, 59.8%** | **$278,535, 59.8%** |
| Rules baseline on May | $474,219 | $474,219 |
| Logistic-regression policy on May | $472,022 | $472,921 |

LightGBM (4.7.0 locally, 4.6.0 on serverless) gives the same model to six decimals. The
logistic regression moves by 0.2%: scikit-learn is 1.9.1 locally and 1.7.2 on serverless.
The point-in-time sample checked 4,491 rows rather than 4,495 because Spark returns rows
in a different order there, so the seeded top-up sample differs; both found no
differences. The job's two test-month reads are logged as a reproduction, not a new
result. The comparison was run with the reports downloaded to `data/databricks-reports/`
(not committed).
