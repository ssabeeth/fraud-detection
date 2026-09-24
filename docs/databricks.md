# Running phases 2-5 on Databricks Free Edition

The same code that runs locally runs as a Databricks Job, defined in an Asset Bundle
(`databricks.yml`, `databricks/resources.yml`). The job's tasks are the CLI commands of
the local pipeline, run from the project's wheel on serverless compute:

`bronze → silver → gold → (profile) → features → check_pit → train → evaluate → policy`

Free Edition blocks most outbound internet, so the data is never downloaded inside the
workspace: you download it on your machine and upload the two CSVs to a Unity Catalog
volume. The job deletes nothing in the raw volume; delete the CSVs yourself once bronze
has run, so the lake holds the only copy, as it does locally.

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

Then delete the raw CSVs from the volume:

```bash
databricks fs rm dbfs:/Volumes/workspace/fraud/raw/train_transaction.csv
databricks fs rm dbfs:/Volumes/workspace/fraud/raw/train_identity.csv
```

## Differences from the local run

- Delta tables are written by path into the `lake` volume rather than registered in the
  catalog, so the jobs stay identical to the local ones.
- Serverless compute does not cache DataFrames; the profile job skips `.cache()` there.
- The modelling tasks pull the training months into pandas on the driver (1.2 GB with
  all columns). If the serverless environment runs out of memory, run the modelling
  tasks locally against the downloaded lake; the data pipeline tasks are unaffected.

## Status

Written and linted locally; **not yet deployed**, because it needs the owner's workspace
and CLI login. See PROGRESS.md.
