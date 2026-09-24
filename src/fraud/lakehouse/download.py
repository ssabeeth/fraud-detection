"""Download the two training files from Kaggle.

Needs the owner's token (``~/.kaggle/access_token`` or ``~/.kaggle/kaggle.json``) and
the competition rules accepted on the Kaggle website. The Kaggle test files have no
labels and are never downloaded.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

COMPETITION = "ieee-fraud-detection"
FILES = ("train_transaction.csv", "train_identity.csv")


def download(raw_dir: Path) -> list[Path]:
    """Download and unzip each file into ``raw_dir``. Returns the CSV paths."""
    # Imported here: the kaggle package authenticates on import in some versions.
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    out = []
    for name in FILES:
        target = raw_dir / name
        if target.exists():
            log.info("%s already present, skipping download", target)
            out.append(target)
            continue
        log.info("downloading %s from %s", name, COMPETITION)
        api.competition_download_file(COMPETITION, name, path=str(raw_dir), quiet=False)
        archive = raw_dir / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                z.extract(name, raw_dir)
            archive.unlink()
        if not target.exists():
            raise FileNotFoundError(f"download finished but {target} is missing")
        out.append(target)
    return out
