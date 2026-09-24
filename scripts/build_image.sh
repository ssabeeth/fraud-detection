#!/usr/bin/env bash
# Build the scoring image with a given LightGBM bundle.
#   scripts/build_image.sh [MODEL_DIR] [TAG]
# MODEL_DIR defaults to data/models/lightgbm. CI passes a bundle trained on the synthetic
# fixtures, so the image it publishes contains nothing learnt from the competition data.
set -euo pipefail
model_dir="${1:-data/models/lightgbm}"
tag="${2:-fraud-api:local}"
rm -rf docker/model
mkdir -p docker/model
cp "$model_dir"/{model.lgb,preprocessor.json,calibrator.json,meta.json} docker/model/
docker build -f docker/Dockerfile -t "$tag" .
