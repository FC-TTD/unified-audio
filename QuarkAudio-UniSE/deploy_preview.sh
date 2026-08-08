#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs portal
docker build --network host -f Dockerfile.preview -t quarkaudio-unise-preview:edge .
docker compose -f compose.preview.yaml up -d
docker compose -f compose.preview.yaml ps
