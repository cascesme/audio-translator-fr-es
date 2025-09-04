#!/usr/bin/env bash
set -e

if [[ $# -eq 0 ]]; then
  python /app/pipeline.py --help
else
  python /app/pipeline.py "$@"
fi
