#!/usr/bin/env bash
# Single local check: lint, format, and tests. CI runs this same script.
set -euo pipefail

cd "$(dirname "$0")/.."

uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
