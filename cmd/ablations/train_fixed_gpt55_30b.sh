#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec "${SCRIPT_DIR}/../games/train_fixed_gpt55_30b.sh" "$@"
