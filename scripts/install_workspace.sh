#!/usr/bin/env bash
# Creates a NEW EasyCarla workspace, leaving existing experiments untouched.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${1:?Usage: bash scripts/install_workspace.sh /absolute/path/to/new/EasyCarla-RL}
if [ -e "$DEST" ]; then echo "Destination exists; choose an empty/new path" >&2; exit 1; fi
mkdir -p "$(dirname "$DEST")"
git clone https://github.com/silverwingsbot/EasyCarla-RL.git "$DEST"
if [ -n "${EASYCARLA_REF:-}" ]; then git -C "$DEST" checkout --detach "$EASYCARLA_REF"; fi
git -C "$DEST" rev-parse HEAD > "$DEST/upstream_commit.txt"
# Restore exact agent/env files archived in the historical run; keep upstream utils/setup.
cp -a "$REPO/upstream_snapshot/." "$DEST/"
cp "$REPO"/*.py "$DEST/"
cp -a "$REPO/configs" "$REPO/tests" "$REPO/scripts" "$DEST/"
cp "$REPO/dashboard/"*.py "$REPO/dashboard/research_monitor.html" "$DEST/"
mkdir -p "$DEST/data" "$DEST/da_runs" "$DEST/logs" "$DEST/demo_runs"
# Preserve the upstream license and provide the local modification notice.
cp "$REPO/NOTICE" "$DEST/USPI_NOTICE"
cd "$DEST"
python -m pip install -e . --no-deps --no-build-isolation
printf 'Workspace ready: %s\n' "$DEST"
