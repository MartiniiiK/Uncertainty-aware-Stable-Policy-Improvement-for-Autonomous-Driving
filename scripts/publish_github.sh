#!/usr/bin/env bash
# Run ONLY in the extracted publication package, not the original server workspace.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OWNER=${OWNER:-MartiniiiK}
REPO=${REPO:-Uncertainty-aware-Stable-Policy-Improvement-for-Autonomous-Driving}
VISIBILITY=${VISIBILITY:-private}
case "$VISIBILITY" in private|public) ;; *) echo 'VISIBILITY must be private or public'; exit 1;; esac
command -v gh >/dev/null || { echo 'Install GitHub CLI (https://cli.github.com/) first'; exit 1; }
gh auth status --hostname github.com
LOGIN=$(gh api user --jq .login)
if [ "$LOGIN" != "$OWNER" ]; then echo "Authenticated as $LOGIN, expected $OWNER"; exit 1; fi
if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then echo 'Repository exists; no changes made. Use a separately reviewed normal push or new name.'; exit 1; fi
if [ -d .git ]; then echo 'Existing local git directory; refusing automatic publication'; exit 1; fi
git init -b main
# Explicit allowlist avoids accidentally committing data or credentials.
git add README.md LICENSE NOTICE .gitignore *.py configs scripts docs tests results dashboard upstream_snapshot
# Refuse common accidental binary/secrets patterns even if added under allowlisted dirs.
if git diff --cached --name-only | grep -Ei '(^|/)(\.env($|\.)|id_rsa|id_ed25519)|\.(pt|pth|h5|hdf5|pem|key|zip|log)$'; then echo 'Potential sensitive/large file staged; stop and inspect'; exit 1; fi
git diff --cached --stat
git commit -m 'Initial release: USPI research implementation and EasyCarla reproducibility'
gh repo create "$OWNER/$REPO" --"$VISIBILITY" --description 'Uncertainty-aware stable policy improvement for autonomous driving in EasyCarla / CARLA' --source=. --remote=origin --push
gh repo view "$OWNER/$REPO" --json url,visibility
