#!/usr/bin/env bash
#
# Sync this fork (patriceshingu-wq/zero-2-ai-systems-engineer) with the
# instructor's original repo (Here2ServeU/zero-2-ai-systems-engineer).
#
# Run this from inside the zero-2-ai-systems-engineer folder:
#   ./scripts/sync-upstream.sh
#
# What it does, in order:
#   1. Makes sure a remote named "upstream" points at the instructor's repo
#      (adds it the first time; does nothing if it's already there).
#   2. Downloads the instructor's latest commits (doesn't touch your files yet).
#   3. Merges those commits into your local main branch.
#   4. Pushes the updated main back up to your fork on GitHub.

set -euo pipefail

UPSTREAM_URL="https://github.com/Here2ServeU/zero-2-ai-systems-engineer.git"

if ! git remote | grep -qx "upstream"; then
  echo "Adding 'upstream' remote -> $UPSTREAM_URL"
  git remote add upstream "$UPSTREAM_URL"
fi

echo "Fetching upstream..."
git fetch upstream

echo "Switching to main..."
git checkout main

echo "Merging upstream/main into main..."
git merge upstream/main

echo "Pushing updated main to your fork (origin)..."
git push origin main

echo "Done — your fork is up to date with the instructor's repo."
