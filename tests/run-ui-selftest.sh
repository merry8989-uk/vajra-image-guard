#!/usr/bin/env bash
#
# Drives the real UI in headless Chromium and reports pass/fail.
#
#   bash tests/run-ui-selftest.sh [path-to-chromium]
#
# The driver page is copied into docs/ so it loads the site with the same relative
# paths a user would, then deleted again.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
CHROME="${1:-${CHROME:-chromium}}"

if ! command -v "$CHROME" >/dev/null 2>&1; then
  echo "error: $CHROME not found; pass the browser binary as the first argument" >&2
  exit 1
fi

cp tests/ui.driver.html docs/_selftest.html
trap 'rm -f docs/_selftest.html' EXIT

DOM="$(mktemp)"
"$CHROME" --headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage \
  --allow-file-access-from-files --virtual-time-budget=900000 \
  --dump-dom "file://$ROOT/docs/_selftest.html" > "$DOM" 2>/dev/null

sed -n '/<pre id="log">/,/<\/pre>/p' "$DOM" | sed 's/<[^>]*>//g'
TITLE="$(grep -o '<title>[^<]*</title>' "$DOM" | sed 's/<[^>]*>//g')"
rm -f "$DOM"

echo
if [ "$TITLE" = "PASS" ]; then
  echo "ui self-test: PASS"
else
  echo "ui self-test: FAIL"
  exit 1
fi
