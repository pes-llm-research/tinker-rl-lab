#!/usr/bin/env bash
# Reviewer-response scorer.
# STDOUT: single integer = addressed_count
# STDERR: human-readable breakdown + severity-weighted score
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YAML="$REPO_ROOT/paper/reviewer_points.yaml"
PAPER_DIR="$REPO_ROOT/paper"
EXTRACT="$REPO_ROOT/scripts/_reviewer_points_extract.py"

if [ ! -f "$YAML" ]; then
  echo "ERROR: missing $YAML" >&2
  exit 2
fi
if [ ! -f "$EXTRACT" ]; then
  echo "ERROR: missing $EXTRACT" >&2
  exit 2
fi

MARKERS_FILE="$(mktemp)"
TEX_BLOB="$(mktemp)"
trap 'rm -f "$MARKERS_FILE" "$TEX_BLOB"' EXIT

python3 "$EXTRACT" "$YAML" > "$MARKERS_FILE"

find "$PAPER_DIR" -type f -name '*.tex' -print0 \
  | xargs -0 cat 2>/dev/null > "$TEX_BLOB" || true

ADDRESSED=0
TOTAL=0
WEIGHTED=0
WEIGHTED_MAX=0
DETAILS=""

sev_weight() {
  case "$1" in
    critical) echo 3 ;;
    high)     echo 2 ;;
    medium)   echo 1 ;;
    *)        echo 1 ;;
  esac
}

while IFS='|' read -r id marker sev; do
  [ -z "${marker:-}" ] && continue
  TOTAL=$((TOTAL+1))
  w=$(sev_weight "${sev:-medium}")
  WEIGHTED_MAX=$((WEIGHTED_MAX + w))
  if grep -qF "$marker" "$TEX_BLOB"; then
    ADDRESSED=$((ADDRESSED+1))
    WEIGHTED=$((WEIGHTED + w))
    DETAILS="${DETAILS}
  [+] ${id}  (${sev})  ADDRESSED"
  else
    DETAILS="${DETAILS}
  [ ] ${id}  (${sev})"
  fi
done < "$MARKERS_FILE"

{
  echo "=== reviewer_response_score ==="
  echo "addressed: $ADDRESSED / $TOTAL"
  echo "severity-weighted: $WEIGHTED / $WEIGHTED_MAX"
  printf '%s\n' "$DETAILS"
} >&2

echo "$ADDRESSED"
