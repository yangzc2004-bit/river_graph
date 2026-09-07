#!/usr/bin/env bash
# Download WQP 'Organic carbon' results per hydrologic unit code.
# Usage: bash scripts/download_wqp_by_huc.sh data/raw/mrb_huc4.txt
# Resumable: skips codes whose output file is already valid.
set -u
LIST=${1:?usage: download_wqp_by_huc.sh <file-with-huc-codes>}
OUT=data/raw/wqp_by_huc
mkdir -p "$OUT"

valid() {
  local f=$1
  [ -s "$f" ] || return 1
  head -1 "$f" | grep -q "^Org_Identifier" || return 1
  ! grep -q "INCOMPLETE DATA\|overloaded\|Service unavailable" "$f" || return 1
  return 0
}

FAIL=0
while read -r code; do
  [ -z "$code" ] && continue
  f="$OUT/huc_$code.csv"
  if valid "$f"; then echo "huc $code already done, skip"; continue; fi
  ok=0
  for attempt in 1 2 3 4; do
    curl -s --max-time 280 \
      "https://www.waterqualitydata.us/wqx3/Result/search?huc=$code&providers=NWIS&characteristicName=Organic%20carbon&dataProfile=narrow&mimeType=csv" \
      -o "$f"
    if valid "$f"; then
      echo "huc $code OK ($(($(wc -l < "$f") - 1)) rows)"
      ok=1; break
    fi
    echo "huc $code bad response, retry $attempt"
    sleep $((attempt * 20))
  done
  [ $ok -eq 0 ] && { echo "huc $code FAILED"; FAIL=1; }
  sleep 3
done < "$LIST"
echo "DONE (fail=$FAIL)"
