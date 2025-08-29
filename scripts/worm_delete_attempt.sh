#!/usr/bin/env bash
set -euo pipefail
BUCKET="${1:-}"
if [[ -z "${BUCKET}" ]]; then
	echo "Usage: $0 <bucket-name>" >&2
	exit 1
fi
KEY="$(date +%F)/audit.parquet"
echo "Attempting delete on s3://${BUCKET}/${KEY} (should fail if WORM)"
set +e
aws s3api delete-object --bucket "$BUCKET" --key "$KEY"
rc=$?
set -e
if [[ $rc -eq 0 ]]; then
	echo "DELETE SUCCEEDED — WORM MISCONFIGURED" >&2
	exit 1
else
	echo "WORM OK: delete blocked"
fi

