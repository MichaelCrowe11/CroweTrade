#!/usr/bin/env bash
# Sign, notarize, staple, and Gatekeeper-check Electron Builder DMGs.
#
# Notarization auth — one of two paths:
#   A) App Store Connect API key (preferred):
#      APPLE_API_KEY_PATH     path to the .p8 private key file
#      APPLE_API_KEY_ID       10-char key ID (e.g. Y83U4S42GU)
#      APPLE_API_ISSUER_ID    UUID issuer ID from App Store Connect
#   B) Apple ID + app-specific password:
#      APPLE_ID
#      APPLE_APP_SPECIFIC_PASSWORD or APPLE_ID_PASSWORD
#
# Either path also requires APPLE_TEAM_ID.
#
# Signing uses the first matching Developer ID Application identity in the
# active keychain. If CSC_LINK and CSC_KEY_PASSWORD are present, this script
# imports that p12/base64 certificate into a temporary keychain first.

set -euo pipefail

: "${APPLE_TEAM_ID:?set APPLE_TEAM_ID}"

USE_API_KEY=false
if [[ -n "${APPLE_API_KEY_PATH:-}" && -n "${APPLE_API_KEY_ID:-}" ]]; then
  USE_API_KEY=true
  if [[ ! -f "$APPLE_API_KEY_PATH" ]]; then
    echo "APPLE_API_KEY_PATH does not exist: $APPLE_API_KEY_PATH" >&2
    exit 1
  fi
  echo "[notarize] using App Store Connect API key (key-id: $APPLE_API_KEY_ID)"
else
  : "${APPLE_ID:?set APPLE_ID (or APPLE_API_KEY_PATH + APPLE_API_KEY_ID)}"
  APPLE_ID_PASSWORD="${APPLE_ID_PASSWORD:-${APPLE_APP_SPECIFIC_PASSWORD:-}}"
  if [[ -z "$APPLE_ID_PASSWORD" ]]; then
    echo "set APPLE_APP_SPECIFIC_PASSWORD or APPLE_ID_PASSWORD (or use APPLE_API_KEY_PATH)" >&2
    exit 1
  fi
  echo "[notarize] using Apple ID + app-specific password"
fi

DMG_DIR="${1:-release}"
CSC_NAME="${CSC_NAME:-Michael Crowe (6QLMV9UCPP)}"
TEMP_KEYCHAIN=""
TEMP_CERT=""

cleanup() {
  if [[ -n "$TEMP_KEYCHAIN" ]]; then
    security delete-keychain "$TEMP_KEYCHAIN" >/dev/null 2>&1 || true
  fi
  if [[ -n "$TEMP_CERT" ]]; then
    rm -f "$TEMP_CERT"
  fi
}
trap cleanup EXIT

find_identity() {
  local keychain="${1:-}"
  if [[ -n "$keychain" ]]; then
    security find-identity -v -p codesigning "$keychain" 2>/dev/null
  else
    security find-identity -v -p codesigning 2>/dev/null
  fi | awk -v name="$CSC_NAME" '
    /Developer ID Application/ {
      if (name == "" || index($0, name) > 0) {
        print $2
        exit
      }
    }
  '
}

decode_csc_link() {
  local link="$1"
  local output="$2"

  if [[ -f "$link" ]]; then
    cp "$link" "$output"
    return
  fi

  if printf '%s' "$link" | base64 --decode >"$output" 2>/dev/null; then
    return
  fi

  printf '%s' "$link" | base64 -D >"$output"
}

import_csc_link() {
  if [[ -z "${CSC_LINK:-}" || -z "${CSC_KEY_PASSWORD:-}" ]]; then
    return
  fi

  local tmp_base keychain_password current_keychains keychain
  tmp_base="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
  TEMP_CERT="$(mktemp "$tmp_base/crowetrade-certificate.XXXXXX.p12")"
  TEMP_KEYCHAIN="$(mktemp -u "$tmp_base/crowetrade-signing.XXXXXX.keychain-db")"
  keychain_password="$(uuidgen 2>/dev/null || date +%s)"

  decode_csc_link "$CSC_LINK" "$TEMP_CERT"

  security create-keychain -p "$keychain_password" "$TEMP_KEYCHAIN"
  security set-keychain-settings -lut 21600 "$TEMP_KEYCHAIN"
  security unlock-keychain -p "$keychain_password" "$TEMP_KEYCHAIN"
  security import "$TEMP_CERT" \
    -k "$TEMP_KEYCHAIN" \
    -P "$CSC_KEY_PASSWORD" \
    -T /usr/bin/codesign \
    -T /usr/bin/productbuild

  current_keychains=()
  while IFS= read -r keychain; do
    keychain="${keychain//\"/}"
    keychain="${keychain#"${keychain%%[![:space:]]*}"}"
    keychain="${keychain%"${keychain##*[![:space:]]}"}"
    if [[ -n "$keychain" && "$keychain" != "$TEMP_KEYCHAIN" ]]; then
      current_keychains+=("$keychain")
    fi
  done < <(security list-keychains -d user)

  security list-keychains -d user -s "$TEMP_KEYCHAIN" "${current_keychains[@]}"
  security set-key-partition-list \
    -S apple-tool:,apple:,codesign: \
    -s \
    -k "$keychain_password" \
    "$TEMP_KEYCHAIN"
}

shopt -s nullglob
dmgs=("$DMG_DIR"/*.dmg)
if [[ ${#dmgs[@]} -eq 0 ]]; then
  echo "no .dmg files found in $DMG_DIR" >&2
  exit 1
fi

identity="$(find_identity)"
if [[ -z "$identity" ]]; then
  echo "[notarize] no active Developer ID identity found; importing CSC_LINK"
  import_csc_link
  identity="$(find_identity "$TEMP_KEYCHAIN")"
fi

if [[ -z "$identity" ]]; then
  echo "no Developer ID Application identity found for $CSC_NAME" >&2
  exit 1
fi

echo "[notarize] using signing identity: $identity"

for dmg in "${dmgs[@]}"; do
  echo
  echo "=== $dmg ==="
  codesign --force --sign "$identity" "$dmg"
  if [[ "$USE_API_KEY" == "true" ]]; then
    notary_args=(
      --key "$APPLE_API_KEY_PATH"
      --key-id "$APPLE_API_KEY_ID"
      --team-id "$APPLE_TEAM_ID"
    )
    if [[ -n "${APPLE_API_ISSUER_ID:-}" ]]; then
      notary_args+=(--issuer "$APPLE_API_ISSUER_ID")
    fi
    xcrun notarytool submit "$dmg" "${notary_args[@]}" --wait
  else
    xcrun notarytool submit "$dmg" \
      --apple-id "$APPLE_ID" \
      --password "$APPLE_ID_PASSWORD" \
      --team-id "$APPLE_TEAM_ID" \
      --wait
  fi
  xcrun stapler staple "$dmg"
  xcrun stapler validate "$dmg"
  spctl -a -vv -t install "$dmg" >/dev/null
done

echo
echo "[notarize] all DMGs signed, notarized, stapled, and Gatekeeper accepted."
