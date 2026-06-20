#!/usr/bin/env bash
# Register a cloned voice on the server.
#
# Engines that support it: omnivoice, pocket, xtts.
# Engines that don't: kitten, kokoro, piper, bark — server returns 400.

set -euo pipefail

VOICE_ID="${1:-viraj}"
AUDIO_FILE="${2:-./Viraj.mp3}"
REF_TEXT="${3:-}"
HOST="${TTS_HOST:-http://localhost:8001}"

curl -X POST "$HOST/v1/voices/upload" \
  -F "voice_id=$VOICE_ID" \
  -F "ref_text=$REF_TEXT" \
  -F "file=@$AUDIO_FILE"
echo
