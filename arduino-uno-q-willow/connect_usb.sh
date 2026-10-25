#!/usr/bin/env bash
set -euo pipefail

if command -v adb >/dev/null 2>&1; then
  BOARD_ADB="$(command -v adb)"
else
  BOARD_ADB=""
  for candidate in "$HOME"/Library/Arduino15/packages/arduino/tools/adb/*/adb; do
    if [[ -x "$candidate" ]]; then BOARD_ADB="$candidate"; fi
  done
fi

if [[ -z "$BOARD_ADB" ]]; then
  echo "ADB was not found. Install Arduino IDE or Arduino App Lab first." >&2
  exit 1
fi

if [[ "$("$BOARD_ADB" devices | awk '$2 == "device" { count++ } END { print count+0 }')" != "1" ]]; then
  echo "Connect one powered UNO Q by USB-C, then run this command again." >&2
  "$BOARD_ADB" devices -l >&2
  exit 1
fi

if ! "$BOARD_ADB" shell arduino-app-cli app list | grep -Eq '^user:WillowRouteLab[[:space:]].*[[:space:]]running[[:space:]]'; then
  echo "Starting Willow Route Lab on the UNO Q..."
  "$BOARD_ADB" shell env TMPDIR=/tmp arduino-app-cli app start /home/arduino/ArduinoApps/WillowRouteLab
fi

"$BOARD_ADB" forward tcp:7000 tcp:7000 >/dev/null
echo "UNO Q connected. Open http://127.0.0.1:7000/"
