#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
BOARD_APP="/home/arduino/ArduinoApps/WillowRouteLab"

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
  echo "Connect one powered UNO Q by USB-C, then retry." >&2
  exit 1
fi

for relative in \
  sketch/sketch.ino \
  python/main.py python/bonus.py \
  assets/index.html assets/app.js assets/style.css assets/bonus-depths.json; do
  "$BOARD_ADB" push "$APP_DIR/$relative" "$BOARD_APP/$relative" >/dev/null
done

"$BOARD_ADB" shell env TMPDIR=/tmp arduino-app-cli app restart "$BOARD_APP"
"$BOARD_ADB" forward tcp:7000 tcp:7000 >/dev/null
echo "Updated UNO Q app. Open http://127.0.0.1:7000/"
