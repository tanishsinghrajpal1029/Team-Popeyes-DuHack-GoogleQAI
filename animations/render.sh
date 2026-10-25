#!/usr/bin/env bash
# Render every scene into figures/anim/:
#   <Scene>.mp4            1080p60, for the talk and the video
#   sections/<Scene>_*.mp4 one clip per beat (one slide click each)
#   <Scene>.gif            480p15, for the README
# Data comes from animations/data/*.json; regenerate it with `.venv/bin/python animations/prep.py`.
set -euo pipefail
cd "$(dirname "$0")"
MANIM=../.venv/bin/manim
OUT=../figures/anim
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(DickeIntro ScoreTail WTree Synthesis Placement)
mkdir -p "$OUT/sections"
for s in "${SCENES[@]}"; do
  $MANIM -qh --save_sections --media_dir media scenes.py "$s"
  cp "media/videos/scenes/1080p60/$s.mp4" "$OUT/$s.mp4"
  cp media/videos/scenes/1080p60/sections/"${s}"_*.mp4 "$OUT/sections/"
  $MANIM -ql --format=gif --media_dir media scenes.py "$s"
  cp "$(ls -t media/videos/scenes/480p15/"${s}"*.gif | head -1)" "$OUT/$s.gif"
done
