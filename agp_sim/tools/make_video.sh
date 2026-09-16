#!/usr/bin/env bash
# make_video.sh <run_dir> [fps]  -> <run_dir>/video.mp4 from the simulator child's 3-camera mosaics (one every 4 sim steps)
set -euo pipefail
R="$1"; FPS="${2:-15}"
FR="$R/sim/sim/video-frames"
[ -d "$FR" ] || { echo "no video-frames under $R"; exit 2; }
ffmpeg -y -loglevel error -framerate "$FPS" -pattern_type glob -i "$FR/*.png" -c:v libx264 -pix_fmt yuv420p -vf "scale=768:256" "$R/video.mp4"
ls -la "$R/video.mp4"
