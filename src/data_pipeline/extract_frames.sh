#!/bin/bash

VIDEO=$1
OUTDIR=$2

if [[ -z "$VIDEO" || -z "$OUTDIR" ]]; then
    echo "Usage: $0 <video_file> <output_dir>"
    exit 1
fi

mkdir -p "$OUTDIR"
ffmpeg -i "$VIDEO" -vf fps=1 "$OUTDIR/frame_%04d.png"
echo "📸 Frames extracted to $OUTDIR"
