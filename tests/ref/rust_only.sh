#!/bin/bash
# Run in WSL: rust_only.sh <ggml model.bin> <audio.wav>  (Rust transcript only, with timing)
D=$(mktemp -d); cd "$D"
python3 /mnt/f/rs-whisper/tests/ref/gen.py "$2"
time $HOME/rs-whisper-target/release/rs-whisper-transcribe "$1" samples.f32 enc.f32
