#!/bin/bash
# Run in WSL after model_run.sh: log-mel with the REAL mel filters read from the model file.
# Usage: mel_real.sh <jfk.wav>
set -e
D=$(mktemp -d); cd "$D"
export PATH=$HOME/.cargo/bin:$PATH
python3 /mnt/f/rs-whisper/tests/ref/gen.py "$1"
cp /tmp/real_filters.f32 filters.f32
g++ -O2 -std=c++17 -o ref /mnt/f/rs-whisper/tests/ref/ref.cpp
./ref samples.f32 filters.f32 80 cpp.f32
$HOME/rs-whisper-target/release/rs-whisper-mel samples.f32 filters.f32 80 rust.f32
python3 /mnt/f/rs-whisper/tests/ref/cmp.py cpp.f32 rust.f32
