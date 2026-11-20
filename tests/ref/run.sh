#!/bin/bash
# WSL で実行: C++基準版とRust版を同じ入力で動かして比較する。使い方: run.sh <jfk.wav>
set -e; D=$(mktemp -d); cd "$D"
python3 /mnt/f/rs-whisper/tests/ref/gen.py "$1"
g++ -O2 -std=c++17 -o ref /mnt/f/rs-whisper/tests/ref/ref.cpp
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release -q --manifest-path /mnt/f/rs-whisper/Cargo.toml --target-dir $HOME/rs-whisper-target
./ref samples.f32 filters.f32 80 cpp.f32
$HOME/rs-whisper-target/release/rs-whisper-mel samples.f32 filters.f32 80 rust.f32
python3 /mnt/f/rs-whisper/tests/ref/cmp.py cpp.f32 rust.f32
