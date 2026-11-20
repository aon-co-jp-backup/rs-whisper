#!/bin/bash
# Run in WSL: transcribe_run.sh <ggml model.bin> <audio.wav>
# Compares the Rust greedy transcript with whisper.cpp (whisper-cli, beam 1, no timestamps, English).
set -e
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release -q --manifest-path /mnt/f/rs-whisper/Cargo.toml --target-dir $HOME/rs-whisper-target
D=$(mktemp -d); cd "$D"
python3 /mnt/f/rs-whisper/tests/ref/gen.py "$2"
time $HOME/rs-whisper-target/release/rs-whisper-transcribe "$1" samples.f32 > rust.txt 2> rust.err
cat rust.err
/root/wcpp-build/bin/whisper-cli -m "$1" -f "$2" -l en -nt -bs 1 -bo 1 -nf -t 4 > cpp.raw 2> cpp.err || true
sed 's/^ *//;s/ *$//' cpp.raw | tr -d '\n' > cpp.txt
sed 's/^ *//;s/ *$//' rust.txt | tr -d '\n' > rust2.txt
echo "RUST: $(cat rust2.txt)"
echo "CPP : $(cat cpp.txt)"
if cmp -s cpp.txt rust2.txt; then echo "TRANSCRIPT IDENTICAL"; else echo "TRANSCRIPT DIFFERS"; fi
