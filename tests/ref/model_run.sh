#!/bin/bash
# Run in WSL: model_run.sh <ggml model.bin>. Compares a Python reader with the Rust loader,
# on the given model and on a synthetic file with tensors of several types.
set -e
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release -q --manifest-path /mnt/f/rs-whisper/Cargo.toml --target-dir $HOME/rs-whisper-target
B=$HOME/rs-whisper-target/release/rs-whisper-model
python3 /mnt/f/rs-whisper/tests/ref/mk_synth.py "$1" /tmp/synth.bin
for f in "$1" /tmp/synth.bin; do
  python3 /mnt/f/rs-whisper/tests/ref/model_ref.py "$f" > /tmp/m_py.txt
  $B "$f" /tmp/real_filters.f32 > /tmp/m_rs.txt
  diff /tmp/m_py.txt /tmp/m_rs.txt && echo "IDENTICAL ($(wc -l < /tmp/m_rs.txt) lines): $f"
done
