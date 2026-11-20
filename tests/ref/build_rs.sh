#!/bin/bash
# Run in WSL: builds rs-whisper in release mode and shows the last lines of output.
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release --manifest-path /mnt/f/rs-whisper/Cargo.toml --target-dir $HOME/rs-whisper-target 2>&1 | tail -30
