#!/bin/bash
# Run in WSL: builds the whisper.cpp reference CLI (CPU, no native flags) into /root/wcpp-build.
set -e
cmake -S /mnt/f/_work/whisper.cpp -B /root/wcpp-build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DWHISPER_BUILD_TESTS=OFF > /tmp/cmake.log 2>&1
cmake --build /root/wcpp-build -j8 --target whisper-cli > /tmp/build.log 2>&1
echo BUILD-OK
