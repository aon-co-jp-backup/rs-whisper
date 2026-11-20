# rs-whisper

**日本語**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp)(MIT)を、段階的に Rust へ移植するリポジトリです。いまできているのは段階1「log-mel スペクトログラム(dft/fft/メル変換)」で、C++ 版と同じ入力(`jfk.wav`)で **32.8万個の値がすべてビット単位で一致**することを確認済みです(`tests/ref/run.sh`)。モデル読み込み・エンコーダ・デコーダは未移植です(計画は [PORTING.md](PORTING.md))。

**English**: A step-by-step Rust port of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT). Stage 1, the log-mel spectrogram (dft/fft/mel), is done and verified: on the same input (`jfk.wav`) **all 328,000 values are bit-identical** to the C++ version (`tests/ref/run.sh`). Model loading, encoder and decoder are not ported yet (plan in [PORTING.md](PORTING.md)).

License: MIT (derived from whisper.cpp, (c) The ggml authors).
