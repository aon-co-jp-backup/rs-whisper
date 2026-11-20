# rs-whisper

**日本語**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp)(MIT)を、段階的に Rust へ移植するリポジトリです。いまできているのは段階1「log-mel スペクトログラム」と段階2「ggml モデルの読み込み」で、C++ 版と同じ入力(`jfk.wav`)で **32.8万個の値がすべてビット単位で一致**することを確認済みです(`tests/ref/run.sh`)。エンコーダ・デコーダは未移植です(計画は [PORTING.md](PORTING.md))。

**English**: A step-by-step Rust port of [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT). Stages 1 (log-mel) and 2 (ggml model loader, verified against an independent Python reader and with the real mel filters) are done; stage 1 is verified: on the same input (`jfk.wav`) **all 328,000 values are bit-identical** to the C++ version (`tests/ref/run.sh`). Encoder and decoder are not ported yet (plan in [PORTING.md](PORTING.md)).

License: MIT (derived from whisper.cpp, (c) The ggml authors).
