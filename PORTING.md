# rs-whisper 移植計画 / Porting plan (2026-11-20)

## 試験のしかた / How to verify
WSL(Ubuntu-24.04)で `bash tests/ref/run.sh <jfk.wav>`。元の関数を取り出した `tests/ref/ref.cpp`(g++ -O2)と Rust 版に同じ入力・同じフィルタを渡し、出力を `cmp.py` で比較する。2026-11-20: values=328000 exact=328000 max_abs_diff=0。
注意: 一致は glibc の sinf/cosf を使う環境(Linux)での結果。メルフィルタは試験用の合成値(本物のモデル内のフィルタでの比較は段階2)。

## whisper.cpp の規模(取得時点 fc89e85)
`src/` 約1.6万行(うち `whisper.cpp` 9395行)、`ggml/src` 約42.8万行(多くはバックエンド: CUDA/Metal/Vulkan 等)、`examples/server` 約1.2万行、`examples/cli` 約1400行。

## 段階 / Stages
1. ✅ log-mel(dft/fft/メル)。
2. モデル(ggml 形式)読み込み: ハイパーパラメータ・語彙・メルフィルタ・テンソル。実モデル `ggml-tiny` で、フィルタとテンソルの値の一致を確認。
3. 推論の中身: 既存 Rust 資産を使う。`open-cuda/crates/open-cuda-whisper`(921行、MIT/Apache 相当をワークスペースのライセンスで要確認)、`open-cuda`/`open-cpu` の行列演算。エンコーダ出力を C++ 版と許容誤差内で比較。
4. デコーダ・トークナイザ(C++ は `std::regex` を使う。Rust で同じ分割になるかを試験)。
5. `examples/cli` 相当、次に `examples/server`(REST は作らず Connect で)。
6. ggml のバックエンド(GPU)は移植しない。`open-cuda` 系を使う。

## C/C++ が主のリポジトリ(F:\ 直下の機械的な洗い出し、ベンダー同梱物は未選別)
- `whisper.cpp`(上記)。
- `open-media`: C/C++ 系 398 ファイル、Fortran(.f90)124 ファイル。同梱の第三者コードの可能性が高く、**自作分かの確認が必要**。
- `make-disk`・`open-audio-sr`・`open-music-llm`: 9千超の C/C++ ファイル(第三者の同梱 or 仮想環境の可能性、**未確認**)。
- ほか(`MFastAPI`・`RPoRea`・`open-web-server` など)は C/C++ が 1〜6 ファイル(橋渡し・例)。
- VPS(`/root/repository/*`)は未調査(`pl2pl c2rust` は単純な C/C++ 向けで、whisper 規模は対象外)。

## 他の言語(2026-11-20 追加指示。ファイル数の目安のみ、方針は未決)
- Python: `open-network-amp` 155、`open-av-amp` 99、`2FastAPI` 85、`runo` 97、`open-media` 5037(多くは同梱物の可能性)。`2FastAPI`/`MFastAPI`/`RMojo` は Python 向け製品そのものなので、移植ではなく対象外にするか要判断。
- PHP: `audiocafe-tokyo` 61、`runo` 65、`pl2pl` 29。
- Kotlin/Swift(Android/iOS の画面): 多数。Rust 化は UI の作り直しになるため要判断。
- JavaScript/TypeScript: Node.js を使わない方針に沿い別途(`NODEJS-REMAINING-20261119.md`)。
- Java: `Java_to`・`pl2pl`・`aruaru-IME`・`open-backup`(各数ファイル)。
- Fortran: `open-media` の 124 ファイル(`pl2pl f2rust` あり)。
順序案: 自作かどうかの選別 → 小さい C/Fortran/Java から `pl2pl` で変換し出力一致を試験 → Python/PHP は機能単位で。
