//! rs-whisper-transcribe <ggml model.bin> <samples.f32>: greedy English transcription (CPU).
use rs_whisper::{infer::Whisper, log_mel_spectrogram, model};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("model");
    let m = model::load(&bytes).expect("load");
    let samples: Vec<f32> = std::fs::read(&a[2])
        .expect("samples")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let n_mel = m.n_mel as usize;
    let (mel, n_len) = log_mel_spectrogram(&samples, &m.filters, n_mel);
    let w = Whisper::new(&m, &bytes);
    let t0 = std::time::Instant::now();
    let enc = w.encode(&mel, n_mel, n_len);
    eprintln!("encode: {:.1}s", t0.elapsed().as_secs_f32());
    if let Some(p) = a.get(3) {
        let b: Vec<u8> = enc.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(p, b).unwrap();
    }
    let (text, ids) = w.transcribe_greedy(&enc, 224);
    eprintln!("tokens: {:?}", ids);
    println!("{}", text);
}
