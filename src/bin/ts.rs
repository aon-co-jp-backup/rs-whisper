//! rs-whisper-ts <ggml model.bin> <samples.f32>: print timestamped segments.
use rs_whisper::{infer::Whisper, log_mel_spectrogram, model};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("model");
    let m = model::load(&bytes).expect("load");
    let x: Vec<f32> = std::fs::read(&a[2]).expect("samples").chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let n_mel = m.n_mel as usize;
    let (mel, n_len) = log_mel_spectrogram(&x, &m.filters, n_mel);
    let mut w = Whisper::new(&m, &bytes);
    let secs = x.len() as f32 / 16000.0;
    w.audio_ctx = ((secs * 50.0).ceil() as usize + 50).clamp(100, 1500);
    let enc = w.encode(&mel, n_mel, n_len);
    for (s, e, t) in w.transcribe_segments(&enc, 224) {
        println!("[{:.2} -> {:.2}] {}", s, e, t);
    }
    let (txt, ids) = w.transcribe_greedy(&enc, 224);
    println!("{txt}");
    let tm = w.token_times(&enc, &ids);
    for (id, t0) in ids.iter().zip(tm.iter()) {
        println!("{:6.2} {:?}", t0, String::from_utf8_lossy(&w.vocab[*id as usize]));
    }
}
