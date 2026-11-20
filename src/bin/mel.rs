//! 使い方: rs-whisper-mel <samples.f32> <filters.f32> <n_mel> <out.f32>(いずれもリトルエンディアン raw)
use std::fs;
fn rd(p: &str) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (mel, _n) = rs_whisper::log_mel_spectrogram(&rd(&a[1]), &rd(&a[2]), a[3].parse().unwrap());
    let mut o = Vec::new();
    for v in mel { o.extend_from_slice(&v.to_le_bytes()); }
    fs::write(&a[4], o).unwrap();
}
