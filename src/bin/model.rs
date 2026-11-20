//! rs-whisper-model <model.bin> [filters_out.f32]: prints a fixed-format summary.
use rs_whisper::model::load;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("read");
    let m = load(&bytes).expect("load");
    let h = &m.hparams;
    println!(
        "hparams {} {} {} {} {} {} {} {} {} {} {}",
        h.n_vocab, h.n_audio_ctx, h.n_audio_state, h.n_audio_head, h.n_audio_layer,
        h.n_text_ctx, h.n_text_state, h.n_text_head, h.n_text_layer, h.n_mels, h.ftype
    );
    println!(
        "filters {} {} sum={:.6}",
        m.n_mel,
        m.n_fft,
        m.filters.iter().map(|&x| x as f64).sum::<f64>()
    );
    let vb: usize = m.vocab.iter().map(|v| v.len()).sum();
    println!("vocab {} bytes={}", m.vocab.len(), vb);
    println!("tensors {}", m.tensors.len());
    for t in &m.tensors {
        println!("T {} {:?} {} {}", t.name, t.dims, t.ttype, t.nbytes);
    }
    if let Some(o) = a.get(2) {
        let b: Vec<u8> = m.filters.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(o, b).unwrap();
    }
}
