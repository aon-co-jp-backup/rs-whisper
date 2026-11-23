//! rs-whisper-stream <ggml model.bin> <samples.f32> [max_seconds]: split 16 kHz f32 audio at silences
//! (energy based) and transcribe each segment with a window fitted to its length. Simulates streaming.
use rs_whisper::{infer::Whisper, log_mel_spectrogram, model};

const SR: usize = 16000;
const FRAME: usize = SR / 50; // 20 ms

/// Segment boundaries (start, end) in samples. A pause of >= 0.3 s ends a segment; a segment never exceeds `max_s`.
fn segments(x: &[f32], max_s: f32, min_s: f32) -> Vec<(usize, usize)> {
    let rms: Vec<f32> = x
        .chunks(FRAME)
        .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let peak = rms.iter().cloned().fold(0.0f32, f32::max);
    let thr = (peak * 0.05).max(0.003);
    let max_f = (max_s * 50.0) as usize;
    let min_f = (min_s * 50.0) as usize;
    let (mut out, mut start, mut quiet) = (Vec::new(), None::<usize>, 0usize);
    for (i, &r) in rms.iter().enumerate() {
        if r >= thr {
            if start.is_none() {
                start = Some(i);
            }
            quiet = 0;
        } else if let Some(s) = start {
            quiet += 1;
            if quiet >= 15 && (i + 1 - quiet - s) >= min_f {
                out.push((s, i + 1 - quiet));
                start = None;
                quiet = 0;
            }
        }
        if let Some(s) = start {
            if i + 1 - s >= max_f {
                out.push((s, i + 1));
                start = None;
                quiet = 0;
            }
        }
    }
    if let Some(s) = start {
        out.push((s, rms.len()));
    }
    // Merge neighbours: gaps under 1 s are joined while the segment stays within max_s (very short pieces recognise badly).
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in out {
        if let Some(l) = merged.last_mut() {
            if a - l.1 < 50 && b - l.0 <= max_f {
                l.1 = b;
                continue;
            }
        }
        merged.push((a, b));
    }
    merged.into_iter().map(|(a, b)| (a * FRAME, (b * FRAME).min(x.len()))).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).expect("model");
    let m = model::load(&bytes).expect("load");
    let x: Vec<f32> = std::fs::read(&a[2])
        .expect("samples")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let max_s: f32 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(8.0);
    let min_s: f32 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(2.5);
    let pad = (0.3 * SR as f32) as usize; // overlap with the neighbours; duplicates are merged afterwards
    let mut merged: Vec<String> = Vec::new();
    let n_mel = m.n_mel as usize;
    let mut w = Whisper::new(&m, &bytes);
    let total = std::time::Instant::now();
    for (s0, e0) in segments(&x, max_s, min_s) {
        let (s, e) = (s0.saturating_sub(pad), (e0 + pad).min(x.len()));
        let seg = &x[s..e];
        let secs = seg.len() as f32 / SR as f32;
        let (mel, n_len) = log_mel_spectrogram(seg, &m.filters, n_mel);
        w.audio_ctx = ((secs * 50.0).ceil() as usize + 50).clamp(100, 1500); // length + 1 s margin
        let t0 = std::time::Instant::now();
        let enc = w.encode(&mel, n_mel, n_len);
        let (text, _) = w.transcribe_greedy(&enc, (secs * 6.0) as usize + 8);
        let mut words: Vec<String> = text.split_whitespace().map(String::from).collect();
        rs_whisper::merge::drop_dangling(&mut words);
        let dropped = rs_whisper::merge::append_dedup(&mut merged, &words);
        println!("[{:5.2}-{:5.2}s] ({:.1}s to process, window {} ) {}", s as f32 / SR as f32, e as f32 / SR as f32, t0.elapsed().as_secs_f32(), w.audio_ctx, text.trim());
        if dropped > 0 {
            println!("    (overlap: dropped {} repeated word(s))", dropped);
        }
    }
    println!("MERGED: {}", merged.join(" "));
    eprintln!("total {:.1}s for {:.1}s of audio", total.elapsed().as_secs_f32(), x.len() as f32 / SR as f32);
}
