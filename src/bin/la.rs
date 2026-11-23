//! rs-whisper-la <ggml model.bin> <samples.f32> [step_seconds] [max_buffer_seconds]: LocalAgreement streaming
//! simulation (as in ufal/whisper_streaming): every step the buffer is transcribed again and only the prefix on
//! which two consecutive hypotheses agree is emitted. When the buffer grows beyond max_buffer_seconds, the finalized
//! part is cut off at a silence (word positions are estimated from the share of voiced frames; there are no timestamps).
use rs_whisper::{infer::Whisper, log_mel_spectrogram, model};

const SR: usize = 16000;
const FRAME: usize = SR / 50;

fn norm(w: &str) -> String {
    w.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

/// Frame energies of `x` and the voiced threshold.
fn voiced(x: &[f32]) -> Vec<bool> {
    let rms: Vec<f32> = x.chunks(FRAME).map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt()).collect();
    let peak = rms.iter().cloned().fold(0.0f32, f32::max);
    let thr = (peak * 0.05).max(0.003);
    rms.iter().map(|&r| r >= thr).collect()
}

/// Latest cut position (sample index) inside a >= 0.3 s silence for which the estimated number of words before it
/// does not exceed `finalized`. Returns (sample index, words before it).
fn cut_point(buf: &[f32], n_words: usize, finalized: usize) -> Option<(usize, usize)> {
    let v = voiced(buf);
    let total_voiced = v.iter().filter(|&&b| b).count().max(1);
    let mut best = None;
    let (mut voiced_before, mut quiet_start, mut quiet) = (0usize, 0usize, 0usize);
    for (i, &b) in v.iter().enumerate() {
        if b {
            quiet = 0;
            voiced_before += 1;
        } else {
            if quiet == 0 {
                quiet_start = i;
            }
            quiet += 1;
            if quiet == 15 && voiced_before > 0 {
                let est = (voiced_before as f32 / total_voiced as f32 * n_words as f32).floor() as usize;
                let mid = (quiet_start + 8) * FRAME;
                if est <= finalized && est > 0 {
                    best = Some((mid.min(buf.len()), est));
                }
            }
        }
    }
    best
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
    let step: f32 = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(2.0);
    let max_buf: f32 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(8.0);
    let n_mel = m.n_mel as usize;
    let mut w = Whisper::new(&m, &bytes);
    let total = x.len() as f32 / SR as f32;
    let (mut start, mut prev, mut emitted, mut t, mut busy) = (0usize, Vec::<String>::new(), 0usize, step, 0f32);
    let mut all: Vec<String> = Vec::new();
    loop {
        let end = ((t.min(total)) * SR as f32) as usize;
        let last = end >= x.len();
        let buf = &x[start..end];
        let secs = buf.len() as f32 / SR as f32;
        let (mel, n_len) = log_mel_spectrogram(buf, &m.filters, n_mel);
        w.audio_ctx = ((secs * 50.0).ceil() as usize + 50).clamp(100, 1500);
        let t0 = std::time::Instant::now();
        let enc = w.encode(&mel, n_mel, n_len);
        let (text, _) = w.transcribe_greedy(&enc, (secs * 6.0) as usize + 8);
        let cost = t0.elapsed().as_secs_f32();
        busy += cost;
        let mut words: Vec<String> = text.split_whitespace().map(String::from).collect();
        if !last {
            rs_whisper::merge::drop_dangling(&mut words);
        }
        let agree = if last { words.len() } else { words.iter().zip(&prev).take_while(|(p, q)| norm(p) == norm(q)).count() };
        if agree > emitted {
            let new = words[emitted..agree].to_vec();
            println!("[audio {:5.1}s, buffer {:4.1}s, cost {:.1}s] final: {}", end as f32 / SR as f32, secs, cost, new.join(" "));
            all.extend(new);
            emitted = agree;
        } else {
            println!("[audio {:5.1}s, buffer {:4.1}s, cost {:.1}s] (nothing new agreed)", end as f32 / SR as f32, secs, cost);
        }
        prev = words;
        if last {
            break;
        }
        if secs > max_buf {
            if let Some((cut, est)) = cut_point(buf, prev.len(), emitted) {
                println!("    cut buffer at {:.1}s (dropping ~{} finalized words)", (start + cut) as f32 / SR as f32, est);
                start += cut;
                prev.drain(..est);
                emitted -= est;
            }
        }
        t += step;
    }
    println!("FINAL: {}", all.join(" "));
    eprintln!("compute {:.1}s for {:.1}s of audio (steps of {:.1}s, buffer limit {:.1}s)", busy, total, step, max_buf);
}
