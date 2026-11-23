//! rs-whisper-la <ggml model.bin> <samples.f32> [step_seconds] [max_buffer_seconds]: LocalAgreement streaming
//! simulation (as in ufal/whisper_streaming): every step the buffer is transcribed again and only the prefix on
//! which two consecutive hypotheses agree is emitted. When the buffer grows beyond max_buffer_seconds, the finalized
//! part is cut off at the real start time of the first non-final word (cross-attention DTW word times).
use rs_whisper::{infer::Whisper, log_mel_spectrogram, model};

const SR: usize = 16000;

fn norm(w: &str) -> String {
    w.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
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
        let (text, ids) = w.transcribe_greedy(&enc, (secs * 6.0) as usize + 8);
        let cost = t0.elapsed().as_secs_f32();
        busy += cost;
        let mut words: Vec<String> = text.split_whitespace().map(String::from).collect();
        {
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
            // real word times from cross-attention DTW: cut just before the first not-yet-final word
            let k = emitted.min(prev.len());
            if k > 0 {
                let tt = w.token_times(&enc, &ids);
                let mut starts: Vec<f32> = Vec::new();
                for (i, &id) in ids.iter().enumerate() {
                    let b = &w.vocab[id as usize];
                    if starts.is_empty() || b.first() == Some(&b' ') {
                        starts.push(tt[i]);
                    }
                }
                if k < starts.len() {
                    let cut = (((starts[k] - 0.15).max(0.0)) * SR as f32) as usize;
                    if cut > 0 && cut < buf.len() {
                        println!("    cut buffer at {:.2}s (dropping {} finalized words, real word times)", (start + cut) as f32 / SR as f32, k);
                        start += cut;
                        prev.drain(..k);
                        emitted -= k;
                    }
                }
            }
        }
        t += step;
    }
    println!("FINAL: {}", all.join(" "));
    eprintln!("compute {:.1}s for {:.1}s of audio (steps of {:.1}s, buffer limit {:.1}s)", busy, total, step, max_buf);
}
