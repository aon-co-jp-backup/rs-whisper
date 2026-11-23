//! rs-whisper-la <ggml model.bin> <samples.f32> [step_seconds]: LocalAgreement streaming simulation
//! (as in ufal/whisper_streaming): every step the whole buffer (up to 15 s) is transcribed again and only the
//! prefix on which two consecutive hypotheses agree is emitted. Prints when each word became final.
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
    let n_mel = m.n_mel as usize;
    let mut w = Whisper::new(&m, &bytes);
    let total = x.len() as f32 / SR as f32;
    assert!(total <= 15.0, "this prototype keeps the whole buffer: audio must be at most 15 s");
    let mut prev: Vec<String> = Vec::new();
    let mut emitted = 0usize;
    let mut t = step;
    let mut busy = 0f32;
    loop {
        let end_s = t.min(total);
        let last = end_s >= total;
        let buf = &x[..(end_s * SR as f32) as usize];
        let (mel, n_len) = log_mel_spectrogram(buf, &m.filters, n_mel);
        w.audio_ctx = ((end_s * 50.0).ceil() as usize + 50).clamp(100, 1500);
        let t0 = std::time::Instant::now();
        let enc = w.encode(&mel, n_mel, n_len);
        let (text, _) = w.transcribe_greedy(&enc, (end_s * 6.0) as usize + 8);
        let cost = t0.elapsed().as_secs_f32();
        busy += cost;
        let mut words: Vec<String> = text.split_whitespace().map(String::from).collect();
        if !last {
            rs_whisper::merge::drop_dangling(&mut words);
        }
        let agree = if last {
            words.len()
        } else {
            words.iter().zip(&prev).take_while(|(p, q)| norm(p) == norm(q)).count()
        };
        if agree > emitted {
            println!("[audio {:5.1}s, cost {:.1}s] final: {}", end_s, cost, words[emitted..agree].join(" "));
            emitted = agree;
        } else {
            println!("[audio {:5.1}s, cost {:.1}s] (nothing new agreed; hypothesis: {})", end_s, cost, words.join(" "));
        }
        prev = words;
        if last {
            break;
        }
        t += step;
    }
    eprintln!("compute {:.1}s for {:.1}s of audio (steps of {:.1}s)", busy, total, step);
}
