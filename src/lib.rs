//! rs-whisper: whisper.cpp (MIT, (c) The ggml authors) の Rust 移植。段階1 = log-mel スペクトログラム。
//! 元: whisper.cpp `src/whisper.cpp` の fft / dft / log_mel_spectrogram(単一スレッド相当、結果は同一)。
//! 演算の順序・型(f32 の表、f64 の積算)を元と揃え、同じ入力で同じ出力になるようにしてある。

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
const SIN_COS_N: usize = N_FFT;

pub struct Cache {
    sin: [f32; SIN_COS_N],
    cos: [f32; SIN_COS_N],
    pub hann: [f32; N_FFT],
}

impl Cache {
    pub fn new() -> Self {
        let mut sin = [0f32; SIN_COS_N];
        let mut cos = [0f32; SIN_COS_N];
        for i in 0..SIN_COS_N {
            let theta = (2.0 * std::f64::consts::PI * i as f64) / SIN_COS_N as f64;
            sin[i] = (theta as f32).sin();
            cos[i] = (theta as f32).cos();
        }
        let mut hann = [0f32; N_FFT];
        for (i, h) in hann.iter_mut().enumerate() {
            let a = ((2.0 * std::f64::consts::PI * i as f64) / N_FFT as f64) as f32;
            *h = (0.5 * (1.0 - a.cos() as f64)) as f32;
        }
        Cache { sin, cos, hann }
    }

    fn dft(&self, inp: &[f32], n: usize, out: &mut [f32]) {
        let step = SIN_COS_N / n;
        for k in 0..n {
            let (mut re, mut im) = (0f32, 0f32);
            for j in 0..n {
                let idx = (k * j * step) % SIN_COS_N;
                re += inp[j] * self.cos[idx];
                im -= inp[j] * self.sin[idx];
            }
            out[k * 2] = re;
            out[k * 2 + 1] = im;
        }
    }

    /// Cooley-Tukey。`inp` は入力+作業領域(長さ 2n 以上)、`out` は出力+作業領域(長さ 4n 以上)。
    pub fn fft(&self, inp: &mut [f32], n: usize, out: &mut [f32]) {
        if n == 1 {
            out[0] = inp[0];
            out[1] = 0.0;
            return;
        }
        let half = n / 2;
        if n - half * 2 == 1 {
            self.dft(inp, n, out);
            return;
        }
        let (a, even) = inp.split_at_mut(n);
        for i in 0..half {
            even[i] = a[2 * i];
        }
        let (o1, rest) = out.split_at_mut(2 * n);
        self.fft(even, half, rest);
        for i in 0..half {
            even[i] = a[2 * i + 1];
        }
        self.fft(even, half, &mut rest[n..]);
        let step = SIN_COS_N / n;
        for k in 0..half {
            let idx = k * step;
            let re = self.cos[idx];
            let im = -self.sin[idx];
            let (re_o, im_o) = (rest[n + 2 * k], rest[n + 2 * k + 1]);
            let (re_e, im_e) = (rest[2 * k], rest[2 * k + 1]);
            o1[2 * k] = re_e + re * re_o - im * im_o;
            o1[2 * k + 1] = im_e + re * im_o + im * re_o;
            o1[2 * (k + half)] = re_e - re * re_o + im * im_o;
            o1[2 * (k + half) + 1] = im_e - re * im_o - im * re_o;
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

/// 戻り値: (mel[n_mel * n_len], n_len)。`filters` は n_mel * (N_FFT/2+1)。
pub fn log_mel_spectrogram(samples: &[f32], filters: &[f32], n_mel: usize) -> (Vec<f32>, usize) {
    let c = Cache::new();
    let n_fft = 1 + N_FFT / 2;
    assert_eq!(filters.len(), n_mel * n_fft);
    let n = samples.len();
    let stage1 = SAMPLE_RATE * 30;
    let stage2 = N_FFT / 2;
    let mut p = vec![0f32; n + stage1 + stage2 * 2];
    p[stage2..stage2 + n].copy_from_slice(samples);
    let n_reflect = stage2.min(n.saturating_sub(1));
    for i in 0..n_reflect {
        p[stage2 - n_reflect + i] = samples[n_reflect - i];
    }
    let n_len = (p.len() - N_FFT) / HOP_LENGTH;
    let n_samples = n + stage2;
    let mut mel = vec![0f32; n_mel * n_len];
    let mut fft_in = vec![0f32; N_FFT * 2];
    let mut fft_out = vec![0f32; N_FFT * 8];
    let lim = (n_samples / HOP_LENGTH + 1).min(n_len);
    for i in 0..lim {
        let off = i * HOP_LENGTH;
        let m = N_FFT.min(n_samples - off);
        for j in 0..m {
            fft_in[j] = c.hann[j] * p[off + j];
        }
        if n_samples - off < N_FFT {
            for v in fft_in[n_samples - off..].iter_mut() {
                *v = 0.0;
            }
        }
        c.fft(&mut fft_in, N_FFT, &mut fft_out);
        for j in 0..n_fft {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
        for j in 0..n_mel {
            let mut sum = 0f64;
            let mut k = 0;
            while k + 3 < n_fft {
                let f = &filters[j * n_fft + k..];
                // 元は float 同士の積と和を f32 で計算してから double へ足す
                let t = fft_out[k] * f[0] + fft_out[k + 1] * f[1] + fft_out[k + 2] * f[2] + fft_out[k + 3] * f[3];
                sum += t as f64;
                k += 4;
            }
            while k < n_fft {
                sum += (fft_out[k] * filters[j * n_fft + k]) as f64;
                k += 1;
            }
            mel[j * n_len + i] = sum.max(1e-10).log10() as f32;
        }
    }
    let fill = (1e-10f64).log10() as f32;
    for i in lim..n_len {
        for j in 0..n_mel {
            mel[j * n_len + i] = fill;
        }
    }
    let mut mmax = -1e20f64;
    for &v in &mel {
        if (v as f64) > mmax {
            mmax = v as f64;
        }
    }
    mmax -= 8.0;
    for v in mel.iter_mut() {
        if (*v as f64) < mmax {
            *v = mmax as f32;
        }
        *v = ((*v as f64 + 4.0) / 4.0) as f32;
    }
    (mel, n_len)
}
