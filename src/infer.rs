//! Stages 3 and 4: encoder, decoder and greedy decoding on real ggml weights (CPU, f32).
//! Written from the Whisper architecture; compared with whisper.cpp by transcript.
use crate::model::{f16_to_f32, Model};
use std::collections::HashMap;

pub const N_CTX: usize = 1500;
pub const N_FRAMES: usize = 3000;

struct W {
    t: HashMap<String, (Vec<i32>, Vec<f32>)>,
}
impl W {
    fn new(m: &Model, bytes: &[u8]) -> W {
        let mut t = HashMap::new();
        for ti in &m.tensors {
            let d = &bytes[ti.offset..ti.offset + ti.nbytes];
            let v: Vec<f32> = match ti.ttype {
                0 => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                1 => d.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
                _ => panic!("only f32/f16 weights are supported (ttype {})", ti.ttype),
            };
            t.insert(ti.name.clone(), (ti.dims.clone(), v));
        }
        W { t }
    }
    fn g(&self, n: &str) -> &[f32] {
        &self.t.get(n).unwrap_or_else(|| panic!("missing tensor {n}")).1
    }
}

fn par_rows<F: Fn(usize, &mut [f32]) + Sync>(out: &mut [f32], row: usize, f: F) {
    let n = out.len() / row;
    let th = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(16);
    let per = (n + th - 1) / th;
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per * row).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (ri, r) in chunk.chunks_mut(row).enumerate() {
                    f(ci * per + ri, r);
                }
            });
        }
    });
}

/// y[t][o] = sum_i w[o][i] * x[t][i] + b[o]
fn linear(x: &[f32], t: usize, i: usize, w: &[f32], o: usize, b: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0f32; t * o];
    par_rows(&mut y, o, |ti, row| {
        let xr = &x[ti * i..(ti + 1) * i];
        for oi in 0..o {
            let wr = &w[oi * i..(oi + 1) * i];
            let mut s = 0f32;
            for k in 0..i {
                s += wr[k] * xr[k];
            }
            row[oi] = s + b.map_or(0.0, |b| b[oi]);
        }
    });
    y
}

fn layer_norm(x: &mut [f32], d: usize, g: &[f32], b: &[f32]) {
    for r in x.chunks_mut(d) {
        let mean = r.iter().sum::<f32>() / d as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for k in 0..d {
            r[k] = (r[k] - mean) * inv * g[k] + b[k];
        }
    }
}

fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + (0.797_884_6 * v * (1.0 + 0.044715 * v * v)).tanh())
}

/// Multi-head attention. q: [tq][d], k,v: [tk][d]. causal: query i sees keys <= i + off.
fn attention(q: &[f32], tq: usize, k: &[f32], v: &[f32], tk: usize, d: usize, heads: usize, causal: bool) -> Vec<f32> {
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; tq * d];
    par_rows(&mut out, d, |qi, row| {
        let lim = if causal { (qi + 1).min(tk) } else { tk };
        let mut sc = vec![0f32; lim];
        for h in 0..heads {
            let qh = &q[qi * d + h * hd..qi * d + (h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..lim {
                let kh = &k[j * d + h * hd..j * d + (h + 1) * hd];
                let mut s = 0f32;
                for c in 0..hd {
                    s += qh[c] * kh[c];
                }
                sc[j] = s * scale;
                if sc[j] > mx {
                    mx = sc[j];
                }
            }
            let mut sum = 0f32;
            for j in 0..lim {
                sc[j] = (sc[j] - mx).exp();
                sum += sc[j];
            }
            for j in 0..lim {
                let p = sc[j] / sum;
                let vh = &v[j * d + h * hd..j * d + (h + 1) * hd];
                for c in 0..hd {
                    row[h * hd + c] += p * vh[c];
                }
            }
        }
    });
    out
}

pub struct Whisper {
    w: W,
    /// Encoder context length (1500 = full 30 s window; smaller = shorter window, like whisper.cpp -ac).
    pub audio_ctx: usize,
    pub n_state: usize,
    heads: usize,
    n_enc_layer: usize,
    n_dec_layer: usize,
    pub n_vocab: usize,
    pub vocab: Vec<Vec<u8>>,
}

impl Whisper {
    pub fn new(m: &Model, bytes: &[u8]) -> Whisper {
        let h = &m.hparams;
        Whisper {
            w: W::new(m, bytes),
            audio_ctx: N_CTX,
            n_state: h.n_audio_state as usize,
            heads: h.n_audio_head as usize,
            n_enc_layer: h.n_audio_layer as usize,
            n_dec_layer: h.n_text_layer as usize,
            n_vocab: h.n_vocab as usize,
            vocab: m.vocab.clone(),
        }
    }

    /// mel: [n_mel][n_len] (first N_FRAMES frames used) -> encoder output [N_CTX][n_state]
    pub fn encode(&self, mel: &[f32], n_mel: usize, n_len: usize) -> Vec<f32> {
        let d = self.n_state;
        let nctx = self.audio_ctx.clamp(1, N_CTX);
        let nframes = nctx * 2;
        // conv1: k=3,s=1,p=1 : [n_mel][3000] -> [3000][d]
        let w1 = self.w.g("encoder.conv1.weight");
        let b1 = self.w.g("encoder.conv1.bias");
        let mut x1 = vec![0f32; nframes * d];
        par_rows(&mut x1, d, |t, row| {
            for o in 0..d {
                let mut s = b1[o];
                for c in 0..n_mel {
                    for kk in 0..3 {
                        let ti = t as isize + kk as isize - 1;
                        if ti >= 0 && (ti as usize) < nframes {
                            s += w1[(o * n_mel + c) * 3 + kk] * mel[c * n_len + ti as usize];
                        }
                    }
                }
                row[o] = gelu(s);
            }
        });
        // conv2: k=3,s=2,p=1 : [3000][d] -> [1500][d]
        let w2 = self.w.g("encoder.conv2.weight");
        let b2 = self.w.g("encoder.conv2.bias");
        let mut x = vec![0f32; nctx * d];
        par_rows(&mut x, d, |t, row| {
            for o in 0..d {
                let mut s = b2[o];
                for kk in 0..3 {
                    let ti = (t * 2) as isize + kk as isize - 1;
                    if ti >= 0 && (ti as usize) < nframes {
                        let xr = &x1[ti as usize * d..(ti as usize + 1) * d];
                        let wr = &w2[(o * d) * 3..];
                        for c in 0..d {
                            s += wr[c * 3 + kk] * xr[c];
                        }
                    }
                }
                row[o] = gelu(s);
            }
        });
        let pe = self.w.g("encoder.positional_embedding");
        for i in 0..nctx * d {
            x[i] += pe[i];
        }
        for l in 0..self.n_enc_layer {
            let p = format!("encoder.blocks.{l}.");
            let g = |n: &str| self.w.g(&format!("{p}{n}"));
            let mut a = x.clone();
            layer_norm(&mut a, d, g("attn_ln.weight"), g("attn_ln.bias"));
            let q = linear(&a, nctx, d, g("attn.query.weight"), d, Some(g("attn.query.bias")));
            let k = linear(&a, nctx, d, g("attn.key.weight"), d, None);
            let v = linear(&a, nctx, d, g("attn.value.weight"), d, Some(g("attn.value.bias")));
            let at = attention(&q, nctx, &k, &v, nctx, d, self.heads, false);
            let o = linear(&at, nctx, d, g("attn.out.weight"), d, Some(g("attn.out.bias")));
            for i in 0..x.len() {
                x[i] += o[i];
            }
            let mut m = x.clone();
            layer_norm(&mut m, d, g("mlp_ln.weight"), g("mlp_ln.bias"));
            let mut h = linear(&m, nctx, d, g("mlp.0.weight"), d * 4, Some(g("mlp.0.bias")));
            for v in h.iter_mut() {
                *v = gelu(*v);
            }
            let o = linear(&h, nctx, d * 4, g("mlp.2.weight"), d, Some(g("mlp.2.bias")));
            for i in 0..x.len() {
                x[i] += o[i];
            }
        }
        layer_norm(&mut x, d, self.w.g("encoder.ln_post.weight"), self.w.g("encoder.ln_post.bias"));
        x
    }

    /// Logits of the last position for the token sequence `toks`.
    fn logits(&self, toks: &[u32], enc: &[f32], cross: &[(Vec<f32>, Vec<f32>)]) -> Vec<f32> {
        let d = self.n_state;
        let t = toks.len();
        let emb = self.w.g("decoder.token_embedding.weight");
        let pos = self.w.g("decoder.positional_embedding");
        let mut x = vec![0f32; t * d];
        for (i, &tk) in toks.iter().enumerate() {
            for c in 0..d {
                x[i * d + c] = emb[tk as usize * d + c] + pos[i * d + c];
            }
        }
        let _ = enc;
        for l in 0..self.n_dec_layer {
            let p = format!("decoder.blocks.{l}.");
            let g = |n: &str| self.w.g(&format!("{p}{n}"));
            let mut a = x.clone();
            layer_norm(&mut a, d, g("attn_ln.weight"), g("attn_ln.bias"));
            let q = linear(&a, t, d, g("attn.query.weight"), d, Some(g("attn.query.bias")));
            let k = linear(&a, t, d, g("attn.key.weight"), d, None);
            let v = linear(&a, t, d, g("attn.value.weight"), d, Some(g("attn.value.bias")));
            let at = attention(&q, t, &k, &v, t, d, self.heads, true);
            let o = linear(&at, t, d, g("attn.out.weight"), d, Some(g("attn.out.bias")));
            for i in 0..x.len() {
                x[i] += o[i];
            }
            let mut c = x.clone();
            layer_norm(&mut c, d, g("cross_attn_ln.weight"), g("cross_attn_ln.bias"));
            let q = linear(&c, t, d, g("cross_attn.query.weight"), d, Some(g("cross_attn.query.bias")));
            let at = attention(&q, t, &cross[l].0, &cross[l].1, enc.len() / d, d, self.heads, false);
            let o = linear(&at, t, d, g("cross_attn.out.weight"), d, Some(g("cross_attn.out.bias")));
            for i in 0..x.len() {
                x[i] += o[i];
            }
            let mut m = x.clone();
            layer_norm(&mut m, d, g("mlp_ln.weight"), g("mlp_ln.bias"));
            let mut h = linear(&m, t, d, g("mlp.0.weight"), d * 4, Some(g("mlp.0.bias")));
            for v in h.iter_mut() {
                *v = gelu(*v);
            }
            let o = linear(&h, t, d * 4, g("mlp.2.weight"), d, Some(g("mlp.2.bias")));
            for i in 0..x.len() {
                x[i] += o[i];
            }
        }
        let mut last = x[(t - 1) * d..].to_vec();
        layer_norm(&mut last, d, self.w.g("decoder.ln.weight"), self.w.g("decoder.ln.bias"));
        linear(&last, 1, d, emb, self.n_vocab, None)
    }

    /// Greedy decoding without timestamps, English, transcribe. Returns text and token ids.
    pub fn transcribe_greedy(&self, enc: &[f32], max_tokens: usize) -> (String, Vec<u32>) {
        let d = self.n_state;
        let cross: Vec<(Vec<f32>, Vec<f32>)> = (0..self.n_dec_layer)
            .map(|l| {
                let p = format!("decoder.blocks.{l}.cross_attn.");
                (
                    linear(enc, enc.len() / d, d, self.w.g(&format!("{p}key.weight")), d, None),
                    linear(enc, enc.len() / d, d, self.w.g(&format!("{p}value.weight")), d, Some(self.w.g(&format!("{p}value.bias")))),
                )
            })
            .collect();
        // multilingual vocabulary (51865): eot=50257, sot=50258, en=50259, transcribe=50359, notimestamps=50363
        let (eot, sot, en, transcribe, nots) = (50257u32, 50258u32, 50259u32, 50359u32, 50363u32);
        let space = self.vocab.iter().position(|v| v == b" ").map(|p| p as u32);
        let mut toks = vec![sot, en, transcribe, nots];
        let n_prompt = toks.len();
        for step in 0..max_tokens {
            let mut lg = self.logits(&toks, enc, &cross);
            for i in 50258..self.n_vocab {
                lg[i] = f32::NEG_INFINITY;
            }
            if step == 0 {
                lg[eot as usize] = f32::NEG_INFINITY;
                if let Some(s) = space {
                    lg[s as usize] = f32::NEG_INFINITY;
                }
            }
            let mut best = 0usize;
            for i in 1..lg.len() {
                if lg[i] > lg[best] {
                    best = i;
                }
            }
            if best as u32 == eot {
                break;
            }
            toks.push(best as u32);
        }
        let mut bytes = Vec::new();
        for &t in &toks[n_prompt..] {
            bytes.extend_from_slice(&self.vocab[t as usize]);
        }
        (String::from_utf8_lossy(&bytes).into_owned(), toks[n_prompt..].to_vec())
    }
}
