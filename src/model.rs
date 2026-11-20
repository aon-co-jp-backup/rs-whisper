//! Stage 2: loader for the ggml model file format used by whisper.cpp.
//! Reads the header, mel filters, vocabulary and a directory of tensors
//! (name, shape, type, offset, size). Weights are not copied: only their
//! position and size are kept so they can be read on demand.
use std::fmt;

pub const GGML_MAGIC: u32 = 0x6767_6d6c;

#[derive(Debug)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for Error {}
type R<T> = Result<T, Error>;
fn err<T>(s: impl Into<String>) -> R<T> {
    Err(Error(s.into()))
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Hparams {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    pub ftype: i32,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<i32>,
    pub ttype: i32,
    pub offset: usize,
    pub nbytes: usize,
}

#[derive(Debug)]
pub struct Model {
    pub hparams: Hparams,
    pub n_mel: i32,
    pub n_fft: i32,
    pub filters: Vec<f32>,
    pub vocab: Vec<Vec<u8>>,
    pub tensors: Vec<TensorInfo>,
}

/// (elements per block, bytes per block) for each ggml type id.
pub fn type_layout(t: i32) -> Option<(usize, usize)> {
    Some(match t {
        0 => (1, 4),   // f32
        1 => (1, 2),   // f16
        2 => (32, 18), // q4_0
        3 => (32, 20), // q4_1
        6 => (32, 22), // q5_0
        7 => (32, 24), // q5_1
        8 => (32, 34), // q8_0
        _ => return None,
    })
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        if self.p + n > self.b.len() {
            return err("unexpected end of file");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn i32(&mut self) -> R<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn eof(&self) -> bool {
        self.p >= self.b.len()
    }
}

pub fn load(bytes: &[u8]) -> R<Model> {
    let mut r = Rd { b: bytes, p: 0 };
    if r.u32()? != GGML_MAGIC {
        return err("bad magic");
    }
    let h = Hparams {
        n_vocab: r.i32()?,
        n_audio_ctx: r.i32()?,
        n_audio_state: r.i32()?,
        n_audio_head: r.i32()?,
        n_audio_layer: r.i32()?,
        n_text_ctx: r.i32()?,
        n_text_state: r.i32()?,
        n_text_head: r.i32()?,
        n_text_layer: r.i32()?,
        n_mels: r.i32()?,
        ftype: r.i32()?,
    };
    let n_mel = r.i32()?;
    let n_fft = r.i32()?;
    if n_mel <= 0 || n_fft <= 0 || (n_mel as i64) * (n_fft as i64) > 1 << 24 {
        return err("bad filter size");
    }
    let raw = r.take((n_mel * n_fft) as usize * 4)?;
    let filters: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let nv = r.i32()?;
    if nv < 0 || nv > 1 << 20 {
        return err("bad vocab size");
    }
    let mut vocab = Vec::with_capacity(nv as usize);
    for _ in 0..nv {
        let len = r.u32()? as usize;
        vocab.push(r.take(len)?.to_vec());
    }
    let mut tensors = Vec::new();
    while !r.eof() {
        let n_dims = r.i32()?;
        let length = r.i32()?;
        let ttype = r.i32()?;
        if !(0..=4).contains(&n_dims) {
            return err(format!("invalid n_dims {n_dims}"));
        }
        let (blk, bsz) = match type_layout(ttype) {
            Some(x) => x,
            None => return err(format!("unsupported ttype {ttype}")),
        };
        let mut dims = Vec::new();
        let mut ne: usize = 1;
        for _ in 0..n_dims {
            let d = r.i32()?;
            if d < 0 {
                return err("negative dim");
            }
            ne = ne.checked_mul(d as usize).ok_or(Error("overflow".into()))?;
            dims.push(d);
        }
        if length < 0 {
            return err("bad name length");
        }
        let name = String::from_utf8_lossy(r.take(length as usize)?).into_owned();
        if ne % blk != 0 {
            return err(format!("tensor {name}: size not a multiple of block"));
        }
        let nbytes = ne / blk * bsz;
        let offset = r.p;
        r.take(nbytes)?;
        tensors.push(TensorInfo { name, dims, ttype, offset, nbytes });
    }
    Ok(Model { hparams: h, n_mel, n_fft, filters, vocab, tensors })
}

impl Model {
    /// Returns an f32 or f16 tensor as f32 values.
    pub fn tensor_f32(&self, bytes: &[u8], name: &str) -> Option<Vec<f32>> {
        let t = self.tensors.iter().find(|t| t.name == name)?;
        let d = &bytes[t.offset..t.offset + t.nbytes];
        match t.ttype {
            0 => Some(d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
            1 => Some(
                d.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
            ),
            _ => None,
        }
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 113u32;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
