//! Helpers for the `run` CLI command: input loading, fp16 conversion, top-k.

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::v1::f32_to_f16;

/// How to turn an input file into the fp16 tensor the stick expects.
pub struct Preprocess {
    pub width: u32,
    pub height: u32,
    /// Per-channel mean subtracted after scaling (in the channel order given by `bgr`).
    pub mean: [f32; 3],
    /// Multiplier applied to 0..255 pixel values before the mean is subtracted.
    pub scale: f32,
    /// true = B,G,R channel order (Caffe convention), false = R,G,B.
    pub bgr: bool,
}

/// Input encodings accepted by `load_input` / `decode_input`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// Little-endian f32 values, already preprocessed, HWC order.
    F32,
    /// Little-endian fp16 values, already preprocessed, HWC order.
    F16,
    /// png/jpeg bytes: decoded, resized, preprocessed.
    Image,
}

impl InputKind {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "raw" | "f32" => Self::F32,
            "f16" => Self::F16,
            _ => Self::Image,
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "f32" | "raw" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "image" | "img" | "jpeg" | "jpg" | "png" => Some(Self::Image),
            _ => None,
        }
    }
}

/// Load an input file.
/// * `.raw` / `.f32`: little-endian f32 values, already preprocessed, HWC order.
/// * `.f16`: little-endian fp16 values, already preprocessed, HWC order.
/// * anything else: decoded as an image (png/jpeg), resized to width x height, preprocessed.
pub fn load_input(path: &Path, pre: &Preprocess) -> Result<Vec<u16>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    decode_input(&b, InputKind::from_extension(ext), pre).with_context(|| path.display().to_string())
}

/// Turn raw bytes into the fp16 tensor the stick expects.
pub fn decode_input(bytes: &[u8], kind: InputKind, pre: &Preprocess) -> Result<Vec<u16>> {
    match kind {
        InputKind::F32 => {
            if bytes.len() % 4 != 0 {
                bail!("f32 input: size is not a multiple of 4");
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32_to_f16(f32::from_le_bytes(c.try_into().unwrap())))
                .collect())
        }
        InputKind::F16 => {
            if bytes.len() % 2 != 0 {
                bail!("f16 input: odd byte count");
            }
            Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        }
        InputKind::Image => {
            let img = image::load_from_memory(bytes).context("decode image")?;
            Ok(preprocess_dynamic(img, pre).iter().map(|&v| f32_to_f16(v)).collect())
        }
    }
}

/// Resize (triangle filter) and normalise an image into f32 HWC.
fn preprocess_dynamic(img: image::DynamicImage, pre: &Preprocess) -> Vec<f32> {
    let img = img
        .resize_exact(pre.width, pre.height, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let mut out = Vec::with_capacity((pre.width * pre.height * 3) as usize);
    for px in img.pixels() {
        let [r, g, b] = px.0;
        let ch = if pre.bgr { [b, g, r] } else { [r, g, b] };
        for (i, v) in ch.iter().enumerate() {
            out.push(*v as f32 * pre.scale - pre.mean[i]);
        }
    }
    out
}

pub fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(k).map(|i| (i, values[i])).collect()
}

pub fn load_labels(path: &Path) -> Result<Vec<String>> {
    Ok(std::fs::read_to_string(path)
        .with_context(|| format!("read labels {}", path.display()))?
        .lines()
        .map(|s| s.to_string())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_f32_and_f16() {
        let pre = Preprocess { width: 1, height: 1, mean: [0.0; 3], scale: 1.0, bgr: false };
        let b: Vec<u8> = [1.0f32, -2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(decode_input(&b, InputKind::F32, &pre).unwrap(), vec![0x3c00, 0xc000]);
        assert!(decode_input(&b[..7], InputKind::F32, &pre).is_err());
        assert_eq!(decode_input(&[0x00, 0x3c], InputKind::F16, &pre).unwrap(), vec![0x3c00]);
        assert!(decode_input(&[0x00], InputKind::F16, &pre).is_err());
    }

    #[test]
    fn decode_image_rejects_garbage_and_applies_mean() {
        let pre = Preprocess { width: 2, height: 2, mean: [10.0, 20.0, 30.0], scale: 1.0, bgr: false };
        assert!(decode_input(b"not an image", InputKind::Image, &pre).is_err());
        // 1x1 red PNG built in memory
        let mut png = Vec::new();
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([100, 50, 0]));
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let t = decode_input(&png, InputKind::Image, &pre).unwrap();
        assert_eq!(t.len(), 2 * 2 * 3);
        assert_eq!(t[0], f32_to_f16(90.0));
        assert_eq!(t[1], f32_to_f16(30.0));
        assert_eq!(t[2], f32_to_f16(-30.0));
    }

    #[test]
    fn input_kind_parsing() {
        assert_eq!(InputKind::parse("F32"), Some(InputKind::F32));
        assert_eq!(InputKind::parse("jpeg"), Some(InputKind::Image));
        assert_eq!(InputKind::parse("xml"), None);
        assert_eq!(InputKind::from_extension("RAW"), InputKind::F32);
    }

    #[test]
    fn topk_orders() {
        let t = top_k(&[0.1, 0.9, 0.5, 0.7], 3);
        assert_eq!(t.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 3, 2]);
    }
}
