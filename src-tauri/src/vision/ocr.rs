// OCR inference kernel for scanned / image-only PDF pages.
//
// Pipeline (PP-OCRv4 mobile, run via `ort`, mirroring the TSR integration):
//   1. detect  — DBNet produces a per-pixel text probability map; we binarize
//                it, find connected components, take their axis-aligned boxes,
//                and "unclip" (expand) each box. → text-line boxes.
//   2. recognize — each box is cropped, resized to height 48, and fed to the
//                  CRNN/SVTR recognizer; the output sequence is CTC-greedy
//                  decoded against ppocr_keys_v1.txt. → text + confidence.
//
// The pure pieces (dict load, CTC decode, DB box extraction, unclip, coord
// mapping) are NOT feature-gated and are unit-tested with synthetic tensors —
// they carry the algorithmic risk and must be correct independent of the model.
// The ONNX session + image preprocessing + orchestration are gated behind
// `ocr-onnx`, exactly like `tsr-onnx`.

#![allow(dead_code)] // Orchestration is wired into the index pipeline in PR-3.

/// A detected text-line box in pixel coordinates of the detection input image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DetBox {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl DetBox {
    fn width(&self) -> f64 {
        (self.x1 - self.x0).max(0.0)
    }
    fn height(&self) -> f64 {
        (self.y1 - self.y0).max(0.0)
    }
}

/// One recognized text line: reading text, confidence, and a normalized
/// [x0,y0,x1,y1] bbox (0..1 in the original page's coordinate space).
#[derive(Debug, Clone)]
pub(crate) struct OcrLine {
    pub text: String,
    pub confidence: f64,
    pub bbox: [f64; 4],
}

// --- character dictionary ---------------------------------------------------

/// Build the recognizer's character set from a ppocr keys file.
///
/// PaddleOCR's `CTCLabelDecode` prepends a CTC blank at index 0, then the dict
/// characters, then appends a space. So `charset[0]` is blank (never emitted),
/// `charset[i]` for i in 1..=N is dict line i-1, and the final entry is " ".
pub(crate) fn build_charset(dict_text: &str) -> Vec<String> {
    let mut charset = Vec::with_capacity(8);
    charset.push(String::new()); // index 0 = CTC blank
    for line in dict_text.split('\n') {
        // Keep the line verbatim except for a trailing CR (Windows newlines);
        // dict entries can be a single space-less glyph, so do NOT trim spaces.
        let ch = line.strip_suffix('\r').unwrap_or(line);
        if ch.is_empty() {
            continue;
        }
        charset.push(ch.to_string());
    }
    charset.push(" ".to_string()); // trailing space char, per PaddleOCR
    charset
}

// --- CTC greedy decode ------------------------------------------------------

/// Greedy-decode a CRNN/SVTR output `[num_steps, num_classes]` (row-major) into
/// text. For each timestep take the argmax class; collapse runs of the same
/// class; drop the blank (index 0); map the rest through `charset`. Confidence
/// is the mean max-probability over the emitted (non-blank, non-repeat) steps.
pub(crate) fn ctc_greedy_decode(
    logits: &[f32],
    num_steps: usize,
    num_classes: usize,
    charset: &[String],
) -> (String, f64) {
    if num_classes == 0 || num_steps == 0 || logits.len() < num_steps * num_classes {
        return (String::new(), 0.0);
    }
    let mut text = String::new();
    let mut prob_sum = 0.0f64;
    let mut emitted = 0usize;
    let mut prev_class = usize::MAX;
    for step in 0..num_steps {
        let row = &logits[step * num_classes..(step + 1) * num_classes];
        let mut best_idx = 0usize;
        let mut best_val = row[0];
        for (idx, &val) in row.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        // Collapse repeats and skip blank (class 0).
        if best_idx != prev_class && best_idx != 0 {
            if let Some(ch) = charset.get(best_idx) {
                text.push_str(ch);
                prob_sum += f64::from(best_val);
                emitted += 1;
            }
        }
        prev_class = best_idx;
    }
    let confidence = if emitted > 0 {
        prob_sum / emitted as f64
    } else {
        0.0
    };
    (text, confidence)
}

// --- DBNet postprocessing: probability map -> text-line boxes ----------------

/// Extract axis-aligned text-line boxes from a DBNet probability map.
///
/// `prob` is row-major `[height * width]` in 0..1. We binarize at `bin_thresh`,
/// find 4-connectivity connected components, drop tiny ones, require the mean
/// in-box probability to exceed `box_thresh`, and return each component's
/// axis-aligned bounding box. Boxes are in the probability map's pixel space.
///
/// Note: PaddleOCR uses polygon contours + Vatti unclip; for v1 we use
/// axis-aligned component boxes, which is robust and sufficient for the
/// horizontal text lines found in scanned documents. `unclip_box` then expands
/// each box to recover the margin that DB shrinks during training.
pub(crate) fn extract_text_boxes(
    prob: &[f32],
    width: usize,
    height: usize,
    bin_thresh: f32,
    box_thresh: f32,
    min_box_side: usize,
) -> Vec<DetBox> {
    if width == 0 || height == 0 || prob.len() < width * height {
        return Vec::new();
    }
    let n = width * height;
    let mut visited = vec![false; n];
    let mut boxes = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..n {
        if visited[start] || prob[start] < bin_thresh {
            continue;
        }
        // Flood fill this component (4-connectivity).
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut prob_sum = 0.0f64;
        let mut count = 0usize;
        stack.clear();
        stack.push(start);
        visited[start] = true;
        while let Some(idx) = stack.pop() {
            let x = idx % width;
            let y = idx / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            prob_sum += f64::from(prob[idx]);
            count += 1;
            // Neighbors.
            if x > 0 {
                let nb = idx - 1;
                if !visited[nb] && prob[nb] >= bin_thresh {
                    visited[nb] = true;
                    stack.push(nb);
                }
            }
            if x + 1 < width {
                let nb = idx + 1;
                if !visited[nb] && prob[nb] >= bin_thresh {
                    visited[nb] = true;
                    stack.push(nb);
                }
            }
            if y > 0 {
                let nb = idx - width;
                if !visited[nb] && prob[nb] >= bin_thresh {
                    visited[nb] = true;
                    stack.push(nb);
                }
            }
            if y + 1 < height {
                let nb = idx + width;
                if !visited[nb] && prob[nb] >= bin_thresh {
                    visited[nb] = true;
                    stack.push(nb);
                }
            }
        }
        let box_w = max_x - min_x + 1;
        let box_h = max_y - min_y + 1;
        if box_w < min_box_side || box_h < min_box_side {
            continue;
        }
        let mean_prob = (prob_sum / count.max(1) as f64) as f32;
        if mean_prob < box_thresh {
            continue;
        }
        boxes.push(DetBox {
            x0: min_x as f64,
            y0: min_y as f64,
            // +1 so the box covers the inclusive max pixel.
            x1: (max_x + 1) as f64,
            y1: (max_y + 1) as f64,
        });
    }
    boxes
}

/// Expand an axis-aligned box outward to recover DB's training-time shrink.
/// PaddleOCR uses distance = area * ratio / perimeter (Vatti offset); for an
/// axis-aligned rectangle that distance is applied to every side. Result is
/// clamped to the image bounds.
pub(crate) fn unclip_box(b: &DetBox, ratio: f64, width: f64, height: f64) -> DetBox {
    let w = b.width();
    let h = b.height();
    if w <= 0.0 || h <= 0.0 {
        return *b;
    }
    let area = w * h;
    let perimeter = 2.0 * (w + h);
    let distance = area * ratio / perimeter.max(1e-6);
    DetBox {
        x0: (b.x0 - distance).max(0.0),
        y0: (b.y0 - distance).max(0.0),
        x1: (b.x1 + distance).min(width),
        y1: (b.y1 + distance).min(height),
    }
}

/// Map a box from detection-input pixel space to a normalized 0..1 bbox.
/// `scale_x`/`scale_y` convert det-input pixels to the original image's pixels;
/// `orig_w`/`orig_h` are the original image dimensions used for normalization.
pub(crate) fn det_box_to_normalized_bbox(
    b: &DetBox,
    scale_x: f64,
    scale_y: f64,
    orig_w: f64,
    orig_h: f64,
) -> [f64; 4] {
    let ow = orig_w.max(1.0);
    let oh = orig_h.max(1.0);
    [
        ((b.x0 * scale_x) / ow).clamp(0.0, 1.0),
        ((b.y0 * scale_y) / oh).clamp(0.0, 1.0),
        ((b.x1 * scale_x) / ow).clamp(0.0, 1.0),
        ((b.y1 * scale_y) / oh).clamp(0.0, 1.0),
    ]
}

// --- orchestration ----------------------------------------------------------

/// DBNet binarization threshold on the probability map.
const DET_BIN_THRESH: f32 = 0.3;
/// Minimum mean in-box probability for a detected region to count.
const DET_BOX_THRESH: f32 = 0.5;
/// Drop boxes whose shorter side (in det-input pixels) is below this.
const DET_MIN_BOX_SIDE: usize = 3;
/// DB unclip ratio (PaddleOCR default ~1.5..1.6).
const DET_UNCLIP_RATIO: f64 = 1.6;
/// Recognizer fixed input height for PP-OCRv4.
const REC_INPUT_HEIGHT: usize = 48;
/// Drop recognized lines below this confidence.
const REC_MIN_CONFIDENCE: f64 = 0.5;

/// Recognize all text lines on an already-rendered page image (RGBA bitmap as
/// produced by pdfium, given as raw rgba8 + dimensions).
///
/// IMPORTANT: this takes a pre-rendered bitmap, NOT a PDF path. The caller (the
/// index loop) renders the page using its already-open pdfium handle and passes
/// the pixels in. OCR must never bind pdfium itself — doing so while the index
/// loop holds a live pdfium document deadlocks under the `thread_safe` global
/// lock.
///
/// Returns reading lines with normalized (0..1) bboxes in the image's
/// coordinate space. Without the `ocr-onnx` feature this is a no-op.
#[cfg(not(feature = "ocr-onnx"))]
pub(crate) fn recognize_image_rgba(
    _rgba: &[u8],
    _width: u32,
    _height: u32,
) -> Result<Vec<OcrLine>, String> {
    Ok(Vec::new())
}

#[cfg(feature = "ocr-onnx")]
pub(crate) fn recognize_image_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<OcrLine>, String> {
    let dir = match super::ocr_model_dir_from_env_or_default() {
        Some(dir) if dir.is_dir() => dir,
        _ => return Ok(Vec::new()), // models absent -> behave as no-op
    };
    let det_path = dir.join("ch_PP-OCRv4_det_infer.onnx");
    let rec_path = dir.join("ch_PP-OCRv4_rec_infer.onnx");
    let dict_path = dir.join("ppocr_keys_v1.txt");
    for path in [&det_path, &rec_path, &dict_path] {
        if !path.is_file() {
            return Err(format!("OCR model file missing: {}", path.display()));
        }
    }
    let dict_text = std::fs::read_to_string(&dict_path)
        .map_err(|err| format!("Failed to read OCR dictionary: {err}"))?;
    let charset = build_charset(&dict_text);

    // Build an RGB image from the caller-provided RGBA bitmap (no pdfium here).
    let rgba_image = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or_else(|| "OCR page bitmap has wrong length for its dimensions".to_string())?;
    let page_image = image::DynamicImage::ImageRgba8(rgba_image).to_rgb8();

    let boxes = detect_boxes(&det_path, &page_image)?;
    let mut rec_session = onnx::build_session(&rec_path, "OCR rec")?;
    let orig_w = f64::from(page_image.width());
    let orig_h = f64::from(page_image.height());

    let mut lines = Vec::new();
    let mut source_order = 0u32;
    for b in boxes {
        let crop = onnx::crop_box(&page_image, &b);
        if crop.width() == 0 || crop.height() == 0 {
            continue;
        }
        let (text, confidence) = recognize_crop(&mut rec_session, &crop, &charset)?;
        let trimmed = normalize_ocr_text(&text);
        if trimmed.is_empty() || confidence < REC_MIN_CONFIDENCE {
            continue;
        }
        // Detection ran on the page image directly (scale 1.0), so the box is
        // already in page-image pixels; normalize against the page size.
        let bbox = det_box_to_normalized_bbox(&b, 1.0, 1.0, orig_w, orig_h);
        lines.push(OcrLine {
            text: trimmed,
            confidence,
            bbox,
        });
        source_order += 1;
        let _ = source_order;
    }
    Ok(lines)
}

/// Collapse internal whitespace runs and trim a recognized line.
fn normalize_ocr_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(feature = "ocr-onnx")]
fn detect_boxes(
    det_path: &std::path::Path,
    page_image: &image::RgbImage,
) -> Result<Vec<DetBox>, String> {
    let mut session = onnx::build_session(det_path, "OCR det")?;
    // DBNet needs H/W as multiples of 32. We feed the page image at its native
    // size (padded up), so detection coordinates map 1:1 back to page pixels.
    let (prob, det_w, det_h) = onnx::run_detection(&mut session, page_image)?;
    let mut boxes = extract_text_boxes(
        &prob,
        det_w,
        det_h,
        DET_BIN_THRESH,
        DET_BOX_THRESH,
        DET_MIN_BOX_SIDE,
    );
    let pw = f64::from(page_image.width());
    let ph = f64::from(page_image.height());
    for b in &mut boxes {
        *b = unclip_box(b, DET_UNCLIP_RATIO, pw, ph);
    }
    Ok(boxes)
}

#[cfg(feature = "ocr-onnx")]
fn recognize_crop(
    session: &mut ort::session::Session,
    crop: &image::RgbImage,
    charset: &[String],
) -> Result<(String, f64), String> {
    let (logits, steps, classes) = onnx::run_recognition(session, crop)?;
    Ok(ctc_greedy_decode(&logits, steps, classes, charset))
}

/// ONNX plumbing for OCR (session build, image→tensor, inference).
/// Mirrors the TSR helpers but with OCR-specific preprocessing:
/// DET uses ImageNet mean/std + multiple-of-32 padding; REC uses (x/255-0.5)/0.5
/// at fixed height 48.
#[cfg(feature = "ocr-onnx")]
mod onnx {
    use super::DetBox;
    use super::REC_INPUT_HEIGHT;
    use image::RgbImage;
    use ort::value::Tensor;

    pub(super) fn build_session(
        path: &std::path::Path,
        label: &str,
    ) -> Result<ort::session::Session, String> {
        ort::session::Session::builder()
            .map_err(|err| format!("Failed to create {label} ONNX session builder: {err}"))?
            .commit_from_file(path)
            .map_err(|err| {
                format!(
                    "Failed to load {label} ONNX model {}: {err}",
                    path.display()
                )
            })
    }

    pub(super) fn crop_box(image: &RgbImage, b: &DetBox) -> RgbImage {
        let x = (b.x0.floor().max(0.0)) as u32;
        let y = (b.y0.floor().max(0.0)) as u32;
        let x1 = (b.x1.ceil().max(0.0) as u32).min(image.width());
        let y1 = (b.y1.ceil().max(0.0) as u32).min(image.height());
        let w = x1.saturating_sub(x);
        let h = y1.saturating_sub(y);
        if w == 0 || h == 0 {
            return RgbImage::new(0, 0);
        }
        image::imageops::crop_imm(image, x, y, w, h).to_image()
    }

    /// Run DBNet. Pads the page image up to multiples of 32, normalizes with
    /// ImageNet stats, and returns (probability_map, width, height) where the
    /// dimensions equal the padded input (1:1 with page pixels in the valid
    /// region; the padded margin is background and yields no boxes).
    pub(super) fn run_detection(
        session: &mut ort::session::Session,
        page_image: &RgbImage,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        let src_w = page_image.width() as usize;
        let src_h = page_image.height() as usize;
        let pad_w = src_w.div_ceil(32) * 32;
        let pad_h = src_h.div_ceil(32) * 32;
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut data = vec![0.0f32; 3 * pad_h * pad_w];
        for y in 0..src_h {
            for x in 0..src_w {
                let pixel = page_image.get_pixel(x as u32, y as u32).0;
                for c in 0..3 {
                    let value = (f32::from(pixel[c]) / 255.0 - mean[c]) / std[c];
                    data[(c * pad_h + y) * pad_w + x] = value;
                }
            }
        }
        let outputs = session
            .run(ort::inputs![Tensor::<f32>::from_array((
                vec![1i64, 3, pad_h as i64, pad_w as i64],
                data
            ))
            .map_err(|err| format!(
                "Failed to build OCR det tensor: {err}"
            ))?])
            .map_err(|err| format!("OCR det inference failed: {err}"))?;
        let (_, value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "OCR det produced no output".to_string())?;
        let (shape, prob) = value
            .try_extract_tensor::<f32>()
            .map_err(|err| format!("Failed to extract OCR det output: {err}"))?;
        // shape = [1,1,H,W]
        let h = *shape.get(2).unwrap_or(&0) as usize;
        let w = *shape.get(3).unwrap_or(&0) as usize;
        Ok((prob.to_vec(), w, h))
    }

    /// Run the recognizer on a single line crop. Resizes to fixed height 48,
    /// width proportional (min 16), normalizes with (x/255-0.5)/0.5, and
    /// returns (logits, num_steps, num_classes).
    pub(super) fn run_recognition(
        session: &mut ort::session::Session,
        crop: &RgbImage,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        let target_h = REC_INPUT_HEIGHT;
        let src_w = crop.width().max(1) as usize;
        let src_h = crop.height().max(1) as usize;
        let ratio = target_h as f32 / src_h as f32;
        let target_w = ((src_w as f32 * ratio).round() as usize).clamp(16, 2048);
        let resized = image::imageops::resize(
            crop,
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::Triangle,
        );
        let mut data = vec![0.0f32; 3 * target_h * target_w];
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = resized.get_pixel(x as u32, y as u32).0;
                for c in 0..3 {
                    let value = (f32::from(pixel[c]) / 255.0 - 0.5) / 0.5;
                    data[(c * target_h + y) * target_w + x] = value;
                }
            }
        }
        let outputs = session
            .run(ort::inputs![Tensor::<f32>::from_array((
                vec![1i64, 3, target_h as i64, target_w as i64],
                data
            ))
            .map_err(|err| format!(
                "Failed to build OCR rec tensor: {err}"
            ))?])
            .map_err(|err| format!("OCR rec inference failed: {err}"))?;
        let (_, value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "OCR rec produced no output".to_string())?;
        let (shape, logits) = value
            .try_extract_tensor::<f32>()
            .map_err(|err| format!("Failed to extract OCR rec output: {err}"))?;
        // shape = [1, T, num_classes]
        let steps = *shape.get(1).unwrap_or(&0) as usize;
        let classes = *shape.get(2).unwrap_or(&0) as usize;
        Ok((logits.to_vec(), steps, classes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charset_has_blank_first_and_space_last() {
        let charset = build_charset("a\nb\n中\n");
        assert_eq!(charset.first().map(String::as_str), Some(""));
        assert_eq!(charset.last().map(String::as_str), Some(" "));
        // blank + 3 dict chars + space
        assert_eq!(charset.len(), 5);
        assert_eq!(charset[1], "a");
        assert_eq!(charset[3], "中");
    }

    #[test]
    fn ctc_collapses_repeats_and_drops_blank() {
        // charset: [blank, "h", "i"]  (num_classes = 3)
        let charset = vec!["".to_string(), "h".to_string(), "i".to_string()];
        // steps (argmax): h, h, blank, i  -> "hi"
        // logits row-major [step][class], make the intended class dominate.
        let rows: [[f32; 3]; 4] = [
            [0.1, 0.8, 0.1],   // h
            [0.1, 0.7, 0.2],   // h (repeat -> collapsed)
            [0.9, 0.05, 0.05], // blank (dropped)
            [0.1, 0.2, 0.7],   // i
        ];
        let logits: Vec<f32> = rows.iter().flatten().copied().collect();
        let (text, conf) = ctc_greedy_decode(&logits, 4, 3, &charset);
        assert_eq!(text, "hi");
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn ctc_handles_empty_and_all_blank() {
        let charset = vec!["".to_string(), "x".to_string()];
        let (t, c) = ctc_greedy_decode(&[], 0, 2, &charset);
        assert_eq!(t, "");
        assert_eq!(c, 0.0);
        // all-blank steps -> empty text
        let logits = vec![0.9, 0.1, 0.9, 0.1];
        let (t2, c2) = ctc_greedy_decode(&logits, 2, 2, &charset);
        assert_eq!(t2, "");
        assert_eq!(c2, 0.0);
    }

    #[test]
    fn extract_boxes_finds_two_separated_blobs() {
        // 10x5 prob map with two horizontal blobs separated by a cold column.
        let w = 10;
        let h = 5;
        let mut prob = vec![0.0f32; w * h];
        // blob A: x 0..3, y 1..3
        for y in 1..3 {
            for x in 0..3 {
                prob[y * w + x] = 0.9;
            }
        }
        // blob B: x 6..9, y 1..3
        for y in 1..3 {
            for x in 6..9 {
                prob[y * w + x] = 0.9;
            }
        }
        let mut boxes = extract_text_boxes(&prob, w, h, 0.3, 0.5, 1);
        boxes.sort_by(|a, b| a.x0.partial_cmp(&b.x0).unwrap());
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].x0, 0.0);
        assert_eq!(boxes[0].x1, 3.0);
        assert_eq!(boxes[1].x0, 6.0);
        assert_eq!(boxes[1].x1, 9.0);
    }

    #[test]
    fn extract_boxes_drops_tiny_and_low_prob() {
        let w = 6;
        let h = 6;
        let mut prob = vec![0.0f32; w * h];
        prob[0] = 0.9; // single pixel -> too small with min_box_side=2
                       // a low-prob 3x3 region (mean below box_thresh)
        for y in 2..5 {
            for x in 2..5 {
                prob[y * w + x] = 0.35; // above bin 0.3 but mean < box_thresh 0.5
            }
        }
        let boxes = extract_text_boxes(&prob, w, h, 0.3, 0.5, 2);
        assert!(boxes.is_empty(), "got {boxes:?}");
    }

    #[test]
    fn unclip_expands_and_clamps() {
        let b = DetBox {
            x0: 10.0,
            y0: 10.0,
            x1: 30.0,
            y1: 14.0,
        };
        let u = unclip_box(&b, 1.5, 100.0, 100.0);
        assert!(u.x0 < b.x0 && u.y0 < b.y0);
        assert!(u.x1 > b.x1 && u.y1 > b.y1);
        // clamp: a box at the edge cannot go negative
        let edge = DetBox {
            x0: 0.0,
            y0: 0.0,
            x1: 5.0,
            y1: 5.0,
        };
        let ue = unclip_box(&edge, 5.0, 100.0, 100.0);
        assert_eq!(ue.x0, 0.0);
        assert_eq!(ue.y0, 0.0);
    }

    // End-to-end smoke test against a real scanned PDF + bundled models.
    // Ignored by default (needs the `ocr-onnx` feature, the models, and a PDF).
    // Run with:
    //   LUMENFOLIO_OCR_SMOKE_PDF=/tmp/ocr-test/scanned_test.pdf \
    //   cargo test --features ocr-onnx --lib vision::ocr::tests::e2e_recognize_scanned_pdf -- --ignored --nocapture
    #[cfg(feature = "ocr-onnx")]
    #[test]
    #[ignore]
    fn e2e_recognize_scanned_pdf() {
        let pdf = std::env::var("LUMENFOLIO_OCR_SMOKE_PDF")
            .expect("set LUMENFOLIO_OCR_SMOKE_PDF to a scanned PDF path");
        // Standalone test: no live pdfium document is held here, so binding a
        // fresh pdfium just to render the page bitmap is safe.
        let image_path = super::super::render_pdf_page_image(std::path::Path::new(&pdf), 1)
            .expect("render page image");
        let rgba = image::ImageReader::open(&image_path)
            .expect("open page image")
            .decode()
            .expect("decode page image")
            .to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());
        let lines = super::recognize_image_rgba(rgba.as_raw(), w, h)
            .expect("OCR recognize_image_rgba failed");
        println!("OCR recognized {} lines:", lines.len());
        for line in &lines {
            println!(
                "  conf={:.3} bbox=[{:.3},{:.3},{:.3},{:.3}]  {:?}",
                line.confidence, line.bbox[0], line.bbox[1], line.bbox[2], line.bbox[3], line.text
            );
        }
        assert!(
            !lines.is_empty(),
            "expected OCR to recover at least one line"
        );
    }

    #[test]
    fn normalized_bbox_scales_and_clamps() {
        // det-input box at 2x downscale of a 200x100 original.
        let b = DetBox {
            x0: 0.0,
            y0: 0.0,
            x1: 50.0,
            y1: 25.0,
        };
        let bbox = det_box_to_normalized_bbox(&b, 2.0, 2.0, 200.0, 100.0);
        assert_eq!(bbox, [0.0, 0.0, 0.5, 0.5]);
        // out-of-range stays clamped to 1.0
        let big = DetBox {
            x0: 0.0,
            y0: 0.0,
            x1: 1000.0,
            y1: 1000.0,
        };
        let bbox2 = det_box_to_normalized_bbox(&big, 2.0, 2.0, 200.0, 100.0);
        assert_eq!(bbox2, [0.0, 0.0, 1.0, 1.0]);
    }
}
