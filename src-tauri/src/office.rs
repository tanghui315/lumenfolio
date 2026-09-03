//! Knowledge-base pivot (P3): text extraction from Office formats.
//!
//! Office files are ZIP archives of XML. We pull readable text only (for the
//! chunk → graph → claims pipeline); fidelity rendering is the frontend
//! preview's job (docx/xlsx), not this extractor. Returns `(text, role)` blocks
//! in document order, matching the shape the text index path expects.

use std::io::Read;
use std::path::Path;

use calamine::{Data, DataType as _, Reader as _};
use quick_xml::events::Event;
use quick_xml::reader::Reader;

/// Cap to keep a runaway spreadsheet/deck from flooding the index.
const MAX_BLOCKS: usize = 5000;

/// One indexable unit of an Office file.
///
/// `page` is the 1-based slide number for pptx, and 0 for formats with no
/// pagination (docx flows, xlsx grids). A real slide number is what lets a deck
/// reuse the PDF-side page tools — `open_pages` and page-anchored citations —
/// instead of needing a pptx-specific reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExtractedBlock {
    pub text: String,
    pub role: String,
    pub page: u32,
}

impl ExtractedBlock {
    fn new(text: impl Into<String>, role: &str, page: u32) -> Self {
        Self {
            text: text.into(),
            role: role.to_string(),
            page,
        }
    }
}

pub(crate) fn extract_office_blocks(
    path: &Path,
    content_type: &str,
) -> Result<Vec<ExtractedBlock>, String> {
    let blocks = match content_type {
        "docx" => extract_docx(path)?,
        "pptx" => extract_pptx(path)?,
        "xlsx" => extract_xlsx(path)?,
        other => return Err(format!("Unsupported office content type '{other}'")),
    };
    Ok(blocks.into_iter().take(MAX_BLOCKS).collect())
}

/// Whether a content type is an Office source indexed via this extractor.
pub(crate) fn is_office_source(content_type: &str) -> bool {
    matches!(content_type, "docx" | "xlsx" | "pptx")
}

/// Map a lowercase file extension to an Office content type, if supported.
pub(crate) fn office_content_type_for_ext(ext: &str) -> Option<&'static str> {
    match ext {
        "docx" => Some("docx"),
        "xlsx" => Some("xlsx"),
        "pptx" => Some("pptx"),
        _ => None,
    }
}

fn read_zip_entry(path: &Path, entry: &str) -> Result<Option<String>, String> {
    let file = std::fs::File::open(path).map_err(|err| format!("Failed to open file: {err}"))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|err| format!("Not a valid Office (ZIP) file: {err}"))?;
    let mut zf = match archive.by_name(entry) {
        Ok(zf) => zf,
        Err(_) => return Ok(None),
    };
    let mut contents = String::new();
    zf.read_to_string(&mut contents)
        .map_err(|err| format!("Failed to read {entry}: {err}"))?;
    Ok(Some(contents))
}

fn zip_entry_names(path: &Path) -> Result<Vec<String>, String> {
    let file = std::fs::File::open(path).map_err(|err| format!("Failed to open file: {err}"))?;
    let archive = zip::ZipArchive::new(file)
        .map_err(|err| format!("Not a valid Office (ZIP) file: {err}"))?;
    Ok(archive.file_names().map(|name| name.to_string()).collect())
}

// ---------------------------------------------------------------------------
// DOCX — word/document.xml: <w:p> paragraphs, <w:t> text runs, heading via
// <w:pStyle w:val="Heading…">.
// ---------------------------------------------------------------------------

fn extract_docx(path: &Path) -> Result<Vec<ExtractedBlock>, String> {
    let Some(xml) = read_zip_entry(path, "word/document.xml")? else {
        return Err("docx is missing word/document.xml".to_string());
    };
    let mut reader = Reader::from_str(&xml);
    reader.config_mut().trim_text(false);
    let mut blocks = Vec::new();
    let mut paragraph = String::new();
    let mut role = "body".to_string();
    let mut in_text = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"w:p" => {
                    paragraph.clear();
                    role = "body".to_string();
                }
                b"w:t" => in_text = true,
                _ => {}
            },
            Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"w:pStyle" {
                    if let Some(style) = attr_value(&e, b"w:val") {
                        if style.starts_with("Heading") || style.starts_with("Title") {
                            role = "heading".to_string();
                        }
                    }
                }
            }
            Ok(Event::Text(e)) if in_text => {
                if let Ok(text) = e.xml_content(quick_xml::XmlVersion::Implicit1_0) {
                    paragraph.push_str(&text);
                }
            }
            // Entities (&amp; &lt; …) and char refs arrive as separate events.
            Ok(Event::GeneralRef(e)) if in_text => {
                if let Some(text) = resolve_entity(&e) {
                    paragraph.push_str(&text);
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"w:t" => in_text = false,
                b"w:p" => {
                    let trimmed = paragraph.trim();
                    if !trimmed.is_empty() {
                        blocks.push(ExtractedBlock::new(trimmed, &role, 0));
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(err) => return Err(format!("Failed to parse docx XML: {err}")),
            _ => {}
        }
    }
    Ok(blocks)
}

// ---------------------------------------------------------------------------
// PPTX — one block per slide, carrying the real slide number.
//
// Previously every <a:p> paragraph became its own block, which for a real deck
// meant ~380 blocks averaging 7 characters ("01", "展示风格"): useless as
// retrieval units, and all pinned to page 0 so nothing could say which slide an
// answer came from. Now a slide is one block on `page = N`, which also lets a
// deck reuse the PDF-side page tooling (open_pages, page-anchored citations)
// rather than needing a pptx-specific reader.
//
// Text is gathered from everything the deck actually carries, not just the
// shapes: speaker notes (often the fullest narration — this deck averages ~100
// characters of shape text per slide but has 23 notes pages), SmartArt diagrams
// and chart labels. All three hang off the slide's .rels, which is the only
// reliable link — notesSlideN does NOT always belong to slideN.
// ---------------------------------------------------------------------------

/// Speaker notes repeat the slide number as a standalone run; drop that noise.
fn is_slide_number_noise(text: &str, page: u32) -> bool {
    text.trim().parse::<u32>() == Ok(page)
}

fn extract_pptx(path: &Path) -> Result<Vec<ExtractedBlock>, String> {
    let mut slide_names: Vec<String> = zip_entry_names(path)?
        .into_iter()
        .filter(|name| name.starts_with("ppt/slides/slide") && name.ends_with(".xml"))
        .collect();
    // Sort by the numeric suffix so slides come out in deck order (slide2 < slide10).
    slide_names.sort_by_key(|name| slide_number(name));

    let mut blocks = Vec::new();
    for (index, name) in slide_names.iter().enumerate() {
        let page = (index + 1) as u32;
        let Some(xml) = read_zip_entry(path, name)? else {
            continue;
        };

        let mut parts: Vec<String> = parse_pptx_slide(&xml)?
            .into_iter()
            .map(|paragraph| paragraph.trim().to_string())
            .filter(|paragraph| !paragraph.is_empty())
            .collect();

        // Follow this slide's relationships to its notes, diagrams and charts.
        let rels_name = name.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels";
        let rel_targets = match read_zip_entry(path, &rels_name)? {
            Some(rels) => parse_rel_targets(&rels),
            None => Vec::new(),
        };
        let mut notes = String::new();
        for target in rel_targets {
            let entry = resolve_rel_target("ppt/slides", &target);
            let Some(part_xml) = read_zip_entry(path, &entry)? else {
                continue;
            };
            if entry.contains("/notesSlides/") {
                notes = collect_xml_text(&part_xml, false)
                    .into_iter()
                    .filter(|line| !is_slide_number_noise(line, page))
                    .collect::<Vec<_>>()
                    .join(" ");
            } else if entry.contains("/diagrams/data") || entry.contains("/charts/chart") {
                // SmartArt labels and chart titles/series are content the shapes
                // themselves never spell out.
                parts.extend(collect_xml_text(&part_xml, entry.contains("/charts/chart")));
            }
        }

        let body = parts.join(" · ");
        if !body.trim().is_empty() {
            // Lead with the slide reference so the model can cite it in prose,
            // the same shape as the xlsx `Sheet!row` prefix.
            blocks.push(ExtractedBlock::new(
                format!("Slide {page} · {body}"),
                "heading",
                page,
            ));
        }
        if !notes.trim().is_empty() {
            // A separate block: notes are narration rather than what is on screen,
            // and keeping them apart lets retrieval surface either one alone.
            blocks.push(ExtractedBlock::new(
                format!("Slide {page} notes · {}", notes.trim()),
                "body",
                page,
            ));
        }
    }
    Ok(blocks)
}

/// A picture pulled out of a deck, ready to register as a visual asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PptxImage {
    /// 1-based slide the picture appears on.
    pub page: u32,
    /// Where the bytes were written on disk.
    pub image_path: String,
}

/// Below this a picture is a bullet glyph, divider or icon — registering those
/// would bury the real figures in noise.
const MIN_IMAGE_BYTES: usize = 16 * 1024;

fn is_image_entry(entry: &str) -> bool {
    let lower = entry.to_ascii_lowercase();
    // Only what a vision model can actually read; decks also carry emf/wmf/video.
    [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
        .iter()
        .any(|ext| lower.ends_with(ext))
}

/// Export a deck's slide pictures to `out_dir` and report which slide each one
/// belongs to.
///
/// Slide text alone misses most of what a deck communicates — this one carries
/// 129 pictures and averages ~100 characters of text per slide, so the
/// architecture diagrams and screenshots are simply absent from the index.
/// Registering them as visual assets lets the existing vision tools read them,
/// with no pptx-specific plumbing.
///
/// Two kinds of noise are filtered: anything tiny (icons, rules) and anything
/// that appears on more than half the slides (logos, template furniture).
pub(crate) fn extract_pptx_media(path: &Path, out_dir: &Path) -> Result<Vec<PptxImage>, String> {
    let mut slide_names: Vec<String> = zip_entry_names(path)?
        .into_iter()
        .filter(|name| name.starts_with("ppt/slides/slide") && name.ends_with(".xml"))
        .collect();
    slide_names.sort_by_key(|name| slide_number(name));
    let slide_count = slide_names.len();
    if slide_count == 0 {
        return Ok(Vec::new());
    }

    // (slide, media entry) pairs, plus how many distinct slides use each entry.
    let mut usages: Vec<(u32, String)> = Vec::new();
    let mut slides_per_entry: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (index, name) in slide_names.iter().enumerate() {
        let page = (index + 1) as u32;
        let rels_name = name.replace("ppt/slides/", "ppt/slides/_rels/") + ".rels";
        let Some(rels) = read_zip_entry(path, &rels_name)? else {
            continue;
        };
        let mut seen_on_this_slide = std::collections::HashSet::new();
        for target in parse_rel_targets(&rels) {
            let entry = resolve_rel_target("ppt/slides", &target);
            if !entry.contains("/media/") || !is_image_entry(&entry) {
                continue;
            }
            if !seen_on_this_slide.insert(entry.clone()) {
                continue;
            }
            *slides_per_entry.entry(entry.clone()).or_insert(0) += 1;
            usages.push((page, entry));
        }
    }

    let template_threshold = (slide_count / 2).max(1);
    let file = std::fs::File::open(path).map_err(|err| format!("Failed to open file: {err}"))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|err| format!("Not a valid Office (ZIP) file: {err}"))?;
    let mut written: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut images = Vec::new();
    for (page, entry) in usages {
        if slides_per_entry.get(&entry).copied().unwrap_or(0) > template_threshold {
            continue;
        }
        // One file on disk even when a picture is reused across slides.
        let image_path = match written.get(&entry) {
            Some(existing) => existing.clone(),
            None => {
                let mut zf = match archive.by_name(&entry) {
                    Ok(zf) => zf,
                    Err(_) => continue,
                };
                let mut bytes = Vec::new();
                if zf.read_to_end(&mut bytes).is_err() || bytes.len() < MIN_IMAGE_BYTES {
                    continue;
                }
                let file_name = entry.rsplit('/').next().unwrap_or("image");
                let out_path = out_dir.join(sanitize_media_name(file_name));
                if let Some(parent) = out_path.parent() {
                    std::fs::create_dir_all(parent).map_err(|err| {
                        format!(
                            "Failed to create media directory {}: {err}",
                            parent.display()
                        )
                    })?;
                }
                std::fs::write(&out_path, &bytes)
                    .map_err(|err| format!("Failed to write {}: {err}", out_path.display()))?;
                let as_string = out_path.to_string_lossy().to_string();
                written.insert(entry.clone(), as_string.clone());
                as_string
            }
        };
        images.push(PptxImage { page, image_path });
    }
    Ok(images)
}

fn sanitize_media_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `Target="…"` values from a `.rels` part, in document order.
fn parse_rel_targets(xml: &str) -> Vec<String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut targets = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) => {
                if e.name().as_ref() == b"Relationship" {
                    if let Some(target) = attr_value(&e, b"Target") {
                        targets.push(target);
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    targets
}

/// Resolve a relationship target against the part's directory, collapsing the
/// leading `../` segments OOXML uses ("../notesSlides/x.xml" from "ppt/slides"
/// → "ppt/notesSlides/x.xml").
fn resolve_rel_target(base_dir: &str, target: &str) -> String {
    if let Some(absolute) = target.strip_prefix('/') {
        return absolute.to_string();
    }
    let mut segments: Vec<&str> = base_dir.split('/').filter(|s| !s.is_empty()).collect();
    for segment in target.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    segments.join("/")
}

/// Readable text from an arbitrary OOXML part: `<a:t>` runs, plus `<c:v>` chart
/// values when `include_chart_values`. Blank and duplicate-adjacent runs are
/// dropped; chart values are capped since a data series can be huge.
fn collect_xml_text(xml: &str, include_chart_values: bool) -> Vec<String> {
    const MAX_CHART_VALUES: usize = 60;
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut capture = false;
    let mut is_chart_value = false;
    let mut chart_values = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"a:t" => {
                    capture = true;
                    is_chart_value = false;
                    current.clear();
                }
                b"c:v" if include_chart_values => {
                    capture = true;
                    is_chart_value = true;
                    current.clear();
                }
                _ => {}
            },
            Ok(Event::Text(e)) if capture => {
                if let Ok(text) = e.xml_content(quick_xml::XmlVersion::Implicit1_0) {
                    current.push_str(&text);
                }
            }
            Ok(Event::GeneralRef(e)) if capture => {
                if let Some(text) = resolve_entity(&e) {
                    current.push_str(&text);
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name();
                if (name.as_ref() == b"a:t") || (include_chart_values && name.as_ref() == b"c:v") {
                    capture = false;
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        if is_chart_value {
                            if chart_values < MAX_CHART_VALUES {
                                chart_values += 1;
                                out.push(trimmed.to_string());
                            }
                        } else {
                            out.push(trimmed.to_string());
                        }
                    }
                    current.clear();
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out.dedup();
    out
}

fn slide_number(name: &str) -> u32 {
    name.trim_start_matches("ppt/slides/slide")
        .trim_end_matches(".xml")
        .parse()
        .unwrap_or(u32::MAX)
}

fn parse_pptx_slide(xml: &str) -> Result<Vec<String>, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    let mut in_text = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"a:p" => current.clear(),
                b"a:t" => in_text = true,
                _ => {}
            },
            Ok(Event::Text(e)) if in_text => {
                if let Ok(text) = e.xml_content(quick_xml::XmlVersion::Implicit1_0) {
                    current.push_str(&text);
                }
            }
            Ok(Event::GeneralRef(e)) if in_text => {
                if let Some(text) = resolve_entity(&e) {
                    current.push_str(&text);
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"a:t" => in_text = false,
                b"a:p" => {
                    if !current.trim().is_empty() {
                        paragraphs.push(current.clone());
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(err) => return Err(format!("Failed to parse pptx slide XML: {err}")),
            _ => {}
        }
    }
    Ok(paragraphs)
}

// ---------------------------------------------------------------------------
// XLSX — calamine reads cells directly. One heading per sheet, then one block
// per non-empty row (cells joined with " | ").
// ---------------------------------------------------------------------------

/// Extract a spreadsheet into indexable blocks. Each data row becomes ONE block
/// (= one retrieval chunk) rendered as a header-keyed record, e.g.
/// "Region: West | Qty: 5 | Revenue: 3140". Two deliberate choices:
///   * column identity travels *with each value* (the header key), so a row is
///     self-describing even after chunking and search returns a complete record;
///   * cells are read positionally — the old code dropped empty cells, which
///     silently shifted every value left of a gap into the wrong column.
/// The full aligned grid with A1 coordinates is the `read_sheet` tool's job.
fn extract_xlsx(path: &Path) -> Result<Vec<ExtractedBlock>, String> {
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(path).map_err(|err| format!("Failed to open xlsx: {err}"))?;
    let mut blocks = Vec::new();
    for name in workbook.sheet_names().to_vec() {
        let Ok(range) = workbook.worksheet_range(&name) else {
            continue;
        };
        let (height, width) = (range.height(), range.width());
        blocks.push(ExtractedBlock::new(
            format!("Sheet: {name} ({height} rows × {width} cols)"),
            "heading",
            0,
        ));
        if width == 0 {
            continue;
        }
        // The first row carrying any value is the header; its labels key the data
        // rows below. A pure title row above the header degrades gracefully (the
        // full grid is available via read_sheet).
        let (start_row, _) = range.start().unwrap_or((0, 0));
        let mut header: Vec<String> = Vec::new();
        let mut header_seen = false;
        for (offset, row) in range.rows().enumerate() {
            // Real 1-based worksheet row, honoring a used range that starts below A1.
            let row_no = start_row as usize + offset + 1;
            if !header_seen {
                let cells: Vec<String> = row.iter().map(cell_to_string).collect();
                if cells.iter().any(|cell| !cell.is_empty()) {
                    let labels: Vec<&str> = cells
                        .iter()
                        .filter(|cell| !cell.is_empty())
                        .map(String::as_str)
                        .collect();
                    blocks.push(ExtractedBlock::new(
                        format!("Columns: {}", labels.join(" | ")),
                        "body",
                        0,
                    ));
                    header = cells;
                    header_seen = true;
                }
                continue;
            }
            if let Some(record) = xlsx_record(&header, row) {
                // Lead with an Excel-style `Sheet!row` reference. It lets the model
                // cite a row precisely, and it is what the viewer parses back out of
                // a citation to scroll to and highlight that row — the indexed text
                // ("Region: West | …") deliberately does not match the rendered
                // cells, so a text search could never find it.
                blocks.push(ExtractedBlock::new(
                    format!("{name}!{row_no} · {record}"),
                    "body",
                    0,
                ));
            }
        }
        // One block of formulas per sheet, so "how is the total computed?" is
        // retrievable (the value rows only carry results). Capped to stay a chunk.
        if let Ok(formula_range) = workbook.worksheet_formula(&name) {
            let formulas = collect_formulas(&formula_range);
            if !formulas.is_empty() {
                const MAX_FORMULAS: usize = 50;
                let shown: Vec<String> = formulas
                    .iter()
                    .take(MAX_FORMULAS)
                    .map(|(address, formula)| format!("{address}: {formula}"))
                    .collect();
                let mut text = format!("Formulas in {name} — {}", shown.join(" | "));
                let extra = formulas.len().saturating_sub(MAX_FORMULAS);
                if extra > 0 {
                    text.push_str(&format!(" | (+{extra} more)"));
                }
                blocks.push(ExtractedBlock::new(text, "body", 0));
            }
        }
    }
    Ok(blocks)
}

/// One data row → a header-keyed record ("Region: West | Revenue: 3140"), or None
/// if the row has no values. Cells are read POSITIONALLY and keyed by the header
/// at the same index, so an empty cell in the middle can't shift the values right
/// of it into the wrong column (the bug in the drop-empties approach). Returns
/// None for an all-empty row.
fn xlsx_record(header: &[String], row: &[Data]) -> Option<String> {
    let record: Vec<String> = row
        .iter()
        .enumerate()
        .filter_map(|(col, cell)| {
            let value = cell_to_string(cell);
            if value.is_empty() {
                return None;
            }
            match header
                .get(col)
                .map(String::as_str)
                .filter(|h| !h.is_empty())
            {
                Some(key) => Some(format!("{key}: {value}")),
                None => Some(value),
            }
        })
        .collect();
    if record.is_empty() {
        None
    } else {
        Some(record.join(" | "))
    }
}

/// Spreadsheet column index (0-based) → letter(s): 0→A, 25→Z, 26→AA. Used to give
/// the agent real A1 addresses when it reads a sheet.
fn column_letter(index: u32) -> String {
    let mut n = index + 1;
    let mut label = String::new();
    while n > 0 {
        let rem = ((n - 1) % 26) as u8;
        label.insert(0, (b'A' + rem) as char);
        n = (n - 1) / 26;
    }
    label
}

/// The non-empty formulas of a sheet as `(A1 address, formula)` pairs, e.g.
/// ("B7", "=SUM(B2:B6)"). calamine stores the formula body without the leading
/// `=`, so we add it. The value grid shows only computed results; this exposes
/// the logic behind them.
fn collect_formulas(formulas: &calamine::Range<String>) -> Vec<(String, String)> {
    let (start_row, start_col) = formulas.start().unwrap_or((0, 0));
    let mut out = Vec::new();
    for (r, row) in formulas.rows().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            let body = cell.trim();
            if body.is_empty() {
                continue;
            }
            let address = format!(
                "{}{}",
                column_letter(start_col + c as u32),
                start_row as usize + r + 1
            );
            let formula = if body.starts_with('=') {
                body.to_string()
            } else {
                format!("={body}")
            };
            out.push((address, formula));
        }
    }
    out
}

/// Render a spreadsheet as A1-addressable Markdown tables for the `read_sheet`
/// tool: a `#` row-number column plus one column per letter, so the agent can
/// cite "B7". `sheet_filter` selects one sheet by name (case-insensitive); None
/// renders all. Capped at `max_cells` total so a huge workbook can't blow the
/// context — truncation is stated in the text.
pub(crate) fn read_xlsx_markdown(
    path: &Path,
    sheet_filter: Option<&str>,
    max_cells: usize,
) -> Result<String, String> {
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook(path).map_err(|err| format!("Failed to open xlsx: {err}"))?;
    let all_names = workbook.sheet_names().to_vec();
    let wanted = sheet_filter.map(str::trim).filter(|s| !s.is_empty());
    let names: Vec<String> = match wanted {
        Some(filter) => all_names
            .iter()
            .filter(|n| n.eq_ignore_ascii_case(filter))
            .cloned()
            .collect(),
        None => all_names.clone(),
    };
    if names.is_empty() {
        return Err(format!(
            "No sheet named '{}'. Sheets: {}.",
            wanted.unwrap_or(""),
            all_names.join(", ")
        ));
    }

    let mut out = String::new();
    let mut cells_used = 0usize;
    for name in names {
        let Ok(range) = workbook.worksheet_range(&name) else {
            continue;
        };
        let (height, width) = (range.height(), range.width());
        out.push_str(&format!(
            "### Sheet: {name} ({height} rows × {width} cols)\n\n"
        ));
        if width == 0 || height == 0 {
            out.push_str("_(empty)_\n\n");
            continue;
        }
        let (start_row, start_col) = range.start().unwrap_or((0, 0));
        out.push_str("| # |");
        for col in 0..width as u32 {
            out.push_str(&format!(" {} |", column_letter(start_col + col)));
        }
        out.push('\n');
        out.push_str("|---|");
        for _ in 0..width {
            out.push_str("---|");
        }
        out.push('\n');
        for (index, row) in range.rows().enumerate() {
            if cells_used + width > max_cells {
                out.push_str(&format!(
                    "\n_[truncated: {height} rows total; showing the first {index}]_\n"
                ));
                break;
            }
            out.push_str(&format!("| {} |", start_row as usize + index + 1));
            for cell in row {
                let value = cell_to_string(cell).replace('|', "\\|").replace('\n', " ");
                out.push_str(&format!(" {value} |"));
            }
            out.push('\n');
            cells_used += width;
        }
        out.push('\n');
        // The grid shows computed values; list the formulas behind them with their
        // A1 addresses so the model can explain or verify a number.
        if let Ok(formula_range) = workbook.worksheet_formula(&name) {
            let formulas = collect_formulas(&formula_range);
            if !formulas.is_empty() {
                const MAX_FORMULAS: usize = 200;
                out.push_str("**Formulas**\n\n");
                for (address, formula) in formulas.iter().take(MAX_FORMULAS) {
                    out.push_str(&format!("- {address}: {formula}\n"));
                }
                if formulas.len() > MAX_FORMULAS {
                    out.push_str(&format!("- _(+{} more)_\n", formulas.len() - MAX_FORMULAS));
                }
                out.push('\n');
            }
        }
    }
    Ok(out)
}

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(s) => s.trim().to_string(),
        Data::Float(f) => {
            // Render integers without a trailing ".0".
            if f.fract() == 0.0 && f.abs() < 1e15 {
                format!("{}", *f as i64)
            } else {
                f.to_string()
            }
        }
        Data::Int(i) => i.to_string(),
        Data::Bool(b) => b.to_string(),
        // ExcelDateTime's own Display prints the raw serial number (e.g. "44484"),
        // which is meaningless. Convert to a real calendar date; drop a midnight
        // time so a plain date reads "2021-10-15", a timestamp "2021-10-15 19:00".
        Data::DateTime(dt) => cell
            .as_datetime()
            .map(|ndt| {
                let text = ndt.format("%Y-%m-%d %H:%M").to_string();
                text.strip_suffix(" 00:00")
                    .map(str::to_string)
                    .unwrap_or(text)
            })
            .unwrap_or_else(|| dt.as_f64().to_string()),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("{e:?}"),
    }
}

/// Resolve a `&ref;` entity event: numeric char refs (`&#123;`) and the five
/// predefined XML entities. Unknown named entities are dropped.
fn resolve_entity(e: &quick_xml::events::BytesRef<'_>) -> Option<String> {
    if let Ok(Some(ch)) = e.resolve_char_ref() {
        return Some(ch.to_string());
    }
    match e.decode().ok()?.as_ref() {
        "amp" => Some("&".to_string()),
        "lt" => Some("<".to_string()),
        "gt" => Some(">".to_string()),
        "quot" => Some("\"".to_string()),
        "apos" => Some("'".to_string()),
        _ => None,
    }
}

fn attr_value(e: &quick_xml::events::BytesStart<'_>, key: &[u8]) -> Option<String> {
    e.attributes().flatten().find_map(|attr| {
        if attr.key.as_ref() == key {
            attr.unescape_value().ok().map(|value| value.to_string())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn extract_docx_reads_paragraphs_and_headings() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lumenfolio-docx-{}-{suffix}.docx",
            std::process::id()
        ));
        let file = std::fs::File::create(&path).expect("create");
        let mut writer = zip::ZipWriter::new(file);
        writer
            .start_file(
                "word/document.xml",
                zip::write::SimpleFileOptions::default(),
            )
            .expect("entry");
        let xml = r#"<?xml version="1.0"?><w:document xmlns:w="x"><w:body>
            <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>My Title</w:t></w:r></w:p>
            <w:p><w:r><w:t xml:space="preserve">Hello </w:t></w:r><w:r><w:t>world &amp; more.</w:t></w:r></w:p>
            <w:p><w:r><w:t>  </w:t></w:r></w:p>
        </w:body></w:document>"#;
        writer.write_all(xml.as_bytes()).expect("write");
        writer.finish().expect("finish");

        let blocks = extract_office_blocks(&path, "docx").expect("extract");
        assert_eq!(
            blocks,
            vec![
                ExtractedBlock::new("My Title", "heading", 0),
                // Runs concatenate within a paragraph; entity is unescaped; the
                // whitespace-only paragraph is dropped.
                ExtractedBlock::new("Hello world & more.", "body", 0),
            ]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn xlsx_cell_renders_integer_floats_without_decimal() {
        assert_eq!(cell_to_string(&Data::Float(42.0)), "42");
        assert_eq!(cell_to_string(&Data::Float(3.5)), "3.5");
        assert_eq!(cell_to_string(&Data::String("  hi ".to_string())), "hi");
        assert_eq!(cell_to_string(&Data::Empty), "");
    }

    #[test]
    fn xlsx_record_keeps_columns_aligned_across_empty_cells() {
        let header = vec![
            "Region".to_string(),
            "Qty".to_string(),
            "Revenue".to_string(),
        ];
        // Qty (middle) is empty. The old drop-empties join produced "West | 3140",
        // shifting Revenue under Qty; header keys pin each value to its column.
        let row = [
            Data::String("West".to_string()),
            Data::Empty,
            Data::Float(3140.0),
        ];
        let record = xlsx_record(&header, &row).expect("record");
        assert_eq!(record, "Region: West | Revenue: 3140");
    }

    #[test]
    fn xlsx_record_is_none_for_a_blank_row() {
        let header = vec!["A".to_string()];
        assert!(xlsx_record(&header, &[Data::Empty, Data::Empty]).is_none());
    }

    #[test]
    fn column_letter_maps_indices_to_a1_letters() {
        assert_eq!(column_letter(0), "A");
        assert_eq!(column_letter(25), "Z");
        assert_eq!(column_letter(26), "AA");
        assert_eq!(column_letter(701), "ZZ");
        assert_eq!(column_letter(702), "AAA");
    }

    #[test]
    fn xlsx_datetime_renders_a_calendar_date_not_the_serial() {
        // Serial 44484 in the 1900 date system is 2021-10-15; the old Display path
        // printed "44484". A midnight time is dropped; a real time is kept.
        let date =
            calamine::ExcelDateTime::new(44484.0, calamine::ExcelDateTimeType::DateTime, false);
        assert_eq!(cell_to_string(&Data::DateTime(date)), "2021-10-15");
        let stamp =
            calamine::ExcelDateTime::new(44484.5, calamine::ExcelDateTimeType::DateTime, false);
        assert_eq!(cell_to_string(&Data::DateTime(stamp)), "2021-10-15 12:00");
    }

    #[test]
    fn collect_formulas_addresses_cells_and_adds_the_equals() {
        let cells = vec![
            calamine::Cell::new((0, 1), "SUM(B1:B3)".to_string()), // B1, stored without '='
            calamine::Cell::new((6, 1), "=B1*2".to_string()),      // B7, already has '='
        ];
        let range = calamine::Range::from_sparse(cells);
        assert_eq!(
            collect_formulas(&range),
            vec![
                ("B1".to_string(), "=SUM(B1:B3)".to_string()),
                ("B7".to_string(), "=B1*2".to_string()),
            ]
        );
    }

    #[test]
    fn slide_number_sorts_numerically() {
        assert!(slide_number("ppt/slides/slide2.xml") < slide_number("ppt/slides/slide10.xml"));
    }

    #[test]
    fn resolve_rel_target_collapses_parent_segments() {
        assert_eq!(
            resolve_rel_target("ppt/slides", "../notesSlides/notesSlide3.xml"),
            "ppt/notesSlides/notesSlide3.xml"
        );
        assert_eq!(
            resolve_rel_target("ppt/slides", "../diagrams/data1.xml"),
            "ppt/diagrams/data1.xml"
        );
        // Same-directory and absolute forms both appear in the wild.
        assert_eq!(
            resolve_rel_target("ppt/slides", "slide2.xml"),
            "ppt/slides/slide2.xml"
        );
        assert_eq!(
            resolve_rel_target("ppt/slides", "/ppt/media/image1.png"),
            "ppt/media/image1.png"
        );
    }

    #[test]
    fn collect_xml_text_reads_runs_and_gates_chart_values() {
        let xml = r#"<r xmlns:a="x" xmlns:c="y"><a:t>Revenue</a:t><c:v>42</c:v><a:t>  </a:t></r>"#;
        // Chart values are opt-in: a diagram part must not pull in numeric noise.
        assert_eq!(collect_xml_text(xml, false), vec!["Revenue".to_string()]);
        assert_eq!(
            collect_xml_text(xml, true),
            vec!["Revenue".to_string(), "42".to_string()]
        );
    }

    /// The P2/P3 contract end to end: one block per slide carrying the real slide
    /// number, notes pulled in through the slide's .rels as their own block.
    #[test]
    fn extract_pptx_emits_one_block_per_slide_with_notes() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lumenfolio-pptx-{}-{suffix}.pptx",
            std::process::id()
        ));
        let file = std::fs::File::create(&path).expect("create");
        let mut writer = zip::ZipWriter::new(file);
        let opts = zip::write::SimpleFileOptions::default();
        let mut put = |name: &str, body: &str| {
            writer.start_file(name, opts).expect("entry");
            writer.write_all(body.as_bytes()).expect("write");
        };
        put(
            "ppt/slides/slide1.xml",
            r#"<p:sld xmlns:a="x"><a:p><a:r><a:t>企业知识库</a:t></a:r></a:p><a:p><a:t>升级方案</a:t></a:p></p:sld>"#,
        );
        // Deliberately notesSlide7 for slide1: the numbers do NOT have to line up,
        // which is why the link is resolved through the .rels part.
        put(
            "ppt/slides/_rels/slide1.xml.rels",
            r#"<Relationships><Relationship Id="rId1" Target="../notesSlides/notesSlide7.xml"/></Relationships>"#,
        );
        put(
            "ppt/notesSlides/notesSlide7.xml",
            r#"<p:notes xmlns:a="x"><a:p><a:t>各位来宾上午好</a:t></a:p><a:p><a:t>1</a:t></a:p></p:notes>"#,
        );
        put(
            "ppt/slides/slide2.xml",
            r#"<p:sld xmlns:a="x"><a:p><a:t>第二页</a:t></a:p></p:sld>"#,
        );
        writer.finish().expect("finish");

        let blocks = extract_office_blocks(&path, "pptx").expect("extract");
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            blocks,
            vec![
                // Paragraphs are joined into ONE block per slide (was one block per
                // paragraph, averaging 7 characters), prefixed for prose citation.
                ExtractedBlock::new("Slide 1 · 企业知识库 · 升级方案", "heading", 1),
                // Notes ride on the same page but stay a separate block. The bare
                // "1" run is slide-number noise and is dropped.
                ExtractedBlock::new("Slide 1 notes · 各位来宾上午好", "body", 1),
                ExtractedBlock::new("Slide 2 · 第二页", "heading", 2),
            ]
        );
    }

    /// Pictures are exported per slide, with the two noise classes filtered: a
    /// tiny icon and a logo repeated across the deck.
    #[test]
    fn extract_pptx_media_filters_icons_and_template_logos() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lumenfolio-pptxmedia-{}-{suffix}.pptx",
            std::process::id()
        ));
        let out_dir = std::env::temp_dir().join(format!("lumenfolio-pptxmedia-out-{suffix}"));
        let file = std::fs::File::create(&path).expect("create");
        let mut writer = zip::ZipWriter::new(file);
        let opts = zip::write::SimpleFileOptions::default();
        let big = vec![b'x'; MIN_IMAGE_BYTES + 1];
        let small = vec![b'x'; 128];
        {
            let mut put = |name: &str, body: &[u8]| {
                writer.start_file(name, opts).expect("entry");
                writer.write_all(body).expect("write");
            };
            for slide in 1..=4 {
                put(
                    &format!("ppt/slides/slide{slide}.xml"),
                    br#"<p:sld xmlns:a="x"><a:p><a:t>t</a:t></a:p></p:sld>"#,
                );
                // Every slide carries the logo; only slide 1 has a real figure.
                let extra = if slide == 1 {
                    r#"<Relationship Id="rId2" Target="../media/figure.png"/>
                       <Relationship Id="rId3" Target="../media/icon.png"/>
                       <Relationship Id="rId4" Target="../media/movie.mp4"/>"#
                } else {
                    ""
                };
                put(
                    &format!("ppt/slides/_rels/slide{slide}.xml.rels"),
                    format!(
                        r#"<Relationships><Relationship Id="rId1" Target="../media/logo.png"/>{extra}</Relationships>"#
                    )
                    .as_bytes(),
                );
            }
            put("ppt/media/figure.png", &big);
            put("ppt/media/logo.png", &big);
            put("ppt/media/icon.png", &small);
            put("ppt/media/movie.mp4", &big);
        }
        writer.finish().expect("finish");

        let images = extract_pptx_media(&path, &out_dir).expect("media");
        let _ = std::fs::remove_file(&path);

        // figure.png only: the logo is on every slide (template furniture), the
        // icon is under the size floor, and the video is not a readable image.
        assert_eq!(images.len(), 1, "unexpected: {images:?}");
        assert_eq!(images[0].page, 1);
        assert!(images[0].image_path.ends_with("figure.png"), "{images:?}");
        assert!(std::path::Path::new(&images[0].image_path).exists());
        let _ = std::fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn parse_pptx_slide_collects_text_runs_per_paragraph() {
        let xml = r#"<p:sld xmlns:a="x"><a:p><a:r><a:t>Hello </a:t></a:r><a:r><a:t>world</a:t></a:r></a:p><a:p><a:t>Second</a:t></a:p></p:sld>"#;
        let paragraphs = parse_pptx_slide(xml).expect("parse");
        assert_eq!(
            paragraphs,
            vec!["Hello world".to_string(), "Second".to_string()]
        );
    }
}
