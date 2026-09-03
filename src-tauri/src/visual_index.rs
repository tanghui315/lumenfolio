use rusqlite::{params, OptionalExtension};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};
use tauri::{Emitter, Manager, State};

use crate::{
    runtime, truncate_for_error, vision, AgentSessionState, AppDatabase, VisualIndexEventOutput,
    VisualIndexResult, CURRENT_INDEX_VERSION, MAX_RUST_TSR_TABLES, MAX_RUST_VISUAL_CROPS,
    MIN_RUST_TSR_CONFIDENCE,
};

#[tauri::command]
pub(crate) fn run_document_visual_index(
    document_id: String,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
) -> Result<VisualIndexResult, String> {
    run_document_visual_index_job(&document_id, &database, &agent_sessions)
}

#[tauri::command]
pub(crate) fn enqueue_document_visual_index(
    document_id: String,
    app: tauri::AppHandle,
) -> Result<VisualIndexResult, String> {
    let document_id = document_id.trim().to_string();
    if document_id.is_empty() {
        return Err("No document selected for visual index".to_string());
    }
    let database = app.state::<AppDatabase>();
    let queued_now = queue_visual_index_command(&document_id, &database)?;
    let queued = VisualIndexResult {
        document_id: document_id.clone(),
        status: if queued_now { "queued" } else { "running" }.to_string(),
        version: CURRENT_INDEX_VERSION,
        error: String::new(),
        visual_assets: 0,
        tables: 0,
        table_facts: 0,
    };
    if !queued_now {
        return Ok(queued);
    }
    let task_app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        emit_visual_index_event(
            &task_app,
            VisualIndexEventOutput {
                document_id: document_id.clone(),
                status: "running".to_string(),
                version: CURRENT_INDEX_VERSION,
                error: String::new(),
                visual_assets: 0,
                tables: 0,
                table_facts: 0,
            },
        );
        let result = {
            let database = task_app.state::<AppDatabase>();
            let agent_sessions = task_app.state::<AgentSessionState>();
            run_document_visual_index_job(&document_id, &database, &agent_sessions)
        };
        let event = match result {
            Ok(result) => VisualIndexEventOutput::from(result),
            Err(err) => VisualIndexEventOutput {
                document_id,
                status: "failed".to_string(),
                version: CURRENT_INDEX_VERSION,
                error: err,
                visual_assets: 0,
                tables: 0,
                table_facts: 0,
            },
        };
        emit_visual_index_event(&task_app, event);
    });
    Ok(queued)
}

pub(super) fn queue_visual_index_command(
    document_id: &str,
    database: &AppDatabase,
) -> Result<bool, String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual index queue transaction: {err}"))?;
    let document_exists = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM documents WHERE id = ?1 LIMIT 1)",
            params![document_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value != 0)
        .map_err(|err| format!("Failed to inspect visual index document: {err}"))?;
    if !document_exists {
        return Err("Document is no longer available for visual index".to_string());
    }
    let existing = tx
        .query_row(
            "SELECT status
             FROM document_index_jobs
             WHERE document_id = ?1 AND job_type = 'visual_tsr'",
            params![document_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|err| format!("Failed to inspect visual index job: {err}"))?;
    if matches!(existing.as_deref(), Some("queued" | "running")) {
        return Ok(false);
    }
    tx.execute(
        "INSERT INTO document_index_jobs
            (id, document_id, job_type, status, version, attempts, error, created_at, updated_at, started_at, finished_at)
         VALUES (?1, ?2, 'visual_tsr', 'queued', ?3, 0, '', unixepoch(), unixepoch(), 0, 0)
         ON CONFLICT(document_id, job_type) DO UPDATE SET
           status = 'queued',
           version = excluded.version,
           error = '',
           updated_at = unixepoch(),
           started_at = 0,
           finished_at = 0",
        params![
            format!("index-job-{document_id}-visual-tsr"),
            document_id,
            CURRENT_INDEX_VERSION,
        ],
    )
    .map_err(|err| format!("Failed to queue visual index job: {err}"))?;
    tx.commit()
        .map_err(|err| format!("Failed to commit visual index queue: {err}"))?;
    Ok(true)
}

fn run_document_visual_index_job(
    document_id: &str,
    database: &AppDatabase,
    agent_sessions: &AgentSessionState,
) -> Result<VisualIndexResult, String> {
    let result = mark_visual_index_job_running_db(database, document_id)
        .and_then(|_| rebuild_visual_evidence_job(database, document_id))
        .and_then(|_| enrich_visual_assets_with_rust_crops(database, document_id))
        .and_then(|_| enrich_visual_evidence_with_rust_tsr(database, document_id))
        .and_then(|_| mark_visual_index_job_succeeded_db(database, document_id));
    match result {
        Ok((visual_assets, tables, table_facts)) => {
            agent_sessions.clear_session(&crate::migrated_session_id(document_id));
            Ok(VisualIndexResult {
                document_id: document_id.to_string(),
                status: "succeeded".to_string(),
                version: CURRENT_INDEX_VERSION,
                error: String::new(),
                visual_assets,
                tables,
                table_facts,
            })
        }
        Err(err) => {
            mark_visual_index_job_failed_db(database, document_id, &err)?;
            Ok(VisualIndexResult {
                document_id: document_id.to_string(),
                status: "failed".to_string(),
                version: CURRENT_INDEX_VERSION,
                error: err,
                visual_assets: 0,
                tables: 0,
                table_facts: 0,
            })
        }
    }
}

impl From<VisualIndexResult> for VisualIndexEventOutput {
    fn from(value: VisualIndexResult) -> Self {
        Self {
            document_id: value.document_id,
            status: value.status,
            version: value.version,
            error: value.error,
            visual_assets: value.visual_assets,
            tables: value.tables,
            table_facts: value.table_facts,
        }
    }
}

fn emit_visual_index_event(app: &tauri::AppHandle, event: VisualIndexEventOutput) {
    if let Err(err) = app.emit("lumenfolio://visual-index", event) {
        log::warn!("Failed to emit visual-index event: {err}");
    }
}

fn mark_visual_index_job_running_db(
    database: &AppDatabase,
    document_id: &str,
) -> Result<(), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual index running transaction: {err}"))?;
    mark_visual_index_job_running(&tx, document_id)?;
    tx.commit()
        .map_err(|err| format!("Failed to commit visual index running state: {err}"))
}

fn rebuild_visual_evidence_job(database: &AppDatabase, document_id: &str) -> Result<(), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual evidence rebuild transaction: {err}"))?;
    runtime::rag::rebuild_visual_evidence(&tx, document_id)?;
    tx.commit()
        .map_err(|err| format!("Failed to commit visual evidence rebuild: {err}"))
}

fn mark_visual_index_job_succeeded_db(
    database: &AppDatabase,
    document_id: &str,
) -> Result<(u32, u32, u32), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual index finish transaction: {err}"))?;
    mark_visual_index_job_finished(&tx, document_id, "succeeded", "")?;
    let counts = visual_index_counts(&tx, document_id)?;
    tx.commit()
        .map_err(|err| format!("Failed to commit visual index finish: {err}"))?;
    Ok(counts)
}

fn mark_visual_index_job_failed_db(
    database: &AppDatabase,
    document_id: &str,
    error: &str,
) -> Result<(), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual index failure transaction: {err}"))?;
    mark_visual_index_job_finished(&tx, document_id, "failed", error)?;
    tx.commit()
        .map_err(|err| format!("Failed to commit visual index failure: {err}"))
}

struct VisualCropUpdate {
    asset_id: String,
    crop_path: String,
    /// OCR text recognized inside the crop. The crop image is still kept as the
    /// primary evidence; this is additive searchable text (empty when OCR is
    /// disabled or finds nothing).
    ocr_text: String,
}

struct TsrRecognitionPlan {
    asset: vision::VisualTableAsset,
    lines: Vec<vision::PdfTextLine>,
}

struct TsrRecognitionUpdate {
    asset: vision::VisualTableAsset,
    response: vision::TableRecognitionResult,
}

fn enrich_visual_evidence_with_rust_tsr(
    database: &AppDatabase,
    document_id: &str,
) -> Result<(), String> {
    let (pdf_path, plans) = load_tsr_recognition_plan(database, document_id)?;
    let mut tsr_errors = Vec::new();
    let mut updates = Vec::new();
    for plan in plans {
        match vision::recognize_table(&pdf_path, &plan.asset) {
            Ok(Some(mut response)) if !response.cells.is_empty() => {
                vision::attach_pdf_text_to_table_cells(&mut response, &plan.asset, &plan.lines);
                if is_usable_tsr_result(&response) {
                    updates.push(TsrRecognitionUpdate {
                        asset: plan.asset,
                        response,
                    });
                } else {
                    log::warn!(
                        "Ignored low-quality Rust TSR result for {}: confidence={:.3} cells={}",
                        plan.asset.id,
                        response.confidence,
                        response.cells.len()
                    );
                }
            }
            Ok(_) => {}
            Err(err) => tsr_errors.push(format!("{}: {err}", plan.asset.id)),
        }
    }
    if !tsr_errors.is_empty() {
        log::warn!(
            "Rust TSR fell back to pdf_bbox evidence for {}: {}",
            document_id,
            tsr_errors.join("; ")
        );
    }
    write_tsr_recognition_updates(database, document_id, updates)?;
    Ok(())
}

pub(super) fn is_usable_tsr_result(response: &vision::TableRecognitionResult) -> bool {
    if response.confidence < MIN_RUST_TSR_CONFIDENCE {
        return false;
    }
    let text_cells = response
        .cells
        .iter()
        .filter(|cell| !cell.text.trim().is_empty())
        .collect::<Vec<_>>();
    if text_cells.is_empty() {
        return false;
    }
    let distinct_rows = text_cells
        .iter()
        .map(|cell| cell.row_index)
        .collect::<HashSet<_>>();
    let distinct_columns = text_cells
        .iter()
        .map(|cell| cell.col_index)
        .collect::<HashSet<_>>();
    let has_data_cell = text_cells
        .iter()
        .any(|cell| !cell.is_header && cell.row_index > 0 && cell.col_index > 0);
    distinct_rows.len() >= 2 && distinct_columns.len() >= 2 && has_data_cell
}

fn enrich_visual_assets_with_rust_crops(
    database: &AppDatabase,
    document_id: &str,
) -> Result<(), String> {
    let (pdf_path, assets) = load_visual_crop_plan(database, document_id)?;
    let mut crop_errors = Vec::new();
    let mut updates = Vec::new();
    for asset in assets {
        if !asset.image_path.trim().is_empty() && PathBuf::from(asset.image_path.trim()).is_file() {
            continue;
        }
        match vision::render_visual_asset_crop(&pdf_path, &asset) {
            Ok(Some(crop_path)) if !crop_path.trim().is_empty() => {
                // Recognize any text inside the crop and store it ALONGSIDE the
                // crop image (the image itself stays as the primary evidence for
                // multimodal use). OCR failure must never drop the crop.
                let ocr_text = ocr_text_for_crop(&crop_path);
                updates.push(VisualCropUpdate {
                    asset_id: asset.id,
                    crop_path,
                    ocr_text,
                });
            }
            Ok(_) => {}
            Err(err) => crop_errors.push(format!("{}: {err}", asset.id)),
        }
    }
    if !crop_errors.is_empty() {
        log::warn!(
            "Rust visual crop generation skipped for {}: {}",
            document_id,
            crop_errors.join("; ")
        );
    }
    write_visual_crop_updates(database, document_id, updates)?;
    Ok(())
}

/// OCR the recognized text inside a rendered crop image. Returns an empty
/// string when OCR is disabled (no `ocr-onnx` feature), the file can't be read,
/// or nothing is recognized — the crop image is never affected.
#[cfg(feature = "ocr-onnx")]
fn ocr_text_for_crop(crop_path: &str) -> String {
    let path = crop_path.trim();
    if path.is_empty() {
        return String::new();
    }
    let image = match image::ImageReader::open(path).and_then(|reader| {
        reader
            .decode()
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
    }) {
        Ok(image) => image.to_rgba8(),
        Err(err) => {
            log::warn!("OCR skipped for crop {path}: {err}");
            return String::new();
        }
    };
    let (w, h) = (image.width(), image.height());
    match vision::ocr_recognize_image_rgba(image.as_raw(), w, h) {
        Ok(lines) => lines
            .into_iter()
            .map(|line| line.text)
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Err(err) => {
            log::warn!("OCR failed for crop {path}: {err}");
            String::new()
        }
    }
}

#[cfg(not(feature = "ocr-onnx"))]
fn ocr_text_for_crop(_crop_path: &str) -> String {
    String::new()
}

fn load_visual_crop_plan(
    database: &AppDatabase,
    document_id: &str,
) -> Result<(PathBuf, Vec<vision::VisualTableAsset>), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual crop plan transaction: {err}"))?;
    let pdf_path = PathBuf::from(document_pdf_path(&tx, document_id)?);
    let assets = visual_assets_for_crop(&tx, document_id)?
        .into_iter()
        .take(MAX_RUST_VISUAL_CROPS)
        .collect::<Vec<_>>();
    tx.commit()
        .map_err(|err| format!("Failed to commit visual crop plan read: {err}"))?;
    Ok((pdf_path, assets))
}

fn write_visual_crop_updates(
    database: &AppDatabase,
    document_id: &str,
    updates: Vec<VisualCropUpdate>,
) -> Result<(), String> {
    if updates.is_empty() {
        return Ok(());
    }
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual crop update transaction: {err}"))?;
    for update in updates {
        update_visual_asset_crop_path(
            &tx,
            document_id,
            &update.asset_id,
            &update.crop_path,
            &update.ocr_text,
        )?;
    }
    tx.commit()
        .map_err(|err| format!("Failed to commit visual crop updates: {err}"))
}

fn load_tsr_recognition_plan(
    database: &AppDatabase,
    document_id: &str,
) -> Result<(PathBuf, Vec<TsrRecognitionPlan>), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual TSR plan transaction: {err}"))?;
    let pdf_path = PathBuf::from(document_pdf_path(&tx, document_id)?);
    let assets = table_visual_assets_for_tsr(&tx, document_id)?;
    let mut plans = Vec::new();
    for asset in assets.into_iter().take(MAX_RUST_TSR_TABLES) {
        let lines = page_text_lines_for_tsr(&tx, document_id, asset.page_no)?;
        plans.push(TsrRecognitionPlan { asset, lines });
    }
    tx.commit()
        .map_err(|err| format!("Failed to commit visual TSR plan read: {err}"))?;
    Ok((pdf_path, plans))
}

fn write_tsr_recognition_updates(
    database: &AppDatabase,
    document_id: &str,
    updates: Vec<TsrRecognitionUpdate>,
) -> Result<(), String> {
    if updates.is_empty() {
        return Ok(());
    }
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start visual TSR update transaction: {err}"))?;
    for update in updates {
        insert_tsr_table_result(&tx, document_id, &update.asset, &update.response)?;
    }
    tx.commit()
        .map_err(|err| format!("Failed to commit visual TSR updates: {err}"))
}

fn page_text_lines_for_tsr(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
    page_no: u32,
) -> Result<Vec<vision::PdfTextLine>, String> {
    let mut stmt = tx
        .prepare(
            "SELECT text, bbox_json
             FROM document_lines
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY line_no",
        )
        .map_err(|err| format!("Failed to prepare Rust TSR line read: {err}"))?;
    let rows = stmt
        .query_map(params![document_id, page_no], |row| {
            let bbox_json: String = row.get(1)?;
            Ok(vision::PdfTextLine {
                text: row.get(0)?,
                bbox: serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([])),
            })
        })
        .map_err(|err| format!("Failed to read Rust TSR page lines: {err}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect Rust TSR page lines: {err}"))
}

fn document_pdf_path(tx: &rusqlite::Transaction<'_>, document_id: &str) -> Result<String, String> {
    tx.query_row(
        "SELECT path FROM documents WHERE id = ?1 LIMIT 1",
        params![document_id],
        |row| row.get::<_, String>(0),
    )
    .map_err(|err| format!("Failed to load document path for Rust TSR: {err}"))
}

fn table_visual_assets_for_tsr(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<Vec<vision::VisualTableAsset>, String> {
    let mut stmt = tx
        .prepare(
            "SELECT id, page_no, asset_type, caption, bbox_json, image_path
             FROM document_visual_assets
             WHERE document_id = ?1 AND asset_type = 'table'
             ORDER BY page_no, id",
        )
        .map_err(|err| format!("Failed to prepare visual TSR asset read: {err}"))?;
    let rows = stmt
        .query_map(params![document_id], |row| {
            let bbox_json: String = row.get(4)?;
            Ok(vision::VisualTableAsset {
                id: row.get(0)?,
                page_no: row.get(1)?,
                asset_type: row.get(2)?,
                caption: row.get(3)?,
                bbox: serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([])),
                image_path: row.get(5)?,
            })
        })
        .map_err(|err| format!("Failed to read visual TSR assets: {err}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect visual TSR assets: {err}"))
}

fn visual_assets_for_crop(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<Vec<vision::VisualTableAsset>, String> {
    let mut stmt = tx
        .prepare(
            "SELECT id, page_no, asset_type, caption, bbox_json, image_path
             FROM document_visual_assets
             WHERE document_id = ?1 AND asset_type IN ('figure', 'chart', 'table', 'image')
             ORDER BY page_no, id",
        )
        .map_err(|err| format!("Failed to prepare visual crop asset read: {err}"))?;
    let rows = stmt
        .query_map(params![document_id], |row| {
            let bbox_json: String = row.get(4)?;
            Ok(vision::VisualTableAsset {
                id: row.get(0)?,
                page_no: row.get(1)?,
                asset_type: row.get(2)?,
                caption: row.get(3)?,
                bbox: serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([])),
                image_path: row.get(5)?,
            })
        })
        .map_err(|err| format!("Failed to read visual crop assets: {err}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect visual crop assets: {err}"))
}

fn update_visual_asset_crop_path(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
    asset_id: &str,
    crop_path: &str,
    ocr_text: &str,
) -> Result<(), String> {
    // image_path is set to the (preserved) crop; ocr_text is only overwritten
    // when we actually recognized something, so a re-run without OCR never
    // wipes previously stored text.
    tx.execute(
        "UPDATE document_visual_assets
         SET image_path = ?3,
             ocr_text = CASE WHEN ?4 != '' THEN ?4 ELSE ocr_text END,
             source = CASE WHEN source = '' THEN 'pdf_crop' ELSE source END,
             confidence = max(confidence, 0.72)
         WHERE id = ?1 AND document_id = ?2",
        params![asset_id, document_id, crop_path, ocr_text],
    )
    .map_err(|err| format!("Failed to update visual asset crop path: {err}"))?;
    Ok(())
}

pub(super) fn insert_tsr_table_result(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
    asset: &vision::VisualTableAsset,
    response: &vision::TableRecognitionResult,
) -> Result<(), String> {
    let source = if response.source.trim().is_empty() {
        "rust_tsr"
    } else {
        response.source.trim()
    };
    let table_id = format!(
        "table-{}-tsr-{}",
        document_id,
        stable_key_fragment(&asset.id)
    );
    tx.execute(
        "INSERT OR REPLACE INTO document_tables
            (id, document_id, page_no, caption, bbox_json, visual_asset_id, source, confidence, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, unixepoch())",
        params![
            table_id,
            document_id,
            asset.page_no,
            asset.caption,
            asset.bbox.to_string(),
            asset.id,
            source,
            normalized_confidence(response.confidence, 0.82),
        ],
    )
    .map_err(|err| format!("Failed to insert Rust TSR table: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_facts_fts WHERE table_id = ?1",
        params![table_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear Rust TSR table FTS rows: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_facts WHERE table_id = ?1",
        params![table_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear Rust TSR table facts: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_cells WHERE table_id = ?1",
        params![table_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear Rust TSR table cells: {err}"))?;

    let mut cells = response.cells.clone();
    cells.sort_by_key(|cell| (cell.row_index, cell.col_index));
    let mut header_by_col: HashMap<u32, String> = HashMap::new();
    let mut row_label_by_row: HashMap<u32, String> = HashMap::new();
    for cell in &cells {
        let text = cell.text.trim();
        if text.is_empty() {
            continue;
        }
        if cell.is_header || cell.row_index == 0 {
            header_by_col
                .entry(cell.col_index)
                .and_modify(|label| {
                    if !label.contains(text) {
                        *label = format!("{label} {text}");
                    }
                })
                .or_insert_with(|| text.to_string());
        } else if cell.col_index == 0 {
            row_label_by_row
                .entry(cell.row_index)
                .or_insert_with(|| text.to_string());
        }
    }

    for (ordinal, cell) in cells.iter().enumerate() {
        let text = cell.text.trim();
        if text.is_empty() {
            continue;
        }
        let bbox_json = if cell.bbox.is_null() {
            serde_json::json!([])
        } else {
            cell.bbox.clone()
        };
        let cell_id = format!(
            "{}-cell-r{}-c{}-{}",
            table_id,
            cell.row_index,
            cell.col_index,
            ordinal + 1
        );
        tx.execute(
            "INSERT INTO document_table_cells
                (id, table_id, row_index, col_index, row_span, col_span, text, bbox_json, is_header, confidence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                cell_id,
                table_id,
                cell.row_index,
                cell.col_index,
                cell.row_span.max(1),
                cell.col_span.max(1),
                text,
                bbox_json.to_string(),
                if cell.is_header { 1 } else { 0 },
                normalized_confidence(cell.confidence, response.confidence),
            ],
        )
        .map_err(|err| format!("Failed to insert Rust TSR table cell: {err}"))?;

        if cell.is_header || cell.row_index == 0 || cell.col_index == 0 {
            continue;
        }
        let row_label = row_label_by_row
            .get(&cell.row_index)
            .cloned()
            .unwrap_or_else(|| format!("Row {}", cell.row_index));
        let column_label = header_by_col
            .get(&cell.col_index)
            .cloned()
            .unwrap_or_else(|| format!("Column {}", cell.col_index));
        let fact_text = format!(
            "{} | {} | {} = {}",
            asset.caption, row_label, column_label, text
        );
        let fact_id = format!("{cell_id}-fact");
        let fact_bbox = if bbox_json.is_array()
            && bbox_json.as_array().is_some_and(|items| !items.is_empty())
        {
            bbox_json
        } else {
            asset.bbox.clone()
        };
        tx.execute(
            "INSERT INTO document_table_facts
                (id, document_id, table_id, page_no, row_label, column_label, value_text,
                 fact_text, bbox_json, source, confidence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                fact_id,
                document_id,
                table_id,
                asset.page_no,
                row_label,
                column_label,
                text,
                fact_text,
                fact_bbox.to_string(),
                source,
                normalized_confidence(cell.confidence, response.confidence),
            ],
        )
        .map_err(|err| format!("Failed to insert Rust TSR table fact: {err}"))?;
        tx.execute(
            "INSERT INTO document_table_facts_fts (fact_id, document_id, table_id, text)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                fact_id,
                document_id,
                table_id,
                crate::search_text::index_text(&fact_text)
            ],
        )
        .map_err(|err| format!("Failed to insert Rust TSR table fact FTS row: {err}"))?;
    }
    let worker_text = [response.markdown.as_str(), response.ocr_text.as_str()]
        .into_iter()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let worker_text = if worker_text.is_empty() {
        truncate_for_error(&response.html, 1600)
    } else {
        worker_text
    };
    if !response.crop_path.trim().is_empty() || !worker_text.trim().is_empty() {
        tx.execute(
            "UPDATE document_visual_assets
             SET image_path = CASE WHEN ?3 != '' THEN ?3 ELSE image_path END,
                 nearby_text = CASE WHEN ?4 != '' THEN ?4 ELSE nearby_text END,
                 source = ?5,
                 confidence = max(confidence, ?6)
             WHERE id = ?1 AND document_id = ?2",
            params![
                asset.id,
                document_id,
                response.crop_path.trim(),
                worker_text.trim(),
                source,
                normalized_confidence(response.confidence, 0.82),
            ],
        )
        .map_err(|err| format!("Failed to update worker visual asset metadata: {err}"))?;
    }
    Ok(())
}

fn normalized_confidence(value: f64, fallback: f64) -> f64 {
    let value = if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    };
    value.clamp(0.0, 1.0)
}

fn stable_key_fragment(value: &str) -> String {
    let mut result = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(32)
        .collect::<String>();
    if result.is_empty() {
        result = "asset".to_string();
    }
    result
}

pub(super) fn document_tree_ready(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<bool, String> {
    tx.query_row(
        "SELECT EXISTS(
           SELECT 1
           FROM structure_tree_nodes
           WHERE document_id = ?1
           LIMIT 1
         )",
        params![document_id],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value != 0)
    .map_err(|err| format!("Failed to inspect rebuilt structure tree: {err}"))
}

pub(super) struct VisualIndexJobState {
    pub(super) status: String,
    pub(super) version: i64,
    pub(super) error: String,
}

pub(super) fn queue_visual_index_job(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<(), String> {
    tx.execute(
        "INSERT INTO document_index_jobs
            (id, document_id, job_type, status, version, attempts, error, created_at, updated_at, started_at, finished_at)
         VALUES (?1, ?2, 'visual_tsr', 'pending', ?3, 0, '', unixepoch(), unixepoch(), 0, 0)
         ON CONFLICT(document_id, job_type) DO UPDATE SET
           status = 'pending',
           version = excluded.version,
           error = '',
           updated_at = unixepoch(),
           started_at = 0,
           finished_at = 0",
        params![
            format!("index-job-{document_id}-visual-tsr"),
            document_id,
            CURRENT_INDEX_VERSION,
        ],
    )
    .map_err(|err| format!("Failed to queue visual index job: {err}"))?;
    Ok(())
}

pub(super) fn visual_index_job_state(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<VisualIndexJobState, String> {
    tx.query_row(
        "SELECT status, version, error
         FROM document_index_jobs
         WHERE document_id = ?1 AND job_type = 'visual_tsr'",
        params![document_id],
        |row| {
            Ok(VisualIndexJobState {
                status: row.get(0)?,
                version: row.get(1)?,
                error: row.get(2)?,
            })
        },
    )
    .map_err(|err| format!("Failed to read visual index job: {err}"))
}

pub(super) fn mark_visual_index_job_running(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<(), String> {
    tx.execute(
        "INSERT INTO document_index_jobs
            (id, document_id, job_type, status, version, attempts, error, created_at, updated_at, started_at, finished_at)
         VALUES (?1, ?2, 'visual_tsr', 'running', ?3, 1, '', unixepoch(), unixepoch(), unixepoch(), 0)
         ON CONFLICT(document_id, job_type) DO UPDATE SET
           status = 'running',
           version = excluded.version,
           attempts = attempts + 1,
           error = '',
           updated_at = unixepoch(),
           started_at = unixepoch(),
           finished_at = 0",
        params![
            format!("index-job-{document_id}-visual-tsr"),
            document_id,
            CURRENT_INDEX_VERSION,
        ],
    )
    .map_err(|err| format!("Failed to mark visual index running: {err}"))?;
    Ok(())
}

pub(super) fn mark_visual_index_job_finished(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
    status: &str,
    error: &str,
) -> Result<(), String> {
    tx.execute(
        "UPDATE document_index_jobs
         SET status = ?2,
             version = ?3,
             error = ?4,
             updated_at = unixepoch(),
             finished_at = unixepoch()
         WHERE document_id = ?1 AND job_type = 'visual_tsr'",
        params![document_id, status, CURRENT_INDEX_VERSION, error],
    )
    .map_err(|err| format!("Failed to mark visual index finished: {err}"))?;
    Ok(())
}

fn visual_index_counts(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<(u32, u32, u32), String> {
    let visual_assets = count_document_rows(tx, "document_visual_assets", document_id)?;
    let tables = count_document_rows(tx, "document_tables", document_id)?;
    let table_facts = count_document_rows(tx, "document_table_facts", document_id)?;
    Ok((visual_assets, tables, table_facts))
}

fn count_document_rows(
    tx: &rusqlite::Transaction<'_>,
    table_name: &str,
    document_id: &str,
) -> Result<u32, String> {
    let sql = format!("SELECT COUNT(*) FROM {table_name} WHERE document_id = ?1");
    tx.query_row(&sql, params![document_id], |row| row.get::<_, u32>(0))
        .map_err(|err| format!("Failed to count {table_name}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::sync::Mutex;

    #[test]
    fn visual_index_job_moves_from_pending_to_succeeded() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE document_index_jobs (
                  id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  job_type TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  version INTEGER NOT NULL DEFAULT 0,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  error TEXT NOT NULL DEFAULT '',
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  started_at INTEGER NOT NULL DEFAULT 0,
                  finished_at INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(document_id, job_type)
                );",
        )
        .expect("job schema");
        let tx = conn.transaction().expect("transaction");

        queue_visual_index_job(&tx, "doc").expect("queue visual job");
        let pending = visual_index_job_state(&tx, "doc").expect("pending state");
        assert_eq!(pending.status, "pending");
        assert_eq!(pending.version, CURRENT_INDEX_VERSION);

        mark_visual_index_job_running(&tx, "doc").expect("mark running");
        mark_visual_index_job_finished(&tx, "doc", "succeeded", "").expect("mark succeeded");
        let succeeded = visual_index_job_state(&tx, "doc").expect("succeeded state");
        assert_eq!(succeeded.status, "succeeded");
        assert_eq!(succeeded.error, "");
    }

    #[test]
    fn visual_index_command_claims_pending_and_rejects_duplicate_queue() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE document_index_jobs (
                  id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  job_type TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  version INTEGER NOT NULL DEFAULT 0,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  error TEXT NOT NULL DEFAULT '',
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  started_at INTEGER NOT NULL DEFAULT 0,
                  finished_at INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(document_id, job_type)
                );
                CREATE TABLE documents (
                  id TEXT PRIMARY KEY
                );
                INSERT INTO documents (id) VALUES ('doc');",
        )
        .expect("job schema");
        {
            let tx = conn.transaction().expect("transaction");
            queue_visual_index_job(&tx, "doc").expect("pending visual job");
            tx.commit().expect("commit pending job");
        }
        let database = AppDatabase {
            conn: Mutex::new(conn),
        };

        assert!(queue_visual_index_command("doc", &database).expect("claim pending visual job"));
        assert!(!queue_visual_index_command("doc", &database).expect("reject duplicate queue"));

        let conn = database.conn.lock().expect("db lock");
        let job_status: String = conn
            .query_row(
                "SELECT status
                     FROM document_index_jobs
                     WHERE document_id = 'doc' AND job_type = 'visual_tsr'",
                [],
                |row| row.get(0),
            )
            .expect("visual job status");
        assert_eq!(job_status, "queued");
    }

    #[test]
    fn visual_index_command_rejects_missing_document() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (
                  id TEXT PRIMARY KEY
                );
                CREATE TABLE document_index_jobs (
                  id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  job_type TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  version INTEGER NOT NULL DEFAULT 0,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  error TEXT NOT NULL DEFAULT '',
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  started_at INTEGER NOT NULL DEFAULT 0,
                  finished_at INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(document_id, job_type)
                );",
        )
        .expect("job schema");
        let database = AppDatabase {
            conn: Mutex::new(conn),
        };

        let err = queue_visual_index_command("missing", &database)
            .expect_err("missing document should reject visual queue");

        assert!(err.contains("Document is no longer available"));
    }

    #[test]
    fn low_quality_tsr_result_is_not_usable() {
        let result = vision::TableRecognitionResult {
            cells: vec![
                vision::TableRecognitionCell {
                    row_index: 1,
                    col_index: 0,
                    row_span: 1,
                    col_span: 1,
                    text: "LLMLingua2".to_string(),
                    bbox: serde_json::json!([[0.13, 0.20, 0.25, 0.22]]),
                    is_header: false,
                    confidence: 0.08,
                },
                vision::TableRecognitionCell {
                    row_index: 1,
                    col_index: 5,
                    row_span: 1,
                    col_span: 1,
                    text: "sitter) on Long Code Completion".to_string(),
                    bbox: serde_json::json!([[0.23, 0.73, 0.54, 0.77]]),
                    is_header: false,
                    confidence: 0.08,
                },
            ],
            html: String::new(),
            markdown: String::new(),
            ocr_text: String::new(),
            crop_path: String::new(),
            confidence: 0.08,
            source: "slanet_onnx:test".to_string(),
        };

        assert!(!is_usable_tsr_result(&result));
    }

    #[test]
    fn rust_tsr_table_result_writes_cells_and_facts() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE document_visual_assets (
                  id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  page_no INTEGER NOT NULL,
                  asset_type TEXT NOT NULL,
                  caption TEXT NOT NULL DEFAULT '',
                  bbox_json TEXT NOT NULL DEFAULT '[]',
                  image_path TEXT NOT NULL DEFAULT '',
                  nearby_text TEXT NOT NULL DEFAULT '',
                  linked_block_ids_json TEXT NOT NULL DEFAULT '[]',
                  source TEXT NOT NULL DEFAULT 'pdf_bbox',
                  confidence REAL NOT NULL DEFAULT 0.0,
                  created_at INTEGER NOT NULL
                );
                CREATE TABLE document_tables (
                  id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  page_no INTEGER NOT NULL,
                  caption TEXT NOT NULL DEFAULT '',
                  bbox_json TEXT NOT NULL DEFAULT '[]',
                  visual_asset_id TEXT NOT NULL DEFAULT '',
                  source TEXT NOT NULL DEFAULT 'pdf_bbox',
                  confidence REAL NOT NULL DEFAULT 0.0,
                  created_at INTEGER NOT NULL
                );
                CREATE TABLE document_table_cells (
                  id TEXT PRIMARY KEY,
                  table_id TEXT NOT NULL,
                  row_index INTEGER NOT NULL,
                  col_index INTEGER NOT NULL,
                  row_span INTEGER NOT NULL DEFAULT 1,
                  col_span INTEGER NOT NULL DEFAULT 1,
                  text TEXT NOT NULL,
                  bbox_json TEXT NOT NULL DEFAULT '[]',
                  is_header INTEGER NOT NULL DEFAULT 0,
                  confidence REAL NOT NULL DEFAULT 0.0
                );
                CREATE TABLE document_table_facts (
                  id TEXT PRIMARY KEY,
                  document_id TEXT NOT NULL,
                  table_id TEXT NOT NULL,
                  page_no INTEGER NOT NULL,
                  row_label TEXT NOT NULL DEFAULT '',
                  column_label TEXT NOT NULL DEFAULT '',
                  value_text TEXT NOT NULL DEFAULT '',
                  fact_text TEXT NOT NULL,
                  bbox_json TEXT NOT NULL DEFAULT '[]',
                  source TEXT NOT NULL DEFAULT 'pdf_bbox',
                  confidence REAL NOT NULL DEFAULT 0.0
                );
                CREATE VIRTUAL TABLE document_table_facts_fts
                  USING fts5(fact_id UNINDEXED, document_id UNINDEXED, table_id UNINDEXED, text);
                INSERT INTO document_visual_assets
                  (id, document_id, page_no, asset_type, caption, bbox_json, created_at)
                VALUES
                  ('asset-1', 'doc', 3, 'table', 'Table 1: Scores',
                   '[[0.1,0.1,0.9,0.6]]', unixepoch());",
        )
        .expect("visual schema");
        let tx = conn.transaction().expect("transaction");
        let asset = vision::VisualTableAsset {
            id: "asset-1".to_string(),
            page_no: 3,
            asset_type: "table".to_string(),
            caption: "Table 1: Scores".to_string(),
            bbox: serde_json::json!([[0.1, 0.1, 0.9, 0.6]]),
            image_path: String::new(),
        };
        let response = vision::TableRecognitionResult {
            cells: vec![
                vision::TableRecognitionCell {
                    row_index: 0,
                    col_index: 0,
                    row_span: 1,
                    col_span: 1,
                    text: "Benchmark".to_string(),
                    bbox: serde_json::json!([]),
                    is_header: true,
                    confidence: 0.9,
                },
                vision::TableRecognitionCell {
                    row_index: 0,
                    col_index: 1,
                    row_span: 1,
                    col_span: 1,
                    text: "GLM-5".to_string(),
                    bbox: serde_json::json!([]),
                    is_header: true,
                    confidence: 0.9,
                },
                vision::TableRecognitionCell {
                    row_index: 1,
                    col_index: 0,
                    row_span: 1,
                    col_span: 1,
                    text: "SWE-bench Verified".to_string(),
                    bbox: serde_json::json!([]),
                    is_header: false,
                    confidence: 0.9,
                },
                vision::TableRecognitionCell {
                    row_index: 1,
                    col_index: 1,
                    row_span: 1,
                    col_span: 1,
                    text: "77.8".to_string(),
                    bbox: serde_json::json!([]),
                    is_header: false,
                    confidence: 0.9,
                },
            ],
            html: "<table></table>".to_string(),
            markdown: "| Benchmark | GLM-5 |".to_string(),
            ocr_text: String::new(),
            crop_path: "/tmp/table.png".to_string(),
            confidence: 0.9,
            source: "rust_tsr".to_string(),
        };
        insert_tsr_table_result(&tx, "doc", &asset, &response).expect("Rust TSR table insert");
        tx.commit().expect("commit");

        let fact = conn
            .query_row(
                "SELECT fact_text
                     FROM document_table_facts
                     WHERE document_id = 'doc' AND source = 'rust_tsr'
                     LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("Rust TSR table fact");
        assert!(fact.contains("SWE-bench Verified"));
        assert!(fact.contains("GLM-5 = 77.8"));
    }
}
