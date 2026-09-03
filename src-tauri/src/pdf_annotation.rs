use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{ipc::Response, State};

#[cfg(unix)]
use std::fs::File;

use crate::{AppDatabase, PdfRegistry};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SaveDocumentPdfInput {
    pub(crate) document_id: String,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SaveDocumentPdfAsInput {
    pub(crate) document_id: String,
    pub(crate) default_name: String,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SavePdfAsInput {
    pub(crate) default_name: String,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SavePdfAtPathInput {
    pub(crate) path: String,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PdfSaveOutput {
    pub(crate) path: String,
    pub(crate) size: u64,
}

#[tauri::command]
pub(crate) fn save_pdf_document(
    input: SaveDocumentPdfInput,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<PdfSaveOutput, String> {
    let document_id = require_document_id(&input.document_id)?;
    let path = registered_document_path(&registry, &document_id)?;
    let output = write_pdf_to_existing_path(&path, &input.bytes)?;
    update_document_metadata(&database, &document_id, &path, output.size)?;
    Ok(output)
}

#[tauri::command]
pub(crate) fn save_pdf_document_as(
    input: SaveDocumentPdfAsInput,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<Option<PdfSaveOutput>, String> {
    let document_id = require_document_id(&input.document_id)?;
    let Some(path) = choose_pdf_save_path(&input.default_name)? else {
        return Ok(None);
    };

    ensure_document_path_available(&database, &document_id, &path)?;
    let output = write_pdf_to_new_or_existing_path(&path, &input.bytes)?;
    update_document_metadata(&database, &document_id, &path, output.size)?;
    update_registered_document_path(&registry, &document_id, &path)?;
    Ok(Some(output))
}

/// Save a generated translation artifact as a normal user-chosen PDF. The
/// artifact itself is managed cache data, so this command intentionally never
/// writes back into the pdf2zh artifact tree.
#[tauri::command]
pub(crate) fn save_pdf_as(input: SavePdfAsInput) -> Result<Option<PdfSaveOutput>, String> {
    let Some(path) = choose_pdf_save_path(&input.default_name)? else {
        return Ok(None);
    };
    write_pdf_to_new_or_existing_path(&path, &input.bytes).map(Some)
}

/// Direct saves after a translation artifact has been saved as a regular PDF.
/// The target must already exist, which prevents this command from becoming a
/// general arbitrary-file creation API; new paths always go through the native
/// Save As dialog above.
#[tauri::command]
pub(crate) fn save_pdf_at_path(input: SavePdfAtPathInput) -> Result<PdfSaveOutput, String> {
    let path = canonical_existing_pdf_path(&input.path)?;
    write_pdf_to_existing_path(&path, &input.bytes)
}

/// Read a PDF the user previously selected through the native Save As flow.
/// This is intentionally separate from `read_pdf_artifact_bytes`, whose path
/// guard only admits pdf2zh cache artifacts.
#[tauri::command]
pub(crate) async fn read_saved_pdf_bytes(path: String) -> Result<Response, String> {
    crate::run_blocking_io(move || {
        let path = canonical_existing_pdf_path(&path)?;
        let bytes =
            fs::read(&path).map_err(|err| crate::map_file_read_error("saved PDF", &path, err))?;
        Ok(Response::new(bytes))
    })
    .await
}

fn require_document_id(raw: &str) -> Result<String, String> {
    let document_id = raw.trim();
    if document_id.is_empty() {
        return Err("No PDF document selected".to_string());
    }
    Ok(document_id.to_string())
}

fn registered_document_path(
    registry: &State<'_, PdfRegistry>,
    document_id: &str,
) -> Result<PathBuf, String> {
    let paths = registry
        .paths
        .lock()
        .map_err(|_| "PDF registry lock was poisoned".to_string())?;
    let path = paths
        .get(document_id)
        .cloned()
        .ok_or_else(|| "Unknown PDF document id".to_string())?;
    canonical_existing_pdf_path(&path.to_string_lossy())
}

fn update_registered_document_path(
    registry: &State<'_, PdfRegistry>,
    document_id: &str,
    path: &Path,
) -> Result<(), String> {
    let mut paths = registry
        .paths
        .lock()
        .map_err(|_| "PDF registry lock was poisoned".to_string())?;
    paths.insert(document_id.to_string(), path.to_path_buf());
    Ok(())
}

fn update_document_metadata(
    database: &State<'_, AppDatabase>,
    document_id: &str,
    path: &Path,
    size: u64,
) -> Result<(), String> {
    let modified = file_modified_epoch(path)?;
    let path = path.to_string_lossy().to_string();
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let updated = conn
        .execute(
            "UPDATE documents
             SET path = ?2, file_size = ?3, modified = ?4, updated_at = unixepoch()
             WHERE id = ?1",
            rusqlite::params![document_id, path, size, modified],
        )
        .map_err(|err| format!("Failed to update saved PDF metadata: {err}"))?;
    if updated == 0 {
        return Err("PDF document no longer exists".to_string());
    }
    Ok(())
}

fn ensure_document_path_available(
    database: &State<'_, AppDatabase>,
    document_id: &str,
    path: &Path,
) -> Result<(), String> {
    let path = path.to_string_lossy().to_string();
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let existing: Option<String> = conn
        .query_row(
            "SELECT id FROM documents WHERE path = ?1",
            rusqlite::params![path],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| format!("Failed to validate Save As path: {err}"))?;
    if existing.as_deref().is_some_and(|id| id != document_id) {
        return Err("That PDF is already open as another document".to_string());
    }
    Ok(())
}

fn choose_pdf_save_path(default_name: &str) -> Result<Option<PathBuf>, String> {
    let default_name = normalized_pdf_file_name(default_name);
    let Some(path) = rfd::FileDialog::new()
        .set_title("Save annotated PDF")
        .set_file_name(&default_name)
        .add_filter("PDF", &["pdf"])
        .save_file()
    else {
        return Ok(None);
    };
    prepare_new_pdf_path(&path).map(Some)
}

fn normalized_pdf_file_name(raw: &str) -> String {
    let candidate = Path::new(raw.trim())
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or("annotated.pdf")
        .trim();
    if candidate.to_ascii_lowercase().ends_with(".pdf") {
        candidate.to_string()
    } else {
        format!("{candidate}.pdf")
    }
}

fn canonical_existing_pdf_path(raw: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(raw.trim());
    if raw.trim().is_empty() {
        return Err("No PDF path is available".to_string());
    }
    let path = path
        .canonicalize()
        .map_err(|err| format!("Failed to resolve PDF path: {err}"))?;
    if !path.is_file() {
        return Err("PDF save target is not a file".to_string());
    }
    ensure_pdf_extension(&path)?;
    Ok(path)
}

fn prepare_new_pdf_path(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| "PDF save target has no parent folder".to_string())?
        .canonicalize()
        .map_err(|err| format!("Failed to resolve PDF save folder: {err}"))?;
    if !parent.is_dir() {
        return Err("PDF save target parent is not a folder".to_string());
    }
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "PDF save target has no file name".to_string())?;
    let target = parent.join(file_name);
    ensure_pdf_extension(&target)?;
    Ok(target)
}

fn ensure_pdf_extension(path: &Path) -> Result<(), String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
    {
        Ok(())
    } else {
        Err("PDF save target must use a .pdf extension".to_string())
    }
}

fn write_pdf_to_existing_path(path: &Path, bytes: &[u8]) -> Result<PdfSaveOutput, String> {
    let path = canonical_existing_pdf_path(&path.to_string_lossy())?;
    write_pdf_atomically(&path, bytes)
}

fn write_pdf_to_new_or_existing_path(path: &Path, bytes: &[u8]) -> Result<PdfSaveOutput, String> {
    let path = prepare_new_pdf_path(path)?;
    write_pdf_atomically(&path, bytes)
}

fn write_pdf_atomically(path: &Path, bytes: &[u8]) -> Result<PdfSaveOutput, String> {
    validate_pdf_bytes(bytes)?;
    let parent = path
        .parent()
        .ok_or_else(|| "PDF save target has no parent folder".to_string())?;
    let temp = adjacent_temporary_path(path, "tmp");
    let backup = adjacent_temporary_path(path, "previous");

    write_and_sync(&temp, bytes)?;
    let had_original = path.exists();
    if had_original {
        fs::rename(path, &backup).map_err(|err| {
            let _ = fs::remove_file(&temp);
            format!("Failed to stage existing PDF for replacement: {err}")
        })?;
    }

    if let Err(err) = fs::rename(&temp, path) {
        if had_original {
            let _ = fs::rename(&backup, path);
        }
        let _ = fs::remove_file(&temp);
        return Err(format!(
            "Failed to replace PDF with annotated version: {err}"
        ));
    }

    if had_original {
        let _ = fs::remove_file(&backup);
    }
    sync_directory(parent);

    Ok(PdfSaveOutput {
        path: path.to_string_lossy().to_string(),
        size: bytes.len() as u64,
    })
}

fn validate_pdf_bytes(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < 8 || !bytes.starts_with(b"%PDF-") {
        return Err("Annotated output is not a valid PDF byte stream".to_string());
    }
    Ok(())
}

fn write_and_sync(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("Failed to create temporary PDF save file: {err}"))?;
    file.write_all(bytes)
        .map_err(|err| format!("Failed to write annotated PDF: {err}"))?;
    file.sync_all()
        .map_err(|err| format!("Failed to flush annotated PDF: {err}"))?;
    Ok(())
}

fn adjacent_temporary_path(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("document.pdf");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{file_name}.lumenfolio-{suffix}-{}-{nonce}",
        std::process::id()
    ))
}

fn file_modified_epoch(path: &Path) -> Result<i64, String> {
    let modified = fs::metadata(path)
        .map_err(|err| format!("Failed to read saved PDF metadata: {err}"))?
        .modified()
        .map_err(|err| format!("Failed to read saved PDF modification time: {err}"))?
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("Saved PDF modification time is before the Unix epoch: {err}"))?
        .as_secs();
    Ok(i64::try_from(modified).unwrap_or(i64::MAX))
}

#[cfg(unix)]
fn sync_directory(path: &Path) {
    if let Ok(file) = File::open(path) {
        let _ = file.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf-save-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn normalizes_pdf_save_names_without_paths() {
        assert_eq!(normalized_pdf_file_name("paper"), "paper.pdf");
        assert_eq!(normalized_pdf_file_name("paper.PDF"), "paper.PDF");
        assert_eq!(normalized_pdf_file_name("nested/paper"), "paper.pdf");
        assert_eq!(normalized_pdf_file_name(""), "annotated.pdf");
    }

    #[test]
    fn rejects_non_pdf_output_bytes() {
        assert!(validate_pdf_bytes(b"not a pdf").is_err());
        assert!(validate_pdf_bytes(b"%PDF-1.7\n").is_ok());
    }

    #[test]
    fn atomically_replaces_an_existing_pdf() {
        let dir = temp_dir("replace");
        let target = dir.join("paper.pdf");
        fs::write(&target, b"%PDF-1.4\nold").expect("seed");

        let output = write_pdf_atomically(&target, b"%PDF-1.7\nnew").expect("save");
        assert_eq!(output.size, 12);
        assert_eq!(fs::read(&target).expect("read"), b"%PDF-1.7\nnew");
        assert!(fs::read_dir(&dir).expect("read dir").all(|entry| !entry
            .expect("entry")
            .file_name()
            .to_string_lossy()
            .contains("lumenfolio-")));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn prepares_new_target_in_existing_folder() {
        let dir = temp_dir("target");
        let target = prepare_new_pdf_path(&dir.join("new.pdf")).expect("prepare");
        assert_eq!(target, dir.join("new.pdf"));
        assert!(prepare_new_pdf_path(&dir.join("new.txt")).is_err());
        let _ = fs::remove_dir_all(dir);
    }
}
