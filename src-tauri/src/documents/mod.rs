use std::{
    collections::HashSet,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use rusqlite::{params, Connection, OptionalExtension};
use tauri::State;

use crate::{AppDatabase, PdfDocument, PdfRegistry, CURRENT_INDEX_VERSION};

pub(crate) fn replace_registry_paths(
    registry: &State<'_, PdfRegistry>,
    documents: &[PdfDocument],
) -> Result<(), String> {
    let mut paths = registry
        .paths
        .lock()
        .map_err(|_| "PDF registry lock was poisoned".to_string())?;
    paths.clear();
    for doc in documents {
        paths.insert(doc.id.clone(), PathBuf::from(&doc.path));
    }
    Ok(())
}

pub(crate) fn upsert_registry_paths(
    registry: &State<'_, PdfRegistry>,
    documents: &[PdfDocument],
) -> Result<(), String> {
    let mut paths = registry
        .paths
        .lock()
        .map_err(|_| "PDF registry lock was poisoned".to_string())?;
    for doc in documents {
        paths.insert(doc.id.clone(), PathBuf::from(&doc.path));
    }
    Ok(())
}

pub(crate) fn remove_registry_paths(
    registry: &State<'_, PdfRegistry>,
    documents: &[PdfDocument],
) -> Result<(), String> {
    let mut paths = registry
        .paths
        .lock()
        .map_err(|_| "PDF registry lock was poisoned".to_string())?;
    for doc in documents {
        paths.remove(&doc.id);
    }
    Ok(())
}

pub(crate) fn load_documents_for_root(
    database: &State<'_, AppDatabase>,
    workspace_root_id: &str,
) -> Result<Vec<PdfDocument>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    load_documents_for_root_conn(&conn, workspace_root_id)
}

pub(crate) fn load_documents_for_root_conn(
    conn: &Connection,
    workspace_root_id: &str,
) -> Result<Vec<PdfDocument>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT documents.id, documents.workspace_root_id, documents.title,
                    documents.short_title, documents.path, documents.file_size,
                    documents.modified, documents.page_count, documents.last_page,
                    documents.index_version,
                    CASE
                      WHEN documents.index_status != 'indexed' THEN documents.index_status
                      WHEN documents.index_version = ?2 THEN documents.index_status
                      ELSE 'stale'
                    END AS index_status,
                    documents.index_version = ?2 AND EXISTS(
                      SELECT 1
                      FROM structure_tree_nodes stn
                      WHERE stn.document_id = documents.id
                      LIMIT 1
                    ) AS tree_ready,
                    COALESCE(vij.status, 'pending') AS visual_index_status,
                    COALESCE(vij.version, 0) AS visual_index_version,
                    COALESCE(vij.error, '') AS visual_index_error,
                    documents.content_type,
                    documents.collection_id,
                    documents.position
	             FROM documents
	             LEFT JOIN document_index_jobs vij
	               ON vij.document_id = documents.id
	              AND vij.job_type = 'visual_tsr'
	             WHERE documents.workspace_root_id = ?1
	             ORDER BY documents.position ASC, lower(documents.short_title) ASC",
        )
        .map_err(|err| format!("Failed to load documents: {err}"))?;

    let documents = stmt
        .query_map(params![workspace_root_id, CURRENT_INDEX_VERSION], |row| {
            Ok(PdfDocument {
                id: row.get(0)?,
                workspace_root_id: row.get(1)?,
                title: row.get(2)?,
                short_title: row.get(3)?,
                path: row.get(4)?,
                size: row.get(5)?,
                modified: row.get(6)?,
                page_count: row.get(7)?,
                current_page: row.get(8)?,
                index_version: row.get(9)?,
                current_index_version: CURRENT_INDEX_VERSION,
                index_status: row.get(10)?,
                tree_ready: row.get::<_, i64>(11)? != 0,
                visual_index_status: row.get(12)?,
                visual_index_version: row.get(13)?,
                visual_index_error: row.get(14)?,
                content_type: row.get(15)?,
                collection_id: row.get(16)?,
                position: row.get(17)?,
            })
        })
        .map_err(|err| format!("Failed to load documents: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to load documents: {err}"))?;

    Ok(documents)
}

pub(crate) fn persist_workspace_scan(
    database: &State<'_, AppDatabase>,
    workspace_root_id: &str,
    root_path: &Path,
    docs: &[PdfDocument],
) -> Result<(), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start workspace transaction: {err}"))?;
    tx.execute(
        "INSERT INTO workspace_roots (id, path, name, created_at, updated_at, last_opened_at)
         VALUES (?1, ?2, ?3, unixepoch(), unixepoch(), unixepoch())
         ON CONFLICT(id) DO UPDATE SET
           path = excluded.path,
           name = excluded.name,
           updated_at = unixepoch(),
           last_opened_at = unixepoch()",
        params![
            workspace_root_id,
            root_path.to_string_lossy(),
            root_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Workspace")
        ],
    )
    .map_err(|err| format!("Failed to save workspace root: {err}"))?;

    let current_doc_ids = docs
        .iter()
        .map(|doc| doc.id.as_str())
        .collect::<HashSet<_>>();
    let existing_doc_ids = load_existing_document_ids(&tx, workspace_root_id)?;

    for stale_doc_id in existing_doc_ids
        .iter()
        .filter(|doc_id| !current_doc_ids.contains(doc_id.as_str()))
    {
        tx.execute(
            "DELETE FROM document_chunks_fts WHERE document_id = ?1",
            params![stale_doc_id],
        )
        .map_err(|err| format!("Failed to delete stale FTS rows: {err}"))?;
        tx.execute("DELETE FROM documents WHERE id = ?1", params![stale_doc_id])
            .map_err(|err| format!("Failed to delete stale document: {err}"))?;
    }

    for doc in docs {
        tx.execute(
            "INSERT INTO documents
                (id, workspace_root_id, path, title, short_title, file_size, modified,
                 page_count, last_page, content_type, index_status, index_version, created_at,
                 updated_at, last_opened_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?12, ?10, 0, unixepoch(),
                     unixepoch(), unixepoch())
             ON CONFLICT(id) DO UPDATE SET
               -- Preserve the source kind: without this an imported docx/xlsx/pptx
               -- fell back to the column default 'pdf', so it opened in the PDF
               -- viewer (Invalid PDF structure) and PDF-path indexing failed.
               content_type = excluded.content_type,
               -- Nested workspace folders share files (e.g. ~/Documents contains
               -- ~/Documents/.../test_pdf). The MORE SPECIFIC (deeper) folder
               -- owns the file: only reassign to the scanning root when its path
               -- is deeper (longer) than the doc's current root, so a shallower
               -- folder added later can't steal docs from a deeper one.
               workspace_root_id = CASE
                 WHEN length(COALESCE((SELECT path FROM workspace_roots WHERE id = documents.workspace_root_id), ''))
                      < length(COALESCE((SELECT path FROM workspace_roots WHERE id = excluded.workspace_root_id), ''))
                   THEN excluded.workspace_root_id
                 ELSE documents.workspace_root_id
               END,
               path = excluded.path,
               title = excluded.title,
               short_title = excluded.short_title,
               file_size = excluded.file_size,
               modified = excluded.modified,
               updated_at = unixepoch(),
               index_status = CASE
                 WHEN documents.modified != excluded.modified
                   OR documents.file_size != excluded.file_size
                   OR (
                     documents.index_status = 'indexed'
                     AND documents.index_version != ?11
                   )
                 THEN 'stale'
                 ELSE documents.index_status
               END",
            params![
                doc.id,
                doc.workspace_root_id,
                doc.path,
                doc.title,
                doc.short_title,
                doc.size,
                doc.modified,
                doc.page_count,
                doc.current_page,
                doc.index_status,
                CURRENT_INDEX_VERSION,
                doc.content_type,
            ],
        )
        .map_err(|err| format!("Failed to save document: {err}"))?;
    }

    tx.commit()
        .map_err(|err| format!("Failed to commit workspace scan: {err}"))
}

/// Collect the PDFs DIRECTLY inside `dir` (one level only — subdirectories are
/// skipped, see the `is_dir` branch). Each scanned folder maps to exactly the
/// files it contains, so the sidebar never flattens nested folders together.
pub(crate) fn collect_pdfs(
    dir: &Path,
    workspace_root_id: &str,
    docs: &mut Vec<PdfDocument>,
) -> Result<(), String> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            log::warn!("Skipping unreadable directory {}: {err}", dir.display());
            return Ok(());
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                log::warn!(
                    "Skipping unreadable directory entry in {}: {err}",
                    dir.display()
                );
                continue;
            }
        };
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();

        if should_ignore_name(&name) {
            continue;
        }

        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(err) => {
                log::warn!(
                    "Skipping entry with unreadable type {}: {err}",
                    path.display()
                );
                continue;
            }
        };

        if file_type.is_symlink() {
            continue;
        }

        if file_type.is_dir() {
            // One level only: subdirectory PDFs are intentionally NOT collected.
            // A scanned folder shows just the PDFs directly inside it; to import a
            // subfolder, add it as its own folder. (Recursing here flattened every
            // nested PDF under the root with no indication of its origin folder.)
            continue;
        }

        if !file_type.is_file() {
            continue;
        }

        // Knowledge-base pivot: a scanned folder is no longer PDF-only. Collect any
        // file-backed source build_document_for_path understands (PDF + Office); it
        // returns None for everything else, so the filter and the builder share one
        // source of truth for "supported extension".
        let supported = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .is_some_and(|extension| {
                extension == "pdf"
                    || crate::office::office_content_type_for_ext(&extension).is_some()
            });
        if !supported {
            continue;
        }

        if let Some(doc) = build_document_for_path(&path, workspace_root_id)? {
            docs.push(doc);
        }
    }

    Ok(())
}

/// Whether a file extension is an importable source (file-backed or text).
pub(crate) fn is_importable_ext(ext: &str) -> bool {
    matches!(ext, "pdf" | "md" | "markdown" | "txt" | "text")
        || crate::office::office_content_type_for_ext(ext).is_some()
}

/// Knowledge-base pivot (Collections): recursively collect importable file paths
/// under a directory (one-shot — no live scan binding). Depth-capped to avoid
/// pathological trees; ignores dot/build dirs and symlinks.
pub(crate) fn collect_import_files(
    dir: &Path,
    depth: usize,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    if depth > 12 {
        return Ok(());
    }
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            log::warn!("Skipping unreadable directory {}: {err}", dir.display());
            return Ok(());
        }
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if should_ignore_name(&name) {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            collect_import_files(&path, depth + 1, out)?;
        } else if file_type.is_file() {
            let importable = path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| extension.to_ascii_lowercase())
                .is_some_and(|extension| is_importable_ext(&extension));
            if importable {
                out.push(path);
            }
        }
    }
    Ok(())
}

/// File the given documents into a collection. Only an explicit target moves a
/// source: a `None` target is a no-op — re-importing an already-filed source
/// must NOT unfile it, and brand-new sources already default to unfiled. A
/// stale/deleted target is ignored rather than orphaning docs into a ghost id.
pub(crate) fn assign_collection(
    database: &State<'_, AppDatabase>,
    document_ids: &[String],
    collection_id: Option<&str>,
) -> Result<(), String> {
    let Some(target) = collection_id else {
        return Ok(());
    };
    if document_ids.is_empty() {
        return Ok(());
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let exists: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM collections WHERE id = ?1",
            params![target],
            |row| row.get(0),
        )
        .map_err(|err| format!("Failed to check target collection: {err}"))?;
    if exists == 0 {
        return Ok(());
    }
    let base = next_document_position(&conn, Some(target))?;
    let mut stmt = conn
        .prepare(
            "UPDATE documents SET collection_id = ?2, position = ?3, updated_at = unixepoch()
             WHERE id = ?1",
        )
        .map_err(|err| format!("Failed to prepare collection assignment: {err}"))?;
    for (offset, id) in document_ids.iter().enumerate() {
        stmt.execute(params![id, target, base + offset as i64])
            .map_err(|err| format!("Failed to assign collection: {err}"))?;
    }
    Ok(())
}

pub(crate) fn build_document_for_path(
    path: &Path,
    workspace_root_id: &str,
) -> Result<Option<PdfDocument>, String> {
    let canonical_path = match path.canonicalize() {
        Ok(canonical_path) => canonical_path,
        Err(err) => {
            log::warn!("Skipping unresolved PDF {}: {err}", path.display());
            return Ok(None);
        }
    };
    let metadata = match fs::metadata(&canonical_path) {
        Ok(metadata) => metadata,
        Err(err) => {
            log::warn!(
                "Skipping PDF with unreadable metadata {}: {err}",
                canonical_path.display()
            );
            return Ok(None);
        }
    };
    if !metadata.is_file() {
        return Ok(None);
    }
    // Knowledge-base pivot (P3): file-backed sources are PDFs and Office formats.
    // Office files index from disk (not body_md) and are previewed client-side.
    let ext = canonical_path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let content_type = if ext == "pdf" {
        "pdf"
    } else if let Some(office) = crate::office::office_content_type_for_ext(&ext) {
        office
    } else {
        return Ok(None);
    };
    let title = canonical_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or("Untitled")
        .to_string();

    Ok(Some(PdfDocument {
        id: stable_path_id(content_type, &canonical_path),
        workspace_root_id: workspace_root_id.to_string(),
        short_title: title.clone(),
        title,
        content_type: content_type.to_string(),
        collection_id: None,
        position: 0,
        path: canonical_path.to_string_lossy().to_string(),
        size: metadata.len(),
        modified: metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_secs()),
        page_count: 0,
        current_page: 1,
        index_status: "pending".to_string(),
        index_version: 0,
        current_index_version: CURRENT_INDEX_VERSION,
        tree_ready: false,
        visual_index_status: "pending".to_string(),
        visual_index_version: 0,
        visual_index_error: String::new(),
    }))
}

pub(crate) fn additive_upsert_documents(
    database: &State<'_, AppDatabase>,
    workspace_root_id: &str,
    root_path: &Path,
    docs: &[PdfDocument],
) -> Result<(), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start workspace transaction: {err}"))?;
    tx.execute(
        "INSERT INTO workspace_roots (id, path, name, created_at, updated_at, last_opened_at)
         VALUES (?1, ?2, ?3, unixepoch(), unixepoch(), unixepoch())
         ON CONFLICT(id) DO UPDATE SET
           path = excluded.path,
           name = excluded.name,
           updated_at = unixepoch(),
           last_opened_at = unixepoch()",
        params![
            workspace_root_id,
            root_path.to_string_lossy(),
            root_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Workspace")
        ],
    )
    .map_err(|err| format!("Failed to save workspace root: {err}"))?;

    for doc in docs {
        tx.execute(
            "INSERT INTO documents
                (id, workspace_root_id, path, title, short_title, file_size, modified,
                 page_count, last_page, content_type, index_status, index_version, created_at,
                 updated_at, last_opened_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?12, ?10, 0, unixepoch(),
                     unixepoch(), unixepoch())
             ON CONFLICT(id) DO UPDATE SET
               -- Preserve the source kind: without this an imported docx/xlsx/pptx
               -- fell back to the column default 'pdf', so it opened in the PDF
               -- viewer (Invalid PDF structure) and PDF-path indexing failed.
               content_type = excluded.content_type,
               -- Nested workspace folders share files (e.g. ~/Documents contains
               -- ~/Documents/.../test_pdf). The MORE SPECIFIC (deeper) folder
               -- owns the file: only reassign to the scanning root when its path
               -- is deeper (longer) than the doc's current root, so a shallower
               -- folder added later can't steal docs from a deeper one.
               workspace_root_id = CASE
                 WHEN length(COALESCE((SELECT path FROM workspace_roots WHERE id = documents.workspace_root_id), ''))
                      < length(COALESCE((SELECT path FROM workspace_roots WHERE id = excluded.workspace_root_id), ''))
                   THEN excluded.workspace_root_id
                 ELSE documents.workspace_root_id
               END,
               path = excluded.path,
               title = excluded.title,
               short_title = excluded.short_title,
               file_size = excluded.file_size,
               modified = excluded.modified,
               updated_at = unixepoch(),
               index_status = CASE
                 WHEN documents.modified != excluded.modified
                   OR documents.file_size != excluded.file_size
                   OR (
                     documents.index_status = 'indexed'
                     AND documents.index_version != ?11
                   )
                 THEN 'stale'
                 ELSE documents.index_status
               END",
            params![
                doc.id,
                doc.workspace_root_id,
                doc.path,
                doc.title,
                doc.short_title,
                doc.size,
                doc.modified,
                doc.page_count,
                doc.current_page,
                doc.index_status,
                CURRENT_INDEX_VERSION,
                doc.content_type,
            ],
        )
        .map_err(|err| format!("Failed to save document: {err}"))?;
    }

    tx.commit()
        .map_err(|err| format!("Failed to commit workspace additive upsert: {err}"))
}

pub(crate) fn lookup_workspace_root_path(
    database: &State<'_, AppDatabase>,
    workspace_root_id: &str,
) -> Result<Option<PathBuf>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let path: Option<String> = conn
        .query_row(
            "SELECT path FROM workspace_roots WHERE id = ?1",
            params![workspace_root_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| format!("Failed to look up workspace root: {err}"))?;
    Ok(path.map(PathBuf::from))
}

pub(crate) fn unique_destination_path(root_path: &Path, file_name: &OsStr) -> PathBuf {
    let initial = root_path.join(file_name);
    if !initial.exists() {
        return initial;
    }
    let name_string = file_name.to_string_lossy().into_owned();
    let (stem, extension) = match name_string.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => (stem.to_string(), Some(ext.to_string())),
        _ => (name_string.clone(), None),
    };
    for counter in 1u32..10_000 {
        let candidate_name = match &extension {
            Some(ext) => format!("{stem} ({counter}).{ext}"),
            None => format!("{stem} ({counter})"),
        };
        let candidate = root_path.join(&candidate_name);
        if !candidate.exists() {
            return candidate;
        }
    }
    root_path.join(file_name)
}

pub(crate) fn stable_path_id(prefix: &str, path: &Path) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in path.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{prefix}-{hash:016x}")
}

// ---------------------------------------------------------------------------
// Knowledge-base pivot (P2): disk-decoupled authored sources.
//
// Notes, web clips and imported markdown live in a single virtual "Knowledge
// Base" workspace root rather than a scanned directory. Their authored body is
// stored in `documents.body_md`; saving re-indexes straight from the DB (no
// file on disk). The synthetic `path` (`<kind>:<id>`) only satisfies the
// NOT NULL / UNIQUE constraint — it is never read from the filesystem.
// ---------------------------------------------------------------------------

/// Stable id of the virtual Knowledge Base root that owns authored sources.
pub(crate) const KNOWLEDGE_ROOT_ID: &str = "root-knowledge-base";
pub(crate) const KNOWLEDGE_ROOT_PATH: &str = "lumenfolio://knowledge";
const KNOWLEDGE_ROOT_NAME: &str = "Knowledge Base";

/// Content types whose body is editable in the Markdown editor.
pub(crate) const EDITABLE_CONTENT_TYPES: &[&str] = &["note", "markdown", "text", "web"];

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TextDocumentBody {
    pub document_id: String,
    pub title: String,
    pub body_md: String,
    pub source_url: String,
    pub content_type: String,
}

fn ensure_knowledge_root(conn: &Connection) -> Result<(), String> {
    conn.execute(
        "INSERT INTO workspace_roots (id, path, name, created_at, updated_at, last_opened_at)
         VALUES (?1, ?2, ?3, unixepoch(), unixepoch(), unixepoch())
         ON CONFLICT(id) DO NOTHING",
        params![KNOWLEDGE_ROOT_ID, KNOWLEDGE_ROOT_PATH, KNOWLEDGE_ROOT_NAME],
    )
    .map_err(|err| format!("Failed to ensure knowledge root: {err}"))?;
    Ok(())
}

fn new_text_document_id(content_type: &str) -> String {
    format!(
        "{content_type}-{:016x}{:016x}",
        rand::random::<u64>(),
        rand::random::<u64>()
    )
}

/// Create a disk-decoupled authored source (note / web clip / imported md/txt)
/// in the Knowledge Base root. Returns the new document id. The row starts
/// `index_status = 'pending'`; the caller enqueues a reindex to chunk the body.
/// Next manual-order slot for a new/moved document inside `collection_id`
/// (NULL = the root). Mirrors the collections.position scheme so a freshly
/// created or filed source appends to the bottom of its siblings rather than
/// jumping into the middle by title. `IS ?1` matches NULL correctly.
pub(crate) fn next_document_position(
    conn: &Connection,
    collection_id: Option<&str>,
) -> Result<i64, String> {
    conn.query_row(
        "SELECT COALESCE(MAX(position), -1) + 1 FROM documents WHERE collection_id IS ?1",
        params![collection_id],
        |row| row.get(0),
    )
    .map_err(|err| format!("Failed to compute document position: {err}"))
}

pub(crate) fn create_text_document(
    database: &State<'_, AppDatabase>,
    content_type: &str,
    title: &str,
    body_md: &str,
    source_url: Option<&str>,
    collection_id: Option<&str>,
) -> Result<String, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    ensure_knowledge_root(&conn)?;
    let id = new_text_document_id(content_type);
    let path = format!("{content_type}:{id}");
    let title = {
        let trimmed = title.trim();
        if trimmed.is_empty() {
            "Untitled".to_string()
        } else {
            trimmed.to_string()
        }
    };
    let position = next_document_position(&conn, collection_id)?;
    conn.execute(
        "INSERT INTO documents
            (id, workspace_root_id, path, title, short_title, file_size, modified,
             page_count, last_page, content_type, body_md, source_url, collection_id,
             position, index_status, index_version, created_at, updated_at, last_opened_at)
         VALUES (?1, ?2, ?3, ?4, ?4, ?5, unixepoch(), 0, 1, ?6, ?7, ?8, ?9, ?10,
                 'pending', 0, unixepoch(), unixepoch(), unixepoch())",
        params![
            id,
            KNOWLEDGE_ROOT_ID,
            path,
            title,
            body_md.len() as i64,
            content_type,
            body_md,
            source_url,
            collection_id,
            position,
        ],
    )
    .map_err(|err| format!("Failed to create text document: {err}"))?;
    mirror_to_vault(&conn, &id, &title, body_md);
    Ok(id)
}

/// Write an authored source through to its Markdown file. Best-effort: the row
/// is already committed, and a vault problem (read-only volume, a sync client
/// holding the file) must never turn into a failed save for the user.
fn mirror_to_vault(conn: &Connection, document_id: &str, title: &str, body_md: &str) {
    if let Err(err) = crate::vault::write_note(conn, document_id, title, body_md) {
        log::warn!("Failed to mirror note {document_id} to the vault: {err}");
    }
}

/// Update an editable source's title + body and mark it stale so the next
/// reindex re-chunks the new content. Errors if the row is missing or not an
/// editable content type.
pub(crate) fn update_text_document_body(
    database: &State<'_, AppDatabase>,
    document_id: &str,
    title: &str,
    body_md: &str,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let title = {
        let trimmed = title.trim();
        if trimmed.is_empty() {
            "Untitled".to_string()
        } else {
            trimmed.to_string()
        }
    };
    let affected = conn
        .execute(
            "UPDATE documents
             SET title = ?2, short_title = ?2, body_md = ?3, file_size = ?4,
                 modified = unixepoch(), updated_at = unixepoch(),
                 index_status = 'stale'
             WHERE id = ?1
               AND content_type IN ('note', 'markdown', 'text', 'web')",
            params![document_id, title, body_md, body_md.len() as i64],
        )
        .map_err(|err| format!("Failed to update text document: {err}"))?;
    if affected == 0 {
        return Err("Note not found or is not an editable source".to_string());
    }
    mirror_to_vault(&conn, document_id, &title, body_md);
    Ok(())
}

/// Load an authored source's body for the editor.
pub(crate) fn load_text_document_body(
    database: &State<'_, AppDatabase>,
    document_id: &str,
) -> Result<TextDocumentBody, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.query_row(
        "SELECT title, COALESCE(body_md, ''), COALESCE(source_url, ''), content_type
         FROM documents WHERE id = ?1",
        params![document_id],
        |row| {
            Ok(TextDocumentBody {
                document_id: document_id.to_string(),
                title: row.get(0)?,
                body_md: row.get(1)?,
                source_url: row.get(2)?,
                content_type: row.get(3)?,
            })
        },
    )
    .optional()
    .map_err(|err| format!("Failed to load text document: {err}"))?
    .ok_or_else(|| "Document not found".to_string())
}

fn load_existing_document_ids(
    tx: &rusqlite::Transaction<'_>,
    workspace_root_id: &str,
) -> Result<Vec<String>, String> {
    // Knowledge-base pivot (P2): directory rescans reconcile only file-backed
    // PDFs (the only kind `collect_pdfs` produces). Notes / web clips / imported
    // markdown are not on the scanned disk tree, so they must never be reaped by
    // the stale-deletion pass — restrict reconciliation to content_type='pdf'.
    let mut stmt = tx
        .prepare("SELECT id FROM documents WHERE workspace_root_id = ?1 AND content_type = 'pdf'")
        .map_err(|err| format!("Failed to load existing documents: {err}"))?;
    let document_ids = stmt
        .query_map(params![workspace_root_id], |row| row.get::<_, String>(0))
        .map_err(|err| format!("Failed to load existing documents: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to load existing documents: {err}"))?;

    Ok(document_ids)
}

fn should_ignore_name(name: &str) -> bool {
    name.starts_with('.') || matches!(name, "node_modules" | "target" | "dist" | "build" | ".git")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_documents_qualifies_joined_id_columns() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(&format!(
            "
            CREATE TABLE documents (
              id TEXT PRIMARY KEY,
              workspace_root_id TEXT NOT NULL,
              path TEXT NOT NULL UNIQUE,
              title TEXT NOT NULL,
              short_title TEXT NOT NULL,
              file_size INTEGER NOT NULL,
              modified INTEGER NOT NULL,
              page_count INTEGER NOT NULL DEFAULT 0,
              last_page INTEGER NOT NULL DEFAULT 1,
              content_type TEXT NOT NULL DEFAULT 'pdf',
              collection_id TEXT,
              position INTEGER NOT NULL DEFAULT 0,
              index_status TEXT NOT NULL DEFAULT 'pending',
              index_version INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE structure_tree_nodes (
              id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL
            );
            CREATE TABLE document_index_jobs (
              id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL,
              job_type TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              version INTEGER NOT NULL DEFAULT 0,
              error TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO documents
              (id, workspace_root_id, path, title, short_title, file_size, modified,
               page_count, last_page, index_status, index_version)
            VALUES
              ('doc-1', 'root-1', '/tmp/a.pdf', 'A', 'A', 1, 0, 2, 1,
               'indexed', {CURRENT_INDEX_VERSION});
            INSERT INTO structure_tree_nodes (id, document_id)
            VALUES ('node-1', 'doc-1');
            INSERT INTO document_index_jobs
              (id, document_id, job_type, status, version, error)
            VALUES
              ('job-1', 'doc-1', 'visual_tsr', 'succeeded', {CURRENT_INDEX_VERSION}, '');
            ",
        ))
        .expect("schema");

        let docs = load_documents_for_root_conn(&conn, "root-1").expect("load documents");

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "doc-1");
        assert!(docs[0].tree_ready);
        assert_eq!(docs[0].visual_index_status, "succeeded");
    }
}
