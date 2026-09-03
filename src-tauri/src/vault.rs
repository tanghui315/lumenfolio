//! Notes on disk: a plain-Markdown mirror of every authored note.
//!
//! Notes used to live only in `documents.body_md`. That made the SQLite file the
//! single point of failure for the one kind of data the user cannot regenerate —
//! everything else (chunks, blocks, FTS, visual assets) is a derived index, and
//! PDFs/Office files already exist on disk.
//!
//! It also blocked the obvious way to protect them. A SQLite database must never
//! sit in a cloud-sync folder: the sync process copies the file while it is open
//! and mid-write, which is one of the corruption modes SQLite documents. Plain
//! files have no such problem, so mirroring notes to a user-chosen folder means
//! iCloud / Dropbox / Syncthing / WebDAV all work with no integration on our side.
//!
//! Direction of truth: the database stays authoritative and every save writes
//! through to the file. Files are the durable copy and the recovery path
//! (`import_orphans`), NOT a second live editing surface — picking up external
//! edits needs conflict resolution and is deliberately not attempted here.

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};

pub(crate) const VAULT_DIR_SETTING: &str = "notes_vault_dir";

/// Where notes are mirrored. Configurable; defaults under the user's Documents
/// so it is somewhere they can point a sync client at without hunting.
pub(crate) fn vault_dir(conn: &Connection) -> Option<PathBuf> {
    let configured: Option<String> = conn
        .query_row(
            "SELECT value FROM app_settings WHERE key = ?1",
            params![VAULT_DIR_SETTING],
            |row| row.get(0),
        )
        .ok();
    let configured = configured.map(|value| value.trim().to_string());
    match configured {
        // An explicitly blank setting means the user turned mirroring off.
        Some(value) if value.is_empty() => None,
        Some(value) => Some(PathBuf::from(value)),
        None => default_vault_dir(),
    }
}

/// `~/Documents/Lumenfolio/Notes`, resolved from the environment rather than by
/// adding a directories crate for one path. Falls back to the home directory if
/// there is no Documents folder.
fn default_vault_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    let documents = home.join("Documents");
    let base = if documents.is_dir() { documents } else { home };
    Some(base.join("Lumenfolio").join("Notes"))
}

/// Turn a note title into a safe file stem, Obsidian-style (the title IS the
/// file name). Path separators and characters Windows rejects are replaced, and
/// the result is length-capped to stay under filesystem limits.
pub(crate) fn file_stem_for_title(title: &str, document_id: &str) -> String {
    let cleaned: String = title
        .trim()
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '-',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect();
    let cleaned = cleaned.trim().trim_matches('.').trim().to_string();
    // Cap by characters, not bytes, so a CJK title is not cut mid-codepoint.
    let capped: String = cleaned.chars().take(80).collect();
    let capped = capped.trim().to_string();
    if capped.is_empty() {
        // Untitled or punctuation-only: fall back to the id so the file still
        // has a stable, unique name.
        return document_id.to_string();
    }
    capped
}

/// Serialize a note. The id lives in YAML front matter so a recovered file maps
/// back to its row unambiguously — matching on the title would break the moment
/// two notes share one, or a title is edited.
fn note_file_contents(document_id: &str, title: &str, body_md: &str) -> String {
    let safe_title = title.replace('\n', " ");
    let body = body_md.strip_prefix('\n').unwrap_or(body_md);
    format!(
        "---\nid: {document_id}\ntitle: \"{}\"\n---\n\n{body}",
        safe_title.replace('"', "'")
    )
}

/// Split stored front matter back off a file, returning `(id, body)`.
pub(crate) fn parse_note_file(contents: &str) -> (Option<String>, String) {
    let Some(rest) = contents.strip_prefix("---\n") else {
        return (None, contents.to_string());
    };
    let Some(end) = rest.find("\n---\n") else {
        return (None, contents.to_string());
    };
    let (front, body) = rest.split_at(end);
    let id = front.lines().find_map(|line| {
        line.strip_prefix("id:")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    });
    let body = body.strip_prefix("\n---\n").unwrap_or(body);
    (id, body.trim_start_matches('\n').to_string())
}

fn note_path(dir: &Path, title: &str, document_id: &str) -> PathBuf {
    dir.join(format!("{}.md", file_stem_for_title(title, document_id)))
}

/// Mirror one note to disk, removing any file left behind by a previous title.
///
/// Best-effort by design: the caller has already committed to the database, and
/// a mirror failure (read-only volume, sync client holding a lock) must not fail
/// the user's save. Errors are returned for logging, not propagated as save
/// failures.
pub(crate) fn write_note(
    conn: &Connection,
    document_id: &str,
    title: &str,
    body_md: &str,
) -> Result<(), String> {
    let Some(dir) = vault_dir(conn) else {
        return Ok(());
    };
    std::fs::create_dir_all(&dir)
        .map_err(|err| format!("Failed to create vault directory {}: {err}", dir.display()))?;
    let target = note_path(&dir, title, document_id);
    // A rename leaves the old file orphaned; find it by id rather than guessing
    // the previous title.
    remove_other_files_for_id(&dir, document_id, Some(&target));
    std::fs::write(&target, note_file_contents(document_id, title, body_md))
        .map_err(|err| format!("Failed to write note file {}: {err}", target.display()))
}

/// Delete a note's mirror. Called when the source is deleted so the vault does
/// not resurrect it on the next import.
pub(crate) fn delete_note(conn: &Connection, document_id: &str) {
    let Some(dir) = vault_dir(conn) else {
        return;
    };
    remove_other_files_for_id(&dir, document_id, None);
}

/// Remove every file in `dir` whose front matter carries `document_id`, except
/// `keep`. Scanning by id (not by name) is what makes renames and duplicates
/// self-healing.
fn remove_other_files_for_id(dir: &Path, document_id: &str, keep: Option<&Path>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }
        if keep.is_some_and(|keep| keep == path) {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(&path) else {
            continue;
        };
        if parse_note_file(&contents).0.as_deref() == Some(document_id) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// A note file on disk with no matching row — what recovery reads.
pub(crate) struct OrphanNote {
    pub title: String,
    pub body_md: String,
}

/// Vault files whose id is absent from the database.
///
/// This is the restore path after a lost or reset database. It is deliberately
/// one-way and additive: files that still have a row are skipped entirely rather
/// than compared, so this can never overwrite current work with a stale copy.
/// Two-way merge would need real conflict resolution.
pub(crate) fn import_orphans(conn: &Connection) -> Result<Vec<OrphanNote>, String> {
    let Some(dir) = vault_dir(conn) else {
        return Ok(Vec::new());
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    let mut orphans = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(&path) else {
            continue;
        };
        let (id, body) = parse_note_file(&contents);
        if let Some(id) = &id {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM documents WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            if exists > 0 {
                continue;
            }
        }
        let title = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("Untitled")
            .to_string();
        orphans.push(OrphanNote {
            title,
            body_md: body,
        });
    }
    Ok(orphans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_stem_sanitizes_and_falls_back_to_the_id() {
        assert_eq!(
            file_stem_for_title("Weekly notes", "note-1"),
            "Weekly notes"
        );
        // Path separators and Windows-illegal characters cannot reach the name.
        assert_eq!(file_stem_for_title("a/b:c?d", "note-1"), "a-b-c-d");
        assert_eq!(file_stem_for_title("   ", "note-1"), "note-1");
        // CJK titles are capped by character, never mid-codepoint.
        let long = "长".repeat(200);
        assert_eq!(file_stem_for_title(&long, "note-1").chars().count(), 80);
    }

    #[test]
    fn note_file_round_trips_through_front_matter() {
        let text = note_file_contents("note-7", "My title", "# Body\n\nHello [[link]].\n");
        let (id, body) = parse_note_file(&text);
        assert_eq!(id.as_deref(), Some("note-7"));
        // Body is byte-identical: wikilinks and spacing must survive the mirror.
        assert_eq!(body, "# Body\n\nHello [[link]].\n");
    }

    fn test_conn(dir: &Path) -> Connection {
        let conn = Connection::open_in_memory().expect("open");
        conn.execute_batch(
            "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,
                 updated_at INTEGER NOT NULL DEFAULT 0);
             CREATE TABLE documents (id TEXT PRIMARY KEY);",
        )
        .expect("schema");
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?1, ?2)",
            params![VAULT_DIR_SETTING, dir.to_string_lossy()],
        )
        .expect("setting");
        conn
    }

    /// The whole point of the mirror: a note reaches disk, a rename does not
    /// leave a duplicate behind, deleting removes it, and a file left without a
    /// row is recoverable.
    #[test]
    fn notes_mirror_rename_delete_and_recover() {
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-vault-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let conn = test_conn(&dir);

        write_note(&conn, "note-1", "First title", "body one\n").expect("write");
        assert!(dir.join("First title.md").exists());

        // Rename: the new file appears and the old one is cleaned up, so the
        // vault never accumulates stale copies of the same note.
        write_note(&conn, "note-1", "Second title", "body one\n").expect("rename");
        assert!(dir.join("Second title.md").exists());
        assert!(!dir.join("First title.md").exists());

        // A file whose id has no row is an orphan → offered for recovery.
        let orphans = import_orphans(&conn).expect("orphans");
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].title, "Second title");
        assert_eq!(orphans[0].body_md, "body one\n");

        // With the row present it is current work, not a recovery candidate —
        // import must never overwrite it.
        conn.execute("INSERT INTO documents (id) VALUES ('note-1')", [])
            .expect("row");
        assert!(import_orphans(&conn).expect("orphans").is_empty());

        delete_note(&conn, "note-1");
        assert!(!dir.join("Second title.md").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_note_file_tolerates_a_plain_markdown_file() {
        // A file the user dropped in by hand has no front matter.
        let (id, body) = parse_note_file("# Just markdown\n");
        assert!(id.is_none());
        assert_eq!(body, "# Just markdown\n");
    }
}
