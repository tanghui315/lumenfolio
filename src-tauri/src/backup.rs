//! Database snapshots.
//!
//! Notes are already safe on disk as Markdown (see `vault`), but the rest of the
//! user's authored state — how sources are filed into collections, and every
//! chat they have had with their documents — exists only in SQLite. This makes a
//! restorable copy of it.
//!
//! Snapshots are taken with `VACUUM INTO`, never by copying the file. Copying a
//! live SQLite database is the corruption mode this whole design exists to avoid:
//! the reader can observe a half-written page set. `VACUUM INTO` runs inside a
//! read transaction and emits a consistent, already-compacted database — safe to
//! run while the app is in use, and small enough to upload.
//!
//! Restoring never writes over the database while it is open. The chosen
//! snapshot is staged beside it and swapped in at the next launch, before any
//! connection exists.

use std::path::{Path, PathBuf};

use rusqlite::Connection;

pub(crate) const BACKUP_DIR_SETTING: &str = "backup_dir";
pub(crate) const BACKUP_KEEP_SETTING: &str = "backup_keep";
/// Hours between automatic snapshots; 0 (the default) means manual only.
pub(crate) const BACKUP_INTERVAL_SETTING: &str = "backup_interval_hours";
/// Unix seconds of the last successful snapshot, automatic or manual.
pub(crate) const BACKUP_LAST_AT_SETTING: &str = "backup_last_at";

/// Whether an automatic snapshot is due.
///
/// A manual backup also stamps `last_at`, so taking one by hand postpones the
/// next automatic one rather than being ignored by it. An unknown `last_at`
/// (never backed up) is due immediately, which is the useful default: the user
/// has just switched the schedule on.
pub(crate) fn auto_backup_due(last_at: Option<i64>, interval_hours: u32, now: i64) -> bool {
    if interval_hours == 0 {
        return false;
    }
    match last_at {
        Some(last) => {
            // A clock that jumped backwards would otherwise wedge this off
            // forever, so treat any future stamp as due.
            if last > now {
                return true;
            }
            now - last >= i64::from(interval_hours) * 3600
        }
        None => true,
    }
}
/// Suffix of a snapshot staged for the next launch to swap in.
const RESTORE_SUFFIX: &str = ".restore";
/// Snapshots older than the keep count are pruned; a floor stops a bad setting
/// from deleting everything.
pub(crate) const DEFAULT_KEEP: usize = 10;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackupEntry {
    pub name: String,
    pub path: String,
    pub size: u64,
    /// Unix seconds, parsed from the file name so listing does not depend on
    /// filesystem timestamps surviving a copy between machines.
    pub created_at: i64,
}

/// `lumenfolio-20260731-143022.sqlite` — sorts chronologically as text, and is
/// readable when the user is picking one to restore.
pub(crate) fn snapshot_name(now: chrono::DateTime<chrono::Local>) -> String {
    format!("lumenfolio-{}.sqlite", now.format("%Y%m%d-%H%M%S"))
}

fn parse_snapshot_time(name: &str) -> Option<i64> {
    let stem = name.strip_prefix("lumenfolio-")?.strip_suffix(".sqlite")?;
    let naive = chrono::NaiveDateTime::parse_from_str(stem, "%Y%m%d-%H%M%S").ok()?;
    Some(naive.and_utc().timestamp())
}

/// Write a consistent snapshot of `conn`'s database to `dest`.
///
/// `VACUUM INTO` refuses to overwrite, so a stale file at the target is removed
/// first — otherwise a second backup within the same second would fail.
pub(crate) fn write_snapshot(conn: &Connection, dest: &Path) -> Result<u64, String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create backup directory {}: {err}",
                parent.display()
            )
        })?;
    }
    if dest.exists() {
        std::fs::remove_file(dest)
            .map_err(|err| format!("Failed to replace {}: {err}", dest.display()))?;
    }
    // Bound as a literal: VACUUM INTO does not accept a bound parameter.
    let escaped = dest.to_string_lossy().replace('\'', "''");
    conn.execute_batch(&format!("VACUUM INTO '{escaped}'"))
        .map_err(|err| format!("Failed to write snapshot: {err}"))?;
    std::fs::metadata(dest)
        .map(|meta| meta.len())
        .map_err(|err| format!("Snapshot was not written: {err}"))
}

/// Snapshots in `dir`, newest first.
pub(crate) fn list_snapshots(dir: &Path) -> Vec<BackupEntry> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<BackupEntry> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?.to_string();
            let created_at = parse_snapshot_time(&name)?;
            let size = entry.metadata().ok().map(|meta| meta.len()).unwrap_or(0);
            Some(BackupEntry {
                name,
                path: path.to_string_lossy().to_string(),
                size,
                created_at,
            })
        })
        .collect();
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    out
}

/// Delete all but the newest `keep` snapshots. Returns how many were removed.
pub(crate) fn prune_snapshots(dir: &Path, keep: usize) -> usize {
    let keep = keep.max(1);
    let snapshots = list_snapshots(dir);
    let mut removed = 0;
    for entry in snapshots.into_iter().skip(keep) {
        if std::fs::remove_file(&entry.path).is_ok() {
            removed += 1;
        }
    }
    removed
}

fn restore_marker(db_path: &Path) -> PathBuf {
    let mut marker = db_path.as_os_str().to_os_string();
    marker.push(RESTORE_SUFFIX);
    PathBuf::from(marker)
}

/// Stage `snapshot` to replace the live database at the next launch.
///
/// Deliberately does not touch the database itself: it is open, and other
/// connections (the index writer) may hold it. Validity is checked here rather
/// than at startup so the user learns immediately if they picked a bad file.
pub(crate) fn stage_restore(snapshot: &Path, db_path: &Path) -> Result<(), String> {
    let probe =
        Connection::open(snapshot).map_err(|err| format!("Cannot open that snapshot: {err}"))?;
    let ok: String = probe
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .map_err(|err| format!("Cannot verify that snapshot: {err}"))?;
    if ok != "ok" {
        return Err(format!(
            "That snapshot is damaged ({ok}) and was not staged."
        ));
    }
    let documents: i64 = probe
        .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
        .map_err(|_| "That file is not a Lumenfolio database.".to_string())?;
    drop(probe);
    log::info!(
        "Staging restore of {} ({documents} documents)",
        snapshot.display()
    );
    std::fs::copy(snapshot, restore_marker(db_path))
        .map_err(|err| format!("Failed to stage the restore: {err}"))?;
    Ok(())
}

/// Swap in a staged snapshot. MUST run before the database is opened.
///
/// The database being replaced is kept beside it rather than deleted, so a
/// restore of the wrong snapshot is itself recoverable.
pub(crate) fn apply_staged_restore(db_path: &Path) -> bool {
    let marker = restore_marker(db_path);
    if !marker.exists() {
        return false;
    }
    if db_path.exists() {
        let mut replaced = db_path.as_os_str().to_os_string();
        replaced.push(".replaced");
        let replaced = PathBuf::from(replaced);
        let _ = std::fs::remove_file(&replaced);
        if let Err(err) = std::fs::rename(db_path, &replaced) {
            log::warn!("Restore aborted: could not move the current database aside: {err}");
            return false;
        }
    }
    // WAL/SHM belong to the old database; leaving them would be read as that
    // database's journal and corrupt the restored file.
    for suffix in ["-wal", "-shm"] {
        let mut side = db_path.as_os_str().to_os_string();
        side.push(suffix);
        let _ = std::fs::remove_file(PathBuf::from(side));
    }
    match std::fs::rename(&marker, db_path) {
        Ok(()) => {
            log::info!("Restored the database from a staged snapshot");
            true
        }
        Err(err) => {
            log::warn!("Restore failed while swapping the snapshot in: {err}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-backup-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn seeded_db(path: &Path) -> Connection {
        let conn = Connection::open(path).expect("open");
        conn.execute_batch(
            "CREATE TABLE documents (id TEXT PRIMARY KEY);
             INSERT INTO documents (id) VALUES ('d1'), ('d2');",
        )
        .expect("seed");
        conn
    }

    #[test]
    fn auto_backup_is_due_only_after_the_configured_interval() {
        const DAY: i64 = 24 * 3600;
        // Off by default: no schedule, never due, whatever the history.
        assert!(!auto_backup_due(None, 0, DAY));
        assert!(!auto_backup_due(Some(0), 0, DAY * 10));
        // Never backed up but scheduled → due now.
        assert!(auto_backup_due(None, 24, DAY));
        // Inside the window → not yet; at/after it → due.
        assert!(!auto_backup_due(Some(DAY), 24, DAY + 3600));
        assert!(auto_backup_due(Some(DAY), 24, DAY * 2));
        assert!(auto_backup_due(Some(DAY), 24, DAY * 3));
        // Weekly honors its own interval.
        assert!(!auto_backup_due(Some(DAY), 24 * 7, DAY * 4));
        assert!(auto_backup_due(Some(DAY), 24 * 7, DAY * 8));
        // A backwards clock jump must not wedge the schedule off forever.
        assert!(auto_backup_due(Some(DAY * 10), 24, DAY));
    }

    #[test]
    fn snapshot_names_sort_chronologically_and_round_trip() {
        let early = chrono::NaiveDate::from_ymd_opt(2026, 7, 31)
            .unwrap()
            .and_hms_opt(9, 5, 3)
            .unwrap();
        let name = format!("lumenfolio-{}.sqlite", early.format("%Y%m%d-%H%M%S"));
        assert_eq!(name, "lumenfolio-20260731-090503.sqlite");
        assert_eq!(
            parse_snapshot_time(&name),
            Some(early.and_utc().timestamp())
        );
        // Text order matches time order, which is what listing relies on.
        assert!("lumenfolio-20260731-090503.sqlite" < "lumenfolio-20260731-143022.sqlite");
        assert_eq!(parse_snapshot_time("notes.md"), None);
    }

    #[test]
    fn snapshot_is_a_usable_copy_and_overwrites_a_stale_file() {
        let dir = temp_dir("snap");
        let conn = seeded_db(&dir.join("live.sqlite"));
        let dest = dir.join("snap.sqlite");

        let size = write_snapshot(&conn, &dest).expect("snapshot");
        assert!(size > 0);
        let restored = Connection::open(&dest).expect("open snapshot");
        let count: i64 = restored
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .expect("count");
        assert_eq!(count, 2);
        drop(restored);

        // VACUUM INTO refuses an existing target; a second backup must still work.
        conn.execute("INSERT INTO documents (id) VALUES ('d3')", [])
            .expect("insert");
        write_snapshot(&conn, &dest).expect("second snapshot");
        let restored = Connection::open(&dest).expect("reopen");
        let count: i64 = restored
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .expect("count");
        assert_eq!(count, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_keeps_only_the_newest() {
        let dir = temp_dir("prune");
        for name in [
            "lumenfolio-20260731-090000.sqlite",
            "lumenfolio-20260731-100000.sqlite",
            "lumenfolio-20260731-110000.sqlite",
            "unrelated.txt",
        ] {
            std::fs::write(dir.join(name), b"x").expect("write");
        }
        assert_eq!(prune_snapshots(&dir, 2), 1);
        let left = list_snapshots(&dir);
        assert_eq!(left.len(), 2);
        // Newest first, and the oldest is the one that went.
        assert_eq!(left[0].name, "lumenfolio-20260731-110000.sqlite");
        assert_eq!(left[1].name, "lumenfolio-20260731-100000.sqlite");
        // A non-snapshot file in the folder is never touched.
        assert!(dir.join("unrelated.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restore_stages_and_swaps_keeping_the_replaced_database() {
        let dir = temp_dir("restore");
        let db_path = dir.join("live.sqlite");
        let conn = seeded_db(&db_path);
        let snapshot = dir.join("snap.sqlite");
        write_snapshot(&conn, &snapshot).expect("snapshot");
        // Diverge the live database after the snapshot.
        conn.execute("DELETE FROM documents", []).expect("clear");
        drop(conn);
        // A stale WAL must not survive the swap.
        std::fs::write(dir.join("live.sqlite-wal"), b"stale").expect("wal");

        stage_restore(&snapshot, &db_path).expect("stage");
        assert!(apply_staged_restore(&db_path));

        let conn = Connection::open(&db_path).expect("reopen");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .expect("count");
        assert_eq!(count, 2, "the snapshot's rows must be back");
        // The replaced database is kept, so restoring the wrong file is undoable.
        assert!(dir.join("live.sqlite.replaced").exists());
        assert!(!dir.join("live.sqlite-wal").exists());
        // Idempotent: nothing staged, nothing to do.
        assert!(!apply_staged_restore(&db_path));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn staging_rejects_a_file_that_is_not_a_lumenfolio_database() {
        let dir = temp_dir("reject");
        let db_path = dir.join("live.sqlite");
        drop(seeded_db(&db_path));
        let bogus = dir.join("bogus.sqlite");
        std::fs::write(&bogus, b"not a database").expect("write");
        assert!(stage_restore(&bogus, &db_path).is_err());

        // A valid SQLite file that is not ours is rejected too.
        let other = dir.join("other.sqlite");
        Connection::open(&other)
            .expect("open")
            .execute_batch("CREATE TABLE t (x)")
            .expect("schema");
        assert!(stage_restore(&other, &db_path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
