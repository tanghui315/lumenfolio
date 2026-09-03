//! Knowledge-base pivot (Collections): user-authored logical folders.
//!
//! Collections are a pure metadata layer — nestable, user-named, decoupled from
//! disk directories. A source's home is `documents.collection_id` (single
//! membership, folder metaphor; NULL = unfiled "Inbox"). Nothing here touches
//! files on disk: importing records a real path, filing only moves the pointer.

use rusqlite::params;
use tauri::State;

use crate::AppDatabase;

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CollectionNode {
    pub id: String,
    pub parent_id: Option<String>,
    pub name: String,
    pub position: i64,
}

fn new_collection_id() -> String {
    format!(
        "col-{:016x}{:016x}",
        rand::random::<u64>(),
        rand::random::<u64>()
    )
}

fn normalize_parent(parent_id: Option<String>) -> Option<String> {
    parent_id
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Load the whole collection forest (flat; the frontend nests by parent_id).
#[tauri::command]
pub(crate) fn load_collections(
    database: State<'_, AppDatabase>,
) -> Result<Vec<CollectionNode>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT id, parent_id, name, position
             FROM collections
             ORDER BY position ASC, lower(name) ASC",
        )
        .map_err(|err| format!("Failed to load collections: {err}"))?;
    let rows = stmt
        .query_map([], |row| {
            Ok(CollectionNode {
                id: row.get(0)?,
                parent_id: row.get(1)?,
                name: row.get(2)?,
                position: row.get(3)?,
            })
        })
        .map_err(|err| format!("Failed to load collections: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to load collections: {err}"))?;
    Ok(rows)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CreateCollectionInput {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub parent_id: Option<String>,
}

#[tauri::command]
pub(crate) fn create_collection(
    input: CreateCollectionInput,
    database: State<'_, AppDatabase>,
) -> Result<CollectionNode, String> {
    let name = {
        let trimmed = input.name.trim();
        if trimmed.is_empty() {
            "Untitled".to_string()
        } else {
            trimmed.to_string()
        }
    };
    let parent_id = normalize_parent(input.parent_id);
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    if let Some(parent) = &parent_id {
        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM collections WHERE id = ?1",
                params![parent],
                |row| row.get(0),
            )
            .map_err(|err| format!("Failed to check parent collection: {err}"))?;
        if exists == 0 {
            return Err("Parent collection no longer exists".to_string());
        }
    }
    let position: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM collections WHERE parent_id IS ?1",
            params![parent_id],
            |row| row.get(0),
        )
        .map_err(|err| format!("Failed to compute collection position: {err}"))?;
    let id = new_collection_id();
    conn.execute(
        "INSERT INTO collections (id, parent_id, name, position, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, unixepoch(), unixepoch())",
        params![id, parent_id, name, position],
    )
    .map_err(|err| format!("Failed to create collection: {err}"))?;
    Ok(CollectionNode {
        id,
        parent_id,
        name,
        position,
    })
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RenameCollectionInput {
    pub id: String,
    #[serde(default)]
    pub name: String,
}

#[tauri::command]
pub(crate) fn rename_collection(
    input: RenameCollectionInput,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let name = input.name.trim();
    if name.is_empty() {
        return Err("Collection name cannot be empty".to_string());
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let affected = conn
        .execute(
            "UPDATE collections SET name = ?2, updated_at = unixepoch() WHERE id = ?1",
            params![input.id.trim(), name],
        )
        .map_err(|err| format!("Failed to rename collection: {err}"))?;
    if affected == 0 {
        return Err("Collection not found".to_string());
    }
    Ok(())
}

/// Delete a collection and its whole subtree. Documents anywhere in that subtree
/// drop to unfiled (collection_id → NULL); the source rows and the files on disk
/// are never touched. Runs the subtree walk explicitly so behavior is identical
/// whether or not the SQLite build enforces the FKs.
#[tauri::command]
pub(crate) fn delete_collection(
    id: String,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let id = id.trim().to_string();
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start delete transaction: {err}"))?;
    const SUBTREE_CTE: &str = "WITH RECURSIVE sub(id) AS (
            SELECT ?1
            UNION ALL
            SELECT c.id FROM collections c JOIN sub ON c.parent_id = sub.id
        )";
    tx.execute(
        &format!(
            "UPDATE documents SET collection_id = NULL
             WHERE collection_id IN ({SUBTREE_CTE} SELECT id FROM sub)"
        ),
        params![id],
    )
    .map_err(|err| format!("Failed to unfile documents: {err}"))?;
    tx.execute(
        &format!("DELETE FROM collections WHERE id IN ({SUBTREE_CTE} SELECT id FROM sub)"),
        params![id],
    )
    .map_err(|err| format!("Failed to delete collection: {err}"))?;
    tx.commit()
        .map_err(|err| format!("Failed to commit collection delete: {err}"))?;
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MoveDocumentInput {
    pub document_id: String,
    #[serde(default)]
    pub collection_id: Option<String>,
}

#[tauri::command]
pub(crate) fn move_document_to_collection(
    input: MoveDocumentInput,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let collection_id = normalize_parent(input.collection_id);
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    if let Some(target) = &collection_id {
        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM collections WHERE id = ?1",
                params![target],
                |row| row.get(0),
            )
            .map_err(|err| format!("Failed to check target collection: {err}"))?;
        if exists == 0 {
            return Err("Target collection no longer exists".to_string());
        }
    }
    // Append to the bottom of the target collection's manual order.
    let position = crate::documents::next_document_position(&conn, collection_id.as_deref())?;
    let affected = conn
        .execute(
            "UPDATE documents SET collection_id = ?2, position = ?3, updated_at = unixepoch()
             WHERE id = ?1",
            params![input.document_id.trim(), collection_id, position],
        )
        .map_err(|err| format!("Failed to move document: {err}"))?;
    if affected == 0 {
        return Err("Document not found".to_string());
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MoveCollectionInput {
    pub id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
}

/// Reparent a collection. Rejects moving a collection under itself or one of its
/// own descendants (which would orphan a cycle).
#[tauri::command]
pub(crate) fn move_collection(
    input: MoveCollectionInput,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let id = input.id.trim().to_string();
    let parent_id = normalize_parent(input.parent_id);
    if parent_id.as_deref() == Some(id.as_str()) {
        return Err("A collection cannot be its own parent".to_string());
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    if let Some(target) = &parent_id {
        // Reject if `target` is inside `id`'s own subtree.
        let in_subtree: i64 = conn
            .query_row(
                "WITH RECURSIVE sub(id) AS (
                    SELECT ?1
                    UNION ALL
                    SELECT c.id FROM collections c JOIN sub ON c.parent_id = sub.id
                 )
                 SELECT COUNT(*) FROM sub WHERE id = ?2",
                params![id, target],
                |row| row.get(0),
            )
            .map_err(|err| format!("Failed to check for a cycle: {err}"))?;
        if in_subtree > 0 {
            return Err("Cannot move a collection into its own descendant".to_string());
        }
    }
    // Append to the bottom of the new parent's manual order (its old position is
    // meaningless under a different parent).
    let position: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM collections WHERE parent_id IS ?1",
            params![parent_id],
            |row| row.get(0),
        )
        .map_err(|err| format!("Failed to compute collection position: {err}"))?;
    let affected = conn
        .execute(
            "UPDATE collections SET parent_id = ?2, position = ?3, updated_at = unixepoch()
             WHERE id = ?1",
            params![id, parent_id, position],
        )
        .map_err(|err| format!("Failed to move collection: {err}"))?;
    if affected == 0 {
        return Err("Collection not found".to_string());
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ReorderInput {
    /// Sibling scope: the collection the documents belong to, or the parent of
    /// the collections being reordered. None = the tree root.
    #[serde(default)]
    pub parent_id: Option<String>,
    /// The sibling ids in their new top-to-bottom order.
    #[serde(default)]
    pub ordered_ids: Vec<String>,
}

/// Renumber a sibling list 0..N inside a single transaction. `update_sql` is an
/// UPDATE with ?1=id, ?2=new index, ?3=scope; its `... IS ?3` guard makes a
/// stale or foreign id a no-op rather than pulling a row out of another parent.
fn reorder_rows(
    conn: &mut rusqlite::Connection,
    update_sql: &str,
    scope: Option<&str>,
    ordered_ids: &[String],
) -> Result<(), String> {
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start reorder transaction: {err}"))?;
    {
        let mut stmt = tx
            .prepare(update_sql)
            .map_err(|err| format!("Failed to prepare reorder: {err}"))?;
        for (index, id) in ordered_ids.iter().enumerate() {
            stmt.execute(params![id.trim(), index as i64, scope])
                .map_err(|err| format!("Failed to reorder row: {err}"))?;
        }
    }
    tx.commit()
        .map_err(|err| format!("Failed to commit reorder: {err}"))
}

const REORDER_DOCUMENTS_SQL: &str = "UPDATE documents SET position = ?2, updated_at = unixepoch()
     WHERE id = ?1 AND collection_id IS ?3";
const REORDER_COLLECTIONS_SQL: &str =
    "UPDATE collections SET position = ?2, updated_at = unixepoch()
     WHERE id = ?1 AND parent_id IS ?3";

/// Persist a manual reorder of documents within one collection (or the root).
/// The frontend sends the full sibling list in its new top-to-bottom order.
#[tauri::command]
pub(crate) fn reorder_documents(
    input: ReorderInput,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let scope = normalize_parent(input.parent_id);
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    reorder_rows(
        &mut conn,
        REORDER_DOCUMENTS_SQL,
        scope.as_deref(),
        &input.ordered_ids,
    )
}

/// Persist a manual reorder of sibling collections under one parent (or root).
#[tauri::command]
pub(crate) fn reorder_collections(
    input: ReorderInput,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let scope = normalize_parent(input.parent_id);
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    reorder_rows(
        &mut conn,
        REORDER_COLLECTIONS_SQL,
        scope.as_deref(),
        &input.ordered_ids,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn seed(conn: &Connection) {
        conn.execute_batch(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, collection_id TEXT,
                 position INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0);
             CREATE TABLE collections (id TEXT PRIMARY KEY, parent_id TEXT,
                 position INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0);",
        )
        .unwrap();
    }

    fn positions(conn: &Connection, scope_is_null: bool) -> Vec<(String, i64)> {
        let where_clause = if scope_is_null {
            "collection_id IS NULL"
        } else {
            "collection_id = 'c1'"
        };
        let mut stmt = conn
            .prepare(&format!(
                "SELECT id, position FROM documents WHERE {where_clause} ORDER BY position"
            ))
            .unwrap();
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn reorder_documents_renumbers_siblings_in_the_given_order() {
        let mut conn = Connection::open_in_memory().unwrap();
        seed(&conn);
        conn.execute_batch(
            "INSERT INTO documents (id, collection_id, position) VALUES
                ('a', 'c1', 0), ('b', 'c1', 1), ('c', 'c1', 2);",
        )
        .unwrap();
        // New order: c, a, b.
        reorder_rows(
            &mut conn,
            REORDER_DOCUMENTS_SQL,
            Some("c1"),
            &["c".into(), "a".into(), "b".into()],
        )
        .unwrap();
        assert_eq!(
            positions(&conn, false),
            vec![("c".into(), 0), ("a".into(), 1), ("b".into(), 2)]
        );
    }

    #[test]
    fn reorder_documents_scope_guard_ignores_foreign_ids() {
        let mut conn = Connection::open_in_memory().unwrap();
        seed(&conn);
        conn.execute_batch(
            "INSERT INTO documents (id, collection_id, position) VALUES
                ('a', 'c1', 0), ('b', 'c1', 1), ('x', 'c2', 5);",
        )
        .unwrap();
        // 'x' belongs to c2; passing it in a c1 reorder must not move it out of c2.
        reorder_rows(
            &mut conn,
            REORDER_DOCUMENTS_SQL,
            Some("c1"),
            &["b".into(), "x".into(), "a".into()],
        )
        .unwrap();
        let x_collection: String = conn
            .query_row(
                "SELECT collection_id FROM documents WHERE id = 'x'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(x_collection, "c2");
        // b and a were renumbered by their index in the list (0 and 2); x's slot (1)
        // hit nothing.
        let b_pos: i64 = conn
            .query_row("SELECT position FROM documents WHERE id = 'b'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let a_pos: i64 = conn
            .query_row("SELECT position FROM documents WHERE id = 'a'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!((b_pos, a_pos), (0, 2));
    }

    #[test]
    fn reorder_documents_handles_the_null_root_scope() {
        let mut conn = Connection::open_in_memory().unwrap();
        seed(&conn);
        conn.execute_batch(
            "INSERT INTO documents (id, collection_id, position) VALUES
                ('a', NULL, 0), ('b', NULL, 1);",
        )
        .unwrap();
        reorder_rows(
            &mut conn,
            REORDER_DOCUMENTS_SQL,
            None,
            &["b".into(), "a".into()],
        )
        .unwrap();
        assert_eq!(
            positions(&conn, true),
            vec![("b".into(), 0), ("a".into(), 1)]
        );
    }

    #[test]
    fn reorder_collections_renumbers_under_a_parent() {
        let mut conn = Connection::open_in_memory().unwrap();
        seed(&conn);
        conn.execute_batch(
            "INSERT INTO collections (id, parent_id, position) VALUES
                ('a', 'p', 0), ('b', 'p', 1), ('c', 'p', 2);",
        )
        .unwrap();
        reorder_rows(
            &mut conn,
            REORDER_COLLECTIONS_SQL,
            Some("p"),
            &["b".into(), "c".into(), "a".into()],
        )
        .unwrap();
        let mut stmt = conn
            .prepare("SELECT id FROM collections WHERE parent_id = 'p' ORDER BY position")
            .unwrap();
        let order: Vec<String> = stmt
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(order, vec!["b", "c", "a"]);
    }
}
