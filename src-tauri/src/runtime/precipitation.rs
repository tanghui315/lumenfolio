//! Knowledge precipitation — Stream 1 (index-time, per-document).
//!
//! After a document is indexed, one bounded LLM call distills durable knowledge
//! artifacts (a short summary + entities + concepts + keywords) into
//! `document_artifacts`, seeding the cross-document knowledge graph. The input
//! to the LLM is a SAMPLE (title + abstract + section titles + sampled body),
//! never the whole PDF — see `gather_source_text`. A SHA256 of that sample is
//! cached in `document_knowledge.source_hash` so unchanged documents never call
//! the LLM twice.
//!
//! Stream 2 (conversation-time, near-zero-LLM) lives in `precipitate_turn`.

use rusqlite::{params, OptionalExtension};
use serde::Deserialize;

use crate::{stable_text_hash, AppDatabase, OpenAiCompatibleProvider};

/// Max characters of sampled source text fed to the extractor (hard cap so a
/// long PDF never balloons the prompt).
const SOURCE_CHAR_BUDGET: usize = 9000;
const MAX_ENTITIES: usize = 24;
const MAX_CONCEPTS: usize = 24;
const MAX_KEYWORDS: usize = 16;

#[derive(Debug, Clone)]
pub(crate) struct PrecipitationResult {
    pub document_id: String,
    pub status: String,
    pub summary: String,
    pub entities: usize,
    pub concepts: usize,
    pub keywords: usize,
}

// ---- LLM output shape ------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ExtractionOut {
    #[serde(default)]
    summary: String,
    #[serde(default)]
    entities: Vec<ExtractItem>,
    #[serde(default)]
    concepts: Vec<ExtractItem>,
    #[serde(default)]
    keywords: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExtractItem {
    #[serde(default)]
    name: String,
    #[serde(rename = "type", default)]
    item_type: String,
    #[serde(default)]
    detail: String,
    #[serde(default)]
    salience: f64,
}

// ---- Queue (called inside the index transaction) ---------------------------

/// Mark a document as needing (re)precipitation. Mirrors `queue_visual_index_job`:
/// runs inside the index transaction, does NOT commit. Always sets `pending` so a
/// re-index re-evaluates; the SHA256 guard in `run_precipitation_job` prevents a
/// redundant LLM call when the sampled text is unchanged.
pub fn queue_precipitation_job(
    tx: &rusqlite::Transaction<'_>,
    document_id: &str,
) -> Result<(), String> {
    tx.execute(
        "INSERT INTO document_knowledge (document_id, status, source_hash, artifact_version, summary, error, updated_at)
         VALUES (?1, 'pending', '', 0, '', '', unixepoch())
         ON CONFLICT(document_id) DO UPDATE SET
           status = 'pending',
           error = '',
           updated_at = unixepoch()",
        params![document_id],
    )
    .map_err(|err| format!("Failed to queue precipitation job: {err}"))?;
    Ok(())
}

// ---- Run (async; called from the enqueue command via spawn) ----------------

/// Distill knowledge for one document. DB access happens in short scoped locks
/// (never held across the `.await`). Returns a skipped result (no LLM) when the
/// sampled text is unchanged since the last successful run.
pub async fn run_precipitation_job(
    document_id: &str,
    database: &AppDatabase,
    provider: &OpenAiCompatibleProvider,
) -> Result<PrecipitationResult, String> {
    // 1. Gather sampled input + prior hash/status (scoped lock).
    let (title, source_text, prior_hash, prior_status) = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM documents WHERE id = ?1 LIMIT 1)",
                params![document_id],
                |row| row.get::<_, i64>(0),
            )
            .map(|value| value != 0)
            .map_err(|err| format!("Failed to inspect document: {err}"))?;
        if !exists {
            return Err("Document is no longer available".to_string());
        }
        let title: String = conn
            .query_row(
                "SELECT title FROM documents WHERE id = ?1",
                params![document_id],
                |row| row.get(0),
            )
            .unwrap_or_default();
        let (prior_hash, prior_status): (String, String) = conn
            .query_row(
                "SELECT source_hash, status FROM document_knowledge WHERE document_id = ?1",
                params![document_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|err| format!("Failed to read knowledge state: {err}"))?
            .unwrap_or_default();
        let source_text = gather_source_text(&conn, document_id);
        (title, source_text, prior_hash, prior_status)
    };

    if source_text.trim().is_empty() {
        mark_status(database, document_id, "skipped", "", "")?;
        return Ok(skipped(document_id));
    }

    let hash = stable_text_hash(&format!("{title}\n\n{source_text}"));
    // Unchanged + already done → skip the LLM entirely.
    if hash == prior_hash && prior_status == "done" {
        return Ok(skipped(document_id));
    }

    mark_status(database, document_id, "running", &prior_hash, "")?;

    // 2. One bounded LLM call (the only token cost).
    let user_prompt = build_extraction_prompt(&title, &source_text);
    let raw = match crate::llm::chat::run_simple_completion(
        provider,
        EXTRACTION_SYSTEM,
        &user_prompt,
        0.1,
        60,
    )
    .await
    {
        Ok(text) => text,
        Err(err) => {
            mark_status(database, document_id, "failed", &prior_hash, &err)?;
            return Err(err);
        }
    };

    let parsed = match parse_extraction(&raw) {
        Ok(parsed) => parsed,
        Err(err) => {
            mark_status(database, document_id, "failed", &prior_hash, &err)?;
            return Err(err);
        }
    };

    // 3. Store artifacts + mark done (scoped lock, single transaction).
    let summary = parsed.summary.trim().to_string();
    let (entities, concepts, keywords) = store_artifacts(database, document_id, &parsed, &hash)?;

    Ok(PrecipitationResult {
        document_id: document_id.to_string(),
        status: "done".to_string(),
        summary,
        entities,
        concepts,
        keywords,
    })
}

fn skipped(document_id: &str) -> PrecipitationResult {
    PrecipitationResult {
        document_id: document_id.to_string(),
        status: "done".to_string(),
        summary: String::new(),
        entities: 0,
        concepts: 0,
        keywords: 0,
    }
}

fn mark_status(
    database: &AppDatabase,
    document_id: &str,
    status: &str,
    source_hash: &str,
    error: &str,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "INSERT INTO document_knowledge (document_id, status, source_hash, artifact_version, summary, error, updated_at)
         VALUES (?1, ?2, ?3, 0, '', ?4, unixepoch())
         ON CONFLICT(document_id) DO UPDATE SET
           status = ?2,
           source_hash = ?3,
           error = ?4,
           updated_at = unixepoch()",
        params![document_id, status, source_hash, error],
    )
    .map_err(|err| format!("Failed to update knowledge status: {err}"))?;
    Ok(())
}

fn store_artifacts(
    database: &AppDatabase,
    document_id: &str,
    parsed: &ExtractionOut,
    hash: &str,
) -> Result<(usize, usize, usize), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to open precipitation tx: {err}"))?;

    tx.execute(
        "DELETE FROM document_artifacts WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear old artifacts: {err}"))?;

    let summary = parsed.summary.trim();
    if !summary.is_empty() {
        tx.execute(
            "INSERT INTO document_artifacts (id, document_id, kind, name, normalized, detail, salience, metadata_json, created_at)
             VALUES (?1, ?2, 'summary', '', '', ?3, 1.0, '{}', unixepoch())",
            params![format!("art-{document_id}-summary"), document_id, summary],
        )
        .map_err(|err| format!("Failed to store summary: {err}"))?;
    }

    let mut insert_items = |kind: &str,
                            items: &[ExtractItem],
                            cap: usize|
     -> Result<usize, String> {
        let mut count = 0usize;
        for (index, item) in items.iter().take(cap).enumerate() {
            let name = item.name.trim();
            if name.is_empty() {
                continue;
            }
            let normalized = normalize_key(name);
            let salience = item.salience.clamp(0.0, 1.0);
            let metadata = if item.item_type.trim().is_empty() {
                "{}".to_string()
            } else {
                serde_json::json!({ "type": item.item_type.trim() }).to_string()
            };
            tx.execute(
                "INSERT INTO document_artifacts (id, document_id, kind, name, normalized, detail, salience, metadata_json, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, unixepoch())",
                params![
                    format!("art-{document_id}-{kind}-{index}"),
                    document_id,
                    kind,
                    name,
                    normalized,
                    item.detail.trim(),
                    salience,
                    metadata,
                ],
            )
            .map_err(|err| format!("Failed to store {kind}: {err}"))?;
            count += 1;
        }
        Ok(count)
    };

    let entities = insert_items("entity", &parsed.entities, MAX_ENTITIES)?;
    let concepts = insert_items("concept", &parsed.concepts, MAX_CONCEPTS)?;

    let mut keyword_count = 0usize;
    let mut keyword_names: Vec<String> = Vec::new();
    for (index, keyword) in parsed.keywords.iter().take(MAX_KEYWORDS).enumerate() {
        let keyword = keyword.trim();
        if keyword.is_empty() {
            continue;
        }
        tx.execute(
            "INSERT INTO document_artifacts (id, document_id, kind, name, normalized, detail, salience, metadata_json, created_at)
             VALUES (?1, ?2, 'keyword', ?3, ?4, '', 0.0, '{}', unixepoch())",
            params![
                format!("art-{document_id}-keyword-{index}"),
                document_id,
                keyword,
                normalize_key(keyword),
            ],
        )
        .map_err(|err| format!("Failed to store keyword: {err}"))?;
        keyword_names.push(keyword.to_string());
        keyword_count += 1;
    }

    // Back-fill the document's keywords onto its structure-tree ROOT node so the
    // RAG inspect_tree keyword match can use them (doc-level; sections unchanged).
    if !keyword_names.is_empty() {
        if let Ok(keywords_json) = serde_json::to_string(&keyword_names) {
            let _ = tx.execute(
                "UPDATE structure_tree_nodes SET keywords_json = ?2, updated_at = unixepoch()
                 WHERE document_id = ?1 AND level = 0",
                params![document_id, keywords_json],
            );
        }
    }

    tx.execute(
        "INSERT INTO document_knowledge (document_id, status, source_hash, artifact_version, summary, error, updated_at)
         VALUES (?1, 'done', ?2, 1, ?3, '', unixepoch())
         ON CONFLICT(document_id) DO UPDATE SET
           status = 'done',
           source_hash = ?2,
           artifact_version = document_knowledge.artifact_version + 1,
           summary = ?3,
           error = '',
           updated_at = unixepoch()",
        params![document_id, hash, summary],
    )
    .map_err(|err| format!("Failed to finalize knowledge: {err}"))?;

    tx.commit()
        .map_err(|err| format!("Failed to commit precipitation: {err}"))?;
    Ok((entities, concepts, keyword_count))
}

// ---- Source sampling -------------------------------------------------------

/// Build the sampled extractor input: abstract (if any) + section titles +
/// in-order body blocks until the char budget is hit. Never the whole PDF.
fn gather_source_text(conn: &rusqlite::Connection, document_id: &str) -> String {
    let mut out = String::new();

    // Abstract block(s).
    if let Ok(mut stmt) = conn.prepare(
        "SELECT text FROM document_blocks
         WHERE document_id = ?1 AND block_role = 'abstract'
         ORDER BY page_no, block_index",
    ) {
        let rows = stmt
            .query_map(params![document_id], |row| row.get::<_, String>(0))
            .map(|iter| iter.filter_map(Result::ok).collect::<Vec<_>>())
            .unwrap_or_default();
        if !rows.is_empty() {
            out.push_str("ABSTRACT:\n");
            out.push_str(&rows.join("\n"));
            out.push_str("\n\n");
        }
    }

    // Section titles (the document's outline).
    if let Ok(mut stmt) = conn.prepare(
        "SELECT title FROM structure_tree_nodes
         WHERE document_id = ?1 AND level >= 1
         ORDER BY order_index",
    ) {
        let titles = stmt
            .query_map(params![document_id], |row| row.get::<_, String>(0))
            .map(|iter| {
                iter.filter_map(Result::ok)
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !titles.is_empty() {
            out.push_str("SECTIONS:\n");
            out.push_str(&titles.join(" / "));
            out.push_str("\n\n");
        }
    }

    // In-order body blocks until the budget is reached.
    if out.len() < SOURCE_CHAR_BUDGET {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT text FROM document_blocks
             WHERE document_id = ?1 AND block_role NOT IN ('abstract')
             ORDER BY page_no, block_index",
        ) {
            if let Ok(iter) = stmt.query_map(params![document_id], |row| row.get::<_, String>(0)) {
                out.push_str("BODY:\n");
                for text in iter.filter_map(Result::ok) {
                    let text = text.trim();
                    if text.is_empty() {
                        continue;
                    }
                    if out.len() + text.len() > SOURCE_CHAR_BUDGET {
                        // Append a char-boundary-safe prefix of this block, then stop.
                        let remaining = SOURCE_CHAR_BUDGET.saturating_sub(out.len());
                        let mut prefix_end = 0;
                        for (i, c) in text.char_indices() {
                            let next = i + c.len_utf8();
                            if next > remaining {
                                break;
                            }
                            prefix_end = next;
                        }
                        out.push_str(&text[..prefix_end]);
                        break;
                    }
                    out.push_str(text);
                    out.push('\n');
                }
            }
        }
    }

    out.trim().to_string()
}

// ---- Prompt + parsing ------------------------------------------------------

const EXTRACTION_SYSTEM: &str = "You are a research-paper knowledge extractor. Read the (sampled) document text and return ONE JSON object only — no prose, no markdown fences. Use the document's own language for names where natural, but keep canonical full names for methods/datasets/systems so they match across papers.";

fn build_extraction_prompt(title: &str, source_text: &str) -> String {
    format!(
        "Title: {title}\n\n{source_text}\n\n\
Return ONLY this JSON object:\n\
{{\n\
  \"summary\": \"2-4 sentences: what this paper is about, its core contribution/method/result\",\n\
  \"entities\": [{{\"name\": \"\", \"type\": \"method|model|system|dataset|metric|organization|other\", \"detail\": \"one short clause\", \"salience\": 0.0}}],\n\
  \"concepts\": [{{\"name\": \"\", \"detail\": \"one short clause\", \"salience\": 0.0}}],\n\
  \"keywords\": [\"\"]\n\
}}\n\
Rules: salience in 0..1. Prefer canonical full names. entities<=24, concepts<=24, keywords<=16. Output JSON only."
    )
}

/// Tolerant JSON parse: models sometimes wrap output in ```json fences or add a
/// stray sentence. Extract the outermost {...} and parse that.
fn parse_extraction(raw: &str) -> Result<ExtractionOut, String> {
    let trimmed = raw.trim();
    let start = trimmed.find('{');
    let end = trimmed.rfind('}');
    let slice = match (start, end) {
        (Some(s), Some(e)) if e > s => &trimmed[s..=e],
        _ => return Err("Extractor returned no JSON object".to_string()),
    };
    serde_json::from_str::<ExtractionOut>(slice)
        .map_err(|err| format!("Failed to parse extraction JSON: {err}"))
}

/// Normalize a name for cross-document matching (alias collapsing). Deterministic,
/// no NLP/embeddings: lowercases, maps separators to spaces, strips edge
/// punctuation, and conservatively singularizes each token so trivial variants
/// ("Denitrifying Bacteria" / "denitrifying-bacterium"? no — only plural→singular)
/// collapse to one key.
///
/// LIMIT: this is a SURFACE alias merge. True semantic consolidation (e.g.
/// "LSTM" ⇄ "Long Short-Term Memory", synonyms) needs embeddings, which the app
/// does not have yet — out of scope.
pub(crate) fn normalize_key(name: &str) -> String {
    let lowered = name.trim().to_lowercase();
    lowered
        .split(|c: char| c.is_whitespace() || c == '-' || c == '_' || c == '/')
        .map(|tok| tok.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|tok| !tok.is_empty())
        .map(singularize_token)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Conservative English singularization (only common, safe cases).
fn singularize_token(token: &str) -> String {
    let len = token.chars().count();
    if len <= 3 {
        return token.to_string();
    }
    if let Some(stem) = token.strip_suffix("ies") {
        if stem.chars().count() >= 2 {
            return format!("{stem}y");
        }
    }
    for suffix in ["sses", "ches", "shes", "xes", "zes"] {
        if let Some(stem) = token.strip_suffix(suffix) {
            // e.g. "classes" -> "class", "boxes" -> "box"
            return format!("{stem}{}", &suffix[..suffix.len() - 2]);
        }
    }
    if token.ends_with("ss")
        || token.ends_with("us")
        || token.ends_with("is")
        || token.ends_with("os")
    {
        return token.to_string();
    }
    if let Some(stem) = token.strip_suffix('s') {
        // Require a reasonably long stem so short words ending in 's' (bias, lens,
        // news, …) are NOT mangled. Under-merging is safer than mangling.
        if stem.chars().count() >= 4 {
            return stem.to_string();
        }
    }
    token.to_string()
}

// ---- Stream 2: conversation-time precipitation (zero extra LLM) ------------

/// Distill one completed chat turn into durable knowledge: write `claim` rows and
/// `co_citation` doc<->doc edges among the documents the answer actually drew on.
/// Reuses data already produced by the turn (claims + citations) — no LLM call.
/// Idempotent per `turn_id`. Runs inside the caller's existing connection lock.
pub fn precipitate_turn(
    conn: &rusqlite::Connection,
    turn_id: &str,
    session_id: &str,
    focus_document_id: &str,
    question: &str,
    claim_texts: &[&str],
    cited_document_ids: &[&str],
    citations_json: &str,
) -> Result<(), String> {
    let turn_id = turn_id.trim();
    if turn_id.is_empty() {
        return Ok(());
    }
    let focus_document_id = focus_document_id.trim();
    // Library-wide (no-focus) turns have no focus document. Anchor the claims on the
    // first cited document so cross-library answers still enrich the graph instead of
    // being silently dropped. (knowledge_claims.document_id is NOT NULL + FK, so it
    // must reference a real document — an empty id would violate the FK.)
    let anchor_document_id = if focus_document_id.is_empty() {
        cited_document_ids
            .iter()
            .map(|id| id.trim())
            .find(|id| !id.is_empty())
            .unwrap_or("")
    } else {
        focus_document_id
    };
    if anchor_document_id.is_empty() {
        return Ok(());
    }
    // Idempotent: a turn already precipitated is not re-counted (would inflate edges).
    let already: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM knowledge_claims WHERE turn_id = ?1)",
            params![turn_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value != 0)
        .unwrap_or(false);
    if already {
        return Ok(());
    }

    // Distinct document set: focus + every cited document (the real cross-doc reach).
    let mut docs: Vec<String> = Vec::new();
    let mut push = |id: &str| {
        let id = id.trim();
        if !id.is_empty() && !docs.iter().any(|d| d == id) {
            docs.push(id.to_string());
        }
    };
    push(anchor_document_id);
    for id in cited_document_ids {
        push(id);
    }
    let doc_ids_json = serde_json::to_string(&docs).unwrap_or_else(|_| "[]".to_string());

    // 1. Claims → knowledge_claims (one row per claim).
    for (index, text) in claim_texts.iter().enumerate() {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        conn.execute(
            "INSERT INTO knowledge_claims
                (id, document_id, turn_id, session_id, claim, question, doc_ids_json, citations_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, unixepoch())",
            params![
                format!("kc-{turn_id}-{index}"),
                anchor_document_id,
                turn_id,
                session_id,
                text,
                question,
                doc_ids_json,
                citations_json,
            ],
        )
        .map_err(|err| format!("Failed to store knowledge claim: {err}"))?;
    }

    // 2. co_citation edges among the document set (weight accumulates per turn).
    if docs.len() >= 2 {
        let evidence = serde_json::json!({ "lastQuestion": question }).to_string();
        for i in 0..docs.len() {
            for j in (i + 1)..docs.len() {
                let (mut a, mut b) = (docs[i].as_str(), docs[j].as_str());
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                conn.execute(
                    "INSERT INTO document_links (id, doc_a, doc_b, basis, weight, evidence_json, updated_at)
                     VALUES (?1, ?2, ?3, 'co_citation', 1, ?4, unixepoch())
                     ON CONFLICT(doc_a, doc_b, basis) DO UPDATE SET
                       weight = document_links.weight + 1,
                       evidence_json = ?4,
                       updated_at = unixepoch()",
                    params![format!("dl-cocite-{a}-{b}"), a, b, evidence],
                )
                .map_err(|err| format!("Failed to store co-citation edge: {err}"))?;
            }
        }
    }
    Ok(())
}

// ---- Tauri commands --------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct KnowledgeEventOutput {
    pub document_id: String,
    pub status: String,
    pub summary: String,
    pub entities: usize,
    pub concepts: usize,
    pub keywords: usize,
    pub error: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactOut {
    name: String,
    detail: String,
    salience: f64,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct KnowledgeCard {
    status: String,
    summary: String,
    entities: Vec<ArtifactOut>,
    concepts: Vec<ArtifactOut>,
    keywords: Vec<String>,
    error: String,
}

/// Trigger Stream-1 precipitation for a document. Resolves the chat provider
/// (default when none selected), then runs the job off the request via spawn,
/// emitting `lumenfolio://knowledge` on completion. Returns immediately ("queued").
#[tauri::command]
pub(crate) async fn enqueue_document_knowledge(
    document_id: String,
    model_provider_id: Option<String>,
    model_key: Option<String>,
    app: tauri::AppHandle,
) -> Result<KnowledgeEventOutput, String> {
    use tauri::{Emitter, Manager};
    let document_id = document_id.trim().to_string();
    if document_id.is_empty() {
        return Err("No document for knowledge precipitation".to_string());
    }
    let provider = {
        let database = app.state::<AppDatabase>();
        let (provider, _model_key) = crate::providers::resolve_chat_provider(
            &database,
            model_provider_id.as_deref().unwrap_or("").trim(),
            model_key.as_deref(),
        )?;
        provider
    };

    let task_app = app.clone();
    let doc = document_id.clone();
    tauri::async_runtime::spawn(async move {
        let result = {
            let database = task_app.state::<AppDatabase>();
            run_precipitation_job(&doc, &database, &provider).await
        };
        let event = match result {
            Ok(r) => KnowledgeEventOutput {
                document_id: r.document_id,
                status: r.status,
                summary: r.summary,
                entities: r.entities,
                concepts: r.concepts,
                keywords: r.keywords,
                error: String::new(),
            },
            Err(err) => KnowledgeEventOutput {
                document_id: doc.clone(),
                status: "failed".to_string(),
                summary: String::new(),
                entities: 0,
                concepts: 0,
                keywords: 0,
                error: err,
            },
        };
        let _ = task_app.emit("lumenfolio://knowledge", &event);
    });

    Ok(KnowledgeEventOutput {
        document_id,
        status: "queued".to_string(),
        summary: String::new(),
        entities: 0,
        concepts: 0,
        keywords: 0,
        error: String::new(),
    })
}

/// Read the precipitated knowledge card for a document (status + summary +
/// entity/concept/keyword artifacts).
#[tauri::command]
pub(crate) fn get_document_knowledge(
    document_id: String,
    database: tauri::State<'_, AppDatabase>,
) -> Result<KnowledgeCard, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let (status, summary, error) = conn
        .query_row(
            "SELECT status, summary, error FROM document_knowledge WHERE document_id = ?1",
            params![document_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|err| format!("Failed to read knowledge state: {err}"))?
        .unwrap_or_else(|| ("idle".to_string(), String::new(), String::new()));

    let load_items = |kind: &str| -> Result<Vec<ArtifactOut>, String> {
        let mut stmt = conn
            .prepare(
                "SELECT name, detail, salience FROM document_artifacts
                 WHERE document_id = ?1 AND kind = ?2
                 ORDER BY salience DESC, name",
            )
            .map_err(|err| format!("Failed to prepare artifact query: {err}"))?;
        let items = stmt
            .query_map(params![document_id, kind], |row| {
                Ok(ArtifactOut {
                    name: row.get(0)?,
                    detail: row.get(1)?,
                    salience: row.get(2)?,
                })
            })
            .map_err(|err| format!("Failed to query artifacts: {err}"))?
            .filter_map(Result::ok)
            .collect();
        Ok(items)
    };

    let entities = load_items("entity")?;
    let concepts = load_items("concept")?;
    let keywords = {
        let mut stmt = conn
            .prepare(
                "SELECT name FROM document_artifacts
                 WHERE document_id = ?1 AND kind = 'keyword' ORDER BY name",
            )
            .map_err(|err| format!("Failed to prepare keyword query: {err}"))?;
        let list: Vec<String> = stmt
            .query_map(params![document_id], |row| row.get::<_, String>(0))
            .map_err(|err| format!("Failed to query keywords: {err}"))?
            .filter_map(Result::ok)
            .collect();
        list
    };

    Ok(KnowledgeCard {
        status,
        summary,
        entities,
        concepts,
        keywords,
        error,
    })
}

/// Force a re-run: clears the cached source hash so the next job won't skip,
/// then enqueues precipitation.
#[tauri::command]
pub(crate) async fn reprecipitate_document(
    document_id: String,
    model_provider_id: Option<String>,
    model_key: Option<String>,
    app: tauri::AppHandle,
) -> Result<KnowledgeEventOutput, String> {
    use tauri::Manager;
    let document_id = document_id.trim().to_string();
    if document_id.is_empty() {
        return Err("No document for knowledge precipitation".to_string());
    }
    {
        let database = app.state::<AppDatabase>();
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        conn.execute(
            "UPDATE document_knowledge SET source_hash = '', status = 'pending', updated_at = unixepoch()
             WHERE document_id = ?1",
            params![document_id],
        )
        .map_err(|err| format!("Failed to reset knowledge cache: {err}"))?;
    }
    enqueue_document_knowledge(document_id, model_provider_id, model_key, app).await
}

/// Consolidate existing knowledge: re-apply the (improved) alias normalization to
/// every stored artifact in place, so variants collapse to one key across the
/// whole library — no LLM, instant. Returns the number of rows updated.
#[tauri::command]
pub(crate) fn consolidate_knowledge(
    database: tauri::State<'_, AppDatabase>,
) -> Result<usize, String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to open consolidate tx: {err}"))?;
    let rows: Vec<(String, String, String)> = {
        let mut stmt = tx
            .prepare("SELECT id, name, normalized FROM document_artifacts WHERE name <> ''")
            .map_err(|err| format!("Failed to prepare consolidate query: {err}"))?;
        let collected = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|err| format!("Failed to read artifacts: {err}"))?
            .filter_map(Result::ok)
            .collect();
        collected
    };
    let mut updated = 0usize;
    for (id, name, normalized) in rows {
        let next = normalize_key(&name);
        if next != normalized {
            tx.execute(
                "UPDATE document_artifacts SET normalized = ?2 WHERE id = ?1",
                params![id, next],
            )
            .map_err(|err| format!("Failed to update artifact: {err}"))?;
            updated += 1;
        }
    }
    tx.commit()
        .map_err(|err| format!("Failed to commit consolidate: {err}"))?;
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_with_fences_and_prose() {
        let raw = "Sure, here:\n```json\n{\"summary\":\"S\",\"entities\":[{\"name\":\"LSTM\",\"type\":\"model\",\"salience\":0.9}],\"concepts\":[],\"keywords\":[\"a\",\"b\"]}\n```";
        let parsed = parse_extraction(raw).expect("should parse");
        assert_eq!(parsed.summary, "S");
        assert_eq!(parsed.entities.len(), 1);
        assert_eq!(parsed.entities[0].name, "LSTM");
        assert_eq!(parsed.keywords, vec!["a", "b"]);
    }

    #[test]
    fn normalize_key_lowercases_and_collapses() {
        assert_eq!(
            normalize_key("  Denitrifying   Bacteria "),
            "denitrifying bacteria"
        );
    }

    #[test]
    fn normalize_key_merges_aliases() {
        // hyphen/underscore separators, plurals, edge punctuation all collapse.
        assert_eq!(
            normalize_key("Neural-Networks"),
            normalize_key("neural networks")
        );
        assert_eq!(normalize_key("Models"), "model");
        assert_eq!(normalize_key("Batches"), "batch");
        assert_eq!(normalize_key("Strategies"), "strategy");
        // safe: short tokens and -ss/-us/-is unchanged.
        assert_eq!(normalize_key("bias"), "bias");
        assert_eq!(normalize_key("class"), "class");
    }

    #[test]
    fn parse_extraction_rejects_non_json() {
        assert!(parse_extraction("no json here").is_err());
    }
}
