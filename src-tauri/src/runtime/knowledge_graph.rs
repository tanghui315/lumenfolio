//! Cross-document knowledge graph (P2) — read-side aggregation over the
//! precipitated artifacts (Stream 1) and conversation edges (Stream 2).
//!
//! `get_knowledge_graph` returns the raw material (documents + entity/concept
//! artifacts + doc<->doc links); the frontend builds nodes/edges, relevance and
//! communities (mirrors llm_wiki's front-end graph construction).
//!
//! `get_related_documents` is the reading-time "related papers" query: rank other
//! documents by shared concepts/entities + co-citation, with the shared concept
//! names as the human-readable reason.

use rusqlite::params;
use serde::Serialize;

use crate::AppDatabase;

// Relevance weights (adapted from llm_wiki's graph-relevance.ts, tuned for our
// concept-overlap domain). co_citation is the strongest signal (a real reasoning
// link); Adamic-Adar rewards RARE shared concepts; raw overlap is the baseline.
const W_OVERLAP: f64 = 1.0; // per shared concept/entity (sourceOverlap-like)
const W_ADAMIC_ADAR: f64 = 2.0; // rare shared concepts weigh more (common-neighbor)
const W_CO_CITATION: f64 = 3.0; // conversation co-citation (directLink-like)
const ENTITY_AFFINITY: f64 = 1.3; // a shared entity (method/dataset) > a shared generic concept
const RELATED_LIMIT: usize = 12;
const MAX_SHARED_NAMES: usize = 6;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GraphDocument {
    id: String,
    title: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GraphArtifact {
    document_id: String,
    kind: String,
    name: String,
    normalized: String,
    salience: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GraphLink {
    doc_a: String,
    doc_b: String,
    basis: String,
    weight: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct KnowledgeGraph {
    documents: Vec<GraphDocument>,
    artifacts: Vec<GraphArtifact>,
    links: Vec<GraphLink>,
}

/// Whole-library graph material. Only documents that carry knowledge artifacts
/// are included (a doc with no precipitated concepts has nothing to connect on).
#[tauri::command]
pub(crate) fn get_knowledge_graph(
    database: tauri::State<'_, AppDatabase>,
) -> Result<KnowledgeGraph, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;

    // Documents that have at least one entity/concept artifact.
    let documents = {
        let mut stmt = conn
            .prepare(
                "SELECT d.id, d.title FROM documents d
                 WHERE EXISTS (
                   SELECT 1 FROM document_artifacts a
                   WHERE a.document_id = d.id AND a.kind IN ('entity','concept')
                 )
                 ORDER BY d.title",
            )
            .map_err(|err| format!("Failed to prepare documents query: {err}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(GraphDocument {
                    id: row.get(0)?,
                    title: row.get(1)?,
                })
            })
            .map_err(|err| format!("Failed to query documents: {err}"))?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        rows
    };

    let artifacts = {
        let mut stmt = conn
            .prepare(
                "SELECT document_id, kind, name, normalized, salience
                 FROM document_artifacts
                 WHERE kind IN ('entity','concept') AND normalized != ''
                 ORDER BY salience DESC",
            )
            .map_err(|err| format!("Failed to prepare artifacts query: {err}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(GraphArtifact {
                    document_id: row.get(0)?,
                    kind: row.get(1)?,
                    name: row.get(2)?,
                    normalized: row.get(3)?,
                    salience: row.get(4)?,
                })
            })
            .map_err(|err| format!("Failed to query artifacts: {err}"))?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        rows
    };

    let links = {
        let mut stmt = conn
            .prepare(
                "SELECT doc_a, doc_b, basis, weight FROM document_links
                 ORDER BY weight DESC",
            )
            .map_err(|err| format!("Failed to prepare links query: {err}"))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(GraphLink {
                    doc_a: row.get(0)?,
                    doc_b: row.get(1)?,
                    basis: row.get(2)?,
                    weight: row.get(3)?,
                })
            })
            .map_err(|err| format!("Failed to query links: {err}"))?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        rows
    };

    Ok(KnowledgeGraph {
        documents,
        artifacts,
        links,
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RelatedDocument {
    pub document_id: String,
    pub title: String,
    pub score: f64,
    pub shared_concepts: Vec<String>,
    pub co_citation: f64,
}

/// Edge between two RELATED documents (not involving the focus document — those
/// relations are already carried by the `RelatedDocument` entries themselves).
/// Lets the reading-time mini ego-graph show cluster structure among neighbours.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RelatedLink {
    pub doc_a: String,
    pub doc_b: String,
    pub shared_count: i64,
    pub co_citation: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RelatedDocumentsPayload {
    pub related: Vec<RelatedDocument>,
    pub links: Vec<RelatedLink>,
}

/// Documents related to `document_id`, ranked by shared concepts/entities +
/// co-citation, plus the inter-links among those related documents. The shared
/// concept names are the "why related" reason shown in the reading-time panel.
#[tauri::command]
pub(crate) fn get_related_documents(
    document_id: String,
    database: tauri::State<'_, AppDatabase>,
) -> Result<RelatedDocumentsPayload, String> {
    let document_id = document_id.trim().to_string();
    if document_id.is_empty() {
        return Ok(RelatedDocumentsPayload {
            related: Vec::new(),
            links: Vec::new(),
        });
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let related = related_documents(&conn, &document_id)?;
    let ids: Vec<&str> = related.iter().map(|r| r.document_id.as_str()).collect();
    let links = related_inter_links(&conn, &ids)?;
    Ok(RelatedDocumentsPayload { related, links })
}

/// Inter-links among a small id set (the related list, ≤ RELATED_LIMIT docs):
/// shared concept/entity counts + co-citation weight per unordered pair.
fn related_inter_links(
    conn: &rusqlite::Connection,
    ids: &[&str],
) -> Result<Vec<RelatedLink>, String> {
    use std::collections::HashMap;
    if ids.len() < 2 {
        return Ok(Vec::new());
    }
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let id_params = rusqlite::params_from_iter(ids.iter().chain(ids.iter()));

    let mut pairs: HashMap<(String, String), (i64, f64)> = HashMap::new();

    {
        let sql = format!(
            "SELECT a.document_id, b.document_id, COUNT(DISTINCT a.normalized)
             FROM document_artifacts a
             JOIN document_artifacts b
               ON a.normalized = b.normalized
              AND a.kind = b.kind
              AND a.document_id < b.document_id
             WHERE a.kind IN ('entity','concept') AND a.normalized <> ''
               AND a.document_id IN ({placeholders})
               AND b.document_id IN ({placeholders})
             GROUP BY a.document_id, b.document_id"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|err| format!("Failed to prepare inter-links query: {err}"))?;
        let rows = stmt
            .query_map(id_params, |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(|err| format!("Failed to query inter-links: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (a, b, count) = row;
            pairs.entry((a, b)).or_insert((0, 0.0)).0 = count;
        }
    }

    {
        // document_links rows already store doc_a < doc_b, matching the pair key.
        let sql = format!(
            "SELECT doc_a, doc_b, weight FROM document_links
             WHERE basis = 'co_citation'
               AND doc_a IN ({placeholders})
               AND doc_b IN ({placeholders})"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|err| format!("Failed to prepare inter co-citation query: {err}"))?;
        let id_params = rusqlite::params_from_iter(ids.iter().chain(ids.iter()));
        let rows = stmt
            .query_map(id_params, |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .map_err(|err| format!("Failed to query inter co-citation: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (a, b, weight) = row;
            pairs.entry((a, b)).or_insert((0, 0.0)).1 += weight;
        }
    }

    Ok(pairs
        .into_iter()
        .map(
            |((doc_a, doc_b), (shared_count, co_citation))| RelatedLink {
                doc_a,
                doc_b,
                shared_count,
                co_citation,
            },
        )
        .collect())
}

// ---- Post-answer recommendations (turn-contextual) -------------------------

const TURN_RELATED_LIMIT: usize = 5;
const TURN_CLAIMS_LIMIT: usize = 4;
// A recommendation block auto-expands in the chat only when the top neighbour
// shares at least this many concepts; otherwise it stays a quiet affordance.
const CONFIDENT_MIN_SHARED: usize = 2;
// Recent claims scanned when matching prior knowledge to this turn's concepts.
const CLAIMS_SCAN_LIMIT: usize = 200;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RelatedClaim {
    pub document_id: String,
    pub title: String,
    pub claim: String,
    pub question: String,
    pub shared_concepts: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TurnRecommendationsPayload {
    pub related: Vec<RelatedDocument>,
    pub claims: Vec<RelatedClaim>,
    pub confident: bool,
}

/// Whether `needle` (already lowercased) occurs in `haystack` as a concept
/// mention. For ASCII needles we require word boundaries so a short name like
/// "ai"/"gan" doesn't match inside "maintain"/"organ"; CJK and other scripts have
/// no word separators, so there we fall back to plain substring containment.
fn concept_mentioned(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let word_like = needle
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b' ' || b == b'-');
    if !word_like {
        return haystack.contains(needle);
    }
    let bytes = haystack.as_bytes();
    let nlen = needle.len();
    let mut from = 0;
    while let Some(rel) = haystack[from..].find(needle) {
        let start = from + rel;
        let end = start + nlen;
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        from = start + 1; // needle is ASCII here, so +1 lands on a char boundary
    }
    false
}

/// Recommendations shown after a completed chat turn: other library papers and
/// prior knowledge (precipitated claims) relevant to *what this turn was about*.
///
/// Unlike `get_related_documents` (keyed on the whole focus document), the seed
/// here is the subset of the focus doc's concepts/entities actually mentioned in
/// this turn's question + distilled claims — so the picks track the question, not
/// the paper. Pure SQL over already-precipitated data: no LLM call (honouring the
/// zero-LLM conversation-time contract).
#[tauri::command]
pub(crate) fn get_turn_recommendations(
    turn_id: String,
    database: tauri::State<'_, AppDatabase>,
) -> Result<TurnRecommendationsPayload, String> {
    use std::collections::{HashMap, HashSet};
    let turn_id = turn_id.trim().to_string();
    let empty = TurnRecommendationsPayload {
        related: Vec::new(),
        claims: Vec::new(),
        confident: false,
    };
    if turn_id.is_empty() {
        return Ok(empty);
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;

    // 1. The turn's focus doc + the text it was about (question + every claim).
    let mut focus_document_id = String::new();
    let mut text_parts: Vec<String> = Vec::new();
    {
        let mut stmt = conn
            .prepare("SELECT document_id, question, claim FROM knowledge_claims WHERE turn_id = ?1")
            .map_err(|err| format!("Failed to prepare turn claims query: {err}"))?;
        let rows = stmt
            .query_map(params![turn_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|err| format!("Failed to query turn claims: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (doc, question, claim) = row;
            if focus_document_id.is_empty() {
                focus_document_id = doc;
            }
            if !question.trim().is_empty() {
                text_parts.push(question);
            }
            text_parts.push(claim);
        }
    }
    // No precipitated claims for this turn (e.g. knowledge disabled, or the answer
    // produced none) → nothing turn-specific to recommend on.
    if focus_document_id.is_empty() {
        return Ok(empty);
    }

    // 2. Seed concepts: focus-doc concepts/entities whose name appears in the
    //    turn text, falling back to the doc's top-salience concepts.
    let haystack = text_parts.join(" \n ").to_lowercase();
    let mut seed: HashMap<String, String> = HashMap::new(); // normalized -> display name
    let mut fallback: Vec<(String, String)> = Vec::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT normalized, name FROM document_artifacts
                 WHERE document_id = ?1 AND kind IN ('entity','concept') AND normalized <> ''
                 ORDER BY salience DESC",
            )
            .map_err(|err| format!("Failed to prepare seed query: {err}"))?;
        let rows = stmt
            .query_map(params![focus_document_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|err| format!("Failed to query seed concepts: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (normalized, name) = row;
            let needle = name.trim().to_lowercase();
            if needle.len() >= 2 && concept_mentioned(&haystack, &needle) {
                seed.entry(normalized.clone())
                    .or_insert_with(|| name.clone());
            }
            if fallback.len() < 8 {
                fallback.push((normalized, name));
            }
        }
    }
    if seed.is_empty() {
        for (normalized, name) in fallback {
            seed.entry(normalized).or_insert(name);
        }
    }
    if seed.is_empty() {
        return Ok(empty);
    }
    let seed_keys: Vec<String> = seed.keys().cloned().collect();

    // 3. Document frequency per concept (Adamic-Adar rarity). Scoped to the seed
    //    keys (only their df is read) so this stays cheap on the chat hot path
    //    instead of grouping the whole artifact table every answer.
    let doc_freq: HashMap<String, i64> = {
        let placeholders = seed_keys.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT normalized, COUNT(DISTINCT document_id)
             FROM document_artifacts
             WHERE kind IN ('entity','concept') AND normalized IN ({placeholders})
             GROUP BY normalized"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|err| format!("Failed to prepare doc-frequency query: {err}"))?;
        let bind = rusqlite::params_from_iter(seed_keys.iter().cloned());
        let map = stmt
            .query_map(bind, |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|err| format!("Failed to query doc frequency: {err}"))?
            .filter_map(Result::ok)
            .collect();
        map
    };

    // 4. Rank OTHER documents by how many seed concepts they share.
    struct Acc {
        title: String,
        seen: HashSet<String>,
        names: Vec<String>,
        overlap: f64,
        adamic_adar: f64,
    }
    let mut acc: HashMap<String, Acc> = HashMap::new();
    {
        let placeholders = seed_keys.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT a.document_id, d.title, a.name, a.normalized, a.kind
             FROM document_artifacts a
             JOIN documents d ON d.id = a.document_id
             WHERE a.document_id <> ?1
               AND a.kind IN ('entity','concept')
               AND a.normalized IN ({placeholders})"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|err| format!("Failed to prepare turn-related query: {err}"))?;
        let bind = rusqlite::params_from_iter(
            std::iter::once(focus_document_id.clone()).chain(seed_keys.iter().cloned()),
        );
        let rows = stmt
            .query_map(bind, |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(|err| format!("Failed to query turn-related: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (other_id, title, name, normalized, kind) = row;
            let entry = acc.entry(other_id).or_insert_with(|| Acc {
                title,
                seen: HashSet::new(),
                names: Vec::new(),
                overlap: 0.0,
                adamic_adar: 0.0,
            });
            if !entry.seen.insert(normalized.clone()) {
                continue;
            }
            let affinity = if kind == "entity" {
                ENTITY_AFFINITY
            } else {
                1.0
            };
            entry.overlap += affinity;
            let df = (*doc_freq.get(&normalized).unwrap_or(&2)).max(2) as f64;
            entry.adamic_adar += affinity / df.ln();
            if entry.names.len() < MAX_SHARED_NAMES {
                entry.names.push(name);
            }
        }
    }

    let mut related: Vec<RelatedDocument> = acc
        .into_iter()
        .map(|(document_id, item)| RelatedDocument {
            document_id,
            title: item.title,
            score: item.overlap * W_OVERLAP + item.adamic_adar * W_ADAMIC_ADAR,
            shared_concepts: item.names,
            co_citation: 0.0,
        })
        .filter(|item| item.score > 0.0)
        .collect();
    related.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.title.cmp(&b.title))
    });
    related.truncate(TURN_RELATED_LIMIT);

    // 5. Prior knowledge: recent claims from OTHER turns whose text mentions a
    //    seed concept. The matched concept names are the "why" shown to the user.
    let seed_needles: Vec<(String, String)> = seed
        .iter()
        .map(|(_, name)| (name.trim().to_lowercase(), name.clone()))
        .filter(|(needle, _)| needle.len() >= 2)
        .collect();
    let mut claims: Vec<RelatedClaim> = Vec::new();
    if !seed_needles.is_empty() {
        let mut seen_claims: HashSet<String> = HashSet::new();
        let mut stmt = conn
            .prepare(
                "SELECT kc.document_id, d.title, kc.claim, kc.question
                 FROM knowledge_claims kc
                 JOIN documents d ON d.id = kc.document_id
                 WHERE kc.turn_id <> ?1 AND kc.document_id <> ?2
                 ORDER BY kc.created_at DESC
                 LIMIT ?3",
            )
            .map_err(|err| format!("Failed to prepare related-claims query: {err}"))?;
        let rows = stmt
            .query_map(
                params![turn_id, focus_document_id, CLAIMS_SCAN_LIMIT as i64],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .map_err(|err| format!("Failed to query related claims: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (document_id, title, claim, question) = row;
            let claim_lc = claim.to_lowercase();
            let mut matched: Vec<String> = Vec::new();
            for (needle, name) in &seed_needles {
                if concept_mentioned(&claim_lc, needle) && !matched.iter().any(|m| m == name) {
                    matched.push(name.clone());
                }
            }
            if matched.is_empty() {
                continue;
            }
            let dedup_key = claim_lc.clone();
            if !seen_claims.insert(dedup_key) {
                continue;
            }
            matched.truncate(MAX_SHARED_NAMES);
            claims.push(RelatedClaim {
                document_id,
                title,
                claim,
                question,
                shared_concepts: matched,
            });
            if claims.len() >= TURN_CLAIMS_LIMIT {
                break;
            }
        }
    }

    // Auto-expand only when the top pick shares ≥2 concepts: one shared concept —
    // even a rare one — scores ~3.9 (1·W_OVERLAP + 1/ln(2)·W_ADAMIC_ADAR), so a
    // score gate would auto-expand almost everything and defeat the quiet default.
    let confident = related
        .first()
        .is_some_and(|top| top.shared_concepts.len() >= CONFIDENT_MIN_SHARED);

    Ok(TurnRecommendationsPayload {
        related,
        claims,
        confident,
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LibraryHit {
    pub document_id: String,
    pub title: String,
    pub matched_concepts: Vec<String>,
    pub score: i64,
}

/// Whole-library concept/entity search (NOT scoped to any focus document), used
/// by the `search_library_knowledge` RAG tool. Ranks documents by how many of
/// their precipitated concepts/entities match the query tokens.
pub(crate) fn search_library(
    conn: &rusqlite::Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<LibraryHit>, String> {
    use std::collections::HashMap;
    // Byte length, not char count, so short CJK tokens (≥3 bytes) survive while
    // 1–2 byte ASCII stopwords are dropped.
    let tokens: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(str::to_string)
        .collect();
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    struct Acc {
        title: String,
        names: Vec<String>,
        seen: std::collections::HashSet<String>,
    }
    let mut acc: HashMap<String, Acc> = HashMap::new();

    // One LIKE pass per token; a matched artifact contributes its document + name.
    let mut stmt = conn
        .prepare(
            "SELECT a.document_id, d.title, a.name, a.normalized
             FROM document_artifacts a
             JOIN documents d ON d.id = a.document_id
             WHERE a.kind IN ('entity','concept')
               AND (lower(a.name) LIKE ?1 OR a.normalized LIKE ?1)",
        )
        .map_err(|err| format!("Failed to prepare library search: {err}"))?;
    for token in &tokens {
        let pattern = format!("%{token}%");
        let rows = stmt
            .query_map(params![pattern], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|err| format!("Failed to query library search: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (doc_id, title, name) = row;
            let entry = acc.entry(doc_id).or_insert_with(|| Acc {
                title,
                names: Vec::new(),
                seen: std::collections::HashSet::new(),
            });
            if entry.seen.insert(name.to_lowercase()) && entry.names.len() < 8 {
                entry.names.push(name);
            }
        }
    }

    let mut hits: Vec<LibraryHit> = acc
        .into_iter()
        .map(|(document_id, item)| LibraryHit {
            document_id,
            title: item.title,
            score: item.names.len() as i64,
            matched_concepts: item.names,
        })
        .collect();
    hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.title.cmp(&b.title)));
    hits.truncate(limit);
    Ok(hits)
}

/// Core ranking used by both the command and the `query_knowledge_graph` RAG tool.
/// Caller holds the connection lock.
pub(crate) fn related_documents(
    conn: &rusqlite::Connection,
    document_id: &str,
) -> Result<Vec<RelatedDocument>, String> {
    let document_id = document_id.trim();
    if document_id.is_empty() {
        return Ok(Vec::new());
    }

    use std::collections::{HashMap, HashSet};
    struct Acc {
        title: String,
        seen: HashSet<String>, // normalized keys already counted for this doc
        names: Vec<String>,    // display names of shared concepts
        overlap: f64,          // weighted shared-concept count (entity affinity)
        adamic_adar: f64,      // sum 1/ln(docFreq): rare shared concepts weigh more
        co_citation: f64,
    }
    impl Acc {
        fn new(title: String) -> Self {
            Acc {
                title,
                seen: HashSet::new(),
                names: Vec::new(),
                overlap: 0.0,
                adamic_adar: 0.0,
                co_citation: 0.0,
            }
        }
    }
    let mut acc: HashMap<String, Acc> = HashMap::new();

    // Document frequency per normalized concept/entity (for Adamic-Adar rarity).
    let doc_freq: HashMap<String, i64> = {
        let mut stmt = conn
            .prepare(
                "SELECT normalized, COUNT(DISTINCT document_id)
                 FROM document_artifacts
                 WHERE kind IN ('entity','concept') AND normalized <> ''
                 GROUP BY normalized",
            )
            .map_err(|err| format!("Failed to prepare doc-frequency query: {err}"))?;
        let map = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|err| format!("Failed to query doc frequency: {err}"))?
            .filter_map(Result::ok)
            .collect::<HashMap<_, _>>();
        map
    };

    // Shared concepts/entities: another document is related when it shares a
    // normalized artifact with the focus document. Each shared item contributes
    // an overlap term + an Adamic-Adar term (rarer concept → larger).
    {
        let mut stmt = conn
            .prepare(
                "SELECT b.document_id, d.title, a.name, a.normalized, a.kind
                 FROM document_artifacts a
                 JOIN document_artifacts b
                   ON a.normalized = b.normalized
                  AND a.kind = b.kind
                  AND b.document_id <> a.document_id
                 JOIN documents d ON d.id = b.document_id
                 WHERE a.document_id = ?1
                   AND a.kind IN ('entity','concept')
                   AND a.normalized <> ''",
            )
            .map_err(|err| format!("Failed to prepare related query: {err}"))?;
        let rows = stmt
            .query_map(params![document_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })
            .map_err(|err| format!("Failed to query related: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (other_id, title, name, normalized, kind) = row;
            let entry = acc.entry(other_id).or_insert_with(|| Acc::new(title));
            if !entry.seen.insert(normalized.clone()) {
                continue; // already counted this shared concept for this doc
            }
            let affinity = if kind == "entity" {
                ENTITY_AFFINITY
            } else {
                1.0
            };
            entry.overlap += affinity;
            let df = (*doc_freq.get(&normalized).unwrap_or(&2)).max(2) as f64;
            entry.adamic_adar += affinity / df.ln();
            if entry.names.len() < 64 {
                entry.names.push(name);
            }
        }
    }

    // Co-citation edges that involve the focus document.
    {
        let mut stmt = conn
            .prepare(
                "SELECT CASE WHEN l.doc_a = ?1 THEN l.doc_b ELSE l.doc_a END AS other_id,
                        d.title, l.weight
                 FROM document_links l
                 JOIN documents d
                   ON d.id = CASE WHEN l.doc_a = ?1 THEN l.doc_b ELSE l.doc_a END
                 WHERE l.basis = 'co_citation' AND (l.doc_a = ?1 OR l.doc_b = ?1)",
            )
            .map_err(|err| format!("Failed to prepare co-citation query: {err}"))?;
        let rows = stmt
            .query_map(params![document_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .map_err(|err| format!("Failed to query co-citation: {err}"))?;
        for row in rows.filter_map(Result::ok) {
            let (other_id, title, weight) = row;
            let entry = acc.entry(other_id).or_insert_with(|| Acc::new(title));
            entry.co_citation += weight;
        }
    }

    let mut related: Vec<RelatedDocument> = acc
        .into_iter()
        .map(|(document_id, item)| {
            let score = item.overlap * W_OVERLAP
                + item.adamic_adar * W_ADAMIC_ADAR
                + item.co_citation * W_CO_CITATION;
            let mut shared = item.names;
            shared.truncate(MAX_SHARED_NAMES);
            RelatedDocument {
                document_id,
                title: item.title,
                score,
                shared_concepts: shared,
                co_citation: item.co_citation,
            }
        })
        .filter(|item| item.score > 0.0)
        .collect();

    related.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.title.cmp(&b.title))
    });
    related.truncate(RELATED_LIMIT);
    Ok(related)
}
