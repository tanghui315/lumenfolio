use rusqlite::{params, Connection, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::runtime::agent::lexicon::{leading_reference_number, requested_table_number};

/// A cited piece of evidence.
///
/// Knowledge-base pivot — anchor convention: a citation is anchored either by
/// PAGE (a `page > 0` + `bbox_list` location in a paginated source: PDF, or
/// Office-as-PDF later) or as a REFERENCE (`page == 0`, non-paginated: web clip,
/// note, markdown, trending — located by source/section, not page). The
/// `page == 0` sentinel is authoritative today; P2 adds an explicit stored kind
/// plus precise chunk-id/offset locators for non-paged sources. Use
/// [`Citation::anchor`] instead of testing `page` directly.
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    pub id: String,
    pub label: String,
    pub page: u32,
    pub block_id: String,
    pub section_title: Option<String>,
    pub quote: String,
    pub bbox_list: serde_json::Value,
    pub document_id: String,
    pub source: String,
}

/// How a [`Citation`] is located in its source. See the `Citation` doc comment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CitationAnchor {
    /// A page + bbox location in a paginated source (clickable → highlight).
    Paged,
    /// A non-paginated reference (web/note/markdown/trending), located by
    /// source/section rather than page.
    Reference,
}

impl Citation {
    /// The anchor kind, derived from the `page == 0` convention (see struct doc).
    pub fn anchor(&self) -> CitationAnchor {
        if self.page > 0 {
            CitationAnchor::Paged
        } else {
            CitationAnchor::Reference
        }
    }
}

#[derive(Clone)]
pub struct EvidenceCandidate {
    pub chunk_id: String,
    pub document_id: String,
    pub page: u32,
    pub block_id: String,
    pub section_title: Option<String>,
    pub quote: String,
    pub bbox_list: serde_json::Value,
    pub score: f64,
    pub source: String,
    pub tree_node_id: Option<String>,
    pub block_role: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalTraceTreeNode {
    pub id: String,
    pub title: String,
    pub page: u32,
    pub block_index: u32,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalTraceCandidate {
    pub source: String,
    pub page: u32,
    pub block_id: String,
    pub tree_node_id: Option<String>,
    pub section_title: Option<String>,
    pub quote: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalTraceToolCall {
    pub tool: String,
    pub status: String,
    pub input: serde_json::Value,
    pub result_count: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalTrace {
    pub run_id: String,
    pub intent: String,
    pub tree_nodes: Vec<RetrievalTraceTreeNode>,
    pub candidates: Vec<RetrievalTraceCandidate>,
    pub tool_calls: Vec<RetrievalTraceToolCall>,
    pub finalize_gate: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RagToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: serde_json::Value,
}

pub struct StructureBlockSeed {
    pub page_no: u32,
    pub block_index: u32,
    pub text: String,
    pub bbox_list: serde_json::Value,
    pub role: String,
    pub region_index: u32,
    pub region_id: String,
}

#[derive(Clone, Debug)]
pub struct OutlineSeed {
    pub title: String,
    pub level: u32,
    pub page_no: u32,
    pub order_index: u32,
}

#[derive(Clone, Debug)]
pub struct OutlineRange {
    pub title: String,
    pub level: u32,
    pub page_no: u32,
    pub page_end: u32,
    pub block_start_index: u32,
    pub block_end_index: u32,
    pub order_index: u32,
}

pub struct RetrievalRequest<'a> {
    pub document_id: &'a str,
    pub question: &'a str,
    pub retrieval_query: Option<&'a str>,
    pub selected_text: Option<&'a str>,
    pub selected_block_id: Option<&'a str>,
    pub selected_bbox_list: Option<serde_json::Value>,
    pub client_context: Option<&'a str>,
    pub page: Option<u32>,
    pub page_mode: Option<&'a str>,
    pub page_source: Option<&'a str>,
    pub context_budget: crate::model_catalog::ModelContextBudget,
    pub force_document_start: bool,
}

pub struct RetrievalRun {
    pub id: String,
    pub intent: String,
    pub prompt_context: String,
    pub citations: Vec<Citation>,
    pub trace: RetrievalTrace,
    pub context_budget: crate::model_catalog::ModelContextBudget,
}

pub struct RagToolExecutionOutput {
    pub citations: Vec<Citation>,
    pub trace_candidates: Vec<RetrievalTraceCandidate>,
    pub tree_nodes: Vec<RetrievalTraceTreeNode>,
    pub tool_call: RetrievalTraceToolCall,
}

#[derive(Clone, Copy, Debug)]
struct InitialRetrievalLimits {
    tree: usize,
    per_section: u32,
    fts: u32,
    table_anchors: u32,
    open_table: u32,
    page_blocks: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct RagToolCapabilities {
    pub vision_enabled: bool,
    /// Whether the chat's "联网" toggle is on — gates the web_search/web_fetch tools.
    pub web_enabled: bool,
    pub max_quote_chars: usize,
}

impl Default for RagToolCapabilities {
    fn default() -> Self {
        Self {
            vision_enabled: false,
            web_enabled: false,
            max_quote_chars: default_citation_quote_chars(),
        }
    }
}

#[cfg(test)]
fn merge_retrieval_citations(accumulated: &mut Vec<Citation>, citations: &[Citation]) {
    merge_retrieval_citations_with_budget(
        accumulated,
        citations,
        &crate::model_catalog::ModelContextBudget::default(),
    );
}

pub fn merge_retrieval_citations_with_budget(
    accumulated: &mut Vec<Citation>,
    citations: &[Citation],
    budget: &crate::model_catalog::ModelContextBudget,
) {
    let mut seen = accumulated
        .iter()
        .map(citation_dedupe_key)
        .collect::<std::collections::HashSet<_>>();
    for citation in citations {
        let key = citation_dedupe_key(citation);
        if seen.insert(key.clone()) {
            if accumulated.len() >= budget.max_accumulated_citations {
                if let Some(remove_index) = capped_citation_replacement_index(accumulated, citation)
                {
                    accumulated.remove(remove_index);
                } else {
                    continue;
                }
            }
            accumulated.push(citation.clone());
            continue;
        }
        if let Some(existing) = accumulated
            .iter_mut()
            .find(|existing| citation_dedupe_key(existing) == key)
        {
            if should_replace_citation(existing, citation) {
                *existing = citation.clone();
            }
        }
    }
    prefer_open_table_citations(accumulated);
    order_citations_for_prompt(accumulated);
    relabel_citations(accumulated);
}

pub fn apply_retrieval_citations(run: &mut RetrievalRun, citations: Vec<Citation>) {
    let mut citations = citations;
    prefer_open_table_citations(&mut citations);
    order_citations_for_prompt(&mut citations);
    relabel_citations(&mut citations);
    run.prompt_context = build_prompt_context(&citations, &run.context_budget);
    run.citations = citations;
}

fn should_replace_citation(existing: &Citation, incoming: &Citation) -> bool {
    if existing.source == "selection" {
        return false;
    }
    if existing.section_title.is_none() && incoming.section_title.is_some() {
        return true;
    }
    citation_source_rank(&incoming.source) > citation_source_rank(&existing.source)
}

fn capped_citation_replacement_index(
    accumulated: &[Citation],
    incoming: &Citation,
) -> Option<usize> {
    let incoming_rank = citation_source_rank(&incoming.source);
    if contextual_source_should_get_slot(accumulated, incoming) {
        if let Some((index, _)) = accumulated
            .iter()
            .enumerate()
            .filter(|(_, citation)| citation.source != "selection")
            .filter(|(_, citation)| {
                matches!(
                    citation.source.as_str(),
                    "table_fact" | "table_anchor" | "fts"
                )
            })
            .min_by_key(|(_, citation)| citation_source_rank(&citation.source))
        {
            return Some(index);
        }
        if let Some((index, _)) = accumulated
            .iter()
            .enumerate()
            .filter(|(_, citation)| citation.source != "selection")
            .find(|(_, citation)| matches!(citation.source.as_str(), "open_table"))
        {
            return Some(index);
        }
    }
    accumulated
        .iter()
        .enumerate()
        .filter(|(_, citation)| {
            !matches!(citation.source.as_str(), "selection" | "open_table_context")
        })
        .filter(|(_, citation)| citation_source_rank(&citation.source) < incoming_rank)
        .min_by_key(|(_, citation)| citation_source_rank(&citation.source))
        .map(|(index, _)| index)
}

fn contextual_source_should_get_slot(accumulated: &[Citation], incoming: &Citation) -> bool {
    if !matches!(
        incoming.source.as_str(),
        "open_table_context"
            | "open_section"
            | "open_pages"
            | "current_view"
            | "read_tree_node_lines"
            | "analyze_page"
    ) {
        return false;
    }
    !accumulated
        .iter()
        .any(|citation| citation.source == incoming.source)
}

fn citation_source_rank(source: &str) -> u8 {
    match source {
        "selection" => 100,
        "current_view" => 90,
        "open_table" => 75,
        "open_table_context" => 74,
        "table_fact" => 70,
        "table_anchor" => 65,
        "analyze_visual" => 70,
        "analyze_page" => 70,
        "open_visual" => 65,
        "visual_asset" => 60,
        "visual_anchor" => 55,
        "inspect_objects" => 55,
        "open_section" => 40,
        "open_pages" => 30,
        "fts" => 20,
        "client-context" => 10,
        _ => 0,
    }
}

fn prefer_open_table_citations(citations: &mut Vec<Citation>) {
    let opened_tables = citations
        .iter()
        .filter(|citation| citation.source == "open_table")
        .map(|citation| (citation.document_id.clone(), citation.block_id.clone()))
        .collect::<std::collections::HashSet<_>>();
    if opened_tables.is_empty() {
        return;
    }
    citations.retain(|citation| {
        !matches!(citation.source.as_str(), "table_fact" | "table_anchor")
            || !opened_tables.contains(&(citation.document_id.clone(), citation.block_id.clone()))
    });
}

fn order_citations_for_prompt(citations: &mut [Citation]) {
    citations.sort_by(|left, right| {
        citation_source_rank(&right.source)
            .cmp(&citation_source_rank(&left.source))
            .then_with(|| left.page.cmp(&right.page))
    });
}

struct StructureTreeHit {
    id: String,
    title: String,
    page: u32,
    page_end: u32,
    block_index: u32,
    score: f64,
}

#[derive(Clone)]
struct HeadingSeed {
    title: String,
    level: u32,
    page_no: u32,
    block_index: u32,
    bbox_list: serde_json::Value,
    kind: &'static str,
}

struct TreeNodeInsert<'a> {
    id: String,
    document_id: &'a str,
    parent_id: Option<String>,
    title: &'a str,
    level: u32,
    page_start: u32,
    page_end: u32,
    block_start_index: u32,
    block_end_index: u32,
    keywords: Vec<String>,
    visual_hint: serde_json::Value,
    order_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RagToolName {
    InspectTree,
    OpenSection,
    ReadTreeNodeLines,
    SearchChunks,
    OpenPages,
    ResolveTableAnchor,
    ResolveVisualAnchor,
    InspectObjects,
    InspectTables,
    OpenTable,
    SearchTableFacts,
    InspectVisuals,
    OpenVisual,
    RecallChatHistory,
    QueryKnowledgeGraph,
    SearchLibraryKnowledge,
    ListTrendingPapers,
    ListSources,
    ReadNoteSource,
    ProposeNoteEdit,
    ReadSheet,
    WebSearch,
    WebFetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageOpenMode {
    Overview,
    Header,
    Full,
}

impl PageOpenMode {
    pub fn from_str(value: Option<&str>) -> Self {
        match value.map(str::trim).map(str::to_lowercase).as_deref() {
            Some("header") => Self::Header,
            Some("full") => Self::Full,
            _ => Self::Overview,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Overview => "overview",
            Self::Header => "header",
            Self::Full => "full",
        }
    }
}

impl RagToolName {
    fn as_str(self) -> &'static str {
        match self {
            Self::InspectTree => "inspect_tree",
            Self::OpenSection => "open_section",
            Self::ReadTreeNodeLines => "read_tree_node_lines",
            Self::SearchChunks => "search_chunks",
            Self::OpenPages => "open_pages",
            Self::ResolveTableAnchor => "resolve_table_anchor",
            Self::ResolveVisualAnchor => "resolve_visual_anchor",
            Self::InspectObjects => "inspect_objects",
            Self::InspectTables => "inspect_tables",
            Self::OpenTable => "open_table",
            Self::SearchTableFacts => "search_table_facts",
            Self::InspectVisuals => "inspect_visuals",
            Self::OpenVisual => "open_visual",
            Self::RecallChatHistory => "recall_chat_history",
            Self::QueryKnowledgeGraph => "query_knowledge_graph",
            Self::SearchLibraryKnowledge => "search_library_knowledge",
            Self::ListTrendingPapers => "list_trending_papers",
            Self::ListSources => "list_sources",
            Self::ReadNoteSource => "read_note_source",
            Self::ProposeNoteEdit => "propose_note_edit",
            Self::ReadSheet => "read_sheet",
            Self::WebSearch => "web_search",
            Self::WebFetch => "web_fetch",
        }
    }
}

pub fn rag_tool_specs_for_capabilities(
    vision_enabled: bool,
    web_enabled: bool,
) -> Vec<RagToolSpec> {
    let mut specs = vec![
        RagToolSpec {
            name: "inspect_tree",
            description: "Find likely document structure nodes by semantic section query. Returns tree node ids and titles, not text evidence. Optionally pass documentId to inspect a referenced document.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 },
                    "documentId": { "type": "string", "description": "Optional referenced document id to inspect instead of the primary document" }
                },
                "required": ["query"]
            }),
        },
        RagToolSpec {
            name: "open_section",
            description: "Open text blocks from known structure tree nodes, or first inspect matching nodes from a query when node ids are absent. Optionally pass documentId to open from a referenced document (use that document's own tree node ids).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "treeNodeIds": { "type": "array", "items": { "type": "string" } },
                    "query": { "type": "string" },
                    "perSectionLimit": { "type": "integer", "minimum": 1, "maximum": 20 },
                    "documentId": { "type": "string", "description": "Optional referenced document id to open from instead of the primary document" }
                }
            }),
        },
        RagToolSpec {
            name: "read_tree_node_lines",
            description: "Read line-level evidence from known structure tree nodes. Prefer this after inspect_tree when detailed text inside a method, algorithm, or result section is needed. Optionally pass documentId to read from a referenced document (use that document's own tree node ids).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "treeNodeIds": { "type": "array", "items": { "type": "string" } },
                    "treeNodeId": { "type": "string" },
                    "nodeIds": { "type": "array", "items": { "type": "string" } },
                    "nodeId": { "type": "string" },
                    "query": { "type": "string" },
                    "lineLimit": { "type": "integer", "minimum": 1, "maximum": 30 },
                    "documentId": { "type": "string", "description": "Optional referenced document id to read from instead of the primary document" }
                }
            }),
        },
        RagToolSpec {
            name: "search_chunks",
            description: "Search chunk text locally. mode=keyword (default) uses FTS with ranking; mode=literal does an exact case-insensitive substring match (use for precise tokens like F1-score, θ, Eq.(3), identifiers). Optionally pass documentId to search a referenced document.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 },
                    "mode": { "type": "string", "enum": ["keyword", "literal"], "description": "keyword=FTS ranked (default); literal=exact substring" },
                    "documentId": { "type": "string", "description": "Optional referenced document id to search instead of the primary document" }
                },
                "required": ["query"]
            }),
        },
        RagToolSpec {
            name: "open_pages",
            description: "Open page-level evidence. Supports overview, header, or full page modes. Full mode returns line-ordered page chunks and is the fallback when table/visual structure extraction is incomplete. Optionally pass documentId to open a page from a referenced document.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "page": { "type": "integer", "minimum": 1 },
                    "mode": { "type": "string", "enum": ["overview", "header", "full"] },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 80 },
                    "documentId": { "type": "string", "description": "Optional referenced document id to open a page from instead of the primary document" }
                },
                "required": ["page"]
            }),
        },
        RagToolSpec {
            name: "resolve_table_anchor",
            description: "Resolve an explicit table reference such as Table 3 or 表3 to the indexed table id, page, caption, and bbox. Use before inspect_tables/open_table when the user names a table number.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "tableNumber": { "type": ["string", "integer"] },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 10 }
                }
            }),
        },
        RagToolSpec {
            name: "resolve_visual_anchor",
            description: "Resolve an explicit visual reference such as Figure 3, Fig. 3, 图3, or Chart 2 to an indexed visual asset id, page, caption, bbox, and crop path.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "visualNumber": { "type": ["string", "integer"] },
                    "assetType": { "type": "string", "enum": ["figure", "chart", "table", "image"] },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 10 }
                }
            }),
        },
        RagToolSpec {
            name: "inspect_objects",
            description: "List indexed visual objects such as tables, figures, charts, and images by query, page, or structure tree node ids. Use after inspect_tree or open_pages to discover objects on a relevant page/section.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "page": { "type": "integer", "minimum": 1 },
                    "treeNodeIds": { "type": "array", "items": { "type": "string" } },
                    "treeNodeId": { "type": "string" },
                    "nodeIds": { "type": "array", "items": { "type": "string" } },
                    "nodeId": { "type": "string" },
                    "assetType": { "type": "string", "enum": ["figure", "chart", "table", "image"] },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 }
                }
            }),
        },
        RagToolSpec {
            name: "inspect_tables",
            description: "Find indexed tables by caption, nearby text, row labels, column labels, or table facts. Use first for SOTA, benchmark, metric, score, and table questions.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "tableNumber": { "type": ["string", "integer"] },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 }
                }
            }),
        },
        RagToolSpec {
            name: "open_table",
            description: "Open a structured indexed table and return a table context bundle plus row/column facts. Prefer this after inspect_tables when exact scores, benchmark comparisons, or table explanations are needed.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "tableId": { "type": "string" },
                    "tableIds": { "type": "array", "items": { "type": "string" } },
                    "query": { "type": "string" },
                    "tableNumber": { "type": ["string", "integer"] },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 40 }
                }
            }),
        },
        RagToolSpec {
            name: "search_table_facts",
            description: "Search normalized table facts with SQLite FTS. Use for benchmark, SOTA, metric, score, result, and model-vs-model questions.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 40 }
                },
                "required": ["query"]
            }),
        },
        RagToolSpec {
            name: "inspect_visuals",
            description: "Find indexed visual assets such as figures, charts, tables, and images by caption or nearby text.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "assetType": { "type": "string", "enum": ["figure", "chart", "table", "image"] },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 }
                },
                "required": ["query"]
            }),
        },
        RagToolSpec {
            name: "open_visual",
            description: "Open a visual asset by id or query and return caption, page, bbox, crop path when available, and nearby text.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "assetId": { "type": "string" },
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 10 }
                }
            }),
        },
        RagToolSpec {
            name: "recall_chat_history",
            description: "Recall PRIOR conversation turns in this document's chat. The last few turns are ALREADY in context — use this ONLY to find OLDER discussion or to locate a specific past topic by keyword. Omit query to get the most recent turns. mode=keyword (default) ranks by term overlap; mode=literal does exact substring match.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "mode": { "type": "string", "enum": ["keyword", "literal"], "description": "keyword=term-overlap ranked (default); literal=exact substring" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 8 }
                }
            }),
        },
        RagToolSpec {
            name: "query_knowledge_graph",
            description: "Find OTHER workspace documents related to the focus document via the knowledge graph (shared concepts/entities + conversation co-citation). Use to discover and then route to related papers (call other tools with their documentId) when the answer likely spans multiple documents. Returns related documents with the shared concepts as the reason.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20, "description": "Max related documents to return (default 8)" }
                }
            }),
        },
        RagToolSpec {
            name: "search_library_knowledge",
            description: "Search the WHOLE workspace library (all documents, not just the focus document) by topic/concept/entity, using the precipitated knowledge graph. Use for questions like \"which of my papers are about X\". Returns matching documents with the matched concepts as the reason; route to one with its documentId for details.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Topic/concept/entity to search for across the library" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20, "description": "Max documents to return (default 10)" }
                },
                "required": ["query"]
            }),
        },
        RagToolSpec {
            name: "list_trending_papers",
            description: "List the Hugging Face trending papers the user is browsing (cached from the Trending view). Use for questions about \"trending papers\" / \"what's trending\" / which trending papers relate to a topic. period defaults to what the user is currently viewing. Optional query filters by title/abstract. Returns titles, abstracts, arXiv ids and HF links. Returns nothing if the user hasn't opened the Trending view yet.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "period": { "type": "string", "enum": ["daily", "weekly", "monthly"], "description": "Which cached trending list (defaults to the user's current period)" },
                    "query": { "type": "string", "description": "Optional keyword(s) to filter by title/abstract" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 30, "description": "Max papers to return (default 12)" }
                }
            }),
        },
    ];
    if vision_enabled {
        specs.push(RagToolSpec {
            name: "analyze_visual",
            description: "Analyze an indexed visual crop with a vision-capable model. Use only after inspect_visuals/open_visual when chart or figure content cannot be answered from caption and nearby text.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "assetId": { "type": "string" },
                    "question": { "type": "string" }
                },
                "required": ["assetId", "question"]
            }),
        });
        specs.push(RagToolSpec {
            name: "analyze_page",
            description: "Analyze a full PDF page screenshot with a vision-capable model. Use as page-level fallback when object/section tools identify a relevant page but caption/text evidence is insufficient.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "page": { "type": "integer", "minimum": 1 },
                    "question": { "type": "string" }
                },
                "required": ["page", "question"]
            }),
        });
    }
    // The library's table of contents. Every other tool needs a documentId to read
    // one source; without an enumeration there is no way to obtain one except by
    // guessing a topic. That matters most for an external MCP client, which cannot
    // see the sidebar.
    specs.push(RagToolSpec {
        name: "list_sources",
        description: "List the sources in the knowledge base — id, title, kind, collection and index state. Use this to discover what exists and to get the `documentId` the per-source tools need. Optionally filter by a title substring or by kind (pdf, docx, xlsx, pptx, note, markdown, web). For 'what do I have about X', prefer search_library_knowledge, which searches meaning rather than titles.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Case-insensitive substring of the title" },
                "contentType": { "type": "string", "description": "Restrict to one kind: pdf, docx, xlsx, pptx, note, markdown, text, web" },
                "limit": { "type": "integer", "minimum": 1, "maximum": 100, "description": "Max sources (default 30)" }
            }
        }),
    });
    // Authored sources (notes / markdown / clips) keep their Markdown in body_md, and
    // the user edits it live. Retrieval tools only ever see the *indexed* chunks, which
    // lag a save and are split up — useless when the question is about the draft as a
    // whole ("tidy this up", "what did I miss"). This returns the exact current text.
    specs.push(RagToolSpec {
        name: "read_note_source",
        description: "Read the FULL current Markdown of an authored source (note, markdown file, or web clip) — the exact text as saved, not indexed excerpts. Use this when the user asks about the note they are writing (summarize/improve/continue/check it), or whenever you need its verbatim text rather than a search hit. Returns an error for PDFs and Office files, which have no editable body — use search_chunks / open_pages for those.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "documentId": { "type": "string", "description": "Source id; defaults to the focus document" }
            }
        }),
    });
    // Editing is a *proposal*, never a write. The user is typing into the same note,
    // so a direct body_md write would race their autosave and one side would be lost
    // silently. Instead the proposal rides back on this tool call's args, the UI shows
    // it, and the editor — the single writer — applies it if the user accepts.
    specs.push(RagToolSpec {
        name: "propose_note_edit",
        description: "Propose a change to the current authored source. PREFER `edits`: a list of exact string replacements, which touches only what you name and leaves the rest of the note — including anything the user is typing right now — alone. Use `content` (a complete rewrite) only when the whole note genuinely changes. Call read_note_source first and copy `oldText` VERBATIM from it, including whitespace and line breaks; each oldText must match exactly once, so include surrounding lines if it would otherwise be ambiguous. Preserve [[wikilinks]]. This does NOT save: the user reviews and applies it. After calling, briefly say what you changed; do not repeat the note's text in your answer.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "Precise replacements, applied in order to the ORIGINAL note. Preferred over `content`.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string", "description": "Exact text from the note to replace; must occur exactly once" },
                            "newText": { "type": "string", "description": "Replacement text; empty string deletes" }
                        },
                        "required": ["oldText", "newText"]
                    }
                },
                "content": { "type": "string", "description": "Complete new Markdown body — only for a full rewrite" },
                "summary": { "type": "string", "description": "One short line describing the change" },
                "documentId": { "type": "string", "description": "Source id; defaults to the focus document" }
            }
        }),
    });
    // Spreadsheets index as one self-describing record per row (good for search),
    // but questions about layout, totals, or a specific cell need the aligned grid.
    // This returns it with real A1 addresses so the model can read and cite cells.
    specs.push(RagToolSpec {
        name: "read_sheet",
        description: "Read a spreadsheet (.xlsx) as an A1-addressable Markdown grid — a '#' row-number column plus one column per letter (A, B, C…), so cell B7 is column B of row 7. Use this when the question needs the table's layout, a whole column/row, totals, or a specific cell, rather than a keyword hit from search_chunks. Pass `sheet` to pick one tab (else all are returned). Only works on .xlsx sources; PDFs and other Office files return an error.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "sheet": { "type": "string", "description": "Sheet/tab name to read (case-insensitive); omit to read every sheet" },
                "documentId": { "type": "string", "description": "Source id; defaults to the focus document" }
            }
        }),
    });
    if web_enabled {
        specs.push(RagToolSpec {
            name: "web_search",
            description: "Search the public web for current/external information not in the user's documents (news, recent papers, docs, definitions). Returns titles, URLs and snippets. Cite sources as Markdown links in your answer. Use only when the document library cannot answer.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The web search query" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 10, "description": "Max results (default 5)" }
                },
                "required": ["query"]
            }),
        });
        specs.push(RagToolSpec {
            name: "web_fetch",
            description: "Fetch a specific web page (by URL) and return its readable text. Use to read a result returned by web_search, or a URL the user provided. Cite the page as a Markdown link.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "The http(s) URL to fetch" }
                },
                "required": ["url"]
            }),
        });
    }
    specs
}

pub fn is_registered_rag_tool(tool: &str) -> bool {
    is_registered_rag_tool_for_capabilities(tool, RagToolCapabilities::default())
}

pub fn is_registered_rag_tool_for_capabilities(
    tool: &str,
    capabilities: RagToolCapabilities,
) -> bool {
    normalize_rag_tool_name(tool, capabilities).is_some()
}

fn normalize_rag_tool_name(tool: &str, capabilities: RagToolCapabilities) -> Option<&'static str> {
    rag_tool_specs_for_capabilities(capabilities.vision_enabled, capabilities.web_enabled)
        .into_iter()
        .find(|spec| spec.name == tool)
        .map(|spec| spec.name)
}

struct RagToolRegistry<'a> {
    conn: &'a Connection,
    document_id: &'a str,
    max_quote_chars: usize,
}

impl<'a> RagToolRegistry<'a> {
    fn new(conn: &'a Connection, document_id: &'a str, max_quote_chars: usize) -> Self {
        Self {
            conn,
            document_id,
            max_quote_chars,
        }
    }

    fn inspect_tree(&self, query: &str, limit: usize) -> Result<Vec<StructureTreeHit>, String> {
        inspect_tree(self.conn, self.document_id, query, limit)
    }

    fn open_sections(
        &self,
        tree_hits: &[StructureTreeHit],
        per_section_limit: u32,
        query: &str,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        open_sections(
            self.conn,
            self.document_id,
            tree_hits,
            per_section_limit,
            query,
        )
    }

    fn read_tree_node_lines(
        &self,
        tree_hits: &[StructureTreeHit],
        line_limit: u32,
        query: &str,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        read_tree_node_lines(self.conn, self.document_id, tree_hits, line_limit, query)
    }

    fn search_chunks(&self, query: &str, limit: u32) -> Result<Vec<EvidenceCandidate>, String> {
        search_chunks(self.conn, self.document_id, query, limit)
    }

    fn search_chunks_literal(
        &self,
        query: &str,
        limit: u32,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        search_chunks_literal(self.conn, self.document_id, query, limit)
    }

    fn recall_chat_history(
        &self,
        query: &str,
        literal: bool,
        limit: u32,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        recall_chat_history(self.conn, self.document_id, query, literal, limit)
    }

    fn open_pages(
        &self,
        page: u32,
        mode: PageOpenMode,
        limit: u32,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        read_page_blocks(self.conn, self.document_id, page, mode, limit)
    }

    fn inspect_tables(&self, query: &str, limit: u32) -> Result<Vec<TableHit>, String> {
        inspect_tables(self.conn, self.document_id, query, limit)
    }

    fn resolve_table_anchors(&self, query: &str, limit: u32) -> Result<Vec<TableHit>, String> {
        resolve_table_anchors(self.conn, self.document_id, query, limit)
    }

    fn resolve_visual_anchors(
        &self,
        query: &str,
        asset_type: Option<&str>,
        limit: u32,
    ) -> Result<Vec<VisualAssetHit>, String> {
        resolve_visual_anchors(self.conn, self.document_id, query, asset_type, limit)
    }

    fn inspect_objects(
        &self,
        query: &str,
        asset_type: Option<&str>,
        page: Option<u32>,
        tree_hits: &[StructureTreeHit],
        limit: u32,
    ) -> Result<Vec<VisualAssetHit>, String> {
        inspect_objects(
            self.conn,
            self.document_id,
            query,
            asset_type,
            page,
            tree_hits,
            limit,
        )
    }

    fn open_tables(
        &self,
        table_ids: &[String],
        query: &str,
        limit: u32,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        open_tables(self.conn, self.document_id, table_ids, query, limit)
    }

    fn search_table_facts(
        &self,
        query: &str,
        limit: u32,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        search_table_facts(self.conn, self.document_id, query, limit)
    }

    fn inspect_visuals(
        &self,
        query: &str,
        asset_type: Option<&str>,
        limit: u32,
    ) -> Result<Vec<VisualAssetHit>, String> {
        inspect_visuals(self.conn, self.document_id, query, asset_type, limit)
    }

    fn open_visuals(
        &self,
        asset_id: Option<&str>,
        query: &str,
        limit: u32,
    ) -> Result<Vec<EvidenceCandidate>, String> {
        open_visuals(self.conn, self.document_id, asset_id, query, limit)
    }
}

#[cfg(test)]
fn execute_rag_tool_call(
    conn: &Connection,
    document_id: &str,
    tool: &str,
    args: &serde_json::Value,
    fallback_query: &str,
) -> RagToolExecutionOutput {
    execute_rag_tool_call_for_capabilities(
        conn,
        document_id,
        &[],
        tool,
        args,
        fallback_query,
        RagToolCapabilities::default(),
    )
}

pub fn execute_rag_tool_call_for_capabilities(
    conn: &Connection,
    document_id: &str,
    reference_document_ids: &[&str],
    tool: &str,
    args: &serde_json::Value,
    fallback_query: &str,
    capabilities: RagToolCapabilities,
) -> RagToolExecutionOutput {
    // Multi-document routing: a tool call may name a `documentId` to search one of
    // the user's "@-referenced" documents instead of the primary one. Honor it only
    // when it is the primary doc or appears in the whitelist (guards against the
    // model hallucinating an arbitrary id); otherwise fall back to the primary doc.
    let target_document_id = args
        .get("documentId")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .filter(|id| *id == document_id || reference_document_ids.contains(id))
        .unwrap_or(document_id);
    let registry = RagToolRegistry::new(conn, target_document_id, capabilities.max_quote_chars);
    let normalized_tool = match normalize_rag_tool_name(tool, capabilities) {
        Some(tool) => tool,
        None => {
            return fallback_search_output(
                &registry,
                tool,
                args,
                fallback_query,
                "unknown RAG tool",
            );
        }
    };

    let execution = match normalized_tool {
        "inspect_tree" => execute_inspect_tree_tool(&registry, args, fallback_query),
        "open_section" => execute_open_section_tool(&registry, args, fallback_query),
        "read_tree_node_lines" => {
            execute_read_tree_node_lines_tool(&registry, args, fallback_query)
        }
        "open_pages" => execute_open_pages_tool(&registry, args),
        "resolve_table_anchor" => {
            execute_resolve_table_anchor_tool(&registry, args, fallback_query)
        }
        "resolve_visual_anchor" => {
            execute_resolve_visual_anchor_tool(&registry, args, fallback_query)
        }
        "inspect_objects" => execute_inspect_objects_tool(&registry, args, fallback_query),
        "inspect_tables" => execute_inspect_tables_tool(&registry, args, fallback_query),
        "open_table" => execute_open_table_tool(&registry, args, fallback_query),
        "search_table_facts" => execute_search_table_facts_tool(&registry, args, fallback_query),
        "inspect_visuals" => execute_inspect_visuals_tool(&registry, args, fallback_query),
        "open_visual" => execute_open_visual_tool(&registry, args, fallback_query),
        "analyze_visual" => execute_analyze_visual_tool(&registry, args, fallback_query),
        "analyze_page" => execute_analyze_page_tool(&registry, args, fallback_query),
        // Chat-history recall is always scoped to the PRIMARY document, never the
        // routed target — the model must not be able to read a referenced doc's
        // private chat history via a stray documentId arg. Build a primary registry.
        "recall_chat_history" => {
            let primary_registry =
                RagToolRegistry::new(conn, document_id, capabilities.max_quote_chars);
            execute_recall_chat_history_tool(&primary_registry, args)
        }
        // Cross-document discovery is always scoped to the PRIMARY document (its
        // relations), never a routed referenced doc.
        "query_knowledge_graph" => {
            let primary_registry =
                RagToolRegistry::new(conn, document_id, capabilities.max_quote_chars);
            execute_query_knowledge_graph_tool(&primary_registry, args)
        }
        // Library-wide / cross-surface tools: not scoped to any single document.
        "search_library_knowledge" => execute_search_library_knowledge_tool(&registry, args),
        "list_trending_papers" => execute_list_trending_papers_tool(&registry, args),
        // Web tools: not document-scoped. The Exa key (if any) lives in app_settings.
        "list_sources" => execute_list_sources_tool(&registry, args),
        "read_note_source" => execute_read_note_source_tool(&registry),
        "propose_note_edit" => execute_propose_note_edit_tool(&registry, args),
        "read_sheet" => execute_read_sheet_tool(&registry, args),
        "web_search" => execute_web_search_tool(&registry, args, fallback_query),
        "web_fetch" => execute_web_fetch_tool(&registry, args),
        _ => execute_search_chunks_tool(&registry, args, fallback_query),
    };

    match execution {
        Ok(output) => output,
        Err(err) => fallback_search_output(&registry, normalized_tool, args, fallback_query, &err),
    }
}

pub fn apply_tool_execution(run: &mut RetrievalRun, output: &RagToolExecutionOutput) {
    run.trace.tool_calls.push(output.tool_call.clone());
    run.trace
        .candidates
        .extend(output.trace_candidates.iter().cloned());
    append_unique_tree_nodes(&mut run.trace.tree_nodes, &output.tree_nodes);
}

fn execute_inspect_tree_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = string_arg(args, "query").unwrap_or(fallback_query);
    let expanded_query = expand_query_for_retrieval(query);
    let limit = usize_arg(args, "limit", 8, 1, 20);
    let hits = registry.inspect_tree(&expanded_query, limit)?;
    let tree_nodes = hits
        .iter()
        .map(|hit| RetrievalTraceTreeNode {
            id: hit.id.clone(),
            title: hit.title.clone(),
            page: hit.page,
            block_index: hit.block_index,
            score: hit.score,
        })
        .collect::<Vec<_>>();
    Ok(RagToolExecutionOutput {
        citations: Vec::new(),
        trace_candidates: Vec::new(),
        tree_nodes,
        tool_call: tool_success_call(
            RagToolName::InspectTree,
            serde_json::json!({ "query": expanded_query, "limit": limit }),
            hits.len(),
        ),
    })
}

fn execute_open_section_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let node_ids = tree_node_ids_arg(args);
    let section_query = string_arg(args, "query").unwrap_or(fallback_query);
    let hits = if node_ids.is_empty() {
        registry.inspect_tree(section_query, 8)?
    } else {
        lookup_tree_hits(registry.conn, registry.document_id, &node_ids)?
    };
    let limit = u32_arg(args, "perSectionLimit", 10, 1, 20);
    let expanded_section_query = expand_query_for_retrieval(section_query);
    let candidates = registry.open_sections(&hits, limit, &expanded_section_query)?;
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    let tree_nodes = hits
        .iter()
        .map(|hit| RetrievalTraceTreeNode {
            id: hit.id.clone(),
            title: hit.title.clone(),
            page: hit.page,
            block_index: hit.block_index,
            score: hit.score,
        })
        .collect::<Vec<_>>();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes,
        tool_call: tool_success_call(
            RagToolName::OpenSection,
            serde_json::json!({
                "treeNodeIds": hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
                "query": section_query,
                "perSectionLimit": limit,
            }),
            candidates.len(),
        ),
    })
}

fn execute_read_tree_node_lines_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let node_ids = tree_node_ids_arg(args);
    let line_query = string_arg(args, "query").unwrap_or(fallback_query);
    let expanded_line_query = expand_query_for_retrieval(line_query);
    let hits = if node_ids.is_empty() {
        registry.inspect_tree(&expanded_line_query, 8)?
    } else {
        lookup_tree_hits(registry.conn, registry.document_id, &node_ids)?
    };
    let limit = u32_arg(args, "lineLimit", 12, 1, 30);
    let candidates = registry.read_tree_node_lines(&hits, limit, &expanded_line_query)?;
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    let tree_nodes = hits
        .iter()
        .map(|hit| RetrievalTraceTreeNode {
            id: hit.id.clone(),
            title: hit.title.clone(),
            page: hit.page,
            block_index: hit.block_index,
            score: hit.score,
        })
        .collect::<Vec<_>>();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes,
        tool_call: tool_success_call(
            RagToolName::ReadTreeNodeLines,
            serde_json::json!({
                "treeNodeIds": hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
                "query": line_query,
                "lineLimit": limit,
            }),
            candidates.len(),
        ),
    })
}

fn execute_search_chunks_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = string_arg(args, "query")
        .filter(|query| *query != "broad_context")
        .unwrap_or(fallback_query);
    let limit = u32_arg(args, "limit", 6, 1, 20);
    // mode: "keyword" (FTS, ranked, default) | "literal" (exact substring).
    let literal = args
        .get("mode")
        .and_then(|value| value.as_str())
        .map(|mode| mode.trim().eq_ignore_ascii_case("literal"))
        .unwrap_or(false);
    let mode = if literal { "literal" } else { "keyword" };
    let candidates = if literal {
        registry.search_chunks_literal(query, limit)?
    } else {
        registry.search_chunks(query, limit)?
    };
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::SearchChunks,
            serde_json::json!({ "query": query, "limit": limit, "mode": mode }),
            candidates.len(),
        ),
    })
}

fn execute_recall_chat_history_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    // Optional query: omit to fetch the most recent turns.
    let query = string_arg(args, "query").unwrap_or("");
    let limit = u32_arg(args, "limit", 5, 1, 8);
    let literal = args
        .get("mode")
        .and_then(|value| value.as_str())
        .map(|mode| mode.trim().eq_ignore_ascii_case("literal"))
        .unwrap_or(false);
    let mode = if literal { "literal" } else { "keyword" };
    let candidates = registry.recall_chat_history(query, literal, limit)?;
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::RecallChatHistory,
            serde_json::json!({ "query": query, "limit": limit, "mode": mode }),
            candidates.len(),
        ),
    })
}

/// Cross-document discovery: return documents related to the focus document as
/// pseudo-citations (page 0, no bbox) whose quote names the shared concepts. The
/// agent can read these and then route to a related doc via its `documentId`.
/// Also surfaces a "gap" note when the focus document has no relations yet.
fn execute_query_knowledge_graph_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let limit = u32_arg(args, "limit", 8, 1, 20) as usize;
    let related =
        crate::runtime::knowledge_graph::related_documents(registry.conn, registry.document_id)?;
    let citations: Vec<Citation> = related
        .iter()
        .take(limit)
        .enumerate()
        .map(|(index, item)| {
            let mut reason = if item.shared_concepts.is_empty() {
                String::new()
            } else {
                format!(" — shares: {}", item.shared_concepts.join(", "))
            };
            if item.co_citation > 0.0 {
                reason.push_str(&format!("; co-cited x{}", item.co_citation as i64));
            }
            Citation {
                id: format!("kg-{}-{index}", registry.document_id),
                label: format!("[{}]", index + 1),
                page: 0,
                block_id: String::new(),
                section_title: Some("related document".to_string()),
                quote: format!("Related document: {}{reason}", item.title),
                bbox_list: serde_json::json!([]),
                document_id: item.document_id.clone(),
                source: "knowledge_graph".to_string(),
            }
        })
        .collect();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::QueryKnowledgeGraph,
            serde_json::json!({ "documentId": registry.document_id, "limit": limit }),
            related.len(),
        ),
    })
}

fn execute_search_library_knowledge_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let query = string_arg(args, "query").unwrap_or_default().to_string();
    let limit = u32_arg(args, "limit", 10, 1, 20) as usize;
    let hits = crate::runtime::knowledge_graph::search_library(registry.conn, &query, limit)?;
    let citations: Vec<Citation> = hits
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let reason = if item.matched_concepts.is_empty() {
                String::new()
            } else {
                format!(" — matches: {}", item.matched_concepts.join(", "))
            };
            Citation {
                // Key by document, not position: a turn may call this tool twice
                // (refined query), and a positional `lib-{index}` would collide
                // (both start at lib-0), making evidence chips drop/overwrite each
                // other. search_library returns one hit per document, so this is
                // unique within a call and dedups the same source across calls.
                id: format!("lib-{}", item.document_id),
                label: format!("[{}]", index + 1),
                page: 0,
                block_id: String::new(),
                section_title: Some("library document".to_string()),
                quote: format!("Library document: {}{reason}", item.title),
                bbox_list: serde_json::json!([]),
                document_id: item.document_id.clone(),
                source: "knowledge_graph".to_string(),
            }
        })
        .collect();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::SearchLibraryKnowledge,
            serde_json::json!({ "query": query, "limit": limit }),
            hits.len(),
        ),
    })
}

#[derive(serde::Deserialize)]
struct CachedTrendingPaper {
    #[serde(rename = "arxivId", default)]
    arxiv_id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    upvotes: i64,
}

fn execute_list_trending_papers_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let period = match string_arg(args, "period") {
        Some("weekly") => "weekly",
        Some("monthly") => "monthly",
        _ => "daily",
    };
    let query = string_arg(args, "query").unwrap_or_default().to_lowercase();
    let limit = u32_arg(args, "limit", 12, 1, 30) as usize;

    let payload: Option<String> = registry
        .conn
        .query_row(
            "SELECT payload_json FROM trending_cache WHERE period = ?1",
            rusqlite::params![period],
            |row| row.get(0),
        )
        .ok();
    let papers: Vec<CachedTrendingPaper> = payload
        .as_deref()
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or_default();

    // Optional keyword filter: keep papers whose title/abstract contains any
    // query token (≥3 chars). No query → keep all.
    // Byte length, not char count: a 1–2 character CJK token (e.g. "训练") is ≥3
    // bytes and must be kept; only short ASCII stopwords (≤2 bytes) are dropped.
    let tokens: Vec<String> = query
        .split_whitespace()
        .filter(|t| t.len() >= 3)
        .map(str::to_string)
        .collect();
    let citations: Vec<Citation> = papers
        .iter()
        .filter(|paper| {
            if tokens.is_empty() {
                return true;
            }
            let haystack = format!("{} {}", paper.title, paper.summary).to_lowercase();
            tokens.iter().any(|token| haystack.contains(token))
        })
        .take(limit)
        .enumerate()
        .map(|(index, paper)| {
            let summary = truncate_chars(paper.summary.trim(), registry.max_quote_chars);
            Citation {
                id: format!("trending-{period}-{index}"),
                label: format!("[{}]", index + 1),
                page: 0,
                block_id: String::new(),
                section_title: Some(format!("trending paper ({period})")),
                quote: format!(
                    "{} (arXiv:{}, {} upvotes): {}",
                    paper.title.trim(),
                    paper.arxiv_id.trim(),
                    paper.upvotes,
                    summary
                ),
                bbox_list: serde_json::json!([]),
                document_id: String::new(),
                source: "trending".to_string(),
            }
        })
        .collect();
    let count = citations.len();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::ListTrendingPapers,
            serde_json::json!({ "period": period, "query": query, "limit": limit }),
            count,
        ),
    })
}

/// Read the configured Exa API key from app_settings (empty/absent → keyless mode).
fn exa_api_key(conn: &Connection) -> Option<String> {
    crate::load_app_setting(conn, "exa_api_key")
        .ok()
        .flatten()
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
}

/// `web_search`: query the public web (Exa when keyed, else DuckDuckGo). Results
/// are returned as non-document citations (page 0, no document) carrying
/// title/URL/snippet — the model cites them as Markdown links in the answer.
/// Hard cap on the body handed to the model. Notes are normally far smaller than
/// this; the cap only stops a pathological import from blowing the context budget,
/// and truncation is stated in the text so the model never assumes it saw the end.
const MAX_NOTE_SOURCE_CHARS: usize = 24_000;

/// Return the verbatim Markdown of an authored source. File-backed documents (PDF,
/// Office) carry no body_md — that is reported as an error rather than an empty
/// result, so the model routes to the retrieval tools instead of retrying this one.
fn execute_read_note_source_tool(
    registry: &RagToolRegistry<'_>,
) -> Result<RagToolExecutionOutput, String> {
    let row: Option<(String, String, Option<String>)> = registry
        .conn
        .query_row(
            "SELECT title, content_type, body_md FROM documents WHERE id = ?1",
            params![registry.document_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .ok();
    let Some((title, content_type, body_md)) = row else {
        return Err("Document not found".to_string());
    };
    let body = body_md.unwrap_or_default();
    if body.trim().is_empty() {
        return Err(format!(
            "'{title}' is a {content_type} source with no editable Markdown body. Use search_chunks / open_pages to read it."
        ));
    }
    let total_chars = body.chars().count();
    let truncated = total_chars > MAX_NOTE_SOURCE_CHARS;
    let text: String = if truncated {
        let head: String = body.chars().take(MAX_NOTE_SOURCE_CHARS).collect();
        format!("{head}\n\n[truncated: showing the first {MAX_NOTE_SOURCE_CHARS} of {total_chars} characters]")
    } else {
        body
    };
    let citation = Citation {
        // Keyed by document so repeated calls in one turn dedupe instead of stacking.
        id: format!("note-body-{}", registry.document_id),
        label: "[note]".to_string(),
        page: 0,
        block_id: String::new(),
        section_title: Some(format!("{title} (full source)")),
        quote: text,
        bbox_list: serde_json::json!([]),
        document_id: registry.document_id.to_string(),
        source: "note_source".to_string(),
    };
    Ok(RagToolExecutionOutput {
        citations: vec![citation],
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::ReadNoteSource,
            serde_json::json!({ "documentId": registry.document_id }),
            1,
        ),
    })
}

/// Enumerate the library's sources.
///
/// Every per-source tool needs a `documentId`, and nothing else hands one out
/// except `search_library_knowledge`, which matches on precipitated meaning and
/// so cannot answer "what is actually in here". An external MCP client has no
/// sidebar to read, making this its entry point.
fn execute_list_sources_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let query = string_arg(args, "query")
        .unwrap_or("")
        .trim()
        .to_lowercase();
    let content_type = string_arg(args, "contentType")
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty());
    let limit = u32_arg(args, "limit", 30, 1, 100);

    let mut sql = String::from(
        "SELECT d.id, d.title, d.content_type, d.index_status, d.page_count,
                COALESCE(c.name, '')
         FROM documents d
         LEFT JOIN collections c ON c.id = d.collection_id
         WHERE 1 = 1",
    );
    if !query.is_empty() {
        sql.push_str(" AND instr(lower(d.title), ?1) > 0");
    }
    if content_type.is_some() {
        sql.push_str(if query.is_empty() {
            " AND lower(d.content_type) = ?1"
        } else {
            " AND lower(d.content_type) = ?2"
        });
    }
    // Most recently touched first: what the user worked on last is the likeliest
    // thing a caller is asking about.
    sql.push_str(" ORDER BY d.last_opened_at DESC, d.updated_at DESC LIMIT ");
    sql.push_str(&limit.to_string());

    let mut stmt = registry
        .conn
        .prepare(&sql)
        .map_err(|err| format!("Failed to list sources: {err}"))?;
    let map_row = |row: &rusqlite::Row<'_>| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, String>(5)?,
        ))
    };
    let rows = match (query.is_empty(), &content_type) {
        (true, None) => stmt.query_map([], map_row),
        (false, None) => stmt.query_map(params![query], map_row),
        (true, Some(kind)) => stmt.query_map(params![kind], map_row),
        (false, Some(kind)) => stmt.query_map(params![query, kind], map_row),
    }
    .map_err(|err| format!("Failed to list sources: {err}"))?
    .collect::<Result<Vec<_>, _>>()
    .map_err(|err| format!("Failed to list sources: {err}"))?;

    let citations: Vec<Citation> = rows
        .iter()
        .enumerate()
        .map(|(index, (id, title, kind, status, pages, collection))| {
            let mut facts = vec![format!("kind: {kind}")];
            if !collection.is_empty() {
                facts.push(format!("collection: {collection}"));
            }
            if *pages > 1 {
                facts.push(format!("pages: {pages}"));
            }
            // State the index status: a source still indexing answers poorly, and
            // the caller should know that rather than conclude the content is thin.
            if status != "indexed" {
                facts.push(format!("index: {status}"));
            }
            Citation {
                // Keyed by document so repeated calls dedupe rather than stack.
                id: format!("source-{id}"),
                label: format!("[{}]", index + 1),
                page: 0,
                block_id: String::new(),
                section_title: Some("library source".to_string()),
                quote: format!("{title} — documentId: {id} ({})", facts.join(", ")),
                bbox_list: serde_json::json!([]),
                document_id: id.clone(),
                source: "library_index".to_string(),
            }
        })
        .collect();
    let count = citations.len();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::ListSources,
            serde_json::json!({ "query": query, "contentType": content_type, "limit": limit }),
            count,
        ),
    })
}

/// Hard cap on cells rendered for the model, so a large workbook can't blow the
/// context budget. Truncation past this is stated in the returned text.
const MAX_SHEET_CELLS: usize = 6_000;

/// Return an .xlsx source as an A1-addressable Markdown grid. Errors (rather than
/// returning empty) for non-spreadsheet sources so the model routes elsewhere.
fn execute_read_sheet_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let row: Option<(String, String, String)> = registry
        .conn
        .query_row(
            "SELECT title, content_type, path FROM documents WHERE id = ?1",
            params![registry.document_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .ok();
    let Some((title, content_type, path)) = row else {
        return Err("Document not found".to_string());
    };
    if content_type != "xlsx" {
        return Err(format!(
            "'{title}' is a {content_type} source, not a spreadsheet. Use read_note_source for notes or search_chunks / open_pages for PDFs and other files."
        ));
    }
    let sheet = args
        .get("sheet")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let grid =
        crate::office::read_xlsx_markdown(std::path::Path::new(&path), sheet, MAX_SHEET_CELLS)?;
    let section_title = match sheet {
        Some(name) => format!("{title} · {name}"),
        None => format!("{title} (spreadsheet)"),
    };
    let citation = Citation {
        // Keyed by document + sheet so repeated calls in one turn dedupe.
        id: format!("sheet-{}-{}", registry.document_id, sheet.unwrap_or("all")),
        label: "[sheet]".to_string(),
        page: 0,
        block_id: String::new(),
        section_title: Some(section_title),
        quote: grid,
        bbox_list: serde_json::json!([]),
        document_id: registry.document_id.to_string(),
        source: "sheet_source".to_string(),
    };
    Ok(RagToolExecutionOutput {
        citations: vec![citation],
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::ReadSheet,
            serde_json::json!({ "documentId": registry.document_id, "sheet": sheet }),
            1,
        ),
    })
}

/// Record a proposed rewrite of an authored source. Deliberately does NOT write:
/// the user is editing the same note, and a body_md write here would race the
/// editor's debounced autosave — whichever landed second would silently erase the
/// other. The editor stays the single writer; this only validates the proposal and
/// acknowledges it. The proposed text reaches the UI on this call's own args (the
/// trace carries them), so it is never duplicated back into the model's context.
fn execute_propose_note_edit_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let content = string_arg(args, "content").unwrap_or_default();
    let raw_edits = args.get("edits").and_then(|value| value.as_array());
    let has_edits = raw_edits.is_some_and(|list| !list.is_empty());
    if !has_edits && content.trim().is_empty() {
        return Err(
            "propose_note_edit needs either `edits` (preferred) or a complete `content` rewrite"
                .to_string(),
        );
    }
    let row: Option<(String, String, Option<String>)> = registry
        .conn
        .query_row(
            "SELECT title, content_type, body_md FROM documents WHERE id = ?1",
            params![registry.document_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .ok();
    let Some((title, content_type, body_md)) = row else {
        return Err("Document not found".to_string());
    };
    let body = body_md.unwrap_or_default();
    if body.trim().is_empty() {
        return Err(format!(
            "'{title}' is a {content_type} source and cannot be edited — only notes, markdown files and web clips have an editable body."
        ));
    }
    let summary = string_arg(args, "summary").unwrap_or_default();

    // Precise mode. Resolve every oldText against the note NOW: the model gets its
    // mistakes back immediately (ambiguous / not found, with the text echoed) instead
    // of the user discovering them at apply time. Resolution also rewrites each
    // oldText to the note's verbatim slice, so the apply step downstream only ever
    // needs a plain exact match and this matcher stays in one place.
    let (proposed_content, resolved_edits, line_count) = if has_edits {
        let mut parsed = Vec::new();
        for (index, item) in raw_edits.unwrap_or(&Vec::new()).iter().enumerate() {
            let old_text = item
                .get("oldText")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string();
            let new_text = item
                .get("newText")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!(
                        "Edit {}: newText is required (use \"\" to delete)",
                        index + 1
                    )
                })?
                .to_string();
            parsed.push(crate::runtime::note_edit::NoteEdit { old_text, new_text });
        }
        let (next, resolved) = crate::runtime::note_edit::apply_edits(&body, &parsed)?;
        let count = resolved.len();
        let json = resolved
            .into_iter()
            .map(|item| serde_json::json!({ "oldText": item.old_text, "newText": item.new_text }))
            .collect::<Vec<_>>();
        (next, serde_json::Value::Array(json), count)
    } else {
        (
            content.to_string(),
            serde_json::Value::Null,
            content.lines().count(),
        )
    };
    // A short ack, not the text: the model already has its own proposal, so echoing
    // it back as evidence would just burn context.
    let citation = Citation {
        id: format!("note-edit-{}", registry.document_id),
        label: "[edit]".to_string(),
        page: 0,
        block_id: String::new(),
        section_title: Some(format!("{title} (proposed edit)")),
        quote: format!(
            "Proposed {} for '{title}'{}. Awaiting the user's approval — it is not saved yet.",
            if has_edits {
                format!("{line_count} precise edit(s)")
            } else {
                format!("a full rewrite ({line_count} lines)")
            },
            if summary.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", summary.trim())
            }
        ),
        bbox_list: serde_json::json!([]),
        document_id: registry.document_id.to_string(),
        source: "note_edit_proposal".to_string(),
    };
    Ok(RagToolExecutionOutput {
        citations: vec![citation],
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::ProposeNoteEdit,
            // What the UI reads. `edits` carry note-verbatim oldText (already resolved),
            // so applying is a plain exact match; `content` is the resulting full text,
            // used for the preview and as the fallback for whole-note rewrites.
            serde_json::json!({
                "documentId": registry.document_id,
                "summary": summary,
                "edits": resolved_edits,
                "content": proposed_content,
            }),
            1,
        ),
    })
}

fn execute_web_search_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = string_arg(args, "query")
        .unwrap_or(fallback_query)
        .trim()
        .to_string();
    let limit = u32_arg(args, "limit", 5, 1, 10) as usize;
    let api_key = exa_api_key(registry.conn);
    let hits = crate::runtime::web_search::web_search(api_key.as_deref(), &query, limit)?;
    let citations: Vec<Citation> = hits
        .iter()
        .enumerate()
        .map(|(index, hit)| {
            let snippet = truncate_chars(hit.snippet.trim(), registry.max_quote_chars);
            Citation {
                id: format!("web-{index}"),
                label: format!("[{}]", index + 1),
                page: 0,
                block_id: String::new(),
                section_title: Some("web result".to_string()),
                quote: format!("{}\n{}\n{}", hit.title.trim(), hit.url.trim(), snippet),
                bbox_list: serde_json::json!([]),
                document_id: String::new(),
                source: "web_search".to_string(),
            }
        })
        .collect();
    let count = citations.len();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::WebSearch,
            serde_json::json!({ "query": query, "limit": limit, "engine": if api_key.is_some() { "exa" } else { "duckduckgo" } }),
            count,
        ),
    })
}

/// `web_fetch`: retrieve a URL and return its readable text as a single citation.
fn execute_web_fetch_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let url = string_arg(args, "url")
        .ok_or_else(|| "web_fetch requires a url".to_string())?
        .to_string();
    let text = crate::runtime::web_search::web_fetch(&url, registry.max_quote_chars)?;
    let citation = Citation {
        id: "web-fetch-0".to_string(),
        label: "[1]".to_string(),
        page: 0,
        block_id: String::new(),
        section_title: Some("web page".to_string()),
        quote: format!("{url}\n{text}"),
        bbox_list: serde_json::json!([]),
        document_id: String::new(),
        source: "web_fetch".to_string(),
    };
    Ok(RagToolExecutionOutput {
        citations: vec![citation],
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(RagToolName::WebFetch, serde_json::json!({ "url": url }), 1),
    })
}

fn execute_open_pages_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
) -> Result<RagToolExecutionOutput, String> {
    let page = u32_arg(args, "page", 1, 1, u32::MAX);
    let mode = PageOpenMode::from_str(args.get("mode").and_then(|value| value.as_str()));
    let limit = u32_arg(args, "limit", 8, 1, 80);
    let candidates = registry.open_pages(page, mode, limit)?;
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::OpenPages,
            serde_json::json!({ "page": page, "mode": mode.as_str(), "limit": limit }),
            candidates.len(),
        ),
    })
}

fn execute_inspect_tables_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = table_anchor_query(args, fallback_query);
    let limit = u32_arg(args, "limit", 8, 1, 20);
    let hits = registry.inspect_tables(&query, limit)?;
    let candidates = hits.iter().map(table_hit_to_candidate).collect::<Vec<_>>();
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::InspectTables,
            serde_json::json!({ "query": query, "limit": limit }),
            hits.len(),
        ),
    })
}

fn execute_resolve_table_anchor_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = table_anchor_query(args, fallback_query);
    let limit = u32_arg(args, "limit", 4, 1, 10);
    let hits = registry.resolve_table_anchors(&query, limit)?;
    let candidates = hits
        .iter()
        .map(table_hit_to_anchor_candidate)
        .collect::<Vec<_>>();
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::ResolveTableAnchor,
            serde_json::json!({ "query": query, "limit": limit }),
            hits.len(),
        ),
    })
}

fn execute_resolve_visual_anchor_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = visual_anchor_query(args, fallback_query);
    let asset_type = string_arg(args, "assetType");
    let limit = u32_arg(args, "limit", 4, 1, 10);
    let hits = registry.resolve_visual_anchors(&query, asset_type, limit)?;
    let candidates = hits
        .iter()
        .map(visual_hit_to_anchor_candidate)
        .collect::<Vec<_>>();
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::ResolveVisualAnchor,
            serde_json::json!({
                "query": query,
                "assetType": asset_type,
                "limit": limit
            }),
            hits.len(),
        ),
    })
}

fn execute_inspect_objects_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let node_ids = tree_node_ids_arg(args);
    let tree_hits = if node_ids.is_empty() {
        Vec::new()
    } else {
        lookup_tree_hits(registry.conn, registry.document_id, &node_ids)?
    };
    let page = optional_u32_arg(args, "page");
    let query = object_query(args, fallback_query, page, &tree_hits);
    let asset_type = string_arg(args, "assetType");
    let limit = u32_arg(args, "limit", 8, 1, 20);
    let hits = registry.inspect_objects(&query, asset_type, page, &tree_hits, limit)?;
    let candidates = hits
        .iter()
        .map(visual_hit_to_object_candidate)
        .collect::<Vec<_>>();
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    let tree_nodes = tree_hits
        .iter()
        .map(|hit| RetrievalTraceTreeNode {
            id: hit.id.clone(),
            title: hit.title.clone(),
            page: hit.page,
            block_index: hit.block_index,
            score: hit.score,
        })
        .collect::<Vec<_>>();
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes,
        tool_call: tool_success_call(
            RagToolName::InspectObjects,
            serde_json::json!({
                "query": query,
                "page": page,
                "treeNodeIds": tree_hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
                "assetType": asset_type,
                "limit": limit
            }),
            hits.len(),
        ),
    })
}

fn execute_open_table_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let mut table_ids = string_array_arg(args, "tableIds");
    if let Some(table_id) = string_arg(args, "tableId") {
        if !table_ids.iter().any(|item| item == table_id) {
            table_ids.push(table_id.to_string());
        }
    }
    let query = table_anchor_query(args, fallback_query);
    if table_ids.is_empty() {
        table_ids = registry
            .resolve_table_anchors(&query, 4)?
            .into_iter()
            .map(|hit| hit.id)
            .collect();
    }
    if table_ids.is_empty() {
        table_ids = registry
            .inspect_tables(&query, 4)?
            .into_iter()
            .map(|hit| hit.id)
            .collect();
    }
    let limit = u32_arg(args, "limit", 24, 1, 40);
    let candidates = registry.open_tables(&table_ids, &query, limit)?;
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::OpenTable,
            serde_json::json!({
                "tableIds": table_ids,
                "query": query,
                "limit": limit
            }),
            candidates.len(),
        ),
    })
}

fn execute_search_table_facts_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = string_arg(args, "query").unwrap_or(fallback_query);
    let limit = u32_arg(args, "limit", 16, 1, 40);
    let candidates = registry.search_table_facts(query, limit)?;
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::SearchTableFacts,
            serde_json::json!({ "query": query, "limit": limit }),
            candidates.len(),
        ),
    })
}

fn execute_inspect_visuals_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let query = string_arg(args, "query").unwrap_or(fallback_query);
    let asset_type = string_arg(args, "assetType");
    let limit = u32_arg(args, "limit", 8, 1, 20);
    let hits = registry.inspect_visuals(query, asset_type, limit)?;
    let candidates = hits.iter().map(visual_hit_to_candidate).collect::<Vec<_>>();
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::InspectVisuals,
            serde_json::json!({ "query": query, "assetType": asset_type, "limit": limit }),
            hits.len(),
        ),
    })
}

fn execute_open_visual_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let asset_id = string_arg(args, "assetId");
    let query = string_arg(args, "query").unwrap_or(fallback_query);
    let limit = u32_arg(args, "limit", 4, 1, 10);
    let candidates = registry.open_visuals(asset_id, query, limit)?;
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    Ok(RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: tool_success_call(
            RagToolName::OpenVisual,
            serde_json::json!({
                "assetId": asset_id,
                "query": query,
                "limit": limit
            }),
            candidates.len(),
        ),
    })
}

fn execute_analyze_visual_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let _ = (registry, args, fallback_query);
    Err("analyze_visual requires the async multimodal provider path".to_string())
}

fn execute_analyze_page_tool(
    registry: &RagToolRegistry<'_>,
    args: &serde_json::Value,
    fallback_query: &str,
) -> Result<RagToolExecutionOutput, String> {
    let _ = (registry, args, fallback_query);
    Err("analyze_page requires the async multimodal provider path".to_string())
}

fn fallback_search_output(
    registry: &RagToolRegistry<'_>,
    tool: &str,
    args: &serde_json::Value,
    fallback_query: &str,
    error: &str,
) -> RagToolExecutionOutput {
    let candidates = registry
        .search_chunks(fallback_query, 6)
        .unwrap_or_else(|search_err| {
            log::warn!("RAG fallback search failed after {tool} error {error}: {search_err}");
            Vec::new()
        });
    let citations = candidates_to_citations(&candidates, registry.max_quote_chars);
    let trace_candidates = trace_candidates_from_candidates(&candidates);
    RagToolExecutionOutput {
        citations,
        trace_candidates,
        tree_nodes: Vec::new(),
        tool_call: RetrievalTraceToolCall {
            tool: "search_chunks".to_string(),
            status: "fallback".to_string(),
            input: serde_json::json!({
                "fallbackFrom": tool,
                "fallbackReason": error,
                "originalArgs": args,
                "query": fallback_query,
                "limit": 6,
            }),
            result_count: candidates.len(),
            error: Some(error.to_string()),
        },
    }
}

fn tool_success_call(
    tool: RagToolName,
    input: serde_json::Value,
    result_count: usize,
) -> RetrievalTraceToolCall {
    RetrievalTraceToolCall {
        tool: tool.as_str().to_string(),
        status: "ok".to_string(),
        input,
        result_count,
        error: None,
    }
}

fn string_arg<'a>(args: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn table_anchor_query(args: &serde_json::Value, fallback_query: &str) -> String {
    let query = string_arg(args, "query").unwrap_or(fallback_query).trim();
    let Some(table_number) = table_number_arg(args, "tableNumber") else {
        return query.to_string();
    };
    if requested_table_number(query).is_some() {
        query.to_string()
    } else {
        format!("Table {table_number} {query}").trim().to_string()
    }
}

fn visual_anchor_query(args: &serde_json::Value, fallback_query: &str) -> String {
    let query = string_arg(args, "query").unwrap_or(fallback_query).trim();
    let Some(visual_number) = visual_number_arg(args, "visualNumber") else {
        return query.to_string();
    };
    if requested_visual_anchor(query).is_some() {
        query.to_string()
    } else {
        format!("Figure {visual_number} {query}").trim().to_string()
    }
}

fn object_query(
    args: &serde_json::Value,
    fallback_query: &str,
    page: Option<u32>,
    tree_hits: &[StructureTreeHit],
) -> String {
    if let Some(query) = string_arg(args, "query") {
        return query.to_string();
    }
    if page.is_some() || !tree_hits.is_empty() {
        String::new()
    } else {
        fallback_query.trim().to_string()
    }
}

fn table_number_arg(args: &serde_json::Value, key: &str) -> Option<String> {
    match args.get(key)? {
        serde_json::Value::String(value) => requested_table_number(value)
            .or_else(|| leading_reference_number(value))
            .filter(|value| !value.is_empty()),
        serde_json::Value::Number(value) => value
            .as_u64()
            .filter(|number| *number > 0)
            .map(|number| number.to_string()),
        _ => None,
    }
}

fn visual_number_arg(args: &serde_json::Value, key: &str) -> Option<String> {
    match args.get(key)? {
        serde_json::Value::String(value) => requested_visual_anchor(value)
            .map(|anchor| anchor.number)
            .or_else(|| leading_reference_number(value))
            .filter(|value| !value.is_empty()),
        serde_json::Value::Number(value) => value
            .as_u64()
            .filter(|number| *number > 0)
            .map(|number| number.to_string()),
        _ => None,
    }
}

fn optional_u32_arg(args: &serde_json::Value, key: &str) -> Option<u32> {
    args.get(key)
        .and_then(|value| value.as_u64())
        .filter(|value| *value > 0)
        .and_then(|value| u32::try_from(value).ok())
}

fn u32_arg(args: &serde_json::Value, key: &str, default: u32, min: u32, max: u32) -> u32 {
    args.get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default.clamp(min, max))
}

fn usize_arg(args: &serde_json::Value, key: &str, default: usize, min: usize, max: usize) -> usize {
    args.get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default.clamp(min, max))
}

fn tree_node_ids_arg(args: &serde_json::Value) -> Vec<String> {
    let mut ids = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for key in ["treeNodeIds", "nodeIds"] {
        if let Some(items) = args.get(key).and_then(|value| value.as_array()) {
            for id in items.iter().filter_map(|item| {
                item.as_str()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            }) {
                if seen.insert(id.to_string()) {
                    ids.push(id.to_string());
                }
            }
        }
    }
    for key in ["treeNodeId", "nodeId"] {
        if let Some(id) = string_arg(args, key) {
            if seen.insert(id.to_string()) {
                ids.push(id.to_string());
            }
        }
    }
    ids
}

fn string_array_arg(args: &serde_json::Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn append_unique_tree_nodes(
    target: &mut Vec<RetrievalTraceTreeNode>,
    nodes: &[RetrievalTraceTreeNode],
) {
    let mut seen = target
        .iter()
        .map(|node| node.id.clone())
        .collect::<std::collections::HashSet<_>>();
    for node in nodes {
        if seen.insert(node.id.clone()) {
            target.push(node.clone());
        }
    }
}

/// How many neighbouring blocks (above + below, same page) to fold into an FTS /
/// literal hit. FTS indexes one block per chunk, so a raw hit shows the model —
/// and highlights — a single line/paragraph. A radius of 1 gives a readable
/// context window without ballooning tokens.
const CHUNK_CONTEXT_RADIUS: i64 = 1;

/// Widen a single-block FTS/literal hit to include its immediate neighbours on
/// the same page (±`radius` blocks by `block_index`). The `quote` (what the LLM
/// is shown) and the `bbox_list` (what the reader highlights) are expanded
/// TOGETHER, so the marked region always equals the model-visible range. The hit
/// keeps its original `block_id`/`page` (identity, scroll anchor) — only the text
/// and geometry grow. A no-op when the block has no neighbours.
fn expand_candidate_to_window(conn: &Connection, candidate: &mut EvidenceCandidate, radius: i64) {
    if candidate.block_id.is_empty() || radius <= 0 {
        return;
    }
    let mut stmt = match conn.prepare(
        "SELECT b2.text, b2.bbox_json
         FROM document_blocks b1
         JOIN document_blocks b2
           ON b2.document_id = b1.document_id
          AND b2.page_no = b1.page_no
          AND b2.block_index BETWEEN b1.block_index - ?2 AND b1.block_index + ?2
         WHERE b1.id = ?1
         ORDER BY b2.block_index",
    ) {
        Ok(stmt) => stmt,
        Err(_) => return,
    };
    let collected = stmt.query_map(params![candidate.block_id, radius], |row| {
        let text: String = row.get(0)?;
        let bbox_json: String = row.get(1)?;
        Ok((text, bbox_json))
    });
    let rows: Vec<(String, String)> = match collected {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(_) => return,
    };
    // Only the hit block itself (or nothing) — leave it untouched.
    if rows.len() <= 1 {
        return;
    }
    let mut texts: Vec<String> = Vec::with_capacity(rows.len());
    let mut bboxes: Vec<serde_json::Value> = Vec::new();
    for (text, bbox_json) in rows {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            texts.push(trimmed.to_string());
        }
        if let Ok(serde_json::Value::Array(list)) =
            serde_json::from_str::<serde_json::Value>(&bbox_json)
        {
            bboxes.extend(list);
        }
    }
    if !texts.is_empty() {
        candidate.quote = texts.join("\n");
    }
    if !bboxes.is_empty() {
        candidate.bbox_list = serde_json::Value::Array(bboxes);
    }
}

pub fn search_chunks(
    conn: &Connection,
    document_id: &str,
    query: &str,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let limit = limit.clamp(1, 20);
    let mut stmt = conn
        .prepare(
            "SELECT c.id, c.document_id, c.page_no, c.block_ids_json, c.text, c.bbox_refs_json,
                    bm25(document_chunks_fts) AS rank
             FROM document_chunks_fts
             JOIN document_chunks c ON c.id = document_chunks_fts.chunk_id
             WHERE document_chunks_fts.document_id = ?1
               AND document_chunks_fts MATCH ?2
             ORDER BY rank
             LIMIT ?3",
        )
        .map_err(|err| format!("Failed to prepare chunk search: {err}"))?;

    let mut rows = stmt
        .query_map(
            params![document_id, escape_fts_query(query), limit],
            |row| {
                let block_ids_json: String = row.get(3)?;
                let bbox_refs_json: String = row.get(5)?;
                let block_ids: Vec<String> =
                    serde_json::from_str(&block_ids_json).unwrap_or_else(|_| Vec::new());
                let bbox_list: serde_json::Value =
                    serde_json::from_str(&bbox_refs_json).unwrap_or_else(|_| serde_json::json!([]));
                Ok(EvidenceCandidate {
                    chunk_id: row.get(0)?,
                    document_id: row.get(1)?,
                    page: row.get(2)?,
                    block_id: block_ids.first().cloned().unwrap_or_default(),
                    section_title: None,
                    quote: row.get(4)?,
                    bbox_list,
                    score: row.get(6)?,
                    source: "fts".to_string(),
                    tree_node_id: None,
                    block_role: None,
                })
            },
        )
        .map_err(|err| format!("Failed to search chunks: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read chunk search results: {err}"))?;
    // Fold ±1 neighbouring block into each hit so the LLM-visible range (and the
    // reader highlight) is a real context window, not a single line.
    for candidate in rows.iter_mut() {
        expand_candidate_to_window(conn, candidate, CHUNK_CONTEXT_RADIUS);
    }
    Ok(rows)
}

/// Exact-substring ("literal") chunk search. Complements `search_chunks` (FTS):
/// FTS tokenizes and ranks, which splits precise tokens like `F1-score`, `θ`,
/// `Eq.(3)`, or `snake_case`; this matches the raw query as a case-insensitive
/// substring so those survive. No ranking — ordered by page then chunk id.
pub fn search_chunks_literal(
    conn: &Connection,
    document_id: &str,
    query: &str,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let limit = limit.clamp(1, 20);
    // NOTE: SQLite's built-in lower() only ASCII-case-folds, so this is
    // case-insensitive for ASCII (covering identifiers like F1-score, snake_case)
    // and exact-byte for non-ASCII. CJK has no case so it's unaffected; the only
    // gap is opposite-case non-ASCII letters (e.g. Θ vs θ), which the precise-token
    // use case rarely hits. Order by page then chunk id (stable within a page).
    let mut stmt = conn
        .prepare(
            "SELECT c.id, c.document_id, c.page_no, c.block_ids_json, c.text, c.bbox_refs_json
             FROM document_chunks c
             WHERE c.document_id = ?1
               AND instr(lower(c.text), lower(?2)) > 0
             ORDER BY c.page_no, c.id
             LIMIT ?3",
        )
        .map_err(|err| format!("Failed to prepare literal chunk search: {err}"))?;

    let mut rows = stmt
        .query_map(params![document_id, query, limit], |row| {
            let block_ids_json: String = row.get(3)?;
            let bbox_refs_json: String = row.get(5)?;
            let block_ids: Vec<String> =
                serde_json::from_str(&block_ids_json).unwrap_or_else(|_| Vec::new());
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_refs_json).unwrap_or_else(|_| serde_json::json!([]));
            Ok(EvidenceCandidate {
                chunk_id: row.get(0)?,
                document_id: row.get(1)?,
                page: row.get(2)?,
                block_id: block_ids.first().cloned().unwrap_or_default(),
                section_title: None,
                quote: row.get(4)?,
                bbox_list,
                score: 0.0,
                source: "literal".to_string(),
                tree_node_id: None,
                block_role: None,
            })
        })
        .map_err(|err| format!("Failed to run literal chunk search: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read literal chunk search results: {err}"))?;
    // Match search_chunks: widen each hit to a ±1-block context window.
    for candidate in rows.iter_mut() {
        expand_candidate_to_window(conn, candidate, CHUNK_CONTEXT_RADIUS);
    }
    Ok(rows)
}

/// Recall PRIOR conversation turns for a document from the persisted `chat_turns`
/// table. The agent's in-context memory only carries the last few turns (compacted);
/// this lets it reach OLDER turns or locate a past topic by keyword. Reads the full
/// persisted history (also survives app restart, unlike the in-memory session).
///
/// Modes: empty `query` → most recent `limit` turns; `literal` → exact case-insensitive
/// substring; otherwise → keyword term-overlap ranking. Per-doc chat history is small,
/// so we over-fetch a recent window and filter/rank in Rust (no FTS table needed).
pub fn recall_chat_history(
    conn: &Connection,
    document_id: &str,
    query: &str,
    literal: bool,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    let query = query.trim();
    let limit = limit.clamp(1, 8) as usize;
    let window = (limit.saturating_mul(6)).clamp(limit, 200) as i64;

    // Filter to the current index version, matching the visible-history loader
    // (lib.rs load_stored_chat_turns) so recall never surfaces turns the UI hides
    // after a reindex.
    let mut stmt = conn
        .prepare(
            "SELECT id, user_message, assistant_answer
             FROM chat_turns
             WHERE document_id = ?1 AND index_version = ?2
             ORDER BY created_at DESC, rowid DESC
             LIMIT ?3",
        )
        .map_err(|err| format!("Failed to prepare chat history recall: {err}"))?;
    // Newest-first window.
    let rows = stmt
        .query_map(
            params![document_id, crate::CURRENT_INDEX_VERSION, window],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(|err| format!("Failed to recall chat history: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read chat history rows: {err}"))?;

    let selected: Vec<(String, String, String)> = if query.is_empty() {
        rows.into_iter().take(limit).collect()
    } else if literal {
        let needle = query.to_lowercase();
        rows.into_iter()
            .filter(|(_, q, a)| {
                q.to_lowercase().contains(&needle) || a.to_lowercase().contains(&needle)
            })
            .take(limit)
            .collect()
    } else {
        // Tokenize for keyword overlap. query_terms splits on non-alphanumeric, which
        // does NOT split CJK (Han chars are alphanumeric) — so a Chinese phrase stays
        // one token and whole-phrase matching is too strict. Add CJK single-char terms
        // as a fallback signal so reworded Chinese queries still rank prior turns.
        let mut terms = query_terms(query);
        let cjk_chars: Vec<String> = query
            .chars()
            .filter(|ch| is_cjk(*ch))
            .map(|ch| ch.to_string())
            .collect();
        terms.extend(cjk_chars);
        terms.sort();
        terms.dedup();

        let mut ranked: Vec<(usize, (String, String, String))> = rows
            .into_iter()
            .map(|(id, q, a)| {
                let haystack = format!("{q} {a}").to_lowercase();
                let score = terms
                    .iter()
                    .filter(|term| haystack.contains(term.as_str()))
                    .count();
                (score, (id, q, a))
            })
            .collect();
        // Stable sort by descending score keeps newest-first order among ties (input
        // is already newest-first). When nothing matches (score all 0), this degrades
        // to "most recent turns" instead of returning empty.
        ranked.sort_by(|a, b| b.0.cmp(&a.0));
        ranked.into_iter().take(limit).map(|(_, row)| row).collect()
    };

    let candidates = selected
        .into_iter()
        .map(
            |(turn_id, user_message, assistant_answer)| EvidenceCandidate {
                chunk_id: turn_id.clone(),
                document_id: document_id.to_string(),
                page: 0,
                block_id: turn_id,
                section_title: Some("Chat history".to_string()),
                quote: format!(
                    "Q: {}\nA: {}",
                    truncate_chars(user_message.trim(), 200),
                    truncate_chars(assistant_answer.trim(), 320)
                ),
                bbox_list: serde_json::json!([]),
                score: 0.0,
                source: "chat_history".to_string(),
                tree_node_id: None,
                block_role: None,
            },
        )
        .collect();
    Ok(candidates)
}

// --- Workspace manifest (multi-document "file tree" for the agent) ----------

/// Soft cap on how many documents are visible to the agent at once (the
/// `documentId`-routing whitelist). The focus document and any @-referenced
/// documents are always included; remaining slots are filled by local relevance
/// to the question. Prevents an enormous library from blowing up the prompt.
pub const WORKSPACE_MANIFEST_MAX_DOCS: usize = 50;
/// At or below this many indexed documents the whole manifest is inlined into the
/// agent's context. Above it, the library is "large": the agent gets only the
/// focus + @-referenced docs inline, plus `search_library`/`list_documents` tools
/// to discover the rest on demand (progressive disclosure — keeps the prompt
/// bounded no matter how big the library grows).
pub const WORKSPACE_MANIFEST_INLINE_MAX_DOCS: usize = 25;
/// Beyond the focus + @-referenced docs, how many of the highest-scoring docs
/// also get their abstract shown (the rest are listed title-only).
const MANIFEST_ABSTRACT_DOCS_TOP_K: usize = 8;
/// Abstract truncation (chars) for a detailed manifest entry.
const MANIFEST_ABSTRACT_CHARS: usize = 300;

struct ManifestRow {
    id: String,
    title: String,
    rel_dir: String,
    page_count: u32,
    abstract_text: String,
}

pub struct DocManifestEntry {
    pub document_id: String,
    pub title: String,
    pub rel_dir: String,
    pub page_count: u32,
    /// Truncated abstract when this entry is "detailed"; empty otherwise.
    pub summary: String,
    pub is_focus: bool,
    pub is_referenced: bool,
}

pub struct WorkspaceManifest {
    pub entries: Vec<DocManifestEntry>,
    /// The `documentId` whitelist (== entry ids), for the tool-dispatch guard.
    pub document_ids: Vec<String>,
    /// Total indexed documents in the workspace (before any cap). Drives the
    /// large-library "progressive disclosure" decision.
    pub total_indexed: usize,
    /// Every indexed document id (uncapped) — the dispatch whitelist for a large
    /// library, where the agent discovers documents on demand via `search_library`
    /// rather than from the inlined manifest.
    pub all_document_ids: Vec<String>,
}

/// A ranked workspace document hit for the agent's `search_library` /
/// `list_documents` tools (progressive disclosure of a large library).
pub struct WorkspaceDocHit {
    pub document_id: String,
    pub title: String,
    pub rel_dir: String,
    pub page_count: u32,
    pub summary: String,
}

impl WorkspaceManifest {
    /// Whether the library is large enough to switch from inlining the whole
    /// manifest to on-demand discovery via `search_library`/`list_documents`.
    pub fn is_large(&self) -> bool {
        self.total_indexed > WORKSPACE_MANIFEST_INLINE_MAX_DOCS
    }

    /// Render the manifest as a prompt block the agent uses to locate documents
    /// and route `documentId` tool calls. Every doc gets title + dir + page
    /// count; only the focus, @-referenced, and top-ranked docs get an abstract.
    pub fn to_prompt_block(&self) -> String {
        if self.entries.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "Workspace documents (the user's whole library). To gather evidence from any of them, pass its id as the `documentId` argument on a retrieval tool:\n",
        );
        for entry in &self.entries {
            push_manifest_entry(&mut out, entry);
        }
        out
    }

    /// Compact prompt block for a large library: only the focus + @-referenced
    /// docs inline, plus a note that the rest are discoverable via the
    /// `search_library`/`list_documents` tools. Keeps the prompt bounded.
    pub fn to_prompt_block_compact(&self) -> String {
        let pinned: Vec<&DocManifestEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.is_focus || entry.is_referenced)
            .collect();
        let mut out = format!(
            "Workspace documents: the user's library has {} indexed documents — too many to list in full here. The focus and any @-referenced documents are shown below. To find OTHER relevant documents in the library, call the `search_library` tool with a query (or `list_documents` to browse the most recent). Then gather evidence from a document by passing its id as the `documentId` argument on a retrieval tool.\n",
            self.total_indexed
        );
        for entry in pinned {
            push_manifest_entry(&mut out, entry);
        }
        out
    }
}

/// Render one manifest entry line (+ abstract when present) into `out`.
fn push_manifest_entry(out: &mut String, entry: &DocManifestEntry) {
    let mut tags = String::new();
    if entry.is_focus {
        tags.push_str(" [CURRENT FOCUS]");
    }
    if entry.is_referenced {
        tags.push_str(" [@referenced]");
    }
    let dir = if entry.rel_dir.is_empty() {
        String::new()
    } else {
        format!("{}/ ", entry.rel_dir)
    };
    out.push_str(&format!(
        "- [{}] {}\"{}\" ({}p){}\n",
        entry.document_id, dir, entry.title, entry.page_count, tags
    ));
    if !entry.summary.is_empty() {
        out.push_str(&format!("    Abstract: {}\n", entry.summary));
    }
}

fn manifest_dir_hint(path: &str) -> String {
    std::path::Path::new(path)
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_string()
}

fn manifest_summary(text: &str) -> String {
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_chars(&cleaned, MANIFEST_ABSTRACT_CHARS)
}

/// Build the workspace manifest: all indexed documents (title + dir + page
/// count), with abstracts for the focus, @-referenced, and locally-most-relevant
/// docs. Pure local ranking (question-term overlap with title+abstract) — no LLM.
/// Load every indexed document's manifest row (id, title, dir, page count,
/// abstract), most-recently-opened first. Shared by the manifest builder and the
/// `search_library` ranking.
fn load_manifest_rows(conn: &Connection) -> Result<Vec<ManifestRow>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT d.id,
                    COALESCE(NULLIF(d.short_title, ''), d.title),
                    d.path,
                    d.page_count,
                    COALESCE((SELECT group_concat(b.text, ' ')
                              FROM document_blocks b
                              WHERE b.document_id = d.id AND b.block_role = 'abstract'), '')
             FROM documents d
             WHERE d.index_status = 'indexed'
             ORDER BY d.last_opened_at DESC",
        )
        .map_err(|err| format!("Failed to prepare workspace manifest query: {err}"))?;
    let rows = stmt
        .query_map([], |row| {
            let path: String = row.get(2)?;
            Ok(ManifestRow {
                id: row.get(0)?,
                title: row.get(1)?,
                rel_dir: manifest_dir_hint(&path),
                page_count: row.get::<_, i64>(3)?.max(0) as u32,
                abstract_text: row.get(4)?,
            })
        })
        .map_err(|err| format!("Failed to load workspace manifest: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read workspace manifest: {err}"))?;
    Ok(rows)
}

/// Term-overlap score of a manifest row against pre-tokenized query `terms`.
fn manifest_row_score(row: &ManifestRow, terms: &[String]) -> usize {
    if terms.is_empty() {
        return 0;
    }
    let haystack = format!("{} {}", row.title, row.abstract_text).to_lowercase();
    terms
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .count()
}

/// Rank indexed documents by local term overlap with `query` (title + abstract),
/// for the agent's `search_library` tool. `exclude` ids are skipped (e.g. docs
/// already pinned in the compact manifest). An empty query returns the most
/// recently opened documents (browse). Returns up to `limit` hits, best first.
pub fn search_workspace_documents(
    conn: &Connection,
    query: &str,
    limit: usize,
    exclude: &[&str],
) -> Result<Vec<WorkspaceDocHit>, String> {
    let rows = load_manifest_rows(conn)?;
    let terms = query_terms(query);
    let excluded: std::collections::HashSet<&str> = exclude.iter().copied().collect();
    let mut scored: Vec<(ManifestRow, usize)> = rows
        .into_iter()
        .filter(|row| !excluded.contains(row.id.as_str()))
        .map(|row| {
            let score = manifest_row_score(&row, &terms);
            (row, score)
        })
        .collect();
    // For a real topical query, return ONLY actual term matches — never pad with
    // recency-ranked unrelated docs, or the model treats a stale doc as a match
    // and routes retrieval to it. An empty query (browse/list) keeps everything.
    if !terms.is_empty() {
        scored.retain(|(_, score)| *score > 0);
    }
    // Best score first; the SELECT's recency order is the stable tiebreak.
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(scored
        .into_iter()
        .take(limit.max(1))
        .map(|(row, _score)| WorkspaceDocHit {
            document_id: row.id,
            title: row.title,
            rel_dir: row.rel_dir,
            page_count: row.page_count,
            summary: manifest_summary(&row.abstract_text),
        })
        .collect())
}

pub fn load_workspace_manifest(
    conn: &Connection,
    question: &str,
    focus_document_id: &str,
    reference_document_ids: &[&str],
) -> Result<WorkspaceManifest, String> {
    let rows = load_manifest_rows(conn)?;
    let total_indexed = rows.len();
    let all_document_ids: Vec<String> = rows.iter().map(|row| row.id.clone()).collect();

    let focus = focus_document_id.trim();
    let referenced: std::collections::HashSet<&str> =
        reference_document_ids.iter().copied().collect();
    let terms = query_terms(question);

    // Split into always-included (focus + @) and the locally-ranked rest. Ranking
    // shares `manifest_row_score` with `search_workspace_documents` so the inlined
    // manifest and the search_library tool can never rank the same library
    // differently.
    let mut priority: Vec<&ManifestRow> = Vec::new();
    let mut rest: Vec<(&ManifestRow, usize)> = Vec::new();
    for row in &rows {
        if row.id == focus || referenced.contains(row.id.as_str()) {
            priority.push(row);
        } else {
            rest.push((row, manifest_row_score(row, &terms)));
        }
    }
    priority.sort_by_key(|row| usize::from(row.id != focus)); // focus leads
    rest.sort_by(|a, b| b.1.cmp(&a.1)); // by score desc; recency order is the stable tiebreak

    let mut entries: Vec<DocManifestEntry> = Vec::new();
    for row in priority {
        entries.push(DocManifestEntry {
            document_id: row.id.clone(),
            title: row.title.clone(),
            rel_dir: row.rel_dir.clone(),
            page_count: row.page_count,
            summary: manifest_summary(&row.abstract_text),
            is_focus: row.id == focus,
            is_referenced: referenced.contains(row.id.as_str()),
        });
    }
    let mut detailed_quota = MANIFEST_ABSTRACT_DOCS_TOP_K;
    for (row, score) in rest {
        if entries.len() >= WORKSPACE_MANIFEST_MAX_DOCS {
            break;
        }
        let detailed = detailed_quota > 0 && score > 0;
        if detailed {
            detailed_quota -= 1;
        }
        entries.push(DocManifestEntry {
            document_id: row.id.clone(),
            title: row.title.clone(),
            rel_dir: row.rel_dir.clone(),
            page_count: row.page_count,
            summary: if detailed {
                manifest_summary(&row.abstract_text)
            } else {
                String::new()
            },
            is_focus: false,
            is_referenced: false,
        });
    }
    let document_ids = entries
        .iter()
        .map(|entry| entry.document_id.clone())
        .collect();
    Ok(WorkspaceManifest {
        entries,
        document_ids,
        total_indexed,
        all_document_ids,
    })
}

pub fn rebuild_structure_tree(
    tx: &Transaction<'_>,
    document_id: &str,
    blocks: &[StructureBlockSeed],
    outlines: &[OutlineSeed],
) -> Result<(), String> {
    tx.execute(
        "DELETE FROM structure_tree_nodes WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear structure tree: {err}"))?;

    if blocks.is_empty() {
        return Ok(());
    }

    let first_page = blocks.first().map(|block| block.page_no).unwrap_or(1);
    let last_page = blocks
        .last()
        .map(|block| block.page_no)
        .unwrap_or(first_page);
    let last_block_by_page = blocks.iter().fold(HashMap::new(), |mut acc, block| {
        acc.insert(block.page_no, block.block_index);
        acc
    });
    let has_region_metadata = blocks
        .iter()
        .any(|block| block.region_index > 0 || !block.region_id.is_empty());
    let root_id = format!("tree-{document_id}-root");
    insert_tree_node(
        tx,
        TreeNodeInsert {
            id: root_id.clone(),
            document_id,
            parent_id: None,
            title: "Document",
            level: 0,
            page_start: first_page,
            page_end: last_page,
            block_start_index: blocks.first().map(|block| block.block_index).unwrap_or(0),
            block_end_index: blocks.last().map(|block| block.block_index).unwrap_or(0),
            keywords: Vec::new(),
            visual_hint: serde_json::json!({
                "kind": "root",
                "regionAware": has_region_metadata
            }),
            order_index: 0,
        },
    )?;

    if !outlines.is_empty() {
        return rebuild_outline_structure_tree(
            tx,
            document_id,
            &root_id,
            blocks,
            outlines,
            last_page,
        );
    }

    let headings = detect_headings(blocks);
    if headings.is_empty() {
        return Ok(());
    }

    let heading_ids = (0..headings.len())
        .map(|index| format!("tree-{document_id}-{}", index + 1))
        .collect::<Vec<_>>();
    for (index, heading) in headings.iter().enumerate() {
        let parent_id = headings[..index]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, candidate)| candidate.level < heading.level)
            .map(|(parent_index, _)| heading_ids[parent_index].clone())
            .unwrap_or_else(|| root_id.clone());
        let (page_end, block_end_index) = resolve_section_end(
            heading,
            headings.get(
                headings
                    .iter()
                    .enumerate()
                    .skip(index + 1)
                    .find(|(_, candidate)| candidate.level <= heading.level)
                    .map(|(next_index, _)| next_index)
                    .unwrap_or(headings.len()),
            ),
            last_page,
            &last_block_by_page,
        );
        insert_tree_node(
            tx,
            TreeNodeInsert {
                id: heading_ids[index].clone(),
                document_id,
                parent_id: Some(parent_id),
                title: &heading.title,
                level: heading.level,
                page_start: heading.page_no,
                page_end,
                block_start_index: heading.block_index,
                block_end_index,
                keywords: keywords_for_title(&heading.title),
                visual_hint: serde_json::json!({
                    "kind": heading.kind,
                    "bboxList": heading.bbox_list,
                }),
                order_index: (index + 1) as u32,
            },
        )?;
    }

    Ok(())
}

fn rebuild_outline_structure_tree(
    tx: &Transaction<'_>,
    document_id: &str,
    root_id: &str,
    blocks: &[StructureBlockSeed],
    outlines: &[OutlineSeed],
    page_count: u32,
) -> Result<(), String> {
    let outline_ranges = outline_ranges(outlines, page_count, blocks);
    if outline_ranges.is_empty() {
        return Ok(());
    }

    let heading_ids = (0..outline_ranges.len())
        .map(|index| format!("tree-{document_id}-outline-{}", index + 1))
        .collect::<Vec<_>>();
    for (index, outline) in outline_ranges.iter().enumerate() {
        let parent_id = outline_ranges[..index]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, candidate)| candidate.level < outline.level)
            .map(|(parent_index, _)| heading_ids[parent_index].clone())
            .unwrap_or_else(|| root_id.to_string());
        insert_tree_node(
            tx,
            TreeNodeInsert {
                id: heading_ids[index].clone(),
                document_id,
                parent_id: Some(parent_id),
                title: &outline.title,
                level: outline.level,
                page_start: outline.page_no,
                page_end: outline.page_end,
                block_start_index: outline.block_start_index,
                block_end_index: outline.block_end_index,
                keywords: keywords_for_title(&outline.title),
                visual_hint: serde_json::json!({
                    "kind": "outline",
                    "source": "pdf_outline",
                }),
                order_index: (index + 1) as u32,
            },
        )?;
    }
    Ok(())
}

pub fn outline_ranges(
    outlines: &[OutlineSeed],
    page_count: u32,
    blocks: &[StructureBlockSeed],
) -> Vec<OutlineRange> {
    let mut outlines = outlines
        .iter()
        .filter(|outline| !outline.title.trim().is_empty() && outline.page_no > 0)
        .cloned()
        .collect::<Vec<_>>();
    outlines.sort_by_key(|outline| (outline.order_index, outline.page_no, outline.level));
    let last_page = blocks
        .iter()
        .map(|block| block.page_no)
        .max()
        .unwrap_or(page_count.max(1));
    let last_page = last_page.max(page_count.max(1));
    let last_block_by_page = blocks.iter().fold(HashMap::new(), |mut acc, block| {
        acc.insert(block.page_no, block.block_index);
        acc
    });
    outlines
        .iter()
        .enumerate()
        .map(|(index, outline)| {
            let start_page = outline.page_no.clamp(1, last_page);
            let start_block = outline_start_block(blocks, start_page, &outline.title);
            let next_boundary = outlines
                .iter()
                .enumerate()
                .skip(index + 1)
                .find(|(_, candidate)| candidate.level <= outline.level);
            let (page_end, block_end_index) = if let Some((_, next)) = next_boundary {
                let next_page = next.page_no.clamp(1, last_page);
                let next_start_block = outline_start_block(blocks, next_page, &next.title);
                if next_page == start_page {
                    (
                        start_page,
                        next_start_block.saturating_sub(1).max(start_block),
                    )
                } else if next_start_block <= 1 {
                    let page_end = next_page.saturating_sub(1).max(start_page);
                    (
                        page_end,
                        last_block_by_page
                            .get(&page_end)
                            .copied()
                            .unwrap_or(start_block),
                    )
                } else {
                    (next_page, next_start_block - 1)
                }
            } else {
                (
                    last_page.max(start_page),
                    last_block_by_page
                        .get(&last_page)
                        .copied()
                        .unwrap_or(start_block),
                )
            };
            OutlineRange {
                title: normalize_heading_text(&outline.title),
                level: outline.level.clamp(1, 6),
                page_no: start_page,
                page_end,
                block_start_index: start_block,
                block_end_index,
                order_index: outline.order_index,
            }
        })
        .collect()
}

fn outline_start_block(blocks: &[StructureBlockSeed], page_no: u32, title: &str) -> u32 {
    let title_key = normalize_title_key(title);
    blocks
        .iter()
        .filter(|block| block.page_no == page_no)
        .find(|block| {
            let block_key = normalize_title_key(&block.text);
            !title_key.is_empty()
                && !block_key.is_empty()
                && (block_key == title_key
                    || block_key.contains(title_key.as_str())
                    || title_key.contains(block_key.as_str()))
        })
        .map(|block| block.block_index)
        .or_else(|| {
            blocks
                .iter()
                .filter(|block| block.page_no == page_no)
                .map(|block| block.block_index)
                .min()
        })
        .unwrap_or(1)
}

fn resolve_section_end(
    heading: &HeadingSeed,
    next_heading: Option<&HeadingSeed>,
    last_page: u32,
    last_block_by_page: &HashMap<u32, u32>,
) -> (u32, u32) {
    let last_block_index = |page_no: u32| {
        last_block_by_page
            .get(&page_no)
            .copied()
            .unwrap_or(heading.block_index)
    };

    let Some(next_heading) = next_heading else {
        return (last_page.max(heading.page_no), last_block_index(last_page));
    };

    if next_heading.page_no == heading.page_no {
        return (
            heading.page_no,
            next_heading
                .block_index
                .saturating_sub(1)
                .max(heading.block_index),
        );
    }

    if next_heading.block_index <= 1 {
        let page_end = next_heading.page_no.saturating_sub(1).max(heading.page_no);
        return (page_end, last_block_index(page_end));
    }

    (next_heading.page_no, next_heading.block_index - 1)
}

fn insert_tree_node(tx: &Transaction<'_>, node: TreeNodeInsert<'_>) -> Result<(), String> {
    tx.execute(
        "INSERT INTO structure_tree_nodes
            (id, document_id, parent_id, title, level, page_start, page_end,
             block_start_index, block_end_index, keywords_json, visual_hint_json,
             order_index, tree_version, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, 1, unixepoch(), unixepoch())",
        params![
            node.id,
            node.document_id,
            node.parent_id,
            node.title,
            node.level,
            node.page_start,
            node.page_end,
            node.block_start_index,
            node.block_end_index,
            serde_json::to_string(&node.keywords)
                .map_err(|err| format!("Failed to encode tree keywords: {err}"))?,
            node.visual_hint.to_string(),
            node.order_index,
        ],
    )
    .map_err(|err| format!("Failed to insert structure tree node: {err}"))?;
    Ok(())
}

fn detect_headings(blocks: &[StructureBlockSeed]) -> Vec<HeadingSeed> {
    let mut headings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut active_major: Option<HeadingSeed> = None;

    if let Some(header) = first_header_seed(blocks) {
        seen.insert(heading_seen_key(&header));
        headings.push(header);
    }

    for block in blocks.iter().filter(|block| block.role == "abstract") {
        let heading = HeadingSeed {
            title: "Abstract".to_string(),
            level: 1,
            page_no: block.page_no,
            block_index: block.block_index,
            bbox_list: block.bbox_list.clone(),
            kind: "abstract",
        };
        if seen.insert(heading_seen_key(&heading)) {
            headings.push(heading);
        }
    }

    for block in blocks {
        let heading = if block.role == "heading" {
            let Some((level, title)) = classify_heading(&block.text) else {
                continue;
            };
            if is_repeated_running_heading(block, &title, active_major.as_ref()) {
                continue;
            }
            if is_table_header_noise(blocks, block, &title) {
                continue;
            }
            let level = contextual_heading_level(level, &title, active_major.as_ref());
            HeadingSeed {
                title,
                level,
                page_no: block.page_no,
                block_index: block.block_index,
                bbox_list: block.bbox_list.clone(),
                kind: "heading",
            }
        } else if block.role == "body" {
            let Some(title) = infer_body_subheading(blocks, block, active_major.as_ref()) else {
                continue;
            };
            HeadingSeed {
                title,
                level: 2,
                page_no: block.page_no,
                block_index: block.block_index,
                bbox_list: block.bbox_list.clone(),
                kind: "inferred_subheading",
            }
        } else {
            continue;
        };
        if seen.insert(heading_seen_key(&heading)) {
            if heading.level == 1 && heading.kind == "heading" {
                active_major = Some(heading.clone());
            }
            headings.push(heading);
        }
    }
    headings.sort_by_key(|heading| (heading.page_no, heading.block_index, heading.level));
    headings
}

fn is_repeated_running_heading(
    block: &StructureBlockSeed,
    title: &str,
    active_major: Option<&HeadingSeed>,
) -> bool {
    if block.block_index > 1 || !is_all_caps_heading(title) {
        return false;
    }
    active_major
        .map(|major| {
            major.page_no < block.page_no
                && normalize_title_key(&major.title) == normalize_title_key(title)
        })
        .unwrap_or(false)
}

fn contextual_heading_level(level: u32, title: &str, active_major: Option<&HeadingSeed>) -> u32 {
    let Some(major) = active_major else {
        return level;
    };
    let major_key = normalize_title_key(&major.title);
    let title_key = normalize_title_key(title);
    if major_key == "approach" && matches!(title_key.as_str(), "approach" | "method" | "methods") {
        return 2;
    }
    if major_key == "results" && matches!(title_key.as_str(), "experiments" | "results") {
        return 2;
    }
    level
}

fn infer_body_subheading(
    blocks: &[StructureBlockSeed],
    block: &StructureBlockSeed,
    active_major: Option<&HeadingSeed>,
) -> Option<String> {
    active_major?;
    if is_before_first_page_caption(blocks, block) {
        return None;
    }
    let title = normalize_inferred_heading_text(&block.text);
    if !is_title_like_subheading(&title) {
        return None;
    }
    Some(title)
}

fn normalize_inferred_heading_text(text: &str) -> String {
    let title = normalize_heading_text(text);
    if let Some((prefix, suffix)) = title.rsplit_once('.') {
        if suffix.trim().chars().count() == 1 && suffix.trim().chars().all(char::is_uppercase) {
            return prefix.trim().to_string();
        }
    }
    title
}

fn is_title_like_subheading(title: &str) -> bool {
    if title.is_empty() || title.len() > 90 {
        return false;
    }
    let words = title.split_whitespace().collect::<Vec<_>>();
    if words.len() < 2 || words.len() > 8 {
        return false;
    }
    if title.contains(',') || title.contains(':') || title.contains(';') || title.contains("...") {
        return false;
    }
    let alpha_count = title.chars().filter(|ch| ch.is_alphabetic()).count();
    if alpha_count < 8 {
        return false;
    }
    let uppercase_words = words
        .iter()
        .filter(|word| {
            let trimmed = word.trim_matches(|ch: char| !ch.is_alphanumeric());
            trimmed
                .chars()
                .next()
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false)
                || matches!(
                    trimmed.to_lowercase().as_str(),
                    "and" | "or" | "with" | "for" | "of" | "on" | "in" | "the" | "to"
                )
        })
        .count();
    uppercase_words == words.len()
}

fn is_before_first_page_caption(blocks: &[StructureBlockSeed], block: &StructureBlockSeed) -> bool {
    blocks
        .iter()
        .filter(|candidate| candidate.page_no == block.page_no && candidate.role == "caption")
        .map(|candidate| candidate.block_index)
        .min()
        .map(|first_caption| block.block_index < first_caption)
        .unwrap_or(false)
}

fn is_table_header_noise(
    blocks: &[StructureBlockSeed],
    block: &StructureBlockSeed,
    title: &str,
) -> bool {
    let key = normalize_title_key(title);
    if !matches!(key.as_str(), "method" | "methods") {
        return false;
    }
    blocks.iter().any(|candidate| {
        candidate.page_no == block.page_no
            && candidate.role == "caption"
            && candidate
                .text
                .trim_start()
                .to_lowercase()
                .starts_with("table")
            && candidate.block_index.abs_diff(block.block_index) <= 5
    })
}

fn is_all_caps_heading(title: &str) -> bool {
    let alpha = title
        .chars()
        .filter(|ch| ch.is_alphabetic())
        .collect::<Vec<_>>();
    !alpha.is_empty() && alpha.iter().all(|ch| ch.is_uppercase())
}

fn normalize_title_key(title: &str) -> String {
    title
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn first_header_seed(blocks: &[StructureBlockSeed]) -> Option<HeadingSeed> {
    let mut header_blocks = blocks
        .iter()
        .filter(|block| {
            block.page_no == 1
                && matches!(block.role.as_str(), "title" | "authors" | "affiliations")
        })
        .collect::<Vec<_>>();
    if header_blocks.is_empty() {
        return None;
    }

    header_blocks.sort_by_key(|block| block.block_index);
    let first = header_blocks.first()?;
    let bbox_list = serde_json::Value::Array(
        header_blocks
            .iter()
            .flat_map(|block| block.bbox_list.as_array().into_iter().flatten().cloned())
            .collect(),
    );
    Some(HeadingSeed {
        title: "Paper header".to_string(),
        level: 1,
        page_no: 1,
        block_index: first.block_index,
        bbox_list,
        kind: "paper_header",
    })
}

fn heading_seen_key(heading: &HeadingSeed) -> (u32, u32, String) {
    (
        heading.page_no,
        heading.block_index,
        heading.title.to_lowercase(),
    )
}

fn classify_heading(text: &str) -> Option<(u32, String)> {
    let title = normalize_heading_text(text);
    if title.is_empty() || title.len() > 120 || title.split_whitespace().count() > 14 {
        return None;
    }
    let alpha_count = title.chars().filter(|ch| ch.is_alphabetic()).count();
    if alpha_count < 4 {
        return None;
    }

    let lower = title.to_lowercase();
    let named_level = match lower.as_str() {
        "introduction" => Some(1),
        "background" => Some(1),
        "related work" => Some(1),
        "method" | "methods" | "methodology" | "approach" => Some(1),
        "experiments" | "evaluation" | "results" => Some(1),
        "discussion" | "limitations" | "conclusion" | "references" => Some(1),
        _ => None,
    };
    if let Some(level) = named_level {
        return Some((level, title));
    }

    let number_token = title.split_whitespace().next()?;
    let normalized_token = number_token.trim_end_matches('.');
    if normalized_token.is_empty()
        || !normalized_token.starts_with(|ch: char| ch.is_ascii_digit())
        || !normalized_token
            .chars()
            .all(|ch| ch.is_ascii_digit() || ch == '.')
    {
        return None;
    }
    let level = normalized_token
        .split('.')
        .filter(|part| !part.is_empty())
        .count();
    if level == 0 {
        return None;
    }
    if normalized_token
        .split('.')
        .next()
        .and_then(|part| part.parse::<u32>().ok())
        .map(|section| section > 20)
        .unwrap_or(true)
    {
        return None;
    }
    let rest = title[number_token.len()..].trim();
    if rest.chars().filter(|ch| ch.is_alphabetic()).count() < 4 {
        return None;
    }
    if !normalized_token.contains('.') {
        let words = rest.split_whitespace().collect::<Vec<_>>();
        if words.len() == 1 && !matches!(rest.to_lowercase().as_str(), "approach" | "method") {
            return None;
        }
    }
    let level = level.clamp(1, 4) as u32;
    Some((level, title))
}

fn normalize_heading_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|ch: char| matches!(ch, ':' | '.' | '-' | ' '))
        .to_string()
}

fn keywords_for_title(title: &str) -> Vec<String> {
    query_terms(title)
        .into_iter()
        .filter(|term| term.len() > 2)
        .collect()
}

pub fn build_retrieval_run(
    conn: &Connection,
    request: RetrievalRequest<'_>,
) -> Result<RetrievalRun, String> {
    let intent = infer_intent(request.question);
    // Knowledge-base pivot (P0-d): focus-optional retrieval. With no focus
    // document, skip the single-doc seeding chain and return an empty run — the
    // agentic loop then relies on the library-wide tools (search_library_knowledge
    // / query_knowledge_graph) to gather evidence across the whole library.
    // NB: persisting a no-focus turn additionally needs chat_turns.document_id
    // nullable (a data-bearing table rebuild deferred to P1, where the no-focus
    // "ask my knowledge base" home is the consumer). This is the retrieval seam.
    if request.document_id.trim().is_empty() {
        let run_id = retrieval_run_id(conn)?;
        let finalize_gate = build_finalize_gate(&[], &request.context_budget);
        let trace = build_retrieval_trace(&run_id, &intent, &[], &[], &[], &finalize_gate);
        return Ok(RetrievalRun {
            id: run_id,
            intent,
            prompt_context: String::new(),
            citations: Vec::new(),
            trace,
            context_budget: request.context_budget,
        });
    }
    let tools = RagToolRegistry::new(
        conn,
        request.document_id,
        request.context_budget.max_quote_chars,
    );
    let mut tool_calls = Vec::new();
    let page_mode = PageOpenMode::from_str(request.page_mode);
    let initial_limits = initial_retrieval_limits(&request.context_budget);
    let raw_retrieval_query = request
        .retrieval_query
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(request.question);
    let retrieval_query = expand_query_for_retrieval(raw_retrieval_query);
    let retrieval_query = retrieval_query.as_str();
    let mut selected = Vec::new();
    if let Some(selected_text) = request
        .selected_text
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        selected.push(EvidenceCandidate {
            chunk_id: "selection".to_string(),
            document_id: request.document_id.to_string(),
            page: request.page.unwrap_or(1),
            block_id: request.selected_block_id.unwrap_or("").to_string(),
            section_title: None,
            quote: selected_text.to_string(),
            bbox_list: request
                .selected_bbox_list
                .clone()
                .unwrap_or_else(|| serde_json::json!([])),
            score: 0.0,
            source: "selection".to_string(),
            tree_node_id: None,
            block_role: Some("selection".to_string()),
        });
    }

    let tree_hits = match tools.inspect_tree(retrieval_query, initial_limits.tree) {
        Ok(results) => {
            push_tool_success(
                &mut tool_calls,
                RagToolName::InspectTree,
                serde_json::json!({ "query": retrieval_query, "limit": initial_limits.tree }),
                results.len(),
            );
            results
        }
        Err(err) => {
            push_tool_error(
                &mut tool_calls,
                RagToolName::InspectTree,
                serde_json::json!({ "query": retrieval_query, "limit": initial_limits.tree }),
                &err,
            );
            return Err(err);
        }
    };
    let tree_node_ids = tree_hits
        .iter()
        .map(|hit| hit.id.clone())
        .collect::<Vec<_>>();
    let mut section_context =
        match tools.open_sections(&tree_hits, initial_limits.per_section, retrieval_query) {
            Ok(results) => {
                push_tool_success(
                    &mut tool_calls,
                    RagToolName::OpenSection,
                    serde_json::json!({
                        "treeNodeIds": tree_node_ids.clone(),
                        "perSectionLimit": initial_limits.per_section,
                    }),
                    results.len(),
                );
                results
            }
            Err(err) => {
                push_tool_error(
                    &mut tool_calls,
                    RagToolName::OpenSection,
                    serde_json::json!({
                        "treeNodeIds": tree_node_ids.clone(),
                        "perSectionLimit": initial_limits.per_section,
                    }),
                    &err,
                );
                return Err(err);
            }
        };
    let mut fts = match tools.search_chunks(retrieval_query, initial_limits.fts) {
        Ok(results) => {
            push_tool_success(
                &mut tool_calls,
                RagToolName::SearchChunks,
                serde_json::json!({ "query": retrieval_query, "limit": initial_limits.fts }),
                results.len(),
            );
            results
        }
        Err(err) if is_recoverable_fts_error(&err) => {
            log::warn!(
                "FTS retrieval skipped for question {:?}: {err}",
                retrieval_query
            );
            push_tool_error(
                &mut tool_calls,
                RagToolName::SearchChunks,
                serde_json::json!({ "query": retrieval_query, "limit": initial_limits.fts }),
                &err,
            );
            Vec::new()
        }
        Err(err) => {
            push_tool_error(
                &mut tool_calls,
                RagToolName::SearchChunks,
                serde_json::json!({ "query": retrieval_query, "limit": initial_limits.fts }),
                &err,
            );
            return Err(err);
        }
    };
    let mut current_view_table_context = if matches!(request.page_source, Some("current_view"))
        && should_read_table_evidence(request.question)
    {
        let table_hits = match tools
            .resolve_table_anchors(retrieval_query, initial_limits.table_anchors)
        {
            Ok(results) => {
                push_tool_success(
                    &mut tool_calls,
                    RagToolName::ResolveTableAnchor,
                    serde_json::json!({ "query": retrieval_query, "limit": initial_limits.table_anchors }),
                    results.len(),
                );
                results
            }
            Err(err) => {
                push_tool_error(
                    &mut tool_calls,
                    RagToolName::ResolveTableAnchor,
                    serde_json::json!({ "query": retrieval_query, "limit": initial_limits.table_anchors }),
                    &err,
                );
                Vec::new()
            }
        };
        let table_ids = table_hits
            .into_iter()
            .filter(|hit| request.page.is_none_or(|page| hit.page_no == page))
            .map(|hit| hit.id)
            .collect::<Vec<_>>();
        if table_ids.is_empty() {
            Vec::new()
        } else {
            match tools.open_tables(&table_ids, retrieval_query, initial_limits.open_table) {
                Ok(results) => {
                    push_tool_success(
                        &mut tool_calls,
                        RagToolName::OpenTable,
                        serde_json::json!({
                            "tableIds": table_ids,
                            "query": retrieval_query,
                            "limit": initial_limits.open_table
                        }),
                        results.len(),
                    );
                    results
                }
                Err(err) => {
                    push_tool_error(
                        &mut tool_calls,
                        RagToolName::OpenTable,
                        serde_json::json!({
                            "tableIds": table_ids,
                            "query": retrieval_query,
                            "limit": initial_limits.open_table
                        }),
                        &err,
                    );
                    Vec::new()
                }
            }
        }
    } else {
        Vec::new()
    };
    let should_open_document_start =
        request.force_document_start || should_read_document_start(request.question, &intent);
    let page_seed = request
        .page
        .or_else(|| should_open_document_start.then_some(1))
        .or_else(|| fts.first().map(|candidate| candidate.page));
    let mut page_context = match page_seed {
        Some(page) => match tools.open_pages(page, page_mode, initial_limits.page_blocks) {
            Ok(results) => {
                push_tool_success(
                    &mut tool_calls,
                    RagToolName::OpenPages,
                    serde_json::json!({ "page": page, "mode": page_mode.as_str(), "limit": initial_limits.page_blocks }),
                    results.len(),
                );
                results
            }
            Err(err) => {
                push_tool_error(
                    &mut tool_calls,
                    RagToolName::OpenPages,
                    serde_json::json!({ "page": page, "mode": page_mode.as_str(), "limit": initial_limits.page_blocks }),
                    &err,
                );
                return Err(err);
            }
        },
        None => Vec::new(),
    };
    if matches!(request.page_source, Some("current_view")) {
        mark_current_view_page_context(&mut page_context);
    }

    let mut candidates = Vec::new();
    candidates.append(&mut selected);
    if matches!(request.page_source, Some("current_view")) {
        candidates.append(&mut page_context);
        candidates.append(&mut current_view_table_context);
        candidates.append(&mut section_context);
        candidates.append(&mut fts);
    } else if should_open_document_start {
        candidates.append(&mut page_context);
        candidates.append(&mut section_context);
        candidates.append(&mut fts);
    } else {
        candidates.append(&mut section_context);
        candidates.append(&mut fts);
        candidates.append(&mut page_context);
    }

    if let Some(client_context) = request
        .client_context
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .filter(|value| {
            request
                .selected_text
                .map(str::trim)
                .map(|selected| normalize_for_dedupe(selected) != normalize_for_dedupe(value))
                .unwrap_or(true)
        })
    {
        candidates.push(EvidenceCandidate {
            chunk_id: "client-context".to_string(),
            document_id: request.document_id.to_string(),
            page: request.page.unwrap_or(1),
            block_id: String::new(),
            section_title: None,
            quote: client_context.to_string(),
            bbox_list: serde_json::json!([]),
            score: 0.0,
            source: "client-context".to_string(),
            tree_node_id: None,
            block_role: Some("client-context".to_string()),
        });
    }

    let candidates = dedupe_candidates(candidates, request.context_budget.max_initial_citations);
    let mut citations =
        candidates_to_citations(&candidates, request.context_budget.max_quote_chars);
    order_citations_for_prompt(&mut citations);
    relabel_citations(&mut citations);
    let prompt_context = build_prompt_context(&citations, &request.context_budget);
    let run_id = retrieval_run_id(conn)?;
    let finalize_gate = build_finalize_gate(&citations, &request.context_budget);
    let trace = build_retrieval_trace(
        &run_id,
        &intent,
        &tree_hits,
        &candidates,
        &tool_calls,
        &finalize_gate,
    );

    record_retrieval_run(
        conn,
        &run_id,
        &request,
        &intent,
        &tree_node_ids,
        &candidates,
        &finalize_gate,
    )?;

    Ok(RetrievalRun {
        id: run_id,
        intent,
        prompt_context,
        citations,
        trace,
        context_budget: request.context_budget,
    })
}

fn initial_retrieval_limits(
    budget: &crate::model_catalog::ModelContextBudget,
) -> InitialRetrievalLimits {
    if budget.max_initial_citations >= 96 {
        return InitialRetrievalLimits {
            tree: 12,
            per_section: 16,
            fts: 12,
            table_anchors: 6,
            open_table: 32,
            page_blocks: 16,
        };
    }
    if budget.max_initial_citations >= 48 {
        return InitialRetrievalLimits {
            tree: 10,
            per_section: 12,
            fts: 8,
            table_anchors: 4,
            open_table: 24,
            page_blocks: 12,
        };
    }
    InitialRetrievalLimits {
        tree: 8,
        per_section: 10,
        fts: 6,
        table_anchors: 4,
        open_table: 24,
        page_blocks: 8,
    }
}

fn mark_current_view_page_context(candidates: &mut [EvidenceCandidate]) {
    for candidate in candidates {
        candidate.source = "current_view".to_string();
        candidate.block_role = Some("current_view_page".to_string());
        candidate.section_title = Some(match candidate.section_title.as_deref() {
            Some(title) if !title.trim().is_empty() => {
                format!("Current view page evidence: {}", title.trim())
            }
            _ => format!("Current view page {}", candidate.page),
        });
    }
}

fn push_tool_success(
    tool_calls: &mut Vec<RetrievalTraceToolCall>,
    tool: RagToolName,
    input: serde_json::Value,
    result_count: usize,
) {
    tool_calls.push(RetrievalTraceToolCall {
        tool: tool.as_str().to_string(),
        status: "ok".to_string(),
        input,
        result_count,
        error: None,
    });
}

fn push_tool_error(
    tool_calls: &mut Vec<RetrievalTraceToolCall>,
    tool: RagToolName,
    input: serde_json::Value,
    error: &str,
) {
    tool_calls.push(RetrievalTraceToolCall {
        tool: tool.as_str().to_string(),
        status: "error".to_string(),
        input,
        result_count: 0,
        error: Some(error.to_string()),
    });
}

fn should_read_document_start(question: &str, intent: &str) -> bool {
    if intent == "summarize" {
        return true;
    }
    let normalized = question.to_lowercase();
    normalized.contains("what is this paper about")
        || normalized.contains("what is this article about")
        || normalized.contains("summarize this paper")
        || normalized.contains("这篇")
            && (normalized.contains("讲的什么")
                || normalized.contains("讲什么")
                || normalized.contains("关于什么")
                || normalized.contains("主要内容")
                || normalized.contains("总结")
                || normalized.contains("概括")
                || normalized.contains("摘要")
                || normalized.contains("重要结论")
                || normalized.contains("主要结论")
                || normalized.contains("核心结论"))
        || normalized.contains("论文")
            && (normalized.contains("讲的什么")
                || normalized.contains("讲什么")
                || normalized.contains("关于什么")
                || normalized.contains("主要内容")
                || normalized.contains("总结")
                || normalized.contains("概括")
                || normalized.contains("摘要")
                || normalized.contains("重要结论")
                || normalized.contains("主要结论")
                || normalized.contains("核心结论"))
}

fn should_read_table_evidence(question: &str) -> bool {
    let normalized = question.to_lowercase();
    contains_any(
        &normalized,
        &[
            "table",
            "sota",
            "benchmark",
            "metric",
            "score",
            "result",
            "performance",
            "leaderboard",
            "表格",
            "指标",
            "分数",
            "成绩",
            "结果",
            "评测",
            "实验",
            "性能",
            "突出",
            "领先",
        ],
    ) || contains_numbered_marker(&normalized, "表")
}

fn contains_numbered_marker(value: &str, marker: &str) -> bool {
    value.contains(marker)
        && value.chars().any(|ch| {
            ch.is_ascii_digit()
                || matches!(
                    ch,
                    '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九'
                )
        })
}

fn expand_query_for_retrieval(query: &str) -> String {
    let trimmed = query.trim();
    let normalized = trimmed.to_lowercase();
    let mut parts = vec![trimmed.to_string()];

    if contains_any(
        &normalized,
        &[
            "这篇文章",
            "这篇论文",
            "文章讲",
            "论文讲",
            "讲了什么",
            "讲述了什么",
            "关于什么",
            "主要内容",
            "总结",
            "概括",
            "摘要",
            "what is this paper about",
            "what is this article about",
            "summarize this paper",
            "overview",
            "paper structure",
            "article structure",
            "main idea",
            "main contribution",
            "contribution",
            "contributions",
        ],
    ) {
        parts.push(
            "abstract introduction contribution contributions method approach experiments evaluation results conclusion"
                .to_string(),
        );
    }
    if contains_any(
        &normalized,
        &[
            "方法",
            "设计",
            "算法",
            "流程",
            "原理",
            "机制",
            "架构",
            "框架",
            "怎么做",
            "如何实现",
            "principle",
            "mechanism",
            "architecture",
            "method",
            "approach",
            "methodology",
            "algorithm",
            "framework",
        ],
    ) {
        parts.push(
            "method approach methodology algorithm framework architecture workflow pruning pipeline"
                .to_string(),
        );
    }
    if contains_any(
        &normalized,
        &[
            "实验",
            "评测",
            "结果",
            "指标",
            "效果",
            "成绩",
            "sota",
            "分数",
            "benchmark",
            "metric",
            "score",
            "experiment",
            "evaluation",
            "result",
            "performance",
        ],
    ) {
        parts.push(
            "experiments evaluation results benchmark performance metrics score sota table leaderboard case study".to_string(),
        );
    }
    if contains_any(
        &normalized,
        &[
            "作者",
            "署名",
            "机构",
            "单位",
            "标题",
            "author",
            "affiliation",
            "title",
        ],
    ) {
        parts.push("paper header author authors affiliation title".to_string());
    }
    if contains_any(
        &normalized,
        &[
            "引用",
            "参考文献",
            "相关工作",
            "出处",
            "来源",
            "reference",
            "citation",
            "related work",
        ],
    ) {
        parts.push("references related work citation bibliography prior work".to_string());
    }
    if contains_any(&normalized, &["图", "表", "figure", "table", "caption"]) {
        parts.push("figure table caption".to_string());
    }

    parts
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

#[derive(Clone)]
struct PdfLine {
    page_no: u32,
    line_no: u32,
    text: String,
    rect: Option<Rect>,
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl Rect {
    fn union(self, other: Rect) -> Rect {
        Rect {
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
            x2: self.x2.max(other.x2),
            y2: self.y2.max(other.y2),
        }
    }

    fn center_x(self) -> f64 {
        (self.x1 + self.x2) / 2.0
    }

    fn center_y(self) -> f64 {
        (self.y1 + self.y2) / 2.0
    }
}

#[derive(Clone)]
struct TableHit {
    id: String,
    document_id: String,
    page_no: u32,
    caption: String,
    facts: String,
    bbox_list: serde_json::Value,
    source: String,
    confidence: f64,
    score: f64,
}

#[derive(Clone)]
struct VisualAssetHit {
    id: String,
    document_id: String,
    page_no: u32,
    asset_type: String,
    caption: String,
    bbox_list: serde_json::Value,
    caption_bbox_list: serde_json::Value,
    image_path: String,
    nearby_text: String,
    ocr_text: String,
    source: String,
    confidence: f64,
    score: f64,
}

struct VisualHitQuery<'a> {
    query: &'a str,
    asset_type: Option<&'a str>,
    page: Option<u32>,
    tree_hits: &'a [StructureTreeHit],
    limit: u32,
    anchor_only: bool,
}

#[derive(Clone)]
struct ReconstructedTable {
    visual_asset_id: String,
    page_no: u32,
    caption: String,
    bbox: Rect,
    columns: Vec<TableColumn>,
    rows: Vec<TableRow>,
}

#[derive(Clone)]
struct TableColumn {
    index: usize,
    label: String,
    x: f64,
    bbox: Option<Rect>,
}

#[derive(Clone)]
struct TableRow {
    index: usize,
    label: String,
    category: Option<String>,
    bbox: Rect,
    cells: Vec<TableValueCell>,
}

#[derive(Clone)]
struct TableValueCell {
    column_index: usize,
    text: String,
    bbox: Rect,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TableCandidateDirection {
    AboveCaption,
    BelowCaption,
}

struct TableLineCandidateSet {
    direction: TableCandidateDirection,
    lines: Vec<PdfLine>,
}

pub fn rebuild_visual_evidence(tx: &Transaction<'_>, document_id: &str) -> Result<(), String> {
    clear_visual_evidence(tx, document_id)?;
    let lines = load_pdf_lines(tx, document_id)?;
    let visual_assets = load_visual_caption_assets(tx, document_id)?;
    for asset in &visual_assets {
        insert_visual_asset(tx, asset)?;
    }

    for table in reconstruct_tables(&lines, &visual_assets) {
        insert_reconstructed_table(tx, document_id, &table)?;
    }
    Ok(())
}

fn clear_visual_evidence(tx: &Transaction<'_>, document_id: &str) -> Result<(), String> {
    tx.execute(
        "DELETE FROM document_table_facts_fts WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear table fact FTS rows: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_facts WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear table facts: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_cells
         WHERE table_id IN (SELECT id FROM document_tables WHERE document_id = ?1)",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear table cells: {err}"))?;
    tx.execute(
        "DELETE FROM document_tables WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear document tables: {err}"))?;
    tx.execute(
        "DELETE FROM document_visual_assets WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear visual assets: {err}"))?;
    Ok(())
}

fn load_pdf_lines(conn: &Connection, document_id: &str) -> Result<Vec<PdfLine>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, page_no, line_no, block_id, block_index, text, bbox_json
             FROM document_lines
             WHERE document_id = ?1
             ORDER BY page_no, line_no",
        )
        .map_err(|err| format!("Failed to prepare PDF line read: {err}"))?;
    let rows = stmt
        .query_map(params![document_id], |row| {
            let bbox_json: String = row.get(6)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            Ok(PdfLine {
                page_no: row.get(1)?,
                line_no: row.get(2)?,
                text: row.get(5)?,
                rect: rect_from_bbox_list(&bbox_list),
            })
        })
        .map_err(|err| format!("Failed to read PDF lines: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect PDF lines: {err}"))?;
    Ok(rows)
}

fn load_visual_caption_assets(
    conn: &Connection,
    document_id: &str,
) -> Result<Vec<VisualAssetHit>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, document_id, page_no, block_index, text, bbox_json
             FROM document_blocks
             WHERE document_id = ?1 AND block_role = 'caption'
             ORDER BY page_no, block_index",
        )
        .map_err(|err| format!("Failed to prepare visual caption read: {err}"))?;
    let rows = stmt
        .query_map(params![document_id], |row| {
            let page_no = row.get::<_, u32>(2)?;
            let block_index = row.get::<_, u32>(3)?;
            let caption = row.get::<_, String>(4)?;
            let bbox_json: String = row.get(5)?;
            let caption_bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            let asset_type = classify_visual_asset_type(&caption);
            let bbox_list = estimate_visual_bbox(asset_type, &caption_bbox_list)
                .unwrap_or_else(|| caption_bbox_list.clone());
            Ok(VisualAssetHit {
                id: format!(
                    "visual-{}-p{}-b{}",
                    row.get::<_, String>(1)?,
                    page_no,
                    block_index
                ),
                document_id: row.get(1)?,
                page_no,
                asset_type: asset_type.to_string(),
                caption,
                bbox_list,
                caption_bbox_list,
                image_path: String::new(),
                nearby_text: String::new(),
                ocr_text: String::new(),
                source: "caption".to_string(),
                confidence: 0.62,
                score: 1.0,
            })
        })
        .map_err(|err| format!("Failed to read visual captions: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect visual captions: {err}"))?;

    let mut assets = Vec::new();
    for mut asset in rows {
        asset.nearby_text = nearby_page_text(conn, document_id, asset.page_no, &asset.caption)?;
        if asset.asset_type == "table"
            || asset.asset_type == "figure"
            || asset.asset_type == "chart"
        {
            assets.push(asset);
        }
    }
    Ok(assets)
}

fn nearby_page_text(
    conn: &Connection,
    document_id: &str,
    page_no: u32,
    caption: &str,
) -> Result<String, String> {
    let mut stmt = conn
        .prepare(
            "SELECT text
             FROM document_blocks
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY block_index
             LIMIT 40",
        )
        .map_err(|err| format!("Failed to prepare nearby visual text read: {err}"))?;
    let caption_key = normalize_for_dedupe(caption);
    let rows = stmt
        .query_map(params![document_id, page_no], |row| row.get::<_, String>(0))
        .map_err(|err| format!("Failed to read nearby visual text: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect nearby visual text: {err}"))?;
    Ok(rows
        .into_iter()
        .map(|text| normalize_for_dedupe(&text))
        .filter(|text| !text.is_empty() && normalize_for_dedupe(text) != caption_key)
        .take(4)
        .collect::<Vec<_>>()
        .join("\n"))
}

fn classify_visual_asset_type(caption: &str) -> &'static str {
    let lower = caption.to_lowercase();
    if lower.starts_with("table") {
        "table"
    } else if lower.contains("benchmark")
        || lower.contains("result")
        || lower.contains("score")
        || lower.contains("bar")
        || lower.contains("plot")
    {
        "chart"
    } else if lower.starts_with("figure") || lower.starts_with("fig.") {
        "figure"
    } else {
        "image"
    }
}

fn estimate_visual_bbox(
    asset_type: &str,
    caption_bbox_list: &serde_json::Value,
) -> Option<serde_json::Value> {
    let caption = rect_from_bbox_list(caption_bbox_list)?;
    if asset_type == "table" {
        let estimated = Rect {
            x1: 0.04_f64.min(caption.x1),
            y1: (caption.y1 - 0.10).max(0.03),
            x2: 0.96_f64.max(caption.x2),
            y2: (caption.y2 + 0.56).min(0.97),
        };
        return Some(rect_to_bbox_json(estimated));
    }
    if !matches!(asset_type, "figure" | "chart" | "image") {
        return Some(rect_to_bbox_json(caption));
    }
    let estimated = Rect {
        x1: 0.06_f64.min(caption.x1),
        y1: (caption.y1 - 0.42).max(0.03),
        x2: 0.94_f64.max(caption.x2),
        y2: caption.y2,
    };
    Some(rect_to_bbox_json(estimated))
}

fn insert_visual_asset(tx: &Transaction<'_>, asset: &VisualAssetHit) -> Result<(), String> {
    tx.execute(
        "INSERT INTO document_visual_assets
            (id, document_id, page_no, asset_type, caption, bbox_json, image_path,
             nearby_text, linked_block_ids_json, source, confidence, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, '[]', ?9, ?10, unixepoch())",
        params![
            asset.id,
            asset.document_id,
            asset.page_no,
            asset.asset_type,
            asset.caption,
            asset.bbox_list.to_string(),
            asset.image_path,
            asset.nearby_text,
            asset.source,
            asset.confidence,
        ],
    )
    .map_err(|err| format!("Failed to insert visual asset: {err}"))?;
    Ok(())
}

fn reconstruct_tables(
    lines: &[PdfLine],
    visual_assets: &[VisualAssetHit],
) -> Vec<ReconstructedTable> {
    let mut tables = Vec::new();
    for asset in visual_assets
        .iter()
        .filter(|asset| asset.asset_type == "table")
    {
        let page_lines = lines
            .iter()
            .filter(|line| line.page_no == asset.page_no)
            .cloned()
            .collect::<Vec<_>>();
        if let Some(table) = reconstruct_table_from_page_lines(asset, &page_lines) {
            tables.push(table);
        }
    }
    tables
}

fn reconstruct_table_from_page_lines(
    asset: &VisualAssetHit,
    lines: &[PdfLine],
) -> Option<ReconstructedTable> {
    let caption_rect = rect_from_bbox_list(&asset.caption_bbox_list)
        .or_else(|| rect_from_bbox_list(&asset.bbox_list));
    let reconstructed = table_line_candidate_sets(lines, caption_rect)
        .into_iter()
        .filter_map(|candidate| {
            reconstruct_table_from_candidate_lines(asset, caption_rect, candidate.lines)
                .map(|table| (candidate.direction, table))
        })
        .collect::<Vec<_>>();
    reconstructed
        .iter()
        .filter(|(direction, table)| {
            *direction == TableCandidateDirection::AboveCaption
                && reconstructed_table_score(table) >= 16.0
        })
        .max_by(|(_, left), (_, right)| compare_reconstructed_table_score(left, right))
        .map(|(_, table)| table.clone())
        .or_else(|| {
            reconstructed
                .into_iter()
                .max_by(|(_, left), (_, right)| compare_reconstructed_table_score(left, right))
                .map(|(_, table)| table)
        })
}

fn table_line_candidate_sets(
    lines: &[PdfLine],
    caption_rect: Option<Rect>,
) -> Vec<TableLineCandidateSet> {
    let Some(caption) = caption_rect else {
        return vec![TableLineCandidateSet {
            direction: TableCandidateDirection::BelowCaption,
            lines: lines.to_vec(),
        }];
    };
    let previous_caption_y2 = lines
        .iter()
        .filter_map(|line| {
            let rect = line.rect?;
            (is_table_caption_text(&line.text) && rect.y2 < caption.y1 - 0.001).then_some(rect.y2)
        })
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let next_caption_y1 = lines
        .iter()
        .filter_map(|line| {
            let rect = line.rect?;
            (is_table_caption_text(&line.text) && rect.y1 > caption.y2 + 0.001).then_some(rect.y1)
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

    let below_max = next_caption_y1.unwrap_or((caption.y2 + 0.58).min(1.0));
    let below = lines
        .iter()
        .filter(|line| {
            let Some(rect) = line.rect else {
                return false;
            };
            rect.y1 > caption.y2 && rect.y1 < below_max
        })
        .cloned()
        .collect::<Vec<_>>();
    let above_min = previous_caption_y2.unwrap_or((caption.y1 - 0.32).max(0.0));
    let above = lines
        .iter()
        .filter(|line| {
            let Some(rect) = line.rect else {
                return false;
            };
            rect.y2 < caption.y1 && rect.y2 > above_min
        })
        .cloned()
        .collect::<Vec<_>>();
    vec![
        TableLineCandidateSet {
            direction: TableCandidateDirection::AboveCaption,
            lines: above,
        },
        TableLineCandidateSet {
            direction: TableCandidateDirection::BelowCaption,
            lines: below,
        },
    ]
    .into_iter()
    .filter(|candidate| !candidate.lines.is_empty())
    .collect()
}

fn reconstruct_table_from_candidate_lines(
    asset: &VisualAssetHit,
    caption_rect: Option<Rect>,
    candidate_lines: Vec<PdfLine>,
) -> Option<ReconstructedTable> {
    let mut rows = cluster_table_rows(candidate_lines);
    if rows.is_empty() {
        return None;
    }

    let first_data_index = rows.iter().position(|row| {
        row_label_from_lines(row).is_some() && row.iter().any(|line| is_value_text(&line.text))
    })?;
    let data_rows = rows.split_off(first_data_index);
    let header_rows = rows;
    let columns = infer_table_columns(&data_rows, &header_rows);
    if columns.is_empty() {
        return None;
    }
    let mut category = None;
    let mut table_rows = Vec::new();
    for row in data_rows {
        if !row.iter().any(|line| is_value_text(&line.text)) {
            if let Some(label) = row
                .iter()
                .filter(|line| !is_value_text(&line.text))
                .map(|line| line.text.trim())
                .find(|text| is_category_text(text))
            {
                category = Some(label.to_string());
            }
            continue;
        }
        let label = row_label_from_lines(&row);
        let Some(label) = label else {
            continue;
        };
        let cells = row
            .iter()
            .filter(|line| is_value_text(&line.text))
            .filter_map(|line| {
                let rect = line.rect?;
                let column_index = nearest_column_index(rect.center_x(), &columns)?;
                Some(TableValueCell {
                    column_index,
                    text: normalize_table_value(&line.text),
                    bbox: rect,
                })
            })
            .filter(|cell| !cell.text.is_empty())
            .collect::<Vec<_>>();
        if cells.is_empty() {
            continue;
        }
        let bbox = row
            .iter()
            .filter_map(|line| line.rect)
            .reduce(Rect::union)?;
        table_rows.push(TableRow {
            index: table_rows.len() + 1,
            label,
            category: category.clone(),
            bbox,
            cells,
        });
    }
    if table_rows.is_empty() {
        return None;
    }
    let bbox = table_rows
        .iter()
        .map(|row| row.bbox)
        .chain(caption_rect)
        .reduce(Rect::union)?;
    Some(ReconstructedTable {
        visual_asset_id: asset.id.clone(),
        page_no: asset.page_no,
        caption: asset.caption.clone(),
        bbox,
        columns,
        rows: table_rows,
    })
}

fn compare_reconstructed_table_score(
    left: &ReconstructedTable,
    right: &ReconstructedTable,
) -> std::cmp::Ordering {
    reconstructed_table_score(left)
        .partial_cmp(&reconstructed_table_score(right))
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn reconstructed_table_score(table: &ReconstructedTable) -> f64 {
    let cell_count = table.rows.iter().map(|row| row.cells.len()).sum::<usize>() as f64;
    let labeled_columns = table
        .columns
        .iter()
        .filter(|column| !column.label.starts_with("Column "))
        .count() as f64;
    let table_text = table
        .columns
        .iter()
        .map(|column| column.label.as_str())
        .chain(table.rows.iter().flat_map(|row| {
            std::iter::once(row.label.as_str())
                .chain(row.cells.iter().map(|cell| cell.text.as_str()))
        }))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let caption_hits = meaningful_caption_terms(&table.caption)
        .iter()
        .filter(|term| table_text.contains(term.as_str()))
        .count() as f64;
    table.rows.len() as f64 * 8.0 + cell_count * 2.0 + labeled_columns * 3.0 + caption_hits * 8.0
}

fn row_label_from_lines(row: &[PdfLine]) -> Option<String> {
    let first_value_x = row
        .iter()
        .filter(|line| is_value_text(&line.text))
        .filter_map(|line| line.rect.map(|rect| rect.x1))
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))?;
    let mut label_parts = row
        .iter()
        .filter(|line| !is_value_text(&line.text))
        .filter_map(|line| {
            let rect = line.rect?;
            (rect.x1 < first_value_x).then_some((rect.x1, normalize_table_label(&line.text)))
        })
        .filter(|(_, text)| is_row_label_text(text))
        .collect::<Vec<_>>();
    label_parts.sort_by(|(left_x, _), (right_x, _)| {
        left_x
            .partial_cmp(right_x)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let label = label_parts
        .into_iter()
        .map(|(_, text)| text)
        .collect::<Vec<_>>()
        .join(" ");
    (!label.is_empty()).then_some(label)
}

fn meaningful_caption_terms(caption: &str) -> Vec<String> {
    query_terms(caption)
        .into_iter()
        .filter(|term| {
            !matches!(
                term.as_str(),
                "table"
                    | "figure"
                    | "fig"
                    | "the"
                    | "and"
                    | "with"
                    | "from"
                    | "after"
                    | "between"
                    | "comparison"
                    | "results"
                    | "result"
                    | "rate"
            ) && !term.chars().all(|ch| ch.is_ascii_digit())
        })
        .collect()
}

fn is_table_caption_text(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    lower.starts_with("table ") || lower.starts_with("table:")
}

fn cluster_table_rows(mut lines: Vec<PdfLine>) -> Vec<Vec<PdfLine>> {
    lines.sort_by(|left, right| {
        left.rect
            .map(|rect| rect.center_y())
            .unwrap_or(0.0)
            .partial_cmp(&right.rect.map(|rect| rect.center_y()).unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.line_no.cmp(&right.line_no))
    });
    let mut rows: Vec<Vec<PdfLine>> = Vec::new();
    for line in lines {
        let Some(rect) = line.rect else {
            continue;
        };
        if let Some(row) = rows.last_mut() {
            let row_y = row
                .iter()
                .filter_map(|item| item.rect.map(|rect| rect.center_y()))
                .sum::<f64>()
                / row.len().max(1) as f64;
            if (rect.center_y() - row_y).abs() <= 0.010 {
                row.push(line);
                continue;
            }
        }
        rows.push(vec![line]);
    }
    for row in &mut rows {
        row.sort_by(|left, right| {
            left.rect
                .map(|rect| rect.x1)
                .unwrap_or(0.0)
                .partial_cmp(&right.rect.map(|rect| rect.x1).unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    rows
}

fn infer_table_columns(
    data_rows: &[Vec<PdfLine>],
    header_rows: &[Vec<PdfLine>],
) -> Vec<TableColumn> {
    let mut xs = Vec::new();
    for line in data_rows
        .iter()
        .flat_map(|row| row.iter())
        .filter(|line| is_value_text(&line.text))
    {
        if let Some(rect) = line.rect {
            xs.push(rect.center_x());
        }
    }
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mut column_xs: Vec<f64> = Vec::new();
    for x in xs {
        if let Some(last) = column_xs.last_mut() {
            if (x - *last).abs() < 0.035 {
                *last = (*last + x) / 2.0;
                continue;
            }
        }
        column_xs.push(x);
    }
    let mut columns = column_xs
        .into_iter()
        .enumerate()
        .map(|(index, x)| TableColumn {
            index,
            label: String::new(),
            x,
            bbox: None,
        })
        .collect::<Vec<_>>();
    assign_column_labels(&mut columns, header_rows);
    for column in &mut columns {
        if column.label.is_empty() {
            column.label = format!("Column {}", column.index + 1);
        }
    }
    columns
}

fn assign_column_labels(columns: &mut [TableColumn], header_rows: &[Vec<PdfLine>]) {
    for row in header_rows {
        for line in row {
            let Some(rect) = line.rect else {
                continue;
            };
            if is_table_column_header_noise(&line.text) {
                continue;
            }
            if rect.center_x() < 0.33 {
                continue;
            }
            let labels = split_header_labels(&line.text);
            let covered = columns
                .iter()
                .enumerate()
                .filter(|(_, column)| column.x >= rect.x1 - 0.015 && column.x <= rect.x2 + 0.015)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if labels.len() > 1 && labels.len() <= covered.len() {
                for (label, column_index) in labels.into_iter().zip(covered) {
                    append_column_label(&mut columns[column_index], &label, rect);
                }
                continue;
            }
            if let Some(index) = nearest_column_index(rect.center_x(), columns) {
                append_column_label(&mut columns[index], &line.text, rect);
            }
        }
    }
}

fn is_table_column_header_noise(text: &str) -> bool {
    let normalized = normalize_table_label(text);
    let lower = normalized.to_lowercase();
    lower.starts_with("table ")
        || lower.starts_with("fig. ")
        || lower.starts_with("figure ")
        || lower.contains("results marked with")
        || lower.contains("highest score")
        || lower.contains("second highest")
        || lower.contains("scores are recorded")
        || lower.contains("fixing some ambiguous")
        || lower.contains("evaluated on a verified")
        || lower == "bolded"
        || lower == "†"
        || lower == ","
        || (normalized.chars().count() > 72 && !contains_known_model_header(&normalized))
}

fn append_column_label(column: &mut TableColumn, label: &str, rect: Rect) {
    let label = normalize_table_label(label);
    if label.is_empty() || is_value_text(&label) {
        return;
    }
    if column.label.is_empty() {
        column.label = label;
    } else if !column.label.to_lowercase().contains(&label.to_lowercase()) {
        column.label = format!("{} {}", column.label, label);
    }
    column.bbox = Some(column.bbox.map(|bbox| bbox.union(rect)).unwrap_or(rect));
}

fn split_header_labels(text: &str) -> Vec<String> {
    let normalized = normalize_table_label(text);
    if is_table_column_header_noise(&normalized) {
        return Vec::new();
    }
    let known = [
        "GLM-5", "GLM-4.7", "DeepSeek", "-V3.2", "Kimi", "K2.5", "Claude", "Opus 4.5", "Gemini",
        "3 Pro", "GPT-5.2", "(xhigh)",
    ];
    let matches = known
        .iter()
        .filter(|item| normalized.contains(**item))
        .map(|item| item.to_string())
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return matches;
    }
    normalized
        .split("  ")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn contains_known_model_header(text: &str) -> bool {
    [
        "GLM-5", "GLM-4.7", "DeepSeek", "Kimi", "Claude", "Gemini", "GPT-5.2",
    ]
    .iter()
    .any(|model| text.contains(model))
}

fn nearest_column_index(x: f64, columns: &[TableColumn]) -> Option<usize> {
    columns
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (left.x - x)
                .abs()
                .partial_cmp(&(right.x - x).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
}

fn insert_reconstructed_table(
    tx: &Transaction<'_>,
    document_id: &str,
    table: &ReconstructedTable,
) -> Result<(), String> {
    let table_id = format!(
        "table-{}-p{}-{}-{}",
        document_id,
        table.page_no,
        stable_fragment(&table.visual_asset_id),
        stable_fragment(&table.caption)
    );
    let visual_asset_id = table.visual_asset_id.clone();
    let bbox_json = rect_to_bbox_json(table.bbox).to_string();
    tx.execute(
        "INSERT OR REPLACE INTO document_tables
            (id, document_id, page_no, caption, bbox_json, visual_asset_id, source, confidence, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pdf_bbox', 0.74, unixepoch())",
        params![
            table_id,
            document_id,
            table.page_no,
            table.caption,
            bbox_json,
            visual_asset_id,
        ],
    )
    .map_err(|err| format!("Failed to insert reconstructed table: {err}"))?;
    tx.execute(
        "UPDATE document_visual_assets
         SET bbox_json = ?3,
             source = 'pdf_bbox',
             confidence = max(confidence, 0.74)
         WHERE id = ?1 AND document_id = ?2",
        params![visual_asset_id, document_id, bbox_json],
    )
    .map_err(|err| format!("Failed to update table visual asset bbox: {err}"))?;

    for column in &table.columns {
        let cell_id = format!("{table_id}-h-{}", column.index + 1);
        let bbox_json = column
            .bbox
            .map(rect_to_bbox_json)
            .unwrap_or_else(|| serde_json::json!([]))
            .to_string();
        tx.execute(
            "INSERT INTO document_table_cells
                (id, table_id, row_index, col_index, row_span, col_span, text, bbox_json, is_header, confidence)
             VALUES (?1, ?2, 0, ?3, 1, 1, ?4, ?5, 1, 0.68)",
            params![cell_id, table_id, column.index as u32 + 1, column.label, bbox_json],
        )
        .map_err(|err| format!("Failed to insert table header cell: {err}"))?;
    }

    for row in &table.rows {
        let row_cell_id = format!("{table_id}-r{}-label", row.index);
        tx.execute(
            "INSERT INTO document_table_cells
                (id, table_id, row_index, col_index, row_span, col_span, text, bbox_json, is_header, confidence)
             VALUES (?1, ?2, ?3, 0, 1, 1, ?4, ?5, 1, 0.70)",
            params![
                row_cell_id,
                table_id,
                row.index as u32,
                row.label,
                rect_to_bbox_json(row.bbox).to_string(),
            ],
        )
        .map_err(|err| format!("Failed to insert table row label cell: {err}"))?;
        for (cell_ordinal, cell) in row.cells.iter().enumerate() {
            let Some(column) = table.columns.get(cell.column_index) else {
                continue;
            };
            let cell_id = format!(
                "{table_id}-r{}-c{}-v{}",
                row.index,
                column.index + 1,
                cell_ordinal + 1
            );
            let bbox_json = rect_to_bbox_json(cell.bbox).to_string();
            tx.execute(
                "INSERT INTO document_table_cells
                    (id, table_id, row_index, col_index, row_span, col_span, text, bbox_json, is_header, confidence)
                 VALUES (?1, ?2, ?3, ?4, 1, 1, ?5, ?6, 0, 0.72)",
                params![
                    cell_id,
                    table_id,
                    row.index as u32,
                    column.index as u32 + 1,
                    cell.text,
                    bbox_json,
                ],
            )
            .map_err(|err| format!("Failed to insert table value cell: {err}"))?;
            let row_label = row
                .category
                .as_ref()
                .map(|category| format!("{category} / {}", row.label))
                .unwrap_or_else(|| row.label.clone());
            let fact_text = format!(
                "{} | {} | {} = {}",
                table.caption, row_label, column.label, cell.text
            );
            let fact_id = format!("{cell_id}-fact");
            tx.execute(
                "INSERT INTO document_table_facts
                    (id, document_id, table_id, page_no, row_label, column_label, value_text,
                     fact_text, bbox_json, source, confidence)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'pdf_bbox', 0.72)",
                params![
                    fact_id,
                    document_id,
                    table_id,
                    table.page_no,
                    row_label,
                    column.label,
                    cell.text,
                    fact_text,
                    bbox_json,
                ],
            )
            .map_err(|err| format!("Failed to insert table fact: {err}"))?;
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
            .map_err(|err| format!("Failed to insert table fact FTS row: {err}"))?;
        }
    }
    Ok(())
}

fn inspect_tables(
    conn: &Connection,
    document_id: &str,
    query: &str,
    limit: u32,
) -> Result<Vec<TableHit>, String> {
    ranked_table_hits(conn, document_id, query, limit, false)
}

fn resolve_table_anchors(
    conn: &Connection,
    document_id: &str,
    query: &str,
    limit: u32,
) -> Result<Vec<TableHit>, String> {
    ranked_table_hits(conn, document_id, query, limit, true)
}

fn ranked_table_hits(
    conn: &Connection,
    document_id: &str,
    query: &str,
    limit: u32,
    anchor_only: bool,
) -> Result<Vec<TableHit>, String> {
    let terms = query_terms(query);
    let requested_number = requested_table_number(query);
    let mut stmt = conn
        .prepare(
            "SELECT t.id, t.document_id, t.page_no, t.caption, t.bbox_json, t.source, t.confidence,
                    COALESCE(group_concat(f.fact_text, ' '), '') AS facts
             FROM document_tables t
             LEFT JOIN document_table_facts f ON f.table_id = t.id
             WHERE t.document_id = ?1
             GROUP BY t.id
             ORDER BY t.page_no",
        )
        .map_err(|err| format!("Failed to prepare table inspection: {err}"))?;
    let mut hits = stmt
        .query_map(params![document_id], |row| {
            let bbox_json: String = row.get(4)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            let caption = row.get::<_, String>(3)?;
            let facts = row.get::<_, String>(7)?;
            let haystack = format!("{caption} {facts}").to_lowercase();
            let mut score = relevance_score_from_terms(&haystack, &terms);
            if let Some(requested) = requested_number.as_deref() {
                if table_caption_number(&caption).as_deref() == Some(requested) {
                    score += 100.0;
                }
            }
            Ok(TableHit {
                id: row.get(0)?,
                document_id: row.get(1)?,
                page_no: row.get(2)?,
                caption,
                facts,
                bbox_list,
                source: row.get(5)?,
                confidence: row.get(6)?,
                score,
            })
        })
        .map_err(|err| format!("Failed to inspect tables: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect table hits: {err}"))?;
    if let Some(requested) = requested_number.as_deref() {
        let exact_hits = hits
            .iter()
            .filter(|hit| table_caption_number(&hit.caption).as_deref() == Some(requested))
            .cloned()
            .collect::<Vec<_>>();
        if !exact_hits.is_empty() {
            hits = exact_hits;
        } else if anchor_only {
            hits.clear();
        }
    } else if anchor_only {
        hits.clear();
    }
    hits.retain(|hit| hit.score > 0.0 || terms.is_empty());
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.page_no.cmp(&right.page_no))
    });
    hits.truncate(limit.clamp(1, 20) as usize);
    Ok(hits)
}

fn open_tables(
    conn: &Connection,
    document_id: &str,
    table_ids: &[String],
    query: &str,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    let mut candidates = Vec::new();
    let terms = query_terms(query);
    for table_id in table_ids {
        if let Some(context_candidate) =
            open_table_context_candidate(conn, document_id, table_id, &terms)?
        {
            candidates.push(context_candidate);
        }
        let mut stmt = conn
            .prepare(
                "SELECT f.id, f.document_id, f.table_id, f.page_no, t.caption,
                        f.fact_text, f.bbox_json, f.confidence
                 FROM document_table_facts f
                 JOIN document_tables t ON t.id = f.table_id
                 WHERE f.document_id = ?1 AND f.table_id = ?2
                 ORDER BY f.row_label, f.column_label
                 LIMIT 200",
            )
            .map_err(|err| format!("Failed to prepare table open: {err}"))?;
        let rows = stmt
            .query_map(params![document_id, table_id], |row| {
                let bbox_json: String = row.get(6)?;
                let bbox_list: serde_json::Value =
                    serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
                let page = row.get::<_, u32>(3)?;
                let table_id = row.get::<_, String>(2)?;
                let caption = row.get::<_, String>(4)?;
                let quote = row.get::<_, String>(5)?;
                let score = relevance_score_from_terms(
                    &format!("{caption} {quote}").to_lowercase(),
                    &terms,
                );
                Ok(EvidenceCandidate {
                    chunk_id: row.get(0)?,
                    document_id: row.get(1)?,
                    page,
                    block_id: table_id,
                    section_title: Some(caption),
                    quote,
                    bbox_list,
                    score,
                    source: "open_table".to_string(),
                    tree_node_id: None,
                    block_role: Some("table_fact".to_string()),
                })
            })
            .map_err(|err| format!("Failed to open table: {err}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("Failed to collect table facts: {err}"))?;
        candidates.extend(rows);
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.page.cmp(&right.page))
            .then_with(|| left.quote.cmp(&right.quote))
    });
    candidates.truncate(limit.clamp(1, 40) as usize);
    Ok(candidates)
}

fn open_table_context_candidate(
    conn: &Connection,
    document_id: &str,
    table_id: &str,
    terms: &[String],
) -> Result<Option<EvidenceCandidate>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, document_id, page_no, caption, bbox_json
             FROM document_tables
             WHERE document_id = ?1 AND id = ?2
             LIMIT 1",
        )
        .map_err(|err| format!("Failed to prepare table context read: {err}"))?;
    let mut rows = stmt
        .query(params![document_id, table_id])
        .map_err(|err| format!("Failed to read table context: {err}"))?;
    let Some(row) = rows
        .next()
        .map_err(|err| format!("Failed to step table context row: {err}"))?
    else {
        return Ok(None);
    };
    let table_id = row
        .get::<_, String>(0)
        .map_err(|err| format!("Failed to read table context id: {err}"))?;
    let document_id = row
        .get::<_, String>(1)
        .map_err(|err| format!("Failed to read table context document id: {err}"))?;
    let page = row
        .get::<_, u32>(2)
        .map_err(|err| format!("Failed to read table context page: {err}"))?;
    let caption = row
        .get::<_, String>(3)
        .map_err(|err| format!("Failed to read table context caption: {err}"))?;
    let bbox_json = row
        .get::<_, String>(4)
        .map_err(|err| format!("Failed to read table context bbox: {err}"))?;
    let bbox_list: serde_json::Value =
        serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
    let section_title = section_title_for_page(conn, &document_id, page)
        .unwrap_or_else(|| caption.clone())
        .trim()
        .to_string();
    let nearby_text = table_nearby_page_text(conn, &document_id, page, &bbox_list)?;
    let mut parts = Vec::new();
    if !section_title.is_empty() {
        parts.push(format!("Section: {section_title}"));
    }
    if !caption.trim().is_empty() {
        parts.push(format!("Caption: {}", caption.trim()));
    }
    if !nearby_text.trim().is_empty() {
        parts.push(format!("Nearby text:\n{}", nearby_text.trim()));
    }
    if parts.is_empty() {
        return Ok(None);
    }
    let quote = parts.join("\n\n");
    let score = 10_000.0 + relevance_score_from_terms(&quote.to_lowercase(), terms);
    Ok(Some(EvidenceCandidate {
        chunk_id: format!("table-context-{table_id}"),
        document_id,
        page,
        block_id: format!("{table_id}:context"),
        section_title: Some(section_title),
        quote,
        bbox_list,
        score,
        source: "open_table_context".to_string(),
        tree_node_id: None,
        block_role: Some("table_context".to_string()),
    }))
}

/// The on-disk crop path for an indexed visual asset by id (the `block_id` a
/// visual citation carries). Lets a vision-capable local agent see the actual
/// figure crop in Mode B.
pub(crate) fn visual_asset_image_path(
    conn: &Connection,
    document_id: &str,
    asset_id: &str,
) -> Option<String> {
    conn.query_row(
        "SELECT image_path FROM document_visual_assets
         WHERE document_id = ?1 AND id = ?2 AND image_path != '' LIMIT 1",
        params![document_id, asset_id],
        |row| row.get::<_, String>(0),
    )
    .ok()
    .filter(|path: &String| !path.trim().is_empty())
}

pub(crate) fn section_title_for_page(
    conn: &Connection,
    document_id: &str,
    page: u32,
) -> Option<String> {
    let mut stmt = conn
        .prepare(
            "SELECT title
             FROM structure_tree_nodes
             WHERE document_id = ?1
               AND page_start <= ?2
               AND page_end >= ?2
             ORDER BY level DESC, order_index DESC
             LIMIT 1",
        )
        .ok()?;
    stmt.query_row(params![document_id, page], |row| row.get::<_, String>(0))
        .ok()
}

fn table_nearby_page_text(
    conn: &Connection,
    document_id: &str,
    page: u32,
    table_bbox_list: &serde_json::Value,
) -> Result<String, String> {
    let table_rect = rect_from_bbox_list(table_bbox_list);
    let mut stmt = conn
        .prepare(
            "SELECT line_no, text, bbox_json
             FROM document_lines
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY line_no
             LIMIT 300",
        )
        .map_err(|err| format!("Failed to prepare table context lines: {err}"))?;
    let rows = stmt
        .query_map(params![document_id, page], |row| {
            let bbox_json: String = row.get(2)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?, bbox_list))
        })
        .map_err(|err| format!("Failed to read table context lines: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect table context lines: {err}"))?;
    let mut selected = Vec::new();
    for (_line_no, text, bbox_list) in rows {
        let text = normalize_for_dedupe(&text);
        if text.is_empty() || is_page_noise(&text) {
            continue;
        }
        let include = if let Some(table_rect) = table_rect {
            rect_from_bbox_list(&bbox_list).is_some_and(|line_rect| {
                line_rect.y2 >= (table_rect.y1 - 0.18).max(0.0)
                    && line_rect.y1 <= (table_rect.y2 + 0.22).min(1.0)
            })
        } else {
            selected.len() < 24
        };
        if include {
            selected.push(text);
        }
        if selected.len() >= 36 {
            break;
        }
    }
    Ok(selected.join("\n"))
}

fn search_table_facts(
    conn: &Connection,
    document_id: &str,
    query: &str,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn
        .prepare(
            "SELECT f.id, f.document_id, f.table_id, f.page_no, t.caption, f.fact_text,
                    f.bbox_json, bm25(document_table_facts_fts) AS rank
             FROM document_table_facts_fts
             JOIN document_table_facts f ON f.id = document_table_facts_fts.fact_id
             JOIN document_tables t ON t.id = f.table_id
             WHERE document_table_facts_fts.document_id = ?1
               AND document_table_facts_fts MATCH ?2
             ORDER BY rank
             LIMIT ?3",
        )
        .map_err(|err| format!("Failed to prepare table fact search: {err}"))?;
    let rows = stmt
        .query_map(
            params![document_id, escape_fts_query(query), limit.clamp(1, 40)],
            |row| {
                let bbox_json: String = row.get(6)?;
                let bbox_list: serde_json::Value =
                    serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
                let table_id = row.get::<_, String>(2)?;
                Ok(EvidenceCandidate {
                    chunk_id: row.get(0)?,
                    document_id: row.get(1)?,
                    page: row.get(3)?,
                    block_id: table_id,
                    section_title: Some(row.get(4)?),
                    quote: row.get(5)?,
                    bbox_list,
                    score: row.get(7)?,
                    source: "table_fact".to_string(),
                    tree_node_id: None,
                    block_role: Some("table_fact".to_string()),
                })
            },
        )
        .map_err(|err| format!("Failed to search table facts: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect table fact search results: {err}"))?;
    Ok(rows)
}

fn inspect_visuals(
    conn: &Connection,
    document_id: &str,
    query: &str,
    asset_type: Option<&str>,
    limit: u32,
) -> Result<Vec<VisualAssetHit>, String> {
    ranked_visual_hits(
        conn,
        document_id,
        VisualHitQuery {
            query,
            asset_type,
            page: None,
            tree_hits: &[],
            limit,
            anchor_only: false,
        },
    )
}

fn resolve_visual_anchors(
    conn: &Connection,
    document_id: &str,
    query: &str,
    asset_type: Option<&str>,
    limit: u32,
) -> Result<Vec<VisualAssetHit>, String> {
    ranked_visual_hits(
        conn,
        document_id,
        VisualHitQuery {
            query,
            asset_type,
            page: None,
            tree_hits: &[],
            limit,
            anchor_only: true,
        },
    )
}

fn inspect_objects(
    conn: &Connection,
    document_id: &str,
    query: &str,
    asset_type: Option<&str>,
    page: Option<u32>,
    tree_hits: &[StructureTreeHit],
    limit: u32,
) -> Result<Vec<VisualAssetHit>, String> {
    ranked_visual_hits(
        conn,
        document_id,
        VisualHitQuery {
            query,
            asset_type,
            page,
            tree_hits,
            limit,
            anchor_only: false,
        },
    )
}

fn ranked_visual_hits(
    conn: &Connection,
    document_id: &str,
    options: VisualHitQuery<'_>,
) -> Result<Vec<VisualAssetHit>, String> {
    let terms = query_terms(options.query);
    let asset_type = options
        .asset_type
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_lowercase);
    let requested_anchor = requested_visual_anchor(options.query);
    let mut stmt = conn
        .prepare(
            "SELECT id, document_id, page_no, asset_type, caption, bbox_json, image_path,
                    nearby_text, ocr_text, source, confidence
             FROM document_visual_assets
             WHERE document_id = ?1
             ORDER BY page_no",
        )
        .map_err(|err| format!("Failed to prepare visual inspection: {err}"))?;
    let mut hits = stmt
        .query_map(params![document_id], |row| {
            let bbox_json: String = row.get(5)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            let caption = row.get::<_, String>(4)?;
            let nearby_text = row.get::<_, String>(7)?;
            // OCR text recovered from inside the figure/chart/image crop makes
            // the picture's own words searchable, alongside its caption and the
            // surrounding body text.
            let ocr_text = row.get::<_, String>(8)?;
            let haystack = format!("{caption} {nearby_text} {ocr_text}").to_lowercase();
            let mut score = relevance_score_from_terms(&haystack, &terms);
            if let Some(requested) = requested_anchor.as_ref() {
                if visual_caption_number(&caption).as_deref() == Some(requested.number.as_str()) {
                    score += 100.0;
                }
            }
            Ok(VisualAssetHit {
                id: row.get(0)?,
                document_id: row.get(1)?,
                page_no: row.get(2)?,
                asset_type: row.get(3)?,
                caption,
                bbox_list,
                caption_bbox_list: serde_json::json!([]),
                image_path: row.get(6)?,
                nearby_text,
                ocr_text,
                source: row.get(9)?,
                confidence: row.get(10)?,
                score,
            })
        })
        .map_err(|err| format!("Failed to inspect visuals: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect visual hits: {err}"))?;
    let location_scoped = options.page.is_some() || !options.tree_hits.is_empty();
    hits.retain(|hit| {
        let type_matches = asset_type
            .as_deref()
            .map(|wanted| hit.asset_type == wanted)
            .unwrap_or(true);
        let page_matches = options
            .page
            .map(|wanted| hit.page_no == wanted)
            .unwrap_or(true);
        let tree_matches = options.tree_hits.is_empty()
            || options
                .tree_hits
                .iter()
                .any(|node| hit.page_no >= node.page && hit.page_no <= node.page_end);
        type_matches
            && page_matches
            && tree_matches
            && (hit.score > 0.0 || terms.is_empty() || location_scoped)
    });
    if let Some(requested) = requested_anchor.as_ref() {
        let exact_hits = hits
            .iter()
            .filter(|hit| {
                visual_caption_number(&hit.caption).as_deref() == Some(requested.number.as_str())
                    && requested
                        .asset_type
                        .as_deref()
                        .map(|wanted| visual_anchor_asset_type_matches(wanted, &hit.asset_type))
                        .unwrap_or(true)
            })
            .cloned()
            .collect::<Vec<_>>();
        if !exact_hits.is_empty() {
            hits = exact_hits;
        } else if options.anchor_only {
            hits.clear();
        }
    } else if options.anchor_only {
        hits.clear();
    }
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.page_no.cmp(&right.page_no))
    });
    hits.truncate(options.limit.clamp(1, 20) as usize);
    Ok(hits)
}

fn open_visuals(
    conn: &Connection,
    document_id: &str,
    asset_id: Option<&str>,
    query: &str,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    let hits = if let Some(asset_id) = asset_id.map(str::trim).filter(|value| !value.is_empty()) {
        let mut stmt = conn
            .prepare(
                "SELECT id, document_id, page_no, asset_type, caption, bbox_json, image_path,
                        nearby_text, ocr_text, source, confidence
                 FROM document_visual_assets
                 WHERE document_id = ?1 AND id = ?2
                 LIMIT 1",
            )
            .map_err(|err| format!("Failed to prepare visual open: {err}"))?;
        let rows = stmt.query_map(params![document_id, asset_id], |row| {
            let bbox_json: String = row.get(5)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            Ok(VisualAssetHit {
                id: row.get(0)?,
                document_id: row.get(1)?,
                page_no: row.get(2)?,
                asset_type: row.get(3)?,
                caption: row.get(4)?,
                bbox_list,
                caption_bbox_list: serde_json::json!([]),
                image_path: row.get(6)?,
                nearby_text: row.get(7)?,
                ocr_text: row.get(8)?,
                source: row.get(9)?,
                confidence: row.get(10)?,
                score: 1.0,
            })
        });
        let collected = rows
            .map_err(|err| format!("Failed to open visual asset: {err}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("Failed to collect visual asset: {err}"))?;
        collected
    } else {
        inspect_visuals(conn, document_id, query, None, limit)?
    };
    Ok(hits.iter().map(visual_hit_to_open_candidate).collect())
}

fn table_hit_to_candidate(hit: &TableHit) -> EvidenceCandidate {
    let facts = if hit.facts.trim().is_empty() {
        "No structured table facts were extracted.".to_string()
    } else {
        hit.facts.clone()
    };
    EvidenceCandidate {
        chunk_id: hit.id.clone(),
        document_id: hit.document_id.clone(),
        page: hit.page_no,
        block_id: hit.id.clone(),
        section_title: Some(hit.caption.clone()),
        quote: format!(
            "Indexed table on page {}: {}\nFacts: {}\nsource={} confidence={:.2}",
            hit.page_no, hit.caption, facts, hit.source, hit.confidence
        ),
        bbox_list: hit.bbox_list.clone(),
        score: hit.score,
        source: "inspect_tables".to_string(),
        tree_node_id: None,
        block_role: Some("table".to_string()),
    }
}

fn table_hit_to_anchor_candidate(hit: &TableHit) -> EvidenceCandidate {
    EvidenceCandidate {
        chunk_id: format!("table-anchor-{}", hit.id),
        document_id: hit.document_id.clone(),
        page: hit.page_no,
        block_id: hit.id.clone(),
        section_title: Some(hit.caption.clone()),
        quote: format!(
            "Resolved table anchor: {} on page {}\ntableId={}\nsource={} confidence={:.2}",
            hit.caption, hit.page_no, hit.id, hit.source, hit.confidence
        ),
        bbox_list: hit.bbox_list.clone(),
        score: hit.score,
        source: "table_anchor".to_string(),
        tree_node_id: None,
        block_role: Some("table".to_string()),
    }
}

fn visual_hit_to_anchor_candidate(hit: &VisualAssetHit) -> EvidenceCandidate {
    let crop = if hit.image_path.trim().is_empty() {
        "cropPath unavailable"
    } else {
        hit.image_path.as_str()
    };
    EvidenceCandidate {
        chunk_id: format!("visual-anchor-{}", hit.id),
        document_id: hit.document_id.clone(),
        page: hit.page_no,
        block_id: hit.id.clone(),
        section_title: Some(hit.caption.clone()),
        quote: format!(
            "Resolved visual anchor: {} on page {}\nassetId={}\nassetType={}\nsource={} confidence={:.2}\n{}",
            hit.caption, hit.page_no, hit.id, hit.asset_type, hit.source, hit.confidence, crop
        ),
        bbox_list: hit.bbox_list.clone(),
        score: hit.score,
        source: "visual_anchor".to_string(),
        tree_node_id: None,
        block_role: Some(hit.asset_type.clone()),
    }
}

fn visual_hit_to_object_candidate(hit: &VisualAssetHit) -> EvidenceCandidate {
    let crop = if hit.image_path.trim().is_empty() {
        "cropPath unavailable"
    } else {
        hit.image_path.as_str()
    };
    EvidenceCandidate {
        chunk_id: format!("object-{}", hit.id),
        document_id: hit.document_id.clone(),
        page: hit.page_no,
        block_id: hit.id.clone(),
        section_title: Some(format!("{} on page {}", hit.asset_type, hit.page_no)),
        quote: format!(
            "Indexed object: {}\nassetId={}\nPage: {}\nCaption: {}\nNearby text: {}\n{}",
            hit.asset_type, hit.id, hit.page_no, hit.caption, hit.nearby_text, crop
        ),
        bbox_list: hit.bbox_list.clone(),
        score: hit.score,
        source: "inspect_objects".to_string(),
        tree_node_id: None,
        block_role: Some(hit.asset_type.clone()),
    }
}

/// Render the in-image OCR text as a labeled evidence line, or nothing when
/// empty (most images have no recoverable text, so we don't add a blank field).
fn visual_ocr_text_suffix(ocr_text: &str) -> String {
    let trimmed = ocr_text.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!("\nText in image (OCR): {trimmed}")
    }
}

fn visual_hit_to_candidate(hit: &VisualAssetHit) -> EvidenceCandidate {
    let crop = if hit.image_path.trim().is_empty() {
        "cropPath unavailable"
    } else {
        hit.image_path.as_str()
    };
    EvidenceCandidate {
        chunk_id: hit.id.clone(),
        document_id: hit.document_id.clone(),
        page: hit.page_no,
        block_id: hit.id.clone(),
        section_title: Some(format!("{} on page {}", hit.asset_type, hit.page_no)),
        quote: format!(
            "Visual asset: {}\nCaption: {}\nNearby text: {}{}\n{}",
            hit.asset_type,
            hit.caption,
            hit.nearby_text,
            visual_ocr_text_suffix(&hit.ocr_text),
            crop
        ),
        bbox_list: hit.bbox_list.clone(),
        score: hit.score,
        source: "visual_asset".to_string(),
        tree_node_id: None,
        block_role: Some(hit.asset_type.clone()),
    }
}

fn visual_hit_to_open_candidate(hit: &VisualAssetHit) -> EvidenceCandidate {
    let crop = if hit.image_path.trim().is_empty() {
        "cropPath unavailable"
    } else {
        hit.image_path.as_str()
    };
    EvidenceCandidate {
        chunk_id: format!("open-visual-{}", hit.id),
        document_id: hit.document_id.clone(),
        page: hit.page_no,
        block_id: hit.id.clone(),
        section_title: Some(format!("{} on page {}", hit.asset_type, hit.page_no)),
        quote: format!(
            "Opened visual asset: {}\nassetId={}\nCaption: {}\nNearby text: {}{}\n{}",
            hit.asset_type,
            hit.id,
            hit.caption,
            hit.nearby_text,
            visual_ocr_text_suffix(&hit.ocr_text),
            crop
        ),
        bbox_list: hit.bbox_list.clone(),
        score: hit.score,
        source: "open_visual".to_string(),
        tree_node_id: None,
        block_role: Some(hit.asset_type.clone()),
    }
}

fn relevance_score_from_terms(haystack: &str, terms: &[String]) -> f64 {
    if terms.is_empty() {
        return 1.0;
    }
    terms
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .count() as f64
}

fn rect_from_bbox_list(value: &serde_json::Value) -> Option<Rect> {
    value
        .as_array()?
        .iter()
        .filter_map(|item| item.as_array())
        .filter_map(|bbox| {
            let x1 = bbox.first()?.as_f64()?;
            let y1 = bbox.get(1)?.as_f64()?;
            let x2 = bbox.get(2)?.as_f64()?;
            let y2 = bbox.get(3)?.as_f64()?;
            (x2 > x1 && y2 > y1).then_some(Rect { x1, y1, x2, y2 })
        })
        .reduce(Rect::union)
}

fn rect_to_bbox_json(rect: Rect) -> serde_json::Value {
    serde_json::json!([[rect.x1, rect.y1, rect.x2, rect.y2]])
}

fn is_row_label_text(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.chars().any(|ch| ch.is_alphabetic())
        && trimmed.chars().count() >= 3
        && !trimmed.to_lowercase().starts_with("table")
}

fn is_category_text(text: &str) -> bool {
    let normalized = text.trim();
    let words = normalized.split_whitespace().count();
    words <= 4
        && normalized.chars().any(|ch| ch.is_alphabetic())
        && !is_value_text(normalized)
        && !normalized.to_lowercase().starts_with("table")
}

fn is_value_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed == "-" || trimmed == "†" {
        return true;
    }
    trimmed.chars().any(|ch| ch.is_ascii_digit())
        && trimmed.chars().all(|ch| {
            ch.is_ascii_digit()
                || ch.is_whitespace()
                || matches!(
                    ch,
                    '.' | ',' | '-' | '/' | '%' | '*' | '†' | '(' | ')' | '+' | ':'
                )
        })
}

fn normalize_table_label(text: &str) -> String {
    text.replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|ch: char| matches!(ch, ',' | ';' | '|'))
        .to_string()
}

fn normalize_table_value(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn stable_fragment(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn read_page_blocks(
    conn: &Connection,
    document_id: &str,
    page: u32,
    mode: PageOpenMode,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    if mode == PageOpenMode::Full {
        let line_chunks = read_page_line_chunks(conn, document_id, page, limit)?;
        if !line_chunks.is_empty() {
            return Ok(line_chunks);
        }
    }

    let mut stmt = conn
        .prepare(
            "SELECT id, document_id, page_no, block_index, text, bbox_json, block_role
             FROM document_blocks
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY block_index
             LIMIT 80",
        )
        .map_err(|err| format!("Failed to prepare page block read: {err}"))?;

    let rows = stmt
        .query_map(params![document_id, page], |row| {
            let bbox_json: String = row.get(5)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            let block_id: String = row.get(0)?;
            Ok(EvidenceCandidate {
                chunk_id: format!("page-block-{block_id}"),
                document_id: row.get(1)?,
                page: row.get(2)?,
                block_id,
                section_title: None,
                quote: row.get(4)?,
                bbox_list,
                score: row.get::<_, u32>(3)? as f64,
                source: "open_pages".to_string(),
                tree_node_id: None,
                block_role: Some(row.get(6)?),
            })
        })
        .map_err(|err| format!("Failed to read page blocks: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect page blocks: {err}"))?;
    Ok(match mode {
        PageOpenMode::Overview => compact_page_blocks(document_id, page, rows, limit),
        PageOpenMode::Header => compact_header_blocks(document_id, page, rows),
        PageOpenMode::Full => full_page_blocks(rows, limit),
    })
}

fn compact_page_blocks(
    document_id: &str,
    page: u32,
    rows: Vec<EvidenceCandidate>,
    limit: u32,
) -> Vec<EvidenceCandidate> {
    let max_parts = limit.clamp(1, 6) as usize;
    let mut parts = Vec::new();
    let mut bbox_refs = Vec::new();

    for row in rows {
        let text = normalize_for_dedupe(&row.quote);
        if text.is_empty() || is_page_noise(&text) {
            continue;
        }
        let is_header_role = matches!(
            row.block_role.as_deref(),
            Some("title" | "authors" | "affiliations" | "abstract")
        );
        let is_title_like = parts.is_empty() && text.chars().count() >= 12;
        let is_meaningful = is_header_role
            || text.chars().count() >= 80
            || text.to_lowercase().contains("abstract")
            || text.to_lowercase().contains("introduction")
            || is_title_like;
        if !is_meaningful {
            continue;
        }
        append_bbox_refs(&mut bbox_refs, &row.bbox_list);
        parts.push(text);
        if parts.len() >= max_parts {
            break;
        }
    }

    if parts.is_empty() {
        return Vec::new();
    }

    vec![EvidenceCandidate {
        chunk_id: format!("page-overview-{page}"),
        document_id: document_id.to_string(),
        page,
        block_id: format!("page-overview-{page}"),
        section_title: Some(format!("Page {page} overview")),
        quote: parts.join("\n\n"),
        bbox_list: serde_json::Value::Array(bbox_refs),
        score: 0.0,
        source: "open_pages".to_string(),
        tree_node_id: None,
        block_role: Some("page_overview".to_string()),
    }]
}

fn compact_header_blocks(
    document_id: &str,
    page: u32,
    rows: Vec<EvidenceCandidate>,
) -> Vec<EvidenceCandidate> {
    let mut parts = Vec::new();
    let mut bbox_refs = Vec::new();

    for row in rows.into_iter().filter(|row| {
        matches!(
            row.block_role.as_deref(),
            Some("title" | "authors" | "affiliations")
        )
    }) {
        let text = normalize_for_dedupe(&row.quote);
        if text.is_empty() {
            continue;
        }
        append_bbox_refs(&mut bbox_refs, &row.bbox_list);
        parts.push(text);
    }

    if parts.is_empty() {
        return Vec::new();
    }

    vec![EvidenceCandidate {
        chunk_id: format!("page-header-{page}"),
        document_id: document_id.to_string(),
        page,
        block_id: format!("page-header-{page}"),
        section_title: Some(format!("Page {page} header")),
        quote: parts.join("\n"),
        bbox_list: serde_json::Value::Array(bbox_refs),
        score: 0.0,
        source: "open_pages".to_string(),
        tree_node_id: None,
        block_role: Some("page_header".to_string()),
    }]
}

fn full_page_blocks(rows: Vec<EvidenceCandidate>, limit: u32) -> Vec<EvidenceCandidate> {
    rows.into_iter()
        .filter(|row| {
            let text = normalize_for_dedupe(&row.quote);
            !text.is_empty() && !is_page_noise(&text)
        })
        .take(limit.clamp(1, 80) as usize)
        .collect()
}

fn read_page_line_chunks(
    conn: &Connection,
    document_id: &str,
    page: u32,
    limit: u32,
) -> Result<Vec<EvidenceCandidate>, String> {
    const LINES_PER_CHUNK: usize = 12;

    let mut stmt = conn
        .prepare(
            "SELECT id, document_id, page_no, line_no, text, bbox_json
             FROM document_lines
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY line_no
             LIMIT 300",
        )
        .map_err(|err| format!("Failed to prepare page line read: {err}"))?;
    let rows = stmt
        .query_map(params![document_id, page], |row| {
            let bbox_json: String = row.get(5)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, u32>(2)?,
                row.get::<_, u32>(3)?,
                row.get::<_, String>(4)?,
                bbox_list,
            ))
        })
        .map_err(|err| format!("Failed to read page lines: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect page lines: {err}"))?;

    let max_chunks = limit.clamp(1, 80) as usize;
    let mut chunks = Vec::new();
    let mut parts: Vec<String> = Vec::new();
    let mut bbox_refs = Vec::new();
    let mut start_line = 0;
    let mut end_line = 0;
    let mut document_id_owned = document_id.to_string();

    for (_id, row_document_id, row_page, line_no, text, bbox_list) in rows {
        let text = normalize_for_dedupe(&text);
        if text.is_empty() || is_page_noise(&text) {
            continue;
        }
        if parts.is_empty() {
            start_line = line_no;
            document_id_owned = row_document_id;
        }
        end_line = line_no;
        append_bbox_refs(&mut bbox_refs, &bbox_list);
        parts.push(text);
        if parts.len() >= LINES_PER_CHUNK {
            chunks.push(page_line_chunk_candidate(
                &document_id_owned,
                row_page,
                start_line,
                end_line,
                chunks.len(),
                &parts,
                &bbox_refs,
            ));
            parts.clear();
            bbox_refs.clear();
            if chunks.len() >= max_chunks {
                return Ok(chunks);
            }
        }
    }

    if !parts.is_empty() && chunks.len() < max_chunks {
        chunks.push(page_line_chunk_candidate(
            &document_id_owned,
            page,
            start_line,
            end_line,
            chunks.len(),
            &parts,
            &bbox_refs,
        ));
    }

    Ok(chunks)
}

fn page_line_chunk_candidate(
    document_id: &str,
    page: u32,
    start_line: u32,
    end_line: u32,
    chunk_index: usize,
    parts: &[String],
    bbox_refs: &[serde_json::Value],
) -> EvidenceCandidate {
    let block_id = format!("page-lines-{page}-{start_line}-{end_line}");
    EvidenceCandidate {
        chunk_id: block_id.clone(),
        document_id: document_id.to_string(),
        page,
        block_id,
        section_title: Some(format!("Page {page} lines {start_line}-{end_line}")),
        quote: parts.join("\n"),
        bbox_list: serde_json::Value::Array(bbox_refs.to_vec()),
        score: chunk_index as f64,
        source: "open_pages".to_string(),
        tree_node_id: None,
        block_role: Some("page_lines".to_string()),
    }
}

fn append_bbox_refs(target: &mut Vec<serde_json::Value>, bbox_list: &serde_json::Value) {
    if let Some(items) = bbox_list.as_array() {
        target.extend(items.iter().cloned());
    }
}

fn is_page_noise(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.chars().count() <= 2 {
        return true;
    }
    trimmed
        .chars()
        .all(|ch| ch.is_ascii_digit() || ch.is_whitespace() || matches!(ch, '.' | '-' | '/'))
}

fn inspect_tree(
    conn: &Connection,
    document_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<StructureTreeHit>, String> {
    let query_terms = query_terms(query);
    if query_terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut stmt = conn
        .prepare(
            "SELECT id, title, keywords_json, level, order_index, page_start, page_end, block_start_index
             FROM structure_tree_nodes
             WHERE document_id = ?1 AND level > 0
             ORDER BY order_index",
        )
        .map_err(|err| format!("Failed to prepare tree inspection: {err}"))?;
    let rows = stmt
        .query_map(params![document_id], |row| {
            let keywords_json: String = row.get(2)?;
            let keywords: Vec<String> =
                serde_json::from_str(&keywords_json).unwrap_or_else(|_| Vec::new());
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                keywords,
                row.get::<_, u32>(3)?,
                row.get::<_, u32>(4)?,
                row.get::<_, u32>(5)?,
                row.get::<_, u32>(6)?,
                row.get::<_, u32>(7)?,
            ))
        })
        .map_err(|err| format!("Failed to inspect structure tree: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect structure tree rows: {err}"))?;

    let mut hits = rows
        .into_iter()
        .filter_map(
            |(id, title, keywords, level, order_index, page, page_end, block_index)| {
                let haystack = format!("{} {}", title, keywords.join(" ")).to_lowercase();
                let mut score = query_terms
                    .iter()
                    .filter(|term| haystack.contains(term.as_str()))
                    .count() as f64;
                if score <= 0.0 {
                    return None;
                }
                score += (6_u32.saturating_sub(level) as f64) * 0.05;
                score += 1.0 / ((order_index + 1) as f64 * 100.0);
                Some(StructureTreeHit {
                    id,
                    title,
                    page,
                    page_end,
                    block_index,
                    score,
                })
            },
        )
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(limit);
    Ok(hits)
}

fn lookup_tree_hits(
    conn: &Connection,
    document_id: &str,
    node_ids: &[String],
) -> Result<Vec<StructureTreeHit>, String> {
    if node_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut hits = Vec::new();
    for node_id in node_ids {
        let hit = conn
            .query_row(
                "SELECT id, title, page_start, page_end, block_start_index FROM structure_tree_nodes
                 WHERE document_id = ?1 AND id = ?2 AND level > 0",
                params![document_id, node_id],
                |row| {
                    Ok(StructureTreeHit {
                        id: row.get(0)?,
                        title: row.get(1)?,
                        page: row.get(2)?,
                        page_end: row.get(3)?,
                        block_index: row.get(4)?,
                        score: 1.0,
                    })
                },
            )
            .map_err(|err| format!("Failed to look up tree node {node_id}: {err}"))?;
        hits.push(hit);
    }
    Ok(hits)
}

fn open_sections(
    conn: &Connection,
    document_id: &str,
    tree_hits: &[StructureTreeHit],
    per_section_limit: u32,
    query: &str,
) -> Result<Vec<EvidenceCandidate>, String> {
    let mut candidates = Vec::new();
    let query_terms = query_terms(query);
    for hit in tree_hits {
        let mut stmt = conn
            .prepare(
                "SELECT b.id, b.document_id, b.page_no, b.block_index, b.text, b.bbox_json, b.block_role
                 FROM structure_tree_nodes n
                 JOIN document_blocks b
                   ON b.document_id = n.document_id
                  AND b.page_no BETWEEN n.page_start AND n.page_end
                  AND (b.page_no > n.page_start OR b.block_index >= n.block_start_index)
                  AND (b.page_no < n.page_end OR b.block_index <= n.block_end_index)
                 WHERE n.document_id = ?1 AND n.id = ?2
                 ORDER BY b.page_no, b.block_index
                 LIMIT ?3",
            )
            .map_err(|err| format!("Failed to prepare section open: {err}"))?;
        let scan_limit = per_section_limit.saturating_mul(6).clamp(20, 120);
        let rows = stmt
            .query_map(params![document_id, hit.id, scan_limit], |row| {
                let bbox_json: String = row.get(5)?;
                let bbox_list: serde_json::Value =
                    serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
                let block_id: String = row.get(0)?;
                let page = row.get::<_, u32>(2)?;
                let block_index = row.get::<_, u32>(3)?;
                let text = row.get::<_, String>(4)?;
                let block_role = row.get::<_, String>(6)?;
                let relevance = section_block_relevance_score(&text, &block_role, &query_terms);
                Ok((
                    relevance,
                    page,
                    block_index,
                    EvidenceCandidate {
                        chunk_id: format!("section-{}-{block_id}", hit.id),
                        document_id: row.get(1)?,
                        page,
                        block_id,
                        section_title: Some(hit.title.clone()),
                        quote: format!("Section: {}\n{}", hit.title, text),
                        bbox_list,
                        score: hit.score + relevance,
                        source: "open_section".to_string(),
                        tree_node_id: Some(hit.id.clone()),
                        block_role: Some(block_role),
                    },
                ))
            })
            .map_err(|err| format!("Failed to open section: {err}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("Failed to collect section blocks: {err}"))?;
        let mut selected = rows;
        if selected.iter().any(|(score, _, _, _)| *score > 0.0) {
            selected.sort_by(|left, right| {
                right
                    .0
                    .partial_cmp(&left.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.1.cmp(&right.1))
                    .then_with(|| left.2.cmp(&right.2))
            });
        }
        selected.truncate(per_section_limit.clamp(1, 20) as usize);
        selected.sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.2.cmp(&right.2)));
        candidates.extend(selected.into_iter().map(|(_, _, _, candidate)| candidate));
    }
    Ok(interleave_section_candidates(candidates))
}

fn interleave_section_candidates(candidates: Vec<EvidenceCandidate>) -> Vec<EvidenceCandidate> {
    let mut groups: Vec<(String, Vec<EvidenceCandidate>)> = Vec::new();
    for candidate in candidates {
        let key = candidate
            .tree_node_id
            .clone()
            .unwrap_or_else(|| candidate.section_title.clone().unwrap_or_default());
        if let Some((_, group)) = groups.iter_mut().find(|(group_key, _)| *group_key == key) {
            group.push(candidate);
        } else {
            groups.push((key, vec![candidate]));
        }
    }
    if groups.len() <= 1 {
        return groups
            .into_iter()
            .flat_map(|(_, group)| group)
            .collect::<Vec<_>>();
    }

    let total = groups.iter().map(|(_, group)| group.len()).sum::<usize>();
    let mut result = Vec::with_capacity(total);
    let mut index = 0;
    while result.len() < total {
        let mut advanced = false;
        for (_, group) in &groups {
            if let Some(candidate) = group.get(index) {
                result.push(candidate.clone());
                advanced = true;
            }
        }
        if !advanced {
            break;
        }
        index += 1;
    }
    result
}

fn read_tree_node_lines(
    conn: &Connection,
    document_id: &str,
    tree_hits: &[StructureTreeHit],
    line_limit: u32,
    query: &str,
) -> Result<Vec<EvidenceCandidate>, String> {
    let mut candidates = Vec::new();
    let query_terms = query_terms(query);
    let line_limit = line_limit.clamp(1, 30);
    for hit in tree_hits {
        let mut stmt = conn
            .prepare(
                "SELECT l.id, l.document_id, l.page_no, l.line_no, l.text, l.bbox_json, l.block_id
                 FROM structure_tree_nodes n
                 JOIN document_lines l
                   ON l.document_id = n.document_id
                  AND l.page_no BETWEEN n.page_start AND n.page_end
                  AND (
                    (l.block_index = 0 AND n.block_start_index = 0 AND n.block_end_index = 0)
                    OR (
                      l.block_index > 0
                      AND
                      (l.page_no > n.page_start OR l.block_index >= n.block_start_index)
                      AND (l.page_no < n.page_end OR l.block_index <= n.block_end_index)
                    )
                  )
                 WHERE n.document_id = ?1 AND n.id = ?2
                 ORDER BY l.page_no, l.line_no
                 LIMIT ?3",
            )
            .map_err(|err| format!("Failed to prepare tree node line read: {err}"))?;
        let scan_limit = line_limit.saturating_mul(12).clamp(40, 240);
        let rows = stmt
            .query_map(params![document_id, hit.id, scan_limit], |row| {
                let line_id: String = row.get(0)?;
                let page = row.get::<_, u32>(2)?;
                let line_no = row.get::<_, u32>(3)?;
                let text = row.get::<_, String>(4)?;
                let bbox_json: String = row.get(5)?;
                let block_id: String = row.get(6)?;
                let bbox_list: serde_json::Value =
                    serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
                let relevance = section_block_relevance_score(&text, "line", &query_terms);
                Ok((
                    relevance,
                    page,
                    line_no,
                    EvidenceCandidate {
                        chunk_id: format!("tree-line-{}-{line_id}", hit.id),
                        document_id: row.get(1)?,
                        page,
                        block_id: if block_id.is_empty() {
                            line_id
                        } else {
                            block_id
                        },
                        section_title: Some(hit.title.clone()),
                        quote: format!("Section: {}\nLine {line_no}: {text}", hit.title),
                        bbox_list,
                        score: hit.score + relevance,
                        source: "read_tree_node_lines".to_string(),
                        tree_node_id: Some(hit.id.clone()),
                        block_role: Some("line".to_string()),
                    },
                ))
            })
            .map_err(|err| format!("Failed to read tree node lines: {err}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("Failed to collect tree node lines: {err}"))?;
        let mut selected = rows;
        if selected.iter().any(|(score, _, _, _)| *score > 0.0) {
            selected.sort_by(|left, right| {
                right
                    .0
                    .partial_cmp(&left.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.1.cmp(&right.1))
                    .then_with(|| left.2.cmp(&right.2))
            });
        }
        selected.truncate(line_limit as usize);
        selected.sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.2.cmp(&right.2)));
        candidates.extend(selected.into_iter().map(|(_, _, _, candidate)| candidate));
    }
    Ok(candidates)
}

fn section_block_relevance_score(text: &str, block_role: &str, query_terms: &[String]) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let mut score = query_terms
        .iter()
        .filter(|term| lower.contains(term.as_str()))
        .count() as f64;
    if score > 0.0 {
        if matches!(block_role, "heading" | "caption") {
            score += 0.25;
        }
        if text.chars().filter(|ch| ch.is_alphabetic()).count() < 20 {
            score -= 0.2;
        }
    }
    score.max(0.0)
}

fn dedupe_candidates(
    candidates: Vec<EvidenceCandidate>,
    max_candidates: usize,
) -> Vec<EvidenceCandidate> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for candidate in candidates {
        let key = if candidate.block_id.is_empty() {
            format!(
                "{}:{}",
                candidate.page,
                normalize_for_dedupe(&candidate.quote)
            )
        } else if matches!(candidate.source.as_str(), "table_fact" | "open_table") {
            format!(
                "{}:{}:{}",
                candidate.document_id,
                candidate.block_id,
                normalize_for_dedupe(&candidate.quote)
            )
        } else {
            format!("{}:{}", candidate.document_id, candidate.block_id)
        };
        if seen.insert(key) {
            result.push(candidate);
        }
    }
    result.truncate(max_candidates.max(1));
    result
}

fn candidates_to_citations(
    candidates: &[EvidenceCandidate],
    max_quote_chars: usize,
) -> Vec<Citation> {
    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| Citation {
            id: format!("rag-c-{}-{}", candidate.chunk_id, index),
            label: format!("[{}]", index + 1),
            page: candidate.page,
            block_id: candidate.block_id.clone(),
            section_title: candidate.section_title.clone(),
            quote: truncate_chars(&candidate.quote, max_quote_chars),
            bbox_list: candidate.bbox_list.clone(),
            document_id: candidate.document_id.clone(),
            source: candidate.source.clone(),
        })
        .collect()
}

fn default_citation_quote_chars() -> usize {
    crate::model_catalog::ModelContextBudget::default().max_quote_chars
}

fn trace_candidates_from_candidates(
    candidates: &[EvidenceCandidate],
) -> Vec<RetrievalTraceCandidate> {
    candidates
        .iter()
        .map(|candidate| RetrievalTraceCandidate {
            source: candidate.source.clone(),
            page: candidate.page,
            block_id: candidate.block_id.clone(),
            tree_node_id: candidate.tree_node_id.clone(),
            section_title: candidate.section_title.clone(),
            quote: truncate_chars(&candidate.quote, 240),
        })
        .collect()
}

fn citation_dedupe_key(citation: &Citation) -> String {
    if citation.block_id.is_empty() {
        format!(
            "{}:{}",
            citation.page,
            normalize_for_dedupe(&citation.quote)
        )
    } else if matches!(citation.source.as_str(), "table_fact" | "open_table") {
        format!(
            "{}:{}:{}",
            citation.document_id,
            citation.block_id,
            normalize_for_dedupe(&citation.quote)
        )
    } else {
        format!("{}:{}", citation.document_id, citation.block_id)
    }
}

fn relabel_citations(citations: &mut [Citation]) {
    for (index, citation) in citations.iter_mut().enumerate() {
        citation.label = format!("[{}]", index + 1);
    }
}

fn build_prompt_context(
    citations: &[Citation],
    budget: &crate::model_catalog::ModelContextBudget,
) -> String {
    let mut context = String::new();
    for citation in citations {
        if context.len() >= budget.max_context_chars {
            break;
        }
        let entry = format!(
            "{} page {} source={}:\n{}\n\n",
            citation.label, citation.page, citation.source, citation.quote
        );
        if context.len() + entry.len() > budget.max_context_chars {
            let remaining = budget.max_context_chars.saturating_sub(context.len());
            context.push_str(&truncate_chars(&entry, remaining));
            break;
        }
        context.push_str(&entry);
    }
    context
}

fn record_retrieval_run(
    conn: &Connection,
    run_id: &str,
    request: &RetrievalRequest<'_>,
    intent: &str,
    tree_node_ids: &[String],
    candidates: &[EvidenceCandidate],
    finalize_gate: &serde_json::Value,
) -> Result<(), String> {
    let fts_ids = candidates
        .iter()
        .filter(|candidate| candidate.source == "fts")
        .map(|candidate| candidate.chunk_id.as_str())
        .collect::<Vec<_>>();
    let selected_ids = candidates
        .iter()
        .map(|candidate| candidate.chunk_id.as_str())
        .collect::<Vec<_>>();
    let selected_tree_node_ids = candidates
        .iter()
        .filter_map(|candidate| candidate.tree_node_id.as_deref())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let tree_node_ids = if tree_node_ids.is_empty() {
        selected_tree_node_ids
    } else {
        tree_node_ids.iter().map(String::as_str).collect::<Vec<_>>()
    };
    conn.execute(
        "INSERT INTO retrieval_evidence_runs
            (id, document_id, question, intent, tree_node_ids_json, fts_candidate_ids_json,
             wiki_candidate_ids_json, vector_candidate_ids_json, selected_candidate_ids_json,
             finalize_gate_json, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, '[]', '[]', ?7, ?8, unixepoch())",
        params![
            run_id,
            request.document_id,
            request.question,
            intent,
            serde_json::to_string(&tree_node_ids)
                .map_err(|err| format!("Failed to encode tree candidates: {err}"))?,
            serde_json::to_string(&fts_ids)
                .map_err(|err| format!("Failed to encode FTS candidates: {err}"))?,
            serde_json::to_string(&selected_ids)
                .map_err(|err| format!("Failed to encode selected candidates: {err}"))?,
            finalize_gate.to_string(),
        ],
    )
    .map_err(|err| format!("Failed to record retrieval run: {err}"))?;
    Ok(())
}

fn infer_intent(question: &str) -> String {
    let question = question.to_lowercase();
    if question.contains("compare") || question.contains("区别") || question.contains("对比") {
        "compare"
    } else if question.contains("where") || question.contains("在哪") || question.contains("位置")
    {
        "locate"
    } else if question.contains("summarize")
        || question.contains("总结")
        || question.contains("概括")
    {
        "summarize"
    } else {
        "explain"
    }
    .to_string()
}

fn build_finalize_gate(
    citations: &[Citation],
    budget: &crate::model_catalog::ModelContextBudget,
) -> serde_json::Value {
    serde_json::json!({
        "status": if citations.is_empty() { "insufficient_evidence" } else { "accepted" },
        "citation_count": citations.len(),
        "contextBudget": budget,
        "runtime": "m1-deterministic",
    })
}

fn build_retrieval_trace(
    run_id: &str,
    intent: &str,
    tree_hits: &[StructureTreeHit],
    candidates: &[EvidenceCandidate],
    tool_calls: &[RetrievalTraceToolCall],
    finalize_gate: &serde_json::Value,
) -> RetrievalTrace {
    RetrievalTrace {
        run_id: run_id.to_string(),
        intent: intent.to_string(),
        tree_nodes: tree_hits
            .iter()
            .map(|hit| RetrievalTraceTreeNode {
                id: hit.id.clone(),
                title: hit.title.clone(),
                page: hit.page,
                block_index: hit.block_index,
                score: hit.score,
            })
            .collect(),
        candidates: trace_candidates_from_candidates(candidates),
        tool_calls: tool_calls.to_vec(),
        finalize_gate: finalize_gate.clone(),
    }
}

fn retrieval_run_id(conn: &Connection) -> Result<String, String> {
    conn.query_row("SELECT lower(hex(randomblob(8)))", [], |row| {
        row.get::<_, String>(0)
    })
    .map(|value| format!("retrieval-{value}"))
    .map_err(|err| format!("Failed to create retrieval run id: {err}"))
}

fn is_recoverable_fts_error(err: &str) -> bool {
    let err = err.to_lowercase();
    err.contains("fts") || err.contains("match") || err.contains("syntax")
}

fn query_terms(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .map(|term| term.trim().to_lowercase())
        .filter(|term| term.len() > 1)
        .collect()
}

/// True for CJK ideographs (Han) — used to derive single-character keyword terms
/// for chat-history recall, since whitespace/word tokenization doesn't split CJK.
fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VisualAnchorRequest {
    number: String,
    asset_type: Option<String>,
}

fn requested_visual_anchor(value: &str) -> Option<VisualAnchorRequest> {
    let normalized = value.to_lowercase();
    for (marker, asset_type) in [
        ("figure", "figure"),
        ("fig.", "figure"),
        ("fig", "figure"),
        ("chart", "chart"),
        ("image", "image"),
        ("diagram", "figure"),
        ("图表", "chart"),
        ("图像", "image"),
        ("图片", "image"),
        ("图", "figure"),
    ] {
        for (index, _) in normalized.match_indices(marker) {
            let rest = &normalized[index + marker.len()..];
            if let Some(number) = leading_reference_number(rest) {
                return Some(VisualAnchorRequest {
                    number,
                    asset_type: Some(asset_type.to_string()),
                });
            }
        }
    }
    None
}

fn table_caption_number(caption: &str) -> Option<String> {
    requested_table_number(caption)
}

fn visual_caption_number(caption: &str) -> Option<String> {
    requested_visual_anchor(caption).map(|anchor| anchor.number)
}

fn visual_anchor_asset_type_matches(requested: &str, actual: &str) -> bool {
    requested == actual || (requested == "figure" && matches!(actual, "figure" | "chart" | "image"))
}

fn escape_fts_query(query: &str) -> String {
    // CJK needs per-character tokens to be findable at all; English is passed
    // through with the same quote-and-OR shape as before. See search_text.
    crate::search_text::match_query(query)
}

fn normalize_for_dedupe(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut result = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        result.push_str("...");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_ACCUMULATED_CITATIONS: usize = 24;

    #[test]
    fn classify_heading_rejects_numeric_and_chart_noise() {
        assert!(classify_heading("1").is_none());
        assert!(classify_heading("1 1").is_none());
        assert!(classify_heading("1 kept").is_none());
        assert!(classify_heading("60 Agent Rounds").is_none());
        assert_eq!(
            classify_heading("3 Approach"),
            Some((1, "3 Approach".to_string()))
        );
        assert_eq!(
            classify_heading("3.1 Adaptive Pruning"),
            Some((2, "3.1 Adaptive Pruning".to_string()))
        );
    }

    #[test]
    fn detect_headings_builds_nested_sections_and_skips_page_noise() {
        let blocks = vec![
            structure_block(3, 1, "APPROACH", "heading"),
            structure_block(3, 2, "Opening approach paragraph.", "body"),
            structure_block(3, 3, "Approach", "heading"),
            structure_block(3, 4, "Framework overview text.", "body"),
            structure_block(4, 1, "APPROACH", "heading"),
            structure_block(4, 2, "Pruning Pipeline", "body"),
            structure_block(4, 3, "Figure 3 Overview.", "caption"),
            structure_block(4, 4, "Goal Hint Generation", "body"),
            structure_block(4, 5, "Goal hint paragraph.", "body"),
            structure_block(4, 6, "Lightweight Neural Skimmer", "body"),
            structure_block(5, 1, "APPROACH", "heading"),
            structure_block(5, 2, "Training Objective.", "body"),
            structure_block(6, 1, "RESULTS", "heading"),
            structure_block(6, 2, "Experiments", "heading"),
            structure_block(7, 1, "RESULTS", "heading"),
            structure_block(7, 2, "Table 1", "caption"),
            structure_block(7, 3, "Method", "heading"),
            structure_block(
                7,
                4,
                "Comparison with Alternative Context Management Strategies.",
                "body",
            ),
        ];

        let headings = detect_headings(&blocks);
        let titles = headings
            .iter()
            .map(|heading| (heading.title.as_str(), heading.level, heading.kind))
            .collect::<Vec<_>>();

        assert!(titles.contains(&("APPROACH", 1, "heading")));
        assert!(titles.contains(&("Approach", 2, "heading")));
        assert!(titles.contains(&("Goal Hint Generation", 2, "inferred_subheading")));
        assert!(titles.contains(&("Lightweight Neural Skimmer", 2, "inferred_subheading")));
        assert!(titles.contains(&("Training Objective", 2, "inferred_subheading")));
        assert!(titles.contains(&("RESULTS", 1, "heading")));
        assert!(titles.contains(&("Experiments", 2, "heading")));
        assert!(titles.contains(&(
            "Comparison with Alternative Context Management Strategies",
            2,
            "inferred_subheading"
        )));
        assert_eq!(
            headings
                .iter()
                .filter(|heading| heading.title == "APPROACH")
                .count(),
            1
        );
        assert!(!headings.iter().any(|heading| heading.title == "Method"));
        assert!(!headings
            .iter()
            .any(|heading| heading.title == "Pruning Pipeline"));
    }

    #[test]
    fn outline_ranges_use_pdf_outline_boundaries() {
        let blocks = vec![
            structure_block(1, 1, "Title", "title"),
            structure_block(2, 1, "Introduction", "heading"),
            structure_block(2, 2, "Intro body", "body"),
            structure_block(3, 1, "Approach", "heading"),
            structure_block(3, 2, "Goal Hint Generation", "body"),
            structure_block(4, 1, "Lightweight Neural Skimmer", "body"),
            structure_block(6, 1, "Experiments", "heading"),
            structure_block(6, 2, "Benchmarks", "body"),
        ];
        let outlines = vec![
            outline_seed("Introduction", 1, 2, 0),
            outline_seed("Approach", 1, 3, 1),
            outline_seed("Goal Hint Generation", 2, 3, 2),
            outline_seed("Lightweight Neural Skimmer", 2, 4, 3),
            outline_seed("Experiments", 1, 6, 4),
        ];

        let ranges = outline_ranges(&outlines, 6, &blocks);

        assert_eq!(ranges.len(), 5);
        assert_eq!(ranges[1].title, "Approach");
        assert_eq!(ranges[1].page_no, 3);
        assert_eq!(ranges[1].page_end, 5);
        assert_eq!(ranges[2].block_start_index, 2);
        assert_eq!(ranges[3].title, "Lightweight Neural Skimmer");
        assert_eq!(ranges[3].page_end, 5);
        assert_eq!(ranges[4].title, "Experiments");
    }

    #[test]
    fn resolve_section_end_closes_before_next_page_first_block() {
        let heading = HeadingSeed {
            title: "Abstract".to_string(),
            level: 1,
            page_no: 1,
            block_index: 4,
            bbox_list: serde_json::json!([]),
            kind: "abstract",
        };
        let next_heading = HeadingSeed {
            title: "1 Introduction".to_string(),
            level: 1,
            page_no: 2,
            block_index: 1,
            bbox_list: serde_json::json!([]),
            kind: "heading",
        };
        let last_block_by_page = HashMap::from([(1, 9), (2, 6)]);

        assert_eq!(
            resolve_section_end(&heading, Some(&next_heading), 2, &last_block_by_page),
            (1, 9)
        );
    }

    #[test]
    fn header_mode_compacts_only_header_roles() {
        let rows = vec![
            page_candidate("b1", "Paper Title", "title"),
            page_candidate("b2", "Ada Lovelace, Alan Turing", "authors"),
            page_candidate("b3", "Example University", "affiliations"),
            page_candidate(
                "b4",
                "Abstract This should not be in header mode.",
                "abstract",
            ),
        ];

        let compacted = compact_header_blocks("doc", 1, rows);

        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].section_title.as_deref(), Some("Page 1 header"));
        assert!(compacted[0].quote.contains("Ada Lovelace"));
        assert!(compacted[0].quote.contains("Example University"));
        assert!(!compacted[0].quote.contains("Abstract"));
    }

    #[test]
    fn dispatcher_executes_open_pages_header_mode() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE document_blocks (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                block_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                block_role TEXT NOT NULL
            );",
        )
        .expect("schema");
        for (index, text, role) in [
            (1, "Paper Title", "title"),
            (2, "Ada Lovelace, Alan Turing", "authors"),
            (3, "Example University", "affiliations"),
            (4, "Abstract This should be excluded.", "abstract"),
        ] {
            conn.execute(
                "INSERT INTO document_blocks
                    (id, document_id, page_no, block_index, text, bbox_json, block_role)
                 VALUES (?1, 'doc', 1, ?2, ?3, '[[0,0,10,10]]', ?4)",
                params![format!("b{index}"), index, text, role],
            )
            .expect("insert block");
        }

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "open_pages",
            &serde_json::json!({ "page": 1, "mode": "header" }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "open_pages");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.tool_call.result_count, 1);
        assert_eq!(output.citations.len(), 1);
        assert!(output.citations[0].quote.contains("Ada Lovelace"));
        assert!(!output.citations[0].quote.contains("Abstract"));
    }

    #[test]
    fn dispatcher_open_section_uses_query_when_node_ids_missing() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE structure_tree_nodes (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                level INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                block_start_index INTEGER NOT NULL,
                block_end_index INTEGER NOT NULL,
                order_index INTEGER NOT NULL
            );
            CREATE TABLE document_blocks (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                block_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                block_role TEXT NOT NULL
            );",
        )
        .expect("schema");
        conn.execute(
            "INSERT INTO structure_tree_nodes
                (id, document_id, title, keywords_json, level, page_start, page_end,
                 block_start_index, block_end_index, order_index)
             VALUES
                ('node-method', 'doc', '3 Method', '[\"method\",\"approach\"]', 1, 2, 2, 1, 2, 1)",
            [],
        )
        .expect("insert tree node");
        conn.execute(
            "INSERT INTO document_blocks
                (id, document_id, page_no, block_index, text, bbox_json, block_role)
             VALUES
                ('b1', 'doc', 2, 1, 'The method uses an adaptive pruning framework.', '[[0,0,10,10]]', 'body')",
            [],
        )
        .expect("insert block");

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "open_section",
            &serde_json::json!({ "query": "method approach", "perSectionLimit": 10 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "open_section");
        assert_eq!(output.tool_call.result_count, 1);
        assert_eq!(output.tree_nodes[0].id, "node-method");
        assert_eq!(output.tree_nodes[0].page, 2);
        assert_eq!(output.tree_nodes[0].block_index, 1);
        assert!(output.citations[0]
            .quote
            .contains("adaptive pruning framework"));
    }

    #[test]
    fn chinese_method_query_expands_to_tree_keywords() {
        let query = expand_query_for_retrieval(
            "这篇文章的方法具体是怎么设计的？请结合方法章节、算法流程和实验结果说明。",
        );

        assert!(query.contains("method approach methodology algorithm framework"));
        assert!(query.contains("experiments evaluation results benchmark"));
    }

    #[test]
    fn chinese_overview_query_expands_to_paper_structure_keywords() {
        let query = expand_query_for_retrieval("这篇文章讲了什么？");

        assert!(query.contains("abstract introduction contribution"));
        assert!(query.contains("method approach"));
        assert!(query.contains("experiments evaluation results conclusion"));
    }

    #[test]
    fn chinese_method_question_uses_expanded_query_to_open_sections() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE structure_tree_nodes (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                level INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                block_start_index INTEGER NOT NULL,
                block_end_index INTEGER NOT NULL,
                order_index INTEGER NOT NULL
            );
            CREATE TABLE document_blocks (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                block_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                block_role TEXT NOT NULL
            );
            CREATE TABLE document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                block_ids_json TEXT NOT NULL,
                text TEXT NOT NULL,
                bbox_refs_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE document_chunks_fts
                USING fts5(chunk_id UNINDEXED, document_id UNINDEXED, text);
            CREATE TABLE retrieval_evidence_runs (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                question TEXT NOT NULL,
                intent TEXT NOT NULL,
                tree_node_ids_json TEXT NOT NULL DEFAULT '[]',
                fts_candidate_ids_json TEXT NOT NULL DEFAULT '[]',
                wiki_candidate_ids_json TEXT NOT NULL DEFAULT '[]',
                vector_candidate_ids_json TEXT NOT NULL DEFAULT '[]',
                selected_candidate_ids_json TEXT NOT NULL DEFAULT '[]',
                finalize_gate_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL
            );",
        )
        .expect("schema");
        conn.execute(
            "INSERT INTO structure_tree_nodes
                (id, document_id, title, keywords_json, level, page_start, page_end,
                 block_start_index, block_end_index, order_index)
             VALUES
                ('node-approach', 'doc', 'Approach', '[\"approach\",\"method\",\"algorithm\"]', 1, 3, 3, 1, 1, 1),
                ('node-exp', 'doc', 'Experiments', '[\"experiments\",\"evaluation\",\"results\"]', 1, 6, 6, 1, 1, 2)",
            [],
        )
        .expect("insert tree nodes");
        conn.execute(
            "INSERT INTO document_blocks
                (id, document_id, page_no, block_index, text, bbox_json, block_role)
             VALUES
                ('b-method', 'doc', 3, 1, 'The method uses a task-aware adaptive pruning pipeline.', '[[0,0,10,10]]', 'body'),
                ('b-exp', 'doc', 6, 1, 'Experiments report benchmark results for token reduction and task success.', '[[0,0,10,10]]', 'body')",
            [],
        )
        .expect("insert blocks");

        let run = build_retrieval_run(
            &conn,
            RetrievalRequest {
                document_id: "doc",
                question:
                    "这篇文章的方法具体是怎么设计的？请结合方法章节、算法流程和实验结果说明。",
                retrieval_query: None,
                selected_text: None,
                selected_block_id: None,
                selected_bbox_list: None,
                client_context: None,
                page: None,
                page_mode: None,
                page_source: None,
                context_budget: crate::model_catalog::ModelContextBudget::default(),
                force_document_start: false,
            },
        )
        .expect("retrieval run");

        let titles = run
            .trace
            .tree_nodes
            .iter()
            .map(|node| node.title.as_str())
            .collect::<Vec<_>>();
        assert!(titles.contains(&"Approach"));
        assert!(titles.contains(&"Experiments"));
        assert!(run
            .citations
            .iter()
            .any(|citation| citation.quote.contains("adaptive pruning pipeline")));
        assert!(run
            .citations
            .iter()
            .any(|citation| citation.quote.contains("benchmark results")));
    }

    #[test]
    fn open_section_prefers_query_relevant_later_blocks() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE structure_tree_nodes (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                level INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                block_start_index INTEGER NOT NULL,
                block_end_index INTEGER NOT NULL,
                order_index INTEGER NOT NULL
            );
            CREATE TABLE document_blocks (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                block_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                block_role TEXT NOT NULL
            );",
        )
        .expect("schema");
        conn.execute(
            "INSERT INTO structure_tree_nodes
                (id, document_id, title, keywords_json, level, page_start, page_end,
                 block_start_index, block_end_index, order_index)
             VALUES
                ('node-approach', 'doc', 'Approach', '[\"approach\",\"method\",\"algorithm\"]', 1, 3, 3, 1, 20, 1)",
            [],
        )
        .expect("insert tree node");
        for index in 1..=12 {
            let text = if index == 12 {
                "Algorithm 1 describes the pruning pipeline and line-level scoring workflow."
            } else {
                "Background context before the detailed method."
            };
            conn.execute(
                "INSERT INTO document_blocks
                    (id, document_id, page_no, block_index, text, bbox_json, block_role)
                 VALUES (?1, 'doc', 3, ?2, ?3, '[[0,0,10,10]]', 'body')",
                params![format!("b{index}"), index, text],
            )
            .expect("insert block");
        }

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "open_section",
            &serde_json::json!({
                "treeNodeIds": ["node-approach"],
                "query": "algorithm workflow",
                "perSectionLimit": 3
            }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "open_section");
        assert!(output
            .citations
            .iter()
            .any(|citation| citation.quote.contains("Algorithm 1")));
    }

    #[test]
    fn read_tree_node_lines_prefers_query_relevant_later_lines() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE structure_tree_nodes (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                level INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                block_start_index INTEGER NOT NULL,
                block_end_index INTEGER NOT NULL,
                order_index INTEGER NOT NULL
            );
            CREATE TABLE document_lines (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                line_no INTEGER NOT NULL,
                block_id TEXT NOT NULL DEFAULT '',
                block_index INTEGER NOT NULL DEFAULT 0,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL
            );",
        )
        .expect("schema");
        conn.execute(
            "INSERT INTO structure_tree_nodes
                (id, document_id, title, keywords_json, level, page_start, page_end,
                 block_start_index, block_end_index, order_index)
             VALUES
                ('node-approach', 'doc', 'Approach', '[\"approach\",\"method\",\"algorithm\"]', 1, 3, 4, 1, 20, 1)",
            [],
        )
        .expect("insert tree node");
        for index in 1..=18 {
            let text = if index == 16 {
                "Algorithm 1 describes the pruning pipeline and line-level scoring workflow."
            } else {
                "Background context before the detailed method."
            };
            conn.execute(
                "INSERT INTO document_lines
                    (id, document_id, page_no, line_no, block_id, block_index, text, bbox_json)
                 VALUES (?1, 'doc', 3, ?2, 'b1', 1, ?3, '[[0,0,10,10]]')",
                params![format!("l{index}"), index, text],
            )
            .expect("insert line");
        }
        conn.execute(
            "INSERT INTO document_lines
                (id, document_id, page_no, line_no, block_id, block_index, text, bbox_json)
             VALUES
                ('outside', 'doc', 4, 19, 'b25', 25,
                 'Algorithm details from a same-page block outside this tree node.',
                 '[[0,0,10,10]]')",
            [],
        )
        .expect("insert outside line");
        conn.execute(
            "INSERT INTO document_lines
                (id, document_id, page_no, line_no, block_id, block_index, text, bbox_json)
             VALUES
                ('unassigned', 'doc', 4, 20, '', 0,
                 'Algorithm details from an unassigned same-page line outside this tree node.',
                 '[[0,0,10,10]]')",
            [],
        )
        .expect("insert unassigned outside line");

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "read_tree_node_lines",
            &serde_json::json!({
                "treeNodeIds": ["node-approach"],
                "query": "algorithm workflow",
                "lineLimit": 3
            }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "read_tree_node_lines");
        assert_eq!(output.tool_call.status, "ok");
        assert!(output
            .citations
            .iter()
            .any(|citation| citation.quote.contains("Algorithm 1")));
        assert!(output
            .citations
            .iter()
            .all(|citation| !citation.quote.contains("outside this tree node")));
        assert!(output
            .citations
            .iter()
            .all(|citation| !citation.quote.contains("unassigned same-page line")));
        assert!(output
            .citations
            .iter()
            .all(|citation| citation.source == "read_tree_node_lines"));
    }

    #[test]
    fn inspect_tables_anchors_explicit_table_number() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_numbered_table_fact(
            &conn,
            "table-1",
            7,
            "Table 1",
            "+ SWE-Pruner",
            "Success (%)",
            "72.0",
        );
        insert_numbered_table_fact(
            &conn,
            "table-3",
            8,
            "Table 3",
            "SWE-Pruner",
            "Success (%)",
            "64.0",
        );
        insert_numbered_table_fact(
            &conn,
            "table-4",
            8,
            "Table 4",
            "SWE-Pruner",
            "Success (%)",
            "58.63",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "inspect_tables",
            &serde_json::json!({ "query": "Table 3 SWE-Pruner", "limit": 8 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "inspect_tables");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].block_id, "table-3");
        assert_eq!(
            output.citations[0].section_title.as_deref(),
            Some("Table 3")
        );
    }

    #[test]
    fn open_table_uses_explicit_table_anchor_before_broad_matches() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_numbered_table_fact(
            &conn,
            "table-1",
            7,
            "Table 1",
            "+ SWE-Pruner",
            "Success (%)",
            "72.0",
        );
        insert_numbered_table_fact(
            &conn,
            "table-3",
            8,
            "Table 3",
            "SWE-Pruner",
            "Rounds",
            "41.1",
        );
        insert_numbered_table_fact(
            &conn,
            "table-3",
            8,
            "Table 3",
            "SWE-Pruner",
            "Success (%)",
            "64.0",
        );
        insert_numbered_table_fact(
            &conn,
            "table-3",
            8,
            "Table 3",
            "SWE-Pruner",
            "Tokens (M)",
            "0.670",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "open_table",
            &serde_json::json!({ "query": "Table 3 SWE-Pruner", "limit": 8 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "open_table");
        assert_eq!(output.tool_call.status, "ok");
        assert!(output.citations.iter().any(|citation| citation
            .quote
            .contains("Table 3 | SWE-Pruner | Success (%) = 64.0")));
        assert!(output
            .citations
            .iter()
            .all(|citation| citation.section_title.as_deref() == Some("Table 3")));
        assert!(output
            .citations
            .iter()
            .all(|citation| !citation.quote.contains("Table 1")));
    }

    #[test]
    fn resolve_table_anchor_returns_table_id_for_explicit_reference() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_numbered_table_fact(
            &conn,
            "table-3",
            8,
            "Table 3",
            "SWE-Pruner",
            "Tokens (M)",
            "0.670",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "resolve_table_anchor",
            &serde_json::json!({ "query": "表 3 里的 SWE-Pruner" }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "resolve_table_anchor");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].source, "table_anchor");
        assert!(output.citations[0].quote.contains("tableId=table-3"));
    }

    #[test]
    fn resolve_table_anchor_accepts_numeric_table_number_arg() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_numbered_table_fact(
            &conn,
            "table-1",
            7,
            "Table 1",
            "+ SWE-Pruner",
            "Success (%)",
            "72.0",
        );
        insert_numbered_table_fact(
            &conn,
            "table-3",
            8,
            "Table 3",
            "SWE-Pruner",
            "Success (%)",
            "64.0",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "resolve_table_anchor",
            &serde_json::json!({ "query": "SWE-Pruner", "tableNumber": 3 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "resolve_table_anchor");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].block_id, "table-3");
        assert!(output.citations[0].quote.contains("tableId=table-3"));

        let cjk_output = execute_rag_tool_call(
            &conn,
            "doc",
            "resolve_table_anchor",
            &serde_json::json!({ "query": "表三里面的 SWE-Pruner 结果" }),
            "fallback query",
        );

        assert_eq!(cjk_output.tool_call.status, "ok");
        assert_eq!(cjk_output.citations.len(), 1);
        assert_eq!(cjk_output.citations[0].block_id, "table-3");
    }

    #[test]
    fn resolve_visual_anchor_returns_asset_id_for_explicit_reference() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_visual_asset_fixture(
            &conn,
            "fig-2",
            3,
            "figure",
            "Figure 2 Sonnet 4.5 mechanisms.",
            "Token cost distribution over different tool calls.",
        );
        insert_visual_asset_fixture(
            &conn,
            "fig-3",
            4,
            "figure",
            "Figure 3",
            "Interaction workflow and pruning pipeline.",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "resolve_visual_anchor",
            &serde_json::json!({ "query": "图3 说明了什么？" }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "resolve_visual_anchor");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].source, "visual_anchor");
        assert_eq!(output.citations[0].block_id, "fig-3");
        assert!(output.citations[0].quote.contains("assetId=fig-3"));
    }

    #[test]
    fn resolve_visual_anchor_accepts_numeric_visual_number_arg() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_visual_asset_fixture(&conn, "fig-2", 3, "figure", "Figure 2", "Other mechanism.");
        insert_visual_asset_fixture(&conn, "fig-3", 4, "figure", "Figure 3", "Pruning pipeline.");

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "resolve_visual_anchor",
            &serde_json::json!({ "query": "pruning pipeline", "visualNumber": 3 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "resolve_visual_anchor");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].block_id, "fig-3");

        let cjk_output = execute_rag_tool_call(
            &conn,
            "doc",
            "resolve_visual_anchor",
            &serde_json::json!({ "query": "图三说明了什么" }),
            "fallback query",
        );

        assert_eq!(cjk_output.tool_call.status, "ok");
        assert_eq!(cjk_output.citations.len(), 1);
        assert_eq!(cjk_output.citations[0].block_id, "fig-3");
    }

    #[test]
    fn inspect_objects_lists_visual_assets_for_page() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_visual_asset_fixture(
            &conn,
            "fig-3",
            4,
            "figure",
            "Figure 3",
            "Interaction workflow and pruning pipeline.",
        );
        insert_visual_asset_fixture(
            &conn,
            "fig-4",
            5,
            "figure",
            "Figure 4",
            "Latency comparison.",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "inspect_objects",
            &serde_json::json!({ "page": 4, "limit": 8 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "inspect_objects");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].block_id, "fig-3");
        assert!(output.citations[0].quote.contains("assetId=fig-3"));
    }

    #[test]
    fn inspect_objects_keeps_page_scope_when_query_terms_do_not_match() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_visual_asset_fixture(
            &conn,
            "fig-3",
            4,
            "figure",
            "Figure 3",
            "Interaction workflow and pruning pipeline.",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "inspect_objects",
            &serde_json::json!({ "page": 4, "query": "这个图说明了什么", "limit": 8 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "inspect_objects");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].block_id, "fig-3");
    }

    #[test]
    fn inspect_objects_uses_tree_node_page_range() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        conn.execute_batch(
            "CREATE TABLE structure_tree_nodes (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                level INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                block_start_index INTEGER NOT NULL,
                block_end_index INTEGER NOT NULL,
                order_index INTEGER NOT NULL
            );",
        )
        .expect("tree schema");
        conn.execute(
            "INSERT INTO structure_tree_nodes
                (id, document_id, title, keywords_json, level, page_start, page_end,
                 block_start_index, block_end_index, order_index)
             VALUES
                ('node-results', 'doc', 'Results', '[\"results\"]', 1, 7, 8, 1, 20, 1)",
            [],
        )
        .expect("insert tree node");
        insert_visual_asset_fixture(
            &conn,
            "table-3-asset",
            8,
            "table",
            "Table 3",
            "SWE-Pruner results.",
        );
        insert_visual_asset_fixture(
            &conn,
            "fig-5",
            14,
            "figure",
            "Figure 5",
            "Architecture details.",
        );

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "inspect_objects",
            &serde_json::json!({ "treeNodeIds": ["node-results"], "limit": 8 }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "inspect_objects");
        assert_eq!(output.tool_call.status, "ok");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].block_id, "table-3-asset");
        assert_eq!(output.tree_nodes[0].id, "node-results");
        assert_eq!(output.tree_nodes[0].page, 7);
    }

    #[test]
    fn tree_node_ids_arg_preserves_model_order_while_deduping() {
        let ids = tree_node_ids_arg(&serde_json::json!({
            "treeNodeIds": ["node-2", "node-10", "node-2"],
            "nodeId": "node-1"
        }));

        assert_eq!(ids, vec!["node-2", "node-10", "node-1"]);
    }

    #[test]
    fn numeric_args_ignore_values_outside_u32_range() {
        let args = serde_json::json!({
            "page": u64::MAX,
            "limit": u64::MAX
        });

        assert_eq!(optional_u32_arg(&args, "page"), None);
        assert_eq!(u32_arg(&args, "limit", 8, 1, 20), 8);
    }

    #[test]
    fn list_trending_papers_reads_cache_and_filters() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE trending_cache (period TEXT PRIMARY KEY, payload_json TEXT NOT NULL, fetched_at INTEGER NOT NULL);",
        )
        .unwrap();
        let payload = serde_json::json!([
            { "arxivId": "2606.1", "title": "Scaling LLM training", "summary": "About large-model training.", "upvotes": 90 },
            { "arxivId": "2606.2", "title": "A vision paper", "summary": "About image segmentation.", "upvotes": 10 }
        ])
        .to_string();
        conn.execute(
            "INSERT INTO trending_cache (period, payload_json, fetched_at) VALUES ('daily', ?1, 0)",
            rusqlite::params![payload],
        )
        .unwrap();

        // No query → both papers.
        let all = execute_rag_tool_call(
            &conn,
            "doc",
            "list_trending_papers",
            &serde_json::json!({}),
            "",
        );
        assert_eq!(all.tool_call.tool, "list_trending_papers");
        assert_eq!(all.citations.len(), 2);

        // Keyword filter on title/abstract → only the training paper.
        let filtered = execute_rag_tool_call(
            &conn,
            "doc",
            "list_trending_papers",
            &serde_json::json!({ "query": "training" }),
            "",
        );
        assert_eq!(filtered.citations.len(), 1);
        assert!(filtered.citations[0].quote.contains("Scaling LLM training"));

        // An un-cached period yields nothing (rather than erroring).
        let empty = execute_rag_tool_call(
            &conn,
            "doc",
            "list_trending_papers",
            &serde_json::json!({ "period": "weekly" }),
            "",
        );
        assert_eq!(empty.citations.len(), 0);
    }

    #[test]
    fn search_library_knowledge_finds_docs_by_concept() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT NOT NULL);
             CREATE TABLE document_artifacts (document_id TEXT, kind TEXT, name TEXT, normalized TEXT);
             INSERT INTO documents VALUES ('d1','Training methods'), ('d2','Vision models');
             INSERT INTO document_artifacts VALUES
               ('d1','concept','model training','model training'),
               ('d1','entity','GPT','gpt'),
               ('d2','concept','image segmentation','image segmentation');",
        )
        .unwrap();

        let hit = execute_rag_tool_call(
            &conn,
            "d2",
            "search_library_knowledge",
            &serde_json::json!({ "query": "training" }),
            "",
        );
        assert_eq!(hit.tool_call.tool, "search_library_knowledge");
        // Whole-library search: finds d1 even though the focus doc is d2.
        assert_eq!(hit.citations.len(), 1);
        assert_eq!(hit.citations[0].document_id, "d1");
        assert!(hit.citations[0].quote.contains("model training"));
    }

    #[test]
    fn build_retrieval_run_focus_optional_returns_empty_run() {
        // KB pivot (P0-d): no focus document → an empty, valid run (the agent then
        // uses library-wide tools). The empty-doc path only hits retrieval_run_id's
        // table-less query, so a schema-less in-memory conn suffices.
        let conn = Connection::open_in_memory().expect("in-memory db");
        let request = RetrievalRequest {
            document_id: "",
            question: "what do I know about RAG citation granularity?",
            retrieval_query: None,
            selected_text: None,
            selected_block_id: None,
            selected_bbox_list: None,
            client_context: None,
            page: None,
            page_mode: None,
            page_source: None,
            context_budget: crate::model_catalog::ModelContextBudget::default(),
            force_document_start: false,
        };
        let run = build_retrieval_run(&conn, request).expect("focus-optional run");
        assert!(run.citations.is_empty());
        assert!(run.prompt_context.is_empty());
        assert!(!run.id.is_empty());
    }

    #[test]
    fn list_sources_enumerates_the_library_and_filters() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT NOT NULL);
             CREATE TABLE documents (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, content_type TEXT NOT NULL,
                index_status TEXT NOT NULL, page_count INTEGER NOT NULL DEFAULT 0,
                collection_id TEXT, last_opened_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0
             );
             INSERT INTO collections (id, name) VALUES ('c1', 'Research');
             INSERT INTO documents
               (id, title, content_type, index_status, page_count, collection_id, last_opened_at)
             VALUES
               ('pdf-1', 'Attention Paper', 'pdf', 'indexed', 12, 'c1', 300),
               ('xls-1', 'Q3 Budget', 'xlsx', 'indexed', 1, NULL, 200),
               ('note-1', 'Budget notes', 'note', 'indexing', 1, NULL, 100);",
        )
        .expect("schema");
        let registry = RagToolRegistry::new(&conn, "pdf-1", 4_000);

        let all = execute_list_sources_tool(&registry, &serde_json::json!({})).expect("list");
        assert_eq!(all.citations.len(), 3);
        // Most recently opened first, so the likely subject comes first.
        assert!(all.citations[0].quote.starts_with("Attention Paper"));
        // The id is in the text: it is what every per-source tool needs next.
        assert!(all.citations[0].quote.contains("documentId: pdf-1"));
        assert!(all.citations[0].quote.contains("collection: Research"));
        // A source that is not ready says so, rather than reading as thin content.
        assert!(
            all.citations[2].quote.contains("index: indexing"),
            "{}",
            all.citations[2].quote
        );

        // Title filter is case-insensitive and matches mid-string.
        let budget =
            execute_list_sources_tool(&registry, &serde_json::json!({ "query": "budget" }))
                .expect("filtered");
        assert_eq!(budget.citations.len(), 2);

        let sheets =
            execute_list_sources_tool(&registry, &serde_json::json!({ "contentType": "xlsx" }))
                .expect("by kind");
        assert_eq!(sheets.citations.len(), 1);
        assert_eq!(sheets.citations[0].document_id, "xls-1");

        // Both filters together.
        let both = execute_list_sources_tool(
            &registry,
            &serde_json::json!({ "query": "budget", "contentType": "note" }),
        )
        .expect("both");
        assert_eq!(both.citations.len(), 1);
        assert_eq!(both.citations[0].document_id, "note-1");
    }

    #[test]
    fn read_note_source_returns_body_and_rejects_file_backed_documents() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                body_md TEXT
            );
             INSERT INTO documents (id, title, content_type, body_md)
               VALUES ('note-1', 'Draft', 'note', '# Draft\n\nSee [[Other]].');
             INSERT INTO documents (id, title, content_type, body_md)
               VALUES ('pdf-1', 'Paper', 'pdf', NULL);",
        )
        .expect("schema");

        // An authored source hands back its Markdown verbatim — including wikilink
        // syntax, which the model needs to see to preserve it.
        let registry = RagToolRegistry::new(&conn, "note-1", 400);
        let output = execute_read_note_source_tool(&registry).expect("note body");
        assert_eq!(output.citations.len(), 1);
        assert_eq!(output.citations[0].quote, "# Draft\n\nSee [[Other]].");
        assert_eq!(output.citations[0].document_id, "note-1");

        // A PDF has no body: this must be an error (so the model routes to the
        // retrieval tools) rather than an empty success it would retry.
        let registry = RagToolRegistry::new(&conn, "pdf-1", 400);
        let error = match execute_read_note_source_tool(&registry) {
            Ok(_) => panic!("a pdf has no body_md and must be rejected"),
            Err(err) => err,
        };
        assert!(error.contains("no editable Markdown body"), "{error}");

        let registry = RagToolRegistry::new(&conn, "missing", 400);
        assert!(execute_read_note_source_tool(&registry).is_err());
    }

    #[test]
    fn propose_note_edit_records_the_proposal_without_writing() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                body_md TEXT
            );
             INSERT INTO documents (id, title, content_type, body_md)
               VALUES ('note-1', 'Draft', 'note', 'old body');
             INSERT INTO documents (id, title, content_type, body_md)
               VALUES ('pdf-1', 'Paper', 'pdf', NULL);",
        )
        .expect("schema");

        let registry = RagToolRegistry::new(&conn, "note-1", 400);
        let args = serde_json::json!({ "content": "# New\n\nbody", "summary": "tightened" });
        let output = execute_propose_note_edit_tool(&registry, &args).expect("proposal");

        // The proposal rides on the tool call's args — that is what the UI reads.
        assert_eq!(output.tool_call.input["content"], "# New\n\nbody");
        // The citation is a short ack, NOT the text: echoing it would burn context.
        assert_eq!(output.citations[0].source, "note_edit_proposal");
        assert!(!output.citations[0].quote.contains("# New"));

        // Crucially: nothing was written. The editor stays the single writer.
        let body: String = conn
            .query_row(
                "SELECT body_md FROM documents WHERE id = 'note-1'",
                [],
                |r| r.get(0),
            )
            .expect("body");
        assert_eq!(body, "old body");

        // Neither edits nor content, and file-backed sources, are rejected.
        let empty = serde_json::json!({ "content": "   " });
        assert!(execute_propose_note_edit_tool(&registry, &empty).is_err());
        let pdf = RagToolRegistry::new(&conn, "pdf-1", 400);
        assert!(execute_propose_note_edit_tool(&pdf, &args).is_err());
    }

    #[test]
    fn propose_note_edit_resolves_precise_edits_against_the_note() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                body_md TEXT
            );
             INSERT INTO documents (id, title, content_type, body_md)
               VALUES ('note-1', 'Draft', 'note', '# Title\n\nHe said “hi” — ok.\n\ntail\n');",
        )
        .expect("schema");
        let registry = RagToolRegistry::new(&conn, "note-1", 400);

        // The model sends ASCII punctuation it can actually type.
        let args = serde_json::json!({
            "edits": [{ "oldText": "He said \"hi\" - ok.", "newText": "He said hello." }],
            "summary": "reworded",
        });
        let output = execute_propose_note_edit_tool(&registry, &args).expect("proposal");

        // oldText comes back resolved to the note's VERBATIM text (curly punctuation
        // intact), so the apply step needs only an exact match.
        assert_eq!(
            output.tool_call.input["edits"][0]["oldText"],
            "He said “hi” — ok."
        );
        // `content` is the resulting full text, for the preview.
        assert_eq!(
            output.tool_call.input["content"],
            "# Title\n\nHe said hello.\n\ntail\n"
        );
        // Still no write.
        let body: String = conn
            .query_row(
                "SELECT body_md FROM documents WHERE id = 'note-1'",
                [],
                |r| r.get(0),
            )
            .expect("body");
        assert!(body.contains("He said “hi” — ok."));

        // A bad edit fails the call and echoes the text, rather than half-applying.
        let bad = serde_json::json!({ "edits": [{ "oldText": "not present", "newText": "x" }] });
        let err = match execute_propose_note_edit_tool(&registry, &bad) {
            Ok(_) => panic!("missing oldText must fail"),
            Err(err) => err,
        };
        assert!(err.contains("not present"), "{err}");
    }

    #[test]
    fn rag_tool_registry_exposes_supported_tools() {
        let names = rag_tool_specs_for_capabilities(false, false)
            .into_iter()
            .map(|tool| tool.name)
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(
            names,
            std::collections::BTreeSet::from([
                "inspect_tree",
                "inspect_tables",
                "inspect_visuals",
                "inspect_objects",
                "open_table",
                "open_section",
                "open_pages",
                "open_visual",
                "read_tree_node_lines",
                "resolve_table_anchor",
                "resolve_visual_anchor",
                "search_chunks",
                "search_table_facts",
                "recall_chat_history",
                "query_knowledge_graph",
                "search_library_knowledge",
                "list_trending_papers",
                // Always offered: authored sources expose their Markdown body, and both
                // tools report a clear error for file-backed documents that have none.
                "read_note_source",
                "propose_note_edit",
                // Always offered: spreadsheets expose their grid; errors for non-xlsx.
                "read_sheet",
                // Always offered: the library's table of contents, and the only
                // way to obtain a documentId without guessing at a topic.
                "list_sources"
            ])
        );
        let vision_names = rag_tool_specs_for_capabilities(true, false)
            .into_iter()
            .map(|tool| tool.name)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(vision_names.contains("analyze_visual"));
        assert!(vision_names.contains("analyze_page"));
        let web_names = rag_tool_specs_for_capabilities(false, true)
            .into_iter()
            .map(|tool| tool.name)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(web_names.contains("web_search"));
        assert!(web_names.contains("web_fetch"));
        assert!(!web_names.contains("analyze_visual"));
        assert!(is_registered_rag_tool("open_section"));
        assert!(is_registered_rag_tool("read_tree_node_lines"));
        assert!(is_registered_rag_tool("search_table_facts"));
        assert!(!is_registered_rag_tool("analyze_visual"));
        assert!(!is_registered_rag_tool("analyze_page"));
        assert!(is_registered_rag_tool_for_capabilities(
            "analyze_visual",
            RagToolCapabilities {
                vision_enabled: true,
                ..RagToolCapabilities::default()
            }
        ));
        assert!(is_registered_rag_tool_for_capabilities(
            "analyze_page",
            RagToolCapabilities {
                vision_enabled: true,
                ..RagToolCapabilities::default()
            }
        ));
        assert!(!is_registered_rag_tool("unknown"));
    }

    #[test]
    fn rebuild_visual_evidence_generates_glm_style_table_facts() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_table7_fixture(&conn);

        let tx = conn.transaction().expect("transaction");
        rebuild_visual_evidence(&tx, "doc").expect("rebuild visual evidence");
        tx.commit().expect("commit");

        let fact = conn
            .query_row(
                "SELECT fact_text
                 FROM document_table_facts
                 WHERE document_id = 'doc'
                   AND row_label LIKE '%SWE-bench Verified%'
                   AND column_label = 'GLM-5'
                 LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("table fact");

        assert!(fact.contains("SWE-bench Verified"));
        assert!(fact.contains("GLM-5 = 77.8"));

        let terminal_fact = conn
            .query_row(
                "SELECT fact_text
                 FROM document_table_facts
                 WHERE document_id = 'doc'
                   AND row_label LIKE '%Terminal-Bench 2.0%'
                   AND column_label = 'GLM-5'
                 LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("terminal table fact");
        assert!(terminal_fact.contains("Terminal-Bench 2.0"));
        assert!(terminal_fact.contains("GLM-5 = 56.2"));

        let noisy_header_count = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM document_table_facts
                 WHERE document_id = 'doc'
                   AND (column_label LIKE '%highest score%'
                        OR column_label LIKE '%scores are recorded%'
                        OR column_label LIKE '%second highest%')",
                [],
                |row| row.get::<_, u32>(0),
            )
            .expect("noisy header count");
        assert_eq!(noisy_header_count, 0);

        let visual_bbox = conn
            .query_row(
                "SELECT bbox_json
                 FROM document_visual_assets
                 WHERE document_id = 'doc' AND asset_type = 'table'
                 LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("table visual asset");
        let bbox_json: serde_json::Value = serde_json::from_str(&visual_bbox).expect("bbox json");
        let rect = rect_from_bbox_list(&bbox_json).expect("table visual bbox");
        assert!(rect.y2 > 0.36);
    }

    #[test]
    fn rebuild_visual_evidence_handles_table_caption_below_rows() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        conn.execute(
            "INSERT INTO document_blocks
                (id, document_id, page_no, block_index, text, bbox_json, block_role)
             VALUES
                ('b-caption', 'doc', 20, 20,
                 'Table 7 Results on Long Code Completion and Long Code QA.',
                 '[[0.115,0.284,0.884,0.323]]', 'caption')",
            [],
        )
        .expect("caption block");
        for (line_no, text, bbox) in [
            (1, "Long Code Completion", "[[0.36,0.090,0.54,0.103]]"),
            (2, "Method", "[[0.16,0.107,0.22,0.120]]"),
            (3, "ES", "[[0.36,0.134,0.40,0.147]]"),
            (4, "EM", "[[0.42,0.134,0.45,0.147]]"),
            (5, "Full", "[[0.16,0.156,0.19,0.168]]"),
            (6, "65.03", "[[0.36,0.156,0.40,0.168]]"),
            (7, "40.5", "[[0.42,0.156,0.45,0.168]]"),
            (8, "SWE-Pruner", "[[0.16,0.254,0.26,0.265]]"),
            (9, "57.71", "[[0.36,0.254,0.40,0.265]]"),
            (10, "31.0", "[[0.42,0.254,0.45,0.265]]"),
            (21, "Method", "[[0.31,0.346,0.38,0.357]]"),
            (22, "AST Correctness (%)", "[[0.52,0.346,0.68,0.357]]"),
            (23, "Function RAG", "[[0.31,0.499,0.42,0.512]]"),
            (24, "92.3", "[[0.59,0.499,0.62,0.512]]"),
        ] {
            conn.execute(
                "INSERT INTO document_lines
                    (id, document_id, page_no, line_no, block_id, block_index, text, bbox_json)
                 VALUES (?1, 'doc', 20, ?2, '', 0, ?3, ?4)",
                params![format!("caption-below-l{line_no}"), line_no, text, bbox],
            )
            .expect("line insert");
        }

        let tx = conn.transaction().expect("transaction");
        rebuild_visual_evidence(&tx, "doc").expect("rebuild visual evidence");
        tx.commit().expect("commit");

        let (fact_text, bbox_json): (String, String) = conn
            .query_row(
                "SELECT fact_text, bbox_json
                 FROM document_table_facts
                 WHERE document_id = 'doc'
                   AND row_label = 'SWE-Pruner'
                   AND value_text = '57.71'
                 LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("caption-below table fact");
        assert!(fact_text.contains("Table 7"));
        assert!(fact_text.contains("SWE-Pruner"));
        let bbox_json: serde_json::Value = serde_json::from_str(&bbox_json).expect("bbox json");
        let rect = rect_from_bbox_list(&bbox_json).expect("fact bbox");
        assert!(rect.y1 < 0.28);
    }

    #[test]
    fn rebuild_visual_evidence_keeps_neighbor_tables_separate() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        for (id, page_no, block_index, text, bbox) in [
            (
                "table3-caption",
                8,
                33,
                "Table 3 Comparison of context compression strategies on SWE-Bench.",
                "[[0.1152,0.2201,0.8842,0.2453]]",
            ),
            (
                "table4-caption",
                8,
                134,
                "Table 4 Main results on Long Code Completion and Long Code QA tasks.",
                "[[0.1152,0.4589,0.8842,0.4979]]",
            ),
        ] {
            conn.execute(
                "INSERT INTO document_blocks
                    (id, document_id, page_no, block_index, text, bbox_json, block_role)
                 VALUES (?1, 'doc', ?2, ?3, ?4, ?5, 'caption')",
                params![id, page_no, block_index, text, bbox],
            )
            .expect("caption block");
        }
        for (line_no, text, bbox) in [
            (4, "Method", "[[0.2989,0.0932,0.3584,0.1046]]"),
            (5, "Rounds", "[[0.4461,0.0932,0.5035,0.1046]]"),
            (6, "Success (%)", "[[0.5133,0.0932,0.6033,0.1046]]"),
            (7, "Tokens (M)", "[[0.613,0.0932,0.7011,0.1046]]"),
            (8, "Mini SWE Agent", "[[0.2989,0.1134,0.4221,0.126]]"),
            (9, "52.3", "[[0.4604,0.1134,0.4893,0.126]]"),
            (10, "62.0", "[[0.5438,0.1134,0.5727,0.126]]"),
            (11, "0.972", "[[0.6385,0.1134,0.6756,0.126]]"),
            (28, "+", "[[0.2989,0.1889,0.3116,0.2015]]"),
            (29, "SWE-Pruner", "[[0.317,0.1901,0.415,0.2015]]"),
            (30, "41.1", "[[0.4604,0.1889,0.4893,0.2015]]"),
            (31, "64.0", "[[0.5429,0.1901,0.5737,0.2015]]"),
            (32, "0.670", "[[0.6373,0.1901,0.6768,0.2015]]"),
            (33, "Table 3", "[[0.1152,0.2201,0.1716,0.2314]]"),
            (
                34,
                "Comparison of context compression strategies on SWE-Bench.",
                "[[0.1824,0.2201,0.8842,0.2314]]",
            ),
            (38, "Methods", "[[0.1547,0.2821,0.2163,0.2947]]"),
            (123, "SWE-Pruner", "[[0.1547,0.4289,0.2527,0.4402]]"),
            (124, "5.56", "[[0.2987,0.4289,0.3296,0.4402]]"),
            (125, "58.63", "[[0.3491,0.4289,0.3886,0.4402]]"),
            (126, "31.5", "[[0.4081,0.4289,0.4389,0.4402]]"),
            (134, "Table 4", "[[0.1152,0.4589,0.1701,0.4702]]"),
            (
                145,
                "compression constraints.",
                "[[0.1158,0.5875,0.2894,0.6001]]",
            ),
            (146, "5.2", "[[0.1158,0.6165,0.1418,0.6303]]"),
            (
                147,
                "Performance on Single-Turn Tasks",
                "[[0.1622,0.6165,0.4692,0.6303]]",
            ),
        ] {
            conn.execute(
                "INSERT INTO document_lines
                    (id, document_id, page_no, line_no, block_id, block_index, text, bbox_json)
                 VALUES (?1, 'doc', 8, ?2, '', 0, ?3, ?4)",
                params![format!("page8-l{line_no}"), line_no, text, bbox],
            )
            .expect("line insert");
        }

        let tx = conn.transaction().expect("transaction");
        rebuild_visual_evidence(&tx, "doc").expect("rebuild visual evidence");
        tx.commit().expect("commit");

        let swe_pruner_tokens = conn
            .query_row(
                "SELECT f.value_text
                 FROM document_table_facts f
                 JOIN document_tables t ON t.id = f.table_id
                 WHERE t.caption LIKE 'Table 3%'
                   AND f.row_label = 'SWE-Pruner'
                   AND f.column_label = 'Tokens (M)'
                 LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("SWE-Pruner Table 3 tokens");
        assert_eq!(swe_pruner_tokens, "0.670");

        let leaked_heading_count = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM document_table_facts f
                 JOIN document_tables t ON t.id = f.table_id
                 WHERE t.caption LIKE 'Table 3%'
                   AND f.fact_text LIKE '%Performance on Single-Turn Tasks%'",
                [],
                |row| row.get::<_, u32>(0),
            )
            .expect("leaked heading count");
        assert_eq!(leaked_heading_count, 0);
    }

    #[test]
    fn table_caption_asset_gets_broad_crop_bbox_when_rows_are_missing() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        conn.execute(
            "INSERT INTO document_blocks
                (id, document_id, page_no, block_index, text, bbox_json, block_role)
             VALUES
                ('b-caption', 'doc', 2, 1,
                 'Table 2: Ablation scores.',
                 '[[0.20,0.20,0.70,0.23]]', 'caption')",
            [],
        )
        .expect("insert table caption");

        let tx = conn.transaction().expect("transaction");
        rebuild_visual_evidence(&tx, "doc").expect("rebuild visual evidence");
        tx.commit().expect("commit");

        let visual_bbox = conn
            .query_row(
                "SELECT bbox_json
                 FROM document_visual_assets
                 WHERE document_id = 'doc' AND asset_type = 'table'
                 LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("table visual asset");
        let bbox_json: serde_json::Value = serde_json::from_str(&visual_bbox).expect("bbox json");
        let rect = rect_from_bbox_list(&bbox_json).expect("table visual bbox");

        assert!(rect.y1 < 0.20);
        assert!(rect.y2 > 0.70);
        assert!(rect.x1 <= 0.04);
        assert!(rect.x2 >= 0.96);
    }

    #[test]
    fn rebuild_visual_evidence_allows_multiple_values_in_same_row_column() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_table7_fixture(&conn);
        conn.execute(
            "INSERT INTO document_lines
                (id, document_id, page_no, line_no, block_id, block_index, text, bbox_json)
             VALUES ('l87b', 'doc', 23, 94, '', 0, '77.9', '[[0.371,0.3612,0.399,0.3738]]')",
            [],
        )
        .expect("duplicate same-column value");

        let tx = conn.transaction().expect("transaction");
        rebuild_visual_evidence(&tx, "doc").expect("rebuild visual evidence with duplicate cells");
        tx.commit().expect("commit");

        let count = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM document_table_cells
                 WHERE table_id IN (SELECT id FROM document_tables WHERE document_id = 'doc')
                   AND row_index = 1
                   AND col_index = 1",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("duplicate value cell count");
        assert_eq!(count, 2);
    }

    #[test]
    fn dispatcher_searches_generated_table_facts() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_table7_fixture(&conn);
        let tx = conn.transaction().expect("transaction");
        rebuild_visual_evidence(&tx, "doc").expect("rebuild visual evidence");
        tx.commit().expect("commit");

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "search_table_facts",
            &serde_json::json!({
                "query": "SWE-bench Verified GLM-5",
                "limit": 8
            }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "search_table_facts");
        assert_eq!(output.tool_call.status, "ok");
        assert!(output
            .citations
            .iter()
            .any(|citation| citation.quote.contains("GLM-5 = 77.8")));
        assert!(output
            .citations
            .iter()
            .all(|citation| citation.source == "table_fact"));
    }

    #[test]
    fn dispatcher_opens_tables_and_visual_assets() {
        let mut conn = Connection::open_in_memory().expect("in-memory db");
        create_visual_evidence_schema(&conn);
        insert_table7_fixture(&conn);
        let tx = conn.transaction().expect("transaction");
        rebuild_visual_evidence(&tx, "doc").expect("rebuild visual evidence");
        tx.commit().expect("commit");

        let inspect_tables = execute_rag_tool_call(
            &conn,
            "doc",
            "inspect_tables",
            &serde_json::json!({ "query": "SWE-bench benchmark", "limit": 4 }),
            "fallback query",
        );
        assert_eq!(inspect_tables.tool_call.tool, "inspect_tables");
        assert_eq!(inspect_tables.tool_call.status, "ok");
        assert_eq!(inspect_tables.citations.len(), 1);
        assert!(inspect_tables
            .citations
            .iter()
            .any(|citation| citation.quote.contains("GLM-5 = 77.8")));
        let table_id = inspect_tables.citations[0].block_id.clone();

        let open_table = execute_rag_tool_call(
            &conn,
            "doc",
            "open_table",
            &serde_json::json!({ "tableId": table_id, "limit": 8 }),
            "fallback query",
        );
        assert_eq!(open_table.tool_call.tool, "open_table");
        assert!(open_table
            .citations
            .iter()
            .any(|citation| citation.source == "open_table_context"
                && citation.quote.contains("Caption:")
                && citation.quote.contains("Nearby text:")));
        assert!(open_table
            .citations
            .iter()
            .any(|citation| citation.quote.contains("GLM-5 = 77.8")));

        let inspect_visuals = execute_rag_tool_call(
            &conn,
            "doc",
            "inspect_visuals",
            &serde_json::json!({ "query": "comparison models", "assetType": "table" }),
            "fallback query",
        );
        assert_eq!(inspect_visuals.tool_call.tool, "inspect_visuals");
        assert_eq!(inspect_visuals.tool_call.status, "ok");
        assert_eq!(inspect_visuals.citations.len(), 1);
        let asset_id = inspect_visuals.citations[0].block_id.clone();

        let open_visual = execute_rag_tool_call(
            &conn,
            "doc",
            "open_visual",
            &serde_json::json!({ "assetId": asset_id }),
            "fallback query",
        );
        assert_eq!(open_visual.tool_call.tool, "open_visual");
        assert_eq!(open_visual.citations[0].source, "open_visual");
        assert!(open_visual.citations[0]
            .quote
            .contains("Opened visual asset: table"));
    }

    #[test]
    fn analyze_visual_requires_vision_capability() {
        let conn = Connection::open_in_memory().expect("in-memory db");

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "analyze_visual",
            &serde_json::json!({ "assetId": "visual-doc-p1-b1", "question": "What is shown?" }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "search_chunks");
        assert_eq!(output.tool_call.status, "fallback");
        assert!(output
            .tool_call
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("unknown RAG tool"));
    }

    #[test]
    fn dispatcher_unknown_tool_falls_back_without_panic() {
        let conn = Connection::open_in_memory().expect("in-memory db");

        let output = execute_rag_tool_call(
            &conn,
            "doc",
            "unknown_tool",
            &serde_json::json!({ "bad": true }),
            "fallback query",
        );

        assert_eq!(output.tool_call.tool, "search_chunks");
        assert_eq!(output.tool_call.status, "fallback");
        assert_eq!(output.tool_call.result_count, 0);
        assert!(output.tool_call.error.is_some());
    }

    #[test]
    fn merging_duplicate_citations_keeps_richer_section_metadata() {
        let mut accumulated = vec![citation("fts", None, "Same block text")];
        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "open_section",
                Some("3 Method"),
                "Section: 3 Method\nSame block text",
            )],
        );

        assert_eq!(accumulated.len(), 1);
        assert_eq!(accumulated[0].source, "open_section");
        assert_eq!(accumulated[0].section_title.as_deref(), Some("3 Method"));
        assert_eq!(accumulated[0].label, "[1]");
    }

    #[test]
    fn table_fact_citations_are_deduped_per_fact_not_per_table() {
        let mut accumulated = vec![citation(
            "table_fact",
            Some("Table 3"),
            "Table 3 | SWE-Pruner | Rounds = 41.1",
        )];
        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "table_fact",
                Some("Table 3"),
                "Table 3 | SWE-Pruner | Tokens (M) = 0.670",
            )],
        );

        assert_eq!(accumulated.len(), 2);
        assert_eq!(accumulated[0].label, "[1]");
        assert_eq!(accumulated[1].label, "[2]");
    }

    #[test]
    fn open_table_replaces_same_table_fact_quote() {
        let mut accumulated = vec![citation(
            "table_fact",
            Some("Table 3"),
            "Table 3 | SWE-Pruner | Tokens (M) = 0.670",
        )];
        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "open_table",
                Some("Table 3"),
                "Table 3 | SWE-Pruner | Tokens (M) = 0.670",
            )],
        );

        assert_eq!(accumulated.len(), 1);
        assert_eq!(accumulated[0].source, "open_table");
    }

    #[test]
    fn open_table_drops_stale_table_facts_from_same_table() {
        let mut accumulated = vec![citation(
            "table_fact",
            Some("Table 3"),
            "Table 3 | unrelated heading | Column 1 = 5.2",
        )];
        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "open_table",
                Some("Table 3"),
                "Table 3 | SWE-Pruner | Tokens (M) = 0.670",
            )],
        );

        assert_eq!(accumulated.len(), 1);
        assert_eq!(accumulated[0].source, "open_table");
        assert!(accumulated[0].quote.contains("SWE-Pruner"));
    }

    #[test]
    fn open_table_drops_resolved_anchor_from_same_table() {
        let mut accumulated = vec![citation(
            "table_anchor",
            Some("Table 3"),
            "Resolved table anchor: Table 3 on page 8\ntableId=b1",
        )];
        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "open_table",
                Some("Table 3"),
                "Table 3 | SWE-Pruner | Tokens (M) = 0.670",
            )],
        );

        assert_eq!(accumulated.len(), 1);
        assert_eq!(accumulated[0].source, "open_table");
    }

    #[test]
    fn open_visual_replaces_visual_anchor_from_same_asset() {
        let mut accumulated = vec![citation(
            "visual_anchor",
            Some("Figure 3"),
            "Resolved visual anchor: Figure 3 on page 4\nassetId=b1",
        )];
        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "open_visual",
                Some("figure on page 4"),
                "Visual asset: figure\nCaption: Figure 3\nNearby text: Pipeline details.",
            )],
        );

        assert_eq!(accumulated.len(), 1);
        assert_eq!(accumulated[0].source, "open_visual");
        assert!(accumulated[0].quote.contains("Pipeline details"));
    }

    #[test]
    fn open_table_rank_stays_above_table_anchor() {
        assert!(citation_source_rank("open_table") > citation_source_rank("table_anchor"));
        assert!(citation_source_rank("table_fact") > citation_source_rank("table_anchor"));
    }

    #[test]
    fn open_visual_rank_stays_above_visual_anchor() {
        assert!(citation_source_rank("open_visual") > citation_source_rank("visual_anchor"));
        assert!(citation_source_rank("analyze_visual") > citation_source_rank("open_visual"));
        assert!(citation_source_rank("analyze_page") > citation_source_rank("inspect_objects"));
    }

    #[test]
    fn initial_retrieval_limits_scale_with_context_budget() {
        let small = initial_retrieval_limits(&crate::model_catalog::ModelContextBudget::default());
        let long = initial_retrieval_limits(
            &crate::model_catalog::ModelContextBudget::from_model_limits(256_000, 16_384, "test"),
        );

        assert!(long.tree > small.tree);
        assert!(long.per_section > small.per_section);
        assert!(long.page_blocks > small.page_blocks);
    }

    #[test]
    fn high_rank_table_evidence_can_enter_when_citation_cap_is_full() {
        let mut accumulated = (0..MAX_ACCUMULATED_CITATIONS)
            .map(|index| Citation {
                id: format!("fts-{index}"),
                label: format!("[{}]", index + 1),
                page: index as u32 + 1,
                block_id: format!("block-{index}"),
                section_title: None,
                quote: format!("Plain text evidence {index}"),
                bbox_list: serde_json::json!([]),
                document_id: "doc".to_string(),
                source: "fts".to_string(),
            })
            .collect::<Vec<_>>();

        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "open_table",
                Some("Table 3"),
                "Table 3 | SWE-Pruner | Tokens (M) = 0.670",
            )],
        );

        assert_eq!(accumulated.len(), MAX_ACCUMULATED_CITATIONS);
        assert!(accumulated
            .iter()
            .any(|citation| citation.source == "open_table"));
    }

    #[test]
    fn contextual_evidence_gets_a_slot_when_table_facts_fill_cap() {
        let mut accumulated = (0..MAX_ACCUMULATED_CITATIONS)
            .map(|index| Citation {
                id: format!("table-{index}"),
                label: format!("[{}]", index + 1),
                page: 8,
                block_id: format!("table-block-{index}"),
                section_title: Some("Table 3".to_string()),
                quote: format!("Table fact {index}"),
                bbox_list: serde_json::json!([]),
                document_id: "doc".to_string(),
                source: "table_fact".to_string(),
            })
            .collect::<Vec<_>>();

        merge_retrieval_citations(
            &mut accumulated,
            &[citation(
                "open_section",
                Some("Case Study"),
                "SWE-Pruner demonstrates consistent effectiveness across different model architectures.",
            )],
        );

        assert_eq!(accumulated.len(), MAX_ACCUMULATED_CITATIONS);
        assert!(accumulated
            .iter()
            .any(|citation| citation.source == "open_section"));
    }

    #[test]
    fn open_table_context_survives_many_open_table_rows() {
        let mut accumulated = (0..MAX_ACCUMULATED_CITATIONS)
            .map(|index| Citation {
                id: format!("table-fact-{index}"),
                label: format!("[{}]", index + 1),
                page: 19,
                block_id: format!("table-fact-block-{index}"),
                section_title: Some("Table 6".to_string()),
                quote: format!("Table 6 | stale fact {index}"),
                bbox_list: serde_json::json!([]),
                document_id: "doc".to_string(),
                source: "table_fact".to_string(),
            })
            .collect::<Vec<_>>();
        let mut incoming = vec![Citation {
            id: "table-context-6".to_string(),
            label: "[1]".to_string(),
            page: 19,
            block_id: "table-6:context".to_string(),
            section_title: Some("Table 6".to_string()),
            quote: "Caption: Table 6 Average TTFT.\n\nNearby text:\nThe latency analysis reveals critical insights.".to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "open_table_context".to_string(),
        }];
        incoming.extend((0..40).map(|index| Citation {
            id: format!("open-table-{index}"),
            label: format!("[{}]", index + 2),
            page: 19,
            block_id: format!("table-6-row-{index}"),
            section_title: Some("Table 6".to_string()),
            quote: format!("Table 6 | row {index} | value = {index}"),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "open_table".to_string(),
        }));

        merge_retrieval_citations(&mut accumulated, &incoming);

        assert_eq!(accumulated.len(), MAX_ACCUMULATED_CITATIONS);
        assert!(accumulated
            .iter()
            .any(|citation| citation.source == "open_table_context"));
        assert!(accumulated
            .iter()
            .any(|citation| citation.source == "open_table"));
    }

    fn citation(source: &str, section_title: Option<&str>, quote: &str) -> Citation {
        Citation {
            id: format!("citation-{source}"),
            label: "[1]".to_string(),
            page: 2,
            block_id: "b1".to_string(),
            section_title: section_title.map(str::to_string),
            quote: quote.to_string(),
            bbox_list: serde_json::json!([[0, 0, 10, 10]]),
            document_id: "doc".to_string(),
            source: source.to_string(),
        }
    }

    fn page_candidate(block_id: &str, quote: &str, role: &str) -> EvidenceCandidate {
        EvidenceCandidate {
            chunk_id: format!("chunk-{block_id}"),
            document_id: "doc".to_string(),
            page: 1,
            block_id: block_id.to_string(),
            section_title: None,
            quote: quote.to_string(),
            bbox_list: serde_json::json!([[0, 0, 10, 10]]),
            score: 0.0,
            source: "open_pages".to_string(),
            tree_node_id: None,
            block_role: Some(role.to_string()),
        }
    }

    fn structure_block(
        page_no: u32,
        block_index: u32,
        text: &str,
        role: &str,
    ) -> StructureBlockSeed {
        StructureBlockSeed {
            page_no,
            block_index,
            text: text.to_string(),
            bbox_list: serde_json::json!([]),
            role: role.to_string(),
            region_index: 0,
            region_id: String::new(),
        }
    }

    fn outline_seed(title: &str, level: u32, page_no: u32, order_index: u32) -> OutlineSeed {
        OutlineSeed {
            title: title.to_string(),
            level,
            page_no,
            order_index,
        }
    }

    fn create_visual_evidence_schema(conn: &Connection) {
        conn.execute_batch(
            "CREATE TABLE document_blocks (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                block_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                block_role TEXT NOT NULL
            );
            CREATE TABLE document_lines (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                line_no INTEGER NOT NULL,
                block_id TEXT NOT NULL DEFAULT '',
                block_index INTEGER NOT NULL DEFAULT 0,
                text TEXT NOT NULL,
                bbox_json TEXT NOT NULL
            );
            CREATE TABLE document_visual_assets (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                asset_type TEXT NOT NULL,
                caption TEXT NOT NULL DEFAULT '',
                bbox_json TEXT NOT NULL DEFAULT '[]',
                image_path TEXT NOT NULL DEFAULT '',
                ocr_text TEXT NOT NULL DEFAULT '',
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
                USING fts5(fact_id UNINDEXED, document_id UNINDEXED, table_id UNINDEXED, text);",
        )
        .expect("visual evidence schema");
    }

    fn insert_numbered_table_fact(
        conn: &Connection,
        table_id: &str,
        page_no: u32,
        caption: &str,
        row_label: &str,
        column_label: &str,
        value_text: &str,
    ) {
        conn.execute(
            "INSERT OR IGNORE INTO document_tables
                (id, document_id, page_no, caption, bbox_json, source, confidence, created_at)
             VALUES (?1, 'doc', ?2, ?3, '[[0.1,0.1,0.9,0.4]]', 'pdf_bbox', 0.9, 1)",
            params![table_id, page_no, caption],
        )
        .expect("table insert");
        let fact_id = format!(
            "{}-{}-{}",
            table_id,
            normalize_for_dedupe(row_label),
            normalize_for_dedupe(column_label)
        );
        let fact_text = format!("{caption} | {row_label} | {column_label} = {value_text}");
        conn.execute(
            "INSERT OR REPLACE INTO document_table_facts
                (id, document_id, table_id, page_no, row_label, column_label, value_text,
                 fact_text, bbox_json, source, confidence)
             VALUES (?1, 'doc', ?2, ?3, ?4, ?5, ?6, ?7, '[[0.1,0.1,0.2,0.2]]', 'pdf_bbox', 0.9)",
            params![
                fact_id,
                table_id,
                page_no,
                row_label,
                column_label,
                value_text,
                fact_text
            ],
        )
        .expect("table fact insert");
    }

    fn insert_visual_asset_fixture(
        conn: &Connection,
        asset_id: &str,
        page_no: u32,
        asset_type: &str,
        caption: &str,
        nearby_text: &str,
    ) {
        conn.execute(
            "INSERT INTO document_visual_assets
                (id, document_id, page_no, asset_type, caption, bbox_json, image_path,
                 nearby_text, linked_block_ids_json, source, confidence, created_at)
             VALUES (?1, 'doc', ?2, ?3, ?4, '[[0.1,0.1,0.9,0.5]]',
                     '/tmp/object.png', ?5, '[]', 'caption', 0.9, 1)",
            params![asset_id, page_no, asset_type, caption, nearby_text],
        )
        .expect("visual asset insert");
    }

    fn insert_table7_fixture(conn: &Connection) {
        conn.execute(
            "INSERT INTO document_blocks
                (id, document_id, page_no, block_index, text, bbox_json, block_role)
             VALUES
                ('b-caption', 'doc', 23, 1,
                 'Table 7: Comparison between GLM-5 and open-source/proprietary models.',
                 '[[0.176,0.0968,0.8252,0.1094]]', 'caption')",
            [],
        )
        .expect("caption block");
        for (line_no, text, bbox) in [
            (
                1,
                "Table 7: Comparison between GLM-5 and open-source/proprietary models.",
                "[[0.176,0.0968,0.8252,0.1094]]",
            ),
            (
                5,
                "Bench 2.0, fixing some ambiguous instructions. The GDPval-AA Elo scores are recorded on 15th",
                "[[0.1765,0.1243,0.8235,0.1369]]",
            ),
            (
                6,
                "Feb., 2026. The highest score for each benchmark is",
                "[[0.1765,0.1381,0.5172,0.1507]]",
            ),
            (7, "bolded", "[[0.5213,0.1381,0.5683,0.1507]]"),
            (
                8,
                ", and the second highest is underlined.",
                "[[0.5683,0.1381,0.817,0.1507]]",
            ),
            (9, "DeepSeek", "[[0.494,0.158,0.5619,0.1706]]"),
            (10, "Kimi", "[[0.5764,0.158,0.6116,0.1706]]"),
            (11, "Claude", "[[0.632,0.158,0.6818,0.1706]]"),
            (12, "Gemini", "[[0.6974,0.158,0.749,0.1706]]"),
            (13, "GPT-5.2", "[[0.7588,0.158,0.8165,0.1706]]"),
            (14, "GLM-5 GLM-4.7", "[[0.3574,0.1659,0.494,0.1785]]"),
            (15, "-V3.2", "[[0.5092,0.1738,0.5467,0.1864]]"),
            (16, "K2.5", "[[0.5775,0.1738,0.6105,0.1864]]"),
            (17, "Opus 4.5", "[[0.6262,0.1738,0.6877,0.1864]]"),
            (18, "3 Pro", "[[0.7046,0.1738,0.7418,0.1864]]"),
            (19, "(xhigh)", "[[0.7628,0.1738,0.8125,0.1864]]"),
            (85, "Coding", "[[0.1835,0.3454,0.2314,0.358]]"),
            (86, "SWE-bench Verified", "[[0.1835,0.3612,0.3178,0.3738]]"),
            (87, "77.8", "[[0.3694,0.3612,0.3979,0.3738]]"),
            (88, "73.8", "[[0.4377,0.3612,0.4662,0.3738]]"),
            (89, "73.1", "[[0.5137,0.3612,0.5422,0.3738]]"),
            (90, "76.8", "[[0.5798,0.3612,0.6083,0.3738]]"),
            (91, "80.9", "[[0.6427,0.3612,0.6712,0.3738]]"),
            (92, "76.2", "[[0.709,0.3612,0.7375,0.3738]]"),
            (93, "80.0", "[[0.7734,0.3612,0.8019,0.3738]]"),
            (
                102,
                "Terminal-Bench 2.0",
                "[[0.1835,0.3929,0.3135,0.4055]]",
            ),
            (103, "56.2 /", "[[0.3651,0.393,0.4022,0.4056]]"),
            (104, "41.0", "[[0.4377,0.4008,0.4662,0.4134]]"),
            (105, "39.3", "[[0.5137,0.4008,0.5422,0.4134]]"),
            (106, "50.8", "[[0.5798,0.4008,0.6083,0.4134]]"),
            (107, "59.3", "[[0.6427,0.4008,0.6712,0.4134]]"),
            (108, "54.2", "[[0.709,0.4008,0.7375,0.4134]]"),
            (109, "54.0", "[[0.7734,0.4008,0.8019,0.4134]]"),
            (
                110,
                "(Terminus-2)",
                "[[0.1872,0.41,0.2643,0.4213]]",
            ),
            (111, "60.7", "[[0.366,0.4078,0.3945,0.4204]]"),
            (112, "†", "[[0.3945,0.407,0.4004,0.4158]]"),
        ] {
            conn.execute(
                "INSERT INTO document_lines
                    (id, document_id, page_no, line_no, block_id, block_index, text, bbox_json)
                 VALUES (?1, 'doc', 23, ?2, '', 0, ?3, ?4)",
                params![format!("l{line_no}"), line_no, text, bbox],
            )
            .expect("line insert");
        }
    }

    fn setup_cross_doc_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                short_title TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE structure_tree_nodes (
                id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                level INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                block_start_index INTEGER NOT NULL,
                block_end_index INTEGER NOT NULL,
                order_index INTEGER NOT NULL
            );
            CREATE TABLE document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                block_ids_json TEXT NOT NULL,
                text TEXT NOT NULL,
                bbox_refs_json TEXT NOT NULL
            );",
        )
        .expect("cross-doc schema");
        conn
    }

    fn insert_cross_doc_chunk(conn: &Connection, document_id: &str, chunk_id: &str, text: &str) {
        conn.execute(
            "INSERT INTO document_chunks
                (id, document_id, page_no, block_ids_json, text, bbox_refs_json)
             VALUES (?1, ?2, 1, '[]', ?3, '[]')",
            params![chunk_id, document_id, text],
        )
        .expect("insert chunk");
    }

    fn cross_doc_caps() -> RagToolCapabilities {
        RagToolCapabilities {
            vision_enabled: false,
            web_enabled: false,
            max_quote_chars: 400,
        }
    }

    #[test]
    fn search_chunks_literal_matches_exact_token() {
        let conn = setup_cross_doc_conn();
        // FTS tokenization can split tokens like "F1-score"; the literal substring
        // path must still match the raw string regardless of case.
        insert_cross_doc_chunk(
            &conn,
            "doc-1",
            "chunk-1",
            "We report an F1-score of 0.91 on the dev set.",
        );
        let hits = search_chunks_literal(&conn, "doc-1", "f1-score", 5).expect("literal search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, "literal");
    }

    #[test]
    fn search_chunks_literal_handles_empty_query() {
        let conn = setup_cross_doc_conn();
        let hits = search_chunks_literal(&conn, "doc-1", "   ", 5).expect("literal search");
        assert!(hits.is_empty());
    }

    #[test]
    fn dispatch_routes_to_whitelisted_reference_document() {
        let conn = setup_cross_doc_conn();
        insert_cross_doc_chunk(&conn, "primary", "p1", "primary doc mentions apples");
        insert_cross_doc_chunk(&conn, "ref-1", "r1", "reference doc mentions apples too");
        let args =
            serde_json::json!({ "query": "apples", "mode": "literal", "documentId": "ref-1" });
        let out = execute_rag_tool_call_for_capabilities(
            &conn,
            "primary",
            &["ref-1"],
            "search_chunks",
            &args,
            "apples",
            cross_doc_caps(),
        );
        assert!(!out.citations.is_empty());
        assert!(out.citations.iter().all(|c| c.document_id == "ref-1"));
    }

    #[test]
    fn dispatch_ignores_non_whitelisted_document_id() {
        let conn = setup_cross_doc_conn();
        insert_cross_doc_chunk(&conn, "primary", "p1", "primary doc mentions apples");
        insert_cross_doc_chunk(&conn, "evil", "e1", "evil doc mentions apples");
        // "evil" is not in the whitelist -> dispatch must fall back to the primary doc.
        let args =
            serde_json::json!({ "query": "apples", "mode": "literal", "documentId": "evil" });
        let out = execute_rag_tool_call_for_capabilities(
            &conn,
            "primary",
            &[],
            "search_chunks",
            &args,
            "apples",
            cross_doc_caps(),
        );
        assert!(out.citations.iter().all(|c| c.document_id == "primary"));
    }

    #[test]
    fn workspace_manifest_ranks_and_tags_documents() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (
               id TEXT PRIMARY KEY,
               short_title TEXT NOT NULL DEFAULT '',
               title TEXT NOT NULL DEFAULT '',
               path TEXT NOT NULL DEFAULT '',
               page_count INTEGER NOT NULL DEFAULT 0,
               index_status TEXT NOT NULL DEFAULT 'indexed',
               last_opened_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE document_blocks (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               text TEXT NOT NULL,
               block_role TEXT NOT NULL DEFAULT 'body'
             );
             INSERT INTO documents (id, short_title, title, path, page_count, index_status, last_opened_at) VALUES
               ('focus', 'Focus', 'Focus Paper', '/lib/a/focus.pdf', 10, 'indexed', 100),
               ('ref',   'Refd',  'Referenced',  '/lib/b/ref.pdf',   8,  'indexed', 90),
               ('hit',   'Hitt',  'Relevant',    '/lib/c/hit.pdf',   6,  'indexed', 80),
               ('miss',  'Misss', 'Unrelated',   '/lib/c/miss.pdf',  4,  'indexed', 70),
               ('pending','Pend', 'Not Indexed', '/lib/c/pend.pdf',  4,  'pending', 60);
             INSERT INTO document_blocks (id, document_id, text, block_role) VALUES
               ('b1', 'hit', 'This paper studies transformer attention mechanisms.', 'abstract'),
               ('b2', 'miss', 'A cookbook about soup recipes.', 'abstract');",
        )
        .expect("seed");

        let manifest = load_workspace_manifest(&conn, "transformer attention", "focus", &["ref"])
            .expect("manifest");

        // Pending (not indexed) doc is excluded.
        assert!(!manifest.document_ids.iter().any(|id| id == "pending"));
        // Focus leads and is tagged; @-referenced doc is tagged.
        assert_eq!(manifest.entries[0].document_id, "focus");
        assert!(manifest.entries[0].is_focus);
        assert!(manifest
            .entries
            .iter()
            .any(|e| e.document_id == "ref" && e.is_referenced));
        // The relevant doc outranks the unrelated one and got a summary.
        let hit = manifest
            .entries
            .iter()
            .find(|e| e.document_id == "hit")
            .expect("hit");
        assert!(hit.summary.contains("attention"));
        assert_eq!(hit.rel_dir, "c");
        let prompt = manifest.to_prompt_block();
        assert!(prompt.contains("[CURRENT FOCUS]"));
        assert!(prompt.contains("[@referenced]"));
        // Four indexed docs (pending excluded) — well under the large threshold.
        assert_eq!(manifest.total_indexed, 4);
        assert!(!manifest.is_large());
        assert_eq!(manifest.all_document_ids.len(), 4);
        assert!(!manifest.all_document_ids.iter().any(|id| id == "pending"));
    }

    #[test]
    fn search_workspace_documents_ranks_excludes_and_browses() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE documents (
               id TEXT PRIMARY KEY,
               short_title TEXT NOT NULL DEFAULT '',
               title TEXT NOT NULL DEFAULT '',
               path TEXT NOT NULL DEFAULT '',
               page_count INTEGER NOT NULL DEFAULT 0,
               index_status TEXT NOT NULL DEFAULT 'indexed',
               last_opened_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE document_blocks (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               text TEXT NOT NULL,
               block_role TEXT NOT NULL DEFAULT 'body'
             );
             INSERT INTO documents (id, short_title, title, path, page_count, index_status, last_opened_at) VALUES
               ('focus', 'Focus', 'Focus Paper', '/lib/a/focus.pdf', 10, 'indexed', 100),
               ('hit',   'Hitt',  'Relevant',    '/lib/c/hit.pdf',   6,  'indexed', 80),
               ('miss',  'Misss', 'Unrelated',   '/lib/c/miss.pdf',  4,  'indexed', 70),
               ('pending','Pend', 'Not Indexed', '/lib/c/pend.pdf',  4,  'pending', 60);
             INSERT INTO document_blocks (id, document_id, text, block_role) VALUES
               ('b1', 'hit', 'This paper studies transformer attention mechanisms.', 'abstract'),
               ('b2', 'miss', 'A cookbook about soup recipes.', 'abstract');",
        )
        .expect("seed");

        // Topical search: the relevant doc ranks first; the non-indexed doc is gone.
        let hits =
            search_workspace_documents(&conn, "transformer attention", 10, &[]).expect("search");
        assert_eq!(hits[0].document_id, "hit");
        assert!(hits.iter().all(|hit| hit.document_id != "pending"));

        // `exclude` removes already-pinned docs.
        let excluded =
            search_workspace_documents(&conn, "transformer", 10, &["hit"]).expect("search");
        assert!(excluded.iter().all(|hit| hit.document_id != "hit"));

        // Empty query = browse by recency (most recently opened first).
        let browse = search_workspace_documents(&conn, "", 10, &[]).expect("browse");
        assert_eq!(browse[0].document_id, "focus");
    }

    #[test]
    fn compact_prompt_block_lists_only_pinned_for_large_library() {
        let entries = vec![
            DocManifestEntry {
                document_id: "focus".into(),
                title: "Focus Paper".into(),
                rel_dir: "a".into(),
                page_count: 10,
                summary: String::new(),
                is_focus: true,
                is_referenced: false,
            },
            DocManifestEntry {
                document_id: "ref".into(),
                title: "Referenced".into(),
                rel_dir: "b".into(),
                page_count: 8,
                summary: String::new(),
                is_focus: false,
                is_referenced: true,
            },
            DocManifestEntry {
                document_id: "other".into(),
                title: "Some Other Paper".into(),
                rel_dir: "c".into(),
                page_count: 5,
                summary: String::new(),
                is_focus: false,
                is_referenced: false,
            },
        ];
        let manifest = WorkspaceManifest {
            entries,
            document_ids: Vec::new(),
            total_indexed: 40,
            all_document_ids: Vec::new(),
        };
        assert!(manifest.is_large());
        let compact = manifest.to_prompt_block_compact();
        assert!(compact.contains("40 indexed documents"));
        assert!(compact.contains("search_library"));
        // Only the pinned (focus + @referenced) docs are inlined.
        assert!(compact.contains("Focus Paper"));
        assert!(compact.contains("Referenced"));
        assert!(!compact.contains("Some Other Paper"));
    }

    fn setup_chat_history_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE chat_turns (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_answer TEXT NOT NULL,
                index_version INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                rowid_helper INTEGER
            );",
        )
        .expect("chat_turns schema");
        conn
    }

    fn insert_chat_turn(
        conn: &Connection,
        id: &str,
        document_id: &str,
        user_message: &str,
        assistant_answer: &str,
        created_at: i64,
    ) {
        insert_chat_turn_versioned(
            conn,
            id,
            document_id,
            user_message,
            assistant_answer,
            created_at,
            crate::CURRENT_INDEX_VERSION,
        );
    }

    fn insert_chat_turn_versioned(
        conn: &Connection,
        id: &str,
        document_id: &str,
        user_message: &str,
        assistant_answer: &str,
        created_at: i64,
        index_version: i64,
    ) {
        conn.execute(
            "INSERT INTO chat_turns
                (id, document_id, user_message, assistant_answer, index_version, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id,
                document_id,
                user_message,
                assistant_answer,
                index_version,
                created_at
            ],
        )
        .expect("insert chat turn");
    }

    #[test]
    fn recall_chat_history_returns_recent_turns_when_query_empty() {
        let conn = setup_chat_history_conn();
        insert_chat_turn(&conn, "t1", "doc", "first question", "first answer", 1);
        insert_chat_turn(&conn, "t2", "doc", "second question", "second answer", 2);
        let hits = recall_chat_history(&conn, "doc", "", false, 5).expect("recall");
        assert_eq!(hits.len(), 2);
        // Newest-first.
        assert_eq!(hits[0].block_id, "t2");
        assert_eq!(hits[0].source, "chat_history");
        assert!(hits[0].quote.contains("second question"));
    }

    #[test]
    fn recall_chat_history_keyword_ranks_by_term_overlap() {
        let conn = setup_chat_history_conn();
        insert_chat_turn(
            &conn,
            "t1",
            "doc",
            "we discussed pruning",
            "the pruning method",
            1,
        );
        insert_chat_turn(
            &conn,
            "t2",
            "doc",
            "unrelated weather chat",
            "sunny today",
            2,
        );
        let hits = recall_chat_history(&conn, "doc", "pruning method", false, 5).expect("recall");
        // The matching turn ranks first; non-matching turns degrade to recent (no
        // longer filtered out, so the agent still gets a fallback instead of empty).
        assert_eq!(hits[0].block_id, "t1");
    }

    #[test]
    fn recall_chat_history_keyword_ranks_reworded_cjk_query() {
        let conn = setup_chat_history_conn();
        // Stored turn uses "剪枝方法"; the query rewords it as "模型剪枝的方法".
        // Han chars aren't split by query_terms, so single-char CJK overlap must
        // still rank the relevant turn above an unrelated one.
        insert_chat_turn(
            &conn,
            "t1",
            "doc",
            "请解释剪枝方法",
            "剪枝方法是一种压缩技术",
            1,
        );
        insert_chat_turn(&conn, "t2", "doc", "今天天气怎么样", "晴天", 2);
        let hits = recall_chat_history(&conn, "doc", "模型剪枝的方法", false, 5).expect("recall");
        assert_eq!(hits[0].block_id, "t1");
    }

    #[test]
    fn recall_chat_history_excludes_stale_index_version() {
        let conn = setup_chat_history_conn();
        // A turn from an older index version must NOT be recalled (matches the
        // visible-history loader, which filters by CURRENT_INDEX_VERSION).
        insert_chat_turn_versioned(
            &conn,
            "stale",
            "doc",
            "old pruning question",
            "old answer",
            1,
            crate::CURRENT_INDEX_VERSION - 1,
        );
        insert_chat_turn(
            &conn,
            "fresh",
            "doc",
            "new pruning question",
            "new answer",
            2,
        );
        let hits = recall_chat_history(&conn, "doc", "pruning", false, 5).expect("recall");
        assert!(hits.iter().all(|h| h.block_id != "stale"));
        assert!(hits.iter().any(|h| h.block_id == "fresh"));
    }

    #[test]
    fn recall_chat_history_literal_matches_exact_substring() {
        let conn = setup_chat_history_conn();
        insert_chat_turn(
            &conn,
            "t1",
            "doc",
            "what is the F1-score",
            "the F1-score is 0.9",
            1,
        );
        insert_chat_turn(&conn, "t2", "doc", "general question", "general answer", 2);
        let hits = recall_chat_history(&conn, "doc", "f1-score", true, 5).expect("recall");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].block_id, "t1");
    }

    #[test]
    fn recall_chat_history_scopes_to_document() {
        let conn = setup_chat_history_conn();
        insert_chat_turn(&conn, "t1", "doc-a", "apples in a", "answer a", 1);
        insert_chat_turn(&conn, "t2", "doc-b", "apples in b", "answer b", 2);
        let hits = recall_chat_history(&conn, "doc-a", "apples", false, 5).expect("recall");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_id, "doc-a");
    }
}
