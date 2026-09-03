use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{Emitter, Manager, State};

mod agent_judge;
mod backup;
mod collections;
pub mod debug_probe;
mod diagnostics;
mod document_index;
mod document_translation;
mod documents;
mod indexing;
mod knowledge_api;
mod llm;
mod local_agent;
mod model_catalog;
mod net;
mod office;
mod pdf2zh_sidecar;
mod pdf_annotation;
mod pdf_index;
mod pdf_layout_dump;
mod providers;
mod runtime;
mod search_text;
mod storage;
mod translation;
mod trending;
mod update_check;
mod vault;
mod vision;
mod visual_index;

const CURRENT_INDEX_VERSION: i64 = 28;
const MICROSOFT_TRANSLATOR_DEFAULT_ENDPOINT: &str = "https://api.cognitive.microsofttranslator.com";
const MAX_LLM_JUDGE_STEPS: u32 = 8;
const CURRENT_VIEW_JUDGE_TIMEOUT_SECS: u64 = 25;
const MAX_RUST_TSR_TABLES: usize = 12;
const MAX_RUST_VISUAL_CROPS: usize = 48;
const MIN_RUST_TSR_CONFIDENCE: f64 = 0.15;
const TEXT_INDEX_JOB_TYPE: &str = "text_pdf";

#[derive(Default)]
struct PdfRegistry {
    paths: Mutex<HashMap<String, PathBuf>>,
}

struct AppDatabase {
    conn: Mutex<Connection>,
}

/// In-flight chat generations, keyed by their activity event id, so `stop_ask_document`
/// can cancel one cooperatively (streaming loops check the token; local-agent
/// subprocesses are killed). Entries are removed when the generation finishes.
#[derive(Default)]
pub(crate) struct AskCancellations(
    Mutex<std::collections::HashMap<String, tokio_util::sync::CancellationToken>>,
);

/// The cancellation token for an in-flight generation, if one is registered.
pub(crate) fn cancellation_token(
    app: &tauri::AppHandle,
    event_id: &str,
) -> Option<tokio_util::sync::CancellationToken> {
    app.try_state::<AskCancellations>().and_then(|state| {
        state
            .0
            .lock()
            .ok()
            .and_then(|map| map.get(event_id).cloned())
    })
}

type AgentSessionState = runtime::agent::AgentSessionStore;

#[derive(Clone, Serialize)]
struct PdfDocument {
    id: String,
    workspace_root_id: String,
    title: String,
    short_title: String,
    path: String,
    /// Source kind: 'pdf' today; the KB pivot adds 'docx'|'web'|'note'|… later.
    content_type: String,
    /// Logical collection membership (None = unfiled). Decoupled from disk.
    collection_id: Option<String>,
    /// Manual sibling order within the collection (ascending). Ties break on title.
    position: i64,
    size: u64,
    modified: u64,
    page_count: u32,
    current_page: u32,
    index_status: String,
    index_version: i64,
    current_index_version: i64,
    tree_ready: bool,
    visual_index_status: String,
    visual_index_version: i64,
    visual_index_error: String,
}

#[derive(Serialize)]
struct WorkspaceSnapshot {
    roots: Vec<WorkspaceRootSnapshot>,
}

#[derive(Serialize)]
struct WorkspaceRootSnapshot {
    root: WorkspaceRoot,
    documents: Vec<PdfDocument>,
}

#[derive(Serialize)]
struct WorkspaceRoot {
    id: String,
    path: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpsertDocumentIndexInput {
    document_id: String,
    page_count: u32,
    pages: Vec<PageIndexInput>,
    outlines: Option<Vec<OutlineIndexInput>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageIndexInput {
    page_no: u32,
    width: f64,
    height: f64,
    text: String,
    blocks: Vec<BlockIndexInput>,
    lines: Option<Vec<LineIndexInput>>,
    #[serde(default)]
    units: Option<Vec<TextUnitIndexInput>>,
    #[serde(default)]
    regions: Option<Vec<LayoutRegionIndexInput>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BlockIndexInput {
    id: String,
    block_index: u32,
    text: String,
    bbox_list: serde_json::Value,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    region_index: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LineIndexInput {
    line_no: u32,
    text: String,
    bbox_list: serde_json::Value,
    #[serde(default)]
    source_order_list: Option<Vec<u32>>,
    #[serde(default)]
    region_index: Option<u32>,
    #[serde(default)]
    role_hint: Option<String>,
    #[serde(default)]
    baseline: Option<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextUnitIndexInput {
    source_order: u32,
    text: String,
    bbox: serde_json::Value,
    #[serde(default)]
    font_size: f64,
    #[serde(default)]
    font_name: String,
    #[serde(default)]
    font_flags: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LayoutRegionIndexInput {
    region_index: u32,
    kind: String,
    bbox: serde_json::Value,
    line_numbers: Vec<u32>,
    confidence: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OutlineIndexInput {
    title: String,
    level: u32,
    page_start: u32,
    order_index: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DocumentIndexResult {
    page_count: u32,
    index_version: i64,
    tree_ready: bool,
    visual_index_status: String,
    visual_index_version: i64,
    visual_index_error: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct DocumentIndexEventOutput {
    document_id: String,
    status: String,
    progress_percent: u32,
    stage: String,
    stage_label: String,
    page_count: u32,
    index_version: i64,
    tree_ready: bool,
    visual_index_status: String,
    visual_index_version: i64,
    visual_index_error: String,
    error: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VisualIndexResult {
    document_id: String,
    status: String,
    version: i64,
    error: String,
    visual_assets: u32,
    tables: u32,
    table_facts: u32,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct VisualIndexEventOutput {
    document_id: String,
    status: String,
    version: i64,
    error: String,
    visual_assets: u32,
    tables: u32,
    table_facts: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResult {
    chunk_id: String,
    document_id: String,
    page: u32,
    block_id: String,
    quote: String,
    bbox_list: serde_json::Value,
    rank: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReadingStateInput {
    document_id: String,
    page: u32,
    zoom: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TranslateTextInput {
    document_id: String,
    page: u32,
    block_id: Option<String>,
    text: String,
    target_lang: String,
    provider: Option<String>,
    source_version: Option<String>,
    force_refresh: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TranslateTextOutput {
    translated_text: String,
    provider: String,
    cached: bool,
    source_hash: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AskDocumentInput {
    document_id: String,
    /// Agent workspace session this turn belongs to. When empty (legacy callers),
    /// the backend falls back to the per-document session `migrated-<document_id>`,
    /// keeping single-document behavior identical to before the session refactor.
    #[serde(default)]
    session_id: Option<String>,
    question: String,
    locale: Option<String>,
    model_provider_id: Option<String>,
    model_key: Option<String>,
    selected_text: Option<String>,
    selected_block_id: Option<String>,
    selected_bbox_list: Option<serde_json::Value>,
    image_data_url: Option<String>,
    page: Option<u32>,
    viewport_context: Option<ViewportContextInput>,
    max_retrieval_steps: Option<u32>,
    retrieval_attempt_offset: Option<u32>,
    activity_event_id: Option<String>,
    /// Other indexed documents the user "@-referenced" in the composer. The agent
    /// may search these as an additional retrieval dimension (see runtime::rag
    /// documentId routing). Normalized + capped at MAX_REFERENCE_DOCS before use.
    #[serde(default)]
    reference_document_ids: Option<Vec<String>>,
    /// Whether knowledge precipitation is enabled (Stream 2 conversation
    /// sedimentation). Defaults to true for legacy callers.
    #[serde(default)]
    knowledge_enabled: Option<bool>,
    /// Whether the chat's "联网" (web search) toggle is on for this turn. Gates the
    /// web_search/web_fetch tools in the agentic + tool-calling paths. Default off.
    #[serde(default)]
    web_enabled: Option<bool>,
    /// Which app surface the user is on right now (reader / trending / graph),
    /// so the agent can resolve "the current trending papers" to the right list
    /// and knows when the question is about something other than the focus PDF.
    #[serde(default)]
    view_context: Option<ViewContextInput>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewContextInput {
    /// "reader" | "trending" | "graph" (free-form; only used for an ambient hint).
    surface: Option<String>,
    /// daily | weekly | monthly — the trending period currently shown.
    trending_period: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewportContextInput {
    active_page: Option<u32>,
    #[serde(default)]
    visible_pages: Vec<u32>,
    selection_preview: Option<String>,
    captured_at: Option<i64>,
    sensitivity: Option<String>,
    source: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoadChatTurnsInput {
    /// Focus document (legacy field). Used only to derive the fallback session id
    /// when `session_id` is absent.
    #[serde(default)]
    document_id: String,
    #[serde(default)]
    session_id: Option<String>,
    limit: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClearChatTurnsInput {
    #[serde(default)]
    document_id: String,
    #[serde(default)]
    session_id: Option<String>,
    turn_ids: Option<Vec<String>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatHistoryMessageOutput {
    id: String,
    turn_id: String,
    role: String,
    content: String,
    reasoning_content: Option<String>,
    provider: Option<String>,
    citations: Vec<runtime::rag::Citation>,
    claims: Vec<AskDocumentClaim>,
    retrieval_trace: Option<serde_json::Value>,
    image_data_url: Option<String>,
    #[serde(default)]
    referenced_document_ids: Vec<String>,
    created_at: i64,
}

struct StoredChatTurn {
    id: String,
    provider_id: Option<String>,
    provider_label: String,
    user_message: String,
    assistant_answer: String,
    reasoning_content: Option<String>,
    selected_text: Option<String>,
    image_data_url: Option<String>,
    citations: Vec<runtime::rag::Citation>,
    claims: Vec<AskDocumentClaim>,
    retrieval_trace: serde_json::Value,
    referenced_document_ids: Vec<String>,
    created_at: i64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AskDocumentOutput {
    answer: String,
    reasoning_content: Option<String>,
    provider: String,
    claims: Vec<AskDocumentClaim>,
    citations: Vec<runtime::rag::Citation>,
    retrieval_trace: runtime::agent::AgentTrace,
    can_continue_retrieval: bool,
    retrieval_attempt_count: u32,
    retrieval_budget_exhausted: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentActivityEventOutput {
    event_id: String,
    event: runtime::agent::AgentTraceEvent,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AnswerDeltaEventOutput {
    event_id: String,
    delta: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReasoningDeltaEventOutput {
    event_id: String,
    delta: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AskDocumentDoneEventOutput {
    event_id: String,
    result: AskDocumentOutput,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AskDocumentErrorEventOutput {
    event_id: String,
    message: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AskDocumentStoppedEventOutput {
    event_id: String,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AskDocumentClaim {
    text: String,
    citation_ids: Vec<String>,
    citation_labels: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveModelProviderInput {
    id: Option<String>,
    name: String,
    provider_type: String,
    base_url: String,
    models: Vec<ModelProviderModelInput>,
    default_model_key: Option<String>,
    api_key: Option<String>,
    enabled: bool,
    is_default: bool,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelProviderModelInput {
    key: Option<String>,
    model_id: String,
    nickname: Option<String>,
    capabilities: Option<Vec<String>>,
    enabled: Option<bool>,
    is_default_chat_model: Option<bool>,
    #[serde(default)]
    context_window_override: Option<u32>,
    #[serde(default)]
    detected_context_window: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeleteModelProviderInput {
    id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestModelProviderInput {
    id: Option<String>,
    name: String,
    provider_type: String,
    base_url: String,
    model_id: String,
    api_key: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FetchProviderModelsInput {
    id: Option<String>,
    provider_type: String,
    base_url: String,
    api_key: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TestTranslationProviderInput {
    provider: String,
    microsoft_endpoint: Option<String>,
    microsoft_region: Option<String>,
    microsoft_api_key: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelProviderOutput {
    id: String,
    name: String,
    provider_type: String,
    base_url: String,
    models: Vec<ModelProviderModelOutput>,
    default_model_key: String,
    enabled: bool,
    is_default: bool,
    has_api_key: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelProviderModelOutput {
    key: String,
    model_id: String,
    nickname: String,
    capabilities: Vec<String>,
    enabled: bool,
    is_default_chat_model: bool,
    context_window_override: Option<u32>,
    detected_context_window: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelProviderTestOutput {
    ok: bool,
    message: String,
}

#[derive(Deserialize)]
struct OpenAiModelsResponse {
    data: Vec<OpenAiModelItem>,
}

#[derive(Deserialize)]
struct OpenAiModelItem {
    id: String,
    // Different OpenAI-compatible servers expose the context window under
    // different keys (OpenRouter: context_length, vLLM: max_model_len, LM Studio:
    // max_context_length) and varying types (int / float / string). Capture as raw
    // JSON and coerce, so a present-but-oddly-typed value never fails the WHOLE
    // /models decode (which would break model import for that provider).
    #[serde(default)]
    context_length: Option<serde_json::Value>,
    #[serde(default)]
    max_model_len: Option<serde_json::Value>,
    #[serde(default)]
    max_context_length: Option<serde_json::Value>,
}

impl OpenAiModelItem {
    fn detected_context_window(&self) -> Option<u32> {
        [
            &self.context_length,
            &self.max_model_len,
            &self.max_context_length,
        ]
        .into_iter()
        .flatten()
        .find_map(coerce_context_window)
    }
}

/// Coerce a JSON context-window value (int / float / numeric string) to a token
/// count, ignoring anything that isn't a sane positive window.
fn coerce_context_window(value: &serde_json::Value) -> Option<u32> {
    let tokens = match value {
        serde_json::Value::Number(number) => number.as_f64(),
        serde_json::Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }?;
    if tokens.is_finite() && tokens >= 1024.0 && tokens <= u32::MAX as f64 {
        Some(tokens as u32)
    } else {
        None
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FetchProviderModelsOutput {
    model_ids: Vec<String>,
    /// model_id → detected context window (tokens), for models whose `/models`
    /// entry exposed it. The frontend stores this as each model's auto-detected
    /// window (overridable by the user).
    context_windows: std::collections::HashMap<String, u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveTranslationSettingsInput {
    provider: String,
    enable_fallback: Option<bool>,
    microsoft_endpoint: Option<String>,
    microsoft_region: Option<String>,
    microsoft_api_key: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TranslationSettingsOutput {
    provider: String,
    enable_fallback: bool,
    microsoft_endpoint: String,
    microsoft_region: String,
    microsoft_has_api_key: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveWebSearchSettingsInput {
    /// New Exa API key. None / omitted → keep the existing key. Empty string →
    /// clear it (revert to the keyless DuckDuckGo fallback).
    exa_api_key: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WebSearchSettingsOutput {
    /// Whether an Exa key is stored — the key itself is never sent to the frontend.
    exa_has_api_key: bool,
}

struct TranslationCacheRecord {
    translated_text: String,
    provider_label: String,
}

struct TranslationCacheWriteInput<'a> {
    record: TranslationCacheRecord,
    document_id: &'a str,
    page: u32,
    block_id: &'a str,
    source_hash: &'a str,
    target_lang: &'a str,
    source_text: &'a str,
}

struct StreamedChatText {
    answer: String,
    reasoning_content: String,
}

struct AskAnswerResult {
    answer: String,
    reasoning_content: Option<String>,
    claims: Vec<AskDocumentClaim>,
}

struct StoredModelProvider {
    id: String,
    name: String,
    provider_type: String,
    base_url: String,
    models: Vec<StoredProviderModel>,
    default_model_key: String,
    enabled: bool,
    is_default: bool,
    has_api_key: bool,
    api_key_local: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredProviderModel {
    key: String,
    model_id: String,
    nickname: String,
    capabilities: Vec<String>,
    enabled: bool,
    is_default_chat_model: bool,
    /// User-set context window (tokens). When present it overrides everything —
    /// the authoritative "this model really has N tokens" escape hatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context_window_override: Option<u32>,
    /// Context window auto-detected from the provider's `/models` endpoint.
    /// Used when there is no user override (beats the bundled catalog/default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    detected_context_window: Option<u32>,
}

pub(crate) struct OpenAiCompatibleProvider {
    base_url: String,
    api_key: Option<String>,
    model: String,
    capabilities: Vec<String>,
    model_profile: model_catalog::ResolvedModelProfile,
    context_budget: model_catalog::ModelContextBudget,
}

pub(crate) struct MicrosoftTranslatorProvider {
    endpoint: String,
    region: String,
    api_key: String,
}

pub(crate) struct TranslationAttempt {
    pub(crate) provider: TranslationProvider,
}

pub(crate) enum TranslationProvider {
    LocalPlaceholder,
    GoogleWeb,
    Microsoft(MicrosoftTranslatorProvider),
    OpenAiCompatible(Box<OpenAiCompatibleProvider>),
    Unavailable {
        cache_key: String,
        label: String,
        message: String,
    },
}

#[derive(Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiChatMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize, Deserialize)]
struct OpenAiChatMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChatChoice>,
}

#[derive(Deserialize)]
struct OpenAiChatChoice {
    message: OpenAiChatMessage,
}

#[derive(Deserialize)]
struct OpenAiChatChunk {
    choices: Vec<OpenAiChatChunkChoice>,
}

#[derive(Deserialize)]
struct OpenAiChatChunkChoice {
    delta: Option<OpenAiChatChunkDelta>,
}

#[derive(Deserialize)]
struct OpenAiChatChunkDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    // Streamed native tool calls (the agent loop accumulates these across chunks
    // by `index`). Ignored by the plain answer-stream path, which never offers tools.
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiStreamToolCall>>,
}

/// One fragment of a streamed tool call. The `id`/`function.name` arrive in the
/// first chunk for a given `index`; the `function.arguments` are split across
/// subsequent chunks and must be concatenated in arrival order.
#[derive(Deserialize)]
struct OpenAiStreamToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiStreamToolCallFunction>,
}

#[derive(Deserialize)]
struct OpenAiStreamToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct MicrosoftTranslateResponseItem {
    translations: Vec<MicrosoftTranslationItem>,
}

#[derive(Deserialize)]
struct MicrosoftTranslationItem {
    text: String,
}

#[tauri::command]
fn choose_workspace() -> Result<Option<String>, String> {
    Ok(rfd::FileDialog::new()
        .set_title("Choose Lumenfolio workspace")
        .pick_folder()
        .map(|path| path.to_string_lossy().to_string()))
}

/// Native multi-select file picker for adding PDFs to a folder. Returns the
/// chosen file paths (empty if the dialog was cancelled), which the frontend
/// feeds into the same `import_workspace_paths` flow as a drag-and-drop.
#[tauri::command]
fn choose_pdf_files() -> Result<Vec<String>, String> {
    Ok(rfd::FileDialog::new()
        .set_title("Add documents")
        // Knowledge-base pivot (P2/P3): accept editable text + Office sources
        // alongside PDFs; ingestion dispatches by extension in import_workspace_paths.
        .add_filter(
            "Documents",
            &[
                "pdf", "docx", "xlsx", "pptx", "md", "markdown", "txt", "text",
            ],
        )
        .add_filter("PDF", &["pdf"])
        .add_filter("Office", &["docx", "xlsx", "pptx"])
        .add_filter("Markdown / Text", &["md", "markdown", "txt", "text"])
        .pick_files()
        .map(|paths| {
            paths
                .into_iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default())
}

/// Save text (e.g. a chat exported as Markdown) to a user-chosen file via the
/// native "Save as" dialog. Returns the saved path, or None if cancelled.
#[tauri::command]
fn export_markdown_file(default_name: String, content: String) -> Result<Option<String>, String> {
    let Some(path) = rfd::FileDialog::new()
        .set_title("Export chat as Markdown")
        .set_file_name(&default_name)
        .add_filter("Markdown", &["md"])
        .save_file()
    else {
        return Ok(None);
    };
    std::fs::write(&path, content).map_err(|err| format!("Failed to write file: {err}"))?;
    Ok(Some(path.to_string_lossy().to_string()))
}

#[tauri::command]
fn open_path_in_file_manager(path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("No path is available to open".to_string());
    }

    let path = PathBuf::from(trimmed);
    let target = path
        .canonicalize()
        .map_err(|err| format!("Failed to resolve path: {err}"))?;

    open_file_manager_target(&target)
}

/// Reveal a source's file in Finder / Explorer.
///
/// Importing references a file where it already lives rather than copying it, so
/// the library shows a title with no hint of where the bytes are. This answers
/// "where did that file go?" directly, and surfaces the one real failure mode of
/// referencing in place: a source whose file was later moved or deleted reports
/// that plainly instead of silently doing nothing.
#[tauri::command]
fn reveal_document_in_file_manager(
    document_id: String,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let document_id = document_id.trim().to_string();
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let path: Option<String> = conn
        .query_row(
            "SELECT path FROM documents WHERE id = ?1",
            params![document_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| format!("Failed to look up document: {err}"))?;
    let Some(path) = path else {
        return Err("Document no longer exists".to_string());
    };
    // Authored sources are rows, not files: their `path` is a synthetic
    // "note:<id>" marker. Point at the Markdown mirror when there is one.
    let is_authored = ["note:", "markdown:", "text:", "web:"]
        .iter()
        .any(|scheme| path.starts_with(scheme));
    if is_authored {
        let title: String = conn
            .query_row(
                "SELECT title FROM documents WHERE id = ?1",
                params![document_id],
                |row| row.get(0),
            )
            .map_err(|err| format!("Failed to look up note: {err}"))?;
        let Some(dir) = vault::vault_dir(&conn) else {
            return Err("This note has no file on disk. Set a notes folder in Settings to keep notes as .md files.".to_string());
        };
        let file = dir.join(format!(
            "{}.md",
            vault::file_stem_for_title(&title, &document_id)
        ));
        if !file.exists() {
            return Err("This note has not been written to the notes folder yet.".to_string());
        }
        drop(conn);
        return open_file_manager_target(&file);
    }
    drop(conn);
    let target = PathBuf::from(&path);
    if !target.exists() {
        return Err(format!(
            "The file is no longer at {path} — it was moved or deleted outside Lumenfolio."
        ));
    }
    let target = target
        .canonicalize()
        .map_err(|err| format!("Failed to resolve {path}: {err}"))?;
    open_file_manager_target(&target)
}

/// Open an http(s) URL in the user's default browser (e.g. a Trending paper's
/// Hugging Face page). Restricted to http/https so it can't launch arbitrary
/// commands or local files.
#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    let url = url.trim();
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        return Err("Only http(s) URLs can be opened".to_string());
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(url)
            .spawn()
            .map_err(|err| format!("Failed to open URL: {err}"))?;
        return Ok(());
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .map_err(|err| format!("Failed to open URL: {err}"))?;
        return Ok(());
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(url)
            .spawn()
            .map_err(|err| format!("Failed to open URL: {err}"))?;
        return Ok(());
    }
    #[allow(unreachable_code)]
    Err("Opening a URL is not supported on this platform".to_string())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalAgentTestResult {
    ok: bool,
    message: String,
    tool_calls: u32,
}

/// Settings "Test connection": verify the local agent can actually reach
/// Lumenfolio's MCP tools end-to-end. Runs a minimal agentic round-trip against
/// the first indexed document and reports how many tools the agent managed to
/// call (0 ⇒ the agent ran but the MCP plumbing didn't engage). Slow by nature
/// (a real CLI turn), so the frontend shows a spinner.
#[tauri::command]
async fn test_local_agent_connection(
    provider_id: String,
    database: State<'_, AppDatabase>,
) -> Result<LocalAgentTestResult, String> {
    let kind = local_agent::provider_id_kind(&provider_id)
        .ok_or_else(|| "Not a local agent provider".to_string())?;

    let (db_path, document_id) = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        let path = conn.path().map(|p| p.to_string());
        let doc: Option<String> = conn
            .query_row(
                "SELECT document_id FROM document_chunks LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .ok()
            .flatten();
        (path, doc)
    };

    let Some(db_path) = db_path else {
        return Ok(LocalAgentTestResult {
            ok: false,
            tool_calls: 0,
            message: "No database is available to test against.".to_string(),
        });
    };
    let Some(document_id) = document_id else {
        return Ok(LocalAgentTestResult {
            ok: false,
            tool_calls: 0,
            message: "Index a document first, then test the connection.".to_string(),
        });
    };

    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let counter_cb = counter.clone();
    let on_tool = move |ev: local_agent::AgentToolEvent| {
        if ev.phase == local_agent::AgentToolPhase::Started {
            counter_cb.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    };
    let prompt = local_agent::build_agentic_prompt(
        "Call the document search tool once for any keyword you like, then reply with exactly: OK.",
        "",
        None,
        Some("en"),
    );

    match local_agent::generate_answer_agentic(
        kind,
        std::path::PathBuf::from(db_path),
        document_id,
        prompt,
        None,
        false,
        tokio_util::sync::CancellationToken::new(),
        on_tool,
        // Connection health check — the answer text is discarded, so don't stream it.
        |_delta: String| {},
    )
    .await
    {
        Ok(_) => {
            let n = counter.load(std::sync::atomic::Ordering::Relaxed);
            Ok(LocalAgentTestResult {
                ok: n > 0,
                tool_calls: n,
                message: if n > 0 {
                    format!("Connected — the agent called {n} tool(s) over MCP.")
                } else {
                    "The agent ran but didn't call any Lumenfolio tools.".to_string()
                },
            })
        }
        Err(err) => Ok(LocalAgentTestResult {
            ok: false,
            tool_calls: counter.load(std::sync::atomic::Ordering::Relaxed),
            message: err,
        }),
    }
}

fn open_file_manager_target(target: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let mut command = Command::new("open");
        if target.is_file() {
            command.arg("-R");
        }
        command.arg(target);
        command
            .spawn()
            .map_err(|err| format!("Failed to open Finder: {err}"))?;
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        let mut command = Command::new("explorer");
        if target.is_file() {
            command.arg(format!("/select,{}", target.display()));
        } else {
            command.arg(target);
        }
        command
            .spawn()
            .map_err(|err| format!("Failed to open File Explorer: {err}"))?;
        return Ok(());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let open_target = if target.is_file() {
            target.parent().unwrap_or(target)
        } else {
            target
        };
        Command::new("xdg-open")
            .arg(open_target)
            .spawn()
            .map_err(|err| format!("Failed to open file manager: {err}"))?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Err("Opening a file manager is not supported on this platform".to_string())
}

#[tauri::command]
fn scan_workspace_pdfs(
    root: String,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<WorkspaceRootSnapshot, String> {
    let root_path = PathBuf::from(root)
        .canonicalize()
        .map_err(|err| format!("Failed to open workspace: {err}"))?;

    let workspace_root_id = documents::stable_path_id("root", &root_path);
    let mut docs = Vec::new();
    documents::collect_pdfs(&root_path, &workspace_root_id, &mut docs)?;
    docs.sort_by(|left, right| {
        left.short_title
            .to_lowercase()
            .cmp(&right.short_title.to_lowercase())
    });

    documents::persist_workspace_scan(&database, &workspace_root_id, &root_path, &docs)?;
    let documents = documents::load_documents_for_root(&database, &workspace_root_id)?;
    documents::upsert_registry_paths(&registry, &documents)?;

    Ok(WorkspaceRootSnapshot {
        root: WorkspaceRoot {
            id: workspace_root_id,
            path: root_path.to_string_lossy().to_string(),
        },
        documents,
    })
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ImportWorkspacePathsArgs {
    /// Logical collection to file the imports into (None = unfiled/inbox).
    #[serde(default)]
    target_collection_id: Option<String>,
    #[serde(default)]
    source_paths: Vec<String>,
}

/// Knowledge-base pivot (Collections): import sources into a logical collection.
/// Files (and, recursively, a folder's contents) are registered by their real
/// path — never copied, never bound to a live disk scan — and filed into the
/// target collection. Returns the refreshed Knowledge Base doc pool.
///
/// Runs off the UI thread: `canonicalize` on an iCloud / File Provider path can
/// block for a long time, and a sync command would freeze the window.
#[tauri::command]
async fn import_workspace_paths(
    args: ImportWorkspacePathsArgs,
    app: tauri::AppHandle,
) -> Result<Vec<WorkspaceRootSnapshot>, String> {
    run_blocking_io(move || {
        let registry = app.state::<PdfRegistry>();
        let database = app.state::<AppDatabase>();
        import_workspace_paths_inner(args, &registry, &database, &app)
    })
    .await
}

fn import_workspace_paths_inner(
    args: ImportWorkspacePathsArgs,
    registry: &State<'_, PdfRegistry>,
    database: &State<'_, AppDatabase>,
    app: &tauri::AppHandle,
) -> Result<Vec<WorkspaceRootSnapshot>, String> {
    let ImportWorkspacePathsArgs {
        target_collection_id,
        source_paths,
    } = args;
    let collection_id = target_collection_id
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    // Gather every importable file: individual files directly, directories
    // recursively (one-shot — no persistent scan/reconcile binding to disk).
    let mut file_paths: Vec<PathBuf> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for raw in source_paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let canonical = match PathBuf::from(trimmed).canonicalize() {
            Ok(path) => path,
            Err(err) => {
                errors.push(format!("Cannot access {trimmed}: {err}"));
                continue;
            }
        };
        let metadata = match fs::metadata(&canonical) {
            Ok(metadata) => metadata,
            Err(err) => {
                errors.push(format!("Cannot read {}: {err}", canonical.display()));
                continue;
            }
        };
        if metadata.is_dir() {
            documents::collect_import_files(&canonical, 0, &mut file_paths)?;
        } else if metadata.is_file() {
            file_paths.push(canonical);
        }
    }
    // A file can arrive via both a directory walk and directly — dedup.
    file_paths.sort();
    file_paths.dedup();

    let mut file_backed: Vec<PathBuf> = Vec::new();
    let mut text_files: Vec<(PathBuf, &'static str)> = Vec::new();
    for path in file_paths {
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "pdf" | "docx" | "xlsx" | "pptx" => file_backed.push(path),
            "md" | "markdown" => text_files.push((path, "markdown")),
            "txt" | "text" => text_files.push((path, "text")),
            _ => {}
        }
    }

    let mut imported_ids: Vec<String> = Vec::new();

    // File-backed sources register under the Knowledge storage root — their real
    // path is preserved (for reading/preview), disk location no longer drives
    // organization. Existing rows dedup by stable id.
    if !file_backed.is_empty() {
        let root_path = PathBuf::from(documents::KNOWLEDGE_ROOT_PATH);
        let mut docs = Vec::new();
        for path in &file_backed {
            match documents::build_document_for_path(path, documents::KNOWLEDGE_ROOT_ID) {
                Ok(Some(doc)) => {
                    imported_ids.push(doc.id.clone());
                    docs.push(doc);
                }
                Ok(None) => {}
                Err(err) => errors.push(format!("Cannot import {}: {err}", path.display())),
            }
        }
        if !docs.is_empty() {
            documents::additive_upsert_documents(
                &database,
                documents::KNOWLEDGE_ROOT_ID,
                &root_path,
                &docs,
            )?;
            let snapshot_docs =
                documents::load_documents_for_root(&database, documents::KNOWLEDGE_ROOT_ID)?;
            documents::upsert_registry_paths(&registry, &snapshot_docs)?;
            for doc in &docs {
                let _ = document_index::enqueue_document_reindex(doc.id.clone(), app.clone());
            }
        }
    }

    // Text sources: content copied into body_md (editable, decoupled from file).
    for (path, content_type) in &text_files {
        let body_md = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(err) => {
                errors.push(format!("Cannot read {}: {err}", path.display()));
                continue;
            }
        };
        let title = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("Untitled")
            .to_string();
        match documents::create_text_document(
            &database,
            content_type,
            &title,
            &body_md,
            None,
            collection_id.as_deref(),
        ) {
            Ok(document_id) => {
                let _ = document_index::enqueue_document_reindex(document_id.clone(), app.clone());
                imported_ids.push(document_id);
            }
            Err(err) => errors.push(format!("Cannot import {}: {err}", path.display())),
        }
    }

    // File the newly-imported sources into the target collection.
    documents::assign_collection(&database, &imported_ids, collection_id.as_deref())?;

    if imported_ids.is_empty() && !errors.is_empty() {
        return Err(errors.join("; "));
    }

    Ok(vec![knowledge_root_snapshot(&database)?])
}

// ---------------------------------------------------------------------------
// Knowledge-base pivot (P2): authored sources (notes / web clips / md imports).
//
// These are disk-decoupled — created, edited and re-indexed straight from the
// `documents.body_md` column inside the virtual "Knowledge Base" root. The
// frontend merges the returned snapshot into its workspace tree and selects the
// new id; `update_note` re-chunks the body via the shared reindex path.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CreateSourceOutput {
    document_id: String,
    snapshot: WorkspaceRootSnapshot,
}

fn knowledge_root_snapshot(
    database: &State<'_, AppDatabase>,
) -> Result<WorkspaceRootSnapshot, String> {
    let documents = documents::load_documents_for_root(database, documents::KNOWLEDGE_ROOT_ID)?;
    Ok(WorkspaceRootSnapshot {
        root: WorkspaceRoot {
            id: documents::KNOWLEDGE_ROOT_ID.to_string(),
            path: "lumenfolio://knowledge".to_string(),
        },
        documents,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateNoteSourceInput {
    #[serde(default)]
    title: String,
    #[serde(default)]
    body_md: String,
    #[serde(default)]
    collection_id: Option<String>,
}

#[tauri::command]
fn create_note_source(
    input: CreateNoteSourceInput,
    database: State<'_, AppDatabase>,
    app: tauri::AppHandle,
) -> Result<CreateSourceOutput, String> {
    let collection_id = input
        .collection_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let document_id = documents::create_text_document(
        &database,
        "note",
        &input.title,
        &input.body_md,
        None,
        collection_id,
    )?;
    // Index the new note (chunks → artifacts → graph) in the background so it
    // is immediately askable / linkable. Failure to enqueue is non-fatal —
    // selecting the note re-triggers indexing.
    let _ = document_index::enqueue_document_reindex(document_id.clone(), app);
    let snapshot = knowledge_root_snapshot(&database)?;
    Ok(CreateSourceOutput {
        document_id,
        snapshot,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateNoteSourceInput {
    document_id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    body_md: String,
}

#[tauri::command]
fn update_note_source(
    input: UpdateNoteSourceInput,
    database: State<'_, AppDatabase>,
    app: tauri::AppHandle,
) -> Result<DocumentIndexEventOutput, String> {
    let document_id = input.document_id.trim().to_string();
    documents::update_text_document_body(&database, &document_id, &input.title, &input.body_md)?;
    document_index::enqueue_document_reindex(document_id, app)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VaultSettings {
    /// Where notes are mirrored; empty when the user has turned mirroring off.
    dir: String,
    /// True when the path came from the setting rather than the built-in default.
    customized: bool,
}

/// Where the live database lives, so backup commands can stage a restore next
/// to it without re-deriving the path.
pub(crate) struct DatabasePath(pub PathBuf);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BackupSettings {
    dir: String,
    keep: u32,
    /// Hours between automatic snapshots; 0 = manual only.
    interval_hours: u32,
    last_at: Option<i64>,
    entries: Vec<backup::BackupEntry>,
}

fn backup_settings_snapshot(conn: &Connection) -> Result<BackupSettings, String> {
    let dir = resolved_backup_dir(conn)?;
    Ok(BackupSettings {
        dir: dir
            .as_ref()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default(),
        keep: resolved_keep(conn),
        interval_hours: load_app_setting(conn, backup::BACKUP_INTERVAL_SETTING)?
            .and_then(|value| value.trim().parse::<u32>().ok())
            .unwrap_or(0),
        last_at: load_app_setting(conn, backup::BACKUP_LAST_AT_SETTING)?
            .and_then(|value| value.trim().parse::<i64>().ok()),
        entries: dir
            .map(|path| backup::list_snapshots(&path))
            .unwrap_or_default(),
    })
}

/// Take a snapshot into the configured folder, prune, and stamp the time.
/// Shared by the manual command and the scheduled check so both behave alike —
/// notably, a manual backup postpones the next automatic one.
fn run_backup(conn: &Connection) -> Result<PathBuf, String> {
    let Some(dir) = resolved_backup_dir(conn)? else {
        return Err("Choose a backup folder first.".to_string());
    };
    let dest = dir.join(backup::snapshot_name(chrono::Local::now()));
    backup::write_snapshot(conn, &dest)?;
    backup::prune_snapshots(&dir, resolved_keep(conn) as usize);
    save_app_setting(
        conn,
        backup::BACKUP_LAST_AT_SETTING,
        &chrono::Local::now().timestamp().to_string(),
    )?;
    Ok(dest)
}

/// Run a scheduled snapshot if one is due. Called once at startup, off the
/// critical path: a backup that fails (folder on an unmounted volume, no space)
/// is logged and forgotten, never surfaced as a startup error.
fn maybe_auto_backup(database: &AppDatabase) {
    let Ok(conn) = database.conn.lock() else {
        return;
    };
    let due = (|| -> Result<bool, String> {
        if resolved_backup_dir(&conn)?.is_none() {
            return Ok(false);
        }
        let interval = load_app_setting(&conn, backup::BACKUP_INTERVAL_SETTING)?
            .and_then(|value| value.trim().parse::<u32>().ok())
            .unwrap_or(0);
        let last_at = load_app_setting(&conn, backup::BACKUP_LAST_AT_SETTING)?
            .and_then(|value| value.trim().parse::<i64>().ok());
        Ok(backup::auto_backup_due(
            last_at,
            interval,
            chrono::Local::now().timestamp(),
        ))
    })()
    .unwrap_or(false);
    if !due {
        return;
    }
    match run_backup(&conn) {
        Ok(path) => log::info!("Scheduled backup written to {}", path.display()),
        Err(err) => log::warn!("Scheduled backup skipped: {err}"),
    }
}

fn resolved_backup_dir(conn: &Connection) -> Result<Option<PathBuf>, String> {
    let configured = load_app_setting(conn, backup::BACKUP_DIR_SETTING)?
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    Ok(configured.map(PathBuf::from))
}

fn resolved_keep(conn: &Connection) -> u32 {
    load_app_setting(conn, backup::BACKUP_KEEP_SETTING)
        .ok()
        .flatten()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(backup::DEFAULT_KEEP as u32)
}

#[tauri::command]
fn load_backup_settings(database: State<'_, AppDatabase>) -> Result<BackupSettings, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    backup_settings_snapshot(&conn)
}

#[tauri::command]
fn save_backup_settings(
    dir: String,
    keep: u32,
    interval_hours: u32,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    save_app_setting(&conn, backup::BACKUP_DIR_SETTING, dir.trim())?;
    save_app_setting(&conn, backup::BACKUP_KEEP_SETTING, &keep.max(1).to_string())?;
    save_app_setting(
        &conn,
        backup::BACKUP_INTERVAL_SETTING,
        &interval_hours.to_string(),
    )
}

#[tauri::command]
fn choose_backup_dir() -> Result<Option<String>, String> {
    Ok(rfd::FileDialog::new()
        .set_title("Choose a folder for database backups")
        .pick_folder()
        .map(|path| path.to_string_lossy().to_string()))
}

/// Take a snapshot now and prune older ones past the keep count.
#[tauri::command]
fn create_backup_now(database: State<'_, AppDatabase>) -> Result<BackupSettings, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    run_backup(&conn)?;
    backup_settings_snapshot(&conn)
}

/// Stage a snapshot to be swapped in at the next launch. Returns nothing —
/// the frontend tells the user to restart, which is when it actually applies.
#[tauri::command]
fn restore_backup(
    path: String,
    database: State<'_, AppDatabase>,
    db_path: State<'_, DatabasePath>,
) -> Result<(), String> {
    let snapshot = PathBuf::from(path.trim());
    if !snapshot.exists() {
        return Err("That backup file no longer exists.".to_string());
    }
    // Take the lock only to prove the app is not mid-write; staging itself never
    // touches the live database.
    let _guard = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    backup::stage_restore(&snapshot, &db_path.0)
}

/// Native folder picker for the notes vault. Separate from `choose_workspace`
/// only so the dialog title names what is being chosen.
#[tauri::command]
fn choose_vault_dir() -> Result<Option<String>, String> {
    Ok(rfd::FileDialog::new()
        .set_title("Choose the folder to keep notes in")
        .pick_folder()
        .map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
fn load_vault_settings(database: State<'_, AppDatabase>) -> Result<VaultSettings, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    Ok(VaultSettings {
        dir: vault::vault_dir(&conn)
            .map(|dir| dir.to_string_lossy().to_string())
            .unwrap_or_default(),
        customized: load_app_setting(&conn, vault::VAULT_DIR_SETTING)?.is_some(),
    })
}

/// Point the Markdown mirror at a folder — typically one the user already syncs
/// (iCloud / Dropbox / Syncthing / a WebDAV mount). An empty string turns
/// mirroring off. Existing notes are re-exported so the folder is complete
/// immediately rather than only holding notes saved from now on.
#[tauri::command]
fn set_vault_dir(dir: String, database: State<'_, AppDatabase>) -> Result<u32, String> {
    let dir = dir.trim().to_string();
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    save_app_setting(&conn, vault::VAULT_DIR_SETTING, &dir)?;
    if dir.is_empty() {
        return Ok(0);
    }
    export_all_notes(&conn)
}

/// Write every authored source to the vault. Used when the folder changes and as
/// a manual "make sure everything is on disk" action.
#[tauri::command]
fn export_notes_to_vault(database: State<'_, AppDatabase>) -> Result<u32, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    export_all_notes(&conn)
}

fn export_all_notes(conn: &Connection) -> Result<u32, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, title, COALESCE(body_md, '') FROM documents
             WHERE content_type IN ('note', 'markdown', 'text', 'web')",
        )
        .map_err(|err| format!("Failed to list notes: {err}"))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|err| format!("Failed to list notes: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to list notes: {err}"))?;
    let mut written = 0u32;
    for (id, title, body) in rows {
        match vault::write_note(conn, &id, &title, &body) {
            Ok(()) => written += 1,
            Err(err) => log::warn!("Failed to export note {id}: {err}"),
        }
    }
    Ok(written)
}

/// Recover notes present in the vault but missing from the database — the path
/// back after a lost or reset database. Additive only: a file whose id still has
/// a row is skipped, so this can never overwrite current work with a stale copy.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImportVaultOutput {
    imported: u32,
    /// Same shape the note-create path returns, so the sidebar refreshes without
    /// a second round trip.
    snapshot: WorkspaceRootSnapshot,
}

#[tauri::command]
fn import_notes_from_vault(
    database: State<'_, AppDatabase>,
    app: tauri::AppHandle,
) -> Result<ImportVaultOutput, String> {
    let orphans = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        vault::import_orphans(&conn)?
    };
    let mut imported = 0u32;
    for orphan in orphans {
        // Goes through the normal create path, so the note is mirrored back with
        // a fresh id and indexed like any other.
        match documents::create_text_document(
            &database,
            "note",
            &orphan.title,
            &orphan.body_md,
            None,
            None,
        ) {
            Ok(document_id) => {
                imported += 1;
                let _ = document_index::enqueue_document_reindex(document_id, app.clone());
            }
            Err(err) => log::warn!("Failed to import vault note '{}': {err}", orphan.title),
        }
    }
    Ok(ImportVaultOutput {
        imported,
        snapshot: knowledge_root_snapshot(&database)?,
    })
}

#[tauri::command]
fn load_note_source(
    document_id: String,
    database: State<'_, AppDatabase>,
) -> Result<documents::TextDocumentBody, String> {
    documents::load_text_document_body(&database, document_id.trim())
}

/// Derive a short, human title for a web clip: the first non-trivial line of the
/// extracted text, else the URL host. Capped so it fits the sidebar.
fn web_clip_title(extracted: &str, url: &str) -> String {
    let from_text = extracted
        .lines()
        .map(str::trim)
        .find(|line| line.len() >= 8 && line.len() <= 120);
    let title = match from_text {
        Some(line) => line.to_string(),
        None => url
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .split('/')
            .next()
            .unwrap_or(url)
            .to_string(),
    };
    let title = title.trim();
    if title.chars().count() > 80 {
        format!("{}…", title.chars().take(80).collect::<String>())
    } else {
        title.to_string()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClipWebPageInput {
    url: String,
    #[serde(default)]
    collection_id: Option<String>,
}

#[tauri::command]
fn clip_web_page(
    input: ClipWebPageInput,
    database: State<'_, AppDatabase>,
    app: tauri::AppHandle,
) -> Result<CreateSourceOutput, String> {
    let url = input.url.trim();
    if url.is_empty() {
        return Err("No URL provided".to_string());
    }
    // Accept bare URLs (e.g. "arxiv.org/abs/1234") by defaulting to https — web_fetch
    // hard-requires an http(s) scheme and would otherwise reject them with a cryptic
    // low-level error.
    let normalized_url = if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{url}")
    };
    let url = normalized_url.as_str();
    // Reuse the chat web_fetch extractor (HTML stripped → readable text). A clip
    // is just a captured snapshot stored as an editable 'web' source so it flows
    // through the same chunk → graph → claims pipeline as notes.
    let extracted = runtime::web_search::web_fetch(url, 40_000)?;
    if extracted.trim().is_empty() {
        return Err("No readable text could be extracted from that page".to_string());
    }
    let title = web_clip_title(&extracted, url);
    // Front-matter keeps the origin visible in the editor body too (and survives
    // export), beyond the structured source_url column.
    let body_md = format!("> [{title}]({url})\n\n{extracted}");
    let collection_id = input
        .collection_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let document_id = documents::create_text_document(
        &database,
        "web",
        &title,
        &body_md,
        Some(url),
        collection_id,
    )?;
    let _ = document_index::enqueue_document_reindex(document_id.clone(), app);
    let snapshot = knowledge_root_snapshot(&database)?;
    Ok(CreateSourceOutput {
        document_id,
        snapshot,
    })
}

// Knowledge-base pivot (P2.5): wikilink graph for the editor — outbound
// [[links]] (resolved by current title, so newly-created targets light up) and
// backlinks (notes whose [[Title]] points at this document).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NoteLinkOut {
    title: String,
    document_id: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NoteBacklink {
    document_id: String,
    title: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NoteLinksOutput {
    outbound: Vec<NoteLinkOut>,
    backlinks: Vec<NoteBacklink>,
}

#[tauri::command]
fn load_note_links(
    document_id: String,
    database: State<'_, AppDatabase>,
) -> Result<NoteLinksOutput, String> {
    let document_id = document_id.trim();
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;

    let mut outbound_stmt = conn
        .prepare(
            "SELECT l.target_title,
                    (SELECT t.id FROM documents t
                       WHERE lower(t.title) = lower(l.target_title)
                         AND t.id != l.source_document_id
                       LIMIT 1)
             FROM note_links l
             WHERE l.source_document_id = ?1
             ORDER BY l.id",
        )
        .map_err(|err| format!("Failed to prepare outbound links: {err}"))?;
    let outbound = outbound_stmt
        .query_map(params![document_id], |row| {
            Ok(NoteLinkOut {
                title: row.get(0)?,
                document_id: row.get(1)?,
            })
        })
        .map_err(|err| format!("Failed to load outbound links: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to load outbound links: {err}"))?;
    drop(outbound_stmt);

    let mut backlink_stmt = conn
        .prepare(
            "SELECT DISTINCT s.id, s.title
             FROM note_links l
             JOIN documents s ON s.id = l.source_document_id
             JOIN documents d ON d.id = ?1
             WHERE lower(l.target_title) = lower(d.title)
               AND s.id != d.id
             ORDER BY lower(s.title)",
        )
        .map_err(|err| format!("Failed to prepare backlinks: {err}"))?;
    let backlinks = backlink_stmt
        .query_map(params![document_id], |row| {
            Ok(NoteBacklink {
                document_id: row.get(0)?,
                title: row.get(1)?,
            })
        })
        .map_err(|err| format!("Failed to load backlinks: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to load backlinks: {err}"))?;

    Ok(NoteLinksOutput {
        outbound,
        backlinks,
    })
}

#[tauri::command]
fn remove_workspace_root(
    root_id: String,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let trimmed_root_id = root_id.trim();
    if trimmed_root_id.is_empty() {
        return Err("No workspace selected".to_string());
    }

    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start remove-workspace transaction: {err}"))?;

    let root_documents = documents::load_documents_for_root_conn(&tx, trimmed_root_id)?;
    let deleted = tx
        .execute(
            "DELETE FROM workspace_roots WHERE id = ?1",
            params![trimmed_root_id],
        )
        .map_err(|err| format!("Failed to remove workspace root: {err}"))?;

    if deleted == 0 {
        return Err("Selected workspace no longer exists".to_string());
    }

    tx.commit()
        .map_err(|err| format!("Failed to commit workspace removal: {err}"))?;
    documents::remove_registry_paths(&registry, &root_documents)?;
    Ok(())
}

/// Permanently delete one document from the knowledge base: its index, chunks,
/// notes, claims, graph edges, chat turns and translations. For a note (synthetic
/// `note:` path) this is a complete removal; for a file-backed source (PDF/Office)
/// it removes the library entry only — the original file on disk is left untouched
/// (and a later rescan of that folder would re-add it).
#[tauri::command]
fn delete_document(
    document_id: String,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
) -> Result<(), String> {
    let document_id = document_id.trim().to_string();
    if document_id.is_empty() {
        return Err("No document selected".to_string());
    }
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM documents WHERE id = ?1",
            params![document_id],
            |_| Ok(()),
        )
        .optional()
        .map_err(|err| format!("Failed to look up document: {err}"))?
        .is_some();
    if !exists {
        return Err("Document no longer exists".to_string());
    }
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start delete-document transaction: {err}"))?;
    // FTS5 virtual tables carry no foreign key, so the cascade below cannot reach
    // them — delete their rows explicitly first.
    tx.execute(
        "DELETE FROM document_chunks_fts WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear chunk FTS rows: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_facts_fts WHERE document_id = ?1",
        params![document_id],
    )
    .map_err(|err| format!("Failed to clear table-fact FTS rows: {err}"))?;
    // Every other document-scoped table has ON DELETE CASCADE to documents(id), so
    // removing the row cleans pages/blocks/lines/chunks/tables/artifacts/claims/
    // links/notes/translations in one shot (connection runs foreign_keys=ON).
    // chat_turns is the deliberate exception: its FK is ON DELETE SET NULL, so the
    // conversation survives — affected turns just become library-wide (document_id
    // → NULL). Deleting a document must never destroy chat history.
    let deleted = tx
        .execute("DELETE FROM documents WHERE id = ?1", params![document_id])
        .map_err(|err| format!("Failed to delete document: {err}"))?;
    if deleted == 0 {
        return Err("Document no longer exists".to_string());
    }
    tx.commit()
        .map_err(|err| format!("Failed to commit document deletion: {err}"))?;
    // Remove the note's Markdown mirror too, or the next vault import would
    // resurrect a source the user just deleted.
    vault::delete_note(&conn, &document_id);
    drop(conn);
    // Drop the registry path mapping (a no-op for synthetic note: paths).
    if let Ok(mut paths) = registry.paths.lock() {
        paths.remove(&document_id);
    }
    // Invalidate any cached working memory for this document's default session.
    agent_sessions.clear_session(&migrated_session_id(&document_id));
    Ok(())
}

/// Clear the Unfiled bucket: permanently remove every document that has no logical
/// collection (`collection_id IS NULL`) from the library, in one transaction.
///
/// Robustness contract (this command exists precisely because the old frontend
/// loop was unsafe):
/// - "Unfiled" is decided by the DATABASE, never the frontend's in-memory view. A
///   client whose collections list hasn't loaded would classify every document as
///   unfiled; keying off `collection_id IS NULL` here makes that impossible.
/// - A document filed into ANY collection is never touched, even a dangling one.
/// - Chat history survives (chat_turns.document_id is ON DELETE SET NULL).
/// - Original files on disk are left untouched; a rescan re-adds file-backed docs.
///
/// Returns the ids that were removed so the caller can update its view.
#[tauri::command]
fn clear_unfiled_documents(
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
) -> Result<Vec<String>, String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    // Snapshot the unfiled ids first — needed to prune the registry / working
    // memory after the rows are gone.
    let ids: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT id FROM documents WHERE collection_id IS NULL")
            .map_err(|err| format!("Failed to list unfiled documents: {err}"))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|err| format!("Failed to list unfiled documents: {err}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format!("Failed to list unfiled documents: {err}"))?;
        rows
    };
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start clear-unfiled transaction: {err}"))?;
    // FTS5 virtual tables carry no foreign key, so the document cascade cannot
    // reach them — clear their rows explicitly first.
    tx.execute(
        "DELETE FROM document_chunks_fts
         WHERE document_id IN (SELECT id FROM documents WHERE collection_id IS NULL)",
        [],
    )
    .map_err(|err| format!("Failed to clear chunk FTS rows: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_facts_fts
         WHERE document_id IN (SELECT id FROM documents WHERE collection_id IS NULL)",
        [],
    )
    .map_err(|err| format!("Failed to clear table-fact FTS rows: {err}"))?;
    tx.execute("DELETE FROM documents WHERE collection_id IS NULL", [])
        .map_err(|err| format!("Failed to clear unfiled documents: {err}"))?;
    tx.commit()
        .map_err(|err| format!("Failed to commit clear-unfiled: {err}"))?;
    drop(conn);
    if let Ok(mut paths) = registry.paths.lock() {
        for id in &ids {
            paths.remove(id);
        }
    }
    for id in &ids {
        agent_sessions.clear_session(&migrated_session_id(id));
    }
    Ok(ids)
}

#[tauri::command]
fn load_last_workspace(
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<Option<WorkspaceSnapshot>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;

    let mut stmt = conn
        .prepare("SELECT id, path FROM workspace_roots ORDER BY created_at ASC, rowid ASC")
        .map_err(|err| format!("Failed to load workspaces: {err}"))?;
    let roots = stmt
        .query_map([], |row| {
            Ok(WorkspaceRoot {
                id: row.get(0)?,
                path: row.get(1)?,
            })
        })
        .map_err(|err| format!("Failed to load workspaces: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to load workspaces: {err}"))?;
    drop(stmt);

    if roots.is_empty() {
        return Ok(None);
    }

    let mut snapshots = Vec::new();
    let mut registry_documents = Vec::new();
    for root in roots {
        let documents = documents::load_documents_for_root_conn(&conn, &root.id)?;
        registry_documents.extend(documents.iter().cloned());
        snapshots.push(WorkspaceRootSnapshot { root, documents });
    }
    drop(conn);

    documents::replace_registry_paths(&registry, &registry_documents)?;

    Ok(Some(WorkspaceSnapshot { roots: snapshots }))
}

#[tauri::command]
fn mark_document_stale(
    document_id: String,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
) -> Result<(), String> {
    let mut conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("Failed to start reindex transaction: {err}"))?;
    tx.execute(
        "UPDATE documents
         SET index_status = 'stale',
             updated_at = unixepoch()
         WHERE id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to mark document stale: {err}"))?;
    tx.execute(
        "DELETE FROM structure_tree_nodes WHERE document_id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear structure tree: {err}"))?;
    tx.execute(
        "DELETE FROM document_outlines WHERE document_id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear document outlines: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_facts_fts WHERE document_id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear table fact FTS rows: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_facts WHERE document_id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear table facts: {err}"))?;
    tx.execute(
        "DELETE FROM document_table_cells
         WHERE table_id IN (SELECT id FROM document_tables WHERE document_id = ?1)",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear table cells: {err}"))?;
    tx.execute(
        "DELETE FROM document_tables WHERE document_id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear document tables: {err}"))?;
    tx.execute(
        "DELETE FROM document_visual_assets WHERE document_id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear visual assets: {err}"))?;
    tx.execute(
        "DELETE FROM document_index_jobs WHERE document_id = ?1",
        params![document_id.as_str()],
    )
    .map_err(|err| format!("Failed to clear document index jobs: {err}"))?;
    tx.commit()
        .map_err(|err| format!("Failed to commit reindex request: {err}"))?;
    // Invalidate cached working memory for this document's default (per-document)
    // session. Memory is keyed by session id now; multi-document sessions refresh
    // on the next ask via fresh retrieval.
    agent_sessions.clear_session(&migrated_session_id(&document_id));
    Ok(())
}

#[tauri::command]
fn search_document_chunks(
    document_id: String,
    query: String,
    limit: Option<u32>,
    database: State<'_, AppDatabase>,
) -> Result<Vec<SearchResult>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let limit = limit.unwrap_or(8).clamp(1, 20);
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    runtime::rag::search_chunks(&conn, &document_id, query, limit).map(|results| {
        results
            .into_iter()
            .map(|candidate| SearchResult {
                chunk_id: candidate.chunk_id,
                document_id: candidate.document_id,
                page: candidate.page,
                block_id: candidate.block_id,
                quote: candidate.quote,
                bbox_list: candidate.bbox_list,
                rank: candidate.score,
            })
            .collect()
    })
}

#[tauri::command]
fn update_document_reading_state(
    input: ReadingStateInput,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "UPDATE documents
         SET last_page = ?2, zoom = ?3, last_opened_at = unixepoch(), updated_at = unixepoch()
         WHERE id = ?1",
        params![input.document_id, input.page, input.zoom],
    )
    .map_err(|err| format!("Failed to update reading state: {err}"))?;
    Ok(())
}

#[tauri::command]
async fn translate_text(
    input: TranslateTextInput,
    database: State<'_, AppDatabase>,
) -> Result<TranslateTextOutput, String> {
    let document_id = input.document_id;
    let source_text = input.text.trim().to_string();
    if source_text.is_empty() {
        return Err("No text to translate".to_string());
    }

    let target_lang = input.target_lang.trim().to_string();
    if target_lang.is_empty() {
        return Err("No target language selected".to_string());
    }

    let enable_fallback = translation_fallback_enabled(&database);
    let selected_provider_name =
        resolve_selected_translation_provider_name(input.provider.as_deref(), &database);
    let attempts = build_translation_attempts(&selected_provider_name, &database, enable_fallback);
    let source_hash = stable_text_hash(&format!(
        "{}\n{}",
        input.source_version.unwrap_or_default().trim(),
        source_text
    ));
    let block_id = input.block_id.unwrap_or_default();
    let force_refresh = input.force_refresh.unwrap_or(false);
    let mut errors = Vec::new();

    if !force_refresh {
        for (index, attempt) in attempts.iter().enumerate() {
            if let Some(translated_text) = read_translation_cache(
                &database,
                &document_id,
                &source_hash,
                &target_lang,
                &attempt.provider,
            )? {
                return Ok(TranslateTextOutput {
                    translated_text,
                    provider: format_translation_attempt_chain(&attempts, index),
                    cached: true,
                    source_hash,
                });
            }
        }
    }

    for (index, attempt) in attempts.iter().enumerate() {
        match translation::translate_with_provider(&source_text, &target_lang, &attempt.provider)
            .await
        {
            Ok(translated_text) => {
                write_translation_cache(
                    &database,
                    TranslationCacheWriteInput {
                        record: TranslationCacheRecord {
                            translated_text: translated_text.clone(),
                            provider_label: attempt.provider.cache_key(),
                        },
                        document_id: &document_id,
                        page: input.page,
                        block_id: &block_id,
                        source_hash: &source_hash,
                        target_lang: &target_lang,
                        source_text: &source_text,
                    },
                )?;

                return Ok(TranslateTextOutput {
                    translated_text,
                    provider: format_translation_attempt_chain(&attempts, index),
                    cached: false,
                    source_hash,
                });
            }
            Err(err) => {
                errors.push(format!("{}: {err}", attempt.provider.label()));
            }
        }
    }

    if errors.is_empty() {
        return Err("No translation providers are available".to_string());
    }
    let provider_chain = format_translation_attempt_chain(&attempts, attempts.len() - 1);
    Err(format!(
        "Translation failed after trying {}: {}",
        provider_chain,
        errors.join(" | ")
    ))
}

#[tauri::command]
fn list_model_providers(
    database: State<'_, AppDatabase>,
) -> Result<Vec<ModelProviderOutput>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    providers::load_model_provider_outputs(&conn)
}

/// Deterministic id of a document's fallback ("migrated") session — the one the
/// migration back-fills and that legacy single-document callers resolve to.
///
/// The same convention is encoded as the `'migrated-' || document_id` SQL
/// expression in `migrate_chat_turns_to_sessions` (storage/db.rs); keep them in
/// sync. Centralizing the Rust side here keeps the string literal out of the
/// reindex/clear paths that need it.
pub(crate) fn migrated_session_id(document_id: &str) -> String {
    format!("migrated-{document_id}")
}

/// Resolve the session a request targets. An explicit, non-empty `session_id`
/// always wins; otherwise we fall back to the deterministic per-document session
/// so legacy single-document callers keep working and stay consistent with the
/// migration back-fill.
fn resolve_session_id(session_id: Option<&str>, document_id: &str) -> Result<String, String> {
    if let Some(session_id) = session_id.map(str::trim).filter(|value| !value.is_empty()) {
        return Ok(session_id.to_string());
    }
    let document_id = document_id.trim();
    if document_id.is_empty() {
        return Err("No session or document selected".to_string());
    }
    Ok(migrated_session_id(document_id))
}

#[tauri::command]
fn load_chat_turns(
    input: LoadChatTurnsInput,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
) -> Result<Vec<ChatHistoryMessageOutput>, String> {
    let session_id = resolve_session_id(input.session_id.as_deref(), &input.document_id)?;
    let limit = input.limit.unwrap_or(40).clamp(1, 120);
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let turns = load_stored_chat_turns(&conn, &session_id, limit)?;
    restore_agent_session_from_turns(&agent_sessions, &session_id, &turns);
    Ok(chat_turns_to_messages(turns))
}

#[tauri::command]
fn clear_chat_turns(
    input: ClearChatTurnsInput,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
) -> Result<(), String> {
    let session_id = resolve_session_id(input.session_id.as_deref(), &input.document_id)?;
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    if let Some(turn_ids) = input.turn_ids {
        for turn_id in turn_ids
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            conn.execute(
                "DELETE FROM chat_turns WHERE session_id = ?1 AND id = ?2",
                params![session_id, turn_id],
            )
            .map_err(|err| format!("Failed to clear chat history: {err}"))?;
        }
    } else {
        conn.execute(
            "DELETE FROM chat_turns WHERE session_id = ?1",
            params![session_id],
        )
        .map_err(|err| format!("Failed to clear chat history: {err}"))?;
    }
    agent_sessions.clear_session(&session_id);
    Ok(())
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct NoteOutput {
    id: String,
    document_id: String,
    page: u32,
    bbox_list: Vec<Vec<f64>>,
    quote_text: String,
    content: String,
    created_at: i64,
    updated_at: i64,
}

fn read_note_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<NoteOutput> {
    let bbox_list_json: String = row.get(3)?;
    let bbox_list = serde_json::from_str::<Vec<Vec<f64>>>(&bbox_list_json).unwrap_or_default();
    Ok(NoteOutput {
        id: row.get(0)?,
        document_id: row.get(1)?,
        page: row.get(2)?,
        bbox_list,
        quote_text: row.get(4)?,
        content: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

fn load_note_by_id(conn: &Connection, id: &str) -> Result<Option<NoteOutput>, String> {
    use rusqlite::OptionalExtension;
    conn.query_row(
        "SELECT id, document_id, page, bbox_list_json, quote_text, content, created_at, updated_at
         FROM notes WHERE id = ?1",
        params![id],
        read_note_row,
    )
    .optional()
    .map_err(|err| format!("Failed to load note: {err}"))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoadNotesInput {
    document_id: String,
}

#[tauri::command]
fn load_notes(
    input: LoadNotesInput,
    database: State<'_, AppDatabase>,
) -> Result<Vec<NoteOutput>, String> {
    let document_id = input.document_id.trim();
    if document_id.is_empty() {
        return Err("No document selected".to_string());
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT id, document_id, page, bbox_list_json, quote_text, content, created_at, updated_at
             FROM notes
             WHERE document_id = ?1
             ORDER BY created_at DESC, rowid DESC",
        )
        .map_err(|err| format!("Failed to prepare notes query: {err}"))?;
    let notes = stmt
        .query_map(params![document_id], read_note_row)
        .map_err(|err| format!("Failed to load notes: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read notes: {err}"))?;
    Ok(notes)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateNoteInput {
    document_id: String,
    page: u32,
    #[serde(default)]
    bbox_list: Vec<Vec<f64>>,
    #[serde(default)]
    quote_text: String,
    #[serde(default)]
    content: String,
}

#[tauri::command]
fn create_note(
    input: CreateNoteInput,
    database: State<'_, AppDatabase>,
) -> Result<NoteOutput, String> {
    let document_id = input.document_id.trim();
    if document_id.is_empty() {
        return Err("No document selected".to_string());
    }
    let note_id = stable_text_hash(&format!(
        "note:{}:{}:{}",
        document_id,
        input.page,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default()
    ));
    let bbox_list_json = serde_json::to_string(&input.bbox_list)
        .map_err(|err| format!("Failed to encode note bbox list: {err}"))?;
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "INSERT INTO notes
            (id, document_id, page, bbox_list_json, quote_text, content,
             created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, unixepoch(), unixepoch())",
        params![
            note_id,
            document_id,
            input.page,
            bbox_list_json,
            input.quote_text,
            input.content,
        ],
    )
    .map_err(|err| format!("Failed to create note: {err}"))?;
    load_note_by_id(&conn, &note_id)?.ok_or_else(|| "Note not found after creation".to_string())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateNoteInput {
    id: String,
    content: String,
}

#[tauri::command]
fn update_note(
    input: UpdateNoteInput,
    database: State<'_, AppDatabase>,
) -> Result<NoteOutput, String> {
    let note_id = input.id.trim();
    if note_id.is_empty() {
        return Err("No note specified".to_string());
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let affected = conn
        .execute(
            "UPDATE notes SET content = ?2, updated_at = unixepoch() WHERE id = ?1",
            params![note_id, input.content],
        )
        .map_err(|err| format!("Failed to update note: {err}"))?;
    if affected == 0 {
        return Err("Note not found".to_string());
    }
    load_note_by_id(&conn, note_id)?.ok_or_else(|| "Note not found".to_string())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeleteNoteInput {
    id: String,
}

#[tauri::command]
fn delete_note(input: DeleteNoteInput, database: State<'_, AppDatabase>) -> Result<(), String> {
    let note_id = input.id.trim();
    if note_id.is_empty() {
        return Err("No note specified".to_string());
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let affected = conn
        .execute("DELETE FROM notes WHERE id = ?1", params![note_id])
        .map_err(|err| format!("Failed to delete note: {err}"))?;
    if affected == 0 {
        return Err("Note not found".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Agent workspace sessions
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatSessionOutput {
    id: String,
    title: String,
    focus_document_id: Option<String>,
    focus_document_title: Option<String>,
    referenced_document_ids: Vec<String>,
    turn_count: i64,
    created_at: i64,
    updated_at: i64,
}

fn read_chat_session_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChatSessionOutput> {
    let referenced_json: String = row.get(3)?;
    let referenced_document_ids =
        serde_json::from_str::<Vec<String>>(&referenced_json).unwrap_or_default();
    Ok(ChatSessionOutput {
        id: row.get(0)?,
        title: row.get(1)?,
        focus_document_id: optional_non_empty(row.get::<_, Option<String>>(2)?.unwrap_or_default()),
        referenced_document_ids,
        focus_document_title: optional_non_empty(
            row.get::<_, Option<String>>(4)?.unwrap_or_default(),
        ),
        turn_count: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

const CHAT_SESSION_SELECT: &str = "SELECT s.id, s.title, s.focus_document_id,
        s.referenced_document_ids_json,
        COALESCE(NULLIF(d.short_title, ''), d.title),
        (SELECT COUNT(*) FROM chat_turns t WHERE t.session_id = s.id),
        s.created_at, s.updated_at
     FROM chat_sessions s
     LEFT JOIN documents d ON d.id = s.focus_document_id";

fn load_chat_session_by_id(
    conn: &Connection,
    session_id: &str,
) -> Result<Option<ChatSessionOutput>, String> {
    let sql = format!("{CHAT_SESSION_SELECT} WHERE s.id = ?1");
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|err| format!("Failed to prepare chat session query: {err}"))?;
    let mut rows = stmt
        .query_map(params![session_id], read_chat_session_row)
        .map_err(|err| format!("Failed to load chat session: {err}"))?;
    match rows.next() {
        Some(row) => {
            Ok(Some(row.map_err(|err| {
                format!("Failed to read chat session: {err}")
            })?))
        }
        None => Ok(None),
    }
}

#[tauri::command]
fn list_chat_sessions(database: State<'_, AppDatabase>) -> Result<Vec<ChatSessionOutput>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let sql = format!("{CHAT_SESSION_SELECT} ORDER BY s.updated_at DESC, s.rowid DESC");
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|err| format!("Failed to prepare chat sessions query: {err}"))?;
    let sessions = stmt
        .query_map([], read_chat_session_row)
        .map_err(|err| format!("Failed to load chat sessions: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read chat sessions: {err}"))?;
    Ok(sessions)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateChatSessionInput {
    #[serde(default)]
    focus_document_id: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

#[tauri::command]
fn create_chat_session(
    input: CreateChatSessionInput,
    database: State<'_, AppDatabase>,
) -> Result<ChatSessionOutput, String> {
    let focus = input
        .focus_document_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let title = input
        .title
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
        .to_string();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let session_id = format!(
        "session-{}",
        stable_text_hash(&format!("{}:{nanos}", focus.unwrap_or("")))
    );
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "INSERT INTO chat_sessions
            (id, title, focus_document_id, referenced_document_ids_json, created_at, updated_at)
         VALUES (?1, ?2, ?3, '[]', unixepoch(), unixepoch())",
        params![session_id, title, focus],
    )
    .map_err(|err| format!("Failed to create chat session: {err}"))?;
    load_chat_session_by_id(&conn, &session_id)?
        .ok_or_else(|| "Chat session not found after creation".to_string())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RenameChatSessionInput {
    id: String,
    title: String,
}

#[tauri::command]
fn rename_chat_session(
    input: RenameChatSessionInput,
    database: State<'_, AppDatabase>,
) -> Result<ChatSessionOutput, String> {
    let session_id = input.id.trim();
    if session_id.is_empty() {
        return Err("No session specified".to_string());
    }
    let title = input.title.trim();
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let affected = conn
        .execute(
            "UPDATE chat_sessions SET title = ?2, updated_at = unixepoch() WHERE id = ?1",
            params![session_id, title],
        )
        .map_err(|err| format!("Failed to rename chat session: {err}"))?;
    if affected == 0 {
        return Err("Chat session not found".to_string());
    }
    load_chat_session_by_id(&conn, session_id)?.ok_or_else(|| "Chat session not found".to_string())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateChatSessionFocusInput {
    id: String,
    #[serde(default)]
    focus_document_id: Option<String>,
}

#[tauri::command]
fn update_chat_session_focus(
    input: UpdateChatSessionFocusInput,
    database: State<'_, AppDatabase>,
) -> Result<ChatSessionOutput, String> {
    let session_id = input.id.trim();
    if session_id.is_empty() {
        return Err("No session specified".to_string());
    }
    let focus = input
        .focus_document_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let affected = conn
        .execute(
            "UPDATE chat_sessions SET focus_document_id = ?2, updated_at = unixepoch() WHERE id = ?1",
            params![session_id, focus],
        )
        .map_err(|err| format!("Failed to update chat session focus: {err}"))?;
    if affected == 0 {
        return Err("Chat session not found".to_string());
    }
    load_chat_session_by_id(&conn, session_id)?.ok_or_else(|| "Chat session not found".to_string())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeleteChatSessionInput {
    id: String,
}

#[tauri::command]
fn delete_chat_session(
    input: DeleteChatSessionInput,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
) -> Result<(), String> {
    let session_id = input.id.trim();
    if session_id.is_empty() {
        return Err("No session specified".to_string());
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    // chat_turns has no foreign key to chat_sessions (its document_id FK is now
    // ON DELETE SET NULL), so deleting a session never cascades to its turns —
    // remove this session's turns explicitly before dropping the session row.
    conn.execute(
        "DELETE FROM chat_turns WHERE session_id = ?1",
        params![session_id],
    )
    .map_err(|err| format!("Failed to delete chat session turns: {err}"))?;
    conn.execute(
        "DELETE FROM chat_sessions WHERE id = ?1",
        params![session_id],
    )
    .map_err(|err| format!("Failed to delete chat session: {err}"))?;
    drop(conn);
    agent_sessions.clear_session(session_id);
    Ok(())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateSessionTitleInput {
    session_id: String,
    #[serde(default)]
    model_provider_id: Option<String>,
    #[serde(default)]
    model_key: Option<String>,
}

/// Generate a short LLM title for a session from its opening exchange and store
/// it. Best-effort: returns the new title on success; surfaces an error the
/// frontend can ignore (keeping the temporary title) on any failure.
#[tauri::command]
async fn generate_session_title(
    input: GenerateSessionTitleInput,
    database: State<'_, AppDatabase>,
) -> Result<String, String> {
    let session_id = input.session_id.trim().to_string();
    if session_id.is_empty() {
        return Err("No session specified".to_string());
    }
    // Pull the first user/assistant exchange for this session.
    let (question, answer) = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        conn.query_row(
            "SELECT user_message, assistant_answer
             FROM chat_turns
             WHERE session_id = ?1
             ORDER BY created_at ASC, rowid ASC
             LIMIT 1",
            params![session_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(|err| format!("No conversation to title yet: {err}"))?
    };
    let question = question.trim();
    if question.is_empty() {
        return Err("No question to summarize".to_string());
    }
    let provider_id = input.model_provider_id.as_deref().unwrap_or("").trim();
    // Local-agent providers (Codex/Claude CLI) have no OpenAI-compatible endpoint
    // for a cheap title call — derive a heuristic title from the question instead
    // of failing (which would leave the session untitled).
    let raw_title = if local_agent::provider_id_kind(provider_id).is_some() {
        question.to_string()
    } else {
        let (provider, _) =
            providers::resolve_chat_provider(&database, provider_id, input.model_key.as_deref())?;
        llm::chat::generate_session_title_with_openai_compatible(question, &answer, &provider)
            .await?
    };
    let title = clamp_session_title(&raw_title);
    if title.is_empty() {
        return Err("Generated title was empty".to_string());
    }
    {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        conn.execute(
            "UPDATE chat_sessions SET title = ?2, updated_at = unixepoch() WHERE id = ?1",
            params![session_id, title],
        )
        .map_err(|err| format!("Failed to save generated title: {err}"))?;
    }
    Ok(title)
}

/// Hard cap the generated title at 12 characters (by Unicode scalar), trimming
/// trailing whitespace left by truncation. Keeps tab labels uniform.
fn clamp_session_title(raw: &str) -> String {
    let trimmed = raw.trim();
    let capped: String = trimmed.chars().take(12).collect();
    capped.trim_end().to_string()
}

#[tauri::command]
async fn ask_document(
    input: AskDocumentInput,
    database: State<'_, AppDatabase>,
    agent_sessions: State<'_, AgentSessionState>,
    app: tauri::AppHandle,
) -> Result<AskDocumentOutput, String> {
    run_ask_document(input, &database, &agent_sessions, &app).await
}

#[tauri::command]
fn ask_document_stream(input: AskDocumentInput, app: tauri::AppHandle) -> Result<(), String> {
    let event_id = input
        .activity_event_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| "Missing activity event id".to_string())?;
    log::info!("chat stream command accepted event_id={event_id}");
    // Register a cancellation token so `stop_ask_document` can interrupt this run.
    if let Some(state) = app.try_state::<AskCancellations>() {
        if let Ok(mut map) = state.0.lock() {
            map.insert(event_id.clone(), tokio_util::sync::CancellationToken::new());
        }
    }
    let task_app = app.clone();
    let cleanup_event_id = event_id.clone();
    tauri::async_runtime::spawn(async move {
        let result = {
            let database = task_app.state::<AppDatabase>();
            let agent_sessions = task_app.state::<AgentSessionState>();
            run_ask_document(input, &database, &agent_sessions, &task_app).await
        };
        // Did the user stop this run? Check before removing the token.
        let was_cancelled = cancellation_token(&task_app, &cleanup_event_id)
            .map(|token| token.is_cancelled())
            .unwrap_or(false);
        if let Some(state) = task_app.try_state::<AskCancellations>() {
            if let Ok(mut map) = state.0.lock() {
                map.remove(&cleanup_event_id);
            }
        }
        match result {
            // The HTTP path returns the partial answer it streamed before the stop,
            // so a stopped HTTP turn still completes normally (with the partial).
            Ok(output) => {
                log::info!(
                    "chat stream command completed event_id={} answer_chars={}",
                    event_id,
                    output.answer.chars().count()
                );
                if let Err(err) = task_app.emit(
                    "lumenfolio://ask-document-done",
                    AskDocumentDoneEventOutput {
                        event_id,
                        result: output,
                    },
                ) {
                    log::warn!("Failed to emit ask-document-done: {err}");
                }
            }
            // A non-streaming path (agent loop / local CLI) that was stopped aborts
            // with an error — surface it as a quiet "stopped", not a failure.
            Err(_) if was_cancelled => {
                log::info!("chat stream stopped by user event_id={event_id}");
                let _ = task_app.emit(
                    "lumenfolio://ask-document-stopped",
                    AskDocumentStoppedEventOutput { event_id },
                );
            }
            Err(err) => {
                log::warn!("chat stream command failed event_id={event_id}: {err}");
                if let Err(emit_err) = task_app.emit(
                    "lumenfolio://ask-document-error",
                    AskDocumentErrorEventOutput {
                        event_id,
                        message: err,
                    },
                ) {
                    log::warn!("Failed to emit ask-document-error: {emit_err}");
                }
            }
        }
    });
    Ok(())
}

/// Stop an in-flight chat generation (the user pressed "stop"). Cooperative: the
/// HTTP/agent-loop streaming loops break on the token and return what they have so
/// far; a local-agent subprocess is killed. The frontend finalizes the message
/// optimistically, so this just needs to fire-and-forget.
#[tauri::command]
fn stop_ask_document(event_id: String, app: tauri::AppHandle) -> Result<(), String> {
    match cancellation_token(&app, &event_id) {
        Some(token) => {
            token.cancel();
            log::info!("stop requested for in-flight generation event_id={event_id}");
        }
        // No token means the run already finished — the frontend flushes the
        // typewriter on click, so the stop still settles the bubble visually.
        None => log::debug!("stop requested but generation already done event_id={event_id}"),
    }
    Ok(())
}

/// Whether the question explicitly names one of the OTHER (non-focus) workspace
/// documents — by its filename stem or its (distinctive) title. When it does, the
/// focus document's "current view" is noise: the user is asking about a different
/// paper, so seeding evidence from the page they happen to be looking at floods
/// the evidence with the wrong document. We use this to suppress current-view
/// seeding and let retrieval route to the named document instead.
/// Whether `needle` appears in `haystack` delimited by non-alphanumeric
/// boundaries (a whole-token match), so a short filename stem like "data" does
/// NOT match inside "database". Both are already lowercased.
fn mentions_token(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let mut from = 0;
    while let Some(offset) = haystack[from..].find(needle) {
        let start = from + offset;
        let end = start + needle.len();
        let before_ok = start == 0
            || !haystack[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric);
        let after_ok = end >= haystack.len()
            || !haystack[end..]
                .chars()
                .next()
                .is_some_and(char::is_alphanumeric);
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// Whether the question explicitly names a NON-focus indexed document (by filename
/// stem or distinctive title). One scan over all indexed docs (not the capped
/// manifest), so it still fires for documents the large-library manifest omitted.
fn question_targets_non_focus_document(
    conn: &Connection,
    question: &str,
    focus_document_id: &str,
) -> bool {
    let normalized = question.to_lowercase();
    let Ok(mut stmt) = conn.prepare(
        "SELECT id, COALESCE(NULLIF(short_title, ''), title), path
         FROM documents WHERE index_status = 'indexed'",
    ) else {
        return false;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    }) else {
        return false;
    };
    for (id, title, path) in rows.flatten() {
        if id == focus_document_id {
            continue;
        }
        // The filename the user most likely typed (e.g. "forge_ieee_preprint").
        let stem = std::path::Path::new(&path)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_lowercase();
        if stem.chars().count() >= 4 && mentions_token(&normalized, &stem) {
            return true;
        }
        // Or a distinctive document title spelled out in the question.
        let title = title.trim().to_lowercase();
        if title.chars().count() >= 8 && mentions_token(&normalized, &title) {
            return true;
        }
    }
    false
}

async fn current_view_decision_for_input(
    question: &str,
    input: &AskDocumentInput,
    metadata: Option<&serde_json::Value>,
    provider: Option<&OpenAiCompatibleProvider>,
) -> Option<llm::chat::LlmCurrentViewDecision> {
    let selected_text = input
        .selected_text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if selected_text.is_some() {
        return Some(llm::chat::LlmCurrentViewDecision {
            relevance: "high".to_string(),
            mode: "selection".to_string(),
            reason: "User selected text, so selected evidence has priority.".to_string(),
            should_use_current_view: true,
        });
    }
    let (Some(metadata), Some(provider)) = (metadata, provider) else {
        return None;
    };
    if input
        .image_data_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
    {
        return None;
    }
    let judge_result = tokio::time::timeout(
        Duration::from_secs(CURRENT_VIEW_JUDGE_TIMEOUT_SECS),
        llm::chat::judge_current_view_relevance_with_openai_compatible(
            question, metadata, provider,
        ),
    )
    .await
    .map_err(|_| {
        format!("Current-view relevance judge timed out after {CURRENT_VIEW_JUDGE_TIMEOUT_SECS}s")
    });
    match judge_result {
        Ok(Ok(decision)) => Some(decision),
        Ok(Err(err)) | Err(err) => {
            log::warn!("Current-view relevance judge skipped: {err}");
            None
        }
    }
}

fn current_view_retrieval_hint(
    input: &AskDocumentInput,
    decision: Option<&llm::chat::LlmCurrentViewDecision>,
) -> (Option<u32>, Option<&'static str>, Option<&'static str>) {
    let selected_text = input
        .selected_text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if selected_text.is_some() {
        return (
            input.page.or_else(|| viewport_active_page(input)),
            Some("overview"),
            None,
        );
    }
    if input.viewport_context.is_none() {
        return (input.page, None, None);
    }
    let Some(decision) = decision else {
        return (None, None, None);
    };
    if !decision.should_use_current_view || decision.mode == "none" {
        return (None, None, None);
    }
    let page = viewport_active_page(input).or(input.page);
    let mode = match decision.mode.as_str() {
        "full" | "visible" => Some("full"),
        "selection" | "semantic_chunks" => Some("overview"),
        _ => None,
    };
    (page, mode, Some("current_view"))
}

fn viewport_active_page(input: &AskDocumentInput) -> Option<u32> {
    input
        .viewport_context
        .as_ref()
        .and_then(|context| context.active_page)
        .filter(|page| *page > 0)
}

fn build_current_view_gate_metadata(
    database: &AppDatabase,
    document_id: &str,
    viewport: Option<&ViewportContextInput>,
    page_hint: Option<u32>,
    selected_text: Option<&str>,
) -> Result<Option<serde_json::Value>, String> {
    let Some(active_page) = viewport
        .and_then(|context| context.active_page)
        .or(page_hint)
        .filter(|page| *page > 0)
    else {
        return Ok(None);
    };
    let mut visible_pages = viewport
        .map(|context| {
            context
                .visible_pages
                .iter()
                .copied()
                .filter(|page| *page > 0)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if visible_pages.is_empty() {
        visible_pages.push(active_page);
    }
    visible_pages.sort_unstable();
    visible_pages.dedup();
    visible_pages.truncate(4);

    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let section_title = current_view_section_title(&conn, document_id, active_page);
    let tables = current_view_table_metadata(&conn, document_id, &visible_pages)?;
    let visuals = current_view_visual_metadata(&conn, document_id, &visible_pages)?;
    let selection_preview = selected_text
        .or_else(|| viewport.and_then(|context| context.selection_preview.as_deref()))
        .map(|value| truncate_for_error(value.trim(), 500))
        .filter(|value| !value.is_empty());
    Ok(Some(serde_json::json!({
        "activePage": active_page,
        "visiblePages": visible_pages,
        "sectionTitle": section_title,
        "selectionPreview": selection_preview,
        "capturedAt": viewport.and_then(|context| context.captured_at),
        "source": viewport.and_then(|context| context.source.as_deref()),
        "sensitivity": viewport
            .and_then(|context| context.sensitivity.as_deref())
            .unwrap_or("normal"),
        "visibleObjects": {
            "tables": tables,
            "visuals": visuals
        }
    })))
}

fn current_view_section_title(conn: &Connection, document_id: &str, page: u32) -> Option<String> {
    conn.query_row(
        "SELECT title
         FROM structure_tree_nodes
         WHERE document_id = ?1 AND page_start <= ?2 AND page_end >= ?2
         ORDER BY level DESC, order_index DESC
         LIMIT 1",
        params![document_id, page],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .ok()
    .flatten()
}

fn current_view_table_metadata(
    conn: &Connection,
    document_id: &str,
    visible_pages: &[u32],
) -> Result<Vec<serde_json::Value>, String> {
    let mut items = Vec::new();
    let mut stmt = conn
        .prepare(
            "SELECT id, page_no, caption, source, confidence
             FROM document_tables
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY page_no, caption
             LIMIT 20",
        )
        .map_err(|err| format!("Failed to prepare current-view table metadata: {err}"))?;
    for page in visible_pages {
        let rows = stmt
            .query_map(params![document_id, page], |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "page": row.get::<_, u32>(1)?,
                    "caption": row.get::<_, String>(2)?,
                    "source": row.get::<_, String>(3)?,
                    "confidence": row.get::<_, f64>(4)?,
                }))
            })
            .map_err(|err| format!("Failed to query current-view tables: {err}"))?;
        for row in rows {
            items.push(row.map_err(|err| format!("Failed to read current-view table: {err}"))?);
        }
    }
    Ok(items)
}

fn current_view_visual_metadata(
    conn: &Connection,
    document_id: &str,
    visible_pages: &[u32],
) -> Result<Vec<serde_json::Value>, String> {
    let mut items = Vec::new();
    let mut stmt = conn
        .prepare(
            "SELECT id, page_no, asset_type, caption, source, confidence
             FROM document_visual_assets
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY page_no, asset_type, caption
             LIMIT 20",
        )
        .map_err(|err| format!("Failed to prepare current-view visual metadata: {err}"))?;
    for page in visible_pages {
        let rows = stmt
            .query_map(params![document_id, page], |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "page": row.get::<_, u32>(1)?,
                    "type": row.get::<_, String>(2)?,
                    "caption": row.get::<_, String>(3)?,
                    "source": row.get::<_, String>(4)?,
                    "confidence": row.get::<_, f64>(5)?,
                }))
            })
            .map_err(|err| format!("Failed to query current-view visuals: {err}"))?;
        for row in rows {
            items.push(row.map_err(|err| format!("Failed to read current-view visual: {err}"))?);
        }
    }
    Ok(items)
}

/// Upper bound on how many "@-referenced" documents a single question may pull in,
/// to keep retrieval token/latency cost bounded.
const MAX_REFERENCE_DOCS: usize = 4;

/// Normalize the raw reference-document id list from the UI: trim, drop blanks,
/// drop the primary document itself, dedupe (order-preserving), cap at
/// MAX_REFERENCE_DOCS. The returned owned Vec is the single source of truth for
/// both the agent run request and chat-turn persistence.
fn normalize_reference_document_ids(raw: &[String], primary_document_id: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for id in raw {
        let id = id.trim();
        if id.is_empty() || id == primary_document_id {
            continue;
        }
        if seen.insert(id.to_string()) {
            out.push(id.to_string());
            if out.len() >= MAX_REFERENCE_DOCS {
                break;
            }
        }
    }
    out
}

// Local agents (Codex / Claude CLI) have very large context windows, so we feed
// them a far richer conversation memory than the token-budgeted API path: the most
// recent turns' FULL question + answer, rather than 3 turns of ~260-char previews.
const LOCAL_AGENT_MEMORY_TURNS: usize = 20;
const LOCAL_AGENT_MEMORY_ANSWER_CHARS: usize = 8_000;
const LOCAL_AGENT_MEMORY_TOTAL_CHARS: usize = 80_000;

fn cap_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max).collect();
    out.push('…');
    out
}

/// Build a generous conversation-memory block for the local-agent prompt from the
/// session's stored turns (full Q + A), newest-bounded by a total budget and shown
/// oldest-first. Returns an empty string when there is no prior history.
fn build_local_agent_memory(conn: &Connection, session_id: &str) -> String {
    let mut stmt = match conn.prepare(
        "SELECT user_message, assistant_answer
         FROM chat_turns
         WHERE session_id = ?1
         ORDER BY created_at DESC, rowid DESC
         LIMIT ?2",
    ) {
        Ok(stmt) => stmt,
        Err(_) => return String::new(),
    };
    let rows = stmt.query_map(
        params![session_id, LOCAL_AGENT_MEMORY_TURNS as i64],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    );
    let turns_desc: Vec<(String, String)> = match rows {
        Ok(iter) => iter.filter_map(Result::ok).collect(),
        Err(_) => return String::new(),
    };

    // Walk newest→oldest, accepting turns until the total budget is hit (so the
    // most recent context is always kept), then present them oldest→newest.
    let mut selected: Vec<(String, String)> = Vec::new();
    let mut total = 0usize;
    for (question, answer) in turns_desc {
        let question = question.trim().to_string();
        let answer = cap_chars(answer.trim(), LOCAL_AGENT_MEMORY_ANSWER_CHARS);
        if question.is_empty() && answer.is_empty() {
            continue;
        }
        let block_len = question.len() + answer.len() + 16;
        if !selected.is_empty() && total + block_len > LOCAL_AGENT_MEMORY_TOTAL_CHARS {
            break;
        }
        total += block_len;
        selected.push((question, answer));
    }
    if selected.is_empty() {
        return String::new();
    }
    selected.reverse();

    let mut out = String::from("Recent conversation (most recent last):\n");
    for (question, answer) in selected {
        out.push_str(&format!("- Q: {question}\n  A: {answer}\n"));
    }
    out
}

/// Human-readable label for a Lumenfolio MCP tool name, for the local-agent trace.
fn friendly_tool_label(tool: &str) -> &str {
    match tool {
        "search_chunks" => "Searching the document",
        "open_pages" => "Reading pages",
        "open_section" => "Reading a section",
        "open_table" | "search_table_facts" | "resolve_table_anchor" => "Reading a table",
        "inspect_tables" => "Inspecting tables",
        "inspect_tree" | "read_tree_node_lines" => "Inspecting structure",
        "inspect_visuals" | "open_visual" | "analyze_visual" | "inspect_objects"
        | "analyze_page" => "Inspecting figures",
        "query_knowledge_graph" => "Querying the knowledge graph",
        "search_library_knowledge" => "Searching your library",
        "list_trending_papers" => "Checking trending papers",
        "recall_chat_history" => "Recalling the conversation",
        _ => "Using a document tool",
    }
}

/// Collapse repeated citations an agentic run may serve across multiple tool
/// calls. Keys on the block (stable id) when present, else page + quote text.
fn dedup_citations(citations: Vec<runtime::rag::Citation>) -> Vec<runtime::rag::Citation> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(citations.len());
    for c in citations {
        let key = if !c.block_id.is_empty() {
            format!("{}::{}::{}", c.document_id, c.page, c.block_id)
        } else {
            format!("{}::{}::{}", c.document_id, c.page, c.quote)
        };
        if seen.insert(key) {
            out.push(c);
        }
    }
    out
}

async fn run_ask_document(
    input: AskDocumentInput,
    database: &AppDatabase,
    agent_sessions: &AgentSessionState,
    app: &tauri::AppHandle,
) -> Result<AskDocumentOutput, String> {
    let question = input.question.trim();
    if question.is_empty() {
        return Err("No question to ask".to_string());
    }
    let document_id = input.document_id.trim();
    // KB pivot (P1): a library-wide "ask my knowledge base" turn has no focus
    // document — allowed as long as it carries a session. Downstream is forgiving:
    // doc-scoped retrieval seeds empty (P0-d) and the agent leans on the library-
    // wide tools. Existing callers always send a document_id, so the focused path
    // is unchanged.
    let has_session = input
        .session_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    if document_id.is_empty() && !has_session {
        return Err("No document or session selected".to_string());
    }
    // The conversation this turn belongs to. Memory + persistence key off this;
    // retrieval targets `document_id` when present, else goes library-wide.
    let session_id = resolve_session_id(input.session_id.as_deref(), document_id)?;
    let reference_document_ids = normalize_reference_document_ids(
        input.reference_document_ids.as_deref().unwrap_or(&[]),
        document_id,
    );
    // Borrowed view of the @-referenced docs (priority hint + manifest tagging).
    let reference_document_id_refs: Vec<&str> =
        reference_document_ids.iter().map(String::as_str).collect();

    // Build the workspace manifest once: the agent's "file tree" for multi-document
    // routing. Its document ids form the tool-dispatch whitelist (focus + @ + the
    // locally-ranked rest, capped). Failure degrades to focus-only behavior.
    let workspace_manifest = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        runtime::rag::load_workspace_manifest(
            &conn,
            question,
            document_id,
            &reference_document_id_refs,
        )
        .unwrap_or_else(|err| {
            log::warn!("Failed to build workspace manifest: {err}");
            runtime::rag::WorkspaceManifest {
                entries: Vec::new(),
                document_ids: Vec::new(),
                total_indexed: 0,
                all_document_ids: Vec::new(),
            }
        })
    };
    let workspace_manifest_text = workspace_manifest.to_prompt_block();
    let visible_document_id_refs: Vec<&str> = workspace_manifest
        .document_ids
        .iter()
        .map(String::as_str)
        .collect();
    // Progressive disclosure: for a large library the unified loop does NOT inline
    // the whole manifest. It gets a compact block (focus + @-referenced) plus
    // `search_library`/`list_documents` tools, and may route to ANY indexed doc —
    // so its dispatch whitelist is the full indexed set, not the capped manifest.
    let library_is_large = workspace_manifest.is_large();
    let unified_loop_manifest_text = if library_is_large {
        workspace_manifest.to_prompt_block_compact()
    } else {
        workspace_manifest_text.clone()
    };
    let unified_visible_document_id_refs: Vec<&str> = if library_is_large {
        workspace_manifest
            .all_document_ids
            .iter()
            .map(String::as_str)
            .collect()
    } else {
        visible_document_id_refs.clone()
    };

    let selected_provider_id = input.model_provider_id.as_deref().unwrap_or("").trim();
    let provider_result = providers::resolve_chat_provider(
        database,
        selected_provider_id,
        input.model_key.as_deref(),
    );
    let context_budget = provider_result
        .as_ref()
        .ok()
        .map(|(provider, _)| provider.context_budget.clone())
        .unwrap_or_default();
    let model_budget_detail = provider_result
        .as_ref()
        .ok()
        .map(|(provider, _)| {
            format!(
                "model={} profile_source={} context_tokens={} evidence_tokens={} max_citations={}",
                provider.model_profile.model_id,
                provider.model_profile.source,
                provider.context_budget.model_context_tokens,
                provider.context_budget.evidence_tokens,
                provider.context_budget.max_accumulated_citations
            )
        })
        .unwrap_or_else(|| {
            format!(
                "model=unknown profile_source={} context_tokens={} evidence_tokens={} max_citations={}",
                context_budget.source,
                context_budget.model_context_tokens,
                context_budget.evidence_tokens,
                context_budget.max_accumulated_citations
            )
        });
    let activity_event_id = input
        .activity_event_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);

    // ── Layer 1: conversational fast-path ──────────────────────────────────────
    // A bare greeting / thanks / "who are you" has nothing to retrieve. Answer it
    // directly and skip the whole evidence loop, instead of grinding the step budget
    // and refusing with "insufficient evidence". Conservative by design
    // (runtime::conversation::is_smalltalk): only short, unambiguous smalltalk with no
    // focus document and no text selection — a real question is never brushed off. If
    // no chat provider is configured we fall through so the normal path surfaces the
    // usual "configure a model" error.
    let has_reader_context = !document_id.is_empty()
        || input
            .selected_text
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty());
    if runtime::conversation::is_smalltalk(question, has_reader_context) {
        if let Ok((provider, provider_label)) = provider_result.as_ref() {
            let system = "You are Lumenfolio, a friendly assistant for the user's personal \
                knowledge base. The user sent a brief conversational message (a greeting, thanks, \
                or a question about you), NOT a question about their documents. Reply briefly and \
                warmly in the SAME language as the user's message. Do not mention retrieval, \
                evidence, citations, or documents. You may add one short sentence inviting them to \
                ask about their library.";
            let answer = llm::chat::run_simple_completion(provider, system, question, 0.7, 30)
                .await
                .unwrap_or_else(|_| runtime::conversation::default_reply(question));
            let is_zh = input
                .locale
                .as_deref()
                .is_some_and(|locale| locale.starts_with("zh"))
                || question
                    .chars()
                    .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
            let (ev_title, ev_summary) = if is_zh {
                ("直接回复", "寒暄消息 — 直接作答，无需检索")
            } else {
                (
                    "Direct reply",
                    "Conversational message — answered directly, no retrieval needed",
                )
            };
            let mut trace = runtime::agent::AgentTrace {
                run_id: String::new(),
                intent: "smalltalk".to_string(),
                tree_nodes: Vec::new(),
                candidates: Vec::new(),
                finalize_gate: serde_json::json!({
                    "status": "conversational",
                    "runtime": "smalltalk-direct",
                    "citation_count": 0,
                }),
                evidence_chain: Vec::new(),
                events: vec![runtime::agent::AgentTraceEvent::new(
                    "smalltalk",
                    "answer_direct",
                    "completed",
                    ev_title,
                    ev_summary,
                    "smalltalk gate matched; skipped the retrieval loop",
                )],
                session_summary: None,
                compact: None,
            };
            trace.renumber_events();
            if let Some(event_id) = activity_event_id.as_deref() {
                agent_judge::emit_agent_activity(app, Some(event_id), trace.events[0].clone());
            }
            let provider_id_opt = if selected_provider_id.is_empty() {
                None
            } else {
                Some(selected_provider_id)
            };
            let no_citations: Vec<runtime::rag::Citation> = Vec::new();
            let no_claims: Vec<AskDocumentClaim> = Vec::new();
            runtime::agent::record_completed_turn(
                agent_sessions,
                runtime::agent::CompletedTurnRecord {
                    session_key: &session_id,
                    provider_id: provider_id_opt,
                    question,
                    answer: &answer,
                    selected_text: input.selected_text.as_deref(),
                    citations: &no_citations,
                    trace: &trace,
                },
            );
            if let Err(err) = persist_chat_turn(
                database,
                ChatTurnPersistInput {
                    turn_id: activity_event_id.as_deref(),
                    session_id: &session_id,
                    document_id,
                    provider_id: provider_id_opt,
                    model_key: input.model_key.as_deref(),
                    provider_label: provider_label.as_str(),
                    user_message: question,
                    assistant_answer: &answer,
                    reasoning_content: None,
                    selected_text: input.selected_text.as_deref(),
                    image_data_url: input.image_data_url.as_deref(),
                    citations: &no_citations,
                    claims: &no_claims,
                    retrieval_trace: &trace,
                    referenced_document_ids: &reference_document_ids,
                    knowledge_enabled: input.knowledge_enabled.unwrap_or(true),
                },
            ) {
                log::warn!("Failed to persist smalltalk turn: {err}");
            }
            return Ok(AskDocumentOutput {
                answer,
                reasoning_content: None,
                provider: provider_label.clone(),
                claims: Vec::new(),
                citations: Vec::new(),
                retrieval_trace: trace,
                can_continue_retrieval: false,
                retrieval_attempt_count: 0,
                retrieval_budget_exhausted: false,
            });
        }
    }

    agent_judge::emit_agent_activity(
        app,
        activity_event_id.as_deref(),
        runtime::agent::AgentTraceEvent::new(
            "start",
            "start",
            "running",
            "Starting document agent",
            "Preparing local retrieval and evidence checks",
            format!("question={question} {model_budget_detail}"),
        ),
    );
    let current_view_metadata = build_current_view_gate_metadata(
        database,
        document_id,
        input.viewport_context.as_ref(),
        input.page,
        input.selected_text.as_deref(),
    )?;
    // If the question explicitly names another workspace document, the focus
    // document's current page is noise — suppress current-view seeding so the
    // retrieval loop routes to the named document instead of flooding the
    // evidence with the page the user happens to be viewing. (Selected text is an
    // explicit current-view intent and is handled inside the decision fn, so this
    // only matters for the no-selection case.)
    let targets_other_document = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        question_targets_non_focus_document(&conn, question, document_id)
    };
    let current_view_decision = if targets_other_document
        && input
            .selected_text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
    {
        log::info!("Suppressing current-view seeding: question names a non-focus document");
        None
    } else {
        current_view_decision_for_input(
            question,
            &input,
            current_view_metadata.as_ref(),
            provider_result.as_ref().ok().map(|(provider, _)| provider),
        )
        .await
    };
    let current_view_event = current_view_decision.as_ref().map(|decision| {
        runtime::agent::AgentTraceEvent::new(
            "judge_result",
            "current_view",
            "completed",
            "Current view relevance",
            format!(
                "relevance={} mode={} use_current_view={}",
                decision.relevance, decision.mode, decision.should_use_current_view
            ),
            format!("current_view_gate reason={}", decision.reason),
        )
        .with_judge(serde_json::json!({
            "runtime": "current-view-llm-router",
            "relevance": decision.relevance,
            "mode": decision.mode,
            "shouldUseCurrentView": decision.should_use_current_view,
            "reason": decision.reason,
            "metadata": current_view_metadata.clone()
        }))
    });
    if let Some(event) = current_view_event.as_ref() {
        agent_judge::emit_agent_activity(app, activity_event_id.as_deref(), event.clone());
    }
    let (retrieval_page, retrieval_page_mode, retrieval_page_source) =
        current_view_retrieval_hint(&input, current_view_decision.as_ref());
    let mut agent_run = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        let event_id = activity_event_id.clone();
        runtime::agent::run_turn_with_activity(
            &conn,
            agent_sessions,
            runtime::agent::AgentRunRequest {
                document_id,
                session_key: &session_id,
                visible_document_ids: visible_document_id_refs.clone(),
                question,
                provider_id: Some(selected_provider_id),
                context_budget,
                selected_text: input.selected_text.as_deref(),
                selected_block_id: input.selected_block_id.as_deref(),
                selected_bbox_list: input.selected_bbox_list.clone(),
                page: retrieval_page,
                page_mode: retrieval_page_mode,
                page_source: retrieval_page_source,
                max_retrieval_steps: input.max_retrieval_steps,
                retrieval_attempt_offset: input.retrieval_attempt_offset.unwrap_or(0),
            },
            |event| {
                if let Some(event_id) = &event_id {
                    let _ = app.emit(
                        "lumenfolio://agent-activity",
                        AgentActivityEventOutput {
                            event_id: event_id.clone(),
                            event: event.clone(),
                        },
                    );
                }
            },
        )?
    };
    if let Some(event) = current_view_event {
        agent_run.trace.events.insert(0, event);
    }

    // The unified loop produces the answer itself (retrieval + answering share one
    // growing context). When it runs, we skip the M4 judge AND the separate answer
    // generation below. `None` means we fall through to the legacy M3+M4 path.
    let mut unified_answer: Option<AskAnswerResult> = None;
    // Whether the unified loop ran this turn (Ok OR Err). When it did, we must NOT
    // also run the M4 judge: the loop already mutated `agent_run` (citations,
    // ledger, trace), so a second retrieval pass would double-count evidence. On a
    // unified-loop error we instead fall through to the best-effort/refusal answer
    // path with whatever evidence was already gathered.
    let mut unified_attempted = false;

    // Local-agent providers (Codex / Claude CLI): Mode A — the seed retrieval above
    // already gathered the evidence + citations into `agent_run`; we use the user's
    // local agent purely to GENERATE the answer from that evidence (no HTTP, no
    // tools). Producing the answer into `unified_answer` lets the shared persistence
    // + return tail handle it exactly like the HTTP loop's result.
    if let Some(kind) = local_agent::provider_id_kind(selected_provider_id) {
        unified_attempted = true;
        // Stop button: the same token registered for this turn, so the CLI subprocess
        // is killed when the user cancels (default = a never-cancelled token).
        let agent_cancel = activity_event_id
            .as_deref()
            .and_then(|id| cancellation_token(app, id))
            .unwrap_or_default();

        // Mode B (agentic, P2): whenever a focused, indexed document exists, let the
        // local CLI call Lumenfolio's tools over a loopback MCP server (document +
        // library + trending + knowledge-graph tools) and do its own multi-step
        // retrieval. Only an unindexed/missing document falls back to Mode A.
        // A short note tells the agent what the user is looking at, so off the reader
        // (e.g. Trending) it reaches for list_trending_papers rather than the open doc.
        let view_note = input.view_context.as_ref().and_then(|v| {
            match v.surface.as_deref() {
                Some("trending") => {
                    let period = v.trending_period.as_deref().unwrap_or("daily");
                    Some(format!(
                        "The user is currently viewing the Trending Papers list (period: {period}). \
For questions about what is trending or high-value there, call list_trending_papers \
(period=\"{period}\") instead of searching the open document."
                    ))
                }
                Some("graph") => Some(
                    "The user is currently viewing the cross-document Knowledge Graph; for \
questions spanning their library use search_library_knowledge / query_knowledge_graph."
                        .to_string(),
                ),
                _ => None,
            }
        });
        let (db_path, doc_indexed, local_memory) = match database.conn.lock() {
            Ok(conn) => {
                let path = conn.path().map(|p| p.to_string());
                let indexed = path.is_some()
                    && conn
                        .query_row(
                            "SELECT 1 FROM document_chunks WHERE document_id = ?1 LIMIT 1",
                            params![document_id],
                            |_| Ok(()),
                        )
                        .optional()
                        .ok()
                        .flatten()
                        .is_some();
                // Rich conversation memory (full recent turns) — local agents have the
                // context window to use it, unlike the token-budgeted API path.
                let memory = build_local_agent_memory(&conn, &session_id);
                (path, indexed, memory)
            }
            Err(_) => (None, false, String::new()),
        };
        // Prefer the generous memory; fall back to the compacted one (e.g. if the DB
        // read found nothing).
        let session_memory = if local_memory.trim().is_empty() {
            agent_run.session_context.trim().to_string()
        } else {
            local_memory
        };

        if doc_indexed {
            agent_judge::emit_agent_activity(
                app,
                activity_event_id.as_deref(),
                runtime::agent::AgentTraceEvent::new(
                    "tool_call",
                    "finalize_answer",
                    "running",
                    "Researching with your local agent",
                    "Letting your local Codex/Claude call Lumenfolio's tools to gather evidence, then answer",
                    "local-agent agentic retrieval",
                ),
            );
            let prompt = local_agent::build_agentic_prompt(
                question,
                &session_memory,
                view_note.as_deref(),
                input.locale.as_deref(),
            );
            // Live trace: relay each MCP tool-call step to the chat activity drawer.
            let trace_app = app.clone();
            let trace_event_id = activity_event_id.clone();
            let on_tool = move |ev: local_agent::AgentToolEvent| {
                let label = friendly_tool_label(&ev.tool);
                let (status, title) = match ev.phase {
                    local_agent::AgentToolPhase::Started => ("running", label.to_string()),
                    local_agent::AgentToolPhase::Completed => (
                        if ev.ok { "completed" } else { "failed" },
                        label.to_string(),
                    ),
                };
                agent_judge::emit_agent_activity(
                    &trace_app,
                    trace_event_id.as_deref(),
                    runtime::agent::AgentTraceEvent::new(
                        "tool_call",
                        ev.tool.clone(),
                        status,
                        title,
                        format!("local agent · {}", ev.tool),
                        String::new(),
                    ),
                );
            };
            // Live answer streaming (Claude stream-json): relay each token delta to
            // the chat bubble via the same answer-delta event the HTTP path uses.
            let answer_app = app.clone();
            let answer_event_id = activity_event_id.clone();
            let on_answer = move |delta: String| {
                if let Some(event_id) = answer_event_id.as_deref() {
                    let _ = answer_app.emit(
                        "lumenfolio://answer-delta",
                        AnswerDeltaEventOutput {
                            event_id: event_id.to_string(),
                            delta,
                        },
                    );
                }
            };
            match local_agent::generate_answer_agentic(
                kind,
                std::path::PathBuf::from(db_path.unwrap_or_default()),
                document_id.to_string(),
                prompt,
                input.image_data_url.clone(),
                input.web_enabled.unwrap_or(false),
                agent_cancel.clone(),
                on_tool,
                on_answer,
            )
            .await
            {
                Ok(outcome) => {
                    // Surface the evidence the agent actually grounded on (what the
                    // MCP server served), replacing the seed retrieval's citations.
                    // Swap in the agent's own trace candidates (they carry section
                    // titles per block) and rebuild the chain from the same set so the
                    // evidence chips resolve to real bboxes (click → highlight) AND
                    // show section labels.
                    if !outcome.citations.is_empty() {
                        let citations = dedup_citations(outcome.citations);
                        agent_run.trace.candidates = outcome.candidates;
                        agent_run.trace.rebuild_evidence_chain(&citations);
                        agent_run.retrieval_run.citations = citations;
                    }
                    unified_answer = Some(AskAnswerResult {
                        answer: outcome.answer,
                        reasoning_content: None,
                        claims: Vec::new(),
                    })
                }
                // User stopped it: propagate so the turn finalizes as "stopped".
                Err(err) if err == llm::agent_loop::GENERATION_STOPPED => return Err(err),
                Err(err) => {
                    unified_answer = Some(AskAnswerResult {
                        answer: format!("⚠️ {err}"),
                        reasoning_content: None,
                        claims: Vec::new(),
                    })
                }
            }
        } else {
            // Mode A: the seed retrieval already gathered evidence + citations into
            // `agent_run`; use the local agent purely to GENERATE the answer (no tools).
            agent_judge::emit_agent_activity(
                app,
                activity_event_id.as_deref(),
                runtime::agent::AgentTraceEvent::new(
                    "tool_call",
                    "finalize_answer",
                    "running",
                    "Generating answer with the local agent",
                    "Answering from retrieved evidence using your local Codex/Claude CLI",
                    "local-agent generation",
                ),
            );
            let prompt = local_agent::build_prompt(
                question,
                agent_run.retrieval_run.prompt_context.trim(),
                &session_memory,
                input.locale.as_deref(),
            );
            match local_agent::generate_answer(
                kind,
                prompt,
                input.image_data_url.clone(),
                agent_cancel.clone(),
            )
            .await
            {
                Ok(answer) => {
                    unified_answer = Some(AskAnswerResult {
                        answer,
                        reasoning_content: None,
                        claims: Vec::new(),
                    })
                }
                // User stopped it: propagate so the turn finalizes as "stopped".
                Err(err) if err == llm::agent_loop::GENERATION_STOPPED => return Err(err),
                Err(err) => {
                    unified_answer = Some(AskAnswerResult {
                        answer: format!("⚠️ {err}"),
                        reasoning_content: None,
                        claims: Vec::new(),
                    })
                }
            }
        }
    }

    if unified_answer.is_none() {
        if let Ok((provider, _)) = provider_result.as_ref() {
            // Strong (native tool-calling) models go through the unified agent loop;
            // others (and image questions) use the legacy M4 judge + answer path.
            if llm::agent_loop::should_use_unified_loop(provider, &input) {
                unified_attempted = true;
                // Clone the session memory block up front so the loop can borrow
                // `agent_run` mutably (to accumulate citations) without conflicting.
                let session_context = agent_run.session_context.clone();
                match llm::agent_loop::run_unified_agent_loop(
                    llm::agent_loop::UnifiedLoopInput {
                        input: &input,
                        database,
                        app,
                        question,
                        document_id,
                        visible_document_ids: &unified_visible_document_id_refs,
                        workspace_manifest: &unified_loop_manifest_text,
                        library_is_large,
                        session_context: &session_context,
                        provider,
                        activity_event_id: activity_event_id.as_deref(),
                    },
                    &mut agent_run,
                )
                .await
                {
                    Ok(result) => unified_answer = Some(result),
                    // User stopped it: propagate so the turn finalizes as "stopped"
                    // instead of falling through to another (best-effort) answer attempt.
                    Err(err) if err == llm::agent_loop::GENERATION_STOPPED => return Err(err),
                    Err(err) => {
                        log::warn!(
                            "Unified agent loop failed; answering from gathered evidence: {err}"
                        );
                    }
                }
            }
            if !unified_attempted {
                if let Err(err) = agent_judge::improve_retrieval_with_llm_judge(
                    agent_judge::LlmJudgeLoopInput {
                        input: &input,
                        database,
                        app: Some(app),
                        question,
                        document_id,
                        visible_document_ids: &visible_document_id_refs,
                        workspace_manifest: &workspace_manifest_text,
                        provider,
                        activity_event_id: activity_event_id.as_deref(),
                    },
                    &mut agent_run,
                )
                .await
                {
                    log::warn!("LLM answerability judge skipped: {err}");
                }
            }
        } else if let Err(err) = provider_result.as_ref() {
            let judge_required = !agent_judge::has_image_context(&input)
                && agent_judge::requires_llm_judge_for_answer(question);
            if judge_required {
                let gate = serde_json::json!({
                    "status": "insufficient",
                    "reason": format!(
                        "M4 LLM judge is required for this question, but no configured chat provider was available: {err}"
                    ),
                    "missing": serde_json::json!(["LLM evidence judge"]),
                    "nextToolCall": serde_json::Value::Null,
                    "citationCount": agent_run.retrieval_run.citations.len(),
                    "runtime": "m4-llm-judge",
                    "skipped": false
                });
                agent_run.retrieval_run.trace.finalize_gate = gate.clone();
                agent_run.trace.finalize_gate = gate.clone();
            }
            let event = runtime::agent::AgentTraceEvent::new(
            "judge_result",
            "finalize_answer",
            if judge_required { "error" } else { "skipped" },
            "M4 LLM evidence check",
            if judge_required {
                "No configured chat provider was available; refusing local answerability for this question"
            } else {
                "No configured chat provider was available; LLM evidence judge is not used for image-context requests"
            },
            format!("llm_judge skipped: provider unavailable: {err}"),
        )
        .with_judge(serde_json::json!({
            "status": agent_run
                .trace
                .finalize_gate
                .get("status")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown"),
            "reason": format!("M4 LLM judge unavailable: {err}"),
            "missing": agent_run
                .trace
                .finalize_gate
                .get("missing")
                .cloned()
                .unwrap_or_else(|| serde_json::json!([])),
            "nextToolCall": serde_json::Value::Null,
            "citationCount": agent_run.retrieval_run.citations.len(),
            "runtime": "m4-llm-judge",
            "skipped": !judge_required
        }));
            agent_judge::emit_agent_activity(app, activity_event_id.as_deref(), event.clone());
            agent_run.trace.events.push(event);
        }
    } // end: if unified_answer.is_none()

    attach_context_budget_to_agent_run(&mut agent_run);

    let retrieval_budget_exhausted = agent_judge::retrieval_budget_exhausted(&agent_run);
    let retrieval_attempt_count = agent_judge::retrieval_attempt_count(&agent_run);
    // The unified loop already produced (and streamed) its answer. The refusal /
    // best-effort gating and the separate answer generator below apply only to the
    // legacy M3+M4 fallback path. `provider_label` is needed by both paths for
    // persistence/output, so derive it once without consuming `provider_result`.
    let provider_label = provider_result
        .as_ref()
        .map(|(_, label)| label.clone())
        .unwrap_or_else(|_| "Local retrieval".to_string());

    let answer_result = if let Some(result) = unified_answer {
        result
    } else {
        // Refuse with the "insufficient evidence" message ONLY when the retrieval
        // loop is neither answerable NOR has enough accumulated evidence for a
        // best-effort, caveated answer. When best-effort applies, fall through to
        // the normal answer generator (told to state its limits) so open-ended
        // questions degrade to "here's what I can tell, with stated uncertainty".
        if !agent_judge::has_image_context(&input)
            && !agent_judge::retrieval_is_answerable(&agent_run)
            && !agent_judge::should_answer_best_effort(&agent_run)
        {
            let answer =
                insufficient_evidence_answer(question, &agent_run, input.locale.as_deref());
            runtime::agent::record_completed_turn(
                agent_sessions,
                runtime::agent::CompletedTurnRecord {
                    session_key: &session_id,
                    provider_id: if selected_provider_id.is_empty() {
                        None
                    } else {
                        Some(selected_provider_id)
                    },
                    question,
                    answer: &answer,
                    selected_text: input.selected_text.as_deref(),
                    citations: &agent_run.retrieval_run.citations,
                    trace: &agent_run.trace,
                },
            );
            if let Err(err) = persist_chat_turn(
                database,
                ChatTurnPersistInput {
                    turn_id: activity_event_id.as_deref(),
                    session_id: &session_id,
                    document_id,
                    provider_id: if selected_provider_id.is_empty() {
                        None
                    } else {
                        Some(selected_provider_id)
                    },
                    model_key: input.model_key.as_deref(),
                    provider_label: "Local retrieval",
                    user_message: question,
                    assistant_answer: &answer,
                    reasoning_content: None,
                    selected_text: input.selected_text.as_deref(),
                    image_data_url: input.image_data_url.as_deref(),
                    citations: &agent_run.retrieval_run.citations,
                    claims: &[],
                    retrieval_trace: &agent_run.trace,
                    referenced_document_ids: &reference_document_ids,
                    knowledge_enabled: input.knowledge_enabled.unwrap_or(true),
                },
            ) {
                log::warn!("Failed to persist insufficient-evidence chat turn: {err}");
            }
            return Ok(AskDocumentOutput {
                answer,
                reasoning_content: None,
                provider: "Local retrieval".to_string(),
                claims: Vec::new(),
                citations: agent_run.retrieval_run.citations,
                retrieval_trace: agent_run.trace,
                can_continue_retrieval: false,
                retrieval_attempt_count,
                retrieval_budget_exhausted,
            });
        }

        // We're answering. If the gate never reached "answerable" (e.g. the judge
        // timed out) but we have enough evidence to answer best-effort, stamp the
        // gate so the UI shows "answered with available evidence".
        if !agent_judge::has_image_context(&input)
            && agent_judge::should_answer_best_effort(&agent_run)
        {
            for gate in [
                &mut agent_run.retrieval_run.trace.finalize_gate,
                &mut agent_run.trace.finalize_gate,
            ] {
                if let Some(object) = gate.as_object_mut() {
                    object.insert("bestEffort".to_string(), serde_json::Value::Bool(true));
                }
            }
        }

        let (provider, _) = provider_result?;
        if let Some(event_id) = activity_event_id.as_deref() {
            let _ = app.emit(
                "lumenfolio://agent-activity",
                AgentActivityEventOutput {
                    event_id: event_id.to_string(),
                    event: runtime::agent::AgentTraceEvent::new(
                        "answer_start",
                        "generate_answer",
                        "running",
                        "Generating answer",
                        "Streaming answer from the configured chat model",
                        "streaming answer from the configured chat model",
                    ),
                },
            );
        }
        llm::chat::ask_with_openai_compatible(
            question,
            &input,
            &agent_run,
            &provider,
            &workspace_manifest_text,
            app,
            activity_event_id.as_deref(),
        )
        .await?
    };
    runtime::agent::record_completed_turn(
        agent_sessions,
        runtime::agent::CompletedTurnRecord {
            session_key: &session_id,
            provider_id: if selected_provider_id.is_empty() {
                None
            } else {
                Some(selected_provider_id)
            },
            question,
            answer: &answer_result.answer,
            selected_text: input.selected_text.as_deref(),
            citations: &agent_run.retrieval_run.citations,
            trace: &agent_run.trace,
        },
    );
    if let Err(err) = persist_chat_turn(
        database,
        ChatTurnPersistInput {
            turn_id: activity_event_id.as_deref(),
            session_id: &session_id,
            document_id,
            provider_id: if selected_provider_id.is_empty() {
                None
            } else {
                Some(selected_provider_id)
            },
            model_key: input.model_key.as_deref(),
            provider_label: &provider_label,
            user_message: question,
            assistant_answer: &answer_result.answer,
            reasoning_content: answer_result.reasoning_content.as_deref(),
            selected_text: input.selected_text.as_deref(),
            image_data_url: input.image_data_url.as_deref(),
            citations: &agent_run.retrieval_run.citations,
            claims: &answer_result.claims,
            retrieval_trace: &agent_run.trace,
            referenced_document_ids: &reference_document_ids,
            knowledge_enabled: input.knowledge_enabled.unwrap_or(true),
        },
    ) {
        log::warn!("Failed to persist chat turn: {err}");
    }
    Ok(AskDocumentOutput {
        answer: answer_result.answer,
        reasoning_content: answer_result.reasoning_content,
        provider: provider_label,
        claims: answer_result.claims,
        citations: agent_run.retrieval_run.citations,
        retrieval_trace: agent_run.trace,
        can_continue_retrieval: false,
        retrieval_attempt_count,
        retrieval_budget_exhausted,
    })
}

struct ChatTurnPersistInput<'a> {
    turn_id: Option<&'a str>,
    session_id: &'a str,
    document_id: &'a str,
    provider_id: Option<&'a str>,
    model_key: Option<&'a str>,
    provider_label: &'a str,
    user_message: &'a str,
    assistant_answer: &'a str,
    reasoning_content: Option<&'a str>,
    selected_text: Option<&'a str>,
    image_data_url: Option<&'a str>,
    citations: &'a [runtime::rag::Citation],
    claims: &'a [AskDocumentClaim],
    retrieval_trace: &'a runtime::agent::AgentTrace,
    referenced_document_ids: &'a [String],
    knowledge_enabled: bool,
}

fn persist_chat_turn(
    database: &AppDatabase,
    input: ChatTurnPersistInput<'_>,
) -> Result<(), String> {
    let turn_id = input
        .turn_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| {
            stable_text_hash(&format!(
                "chat-turn:{}:{}:{}",
                input.document_id,
                input.user_message,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_nanos())
                    .unwrap_or_default()
            ))
        });
    let citations_json = serde_json::to_string(input.citations)
        .map_err(|err| format!("Failed to encode chat citations: {err}"))?;
    let claims_json = serde_json::to_string(input.claims)
        .map_err(|err| format!("Failed to encode chat claims: {err}"))?;
    let retrieval_trace_json = serde_json::to_string(input.retrieval_trace)
        .map_err(|err| format!("Failed to encode retrieval trace: {err}"))?;
    let referenced_document_ids_json = serde_json::to_string(input.referenced_document_ids)
        .map_err(|err| format!("Failed to encode referenced document ids: {err}"))?;
    // KB pivot (P1): a library-wide / no-focus turn has no document — store NULL
    // (the column is nullable; the FK rejects an empty-string id).
    let document_id_param: Option<&str> = Some(input.document_id)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    // Make sure the session row exists before we attach turns to it. This keeps
    // persistence self-sufficient for legacy callers (no explicit session) and
    // back-fills the focus document from the turn being saved.
    ensure_chat_session_row(&conn, input.session_id, document_id_param)?;
    conn.execute(
        "INSERT INTO chat_turns
            (id, session_id, document_id, provider_id, model_key, provider_label, user_message,
             assistant_answer, reasoning_content, selected_text, image_data_url, citations_json,
             claims_json, retrieval_trace_json, referenced_document_ids_json, index_version,
             created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, unixepoch(), unixepoch())",
        params![
            turn_id,
            input.session_id,
            document_id_param,
            input.provider_id.unwrap_or(""),
            input.model_key.unwrap_or(""),
            input.provider_label,
            input.user_message,
            input.assistant_answer,
            input.reasoning_content.unwrap_or(""),
            input.selected_text.unwrap_or(""),
            input.image_data_url.unwrap_or(""),
            citations_json,
            claims_json,
            retrieval_trace_json,
            referenced_document_ids_json,
            CURRENT_INDEX_VERSION,
        ],
    )
    .map_err(|err| format!("Failed to save chat turn: {err}"))?;
    // Surface this session at the top of the recency-ordered session list.
    conn.execute(
        "UPDATE chat_sessions SET updated_at = unixepoch() WHERE id = ?1",
        params![input.session_id],
    )
    .map_err(|err| format!("Failed to bump chat session recency: {err}"))?;

    // Knowledge precipitation — Stream 2 (zero extra LLM): distill this turn's
    // claims + co-citation doc<->doc edges from data we just stored. Gated on the
    // knowledge setting; best-effort (a failure must not fail the turn save).
    if input.knowledge_enabled {
        let claim_texts: Vec<&str> = input
            .claims
            .iter()
            .map(|claim| claim.text.as_str())
            .collect();
        let cited_document_ids: Vec<&str> = input
            .citations
            .iter()
            .map(|citation| citation.document_id.as_str())
            .collect();
        if let Err(err) = runtime::precipitation::precipitate_turn(
            &conn,
            &turn_id,
            input.session_id,
            input.document_id,
            input.user_message,
            &claim_texts,
            &cited_document_ids,
            &citations_json,
        ) {
            log::warn!("Stream-2 precipitation failed for turn {turn_id}: {err}");
        }
    }
    Ok(())
}

/// Idempotently create a session row if missing. When `focus_document_id` is
/// provided and non-empty it is used as the focus for a freshly created row;
/// existing rows are never overwritten (INSERT OR IGNORE), so a user's explicit
/// focus choice is preserved.
fn ensure_chat_session_row(
    conn: &Connection,
    session_id: &str,
    focus_document_id: Option<&str>,
) -> Result<(), String> {
    let focus = focus_document_id
        .map(str::trim)
        .filter(|value| !value.is_empty());
    conn.execute(
        "INSERT OR IGNORE INTO chat_sessions
            (id, title, focus_document_id, referenced_document_ids_json, created_at, updated_at)
         VALUES (?1, '', ?2, '[]', unixepoch(), unixepoch())",
        params![session_id, focus],
    )
    .map_err(|err| format!("Failed to ensure chat session: {err}"))?;
    Ok(())
}

fn load_stored_chat_turns(
    conn: &Connection,
    session_id: &str,
    limit: u32,
) -> Result<Vec<StoredChatTurn>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, provider_id, provider_label, user_message, assistant_answer,
                    reasoning_content, selected_text, image_data_url, citations_json, claims_json,
                    retrieval_trace_json, created_at, referenced_document_ids_json
             FROM (
               SELECT rowid AS turn_rowid, *
               FROM chat_turns
               WHERE session_id = ?1 AND index_version = ?2
               ORDER BY created_at DESC, rowid DESC
               LIMIT ?3
             )
             ORDER BY created_at ASC, turn_rowid ASC",
        )
        .map_err(|err| format!("Failed to prepare chat turns query: {err}"))?;
    let turns = stmt
        .query_map(
            params![session_id, CURRENT_INDEX_VERSION, limit],
            read_stored_chat_turn_row,
        )
        .map_err(|err| format!("Failed to load chat turns: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read chat turns: {err}"))?;
    Ok(turns)
}

fn read_stored_chat_turn_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredChatTurn> {
    let citations_json: String = row.get(8)?;
    let claims_json: String = row.get(9)?;
    let retrieval_trace_json: String = row.get(10)?;
    let citations =
        serde_json::from_str::<Vec<runtime::rag::Citation>>(&citations_json).unwrap_or_default();
    let claims = serde_json::from_str::<Vec<AskDocumentClaim>>(&claims_json).unwrap_or_default();
    let retrieval_trace =
        serde_json::from_str::<serde_json::Value>(&retrieval_trace_json).unwrap_or_default();
    let referenced_document_ids_json: String = row.get(12)?;
    let referenced_document_ids =
        serde_json::from_str::<Vec<String>>(&referenced_document_ids_json).unwrap_or_default();
    let provider_id = optional_non_empty(row.get::<_, String>(1)?);
    Ok(StoredChatTurn {
        id: row.get(0)?,
        provider_id,
        provider_label: row.get(2)?,
        user_message: row.get(3)?,
        assistant_answer: row.get(4)?,
        reasoning_content: optional_non_empty(row.get::<_, String>(5)?),
        selected_text: optional_non_empty(row.get::<_, String>(6)?),
        image_data_url: optional_non_empty(row.get::<_, String>(7)?),
        citations,
        claims,
        retrieval_trace,
        referenced_document_ids,
        created_at: row.get(11)?,
    })
}

fn optional_non_empty(value: String) -> Option<String> {
    let value = value.trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn chat_turns_to_messages(turns: Vec<StoredChatTurn>) -> Vec<ChatHistoryMessageOutput> {
    turns
        .into_iter()
        .flat_map(|turn| {
            let user = ChatHistoryMessageOutput {
                id: format!("{}:user", turn.id),
                turn_id: turn.id.clone(),
                role: "user".to_string(),
                content: turn.user_message.clone(),
                reasoning_content: None,
                provider: None,
                citations: Vec::new(),
                claims: Vec::new(),
                retrieval_trace: None,
                image_data_url: turn.image_data_url.clone(),
                referenced_document_ids: turn.referenced_document_ids.clone(),
                created_at: turn.created_at,
            };
            let assistant = ChatHistoryMessageOutput {
                id: format!("{}:assistant", turn.id),
                turn_id: turn.id.clone(),
                role: "assistant".to_string(),
                content: turn.assistant_answer.clone(),
                reasoning_content: turn.reasoning_content.clone(),
                provider: Some(turn.provider_label.clone())
                    .filter(|value| !value.trim().is_empty()),
                citations: turn.citations.clone(),
                claims: turn.claims.clone(),
                retrieval_trace: Some(turn.retrieval_trace.clone()),
                image_data_url: None,
                referenced_document_ids: Vec::new(),
                created_at: turn.created_at,
            };
            [user, assistant]
        })
        .collect()
}

fn restore_agent_session_from_turns(
    agent_sessions: &AgentSessionState,
    session_id: &str,
    turns: &[StoredChatTurn],
) {
    agent_sessions.clear_session(session_id);
    for turn in turns
        .iter()
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let tree_titles = turn
            .retrieval_trace
            .get("treeNodes")
            .and_then(|value| value.as_array())
            .map(|nodes| {
                nodes
                    .iter()
                    .filter_map(|node| node.get("title").and_then(|title| title.as_str()))
                    .take(4)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        runtime::agent::record_restored_turn(
            agent_sessions,
            runtime::agent::RestoredTurnRecord {
                session_key: session_id,
                provider_id: turn.provider_id.as_deref(),
                question: &turn.user_message,
                answer: &turn.assistant_answer,
                selected_text: turn.selected_text.as_deref(),
                citations: &turn.citations,
                tree_titles,
            },
        );
    }
}

fn attach_context_budget_to_agent_run(agent_run: &mut runtime::agent::AgentRunResult) {
    if let Ok(context_budget) = serde_json::to_value(&agent_run.retrieval_run.context_budget) {
        for gate in [
            &mut agent_run.retrieval_run.trace.finalize_gate,
            &mut agent_run.trace.finalize_gate,
        ] {
            if let Some(object) = gate.as_object_mut() {
                object.insert("contextBudget".to_string(), context_budget.clone());
            }
        }
    }
    // P4-4: assign a monotonic seq right before the trace is serialized so
    // the frontend has a stable ordering even after M4 LLM-judge events are
    // appended (and out-of-order ts values from different runtimes can occur).
    agent_run.trace.renumber_events();
}

fn insufficient_evidence_answer(
    question: &str,
    agent_run: &runtime::agent::AgentRunResult,
    locale: Option<&str>,
) -> String {
    let gate = &agent_run.retrieval_run.trace.finalize_gate;
    let reason = gate
        .get("reason")
        .and_then(|value| value.as_str())
        .unwrap_or("The retrieved evidence is not sufficient.");
    let missing = gate
        .get("missing")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|value| !value.is_empty());
    let budget_exhausted = gate
        .get("budgetExhausted")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let max_attempts = gate
        .get("maxAttempts")
        .and_then(|value| value.as_u64())
        .unwrap_or(20);
    let is_chinese = llm::chat::answer_language_for_question(question, locale) == "Chinese";
    let mut message = if is_chinese {
        if budget_exhausted {
            format!("我已经完成本轮 {max_attempts} 次检索，但还没有拿到足够可靠的证据来回答这个问题。\n\n原因：{reason}\n\n这类情况继续重复同一轮检索通常不会带来新证据。建议把问题改得更具体，或先选中文档中的相关段落再提问。")
        } else if let Some(missing) = &missing {
            format!("我还没有拿到足够可靠的证据来回答这个问题。\n\n原因：{reason}\n缺少：{missing}\n\n你可以换一个更具体的问题，或者先选中文档中的相关段落再提问。")
        } else {
            format!("我还没有拿到足够可靠的证据来回答这个问题。\n\n原因：{reason}\n\n你可以换一个更具体的问题，或者先选中文档中的相关段落再提问。")
        }
    } else if budget_exhausted {
        format!("I completed this retrieval budget of {max_attempts} steps, but I still do not have enough reliable evidence to answer.\n\nReason: {reason}\n\nRepeating the same retrieval loop is unlikely to add new evidence. Try asking a more specific question, or select a relevant passage in the PDF before asking.")
    } else if let Some(missing) = &missing {
        format!("I do not have enough reliable evidence to answer yet.\n\nReason: {reason}\nMissing: {missing}\n\nTry asking a more specific question, or select a relevant passage in the PDF before asking.")
    } else {
        format!("I do not have enough reliable evidence to answer yet.\n\nReason: {reason}\n\nTry asking a more specific question, or select a relevant passage in the PDF before asking.")
    };
    // Be transparent about what was already examined so the user knows the
    // answer is "not found after looking", not "didn't bother to look".
    if let Some(coverage) = agent_run.ledger.coverage_summary() {
        if is_chinese {
            message.push_str(&format!("\n\n（已检索过：{coverage}）"));
        } else {
            message.push_str(&format!("\n\n(Already reviewed: {coverage})"));
        }
    }
    message
}

fn read_translation_cache(
    database: &State<'_, AppDatabase>,
    document_id: &str,
    source_hash: &str,
    target_lang: &str,
    provider: &TranslationProvider,
) -> Result<Option<String>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.query_row(
        "SELECT translated_text
         FROM translations
         WHERE document_id = ?1
           AND source_hash = ?2
           AND target_lang = ?3
           AND provider = ?4
         LIMIT 1",
        params![document_id, source_hash, target_lang, provider.cache_key()],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|err| format!("Failed to read translation cache: {err}"))
}

fn write_translation_cache(
    database: &State<'_, AppDatabase>,
    input: TranslationCacheWriteInput<'_>,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "INSERT INTO translations
            (document_id, page_no, block_id, source_hash, target_lang, provider,
             source_text, translated_text, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, unixepoch(), unixepoch())
         ON CONFLICT(document_id, source_hash, target_lang, provider) DO UPDATE SET
           page_no = excluded.page_no,
           block_id = excluded.block_id,
           source_text = excluded.source_text,
           translated_text = excluded.translated_text,
           updated_at = unixepoch()",
        params![
            input.document_id,
            input.page,
            input.block_id,
            input.source_hash,
            input.target_lang,
            input.record.provider_label,
            input.source_text,
            input.record.translated_text
        ],
    )
    .map_err(|err| format!("Failed to cache translation: {err}"))?;
    Ok(())
}

#[tauri::command]
fn load_translation_settings(
    database: State<'_, AppDatabase>,
) -> Result<TranslationSettingsOutput, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    Ok(TranslationSettingsOutput {
        provider: load_app_setting(&conn, "translation_provider")?
            .unwrap_or_else(|| "google-web".to_string()),
        enable_fallback: load_app_setting(&conn, "translation_enable_fallback")?
            .map(|value| parse_bool_setting(&value))
            .unwrap_or(true),
        microsoft_endpoint: load_app_setting(&conn, "translation_microsoft_endpoint")?
            .unwrap_or_else(|| MICROSOFT_TRANSLATOR_DEFAULT_ENDPOINT.to_string()),
        microsoft_region: load_app_setting(&conn, "translation_microsoft_region")?
            .unwrap_or_default(),
        microsoft_has_api_key: load_app_setting(&conn, "translation_microsoft_api_key")?
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false),
    })
}

#[tauri::command]
fn save_translation_settings(
    input: SaveTranslationSettingsInput,
    database: State<'_, AppDatabase>,
) -> Result<TranslationSettingsOutput, String> {
    let provider = normalize_translation_provider_name(&input.provider);
    let enable_fallback = input.enable_fallback.unwrap_or(true);
    if !is_supported_translation_setting(&provider) {
        return Err("Unsupported translation provider".to_string());
    }
    let microsoft_endpoint = normalize_base_url(
        input
            .microsoft_endpoint
            .as_deref()
            .unwrap_or(MICROSOFT_TRANSLATOR_DEFAULT_ENDPOINT),
    );
    let microsoft_region = input
        .microsoft_region
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_string();
    let provided_microsoft_api_key = input
        .microsoft_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if provider == "microsoft" {
        if microsoft_endpoint.is_empty() {
            return Err("Microsoft Translator endpoint is required".to_string());
        }
        if microsoft_region.is_empty() {
            return Err("Microsoft Translator region is required".to_string());
        }
    }

    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let previous_microsoft_api_key =
        load_app_setting(&conn, "translation_microsoft_api_key")?.unwrap_or_default();
    let microsoft_api_key = provided_microsoft_api_key
        .map(ToString::to_string)
        .unwrap_or(previous_microsoft_api_key);
    if provider == "microsoft" && microsoft_api_key.trim().is_empty() {
        return Err("Microsoft Translator API key is required".to_string());
    }
    save_app_setting(&conn, "translation_provider", &provider)?;
    save_app_setting(
        &conn,
        "translation_enable_fallback",
        if enable_fallback { "1" } else { "0" },
    )?;
    if provider == "microsoft" {
        save_app_setting(&conn, "translation_microsoft_endpoint", &microsoft_endpoint)?;
        save_app_setting(&conn, "translation_microsoft_region", &microsoft_region)?;
        save_app_setting(&conn, "translation_microsoft_api_key", &microsoft_api_key)?;
    }
    Ok(TranslationSettingsOutput {
        provider,
        enable_fallback,
        microsoft_endpoint,
        microsoft_region,
        microsoft_has_api_key: !microsoft_api_key.trim().is_empty(),
    })
}

#[tauri::command]
fn load_web_search_settings(
    database: State<'_, AppDatabase>,
) -> Result<WebSearchSettingsOutput, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    Ok(WebSearchSettingsOutput {
        exa_has_api_key: load_app_setting(&conn, "exa_api_key")?
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false),
    })
}

#[tauri::command]
fn save_web_search_settings(
    input: SaveWebSearchSettingsInput,
    database: State<'_, AppDatabase>,
) -> Result<WebSearchSettingsOutput, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    // None → leave the stored key untouched; Some("") → clear; Some(key) → set.
    if let Some(raw) = input.exa_api_key {
        save_app_setting(&conn, "exa_api_key", raw.trim())?;
    }
    Ok(WebSearchSettingsOutput {
        exa_has_api_key: load_app_setting(&conn, "exa_api_key")?
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false),
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProxySettingsOutput {
    proxy_url: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveProxySettingsInput {
    proxy_url: String,
}

#[tauri::command]
fn load_proxy_settings(database: State<'_, AppDatabase>) -> Result<ProxySettingsOutput, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    Ok(ProxySettingsOutput {
        proxy_url: load_app_setting(&conn, "proxy_url")?.unwrap_or_default(),
    })
}

#[tauri::command]
fn save_proxy_settings(
    input: SaveProxySettingsInput,
    database: State<'_, AppDatabase>,
) -> Result<ProxySettingsOutput, String> {
    let trimmed = input.proxy_url.trim().to_string();
    {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        save_app_setting(&conn, "proxy_url", &trimmed)?;
    }
    // Update the live global so new HTTP clients pick it up without a restart.
    net::set_proxy(if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.clone())
    });
    Ok(ProxySettingsOutput { proxy_url: trimmed })
}

#[tauri::command]
fn save_model_provider(
    input: SaveModelProviderInput,
    database: State<'_, AppDatabase>,
) -> Result<ModelProviderOutput, String> {
    let provider_id = input
        .id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| providers::new_model_provider_id(input.name.trim()));
    let name = input.name.trim();
    let provider_type = input.provider_type.trim();
    let base_url = normalize_base_url(&input.base_url);
    let mut models =
        providers::normalize_provider_models(provider_type, &provider_id, input.models)?;
    let default_model_key =
        providers::resolve_default_model_key(&models, input.default_model_key.as_deref())?;
    providers::apply_default_model_key(&mut models, &default_model_key);

    if name.is_empty() {
        return Err("Provider name is required".to_string());
    }
    if !providers::is_openai_compatible_provider_type(provider_type) {
        return Err("Only OpenAI-compatible providers are supported for now".to_string());
    }
    if base_url.is_empty() {
        return Err("Base URL is required".to_string());
    }
    let models_json = serde_json::to_string(&models)
        .map_err(|err| format!("Failed to encode provider models: {err}"))?;

    let provided_api_key = input
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let secret_ref = providers::legacy_api_key_secret_ref(&provider_id);

    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let (_previous_has_api_key, previous_api_key_local) = conn
        .query_row(
            "SELECT has_api_key, api_key_local FROM model_providers WHERE id = ?1",
            params![&provider_id],
            |row| Ok((row.get::<_, bool>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|err| format!("Failed to read existing provider key state: {err}"))?
        .unwrap_or((false, String::new()));
    let api_key_local = provided_api_key
        .map(ToString::to_string)
        .unwrap_or(previous_api_key_local);
    let has_api_key = !api_key_local.trim().is_empty();
    let existing_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM model_providers", [], |row| row.get(0))
        .map_err(|err| format!("Failed to count model providers: {err}"))?;
    let should_be_default = input.is_default || existing_count == 0;
    if should_be_default {
        conn.execute("UPDATE model_providers SET is_default = 0", [])
            .map_err(|err| format!("Failed to update default provider: {err}"))?;
    }

    conn.execute(
        "INSERT INTO model_providers
            (id, name, provider_type, base_url, model, models_json, default_model_key, enabled, is_default,
             api_key_secret_ref, has_api_key, api_key_local, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, unixepoch(), unixepoch())
         ON CONFLICT(id) DO UPDATE SET
           name = excluded.name,
           provider_type = excluded.provider_type,
           base_url = excluded.base_url,
           model = excluded.model,
           models_json = excluded.models_json,
           default_model_key = excluded.default_model_key,
           enabled = excluded.enabled,
           is_default = excluded.is_default,
           api_key_secret_ref = excluded.api_key_secret_ref,
           has_api_key = excluded.has_api_key,
           api_key_local = excluded.api_key_local,
           updated_at = unixepoch()",
        params![
            &provider_id,
            name,
            provider_type,
            &base_url,
            models
                .iter()
                .find(|model| model.key == default_model_key)
                .map(|model| model.model_id.as_str())
                .unwrap_or(""),
            models_json,
            default_model_key,
            input.enabled,
            should_be_default,
            &secret_ref,
            has_api_key,
            api_key_local
        ],
    )
    .map_err(|err| format!("Failed to save model provider: {err}"))?;

    providers::load_model_provider_by_id(&conn, &provider_id)?
        .map(providers::provider_output)
        .ok_or_else(|| "Saved provider was not found".to_string())
}

#[tauri::command]
fn delete_model_provider(
    input: DeleteModelProviderInput,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let provider_id = input.id.trim();
    if provider_id.is_empty() {
        return Err("Provider id is required".to_string());
    }

    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let deleted_count = conn
        .execute(
            "DELETE FROM model_providers WHERE id = ?1",
            params![provider_id],
        )
        .map_err(|err| format!("Failed to delete model provider: {err}"))?;
    if deleted_count == 0 {
        return Err("Provider was not found".to_string());
    }

    let default_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM model_providers WHERE is_default = 1",
            [],
            |row| row.get(0),
        )
        .map_err(|err| format!("Failed to inspect default model provider: {err}"))?;

    if default_count == 0 {
        let fallback_provider_id = conn
            .query_row(
                "SELECT id FROM model_providers ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|err| format!("Failed to choose replacement default provider: {err}"))?;

        if let Some(fallback_provider_id) = fallback_provider_id {
            conn.execute(
                "UPDATE model_providers SET is_default = CASE WHEN id = ?1 THEN 1 ELSE 0 END",
                params![fallback_provider_id],
            )
            .map_err(|err| format!("Failed to assign replacement default provider: {err}"))?;
        }
    }

    Ok(())
}

#[tauri::command]
async fn test_model_provider(
    input: TestModelProviderInput,
    database: State<'_, AppDatabase>,
) -> Result<ModelProviderTestOutput, String> {
    let provider_type = input.provider_type.trim();
    if !providers::is_openai_compatible_provider_type(provider_type) {
        return Err("Only OpenAI-compatible providers are supported for now".to_string());
    }

    let base_url = normalize_base_url(&input.base_url);
    let model = input.model_id.trim().to_string();
    if base_url.is_empty() {
        return Err("Base URL is required".to_string());
    }
    if model.is_empty() {
        return Err("Model is required".to_string());
    }

    let api_key = input
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            input.id.as_deref().map(str::trim).and_then(|id| {
                providers::read_local_model_provider_api_key(&database, id)
                    .ok()
                    .flatten()
            })
        });

    let model_profile =
        model_catalog::resolve_model_profile(&input.provider_type, &base_url, &model);
    let context_budget = model_profile.context_budget();
    let provider = OpenAiCompatibleProvider {
        base_url,
        api_key,
        model,
        capabilities: vec!["text".to_string()],
        model_profile,
        context_budget,
    };
    llm::chat::test_openai_compatible_provider(&provider).await?;
    Ok(ModelProviderTestOutput {
        ok: true,
        message: format!("{} connected", input.name.trim()),
    })
}

#[tauri::command]
async fn fetch_provider_models(
    input: FetchProviderModelsInput,
    database: State<'_, AppDatabase>,
) -> Result<FetchProviderModelsOutput, String> {
    let provider_type = input.provider_type.trim();
    if !providers::is_openai_compatible_provider_type(provider_type) {
        return Err("Only OpenAI-compatible providers are supported for now".to_string());
    }

    let base_url = normalize_base_url(&input.base_url);
    if base_url.is_empty() {
        return Err("Base URL is required".to_string());
    }

    let api_key = input
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            input.id.as_deref().map(str::trim).and_then(|id| {
                providers::read_local_model_provider_api_key(&database, id)
                    .ok()
                    .flatten()
            })
        });

    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| format!("Failed to create model list client: {err}"))?;
    let mut builder = client.get(format!("{base_url}/models"));
    if let Some(api_key) = &api_key {
        builder = builder.bearer_auth(api_key);
    }

    let response = builder
        .send()
        .await
        .map_err(|err| format!("Model list request failed: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Model list request returned {status}: {}",
            truncate_for_error(&body, 600)
        ));
    }

    let parsed = response
        .json::<OpenAiModelsResponse>()
        .await
        .map_err(|err| format!("Failed to decode model list response: {err}"))?;
    let mut context_windows = std::collections::HashMap::new();
    let mut model_ids = Vec::new();
    for item in parsed.data {
        let id = item.id.trim().to_string();
        if id.is_empty() {
            continue;
        }
        if let Some(window) = item.detected_context_window() {
            context_windows.insert(id.clone(), window);
        }
        model_ids.push(id);
    }
    model_ids.sort();
    model_ids.dedup();

    Ok(FetchProviderModelsOutput {
        model_ids,
        context_windows,
    })
}

#[tauri::command]
async fn test_translation_provider(
    input: TestTranslationProviderInput,
    database: State<'_, AppDatabase>,
) -> Result<ModelProviderTestOutput, String> {
    let provider_name = normalize_translation_provider_name(&input.provider);
    if !is_supported_translation_setting(&provider_name) {
        return Err("Unsupported translation provider".to_string());
    }

    let provider = if provider_name == "microsoft" {
        resolve_microsoft_translation_provider_from_input(&input, &database)
    } else {
        resolve_translation_provider(Some(&provider_name), &database)
    };
    let translated_text = translation::translate_with_provider("hello", "zh", &provider).await?;
    if translated_text.trim().is_empty() {
        return Err("Translation provider returned an empty response".to_string());
    }

    Ok(ModelProviderTestOutput {
        ok: true,
        message: format!("{}: {}", provider.label(), translated_text.trim()),
    })
}

/// Run blocking filesystem work off the Tauri UI thread.
///
/// Sync commands execute on the main thread. `std::fs::read` / `canonicalize`
/// of an iCloud, File Provider, or network path can sit in the kernel for tens
/// of seconds and freeze the window (macOS rainbow cursor) until it returns
/// `ETIMEDOUT`. `async` plus `spawn_blocking` keeps the event loop free so the
/// viewer can keep showing its loading state.
pub(crate) async fn run_blocking_io<T, F>(task: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(task)
        .await
        .map_err(|err| format!("Background I/O task failed: {err}"))?
}

pub(crate) fn map_file_read_error(what: &str, path: &Path, err: std::io::Error) -> String {
    if err.kind() == std::io::ErrorKind::TimedOut || err.raw_os_error() == Some(60) {
        format!(
            "Failed to read {what}: timed out while opening {}. The file may still be downloading from iCloud or a network drive.",
            path.display()
        )
    } else {
        format!("Failed to read {what}: {err}")
    }
}

fn registry_path(registry: &PdfRegistry, doc_id: &str, unknown: &str) -> Result<PathBuf, String> {
    let paths = registry
        .paths
        .lock()
        .map_err(|_| "PDF registry lock was poisoned".to_string())?;
    paths
        .get(doc_id)
        .cloned()
        .ok_or_else(|| unknown.to_string())
}

#[tauri::command]
async fn read_pdf_bytes(
    doc_id: String,
    registry: State<'_, PdfRegistry>,
) -> Result<tauri::ipc::Response, String> {
    let path = registry_path(&registry, &doc_id, "Unknown PDF document id")?;

    // Return raw bytes via `Response` so the webview receives an ArrayBuffer.
    // A plain `Vec<u8>` would serialize as a JSON array of numbers, which costs
    // multiple seconds of main-thread deserialization for multi-MB PDFs.
    let bytes = run_blocking_io(move || {
        fs::read(&path).map_err(|err| map_file_read_error("PDF", &path, err))
    })
    .await?;
    Ok(tauri::ipc::Response::new(bytes))
}

/// Knowledge-base pivot (P3): serve a file-backed document's raw bytes to the
/// webview (ArrayBuffer) for client-side Office preview (docx/xlsx). Same
/// registry-backed path resolution as PDFs.
#[tauri::command]
async fn read_document_bytes(
    doc_id: String,
    registry: State<'_, PdfRegistry>,
) -> Result<tauri::ipc::Response, String> {
    let path = registry_path(&registry, &doc_id, "Unknown document id")?;
    let bytes = run_blocking_io(move || {
        fs::read(&path).map_err(|err| map_file_read_error("document", &path, err))
    })
    .await?;
    Ok(tauri::ipc::Response::new(bytes))
}

pub(crate) fn stable_text_hash(text: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("txt-{hash:016x}")
}

fn parse_bool_setting(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub(crate) fn translation_fallback_enabled(database: &State<'_, AppDatabase>) -> bool {
    let conn = match database.conn.lock() {
        Ok(conn) => conn,
        Err(_) => return true,
    };
    load_app_setting(&conn, "translation_enable_fallback")
        .ok()
        .flatten()
        .map(|value| parse_bool_setting(&value))
        .unwrap_or(true)
}

fn resolve_selected_translation_provider_name(
    requested_provider: Option<&str>,
    database: &State<'_, AppDatabase>,
) -> String {
    requested_provider
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(normalize_translation_provider_name)
        .or_else(|| load_configured_translation_provider_name(database))
        .unwrap_or_else(|| "google-web".to_string())
}

pub(crate) fn build_translation_attempts(
    selected_provider_name: &str,
    database: &State<'_, AppDatabase>,
    enable_fallback: bool,
) -> Vec<TranslationAttempt> {
    let mut attempts = vec![TranslationAttempt {
        provider: resolve_translation_provider(Some(selected_provider_name), database),
    }];

    if !enable_fallback {
        return attempts;
    }

    match selected_provider_name {
        "google-web" => {
            if let Some(provider) = configured_microsoft_translation_provider(database) {
                attempts.push(TranslationAttempt { provider });
            }
            if let Some(provider) = resolve_llm_translation_provider(database) {
                attempts.push(TranslationAttempt { provider });
            }
            attempts.push(TranslationAttempt {
                provider: TranslationProvider::LocalPlaceholder,
            });
        }
        "microsoft" => {
            if let Some(provider) = resolve_llm_translation_provider(database) {
                attempts.push(TranslationAttempt { provider });
            }
            attempts.push(TranslationAttempt {
                provider: TranslationProvider::LocalPlaceholder,
            });
        }
        "llm" => {
            attempts.push(TranslationAttempt {
                provider: TranslationProvider::LocalPlaceholder,
            });
        }
        "local-placeholder" => {}
        _ => {
            attempts.push(TranslationAttempt {
                provider: TranslationProvider::LocalPlaceholder,
            });
        }
    }

    dedupe_translation_attempts(attempts)
}

fn dedupe_translation_attempts(attempts: Vec<TranslationAttempt>) -> Vec<TranslationAttempt> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for attempt in attempts {
        let cache_key = attempt.provider.cache_key();
        if seen.insert(cache_key) {
            deduped.push(attempt);
        }
    }
    deduped
}

pub(crate) fn format_translation_attempt_chain(
    attempts: &[TranslationAttempt],
    end_index: usize,
) -> String {
    attempts
        .iter()
        .take(end_index + 1)
        .map(|attempt| attempt.provider.label())
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn resolve_translation_provider(
    requested_provider: Option<&str>,
    database: &State<'_, AppDatabase>,
) -> TranslationProvider {
    let requested = resolve_selected_translation_provider_name(requested_provider, database);
    let requested = requested.as_str();
    if requested == "local-placeholder" {
        return TranslationProvider::LocalPlaceholder;
    }
    if requested == "google-web" {
        return TranslationProvider::GoogleWeb;
    }
    if requested == "microsoft" {
        return resolve_microsoft_translation_provider(database);
    }
    if requested == "llm" {
        if let Some(provider) = resolve_llm_translation_provider(database) {
            return provider;
        }
        return TranslationProvider::Unavailable {
            cache_key: "llm".to_string(),
            label: "llm".to_string(),
            message: "LLM translation provider is not configured. Add a model provider in Settings or set translation environment variables.".to_string(),
        };
    }

    let stored_setting = requested.to_string();

    match stored_setting.as_str() {
        "google-web" => return TranslationProvider::GoogleWeb,
        "microsoft" => return resolve_microsoft_translation_provider(database),
        "local-placeholder" => return TranslationProvider::LocalPlaceholder,
        "llm" => {
            if let Some(provider) = resolve_llm_translation_provider(database) {
                return provider;
            }
        }
        _ => {
            if let Some(provider) = resolve_stored_translation_provider(&stored_setting, database) {
                return provider;
            }
        }
    }

    let configured_provider = env_var("LUMENFOLIO_TRANSLATION_PROVIDER")
        .unwrap_or_else(|| stored_setting.to_string())
        .to_lowercase();
    let base_url = env_var("LUMENFOLIO_TRANSLATION_BASE_URL")
        .or_else(|| env_var("OPENAI_BASE_URL"))
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
    let api_key = env_var("LUMENFOLIO_TRANSLATION_API_KEY").or_else(|| env_var("OPENAI_API_KEY"));
    let model = env_var("LUMENFOLIO_TRANSLATION_MODEL").or_else(|| env_var("OPENAI_MODEL"));

    if matches!(
        configured_provider.as_str(),
        "openai-compatible" | "openai" | "deepseek" | "openrouter"
    ) {
        if let Some(model) = model {
            let model_profile =
                model_catalog::resolve_model_profile(&configured_provider, &base_url, &model);
            let context_budget = model_profile.context_budget();
            return TranslationProvider::OpenAiCompatible(Box::new(OpenAiCompatibleProvider {
                base_url,
                api_key,
                model,
                capabilities: vec!["text".to_string()],
                model_profile,
                context_budget,
            }));
        }
    }

    TranslationProvider::LocalPlaceholder
}

fn load_configured_translation_provider_name(database: &State<'_, AppDatabase>) -> Option<String> {
    let conn = database.conn.lock().ok()?;
    load_app_setting(&conn, "translation_provider")
        .ok()
        .flatten()
        .map(|value| normalize_translation_provider_name(&value))
}

fn normalize_translation_provider_name(provider: &str) -> String {
    let value = provider.trim().to_lowercase();
    if value.is_empty() {
        "google-web".to_string()
    } else {
        value
    }
}

fn is_supported_translation_setting(provider: &str) -> bool {
    matches!(
        provider,
        "google-web" | "microsoft" | "llm" | "local-placeholder"
    )
}

fn resolve_microsoft_translation_provider(
    database: &State<'_, AppDatabase>,
) -> TranslationProvider {
    let Some(provider) = load_microsoft_translation_provider(database) else {
        return TranslationProvider::Unavailable {
            cache_key: "microsoft".to_string(),
            label: "microsoft".to_string(),
            message: "Microsoft Translator is not configured. Add endpoint, region, and API key in Settings.".to_string(),
        };
    };
    TranslationProvider::Microsoft(provider)
}

fn configured_microsoft_translation_provider(
    database: &State<'_, AppDatabase>,
) -> Option<TranslationProvider> {
    load_microsoft_translation_provider(database).map(TranslationProvider::Microsoft)
}

fn resolve_microsoft_translation_provider_from_input(
    input: &TestTranslationProviderInput,
    database: &State<'_, AppDatabase>,
) -> TranslationProvider {
    let endpoint = normalize_base_url(
        input
            .microsoft_endpoint
            .as_deref()
            .unwrap_or(MICROSOFT_TRANSLATOR_DEFAULT_ENDPOINT),
    );
    let region = input
        .microsoft_region
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_string();
    let api_key = input
        .microsoft_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| load_local_microsoft_api_key(database));

    match (endpoint.is_empty(), region.is_empty(), api_key) {
        (false, false, Some(api_key)) => {
            TranslationProvider::Microsoft(MicrosoftTranslatorProvider {
                endpoint,
                region,
                api_key,
            })
        }
        _ => resolve_microsoft_translation_provider(database),
    }
}

fn load_microsoft_translation_provider(
    database: &State<'_, AppDatabase>,
) -> Option<MicrosoftTranslatorProvider> {
    let conn = database.conn.lock().ok()?;
    let endpoint = load_app_setting(&conn, "translation_microsoft_endpoint")
        .ok()
        .flatten()
        .unwrap_or_else(|| MICROSOFT_TRANSLATOR_DEFAULT_ENDPOINT.to_string());
    let region = load_app_setting(&conn, "translation_microsoft_region")
        .ok()
        .flatten()
        .unwrap_or_default();
    let api_key = load_local_microsoft_api_key_from_conn(&conn)?;

    let endpoint = normalize_base_url(&endpoint);
    let region = region.trim().to_string();
    if endpoint.is_empty() || region.is_empty() {
        return None;
    }
    Some(MicrosoftTranslatorProvider {
        endpoint,
        region,
        api_key,
    })
}

fn load_local_microsoft_api_key(database: &State<'_, AppDatabase>) -> Option<String> {
    let conn = database.conn.lock().ok()?;
    load_local_microsoft_api_key_from_conn(&conn)
}

fn load_local_microsoft_api_key_from_conn(conn: &Connection) -> Option<String> {
    load_app_setting(conn, "translation_microsoft_api_key")
        .ok()
        .flatten()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn resolve_stored_translation_provider(
    requested_provider_id: &str,
    database: &State<'_, AppDatabase>,
) -> Option<TranslationProvider> {
    let conn = database.conn.lock().ok()?;
    let stored = if requested_provider_id.is_empty() {
        providers::load_default_model_provider(&conn).ok().flatten()
    } else {
        providers::load_model_provider_by_id(&conn, requested_provider_id)
            .ok()
            .flatten()
    }?;
    if !providers::is_openai_compatible_provider_type(&stored.provider_type) {
        return None;
    }
    let selected_model =
        providers::pick_chat_model(&stored, Some(&stored.default_model_key)).ok()?;
    let model_profile = model_catalog::resolve_model_profile(
        &stored.provider_type,
        &stored.base_url,
        &selected_model.model_id,
    );
    let context_budget = model_profile.context_budget();

    Some(TranslationProvider::OpenAiCompatible(Box::new(
        OpenAiCompatibleProvider {
            base_url: stored.base_url.clone(),
            api_key: Some(stored.api_key_local.trim().to_string())
                .filter(|value| !value.is_empty()),
            model: selected_model.model_id.clone(),
            capabilities: selected_model.capabilities.clone(),
            model_profile,
            context_budget,
        },
    )))
}

fn resolve_llm_translation_provider(
    database: &State<'_, AppDatabase>,
) -> Option<TranslationProvider> {
    if let Some(provider) = resolve_stored_translation_provider("", database) {
        return Some(provider);
    }

    let configured_provider = env_var("LUMENFOLIO_TRANSLATION_PROVIDER")
        .unwrap_or_else(|| "openai-compatible".to_string())
        .to_lowercase();
    let base_url = env_var("LUMENFOLIO_TRANSLATION_BASE_URL")
        .or_else(|| env_var("OPENAI_BASE_URL"))
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
    let api_key = env_var("LUMENFOLIO_TRANSLATION_API_KEY").or_else(|| env_var("OPENAI_API_KEY"));
    let model = env_var("LUMENFOLIO_TRANSLATION_MODEL").or_else(|| env_var("OPENAI_MODEL"));

    if matches!(
        configured_provider.as_str(),
        "openai-compatible" | "openai" | "deepseek" | "openrouter"
    ) {
        if let Some(model) = model {
            let model_profile =
                model_catalog::resolve_model_profile(&configured_provider, &base_url, &model);
            let context_budget = model_profile.context_budget();
            return Some(TranslationProvider::OpenAiCompatible(Box::new(
                OpenAiCompatibleProvider {
                    base_url,
                    api_key,
                    model,
                    capabilities: vec!["text".to_string()],
                    model_profile,
                    context_budget,
                },
            )));
        }
    }

    None
}

fn env_var(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(crate) fn load_app_setting(conn: &Connection, key: &str) -> Result<Option<String>, String> {
    conn.query_row(
        "SELECT value FROM app_settings WHERE key = ?1 LIMIT 1",
        params![key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(|err| format!("Failed to load app setting {key}: {err}"))
}

pub(crate) fn save_app_setting(conn: &Connection, key: &str, value: &str) -> Result<(), String> {
    conn.execute(
        "INSERT INTO app_settings (key, value, updated_at)
         VALUES (?1, ?2, unixepoch())
         ON CONFLICT(key) DO UPDATE SET
           value = excluded.value,
           updated_at = unixepoch()",
        params![key, value],
    )
    .map_err(|err| format!("Failed to save app setting {key}: {err}"))?;
    Ok(())
}

fn normalize_base_url(base_url: &str) -> String {
    base_url.trim().trim_end_matches('/').to_string()
}

fn truncate_for_error(value: &str, max_chars: usize) -> String {
    let mut result = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        result.push_str("...");
    }
    result
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn map_file_read_error_explains_timeout() {
        let err = std::io::Error::from(std::io::ErrorKind::TimedOut);
        let message = map_file_read_error("PDF", Path::new("/tmp/paper.pdf"), err);
        assert!(message.contains("timed out while opening"));
        assert!(message.contains("iCloud"));
        assert!(message.contains("/tmp/paper.pdf"));
    }

    #[test]
    fn map_file_read_error_keeps_ordinary_io_errors() {
        let err = std::io::Error::from(std::io::ErrorKind::NotFound);
        let message = map_file_read_error("PDF", Path::new("/tmp/missing.pdf"), err);
        assert!(message.starts_with("Failed to read PDF:"));
        assert!(!message.contains("iCloud"));
    }

    #[test]
    fn mentions_token_requires_word_boundaries() {
        // Whole-token match.
        assert!(mentions_token(
            "what is forge_ieee_preprint.pdf about",
            "forge_ieee_preprint"
        ));
        assert!(mentions_token("summarize data today", "data"));
        // Substring inside a larger word must NOT match.
        assert!(!mentions_token("explain the database schema", "data"));
        assert!(!mentions_token("metadata pipeline", "data"));
        assert!(!mentions_token("anything", ""));
    }

    #[test]
    fn coerce_context_window_accepts_int_float_string() {
        assert_eq!(
            coerce_context_window(&serde_json::json!(128000)),
            Some(128000)
        );
        assert_eq!(
            coerce_context_window(&serde_json::json!(128000.0)),
            Some(128000)
        );
        assert_eq!(
            coerce_context_window(&serde_json::json!("128000")),
            Some(128000)
        );
        // Junk / too-small / non-numeric → None (never breaks the decode).
        assert_eq!(coerce_context_window(&serde_json::json!("n/a")), None);
        assert_eq!(coerce_context_window(&serde_json::json!(512)), None);
        assert_eq!(coerce_context_window(&serde_json::json!(true)), None);
    }

    #[test]
    fn budget_exhausted_chinese_answer_does_not_suggest_continue_retrieval() {
        let run = insufficient_run(true);
        let answer = insufficient_evidence_answer("这篇文章讲了什么？", &run, Some("zh"));

        assert!(agent_judge::retrieval_budget_exhausted(&run));
        assert!(!answer.contains("继续检索"));
        assert!(answer.contains("更具体"));
        assert!(answer.contains("选中文档"));
    }

    #[test]
    fn budget_exhausted_english_answer_does_not_suggest_continue_retrieval() {
        let run = insufficient_run(true);
        let answer = insufficient_evidence_answer("What is this paper about?", &run, Some("en"));

        assert!(agent_judge::retrieval_budget_exhausted(&run));
        assert!(!answer.contains("continue retrieval"));
        assert!(answer.contains("Repeating the same retrieval loop"));
        assert!(answer.contains("select a relevant passage"));
    }

    #[test]
    fn llm_judge_gets_one_round_even_after_local_budget_exhaustion() {
        let (tool_steps, judge_rounds) = agent_judge::llm_judge_budget(20, 20);

        assert_eq!(tool_steps, 0);
        assert_eq!(judge_rounds, 1);
    }

    #[test]
    fn llm_judge_reserves_final_rejudge_after_tool_steps() {
        let (tool_steps, judge_rounds) = agent_judge::llm_judge_budget(20, 18);

        assert_eq!(tool_steps, 2);
        assert_eq!(judge_rounds, 3);
    }

    #[test]
    fn llm_hard_gate_rejects_answerable_without_citations() {
        let mut decision = llm::chat::LlmAnswerabilityDecision {
            status: "answerable".to_string(),
            reason: "enough".to_string(),
            missing: Vec::new(),
            next_tool_call: None,
        };

        agent_judge::enforce_llm_judge_hard_gate("What is this paper about?", &[], &mut decision);

        assert_eq!(decision.status, "insufficient");
        assert!(decision.reason.contains("no citations"));
    }

    #[test]
    fn llm_hard_gate_requires_open_table_for_table_questions() {
        let citation = runtime::rag::Citation {
            id: "c1".to_string(),
            label: "[1]".to_string(),
            page: 8,
            block_id: "table-3".to_string(),
            section_title: Some("Table 3".to_string()),
            quote: "Table 3 | SWE-Pruner | Tokens (M) = 0.670".to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "table_fact".to_string(),
        };
        let mut decision = llm::chat::LlmAnswerabilityDecision {
            status: "answerable".to_string(),
            reason: "enough".to_string(),
            missing: Vec::new(),
            next_tool_call: None,
        };

        agent_judge::enforce_llm_judge_hard_gate(
            "文章里面 Table 3 的 SWE-Pruner 的数值是什么结果？",
            &[citation],
            &mut decision,
        );

        assert_eq!(decision.status, "needs_more_evidence");
        let next_tool = decision.next_tool_call.expect("open table tool");
        assert_eq!(next_tool.tool, "open_table");
    }

    #[test]
    fn llm_hard_gate_accepts_requested_open_table_evidence() {
        let citation = runtime::rag::Citation {
            id: "c1".to_string(),
            label: "[1]".to_string(),
            page: 8,
            block_id: "table-3".to_string(),
            section_title: Some("Table 3".to_string()),
            quote: "Table 3 | SWE-Pruner | Tokens (M) = 0.670".to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "open_table".to_string(),
        };
        let mut decision = llm::chat::LlmAnswerabilityDecision {
            status: "answerable".to_string(),
            reason: "enough".to_string(),
            missing: Vec::new(),
            next_tool_call: None,
        };

        agent_judge::enforce_llm_judge_hard_gate(
            "文章里面 Table 3 的 SWE-Pruner 的数值是什么结果？",
            &[citation],
            &mut decision,
        );

        assert_eq!(decision.status, "answerable");
        assert!(decision.next_tool_call.is_none());
    }

    #[test]
    fn llm_hard_gate_accepts_current_view_requested_table_evidence() {
        let citation = runtime::rag::Citation {
            id: "c1".to_string(),
            label: "[1]".to_string(),
            page: 8,
            block_id: "page-lines-8-1-12".to_string(),
            section_title: Some("Current view page evidence: Page 8 lines 1-12".to_string()),
            quote: "Table 3\nMethod Rounds Success (%) Tokens (M)\nSWE-Pruner 41.1 64.0 0.670"
                .to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "current_view".to_string(),
        };
        let mut decision = llm::chat::LlmAnswerabilityDecision {
            status: "answerable".to_string(),
            reason: "enough".to_string(),
            missing: Vec::new(),
            next_tool_call: None,
        };

        agent_judge::enforce_llm_judge_hard_gate(
            "文章里面 Table 3 的 SWE-Pruner 的数值是什么结果？",
            &[citation],
            &mut decision,
        );

        assert_eq!(decision.status, "answerable");
        assert!(decision.next_tool_call.is_none());
    }

    #[test]
    fn llm_judge_is_required_for_all_text_rag_questions() {
        assert!(agent_judge::requires_llm_judge_for_answer(
            "这篇文章讲了什么？"
        ));
        assert!(agent_judge::requires_llm_judge_for_answer(
            "这个方法是什么？"
        ));
        assert!(agent_judge::requires_llm_judge_for_answer(
            "Table 3 里面提到的 SWE-Pruner 是什么指标？"
        ));
    }

    #[test]
    fn retrieval_answerability_requires_m4_llm_gate() {
        let local_run = run_with_finalize_gate(serde_json::json!({
            "status": "answerable",
            "reason": "local rule found citations",
            "runtime": "m3-rule-guard"
        }));
        assert!(!agent_judge::retrieval_is_answerable(&local_run));

        let llm_run = run_with_finalize_gate(serde_json::json!({
            "status": "answerable",
            "reason": "LLM judge accepted evidence",
            "runtime": "m4-llm-judge"
        }));
        assert!(agent_judge::retrieval_is_answerable(&llm_run));
    }

    fn dummy_citation(id: &str) -> runtime::rag::Citation {
        runtime::rag::Citation {
            id: id.to_string(),
            label: "[1]".to_string(),
            page: 1,
            block_id: id.to_string(),
            section_title: None,
            quote: "evidence".to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "fts".to_string(),
        }
    }

    #[test]
    fn dedup_citations_collapses_repeats_by_block_then_page_quote() {
        // Same block id (even with differing citation ids) → one entry.
        let a1 = dummy_citation("blk-a");
        let mut a2 = dummy_citation("blk-a");
        a2.id = "different-id".to_string();
        // Distinct block id → kept separately.
        let b = dummy_citation("blk-b");
        // Empty block id falls back to page+quote: same page+quote → collapsed.
        let mut q1 = dummy_citation("");
        q1.block_id = String::new();
        let mut q2 = dummy_citation("");
        q2.block_id = String::new();
        let out = dedup_citations(vec![a1, a2, b, q1, q2]);
        // blk-a (1) + blk-b (1) + page1/"evidence" (1) = 3.
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn best_effort_answers_when_not_answerable_but_evidence_accumulated() {
        // Judge ended needs_more_evidence but several citations were gathered: an
        // open-ended question should get a caveated answer, not a refusal.
        let mut run = run_with_finalize_gate(serde_json::json!({
            "status": "needs_more_evidence",
            "reason": "could not confirm a direct citation between the two papers",
            "runtime": "m4-llm-judge"
        }));
        run.retrieval_run.citations = vec![dummy_citation("a"), dummy_citation("b")];
        assert!(!agent_judge::retrieval_is_answerable(&run));
        assert!(agent_judge::should_answer_best_effort(&run));
    }

    #[test]
    fn best_effort_refuses_when_evidence_is_empty() {
        // No usable evidence at all → still refuse rather than fabricate.
        let mut run = run_with_finalize_gate(serde_json::json!({
            "status": "insufficient",
            "reason": "no evidence found",
            "runtime": "m4-llm-judge"
        }));
        run.retrieval_run.citations = vec![dummy_citation("a")]; // below the min of 2
        assert!(!agent_judge::should_answer_best_effort(&run));
    }

    #[test]
    fn best_effort_does_not_trigger_when_already_answerable() {
        let mut run = run_with_finalize_gate(serde_json::json!({
            "status": "answerable",
            "reason": "evidence supports the answer",
            "runtime": "m4-llm-judge"
        }));
        run.retrieval_run.citations = vec![dummy_citation("a"), dummy_citation("b")];
        // Already answerable → best-effort path is not needed (returns false).
        assert!(!agent_judge::should_answer_best_effort(&run));
    }

    #[test]
    fn judge_tool_counts_open_table_replacement_as_new_evidence() {
        let mut run = insufficient_run(false);
        run.retrieval_run.citations = vec![runtime::rag::Citation {
            id: "old".to_string(),
            label: "[1]".to_string(),
            page: 8,
            block_id: "table-3".to_string(),
            section_title: Some("Table 3".to_string()),
            quote: "Table 3 | SWE-Pruner | Tokens (M) = 0.670".to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "table_fact".to_string(),
        }];
        let output = runtime::rag::RagToolExecutionOutput {
            citations: vec![runtime::rag::Citation {
                id: "new".to_string(),
                label: "[1]".to_string(),
                page: 8,
                block_id: "table-3".to_string(),
                section_title: Some("Table 3".to_string()),
                quote: "Table 3 | SWE-Pruner | Tokens (M) = 0.670".to_string(),
                bbox_list: serde_json::json!([]),
                document_id: "doc".to_string(),
                source: "open_table".to_string(),
            }],
            trace_candidates: Vec::new(),
            tree_nodes: Vec::new(),
            tool_call: runtime::rag::RetrievalTraceToolCall {
                tool: "open_table".to_string(),
                status: "ok".to_string(),
                input: serde_json::json!({ "query": "Table 3 SWE-Pruner", "limit": 40 }),
                result_count: 1,
                error: None,
            },
        };

        let (gained_citations, _) = agent_judge::apply_judge_tool_output(&mut run, &output);

        assert_eq!(gained_citations, 1);
        assert_eq!(run.retrieval_run.citations.len(), 1);
        assert_eq!(run.retrieval_run.citations[0].source, "open_table");
    }

    #[test]
    fn llm_hard_gate_does_not_match_table_1_against_table_10() {
        let citation = runtime::rag::Citation {
            id: "c1".to_string(),
            label: "[1]".to_string(),
            page: 10,
            block_id: "table-10".to_string(),
            section_title: Some("Table 10 length.".to_string()),
            quote: "Table 10 length. | Pruner | Tokens = 806,220".to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc".to_string(),
            source: "open_table".to_string(),
        };
        let mut decision = llm::chat::LlmAnswerabilityDecision {
            status: "answerable".to_string(),
            reason: "enough".to_string(),
            missing: Vec::new(),
            next_tool_call: None,
        };

        agent_judge::enforce_llm_judge_hard_gate(
            "Table 1 的结果是什么？",
            &[citation],
            &mut decision,
        );

        assert_eq!(decision.status, "needs_more_evidence");
        assert!(decision.reason.contains("table 1"));
    }

    #[test]
    fn table_number_parser_handles_cjk_table_markers() {
        assert_eq!(
            agent_judge::requested_table_number("表3 的 SWE-Pruner 数值是什么？").as_deref(),
            Some("3")
        );
        assert_eq!(
            agent_judge::requested_table_number("第7表里面 GLM-5 的分数").as_deref(),
            Some("7")
        );
        assert_eq!(
            agent_judge::requested_table_number("表三的 SWE-Pruner 结果是什么？").as_deref(),
            Some("3")
        );
        assert_eq!(
            agent_judge::requested_table_number("第十二表里面的指标").as_deref(),
            Some("12")
        );
        assert!(agent_judge::question_needs_table_evidence("表 8 的结果"));
        assert!(agent_judge::question_needs_table_evidence("表三的结果"));
        assert!(!agent_judge::question_needs_table_evidence(
            "这个方法的表现如何？"
        ));
    }

    #[test]
    fn chat_history_load_filters_stale_index_versions() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE chat_turns (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL DEFAULT '',
              document_id TEXT NOT NULL,
              provider_id TEXT NOT NULL DEFAULT '',
              model_key TEXT NOT NULL DEFAULT '',
              provider_label TEXT NOT NULL DEFAULT '',
              user_message TEXT NOT NULL,
              assistant_answer TEXT NOT NULL,
              reasoning_content TEXT NOT NULL DEFAULT '',
              selected_text TEXT NOT NULL DEFAULT '',
              image_data_url TEXT NOT NULL DEFAULT '',
              citations_json TEXT NOT NULL DEFAULT '[]',
              claims_json TEXT NOT NULL DEFAULT '[]',
              retrieval_trace_json TEXT NOT NULL DEFAULT '{}',
              referenced_document_ids_json TEXT NOT NULL DEFAULT '[]',
              index_version INTEGER NOT NULL DEFAULT 0,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL
            );",
        )
        .expect("chat schema");
        conn.execute(
            "INSERT INTO chat_turns
               (id, session_id, document_id, user_message, assistant_answer, citations_json,
                claims_json, retrieval_trace_json, index_version, created_at, updated_at)
             VALUES
               ('stale', 'sess', 'doc', 'old question', 'old answer', '[]', '[]', '{}', ?1, 1, 1),
               ('fresh', 'sess', 'doc', 'new question', 'new answer', '[]', '[]', '{}', ?2, 2, 2)",
            params![CURRENT_INDEX_VERSION - 1, CURRENT_INDEX_VERSION],
        )
        .expect("chat rows");

        let turns = load_stored_chat_turns(&conn, "sess", 40).expect("chat turns");

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].id, "fresh");
        assert_eq!(turns[0].assistant_answer, "new answer");
    }

    #[test]
    fn local_agent_memory_keeps_full_answers_in_chronological_order() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE chat_turns (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL DEFAULT '',
              user_message TEXT NOT NULL,
              assistant_answer TEXT NOT NULL,
              created_at INTEGER NOT NULL
            );",
        )
        .expect("schema");
        // Two sessions; only 'sess' turns should appear, oldest-first.
        conn.execute_batch(
            "INSERT INTO chat_turns (id, session_id, user_message, assistant_answer, created_at) VALUES
               ('t1', 'sess',  'first question',  'first FULL answer that is well past any 260-char preview cap', 1),
               ('t2', 'sess',  'second question', 'second answer', 2),
               ('o1', 'other', 'other question',  'other answer', 5);",
        )
        .expect("rows");

        let memory = build_local_agent_memory(&conn, "sess");
        // Both turns present, full text, chronological (first before second).
        let first = memory.find("first question").expect("turn 1");
        let second = memory.find("second question").expect("turn 2");
        assert!(first < second, "oldest turn should come first");
        assert!(memory.contains("first FULL answer that is well past any 260-char preview cap"));
        // Other session is excluded.
        assert!(!memory.contains("other question"));

        assert!(build_local_agent_memory(&conn, "empty-session").is_empty());
    }

    #[test]
    fn cap_chars_truncates_with_ellipsis() {
        assert_eq!(cap_chars("hello", 10), "hello");
        assert_eq!(cap_chars("hello", 3), "hel…");
    }

    fn insufficient_run(budget_exhausted: bool) -> runtime::agent::AgentRunResult {
        run_with_finalize_gate(serde_json::json!({
            "status": "insufficient",
            "reason": "The question asks for a definition, but no definitional evidence is available.",
            "missing": ["definition"],
            "attempt": 19,
            "maxAttempts": 20,
            "budgetExhausted": budget_exhausted,
            "runtime": "test"
        }))
    }

    fn run_with_finalize_gate(finalize_gate: serde_json::Value) -> runtime::agent::AgentRunResult {
        let retrieval_trace = runtime::rag::RetrievalTrace {
            run_id: "test-run".to_string(),
            intent: "explain".to_string(),
            tree_nodes: Vec::new(),
            candidates: Vec::new(),
            tool_calls: Vec::new(),
            finalize_gate,
        };
        let trace = runtime::agent::AgentTrace::from_retrieval(
            retrieval_trace.clone(),
            &[],
            Vec::new(),
            None,
            None,
        );
        runtime::agent::AgentRunResult {
            retrieval_run: runtime::rag::RetrievalRun {
                id: "test-run".to_string(),
                intent: "explain".to_string(),
                prompt_context: String::new(),
                citations: Vec::new(),
                trace: retrieval_trace,
                context_budget: model_catalog::ModelContextBudget::default(),
            },
            trace,
            session_context: String::new(),
            attempts: runtime::agent::RetrievalAttempts::default(),
            ledger: runtime::agent::RetrievalLedger::new(),
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    model_catalog::warmup_model_catalog();
    tauri::Builder::default()
        .manage(PdfRegistry::default())
        .manage(pdf2zh_sidecar::Pdf2zhSidecarState::default())
        .manage(AgentSessionState::default())
        .manage(AskCancellations::default())
        .invoke_handler(tauri::generate_handler![
            choose_workspace,
            choose_pdf_files,
            open_path_in_file_manager,
            pdf_layout_dump::dump_pdf_layout,
            scan_workspace_pdfs,
            import_workspace_paths,
            create_note_source,
            update_note_source,
            load_note_source,
            load_note_links,
            clip_web_page,
            collections::load_collections,
            collections::create_collection,
            collections::rename_collection,
            collections::delete_collection,
            collections::move_document_to_collection,
            collections::move_collection,
            knowledge_api::load_knowledge_api_settings,
            knowledge_api::save_knowledge_api_settings,
            knowledge_api::rotate_knowledge_api_token,
            load_backup_settings,
            save_backup_settings,
            choose_backup_dir,
            create_backup_now,
            restore_backup,
            reveal_document_in_file_manager,
            choose_vault_dir,
            load_vault_settings,
            set_vault_dir,
            export_notes_to_vault,
            import_notes_from_vault,
            collections::reorder_documents,
            collections::reorder_collections,
            trending::fetch_trending_papers,
            trending::add_trending_paper,
            runtime::precipitation::enqueue_document_knowledge,
            runtime::precipitation::get_document_knowledge,
            runtime::precipitation::reprecipitate_document,
            runtime::precipitation::consolidate_knowledge,
            runtime::knowledge_graph::get_knowledge_graph,
            runtime::knowledge_graph::get_related_documents,
            runtime::knowledge_graph::get_turn_recommendations,
            update_check::check_for_update,
            open_external_url,
            local_agent::get_local_agent_status,
            test_local_agent_connection,
            export_markdown_file,
            load_last_workspace,
            read_pdf_bytes,
            read_document_bytes,
            pdf_annotation::save_pdf_document,
            pdf_annotation::save_pdf_document_as,
            pdf_annotation::save_pdf_as,
            pdf_annotation::save_pdf_at_path,
            pdf_annotation::read_saved_pdf_bytes,
            pdf2zh_sidecar::read_pdf_artifact_bytes,
            pdf2zh_sidecar::probe_pdf_translation_runtime,
            pdf2zh_sidecar::start_pdf_translation,
            pdf2zh_sidecar::cancel_pdf_translation,
            pdf2zh_sidecar::request_pdf_translation_pages,
            pdf2zh_sidecar::clear_pdf_translation_cache,
            document_index::upsert_document_index,
            document_index::repair_document_index_from_cache,
            document_index::enqueue_document_reindex,
            visual_index::run_document_visual_index,
            visual_index::enqueue_document_visual_index,
            mark_document_stale,
            search_document_chunks,
            translate_text,
            document_translation::start_document_translation,
            document_translation::get_page_translation,
            document_translation::cancel_translation_job,
            load_chat_turns,
            clear_chat_turns,
            list_chat_sessions,
            create_chat_session,
            rename_chat_session,
            update_chat_session_focus,
            delete_chat_session,
            generate_session_title,
            load_notes,
            create_note,
            update_note,
            delete_note,
            ask_document,
            ask_document_stream,
            stop_ask_document,
            load_translation_settings,
            save_translation_settings,
            load_web_search_settings,
            save_web_search_settings,
            load_proxy_settings,
            save_proxy_settings,
            remove_workspace_root,
            delete_document,
            clear_unfiled_documents,
            list_model_providers,
            save_model_provider,
            delete_model_provider,
            fetch_provider_models,
            test_model_provider,
            test_translation_provider,
            update_document_reading_state
        ])
        .setup(|app| {
            // Tell the model loader where the bundled `.models/` lives. In a
            // packaged app the resources sit under the Tauri resource dir; a
            // `../.models/...` bundle entry is flattened to `<resource>/_up_/.models`.
            // We register whichever candidate actually contains `.models` so the
            // shared path-resolution logic finds TSR/OCR models in dev and prod.
            if let Ok(resource_dir) = app.path().resource_dir() {
                for candidate in [resource_dir.join("_up_"), resource_dir.clone()] {
                    if candidate.join(".models").is_dir() {
                        vision::register_resource_models_dir(candidate);
                        break;
                    }
                }
            }

            let app_data_dir = app
                .path()
                .app_data_dir()
                .map_err(|err| format!("Failed to resolve app data dir: {err}"))?;
            fs::create_dir_all(&app_data_dir)
                .map_err(|err| format!("Failed to create app data dir: {err}"))?;
            let db_path = app_data_dir.join("lumenfolio.sqlite");
            // A restore staged in a previous session is swapped in here — before
            // any connection exists, since replacing an open database is exactly
            // the corruption this feature is meant to protect against.
            backup::apply_staged_restore(&db_path);
            let conn = storage::open_database(&db_path)?;
            app.manage(DatabasePath(db_path.clone()));
            // Seed the outbound proxy from settings before any HTTP client is
            // built (GUI apps don't inherit the shell's HTTPS_PROXY).
            if let Ok(Some(proxy)) = load_app_setting(&conn, "proxy_url") {
                net::set_proxy(Some(proxy));
            }
            app.manage(AppDatabase {
                conn: Mutex::new(conn),
            });
            app.manage(knowledge_api::KnowledgeApiState::default());
            // Scheduled snapshot, if one is due. Off the startup path: it takes
            // the database lock, and on a large library VACUUM INTO is long
            // enough to be felt as a slow launch.
            {
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    maybe_auto_backup(&handle.state::<AppDatabase>());
                });
            }
            // Bring the knowledge API up if the user left it on. Off the startup
            // path too: a port taken by another process must not stop the app.
            {
                let handle = app.handle().clone();
                let api_db_path = db_path.clone();
                tauri::async_runtime::spawn(async move {
                    knowledge_api::start_if_enabled(
                        api_db_path,
                        &handle.state::<AppDatabase>(),
                        &handle.state::<knowledge_api::KnowledgeApiState>(),
                    )
                    .await;
                });
            }

            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .build(),
            )?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
