//! Unified native tool-calling agent loop (P1 of the unified-loop refactor).
//!
//! This is the modern, codex/opencode-style replacement for the fragmented
//! "M4 judge → separate answer generation" pipeline: ONE growing message history
//! that the model always sees in full. The model decides which retrieval tools to
//! call (via the provider's native `tools`/`tool_calls`), the tool results are
//! appended back into the SAME history, and the loop ends when the model stops
//! requesting tools and writes the answer. Because retrieval and answering share
//! one context, the model has true global awareness of everything it explored.
//!
//! Design choice (deliberate, to avoid the biggest pit): the tool-DECISION rounds
//! are NON-streaming (robust JSON parsing of `tool_calls`), while the FINAL answer
//! reuses the proven streaming reader (`read_openai_answer_stream`, with its
//! ThinkSplitter + SSE handling) so the UI still streams tokens. Both phases send
//! the identical full message history.
//!
//! Routing is gated by the model's `tool_call` capability (models.dev profile).
//! Models without native tool-calling fall back to the existing M3+M4 path, so
//! weak/local models keep working unchanged.

use std::time::Duration;

use tauri::Emitter;

use crate::runtime::agent::{tool_call_event, tool_result_event};
use crate::{
    llm, normalize_base_url, optional_non_empty, runtime, truncate_for_error,
    AgentActivityEventOutput, AnswerDeltaEventOutput, AskAnswerResult, AskDocumentInput,
    OpenAiCompatibleProvider,
};

/// Per tool-round HTTP timeout. Generous: a round may run an expensive retrieval
/// plan server-side before the provider responds.
const UNIFIED_TOOL_ROUND_TIMEOUT_SECS: u64 = 90;
/// Default upper bound on tool-calling rounds before we force a final answer.
const DEFAULT_MAX_TOOL_ROUNDS: u32 = 8;
/// Hard clamp so a stray `max_retrieval_steps` can't run the loop forever.
const MAX_TOOL_ROUNDS_CLAMP: u32 = 12;
/// Rough chars-per-token for the context-budget estimate. Intentionally low (so
/// it OVER-estimates tokens) — we'd rather compact a little early than overflow.
const CHARS_PER_TOKEN: usize = 3;
/// The most recent messages are never pruned — keeps the last couple of tool
/// rounds (and the question) intact so the model still has fresh evidence.
const UNIFIED_KEEP_RECENT_MESSAGES: usize = 8;
/// Floor for the tool-history char budget, so an odd/small model budget never
/// prunes almost everything.
const MIN_HISTORY_CHARS: usize = 6_000;
/// Placeholder swapped in for an elided old tool result.
const ELIDED_TOOL_RESULT: &str = "[earlier tool result elided to fit the context window]";

pub(crate) struct UnifiedLoopInput<'a> {
    pub(crate) input: &'a AskDocumentInput,
    pub(crate) database: &'a crate::AppDatabase,
    pub(crate) app: &'a tauri::AppHandle,
    pub(crate) question: &'a str,
    pub(crate) document_id: &'a str,
    /// Dispatch whitelist for cross-document `documentId` tool routing.
    pub(crate) visible_document_ids: &'a [&'a str],
    /// Pre-rendered workspace manifest (titles/dirs/abstracts) — the model's
    /// "library view", injected so it can answer library questions and route
    /// cross-document tool calls. For a large library this is the COMPACT block
    /// (focus + @-referenced only); the rest are discovered via tools.
    pub(crate) workspace_manifest: &'a str,
    /// Large library: expose `search_library`/`list_documents` discovery tools
    /// instead of relying on a fully-inlined manifest (progressive disclosure).
    pub(crate) library_is_large: bool,
    /// Session memory block (recent turns / selection), already rendered.
    pub(crate) session_context: &'a str,
    pub(crate) provider: &'a OpenAiCompatibleProvider,
    pub(crate) activity_event_id: Option<&'a str>,
}

/// Whether to route this turn through the unified loop. Requires a model that the
/// catalog marks as supporting native tool-calling and a non-image question
/// (image questions keep the existing vision path).
pub(crate) fn should_use_unified_loop(
    provider: &OpenAiCompatibleProvider,
    input: &AskDocumentInput,
) -> bool {
    if provider.model_profile.tool_call != Some(true) {
        return false;
    }
    let has_image = input
        .image_data_url
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    !has_image
}

// ---- Non-streaming tool-round response shapes -----------------------------

/// A native tool call ready for the loop to execute — reassembled from the
/// streamed fragments (id/name/arguments).
struct ToolCallEntry {
    id: String,
    function: ToolCallFunction,
}

struct ToolCallFunction {
    name: String,
    arguments: String,
}

pub(crate) async fn run_unified_agent_loop(
    ctx: UnifiedLoopInput<'_>,
    agent_run: &mut runtime::agent::AgentRunResult,
) -> Result<AskAnswerResult, String> {
    let vision_enabled = ctx
        .provider
        .capabilities
        .iter()
        .any(|capability| capability == "vision");
    let web_enabled = ctx.input.web_enabled.unwrap_or(false);
    let rag_capabilities = runtime::rag::RagToolCapabilities {
        vision_enabled,
        web_enabled,
        max_quote_chars: agent_run.retrieval_run.context_budget.max_quote_chars,
    };
    let tools = build_openai_tools(vision_enabled, web_enabled, ctx.library_is_large);
    let mut messages = build_initial_messages(&ctx, agent_run);

    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(UNIFIED_TOOL_ROUND_TIMEOUT_SECS))
        .build()
        .map_err(|err| format!("Failed to create agent loop client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&ctx.provider.base_url)
    );
    let max_rounds = ctx
        .input
        .max_retrieval_steps
        .unwrap_or(DEFAULT_MAX_TOOL_ROUNDS)
        .clamp(1, MAX_TOOL_ROUNDS_CLAMP);
    // Keep the accumulating tool history within the model's context window — a
    // long tool-calling turn would otherwise overflow a small-context model.
    let char_budget = history_char_budget(&agent_run.retrieval_run.context_budget);

    for round in 0..max_rounds {
        compact_history(
            ctx.app,
            ctx.activity_event_id,
            agent_run,
            &mut messages,
            char_budget,
        );
        let request = serde_json::json!({
            "model": ctx.provider.model,
            "messages": messages,
            "temperature": 0.1,
            // Stream the round: answer/reasoning content paints live (true token
            // streaming) while native tool calls are reassembled from the deltas.
            "stream": true,
            "tools": tools,
            "tool_choice": "auto",
        });
        let round_result = stream_tool_round(
            &client,
            &endpoint,
            ctx.provider,
            &request,
            ctx.app,
            ctx.activity_event_id,
        )
        .await?;
        let content_str = round_result.content.clone();
        let reasoning_str = round_result.reasoning.clone();
        // Reassemble streamed tool-call fragments into the loop's tool-call shape,
        // synthesizing a stable id when the server omitted one and dropping any
        // empty slot (a stray index that never received a name).
        let tool_calls: Vec<ToolCallEntry> = round_result
            .tool_calls
            .into_iter()
            .enumerate()
            .filter(|(_, call)| !call.name.trim().is_empty())
            .map(|(index, call)| ToolCallEntry {
                id: if call.id.trim().is_empty() {
                    format!("call_{round}_{index}")
                } else {
                    call.id
                },
                function: ToolCallFunction {
                    name: call.name,
                    arguments: call.arguments,
                },
            })
            .collect();

        if !tool_calls.is_empty() {
            // --- OpenAI-native structured tool calls ---
            log::info!(
                "unified_loop round={} native tool call(s)={}",
                round + 1,
                tool_calls.len()
            );
            messages.push(assistant_tool_call_message(
                &serde_json::Value::String(content_str.clone()),
                &tool_calls,
            ));
            for call in &tool_calls {
                let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|err| {
                        log::warn!(
                            "unified_loop tool={} had unparseable arguments ({err}); using empty args: {}",
                            call.function.name,
                            truncate_for_error(&call.function.arguments, 200)
                        );
                        serde_json::json!({})
                    });
                let rendered = run_one_tool(
                    &ctx,
                    agent_run,
                    rag_capabilities,
                    &call.function.name,
                    &args,
                )?;
                messages.push(tool_result_message(&call.id, &rendered));
            }
            continue;
        }

        // --- DSML text tool calls (e.g. DeepSeek V4) — COMPATIBILITY FALLBACK ---
        // Reached ONLY when there were no native `tool_calls` above, so normal
        // tool-calling models are never affected by this branch. Some endpoints
        // (a passthrough proxy, or a server without DSML tool parsing) return a
        // DeepSeek tool call as TEXT in `content` (`<｜DSML｜tool_calls>…`) instead of
        // structured tool_calls. The canonical fix is server-side parsing — the
        // official DeepSeek API, or vLLM/SGLang with `--tool-call-parser deepseek_v4`,
        // expose proper `tool_calls` and take the native path above. This block is
        // a client-side shim for non-conformant endpoints: parse the DSML so the
        // agent can still explore, then feed results back as
        // `<tool_result>…</tool_result>` in a user message (the DSML protocol).
        let dsml_calls = parse_dsml_tool_calls(&content_str);
        if dsml_calls.is_empty() {
            // No tool call of either kind — this response IS the model's answer.
            // Use it directly: no separate re-generation step (which is what made
            // an interleaved model emit a trailing, un-runnable tool call). The
            // loop is closed — the model either keeps calling tools above or
            // answers here.
            return build_unified_answer(
                &ctx,
                agent_run,
                &content_str,
                &reasoning_str,
                round_result.emitted_answer,
            );
        }
        log::info!(
            "unified_loop round={} DSML tool call(s)={}",
            round + 1,
            dsml_calls.len()
        );
        messages.push(serde_json::json!({ "role": "assistant", "content": content_str }));
        let mut tool_results = String::new();
        for call in &dsml_calls {
            let rendered = run_one_tool(&ctx, agent_run, rag_capabilities, &call.name, &call.args)?;
            tool_results.push_str(&format!("<tool_result>{rendered}</tool_result>\n"));
        }
        messages.push(serde_json::json!({ "role": "user", "content": tool_results }));
    }

    // Round cap reached while the model still wanted tools — force a final answer
    // (no tools offered) so it commits to prose instead of looping forever.
    compact_history(
        ctx.app,
        ctx.activity_event_id,
        agent_run,
        &mut messages,
        char_budget,
    );
    let (content, reasoning, emitted_answer) =
        force_final_answer(&ctx, &client, &endpoint, &messages).await?;
    build_unified_answer(&ctx, agent_run, &content, &reasoning, emitted_answer)
}

/// A tool call parsed out of DeepSeek's DSML text format.
struct DsmlCall {
    name: String,
    args: serde_json::Value,
}

/// Parse DSML tool calls that a model emitted as plain text in `content`
/// (DeepSeek V4 "Tool Calling (DSML Format)"). Delimiters use ASCII `|` or
/// full-width `｜` (U+FF5C), sometimes doubled, e.g. `<｜DSML｜invoke name="…">`.
/// A `string="true"` parameter is a raw string; `string="false"` is JSON.
///
/// Fallback only: this exists for endpoints that don't parse DSML server-side
/// (the official DeepSeek API and vLLM/SGLang with the deepseek_v4 tool parser
/// already expose structured `tool_calls`). It is called only when native
/// `tool_calls` were absent, and returns an empty vec for ordinary prose — so it
/// never interferes with other models' normal tool calling.
fn parse_dsml_tool_calls(content: &str) -> Vec<DsmlCall> {
    use regex::Regex;
    use std::sync::OnceLock;
    static INVOKE_RE: OnceLock<Option<Regex>> = OnceLock::new();
    static PARAM_RE: OnceLock<Option<Regex>> = OnceLock::new();
    let invoke_re = INVOKE_RE
        .get_or_init(|| {
            Regex::new(
                r#"(?s)<[|\x{ff5c}]+[^|\x{ff5c}>]*[|\x{ff5c}]+invoke\s+name="([^"]+)"\s*>(.*?)</[|\x{ff5c}]+[^|\x{ff5c}>]*[|\x{ff5c}]+invoke>"#,
            )
            .ok()
        })
        .as_ref();
    let param_re = PARAM_RE
        .get_or_init(|| {
            Regex::new(
                r#"(?s)<[|\x{ff5c}]+[^|\x{ff5c}>]*[|\x{ff5c}]+parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</[|\x{ff5c}]+[^|\x{ff5c}>]*[|\x{ff5c}]+parameter>"#,
            )
            .ok()
        })
        .as_ref();
    let (Some(invoke_re), Some(param_re)) = (invoke_re, param_re) else {
        return Vec::new();
    };
    let mut calls = Vec::new();
    for invoke in invoke_re.captures_iter(content) {
        let name = invoke[1].trim().to_string();
        if name.is_empty() {
            continue;
        }
        let mut args = serde_json::Map::new();
        for param in param_re.captures_iter(&invoke[2]) {
            let key = param[1].trim().to_string();
            let raw = param[3].trim();
            let value = if &param[2] == "true" {
                serde_json::Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw)
                    .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
            };
            args.insert(key, value);
        }
        calls.push(DsmlCall {
            name,
            args: serde_json::Value::Object(args),
        });
    }
    calls
}

/// Execute a single tool call (library-discovery or RAG), emit its trace events,
/// merge its citations into the run, and return the rendered result text. Shared
/// by the native-tool-calls and DSML-tool-calls branches.
fn run_one_tool(
    ctx: &UnifiedLoopInput<'_>,
    agent_run: &mut runtime::agent::AgentRunResult,
    rag_capabilities: runtime::rag::RagToolCapabilities,
    name: &str,
    args: &serde_json::Value,
) -> Result<String, String> {
    // Default list_trending_papers to the period the user is actually viewing.
    // The model is also hinted in the prompt, but it may omit the arg — the tool
    // would then fall back to 'daily' and answer about the wrong list.
    let injected_args;
    let args = if name == "list_trending_papers"
        && args.get("period").and_then(|v| v.as_str()).is_none()
    {
        match ctx
            .input
            .view_context
            .as_ref()
            .and_then(|v| v.trending_period.as_deref())
            .filter(|p| matches!(*p, "daily" | "weekly" | "monthly"))
        {
            Some(period) => {
                let mut obj = args.as_object().cloned().unwrap_or_default();
                obj.insert(
                    "period".to_string(),
                    serde_json::Value::String(period.to_string()),
                );
                injected_args = serde_json::Value::Object(obj);
                &injected_args
            }
            None => args,
        }
    } else {
        args
    };

    let fallback_query = args
        .get("query")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| ctx.question.to_string());

    let start_event = tool_call_event(
        name.to_string(),
        args.clone(),
        format!("Calling {name}"),
        String::new(),
        format!(
            "unified_loop tool={name} args={}",
            truncate_for_error(&args.to_string(), 200)
        ),
    );
    emit_activity(ctx.app, ctx.activity_event_id, start_event.clone());
    agent_run.trace.events.push(start_event);

    // Workspace-level discovery tools (large library): list/search docs. They
    // produce no citations — the model uses the returned ids to route a normal
    // retrieval tool to a discovered document.
    if is_library_tool(name) {
        let (rendered, count) = {
            let conn = ctx
                .database
                .conn
                .lock()
                .map_err(|_| "SQLite lock was poisoned".to_string())?;
            execute_library_tool(&conn, name, args)
        };
        let result_event = runtime::agent::AgentTraceEvent::new(
            "tool_result",
            name.to_string(),
            "completed",
            format!("{name} result"),
            format!("{name} returned {count} documents"),
            format!("unified_loop library tool={name} results={count}"),
        )
        .with_tool(name.to_string(), args.clone());
        emit_activity(ctx.app, ctx.activity_event_id, result_event.clone());
        agent_run.trace.events.push(result_event);
        return Ok(rendered);
    }

    let output = {
        let conn = ctx
            .database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        runtime::rag::execute_rag_tool_call_for_capabilities(
            &conn,
            ctx.document_id,
            ctx.visible_document_ids,
            name,
            args,
            &fallback_query,
            rag_capabilities,
        )
    };
    let result_event = tool_result_event(
        &output,
        output.tool_call.tool.clone(),
        format!(
            "unified_loop tool={} results={}",
            output.tool_call.tool, output.tool_call.result_count
        ),
    );
    emit_activity(ctx.app, ctx.activity_event_id, result_event.clone());
    agent_run.trace.events.push(result_event);
    let rendered = render_tool_output(name, &output);
    // Merge the gained citations into the shared run (budget-aware merge +
    // coverage + trace sync) — the same accounting the M4 loop uses.
    crate::agent_judge::apply_judge_tool_output(agent_run, &output);
    Ok(rendered)
}

/// Build the OpenAI `tools` array from the RAG tool specs, plus the library
/// discovery tools when the workspace is large (progressive disclosure).
fn build_openai_tools(
    vision_enabled: bool,
    web_enabled: bool,
    library_is_large: bool,
) -> Vec<serde_json::Value> {
    let mut tools: Vec<serde_json::Value> =
        runtime::rag::rag_tool_specs_for_capabilities(vision_enabled, web_enabled)
            .into_iter()
            // The unified loop runs the SYNCHRONOUS RAG executor, which only stubs
            // analyze_visual/analyze_page (they need the async multimodal provider
            // path). Don't advertise tools we can't fulfil — the model would call
            // them and silently get a text-search fallback instead of vision.
            .filter(|spec| !matches!(spec.name, "analyze_visual" | "analyze_page"))
            .map(|spec| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    }
                })
            })
            .collect();
    if library_is_large {
        tools.extend(library_discovery_tool_specs());
    }
    tools
}

/// Conservative char budget for the accumulating tool history, derived from the
/// model's context window: reserve the output budget, then keep ~70% headroom for
/// the system/question/answer. Floored so a tiny budget never prunes everything.
fn history_char_budget(budget: &crate::model_catalog::ModelContextBudget) -> usize {
    let history_tokens = budget
        .model_context_tokens
        .saturating_sub(budget.model_output_tokens)
        .saturating_mul(7)
        / 10;
    history_tokens
        .saturating_mul(CHARS_PER_TOKEN)
        .max(MIN_HISTORY_CHARS)
}

/// Rough character size of the message history (content + tool-call args, with a
/// small per-message overhead). Only used to decide WHEN to compact.
fn estimate_message_chars(messages: &[serde_json::Value]) -> usize {
    messages
        .iter()
        .map(|message| {
            let mut chars = message
                .get("content")
                .and_then(|value| value.as_str())
                .map(str::len)
                .unwrap_or(0);
            if let Some(calls) = message.get("tool_calls").and_then(|value| value.as_array()) {
                for call in calls {
                    let function = call.get("function");
                    chars += function
                        .and_then(|f| f.get("arguments"))
                        .and_then(|value| value.as_str())
                        .map(str::len)
                        .unwrap_or(0);
                    chars += function
                        .and_then(|f| f.get("name"))
                        .and_then(|value| value.as_str())
                        .map(str::len)
                        .unwrap_or(0);
                }
            }
            chars + 16
        })
        .sum()
}

/// Prune the OLDEST tool-result messages — replacing their content with a short
/// placeholder, but keeping the assistant/tool message structure intact so each
/// tool result still pairs with its `tool_call_id` — until the history fits
/// `char_budget`. The most recent `UNIFIED_KEEP_RECENT_MESSAGES` messages are
/// never touched (fresh evidence stays whole). Returns how many were elided.
fn compact_messages_to_budget(messages: &mut [serde_json::Value], char_budget: usize) -> usize {
    if estimate_message_chars(messages) <= char_budget {
        return 0;
    }
    let keep_from = messages.len().saturating_sub(UNIFIED_KEEP_RECENT_MESSAGES);
    let mut pruned = 0;
    for index in 0..keep_from {
        if estimate_message_chars(messages) <= char_budget {
            break;
        }
        let is_tool = messages[index].get("role").and_then(|value| value.as_str()) == Some("tool");
        if !is_tool {
            continue;
        }
        let already_elided = messages[index]
            .get("content")
            .and_then(|value| value.as_str())
            == Some(ELIDED_TOOL_RESULT);
        if already_elided {
            continue;
        }
        messages[index]["content"] = serde_json::Value::String(ELIDED_TOOL_RESULT.to_string());
        pruned += 1;
    }
    pruned
}

/// Compact `messages` to fit `char_budget` and, if anything was elided, record a
/// trace event so the compaction is visible in the Debug trace.
fn compact_history(
    app: &tauri::AppHandle,
    activity_event_id: Option<&str>,
    agent_run: &mut runtime::agent::AgentRunResult,
    messages: &mut [serde_json::Value],
    char_budget: usize,
) {
    let pruned = compact_messages_to_budget(messages, char_budget);
    if pruned == 0 {
        return;
    }
    log::info!("unified_loop compacted {pruned} old tool result(s) to fit the context window");
    let event = runtime::agent::AgentTraceEvent::new(
        "context_compacted",
        "generate_answer",
        "completed",
        "Compacted context",
        format!("Elided {pruned} older tool result(s) to fit the context window"),
        format!("unified_loop compacted {pruned} old tool results"),
    );
    emit_activity(app, activity_event_id, event.clone());
    agent_run.trace.events.push(event);
}

/// The two workspace-discovery tools for a large library. They return a ranked
/// list of documents (id/title/dir/abstract) — NOT evidence; the model then
/// routes a normal retrieval tool to a discovered doc via its `documentId`.
fn library_discovery_tool_specs() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "search_library",
                "description": "Search the user's whole document library by topic to find which documents are relevant. Returns matching documents (id, title, folder, abstract). Use a document's id as the `documentId` argument on a retrieval tool to then read it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "What to look for across the library (topic, title, keywords)." },
                        "limit": { "type": "integer", "description": "Max documents to return (default 10).", "minimum": 1, "maximum": 30 }
                    },
                    "required": ["query"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_documents",
                "description": "Browse the user's document library (most recently opened first). Returns documents (id, title, folder, abstract). Use when you need an overview of what is in the library rather than a topical search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": { "type": "integer", "description": "Max documents to return (default 15).", "minimum": 1, "maximum": 30 }
                    }
                }
            }
        }),
    ]
}

/// Whether a tool name is one of the workspace-level discovery tools (handled
/// directly in the loop rather than the per-document RAG executor).
fn is_library_tool(name: &str) -> bool {
    matches!(name, "search_library" | "list_documents")
}

/// Execute a library discovery tool: rank/list workspace documents and render
/// the result as text for the model. Returns (rendered_text, hit_count).
fn execute_library_tool(
    conn: &rusqlite::Connection,
    tool: &str,
    args: &serde_json::Value,
) -> (String, usize) {
    let default_limit = if tool == "list_documents" { 15 } else { 10 };
    let limit = args
        .get("limit")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or(default_limit)
        .clamp(1, 30);
    let query = args
        .get("query")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .unwrap_or("");
    let hits =
        runtime::rag::search_workspace_documents(conn, query, limit, &[]).unwrap_or_default();
    if hits.is_empty() {
        return (format!("{tool} found no documents in the library."), 0);
    }
    let mut rendered = format!("{tool} found {} document(s):\n", hits.len());
    for hit in &hits {
        let dir = if hit.rel_dir.is_empty() {
            String::new()
        } else {
            format!("{}/ ", hit.rel_dir)
        };
        rendered.push_str(&format!(
            "- [{}] {}\"{}\" ({}p)\n",
            hit.document_id, dir, hit.title, hit.page_count
        ));
        if !hit.summary.is_empty() {
            rendered.push_str(&format!("    Abstract: {}\n", hit.summary));
        }
    }
    (rendered, hits.len())
}

/// Ambient one-liner telling the model which app surface the user is on, so
/// "the current trending papers" resolves to the right cached list and so the
/// model knows the question may not be about the focus PDF.
fn current_surface_hint(view: Option<&crate::ViewContextInput>) -> Option<String> {
    let view = view?;
    match view.surface.as_deref() {
        Some("trending") => {
            let period = match view.trending_period.as_deref() {
                Some("weekly") => "Weekly",
                Some("monthly") => "Monthly",
                _ => "Daily",
            };
            Some(format!(
                "The user is currently browsing the {period} Trending Papers list (Hugging Face). \
For questions about \"the trending papers\", call list_trending_papers with period=\"{}\".",
                view.trending_period.as_deref().unwrap_or("daily")
            ))
        }
        Some("graph") => Some(
            "The user is currently viewing the cross-document Knowledge Graph of their library."
                .to_string(),
        ),
        _ => None,
    }
}

fn build_initial_messages(
    ctx: &UnifiedLoopInput<'_>,
    agent_run: &runtime::agent::AgentRunResult,
) -> Vec<serde_json::Value> {
    let answer_language =
        llm::chat::answer_language_for_question(ctx.question, ctx.input.locale.as_deref());
    let system = format!(
        "You are Lumenfolio, a careful academic PDF reading assistant. Answer in {answer_language}. \
You can call retrieval tools to read the user's PDFs (search passages, open sections, open tables and pages, inspect the structure, recall prior chat). \
Call the tools you need to gather evidence, then write the answer. Use only evidence you retrieved or that is already provided below — do not invent facts. \
When the question is about the user's document library/workspace itself — which documents or papers they have, what is in the sidebar/list, which of their papers is about a topic — the 'Workspace documents' list below is authoritative: answer directly from it (list the relevant titles), no retrieval is needed. Use search_library_knowledge to find library papers by topic across ALL documents, and list_trending_papers for questions about the Hugging Face trending list the user is browsing. \
Prefer the focus document; only pass another document's id as the `documentId` tool argument when the question genuinely needs cross-document evidence, and only use an id listed in 'Workspace documents'. \
When you have enough evidence, stop calling tools and reply with a structured Markdown answer (a short direct answer first, then concise paragraphs or lists). Do not return JSON. The final answer must be plain Markdown prose only — never write tool-call syntax, function calls, or any `<|...|>` markup in the answer itself. If the evidence is insufficient, say so clearly and state what is missing."
    );

    let mut context = String::new();
    if !ctx.session_context.trim().is_empty() {
        context.push_str("Conversation memory:\n");
        context.push_str(ctx.session_context.trim_end());
        context.push_str("\n\n");
    }
    if !ctx.workspace_manifest.trim().is_empty() {
        context.push_str(ctx.workspace_manifest.trim_end());
        context.push_str("\n\n");
    }
    let seed_evidence = agent_run.retrieval_run.prompt_context.trim();
    if !seed_evidence.is_empty() {
        context
            .push_str("Initial evidence already gathered (you may rely on it or retrieve more):\n");
        context.push_str(seed_evidence);
        context.push_str("\n\n");
    }
    if let Some(page) = ctx.input.page {
        context.push_str(&format!("The user is currently viewing page {page}.\n\n"));
    }
    if let Some(hint) = current_surface_hint(ctx.input.view_context.as_ref()) {
        context.push_str(&hint);
        context.push_str("\n\n");
    }
    context.push_str("Question:\n");
    context.push_str(ctx.question);

    vec![
        serde_json::json!({ "role": "system", "content": system }),
        serde_json::json!({ "role": "user", "content": context }),
    ]
}

fn assistant_tool_call_message(
    content: &serde_json::Value,
    tool_calls: &[ToolCallEntry],
) -> serde_json::Value {
    let calls: Vec<serde_json::Value> = tool_calls
        .iter()
        .map(|call| {
            serde_json::json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            })
        })
        .collect();
    // Preserve any assistant preface text; null when empty (OpenAI tool-call shape).
    let content_value = match content {
        serde_json::Value::String(text) if !text.trim().is_empty() => {
            serde_json::Value::String(text.clone())
        }
        _ => serde_json::Value::Null,
    };
    serde_json::json!({
        "role": "assistant",
        "content": content_value,
        "tool_calls": calls,
    })
}

fn tool_result_message(tool_call_id: &str, rendered: &str) -> serde_json::Value {
    serde_json::json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": rendered,
    })
}

/// Render a tool execution result into text the model reads as the tool message.
fn render_tool_output(tool: &str, output: &runtime::rag::RagToolExecutionOutput) -> String {
    if output.tool_call.status == "error" {
        if let Some(error) = &output.tool_call.error {
            return format!("Tool {tool} failed: {error}");
        }
    }
    if output.citations.is_empty() {
        if !output.tree_nodes.is_empty() {
            let nodes = output
                .tree_nodes
                .iter()
                .take(20)
                .map(|node| format!("- {} (p.{}) [id: {}]", node.title, node.page, node.id))
                .collect::<Vec<_>>()
                .join("\n");
            return format!(
                "Tool {tool} found these sections (open one for its passages):\n{nodes}"
            );
        }
        return format!("Tool {tool} returned no new evidence.");
    }
    let mut rendered = format!(
        "Tool {tool} returned {} evidence snippet(s):\n",
        output.citations.len()
    );
    for (index, citation) in output.citations.iter().enumerate().take(40) {
        let section = citation
            .section_title
            .as_deref()
            .map(|title| format!(" · {title}"))
            .unwrap_or_default();
        rendered.push_str(&format!(
            "[{}] (p.{}{}) {}\n",
            index + 1,
            citation.page,
            section,
            truncate_for_error(citation.quote.trim(), 600)
        ));
    }
    rendered
}

/// Send one streaming round: POST the request, then read the SSE stream — content
/// paints live, native tool calls are reassembled, and a stop cancels the read.
async fn stream_tool_round(
    client: &reqwest::Client,
    endpoint: &str,
    provider: &OpenAiCompatibleProvider,
    request: &serde_json::Value,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<llm::openai_stream::UnifiedStreamRound, String> {
    let mut builder = client.post(endpoint).json(request);
    if let Some(api_key) = &provider.api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = builder
        .send()
        .await
        .map_err(|err| format!("Agent loop request failed: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Agent loop provider returned {status}: {}",
            truncate_for_error(&body, 600)
        ));
    }
    llm::openai_stream::read_unified_tool_round_stream(response, app, answer_event_id).await
}

/// Error sentinel for a user-cancelled generation; the frontend ignores the
/// resulting done/error event for a stopped turn, so the text is irrelevant.
pub(crate) const GENERATION_STOPPED: &str = "Generation stopped by user.";

/// Final answer: stream from the SAME full message history (tool results included)
/// with tools disabled so the model writes prose instead of calling more tools.
/// Build the final answer from the model's TERMINAL response — the round in which
/// it stopped requesting tools. There is no second "answer generation" call, so an
/// interleaved model can't emit a trailing, un-runnable tool call here; the loop
/// is genuinely closed (keep calling tools, or answer). Any stray tool-call markup
/// is stripped as a safety net.
fn build_unified_answer(
    ctx: &UnifiedLoopInput<'_>,
    agent_run: &mut runtime::agent::AgentRunResult,
    content: &str,
    reasoning: &str,
    already_streamed: bool,
) -> Result<AskAnswerResult, String> {
    if let Some(event_id) = ctx.activity_event_id {
        let _ = ctx.app.emit(
            "lumenfolio://agent-activity",
            AgentActivityEventOutput {
                event_id: event_id.to_string(),
                event: runtime::agent::AgentTraceEvent::new(
                    "answer_start",
                    "generate_answer",
                    "running",
                    "Generating answer",
                    "Composing the answer from the gathered evidence",
                    "unified_loop building final answer",
                ),
            },
        );
    }

    let answer = llm::openai_stream::strip_tool_call_markup(content);
    if answer.is_empty() {
        return Err("Agent loop produced an empty answer".to_string());
    }
    let reasoning = llm::openai_stream::strip_tool_call_markup(reasoning);

    // Answerable gate — a first-class verdict alongside the M4 judge
    // (see FinalizeRuntime::is_verdict).
    let gate = serde_json::json!({
        "status": runtime::agent::FinalizeStatus::Answerable.as_str(),
        "reason": "Answered via the unified tool-calling agent loop.",
        "missing": serde_json::json!([]),
        "nextToolCall": serde_json::Value::Null,
        "citationCount": agent_run.retrieval_run.citations.len(),
        "runtime": runtime::agent::FinalizeRuntime::UnifiedLoop.as_str(),
    });
    agent_run.retrieval_run.trace.finalize_gate = gate.clone();
    agent_run.trace.finalize_gate = gate;

    let claims =
        llm::claims::fallback_claims_from_answer(&answer, &agent_run.retrieval_run.citations);
    let answer = llm::claims::strip_known_inline_citation_labels(
        &answer,
        &agent_run.retrieval_run.citations,
    );

    // If the answer round streamed live, its deltas already painted the bubble —
    // re-emitting the whole answer here would duplicate it. Only push the one-shot
    // delta for paths that produced no live tokens (e.g. a withheld/empty stream,
    // or a non-streaming fallback), so the frontend still renders the answer.
    if !already_streamed {
        if let Some(event_id) = ctx.activity_event_id {
            let _ = ctx.app.emit(
                "lumenfolio://answer-delta",
                AnswerDeltaEventOutput {
                    event_id: event_id.to_string(),
                    delta: answer.clone(),
                },
            );
        }
    }

    Ok(AskAnswerResult {
        answer,
        reasoning_content: optional_non_empty(reasoning),
        claims,
    })
}

/// Last resort when the round cap is hit while the model still wants tools: one
/// streaming call with NO tools offered, so it commits to a prose answer instead
/// of requesting yet another tool. The answer paints live like any other round.
/// Returns (content, reasoning, already_streamed).
async fn force_final_answer(
    ctx: &UnifiedLoopInput<'_>,
    client: &reqwest::Client,
    endpoint: &str,
    messages: &[serde_json::Value],
) -> Result<(String, String, bool), String> {
    let request = serde_json::json!({
        "model": ctx.provider.model,
        "messages": messages,
        "temperature": 0.2,
        "stream": true,
    });
    let round = stream_tool_round(
        client,
        endpoint,
        ctx.provider,
        &request,
        ctx.app,
        ctx.activity_event_id,
    )
    .await?;
    Ok((round.content, round.reasoning, round.emitted_answer))
}

fn emit_activity(
    app: &tauri::AppHandle,
    activity_event_id: Option<&str>,
    event: runtime::agent::AgentTraceEvent,
) {
    if let Some(event_id) = activity_event_id {
        let _ = app.emit(
            "lumenfolio://agent-activity",
            AgentActivityEventOutput {
                event_id: event_id.to_string(),
                event,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_output(
        tool: &str,
        citations: Vec<runtime::rag::Citation>,
        tree_nodes: Vec<runtime::rag::RetrievalTraceTreeNode>,
    ) -> runtime::rag::RagToolExecutionOutput {
        runtime::rag::RagToolExecutionOutput {
            citations,
            trace_candidates: Vec::new(),
            tree_nodes,
            tool_call: runtime::rag::RetrievalTraceToolCall {
                tool: tool.to_string(),
                status: "ok".to_string(),
                input: serde_json::json!({}),
                result_count: 0,
                error: None,
            },
        }
    }

    fn citation(page: u32, section: Option<&str>, quote: &str) -> runtime::rag::Citation {
        runtime::rag::Citation {
            id: "rag-c-1".to_string(),
            label: "[1]".to_string(),
            page,
            block_id: "b-1".to_string(),
            section_title: section.map(str::to_string),
            quote: quote.to_string(),
            bbox_list: serde_json::json!([]),
            document_id: "doc-1".to_string(),
            source: "search_chunks".to_string(),
        }
    }

    fn tool_names(tools: &[serde_json::Value]) -> Vec<&str> {
        tools
            .iter()
            .filter_map(|tool| tool["function"]["name"].as_str())
            .collect()
    }

    #[test]
    fn build_openai_tools_maps_specs_to_function_shape() {
        let tools = build_openai_tools(false, false, false);
        assert!(!tools.is_empty());
        // Every entry is a function tool with a name + parameters object.
        for tool in &tools {
            assert_eq!(tool["type"], "function");
            assert!(tool["function"]["name"].is_string());
            assert!(tool["function"]["parameters"].is_object());
        }
        let names = tool_names(&tools);
        assert!(names.contains(&"search_chunks"));
        // Vision-only tools are excluded when vision is disabled.
        assert!(!names.contains(&"analyze_visual"));
        // Library discovery tools only appear for a large library.
        assert!(!names.contains(&"search_library"));
        assert!(!names.contains(&"list_documents"));
    }

    #[test]
    fn build_openai_tools_adds_discovery_tools_for_large_library() {
        let names_owned: Vec<String> = build_openai_tools(false, false, true)
            .iter()
            .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_string))
            .collect();
        assert!(names_owned.iter().any(|name| name == "search_chunks"));
        assert!(names_owned.iter().any(|name| name == "search_library"));
        assert!(names_owned.iter().any(|name| name == "list_documents"));
    }

    #[test]
    fn library_tool_predicate_only_matches_discovery_tools() {
        assert!(is_library_tool("search_library"));
        assert!(is_library_tool("list_documents"));
        assert!(!is_library_tool("search_chunks"));
        assert!(!is_library_tool("open_table"));
    }

    #[test]
    fn render_tool_output_handles_empty_and_populated() {
        let empty = tool_output("search_chunks", Vec::new(), Vec::new());
        assert_eq!(
            render_tool_output("search_chunks", &empty),
            "Tool search_chunks returned no new evidence."
        );

        let populated = tool_output(
            "open_section",
            vec![citation(4, Some("Method"), "We propose X.")],
            Vec::new(),
        );
        let rendered = render_tool_output("open_section", &populated);
        assert!(rendered.contains("returned 1 evidence snippet"));
        assert!(rendered.contains("p.4"));
        assert!(rendered.contains("Method"));
        assert!(rendered.contains("We propose X."));
    }

    #[test]
    fn render_tool_output_lists_tree_nodes_when_no_citations() {
        let nodes = vec![runtime::rag::RetrievalTraceTreeNode {
            id: "n-1".to_string(),
            title: "Introduction".to_string(),
            page: 1,
            block_index: 0,
            score: 1.0,
        }];
        let output = tool_output("inspect_tree", Vec::new(), nodes);
        let rendered = render_tool_output("inspect_tree", &output);
        assert!(rendered.contains("Introduction"));
        assert!(rendered.contains("[id: n-1]"));
    }

    #[test]
    fn assistant_tool_call_message_nulls_empty_content_and_echoes_calls() {
        let calls = vec![ToolCallEntry {
            id: "call_1".to_string(),
            function: ToolCallFunction {
                name: "search_chunks".to_string(),
                arguments: "{\"query\":\"x\"}".to_string(),
            },
        }];
        let message =
            assistant_tool_call_message(&serde_json::Value::String("  ".to_string()), &calls);
        assert_eq!(message["role"], "assistant");
        assert!(message["content"].is_null());
        assert_eq!(message["tool_calls"][0]["id"], "call_1");
        assert_eq!(message["tool_calls"][0]["type"], "function");
        assert_eq!(
            message["tool_calls"][0]["function"]["name"],
            "search_chunks"
        );
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"query\":\"x\"}"
        );
    }

    #[test]
    fn parses_dsml_tool_calls() {
        // Exact delimiter form observed from deepseek-v4-flash (doubled full-width ｜).
        let p = "\u{ff5c}\u{ff5c}DSML\u{ff5c}\u{ff5c}";
        let content = format!(
            "<{p}tool_calls>\n\
             <{p}invoke name=\"search_chunks\">\n\
             <{p}parameter name=\"query\" string=\"true\">DR Tulu results</{p}parameter>\n\
             <{p}parameter name=\"limit\" string=\"false\">5</{p}parameter>\n\
             </{p}invoke>\n\
             </{p}tool_calls>"
        );
        let calls = parse_dsml_tool_calls(&content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search_chunks");
        assert_eq!(calls[0].args["query"], "DR Tulu results");
        assert_eq!(calls[0].args["limit"], 5); // string="false" -> JSON number
                                               // Normal prose yields no tool calls.
        assert!(parse_dsml_tool_calls("This is a normal answer with no markup.").is_empty());
    }

    #[test]
    fn tool_result_message_has_tool_role_and_id() {
        let message = tool_result_message("call_1", "evidence text");
        assert_eq!(message["role"], "tool");
        assert_eq!(message["tool_call_id"], "call_1");
        assert_eq!(message["content"], "evidence text");
    }

    fn round_messages(rounds: usize, tool_chars: usize) -> Vec<serde_json::Value> {
        let big = "x".repeat(tool_chars);
        let mut messages = vec![
            serde_json::json!({ "role": "system", "content": "sys" }),
            serde_json::json!({ "role": "user", "content": "question" }),
        ];
        for index in 0..rounds {
            let id = format!("call_{index}");
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": serde_json::Value::Null,
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": { "name": "search_chunks", "arguments": "{}" }
                }]
            }));
            messages.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": id,
                "content": big,
            }));
        }
        messages
    }

    #[test]
    fn compaction_noop_when_within_budget() {
        let mut messages = round_messages(2, 50);
        let pruned = compact_messages_to_budget(&mut messages, 10_000_000);
        assert_eq!(pruned, 0);
    }

    #[test]
    fn compaction_elides_old_tool_results_and_keeps_recent() {
        let mut messages = round_messages(6, 2_000);
        let before = estimate_message_chars(&messages);
        let pruned = compact_messages_to_budget(&mut messages, 5_000);
        assert!(pruned > 0, "must prune when over budget");
        assert!(estimate_message_chars(&messages) < before);
        // System + question are never touched.
        assert_eq!(messages[0]["content"], "sys");
        assert_eq!(messages[1]["content"], "question");
        // The most recent tool result stays whole (recent window preserved).
        let last_tool = messages
            .iter()
            .rev()
            .find(|message| message["role"] == "tool")
            .expect("a tool message");
        assert_ne!(last_tool["content"].as_str().unwrap(), ELIDED_TOOL_RESULT);
        // An early tool result was elided.
        assert_eq!(messages[3]["content"], ELIDED_TOOL_RESULT);
        // Assistant tool_calls structure is preserved (tool_call_id pairing intact).
        assert_eq!(messages[2]["role"], "assistant");
        assert!(messages[2]["tool_calls"].is_array());
    }

    #[test]
    fn history_char_budget_reserves_output_and_has_floor() {
        let budget =
            crate::model_catalog::ModelContextBudget::from_model_limits(32_000, 4_000, "t");
        let chars = history_char_budget(&budget);
        assert!(chars >= MIN_HISTORY_CHARS);
        assert!(chars < 32_000 * CHARS_PER_TOKEN);
        // A degenerate budget still yields at least the floor.
        let tiny = crate::model_catalog::ModelContextBudget::from_model_limits(1, 1, "t");
        assert!(history_char_budget(&tiny) >= MIN_HISTORY_CHARS);
    }
}
