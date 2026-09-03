use base64::{engine::general_purpose, Engine as _};
use rusqlite::{params, Connection, OptionalExtension};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use tauri::Emitter;

use crate::runtime::agent::{
    lexicon, tool_call_event, tool_result_event,
    trace_preview_from_output as trace_preview_from_rag_output,
};
use crate::{
    llm, normalize_base_url, runtime, truncate_for_error, vision, AgentActivityEventOutput,
    AppDatabase, AskDocumentInput, OpenAiChatRequest, OpenAiChatResponse, OpenAiCompatibleProvider,
    MAX_LLM_JUDGE_STEPS,
};

const M4_LLM_JUDGE_TIMEOUT_SECS: u64 = 45;
/// One extra attempt on a judge timeout/error before giving up — the provider is
/// often just briefly slow, and a single retry recovers most of those without
/// stalling the whole turn.
const M4_LLM_JUDGE_MAX_TRIES: u32 = 2;

pub(crate) struct LlmJudgeLoopInput<'a> {
    pub(crate) input: &'a AskDocumentInput,
    pub(crate) database: &'a AppDatabase,
    pub(crate) app: Option<&'a tauri::AppHandle>,
    pub(crate) question: &'a str,
    pub(crate) document_id: &'a str,
    /// Every document the agent may route a `documentId` tool call to this turn
    /// (focus + all visible workspace docs, capped). The dispatch whitelist.
    /// @-referenced docs are tagged inside `workspace_manifest`.
    pub(crate) visible_document_ids: &'a [&'a str],
    /// Pre-rendered workspace manifest (titles/dirs/abstracts) injected into the
    /// judge prompt so it can pick which document to search.
    pub(crate) workspace_manifest: &'a str,
    pub(crate) provider: &'a OpenAiCompatibleProvider,
    pub(crate) activity_event_id: Option<&'a str>,
}

struct VisualAssetForAnalysis {
    id: String,
    document_id: String,
    page: u32,
    asset_type: String,
    caption: String,
    bbox_list: serde_json::Value,
    image_path: String,
    nearby_text: String,
}

pub(crate) async fn improve_retrieval_with_llm_judge(
    ctx: LlmJudgeLoopInput<'_>,
    agent_run: &mut runtime::agent::AgentRunResult,
) -> Result<(), String> {
    if has_image_context(ctx.input) {
        return Ok(());
    }
    let max_steps = ctx.input.max_retrieval_steps.unwrap_or(20).clamp(1, 20);
    let mut attempt = retrieval_attempt_count(agent_run);
    let (mut remaining_llm_tool_steps, mut remaining_llm_judge_rounds) =
        llm_judge_budget(max_steps, attempt);
    // Inherit the M3 loop's "already attempted" set so the LLM judge never
    // re-requests a tool the rule loop already ran this turn.
    let mut attempted_tools: HashSet<String> =
        agent_run.ledger.attempted_signatures().cloned().collect();
    let mut tool_feedback = Vec::new();
    // Tell the judge up front which regions the M3 loop already examined, so it
    // can reason about coverage instead of re-requesting them.
    if let Some(coverage) = agent_run.ledger.coverage_summary() {
        tool_feedback.push(
            serde_json::json!({
                "observation": format!(
                    "Already examined this turn (no need to re-open): {coverage}"
                )
            })
            .to_string(),
        );
    }
    // Give the judge the whole-workspace manifest (titles/dirs/abstracts) so it can
    // locate the right document and route a tool call to it via the optional
    // `documentId` argument — but only when the question genuinely needs
    // cross-document evidence (rule in the prompt). Basic questions stay on the
    // focus document. The manifest already tags the focus + @-referenced docs.
    if !ctx.workspace_manifest.trim().is_empty() {
        tool_feedback.push(
            serde_json::json!({
                "observation": format!(
                    "{}\nUse another document's id as `documentId` ONLY when the question needs cross-document evidence; otherwise stay on the current focus document. Never invent an id — it must be one listed above.",
                    ctx.workspace_manifest.trim_end()
                )
            })
            .to_string(),
        );
    }

    while remaining_llm_judge_rounds > 0 {
        remaining_llm_judge_rounds = remaining_llm_judge_rounds.saturating_sub(1);
        emit_agent_activity_optional(
            ctx.app,
            ctx.activity_event_id,
            runtime::agent::AgentTraceEvent::new(
                "judge_start",
                "finalize_answer",
                "running",
                "M4 LLM evidence check",
                format!(
                    "Judge attempt {} is deciding whether more evidence is needed",
                    attempt + 1
                ),
                format!("llm_judge attempt={} checking answerability", attempt + 1),
            ),
        );
        // Retry once on timeout/error: the provider is often just briefly slow, so
        // a single retry recovers most blips without aborting the turn.
        let mut judge_result: Result<Result<_, String>, String> =
            Err("M4 LLM judge not attempted".to_string());
        for try_index in 0..M4_LLM_JUDGE_MAX_TRIES {
            judge_result = tokio::time::timeout(
                Duration::from_secs(M4_LLM_JUDGE_TIMEOUT_SECS),
                llm::chat::judge_answerability_with_openai_compatible(
                    ctx.question,
                    &agent_run.retrieval_run.intent,
                    &agent_run.retrieval_run.citations,
                    &tool_feedback,
                    ctx.provider,
                    &agent_run.retrieval_run.context_budget,
                ),
            )
            .await
            .map_err(|_| format!("M4 LLM judge timed out after {M4_LLM_JUDGE_TIMEOUT_SECS}s"));
            if matches!(judge_result, Ok(Ok(_))) {
                break;
            }
            if try_index + 1 < M4_LLM_JUDGE_MAX_TRIES {
                emit_agent_activity_optional(
                    ctx.app,
                    ctx.activity_event_id,
                    runtime::agent::AgentTraceEvent::new(
                        "judge_retry",
                        "finalize_answer",
                        "running",
                        "Retrying evidence check",
                        "The evidence check did not respond in time; retrying once",
                        "llm_judge timed out or errored; retrying once".to_string(),
                    ),
                );
            }
        }
        let mut decision = match judge_result {
            Ok(Ok(decision)) => decision,
            Ok(Err(err)) | Err(err) => {
                // The judge itself failed (timeout/error) even after the retry. The
                // gate stays "insufficient" — we did NOT get a positive answerability
                // verdict — but the downstream best-effort path may still answer from
                // the evidence already gathered. So the wording must not claim we are
                // "refusing to fall back": that fallback exists.
                let gate = serde_json::json!({
                    "status": "insufficient",
                    "reason": format!(
                        "The evidence check was unavailable ({err}); answering from the evidence already gathered if there is enough."
                    ),
                    "missing": serde_json::json!(["LLM evidence judge"]),
                    "nextToolCall": serde_json::Value::Null,
                    "citationCount": agent_run.retrieval_run.citations.len(),
                    "attempt": attempt,
                    "maxAttempts": max_steps,
                    "budgetExhausted": false,
                    "runtime": "m4-llm-judge",
                    "skipped": false
                });
                agent_run.retrieval_run.trace.finalize_gate = gate.clone();
                agent_run.trace.finalize_gate = gate.clone();
                let event = runtime::agent::AgentTraceEvent::new(
                    "judge_result",
                    "finalize_answer",
                    "error",
                    "M4 LLM evidence check",
                    "Evidence check unavailable; will answer from gathered evidence if sufficient",
                    format!("llm_judge unavailable after retry: {err}"),
                )
                .with_judge(gate);
                emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
                agent_run.trace.events.push(event);
                return Ok(());
            }
        };
        enforce_llm_judge_hard_gate(
            ctx.question,
            &agent_run.retrieval_run.citations,
            &mut decision,
        );

        let gate = serde_json::json!({
            "status": decision.status,
            "reason": decision.reason,
            "missing": decision.missing,
            "nextToolCall": decision.next_tool_call,
            "citationCount": agent_run.retrieval_run.citations.len(),
            "attempt": attempt,
            "maxAttempts": max_steps,
            "budgetExhausted": false,
            "runtime": "m4-llm-judge"
        });
        agent_run.retrieval_run.trace.finalize_gate = gate.clone();
        agent_run.trace.finalize_gate = gate.clone();
        let detail = format!(
            "llm_judge attempt={} status={} reason={} missing={} next_tool={}",
            attempt.saturating_add(1),
            agent_run
                .trace
                .finalize_gate
                .get("status")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown"),
            agent_run
                .trace
                .finalize_gate
                .get("reason")
                .and_then(|value| value.as_str())
                .unwrap_or(""),
            format_json_string_list(
                agent_run
                    .trace
                    .finalize_gate
                    .get("missing")
                    .unwrap_or(&serde_json::Value::Null),
            ),
            agent_run
                .trace
                .finalize_gate
                .get("nextToolCall")
                .and_then(|value| value.get("tool"))
                .and_then(|value| value.as_str())
                .unwrap_or("none")
        );
        let event = runtime::agent::AgentTraceEvent::new(
            "judge_result",
            "finalize_answer",
            "completed",
            "M4 LLM evidence check",
            format!(
                "status={} next_tool={}",
                agent_run
                    .trace
                    .finalize_gate
                    .get("status")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown"),
                agent_run
                    .trace
                    .finalize_gate
                    .get("nextToolCall")
                    .and_then(|value| value.get("tool"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("none")
            ),
            detail,
        )
        .with_judge(gate);
        emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
        agent_run.trace.events.push(event);

        let status = agent_run
            .trace
            .finalize_gate
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if status != "needs_more_evidence" {
            return Ok(());
        }
        if remaining_llm_tool_steps == 0 {
            let gate = serde_json::json!({
                "status": "insufficient",
                "reason": agent_run
                    .trace
                    .finalize_gate
                    .get("reason")
                    .and_then(|value| value.as_str())
                    .map(|reason| format!("{reason} Reached the retrieval step limit."))
                    .unwrap_or_else(|| "LLM judge requested more evidence, but the retrieval step limit has been reached.".to_string()),
                "missing": agent_run
                    .trace
                    .finalize_gate
                    .get("missing")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([])),
                "nextToolCall": agent_run
                    .trace
                    .finalize_gate
                    .get("nextToolCall")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
                "citationCount": agent_run.retrieval_run.citations.len(),
                "attempt": attempt,
                "maxAttempts": max_steps,
                "budgetExhausted": true,
                "runtime": "m4-llm-judge"
            });
            agent_run.retrieval_run.trace.finalize_gate = gate.clone();
            agent_run.trace.finalize_gate = gate.clone();
            let event = runtime::agent::AgentTraceEvent::new(
                "judge_stop",
                "finalize_answer",
                "skipped",
                "Stopping retrieval",
                "LLM judge requested more evidence, but no retrieval budget remains",
                "llm_judge stopped: no retrieval budget remains",
            )
            .with_judge(gate);
            emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
            agent_run.trace.events.push(event);
            return Ok(());
        }
        let Some(call) = agent_run
            .trace
            .finalize_gate
            .get("nextToolCall")
            .cloned()
            .and_then(|value| serde_json::from_value::<llm::chat::LlmRagToolCall>(value).ok())
        else {
            return Ok(());
        };
        let vision_enabled = ctx
            .provider
            .capabilities
            .iter()
            .any(|capability| capability == "vision");
        let rag_capabilities = runtime::rag::RagToolCapabilities {
            vision_enabled,
            // Web tools are scoped to the unified-loop / local-agent paths, not the
            // M3/M4 judge-driven retrieval.
            web_enabled: false,
            max_quote_chars: agent_run.retrieval_run.context_budget.max_quote_chars,
        };
        if !runtime::rag::is_registered_rag_tool_for_capabilities(&call.tool, rag_capabilities) {
            let gate = serde_json::json!({
                "status": "insufficient",
                "reason": format!(
                    "LLM judge requested unavailable tool {} for the selected model capabilities.",
                    call.tool
                ),
                "missing": serde_json::json!(["available retrieval tool"]),
                "nextToolCall": serde_json::Value::Null,
                "citationCount": agent_run.retrieval_run.citations.len(),
                "attempt": attempt,
                "maxAttempts": max_steps,
                "budgetExhausted": false,
                "runtime": "m4-llm-judge"
            });
            agent_run.retrieval_run.trace.finalize_gate = gate.clone();
            agent_run.trace.finalize_gate = gate.clone();
            let event = runtime::agent::AgentTraceEvent::new(
                "judge_stop",
                "finalize_answer",
                "error",
                "Stopping retrieval",
                "Judge returned a tool that is unavailable for the selected model",
                format!(
                    "llm_judge returned unavailable tool={} vision_enabled={}",
                    call.tool, vision_enabled
                ),
            )
            .with_judge(gate);
            emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
            agent_run.trace.events.push(event);
            return Ok(());
        }
        let signature = format!("{}:{}", call.tool, call.args);
        // Mirror the attempt into the shared ledger so it is the single source
        // of "what have I already looked at" across the M3 and M4 loops.
        agent_run.ledger.record_tool_call(&call.tool, &call.args);
        if !attempted_tools.insert(signature) {
            if remaining_llm_judge_rounds > 0 {
                let feedback = serde_json::json!({
                    "tool": call.tool,
                    "args": call.args,
                    "reason": call.reason,
                    "observation": "The LLM judge repeated this exact tool request. The tool will not be called again; decide from current evidence, choose a different useful tool, or mark insufficient."
                })
                .to_string();
                let event = runtime::agent::AgentTraceEvent::new(
                    "judge_feedback",
                    "finalize_answer",
                    "skipped",
                    "Tool feedback",
                    "Judge repeated a previous tool request; asking it to decide again",
                    format!(
                        "llm_judge repeated tool={} with same args; retrying judge with feedback",
                        call.tool
                    ),
                )
                .with_tool(call.tool.clone(), call.args.clone());
                emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
                agent_run.trace.events.push(event);
                tool_feedback.push(feedback);
                continue;
            }
            let gate = serde_json::json!({
                "status": "insufficient",
                "reason": format!(
                    "LLM judge repeated the same tool request {}; stopping to avoid a retrieval loop.",
                    call.tool
                ),
                "missing": serde_json::json!(["non-repeating retrieval step"]),
                "nextToolCall": serde_json::Value::Null,
                "citationCount": agent_run.retrieval_run.citations.len(),
                "attempt": attempt,
                "maxAttempts": max_steps,
                "budgetExhausted": false,
                "runtime": "m4-llm-judge"
            });
            agent_run.retrieval_run.trace.finalize_gate = gate.clone();
            agent_run.trace.finalize_gate = gate.clone();
            let event = runtime::agent::AgentTraceEvent::new(
                "judge_stop",
                "finalize_answer",
                "error",
                "Stopping retrieval",
                "Judge repeated the same tool request",
                format!("llm_judge repeated tool={} with same args", call.tool),
            )
            .with_tool(call.tool.clone(), call.args.clone())
            .with_judge(gate);
            emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
            agent_run.trace.events.push(event);
            return Ok(());
        }

        let fallback_query_owned;
        let fallback_query =
            if let Some(query) = call.args.get("query").and_then(|value| value.as_str()) {
                query
            } else {
                fallback_query_owned = judge_tool_fallback_query(
                    ctx.question,
                    &agent_run.retrieval_run.intent,
                    &call.tool,
                );
                fallback_query_owned.as_str()
            };
        let tool_start = tool_call_event(
            call.tool.clone(),
            call.args.clone(),
            format!("Calling {}", call.tool),
            call.reason.clone(),
            format!(
                "llm_judge tool={} args={} fallback_query={}",
                call.tool, call.args, fallback_query
            ),
        );
        emit_agent_activity_optional(ctx.app, ctx.activity_event_id, tool_start.clone());

        let output = if call.tool == "analyze_visual" {
            execute_analyze_visual_with_provider(&ctx, &call, fallback_query).await
        } else if call.tool == "analyze_page" {
            execute_analyze_page_with_provider(&ctx, &call, fallback_query).await
        } else {
            let conn = ctx
                .database
                .conn
                .lock()
                .map_err(|_| "SQLite lock was poisoned".to_string())?;
            Ok(runtime::rag::execute_rag_tool_call_for_capabilities(
                &conn,
                ctx.document_id,
                ctx.visible_document_ids,
                &call.tool,
                &call.args,
                fallback_query,
                rag_capabilities,
            ))
        }?;
        let event = tool_result_event(
            &output,
            output.tool_call.tool.clone(),
            format!(
                "llm_judge tool={} results={} reason={}",
                output.tool_call.tool, output.tool_call.result_count, call.reason
            ),
        );
        emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
        agent_run.trace.events.push(event);

        let mut gained_citations = apply_judge_tool_output(agent_run, &output).0;
        gained_citations = gained_citations.saturating_add(maybe_open_table_page_fallback(
            &ctx,
            agent_run,
            &call,
            &output,
            fallback_query,
        )?);
        if gained_citations == 0
            && matches!(call.tool.as_str(), "inspect_tree" | "read_tree_node_lines")
            && !output.tree_nodes.is_empty()
        {
            let node_ids = output
                .tree_nodes
                .iter()
                .map(|node| serde_json::Value::String(node.id.clone()))
                .collect::<Vec<_>>();
            if call.tool == "inspect_tree" {
                let line_args = serde_json::json!({
                    "treeNodeIds": node_ids.clone(),
                    "query": fallback_query,
                    "lineLimit": 12
                });
                let line_start = tool_call_event(
                    "read_tree_node_lines",
                    line_args.clone(),
                    "Calling read_tree_node_lines",
                    "Inspecting line-level content inside matched sections",
                    format!(
                        "llm_judge follow-up tool=read_tree_node_lines args={} fallback_query={}",
                        line_args, fallback_query
                    ),
                );
                emit_agent_activity_optional(ctx.app, ctx.activity_event_id, line_start.clone());
                let line_output = {
                    let conn = ctx
                        .database
                        .conn
                        .lock()
                        .map_err(|_| "SQLite lock was poisoned".to_string())?;
                    runtime::rag::execute_rag_tool_call_for_capabilities(
                        &conn,
                        ctx.document_id,
                        ctx.visible_document_ids,
                        "read_tree_node_lines",
                        &line_args,
                        fallback_query,
                        rag_capabilities,
                    )
                };
                let event = tool_result_event(
                    &line_output,
                    line_output.tool_call.tool.clone(),
                    format!(
                        "llm_judge follow-up tool={} results={} reason=inspect_tree returned section nodes",
                        line_output.tool_call.tool, line_output.tool_call.result_count
                    ),
                );
                emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
                agent_run.trace.events.push(event);
                gained_citations = gained_citations
                    .saturating_add(apply_judge_tool_output(agent_run, &line_output).0);
            }

            if gained_citations == 0 {
                let open_args = serde_json::json!({
                    "treeNodeIds": node_ids,
                    "query": fallback_query,
                    "perSectionLimit": 10
                });
                let section_start = tool_call_event(
                    "open_section",
                    open_args.clone(),
                    "Calling open_section",
                    "Opening full section evidence after line lookup found nothing new",
                    format!(
                        "llm_judge follow-up tool=open_section args={} fallback_query={}",
                        open_args, fallback_query
                    ),
                );
                emit_agent_activity_optional(ctx.app, ctx.activity_event_id, section_start.clone());
                let section_output = {
                    let conn = ctx
                        .database
                        .conn
                        .lock()
                        .map_err(|_| "SQLite lock was poisoned".to_string())?;
                    runtime::rag::execute_rag_tool_call_for_capabilities(
                        &conn,
                        ctx.document_id,
                        ctx.visible_document_ids,
                        "open_section",
                        &open_args,
                        fallback_query,
                        rag_capabilities,
                    )
                };
                let event = tool_result_event(
                    &section_output,
                    section_output.tool_call.tool.clone(),
                    format!(
                        "llm_judge follow-up tool={} results={} reason=line evidence unavailable",
                        section_output.tool_call.tool, section_output.tool_call.result_count
                    ),
                );
                emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
                agent_run.trace.events.push(event);
                gained_citations = gained_citations
                    .saturating_add(apply_judge_tool_output(agent_run, &section_output).0);
            }
        }
        if gained_citations == 0 {
            let feedback = no_gain_tool_feedback(&call, &output);
            let event = runtime::agent::AgentTraceEvent::new(
                "judge_feedback",
                "finalize_answer",
                "skipped",
                "Tool feedback",
                "Tool returned no new evidence; asking the LLM judge to decide again",
                format!(
                    "llm_judge tool={} produced no new evidence; retrying judge with feedback",
                    call.tool
                ),
            )
            .with_tool(call.tool.clone(), call.args.clone());
            emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
            agent_run.trace.events.push(event);
            tool_feedback.push(feedback);
            attempt = attempt.saturating_add(1);
            remaining_llm_tool_steps = remaining_llm_tool_steps.saturating_sub(1);
            if remaining_llm_judge_rounds > 0 {
                continue;
            }
            let gate = serde_json::json!({
                "status": "insufficient",
                "reason": format!("LLM judge requested {}, but the tool returned no new evidence and no judge rounds remain.", call.tool),
                "missing": agent_run
                    .trace
                    .finalize_gate
                    .get("missing")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([])),
                "nextToolCall": serde_json::Value::Null,
                "citationCount": agent_run.retrieval_run.citations.len(),
                "attempt": attempt,
                "maxAttempts": max_steps,
                "budgetExhausted": false,
                "runtime": "m4-llm-judge"
            });
            agent_run.retrieval_run.trace.finalize_gate = gate.clone();
            agent_run.trace.finalize_gate = gate.clone();
            let event = runtime::agent::AgentTraceEvent::new(
                "judge_stop",
                "finalize_answer",
                "skipped",
                "Stopping retrieval",
                "Tool returned no new evidence and no judge rounds remain",
                format!(
                    "llm_judge stopped: tool={} produced no new evidence and no judge rounds remain",
                    call.tool
                ),
            )
            .with_judge(gate);
            emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
            agent_run.trace.events.push(event);
            return Ok(());
        }
        attempt = attempt.saturating_add(1);
        remaining_llm_tool_steps = remaining_llm_tool_steps.saturating_sub(1);
    }
    Ok(())
}

async fn execute_analyze_visual_with_provider(
    ctx: &LlmJudgeLoopInput<'_>,
    call: &llm::chat::LlmRagToolCall,
    fallback_query: &str,
) -> Result<runtime::rag::RagToolExecutionOutput, String> {
    let asset_id = call
        .args
        .get("assetId")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(asset_id) = asset_id else {
        return Ok(analyze_visual_error_output(
            &call.args,
            "analyze_visual requires assetId",
        ));
    };
    let question = call
        .args
        .get("question")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback_query);
    let asset = {
        let conn = ctx
            .database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        load_visual_asset_for_analysis(&conn, ctx.document_id, asset_id)?
    };
    let Some(asset) = asset else {
        return Ok(analyze_visual_error_output(
            &call.args,
            "visual asset was not found",
        ));
    };
    let image_data_url = match visual_asset_data_url(&asset) {
        Ok(value) => value,
        Err(err) => return Ok(analyze_visual_error_output(&call.args, &err)),
    };

    let prompt = format!(
        "Analyze this visual crop from an academic PDF.\n\
         User question: {question}\n\
         Asset type: {}\n\
         Caption: {}\n\
         Nearby text: {}\n\n\
         Return a concise evidence-focused description. Do not invent values that are not visible.",
        asset.asset_type, asset.caption, asset.nearby_text
    );
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(45))
        .build()
        .map_err(|err| format!("Failed to create visual analysis client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&ctx.provider.base_url)
    );
    let request = OpenAiChatRequest {
        model: ctx.provider.model.clone(),
        temperature: 0.0,
        stream: None,
        messages: vec![
            llm::chat::text_message(
                "system",
                "You are a careful visual evidence analyzer for an academic PDF. Use only the image and supplied caption/nearby text.",
            ),
            llm::chat::user_message_with_optional_image(&prompt, Some(&image_data_url)),
        ],
    };
    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = &ctx.provider.api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = match builder.send().await {
        Ok(response) => response,
        Err(err) => {
            return Ok(analyze_visual_error_output(
                &call.args,
                &format!("visual analysis request failed: {err}"),
            ));
        }
    };
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Ok(analyze_visual_error_output(
            &call.args,
            &format!(
                "visual analysis provider returned {status}: {}",
                truncate_for_error(&body, 400)
            ),
        ));
    }
    let response = response
        .json::<OpenAiChatResponse>()
        .await
        .map_err(|err| format!("Failed to decode visual analysis response: {err}"))?;
    let answer = response
        .choices
        .into_iter()
        .next()
        .map(|choice| llm::chat::extract_chat_response_text(&choice.message.content))
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| "The vision model returned no visual analysis.".to_string());
    let quote = format!(
        "Visual analysis for {} on page {}:\n{}\n\nCaption: {}",
        asset.asset_type, asset.page, answer, asset.caption
    );
    let citation = runtime::rag::Citation {
        id: format!("rag-c-analyze-visual-{}", asset.id),
        label: "[1]".to_string(),
        page: asset.page,
        block_id: asset.id.clone(),
        section_title: Some(format!("{} on page {}", asset.asset_type, asset.page)),
        quote: truncate_for_error(&quote, 1_800),
        bbox_list: asset.bbox_list.clone(),
        document_id: asset.document_id.clone(),
        source: "analyze_visual".to_string(),
    };
    Ok(runtime::rag::RagToolExecutionOutput {
        citations: vec![citation.clone()],
        trace_candidates: vec![runtime::rag::RetrievalTraceCandidate {
            source: citation.source.clone(),
            page: citation.page,
            block_id: citation.block_id.clone(),
            tree_node_id: None,
            section_title: citation.section_title.clone(),
            quote: truncate_for_error(&citation.quote, 240),
        }],
        tree_nodes: Vec::new(),
        tool_call: runtime::rag::RetrievalTraceToolCall {
            tool: "analyze_visual".to_string(),
            status: "ok".to_string(),
            input: call.args.clone(),
            result_count: 1,
            error: None,
        },
    })
}

async fn execute_analyze_page_with_provider(
    ctx: &LlmJudgeLoopInput<'_>,
    call: &llm::chat::LlmRagToolCall,
    fallback_query: &str,
) -> Result<runtime::rag::RagToolExecutionOutput, String> {
    let page = call
        .args
        .get("page")
        .and_then(|value| value.as_u64())
        .filter(|value| *value > 0)
        .and_then(|value| u32::try_from(value).ok());
    let Some(page) = page else {
        return Ok(analyze_page_error_output(
            &call.args,
            "analyze_page requires page",
        ));
    };
    let question = call
        .args
        .get("question")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback_query);
    let pdf_path = {
        let conn = ctx
            .database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        document_pdf_path_for_analysis(&conn, ctx.document_id)?
    };
    let image_data_url =
        match tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
            let page_image = vision::render_pdf_page_image(&pdf_path, page)?;
            image_file_data_url(&page_image)
        })
        .await
        {
            Ok(Ok(value)) => value,
            Ok(Err(err)) => return Ok(analyze_page_error_output(&call.args, &err)),
            Err(err) => {
                return Ok(analyze_page_error_output(
                    &call.args,
                    &format!("page rendering task failed: {err}"),
                ));
            }
        };
    let page_text = {
        let conn = ctx
            .database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        page_text_for_analysis(&conn, ctx.document_id, page)?
    };

    let prompt = format!(
        "Analyze this full page screenshot from an academic PDF.\n\
         User question: {question}\n\
         Page: {page}\n\
         Extracted page text:\n{page_text}\n\n\
         Use the screenshot for visual layout, charts, figures, and tables. Use the text only as supporting context. Return concise evidence-focused findings and do not invent values that are not visible.",
    );
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(45))
        .build()
        .map_err(|err| format!("Failed to create page analysis client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&ctx.provider.base_url)
    );
    let request = OpenAiChatRequest {
        model: ctx.provider.model.clone(),
        temperature: 0.0,
        stream: None,
        messages: vec![
            llm::chat::text_message(
                "system",
                "You are a careful page-level visual evidence analyzer for academic PDFs. Use only the screenshot and supplied page text.",
            ),
            llm::chat::user_message_with_optional_image(&prompt, Some(&image_data_url)),
        ],
    };
    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = &ctx.provider.api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = match builder.send().await {
        Ok(response) => response,
        Err(err) => {
            return Ok(analyze_page_error_output(
                &call.args,
                &format!("page analysis request failed: {err}"),
            ));
        }
    };
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Ok(analyze_page_error_output(
            &call.args,
            &format!(
                "page analysis provider returned {status}: {}",
                truncate_for_error(&body, 400)
            ),
        ));
    }
    let response = response
        .json::<OpenAiChatResponse>()
        .await
        .map_err(|err| format!("Failed to decode page analysis response: {err}"))?;
    let answer = response
        .choices
        .into_iter()
        .next()
        .map(|choice| llm::chat::extract_chat_response_text(&choice.message.content))
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| "The vision model returned no page analysis.".to_string());
    let quote = format!("Full-page visual analysis for page {page}:\n{answer}");
    let citation = runtime::rag::Citation {
        id: format!("rag-c-analyze-page-{page}"),
        label: "[1]".to_string(),
        page,
        block_id: format!("page-image-{page}"),
        section_title: Some(format!("Page {page} screenshot")),
        quote: truncate_for_error(&quote, 1_800),
        bbox_list: serde_json::json!([[0.0, 0.0, 1.0, 1.0]]),
        document_id: ctx.document_id.to_string(),
        source: "analyze_page".to_string(),
    };
    Ok(runtime::rag::RagToolExecutionOutput {
        citations: vec![citation.clone()],
        trace_candidates: vec![runtime::rag::RetrievalTraceCandidate {
            source: citation.source.clone(),
            page: citation.page,
            block_id: citation.block_id.clone(),
            tree_node_id: None,
            section_title: citation.section_title.clone(),
            quote: truncate_for_error(&citation.quote, 240),
        }],
        tree_nodes: Vec::new(),
        tool_call: runtime::rag::RetrievalTraceToolCall {
            tool: "analyze_page".to_string(),
            status: "ok".to_string(),
            input: call.args.clone(),
            result_count: 1,
            error: None,
        },
    })
}

fn analyze_visual_error_output(
    args: &serde_json::Value,
    error: &str,
) -> runtime::rag::RagToolExecutionOutput {
    runtime::rag::RagToolExecutionOutput {
        citations: Vec::new(),
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: runtime::rag::RetrievalTraceToolCall {
            tool: "analyze_visual".to_string(),
            status: "error".to_string(),
            input: args.clone(),
            result_count: 0,
            error: Some(error.to_string()),
        },
    }
}

fn analyze_page_error_output(
    args: &serde_json::Value,
    error: &str,
) -> runtime::rag::RagToolExecutionOutput {
    runtime::rag::RagToolExecutionOutput {
        citations: Vec::new(),
        trace_candidates: Vec::new(),
        tree_nodes: Vec::new(),
        tool_call: runtime::rag::RetrievalTraceToolCall {
            tool: "analyze_page".to_string(),
            status: "error".to_string(),
            input: args.clone(),
            result_count: 0,
            error: Some(error.to_string()),
        },
    }
}

fn load_visual_asset_for_analysis(
    conn: &Connection,
    document_id: &str,
    asset_id: &str,
) -> Result<Option<VisualAssetForAnalysis>, String> {
    conn.query_row(
        "SELECT id, document_id, page_no, asset_type, caption, bbox_json, image_path, nearby_text
         FROM document_visual_assets
         WHERE document_id = ?1 AND id = ?2
         LIMIT 1",
        params![document_id, asset_id],
        |row| {
            let bbox_json: String = row.get(5)?;
            let bbox_list: serde_json::Value =
                serde_json::from_str(&bbox_json).unwrap_or_else(|_| serde_json::json!([]));
            Ok(VisualAssetForAnalysis {
                id: row.get(0)?,
                document_id: row.get(1)?,
                page: row.get(2)?,
                asset_type: row.get(3)?,
                caption: row.get(4)?,
                bbox_list,
                image_path: row.get(6)?,
                nearby_text: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(|err| format!("Failed to load visual asset for analysis: {err}"))
}

fn document_pdf_path_for_analysis(conn: &Connection, document_id: &str) -> Result<PathBuf, String> {
    let path = conn
        .query_row(
            "SELECT path FROM documents WHERE id = ?1 LIMIT 1",
            params![document_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|err| format!("Failed to load document path for page analysis: {err}"))?
        .ok_or_else(|| "Document is no longer available for page analysis".to_string())?;
    let path = PathBuf::from(path);
    if !path.is_file() {
        return Err(format!("PDF is no longer available at {}", path.display()));
    }
    Ok(path)
}

fn page_text_for_analysis(
    conn: &Connection,
    document_id: &str,
    page: u32,
) -> Result<String, String> {
    let mut stmt = conn
        .prepare(
            "SELECT text
             FROM document_blocks
             WHERE document_id = ?1 AND page_no = ?2
             ORDER BY block_index
             LIMIT 80",
        )
        .map_err(|err| format!("Failed to prepare page text for analysis: {err}"))?;
    let rows = stmt
        .query_map(params![document_id, page], |row| row.get::<_, String>(0))
        .map_err(|err| format!("Failed to read page text for analysis: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to collect page text for analysis: {err}"))?;
    let text = rows
        .into_iter()
        .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|text| !text.is_empty())
        .take(24)
        .collect::<Vec<_>>()
        .join("\n");
    Ok(truncate_for_error(&text, 2_400))
}

fn visual_asset_data_url(asset: &VisualAssetForAnalysis) -> Result<String, String> {
    let image_path = asset.image_path.trim();
    if image_path.is_empty() {
        return Err(
            "visual asset has no crop path; run Rust visual indexing with crop support first"
                .to_string(),
        );
    }
    image_file_data_url(&PathBuf::from(image_path))
}

fn image_file_data_url(path: &Path) -> Result<String, String> {
    let metadata =
        fs::metadata(path).map_err(|err| format!("failed to read image metadata: {err}"))?;
    const MAX_VISUAL_CROP_BYTES: u64 = 8 * 1024 * 1024;
    if metadata.len() > MAX_VISUAL_CROP_BYTES {
        return Err(format!(
            "image is too large for inline analysis: {} bytes",
            metadata.len()
        ));
    }
    let mime = match path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("png") => "image/png",
        _ => return Err("image must be a PNG, JPEG, or WebP image".to_string()),
    };
    let bytes = fs::read(path).map_err(|err| format!("failed to read image: {err}"))?;
    Ok(format!(
        "data:{mime};base64,{}",
        general_purpose::STANDARD.encode(bytes)
    ))
}

pub(crate) fn apply_judge_tool_output(
    agent_run: &mut runtime::agent::AgentRunResult,
    output: &runtime::rag::RagToolExecutionOutput,
) -> (usize, usize) {
    let before_citation_signatures = citation_change_signatures(&agent_run.retrieval_run.citations);
    let before_tree_node_count = agent_run.retrieval_run.trace.tree_nodes.len();
    let mut accumulated = agent_run.retrieval_run.citations.clone();
    runtime::rag::merge_retrieval_citations_with_budget(
        &mut accumulated,
        &output.citations,
        &agent_run.retrieval_run.context_budget,
    );
    let gained_citations = citation_change_signatures(&accumulated)
        .difference(&before_citation_signatures)
        .count();
    let latest_gate = agent_run.trace.finalize_gate.clone();
    runtime::rag::apply_tool_execution(&mut agent_run.retrieval_run, output);
    runtime::rag::apply_retrieval_citations(&mut agent_run.retrieval_run, accumulated);
    agent_run.retrieval_run.trace.finalize_gate = latest_gate.clone();
    agent_run.trace.sync_retrieval_view(
        agent_run.retrieval_run.trace.clone(),
        &agent_run.retrieval_run.citations,
    );
    agent_run.trace.finalize_gate = latest_gate;
    let gained_tree_nodes = agent_run
        .retrieval_run
        .trace
        .tree_nodes
        .len()
        .saturating_sub(before_tree_node_count);
    // Record the regions this tool surfaced so the shared ledger can report an
    // honest "already looked at" summary spanning both the M3 and M4 loops.
    agent_run.ledger.record_coverage(&output.citations);
    (gained_citations, gained_tree_nodes)
}

fn citation_change_signatures(citations: &[runtime::rag::Citation]) -> HashSet<String> {
    citations
        .iter()
        .map(citation_change_signature)
        .collect::<HashSet<_>>()
}

fn citation_change_signature(citation: &runtime::rag::Citation) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        citation.source,
        citation.document_id,
        citation.block_id,
        citation.page,
        citation.quote.trim()
    )
}

pub(super) fn enforce_llm_judge_hard_gate(
    question: &str,
    citations: &[runtime::rag::Citation],
    decision: &mut llm::chat::LlmAnswerabilityDecision,
) {
    if decision.status != "answerable" {
        return;
    }
    if citations.is_empty() {
        decision.status = "insufficient".to_string();
        decision.reason = "Hard gate rejected answerable: no citations are available.".to_string();
        decision.missing = vec!["document evidence".to_string()];
        decision.next_tool_call = None;
        return;
    }
    if !question_needs_table_evidence(question) {
        return;
    }
    if has_current_view_table_evidence(question, citations) {
        return;
    }
    if !citations
        .iter()
        .any(|citation| citation.source.as_str() == "open_table")
    {
        decision.status = "needs_more_evidence".to_string();
        decision.reason =
            "Hard gate requires open_table evidence for table or metric questions.".to_string();
        decision.missing = vec!["full table evidence".to_string()];
        decision.next_tool_call = Some(llm::chat::LlmRagToolCall {
            tool: "open_table".to_string(),
            args: serde_json::json!({ "query": question, "limit": 40 }),
            reason: "Open the candidate table before deciding exact table values.".to_string(),
        });
        return;
    }
    if let Some(table_no) = requested_table_number(question) {
        let needle = format!("table {table_no}");
        let has_requested_table = citations.iter().any(|citation| {
            citation.source.as_str() == "open_table"
                && citation
                    .section_title
                    .as_deref()
                    .and_then(requested_table_number)
                    .as_deref()
                    .is_some_and(|candidate_no| candidate_no == table_no)
        });
        if !has_requested_table {
            decision.status = "needs_more_evidence".to_string();
            decision.reason =
                format!("Hard gate requires open_table evidence from the requested {needle}.");
            decision.missing = vec![format!("{needle} open_table evidence")];
            decision.next_tool_call = Some(llm::chat::LlmRagToolCall {
                tool: "open_table".to_string(),
                args: serde_json::json!({
                    "tableNumber": table_no,
                    "query": question,
                    "limit": 40
                }),
                reason: "Open the requested table number before answering.".to_string(),
            });
        }
    }
}

fn has_current_view_table_evidence(question: &str, citations: &[runtime::rag::Citation]) -> bool {
    let requested = requested_table_number(question);
    citations
        .iter()
        .filter(|citation| citation.source.as_str() == "current_view")
        .any(|citation| {
            if let Some(table_no) = requested.as_deref() {
                return citation
                    .section_title
                    .as_deref()
                    .and_then(requested_table_number)
                    .as_deref()
                    == Some(table_no)
                    || requested_table_number(&citation.quote).as_deref() == Some(table_no);
            }
            let haystack = format!(
                "{}\n{}",
                citation.section_title.as_deref().unwrap_or_default(),
                citation.quote
            )
            .to_lowercase();
            haystack.contains("table")
                || haystack.contains("metric")
                || haystack.contains("score")
                || haystack.contains("benchmark")
                || haystack.contains("performance")
                || haystack.contains("表")
                || haystack.contains("指标")
        })
}

pub(super) fn requires_llm_judge_for_answer(_question: &str) -> bool {
    true
}

pub(super) fn question_needs_table_evidence(question: &str) -> bool {
    let normalized = question.to_lowercase();
    requested_table_number(question).is_some()
        || normalized.contains("table")
        || normalized.contains("benchmark")
        || normalized.contains("metric")
        || normalized.contains("score")
        || normalized.contains("sota")
        || normalized.contains("performance")
        || normalized.contains("表格")
        || normalized.contains("指标")
        || normalized.contains("分数")
        || normalized.contains("成绩")
        || normalized.contains("数值")
}

pub(super) fn requested_table_number(question: &str) -> Option<String> {
    lexicon::requested_table_number(question)
}

fn no_gain_tool_feedback(
    call: &llm::chat::LlmRagToolCall,
    output: &runtime::rag::RagToolExecutionOutput,
) -> String {
    let preview = trace_preview_from_rag_output(output)
        .into_iter()
        .map(|item| {
            serde_json::json!({
                "page": item.page,
                "sectionTitle": item.section_title,
                "quote": truncate_for_error(&item.quote, 240),
                "source": item.source
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "tool": call.tool,
        "args": call.args,
        "reason": call.reason,
        "status": output.tool_call.status,
        "resultCount": output.tool_call.result_count,
        "observation": "The tool call completed but added no new unique citations to current evidence.",
        "preview": preview
    })
    .to_string()
}

fn maybe_open_table_page_fallback(
    ctx: &LlmJudgeLoopInput<'_>,
    agent_run: &mut runtime::agent::AgentRunResult,
    call: &llm::chat::LlmRagToolCall,
    output: &runtime::rag::RagToolExecutionOutput,
    fallback_query: &str,
) -> Result<usize, String> {
    let Some(page) = table_page_fallback_page(ctx.question, call, output) else {
        return Ok(0);
    };
    if agent_run.retrieval_run.citations.iter().any(|citation| {
        citation.source == "open_pages"
            && citation.page == page
            && citation
                .section_title
                .as_deref()
                .is_some_and(|title| title.contains(" lines "))
    }) {
        return Ok(0);
    }

    let page_args = serde_json::json!({
        "page": page,
        "mode": "full",
        "limit": 40
    });
    let page_start = tool_call_event(
        "open_pages",
        page_args.clone(),
        "Calling open_pages",
        "Opening the source page because structured table extraction looked sparse or noisy",
        format!(
            "llm_judge follow-up tool=open_pages args={} fallback_query={}",
            page_args, fallback_query
        ),
    );
    emit_agent_activity_optional(ctx.app, ctx.activity_event_id, page_start.clone());

    let page_output = {
        let conn = ctx
            .database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        runtime::rag::execute_rag_tool_call_for_capabilities(
            &conn,
            ctx.document_id,
            // v0: table→page fallback stays single-document (cross-doc tables are Phase 3).
            &[],
            "open_pages",
            &page_args,
            fallback_query,
            runtime::rag::RagToolCapabilities {
                vision_enabled: ctx
                    .provider
                    .capabilities
                    .iter()
                    .any(|capability| capability == "vision"),
                web_enabled: false,
                max_quote_chars: agent_run.retrieval_run.context_budget.max_quote_chars,
            },
        )
    };
    let event = tool_result_event(
        &page_output,
        page_output.tool_call.tool.clone(),
        format!(
            "llm_judge follow-up tool={} results={} reason=structured table evidence was incomplete",
            page_output.tool_call.tool, page_output.tool_call.result_count
        ),
    );
    emit_agent_activity_optional(ctx.app, ctx.activity_event_id, event.clone());
    agent_run.trace.events.push(event);

    Ok(apply_judge_tool_output(agent_run, &page_output).0)
}

fn table_page_fallback_page(
    question: &str,
    call: &llm::chat::LlmRagToolCall,
    output: &runtime::rag::RagToolExecutionOutput,
) -> Option<u32> {
    if call.tool != "open_table" {
        return None;
    }
    let requested = requested_table_number(question)?;
    runtime::agent::table_fallback::sparse_open_table_page(output, Some(requested.as_str()))
}

fn format_json_string_list(value: &serde_json::Value) -> String {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "none".to_string())
}

pub(super) fn emit_agent_activity(
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

fn emit_agent_activity_optional(
    app: Option<&tauri::AppHandle>,
    activity_event_id: Option<&str>,
    event: runtime::agent::AgentTraceEvent,
) {
    if let Some(app) = app {
        emit_agent_activity(app, activity_event_id, event);
    }
}

pub(super) fn has_image_context(input: &AskDocumentInput) -> bool {
    input
        .image_data_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
}

pub(crate) fn retrieval_is_answerable(agent_run: &runtime::agent::AgentRunResult) -> bool {
    use runtime::agent::{FinalizeRuntime, FinalizeStatus};
    let gate = &agent_run.retrieval_run.trace.finalize_gate;
    let status_is_answerable = gate
        .get("status")
        .and_then(|value| value.as_str())
        .is_some_and(|status| status == FinalizeStatus::Answerable.as_str());
    // A real verdict (M4 LLM judge or the unified agent loop) must have cleared the
    // turn — an M3 rule/heuristic seed alone never authorizes an answer.
    let runtime_is_verdict = gate
        .get("runtime")
        .and_then(|value| value.as_str())
        .and_then(FinalizeRuntime::parse)
        .is_some_and(FinalizeRuntime::is_verdict);
    status_is_answerable && runtime_is_verdict
}

/// Minimum accumulated citations to attempt a best-effort answer instead of
/// refusing. Below this, the evidence is too thin to say anything useful.
const BEST_EFFORT_MIN_CITATIONS: usize = 2;

/// Best-effort fallback: the judge loop ended WITHOUT "answerable" (it kept asking
/// for evidence it never found, or marked insufficient), but enough evidence was
/// actually accumulated to give a useful, caveated answer rather than refusing.
/// This makes open-ended questions degrade to "here's what I can tell, with stated
/// limits" instead of "ask a more specific question" — without per-scenario rules.
///
/// Excludes the genuinely-empty case (no citations) so we never fabricate an answer
/// from nothing.
pub(crate) fn should_answer_best_effort(agent_run: &runtime::agent::AgentRunResult) -> bool {
    if retrieval_is_answerable(agent_run) {
        return false;
    }
    agent_run.retrieval_run.citations.len() >= BEST_EFFORT_MIN_CITATIONS
}

fn judge_tool_fallback_query(question: &str, intent: &str, tool: &str) -> String {
    let normalized = question.to_lowercase();
    if matches!(
        tool,
        "inspect_tree" | "read_tree_node_lines" | "open_section"
    ) && (intent == "summarize"
        || normalized.contains("这篇文章")
        || normalized.contains("这篇论文")
        || normalized.contains("文章讲")
        || normalized.contains("论文讲")
        || normalized.contains("关于什么")
        || normalized.contains("主要内容")
        || normalized.contains("总结")
        || normalized.contains("概括")
        || normalized.contains("what is this paper about")
        || normalized.contains("what is this article about")
        || normalized.contains("summarize this paper"))
    {
        return "abstract introduction contribution method approach experiments evaluation results conclusion"
            .to_string();
    }
    question.to_string()
}

pub(super) fn retrieval_budget_exhausted(agent_run: &runtime::agent::AgentRunResult) -> bool {
    agent_run
        .retrieval_run
        .trace
        .finalize_gate
        .get("budgetExhausted")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

pub(super) fn retrieval_attempt_count(agent_run: &runtime::agent::AgentRunResult) -> u32 {
    let from_attempts = agent_run.attempts.used;
    if from_attempts > 0 {
        return from_attempts;
    }
    // Backward-compatible fallback: older `AgentRunResult` instances built
    // outside `run_turn_with_activity` (e.g. legacy test fixtures) may lack
    // populated `attempts`, so we still derive from the trace JSON gate.
    agent_run
        .retrieval_run
        .trace
        .finalize_gate
        .get("attempt")
        .and_then(|value| value.as_u64())
        .map(|attempt| attempt.saturating_add(1) as u32)
        .unwrap_or(1)
}

pub(super) fn llm_judge_budget(max_steps: u32, attempt: u32) -> (u32, u32) {
    let remaining_tool_steps = MAX_LLM_JUDGE_STEPS.min(max_steps.saturating_sub(attempt));
    let judge_rounds = remaining_tool_steps.saturating_add(1).max(1);
    (remaining_tool_steps, judge_rounds)
}
