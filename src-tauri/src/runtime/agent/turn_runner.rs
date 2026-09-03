use rusqlite::Connection;

use crate::runtime::rag::{self, Citation, RetrievalRequest, RetrievalRun};

use super::compact::{compact_session, CompactResult};
use super::context::build_chat_context;
use super::decision::{FinalizeStatus, RetrievalAttempts};
use super::ledger::RetrievalLedger;
use super::memory::{preview_text, summarize_citations, RecentTurnSummary};
use super::policy::{finalize_citations, FinalizeDecision};
use super::protocol::{AgentIntent, AgentStepKind};
use super::session::AgentSessionStore;
use super::trace::{
    event, skipped_event, tool_call_event, tool_result_event, trace_preview_from_output,
    trace_status_from_tool_status, AgentTrace, AgentTraceEvent,
};

const DEFAULT_MAX_RETRIEVAL_STEPS: u32 = 20;
const MAX_RETRIEVAL_STEPS: u32 = 20;
/// No-progress early exit: after this many retrieval rounds with ZERO accumulated
/// evidence, stop instead of grinding to the step limit. More rounds of the same
/// library-wide tools won't conjure evidence that isn't there, so this bounds the
/// wasted work (and the "I searched 20 times, found nothing" refusal) for off-topic,
/// empty-library, or missed-smalltalk messages. Any genuinely retrievable evidence —
/// a focus document's own text — already surfaces in the seed, so a real question is
/// never cut short by this.
const MAX_EMPTY_RETRIEVAL_ATTEMPTS: u32 = 3;

pub struct AgentRunRequest<'a> {
    pub document_id: &'a str,
    /// Conversation this turn belongs to. Used to key in-memory working memory
    /// (recent turns / selection / sections) so memory follows the session, not
    /// the focus document. Retrieval still targets `document_id`.
    pub session_key: &'a str,
    /// The documents this turn may search via `documentId` routing — the focus
    /// doc plus every other visible workspace doc (capped at
    /// WORKSPACE_MANIFEST_MAX_DOCS). Forms the dispatch whitelist in
    /// run_answerability_loop, so a tool call may target any of them.
    pub visible_document_ids: Vec<&'a str>,
    pub question: &'a str,
    pub provider_id: Option<&'a str>,
    pub context_budget: crate::model_catalog::ModelContextBudget,
    pub selected_text: Option<&'a str>,
    pub selected_block_id: Option<&'a str>,
    pub selected_bbox_list: Option<serde_json::Value>,
    pub page: Option<u32>,
    pub page_mode: Option<&'a str>,
    pub page_source: Option<&'a str>,
    pub max_retrieval_steps: Option<u32>,
    pub retrieval_attempt_offset: u32,
}

pub struct AgentRunResult {
    pub retrieval_run: RetrievalRun,
    pub trace: AgentTrace,
    pub session_context: String,
    /// Single source of truth for how many retrieval steps the agent run used.
    /// Read this instead of recomputing from `retrieval_run.trace.finalize_gate`.
    pub attempts: RetrievalAttempts,
    /// Unified "what have I already looked at" record for this turn (tool-call
    /// dedupe + coverage + judge feedback). Populated by the M3 loop and handed
    /// to the M4 LLM-judge loop so it inherits the M3 coverage instead of
    /// re-deriving its own dedupe state.
    pub ledger: RetrievalLedger,
}

pub fn run_turn_with_activity<F>(
    conn: &Connection,
    sessions: &AgentSessionStore,
    request: AgentRunRequest<'_>,
    mut on_event: F,
) -> Result<AgentRunResult, String>
where
    F: FnMut(&AgentTraceEvent),
{
    let snapshot = sessions.snapshot(request.session_key);
    let compacted = snapshot.as_ref().map(compact_session).unwrap_or_else(|| {
        compact_session(&empty_snapshot(request.session_key, request.provider_id))
    });
    let intent = AgentIntent::infer(request.question);
    let max_retrieval_attempts = request
        .max_retrieval_steps
        .unwrap_or(DEFAULT_MAX_RETRIEVAL_STEPS)
        .clamp(1, MAX_RETRIEVAL_STEPS);
    let mut loop_events = Vec::new();
    let (mut retrieval_run, finalize, ledger) = run_answerability_loop(
        conn,
        &request,
        intent,
        max_retrieval_attempts,
        request.retrieval_attempt_offset,
        &mut loop_events,
        &mut on_event,
    )?;

    let session_context = build_chat_context(snapshot.as_ref(), &compacted, &retrieval_run);
    let mut finalize_gate = serde_json::to_value(&finalize)
        .map_err(|err| format!("Failed to encode finalize decision: {err}"))?;
    if let Some(object) = finalize_gate.as_object_mut() {
        object.insert(
            "contextBudget".to_string(),
            serde_json::to_value(&request.context_budget)
                .map_err(|err| format!("Failed to encode context budget: {err}"))?,
        );
    }
    retrieval_run.trace.finalize_gate = finalize_gate;
    let events = build_agent_events(
        &retrieval_run,
        intent,
        compacted.compact.clone(),
        !session_context.trim().is_empty(),
        loop_events,
    );
    let trace = AgentTrace::from_retrieval(
        retrieval_run.trace.clone(),
        &retrieval_run.citations,
        events,
        if session_context.trim().is_empty() {
            None
        } else {
            Some(session_context.clone())
        },
        Some(compacted.compact.clone()),
    );

    let attempts = RetrievalAttempts::new(
        finalize.attempt.saturating_add(1),
        max_retrieval_attempts,
        finalize.budget_exhausted,
    );

    Ok(AgentRunResult {
        retrieval_run,
        trace,
        session_context,
        attempts,
        ledger,
    })
}

fn run_answerability_loop<F>(
    conn: &Connection,
    request: &AgentRunRequest<'_>,
    intent: AgentIntent,
    max_attempts: u32,
    attempt_offset: u32,
    loop_events: &mut Vec<super::trace::AgentTraceEvent>,
    on_event: &mut F,
) -> Result<(RetrievalRun, FinalizeDecision, RetrievalLedger), String>
where
    F: FnMut(&AgentTraceEvent),
{
    let mut attempt = attempt_offset;
    let max_attempts = attempt_offset.saturating_add(max_attempts);
    let mut accumulated_citations = Vec::new();
    let mut ledger = RetrievalLedger::new();
    let start_event = AgentTraceEvent::new(
        "retrieval_round",
        AgentStepKind::SearchChunks.as_str(),
        "running",
        "Gathering initial evidence",
        "Inspecting structure, sections, text search, and nearby pages",
        format!(
            "attempt={} query={}",
            attempt.saturating_add(1),
            request.question
        ),
    );
    on_event(&start_event);

    let mut retrieval_run = rag::build_retrieval_run(
        conn,
        RetrievalRequest {
            document_id: request.document_id,
            question: request.question,
            retrieval_query: None,
            selected_text: request.selected_text,
            selected_block_id: request.selected_block_id,
            selected_bbox_list: request.selected_bbox_list.clone(),
            client_context: None,
            page: request.page,
            page_mode: request.page_mode,
            page_source: request.page_source,
            context_budget: request.context_budget.clone(),
            force_document_start: false,
        },
    )?;
    // The seed runs only the cheap base chain (inspect_tree → open_section →
    // search_chunks → open_pages); table/visual evidence is fetched on demand by
    // the answerability loop. Collapse the seed tool calls into a single
    // `retrieval_seed` event so the trace does not open with a burst of
    // `tool_result` rows.
    let seed_tool_summary = retrieval_run
        .trace
        .tool_calls
        .iter()
        .map(|call| format!("{}={}", call.tool, call.result_count))
        .collect::<Vec<_>>()
        .join(" ");
    let round_event = AgentTraceEvent::new(
        "retrieval_round",
        AgentStepKind::SearchChunks.as_str(),
        "completed",
        "Seed evidence gathered",
        format!(
            "Seeded {} citations from a thin base retrieval",
            retrieval_run.citations.len()
        ),
        format!(
            "retrieval_seed citations={} tools=[{seed_tool_summary}]",
            retrieval_run.citations.len()
        ),
    );
    on_event(&round_event);
    loop_events.push(round_event);
    rag::merge_retrieval_citations_with_budget(
        &mut accumulated_citations,
        &retrieval_run.citations,
        &request.context_budget,
    );
    ledger.record_coverage(&retrieval_run.citations);

    loop {
        rag::apply_retrieval_citations(&mut retrieval_run, accumulated_citations.clone());
        let decision = finalize_citations(
            request.question,
            intent.as_str(),
            &retrieval_run.citations,
            attempt,
            max_attempts,
        );
        let next_tool_name = decision
            .next_tool_call
            .as_ref()
            .map(|call| call.tool.as_str())
            .unwrap_or("none");
        let detail = format!(
            "attempt={} status={} reason={} next_tool={}",
            attempt + 1,
            decision.status,
            decision.reason,
            next_tool_name
        );
        let finalize_event = AgentTraceEvent::new(
            "judge_result",
            AgentStepKind::FinalizeAnswer.as_str(),
            "completed",
            "M3 rule evidence check",
            format!("status={} next_tool={}", decision.status, next_tool_name),
            detail,
        )
        .with_judge(serde_json::to_value(&decision).unwrap_or_else(|_| serde_json::json!({})));
        on_event(&finalize_event);
        loop_events.push(finalize_event);

        if !decision.needs_more_evidence {
            return Ok((retrieval_run, decision, ledger));
        }

        // No-progress early exit (Layer 2): several rounds have yielded zero evidence,
        // so more of the same retrieval won't help. Stop now instead of exhausting the
        // budget and refusing after 20 empty rounds. Only fires on a truly empty
        // accumulation, so it never truncates a question that found anything at all.
        let empty_rounds = attempt.saturating_sub(attempt_offset);
        if accumulated_citations.is_empty() && empty_rounds >= MAX_EMPTY_RETRIEVAL_ATTEMPTS {
            let mut terminal = decision.clone();
            terminal.status = FinalizeStatus::Insufficient;
            terminal.needs_more_evidence = false;
            terminal.next_tool_call = None;
            terminal.reason = format!(
                "{} No evidence surfaced after {} rounds; stopping early instead of exhausting the retrieval budget.",
                terminal.reason.trim(),
                empty_rounds
            );
            let skipped = skipped_event(
                AgentStepKind::SearchChunks,
                format!("no evidence after {empty_rounds} rounds; early exit"),
            );
            on_event(&skipped);
            loop_events.push(skipped);
            return Ok((retrieval_run, terminal, ledger));
        }

        let fallback_query = broad_context_query(intent, attempt);
        let Some((tool, args)) = next_tool_parts(&decision) else {
            return Ok((retrieval_run, decision, ledger));
        };
        let (tool, args, repeated_tool) = if ledger.record_tool_call(tool, &args) {
            (tool, args, false)
        } else if is_header_lookup(tool, &args) {
            let mut terminal_decision = decision.clone();
            terminal_decision.status = FinalizeStatus::Insufficient;
            terminal_decision.needs_more_evidence = false;
            terminal_decision.next_tool_call = None;
            terminal_decision.reason = format!(
                "{} Header lookup already ran without producing header evidence; the document index likely needs to be rebuilt with header roles.",
                terminal_decision.reason.trim()
            );
            let skipped = skipped_event(
                AgentStepKind::OpenPages,
                "header lookup already attempted; stopping instead of broad-search fallback"
                    .to_string(),
            );
            on_event(&skipped);
            loop_events.push(skipped);
            return Ok((retrieval_run, terminal_decision, ledger));
        } else {
            (
                "search_chunks",
                serde_json::json!({ "query": fallback_query, "limit": 8 }),
                true,
            )
        };
        let call_event = tool_call_event(
            tool,
            args.clone(),
            format!("Calling {tool}"),
            decision
                .next_tool_call
                .as_ref()
                .map(|call| call.reason.as_str())
                .unwrap_or("Need more evidence"),
            format!("tool={tool} args={args} fallback_query={fallback_query}"),
        );
        on_event(&call_event);

        let output = rag::execute_rag_tool_call_for_capabilities(
            conn,
            request.document_id,
            &request.visible_document_ids,
            tool,
            &args,
            fallback_query,
            rag::RagToolCapabilities {
                vision_enabled: false,
                web_enabled: false,
                max_quote_chars: request.context_budget.max_quote_chars,
            },
        );
        let tool_event = AgentTraceEvent::new(
            "tool_result",
            step_kind_for_tool(&output.tool_call.tool).as_str(),
            trace_status_from_tool_status(&output.tool_call.status),
            format!("{} result", output.tool_call.tool),
            format!(
                "{} returned {} results{}",
                output.tool_call.tool,
                output.tool_call.result_count,
                if repeated_tool { " after fallback" } else { "" }
            ),
            format!(
                "tool={} status={} results={} reason={}{}",
                output.tool_call.tool,
                output.tool_call.status,
                output.tool_call.result_count,
                decision
                    .next_tool_call
                    .as_ref()
                    .map(|call| call.reason.as_str())
                    .unwrap_or("legacy fallback"),
                if repeated_tool {
                    " repeated_tool_fallback=true"
                } else {
                    ""
                }
            ),
        )
        .with_tool(
            output.tool_call.tool.clone(),
            output.tool_call.input.clone(),
        )
        .with_result(
            output.tool_call.result_count,
            trace_preview_from_output(&output),
            output.tool_call.error.clone(),
        );
        on_event(&tool_event);
        loop_events.push(tool_event);
        rag::merge_retrieval_citations_with_budget(
            &mut accumulated_citations,
            &output.citations,
            &request.context_budget,
        );
        ledger.record_coverage(&output.citations);
        rag::apply_tool_execution(&mut retrieval_run, &output);
        if let Some(page) = table_fallback_page_for_m3(&output, &accumulated_citations) {
            let page_args = serde_json::json!({
                "page": page,
                "mode": "full",
                "limit": 40
            });
            let page_call_event = tool_call_event(
                "open_pages",
                page_args.clone(),
                "Calling open_pages",
                "Opening the source page because structured table extraction looked sparse or noisy",
                format!("tool=open_pages args={page_args} fallback_query={fallback_query}"),
            );
            on_event(&page_call_event);

            let page_output = rag::execute_rag_tool_call_for_capabilities(
                conn,
                request.document_id,
                &request.visible_document_ids,
                "open_pages",
                &page_args,
                fallback_query,
                rag::RagToolCapabilities {
                    vision_enabled: false,
                    web_enabled: false,
                    max_quote_chars: request.context_budget.max_quote_chars,
                },
            );
            let page_tool_event = tool_result_event(
                &page_output,
                AgentStepKind::OpenPages.as_str(),
                format!(
                    "tool={} status={} results={} reason=structured table evidence was incomplete",
                    page_output.tool_call.tool,
                    page_output.tool_call.status,
                    page_output.tool_call.result_count
                ),
            );
            on_event(&page_tool_event);
            loop_events.push(page_tool_event);
            rag::merge_retrieval_citations_with_budget(
                &mut accumulated_citations,
                &page_output.citations,
                &request.context_budget,
            );
            ledger.record_coverage(&page_output.citations);
            rag::apply_tool_execution(&mut retrieval_run, &page_output);
        }
        attempt += 1;
    }
}

fn table_fallback_page_for_m3(
    output: &rag::RagToolExecutionOutput,
    accumulated_citations: &[Citation],
) -> Option<u32> {
    let page = super::table_fallback::sparse_open_table_page(output, None)?;
    if accumulated_citations.iter().any(|citation| {
        citation.source == "open_pages"
            && citation.page == page
            && citation
                .section_title
                .as_deref()
                .is_some_and(|title| title.contains(" lines "))
    }) {
        return None;
    }
    Some(page)
}

fn is_header_lookup(tool: &str, args: &serde_json::Value) -> bool {
    tool == "open_pages"
        && args
            .get("page")
            .and_then(|value| value.as_u64())
            .is_some_and(|page| page == 1)
        && args
            .get("mode")
            .and_then(|value| value.as_str())
            .is_some_and(|mode| mode == "header")
}

fn next_tool_parts(decision: &FinalizeDecision) -> Option<(&str, serde_json::Value)> {
    let call = decision.next_tool_call.as_ref()?;
    rag::is_registered_rag_tool(&call.tool).then_some((call.tool.as_str(), call.args.clone()))
}

fn step_kind_for_tool(tool: &str) -> AgentStepKind {
    match tool {
        "inspect_tree" => AgentStepKind::InspectTree,
        "open_section" => AgentStepKind::OpenSection,
        "open_pages" => AgentStepKind::OpenPages,
        "search_chunks" => AgentStepKind::SearchChunks,
        _ => AgentStepKind::SearchChunks,
    }
}

fn broad_context_query(intent: AgentIntent, attempt: u32) -> &'static str {
    let queries: &[&str] = match intent {
        AgentIntent::Compare => &[
            "comparison difference evaluation results",
            "baseline ablation performance limitation",
            "experiment table metric benchmark",
        ],
        AgentIntent::Locate => &[
            "definition location section page",
            "method term introduced described",
            "figure table algorithm reference",
        ],
        AgentIntent::Summarize | AgentIntent::Explain => &[
            "abstract introduction contribution method approach experiment result conclusion",
            "problem motivation proposed method contribution",
            "system architecture design implementation evaluation",
            "limitations discussion conclusion future work",
            "results experiments benchmark findings",
        ],
    };
    queries[(attempt as usize).saturating_sub(1) % queries.len()]
}

pub struct CompletedTurnRecord<'a> {
    pub session_key: &'a str,
    pub provider_id: Option<&'a str>,
    pub question: &'a str,
    pub answer: &'a str,
    pub selected_text: Option<&'a str>,
    pub citations: &'a [Citation],
    pub trace: &'a AgentTrace,
}

pub fn record_completed_turn(sessions: &AgentSessionStore, record: CompletedTurnRecord<'_>) {
    let tree_titles = record
        .trace
        .tree_nodes
        .iter()
        .take(4)
        .map(|node| node.title.clone())
        .collect::<Vec<_>>();
    sessions.record_turn(
        record.session_key,
        record.provider_id.map(str::to_string),
        RecentTurnSummary {
            question: preview_text(record.question, 220),
            answer_preview: preview_text(record.answer, 260),
            citations: summarize_citations(record.citations, 4),
            selected_text_preview: record
                .selected_text
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| preview_text(value, 180)),
            tree_titles,
        },
    );
}

pub struct RestoredTurnRecord<'a> {
    pub session_key: &'a str,
    pub provider_id: Option<&'a str>,
    pub question: &'a str,
    pub answer: &'a str,
    pub selected_text: Option<&'a str>,
    pub citations: &'a [Citation],
    pub tree_titles: Vec<String>,
}

pub fn record_restored_turn(sessions: &AgentSessionStore, record: RestoredTurnRecord<'_>) {
    sessions.record_turn(
        record.session_key,
        record.provider_id.map(str::to_string),
        RecentTurnSummary {
            question: preview_text(record.question, 220),
            answer_preview: preview_text(record.answer, 260),
            citations: summarize_citations(record.citations, 4),
            selected_text_preview: record
                .selected_text
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| preview_text(value, 180)),
            tree_titles: record.tree_titles,
        },
    );
}

fn build_agent_events(
    _retrieval_run: &RetrievalRun,
    intent: AgentIntent,
    compact: CompactResult,
    has_session_context: bool,
    loop_events: Vec<super::trace::AgentTraceEvent>,
) -> Vec<super::trace::AgentTraceEvent> {
    let mut events = Vec::new();
    if compact.summary_generated || compact.summarized_turns > 0 || has_session_context {
        events.push(event(
            AgentStepKind::SessionCompact,
            format!(
                "retained {} recent turns, summarized {} turns",
                compact.retained_recent_turns, compact.summarized_turns
            ),
        ));
    } else {
        events.push(skipped_event(
            AgentStepKind::SessionCompact,
            "no prior turns needed compaction".to_string(),
        ));
    }
    events.push(event(
        AgentStepKind::IntentClassify,
        format!("intent={}", intent.as_str()),
    ));
    events.extend(loop_events);
    events
}

#[cfg(test)]
fn trace_tool_result_count(retrieval_run: &RetrievalRun, tool: &str) -> Option<usize> {
    let mut saw_tool = false;
    let count = retrieval_run
        .trace
        .tool_calls
        .iter()
        .filter(|call| call.tool == tool)
        .map(|call| {
            saw_tool = true;
            call.result_count
        })
        .sum();

    saw_tool.then_some(count)
}

fn empty_snapshot(
    document_id: &str,
    provider_id: Option<&str>,
) -> super::memory::SessionMemorySnapshot {
    super::memory::SessionMemorySnapshot {
        document_id: document_id.to_string(),
        provider_id: provider_id.map(str::to_string),
        recent_turns: Vec::new(),
        selection: None,
        sections: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::rag::{RetrievalTrace, RetrievalTraceToolCall};

    #[test]
    fn trace_tool_result_count_sums_repeated_tool_calls() {
        let run = RetrievalRun {
            id: "run".to_string(),
            intent: "explain".to_string(),
            prompt_context: String::new(),
            citations: Vec::new(),
            trace: RetrievalTrace {
                run_id: "run".to_string(),
                intent: "explain".to_string(),
                tree_nodes: Vec::new(),
                candidates: Vec::new(),
                tool_calls: vec![
                    tool_call("open_section", 0),
                    tool_call("search_chunks", 2),
                    tool_call("open_section", 5),
                ],
                finalize_gate: serde_json::json!({}),
            },
            context_budget: crate::model_catalog::ModelContextBudget::default(),
        };

        assert_eq!(trace_tool_result_count(&run, "open_section"), Some(5));
        assert_eq!(trace_tool_result_count(&run, "search_chunks"), Some(2));
        assert_eq!(trace_tool_result_count(&run, "open_pages"), None);
    }

    #[test]
    fn header_lookup_signature_is_detected() {
        assert!(is_header_lookup(
            "open_pages",
            &serde_json::json!({ "page": 1, "mode": "header" })
        ));
        assert!(!is_header_lookup(
            "open_pages",
            &serde_json::json!({ "page": 1, "mode": "overview" })
        ));
        assert!(!is_header_lookup(
            "search_chunks",
            &serde_json::json!({ "query": "author" })
        ));
    }

    #[test]
    fn next_tool_parts_rejects_unregistered_tools() {
        let mut decision = FinalizeDecision {
            status: FinalizeStatus::NeedsMoreEvidence,
            citation_count: 0,
            needs_more_evidence: true,
            reason: "test".to_string(),
            missing: Vec::new(),
            next_tool_call: Some(super::super::policy::NextToolCall {
                tool: "not_a_tool".to_string(),
                args: serde_json::json!({}),
                reason: "unregistered".to_string(),
            }),
            attempt: 0,
            max_attempts: 3,
            budget_exhausted: false,
            runtime: super::super::decision::FinalizeRuntime::M3RuleGuard,
        };

        assert!(next_tool_parts(&decision).is_none());
        decision.next_tool_call = Some(super::super::policy::NextToolCall {
            tool: "open_section".to_string(),
            args: serde_json::json!({ "query": "method" }),
            reason: "test".to_string(),
        });

        let (tool, args) = next_tool_parts(&decision).expect("registered tool");
        assert_eq!(tool, "open_section");
        assert_eq!(args["query"], serde_json::json!("method"));
    }

    fn tool_call(tool: &str, result_count: usize) -> RetrievalTraceToolCall {
        RetrievalTraceToolCall {
            tool: tool.to_string(),
            status: "ok".to_string(),
            input: serde_json::json!({}),
            result_count,
            error: None,
        }
    }
}
