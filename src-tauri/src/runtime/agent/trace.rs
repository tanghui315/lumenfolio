use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::runtime::rag::{
    Citation, RagToolExecutionOutput, RetrievalTrace, RetrievalTraceCandidate,
    RetrievalTraceTreeNode,
};

use super::compact::CompactResult;
use super::protocol::AgentStepKind;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTraceEvent {
    pub step: String,
    pub status: String,
    pub detail: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub title: String,
    pub summary: String,
    pub ts: i64,
    /// Monotonic sequence number assigned by `AgentTrace::renumber_events` right
    /// before the trace is handed to the frontend. UI code can rely on `seq`
    /// being strictly increasing within a single trace, even when the wall-clock
    /// timestamps from different runtimes (M3 rule loop, M4 LLM judge) end up
    /// out of order.
    #[serde(default)]
    pub seq: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<AgentTraceTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<AgentTraceResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTraceTool {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTraceResult {
    pub count: usize,
    pub preview: Vec<AgentTracePreview>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTracePreview {
    pub page: u32,
    pub section_title: Option<String>,
    pub quote: String,
    pub source: String,
}

impl AgentTraceEvent {
    pub fn new(
        event_type: impl Into<String>,
        step: impl Into<String>,
        status: impl Into<String>,
        title: impl Into<String>,
        summary: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            step: step.into(),
            status: status.into(),
            detail: detail.into(),
            event_type: event_type.into(),
            title: title.into(),
            summary: summary.into(),
            ts: unix_millis(),
            seq: 0,
            tool: None,
            result: None,
            judge: None,
        }
    }

    pub fn with_tool(mut self, name: impl Into<String>, args: serde_json::Value) -> Self {
        self.tool = Some(AgentTraceTool {
            name: name.into(),
            args,
        });
        self
    }

    pub fn with_result(
        mut self,
        count: usize,
        preview: Vec<AgentTracePreview>,
        error: Option<String>,
    ) -> Self {
        self.result = Some(AgentTraceResult {
            count,
            preview,
            error,
        });
        self
    }

    pub fn with_judge(mut self, judge: serde_json::Value) -> Self {
        self.judge = Some(judge);
        self
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentEvidenceItem {
    pub citation_id: String,
    pub label: String,
    pub page: u32,
    pub block_id: String,
    pub section_title: Option<String>,
    pub source: String,
    pub quote: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTrace {
    pub run_id: String,
    pub intent: String,
    pub tree_nodes: Vec<RetrievalTraceTreeNode>,
    pub candidates: Vec<RetrievalTraceCandidate>,
    pub finalize_gate: serde_json::Value,
    pub evidence_chain: Vec<AgentEvidenceItem>,
    pub events: Vec<AgentTraceEvent>,
    pub session_summary: Option<String>,
    pub compact: Option<CompactResult>,
}

impl AgentTrace {
    pub fn from_retrieval(
        retrieval_trace: RetrievalTrace,
        citations: &[Citation],
        events: Vec<AgentTraceEvent>,
        session_summary: Option<String>,
        compact: Option<CompactResult>,
    ) -> Self {
        Self {
            run_id: retrieval_trace.run_id,
            intent: retrieval_trace.intent,
            tree_nodes: retrieval_trace.tree_nodes,
            evidence_chain: build_evidence_chain(citations, &retrieval_trace.candidates),
            candidates: retrieval_trace.candidates,
            finalize_gate: retrieval_trace.finalize_gate,
            events,
            session_summary,
            compact,
        }
    }

    pub fn sync_retrieval_view(&mut self, retrieval_trace: RetrievalTrace, citations: &[Citation]) {
        self.run_id = retrieval_trace.run_id;
        self.intent = retrieval_trace.intent;
        self.tree_nodes = retrieval_trace.tree_nodes;
        self.evidence_chain = build_evidence_chain(citations, &retrieval_trace.candidates);
        self.candidates = retrieval_trace.candidates;
        self.finalize_gate = retrieval_trace.finalize_gate;
    }

    /// Rebuild the evidence chain against a replacement citation set (e.g. when an
    /// agentic local-CLI run substitutes the seed citations with the ones it actually
    /// grounded on). Keeps the chain's `citationId`s in sync with the citations the UI
    /// renders, so evidence chips resolve to real bboxes. Reuses the existing
    /// candidates only for section-title fallback.
    pub fn rebuild_evidence_chain(&mut self, citations: &[Citation]) {
        self.evidence_chain = build_evidence_chain(citations, &self.candidates);
    }

    /// Assign a strictly increasing `seq` (starting at 1) to every event in
    /// insertion order. Call this once, right before the trace is serialized
    /// for the frontend, so UI code can rely on a stable monotone ordering
    /// regardless of which runtime stage produced the event.
    pub fn renumber_events(&mut self) {
        for (idx, event) in self.events.iter_mut().enumerate() {
            event.seq = (idx as u64) + 1;
        }
    }
}

fn build_evidence_chain(
    citations: &[Citation],
    candidates: &[RetrievalTraceCandidate],
) -> Vec<AgentEvidenceItem> {
    citations
        .iter()
        .map(|citation| {
            let section_title = citation.section_title.clone().or_else(|| {
                candidates
                    .iter()
                    .find(|candidate| {
                        (!citation.block_id.is_empty() && candidate.block_id == citation.block_id)
                            || (candidate.source == citation.source
                                && normalize_for_match(&candidate.quote)
                                    == normalize_for_match(&citation.quote))
                    })
                    .and_then(|candidate| candidate.section_title.clone())
            });
            AgentEvidenceItem {
                citation_id: citation.id.clone(),
                label: citation.label.clone(),
                page: citation.page,
                block_id: citation.block_id.clone(),
                section_title,
                source: citation.source.clone(),
                quote: citation.quote.clone(),
            }
        })
        .collect()
}

fn normalize_for_match(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn event(step: AgentStepKind, detail: impl Into<String>) -> AgentTraceEvent {
    event_with_status(step, "completed", detail)
}

pub fn skipped_event(step: AgentStepKind, detail: impl Into<String>) -> AgentTraceEvent {
    event_with_status(step, "skipped", detail)
}

pub fn event_with_status(
    step: AgentStepKind,
    status: impl Into<String>,
    detail: impl Into<String>,
) -> AgentTraceEvent {
    let step = step.as_str();
    let detail = detail.into();
    AgentTraceEvent::new(step, step, status, step, detail.clone(), detail)
}

/// Map RAG tool execution status (`ok` / `error` / `…`) to a trace event status.
pub fn trace_status_from_tool_status(status: &str) -> &'static str {
    if status == "error" {
        "error"
    } else {
        "completed"
    }
}

/// Build a short preview list (up to 3 entries) from a tool's trace candidates.
pub fn trace_preview_from_output(output: &RagToolExecutionOutput) -> Vec<AgentTracePreview> {
    output
        .trace_candidates
        .iter()
        .take(3)
        .map(|candidate| AgentTracePreview {
            page: candidate.page,
            section_title: candidate.section_title.clone(),
            quote: candidate.quote.clone(),
            source: candidate.source.clone(),
        })
        .collect()
}

/// Construct a uniform "tool_call" trace event.
///
/// Equivalent to:
/// ```ignore
/// AgentTraceEvent::new("tool_call", tool, "running", title, reason, detail)
///     .with_tool(tool, args)
/// ```
pub fn tool_call_event(
    tool: impl Into<String>,
    args: serde_json::Value,
    title: impl Into<String>,
    reason: impl Into<String>,
    detail: impl Into<String>,
) -> AgentTraceEvent {
    let tool = tool.into();
    AgentTraceEvent::new("tool_call", tool.clone(), "running", title, reason, detail)
        .with_tool(tool, args)
}

/// Construct a uniform "tool_result" trace event populated from a RAG tool
/// execution output. `step` is the agent step kind name (e.g.
/// `AgentStepKind::OpenPages.as_str()` or `step_kind_for_tool(tool).as_str()`).
pub fn tool_result_event(
    output: &RagToolExecutionOutput,
    step: impl Into<String>,
    detail: impl Into<String>,
) -> AgentTraceEvent {
    let tool = output.tool_call.tool.clone();
    let result_count = output.tool_call.result_count;
    AgentTraceEvent::new(
        "tool_result",
        step,
        trace_status_from_tool_status(&output.tool_call.status),
        format!("{tool} result"),
        format!("{tool} returned {result_count} results"),
        detail,
    )
    .with_tool(tool, output.tool_call.input.clone())
    .with_result(
        result_count,
        trace_preview_from_output(output),
        output.tool_call.error.clone(),
    )
}

fn unix_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_event(detail: &str) -> AgentTraceEvent {
        AgentTraceEvent::new("e", "step", "completed", "title", detail, detail)
    }

    #[test]
    fn renumber_events_assigns_monotonic_sequence() {
        let mut trace = AgentTrace {
            run_id: "run".into(),
            intent: String::new(),
            tree_nodes: Vec::new(),
            candidates: Vec::new(),
            finalize_gate: serde_json::Value::Null,
            evidence_chain: Vec::new(),
            events: vec![dummy_event("a"), dummy_event("b"), dummy_event("c")],
            session_summary: None,
            compact: None,
        };
        // Mimic the case where two stages set non-monotonic timestamps.
        trace.events[0].ts = 100;
        trace.events[1].ts = 50;
        trace.events[2].ts = 200;
        trace.renumber_events();
        assert_eq!(trace.events[0].seq, 1);
        assert_eq!(trace.events[1].seq, 2);
        assert_eq!(trace.events[2].seq, 3);
    }

    fn dummy_citation(id: &str, page: u32) -> Citation {
        Citation {
            id: id.to_string(),
            label: format!("[{id}]"),
            page,
            block_id: format!("blk-{id}"),
            section_title: None,
            quote: "evidence".to_string(),
            bbox_list: serde_json::json!([[0.1, 0.1, 0.2, 0.2]]),
            document_id: "doc".to_string(),
            source: "fts".to_string(),
        }
    }

    #[test]
    fn rebuild_evidence_chain_syncs_ids_with_new_citations() {
        let mut trace = AgentTrace {
            run_id: "run".into(),
            intent: String::new(),
            tree_nodes: Vec::new(),
            candidates: Vec::new(),
            finalize_gate: serde_json::Value::Null,
            // Stale chain from a prior (seed) citation set.
            evidence_chain: build_evidence_chain(&[dummy_citation("seed", 1)], &[]),
            events: Vec::new(),
            session_summary: None,
            compact: None,
        };
        assert_eq!(trace.evidence_chain[0].citation_id, "seed");

        // Replacing with the agent's citations must re-point the chain so the UI can
        // resolve each chip back to a real citation (and its bbox).
        let agent_cites = vec![dummy_citation("x", 6), dummy_citation("y", 7)];
        trace.rebuild_evidence_chain(&agent_cites);
        let chain_ids: Vec<_> = trace
            .evidence_chain
            .iter()
            .map(|e| e.citation_id.as_str())
            .collect();
        assert_eq!(chain_ids, vec!["x", "y"]);
    }

    #[test]
    fn renumber_events_is_idempotent_for_subsequent_calls() {
        let mut trace = AgentTrace {
            run_id: "run".into(),
            intent: String::new(),
            tree_nodes: Vec::new(),
            candidates: Vec::new(),
            finalize_gate: serde_json::Value::Null,
            evidence_chain: Vec::new(),
            events: vec![dummy_event("a"), dummy_event("b")],
            session_summary: None,
            compact: None,
        };
        trace.renumber_events();
        trace.events.push(dummy_event("c"));
        trace.renumber_events();
        assert_eq!(
            trace.events.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }
}
