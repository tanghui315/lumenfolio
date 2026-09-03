use serde::Serialize;

use crate::runtime::agent::decision::{FinalizeRuntime, FinalizeStatus};
use crate::runtime::agent::lexicon::requested_table_number;
use crate::runtime::rag::Citation;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NextToolCall {
    pub tool: String,
    pub args: serde_json::Value,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FinalizeDecision {
    pub status: FinalizeStatus,
    pub citation_count: usize,
    pub needs_more_evidence: bool,
    pub reason: String,
    pub missing: Vec<String>,
    pub next_tool_call: Option<NextToolCall>,
    pub attempt: u32,
    pub max_attempts: u32,
    pub budget_exhausted: bool,
    pub runtime: FinalizeRuntime,
}

pub fn finalize_citations(
    question: &str,
    intent: &str,
    citations: &[Citation],
    attempt: u32,
    max_attempts: u32,
) -> FinalizeDecision {
    let judge = AnswerabilityJudge::default();
    let mut decision = judge.decide(question, intent, citations);
    finish_decision(&mut decision, citations.len(), attempt, max_attempts);
    decision
}

#[derive(Default)]
struct AnswerabilityJudge {
    guard: RuleGuard,
    heuristic: HeuristicAnswerabilityJudge,
}

impl AnswerabilityJudge {
    fn decide(&self, question: &str, intent: &str, citations: &[Citation]) -> FinalizeDecision {
        let needs = QuestionNeeds::from_question(question, intent);
        let facts = EvidenceFacts::from_citations(citations);
        if let Some(decision) = self.guard.decide(question, &needs, &facts, citations) {
            return decision;
        }
        self.heuristic.decide(&needs, &facts)
    }
}

#[derive(Default)]
struct RuleGuard;

impl RuleGuard {
    fn decide(
        &self,
        question: &str,
        needs: &QuestionNeeds,
        facts: &EvidenceFacts,
        citations: &[Citation],
    ) -> Option<FinalizeDecision> {
        if facts.has_selection {
            return Some(answerable(
                "User-selected text is direct evidence for the question.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.header && (facts.has_header || facts.has_author_evidence) {
            return Some(answerable(
                "The retrieved document header contains the requested paper metadata.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.reference && facts.has_reference {
            return Some(answerable(
                "The retrieved evidence contains reference or related-work context.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.reference {
            return Some(needs_more(
                "The question asks for references or supporting source context, but no reference-related evidence is available.",
                vec!["reference or related-work evidence"],
                NextToolCall {
                    tool: "open_section".to_string(),
                    args: serde_json::json!({
                        "query": "references related work citation bibliography source prior work",
                        "perSectionLimit": 8
                    }),
                    reason: "Open references or related-work sections for source evidence."
                        .to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.overview
            && facts.has_overview
            && facts.evidence_chars >= 280
            && overview_evidence_can_answer(question, needs)
        {
            return Some(answerable(
                "The retrieved page overview contains enough title/abstract context.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.overview && !facts.has_overview {
            return Some(needs_more(
                "The question asks for a document-level overview, but no page overview evidence is available.",
                vec!["title and abstract"],
                NextToolCall {
                    tool: "open_pages".to_string(),
                    args: serde_json::json!({ "page": 1, "mode": "overview" }),
                    reason: "Open the first-page overview for title and abstract context."
                        .to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.location && has_location_evidence(citations, asks_section_location(question)) {
            return Some(answerable(
                "The retrieved evidence identifies the requested page or section location.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.location {
            return Some(needs_more(
                "The question asks where content appears, but the available evidence does not identify a page or section strongly enough.",
                vec!["page or section location"],
                NextToolCall {
                    tool: "open_section".to_string(),
                    args: serde_json::json!({
                        "query": "section page location introduced described definition method reference",
                        "perSectionLimit": 6
                    }),
                    reason: "Open likely sections so the answer can cite a concrete location."
                        .to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if let Some(table_number) = requested_table_number(question) {
            if has_open_table_evidence_for_number(citations, &table_number)
                || has_current_view_table_evidence_for_number(citations, &table_number)
            {
                return Some(answerable(
                    "The requested numbered table is available in structured or current-view evidence; semantic sufficiency must be checked by the LLM evidence judge when available.",
                    FinalizeRuntime::M3RuleGuard,
                ));
            }
            return Some(needs_more(
                "The question asks about a specific numbered table, but that full table has not been opened yet.",
                vec!["full requested table evidence"],
                NextToolCall {
                    tool: "open_table".to_string(),
                    args: serde_json::json!({
                        "tableNumber": table_number,
                        "query": question,
                        "limit": 40
                    }),
                    reason: "Open the requested numbered table before generic definition or broad context search.".to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.method && !facts.has_method {
            return Some(needs_more(
                "The question asks about the method or framework, but no method-section evidence is available.",
                vec!["method section"],
                NextToolCall {
                    tool: "open_section".to_string(),
                    args: serde_json::json!({
                        "query": "method approach methodology algorithm framework",
                        "perSectionLimit": 10
                    }),
                    reason: "Open the method-related section for grounded method evidence."
                        .to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.method && needs.experiment && !facts.has_experiment {
            return Some(needs_more(
                "The question asks to connect method design with experiments or results, but no evaluation-section evidence is available.",
                vec!["experiment or result section"],
                NextToolCall {
                    tool: "open_section".to_string(),
                    args: serde_json::json!({
                        "query": "experiments evaluation results benchmark",
                        "perSectionLimit": 10
                    }),
                    reason: "Open the experiment/result section so the method answer can cite evaluation evidence.".to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.experiment && !facts.has_experiment {
            return Some(needs_more(
                "The question asks about experiments or results, but no evaluation-section evidence is available.",
                vec!["experiment or result section"],
                NextToolCall {
                    tool: if needs.table {
                        "search_table_facts".to_string()
                    } else {
                        "open_section".to_string()
                    },
                    args: serde_json::json!({
                        "query": "experiments evaluation results benchmark metric score SOTA table",
                        "perSectionLimit": 10,
                        "limit": 16
                    }),
                    reason: if needs.table {
                        "Search structured table facts for benchmark scores and metrics."
                            .to_string()
                    } else {
                        "Open the experiment/result section for evaluation evidence."
                            .to_string()
                    },
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.table && !facts.has_table {
            return Some(needs_more(
                "The question asks for table-like metrics or benchmark scores, but no structured table fact evidence is available.",
                vec!["table facts or benchmark scores"],
                NextToolCall {
                    tool: "search_table_facts".to_string(),
                    args: serde_json::json!({
                        "query": "SOTA benchmark metric score performance results table",
                        "limit": 16
                    }),
                    reason: "Search normalized table facts before relying on prose evidence."
                        .to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.table && facts.has_table && !facts.has_open_table {
            return Some(needs_more(
                "The question asks for table-like metrics or benchmark scores; structured facts were found, but the full table has not been opened for row/column coverage.",
                vec!["full table row and column evidence"],
                NextToolCall {
                    tool: "open_table".to_string(),
                    args: serde_json::json!({
                        "query": question,
                        "limit": 40
                    }),
                    reason: "Open the candidate table so the evidence judge can verify target table, row, and metric coverage.".to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.table && facts.has_open_table {
            return Some(answerable(
                "A structured table has been opened; semantic sufficiency must be checked by the LLM evidence judge when available.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.figure && !facts.has_figure {
            return Some(needs_more(
                "The question asks about a figure or table, but no figure/table evidence is available.",
                vec!["figure or table caption"],
                NextToolCall {
                    tool: "inspect_visuals".to_string(),
                    args: serde_json::json!({
                        "query": "figure table chart caption",
                        "limit": 8
                    }),
                    reason: "Inspect indexed visual assets and captions.".to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if facts.has_current_view && facts.evidence_chars >= 280 {
            return Some(answerable(
                "Current-view page evidence is available; semantic sufficiency must be checked by the LLM evidence judge when available.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.definition && has_definition_evidence(question, citations) {
            return Some(answerable(
                "The retrieved evidence contains a definition or introductory explanation.",
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.definition {
            return Some(needs_more(
                "The question asks for a definition, but no definitional evidence is available.",
                vec!["definition or introductory explanation"],
                NextToolCall {
                    tool: "search_chunks".to_string(),
                    args: serde_json::json!({
                        "query": "definition defined means refers to called proposed introduced",
                        "limit": 8
                    }),
                    reason: "Search for definition-style passages and first explanations."
                        .to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if facts.citation_count == 0 {
            return Some(needs_more(
                "No evidence has been retrieved yet.",
                vec!["document evidence"],
                NextToolCall {
                    tool: "open_pages".to_string(),
                    args: serde_json::json!({ "page": 1, "mode": "overview" }),
                    reason: "Open the document start to obtain grounding evidence.".to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        if needs.header {
            return Some(needs_more(
                "The question asks for document header metadata, but no header evidence is available.",
                vec!["title", "authors", "affiliations"],
                NextToolCall {
                    tool: "open_pages".to_string(),
                    args: serde_json::json!({ "page": 1, "mode": "header" }),
                    reason: "Open the first-page header for paper metadata.".to_string(),
                },
                FinalizeRuntime::M3RuleGuard,
            ));
        }
        None
    }
}

#[derive(Default)]
struct HeuristicAnswerabilityJudge;

impl HeuristicAnswerabilityJudge {
    fn decide(&self, needs: &QuestionNeeds, facts: &EvidenceFacts) -> FinalizeDecision {
        if needs.overview && facts.has_overview && facts.evidence_chars >= 280 {
            answerable(
                "The retrieved page overview contains enough title/abstract context.",
                FinalizeRuntime::M3HeuristicJudge,
            )
        } else if needs.overview && !facts.has_overview {
            needs_more(
                "The question asks for a document-level overview, but no page overview evidence is available.",
                vec!["title and abstract"],
                NextToolCall {
                    tool: "open_pages".to_string(),
                    args: serde_json::json!({ "page": 1, "mode": "overview" }),
                    reason: "Open the first-page overview for title and abstract context."
                        .to_string(),
                },
                FinalizeRuntime::M3HeuristicJudge,
            )
        } else if matches!(needs.intent, "explain" | "summarize") && facts.evidence_chars < 360 {
            needs_more(
                "The retrieved evidence is too thin for a grounded explanation.",
                vec!["supporting passage"],
                NextToolCall {
                    tool: "search_chunks".to_string(),
                    args: serde_json::json!({ "query": "broad_context", "page": 1 }),
                    reason: "Broaden retrieval to collect supporting passages.".to_string(),
                },
                FinalizeRuntime::M3HeuristicJudge,
            )
        } else {
            answerable(
                "Retrieved evidence is sufficient for a grounded answer.",
                FinalizeRuntime::M3HeuristicJudge,
            )
        }
    }
}

struct QuestionNeeds<'a> {
    intent: &'a str,
    overview: bool,
    header: bool,
    method: bool,
    experiment: bool,
    figure: bool,
    table: bool,
    definition: bool,
    location: bool,
    reference: bool,
}

impl<'a> QuestionNeeds<'a> {
    fn from_question(question: &str, intent: &'a str) -> Self {
        Self {
            intent,
            overview: asks_document_overview(question, intent),
            header: asks_document_header(question),
            method: asks_method_or_approach(question),
            experiment: asks_experiment_or_result(question),
            figure: asks_figure_or_table(question),
            table: asks_table_metrics(question),
            definition: asks_definition(question),
            location: asks_location(question),
            reference: asks_reference(question),
        }
    }
}

struct EvidenceFacts {
    citation_count: usize,
    evidence_chars: usize,
    has_selection: bool,
    has_overview: bool,
    has_header: bool,
    has_author_evidence: bool,
    has_method: bool,
    has_experiment: bool,
    has_figure: bool,
    has_table: bool,
    has_open_table: bool,
    has_current_view: bool,
    has_reference: bool,
}

impl EvidenceFacts {
    fn from_citations(citations: &[Citation]) -> Self {
        Self {
            citation_count: citations.len(),
            evidence_chars: citations
                .iter()
                .map(|citation| citation.quote.len())
                .sum::<usize>(),
            has_selection: citations
                .iter()
                .any(|citation| citation.source == "selection"),
            has_overview: citations.iter().any(is_page_overview_citation),
            has_header: citations.iter().any(|citation| {
                citation.source == "open_pages"
                    && citation
                        .section_title
                        .as_deref()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains("header")
            }),
            has_author_evidence: citations.iter().any(|citation| {
                let quote = citation.quote.to_lowercase();
                citation.source == "open_pages"
                    && (quote.contains("author")
                        || quote.contains("authors")
                        || quote.contains("university")
                        || quote.contains("institute")
                        || quote.contains("group"))
            }),
            has_method: has_method_evidence(citations),
            has_experiment: has_experiment_evidence(citations),
            has_figure: has_figure_evidence(citations),
            has_table: has_table_evidence(citations),
            has_open_table: has_open_table_evidence(citations),
            has_current_view: citations
                .iter()
                .any(|citation| citation.source == "current_view"),
            has_reference: has_reference_evidence(citations),
        }
    }
}

fn finish_decision(
    decision: &mut FinalizeDecision,
    citation_count: usize,
    attempt: u32,
    max_attempts: u32,
) {
    decision.citation_count = citation_count;
    decision.attempt = attempt;
    decision.max_attempts = max_attempts;
    if decision.needs_more_evidence && attempt + 1 >= max_attempts {
        decision.status = FinalizeStatus::Insufficient;
        decision.needs_more_evidence = false;
        decision.budget_exhausted = true;
        decision.next_tool_call = None;
        decision.reason = format!(
            "{} Reached the retrieval step limit.",
            decision.reason.trim()
        );
    }
}

fn answerable(reason: &str, runtime: FinalizeRuntime) -> FinalizeDecision {
    FinalizeDecision {
        status: FinalizeStatus::Answerable,
        citation_count: 0,
        needs_more_evidence: false,
        reason: reason.to_string(),
        missing: Vec::new(),
        next_tool_call: None,
        attempt: 0,
        max_attempts: 0,
        budget_exhausted: false,
        runtime,
    }
}

fn needs_more(
    reason: &str,
    missing: Vec<&str>,
    next_tool_call: NextToolCall,
    runtime: FinalizeRuntime,
) -> FinalizeDecision {
    FinalizeDecision {
        status: FinalizeStatus::NeedsMoreEvidence,
        citation_count: 0,
        needs_more_evidence: true,
        reason: reason.to_string(),
        missing: missing.into_iter().map(str::to_string).collect(),
        next_tool_call: Some(next_tool_call),
        attempt: 0,
        max_attempts: 0,
        budget_exhausted: false,
        runtime,
    }
}

fn asks_document_header(question: &str) -> bool {
    let normalized = question.to_lowercase();
    normalized.contains("author")
        || normalized.contains("authors")
        || normalized.contains("affiliation")
        || normalized.contains("affiliations")
        || normalized.contains("作者")
        || normalized.contains("署名")
        || normalized.contains("机构")
        || normalized.contains("单位")
        || normalized.contains("标题")
        || normalized.contains("title")
}

fn asks_method_or_approach(question: &str) -> bool {
    let normalized = question.to_lowercase();
    normalized.contains("method")
        || normalized.contains("approach")
        || normalized.contains("methodology")
        || normalized.contains("algorithm")
        || normalized.contains("framework")
        || normalized.contains("principle")
        || normalized.contains("mechanism")
        || normalized.contains("architecture")
        || normalized.contains("方法")
        || normalized.contains("框架")
        || normalized.contains("算法")
        || normalized.contains("原理")
        || normalized.contains("机制")
        || normalized.contains("架构")
        || normalized.contains("怎么做")
        || normalized.contains("如何实现")
}

fn overview_evidence_can_answer(question: &str, needs: &QuestionNeeds<'_>) -> bool {
    if needs.header
        || needs.experiment
        || needs.figure
        || needs.definition
        || needs.location
        || needs.reference
    {
        return false;
    }
    !needs.method || asks_high_level_principle_overview(question)
}

fn asks_high_level_principle_overview(question: &str) -> bool {
    let normalized = question.to_lowercase();
    let asks_principle = normalized.contains("原理")
        || normalized.contains("机制")
        || normalized.contains("principle")
        || normalized.contains("mechanism");
    let asks_specific_design = normalized.contains("具体")
        || normalized.contains("详细")
        || normalized.contains("设计")
        || normalized.contains("算法")
        || normalized.contains("流程")
        || normalized.contains("实现")
        || normalized.contains("architecture")
        || normalized.contains("algorithm")
        || normalized.contains("step")
        || normalized.contains("implementation")
        || normalized.contains("design");
    asks_principle && !asks_specific_design
}

fn asks_experiment_or_result(question: &str) -> bool {
    let normalized = question.to_lowercase();
    normalized.contains("experiment")
        || normalized.contains("evaluation")
        || normalized.contains("result")
        || normalized.contains("benchmark")
        || normalized.contains("performance")
        || normalized.contains("metric")
        || normalized.contains("score")
        || normalized.contains("sota")
        || normalized.contains("实验")
        || normalized.contains("评测")
        || normalized.contains("结果")
        || normalized.contains("指标")
        || normalized.contains("成绩")
        || normalized.contains("分数")
        || normalized.contains("效果")
}

fn asks_table_metrics(question: &str) -> bool {
    let normalized = question.to_lowercase();
    requested_table_number(&normalized).is_some()
        || contains_ascii_word(&normalized, "table")
        || contains_ascii_word(&normalized, "benchmark")
        || contains_ascii_word(&normalized, "metric")
        || contains_ascii_word(&normalized, "metrics")
        || contains_ascii_word(&normalized, "score")
        || contains_ascii_word(&normalized, "scores")
        || contains_ascii_word(&normalized, "sota")
        || contains_ascii_word(&normalized, "performance")
        || normalized.contains("表格")
        || normalized.contains("指标")
        || normalized.contains("分数")
        || normalized.contains("成绩")
}

fn asks_figure_or_table(question: &str) -> bool {
    let normalized = question.to_lowercase();
    contains_ascii_word(&normalized, "figure")
        || contains_ascii_word(&normalized, "figures")
        || contains_ascii_word(&normalized, "table")
        || contains_ascii_word(&normalized, "tables")
        || contains_ascii_word(&normalized, "caption")
        || normalized.contains("图表")
        || normalized.contains("表格")
        || contains_numbered_cjk_marker(&normalized, '图')
        || contains_numbered_cjk_marker(&normalized, '表')
}

/// Match `needle` (lowercase ASCII word) inside `haystack` only when both
/// neighboring characters are non-ASCII-alphanumeric. Prevents accidental hits
/// like `score` inside `scoreboard` or `table` inside `comfortable`.
fn contains_ascii_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    for (idx, _) in haystack.match_indices(needle) {
        let before_ok = idx == 0
            || haystack[..idx]
                .chars()
                .next_back()
                .is_some_and(|ch| !ch.is_ascii_alphanumeric());
        let after_ok = haystack[idx + needle.len()..]
            .chars()
            .next()
            .is_none_or(|ch| !ch.is_ascii_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn asks_definition(question: &str) -> bool {
    let normalized = question.to_lowercase();
    if is_document_overview_definition_collision(&normalized)
        || requested_table_number(&normalized).is_some()
        || normalized.contains("原理")
        || normalized.contains("机制")
        || normalized.contains("principle")
        || normalized.contains("mechanism")
    {
        return false;
    }
    normalized.contains("what is ")
        || normalized.contains("what are ")
        || normalized.contains("what does")
        || normalized.contains("define")
        || normalized.contains("definition")
        || normalized.contains("meaning")
        || normalized.contains("是什么意思")
        || normalized.contains("是什么")
        || normalized.contains("什么是")
        || normalized.contains("指什么")
        || normalized.contains("含义")
        || normalized.contains("定义")
}

fn is_document_overview_definition_collision(normalized: &str) -> bool {
    normalized.contains("what is this paper about")
        || normalized.contains("what is this article about")
        || normalized.contains("这篇文章讲的什么")
        || normalized.contains("这篇文章讲了什么")
        || normalized.contains("这篇文章讲什么")
        || normalized.contains("这篇论文讲的什么")
        || normalized.contains("这篇论文讲了什么")
        || normalized.contains("这篇论文讲什么")
        || normalized.contains("讲的什么")
        || normalized.contains("讲了什么")
        || normalized.contains("讲什么")
        || normalized.contains("关于什么")
}

fn asks_location(question: &str) -> bool {
    let normalized = question.to_lowercase();
    normalized.contains("where")
        || normalized.contains("which page")
        || normalized.contains("what page")
        || normalized.contains("which section")
        || normalized.contains("what section")
        || normalized.contains("located")
        || normalized.contains("location")
        || normalized.contains("appears")
        || normalized.contains("在哪里")
        || normalized.contains("在哪")
        || normalized.contains("哪一页")
        || normalized.contains("第几页")
        || normalized.contains("哪一节")
        || normalized.contains("哪个章节")
        || normalized.contains("位置")
        || normalized.contains("出现在")
}

fn asks_section_location(question: &str) -> bool {
    let normalized = question.to_lowercase();
    normalized.contains("section")
        || normalized.contains("chapter")
        || normalized.contains("哪一节")
        || normalized.contains("哪个章节")
        || normalized.contains("章节")
}

fn asks_reference(question: &str) -> bool {
    let normalized = question.to_lowercase();
    normalized.contains("reference")
        || normalized.contains("references")
        || normalized.contains("citation")
        || normalized.contains("cite")
        || normalized.contains("cited")
        || normalized.contains("bibliography")
        || normalized.contains("related work")
        || normalized.contains("依据")
        || normalized.contains("引用")
        || normalized.contains("参考文献")
        || normalized.contains("出处")
        || normalized.contains("来源")
        || normalized.contains("相关工作")
}

fn contains_numbered_cjk_marker(value: &str, marker: char) -> bool {
    let chars = value.chars().collect::<Vec<_>>();
    chars.iter().enumerate().any(|(index, ch)| {
        if *ch != marker {
            return false;
        }
        chars
            .get(index + 1)
            .copied()
            .filter(|next| {
                next.is_ascii_digit() || matches!(next, '一' | '二' | '三' | '四' | '五')
            })
            .is_some()
            || (chars.get(index + 1) == Some(&' ')
                && chars
                    .get(index + 2)
                    .copied()
                    .filter(|next| {
                        next.is_ascii_digit() || matches!(next, '一' | '二' | '三' | '四' | '五')
                    })
                    .is_some())
    })
}

fn has_method_evidence(citations: &[Citation]) -> bool {
    citations.iter().any(|citation| {
        if is_page_overview_citation(citation) {
            return false;
        }
        citation_matches(
            citation,
            &[
                "method",
                "approach",
                "methodology",
                "algorithm",
                "framework",
            ],
        )
    })
}

fn has_experiment_evidence(citations: &[Citation]) -> bool {
    citations.iter().any(|citation| {
        if is_page_overview_citation(citation) {
            return false;
        }
        if matches!(citation.source.as_str(), "table_fact" | "open_table") {
            return true;
        }
        citation_matches(
            citation,
            &[
                "experiment",
                "evaluation",
                "result",
                "benchmark",
                "performance",
                "metric",
            ],
        )
    })
}

fn has_figure_evidence(citations: &[Citation]) -> bool {
    citations.iter().any(|citation| {
        if matches!(
            citation.source.as_str(),
            "visual_anchor" | "inspect_objects"
        ) {
            return false;
        }
        matches!(
            citation.source.as_str(),
            "visual_asset" | "open_visual" | "analyze_visual" | "analyze_page"
        ) || citation_matches(citation, &["figure", "table", "caption"])
    })
}

fn has_table_evidence(citations: &[Citation]) -> bool {
    citations.iter().any(|citation| {
        matches!(citation.source.as_str(), "table_fact" | "open_table")
            || citation_matches(
                citation,
                &[
                    "table",
                    "benchmark",
                    "metric",
                    "score",
                    "sota",
                    "performance",
                ],
            )
    })
}

fn has_open_table_evidence(citations: &[Citation]) -> bool {
    citations
        .iter()
        .any(|citation| citation.source.as_str() == "open_table")
}

fn has_open_table_evidence_for_number(citations: &[Citation], table_number: &str) -> bool {
    citations
        .iter()
        .filter(|citation| citation.source == "open_table")
        .any(|citation| {
            citation
                .section_title
                .as_deref()
                .and_then(requested_table_number)
                .as_deref()
                == Some(table_number)
                || requested_table_number(&citation.quote).as_deref() == Some(table_number)
        })
}

fn has_current_view_table_evidence_for_number(citations: &[Citation], table_number: &str) -> bool {
    citations
        .iter()
        .filter(|citation| citation.source == "current_view")
        .any(|citation| {
            citation
                .section_title
                .as_deref()
                .and_then(requested_table_number)
                .as_deref()
                == Some(table_number)
                || requested_table_number(&citation.quote).as_deref() == Some(table_number)
        })
}

fn is_page_overview_citation(citation: &Citation) -> bool {
    citation.source == "open_pages"
        && citation
            .section_title
            .as_deref()
            .unwrap_or_default()
            .to_lowercase()
            .contains("overview")
}

fn has_definition_evidence(question: &str, citations: &[Citation]) -> bool {
    let focus_terms = definition_focus_terms(question);
    citations.iter().any(|citation| {
        let haystack = citation_haystack(citation);
        has_definition_marker(&haystack)
            && (focus_terms.is_empty() || focus_terms.iter().any(|term| haystack.contains(term)))
    })
}

fn has_definition_marker(haystack: &str) -> bool {
    let padded = format!(" {haystack} ");
    [
        " definition ",
        " defined ",
        " refers to ",
        " means ",
        " called ",
        " we propose ",
        " proposed ",
        " introduced ",
        " is a ",
        " is an ",
        " are a ",
        " are an ",
    ]
    .iter()
    .any(|marker| padded.contains(marker))
        || haystack.contains("是一种")
        || haystack.contains("是一个")
        || haystack.contains("指的是")
        || haystack.contains("定义为")
        || haystack.contains("提出")
}

fn definition_focus_terms(question: &str) -> Vec<String> {
    let normalized = question.to_lowercase();
    let mut terms = Vec::new();
    for marker in [
        "是什么意思",
        "是什么",
        "什么是",
        "指什么",
        "what is",
        "what are",
        "what does",
        "define",
        "definition",
        "meaning",
    ] {
        if let Some(index) = normalized.find(marker) {
            push_definition_focus_terms(&normalized[..index], &mut terms);
            push_definition_focus_terms(&normalized[index + marker.len()..], &mut terms);
        }
    }
    push_definition_focus_terms(&normalized, &mut terms);
    terms.dedup();
    terms
}

fn push_definition_focus_terms(value: &str, terms: &mut Vec<String>) {
    let cleaned = value
        .trim_matches(|ch: char| ch.is_whitespace() || matches!(ch, '?' | '？' | ':' | '：'))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if cleaned.chars().count() > 2 && !is_definition_stop_term(&cleaned) {
        terms.push(cleaned.clone());
    }
    for term in cleaned
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|term| term.chars().count() > 2)
        .filter(|term| !is_definition_stop_term(term))
    {
        terms.push(term.to_string());
    }
}

fn is_definition_stop_term(value: &str) -> bool {
    matches!(
        value,
        "what"
            | "does"
            | "define"
            | "definition"
            | "meaning"
            | "paper"
            | "article"
            | "this"
            | "that"
            | "这个"
            | "这篇"
            | "论文"
            | "文章"
            | "是什么"
            | "是什么意思"
            | "什么是"
            | "指什么"
            | "含义"
            | "定义"
    )
}

fn has_location_evidence(citations: &[Citation], require_section: bool) -> bool {
    citations.iter().any(|citation| {
        if require_section {
            return citation.section_title.as_deref().is_some_and(|title| {
                !title.trim().is_empty() && !title.to_lowercase().contains("overview")
            });
        }
        citation.page > 0
            && (!citation.block_id.trim().is_empty() || !citation.quote.trim().is_empty())
    })
}

fn has_reference_evidence(citations: &[Citation]) -> bool {
    citations.iter().any(|citation| {
        citation_matches(
            citation,
            &[
                "reference",
                "references",
                "citation",
                "cited",
                "related work",
                "bibliography",
                "prior work",
                "参考文献",
                "引用",
                "相关工作",
            ],
        )
    })
}

fn citation_matches(citation: &Citation, needles: &[&str]) -> bool {
    let haystack = citation_haystack(citation);
    needles.iter().any(|needle| haystack.contains(needle))
}

fn citation_haystack(citation: &Citation) -> String {
    format!(
        "{} {}",
        citation.section_title.as_deref().unwrap_or_default(),
        citation.quote
    )
    .to_lowercase()
}

fn asks_document_overview(question: &str, intent: &str) -> bool {
    if intent == "summarize" {
        return true;
    }
    let normalized = question.to_lowercase();
    normalized.contains("what is this paper about")
        || normalized.contains("what is this article about")
        || normalized.contains("summarize this paper")
        || ((normalized.contains("这篇")
            || normalized.contains("文章")
            || normalized.contains("论文"))
            && (normalized.contains("讲的什么")
                || normalized.contains("讲了什么")
                || normalized.contains("讲什么")
                || normalized.contains("讲述了什么")
                || normalized.contains("关于什么")
                || normalized.contains("主要内容")
                || normalized.contains("总结")
                || normalized.contains("概括")
                || normalized.contains("摘要")
                || normalized.contains("重要结论")
                || normalized.contains("主要结论")
                || normalized.contains("核心结论")))
}

#[cfg(test)]
mod tests;
