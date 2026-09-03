use super::*;

fn citation(source: &str, section_title: Option<&str>, quote: &str) -> Citation {
    Citation {
        id: "c1".to_string(),
        label: "[1]".to_string(),
        page: 1,
        block_id: "b1".to_string(),
        section_title: section_title.map(str::to_string),
        quote: quote.to_string(),
        bbox_list: serde_json::json!([]),
        document_id: "doc".to_string(),
        source: source.to_string(),
    }
}

#[test]
fn empty_evidence_does_not_exit_as_answerable() {
    let decision = finalize_citations("这篇文章讲的什么？", "explain", &[], 2, 3);
    assert_eq!(decision.status, FinalizeStatus::Insufficient);
    assert!(!decision.needs_more_evidence);
    assert!(decision.next_tool_call.is_none());
}

#[test]
fn document_overview_exits_when_page_evidence_is_sufficient() {
    let quote = "MemGPT: Towards LLMs as Operating Systems\n\nAbstract. Large language models are limited by fixed context windows. This paper proposes virtual context management and a memory hierarchy that allows agents to page information between context and external storage while preserving long-running interactions.";
    let decision = finalize_citations(
        "这篇文章讲的什么？",
        "explain",
        &[citation("open_pages", Some("Page 1 overview"), quote)],
        1,
        3,
    );
    assert_eq!(decision.status, FinalizeStatus::Answerable);
    assert!(!decision.needs_more_evidence);
}

#[test]
fn english_document_overview_does_not_route_as_definition() {
    let decision = finalize_citations("What is this paper about?", "explain", &[], 0, 3);

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("overview tool");
    assert_eq!(next_tool.tool, "open_pages");
    assert_eq!(next_tool.args["mode"], serde_json::json!("overview"));
}

#[test]
fn chinese_overview_plus_principle_uses_overview_not_definition() {
    let decision = finalize_citations("这篇文章讲了什么？原理是什么？", "explain", &[], 0, 3);

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    assert!(!decision.reason.contains("definition"));
    let next_tool = decision.next_tool_call.expect("overview tool");
    assert_eq!(next_tool.tool, "open_pages");
    assert_eq!(next_tool.args["mode"], serde_json::json!("overview"));
}

#[test]
fn chinese_overview_plus_principle_accepts_abstract_overview() {
    let quote = "GLM-5: from Vibe Coding to Agentic Engineering\n\nAbstract. We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. Building upon agentic reasoning and coding capabilities, GLM-5 adopts DSA to reduce training and inference costs while maintaining long-context fidelity, and uses asynchronous reinforcement learning infrastructure to improve post-training efficiency.";
    let decision = finalize_citations(
        "这篇文章讲了什么？原理是什么？",
        "explain",
        &[citation("open_pages", Some("Page 1 overview"), quote)],
        1,
        20,
    );

    assert_eq!(
        decision.status,
        FinalizeStatus::Answerable,
        "decision: {decision:?}"
    );
    assert!(!decision.needs_more_evidence);
    assert_eq!(decision.runtime, FinalizeRuntime::M3RuleGuard);
}

#[test]
fn overview_plus_experiment_does_not_accept_abstract_only() {
    let quote = "GLM-5: from Vibe Coding to Agentic Engineering\n\nAbstract. We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. It adopts DSA for long-context fidelity and asynchronous reinforcement learning infrastructure for post-training efficiency.";
    let decision = finalize_citations(
        "这篇文章讲了什么？实验结果怎么样？",
        "explain",
        &[citation("open_pages", Some("Page 1 overview"), quote)],
        1,
        20,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("experiment section tool");
    assert_eq!(next_tool.tool, "open_section");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("experiments evaluation results benchmark metric score SOTA table")
    );
}

#[test]
fn author_question_requests_header_tool_when_missing_metadata() {
    let decision = finalize_citations(
        "这篇论文的作者有哪些？",
        "explain",
        &[citation(
            "fts",
            None,
            "This paper proposes an adaptive context pruning method for coding agents.",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    assert_eq!(
        decision.missing,
        vec![
            "title".to_string(),
            "authors".to_string(),
            "affiliations".to_string()
        ]
    );
    let next_tool = decision.next_tool_call.expect("header tool should be set");
    assert_eq!(next_tool.tool, "open_pages");
    assert_eq!(next_tool.args["page"], serde_json::json!(1));
    assert_eq!(next_tool.args["mode"], serde_json::json!("header"));
}

#[test]
fn author_question_accepts_header_evidence() {
    let decision = finalize_citations(
        "这篇论文的作者有哪些？",
        "explain",
        &[citation(
            "open_pages",
            Some("Page 1 header"),
            "SWE-Pruner\nYuhang Wang, Yuling Shi\nShanghai Jiao Tong University",
        )],
        1,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::Answerable);
    assert!(decision.next_tool_call.is_none());
}

#[test]
fn method_question_requests_open_section() {
    let decision = finalize_citations(
        "这篇论文的方法框架是什么？",
        "explain",
        &[citation(
            "fts",
            None,
            "This paper studies context pruning for coding agents.",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("method section tool");
    assert_eq!(next_tool.tool, "open_section");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("method approach methodology algorithm framework")
    );
}

#[test]
fn method_and_experiment_question_rejects_page_overview_only() {
    let decision = finalize_citations(
        "这篇文章的方法具体是怎么设计的？请结合方法章节、算法流程和实验结果说明。",
        "explain",
        &[citation(
            "open_pages",
            Some("Page 21 overview"),
            "CASE STUDY ON SWE BENCH. The Pruner-augmented agent applies context pruning to file observations and reports token reduction.",
        )],
        0,
        20,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("method section tool");
    assert_eq!(next_tool.tool, "open_section");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("method approach methodology algorithm framework")
    );
}

#[test]
fn experiment_question_requests_open_section() {
    let decision = finalize_citations(
        "实验结果怎么样？",
        "explain",
        &[citation(
            "fts",
            None,
            "This paper studies context pruning for coding agents.",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("experiment section tool");
    assert_eq!(next_tool.tool, "open_section");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("experiments evaluation results benchmark metric score SOTA table")
    );
}

#[test]
fn table_metric_question_opens_full_table_after_structured_fact_hit() {
    let decision = finalize_citations(
        "请列出 Table 7 中 GLM-5 在 SWE-bench Verified 上的分数。",
        "explain",
        &[citation(
            "table_fact",
            Some("Table 7"),
            "Table 7: Comparison between GLM-5 and open-source/proprietary models. Coding / SWE-bench Verified | GLM-5 = 77.8",
        )],
        1,
        20,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("open table tool");
    assert_eq!(next_tool.tool, "open_table");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("请列出 Table 7 中 GLM-5 在 SWE-bench Verified 上的分数。")
    );
}

#[test]
fn table_explanation_question_opens_requested_table_before_definition_search() {
    let decision = finalize_citations("表 6 是什么，我看不懂，解读一下", "explain", &[], 0, 20);

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("open table tool");
    assert_eq!(next_tool.tool, "open_table");
    assert_eq!(next_tool.args["tableNumber"], serde_json::json!("6"));
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("表 6 是什么，我看不懂，解读一下")
    );
    assert!(!decision.reason.contains("definition"));
}

#[test]
fn table_explanation_question_still_opens_table_after_fact_hit() {
    let decision = finalize_citations(
        "表 6 是什么，我看不懂，解读一下",
        "explain",
        &[citation(
            "table_fact",
            Some("Table 6"),
            "Table 6 | Column 1 | Input Length = 64",
        )],
        1,
        20,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("open table tool");
    assert_eq!(next_tool.tool, "open_table");
    assert_eq!(next_tool.args["tableNumber"], serde_json::json!("6"));
}

#[test]
fn table_explanation_question_accepts_exact_open_table() {
    let decision = finalize_citations(
        "表 6 是什么，我看不懂，解读一下",
        "explain",
        &[citation(
            "open_table",
            Some("Table 6"),
            "Table 6: Average TTFT (ms) for different models and input lengths.",
        )],
        1,
        20,
    );

    assert_eq!(
        decision.status,
        FinalizeStatus::Answerable,
        "decision: {decision:?}"
    );
    assert!(!decision.needs_more_evidence);
}

#[test]
fn table_metric_question_can_finish_after_open_table_evidence() {
    let decision = finalize_citations(
        "请列出 Table 7 中 GLM-5 在 SWE-bench Verified 上的分数。",
        "explain",
        &[citation(
            "open_table",
            Some("Table 7"),
            "Table 7: Comparison between GLM-5 and open-source/proprietary models. Coding / SWE-bench Verified | GLM-5 = 77.8",
        )],
        1,
        20,
    );

    assert_eq!(
        decision.status,
        FinalizeStatus::Answerable,
        "decision: {decision:?}"
    );
    assert!(!decision.needs_more_evidence);
    assert!(decision.reason.contains("LLM evidence judge"));
}

#[test]
fn table_metric_phrase_is_not_treated_as_definition_request() {
    let decision = finalize_citations(
        "Table 3 里面提到的 SWE-Pruner 是什么指标？结果是怎么样的？",
        "explain",
        &[citation(
            "open_table",
            Some("Table 3"),
            "Table 3 | SWE-Pruner | Rounds = 41.1\nTable 3 | SWE-Pruner | Success (%) = 64.0\nTable 3 | SWE-Pruner | Tokens (M) = 0.670",
        )],
        1,
        20,
    );

    assert_eq!(
        decision.status,
        FinalizeStatus::Answerable,
        "decision: {decision:?}"
    );
    assert!(!decision
        .missing
        .iter()
        .any(|missing| missing.contains("definition")));
}

#[test]
fn figure_question_requests_caption_search() {
    let decision = finalize_citations(
        "图 1 说明了什么？",
        "explain",
        &[citation(
            "fts",
            None,
            "This paper studies context pruning for coding agents.",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("figure search tool");
    assert_eq!(next_tool.tool, "inspect_visuals");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("figure table chart caption")
    );
}

#[test]
fn figure_question_does_not_accept_visual_anchor_only() {
    let decision = finalize_citations(
        "Figure 3 说明了什么？",
        "explain",
        &[citation(
            "visual_anchor",
            Some("Figure 3"),
            "Resolved visual anchor: Figure 3 on page 4\nassetId=fig-3",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("visual content tool");
    assert_eq!(next_tool.tool, "inspect_visuals");
}

#[test]
fn definition_question_requests_definition_search() {
    let decision = finalize_citations(
        "SWE-Pruner 是什么？",
        "explain",
        &[citation(
            "fts",
            None,
            "The paper evaluates several coding agents on benchmark tasks.",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("definition search tool");
    assert_eq!(next_tool.tool, "search_chunks");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("definition defined means refers to called proposed introduced")
    );
}

#[test]
fn current_view_evidence_defers_definition_semantics_to_llm() {
    let decision = finalize_citations(
        "这些 Task Type 是什么意思？解读一下",
        "explain",
        &[citation(
            "current_view",
            Some("Current view page evidence: Page 17 lines 2-37"),
            "Table 5 Agentic Tasks Taxonomy used for Query Synthesis\nTask Type Instruction for Query Generation\ncode-summarize Summarize the main purpose or functionality of the code, but do not explain every line.\ncode-refactor Suggest a refactoring or improvement for the code.\nfind-relevant-part Ask to locate or identify the part of the code that implements a specific feature or logic.\ncode-optimize Request an optimization for the code.\ncode-locate Ask to pinpoint the location of a bug, feature, or important logic within the code.\ncode-explain Request an explanation for a particular logic, algorithm, or design choice in the code.",
        )],
        0,
        20,
    );

    assert_eq!(decision.status, FinalizeStatus::Answerable);
    assert!(decision.next_tool_call.is_none());
    assert_eq!(decision.runtime, FinalizeRuntime::M3RuleGuard);
}

#[test]
fn current_view_requested_table_does_not_force_open_table_in_m3() {
    let decision = finalize_citations(
        "Table 3 里面 SWE-Pruner 的数值是什么结果？",
        "explain",
        &[citation(
            "current_view",
            Some("Current view page evidence: Page 8 lines 1-12"),
            "Table 3\nMethod Rounds Success (%) Tokens (M)\nSWE-Pruner 41.1 64.0 0.670",
        )],
        0,
        20,
    );

    assert_eq!(decision.status, FinalizeStatus::Answerable);
    assert!(decision.next_tool_call.is_none());
    assert_eq!(decision.runtime, FinalizeRuntime::M3RuleGuard);
}

#[test]
fn definition_question_accepts_definitional_evidence() {
    let decision = finalize_citations(
        "SWE-Pruner 是什么？",
        "explain",
        &[citation(
            "fts",
            Some("Abstract"),
            "In this paper, we propose SWE-Pruner, a self-adaptive context pruning framework for coding agents.",
        )],
        1,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::Answerable);
    assert!(decision.next_tool_call.is_none());
}

#[test]
fn definition_question_rejects_irrelevant_chinese_copula_evidence() {
    let decision = finalize_citations(
        "SWE-Pruner 是什么？",
        "explain",
        &[citation(
            "fts",
            Some("Experiments"),
            "这是实验结果，展示模型在多个基准上的性能。",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("definition search tool");
    assert_eq!(next_tool.tool, "search_chunks");
}

#[test]
fn definition_question_accepts_targeted_chinese_definition() {
    let decision = finalize_citations(
        "SWE-Pruner 是什么？",
        "explain",
        &[citation(
            "fts",
            Some("Abstract"),
            "SWE-Pruner 是一种面向 coding agents 的自适应上下文剪枝框架。",
        )],
        1,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::Answerable);
}

#[test]
fn location_question_requests_section_when_only_plain_chunk_exists() {
    let decision = finalize_citations(
        "这个方法在哪一节介绍？",
        "locate",
        &[citation(
            "fts",
            None,
            "The method uses an adaptive pruning framework.",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("location section tool");
    assert_eq!(next_tool.tool, "open_section");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("section page location introduced described definition method reference")
    );
}

#[test]
fn location_question_accepts_section_evidence() {
    let decision = finalize_citations(
        "这个方法在哪一节介绍？",
        "locate",
        &[citation(
            "open_section",
            Some("3 Method"),
            "Section: 3 Method\nThe method uses an adaptive pruning framework.",
        )],
        1,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::Answerable);
}

#[test]
fn reference_question_requests_related_work_or_references_section() {
    let decision = finalize_citations(
        "这句话的引用来源是什么？",
        "explain",
        &[citation(
            "fts",
            None,
            "The method uses an adaptive pruning framework.",
        )],
        0,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::NeedsMoreEvidence);
    let next_tool = decision.next_tool_call.expect("reference section tool");
    assert_eq!(next_tool.tool, "open_section");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("references related work citation bibliography source prior work")
    );
}

#[test]
fn reference_question_accepts_reference_evidence() {
    let decision = finalize_citations(
        "相关工作引用了哪些方法？",
        "explain",
        &[citation(
            "open_section",
            Some("2 Related Work"),
            "Section: 2 Related Work\nPrior work on context compression includes LongLLMLingua and LLMLingua.",
        )],
        1,
        3,
    );

    assert_eq!(decision.status, FinalizeStatus::Answerable);
}

#[test]
fn cjk_figure_detection_does_not_match_common_words() {
    let decision = finalize_citations(
        "这个方法的表现如何？",
        "explain",
        &[citation(
            "fts",
            None,
            "This paper studies context pruning for coding agents.",
        )],
        0,
        3,
    );

    let next_tool = decision.next_tool_call.expect("method tool should be set");
    assert_eq!(next_tool.tool, "open_section");
    assert_eq!(
        next_tool.args["query"],
        serde_json::json!("method approach methodology algorithm framework")
    );
}

#[test]
fn table_number_parser_handles_cjk_table_markers() {
    assert_eq!(requested_table_number("表 6 是什么"), Some("6".to_string()));
    assert_eq!(
        requested_table_number("表六解读一下"),
        Some("6".to_string())
    );
    assert_eq!(
        requested_table_number("Table VI latency"),
        Some("6".to_string())
    );
    assert_eq!(requested_table_number("这个方法的表现如何？"), None);
}

#[test]
fn ascii_word_boundary_rejects_substring_hits() {
    // Callers always pass a lowercase haystack; mirror that here.
    assert!(!contains_ascii_word("comfortable bench", "table"));
    assert!(!contains_ascii_word("scoreboard view", "score"));
    assert!(!contains_ascii_word("figuration", "figure"));
    assert!(contains_ascii_word("table 3 results", "table"));
    assert!(contains_ascii_word("the score is 90", "score"));
    assert!(contains_ascii_word("see figure 1", "figure"));
}

#[test]
fn chinese_outstanding_phrase_does_not_force_table_route() {
    // "突出表现" 是中文常用搭配，不应仅凭这两个字符触发 table-metrics 路由。
    assert!(!asks_table_metrics("这个方法的突出表现来自于哪里？"));
    // 「方法的表现如何」也是常见 wording — 不应误命中 figure/table 路径。
    assert!(!asks_figure_or_table("这个方法的表现如何？"));
}

#[test]
fn ascii_word_table_routes_only_with_real_word() {
    assert!(asks_figure_or_table("see table 2 for evaluation"));
    assert!(!asks_figure_or_table(
        "this method is portable on small devices"
    ));
}

/// Recall-neutrality: the seed runs only the cheap base chain and skips the
/// table fan-out. This proves the answerability loop re-requests table evidence
/// when a metric question lands with only seed-level prose, so not fetching it
/// in the seed loses no recall — the loop fills the gap on demand.
#[test]
fn table_gap_in_seed_is_recovered_by_loop() {
    let decision = finalize_citations(
        "Report the benchmark score for SWE-Pruner",
        "compare",
        &[citation(
            "open_pages",
            Some("Method"),
            "method overview prose without any benchmark table",
        )],
        0,
        8,
    );
    assert!(
        decision.needs_more_evidence,
        "loop must ask for more evidence, got {decision:?}"
    );
    let next_tool = decision
        .next_tool_call
        .as_ref()
        .map(|call| call.tool.as_str())
        .unwrap_or("");
    assert!(
        matches!(
            next_tool,
            "search_table_facts" | "open_table" | "open_section"
        ),
        "expected a table/evidence tool, got {next_tool}"
    );
}

/// Recall-neutrality for visuals: a figure question with only prose seed
/// evidence must drive the loop to request more evidence rather than answering.
#[test]
fn figure_gap_in_seed_is_recovered_by_loop() {
    let decision = finalize_citations(
        "What does Figure 2 show?",
        "explain",
        &[citation(
            "open_pages",
            Some("Introduction"),
            "unrelated prose",
        )],
        0,
        8,
    );
    assert!(
        decision.needs_more_evidence,
        "loop must ask for more evidence, got {decision:?}"
    );
}
