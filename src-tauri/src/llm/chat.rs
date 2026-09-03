use std::time::Duration;

use super::{claims, openai_stream};
use crate::{
    normalize_base_url, optional_non_empty, runtime, truncate_for_error, AskAnswerResult,
    AskDocumentInput, OpenAiChatMessage, OpenAiChatRequest, OpenAiChatResponse,
    OpenAiCompatibleProvider,
};

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LlmAnswerabilityDecision {
    pub status: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub missing: Vec<String>,
    #[serde(default)]
    pub next_tool_call: Option<LlmRagToolCall>,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LlmRagToolCall {
    pub tool: String,
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(default)]
    pub reason: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LlmCurrentViewDecision {
    pub relevance: String,
    pub mode: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub should_use_current_view: bool,
}

pub(crate) fn text_message(role: &str, content: impl Into<String>) -> OpenAiChatMessage {
    OpenAiChatMessage {
        role: role.to_string(),
        content: serde_json::Value::String(content.into()),
    }
}

pub(crate) fn user_message_with_optional_image(
    text: &str,
    image_data_url: Option<&str>,
) -> OpenAiChatMessage {
    if let Some(image_url) = image_data_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        OpenAiChatMessage {
            role: "user".to_string(),
            content: serde_json::json!([
                { "type": "text", "text": text },
                { "type": "image_url", "image_url": { "url": image_url } }
            ]),
        }
    } else {
        text_message("user", text.to_string())
    }
}

pub(crate) fn extract_chat_response_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(value) => value.trim().to_string(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Minimal non-streaming chat completion: one system + one user message → the
/// assistant's text. Retries once on transient (429 / 5xx / network / decode)
/// failures. Used where we need a single one-shot reply (e.g. knowledge
/// precipitation's JSON extraction), not a token stream.
pub(crate) async fn run_simple_completion(
    provider: &OpenAiCompatibleProvider,
    system: &str,
    user: &str,
    temperature: f32,
    timeout_secs: u64,
) -> Result<String, String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|err| format!("Failed to create completion client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&provider.base_url)
    );
    let request = OpenAiChatRequest {
        model: provider.model.clone(),
        temperature,
        stream: None,
        messages: vec![
            text_message("system", system.to_string()),
            text_message("user", user.to_string()),
        ],
    };
    let mut last_error = String::new();
    for retry in 0..2 {
        let mut builder = client.post(&endpoint).json(&request);
        if let Some(api_key) = &provider.api_key {
            builder = builder.bearer_auth(api_key);
        }
        let response = match builder.send().await {
            Ok(response) => response,
            Err(err) => {
                last_error = format!("Completion request failed: {err}");
                continue;
            }
        };
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            last_error = format!(
                "Completion returned {status}: {}",
                truncate_for_error(&body, 600)
            );
            if status.as_u16() == 429 || status.is_server_error() {
                continue;
            }
            break;
        }
        match serde_json::from_str::<OpenAiChatResponse>(&body) {
            Ok(decoded) => {
                let text = decoded
                    .choices
                    .into_iter()
                    .next()
                    .map(|choice| extract_chat_response_text(&choice.message.content))
                    .filter(|text| !text.trim().is_empty());
                if let Some(text) = text {
                    return Ok(text);
                }
                last_error = "Completion returned an empty response".to_string();
            }
            Err(err) => {
                last_error = format!(
                    "Failed to decode completion on attempt {}: {err}; body={}",
                    retry + 1,
                    truncate_for_error(&body, 600)
                );
            }
        }
    }
    Err(last_error)
}

pub(crate) async fn judge_answerability_with_openai_compatible(
    question: &str,
    intent: &str,
    citations: &[runtime::rag::Citation],
    tool_feedback: &[String],
    provider: &OpenAiCompatibleProvider,
    budget: &crate::model_catalog::ModelContextBudget,
) -> Result<LlmAnswerabilityDecision, String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| format!("Failed to create answerability judge client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&provider.base_url)
    );
    let vision_enabled = provider
        .capabilities
        .iter()
        .any(|capability| capability == "vision");
    let user_content = build_answerability_judge_prompt(
        question,
        intent,
        citations,
        tool_feedback,
        vision_enabled,
        budget,
    )?;
    let request = OpenAiChatRequest {
        model: provider.model.clone(),
        temperature: 0.0,
        stream: None,
        messages: vec![
            text_message(
                "system",
                "You are an answerability judge for an academic PDF RAG agent. Return one JSON object only. Do not answer the user's question. Decide whether the current evidence is sufficient, or choose exactly one available RAG tool to gather missing evidence.",
            ),
            text_message("user", user_content),
        ],
    };
    let mut last_error = String::new();
    let response = {
        let mut decoded = None;
        for retry in 0..2 {
            let mut builder = client.post(&endpoint).json(&request);
            if let Some(api_key) = &provider.api_key {
                builder = builder.bearer_auth(api_key);
            }
            let response = match builder.send().await {
                Ok(response) => response,
                Err(err) => {
                    last_error = format!("Answerability judge request failed: {err}");
                    continue;
                }
            };
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                last_error = format!(
                    "Answerability judge returned {status}: {}",
                    truncate_for_error(&body, 600)
                );
                if status.as_u16() == 429 || status.is_server_error() {
                    continue;
                }
                break;
            }
            match serde_json::from_str::<OpenAiChatResponse>(&body) {
                Ok(response) => {
                    decoded = Some(response);
                    break;
                }
                Err(err) => {
                    last_error = format!(
                        "Failed to decode answerability judge response on attempt {}: {}; body={}",
                        retry + 1,
                        err,
                        truncate_for_error(&body, 600)
                    );
                }
            }
        }
        decoded
    }
    .ok_or_else(|| last_error.clone())?;
    let text = response
        .choices
        .into_iter()
        .next()
        .map(|choice| extract_chat_response_text(&choice.message.content))
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "Answerability judge returned an empty response".to_string())?;
    parse_answerability_decision_for_vision(&text, vision_enabled)
}

pub(crate) async fn judge_current_view_relevance_with_openai_compatible(
    question: &str,
    metadata: &serde_json::Value,
    provider: &OpenAiCompatibleProvider,
) -> Result<LlmCurrentViewDecision, String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|err| format!("Failed to create current-view judge client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&provider.base_url)
    );
    let metadata = serde_json::to_string_pretty(metadata)
        .map_err(|err| format!("Failed to encode current-view metadata: {err}"))?;
    let user_content = format!(
        "Question:\n{question}\n\nCurrent view metadata:\n{metadata}\n\nReturn JSON only:\n{{\"relevance\":\"high|medium|low\",\"mode\":\"none|selection|visible|semantic_chunks|full\",\"shouldUseCurrentView\":true|false,\"reason\":\"short reason\"}}\n\nRules:\n- Decide only whether the current PDF view is likely relevant and which lightweight context mode to fetch. Do not answer the question.\n- Use mode=selection when selection metadata exists and the question refers to it.\n- Use mode=visible when the question uses deictic wording such as this/these/here/above/current page/这个/这些/这里/上面/当前页, or when visible object captions appear relevant.\n- Use mode=semantic_chunks for long-document content questions that appear related to the current page but need targeted chunks rather than the whole page.\n- Use mode=full only when the user explicitly asks for the entire/full/current page or when visible metadata strongly indicates the answer is on this page but line-level context is needed.\n- Use mode=none when the question is clearly independent of the current view.\n- This is a soft routing decision; low relevance must not block global retrieval."
    );
    let request = OpenAiChatRequest {
        model: provider.model.clone(),
        temperature: 0.0,
        stream: None,
        messages: vec![
            text_message(
                "system",
                "You are a current-view relevance router for a PDF RAG agent. Return one JSON object only. Do not answer the user's question.",
            ),
            text_message("user", user_content),
        ],
    };
    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = &provider.api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = builder
        .send()
        .await
        .map_err(|err| format!("Current-view judge request failed: {err}"))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "Current-view judge returned {status}: {}",
            truncate_for_error(&body, 600)
        ));
    }
    let response = serde_json::from_str::<OpenAiChatResponse>(&body).map_err(|err| {
        format!(
            "Failed to decode current-view judge response: {}; body={}",
            err,
            truncate_for_error(&body, 600)
        )
    })?;
    let text = response
        .choices
        .into_iter()
        .next()
        .map(|choice| extract_chat_response_text(&choice.message.content))
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "Current-view judge returned an empty response".to_string())?;
    parse_current_view_decision(&text)
}

/// Single-shot LLM call that condenses a conversation's opening exchange into a
/// short tab title (the "hybrid" title strategy: a temporary truncated title is
/// shown immediately, then quietly replaced by this once the first answer lands).
/// Returns a trimmed title with no surrounding quotes; the caller enforces the
/// final length cap and falls back to the temporary title on any error.
pub(crate) async fn generate_session_title_with_openai_compatible(
    question: &str,
    answer: &str,
    provider: &OpenAiCompatibleProvider,
) -> Result<String, String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|err| format!("Failed to create session-title client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&provider.base_url)
    );
    let answer_excerpt = truncate_chars(answer, 600);
    let user_content = format!(
        "Summarize this conversation as a very short tab title.\n\nUser asked:\n{question}\n\nAssistant answered:\n{answer_excerpt}\n\nRules:\n- Reply with ONLY the title text, nothing else.\n- At most 12 characters (Chinese characters count as one each); fewer is better.\n- No surrounding quotes, no trailing punctuation, no prefix like 'Title:'.\n- Use the same language as the user's question.\n- Capture the core topic, not a generic label."
    );
    let request = OpenAiChatRequest {
        model: provider.model.clone(),
        temperature: 0.0,
        stream: None,
        messages: vec![
            text_message(
                "system",
                "You name chat sessions. Reply with one short title only, no explanation.",
            ),
            text_message("user", user_content),
        ],
    };
    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = &provider.api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = builder
        .send()
        .await
        .map_err(|err| format!("Session-title request failed: {err}"))?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!(
            "Session-title request returned {status}: {}",
            truncate_for_error(&body, 400)
        ));
    }
    let response = serde_json::from_str::<OpenAiChatResponse>(&body).map_err(|err| {
        format!(
            "Failed to decode session-title response: {}; body={}",
            err,
            truncate_for_error(&body, 400)
        )
    })?;
    let text = response
        .choices
        .into_iter()
        .next()
        .map(|choice| extract_chat_response_text(&choice.message.content))
        .map(|text| clean_session_title(&text))
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "Session-title request returned an empty response".to_string())?;
    Ok(text)
}

/// Strip the noise an LLM tends to wrap a short title in: surrounding quotes,
/// a leading "Title:" label, and trailing punctuation/whitespace. Collapses to
/// a single line.
fn clean_session_title(raw: &str) -> String {
    let first_line = raw
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    let mut title = first_line.trim().to_string();
    for prefix in ["Title:", "title:", "标题:", "标题:"] {
        if let Some(stripped) = title.strip_prefix(prefix) {
            title = stripped.trim().to_string();
        }
    }
    let trimmed = title
        .trim_matches(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '"' | '\'' | '“' | '”' | '‘' | '’' | '「' | '」' | '。' | '.'
                )
        })
        .to_string();
    trimmed
}

fn build_answerability_judge_prompt(
    question: &str,
    intent: &str,
    citations: &[runtime::rag::Citation],
    tool_feedback: &[String],
    vision_enabled: bool,
    budget: &crate::model_catalog::ModelContextBudget,
) -> Result<String, String> {
    // Web tools are scoped to the tool-calling/agentic paths, not the M4 judge.
    let tools = serde_json::to_string_pretty(&runtime::rag::rag_tool_specs_for_capabilities(
        vision_enabled,
        false,
    ))
    .map_err(|err| format!("Failed to encode RAG tool specs: {err}"))?;
    let evidence = citations
        .iter()
        .take(budget.judge_max_citations)
        .map(|citation| {
            serde_json::json!({
                "label": citation.label,
                "page": citation.page,
                "source": citation.source,
                "sectionTitle": citation.section_title,
                "quote": truncate_chars(&citation.quote, budget.judge_quote_chars)
            })
        })
        .collect::<Vec<_>>();
    let evidence = serde_json::to_string_pretty(&evidence)
        .map_err(|err| format!("Failed to encode judge evidence: {err}"))?;
    let tool_feedback = if tool_feedback.is_empty() {
        "[]".to_string()
    } else {
        serde_json::to_string_pretty(tool_feedback)
            .map_err(|err| format!("Failed to encode judge tool feedback: {err}"))?
    };
    let allowed_tools = runtime::rag::rag_tool_specs_for_capabilities(vision_enabled, false)
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>()
        .join("|");
    Ok(format!(
        "Question:\n{question}\n\nIntent: {intent}\n\nVision model enabled: {vision_enabled}\n\nAvailable tools:\n{tools}\n\nCurrent evidence:\n{evidence}\n\nTool feedback from previous judge-requested calls:\n{tool_feedback}\n\nReturn JSON with this shape:\n{{\"status\":\"answerable|needs_more_evidence|insufficient\",\"reason\":\"short reason\",\"missing\":[\"...\"],\"nextToolCall\":{{\"tool\":\"{allowed_tools}\",\"args\":{{}},\"reason\":\"why this tool\"}}}}\n\nRules:\n- Use status=answerable only when the evidence directly supports the requested answer.\n- LIBRARY/WORKSPACE questions about the user's document set itself — which documents/papers they have, what is in the sidebar/list, which of their papers is about a topic (e.g. \"我左侧栏中有哪些论文\", \"list my documents\", \"which of my papers is about X\") — are answered from the Workspace documents manifest in the tool feedback, NOT from document evidence. When the question is of this kind, mark status=answerable immediately and do NOT request any retrieval tool.\n- FIRST classify the question's nature and apply the matching sufficiency bar (this overrides the type-specific rules below when they conflict):\n  (a) PRECISE-FACT (a specific table cell/metric, a number, an exact definition, a named entity's value): strict bar — require evidence that matches the exact target; otherwise keep retrieving or mark insufficient.\n  (b) OPEN-SYNTHESIS (relationship, difference, comparison, overview, evaluation, \"can A be combined with B\", pros/cons, how two things relate — e.g. \"这两篇有什么关系/区别\", \"relationship between\", \"compare\"): relaxed bar — once you have TOPIC-LEVEL evidence for each subject (its title, abstract, introduction, or section overview), that is SUFFICIENT to answer by comparing/synthesizing themes. Do NOT require a direct citation linking the subjects (e.g. proof that paper A cites paper B); absence of such a link is itself a valid finding. Mark answerable and let the answer note any uncertainty.\n  (c) For cross-document questions, gather topic-level evidence for EACH referenced document (use its documentId), then judge by (b).\n- Treat tool feedback as authoritative loop state. If a previous tool call returned no new evidence or repeated a previous request, do not request the same tool with the same arguments again. Decide from the current evidence, choose a different useful tool, or mark insufficient.\n- For broad overview questions such as \"what is this paper about\" or \"这篇文章/论文讲了什么\", title/abstract/page overview evidence can be enough, including high-level \"principle/原理/机制\" wording when the user is asking for a conceptual explanation.\n- Evidence with source=current_view is line-ordered content from the user's visible/current PDF page. For deictic questions such as \"这些/这里/当前页/上面/this/these\", treat matching current_view evidence as highly relevant and do not ignore it in favor of global search.\n- For specific method/design/algorithm questions that ask for concrete architecture, steps, implementation details, or algorithm design, page overview alone is not enough; prefer inspect_tree first, then read_tree_node_lines with the found treeNodeIds.\n- For table/metric/SOTA/benchmark/score questions, do not mark answerable just because some table_fact exists. Verify that evidence matches the requested table number/caption, row/entity, and metric/column. If the user names a table number such as Table 3 or 表3, first request resolve_table_anchor or open_table with that tableNumber/query; do not use similarly named rows from another table.\n- For exact numeric table claims, prefer open_table evidence over isolated search_table_facts evidence. Reject evidence from the wrong table number, wrong row/entity, unrelated captions, prose headings, or conflicting values.\n- If a table question asks what metric/which metric/是什么指标/数值/结果, use the opened table's column names and row values as the answer. Do not require a general definition or prose description of the row/entity unless the user explicitly asks what the method is.\n- In Chinese table questions, phrases like \"是什么指标\" usually mean \"which metrics/columns are reported\". If current open_table evidence already contains the requested row/entity and columns such as Rounds, Success (%), Tokens (M), the evidence is sufficient for the table-metric part. Do not keep searching for a method definition unless the question explicitly asks for 方法/原理/定义/架构/怎么做.\n- For table explanation questions such as \"解读一下\", \"看不懂\", \"说明什么\", or \"what does this table show\", caption plus nearby explanatory text is sufficient when it explains the table purpose and key takeaway. Do not require a perfect row/column matrix unless the user asks for exact cell values or comparisons. If open_table facts have incomplete column labels but open_table_context/current_view/open_pages explain the takeaway, mark answerable.\n- For explicit figure/chart/image references such as Figure 3, Fig. 3, 图3, or Chart 2, first request resolve_visual_anchor, then open_visual or analyze_visual when visual content is needed.\n- For broad figure/chart/image questions without an explicit number, use inspect_tree or inspect_objects to discover relevant page/section objects, then open_visual. Use analyze_visual only when Vision model enabled is true and caption/nearby text is not enough.\n- If object or section tools identify a relevant page but still do not provide enough evidence, use page fallback: call analyze_page with that page when Vision model enabled is true and visual/layout evidence is needed; otherwise call open_pages with mode=\"full\" for that page.\n- If no page is known, first use inspect_tree, inspect_objects, inspect_visuals, search_table_facts, or search_chunks to discover one; do not guess a page number.\n- For experiment/result questions, require experiment/result evidence. If the question asks for concrete metrics, use table tools first; otherwise inspect_tree then read_tree_node_lines for experiments/results nodes.\n- Use open_section when line evidence is missing or when broader section context is needed.\n- If status=needs_more_evidence, nextToolCall is required and must use one available tool.\n- The focus document (the one the user is currently reading) is the default search target. The tool feedback includes a Workspace documents manifest: every document in the user's library with its id, folder, title, and (for the most relevant) an abstract; the focus doc is tagged [CURRENT FOCUS] and any user-attached docs are tagged [@referenced]. Use the manifest's titles/abstracts/folders to locate the right document, then search it by adding its id as the `documentId` argument on a tool call (search_chunks/inspect_tree/open_section/read_tree_node_lines/open_pages). Only route to another document when the question genuinely requires cross-document comparison, corroboration, or it clearly lives in a different paper; otherwise stay on the focus document. For a cross-document question, gather topic-level evidence from EACH relevant document via its `documentId`. The `documentId` value must be one of the ids listed in the manifest; never invent an id.\n- Use recall_chat_history only when the question refers to earlier discussion in this chat (e.g. \"like we discussed\", \"the paper you mentioned\", a follow-up to a prior turn): omit query for the most recent turns, or pass a keyword to locate a specific past topic. Do not call it for self-contained questions.\n- If no useful next step exists, use status=insufficient and omit nextToolCall."
    ))
}

#[cfg(test)]
pub(crate) fn parse_answerability_decision(text: &str) -> Result<LlmAnswerabilityDecision, String> {
    parse_answerability_decision_for_vision(text, false)
}

pub(crate) fn parse_answerability_decision_for_vision(
    text: &str,
    vision_enabled: bool,
) -> Result<LlmAnswerabilityDecision, String> {
    let json_text = extract_json_object(text)
        .ok_or_else(|| "Answerability judge did not return a JSON object".to_string())?;
    let value = serde_json::from_str::<serde_json::Value>(json_text)
        .map_err(|err| format!("Failed to parse answerability judge JSON: {err}"))?;
    let status = value
        .get("status")
        .and_then(|status| status.as_str())
        .map(normalize_answerability_status)
        .ok_or_else(|| "Answerability judge JSON is missing status".to_string())?;
    if !matches!(
        status.as_str(),
        "answerable" | "needs_more_evidence" | "insufficient"
    ) {
        return Err(format!(
            "Answerability judge returned unsupported status: {status}"
        ));
    }
    let reason = value
        .get("reason")
        .and_then(|reason| reason.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let missing = normalize_missing_list(value.get("missing"));
    let mut next_tool_call = normalize_next_tool_call(value.get("nextToolCall"))?;

    // P4-2 schema validation: the tool name must be in the registered RAG tool
    // catalogue (vision-gated when applicable), and `args` must be an object.
    if let Some(call) = next_tool_call.as_ref() {
        let capabilities = runtime::rag::RagToolCapabilities {
            vision_enabled,
            ..runtime::rag::RagToolCapabilities::default()
        };
        if !runtime::rag::is_registered_rag_tool_for_capabilities(&call.tool, capabilities) {
            return Err(format!(
                "Answerability judge requested an unknown tool: {}",
                call.tool
            ));
        }
        if !call.args.is_object() {
            return Err(format!(
                "Answerability judge nextToolCall.args must be a JSON object, got: {}",
                call.args
            ));
        }
    }

    if status == "needs_more_evidence" && next_tool_call.is_none() {
        return Err("Answerability judge requested more evidence without a tool call".to_string());
    }
    // Defensive: if status is not needs_more_evidence but a tool was suggested,
    // clear it so downstream loops do not silently keep retrieving.
    if status != "needs_more_evidence" {
        next_tool_call = None;
    }

    Ok(LlmAnswerabilityDecision {
        status,
        reason,
        missing,
        next_tool_call,
    })
}

pub(crate) fn parse_current_view_decision(text: &str) -> Result<LlmCurrentViewDecision, String> {
    let json_text = extract_json_object(text)
        .ok_or_else(|| "Current-view judge did not return a JSON object".to_string())?;
    let value = serde_json::from_str::<serde_json::Value>(json_text)
        .map_err(|err| format!("Failed to parse current-view judge JSON: {err}"))?;
    let relevance = value
        .get("relevance")
        .and_then(|value| value.as_str())
        .map(normalize_current_view_relevance)
        .unwrap_or_else(|| "low".to_string());
    let mode = value
        .get("mode")
        .and_then(|value| value.as_str())
        .map(normalize_current_view_mode)
        .unwrap_or_else(|| "none".to_string());
    let should_use_current_view = value
        .get("shouldUseCurrentView")
        .or_else(|| value.get("should_use_current_view"))
        .and_then(|value| value.as_bool())
        .unwrap_or(mode != "none");
    let reason = value
        .get("reason")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    Ok(LlmCurrentViewDecision {
        relevance,
        mode: mode.clone(),
        reason,
        should_use_current_view: should_use_current_view && mode != "none",
    })
}

fn normalize_answerability_status(status: &str) -> String {
    match status.trim().to_lowercase().as_str() {
        "answerable" | "sufficient" | "enough" => "answerable".to_string(),
        "insufficient" | "not_answerable" | "no_answer" => "insufficient".to_string(),
        "needs_more_evidence" | "need_more_evidence" | "needs_more" | "need_more" | "partial" => {
            "needs_more_evidence".to_string()
        }
        other => other.to_string(),
    }
}

fn normalize_current_view_relevance(value: &str) -> String {
    match value.trim().to_lowercase().as_str() {
        "high" | "relevant" | "strong" => "high".to_string(),
        "medium" | "maybe" | "partial" => "medium".to_string(),
        _ => "low".to_string(),
    }
}

fn normalize_current_view_mode(value: &str) -> String {
    match value.trim().to_lowercase().replace('-', "_").as_str() {
        "selection" => "selection".to_string(),
        "visible" | "viewport" | "current_view" => "visible".to_string(),
        "semantic_chunks" | "semantic" | "chunks" => "semantic_chunks".to_string(),
        "full" | "full_page" | "page" => "full".to_string(),
        _ => "none".to_string(),
    }
}

fn normalize_missing_list(value: Option<&serde_json::Value>) -> Vec<String> {
    match value {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str())
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
            .collect(),
        Some(serde_json::Value::String(item)) => item
            .split([';', '\n'])
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn normalize_next_tool_call(
    value: Option<&serde_json::Value>,
) -> Result<Option<LlmRagToolCall>, String> {
    match value {
        Some(serde_json::Value::Object(_)) => serde_json::from_value::<LlmRagToolCall>(
            value.cloned().unwrap_or(serde_json::Value::Null),
        )
        .map(Some)
        .map_err(|err| format!("Failed to parse nextToolCall: {err}")),
        Some(serde_json::Value::String(tool)) => {
            let tool = tool.trim();
            if tool.is_empty() {
                Ok(None)
            } else {
                Ok(Some(LlmRagToolCall {
                    tool: tool.to_string(),
                    args: serde_json::json!({}),
                    reason: String::new(),
                }))
            }
        }
        _ => Ok(None),
    }
}

fn extract_json_object(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (start < end).then_some(&trimmed[start..=end])
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut result = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        result.push_str("...");
    }
    result
}

pub(crate) async fn ask_with_openai_compatible(
    question: &str,
    input: &AskDocumentInput,
    agent_run: &runtime::agent::AgentRunResult,
    provider: &OpenAiCompatibleProvider,
    workspace_manifest: &str,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<AskAnswerResult, String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|err| format!("Failed to create chat client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&provider.base_url)
    );
    let answer_language = answer_language_for_question(question, input.locale.as_deref());
    let mut user_content = String::new();
    user_content.push_str(&format!(
        "Retrieval run: {}\nIntent: {}\n\n",
        agent_run.retrieval_run.id, agent_run.retrieval_run.intent
    ));
    if !agent_run.session_context.trim().is_empty() {
        user_content.push_str("Conversation memory:\n");
        user_content.push_str(&agent_run.session_context);
        user_content.push_str("\n\n");
    }
    if let Some(page) = input.page {
        user_content.push_str(&format!("Page: {page}\n\n"));
    }
    if !agent_run.retrieval_run.prompt_context.trim().is_empty() {
        user_content.push_str("Evidence sources:\n");
        user_content.push_str(&agent_run.retrieval_run.prompt_context);
        user_content.push_str("\n\n");
    }
    // The user's whole document list (titles/folders/abstracts). Lets the agent
    // answer questions ABOUT the library itself ("which papers do I have", "what's
    // in the sidebar") without retrieving — those facts aren't in the evidence.
    if !workspace_manifest.trim().is_empty() {
        user_content.push_str(workspace_manifest.trim_end());
        user_content.push_str("\n\n");
    }
    // Best-effort signal: when the answerability gate did NOT conclude "answerable"
    // but we are still answering (the retrieval loop gathered usable evidence yet
    // couldn't fully confirm everything), tell the model to answer with what it has
    // and explicitly state the limits — instead of refusing. This keeps open-ended
    // questions useful without per-scenario rules.
    let gate_status = agent_run
        .retrieval_run
        .trace
        .finalize_gate
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if !gate_status.is_empty() && gate_status != "answerable" {
        let gate_missing = agent_run
            .retrieval_run
            .trace
            .finalize_gate
            .get("missing")
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .filter(|value| !value.trim().is_empty());
        user_content.push_str(
            "Note: the evidence may not fully resolve every part of this question. Answer with what the evidence does support, and explicitly state which parts remain uncertain or unconfirmed rather than refusing to answer. Do not invent facts to fill the gaps.",
        );
        if let Some(missing) = gate_missing {
            user_content.push_str(&format!(" Specifically still unconfirmed: {missing}."));
        }
        user_content.push_str("\n\n");
    }
    user_content.push_str("Question:\n");
    user_content.push_str(question);
    user_content.push_str(
        "\n\nWrite a structured Markdown answer only. Do not return JSON. Use a short direct answer first, then organize the supporting explanation with concise paragraphs, bullet lists, or numbered lists when helpful. For summary, conclusion, comparison, method, or experiment questions, prefer clear section labels and lists over one long paragraph. Use only the provided evidence sources. If evidence is insufficient, say so clearly. Do not invent facts beyond the evidence. Stay tightly scoped to the user's question. For table or metric questions, answer the requested table, row/entity, columns, and values first; do not add metrics from other tables, pages, or sections unless they are explicitly requested or strictly necessary to define the requested entity.",
    );

    let request = OpenAiChatRequest {
        model: provider.model.clone(),
        temperature: 0.2,
        stream: Some(true),
        messages: vec![
            text_message(
                "system",
                format!(
                    "You are Lumenfolio, a careful academic PDF reading assistant. Answer in {answer_language}. Use only the provided evidence sources. Do not invent facts beyond the evidence. When the question is about the user's document library/workspace itself — which documents or papers they have, what is in the sidebar/list, which of their papers is about a topic — the provided 'Workspace documents' list is authoritative: answer directly from it (list the relevant titles), it does not need supporting evidence. Return Markdown only, not JSON. Prefer readable structure: a short direct answer, then concise paragraphs, bullet lists, or numbered lists when they improve scanability. Avoid one long undifferentiated paragraph. If the evidence is insufficient, say so clearly. Keep the answer scoped to the user's question; for table or metric questions, do not volunteer extra metrics from other tables or sections. Reply with plain Markdown prose only — never write tool-call syntax, function calls, or any `<|...|>` markup."
                ),
            ),
            user_message_with_optional_image(&user_content, input.image_data_url.as_deref()),
        ],
    };
    let mut builder = client
        .post(endpoint)
        .header("Accept", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .json(&request);
    if let Some(api_key) = &provider.api_key {
        builder = builder.bearer_auth(api_key);
    }

    let response = builder
        .send()
        .await
        .map_err(|err| format!("Chat provider request failed: {err}"))?;
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    log::info!(
        "chat provider response event_id={} status={} content_type={}",
        answer_event_id.unwrap_or("-"),
        status,
        content_type
    );
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Chat provider returned {status}: {}",
            truncate_for_error(&body, 600)
        ));
    }

    let streamed = openai_stream::read_openai_answer_stream(response, app, answer_event_id).await?;
    let answer = streamed.answer.trim().to_string();
    if answer.is_empty() {
        return Err("Chat provider returned an empty response".to_string());
    }
    let claims = claims::fallback_claims_from_answer(&answer, &agent_run.retrieval_run.citations);
    let answer =
        claims::strip_known_inline_citation_labels(&answer, &agent_run.retrieval_run.citations);
    Ok(AskAnswerResult {
        answer,
        reasoning_content: optional_non_empty(streamed.reasoning_content),
        claims,
    })
}

pub(crate) async fn test_openai_compatible_provider(
    provider: &OpenAiCompatibleProvider,
) -> Result<(), String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| format!("Failed to create provider test client: {err}"))?;
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&provider.base_url)
    );
    let request = OpenAiChatRequest {
        model: provider.model.clone(),
        temperature: 0.0,
        stream: None,
        messages: vec![text_message("user", "Reply with OK only.".to_string())],
    };
    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = &provider.api_key {
        builder = builder.bearer_auth(api_key);
    }

    let response = builder
        .send()
        .await
        .map_err(|err| format!("Provider test request failed: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Provider test returned {status}: {}",
            truncate_for_error(&body, 600)
        ));
    }

    let response = response
        .json::<OpenAiChatResponse>()
        .await
        .map_err(|err| format!("Failed to decode provider test response: {err}"))?;
    response
        .choices
        .into_iter()
        .next()
        .map(|choice| extract_chat_response_text(&choice.message.content))
        .filter(|text| !text.trim().is_empty())
        .map(|_| ())
        .ok_or_else(|| "Provider test returned an empty response".to_string())
}

pub(crate) fn answer_language_for_question(question: &str, locale: Option<&str>) -> &'static str {
    if contains_han_text(question) {
        return "Chinese";
    }
    match locale.unwrap_or("en").to_lowercase().as_str() {
        "zh" | "zh-cn" | "cn" => "Chinese",
        _ => "English",
    }
}

fn contains_han_text(value: &str) -> bool {
    value.chars().any(|ch| {
        ('\u{4E00}'..='\u{9FFF}').contains(&ch) || ('\u{3400}'..='\u{4DBF}').contains(&ch)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_answerability_json() {
        let decision = parse_answerability_decision(
            r#"{"status":"needs_more_evidence","reason":"Need method section","missing":["method"],"nextToolCall":{"tool":"open_section","args":{"query":"method"},"reason":"open method"}}"#,
        )
        .expect("decision");

        assert_eq!(decision.status, "needs_more_evidence");
        assert_eq!(
            decision.next_tool_call.expect("tool call").args["query"],
            serde_json::json!("method")
        );
    }

    #[test]
    fn parses_fenced_answerability_json() {
        let decision = parse_answerability_decision(
            "```json\n{\"status\":\"answerable\",\"reason\":\"Enough evidence\"}\n```",
        )
        .expect("decision");

        assert_eq!(decision.status, "answerable");
        assert!(decision.next_tool_call.is_none());
    }

    #[test]
    fn rejects_more_evidence_without_tool() {
        let err = parse_answerability_decision(
            r#"{"status":"needs_more_evidence","reason":"Need more"}"#,
        )
        .expect_err("missing tool should fail");

        assert!(err.contains("without a tool call"));
    }

    #[test]
    fn normalizes_loose_answerability_json() {
        let decision = parse_answerability_decision(
            r#"{"status":"partial","reason":"Need sections","missing":"method; experiments","nextToolCall":"inspect_tree"}"#,
        )
        .expect("loose decision");

        assert_eq!(decision.status, "needs_more_evidence");
        assert_eq!(decision.missing, vec!["method", "experiments"]);
        let call = decision.next_tool_call.expect("tool call");
        assert_eq!(call.tool, "inspect_tree");
        assert_eq!(call.args, serde_json::json!({}));
    }

    #[test]
    fn rejects_unknown_tool_in_next_tool_call() {
        let err = parse_answerability_decision(
            r#"{"status":"needs_more_evidence","reason":"r","nextToolCall":{"tool":"hallucinated_tool","args":{}}}"#,
        )
        .expect_err("unknown tool should fail");
        assert!(err.contains("unknown tool"), "{err}");
    }

    #[test]
    fn rejects_non_object_args_in_next_tool_call() {
        let err = parse_answerability_decision(
            r#"{"status":"needs_more_evidence","reason":"r","nextToolCall":{"tool":"inspect_tree","args":"oops"}}"#,
        )
        .expect_err("non-object args should fail");
        assert!(err.contains("args"), "{err}");
    }

    #[test]
    fn rejects_vision_tool_when_vision_disabled() {
        let err = parse_answerability_decision_for_vision(
            r#"{"status":"needs_more_evidence","reason":"r","nextToolCall":{"tool":"analyze_visual","args":{}}}"#,
            false,
        )
        .expect_err("vision tool should be rejected when capability is off");
        assert!(err.contains("unknown tool"), "{err}");
    }

    #[test]
    fn accepts_vision_tool_when_vision_enabled() {
        let decision = parse_answerability_decision_for_vision(
            r#"{"status":"needs_more_evidence","reason":"r","nextToolCall":{"tool":"analyze_visual","args":{"page":3}}}"#,
            true,
        )
        .expect("vision tool should be allowed when capability is on");
        let call = decision.next_tool_call.expect("tool call");
        assert_eq!(call.tool, "analyze_visual");
    }

    #[test]
    fn clears_tool_call_when_status_not_needs_more_evidence() {
        let decision = parse_answerability_decision(
            r#"{"status":"answerable","reason":"r","nextToolCall":{"tool":"inspect_tree","args":{}}}"#,
        )
        .expect("answerable decision");
        assert_eq!(decision.status, "answerable");
        assert!(decision.next_tool_call.is_none());
    }

    #[test]
    fn parses_current_view_decision_json() {
        let decision = parse_current_view_decision(
            r#"{"relevance":"strong","mode":"viewport","shouldUseCurrentView":true,"reason":"deictic table question"}"#,
        )
        .expect("current view decision");

        assert_eq!(decision.relevance, "high");
        assert_eq!(decision.mode, "visible");
        assert!(decision.should_use_current_view);
    }

    #[test]
    fn current_view_none_never_uses_current_view() {
        let decision = parse_current_view_decision(
            r#"{"relevance":"low","mode":"none","shouldUseCurrentView":true,"reason":"global question"}"#,
        )
        .expect("current view decision");

        assert_eq!(decision.mode, "none");
        assert!(!decision.should_use_current_view);
    }

    #[test]
    fn answerability_prompt_distinguishes_overview_from_specific_method_questions() {
        let prompt = build_answerability_judge_prompt(
            "这篇文章讲了什么？原理是什么？",
            "summarize",
            &[],
            &[],
            false,
            &crate::model_catalog::ModelContextBudget::default(),
        )
        .expect("prompt");

        assert!(prompt.contains("title/abstract/page overview evidence can be enough"));
        assert!(prompt.contains("high-level \"principle/原理/机制\""));
        assert!(prompt.contains("specific method/design/algorithm questions"));
        assert!(prompt.contains("concrete architecture, steps, implementation details"));
        assert!(prompt.contains("open_pages with mode=\"full\""));
        assert!(prompt.contains("If no page is known"));
        assert!(prompt.contains("是什么指标"));

        let vision_prompt = build_answerability_judge_prompt(
            "图1说明了什么？",
            "explain",
            &[],
            &[],
            true,
            &crate::model_catalog::ModelContextBudget::default(),
        )
        .expect("vision prompt");
        assert!(vision_prompt.contains("open_visual"));
        assert!(vision_prompt.contains("analyze_visual"));
        assert!(vision_prompt.contains("analyze_page"));
    }

    #[test]
    fn answerability_prompt_includes_tool_feedback() {
        let prompt = build_answerability_judge_prompt(
            "Table 3 里面提到的 SWE-Pruner 是什么指标？",
            "explain",
            &[],
            &["search_chunks returned results but added no new citations".to_string()],
            false,
            &crate::model_catalog::ModelContextBudget::default(),
        )
        .expect("prompt");

        assert!(prompt.contains("Tool feedback from previous judge-requested calls"));
        assert!(prompt.contains("search_chunks returned results but added no new citations"));
        assert!(prompt.contains("do not request the same tool with the same arguments again"));
    }
}
