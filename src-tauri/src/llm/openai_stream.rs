use tauri::Emitter;

use super::chat::extract_chat_response_text;
use crate::{
    truncate_for_error, AnswerDeltaEventOutput, OpenAiChatChunk, OpenAiChatResponse,
    ReasoningDeltaEventOutput, StreamedChatText,
};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Strip leaked tool-call markup from an answer/reasoning string.
///
/// Some tool-calling models (e.g. DeepSeek "interleaved" reasoning models) emit
/// their tool-call syntax as plain text — `<|DSML|tool_calls>…</|DSML|tool_calls>`
/// or the DeepSeek special tokens — inside `content`/`reasoning_content`. Without
/// this, that raw markup shows up in the answer or the "Thinking" drawer as
/// garbage. The patterns are specific enough not to touch normal prose.
pub(crate) fn strip_tool_call_markup(text: &str) -> String {
    use regex::Regex;
    use std::sync::OnceLock;
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    let patterns = PATTERNS.get_or_init(|| {
        // Delimiters seen in the wild use ASCII `|` or full-width `｜` (U+FF5C),
        // sometimes doubled, e.g. `<｜｜DSML｜｜tool_calls>`. `[|｜]+` matches any run.
        [
            // <…|tool_calls> … </…|tool_calls> (whole pipe-delimited block)
            r"(?s)<[|\x{ff5c}]+[^|\x{ff5c}>]*[|\x{ff5c}]+tool_calls>.*?</[|\x{ff5c}]+[^|\x{ff5c}>]*[|\x{ff5c}]+tool_calls>",
            // DeepSeek special tokens (full-width pipe + ▁ separators)
            "(?s)<\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}>.*?<\u{ff5c}tool\u{2581}calls\u{2581}end\u{ff5c}>",
            // stray leftover markup tags (unterminated blocks / individual tags)
            r"</?[|\x{ff5c}]+[^|\x{ff5c}>]*[|\x{ff5c}]+(?:tool_calls|invoke|parameter|tool_call|tool_sep)[^>]*>",
        ]
        .iter()
        .filter_map(|pattern| Regex::new(pattern).ok())
        .collect()
    });
    let mut cleaned = std::borrow::Cow::Borrowed(text);
    for re in patterns {
        if let std::borrow::Cow::Owned(next) = re.replace_all(&cleaned, "") {
            cleaned = std::borrow::Cow::Owned(next);
        }
    }
    cleaned.trim().to_string()
}

/// Splits inline `<think>…</think>` reasoning out of the assistant `content`
/// stream, for providers (Qwen, GLM, many OpenAI-compatible endpoints) that
/// embed reasoning in the content instead of a separate `reasoning_content`
/// field. Stateful across deltas: a tag can be split across chunk boundaries, so
/// a partial tag prefix is held back until it completes.
#[derive(Default)]
struct ThinkSplitter {
    inside_think: bool,
    pending: String,
}

impl ThinkSplitter {
    /// Route one content delta into (answer_text, reasoning_text).
    fn feed(&mut self, delta: &str) -> (String, String) {
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(delta);
        let mut answer = String::new();
        let mut reasoning = String::new();
        loop {
            let needle = if self.inside_think {
                THINK_CLOSE
            } else {
                THINK_OPEN
            };
            if let Some(pos) = buf.find(needle) {
                let before = &buf[..pos];
                if self.inside_think {
                    reasoning.push_str(before);
                } else {
                    answer.push_str(before);
                }
                buf.drain(..pos + needle.len());
                self.inside_think = !self.inside_think;
            } else {
                // No complete tag; hold back any trailing partial-tag prefix.
                let hold = partial_tag_suffix_len(&buf, needle);
                let split_at = buf.len() - hold;
                let emit = &buf[..split_at];
                if self.inside_think {
                    reasoning.push_str(emit);
                } else {
                    answer.push_str(emit);
                }
                self.pending = buf[split_at..].to_string();
                break;
            }
        }
        (answer, reasoning)
    }

    /// Emit any held-back text at end of stream as literal content.
    fn flush(&mut self) -> (String, String) {
        let pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            (String::new(), String::new())
        } else if self.inside_think {
            (String::new(), pending)
        } else {
            (pending, String::new())
        }
    }
}

/// Byte length of the longest suffix of `buf` that is a proper prefix of
/// `needle` — text held back in case the next delta completes the tag.
fn partial_tag_suffix_len(buf: &str, needle: &str) -> usize {
    let max = buf.len().min(needle.len().saturating_sub(1));
    for k in (1..=max).rev() {
        if buf.ends_with(&needle[..k]) {
            return k;
        }
    }
    0
}

pub(crate) async fn read_openai_answer_stream(
    mut response: reqwest::Response,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<StreamedChatText, String> {
    let mut buffer = Vec::new();
    let mut answer = String::new();
    let mut reasoning_content = String::new();
    let mut splitter = ThinkSplitter::default();
    let mut chunk_count = 0usize;
    let mut total_bytes = 0usize;
    // Stop button: break the read loop when the user cancels, returning what has
    // streamed so far (the request is dropped, so the provider stops too).
    let cancel = answer_event_id.and_then(|id| crate::cancellation_token(app, id));
    loop {
        let next = if let Some(token) = &cancel {
            tokio::select! {
                _ = token.cancelled() => {
                    log::info!(
                        "chat stream cancelled by user event_id={}",
                        answer_event_id.unwrap_or("-")
                    );
                    break;
                }
                res = response.chunk() => res,
            }
        } else {
            response.chunk().await
        };
        let Some(chunk) = next.map_err(|err| format!("Failed to read chat stream: {err}"))? else {
            break;
        };
        chunk_count += 1;
        total_bytes += chunk.len();
        log::debug!(
            "chat stream chunk event_id={} chunk={} bytes={} total_bytes={} buffer_before={}",
            answer_event_id.unwrap_or("-"),
            chunk_count,
            chunk.len(),
            total_bytes,
            buffer.len()
        );
        let answer_chars_before = answer.chars().count();
        buffer.extend_from_slice(&chunk);
        drain_openai_sse_buffer(
            &mut buffer,
            &mut answer,
            &mut reasoning_content,
            &mut splitter,
            app,
            answer_event_id,
        )?;
        let answer_chars_after = answer.chars().count();
        if answer_chars_after > answer_chars_before {
            log::debug!(
                "chat stream parsed delta event_id={} chunk={} new_chars={} total_answer_chars={}",
                answer_event_id.unwrap_or("-"),
                chunk_count,
                answer_chars_after - answer_chars_before,
                answer_chars_after
            );
        }
    }
    drain_openai_sse_buffer(
        &mut buffer,
        &mut answer,
        &mut reasoning_content,
        &mut splitter,
        app,
        answer_event_id,
    )?;
    drain_openai_stream_tail(
        &mut buffer,
        &mut answer,
        &mut reasoning_content,
        &mut splitter,
        app,
        answer_event_id,
    )?;
    // Emit any text held back as a possible (but never-completed) partial tag.
    let (answer_tail, reasoning_tail) = splitter.flush();
    if !answer_tail.is_empty() {
        answer.push_str(&answer_tail);
        emit_answer_delta(app, answer_event_id, answer_tail);
    }
    if !reasoning_tail.is_empty() {
        reasoning_content.push_str(&reasoning_tail);
        emit_reasoning_delta(app, answer_event_id, reasoning_tail);
    }
    log::info!(
        "chat stream read complete event_id={} chunks={} total_bytes={} answer_chars={} reasoning_chars={}",
        answer_event_id.unwrap_or("-"),
        chunk_count,
        total_bytes,
        answer.chars().count(),
        reasoning_content.chars().count()
    );
    // Strip any tool-call markup the model leaked into the answer or its reasoning
    // (interleaved tool-calling models emit it as plain text). The live deltas may
    // have shown it briefly, but the final/persisted value is clean.
    Ok(StreamedChatText {
        answer: strip_tool_call_markup(&answer),
        reasoning_content: strip_tool_call_markup(&reasoning_content),
    })
}

fn drain_openai_sse_buffer(
    buffer: &mut Vec<u8>,
    answer: &mut String,
    reasoning_content: &mut String,
    splitter: &mut ThinkSplitter,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<(), String> {
    while let Some((index, delimiter_len)) = next_sse_frame_boundary(buffer) {
        let frame_bytes = buffer[..index].to_vec();
        buffer.drain(..index + delimiter_len);
        let frame = std::str::from_utf8(&frame_bytes)
            .map_err(|err| format!("Failed to decode chat stream frame: {err}"))?;
        log::debug!(
            "chat stream frame event_id={} bytes={} preview={}",
            answer_event_id.unwrap_or("-"),
            frame_bytes.len(),
            truncate_for_error(frame, 120)
        );
        handle_openai_sse_frame(
            frame,
            answer,
            reasoning_content,
            splitter,
            app,
            answer_event_id,
        )?;
    }
    Ok(())
}

fn next_sse_frame_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    match (
        find_byte_sequence(buffer, b"\n\n"),
        find_byte_sequence(buffer, b"\r\n\r\n"),
    ) {
        (Some(line_feed), Some(carriage_return)) if carriage_return < line_feed => {
            Some((carriage_return, 4))
        }
        (Some(line_feed), _) => Some((line_feed, 2)),
        (None, Some(carriage_return)) => Some((carriage_return, 4)),
        (None, None) => None,
    }
}

fn find_byte_sequence(buffer: &[u8], needle: &[u8]) -> Option<usize> {
    buffer
        .windows(needle.len())
        .position(|window| window == needle)
}

fn handle_openai_sse_frame(
    frame: &str,
    answer: &mut String,
    reasoning_content: &mut String,
    splitter: &mut ThinkSplitter,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<(), String> {
    let mut emit_delta = |delta| emit_answer_delta(app, answer_event_id, delta);
    let mut emit_reasoning_delta = |delta| emit_reasoning_delta(app, answer_event_id, delta);
    handle_openai_sse_frame_with_sink(
        frame,
        answer,
        reasoning_content,
        splitter,
        &mut emit_delta,
        &mut emit_reasoning_delta,
    )
}

fn handle_openai_sse_frame_with_sink<F, G>(
    frame: &str,
    answer: &mut String,
    reasoning_content: &mut String,
    splitter: &mut ThinkSplitter,
    on_delta: &mut F,
    on_reasoning_delta: &mut G,
) -> Result<(), String>
where
    F: FnMut(String),
    G: FnMut(String),
{
    for line in frame.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let chunk = serde_json::from_str::<OpenAiChatChunk>(data)
            .map_err(|err| format!("Failed to decode chat stream chunk: {err}"))?;
        for choice in chunk.choices {
            if let Some(delta) = choice.delta {
                if let Some(reasoning_delta) = delta
                    .reasoning_content
                    .or(delta.reasoning)
                    .filter(|value| !value.is_empty())
                {
                    reasoning_content.push_str(&reasoning_delta);
                    on_reasoning_delta(reasoning_delta);
                }
                if let Some(answer_delta) = delta.content.filter(|value| !value.is_empty()) {
                    // Pull any inline <think>…</think> out into the reasoning sink
                    // so the tags/reasoning never land in the visible answer.
                    let (answer_part, reasoning_part) = splitter.feed(&answer_delta);
                    if !reasoning_part.is_empty() {
                        reasoning_content.push_str(&reasoning_part);
                        on_reasoning_delta(reasoning_part);
                    }
                    if !answer_part.is_empty() {
                        answer.push_str(&answer_part);
                        on_delta(answer_part);
                    }
                }
            }
        }
    }
    Ok(())
}

fn drain_openai_stream_tail(
    buffer: &mut Vec<u8>,
    answer: &mut String,
    reasoning_content: &mut String,
    splitter: &mut ThinkSplitter,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<(), String> {
    let tail = std::str::from_utf8(buffer)
        .map_err(|err| format!("Failed to decode chat stream tail: {err}"))?
        .trim();
    if tail.is_empty() {
        buffer.clear();
        return Ok(());
    }
    if tail.starts_with("data:") {
        let frame = tail.to_string();
        buffer.clear();
        return handle_openai_sse_frame(
            &frame,
            answer,
            reasoning_content,
            splitter,
            app,
            answer_event_id,
        );
    }
    if let Ok(response) = serde_json::from_str::<OpenAiChatResponse>(tail) {
        log::warn!(
            "chat stream fallback non_sse_json event_id={} tail_chars={}",
            answer_event_id.unwrap_or("-"),
            tail.chars().count()
        );
        let text = response
            .choices
            .into_iter()
            .next()
            .map(|choice| extract_chat_response_text(&choice.message.content))
            .unwrap_or_default();
        if !text.is_empty() {
            // Same inline-<think> split for the non-SSE fallback body.
            let (answer_part, reasoning_part) = splitter.feed(&text);
            if !reasoning_part.is_empty() {
                reasoning_content.push_str(&reasoning_part);
                emit_reasoning_delta(app, answer_event_id, reasoning_part);
            }
            if !answer_part.is_empty() {
                answer.push_str(&answer_part);
                emit_answer_delta(app, answer_event_id, answer_part);
            }
        }
        buffer.clear();
        return Ok(());
    }
    Err("Chat provider returned an incomplete stream frame".to_string())
}

fn emit_answer_delta(app: &tauri::AppHandle, answer_event_id: Option<&str>, delta: String) {
    if let Some(event_id) = answer_event_id {
        let delta_chars = delta.chars().count();
        let delta_preview = truncate_for_error(&delta, 80);
        if let Err(err) = app.emit(
            "lumenfolio://answer-delta",
            AnswerDeltaEventOutput {
                event_id: event_id.to_string(),
                delta,
            },
        ) {
            log::warn!("Failed to emit answer delta: {err}");
        } else {
            log::debug!(
                "chat stream emitted delta event_id={} chars={} preview={}",
                event_id,
                delta_chars,
                delta_preview
            );
        }
    }
}

fn emit_reasoning_delta(app: &tauri::AppHandle, answer_event_id: Option<&str>, delta: String) {
    if let Some(event_id) = answer_event_id {
        let delta_chars = delta.chars().count();
        if let Err(err) = app.emit(
            "lumenfolio://reasoning-delta",
            ReasoningDeltaEventOutput {
                event_id: event_id.to_string(),
                delta,
            },
        ) {
            log::warn!("Failed to emit reasoning delta: {err}");
        } else {
            log::debug!(
                "chat stream emitted reasoning delta event_id={} chars={}",
                event_id,
                delta_chars
            );
        }
    }
}

/// One streamed tool call, reassembled from its `index`-keyed fragments.
#[derive(Default, Clone)]
pub(crate) struct UnifiedStreamToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Result of one streamed round of the agent loop.
pub(crate) struct UnifiedStreamRound {
    /// Full assistant content (think-split applied). For a tool round this is
    /// usually empty for native models, or the DSML markup for the text fallback.
    pub content: String,
    /// Reasoning/thinking text, emitted live to the drawer.
    pub reasoning: String,
    /// Native tool calls the model requested this round, in `index` order.
    pub tool_calls: Vec<UnifiedStreamToolCall>,
    /// Whether any answer delta was painted live. When true the caller must NOT
    /// re-emit the whole answer one-shot (that would duplicate the bubble).
    pub emitted_answer: bool,
}

/// Withholds live answer emission until the round's content is confirmed to be
/// prose rather than a tool-call markup block (the DSML text fallback, whose
/// blocks always open with `<|`/`<｜`). Without this, a non-conformant endpoint
/// that returns tool calls as `content` text would paint raw markup into the
/// answer bubble. Accumulation into the full `content` string is unaffected —
/// this only gates what reaches the UI. Adds at most a 1–2 char delay for a
/// genuine answer that happens to start with `<`.
#[derive(Default)]
struct ContentGate {
    /// None = undecided, Some(true) = prose (emit), Some(false) = markup (hold).
    decided: Option<bool>,
    pending: String,
}

impl ContentGate {
    /// Feed a fresh answer fragment; return the slice that is safe to emit now.
    fn admit(&mut self, part: &str) -> String {
        match self.decided {
            Some(true) => part.to_string(),
            Some(false) => String::new(),
            None => {
                self.pending.push_str(part);
                let trimmed = self.pending.trim_start();
                if trimmed.is_empty() {
                    return String::new();
                }
                let mut chars = trimmed.chars();
                let first = chars.next().unwrap();
                if first != '<' {
                    self.decided = Some(true);
                    return std::mem::take(&mut self.pending);
                }
                match chars.next() {
                    None => String::new(), // only "<" so far — wait for the next char
                    Some(c) if c == '|' || c == '\u{ff5c}' => {
                        self.decided = Some(false);
                        String::new()
                    }
                    Some(_) => {
                        self.decided = Some(true);
                        std::mem::take(&mut self.pending)
                    }
                }
            }
        }
    }

    /// Any text held back at end of stream, emitted as prose unless ruled markup.
    fn flush(&mut self) -> String {
        if self.decided == Some(false) {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

/// Read one streaming round of the agent loop. Answer/reasoning content is
/// emitted live (true token streaming) while native `tool_calls` are accumulated
/// across chunks and returned so the loop can execute them. Cancellation mirrors
/// the plain answer stream: a stop aborts the read and surfaces GENERATION_STOPPED.
pub(crate) async fn read_unified_tool_round_stream(
    mut response: reqwest::Response,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<UnifiedStreamRound, String> {
    let mut buffer = Vec::new();
    let mut acc = UnifiedRoundAcc::default();
    let cancel = answer_event_id.and_then(|id| crate::cancellation_token(app, id));
    loop {
        let next = if let Some(token) = &cancel {
            tokio::select! {
                _ = token.cancelled() => {
                    return Err(crate::llm::agent_loop::GENERATION_STOPPED.to_string());
                }
                res = response.chunk() => res,
            }
        } else {
            response.chunk().await
        };
        let Some(chunk) = next.map_err(|err| format!("Failed to read agent stream: {err}"))? else {
            break;
        };
        buffer.extend_from_slice(&chunk);
        drain_unified_sse_buffer(&mut buffer, &mut acc, app, answer_event_id)?;
    }
    drain_unified_sse_buffer(&mut buffer, &mut acc, app, answer_event_id)?;
    drain_unified_stream_tail(&mut buffer, &mut acc, app, answer_event_id)?;
    // Emit (or just absorb) any text the think-splitter / content-gate held back.
    let (answer_tail, reasoning_tail) = acc.splitter.flush();
    if !reasoning_tail.is_empty() {
        acc.reasoning.push_str(&reasoning_tail);
        if !acc.tool_call_seen {
            emit_reasoning_delta(app, answer_event_id, reasoning_tail);
        }
    }
    if !answer_tail.is_empty() {
        acc.answer.push_str(&answer_tail);
    }
    let gated = acc.gate.flush();
    if !gated.is_empty() && !acc.tool_call_seen {
        emit_answer_delta(app, answer_event_id, gated);
        acc.emitted_answer = true;
    }
    Ok(UnifiedStreamRound {
        content: acc.answer,
        reasoning: acc.reasoning,
        tool_calls: acc.tool_calls,
        emitted_answer: acc.emitted_answer,
    })
}

#[derive(Default)]
struct UnifiedRoundAcc {
    answer: String,
    reasoning: String,
    splitter: ThinkSplitter,
    gate: ContentGate,
    tool_calls: Vec<UnifiedStreamToolCall>,
    tool_call_seen: bool,
    emitted_answer: bool,
}

fn drain_unified_sse_buffer(
    buffer: &mut Vec<u8>,
    acc: &mut UnifiedRoundAcc,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<(), String> {
    while let Some((index, delimiter_len)) = next_sse_frame_boundary(buffer) {
        let frame_bytes = buffer[..index].to_vec();
        buffer.drain(..index + delimiter_len);
        let frame = std::str::from_utf8(&frame_bytes)
            .map_err(|err| format!("Failed to decode agent stream frame: {err}"))?;
        handle_unified_frame(frame, acc, app, answer_event_id)?;
    }
    Ok(())
}

fn handle_unified_frame(
    frame: &str,
    acc: &mut UnifiedRoundAcc,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<(), String> {
    for line in frame.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let chunk = serde_json::from_str::<crate::OpenAiChatChunk>(data)
            .map_err(|err| format!("Failed to decode agent stream chunk: {err}"))?;
        for choice in chunk.choices {
            let Some(delta) = choice.delta else { continue };
            if let Some(calls) = delta.tool_calls {
                for call in calls {
                    acc.tool_call_seen = true;
                    while acc.tool_calls.len() <= call.index {
                        acc.tool_calls.push(UnifiedStreamToolCall::default());
                    }
                    let slot = &mut acc.tool_calls[call.index];
                    if let Some(id) = call.id {
                        if !id.is_empty() {
                            slot.id = id;
                        }
                    }
                    if let Some(func) = call.function {
                        if let Some(name) = func.name {
                            if !name.is_empty() {
                                slot.name = name;
                            }
                        }
                        if let Some(arguments) = func.arguments {
                            slot.arguments.push_str(&arguments);
                        }
                    }
                }
            }
            if let Some(reasoning_delta) = delta
                .reasoning_content
                .or(delta.reasoning)
                .filter(|value| !value.is_empty())
            {
                acc.reasoning.push_str(&reasoning_delta);
                if !acc.tool_call_seen {
                    emit_reasoning_delta(app, answer_event_id, reasoning_delta);
                }
            }
            if let Some(answer_delta) = delta.content.filter(|value| !value.is_empty()) {
                let (answer_part, reasoning_part) = acc.splitter.feed(&answer_delta);
                if !reasoning_part.is_empty() {
                    acc.reasoning.push_str(&reasoning_part);
                    if !acc.tool_call_seen {
                        emit_reasoning_delta(app, answer_event_id, reasoning_part);
                    }
                }
                if !answer_part.is_empty() {
                    acc.answer.push_str(&answer_part);
                    // Paint live only while this still looks like an answer round.
                    if !acc.tool_call_seen {
                        let emit = acc.gate.admit(&answer_part);
                        if !emit.is_empty() {
                            emit_answer_delta(app, answer_event_id, emit);
                            acc.emitted_answer = true;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Handle a trailing partial frame or a whole non-SSE JSON body (a provider that
/// ignored `stream: true` and returned one completion object). Keeps the loop
/// working against endpoints that don't actually stream.
fn drain_unified_stream_tail(
    buffer: &mut Vec<u8>,
    acc: &mut UnifiedRoundAcc,
    app: &tauri::AppHandle,
    answer_event_id: Option<&str>,
) -> Result<(), String> {
    let tail = std::str::from_utf8(buffer)
        .map_err(|err| format!("Failed to decode agent stream tail: {err}"))?
        .trim()
        .to_string();
    buffer.clear();
    if tail.is_empty() {
        return Ok(());
    }
    if tail.starts_with("data:") {
        return handle_unified_frame(&tail, acc, app, answer_event_id);
    }
    // Non-SSE fallback: parse a full chat-completion object for content + tool calls.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&tail) {
        let message = &value["choices"][0]["message"];
        if let Some(calls) = message["tool_calls"].as_array() {
            for call in calls {
                acc.tool_call_seen = true;
                acc.tool_calls.push(UnifiedStreamToolCall {
                    id: call["id"].as_str().unwrap_or_default().to_string(),
                    name: call["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    arguments: call["function"]["arguments"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                });
            }
        }
        let content = extract_chat_response_text(&message["content"]);
        if !content.is_empty() {
            let (answer_part, reasoning_part) = acc.splitter.feed(&content);
            if !reasoning_part.is_empty() {
                acc.reasoning.push_str(&reasoning_part);
                if !acc.tool_call_seen {
                    emit_reasoning_delta(app, answer_event_id, reasoning_part);
                }
            }
            if !answer_part.is_empty() {
                acc.answer.push_str(&answer_part);
                if !acc.tool_call_seen {
                    let emit = acc.gate.admit(&answer_part);
                    if !emit.is_empty() {
                        emit_answer_delta(app, answer_event_id, emit);
                        acc.emitted_answer = true;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_gate_emits_plain_prose_immediately() {
        let mut gate = ContentGate::default();
        assert_eq!(gate.admit("The "), "The ");
        assert_eq!(gate.admit("answer."), "answer.");
        assert_eq!(gate.flush(), "");
    }

    #[test]
    fn content_gate_withholds_dsml_markup() {
        // Tool-call text fallback: opens with `<｜` — must never reach the bubble.
        let mut gate = ContentGate::default();
        assert_eq!(gate.admit("<\u{ff5c}"), "");
        assert_eq!(gate.admit("\u{ff5c}DSML\u{ff5c}\u{ff5c}tool_calls>"), "");
        assert_eq!(gate.flush(), "");
    }

    #[test]
    fn content_gate_keeps_prose_starting_with_angle_bracket() {
        // A genuine answer may start with "<" (e.g. "<5%"); only "<|" is markup.
        let mut gate = ContentGate::default();
        assert_eq!(gate.admit("<"), ""); // undecided on a lone "<"
        assert_eq!(gate.admit("5% of cases"), "<5% of cases");
    }

    #[test]
    fn content_gate_preserves_leading_whitespace_before_prose() {
        let mut gate = ContentGate::default();
        assert_eq!(gate.admit("  "), "");
        assert_eq!(gate.admit("hi"), "  hi");
    }

    #[test]
    fn sse_boundary_waits_for_complete_utf8_frame() {
        let mut buffer = b"data: {\"choices\":[{\"delta\":{\"content\":\"".to_vec();
        buffer.extend_from_slice("中文".as_bytes());

        assert_eq!(next_sse_frame_boundary(&buffer), None);

        buffer.extend_from_slice(b"\"}}]}\n\n");
        assert_eq!(
            next_sse_frame_boundary(&buffer),
            Some((buffer.len() - 2, 2))
        );
    }

    #[test]
    fn sse_boundary_supports_crlf_frames() {
        let buffer = b"data: {\"choices\":[]}\r\n\r\n";

        assert_eq!(next_sse_frame_boundary(buffer), Some((buffer.len() - 4, 4)));
    }

    #[test]
    fn sse_frame_emits_each_delta_incrementally() {
        let frame = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"你\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"好\"}}]}\n"
        );
        let mut answer = String::new();
        let mut reasoning_content = String::new();
        let mut deltas = Vec::new();
        let mut reasoning_deltas = Vec::new();

        handle_openai_sse_frame_with_sink(
            frame,
            &mut answer,
            &mut reasoning_content,
            &mut ThinkSplitter::default(),
            &mut |delta| {
                deltas.push(delta);
            },
            &mut |delta| {
                reasoning_deltas.push(delta);
            },
        )
        .expect("stream frame should parse");

        assert_eq!(answer, "你好");
        assert_eq!(reasoning_content, "");
        assert_eq!(deltas, vec!["你".to_string(), "好".to_string()]);
        assert!(reasoning_deltas.is_empty());
    }

    #[test]
    fn sse_frame_captures_deepseek_reasoning_content() {
        let frame = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"先分析\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"答案\"}}]}\n"
        );
        let mut answer = String::new();
        let mut reasoning_content = String::new();
        let mut deltas = Vec::new();
        let mut reasoning_deltas = Vec::new();

        handle_openai_sse_frame_with_sink(
            frame,
            &mut answer,
            &mut reasoning_content,
            &mut ThinkSplitter::default(),
            &mut |delta| {
                deltas.push(delta);
            },
            &mut |delta| {
                reasoning_deltas.push(delta);
            },
        )
        .expect("stream frame should parse");

        assert_eq!(answer, "答案");
        assert_eq!(reasoning_content, "先分析");
        assert_eq!(deltas, vec!["答案".to_string()]);
        assert_eq!(reasoning_deltas, vec!["先分析".to_string()]);
    }

    #[test]
    fn inline_think_block_is_routed_to_reasoning() {
        let mut splitter = ThinkSplitter::default();
        let (answer, reasoning) = splitter.feed("<think>let me reason</think>The answer.");
        assert_eq!(answer, "The answer.");
        assert_eq!(reasoning, "let me reason");
        assert!(!splitter.inside_think);
    }

    #[test]
    fn inline_think_tags_split_across_deltas() {
        // The tag itself, and the boundaries, arrive in separate streamed chunks.
        let mut splitter = ThinkSplitter::default();
        let mut answer = String::new();
        let mut reasoning = String::new();
        for chunk in ["Hello <thi", "nk>reason ", "here</thi", "nk> world"] {
            let (a, r) = splitter.feed(chunk);
            answer.push_str(&a);
            reasoning.push_str(&r);
        }
        let (a, r) = splitter.flush();
        answer.push_str(&a);
        reasoning.push_str(&r);
        assert_eq!(answer, "Hello  world");
        assert_eq!(reasoning, "reason here");
    }

    #[test]
    fn strips_leaked_tool_call_markup() {
        let dsml = "Here is the answer.\n\
            <|DSML|tool_calls>\n\
            <|DSML|invoke name=\"search_chunks\">\n\
            <|DSML|parameter name=\"query\" string=\"true\">DR Tulu</|DSML|parameter>\n\
            </|DSML|invoke>\n\
            </|DSML|tool_calls>";
        let cleaned = strip_tool_call_markup(dsml);
        assert_eq!(cleaned, "Here is the answer.");
        assert!(!cleaned.contains("DSML"));
        assert!(!cleaned.contains("tool_calls"));
    }

    #[test]
    fn strips_fullwidth_doubled_pipe_markup() {
        // Exact format observed from deepseek-v4-flash (full-width ｜, doubled).
        let p = "\u{ff5c}\u{ff5c}DSML\u{ff5c}\u{ff5c}";
        let raw = format!(
            "<{p}tool_calls>\n<{p}invoke name=\"search_chunks\">\n<{p}parameter name=\"query\" string=\"true\">DR Tulu-8B outperforms</{p}parameter>\n</{p}invoke>\n</{p}tool_calls>"
        );
        let cleaned = strip_tool_call_markup(&raw);
        assert_eq!(cleaned, "");
        assert!(!cleaned.contains("tool_calls"));
        assert!(!cleaned.contains("DSML"));
        // With surrounding prose, only the markup block is removed.
        let mixed = format!("Here is the summary.\n{raw}");
        assert_eq!(strip_tool_call_markup(&mixed), "Here is the summary.");
    }

    #[test]
    fn strips_deepseek_tool_call_tokens() {
        // Block delimited by DeepSeek <｜tool▁calls▁begin｜> … <｜tool▁calls▁end｜>.
        let wrapped = "答案。<\u{ff5c}tool\u{2581}calls\u{2581}begin\u{ff5c}>foo<\u{ff5c}tool\u{2581}calls\u{2581}end\u{ff5c}>";
        assert_eq!(strip_tool_call_markup(wrapped), "答案。");
    }

    #[test]
    fn tool_call_stripper_leaves_normal_prose() {
        let prose = "We compare A | B, and note x < y in Table 2.";
        assert_eq!(strip_tool_call_markup(prose), prose);
    }

    #[test]
    fn non_think_angle_brackets_pass_through() {
        // A stray "<" that is not a <think> tag must not be swallowed.
        let mut splitter = ThinkSplitter::default();
        let (answer, reasoning) = splitter.feed("a < b and c <tag> d");
        let (tail_a, _) = splitter.flush();
        assert_eq!(format!("{answer}{tail_a}"), "a < b and c <tag> d");
        assert_eq!(reasoning, "");
    }
}
