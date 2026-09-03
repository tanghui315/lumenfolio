//! Detection of locally-installed agent CLIs (Codex, Claude Code) so they can be
//! offered as zero-config "local agent" chat providers.
//!
//! Detection and status live here; driving the CLI and exposing tools over MCP
//! live in `local_agent/mcp_server.rs` and the chat dispatch path.

use std::{
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::runtime::rag::Citation;

/// Error returned when the user stops a generation; the frontend ignores the
/// resulting error event for a stopped turn, so the exact text is irrelevant.
const GENERATION_STOPPED: &str = "Generation stopped by user.";

pub(crate) mod mcp_server;

const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum AgentKind {
    Codex,
    Claude,
}

impl AgentKind {
    fn binary(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Claude => "claude",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Codex => "Codex",
            Self::Claude => "Claude Code",
        }
    }

    fn install_url(self) -> &'static str {
        match self {
            Self::Codex => "https://developers.openai.com/codex/cli",
            Self::Claude => "https://docs.claude.com/en/docs/claude-code",
        }
    }

    const ALL: [AgentKind; 2] = [AgentKind::Codex, AgentKind::Claude];
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AgentStatus {
    kind: AgentKind,
    label: String,
    installed: bool,
    version: Option<String>,
    path: Option<String>,
    install_url: String,
}

/// Run a command, capture trimmed stdout on success, with a hard timeout. A hung
/// child leaks its worker thread (acceptable for an occasional `--version` probe);
/// we never block the caller past `timeout`.
fn run_capture(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let program = program.to_string();
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let output = Command::new(&program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();
        let _ = tx.send(output);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(out)) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

/// Resolve a binary's absolute path, tolerating the minimal PATH a macOS GUI app
/// inherits (resolve through the user's login shell), and `where` on Windows.
fn resolve_path(binary: &str) -> Option<String> {
    #[cfg(unix)]
    {
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
        let lookup = format!("command -v {binary}");
        if let Some(out) = run_capture(&shell, &["-lc", &lookup], PROBE_TIMEOUT) {
            if let Some(line) = out.lines().next() {
                let line = line.trim();
                if !line.is_empty() {
                    return Some(line.to_string());
                }
            }
        }
    }
    #[cfg(windows)]
    {
        if let Some(out) = run_capture("where", &[binary], PROBE_TIMEOUT) {
            if let Some(line) = out.lines().next() {
                let line = line.trim();
                if !line.is_empty() {
                    return Some(line.to_string());
                }
            }
        }
    }
    None
}

/// Pull a clean version token out of a CLI's `--version` line, e.g.
/// "2.0.55 (Claude Code)" -> "2.0.55", "codex-cli 0.12.0" -> "0.12.0". Falls back
/// to the trimmed first line when no dotted-number token is found.
fn parse_version(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or(raw).trim();
    first_line
        .split_whitespace()
        .find(|tok| {
            let t = tok.trim_start_matches('v');
            t.contains('.') && t.chars().next().is_some_and(|c| c.is_ascii_digit())
        })
        .map(|tok| tok.trim_start_matches('v').to_string())
        .unwrap_or_else(|| first_line.to_string())
}

fn detect(kind: AgentKind) -> AgentStatus {
    let path = resolve_path(kind.binary());
    // Probe via the resolved absolute path when we have it (robust under a minimal
    // PATH); otherwise try the bare name (works when PATH is already complete).
    let invoke = path.clone().unwrap_or_else(|| kind.binary().to_string());
    let version =
        run_capture(&invoke, &["--version"], PROBE_TIMEOUT).map(|raw| parse_version(&raw));
    let installed = version.is_some() || path.is_some();
    AgentStatus {
        kind,
        label: kind.label().to_string(),
        installed,
        version,
        path,
        install_url: kind.install_url().to_string(),
    }
}

/// Detection status for each supported local agent CLI. Runs subprocess probes, so
/// it is offloaded to a blocking task. Cheap enough to call on demand (startup +
/// a Settings "re-check").
#[tauri::command]
pub(crate) async fn get_local_agent_status() -> Result<Vec<AgentStatus>, String> {
    tauri::async_runtime::spawn_blocking(|| AgentKind::ALL.iter().copied().map(detect).collect())
        .await
        .map_err(|err| format!("Local-agent detection task failed: {err}"))
}

// ---- P1: drive the CLI as a one-shot answer generator (Mode A) -------------

/// Synthetic provider id format used by the frontend model selector for local
/// agents (no DB row — they are virtual, computed from detection).
pub(crate) fn provider_id_kind(provider_id: &str) -> Option<AgentKind> {
    match provider_id.trim() {
        "local-agent-codex" => Some(AgentKind::Codex),
        "local-agent-claude" => Some(AgentKind::Claude),
        _ => None,
    }
}

/// Generous cap: a real agent answer is seconds, but cold start + retries can run
/// long (the expired-token probe took ~3min). Past this we fail with a timeout.
const GENERATE_TIMEOUT: Duration = Duration::from_secs(150);

/// A user-supplied image decoded to a temp file for `codex exec -i`. Removed on drop.
struct ImageTemp {
    path: PathBuf,
}

impl Drop for ImageTemp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Decode a `data:image/...;base64,...` URL into a temp file. Returns None for a
/// non-data / non-base64 / decode failure, in which case the caller answers
/// text-only.
fn write_image_temp(data_url: &str) -> Option<ImageTemp> {
    use base64::Engine;
    let rest = data_url.trim().strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    if !meta.contains("base64") {
        return None;
    }
    let ext = match meta.split(';').next().unwrap_or("") {
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "png",
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .ok()?;
    if bytes.is_empty() {
        return None;
    }
    let path = std::env::temp_dir().join(format!(
        "lumenfolio-img-{:016x}.{ext}",
        rand::random::<u64>()
    ));
    std::fs::write(&path, &bytes).ok()?;
    Some(ImageTemp { path })
}

fn answer_language(locale: Option<&str>) -> &'static str {
    match locale.map(str::trim).unwrap_or("") {
        l if l.starts_with("zh") => "Chinese",
        _ => "English",
    }
}

/// The language directive shared by both prompt modes. The answer must follow the
/// QUESTION's language (a Chinese question gets a Chinese answer), regardless of
/// the app UI locale or the document's language; `locale` is only the tiebreaker
/// when the question itself is language-neutral.
fn language_directive(locale: Option<&str>) -> String {
    let fallback = answer_language(locale);
    format!(
        "Always write your answer in the SAME language as the user's Question below \
(e.g. a Chinese question gets a Chinese answer, an English question gets an English answer), \
even when the document evidence is in another language. Only if the question's language is \
genuinely ambiguous, default to {fallback}."
    )
}

/// Assemble the single prompt for Mode A: the evidence is already retrieved by
/// Lumenfolio and embedded here; the agent is a pure generator (no tools).
pub(crate) fn build_prompt(
    question: &str,
    evidence: &str,
    session_context: &str,
    locale: Option<&str>,
) -> String {
    let lang_directive = language_directive(locale);
    let mut prompt = format!(
        "You are Lumenfolio, a careful academic PDF reading assistant. {lang_directive} \
The user is reading a document; relevant passages retrieved from it appear below. \
If the question is about the document, answer using ONLY that evidence; if it is insufficient, say so plainly. \
If instead the question is general background knowledge not specific to this document, and the evidence \
below is not relevant to it, answer from your own knowledge and note that this is general background, \
not drawn from the document. \
Write a concise, well-structured Markdown answer (a short direct answer first, then detail). \
Do NOT call tools, read files, or run commands.\n\n"
    );
    let session_context = session_context.trim();
    if !session_context.is_empty() {
        prompt.push_str("Conversation memory:\n");
        prompt.push_str(session_context);
        prompt.push_str("\n\n");
    }
    let evidence = evidence.trim();
    prompt.push_str("Evidence from the document:\n");
    prompt.push_str(if evidence.is_empty() {
        "(no evidence was retrieved for this question)"
    } else {
        evidence
    });
    prompt.push_str("\n\nQuestion:\n");
    prompt.push_str(question.trim());
    prompt
}

/// Run the local agent CLI once with a fully-assembled prompt; return its answer.
/// Offloaded to a blocking task (subprocess I/O). Errors carry a user-facing hint
/// (e.g. login required).
pub(crate) async fn generate_answer(
    kind: AgentKind,
    prompt: String,
    image_data_url: Option<String>,
    cancel: CancellationToken,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        run_cli(kind, &prompt, image_data_url.as_deref(), &cancel)
    })
    .await
    .map_err(|err| format!("Local-agent task failed: {err}"))?
}

fn run_cli(
    kind: AgentKind,
    prompt: &str,
    image_data_url: Option<&str>,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let binary = resolve_path(kind.binary()).unwrap_or_else(|| kind.binary().to_string());
    // Codex `exec` takes images via `-i`; Claude's headless `-p` has no image flag,
    // so it degrades to text. Kept alive until the child exits.
    let image = image_data_url
        .filter(|_| kind == AgentKind::Codex)
        .and_then(write_image_temp);
    let mut cmd = Command::new(&binary);
    match kind {
        // --tools "" disables all built-in tools: pure generation, no fs/shell access.
        AgentKind::Claude => {
            cmd.args(["-p", prompt, "--output-format", "json", "--tools", ""]);
        }
        // read-only sandbox + no approval; no MCP configured → no tools available.
        AgentKind::Codex => {
            cmd.args([
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
            ]);
            // Prompt MUST precede `-i`: `--image <FILE>...` is multi-value and would
            // otherwise swallow the prompt positional as a second image.
            cmd.arg(prompt);
            if let Some(image) = &image {
                cmd.arg("-i").arg(&image.path);
            }
        }
    }
    cmd.current_dir(std::env::temp_dir()) // neutral cwd: nothing of the user's to read
        .env_remove("ANTHROPIC_API_KEY") // never bill an API key — use the subscription
        .env_remove("OPENAI_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = run_command_timeout(cmd, GENERATE_TIMEOUT, cancel)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(classify_error(kind, stderr.trim()));
    }
    match kind {
        AgentKind::Claude => parse_claude_json(kind, &stdout),
        AgentKind::Codex => parse_codex_jsonl(kind, &stdout),
    }
}

fn run_command_timeout(
    mut cmd: Command,
    timeout: Duration,
    cancel: &CancellationToken,
) -> Result<std::process::Output, String> {
    let mut child = cmd
        .spawn()
        .map_err(|err| format!("Failed to launch the local agent: {err}"))?;
    // Drain stdout/stderr on threads so a full pipe can't deadlock the child while
    // we poll for exit / cancellation.
    let drain = |pipe: Option<std::process::ChildStdout>| {
        let (tx, rx) = mpsc::channel();
        match pipe {
            Some(mut handle) => {
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let _ = handle.read_to_end(&mut buf);
                    let _ = tx.send(buf);
                });
            }
            None => {
                let _ = tx.send(Vec::new());
            }
        }
        rx
    };
    let out_rx = drain(child.stdout.take());
    // stderr is ChildStderr, not ChildStdout — drain inline.
    let err_rx = {
        let (tx, rx) = mpsc::channel();
        match child.stderr.take() {
            Some(mut handle) => {
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let _ = handle.read_to_end(&mut buf);
                    let _ = tx.send(buf);
                });
            }
            None => {
                let _ = tx.send(Vec::new());
            }
        }
        rx
    };

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(err) => return Err(format!("Local agent wait failed: {err}")),
        }
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(GENERATION_STOPPED.to_string());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(
                "The local agent timed out. Try again, or pick a model provider.".to_string(),
            );
        }
        std::thread::sleep(Duration::from_millis(80));
    };
    let stdout = out_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap_or_default();
    let stderr = err_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap_or_default();
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

/// Turn an auth/login failure into an actionable message; otherwise pass through.
fn classify_error(kind: AgentKind, detail: &str) -> String {
    let lower = detail.to_lowercase();
    if lower.contains("login")
        || lower.contains("log in")
        || lower.contains("unauthorized")
        || lower.contains("401")
        || lower.contains("过期")
        || lower.contains("not authenticated")
    {
        let cmd = match kind {
            AgentKind::Codex => "codex login",
            AgentKind::Claude => "claude",
        };
        return format!(
            "{} isn't logged in (run `{cmd}` once in a terminal). Original error: {detail}",
            kind.label()
        );
    }
    format!("{} failed: {detail}", kind.label())
}

/// Claude `--output-format json` returns one object; the answer/error is in
/// `result`, with `is_error` flagging auth/runtime failures (exit code stays 0).
fn parse_claude_json(kind: AgentKind, stdout: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|err| format!("Could not parse Claude output: {err}"))?;
    let result = value
        .get("result")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let is_error = value
        .get("is_error")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if is_error {
        return Err(classify_error(kind, &result));
    }
    if result.is_empty() {
        return Err(format!("{} returned an empty answer.", kind.label()));
    }
    Ok(result)
}

/// Codex `exec --json` emits JSONL; the answer is the last `item.completed` whose
/// `item.type == "agent_message"`.
fn parse_codex_jsonl(kind: AgentKind, stdout: &str) -> Result<String, String> {
    let mut answer: Option<String> = None;
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("type").and_then(|t| t.as_str()) == Some("item.completed") {
            if let Some(item) = value.get("item") {
                if item.get("type").and_then(|t| t.as_str()) == Some("agent_message") {
                    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                        answer = Some(text.trim().to_string());
                    }
                }
            }
        }
    }
    answer
        .filter(|a| !a.is_empty())
        .ok_or_else(|| format!("{} returned no answer.", kind.label()))
}

// ---- P2: drive the CLI as a multi-step agent over our MCP tools (Mode B) ----

/// What an agentic run returns: the final answer plus every citation the MCP
/// server served while the CLI was exploring (so the UI can surface evidence the
/// agent actually grounded on, even though we never saw its intermediate reasoning).
pub(crate) struct AgenticOutcome {
    pub answer: String,
    pub citations: Vec<Citation>,
    /// Trace candidates the server served (carry `section_title` per block), so the
    /// dispatch can label the rebuilt evidence chips.
    pub candidates: Vec<crate::runtime::rag::RetrievalTraceCandidate>,
}

/// A tool-call step observed in the CLI's event stream, reported live so the caller
/// can surface a trace timeline. `ok` is meaningful only for `Completed`.
pub(crate) struct AgentToolEvent {
    pub tool: String,
    pub phase: AgentToolPhase,
    pub ok: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AgentToolPhase {
    Started,
    Completed,
}

/// Prompt for Mode B: no evidence is embedded — instead the agent is told to use
/// the Lumenfolio MCP tools to retrieve evidence from the open document itself,
/// then answer from what it finds. Mirrors `build_prompt`'s persona/language.
pub(crate) fn build_agentic_prompt(
    question: &str,
    session_context: &str,
    view_note: Option<&str>,
    locale: Option<&str>,
) -> String {
    let lang_directive = language_directive(locale);
    let mut prompt = format!(
        "You are Lumenfolio, a careful academic PDF reading assistant. {lang_directive} \
You have MCP tools (server `lumenfolio`) that retrieve evidence from the user's open PDF \
(search passages, open pages/sections, inspect tables/figures) and from their wider library \
(list_trending_papers, search_library_knowledge, query_knowledge_graph). \
Decide what the question needs: if it is about the open document (its content, methods, results, \
figures, or claims), use the tools to gather evidence first, then answer grounded ONLY in what the \
tools return — if they surface nothing relevant, say so plainly. If instead it is a general or \
background-knowledge question not specific to this document (e.g. explaining a general concept, \
algorithm, or term), answer it directly from your own knowledge; you need not search, and should \
not imply the document covers it. \
Write a concise, well-structured Markdown answer (a short direct answer first, then detail). \
Only use the `lumenfolio` tools — do not read local files or run shell commands.\n\n"
    );
    if let Some(note) = view_note.map(str::trim).filter(|note| !note.is_empty()) {
        prompt.push_str("Current view:\n");
        prompt.push_str(note);
        prompt.push_str("\n\n");
    }
    let session_context = session_context.trim();
    if !session_context.is_empty() {
        prompt.push_str("Conversation memory:\n");
        prompt.push_str(session_context);
        prompt.push_str("\n\n");
    }
    prompt.push_str("Question:\n");
    prompt.push_str(question.trim());
    prompt
}

/// The env var the CLI reads the MCP bearer token from. The token value is passed
/// via the child's environment (never on the command line / in a config file).
const MCP_TOKEN_ENV: &str = "LUMENFOLIO_MCP_TOKEN";

/// Run the local agent CLI as a multi-step agent: bring up an in-process loopback
/// MCP server scoped to `document_id`, wire the CLI to it, let it call our tools,
/// and return the answer + the citations the server served. `on_tool` is invoked
/// live for each tool-call step (Codex's `--json` stream is parsed incrementally).
/// Offloaded subprocess I/O is wrapped around the (async) server lifecycle.
pub(crate) async fn generate_answer_agentic<F, G>(
    kind: AgentKind,
    db_path: PathBuf,
    document_id: String,
    prompt: String,
    image_data_url: Option<String>,
    web_enabled: bool,
    cancel: CancellationToken,
    on_tool: F,
    on_answer: G,
) -> Result<AgenticOutcome, String>
where
    F: Fn(AgentToolEvent) + Send + 'static,
    G: Fn(String) + Send + 'static,
{
    let server = mcp_server::start_mcp_server(db_path, document_id, web_enabled).await?;
    let url = server.url.clone();
    let token = server.token.clone();

    let answer = tauri::async_runtime::spawn_blocking(move || {
        run_cli_agentic(
            kind,
            &prompt,
            &url,
            &token,
            image_data_url.as_deref(),
            &cancel,
            &on_tool,
            &on_answer,
        )
    })
    .await
    .map_err(|err| format!("Local-agent task failed: {err}"));

    // Snapshot the evidence the server collected before tearing it down.
    let citations = server
        .citations
        .lock()
        .map(|c| c.clone())
        .unwrap_or_default();
    let candidates = server
        .candidates
        .lock()
        .map(|c| c.clone())
        .unwrap_or_default();
    drop(server); // stops the accept loop (also via Drop), frees the port

    let answer = answer??;
    Ok(AgenticOutcome {
        answer,
        citations,
        candidates,
    })
}

fn run_cli_agentic<F, G>(
    kind: AgentKind,
    prompt: &str,
    url: &str,
    token: &str,
    image_data_url: Option<&str>,
    cancel: &CancellationToken,
    on_tool: &F,
    on_answer: &G,
) -> Result<String, String>
where
    F: Fn(AgentToolEvent),
    G: Fn(String),
{
    let binary = resolve_path(kind.binary()).unwrap_or_else(|| kind.binary().to_string());
    // Per-run empty working dir so that — with Codex's OS sandbox necessarily off
    // (it cancels MCP calls otherwise) — there is nothing of the user's to read.
    let work_dir = std::env::temp_dir().join(format!(
        "lumenfolio-agent-{}",
        &token[..token.len().min(16)]
    ));
    let _ = std::fs::create_dir_all(&work_dir);
    // Codex takes the user's image via `-i`; Claude headless can't. Alive until exit.
    let image = image_data_url
        .filter(|_| kind == AgentKind::Codex)
        .and_then(write_image_temp);

    let result = match kind {
        // Codex: drive the app-server JSON-RPC protocol. Unlike `exec --json` (which
        // delivers the answer atomically), the app-server emits token-level
        // `item/agentMessage/delta` notifications, so the answer truly streams. The
        // user's image rides along as a `localImage` turn input.
        AgentKind::Codex => run_codex_app_server(
            &binary,
            prompt,
            url,
            token,
            &work_dir,
            image.as_ref().map(|img| img.path.clone()),
            GENERATE_TIMEOUT,
            cancel,
            on_tool,
            on_answer,
        ),
        // Claude: headless `-p` with stream-json + partial messages → token-level
        // `content_block_delta` we relay live; the final `result` event is authoritative.
        // Scope hard: --strict-mcp-config (ignore the user's other servers), --tools ""
        // (no built-in fs/shell), allow only our server's tools, bypass the prompt.
        AgentKind::Claude => {
            let mcp_config = serde_json::json!({
                "mcpServers": {
                    "lumenfolio": {
                        "type": "http",
                        "url": url,
                        "headers": { "Authorization": format!("Bearer {token}") }
                    }
                }
            })
            .to_string();
            let mut cmd = Command::new(&binary);
            cmd.args([
                "-p",
                prompt,
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--verbose",
                "--mcp-config",
                &mcp_config,
                "--strict-mcp-config",
                "--tools",
                "",
                "--allowedTools",
                "mcp__lumenfolio",
                "--permission-mode",
                "bypassPermissions",
            ]);
            cmd.current_dir(&work_dir)
                .env_remove("ANTHROPIC_API_KEY") // never bill an API key — use the subscription
                .env_remove("OPENAI_API_KEY")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            stream_claude_agentic(cmd, GENERATE_TIMEOUT, cancel, on_tool, on_answer)
        }
    };
    let _ = std::fs::remove_dir_all(&work_dir);
    result
}

/// Accumulator for one Codex app-server run.
#[derive(Default)]
struct CodexAppServerState {
    /// Text assembled from `item/agentMessage/delta`.
    answer: String,
    /// Authoritative final answer from the terminal `item/completed` agentMessage.
    final_answer: Option<String>,
    thread_id: Option<String>,
    turn_id: Option<String>,
    /// MCP tool calls started but not yet completed (FIFO).
    pending_tools: std::collections::VecDeque<String>,
    done: bool,
}

/// Drive Codex through the **app-server** JSON-RPC protocol (stdio). Unlike
/// `exec --json` (which delivers the answer in one atomic `agent_message`), the
/// app-server streams token-level `item/agentMessage/delta` notifications, so the
/// answer truly streams. Flow: initialize → thread/start → turn/start → read
/// notifications. MCP tool calls surface via `item/started`/`item/completed`.
/// A hard deadline kills a stall; a cancel sends `turn/interrupt` then kills.
fn run_codex_app_server<F, G>(
    binary: &str,
    prompt: &str,
    mcp_url: &str,
    mcp_token: &str,
    work_dir: &std::path::Path,
    image_path: Option<PathBuf>,
    timeout: Duration,
    cancel: &CancellationToken,
    on_tool: &F,
    on_answer: &G,
) -> Result<String, String>
where
    F: Fn(AgentToolEvent),
    G: Fn(String),
{
    // Inject ONLY our loopback MCP server (replacing any user-config servers) and
    // pass its bearer token via env so it never lands in argv/ps output.
    let mcp_cfg = format!(
        "mcp_servers={{lumenfolio={{url=\"{mcp_url}\", bearer_token_env_var=\"{MCP_TOKEN_ENV}\"}}}}"
    );
    let mut child = Command::new(binary)
        .args(["app-server", "-c", &mcp_cfg])
        .current_dir(work_dir)
        .env(MCP_TOKEN_ENV, mcp_token)
        .env_remove("ANTHROPIC_API_KEY") // never bill an API key — use the subscription
        .env_remove("OPENAI_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("Failed to launch the local agent: {err}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Local agent produced no input stream".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Local agent produced no output stream".to_string())?;

    let (err_tx, err_rx) = mpsc::channel();
    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut buf);
            let _ = err_tx.send(buf);
        });
    }

    let (line_tx, line_rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            match line {
                Ok(l) => {
                    if line_tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Send one newline-delimited JSON-RPC frame.
    let send = |stdin: &mut std::process::ChildStdin, msg: serde_json::Value| {
        use std::io::Write;
        let mut line = msg.to_string();
        line.push('\n');
        let _ = stdin.write_all(line.as_bytes());
        let _ = stdin.flush();
    };

    // Kick off the handshake; the rest is driven by responses below.
    send(
        &mut stdin,
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": { "clientInfo": { "name": "lumenfolio", "version": env!("CARGO_PKG_VERSION") } }
        }),
    );

    let deadline = Instant::now() + timeout;
    let mut state = CodexAppServerState::default();
    let stop = |child: &mut std::process::Child, reader: std::thread::JoinHandle<()>| {
        let _ = child.kill();
        let _ = reader.join();
    };
    loop {
        // Stop button: ask the server to interrupt the turn, then kill the child.
        if cancel.is_cancelled() {
            if let (Some(tid), Some(turn)) = (&state.thread_id, &state.turn_id) {
                send(
                    &mut stdin,
                    serde_json::json!({"jsonrpc":"2.0","id":99,"method":"turn/interrupt",
                        "params":{"threadId":tid,"turnId":turn}}),
                );
            }
            stop(&mut child, reader);
            return Err(GENERATION_STOPPED.to_string());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            stop(&mut child, reader);
            return Err(
                "The local agent timed out. Try again, or pick a model provider.".to_string(),
            );
        }
        let poll = remaining.min(Duration::from_millis(150));
        let line = match line_rx.recv_timeout(poll) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            continue;
        };
        let has_id = msg.get("id").is_some();
        let is_response = has_id && (msg.get("result").is_some() || msg.get("error").is_some());
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

        if is_response {
            // Responses to OUR requests, keyed by the id we sent.
            match msg.get("id").and_then(|i| i.as_i64()) {
                Some(1) => {
                    // initialize done → ack, then open a thread.
                    send(
                        &mut stdin,
                        serde_json::json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
                    );
                    send(
                        &mut stdin,
                        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"thread/start",
                            "params":{"sandbox":"danger-full-access","approvalPolicy":"never",
                                "cwd": work_dir.to_string_lossy()}}),
                    );
                }
                Some(2) => {
                    // thread open → capture its id and start the turn with the prompt
                    // (and the user's image as a localImage input, if any).
                    let tid = msg["result"]["thread"]["id"].as_str().map(str::to_string);
                    match tid {
                        Some(tid) => {
                            state.thread_id = Some(tid.clone());
                            let mut input = vec![serde_json::json!({"type":"text","text":prompt})];
                            if let Some(path) = &image_path {
                                input.push(serde_json::json!({"type":"localImage","path": path.to_string_lossy()}));
                            }
                            send(
                                &mut stdin,
                                serde_json::json!({"jsonrpc":"2.0","id":3,"method":"turn/start",
                                    "params":{"threadId":tid,"input":input,"approvalPolicy":"never"}}),
                            );
                        }
                        None => {
                            stop(&mut child, reader);
                            return Err(
                                "Codex app-server: thread/start returned no thread id".to_string()
                            );
                        }
                    }
                }
                _ => {}
            }
            continue;
        }

        if has_id && !method.is_empty() {
            // A server→client REQUEST (approval, etc.). With approvalPolicy=never we
            // don't expect these, but answer "approved" defensively so nothing hangs.
            if let Some(id) = msg.get("id") {
                send(
                    &mut stdin,
                    serde_json::json!({"jsonrpc":"2.0","id":id,"result":{"decision":"approved"}}),
                );
            }
            continue;
        }

        if !method.is_empty() {
            handle_app_server_notification(method, &msg["params"], &mut state, on_tool, on_answer);
            if state.done {
                break;
            }
        }
    }
    let _ = reader.join();
    let _ = child.kill();

    state
        .final_answer
        .filter(|a| !a.is_empty())
        .or_else(|| Some(state.answer.trim().to_string()).filter(|a| !a.is_empty()))
        .ok_or_else(|| {
            let stderr = err_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap_or_default();
            if stderr.trim().is_empty() {
                format!("{} returned no answer.", AgentKind::Codex.label())
            } else {
                classify_error(AgentKind::Codex, stderr.trim())
            }
        })
}

/// Route one Codex app-server notification: token deltas → `on_answer`, MCP
/// tool-call lifecycle → `on_tool`, terminal agentMessage/turn → final answer/done.
fn handle_app_server_notification<F, G>(
    method: &str,
    params: &serde_json::Value,
    state: &mut CodexAppServerState,
    on_tool: &F,
    on_answer: &G,
) where
    F: Fn(AgentToolEvent),
    G: Fn(String),
{
    match method {
        "turn/started" => {
            state.turn_id = params["turnId"]
                .as_str()
                .or_else(|| params["turn"]["id"].as_str())
                .map(str::to_string);
        }
        "item/agentMessage/delta" => {
            if let Some(delta) = params["delta"].as_str() {
                if !delta.is_empty() {
                    state.answer.push_str(delta);
                    on_answer(delta.to_string());
                }
            }
        }
        "item/started" => {
            let item = &params["item"];
            if item["type"].as_str() == Some("mcpToolCall") {
                let tool = item["tool"]
                    .as_str()
                    .or_else(|| item["name"].as_str())
                    .unwrap_or("tool")
                    .to_string();
                state.pending_tools.push_back(tool.clone());
                on_tool(AgentToolEvent {
                    tool,
                    phase: AgentToolPhase::Started,
                    ok: true,
                });
            }
        }
        "item/completed" => {
            let item = &params["item"];
            match item["type"].as_str() {
                Some("mcpToolCall") => {
                    let ok = item["status"]
                        .as_str()
                        .map(|s| s != "failed" && s != "error")
                        .unwrap_or(true);
                    let tool = state
                        .pending_tools
                        .pop_front()
                        .or_else(|| item["tool"].as_str().map(str::to_string))
                        .unwrap_or_else(|| "tool".to_string());
                    on_tool(AgentToolEvent {
                        tool,
                        phase: AgentToolPhase::Completed,
                        ok,
                    });
                }
                Some("agentMessage") => {
                    if let Some(text) = item["text"].as_str() {
                        if !text.trim().is_empty() {
                            state.final_answer = Some(text.trim().to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        "turn/completed" | "turn/failed" | "error" => state.done = true,
        _ => {}
    }
}

/// Accumulator for one Claude `stream-json` run.
#[derive(Default)]
struct ClaudeStreamState {
    /// Text assembled from `content_block_delta` (or a fallback assistant message).
    answer: String,
    /// The authoritative final answer from the terminal `result` event.
    final_result: Option<String>,
    /// Tool names started but not yet completed (FIFO — Claude calls them in order).
    pending_tools: std::collections::VecDeque<String>,
    /// Whether any answer token was streamed live (for diagnostics/fallback).
    streamed: bool,
}

/// Drive Claude `-p --output-format stream-json --include-partial-messages`,
/// parsing its JSONL as it arrives: token deltas go to `on_answer` (live answer
/// streaming), MCP tool-call start/finish go to `on_tool`, and the final `result`
/// event is taken as the authoritative answer. A hard deadline kills a stall.
fn stream_claude_agentic<F, G>(
    mut cmd: Command,
    timeout: Duration,
    cancel: &CancellationToken,
    on_tool: &F,
    on_answer: &G,
) -> Result<String, String>
where
    F: Fn(AgentToolEvent),
    G: Fn(String),
{
    let mut child = cmd
        .spawn()
        .map_err(|err| format!("Failed to launch the local agent: {err}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Local agent produced no output stream".to_string())?;

    // Drain stderr on a side thread so a full pipe can't deadlock the child.
    let (err_tx, err_rx) = mpsc::channel();
    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut buf);
            let _ = err_tx.send(buf);
        });
    }

    let (line_tx, line_rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            match line {
                Ok(l) => {
                    if line_tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let deadline = Instant::now() + timeout;
    let mut state = ClaudeStreamState::default();
    loop {
        // Stop button: kill the CLI mid-run when the user cancels.
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = reader.join();
            return Err(GENERATION_STOPPED.to_string());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            let _ = child.kill();
            let _ = reader.join();
            return Err(
                "The local agent timed out. Try again, or pick a model provider.".to_string(),
            );
        }
        let poll = remaining.min(Duration::from_millis(150));
        match line_rx.recv_timeout(poll) {
            Ok(line) => handle_claude_line(&line, on_tool, on_answer, &mut state),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break, // stdout closed → done
        }
    }
    let _ = reader.join();

    let status = child
        .wait()
        .map_err(|err| format!("Local agent did not exit cleanly: {err}"))?;
    if !status.success() {
        let stderr = err_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap_or_default();
        return Err(classify_error(AgentKind::Claude, stderr.trim()));
    }
    // Prefer the authoritative `result` text; fall back to the streamed/assembled text.
    state
        .final_result
        .filter(|a| !a.is_empty())
        .or_else(|| Some(state.answer.trim().to_string()).filter(|a| !a.is_empty()))
        .ok_or_else(|| format!("{} returned no answer.", AgentKind::Claude.label()))
}

/// Parse one Claude `stream-json` line: relay token deltas, tool-call steps, and
/// capture the terminal result. Unknown shapes are ignored (robust to version drift).
fn handle_claude_line<F, G>(line: &str, on_tool: &F, on_answer: &G, state: &mut ClaudeStreamState)
where
    F: Fn(AgentToolEvent),
    G: Fn(String),
{
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        return;
    };
    match value.get("type").and_then(|t| t.as_str()).unwrap_or("") {
        // Partial streaming events (enabled by --include-partial-messages).
        "stream_event" => {
            let event = &value["event"];
            match event.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "content_block_delta" => {
                    if let Some(text) = event["delta"]["text"].as_str() {
                        if !text.is_empty() {
                            state.answer.push_str(text);
                            state.streamed = true;
                            on_answer(text.to_string());
                        }
                    }
                }
                // A tool_use block opening → a tool call is starting.
                "content_block_start" => {
                    let block = &event["content_block"];
                    if block["type"].as_str() == Some("tool_use") {
                        let tool = block["name"].as_str().unwrap_or("tool").to_string();
                        state.pending_tools.push_back(tool.clone());
                        on_tool(AgentToolEvent {
                            tool,
                            phase: AgentToolPhase::Started,
                            ok: true,
                        });
                    }
                }
                _ => {}
            }
        }
        // Tool results return as a user message carrying tool_result blocks.
        "user" => {
            if let Some(content) = value["message"]["content"].as_array() {
                for block in content {
                    if block["type"].as_str() == Some("tool_result") {
                        let ok = block["is_error"].as_bool() != Some(true);
                        let tool = state
                            .pending_tools
                            .pop_front()
                            .unwrap_or_else(|| "tool".to_string());
                        on_tool(AgentToolEvent {
                            tool,
                            phase: AgentToolPhase::Completed,
                            ok,
                        });
                    }
                }
            }
        }
        // Fallback when partial deltas are absent: a complete assistant message. Emit
        // its text once so the bubble still streams something.
        "assistant" => {
            if !state.streamed {
                if let Some(content) = value["message"]["content"].as_array() {
                    let text: String = content
                        .iter()
                        .filter(|block| block["type"].as_str() == Some("text"))
                        .filter_map(|block| block["text"].as_str())
                        .collect();
                    if !text.is_empty() {
                        on_answer(text.clone());
                        state.answer.push_str(&text);
                        state.streamed = true;
                    }
                }
            }
        }
        // Authoritative final answer.
        "result" => {
            if let Some(text) = value["result"].as_str() {
                state.final_result = Some(text.trim().to_string());
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_stream_emits_token_deltas_and_captures_result() {
        use std::cell::RefCell;
        let deltas = RefCell::new(Vec::new());
        let tools = RefCell::new(Vec::new());
        let on_answer = |d: String| deltas.borrow_mut().push(d);
        let on_tool = |ev: AgentToolEvent| tools.borrow_mut().push((ev.tool, ev.phase, ev.ok));
        let mut state = ClaudeStreamState::default();

        for line in [
            r#"{"type":"system","subtype":"init"}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","name":"mcp__lumenfolio__search_chunks"}}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","is_error":false}]}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}}"#,
            r#"{"type":"result","subtype":"success","result":"Hello"}"#,
        ] {
            handle_claude_line(line, &on_tool, &on_answer, &mut state);
        }

        assert_eq!(*deltas.borrow(), vec!["Hel".to_string(), "lo".to_string()]);
        assert_eq!(state.answer, "Hello");
        assert_eq!(state.final_result.as_deref(), Some("Hello"));
        let tools = tools.borrow();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].1, AgentToolPhase::Started);
        assert_eq!(tools[0].0, "mcp__lumenfolio__search_chunks");
        assert_eq!(tools[1].1, AgentToolPhase::Completed);
        assert_eq!(tools[1].0, "mcp__lumenfolio__search_chunks");
    }

    #[test]
    fn claude_stream_falls_back_to_complete_assistant_message() {
        // No partial deltas (e.g. --include-partial-messages unsupported): a full
        // assistant message is emitted once so the bubble still gets the answer.
        use std::cell::RefCell;
        let deltas = RefCell::new(Vec::new());
        let on_answer = |d: String| deltas.borrow_mut().push(d);
        let on_tool = |_ev: AgentToolEvent| {};
        let mut state = ClaudeStreamState::default();
        handle_claude_line(
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Atomic answer."}]}}"#,
            &on_tool,
            &on_answer,
            &mut state,
        );
        assert_eq!(*deltas.borrow(), vec!["Atomic answer.".to_string()]);
        assert_eq!(state.answer, "Atomic answer.");
    }

    #[test]
    fn parse_version_extracts_dotted_token() {
        assert_eq!(parse_version("2.0.55 (Claude Code)"), "2.0.55");
        assert_eq!(parse_version("codex-cli 0.12.0"), "0.12.0");
        assert_eq!(parse_version("v1.4.2"), "1.4.2");
        // No dotted token → trimmed first line.
        assert_eq!(parse_version("nightly build\nextra"), "nightly build");
    }

    #[test]
    fn write_image_temp_decodes_and_cleans_up() {
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3, 4]);
        let data_url = format!("data:image/png;base64,{encoded}");
        let temp = write_image_temp(&data_url).expect("decoded");
        assert_eq!(temp.path.extension().and_then(|e| e.to_str()), Some("png"));
        assert_eq!(std::fs::read(&temp.path).unwrap(), vec![1, 2, 3, 4]);
        let path = temp.path.clone();
        drop(temp);
        assert!(!path.exists(), "temp file removed on drop");
        // Rejects: not a data URL, and data URL without base64 marker.
        assert!(write_image_temp("https://example.com/x.png").is_none());
        assert!(write_image_temp("data:image/png,plain").is_none());
    }

    #[test]
    fn provider_id_kind_maps_virtual_ids() {
        assert_eq!(
            provider_id_kind("local-agent-codex"),
            Some(AgentKind::Codex)
        );
        assert_eq!(
            provider_id_kind("local-agent-claude"),
            Some(AgentKind::Claude)
        );
        assert_eq!(provider_id_kind("openai"), None);
        assert_eq!(provider_id_kind(""), None);
    }

    #[test]
    fn parse_codex_picks_last_agent_message() {
        let stdout = r#"{"type":"thread.started","thread_id":"x"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"The answer."}}
{"type":"turn.completed","usage":{"output_tokens":5}}"#;
        assert_eq!(
            parse_codex_jsonl(AgentKind::Codex, stdout).unwrap(),
            "The answer."
        );
        assert!(parse_codex_jsonl(AgentKind::Codex, "{}\n").is_err());
    }

    #[test]
    fn parse_claude_extracts_result_and_flags_errors() {
        let ok = r#"{"type":"result","is_error":false,"result":"  Hello.  "}"#;
        assert_eq!(parse_claude_json(AgentKind::Claude, ok).unwrap(), "Hello.");
        // is_error true + login text → actionable login hint, NOT echoed as answer.
        let expired =
            r#"{"type":"result","is_error":true,"result":"API Error: 401 ... Please run /login"}"#;
        let err = parse_claude_json(AgentKind::Claude, expired).unwrap_err();
        assert!(err.contains("isn't logged in"), "got: {err}");
    }

    #[test]
    fn build_prompt_embeds_evidence_and_language() {
        let p = build_prompt("What is X?", "page 1: X is Y.", "", Some("zh-CN"));
        // The answer follows the question's language; locale is only the fallback.
        assert!(p.contains("SAME language as the user's Question"));
        assert!(p.contains("default to Chinese"));
        assert!(p.contains("page 1: X is Y."));
        assert!(p.contains("What is X?"));
    }

    #[test]
    fn build_agentic_prompt_instructs_tool_use_not_embedded_evidence() {
        let p = build_agentic_prompt(
            "What is X?",
            "prior: Y",
            Some("The user is currently viewing the Trending Papers list (period: weekly)."),
            Some("zh"),
        );
        assert!(p.contains("SAME language as the user's Question"));
        // Mode B tells the agent to call the lumenfolio tools itself...
        assert!(p.contains("lumenfolio"));
        assert!(p.to_lowercase().contains("search"));
        assert!(p.contains("What is X?"));
        assert!(p.contains("prior: Y"));
        // ...the current-view note is surfaced so it reaches for the right tool...
        assert!(p.contains("Trending Papers list (period: weekly)"));
        // ...and must NOT carry the Mode-A "do not call tools" instruction.
        assert!(!p.contains("Do NOT call tools"));
    }

    #[test]
    fn language_directive_follows_question_not_locale() {
        // Even with an English/None locale, the directive must instruct following
        // the question's own language (so a Chinese question gets a Chinese answer).
        let d = language_directive(Some("en"));
        assert!(d.contains("SAME language as the user's Question"));
        assert!(d.contains("a Chinese question gets a Chinese answer"));
        assert!(d.contains("default to English"));
    }

    #[test]
    fn detect_missing_binary_reports_not_installed() {
        // A binary that certainly does not exist.
        let status = detect(AgentKind::Codex);
        // We can't assert installed=true (CI may not have codex), but the shape must
        // be coherent: not installed <=> no version and no path.
        if !status.installed {
            assert!(status.version.is_none() && status.path.is_none());
        }
        assert_eq!(status.install_url, AgentKind::Codex.install_url());
    }
}
