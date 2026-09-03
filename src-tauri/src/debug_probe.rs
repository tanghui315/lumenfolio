use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use std::{
    env,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use crate::{
    agent_judge, llm, normalize_base_url, providers, runtime, storage, truncate_for_error,
    AppDatabase, AskDocumentInput, OpenAiChatRequest, OpenAiChatResponse, OpenAiCompatibleProvider,
};

#[derive(Debug)]
struct ProbeConfig {
    db_path: PathBuf,
    document: Option<String>,
    prompt: Option<String>,
    provider_id: String,
    model_key: Option<String>,
    locale: Option<String>,
    current_page: Option<u32>,
    max_steps: u32,
    json: bool,
    no_answer: bool,
    skip_llm_judge: bool,
    list_documents: bool,
    list_providers: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProbeDocument {
    id: String,
    title: String,
    path: String,
    index_status: String,
    index_version: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProbeOutput {
    document: ProbeDocument,
    provider: String,
    question: String,
    current_view: Option<serde_json::Value>,
    answerable: bool,
    final_gate: serde_json::Value,
    answer: Option<String>,
    citations: Vec<runtime::rag::Citation>,
    tool_calls: Vec<runtime::rag::RetrievalTraceToolCall>,
    events: Vec<runtime::agent::AgentTraceEvent>,
    prompt_context_chars: usize,
}

pub async fn run_from_env() -> Result<(), String> {
    let config = ProbeConfig::from_args(env::args().skip(1).collect())?;
    let conn = open_probe_database(&config.db_path)?;
    if config.list_documents {
        print_documents(&conn, config.json)?;
        return Ok(());
    }
    if config.list_providers {
        print_providers(&conn, config.json)?;
        return Ok(());
    }

    let document_key = config.document.as_deref().ok_or_else(|| {
        "Missing --document. Use --list-documents to inspect indexed PDFs.".to_string()
    })?;
    let question = config
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Missing --prompt.".to_string())?;
    let document = resolve_document(&conn, document_key)?;
    let database = AppDatabase {
        conn: Mutex::new(conn),
    };
    let sessions = runtime::agent::AgentSessionStore::default();
    let (provider, provider_label) = providers::resolve_chat_provider(
        &database,
        &config.provider_id,
        config.model_key.as_deref(),
    )?;
    let input = AskDocumentInput {
        document_id: document.id.clone(),
        session_id: None,
        question: question.to_string(),
        locale: config.locale.clone(),
        model_provider_id: if config.provider_id.is_empty() {
            None
        } else {
            Some(config.provider_id.clone())
        },
        model_key: config.model_key.clone(),
        selected_text: None,
        selected_block_id: None,
        selected_bbox_list: None,
        image_data_url: None,
        page: config.current_page,
        viewport_context: config.current_page.map(|page| crate::ViewportContextInput {
            active_page: Some(page),
            visible_pages: vec![page],
            selection_preview: None,
            captured_at: None,
            sensitivity: Some("normal".to_string()),
            source: Some("agentic_probe".to_string()),
        }),
        max_retrieval_steps: Some(config.max_steps),
        retrieval_attempt_offset: None,
        activity_event_id: None,
        reference_document_ids: None,
        knowledge_enabled: None,
        web_enabled: Some(env::var("LUMEN_VERIFY_WEB").is_ok()),
        view_context: None,
    };
    let current_view_metadata = crate::build_current_view_gate_metadata(
        &database,
        &document.id,
        input.viewport_context.as_ref(),
        input.page,
        input.selected_text.as_deref(),
    )?;
    let current_view_decision = crate::current_view_decision_for_input(
        question,
        &input,
        current_view_metadata.as_ref(),
        Some(&provider),
    )
    .await;
    let (retrieval_page, retrieval_page_mode, retrieval_page_source) =
        crate::current_view_retrieval_hint(&input, current_view_decision.as_ref());

    let mut agent_run = {
        let conn = database
            .conn
            .lock()
            .map_err(|_| "SQLite lock was poisoned".to_string())?;
        runtime::agent::run_turn_with_activity(
            &conn,
            &sessions,
            runtime::agent::AgentRunRequest {
                document_id: &document.id,
                session_key: &document.id,
                visible_document_ids: Vec::new(),
                question,
                provider_id: input.model_provider_id.as_deref(),
                context_budget: provider.context_budget.clone(),
                selected_text: None,
                selected_block_id: None,
                selected_bbox_list: None,
                page: retrieval_page,
                page_mode: retrieval_page_mode,
                page_source: retrieval_page_source,
                max_retrieval_steps: Some(config.max_steps),
                retrieval_attempt_offset: 0,
            },
            |_event| {},
        )?
    };

    if !config.skip_llm_judge {
        agent_judge::improve_retrieval_with_llm_judge(
            agent_judge::LlmJudgeLoopInput {
                input: &input,
                database: &database,
                app: None,
                question,
                document_id: &document.id,
                visible_document_ids: &[],
                workspace_manifest: "",
                provider: &provider,
                activity_event_id: None,
            },
            &mut agent_run,
        )
        .await?;
    }

    let answerable = agent_judge::retrieval_is_answerable(&agent_run);
    let answer = if answerable && !config.no_answer {
        Some(generate_probe_answer(question, &input, &agent_run, &provider).await?)
    } else {
        None
    };
    let output = ProbeOutput {
        document,
        provider: provider_label,
        question: question.to_string(),
        current_view: current_view_decision.as_ref().map(|decision| {
            serde_json::json!({
                "relevance": decision.relevance,
                "mode": decision.mode,
                "shouldUseCurrentView": decision.should_use_current_view,
                "reason": decision.reason,
                "metadata": current_view_metadata,
            })
        }),
        answerable,
        final_gate: agent_run.retrieval_run.trace.finalize_gate.clone(),
        answer,
        citations: agent_run.retrieval_run.citations.clone(),
        tool_calls: agent_run.retrieval_run.trace.tool_calls.clone(),
        events: agent_run.trace.events.clone(),
        prompt_context_chars: agent_run.retrieval_run.prompt_context.chars().count(),
    };
    print_probe_output(&output, config.json)
}

impl ProbeConfig {
    fn from_args(args: Vec<String>) -> Result<Self, String> {
        let mut db_path = env::var("LUMENFOLIO_DB_PATH").ok().map(PathBuf::from);
        let mut document = None;
        let mut prompt = None;
        let mut provider_id = String::new();
        let mut model_key = None;
        let mut locale = None;
        let mut current_page = None;
        let mut max_steps = 20_u32;
        let mut json = false;
        let mut no_answer = false;
        let mut skip_llm_judge = false;
        let mut list_documents = false;
        let mut list_providers = false;
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--help" | "-h" => return Err(usage()),
                "--db" => {
                    index += 1;
                    db_path = Some(PathBuf::from(required_arg(&args, index, "--db")?));
                }
                "--document" | "--document-id" | "--pdf" => {
                    index += 1;
                    document = Some(required_arg(&args, index, "--document")?);
                }
                "--prompt" | "--question" => {
                    index += 1;
                    prompt = Some(required_arg(&args, index, "--prompt")?);
                }
                "--provider" | "--provider-id" => {
                    index += 1;
                    provider_id = required_arg(&args, index, "--provider")?;
                }
                "--model-key" => {
                    index += 1;
                    model_key = Some(required_arg(&args, index, "--model-key")?);
                }
                "--locale" => {
                    index += 1;
                    locale = Some(required_arg(&args, index, "--locale")?);
                }
                "--page" | "--current-page" => {
                    index += 1;
                    current_page = Some(
                        required_arg(&args, index, "--page")?
                            .parse::<u32>()
                            .map_err(|err| format!("Invalid --page: {err}"))?,
                    );
                }
                "--max-steps" => {
                    index += 1;
                    max_steps = required_arg(&args, index, "--max-steps")?
                        .parse::<u32>()
                        .map_err(|err| format!("Invalid --max-steps: {err}"))?
                        .clamp(1, 20);
                }
                "--json" => json = true,
                "--no-answer" => no_answer = true,
                "--skip-llm-judge" => skip_llm_judge = true,
                "--list-documents" => list_documents = true,
                "--list-providers" => list_providers = true,
                other => return Err(format!("Unknown argument: {other}\n\n{}", usage())),
            }
            index += 1;
        }
        let db_path = db_path.or_else(default_db_path).ok_or_else(|| {
            "Missing --db and could not infer the default app database path.".to_string()
        })?;
        Ok(Self {
            db_path,
            document,
            prompt,
            provider_id,
            model_key,
            locale,
            current_page,
            max_steps,
            json,
            no_answer,
            skip_llm_judge,
            list_documents,
            list_providers,
        })
    }
}

fn required_arg(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn usage() -> String {
    "Usage:\n  cargo run --manifest-path src-tauri/Cargo.toml --bin agentic_probe -- \\\n    --db /path/to/lumenfolio.sqlite \\\n    --document 2601.16746v2.pdf \\\n    --prompt \"Table 3 里面提到的 SWE-Pruner 是什么指标？结果是怎么样的？\" \\\n    --json\n\nOptions:\n  --list-documents        List indexed documents in the selected DB.\n  --list-providers        List configured model providers.\n  --provider <id>         Use a specific provider id; default provider is used when omitted.\n  --model-key <key>       Use a specific configured model key.\n  --page <n>              Simulate the active PDF page for current-view routing.\n  --max-steps <1..20>     Retrieval budget, default 20.\n  --no-answer             Stop after retrieval/judge, without final answer generation.\n  --skip-llm-judge        Skip the M4 LLM evidence judge and report the deterministic M3 retrieval gate.\n  --json                  Emit machine-readable JSON."
        .to_string()
}

fn default_db_path() -> Option<PathBuf> {
    let home = env::var("HOME").ok()?;
    let candidates = [
        "Library/Application Support/com.sotarium.lumenfolio/lumenfolio.sqlite",
        "Library/Application Support/Lumenfolio/lumenfolio.sqlite",
        "Library/Application Support/Lumenfolio Desktop/lumenfolio.sqlite",
    ];
    candidates
        .iter()
        .map(|path| PathBuf::from(&home).join(path))
        .find(|path| path.is_file())
}

fn open_probe_database(path: &Path) -> Result<Connection, String> {
    let conn = storage::open_database(path)?;
    Ok(conn)
}

fn resolve_document(conn: &Connection, key: &str) -> Result<ProbeDocument, String> {
    if let Some(document) = query_document(conn, "id = ?1", key)? {
        return Ok(document);
    }
    if let Some(document) = query_document(conn, "path = ?1", key)? {
        return Ok(document);
    }
    let like = format!("%{key}%");
    let mut stmt = conn
        .prepare(
            "SELECT id, title, path, index_status, index_version
             FROM documents
             WHERE title LIKE ?1 OR short_title LIKE ?1 OR path LIKE ?1
             ORDER BY last_opened_at DESC, updated_at DESC
             LIMIT 10",
        )
        .map_err(|err| format!("Failed to prepare document lookup: {err}"))?;
    let matches = stmt
        .query_map(params![like], read_probe_document)
        .map_err(|err| format!("Failed to lookup document: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read document lookup results: {err}"))?;
    match matches.len() {
        1 => Ok(matches.into_iter().next().expect("one document")),
        0 => Err(format!(
            "No document matched '{key}'. Use --list-documents."
        )),
        _ => Err(format!(
            "Document key '{key}' matched multiple PDFs:\n{}",
            matches
                .iter()
                .map(|doc| format!("- {} | {} | {}", doc.id, doc.title, doc.path))
                .collect::<Vec<_>>()
                .join("\n")
        )),
    }
}

fn query_document(
    conn: &Connection,
    predicate: &str,
    key: &str,
) -> Result<Option<ProbeDocument>, String> {
    conn.query_row(
        &format!(
            "SELECT id, title, path, index_status, index_version
             FROM documents
             WHERE {predicate}
             LIMIT 1"
        ),
        params![key],
        read_probe_document,
    )
    .optional()
    .map_err(|err| format!("Failed to query document: {err}"))
}

fn read_probe_document(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProbeDocument> {
    Ok(ProbeDocument {
        id: row.get(0)?,
        title: row.get(1)?,
        path: row.get(2)?,
        index_status: row.get(3)?,
        index_version: row.get(4)?,
    })
}

fn print_documents(conn: &Connection, json: bool) -> Result<(), String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, title, path, index_status, index_version
             FROM documents
             ORDER BY last_opened_at DESC, updated_at DESC
             LIMIT 100",
        )
        .map_err(|err| format!("Failed to prepare document list: {err}"))?;
    let documents = stmt
        .query_map([], read_probe_document)
        .map_err(|err| format!("Failed to list documents: {err}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Failed to read documents: {err}"))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&documents)
                .map_err(|err| format!("Failed to encode documents: {err}"))?
        );
    } else {
        for doc in documents {
            println!(
                "{} | {} | {} | {} | v{}",
                doc.id, doc.title, doc.path, doc.index_status, doc.index_version
            );
        }
    }
    Ok(())
}

fn print_providers(conn: &Connection, json: bool) -> Result<(), String> {
    let providers = providers::load_model_provider_outputs(conn)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&providers)
                .map_err(|err| format!("Failed to encode providers: {err}"))?
        );
    } else {
        for provider in providers {
            println!(
                "{} | {} | default={} | enabled={}",
                provider.id, provider.name, provider.is_default, provider.enabled
            );
            for model in provider.models {
                println!(
                    "  - {} | {} | default={} | capabilities={}",
                    model.key,
                    model.model_id,
                    model.is_default_chat_model,
                    model.capabilities.join(",")
                );
            }
        }
    }
    Ok(())
}

async fn generate_probe_answer(
    question: &str,
    input: &AskDocumentInput,
    agent_run: &runtime::agent::AgentRunResult,
    provider: &OpenAiCompatibleProvider,
) -> Result<String, String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|err| format!("Failed to create chat client: {err}"))?;
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
    if !agent_run.retrieval_run.prompt_context.trim().is_empty() {
        user_content.push_str("Evidence sources:\n");
        user_content.push_str(&agent_run.retrieval_run.prompt_context);
        user_content.push_str("\n\n");
    }
    user_content.push_str("Question:\n");
    user_content.push_str(question);
    user_content.push_str(
        "\n\nWrite a structured Markdown answer only. Do not return JSON. Use a short direct answer first, then organize the supporting explanation with concise paragraphs, bullet lists, or numbered lists when helpful. For summary, conclusion, comparison, method, or experiment questions, prefer clear section labels and lists over one long paragraph. Use only the provided evidence sources. If evidence is insufficient, say so clearly. Do not invent facts beyond the evidence. Stay tightly scoped to the user's question. For table or metric questions, answer the requested table, row/entity, columns, and values first; do not add metrics from other tables, pages, or sections unless they are explicitly requested or strictly necessary to define the requested entity.",
    );
    let answer_language =
        llm::chat::answer_language_for_question(question, input.locale.as_deref());
    let request = OpenAiChatRequest {
        model: provider.model.clone(),
        temperature: 0.2,
        stream: None,
        messages: vec![
            llm::chat::text_message(
                "system",
                format!(
                    "You are Lumenfolio, a careful academic PDF reading assistant. Answer in {answer_language}. Use only the provided evidence sources. Do not invent facts beyond the evidence. Return Markdown only, not JSON. Prefer readable structure: a short direct answer, then concise paragraphs, bullet lists, or numbered lists when they improve scanability. Avoid one long undifferentiated paragraph. If the evidence is insufficient, say so clearly. Keep the answer scoped to the user's question; for table or metric questions, do not volunteer extra metrics from other tables or sections."
                ),
            ),
            llm::chat::user_message_with_optional_image(&user_content, None),
        ],
    };
    let endpoint = format!(
        "{}/chat/completions",
        normalize_base_url(&provider.base_url)
    );
    let mut builder = client.post(endpoint).json(&request);
    if let Some(api_key) = &provider.api_key {
        builder = builder.bearer_auth(api_key);
    }
    let response = builder
        .send()
        .await
        .map_err(|err| format!("Chat provider request failed: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Chat provider returned {status}: {}",
            truncate_for_error(&body, 600)
        ));
    }
    let response = response
        .json::<OpenAiChatResponse>()
        .await
        .map_err(|err| format!("Failed to decode answer response: {err}"))?;
    response
        .choices
        .into_iter()
        .next()
        .map(|choice| llm::chat::extract_chat_response_text(&choice.message.content))
        .filter(|answer| !answer.trim().is_empty())
        .ok_or_else(|| "Chat provider returned an empty answer".to_string())
}

fn print_probe_output(output: &ProbeOutput, json: bool) -> Result<(), String> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(output)
                .map_err(|err| format!("Failed to encode probe output: {err}"))?
        );
        return Ok(());
    }
    println!(
        "Document: {} | {}",
        output.document.id, output.document.title
    );
    println!("Provider: {}", output.provider);
    println!("Question: {}", output.question);
    println!("Answerable: {}", output.answerable);
    println!(
        "Final gate: {}",
        serde_json::to_string_pretty(&output.final_gate)
            .map_err(|err| format!("Failed to encode final gate: {err}"))?
    );
    println!("Tool calls:");
    for call in &output.tool_calls {
        println!(
            "- {} status={} results={} error={}",
            call.tool,
            call.status,
            call.result_count,
            call.error.as_deref().unwrap_or("")
        );
    }
    println!("Citations:");
    for citation in &output.citations {
        println!(
            "- {} p{} source={} section={} quote={}",
            citation.label,
            citation.page,
            citation.source,
            citation.section_title.as_deref().unwrap_or(""),
            truncate_for_error(&citation.quote, 180)
        );
    }
    if let Some(answer) = &output.answer {
        println!("\nAnswer:\n{answer}");
    }
    Ok(())
}

/// Manual harness for P2-3 (Mode B MCP server live verification): start the
/// in-process loopback MCP server against a real on-disk DB + document, print
/// its URL + bearer token, and keep it alive long enough for an external CLI
/// (`codex` / `claude`) to connect and exercise the tools. Reads:
///   LUMEN_VERIFY_DB   — absolute path to lumenfolio.sqlite
///   LUMEN_VERIFY_DOC  — document id to scope the tools to
///   LUMEN_VERIFY_SECS — keepalive seconds (default 600)
/// Not wired into the app; invoked via the `mcp_verify` bin.
pub async fn run_mcp_verify_from_env() -> Result<(), String> {
    use std::io::Write;

    let db = env::var("LUMEN_VERIFY_DB").map_err(|_| "set LUMEN_VERIFY_DB".to_string())?;
    let doc = env::var("LUMEN_VERIFY_DOC").map_err(|_| "set LUMEN_VERIFY_DOC".to_string())?;
    let secs: u64 = env::var("LUMEN_VERIFY_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let server = crate::local_agent::mcp_server::start_mcp_server(
        PathBuf::from(db),
        doc,
        env::var("LUMEN_VERIFY_WEB").is_ok(),
    )
    .await?;
    println!("MCP_URL={}", server.url);
    println!("MCP_TOKEN={}", server.token);
    println!("MCP_READY alive={secs}s");
    std::io::stdout().flush().ok();

    tokio::time::sleep(Duration::from_secs(secs)).await;

    let served = server.citations.lock().map(|c| c.len()).unwrap_or(0);
    println!("CITATIONS_SERVED={served}");
    Ok(())
}

/// Full-stack probe for P2-4 (Mode B agentic dispatch): drive `generate_answer_agentic`
/// end-to-end — bring up the MCP server, run the real CLI against it, and report the
/// answer + the citations the server captured. Reads:
///   LUMEN_VERIFY_DB / LUMEN_VERIFY_DOC — db path + document id
///   LUMEN_VERIFY_KIND — "codex" (default) | "claude"
///   LUMEN_VERIFY_Q    — the question to ask
pub async fn run_agentic_probe_from_env() -> Result<(), String> {
    let db = env::var("LUMEN_VERIFY_DB").map_err(|_| "set LUMEN_VERIFY_DB".to_string())?;
    let doc = env::var("LUMEN_VERIFY_DOC").map_err(|_| "set LUMEN_VERIFY_DOC".to_string())?;
    let question = env::var("LUMEN_VERIFY_Q")
        .unwrap_or_else(|_| "What is the main contribution of this paper?".to_string());
    let kind = match env::var("LUMEN_VERIFY_KIND").unwrap_or_default().as_str() {
        "claude" => crate::local_agent::AgentKind::Claude,
        _ => crate::local_agent::AgentKind::Codex,
    };

    // Optional image: LUMEN_VERIFY_IMG=<path> → base64 data URL, exercising the `-i` path.
    let image_data_url = env::var("LUMEN_VERIFY_IMG").ok().and_then(|path| {
        use base64::Engine;
        let bytes = std::fs::read(&path).ok()?;
        let mime = if path.ends_with(".jpg") || path.ends_with(".jpeg") {
            "image/jpeg"
        } else {
            "image/png"
        };
        Some(format!(
            "data:{mime};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&bytes)
        ))
    });

    let view_note = env::var("LUMEN_VERIFY_VIEW").ok();
    let prompt =
        crate::local_agent::build_agentic_prompt(&question, "", view_note.as_deref(), Some("en"));
    let outcome = crate::local_agent::generate_answer_agentic(
        kind,
        PathBuf::from(db),
        doc,
        prompt,
        image_data_url,
        env::var("LUMEN_VERIFY_WEB").is_ok(),
        tokio_util::sync::CancellationToken::new(),
        |ev| {
            let phase = match ev.phase {
                crate::local_agent::AgentToolPhase::Started => "→ calling",
                crate::local_agent::AgentToolPhase::Completed => {
                    if ev.ok {
                        "✓ done"
                    } else {
                        "✗ failed"
                    }
                }
            };
            println!("TRACE {phase} {}", ev.tool);
        },
        |delta: String| {
            use std::io::Write;
            print!("{delta}");
            let _ = std::io::stdout().flush();
        },
    )
    .await?;

    let max_quote = outcome
        .citations
        .iter()
        .map(|c| c.quote.chars().count())
        .max()
        .unwrap_or(0);
    println!(
        "CITATIONS_CAPTURED={} CANDIDATES_CAPTURED={} MAX_QUOTE_CHARS={}",
        outcome.citations.len(),
        outcome.candidates.len(),
        max_quote
    );
    // Mirror build_evidence_chain's block-match to confirm section titles now resolve.
    for c in outcome.citations.iter().take(8) {
        let bbox_len = c.bbox_list.as_array().map(|a| a.len()).unwrap_or(0);
        let section = c.section_title.clone().or_else(|| {
            outcome
                .candidates
                .iter()
                .find(|cand| !c.block_id.is_empty() && cand.block_id == c.block_id)
                .and_then(|cand| cand.section_title.clone())
        });
        println!(
            "- [{}] p{} src={} bbox_items={} section={:?} : {}",
            c.label,
            c.page,
            c.source,
            bbox_len,
            section.as_deref().unwrap_or("<none>"),
            truncate_for_error(&c.quote, 60)
        );
    }
    println!("\nANSWER:\n{}", outcome.answer);
    Ok(())
}
