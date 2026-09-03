mod manager;
mod paths;
mod protocol;

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, UNIX_EPOCH},
};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};

pub(crate) use manager::Pdf2zhSidecarState;
use protocol::{
    PreflightPayload, ProbePayload, SidecarEvent, SidecarRequest, TranslatePayload,
    TranslationEnginePayload,
};

use crate::{stable_text_hash, translation_fallback_enabled, AppDatabase, PdfRegistry};

const PDF_TRANSLATION_CACHE_VERSION: &str = "pdf2zh-sidecar-v1";
const MAX_PRIORITY_PARTIAL_PAGES: usize = 24;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StartPdfTranslationInput {
    document_id: String,
    target_lang: String,
    #[serde(default)]
    source_lang: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    artifact_mode: Option<String>,
    #[serde(default)]
    pages: Option<String>,
    #[serde(default)]
    priority_page: Option<u32>,
    #[serde(default)]
    visible_pages: Vec<u32>,
    #[serde(default)]
    nearby_page_radius: Option<u32>,
    #[serde(default)]
    page_count: Option<u32>,
    #[serde(default)]
    force_refresh: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CancelPdfTranslationInput {
    job_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ClearPdfTranslationCacheInput {
    #[serde(default)]
    document_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RequestPdfTranslationPagesInput {
    job_id: String,
    pages: Vec<u32>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PdfTranslationPageRequestOutput {
    job_id: String,
    requested_pages: Vec<u32>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PdfTranslationRuntimeOutput {
    ok: bool,
    protocol_version: String,
    sidecar_version: String,
    pdf2zh_version: String,
    babeldoc_version: String,
    python: String,
    app_data_dir: String,
    log_dir: String,
    cache_dir: String,
    artifacts_dir: String,
    tmp_dir: String,
    error: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PdfTranslationJobOutput {
    job_id: String,
    document_id: String,
    target_lang: String,
    source_lang: String,
    provider_key: String,
    cache_key: String,
    status: String,
    phase: String,
    progress_percent: u32,
    current_page: Option<u32>,
    mono_pdf_path: String,
    dual_pdf_path: String,
    artifact_scope: String,
    artifact_pages: String,
    error: String,
    cached: bool,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PdfTranslationEventOutput {
    job_id: String,
    document_id: String,
    target_lang: String,
    source_lang: String,
    provider_key: String,
    cache_key: String,
    status: String,
    phase: String,
    progress_percent: u32,
    current_page: Option<u32>,
    mono_pdf_path: String,
    dual_pdf_path: String,
    artifact_scope: String,
    artifact_pages: String,
    error: String,
    cached: bool,
}

#[tauri::command]
pub(crate) fn probe_pdf_translation_runtime(
    app: AppHandle,
) -> Result<PdfTranslationRuntimeOutput, String> {
    let paths = paths::Pdf2zhPaths::ensure(&app)?;
    let payload = ProbePayload {
        app_data_dir: paths.root.to_string_lossy().to_string(),
        log_dir: paths.logs.to_string_lossy().to_string(),
        cache_dir: paths.cache.to_string_lossy().to_string(),
        artifacts_dir: paths.artifacts.to_string_lossy().to_string(),
        tmp_dir: paths.tmp.to_string_lossy().to_string(),
    };
    let request = SidecarRequest::new("probe".to_string(), "probe", payload);
    let result = manager::run_request_blocking(&app, request, &paths.logs, None, None, |_| {});
    let mut output = PdfTranslationRuntimeOutput {
        ok: false,
        protocol_version: protocol::PROTOCOL_VERSION.to_string(),
        sidecar_version: String::new(),
        pdf2zh_version: String::new(),
        babeldoc_version: String::new(),
        python: String::new(),
        app_data_dir: paths.root.to_string_lossy().to_string(),
        log_dir: paths.logs.to_string_lossy().to_string(),
        cache_dir: paths.cache.to_string_lossy().to_string(),
        artifacts_dir: paths.artifacts.to_string_lossy().to_string(),
        tmp_dir: paths.tmp.to_string_lossy().to_string(),
        error: String::new(),
    };
    match result {
        Ok(event) => {
            output.ok = true;
            output.sidecar_version = event.sidecar_version;
            output.pdf2zh_version = event.pdf2zh_version;
            output.babeldoc_version = event.babeldoc_version;
            output.python = event.python;
            Ok(output)
        }
        Err(err) => {
            output.error = err;
            Ok(output)
        }
    }
}

// `(async)` forces Tauri to run this command on a worker thread instead of the
// main (UI) thread. The body is synchronous and briefly blocks on a sidecar
// "preflight" round-trip (~seconds); running it on the main thread froze the UI
// on every translate click. The actual translation already runs on a spawned
// thread inside the impl, so only the preflight needed to get off the main thread.
#[tauri::command(async)]
pub(crate) fn start_pdf_translation(
    input: StartPdfTranslationInput,
    app: AppHandle,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<PdfTranslationJobOutput, String> {
    start_pdf_translation_impl(input, app, registry, database)
}

fn start_pdf_translation_impl<R: Runtime>(
    input: StartPdfTranslationInput,
    app: AppHandle<R>,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<PdfTranslationJobOutput, String> {
    let input = normalize_start_input(input)?;
    let source_pdf = source_pdf_path(&registry, &input.document_id)?;
    let source_info = file_source_identity(&source_pdf)?;
    let paths = paths::Pdf2zhPaths::ensure(&app)?;
    let provider_candidates = pdf_provider_candidates(
        input.provider.as_deref(),
        translation_fallback_enabled(&database),
    );
    let provider_plans = provider_candidates
        .into_iter()
        .map(|provider| {
            let provider_key = provider_key(&provider);
            let cache_key = build_cache_key(&source_info.hash, &input, &provider_key);
            (provider, provider_key, cache_key)
        })
        .collect::<Vec<_>>();
    if !input.force_refresh.unwrap_or(false) {
        for (_, _, cache_key) in &provider_plans {
            if let Some(hit) = read_cache_hit(&database, &input.document_id, cache_key)? {
                emit_pdf_translation_event(&app, &hit);
                return Ok(hit);
            }
        }
    }
    let mut preflight_errors = Vec::new();
    let mut selected_provider = None;
    for (provider, provider_key, cache_key) in provider_plans {
        match preflight_pdf_translation_provider(&app, &paths, &input, &provider) {
            Ok(()) => {
                selected_provider = Some((provider, provider_key, cache_key));
                break;
            }
            Err(err) => {
                preflight_errors
                    .push((provider.provider.clone(), classify_pdf_provider_error(&err)));
            }
        }
    }
    let Some((provider, provider_key, cache_key)) = selected_provider else {
        return Err(format_pdf_translation_unavailable(&preflight_errors));
    };

    let job_id = format!("pdf-translation-{}", stable_text_hash(&cache_key));
    upsert_pdf_translation_job(
        &database,
        &job_id,
        &input,
        &provider_key,
        &cache_key,
        &source_info,
    )?;
    let output = read_pdf_translation_job(&database, &job_id)?
        .ok_or_else(|| format!("PDF translation job was not persisted: {job_id}"))?;
    emit_pdf_translation_event(&app, &output);

    let task_app = app.clone();
    std::thread::spawn(move || {
        run_pdf_translation_job(task_app, input, source_pdf, provider, paths, job_id);
    });
    Ok(output)
}

#[tauri::command]
pub(crate) fn cancel_pdf_translation(
    input: CancelPdfTranslationInput,
    app: AppHandle,
    database: State<'_, AppDatabase>,
    state: State<'_, Pdf2zhSidecarState>,
) -> Result<(), String> {
    cancel_pdf_translation_impl(input, app, database, state)
}

fn cancel_pdf_translation_impl<R: Runtime>(
    input: CancelPdfTranslationInput,
    app: AppHandle<R>,
    database: State<'_, AppDatabase>,
    state: State<'_, Pdf2zhSidecarState>,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    cancel_pdf_translation_job_in_conn(&conn, &input.job_id)?;
    drop(conn);
    let _ = state.kill(&input.job_id);
    if let Some(job) = read_pdf_translation_job(&database, &input.job_id)? {
        emit_pdf_translation_event(&app, &job);
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn request_pdf_translation_pages(
    input: RequestPdfTranslationPagesInput,
    database: State<'_, AppDatabase>,
) -> Result<PdfTranslationPageRequestOutput, String> {
    let pages = normalize_page_list(&input.pages);
    if pages.is_empty() {
        return Ok(PdfTranslationPageRequestOutput {
            job_id: input.job_id,
            requested_pages: Vec::new(),
        });
    }
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let requested_pages = insert_page_requests_in_conn(&conn, &input.job_id, &pages)?;
    Ok(PdfTranslationPageRequestOutput {
        job_id: input.job_id,
        requested_pages,
    })
}

fn cancel_pdf_translation_job_in_conn(conn: &Connection, job_id: &str) -> Result<usize, String> {
    conn.execute(
        "UPDATE pdf_translation_jobs
         SET status = 'canceled',
             phase = 'canceled',
             error = '',
             updated_at = unixepoch(),
             finished_at = unixepoch()
         WHERE id = ?1 AND status IN ('queued', 'running', 'partial')",
        params![job_id],
    )
    .map_err(|err| format!("Failed to cancel PDF translation job: {err}"))
}

#[tauri::command]
pub(crate) fn clear_pdf_translation_cache(
    input: ClearPdfTranslationCacheInput,
    app: AppHandle,
    database: State<'_, AppDatabase>,
) -> Result<(), String> {
    let paths = paths::Pdf2zhPaths::ensure(&app)?;
    let artifacts_root = paths
        .artifacts
        .canonicalize()
        .map_err(|err| format!("Failed to resolve artifacts dir: {err}"))?;
    let rows = artifact_paths_for_clear(&database, input.document_id.as_deref())?;
    for path in rows {
        let candidate = PathBuf::from(path);
        let Ok(canonical) = candidate.canonicalize() else {
            continue;
        };
        if canonical.starts_with(&artifacts_root) && canonical.is_file() {
            fs::remove_file(&canonical).map_err(|err| {
                format!("Failed to remove artifact {}: {err}", canonical.display())
            })?;
        }
    }
    clear_cache_rows(&database, input.document_id.as_deref())
}

#[tauri::command]
pub(crate) async fn read_pdf_artifact_bytes(
    path: String,
    app: AppHandle,
) -> Result<tauri::ipc::Response, String> {
    crate::run_blocking_io(move || {
        let paths = paths::Pdf2zhPaths::ensure(&app)?;
        let artifacts_root = paths
            .artifacts
            .canonicalize()
            .map_err(|err| format!("Failed to resolve artifacts dir: {err}"))?;
        let target = PathBuf::from(path)
            .canonicalize()
            .map_err(|err| format!("Failed to resolve PDF artifact: {err}"))?;
        if !target.starts_with(&artifacts_root) {
            return Err("PDF artifact path is outside Lumenfolio managed storage".to_string());
        }
        if target.extension().and_then(|ext| ext.to_str()) != Some("pdf") {
            return Err("PDF artifact is not a PDF file".to_string());
        }
        // Return an ArrayBuffer (raw bytes) instead of a JSON number array — see the
        // note in `read_pdf_bytes`; multi-MB artifacts otherwise block the webview
        // main thread for seconds while deserializing.
        let bytes = fs::read(&target)
            .map_err(|err| crate::map_file_read_error("PDF artifact", &target, err))?;
        Ok(tauri::ipc::Response::new(bytes))
    })
    .await
}

fn run_pdf_translation_job(
    app: AppHandle<impl Runtime>,
    input: StartPdfTranslationInput,
    source_pdf: PathBuf,
    engine: TranslationEnginePayload,
    paths: paths::Pdf2zhPaths,
    job_id: String,
) {
    let database = app.state::<AppDatabase>();
    let state = app.state::<Pdf2zhSidecarState>();
    let _slot = match wait_for_translation_slot(&app, &database, &state, &job_id) {
        Ok(slot) => slot,
        Err(err) => {
            if is_job_canceled(&database, &job_id).unwrap_or(false) {
                if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
                    emit_pdf_translation_event(&app, &job);
                }
                return;
            }
            fail_job(&app, &database, &job_id, &err);
            return;
        }
    };
    match mark_job_running(&database, &job_id) {
        Ok(true) => {}
        Ok(false) if is_job_canceled(&database, &job_id).unwrap_or(false) => {
            if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
                emit_pdf_translation_event(&app, &job);
            }
            return;
        }
        Ok(false) => {
            fail_job(
                &app,
                &database,
                &job_id,
                "PDF translation job could not transition to running from its current state",
            );
            return;
        }
        Err(err) => {
            fail_job(&app, &database, &job_id, &err);
            return;
        }
    }
    if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
        emit_pdf_translation_event(&app, &job);
    }

    let job_artifact_dir = paths.artifacts.join(&job_id);
    if let Err(err) = fs::create_dir_all(&job_artifact_dir) {
        fail_job(
            &app,
            &database,
            &job_id,
            &format!("Failed to create artifact dir: {err}"),
        );
        return;
    }

    let requested_pages = input.pages.clone().filter(|value| !value.trim().is_empty());
    if requested_pages.is_none() {
        let mut completed_partial_pages = HashSet::new();
        let mut partial_queue = priority_page_requests(&input);
        loop {
            if partial_queue.is_empty() {
                partial_queue =
                    requested_partial_pages(&database, &job_id, &completed_partial_pages)
                        .unwrap_or_default();
                if partial_queue.is_empty() {
                    break;
                }
            }
            let partial_pages = partial_queue.remove(0);
            if completed_partial_pages.contains(&partial_pages) {
                continue;
            }
            let partial_dir = job_artifact_dir.join(format!("partial-p{partial_pages}"));
            if let Err(err) = fs::create_dir_all(&partial_dir) {
                fail_job(
                    &app,
                    &database,
                    &job_id,
                    &format!("Failed to create partial artifact dir: {err}"),
                );
                return;
            }
            let partial_payload = build_translate_payload(
                &input,
                &source_pdf,
                &engine,
                &paths,
                &job_id,
                &partial_dir,
                Some(partial_pages.clone()),
                true,
            );
            match run_translate_request(&app, &database, &state, &paths, &job_id, partial_payload) {
                Ok(event) => {
                    if let Err(err) = validate_finished_artifacts(&event, &partial_dir) {
                        fail_job(&app, &database, &job_id, &err);
                        return;
                    }
                    if let Err(err) = mark_job_partial(&database, &job_id, &event, &partial_pages) {
                        fail_job(&app, &database, &job_id, &err);
                        return;
                    }
                    completed_partial_pages.insert(partial_pages.clone());
                    if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
                        emit_pdf_translation_event(&app, &job);
                    }
                }
                Err(err) => {
                    if is_job_canceled(&database, &job_id).unwrap_or(false) {
                        if let Some(job) =
                            read_pdf_translation_job(&database, &job_id).ok().flatten()
                        {
                            emit_pdf_translation_event(&app, &job);
                        }
                        return;
                    }
                    fail_job(&app, &database, &job_id, &err);
                    return;
                }
            }
            if is_job_canceled(&database, &job_id).unwrap_or(false) {
                if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
                    emit_pdf_translation_event(&app, &job);
                }
                return;
            }
        }
    }

    let full_dir = job_artifact_dir.join("full");
    if let Err(err) = fs::create_dir_all(&full_dir) {
        fail_job(
            &app,
            &database,
            &job_id,
            &format!("Failed to create full artifact dir: {err}"),
        );
        return;
    }
    let payload = build_translate_payload(
        &input,
        &source_pdf,
        &engine,
        &paths,
        &job_id,
        &full_dir,
        requested_pages,
        input.pages.is_some(),
    );
    let result = run_translate_request(&app, &database, &state, &paths, &job_id, payload);
    match result {
        Ok(event) => {
            if let Err(err) = validate_finished_artifacts(&event, &full_dir) {
                fail_job(&app, &database, &job_id, &err);
                return;
            }
            match mark_job_succeeded(&database, &job_id, &event) {
                Ok(true) => {}
                Ok(false) if is_job_canceled(&database, &job_id).unwrap_or(false) => {
                    if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
                        emit_pdf_translation_event(&app, &job);
                    }
                    return;
                }
                Ok(false) => {
                    fail_job(
                        &app,
                        &database,
                        &job_id,
                        "PDF translation job could not be finalized from its current state",
                    );
                    return;
                }
                Err(err) => {
                    fail_job(&app, &database, &job_id, &err);
                    return;
                }
            }
            if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
                emit_pdf_translation_event(&app, &job);
            }
        }
        Err(err) => {
            if is_job_canceled(&database, &job_id).unwrap_or(false) {
                if let Some(job) = read_pdf_translation_job(&database, &job_id).ok().flatten() {
                    emit_pdf_translation_event(&app, &job);
                }
                return;
            }
            fail_job(&app, &database, &job_id, &err);
        }
    }
}

struct TranslationSlotGuard<'a> {
    state: &'a Pdf2zhSidecarState,
    job_id: String,
}

impl Drop for TranslationSlotGuard<'_> {
    fn drop(&mut self) {
        self.state.release_job_slot(&self.job_id);
    }
}

fn wait_for_translation_slot<'a>(
    app: &AppHandle<impl Runtime>,
    database: &State<'_, AppDatabase>,
    state: &'a Pdf2zhSidecarState,
    job_id: &str,
) -> Result<TranslationSlotGuard<'a>, String> {
    mark_job_waiting_for_slot(database, job_id)?;
    if let Some(job) = read_pdf_translation_job(database, job_id)? {
        emit_pdf_translation_event(app, &job);
    }
    loop {
        if is_job_canceled(database, job_id)? {
            return Err("PDF translation job was canceled before it started".to_string());
        }
        if state.try_acquire_job_slot(job_id)? {
            return Ok(TranslationSlotGuard {
                state,
                job_id: job_id.to_string(),
            });
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn build_translate_payload(
    input: &StartPdfTranslationInput,
    source_pdf: &Path,
    engine: &TranslationEnginePayload,
    paths: &paths::Pdf2zhPaths,
    job_id: &str,
    output_dir: &Path,
    pages: Option<String>,
    only_include_translated_page: bool,
) -> TranslatePayload {
    TranslatePayload {
        job_id: job_id.to_string(),
        input_pdf_path: source_pdf.to_string_lossy().to_string(),
        output_dir: output_dir.to_string_lossy().to_string(),
        log_dir: paths.logs.to_string_lossy().to_string(),
        cache_dir: paths.cache.to_string_lossy().to_string(),
        source_lang: input
            .source_lang
            .clone()
            .unwrap_or_else(|| "en".to_string()),
        target_lang: input.target_lang.clone(),
        pages,
        artifact_mode: input
            .artifact_mode
            .clone()
            .unwrap_or_else(|| "both".to_string()),
        only_include_translated_page,
        force_refresh: input.force_refresh.unwrap_or(false),
        engine: engine.clone(),
    }
}

fn run_translate_request(
    app: &AppHandle<impl Runtime>,
    database: &State<'_, AppDatabase>,
    state: &State<'_, Pdf2zhSidecarState>,
    paths: &paths::Pdf2zhPaths,
    job_id: &str,
    payload: TranslatePayload,
) -> Result<SidecarEvent, String> {
    let request = SidecarRequest::new(job_id.to_string(), "translate", payload);
    manager::run_request_blocking(
        app,
        request,
        &paths.logs,
        Some(state),
        Some(job_id),
        |event| {
            if matches!(event.event.as_str(), "progress" | "artifact_ready") {
                let _ = update_job_progress(database, job_id, event);
                if let Some(job) = read_pdf_translation_job(database, job_id).ok().flatten() {
                    emit_pdf_translation_event(app, &job);
                }
            }
        },
    )
}

#[cfg(test)]
fn priority_page_request(input: &StartPdfTranslationInput) -> Option<String> {
    let pages = priority_pages(input);
    if pages.is_empty() {
        None
    } else {
        Some(
            pages
                .into_iter()
                .map(|page| page.to_string())
                .collect::<Vec<_>>()
                .join(","),
        )
    }
}

fn priority_page_requests(input: &StartPdfTranslationInput) -> Vec<String> {
    priority_pages(input)
        .into_iter()
        .map(|page| page.to_string())
        .collect()
}

fn priority_pages(input: &StartPdfTranslationInput) -> Vec<u32> {
    let mut pages = Vec::new();
    let page_count = input.page_count.filter(|value| *value > 0);
    let radius = input.nearby_page_radius.unwrap_or(0).min(5);

    if let Some(priority_page) = input.priority_page {
        push_priority_page(&mut pages, priority_page, page_count);
    }
    for page in &input.visible_pages {
        push_priority_page(&mut pages, *page, page_count);
    }
    if radius > 0 {
        for center in input
            .priority_page
            .into_iter()
            .chain(input.visible_pages.iter().copied())
        {
            for offset in 1..=radius {
                if center > offset {
                    push_priority_page(&mut pages, center - offset, page_count);
                }
                push_priority_page(&mut pages, center + offset, page_count);
            }
        }
    }

    pages
}

fn push_priority_page(pages: &mut Vec<u32>, page: u32, page_count: Option<u32>) {
    if pages.len() >= MAX_PRIORITY_PARTIAL_PAGES {
        return;
    }
    if page == 0 {
        return;
    }
    if let Some(page_count) = page_count {
        if page > page_count {
            return;
        }
    }
    if !pages.contains(&page) {
        pages.push(page);
    }
}

fn first_page_from_pages(pages: &str) -> Option<u32> {
    pages
        .split(',')
        .next()?
        .split('-')
        .next()?
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|page| *page > 0)
}

fn normalize_start_input(
    mut input: StartPdfTranslationInput,
) -> Result<StartPdfTranslationInput, String> {
    input.document_id = input.document_id.trim().to_string();
    input.target_lang = input.target_lang.trim().to_string();
    if input.document_id.is_empty() {
        return Err("No document selected".to_string());
    }
    if input.target_lang.is_empty() {
        return Err("No target language selected".to_string());
    }
    if let Some(pages) = input.pages.as_mut() {
        *pages = normalize_pages(pages)?;
    }
    if input.priority_page == Some(0) {
        return Err("Priority page must use a 1-based page number".to_string());
    }
    if input.visible_pages.iter().any(|page| *page == 0) {
        return Err("Visible pages must use 1-based page numbers".to_string());
    }
    if input.page_count == Some(0) {
        input.page_count = None;
    }
    if let Some(mode) = input.artifact_mode.as_mut() {
        let normalized = mode.trim().to_ascii_lowercase();
        if !matches!(normalized.as_str(), "mono" | "dual" | "both") {
            return Err("Unsupported PDF translation artifact mode".to_string());
        }
        *mode = normalized;
    }
    Ok(input)
}

fn normalize_pages(pages: &str) -> Result<String, String> {
    let value = pages.trim();
    if value.is_empty() {
        return Ok(String::new());
    }
    for part in value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("Invalid pages expression".to_string());
        }
        for token in part.split('-').filter(|token| !token.trim().is_empty()) {
            let page = token
                .trim()
                .parse::<u32>()
                .map_err(|_| "Pages must use 1-based page numbers".to_string())?;
            if page == 0 {
                return Err("Pages must use 1-based page numbers".to_string());
            }
        }
    }
    Ok(value.to_string())
}

fn normalize_page_list(pages: &[u32]) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for page in pages {
        if *page == 0 || seen.contains(page) {
            continue;
        }
        seen.insert(*page);
        normalized.push(*page);
        if normalized.len() >= MAX_PRIORITY_PARTIAL_PAGES {
            break;
        }
    }
    normalized
}

fn source_pdf_path(
    registry: &State<'_, PdfRegistry>,
    document_id: &str,
) -> Result<PathBuf, String> {
    let paths = registry
        .paths
        .lock()
        .map_err(|_| "PDF registry lock was poisoned".to_string())?;
    let path = paths
        .get(document_id)
        .cloned()
        .ok_or_else(|| "Unknown PDF document id".to_string())?;
    path.canonicalize()
        .map_err(|err| format!("Failed to resolve source PDF: {err}"))
}

struct SourceIdentity {
    hash: String,
    size: u64,
    modified: u64,
}

fn file_source_identity(path: &Path) -> Result<SourceIdentity, String> {
    let metadata =
        fs::metadata(path).map_err(|err| format!("Failed to read source PDF metadata: {err}"))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_secs())
        .unwrap_or_default();
    let bytes = fs::read(path).map_err(|err| format!("Failed to hash source PDF: {err}"))?;
    Ok(SourceIdentity {
        hash: stable_text_hash(&format!(
            "{}\n{}\n{}",
            metadata.len(),
            modified,
            stable_bytes_hash(&bytes)
        )),
        size: metadata.len(),
        modified,
    })
}

fn stable_bytes_hash(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("bin-{hash:016x}")
}

fn pdf_provider_candidates(
    provider: Option<&str>,
    enable_fallback: bool,
) -> Vec<TranslationEnginePayload> {
    let selected = normalize_pdf_provider_name(provider);
    let mut providers = vec![selected.clone()];
    if enable_fallback {
        match selected.as_str() {
            "google-web" => providers.push("microsoft".to_string()),
            "microsoft" | "bing" => providers.push("google-web".to_string()),
            _ => {}
        }
    }
    providers
        .into_iter()
        .map(|provider| build_engine_payload(Some(&provider)))
        .fold(Vec::new(), |mut engines, engine| {
            if !engines
                .iter()
                .any(|item: &TranslationEnginePayload| item.provider == engine.provider)
            {
                engines.push(engine);
            }
            engines
        })
}

fn normalize_pdf_provider_name(provider: Option<&str>) -> String {
    let value = provider.unwrap_or("google-web").trim().to_ascii_lowercase();
    match value.as_str() {
        "" => "google-web".to_string(),
        "google" => "google-web".to_string(),
        "bing" => "microsoft".to_string(),
        other => other.to_string(),
    }
}

fn build_engine_payload(provider: Option<&str>) -> TranslationEnginePayload {
    let provider = normalize_pdf_provider_name(provider);
    let uses_llm_config = matches!(provider.as_str(), "llm" | "openai" | "openai-compatible");
    TranslationEnginePayload {
        provider,
        base_url: uses_llm_config
            .then(|| std::env::var("LUMENFOLIO_PDF2ZH_BASE_URL").ok())
            .flatten(),
        model: uses_llm_config
            .then(|| std::env::var("LUMENFOLIO_PDF2ZH_MODEL").ok())
            .flatten(),
        api_key: uses_llm_config
            .then(|| std::env::var("LUMENFOLIO_PDF2ZH_API_KEY").ok())
            .flatten(),
    }
}

fn preflight_pdf_translation_provider(
    app: &AppHandle<impl Runtime>,
    paths: &paths::Pdf2zhPaths,
    input: &StartPdfTranslationInput,
    engine: &TranslationEnginePayload,
) -> Result<(), String> {
    let output_dir = paths.tmp.join("preflight");
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("Failed to create PDF translation preflight dir: {err}"))?;
    let payload = PreflightPayload {
        log_dir: paths.logs.to_string_lossy().to_string(),
        cache_dir: paths.cache.to_string_lossy().to_string(),
        output_dir: output_dir.to_string_lossy().to_string(),
        source_lang: input
            .source_lang
            .clone()
            .unwrap_or_else(|| "en".to_string()),
        target_lang: input.target_lang.clone(),
        engine: engine.clone(),
    };
    let request_id = format!(
        "pdf-preflight-{}",
        stable_text_hash(&format!(
            "{}\n{}\n{}",
            engine.provider, payload.source_lang, payload.target_lang
        ))
    );
    let request = SidecarRequest::new(request_id, "preflight", payload);
    manager::run_request_blocking(app, request, &paths.logs, None, None, |_| {}).map(|_| ())
}

fn classify_pdf_provider_error(error: &str) -> String {
    let lower = error.to_ascii_lowercase();
    if lower.contains("ssl") || lower.contains("tls") || lower.contains("eof") {
        "TLS connection failed; the translation endpoint may be blocked, interrupted, or routed through an incompatible proxy.".to_string()
    } else if lower.contains("timeout") || lower.contains("timed out") {
        "Request timed out; the translation endpoint is not reachable from this network."
            .to_string()
    } else if lower.contains("401") || lower.contains("403") || lower.contains("unauthorized") {
        "Authentication or access was rejected by the translation provider.".to_string()
    } else if lower.contains("unsupported pdf translation provider") {
        "This provider is not supported by the PDF translation sidecar.".to_string()
    } else if lower.contains("not configured") || lower.contains("api key") {
        "The translation provider is not configured.".to_string()
    } else {
        truncate_for_pdf_error(error, 220)
    }
}

fn format_pdf_translation_unavailable(errors: &[(String, String)]) -> String {
    let labels = errors
        .iter()
        .map(|(provider, _)| pdf_provider_label(provider))
        .fold(Vec::<&'static str>::new(), |mut labels, label| {
            if !labels.contains(&label) {
                labels.push(label);
            }
            labels
        })
        .join(" and ");
    let details = errors
        .iter()
        .map(|(provider, error)| format!("{}: {}", pdf_provider_label(provider), error))
        .collect::<Vec<_>>()
        .join("; ");
    if details.is_empty() {
        "Translation is unavailable. No PDF translation provider could be checked.".to_string()
    } else {
        format!(
            "Translation is unavailable. {labels} did not pass the network/provider check. {details}"
        )
    }
}

fn pdf_provider_label(provider: &str) -> &'static str {
    match provider {
        "google-web" | "google" => "Google Web",
        "microsoft" | "bing" => "Microsoft/Bing Web",
        "llm" | "openai" | "openai-compatible" => "LLM provider",
        _ => "Provider",
    }
}

fn truncate_for_pdf_error(value: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for ch in value.chars().take(max_chars) {
        output.push(ch);
    }
    if value.chars().count() > max_chars {
        output.push_str("...");
    }
    output
}

fn provider_key(engine: &TranslationEnginePayload) -> String {
    let fingerprint = stable_text_hash(&format!(
        "{}\n{}\n{}",
        engine.provider,
        engine.base_url.clone().unwrap_or_default(),
        engine.model.clone().unwrap_or_default()
    ));
    format!("{}:{fingerprint}", engine.provider)
}

fn build_cache_key(
    source_hash: &str,
    input: &StartPdfTranslationInput,
    provider_key: &str,
) -> String {
    stable_text_hash(&format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        PDF_TRANSLATION_CACHE_VERSION,
        source_hash,
        input
            .source_lang
            .clone()
            .unwrap_or_else(|| "en".to_string()),
        input.target_lang,
        provider_key,
        input
            .artifact_mode
            .clone()
            .unwrap_or_else(|| "both".to_string()),
        input.pages.clone().unwrap_or_default()
    ))
}

fn insert_page_requests_in_conn(
    conn: &Connection,
    job_id: &str,
    pages: &[u32],
) -> Result<Vec<u32>, String> {
    let status: Option<String> = conn
        .query_row(
            "SELECT status FROM pdf_translation_jobs WHERE id = ?1",
            params![job_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| format!("Failed to read PDF translation job: {err}"))?;
    if !matches!(status.as_deref(), Some("queued" | "running" | "partial")) {
        return Ok(Vec::new());
    }
    let mut requested = Vec::new();
    for page in pages {
        let event_id = stable_text_hash(&format!("{job_id}:page_request:{page}"));
        let updated = conn
            .execute(
                "INSERT INTO pdf_translation_events
                   (id, job_id, event_type, payload_json, created_at)
                 VALUES (?1, ?2, 'page_request', ?3, unixepoch())
                 ON CONFLICT(id) DO NOTHING",
                params![
                    event_id,
                    job_id,
                    serde_json::json!({ "page": page }).to_string(),
                ],
            )
            .map_err(|err| format!("Failed to persist PDF translation page request: {err}"))?;
        if updated > 0 {
            requested.push(*page);
        }
    }
    Ok(requested)
}

fn requested_partial_pages(
    database: &State<'_, AppDatabase>,
    job_id: &str,
    completed_pages: &HashSet<String>,
) -> Result<Vec<String>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT payload_json
             FROM pdf_translation_events
             WHERE job_id = ?1 AND event_type = 'page_request'
             ORDER BY created_at ASC",
        )
        .map_err(|err| format!("Failed to read PDF translation page requests: {err}"))?;
    let mut rows = stmt
        .query(params![job_id])
        .map_err(|err| format!("Failed to read PDF translation page requests: {err}"))?;
    let mut pages = Vec::new();
    let mut seen = HashSet::new();
    while let Some(row) = rows
        .next()
        .map_err(|err| format!("Failed to read PDF translation page request: {err}"))?
    {
        let payload: String = row
            .get(0)
            .map_err(|err| format!("Failed to decode PDF translation page request: {err}"))?;
        let page = serde_json::from_str::<serde_json::Value>(&payload)
            .ok()
            .and_then(|value| value.get("page").and_then(|page| page.as_u64()))
            .filter(|page| *page > 0)
            .map(|page| page.to_string());
        let Some(page) = page else {
            continue;
        };
        if completed_pages.contains(&page) || seen.contains(&page) {
            continue;
        }
        seen.insert(page.clone());
        pages.push(page);
        if pages.len() >= MAX_PRIORITY_PARTIAL_PAGES {
            break;
        }
    }
    Ok(pages)
}

fn upsert_pdf_translation_job(
    database: &State<'_, AppDatabase>,
    job_id: &str,
    input: &StartPdfTranslationInput,
    provider_key: &str,
    cache_key: &str,
    source: &SourceIdentity,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "INSERT INTO pdf_translation_jobs
           (id, document_id, source_hash, source_size, source_modified, target_lang,
            source_lang, provider_key, cache_key, status, phase, progress_percent,
            current_page, mono_pdf_path, dual_pdf_path, error, created_at, updated_at,
            started_at, finished_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'queued', 'queued', 0,
                 ?10, '', '', '', unixepoch(), unixepoch(), 0, 0)
         ON CONFLICT(id) DO UPDATE SET
           status = 'queued',
           phase = 'queued',
           progress_percent = 0,
           current_page = excluded.current_page,
           mono_pdf_path = '',
           dual_pdf_path = '',
           error = '',
           updated_at = unixepoch(),
           started_at = 0,
           finished_at = 0",
        params![
            job_id,
            input.document_id,
            source.hash,
            source.size as i64,
            source.modified as i64,
            input.target_lang,
            input
                .source_lang
                .clone()
                .unwrap_or_else(|| "en".to_string()),
            provider_key,
            cache_key,
            input.priority_page.unwrap_or(0)
        ],
    )
    .map_err(|err| format!("Failed to create PDF translation job: {err}"))?;
    Ok(())
}

fn read_cache_hit(
    database: &State<'_, AppDatabase>,
    document_id: &str,
    cache_key: &str,
) -> Result<Option<PdfTranslationJobOutput>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    read_cache_hit_from_conn(&conn, document_id, cache_key)
}

fn read_cache_hit_from_conn(
    conn: &Connection,
    document_id: &str,
    cache_key: &str,
) -> Result<Option<PdfTranslationJobOutput>, String> {
    let row = conn
        .query_row(
            "SELECT jobs.id, jobs.document_id, jobs.target_lang, jobs.source_lang,
                    jobs.provider_key, jobs.cache_key, artifacts.mono_pdf_path,
                    artifacts.dual_pdf_path
             FROM pdf_translation_artifacts artifacts
             JOIN pdf_translation_jobs jobs ON jobs.cache_key = artifacts.cache_key
             WHERE artifacts.document_id = ?1 AND artifacts.cache_key = ?2
             ORDER BY artifacts.created_at DESC
             LIMIT 1",
            params![document_id, cache_key],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                ))
            },
        )
        .optional()
        .map_err(|err| format!("Failed to read PDF translation cache: {err}"))?;
    let Some((job_id, document_id, target_lang, source_lang, provider_key, cache_key, mono, dual)) =
        row
    else {
        return Ok(None);
    };
    let mono_exists = mono.is_empty() || Path::new(&mono).is_file();
    let dual_exists = dual.is_empty() || Path::new(&dual).is_file();
    if !mono_exists || !dual_exists {
        return Ok(None);
    }
    Ok(Some(PdfTranslationJobOutput {
        job_id,
        document_id,
        target_lang,
        source_lang,
        provider_key,
        cache_key,
        status: "succeeded".to_string(),
        phase: "cache_hit".to_string(),
        progress_percent: 100,
        current_page: None,
        mono_pdf_path: mono,
        dual_pdf_path: dual,
        artifact_scope: "full".to_string(),
        artifact_pages: String::new(),
        error: String::new(),
        cached: true,
    }))
}

fn read_pdf_translation_job(
    database: &State<'_, AppDatabase>,
    job_id: &str,
) -> Result<Option<PdfTranslationJobOutput>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.query_row(
        "SELECT id, document_id, target_lang, source_lang, provider_key, cache_key,
                status, phase, progress_percent, current_page, mono_pdf_path,
                dual_pdf_path, error
         FROM pdf_translation_jobs
         WHERE id = ?1",
        params![job_id],
        |row| {
            let current_page = row.get::<_, i64>(9)?;
            Ok(PdfTranslationJobOutput {
                job_id: row.get(0)?,
                document_id: row.get(1)?,
                target_lang: row.get(2)?,
                source_lang: row.get(3)?,
                provider_key: row.get(4)?,
                cache_key: row.get(5)?,
                status: row.get(6)?,
                phase: row.get(7)?,
                progress_percent: row.get(8)?,
                current_page: (current_page > 0).then_some(current_page as u32),
                mono_pdf_path: row.get(10)?,
                dual_pdf_path: row.get(11)?,
                artifact_scope: if row.get::<_, String>(6)? == "partial" {
                    "partial".to_string()
                } else {
                    "full".to_string()
                },
                artifact_pages: if current_page > 0 {
                    current_page.to_string()
                } else {
                    String::new()
                },
                error: row.get(12)?,
                cached: false,
            })
        },
    )
    .optional()
    .map_err(|err| format!("Failed to read PDF translation job: {err}"))
}

fn mark_job_waiting_for_slot(
    database: &State<'_, AppDatabase>,
    job_id: &str,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "UPDATE pdf_translation_jobs
         SET phase = 'waiting_for_slot',
             updated_at = unixepoch()
         WHERE id = ?1 AND status = 'queued'",
        params![job_id],
    )
    .map_err(|err| format!("Failed to mark PDF translation job waiting: {err}"))?;
    Ok(())
}

fn mark_job_running(database: &State<'_, AppDatabase>, job_id: &str) -> Result<bool, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    mark_job_running_in_conn(&conn, job_id)
}

fn mark_job_running_in_conn(conn: &Connection, job_id: &str) -> Result<bool, String> {
    let updated = conn
        .execute(
            "UPDATE pdf_translation_jobs
         SET status = 'running',
             phase = 'starting',
             started_at = CASE WHEN started_at = 0 THEN unixepoch() ELSE started_at END,
             updated_at = unixepoch()
         WHERE id = ?1 AND status = 'queued'",
            params![job_id],
        )
        .map_err(|err| format!("Failed to mark PDF translation job running: {err}"))?;
    Ok(updated > 0)
}

fn update_job_progress(
    database: &State<'_, AppDatabase>,
    job_id: &str,
    event: &SidecarEvent,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "UPDATE pdf_translation_jobs
         SET phase = ?2,
             progress_percent = ?3,
             current_page = ?4,
             updated_at = unixepoch()
         WHERE id = ?1 AND status IN ('running', 'partial')",
        params![
            job_id,
            if event.phase.is_empty() {
                "translating"
            } else {
                event.phase.as_str()
            },
            event.progress_percent.min(99),
            event
                .current_page
                .or_else(|| first_page_from_pages(&event.artifact_pages))
                .unwrap_or(0)
        ],
    )
    .map_err(|err| format!("Failed to update PDF translation progress: {err}"))?;
    Ok(())
}

fn mark_job_partial(
    database: &State<'_, AppDatabase>,
    job_id: &str,
    event: &SidecarEvent,
    artifact_pages: &str,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "UPDATE pdf_translation_jobs
         SET status = 'partial',
             phase = 'partial_ready',
             progress_percent = MAX(progress_percent, ?2),
             current_page = ?3,
             mono_pdf_path = ?4,
             dual_pdf_path = ?5,
             error = '',
             updated_at = unixepoch()
         WHERE id = ?1 AND status IN ('running', 'partial')",
        params![
            job_id,
            event.progress_percent.max(35).min(99),
            first_page_from_pages(artifact_pages)
                .or(event.current_page)
                .unwrap_or(0),
            event.mono_pdf_path,
            event.dual_pdf_path,
        ],
    )
    .map_err(|err| format!("Failed to mark PDF translation partial artifact ready: {err}"))?;
    conn.execute(
        "INSERT INTO pdf_translation_events
           (id, job_id, event_type, payload_json, created_at)
         VALUES (?1, ?2, 'artifact_ready', ?3, unixepoch())
         ON CONFLICT(id) DO UPDATE SET
           payload_json = excluded.payload_json,
           created_at = unixepoch()",
        params![
            stable_text_hash(&format!(
                "{job_id}:artifact_ready:{artifact_pages}:{}:{}",
                event.mono_pdf_path, event.dual_pdf_path
            )),
            job_id,
            serde_json::json!({
                "artifactScope": if event.artifact_scope.is_empty() { "partial" } else { event.artifact_scope.as_str() },
                "artifactPages": artifact_pages,
                "monoPdfPath": event.mono_pdf_path,
                "dualPdfPath": event.dual_pdf_path,
            })
            .to_string()
        ],
    )
    .map_err(|err| format!("Failed to persist PDF translation partial event: {err}"))?;
    Ok(())
}

fn mark_job_succeeded(
    database: &State<'_, AppDatabase>,
    job_id: &str,
    event: &SidecarEvent,
) -> Result<bool, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    mark_job_succeeded_in_conn(&conn, job_id, event)
}

fn mark_job_succeeded_in_conn(
    conn: &Connection,
    job_id: &str,
    event: &SidecarEvent,
) -> Result<bool, String> {
    let updated = conn
        .execute(
            "UPDATE pdf_translation_jobs
         SET status = 'succeeded',
             phase = 'finished',
             progress_percent = 100,
             current_page = 0,
             mono_pdf_path = ?2,
             dual_pdf_path = ?3,
             error = '',
             updated_at = unixepoch(),
             finished_at = unixepoch()
         WHERE id = ?1 AND status IN ('running', 'partial')",
            params![job_id, event.mono_pdf_path, event.dual_pdf_path],
        )
        .map_err(|err| format!("Failed to mark PDF translation job succeeded: {err}"))?;
    if updated == 0 {
        return Ok(false);
    }
    conn.execute(
        "INSERT INTO pdf_translation_artifacts
           (cache_key, document_id, job_id, mono_pdf_path, dual_pdf_path, created_at)
         SELECT cache_key, document_id, id, mono_pdf_path, dual_pdf_path, unixepoch()
         FROM pdf_translation_jobs
         WHERE id = ?1
         ON CONFLICT(cache_key) DO UPDATE SET
           job_id = excluded.job_id,
           mono_pdf_path = excluded.mono_pdf_path,
           dual_pdf_path = excluded.dual_pdf_path,
           created_at = unixepoch()",
        params![job_id],
    )
    .map_err(|err| format!("Failed to persist PDF translation artifact: {err}"))?;
    conn.execute(
        "INSERT INTO pdf_translation_events
           (id, job_id, event_type, payload_json, created_at)
         VALUES (?1, ?2, 'finish', ?3, unixepoch())",
        params![
            stable_text_hash(&format!("{job_id}:finish")),
            job_id,
            serde_json::json!({
                "monoPdfPath": event.mono_pdf_path,
                "dualPdfPath": event.dual_pdf_path,
            })
            .to_string()
        ],
    )
    .map_err(|err| format!("Failed to persist PDF translation finish event: {err}"))?;
    Ok(true)
}

fn validate_finished_artifacts(
    event: &SidecarEvent,
    job_artifact_dir: &Path,
) -> Result<(), String> {
    if event.mono_pdf_path.is_empty() && event.dual_pdf_path.is_empty() {
        return Err("PDF translation finished without any PDF artifact".to_string());
    }
    let root = job_artifact_dir
        .canonicalize()
        .map_err(|err| format!("Failed to resolve artifact output dir: {err}"))?;
    for artifact_path in [&event.mono_pdf_path, &event.dual_pdf_path] {
        if artifact_path.is_empty() {
            continue;
        }
        let artifact = PathBuf::from(artifact_path)
            .canonicalize()
            .map_err(|err| format!("Failed to resolve PDF translation artifact: {err}"))?;
        if !artifact.starts_with(&root) {
            return Err(
                "PDF translation artifact path is outside the job output directory".to_string(),
            );
        }
        if artifact.extension().and_then(|ext| ext.to_str()) != Some("pdf") {
            return Err("PDF translation artifact is not a PDF file".to_string());
        }
    }
    Ok(())
}

fn fail_job(
    app: &AppHandle<impl Runtime>,
    database: &State<'_, AppDatabase>,
    job_id: &str,
    error: &str,
) {
    let sanitized = redact_secret_text(error);
    if let Ok(conn) = database.conn.lock() {
        let _ = fail_job_in_conn(&conn, job_id, &sanitized);
    }
    if let Some(job) = read_pdf_translation_job(database, job_id).ok().flatten() {
        emit_pdf_translation_event(app, &job);
    }
}

fn fail_job_in_conn(conn: &Connection, job_id: &str, error: &str) -> Result<bool, String> {
    let updated = conn
        .execute(
            "UPDATE pdf_translation_jobs
             SET status = 'failed',
                 phase = 'failed',
                 error = ?2,
                 updated_at = unixepoch(),
                 finished_at = unixepoch()
             WHERE id = ?1 AND status IN ('queued', 'running', 'partial')",
            params![job_id, error],
        )
        .map_err(|err| format!("Failed to mark PDF translation job failed: {err}"))?;
    Ok(updated > 0)
}

fn is_job_canceled(database: &State<'_, AppDatabase>, job_id: &str) -> Result<bool, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    let status: Option<String> = conn
        .query_row(
            "SELECT status FROM pdf_translation_jobs WHERE id = ?1",
            params![job_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| format!("Failed to read PDF translation job status: {err}"))?;
    Ok(status.as_deref() == Some("canceled"))
}

fn emit_pdf_translation_event(app: &AppHandle<impl Runtime>, job: &PdfTranslationJobOutput) {
    let event = PdfTranslationEventOutput {
        job_id: job.job_id.clone(),
        document_id: job.document_id.clone(),
        target_lang: job.target_lang.clone(),
        source_lang: job.source_lang.clone(),
        provider_key: job.provider_key.clone(),
        cache_key: job.cache_key.clone(),
        status: job.status.clone(),
        phase: job.phase.clone(),
        progress_percent: job.progress_percent,
        current_page: job.current_page,
        mono_pdf_path: job.mono_pdf_path.clone(),
        dual_pdf_path: job.dual_pdf_path.clone(),
        artifact_scope: job.artifact_scope.clone(),
        artifact_pages: job.artifact_pages.clone(),
        error: job.error.clone(),
        cached: job.cached,
    };
    let _ = app.emit("lumenfolio://pdf-translation", event);
}

fn artifact_paths_for_clear(
    database: &State<'_, AppDatabase>,
    document_id: Option<&str>,
) -> Result<Vec<String>, String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    artifact_paths_for_clear_from_conn(&conn, document_id)
}

fn artifact_paths_for_clear_from_conn(
    conn: &Connection,
    document_id: Option<&str>,
) -> Result<Vec<String>, String> {
    let sql = if document_id.is_some() {
        "SELECT mono_pdf_path FROM pdf_translation_artifacts WHERE document_id = ?1
         UNION
         SELECT dual_pdf_path FROM pdf_translation_artifacts WHERE document_id = ?1
         UNION
         SELECT mono_pdf_path FROM pdf_translation_jobs WHERE document_id = ?1
         UNION
         SELECT dual_pdf_path FROM pdf_translation_jobs WHERE document_id = ?1"
    } else {
        "SELECT mono_pdf_path FROM pdf_translation_artifacts
         UNION
         SELECT dual_pdf_path FROM pdf_translation_artifacts
         UNION
         SELECT mono_pdf_path FROM pdf_translation_jobs
         UNION
         SELECT dual_pdf_path FROM pdf_translation_jobs"
    };
    let mut stmt = conn
        .prepare(sql)
        .map_err(|err| format!("Failed to read PDF translation artifacts: {err}"))?;
    let rows = if let Some(document_id) = document_id {
        stmt.query_map(params![document_id], |row| row.get::<_, String>(0))
            .map_err(|err| format!("Failed to read PDF translation artifacts: {err}"))?
            .collect::<Result<Vec<_>, _>>()
    } else {
        stmt.query_map([], |row| row.get::<_, String>(0))
            .map_err(|err| format!("Failed to read PDF translation artifacts: {err}"))?
            .collect::<Result<Vec<_>, _>>()
    }
    .map_err(|err| format!("Failed to decode PDF translation artifacts: {err}"))?;
    Ok(rows.into_iter().filter(|path| !path.is_empty()).collect())
}

fn clear_cache_rows(
    database: &State<'_, AppDatabase>,
    document_id: Option<&str>,
) -> Result<(), String> {
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    if let Some(document_id) = document_id {
        conn.execute(
            "DELETE FROM pdf_translation_artifacts WHERE document_id = ?1",
            params![document_id],
        )
        .map_err(|err| format!("Failed to clear PDF translation artifacts: {err}"))?;
        conn.execute(
            "DELETE FROM pdf_translation_events
             WHERE job_id IN (SELECT id FROM pdf_translation_jobs WHERE document_id = ?1)",
            params![document_id],
        )
        .map_err(|err| format!("Failed to clear PDF translation events: {err}"))?;
        conn.execute(
            "DELETE FROM pdf_translation_jobs WHERE document_id = ?1",
            params![document_id],
        )
        .map_err(|err| format!("Failed to clear PDF translation jobs: {err}"))?;
    } else {
        conn.execute("DELETE FROM pdf_translation_artifacts", [])
            .map_err(|err| format!("Failed to clear PDF translation artifacts: {err}"))?;
        conn.execute("DELETE FROM pdf_translation_events", [])
            .map_err(|err| format!("Failed to clear PDF translation events: {err}"))?;
        conn.execute("DELETE FROM pdf_translation_jobs", [])
            .map_err(|err| format!("Failed to clear PDF translation jobs: {err}"))?;
    }
    Ok(())
}

fn redact_secret_text(value: &str) -> String {
    let mut redacted = String::with_capacity(value.len());
    for token in value.split_whitespace() {
        let lower = token.to_ascii_lowercase();
        let next = if lower.contains("key=")
            || lower.contains("api_key")
            || lower.starts_with("sk-")
            || lower.contains("authorization")
        {
            "[REDACTED]"
        } else {
            token
        };
        if !redacted.is_empty() {
            redacted.push(' ');
        }
        redacted.push_str(next);
    }
    redacted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::sync::{Mutex as StdMutex, OnceLock};

    #[cfg(unix)]
    static SIDECAR_ENV_LOCK: OnceLock<StdMutex<()>> = OnceLock::new();

    #[cfg(unix)]
    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    #[cfg(unix)]
    struct EnvValueGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    #[cfg(unix)]
    impl EnvGuard {
        fn set(key: &'static str, value: &Path) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    #[cfg(unix)]
    impl EnvValueGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    #[cfg(unix)]
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[cfg(unix)]
    impl Drop for EnvValueGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn page_ranges_reject_zero_based_pages() {
        assert!(normalize_pages("1,3-5").is_ok());
        assert!(normalize_pages("0").is_err());
        assert!(normalize_pages("1,0-2").is_err());
    }

    #[test]
    fn priority_page_request_uses_one_based_page() {
        let input = StartPdfTranslationInput {
            document_id: "doc".to_string(),
            target_lang: "zh".to_string(),
            source_lang: None,
            provider: None,
            artifact_mode: None,
            pages: None,
            priority_page: Some(7),
            visible_pages: Vec::new(),
            nearby_page_radius: None,
            page_count: None,
            force_refresh: None,
        };
        assert_eq!(priority_page_request(&input), Some("7".to_string()));
    }

    #[test]
    fn priority_page_request_includes_visible_and_nearby_pages_in_order() {
        let input = StartPdfTranslationInput {
            document_id: "doc".to_string(),
            target_lang: "zh".to_string(),
            source_lang: None,
            provider: None,
            artifact_mode: None,
            pages: None,
            priority_page: Some(7),
            visible_pages: vec![8, 7, 10],
            nearby_page_radius: Some(1),
            page_count: Some(10),
            force_refresh: None,
        };
        assert_eq!(
            priority_page_request(&input),
            Some("7,8,10,6,9".to_string())
        );
    }

    #[test]
    fn priority_page_requests_split_pages_into_ordered_partial_jobs() {
        let input = StartPdfTranslationInput {
            document_id: "doc".to_string(),
            target_lang: "zh".to_string(),
            source_lang: None,
            provider: None,
            artifact_mode: None,
            pages: None,
            priority_page: Some(5),
            visible_pages: vec![6, 5],
            nearby_page_radius: Some(1),
            page_count: Some(7),
            force_refresh: None,
        };
        assert_eq!(
            priority_page_requests(&input),
            vec![
                "5".to_string(),
                "6".to_string(),
                "4".to_string(),
                "7".to_string(),
            ]
        );
    }

    #[test]
    fn priority_page_request_caps_partial_page_count() {
        let input = StartPdfTranslationInput {
            document_id: "doc".to_string(),
            target_lang: "zh".to_string(),
            source_lang: None,
            provider: None,
            artifact_mode: None,
            pages: None,
            priority_page: Some(1),
            visible_pages: (2..=100).collect(),
            nearby_page_radius: Some(5),
            page_count: Some(100),
            force_refresh: None,
        };
        let pages = priority_page_request(&input).expect("priority pages");
        assert_eq!(pages.split(',').count(), MAX_PRIORITY_PARTIAL_PAGES);
    }

    #[test]
    fn first_page_from_pages_accepts_ranges() {
        assert_eq!(first_page_from_pages("7-9"), Some(7));
        assert_eq!(first_page_from_pages(" 3,5 "), Some(3));
        assert_eq!(first_page_from_pages("0"), None);
    }

    #[test]
    fn redacts_common_secret_shapes() {
        let value = redact_secret_text("failed api_key=sk-test Authorization bearer sk-live");
        assert!(!value.contains("sk-test"));
        assert!(!value.contains("sk-live"));
        assert!(value.contains("[REDACTED]"));
    }

    #[test]
    fn byte_hash_is_stable_for_binary_pdf_bytes() {
        assert_eq!(
            stable_bytes_hash(&[0, 159, 146, 150]),
            stable_bytes_hash(&[0, 159, 146, 150])
        );
        assert_ne!(
            stable_bytes_hash(&[0, 159, 146, 150]),
            stable_bytes_hash(&[0, 159, 146, 151])
        );
    }

    #[test]
    fn cache_key_changes_when_translation_inputs_change() {
        let mut input = StartPdfTranslationInput {
            document_id: "doc".to_string(),
            target_lang: "zh".to_string(),
            source_lang: Some("en".to_string()),
            provider: Some("google-web".to_string()),
            artifact_mode: Some("both".to_string()),
            pages: None,
            priority_page: Some(1),
            visible_pages: Vec::new(),
            nearby_page_radius: None,
            page_count: None,
            force_refresh: None,
        };
        let first = build_cache_key("source-a", &input, "google-web:stable");
        input.target_lang = "ja".to_string();
        assert_ne!(
            first,
            build_cache_key("source-a", &input, "google-web:stable")
        );
        input.target_lang = "zh".to_string();
        input.pages = Some("1".to_string());
        assert_ne!(
            first,
            build_cache_key("source-a", &input, "google-web:stable")
        );
    }

    #[test]
    fn pdf_provider_candidates_use_free_web_fallback_pair() {
        let google_first = pdf_provider_candidates(Some("google-web"), true)
            .into_iter()
            .map(|engine| engine.provider)
            .collect::<Vec<_>>();
        assert_eq!(google_first, vec!["google-web", "microsoft"]);

        let microsoft_first = pdf_provider_candidates(Some("bing"), true)
            .into_iter()
            .map(|engine| engine.provider)
            .collect::<Vec<_>>();
        assert_eq!(microsoft_first, vec!["microsoft", "google-web"]);

        let no_fallback = pdf_provider_candidates(Some("google-web"), false)
            .into_iter()
            .map(|engine| engine.provider)
            .collect::<Vec<_>>();
        assert_eq!(no_fallback, vec!["google-web"]);
    }

    #[test]
    fn pdf_preflight_errors_are_user_friendly() {
        let message = format_pdf_translation_unavailable(&[
            (
                "google-web".to_string(),
                classify_pdf_provider_error("requests.exceptions.SSLError: EOF"),
            ),
            (
                "microsoft".to_string(),
                classify_pdf_provider_error("request timed out"),
            ),
        ]);
        assert!(message.contains("Translation is unavailable"));
        assert!(message.contains("Google Web"));
        assert!(message.contains("Microsoft/Bing Web"));
        assert!(!message.contains("Future at"));
    }

    #[cfg(unix)]
    #[test]
    fn free_web_engine_payload_ignores_llm_environment() {
        let _env_lock = SIDECAR_ENV_LOCK
            .get_or_init(|| StdMutex::new(()))
            .lock()
            .unwrap();
        let _base_url = EnvValueGuard::set("LUMENFOLIO_PDF2ZH_BASE_URL", "https://llm.example/v1");
        let _model = EnvValueGuard::set("LUMENFOLIO_PDF2ZH_MODEL", "llm-model");
        let _api_key = EnvValueGuard::set("LUMENFOLIO_PDF2ZH_API_KEY", "sk-test");

        let google = build_engine_payload(Some("google-web"));
        assert!(google.base_url.is_none());
        assert!(google.model.is_none());
        assert!(google.api_key.is_none());

        let llm = build_engine_payload(Some("llm"));
        assert_eq!(llm.base_url.as_deref(), Some("https://llm.example/v1"));
        assert_eq!(llm.model.as_deref(), Some("llm-model"));
        assert_eq!(llm.api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    fn cancel_job_only_marks_active_statuses_canceled() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               updated_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );",
        )
        .unwrap();

        for status in ["queued", "running", "partial", "succeeded", "failed"] {
            conn.execute(
                "INSERT INTO pdf_translation_jobs (id, status, phase, error)
                 VALUES (?1, ?2, 'before', 'stale')",
                params![format!("job-{status}"), status],
            )
            .unwrap();
        }

        assert_eq!(
            cancel_pdf_translation_job_in_conn(&conn, "job-queued").unwrap(),
            1
        );
        assert_eq!(
            cancel_pdf_translation_job_in_conn(&conn, "job-running").unwrap(),
            1
        );
        assert_eq!(
            cancel_pdf_translation_job_in_conn(&conn, "job-partial").unwrap(),
            1
        );
        assert_eq!(
            cancel_pdf_translation_job_in_conn(&conn, "job-succeeded").unwrap(),
            0
        );
        assert_eq!(
            cancel_pdf_translation_job_in_conn(&conn, "job-failed").unwrap(),
            0
        );

        let rows = conn
            .prepare("SELECT id, status, phase, error, finished_at FROM pdf_translation_jobs")
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        for (id, status, phase, error, finished_at) in rows {
            if matches!(id.as_str(), "job-queued" | "job-running" | "job-partial") {
                assert_eq!(status, "canceled");
                assert_eq!(phase, "canceled");
                assert_eq!(error, "");
                assert!(finished_at > 0);
            } else {
                assert!(matches!(status.as_str(), "succeeded" | "failed"));
                assert_eq!(phase, "before");
                assert_eq!(error, "stale");
                assert_eq!(finished_at, 0);
            }
        }
    }

    #[test]
    fn cancel_command_marks_running_job_canceled() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL DEFAULT 'doc-1',
               target_lang TEXT NOT NULL DEFAULT 'zh',
               source_lang TEXT NOT NULL DEFAULT 'en',
               provider_key TEXT NOT NULL DEFAULT 'google-web:test',
               cache_key TEXT NOT NULL DEFAULT 'cache-1',
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               progress_percent INTEGER NOT NULL DEFAULT 0,
               current_page INTEGER NOT NULL DEFAULT 0,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               updated_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs
             (id, status, phase, progress_percent, current_page, mono_pdf_path)
             VALUES ('job-running', 'running', 'translating', 45, 7, '/tmp/partial.pdf')",
            [],
        )
        .unwrap();

        let app = tauri::test::mock_builder()
            .manage(AppDatabase {
                conn: std::sync::Mutex::new(conn),
            })
            .manage(Pdf2zhSidecarState::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();

        cancel_pdf_translation_impl(
            CancelPdfTranslationInput {
                job_id: "job-running".to_string(),
            },
            app.handle().clone(),
            app.state::<AppDatabase>(),
            app.state::<Pdf2zhSidecarState>(),
        )
        .unwrap();

        let database = app.state::<AppDatabase>();
        let conn = database.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT status, phase, error, mono_pdf_path, finished_at
                 FROM pdf_translation_jobs WHERE id = 'job-running'",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(row.0, "canceled");
        assert_eq!(row.1, "canceled");
        assert_eq!(row.2, "");
        assert_eq!(row.3, "/tmp/partial.pdf");
        assert!(row.4 > 0);
    }

    #[test]
    fn running_and_failed_updates_do_not_override_canceled_jobs() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               started_at INTEGER NOT NULL DEFAULT 0,
               updated_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs (id, status, phase)
             VALUES ('job-queued', 'queued', 'queued'),
                    ('job-canceled', 'canceled', 'canceled'),
                    ('job-succeeded', 'succeeded', 'finished')",
            [],
        )
        .unwrap();

        assert!(mark_job_running_in_conn(&conn, "job-queued").unwrap());
        assert!(!mark_job_running_in_conn(&conn, "job-canceled").unwrap());

        assert!(!fail_job_in_conn(&conn, "job-canceled", "late error").unwrap());
        assert!(!fail_job_in_conn(&conn, "job-succeeded", "late error").unwrap());
        assert!(fail_job_in_conn(&conn, "job-queued", "active error").unwrap());

        let rows = conn
            .prepare("SELECT id, status, phase, error FROM pdf_translation_jobs ORDER BY id")
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(
            rows,
            vec![
                (
                    "job-canceled".to_string(),
                    "canceled".to_string(),
                    "canceled".to_string(),
                    "".to_string(),
                ),
                (
                    "job-queued".to_string(),
                    "failed".to_string(),
                    "failed".to_string(),
                    "active error".to_string(),
                ),
                (
                    "job-succeeded".to_string(),
                    "succeeded".to_string(),
                    "finished".to_string(),
                    "".to_string(),
                ),
            ]
        );
    }

    #[test]
    fn succeeded_job_update_does_not_override_canceled_status() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               cache_key TEXT NOT NULL UNIQUE,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               progress_percent INTEGER NOT NULL DEFAULT 0,
               current_page INTEGER NOT NULL DEFAULT 0,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               updated_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL
             );
             CREATE TABLE pdf_translation_events (
               id TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload_json TEXT NOT NULL DEFAULT '{}',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs (id, document_id, cache_key, status, phase)
             VALUES ('job-running', 'doc-1', 'cache-running', 'running', 'translating'),
                    ('job-canceled', 'doc-1', 'cache-canceled', 'canceled', 'canceled')",
            [],
        )
        .unwrap();

        let mono = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-finish-{}.pdf",
            stable_text_hash("finish-race-test")
        ));
        let event = test_sidecar_event(&mono);

        assert!(mark_job_succeeded_in_conn(&conn, "job-running", &event).unwrap());
        assert!(!mark_job_succeeded_in_conn(&conn, "job-canceled", &event).unwrap());

        let canceled_status: String = conn
            .query_row(
                "SELECT status FROM pdf_translation_jobs WHERE id = 'job-canceled'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(canceled_status, "canceled");

        let artifact_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pdf_translation_artifacts",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(artifact_count, 1);
    }

    #[test]
    fn repeated_partial_updates_keep_event_history_and_latest_artifact() {
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-partials-{}",
            stable_text_hash("partial-history-test")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let page_5 = dir.join("page-5.pdf");
        let page_6 = dir.join("page-6.pdf");
        fs::write(&page_5, b"%PDF-1.7\np5").unwrap();
        fs::write(&page_6, b"%PDF-1.7\np6").unwrap();

        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               progress_percent INTEGER NOT NULL DEFAULT 0,
               current_page INTEGER NOT NULL DEFAULT 0,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               updated_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE pdf_translation_events (
               id TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload_json TEXT NOT NULL DEFAULT '{}',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs (id, status, phase)
             VALUES ('job-1', 'running', 'translating')",
            [],
        )
        .unwrap();

        let event_5 = test_sidecar_event(&page_5);
        let event_6 = test_sidecar_event(&page_6);
        let database = AppDatabase {
            conn: std::sync::Mutex::new(conn),
        };
        let app = tauri::test::mock_builder()
            .manage(database)
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let database = app.state::<AppDatabase>();

        mark_job_partial(&database, "job-1", &event_5, "5").unwrap();
        mark_job_partial(&database, "job-1", &event_6, "6").unwrap();

        let conn = database.conn.lock().unwrap();
        let latest = conn
            .query_row(
                "SELECT status, phase, current_page, mono_pdf_path FROM pdf_translation_jobs WHERE id = 'job-1'",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(latest.0, "partial");
        assert_eq!(latest.1, "partial_ready");
        assert_eq!(latest.2, 6);
        assert_eq!(latest.3, page_6.to_string_lossy());

        let event_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pdf_translation_events WHERE job_id = 'job-1' AND event_type = 'artifact_ready'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(event_count, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn page_requests_only_append_new_pages_for_active_jobs() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               status TEXT NOT NULL
             );
             CREATE TABLE pdf_translation_events (
               id TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload_json TEXT NOT NULL DEFAULT '{}',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs (id, status)
             VALUES ('job-running', 'running'),
                    ('job-done', 'succeeded')",
            [],
        )
        .unwrap();

        let pages = normalize_page_list(&[2, 0, 2, 3, 4]);
        assert_eq!(pages, vec![2, 3, 4]);
        assert_eq!(
            insert_page_requests_in_conn(&conn, "job-running", &pages).unwrap(),
            vec![2, 3, 4]
        );
        assert_eq!(
            insert_page_requests_in_conn(&conn, "job-running", &[3, 5]).unwrap(),
            vec![5]
        );
        assert!(insert_page_requests_in_conn(&conn, "job-done", &[6])
            .unwrap()
            .is_empty());

        let database = AppDatabase {
            conn: std::sync::Mutex::new(conn),
        };
        let app = tauri::test::mock_builder()
            .manage(database)
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let mut completed = HashSet::new();
        completed.insert("3".to_string());
        assert_eq!(
            requested_partial_pages(&app.state::<AppDatabase>(), "job-running", &completed)
                .unwrap(),
            vec!["2".to_string(), "4".to_string(), "5".to_string()]
        );
    }

    #[test]
    fn cache_hit_returns_succeeded_output_when_artifacts_exist() {
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-cache-{}",
            stable_text_hash("cache-hit-test")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mono = dir.join("mono.pdf");
        let dual = dir.join("dual.pdf");
        fs::write(&mono, b"%PDF-1.7\nmono").unwrap();
        fs::write(&dual, b"%PDF-1.7\ndual").unwrap();

        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               target_lang TEXT NOT NULL,
               source_lang TEXT NOT NULL,
               provider_key TEXT NOT NULL,
               cache_key TEXT NOT NULL
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL,
               dual_pdf_path TEXT NOT NULL,
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs
             (id, document_id, target_lang, source_lang, provider_key, cache_key)
             VALUES ('job-1', 'doc-1', 'zh', 'en', 'google-web:abc', 'cache-1')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_artifacts
             (cache_key, document_id, job_id, mono_pdf_path, dual_pdf_path, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 1)",
            params![
                "cache-1",
                "doc-1",
                "job-1",
                mono.to_string_lossy(),
                dual.to_string_lossy()
            ],
        )
        .unwrap();

        let hit = read_cache_hit_from_conn(&conn, "doc-1", "cache-1")
            .unwrap()
            .expect("cache hit");
        assert_eq!(hit.status, "succeeded");
        assert_eq!(hit.phase, "cache_hit");
        assert!(hit.cached);
        assert_eq!(hit.artifact_scope, "full");

        fs::remove_file(&dual).unwrap();
        assert!(read_cache_hit_from_conn(&conn, "doc-1", "cache-1")
            .unwrap()
            .is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn clear_cache_collects_final_and_partial_job_artifacts() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT ''
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs
             (id, document_id, mono_pdf_path, dual_pdf_path)
             VALUES
             ('job-partial', 'doc-1', '/tmp/partial-mono.pdf', '/tmp/partial-dual.pdf'),
             ('job-other', 'doc-2', '/tmp/other-mono.pdf', '/tmp/other-dual.pdf')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_artifacts
             (cache_key, document_id, job_id, mono_pdf_path, dual_pdf_path, created_at)
             VALUES
             ('cache-1', 'doc-1', 'job-full', '/tmp/full-mono.pdf', '/tmp/full-dual.pdf', 1)",
            [],
        )
        .unwrap();

        let mut doc_paths = artifact_paths_for_clear_from_conn(&conn, Some("doc-1")).unwrap();
        doc_paths.sort();
        assert_eq!(
            doc_paths,
            vec![
                "/tmp/full-dual.pdf".to_string(),
                "/tmp/full-mono.pdf".to_string(),
                "/tmp/partial-dual.pdf".to_string(),
                "/tmp/partial-mono.pdf".to_string(),
            ]
        );

        let mut all_paths = artifact_paths_for_clear_from_conn(&conn, None).unwrap();
        all_paths.sort();
        assert!(all_paths.contains(&"/tmp/other-mono.pdf".to_string()));
        assert!(all_paths.contains(&"/tmp/full-mono.pdf".to_string()));
    }

    #[test]
    fn start_command_returns_cache_hit_without_launching_worker() {
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-command-cache-{}",
            stable_text_hash("command-cache-hit-test")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.pdf");
        let mono = dir.join("mono.pdf");
        let dual = dir.join("dual.pdf");
        fs::write(&source, b"%PDF-1.7\nsource").unwrap();
        fs::write(&mono, b"%PDF-1.7\nmono").unwrap();
        fs::write(&dual, b"%PDF-1.7\ndual").unwrap();

        let input = StartPdfTranslationInput {
            document_id: "doc-1".to_string(),
            target_lang: "zh".to_string(),
            source_lang: Some("en".to_string()),
            provider: Some("google-web".to_string()),
            artifact_mode: Some("both".to_string()),
            pages: None,
            priority_page: None,
            visible_pages: Vec::new(),
            nearby_page_radius: None,
            page_count: Some(1),
            force_refresh: None,
        };
        let source_info = file_source_identity(&source).unwrap();
        let engine = build_engine_payload(input.provider.as_deref());
        let provider_key = provider_key(&engine);
        let cache_key = build_cache_key(&source_info.hash, &input, &provider_key);

        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               target_lang TEXT NOT NULL,
               source_lang TEXT NOT NULL,
               provider_key TEXT NOT NULL,
               cache_key TEXT NOT NULL
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL,
               dual_pdf_path TEXT NOT NULL,
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_jobs
             (id, document_id, target_lang, source_lang, provider_key, cache_key)
             VALUES ('job-cache', 'doc-1', 'zh', 'en', ?1, ?2)",
            params![provider_key, cache_key],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO pdf_translation_artifacts
             (cache_key, document_id, job_id, mono_pdf_path, dual_pdf_path, created_at)
             VALUES (?1, 'doc-1', 'job-cache', ?2, ?3, 1)",
            params![cache_key, mono.to_string_lossy(), dual.to_string_lossy()],
        )
        .unwrap();

        let app = tauri::test::mock_builder()
            .manage(PdfRegistry::default())
            .manage(Pdf2zhSidecarState::default())
            .manage(AppDatabase {
                conn: std::sync::Mutex::new(conn),
            })
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        app.state::<PdfRegistry>()
            .paths
            .lock()
            .unwrap()
            .insert("doc-1".to_string(), source);

        let output = start_pdf_translation_impl(
            input,
            app.handle().clone(),
            app.state::<PdfRegistry>(),
            app.state::<AppDatabase>(),
        )
        .unwrap();

        assert_eq!(output.status, "succeeded");
        assert_eq!(output.phase, "cache_hit");
        assert!(output.cached);
        assert_eq!(output.mono_pdf_path, mono.to_string_lossy());
        assert_eq!(output.dual_pdf_path, dual.to_string_lossy());
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn start_command_runs_fake_sidecar_to_partial_and_full_success() {
        use std::os::unix::fs::PermissionsExt;

        let _env_lock = SIDECAR_ENV_LOCK
            .get_or_init(|| StdMutex::new(()))
            .lock()
            .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-start-job-{}",
            stable_text_hash("fake-sidecar-start-job-test")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake_pdf2zh_sidecar.py");
        fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json
import pathlib
import sys

request = json.loads(sys.stdin.readline())
payload = request["payload"]

def emit(event, **payload):
    print(json.dumps({"id": request["id"], "event": event, **payload}), flush=True)

if request.get("kind") == "preflight":
    emit("preflight_result", status="succeeded", phase="preflight_ready")
    sys.exit(0)

output_dir = pathlib.Path(payload["outputDir"])
output_dir.mkdir(parents=True, exist_ok=True)
pages = payload.get("pages") or "full"
mono = output_dir / f"mono-{pages}.pdf"
dual = output_dir / f"dual-{pages}.pdf"
mono.write_bytes(b"%PDF-1.7\nmono\n")
dual.write_bytes(b"%PDF-1.7\ndual\n")

emit("progress", status="running", phase="translating", progressPercent=20, currentPage=1)
emit(
    "finish",
    status="succeeded",
    phase="finished",
    progressPercent=100,
    monoPdfPath=str(mono),
    dualPdfPath=str(dual),
)
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();
        let _env_guard = EnvGuard::set("LUMENFOLIO_PDF2ZH_SIDECAR", &script);

        let source = dir.join("source.pdf");
        fs::write(&source, b"%PDF-1.7\nsource").unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               source_hash TEXT NOT NULL,
               source_size INTEGER NOT NULL DEFAULT 0,
               source_modified INTEGER NOT NULL DEFAULT 0,
               target_lang TEXT NOT NULL,
               source_lang TEXT NOT NULL DEFAULT 'en',
               provider_key TEXT NOT NULL,
               cache_key TEXT NOT NULL UNIQUE,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               progress_percent INTEGER NOT NULL DEFAULT 0,
               current_page INTEGER NOT NULL DEFAULT 0,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL,
               updated_at INTEGER NOT NULL,
               started_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL
             );
             CREATE TABLE pdf_translation_events (
               id TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload_json TEXT NOT NULL DEFAULT '{}',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        let app = tauri::test::mock_builder()
            .manage(PdfRegistry::default())
            .manage(Pdf2zhSidecarState::default())
            .manage(AppDatabase {
                conn: std::sync::Mutex::new(conn),
            })
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        app.state::<PdfRegistry>()
            .paths
            .lock()
            .unwrap()
            .insert("doc-1".to_string(), source);

        let output = start_pdf_translation_impl(
            StartPdfTranslationInput {
                document_id: "doc-1".to_string(),
                target_lang: "zh".to_string(),
                source_lang: Some("en".to_string()),
                provider: Some("google-web".to_string()),
                artifact_mode: Some("both".to_string()),
                pages: None,
                priority_page: Some(1),
                visible_pages: vec![1],
                nearby_page_radius: Some(0),
                page_count: Some(1),
                force_refresh: Some(true),
            },
            app.handle().clone(),
            app.state::<PdfRegistry>(),
            app.state::<AppDatabase>(),
        )
        .unwrap();
        assert_eq!(output.status, "queued");

        let database = app.state::<AppDatabase>();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let finished = loop {
            let job = read_pdf_translation_job(&database, &output.job_id)
                .unwrap()
                .expect("job row");
            if job.status == "succeeded" {
                break job;
            }
            if job.status == "failed" {
                panic!("fake sidecar job failed: {}", job.error);
            }
            if std::time::Instant::now() > deadline {
                panic!("timed out waiting for fake sidecar job, latest={job:?}");
            }
            thread::sleep(Duration::from_millis(50));
        };

        assert_eq!(finished.phase, "finished");
        assert_eq!(finished.progress_percent, 100);
        assert!(Path::new(&finished.mono_pdf_path).is_file());
        assert!(Path::new(&finished.dual_pdf_path).is_file());
        let conn = database.conn.lock().unwrap();
        let artifact_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pdf_translation_artifacts",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(artifact_count, 1);
        let partial_event_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pdf_translation_events WHERE event_type = 'artifact_ready'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(partial_event_count, 1);
        drop(conn);
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn start_command_falls_back_to_microsoft_when_google_preflight_fails() {
        use std::os::unix::fs::PermissionsExt;

        let _env_lock = SIDECAR_ENV_LOCK
            .get_or_init(|| StdMutex::new(()))
            .lock()
            .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-provider-fallback-{}",
            stable_text_hash("fake-sidecar-provider-fallback-test")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake_pdf2zh_sidecar_provider_fallback.py");
        fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json
import pathlib
import sys

request = json.loads(sys.stdin.readline())
payload = request["payload"]
provider = payload.get("engine", {}).get("provider")

def emit(event, **payload):
    print(json.dumps({"id": request["id"], "event": event, **payload}), flush=True)

if request.get("kind") == "preflight":
    if provider == "google-web":
        emit("error", status="failed", phase="failed", errorCode="SSLError", message="SSL EOF")
        sys.exit(0)
    emit("preflight_result", status="succeeded", phase="preflight_ready")
    sys.exit(0)

output_dir = pathlib.Path(payload["outputDir"])
output_dir.mkdir(parents=True, exist_ok=True)
mono = output_dir / "mono.pdf"
dual = output_dir / "dual.pdf"
mono.write_bytes(b"%PDF-1.7\nmono\n")
dual.write_bytes(b"%PDF-1.7\ndual\n")
emit(
    "finish",
    status="succeeded",
    phase="finished",
    progressPercent=100,
    monoPdfPath=str(mono),
    dualPdfPath=str(dual),
)
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();
        let _env_guard = EnvGuard::set("LUMENFOLIO_PDF2ZH_SIDECAR", &script);

        let source = dir.join("source.pdf");
        fs::write(&source, b"%PDF-1.7\nsource").unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               source_hash TEXT NOT NULL,
               source_size INTEGER NOT NULL DEFAULT 0,
               source_modified INTEGER NOT NULL DEFAULT 0,
               target_lang TEXT NOT NULL,
               source_lang TEXT NOT NULL DEFAULT 'en',
               provider_key TEXT NOT NULL,
               cache_key TEXT NOT NULL UNIQUE,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               progress_percent INTEGER NOT NULL DEFAULT 0,
               current_page INTEGER NOT NULL DEFAULT 0,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL,
               updated_at INTEGER NOT NULL,
               started_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL
             );
             CREATE TABLE pdf_translation_events (
               id TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload_json TEXT NOT NULL DEFAULT '{}',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        let app = tauri::test::mock_builder()
            .manage(PdfRegistry::default())
            .manage(Pdf2zhSidecarState::default())
            .manage(AppDatabase {
                conn: std::sync::Mutex::new(conn),
            })
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        app.state::<PdfRegistry>()
            .paths
            .lock()
            .unwrap()
            .insert("doc-1".to_string(), source);

        let output = start_pdf_translation_impl(
            StartPdfTranslationInput {
                document_id: "doc-1".to_string(),
                target_lang: "zh".to_string(),
                source_lang: Some("en".to_string()),
                provider: Some("google-web".to_string()),
                artifact_mode: Some("both".to_string()),
                pages: Some("1".to_string()),
                priority_page: Some(1),
                visible_pages: vec![1],
                nearby_page_radius: Some(0),
                page_count: Some(1),
                force_refresh: Some(true),
            },
            app.handle().clone(),
            app.state::<PdfRegistry>(),
            app.state::<AppDatabase>(),
        )
        .unwrap();
        assert_eq!(output.status, "queued");
        assert!(output.provider_key.starts_with("microsoft:"));

        let database = app.state::<AppDatabase>();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let job = read_pdf_translation_job(&database, &output.job_id)
                .unwrap()
                .expect("job row");
            if job.status == "succeeded" {
                assert!(job.provider_key.starts_with("microsoft:"));
                break;
            }
            if job.status == "failed" {
                panic!("fallback fake sidecar job failed: {}", job.error);
            }
            if std::time::Instant::now() > deadline {
                panic!("timed out waiting for fallback job, latest={job:?}");
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn cancel_command_kills_running_fake_sidecar_and_keeps_canceled_terminal_state() {
        use std::os::unix::fs::PermissionsExt;

        let _env_lock = SIDECAR_ENV_LOCK
            .get_or_init(|| StdMutex::new(()))
            .lock()
            .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-cancel-job-{}",
            stable_text_hash("fake-sidecar-cancel-job-test")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let started_marker = dir.join("sidecar-started");
        let marker_literal = serde_json::to_string(&started_marker.to_string_lossy()).unwrap();
        let script = dir.join("fake_pdf2zh_sidecar_hang.py");
        fs::write(
            &script,
            format!(
                r#"#!/usr/bin/env python3
import json
import pathlib
import sys
import time

request = json.loads(sys.stdin.readline())
marker = pathlib.Path({marker_literal})

def emit(event, **payload):
    print(json.dumps({{"id": request["id"], "event": event, **payload}}), flush=True)

if request.get("kind") == "preflight":
    emit("preflight_result", status="succeeded", phase="preflight_ready")
    sys.exit(0)

marker.write_text("started", encoding="utf-8")
emit("progress", status="running", phase="translating", progressPercent=10, currentPage=1)
while True:
    time.sleep(1)
"#
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();
        let _env_guard = EnvGuard::set("LUMENFOLIO_PDF2ZH_SIDECAR", &script);

        let source = dir.join("source.pdf");
        fs::write(&source, b"%PDF-1.7\nsource").unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               source_hash TEXT NOT NULL,
               source_size INTEGER NOT NULL DEFAULT 0,
               source_modified INTEGER NOT NULL DEFAULT 0,
               target_lang TEXT NOT NULL,
               source_lang TEXT NOT NULL DEFAULT 'en',
               provider_key TEXT NOT NULL,
               cache_key TEXT NOT NULL UNIQUE,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               progress_percent INTEGER NOT NULL DEFAULT 0,
               current_page INTEGER NOT NULL DEFAULT 0,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL,
               updated_at INTEGER NOT NULL,
               started_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL
             );
             CREATE TABLE pdf_translation_events (
               id TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload_json TEXT NOT NULL DEFAULT '{}',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        let app = tauri::test::mock_builder()
            .manage(PdfRegistry::default())
            .manage(Pdf2zhSidecarState::default())
            .manage(AppDatabase {
                conn: std::sync::Mutex::new(conn),
            })
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        app.state::<PdfRegistry>()
            .paths
            .lock()
            .unwrap()
            .insert("doc-1".to_string(), source);

        let output = start_pdf_translation_impl(
            StartPdfTranslationInput {
                document_id: "doc-1".to_string(),
                target_lang: "zh".to_string(),
                source_lang: Some("en".to_string()),
                provider: Some("google-web".to_string()),
                artifact_mode: Some("both".to_string()),
                pages: None,
                priority_page: Some(1),
                visible_pages: vec![1],
                nearby_page_radius: Some(0),
                page_count: Some(1),
                force_refresh: Some(true),
            },
            app.handle().clone(),
            app.state::<PdfRegistry>(),
            app.state::<AppDatabase>(),
        )
        .unwrap();
        assert_eq!(output.status, "queued");

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !started_marker.exists() {
            if std::time::Instant::now() > deadline {
                panic!("timed out waiting for fake sidecar to start");
            }
            thread::sleep(Duration::from_millis(50));
        }

        cancel_pdf_translation_impl(
            CancelPdfTranslationInput {
                job_id: output.job_id.clone(),
            },
            app.handle().clone(),
            app.state::<AppDatabase>(),
            app.state::<Pdf2zhSidecarState>(),
        )
        .unwrap();

        let database = app.state::<AppDatabase>();
        let state = app.state::<Pdf2zhSidecarState>();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let job = read_pdf_translation_job(&database, &output.job_id)
                .unwrap()
                .expect("job row");
            if job.status == "canceled" {
                if state.try_acquire_job_slot("after-cancel").unwrap() {
                    state.release_job_slot("after-cancel");
                    assert_eq!(job.phase, "canceled");
                    assert!(job.error.is_empty());
                    break;
                }
            }
            if job.status == "failed" {
                panic!("canceled fake sidecar job became failed: {}", job.error);
            }
            if std::time::Instant::now() > deadline {
                panic!("timed out waiting for canceled job cleanup, latest={job:?}");
            }
            thread::sleep(Duration::from_millis(50));
        }

        let conn = database.conn.lock().unwrap();
        let artifact_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pdf_translation_artifacts",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(artifact_count, 0);
        drop(conn);
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn start_command_marks_job_failed_when_fake_sidecar_exits_without_terminal_event() {
        use std::os::unix::fs::PermissionsExt;

        let _env_lock = SIDECAR_ENV_LOCK
            .get_or_init(|| StdMutex::new(()))
            .lock()
            .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-crash-job-{}",
            stable_text_hash("fake-sidecar-crash-job-test")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake_pdf2zh_sidecar_crash.py");
        fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json
import sys

request = json.loads(sys.stdin.readline())
if request.get("kind") == "preflight":
    print(json.dumps({
        "id": request["id"],
        "event": "preflight_result",
        "status": "succeeded",
        "phase": "preflight_ready",
    }), flush=True)
    sys.exit(0)
print(json.dumps({
    "id": request["id"],
    "event": "progress",
    "status": "running",
    "phase": "translating",
    "progressPercent": 12,
    "currentPage": 1,
}), flush=True)
sys.exit(23)
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();
        let _env_guard = EnvGuard::set("LUMENFOLIO_PDF2ZH_SIDECAR", &script);

        let source = dir.join("source.pdf");
        fs::write(&source, b"%PDF-1.7\nsource").unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pdf_translation_jobs (
               id TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               source_hash TEXT NOT NULL,
               source_size INTEGER NOT NULL DEFAULT 0,
               source_modified INTEGER NOT NULL DEFAULT 0,
               target_lang TEXT NOT NULL,
               source_lang TEXT NOT NULL DEFAULT 'en',
               provider_key TEXT NOT NULL,
               cache_key TEXT NOT NULL UNIQUE,
               status TEXT NOT NULL,
               phase TEXT NOT NULL DEFAULT '',
               progress_percent INTEGER NOT NULL DEFAULT 0,
               current_page INTEGER NOT NULL DEFAULT 0,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               error TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL,
               updated_at INTEGER NOT NULL,
               started_at INTEGER NOT NULL DEFAULT 0,
               finished_at INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE pdf_translation_artifacts (
               cache_key TEXT PRIMARY KEY,
               document_id TEXT NOT NULL,
               job_id TEXT NOT NULL,
               mono_pdf_path TEXT NOT NULL DEFAULT '',
               dual_pdf_path TEXT NOT NULL DEFAULT '',
               created_at INTEGER NOT NULL
             );
             CREATE TABLE pdf_translation_events (
               id TEXT PRIMARY KEY,
               job_id TEXT NOT NULL,
               event_type TEXT NOT NULL,
               payload_json TEXT NOT NULL DEFAULT '{}',
               created_at INTEGER NOT NULL
             );",
        )
        .unwrap();
        let app = tauri::test::mock_builder()
            .manage(PdfRegistry::default())
            .manage(Pdf2zhSidecarState::default())
            .manage(AppDatabase {
                conn: std::sync::Mutex::new(conn),
            })
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        app.state::<PdfRegistry>()
            .paths
            .lock()
            .unwrap()
            .insert("doc-1".to_string(), source);

        let output = start_pdf_translation_impl(
            StartPdfTranslationInput {
                document_id: "doc-1".to_string(),
                target_lang: "zh".to_string(),
                source_lang: Some("en".to_string()),
                provider: Some("google-web".to_string()),
                artifact_mode: Some("both".to_string()),
                pages: None,
                priority_page: Some(1),
                visible_pages: vec![1],
                nearby_page_radius: Some(0),
                page_count: Some(1),
                force_refresh: Some(true),
            },
            app.handle().clone(),
            app.state::<PdfRegistry>(),
            app.state::<AppDatabase>(),
        )
        .unwrap();
        assert_eq!(output.status, "queued");

        let database = app.state::<AppDatabase>();
        let state = app.state::<Pdf2zhSidecarState>();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let job = read_pdf_translation_job(&database, &output.job_id)
                .unwrap()
                .expect("job row");
            if job.status == "failed" {
                assert_eq!(job.phase, "failed");
                assert!(job.error.contains("without a terminal event"));
                if state.try_acquire_job_slot("after-crash").unwrap() {
                    state.release_job_slot("after-crash");
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                panic!("timed out waiting for fake sidecar crash recovery, latest={job:?}");
            }
            thread::sleep(Duration::from_millis(50));
        }

        let conn = database.conn.lock().unwrap();
        let artifact_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pdf_translation_artifacts",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(artifact_count, 0);
        drop(conn);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validates_artifacts_stay_inside_job_dir_and_are_pdfs() {
        let root = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-artifact-{}",
            stable_text_hash("artifact-validation-test")
        ));
        let outside = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-outside-{}.pdf",
            stable_text_hash("artifact-validation-test")
        ));
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_file(&outside);
        fs::create_dir_all(&root).unwrap();
        let mono = root.join("mono.pdf");
        let text = root.join("mono.txt");
        fs::write(&mono, b"%PDF-1.7\nmono").unwrap();
        fs::write(&text, b"not a pdf").unwrap();
        fs::write(&outside, b"%PDF-1.7\noutside").unwrap();

        let mut event = test_sidecar_event(&mono);
        assert!(validate_finished_artifacts(&event, &root).is_ok());

        event.mono_pdf_path = outside.to_string_lossy().to_string();
        assert!(validate_finished_artifacts(&event, &root).is_err());

        event.mono_pdf_path = text.to_string_lossy().to_string();
        assert!(validate_finished_artifacts(&event, &root).is_err());

        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_file(&outside);
    }

    #[test]
    fn source_identity_changes_when_pdf_bytes_change() {
        let path = std::env::temp_dir().join(format!(
            "lumenfolio-pdf2zh-source-{}.pdf",
            stable_text_hash("source-identity-test")
        ));
        fs::write(&path, b"%PDF-1.7\none").unwrap();
        let first = file_source_identity(&path).unwrap();
        fs::write(&path, b"%PDF-1.7\ntwo").unwrap();
        let second = file_source_identity(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_ne!(first.hash, second.hash);
    }

    fn test_sidecar_event(mono_pdf_path: &Path) -> SidecarEvent {
        SidecarEvent {
            id: "job-1".to_string(),
            event: "finish".to_string(),
            status: "succeeded".to_string(),
            phase: "finished".to_string(),
            progress_percent: 100,
            current_page: None,
            retryable: false,
            error_code: String::new(),
            message: String::new(),
            sidecar_version: String::new(),
            pdf2zh_version: String::new(),
            babeldoc_version: String::new(),
            python: String::new(),
            mono_pdf_path: mono_pdf_path.to_string_lossy().to_string(),
            dual_pdf_path: String::new(),
            artifact_scope: "full".to_string(),
            artifact_pages: String::new(),
        }
    }
}
