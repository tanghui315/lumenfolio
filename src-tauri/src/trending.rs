//! Trending Papers: an optional online discovery feature. Fetches the Hugging
//! Face "trending" papers list and lets the user download one into a managed
//! "Trending Papers" workspace folder (which then behaves like any other folder:
//! indexed, reopenable, chat/notes/translation). local-first: nothing is fetched
//! unless the user opens the discovery view, and a PDF is downloaded only on an
//! explicit "add" action.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tauri::{Manager, State};

use crate::{documents, AppDatabase, PdfRegistry, WorkspaceRoot, WorkspaceRootSnapshot};

const DEFAULT_TRENDING_LIMIT: u32 = 30;

// ---- Hugging Face daily_papers API shapes ---------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct HfDailyPaperItem {
    paper: HfPaper,
    #[serde(default)]
    upvotes: i64,
    /// HF's auto-generated card thumbnail (present for almost every paper). The
    /// real source of the cards' preview images — NOT `paper.mediaUrls`.
    #[serde(default)]
    thumbnail: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct HfPaper {
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    authors: Vec<HfAuthor>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    media_urls: Vec<String>,
    // The date endpoint carries upvotes at the item level; the week/month
    // endpoints carry it here on the paper instead. map_item reads both.
    #[serde(default)]
    upvotes: i64,
}

#[derive(Deserialize)]
struct HfAuthor {
    #[serde(default)]
    name: String,
}

// ---- Output to the frontend ------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrendingPaper {
    arxiv_id: String,
    title: String,
    authors: Vec<String>,
    summary: String,
    upvotes: i64,
    published_at: String,
    thumbnail_url: Option<String>,
    hf_url: String,
    pdf_url: String,
}

fn map_item(item: HfDailyPaperItem) -> TrendingPaper {
    let id = item.paper.id.trim().to_string();
    // Upvotes live at the item level (date endpoint) or on the paper (week/month
    // endpoints); take whichever is present.
    let upvotes = item.upvotes.max(item.paper.upvotes);
    let authors = item
        .paper
        .authors
        .into_iter()
        .map(|author| author.name.trim().to_string())
        .filter(|name| !name.is_empty())
        .collect();
    // Prefer HF's auto-generated card thumbnail; fall back to a submitted media
    // URL only if it's actually an image (mediaUrls can hold a video).
    let thumbnail_url = item
        .thumbnail
        .map(|url| url.trim().to_string())
        .filter(|url| !url.is_empty())
        .or_else(|| {
            item.paper
                .media_urls
                .iter()
                .find(|url| is_image_url(url))
                .map(|url| url.trim().to_string())
        });
    TrendingPaper {
        title: item.paper.title.trim().to_string(),
        authors,
        summary: item.paper.summary.trim().to_string(),
        upvotes,
        published_at: item.paper.published_at.unwrap_or_default(),
        thumbnail_url,
        hf_url: format!("https://huggingface.co/papers/{id}"),
        pdf_url: format!("https://arxiv.org/pdf/{id}.pdf"),
        arxiv_id: id,
    }
}

fn is_image_url(url: &str) -> bool {
    let path = url.split('?').next().unwrap_or(url).to_lowercase();
    [".png", ".jpg", ".jpeg", ".webp", ".gif"]
        .iter()
        .any(|ext| path.ends_with(ext))
}

const TRENDING_FALLBACK_QUERY: &str = "sort=trending";

/// Build the HF API query for a period + caller-computed value (everything after
/// `?`, minus the limit which the caller appends). daily → `date=YYYY-MM-DD`,
/// weekly → `week=YYYY-Www`, monthly → `month=YYYY-MM`. NOTE: the scoped
/// (week/month/date) endpoints must NOT carry `sort=trending` — that flag makes
/// HF ignore the scope and return the global trending list. They're already
/// ordered by upvotes within the period. A missing/invalid value (or an
/// unrecognised period) falls back to the latest global trending list. The value
/// is sanitised to a conservative charset before it reaches the query string.
fn trending_query(period: Option<&str>, value: Option<&str>) -> String {
    let value = value
        .map(str::trim)
        .filter(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'));
    match (period, value) {
        (Some("daily"), Some(v)) => format!("date={v}"),
        (Some("weekly"), Some(v)) => format!("week={v}"),
        (Some("monthly"), Some(v)) => format!("month={v}"),
        _ => TRENDING_FALLBACK_QUERY.to_string(),
    }
}

async fn fetch_trending_query(
    client: &reqwest::Client,
    query: &str,
    limit: u32,
) -> Result<Vec<TrendingPaper>, String> {
    let url = format!("https://huggingface.co/api/daily_papers?{query}&limit={limit}");
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .header("User-Agent", "Lumenfolio")
        .send()
        .await
        .map_err(|err| format!("Trending request failed: {err}"))?;
    if !response.status().is_success() {
        return Err(format!("Trending request returned {}", response.status()));
    }
    let items = response
        .json::<Vec<HfDailyPaperItem>>()
        .await
        .map_err(|err| format!("Failed to decode trending response: {err}"))?;
    Ok(items
        .into_iter()
        .filter(|item| !item.paper.id.trim().is_empty())
        .map(map_item)
        .collect())
}

/// Fetch the HF trending papers (recency + upvotes). Online-only; returns an
/// error when offline so the frontend can show an offline state. `period` is
/// "daily" (default), "weekly", or "monthly"; `value` is the caller-computed
/// date scope (e.g. "2026-06-10" / "2026-W24" / "2026-06").
#[tauri::command]
pub(crate) async fn fetch_trending_papers(
    period: Option<String>,
    value: Option<String>,
    limit: Option<u32>,
    database: State<'_, AppDatabase>,
) -> Result<Vec<TrendingPaper>, String> {
    let limit = limit.unwrap_or(DEFAULT_TRENDING_LIMIT).clamp(1, 100);
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|err| format!("Failed to create trending client: {err}"))?;
    let query = trending_query(period.as_deref(), value.as_deref());
    let mut papers = fetch_trending_query(&client, &query, limit).await?;
    // A scoped list (date/week/month) can be empty at a period boundary — today's
    // daily not published yet, or the first hours of a new ISO week/month. Fall
    // back to the latest global trending so the view (and the agent's cache) is
    // never blank. (Skip when we already used the global query.)
    if papers.is_empty() && query != TRENDING_FALLBACK_QUERY {
        papers = fetch_trending_query(&client, TRENDING_FALLBACK_QUERY, limit).await?;
    }
    // Warm the cache the agent's list_trending_papers tool reads (best-effort:
    // never fail the user-facing fetch over a cache write).
    let cache_period = normalize_period(period.as_deref());
    if let Err(err) = cache_trending(&database, cache_period, &papers) {
        eprintln!("[trending] cache write failed: {err}");
    }
    Ok(papers)
}

/// Canonical period key for the cache ('daily' | 'weekly' | 'monthly').
fn normalize_period(period: Option<&str>) -> &'static str {
    match period {
        Some("weekly") => "weekly",
        Some("monthly") => "monthly",
        _ => "daily",
    }
}

fn cache_trending(
    database: &AppDatabase,
    period: &str,
    papers: &[TrendingPaper],
) -> Result<(), String> {
    let payload = serde_json::to_string(papers)
        .map_err(|err| format!("Failed to serialize trending cache: {err}"))?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let conn = database
        .conn
        .lock()
        .map_err(|_| "SQLite lock was poisoned".to_string())?;
    conn.execute(
        "INSERT INTO trending_cache (period, payload_json, fetched_at)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(period) DO UPDATE SET payload_json = excluded.payload_json, fetched_at = excluded.fetched_at",
        rusqlite::params![period, payload, now],
    )
    .map_err(|err| format!("Failed to write trending cache: {err}"))?;
    Ok(())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AddTrendingResult {
    /// The (re)scanned "Trending Papers" workspace root, so the frontend can
    /// upsert it into the sidebar.
    snapshot: WorkspaceRootSnapshot,
    /// The document id of the just-added paper, so the frontend can open it.
    document_id: String,
}

/// Download a trending paper's PDF into the managed "Trending Papers" folder (if
/// not already there), (re)scan + index that folder, and return the folder
/// snapshot plus the new document id.
#[tauri::command]
pub(crate) async fn add_trending_paper(
    arxiv_id: String,
    title: Option<String>,
    app: tauri::AppHandle,
    registry: State<'_, PdfRegistry>,
    database: State<'_, AppDatabase>,
) -> Result<AddTrendingResult, String> {
    let arxiv_id = arxiv_id.trim().to_string();
    if arxiv_id.is_empty() {
        return Err("No paper id was provided".to_string());
    }

    let dir = trending_dir(&app)?;

    // Dedup by arXiv id regardless of the title in the filename.
    let file_path = existing_paper_path(&dir, &arxiv_id)
        .unwrap_or_else(|| dir.join(trending_file_name(&arxiv_id, title.as_deref())));

    if !file_path.exists() {
        download_pdf(&format!("https://arxiv.org/pdf/{arxiv_id}.pdf"), &file_path).await?;
    }

    // Register just this paper additively — never a full-folder reconcile, which
    // would DELETE a previously-added paper that happens to be transiently
    // unscannable (cloud-synced/locked) at this moment. The doc id comes from the
    // same builder that creates it, so it always matches the snapshot.
    let workspace_root_id = documents::stable_path_id("root", &dir);
    let doc = documents::build_document_for_path(&file_path, &workspace_root_id)?
        .ok_or_else(|| "Failed to read the downloaded paper".to_string())?;
    let document_id = doc.id.clone();
    documents::additive_upsert_documents(&database, &workspace_root_id, &dir, &[doc])?;
    let snapshot_docs = documents::load_documents_for_root(&database, &workspace_root_id)?;
    documents::upsert_registry_paths(&registry, &snapshot_docs)?;

    Ok(AddTrendingResult {
        snapshot: WorkspaceRootSnapshot {
            root: WorkspaceRoot {
                id: workspace_root_id,
                path: dir.to_string_lossy().to_string(),
            },
            documents: snapshot_docs,
        },
        document_id,
    })
}

/// The managed folder: `<Documents>/Lumenfolio/Trending Papers/` (created on
/// demand). User-visible so PDFs are accessible in the file manager.
fn trending_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let base = app
        .path()
        .document_dir()
        .map_err(|err| format!("Failed to resolve the Documents folder: {err}"))?;
    let dir = base.join("Lumenfolio").join("Trending Papers");
    fs::create_dir_all(&dir)
        .map_err(|err| format!("Failed to create the Trending Papers folder: {err}"))?;
    Ok(dir)
}

/// An existing file for this arXiv id (named `<id>.pdf` or `<id> <title>.pdf`).
fn existing_paper_path(dir: &Path, arxiv_id: &str) -> Option<PathBuf> {
    let exact = format!("{arxiv_id}.pdf");
    let prefix = format!("{arxiv_id} ");
    fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == exact || name.starts_with(&prefix))
        })
}

fn trending_file_name(arxiv_id: &str, title: Option<&str>) -> String {
    match title.map(sanitize_title).filter(|slug| !slug.is_empty()) {
        Some(slug) => format!("{arxiv_id} {slug}.pdf"),
        None => format!("{arxiv_id}.pdf"),
    }
}

/// Filesystem-safe, readable slug for the title (alnum/space/-/_; collapsed).
fn sanitize_title(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch == ' ' || ch == '-' || ch == '_' {
                ch
            } else {
                ' '
            }
        })
        .collect();
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(80)
        .collect()
}

async fn download_pdf(url: &str, dest: &Path) -> Result<(), String> {
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|err| format!("Failed to create download client: {err}"))?;
    let response = client
        .get(url)
        .header("User-Agent", "Lumenfolio")
        .send()
        .await
        .map_err(|err| format!("PDF download failed: {err}"))?;
    if !response.status().is_success() {
        return Err(format!("PDF download returned {}", response.status()));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|err| format!("Failed to read the downloaded PDF: {err}"))?;
    if !bytes.starts_with(b"%PDF") {
        return Err(
            "The downloaded file is not a valid PDF (the paper may not be on arXiv yet)."
                .to_string(),
        );
    }
    fs::write(dest, &bytes)
        .map_err(|err| format!("Failed to save the PDF to {}: {err}", dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_hf_item_to_trending_paper() {
        let json = r#"{
            "paper": {
                "id": "2606.05515",
                "title": "  BRepCLIP  ",
                "summary": "An abstract.",
                "authors": [{"name": "Alice"}, {"name": " "}, {"name": "Bob"}],
                "publishedAt": "2026-06-03T00:00:00.000Z"
            },
            "upvotes": 42,
            "thumbnail": "https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2606.05515.png"
        }"#;
        let item: HfDailyPaperItem = serde_json::from_str(json).unwrap();
        let paper = map_item(item);
        assert_eq!(paper.arxiv_id, "2606.05515");
        assert_eq!(paper.title, "BRepCLIP");
        assert_eq!(paper.authors, vec!["Alice".to_string(), "Bob".to_string()]);
        assert_eq!(paper.upvotes, 42);
        assert_eq!(paper.hf_url, "https://huggingface.co/papers/2606.05515");
        assert_eq!(paper.pdf_url, "https://arxiv.org/pdf/2606.05515.pdf");
        // The top-level `thumbnail` is preferred over mediaUrls.
        assert_eq!(
            paper.thumbnail_url.as_deref(),
            Some("https://cdn-thumbnails.huggingface.co/social-thumbnails/papers/2606.05515.png")
        );
    }

    #[test]
    fn upvotes_fall_back_to_paper_level_for_week_month_endpoints() {
        // week/month responses omit the item-level upvotes and put it on the paper.
        let json = r#"{
            "paper": { "id": "2606.00001", "title": "T", "upvotes": 90 }
        }"#;
        let item: HfDailyPaperItem = serde_json::from_str(json).unwrap();
        assert_eq!(map_item(item).upvotes, 90);
    }

    #[test]
    fn trending_query_maps_period_and_sanitises_value() {
        assert_eq!(
            trending_query(Some("daily"), Some("2026-06-10")),
            "date=2026-06-10"
        );
        assert_eq!(
            trending_query(Some("weekly"), Some("2026-W24")),
            "week=2026-W24"
        );
        assert_eq!(
            trending_query(Some("monthly"), Some("2026-06")),
            "month=2026-06"
        );
        // No sort=trending on the scoped queries (it would override the scope).
        assert!(!trending_query(Some("weekly"), Some("2026-W24")).contains("sort"));
        // Missing value or unrecognised period → latest global trending.
        assert_eq!(trending_query(None, None), "sort=trending");
        assert_eq!(trending_query(Some("weekly"), None), "sort=trending");
        // A value with unexpected characters is rejected (no query injection).
        assert_eq!(
            trending_query(Some("weekly"), Some("2026-W24&x=1")),
            "sort=trending"
        );
    }

    #[test]
    fn thumbnail_falls_back_to_image_media_url_only() {
        // No top-level thumbnail; mediaUrls has a video then an image — pick the image.
        let json = r#"{
            "paper": {
                "id": "1",
                "title": "T",
                "mediaUrls": ["https://x/clip.mp4", "https://x/fig.jpg"]
            }
        }"#;
        let item: HfDailyPaperItem = serde_json::from_str(json).unwrap();
        assert_eq!(
            map_item(item).thumbnail_url.as_deref(),
            Some("https://x/fig.jpg")
        );

        // No thumbnail and only a video → no thumbnail (don't put a video in <img>).
        let json =
            r#"{ "paper": { "id": "2", "title": "T", "mediaUrls": ["https://x/clip.mp4"] } }"#;
        let item: HfDailyPaperItem = serde_json::from_str(json).unwrap();
        assert_eq!(map_item(item).thumbnail_url, None);
    }

    #[test]
    fn file_name_dedup_and_sanitize() {
        assert_eq!(
            trending_file_name("2606.05515", Some("Deep: Research/Models?")),
            "2606.05515 Deep Research Models.pdf"
        );
        assert_eq!(trending_file_name("2606.05515", None), "2606.05515.pdf");
        assert_eq!(
            trending_file_name("2606.05515", Some("   ")),
            "2606.05515.pdf"
        );
    }
}
