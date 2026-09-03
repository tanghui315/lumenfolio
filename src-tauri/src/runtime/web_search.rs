//! Web search + fetch for the chat agent.
//!
//! `web_search` uses Exa (api.exa.ai) when an API key is configured, otherwise a
//! keyless DuckDuckGo HTML fallback. `web_fetch` retrieves a URL and extracts
//! readable text. The agent's tool dispatch is synchronous but is called from
//! inside a tokio runtime, so all HTTP here runs on a dedicated scoped thread with
//! its own current-thread runtime (building/blocking a runtime inline would panic).

use std::time::Duration;

const HTTP_TIMEOUT: Duration = Duration::from_secs(20);
const USER_AGENT: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Lumenfolio/1.0";

/// One web result: the model weaves these into the answer as Markdown links.
#[derive(Clone, Debug)]
pub struct WebHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Search the public web. With an Exa key → Exa semantic search (richer, with
/// page text); without → a keyless DuckDuckGo fallback (titles + snippets only).
pub fn web_search(api_key: Option<&str>, query: &str, limit: usize) -> Result<Vec<WebHit>, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err("web_search requires a non-empty query".to_string());
    }
    match api_key.map(str::trim).filter(|k| !k.is_empty()) {
        Some(key) => exa_search(key, query, limit),
        None => duckduckgo_search(query, limit),
    }
}

/// Fetch a URL and return its readable text (HTML stripped), capped to `max_chars`.
pub fn web_fetch(url: &str, max_chars: usize) -> Result<String, String> {
    let url = url.trim();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("web_fetch requires an http(s) URL".to_string());
    }
    let url = url.to_string();
    let body = run_http(move || async move {
        let client = http_client()?;
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|err| format!("fetch failed: {err}"))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|err| format!("read failed: {err}"))?;
        if !status.is_success() {
            return Err(format!("fetch returned {status}"));
        }
        Ok(text)
    })?;
    let text = strip_html(&body);
    Ok(truncate_chars(&text, max_chars))
}

fn exa_search(api_key: &str, query: &str, limit: usize) -> Result<Vec<WebHit>, String> {
    let api_key = api_key.to_string();
    let query = query.to_string();
    let body = serde_json::json!({
        "query": query,
        "numResults": limit,
        "contents": { "text": { "maxCharacters": 1200 } }
    });
    let value: serde_json::Value = run_http(move || async move {
        let client = http_client()?;
        let resp = client
            .post("https://api.exa.ai/search")
            .header("x-api-key", &api_key)
            .json(&body)
            .send()
            .await
            .map_err(|err| format!("Exa request failed: {err}"))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|err| format!("Exa read failed: {err}"))?;
        if !status.is_success() {
            return Err(format!(
                "Exa returned {status}: {}",
                text.chars().take(300).collect::<String>()
            ));
        }
        serde_json::from_str::<serde_json::Value>(&text)
            .map_err(|err| format!("Exa decode failed: {err}"))
    })?;
    let results = value
        .get("results")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let hits = results
        .iter()
        .take(limit)
        .map(|r| WebHit {
            title: r["title"].as_str().unwrap_or("").trim().to_string(),
            url: r["url"].as_str().unwrap_or("").trim().to_string(),
            snippet: r["text"]
                .as_str()
                .or_else(|| r["snippet"].as_str())
                .unwrap_or("")
                .trim()
                .chars()
                .take(1200)
                .collect(),
        })
        .filter(|hit| !hit.url.is_empty())
        .collect();
    Ok(hits)
}

/// Keyless fallback: scrape the DuckDuckGo HTML endpoint. Best-effort — DDG may
/// rate-limit or change markup; on failure we surface a clear error so the model
/// (and the user) know to configure Exa for reliable search.
fn duckduckgo_search(query: &str, limit: usize) -> Result<Vec<WebHit>, String> {
    let query = query.to_string();
    let html = run_http(move || async move {
        let client = http_client()?;
        let resp = client
            .post("https://html.duckduckgo.com/html/")
            .form(&[("q", query.as_str())])
            .send()
            .await
            .map_err(|err| format!("DuckDuckGo request failed: {err}"))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|err| format!("DuckDuckGo read failed: {err}"))?;
        if !status.is_success() {
            return Err(format!("DuckDuckGo returned {status}"));
        }
        Ok(text)
    })?;
    let hits = parse_duckduckgo_html(&html, limit);
    if hits.is_empty() {
        return Err(
            "No web results (DuckDuckGo fallback returned nothing). Configure an Exa API key in Settings for reliable web search.".to_string(),
        );
    }
    Ok(hits)
}

fn parse_duckduckgo_html(html: &str, limit: usize) -> Vec<WebHit> {
    use regex::Regex;
    use std::sync::OnceLock;
    static LINK_RE: OnceLock<Regex> = OnceLock::new();
    static SNIPPET_RE: OnceLock<Regex> = OnceLock::new();
    let link_re = LINK_RE.get_or_init(|| {
        Regex::new(r#"(?s)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>"#).unwrap()
    });
    let snippet_re = SNIPPET_RE.get_or_init(|| {
        Regex::new(r#"(?s)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>"#).unwrap()
    });
    let snippets: Vec<String> = snippet_re
        .captures_iter(html)
        .map(|c| strip_html(&c[1]))
        .collect();
    link_re
        .captures_iter(html)
        .take(limit)
        .enumerate()
        .filter_map(|(index, caps)| {
            let url = normalize_ddg_url(&caps[1]);
            let title = strip_html(&caps[2]);
            if url.is_empty() {
                return None;
            }
            Some(WebHit {
                title,
                url,
                snippet: snippets.get(index).cloned().unwrap_or_default(),
            })
        })
        .collect()
}

/// DDG result hrefs are redirects like `//duckduckgo.com/l/?uddg=<encoded>&…`.
/// Pull the real destination out of the `uddg` param; otherwise normalize scheme.
fn normalize_ddg_url(href: &str) -> String {
    if let Some(idx) = href.find("uddg=") {
        let rest = &href[idx + 5..];
        let encoded = rest.split('&').next().unwrap_or("");
        return percent_decode(encoded);
    }
    if let Some(stripped) = href.strip_prefix("//") {
        return format!("https://{stripped}");
    }
    href.to_string()
}

/// Minimal percent-decoder (avoids pulling in a urlencoding crate).
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Strip HTML to plain text: drop script/style blocks and tags, decode the common
/// entities, and collapse whitespace.
fn strip_html(html: &str) -> String {
    use regex::Regex;
    use std::sync::OnceLock;
    static BLOCK_RE: OnceLock<Regex> = OnceLock::new();
    static TAG_RE: OnceLock<Regex> = OnceLock::new();
    static WS_RE: OnceLock<Regex> = OnceLock::new();
    // No backreferences in the `regex` crate — list each block's full pattern.
    let block_re = BLOCK_RE.get_or_init(|| {
        Regex::new(
            r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<noscript[^>]*>.*?</noscript>|<head[^>]*>.*?</head>",
        )
        .unwrap()
    });
    let tag_re = TAG_RE.get_or_init(|| Regex::new(r"(?s)<[^>]+>").unwrap());
    let ws_re = WS_RE.get_or_init(|| Regex::new(r"\s+").unwrap());
    let without_blocks = block_re.replace_all(html, " ");
    let without_tags = tag_re.replace_all(&without_blocks, " ");
    let decoded = without_tags
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'");
    ws_re.replace_all(&decoded, " ").trim().to_string()
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect()
}

fn http_client() -> Result<reqwest::Client, String> {
    crate::net::client_builder()
        .timeout(HTTP_TIMEOUT)
        .user_agent(USER_AGENT)
        .build()
        .map_err(|err| format!("web client build failed: {err}"))
}

/// Run an async HTTP future to completion on a fresh thread with its own
/// current-thread runtime, so the synchronous tool dispatch can call it even when
/// invoked from within a tokio runtime.
fn run_http<F, Fut, T>(make_future: F) -> Result<T, String>
where
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = Result<T, String>>,
    T: Send,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|err| format!("web runtime build failed: {err}"))?;
                rt.block_on(make_future())
            })
            .join()
            .map_err(|_| "web request thread panicked".to_string())?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_html_removes_tags_scripts_and_entities() {
        let html = "<div>Hello <script>alert(1)</script><b>world</b> &amp; more&nbsp;text</div>";
        assert_eq!(strip_html(html), "Hello world & more text");
    }

    #[test]
    fn percent_decode_handles_encoded_url() {
        assert_eq!(
            percent_decode("https%3A%2F%2Fexample.com%2Fa%20b"),
            "https://example.com/a b"
        );
    }

    #[test]
    fn normalize_ddg_url_extracts_uddg_target() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Farxiv.org%2Fabs%2F1234&rut=xyz";
        assert_eq!(normalize_ddg_url(href), "https://arxiv.org/abs/1234");
    }

    #[test]
    fn parse_duckduckgo_html_pairs_links_and_snippets() {
        let html = r#"
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com">First <b>title</b></a>
            <a class="result__snippet">First snippet</a>
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fb.com">Second</a>
            <a class="result__snippet">Second snippet</a>
        "#;
        let hits = parse_duckduckgo_html(html, 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://a.com");
        assert_eq!(hits[0].title, "First title");
        assert_eq!(hits[0].snippet, "First snippet");
        assert_eq!(hits[1].url, "https://b.com");
    }
}
