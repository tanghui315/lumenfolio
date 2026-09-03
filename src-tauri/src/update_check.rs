//! Lightweight "a newer version exists" check (notify-only — no auto-install).
//!
//! Compares the running app version (`CARGO_PKG_VERSION`) against the highest
//! semver tag published on the GitHub repo. The frontend shows a dismissible
//! banner and links out to the tags page; the user updates manually. This fits
//! the project's release flow (annotated tags, no signed installer artifacts),
//! and deliberately avoids the Tauri updater plugin's signing/hosting machinery.

use std::time::Duration;

use serde::{Deserialize, Serialize};

const REPO: &str = "tanghui315/lumenfolio";

#[derive(Deserialize)]
struct GhTag {
    name: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UpdateInfo {
    /// The running version (without a leading "v").
    current: String,
    /// The highest published version (without a leading "v"); equals `current`
    /// when no newer tag is found.
    latest: String,
    has_update: bool,
    /// Where to send the user to get the new version.
    url: String,
}

/// Parse a `vX.Y.Z` (or `X.Y.Z`) tag into a comparable tuple. A trailing
/// pre-release/build suffix on the patch (e.g. `1.2.3-beta`) is tolerated by
/// taking the leading digits. Returns None for non-semver tags.
fn parse_semver(raw: &str) -> Option<(u32, u32, u32)> {
    let s = raw.trim().trim_start_matches('v');
    let mut parts = s.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_raw = parts.next().unwrap_or("0");
    let patch_digits: String = patch_raw
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let patch = patch_digits.parse().unwrap_or(0);
    Some((major, minor, patch))
}

/// Check GitHub for a newer release tag. Online-only; returns an error when
/// offline (the frontend silently ignores it — no banner).
#[tauri::command]
pub(crate) async fn check_for_update() -> Result<UpdateInfo, String> {
    let current = env!("CARGO_PKG_VERSION").to_string();
    let url = format!("https://api.github.com/repos/{REPO}/tags?per_page=100");
    let client = crate::net::client_builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|err| format!("Failed to create update client: {err}"))?;
    let response = client
        .get(&url)
        // GitHub's API requires a User-Agent; unauthenticated is fine (60 req/hr).
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "Lumenfolio")
        .send()
        .await
        .map_err(|err| format!("Update check request failed: {err}"))?;
    if !response.status().is_success() {
        return Err(format!("Update check returned {}", response.status()));
    }
    let tags = response
        .json::<Vec<GhTag>>()
        .await
        .map_err(|err| format!("Failed to decode tags: {err}"))?;

    // Tags aren't guaranteed to be returned in version order — pick the max semver.
    let current_v = parse_semver(&current);
    let mut best: Option<((u32, u32, u32), String)> = None;
    for tag in &tags {
        if let Some(v) = parse_semver(&tag.name) {
            if best.as_ref().is_none_or(|(bv, _)| v > *bv) {
                best = Some((v, tag.name.clone()));
            }
        }
    }

    let (latest_display, has_update) = match (best, current_v) {
        (Some((latest_v, latest_name)), Some(cur)) => (
            latest_name.trim_start_matches('v').to_string(),
            latest_v > cur,
        ),
        _ => (current.clone(), false),
    };

    Ok(UpdateInfo {
        current,
        latest: latest_display,
        has_update,
        url: format!("https://github.com/{REPO}/tags"),
    })
}
