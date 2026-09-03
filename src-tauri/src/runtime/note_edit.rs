//! Precise (partial) edits for authored Markdown sources.
//!
//! Design notes, cross-checked against three shipped coding agents:
//!
//! * **Exact match, with one narrow tolerance.** Claude Code does no fuzzy matching
//!   at all; opencode shipped a nine-strategy fuzzy chain, destroyed user code with
//!   it (their commit "prevent destructive edit matches"), and its V2 rewrite is back
//!   to exact-only. So: no whitespace collapsing, no indentation re-inference, no
//!   block anchors. Those are extra dangerous for Markdown, where leading whitespace
//!   is semantic (nested lists, code fences) and two trailing spaces are a hard line
//!   break.
//!
//! * **Except typography.** Codex folds dashes/curly quotes/NBSP to ASCII before
//!   matching, and Claude Code separately normalizes curly quotes because the model
//!   "can't output curly quotes". Two independent implementations converged here, and
//!   prose notes — especially CJK ones — are full of “ ” ‘ ’ — … and NBSP. This is the
//!   one tolerance worth having.
//!
//! * **A match must be unique.** Codex takes the opposite line (first match wins,
//!   ordering enforced by a monotonic cursor); that suits its patch envelope but is a
//!   poor fit for prose, which repeats itself far more than code ("## Notes", "- [ ]",
//!   blank lines). 0 matches and >1 matches are both errors here, never a guess.
//!
//! * **Resolve to the original slice.** A normalized hit maps back to the exact text
//!   as it appears in the note (Claude Code's `findActualString`). Everything
//!   downstream — the proposal, the UI, the apply step — then deals in exact strings
//!   only, so this matching logic never needs a second implementation in JS.

/// One requested replacement. `old_text` is what the model believes is in the note.
#[derive(Debug, Clone)]
pub struct NoteEdit {
    pub old_text: String,
    pub new_text: String,
}

/// A replacement whose `old_text` has been resolved to the note's real bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedNoteEdit {
    /// Verbatim slice of the note — safe for a downstream exact-match apply.
    pub old_text: String,
    pub new_text: String,
}

/// Fold typographic characters the model cannot reliably reproduce down to ASCII.
/// Every arm maps one char to one char, which is what lets a normalized hit be mapped
/// back to an exact char range in the original.
fn normalize_char(c: char) -> char {
    match c {
        // Hyphens, dashes, minus.
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => '-',
        // Single quotes / apostrophes.
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => '\'',
        // Double quotes.
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' => '"',
        // Odd spaces → plain space. NOT newlines: in Markdown a line break is
        // structure, so folding it would let an edit match across a paragraph
        // boundary it never meant to touch.
        '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
        other => other,
    }
}

fn normalize_chars(chars: &[char]) -> Vec<char> {
    chars.iter().copied().map(normalize_char).collect()
}

/// Char indices at which `needle` occurs in `haystack`.
fn find_all(haystack: &[char], needle: &[char]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    (0..=haystack.len() - needle.len())
        .filter(|&start| &haystack[start..start + needle.len()] == needle)
        .collect()
}

fn byte_offset(chars: &[char], char_index: usize) -> usize {
    chars[..char_index].iter().map(|c| c.len_utf8()).sum()
}

/// Resolve `old_text` to the exact substring of `body` that it refers to.
///
/// Exact match first, then a typography-normalized match. Either way the match must
/// be unique. Errors name the problem *and* echo the text back, because that is what
/// lets the model correct itself on the next call rather than guess again.
pub fn resolve_edit_target(body: &str, old_text: &str) -> Result<String, String> {
    if old_text.is_empty() {
        return Err(
            "oldText cannot be empty. Provide the exact text to replace, or use `content` for a full rewrite."
                .to_string(),
        );
    }
    let body_chars: Vec<char> = body.chars().collect();
    let needle_chars: Vec<char> = old_text.chars().collect();

    let exact = find_all(&body_chars, &needle_chars);
    match exact.len() {
        1 => return Ok(old_text.to_string()),
        n if n > 1 => {
            return Err(format!(
                "Found {n} matches for oldText, so the target is ambiguous. Include more surrounding lines to make it unique.\noldText:\n{old_text}"
            ))
        }
        _ => {}
    }

    // Fall back to typography-insensitive matching, then return the note's own text so
    // the note keeps its curly quotes / dashes.
    let normalized_body = normalize_chars(&body_chars);
    let normalized_needle = normalize_chars(&needle_chars);
    let hits = find_all(&normalized_body, &normalized_needle);
    match hits.len() {
        1 => {
            let start = hits[0];
            let end = start + needle_chars.len();
            Ok(body[byte_offset(&body_chars, start)..byte_offset(&body_chars, end)].to_string())
        }
        n if n > 1 => Err(format!(
            "Found {n} matches for oldText, so the target is ambiguous. Include more surrounding lines to make it unique.\noldText:\n{old_text}"
        )),
        _ => Err(format!(
            "Could not find oldText in the note. It must match the note's text exactly, including whitespace and line breaks. Call read_note_source and copy the text verbatim.\noldText:\n{old_text}"
        )),
    }
}

/// Resolve and apply every edit to `body`, returning the new text and the resolved
/// edits. Applied to an in-memory copy and validated as a set: if any edit fails,
/// the whole call fails and the caller is left with nothing half-applied.
pub fn apply_edits(
    body: &str,
    edits: &[NoteEdit],
) -> Result<(String, Vec<ResolvedNoteEdit>), String> {
    if edits.is_empty() {
        return Err("edits cannot be empty".to_string());
    }
    let mut current = body.to_string();
    let mut resolved: Vec<ResolvedNoteEdit> = Vec::new();

    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text == edit.new_text {
            return Err(format!(
                "Edit {} does nothing: oldText and newText are identical.",
                index + 1
            ));
        }
        // Cascade guard: edit N must not target text edit N-1 just inserted, which is
        // almost always the model losing track of what it already changed.
        if let Some(previous) = resolved
            .iter()
            .position(|prior| !prior.new_text.is_empty() && prior.new_text.contains(&edit.old_text))
        {
            return Err(format!(
                "Edit {} targets text that edit {} inserts. Edits are applied in order to the ORIGINAL note — re-read it and describe each change against the original text.",
                index + 1,
                previous + 1
            ));
        }
        let target = resolve_edit_target(&current, &edit.old_text)
            .map_err(|err| format!("Edit {}: {err}", index + 1))?;
        let start = current
            .find(&target)
            .ok_or_else(|| format!("Edit {}: resolved target vanished", index + 1))?;
        // Deleting a whole line should take its newline with it, or the note is left
        // with a stray blank line where the content used to be.
        let (start, end, replacement) = if edit.new_text.is_empty()
            && !target.ends_with('\n')
            && current[start + target.len()..].starts_with('\n')
        {
            (start, start + target.len() + 1, String::new())
        } else {
            (start, start + target.len(), edit.new_text.clone())
        };
        current = format!("{}{}{}", &current[..start], replacement, &current[end..]);
        resolved.push(ResolvedNoteEdit {
            old_text: target,
            new_text: edit.new_text.clone(),
        });
    }

    if current == body {
        return Err("The edits leave the note unchanged.".to_string());
    }
    Ok((current, resolved))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(old: &str, new: &str) -> NoteEdit {
        NoteEdit {
            old_text: old.to_string(),
            new_text: new.to_string(),
        }
    }

    #[test]
    fn exact_unique_match_resolves() {
        let body = "# Title\n\nalpha beta\n\ngamma\n";
        assert_eq!(
            resolve_edit_target(body, "alpha beta").unwrap(),
            "alpha beta"
        );
    }

    #[test]
    fn ambiguous_match_is_rejected_never_guessed() {
        // Prose repeats itself far more than code — this is the case codex's
        // first-match-wins would silently get wrong.
        let body = "- [ ] todo\n- [ ] todo\n";
        let err = resolve_edit_target(body, "- [ ] todo").unwrap_err();
        assert!(err.contains("Found 2 matches"), "{err}");
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn missing_match_echoes_the_text_back() {
        let body = "hello world\n";
        let err = resolve_edit_target(body, "goodbye world").unwrap_err();
        assert!(err.contains("Could not find oldText"), "{err}");
        // The echo is what lets the model self-correct instead of retrying blind.
        assert!(err.contains("goodbye world"), "{err}");
    }

    #[test]
    fn typography_is_tolerated_and_the_note_keeps_its_own() {
        // The model cannot reliably emit curly quotes / em dashes, so it sends ASCII.
        let body = "He said “hello” — really.\n";
        let resolved = resolve_edit_target(body, "He said \"hello\" - really.").unwrap();
        // Resolves to the note's real text, curly punctuation intact.
        assert_eq!(resolved, "He said “hello” — really.");
    }

    #[test]
    fn cjk_full_width_space_is_tolerated() {
        let body = "第一段　第二段\n";
        let resolved = resolve_edit_target(body, "第一段 第二段").unwrap();
        assert_eq!(resolved, "第一段　第二段");
    }

    #[test]
    fn markdown_hard_line_break_is_preserved() {
        // Two trailing spaces are a hard line break: matching must not be
        // whitespace-insensitive, or an edit could silently drop them.
        let body = "line one  \nline two\n";
        assert!(resolve_edit_target(body, "line one\nline two").is_err());
        assert_eq!(
            resolve_edit_target(body, "line one  \nline two").unwrap(),
            "line one  \nline two"
        );
    }

    #[test]
    fn newlines_are_not_folded_into_spaces() {
        // A paragraph break must never match a single space, or an edit could reach
        // across a boundary it never meant to touch.
        let body = "alpha\n\nbeta\n";
        assert!(resolve_edit_target(body, "alpha beta").is_err());
    }

    #[test]
    fn edits_apply_in_order_and_preserve_the_rest() {
        let body = "# Title\n\nold one\n\nkeep me\n\nold two\n";
        let (next, resolved) = apply_edits(
            body,
            &[edit("old one", "new one"), edit("old two", "new two")],
        )
        .unwrap();
        assert_eq!(next, "# Title\n\nnew one\n\nkeep me\n\nnew two\n");
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].old_text, "old one");
    }

    #[test]
    fn one_bad_edit_fails_the_whole_set() {
        let body = "alpha\n\nbeta\n";
        let err = apply_edits(body, &[edit("alpha", "ALPHA"), edit("nope", "x")]).unwrap_err();
        assert!(err.contains("Edit 2"), "{err}");
        // Nothing is returned, so the caller cannot half-apply.
    }

    #[test]
    fn deleting_a_line_takes_its_newline() {
        let body = "keep\ndrop me\ntail\n";
        let (next, _) = apply_edits(body, &[edit("drop me", "")]).unwrap();
        assert_eq!(next, "keep\ntail\n", "a stray blank line was left behind");
    }

    #[test]
    fn wikilinks_survive_a_precise_edit() {
        let body = "See [[Attention]] for context.\n";
        let (next, _) = apply_edits(body, &[edit("for context", "for the details")]).unwrap();
        assert_eq!(next, "See [[Attention]] for the details.\n");
    }

    #[test]
    fn identical_old_and_new_is_rejected() {
        let body = "alpha\n";
        assert!(apply_edits(body, &[edit("alpha", "alpha")]).is_err());
    }

    #[test]
    fn editing_text_a_previous_edit_inserted_is_rejected() {
        let body = "alpha\n";
        let err = apply_edits(body, &[edit("alpha", "beta"), edit("beta", "gamma")]).unwrap_err();
        assert!(err.contains("edit 1"), "{err}");
    }

    #[test]
    fn empty_old_text_is_rejected() {
        assert!(resolve_edit_target("alpha\n", "").is_err());
        assert!(apply_edits("alpha\n", &[edit("", "x")]).is_err());
    }
}
