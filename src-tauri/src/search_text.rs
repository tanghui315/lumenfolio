//! CJK-aware text preparation for the FTS5 indexes.
//!
//! FTS5's default `unicode61` tokenizer does not segment CJK: a whole run of Han
//! characters becomes ONE token. Indexing "企业知识库升级建设方案" and searching
//! "知识库" therefore matched nothing — only a query anchored at the start of the
//! run (`企业*`) ever hit — so Chinese retrieval was broken across every document
//! type, not just one format.
//!
//! Fix: index each CJK character as its own token (spaces inserted at write
//! time) and turn a CJK query into overlapping bigram phrases.
//!
//! `tokenize='trigram'` was the obvious alternative and is rejected on purpose:
//! it silently drops every query shorter than three characters, so "北京" and
//! "AI" both stop matching. That trades one bug for a subtler one. Here, text
//! containing no CJK is returned untouched, so English tokenization, BM25
//! ranking and prefix (`term*`) syntax stay byte-for-byte identical.

fn is_cjk(c: char) -> bool {
    matches!(u32::from(c),
        0x3400..=0x4DBF     // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF   // CJK Unified Ideographs
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
        | 0x3040..=0x309F   // Hiragana
        | 0x30A0..=0x30FF   // Katakana
        | 0xAC00..=0xD7AF   // Hangul syllables
    )
}

/// Prepare text for storage in an FTS table: every CJK character becomes its own
/// token. Text with no CJK is returned unchanged.
///
/// Only the FTS mirror stores this form. Everything the user sees (citation
/// quotes, chunk text) is read from the source tables, so the spacing can never
/// leak into the UI.
pub(crate) fn index_text(text: &str) -> String {
    if !text.chars().any(is_cjk) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + text.len() / 2);
    for c in text.chars() {
        if is_cjk(c) {
            if !out.ends_with(' ') {
                out.push(' ');
            }
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn quote(term: &str) -> String {
    format!("\"{}\"", term.replace('"', "\"\""))
}

/// Build an FTS5 MATCH expression for `query`, OR-ing the clauses as before.
///
/// A CJK run becomes overlapping bigram phrases ("知识库" → `"知 识" OR "识 库"`):
/// that reaches near-segmenter recall without shipping a dictionary, and BM25
/// still ranks a chunk matching more of them higher. Matching the whole run as
/// one phrase would be far too strict for a natural-language question.
pub(crate) fn match_query(query: &str) -> String {
    let mut clauses: Vec<String> = Vec::new();
    for term in query.split_whitespace() {
        let chars: Vec<char> = term.chars().collect();
        let mut index = 0;
        while index < chars.len() {
            if is_cjk(chars[index]) {
                let start = index;
                while index < chars.len() && is_cjk(chars[index]) {
                    index += 1;
                }
                let run = &chars[start..index];
                if run.len() == 1 {
                    clauses.push(quote(&run[0].to_string()));
                } else {
                    for pair in run.windows(2) {
                        clauses.push(quote(&format!("{} {}", pair[0], pair[1])));
                    }
                }
            } else {
                let start = index;
                while index < chars.len() && !is_cjk(chars[index]) {
                    index += 1;
                }
                let run: String = chars[start..index].iter().collect();
                let trimmed = run.trim();
                if !trimmed.is_empty() {
                    clauses.push(quote(trimmed));
                }
            }
        }
    }
    clauses.join(" OR ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};

    #[test]
    fn english_text_and_queries_are_untouched() {
        assert_eq!(index_text("AI For Better Code"), "AI For Better Code");
        // Identical to the previous quote-and-OR behavior.
        assert_eq!(
            match_query("hybrid retrieval"),
            "\"hybrid\" OR \"retrieval\""
        );
    }

    #[test]
    fn cjk_text_is_split_into_per_character_tokens() {
        assert_eq!(index_text("知识库"), " 知 识 库 ");
        // Mixed content keeps the ASCII run intact.
        assert_eq!(index_text("AI知识库"), "AI 知 识 库 ");
    }

    #[test]
    fn cjk_queries_become_overlapping_bigrams() {
        assert_eq!(match_query("知识库"), "\"知 识\" OR \"识 库\"");
        assert_eq!(match_query("库"), "\"库\"");
        // A mixed term yields one clause per run.
        assert_eq!(match_query("AI战略"), "\"AI\" OR \"战 略\"");
    }

    /// The real contract: text written through `index_text` must be findable by
    /// `match_query`, including a term buried in the middle of a Han run — the
    /// exact case the default tokenizer could not match.
    #[test]
    fn round_trips_through_a_real_fts_index() {
        let conn = Connection::open_in_memory().expect("open");
        conn.execute_batch("CREATE VIRTUAL TABLE t USING fts5(text);")
            .expect("schema");
        for line in [
            "企业知识库升级建设方案",
            "北京理工项目进展 AI For Better Code",
        ] {
            conn.execute(
                "INSERT INTO t (text) VALUES (?1)",
                params![index_text(line)],
            )
            .expect("insert");
        }

        let hits = |q: &str| -> i64 {
            conn.query_row(
                "SELECT COUNT(*) FROM t WHERE t MATCH ?1",
                params![match_query(q)],
                |row| row.get(0),
            )
            .expect("match")
        };

        assert_eq!(hits("知识库"), 1, "mid-run CJK term must match");
        assert_eq!(hits("建设方案"), 1);
        assert_eq!(
            hits("北京"),
            1,
            "two-character terms must match (trigram cannot)"
        );
        assert_eq!(hits("AI"), 1, "two-letter ASCII must still match");
        assert_eq!(hits("Better"), 1);
        assert_eq!(hits("完全不存在的词"), 0);
    }
}
