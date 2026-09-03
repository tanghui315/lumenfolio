//! Shared lexicon helpers for table numbers, CJK digits, and Roman numerals.
//!
//! Historically these helpers were duplicated across `finalize.rs`,
//! `agent_judge.rs`, and `runtime/rag/mod.rs`. They now live here as the single
//! source of truth so that intent classification, evidence checks, and RAG
//! tools all agree on how to parse "Table 3", "表 6", "Table VI", and "第3表".

/// Extract a referenced table number from free-form question/quote text.
///
/// Supports the following forms (case insensitive):
/// - `Table N`, `Table VI`, `Table 一`
/// - `表 N`, `表 六`, `表格 N`, `表格三`
/// - `第 N 表`, `第三表`, `第十二表`
pub fn requested_table_number(value: &str) -> Option<String> {
    let normalized = value.to_lowercase();

    for marker in ["table", "表格", "表"] {
        for (index, _) in normalized.match_indices(marker) {
            let rest = &normalized[index + marker.len()..];
            if let Some(number) = leading_reference_number(rest) {
                return Some(number);
            }
        }
    }

    let chars: Vec<char> = normalized.chars().collect();
    for (index, ch) in chars.iter().enumerate() {
        if *ch != '第' {
            continue;
        }
        let mut cursor = index + 1;
        while cursor < chars.len() && chars[cursor].is_whitespace() {
            cursor += 1;
        }
        let digit_start = cursor;
        while cursor < chars.len() && chars[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor > digit_start && chars.get(cursor) == Some(&'表') {
            let digits: String = chars[digit_start..cursor].iter().collect();
            if let Ok(number) = digits.parse::<u32>() {
                if number > 0 {
                    return Some(number.to_string());
                }
            }
            continue;
        }
        let cjk_start = cursor;
        while cursor < chars.len() && is_cjk_number_char(chars[cursor]) {
            cursor += 1;
        }
        if cursor > cjk_start && chars.get(cursor) == Some(&'表') {
            let cjk: String = chars[cjk_start..cursor].iter().collect();
            if let Some(number) = parse_cjk_number(&cjk) {
                return Some(number.to_string());
            }
        }
    }
    None
}

/// Parse the leading reference number from `value` after stripping common
/// connector punctuation. Supports ASCII digits, CJK digits, and lowercase
/// Roman numerals.
pub fn leading_reference_number(value: &str) -> Option<String> {
    let trimmed = value.trim_start_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, ':' | '#' | '-' | '_' | '.' | '：')
    });
    let digits = trimmed
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if !digits.is_empty() {
        return digits
            .parse::<u32>()
            .ok()
            .filter(|number| *number > 0)
            .map(|number| number.to_string());
    }
    let cjk = trimmed
        .chars()
        .take_while(|ch| is_cjk_number_char(*ch))
        .collect::<String>();
    if let Some(number) = parse_cjk_number(&cjk) {
        return Some(number.to_string());
    }
    let roman = trimmed
        .chars()
        .take_while(|ch| matches!(ch, 'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'))
        .collect::<String>();
    if roman.is_empty()
        || trimmed[roman.len()..]
            .chars()
            .next()
            .is_some_and(|ch| ch.is_alphabetic())
    {
        None
    } else {
        roman_to_number(&roman).map(|number| number.to_string())
    }
}

pub fn is_cjk_number_char(ch: char) -> bool {
    matches!(
        ch,
        '零' | '〇' | '一' | '二' | '两' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十'
    )
}

pub fn parse_cjk_number(value: &str) -> Option<u32> {
    if value.is_empty() {
        return None;
    }
    let chars = value.chars().collect::<Vec<_>>();
    if let Some(ten_index) = chars.iter().position(|ch| *ch == '十') {
        let tens = if ten_index == 0 {
            1
        } else {
            cjk_digit(chars[ten_index - 1])?
        };
        let ones = match chars.get(ten_index + 1).copied() {
            Some(ch) => cjk_digit(ch)?,
            None => 0,
        };
        return Some(tens * 10 + ones);
    }
    if chars.len() == 1 {
        return cjk_digit(chars[0]);
    }
    None
}

pub fn cjk_digit(ch: char) -> Option<u32> {
    match ch {
        '零' | '〇' => Some(0),
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

pub fn roman_to_number(value: &str) -> Option<u32> {
    let mut total = 0_i32;
    let mut previous = 0_i32;
    for ch in value.chars().rev() {
        let current = match ch {
            'i' => 1,
            'v' => 5,
            'x' => 10,
            'l' => 50,
            'c' => 100,
            'd' => 500,
            'm' => 1000,
            _ => return None,
        };
        if current < previous {
            total -= current;
        } else {
            total += current;
            previous = current;
        }
    }
    (total > 0).then_some(total as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_form_a_matches() {
        assert_eq!(requested_table_number("Table 3 results"), Some("3".into()));
    }

    #[test]
    fn cjk_form_a_matches() {
        assert_eq!(requested_table_number("表 6 是什么"), Some("6".into()));
        assert_eq!(requested_table_number("表六解读一下"), Some("6".into()));
    }

    #[test]
    fn roman_form_a_matches() {
        assert_eq!(requested_table_number("Table VI latency"), Some("6".into()));
    }

    #[test]
    fn form_b_di_n_biao_ascii() {
        assert_eq!(
            requested_table_number("第3表里面 GLM-5 的分数"),
            Some("3".into())
        );
    }

    #[test]
    fn form_b_di_n_biao_cjk_two_digits() {
        assert_eq!(
            requested_table_number("第十二表里面的指标"),
            Some("12".into())
        );
    }

    #[test]
    fn no_table_marker_returns_none() {
        assert_eq!(requested_table_number("这个方法的表现如何？"), None);
    }

    #[test]
    fn cjk_digit_basics() {
        assert_eq!(cjk_digit('三'), Some(3));
        assert_eq!(cjk_digit('两'), Some(2));
        assert_eq!(cjk_digit('a'), None);
    }

    #[test]
    fn parse_cjk_number_basics() {
        assert_eq!(parse_cjk_number("六"), Some(6));
        assert_eq!(parse_cjk_number("十"), Some(10));
        assert_eq!(parse_cjk_number("十二"), Some(12));
        assert_eq!(parse_cjk_number("三十"), Some(30));
        assert_eq!(parse_cjk_number("三十五"), Some(35));
        assert_eq!(parse_cjk_number(""), None);
    }

    #[test]
    fn roman_to_number_basics() {
        assert_eq!(roman_to_number("i"), Some(1));
        assert_eq!(roman_to_number("vi"), Some(6));
        assert_eq!(roman_to_number("ix"), Some(9));
        assert_eq!(roman_to_number("xiv"), Some(14));
        assert_eq!(roman_to_number("z"), None);
    }
}
