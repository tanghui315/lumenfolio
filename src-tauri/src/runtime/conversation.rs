//! Conversational fast-path detection.
//!
//! The retrieval pipeline treats every message as a document question and runs the
//! full agentic evidence loop. A bare greeting ("你好", "thanks") has nothing to
//! retrieve, so it burns the whole step budget and then refuses with "insufficient
//! evidence" — a bad experience for a message that just wanted a hello back.
//!
//! `is_smalltalk` is a HIGH-PRECISION, LOW-RECALL gate: it only fires on short,
//! unambiguous conversational messages. The error asymmetry drives the design —
//! misclassifying a real question as smalltalk (skipping retrieval on something the
//! user actually wanted answered) is far worse than missing a greeting (which just
//! falls back to the normal, if wasteful, path). So the rules err hard toward NOT
//! short-circuiting: any "real question" signal, or any open document / selection,
//! disqualifies the message.

/// Curated, unambiguous conversational phrases. These match the whole normalized
/// message and OVERRIDE the question-signal check — a few (e.g. "你是谁") contain
/// interrogative-looking characters but are plainly chit-chat about the assistant.
const SMALLTALK_PHRASES: &[&str] = &[
    // Chinese greetings / acknowledgements / farewells.
    "你好",
    "您好",
    "你好呀",
    "你好啊",
    "哈喽",
    "嗨",
    "hi",
    "在吗",
    "在么",
    "在不在",
    "早",
    "早安",
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
    "晚安",
    "你好吗",
    "最近好吗",
    "谢谢",
    "谢谢你",
    "多谢",
    "感谢",
    "感谢你",
    "辛苦了",
    "麻烦了",
    "好的",
    "好",
    "收到",
    "嗯",
    "嗯嗯",
    "ok",
    "okay",
    "好嘞",
    "行",
    "可以",
    "没事了",
    "算了",
    "再见",
    "拜拜",
    "回头见",
    "晚点聊",
    // Meta questions about the assistant itself (not the library).
    "你是谁",
    "你叫什么",
    "你叫什么名字",
    "你是什么",
    "你能做什么",
    "你会什么",
    "介绍一下你自己",
    "自我介绍",
    "你好厉害",
    // English.
    "hello",
    "hey",
    "yo",
    "hiya",
    "heya",
    "howdy",
    "hi there",
    "hello there",
    "thanks",
    "thank you",
    "thanks a lot",
    "thx",
    "ty",
    "cheers",
    "much appreciated",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "goodnight",
    "bye",
    "goodbye",
    "see you",
    "see ya",
    "later",
    "ok",
    "okay",
    "cool",
    "nice",
    "who are you",
    "what are you",
    "what can you do",
    "introduce yourself",
];

/// Greeting / thanks / farewell tokens. A message that STARTS WITH one of these and
/// carries no real-question signal is treated as smalltalk even if not an exact
/// phrase (e.g. "你好~~" after punctuation stripping, "thanks so much").
const GREETING_PREFIXES: &[&str] = &[
    "你好",
    "您好",
    "哈喽",
    "嗨",
    "早上好",
    "晚上好",
    "晚安",
    "谢谢",
    "多谢",
    "感谢",
    "再见",
    "拜拜",
    "hello",
    "hi ",
    "hey ",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
];

/// Substrings that mark a real request — an actual question or an instruction to act
/// on the library. Any of these disqualifies a message from the token-based rule
/// (the curated exact list still wins, so deliberate exceptions like "你是谁" work).
const QUESTION_SIGNALS: &[&str] = &[
    // Chinese interrogatives / imperatives.
    "什么",
    "为什么",
    "为何",
    "怎么",
    "怎样",
    "如何",
    "多少",
    "哪",
    "啥",
    "是不是",
    "能不能",
    "可不可以",
    "有没有",
    "总结",
    "概括",
    "解释",
    "说明",
    "分析",
    "翻译",
    "对比",
    "区别",
    "列出",
    "查",
    "找",
    "帮我",
    "帮忙",
    "讲了",
    "讲的",
    "介绍一下这",
    // English interrogatives / imperatives.
    "what ",
    "why",
    "how ",
    "which",
    "where",
    "when",
    "explain",
    "summar",
    "analyz",
    "translate",
    "compare",
    "list ",
    "find ",
    "search",
    "help me",
    "tell me about",
];

/// Whether `question` is a short, unambiguous conversational message that needs no
/// retrieval. `has_context` is true when a focus document is open or the user has a
/// text selection — in that case they are almost certainly asking about it, so we
/// never short-circuit.
pub fn is_smalltalk(question: &str, has_context: bool) -> bool {
    if has_context {
        return false;
    }
    let trimmed = question.trim();
    let char_count = trimmed.chars().count();
    if char_count == 0 || char_count > 24 {
        return false;
    }
    let norm = normalize(trimmed);
    if norm.is_empty() {
        return false;
    }
    // 1. Exact curated phrase — overrides everything.
    if SMALLTALK_PHRASES.contains(&norm.as_str()) {
        return true;
    }
    // 2. Greeting/thanks/farewell prefix with no request signal. The real guard is
    //    the question-signal check, not length: "你好，帮我总结这篇" opens with a
    //    greeting but carries 帮我/总结, so it is excluded and routed to retrieval.
    if starts_with_greeting(&norm) && !has_question_signal(&norm) {
        return true;
    }
    false
}

/// A canned, language-matched greeting used only when the direct LLM completion
/// fails (network / provider error) — so a smalltalk turn never dead-ends on an
/// error message. Mirrors the user's script (Han → Chinese, else English).
pub fn default_reply(question: &str) -> String {
    if question
        .chars()
        .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
    {
        "你好！我是你的知识库助手，有什么可以帮你的吗？".to_string()
    } else {
        "Hi! I'm your knowledge-base assistant — how can I help?".to_string()
    }
}

/// Lowercase and strip surrounding whitespace plus trailing conversational
/// punctuation / emoji-ish filler so "Hello!!!" and "你好～" normalize to the phrase.
fn normalize(raw: &str) -> String {
    let lowered = raw.to_lowercase();
    lowered
        .trim_matches(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '!' | '！'
                        | '。'
                        | '，'
                        | ','
                        | '.'
                        | '~'
                        | '～'
                        | '?'
                        | '？'
                        | '呀'
                        | '啊'
                        | '哦'
                        | '呢'
                        | '…'
                        | '\u{3000}'
                )
        })
        .to_string()
}

fn has_question_signal(norm: &str) -> bool {
    if norm.contains('?') || norm.contains('？') {
        return true;
    }
    QUESTION_SIGNALS.iter().any(|sig| norm.contains(sig))
}

fn starts_with_greeting(norm: &str) -> bool {
    GREETING_PREFIXES
        .iter()
        .any(|prefix| norm.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greetings_and_thanks_are_smalltalk() {
        for q in [
            "你好",
            "你好！",
            "您好",
            "你好~",
            "在吗",
            "谢谢",
            "谢谢你",
            "谢谢啦",
            "早上好",
            "晚安",
            "你是谁",
            "你能做什么",
            "好的",
            "hi",
            "Hello!",
            "hey",
            "thanks",
            "Thank you",
            "thanks so much",
            "good morning",
            "who are you",
        ] {
            assert!(is_smalltalk(q, false), "should be smalltalk: {q:?}");
        }
    }

    #[test]
    fn real_questions_are_not_smalltalk() {
        for q in [
            "总结这篇论文",
            "你好，帮我总结这篇",
            "什么是注意力机制",
            "DOPD 是什么意思",
            "这两篇有什么区别",
            "你好我想问一下这篇文章讲了啥",
            "解释一下 Table 3",
            "how does attention work",
            "what is DOPD",
            "summarize this paper",
            "compare these two papers",
            "帮我找一下关于蒸馏的论文",
        ] {
            assert!(!is_smalltalk(q, false), "should NOT be smalltalk: {q:?}");
        }
    }

    #[test]
    fn context_disables_the_gate() {
        // A greeting while reading a document / with a selection still runs the
        // normal path — the user may be about to ask about what they're looking at,
        // and we must never short-circuit real intent.
        assert!(is_smalltalk("你好", false));
        assert!(!is_smalltalk("你好", true));
    }

    #[test]
    fn long_messages_are_never_smalltalk() {
        // Even greeting-flavoured, anything past the length cap goes to retrieval.
        let long = "你好你好你好你好你好你好你好你好你好你好你好你好你好";
        assert!(!is_smalltalk(long, false));
    }

    #[test]
    fn empty_is_not_smalltalk() {
        assert!(!is_smalltalk("", false));
        assert!(!is_smalltalk("   ", false));
    }
}
