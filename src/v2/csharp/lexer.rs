//! C# lexer for the Add-Type dialect.
//!
//! [`cs_lex`] runs over the raw C# region and returns significant tokens with
//! absolute spans. Whitespace and comments are skipped; string and character
//! literals are taken whole. The lexer never fails: anything it does not
//! recognize becomes an [`Unknown`](super::tokens::CsTokenKind::Unknown) token so the parser
//! can recover.

use super::tokens::{CsToken, CsTokenKind as K};
use crate::v2::span::Span;

/// Lexes a C# source region into significant tokens.
///
/// `base` is the absolute byte offset of `code` within the original file, so
/// the returned spans index the file directly. The final token is always
/// [`Eof`](super::tokens::CsTokenKind::Eof).
pub fn cs_lex(code: &str, base: usize) -> Vec<CsToken> {
    let b = code.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    let mut i = 0usize;

    let push = |out: &mut Vec<CsToken>, kind: K, start: usize, end: usize| {
        out.push(CsToken {
            kind,
            text: code[start..end].to_string(),
            span: Span::new(base + start, base + end),
        });
    };

    while i < n {
        let c = b[i];
        // Whitespace.
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Comments.
        if c == b'/' && i + 1 < n && b[i + 1] == b'/' {
            i += 2;
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < n && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }
        // Verbatim or interpolated string prefixes: @" $" $@" @$"
        if c == b'@' || c == b'$' {
            if let Some((prefix, verbatim)) = string_prefix_len(b, i) {
                let start = i;
                i += prefix;
                i = scan_string_body(b, n, i, verbatim);
                push(&mut out, K::Str, start, i);
                continue;
            }
        }
        // Identifiers (including @-escaped identifiers).
        if is_ident_start(c) || (c == b'@' && i + 1 < n && is_ident_start(b[i + 1])) {
            let start = i;
            if c == b'@' {
                i += 1;
            }
            while i < n && is_ident_continue(b[i]) {
                i += 1;
            }
            push(&mut out, K::Ident, start, i);
            continue;
        }
        // Plain string / char.
        if c == b'"' {
            let start = i;
            i = scan_string_body(b, n, i + 1, false);
            push(&mut out, K::Str, start, i);
            continue;
        }
        if c == b'\'' {
            let start = i;
            i = scan_char_body(b, n, i + 1);
            push(&mut out, K::Char, start, i);
            continue;
        }
        // Numbers.
        if c.is_ascii_digit() || (c == b'.' && i + 1 < n && b[i + 1].is_ascii_digit()) {
            let start = i;
            i = scan_number(b, n, i);
            push(&mut out, K::Number, start, i);
            continue;
        }
        // Multi-character punctuation.
        if c == b':' && i + 1 < n && b[i + 1] == b':' {
            push(&mut out, K::ColonColon, i, i + 2);
            i += 2;
            continue;
        }
        if c == b'=' && i + 1 < n && b[i + 1] == b'>' {
            push(&mut out, K::Arrow, i, i + 2);
            i += 2;
            continue;
        }
        // Single-character punctuation.
        let single = match c {
            b'{' => Some(K::LBrace),
            b'}' => Some(K::RBrace),
            b'(' => Some(K::LParen),
            b')' => Some(K::RParen),
            b'[' => Some(K::LBracket),
            b']' => Some(K::RBracket),
            b';' => Some(K::Semicolon),
            b',' => Some(K::Comma),
            b'.' => Some(K::Dot),
            b':' => Some(K::Colon),
            b'<' => Some(K::Lt),
            b'>' => Some(K::Gt),
            b'=' => Some(K::Assign),
            _ => None,
        };
        if let Some(kind) = single {
            push(&mut out, kind, i, i + 1);
            i += 1;
            continue;
        }
        // Other operator characters group into one Op token.
        if is_op_char(c) {
            let start = i;
            while i < n && is_op_char(b[i]) {
                i += 1;
            }
            push(&mut out, K::Op, start, i);
            continue;
        }
        // Anything else: a single Unknown byte (advance by a full UTF-8 char).
        let start = i;
        i += utf8_len(c);
        i = i.min(n);
        push(&mut out, K::Unknown, start, i);
    }

    out.push(CsToken {
        kind: K::Eof,
        text: String::new(),
        span: Span::new(base + n, base + n),
    });
    out
}

/// `@"`, `$"`, `$@"`, or `@$"` → (prefix length, verbatim?). Interpolated
/// strings are scanned as verbatim-ish only in that doubled quotes escape; for
/// the dialect this is close enough since literal bodies are not renamed.
fn string_prefix_len(b: &[u8], i: usize) -> Option<(usize, bool)> {
    match (b.get(i), b.get(i + 1), b.get(i + 2)) {
        (Some(b'@'), Some(b'$'), Some(b'"')) | (Some(b'$'), Some(b'@'), Some(b'"')) => {
            Some((3, true))
        }
        (Some(b'@'), Some(b'"'), _) => Some((2, true)),
        (Some(b'$'), Some(b'"'), _) => Some((2, false)),
        _ => None,
    }
}

/// Scans a string body starting just past the opening quote, returning the
/// index just past the closing quote. In verbatim mode `""` is an escaped
/// quote; otherwise `\` escapes the next byte.
fn scan_string_body(b: &[u8], n: usize, mut i: usize, verbatim: bool) -> usize {
    while i < n {
        if verbatim {
            if b[i] == b'"' {
                if i + 1 < n && b[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                return i + 1;
            }
            i += 1;
        } else {
            match b[i] {
                b'\\' => i += 2,
                b'"' => return i + 1,
                b'\n' => return i, // unterminated; stop at line end
                _ => i += 1,
            }
        }
    }
    n
}

/// Scans a char body starting just past the opening quote.
fn scan_char_body(b: &[u8], n: usize, mut i: usize) -> usize {
    while i < n {
        match b[i] {
            b'\\' => i += 2,
            b'\'' => return i + 1,
            b'\n' => return i,
            _ => i += 1,
        }
    }
    n
}

/// Scans a numeric literal. Deliberately loose: exact numeric form does not
/// affect refactoring, only that the run is consumed as one token.
fn scan_number(b: &[u8], n: usize, mut i: usize) -> usize {
    // Hex / binary prefix.
    if b[i] == b'0' && i + 1 < n && (b[i + 1] | 0x20 == b'x' || b[i + 1] | 0x20 == b'b') {
        i += 2;
        while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        return i;
    }
    while i < n {
        let c = b[i];
        if c.is_ascii_digit() || c == b'_' || c == b'.' {
            // A '.' is part of the number only if followed by a digit (so
            // `x.Method` and ranges are not swallowed).
            if c == b'.' && !(i + 1 < n && b[i + 1].is_ascii_digit()) {
                break;
            }
            i += 1;
        } else if c | 0x20 == b'e' && i + 1 < n && (b[i + 1] == b'+' || b[i + 1] == b'-') {
            i += 2;
        } else if matches!(c | 0x20, b'f' | b'd' | b'm' | b'l' | b'u') {
            i += 1;
        } else {
            break;
        }
    }
    i
}

fn is_ident_start(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphabetic() || c >= 0x80
}

fn is_ident_continue(c: u8) -> bool {
    c == b'_' || c.is_ascii_alphanumeric() || c >= 0x80
}

fn is_op_char(c: u8) -> bool {
    matches!(
        c,
        b'+' | b'-' | b'*' | b'/' | b'%' | b'&' | b'|' | b'^' | b'!' | b'~' | b'?' | b'@' | b'$'
    )
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(code: &str) -> Vec<K> {
        cs_lex(code, 0).iter().map(|t| t.kind).collect()
    }

    #[test]
    fn spans_are_offset_by_base() {
        let toks = cs_lex("class A", 100);
        assert_eq!(toks[0].text, "class");
        assert_eq!(toks[0].span.start, 100);
        assert_eq!(toks[1].text, "A");
        assert_eq!(toks[1].span.start, 106);
    }

    #[test]
    fn comments_and_strings_are_not_tokenized_inside() {
        // The identifier `secret` lives in a comment and a string; neither
        // should produce an Ident token.
        let toks = cs_lex("a // secret\nb /* secret */ \"secret\" c", 0);
        let idents: Vec<&str> = toks
            .iter()
            .filter(|t| t.kind == K::Ident)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(idents, vec!["a", "b", "c"]);
    }

    #[test]
    fn verbatim_string_handles_doubled_quote() {
        let toks = cs_lex("@\"a\"\"b\" x", 0);
        assert_eq!(toks[0].kind, K::Str);
        assert_eq!(toks[0].text, "@\"a\"\"b\"");
        assert_eq!(toks[1].text, "x");
    }

    #[test]
    fn dot_after_number_vs_member_access() {
        assert_eq!(kinds("1.5"), vec![K::Number, K::Eof]);
        assert_eq!(kinds("x.y"), vec![K::Ident, K::Dot, K::Ident, K::Eof]);
    }

    #[test]
    fn punctuation_classified() {
        assert_eq!(
            kinds("a::b => (c)"),
            vec![
                K::Ident,
                K::ColonColon,
                K::Ident,
                K::Arrow,
                K::LParen,
                K::Ident,
                K::RParen,
                K::Eof
            ]
        );
    }
}
