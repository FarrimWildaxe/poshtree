//! The v2 token: a syntactic kind, the exact source text, a byte span, and
//! the trivia that surrounds it.
//!
//! Several token kinds were inherited from the former v1 lexer, but the
//! native token stream and payloads are different:
//!
//! * Newlines and comments are [`Trivia`], not significant tokens. Check
//!   [`Token::starts_line`] and [`Token::ends_line`] for unescaped line breaks.
//! * [`TokenKind::VerbatimArgs`] holds the raw remainder of a line after `--%`,
//!   where even `#` does not start a comment.
//! * `value` keeps the raw token spelling, including original casing. Decoded
//!   values are derived by the parser rather than stored on lexer tokens.

use super::span::Span;
use super::trivia::{Trivia, TriviaKind};
use std::fmt;

/// Syntactic category of a significant token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Literals
    /// `$x`, `${a b}`, `$env:PATH`, `$global:y`, `$_`, `$?`, splatted
    /// `@args` / `@$args`
    Variable,
    /// `1`, `0xFF`, `1.5`, `1kb`, `1e3`, `0b1010`
    Number,
    /// `'literal'`
    StringSq,
    /// `"interpolated $(1 + 1)"`
    StringDq,
    /// `@' ... '@`
    HereStringSq,
    /// `@" ... "@`
    HereStringDq,

    // Words
    /// Bareword / command name / argument (e.g. `Get-ChildItem`, `C:\tmp`,
    /// `*.txt`)
    Generic,
    /// `if`, `foreach`, `function`, `return`, ...
    Keyword,
    /// `-Path`, `-Force`, `-ErrorAction:`
    Parameter,
    /// `-eq`, `-and`, `+`, `=`, `-f`, `..`, `&&`, `?.`, `--%`
    Operator,

    // Punctuation / structure
    Pipe,
    Amp,
    Semicolon,
    Comma,
    Dot,
    DoubleColon,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    /// `$(`
    DollarParen,
    /// `@(`
    AtParen,
    /// `@{`
    AtBrace,
    /// `>`, `>>`, `2>&1`, `*>`, `<`
    Redirect,

    /// Everything after `--%` to the end of the line, byte-for-byte.
    VerbatimArgs,

    Eof,
    Unknown,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A significant token plus the trivia attached to it.
///
/// Attachment policy (Roslyn-style, chosen so a formatter can find a
/// trailing same-line comment without re-scanning):
///
/// * `leading`: everything between the previous token's trailing trivia and
///   this token: whitespace, blank lines, comments, block comments, line
///   continuations.
/// * `trailing`: at most spaces/tabs, then at most one `# line comment`,
///   then at most one newline, in that order. Never crosses a line break.
/// * The `Eof` token carries the file's final trivia as `leading`.
///
/// The invariant the whole module is built on: concatenating
/// `leading + value + trailing` over all tokens reproduces the input
/// byte-for-byte. [`super::reconstruct`] does exactly that and the test
/// suite enforces it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    /// Exact source text of the token itself, trivia excluded.
    pub value: String,
    /// Byte span of `value` in the original source.
    pub span: Span,
    pub leading: Vec<Trivia>,
    pub trailing: Vec<Trivia>,
}

impl Token {
    /// Span covering the token and all of its attached trivia.
    pub fn full_span(&self) -> Span {
        let start = self
            .leading
            .first()
            .map_or(self.span.start, |t| t.span.start);
        let end = self.trailing.last().map_or(self.span.end, |t| t.span.end);
        Span::new(start, end)
    }

    /// Appends `leading + value + trailing` to `out`, byte-for-byte.
    pub fn write_full(&self, out: &mut String) {
        for t in &self.leading {
            out.push_str(&t.text);
        }
        out.push_str(&self.value);
        for t in &self.trailing {
            out.push_str(&t.text);
        }
    }

    /// Comments attached in front of the token (a formatter's "leading
    /// comments" of the node that starts here).
    pub fn leading_comments(&self) -> impl Iterator<Item = &Trivia> {
        self.leading.iter().filter(|t| t.is_comment())
    }

    /// The `# comment` sitting on the same line after the token, if any.
    pub fn trailing_comment(&self) -> Option<&Trivia> {
        self.trailing.iter().find(|t| t.is_comment())
    }

    /// True when a real (unescaped) line break sits in this token's leading
    /// trivia, so the token opens a new physical line. A backtick
    /// continuation (`` ` `` then a newline) deliberately does not count: it
    /// joins the next line to this statement, which is the whole point of the
    /// continuation.
    pub fn starts_line(&self) -> bool {
        self.leading.iter().any(|t| t.kind == TriviaKind::Newline)
    }

    /// True when a real (unescaped) newline sits in this token's *trailing*
    /// trivia, so the next token opens a new physical line. The mirror of
    /// [`starts_line`](Self::starts_line): a newline between two tokens can be
    /// attached to either side, so to tell whether a line break sits here, check
    /// the previous token's `ends_line` as well as the next token's
    /// `starts_line`. A backtick continuation does not count, for the same
    /// reason it does not count for `starts_line`.
    pub fn ends_line(&self) -> bool {
        self.trailing.iter().any(|t| t.kind == TriviaKind::Newline)
    }

    /// Case-insensitive comparison of the raw token text, the way
    /// PowerShell compares keywords, command names, operators, and
    /// parameters.
    pub fn value_eq_ci(&self, other: &str) -> bool {
        self.value.eq_ignore_ascii_case(other)
    }
}

/// A recoverable problem found while lexing. The lexer never panics and
/// never stops: the affected text still lands in a token (or trivia) so the
/// byte-for-byte invariant holds even for malformed input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)
    }
}

impl std::error::Error for LexError {}

/// Named operators, compared case-insensitively, without the leading hyphen.
/// Base names only: the `c`/`i` case-prefixed comparison spellings (`ceq`,
/// `imatch`, ...) are derived in the crate-private `ops` module rather than
/// listed here.
pub const NAMED_OPERATORS: &[&str] = &[
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "like",
    "notlike",
    "match",
    "notmatch",
    "replace",
    "contains",
    "notcontains",
    "in",
    "notin",
    "is",
    "isnot",
    "as",
    "and",
    "or",
    "xor",
    "not",
    "band",
    "bor",
    "bxor",
    "bnot",
    "shl",
    "shr",
    "join",
    "split",
    "f",
];

/// Keywords, compared case-insensitively. Context sensitivity (e.g. `in`
/// only inside `foreach`) is the parser's job.
pub const KEYWORDS: &[&str] = &[
    "if",
    "elseif",
    "else",
    "switch",
    "while",
    "for",
    "foreach",
    "do",
    "until",
    "break",
    "continue",
    "function",
    "filter",
    "workflow",
    "return",
    "throw",
    "trap",
    "try",
    "catch",
    "finally",
    "param",
    "begin",
    "process",
    "end",
    "dynamicparam",
    "data",
    "class",
    "enum",
    "using",
    "in",
    "exit",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::trivia::TriviaKind;

    fn trivia(kind: TriviaKind, text: &str, start: usize) -> Trivia {
        Trivia {
            kind,
            text: text.to_string(),
            span: Span::new(start, start + text.len()),
        }
    }

    #[test]
    fn full_span_and_write_full() {
        let tok = Token {
            kind: TokenKind::Generic,
            value: "ls".to_string(),
            span: Span::new(2, 4),
            leading: vec![trivia(TriviaKind::Whitespace, "  ", 0)],
            trailing: vec![
                trivia(TriviaKind::Whitespace, " ", 4),
                trivia(TriviaKind::LineComment, "# x", 5),
                trivia(TriviaKind::Newline, "\n", 8),
            ],
        };
        assert_eq!(tok.full_span(), Span::new(0, 9));
        let mut s = String::new();
        tok.write_full(&mut s);
        assert_eq!(s, "  ls # x\n");
        assert_eq!(tok.trailing_comment().unwrap().text, "# x");
        assert!(!tok.starts_line());
        assert!(tok.value_eq_ci("LS"));
    }

    #[test]
    fn starts_line_via_leading_newline() {
        let tok = Token {
            kind: TokenKind::Generic,
            value: "x".to_string(),
            span: Span::new(1, 2),
            leading: vec![trivia(TriviaKind::Newline, "\n", 0)],
            trailing: vec![],
        };
        assert!(tok.starts_line());
        assert_eq!(tok.leading_comments().count(), 0);
    }
}
