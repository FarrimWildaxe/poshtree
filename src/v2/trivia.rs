//! Trivia: every byte of the source that is not a significant token.
//!
//! This is the load-bearing difference between `v1` and `v2`. The v1 lexer
//! emits `Newline` and `Comment` tokens and drops plain whitespace, and the
//! v1 unparser regenerates spacing from scratch. In v2 nothing is dropped:
//! whitespace, newlines, comments, and backtick line continuations are kept
//! verbatim and attached to the significant token next to them. A formatter
//! reads them to place comments; a codemod leaves them untouched.

use super::span::Span;

/// What kind of non-significant text a piece of trivia is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriviaKind {
    /// Spaces, tabs, and other non-newline whitespace.
    Whitespace,
    /// Exactly one line break: `\n`, `\r\n`, or a bare `\r`.
    Newline,
    /// A `# ...` comment, without the line break that ends it.
    LineComment,
    /// A `<# ... #>` comment, delimiters included.
    BlockComment,
    /// A backtick immediately followed by a line break.
    LineContinuation,
}

/// One run of non-significant text, kept byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trivia {
    pub kind: TriviaKind,
    /// The exact source text, e.g. `"  "`, `"\r\n"`, `"# fix me"`.
    pub text: String,
    pub span: Span,
}

impl Trivia {
    pub fn is_comment(&self) -> bool {
        matches!(
            self.kind,
            TriviaKind::LineComment | TriviaKind::BlockComment
        )
    }

    /// True for newlines and line continuations, the two ways a physical
    /// line can end inside trivia.
    pub fn is_line_break(&self) -> bool {
        matches!(
            self.kind,
            TriviaKind::Newline | TriviaKind::LineContinuation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_helpers() {
        let t = |kind, text: &str| Trivia {
            kind,
            text: text.to_string(),
            span: Span::new(0, text.len()),
        };
        assert!(t(TriviaKind::LineComment, "# hi").is_comment());
        assert!(t(TriviaKind::BlockComment, "<# hi #>").is_comment());
        assert!(!t(TriviaKind::Whitespace, "  ").is_comment());
        assert!(t(TriviaKind::Newline, "\r\n").is_line_break());
        assert!(t(TriviaKind::LineContinuation, "`\n").is_line_break());
        assert!(!t(TriviaKind::LineComment, "# hi").is_line_break());
    }
}
