//! C# tokens for the Add-Type dialect.
//!
//! The lexer discards trivia (whitespace and comments) and lexes each string
//! or character literal as a single token, so identifiers inside comments and
//! strings never appear as tokens. Every token carries an absolute source span
//! (already offset to the original file), so a token is a ready-made edit
//! target.

use crate::v2::span::Span;

/// The kind of a C# token. Identifier-like tokens are all [`Ident`]; the parser
/// classifies keywords by text, which also handles C#'s contextual keywords.
///
/// [`Ident`]: CsTokenKind::Ident
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsTokenKind {
    /// An identifier or keyword (keywords are recognized by the parser).
    Ident,
    /// A numeric literal.
    Number,
    /// A string literal: regular, `@"verbatim"`, or `$"interpolated"`.
    Str,
    /// A character literal.
    Char,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `::`
    ColonColon,
    /// `:`
    Colon,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `=`
    Assign,
    /// `=>`
    Arrow,
    /// Any other operator or punctuation.
    Op,
    /// End of input.
    Eof,
    /// A byte that did not start any known token.
    Unknown,
}

/// A C# token: its kind, exact source text, and absolute span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsToken {
    /// What kind of token this is.
    pub kind: CsTokenKind,
    /// The exact source text of the token.
    pub text: String,
    /// Absolute byte span in the original file.
    pub span: Span,
}

impl CsToken {
    /// Whether this token is an identifier whose text equals `kw` (a keyword
    /// check; C# is case-sensitive, so the comparison is exact).
    pub fn is_kw(&self, kw: &str) -> bool {
        self.kind == CsTokenKind::Ident && self.text == kw
    }
}
