//! Token definitions for the PowerShell lexer.
//!
//! PowerShell tokenization is context-sensitive (command mode vs expression
//! mode). The lexer produces a flat token stream and defers the hardest
//! disambiguation to the parser, but resolves the classic ambiguities locally:
//!
//! * `Get-ChildItem`   → a single GENERIC command-name token
//! * `-eq` / `-and`   → OPERATOR tokens  
//! * `-Path`           → PARAMETER tokens

use std::fmt;

/// Every distinct syntactic category produced by the lexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    // --- literals ---
    /// `$x`, `${a b}`, `$env:PATH`, `$global:y`, `$_`, `$?`
    Variable,
    /// `1`, `0xFF`, `1.5`, `1kb`, `1e3`
    Number,
    /// `'literal'`
    StringSq,
    /// `"interpolated"`
    StringDq,
    /// `@' ... '@`
    HereStringSq,
    /// `@" ... "@`
    HereStringDq,

    // --- words ---
    /// Bareword / command name / argument (e.g. `Get-ChildItem`, `foo`)
    Generic,
    /// `if`, `foreach`, `function`, `return`, …
    Keyword,
    /// `-Path`, `-Force`
    Parameter,
    /// `-eq`, `-and`, `+`, `=`, `-f`, `..`
    Operator,

    // --- punctuation / structure ---
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
    /// `>`, `>>`, `2>&1`, …
    Redirect,

    Newline,
    Comment,
    Eof,
    Unknown,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Named operators – compared case-insensitively, without the leading hyphen.
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
    "ceq",
    "cne",
    "cgt",
    "cge",
    "clt",
    "cle",
    "clike",
    "cnotlike",
    "cmatch",
    "cnotmatch",
    "creplace",
    "ccontains",
    "cnotcontains",
    "ieq",
    "ine",
    "igt",
    "ige",
    "ilt",
    "ile",
    "ilike",
    "inotlike",
    "imatch",
    "inotmatch",
    "ireplace",
];

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

/// A single token produced by the lexer.
#[derive(Debug, Clone)]
pub struct Token {
    pub ty: TokenType,
    /// Raw source text of the token.
    pub value: String,
    pub line: u32,
    pub col: u32,
    /// Byte offset into the source string (start of this token).
    pub pos: usize,
    /// Decoded / normalised payload (e.g. string contents without quotes).
    pub text: Option<String>,
    /// Variable scope extracted by the lexer (for `$env:PATH` → `"env"`).
    pub scope: Option<String>,
    /// `true` if this is a splatted variable (`@args`).
    pub splat: bool,
}

impl Token {
    pub fn new(ty: TokenType, value: impl Into<String>, line: u32, col: u32, pos: usize) -> Self {
        Token {
            ty,
            value: value.into(),
            line,
            col,
            pos,
            text: None,
            scope: None,
            splat: false,
        }
    }

    /// Convenience: the decoded text if present, otherwise the raw value.
    pub fn text_or_value(&self) -> &str {
        self.text.as_deref().unwrap_or(&self.value)
    }
}
