//! The v2 lexer: PowerShell source to trivia-bearing tokens, losslessly.
//!
//! The contract, in priority order:
//!
//! 1. **Byte fidelity.** Every byte of the input ends up in exactly one
//!    token's `leading` trivia, `value`, or `trailing` trivia, in source
//!    order. [`super::reconstruct`] therefore reproduces the input exactly,
//!    including on malformed input. The test suite enforces this.
//! 2. **No panics.** Broken input (unterminated strings, dangling escapes)
//!    produces a [`LexError`] and a best-effort token, never a crash.
//! 3. **v1-compatible kinds.** Classification matches the v1 lexer where it
//!    matters for a parser port: the [`KEYWORDS`] and [`NAMED_OPERATORS`]
//!    tables are equal to v1's (v2 keeps its own copies, locked to v1's by
//!    a test, so v2 has no compile-time dependency on v1 here), they decide
//!    `Keyword` vs `Generic` and `Operator` vs `Parameter`,
//!    `?.`/`?[`/`??`/`??=` are single operators, and the
//!    `.5`-vs-member-access call uses v1's previous-token rule.
//!
//! Like v1, this lexer does not model PowerShell's command/expression mode
//! split; genuinely context-sensitive calls are deferred to the parser.
//! Where the two lexers disagree, v2 is the more cohesive one (see the
//! differences list in [`crate::v2`]); where real PowerShell would still
//! have glued more bytes into one token, the v2 stream contains adjacent
//! tokens with no trivia between them (`a.span.end == b.span.start`), the
//! same byte-adjacency signal the v1 parser already uses when it joins
//! glued tokens. Kind assignment can be heuristic; spans and bytes never
//! are.

use super::span::Span;
use super::tokens::{LexError, Token, TokenKind, KEYWORDS, NAMED_OPERATORS};
use super::trivia::{Trivia, TriviaKind};

/// Result of lexing one source text.
#[derive(Debug, Clone)]
pub struct LexOutput {
    pub tokens: Vec<Token>,
    pub errors: Vec<LexError>,
}

/// Lexes `src` into a lossless token stream. The last token is always
/// [`TokenKind::Eof`]; it carries the file's final trivia.
///
/// Unlike v1's `tokenize`, this does **not** strip a UTF-8
/// BOM: a leading BOM survives as whitespace trivia on the first token, so
/// the byte-for-byte invariant holds. Strip it first with
/// [`crate::encoding::strip_bom`] if you do not want it in the stream.
pub fn lex(src: &str) -> LexOutput {
    Lexer::new(src).run()
}

fn is_keyword(word: &str) -> bool {
    KEYWORDS.contains(&word.to_lowercase().as_str())
}

fn is_named_operator(word: &str) -> bool {
    crate::ops::is_named_operator_word(word, NAMED_OPERATORS)
}

/// Non-newline blank characters that become [`TriviaKind::Whitespace`]. The
/// BOM is included so a file that starts with one still round-trips; v1
/// strips it instead.
fn is_blank(c: char) -> bool {
    (c.is_whitespace() && c != '\r' && c != '\n') || c == '\u{feff}'
}

/// Characters that keep a bareword going. The stop set is the delimiters
/// that always start a new token; `#`, `@`, `.`, `:`, `?`, `\`
/// and `-` continue a word, which is how `C:\tmp`, `user@host`, `*.txt`
/// and `Get-ChildItem` each stay one `Generic`. `=` stops a word so
/// `@{a=1}` lexes as four tokens; a `key=value` command argument becomes
/// adjacent tokens the parser may glue.
fn is_word_continue(c: char) -> bool {
    !c.is_whitespace()
        && !matches!(
            c,
            '|' | ';'
                | ','
                | '('
                | ')'
                | '{'
                | '}'
                | '['
                | ']'
                | '&'
                | '"'
                | '\''
                | '<'
                | '>'
                | '$'
                | '`'
                | '='
                | '\u{feff}'
        )
}

/// The v1 lexer decides `.5`-is-a-number by looking at the previous token:
/// after one of these kinds, a `.` is member access instead. Same set here.
fn allows_member(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Variable
            | TokenKind::RParen
            | TokenKind::RBracket
            | TokenKind::StringDq
            | TokenKind::StringSq
            | TokenKind::Generic
    )
}

/// Trivia that v1 would have emitted as a `Newline` or `Comment` token,
/// which resets its member-access context.
fn is_separator(t: &Trivia) -> bool {
    matches!(
        t.kind,
        TriviaKind::Newline | TriviaKind::LineComment | TriviaKind::BlockComment
    )
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    errors: Vec<LexError>,
    /// Set after lexing `--%`: the rest of the line is verbatim.
    verbatim_pending: bool,
    /// Kind of the last significant token, for the member-access rule.
    prev_kind: Option<TokenKind>,
    /// True when the previous token's trailing trivia ended its line (a
    /// newline or comment was attached there instead of to the next
    /// token's leading trivia).
    pending_separator: bool,
    /// True when a `.` at the current token would be member access. v1 gets
    /// this from `tokens.last()`, whose stream still contains `Newline` and
    /// `Comment` tokens; in v2 those are trivia, so the flag is computed
    /// from `prev_kind` plus the trivia on both sides of the gap.
    member_context: bool,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            pos: 0,
            errors: Vec::new(),
            verbatim_pending: false,
            prev_kind: None,
            pending_separator: false,
            member_context: false,
        }
    }

    // Low-level cursor

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn peek2(&self) -> Option<char> {
        self.rest().chars().nth(1)
    }

    fn bump(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    /// Consume a backtick and the character it escapes (if any). Shared by the
    /// double-quoted string, subexpression, and braced-variable scanners.
    fn skip_backtick_escape(&mut self) {
        self.bump(); // `
        if self.peek().is_some() {
            self.bump();
        }
    }

    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.errors.push(LexError {
            span,
            message: message.into(),
        });
    }

    fn trivia_from(&self, kind: TriviaKind, start: usize) -> Trivia {
        Trivia {
            kind,
            text: self.src[start..self.pos].to_string(),
            span: Span::new(start, self.pos),
        }
    }

    // Driver

    fn run(mut self) -> LexOutput {
        let mut tokens = Vec::new();
        loop {
            // After `--%`, the rest of the line is one verbatim token and
            // gets no leading trivia: a space or `#` there is argument text.
            if self.verbatim_pending {
                self.verbatim_pending = false;
                if !self.at_eof() && !matches!(self.peek(), Some('\r' | '\n')) {
                    let start = self.pos;
                    while let Some(c) = self.peek() {
                        if c == '\r' || c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                    let span = Span::new(start, self.pos);
                    let trailing = self.scan_trailing_trivia();
                    self.prev_kind = Some(TokenKind::VerbatimArgs);
                    self.pending_separator = trailing.iter().any(is_separator);
                    tokens.push(Token {
                        kind: TokenKind::VerbatimArgs,
                        value: self.src[span.start..span.end].to_string(),
                        span,
                        leading: Vec::new(),
                        trailing,
                    });
                    continue;
                }
            }

            let leading = self.scan_leading_trivia();
            if self.at_eof() {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    value: String::new(),
                    span: Span::new(self.pos, self.pos),
                    leading,
                    trailing: Vec::new(),
                });
                break;
            }

            // v1 sees Newline and Comment as tokens, and either one resets
            // its member-access context. Mirror that from the trivia on
            // both sides of the gap: a newline may sit in the previous
            // token's trailing trivia or in this token's leading trivia.
            let separated = self.pending_separator || leading.iter().any(is_separator);
            self.member_context = !separated && self.prev_kind.is_some_and(allows_member);

            let start = self.pos;
            let kind = self.scan_significant();
            debug_assert!(self.pos > start, "lexer failed to make progress");
            let span = Span::new(start, self.pos);
            let trailing = if self.verbatim_pending {
                Vec::new() // keep the bytes after `--%` out of trivia
            } else {
                self.scan_trailing_trivia()
            };
            self.prev_kind = Some(kind);
            self.pending_separator = trailing.iter().any(is_separator);
            tokens.push(Token {
                kind,
                value: self.src[span.start..span.end].to_string(),
                span,
                leading,
                trailing,
            });
        }
        LexOutput {
            tokens,
            errors: self.errors,
        }
    }

    // Trivia

    fn scan_leading_trivia(&mut self) -> Vec<Trivia> {
        let mut out = Vec::new();
        loop {
            let start = self.pos;
            match self.peek() {
                Some('\r') => {
                    self.bump();
                    if self.peek() == Some('\n') {
                        self.bump();
                    }
                    out.push(self.trivia_from(TriviaKind::Newline, start));
                }
                Some('\n') => {
                    self.bump();
                    out.push(self.trivia_from(TriviaKind::Newline, start));
                }
                Some(c) if is_blank(c) => {
                    while matches!(self.peek(), Some(c) if is_blank(c)) {
                        self.bump();
                    }
                    out.push(self.trivia_from(TriviaKind::Whitespace, start));
                }
                Some('#') => {
                    self.skip_line_comment();
                    out.push(self.trivia_from(TriviaKind::LineComment, start));
                }
                Some('<') if self.rest().starts_with("<#") => {
                    self.skip_block_comment();
                    out.push(self.trivia_from(TriviaKind::BlockComment, start));
                }
                Some('`') if matches!(self.peek2(), Some('\r' | '\n')) => {
                    self.bump(); // `
                    if self.peek() == Some('\r') {
                        self.bump();
                    }
                    if self.peek() == Some('\n') {
                        self.bump();
                    }
                    out.push(self.trivia_from(TriviaKind::LineContinuation, start));
                }
                _ => break,
            }
        }
        out
    }

    /// Trailing trivia: spaces/tabs, then at most one line comment, then at
    /// most one newline. Anything else (a block comment, a continuation, a
    /// second line) belongs to the next token's leading trivia.
    fn scan_trailing_trivia(&mut self) -> Vec<Trivia> {
        let mut out = Vec::new();
        let start = self.pos;
        while matches!(self.peek(), Some(c) if is_blank(c)) {
            self.bump();
        }
        if self.pos > start {
            out.push(self.trivia_from(TriviaKind::Whitespace, start));
        }
        if self.peek() == Some('#') {
            let start = self.pos;
            self.skip_line_comment();
            out.push(self.trivia_from(TriviaKind::LineComment, start));
        }
        let start = self.pos;
        match self.peek() {
            Some('\r') => {
                self.bump();
                if self.peek() == Some('\n') {
                    self.bump();
                }
                out.push(self.trivia_from(TriviaKind::Newline, start));
            }
            Some('\n') => {
                self.bump();
                out.push(self.trivia_from(TriviaKind::Newline, start));
            }
            _ => {}
        }
        out
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\r' || c == '\n' {
                break;
            }
            self.bump();
        }
    }

    fn skip_block_comment(&mut self) {
        let start = self.pos;
        self.bump(); // <
        self.bump(); // #
        loop {
            if self.rest().starts_with("#>") {
                self.bump();
                self.bump();
                return;
            }
            if self.at_eof() {
                self.error(Span::new(start, self.pos), "unterminated block comment");
                return;
            }
            self.bump();
        }
    }

    // Significant tokens

    fn scan_significant(&mut self) -> TokenKind {
        let c = self.peek().expect("checked by caller");
        match c {
            '$' => self.scan_dollar(),
            '@' => self.scan_at(),
            '\'' => {
                self.skip_sq_body();
                TokenKind::StringSq
            }
            '"' => {
                self.skip_dq_body();
                TokenKind::StringDq
            }
            '0'..='9' => self.scan_number_or_redirect(),
            '(' => self.one(TokenKind::LParen),
            ')' => self.one(TokenKind::RParen),
            '{' => self.one(TokenKind::LBrace),
            '}' => self.one(TokenKind::RBrace),
            '[' => self.one(TokenKind::LBracket),
            ']' => self.one(TokenKind::RBracket),
            ';' => self.one(TokenKind::Semicolon),
            ',' => self.one(TokenKind::Comma),
            '|' => {
                self.bump();
                if self.peek() == Some('|') {
                    self.bump();
                    TokenKind::Operator // `||` pipeline chain
                } else {
                    TokenKind::Pipe
                }
            }
            '&' => {
                self.bump();
                if self.peek() == Some('&') {
                    self.bump();
                    TokenKind::Operator // `&&` pipeline chain
                } else {
                    TokenKind::Amp
                }
            }
            '>' => {
                self.bump();
                if self.peek() == Some('>') {
                    self.bump();
                }
                TokenKind::Redirect
            }
            '<' => self.one(TokenKind::Redirect), // reserved; `<#` is trivia
            '.' => self.scan_dot(),
            ':' => {
                if self.rest().starts_with("::") {
                    self.bump();
                    self.bump();
                    TokenKind::DoubleColon
                } else {
                    self.one(TokenKind::Operator) // bare `:` as in v1
                }
            }
            '-' => self.scan_dash(),
            '+' => {
                self.bump();
                if matches!(self.peek(), Some('+' | '=')) {
                    self.bump();
                }
                TokenKind::Operator
            }
            '=' => self.one(TokenKind::Operator),
            '!' => self.one(TokenKind::Operator),
            '%' => {
                self.bump();
                if self.peek() == Some('=') {
                    self.bump();
                }
                TokenKind::Operator
            }
            '*' => self.scan_star(),
            '/' => self.scan_slash(),
            '?' => self.scan_question(),
            '`' => {
                // An escape at token start (`` `a ``); `` ` `` before a
                // newline was already taken as continuation trivia.
                let start = self.pos;
                self.scan_word();
                if self.pos == start {
                    self.bump();
                    self.error(Span::new(start, self.pos), "dangling escape character");
                    TokenKind::Unknown
                } else {
                    TokenKind::Generic
                }
            }
            _ => {
                // Letters, `_`, `\`, `~`, `^`, full Unicode: a bareword.
                let start = self.pos;
                self.scan_word();
                if is_keyword(&self.src[start..self.pos]) {
                    TokenKind::Keyword
                } else {
                    TokenKind::Generic
                }
            }
        }
    }

    fn one(&mut self, kind: TokenKind) -> TokenKind {
        self.bump();
        kind
    }

    /// Consumes bareword characters, honoring backtick escapes (`` a` b ``
    /// is one word). Stops before a backtick that precedes a newline or EOF.
    fn scan_word(&mut self) {
        loop {
            match self.peek() {
                Some('`') => match self.peek2() {
                    Some('\r' | '\n') | None => break,
                    Some(_) => {
                        self.bump();
                        self.bump();
                    }
                },
                Some(c) if is_word_continue(c) => self.bump(),
                _ => break,
            }
        }
    }

    fn scan_dollar(&mut self) -> TokenKind {
        let start = self.pos;
        self.bump(); // $
        match self.peek() {
            Some('(') => self.one(TokenKind::DollarParen),
            Some('{') => {
                self.bump();
                loop {
                    match self.peek() {
                        None => {
                            self.error(Span::new(start, self.pos), "unterminated braced variable");
                            break;
                        }
                        Some('}') => {
                            self.bump();
                            break;
                        }
                        Some('`') => self.skip_backtick_escape(),
                        Some(_) => self.bump(),
                    }
                }
                TokenKind::Variable
            }
            Some('$' | '?' | '^') => self.one(TokenKind::Variable),
            Some(c) if c.is_alphanumeric() || c == '_' => {
                loop {
                    while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
                        self.bump();
                    }
                    // scope or drive qualifier: `$env:PATH`, `$global:x`
                    if self.peek() == Some(':')
                        && matches!(self.peek2(), Some(c) if c.is_alphanumeric() || c == '_')
                    {
                        self.bump();
                    } else {
                        break;
                    }
                }
                TokenKind::Variable
            }
            _ => {
                self.error(Span::new(start, self.pos), "lone `$`");
                TokenKind::Unknown
            }
        }
    }

    fn scan_at(&mut self) -> TokenKind {
        let start = self.pos;
        self.bump(); // @
        match self.peek() {
            Some('(') => self.one(TokenKind::AtParen),
            Some('{') => self.one(TokenKind::AtBrace),
            Some(q @ ('\'' | '"')) => self.scan_here_string(start, q),
            Some('$') => {
                // `@$list` splatting, accepted by the v1 lexer too.
                self.bump();
                while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
                    self.bump();
                }
                TokenKind::Variable
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
                    self.bump();
                }
                TokenKind::Variable // splatting: @params
            }
            _ => {
                self.error(Span::new(start, self.pos), "lone `@`");
                TokenKind::Unknown
            }
        }
    }

    /// `@'`/`@"` here-string. The terminator quote must sit at the start of
    /// a line, so the scan looks for the next line break immediately
    /// followed by `'@`/`"@`; the body in between is verbatim and never
    /// inspected. Like v1, a lone `\r` counts as a line break.
    fn scan_here_string(&mut self, start: usize, quote: char) -> TokenKind {
        let kind = if quote == '\'' {
            TokenKind::HereStringSq
        } else {
            TokenKind::HereStringDq
        };
        self.bump(); // quote
        let lf = format!("\n{quote}@");
        let cr = format!("\r{quote}@");
        let rest = self.rest();
        let found = match (rest.find(&lf), rest.find(&cr)) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        match found {
            Some(idx) => self.pos += idx + lf.len(),
            None => {
                self.pos = self.src.len();
                self.error(Span::new(start, self.pos), "unterminated here-string");
            }
        }
        kind
    }

    /// Single-quoted string body: `''` is an escaped quote, nothing else is
    /// special, newlines are allowed.
    fn skip_sq_body(&mut self) {
        let start = self.pos;
        self.bump(); // '
        loop {
            match self.peek() {
                None => {
                    self.error(Span::new(start, self.pos), "unterminated string");
                    return;
                }
                Some('\'') => {
                    self.bump();
                    if self.peek() == Some('\'') {
                        self.bump(); // doubled quote, keep going
                    } else {
                        return;
                    }
                }
                Some(_) => self.bump(),
            }
        }
    }

    /// Double-quoted string body: backtick escapes, `""` doubling, and
    /// `$( ... )` subexpressions, which may themselves contain strings,
    /// comments, and here-strings. The whole thing stays one token, exactly
    /// as in v1; v2 just guarantees its bytes survive.
    fn skip_dq_body(&mut self) {
        let start = self.pos;
        self.bump(); // "
        loop {
            match self.peek() {
                None => {
                    self.error(Span::new(start, self.pos), "unterminated string");
                    return;
                }
                Some('`') => self.skip_backtick_escape(),
                Some('"') => {
                    self.bump();
                    if self.peek() == Some('"') {
                        self.bump(); // doubled quote, keep going
                    } else {
                        return;
                    }
                }
                Some('$') if self.peek2() == Some('(') => self.skip_subexpression(),
                Some(_) => self.bump(),
            }
        }
    }

    /// Balanced `$( ... )` inside a double-quoted string. Tracks paren depth
    /// while skipping nested strings, comments, and here-strings so that a
    /// `)` inside `"...)..."` or `# ...)` does not close the subexpression.
    fn skip_subexpression(&mut self) {
        let start = self.pos;
        self.bump(); // $
        self.bump(); // (
        let mut depth = 1usize;
        while depth > 0 {
            match self.peek() {
                None => {
                    self.error(
                        Span::new(start, self.pos),
                        "unterminated subexpression in string",
                    );
                    return;
                }
                Some('(') => {
                    self.bump();
                    depth += 1;
                }
                Some(')') => {
                    self.bump();
                    depth -= 1;
                }
                Some('\'') => self.skip_sq_body(),
                Some('"') => self.skip_dq_body(),
                Some('`') => self.skip_backtick_escape(),
                Some('<') if self.rest().starts_with("<#") => self.skip_block_comment(),
                Some('#') => self.skip_line_comment(),
                Some('@') if matches!(self.peek2(), Some('\'' | '"')) => {
                    let here_start = self.pos;
                    let quote = self.peek2().expect("just matched");
                    self.bump(); // @
                    self.scan_here_string(here_start, quote);
                }
                Some(_) => self.bump(),
            }
        }
    }

    /// First char is a digit: either a redirection (`2>`, `2>&1`) or a
    /// number. A number that runs straight into letters that are not a
    /// known suffix becomes a `Generic` bareword (`7zip`).
    fn scan_number_or_redirect(&mut self) -> TokenKind {
        let bytes = self.rest().as_bytes();
        let mut digits = 0;
        while bytes.get(digits).is_some_and(u8::is_ascii_digit) {
            digits += 1;
        }
        if bytes.get(digits) == Some(&b'>') {
            self.pos += digits;
            return self.finish_redirect();
        }

        let start = self.pos;
        let radix_without_digits = self.scan_number_core();
        self.scan_number_suffix();
        if matches!(self.peek(), Some(c) if c.is_alphabetic() || c == '_') {
            self.scan_word();
            return TokenKind::Generic;
        }
        if radix_without_digits {
            // `0x` / `0b` with nothing after the prefix. The token stays a
            // Number so the input still round-trips; the error records that no
            // PowerShell engine would accept the literal.
            self.error(Span::new(start, self.pos), "radix prefix with no digits");
        }
        TokenKind::Number
    }

    /// At `>` after a stream prefix (digits or `*`): `>`, `>>`, or `>&n`.
    fn finish_redirect(&mut self) -> TokenKind {
        self.bump(); // >
        if self.peek() == Some('>') {
            self.bump();
        } else if self.peek() == Some('&') && matches!(self.peek2(), Some(c) if c.is_ascii_digit())
        {
            self.bump();
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        TokenKind::Redirect
    }

    /// Returns true when a `0x`/`0b` radix prefix was consumed with no digits
    /// after it; the caller reports it if the token stays a Number.
    fn scan_number_core(&mut self) -> bool {
        let rest = self.rest();
        if rest.len() >= 2 && (rest.starts_with("0x") || rest.starts_with("0X")) {
            self.bump();
            self.bump();
            let digits_start = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_hexdigit()) {
                self.bump();
            }
            return self.pos == digits_start;
        }
        if rest.len() >= 2 && (rest.starts_with("0b") || rest.starts_with("0B")) {
            self.bump();
            self.bump();
            let digits_start = self.pos;
            // Stops at the first non-binary digit, leaving the remainder to
            // re-lex (so `0b1012` is `0b101` then `2`). PowerShell rejects such
            // a literal anyway; splitting it keeps the lexer lenient rather
            // than emitting a combined malformed-number token.
            while matches!(self.peek(), Some('0' | '1')) {
                self.bump();
            }
            return self.pos == digits_start;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if self.peek() == Some('.') && matches!(self.peek2(), Some(c) if c.is_ascii_digit()) {
            self.bump();
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
        }
        self.scan_exponent();
        false
    }

    /// `e3`, `E+5`, `e-2`; consumed only when at least one digit follows.
    fn scan_exponent(&mut self) {
        let bytes = self.rest().as_bytes();
        if matches!(bytes.first(), Some(b'e' | b'E')) {
            let mut i = 1;
            if matches!(bytes.get(i), Some(b'+' | b'-')) {
                i += 1;
            }
            if bytes.get(i).is_some_and(u8::is_ascii_digit) {
                for _ in 0..=i {
                    self.bump();
                }
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.bump();
                }
            }
        }
    }

    /// Type suffix (`l`, `d`, `u`, `ul`, ...) then multiplier (`kb`..`pb`),
    /// both case-insensitive, longest match first. A superset of v1, which
    /// stops at `l`/`d`/`u` plus the multipliers.
    fn scan_number_suffix(&mut self) {
        const TYPES: &[&str] = &["ul", "uy", "us", "u", "l", "y", "s", "n", "d"];
        const MULTIPLIERS: &[&str] = &["kb", "mb", "gb", "tb", "pb"];
        let take = |lexer: &mut Self, set: &[&str]| {
            for cand in set {
                let matched = lexer
                    .rest()
                    .get(..cand.len())
                    .is_some_and(|p| p.eq_ignore_ascii_case(cand));
                if matched {
                    lexer.pos += cand.len();
                    return;
                }
            }
        };
        take(self, TYPES);
        take(self, MULTIPLIERS);
    }

    fn scan_dot(&mut self) -> TokenKind {
        self.bump(); // .
        if self.peek() == Some('.') {
            self.bump();
            return TokenKind::Operator; // `..` range
        }
        if !self.member_context && matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            // `.5` is a number, but `$x.5` is member access; same
            // previous-token rule as the v1 lexer.
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.bump();
            }
            self.scan_exponent();
            self.scan_number_suffix();
            return TokenKind::Number;
        }
        TokenKind::Dot
    }

    fn scan_dash(&mut self) -> TokenKind {
        if self.rest().starts_with("--%") {
            self.bump();
            self.bump();
            self.bump();
            self.verbatim_pending = true;
            return TokenKind::Operator;
        }
        let start = self.pos;
        self.bump(); // -
        match self.peek() {
            Some('=') => self.one(TokenKind::Operator), // -=
            Some(c) if c.is_ascii_digit() => TokenKind::Operator, // minus; number follows
            Some(c) if c == '-' || c.is_alphabetic() || c == '_' || c == '?' => {
                self.scan_word();
                let value = &self.src[start..self.pos];
                let core = value.trim_start_matches('-');
                if core.is_empty() {
                    return TokenKind::Operator; // `--` decrement
                }
                let (core, had_colon) = match core.strip_suffix(':') {
                    Some(stripped) => (stripped, true),
                    None => (core, false),
                };
                if !had_colon && is_named_operator(core) {
                    TokenKind::Operator // -eq, -and, -f, -ireplace, ...
                } else {
                    TokenKind::Parameter // -Path, -Force, -ErrorAction:, --force
                }
            }
            _ => TokenKind::Operator, // bare minus
        }
    }

    fn scan_star(&mut self) -> TokenKind {
        if self.rest().as_bytes().get(1) == Some(&b'>') {
            self.bump(); // *
            return self.finish_redirect(); // *>, *>>, *>&1
        }
        if self.rest().as_bytes().get(1) == Some(&b'=') {
            self.bump();
            self.bump();
            return TokenKind::Operator; // *=
        }
        let start = self.pos;
        self.scan_word();
        if &self.src[start..self.pos] == "*" {
            TokenKind::Operator // multiplication / lone wildcard
        } else {
            TokenKind::Generic // `*.txt`
        }
    }

    fn scan_slash(&mut self) -> TokenKind {
        self.bump(); // /
        if self.peek() == Some('=') {
            self.bump();
            return TokenKind::Operator;
        }
        if matches!(self.peek(), Some(c) if c.is_alphabetic() || matches!(c, '_' | '.' | '/' | '\\' | '~'))
        {
            self.scan_word();
            return TokenKind::Generic; // `/usr/bin/env`
        }
        TokenKind::Operator // division
    }

    fn scan_question(&mut self) -> TokenKind {
        // Multi-char forms first, exactly the set v1 emits as operators.
        for op in ["??=", "??", "?.", "?["] {
            if self.rest().starts_with(op) {
                for _ in 0..op.len() {
                    self.bump();
                }
                return TokenKind::Operator;
            }
        }
        let start = self.pos;
        self.scan_word();
        if &self.src[start..self.pos] == "?" {
            TokenKind::Operator // ternary / Where-Object alias, as in v1
        } else {
            TokenKind::Generic // a wildcard word like `?abc`
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::reconstruct;

    fn lex_ok(src: &str) -> Vec<Token> {
        let out = lex(src);
        assert_eq!(reconstruct(&out.tokens), src, "round trip failed");
        assert!(out.errors.is_empty(), "unexpected errors: {:?}", out.errors);
        out.tokens
    }

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex_ok(src).iter().map(|t| t.kind).collect()
    }

    fn values(src: &str) -> Vec<String> {
        lex_ok(src)
            .iter()
            .filter(|t| t.kind != TokenKind::Eof)
            .map(|t| t.value.clone())
            .collect()
    }

    use TokenKind::*;

    #[test]
    fn pipeline_with_filter_block() {
        let src = "Get-ChildItem -Path C:\\tmp -Recurse | Where-Object { $_.Length -gt 1kb }";
        assert_eq!(
            kinds(src),
            vec![
                Generic, Parameter, Generic, Parameter, Pipe, Generic, LBrace, Variable, Dot,
                Generic, Operator, Number, RBrace, Eof
            ]
        );
    }

    #[test]
    fn keywords_and_assignment() {
        assert_eq!(
            kinds("if ($x -eq 1) { return } else { exit }"),
            vec![
                Keyword, LParen, Variable, Operator, Number, RParen, LBrace, Keyword, RBrace,
                Keyword, LBrace, Keyword, RBrace, Eof
            ]
        );
        assert_eq!(kinds("$a = 5"), vec![Variable, Operator, Number, Eof]);
    }

    #[test]
    fn dash_classification_uses_v1_tables() {
        assert_eq!(
            values("-eq -iMatch -Path -Path: --force -- -f -and"),
            vec!["-eq", "-iMatch", "-Path", "-Path:", "--force", "--", "-f", "-and"]
        );
        assert_eq!(
            kinds("-eq -iMatch -Path -Path: --force -- -f -and"),
            vec![
                Operator, Operator, Parameter, Parameter, Parameter, Operator, Operator, Operator,
                Eof
            ]
        );
        // `$i--` and subtraction
        assert_eq!(kinds("$i--"), vec![Variable, Operator, Eof]);
        assert_eq!(kinds("3 - 1"), vec![Number, Operator, Number, Eof]);
    }

    #[test]
    fn case_prefixed_operator_spellings_are_operators() {
        // Every prefixable comparison name in plain, `c`, and `i` spellings.
        // Derived, not sampled, so a new entry in the set is covered too.
        for base in crate::ops::CASE_PREFIXABLE {
            for prefix in ["", "c", "i", "C", "I"] {
                let src = format!("1 -{prefix}{base} 2");
                let out = lex(&src);
                assert_eq!(
                    out.tokens[1].kind, Operator,
                    "-{prefix}{base} should lex as an operator"
                );
            }
        }
        // The prefix does not apply outside the comparison set.
        for op in ["-cand", "-cor", "-cjoin", "-cis", "-cisnot", "-cshl"] {
            let out = lex(&format!("1 {op} 2"));
            assert_eq!(out.tokens[1].kind, Parameter, "{op} is not an operator");
        }
    }

    #[test]
    fn radix_prefix_without_digits_is_reported() {
        for src in ["0x", "0b", "$a = 0x"] {
            let out = lex(src);
            assert_eq!(out.errors.len(), 1, "{src}: {:?}", out.errors);
            assert!(out.errors[0].message.contains("radix"));
            // Lossless: the token text is preserved.
            assert_eq!(crate::v2::reconstruct(&out.tokens), src);
        }
        for src in ["0x1F", "0b1010", "0xFFL", "0x10kb", "0xZZ", "0"] {
            assert!(lex(src).errors.is_empty(), "{src} should not error");
        }
    }

    #[test]
    fn comments_become_trivia() {
        let toks = lex_ok("# leading\nls # trailing\n<# block #> pwd");
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Generic, Generic, Eof]
        );
        let ls = &toks[0];
        assert_eq!(ls.leading_comments().next().unwrap().text, "# leading");
        assert_eq!(ls.trailing_comment().unwrap().text, "# trailing");
        let pwd = &toks[1];
        assert_eq!(pwd.leading_comments().next().unwrap().text, "<# block #>");
    }

    #[test]
    fn crlf_and_blank_lines() {
        let src = "ls\r\n\r\n  pwd\r\n";
        let toks = lex_ok(src);
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Generic, Generic, Eof]
        );
        assert!(toks[1].starts_line());
    }

    #[test]
    fn here_strings() {
        let src = "@'\nliteral '@ not end\n'@\n$x = @\"\nhello $name\n\"@";
        assert_eq!(
            kinds(src),
            vec![HereStringSq, Variable, Operator, HereStringDq, Eof]
        );
        // a lone \r counts as a line break before the terminator, like v1
        let toks = lex_ok("@'\rbody\r'@");
        assert_eq!(toks[0].kind, HereStringSq);
    }

    #[test]
    fn dq_string_with_nested_subexpression() {
        // The `)` inside the inner string and the `#` comment must not
        // close the subexpression early.
        let src = "\"v: $( 'a)b' + \")(\" # )\n + 2 ) end\"";
        assert_eq!(kinds(src), vec![StringDq, Eof]);
        // doubled-quote escape
        assert_eq!(kinds("\"say \"\"hi\"\"\""), vec![StringDq, Eof]);
        assert_eq!(kinds("'it''s'"), vec![StringSq, Eof]);
    }

    #[test]
    fn verbatim_arguments() {
        let src = "ping --% -n 1 # still args\nnext";
        let toks = lex_ok(src);
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Generic, Operator, VerbatimArgs, Generic, Eof]
        );
        assert_eq!(toks[2].value, " -n 1 # still args");
        assert!(toks[2].trailing_comment().is_none());
    }

    #[test]
    fn redirections() {
        assert_eq!(
            kinds("cmd 2>&1 *> all.log >> out.txt"),
            vec![Generic, Redirect, Redirect, Generic, Redirect, Generic, Eof]
        );
        assert_eq!(
            values("cmd 2>&1 *> all.log >> out.txt"),
            vec!["cmd", "2>&1", "*>", "all.log", ">>", "out.txt"]
        );
    }

    #[test]
    fn backtick_escapes_and_continuation() {
        // escaped space keeps the word together
        assert_eq!(values("Write-Host a` b"), vec!["Write-Host", "a` b"]);
        // the continuation is trivia on the next token, and because it joins
        // the lines it does not open a new logical line: `1 + `<nl>`2` is the
        // single expression `1 + 2`.
        let toks = lex_ok("1 + `\n2");
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Number, Operator, Number, Eof]
        );
        assert_eq!(toks[2].leading.len(), 1);
        assert_eq!(toks[2].leading[0].kind, TriviaKind::LineContinuation);
        assert!(!toks[2].starts_line());
    }

    #[test]
    fn variables_and_splatting() {
        assert_eq!(
            values("$env:PATH ${a b} $_ $? @args @$list $global:x"),
            vec![
                "$env:PATH",
                "${a b}",
                "$_",
                "$?",
                "@args",
                "@$list",
                "$global:x"
            ]
        );
        assert!(kinds("$env:PATH ${a b} $_ $? @args @$list $global:x")
            .iter()
            .take(7)
            .all(|k| *k == Variable));
    }

    #[test]
    fn numbers() {
        let src = "0xFF 0b101 1.5e3 4kb 1ul .5 7zip";
        assert_eq!(
            kinds(src),
            vec![Number, Number, Number, Number, Number, Number, Generic, Eof]
        );
        assert_eq!(kinds("1..5"), vec![Number, Operator, Number, Eof]);
    }

    #[test]
    fn dot_member_access_vs_leading_dot_number() {
        // v1 rule: after Variable/RParen/RBracket/strings/Generic, a `.` is
        // member access even when a digit follows.
        assert_eq!(kinds("$x.5"), vec![Variable, Dot, Number, Eof]);
        assert_eq!(kinds("$x = .5"), vec![Variable, Operator, Number, Eof]);
        // a line break resets the context, as it does in v1's token stream
        let toks = lex_ok("foo\n.5");
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Generic, Number, Eof]
        );
        assert_eq!(toks[1].value, ".5");
    }

    #[test]
    fn null_conditional_operators_match_v1() {
        assert_eq!(kinds("$x?.Length"), vec![Variable, Operator, Generic, Eof]);
        assert_eq!(
            kinds("$a?[0]"),
            vec![Variable, Operator, Number, RBracket, Eof]
        );
        assert_eq!(kinds("$a ?? $b"), vec![Variable, Operator, Variable, Eof]);
        assert_eq!(kinds("$a ??= 1"), vec![Variable, Operator, Number, Eof]);
    }

    #[test]
    fn chains_subexpr_and_misc_punctuation() {
        assert_eq!(
            kinds("a && b || c"),
            vec![Generic, Operator, Generic, Operator, Generic, Eof]
        );
        assert_eq!(
            kinds("$(1) @(2) @{a=1} [int]::MaxValue"),
            vec![
                DollarParen,
                Number,
                RParen,
                AtParen,
                Number,
                RParen,
                AtBrace,
                Generic,
                Operator,
                Number,
                RBrace,
                LBracket,
                Generic,
                RBracket,
                DoubleColon,
                Generic,
                Eof
            ]
        );
        // a label lexes as `:` Operator + word, the v1 shape
        assert_eq!(
            kinds(":outer foreach"),
            vec![Operator, Generic, Keyword, Eof]
        );
    }

    #[test]
    fn bom_survives_as_whitespace_trivia() {
        let src = "\u{feff}ls";
        let toks = lex_ok(src);
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Generic, Eof]
        );
        assert_eq!(toks[0].leading.len(), 1);
        assert_eq!(toks[0].leading[0].kind, TriviaKind::Whitespace);
        assert_eq!(toks[0].leading[0].text, "\u{feff}");
    }

    #[test]
    fn unterminated_inputs_error_but_stay_lossless() {
        for src in [
            "'open",
            "\"open",
            "@'\nno end",
            "<# no end",
            "${no end",
            "\"a $(1 + ",
            "x `",
        ] {
            let out = lex(src);
            assert_eq!(reconstruct(&out.tokens), src, "round trip for {src:?}");
            assert!(!out.errors.is_empty(), "expected an error for {src:?}");
        }
    }

    #[test]
    fn unicode_is_boundary_safe() {
        let src = "Write-Output 'zażółć 🦀' # ok 🚀";
        let toks = lex_ok(src);
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![Generic, StringSq, Eof]
        );
    }

    #[test]
    fn empty_and_trivia_only_inputs() {
        let toks = lex_ok("");
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].kind, Eof);

        let toks = lex_ok("  # just a comment\n");
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].leading.len(), 3); // ws, comment, newline
    }
}
