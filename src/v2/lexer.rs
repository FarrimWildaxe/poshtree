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
//! 3. **Token classification.** The [`KEYWORDS`] and [`NAMED_OPERATORS`]
//!    tables support `Keyword` vs `Generic` and `Operator` vs `Parameter`
//!    classification. `?.`/`?[`/`??`/`??=` are single operators; the
//!    `.5`-vs-member-access decision depends on the preceding token and trivia.
//!
//! The lexer does not model PowerShell's command/expression mode split;
//! context-sensitive decisions are deferred to the parser. Where a source
//! word is split across tokens, byte adjacency (`a.span.end == b.span.start`)
//! lets the parser rejoin them when needed. See the [lexing details](crate::v2)
//! for examples. Kind assignment can be heuristic; spans and bytes never are.

use std::convert::Infallible;

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
/// Unlike the former v1 `tokenize`, this retains a leading UTF-8 BOM as
/// whitespace trivia so reconstruction stays exact. Strip it first with
/// [`crate::encoding::strip_bom`] if you do not want it in the stream.
pub fn lex(src: &str) -> LexOutput {
    let mut tokens = Vec::new();
    let errors = match Lexer::new(src).scan_into(
        |trivia: &mut Vec<Trivia>, kind, span| {
            trivia.push(Trivia {
                kind,
                text: span.slice(src).to_string(),
                span,
            });
        },
        || Ok(()),
        |kind, span, leading, trailing| {
            tokens.push(Token {
                kind,
                value: span.slice(src).to_string(),
                span,
                leading,
                trailing,
            });
            Ok::<(), Infallible>(())
        },
    ) {
        Ok(errors) => errors,
        Err(never) => match never {},
    };
    LexOutput { tokens, errors }
}

/// Emits completed kinds and spans without materializing token or trivia text.
/// A receiver or scan-work refusal stops scanning without undoing receiver effects.
#[cfg(test)]
pub(super) fn lex_records_into<E>(
    src: &str,
    charge_scan_work: impl FnMut() -> Result<(), E>,
    mut emit: impl FnMut(TokenKind, Span) -> Result<(), E>,
) -> Result<Vec<LexError>, E> {
    Lexer::new(src).scan_into(
        |_, _, _| {},
        charge_scan_work,
        |kind, span, (), ()| emit(kind, span),
    )
}

/// Byte length of an error-free `$` variable prefix, using the token scanner's
/// grammar. Other dollar tokens (notably `$(`) and malformed variables do not
/// qualify; diagnostics stay local to this probe.
pub(super) fn variable_reference_len(src: &str) -> Option<usize> {
    if !src.starts_with('$') {
        return None;
    }
    let mut lexer = Lexer::new(src);
    let kind = lexer.scan_dollar();
    (kind == TokenKind::Variable && lexer.errors.is_empty()).then_some(lexer.pos)
}

/// Exclusive byte end of a closed root `$()` prefix, using the string scanner's
/// boundary grammar. Non-prefixes and unclosed roots return `None`. Probe
/// diagnostics stay local; callers retain their existing diagnostic policy.
pub(super) fn scan_subexpression_end(src: &str) -> Option<usize> {
    if !src.starts_with("$(") {
        return None;
    }
    let mut lexer = Lexer::new(src);
    match lexer.skip_subexpression(&mut || Ok::<(), Infallible>(())) {
        Ok(()) => {}
        Err(never) => match never {},
    }
    // This scanner reports only unclosed contexts. Any unclosed descendant
    // also leaves the root unclosed, even if the input ends with a nested `)`.
    lexer.errors.is_empty().then_some(lexer.pos)
}

/// Exclusive byte end of the first here-string terminator in the suffix
/// after the opener, for the given ASCII opener quote.
pub(super) fn here_terminator_end<E>(
    suffix: &str,
    quote: u8,
    mut charge: impl FnMut() -> Result<(), E>,
) -> Result<Option<usize>, E> {
    enum State {
        Body,
        AfterLineBreak,
        AfterQuote,
    }

    charge()?;
    let mut state = State::Body;
    for (index, byte) in suffix.as_bytes().iter().enumerate() {
        // Borrow the byte in the loop header, but read it only after admission.
        charge()?;
        state = match (*byte, state) {
            // A new break can start a candidate even if the previous one failed.
            (b'\r' | b'\n', _) => State::AfterLineBreak,
            (byte, State::AfterLineBreak) if byte == quote => State::AfterQuote,
            (b'@', State::AfterQuote) => return Ok(Some(index + 1)),
            _ => State::Body,
        };
    }
    Ok(None)
}

fn is_keyword(word: &str) -> bool {
    KEYWORDS.contains(&word.to_lowercase().as_str())
}

fn is_named_operator(word: &str) -> bool {
    crate::ops::is_named_operator_word(word, NAMED_OPERATORS)
}

/// Non-newline blank characters that become [`TriviaKind::Whitespace`]. The
/// BOM is included so a file that starts with one still round-trips.
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

/// A `.` after one of these token kinds is member access, even if a digit
/// follows, unless separating trivia resets the context.
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

/// Trivia that resets member-access context between significant tokens.
fn is_separator(kind: TriviaKind) -> bool {
    matches!(
        kind,
        TriviaKind::Newline | TriviaKind::LineComment | TriviaKind::BlockComment
    )
}

/// Suspended contexts inside a string's subexpression. Keeping these flat
/// avoids growing the Rust call stack when quotes and `$()` alternate.
#[derive(Debug)]
enum StringScanFrame {
    DoubleQuoted { start: usize },
    Subexpression { start: usize, depth: usize },
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
    /// True when a `.` at the current token would be member access, based on
    /// `prev_kind` and separating trivia on either side of the gap.
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

    /// Walk to CR, LF, or EOF, leaving any delimiter for the caller.
    fn skip_to_line_end<E>(
        &mut self,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        // Admit the initial bounds check even at EOF. Each subsequent unit
        // covers the byte read, comparison, advance, and next bounds check.
        charge_scan_work()?;
        let bytes = self.src.as_bytes();
        let mut end = self.pos;
        while end < bytes.len() {
            charge_scan_work()?;
            if matches!(bytes[end], b'\r' | b'\n') {
                break;
            }
            end += 1;
        }
        // ASCII CR/LF and EOF are UTF-8 boundaries.
        self.pos = end;
        Ok(())
    }

    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.errors.push(LexError {
            span,
            message: message.into(),
        });
    }

    // Driver

    // Each side of a token starts with an empty trivia accumulator (T::default).
    // A receiver or scan-work error stops scanning; diagnostics are returned on completion.
    fn scan_into<T: Default, E>(
        mut self,
        mut push_trivia: impl FnMut(&mut T, TriviaKind, Span),
        mut charge_scan_work: impl FnMut() -> Result<(), E>,
        mut emit_token: impl FnMut(TokenKind, Span, T, T) -> Result<(), E>,
    ) -> Result<Vec<LexError>, E> {
        loop {
            // After `--%`, the rest of the line is one verbatim token and
            // gets no leading trivia: a space or `#` there is argument text.
            if self.verbatim_pending {
                self.verbatim_pending = false;
                let start = self.pos;
                self.skip_to_line_end(&mut charge_scan_work)?;
                if self.pos != start {
                    let span = Span::new(start, self.pos);
                    let (trailing, separated) =
                        self.scan_trailing_trivia(&mut push_trivia, &mut charge_scan_work)?;
                    self.prev_kind = Some(TokenKind::VerbatimArgs);
                    self.pending_separator = separated;
                    emit_token(TokenKind::VerbatimArgs, span, T::default(), trailing)?;
                    continue;
                }
            }

            let (leading, leading_separator) =
                self.scan_leading_trivia(&mut push_trivia, &mut charge_scan_work)?;
            if self.at_eof() {
                let span = Span::new(self.pos, self.pos);
                emit_token(TokenKind::Eof, span, leading, T::default())?;
                return Ok(self.errors);
            }

            // A newline or comment resets member-access context whether it
            // sits in the previous token's trailing trivia or this token's
            // leading trivia.
            let separated = self.pending_separator || leading_separator;
            self.member_context = !separated && self.prev_kind.is_some_and(allows_member);

            let start = self.pos;
            let kind = self.scan_significant(&mut charge_scan_work)?;
            debug_assert!(self.pos > start, "lexer failed to make progress");
            let span = Span::new(start, self.pos);
            let (trailing, trailing_separator) = if self.verbatim_pending {
                (T::default(), false) // keep the bytes after `--%` out of trivia
            } else {
                self.scan_trailing_trivia(&mut push_trivia, &mut charge_scan_work)?
            };
            self.prev_kind = Some(kind);
            self.pending_separator = trailing_separator;
            emit_token(kind, span, leading, trailing)?;
        }
    }

    // Trivia

    fn scan_leading_trivia<T: Default, E>(
        &mut self,
        push_trivia: &mut impl FnMut(&mut T, TriviaKind, Span),
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(T, bool), E> {
        let mut out = T::default();
        let mut separated = false;
        loop {
            let start = self.pos;
            let kind = match self.peek() {
                Some('\r') => {
                    self.bump();
                    if self.peek() == Some('\n') {
                        self.bump();
                    }
                    TriviaKind::Newline
                }
                Some('\n') => {
                    self.bump();
                    TriviaKind::Newline
                }
                Some(c) if is_blank(c) => {
                    while matches!(self.peek(), Some(c) if is_blank(c)) {
                        self.bump();
                    }
                    TriviaKind::Whitespace
                }
                Some('#') => {
                    self.skip_line_comment(charge_scan_work)?;
                    TriviaKind::LineComment
                }
                Some('<') => {
                    if !self.try_skip_block_comment(charge_scan_work)? {
                        break;
                    }
                    TriviaKind::BlockComment
                }
                Some('`') if matches!(self.peek2(), Some('\r' | '\n')) => {
                    self.bump(); // `
                    if self.peek() == Some('\r') {
                        self.bump();
                    }
                    if self.peek() == Some('\n') {
                        self.bump();
                    }
                    TriviaKind::LineContinuation
                }
                _ => break,
            };
            push_trivia(&mut out, kind, Span::new(start, self.pos));
            separated |= is_separator(kind);
        }
        Ok((out, separated))
    }

    /// Trailing trivia: spaces/tabs, then at most one line comment, then at
    /// most one newline. Anything else (a block comment, a continuation, a
    /// second line) belongs to the next token's leading trivia.
    fn scan_trailing_trivia<T: Default, E>(
        &mut self,
        push_trivia: &mut impl FnMut(&mut T, TriviaKind, Span),
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(T, bool), E> {
        let mut out = T::default();
        let mut separated = false;
        let start = self.pos;
        while matches!(self.peek(), Some(c) if is_blank(c)) {
            self.bump();
        }
        if self.pos > start {
            push_trivia(&mut out, TriviaKind::Whitespace, Span::new(start, self.pos));
        }
        if self.peek() == Some('#') {
            let start = self.pos;
            self.skip_line_comment(charge_scan_work)?;
            let span = Span::new(start, self.pos);
            push_trivia(&mut out, TriviaKind::LineComment, span);
            separated = true;
        }
        let start = self.pos;
        match self.peek() {
            Some('\r') => {
                self.bump();
                if self.peek() == Some('\n') {
                    self.bump();
                }
            }
            Some('\n') => self.bump(),
            _ => return Ok((out, separated)),
        }
        push_trivia(&mut out, TriviaKind::Newline, Span::new(start, self.pos));
        Ok((out, true))
    }

    fn skip_line_comment<E>(
        &mut self,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        self.skip_to_line_end(charge_scan_work)
    }

    /// After the caller sees `<`, try to consume a block comment. A missing
    /// opener or a work refusal leaves the cursor and diagnostics unchanged.
    fn try_skip_block_comment<E>(
        &mut self,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<bool, E> {
        // Entry covers the second-byte bounds check, even for a lone `<`.
        charge_scan_work()?;
        let start = self.pos;
        let bytes = self.src.as_bytes();
        let mut end = start + 1;
        if end == bytes.len() {
            return Ok(false);
        }
        charge_scan_work()?;
        if bytes[end] != b'#' {
            return Ok(false);
        }
        end += 1;
        // The opener's `#` cannot begin a closer. Admit each later byte
        // before reading it, including both bytes of the first body `#>`.
        let mut previous_is_hash = false;
        while end < bytes.len() {
            charge_scan_work()?;
            let byte = bytes[end];
            end += 1;
            if previous_is_hash && byte == b'>' {
                self.pos = end; // After ASCII `>`, a UTF-8 boundary.
                return Ok(true);
            }
            previous_is_hash = byte == b'#';
        }
        self.pos = end; // EOF is also a UTF-8 boundary.
        self.error(Span::new(start, end), "unterminated block comment");
        Ok(true)
    }

    // Significant tokens

    fn scan_significant<E>(
        &mut self,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<TokenKind, E> {
        let c = self.peek().expect("checked by caller");
        let kind = match c {
            '$' => self.scan_dollar(),
            '@' => self.scan_at(charge_scan_work)?,
            '\'' => {
                self.skip_sq_body();
                TokenKind::StringSq
            }
            '"' => {
                self.skip_dq_body(charge_scan_work)?;
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
        };
        Ok(kind)
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

    fn scan_at<E>(
        &mut self,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<TokenKind, E> {
        let start = self.pos;
        self.bump(); // @
        let kind = match self.peek() {
            Some('(') => self.one(TokenKind::AtParen),
            Some('{') => self.one(TokenKind::AtBrace),
            Some(q @ ('\'' | '"')) => self.scan_here_string(start, q, charge_scan_work)?,
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
        };
        Ok(kind)
    }

    /// `@'`/`@"` here-string. The terminator quote must sit at the start of
    /// a line, so the scan looks for the next line break immediately
    /// followed by `'@`/`"@`; the body in between is otherwise verbatim.
    /// Like v1, a lone `\r` counts as a line break.
    fn scan_here_string<E>(
        &mut self,
        start: usize,
        quote: char,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<TokenKind, E> {
        let kind = if quote == '\'' {
            TokenKind::HereStringSq
        } else {
            TokenKind::HereStringDq
        };
        self.bump(); // quote
        match here_terminator_end(self.rest(), quote as u8, charge_scan_work)? {
            Some(end) => self.pos += end,
            None => {
                self.pos = self.src.len();
                self.error(Span::new(start, self.pos), "unterminated here-string");
            }
        }
        Ok(kind)
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
    /// comments, and here-strings. The whole thing stays one token; nested
    /// quotes and subexpressions use heap frames rather than recursive calls.
    fn skip_dq_body<E>(
        &mut self,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        let start = self.pos;
        self.bump(); // "
        loop {
            match self.peek() {
                None => {
                    self.error(Span::new(start, self.pos), "unterminated string");
                    return Ok(());
                }
                Some('`') => self.skip_backtick_escape(),
                Some('"') => {
                    self.bump();
                    if self.peek() == Some('"') {
                        self.bump(); // doubled quote, keep going
                    } else {
                        return Ok(());
                    }
                }
                Some('$') if self.peek2() == Some('(') => {
                    self.skip_subexpression(charge_scan_work)?;
                }
                Some(_) => self.bump(),
            }
        }
    }

    /// Balanced `$( ... )` inside a double-quoted string. Tracks paren depth
    /// while skipping nested strings, comments, and here-strings so that a
    /// `)` inside `"...)..."` or `# ...)` does not close the subexpression.
    fn skip_subexpression<E>(
        &mut self,
        charge_scan_work: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<(), E> {
        let start = self.pos;
        self.bump(); // $
        self.bump(); // (
        let mut frames = vec![StringScanFrame::Subexpression { start, depth: 1 }];
        // Every push consumes an opener, bounding frames by input bytes.
        // Each iteration consumes input or pops a frame; EOF unwinds inside
        // out, preserving the recursive scanner's diagnostic order and spans.
        while let Some(frame) = frames.last_mut() {
            match frame {
                StringScanFrame::DoubleQuoted { start } => match self.peek() {
                    None => {
                        self.error(Span::new(*start, self.pos), "unterminated string");
                        frames.pop();
                    }
                    Some('`') => self.skip_backtick_escape(),
                    Some('"') => {
                        self.bump();
                        if self.peek() == Some('"') {
                            self.bump(); // doubled quote, keep going
                        } else {
                            frames.pop();
                        }
                    }
                    Some('$') if self.peek2() == Some('(') => {
                        let start = self.pos;
                        self.bump(); // $
                        self.bump(); // (
                        frames.push(StringScanFrame::Subexpression { start, depth: 1 });
                    }
                    Some(_) => self.bump(),
                },
                StringScanFrame::Subexpression { start, depth } => match self.peek() {
                    None => {
                        self.error(
                            Span::new(*start, self.pos),
                            "unterminated subexpression in string",
                        );
                        frames.pop();
                    }
                    Some('(') => {
                        self.bump();
                        *depth += 1;
                    }
                    Some(')') => {
                        self.bump();
                        *depth -= 1;
                        if *depth == 0 {
                            frames.pop();
                        }
                    }
                    Some('\'') => self.skip_sq_body(),
                    Some('"') => {
                        let start = self.pos;
                        self.bump(); // "
                        frames.push(StringScanFrame::DoubleQuoted { start });
                    }
                    Some('`') => self.skip_backtick_escape(),
                    Some('<') => {
                        if !self.try_skip_block_comment(charge_scan_work)? {
                            self.pos += 1; // Consume only the ASCII `<`.
                        }
                    }
                    Some('#') => self.skip_line_comment(charge_scan_work)?,
                    Some('@') if matches!(self.peek2(), Some('\'' | '"')) => {
                        let here_start = self.pos;
                        let quote = self.peek2().expect("just matched");
                        self.bump(); // @
                        self.scan_here_string(here_start, quote, charge_scan_work)?;
                    }
                    Some(_) => self.bump(),
                },
            }
        }
        Ok(())
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
    /// both case-insensitive, longest match first. The former v1 lexer
    /// recognized only `l`/`d`/`u` plus the multipliers.
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
            // `.5` is a number without the member context present in `$x.5`.
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
        // Match the longest operator form before scanning wildcard words.
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

    fn expected_token(kind: TokenKind, value: &str, start: usize, end: usize) -> Token {
        Token {
            kind,
            value: value.into(),
            span: Span::new(start, end),
            leading: Vec::new(),
            trailing: Vec::new(),
        }
    }

    fn expected_trivia(kind: TriviaKind, text: &str, start: usize, end: usize) -> Trivia {
        Trivia {
            kind,
            text: text.into(),
            span: Span::new(start, end),
        }
    }

    use TokenKind::*;

    #[test]
    fn lex_line_comments_preserve_trivia_spans_and_reset_member_context() {
        let src = "\u{feff}#\n$x #é\r\n.5";
        let mut variable = expected_token(Variable, "$x", 5, 7);
        variable.leading = vec![
            expected_trivia(TriviaKind::Whitespace, "\u{feff}", 0, 3),
            expected_trivia(TriviaKind::LineComment, "#", 3, 4),
            expected_trivia(TriviaKind::Newline, "\n", 4, 5),
        ];
        variable.trailing = vec![
            expected_trivia(TriviaKind::Whitespace, " ", 7, 8),
            expected_trivia(TriviaKind::LineComment, "#é", 8, 11),
            expected_trivia(TriviaKind::Newline, "\r\n", 11, 13),
        ];

        let tokens = lex_ok(src);

        assert_eq!(
            tokens,
            [
                variable,
                expected_token(Number, ".5", 13, 15),
                expected_token(Eof, "", 15, 15),
            ],
        );
    }

    #[test]
    fn lex_block_comment_stays_leading_and_resets_member_context() {
        let src = "$x <#é\r\n##>.5";
        let mut variable = expected_token(Variable, "$x", 0, 2);
        variable.trailing = vec![expected_trivia(TriviaKind::Whitespace, " ", 2, 3)];
        let mut number = expected_token(Number, ".5", 12, 14);
        number.leading = vec![expected_trivia(
            TriviaKind::BlockComment,
            "<#é\r\n##>",
            3,
            12,
        )];

        let tokens = lex_ok(src);

        assert_eq!(tokens, [variable, number, expected_token(Eof, "", 14, 14)]);
    }

    macro_rules! record_emission_cases {
        ($($name:ident: $src:literal, [$(($kind:ident, $start:literal, $end:literal)),+ $(,)?], $diagnostics:expr;)*) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;
                    let expected = [$(Some(($kind, Span::new($start, $end)))),+];
                    let mut records = expected.map(|_| None);
                    let mut slots = records.iter_mut();

                    let result = lex_records_into(src, || Ok(()), |kind, span| {
                        let record = (kind, span);
                        *slots.next().ok_or(record)? = Some(record);
                        Ok::<(), (TokenKind, Span)>(())
                    });

                    assert_eq!(result.as_ref().err(), None, "unexpected extra record");
                    assert_eq!(records, expected);
                    assert_eq!(result.expect("no extra record was offered"), $diagnostics);
                }
            )*
        };
    }

    record_emission_cases! {
        record_emission_empty_source_emits_only_eof:
            "", [(Eof, 0, 0)], [];
        record_emission_bom_and_unicode_here_string_use_byte_spans:
            "\u{feff}@'\né\n'@;", [(HereStringSq, 3, 11), (Semicolon, 11, 12), (Eof, 12, 12)], [];
        record_emission_stop_parsing_with_tail_emits_verbatim_record:
            "--% #x\r\n;", [(Operator, 0, 3), (VerbatimArgs, 3, 6), (Semicolon, 8, 9), (Eof, 9, 9)], [];
        record_emission_stop_parsing_before_newline_skips_empty_tail:
            "--%\r\n;", [(Operator, 0, 3), (Semicolon, 5, 6), (Eof, 6, 6)], [];
        record_emission_stop_parsing_at_eof_skips_empty_tail:
            "--%", [(Operator, 0, 3), (Eof, 3, 3)], [];
        record_emission_separators_reset_dot_member_context:
            "$x.5\n$x\r\n.5\n$x <# #> .5",
            [
                (Variable, 0, 2), (Dot, 2, 3), (Number, 3, 4),
                (Variable, 5, 7), (Number, 9, 11),
                (Variable, 12, 14), (Number, 21, 23), (Eof, 23, 23),
            ],
            [];
        record_emission_malformed_tokens_return_diagnostics_in_source_order:
            "$ @\n", [(Unknown, 0, 1), (Unknown, 2, 3), (Eof, 4, 4)],
            [
                LexError { span: Span::new(0, 1), message: "lone `$`".into() },
                LexError { span: Span::new(2, 3), message: "lone `@`".into() },
            ];
    }

    #[test]
    fn record_emission_block_comment_offers_eof_only_after_admitting_closer() {
        #[derive(Debug, PartialEq, Eq)]
        struct WorkRefused;

        let semicolon = (Semicolon, Span::new(0, 1));
        let eof = (Eof, Span::new(6, 6));
        for (budget, expected_result, expected_records) in [
            (4_usize, Err(WorkRefused), vec![semicolon]),
            (5, Ok(vec![]), vec![semicolon, eof]),
        ] {
            // The body `<#` does not nest; its `#` closes with the next byte.
            let mut remaining = budget;
            let mut records = Vec::new();

            let result = lex_records_into(
                ";<#<#>",
                || {
                    remaining = remaining.checked_sub(1).ok_or(WorkRefused)?;
                    Ok(())
                },
                |kind, span| {
                    records.push((kind, span));
                    Ok(())
                },
            );

            assert_eq!(
                (result, records, remaining),
                (expected_result, expected_records, 0),
                "budget {budget}",
            );
        }
    }

    #[test]
    fn record_emission_full_receiver_refuses_candidate_and_stops() {
        #[derive(Debug, PartialEq, Eq)]
        struct ReceiverFull((TokenKind, Span));

        let src = ";,";
        let mut published = [None; 1];
        let mut attempts = 0;

        let result = lex_records_into(
            src,
            || Ok(()),
            |kind, span| {
                let record = (kind, span);
                let slot = published.get_mut(attempts);
                attempts += 1;
                *slot.ok_or(ReceiverFull(record))? = Some(record);
                Ok(())
            },
        );

        assert_eq!(result, Err(ReceiverFull((Comma, Span::new(1, 2)))));
        assert_eq!(published, [Some((Semicolon, Span::new(0, 1)))]);
        assert_eq!(attempts, 2, "EOF must not be offered after refusal");
    }

    macro_rules! subexpression_end_cases {
        ($($name:ident: $src:literal, $end:expr;)*) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;

                    let end = scan_subexpression_end(src);

                    assert_eq!(end, $end);
                }
            )*
        };
    }

    subexpression_end_cases! {
        subexpression_end_empty_is_none: "", None;
        subexpression_end_non_prefix_is_none: "x$(1)", None;
        subexpression_end_lone_dollar_is_none: "$", None;
        subexpression_end_plain_paren_is_none: "(1)", None;
        subexpression_end_empty_is_exclusive: "$()tail", Some(3);
        subexpression_end_unicode_counts_bytes: "$(é)tail", Some(5);
        subexpression_end_ignores_malformed_tail: "$(1) \"${open", Some(4);
        subexpression_end_single_quote_hides_closer: "$(')')tail", Some(6);
        subexpression_end_double_quote_hides_opener: "$( \"(\" )tail", Some(8);
        subexpression_end_single_doubled_quote_hides_parens: "$('a'')(')tail", Some(10);
        subexpression_end_double_doubled_quote_hides_parens: "$(\"a\"\")(\")tail", Some(10);
        subexpression_end_code_escape_hides_closer: "$( `) )tail", Some(7);
        subexpression_end_line_comment_hides_parens: "$(1 # ) (\r\n)tail", Some(12);
        subexpression_end_block_comment_is_not_nested: "$(<# <# #>)tail", Some(11);
        subexpression_end_single_here_body_is_verbatim: "$(@'\n) (\n'@)tail", Some(12);
        subexpression_end_double_here_body_is_verbatim: "$(@\"\n$( ' )\n\"@)tail", Some(15);
        subexpression_end_nested_code_waits_for_root: "$($(1))tail", Some(7);
        subexpression_end_nested_expandable_waits_for_root: "$(\"$(1)\")tail", Some(9);
        subexpression_end_unclosed_root_is_none: "$(1", None;
        subexpression_end_nested_code_closer_is_not_root: "$($(1)", None;
        subexpression_end_nested_expandable_closer_is_not_root: "$(\"$(1)", None;
        subexpression_end_quoted_final_closer_is_not_root: "$(')", None;
        subexpression_end_comment_final_closer_is_not_root: "$(# )", None;
        subexpression_end_unclosed_block_is_none: "$(<# )", None;
        subexpression_end_unclosed_here_is_none: "$(@\"\n)", None;
        subexpression_end_dangling_escape_is_none: "$(`", None;
    }

    macro_rules! variable_reference_cases {
        ($($name:ident: $src:literal, $raw:literal, $end:literal;)*) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;

                    let consumed = variable_reference_len(src);
                    let out = lex(src);

                    assert_eq!(consumed, Some($end));
                    assert_eq!(
                        (out.tokens[0].kind, out.tokens[0].value.as_str(), out.tokens[0].span),
                        (Variable, $raw, Span::new(0, $end)),
                    );
                    assert_eq!(out.errors, []);
                }
            )*
        };
    }

    variable_reference_cases! {
        variable_reference_ascii_prefix: "$name!", "$name", 5;
        variable_reference_unicode_initial_bytes: "$é!", "$é", 3;
        variable_reference_unicode_continuation_bytes: "$aé!", "$aé", 4;
        variable_reference_nonlatin_bytes: "$变量!", "$变量", 7;
        variable_reference_numeric_initial: "$123!", "$123", 4;
        variable_reference_numeric_qualifier: "$env:1!", "$env:1", 6;
        variable_reference_repeated_qualifiers: "$a:b:c!", "$a:b:c", 6;
        variable_reference_unicode_qualifiers: "$é:变量!", "$é:变量", 10;
        variable_reference_underscore: "$_!", "$_", 2;
        variable_reference_question: "$?!", "$?", 2;
        variable_reference_caret: "$^!", "$^", 2;
        variable_reference_dollar: "$$!", "$$", 2;
        variable_reference_escaped_braced_terminator: "${a`}b}!", "${a`}b}", 7;
        variable_reference_escaped_unicode_in_braces: "${`é}!", "${`é}", 6;
        variable_reference_empty_braces: "${}!", "${}", 3;
        variable_reference_trailing_colon_is_not_qualifier: "$x:", "$x", 2;
        variable_reference_combining_mark_stops_name: "$a\u{301}", "$a", 2;
    }

    macro_rules! rejected_variable_reference_cases {
        ($($name:ident: $src:literal, $kind:ident, $raw:literal, $end:literal,
            [$($message:literal),* $(,)?];)*) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;
                    let expected_errors: Vec<LexError> = vec![$(
                        LexError {
                            span: Span::new(0, $end),
                            message: $message.into(),
                        },
                    )*];

                    let consumed = variable_reference_len(src);
                    let out = lex(src);

                    assert_eq!(consumed, None);
                    assert_eq!(out.tokens[0], expected_token($kind, $raw, 0, $end));
                    assert_eq!(out.errors, expected_errors);
                }
            )*
        };
    }

    rejected_variable_reference_cases! {
        variable_reference_empty_input_is_none: "", Eof, "", 0, [];
        variable_reference_non_dollar_is_none: "é", Generic, "é", 2, [];
        variable_reference_splat_is_none: "@x", Variable, "@x", 2, [];
        variable_reference_subexpression_kind_is_none: "$(1)", DollarParen, "$(", 2, [];
        variable_reference_lone_dollar_is_none: "$", Unknown, "$", 1, ["lone `$`"];
        variable_reference_punctuation_is_none: "$!", Unknown, "$", 1, ["lone `$`"];
        variable_reference_unclosed_braces_is_none:
            "${x", Variable, "${x", 3, ["unterminated braced variable"];
        variable_reference_only_escaped_terminator_is_none:
            "${a`}", Variable, "${a`}", 5, ["unterminated braced variable"];
        variable_reference_dangling_braced_escape_is_none:
            "${a`", Variable, "${a`", 4, ["unterminated braced variable"];
    }

    #[test]
    fn variable_reference_valid_prefix_does_not_inspect_malformed_tail() {
        let src = "$é ${open";

        let consumed = variable_reference_len(src);
        let out = lex(src);

        assert_eq!(consumed, Some(3));
        assert_eq!(
            (
                out.tokens[0].kind,
                out.tokens[0].value.as_str(),
                out.tokens[0].span
            ),
            (Variable, "$é", Span::new(0, 3)),
        );
        assert_eq!(
            out.errors,
            vec![LexError {
                span: Span::new(4, 10),
                message: "unterminated braced variable".into(),
            }],
        );
    }

    // Literal v2 expectations, characterized before replacing the recursive
    // scanner. In particular, do not import the broader v1 boundary grammar.
    macro_rules! dq_lex_cases {
        ($($name:ident: $src:literal, $end:literal,
            [$(($start:literal, $message:literal)),* $(,)?];)*) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;
                    let expected_tokens = vec![
                        expected_token(StringDq, $src, 0, $end),
                        expected_token(Eof, "", $end, $end),
                    ];
                    let expected_errors: Vec<LexError> = vec![$(
                        LexError {
                            span: Span::new($start, $end),
                            message: $message.into(),
                        },
                    )*];

                    let out = lex(src);

                    assert_eq!(out.tokens, expected_tokens);
                    assert_eq!(out.errors, expected_errors);
                    assert_eq!(reconstruct(&out.tokens), src);
                }
            )*
        };
    }

    dq_lex_cases! {
        lex_dq_nested_expandables_stay_one_token:
            r#""a $("b $(1)") c""#, 17, [];
        lex_dq_doubled_quotes_hide_parentheses:
            r#""$( 'a'')(' + "b"")(" )""#, 24, [];
        lex_dq_comments_hide_parentheses:
            "\"$( (1) <# ) ( #> # ) (\r\n 2 )\"", 30, [];
        lex_dq_code_escapes_hide_parentheses_and_quotes:
            r#""$( `( `) `" `' `$ 1 ) tail""#, 28, [];
        lex_dq_string_escapes_preserve_dollar_parity:
            r#""`$( ) `"q`" ``$(1)""#, 20, [];
        lex_dq_single_quoted_here_body_is_verbatim:
            "\"$( @'\n$( \" ) <#\n'@ + 1 ) tail\"", 31, [];
        lex_dq_double_quoted_here_body_is_verbatim:
            "\"$( @\"\r\n$( ' ) #\r\n\"@ + 2 ) tail\"", 32, [];
        lex_dq_here_terminator_accepts_lone_cr:
            "\"$( @'\r)\r'@ )\"", 14, [];
        lex_dq_indented_here_marker_does_not_terminate:
            "\"$( @\"\n)\n \"@\nstill )\n\"@ )\"", 26, [];
        lex_dq_block_comment_stops_at_first_terminator:
            r#""$( <# <# #> ) done""#, 20, [];
        lex_dq_braced_variable_does_not_hide_parenthesis:
            r#""$( ${x)y} tail""#, 16, [];
        lex_dq_code_subexpressions_balance_as_parentheses:
            r#""$( $( (1) ) )""#, 15, [];
        lex_dq_unclosed_nested_contexts_report_innermost_first:
            r#""$( "x $(é"#, 11, [
                (7, "unterminated subexpression in string"),
                (4, "unterminated string"),
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_single_quote_reports_before_enclosing_contexts:
            "\"$( 'é", 7, [
                (4, "unterminated string"),
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_block_comment_reports_before_enclosing_contexts:
            "\"$( <# é", 9, [
                (4, "unterminated block comment"),
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_double_here_reports_before_enclosing_contexts:
            "\"$( @\"\né", 9, [
                (4, "unterminated here-string"),
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_single_here_reports_before_enclosing_contexts:
            "\"$( @'\ré", 9, [
                (4, "unterminated here-string"),
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_line_comment_only_reports_enclosing_contexts:
            "\"$( # é", 8, [
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_dangling_code_escape_only_reports_enclosing_contexts:
            "\"$( `", 5, [
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_inner_quote_reports_before_enclosing_contexts:
            "\"$( \"x", 6, [
                (4, "unterminated string"),
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_subexpression_reports_before_string:
            "\"$(1", 4, [
                (1, "unterminated subexpression in string"),
                (0, "unterminated string"),
            ];
        lex_dq_unclosed_string_reports_its_opening_quote:
            "\"plain", 6, [(0, "unterminated string")];
    }

    #[test]
    fn lex_dq_nested_expandables_keep_surrounding_trivia_and_following_token() {
        let src = "# h\r\n \"a $(\"b $(1)\") c\" # t\r\nnext";
        let mut string = expected_token(StringDq, r#""a $("b $(1)") c""#, 6, 23);
        string.leading = vec![
            expected_trivia(TriviaKind::LineComment, "# h", 0, 3),
            expected_trivia(TriviaKind::Newline, "\r\n", 3, 5),
            expected_trivia(TriviaKind::Whitespace, " ", 5, 6),
        ];
        string.trailing = vec![
            expected_trivia(TriviaKind::Whitespace, " ", 23, 24),
            expected_trivia(TriviaKind::LineComment, "# t", 24, 27),
            expected_trivia(TriviaKind::Newline, "\r\n", 27, 29),
        ];
        let expected_tokens = vec![
            string,
            expected_token(Generic, "next", 29, 33),
            expected_token(Eof, "", 33, 33),
        ];

        let out = lex(src);

        assert_eq!(out.tokens, expected_tokens);
        assert_eq!(out.errors, []);
        assert_eq!(reconstruct(&out.tokens), src);
    }

    #[test]
    fn lex_dq_unclosed_nested_contexts_keep_bom_and_utf8_byte_spans() {
        let src = "\u{feff}\"$( \"x $(é";
        let mut string = expected_token(StringDq, "\"$( \"x $(é", 3, 14);
        string.leading = vec![expected_trivia(TriviaKind::Whitespace, "\u{feff}", 0, 3)];
        let expected_tokens = vec![string, expected_token(Eof, "", 14, 14)];
        let expected_errors = vec![
            LexError {
                span: Span::new(10, 14),
                message: "unterminated subexpression in string".into(),
            },
            LexError {
                span: Span::new(7, 14),
                message: "unterminated string".into(),
            },
            LexError {
                span: Span::new(4, 14),
                message: "unterminated subexpression in string".into(),
            },
            LexError {
                span: Span::new(3, 14),
                message: "unterminated string".into(),
            },
        ];

        let out = lex(src);

        assert_eq!(out.tokens, expected_tokens);
        assert_eq!(out.errors, expected_errors);
        assert_eq!(reconstruct(&out.tokens), src);
    }

    #[test]
    fn lex_dq_4096_nested_expandables_terminate_losslessly_on_normal_stack() {
        // Bounded stress input, run only after removing the recursive cycle.
        let src = format!("{}0{}", "\"$(".repeat(4096), ")\"".repeat(4096));
        let expected_tokens = vec![
            expected_token(StringDq, &src, 0, 20481),
            expected_token(Eof, "", 20481, 20481),
        ];

        let out = lex(&src);

        assert_eq!(out.tokens, expected_tokens);
        assert_eq!(out.errors, []);
        assert_eq!(reconstruct(&out.tokens), src);
    }

    #[test]
    fn lex_dq_4096_unclosed_expandables_unwind_losslessly_on_normal_stack() {
        let src = format!("{}'é", "\"$(".repeat(4096));
        let expected_tokens = vec![
            expected_token(StringDq, &src, 0, 12291),
            expected_token(Eof, "", 12291, 12291),
        ];
        let expected_first_errors = [
            LexError {
                span: Span::new(12288, 12291),
                message: "unterminated string".into(),
            },
            LexError {
                span: Span::new(12286, 12291),
                message: "unterminated subexpression in string".into(),
            },
            LexError {
                span: Span::new(12285, 12291),
                message: "unterminated string".into(),
            },
        ];
        let expected_last_errors = [
            LexError {
                span: Span::new(1, 12291),
                message: "unterminated subexpression in string".into(),
            },
            LexError {
                span: Span::new(0, 12291),
                message: "unterminated string".into(),
            },
        ];

        let out = lex(&src);

        assert_eq!(out.tokens, expected_tokens);
        assert_eq!(
            (out.errors.len(), &out.errors[..3], &out.errors[8191..]),
            (
                8193,
                expected_first_errors.as_slice(),
                expected_last_errors.as_slice(),
            )
        );
        assert_eq!(reconstruct(&out.tokens), src);
    }

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
    fn here_terminator_contract_closed_single_quote_preserves_bom_and_semicolon() {
        let src = "\u{feff}@'\né\n'@;";
        let mut string = expected_token(HereStringSq, "@'\né\n'@", 3, 11);
        string.leading = vec![expected_trivia(TriviaKind::Whitespace, "\u{feff}", 0, 3)];
        let expected_tokens = vec![
            string,
            expected_token(Semicolon, ";", 11, 12),
            expected_token(Eof, "", 12, 12),
        ];

        let out = lex(src);

        assert_eq!(out.tokens, expected_tokens);
        assert_eq!(out.errors, []);
    }

    #[test]
    fn here_terminator_contract_unclosed_double_quote_preserves_bom_and_error_span() {
        let src = "\u{feff}@\"\né";
        let mut string = expected_token(HereStringDq, "@\"\né", 3, 8);
        string.leading = vec![expected_trivia(TriviaKind::Whitespace, "\u{feff}", 0, 3)];
        let expected_tokens = vec![string, expected_token(Eof, "", 8, 8)];
        let expected_errors = vec![LexError {
            span: Span::new(3, 8),
            message: "unterminated here-string".into(),
        }];

        let out = lex(src);

        assert_eq!(out.tokens, expected_tokens);
        assert_eq!(out.errors, expected_errors);
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
