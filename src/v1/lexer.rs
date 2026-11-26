//! A pragmatic PowerShell tokenizer.
//!
//! Two goals shape it. It has to survive obfuscated and adversarial input
//! without crashing or stalling, since the whole point is to read scripts that
//! are trying not to be read. And it has to recognise the constructs that
//! actually show up in real scripts: every string flavour, variables with
//! scopes, the Verb-Noun / -Parameter / -operator split, member and static
//! access, sub-expressions, arrays, hashtables, here-strings, comments, and
//! line continuations.

use super::tokens::{Token, TokenType, KEYWORDS, NAMED_OPERATORS};

fn is_word_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_word_cont(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}
fn is_hex(c: char) -> bool {
    c.is_ascii_hexdigit()
}

/// Whether a `$(` span closed, with the exclusive character index after `)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SubexpressionEnd {
    Closed(usize),
    Unterminated,
}

/// Find the end of a subexpression whose `$` is at character index `start`.
///
/// This is a lexical boundary scan, not a grammar check. Here-string headers
/// and physical-line terminators follow the lexer below. Each character is
/// visited once; nested expandable strings use heap frames, never recursion.
/// Missing openers or any unclosed context return `Unterminated`.
pub(super) fn scan_subexpression_end(chars: &[char], start: usize) -> SubexpressionEnd {
    let Some(tail) = chars.get(start..) else {
        return SubexpressionEnd::Unterminated;
    };
    if !tail.starts_with(&['$', '(']) {
        return SubexpressionEnd::Unterminated;
    }

    enum Context {
        Code(usize),
        SingleQuoted,
        DoubleQuoted,
        LineComment,
        BlockComment(usize),
        HereString(char),
    }

    let mut contexts = vec![Context::Code(1)];
    let mut i = start + 2;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match contexts
            .last_mut()
            .expect("root context returns when closed")
        {
            Context::Code(depth) => match (c, next) {
                ('`', _) => i += 1 + usize::from(next.is_some()),
                ('(', _) => {
                    *depth += 1;
                    i += 1;
                }
                (')', _) => {
                    *depth -= 1;
                    i += 1;
                    if *depth == 0 {
                        contexts.pop();
                        if contexts.is_empty() {
                            return SubexpressionEnd::Closed(i);
                        }
                    }
                }
                ('\'', _) => {
                    contexts.push(Context::SingleQuoted);
                    i += 1;
                }
                ('"', _) => {
                    contexts.push(Context::DoubleQuoted);
                    i += 1;
                }
                ('#', _) => {
                    contexts.push(Context::LineComment);
                    i += 1;
                }
                ('<', Some('#')) => {
                    contexts.push(Context::BlockComment(1));
                    i += 2;
                }
                ('@', Some(quote @ ('\'' | '"'))) => {
                    contexts.push(Context::HereString(quote));
                    i += 2;
                    while i < chars.len() && !matches!(chars[i], '\r' | '\n') {
                        i += 1;
                    }
                }
                _ => i += 1,
            },
            Context::SingleQuoted => {
                if c == '\'' {
                    if next == Some('\'') {
                        i += 2;
                        continue;
                    }
                    contexts.pop();
                }
                i += 1;
            }
            Context::DoubleQuoted => match (c, next) {
                ('`', _) => i += 1 + usize::from(next.is_some()),
                ('"', Some('"')) => i += 2,
                ('"', _) => {
                    contexts.pop();
                    i += 1;
                }
                ('$', Some('(')) => {
                    contexts.push(Context::Code(1));
                    i += 2;
                }
                _ => i += 1,
            },
            Context::LineComment => {
                if matches!(c, '\r' | '\n') {
                    contexts.pop();
                }
                i += 1;
            }
            Context::BlockComment(depth) => match (c, next) {
                ('<', Some('#')) => {
                    *depth += 1;
                    i += 2;
                }
                ('#', Some('>')) => {
                    *depth -= 1;
                    if *depth == 0 {
                        contexts.pop();
                    }
                    i += 2;
                }
                _ => i += 1,
            },
            Context::HereString(quote) => {
                let at_line_start = i == 0 || matches!(chars[i - 1], '\r' | '\n');
                if at_line_start && c == *quote && next == Some('@') {
                    contexts.pop();
                    i += 2;
                } else if *quote == '"' && c == '`' {
                    i += 1 + usize::from(next.is_some());
                } else if *quote == '"' && c == '$' && next == Some('(') {
                    contexts.push(Context::Code(1));
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
    }
    SubexpressionEnd::Unterminated
}

/// State of a running lexer pass.
struct Lexer {
    chars: Vec<char>, // owned copy for char-indexed slicing
    n: usize,
    i: usize,    // current char index into `chars`
    byte: usize, // byte offset into the source matching char index `i`
    line: u32,
    col: u32,
    tokens: Vec<Token>,
}

impl Lexer {
    fn new(source: &str) -> Self {
        let chars: Vec<char> = source.chars().collect();
        let n = chars.len();
        Lexer {
            chars,
            n,
            i: 0,
            byte: 0,
            line: 1,
            col: 1,
            tokens: Vec::new(),
        }
    }

    fn peek(&self, ahead: usize) -> char {
        self.chars.get(self.i + ahead).copied().unwrap_or('\0')
    }

    fn advance(&mut self) -> char {
        let ch = self.chars[self.i];
        self.i += 1;
        self.byte += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        ch
    }

    fn match_str(&mut self, s: &str) -> bool {
        if self.starts_with(s) {
            for _ in s.chars() {
                self.advance();
            }
            true
        } else {
            false
        }
    }

    fn starts_with(&self, s: &str) -> bool {
        // Char-by-char comparison avoids allocating a temporary Vec<char> for
        // `s`, which matters because this runs for nearly every operator and
        // punctuation token.
        s.chars()
            .enumerate()
            .all(|(off, c)| self.chars.get(self.i + off) == Some(&c))
    }

    fn slice_str(&self, start: usize, end: usize) -> String {
        self.chars[start..end].iter().collect()
    }

    fn add(&mut self, ty: TokenType, value: String, line: u32, col: u32, pos: usize) -> &mut Token {
        self.tokens.push(Token::new(ty, value, line, col, pos));
        self.tokens.last_mut().unwrap()
    }

    // Main loop

    fn tokenize(mut self) -> Vec<Token> {
        while self.i < self.n {
            let ch = self.peek(0);

            // line continuation: backtick immediately before a newline
            if ch == '`' && (self.peek(1) == '\n' || self.peek(1) == '\r') {
                self.advance(); // backtick
                if self.peek(0) == '\r' {
                    self.advance();
                }
                if self.peek(0) == '\n' {
                    self.advance();
                }
                continue;
            }

            if " \t\x0c\x0b\u{00a0}".contains(ch) {
                self.advance();
                continue;
            }

            if ch == '\r' || ch == '\n' {
                self.read_newline();
                continue;
            }

            if ch == '#' {
                self.read_line_comment();
                continue;
            }

            if ch == '<' && self.peek(1) == '#' {
                self.read_block_comment();
                continue;
            }

            if ch == '$' {
                // `$(` opens a sub-expression; everything else is a variable.
                if self.peek(1) == '(' {
                    self.read_simple("$(", TokenType::DollarParen);
                    continue;
                }
                self.read_variable();
                continue;
            }

            if ch == '@' {
                let nxt = self.peek(1);
                if nxt == '\'' || nxt == '"' {
                    self.read_here_string(nxt);
                    continue;
                }
                if nxt == '(' {
                    self.read_simple("@(", TokenType::AtParen);
                    continue;
                }
                if nxt == '{' {
                    self.read_simple("@{", TokenType::AtBrace);
                    continue;
                }
                if nxt == '$' || is_word_start(nxt) {
                    self.read_splat();
                    continue;
                }
                self.read_simple("@", TokenType::Unknown);
                continue;
            }

            if ch == '\'' {
                self.read_single_quoted();
                continue;
            }
            if ch == '"' {
                self.read_double_quoted();
                continue;
            }

            // Redirection: `<`, `>`, `>>`, an optional stream digit or `*` glued
            // to `>` (e.g. `2>`), and an optional `&n` handle merge (e.g.
            // `2>&1`). Checked before numbers so a stream digit is not eaten.
            if ch == '>' || ch == '<' || ((ch == '*' || is_digit(ch)) && self.peek(1) == '>') {
                self.read_redirection();
                continue;
            }

            if is_digit(ch) || (ch == '.' && is_digit(self.peek(1)) && !self.prev_allows_member()) {
                self.read_number();
                continue;
            }

            if ch == '-' {
                self.read_hyphen();
                continue;
            }

            if self.read_operator_or_punct() {
                continue;
            }

            if is_word_start(ch) {
                self.read_word();
                continue;
            }

            // unknown single char
            let (line, col, pos) = (self.line, self.col, self.byte);
            let c = self.advance();
            self.add(TokenType::Unknown, c.to_string(), line, col, pos);
        }

        let (line, col, pos) = (self.line, self.col, self.byte);
        self.add(TokenType::Eof, String::new(), line, col, pos);
        self.tokens
    }

    // Readers

    fn read_newline(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let mut val = String::new();
        if self.peek(0) == '\r' {
            val.push(self.advance());
        }
        if self.peek(0) == '\n' {
            val.push(self.advance());
        }
        if val.is_empty() {
            val.push(self.advance());
        }
        self.add(TokenType::Newline, val, line, col, pos);
    }

    fn read_line_comment(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        while self.i < self.n && self.peek(0) != '\r' && self.peek(0) != '\n' {
            self.advance();
        }
        let val = self.slice_str(start, self.i);
        self.add(TokenType::Comment, val, line, col, pos);
    }

    fn read_block_comment(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        self.advance();
        self.advance(); // "<#"
        while self.i < self.n && !(self.peek(0) == '#' && self.peek(1) == '>') {
            self.advance();
        }
        self.match_str("#>");
        let val = self.slice_str(start, self.i);
        self.add(TokenType::Comment, val, line, col, pos);
    }

    fn read_simple(&mut self, lit: &str, ty: TokenType) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        self.match_str(lit);
        self.add(ty, lit.to_owned(), line, col, pos);
    }

    fn read_variable(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        self.advance(); // '$'
        let mut scope: Option<String> = None;

        let name = if self.peek(0) == '{' {
            self.advance();
            let mut buf = String::new();
            while self.i < self.n && self.peek(0) != '}' {
                buf.push(self.advance());
            }
            self.match_str("}");
            buf
        } else if "?^$_".contains(self.peek(0)) {
            self.advance().to_string()
        } else {
            let mut buf = String::new();
            while self.i < self.n && (is_word_cont(self.peek(0)) || self.peek(0) == ':') {
                let c = self.advance();
                buf.push(c);
                if c == ':' && scope.is_none() {
                    scope = Some(buf[..buf.len() - 1].to_owned());
                    buf.clear();
                }
            }
            buf
        };
        let val = self.slice_str(start, self.i);
        let tok = self.add(TokenType::Variable, val, line, col, pos);
        tok.text = Some(name);
        tok.scope = scope;
    }

    fn read_splat(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        self.advance(); // '@'
        if self.peek(0) == '$' {
            self.advance();
        }
        while self.i < self.n && is_word_cont(self.peek(0)) {
            self.advance();
        }
        let val = self.slice_str(start, self.i);
        let name = val.trim_start_matches(['@', '$']).to_owned();
        let tok = self.add(TokenType::Variable, val, line, col, pos);
        tok.text = Some(name);
        tok.splat = true;
    }

    fn read_here_string(&mut self, quote: char) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        self.advance(); // '@'
        self.advance(); // quote
                        // skip to end of physical line
        while self.i < self.n && self.peek(0) != '\r' && self.peek(0) != '\n' {
            self.advance();
        }
        let terminator: String = format!("{quote}@");
        let term_chars: Vec<char> = terminator.chars().collect();
        let body_start = self.i;
        while self.i < self.n {
            let at_line_start =
                self.i == 0 || self.chars[self.i - 1] == '\r' || self.chars[self.i - 1] == '\n';
            if at_line_start && self.chars[self.i..].starts_with(&term_chars) {
                break;
            }
            self.advance();
        }
        let body: String = self.chars[body_start..self.i]
            .iter()
            .collect::<String>()
            .trim_matches(['\r', '\n'])
            .to_owned();
        self.match_str(&terminator);
        let val = self.slice_str(start, self.i);
        let ty = if quote == '\'' {
            TokenType::HereStringSq
        } else {
            TokenType::HereStringDq
        };
        let tok = self.add(ty, val, line, col, pos);
        tok.text = Some(body);
    }

    fn read_single_quoted(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        self.advance(); // opening '
        let mut buf = String::new();
        loop {
            if self.i >= self.n {
                break;
            }
            let c = self.peek(0);
            if c == '\'' {
                if self.peek(1) == '\'' {
                    self.advance();
                    self.advance();
                    buf.push('\'');
                } else {
                    self.advance();
                    break;
                }
            } else {
                buf.push(self.advance());
            }
        }
        let val = self.slice_str(start, self.i);
        let tok = self.add(TokenType::StringSq, val, line, col, pos);
        tok.text = Some(buf);
    }

    fn read_double_quoted(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        self.advance(); // opening "
        let mut buf = String::new();
        let mut subexpression_end = self.i;
        loop {
            if self.i >= self.n {
                break;
            }
            let c = self.peek(0);
            if self.i >= subexpression_end && c == '$' && self.peek(1) == '(' {
                subexpression_end = match scan_subexpression_end(&self.chars, self.i) {
                    SubexpressionEnd::Closed(end) => end,
                    SubexpressionEnd::Unterminated => self.n,
                };
            }
            if c == '`' {
                self.advance();
                if self.i < self.n {
                    let esc = self.advance();
                    buf.push(match esc {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        'a' => '\x07',
                        'b' => '\x08',
                        other => other,
                    });
                }
            } else if c == '"' {
                if self.peek(1) == '"' {
                    self.advance();
                    self.advance();
                    buf.push('"');
                } else if self.i >= subexpression_end {
                    self.advance();
                    break;
                } else {
                    // Only the boundary changes: legacy payload decoding still
                    // applies inside subexpressions until the parser uses raw text.
                    buf.push(self.advance());
                }
            } else {
                buf.push(self.advance());
            }
        }
        let val = self.slice_str(start, self.i);
        let tok = self.add(TokenType::StringDq, val, line, col, pos);
        tok.text = Some(buf);
    }

    fn read_number(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        if self.peek(0) == '0' && (self.peek(1) == 'x' || self.peek(1) == 'X') {
            self.advance();
            self.advance();
            while self.i < self.n && is_hex(self.peek(0)) {
                self.advance();
            }
        } else {
            let mut seen_dot = false;
            loop {
                if self.i >= self.n {
                    break;
                }
                let c = self.peek(0);
                if is_digit(c) {
                    self.advance();
                } else if c == '.' && !seen_dot && is_digit(self.peek(1)) {
                    seen_dot = true;
                    self.advance();
                } else if (c == 'e' || c == 'E')
                    && (is_digit(self.peek(1)) || self.peek(1) == '+' || self.peek(1) == '-')
                {
                    self.advance();
                    if self.peek(0) == '+' || self.peek(0) == '-' {
                        self.advance();
                    }
                } else {
                    break;
                }
            }
        }
        // numeric suffix
        let sfx2 = self
            .slice_str(self.i, (self.i + 2).min(self.n))
            .to_lowercase();
        if ["kb", "mb", "gb", "tb", "pb"].contains(&sfx2.as_str()) {
            self.advance();
            self.advance();
        } else if self.i < self.n && "ldu".contains(self.chars[self.i].to_ascii_lowercase()) {
            self.advance();
        }
        let val = self.slice_str(start, self.i);
        self.add(TokenType::Number, val, line, col, pos);
    }

    fn read_hyphen(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let nxt = self.peek(1);
        if nxt == '=' {
            self.read_simple("-=", TokenType::Operator);
            return;
        }
        if !is_word_start(nxt) {
            self.advance();
            if self.peek(0) == '-' {
                self.advance();
                self.add(TokenType::Operator, "--".into(), line, col, pos);
            } else {
                self.add(TokenType::Operator, "-".into(), line, col, pos);
            }
            return;
        }
        self.advance(); // consume '-'
        let wstart = self.i;
        while self.i < self.n && is_word_cont(self.peek(0)) {
            self.advance();
        }
        let word = self.slice_str(wstart, self.i);
        // Compute the lowercase form once: it both decides the classification
        // and, for a named operator, becomes the normalised `text`.
        let lower = word.to_lowercase();
        if crate::ops::is_named_operator_word(&lower, NAMED_OPERATORS) {
            let tok = self.add(TokenType::Operator, format!("-{word}"), line, col, pos);
            tok.text = Some(lower);
        } else {
            let tok = self.add(TokenType::Parameter, format!("-{word}"), line, col, pos);
            tok.text = Some(word);
        }
    }

    fn read_redirection(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let mut text = String::new();
        // optional stream specifier (`1`-`6` or `*`) glued to `>`
        let c = self.peek(0);
        if (c == '*' || is_digit(c)) && self.peek(1) == '>' {
            text.push(self.advance());
        }
        if self.peek(0) == '<' {
            text.push(self.advance());
        } else {
            text.push(self.advance()); // '>'
            if self.peek(0) == '>' {
                text.push(self.advance()); // second '>' for '>>'
            }
            // handle merge such as `2>&1`: `&` followed by a stream digit
            if self.peek(0) == '&' && is_digit(self.peek(1)) {
                text.push(self.advance()); // '&'
                text.push(self.advance()); // digit
            }
        }
        self.add(TokenType::Redirect, text, line, col, pos);
    }

    fn read_operator_or_punct(&mut self) -> bool {
        let (line, col, pos) = (self.line, self.col, self.byte);

        let three: String = self.chars[self.i..(self.i + 3).min(self.n)]
            .iter()
            .collect();
        if three == "??=" {
            self.match_str("??=");
            self.add(TokenType::Operator, "??=".to_owned(), line, col, pos);
            return true;
        }

        let two: String = self.chars[self.i..(self.i + 2).min(self.n)]
            .iter()
            .collect();
        // null-conditional access operators (PowerShell 7): emitting `?.` / `?[`
        // as single tokens lets the parser treat them like `.` / `[` without
        // re-checking that the `?` is glued to the next token.
        if two == "?." || two == "?[" {
            self.match_str(&two);
            self.add(TokenType::Operator, two, line, col, pos);
            return true;
        }
        macro_rules! tok2 {
            ($lit:expr, $ty:expr) => {
                if two == $lit {
                    self.match_str($lit);
                    self.add($ty, $lit.to_owned(), line, col, pos);
                    return true;
                }
            };
        }
        tok2!("::", TokenType::DoubleColon);
        tok2!("++", TokenType::Operator);
        tok2!("..", TokenType::Operator);
        for op in &["+=", "-=", "*=", "/=", "%=", "&&", "||", "??"] {
            if two == *op {
                self.match_str(op);
                self.add(TokenType::Operator, op.to_string(), line, col, pos);
                return true;
            }
        }

        let c = self.peek(0);
        let single: Option<TokenType> = match c {
            '|' => Some(TokenType::Pipe),
            '&' => Some(TokenType::Amp),
            ';' => Some(TokenType::Semicolon),
            ',' => Some(TokenType::Comma),
            '.' => Some(TokenType::Dot),
            '(' => Some(TokenType::LParen),
            ')' => Some(TokenType::RParen),
            '{' => Some(TokenType::LBrace),
            '}' => Some(TokenType::RBrace),
            '[' => Some(TokenType::LBracket),
            ']' => Some(TokenType::RBracket),
            _ => None,
        };
        if let Some(ty) = single {
            self.advance();
            self.add(ty, c.to_string(), line, col, pos);
            return true;
        }
        if "+*/%=?:".contains(c) {
            self.advance();
            self.add(TokenType::Operator, c.to_string(), line, col, pos);
            return true;
        }
        false
    }

    fn read_word(&mut self) {
        let (line, col, pos) = (self.line, self.col, self.byte);
        let start = self.i;
        loop {
            if self.i >= self.n {
                break;
            }
            let c = self.peek(0);
            if is_word_cont(c) {
                self.advance();
            } else if c == '-' && is_word_start(self.peek(1)) {
                // hyphen with no surrounding space → part of Verb-Noun
                self.advance();
            } else {
                break;
            }
        }
        let word = self.slice_str(start, self.i);
        // One lowercase pass drives the keyword check; a keyword moves `word`
        // into the token value and reuses the lowercase form as its `text`.
        let lower = word.to_lowercase();
        if KEYWORDS.contains(&lower.as_str()) {
            let tok = self.add(TokenType::Keyword, word, line, col, pos);
            tok.text = Some(lower);
        } else {
            let tok = self.add(TokenType::Generic, word.clone(), line, col, pos);
            tok.text = Some(word);
        }
    }

    fn prev_allows_member(&self) -> bool {
        if let Some(t) = self.tokens.last() {
            matches!(
                t.ty,
                TokenType::Variable
                    | TokenType::RParen
                    | TokenType::RBracket
                    | TokenType::StringDq
                    | TokenType::StringSq
                    | TokenType::Generic
            )
        } else {
            false
        }
    }
}

/// Tokenize a PowerShell source string.
pub fn tokenize(source: &str) -> Vec<Token> {
    let source = crate::encoding::strip_bom(source);
    Lexer::new(source).tokenize()
}

/// Return `source` with comment spans replaced by spaces.
///
/// Each comment byte is replaced by an ASCII space, except CR and LF bytes.
/// This preserves byte length, line endings, and byte offsets outside comments,
/// including any leading BOM, but not Unicode character counts or display columns.
pub fn strip_comments(source: &str) -> String {
    strip_comments_tokens(&tokenize(source), source)
}

/// [`strip_comments`] over an already-computed token stream, so callers that
/// have tokenized do not pay for a second lex pass.
///
/// `tokens` must match `source`, as generated by [`tokenize`]. Token positions
/// are relative to the BOM-stripped source; comment offsets are translated back
/// to the original source before masking, preserving any leading BOM.
/// Byte padding preserves byte length and line endings, not Unicode character
/// counts or display columns.
pub fn strip_comments_tokens(tokens: &[Token], source: &str) -> String {
    let bom_offset = source.len() - crate::encoding::strip_bom(source).len();
    let mut bytes = source.as_bytes().to_vec();

    for tok in tokens {
        if tok.ty != TokenType::Comment {
            continue;
        }
        let start = tok.pos + bom_offset; // byte offset in the original source
        let end = (start + tok.value.len()).min(bytes.len());
        for b in &mut bytes[start..end] {
            // Preserve newlines so line numbers are unaffected; blank the rest.
            if *b != b'\r' && *b != b'\n' {
                *b = b' ';
            }
        }
    }
    // Only ASCII spaces were written over whole-character comment spans, so the
    // buffer is still valid UTF-8.
    String::from_utf8(bytes).unwrap_or_else(|_| source.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types(src: &str) -> Vec<TokenType> {
        tokenize(src).into_iter().map(|t| t.ty).collect()
    }

    fn assert_tokens_eq(actual: Vec<Token>, expected: Vec<Token>) {
        let metadata = |token: Token| {
            (
                token.ty,
                token.value,
                token.line,
                token.col,
                token.pos,
                token.text,
                token.scope,
                token.splat,
            )
        };
        assert_eq!(
            actual.into_iter().map(metadata).collect::<Vec<_>>(),
            expected.into_iter().map(metadata).collect::<Vec<_>>()
        );
    }

    macro_rules! quoted_subexpression_cases {
        ($($name:ident: ($source:expr, $raw:expr, $text:expr, [$($tail:expr),* $(,)?])),* $(,)?) => {
            $(
                #[test]
                fn $name() {
                    let mut string = Token::new(TokenType::StringDq, $raw, 1, 1, 0);
                    string.text = Some($text.into());
                    let expected = vec![string, $($tail),*];

                    let tokens = tokenize($source);

                    assert_tokens_eq(tokens, expected);
                }
            )*
        };
    }

    quoted_subexpression_cases! {
        tokenize_subexpression_empty_body_preserves_following_statement: (
            r#""$()";1"#, r#""$()""#, "$()", [
                Token::new(TokenType::Semicolon, ";", 1, 6, 5),
                Token::new(TokenType::Number, "1", 1, 7, 6),
                Token::new(TokenType::Eof, "", 1, 8, 7),
            ]
        ),
        tokenize_subexpression_sequential_bodies_preserves_following_statement: (
            r#""$(")")x$(')')";1"#, r#""$(")")x$(')')""#, r#"$(")")x$(')')"#, [
                Token::new(TokenType::Semicolon, ";", 1, 16, 15),
                Token::new(TokenType::Number, "1", 1, 17, 16),
                Token::new(TokenType::Eof, "", 1, 18, 17),
            ]
        ),
        tokenize_subexpression_even_backticks_leave_opener_active: (
            r#""``$(")")";1"#, r#""``$(")")""#, r#"`$(")")"#, [
                Token::new(TokenType::Semicolon, ";", 1, 11, 10),
                Token::new(TokenType::Number, "1", 1, 12, 11),
                Token::new(TokenType::Eof, "", 1, 13, 12),
            ]
        ),
        tokenize_subexpression_escaped_nested_dollar_is_literal_in_inner_string: (
            r#""$("`$( )")";1"#, r#""$("`$( )")""#, r#"$("$( )")"#, [
                Token::new(TokenType::Semicolon, ";", 1, 13, 12),
                Token::new(TokenType::Number, "1", 1, 14, 13),
                Token::new(TokenType::Eof, "", 1, 15, 14),
            ]
        ),
        tokenize_subexpression_double_quoted_paren_preserves_following_statement: (
            r#""$(")")";1"#, r#""$(")")""#, r#"$(")")"#, [
                Token::new(TokenType::Semicolon, ";", 1, 9, 8),
                Token::new(TokenType::Number, "1", 1, 10, 9),
                Token::new(TokenType::Eof, "", 1, 11, 10),
            ]
        ),
        tokenize_subexpression_single_quoted_paren_preserves_following_statement: (
            r#""$(')')";1"#, r#""$(')')""#, "$(')')", [
                Token::new(TokenType::Semicolon, ";", 1, 9, 8),
                Token::new(TokenType::Number, "1", 1, 10, 9),
                Token::new(TokenType::Eof, "", 1, 11, 10),
            ]
        ),
        tokenize_subexpression_nested_parens_preserves_following_statement: (
            r#""$((1 + (2)))";1"#, r#""$((1 + (2)))""#, "$((1 + (2)))", [
                Token::new(TokenType::Semicolon, ";", 1, 15, 14),
                Token::new(TokenType::Number, "1", 1, 16, 15),
                Token::new(TokenType::Eof, "", 1, 17, 16),
            ]
        ),
        tokenize_subexpression_nested_expandable_strings_preserves_following_statement: (
            r#""$("a$(")")b")";1"#, r#""$("a$(")")b")""#, r#"$("a$(")")b")"#, [
                Token::new(TokenType::Semicolon, ";", 1, 16, 15),
                Token::new(TokenType::Number, "1", 1, 17, 16),
                Token::new(TokenType::Eof, "", 1, 18, 17),
            ]
        ),
        tokenize_subexpression_doubled_quotes_keeps_legacy_normalization: (
            r#""$("a"")""b" + 'c'')d')";1"#,
            r#""$("a"")""b" + 'c'')d')""#,
            r#"$("a")"b" + 'c'')d')"#, [
                Token::new(TokenType::Semicolon, ";", 1, 25, 24),
                Token::new(TokenType::Number, "1", 1, 26, 25),
                Token::new(TokenType::Eof, "", 1, 27, 26),
            ]
        ),
        tokenize_subexpression_escaped_paren_keeps_legacy_normalization: (
            r#""$(`) + 1)";1"#, r#""$(`) + 1)""#, "$() + 1)", [
                Token::new(TokenType::Semicolon, ";", 1, 12, 11),
                Token::new(TokenType::Number, "1", 1, 13, 12),
                Token::new(TokenType::Eof, "", 1, 14, 13),
            ]
        ),
        tokenize_subexpression_escaped_inner_quote_preserves_following_statement: (
            r#""$("a`")b")";1"#, r#""$("a`")b")""#, r#"$("a")b")"#, [
                Token::new(TokenType::Semicolon, ";", 1, 13, 12),
                Token::new(TokenType::Number, "1", 1, 14, 13),
                Token::new(TokenType::Eof, "", 1, 15, 14),
            ]
        ),
        tokenize_subexpression_line_comment_paren_preserves_unicode_crlf_positions: (
            "\"é$(# ) \"\r\n1)終\";1", "\"é$(# ) \"\r\n1)終\"", "é$(# ) \"\r\n1)終", [
                Token::new(TokenType::Semicolon, ";", 2, 5, 18),
                Token::new(TokenType::Number, "1", 2, 6, 19),
                Token::new(TokenType::Eof, "", 2, 7, 20),
            ]
        ),
        tokenize_subexpression_nested_block_comments_preserves_following_statement: (
            r#""$(<# ) " <# ) #> " #>1)";1"#,
            r#""$(<# ) " <# ) #> " #>1)""#, r#"$(<# ) " <# ) #> " #>1)"#, [
                Token::new(TokenType::Semicolon, ";", 1, 26, 25),
                Token::new(TokenType::Number, "1", 1, 27, 26),
                Token::new(TokenType::Eof, "", 1, 28, 27),
            ]
        ),
        tokenize_subexpression_single_here_string_preserves_following_statement: (
            "\"$(@'\r\n) \"\r\n'@\r\n)\";1", "\"$(@'\r\n) \"\r\n'@\r\n)\"", "$(@'\r\n) \"\r\n'@\r\n)", [
                Token::new(TokenType::Semicolon, ";", 4, 3, 18),
                Token::new(TokenType::Number, "1", 4, 4, 19),
                Token::new(TokenType::Eof, "", 4, 5, 20),
            ]
        ),
        tokenize_subexpression_double_here_string_preserves_following_statement: (
            "\"$(@\"\n) '\n\"@\n)\";1", "\"$(@\"\n) '\n\"@\n)\"", "$(@\"\n) '\n\"@\n)", [
                Token::new(TokenType::Semicolon, ";", 4, 3, 15),
                Token::new(TokenType::Number, "1", 4, 4, 16),
                Token::new(TokenType::Eof, "", 4, 5, 17),
            ]
        ),
        tokenize_subexpression_indented_here_string_terminator_stays_in_body: (
            "\"$(@'\n '@ ) \"\n'@\n)\"", "\"$(@'\n '@ ) \"\n'@\n)\"", "$(@'\n '@ ) \"\n'@\n)", [
                Token::new(TokenType::Eof, "", 4, 3, 19),
            ]
        ),
        tokenize_subexpression_double_here_string_nested_expansion_ignores_false_terminator: (
            "\"$(@\"\n$( '\n\"@\n)' )\n\"@\n)\";1",
            "\"$(@\"\n$( '\n\"@\n)' )\n\"@\n)\"", "$(@\"\n$( '\n\"@\n)' )\n\"@\n)", [
                Token::new(TokenType::Semicolon, ";", 6, 3, 24),
                Token::new(TokenType::Number, "1", 6, 4, 25),
                Token::new(TokenType::Eof, "", 6, 5, 26),
            ]
        ),
        tokenize_subexpression_missing_paren_consumes_to_eof: (
            r#""$(1";1"#, r#""$(1";1"#, r#"$(1";1"#, [
                Token::new(TokenType::Eof, "", 1, 8, 7),
            ]
        ),
        tokenize_subexpression_unterminated_single_quote_consumes_to_eof: (
            r#""$(')";1"#, r#""$(')";1"#, r#"$(')";1"#, [
                Token::new(TokenType::Eof, "", 1, 9, 8),
            ]
        ),
        tokenize_subexpression_unterminated_double_quote_consumes_to_eof: (
            r#""$(")';1"#, r#""$(")';1"#, r#"$(")';1"#, [
                Token::new(TokenType::Eof, "", 1, 9, 8),
            ]
        ),
        tokenize_subexpression_unterminated_block_comment_consumes_to_eof: (
            r#""$(<# )";1"#, r#""$(<# )";1"#, r#"$(<# )";1"#, [
                Token::new(TokenType::Eof, "", 1, 11, 10),
            ]
        ),
        tokenize_subexpression_line_comment_without_newline_consumes_to_eof: (
            r#""$(# )";1"#, r#""$(# )";1"#, r#"$(# )";1"#, [
                Token::new(TokenType::Eof, "", 1, 10, 9),
            ]
        ),
        tokenize_subexpression_unterminated_here_string_consumes_to_eof: (
            "\"$(@\"\n)\";1", "\"$(@\"\n)\";1", "$(@\"\n)\";1", [
                Token::new(TokenType::Eof, "", 2, 5, 10),
            ]
        ),
        tokenize_subexpression_unterminated_single_here_string_consumes_to_eof: (
            "\"$(@'\n)\";1", "\"$(@'\n)\";1", "$(@'\n)\";1", [
                Token::new(TokenType::Eof, "", 2, 5, 10),
            ]
        ),
        tokenize_subexpression_trailing_backtick_consumes_to_eof_without_inventing_text: (
            "\"$(1`", "\"$(1`", "$(1", [
                Token::new(TokenType::Eof, "", 1, 6, 5),
            ]
        ),
        tokenize_subexpression_closed_at_eof_without_outer_quote_preserves_payload: (
            r#""$(")")"#, r#""$(")")"#, r#"$(")")"#, [
                Token::new(TokenType::Eof, "", 1, 8, 7),
            ]
        ),
        tokenize_escaped_subexpression_opener_retains_legacy_decoding: (
            r#""`$(x) `` `n`t`r`0`a`b ""q""";1"#,
            r#""`$(x) `` `n`t`r`0`a`b ""q""""#, "$(x) ` \n\t\r\0\x07\x08 \"q\"", [
                Token::new(TokenType::Semicolon, ";", 1, 30, 29),
                Token::new(TokenType::Number, "1", 1, 31, 30),
                Token::new(TokenType::Eof, "", 1, 32, 31),
            ]
        ),
        tokenize_subexpression_without_inner_quotes_retains_legacy_decoding: (
            r#""$(`n + 'a""b')";1"#, r#""$(`n + 'a""b')""#, "$(\n + 'a\"b')", [
                Token::new(TokenType::Semicolon, ";", 1, 17, 16),
                Token::new(TokenType::Number, "1", 1, 18, 17),
                Token::new(TokenType::Eof, "", 1, 19, 18),
            ]
        ),
    }

    #[test]
    fn tokenize_subexpression_deep_parentheses_preserves_following_statement_without_recursion() {
        let body = format!("$({}1{})", "(".repeat(8192), ")".repeat(8192));
        let raw = format!("\"{body}\"");
        let source = format!("{raw};1");
        let mut string = Token::new(TokenType::StringDq, raw, 1, 1, 0);
        string.text = Some(body);
        let expected = vec![
            string,
            Token::new(TokenType::Semicolon, ";", 1, 16391, 16390),
            Token::new(TokenType::Number, "1", 1, 16392, 16391),
            Token::new(TokenType::Eof, "", 1, 16393, 16392),
        ];

        let tokens = tokenize(&source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_subexpression_deep_expandable_strings_preserves_following_statement_without_recursion(
    ) {
        let body = format!("{}1{}", "$(\"a".repeat(2048), "b\")".repeat(2048));
        let raw = format!("\"{body}\"");
        let source = format!("{raw};1");
        let mut string = Token::new(TokenType::StringDq, raw, 1, 1, 0);
        string.text = Some(body);
        let expected = vec![
            string,
            Token::new(TokenType::Semicolon, ";", 1, 14340, 14339),
            Token::new(TokenType::Number, "1", 1, 14341, 14340),
            Token::new(TokenType::Eof, "", 1, 14342, 14341),
        ];

        let tokens = tokenize(&source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_subexpression_deep_unterminated_expandable_strings_consumes_to_eof_without_recursion(
    ) {
        let body = format!("{}1", "$(\"a".repeat(2048));
        let source = format!("\"{body}");
        let mut string = Token::new(TokenType::StringDq, &source, 1, 1, 0);
        string.text = Some(body);
        let expected = vec![string, Token::new(TokenType::Eof, "", 1, 8195, 8194)];

        let tokens = tokenize(&source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_ordinary_variable_preserves_name_and_positions() {
        let source = "$name;";
        let mut variable = Token::new(TokenType::Variable, "$name", 1, 1, 0);
        variable.text = Some("name".into());
        let expected = vec![
            variable,
            Token::new(TokenType::Semicolon, ";", 1, 6, 5),
            Token::new(TokenType::Eof, "", 1, 7, 6),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_braced_variable_preserves_unicode_spaces_and_positions() {
        let source = "${é name};";
        let mut variable = Token::new(TokenType::Variable, "${é name}", 1, 1, 0);
        variable.text = Some("é name".into());
        let expected = vec![
            variable,
            Token::new(TokenType::Semicolon, ";", 1, 10, 10),
            Token::new(TokenType::Eof, "", 1, 11, 11),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_special_variable_preserves_question_mark_and_positions() {
        let source = "$?;";
        let mut variable = Token::new(TokenType::Variable, "$?", 1, 1, 0);
        variable.text = Some("?".into());
        let expected = vec![
            variable,
            Token::new(TokenType::Semicolon, ";", 1, 3, 2),
            Token::new(TokenType::Eof, "", 1, 4, 3),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_empty_braced_variable_preserves_empty_name_and_positions() {
        let source = "${};";
        let mut variable = Token::new(TokenType::Variable, "${}", 1, 1, 0);
        variable.text = Some("".into());
        let expected = vec![
            variable,
            Token::new(TokenType::Semicolon, ";", 1, 4, 3),
            Token::new(TokenType::Eof, "", 1, 5, 4),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_unterminated_braced_variable_consumes_through_eof() {
        let source = "${é name\nx";
        let mut variable = Token::new(TokenType::Variable, "${é name\nx", 1, 1, 0);
        variable.text = Some("é name\nx".into());
        let expected = vec![variable, Token::new(TokenType::Eof, "", 2, 2, 11)];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_adjacent_subtract_assignment_emits_one_operator_and_positions() {
        let source = "$x-=1";
        let mut variable = Token::new(TokenType::Variable, "$x", 1, 1, 0);
        variable.text = Some("x".into());
        let expected = vec![
            variable,
            Token::new(TokenType::Operator, "-=", 1, 3, 2),
            Token::new(TokenType::Number, "1", 1, 5, 4),
            Token::new(TokenType::Eof, "", 1, 6, 5),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_spaced_subtract_assignment_emits_one_operator_and_positions() {
        let source = "$x -= 1";
        let mut variable = Token::new(TokenType::Variable, "$x", 1, 1, 0);
        variable.text = Some("x".into());
        let expected = vec![
            variable,
            Token::new(TokenType::Operator, "-=", 1, 4, 3),
            Token::new(TokenType::Number, "1", 1, 7, 6),
            Token::new(TokenType::Eof, "", 1, 8, 7),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_subtract_assignment_at_eof_preserves_operator_and_eof_positions() {
        let source = "-=";
        let expected = vec![
            Token::new(TokenType::Operator, "-=", 1, 1, 0),
            Token::new(TokenType::Eof, "", 1, 3, 2),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_hyphenated_command_preserves_name_and_positions() {
        let source = "Get-Process";
        let mut command = Token::new(TokenType::Generic, "Get-Process", 1, 1, 0);
        command.text = Some("Get-Process".into());
        let expected = vec![command, Token::new(TokenType::Eof, "", 1, 12, 11)];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_hyphenated_parameter_preserves_case_and_positions() {
        let source = "-Path";
        let mut parameter = Token::new(TokenType::Parameter, "-Path", 1, 1, 0);
        parameter.text = Some("Path".into());
        let expected = vec![parameter, Token::new(TokenType::Eof, "", 1, 6, 5)];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_named_operator_preserves_case_normalized_text_and_positions() {
        let source = "-Eq";
        let mut operator = Token::new(TokenType::Operator, "-Eq", 1, 1, 0);
        operator.text = Some("eq".into());
        let expected = vec![operator, Token::new(TokenType::Eof, "", 1, 4, 3)];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_decrement_preserves_operator_and_positions() {
        let source = "--";
        let expected = vec![
            Token::new(TokenType::Operator, "--", 1, 1, 0),
            Token::new(TokenType::Eof, "", 1, 3, 2),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_subtraction_preserves_separate_operator_and_positions() {
        let source = "$x-1";
        let mut variable = Token::new(TokenType::Variable, "$x", 1, 1, 0);
        variable.text = Some("x".into());
        let expected = vec![
            variable,
            Token::new(TokenType::Operator, "-", 1, 3, 2),
            Token::new(TokenType::Number, "1", 1, 4, 3),
            Token::new(TokenType::Eof, "", 1, 5, 4),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn tokenize_separated_hyphen_equals_preserves_two_operators_and_positions() {
        let source = "- =";
        let expected = vec![
            Token::new(TokenType::Operator, "-", 1, 1, 0),
            Token::new(TokenType::Operator, "=", 1, 3, 2),
            Token::new(TokenType::Eof, "", 1, 4, 3),
        ];

        let tokens = tokenize(source);

        assert_tokens_eq(tokens, expected);
    }

    #[test]
    fn operators_keywords_and_redirections_unchanged() {
        // Reducing allocations in `read_hyphen`, `read_word`, and `match_str`
        // must not change the token stream.
        let one = |src: &str| {
            let mut t = tokenize(src);
            t.retain(|t| !matches!(t.ty, TokenType::Eof));
            t
        };

        // Named operator: case-insensitive; value keeps case, text is lowercased.
        let t = one("-Eq");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].ty, TokenType::Operator);
        assert_eq!(
            (t[0].value.as_str(), t[0].text.as_deref()),
            ("-Eq", Some("eq"))
        );

        // Parameter keeps its original case in both value and text.
        let t = one("-Path");
        assert_eq!(t[0].ty, TokenType::Parameter);
        assert_eq!(
            (t[0].value.as_str(), t[0].text.as_deref()),
            ("-Path", Some("Path"))
        );

        // Keyword: case-insensitive; text is lowercased.
        let t = one("Function");
        assert_eq!(t[0].ty, TokenType::Keyword);
        assert_eq!(
            (t[0].value.as_str(), t[0].text.as_deref()),
            ("Function", Some("function"))
        );

        // A Verb-Noun command stays one Generic token with its original case.
        let t = one("Get-Process");
        assert_eq!(t[0].ty, TokenType::Generic);
        assert_eq!(t[0].value, "Get-Process");

        // Multi-char redirection/operator matching (match_str / starts_with).
        let t = one(">>");
        assert_eq!((t[0].ty, t[0].value.as_str()), (TokenType::Redirect, ">>"));
        let t = one("1..3");
        assert_eq!((t[1].ty, t[1].value.as_str()), (TokenType::Operator, ".."));
    }

    #[test]
    fn dollar_paren_is_a_distinct_token() {
        // Regression: `$(` must tokenize as DollarParen, not Variable + LParen.
        let toks = tokenize("$(Get-Date)");
        assert_eq!(toks[0].ty, TokenType::DollarParen, "got {:?}", toks[0]);
        assert!(!types("$(Get-Date)").contains(&TokenType::Variable));
    }

    #[test]
    fn plain_variable_still_lexes() {
        let toks = tokenize("$env:PATH");
        assert_eq!(toks[0].ty, TokenType::Variable);
        assert_eq!(toks[0].scope.as_deref(), Some("env"));
        assert_eq!(toks[0].text.as_deref(), Some("PATH"));
    }

    #[test]
    fn token_pos_is_a_byte_offset() {
        // After a 2-byte 'é', the next token's byte offset must account for it.
        let toks = tokenize("é $x");
        let var = toks.iter().find(|t| t.ty == TokenType::Variable).unwrap();
        assert_eq!(&"é $x"[var.pos..var.pos + 2], "$x");
    }

    #[test]
    fn strip_comments_leading_bom_preserves_bom_and_blanks_comment() {
        let source = "\u{feff}#x";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "\u{feff}  ");
    }

    #[test]
    fn strip_comments_tokens_leading_bom_preserves_unicode_comment_byte_ranges() {
        let source = "\u{feff}#é\r\n$x #終";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "\u{feff}   \r\n$x     ");
        assert_eq!(stripped.len(), 15);
        assert_eq!(&stripped[8..10], "$x");
    }

    #[test]
    fn strip_comments_preserves_byte_length_with_unicode() {
        // Regression: comment blanking must not corrupt multi-byte text or
        // shift byte offsets.
        let src = "café # cömment\nWrite-Output 1\n";

        let stripped = strip_comments(src);

        assert_eq!(stripped.len(), src.len());
        assert_eq!(stripped, "café           \nWrite-Output 1\n");
    }

    #[test]
    fn strip_comments_tokens_with_unicode_preserves_code_and_masks_comment_bytes() {
        let source = "café # cömment\nWrite-Output 1\n";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "café           \nWrite-Output 1\n");
        assert_eq!(stripped.len(), source.len());
    }

    #[test]
    fn strip_comments_multiline_block_preserves_bom_code_and_line_endings() {
        let source = "\u{feff}'é';<#終\r\n🙂\n#>Write-Output 1\r\n";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "\u{feff}'é';     \r\n    \n  Write-Output 1\r\n");
        assert_eq!(stripped.len(), source.len());
    }

    #[test]
    fn strip_comments_tokens_multiline_block_preserves_bom_code_and_line_endings() {
        let source = "\u{feff}'é';<#終\r\n🙂\n#>Write-Output 1\r\n";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "\u{feff}'é';     \r\n    \n  Write-Output 1\r\n");
        assert_eq!(stripped.len(), source.len());
    }

    #[test]
    fn strip_comments_line_comment_at_eof_masks_through_last_byte() {
        let source = "$x=1; #終🙂";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "$x=1;         ");
        assert_eq!(stripped.len(), source.len());
    }

    #[test]
    fn strip_comments_tokens_line_comment_at_eof_masks_through_last_byte() {
        let source = "$x=1; #終🙂";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "$x=1;         ");
        assert_eq!(stripped.len(), source.len());
    }

    #[test]
    fn strip_comments_unterminated_block_masks_through_eof_and_preserves_line_endings() {
        let source = "$x=1;<#é\r\n終\n$y=2";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "$x=1;    \r\n   \n    ");
        assert_eq!(stripped.len(), source.len());
    }

    #[test]
    fn strip_comments_tokens_unterminated_block_masks_through_eof_and_preserves_line_endings() {
        let source = "$x=1;<#é\r\n終\n$y=2";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "$x=1;    \r\n   \n    ");
        assert_eq!(stripped.len(), source.len());
    }

    #[test]
    fn strip_comments_empty_source_returns_empty_output() {
        let source = "";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "");
    }

    #[test]
    fn strip_comments_tokens_empty_source_returns_empty_output() {
        let source = "";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "");
    }

    #[test]
    fn strip_comments_bom_only_preserves_bom() {
        let source = "\u{feff}";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "\u{feff}");
    }

    #[test]
    fn strip_comments_tokens_bom_only_preserves_bom() {
        let source = "\u{feff}";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "\u{feff}");
    }

    #[test]
    fn strip_comments_comment_free_without_bom_preserves_source() {
        let source = "\tWrite-Output 'café'\r\n$x = 1\n";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "\tWrite-Output 'café'\r\n$x = 1\n");
    }

    #[test]
    fn strip_comments_tokens_comment_free_without_bom_preserves_source() {
        let source = "\tWrite-Output 'café'\r\n$x = 1\n";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "\tWrite-Output 'café'\r\n$x = 1\n");
    }

    #[test]
    fn strip_comments_quoted_markers_preserves_strings_and_masks_real_comment() {
        let source = "'# <#é#>'; \"# <#終#>\"; #x\r\n";

        let stripped = strip_comments(source);

        assert_eq!(stripped, "'# <#é#>'; \"# <#終#>\";   \r\n");
    }

    #[test]
    fn strip_comments_tokens_quoted_markers_preserves_strings_and_masks_real_comment() {
        let source = "'# <#é#>'; \"# <#終#>\"; #x\r\n";
        let tokens = tokenize(source);

        let stripped = strip_comments_tokens(&tokens, source);

        assert_eq!(stripped, "'# <#é#>'; \"# <#終#>\";   \r\n");
    }
}
