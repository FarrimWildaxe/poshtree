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
        let name: String;

        if self.peek(0) == '{' {
            self.advance();
            let mut buf = String::new();
            while self.i < self.n && self.peek(0) != '}' {
                buf.push(self.advance());
            }
            self.match_str("}");
            name = buf;
        } else if "?^$_".contains(self.peek(0)) {
            name = self.advance().to_string();
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
            name = buf;
        }
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
        loop {
            if self.i >= self.n {
                break;
            }
            let c = self.peek(0);
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
                } else {
                    self.advance();
                    break;
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
/// Byte offsets, line counts and column positions are preserved so any
/// line/column computed against the result still matches the original file.
pub fn strip_comments(source: &str) -> String {
    strip_comments_tokens(&tokenize(source), source)
}

/// [`strip_comments`] over an already-computed token stream, so callers that
/// have tokenized do not pay for a second lex pass.
pub fn strip_comments_tokens(tokens: &[Token], source: &str) -> String {
    let mut bytes = source.as_bytes().to_vec();

    for tok in tokens {
        if tok.ty != TokenType::Comment {
            continue;
        }
        let start = tok.pos; // byte offset
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

    #[test]
    fn operators_keywords_and_redirections_unchanged() {
        // Guards the read_hyphen / read_word / match_str cleanups: the cheaper
        // allocation paths must produce identical tokens.
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
    fn strip_comments_preserves_byte_length_with_unicode() {
        // Regression: comment blanking must not corrupt multi-byte text or
        // shift byte offsets.
        let src = "café # cömment\nWrite-Output 1\n";
        let stripped = strip_comments(src);
        assert_eq!(stripped.len(), src.len());
        assert!(stripped.starts_with("café "));
        assert!(!stripped.contains("cömment"));
        assert!(stripped.contains("Write-Output 1"));
    }
}
