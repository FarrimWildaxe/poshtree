//! A native recursive-descent parser for PowerShell, working directly on v2
//! tokens and producing a [`Node`] tree with native [`TokenRange`]s.
//!
//! This is the v2 counterpart to the v1 parser. It depends on no `v1` code,
//! which is what lets the `v2` feature stand alone. Because v2 keeps newlines
//! as trivia rather than tokens, statement boundaries are found with
//! [`Token::starts_line`](crate::v2::Token::starts_line) and `;`.
//!
//! Error handling mirrors the v1 parser in spirit: errors are collected into
//! [`ParseOutput::errors`] and the parser keeps going, producing
//! [`NodeKind::Error`] nodes where it cannot proceed, so a tree always comes
//! back.
//!
//! # Add-Type source cache
//!
//! Inline C# extraction accepts eligible string literals directly or through a
//! parser-local cache of plain, unscoped variable assignments. Recognized writes
//! (`=`, `+=`, `-=`, `*=`, `/=`, `%=`, `??=`) invalidate that variable's entry
//! after parsing the RHS, so an `Add-Type` on the RHS can still use the old entry.
//! Prefix/postfix `++` and `--` on such a variable also invalidate its entry.
//! Only `=` with an eligible string-literal RHS installs or restores an entry;
//! writes to another variable leave it alone.
//!
//! Single-quoted strings and single-quoted here-strings are eligible. For
//! double-quoted forms, eligibility conservatively rejects any unescaped `$`
//! in the raw body; a backtick skips the next character. This does not rely on
//! the interpolation-parts list, which does not cover every reference.
//! Unknown variables and rejected strings do not supply confirmed C# metadata.
//! Missing metadata is neither a safety verdict nor proof that `Add-Type` is
//! absent or cannot execute.
//!
//! This is a bounded syntactic cache in linear parser traversal order, not
//! scope, branch, function, or dataflow analysis. It does not track indirect
//! writes or alternate runtime values. Extracted text remains the raw body
//! with delimiters removed and its source span, not a fully decoded PowerShell
//! runtime string or a general decoded-text-to-source map.
//!
//! # Example
//!
//! ```
//! use poshtree::v2::parse;
//! let out = poshtree::v2::parser::parse("Get-ChildItem -Path . | Sort-Object\n");
//! assert!(out.errors.is_empty());
//! ```

use super::lexer::{lex, scan_subexpression_end, variable_reference_len};
use super::span::{Span, TokenRange};
use super::tokens::TokenKind as T;
use super::tokens::{LexError, Token};

use super::ast::{Node, NodeKind, StringKind};

/// A parse error: a message and the source span it concerns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Human-readable description.
    pub message: String,
    /// The byte span the error concerns.
    pub span: Span,
}

/// The result of [`parse`]: the tree root (always a [`NodeKind::Script`]),
/// any errors encountered (lexer errors first, then parser errors), and the
/// token stream the tree was built from.
#[derive(Debug, Clone)]
pub struct ParseOutput {
    /// The script-block root.
    pub script: Node,
    /// Recoverable errors; an empty vector means a clean parse.
    pub errors: Vec<ParseError>,
    /// The tokens the tree was built from, handed back so a caller that also
    /// needs them (a formatter, an analyzer) does not have to lex a second
    /// time. With [`parse`] these come from the internal [`lex`] call; with
    /// [`parse_tokens`] they are the tokens passed in.
    pub tokens: Vec<Token>,
}

/// Parses PowerShell source into a v2 syntax tree.
///
/// This lexes internally. If you have already lexed `src` (to inspect tokens,
/// reconstruct, or check lex errors), call [`parse_tokens`] instead to avoid
/// lexing twice; either way the tokens come back on [`ParseOutput::tokens`].
///
/// Error spans use UTF-8 byte offsets into the original input, including any
/// leading BOM bytes.
pub fn parse(src: &str) -> ParseOutput {
    parse_at_depth(src, 0)
}

fn parse_at_depth(src: &str, initial_depth: u32) -> ParseOutput {
    let lexed = lex(src);
    let lex_errors: Vec<ParseError> = lexed.errors.iter().map(ParseError::from_lex).collect();
    let mut out = parse_tokens_at_depth(src, lexed.tokens, initial_depth);
    if !lex_errors.is_empty() {
        // Lexer errors come first, ahead of the parser errors already in `out`.
        let mut errors = lex_errors;
        errors.append(&mut out.errors);
        out.errors = errors;
    }
    out
}

/// Parses tokens that were already produced by [`lex`] over the same `src`,
/// without lexing again.
///
/// `src` must be the exact string the tokens were lexed from: the parser
/// reads it through the token spans (for example to capture a bracketed type
/// name spanning several tokens), so a mismatched `src` gives wrong spans or
/// panics. Lexer errors are not represented here; inspect [`LexOutput::errors`]
/// from your own `lex` call. The returned [`ParseOutput`] carries the tokens
/// back unchanged.
///
/// ```
/// use poshtree::v2::{lex, parse_tokens, reconstruct};
///
/// let src = "Get-ChildItem | Sort-Object\n";
/// let lexed = lex(src); // lex once
/// let out = parse_tokens(src, lexed.tokens);
/// assert!(out.errors.is_empty());
/// assert_eq!(reconstruct(&out.tokens), src); // tokens handed back, no re-lex
/// ```
///
/// [`LexOutput::errors`]: super::lexer::LexOutput::errors
pub fn parse_tokens(src: &str, tokens: Vec<Token>) -> ParseOutput {
    parse_tokens_at_depth(src, tokens, 0)
}

fn parse_tokens_at_depth(src: &str, tokens: Vec<Token>, initial_depth: u32) -> ParseOutput {
    let mut parser = Parser {
        tokens,
        src,
        pos: 0,
        errors: Vec::new(),
        depth: initial_depth,
        in_command_argument: false,
        suppress_array_comma: false,
        overflowed: false,
        vars: std::collections::HashMap::new(),
    };
    let body = parser.parse_block_body(&[]);
    let end = parser.pos;
    let script = parser.make(NodeKind::Script(body), 0, end);
    let Parser { tokens, errors, .. } = parser;
    ParseOutput {
        script,
        errors,
        tokens,
    }
}

impl ParseError {
    fn from_lex(e: &LexError) -> Self {
        ParseError {
            message: e.to_string(),
            span: e.span,
        }
    }
}

struct Parser<'a> {
    tokens: Vec<Token>,
    src: &'a str,
    pos: usize,
    errors: Vec<ParseError>,
    depth: u32,
    /// True while parsing a top-level command-argument expression. The glued
    /// `*` reinterpretation is suppressed here, because in argument position
    /// `$x*2` is a glob continuation, not multiplication. Grouping constructs
    /// (parens, script blocks, statements) clear it, since they re-enter pure
    /// expression context.
    in_command_argument: bool,
    /// When set, `parse_array_literal` does not collect a trailing comma into an
    /// array, so the comma stays available as a separator. Used for parameter
    /// defaults, where a top-level comma separates parameters rather than
    /// building an array. Reset inside any grouping (parens, `@()`, `@{}`,
    /// index) so commas there still build arrays.
    suppress_array_comma: bool,
    /// Set once the depth limit is hit. While set, the recursive entry points
    /// stop descending and drain one token at a time, so recovery from
    /// pathological nesting stays linear in the token count instead of letting
    /// a parent re-drive the parse of the levels beneath it.
    overflowed: bool,
    /// Eligible string bodies and source spans for plain, unscoped variables in
    /// parser traversal order, keyed ASCII-case-insensitively. Recognized writes
    /// invalidate entries; only eligible literal `=` assignments restore them.
    /// Not a scope model.
    vars: std::collections::HashMap<String, (String, Span)>,
}

/// Guards against pathological nesting blowing the stack on adversarial input.
const MAX_DEPTH: u32 = 200;

impl Parser<'_> {
    // Cursor

    fn kind(&self) -> T {
        self.tokens[self.pos].kind
    }

    fn value(&self) -> &str {
        &self.tokens[self.pos].value
    }

    fn lower(&self) -> String {
        self.value().to_ascii_lowercase()
    }

    fn at(&self, k: T) -> bool {
        self.kind() == k
    }

    fn at_end(&self) -> bool {
        self.at(T::Eof)
    }

    fn at_kw(&self, kw: &str) -> bool {
        self.at(T::Keyword) && self.tokens[self.pos].value_eq_ci(kw)
    }

    fn at_op(&self, op: &str) -> bool {
        self.at(T::Operator) && self.value() == op
    }

    /// Whether a real line break sits immediately before the current token.
    /// A newline between two tokens may be attached to the previous token's
    /// trailing trivia or this token's leading trivia (a here-string, for one,
    /// keeps its closing newline as trailing trivia), so a correct statement
    /// boundary check looks at both sides. Backtick continuations are
    /// `LineContinuation`, not `Newline`, so they correctly do not count.
    fn starts_line(&self) -> bool {
        let here = self.tokens[self.pos].starts_line();
        let prev = self
            .pos
            .checked_sub(1)
            .and_then(|i| self.tokens.get(i))
            .is_some_and(|t| t.ends_line());
        here || prev
    }

    /// Consumes the current token and returns its index. Never advances past
    /// the final `Eof` token.
    fn bump(&mut self) -> usize {
        let i = self.pos;
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        i
    }

    fn expect(&mut self, k: T, what: &str) {
        if self.at(k) {
            self.bump();
        } else {
            self.error(&format!("expected {what}"));
        }
    }

    fn error(&mut self, message: &str) {
        let span = self.tokens[self.pos].span;
        self.errors.push(ParseError {
            message: message.to_string(),
            span,
        });
    }

    // Node construction

    fn make(&self, kind: NodeKind, first: usize, end: usize) -> Node {
        Node {
            kind,
            span: self.span_of(first, end),
            range: TokenRange { first, end },
        }
    }

    fn span_of(&self, first: usize, end: usize) -> Span {
        let n = self.tokens.len();
        if first >= n {
            let last = self.tokens.last().map(|t| t.span.end).unwrap_or(0);
            return Span::new(last, last);
        }
        let s = self.tokens[first].span.start;
        let e = if end > first && end <= n {
            self.tokens[end - 1].span.end
        } else {
            s
        };
        Span::new(s, e.max(s))
    }

    /// Consumes one token as a leaf node of the given kind.
    fn leaf(&mut self, kind: NodeKind) -> Node {
        let i = self.bump();
        self.make(kind, i, i + 1)
    }

    // Predicates

    fn is_value_start(&self) -> bool {
        match self.kind() {
            T::Variable
            | T::Number
            | T::StringSq
            | T::StringDq
            | T::HereStringSq
            | T::HereStringDq
            | T::LParen
            | T::DollarParen
            | T::AtParen
            | T::AtBrace
            | T::LBrace
            | T::LBracket => true,
            T::Comma => true,
            T::Operator => is_unary_operator_word(&self.lower()),
            _ => false,
        }
    }

    fn is_command_start(&self) -> bool {
        if matches!(self.kind(), T::Generic | T::Amp | T::Dot | T::Keyword) {
            return true;
        }
        // `%` and `?` are the ForEach-Object / Where-Object aliases. The lexer
        // always emits them as operators (it is context-free by design), but in
        // command position they name a command. Expression uses (modulo,
        // ternary) are unaffected: those sites never ask is_command_start.
        self.at(T::Operator) && matches!(self.value(), "%" | "?")
    }

    /// Tokens that can begin a command argument, mirroring v1's set. Used to
    /// decide parameter-argument and redirection-target binding.
    fn starts_argument(&self) -> bool {
        matches!(
            self.kind(),
            T::Variable
                | T::Number
                | T::StringSq
                | T::StringDq
                | T::HereStringSq
                | T::HereStringDq
                | T::Generic
                | T::LParen
                | T::LBrace
                | T::LBracket
                | T::AtParen
                | T::AtBrace
                | T::DollarParen
                | T::Keyword
                | T::Parameter
        )
    }

    fn at_command_boundary(&self) -> bool {
        self.at_end()
            || matches!(
                self.kind(),
                T::Pipe | T::Semicolon | T::RParen | T::RBrace | T::RBracket
            )
            || (self.at(T::Operator) && matches!(self.value(), "&&" | "||"))
            || self.starts_line()
    }

    fn enter(&mut self) -> bool {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.overflowed = true;
        }
        self.overflowed
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    // Statements

    fn skip_separators(&mut self) {
        while self.at(T::Semicolon) {
            self.bump();
        }
    }

    fn parse_block_body(&mut self, closers: &[T]) -> Vec<Node> {
        let mut stmts = Vec::new();
        // Like v1: an optional `param(...)` block, only as the leading statement.
        self.skip_separators();
        if self.at_kw("param") && self.peek_is_lparen() {
            stmts.push(self.parse_param_block());
        }
        loop {
            self.skip_separators();
            if self.at_end() || closers.contains(&self.kind()) {
                break;
            }
            let before = self.pos;
            let stmt = self.parse_statement();
            stmts.push(stmt);
            if self.pos == before {
                // No progress: report and skip a token so we cannot loop.
                self.error("unexpected token");
                self.bump();
            }
        }
        stmts
    }

    fn parse_braced_block(&mut self) -> Node {
        let start = self.pos;
        self.expect(T::LBrace, "'{'");
        let body = self.parse_block_body(&[T::RBrace]);
        self.expect(T::RBrace, "'}'");
        self.make(NodeKind::Script(body), start, self.pos)
    }

    fn parse_statement(&mut self) -> Node {
        if self.enter() {
            self.leave();
            // Past the depth limit, consume one token and return an error node
            // spanning it. Consuming keeps the enclosing block loop making
            // progress without its no-progress path logging a second error per
            // token, so a pathological input drains linearly instead of
            // emitting several errors per level.
            let start = self.pos;
            self.error("statement nested too deeply");
            if !self.at_end() {
                self.bump();
            }
            return self.make(NodeKind::Error("too deep".into()), start, self.pos);
        }
        let saved = self.in_command_argument;
        self.in_command_argument = false;
        let saved_comma = self.suppress_array_comma;
        self.suppress_array_comma = false;
        let node = self.parse_statement_inner();
        self.in_command_argument = saved;
        self.suppress_array_comma = saved_comma;
        self.leave();
        node
    }

    fn parse_statement_inner(&mut self) -> Node {
        // A loop/switch label: `:outer foreach (...) { ... }`. The lexer emits
        // `:` then a word. PowerShell allows a label only before a loop or
        // `switch`, so require one of those keywords two tokens ahead before
        // wrapping; otherwise `:` is left to parse as it otherwise would, and a
        // bare command like `:lbl Get-Process` is not mistaken for a label.
        if self.at_op(":") && self.peek_kind() == T::Generic {
            let after_label = self.peek_n(2);
            let is_labelable = after_label.kind == T::Keyword
                && matches!(
                    after_label.value.to_ascii_lowercase().as_str(),
                    "for" | "foreach" | "while" | "do" | "switch"
                );
            if is_labelable {
                let start = self.pos;
                self.bump(); // ':'
                let label = self.value().to_string();
                let label_span = self.tokens[self.pos].span;
                self.bump(); // label word
                let statement = self.parse_statement();
                return self.make(
                    NodeKind::Labeled {
                        label,
                        label_span,
                        statement: Box::new(statement),
                    },
                    start,
                    self.pos,
                );
            }
        }
        if self.at(T::Keyword) {
            match self.lower().as_str() {
                "if" => return self.parse_if(),
                "while" => return self.parse_while(),
                "do" => return self.parse_do(),
                "for" => return self.parse_for(),
                "foreach" if self.peek_is_lparen() => return self.parse_foreach(),
                "switch" => return self.parse_switch(),
                "function" => return self.parse_function(false),
                "filter" => return self.parse_function(true),
                "workflow" => return self.parse_function(false),
                "try" => return self.parse_try(),
                "trap" | "data" | "dynamicparam" => return self.parse_trailing_block(),
                "begin" | "process" | "end" if self.peek_is_lbrace() => {
                    let start = self.pos;
                    self.bump();
                    let body = self.parse_braced_block();
                    return self.make(
                        NodeKind::ScriptBlockExpression(Box::new(body)),
                        start,
                        self.pos,
                    );
                }
                "class" => return self.parse_class(),
                "enum" => return self.parse_enum(),
                "using" => return self.parse_using(),
                "return" | "throw" | "break" | "continue" | "exit" => return self.parse_flow(),
                _ => {}
            }
        }
        self.parse_pipeline_statement()
    }

    fn peek_kind(&self) -> T {
        let next = (self.pos + 1).min(self.tokens.len() - 1);
        self.tokens[next].kind
    }

    /// The token `n` positions ahead of the cursor (0 = current), clamped to
    /// the final token.
    fn peek_n(&self, n: usize) -> &Token {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        &self.tokens[i]
    }

    fn peek_is_lparen(&self) -> bool {
        self.peek_kind() == T::LParen
    }

    fn peek_is_lbrace(&self) -> bool {
        self.peek_kind() == T::LBrace
    }

    fn parse_pipeline_statement(&mut self) -> Node {
        let start = self.pos;
        if self.is_value_start() {
            // Possibly an assignment; parse a target expression first.
            let expr = self.parse_expression();
            if self.at(T::Operator) && is_assignment_op(self.value()) {
                let op = self.value().to_string();
                self.bump();
                // The value may be a full statement (`$x = if (...) {...}`).
                let value = self.parse_statement();
                // Keep the old value available to the RHS, then discard stale
                // source before any later Add-Type command can use it.
                self.invalidate_cached_variable(&expr);
                // Only an eligible literal `=` can restore the syntactic source cache.
                if op == "=" {
                    if let (
                        NodeKind::Variable(raw),
                        NodeKind::StringLiteral {
                            value: sv, kind, ..
                        },
                    ) = (&expr.kind, &value.kind)
                    {
                        let n = raw.trim_start_matches('$');
                        if !n.contains(':') && !n.starts_with('{') && is_constant_string(sv, *kind)
                        {
                            self.vars.insert(
                                n.to_ascii_lowercase(),
                                (
                                    string_inner(sv, *kind).to_string(),
                                    body_span(sv, *kind, value.span),
                                ),
                            );
                        }
                    }
                }
                return self.make(
                    NodeKind::Assignment {
                        target: Box::new(expr),
                        op,
                        value: Box::new(value),
                    },
                    start,
                    self.pos,
                );
            }
            let pipeline = self.collect_pipeline(expr, start);
            return self.finish_chain(pipeline, start);
        }
        let first = self.parse_command();
        let pipeline = self.collect_pipeline(first, start);
        self.finish_chain(pipeline, start)
    }

    fn invalidate_cached_variable(&mut self, node: &Node) {
        if let NodeKind::Variable(raw) = &node.kind {
            let n = raw.trim_start_matches('$');
            if !n.contains(':') && !n.starts_with('{') {
                self.vars.remove(&n.to_ascii_lowercase());
            }
        }
    }

    /// Collects `|`-separated elements into a [`NodeKind::Pipeline`] (a single
    /// element is returned unwrapped).
    fn collect_pipeline(&mut self, first: Node, start: usize) -> Node {
        let mut elements = vec![first];
        while self.at(T::Pipe) {
            self.bump();
            let element = if self.is_command_start() {
                self.parse_command()
            } else {
                self.parse_expression()
            };
            elements.push(element);
        }
        if elements.len() == 1 {
            elements.pop().unwrap()
        } else {
            self.make(NodeKind::Pipeline(elements), start, self.pos)
        }
    }

    /// Folds `&&` / `||` chains left-associatively, matching v1: each right
    /// side is a single pipeline, not another chain.
    fn finish_chain(&mut self, mut node: Node, start: usize) -> Node {
        while self.at(T::Operator) && matches!(self.value(), "&&" | "||") {
            let op = self.value().to_string();
            self.bump();
            let r_start = self.pos;
            let rfirst = if self.is_command_start() || !self.is_value_start() {
                self.parse_command()
            } else {
                self.parse_expression()
            };
            let right = self.collect_pipeline(rfirst, r_start);
            node = self.make(
                NodeKind::PipelineChain {
                    left: Box::new(node),
                    op,
                    right: Box::new(right),
                },
                start,
                self.pos,
            );
        }
        node
    }

    fn parse_command(&mut self) -> Node {
        let start = self.pos;
        let (name, invocation) = self.parse_command_name();
        let mut elements = Vec::new();
        let mut redirections = Vec::new();
        loop {
            if self.at_command_boundary() {
                break;
            }
            if self.at(T::Redirect) {
                redirections.push(self.parse_redirection());
            } else if self.at(T::Parameter) {
                elements.push(self.parse_parameter());
            } else {
                elements.push(self.parse_command_argument());
            }
        }
        let csharp = self.maybe_csharp(&name, invocation, &elements, start);
        self.make(
            NodeKind::Command {
                name: Box::new(name),
                invocation,
                elements,
                redirections,
                csharp,
            },
            start,
            self.pos,
        )
    }

    /// For an `Add-Type` command, locate the inline C# and attach a
    /// [`CSharpMemberDef`](NodeKind::CSharpMemberDef) node, mirroring v1.
    fn maybe_csharp(
        &self,
        name: &Node,
        invocation: bool,
        elements: &[Node],
        start: usize,
    ) -> Option<Box<Node>> {
        if invocation {
            return None;
        }
        let NodeKind::BareWord(cmd) = &name.kind else {
            return None;
        };
        if !cmd.eq_ignore_ascii_case("add-type") {
            return None;
        }
        let (code, parameter, code_span) = find_csharp_code(elements, &self.vars)?;
        let mut def = parse_csharp_member_def(&code);
        def.parameter = parameter;
        def.code_span = code_span;
        Some(Box::new(Node {
            kind: NodeKind::CSharpMemberDef(def),
            span: self.span_of(start, self.pos),
            range: TokenRange {
                first: start,
                end: self.pos,
            },
        }))
    }

    /// Parses a command name, returning the name node and whether it was an
    /// `&`/`.` call-operator invocation (in which case the name is an
    /// expression and counts as a child).
    fn parse_command_name(&mut self) -> (Node, bool) {
        if self.at(T::Amp) || self.at(T::Dot) {
            self.bump();
            if self.at_command_boundary() {
                let i = self.pos;
                return (self.make(NodeKind::BareWord(String::new()), i, i), true);
            }
            return (self.parse_command_argument(), true);
        }
        let v = self.value().to_string();
        (self.leaf(NodeKind::BareWord(v)), false)
    }

    fn parse_parameter(&mut self) -> Node {
        let start = self.pos;
        let name = strip_param_dash(self.value());
        self.bump();
        // Bind a following argument as v1 does (`-Path value`): never bind to
        // another parameter, only when an argument actually starts, and only on
        // the same line. A newline ends the command (it is a command boundary),
        // so a trailing switch like `-PassThru` followed by a newline takes no
        // argument and does not swallow the next statement.
        let argument = if self.starts_argument() && !self.at(T::Parameter) && !self.starts_line() {
            Some(Box::new(self.parse_command_argument()))
        } else {
            None
        };
        self.make(
            NodeKind::CommandParameter { name, argument },
            start,
            self.pos,
        )
    }

    fn parse_command_argument(&mut self) -> Node {
        if self.arg_is_expression() {
            let saved = self.in_command_argument;
            self.in_command_argument = true;
            let node = self.parse_expression();
            self.in_command_argument = saved;
            node
        } else {
            let v = self.value().to_string();
            self.leaf(NodeKind::BareWord(v))
        }
    }

    /// In argument position only these introduce an expression; everything
    /// else (including bare numbers and words) is a bareword, as in v1.
    fn arg_is_expression(&self) -> bool {
        matches!(
            self.kind(),
            T::Variable
                | T::LParen
                | T::AtParen
                | T::AtBrace
                | T::DollarParen
                | T::LBrace
                | T::StringSq
                | T::StringDq
                | T::HereStringSq
                | T::HereStringDq
                | T::LBracket
        )
    }

    fn parse_redirection(&mut self) -> Node {
        let start = self.pos;
        let op = self.value().to_string();
        self.bump();
        // Stream merges such as `2>&1` carry their destination in the operator
        // and take no target; a file redirection binds the next argument.
        let target = if !op.contains('&') && self.starts_argument() {
            Some(Box::new(self.parse_command_argument()))
        } else {
            None
        };
        self.make(NodeKind::Redirection { op, target }, start, self.pos)
    }

    fn parse_if(&mut self) -> Node {
        let start = self.pos;
        self.bump(); // if
        let mut conditions = Vec::new();
        let mut blocks = Vec::new();
        self.expect(T::LParen, "'('");
        conditions.push(self.parse_pipeline_statement());
        self.expect(T::RParen, "')'");
        blocks.push(self.parse_braced_block());
        while self.at_kw("elseif") {
            self.bump();
            self.expect(T::LParen, "'('");
            conditions.push(self.parse_pipeline_statement());
            self.expect(T::RParen, "')'");
            blocks.push(self.parse_braced_block());
        }
        let else_block = if self.at_kw("else") {
            self.bump();
            Some(Box::new(self.parse_braced_block()))
        } else {
            None
        };
        self.make(
            NodeKind::If {
                conditions,
                blocks,
                else_block,
            },
            start,
            self.pos,
        )
    }

    fn parse_while(&mut self) -> Node {
        let start = self.pos;
        self.bump();
        self.expect(T::LParen, "'('");
        let condition = self.parse_pipeline_statement();
        self.expect(T::RParen, "')'");
        let body = self.parse_braced_block();
        self.make(
            NodeKind::While {
                condition: Box::new(condition),
                body: Box::new(body),
            },
            start,
            self.pos,
        )
    }

    fn parse_do(&mut self) -> Node {
        let start = self.pos;
        self.bump();
        let body = self.parse_braced_block();
        let until = self.at_kw("until");
        if self.at_kw("while") || self.at_kw("until") {
            self.bump();
        } else {
            self.error("expected 'while' or 'until'");
        }
        self.expect(T::LParen, "'('");
        let condition = self.parse_pipeline_statement();
        self.expect(T::RParen, "')'");
        self.make(
            NodeKind::DoWhile {
                body: Box::new(body),
                condition: Box::new(condition),
                until,
            },
            start,
            self.pos,
        )
    }

    fn parse_for(&mut self) -> Node {
        let start = self.pos;
        self.bump();
        self.expect(T::LParen, "'('");
        let init = self.parse_optional_clause(T::Semicolon);
        self.expect(T::Semicolon, "';'");
        let condition = self.parse_optional_clause(T::Semicolon);
        self.expect(T::Semicolon, "';'");
        let update = self.parse_optional_clause(T::RParen);
        self.expect(T::RParen, "')'");
        let body = self.parse_braced_block();
        self.make(
            NodeKind::For {
                init,
                condition,
                update,
                body: Box::new(body),
            },
            start,
            self.pos,
        )
    }

    fn parse_optional_clause(&mut self, terminator: T) -> Option<Box<Node>> {
        if self.at(terminator) {
            None
        } else {
            Some(Box::new(self.parse_pipeline_statement()))
        }
    }

    fn parse_foreach(&mut self) -> Node {
        let start = self.pos;
        self.bump();
        self.expect(T::LParen, "'('");
        let variable = if self.at(T::Variable) {
            let raw = self.value().to_string();
            self.leaf(NodeKind::Variable(raw))
        } else {
            self.error("expected loop variable");
            let i = self.pos;
            self.make(NodeKind::Variable(String::new()), i, i)
        };
        if self.at_kw("in") {
            self.bump();
        } else {
            self.error("expected 'in'");
        }
        let iterable = self.parse_pipeline_statement();
        self.expect(T::RParen, "')'");
        let body = self.parse_braced_block();
        self.make(
            NodeKind::ForEach {
                variable: Box::new(variable),
                iterable: Box::new(iterable),
                body: Box::new(body),
            },
            start,
            self.pos,
        )
    }

    fn parse_switch(&mut self) -> Node {
        let start = self.pos;
        self.bump();
        // Options before the input: `-Regex`, `-Wildcard`, `-CaseSensitive`,
        // `-Exact` are switches; `-File <path>` takes a path that becomes the
        // input. Each is captured as a CommandParameter. Only `-File` consumes
        // a following value, so the others do not swallow the `(...)` input.
        let mut flags = Vec::new();
        let mut file_input = None;
        while self.at(T::Parameter) {
            let p_start = self.pos;
            let name = self.value().trim_start_matches('-').to_string();
            self.bump();
            let is_file = name.eq_ignore_ascii_case("file");
            // `-File <path>` consumes the path, which becomes the switch input.
            // The flag itself records only its presence, so the path is not
            // duplicated as both the flag's argument and the input.
            let path = if is_file && self.is_value_start() {
                Some(self.parse_command_argument())
            } else {
                None
            };
            let flag = self.make(
                NodeKind::CommandParameter {
                    name,
                    argument: None,
                },
                p_start,
                self.pos,
            );
            flags.push(flag);
            if let Some(p) = path {
                file_input = Some(p);
            }
        }
        let input = if let Some(f) = file_input {
            // `-File <path>`: the path is the input.
            f
        } else if self.at(T::LParen) {
            self.bump();
            let e = self.parse_pipeline_statement();
            self.expect(T::RParen, "')'");
            e
        } else {
            self.parse_command_argument()
        };
        // Case bodies are parsed as ordinary statements inside the braces;
        // command extraction and ranges still work without modeling labels.
        let body = self.parse_braced_block();
        self.make(
            NodeKind::Switch {
                flags,
                input: Box::new(input),
                cases: vec![body],
            },
            start,
            self.pos,
        )
    }

    fn parse_function(&mut self, filter: bool) -> Node {
        let start = self.pos;
        self.bump();
        let name_pos = self.pos;
        let name = if self.at(T::Generic) || self.at(T::Keyword) {
            let v = self.value().to_string();
            self.bump();
            v
        } else {
            self.error("expected function name");
            String::new()
        };
        let name_span = self.span_of(name_pos, self.pos);
        // An optional `function f(...)` parameter list before the body. The
        // alternative `param(...)` block lives inside the body and is handled
        // there; a function uses one form or the other.
        let parameters = if self.at(T::LParen) {
            self.bump();
            self.parse_param_decl_list()
        } else {
            Vec::new()
        };
        let body = self.parse_braced_block();
        self.make(
            NodeKind::Function {
                name,
                name_span,
                filter,
                parameters,
                body: Box::new(body),
            },
            start,
            self.pos,
        )
    }

    fn parse_try(&mut self) -> Node {
        let start = self.pos;
        self.bump();
        let body = self.parse_braced_block();
        let mut catches = Vec::new();
        while self.at_kw("catch") {
            let c_start = self.pos;
            self.bump();
            // Optional exception type filters before the body.
            while !self.at(T::LBrace) && !self.at_end() && !self.at_command_boundary_for_catch() {
                self.bump();
            }
            let c_body = self.parse_braced_block();
            catches.push(self.make(
                NodeKind::Catch {
                    body: Box::new(c_body),
                },
                c_start,
                self.pos,
            ));
        }
        let finally_block = if self.at_kw("finally") {
            self.bump();
            Some(Box::new(self.parse_braced_block()))
        } else {
            None
        };
        self.make(
            NodeKind::Try {
                body: Box::new(body),
                catches,
                finally_block,
            },
            start,
            self.pos,
        )
    }

    fn at_command_boundary_for_catch(&self) -> bool {
        // Stop skipping catch type filters at a brace or a clear terminator.
        matches!(self.kind(), T::RBrace | T::Semicolon)
    }

    /// Parses a keyword whose body is a trailing `{ block }` after an optional
    /// header: `trap [type] {…}`, `data [name] [-opts] {…}`, `dynamicparam
    /// {…}`. Matches v1 by representing the result as a `ScriptBlockExpression`.
    fn parse_trailing_block(&mut self) -> Node {
        let start = self.pos;
        self.bump(); // keyword
        let mut guard = 0;
        while !self.at(T::LBrace) && !self.at_command_boundary() && !self.at_end() && guard < 64 {
            self.bump();
            guard += 1;
        }
        let body = if self.at(T::LBrace) {
            self.parse_braced_block()
        } else {
            self.make(NodeKind::Script(Vec::new()), self.pos, self.pos)
        };
        self.make(
            NodeKind::ScriptBlockExpression(Box::new(body)),
            start,
            self.pos,
        )
    }

    fn parse_using(&mut self) -> Node {
        let start = self.pos;
        self.bump(); // using
        let kind = if matches!(self.kind(), T::Generic | T::Keyword) {
            let k = self.lower();
            if matches!(
                k.as_str(),
                "namespace" | "module" | "assembly" | "type" | "command"
            ) {
                self.bump();
                k
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        // The name is everything to the end of the line, not one token: an
        // assembly or module name can contain dots, slashes, or version
        // numbers that the lexer splits into several tokens, and v1 reads to
        // end of line for the same reason.
        let mut name = String::new();
        while !self.at_line_end() && !self.at_end() {
            name.push_str(self.value());
            self.bump();
        }
        self.make(
            NodeKind::Using {
                kind,
                name: name.trim().to_string(),
            },
            start,
            self.pos,
        )
    }

    fn at_line_end(&self) -> bool {
        self.at_end()
            || self.at(T::Semicolon)
            || self.starts_line()
            || matches!(self.kind(), T::RBrace | T::RParen)
    }

    /// Reads a declaration name token, returning its text and source span. The
    /// span is empty (start == end at the current position) when no name is
    /// present, so rename tooling can skip it harmlessly.
    fn read_name_token_spanned(&mut self) -> (String, Span) {
        if matches!(self.kind(), T::Generic | T::Keyword) {
            let span = self.tokens[self.pos].span;
            let v = self.value().to_string();
            self.bump();
            (v, span)
        } else {
            self.error("expected name");
            let at = self.tokens.get(self.pos).map_or(0, |t| t.span.start);
            (String::new(), Span::new(at, at))
        }
    }

    fn read_dotted_name(&mut self) -> String {
        let mut s = String::new();
        while matches!(self.kind(), T::Generic | T::Keyword | T::Dot) && !self.starts_line() {
            s.push_str(self.value());
            self.bump();
        }
        s
    }

    fn parse_enum(&mut self) -> Node {
        let start = self.pos;
        self.bump(); // enum
        let (name, name_span) = self.read_name_token_spanned();
        let mut base = String::new();
        if self.at_op(":") {
            self.bump();
            base = if self.at(T::LBracket) {
                self.parse_bracket_type().0
            } else {
                self.read_dotted_name()
            };
        }
        self.expect(T::LBrace, "'{'");
        let mut members = Vec::new();
        loop {
            while self.at(T::Comma) {
                self.bump();
            }
            if self.at(T::RBrace) || self.at_end() {
                break;
            }
            if !matches!(self.kind(), T::Generic | T::Keyword) {
                self.bump(); // skip the unexpected token
                continue;
            }
            let m_start = self.pos;
            let mname = self.value().to_string();
            let member_name = self.leaf(NodeKind::BareWord(mname));
            if self.at_op("=") {
                self.bump();
                let val = self.parse_expression();
                members.push(self.make(
                    NodeKind::Assignment {
                        target: Box::new(member_name),
                        op: "=".into(),
                        value: Box::new(val),
                    },
                    m_start,
                    self.pos,
                ));
            } else {
                members.push(member_name);
            }
        }
        self.expect(T::RBrace, "'}'");
        self.make(
            NodeKind::EnumDefinition {
                name,
                name_span,
                base,
                members,
            },
            start,
            self.pos,
        )
    }

    fn parse_class(&mut self) -> Node {
        let start = self.pos;
        self.bump(); // class
        let (name, name_span) = self.read_name_token_spanned();
        let mut bases = Vec::new();
        if self.at_op(":") {
            self.bump();
            loop {
                if self.at(T::LBracket) {
                    bases.push(self.parse_bracket_type().0);
                } else if matches!(self.kind(), T::Generic | T::Keyword) {
                    bases.push(self.read_dotted_name());
                } else {
                    break;
                }
                if self.at(T::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(T::LBrace, "'{'");
        let mut members = Vec::new();
        loop {
            self.skip_separators();
            if self.at(T::RBrace) || self.at_end() {
                break;
            }
            let before = self.pos;
            if let Some(m) = self.parse_class_member(&name) {
                members.push(m);
            }
            if self.pos == before {
                self.bump(); // stall guard
            }
        }
        self.expect(T::RBrace, "'}'");
        self.make(
            NodeKind::ClassDefinition {
                name,
                name_span,
                bases,
                members,
            },
            start,
            self.pos,
        )
    }

    /// Skips a class member's leading attributes (`[Attr(...)]`), modifiers
    /// (`static`, `hidden`), and captures an optional `[type]`.
    fn parse_member_prefix(&mut self) -> String {
        let mut type_name = String::new();
        loop {
            if matches!(self.kind(), T::Generic | T::Keyword) {
                let w = self.lower();
                if w == "static" || w == "hidden" {
                    self.bump();
                    continue;
                }
            }
            if self.at(T::LBracket) {
                let (t, _) = self.parse_bracket_type();
                // A type literal never contains '('; an attribute invocation
                // like `ValidateSet('a')` does, so the latter is dropped rather
                // than overwriting the member type.
                if !t.contains('(') {
                    type_name = t;
                }
                continue;
            }
            break;
        }
        type_name
    }

    fn parse_class_member(&mut self, class_name: &str) -> Option<Node> {
        if self.at(T::RBrace) || self.at_end() {
            return None;
        }
        let start = self.pos;
        let type_name = self.parse_member_prefix();

        // Property: `$name [= default]`.
        if self.at(T::Variable) {
            let name = strip_sigil(self.value());
            self.bump();
            let default = if self.at_op("=") {
                self.bump();
                Some(Box::new(self.parse_expression()))
            } else {
                None
            };
            return Some(self.make(
                NodeKind::ClassMember {
                    member_kind: "property".into(),
                    name,
                    parameters: Vec::new(),
                    default,
                    body: None,
                },
                start,
                self.pos,
            ));
        }

        // Method or constructor: `Name ( params ) { body }`.
        if matches!(self.kind(), T::Generic | T::Keyword) {
            let name = self.value().to_string();
            self.bump();
            let parameters = if self.at(T::LParen) {
                self.parse_method_params()
            } else {
                Vec::new()
            };
            let body = if self.at(T::LBrace) {
                Some(Box::new(self.parse_braced_block()))
            } else {
                None
            };
            let member_kind = if type_name.is_empty() && name.eq_ignore_ascii_case(class_name) {
                "constructor"
            } else {
                "method"
            }
            .to_string();
            return Some(self.make(
                NodeKind::ClassMember {
                    member_kind,
                    name,
                    parameters,
                    default: None,
                    body,
                },
                start,
                self.pos,
            ));
        }

        let v = self.value().to_string();
        self.error("unexpected token in class body");
        Some(self.leaf(NodeKind::Error(v)))
    }

    fn parse_method_params(&mut self) -> Vec<Node> {
        let mut params = Vec::new();
        self.expect(T::LParen, "'('");
        while !self.at(T::RParen) && !self.at_end() {
            let p_start = self.pos;
            let ptype = self.parse_member_prefix();
            if !self.at(T::Variable) {
                break;
            }
            let raw = self.value().to_string();
            let var = self.leaf(NodeKind::Variable(raw));
            let typed = if ptype.is_empty() {
                var
            } else {
                self.make(
                    NodeKind::Cast {
                        type_name: ptype,
                        operand: Box::new(var),
                    },
                    p_start,
                    self.pos,
                )
            };
            let param = if self.at_op("=") {
                self.bump();
                let val = self.parse_expression();
                self.make(
                    NodeKind::Assignment {
                        target: Box::new(typed),
                        op: "=".into(),
                        value: Box::new(val),
                    },
                    p_start,
                    self.pos,
                )
            } else {
                typed
            };
            params.push(param);
            if self.at(T::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(T::RParen, "')'");
        params
    }

    fn parse_flow(&mut self) -> Node {
        let start = self.pos;
        let keyword = self.lower();
        self.bump();
        let value = if self.at_command_boundary() {
            None
        } else {
            Some(Box::new(self.parse_pipeline_statement()))
        };
        self.make(NodeKind::Flow { keyword, value }, start, self.pos)
    }

    fn parse_param_block(&mut self) -> Node {
        let start = self.pos;
        self.bump(); // param
        self.expect(T::LParen, "'('");
        let params = self.parse_param_decl_list();
        self.make(NodeKind::ParamBlock(params), start, self.pos)
    }

    /// Parses `param_decl (',' param_decl)* ')'`, with the opening `(` already
    /// consumed. A malformed entry is skipped past so the list still
    /// terminates. Shared by the `param(...)` block and the `function f(...)`
    /// parameter list.
    fn parse_param_decl_list(&mut self) -> Vec<Node> {
        let mut params = Vec::new();
        while !self.at_end() && !self.at(T::RParen) {
            if let Some(p) = self.parse_param_decl() {
                params.push(p);
            }
            if self.at(T::Comma) {
                self.bump();
            } else if !self.at(T::RParen) {
                self.bump(); // stall guard: drop a stray token
            }
        }
        self.expect(T::RParen, "')'");
        params
    }

    /// Parses one parameter declaration: `[attr][type] $name [= default]`.
    /// Returns `None` when no variable is found (a malformed entry), after
    /// consuming the brackets so the caller's loop still advances.
    fn parse_param_decl(&mut self) -> Option<Node> {
        let start = self.pos;
        let mut attributes = Vec::new();
        // Leading `[...]` brackets: attributes and the type, kept in order.
        while self.at(T::LBracket) {
            let (text, open) = self.parse_bracket_type();
            attributes.push(self.make(NodeKind::TypeExpression(text), open, self.pos));
        }
        if !self.at(T::Variable) {
            return None;
        }
        let name = self.value().to_string();
        let name_span = self.tokens[self.pos].span;
        self.bump();
        // Optional default: `= <expression>`. The comma is suppressed at the
        // top of the default so a top-level comma separates parameters rather
        // than being read as an array element of this default (an array default
        // must be parenthesized, `$x = (1, 2)`, or use `@(...)`). The comma
        // collection resumes inside any grouping.
        let default = if self.at_op("=") {
            self.bump();
            let saved = self.suppress_array_comma;
            self.suppress_array_comma = true;
            let expr = self.parse_ternary();
            self.suppress_array_comma = saved;
            Some(Box::new(expr))
        } else {
            None
        };
        Some(self.make(
            NodeKind::Parameter {
                attributes,
                name,
                name_span,
                default,
            },
            start,
            self.pos,
        ))
    }

    // Expressions

    /// Parses an expression. The comma operator is no longer handled here; it
    /// sits lower, at `parse_array_literal`, between the binary-operator climb
    /// and `parse_unary`, so it binds tighter than `-f`, the arithmetic,
    /// comparison, and logical operators (looser only than casts, index, member
    /// access, subexpressions, and the unary operators). This matches
    /// PowerShell's documented precedence (`about_Operator_Precedence`), where
    /// the comma operator is among the highest. For example, `1,2,1+2` parses
    /// as `(1,2,1)+2` (the array `1,2,1` with `2` appended), as in PowerShell,
    /// not as `@(1,2,3)`.
    fn parse_expression(&mut self) -> Node {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Node {
        let start = self.pos;
        let condition = self.parse_binary(0);
        if self.at_op("?") {
            self.bump();
            let if_true = self.parse_ternary();
            if self.at_op(":") {
                self.bump();
            } else {
                self.error("expected ':' in ternary");
            }
            let if_false = self.parse_ternary();
            return self.make(
                NodeKind::Ternary {
                    condition: Box::new(condition),
                    if_true: Box::new(if_true),
                    if_false: Box::new(if_false),
                },
                start,
                self.pos,
            );
        }
        condition
    }

    fn parse_binary(&mut self, min_prec: u8) -> Node {
        let start = self.pos;
        let mut left = self.parse_array_literal();
        loop {
            if let Some(prec) = self.binary_prec() {
                if prec < min_prec {
                    break;
                }
                let op = self.value().to_string();
                self.bump();
                let right = self.parse_binary(prec + 1);
                left = self.make(
                    NodeKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    start,
                    self.pos,
                );
                continue;
            }
            // `$_*2` glues into one Generic token because the lexer reads `*`
            // followed by a word as a wildcard (argument mode). Here an
            // operand just ended, so the expression-mode reading applies:
            // split the token into `*`-separated numeric segments and fold
            // them left-associatively, leaving the token stream untouched.
            const STAR_PREC: u8 = 7; // same tier as `*` `/` `%`
            if STAR_PREC < min_prec {
                break;
            }
            let Some(segments) = self.glued_star_segments() else {
                break;
            };
            let range_end = self.bump() + 1;
            for (mut seg, delta) in segments {
                let range = TokenRange {
                    first: left.range.first,
                    end: range_end,
                };
                rebase(
                    &mut seg,
                    delta,
                    TokenRange {
                        first: range_end - 1,
                        end: range_end,
                    },
                );
                let span = Span::new(left.span.start, seg.span.end);
                left = Node {
                    kind: NodeKind::Binary {
                        op: "*".to_string(),
                        left: Box::new(left),
                        right: Box::new(seg),
                    },
                    span,
                    range,
                };
            }
        }
        left
    }

    /// When the current token is a Generic of the shape `*<n>` or
    /// `*<n>*<m>...` whose every `*`-separated segment is a numeric literal,
    /// returns the parsed segments paired with their absolute byte offsets,
    /// consuming nothing. Anything else (`*abc`, `*.log`, empty segments from
    /// `**`) returns `None`, so glob arguments keep the wildcard reading, and
    /// the whole path is suppressed in command-argument position, where `$x*2`
    /// is a glob continuation.
    fn glued_star_segments(&self) -> Option<Vec<(Node, usize)>> {
        if self.in_command_argument || !self.at(T::Generic) {
            return None;
        }
        let tok = &self.tokens[self.pos];
        let rest = tok.value.strip_prefix('*')?;
        if rest.is_empty() {
            return None; // a lone `*` is already an Operator
        }
        let mut segments = Vec::new();
        let mut offset = tok.span.start + 1; // past the leading `*`
        for seg in rest.split('*') {
            if seg.is_empty() {
                return None; // `**` stays a glob
            }
            let sub = parse(seg);
            if !sub.errors.is_empty() {
                return None;
            }
            let node = unwrap_single_expression(sub.script)?;
            let is_numeric = match &node.kind {
                NodeKind::Number(_) => true,
                // `-` is a word character (command names), so `*-2` glues too;
                // a sign-wrapped numeric literal is still multiplication.
                NodeKind::Unary { op, operand } => {
                    matches!(op.as_str(), "-" | "+") && matches!(operand.kind, NodeKind::Number(_))
                }
                _ => false,
            };
            if !is_numeric {
                return None; // `*abc` is a glob word, not multiplication
            }
            segments.push((node, offset));
            offset += seg.len() + 1; // past this segment and the next `*`
        }
        Some(segments)
    }

    fn binary_prec(&self) -> Option<u8> {
        if !self.at(T::Operator) {
            return None;
        }
        let tok = &self.tokens[self.pos];
        let eq = |s: &str| tok.value_eq_ci(s);
        let v = tok.value.as_str();
        if eq("??") {
            Some(1)
        } else if eq("-or") || eq("-xor") || eq("-and") {
            // PowerShell gives `-and`, `-or`, and `-xor` equal precedence; they
            // fold left to right. (Microsoft about_Operator_Precedence: the
            // documented example `$true -or $false -and $false` is FALSE.)
            Some(2)
        } else if eq("..") {
            // Range binds tighter than `-f` (the table puts `..` above `-f`),
            // so it sits one rank above the format operator below.
            Some(9)
        } else if is_format_op(v) {
            // `-f` binds tighter than the arithmetic operators and looser than
            // range (about_Operator_Precedence ranks it above `* / %` and
            // `+ -`). Its own rank, not the comparison tier it would otherwise
            // fall into via `is_comparison_op`. Comma binds tighter than `-f`,
            // so the right operand picks up a whole comma list through
            // `parse_array_literal` without a special case.
            Some(8)
        } else if eq("+") || eq("-") {
            Some(6)
        } else if eq("*") || eq("/") || eq("%") {
            Some(7)
        } else if is_comparison_op(v) {
            // Comparison operators (and the type and binary split/join
            // operators, all one equal tier) bind tighter than the bitwise
            // tier below them.
            Some(5)
        } else if is_bitwise_op(v) {
            // Bitwise and shift operators are a tier looser than comparison and
            // tighter than the logical operators (about_Operator_Precedence).
            Some(4)
        } else {
            None
        }
    }

    /// The array-literal level: `unary-expression (, unary-expression)*`. Sits
    /// between the binary-operator climb and `parse_unary`, so the comma binds
    /// tighter than every binary operator (`-f`, arithmetic, comparison,
    /// bitwise, logical, range) and looser than the unary operators, casts,
    /// index, and member access, matching PowerShell's precedence. A single
    /// element with no trailing comma is returned unwrapped. The array nests to
    /// the right (`a, b, c` is `ArrayLiteral[a, ArrayLiteral[b, c]]`), the shape
    /// the comma operator produced when it was handled at the top level.
    fn parse_array_literal(&mut self) -> Node {
        let start = self.pos;
        let first = self.parse_unary();
        if self.at(T::Comma) && !self.suppress_array_comma {
            self.bump();
            let rest = self.parse_array_literal();
            return self.make(NodeKind::ArrayLiteral(vec![first, rest]), start, self.pos);
        }
        first
    }

    fn parse_unary(&mut self) -> Node {
        let start = self.pos;
        if self.at(T::Comma) {
            // Unary comma: `,$x` is a one-element array.
            self.bump();
            let operand = self.parse_unary();
            return self.make(NodeKind::ArrayLiteral(vec![operand]), start, self.pos);
        }
        if self.at(T::Operator) {
            let tok = &self.tokens[self.pos];
            if is_unary_operator_word(&tok.value.to_ascii_lowercase()) {
                let op = self.value().to_string();
                self.bump();
                let operand = self.parse_unary();
                if matches!(op.as_str(), "++" | "--") {
                    self.invalidate_cached_variable(&operand);
                }
                return self.make(
                    NodeKind::Unary {
                        op,
                        operand: Box::new(operand),
                    },
                    start,
                    self.pos,
                );
            }
        }
        if self.at(T::LBracket) {
            let (type_name, b_start) = self.parse_bracket_type();
            if self.is_value_start() {
                let operand = self.parse_unary();
                return self.make(
                    NodeKind::Cast {
                        type_name,
                        operand: Box::new(operand),
                    },
                    b_start,
                    self.pos,
                );
            }
            let type_node = self.make(NodeKind::TypeExpression(type_name), b_start, self.pos);
            return self.parse_postfix_from(type_node, b_start);
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Node {
        let start = self.pos;
        let prim = self.parse_primary();
        self.parse_postfix_from(prim, start)
    }

    fn parse_postfix_from(&mut self, mut node: Node, start: usize) -> Node {
        loop {
            // A newline ends the statement: a postfix operator (`.`, `::`, `[`,
            // `(`, `++`) on the next line is a new statement, not a suffix on
            // this value. Backtick continuations are LineContinuation trivia,
            // not Newline, so they do not trip this and the chain continues.
            if self.starts_line() {
                break;
            }
            let null_dot = self.at(T::Operator) && self.value() == "?.";
            let null_index = self.at(T::Operator) && self.value() == "?[";
            if self.at(T::Dot) || self.at(T::DoubleColon) || null_dot {
                let is_static = self.at(T::DoubleColon);
                self.bump();
                let (member, member_span) =
                    if matches!(self.kind(), T::Generic | T::Keyword | T::Variable) {
                        let v = self.value().to_string();
                        let span = matches!(self.kind(), T::Generic | T::Keyword)
                            .then_some(self.tokens[self.pos].span);
                        self.bump();
                        (v, span)
                    } else {
                        self.error("expected member name");
                        (String::new(), None)
                    };
                if self.at(T::LParen) {
                    let args = self.parse_paren_args();
                    node = self.make(
                        NodeKind::InvokeMember {
                            target: Box::new(node),
                            member,
                            member_span,
                            is_static,
                            args,
                        },
                        start,
                        self.pos,
                    );
                } else {
                    node = self.make(
                        NodeKind::MemberAccess {
                            target: Box::new(node),
                            member,
                            member_span,
                            is_static,
                        },
                        start,
                        self.pos,
                    );
                }
            } else if self.at(T::LBracket) || null_index {
                self.bump();
                let index = self.in_expression_context(|p| p.parse_expression());
                self.expect(T::RBracket, "']'");
                node = self.make(
                    NodeKind::Index {
                        target: Box::new(node),
                        index: Box::new(index),
                    },
                    start,
                    self.pos,
                );
            } else if self.at(T::Operator) && matches!(self.value(), "++" | "--") {
                let op = self.value().to_string();
                self.bump();
                self.invalidate_cached_variable(&node);
                node = self.make(
                    NodeKind::PostfixUnary {
                        op,
                        operand: Box::new(node),
                    },
                    start,
                    self.pos,
                );
            } else {
                break;
            }
        }
        node
    }

    fn parse_paren_args(&mut self) -> Vec<Node> {
        self.bump(); // (
        let mut args = Vec::new();
        if !self.at(T::RParen) && !self.at_end() {
            // v1 parses the whole argument list as one expression, so a comma
            // list becomes a single ArrayLiteral argument.
            let arg = self.in_expression_context(|p| p.parse_expression());
            args.push(arg);
        }
        self.expect(T::RParen, "')'");
        args
    }

    /// Runs `f` with the command-argument flag cleared: grouping constructs
    /// re-enter pure expression context, where glued `*` is multiplication
    /// again (`dir ($_*2)`).
    fn in_expression_context<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let saved = self.in_command_argument;
        let saved_comma = self.suppress_array_comma;
        self.in_command_argument = false;
        self.suppress_array_comma = false;
        let r = f(self);
        self.in_command_argument = saved;
        self.suppress_array_comma = saved_comma;
        r
    }

    fn parse_primary(&mut self) -> Node {
        if self.enter() {
            self.leave();
            // Past the depth limit (sticky once hit): consume one token and
            // return an error node. Consuming guarantees the cursor advances, so
            // an enclosing loop or a parent retry drains the remaining input
            // linearly instead of re-parsing the same tokens.
            let start = self.pos;
            self.error("expression nested too deeply");
            if !self.at_end() {
                self.bump();
            }
            return self.make(NodeKind::Error("too deep".into()), start, self.pos);
        }
        let node = self.parse_primary_inner();
        self.leave();
        node
    }

    fn parse_primary_inner(&mut self) -> Node {
        let start = self.pos;
        match self.kind() {
            T::Variable => {
                let v = self.value().to_string();
                self.leaf(NodeKind::Variable(v))
            }
            T::Number => {
                let v = self.value().to_string();
                self.leaf(NodeKind::Number(v))
            }
            T::StringSq => self.string_leaf(StringKind::Single),
            T::StringDq => self.string_leaf(StringKind::Double),
            T::HereStringSq => self.string_leaf(StringKind::HereSingle),
            T::HereStringDq => self.string_leaf(StringKind::HereDouble),
            T::LParen => {
                self.bump();
                let inner = self.in_expression_context(|p| p.parse_pipeline_statement());
                self.expect(T::RParen, "')'");
                self.make(NodeKind::Paren(Box::new(inner)), start, self.pos)
            }
            T::DollarParen => {
                self.bump();
                let body = self.parse_block_body(&[T::RParen]);
                let script = self.make(NodeKind::Script(body), start + 1, self.pos);
                self.expect(T::RParen, "')'");
                self.make(NodeKind::SubExpression(Box::new(script)), start, self.pos)
            }
            T::AtParen => {
                self.bump();
                let items = self.parse_block_body(&[T::RParen]);
                self.expect(T::RParen, "')'");
                self.make(NodeKind::Array(items), start, self.pos)
            }
            T::AtBrace => self.in_expression_context(|p| p.parse_hashtable()),
            T::LBrace => {
                self.bump();
                let body = self.parse_block_body(&[T::RBrace]);
                let script = self.make(NodeKind::Script(body), start + 1, self.pos);
                self.expect(T::RBrace, "'}'");
                self.make(
                    NodeKind::ScriptBlockExpression(Box::new(script)),
                    start,
                    self.pos,
                )
            }
            T::LBracket => {
                let (type_name, b_start) = self.parse_bracket_type();
                self.make(NodeKind::TypeExpression(type_name), b_start, self.pos)
            }
            T::Generic | T::Keyword => {
                let v = self.value().to_string();
                self.leaf(NodeKind::BareWord(v))
            }
            _ => {
                let v = self.value().to_string();
                self.error(&format!("unexpected token '{v}'"));
                self.leaf(NodeKind::Error(v))
            }
        }
    }

    fn string_leaf(&mut self, kind: StringKind) -> Node {
        let i = self.pos;
        let value = self.tokens[i].value.clone();
        let span = self.tokens[i].span;
        self.pos = (self.pos + 1).min(self.tokens.len() - 1);
        let parts = if matches!(kind, StringKind::Double | StringKind::HereDouble) {
            let inner = string_inner(&value, kind);
            // Guard against pathological interpolation nesting on adversarial
            // input; real code never approaches this depth.
            if inner.contains('$') && self.depth < 64 {
                let base = body_span(&value, kind, span).start;
                extract_interpolations(inner, base, i, self.depth)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        self.make(NodeKind::StringLiteral { kind, value, parts }, i, i + 1)
    }

    fn parse_hashtable(&mut self) -> Node {
        let start = self.pos;
        self.bump(); // @{
        let mut pairs = Vec::new();
        loop {
            self.skip_separators();
            if self.at(T::RBrace) || self.at_end() {
                break;
            }
            let key = self.parse_ternary();
            if self.at_op("=") {
                self.bump();
            } else {
                self.error("expected '=' in hashtable entry");
            }
            let value = self.parse_pipeline_statement();
            pairs.push((key, value));
            self.skip_separators();
        }
        self.expect(T::RBrace, "'}'");
        self.make(NodeKind::Hashtable(pairs), start, self.pos)
    }

    /// Consumes a `[ ... ]` type, including nested brackets (`[int[]]`,
    /// `[Dictionary[string, int]]`), and returns the inner text and the index
    /// of the opening bracket.
    fn parse_bracket_type(&mut self) -> (String, usize) {
        let open = self.pos;
        let open_end = self.tokens[open].span.end;
        self.bump(); // [
        let mut depth = 1usize;
        let mut close_start = open_end;
        while !self.at_end() {
            if self.at(T::LBracket) {
                depth += 1;
                self.bump();
            } else if self.at(T::RBracket) {
                depth -= 1;
                close_start = self.tokens[self.pos].span.start;
                self.bump();
                if depth == 0 {
                    break;
                }
            } else {
                self.bump();
            }
        }
        let name = self.src[open_end..close_start.max(open_end)].to_string();
        (name, open)
    }
}

fn is_assignment_op(op: &str) -> bool {
    matches!(op, "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "??=")
}

/// PowerShell's only format operator, `-f`. Case-insensitive, optional leading
/// `-`, matching how the other operator predicates compare.
fn is_format_op(op: &str) -> bool {
    op.strip_prefix('-').unwrap_or(op).eq_ignore_ascii_case("f")
}

fn is_comparison_op(v: &str) -> bool {
    // Operators are case-insensitive in PowerShell (`-EQ`, `-iLike`). Strip a
    // leading `-`, then match the name. The optional case-sensitivity prefix
    // (`c`/`i`, as in `-ceq` / `-ieq`) is only stripped when the full name does
    // not already match, so operators that begin with `c` or `i` (`-contains`,
    // `-in`, `-is`, `-isnot`) are still recognized.
    let core = v.strip_prefix('-').unwrap_or(v);
    const NAMES: &[&str] = &[
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
        "contains",
        "notcontains",
        "in",
        "notin",
        "is",
        "isnot",
        "as",
        "replace",
        "split",
        "join",
    ];
    let matches_name = |s: &str| NAMES.iter().any(|n| n.eq_ignore_ascii_case(s));
    if matches_name(core) {
        return true;
    }
    match core.as_bytes().first() {
        Some(b'c' | b'C' | b'i' | b'I') => crate::ops::CASE_PREFIXABLE
            .iter()
            .any(|n| n.eq_ignore_ascii_case(&core[1..])),
        _ => false,
    }
}

/// The binary bitwise and shift operators (`-band`, `-bor`, `-bxor`, `-shl`,
/// `-shr`). They form one precedence tier, looser than the comparison operators
/// and tighter than the logical operators, evaluated left to right.
fn is_bitwise_op(v: &str) -> bool {
    let core = v.strip_prefix('-').unwrap_or(v);
    ["band", "bor", "bxor", "shl", "shr"]
        .iter()
        .any(|n| n.eq_ignore_ascii_case(core))
}

fn strip_sigil(var: &str) -> String {
    var.trim_start_matches(['$', '@']).to_string()
}

/// The body of a string, without its delimiters.
pub(super) fn string_inner(value: &str, kind: StringKind) -> &str {
    match kind {
        StringKind::Single => value
            .strip_prefix('\'')
            .and_then(|s| s.strip_suffix('\''))
            .unwrap_or(value),
        StringKind::Double => value
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(value),
        StringKind::HereSingle => strip_here_newlines(
            value
                .strip_prefix("@'")
                .and_then(|s| s.strip_suffix("'@"))
                .unwrap_or(value),
        ),
        StringKind::HereDouble => strip_here_newlines(
            value
                .strip_prefix("@\"")
                .and_then(|s| s.strip_suffix("\"@"))
                .unwrap_or(value),
        ),
    }
}

/// Drops the one newline a here-string carries right after its opening
/// delimiter and right before its closing delimiter.
fn strip_here_newlines(s: &str) -> &str {
    let s = s
        .strip_prefix("\r\n")
        .or_else(|| s.strip_prefix('\n'))
        .unwrap_or(s);
    s.strip_suffix("\r\n")
        .or_else(|| s.strip_suffix('\n'))
        .unwrap_or(s)
}

/// Extracts interpolation nodes (`$var`, `${name}`, `$(...)`) from the body of
/// an expandable string. Parts have source byte spans and an empty token range,
/// since their bytes are owned by the string token.
fn extract_interpolations(inner: &str, base: usize, idx: usize, enclosing_depth: u32) -> Vec<Node> {
    let range = TokenRange {
        first: idx,
        end: idx,
    };
    let n = inner.len();
    let span_at = |a: usize, z: usize| Span::new(base + a, base + z);
    let mut parts = Vec::new();
    let mut i = 0;
    while i < n {
        // Cursor positions are bytes; non-ASCII text and escaped characters
        // advance by whole scalars so slicing stays on UTF-8 boundaries.
        let c = inner[i..].chars().next().expect("i is within inner");
        if c == '`' {
            i += 1;
            if let Some(escaped) = inner[i..].chars().next() {
                i += escaped.len_utf8();
            }
            continue;
        }
        if inner[i..].starts_with("$(") {
            let closed_end = scan_subexpression_end(&inner[i..]).map(|end| i + end);
            let end = closed_end.unwrap_or(n);
            // Strip only a confirmed root closer. An unclosed fragment keeps
            // its exact available remainder, including any nested closer.
            let body_end = closed_end.map_or(n, |end| end - 1);
            let sub_src = &inner[i + 2..body_end];
            // Boundary-probe and fragment diagnostics intentionally stay local,
            // preserving the outer parser's existing error policy.
            // Inherit nesting, but keep the cache and sticky overflow latch
            // fragment-local so recovery cannot poison the enclosing parser.
            let mut script = parse_at_depth(sub_src, enclosing_depth).script;
            // Fragment spans are relative to `sub_src` (base 0). Lift them to
            // absolute source offsets: `sub_src` begins at the byte after `$(`,
            // i.e. `base + i + 2`. Without this, a variable inside the
            // sub-expression keeps a sub_src-relative span and points at the
            // wrong bytes (or, worse, at coincidentally matching ones).
            let delta = base + i + 2;
            rebase(&mut script, delta, range);
            parts.push(Node {
                kind: NodeKind::SubExpression(Box::new(script)),
                span: span_at(i, end),
                range,
            });
            i = end;
            continue;
        }
        if c == '$' {
            if let Some(consumed) = variable_reference_len(&inner[i..]) {
                let raw = inner[i..i + consumed].to_string();
                parts.push(Node {
                    kind: NodeKind::Variable(raw),
                    span: span_at(i, i + consumed),
                    range,
                });
                i += consumed;
                continue;
            }
        }
        i += c.len_utf8();
    }
    parts
}

/// Shifts every span in `node` by `delta` and pins its token range to `range`.
/// Used to lift a sub-expression that was parsed from a detached substring
/// (offsets relative to that substring) into the coordinates of the original
/// source. The token range is set to the string token's placeholder, since the
/// sub-expression's tokens are not part of the outer token stream.
/// Whether a dash-word (lowercased) is a prefix-unary operator. Shared by
/// `is_value_start` (to route the expression) and `parse_unary` (to build the
/// node), so the two cannot disagree about what begins a unary expression.
/// `-split` and `-join` have unary forms; `-split` also takes a `c`/`i` case
/// prefix, `-join` does not.
fn is_unary_operator_word(word: &str) -> bool {
    matches!(
        word,
        "-" | "+"
            | "!"
            | "-not"
            | "-bnot"
            | "++"
            | "--"
            | "-split"
            | "-csplit"
            | "-isplit"
            | "-join"
    )
}

/// Peels single-statement `Script`/`Pipeline` wrappers off a re-parsed
/// fragment, yielding the one expression inside, or `None` when the fragment
/// is not exactly one expression.
fn unwrap_single_expression(node: Node) -> Option<Node> {
    match node.kind {
        NodeKind::Script(mut v) | NodeKind::Pipeline(mut v) => {
            if v.len() == 1 {
                unwrap_single_expression(v.pop().expect("length checked"))
            } else {
                None
            }
        }
        _ => Some(node),
    }
}

fn rebase(node: &mut Node, delta: usize, range: TokenRange) {
    node.span = Span::new(node.span.start + delta, node.span.end + delta);
    node.range = range;
    match &mut node.kind {
        NodeKind::MemberAccess { member_span, .. } | NodeKind::InvokeMember { member_span, .. } => {
            if let Some(span) = member_span {
                // Like the containing node, lift once per detached fragment;
                // nested fragments have already been lifted into this one.
                *span = Span::new(span.start + delta, span.end + delta);
            }
        }
        _ => {}
    }
    node.for_each_child_mut(&mut |child| rebase(child, delta, range));
}

/// Source span of a string's body (what [`string_inner`] returns) given the
/// token's full span, so interpolation parts and Add-Type C# get spans that
/// point at the real bytes rather than the whole string token.
pub(super) fn body_span(value: &str, kind: StringKind, span: Span) -> Span {
    let inner = string_inner(value, kind);
    let off = subslice_offset(value, inner);
    Span::new(span.start + off, span.start + off + inner.len())
}

/// Byte offset of `inner` within `outer`. `inner` must be a subslice of
/// `outer`, as produced by `str::strip_prefix`/`strip_suffix`.
fn subslice_offset(outer: &str, inner: &str) -> usize {
    (inner.as_ptr() as usize) - (outer.as_ptr() as usize)
}

fn strip_param_dash(param: &str) -> String {
    param
        .trim_start_matches('-')
        .trim_end_matches(':')
        .to_string()
}

// Add-Type C# extraction (hand-rolled, dependency-free)

use std::collections::HashMap;

#[cfg(not(feature = "csharp"))]
use super::ast::CSharpParam;
use super::ast::{CSharpImport, CSharpMemberDef};

/// Locates the C# source handed to `Add-Type`: the argument of a
/// `-TypeDefinition`/`-MemberDefinition` parameter (prefix-matched), or the
/// first positional eligible literal or variable with a current source-cache
/// entry. See the module's bounded cache contract. Returns `(code, label, span)`.
fn find_csharp_code(
    elements: &[Node],
    vars: &HashMap<String, (String, Span)>,
) -> Option<(String, String, Span)> {
    const CSHARP_PARAMS: [&str; 2] = ["memberdefinition", "typedefinition"];
    for (i, el) in elements.iter().enumerate() {
        let NodeKind::CommandParameter { name, argument } = &el.kind else {
            continue;
        };
        let pname = name.to_ascii_lowercase();
        if !CSHARP_PARAMS.iter().any(|full| full.starts_with(&pname)) {
            continue;
        }
        let value: Option<&Node> = argument.as_deref().or_else(|| match elements.get(i + 1) {
            Some(e) if !matches!(e.kind, NodeKind::CommandParameter { .. }) => Some(e),
            _ => None,
        });
        if let Some((code, span)) = value.and_then(|v| as_csharp_code(v, vars)) {
            return Some((code, pname, span));
        }
    }
    for el in elements {
        if matches!(el.kind, NodeKind::CommandParameter { .. }) {
            continue;
        }
        if let Some((code, span)) = as_csharp_code(el, vars) {
            return Some((code, "positional".to_string(), span));
        }
    }
    None
}

pub(super) fn is_constant_string(value: &str, kind: StringKind) -> bool {
    if matches!(kind, StringKind::Single | StringKind::HereSingle) {
        return true;
    }
    // Interpolation parts do not cover every reference. Check raw spelling so
    // escaped dollars and escaped backticks remain distinguishable.
    let mut chars = string_inner(value, kind).chars();
    while let Some(c) = chars.next() {
        match c {
            '`' => {
                chars.next();
            }
            '$' => return false,
            _ => {}
        }
    }
    true
}

/// An eligible literal's raw body and source span, or an unscoped variable's
/// current cached body/span. This does not evaluate or decode runtime strings.
fn as_csharp_code(node: &Node, vars: &HashMap<String, (String, Span)>) -> Option<(String, Span)> {
    match &node.kind {
        NodeKind::StringLiteral { value, kind, .. } if is_constant_string(value, *kind) => Some((
            string_inner(value, *kind).to_string(),
            body_span(value, *kind, node.span),
        )),
        NodeKind::Variable(raw) => {
            let n = raw.trim_start_matches('$');
            if n.contains(':') || n.starts_with('{') {
                None
            } else {
                vars.get(&n.to_ascii_lowercase()).cloned()
            }
        }
        _ => None,
    }
}

/// Parses inline C# for `[DllImport]` P/Invoke declarations.
fn parse_csharp_member_def(code: &str) -> CSharpMemberDef {
    let (imports, apis) = extract_imports_apis(code);
    CSharpMemberDef {
        code: code.to_string(),
        // The caller (maybe_csharp) fills this with the real source span; this
        // path only has the extracted text.
        code_span: Span::default(),
        imports,
        apis,
        parameter: String::new(),
    }
}

/// With the C# front-end compiled in, read imports from the real parse tree.
#[cfg(feature = "csharp")]
fn extract_imports_apis(code: &str) -> (Vec<CSharpImport>, Vec<String>) {
    let unit = crate::v2::csharp::parser::cs_parse(code, 0);
    crate::v2::csharp::imports::csharp_imports_and_apis(&unit, code)
}

/// Without it, a lightweight scanner finds `[DllImport]` signatures, so the
/// default build still populates imports with no extra dependency or parser.
#[cfg(not(feature = "csharp"))]
fn extract_imports_apis(code: &str) -> (Vec<CSharpImport>, Vec<String>) {
    let mut imports = Vec::new();
    let mut apis = Vec::new();
    let n = code.len();
    let mut i = 0;
    while i < n {
        if let Some((imp, end)) = try_match_dllimport(code, i) {
            apis.push(imp.function.clone());
            imports.push(imp);
            i = end;
        } else {
            i += 1;
        }
    }
    (imports, apis)
}

/// Tries to match `[DllImport("dll" ...)] <modifiers> <ret> <fn>(<params>)`
/// starting at byte `i`, returning the import and the byte offset just past the
/// parameter list. Mirrors v1's `DLLIMPORT_RE` plus `balanced_parens`.
#[cfg(not(feature = "csharp"))]
fn try_match_dllimport(code: &str, i: usize) -> Option<(CSharpImport, usize)> {
    let b = code.as_bytes();
    let n = b.len();
    if b.get(i) != Some(&b'[') {
        return None;
    }
    let mut j = skip_ws_b(b, i + 1);
    if !matches_ci(b, j, b"dllimport") {
        return None;
    }
    j = skip_ws_b(b, j + b"dllimport".len());
    if b.get(j) != Some(&b'(') {
        return None;
    }
    j = skip_ws_b(b, j + 1);
    let quote = match b.get(j) {
        Some(&b'"') => b'"',
        Some(&b'\'') => b'\'',
        _ => return None,
    };
    let dll_start = j + 1;
    let mut k = dll_start;
    while k < n && b[k] != quote {
        k += 1;
    }
    if k >= n || k == dll_start {
        return None; // unterminated, or empty (regex requires one or more)
    }
    let dll = code[dll_start..k].to_string();
    k += 1; // past closing quote
    while k < n && b[k] != b']' {
        k += 1; // [^]]* up to the attribute close
    }
    if k >= n {
        return None;
    }
    k += 1; // past ']'
            // At least one modifier (with surrounding whitespace) must separate the
            // attribute from the return type, because v1's DLLIMPORT_RE demands the
            // run; accepting zero would also misread `]int` as a return type.
    let mod_start = k;
    loop {
        let ws = skip_ws_b(b, k);
        if let Some(end) = match_cs_modifier(b, ws) {
            k = end;
        } else {
            k = ws;
            break;
        }
    }
    if k == mod_start {
        return None;
    }
    let ret_start = k;
    if k >= n || !(is_alpha_b(b[k]) || b[k] == b'_') {
        return None;
    }
    k += 1;
    while k < n && is_ret_char_b(b[k]) {
        k += 1;
    }
    let returns = code[ret_start..k].to_string();
    let ws2 = skip_ws_b(b, k);
    if ws2 == k {
        return None; // \s+ required before the function name
    }
    k = ws2;
    let fn_start = k;
    if k >= n || !(is_alpha_b(b[k]) || b[k] == b'_') {
        return None;
    }
    k += 1;
    while k < n && is_word_b(b[k]) {
        k += 1;
    }
    let function = code[fn_start..k].to_string();
    k = skip_ws_b(b, k);
    if b.get(k) != Some(&b'(') {
        return None;
    }
    let (params_str, after) = balanced_parens(code, k + 1);
    Some((
        CSharpImport {
            dll,
            function,
            returns,
            params: parse_cs_params(&params_str),
        },
        after,
    ))
}

#[cfg(not(feature = "csharp"))]
fn parse_cs_params(param_str: &str) -> Vec<CSharpParam> {
    const MODS: [&str; 7] = ["ref", "out", "in", "params", "this", "readonly", "scoped"];
    split_top_level(param_str, ',')
        .iter()
        .filter_map(|raw| {
            let cleaned = strip_cs_attrs(raw);
            let no_default = cleaned.split('=').next().unwrap_or("").trim();
            let words: Vec<&str> = no_default
                .split_whitespace()
                .filter(|w| !MODS.contains(w))
                .collect();
            match words.len() {
                0 => None,
                1 => Some(CSharpParam {
                    type_name: words[0].to_string(),
                    name: String::new(),
                }),
                _ => Some(CSharpParam {
                    type_name: words[..words.len() - 1].join(" "),
                    name: words.last().unwrap().to_string(),
                }),
            }
        })
        .collect()
}

/// Replaces `[Attr...]` attribute spans with a space, as v1's `CS_ATTR_RE` does.
#[cfg(not(feature = "csharp"))]
fn strip_cs_attrs(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::new();
    let mut i = 0;
    while i < n {
        if chars[i] == '[' {
            let mut j = i + 1;
            while j < n && chars[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < n && (chars[j].is_ascii_alphabetic() || chars[j] == '_') {
                let mut k = j;
                while k < n && chars[k] != ']' {
                    k += 1;
                }
                if k < n {
                    out.push(' ');
                    i = k + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Splits `s` on `sep` at top level, ignoring separators inside quotes or
/// nested brackets. Empty (whitespace-only) pieces are dropped.
#[cfg(not(feature = "csharp"))]
fn split_top_level(s: &str, sep: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    let mut cur = String::new();
    for c in s.chars() {
        if let Some(q) = quote {
            cur.push(c);
            if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => {
                quote = Some(c);
                cur.push(c);
            }
            '(' | '[' | '{' | '<' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' | '}' | '>' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            _ if c == sep && depth == 0 => parts.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    parts.push(cur);
    parts.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// From a byte offset just past `(`, returns the balanced-paren contents and
/// the offset just past the matching `)`.
#[cfg(not(feature = "csharp"))]
fn balanced_parens(code: &str, start: usize) -> (String, usize) {
    let mut depth = 1usize;
    for (off, c) in code[start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return (code[start..start + off].to_string(), start + off + 1);
                }
            }
            _ => {}
        }
    }
    (code[start..].to_string(), code.len())
}

#[cfg(not(feature = "csharp"))]
fn skip_ws_b(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

#[cfg(not(feature = "csharp"))]
fn matches_ci(b: &[u8], i: usize, kw: &[u8]) -> bool {
    i + kw.len() <= b.len() && b[i..i + kw.len()].eq_ignore_ascii_case(kw)
}

#[cfg(not(feature = "csharp"))]
fn is_alpha_b(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

#[cfg(not(feature = "csharp"))]
fn is_word_b(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

#[cfg(not(feature = "csharp"))]
fn is_ret_char_b(c: u8) -> bool {
    is_word_b(c) || matches!(c, b'.' | b'<' | b'>' | b'[' | b']')
}

#[cfg(not(feature = "csharp"))]
fn match_cs_modifier(b: &[u8], i: usize) -> Option<usize> {
    const MODS: [&[u8]; 7] = [
        b"public",
        b"private",
        b"internal",
        b"protected",
        b"static",
        b"extern",
        b"unsafe",
    ];
    MODS.iter()
        .find(|m| matches_ci(b, i, m))
        .map(|m| i + m.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(src: &str) -> Vec<String> {
        let out = parse(src);
        let mut found = Vec::new();
        out.script.walk(&mut |n| {
            if let NodeKind::Command { name, .. } = &n.kind {
                if let NodeKind::BareWord(s) = &name.kind {
                    found.push(s.clone());
                }
            }
        });
        found
    }

    fn clean(src: &str) -> Node {
        let out = parse(src);
        assert!(
            out.errors.is_empty(),
            "errors for {src:?}: {:?}",
            out.errors
        );
        out.script
    }

    #[test]
    fn parse_lex_errors_after_statement_keep_block_comment_span() {
        let src = "1; <# x";
        let expected = vec![ParseError {
            message: "unterminated block comment at 3..7".into(),
            span: Span::new(3, 7),
        }];

        let out = parse(src);

        assert_eq!(out.errors, expected);
    }

    #[test]
    fn parse_lex_errors_with_unicode_keep_utf8_byte_span() {
        let src = "'ż'; <# 🦀";
        let expected = vec![ParseError {
            message: "unterminated block comment at 6..13".into(),
            span: Span::new(6, 13),
        }];

        let out = parse(src);

        assert_eq!(out.errors, expected);
    }

    #[test]
    fn parse_lex_errors_with_bom_keep_original_utf8_byte_span() {
        let src = "\u{feff}'ż'; <# 🦀";
        let expected = vec![ParseError {
            message: "unterminated block comment at 9..16".into(),
            span: Span::new(9, 16),
        }];

        let out = parse(src);

        assert_eq!(out.errors, expected);
    }

    #[test]
    fn parse_lex_errors_at_source_start_keep_nonempty_span() {
        let src = "<#";
        let expected = vec![ParseError {
            message: "unterminated block comment at 0..2".into(),
            span: Span::new(0, 2),
        }];

        let out = parse(src);

        assert_eq!(out.errors, expected);
    }

    #[test]
    fn parse_lex_errors_in_unterminated_string_keep_token_span() {
        let src = "1; 'open";
        let expected = vec![ParseError {
            message: "unterminated string at 3..8".into(),
            span: Span::new(3, 8),
        }];

        let out = parse(src);

        assert_eq!(out.errors, expected);
    }

    #[test]
    fn parse_lex_errors_with_multiple_diagnostics_keep_spans_and_order() {
        let src = "0x; <# x";
        let expected = vec![
            ParseError {
                message: "radix prefix with no digits at 0..2".into(),
                span: Span::new(0, 2),
            },
            ParseError {
                message: "unterminated block comment at 4..8".into(),
                span: Span::new(4, 8),
            },
        ];

        let out = parse(src);

        assert_eq!(out.errors, expected);
    }

    #[test]
    fn parse_lex_errors_before_parser_errors_keep_spans_and_order() {
        let src = "param( <# x";
        let expected = vec![
            ParseError {
                message: "unterminated block comment at 7..11".into(),
                span: Span::new(7, 11),
            },
            ParseError {
                message: "expected ')'".into(),
                span: Span::new(11, 11),
            },
        ];

        let out = parse(src);

        assert_eq!(out.errors, expected);
    }

    #[test]
    fn parses_a_simple_pipeline() {
        let root = clean("Get-ChildItem -Path . -Recurse | Sort-Object Length\n");
        assert_eq!(root.label(), "ScriptBlock");
        assert_eq!(
            names("Get-ChildItem -Path . | Sort-Object\n"),
            ["Get-ChildItem", "Sort-Object"]
        );
    }

    #[test]
    fn ranges_cover_node_source() {
        let src = "$x = 1 + 2\n";
        let out = parse(src);
        // the whole assignment spans "$x = 1 + 2"
        let stmt = &out.script.children()[0];
        assert_eq!(stmt.label(), "AssignmentStatement");
        assert_eq!(stmt.span.slice(src), "$x = 1 + 2");
    }

    #[test]
    fn assignment_and_binary() {
        let root = clean("$total = $a * 2 + $b\n");
        let assign = &root.children()[0];
        assert_eq!(assign.label(), "AssignmentStatement");
        if let NodeKind::Assignment { value, op, .. } = &assign.kind {
            assert_eq!(op, "=");
            assert_eq!(value.label(), "BinaryExpression"); // + at the top
        } else {
            panic!("not an assignment");
        }
    }

    #[test]
    fn control_flow_blocks() {
        clean("if ($x -gt 1) { 'big' } elseif ($x -eq 1) { 'one' } else { 'small' }\n");
        clean("while ($true) { break }\n");
        clean("for ($i = 0; $i -lt 10; $i++) { $i }\n");
        clean("foreach ($f in $files) { $f.Name }\n");
        clean("do { $x } while ($x)\n");
        clean("try { risky } catch [System.Exception] { recover } finally { cleanup }\n");
    }

    #[test]
    fn functions_and_params() {
        let root = clean("function Get-Thing { param([string]$Name, [int]$Count = 3) $Name }\n");
        let func = &root.children()[0];
        assert_eq!(func.label(), "FunctionDefinition");
    }

    #[test]
    fn member_access_index_invoke() {
        clean("$x.Length\n");
        clean("$a[0]\n");
        clean("[System.Math]::Max(1, 2)\n");
        clean("$obj.Method($arg).Property\n");
    }

    // These cases check the public syntax tree, tokens, and diagnostics against
    // explicit expectations, without a second parser or a source rescan.
    macro_rules! member_parse_cases {
        ($($name:ident: $src:expr,
            ($label:expr, $member:expr, $static:expr, $end:expr, $range_end:expr,
             $target_end:expr, $args:expr),
            ($token_index:expr, $token_kind:ident, $token_text:expr, $ts:expr, $te:expr),
            [$(($error:expr, $es:expr, $ee:expr)),*];)*) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;
                    let expected_errors = vec![$(ParseError {
                        message: $error.into(),
                        span: Span::new($es, $ee),
                    }),*];

                    let out = parse(src);

                    let mut sites = Vec::new();
                    out.script.walk(&mut |node| {
                        let (target, member, is_static, args) = match &node.kind {
                            NodeKind::MemberAccess { target, member, is_static, .. } =>
                                (target, member, *is_static, None),
                            NodeKind::InvokeMember { target, member, is_static, args, .. } =>
                                (target, member, *is_static, Some(args.len())),
                            _ => return,
                        };
                        sites.push((
                            node.label(), member.clone(), is_static, node.span,
                            node.range, target.span, args,
                        ));
                    });
                    assert_eq!(out.errors, expected_errors, "{src:?}");
                    assert_eq!(sites, vec![(
                        $label, $member.to_string(), $static, Span::new(0, $end),
                        TokenRange { first: 0, end: $range_end },
                        Span::new(0, $target_end), $args,
                    )], "{src:?}");
                    let token = &out.tokens[$token_index];
                    assert_eq!(
                        (token.kind, token.value.as_str(), token.span),
                        (T::$token_kind, $token_text, Span::new($ts, $te)),
                        "{src:?}",
                    );
                }
            )*
        };
    }

    member_parse_cases! {
        parse_static_ascii_access_keeps_member_and_ranges: "[T]::M",
            ("MemberAccess", "M", true, 6, 5, 3, None),
            (4, Generic, "M", 5, 6), [];
        parse_static_ascii_call_keeps_member_and_empty_args: "[T]::M()",
            ("InvokeMember", "M", true, 8, 7, 3, Some(0)),
            (4, Generic, "M", 5, 6), [];
        parse_static_keyword_member_retains_keyword_acceptance: "[T]::return",
            ("MemberAccess", "return", true, 11, 5, 3, None),
            (4, Keyword, "return", 5, 11), [];
        parse_static_digit_initial_generic_retains_acceptance: "[T]::9Lives",
            ("MemberAccess", "9Lives", true, 11, 5, 3, None),
            (4, Generic, "9Lives", 5, 11), [];
        parse_static_ascii_alnum_underscore_keeps_literal_member: "[T]::_M2",
            ("MemberAccess", "_M2", true, 8, 5, 3, None),
            (4, Generic, "_M2", 5, 8), [];
        parse_static_digit_initial_unicode_generic_keeps_raw_member: "[T]::9猫_2",
            ("MemberAccess", "9猫_2", true, 11, 5, 3, None),
            (4, Generic, "9猫_2", 5, 11), [];
        parse_static_unicode_initial_keeps_raw_member_and_byte_span: "[T]::猫",
            ("MemberAccess", "猫", true, 8, 5, 3, None),
            (4, Generic, "猫", 5, 8), [];
        parse_static_unicode_call_keeps_raw_member_and_empty_args: "[T]::猫()",
            ("InvokeMember", "猫", true, 10, 7, 3, Some(0)),
            (4, Generic, "猫", 5, 8), [];
        parse_static_mixed_unicode_alnum_underscore_keeps_raw_member: "[T]::M猫_2",
            ("MemberAccess", "M猫_2", true, 11, 5, 3, None),
            (4, Generic, "M猫_2", 5, 11), [];
        parse_static_accented_member_keeps_raw_spelling: "[T]::Méthod",
            ("MemberAccess", "Méthod", true, 12, 5, 3, None),
            (4, Generic, "Méthod", 5, 12), [];
        parse_static_punctuation_generic_retains_current_acceptance: "[T]::Foo-Bar",
            ("MemberAccess", "Foo-Bar", true, 12, 5, 3, None),
            (4, Generic, "Foo-Bar", 5, 12), [];
        parse_static_combining_mark_keeps_unnormalized_generic: "[T]::e\u{301}",
            ("MemberAccess", "e\u{301}", true, 8, 5, 3, None),
            (4, Generic, "e\u{301}", 5, 8), [];
        parse_static_whitespace_keeps_member_and_token_range: "[T] \t:: \tM",
            ("MemberAccess", "M", true, 10, 5, 3, None),
            (4, Generic, "M", 9, 10), [];
        parse_static_comment_before_separator_keeps_member: "[T]<# gap #>::M",
            ("MemberAccess", "M", true, 15, 5, 3, None),
            (4, Generic, "M", 14, 15), [];
        parse_static_comment_after_separator_keeps_member: "[T]::<# gap #>M",
            ("MemberAccess", "M", true, 15, 5, 3, None),
            (4, Generic, "M", 14, 15), [];
        parse_static_continuation_before_separator_keeps_member: "[T]`\n::M",
            ("MemberAccess", "M", true, 8, 5, 3, None),
            (4, Generic, "M", 7, 8), [];
        parse_static_continuation_after_separator_keeps_member: "[T]::`\nM",
            ("MemberAccess", "M", true, 8, 5, 3, None),
            (4, Generic, "M", 7, 8), [];
        parse_static_line_comment_after_separator_retains_current_acceptance: "[T]::# gap\nM",
            ("MemberAccess", "M", true, 12, 5, 3, None),
            (4, Generic, "M", 11, 12), [];
        parse_instance_access_keeps_nonstatic_flag: "$obj.M",
            ("MemberAccess", "M", false, 6, 3, 4, None),
            (2, Generic, "M", 5, 6), [];
        parse_other_static_receiver_keeps_receiver_span: "[Other]::M",
            ("MemberAccess", "M", true, 10, 5, 7, None),
            (4, Generic, "M", 9, 10), [];
        parse_dynamic_member_keeps_variable_text: "[T]::$name",
            ("MemberAccess", "$name", true, 10, 5, 3, None),
            (4, Variable, "$name", 5, 10), [];
        parse_quoted_member_reports_error_and_keeps_empty_member: "[T]::'M'",
            ("MemberAccess", "", true, 5, 4, 3, None),
            (4, StringSq, "'M'", 5, 8), [("expected member name", 5, 8)];
        parse_missing_member_reports_eof_and_keeps_empty_member: "[T]::",
            ("MemberAccess", "", true, 5, 4, 3, None),
            (4, Eof, "", 5, 5), [("expected member name", 5, 5)];
        parse_number_member_remains_unaccepted: "[T]::123",
            ("MemberAccess", "", true, 5, 4, 3, None),
            (4, Number, "123", 5, 8), [("expected member name", 5, 8)];
        parse_unclosed_member_call_keeps_recovered_call_and_diagnostic: "[T]::M(",
            ("InvokeMember", "M", true, 7, 6, 3, Some(0)),
            (4, Generic, "M", 5, 6), [("expected ')'", 7, 7)];
    }

    type MemberSpanShape = (
        &'static str,
        String,
        bool,
        Option<Span>,
        Span,
        TokenRange,
        Span,
        Option<usize>,
    );
    type DetachedSpanShape = (&'static str, Span, TokenRange);

    fn assert_member_spans(
        out: ParseOutput,
        src: &str,
        expected_members: Vec<MemberSpanShape>,
        expected_detached: Vec<DetachedSpanShape>,
        errors: Vec<ParseError>,
    ) {
        let mut members = Vec::new();
        let mut detached = Vec::new();
        out.script.walk(&mut |node| {
            if node.range.is_empty() {
                detached.push((node.label(), node.span, node.range));
            }
            let (target, member, member_span, is_static, args) = match &node.kind {
                NodeKind::MemberAccess {
                    target,
                    member,
                    member_span,
                    is_static,
                } => (target, member, *member_span, *is_static, None),
                NodeKind::InvokeMember {
                    target,
                    member,
                    member_span,
                    is_static,
                    args,
                } => (target, member, *member_span, *is_static, Some(args.len())),
                _ => return,
            };
            members.push((
                node.label(),
                member.clone(),
                is_static,
                member_span,
                node.span,
                node.range,
                target.span,
                args,
            ));
        });

        assert_eq!(
            (members, detached),
            (expected_members, expected_detached),
            "{src:?}"
        );
        assert_eq!(out.errors, errors, "{src:?}");
        assert_eq!(crate::v2::reconstruct(&out.tokens), src);
    }

    // Both public entry points must expose literal original-input coordinates.
    // Expected offsets are constants, never found by rescanning member text.
    macro_rules! member_span_cases {
        ($(($parse:ident, $tokens:ident): $src:expr,
            [$(($label:expr, $member:expr, $static:expr, $member_span:expr,
                ($start:expr, $end:expr), ($first:expr, $last:expr),
                ($target_start:expr, $target_end:expr), $args:expr)),* $(,)?],
            [$(($tag:expr, $ds:expr, $de:expr, $idx:expr)),* $(,)?],
            [$(($error:expr, $es:expr, $ee:expr)),* $(,)?];)*) => {
            $(
                #[test]
                fn $parse() {
                    let src = $src;

                    let out = parse(src);

                    member_span_cases!(@assert out, src,
                        [$(($label, $member, $static, $member_span,
                            ($start, $end), ($first, $last),
                            ($target_start, $target_end), $args)),*],
                        [$(($tag, $ds, $de, $idx)),*],
                        [$(($error, $es, $ee)),*]);
                }

                #[test]
                fn $tokens() {
                    let src = $src;
                    let lexed = lex(src);

                    let out = parse_tokens(src, lexed.tokens);

                    member_span_cases!(@assert out, src,
                        [$(($label, $member, $static, $member_span,
                            ($start, $end), ($first, $last),
                            ($target_start, $target_end), $args)),*],
                        [$(($tag, $ds, $de, $idx)),*],
                        [$(($error, $es, $ee)),*]);
                }
            )*
        };
        (@assert $out:expr, $src:expr,
            [$(($label:expr, $member:expr, $static:expr, $member_span:expr,
                ($start:expr, $end:expr), ($first:expr, $last:expr),
                ($target_start:expr, $target_end:expr), $args:expr)),*],
            [$(($tag:expr, $ds:expr, $de:expr, $idx:expr)),*],
            [$(($error:expr, $es:expr, $ee:expr)),*]) => {
            assert_member_spans($out, $src,
                vec![$(($label, $member.into(), $static, $member_span,
                    Span::new($start, $end), TokenRange { first: $first, end: $last },
                    Span::new($target_start, $target_end), $args)),*],
                vec![$(($tag, Span::new($ds, $de),
                    TokenRange { first: $idx, end: $idx })),*],
                vec![$(ParseError { message: $error.into(), span: Span::new($es, $ee) }),*]);
        };
    }

    member_span_cases! {
        (parse_member_span_static_access, parse_tokens_member_span_static_access): "[T]::M",
            [("MemberAccess", "M", true, Some(Span::new(5, 6)), (0, 6), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_static_call, parse_tokens_member_span_static_call): "[T]::M()",
            [("InvokeMember", "M", true, Some(Span::new(5, 6)), (0, 8), (0, 7), (0, 3), Some(0))], [], [];
        (parse_member_span_instance_access, parse_tokens_member_span_instance_access): "$obj.M",
            [("MemberAccess", "M", false, Some(Span::new(5, 6)), (0, 6), (0, 3), (0, 4), None)], [], [];
        (parse_member_span_instance_call, parse_tokens_member_span_instance_call): "$obj.M()",
            [("InvokeMember", "M", false, Some(Span::new(5, 6)), (0, 8), (0, 5), (0, 4), Some(0))], [], [];
        (parse_member_span_static_keyword, parse_tokens_member_span_static_keyword): "[T]::return",
            [("MemberAccess", "return", true, Some(Span::new(5, 11)), (0, 11), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_static_keyword_call, parse_tokens_member_span_static_keyword_call): "[T]::return()",
            [("InvokeMember", "return", true, Some(Span::new(5, 11)), (0, 13), (0, 7), (0, 3), Some(0))], [], [];
        (parse_member_span_instance_keyword, parse_tokens_member_span_instance_keyword): "$obj.return",
            [("MemberAccess", "return", false, Some(Span::new(5, 11)), (0, 11), (0, 3), (0, 4), None)], [], [];
        (parse_member_span_instance_keyword_call, parse_tokens_member_span_instance_keyword_call): "$obj.return()",
            [("InvokeMember", "return", false, Some(Span::new(5, 11)), (0, 13), (0, 5), (0, 4), Some(0))], [], [];
        (parse_member_span_unicode, parse_tokens_member_span_unicode): "[T]::猫",
            [("MemberAccess", "猫", true, Some(Span::new(5, 8)), (0, 8), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_unicode_call, parse_tokens_member_span_unicode_call): "$obj.猫()",
            [("InvokeMember", "猫", false, Some(Span::new(5, 8)), (0, 10), (0, 5), (0, 4), Some(0))], [], [];
        (parse_member_span_digit_initial_unicode, parse_tokens_member_span_digit_initial_unicode): "[T]::9猫_2",
            [("MemberAccess", "9猫_2", true, Some(Span::new(5, 11)), (0, 11), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_ascii_underscore, parse_tokens_member_span_ascii_underscore): "[T]::_M2",
            [("MemberAccess", "_M2", true, Some(Span::new(5, 8)), (0, 8), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_broad_generic, parse_tokens_member_span_broad_generic): "[T]::Foo-Bar",
            [("MemberAccess", "Foo-Bar", true, Some(Span::new(5, 12)), (0, 12), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_combining_mark, parse_tokens_member_span_combining_mark): "[T]::e\u{301}()",
            [("InvokeMember", "e\u{301}", true, Some(Span::new(5, 8)), (0, 10), (0, 7), (0, 3), Some(0))], [], [];
        (parse_member_span_bom_prefix_unicode_crlf, parse_tokens_member_span_bom_prefix_unicode_crlf):
            "\u{feff}'猫';\r\n[T]::Méthod()",
            [("InvokeMember", "Méthod", true, Some(Span::new(16, 23)), (11, 25), (2, 9), (11, 14), Some(0))], [], [];
        (parse_member_span_instance_prefix_unicode_crlf, parse_tokens_member_span_instance_prefix_unicode_crlf):
            "\"猫\";\r\n$obj.M猫_2()",
            [("InvokeMember", "M猫_2", false, Some(Span::new(13, 19)), (8, 21), (2, 7), (8, 12), Some(0))], [], [];
        (parse_member_span_whitespace, parse_tokens_member_span_whitespace): "[T] \t:: \tM",
            [("MemberAccess", "M", true, Some(Span::new(9, 10)), (0, 10), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_comment_before, parse_tokens_member_span_comment_before): "[T]<# gap #>::M",
            [("MemberAccess", "M", true, Some(Span::new(14, 15)), (0, 15), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_comment_after, parse_tokens_member_span_comment_after): "[T]::<# gap #>M",
            [("MemberAccess", "M", true, Some(Span::new(14, 15)), (0, 15), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_continuation_before, parse_tokens_member_span_continuation_before): "[T]`\n::M",
            [("MemberAccess", "M", true, Some(Span::new(7, 8)), (0, 8), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_continuation_after, parse_tokens_member_span_continuation_after): "[T]::`\nM",
            [("MemberAccess", "M", true, Some(Span::new(7, 8)), (0, 8), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_crlf_continuation_call, parse_tokens_member_span_crlf_continuation_call): "[T]::`\r\n猫()",
            [("InvokeMember", "猫", true, Some(Span::new(8, 11)), (0, 13), (0, 7), (0, 3), Some(0))], [], [];
        (parse_member_span_line_comment, parse_tokens_member_span_line_comment): "[T]::# gap\nM",
            [("MemberAccess", "M", true, Some(Span::new(11, 12)), (0, 12), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_variable_is_none, parse_tokens_member_span_variable_is_none): "[T]::$name",
            [("MemberAccess", "$name", true, None, (0, 10), (0, 5), (0, 3), None)], [], [];
        (parse_member_span_variable_call_is_none, parse_tokens_member_span_variable_call_is_none): "$obj.$name()",
            [("InvokeMember", "$name", false, None, (0, 12), (0, 5), (0, 4), Some(0))], [], [];
        (parse_member_span_missing_is_none, parse_tokens_member_span_missing_is_none): "[T]::",
            [("MemberAccess", "", true, None, (0, 5), (0, 4), (0, 3), None)],
            [], [("expected member name", 5, 5)];
        (parse_member_span_single_quoted_is_none, parse_tokens_member_span_single_quoted_is_none): "[T]::'M'",
            [("MemberAccess", "", true, None, (0, 5), (0, 4), (0, 3), None)],
            [], [("expected member name", 5, 8)];
        (parse_member_span_double_quoted_is_none, parse_tokens_member_span_double_quoted_is_none): "[T]::\"M\"",
            [("MemberAccess", "", true, None, (0, 5), (0, 4), (0, 3), None)],
            [], [("expected member name", 5, 8)];
        (parse_member_span_computed_is_none, parse_tokens_member_span_computed_is_none): "[T]::('M')",
            [("InvokeMember", "", true, None, (0, 10), (0, 7), (0, 3), Some(1))],
            [], [("expected member name", 5, 6)];
        (parse_member_span_number_is_none, parse_tokens_member_span_number_is_none): "[T]::123",
            [("MemberAccess", "", true, None, (0, 5), (0, 4), (0, 3), None)],
            [], [("expected member name", 5, 8)];
        (parse_member_span_incomplete_call_keeps_name, parse_tokens_member_span_incomplete_call_keeps_name): "[T]::M(",
            [("InvokeMember", "M", true, Some(Span::new(5, 6)), (0, 7), (0, 6), (0, 3), Some(0))],
            [], [("expected ')'", 7, 7)];
        (parse_member_span_incomplete_args_keep_name, parse_tokens_member_span_incomplete_args_keep_name): "[T]::M(1",
            [("InvokeMember", "M", true, Some(Span::new(5, 6)), (0, 8), (0, 7), (0, 3), Some(1))],
            [], [("expected ')'", 8, 8)];
        (parse_member_span_incomplete_unicode_call_keeps_name, parse_tokens_member_span_incomplete_unicode_call_keeps_name): "$obj.猫(",
            [("InvokeMember", "猫", false, Some(Span::new(5, 8)), (0, 9), (0, 4), (0, 4), Some(0))],
            [], [("expected ')'", 9, 9)];
    }

    member_span_cases! {
        (parse_member_span_double_fragment, parse_tokens_member_span_double_fragment): r#""$([T]::M)""#,
            [("MemberAccess", "M", true, Some(Span::new(8, 9)), (3, 9), (0, 0), (3, 6), None)],
            [("SubExpression", 1, 10, 0), ("ScriptBlock", 3, 9, 0),
             ("MemberAccess", 3, 9, 0), ("TypeExpression", 3, 6, 0)], [];
        (parse_member_span_unicode_fragment_call, parse_tokens_member_span_unicode_fragment_call): r#""é $([T]::猫())""#,
            [("InvokeMember", "猫", true, Some(Span::new(11, 14)), (6, 16), (0, 0), (6, 9), Some(0))],
            [("SubExpression", 4, 17, 0), ("ScriptBlock", 6, 16, 0),
             ("InvokeMember", 6, 16, 0), ("TypeExpression", 6, 9, 0)], [];
        (parse_member_span_here_fragment_crlf, parse_tokens_member_span_here_fragment_crlf): "@\"\r\n猫 $($obj.M)\r\n\"@",
            [("MemberAccess", "M", false, Some(Span::new(15, 16)), (10, 16), (0, 0), (10, 14), None)],
            [("SubExpression", 8, 17, 0), ("ScriptBlock", 10, 16, 0),
             ("MemberAccess", 10, 16, 0), ("Variable", 10, 14, 0)], [];
        (parse_member_span_repeated_names_are_distinct, parse_tokens_member_span_repeated_names_are_distinct):
            r#"[T]::M;"$([T]::M) $([T]::M())""#,
            [("MemberAccess", "M", true, Some(Span::new(5, 6)), (0, 6), (0, 5), (0, 3), None),
             ("MemberAccess", "M", true, Some(Span::new(15, 16)), (10, 16), (6, 6), (10, 13), None),
             ("InvokeMember", "M", true, Some(Span::new(25, 26)), (20, 28), (6, 6), (20, 23), Some(0))],
            [("SubExpression", 8, 17, 6), ("ScriptBlock", 10, 16, 6),
             ("MemberAccess", 10, 16, 6), ("TypeExpression", 10, 13, 6),
             ("SubExpression", 18, 29, 6), ("ScriptBlock", 20, 28, 6),
             ("InvokeMember", 20, 28, 6), ("TypeExpression", 20, 23, 6)], [];
        (parse_member_span_nested_double_bom, parse_tokens_member_span_nested_double_bom):
            "\u{feff}'猫';\r\n\"$(\"$([T]::猫())\")\"",
            [("InvokeMember", "猫", true, Some(Span::new(22, 25)), (17, 27), (2, 2), (17, 20), Some(0))],
            [("SubExpression", 12, 30, 2), ("ScriptBlock", 14, 29, 2),
             ("StringLiteral", 14, 29, 2), ("SubExpression", 15, 28, 2),
             ("ScriptBlock", 17, 27, 2), ("InvokeMember", 17, 27, 2),
             ("TypeExpression", 17, 20, 2)], [];
        (parse_member_span_nested_here_in_double, parse_tokens_member_span_nested_here_in_double):
            "\"$(@\"\r\n$([T]::M)\r\n\"@)\"",
            [("MemberAccess", "M", true, Some(Span::new(14, 15)), (9, 15), (0, 0), (9, 12), None)],
            [("SubExpression", 1, 21, 0), ("ScriptBlock", 3, 20, 0),
             ("StringLiteral", 3, 20, 0), ("SubExpression", 7, 16, 0),
             ("ScriptBlock", 9, 15, 0), ("MemberAccess", 9, 15, 0),
             ("TypeExpression", 9, 12, 0)], [];
        (parse_member_span_nested_double_in_here, parse_tokens_member_span_nested_double_in_here):
            "@\"\n$(\"$($obj.M())\")\n\"@",
            [("InvokeMember", "M", false, Some(Span::new(13, 14)), (8, 16), (0, 0), (8, 12), Some(0))],
            [("SubExpression", 3, 19, 0), ("ScriptBlock", 5, 18, 0),
             ("StringLiteral", 5, 18, 0), ("SubExpression", 6, 17, 0),
             ("ScriptBlock", 8, 16, 0), ("InvokeMember", 8, 16, 0),
             ("Variable", 8, 12, 0)], [];
        (parse_member_span_empty_fragment_keeps_ranges, parse_tokens_member_span_empty_fragment_keeps_ranges):
            r#""$() $([T]::M)""#,
            [("MemberAccess", "M", true, Some(Span::new(12, 13)), (7, 13), (0, 0), (7, 10), None)],
            [("SubExpression", 1, 4, 0), ("ScriptBlock", 3, 3, 0),
             ("SubExpression", 5, 14, 0), ("ScriptBlock", 7, 13, 0),
             ("MemberAccess", 7, 13, 0), ("TypeExpression", 7, 10, 0)], [];
        (parse_member_span_nested_normal_fragment, parse_tokens_member_span_nested_normal_fragment):
            r#""$($([T]::M))""#,
            [("MemberAccess", "M", true, Some(Span::new(10, 11)), (5, 11), (0, 0), (5, 8), None)],
            [("SubExpression", 1, 13, 0), ("ScriptBlock", 3, 12, 0),
             ("SubExpression", 3, 12, 0), ("ScriptBlock", 5, 11, 0),
             ("MemberAccess", 5, 11, 0), ("TypeExpression", 5, 8, 0)], [];
        (parse_member_span_chained_fragment_names_rebase_once, parse_tokens_member_span_chained_fragment_names_rebase_once):
            r#""$($obj.M().M)""#,
            [("MemberAccess", "M", false, Some(Span::new(12, 13)), (3, 13), (0, 0), (3, 11), None),
             ("InvokeMember", "M", false, Some(Span::new(8, 9)), (3, 11), (0, 0), (3, 7), Some(0))],
            [("SubExpression", 1, 14, 0), ("ScriptBlock", 3, 13, 0),
             ("MemberAccess", 3, 13, 0), ("InvokeMember", 3, 11, 0),
             ("Variable", 3, 7, 0)], [];
        (parse_member_span_fragment_variable_stays_none, parse_tokens_member_span_fragment_variable_stays_none):
            r#""$([T]::$name)""#,
            [("MemberAccess", "$name", true, None, (3, 13), (0, 0), (3, 6), None)],
            [("SubExpression", 1, 14, 0), ("ScriptBlock", 3, 13, 0),
             ("MemberAccess", 3, 13, 0), ("TypeExpression", 3, 6, 0)], [];
        (parse_member_span_fragment_missing_stays_none, parse_tokens_member_span_fragment_missing_stays_none):
            r#""$([T]::)""#,
            [("MemberAccess", "", true, None, (3, 8), (0, 0), (3, 6), None)],
            [("SubExpression", 1, 9, 0), ("ScriptBlock", 3, 8, 0),
             ("MemberAccess", 3, 8, 0), ("TypeExpression", 3, 6, 0)], [];
        (parse_member_span_fragment_quoted_stays_none, parse_tokens_member_span_fragment_quoted_stays_none):
            r#""$([T]::'M')""#,
            [("MemberAccess", "", true, None, (3, 8), (0, 0), (3, 6), None)],
            [("SubExpression", 1, 12, 0), ("ScriptBlock", 3, 11, 0),
             ("MemberAccess", 3, 8, 0), ("TypeExpression", 3, 6, 0),
             ("StringLiteral", 8, 11, 0)], [];
        (parse_member_span_fragment_computed_stays_none, parse_tokens_member_span_fragment_computed_stays_none):
            r#""$([T]::('M'))""#,
            [("InvokeMember", "", true, None, (3, 13), (0, 0), (3, 6), Some(1))],
            [("SubExpression", 1, 14, 0), ("ScriptBlock", 3, 13, 0),
             ("InvokeMember", 3, 13, 0), ("TypeExpression", 3, 6, 0),
             ("StringLiteral", 9, 12, 0)], [];
    }

    #[test]
    fn collections_and_subexpressions() {
        clean("$h = @{ a = 1; b = @(2, 3) }\n");
        clean("$arr = @(1, 2, 3)\n");
        clean("\"value is $($x.Prop)\"\n");
        clean("$sb = { param($n) $n * 2 }\n");
    }

    #[test]
    fn pipeline_chains_and_flow() {
        clean("Test-Path $p && Write-Output 'ok' || Write-Error 'no'\n");
        clean("return $result\n");
        clean("throw 'bad'\n");
    }

    #[test]
    fn call_operators() {
        let root = clean("& $command --arg\n");
        assert_eq!(root.children()[0].label(), "Command");
        clean(". .\\setup.ps1\n");
    }

    #[test]
    fn empty_and_comment_only() {
        assert!(parse("").errors.is_empty());
        assert!(parse("# just a comment\n").errors.is_empty());
        assert_eq!(parse("").script.label(), "ScriptBlock");
    }

    #[test]
    fn definitions_class_enum_using() {
        let root = clean(
            "class Point { [int]$X; Point([int]$x) { $this.X = $x } [int] Get() { return $this.X } }\n",
        );
        let class = &root.children()[0];
        assert_eq!(class.label(), "ClassDefinition");
        // a property, a constructor, and a method
        assert_eq!(class.children().len(), 3);

        let e = clean("enum Color { Red; Green = 2; Blue }\n");
        assert_eq!(e.children()[0].label(), "EnumDefinition");

        let u = clean("using namespace System.Collections.Generic\n");
        assert_eq!(u.children()[0].label(), "UsingStatement");

        let w = clean("workflow Flow { Get-Service }\n");
        assert_eq!(w.children()[0].label(), "FunctionDefinition");
    }

    #[test]
    fn parameter_arguments_and_redirections() {
        let root = clean("Get-Item -Path 'x' -Force > out.txt\n");
        let cmd = &root.children()[0];
        if let NodeKind::Command {
            elements,
            redirections,
            ..
        } = &cmd.kind
        {
            // -Path with a bound argument, then a switch parameter -Force
            assert!(matches!(
                elements[0].kind,
                NodeKind::CommandParameter {
                    argument: Some(_),
                    ..
                }
            ));
            assert!(matches!(
                elements[1].kind,
                NodeKind::CommandParameter { argument: None, .. }
            ));
            assert_eq!(redirections.len(), 1);
        } else {
            panic!("not a command");
        }
    }

    #[test]
    fn trap_and_dynamicparam_are_script_block_expressions() {
        let root = clean("trap { 'oops' }\n");
        assert_eq!(root.children()[0].label(), "ScriptBlockExpression");
    }

    #[test]
    fn string_interpolation_parts() {
        // a variable part and a subexpression part are extracted from a
        // double-quoted string; single-quoted strings have none.
        let root = clean("$s = \"hi $name and $($x.Prop)\"\n");
        let mut labels = Vec::new();
        root.walk(&mut |n| {
            if let NodeKind::StringLiteral { parts, .. } = &n.kind {
                for p in parts {
                    labels.push(p.label());
                }
            }
        });
        assert_eq!(labels, ["Variable", "SubExpression"]);

        let lit = clean("$t = 'no $interpolation here'\n");
        lit.walk(&mut |n| {
            if let NodeKind::StringLiteral { parts, .. } = &n.kind {
                assert!(parts.is_empty(), "single-quoted strings have no parts");
            }
        });
    }

    // Compare the public tree's raw payloads and coordinates, not fragment-local
    // tokens. Every descendant of an interpolation belongs to the outer token.
    fn interpolation_shape(node: &Node) -> (&'static str, String, Span, TokenRange) {
        let (tag, raw) = match &node.kind {
            NodeKind::Script(_) => ("script", ""),
            NodeKind::SubExpression(_) => ("sub", ""),
            NodeKind::StringLiteral { kind, value, .. } => (
                match kind {
                    StringKind::Single => "single",
                    StringKind::Double => "double",
                    StringKind::HereSingle => "here-single",
                    StringKind::HereDouble => "here-double",
                },
                value.as_str(),
            ),
            NodeKind::Variable(raw) => ("variable", raw.as_str()),
            NodeKind::Number(raw) => ("number", raw.as_str()),
            NodeKind::Binary { op, .. } => ("binary", op.as_str()),
            NodeKind::BareWord(raw) => ("bareword", raw.as_str()),
            NodeKind::Command { .. } => ("command", ""),
            NodeKind::Error(raw) => ("error", raw.as_str()),
            other => panic!("unexpected interpolation regression node: {other:?}"),
        };
        (tag, raw.into(), node.span, node.range)
    }

    macro_rules! subexpression_boundary_cases {
        ($(($parse:ident, $tokens:ident): $src:literal,
            [$(($tag:literal, $raw:literal, $start:literal, $end:literal,
                $first:literal, $last:literal)),* $(,)?],
            [$(($message:literal, $err_start:literal, $err_end:literal)),* $(,)?];)*) => {
            $(
                #[test]
                fn $parse() {
                    let src = $src;

                    let out = parse(src);

                    assert_subexpression_boundary(out, src,
                        vec![$(($tag, $raw.into(), Span::new($start, $end),
                            TokenRange { first: $first, end: $last })),*],
                        vec![$(ParseError { message: $message.into(),
                            span: Span::new($err_start, $err_end) }),*]);
                }

                #[test]
                fn $tokens() {
                    let src = $src;
                    let lexed = lex(src);

                    let out = parse_tokens(src, lexed.tokens);

                    // parse_tokens does not import lexer errors; neither entry
                    // point imports boundary-probe or fragment diagnostics.
                    assert_subexpression_boundary(out, src,
                        vec![$(($tag, $raw.into(), Span::new($start, $end),
                            TokenRange { first: $first, end: $last })),*],
                        vec![]);
                }
            )*
        };
    }

    fn assert_subexpression_boundary(
        out: ParseOutput,
        src: &str,
        expected: Vec<(&'static str, String, Span, TokenRange)>,
        errors: Vec<ParseError>,
    ) {
        let NodeKind::Script(statements) = &out.script.kind else {
            panic!("expected script, got {:?}", out.script);
        };
        let actual: Vec<_> = statements
            .iter()
            .flat_map(|statement| {
                let mut shapes = Vec::new();
                statement.walk(&mut |node| shapes.push(interpolation_shape(node)));
                shapes
            })
            .collect();

        assert_eq!(actual, expected);
        assert_eq!(out.errors, errors);
        assert_eq!(crate::v2::reconstruct(&out.tokens), src);
    }

    subexpression_boundary_cases! {
        (parse_boundary_single_quote, parse_tokens_boundary_single_quote):
            r#""$(')') $tail";42"#, [
                ("double", r#""$(')') $tail""#, 0, 14, 0, 1),
                ("sub", "", 1, 7, 0, 0),
                ("script", "", 3, 6, 0, 0),
                ("single", "')'", 3, 6, 0, 0),
                ("variable", "$tail", 8, 13, 0, 0),
                ("number", "42", 15, 17, 2, 3),
            ], [];
        (parse_boundary_double_quote, parse_tokens_boundary_double_quote):
            r#""$("(") $tail";42"#, [
                ("double", r#""$("(") $tail""#, 0, 14, 0, 1),
                ("sub", "", 1, 7, 0, 0),
                ("script", "", 3, 6, 0, 0),
                ("double", r#""(""#, 3, 6, 0, 0),
                ("variable", "$tail", 8, 13, 0, 0),
                ("number", "42", 15, 17, 2, 3),
            ], [];
        (parse_boundary_doubled_single_quote, parse_tokens_boundary_doubled_single_quote):
            r#""$('a'')(') $tail";42"#, [
                ("double", r#""$('a'')(') $tail""#, 0, 18, 0, 1),
                ("sub", "", 1, 11, 0, 0),
                ("script", "", 3, 10, 0, 0),
                ("single", "'a'')('", 3, 10, 0, 0),
                ("variable", "$tail", 12, 17, 0, 0),
                ("number", "42", 19, 21, 2, 3),
            ], [];
        (parse_boundary_doubled_double_quote, parse_tokens_boundary_doubled_double_quote):
            r#""$("a"")(") $tail";42"#, [
                ("double", r#""$("a"")(") $tail""#, 0, 18, 0, 1),
                ("sub", "", 1, 11, 0, 0),
                ("script", "", 3, 10, 0, 0),
                ("double", r#""a"")(""#, 3, 10, 0, 0),
                ("variable", "$tail", 12, 17, 0, 0),
                ("number", "42", 19, 21, 2, 3),
            ], [];
        (parse_boundary_line_comment, parse_tokens_boundary_line_comment):
            "\"$(1 # ) (\r\n; $é) $tail\";42", [
                ("double", "\"$(1 # ) (\r\n; $é) $tail\"", 0, 25, 0, 1),
                ("sub", "", 1, 18, 0, 0),
                ("script", "", 3, 17, 0, 0),
                ("number", "1", 3, 4, 0, 0),
                ("variable", "$é", 14, 17, 0, 0),
                ("variable", "$tail", 19, 24, 0, 0),
                ("number", "42", 26, 28, 2, 3),
            ], [];
        (parse_boundary_block_comment_first_terminator, parse_tokens_boundary_block_comment_first_terminator):
            "\"$(1 <# <# ) ( #> ; $é) $tail\";42", [
                ("double", "\"$(1 <# <# ) ( #> ; $é) $tail\"", 0, 31, 0, 1),
                ("sub", "", 1, 24, 0, 0),
                ("script", "", 3, 23, 0, 0),
                ("number", "1", 3, 4, 0, 0),
                ("variable", "$é", 20, 23, 0, 0),
                ("variable", "$tail", 25, 30, 0, 0),
                ("number", "42", 32, 34, 2, 3),
            ], [];
        (parse_boundary_code_escape, parse_tokens_boundary_code_escape):
            "\"$( `) ; $é) $tail\";42", [
                ("double", "\"$( `) ; $é) $tail\"", 0, 20, 0, 1),
                ("sub", "", 1, 13, 0, 0),
                ("script", "", 4, 12, 0, 0),
                ("command", "", 4, 6, 0, 0),
                ("bareword", "`)", 4, 6, 0, 0),
                ("variable", "$é", 9, 12, 0, 0),
                ("variable", "$tail", 14, 19, 0, 0),
                ("number", "42", 21, 23, 2, 3),
            ], [];
        (parse_boundary_single_here, parse_tokens_boundary_single_here):
            "\"$(@'\n) (\n'@) $tail\";42", [
                ("double", "\"$(@'\n) (\n'@) $tail\"", 0, 20, 0, 1),
                ("sub", "", 1, 13, 0, 0),
                ("script", "", 3, 12, 0, 0),
                ("here-single", "@'\n) (\n'@", 3, 12, 0, 0),
                ("variable", "$tail", 14, 19, 0, 0),
                ("number", "42", 21, 23, 2, 3),
            ], [];
        (parse_boundary_double_here, parse_tokens_boundary_double_here):
            "\"$(@\"\r\n) (\r\n\"@) $tail\";42", [
                ("double", "\"$(@\"\r\n) (\r\n\"@) $tail\"", 0, 22, 0, 1),
                ("sub", "", 1, 15, 0, 0),
                ("script", "", 3, 14, 0, 0),
                ("here-double", "@\"\r\n) (\r\n\"@", 3, 14, 0, 0),
                ("variable", "$tail", 16, 21, 0, 0),
                ("number", "42", 23, 25, 2, 3),
            ], [];
        (parse_boundary_here_crlf_unicode, parse_tokens_boundary_here_crlf_unicode):
            "@\"\r\né $(')') $tail\r\n\"@;42", [
                ("here-double", "@\"\r\né $(')') $tail\r\n\"@", 0, 23, 0, 1),
                ("sub", "", 7, 13, 0, 0),
                ("script", "", 9, 12, 0, 0),
                ("single", "')'", 9, 12, 0, 0),
                ("variable", "$tail", 14, 19, 0, 0),
                ("number", "42", 24, 26, 2, 3),
            ], [];
        (parse_boundary_nested_expandable, parse_tokens_boundary_nested_expandable):
            "\"$(\"$($é)\") $tail\";42", [
                ("double", "\"$(\"$($é)\") $tail\"", 0, 19, 0, 1),
                ("sub", "", 1, 12, 0, 0),
                ("script", "", 3, 11, 0, 0),
                ("double", "\"$($é)\"", 3, 11, 0, 0),
                ("sub", "", 4, 10, 0, 0),
                ("script", "", 6, 9, 0, 0),
                ("variable", "$é", 6, 9, 0, 0),
                ("variable", "$tail", 13, 18, 0, 0),
                ("number", "42", 20, 22, 2, 3),
            ], [];
        (parse_boundary_empty, parse_tokens_boundary_empty):
            "\"$()$tail\";42", [
                ("double", "\"$()$tail\"", 0, 10, 0, 1),
                ("sub", "", 1, 4, 0, 0),
                ("script", "", 3, 3, 0, 0),
                ("variable", "$tail", 4, 9, 0, 0),
                ("number", "42", 11, 13, 2, 3),
            ], [];
        (parse_boundary_nested_empty, parse_tokens_boundary_nested_empty):
            "\"$(\"$()\") $tail\";42", [
                ("double", "\"$(\"$()\") $tail\"", 0, 16, 0, 1),
                ("sub", "", 1, 9, 0, 0),
                ("script", "", 3, 8, 0, 0),
                ("double", "\"$()\"", 3, 8, 0, 0),
                ("sub", "", 4, 7, 0, 0),
                ("script", "", 6, 6, 0, 0),
                ("variable", "$tail", 10, 15, 0, 0),
                ("number", "42", 17, 19, 2, 3),
            ], [];
        (parse_boundary_unclosed_quote_eof, parse_tokens_boundary_unclosed_quote_eof):
            "\"$( 'é", [
                ("double", "\"$( 'é", 0, 7, 0, 1),
                ("sub", "", 1, 7, 0, 0),
                ("script", "", 4, 7, 0, 0),
                ("single", "'é", 4, 7, 0, 0),
            ], [
                ("unterminated string at 4..7", 4, 7),
                ("unterminated subexpression in string at 1..7", 1, 7),
                ("unterminated string at 0..7", 0, 7),
            ];
        (parse_boundary_unclosed_root_keeps_nested_closer, parse_tokens_boundary_unclosed_root_keeps_nested_closer):
            "\"$( $(1)", [
                ("double", "\"$( $(1)", 0, 8, 0, 1),
                ("sub", "", 1, 8, 0, 0),
                ("script", "", 4, 8, 0, 0),
                ("sub", "", 4, 8, 0, 0),
                ("script", "", 6, 7, 0, 0),
                ("number", "1", 6, 7, 0, 0),
            ], [
                ("unterminated subexpression in string at 1..8", 1, 8),
                ("unterminated string at 0..8", 0, 8),
            ];
        (parse_boundary_unclosed_here_fragment_keeps_remainder, parse_tokens_boundary_unclosed_here_fragment_keeps_remainder):
            "@\"\n$(')'; $é\n\"@;42", [
                ("here-double", "@\"\n$(')'; $é\n\"@", 0, 16, 0, 1),
                ("sub", "", 3, 13, 0, 0),
                ("script", "", 5, 13, 0, 0),
                ("single", "')'", 5, 8, 0, 0),
                ("variable", "$é", 10, 13, 0, 0),
                ("number", "42", 17, 19, 2, 3),
            ], [];
        (parse_boundary_fragment_lex_error_stays_local, parse_tokens_boundary_fragment_lex_error_stays_local):
            "\"$(0x) $tail\";42", [
                ("double", "\"$(0x) $tail\"", 0, 13, 0, 1),
                ("sub", "", 1, 6, 0, 0),
                ("script", "", 3, 5, 0, 0),
                ("number", "0x", 3, 5, 0, 0),
                ("variable", "$tail", 7, 12, 0, 0),
                ("number", "42", 14, 16, 2, 3),
            ], [];
        (parse_boundary_fragment_parse_error_stays_local, parse_tokens_boundary_fragment_parse_error_stays_local):
            "\"$(1 +) $tail\";42", [
                ("double", "\"$(1 +) $tail\"", 0, 14, 0, 1),
                ("sub", "", 1, 7, 0, 0),
                ("script", "", 3, 6, 0, 0),
                ("binary", "+", 3, 6, 0, 0),
                ("number", "1", 3, 4, 0, 0),
                ("error", "", 6, 6, 0, 0),
                ("variable", "$tail", 8, 13, 0, 0),
                ("number", "42", 15, 17, 2, 3),
            ], [];
        (parse_boundary_odd_even_backticks, parse_tokens_boundary_odd_even_backticks):
            "\"`$(no) ``$(')') $tail\";42", [
                ("double", "\"`$(no) ``$(')') $tail\"", 0, 23, 0, 1),
                ("sub", "", 10, 16, 0, 0),
                ("script", "", 12, 15, 0, 0),
                ("single", "')'", 12, 15, 0, 0),
                ("variable", "$tail", 17, 22, 0, 0),
                ("number", "42", 24, 26, 2, 3),
            ], [];
        (parse_boundary_bom_unicode_nonzero_token, parse_tokens_boundary_bom_unicode_nonzero_token):
            "\u{feff};\"é $(')') $tail\";42", [
                ("double", "\"é $(')') $tail\"", 4, 21, 1, 2),
                ("sub", "", 8, 14, 1, 1),
                ("script", "", 10, 13, 1, 1),
                ("single", "')'", 10, 13, 1, 1),
                ("variable", "$tail", 15, 20, 1, 1),
                ("number", "42", 22, 24, 3, 4),
            ], [];
    }

    #[derive(Debug, PartialEq, Eq)]
    struct VariablePart {
        raw: String,
        span: Span,
        range: TokenRange,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct VariableString {
        kind: StringKind,
        value: String,
        parts: Vec<VariablePart>,
    }

    fn assert_variable_string(
        out: ParseOutput,
        src: &str,
        kind: StringKind,
        value: &str,
        expected_parts: &[(&str, usize, usize)],
    ) {
        let mut strings = Vec::new();
        out.script.walk(&mut |node| {
            if let NodeKind::StringLiteral { kind, value, parts } = &node.kind {
                strings.push(VariableString {
                    kind: *kind,
                    value: value.clone(),
                    parts: parts
                        .iter()
                        .map(|part| {
                            let NodeKind::Variable(raw) = &part.kind else {
                                panic!("expected variable part, got {part:?}");
                            };
                            assert_eq!(part.span.slice(src), raw);
                            VariablePart {
                                raw: raw.clone(),
                                span: part.span,
                                range: part.range,
                            }
                        })
                        .collect(),
                });
            }
        });
        let expected = vec![VariableString {
            kind,
            value: value.into(),
            parts: expected_parts
                .iter()
                .map(|&(raw, start, end)| VariablePart {
                    raw: raw.into(),
                    span: Span::new(start, end),
                    range: TokenRange { first: 0, end: 0 },
                })
                .collect(),
        }];

        assert_eq!(out.errors, []);
        assert_eq!(strings, expected);
        assert_eq!(crate::v2::reconstruct(&out.tokens), src);
    }

    // Both public entry points have independent literal expectations: lexer
    // output is not the oracle for variable spellings or byte spans.
    macro_rules! variable_interpolation_cases {
        ($(($parse:ident, $tokens:ident): $src:literal, $value:literal, $kind:ident,
            [$(($raw:literal, $start:literal, $end:literal)),* $(,)?];)*) => {
            $(
                #[test]
                fn $parse() {
                    let src = $src;

                    let out = parse(src);

                    assert_variable_string(out, src, StringKind::$kind, $value,
                        &[$(($raw, $start, $end)),*]);
                }

                #[test]
                fn $tokens() {
                    let src = $src;
                    let lexed = lex(src);
                    assert_eq!(lexed.errors, []);

                    let out = parse_tokens(src, lexed.tokens);

                    assert_variable_string(out, src, StringKind::$kind, $value,
                        &[$(($raw, $start, $end)),*]);
                }
            )*
        };
    }

    variable_interpolation_cases! {
        (parse_variable_unicode_initial, parse_tokens_variable_unicode_initial):
            "\"$é\"", "\"$é\"", Double, [("$é", 1, 4)];
        (parse_variable_unicode_continuation, parse_tokens_variable_unicode_continuation):
            "\"$aé\"", "\"$aé\"", Double, [("$aé", 1, 5)];
        (parse_variable_nonlatin, parse_tokens_variable_nonlatin):
            "\"$变量\"", "\"$变量\"", Double, [("$变量", 1, 8)];
        (parse_variable_bom_byte_span, parse_tokens_variable_bom_byte_span):
            "\u{feff}\"$é\"", "\"$é\"", Double, [("$é", 4, 7)];
        (parse_variable_numeric, parse_tokens_variable_numeric):
            "\"$123\"", "\"$123\"", Double, [("$123", 1, 5)];
        (parse_variable_numeric_qualifier, parse_tokens_variable_numeric_qualifier):
            "\"$env:1\"", "\"$env:1\"", Double, [("$env:1", 1, 7)];
        (parse_variable_repeated_qualifiers, parse_tokens_variable_repeated_qualifiers):
            "\"$a:b:c\"", "\"$a:b:c\"", Double, [("$a:b:c", 1, 7)];
        (parse_variable_unicode_qualifier, parse_tokens_variable_unicode_qualifier):
            "\"$é:变量\"", "\"$é:变量\"", Double, [("$é:变量", 1, 11)];
        (parse_variable_specials_source_order, parse_tokens_variable_specials_source_order):
            "\"$_ $? $^ $$\"", "\"$_ $? $^ $$\"", Double,
            [("$_", 1, 3), ("$?", 4, 6), ("$^", 7, 9), ("$$", 10, 12)];
        (parse_variable_escaped_braced_terminator, parse_tokens_variable_escaped_braced_terminator):
            "\"${a`}b}\"", "\"${a`}b}\"", Double, [("${a`}b}", 1, 8)];
        (parse_variable_unicode_braced_terminator, parse_tokens_variable_unicode_braced_terminator):
            "\"${é`}界}\"", "\"${é`}界}\"", Double, [("${é`}界}", 1, 11)];
        (parse_variable_empty_braces, parse_tokens_variable_empty_braces):
            "\"${}\"", "\"${}\"", Double, [("${}", 1, 4)];
        (parse_variable_unclosed_braces, parse_tokens_variable_unclosed_braces):
            "\"${open\"", "\"${open\"", Double, [];
        (parse_variable_only_escaped_brace, parse_tokens_variable_only_escaped_brace):
            "\"${a`}\"", "\"${a`}\"", Double, [];
        (parse_variable_unsupported_dollars, parse_tokens_variable_unsupported_dollars):
            "\"$ $- $: $!\"", "\"$ $- $: $!\"", Double, [];
        (parse_variable_combining_mark_stops_name, parse_tokens_variable_combining_mark_stops_name):
            "\"$a\u{301}\"", "\"$a\u{301}\"", Double, [("$a", 1, 3)];
        (parse_variable_after_multibyte_text, parse_tokens_variable_after_multibyte_text):
            "\"🦀 $aé $9\"", "\"🦀 $aé $9\"", Double,
            [("$aé", 6, 10), ("$9", 11, 13)];
        (parse_variable_odd_backtick_escapes_dollar, parse_tokens_variable_odd_backtick_escapes_dollar):
            "\"`$é\"", "\"`$é\"", Double, [];
        (parse_variable_even_backticks_leave_dollar_active, parse_tokens_variable_even_backticks_leave_dollar_active):
            "\"``$é\"", "\"``$é\"", Double, [("$é", 3, 6)];
        (parse_variable_three_backticks_escape_dollar, parse_tokens_variable_three_backticks_escape_dollar):
            "\"```$é\"", "\"```$é\"", Double, [];
        (parse_variable_escaped_multibyte_character, parse_tokens_variable_escaped_multibyte_character):
            "\"`é$9\"", "\"`é$9\"", Double, [("$9", 4, 6)];
        (parse_variable_here_unicode_initial, parse_tokens_variable_here_unicode_initial):
            "@\"\n$é\n\"@", "@\"\n$é\n\"@", HereDouble, [("$é", 3, 6)];
        (parse_variable_here_crlf_continuation, parse_tokens_variable_here_crlf_continuation):
            "@\"\r\n$aé\r\n\"@", "@\"\r\n$aé\r\n\"@", HereDouble, [("$aé", 4, 8)];
        (parse_variable_here_numeric_and_qualifiers, parse_tokens_variable_here_numeric_and_qualifiers):
            "@\"\n$9 $a:b:c\n\"@", "@\"\n$9 $a:b:c\n\"@", HereDouble,
            [("$9", 3, 5), ("$a:b:c", 6, 12)];
        (parse_variable_here_specials, parse_tokens_variable_here_specials):
            "@\"\n$_ $? $^ $$\n\"@", "@\"\n$_ $? $^ $$\n\"@", HereDouble,
            [("$_", 3, 5), ("$?", 6, 8), ("$^", 9, 11), ("$$", 12, 14)];
        (parse_variable_here_escaped_brace, parse_tokens_variable_here_escaped_brace):
            "@\"\n${a`}b}\n\"@", "@\"\n${a`}b}\n\"@", HereDouble, [("${a`}b}", 3, 10)];
        (parse_variable_here_malformed_braces, parse_tokens_variable_here_malformed_braces):
            "@\"\n${a`}\n\"@", "@\"\n${a`}\n\"@", HereDouble, [];
        (parse_variable_here_dangling_braced_escape, parse_tokens_variable_here_dangling_braced_escape):
            "@\"\n${a`\n\"@", "@\"\n${a`\n\"@", HereDouble, [];
        (parse_variable_here_escaped_dollar, parse_tokens_variable_here_escaped_dollar):
            "@\"\n`$é\n\"@", "@\"\n`$é\n\"@", HereDouble, [];
        (parse_variable_here_even_backticks, parse_tokens_variable_here_even_backticks):
            "@\"\n``$é\n\"@", "@\"\n``$é\n\"@", HereDouble, [("$é", 5, 8)];
        (parse_variable_here_trailing_escape, parse_tokens_variable_here_trailing_escape):
            "@\"\n$é`\n\"@", "@\"\n$é`\n\"@", HereDouble, [("$é", 3, 6)];
        (parse_variable_single_quoted_stays_literal, parse_tokens_variable_single_quoted_stays_literal):
            "'$é ${a`}b}'", "'$é ${a`}b}'", Single, [];
        (parse_variable_here_single_stays_literal, parse_tokens_variable_here_single_stays_literal):
            "@'\n$é ${a`}b}\n'@", "@'\n$é ${a`}b}\n'@", HereSingle, [];
    }

    fn interpolation_depth_source(enclosing_blocks: usize, body: &str) -> String {
        format!(
            "{}\"{}\"{};42",
            "begin {".repeat(enclosing_blocks),
            body,
            "}".repeat(enclosing_blocks)
        )
    }

    fn assert_interpolation_depth(
        out: ParseOutput,
        src: &str,
        strings: Vec<(Span, usize)>,
        leaves: Vec<(&'static str, String, Span, TokenRange)>,
    ) {
        let mut actual_strings = Vec::new();
        let mut actual_leaves = Vec::new();
        // Exercise the ordinary public walk and let the whole output drop on
        // the test harness's stack; these fixtures are bounded even before fixing.
        out.script.walk(&mut |node| match &node.kind {
            NodeKind::StringLiteral { parts, .. } => {
                actual_strings.push((node.span, parts.len()));
            }
            NodeKind::Variable(_) | NodeKind::Number(_) | NodeKind::Error(_) => {
                actual_leaves.push(interpolation_shape(node));
            }
            NodeKind::Flow { keyword, .. } => {
                actual_leaves.push(("flow", keyword.clone(), node.span, node.range));
            }
            _ => {}
        });

        assert_eq!((actual_strings, actual_leaves), (strings, leaves));
        assert_eq!(out.errors, []);
        assert_eq!(crate::v2::reconstruct(&out.tokens), src);
    }

    // One public parse per case, with literal coordinates and no private depth
    // oracle. `begin` blocks add statement nesting without condition expressions.
    macro_rules! interpolation_depth_cases {
        ($(($parse:ident, $tokens:ident): $src:expr,
            [$(($string_start:literal, $string_end:literal, $parts:literal)),* $(,)?],
            [$(($tag:literal, $raw:literal, $start:literal, $end:literal,
                $first:literal, $last:literal)),* $(,)?];)*) => {
            $(
                #[test]
                fn $parse() {
                    let src = $src;

                    let out = parse(&src);

                    assert_interpolation_depth(out, &src,
                        vec![$((Span::new($string_start, $string_end), $parts)),*],
                        vec![$(($tag, $raw.into(), Span::new($start, $end),
                            TokenRange { first: $first, end: $last })),*]);
                }

                #[test]
                fn $tokens() {
                    let src = $src;
                    let lexed = lex(&src);
                    assert_eq!(lexed.errors, []);

                    let out = parse_tokens(&src, lexed.tokens);

                    assert_interpolation_depth(out, &src,
                        vec![$((Span::new($string_start, $string_end), $parts)),*],
                        vec![$(($tag, $raw.into(), Span::new($start, $end),
                            TokenRange { first: $first, end: $last })),*]);
                }
            )*
        };
    }

    interpolation_depth_cases! {
        (parse_depth_63_extracts_parts, parse_tokens_depth_63_extracts_parts):
            interpolation_depth_source(61, "$é $(1)"),
            [(427, 437, 2)], [
                ("variable", "$é", 428, 431, 122, 122),
                ("number", "1", 434, 435, 122, 122),
                ("number", "42", 499, 501, 185, 186),
            ];
        (parse_depth_64_silently_omits_parts, parse_tokens_depth_64_silently_omits_parts):
            interpolation_depth_source(62, "$é $(1)"),
            [(434, 444, 0)], [
                ("number", "42", 507, 509, 188, 189),
            ];
        (parse_depth_detached_levels_reach_cutoff, parse_tokens_depth_detached_levels_reach_cutoff):
            interpolation_depth_source(58, r#"$("$("$é")")"#),
            [(406, 421, 1), (409, 419, 1), (412, 417, 0)], [
                ("number", "42", 480, 482, 176, 177),
            ];
        (parse_depth_fragment_expression_200_allowed, parse_tokens_depth_fragment_expression_200_allowed):
            interpolation_depth_source(60,
                &format!("$({}$é{};42) $tail", "(".repeat(136), ")".repeat(136))),
            [(420, 709, 2)], [
                ("variable", "$é", 559, 562, 120, 120),
                ("number", "42", 699, 701, 120, 120),
                ("variable", "$tail", 703, 708, 120, 120),
                ("number", "42", 770, 772, 182, 183),
            ];
        (parse_depth_fragment_expression_201_recovers_locally, parse_tokens_depth_fragment_expression_201_recovers_locally):
            interpolation_depth_source(60,
                &format!("$({}$é{};42) $tail", "(".repeat(137), ")".repeat(137))),
            [(420, 711, 2)], [
                ("error", "too deep", 560, 563, 120, 120),
                ("error", "too deep", 701, 703, 120, 120),
                ("variable", "$tail", 705, 710, 120, 120),
                ("number", "42", 772, 774, 182, 183),
            ];
        (parse_depth_fragment_statement_200_allowed, parse_tokens_depth_fragment_statement_200_allowed):
            interpolation_depth_source(60,
                &format!("$({}break{};42) $tail", "begin {".repeat(137), "}".repeat(137))),
            [(420, 1535, 2)], [
                ("flow", "break", 1382, 1387, 120, 120),
                ("number", "42", 1525, 1527, 120, 120),
                ("variable", "$tail", 1529, 1534, 120, 120),
                ("number", "42", 1596, 1598, 182, 183),
            ];
        (parse_depth_fragment_statement_201_recovers_locally, parse_tokens_depth_fragment_statement_201_recovers_locally):
            interpolation_depth_source(60,
                &format!("$({}break;42{}) $tail", "begin {".repeat(138), "}".repeat(138))),
            [(420, 1543, 2)], [
                ("error", "too deep", 1389, 1394, 120, 120),
                ("error", "too deep", 1395, 1397, 120, 120),
                ("variable", "$tail", 1537, 1542, 120, 120),
                ("number", "42", 1604, 1606, 182, 183),
            ];
    }

    #[test]
    fn add_type_csharp_extraction() {
        // a DllImport is parsed out of the inline C#, including the function
        // signature, and the carrying parameter is recorded.
        let src = "Add-Type -MemberDefinition '[DllImport(\"user32.dll\")] public static extern int MessageBox(IntPtr hWnd, string text, string caption, uint type);' -Name Win\n";
        let out = parse(src);
        let mut found = None;
        out.script.walk(&mut |n| {
            if let NodeKind::CSharpMemberDef(c) = &n.kind {
                found = Some(c.clone());
            }
        });
        let cs = found.expect("expected a CSharpMemberDef");
        assert_eq!(cs.parameter, "memberdefinition");
        assert_eq!(cs.apis, ["MessageBox"]);
        assert_eq!(cs.imports.len(), 1);
        let imp = &cs.imports[0];
        assert_eq!(imp.dll, "user32.dll");
        assert_eq!(imp.function, "MessageBox");
        assert_eq!(imp.returns, "int");
        assert_eq!(imp.params.len(), 4);
        assert_eq!(imp.params[0].type_name, "IntPtr");
        assert_eq!(imp.params[0].name, "hWnd");

        // constant propagation: a variable assigned a literal resolves
        let via_var = parse("$c = 'class C {}'\nAdd-Type $c\n");
        let mut had = false;
        via_var.script.walk(&mut |n| {
            if let NodeKind::CSharpMemberDef(c) = &n.kind {
                had = true;
                assert_eq!(c.parameter, "positional");
                assert_eq!(c.code, "class C {}");
            }
        });
        assert!(had, "variable-carried C# should be resolved");

        // a non-Add-Type command has no csharp child
        let other = parse("Get-Process -Name foo\n");
        other.script.walk(&mut |n| {
            if let NodeKind::Command { csharp, .. } = &n.kind {
                assert!(csharp.is_none());
            }
        });
    }

    fn add_type_metadata(tree: &Node) -> Vec<Option<(String, String)>> {
        let mut metadata = Vec::new();
        tree.walk(&mut |n| {
            if let NodeKind::Command { name, csharp, .. } = &n.kind {
                if matches!(&name.kind, NodeKind::BareWord(name) if name.eq_ignore_ascii_case("add-type"))
                {
                    metadata.push(csharp.as_deref().map(|node| {
                        let NodeKind::CSharpMemberDef(cs) = &node.kind else {
                            panic!("expected CSharpMemberDef, got {node:?}");
                        };
                        (cs.code.clone(), cs.parameter.clone())
                    }));
                }
            }
        });
        metadata
    }

    fn assert_add_type_cache_invalidated_by(write: &str) {
        let src = format!("$Code = 'class Old {{}}'\nAdd-Type $code\n{write}\nAdd-Type $CODE\n");

        let out = parse(&src);

        assert_eq!(out.errors, Vec::<ParseError>::new());
        assert_eq!(
            add_type_metadata(&out.script),
            vec![Some(("class Old {}".into(), "positional".into())), None]
        );
    }

    #[test]
    fn add_type_cache_command_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe = Get-Content source.cs");
    }

    #[test]
    fn add_type_cache_variable_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe = $unknown");
    }

    #[test]
    fn add_type_cache_nonstring_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe = 42");
    }

    #[test]
    fn add_type_cache_add_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe += 'class New {}'");
    }

    #[test]
    fn add_type_cache_subtract_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe -= 'class New {}'");
    }

    #[test]
    fn add_type_cache_multiply_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe *= 'class New {}'");
    }

    #[test]
    fn add_type_cache_divide_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe /= 'class New {}'");
    }

    #[test]
    fn add_type_cache_remainder_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe %= 'class New {}'");
    }

    #[test]
    fn add_type_cache_coalescing_assignment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe ??= 'class New {}'");
    }

    #[test]
    fn add_type_cache_prefix_increment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("++$CoDe");
    }

    #[test]
    fn add_type_cache_prefix_decrement_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("--$CoDe");
    }

    #[test]
    fn add_type_cache_postfix_increment_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe++");
    }

    #[test]
    fn add_type_cache_postfix_decrement_invalidates_only_later_metadata() {
        assert_add_type_cache_invalidated_by("$CoDe--");
    }

    #[test]
    fn add_type_cache_fresh_literal_restores_metadata_after_invalidation() {
        let src = "$Code = 'class Old {}'\n$CoDe = 42\nAdd-Type $code\n\
                   $CODE = 'class New {}'\nAdd-Type $code\n";

        let out = parse(src);

        assert_eq!(out.errors, Vec::<ParseError>::new());
        assert_eq!(
            add_type_metadata(&out.script),
            vec![None, Some(("class New {}".into(), "positional".into()))]
        );
    }

    #[test]
    fn add_type_cache_write_to_another_variable_preserves_metadata() {
        let src = "$Code = 'class Old {}'\n$other = 42\nAdd-Type $code\n";

        let out = parse(src);

        assert_eq!(out.errors, Vec::<ParseError>::new());
        assert_eq!(
            add_type_metadata(&out.script),
            vec![Some(("class Old {}".into(), "positional".into()))]
        );
    }

    #[test]
    fn add_type_cache_assignment_rhs_sees_old_value_before_invalidation() {
        let src = "$Code = 'class Old {}'\n$CoDe = Add-Type $code\nAdd-Type $code\n";

        let out = parse(src);

        assert_eq!(out.errors, Vec::<ParseError>::new());
        assert_eq!(
            add_type_metadata(&out.script),
            vec![Some(("class Old {}".into(), "positional".into())), None]
        );
    }

    #[test]
    fn add_type_cache_compound_rhs_sees_old_value_before_invalidation() {
        let src = "$Code = 'class Old {}'\n$CoDe += Add-Type $code\nAdd-Type $code\n";

        let out = parse(src);

        assert_eq!(out.errors, Vec::<ParseError>::new());
        assert_eq!(
            add_type_metadata(&out.script),
            vec![Some(("class Old {}".into(), "positional".into())), None]
        );
    }

    macro_rules! nonconstant_add_type_cases {
        ($(($direct:ident, $cached:ident, $literal:literal)),+ $(,)?) => {
            $(
                #[test]
                fn $direct() {
                    let src = concat!("Add-Type -TypeDefinition ", $literal);

                    let out = parse(src);

                    assert_eq!(out.errors, Vec::<ParseError>::new());
                    assert_eq!(add_type_metadata(&out.script), vec![None]);
                }

                #[test]
                fn $cached() {
                    let src = concat!(
                        "$Code = 'class Old {}'\n$CODE = ", $literal,
                        "\nAdd-Type -TypeDefinition $code"
                    );

                    let out = parse(src);

                    assert_eq!(out.errors, Vec::<ParseError>::new());
                    assert_eq!(add_type_metadata(&out.script), vec![None]);
                }
            )+
        };
    }

    nonconstant_add_type_cases!(
        (
            add_type_direct_variable_interpolation_has_no_metadata,
            add_type_cached_variable_interpolation_has_no_metadata,
            "\"class $name {}\""
        ),
        (
            add_type_direct_braced_interpolation_has_no_metadata,
            add_type_cached_braced_interpolation_has_no_metadata,
            "\"class ${name} {}\""
        ),
        (
            add_type_direct_subexpression_has_no_metadata,
            add_type_cached_subexpression_has_no_metadata,
            "\"class $(Get-Name) {}\""
        ),
        (
            add_type_direct_unicode_reference_has_no_metadata,
            add_type_cached_unicode_reference_has_no_metadata,
            "\"class $变量 {}\""
        ),
        (
            add_type_direct_bare_dollar_has_no_metadata,
            add_type_cached_bare_dollar_has_no_metadata,
            "\"class A {} // $\""
        ),
        (
            add_type_direct_escaped_backtick_has_no_metadata,
            add_type_cached_escaped_backtick_has_no_metadata,
            "\"class ``$name {}\""
        ),
        (
            add_type_direct_escaped_then_active_dollar_has_no_metadata,
            add_type_cached_escaped_then_active_dollar_has_no_metadata,
            "\"`$literal $name\""
        ),
        (
            add_type_direct_here_variable_interpolation_has_no_metadata,
            add_type_cached_here_variable_interpolation_has_no_metadata,
            "@\"\nclass $name {}\n\"@"
        ),
        (
            add_type_direct_here_braced_interpolation_has_no_metadata,
            add_type_cached_here_braced_interpolation_has_no_metadata,
            "@\"\nclass ${name} {}\n\"@"
        ),
        (
            add_type_direct_here_subexpression_has_no_metadata,
            add_type_cached_here_subexpression_has_no_metadata,
            "@\"\nclass $(Get-Name) {}\n\"@"
        ),
        (
            add_type_direct_here_unicode_reference_has_no_metadata,
            add_type_cached_here_unicode_reference_has_no_metadata,
            "@\"\nclass $é {}\n\"@"
        ),
        (
            add_type_direct_here_escaped_backtick_has_no_metadata,
            add_type_cached_here_escaped_backtick_has_no_metadata,
            "@\"\nclass ``$name {}\n\"@"
        ),
    );

    #[derive(Debug, PartialEq)]
    struct ConstantMetadata {
        code: String,
        parameter: String,
        code_span: Span,
        command_span: Span,
        metadata_span: Span,
        import_count: usize,
        apis: Vec<String>,
    }

    fn constant_metadata(tree: &Node) -> Vec<Option<ConstantMetadata>> {
        let mut metadata = Vec::new();
        tree.walk(&mut |n| {
            if let NodeKind::Command { name, csharp, .. } = &n.kind {
                if matches!(&name.kind, NodeKind::BareWord(name) if name.eq_ignore_ascii_case("add-type"))
                {
                    metadata.push(csharp.as_deref().map(|node| {
                        let NodeKind::CSharpMemberDef(cs) = &node.kind else {
                            panic!("expected CSharpMemberDef, got {node:?}");
                        };
                        ConstantMetadata {
                            code: cs.code.clone(),
                            parameter: cs.parameter.clone(),
                            code_span: cs.code_span,
                            command_span: n.span,
                            metadata_span: node.span,
                            import_count: cs.imports.len(),
                            apis: cs.apis.clone(),
                        }
                    }));
                }
            }
        });
        metadata
    }

    macro_rules! constant_add_type_cases {
        ($(($name:ident, $src:literal, $code:literal, $body:expr, $cmd:expr)),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    let src = $src;

                    let out = parse(src);

                    assert_eq!(out.errors, Vec::<ParseError>::new());
                    assert_eq!(constant_metadata(&out.script), vec![Some(ConstantMetadata {
                        code: $code.into(),
                        parameter: "typedefinition".into(),
                        code_span: $body,
                        command_span: $cmd,
                        metadata_span: $cmd,
                        import_count: 0,
                        apis: vec![],
                    })]);
                }
            )+
        };
    }

    constant_add_type_cases!(
        (
            add_type_direct_single_quoted_keeps_dollars,
            "Add-Type -TypeDefinition '$name ${name} $(Get-Name) `'",
            "$name ${name} $(Get-Name) `",
            Span::new(26, 53),
            Span::new(0, 54)
        ),
        (
            add_type_cached_single_quoted_keeps_dollars,
            "$Code = '$name ${name} $(Get-Name) `'\nAdd-Type -TypeDefinition $code",
            "$name ${name} $(Get-Name) `",
            Span::new(9, 36),
            Span::new(38, 68)
        ),
        (
            add_type_direct_single_here_keeps_dollars,
            "Add-Type -TypeDefinition @'\n$name ${name} $(Get-Name) `\n'@",
            "$name ${name} $(Get-Name) `",
            Span::new(28, 55),
            Span::new(0, 58)
        ),
        (
            add_type_cached_single_here_keeps_dollars,
            "$Code = @'\n$name ${name} $(Get-Name) `\n'@\nAdd-Type -TypeDefinition $code",
            "$name ${name} $(Get-Name) `",
            Span::new(11, 38),
            Span::new(42, 72)
        ),
        (
            add_type_direct_double_constant_preserves_metadata,
            "Add-Type -TypeDefinition \"class A {}\"",
            "class A {}",
            Span::new(26, 36),
            Span::new(0, 37)
        ),
        (
            add_type_cached_double_constant_preserves_metadata,
            "$Code = \"class A {}\"\nAdd-Type -TypeDefinition $code",
            "class A {}",
            Span::new(9, 19),
            Span::new(21, 51)
        ),
        (
            add_type_direct_double_here_constant_preserves_metadata,
            "Add-Type -TypeDefinition @\"\nclass A {}\n\"@",
            "class A {}",
            Span::new(28, 38),
            Span::new(0, 41)
        ),
        (
            add_type_cached_double_here_constant_preserves_metadata,
            "$Code = @\"\nclass A {}\n\"@\nAdd-Type -TypeDefinition $code",
            "class A {}",
            Span::new(11, 21),
            Span::new(25, 55)
        ),
        (
            add_type_direct_escaped_dollar_is_constant,
            "Add-Type -TypeDefinition \"class `$name {}\"",
            "class `$name {}",
            Span::new(26, 41),
            Span::new(0, 42)
        ),
        (
            add_type_cached_escaped_dollar_is_constant,
            "$Code = \"class `$name {}\"\nAdd-Type -TypeDefinition $code",
            "class `$name {}",
            Span::new(9, 24),
            Span::new(26, 56)
        ),
        (
            add_type_direct_odd_backticks_are_constant,
            "Add-Type -TypeDefinition \"class ```$name {}\"",
            "class ```$name {}",
            Span::new(26, 43),
            Span::new(0, 44)
        ),
        (
            add_type_cached_odd_backticks_are_constant,
            "$Code = \"class ```$name {}\"\nAdd-Type -TypeDefinition $code",
            "class ```$name {}",
            Span::new(9, 26),
            Span::new(28, 58)
        ),
        (
            add_type_direct_here_escaped_dollar_is_constant,
            "Add-Type -TypeDefinition @\"\nclass `$name {}\n\"@",
            "class `$name {}",
            Span::new(28, 43),
            Span::new(0, 46)
        ),
        (
            add_type_cached_here_escaped_dollar_is_constant,
            "$Code = @\"\nclass `$name {}\n\"@\nAdd-Type -TypeDefinition $code",
            "class `$name {}",
            Span::new(11, 26),
            Span::new(30, 60)
        ),
        (
            add_type_fresh_literal_reassignment_uses_new_metadata,
            "$Code = 'class A {}'\n$CODE = 'class B {}'\nAdd-Type -TypeDefinition $code",
            "class B {}",
            Span::new(30, 40),
            Span::new(42, 72)
        ),
        (
            add_type_unicode_bom_uses_original_source_spans,
            "\u{feff}# é\n$Code = 'class A {}'\nAdd-Type -TypeDefinition $code",
            "class A {}",
            Span::new(17, 27),
            Span::new(29, 59)
        ),
    );

    #[test]
    fn interpolation_parts_have_precise_spans() {
        // The variable inside a double-quoted string must point at its own
        // bytes, not at the whole string, so refactors can rewrite it.
        let src = "Write-Output \"hello $name and ${other}\"\n";
        let out = parse(src);
        let mut hits: Vec<String> = Vec::new();
        out.script.walk(&mut |n| {
            if let NodeKind::Variable(_) = &n.kind {
                hits.push(n.span.slice(src).to_string());
            }
        });
        assert_eq!(hits, vec!["$name".to_string(), "${other}".to_string()]);

        // The same must hold inside a here-string, where the body offset is
        // past `@"` and the leading newline.
        let src2 = "$x = @\"\nval=$count\n\"@\n";
        let out2 = parse(src2);
        let mut found = None;
        out2.script.walk(&mut |n| {
            if let NodeKind::Variable(raw) = &n.kind {
                if raw == "$count" {
                    found = Some(n.span.slice(src2).to_string());
                }
            }
        });
        assert_eq!(found.as_deref(), Some("$count"));
    }

    #[test]
    fn subexpression_interpolation_variable_has_correct_span() {
        // Regression for the $(...) span bug: a variable inside a sub-expression
        // interpolation must point at its own bytes. Before the fix this sliced
        // to "Write" (the sub_src-relative span [0,5] read against the source).
        let src = "Write-Output \"x $($file.Name) y\"\n";
        let out = parse(src);
        let mut slices = Vec::new();
        out.script.walk(&mut |n| {
            if let NodeKind::Variable(name) = &n.kind {
                if name == "$file" {
                    slices.push(n.span.slice(src).to_string());
                }
            }
        });
        assert_eq!(slices, vec!["$file".to_string()]);
    }

    #[test]
    fn subexpression_interpolation_span_is_not_a_coincidence() {
        // The dangerous case: the buggy span happened to slice to a real `$q`
        // elsewhere, so a slice-only check passes. Assert on the byte offset.
        let src = "$q = 1\n\"hi $($q.X)\"\n";
        let out = parse(src);
        let mut starts = Vec::new();
        out.script.walk(&mut |n| {
            if let NodeKind::Variable(name) = &n.kind {
                if name == "$q" {
                    starts.push(n.span.start);
                }
            }
        });
        // The assignment's $q is at byte 0; the interpolated $q is at byte 13.
        assert!(
            starts.contains(&0),
            "assignment $q at byte 0, got {starts:?}"
        );
        assert!(
            starts.contains(&13),
            "interpolated $q at byte 13, got {starts:?}"
        );
        assert_eq!(&src[13..15], "$q");
    }

    #[test]
    fn two_subexpressions_get_distinct_spans() {
        let src = "Write-Output \"z $($a.B) $($c) w\"\n";
        let out = parse(src);
        let mut found = std::collections::HashMap::new();
        out.script.walk(&mut |n| {
            if let NodeKind::Variable(name) = &n.kind {
                found.insert(name.clone(), n.span.slice(src).to_string());
            }
        });
        assert_eq!(found.get("$a").map(String::as_str), Some("$a"));
        assert_eq!(found.get("$c").map(String::as_str), Some("$c"));
    }

    #[test]
    fn subexpression_interpolation_after_multibyte_char() {
        // `é` is two bytes; the rebased span must stay byte-accurate (slicing on
        // a non-char-boundary would panic).
        let src = "Write-Output \"é $($x) y\"\n";
        let out = parse(src);
        let mut slice = None;
        out.script.walk(&mut |n| {
            if let NodeKind::Variable(name) = &n.kind {
                if name == "$x" {
                    slice = Some(n.span.slice(src).to_string());
                }
            }
        });
        assert_eq!(slice.as_deref(), Some("$x"));
    }

    #[test]
    fn nested_subexpression_interpolation() {
        // A sub-expression inside a sub-expression: the innermost variable is
        // rebased once per level and ends up at an absolute offset.
        let src = "Write-Output \"v $( $($y) ) w\"\n";
        let out = parse(src);
        let mut slice = None;
        out.script.walk(&mut |n| {
            if let NodeKind::Variable(name) = &n.kind {
                if name == "$y" {
                    slice = Some(n.span.slice(src).to_string());
                }
            }
        });
        assert_eq!(slice.as_deref(), Some("$y"));
    }

    #[test]
    fn subexpression_interpolation_is_lossless() {
        // The fix moves spans only; tokens and the round-trip are untouched.
        let src = "Write-Output \"x $($file.Name) y\"\n";
        let out = parse(src);
        assert_eq!(crate::v2::reconstruct(&out.tokens), src);
    }

    #[test]
    fn parse_tokens_matches_parse_and_hands_tokens_back() {
        let src = "function Get-Thing { param([int]$n) $n + 1 }\nGet-Thing -n 2 | Write-Output\n";

        // parse() lexes internally.
        let from_src = parse(src);
        // parse_tokens() parses tokens lexed once by the caller.
        let lexed = crate::v2::lex(src);
        let from_tokens = parse_tokens(src, lexed.tokens);

        // Same tree and same errors regardless of entry point.
        assert_eq!(from_src.errors, from_tokens.errors);
        assert_eq!(
            format!("{:?}", from_src.script),
            format!("{:?}", from_tokens.script)
        );

        // Both hand the tokens back, and they reconstruct the source exactly,
        // so a caller that also needs tokens never has to lex twice.
        assert_eq!(crate::v2::reconstruct(&from_src.tokens), src);
        assert_eq!(crate::v2::reconstruct(&from_tokens.tokens), src);
    }

    #[test]
    fn fuzz_parser_never_panics() {
        fn next(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        let charset: Vec<char> = "$@{}()[]| ;,.\"'`#-=+*/<>?:&\nabcXYZ012_".chars().collect();
        let mut state: u64 = 0xABCDEF;
        for _ in 0..2000 {
            let len = (next(&mut state) % 60) as usize;
            let src: String = (0..len)
                .map(|_| charset[(next(&mut state) as usize) % charset.len()])
                .collect();
            let _ = parse(&src); // must not panic
        }
    }

    /// The first `Function` node in source order, with its captured name span.
    fn first_function(src: &str) -> (String, Span, bool) {
        let out = parse(src);
        let mut found = None;
        out.script.walk(&mut |n| {
            if found.is_none() {
                if let NodeKind::Function {
                    name,
                    name_span,
                    filter,
                    ..
                } = &n.kind
                {
                    found = Some((name.clone(), *name_span, *filter));
                }
            }
        });
        found.expect("a function node")
    }

    #[test]
    fn function_name_span_covers_the_name_only() {
        let src = "function Get-Thing { param([int]$n) $n + 1 }\n";
        let (name, span, filter) = first_function(src);
        assert_eq!(name, "Get-Thing");
        assert!(!filter);
        // The span addresses just the name, not the whole `function ... { }`.
        assert_eq!(span.slice(src), "Get-Thing");
    }

    #[test]
    fn filter_name_span_covers_the_name_only() {
        let src = "filter Skip-Blank { $_ }\n";
        let (name, span, filter) = first_function(src);
        assert_eq!(name, "Skip-Blank");
        assert!(filter);
        assert_eq!(span.slice(src), "Skip-Blank");
    }

    #[test]
    fn missing_function_name_yields_empty_span() {
        let src = "function { 1 }\n";
        let out = parse(src);
        assert!(!out.errors.is_empty());
        let (name, span, _) = first_function(src);
        assert!(name.is_empty());
        // Empty, and positioned where the name would be (at the `{`).
        assert_eq!(span.start, span.end);
        assert_eq!(&src[span.start..], "{ 1 }\n");
    }

    #[test]
    fn function_name_span_drives_a_rename() {
        // Renaming a definition needs no token-stream lookup: edit name_span.
        let src = "function Get-Thing { 1 }\n";
        let (_, span, _) = first_function(src);
        let edit = crate::v2::TextEdit::replace(span, "Get-Other".to_string());
        let renamed = crate::v2::apply_edits(src, &[edit]).unwrap();
        assert_eq!(renamed, "function Get-Other { 1 }\n");
    }

    #[test]
    fn deeply_nested_statements_are_bounded() {
        // Statement recursion (nested `if { ... }`) is depth-guarded, so a
        // pathological nesting returns with a recorded error instead of
        // recursing until the stack is exhausted. Without the guard this input
        // overflows and aborts the process. Run on an explicit default-sized
        // (8 MiB) stack so the test is independent of the harness default.
        let handle = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(|| {
                let depth = 100_000;
                let src = format!("{}1{}", "if (1) {".repeat(depth), "}".repeat(depth));
                let out = parse(&src);
                out.errors
                    .iter()
                    .any(|e| format!("{e:?}").contains("deeply"))
            })
            .unwrap();
        let hit_guard = handle.join().expect("parser must not overflow the stack");
        assert!(hit_guard, "expected a depth-limit error");
    }

    #[test]
    fn moderate_statement_nesting_still_parses() {
        // The limit is well above realistic block nesting, so ordinary scripts
        // are unaffected.
        let depth = 50;
        let src = format!("{}1{}", "if (1) {".repeat(depth), "}".repeat(depth));
        let out = parse(&src);
        assert!(
            !out.errors
                .iter()
                .any(|e| format!("{e:?}").contains("deeply")),
            "depth {depth} should be under the limit"
        );
    }

    #[test]
    fn interleaved_nesting_recovers_linearly() {
        // Interleaving statement and expression nesting (`if (1) { (` ...) used
        // to blow up: once the depth guard tripped, a non-consuming error let a
        // parent re-drive the parse beneath it, so work grew exponentially in
        // the nesting depth and a short input exhausted memory. Recovery is now
        // linear, so a deeply interleaved input parses quickly. Assert the node
        // count stays proportional to the input rather than exploding.
        let depth = 5000;
        let src = format!("{}1{}", "if (1) {{ (".repeat(depth), ") }".repeat(depth));
        let out = parse(&src);
        let mut nodes = 0usize;
        out.script.walk(&mut |_| nodes += 1);
        // Linear recovery keeps this within a small multiple of the token count.
        // Exponential behavior produced millions of nodes (or OOM) by depth ~50.
        assert!(
            nodes < depth * 20,
            "node count {nodes} is not linear in depth {depth}"
        );
        assert!(out
            .errors
            .iter()
            .any(|e| format!("{e:?}").contains("deeply")));
    }

    #[test]
    fn deeply_nested_expressions_are_bounded() {
        // The same guard covers expression recursion via parse_primary.
        let handle = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(|| {
                let depth = 100_000;
                let src = format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
                let out = parse(&src);
                out.errors
                    .iter()
                    .any(|e| format!("{e:?}").contains("deeply"))
            })
            .unwrap();
        let hit_guard = handle.join().expect("parser must not overflow the stack");
        assert!(hit_guard, "expected a depth-limit error");
    }

    /// A parenthesized rendering of the first statement's expression tree, for
    /// asserting operator associativity and precedence.
    fn expr_shape(src: &str) -> String {
        fn show(n: &Node, src: &str) -> String {
            match &n.kind {
                NodeKind::Binary { op, left, right } => {
                    format!("({} {} {})", show(left, src), op, show(right, src))
                }
                NodeKind::Unary { op, operand } => format!("U({op} {})", show(operand, src)),
                NodeKind::Number(s) | NodeKind::BareWord(s) | NodeKind::Variable(s) => s.clone(),
                NodeKind::StringLiteral { .. } => n.span.slice(src).to_string(),
                NodeKind::Script(v) if v.len() == 1 => show(&v[0], src),
                NodeKind::Pipeline(v) if v.len() == 1 => show(&v[0], src),
                other => format!("{other:?}"),
            }
        }
        let out = parse(src);
        match &out.script.kind {
            NodeKind::Script(v) => show(v.first().expect("a statement"), src),
            _ => unreachable!(),
        }
    }

    #[test]
    fn comparison_is_tighter_than_bitwise_tier() {
        // PowerShell ranks the comparison tier above the bitwise tier, so a
        // comparison mixed with a bitwise or shift operator binds first.
        assert_eq!(expr_shape("1 -bor 2 -eq 3"), "(1 -bor (2 -eq 3))");
        // Bitwise and shift operators share one tier, left-associative.
        assert_eq!(expr_shape("5 -band 6 -shl 1"), "((5 -band 6) -shl 1)");
        assert_eq!(expr_shape("1 -bxor 2 -bor 3"), "((1 -bxor 2) -bor 3)");
    }

    #[test]
    fn logical_binds_looser_than_bitwise() {
        // -and / -or sit below the comparison and bitwise tier.
        assert_eq!(expr_shape("1 -bor 2 -and 3"), "((1 -bor 2) -and 3)");
    }

    #[test]
    fn arithmetic_binds_tighter_than_bitwise() {
        // -band is tier 5; + is tier 6 (tighter), so the addition groups first.
        assert_eq!(expr_shape("1 -band 2 + 3"), "(1 -band (2 + 3))");
    }

    #[test]
    fn containment_and_type_operators_are_recognized() {
        // -contains, -in, -is, -isnot begin with c/i, the case-sensitivity
        // prefix letters. They still parse as binary comparison operators.
        assert_eq!(expr_shape("1 -contains 2"), "(1 -contains 2)");
        assert_eq!(expr_shape("1 -notcontains 2"), "(1 -notcontains 2)");
        assert_eq!(expr_shape("1 -in 2"), "(1 -in 2)");
        assert_eq!(expr_shape("1 -notin 2"), "(1 -notin 2)");
        assert_eq!(expr_shape("1 -isnot 2"), "(1 -isnot 2)");
    }

    #[test]
    fn comparison_operators_are_case_insensitive() {
        // PowerShell accepts any case, plus the explicit c/i prefix forms.
        assert_eq!(expr_shape("1 -EQ 2"), "(1 -EQ 2)");
        assert_eq!(expr_shape("1 -cEq 2"), "(1 -cEq 2)");
        assert_eq!(expr_shape("1 -iLike 2"), "(1 -iLike 2)");
        assert_eq!(expr_shape("1 -CONTAINS 2"), "(1 -CONTAINS 2)");
        // Previously the lexer classified `-cin` as a parameter, so the parser
        // never saw it; the shared operator table fixed the spelling family.
        assert_eq!(expr_shape("1 -cin 2"), "(1 -cin 2)");
        assert_eq!(expr_shape("1 -csplit 2"), "(1 -csplit 2)");
    }

    #[test]
    fn glued_star_is_multiplication_in_expression_position() {
        // The lexer reads `*2` as a glob word; in expression position the
        // parser splits it into `*` and a right operand.
        let out = parse("$_*2\n");
        assert!(out.errors.is_empty());
        let mut found = None;
        out.script.walk(&mut |n| {
            if let NodeKind::Binary { op, right, .. } = &n.kind {
                found = Some((op.clone(), right.span));
            }
        });
        let (op, rhs) = found.expect("a Binary node");
        assert_eq!(op, "*");
        assert_eq!(rhs.slice("$_*2\n"), "2");

        // Precedence composes: tier 7, same as spaced multiplication.
        assert_eq!(expr_shape("1 + 2*3"), "(1 + (2 * 3))");
        assert_eq!(expr_shape("1..2*3"), "((1 .. 2) * 3)");
    }

    /// Binary-node count and error count for a source snippet.
    fn binaries_and_errors(src: &str) -> (usize, usize) {
        let out = parse(src);
        let mut binaries = 0;
        out.script.walk(&mut |n| {
            if matches!(n.kind, NodeKind::Binary { .. }) {
                binaries += 1;
            }
        });
        (binaries, out.errors.len())
    }

    #[test]
    fn glued_star_chains_fold_left_associatively() {
        // `$_*2*3` is (($_ * 2) * 3), as PowerShell groups it, not
        // ($_ * (2 * 3)).
        assert_eq!(expr_shape("2*3*4"), "((2 * 3) * 4)");
        let (binaries, errors) = binaries_and_errors("$_*2*3\n");
        assert_eq!((binaries, errors), (2, 0));
        // A sign-wrapped segment is still a numeric literal: `-` is a word
        // character, so `*-2` glues into the same token.
        let (binaries, errors) = binaries_and_errors("$_*-2\n");
        assert_eq!((binaries, errors), (1, 0));
        let (binaries, errors) = binaries_and_errors("2*-3*4\n");
        assert_eq!((binaries, errors), (2, 0));
    }

    #[test]
    fn glued_star_bails_on_non_numeric_segments() {
        // Bareword, trailing-star, and empty segments keep the wildcard
        // reading instead of inventing a multiplication.
        for src in ["$_*abc\n", "2*abc\n", "$_*2*\n", "$_**2\n"] {
            let (binaries, errors) = binaries_and_errors(src);
            assert_eq!(binaries, 0, "{src} must not become a Binary");
            assert_eq!(errors, 0, "{src}");
        }
    }

    #[test]
    fn glued_star_suppressed_in_argument_position() {
        // In argument position `$x*2` is a glob continuation. Spaced `*2` is
        // a separate glob argument. Neither becomes a multiplication.
        for src in ["dir $x*2\n", "dir $x *2\n", "cmd $a*3\n", "dir $a[0]*2\n"] {
            let (binaries, errors) = binaries_and_errors(src);
            assert_eq!(binaries, 0, "{src} must stay glob-shaped");
            assert_eq!(errors, 0, "{src}");
        }
    }

    #[test]
    fn glued_star_active_inside_grouping_within_arguments() {
        // Parens, script blocks, array and hashtable literals, index
        // brackets, and method-call arguments re-enter expression context
        // even when the construct itself is a command argument.
        for src in [
            "dir ($_*2)\n",
            "dir { $_*2 }\n",
            "dir @($_*2)\n",
            "dir @{a=$_*2}\n",
            "dir $a[$_*2]\n",
            "dir $x.M($_*2)\n",
        ] {
            let (binaries, errors) = binaries_and_errors(src);
            assert!(binaries >= 1, "{src} must multiply inside the grouping");
            assert_eq!(errors, 0, "{src}");
        }
    }

    #[test]
    fn glued_star_does_not_touch_argument_globs() {
        for src in ["dir *.log\n", "copy a* b\n", "*.log | x\n"] {
            let out = parse(src);
            assert!(out.errors.is_empty(), "{src}");
            let mut binary = false;
            out.script.walk(&mut |n| {
                if matches!(n.kind, NodeKind::Binary { .. }) {
                    binary = true;
                }
            });
            assert!(!binary, "{src} must stay a command with glob arguments");
        }
    }

    /// The bareword command names in source order, plus the error count.
    fn command_names(src: &str) -> (usize, Vec<String>) {
        let out = parse(src);
        let mut names = Vec::new();
        out.script.walk(&mut |n| {
            if let NodeKind::Command { name, .. } = &n.kind {
                if let NodeKind::BareWord(w) = &name.kind {
                    names.push(w.clone());
                }
            }
        });
        (out.errors.len(), names)
    }

    #[test]
    fn percent_and_question_are_commands_in_pipeline_position() {
        // `%` and `?` are the ForEach-Object / Where-Object aliases. After a
        // pipe (or at statement start) they name a command, not an operator.
        let (errors, names) = command_names("1..10 | % { $_ * 2 }\n");
        assert_eq!(errors, 0);
        assert_eq!(names, ["%"]);

        let (errors, names) = command_names("ls | ? { $_.Length } | % { $_ }\n");
        assert_eq!(errors, 0);
        assert_eq!(names, ["ls", "?", "%"]);

        let (errors, names) = command_names("% { 1 }\n");
        assert_eq!(errors, 0);
        assert_eq!(names, ["%"]);
    }

    #[test]
    fn function_paren_parameter_list_is_parsed() {
        // `function f(...)` parameters parse into the same Parameter nodes as a
        // `param(...)` block; a `param(...)` block stays distinct.
        fn fn_params(src: &str) -> usize {
            let out = parse(src);
            let mut n = 0;
            out.script.walk(&mut |x| {
                if let NodeKind::Function { parameters, .. } = &x.kind {
                    n = parameters.len();
                }
            });
            n
        }
        let out = parse("function f($a, [int]$b = 1) { }\n");
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        assert_eq!(fn_params("function f($a, [int]$b = 1) { }\n"), 2);
        assert_eq!(fn_params("filter g([string]$s) { $s }\n"), 1);
        assert_eq!(fn_params("function NoP { 1 }\n"), 0);
        // A param() block is not a function-paren list.
        assert_eq!(fn_params("function f { param([int]$n) $n }\n"), 0);
    }

    #[test]
    fn param_block_models_attributes_types_and_defaults() {
        // `param(...)` entries used to collapse to bare variables; each is now
        // a Parameter node carrying its `[...]` brackets and default.
        fn params(src: &str) -> Vec<(usize, bool, String)> {
            let out = parse(src);
            let mut v = Vec::new();
            out.script.walk(&mut |n| {
                if let NodeKind::Parameter {
                    attributes,
                    default,
                    name,
                    ..
                } = &n.kind
                {
                    v.push((attributes.len(), default.is_some(), name.clone()));
                }
            });
            v
        }
        assert_eq!(
            params("function f { param([int]$n) }\n"),
            vec![(1, false, "$n".into())]
        );
        assert_eq!(
            params("function f { param([Parameter(Mandatory)][int]$n) }\n"),
            vec![(2, false, "$n".into())]
        );
        // Two parameters, each with a default, split on the top-level comma.
        assert_eq!(
            params("param([string]$x = 'hi', [int]$y = 5)\n"),
            vec![(1, true, "$x".into()), (1, true, "$y".into())]
        );
        assert_eq!(params("param($plain)\n"), vec![(0, false, "$plain".into())]);
    }

    #[test]
    fn trailing_switch_parameter_does_not_swallow_next_statement() {
        // A command parameter binds an argument only on the same line. A switch
        // like `-PassThru` followed by a newline takes no argument, and the
        // newline ends the command, so the next line is a separate statement.
        fn count(src: &str) -> usize {
            match parse(src).script.kind {
                NodeKind::Script(v) => v.len(),
                _ => 0,
            }
        }
        assert_eq!(count("Get-Foo -PassThru\nGet-Bar\n"), 2);
        assert_eq!(count("Get-Foo -PassThru\n$x = 1\n"), 2);
        assert_eq!(count("Get-Foo -PassThru -Other\nGet-Bar\n"), 2);
        assert_eq!(count("Get-Foo -PassThru\n\nGet-Bar\n"), 2);
        // A parameter with a same-line argument still binds it (one command).
        assert_eq!(count("Get-Foo -Name x\nGet-Bar\n"), 2);
        assert_eq!(count("Get-Foo -Name x\n"), 1);

        // The switch with a trailing newline has no bound argument; the
        // same-line parameter does.
        fn last_param_has_arg(src: &str) -> bool {
            let out = parse(src);
            let mut has = false;
            out.script.walk(&mut |n| {
                if let NodeKind::CommandParameter { argument, .. } = &n.kind {
                    has = argument.is_some();
                }
            });
            has
        }
        assert!(!last_param_has_arg("Get-Foo -PassThru\nGet-Bar\n"));
        assert!(last_param_has_arg("Get-Foo -Name x\n"));

        for src in [
            "Get-Foo -PassThru\nGet-Bar\n",
            "Get-Foo -PassThru -Other\nGet-Bar\n",
            "Get-Foo -Name x\nGet-Bar\n",
        ] {
            assert_eq!(
                crate::v2::reconstruct(&lex(src).tokens),
                src,
                "round-trip {src:?}"
            );
        }
    }

    #[test]
    fn as_type_operator_is_recognized() {
        // `-as` is a binary type operator (about_Type_Operators); `$x -as [int]`
        // is a single conversion, not two statements.
        fn shape(src: &str) -> String {
            let out = parse(src);
            match out.script.kind {
                NodeKind::Script(ref v) if v.len() == 1 => match &v[0].kind {
                    NodeKind::Binary { op, right, .. } => {
                        format!("Binary({op}, {})", right.label())
                    }
                    other => format!("{:?}", std::mem::discriminant(other)),
                },
                NodeKind::Script(ref v) => format!("{} statements", v.len()),
                _ => "not-script".into(),
            }
        }
        assert_eq!(shape("$x -as [int]\n"), "Binary(-as, TypeExpression)");
        assert_eq!(shape("$x -is [int]\n"), "Binary(-is, TypeExpression)");
        // `-as` followed by a comparison binds left: `($x -as [int]) -eq 1`.
        let out = parse("$x -as [int] -eq 1\n");
        if let NodeKind::Script(v) = &out.script.kind {
            assert_eq!(v.len(), 1, "should be a single expression, not split");
        }
        for src in ["$x -as [int]\n", "$x -as [int] -eq 1\n"] {
            assert_eq!(crate::v2::reconstruct(&lex(src).tokens), src);
        }
    }

    #[test]
    fn comparison_binds_tighter_than_bitwise() {
        // about_Operator_Precedence puts the comparison tier above the bitwise
        // tier (which includes the shifts), so `1 -band 2 -eq 3` is
        // `1 -band (2 -eq 3)`, not `(1 -band 2) -eq 3`.
        fn outer(src: &str) -> String {
            let out = parse(src);
            match out.script.kind {
                NodeKind::Script(ref v) => match v.first().map(|n| &n.kind) {
                    Some(NodeKind::Binary { op, right, .. }) => {
                        format!("Binary({op}, right={})", right.label())
                    }
                    _ => "other".into(),
                },
                _ => "not-script".into(),
            }
        }
        // Comparison is the inner (tighter) node; bitwise/shift is outer.
        assert_eq!(
            outer("1 -band 2 -eq 3\n"),
            "Binary(-band, right=BinaryExpression)"
        );
        assert_eq!(
            outer("1 -shl 2 -eq 3\n"),
            "Binary(-shl, right=BinaryExpression)"
        );
        // Bitwise group stays internally equal (left-associative).
        assert_eq!(
            outer("1 -band 2 -bor 3\n"),
            "Binary(-bor, right=NumberLiteral)"
        );
        // Bitwise still binds tighter than logical.
        assert_eq!(
            outer("1 -band 2 -and 3\n"),
            "Binary(-and, right=NumberLiteral)"
        );
        for src in [
            "1 -band 2 -eq 3\n",
            "1 -shl 2 -eq 3\n",
            "1 -band 2 -bor 3\n",
        ] {
            assert_eq!(crate::v2::reconstruct(&lex(src).tokens), src);
        }
    }

    #[test]
    fn comma_binds_tighter_than_binary_operators() {
        // PowerShell ranks the comma operator among the highest, above `-f`,
        // arithmetic, comparison, and logical operators. The documented example
        // is `1,2,1+2`, which PowerShell evaluates as `(1,2,1)+2` (printing
        // `1 2 1 2`), not `@(1,2,3)`. So the comma array is the *left* operand
        // of `+`, and `+` is the outermost node.
        fn outer(src: &str) -> String {
            let out = parse(src);
            match out.script.kind {
                NodeKind::Script(ref v) => match v.first().map(|n| &n.kind) {
                    Some(NodeKind::Binary { op, left, .. }) => {
                        format!("Binary({op}, left={})", left.label())
                    }
                    Some(other) => format!("{:?}", std::mem::discriminant(other)),
                    None => "empty".into(),
                },
                _ => "not-script".into(),
            }
        }
        // `+` is outermost; its left operand is the comma array.
        assert_eq!(outer("1,2,1+2\n"), "Binary(+, left=ArrayLiteral)");
        // `,1 + 2,3` is `(,1) + (2,3)`: `+` outermost, left is the one-element array.
        assert_eq!(outer(",1 + 2,3\n"), "Binary(+, left=ArrayLiteral)");
        // `1,2 -join 'x'` is `(1,2) -join 'x'`.
        assert_eq!(outer("1,2 -join 'x'\n"), "Binary(-join, left=ArrayLiteral)");

        // Inside an explicit group the comma builds the array as usual, and
        // multiple assignment keeps a comma list on each side.
        assert!(parse("$a, $b = 1, 2\n").errors.is_empty());

        for src in [
            "1,2,1+2\n",
            ",1 + 2,3\n",
            "1,2 -join 'x'\n",
            "$a, $b = 1, 2\n",
        ] {
            assert_eq!(
                crate::v2::reconstruct(&lex(src).tokens),
                src,
                "round-trip {src:?}"
            );
        }
    }

    #[test]
    fn logical_and_or_have_equal_precedence() {
        // PowerShell gives `-and`, `-or`, `-xor` equal precedence, folding left
        // to right, so `$true -or $false -and $false` is
        // `($true -or $false) -and $false` (FALSE), not
        // `$true -or ($false -and $false)` (TRUE).
        fn outer_op(src: &str) -> Option<String> {
            let out = parse(src);
            if let NodeKind::Script(v) = &out.script.kind {
                if let Some(NodeKind::Binary { op, .. }) = v.first().map(|n| &n.kind) {
                    return Some(op.clone());
                }
            }
            None
        }
        // Left-to-right means the *last* operator is the outermost node.
        assert_eq!(
            outer_op("$true -or $false -and $false\n").as_deref(),
            Some("-and")
        );
        assert_eq!(outer_op("1 -and 2 -or 3\n").as_deref(), Some("-or"));
        assert_eq!(outer_op("1 -xor 2 -and 3\n").as_deref(), Some("-and"));
        // And `x -and y -or z` is unchanged (already left-associative).
        assert_eq!(outer_op("1 -or 2 -and 3\n").as_deref(), Some("-and"));
        for src in ["$true -or $false -and $false\n", "1 -and 2 -or 3\n"] {
            assert_eq!(crate::v2::reconstruct(&lex(src).tokens), src);
        }
    }

    #[test]
    fn format_operator_binds_tighter_than_arithmetic() {
        // about_Operator_Precedence ranks `-f` above `* / %` and `+ -`, so
        // `"{0}" -f 1 + 2` is `("{0}" -f 1) + 2`, and range stays above `-f`.
        fn outer_op(src: &str) -> Option<String> {
            let out = parse(src);
            if let NodeKind::Script(v) = &out.script.kind {
                if let Some(NodeKind::Binary { op, .. }) = v.first().map(|n| &n.kind) {
                    return Some(op.clone());
                }
            }
            None
        }
        // `+` is the outermost node, so `-f` bound tighter.
        assert_eq!(outer_op("\"{0}\" -f 1 + 2\n").as_deref(), Some("+"));
        assert_eq!(outer_op("\"{0}\" -f 2 * 3\n").as_deref(), Some("*"));
        // Range binds tighter than `-f`, so `-f` is the outermost node here.
        assert_eq!(outer_op("\"{0}\" -f 1..3\n").as_deref(), Some("-f"));
        // The `-f` comma argument list still works.
        let out = parse("\"{0} {1}\" -f $a, $b\n");
        let mut right_is_array = false;
        out.script.walk(&mut |n| {
            if let NodeKind::Binary { op, right, .. } = &n.kind {
                if op == "-f" {
                    right_is_array = matches!(right.kind, NodeKind::ArrayLiteral(_));
                }
            }
        });
        assert!(
            right_is_array,
            "-f right operand should be the argument array"
        );
        for src in [
            "\"{0}\" -f 1 + 2\n",
            "\"{0}\" -f 1..3\n",
            "\"{0} {1}\" -f $a, $b\n",
        ] {
            assert_eq!(crate::v2::reconstruct(&lex(src).tokens), src);
        }
    }

    #[test]
    fn format_operator_binds_its_whole_argument_list() {
        // `-f` formats against the entire right-hand comma list, so its right
        // operand is an array of all arguments, and the top node is the binary
        // `-f`, not an array wrapping it.
        fn top_label(src: &str) -> &'static str {
            let out = parse(src);
            if let NodeKind::Script(v) = &out.script.kind {
                if let Some(first) = v.first() {
                    return first.label();
                }
            }
            "Script"
        }
        fn format_right_is_array(src: &str) -> Option<usize> {
            let out = parse(src);
            let mut len = None;
            out.script.walk(&mut |n| {
                if let NodeKind::Binary { op, right, .. } = &n.kind {
                    if op == "-f" {
                        if let NodeKind::ArrayLiteral(_) = &right.kind {
                            // Count leaf arguments by walking the right subtree.
                            let mut n = 0;
                            right.walk(&mut |x| {
                                if matches!(x.kind, NodeKind::Variable(_) | NodeKind::Number(_)) {
                                    n += 1;
                                }
                            });
                            len = Some(n);
                        }
                    }
                }
            });
            len
        }

        // Two arguments: top is the binary, right is an array of both.
        assert_eq!(top_label("\"{0} {1}\" -f $a, $b\n"), "BinaryExpression");
        assert_eq!(format_right_is_array("\"{0} {1}\" -f $a, $b\n"), Some(2));
        // Three arguments.
        assert_eq!(
            format_right_is_array("\"{0} {1} {2}\" -f 1, 2, 3\n"),
            Some(3)
        );
        // Single argument: the right operand is the value itself, not an array.
        assert_eq!(top_label("\"{0}\" -f $a\n"), "BinaryExpression");
        assert_eq!(format_right_is_array("\"{0}\" -f $a\n"), None);
        // Assignment of a format result is an assignment, not an array assign.
        assert_eq!(
            top_label("$x = \"{0} {1}\" -f $a, $b\n"),
            "AssignmentStatement"
        );

        // Comma binds tighter than the other operators now (PowerShell
        // precedence), so the comma array is an operand of the binary, and the
        // binary operator is the outermost node.
        assert_eq!(top_label("$a + $b, $c\n"), "BinaryExpression");
        assert_eq!(top_label("1 -eq 2, 3\n"), "BinaryExpression");

        // The reshape is token-preserving.
        for src in [
            "\"{0} {1}\" -f $a, $b\n",
            "\"{0}\" -f $a\n",
            "$x = \"{0} {1}\" -f $a, $b\n",
            "$a + $b, $c\n",
        ] {
            assert_eq!(
                crate::v2::reconstruct(&lex(src).tokens),
                src,
                "round-trip {src:?}"
            );
        }
    }

    #[test]
    fn switch_file_path_appears_once_in_the_tree() {
        // The -File path is the switch input; it must not also be stored as the
        // flag's argument, or a walk would visit it twice.
        let src = "switch -File 'data.txt' { 'x' { 1 } }\n";
        let out = parse(src);
        assert!(out.errors.is_empty());
        let mut path_hits = 0;
        let mut flag_count = 0;
        out.script.walk(&mut |n| match &n.kind {
            NodeKind::StringLiteral { .. } if n.span.slice(src) == "'data.txt'" => {
                path_hits += 1;
            }
            NodeKind::Switch { flags, input, .. } => {
                flag_count = flags.len();
                assert_eq!(input.span.slice(src), "'data.txt'", "input is the path");
            }
            _ => {}
        });
        assert_eq!(path_hits, 1, "the -File path must appear exactly once");
        assert_eq!(flag_count, 1, "the -File flag is still recorded");
    }

    #[test]
    fn switch_flags_are_modeled_and_do_not_break_the_input() {
        // `-Regex` and friends used to make the input parse fail; they are now
        // captured as flags, and `-File` takes the path as the input.
        let plain = parse("switch ($x) { 1 { 'a' } }\n");
        let mut n = 0;
        plain.script.walk(&mut |x| {
            if let NodeKind::Switch { flags, .. } = &x.kind {
                n = flags.len();
            }
        });
        assert!(plain.errors.is_empty());
        assert_eq!(n, 0);

        let regex = parse("switch -Regex ($x) { 'a.' { 1 } }\n");
        assert!(regex.errors.is_empty(), "{:?}", regex.errors);
        regex.script.walk(&mut |x| {
            if let NodeKind::Switch { flags, .. } = &x.kind {
                assert_eq!(flags.len(), 1);
            }
        });

        let two = parse("switch -Wildcard -CaseSensitive ($x) { 'a*' { 1 } }\n");
        assert!(two.errors.is_empty());
        two.script.walk(&mut |x| {
            if let NodeKind::Switch { flags, .. } = &x.kind {
                assert_eq!(flags.len(), 2);
            }
        });

        let file = parse("switch -File 'data.txt' { 'x' { 1 } }\n");
        assert!(file.errors.is_empty(), "{:?}", file.errors);
    }

    #[test]
    fn labels_only_wrap_loops_and_switch() {
        // A label before a loop or switch wraps it; a label before a bare
        // command does not (PowerShell allows labels only on loops/switch).
        fn is_labeled(src: &str) -> bool {
            let mut found = false;
            parse(src).script.walk(&mut |n| {
                if matches!(n.kind, NodeKind::Labeled { .. }) {
                    found = true;
                }
            });
            found
        }
        assert!(is_labeled(":outer foreach ($i in 1..3) { break outer }\n"));
        assert!(is_labeled(":sw switch ($x) { 1 { } }\n"));
        assert!(
            !is_labeled(":lbl Get-Process\n"),
            "command must not be labeled"
        );
        assert!(!is_labeled(":foo bar\n"), "bareword must not be labeled");
        // The non-label cases still parse without error.
        assert!(parse(":lbl Get-Process\n").errors.is_empty());
    }

    #[test]
    fn labeled_loops_parse() {
        // A label before a loop or switch wraps it in a Labeled node; the
        // ternary colon is not mistaken for a label.
        for (src, kw) in [
            (":outer foreach ($i in 1..3) { break outer }\n", "foreach"),
            (":x while ($y) { }\n", "while"),
            (":lbl for ($i = 0; $i -lt 3; $i++) { }\n", "for"),
            (":sw switch ($x) { 1 { } }\n", "switch"),
        ] {
            let out = parse(src);
            assert!(out.errors.is_empty(), "{kw}: {:?}", out.errors);
            let mut label = None;
            out.script.walk(&mut |n| {
                if let NodeKind::Labeled { label: l, .. } = &n.kind {
                    label = Some(l.clone());
                }
            });
            assert!(label.is_some(), "{kw} should be labeled");
        }
        let ternary = parse("$x = $a ? 1 : 2\n");
        let mut labeled = false;
        ternary.script.walk(&mut |n| {
            if matches!(n.kind, NodeKind::Labeled { .. }) {
                labeled = true;
            }
        });
        assert!(!labeled, "ternary colon is not a label");
    }

    #[test]
    fn unary_split_and_join_parse_at_expression_start() {
        // `-split`/`-join` have unary forms; at a value position they build a
        // Unary node, while the binary forms (with a left operand) are
        // unaffected.
        assert_eq!(expr_shape("-split 'a b'"), "U(-split 'a b')");
        assert_eq!(expr_shape("-join $a"), "U(-join $a)");
        assert_eq!(expr_shape("-csplit 'a b'"), "U(-csplit 'a b')");
        // Binary forms keep their shape.
        assert_eq!(expr_shape("'a b' -split ' '"), "('a b' -split ' ')");
        assert_eq!(expr_shape("$a -join ','"), "($a -join ',')");
    }

    #[test]
    fn modulo_and_ternary_still_parse_as_operators() {
        // The alias handling is positional; expression uses are untouched.
        assert_eq!(expr_shape("5 % 2"), "(5 % 2)");
        let (errors, names) = command_names("$x = $true ? 1 : 2\n");
        assert_eq!(errors, 0);
        assert!(names.is_empty(), "ternary must not become a command");
    }
}
