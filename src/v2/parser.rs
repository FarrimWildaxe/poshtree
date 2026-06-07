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
//! ```
//! use poshtree::v2::parse;
//! let out = poshtree::v2::parser::parse("Get-ChildItem -Path . | Sort-Object\n");
//! assert!(out.errors.is_empty());
//! ```

use super::lexer::lex;
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

/// The result of [`parse`]: the tree root (always a [`NodeKind::Script`]) and
/// any errors encountered (lexer errors first, then parser errors).
#[derive(Debug, Clone)]
pub struct ParseOutput {
    /// The script-block root.
    pub script: Node,
    /// Recoverable errors; an empty vector means a clean parse.
    pub errors: Vec<ParseError>,
}

/// Parses PowerShell source into a v2 syntax tree.
pub fn parse(src: &str) -> ParseOutput {
    let lexed = lex(src);
    let mut errors: Vec<ParseError> = lexed.errors.iter().map(ParseError::from_lex).collect();
    let mut parser = Parser {
        tokens: lexed.tokens,
        src,
        pos: 0,
        errors: Vec::new(),
        depth: 0,
        vars: std::collections::HashMap::new(),
    };
    let body = parser.parse_block_body(&[]);
    let end = parser.pos;
    let script = parser.make(NodeKind::Script(body), 0, end);
    errors.append(&mut parser.errors);
    ParseOutput { script, errors }
}

impl ParseError {
    fn from_lex(e: &LexError) -> Self {
        ParseError {
            message: e.to_string(),
            span: lex_error_span(e),
        }
    }
}

struct Parser<'a> {
    tokens: Vec<Token>,
    src: &'a str,
    pos: usize,
    errors: Vec<ParseError>,
    depth: u32,
    /// `$plainvar = "literal"` assignments, for recovering inline C# handed to
    /// `Add-Type` through a variable.
    vars: std::collections::HashMap<String, String>,
}

/// Guards against pathological nesting blowing the stack on adversarial input.
const MAX_DEPTH: u32 = 400;

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

    fn starts_line(&self) -> bool {
        self.tokens[self.pos].starts_line()
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
            T::Operator => matches!(
                self.lower().as_str(),
                "-" | "+" | "!" | "-not" | "-bnot" | "++" | "--"
            ),
            _ => false,
        }
    }

    fn is_command_start(&self) -> bool {
        matches!(self.kind(), T::Generic | T::Amp | T::Dot | T::Keyword)
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
        self.depth > MAX_DEPTH
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
                // Constant propagation: remember `$plainvar = "literal"` so a
                // later `Add-Type $plainvar` can recover the inline C#.
                if op == "=" {
                    if let (
                        NodeKind::Variable(raw),
                        NodeKind::StringLiteral {
                            value: sv, kind, ..
                        },
                    ) = (&expr.kind, &value.kind)
                    {
                        let n = raw.trim_start_matches('$');
                        if !n.contains(':') && !n.starts_with('{') {
                            self.vars.insert(
                                n.to_ascii_lowercase(),
                                string_inner(sv, *kind).to_string(),
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
        let (code, parameter) = find_csharp_code(elements, &self.vars)?;
        let mut def = parse_csharp_member_def(&code);
        def.parameter = parameter;
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
        // Bind a following argument as v1 does (`-Path value`); never bind to
        // another parameter, and only when an argument actually starts.
        let argument = if self.starts_argument() && !self.at(T::Parameter) {
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
            self.parse_expression()
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
        // Options like -Regex, -Wildcard, -File <path> tune how cases match;
        // they add nothing to the tree shape, and v1 drops them too, so they
        // are consumed here rather than modeled.
        while self.at(T::Parameter) {
            self.bump();
            if self.is_value_start() {
                let _ = self.parse_command_argument();
            }
        }
        let input = if self.at(T::LParen) {
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
        let name = if self.at(T::Generic) || self.at(T::Keyword) {
            let v = self.value().to_string();
            self.bump();
            v
        } else {
            self.error("expected function name");
            String::new()
        };
        let body = self.parse_braced_block();
        self.make(
            NodeKind::Function {
                name,
                filter,
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

    fn read_name_token(&mut self) -> String {
        if matches!(self.kind(), T::Generic | T::Keyword) {
            let v = self.value().to_string();
            self.bump();
            v
        } else {
            self.error("expected name");
            String::new()
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
        let name = self.read_name_token();
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
        let name = self.read_name_token();
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
        // Like v1: scan to the matching ')', emitting a Variable for every
        // Variable token and ignoring types, defaults, and attributes.
        let mut depth = 1usize;
        let mut params = Vec::new();
        while !self.at_end() && depth > 0 {
            match self.kind() {
                T::LParen => {
                    depth += 1;
                    self.bump();
                }
                T::RParen => {
                    depth -= 1;
                    self.bump();
                    if depth == 0 {
                        break;
                    }
                }
                T::Variable => {
                    let raw = self.value().to_string();
                    params.push(self.leaf(NodeKind::Variable(raw)));
                }
                _ => {
                    self.bump();
                }
            }
        }
        self.make(NodeKind::ParamBlock(params), start, self.pos)
    }

    // Expressions

    fn parse_expression(&mut self) -> Node {
        let start = self.pos;
        let first = self.parse_ternary();
        if self.at(T::Comma) {
            self.bump();
            // Right-associative: `a, b, c` nests as ArrayLiteral[a, [b, c]],
            // matching v1's comma operator.
            let rest = self.parse_expression();
            return self.make(NodeKind::ArrayLiteral(vec![first, rest]), start, self.pos);
        }
        first
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
        let mut left = self.parse_unary();
        while let Some(prec) = self.binary_prec() {
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
        }
        left
    }

    fn binary_prec(&self) -> Option<u8> {
        if !self.at(T::Operator) {
            return None;
        }
        let v = self.lower();
        Some(match v.as_str() {
            "??" => 1,
            "-or" | "-xor" => 2,
            "-and" => 3,
            "-band" | "-bor" | "-bxor" => 3,
            ".." => 8,
            "+" | "-" => 6,
            "*" | "/" | "%" => 7,
            _ if is_comparison_op(&v) => 5,
            _ => return None,
        })
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
            let v = self.lower();
            if matches!(v.as_str(), "-" | "+" | "!" | "-not" | "-bnot" | "++" | "--") {
                let op = self.value().to_string();
                self.bump();
                let operand = self.parse_unary();
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
            let null_dot = self.at(T::Operator) && self.value() == "?.";
            let null_index = self.at(T::Operator) && self.value() == "?[";
            if self.at(T::Dot) || self.at(T::DoubleColon) || null_dot {
                let is_static = self.at(T::DoubleColon);
                self.bump();
                let member = if matches!(self.kind(), T::Generic | T::Keyword | T::Variable) {
                    let v = self.value().to_string();
                    self.bump();
                    v
                } else {
                    self.error("expected member name");
                    String::new()
                };
                if self.at(T::LParen) {
                    let args = self.parse_paren_args();
                    node = self.make(
                        NodeKind::InvokeMember {
                            target: Box::new(node),
                            member,
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
                            is_static,
                        },
                        start,
                        self.pos,
                    );
                }
            } else if self.at(T::LBracket) || null_index {
                self.bump();
                let index = self.parse_expression();
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
            args.push(self.parse_expression());
        }
        self.expect(T::RParen, "')'");
        args
    }

    fn parse_primary(&mut self) -> Node {
        if self.enter() {
            self.leave();
            let i = self.pos;
            self.error("expression nested too deeply");
            return self.make(NodeKind::Error("too deep".into()), i, i);
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
                let inner = self.parse_pipeline_statement();
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
            T::AtBrace => self.parse_hashtable(),
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
                extract_interpolations(inner, span, i)
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

fn is_comparison_op(v: &str) -> bool {
    let core = v.strip_prefix('-').unwrap_or(v);
    // PowerShell spells case-sensitive and case-insensitive comparison as a
    // `c`/`i` prefix on the operator name (`-ceq`, `-ilike`); stripping the
    // prefix lets one list cover all three spellings of each operator.
    let core = core
        .strip_prefix('c')
        .or_else(|| core.strip_prefix('i'))
        .unwrap_or(core);
    matches!(
        core,
        "eq" | "ne"
            | "gt"
            | "ge"
            | "lt"
            | "le"
            | "like"
            | "notlike"
            | "match"
            | "notmatch"
            | "contains"
            | "notcontains"
            | "in"
            | "notin"
            | "is"
            | "isnot"
            | "replace"
            | "split"
            | "join"
            | "f"
            | "shl"
            | "shr"
    )
}

fn strip_sigil(var: &str) -> String {
    var.trim_start_matches(['$', '@']).to_string()
}

/// The body of a string, without its delimiters.
fn string_inner(value: &str, kind: StringKind) -> &str {
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

/// Matches a PowerShell variable reference at the start of `chars` and returns
/// how many characters it spans, mirroring v1's `VAR_REF`:
/// `$` followed by `{...}`, a `name` (optionally `scope:name`), or one of
/// `$_ $? $^ $$`.
fn match_var_ref(chars: &[char]) -> Option<usize> {
    if chars.first() != Some(&'$') || chars.len() < 2 {
        return None;
    }
    let n = chars.len();
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    match chars[1] {
        '{' => {
            let mut k = 2;
            while k < n && chars[k] != '}' {
                k += 1;
            }
            if k < n {
                Some(k + 1)
            } else {
                None
            }
        }
        c if c.is_ascii_alphabetic() || c == '_' => {
            let mut k = 2;
            while k < n && is_word(chars[k]) {
                k += 1;
            }
            if k + 1 < n
                && chars[k] == ':'
                && (chars[k + 1].is_ascii_alphabetic() || chars[k + 1] == '_')
            {
                k += 1; // ':'
                while k < n && is_word(chars[k]) {
                    k += 1;
                }
            }
            Some(k)
        }
        '?' | '^' | '$' => Some(2),
        _ => None,
    }
}

/// Extracts interpolation nodes (`$var`, `${name}`, `$(...)`) from the body of
/// an expandable string, mirroring v1. Parts share the string's span and an
/// empty token range, since their bytes are owned by the string token.
fn extract_interpolations(inner: &str, span: Span, idx: usize) -> Vec<Node> {
    let range = TokenRange {
        first: idx,
        end: idx,
    };
    let chars: Vec<char> = inner.chars().collect();
    let n = chars.len();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < n {
        if chars[i] == '`' {
            i += 2; // backtick escapes the next character
            continue;
        }
        if chars[i] == '$' && i + 1 < n && chars[i + 1] == '(' {
            let mut depth = 0usize;
            let mut j = i + 1;
            while j < n {
                match chars[j] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            let sub_src: String = chars[i + 2..j.min(n)].iter().collect();
            let script = parse(&sub_src).script;
            parts.push(Node {
                kind: NodeKind::SubExpression(Box::new(script)),
                span,
                range,
            });
            i = j + 1;
            continue;
        }
        if chars[i] == '$' {
            if let Some(consumed) = match_var_ref(&chars[i..]) {
                let raw: String = chars[i..i + consumed].iter().collect();
                parts.push(Node {
                    kind: NodeKind::Variable(raw),
                    span,
                    range,
                });
                i += consumed;
                continue;
            }
        }
        i += 1;
    }
    parts
}

fn strip_param_dash(param: &str) -> String {
    param
        .trim_start_matches('-')
        .trim_end_matches(':')
        .to_string()
}

fn lex_error_span(_e: &LexError) -> Span {
    Span::new(0, 0)
}

// Add-Type C# extraction (hand-rolled, dependency-free)

use std::collections::HashMap;

use super::ast::{CSharpImport, CSharpMemberDef, CSharpParam};

/// Locates the C# source handed to `Add-Type`: the argument of a
/// `-TypeDefinition`/`-MemberDefinition` parameter (prefix-matched), or the
/// first positional string or resolvable variable. Returns `(code, label)`.
fn find_csharp_code(elements: &[Node], vars: &HashMap<String, String>) -> Option<(String, String)> {
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
        if let Some(code) = value.and_then(|v| as_csharp_code(v, vars)) {
            return Some((code, pname));
        }
    }
    for el in elements {
        if matches!(el.kind, NodeKind::CommandParameter { .. }) {
            continue;
        }
        if let Some(code) = as_csharp_code(el, vars) {
            return Some((code, "positional".to_string()));
        }
    }
    None
}

/// A string literal's body, or an unscoped variable resolved to the string it
/// was assigned earlier in the script.
fn as_csharp_code(node: &Node, vars: &HashMap<String, String>) -> Option<String> {
    match &node.kind {
        NodeKind::StringLiteral { value, kind, .. } => Some(string_inner(value, *kind).to_string()),
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
    CSharpMemberDef {
        code: code.to_string(),
        imports,
        apis,
        parameter: String::new(),
    }
}

/// Tries to match `[DllImport("dll" ...)] <modifiers> <ret> <fn>(<params>)`
/// starting at byte `i`, returning the import and the byte offset just past the
/// parameter list. Mirrors v1's `DLLIMPORT_RE` plus `balanced_parens`.
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

fn skip_ws_b(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn matches_ci(b: &[u8], i: usize, kw: &[u8]) -> bool {
    i + kw.len() <= b.len() && b[i..i + kw.len()].eq_ignore_ascii_case(kw)
}

fn is_alpha_b(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

fn is_word_b(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn is_ret_char_b(c: u8) -> bool {
    is_word_b(c) || matches!(c, b'.' | b'<' | b'>' | b'[' | b']')
}

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
}
