//! A resilient recursive-descent parser for a broad subset of PowerShell.
//!
//! Key design points:
//! - **Two modes**: command mode (bareword starts) vs expression mode.
//! - **Error recovery**: failed statements become `ErrorNode`; the parser
//!   resynchronises to the next statement terminator, so one bad construct
//!   never aborts the whole parse.
//! - **Operator precedence** follows PowerShell's grammar via a layered descent:
//!   comparison, type, and bitwise/shift operators share one left-associative
//!   level; `??` and the ternary `? :` sit above it; and `&&` / `||` are treated
//!   as pipeline-chain operators (connecting pipelines) rather than expression
//!   operators.
//! - **Coverage**: expressions, pipelines and pipeline chains, assignments
//!   (incl. `??=`), control flow, functions/filters, the `using` / `class` /
//!   `enum` / `trap` / `data` / `dynamicparam` statement forms, full class
//!   members (properties, methods, constructors with attributes, modifiers,
//!   typed/defaulted parameters), and argument-mode lexing scanned from source
//!   (paths, wildcards, operator-led switches such as `/c` or `*.txt`,
//!   `-name:value`, and the `--%` stop-parsing token).
//! - **Null-conditional and redirections**: the `?.` / `?[` operators are
//!   parsed and flagged on the member-access / index nodes; redirections are
//!   captured as structured operator + optional target (a `2>&1`-style stream
//!   merge keeps its destination in the operator and has no target node), and a
//!   `[...]`-led argument glued to more text (such as `[abc]*.txt`) is read as a
//!   wildcard bareword rather than a type/cast.
//! - **Known limits**: member-attribute argument *values* are kept as trimmed
//!   source text (split into positional and named, but not parsed into
//!   expressions); the `[...]`-vs-bareword decision is a glued-character
//!   heuristic rather than full command/expression-mode resolution.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use super::ast::*;
use super::tokens::{Token, TokenType as T};

// Helpers

fn make_loc(tok: &Token) -> Location {
    Location {
        line: tok.line,
        col: tok.col,
        pos: tok.pos,
    }
}

fn box_node(n: AstNode) -> Box<AstNode> {
    Box::new(n)
}

/// Stamp a location onto every node in the subtree (for interpolation nodes).
fn stamp_pos(node: &mut AstNode, loc: &Location) {
    match node {
        AstNode::Variable(n) => n.loc = loc.clone(),
        AstNode::SubExpression(n) => n.loc = loc.clone(),
        AstNode::ScriptBlock(n) => n.loc = loc.clone(),
        _ => {}
    }
    for child in match node {
        AstNode::ScriptBlock(n) => n.statements.iter_mut().collect::<Vec<_>>(),
        _ => vec![],
    } {
        stamp_pos(child, loc);
    }
}

// C# member-definition parsing

// Rust's `regex` crate does not support backreferences, so we expand the
// quote pair into explicit double- and single-quote alternations.
static DLLIMPORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?xi)
        \[\s*DllImport\s*\(\s*              # [DllImport(
        (?:
            "(?P<dll_dq>[^"]+)"             # double-quoted dll name
          | '(?P<dll_sq>[^']+)'             # single-quoted dll name
        )
        [^\]]*\]                            # rest of attribute
        \s*(?:public|private|internal|protected|static|extern|unsafe|\s)+
        (?P<ret>[A-Za-z_][\w.<>\[\]]*)   \s+   # return type
        (?P<fn>[A-Za-z_]\w*)             \s*\(  # function name (
        "#,
    )
    .expect("DLLIMPORT_RE is a valid compile-time regex")
});

static VAR_REF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\$(?:\{[^}]*\}|[A-Za-z_][\w]*(?::[A-Za-z_][\w]*)?|[_?^$])")
        .expect("VAR_REF is a valid compile-time regex")
});

fn balanced_parens(code: &str, start: usize) -> (String, usize) {
    // `start` is a byte offset just past an opening '(' (it comes from a regex
    // match end). Working in byte space keeps this aligned with that offset;
    // '(' and ')' are ASCII, so every match lands on a char boundary even when
    // the surrounding C# contains multi-byte characters.
    let mut depth = 1usize;
    for (off, c) in code[start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return (code[start..start + off].to_owned(), start + off + 1);
                }
            }
            _ => {}
        }
    }
    (code[start..].to_owned(), code.len())
}

fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        // Separators and brackets inside a quoted span are literal text:
        // `ValidateSet('a,b', 'c')` has two arguments, not three.
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => quote = Some(c),
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth = depth.saturating_sub(1),
            _ if c == sep && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + sep.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// Parse a member attribute's inner bracket text (for example
/// `ValidateSet('A','B')` or `Parameter(Mandatory = $true)`) into structured
/// form. Argument values are kept as trimmed source text rather than parsed
/// into expressions.
fn parse_attribute(text: &str) -> Attribute {
    let text = text.trim();
    let (name, args, paren) = match text.find('(') {
        Some(i) if text.ends_with(')') => (text[..i].trim(), &text[i + 1..text.len() - 1], true),
        _ => (text, "", false),
    };
    let mut positional = Vec::new();
    let mut named = Vec::new();
    for arg in split_top_level(args, ',') {
        match split_named_arg(arg) {
            Some((k, v)) => named.push((k, v)),
            None => positional.push(arg.trim().to_owned()),
        }
    }
    Attribute {
        loc: Location::default(),
        name: name.to_owned(),
        paren,
        positional,
        named,
    }
}

/// Split an attribute argument of the form `key = value` on its top-level `=`.
/// Returns `None` (a positional argument) unless the left side is a bare
/// identifier, so `'A'` or `$true` stay positional.
fn split_named_arg(arg: &str) -> Option<(String, String)> {
    let mut depth = 0usize;
    for (i, c) in arg.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            '=' if depth == 0 => {
                let key = arg[..i].trim();
                if !key.is_empty() && key.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    return Some((key.to_owned(), arg[i + 1..].trim().to_owned()));
                }
                return None;
            }
            _ => {}
        }
    }
    None
}

const CS_PARAM_MODS: &[&str] = &["ref", "out", "in", "params", "this", "readonly", "scoped"];

static CS_ATTR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\s*[A-Za-z_][^\]]*\]").expect("CS_ATTR_RE is a valid regex"));

fn parse_cs_params(param_str: &str) -> Vec<CSharpParam> {
    split_top_level(param_str, ',')
        .iter()
        .filter_map(|raw| {
            let cleaned = CS_ATTR_RE.replace_all(raw, " ");
            let no_default = cleaned.split('=').next().unwrap_or("").trim();
            let words: Vec<&str> = no_default
                .split_whitespace()
                .filter(|w| !CS_PARAM_MODS.contains(w))
                .collect();
            if words.is_empty() {
                return None;
            }
            if words.len() == 1 {
                Some(CSharpParam {
                    type_name: words[0].to_owned(),
                    name: String::new(),
                })
            } else {
                Some(CSharpParam {
                    type_name: words[..words.len() - 1].join(" "),
                    name: words.last().unwrap().to_string(),
                })
            }
        })
        .collect()
}

pub fn parse_csharp_member_def(code: &str) -> CSharpMemberDef {
    let mut imports = Vec::new();
    let mut apis = Vec::new();
    for cap in DLLIMPORT_RE.captures_iter(code) {
        let end = cap.get(0).unwrap().end();
        let (params_str, _) = balanced_parens(code, end);
        // Resolve which quote style matched.
        let dll = cap
            .name("dll_dq")
            .or_else(|| cap.name("dll_sq"))
            .map(|m| m.as_str())
            .unwrap_or("")
            .to_owned();
        let imp = CSharpImport {
            dll,
            function: cap["fn"].to_owned(),
            returns: cap["ret"].to_owned(),
            params: parse_cs_params(&params_str),
        };
        apis.push(imp.function.clone());
        imports.push(imp);
    }
    CSharpMemberDef {
        code: code.to_owned(),
        imports,
        apis,
        ..Default::default()
    }
}

/// Extract interpolation nodes from an expandable string body.
fn extract_interpolations(text: &str, loc: &Location) -> Vec<AstNode> {
    let mut parts = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i] == '`' {
            i += 2;
            continue;
        }
        if chars[i] == '$' && i + 1 < n && chars[i + 1] == '(' {
            // $( ... )
            let mut depth = 0;
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
            let inner: String = chars[i + 2..j].iter().collect();
            // The parser is resilient (it never aborts on bad input), so we can
            // parse the sub-expression directly without a panic guard.
            let toks = super::lexer::tokenize(&inner);
            let sub = Parser::new(toks, inner.clone()).parse();
            let mut n = AstNode::SubExpression(SubExpression {
                loc: loc.clone(),
                body: sub,
            });
            stamp_pos(&mut n, loc);
            parts.push(n);
            i = j + 1;
            continue;
        }
        if chars[i] == '$' {
            let rest: String = chars[i..].iter().collect();
            if let Some(m) = VAR_REF.find(&rest) {
                let raw = m.as_str().to_owned();
                let inner = &raw[1..];
                let (scope, name) = if inner.starts_with('{') && inner.ends_with('}') {
                    (None, inner[1..inner.len() - 1].to_owned())
                } else if let Some(colon) = inner.find(':') {
                    (
                        Some(inner[..colon].to_owned()),
                        inner[colon + 1..].to_owned(),
                    )
                } else {
                    (None, inner.to_owned())
                };
                parts.push(AstNode::Variable(Variable {
                    loc: loc.clone(),
                    name,
                    scope,
                    raw,
                    ..Default::default()
                }));
                // `m.end()` is a byte offset into `rest`; `i` is a char index,
                // so advance by the match's char count to stay aligned on
                // non-ASCII variable names.
                i += m.as_str().chars().count();
                continue;
            }
        }
        i += 1;
    }
    parts
}

// Parser

const FLOW_KEYWORDS: &[&str] = &["break", "continue", "exit"];

pub struct Parser {
    toks: Vec<Token>,
    src: String,
    i: usize,
    pub errors: Vec<String>,
    /// `$plainvar` → assigned string-literal value, for resolving inline C#
    /// passed to `Add-Type` by variable (simple constant propagation).
    vars: HashMap<String, String>,
}

struct ParseError {
    msg: String,
    line: u32,
    col: u32,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, source: String) -> Self {
        // drop comments, keep newlines
        let toks = tokens.into_iter().filter(|t| t.ty != T::Comment).collect();
        Parser {
            toks,
            src: source,
            i: 0,
            errors: Vec::new(),
            vars: HashMap::new(),
        }
    }

    fn cur(&self) -> &Token {
        &self.toks[self.i]
    }
    fn peek(&self, ahead: usize) -> &Token {
        let j = (self.i + ahead).min(self.toks.len() - 1);
        &self.toks[j]
    }
    fn at(&self, ty: T) -> bool {
        self.cur().ty == ty
    }
    fn at_any(&self, tys: &[T]) -> bool {
        tys.contains(&self.cur().ty)
    }
    fn eof(&self) -> bool {
        self.cur().ty == T::Eof
    }

    fn next(&mut self) -> &Token {
        let i = self.i;
        if self.toks[i].ty != T::Eof {
            self.i += 1;
        }
        &self.toks[i]
    }

    /// Advance the token cursor to the first token at or beyond the raw byte
    /// offset `end_byte`. Used to resynchronise after scanning a span directly
    /// from the source (argument-mode barewords, the `--%` stop-parsing token).
    fn resync_past(&mut self, end_byte: usize) {
        while !self.eof() && self.cur().pos < end_byte {
            self.next();
        }
    }

    fn eat(&mut self, ty: T) -> Option<&Token> {
        if self.cur().ty == ty {
            let t = self.next();
            Some(t)
        } else {
            None
        }
    }

    fn skip_newlines(&mut self) {
        while self.at_any(&[T::Newline, T::Semicolon]) {
            self.next();
        }
    }

    fn skip_inline_newlines(&mut self) {
        while self.at(T::Newline) {
            self.next();
        }
    }

    fn stmt_end(&self) -> bool {
        self.at_any(&[T::Newline, T::Semicolon, T::Eof, T::RBrace, T::RParen])
    }

    fn is_keyword(&self, name: &str) -> bool {
        self.cur().ty == T::Keyword && self.cur().text.as_deref().unwrap_or("") == name
    }

    // Entry

    pub fn parse(&mut self) -> ScriptBlock {
        self.parse_statement_list(true, &[])
    }

    fn parse_statement_list(&mut self, _top: bool, stop: &[T]) -> ScriptBlock {
        let loc = make_loc(self.cur());
        let mut sb = ScriptBlock {
            loc,
            ..Default::default()
        };
        self.skip_newlines();
        // optional param block
        if self.is_keyword("param") {
            let pb = self.parse_param_block();
            sb.param_block = Some(box_node(AstNode::ParamBlock(pb)));
            self.skip_newlines();
        }
        while !self.eof() && !self.at_any(stop) {
            let saved = self.i;
            match self.parse_statement() {
                Ok(Some(stmt)) => sb.statements.push(stmt),
                Ok(None) => {}
                Err(e) => {
                    let err = self.recover(e);
                    sb.statements.push(AstNode::ErrorNode(err));
                }
            }
            if self.i == saved {
                self.next();
            }
            self.skip_newlines();
        }
        sb
    }

    fn parse_statement(&mut self) -> Result<Option<AstNode>, ParseError> {
        self.skip_inline_newlines();
        let tok = self.cur();

        if tok.ty == T::Keyword {
            let kw = tok.text.clone().unwrap_or_default();
            return Ok(Some(match kw.as_str() {
                "function" | "filter" | "workflow" => self.parse_function()?,
                "if" => self.parse_if()?,
                "while" => self.parse_while()?,
                "do" => self.parse_do()?,
                "for" => self.parse_for()?,
                "foreach" => self.parse_foreach()?,
                "switch" => self.parse_switch()?,
                "try" => self.parse_try()?,
                "return" => {
                    let loc = make_loc(self.next());
                    let val = if self.stmt_end() {
                        None
                    } else {
                        Some(box_node(self.parse_pipeline()?))
                    };
                    AstNode::ReturnStatement(ReturnStatement { loc, value: val })
                }
                "throw" => {
                    let loc = make_loc(self.next());
                    let val = if self.stmt_end() {
                        None
                    } else {
                        Some(box_node(self.parse_pipeline()?))
                    };
                    AstNode::ThrowStatement(ThrowStatement { loc, value: val })
                }
                k if FLOW_KEYWORDS.contains(&k) => {
                    let loc = make_loc(self.next());
                    if !self.stmt_end() {
                        let _ = self.parse_unary();
                    }
                    AstNode::FlowStatement(FlowStatement {
                        loc,
                        keyword: k.to_owned(),
                    })
                }
                "begin" | "process" | "end" => {
                    self.next();
                    AstNode::ScriptBlockExpression(ScriptBlockExpression {
                        loc: make_loc(self.cur()),
                        body: self.parse_block_value()?,
                    })
                }
                "class" => self.parse_class()?,
                "enum" => self.parse_enum()?,
                "using" => self.parse_using()?,
                "trap" | "data" | "dynamicparam" => self.parse_trailing_block()?,
                _ => return self.parse_pipeline_or_assignment(),
            }));
        }

        self.parse_pipeline_or_assignment()
    }

    fn parse_pipeline_or_assignment(&mut self) -> Result<Option<AstNode>, ParseError> {
        let first = if self.looks_like_assignment() {
            self.parse_assignment()?
        } else {
            self.parse_pipeline()?
        };
        // Pipeline-chain operators connect whole pipelines (`a && b || c`),
        // left-associative; the right side is itself a pipeline.
        let mut node = first;
        while self.at(T::Operator) && matches!(self.cur().value.as_str(), "&&" | "||") {
            let op = self.next().value.clone();
            self.skip_inline_newlines();
            let right = self.parse_pipeline()?;
            let loc = node.loc().clone();
            node = AstNode::PipelineChain(PipelineChain {
                loc,
                left: box_node(node),
                operator: op,
                right: box_node(right),
            });
        }
        Ok(Some(node))
    }

    // Assignment

    fn looks_like_assignment(&self) -> bool {
        if !self.at_any(&[T::Variable, T::LBracket]) {
            return false;
        }
        let mut j = self.i;
        let mut depth = 0usize;
        while j < self.toks.len() {
            let t = &self.toks[j];
            match t.ty {
                T::LBracket | T::LParen | T::LBrace | T::AtParen | T::AtBrace | T::DollarParen => {
                    depth += 1
                }
                T::RBracket | T::RParen | T::RBrace => {
                    depth = depth.saturating_sub(1);
                }
                T::Operator
                    if depth == 0
                        && ["=", "+=", "-=", "*=", "/=", "%=", "??="]
                            .contains(&t.value.as_str()) =>
                {
                    return true
                }
                T::Newline | T::Semicolon | T::Pipe | T::Eof if depth == 0 => return false,
                _ => {}
            }
            j += 1;
        }
        false
    }

    fn parse_assignment(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.cur().clone();
        let target = self.parse_unary()?;
        let op = self.next().value.clone();
        self.skip_inline_newlines();
        let value = self
            .parse_statement()?
            .unwrap_or_else(|| AstNode::ErrorNode(ErrorNode::default()));
        // Constant propagation: remember `$plainvar = "literal"` so a later
        // `Add-Type $plainvar` can recover the inline C# source.
        if op == "=" {
            if let (AstNode::Variable(v), AstNode::StringLiteral(s)) = (&target, &value) {
                if v.scope.is_none() {
                    self.vars.insert(v.name.to_lowercase(), s.value.clone());
                }
            }
        }
        Ok(AstNode::AssignmentStatement(AssignmentStatement {
            loc: make_loc(&tok),
            target: Some(box_node(target)),
            operator: op,
            value: Some(box_node(value)),
        }))
    }

    // Control flow

    fn parse_param_block(&mut self) -> ParamBlock {
        let tok = self.next().clone(); // 'param'
        let mut pb = ParamBlock {
            loc: make_loc(&tok),
            ..Default::default()
        };
        self.eat(T::LParen);
        let mut depth = 1usize;
        while !self.eof() && depth > 0 {
            match self.cur().ty {
                T::LParen => {
                    depth += 1;
                    self.next();
                }
                T::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        self.next();
                        break;
                    } else {
                        self.next();
                    }
                }
                T::Variable => {
                    let v = self.cur().clone();
                    pb.parameters.push(AstNode::Variable(Variable {
                        loc: make_loc(&v),
                        name: v.text.clone().unwrap_or_default(),
                        scope: v.scope.clone(),
                        raw: v.value.clone(),
                        ..Default::default()
                    }));
                    self.next();
                }
                _ => {
                    self.next();
                }
            }
        }
        pb
    }

    fn parse_function(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let name = if self.at_any(&[T::Generic, T::Keyword]) {
            self.next().value.clone()
        } else {
            String::new()
        };
        if self.at(T::LParen) {
            self.skip_balanced(T::LParen, T::RParen);
        }
        let body = if self.at(T::LBrace) {
            Some(self.parse_block_value()?)
        } else {
            None
        };
        Ok(AstNode::FunctionDefinition(FunctionDefinition {
            loc: make_loc(&tok),
            name,
            kind: tok.text.clone().unwrap_or_else(|| "function".into()),
            body,
        }))
    }

    /// Read a dotted identifier such as `System.Collections.Hashtable`.
    fn read_dotted_name(&mut self) -> String {
        let mut s = String::new();
        while self.at_any(&[T::Generic, T::Keyword, T::Dot]) {
            s.push_str(&self.next().value);
        }
        s
    }

    fn parse_using(&mut self) -> Result<AstNode, ParseError> {
        let kw = self.next().clone(); // 'using'
        self.skip_inline_newlines();
        let kind = if self.at_any(&[T::Generic, T::Keyword]) {
            let k = self
                .cur()
                .text
                .clone()
                .unwrap_or_else(|| self.cur().value.clone())
                .to_lowercase();
            if ["namespace", "module", "assembly", "type", "command"].contains(&k.as_str()) {
                self.next();
                k
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        // Remainder of the line is the name (dotted identifier, string, or path).
        let mut name = String::new();
        while !self.stmt_end() && !self.eof() {
            name.push_str(&self.next().value);
        }
        Ok(AstNode::UsingStatement(UsingStatement {
            loc: make_loc(&kw),
            kind,
            name: name.trim().to_owned(),
        }))
    }

    fn parse_class(&mut self) -> Result<AstNode, ParseError> {
        let kw = self.next().clone(); // 'class'
        self.skip_inline_newlines();
        let name = if self.at_any(&[T::Generic, T::Keyword]) {
            self.next().value.clone()
        } else {
            String::new()
        };
        // optional base/interface list: ': Base, [IFace], ...'
        let mut bases = Vec::new();
        self.skip_inline_newlines();
        if self.at(T::Operator) && self.cur().value == ":" {
            self.next();
            loop {
                self.skip_inline_newlines();
                if self.at(T::LBracket) {
                    match self.try_parse_type() {
                        Some(t) => bases.push(t),
                        None => break,
                    }
                } else if self.at_any(&[T::Generic, T::Keyword]) {
                    bases.push(self.read_dotted_name());
                } else {
                    break;
                }
                self.skip_inline_newlines();
                if self.eat(T::Comma).is_none() {
                    break;
                }
            }
        }
        self.skip_newlines();
        self.eat(T::LBrace);
        let mut members = Vec::new();
        loop {
            self.skip_newlines();
            while self.eat(T::Semicolon).is_some() {
                self.skip_newlines();
            }
            if self.eof() || self.at(T::RBrace) {
                break;
            }
            let saved = self.i;
            match self.parse_class_member(&name) {
                Ok(Some(m)) => members.push(m),
                Ok(None) => {}
                Err(e) => {
                    let err = self.recover(e);
                    members.push(AstNode::ErrorNode(err));
                }
            }
            if self.i == saved {
                self.next(); // stall guard
            }
        }
        self.eat(T::RBrace);
        Ok(AstNode::ClassDefinition(ClassDefinition {
            loc: make_loc(&kw),
            name,
            bases,
            members,
        }))
    }

    /// Capture the raw inner text of a balanced `[ ... ]` group (without the
    /// outer brackets) and advance past it. Used for member attributes and type
    /// annotations.
    fn capture_bracketed(&mut self) -> Option<String> {
        if !self.at(T::LBracket) {
            return None;
        }
        let start_pos = self.cur().pos;
        let mut depth = 0usize;
        let mut end_pos = start_pos;
        while !self.eof() {
            let t = self.cur();
            end_pos = t.pos + t.value.len();
            match t.ty {
                T::LBracket => {
                    depth += 1;
                    self.next();
                }
                T::RBracket => {
                    depth = depth.saturating_sub(1);
                    self.next();
                    if depth == 0 {
                        break;
                    }
                }
                _ => {
                    self.next();
                }
            }
        }
        let inner = self
            .src
            .get(start_pos + 1..end_pos.saturating_sub(1))
            .unwrap_or("")
            .trim()
            .to_owned();
        Some(inner)
    }

    /// Parse leading `[attr]` / `[type]` brackets and `static` / `hidden`
    /// modifiers. The last bracket is treated as the member's type; earlier
    /// brackets are attributes.
    fn parse_member_prefix(&mut self) -> (Vec<Attribute>, Vec<String>, String) {
        let mut attributes: Vec<Attribute> = Vec::new();
        let mut modifiers = Vec::new();
        let mut type_name = String::new();
        loop {
            self.skip_inline_newlines();
            if self.at_any(&[T::Generic, T::Keyword]) {
                let w = self
                    .cur()
                    .text
                    .clone()
                    .unwrap_or_else(|| self.cur().value.clone())
                    .to_lowercase();
                if w == "static" || w == "hidden" {
                    modifiers.push(w);
                    self.next();
                    continue;
                }
            }
            if self.at(T::LBracket) {
                match self.capture_bracketed() {
                    Some(t) => {
                        // A type literal never contains '('; an attribute
                        // invocation such as `ValidateSet('a','b')` does, so
                        // treat the latter as an attribute rather than letting it
                        // overwrite the member's type. (A paren-less attribute
                        // followed by a real type is still handled by the
                        // last-wins rule below.)
                        if t.contains('(') {
                            attributes.push(parse_attribute(&t));
                        } else {
                            if !type_name.is_empty() {
                                attributes.push(parse_attribute(&std::mem::take(&mut type_name)));
                            }
                            type_name = t;
                        }
                        continue;
                    }
                    None => break,
                }
            }
            break;
        }
        (attributes, modifiers, type_name)
    }

    fn parse_class_member(&mut self, class_name: &str) -> Result<Option<AstNode>, ParseError> {
        self.skip_newlines();
        if self.eof() || self.at(T::RBrace) {
            return Ok(None);
        }
        let start = self.cur().clone();
        let (attributes, modifiers, type_name) = self.parse_member_prefix();
        self.skip_inline_newlines();

        // Property: `$name [= default]`.
        if self.at(T::Variable) {
            let vtok = self.next().clone();
            let name = vtok
                .text
                .clone()
                .unwrap_or_else(|| vtok.value.trim_start_matches('$').to_owned());
            self.skip_inline_newlines();
            let default = if self.at(T::Operator) && self.cur().value == "=" {
                self.next();
                self.skip_inline_newlines();
                Some(box_node(self.parse_expression()?))
            } else {
                None
            };
            return Ok(Some(AstNode::ClassMember(ClassMember {
                loc: make_loc(&start),
                member_kind: "property".to_owned(),
                name,
                type_name,
                attributes,
                modifiers,
                parameters: Vec::new(),
                default,
                body: None,
            })));
        }

        // Method or constructor: `Name ( params ) { body }`.
        if self.at_any(&[T::Generic, T::Keyword]) {
            let name = self.next().value.clone();
            let parameters = if self.at(T::LParen) {
                self.parse_method_params()?
            } else {
                Vec::new()
            };
            self.skip_newlines();
            let body = if self.at(T::LBrace) {
                Some(self.parse_block_value()?)
            } else {
                None
            };
            let member_kind = if type_name.is_empty() && name.eq_ignore_ascii_case(class_name) {
                "constructor"
            } else {
                "method"
            }
            .to_owned();
            return Ok(Some(AstNode::ClassMember(ClassMember {
                loc: make_loc(&start),
                member_kind,
                name,
                type_name,
                attributes,
                modifiers,
                parameters,
                default: None,
                body,
            })));
        }

        Err(ParseError {
            msg: format!(
                "unexpected {:?} {:?} in class body",
                self.cur().ty,
                self.cur().value
            ),
            line: self.cur().line,
            col: self.cur().col,
        })
    }

    /// Parse a method/constructor parameter list `( [type] $a = default, ... )`.
    fn parse_method_params(&mut self) -> Result<Vec<AstNode>, ParseError> {
        let mut params = Vec::new();
        self.eat(T::LParen);
        self.skip_newlines();
        while !self.eof() && !self.at(T::RParen) {
            let pstart = self.cur().clone();
            let (_attrs, _mods, ptype) = self.parse_member_prefix();
            self.skip_inline_newlines();
            if !self.at(T::Variable) {
                break; // cannot make progress on this parameter
            }
            let vtok = self.next().clone();
            let var = AstNode::Variable(Variable {
                loc: make_loc(&vtok),
                name: vtok.text.clone().unwrap_or_default(),
                scope: vtok.scope.clone(),
                splat: vtok.splat,
                raw: vtok.value.clone(),
            });
            let typed = if ptype.is_empty() {
                var
            } else {
                AstNode::CastExpression(CastExpression {
                    loc: make_loc(&pstart),
                    type_name: ptype,
                    expression: box_node(var),
                })
            };
            self.skip_inline_newlines();
            let param = if self.at(T::Operator) && self.cur().value == "=" {
                self.next();
                self.skip_inline_newlines();
                let val = self.parse_expression()?;
                AstNode::AssignmentStatement(AssignmentStatement {
                    loc: make_loc(&pstart),
                    target: Some(box_node(typed)),
                    operator: "=".to_owned(),
                    value: Some(box_node(val)),
                })
            } else {
                typed
            };
            params.push(param);
            self.skip_newlines();
            if self.eat(T::Comma).is_none() {
                break;
            }
            self.skip_newlines();
        }
        self.eat(T::RParen);
        Ok(params)
    }

    fn parse_enum(&mut self) -> Result<AstNode, ParseError> {
        let kw = self.next().clone(); // 'enum'
        self.skip_inline_newlines();
        let name = if self.at_any(&[T::Generic, T::Keyword]) {
            self.next().value.clone()
        } else {
            String::new()
        };
        let mut base = String::new();
        self.skip_inline_newlines();
        if self.at(T::Operator) && self.cur().value == ":" {
            self.next();
            self.skip_inline_newlines();
            base = if self.at(T::LBracket) {
                self.try_parse_type().unwrap_or_default()
            } else {
                self.read_dotted_name()
            };
        }
        self.skip_newlines();
        self.eat(T::LBrace);
        let mut members = Vec::new();
        loop {
            self.skip_newlines();
            while self.eat(T::Comma).is_some() {
                self.skip_newlines();
            }
            if self.eof() || self.at(T::RBrace) {
                break;
            }
            if !self.at_any(&[T::Generic, T::Keyword]) {
                self.next(); // skip the unexpected token, avoid stalling
                continue;
            }
            let mtok = self.next().clone();
            let member_name = AstNode::BareWord(BareWord {
                loc: make_loc(&mtok),
                value: mtok.value.clone(),
            });
            if self.at(T::Operator) && self.cur().value == "=" {
                self.next();
                self.skip_inline_newlines();
                let val = self.parse_expression()?;
                members.push(AstNode::AssignmentStatement(AssignmentStatement {
                    loc: make_loc(&mtok),
                    target: Some(box_node(member_name)),
                    operator: "=".to_owned(),
                    value: Some(box_node(val)),
                }));
            } else {
                members.push(member_name);
            }
        }
        self.eat(T::RBrace);
        Ok(AstNode::EnumDefinition(EnumDefinition {
            loc: make_loc(&kw),
            name,
            base,
            members,
        }))
    }

    /// Parse a keyword whose body is a trailing `{ block }` after an optional
    /// header (`trap [type] {…}`, `data [name] [-opts] {…}`, `dynamicparam
    /// {…}`). The header tokens are skipped; the block is captured so its
    /// contents are still parsed.
    fn parse_trailing_block(&mut self) -> Result<AstNode, ParseError> {
        let kw = self.next().clone();
        let mut guard = 0;
        while !self.at(T::LBrace) && !self.stmt_end() && !self.eof() && guard < 64 {
            self.next();
            guard += 1;
        }
        let body = if self.at(T::LBrace) {
            self.parse_block_value()?
        } else {
            ScriptBlock::default()
        };
        Ok(AstNode::ScriptBlockExpression(ScriptBlockExpression {
            loc: make_loc(&kw),
            body,
        }))
    }

    fn parse_if(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let mut node = IfStatement {
            loc: make_loc(&tok),
            ..Default::default()
        };
        let cond = self.parse_paren_condition()?;
        let body = self.parse_block_value()?;
        node.clauses.push((box_node(cond), body));
        self.skip_inline_newlines();
        while self.is_keyword("elseif") {
            self.next();
            let c = self.parse_paren_condition()?;
            let b = self.parse_block_value()?;
            node.clauses.push((box_node(c), b));
            self.skip_inline_newlines();
        }
        if self.is_keyword("else") {
            self.next();
            node.else_body = Some(self.parse_block_value()?);
        }
        Ok(AstNode::IfStatement(node))
    }

    fn parse_while(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let cond = self.parse_paren_condition()?;
        let body = self.parse_block_value()?;
        Ok(AstNode::WhileStatement(WhileStatement {
            loc: make_loc(&tok),
            condition: Some(box_node(cond)),
            body: Some(body),
            ..Default::default()
        }))
    }

    fn parse_do(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let body = self.parse_block_value()?;
        self.skip_inline_newlines();
        let until = self.is_keyword("until");
        if self.is_keyword("while") || self.is_keyword("until") {
            self.next();
        }
        let cond = self.parse_paren_condition()?;
        Ok(AstNode::WhileStatement(WhileStatement {
            loc: make_loc(&tok),
            condition: Some(box_node(cond)),
            body: Some(body),
            do_while: true,
            until,
        }))
    }

    fn parse_for(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let mut node = ForStatement {
            loc: make_loc(&tok),
            ..Default::default()
        };
        self.eat(T::LParen);
        if !self.at(T::Semicolon) {
            node.initializer = Some(box_node(self.parse_statement()?.unwrap_or_default()));
        }
        self.eat(T::Semicolon);
        if !self.at(T::Semicolon) {
            node.condition = Some(box_node(self.parse_pipeline()?));
        }
        self.eat(T::Semicolon);
        if !self.at(T::RParen) {
            node.iterator = Some(box_node(self.parse_statement()?.unwrap_or_default()));
        }
        self.eat(T::RParen);
        node.body = Some(self.parse_block_value()?);
        Ok(AstNode::ForStatement(node))
    }

    fn parse_foreach(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let mut node = ForEachStatement {
            loc: make_loc(&tok),
            ..Default::default()
        };
        self.eat(T::LParen);
        if self.at(T::Variable) {
            let v = self.next().clone();
            node.variable = Some(box_node(AstNode::Variable(Variable {
                loc: make_loc(&v),
                name: v.text.clone().unwrap_or_default(),
                scope: v.scope.clone(),
                raw: v.value.clone(),
                ..Default::default()
            })));
        }
        if self.is_keyword("in") {
            self.next();
        }
        if !self.at(T::RParen) {
            node.enumerable = Some(box_node(self.parse_pipeline()?));
        }
        self.eat(T::RParen);
        node.body = Some(self.parse_block_value()?);
        Ok(AstNode::ForEachStatement(node))
    }

    fn parse_switch(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let mut node = SwitchStatement {
            loc: make_loc(&tok),
            ..Default::default()
        };
        while self.at(T::Parameter) {
            self.next();
        } // skip flags
        if self.at(T::LParen) {
            node.condition = Some(box_node(self.parse_paren_condition()?));
        }
        if self.at(T::LBrace) {
            node.body = Some(self.parse_block_value()?);
        }
        Ok(AstNode::SwitchStatement(node))
    }

    fn parse_try(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone();
        let mut node = TryStatement {
            loc: make_loc(&tok),
            ..Default::default()
        };
        node.body = Some(self.parse_block_value()?);
        self.skip_inline_newlines();
        while self.is_keyword("catch") {
            self.next();
            while self.at(T::LBracket) {
                self.skip_balanced(T::LBracket, T::RBracket);
                self.eat(T::Comma);
            }
            node.catches.push(self.parse_block_value()?);
            self.skip_inline_newlines();
        }
        if self.is_keyword("finally") {
            self.next();
            node.finally_body = Some(self.parse_block_value()?);
        }
        Ok(AstNode::TryStatement(node))
    }

    fn parse_paren_condition(&mut self) -> Result<AstNode, ParseError> {
        self.eat(T::LParen);
        self.skip_inline_newlines();
        let cond = self.parse_pipeline()?;
        self.skip_inline_newlines();
        self.eat(T::RParen);
        Ok(cond)
    }

    fn parse_block_value(&mut self) -> Result<ScriptBlock, ParseError> {
        self.skip_inline_newlines();
        if !self.at(T::LBrace) {
            let stmt = self.parse_statement()?.unwrap_or_default();
            let loc = stmt.loc().clone();
            return Ok(ScriptBlock {
                loc,
                statements: vec![stmt],
                ..Default::default()
            });
        }
        self.next(); // {
        let sb = self.parse_statement_list(false, &[T::RBrace]);
        self.eat(T::RBrace);
        Ok(sb)
    }

    // Pipelines & commands

    fn parse_pipeline(&mut self) -> Result<AstNode, ParseError> {
        let first = self.parse_pipeline_element()?;
        if !self.at(T::Pipe) {
            return Ok(first);
        }
        let loc = first.loc().clone();
        let mut pipe = Pipeline {
            loc,
            elements: vec![first],
        };
        while self.eat(T::Pipe).is_some() {
            self.skip_inline_newlines();
            pipe.elements.push(self.parse_pipeline_element()?);
        }
        Ok(AstNode::Pipeline(pipe))
    }

    fn parse_pipeline_element(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.cur();
        if tok.ty == T::Generic {
            return self.parse_command();
        }
        if tok.ty == T::Amp
            || (tok.ty == T::Dot
                && self.peek(1).ty != T::Dot
                && matches!(
                    self.peek(1).ty,
                    T::Variable | T::Generic | T::StringSq | T::StringDq | T::LParen
                ))
        {
            return self.parse_invocation_command();
        }
        if tok.ty == T::Keyword
            && ![
                "if", "while", "for", "foreach", "switch", "function", "filter", "workflow", "do",
                "try", "return", "throw", "param",
            ]
            .contains(&tok.text.as_deref().unwrap_or(""))
        {
            return self.parse_command();
        }
        self.parse_expression()
    }

    fn parse_invocation_command(&mut self) -> Result<AstNode, ParseError> {
        let op_tok = self.next().clone();
        let loc = make_loc(&op_tok);
        let mut cmd = Command {
            loc: loc.clone(),
            invocation_operator: Some(op_tok.value),
            ..Default::default()
        };
        cmd.name_expr = Some(box_node(self.parse_primary()?));
        if let Some(ne) = &cmd.name_expr {
            cmd.name = match ne.as_ref() {
                AstNode::StringLiteral(n) => n.value.clone(),
                AstNode::BareWord(n) => n.value.clone(),
                _ => String::new(),
            };
        }
        self.parse_command_elements(&mut cmd)?;
        Ok(AstNode::Command(cmd))
    }

    fn parse_command(&mut self) -> Result<AstNode, ParseError> {
        let (name_tok, name) = self.scan_bareword();
        let mut cmd = Command {
            loc: make_loc(&name_tok),
            name,
            ..Default::default()
        };
        self.parse_command_elements(&mut cmd)?;
        self.maybe_parse_add_type(&mut cmd);
        Ok(AstNode::Command(cmd))
    }

    const CSHARP_PARAMS: &'static [&'static str] = &["memberdefinition", "typedefinition"];

    fn maybe_parse_add_type(&self, cmd: &mut Command) {
        if cmd.name.to_lowercase() != "add-type" {
            return;
        }
        if let Some((code, parameter)) = self.find_csharp_code(&cmd.elements) {
            let mut cs = parse_csharp_member_def(&code);
            cs.parameter = parameter;
            cs.loc = cmd.loc.clone();
            cmd.csharp = Some(Box::new(AstNode::CSharpMemberDef(cs)));
        }
    }

    /// Locate the C# type/member definition handed to `Add-Type`.
    ///
    /// Accepts it as the argument of `-TypeDefinition` / `-MemberDefinition`,
    /// or as a bare positional argument (`-TypeDefinition` is `Position 0` of
    /// Add-Type's source parameter set). A variable argument is resolved to the
    /// string literal it was assigned earlier in the script (simple constant
    /// propagation), so `$c = @"..."@; Add-Type $c` is handled. Returns
    /// `(code, parameter_label)`.
    fn find_csharp_code(&self, elements: &[AstNode]) -> Option<(String, String)> {
        // 1) Named -MemberDefinition / -TypeDefinition.
        for (i, el) in elements.iter().enumerate() {
            let AstNode::CommandParameter(p) = el else {
                continue;
            };
            let pname = p.name.to_lowercase();
            if !Self::CSHARP_PARAMS
                .iter()
                .any(|&full| full.starts_with(&pname))
            {
                continue;
            }
            let value = p.argument.as_deref().or_else(|| match elements.get(i + 1) {
                Some(e) if !matches!(e, AstNode::CommandParameter(_)) => Some(e),
                _ => None,
            });
            if let Some(code) = value.and_then(|v| self.as_csharp_code(v)) {
                return Some((code, pname));
            }
        }
        // 2) First positional string / resolvable-variable argument.
        for el in elements {
            if matches!(el, AstNode::CommandParameter(_)) {
                continue;
            }
            if let Some(code) = self.as_csharp_code(el) {
                return Some((code, "positional".to_owned()));
            }
        }
        None
    }

    /// A string literal's value, or an unscoped variable resolved to the string
    /// it was assigned earlier in the script.
    fn as_csharp_code(&self, node: &AstNode) -> Option<String> {
        match node {
            AstNode::StringLiteral(s) => Some(s.value.clone()),
            AstNode::Variable(v) if v.scope.is_none() => {
                self.vars.get(&v.name.to_lowercase()).cloned()
            }
            _ => None,
        }
    }

    fn parse_command_elements(&mut self, cmd: &mut Command) -> Result<(), ParseError> {
        loop {
            let tok = self.cur().clone();
            if matches!(
                tok.ty,
                T::Newline | T::Semicolon | T::Pipe | T::RBrace | T::RParen | T::RBracket | T::Eof
            ) {
                break;
            }
            // Pipeline-chain operators end the command (handled one level up).
            if tok.ty == T::Operator && matches!(tok.value.as_str(), "&&" | "||") {
                break;
            }
            // `--%` stop-parsing token: the remainder of the line is verbatim.
            if self.src[tok.pos..].starts_with("--%") {
                let begin = tok.pos;
                let rest = &self.src[begin..];
                let end = begin + rest.find('\n').unwrap_or(rest.len());
                let value = self.src.get(begin..end).unwrap_or("").trim_end().to_owned();
                let loc = make_loc(&tok);
                self.resync_past(end);
                cmd.elements
                    .push(AstNode::BareWord(BareWord { loc, value }));
                break;
            }
            if tok.ty == T::Redirect {
                let rtok = self.next().clone();
                let operator = rtok.value.clone();
                // `2>&1`-style stream merges encode their destination in the
                // operator; a file redirection takes the next argument as target.
                let target = if operator.contains('&') {
                    None
                } else if self.starts_argument() {
                    Some(box_node(self.parse_command_argument()?))
                } else {
                    None
                };
                cmd.redirections.push(Redirection {
                    loc: make_loc(&rtok),
                    operator,
                    target,
                });
                continue;
            }
            if tok.ty == T::Parameter {
                let ptok = self.next().clone();
                let param_end = ptok.pos + ptok.value.len();
                let mut param = CommandParameter {
                    loc: make_loc(&ptok),
                    name: ptok
                        .text
                        .clone()
                        .unwrap_or_else(|| ptok.value.trim_start_matches('-').to_owned()),
                    argument: None,
                };
                // `-name:value`: a colon-attached argument with no intervening space.
                if self.at(T::Operator) && self.cur().value == ":" && self.cur().pos == param_end {
                    self.next(); // ':'
                    if !self.stmt_end() && !self.at(T::Parameter) {
                        param.argument = Some(box_node(self.parse_command_argument()?));
                    }
                } else if self.starts_argument() && !self.at(T::Parameter) {
                    param.argument = Some(box_node(self.parse_command_argument()?));
                }
                cmd.elements.push(AstNode::CommandParameter(param));
                continue;
            }
            cmd.elements.push(self.parse_command_argument()?);
        }
        Ok(())
    }

    fn starts_argument(&self) -> bool {
        self.at_any(&[
            T::Variable,
            T::Number,
            T::StringSq,
            T::StringDq,
            T::HereStringSq,
            T::HereStringDq,
            T::Generic,
            T::LParen,
            T::LBrace,
            T::LBracket,
            T::AtParen,
            T::AtBrace,
            T::DollarParen,
            T::Keyword,
            T::Parameter,
        ])
    }

    /// Parse a single command argument. Expression-mode introducers (`$var`,
    /// `(...)`, `@(...)`, `@{...}`, `$(...)`, `{...}`, quoted / here-strings, and
    /// `[...]` type / cast literals) are parsed as expressions; everything else
    /// is read as a bareword in *argument mode* directly from the source, so
    /// paths, wildcards, and switches such as `/c` or `*.txt` each form one
    /// argument.
    fn parse_command_argument(&mut self) -> Result<AstNode, ParseError> {
        // A `[`-led argument is a type/cast literal only when the bracket closes
        // cleanly; if the matching `]` is glued to more bareword text (such as
        // `[abc]*.txt`), the whole run is a wildcard/path bareword instead.
        if self.at(T::LBracket) && self.bracket_run_is_bareword() {
            return Ok(self.scan_argument_run());
        }
        if self.at_any(&[
            T::Variable,
            T::LParen,
            T::AtParen,
            T::AtBrace,
            T::DollarParen,
            T::LBrace,
            T::StringSq,
            T::StringDq,
            T::HereStringSq,
            T::HereStringDq,
            T::LBracket,
        ]) {
            return self.parse_expression();
        }
        Ok(self.scan_argument_run())
    }

    /// In argument position, decide whether a `[`-led run is a wildcard/path
    /// bareword rather than a type or cast literal. Returns `true` when the
    /// matching `]` is immediately followed by a bareword character: anything
    /// other than whitespace, a structural terminator, or one of `$`, `(`, `:`
    /// (which begin a cast target or a `::` static access, i.e. an expression).
    fn bracket_run_is_bareword(&self) -> bool {
        let begin = self.cur().pos;
        let mut depth = 0usize;
        for (off, c) in self.src[begin..].char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        let after = begin + off + c.len_utf8();
                        return match self.src[after..].chars().next() {
                            Some(n) => {
                                !n.is_whitespace()
                                    && !matches!(
                                        n,
                                        '|' | ';'
                                            | ','
                                            | '('
                                            | ')'
                                            | '{'
                                            | '}'
                                            | '&'
                                            | '<'
                                            | '>'
                                            | '$'
                                            | ':'
                                    )
                            }
                            None => false,
                        };
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Scan one bareword argument straight from the source using PowerShell's
    /// permissive argument-mode rules. The run extends until unquoted whitespace
    /// or a structural terminator (`| ; , ( ) { } & < >`, or a leading `#`);
    /// embedded single/double-quoted spans and backtick escapes are kept
    /// verbatim. The token cursor is then resynchronised to the first token at
    /// or beyond the end of the run, and the run always consumes at least the
    /// starting token so the caller is guaranteed to make progress.
    /// Read one bareword argument (see [`Self::scan_bareword`]) as an AST node.
    fn scan_argument_run(&mut self) -> AstNode {
        let (start, value) = self.scan_bareword();
        AstNode::BareWord(BareWord {
            loc: make_loc(&start),
            value,
        })
    }

    /// Core of argument / command-mode scanning: read a bareword run from the
    /// source at the current token, resynchronise the cursor past it, and return
    /// the starting token (for location) and the scanned text. Honours quoted
    /// spans and backtick escapes; always consumes at least the starting token.
    fn scan_bareword(&mut self) -> (Token, String) {
        let start = self.cur().clone();
        let begin = start.pos;
        let s = &self.src[begin..];
        let mut iter = s.char_indices().peekable();
        let mut end_rel = 0usize;
        let mut first = true;
        while let Some((off, c)) = iter.next() {
            let is_term = matches!(
                c,
                ' ' | '\t'
                    | '\r'
                    | '\n'
                    | '|'
                    | ';'
                    | ','
                    | '('
                    | ')'
                    | '{'
                    | '}'
                    | '&'
                    | '<'
                    | '>'
            ) || (c == '#' && first);
            if is_term {
                end_rel = off;
                break;
            }
            first = false;
            match c {
                '`' => {
                    // backtick escape: take the next char literally as well
                    if let Some((noff, nc)) = iter.next() {
                        end_rel = noff + nc.len_utf8();
                    } else {
                        end_rel = off + c.len_utf8();
                    }
                }
                '\'' | '"' => {
                    // keep a balanced quoted span as part of the bareword
                    end_rel = off + c.len_utf8();
                    for (qoff, q) in iter.by_ref() {
                        end_rel = qoff + q.len_utf8();
                        if q == c {
                            break;
                        }
                    }
                }
                _ => end_rel = off + c.len_utf8(),
            }
        }
        // Guarantee progress: never consume less than the starting token.
        let min_end = start.value.len();
        let end = begin + end_rel.max(min_end);
        let value = self.src.get(begin..end).unwrap_or("").to_owned();
        self.resync_past(end);
        (start, value)
    }

    // Expressions

    fn parse_expression(&mut self) -> Result<AstNode, ParseError> {
        self.parse_ternary()
    }

    /// Ternary `<cond> ? <if-true> : <if-false>` (PowerShell 7). Looser than
    /// every other expression operator and right-associative, so
    /// `a ? b : c ? d : e` parses as `a ? b : (c ? d : e)`.
    fn parse_ternary(&mut self) -> Result<AstNode, ParseError> {
        let cond = self.parse_logical()?;
        if self.at(T::Operator) && self.cur().value == "?" {
            let q = self.next().clone();
            self.skip_inline_newlines();
            let if_true = self.parse_ternary()?;
            self.skip_inline_newlines();
            if self.at(T::Operator) && self.cur().value == ":" {
                self.next();
            }
            self.skip_inline_newlines();
            let if_false = self.parse_ternary()?;
            return Ok(AstNode::TernaryExpression(TernaryExpression {
                loc: make_loc(&q),
                condition: box_node(cond),
                if_true: box_node(if_true),
                if_false: box_node(if_false),
            }));
        }
        Ok(cond)
    }

    fn binary_level(
        &mut self,
        ops: &[&str],
        sub: fn(&mut Self) -> Result<AstNode, ParseError>,
    ) -> Result<AstNode, ParseError> {
        let mut left = sub(self)?;
        while self.at(T::Operator) && ops.contains(&self.cur().value.to_lowercase().as_str()) {
            let op_tok = self.next().clone();
            self.skip_inline_newlines();
            let right = sub(self)?;
            let loc = left.loc().clone();
            left = AstNode::BinaryExpression(BinaryExpression {
                loc,
                operator: op_tok.value.to_lowercase(),
                left: box_node(left),
                right: box_node(right),
            });
        }
        Ok(left)
    }

    fn parse_logical(&mut self) -> Result<AstNode, ParseError> {
        self.binary_level(&["-and", "-or", "-xor"], Self::parse_coalesce)
    }
    /// Null-coalescing `??` (PowerShell 7): looser than comparison, tighter than
    /// the logical operators.
    fn parse_coalesce(&mut self) -> Result<AstNode, ParseError> {
        self.binary_level(&["??"], Self::parse_comparison)
    }
    fn parse_comparison(&mut self) -> Result<AstNode, ParseError> {
        // Comparison, type, and bitwise/shift operators all share one precedence
        // level in PowerShell's grammar (left-associative).
        self.binary_level(
            &[
                "-eq",
                "-ne",
                "-gt",
                "-ge",
                "-lt",
                "-le",
                "-like",
                "-notlike",
                "-match",
                "-notmatch",
                "-replace",
                "-contains",
                "-notcontains",
                "-in",
                "-notin",
                "-is",
                "-isnot",
                "-split",
                "-join",
                "-as",
                "-ceq",
                "-cne",
                "-ieq",
                "-ine",
                "-imatch",
                "-cmatch",
                "-band",
                "-bor",
                "-bxor",
                "-shl",
                "-shr",
            ],
            Self::parse_additive,
        )
    }
    fn parse_additive(&mut self) -> Result<AstNode, ParseError> {
        self.binary_level(&["+", "-"], Self::parse_multiplicative)
    }
    fn parse_multiplicative(&mut self) -> Result<AstNode, ParseError> {
        self.binary_level(&["*", "/", "%"], Self::parse_format)
    }
    fn parse_format(&mut self) -> Result<AstNode, ParseError> {
        self.binary_level(&["-f"], Self::parse_range)
    }
    fn parse_range(&mut self) -> Result<AstNode, ParseError> {
        self.binary_level(&[".."], Self::parse_unary)
    }

    fn parse_unary(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.cur();
        // Unary comma: `,$x` is a single-element array.
        if tok.ty == T::Comma {
            let comma = self.next().clone();
            self.skip_inline_newlines();
            let operand = self.parse_unary()?;
            return Ok(AstNode::ArrayLiteral(ArrayLiteral {
                loc: make_loc(&comma),
                elements: vec![operand],
            }));
        }
        if tok.ty == T::Operator
            && [
                "-not", "-bnot", "!", "-", "+", "--", "++", "-split", "-join",
            ]
            .contains(&tok.value.to_lowercase().as_str())
        {
            let op_tok = self.next().clone();
            let operand = self.parse_unary()?;
            return Ok(AstNode::UnaryExpression(UnaryExpression {
                loc: make_loc(&op_tok),
                operator: op_tok.value.to_lowercase(),
                operand: box_node(operand),
                postfix: false,
            }));
        }
        if tok.ty == T::LBracket {
            let saved = self.i;
            if let Some(type_name) = self.try_parse_type() {
                if self.at(T::DoubleColon) {
                    let loc = make_loc(&self.toks[saved]);
                    let target = AstNode::TypeExpression(TypeExpression {
                        loc: loc.clone(),
                        name: type_name,
                    });
                    return self.parse_postfix(target);
                }
                if self.starts_value() {
                    let loc = make_loc(&self.toks[saved]);
                    let expr = self.parse_unary()?;
                    return Ok(AstNode::CastExpression(CastExpression {
                        loc,
                        type_name,
                        expression: box_node(expr),
                    }));
                }
                let loc = make_loc(&self.toks[saved]);
                return Ok(AstNode::TypeExpression(TypeExpression {
                    loc,
                    name: type_name,
                }));
            }
            self.i = saved;
        }
        let primary = self.parse_primary()?;
        self.parse_postfix(primary)
    }

    fn starts_value(&self) -> bool {
        self.at_any(&[
            T::Variable,
            T::Number,
            T::StringSq,
            T::StringDq,
            T::HereStringSq,
            T::HereStringDq,
            T::LParen,
            T::AtParen,
            T::AtBrace,
            T::DollarParen,
            T::LBrace,
            T::LBracket,
            T::Generic,
        ])
    }

    fn try_parse_type(&mut self) -> Option<String> {
        if !self.at(T::LBracket) {
            return None;
        }
        let start = self.i;
        self.next(); // [
        let mut parts = Vec::new();
        let mut depth = 1usize;
        loop {
            let t = self.cur();
            match t.ty {
                T::LBracket => {
                    depth += 1;
                    parts.push("[".to_owned());
                    self.next();
                }
                T::RBracket => {
                    depth -= 1;
                    if depth == 0 {
                        self.next();
                        break;
                    }
                    parts.push("]".to_owned());
                    self.next();
                }
                T::Generic | T::Keyword => {
                    parts.push(t.value.clone());
                    self.next();
                }
                T::Dot => {
                    parts.push(".".to_owned());
                    self.next();
                }
                T::Comma => {
                    parts.push(",".to_owned());
                    self.next();
                }
                T::Operator if t.value == "+" => {
                    parts.push("+".to_owned());
                    self.next();
                }
                T::Number => {
                    parts.push(t.value.clone());
                    self.next();
                }
                _ => {
                    self.i = start;
                    return None;
                }
            }
        }
        Some(parts.join(""))
    }

    fn parse_postfix(&mut self, mut node: AstNode) -> Result<AstNode, ParseError> {
        loop {
            let tok = self.cur().clone();
            let is_qdot = tok.ty == T::Operator && tok.value == "?.";
            let is_qindex = tok.ty == T::Operator && tok.value == "?[";
            if tok.ty == T::Dot || tok.ty == T::DoubleColon || is_qdot {
                let is_static = tok.ty == T::DoubleColon;
                self.next();
                let (member, member_expr) = self.read_member_name()?;
                let loc = make_loc(&tok);
                if self.at(T::LParen) {
                    let args = self.parse_arg_list()?;
                    node = AstNode::InvokeMember(InvokeMember {
                        loc,
                        target: box_node(node),
                        member,
                        member_expr: member_expr.map(box_node),
                        is_static,
                        null_conditional: is_qdot,
                        arguments: args,
                    });
                } else {
                    node = AstNode::MemberAccess(MemberAccess {
                        loc,
                        target: box_node(node),
                        member,
                        member_expr: member_expr.map(box_node),
                        is_static,
                        null_conditional: is_qdot,
                    });
                }
            } else if tok.ty == T::LBracket || is_qindex {
                self.next();
                let idx = if self.at(T::RBracket) {
                    None
                } else {
                    Some(box_node(self.parse_expression()?))
                };
                self.eat(T::RBracket);
                node = AstNode::IndexExpression(IndexExpression {
                    loc: make_loc(&tok),
                    target: box_node(node),
                    index: idx,
                    null_conditional: is_qindex,
                });
            } else if tok.ty == T::Operator && (tok.value == "++" || tok.value == "--") {
                self.next();
                node = AstNode::UnaryExpression(UnaryExpression {
                    loc: make_loc(&tok),
                    operator: tok.value,
                    operand: box_node(node),
                    postfix: true,
                });
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn read_member_name(&mut self) -> Result<(String, Option<AstNode>), ParseError> {
        let tok = self.cur().clone();
        if tok.ty == T::Generic || tok.ty == T::Keyword {
            self.next();
            return Ok((tok.value, None));
        }
        if tok.ty == T::Variable {
            self.next();
            let v = AstNode::Variable(Variable {
                loc: make_loc(&tok),
                name: tok.text.clone().unwrap_or_default(),
                scope: tok.scope.clone(),
                raw: tok.value.clone(),
                ..Default::default()
            });
            return Ok((String::new(), Some(v)));
        }
        if tok.ty == T::StringSq || tok.ty == T::StringDq {
            self.next();
            return Ok((tok.text.clone().unwrap_or_default(), None));
        }
        if tok.ty == T::LBrace {
            let sb = self.parse_block_value()?;
            return Ok((
                String::new(),
                Some(AstNode::ScriptBlockExpression(ScriptBlockExpression {
                    loc: make_loc(&tok),
                    body: sb,
                })),
            ));
        }
        Ok((String::new(), None))
    }

    fn parse_arg_list(&mut self) -> Result<Vec<AstNode>, ParseError> {
        self.eat(T::LParen);
        self.skip_inline_newlines();
        let mut args = Vec::new();
        if !self.at(T::RParen) {
            args.push(self.parse_expression()?);
            while self.eat(T::Comma).is_some() {
                self.skip_inline_newlines();
                args.push(self.parse_expression()?);
            }
        }
        self.skip_inline_newlines();
        self.eat(T::RParen);
        Ok(args)
    }

    // Primary

    fn parse_primary(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.cur().clone();
        match tok.ty {
            T::Variable => {
                self.next();
                let node = AstNode::Variable(Variable {
                    loc: make_loc(&tok),
                    name: tok.text.clone().unwrap_or_default(),
                    scope: tok.scope.clone(),
                    splat: tok.splat,
                    raw: tok.value,
                });
                return self.maybe_array_literal(node);
            }
            T::Number => {
                self.next();
                let node = AstNode::NumberLiteral(NumberLiteral {
                    loc: make_loc(&tok),
                    raw: tok.value.clone(),
                    value: parse_number(&tok.value),
                });
                return self.maybe_array_literal(node);
            }
            T::StringSq | T::StringDq | T::HereStringSq | T::HereStringDq => {
                self.next();
                let kind = match tok.ty {
                    T::StringSq => "single",
                    T::StringDq => "double",
                    T::HereStringSq => "here_single",
                    // The enclosing match arm admits exactly the four string
                    // kinds, so the only one left here is `HereStringDq`.
                    _ => "here_double",
                };
                let text = tok.text.clone().unwrap_or_default();
                let expandable =
                    (tok.ty == T::StringDq || tok.ty == T::HereStringDq) && text.contains('$');
                let parts = if expandable {
                    extract_interpolations(&text, &make_loc(&tok))
                } else {
                    Vec::new()
                };
                let node = AstNode::StringLiteral(StringLiteral {
                    loc: make_loc(&tok),
                    value: text,
                    kind: kind.into(),
                    raw: tok.value,
                    expandable,
                    parts,
                });
                return self.maybe_array_literal(node);
            }
            T::LParen => {
                self.next();
                self.skip_inline_newlines();
                let inner = self.parse_pipeline()?;
                self.skip_inline_newlines();
                self.eat(T::RParen);
                return Ok(AstNode::ParenExpression(ParenExpression {
                    loc: make_loc(&tok),
                    expression: box_node(inner),
                }));
            }
            T::DollarParen => {
                self.next();
                let sb = self.parse_statement_list(false, &[T::RParen]);
                self.eat(T::RParen);
                return Ok(AstNode::SubExpression(SubExpression {
                    loc: make_loc(&tok),
                    body: sb,
                }));
            }
            T::AtParen => {
                self.next();
                let sb = self.parse_statement_list(false, &[T::RParen]);
                self.eat(T::RParen);
                let elems = sb.statements;
                return Ok(AstNode::ArrayExpression(ArrayExpression {
                    loc: make_loc(&tok),
                    elements: elems,
                }));
            }
            T::AtBrace => return self.parse_hashtable(),
            T::LBrace => {
                let sb = self.parse_block_value()?;
                return Ok(AstNode::ScriptBlockExpression(ScriptBlockExpression {
                    loc: make_loc(&tok),
                    body: sb,
                }));
            }
            T::LBracket => {
                if let Some(tn) = self.try_parse_type() {
                    return Ok(AstNode::TypeExpression(TypeExpression {
                        loc: make_loc(&tok),
                        name: tn,
                    }));
                }
            }
            T::Generic | T::Keyword => {
                self.next();
                return Ok(AstNode::BareWord(BareWord {
                    loc: make_loc(&tok),
                    value: tok.value,
                }));
            }
            _ => {}
        }
        Err(ParseError {
            msg: format!("unexpected {:?} {:?}", tok.ty, tok.value),
            line: tok.line,
            col: tok.col,
        })
    }

    fn maybe_array_literal(&mut self, first: AstNode) -> Result<AstNode, ParseError> {
        if !self.at(T::Comma) {
            return Ok(first);
        }
        let loc = first.loc().clone();
        let mut arr = ArrayLiteral {
            loc,
            elements: vec![first],
        };
        while self.eat(T::Comma).is_some() {
            self.skip_inline_newlines();
            arr.elements.push(self.parse_unary()?);
        }
        Ok(AstNode::ArrayLiteral(arr))
    }

    fn parse_hashtable(&mut self) -> Result<AstNode, ParseError> {
        let tok = self.next().clone(); // @{
        let mut node = HashtableExpression {
            loc: make_loc(&tok),
            ..Default::default()
        };
        self.skip_newlines();
        while !self.eof() && !self.at(T::RBrace) {
            let key = self.parse_unary()?;
            let val = if self.at(T::Operator) && self.cur().value == "=" {
                self.next();
                self.skip_inline_newlines();
                self.parse_statement()?.unwrap_or_default()
            } else {
                AstNode::ErrorNode(ErrorNode::default())
            };
            node.entries.push((box_node(key), box_node(val)));
            self.skip_newlines();
        }
        self.eat(T::RBrace);
        Ok(AstNode::HashtableExpression(node))
    }

    // Utilities

    fn skip_balanced(&mut self, open: T, close: T) {
        if !self.at(open) {
            return;
        }
        let mut depth = 0usize;
        while !self.eof() {
            let ty = self.cur().ty;
            if ty == open {
                depth += 1;
            }
            if ty == close {
                depth -= 1;
                if depth == 0 {
                    self.next();
                    return;
                }
            }
            self.next();
        }
    }

    fn recover(&mut self, e: ParseError) -> ErrorNode {
        self.errors.push(e.msg.clone());
        let start_pos = self.cur().pos;
        while !self.eof() && !self.at_any(&[T::Newline, T::Semicolon]) {
            self.next();
        }
        let raw =
            self.src[start_pos.min(self.src.len())..self.cur().pos.min(self.src.len())].to_owned();
        ErrorNode {
            loc: Location {
                line: e.line,
                col: e.col,
                pos: start_pos,
            },
            message: e.msg,
            raw,
        }
    }
}

// Number parsing

fn parse_number(raw: &str) -> Option<f64> {
    let lower = raw.to_lowercase();

    // Binary multiplier suffixes (1kb, 2mb, …) apply to the literal as a whole.
    let (body, mult): (&str, f64) = if let Some(b) = lower.strip_suffix("kb") {
        (b, 1024.0)
    } else if let Some(b) = lower.strip_suffix("mb") {
        (b, 1024.0_f64.powi(2))
    } else if let Some(b) = lower.strip_suffix("gb") {
        (b, 1024.0_f64.powi(3))
    } else if let Some(b) = lower.strip_suffix("tb") {
        (b, 1024.0_f64.powi(4))
    } else if let Some(b) = lower.strip_suffix("pb") {
        (b, 1024.0_f64.powi(5))
    } else {
        (lower.as_str(), 1.0)
    };

    if let Some(hex) = body.strip_prefix("0x") {
        // In a hex literal `a`–`f` (incl. `d`) are digits, not type suffixes.
        // Only strip integer-type suffix letters that are never hex digits,
        // e.g. the `L` in `0x10L` or the `u`/`y`/`s` size markers.
        let hex = hex.trim_end_matches(['l', 'u', 'y', 's']);
        i64::from_str_radix(hex, 16).ok().map(|i| i as f64 * mult)
    } else {
        // Decimal/float: `d` (decimal), `l` (long), `u` (unsigned) are suffixes.
        let dec = body.trim_end_matches(['l', 'd', 'u']);
        if dec.contains('.') || dec.contains('e') {
            dec.parse::<f64>().ok().map(|f| f * mult)
        } else {
            dec.parse::<i64>().ok().map(|i| i as f64 * mult)
        }
    }
}

/// Parse a PowerShell source string into a `ScriptBlock`.
pub fn parse(source: &str) -> (ScriptBlock, Vec<String>) {
    parse_tokens(super::lexer::tokenize(source), source)
}

/// Parse pre-tokenized input, avoiding a redundant lex pass when the caller
/// already has the token stream (the engine tokenizes once and reuses it).
pub fn parse_tokens(tokens: Vec<Token>, source: &str) -> (ScriptBlock, Vec<String>) {
    // Strip a leading BOM so the byte offsets used for source slicing line up
    // with the BOM-less token stream produced by `tokenize`.
    let source = crate::encoding::strip_bom(source);
    let mut p = Parser::new(tokens, source.to_owned());
    let tree = p.parse();
    (tree, p.errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_node(src: &str, pred: impl Fn(&AstNode) -> bool) -> bool {
        let (tree, _) = parse(src);
        let mut found = false;
        AstNode::ScriptBlock(tree).walk(&mut |n| {
            if pred(n) {
                found = true;
            }
        });
        found
    }

    #[test]
    fn parses_subexpression() {
        // Regression: `$( ... )` should produce a SubExpression node now that
        // the lexer emits a DollarParen token.
        assert!(contains_node("$(Get-Date)", |n| matches!(
            n,
            AstNode::SubExpression(_)
        )));
    }

    #[test]
    fn hex_literals_ending_in_letters_are_not_truncated() {
        // Regression: the type-suffix trim used to strip a trailing `d` from
        // hex literals (`0x2d` -> `0x2` = 2). `d`/`a`..`f` are hex digits.
        assert_eq!(parse_number("0x2d"), Some(45.0));
        assert_eq!(parse_number("0x0d"), Some(13.0));
        assert_eq!(parse_number("0xdd"), Some(221.0));
        assert_eq!(parse_number("0xff"), Some(255.0));
        // Genuine integer-type suffixes on hex are still honoured.
        assert_eq!(parse_number("0x10l"), Some(16.0));
        assert_eq!(parse_number("0xffu"), Some(255.0));
    }

    #[test]
    fn decimal_type_suffixes_still_strip() {
        assert_eq!(parse_number("100d"), Some(100.0));
        assert_eq!(parse_number("42l"), Some(42.0));
        assert_eq!(parse_number("2kb"), Some(2048.0));
    }

    #[test]
    fn prefix_decrement_is_a_unary_expression() {
        // Regression: the prefix `--` operator literal had a stray leading space.
        assert!(contains_node("--$x", |n| matches!(
            n,
            AstNode::UnaryExpression(u) if u.operator == "--" && !u.postfix
        )));
    }

    #[test]
    fn recovers_from_unicode_without_panicking() {
        // A lone closing brace forces error recovery, which slices the source;
        // with multi-byte input this must not panic.
        let (_, errors) = parse("café }");
        // whether or not recovery fired, reaching this line means no panic.
        let _ = errors;
    }

    #[test]
    fn assignment_is_detected() {
        assert!(contains_node("$x = 1", |n| matches!(
            n,
            AstNode::AssignmentStatement(_)
        )));
    }

    #[test]
    fn pipeline_splits_on_pipe() {
        let (tree, _) = parse("Get-Process | Where-Object Name");
        let has_pipeline = tree
            .statements
            .iter()
            .any(|s| matches!(s, AstNode::Pipeline(p) if p.elements.len() == 2));
        assert!(has_pipeline);
    }

    // Add-Type C# extraction (named / positional / via variable)

    const WIN32: &str = r#"@"
using System;
using System.Runtime.InteropServices;
public class Win32 {
[DllImport("kernel32")]
public static extern IntPtr CreateThread(IntPtr a);
}
"@"#;

    /// P/Invoke function names extracted from the first `Add-Type` command.
    fn add_type_imports(src: &str) -> Vec<String> {
        let (tree, _) = parse(src);
        let mut funcs = Vec::new();
        AstNode::ScriptBlock(tree).walk(&mut |n| {
            if let AstNode::CSharpMemberDef(cs) = n {
                funcs = cs.imports.iter().map(|i| i.function.clone()).collect();
            }
        });
        funcs
    }

    #[test]
    fn add_type_named_inline_extracts_csharp() {
        let imps = add_type_imports(&format!("Add-Type -TypeDefinition {WIN32}"));
        assert!(imps.contains(&"CreateThread".to_string()), "{imps:?}");
    }

    #[test]
    fn add_type_positional_inline_extracts_csharp() {
        // `Add-Type @"..."@` has no -TypeDefinition, so the source is positional.
        let imps = add_type_imports(&format!("Add-Type {WIN32}"));
        assert!(imps.contains(&"CreateThread".to_string()), "{imps:?}");
    }

    #[test]
    fn dllimport_params_correct_when_non_ascii_precedes() {
        // A multi-byte char before the signature used to desync the byte offset
        // from a char index and corrupt the first parameter's type.
        let code =
            "// caf\u{e9}\n[DllImport(\"k.dll\")] public static extern void F(int a, int b);";
        let def = parse_csharp_member_def(code);
        assert_eq!(def.imports.len(), 1);
        let p = &def.imports[0].params;
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].type_name.as_str(), p[0].name.as_str()), ("int", "a"));
        assert_eq!((p[1].type_name.as_str(), p[1].name.as_str()), ("int", "b"));
    }

    #[test]
    fn dllimport_non_ascii_near_end_does_not_panic() {
        // Same root cause, but with the paren near the end so the bad index
        // could run past the buffer. Must not panic.
        let code = "\u{e9}[DllImport(\"k\")]public static extern void F(int a)";
        let _ = parse_csharp_member_def(code);
    }

    #[test]
    fn leading_bom_is_stripped() {
        let toks = crate::tokenize("\u{feff}$x = 1");
        assert_eq!(toks[0].value, "$x", "BOM should not survive as a token");
        let (_, errs) = parse("\u{feff}$x = 1");
        assert!(errs.is_empty(), "BOM should be stripped, got {errs:?}");
    }

    #[test]
    fn add_type_named_variable_is_resolved() {
        let src = format!("$c = {WIN32}\nAdd-Type -TypeDefinition $c");
        assert!(
            add_type_imports(&src).contains(&"CreateThread".to_string()),
            "named -TypeDefinition with a variable should resolve"
        );
    }

    #[test]
    fn add_type_positional_variable_is_resolved() {
        // The reported snippet's shape: `$Win32 = @"..."@; Add-Type $Win32`.
        let src = format!("$Win32 = {WIN32}\nAdd-Type $Win32");
        assert!(
            add_type_imports(&src).contains(&"CreateThread".to_string()),
            "positional variable should resolve to its assigned here-string"
        );
    }

    #[test]
    fn add_type_variable_resolution_is_case_insensitive() {
        let src = format!("$Win32 = {WIN32}\nAdd-Type $win32");
        assert!(add_type_imports(&src).contains(&"CreateThread".to_string()));
    }

    #[test]
    fn add_type_unresolved_variable_yields_no_csharp() {
        // No prior assignment to resolve, so this must attach nothing instead of panicking.
        assert!(add_type_imports("Add-Type $never_assigned").is_empty());
    }

    fn all_nodes(src: &str) -> Vec<AstNode> {
        let (tree, _) = parse(src);
        let mut v = Vec::new();
        AstNode::ScriptBlock(tree).walk(&mut |n| v.push(n.clone()));
        v
    }

    #[test]
    fn null_conditional_member_and_index() {
        // `?.` / `?[` produce the same nodes as `.` / `[`, flagged
        // null-conditional, and the unparser re-emits the `?.` / `?[` form.
        let ma = all_nodes("$x?.Foo")
            .into_iter()
            .find_map(|n| match n {
                AstNode::MemberAccess(m) => Some(m),
                _ => None,
            })
            .expect("MemberAccess");
        assert!(ma.null_conditional);
        assert_eq!(ma.member, "Foo");

        let ix = all_nodes("$a?[0]")
            .into_iter()
            .find_map(|n| match n {
                AstNode::IndexExpression(i) => Some(i),
                _ => None,
            })
            .expect("IndexExpression");
        assert!(ix.null_conditional);

        // A plain access stays non-null-conditional.
        assert!(all_nodes("$x.Foo")
            .into_iter()
            .any(|n| matches!(n, AstNode::MemberAccess(m) if !m.null_conditional)));

        assert_eq!(crate::unparse_source("$x?.Foo").trim(), "$x?.Foo");
        assert_eq!(crate::unparse_source("$a?[0]").trim(), "$a?[0]");
    }

    #[test]
    fn bracket_led_wildcard_is_bareword_but_cast_is_preserved() {
        // `[abc]*.txt` is a wildcard/path bareword argument...
        assert!(contains_node("Get-Item [abc]*.txt", |n| matches!(
            n,
            AstNode::BareWord(b) if b.value == "[abc]*.txt"
        )));
        // ...while a real cast argument stays a CastExpression...
        assert!(contains_node("Write-Output [int]$x", |n| matches!(
            n,
            AstNode::CastExpression(_)
        )));
        // ...and a lone `[int]` type literal is not turned into a bareword.
        assert!(!contains_node("Write-Output [int]", |n| matches!(
            n,
            AstNode::BareWord(b) if b.value == "[int]"
        )));
    }

    #[test]
    fn structured_redirections() {
        // File redirection: operator plus a parsed target node.
        let cmd = all_nodes("cmd > out.txt")
            .into_iter()
            .find_map(|n| match n {
                AstNode::Command(c) if !c.redirections.is_empty() => Some(c),
                _ => None,
            })
            .expect("Command with redirection");
        assert_eq!(cmd.redirections.len(), 1);
        assert_eq!(cmd.redirections[0].operator, ">");
        assert!(matches!(
            cmd.redirections[0].target.as_deref(),
            Some(AstNode::BareWord(b)) if b.value == "out.txt"
        ));

        // Stream-merge redirection: the operator carries it, no target node.
        let cmd = all_nodes("cmd 2>&1")
            .into_iter()
            .find_map(|n| match n {
                AstNode::Command(c) if !c.redirections.is_empty() => Some(c),
                _ => None,
            })
            .expect("Command with merge redirection");
        assert_eq!(cmd.redirections[0].operator, "2>&1");
        assert!(cmd.redirections[0].target.is_none());

        assert_eq!(
            crate::unparse_source("cmd > out.txt").trim(),
            "cmd > out.txt"
        );
        assert_eq!(crate::unparse_source("cmd 2>&1").trim(), "cmd 2>&1");
    }

    #[test]
    fn structured_member_attributes() {
        let attrs = |src: &str| {
            all_nodes(src)
                .into_iter()
                .find_map(|n| match n {
                    AstNode::ClassMember(m) if !m.attributes.is_empty() => Some(m.attributes),
                    _ => None,
                })
                .unwrap_or_default()
        };

        let a = attrs("class C { [ValidateSet('A','B')] [string]$X }");
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].name, "ValidateSet");
        assert!(a[0].paren);
        assert_eq!(a[0].positional, vec!["'A'".to_string(), "'B'".to_string()]);
        assert!(a[0].named.is_empty());

        let a = attrs("class C { [Parameter(Mandatory=$true)] [string]$X }");
        assert_eq!(a[0].name, "Parameter");
        assert_eq!(
            a[0].named,
            vec![("Mandatory".to_string(), "$true".to_string())]
        );

        // A comma inside a quoted argument is literal text, not a separator.
        let a = attrs("class C { [ValidateSet('A,B', 'C')] [string]$X }");
        assert_eq!(
            a[0].positional,
            vec!["'A,B'".to_string(), "'C'".to_string()]
        );

        // An empty-paren attribute keeps its `()` through a round-trip, so it
        // re-parses as an attribute rather than a bare type name.
        assert!(
            crate::unparse_source("class C { [ValidateNotNull()] [string]$X }")
                .contains("ValidateNotNull()")
        );
    }
}
