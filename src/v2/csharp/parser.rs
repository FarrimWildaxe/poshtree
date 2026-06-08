//! C# parser for the Add-Type dialect.
//!
//! Recursive descent over [`cs_lex`] output. Declarations, parameters, and
//! scopes are modeled precisely, since that is what renaming binds against;
//! statement bodies are parsed as a scope tree with local declarations and
//! identifier references captured, which is enough for the resolver without a
//! full expression grammar. Unmatched input recovers into
//! [`Error`](CsNodeKind::Error) nodes. A depth guard and a guaranteed
//! one-token-minimum of progress per loop keep it from recursing without bound
//! or hanging on adversarial input.

use super::ast::{CsName, CsNode, CsNodeKind, CsTypeKind, CsUnit};
use super::lexer::cs_lex;
use super::tokens::{CsToken, CsTokenKind as K};
use crate::v2::span::Span;

const MAX_DEPTH: usize = 200;

/// Parses a C# source region into a [`CsUnit`].
///
/// `base` is the absolute offset of `code` in the original file; every node
/// span indexes the file directly.
pub fn cs_parse(code: &str, base: usize) -> CsUnit {
    let toks = cs_lex(code, base);
    let mut p = CsParser {
        toks,
        pos: 0,
        depth: 0,
        errors: Vec::new(),
    };
    let start = p.span().start;
    let mut children = Vec::new();
    while !p.at(K::Eof) {
        let before = p.pos;
        if let Some(node) = p.top_item() {
            children.push(node);
        } else {
            children.push(p.recover());
        }
        if p.pos == before {
            // No rule consumed a token; force progress so we always terminate.
            children.push(p.recover_one());
        }
    }
    let end = p.span().end;
    let root = CsNode::new(CsNodeKind::Unit, Span::new(start, end), children);
    CsUnit {
        root,
        errors: p.errors,
    }
}

struct CsParser {
    toks: Vec<CsToken>,
    pos: usize,
    depth: usize,
    errors: Vec<Span>,
}

impl CsParser {
    // token cursor

    fn tok(&self) -> &CsToken {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }
    fn k(&self) -> K {
        self.tok().kind
    }
    fn span(&self) -> Span {
        self.tok().span
    }
    fn at(&self, kind: K) -> bool {
        self.k() == kind
    }
    fn at_kw(&self, kw: &str) -> bool {
        self.tok().is_kw(kw)
    }
    fn advance(&mut self) {
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
    }
    fn eat(&mut self, kind: K) -> bool {
        if self.at(kind) {
            self.advance();
            true
        } else {
            false
        }
    }
    /// End offset of the most recently consumed token.
    fn last_end(&self) -> usize {
        if self.pos == 0 {
            self.span().start
        } else {
            self.toks[self.pos - 1].span.end
        }
    }
    /// Consumes an identifier as a name, if present.
    fn ident_name(&mut self) -> Option<CsName> {
        if self.at(K::Ident) {
            let name = CsName {
                text: self.tok().text.clone(),
                span: self.span(),
            };
            self.advance();
            Some(name)
        } else {
            None
        }
    }

    // recovery

    /// Records an error spanning from `start` to the last consumed token.
    fn error_from(&mut self, start: usize) -> CsNode {
        let span = Span::new(start, self.last_end().max(start));
        self.errors.push(span);
        CsNode::leaf(CsNodeKind::Error, span)
    }

    /// Skips to the next top-level sync point (`;` or a balanced `}`), emitting
    /// an error node over the skipped range.
    fn recover(&mut self) -> CsNode {
        let start = self.span().start;
        let mut depth = 0i32;
        while !self.at(K::Eof) {
            match self.k() {
                K::LBrace => {
                    depth += 1;
                    self.advance();
                }
                K::RBrace => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    self.advance();
                    if depth == 0 {
                        break;
                    }
                }
                K::Semicolon if depth == 0 => {
                    self.advance();
                    break;
                }
                _ => self.advance(),
            }
        }
        self.error_from(start)
    }

    /// Consumes exactly one token as an error (last-resort progress guarantee).
    fn recover_one(&mut self) -> CsNode {
        let start = self.span().start;
        self.advance();
        self.error_from(start)
    }

    /// Member-list recovery: skip stray tokens up to the next plausible member
    /// start (an identifier or `[`), a `;`, or the closing `}`, without
    /// balancing over a block, so a following well-formed member survives.
    fn recover_member(&mut self) -> CsNode {
        let start = self.span().start;
        self.advance(); // always make progress
        while !self.at(K::Eof) && !self.at(K::RBrace) {
            if self.at(K::Ident) || self.at(K::LBracket) {
                break;
            }
            if self.eat(K::Semicolon) {
                break;
            }
            self.advance();
        }
        self.error_from(start)
    }

    // top-level items

    fn top_item(&mut self) -> Option<CsNode> {
        let start = self.span().start;
        let mut attrs = self.parse_attrs();
        self.skip_modifiers();
        if self.at_kw("using") {
            return Some(self.parse_using(start));
        }
        if self.at_kw("namespace") {
            return Some(self.parse_namespace(start, attrs));
        }
        if let Some(kind) = self.type_keyword() {
            return Some(self.parse_type(start, kind, &mut attrs));
        }
        // A bare member with no enclosing type: `Add-Type -MemberDefinition`
        // fragments are class members that PowerShell wraps in a class, so a
        // top-level method/field/property is valid here and parses like one.
        self.parse_member(start, attrs)
    }

    fn parse_using(&mut self, start: usize) -> CsNode {
        // `using ...;` or `using (...)` is only top-level as a directive here.
        self.advance(); // using
        while !self.at(K::Eof) && !self.at(K::Semicolon) {
            self.advance();
        }
        self.eat(K::Semicolon);
        CsNode::leaf(CsNodeKind::Using, Span::new(start, self.last_end()))
    }

    fn parse_namespace(&mut self, start: usize, mut attrs: Vec<CsNode>) -> CsNode {
        self.advance(); // namespace
        let name = self.qualified_name().unwrap_or(CsName {
            text: String::new(),
            span: self.span(),
        });
        let mut children = std::mem::take(&mut attrs);
        if self.at(K::Semicolon) {
            // File-scoped namespace: members follow at top level.
            self.advance();
            while !self.at(K::Eof) {
                let before = self.pos;
                if let Some(item) = self.member_or_type() {
                    children.push(item);
                } else {
                    children.push(self.recover_member());
                }
                if self.pos == before {
                    children.push(self.recover_one());
                }
            }
        } else if self.eat(K::LBrace) {
            self.parse_members_into(&mut children);
            self.eat(K::RBrace);
        }
        CsNode::new(
            CsNodeKind::Namespace(name),
            Span::new(start, self.last_end()),
            children,
        )
    }

    fn parse_type(&mut self, start: usize, kind: CsTypeKind, attrs: &mut Vec<CsNode>) -> CsNode {
        self.advance(); // class/struct/interface/enum
        let name = self.ident_name().unwrap_or(CsName {
            text: String::new(),
            span: self.span(),
        });
        self.skip_generic_params();
        // Base list / constraints: skip to the body.
        while !self.at(K::Eof) && !self.at(K::LBrace) && !self.at(K::Semicolon) {
            self.advance();
        }
        let mut children = std::mem::take(attrs);
        if self.eat(K::LBrace) {
            if kind == CsTypeKind::Enum {
                self.parse_enum_members_into(&mut children);
            } else {
                self.parse_members_into(&mut children);
            }
            self.eat(K::RBrace);
        } else {
            self.eat(K::Semicolon);
        }
        CsNode::new(
            CsNodeKind::Type { kind, name },
            Span::new(start, self.last_end()),
            children,
        )
    }

    fn parse_enum_members_into(&mut self, out: &mut Vec<CsNode>) {
        while !self.at(K::Eof) && !self.at(K::RBrace) {
            let before = self.pos;
            self.parse_attrs(); // members may carry attributes
            if let Some(name) = self.ident_name() {
                let mstart = name.span.start;
                out.push(CsNode::leaf(
                    CsNodeKind::EnumMember(name),
                    Span::new(mstart, self.last_end()),
                ));
                // Skip `= value` and the separating comma.
                while !self.at(K::Eof) && !self.at(K::Comma) && !self.at(K::RBrace) {
                    self.advance();
                }
                self.eat(K::Comma);
            } else {
                self.eat(K::Comma);
            }
            if self.pos == before {
                self.advance();
            }
        }
    }

    fn parse_members_into(&mut self, out: &mut Vec<CsNode>) {
        while !self.at(K::Eof) && !self.at(K::RBrace) {
            let before = self.pos;
            if let Some(m) = self.member_or_type() {
                out.push(m);
            } else {
                out.push(self.recover_member());
            }
            if self.pos == before {
                out.push(self.recover_one());
            }
        }
    }

    /// A nested type or a class member.
    fn member_or_type(&mut self) -> Option<CsNode> {
        let start = self.span().start;
        let mut attrs = self.parse_attrs();
        self.skip_modifiers();
        if let Some(kind) = self.type_keyword() {
            return Some(self.parse_type(start, kind, &mut attrs));
        }
        self.parse_member(start, attrs)
    }

    fn parse_member(&mut self, start: usize, attrs: Vec<CsNode>) -> Option<CsNode> {
        self.enter();
        let r = self.parse_member_inner(start, attrs);
        self.leave();
        r
    }

    fn parse_member_inner(&mut self, start: usize, mut attrs: Vec<CsNode>) -> Option<CsNode> {
        if self.depth >= MAX_DEPTH {
            return Some(self.recover());
        }
        if !self.at(K::Ident) {
            return None;
        }
        // First identifier: either a constructor name or the start of a type.
        let first = self.ident_name().unwrap();
        // Constructor: `Name(` directly.
        if self.at(K::LParen) {
            let params = self.parse_params();
            let body = self.parse_member_tail();
            let mut children = std::mem::take(&mut attrs);
            children.extend(params);
            children.extend(body);
            return Some(CsNode::new(
                CsNodeKind::Ctor(first),
                Span::new(start, self.last_end()),
                children,
            ));
        }
        // Otherwise `first` began a (return/field) type; finish the type, which
        // may contribute references, then read the member name.
        let mut type_refs = vec![root_ref(first.clone())];
        self.finish_type(&mut type_refs, Some(first.text.clone()));

        let Some(name) = self.ident_name() else {
            // No member name: treat what we have as a recovered region (but the
            // type references are still useful, so fold them in).
            let mut children = std::mem::take(&mut attrs);
            children.extend(type_refs);
            let span = Span::new(start, self.last_end().max(start));
            self.errors.push(span);
            return Some(CsNode::new(CsNodeKind::Error, span, children));
        };

        let mut children = std::mem::take(&mut attrs);
        children.extend(type_refs);

        if self.at(K::LParen) {
            // Method.
            let params = self.parse_params();
            children.extend(params);
            children.extend(self.parse_member_tail());
            Some(CsNode::new(
                CsNodeKind::Method(name),
                Span::new(start, self.last_end()),
                children,
            ))
        } else if self.at(K::LBrace) || self.at(K::Arrow) {
            // Property (accessor block or expression-bodied).
            let body = self.parse_member_tail();
            children.extend(body);
            Some(CsNode::new(
                CsNodeKind::Property(name),
                Span::new(start, self.last_end()),
                children,
            ))
        } else {
            // Field: this name plus any after commas; skip initializers. The
            // declared names and any initializer references live directly under
            // the Decl, alongside the type references already in `children`.
            children.push(name_decl(name));
            self.skip_field_initializer(&mut children);
            while self.eat(K::Comma) {
                if let Some(more) = self.ident_name() {
                    children.push(name_decl(more));
                    self.skip_field_initializer(&mut children);
                }
            }
            self.eat(K::Semicolon);
            Some(CsNode::new(
                CsNodeKind::Decl,
                Span::new(start, self.last_end()),
                children,
            ))
        }
    }

    /// The tail of a method/property/ctor: a `{ block }`, an `=> expr;`, or a
    /// terminating `;` (abstract/extern/interface members). Returns the body
    /// nodes (a Block, or captured references).
    fn parse_member_tail(&mut self) -> Vec<CsNode> {
        if self.at(K::LBrace) {
            vec![self.parse_block()]
        } else if self.eat(K::Arrow) {
            let mut refs = Vec::new();
            self.scan_expr(&mut refs);
            self.eat(K::Semicolon);
            refs
        } else {
            self.eat(K::Semicolon);
            Vec::new()
        }
    }

    fn skip_field_initializer(&mut self, out: &mut Vec<CsNode>) {
        if self.eat(K::Assign) {
            self.scan_expr(out);
        }
    }

    // parameters

    fn parse_params(&mut self) -> Vec<CsNode> {
        let mut out = Vec::new();
        if !self.eat(K::LParen) {
            return out;
        }
        while !self.at(K::Eof) && !self.at(K::RParen) {
            let before = self.pos;
            self.parse_attrs();
            self.skip_param_modifiers();
            // Parameter type (contributes references), then the parameter name.
            let mut refs = Vec::new();
            if self.at(K::Ident) {
                let first = self.ident_name().unwrap();
                refs.push(root_ref(first.clone()));
                self.finish_type(&mut refs, Some(first.text.clone()));
            }
            out.extend(refs);
            if let Some(name) = self.ident_name() {
                out.push(CsNode::leaf(CsNodeKind::Param(name.clone()), name.span));
            }
            // Default value, up to the comma or close paren.
            if self.eat(K::Assign) {
                self.scan_expr(&mut out);
            }
            self.eat(K::Comma);
            if self.pos == before {
                self.advance();
            }
        }
        self.eat(K::RParen);
        out
    }

    // statements / blocks

    fn parse_block(&mut self) -> CsNode {
        self.enter();
        let node = self.parse_block_inner();
        self.leave();
        node
    }

    fn parse_block_inner(&mut self) -> CsNode {
        let start = self.span().start;
        if self.depth >= MAX_DEPTH {
            // Too deep: skip a balanced block without recursing.
            return self.skip_balanced_block(start);
        }
        self.eat(K::LBrace);
        let mut children = Vec::new();
        while !self.at(K::Eof) && !self.at(K::RBrace) {
            let before = self.pos;
            self.parse_statement(&mut children);
            if self.pos == before {
                self.advance();
            }
        }
        self.eat(K::RBrace);
        CsNode::new(
            CsNodeKind::Block,
            Span::new(start, self.last_end()),
            children,
        )
    }

    fn skip_balanced_block(&mut self, start: usize) -> CsNode {
        let mut depth = 0i32;
        while !self.at(K::Eof) {
            match self.k() {
                K::LBrace => {
                    depth += 1;
                    self.advance();
                }
                K::RBrace => {
                    depth -= 1;
                    self.advance();
                    if depth <= 0 {
                        break;
                    }
                }
                _ => self.advance(),
            }
        }
        CsNode::leaf(CsNodeKind::Block, Span::new(start, self.last_end()))
    }

    fn parse_statement(&mut self, out: &mut Vec<CsNode>) {
        match self.k() {
            K::LBrace => out.push(self.parse_block()),
            K::Semicolon => self.advance(),
            K::Ident if self.is_control_kw() => self.parse_control(out),
            _ => self.parse_simple_statement(out),
        }
    }

    /// Control flow: the keyword, an optional `( ... )` header (which may
    /// declare locals), and a following statement. The header's declarations
    /// and the body are wrapped in a synthetic block so the declarations scope
    /// to it.
    fn parse_control(&mut self, out: &mut Vec<CsNode>) {
        let start = self.span().start;
        self.advance(); // the control keyword
        let mut scope_children = Vec::new();
        if self.at(K::LParen) {
            self.parse_header_into(&mut scope_children);
        }
        // `catch`/`is` type already consumed in header; now the body or branch.
        if self.at(K::LBrace) {
            scope_children.push(self.parse_block());
        } else if !self.at(K::Eof) && !self.at(K::RBrace) {
            // A single statement (which may itself be control flow).
            let before = self.pos;
            self.parse_statement(&mut scope_children);
            if self.pos == before {
                self.advance();
            }
        }
        // `do { } while(...)`, `try {} catch {}`, `if {} else {}` chains: the
        // outer loop picks up the continuation as the next statement.
        out.push(CsNode::new(
            CsNodeKind::Block,
            Span::new(start, self.last_end()),
            scope_children,
        ));
    }

    /// A `( ... )` control header. Declarations inside (`for`, `foreach`,
    /// `using`, `catch`) become NameDecls; everything else is captured as refs.
    fn parse_header_into(&mut self, out: &mut Vec<CsNode>) {
        self.eat(K::LParen);
        // Try a leading declaration: `Type name` optionally `in`/`=`.
        self.try_local_decl(out);
        self.scan_expr(out);
        self.eat(K::RParen);
    }

    fn parse_simple_statement(&mut self, out: &mut Vec<CsNode>) {
        // Try a local declaration; if it doesn't match, scan as an expression.
        self.try_local_decl(out);
        self.scan_expr(out);
        self.eat(K::Semicolon);
    }

    /// Attempts to parse `Type name (= init)? (, name (= init)?)*` as a local
    /// declaration, emitting NameDecls and capturing initializer references.
    /// Returns whether it matched; leaves the cursor put if it did not.
    fn try_local_decl(&mut self, out: &mut Vec<CsNode>) -> bool {
        if !self.at(K::Ident) {
            return false;
        }
        let save = self.pos;
        let mut type_refs = Vec::new();
        let first = self.ident_name().unwrap();
        let is_var = first.text == "var";
        if !is_var {
            type_refs.push(root_ref(first.clone()));
            self.finish_type(&mut type_refs, Some(first.text.clone()));
        }
        // Must now be at an identifier (the variable name) followed by a token
        // that can end a declarator: `=`, `;`, `,`, `)`, or `in`.
        if !self.at(K::Ident) {
            self.pos = save;
            return false;
        }
        let after_name = self.toks.get(self.pos + 1).map(|t| t.kind);
        let ends_decl = matches!(
            after_name,
            Some(K::Assign) | Some(K::Semicolon) | Some(K::Comma) | Some(K::RParen)
        ) || self.toks.get(self.pos + 1).is_some_and(|t| t.is_kw("in"));
        if !ends_decl {
            self.pos = save;
            return false;
        }
        // Commit.
        out.extend(type_refs);
        while let Some(name) = self.ident_name() {
            out.push(name_decl(name));
            if self.eat(K::Assign) {
                self.scan_expr(out);
            }
            if self.at_kw("in") {
                self.advance();
                self.scan_expr(out);
                break;
            }
            if !self.eat(K::Comma) {
                break;
            }
        }
        true
    }

    /// Scans an expression, capturing identifier references and nested blocks
    /// (lambda bodies, initializers), until a `;`, a `,` at the top level, or a
    /// closing `)`/`]`/`}` that is not balanced within.
    ///
    /// `prev_name` tracks the most recent bare identifier so a member access
    /// (`recv.Member`) can record its receiver; it is cleared by anything that
    /// is not an identifier, so `foo().Member` records no receiver.
    fn scan_expr(&mut self, out: &mut Vec<CsNode>) {
        let mut paren = 0i32;
        let mut bracket = 0i32;
        let mut prev_dot = false;
        let mut prev_name: Option<String> = None;
        loop {
            match self.k() {
                K::Eof => return,
                K::Semicolon if paren == 0 && bracket == 0 => return,
                K::Comma if paren == 0 && bracket == 0 => return,
                K::RParen if paren == 0 => return,
                K::RBracket if bracket == 0 => return,
                K::RBrace => return,
                K::LParen => {
                    paren += 1;
                    prev_dot = false;
                    prev_name = None;
                    self.advance();
                }
                K::RParen => {
                    paren -= 1;
                    prev_dot = false;
                    prev_name = None;
                    self.advance();
                }
                K::LBracket => {
                    bracket += 1;
                    prev_dot = false;
                    prev_name = None;
                    self.advance();
                }
                K::RBracket => {
                    bracket -= 1;
                    prev_dot = false;
                    prev_name = None;
                    self.advance();
                }
                K::LBrace => {
                    // Lambda body, object/collection initializer: a nested scope.
                    out.push(self.parse_block());
                    prev_dot = false;
                    prev_name = None;
                }
                K::Dot | K::ColonColon => {
                    // Keep prev_name: it is the receiver of the name that follows.
                    prev_dot = true;
                    self.advance();
                }
                K::Ident => {
                    let after_dot = prev_dot;
                    let span = self.span();
                    let text = self.tok().text.clone();
                    self.advance();
                    if after_dot {
                        out.push(member_ref(
                            CsName {
                                text: text.clone(),
                                span,
                            },
                            prev_name.take(),
                        ));
                    } else if !skip_as_ref(&text) {
                        out.push(root_ref(CsName {
                            text: text.clone(),
                            span,
                        }));
                    }
                    prev_name = Some(text);
                    prev_dot = false;
                }
                _ => {
                    prev_dot = false;
                    prev_name = None;
                    self.advance();
                }
            }
        }
    }

    // type syntax

    /// After the first identifier of a type has been consumed, consume the rest
    /// of the type syntax (`.Qualified`, `<generics>`, `[]`, `?`, `*`) and push
    /// any identifiers found as references. `prev` is the first segment's text,
    /// so a qualified segment records the segment before it as its receiver.
    fn finish_type(&mut self, out: &mut Vec<CsNode>, mut prev: Option<String>) {
        loop {
            match self.k() {
                K::Dot | K::ColonColon => {
                    self.advance();
                    if let Some(name) = self.ident_name() {
                        let recv = prev.take();
                        prev = Some(name.text.clone());
                        out.push(member_ref(name, recv));
                    }
                }
                K::Lt => self.consume_generic_args(out),
                K::LBracket => {
                    // Array rank `[]` / `[,]`.
                    self.advance();
                    while !self.at(K::Eof) && !self.at(K::RBracket) {
                        self.advance();
                    }
                    self.eat(K::RBracket);
                }
                K::Op if self.tok().text == "?" || self.tok().text == "*" => self.advance(),
                _ => break,
            }
        }
    }

    fn consume_generic_args(&mut self, out: &mut Vec<CsNode>) {
        // Balanced `< ... >`, capturing identifiers (which may be our types).
        let mut depth = 0i32;
        loop {
            match self.k() {
                K::Eof => return,
                K::Lt => {
                    depth += 1;
                    self.advance();
                }
                K::Gt => {
                    depth -= 1;
                    self.advance();
                    if depth <= 0 {
                        return;
                    }
                }
                K::Ident => {
                    let name = CsName {
                        text: self.tok().text.clone(),
                        span: self.span(),
                    };
                    self.advance();
                    if !skip_as_ref(&name.text) {
                        out.push(root_ref(name));
                    }
                }
                K::Semicolon | K::LBrace | K::RBrace => return, // safety
                _ => self.advance(),
            }
        }
    }

    fn skip_generic_params(&mut self) {
        if self.at(K::Lt) {
            let mut sink = Vec::new();
            self.consume_generic_args(&mut sink);
        }
    }

    // small helpers

    fn qualified_name(&mut self) -> Option<CsName> {
        let first = self.ident_name()?;
        let mut text = first.text.clone();
        let mut end = first.span.end;
        while self.at(K::Dot) {
            self.advance();
            if let Some(part) = self.ident_name() {
                text.push('.');
                text.push_str(&part.text);
                end = part.span.end;
            } else {
                break;
            }
        }
        Some(CsName {
            text,
            span: Span::new(first.span.start, end),
        })
    }

    fn type_keyword(&self) -> Option<CsTypeKind> {
        if self.at_kw("class") {
            Some(CsTypeKind::Class)
        } else if self.at_kw("struct") {
            Some(CsTypeKind::Struct)
        } else if self.at_kw("interface") {
            Some(CsTypeKind::Interface)
        } else if self.at_kw("enum") {
            Some(CsTypeKind::Enum)
        } else {
            None
        }
    }

    fn is_control_kw(&self) -> bool {
        CONTROL.contains(&self.tok().text.as_str())
    }

    fn skip_modifiers(&mut self) {
        while self.at(K::Ident) && MODIFIERS.contains(&self.tok().text.as_str()) {
            self.advance();
        }
    }

    fn skip_param_modifiers(&mut self) {
        while self.at(K::Ident) && PARAM_MODIFIERS.contains(&self.tok().text.as_str()) {
            self.advance();
        }
    }

    fn parse_attrs(&mut self) -> Vec<CsNode> {
        let mut attrs = Vec::new();
        while self.at(K::LBracket) {
            let start = self.span().start;
            self.advance();
            let name = self.ident_name();
            let mut depth = 1i32;
            while !self.at(K::Eof) && depth > 0 {
                match self.k() {
                    K::LBracket => {
                        depth += 1;
                        self.advance();
                    }
                    K::RBracket => {
                        depth -= 1;
                        self.advance();
                    }
                    _ => self.advance(),
                }
            }
            if let Some(n) = name {
                attrs.push(CsNode::leaf(
                    CsNodeKind::Attribute(n),
                    Span::new(start, self.last_end()),
                ));
            }
        }
        attrs
    }

    fn enter(&mut self) {
        self.depth += 1;
    }
    fn leave(&mut self) {
        self.depth -= 1;
    }
}

/// Identifiers that are keywords or primitives and so are never captured as
/// renameable references when they appear as a root (not after a dot).
fn skip_as_ref(s: &str) -> bool {
    // REF_SKIP is kept sorted (a test enforces it), so this is a binary search.
    REF_SKIP.binary_search(&s).is_ok()
}

/// Builds a `NameRef` leaf, copying the name's span onto the node.
fn name_ref(name: CsName, after_dot: bool, receiver: Option<String>) -> CsNode {
    let span = name.span;
    CsNode::leaf(
        CsNodeKind::NameRef {
            name,
            after_dot,
            receiver,
        },
        span,
    )
}

/// A root reference (not member access).
fn root_ref(name: CsName) -> CsNode {
    name_ref(name, false, None)
}

/// A member-access reference reached through `receiver`.
fn member_ref(name: CsName, receiver: Option<String>) -> CsNode {
    name_ref(name, true, receiver)
}

/// A declared-name leaf (field or local).
fn name_decl(name: CsName) -> CsNode {
    let span = name.span;
    CsNode::leaf(CsNodeKind::NameDecl(name), span)
}

const MODIFIERS: &[&str] = &[
    "public",
    "private",
    "protected",
    "internal",
    "static",
    "readonly",
    "const",
    "virtual",
    "override",
    "abstract",
    "sealed",
    "partial",
    "extern",
    "unsafe",
    "async",
    "new",
    "volatile",
    "fixed",
    "implicit",
    "explicit",
    "required",
    "file",
];

const PARAM_MODIFIERS: &[&str] = &["ref", "out", "in", "params", "this", "scoped", "readonly"];

const CONTROL: &[&str] = &[
    "if",
    "else",
    "for",
    "foreach",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "try",
    "catch",
    "finally",
    "using",
    "lock",
    "fixed",
    "unsafe",
    "checked",
    "unchecked",
    "return",
    "throw",
    "break",
    "continue",
    "yield",
    "goto",
];

const REF_SKIP: &[&str] = &[
    // Keywords, primitives, and contextual words that are never renameable
    // references at a root position. Kept sorted for binary search; a test
    // enforces the ordering.
    "abstract",
    "add",
    "as",
    "async",
    "await",
    "base",
    "bool",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "checked",
    "class",
    "const",
    "continue",
    "decimal",
    "default",
    "delegate",
    "do",
    "double",
    "dynamic",
    "else",
    "enum",
    "event",
    "explicit",
    "extern",
    "false",
    "finally",
    "float",
    "for",
    "foreach",
    "from",
    "get",
    "global",
    "goto",
    "if",
    "implicit",
    "in",
    "int",
    "interface",
    "internal",
    "is",
    "lock",
    "long",
    "nameof",
    "namespace",
    "new",
    "nint",
    "nuint",
    "null",
    "object",
    "operator",
    "out",
    "override",
    "params",
    "partial",
    "private",
    "protected",
    "public",
    "readonly",
    "ref",
    "remove",
    "return",
    "sbyte",
    "sealed",
    "select",
    "set",
    "short",
    "sizeof",
    "stackalloc",
    "static",
    "string",
    "struct",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "uint",
    "ulong",
    "unchecked",
    "unsafe",
    "ushort",
    "using",
    "value",
    "var",
    "virtual",
    "void",
    "volatile",
    "when",
    "where",
    "while",
    "yield",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::csharp::ast::CsNodeKind;

    fn unit(code: &str) -> CsUnit {
        cs_parse(code, 0)
    }

    /// Collect (kind-label, name-text) for declared symbols.
    fn decls(u: &CsUnit) -> Vec<(String, String)> {
        let mut v = Vec::new();
        u.root.walk(&mut |n| {
            let label = match &n.kind {
                CsNodeKind::Type { kind, .. } => Some(format!("{kind:?}")),
                CsNodeKind::Method(_) => Some("Method".into()),
                CsNodeKind::Ctor(_) => Some("Ctor".into()),
                CsNodeKind::Property(_) => Some("Property".into()),
                CsNodeKind::Param(_) => Some("Param".into()),
                CsNodeKind::NameDecl(_) => Some("Field/Local".into()),
                CsNodeKind::EnumMember(_) => Some("EnumMember".into()),
                _ => None,
            };
            if let (Some(l), Some(name)) = (label, n.declared_name()) {
                v.push((l, name.text.clone()));
            }
        });
        v
    }

    #[test]
    fn parses_class_with_pinvoke_method() {
        let u = unit(
            "public class Win32 {\n  [DllImport(\"user32.dll\")]\n  public static extern int MessageBox(IntPtr h, string text, string caption, uint type);\n}\n",
        );
        assert!(u.errors.is_empty(), "errors: {:?}", u.errors);
        let d = decls(&u);
        assert!(d.contains(&("Class".into(), "Win32".into())));
        assert!(d.contains(&("Method".into(), "MessageBox".into())));
        assert!(d.contains(&("Param".into(), "text".into())));
    }

    #[test]
    fn declared_name_spans_are_exact() {
        let src = "class Foo { public int Bar() { return 0; } }";
        let u = unit(src);
        let mut method_span = None;
        u.root.walk(&mut |n| {
            if let CsNodeKind::Method(name) = &n.kind {
                method_span = Some(name.span);
            }
        });
        assert_eq!(method_span.map(|s| s.slice(src)), Some("Bar"));
    }

    #[test]
    fn fields_locals_and_refs_captured() {
        let src = "class C { int field; void M() { int local = field; var z = Helper(local); } }";
        let u = unit(src);
        let d = decls(&u);
        assert!(d.contains(&("Field/Local".into(), "field".into())));
        assert!(d.contains(&("Field/Local".into(), "local".into())));
        assert!(d.contains(&("Field/Local".into(), "z".into())));
        // `Helper` is a root reference; `field` and `local` are referenced too.
        let mut refs = Vec::new();
        u.root.walk(&mut |n| {
            if let CsNodeKind::NameRef { name, .. } = &n.kind {
                refs.push(name.text.clone());
            }
        });
        assert!(refs.contains(&"Helper".to_string()));
        assert!(refs.contains(&"field".to_string()));
    }

    #[test]
    fn enum_members_parsed() {
        let u = unit("public enum Color { Red, Green = 2, Blue }");
        let d = decls(&u);
        assert!(d.contains(&("EnumMember".into(), "Red".into())));
        assert!(d.contains(&("EnumMember".into(), "Green".into())));
        assert!(d.contains(&("EnumMember".into(), "Blue".into())));
    }

    #[test]
    fn recovers_from_garbage_without_error_explosion() {
        let u = unit("class A { @@@ ### void Ok() {} }");
        // It still finds the well-formed method.
        let d = decls(&u);
        assert!(d.contains(&("Method".into(), "Ok".into())));
    }

    #[test]
    fn ref_skip_is_sorted_for_binary_search() {
        // skip_as_ref binary-searches REF_SKIP, so it must stay strictly sorted
        // (which also rules out duplicates).
        assert!(
            super::REF_SKIP.windows(2).all(|w| w[0] < w[1]),
            "REF_SKIP must be strictly sorted"
        );
    }

    #[test]
    fn fuzz_parser_never_panics() {
        // Deterministic pseudo-random C#-ish soup must never panic or hang.
        let mut state = 0x9e3779b97f4a7c15u64;
        let alphabet: &[u8] = b"class struct{}()[]<>;,.:=>@\"'/ * abcXYZ_0 namespace public static int void return new ";
        for _ in 0..2000 {
            let mut s = String::new();
            let len = {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 120) as usize
            };
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                s.push(alphabet[(state as usize) % alphabet.len()] as char);
            }
            let u = cs_parse(&s, 0);
            // Spans must stay within bounds.
            u.root.walk(&mut |n| {
                assert!(n.span.start <= n.span.end);
                assert!(n.span.end <= s.len());
            });
        }
    }
}
