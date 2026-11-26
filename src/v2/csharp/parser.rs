//! C# parser for the Add-Type dialect.
//!
//! Recursive descent over [`cs_lex`] output models supported declarations,
//! parameters, and scopes for the resolver. Statement bodies capture local
//! declarations and identifier references without a full expression grammar.
//! This is a syntactic subset, not full C# parsing or semantic validation.
//!
//! Type declarations, members, real blocks, and synthetic control-flow scopes
//! share one depth counter. Each guarded entry increments it before checking
//! the private threshold of 200; reaching 200 triggers recovery, not acceptance
//! of 200 nested constructs. Mixed nesting consumes the same budget, including
//! brace-free control chains. It is not a separate allowance for each kind.
//!
//! At the threshold, types, members, and controls emit
//! [`Error`](CsNodeKind::Error) nodes and error spans. A rejected type's span
//! starts before its attributes/modifiers. Real blocks retain the existing
//! behavior: skip their balanced tokens into an empty block without an error.
//! Recovery scans tokens iteratively to a synchronization point or EOF; literal
//! and comment braces do not count. It can leave an enclosing `}` for its
//! caller. Enclosing parse loops enforce progress if a rule consumes nothing,
//! so recovery need not consume a token itself at every boundary. Following
//! declarations can survive recovery; arbitrary malformed input need not retain
//! every declaration.
//!
//! These guards bound parser-produced tree nesting, including for recursive
//! walking and ordinary drop. They do not bound input size or guarantee
//! stack-safe walking/dropping of arbitrarily deep, manually built [`CsNode`]s;
//! the recursive tree representation is unchanged.

use super::ast::{CsName, CsNode, CsNodeKind, CsTypeKind, CsUnit};
use super::lexer::{cs_lex, lex_checked};
use super::local::bases::{BaseClause, BaseOccurrence, BasePath};
use super::local::binding::{Receiver, ReferenceRole, ReferenceSite, SiteOwner};
use super::local::{
    BlockOccurrence, CallableHeaderRow, CallableKind, CallableParameter, CallableParameterType,
    CallableParameterTypeUse, FieldOccurrence, GenericCallableIndexError, LexicalDeclarations,
    LocalOccurrence, MemberModifiers, NamespaceOccurrence, PendingCallableParameter,
    PendingQualifiedCallableIndex, QualifiedTypeUse, TypeHeaderIndex, TypeHeaderRow,
};
#[cfg(test)]
use super::local::{CallableHeaderIndex, TypeIndexError};
use super::tokens::{CsToken, CsTokenKind as K};
use crate::v2::span::Span;
#[cfg(test)]
use std::collections::HashMap;
use std::collections::HashSet;

const MAX_DEPTH: usize = 200;

/// Parses a C# source region into a [`CsUnit`].
///
/// `base` is the absolute offset of `code` in the original file; every node
/// span indexes the file directly.
///
/// Returns supported syntax and recovery nodes, with error spans in
/// [`CsUnit::errors`], not a C# validity verdict. Types, members, blocks, and
/// control scopes share the post-entry depth threshold described in the
/// [module documentation](self). At the threshold, a real block is skipped
/// without a diagnostic; types, members, and controls instead produce errors.
pub fn cs_parse(code: &str, base: usize) -> CsUnit {
    cs_parse_with_base_observer(code, base, None)
}

/// Reports the declaring type's name span if present, the original significant
/// tokens for one base entry, and its terminal name span (`None` for
/// unsupported syntax). Generic argument contents are opaque, and the
/// legacy projection may still keep a name from an unsupported entry.
type BaseEntryObserver<'a> = &'a mut dyn FnMut(Option<Span>, &[CsToken], Option<Span>);

fn cs_parse_with_base_observer(
    code: &str,
    base: usize,
    observe_base_entry: Option<BaseEntryObserver<'_>>,
) -> CsUnit {
    CsParser::new(code, base, observe_base_entry).parse_unit()
}

/// Collects headers during the legacy parse of an already validated region.
#[cfg(test)]
pub(super) fn parse_headers(
    code: &str,
    region: Span,
    callables: Option<&mut Vec<CallableHeaderRow>>,
) -> Result<TypeHeaderIndex, TypeIndexError> {
    let (index, recovery, _) =
        parse_indexed_headers(code, region, callables.map(IndexedCallables::Simple));
    // The legacy parser decides whether recovery occurred, even after a subset refusal.
    if recovery.is_empty() {
        index.map_err(TypeIndexError::UnsupportedSyntax)
    } else {
        Err(TypeIndexError::Recovered(recovery))
    }
}

/// Collects headers with generic methods enabled for this parse only.
#[cfg(test)]
pub(super) fn parse_generic_headers(
    code: &str,
    region: Span,
) -> Result<CallableHeaderIndex, GenericCallableIndexError> {
    let mut callables = Vec::new();
    let (index, recovery, _) = parse_indexed_headers(
        code,
        region,
        Some(IndexedCallables::Generic(&mut callables)),
    );
    match index {
        Ok(types) if recovery.is_empty() => Ok(CallableHeaderIndex { types, callables }),
        index => Err(GenericCallableIndexError::Rejected {
            first_unsupported: index.err(),
            recovery,
        }),
    }
}

/// Returns pending evidence only from a complete, recovery-free parse.
#[cfg(test)]
pub(super) fn parse_qualified_headers(
    code: &str,
    region: Span,
) -> Result<PendingQualifiedCallableIndex, GenericCallableIndexError> {
    parse_qualified_mode(code, region, None)
}

/// Collects lexical declarations alongside pending callable headers.
#[cfg(test)]
pub(super) fn parse_lexical_headers(
    code: &str,
    region: Span,
) -> Result<(PendingQualifiedCallableIndex, LexicalDeclarations), GenericCallableIndexError> {
    let mut declarations = LexicalDeclarations::default();
    let headers = parse_qualified_mode(
        code,
        region,
        Some(LexicalObserver {
            declarations: &mut declarations,
            body: None,
            sites: None,
        }),
    )?;
    Ok((headers, declarations))
}

/// Observes assignments in the same parse that records lexical declarations.
#[cfg(test)]
pub(super) fn parse_binding_headers(
    code: &str,
    region: Span,
    declarations: &mut LexicalDeclarations,
    sites: &mut Vec<ReferenceSite>,
) -> Result<PendingQualifiedCallableIndex, GenericCallableIndexError> {
    parse_observed_headers(code, region, declarations, ObservedSites::Binding(sites))
}

/// Requires coverage of the checked subset. This does not check rename eligibility.
/// `code` is the original slice of the already validated `region`.
pub(super) fn parse_checked_headers(
    code: &str,
    region: Span,
    declarations: &mut LexicalDeclarations,
    sites: &mut Vec<ReferenceSite>,
    bases: &mut Vec<BaseOccurrence>,
    mode: CheckedHeaderMode,
) -> Result<PendingQualifiedCallableIndex, GenericCallableIndexError> {
    parse_observed_headers(
        code,
        region,
        declarations,
        ObservedSites::Checked { sites, bases, mode },
    )
}

fn parse_observed_headers(
    code: &str,
    region: Span,
    declarations: &mut LexicalDeclarations,
    sites: ObservedSites<'_>,
) -> Result<PendingQualifiedCallableIndex, GenericCallableIndexError> {
    parse_qualified_mode(
        code,
        region,
        Some(LexicalObserver {
            declarations,
            body: None,
            sites: Some(sites),
        }),
    )
}

fn parse_qualified_mode(
    code: &str,
    region: Span,
    lexical: Option<LexicalObserver<'_>>,
) -> Result<PendingQualifiedCallableIndex, GenericCallableIndexError> {
    let mut callables = Vec::new();
    let (index, recovery, depth_limit) = parse_indexed_headers(
        code,
        region,
        Some(IndexedCallables::Qualified {
            rows: &mut callables,
            lexical,
        }),
    );
    if let Some(at) = depth_limit.filter(|at| {
        index
            .as_ref()
            .err()
            .is_none_or(|first| at.start <= first.start)
    }) {
        return Err(GenericCallableIndexError::DepthLimit { at, recovery });
    }
    match index {
        Ok(types) if recovery.is_empty() => Ok(PendingQualifiedCallableIndex { types, callables }),
        index => Err(GenericCallableIndexError::Rejected {
            first_unsupported: index.err(),
            recovery,
        }),
    }
}

enum IndexedCallables<'a> {
    #[cfg(test)]
    Simple(&'a mut Vec<CallableHeaderRow>),
    #[cfg(test)]
    Generic(&'a mut Vec<CallableHeaderRow>),
    Qualified {
        rows: &'a mut Vec<CallableHeaderRow<PendingCallableParameter>>,
        lexical: Option<LexicalObserver<'a>>,
    },
}

struct LexicalObserver<'a> {
    declarations: &'a mut LexicalDeclarations,
    body: Option<IndexedCallableBody>,
    /// Only site observers accept and record assignments.
    sites: Option<ObservedSites<'a>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CheckedHeaderMode {
    Rename,
    #[cfg(test)]
    GenericEvidence,
}

enum ObservedSites<'a> {
    #[cfg(test)]
    Binding(&'a mut Vec<ReferenceSite>),
    Checked {
        sites: &'a mut Vec<ReferenceSite>,
        bases: &'a mut Vec<BaseOccurrence>,
        mode: CheckedHeaderMode,
    },
}

impl ObservedSites<'_> {
    fn sites(&mut self) -> &mut Vec<ReferenceSite> {
        match self {
            #[cfg(test)]
            Self::Binding(sites) => sites,
            Self::Checked { sites, .. } => sites,
        }
    }
}

struct ActiveLexicalBody<'a> {
    declarations: &'a mut LexicalDeclarations,
    body: &'a mut IndexedCallableBody,
    rows: &'a mut [CallableHeaderRow<PendingCallableParameter>],
    sites: Option<&'a mut Vec<ReferenceSite>>,
}

struct IndexedCallableBody {
    callable: usize,
    /// Offset of this body's first row in the flat block list.
    first_block: usize,
    /// Index into the flat block list, not the per-callable ordinal.
    current_block: Option<usize>,
}

fn parse_indexed_headers(
    code: &str,
    region: Span,
    callables: Option<IndexedCallables<'_>>,
) -> (Result<TypeHeaderIndex, Span>, Vec<Span>, Option<Span>) {
    let mut index = Ok(TypeHeaderIndex {
        region,
        namespaces: Vec::new(),
        rows: Vec::new(),
    });
    let checked = match &callables {
        Some(IndexedCallables::Qualified {
            lexical:
                Some(LexicalObserver {
                    sites: Some(ObservedSites::Checked { mode, .. }),
                    ..
                }),
            ..
        }) => Some(*mode),
        _ => None,
    };
    let mut depth_limit = None;
    let mut parser = if let Some(mode) = checked {
        let (tokens, lexical_refusal) = lex_checked(code, region.start);
        let coverage = CheckedCoverage {
            roles: vec![None; tokens.len()],
            lexical_refusal,
            depth_limit: (mode == CheckedHeaderMode::Rename).then_some(&mut depth_limit),
        };
        let mut parser = CsParser::from_tokens(tokens, None);
        parser.checked_coverage = Some(coverage);
        parser
    } else {
        CsParser::new(code, region.start, None)
    };
    parser.type_index = Some(&mut index);
    parser.index_callables = callables;
    let bom = '\u{feff}';
    if code.starts_with(bom) {
        // The lexer may include a BOM in an identifier. Refuse it without
        // altering the legacy token stream or its diagnostics.
        parser.reject_type_index(Span::new(region.start, region.start + bom.len_utf8()));
    }
    let unit = parser.parse_unit();
    (index, unit.errors, depth_limit)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoverageRole {
    Syntax,
    Declaration,
    Modifier,
    TypeNamespaceOrBinder,
    ExpressionSite,
    QualifiedRoot,
    QualifiedPrefix,
    QualifiedTerminal,
}

/// Only admitted syntax and exact evidence rows can claim significant tokens.
/// Cursor movement alone (including legacy skips and recovery) claims nothing.
struct CheckedCoverage<'a> {
    roles: Vec<Option<CoverageRole>>,
    lexical_refusal: Option<Span>,
    depth_limit: Option<&'a mut Option<Span>>,
}

impl CheckedCoverage<'_> {
    fn cover_punctuation(&mut self, tokens: &[CsToken], start: usize, end: usize, kinds: &[K]) {
        for (token, role) in tokens[start..end].iter().zip(&mut self.roles[start..end]) {
            if kinds.contains(&token.kind) {
                *role = Some(CoverageRole::Syntax);
            }
        }
    }

    fn first_uncovered(
        &mut self,
        tokens: &[CsToken],
        types: &TypeHeaderIndex,
        callables: &[CallableHeaderRow<PendingCallableParameter>],
        declarations: &LexicalDeclarations,
        sites: &[ReferenceSite],
        bases: &[BaseOccurrence],
    ) -> Option<Span> {
        use CoverageRole::{Declaration, ExpressionSite, TypeNamespaceOrBinder as HeaderName};

        // Claim names from the output evidence, not from speculative lookahead
        // or the legacy AST, so a dropped row leaves an uncovered token.
        let mut claim_error = None;
        let mut claim = |name: &CsName, role| {
            let pos = tokens
                .binary_search_by_key(&name.span.start, |token| token.span.start)
                .ok();
            if let Some(pos) = pos.filter(|&pos| {
                let token = &tokens[pos];
                token.kind == K::Ident && token.span == name.span && token.text == name.text
            }) {
                if self.roles[pos].is_none() {
                    self.roles[pos] = Some(role);
                    return;
                }
            }
            claim_error = earlier_span(claim_error, Some(name.span));
        };

        for namespace in &types.namespaces {
            for segment in &namespace.segments {
                claim(segment, HeaderName);
            }
        }
        for row in &types.rows {
            claim(&row.name, Declaration);
            if let Some(modifier) = &row.public_modifier {
                claim(modifier, CoverageRole::Modifier);
            }
            for binder in &row.type_params {
                claim(binder, HeaderName);
            }
        }
        for base in bases {
            match &base.clause.path {
                BasePath::Bare(name) => claim(name, HeaderName),
                BasePath::Rooted { path, .. } => {
                    claim(&path.root, CoverageRole::QualifiedRoot);
                    for segment in &path.segments {
                        claim(segment, HeaderName);
                    }
                }
            }
        }
        for row in callables {
            claim(&row.name, Declaration);
            for modifier in row.modifiers.tokens() {
                claim(modifier, CoverageRole::Modifier);
            }
            if let Some(ty) = &row.return_type {
                claim(ty, HeaderName);
            }
            #[cfg(test)]
            for binder in &row.type_params {
                claim(binder, HeaderName);
            }
            for parameter in &row.parameters {
                match parameter {
                    PendingCallableParameter::Simple(parameter) => {
                        if let CallableParameterTypeUse::Simple(ty) = &parameter.type_use {
                            claim(ty, HeaderName);
                        }
                        claim(&parameter.name, Declaration);
                    }
                    PendingCallableParameter::Qualified { type_use, name } => {
                        claim(&type_use.root, HeaderName);
                        for segment in &type_use.segments {
                            claim(segment, HeaderName);
                        }
                        claim(name, Declaration);
                    }
                }
            }
        }
        for (ty, name) in declarations
            .fields
            .iter()
            .map(|row| (&row.type_use, &row.name))
            .chain(
                declarations
                    .locals
                    .iter()
                    .map(|row| (&row.type_use, &row.name)),
            )
        {
            claim(ty, HeaderName);
            claim(name, Declaration);
        }
        for field in &declarations.fields {
            for modifier in field.modifiers.tokens() {
                claim(modifier, CoverageRole::Modifier);
            }
        }
        for site in sites {
            claim(&site.name, ExpressionSite);
            // Named receivers already have their own site; `this` and `base`
            // are carried on the selected site rather than emitted separately.
            match &site.receiver {
                Receiver::This(receiver) | Receiver::Base(receiver) => {
                    claim(receiver, ExpressionSite)
                }
                Receiver::Rooted(path) => {
                    claim(&path.root, CoverageRole::QualifiedRoot);
                    let (terminal, prefixes) =
                        path.segments.split_last().expect("validated receiver path");
                    for segment in prefixes {
                        claim(segment, CoverageRole::QualifiedPrefix);
                    }
                    claim(terminal, CoverageRole::QualifiedTerminal);
                }
                _ => {}
            }
        }
        let unclaimed = tokens
            .iter()
            .zip(&self.roles)
            .find(|(token, role)| token.kind != K::Eof && role.is_none())
            .map(|(token, _)| token.span);
        earlier_span(claim_error, unclaimed)
    }
}

fn earlier_span(first: Option<Span>, other: Option<Span>) -> Option<Span> {
    match (first, other) {
        (Some(left), Some(right)) if right.start < left.start => Some(right),
        (None, other) => other,
        (first, _) => first,
    }
}

struct ValidatedCallableHeader<Parameter> {
    row: CallableHeaderRow<Parameter>,
    /// Token offset of `(` from the member start, only for a validated generic method.
    #[cfg(test)]
    generic_open_paren_offset: Option<usize>,
}

/// Validated token evidence passed only to the parse of this member.
enum IndexedMemberHeader {
    Field(FieldOccurrence),
    #[cfg(test)]
    MethodTypeParams {
        open_paren_pos: usize,
    },
}

enum IndexedStatement {
    Local(CsName, CsName),
    Assignment(StatementPlace, StatementPlace),
    Call(StatementPlace, Option<CsName>),
}

struct StatementPlace {
    name: CsName,
    receiver: Receiver,
}

impl StatementPlace {
    fn record(
        self,
        sites: &mut Vec<ReferenceSite>,
        owner: SiteOwner,
        role: ReferenceRole,
        call_argument: Option<Span>,
    ) {
        let mut push = |name, role, receiver, call_argument| {
            sites.push(ReferenceSite {
                ordinal: sites.len(),
                owner,
                name,
                role,
                receiver,
                call_argument,
            });
        };
        if let Receiver::Name(name) = &self.receiver {
            let receiver_role = if role == ReferenceRole::Call {
                ReferenceRole::Read
            } else {
                ReferenceRole::ReceiverName
            };
            push(name.clone(), receiver_role, Receiver::Bare, None);
        }
        push(self.name, role, self.receiver, call_argument);
    }
}

/// Read-only lookahead over the parser's significant tokens, ending in its EOF token.
struct CallableHeaderTokens<'a>(&'a [CsToken]);

impl<'a> CallableHeaderTokens<'a> {
    fn at(&self, kind: K) -> bool {
        self.0[0].kind == kind
    }

    fn take(&mut self, kind: K) -> Result<&'a CsToken, Span> {
        let token = &self.0[0];
        if token.kind == kind && kind != K::Eof {
            self.0 = &self.0[1..];
            Ok(token)
        } else {
            Err(token.span)
        }
    }

    fn name(&mut self) -> Result<CsName, Span> {
        self.take(K::Ident).map(Self::name_evidence)
    }

    fn name_evidence(token: &CsToken) -> CsName {
        CsName {
            text: token.text.clone(),
            span: token.span,
        }
    }

    fn uninitialized_primitive(&mut self) -> Result<(CsName, CsName), Span> {
        let primitive = self.take(K::Ident)?;
        if !matches!(primitive.text.as_str(), "int" | "bool") {
            return Err(primitive.span);
        }
        let name = self.take(K::Ident)?;
        self.take(K::Semicolon)?;
        Ok((Self::name_evidence(primitive), Self::name_evidence(name)))
    }

    fn place_statement(
        &mut self,
        checked: bool,
        class_base: bool,
    ) -> Result<IndexedStatement, Span> {
        let target = self.place(checked, class_base)?;
        let statement = if checked && self.at(K::LParen) {
            self.take(K::LParen)?;
            let argument = if self.at(K::RParen) {
                None
            } else {
                Some(self.take_reference_name()?)
            };
            self.take(K::RParen)?;
            IndexedStatement::Call(target, argument)
        } else {
            self.take(K::Assign)?;
            let read = self.place(checked, class_base)?;
            IndexedStatement::Assignment(target, read)
        };
        self.take(K::Semicolon)?;
        Ok(statement)
    }

    fn place(&mut self, checked: bool, class_base: bool) -> Result<StatementPlace, Span> {
        if checked && self.has_global_root() {
            let (root, mut segments) = self.rooted_names(Self::take_reference_name)?;
            if segments.len() < 2 {
                return Err(self.0[0].span);
            }
            let name = segments.pop().expect("validated path includes a member");
            return Ok(StatementPlace {
                name,
                receiver: Receiver::Rooted(QualifiedTypeUse::new(root, segments)),
            });
        }
        let first = self.take(K::Ident)?;
        let dotted = self.at(K::Dot);
        let is_base = class_base && dotted && first.is_kw("base");
        let first_name = if is_base || (dotted && first.is_kw("this")) {
            Self::name_evidence(first)
        } else {
            Self::reference_name(first)?
        };
        if !dotted {
            return Ok(StatementPlace {
                name: first_name,
                receiver: Receiver::Bare,
            });
        }
        self.take(K::Dot)?;
        Ok(StatementPlace {
            name: self.take_reference_name()?,
            receiver: if first.is_kw("this") {
                Receiver::This(first_name)
            } else if is_base {
                Receiver::Base(first_name)
            } else {
                Receiver::Name(first_name)
            },
        })
    }

    fn has_global_root(&self) -> bool {
        matches!(self.0, [root, separator, ..] if root.is_kw("global") && separator.kind == K::ColonColon)
    }

    fn rooted_names(
        &mut self,
        segment: fn(&mut Self) -> Result<CsName, Span>,
    ) -> Result<(CsName, Vec<CsName>), Span> {
        let root = self.name()?;
        self.take(K::ColonColon)?;
        let mut segments = vec![segment(self)?];
        while self.at(K::Dot) {
            self.take(K::Dot)?;
            segments.push(segment(self)?);
        }
        Ok((root, segments))
    }

    fn take_reference_name(&mut self) -> Result<CsName, Span> {
        Self::reference_name(self.take(K::Ident)?)
    }

    fn reference_name(token: &CsToken) -> Result<CsName, Span> {
        let text = token.text.as_str();
        if skip_as_ref(text)
            || CONTROL.contains(&text)
            || MODIFIERS.contains(&text)
            || PARAM_MODIFIERS.contains(&text)
        {
            return Err(token.span);
        }
        Ok(Self::name_evidence(token))
    }

    fn simple_parameter(
        &mut self,
        #[cfg(test)] method_slots: &HashMap<&'a str, usize>,
    ) -> Result<CallableParameter, Span> {
        let type_name = self.name()?;
        let comparison_type = CallableParameterType::primitive(&type_name.text);
        #[cfg(test)]
        let comparison_type = comparison_type.or_else(|| {
            method_slots
                .get(type_name.text.as_str())
                .copied()
                .map(CallableParameterType::MethodSlot)
        });
        let comparison_type = comparison_type.ok_or(type_name.span)?;
        let name = self.name()?;
        Ok(CallableParameter {
            type_use: CallableParameterTypeUse::Simple(type_name),
            comparison_type,
            name,
        })
    }

    fn qualified_parameter(
        &mut self,
        #[cfg(test)] method_slots: &HashMap<&'a str, usize>,
    ) -> Result<PendingCallableParameter, Span> {
        #[cfg(test)]
        if !method_slots.is_empty() {
            return self
                .simple_parameter(method_slots)
                .map(PendingCallableParameter::Simple);
        }
        if !self.has_global_root() {
            return self
                .simple_parameter(
                    #[cfg(test)]
                    method_slots,
                )
                .map(PendingCallableParameter::Simple);
        }
        let (root, segments) = self.rooted_names(Self::name)?;
        let name = self.name()?;
        Ok(PendingCallableParameter::Qualified {
            type_use: QualifiedTypeUse::new(root, segments),
            name,
        })
    }

    #[cfg(test)]
    fn method_type_params(
        &mut self,
        slots: &mut HashMap<&'a str, usize>,
    ) -> Result<Vec<CsName>, Span> {
        self.take(K::Lt)?;
        let mut names = Vec::new();
        loop {
            let token = self.take(K::Ident)?;
            let text = token.text.as_str();
            if !text.is_ascii()
                || text.starts_with('@')
                || skip_as_ref(text)
                || MODIFIERS.contains(&text)
                || PARAM_MODIFIERS.contains(&text)
                || slots.insert(text, names.len()).is_some()
            {
                return Err(token.span);
            }
            names.push(CsName {
                text: token.text.clone(),
                span: token.span,
            });
            if !self.at(K::Comma) {
                break;
            }
            self.take(K::Comma)?;
        }
        self.take(K::Gt)?;
        Ok(names)
    }
}

#[derive(Clone, Copy, Default)]
struct BaseEntryDelimiters {
    angles: usize,
    brackets: usize,
    parens: usize,
}

impl BaseEntryDelimiters {
    fn has_open_after(&mut self, tokens: &[CsToken]) -> bool {
        for token in tokens {
            match token.kind {
                K::Lt => self.angles += 1,
                K::Gt => self.angles = self.angles.saturating_sub(1),
                // The base parser treats generic argument contents as opaque.
                _ if self.angles != 0 => {}
                K::LBracket => self.brackets += 1,
                K::RBracket => self.brackets = self.brackets.saturating_sub(1),
                K::LParen => self.parens += 1,
                K::RParen => self.parens = self.parens.saturating_sub(1),
                _ => {}
            }
        }
        self.angles != 0 || self.brackets != 0 || self.parens != 0
    }
}

struct CsParser<'a> {
    toks: Vec<CsToken>,
    pos: usize,
    depth: usize,
    errors: Vec<Span>,
    observe_base_entry: Option<BaseEntryObserver<'a>>,
    type_index: Option<&'a mut Result<TypeHeaderIndex, Span>>,
    index_callables: Option<IndexedCallables<'a>>,
    index_namespace: Option<usize>,
    index_containing_type: Option<usize>,
    checked_coverage: Option<CheckedCoverage<'a>>,
}

impl<'a> CsParser<'a> {
    fn new(code: &str, base: usize, observe_base_entry: Option<BaseEntryObserver<'a>>) -> Self {
        Self::from_tokens(cs_lex(code, base), observe_base_entry)
    }

    fn from_tokens(toks: Vec<CsToken>, observe_base_entry: Option<BaseEntryObserver<'a>>) -> Self {
        Self {
            toks,
            pos: 0,
            depth: 0,
            errors: Vec::new(),
            observe_base_entry,
            type_index: None,
            index_callables: None,
            index_namespace: None,
            index_containing_type: None,
            checked_coverage: None,
        }
    }

    fn parse_unit(mut self) -> CsUnit {
        let start = self.span().start;
        let mut children = Vec::new();
        while !self.at(K::Eof) {
            let before = self.pos;
            if let Some(node) = self.top_item() {
                children.push(node);
            } else {
                children.push(self.recover());
            }
            if self.pos == before {
                // No rule consumed a token; force progress so we always terminate.
                children.push(self.recover_one());
            }
        }
        self.finish_checked_coverage();
        let end = self.span().end;
        let root = CsNode::new(CsNodeKind::Unit, Span::new(start, end), children);
        CsUnit {
            root,
            errors: self.errors,
        }
    }

    fn reject_type_index(&mut self, span: Span) {
        if let Some(index @ Ok(_)) = self.type_index.as_deref_mut() {
            // Keep the first refusal and discard any earlier rows.
            *index = Err(span);
        }
    }

    fn finish_checked_coverage(&mut self) {
        let (
            Some(coverage),
            Some(index),
            Some(IndexedCallables::Qualified {
                rows,
                lexical:
                    Some(LexicalObserver {
                        declarations,
                        sites: Some(ObservedSites::Checked { sites, bases, .. }),
                        ..
                    }),
            }),
        ) = (
            &mut self.checked_coverage,
            self.type_index.as_deref_mut(),
            &self.index_callables,
        )
        else {
            return;
        };
        let syntax_refusal = match index {
            // Only a clean traversal can distinguish a coverage gap from an
            // ordinary token left unclaimed by recovery or a parser refusal.
            Ok(types) if self.errors.is_empty() => {
                coverage.first_uncovered(&self.toks, types, rows, declarations, sites, bases)
            }
            Ok(_) => None, // Keep the parser's ordered recovery channel.
            Err(span) => Some(*span),
        };
        // Merge independent lexer refusals after traversal so observation
        // (including the generic-header bridge) can preserve recovery. At the
        // same start the parser's specific span wins, e.g. the region-start BOM.
        if let Some(span) = earlier_span(syntax_refusal, coverage.lexical_refusal) {
            *index = Err(span);
        }
    }

    fn cover_syntax_token(&mut self, pos: usize) {
        if let Some(coverage) = &mut self.checked_coverage {
            coverage.roles[pos] = Some(CoverageRole::Syntax);
        }
    }

    /// Claims punctuation only in a construct already validated by the observer.
    fn cover_punctuation(&mut self, start: usize, end: usize, kinds: &[K]) {
        if let Some(coverage) = &mut self.checked_coverage {
            coverage.cover_punctuation(&self.toks, start, end, kinds);
        }
    }

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
        self.recover_from(self.span().start)
    }

    fn recover_from(&mut self, start: usize) -> CsNode {
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
        let item_start = self.pos;
        if self.at(K::LBracket) {
            self.reject_type_index(self.span());
        }
        let mut attrs = self.parse_attrs();
        self.skip_modifiers();
        if self.type_index.is_some() && !self.at_kw("namespace") && self.type_keyword().is_none() {
            self.reject_type_index(self.span());
        }
        if self.at_kw("using") {
            return Some(self.parse_using(start));
        }
        if self.at_kw("namespace") {
            if self.pos != item_start {
                self.reject_type_index(self.toks[item_start].span);
            }
            return Some(self.parse_namespace(start, attrs));
        }
        if let Some(kind) = self.type_keyword() {
            return Some(self.parse_type(start, kind, &mut attrs));
        }
        // A bare member with no enclosing type: `Add-Type -MemberDefinition`
        // fragments are class members that PowerShell wraps in a class, so a
        // top-level method/field/property is valid here and parses like one.
        self.parse_member(start, attrs, None)
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
        let keyword = self.pos;
        self.advance(); // namespace
        let mut segments = Vec::new();
        let collect = matches!(self.type_index.as_deref(), Some(Ok(_)));
        let name = self.qualified_name(collect.then_some(&mut segments));
        let name = name.unwrap_or(CsName {
            text: String::new(),
            span: self.span(),
        });
        let previous_namespace = self.index_namespace;
        self.enter_index_namespace(keyword, segments);

        let mut children = std::mem::take(&mut attrs);
        if self.at(K::Semicolon) {
            // File-scoped namespace: members follow at top level.
            self.cover_syntax_token(self.pos);
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
            self.cover_syntax_token(self.pos - 1);
            self.parse_members_into(&mut children);
            let _closed = self.eat(K::RBrace);
            if _closed {
                self.cover_syntax_token(self.pos - 1);
            } else {
                self.reject_type_index(self.span());
            }
        }
        self.index_namespace = previous_namespace;
        CsNode::new(
            CsNodeKind::Namespace(name),
            Span::new(start, self.last_end()),
            children,
        )
    }

    /// Only root dispatch enters namespaces; member dispatch refuses their keyword.
    fn enter_index_namespace(&mut self, keyword: usize, segments: Vec<CsName>) {
        let file_scoped = match self.k() {
            K::Semicolon => true,
            K::LBrace => false,
            _ => {
                self.reject_type_index(self.span());
                return;
            }
        };
        if let Some(Ok(index)) = self.type_index.as_deref_mut() {
            // File scopes are exclusive, even after an empty block namespace.
            if file_scoped && (!index.namespaces.is_empty() || !index.rows.is_empty()) {
                self.reject_type_index(self.toks[keyword].span);
                return;
            }
            let ordinal = index.namespaces.len();
            index
                .namespaces
                .push(NamespaceOccurrence { ordinal, segments });
            self.index_namespace = Some(ordinal);
            self.cover_syntax_token(keyword);
        }
    }

    fn parse_type(&mut self, start: usize, kind: CsTypeKind, attrs: &mut Vec<CsNode>) -> CsNode {
        self.enter();
        let node = self.parse_type_inner(start, kind, attrs);
        self.leave();
        node
    }

    fn parse_type_inner(
        &mut self,
        start: usize,
        kind: CsTypeKind,
        attrs: &mut Vec<CsNode>,
    ) -> CsNode {
        if self.depth >= MAX_DEPTH {
            // Keep the declaration's attributes/modifiers in the error span,
            // and consume only its own balanced body without recursing.
            return self.recover_from(start);
        }
        let kind_pos = self.pos;
        self.advance(); // class/struct/interface/enum
        let name = self.ident_name();
        let owner = name.as_ref().map(|name| name.span);
        if name.is_none() {
            self.reject_type_index(self.span());
        }
        let name = name.unwrap_or(CsName {
            text: String::new(),
            span: self.span(),
        });
        let params_start = self.pos;
        let params_closed = self.skip_generic_params();
        let type_params = self.collect_index_type_params(kind, params_start, params_closed);
        let (bases, base_clause) = if let (true, CsTypeKind::Class, Some(owner), Some(params)) = (
            self.checked_coverage.is_some() && self.at(K::Colon),
            kind,
            owner,
            type_params.as_ref(),
        ) {
            let bare_allowed = params.is_empty()
                && self.index_namespace.is_none()
                && self.index_containing_type.is_none();
            self.parse_checked_bases(owner, bare_allowed)
        } else {
            if !self.at(K::LBrace) {
                // The legacy parser can consume tails the index does not support.
                self.reject_type_index(self.span());
            }
            (self.parse_bases(owner), None)
        };
        // Any remaining constraint text before the body.
        while !self.at(K::Eof) && !self.at(K::LBrace) && !self.at(K::Semicolon) {
            self.advance();
        }
        let header_end = self.last_end();
        let mut children = std::mem::take(attrs);
        if self.eat(K::LBrace) {
            let previous_type = self.index_containing_type;
            if let Some(type_params) = type_params {
                self.enter_index_type(
                    kind,
                    &name,
                    type_params,
                    Span::new(self.toks[kind_pos].span.start, header_end),
                    start,
                    kind_pos,
                );
                if let Some(clause) = base_clause {
                    self.record_index_base(clause);
                }
                self.cover_syntax_token(kind_pos);
                self.cover_syntax_token(self.pos - 1);
            }
            if kind == CsTypeKind::Enum {
                if !self.at(K::RBrace) {
                    self.reject_type_index(self.span());
                }
                self.parse_enum_members_into(&mut children);
            } else {
                self.parse_members_into(&mut children);
            }
            let _closed = self.eat(K::RBrace);
            self.leave_index_type(previous_type, _closed);
        } else {
            self.eat(K::Semicolon);
        }
        CsNode::new(
            CsNodeKind::Type { kind, name, bases },
            Span::new(start, self.last_end()),
            children,
        )
    }

    fn enter_index_type(
        &mut self,
        kind: CsTypeKind,
        name: &CsName,
        type_params: Vec<CsName>,
        header_span: Span,
        start: usize,
        kind_pos: usize,
    ) {
        if let Some(Ok(index)) = self.type_index.as_deref_mut() {
            // Reserve the owner's ordinal before any nested declaration is dispatched.
            let ordinal = index.rows.len();
            let public_modifier = self.checked_coverage.as_ref().and_then(|_| {
                // The type modifier check has already admitted at most one
                // token; it is immediately before the kind in the same parse.
                let token = &self.toks[kind_pos.checked_sub(1)?];
                (token.span.start == start && token.is_kw("public"))
                    .then(|| CallableHeaderTokens::name_evidence(token))
            });
            index.rows.push(TypeHeaderRow {
                ordinal,
                namespace: self.index_namespace,
                containing_type: self.index_containing_type,
                kind,
                name: name.clone(),
                public_modifier,
                type_params,
                header_span,
                declaration_span: Span::new(start, header_span.end),
            });
            self.index_containing_type = Some(ordinal);
        }
    }

    fn leave_index_type(&mut self, previous_type: Option<usize>, closed: bool) {
        let ordinal = std::mem::replace(&mut self.index_containing_type, previous_type);
        if !closed {
            self.reject_type_index(self.span());
            return;
        }
        let end = self.last_end();
        if let (Some(ordinal), Some(Ok(index))) = (ordinal, self.type_index.as_deref_mut()) {
            index.rows[ordinal].declaration_span.end = end;
            self.cover_syntax_token(self.pos - 1);
        }
    }

    fn parse_bases(&mut self, owner: Option<Span>) -> Vec<CsName> {
        let mut observe = self.observe_base_entry.take();
        let bases = self.parse_observed_bases(owner, &mut observe);
        self.observe_base_entry = observe;
        bases
    }

    fn parse_checked_bases(
        &mut self,
        owner: Span,
        bare_allowed: bool,
    ) -> (Vec<CsName>, Option<BaseClause>) {
        let colon_pos = self.pos;
        let colon = self.span();
        let entry_pos = colon_pos + 1;
        let entry_start = self.toks[entry_pos].span;
        let mut first_entry = None;
        let mut observe = |reported_owner, tokens: &[CsToken], terminal| {
            first_entry.get_or_insert_with(|| {
                if reported_owner != Some(owner) || tokens[0].span != entry_start {
                    return Err(entry_start);
                }
                Self::checked_base_path(tokens, terminal, bare_allowed)
                    .map(|path| (path, entry_pos + tokens.len()))
            });
        };
        let bases = self.parse_observed_bases(Some(owner), &mut Some(&mut observe));
        let clause = match first_entry.unwrap_or(Err(entry_start)) {
            // The next actual token must be the same brace the legacy parser
            // stopped at. A skipped leading, trailing, or intervening comma
            // cannot masquerade as part of the single admitted entry.
            Ok((path, end)) if end == self.pos && self.at(K::LBrace) => {
                self.cover_punctuation(colon_pos, end, &[K::Colon, K::ColonColon, K::Dot]);
                Some(BaseClause {
                    colon,
                    path,
                    body_open: self.span(),
                })
            }
            Ok((_, end)) => {
                self.reject_type_index(self.toks[end.min(self.pos)].span);
                None
            }
            Err(span) => {
                self.reject_type_index(span);
                None
            }
        };
        (bases, clause)
    }

    fn checked_base_path(
        tokens: &[CsToken],
        terminal: Option<Span>,
        bare_allowed: bool,
    ) -> Result<BasePath, Span> {
        let name = |token: &CsToken| {
            if token.kind != K::Ident {
                return Err(token.span);
            }
            CallableHeaderTokens::reference_name(token)
        };
        let first = &tokens[0];
        if let [root, separator, tail @ ..] = tokens {
            if root.is_kw("global") && separator.kind == K::ColonColon {
                let mut tail = tail.iter();
                let mut segments = vec![name(tail.next().ok_or(separator.span)?)?];
                let mut dots = Vec::new();
                while let Some(dot) = tail.next() {
                    if dot.kind != K::Dot {
                        return Err(dot.span);
                    }
                    dots.push(dot.span);
                    segments.push(name(tail.next().ok_or(dot.span)?)?);
                }
                let path =
                    QualifiedTypeUse::new(CallableHeaderTokens::name_evidence(root), segments);
                if terminal != path.segments.last().map(|segment| segment.span) {
                    return Err(path.span);
                }
                return Ok(BasePath::Rooted {
                    path,
                    separator: separator.span,
                    dots,
                });
            }
        }
        let bare = name(first)?;
        if let Some(extra) = tokens.get(1) {
            return Err(extra.span);
        }
        if !bare_allowed || terminal != Some(first.span) {
            return Err(first.span);
        }
        Ok(BasePath::Bare(bare))
    }

    fn record_index_base(&mut self, clause: BaseClause) {
        if let (
            Some(Ok(_)),
            Some(owner),
            Some(IndexedCallables::Qualified {
                lexical:
                    Some(LexicalObserver {
                        sites: Some(ObservedSites::Checked { bases, .. }),
                        ..
                    }),
                ..
            }),
        ) = (
            self.type_index.as_deref(),
            self.index_containing_type,
            &mut self.index_callables,
        ) {
            bases.push(BaseOccurrence { owner, clause });
        }
    }

    fn parse_observed_bases(
        &mut self,
        owner: Option<Span>,
        observe: &mut Option<BaseEntryObserver<'_>>,
    ) -> Vec<CsName> {
        let mut bases = Vec::new();
        let mut pending = None;
        let mut legacy_end = None;
        if self.eat(K::Colon) {
            while !self.at_base_list_end() {
                if self.eat(K::Comma) {
                    continue;
                }
                let start = self.pos;
                let (entry_start, mut delimiters) =
                    pending.unwrap_or((start, BaseEntryDelimiters::default()));
                let (name, supported) =
                    self.parse_base_entry(&mut legacy_end, delimiters, observe.is_some());
                if self.pos == start {
                    break; // constraint boundary
                }
                // Names beyond the legacy stop are only observed, not cloned
                // into the public AST.
                if let Some(index) = name.filter(|&index| legacy_end.is_none_or(|end| index < end))
                {
                    let token = &self.toks[index];
                    bases.push(CsName {
                        text: token.text.clone(),
                        span: token.span,
                    });
                }
                if let Some(observe) = observe {
                    // Recovery may split an unsupported entry at an inner
                    // comma. Join only the observed ranges, not the legacy names.
                    let terminal = name
                        .filter(|_| supported && entry_start == start)
                        .map(|index| self.toks[index].span);
                    pending = if terminal.is_none()
                        && delimiters.has_open_after(&self.toks[start..self.pos])
                    {
                        Some((entry_start, delimiters))
                    } else {
                        observe(owner, &self.toks[entry_start..self.pos], terminal);
                        None
                    };
                }
            }
        }
        if let (Some(observe), Some((start, _))) = (observe, pending) {
            observe(owner, &self.toks[start..self.pos], None);
        }
        if let Some(end) = legacy_end {
            self.pos = end; // resume with the legacy header skip/recovery
        }
        bases
    }

    fn at_base_list_end(&self) -> bool {
        matches!(self.k(), K::Eof | K::LBrace | K::Semicolon)
    }

    fn parse_base_entry(
        &mut self,
        legacy_end: &mut Option<usize>,
        mut delimiters: BaseEntryDelimiters,
        observing: bool,
    ) -> (Option<usize>, bool) {
        let start = self.pos;
        let mut delimiters_scanned_to = start;
        let mut at_end = |parser: &Self| {
            if parser.at(K::Comma) || parser.at_base_list_end() {
                return true;
            }
            if !parser.at_kw("where") {
                return false;
            }
            if !observing {
                return true;
            }
            // Legacy stops at any `where` encountered here. The observer
            // stops only at an outer `where Ident :` constraint.
            legacy_end.get_or_insert(parser.pos);
            if !matches!(
                parser.toks.get(parser.pos + 1..parser.pos + 3),
                Some([parameter, colon]) if parameter.kind == K::Ident && colon.kind == K::Colon
            ) {
                return false;
            }
            let nested = delimiters.has_open_after(&parser.toks[delimiters_scanned_to..parser.pos]);
            delimiters_scanned_to = parser.pos;
            !nested
        };

        // Legacy recovery still looks for the first identifier after stray
        // tokens, but the observer must not treat it as a supported path.
        while !self.at(K::Ident) && !at_end(self) {
            self.advance();
        }
        if at_end(self) {
            return (None, false);
        }
        let mut supported = self.pos == start;
        let mut name = self.pos;
        self.advance();
        let mut first_separator = true;
        while matches!(self.k(), K::Dot | K::ColonColon) {
            supported &= first_separator || self.at(K::Dot);
            first_separator = false;
            self.advance();
            if self.at(K::Ident) {
                name = self.pos;
                self.advance();
            } else {
                supported = false;
            }
        }
        // A trailing balanced generic list is opaque here. Its commas
        // are not base separators and its identifiers are not base names.
        if self.at(K::Lt) {
            supported &= self.consume_generic_args(None);
        }
        while !at_end(self) {
            supported = false;
            if self.at(K::Lt) {
                self.consume_generic_args(None);
            } else {
                self.advance();
            }
        }
        (Some(name), supported)
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
        let member_start = self.pos;
        if self.at(K::LBracket) {
            self.reject_type_index(self.span());
        }
        let mut attrs = self.parse_attrs();
        self.skip_modifiers();
        if let Some(kind) = self.type_keyword() {
            return Some(self.parse_type(start, kind, &mut attrs));
        }
        let indexed_header = self.collect_index_member(member_start);
        let member = self.parse_member(start, attrs, indexed_header);
        if let Some(IndexedCallables::Qualified {
            lexical: Some(lexical),
            ..
        }) = &mut self.index_callables
        {
            // Also clear a reserved occurrence if member recovery skipped its body.
            lexical.body = None;
        }
        member
    }

    fn collect_index_member(&mut self, start: usize) -> Option<IndexedMemberHeader> {
        let (Some(Ok(index)), Some(callables), Some(owner)) = (
            self.type_index.as_deref(),
            self.index_callables.as_mut(),
            self.index_containing_type,
        ) else {
            // Preserve the type-only refusal at the post-modifier token.
            self.reject_type_index(self.span());
            return None;
        };
        let owner = &index.rows[owner];
        if !matches!(owner.kind, CsTypeKind::Class | CsTypeKind::Struct) {
            self.reject_type_index(self.span());
            return None;
        }
        let checked = self.checked_coverage.is_some();
        if start != self.pos && !checked {
            self.reject_type_index(self.toks[start].span);
            return None;
        }
        let modifiers = if checked {
            match Self::checked_member_modifiers(&self.toks[start..self.pos]) {
                Ok(modifiers) => modifiers,
                Err(span) => {
                    self.reject_type_index(span);
                    return None;
                }
            }
        } else {
            MemberModifiers::default()
        };
        if start != self.pos
            && self
                .toks
                .get(self.pos + 1)
                .is_some_and(|token| token.kind == K::LParen)
        {
            // Constructors are still admitted only without modifiers.
            self.reject_type_index(self.toks[start].span);
            return None;
        }
        // The opt-in modes model method slots only, never containing-type slots.
        #[cfg(test)]
        let generic_methods = matches!(callables, IndexedCallables::Generic(_))
            || matches!(
                callables,
                IndexedCallables::Qualified { lexical: Some(lexical), .. }
                    if matches!(lexical.sites, Some(ObservedSites::Checked {
                        mode: CheckedHeaderMode::GenericEvidence, ..
                    }))
            );
        #[cfg(test)]
        let recognize_method_type_params = generic_methods
            && index
                .type_lineage(Some(owner.ordinal))
                .all(|row| row.type_params.is_empty());
        // Legacy parsing is more permissive than this index; look ahead without
        // moving its cursor or relying on what survived the AST projection.
        let member_pos = self.pos;
        let tokens = &self.toks[member_pos..];
        let coverage = &mut self.checked_coverage;
        #[cfg(test)]
        let bridge_type_params = |offset: Option<usize>| {
            offset.map(|offset| IndexedMemberHeader::MethodTypeParams {
                open_paren_pos: member_pos + offset,
            })
        };
        let header = match callables {
            IndexedCallables::Qualified { rows, lexical } => {
                if lexical.is_some() {
                    if let Ok((type_use, name)) =
                        CallableHeaderTokens(tokens).uninitialized_primitive()
                    {
                        return Some(IndexedMemberHeader::Field(FieldOccurrence {
                            containing_type: owner.ordinal,
                            type_use,
                            name,
                            modifiers,
                        }));
                    }
                }
                Self::validate_callable_header(
                    tokens,
                    owner,
                    rows.len(),
                    #[cfg(test)]
                    recognize_method_type_params,
                    lexical.is_none(),
                    CallableHeaderTokens::qualified_parameter,
                )
                .map(|mut header| {
                    header.row.modifiers = modifiers;
                    header.row.header_span.start = self.toks[start].span.start;
                    header.row.declaration_span.start = self.toks[start].span.start;
                    let header_end = header.row.header_span.end;
                    if let Some(lexical) = lexical {
                        lexical.body = Some(IndexedCallableBody {
                            callable: header.row.ordinal,
                            first_block: lexical.declarations.blocks.len(),
                            current_block: None,
                        });
                    }
                    rows.push(header.row);
                    if let Some(coverage) = coverage {
                        let end = member_pos
                            + tokens.partition_point(|token| token.span.start < header_end);
                        coverage.cover_punctuation(
                            &self.toks,
                            member_pos,
                            end,
                            &[
                                K::Lt,
                                K::Gt,
                                K::LParen,
                                K::RParen,
                                K::Comma,
                                K::ColonColon,
                                K::Dot,
                            ],
                        );
                    }
                    #[cfg(test)]
                    return bridge_type_params(header.generic_open_paren_offset);
                    #[cfg(not(test))]
                    None
                })
            }
            #[cfg(test)]
            IndexedCallables::Simple(rows) | IndexedCallables::Generic(rows) => {
                Self::validate_callable_header(
                    tokens,
                    owner,
                    rows.len(),
                    recognize_method_type_params,
                    true,
                    CallableHeaderTokens::simple_parameter,
                )
                .map(|header| {
                    rows.push(header.row);
                    bridge_type_params(header.generic_open_paren_offset)
                })
            }
        };
        match header {
            Ok(header) => header,
            Err(span) => {
                self.reject_type_index(span);
                None
            }
        }
    }

    fn checked_member_modifiers(mut tokens: &[CsToken]) -> Result<MemberModifiers, Span> {
        let mut take = |keyword| {
            let (token, rest) = tokens.split_first()?;
            if !token.is_kw(keyword) {
                return None;
            }
            tokens = rest;
            Some(CallableHeaderTokens::name_evidence(token))
        };
        let public = take("public");
        let static_token = take("static");
        if let Some(token) = tokens.first() {
            return Err(token.span);
        }
        Ok(MemberModifiers {
            public,
            static_token,
        })
    }

    fn validate_callable_header<'tokens, Parameter>(
        member_tokens: &'tokens [CsToken],
        owner: &TypeHeaderRow,
        ordinal: usize,
        #[cfg(test)] recognize_method_type_params: bool,
        require_empty_body: bool,
        parse_parameter: fn(
            &mut CallableHeaderTokens<'tokens>,
            #[cfg(test)] &HashMap<&'tokens str, usize>,
        ) -> Result<Parameter, Span>,
    ) -> Result<ValidatedCallableHeader<Parameter>, Span> {
        let mut tokens = CallableHeaderTokens(member_tokens);
        let first = tokens.name()?;
        let start = first.span.start;
        let (kind, name, return_type) = if tokens.at(K::LParen) {
            if first.text != owner.name.text {
                return Err(first.span);
            }
            (CallableKind::Constructor, first, None)
        } else {
            if !matches!(first.text.as_str(), "void" | "int" | "bool") {
                return Err(first.span);
            }
            (CallableKind::Method, tokens.name()?, Some(first))
        };
        #[cfg(test)]
        let mut method_slots = HashMap::new();
        #[cfg(test)]
        let type_params =
            if recognize_method_type_params && kind == CallableKind::Method && tokens.at(K::Lt) {
                tokens.method_type_params(&mut method_slots)?
            } else {
                Vec::new()
            };
        #[cfg(test)]
        let generic_open_paren_offset =
            (!type_params.is_empty()).then(|| member_tokens.len() - tokens.0.len());
        // With generic evidence disabled, this refuses a method at its
        // actual `<` and leaves the ordinary cursor to report recovery.
        tokens.take(K::LParen)?;
        let mut parameters = Vec::new();
        if !tokens.at(K::RParen) {
            loop {
                parameters.push(parse_parameter(
                    &mut tokens,
                    #[cfg(test)]
                    &method_slots,
                )?);
                if !tokens.at(K::Comma) {
                    break;
                }
                tokens.take(K::Comma)?;
            }
        }
        let header_end = tokens.take(K::RParen)?.span.end;
        let open = tokens.take(K::LBrace)?;
        #[cfg(test)]
        let require_empty_body = require_empty_body || generic_open_paren_offset.is_some();
        let declaration_end = if require_empty_body {
            tokens.take(K::RBrace)?.span.end
        } else {
            // The lexical observer completes this span only on a real closing brace.
            open.span.end
        };
        Ok(ValidatedCallableHeader {
            row: CallableHeaderRow {
                ordinal,
                containing_type: owner.ordinal,
                kind,
                name,
                return_type,
                modifiers: MemberModifiers::default(),
                #[cfg(test)]
                type_params,
                parameters,
                header_span: Span::new(start, header_end),
                declaration_span: Span::new(start, declaration_end),
            },
            #[cfg(test)]
            generic_open_paren_offset,
        })
    }

    fn parse_member(
        &mut self,
        start: usize,
        attrs: Vec<CsNode>,
        indexed_header: Option<IndexedMemberHeader>,
    ) -> Option<CsNode> {
        self.enter();
        let r = self.parse_member_inner(start, attrs, indexed_header);
        self.leave();
        r
    }

    fn parse_member_inner(
        &mut self,
        start: usize,
        mut attrs: Vec<CsNode>,
        indexed_header: Option<IndexedMemberHeader>,
    ) -> Option<CsNode> {
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

        let pending_field = match indexed_header {
            Some(IndexedMemberHeader::Field(field)) => Some(field),
            #[cfg(test)]
            Some(IndexedMemberHeader::MethodTypeParams { open_paren_pos }) => {
                // Only this member's completed lookahead can bridge the binder list.
                self.pos = open_paren_pos;
                None
            }
            None => None,
        };

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
            let _terminated = self.eat(K::Semicolon);
            if let (
                true,
                Some(field),
                Some(IndexedCallables::Qualified {
                    lexical: Some(lexical),
                    ..
                }),
            ) = (_terminated, pending_field, &mut self.index_callables)
            {
                lexical.declarations.fields.push(field);
                self.cover_syntax_token(self.pos - 1);
            }
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
            if self.at(K::LBrace) {
                self.reject_lexical_body(self.span());
            }
            return self.skip_balanced_block(start);
        }
        let open = self.span();
        let _opened = self.eat(K::LBrace);
        if _opened {
            self.enter_index_block(open);
        }
        let mut children = Vec::new();
        while !self.at(K::Eof) && !self.at(K::RBrace) {
            let before = self.pos;
            self.parse_statement(&mut children);
            if self.pos == before {
                self.advance();
            }
        }
        let _closed = self.eat(K::RBrace);
        self.leave_index_block(_closed);
        CsNode::new(
            CsNodeKind::Block,
            Span::new(start, self.last_end()),
            children,
        )
    }

    fn reject_lexical_body(&mut self, span: Span) {
        if self.active_lexical_observer().is_some() {
            self.reject_type_index(span);
        }
    }

    fn enter_index_block(&mut self, open: Span) {
        if let Some(ActiveLexicalBody {
            declarations, body, ..
        }) = self.active_lexical_observer()
        {
            let block = declarations.blocks.len();
            let parent = body.current_block.replace(block);
            declarations.blocks.push(BlockOccurrence {
                callable: body.callable,
                ordinal: block - body.first_block,
                parent: parent.map(|parent| parent - body.first_block),
                span: open,
            });
            self.cover_syntax_token(self.pos - 1);
        }
    }

    fn leave_index_block(&mut self, closed: bool) {
        if !closed {
            self.reject_lexical_body(self.span());
            return;
        }
        let end = self.last_end();
        if let Some(ActiveLexicalBody {
            declarations,
            body,
            rows,
            ..
        }) = self.active_lexical_observer()
        {
            if let Some(block) = body.current_block {
                let row = &mut declarations.blocks[block];
                row.span.end = end;
                body.current_block = row.parent.map(|parent| body.first_block + parent);
                if row.parent.is_none() {
                    rows[body.callable].declaration_span.end = end;
                }
                self.cover_syntax_token(self.pos - 1);
            }
        }
    }

    fn active_lexical_observer(&mut self) -> Option<ActiveLexicalBody<'_>> {
        match (&self.type_index, &mut self.index_callables) {
            (
                Some(index),
                Some(IndexedCallables::Qualified {
                    rows,
                    lexical:
                        Some(LexicalObserver {
                            declarations,
                            body: Some(body),
                            sites,
                        }),
                }),
            ) if index.is_ok() => Some(ActiveLexicalBody {
                declarations,
                body,
                rows,
                sites: sites.as_mut().map(ObservedSites::sites),
            }),
            _ => None,
        }
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
            K::Semicolon => {
                self.reject_lexical_body(self.span());
                self.advance();
            }
            K::Ident if self.is_control_kw() => {
                self.reject_lexical_body(self.span());
                self.parse_control(out);
            }
            _ => self.parse_simple_statement(out),
        }
    }

    /// Control flow: the keyword, an optional `( ... )` header (which may
    /// declare locals), and a following statement. The header's declarations
    /// and the body are wrapped in a synthetic block so the declarations scope
    /// to it.
    fn parse_control(&mut self, out: &mut Vec<CsNode>) {
        // Count the synthetic scope, not every statement: real blocks already
        // enter through parse_block. This also bounds brace-free control chains.
        self.enter();
        self.parse_control_inner(out);
        self.leave();
    }

    fn parse_control_inner(&mut self, out: &mut Vec<CsNode>) {
        if self.depth >= MAX_DEPTH {
            out.push(self.recover());
            return;
        }
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
        let start = self.pos;
        let indexed = self.validate_index_statement();
        // Try a local declaration; if it doesn't match, scan as an expression.
        let _committed = self.try_local_decl(out);
        self.scan_expr(out);
        let terminator = self.pos;
        let _terminated = self.eat(K::Semicolon);
        if let Some((statement, predicted_terminator)) = indexed {
            let ended_as_predicted = _terminated && terminator == predicted_terminator;
            match statement {
                IndexedStatement::Local(type_use, name) if ended_as_predicted && _committed => {
                    self.record_index_local(type_use, name);
                    self.cover_syntax_token(terminator);
                }
                IndexedStatement::Assignment(write, read) if ended_as_predicted && !_committed => {
                    self.record_index_places(write, ReferenceRole::Write, Some(read));
                    self.cover_punctuation(
                        start,
                        self.pos,
                        &[K::Assign, K::Dot, K::ColonColon, K::Semicolon],
                    );
                }
                IndexedStatement::Call(target, argument) if ended_as_predicted && !_committed => {
                    let read = argument.map(|name| StatementPlace {
                        name,
                        receiver: Receiver::Bare,
                    });
                    self.record_index_places(target, ReferenceRole::Call, read);
                    self.cover_punctuation(
                        start,
                        self.pos,
                        &[K::LParen, K::RParen, K::Dot, K::ColonColon, K::Semicolon],
                    );
                }
                _ => self.reject_type_index(self.span()),
            }
        }
    }

    fn validate_index_statement(&mut self) -> Option<(IndexedStatement, usize)> {
        let binding = self.active_lexical_observer()?.sites.is_some();
        // Refuse before expression scanning: a lambda or initializer could
        // otherwise reach the real-block hook even though it is not a statement block.
        let mut tokens = CallableHeaderTokens(&self.toks[self.pos..]);
        let validated = if !binding || self.at_kw("int") || self.at_kw("bool") {
            tokens
                .uninitialized_primitive()
                .map(|(type_use, name)| IndexedStatement::Local(type_use, name))
        } else {
            let checked = self.checked_coverage.is_some();
            let class_base = checked
                && matches!(
                    (self.type_index.as_deref(), self.index_containing_type),
                    (Some(Ok(index)), Some(owner)) if index.rows[owner].kind == CsTypeKind::Class
                );
            tokens.place_statement(checked, class_base)
        };
        match validated {
            Ok(statement) => Some((statement, self.toks.len() - tokens.0.len() - 1)),
            Err(span) => {
                self.reject_type_index(span);
                None
            }
        }
    }

    fn record_index_local(&mut self, type_use: CsName, name: CsName) {
        if let Some(ActiveLexicalBody {
            declarations, body, ..
        }) = self.active_lexical_observer()
        {
            if let Some(block) = body.current_block {
                declarations.locals.push(LocalOccurrence {
                    callable: body.callable,
                    block: block - body.first_block,
                    type_use,
                    name,
                });
            }
        }
    }

    fn record_index_places(
        &mut self,
        target: StatementPlace,
        role: ReferenceRole,
        read: Option<StatementPlace>,
    ) {
        if let Some(ActiveLexicalBody {
            body,
            sites: Some(sites),
            ..
        }) = self.active_lexical_observer()
        {
            if let Some(block) = body.current_block {
                let owner = SiteOwner {
                    callable: body.callable,
                    block: block - body.first_block,
                };
                let call_argument = read
                    .as_ref()
                    .filter(|_| role == ReferenceRole::Call)
                    .map(|argument| argument.name.span);
                target.record(sites, owner, role, call_argument);
                if let Some(read) = read {
                    read.record(sites, owner, ReferenceRole::Read, None);
                }
            }
        }
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
                K::Lt => {
                    self.consume_generic_args(Some(out));
                }
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

    fn consume_generic_args(&mut self, mut out: Option<&mut Vec<CsNode>>) -> bool {
        // Balance `< ... >`, capturing identifiers only for callers that need
        // type references. Recovery stops at the same tokens without a match.
        let mut depth = 0i32;
        loop {
            match self.k() {
                K::Eof | K::Semicolon | K::LBrace | K::RBrace => return false,
                K::Lt => {
                    depth += 1;
                    self.advance();
                }
                K::Gt => {
                    depth -= 1;
                    self.advance();
                    if depth <= 0 {
                        return true;
                    }
                }
                K::Ident => {
                    if let Some(out) = &mut out {
                        if !skip_as_ref(&self.tok().text) {
                            out.push(root_ref(CsName {
                                text: self.tok().text.clone(),
                                span: self.span(),
                            }));
                        }
                    }
                    self.advance();
                }
                _ => self.advance(),
            }
        }
    }

    /// Returns whether the list was absent or closed, without changing recovery.
    fn skip_generic_params(&mut self) -> bool {
        !self.at(K::Lt) || self.consume_generic_args(None)
    }

    fn collect_index_type_params(
        &mut self,
        kind: CsTypeKind,
        start: usize,
        closed: bool,
    ) -> Option<Vec<CsName>> {
        if !matches!(self.type_index.as_deref(), Some(Ok(_))) {
            return None;
        }
        if start == self.pos {
            return Some(Vec::new());
        }
        if kind == CsTypeKind::Enum {
            self.reject_type_index(self.toks[start].span);
            return None;
        }
        // Inspect only what the legacy skip consumed, without moving its cursor.
        // The first excluded token is the close or the legacy recovery boundary.
        let end = self.pos - usize::from(closed);
        match Self::validate_index_type_params(
            &self.toks[start + 1..end],
            closed,
            self.toks[end].span,
        ) {
            Ok(names) => {
                self.cover_punctuation(start, self.pos, &[K::Lt, K::Comma, K::Gt]);
                Some(names)
            }
            Err(span) => {
                self.reject_type_index(span);
                None
            }
        }
    }

    fn validate_index_type_params(
        tokens: &[CsToken],
        closed: bool,
        end: Span,
    ) -> Result<Vec<CsName>, Span> {
        let mut names = Vec::new();
        let mut seen = HashSet::new();
        let mut expect_name = true;
        for token in tokens {
            if expect_name {
                if token.kind != K::Ident
                    || token.is_kw("in")
                    || token.is_kw("out")
                    || !seen.insert(token.text.as_str())
                {
                    return Err(token.span);
                }
                names.push(CsName {
                    text: token.text.clone(),
                    span: token.span,
                });
            } else if token.kind != K::Comma {
                return Err(token.span);
            }
            expect_name = !expect_name;
        }
        if expect_name || !closed {
            return Err(end);
        }
        Ok(names)
    }

    // small helpers

    fn qualified_name(&mut self, mut segments: Option<&mut Vec<CsName>>) -> Option<CsName> {
        let first = self.ident_name();
        if first.is_none() {
            self.reject_type_index(self.span());
        }
        let first = first?;
        let mut text = first.text.clone();
        let start = first.span.start;
        let mut end = first.span.end;
        if let Some(segments) = segments.as_deref_mut() {
            segments.push(first);
        }
        while self.at(K::Dot) {
            let dot = self.pos;
            self.advance();
            if let Some(part) = self.ident_name() {
                text.push('.');
                text.push_str(&part.text);
                end = part.span.end;
                if let Some(segments) = segments.as_deref_mut() {
                    segments.push(part);
                    self.cover_syntax_token(dot);
                }
            } else {
                self.reject_type_index(self.toks[dot].span);
                break;
            }
        }
        Some(CsName {
            text,
            span: Span::new(start, end),
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
        let start = self.pos;
        while self.at(K::Ident) && MODIFIERS.contains(&self.tok().text.as_str()) {
            self.advance();
        }
        // Member validation owns its own modifier refusal, not this type check.
        if start != self.pos && self.type_keyword().is_some() {
            if let Some(IndexedCallables::Qualified { lexical, .. }) = &self.index_callables {
                let modifiers = &self.toks[start..self.pos];
                let unsupported = if lexical
                    .as_ref()
                    .is_some_and(|observer| observer.sites.is_some())
                {
                    let allowed = usize::from(modifiers[0].is_kw("public"));
                    modifiers.get(allowed)
                } else {
                    modifiers.iter().find(|token| token.is_kw("partial"))
                };
                if let Some(token) = unsupported {
                    self.reject_type_index(token.span);
                }
            }
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
        if self.depth == MAX_DEPTH {
            let at = self.span();
            if let Some(limit) = self
                .checked_coverage
                .as_mut()
                .and_then(|coverage| coverage.depth_limit.as_deref_mut())
            {
                limit.get_or_insert(at);
            }
        }
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
    use crate::v2::csharp::local::{index_lexical_declarations, index_qualified_callable_headers};

    fn unit(code: &str) -> CsUnit {
        cs_parse(code, 0)
    }

    fn name(text: &str, start: usize, end: usize) -> CsName {
        CsName {
            text: text.to_owned(),
            span: Span::new(start, end),
        }
    }

    fn recognize_callable(
        code: &str,
        base: usize,
        recognize_method_type_params: bool,
    ) -> Result<CallableHeaderRow, Span> {
        let tokens = cs_lex(code, base);
        let owner = TypeHeaderRow {
            ordinal: 0,
            namespace: None,
            containing_type: None,
            kind: CsTypeKind::Class,
            name: name("C", base + 6, base + 7),
            public_modifier: None,
            type_params: Vec::new(),
            header_span: Span::new(base, base + 7),
            declaration_span: Span::new(base, base + code.len()),
        };
        CsParser::validate_callable_header(
            &tokens[3..],
            &owner,
            0,
            recognize_method_type_params,
            true,
            CallableHeaderTokens::simple_parameter,
        )
        .map(|header| header.row)
    }

    #[test]
    fn callable_lookahead_records_absolute_binder_and_type_use_spans_in_order() {
        let code = "class C{void F<T,U>(U x,T y){/* } */}}";

        let row = recognize_callable(code, 3, true);

        assert_eq!(
            row,
            Ok(CallableHeaderRow {
                ordinal: 0,
                containing_type: 0,
                kind: CallableKind::Method,
                name: name("F", 16, 17),
                return_type: Some(name("void", 11, 15)),
                modifiers: MemberModifiers::default(),
                type_params: vec![name("T", 18, 19), name("U", 20, 21)],
                parameters: vec![
                    CallableParameter {
                        type_use: CallableParameterTypeUse::Simple(name("U", 23, 24)),
                        comparison_type: CallableParameterType::MethodSlot(1),
                        name: name("x", 25, 26),
                    },
                    CallableParameter {
                        type_use: CallableParameterTypeUse::Simple(name("T", 27, 28)),
                        comparison_type: CallableParameterType::MethodSlot(0),
                        name: name("y", 29, 30),
                    },
                ],
                header_span: Span::new(11, 31),
                declaration_span: Span::new(11, 40),
            })
        );
    }

    #[test]
    fn callable_keys_compare_binder_slots_not_their_raw_spellings() {
        let first =
            recognize_callable("class C{int F<T,U>(U x,T y,int z,bool b){}}", 0, true).unwrap();
        let renamed =
            recognize_callable("class C{bool F<A,a>(a p,A q,int r,bool s){}}", 0, true).unwrap();

        let renamed_key = renamed.key();

        assert_eq!(renamed_key, first.key());
    }

    #[test]
    fn callable_keys_distinguish_generic_arity_even_without_parameters() {
        let one_binder = recognize_callable("class C{void F<T>(){}}", 0, true).unwrap();
        let two_binders = recognize_callable("class C{void F<T,U>(){}}", 0, true).unwrap();

        let two_binder_key = two_binders.key();

        assert_ne!(two_binder_key, one_binder.key());
    }

    #[test]
    fn callable_keys_distinguish_method_slot_order() {
        let forward = recognize_callable("class C{void F<T,U>(T x,U y){}}", 0, true).unwrap();
        let reverse = recognize_callable("class C{void F<T,U>(U x,T y){}}", 0, true).unwrap();

        let reverse_key = reverse.key();

        assert_ne!(reverse_key, forward.key());
    }

    #[test]
    fn callable_lookahead_refuses_method_binders_without_explicit_recognition() {
        let code = "class C{void F<T>(){}}";

        let row = recognize_callable(code, 0, false);

        assert_eq!(row, Err(Span::new(14, 15)));
    }

    fn assert_qualified_header_refused(code: &str, start: usize, end: usize) {
        let region = Span::new(0, code.len());

        let index = index_qualified_callable_headers(code, region);

        assert!(
            matches!(
                &index,
                Err(GenericCallableIndexError::Rejected {
                    first_unsupported: Some(span),
                    ..
                }) if *span == Span::new(start, end)
            ),
            "{code:?}: expected refusal at {start}..{end}, got {index:?}"
        );
    }

    #[test]
    fn qualified_header_refuses_roots_other_than_exact_unescaped_global_colon_colon() {
        for (parameter, end) in [
            ("global : : N p", 21),
            ("@global::N p", 22),
            ("Global::N p", 21),
            ("Alias::N p", 20),
            ("N.X p", 16),
        ] {
            assert_qualified_header_refused(
                &format!("class C{{void F({parameter}){{}}}}"),
                15,
                end,
            );
        }
    }

    #[test]
    fn qualified_header_refuses_missing_segments_and_unsupported_path_suffixes() {
        for (parameter, start, end) in [
            ("global::", 23, 24),
            ("global::N. ", 26, 27),
            ("global::N::X p", 24, 26),
            ("global::N<T> p", 24, 25),
            ("global::A.X[] p", 26, 27),
            ("global::N? p", 24, 25),
            ("global::N* p", 24, 25),
            ("global::A.X", 26, 27),
        ] {
            assert_qualified_header_refused(
                &format!("class C{{void F({parameter}){{}}}}"),
                start,
                end,
            );
        }
    }

    #[test]
    fn qualified_header_still_refuses_callables_outside_the_simple_subset() {
        for (code, start, end) in [
            ("class C{void F(global::N p = null){}}", 27, 28),
            ("class C{void F(ref global::N p){}}", 15, 18),
            ("class C{void F(global::N p){;}}", 28, 29),
            ("interface C{void F(global::N p){}}", 12, 16),
            ("class C{public void F(global::N p){}}", 8, 14),
            ("class C{global::N F(int p){}}", 8, 14),
            ("class C{D(global::N p){}}", 8, 9),
        ] {
            assert_qualified_header_refused(code, start, end);
        }
    }

    fn assert_callable_lookahead_refuses(code: &str, start: usize, end: usize) {
        let row = recognize_callable(code, 0, true);

        assert_eq!(row, Err(Span::new(start, end)), "source {code:?}");
    }

    #[test]
    fn callable_lookahead_refuses_malformed_binder_lists_at_the_offending_token_or_eof() {
        for (suffix, start, end) in [
            ("<>", 15, 16),
            ("<T,>", 17, 18),
            ("<T,T>", 17, 18),
            ("<T", 16, 16),
            ("<T U>", 17, 18),
            ("<N.T>", 16, 17),
            ("<T<U>>", 16, 17),
            ("<[A] T>", 15, 16),
        ] {
            assert_callable_lookahead_refuses(&format!("class C{{void F{suffix}"), start, end);
        }
    }

    #[test]
    fn callable_lookahead_refuses_binders_outside_unescaped_ascii_nonreserved_names() {
        for (binder, end) in [
            ("@T", 17),
            ("É", 17),
            ("1", 16),
            ("int", 18),
            ("where", 20),
            ("required", 23),
            ("scoped", 21),
        ] {
            assert_callable_lookahead_refuses(
                &format!("class C{{void F<{binder}>(){{}}}}"),
                15,
                end,
            );
        }
    }

    #[test]
    fn callable_lookahead_refuses_unsupported_parameter_types_or_missing_names() {
        for (suffix, start, end) in [
            ("<T>(@T x){}", 18, 20),
            ("<T>(t x){}", 18, 19),
            ("<T>(ref T x){}", 18, 21),
            ("<T>(T.N x){}", 19, 20),
            ("<T>(T[] x){}", 19, 20),
            ("<T>(T){}", 19, 20),
        ] {
            assert_callable_lookahead_refuses(&format!("class C{{void F{suffix}"), start, end);
        }
    }

    #[test]
    fn callable_lookahead_requires_parameter_delimiters_and_an_empty_body() {
        for (suffix, start, end) in [
            ("<T>;", 17, 18),
            ("<T>(T x", 21, 21),
            ("<T>(T x,)", 22, 23),
            ("<T>() where T : C {}", 20, 25),
            ("<T>(){;}", 20, 21),
            ("<T>(){", 20, 20),
        ] {
            assert_callable_lookahead_refuses(&format!("class C{{void F{suffix}"), start, end);
        }
    }

    #[test]
    fn callable_lookahead_refuses_member_forms_outside_the_existing_subset() {
        for (member, start, end) in [
            ("C<T>(){}", 8, 9),
            ("T F<T>(){}", 8, 9),
            ("int F;", 13, 14),
            ("public void F<T>(){}", 8, 14),
            ("[A] void F<T>(){}", 8, 9),
        ] {
            assert_callable_lookahead_refuses(&format!("class C{{{member}}}"), start, end);
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ObservedBase {
        owner: Option<Span>,
        tokens: Vec<(String, Span)>,
        terminal: Option<Span>,
    }

    fn observe_bases(code: &str, base: usize) -> (CsUnit, Vec<ObservedBase>) {
        let mut entries = Vec::new();
        let unit = cs_parse_with_base_observer(
            code,
            base,
            Some(&mut |owner, tokens, terminal| {
                entries.push(ObservedBase {
                    owner,
                    tokens: tokens.iter().map(|t| (t.text.clone(), t.span)).collect(),
                    terminal,
                });
            }),
        );
        (unit, entries)
    }

    fn observed_base(
        owner: Option<(usize, usize)>,
        tokens: &[(&str, usize, usize)],
        terminal: Option<(usize, usize)>,
    ) -> ObservedBase {
        ObservedBase {
            owner: owner.map(|(start, end)| Span::new(start, end)),
            tokens: tokens
                .iter()
                .map(|&(text, start, end)| (text.to_string(), Span::new(start, end)))
                .collect(),
            terminal: terminal.map(|(start, end)| Span::new(start, end)),
        }
    }

    fn base_names(node: &CsNode) -> Vec<(&str, Span)> {
        let CsNodeKind::Type { bases, .. } = &node.kind else {
            panic!("expected a type declaration, got {:?}", node.kind);
        };
        bases.iter().map(|b| (b.text.as_str(), b.span)).collect()
    }

    #[test]
    fn base_observer_retains_qualified_generic_tokens_and_absolute_terminal_span() {
        let code = "class D : global::A.B<C> {}";

        let (unit, entries) = observe_bases(code, 40);

        assert_eq!(
            entries,
            [observed_base(
                Some((46, 47)),
                &[
                    ("global", 50, 56),
                    ("::", 56, 58),
                    ("A", 58, 59),
                    (".", 59, 60),
                    ("B", 60, 61),
                    ("<", 61, 62),
                    ("C", 62, 63),
                    (">", 63, 64),
                ],
                Some((60, 61)),
            )]
        );
        assert_eq!(
            base_names(&unit.root.children[0]),
            [("B", Span::new(60, 61))]
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_still_reports_nested_bases_after_an_unclosed_declaration_parameter_list() {
        let code = "class D<T { class E : A.B {} }";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [observed_base(
                Some((18, 19)),
                &[("A", 22, 23), (".", 23, 24), ("B", 24, 25)],
                Some((24, 25)),
            )]
        );
        assert_eq!(
            base_names(&unit.root.children[0].children[0]),
            [("B", Span::new(24, 25))]
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_marks_dangling_qualification_unsupported_without_dropping_legacy_name() {
        let code = "class D : A. {}";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [observed_base(
                Some((6, 7)),
                &[("A", 10, 11), (".", 11, 12)],
                None,
            )]
        );
        assert_eq!(
            base_names(&unit.root.children[0]),
            [("A", Span::new(10, 11))]
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_keeps_closed_leading_generics_together_before_the_next_base() {
        let code = "class D : <A,B>, I {}";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[
                        ("<", 10, 11),
                        ("A", 11, 12),
                        (",", 12, 13),
                        ("B", 13, 14),
                        (">", 14, 15),
                    ],
                    None,
                ),
                observed_base(Some((6, 7)), &[("I", 17, 18)], Some((17, 18))),
            ]
        );
        assert_eq!(
            base_names(&unit.root.children[0]),
            [
                ("A", Span::new(11, 12)),
                ("B", Span::new(13, 14)),
                ("I", Span::new(17, 18)),
            ],
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_keeps_unclosed_leading_generics_unsupported_up_to_the_body() {
        let code = "class D : <A,B {}";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [observed_base(
                Some((6, 7)),
                &[("<", 10, 11), ("A", 11, 12), (",", 12, 13), ("B", 13, 14)],
                None,
            )]
        );
        assert_eq!(
            base_names(&unit.root.children[0]),
            [("A", Span::new(11, 12)), ("B", Span::new(13, 14))],
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_retains_qualified_where_and_following_base_after_legacy_stop() {
        let code = "class D : where.Base, I {}";

        let (observed_unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[("where", 10, 15), (".", 15, 16), ("Base", 16, 20)],
                    Some((16, 20)),
                ),
                observed_base(Some((6, 7)), &[("I", 22, 23)], Some((22, 23))),
            ],
        );
        assert_eq!(
            (
                base_names(&observed_unit.root.children[0]),
                &observed_unit.errors[..]
            ),
            (vec![], &[][..]),
        );
        assert_eq!(observed_unit, unit(code));
    }

    #[test]
    fn base_observer_retains_where_inside_an_unsupported_leading_generic() {
        let code = "class D : <A,where>, I {}";

        let (observed_unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[
                        ("<", 10, 11),
                        ("A", 11, 12),
                        (",", 12, 13),
                        ("where", 13, 18),
                        (">", 18, 19),
                    ],
                    None,
                ),
                observed_base(Some((6, 7)), &[("I", 21, 22)], Some((21, 22))),
            ],
        );
        assert_eq!(
            (
                base_names(&observed_unit.root.children[0]),
                &observed_unit.errors[..]
            ),
            (vec![("A", Span::new(11, 12))], &[][..]),
        );
        assert_eq!(observed_unit, unit(code));
    }

    #[test]
    fn base_observer_retains_unqualified_where_but_excludes_outer_constraint() {
        let code = "class D<T> : where, I where /* trivia */ T : X {}";

        let (observed_unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(Some((6, 7)), &[("where", 13, 18)], Some((13, 18))),
                observed_base(Some((6, 7)), &[("I", 20, 21)], Some((20, 21))),
            ],
        );
        assert_eq!(
            (
                base_names(&observed_unit.root.children[0]),
                &observed_unit.errors[..]
            ),
            (vec![], &[][..]),
        );
        assert_eq!(observed_unit, unit(code));
    }

    #[test]
    fn base_observer_does_not_treat_nested_constraint_shape_as_outer_constraint() {
        let code = "class D : <A, where T : X>, I {}";

        let (observed_unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[
                        ("<", 10, 11),
                        ("A", 11, 12),
                        (",", 12, 13),
                        ("where", 14, 19),
                        ("T", 20, 21),
                        (":", 22, 23),
                        ("X", 24, 25),
                        (">", 25, 26),
                    ],
                    None,
                ),
                observed_base(Some((6, 7)), &[("I", 28, 29)], Some((28, 29))),
            ],
        );
        assert_eq!(
            (
                base_names(&observed_unit.root.children[0]),
                &observed_unit.errors[..]
            ),
            (vec![("A", Span::new(11, 12))], &[][..]),
        );
        assert_eq!(observed_unit, unit(code));
    }

    #[test]
    fn base_observer_reports_no_owner_for_a_type_without_a_name() {
        let code = "class : A {}";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(entries, [observed_base(None, &[("A", 8, 9)], Some((8, 9)))]);
        let node = &unit.root.children[0];
        assert_eq!(
            (
                node.declared_name()
                    .map(|name| (name.text.as_str(), name.span)),
                base_names(node),
            ),
            (Some(("", Span::new(6, 7))), vec![("A", Span::new(8, 9))]),
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_keeps_commas_inside_nested_unsupported_delimiters() {
        let code = "class D : [(A,B), C], I {}";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[
                        ("[", 10, 11),
                        ("(", 11, 12),
                        ("A", 12, 13),
                        (",", 13, 14),
                        ("B", 14, 15),
                        (")", 15, 16),
                        (",", 16, 17),
                        ("C", 18, 19),
                        ("]", 19, 20),
                    ],
                    None,
                ),
                observed_base(Some((6, 7)), &[("I", 22, 23)], Some((22, 23))),
            ],
        );
        assert_eq!(
            base_names(&unit.root.children[0]),
            [
                ("A", Span::new(12, 13)),
                ("B", Span::new(14, 15)),
                ("C", Span::new(18, 19)),
                ("I", Span::new(22, 23)),
            ],
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_keeps_generic_commas_inside_entries_and_reports_nested_type_bases() {
        let code = "class D<T> : A<B<X,Y>,Z>, N.I<T> where T : X { class E : Q.R {} }";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[
                        ("A", 13, 14),
                        ("<", 14, 15),
                        ("B", 15, 16),
                        ("<", 16, 17),
                        ("X", 17, 18),
                        (",", 18, 19),
                        ("Y", 19, 20),
                        (">", 20, 21),
                        (",", 21, 22),
                        ("Z", 22, 23),
                        (">", 23, 24),
                    ],
                    Some((13, 14)),
                ),
                observed_base(
                    Some((6, 7)),
                    &[
                        ("N", 26, 27),
                        (".", 27, 28),
                        ("I", 28, 29),
                        ("<", 29, 30),
                        ("T", 30, 31),
                        (">", 31, 32),
                    ],
                    Some((28, 29)),
                ),
                observed_base(
                    Some((53, 54)),
                    &[("Q", 57, 58), (".", 58, 59), ("R", 59, 60)],
                    Some((59, 60)),
                ),
            ]
        );
        let outer = &unit.root.children[0];
        assert_eq!(
            (base_names(outer), base_names(&outer.children[0])),
            (
                vec![("A", Span::new(13, 14)), ("I", Span::new(28, 29))],
                vec![("R", Span::new(59, 60))],
            ),
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_marks_unbalanced_generics_unsupported_and_preserves_nested_recovery() {
        let code = "class D : A<B,C { class E : F {} }";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[
                        ("A", 10, 11),
                        ("<", 11, 12),
                        ("B", 12, 13),
                        (",", 13, 14),
                        ("C", 14, 15),
                    ],
                    None,
                ),
                observed_base(Some((24, 25)), &[("F", 28, 29)], Some((28, 29))),
            ]
        );
        let outer = &unit.root.children[0];
        assert_eq!(
            (base_names(outer), base_names(&outer.children[0])),
            (
                vec![("A", Span::new(10, 11))],
                vec![("F", Span::new(28, 29))]
            ),
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn base_observer_marks_nonterminal_generics_unsupported_without_swallowing_next_base() {
        let code = "class D : Outer<A>.Inner<B>, S {}";

        let (unit, entries) = observe_bases(code, 0);

        assert_eq!(
            entries,
            [
                observed_base(
                    Some((6, 7)),
                    &[
                        ("Outer", 10, 15),
                        ("<", 15, 16),
                        ("A", 16, 17),
                        (">", 17, 18),
                        (".", 18, 19),
                        ("Inner", 19, 24),
                        ("<", 24, 25),
                        ("B", 25, 26),
                        (">", 26, 27),
                    ],
                    None,
                ),
                observed_base(Some((6, 7)), &[("S", 29, 30)], Some((29, 30))),
            ]
        );
        assert_eq!(
            base_names(&unit.root.children[0]),
            [("Outer", Span::new(10, 15)), ("S", Span::new(29, 30))]
        );
        assert_eq!(unit.errors, []);
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

    fn nested_types(count: usize) -> String {
        format!(
            "{}[A] public class Leaf {{}}{}class Sibling {{}} }} class Following {{}}",
            "class N { ".repeat(count - 1),
            "}".repeat(count - 2),
        )
    }

    fn control_prefix(types: usize, blocks: usize, controls: usize) -> String {
        format!(
            "{}void M() {{ {}{}",
            "class N { ".repeat(types),
            "{ ".repeat(blocks),
            "if (ok) ".repeat(controls),
        )
    }

    fn control_fixture(types: usize, blocks: usize, controls: usize, tail: &str) -> String {
        format!(
            "{}{}{} }} void Later() {{}}{}class Following {{}}",
            control_prefix(types, blocks, controls),
            tail,
            "} ".repeat(blocks),
            "} ".repeat(types),
        )
    }

    fn assert_control_fixture(src: &str, names: &[(&str, Span)], errors: &[Span]) -> CsUnit {
        let u = cs_parse(src, 37);

        let mut retained_names = Vec::new();
        let mut error_nodes = Vec::new();
        u.root.walk(&mut |n| {
            if let Some(name) = n.declared_name() {
                // Repeated enclosing types are scaffolding; retain all the
                // distinct declarations, including those after recovery.
                if name.text != "N" {
                    retained_names.push((name.text.clone(), name.span));
                }
            }
            if n.kind == CsNodeKind::Error {
                error_nodes.push((n.span, n.children.len()));
            }
        });
        assert_eq!(u.errors, errors);
        assert_eq!(
            retained_names,
            names
                .iter()
                .map(|&(name, span)| (name.to_string(), span))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            error_nodes,
            errors.iter().map(|&span| (span, 0)).collect::<Vec<_>>(),
        );
        // Public walk and ordinary drop by the caller exercise the bounded tree
        // too, without a larger stack or a different AST representation.
        u
    }

    #[test]
    fn cs_parse_control_limit_shares_type_depth_and_restores_it_for_next_chain() {
        let src = control_fixture(
            195,
            0,
            3,
            "int hidden; if (ok) if (ok) int retry; int after;",
        );

        assert_control_fixture(
            &src,
            &[
                ("M", Span::new(1992, 1993)),
                ("retry", Span::new(2054, 2059)),
                ("after", Span::new(2065, 2070)),
                ("Later", Span::new(2079, 2084)),
                ("Following", Span::new(2485, 2494)),
            ],
            &[Span::new(2014, 2033)],
        );
    }

    #[test]
    fn cs_parse_brace_free_control_chain_at_boundary_preserves_following_declarations() {
        type BoundaryCase<'a> = (usize, &'a [(&'a str, Span)], &'a [Span]);
        let cases: &[BoundaryCase<'_>] = &[
            (
                196,
                &[
                    ("M", Span::new(52, 53)),
                    ("kept", Span::new(1630, 1634)),
                    ("after", Span::new(1640, 1645)),
                    ("Later", Span::new(1654, 1659)),
                    ("Following", Span::new(1672, 1681)),
                ],
                &[],
            ),
            (
                197,
                &[
                    ("M", Span::new(52, 53)),
                    ("after", Span::new(1648, 1653)),
                    ("Later", Span::new(1662, 1667)),
                    ("Following", Span::new(1680, 1689)),
                ],
                &[Span::new(1626, 1643)],
            ),
            (
                198,
                &[
                    ("M", Span::new(52, 53)),
                    ("after", Span::new(1656, 1661)),
                    ("Later", Span::new(1670, 1675)),
                    ("Following", Span::new(1688, 1697)),
                ],
                &[Span::new(1626, 1651)],
            ),
        ];

        for &(controls, names, errors) in cases {
            let src = control_fixture(1, 0, controls, "int kept; int after;");
            assert_control_fixture(&src, names, errors);
        }
    }

    #[test]
    fn cs_parse_mixed_type_block_control_limit_recovers_balanced_tokens_not_literal_braces() {
        let src = control_fixture(
            95,
            50,
            53,
            "{ string hidden = \"};雪\"; /* } { */ char c = '}'; } int after;",
        );

        assert_control_fixture(
            &src,
            &[
                ("M", Span::new(992, 993)),
                ("after", Span::new(1579, 1584)),
                ("Later", Span::new(1693, 1698)),
                ("Following", Span::new(1899, 1908)),
            ],
            &[Span::new(1514, 1574)],
        );
    }

    #[test]
    fn cs_parse_control_body_block_below_limit_retains_locals_without_double_counting() {
        let src = control_fixture(195, 0, 1, "{ int kept; } int after;");

        assert_control_fixture(
            &src,
            &[
                ("M", Span::new(1992, 1993)),
                ("kept", Span::new(2012, 2016)),
                ("after", Span::new(2024, 2029)),
                ("Later", Span::new(2038, 2043)),
                ("Following", Span::new(2444, 2453)),
            ],
            &[],
        );
    }

    #[test]
    fn cs_parse_control_body_block_at_limit_keeps_existing_block_skip_without_error() {
        let src = control_fixture(195, 0, 2, "{ int kept; } int after;");

        let u = assert_control_fixture(
            &src,
            &[
                ("M", Span::new(1992, 1993)),
                ("after", Span::new(2032, 2037)),
                ("Later", Span::new(2046, 2051)),
                ("Following", Span::new(2452, 2461)),
            ],
            &[],
        );
        let mut empty_blocks = Vec::new();
        u.root.walk(&mut |n| {
            if n.kind == CsNodeKind::Block && n.children.is_empty() {
                empty_blocks.push(n.span);
            }
        });
        assert_eq!(empty_blocks, [Span::new(2014, 2027), Span::new(2054, 2056)],);
    }

    #[test]
    fn cs_parse_over_limit_control_without_body_preserves_enclosing_brace_and_later_method() {
        let src = control_fixture(195, 0, 3, "");

        assert_control_fixture(
            &src,
            &[
                ("M", Span::new(1992, 1993)),
                ("Later", Span::new(2030, 2035)),
                ("Following", Span::new(2436, 2445)),
            ],
            &[Span::new(2014, 2021)],
        );
    }

    #[test]
    fn cs_parse_over_limit_control_at_malformed_eof_emits_one_token_bounded_error() {
        let cases = [
            ("(", Span::new(2014, 2023)),
            ("{ string hidden = \"}雪\"; /* { */", Span::new(2014, 2047)),
        ];

        for (tail, error) in cases {
            let src = format!("{}{tail}", control_prefix(195, 0, 3));
            assert_control_fixture(&src, &[("M", Span::new(1992, 1993))], &[error]);
        }
    }

    fn assert_nested_type_boundary(count: usize, declarations: usize, errors: &[Span]) {
        let src = nested_types(count);

        let u = cs_parse(&src, 37);

        assert_eq!(u.errors, errors, "nested types: {count}");
        let mut error_nodes = Vec::new();
        u.root.walk(&mut |n| {
            if n.kind == CsNodeKind::Error {
                error_nodes.push(n.span);
            }
        });
        assert_eq!(
            (decls(&u).len(), error_nodes.as_slice()),
            (declarations, errors),
            "nested types: {count}"
        );
        assert_eq!(
            (
                u.root
                    .children
                    .iter()
                    .map(|n| n.declared_name().map(|name| name.text.as_str()))
                    .collect::<Vec<_>>(),
                u.root.children[0]
                    .children
                    .iter()
                    .map(|n| n.declared_name().map(|name| name.text.as_str()))
                    .collect::<Vec<_>>(),
            ),
            (
                vec![Some("N"), Some("Following")],
                vec![Some("N"), Some("Sibling")],
            ),
            "nested types: {count}"
        );
    }

    #[test]
    fn cs_parse_nested_types_at_depth_boundary_bounds_errors_and_preserves_siblings() {
        let cases: &[(usize, usize, &[Span])] = &[
            (198, 200, &[]),
            (199, 201, &[]),
            (200, 201, &[Span::new(2027, 2051)]),
            (201, 201, &[Span::new(2027, 2062)]),
        ];

        for &(count, declarations, errors) in cases {
            assert_nested_type_boundary(count, declarations, errors);
        }
    }

    #[test]
    fn cs_parse_over_limit_type_at_eof_emits_one_error_from_declaration_start() {
        let src = format!(
            "{}[A] public class Leaf {{ class Body {{",
            "class N { ".repeat(199),
        );

        let u = cs_parse(&src, 37);

        assert_eq!(u.errors, [Span::new(2027, 2063)]);
        assert_eq!(decls(&u).len(), 199);
        let mut error_nodes = Vec::new();
        u.root.walk(&mut |n| {
            if n.kind == CsNodeKind::Error {
                error_nodes.push((n.span, n.children.len()));
            }
        });
        assert_eq!(error_nodes, [(Span::new(2027, 2063), 0)]);
    }

    #[test]
    fn cs_parse_over_limit_type_restores_depth_for_near_limit_sibling() {
        let src = format!(
            "{}[A] public class Leaf {{}} }} class Sibling {{}}{} class Following {{}}",
            "class N { ".repeat(199),
            "}".repeat(198),
        );

        let u = cs_parse(&src, 37);

        assert_eq!(u.errors, [Span::new(2027, 2051)]);
        assert_eq!(decls(&u).len(), 201);
        let mut sibling_spans = Vec::new();
        u.root.walk(&mut |n| {
            if let Some(name) = n.declared_name() {
                if name.text == "Sibling" {
                    sibling_spans.push(name.span);
                }
            }
        });
        assert_eq!(sibling_spans, [Span::new(2060, 2067)]);
    }

    #[test]
    fn cs_parse_over_limit_type_without_body_preserves_enclosing_brace() {
        let src = nested_types(200).replace("Leaf {}", "Leaf");

        let u = cs_parse(&src, 37);

        assert_eq!(u.errors, [Span::new(2027, 2048)]);
        assert_eq!(decls(&u).len(), 201);
        assert_eq!(
            u.root.children[0].children[1]
                .declared_name()
                .map(|name| name.text.as_str()),
            Some("Sibling"),
        );
    }

    #[test]
    fn cs_parse_nested_types_share_depth_limit_with_members() {
        let src = nested_types(199).replace("Leaf {}", "Leaf { void M() {} }");

        let u = cs_parse(&src, 37);

        assert_eq!(u.errors, [Span::new(2041, 2052)]);
        assert_eq!(decls(&u).len(), 201);
    }

    #[test]
    fn cs_parse_nested_types_share_depth_limit_with_blocks() {
        let src = nested_types(198).replace("Leaf {}", "Leaf { void M() { int hidden; } }");

        let u = cs_parse(&src, 37);

        assert_eq!(u.errors, []);
        assert_eq!(decls(&u).len(), 201);
        let mut blocks = Vec::new();
        u.root.walk(&mut |n| {
            if n.kind == CsNodeKind::Block {
                blocks.push((n.span, n.children.len()));
            }
        });
        assert_eq!(blocks, [(Span::new(2040, 2055), 0)]);
    }

    fn assert_lexical_block_depth_refusal(src: &str, open: Span) {
        let source = format!("{}{src}", " ".repeat(37));
        let region = Span::new(37, source.len());

        let index = index_lexical_declarations(&source, region);

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(open),
                recovery: vec![],
            })
        );
    }

    #[test]
    fn lexical_index_refuses_the_actual_opening_brace_of_a_depth_skipped_block() {
        // Same bounded nesting as the public-parser fixture, without the
        // independent attribute refusal. Exercise a skipped root and child.
        for (count, body, open) in [
            (
                198,
                "Leaf { void M() { int hidden; } }",
                Span::new(2036, 2037),
            ),
            (
                197,
                "Leaf { void M() {{ int hidden; }} }",
                Span::new(2027, 2028),
            ),
        ] {
            let src = nested_types(count)
                .replace("[A] ", "")
                .replace("Leaf {}", body);
            assert_lexical_block_depth_refusal(&src, open);
        }
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
