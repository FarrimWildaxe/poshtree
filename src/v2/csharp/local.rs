//! Parse-local declaration evidence, independent of the legacy AST projection.
//!
//! This slice admits braced `class`, `struct`, `interface`, and `enum`
//! declarations, globally, in root namespaces, or nested in other types.
//! A root namespace can be braced, or a single file-scoped namespace can
//! precede all types and apply through region EOF.
//! In type-only mode, type bodies may contain only nested types (enum bodies
//! must be empty and nongeneric). Other types may declare `<Ident (, Ident)*>`
//! with distinct raw names and no variance or attributes. Names, modifiers, and
//! trivia otherwise follow legacy token rules, not full C# syntax or semantic
//! validation.
//! Callable mode additionally admits direct class and struct members:
//! nongeneric methods returning `void`, `int`, or `bool`, and constructors named
//! exactly like their owner. Both require an empty block, no modifiers or
//! attributes, and only by-value `int` or `bool` parameters. Identifier evidence
//! is preserved as raw tokens.
//! Opt-in generic callable mode also admits methods with distinct unescaped
//! ASCII binders and by-value uses of those binders, but only when neither
//! the owner nor an enclosing type declares type parameters.
//! Qualified callable mode admits `global::` parameter paths with one exact
//! nongeneric target in the same region, and refuses partial type declarations.
//! Lexical mode extends qualified callable mode with single `int Ident;` or
//! `bool Ident;` fields and locals, and nonempty bodies made up of those locals
//! and standalone brace blocks. These are syntactic owners, not name bindings.
//! The separate binding observer uses those owners, admits only simple
//! assignments in addition to the lexical body grammar, and restricts types to
//! no modifier or one `public` modifier.
//! The earlier observation modes, including generic method identities, are
//! test-only. Checked rename adds exact evidence to the lexical index, requires inert
//! trivia, accounts for every significant token, and binds proven instance and
//! explicitly rooted static places. Only checked mode admits a single explicit
//! nongeneric class base: a `global::` path, or a bare name between unambiguous
//! global top-level siblings. Inheritance lookup requires a complete reachable
//! graph, but a direct member does not. This does not certify rename eligibility.
//! Checked bodies also admit standalone zero-argument and single-name calls.
//! A single-name call requires recorded exact type evidence, not inference or
//! conversions; that evidence never selects a winner among overloads.
//! A region-start BOM, other items, attributes, unsupported base lists, and
//! constraints are refused. The legacy indexes give recovery precedence over
//! subset refusals; generic, qualified, lexical, and binding modes report the
//! first subset refusal and ordered recovery separately. Qualified lookup
//! requires a complete parse without either kind of parse refusal.

use super::ast::{CsName, CsTypeKind};
use super::parser::{parse_checked_headers, CheckedHeaderMode};
#[cfg(test)]
use super::parser::{
    parse_generic_headers, parse_headers, parse_lexical_headers, parse_qualified_headers,
};
use crate::v2::span::Span;
use binding::{CheckedReferenceBinding, DeclarationId, Receiver};
#[cfg(test)]
use binding::{ReferenceRole, ReferenceSite, SiteOwner};
use std::collections::{HashMap, HashSet};

pub(super) mod bases;
pub(super) mod binding;

/// Header rows in source order from one parse of the given original source region.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct TypeHeaderIndex {
    pub(super) region: Span,
    pub(super) namespaces: Vec<NamespaceOccurrence>,
    pub(super) rows: Vec<TypeHeaderRow>,
}

impl TypeHeaderIndex {
    /// Starts at the given row and walks outward through exact owner occurrences.
    pub(super) fn type_lineage(
        &self,
        ordinal: Option<usize>,
    ) -> impl Iterator<Item = &TypeHeaderRow> {
        std::iter::successors(ordinal.map(|ordinal| &self.rows[ordinal]), |row| {
            row.containing_type.map(|owner| &self.rows[owner])
        })
    }

    /// Outside the exact same owner, each nested link must be explicitly public.
    fn inaccessible_type_link(&self, from: usize, target: usize) -> Option<usize> {
        if from == target {
            return None;
        }
        self.type_lineage(Some(target))
            .find(|row| row.containing_type.is_some() && row.public_modifier.is_none())
            .map(|row| row.ordinal)
    }
}

/// A source occurrence, not a semantic namespace identity.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct NamespaceOccurrence {
    /// Zero-based within this index only, in source order.
    pub(super) ordinal: usize,
    /// Raw name tokens in declaration order, with absolute byte spans.
    pub(super) segments: Vec<CsName>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct TypeHeaderRow {
    /// Zero-based within this index only; not a stable identity across reparses.
    pub(super) ordinal: usize,
    /// Ordinal of the namespace occurrence, or `None` for global.
    pub(super) namespace: Option<usize>,
    /// Ordinal of the immediate containing type, if any.
    pub(super) containing_type: Option<usize>,
    pub(super) kind: CsTypeKind,
    pub(super) name: CsName,
    /// The checked observer's exact `public` token, if present.
    pub(super) public_modifier: Option<CsName>,
    /// In declaration order; arity is the length of this list.
    pub(super) type_params: Vec<CsName>,
    /// From the kind token to the last significant token before the body.
    pub(super) header_span: Span,
    /// From the first modifier (or kind) through the closing brace.
    pub(super) declaration_span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum QualifiedScope {
    Global,
    Namespace(usize),
    Type(usize),
}

#[derive(Default)]
struct QualifiedEntry<'a> {
    namespace: Option<QualifiedNamespace<'a>>,
    types: Vec<&'a TypeHeaderRow>,
}

impl QualifiedEntry<'_> {
    fn type_candidates(&self) -> impl Iterator<Item = QualifiedTypeCandidate> + '_ {
        self.types.iter().map(|row| QualifiedTypeCandidate::Type {
            ordinal: row.ordinal,
            span: row.name.span,
        })
    }
}

struct QualifiedNamespace<'a> {
    scope: usize,
    declarations: Vec<(usize, &'a CsName)>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct QualifiedTypeBinding {
    prefixes: Vec<QualifiedScope>,
    terminal: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum QualifiedTypeFailure {
    Ambiguous {
        segment: Span,
        candidates: Vec<QualifiedTypeCandidate>,
    },
    AmbiguousTypeMember {
        segment: Span,
        /// Proven scopes before the conflicting segment, not a receiver binding.
        prefixes: Vec<QualifiedScope>,
        candidates: Vec<QualifiedTypeCandidate>,
    },
    MissingType {
        segment: Span,
    },
    UnsupportedArity {
        segment: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum QualifiedTypeCandidate {
    Namespace { occurrence: usize, span: Span },
    Type { ordinal: usize, span: Span },
    Member { id: DeclarationId, span: Span },
}

impl QualifiedTypeCandidate {
    pub(super) fn span(&self) -> Span {
        match *self {
            Self::Namespace { span, .. } | Self::Type { span, .. } | Self::Member { span, .. } => {
                span
            }
        }
    }
}

/// Exact paths within one complete index, without merging any occurrence rows.
struct QualifiedTypeLookup<'a> {
    entries: HashMap<(QualifiedScope, &'a str), QualifiedEntry<'a>>,
}

impl<'a> QualifiedTypeLookup<'a> {
    fn new(index: &'a TypeHeaderIndex) -> Self {
        let mut entries = HashMap::<_, QualifiedEntry<'_>>::new();
        let mut namespace_scopes = Vec::with_capacity(index.namespaces.len());
        let mut next_namespace = 0;
        for occurrence in &index.namespaces {
            let mut scope = QualifiedScope::Global;
            for segment in &occurrence.segments {
                let entry = entries
                    .entry((scope, comparison_identifier(&segment.text)))
                    .or_default();
                // Reopenings share the same lookup scope, including dotted prefixes.
                let namespace = entry.namespace.get_or_insert_with(|| {
                    let scope = next_namespace;
                    next_namespace += 1;
                    QualifiedNamespace {
                        scope,
                        declarations: Vec::new(),
                    }
                });
                namespace.declarations.push((occurrence.ordinal, segment));
                scope = QualifiedScope::Namespace(namespace.scope);
            }
            namespace_scopes.push(scope);
        }

        for row in &index.rows {
            let scope = if let Some(owner) = row.containing_type {
                QualifiedScope::Type(owner)
            } else {
                row.namespace
                    .map_or(QualifiedScope::Global, |ordinal| namespace_scopes[ordinal])
            };
            // A generic occurrence still competes for the name, regardless of arity.
            entries
                .entry((scope, comparison_identifier(&row.name.text)))
                .or_default()
                .types
                .push(row);
        }
        Self { entries }
    }

    /// Looks up raw identifier segments from the global scope, not a qualifier token.
    #[cfg(test)]
    fn resolve(&self, segments: &[&str]) -> Option<usize> {
        let (terminal, prefixes) = segments.split_last()?;
        let mut scope = QualifiedScope::Global;
        for segment in prefixes {
            let entry = self.entries.get(&(scope, comparison_identifier(segment)))?;
            scope = match (&entry.namespace, entry.types.as_slice()) {
                (Some(namespace), []) => QualifiedScope::Namespace(namespace.scope),
                (None, [row]) if row.type_params.is_empty() => QualifiedScope::Type(row.ordinal),
                _ => return None,
            };
        }

        let entry = self
            .entries
            .get(&(scope, comparison_identifier(terminal)))?;
        match (&entry.namespace, entry.types.as_slice()) {
            (None, [row]) if row.type_params.is_empty() => match row.kind {
                // Complete indexes already refuse nonempty enum bodies.
                CsTypeKind::Class
                | CsTypeKind::Struct
                | CsTypeKind::Interface
                | CsTypeKind::Enum => Some(row.ordinal),
            },
            _ => None,
        }
    }

    /// In a type position, only namespaces and types compete.
    fn resolve_checked(
        &self,
        path: &QualifiedTypeUse,
    ) -> Result<QualifiedTypeBinding, QualifiedTypeFailure> {
        self.resolve_checked_path(path, |_, _| Vec::new())
    }

    /// A single type name in the global scope, without fabricating a root token.
    fn resolve_global_type(&self, name: &CsName) -> Result<usize, QualifiedTypeFailure> {
        let missing = || QualifiedTypeFailure::MissingType { segment: name.span };
        let entry = self
            .entries
            .get(&(QualifiedScope::Global, comparison_identifier(&name.text)))
            .ok_or_else(missing)?;
        match Self::bind_segment(entry, name.span)? {
            QualifiedScope::Type(ordinal) => Ok(ordinal),
            _ => Err(missing()),
        }
    }

    /// Stops at the first unproven segment. Rooted expression lookup supplies
    /// the direct non-type competitors in each containing type's scope.
    fn resolve_checked_path(
        &self,
        path: &QualifiedTypeUse,
        competing_members: impl Fn(usize, &str) -> Vec<QualifiedTypeCandidate>,
    ) -> Result<QualifiedTypeBinding, QualifiedTypeFailure> {
        self.resolve_checked_segments(path.segments.iter(), path.root.span, competing_members)
    }

    fn resolve_checked_segments<'n>(
        &self,
        mut segments: impl DoubleEndedIterator<Item = &'n CsName>,
        fallback: Span,
        competing_members: impl Fn(usize, &str) -> Vec<QualifiedTypeCandidate>,
    ) -> Result<QualifiedTypeBinding, QualifiedTypeFailure> {
        let terminal = segments
            .next_back()
            .ok_or(QualifiedTypeFailure::MissingType { segment: fallback })?;
        let bind = |scope, segment: &CsName, prefixes: &[QualifiedScope]| {
            let spelling = comparison_identifier(&segment.text);
            let span = segment.span;
            let entry = self
                .entries
                .get(&(scope, spelling))
                .ok_or(QualifiedTypeFailure::MissingType { segment: span })?;
            if let QualifiedScope::Type(owner) = scope {
                let mut candidates = competing_members(owner, spelling);
                if !candidates.is_empty() {
                    // A later leaf, member kind, or visibility cannot
                    // disambiguate this segment in an expression.
                    candidates.extend(entry.type_candidates());
                    candidates.sort_unstable_by_key(|candidate| candidate.span().start);
                    return Err(QualifiedTypeFailure::AmbiguousTypeMember {
                        segment: span,
                        prefixes: prefixes.to_vec(),
                        candidates,
                    });
                }
            }
            Self::bind_segment(entry, span)
        };
        let mut scope = QualifiedScope::Global;
        let mut bound_prefixes = Vec::with_capacity(segments.size_hint().0);
        for segment in segments {
            scope = bind(scope, segment, &bound_prefixes)?;
            bound_prefixes.push(scope);
        }
        match bind(scope, terminal, &bound_prefixes)? {
            QualifiedScope::Type(ordinal) => Ok(QualifiedTypeBinding {
                prefixes: bound_prefixes,
                terminal: ordinal,
            }),
            _ => Err(QualifiedTypeFailure::MissingType {
                segment: terminal.span,
            }),
        }
    }

    fn bind_segment(
        entry: &QualifiedEntry<'_>,
        span: Span,
    ) -> Result<QualifiedScope, QualifiedTypeFailure> {
        let namespace = entry.namespace.as_ref();
        let types = entry.types.as_slice();
        let collision = namespace.is_some() && !types.is_empty();
        // Generic declarations compete but this subset never binds a generic
        // type, including when a nongeneric declaration has the same name.
        if !collision && types.iter().any(|row| !row.type_params.is_empty()) {
            return Err(QualifiedTypeFailure::UnsupportedArity { segment: span });
        }
        match (namespace, types) {
            (Some(namespace), []) => Ok(QualifiedScope::Namespace(namespace.scope)),
            (None, [row]) => Ok(QualifiedScope::Type(row.ordinal)),
            _ => {
                let mut candidates: Vec<_> = entry
                    .type_candidates()
                    .chain(
                        namespace
                            .into_iter()
                            .flat_map(|namespace| &namespace.declarations)
                            .map(|&(occurrence, name)| QualifiedTypeCandidate::Namespace {
                                occurrence,
                                span: name.span,
                            }),
                    )
                    .collect();
                candidates.sort_unstable_by_key(|candidate| candidate.span().start);
                Err(QualifiedTypeFailure::Ambiguous {
                    segment: span,
                    candidates,
                })
            }
        }
    }
}

pub(super) fn comparison_identifier(segment: &str) -> &str {
    segment.strip_prefix('@').unwrap_or(segment)
}

/// The existing namespace and type rows plus callable occurrences in source order.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CallableHeaderIndex {
    pub(super) types: TypeHeaderIndex,
    pub(super) callables: Vec<CallableHeaderRow>,
}

impl CallableHeaderIndex {
    /// Borrows the original parameter evidence in callable and parameter source
    /// order. Each tuple carries the callable occurrence ordinal and zero-based
    /// position in that callable, regardless of signature equality.
    pub(super) fn parameter_occurrences(
        &self,
    ) -> impl Iterator<Item = (usize, usize, &CallableParameter)> + '_ {
        self.callables.iter().flat_map(|row| {
            row.parameters
                .iter()
                .enumerate()
                .map(move |(position, parameter)| (row.ordinal, position, parameter))
        })
    }
}

/// Evidence from one complete parse. All ordinals expire on reparse.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct LexicalDeclarationIndex {
    pub(super) headers: CallableHeaderIndex,
    pub(super) declarations: LexicalDeclarations,
}

/// Occurrences in source order (opening-brace order for blocks).
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct LexicalDeclarations {
    pub(super) fields: Vec<FieldOccurrence>,
    pub(super) blocks: Vec<BlockOccurrence>,
    pub(super) locals: Vec<LocalOccurrence>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct FieldOccurrence {
    /// Ordinal of the immediate class or struct.
    pub(super) containing_type: usize,
    pub(super) type_use: CsName,
    pub(super) name: CsName,
    pub(super) modifiers: MemberModifiers,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct MemberModifiers {
    pub(super) public: Option<CsName>,
    pub(super) static_token: Option<CsName>,
}

impl MemberModifiers {
    pub(super) fn tokens(&self) -> impl Iterator<Item = &CsName> {
        self.public.iter().chain(&self.static_token)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct BlockOccurrence {
    pub(super) callable: usize,
    /// Zero-based within this callable, with the body root at zero.
    pub(super) ordinal: usize,
    /// Block ordinal in the same callable, or `None` for the body root.
    pub(super) parent: Option<usize>,
    /// From the actual opening brace through the actual closing brace.
    pub(super) span: Span,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct LocalOccurrence {
    pub(super) callable: usize,
    /// Ordinal of the immediate brace block within `callable`.
    pub(super) block: usize,
    pub(super) type_use: CsName,
    pub(super) name: CsName,
}

/// Syntactic evidence only; qualified parameters have not been resolved.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct PendingQualifiedCallableIndex {
    pub(super) types: TypeHeaderIndex,
    pub(super) callables: Vec<CallableHeaderRow<PendingCallableParameter>>,
}

#[cfg(test)]
impl PendingQualifiedCallableIndex {
    fn resolve(self) -> Result<CallableHeaderIndex, GenericCallableIndexError> {
        let Self { types, callables } = self;
        let lookup = QualifiedTypeLookup::new(&types);
        let callables = callables
            .into_iter()
            .map(|row| {
                row.resolve(|path| {
                    let segments: Vec<_> = path
                        .segments
                        .iter()
                        .map(|segment| segment.text.as_str())
                        .collect();
                    lookup.resolve(&segments).ok_or(path.span)
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|span| GenericCallableIndexError::Rejected {
                first_unsupported: Some(span),
                recovery: Vec::new(),
            })?;
        Ok(CallableHeaderIndex { types, callables })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CallableKind {
    Method,
    Constructor,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CallableParameter {
    /// The original type-use evidence, with absolute byte spans.
    pub(super) type_use: CallableParameterTypeUse,
    pub(super) comparison_type: CallableParameterType,
    pub(super) name: CsName,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum CallableParameterTypeUse {
    Simple(CsName),
    Qualified(QualifiedTypeUse),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum CallableParameterType {
    Int,
    Bool,
    /// Zero-based binder ordinal in this method's declaration order.
    #[cfg(test)]
    MethodSlot(usize),
    /// Ordinal of the unique target type row, valid only in this parse.
    LocalType(usize),
}

impl CallableParameterType {
    pub(super) fn primitive(spelling: &str) -> Option<Self> {
        match spelling {
            "int" => Some(Self::Int),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum PendingCallableParameter {
    Simple(CallableParameter),
    Qualified {
        type_use: QualifiedTypeUse,
        name: CsName,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct QualifiedTypeUse {
    /// The exact unescaped `global` token.
    pub(super) root: CsName,
    /// Raw path tokens after `::`, without spelling transformations.
    pub(super) segments: Vec<CsName>,
    /// From the root through the final segment, including intervening trivia.
    pub(super) span: Span,
}

impl QualifiedTypeUse {
    pub(super) fn new(root: CsName, segments: Vec<CsName>) -> Self {
        let terminal = segments.last().expect("validated path has a terminal");
        let span = Span::new(root.span.start, terminal.span.end);
        Self {
            root,
            segments,
            span,
        }
    }
}

impl PendingCallableParameter {
    fn resolve<E>(
        self,
        resolve_type: impl FnOnce(&QualifiedTypeUse) -> Result<usize, E>,
    ) -> Result<CallableParameter, E> {
        match self {
            Self::Simple(parameter) => Ok(parameter),
            Self::Qualified { type_use, name } => {
                let ordinal = resolve_type(&type_use)?;
                Ok(CallableParameter {
                    type_use: CallableParameterTypeUse::Qualified(type_use),
                    comparison_type: CallableParameterType::LocalType(ordinal),
                    name,
                })
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CallableHeaderRow<Parameter = CallableParameter> {
    /// Zero-based callable occurrence within this index only.
    pub(super) ordinal: usize,
    /// Ordinal of the immediate containing type in this index only.
    pub(super) containing_type: usize,
    pub(super) kind: CallableKind,
    pub(super) name: CsName,
    /// The exact return keyword token, or `None` for a constructor.
    pub(super) return_type: Option<CsName>,
    pub(super) modifiers: MemberModifiers,
    /// The original method binder tokens in declaration order.
    #[cfg(test)]
    pub(super) type_params: Vec<CsName>,
    pub(super) parameters: Vec<Parameter>,
    /// From the first significant token through the closing parenthesis.
    pub(super) header_span: Span,
    /// From the first significant token through the closing body brace.
    pub(super) declaration_span: Span,
}

/// A comparison key within one parse, not an occurrence ID or a cross-parse handle.
/// Equal keys do not merge rows.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CallableKey<'a> {
    containing_type: usize,
    kind: CallableKind,
    name: &'a str,
    arity: usize,
    /// Parameter comparison types in declaration order.
    parameter_types: Vec<CallableParameterType>,
}

impl CallableHeaderRow {
    /// Compares headers without return type, parameter names, spans, or occurrence ordinal.
    #[cfg(test)]
    pub(super) fn key(&self) -> CallableKey<'_> {
        CallableKey {
            containing_type: self.containing_type,
            kind: self.kind,
            name: &self.name.text,
            arity: self.type_params.len(),
            parameter_types: self.parameter_types().collect(),
        }
    }

    pub(super) fn parameter_types(&self) -> impl Iterator<Item = CallableParameterType> + '_ {
        self.parameters
            .iter()
            .map(|parameter| parameter.comparison_type)
    }
}

impl CallableHeaderRow<PendingCallableParameter> {
    fn resolve<E>(
        self,
        resolve_type: impl Fn(&QualifiedTypeUse) -> Result<usize, E>,
    ) -> Result<CallableHeaderRow, E> {
        let parameters = self
            .parameters
            .into_iter()
            .map(|parameter| parameter.resolve(&resolve_type))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CallableHeaderRow {
            ordinal: self.ordinal,
            containing_type: self.containing_type,
            kind: self.kind,
            name: self.name,
            return_type: self.return_type,
            modifiers: self.modifiers,
            #[cfg(test)]
            type_params: self.type_params,
            parameters,
            header_span: self.header_span,
            declaration_span: self.declaration_span,
        })
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(super) enum TypeIndexError {
    InvalidRegion(Span),
    /// The legacy parser's diagnostics in their original order.
    Recovered(Vec<Span>),
    /// The first unsupported span, possibly at EOF.
    UnsupportedSyntax(Span),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum GenericCallableIndexError {
    InvalidRegion(Span),
    DepthLimit {
        at: Span,
        recovery: Vec<Span>,
    },
    CheckedParameterType {
        use_span: Span,
        failure: QualifiedTypeFailure,
    },
    Rejected {
        /// The first subset refusal (possibly at EOF), or the full use span
        /// of the first failed qualified lookup after a syntax-clean parse.
        first_unsupported: Option<Span>,
        /// All diagnostics from the opt-in parse, in their original order.
        recovery: Vec<Span>,
    },
}

/// Indexes exactly `region`, with absolute byte spans and no partial result.
#[cfg(test)]
pub(super) fn index_type_headers(
    source: &str,
    region: Span,
) -> Result<TypeHeaderIndex, TypeIndexError> {
    let code = source
        .get(region.start..region.end)
        .ok_or(TypeIndexError::InvalidRegion(region))?;
    parse_headers(code, region, None)
}

/// Indexes exactly `region` in callable mode, with absolute byte spans and no partial result.
#[cfg(test)]
pub(super) fn index_callable_headers(
    source: &str,
    region: Span,
) -> Result<CallableHeaderIndex, TypeIndexError> {
    let code = source
        .get(region.start..region.end)
        .ok_or(TypeIndexError::InvalidRegion(region))?;
    let mut callables = Vec::new();
    let types = parse_headers(code, region, Some(&mut callables))?;
    Ok(CallableHeaderIndex { types, callables })
}

/// Indexes exactly `region` with generic methods enabled and no partial result.
#[cfg(test)]
pub(super) fn index_generic_callable_headers(
    source: &str,
    region: Span,
) -> Result<CallableHeaderIndex, GenericCallableIndexError> {
    let code = source
        .get(region.start..region.end)
        .ok_or(GenericCallableIndexError::InvalidRegion(region))?;
    parse_generic_headers(code, region)
}

/// Indexes nongeneric methods and constructors in `region`, with absolute spans.
///
/// Only a complete, recovery-free parse can resolve `global::` parameter paths
/// to type row ordinals from this region. Syntax refusals and ordered recovery
/// take precedence. Otherwise, the first failed lookup in callable and parameter
/// source order refuses the whole index at the full type-use span.
#[cfg(test)]
pub(super) fn index_qualified_callable_headers(
    source: &str,
    region: Span,
) -> Result<CallableHeaderIndex, GenericCallableIndexError> {
    let code = source
        .get(region.start..region.end)
        .ok_or(GenericCallableIndexError::InvalidRegion(region))?;
    parse_qualified_headers(code, region)?.resolve()
}

/// Indexes declaration ownership in exactly `region`, without partial results.
/// Qualified parameters are resolved only after all syntax has been accepted.
#[cfg(test)]
pub(super) fn index_lexical_declarations(
    source: &str,
    region: Span,
) -> Result<LexicalDeclarationIndex, GenericCallableIndexError> {
    let code = source
        .get(region.start..region.end)
        .ok_or(GenericCallableIndexError::InvalidRegion(region))?;
    let (headers, declarations) = parse_lexical_headers(code, region)?;
    Ok(LexicalDeclarationIndex {
        headers: headers.resolve()?,
        declarations,
    })
}

/// Evidence and bindings from a single complete checked observation.
/// Its occurrence IDs expire on reparse.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CheckedIndex {
    pub(super) lexical: LexicalDeclarationIndex,
    pub(super) references: Vec<CheckedReferenceBinding>,
    bases: bases::BaseIndex,
    base_identifiers: Vec<Span>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeclarationRole {
    #[cfg(test)]
    Type,
    Callable(CallableKind),
    Field,
    #[cfg(test)]
    Parameter,
    #[cfg(test)]
    Local,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CheckedDeclaration<'a> {
    pub(super) id: DeclarationId,
    pub(super) name: &'a CsName,
    #[cfg(test)]
    namespace_occurrence: Option<usize>,
    #[cfg(test)]
    key: CheckedDeclarationKey<'a>,
}

/// Comparison within one parse. Equal keys never combine declaration rows.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
struct CheckedDeclarationKey<'a> {
    namespace_path: Vec<&'a str>,
    containing_type: Option<usize>,
    /// Outermost type through immediate containing type.
    type_lineage: Vec<(&'a str, CsTypeKind, usize)>,
    name: &'a str,
    shape: DeclarationShape,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
enum DeclarationShape {
    Type(CsTypeKind, usize),
    Callable {
        kind: CallableKind,
        arity: usize,
        parameters: Vec<CallableParameterType>,
    },
    Field,
    Parameter(usize),
    Local(SiteOwner),
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EvidenceRole {
    NamespaceSegment,
    TypeBinder,
    MethodBinder,
    ReturnType,
    ParameterType,
    QualifiedParameterRoot,
    QualifiedParameterSegment,
    QualifiedReceiverRoot,
    QualifiedReceiverPrefix,
    QualifiedReceiverTerminal,
    MemberModifier,
    FieldType,
    LocalType,
    Reference(ReferenceRole),
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(super) enum CheckedEvidence<'a> {
    Name(&'a CsName),
    ParameterType(&'a CallableParameter),
    Reference(&'a ReferenceSite),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum RenameBaseFailure {
    Cycle { span: Span, from: usize, to: usize },
    Incomplete { span: Span, from: usize },
}

impl CheckedIndex {
    /// Checks every explicit edge and reports the other owners connected to
    /// `owner` by ancestry (not siblings sharing a base).
    pub(super) fn rename_base_relatives(
        &self,
        owner: usize,
    ) -> Result<HashSet<usize>, RenameBaseFailure> {
        let mut related = HashSet::new();
        if self.base_identifiers.is_empty() {
            return Ok(related);
        }
        for row in &self.lexical.headers.types.rows {
            let relatives = self
                .bases
                .graph
                .nearest_ancestors(row.ordinal, |ancestor| {
                    if row.ordinal == owner {
                        // No level stops the walk when collecting the owner's ancestors.
                        related.insert(ancestor);
                        None
                    } else {
                        (ancestor == owner).then_some(row.ordinal)
                    }
                })
                .map_err(|error| match error {
                    bases::BaseLookupError::BaseCycle { span, from, to } => {
                        RenameBaseFailure::Cycle { span, from, to }
                    }
                    bases::BaseLookupError::IncompleteBase { span, from, .. } => {
                        RenameBaseFailure::Incomplete { span, from }
                    }
                })?;
            related.extend(relatives);
        }
        Ok(related)
    }

    /// Verifies that a nongeneric owner and each of its enclosing types are
    /// unambiguous in their namespace or containing type.
    pub(super) fn check_rename_owner(&self, owner: usize) -> Result<(), QualifiedTypeFailure> {
        let types = &self.lexical.headers.types;
        let row = &types.rows[owner];
        let namespace = row
            .namespace
            .iter()
            .flat_map(|&ordinal| &types.namespaces[ordinal].segments);
        let lineage: Vec<_> = types
            .type_lineage(Some(owner))
            .map(|row| &row.name)
            .collect();
        let path = namespace.chain(lineage.into_iter().rev());
        let bound = QualifiedTypeLookup::new(types).resolve_checked_segments(
            path,
            row.name.span,
            |_, _| Vec::new(),
        )?;
        (bound.terminal == owner)
            .then_some(())
            .ok_or(QualifiedTypeFailure::MissingType {
                segment: row.name.span,
            })
    }

    /// Identifier roles only; the parser's keywords, modifiers, primitive
    /// type tokens and `this`/`base`/`global::` introducers are not names.
    pub(super) fn identifier_spans(&self) -> impl Iterator<Item = Span> + '_ {
        let headers = &self.lexical.headers;
        let declarations = &self.lexical.declarations;
        let namespaces = headers.types.namespaces.iter().flat_map(|ns| &ns.segments);
        let types = headers
            .types
            .rows
            .iter()
            .flat_map(|row| std::iter::once(&row.name).chain(&row.type_params));
        #[cfg(test)]
        let callables = headers
            .callables
            .iter()
            .flat_map(|row| std::iter::once(&row.name).chain(&row.type_params));
        #[cfg(not(test))]
        let callables = headers.callables.iter().map(|row| &row.name);
        let parameters = headers.parameter_occurrences().flat_map(|(_, _, row)| {
            let segments = match &row.type_use {
                CallableParameterTypeUse::Qualified(path) => path.segments.as_slice(),
                #[cfg(test)]
                CallableParameterTypeUse::Simple(name)
                    if matches!(row.comparison_type, CallableParameterType::MethodSlot(_)) =>
                {
                    std::slice::from_ref(name)
                }
                CallableParameterTypeUse::Simple(_) => &[],
            };
            std::iter::once(&row.name).chain(segments)
        });
        let references = self.references.iter().flat_map(|reference| {
            let segments = match &reference.site.receiver {
                Receiver::Rooted(path) => path.segments.as_slice(),
                // A named receiver already has its own reference site.
                _ => &[],
            };
            std::iter::once(&reference.site.name).chain(segments)
        });
        namespaces
            .chain(types)
            .chain(callables)
            .chain(parameters)
            .chain(declarations.fields.iter().map(|row| &row.name))
            .chain(declarations.locals.iter().map(|row| &row.name))
            .chain(references)
            .map(|name| name.span)
            .chain(self.base_identifiers.iter().copied())
    }

    /// Finds an exact declaration name span in the requested role only.
    pub(super) fn declaration_at(
        &self,
        span: Span,
        role: DeclarationRole,
    ) -> Option<CheckedDeclaration<'_>> {
        let headers = &self.lexical.headers;
        let (id, name) = match role {
            #[cfg(test)]
            DeclarationRole::Type => {
                let row = headers
                    .types
                    .rows
                    .iter()
                    .find(|row| row.name.span == span)?;
                (DeclarationId::Type(row.ordinal), &row.name)
            }
            DeclarationRole::Callable(kind) => {
                let row = headers
                    .callables
                    .iter()
                    .find(|row| row.kind == kind && row.name.span == span)?;
                (DeclarationId::Callable(row.ordinal), &row.name)
            }
            DeclarationRole::Field => {
                let (id, row) = self
                    .lexical
                    .declarations
                    .field_occurrences()
                    .find(|(_, row)| row.name.span == span)?;
                (id, &row.name)
            }
            #[cfg(test)]
            DeclarationRole::Parameter => {
                let (callable, position, row) = headers
                    .parameter_occurrences()
                    .find(|(_, _, row)| row.name.span == span)?;
                (DeclarationId::Parameter(callable, position), &row.name)
            }
            #[cfg(test)]
            DeclarationRole::Local => {
                let (id, row) = self
                    .lexical
                    .declarations
                    .local_occurrences()
                    .find(|(_, row)| row.name.span == span)?;
                (id, &row.name)
            }
        };
        #[cfg(test)]
        return Some(self.checked_declaration(id, name));
        #[cfg(not(test))]
        Some(CheckedDeclaration { id, name })
    }

    #[cfg(test)]
    fn checked_declaration<'a>(
        &'a self,
        id: DeclarationId,
        name: &'a CsName,
    ) -> CheckedDeclaration<'a> {
        let headers = &self.lexical.headers;
        let types = &headers.types;
        let (scope_type, shape) = match id {
            DeclarationId::Type(ordinal) => {
                let row = &types.rows[ordinal];
                (
                    ordinal,
                    DeclarationShape::Type(row.kind, row.type_params.len()),
                )
            }
            DeclarationId::Callable(ordinal) => {
                let row = &headers.callables[ordinal];
                (
                    row.containing_type,
                    DeclarationShape::Callable {
                        kind: row.kind,
                        arity: row.type_params.len(),
                        parameters: row.parameter_types().collect(),
                    },
                )
            }
            DeclarationId::Field(owner, _) => (owner, DeclarationShape::Field),
            DeclarationId::Parameter(callable, _) => (
                headers.callables[callable].containing_type,
                DeclarationShape::Parameter(callable),
            ),
            DeclarationId::Local(callable, block, _) => (
                headers.callables[callable].containing_type,
                DeclarationShape::Local(SiteOwner { callable, block }),
            ),
        };
        let scope_row = &types.rows[scope_type];
        let containing_type = if matches!(id, DeclarationId::Type(_)) {
            scope_row.containing_type
        } else {
            Some(scope_type)
        };
        let namespace_occurrence = scope_row.namespace;
        let namespace_path = namespace_occurrence
            .iter()
            .flat_map(|&ordinal| &types.namespaces[ordinal].segments)
            .map(|segment| comparison_identifier(&segment.text))
            .collect();
        let mut type_lineage: Vec<_> = types
            .type_lineage(containing_type)
            .map(|row| {
                (
                    comparison_identifier(&row.name.text),
                    row.kind,
                    row.type_params.len(),
                )
            })
            .collect();
        type_lineage.reverse();
        CheckedDeclaration {
            id,
            name,
            namespace_occurrence,
            key: CheckedDeclarationKey {
                namespace_path,
                containing_type,
                type_lineage,
                name: comparison_identifier(&name.text),
                shape,
            },
        }
    }

    pub(super) fn candidate_spans(&self, candidates: &[DeclarationId]) -> Vec<Span> {
        let ids: HashSet<_> = candidates.iter().copied().collect();
        let headers = &self.lexical.headers;
        let declarations = &self.lexical.declarations;
        let mut spans: Vec<_> = declarations
            .field_occurrences()
            .map(|(id, row)| (id, row.name.span))
            .chain(
                declarations
                    .local_occurrences()
                    .map(|(id, row)| (id, row.name.span)),
            )
            .chain(
                headers
                    .parameter_occurrences()
                    .map(|(callable, position, row)| {
                        (DeclarationId::Parameter(callable, position), row.name.span)
                    }),
            )
            .chain(
                headers
                    .callables
                    .iter()
                    .map(|row| (DeclarationId::Callable(row.ordinal), row.name.span)),
            )
            .chain(
                headers
                    .types
                    .rows
                    .iter()
                    .map(|row| (DeclarationId::Type(row.ordinal), row.name.span)),
            )
            .filter_map(|(id, span)| ids.contains(&id).then_some(span))
            .collect();
        spans.sort_unstable_by_key(|span| (span.start, span.end));
        spans
    }

    /// Finds existing evidence by exact span and role, not by overlap.
    #[cfg(test)]
    pub(super) fn evidence_at(
        &self,
        span: Span,
        role: EvidenceRole,
    ) -> Option<CheckedEvidence<'_>> {
        let headers = &self.lexical.headers;
        let exact = |name: &&CsName| name.span == span;
        let name = match role {
            EvidenceRole::NamespaceSegment => headers
                .types
                .namespaces
                .iter()
                .flat_map(|row| &row.segments)
                .find(exact),
            EvidenceRole::TypeBinder => headers
                .types
                .rows
                .iter()
                .flat_map(|row| &row.type_params)
                .find(exact),
            EvidenceRole::MethodBinder => headers
                .callables
                .iter()
                .flat_map(|row| &row.type_params)
                .find(exact),
            EvidenceRole::ReturnType => headers
                .callables
                .iter()
                .filter_map(|row| row.return_type.as_ref())
                .find(exact),
            EvidenceRole::ParameterType => {
                return headers
                    .parameter_occurrences()
                    .map(|(_, _, parameter)| parameter)
                    .find(|parameter| match &parameter.type_use {
                        CallableParameterTypeUse::Simple(name) => name.span == span,
                        CallableParameterTypeUse::Qualified(path) => path.span == span,
                    })
                    .map(CheckedEvidence::ParameterType);
            }
            EvidenceRole::QualifiedParameterRoot | EvidenceRole::QualifiedParameterSegment => {
                headers
                    .parameter_occurrences()
                    .filter_map(|(_, _, parameter)| match &parameter.type_use {
                        CallableParameterTypeUse::Qualified(path) => Some(path),
                        CallableParameterTypeUse::Simple(_) => None,
                    })
                    .flat_map(|path| match role {
                        EvidenceRole::QualifiedParameterRoot => std::slice::from_ref(&path.root),
                        _ => &path.segments,
                    })
                    .find(exact)
            }
            EvidenceRole::QualifiedReceiverRoot
            | EvidenceRole::QualifiedReceiverPrefix
            | EvidenceRole::QualifiedReceiverTerminal => self
                .references
                .iter()
                .filter_map(|reference| match &reference.site.receiver {
                    Receiver::Rooted(path) => Some(path),
                    _ => None,
                })
                .flat_map(|path| {
                    let (terminal, prefixes) =
                        path.segments.split_last().expect("validated receiver path");
                    match role {
                        EvidenceRole::QualifiedReceiverRoot => std::slice::from_ref(&path.root),
                        EvidenceRole::QualifiedReceiverPrefix => prefixes,
                        _ => std::slice::from_ref(terminal),
                    }
                })
                .find(exact),
            EvidenceRole::MemberModifier => headers
                .callables
                .iter()
                .map(|row| &row.modifiers)
                .chain(
                    self.lexical
                        .declarations
                        .fields
                        .iter()
                        .map(|row| &row.modifiers),
                )
                .flat_map(MemberModifiers::tokens)
                .find(exact),
            EvidenceRole::FieldType => self
                .lexical
                .declarations
                .fields
                .iter()
                .map(|row| &row.type_use)
                .find(exact),
            EvidenceRole::LocalType => self
                .lexical
                .declarations
                .locals
                .iter()
                .map(|row| &row.type_use)
                .find(exact),
            EvidenceRole::Reference(role) => {
                return self
                    .references
                    .iter()
                    .map(|reference| &reference.site)
                    .find(|site| site.role == role && site.name.span == span)
                    .map(CheckedEvidence::Reference);
            }
        };
        name.map(CheckedEvidence::Name)
    }
}

/// Certifies checked subset coverage in exactly `region` before binding any
/// sites. Syntax and qualified parameter failures refuse the whole index.
pub(super) fn index_checked_mode(
    source: &str,
    region: Span,
    mode: CheckedHeaderMode,
) -> Result<CheckedIndex, GenericCallableIndexError> {
    let code = source
        .get(region.start..region.end)
        .ok_or(GenericCallableIndexError::InvalidRegion(region))?;
    let mut declarations = LexicalDeclarations::default();
    let mut sites = Vec::new();
    let mut base_occurrences = Vec::new();
    let PendingQualifiedCallableIndex { types, callables } = parse_checked_headers(
        code,
        region,
        &mut declarations,
        &mut sites,
        &mut base_occurrences,
        mode,
    )?;
    let lookup = QualifiedTypeLookup::new(&types);
    let callables = callables
        .into_iter()
        .map(|row| {
            row.resolve(|path| {
                lookup
                    .resolve_checked(path)
                    .map(|binding| binding.terminal)
                    .map_err(|failure| GenericCallableIndexError::CheckedParameterType {
                        use_span: path.span,
                        failure,
                    })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let base_identifiers = base_occurrences
        .iter()
        .flat_map(|base| match &base.clause.path {
            bases::BasePath::Bare(name) => std::slice::from_ref(name),
            bases::BasePath::Rooted { path, .. } => &path.segments,
        })
        .map(|name| name.span)
        .collect();
    let bases = bases::BaseIndex::new(&types, &lookup, base_occurrences);
    let references = binding::bind_checked(
        &types,
        &callables,
        &declarations,
        &lookup,
        &bases.graph,
        sites,
    );
    Ok(CheckedIndex {
        lexical: LexicalDeclarationIndex {
            headers: CallableHeaderIndex { types, callables },
            declarations,
        },
        references,
        bases,
        base_identifiers,
    })
}

#[cfg(test)]
pub(super) fn index_checked(
    source: &str,
    region: Span,
) -> Result<CheckedIndex, GenericCallableIndexError> {
    index_checked_mode(source, region, CheckedHeaderMode::GenericEvidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::csharp::ast::{CsNode, CsNodeKind, CsUnit};
    use crate::v2::csharp::parser::cs_parse;
    use binding::{
        Ambiguity,
        BindingOutcome::{Ambiguous, Bound, Unresolved as Unbound},
        Unresolved,
    };
    use DeclarationId::{Field, Local, Parameter};
    use QualifiedTypeFailure::{MissingType, UnsupportedArity};

    fn name(text: &str, start: usize, end: usize) -> CsName {
        CsName {
            text: text.to_owned(),
            span: Span::new(start, end),
        }
    }

    const METHOD: DeclarationRole = DeclarationRole::Callable(CallableKind::Method);

    fn checked(source: &str) -> Result<CheckedIndex, GenericCallableIndexError> {
        index_checked(source, Span::new(0, source.len()))
    }

    fn checked_outcomes(index: &CheckedIndex) -> Vec<(Span, &binding::BindingOutcome)> {
        index
            .references
            .iter()
            .map(|reference| (reference.site.name.span, &reference.outcome))
            .collect()
    }

    #[test]
    fn checked_calls_bind_sole_methods_by_count_and_keep_the_argument_read() {
        let source = "class C{void F(){}void F(int a){}void G(int v){F();this.F(v);}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(47, 48), &Bound(DeclarationId::Callable(0))),
                (Span::new(56, 57), &Bound(DeclarationId::Callable(1))),
                (Span::new(58, 59), &Bound(Parameter(2, 0))),
            ]
        );
        assert_eq!(
            (
                index.declaration_at(Span::new(13, 14), METHOD).unwrap().id,
                index.declaration_at(Span::new(23, 24), METHOD).unwrap().id,
                index
                    .declaration_at(Span::new(44, 45), DeclarationRole::Parameter)
                    .unwrap()
                    .id,
            ),
            (
                DeclarationId::Callable(0),
                DeclarationId::Callable(1),
                Parameter(2, 0)
            )
        );
        assert_eq!(
            index
                .references
                .iter()
                .map(|row| (row.site.ordinal, row.site.role, row.site.call_argument))
                .collect::<Vec<_>>(),
            [
                (0, ReferenceRole::Call, None),
                (1, ReferenceRole::Call, Some(Span::new(58, 59))),
                (2, ReferenceRole::Read, None),
            ]
        );
    }

    #[test]
    fn checked_calls_keep_all_same_count_overloads_despite_one_exact_witness() {
        let source = "class C{void F(int a){}void F(bool b){}void G(int v){F(v);}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (
                    Span::new(53, 54),
                    &Ambiguous(
                        Ambiguity::Overload,
                        vec![DeclarationId::Callable(0), DeclarationId::Callable(1)]
                    )
                ),
                (Span::new(55, 56), &Bound(Parameter(2, 0))),
            ]
        );
        assert_eq!(
            (
                index.declaration_at(Span::new(13, 14), METHOD).unwrap().id,
                index.declaration_at(Span::new(28, 29), METHOD).unwrap().id,
                &index.references[0].argument,
            ),
            (
                DeclarationId::Callable(0),
                DeclarationId::Callable(1),
                &Some(binding::CallArgumentEvidence {
                    span: Span::new(55, 56),
                    declaration: Some(Parameter(2, 0)),
                    exact_type: Some(CallableParameterType::Int),
                }),
            )
        );
    }

    #[test]
    fn checked_calls_without_a_proven_exact_match_preserve_the_argument_failure() {
        use Unresolved::{NoExactParameterMatch, UnboundArgument};
        for (source, call, argument, refusal, read, exact_type) in [
            (
                "class C{void F(bool a){}void G(int v){F(v);}}",
                Span::new(38, 39),
                Span::new(40, 41),
                NoExactParameterMatch {
                    span: Span::new(40, 41),
                },
                Bound(Parameter(1, 0)),
                Some(CallableParameterType::Int),
            ),
            (
                "class C{void F(bool a){}void G(int v){F(u);}}",
                Span::new(38, 39),
                Span::new(40, 41),
                UnboundArgument {
                    span: Span::new(40, 41),
                },
                Unbound(Unresolved::NotFound),
                None,
            ),
            (
                "class C{void F(int a){}void F(bool b){}void G(global::C p){F(p);}}",
                Span::new(59, 60),
                Span::new(61, 62),
                NoExactParameterMatch {
                    span: Span::new(61, 62),
                },
                Bound(Parameter(2, 0)),
                Some(CallableParameterType::LocalType(0)),
            ),
            (
                "namespace A{class T{}}namespace B{class T{}}class C{void F(global::A.T a){}void G(global::B.T v){F(v);}}",
                Span::new(97, 98),
                Span::new(99, 100),
                NoExactParameterMatch {
                    span: Span::new(99, 100),
                },
                Bound(Parameter(1, 0)),
                Some(CallableParameterType::LocalType(1)),
            ),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_outcomes(&index),
                [(call, &Unbound(refusal)), (argument, &read)],
                "{source}"
            );
            assert_eq!(
                index.references[0]
                    .argument
                    .as_ref()
                    .map(|evidence| (evidence.span, evidence.exact_type)),
                Some((argument, exact_type)),
                "{source}"
            );
        }
    }

    #[test]
    fn checked_call_preserves_both_equal_signature_occurrences() {
        let source = "class C{void F(int a){}void F(int b){}void G(int v){F(v);}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (
                    Span::new(52, 53),
                    &Ambiguous(
                        Ambiguity::Overload,
                        vec![DeclarationId::Callable(0), DeclarationId::Callable(1)]
                    )
                ),
                (Span::new(54, 55), &Bound(Parameter(2, 0))),
            ]
        );
        assert_eq!(
            index.declaration_at(Span::new(13, 14), METHOD).unwrap().key,
            index.declaration_at(Span::new(28, 29), METHOD).unwrap().key
        );
    }

    #[test]
    fn checked_call_records_named_receiver_and_argument_as_separate_reads() {
        let source = "class C{void F(global::C a){}void G(global::C p){p.F(p);}}";

        let index = checked(source).unwrap();

        assert_eq!(
            index
                .references
                .iter()
                .map(|row| (row.site.name.span, row.site.role, &row.outcome))
                .collect::<Vec<_>>(),
            [
                (
                    Span::new(49, 50),
                    ReferenceRole::Read,
                    &Bound(Parameter(1, 0))
                ),
                (
                    Span::new(51, 52),
                    ReferenceRole::Call,
                    &Bound(DeclarationId::Callable(0))
                ),
                (
                    Span::new(53, 54),
                    ReferenceRole::Read,
                    &Bound(Parameter(1, 0))
                ),
            ]
        );
    }

    #[test]
    fn checked_call_uses_the_recorded_type_of_a_read_field_or_local() {
        for (source, site, argument, declaration, exact_type) in [
            (
                "class C{int n;void F(int a){}void G(){F(n);}}",
                38,
                40,
                Field(0, 0),
                CallableParameterType::Int,
            ),
            (
                "class C{void F(bool a){}void G(){bool n;F(n);}}",
                40,
                42,
                DeclarationId::Local(1, 0, 0),
                CallableParameterType::Bool,
            ),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_outcomes(&index),
                [
                    (
                        Span::new(site, site + 1),
                        &Bound(DeclarationId::Callable(0))
                    ),
                    (Span::new(argument, argument + 1), &Bound(declaration)),
                ],
                "{source}"
            );
            assert_eq!(
                index.references[0].argument,
                Some(binding::CallArgumentEvidence {
                    span: Span::new(argument, argument + 1),
                    declaration: Some(declaration),
                    exact_type: Some(exact_type),
                }),
                "{source}"
            );
        }
    }

    #[test]
    fn checked_call_keeps_raw_spelling_and_exact_roles_in_a_utf8_region() {
        let source = "π|class C{void @F(int a){}void G(int @v){this.@F(/*é*/@v);}}|";

        let index = index_checked(source, Span::new(3, 62)).unwrap();

        assert_eq!(
            index.evidence_at(
                Span::new(47, 49),
                EvidenceRole::Reference(ReferenceRole::Call)
            ),
            Some(CheckedEvidence::Reference(&ReferenceSite {
                ordinal: 0,
                owner: SiteOwner {
                    callable: 1,
                    block: 0
                },
                role: ReferenceRole::Call,
                name: name("@F", 47, 49),
                receiver: Receiver::This(name("this", 42, 46)),
                call_argument: Some(Span::new(56, 58)),
            }))
        );
        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(47, 49), &Bound(DeclarationId::Callable(0))),
                (Span::new(56, 58), &Bound(Parameter(1, 0))),
            ]
        );
    }

    #[test]
    fn checked_calls_do_not_bind_past_a_lexical_or_member_blocker() {
        use Unresolved::{BeforeLocalDeclaration, GenericMethod, WrongArgumentCount, WrongKind};
        for (source, site, reason) in [
            ("class C{void F(int a){}void G(){F();}}", 32, WrongArgumentCount),
            ("class C{void F(){}void G(int F){F();}}", 32, WrongKind),
            ("class C{void F(){}void G(){F();int F;}}", 27, BeforeLocalDeclaration),
            ("class C{C(){}void G(){C();}}", 22, WrongKind),
            ("class C{void F<T>(T a){}void F(int a){}void G(int v){F(v);}}", 53, GenericMethod),
            ("class C{int F;void F(int a){}void G(int v){F(v);}}", 43, WrongKind),
            ("class A{public void F(int a){}void F(bool b){} }class B{void G(global::A p,int v){p.F(v);}}", 84, Unresolved::Inaccessible),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_call_outcomes(&index),
                [(Span::new(site, site + 1), &Unbound(reason))],
                "{source}"
            );
        }
    }

    #[test]
    fn checked_type_parameter_blocks_bare_call_but_not_explicit_base_call() {
        let source = "class B{public void F(){}}class D<F>:global::B{void G(){F();base.F();}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_call_outcomes(&index),
            [
                (
                    Span::new(56, 57),
                    &Unbound(Unresolved::TypeParameterInValuePosition {
                        declaration: Span::new(34, 35),
                    }),
                ),
                (Span::new(65, 66), &Bound(DeclarationId::Callable(0))),
            ]
        );
        assert_eq!(
            index.declaration_at(Span::new(20, 21), METHOD).unwrap().id,
            DeclarationId::Callable(0)
        );
    }

    #[test]
    fn checked_type_parameter_in_owner_lineage_prevents_inherited_field_argument_binding() {
        for (source, call, argument, declaration) in [
            (
                "class B{public int T;}class D<T>:global::B{void H(int x){}void G(){H(T);}}",
                Span::new(67, 68),
                Span::new(69, 70),
                Span::new(30, 31),
            ),
            (
                "class B{public int T;}class O<@T>{class D:global::B{void H(int x){}void G(){H(T);}}}",
                Span::new(76, 77),
                Span::new(78, 79),
                Span::new(30, 32),
            ),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_outcomes(&index),
                [
                    (call, &Unbound(Unresolved::UnboundArgument { span: argument })),
                    (
                        argument,
                        &Unbound(Unresolved::TypeParameterInValuePosition { declaration }),
                    ),
                ],
                "{source}"
            );
        }
    }

    #[test]
    fn checked_type_parameter_does_not_displace_a_bound_value_argument() {
        let source =
            "class B{public int T;}class D<T>:global::B{void H(int x){}void G(int T){H(T);}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(72, 73), &Bound(DeclarationId::Callable(0))),
                (Span::new(74, 75), &Bound(Parameter(1, 0))),
            ]
        );
    }

    #[test]
    fn checked_type_parameter_does_not_replace_a_lexical_order_or_conflict_refusal() {
        for (source, call, argument, lexical_outcome) in [
            (
                "class B{public int T;}class D<T>:global::B{void H(int x){}void G(){H(T);int T;}}",
                Span::new(67, 68),
                Span::new(69, 70),
                Unbound(Unresolved::BeforeLocalDeclaration),
            ),
            (
                "class B{public int T;}class D<T>:global::B{void H(int x){}void G(int T){int T;H(T);}}",
                Span::new(78, 79),
                Span::new(80, 81),
                Ambiguous(Ambiguity::DeclarationSpaceConflict, vec![Parameter(1, 0), Local(1, 0, 0)]),
            ),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_outcomes(&index),
                [
                    (call, &Unbound(Unresolved::UnboundArgument { span: argument })),
                    (argument, &lexical_outcome),
                ],
                "{source}"
            );
        }
    }

    fn checked_call_outcomes(index: &CheckedIndex) -> Vec<(Span, &binding::BindingOutcome)> {
        index
            .references
            .iter()
            .filter(|row| row.site.role == ReferenceRole::Call)
            .map(|row| (row.site.name.span, &row.outcome))
            .collect()
    }

    #[test]
    fn checked_calls_require_proven_receivers_and_matching_member_access() {
        use Unresolved::{
            InstanceMemberRequired, StaticMemberRequired, ThisInStaticContext, UnknownReceiver,
            UnprovenType,
        };
        for (source, site, reason) in [
            (
                "class C{void F(){}void G(int p){p.F();}}",
                34,
                UnknownReceiver,
            ),
            (
                "class C{static void F(){}void G(){this.F();}}",
                39,
                InstanceMemberRequired,
            ),
            (
                "class C{void F(){}static void G(){F();}}",
                34,
                StaticMemberRequired,
            ),
            (
                "class C{void F(){}static void G(){this.F();}}",
                39,
                ThisInStaticContext,
            ),
            (
                "class C{void G(){global::T.x();}}",
                27,
                UnprovenType(MissingType {
                    segment: Span::new(25, 26),
                }),
            ),
            (
                "class A{class N{public static void F(){}}}class B{void G(){global::A.N.F();}}",
                71,
                Unresolved::Inaccessible,
            ),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_call_outcomes(&index),
                [(Span::new(site, site + 1), &Unbound(reason))],
                "{source}"
            );
        }
    }

    #[test]
    fn checked_rooted_call_keeps_the_exact_type_proof() {
        let source = "class C{static void F(){}void G(){global::C.F();}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_call_outcomes(&index),
            [(Span::new(44, 45), &Bound(DeclarationId::Callable(0)))]
        );
        assert_eq!(
            (
                index.references[0]
                    .qualified_type
                    .as_ref()
                    .map(|ty| ty.terminal),
                index.evidence_at(Span::new(42, 43), EvidenceRole::QualifiedReceiverTerminal),
            ),
            (Some(0), Some(CheckedEvidence::Name(&name("C", 42, 43))))
        );
    }

    #[test]
    fn checked_inherited_calls_stop_at_the_nearest_declaring_scope() {
        use Unresolved::{Inaccessible, WrongArgumentCount};
        for (source, site, outcome) in [
            ("class A{public void F(){} }class B:A{void G(){base.F();}}", 51, Bound(DeclarationId::Callable(0))),
            ("class A{public void F(){} }class B:A{void F(int a){} }class C:B{void G(){base.F();}}", 78, Unbound(WrongArgumentCount)),
            ("class A{public void F(){} }class B:A{void F(){} }class C:B{void G(){base.F();}}", 73, Unbound(Inaccessible)),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_call_outcomes(&index),
                [(Span::new(site, site + 1), &outcome)],
                "{source}"
            );
        }
    }

    #[test]
    fn checked_inherited_call_refuses_a_match_in_incomplete_ancestry() {
        use bases::{BaseLookupError::IncompleteBase, IncompleteReason::Target};
        let source = "class A:Missing{public void F(){} }class B:A{void G(){base.F();}}";

        let index = checked(source).unwrap();

        let span = Span::new(8, 15);
        assert_eq!(
            checked_call_outcomes(&index),
            [(
                Span::new(59, 60),
                &Unbound(Unresolved::Base(IncompleteBase {
                    span,
                    from: 0,
                    reason: Target(MissingType { segment: span }),
                }))
            )]
        );
    }

    #[test]
    fn checked_this_and_base_select_different_exact_field_occurrences() {
        let source = "class B{public int x;}class D:global::B{int x;void F(){this.x=base.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            (
                checked_outcomes(&index),
                index
                    .lexical
                    .declarations
                    .field_occurrences()
                    .map(|(id, row)| (id, row.name.span))
                    .collect::<Vec<_>>()
            ),
            (
                vec![
                    (Span::new(60, 61), &Bound(Field(1, 0))),
                    (Span::new(67, 68), &Bound(Field(0, 0)))
                ],
                vec![
                    (Field(0, 0), Span::new(19, 20)),
                    (Field(1, 0), Span::new(44, 45))
                ],
            )
        );
    }

    #[test]
    fn checked_inheritance_uses_a_bare_global_sibling_chain_for_proven_instance_receivers() {
        let source = "class A{public int x;}class B:A{}class C:B{void F(global::C p){p.x=this.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(63, 64), &Bound(Parameter(0, 0))),
                (Span::new(65, 66), &Bound(Field(0, 0))),
                (Span::new(72, 73), &Bound(Field(0, 0))),
            ]
        );
    }

    #[test]
    fn checked_inheritance_validates_past_a_matching_field_before_selecting_it() {
        use bases::{BaseLookupError::IncompleteBase, IncompleteReason::Target};
        let source = "class B:global::N.Missing{public int x;}class D:B{void F(){this.x=base.x;}}";
        let incomplete = Unbound(Unresolved::Base(IncompleteBase {
            span: Span::new(8, 25),
            from: 0,
            reason: Target(MissingType {
                segment: Span::new(16, 17),
            }),
        }));

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(64, 65), &incomplete),
                (Span::new(71, 72), &incomplete)
            ]
        );
    }

    #[test]
    fn checked_inheritance_reports_the_cycle_edge_rather_than_not_found() {
        let source = "class A:B{}class B:A{}class D:A{void F(){this.x=base.x;}}";
        let cycle = Unbound(Unresolved::Base(bases::BaseLookupError::BaseCycle {
            span: Span::new(19, 20),
            from: 1,
            to: 0,
        }));

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [(Span::new(46, 47), &cycle), (Span::new(53, 54), &cycle)]
        );
    }

    #[test]
    fn checked_direct_member_does_not_depend_on_an_unproven_base() {
        let source = "class D:B{int x;void F(){this.x=x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(30, 31), &Bound(Field(0, 0))),
                (Span::new(32, 33), &Bound(Field(0, 0)))
            ]
        );
    }

    #[test]
    fn checked_base_lookup_stops_at_the_nearest_declaring_level_even_on_conflict() {
        use Unresolved::{Inaccessible, InstanceMemberRequired, WrongKind};
        for (source, first, second, reason) in [
            (
                "class A{public int x;}class B:A{void x(){}}class D:B{void F(){x=base.x;}}",
                62,
                69,
                WrongKind,
            ),
            (
                "class A{public int x;}class B:A{int x;}class D:B{void F(){x=base.x;}}",
                58,
                65,
                Inaccessible,
            ),
            (
                "class A{public int x;}class B:A{public static int x;}class D:B{void F(){this.x=base.x;}}",
                77,
                84,
                InstanceMemberRequired,
            ),
        ] {
            let index = checked(source).unwrap();

            let conflict = Unbound(reason);
            assert_eq!(
                checked_outcomes(&index),
                [
                    (Span::new(first, first + 1), &conflict),
                    (Span::new(second, second + 1), &conflict)
                ],
                "{source}"
            );
        }
    }

    #[test]
    fn checked_base_receiver_does_not_bind_from_a_static_context() {
        let source = "class A{public int x;}class B:A{}class C:B{static void F(){base.x=base.x;}}";
        let not_instance = Unbound(Unresolved::BaseInStaticContext);

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(64, 65), &not_instance),
                (Span::new(71, 72), &not_instance)
            ]
        );
    }

    #[test]
    fn checked_base_receiver_without_an_explicit_parent_never_selects_its_own_field() {
        let source = "class D{int x;void F(){this.x=base.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(28, 29), &Bound(Field(0, 0))),
                (Span::new(35, 36), &Unbound(Unresolved::NotFound)),
            ]
        );
    }

    #[test]
    fn checked_rooted_static_receiver_can_bind_an_inherited_field() {
        let source = "class B{public static int x;}class A:B{public class N{}}class C{void F(){global::A.x=global::A.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(83, 84), &Bound(Field(0, 0))),
                (Span::new(95, 96), &Bound(Field(0, 0)))
            ]
        );
    }

    #[test]
    fn checked_inheritance_refuses_a_rooted_receiver_with_a_hidden_type() {
        let source = "class B{public static int x;}class O{class H:global::B{}}class C{void F(){global::O.H.x=global::O.H.x;}}";
        let inaccessible = Unbound(Unresolved::Inaccessible);

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(86, 87), &inaccessible),
                (Span::new(100, 101), &inaccessible),
            ]
        );
        assert_eq!(
            index.references[0].qualified_type,
            Some(QualifiedTypeBinding {
                prefixes: vec![QualifiedScope::Type(1)],
                terminal: 2,
            })
        );
    }

    #[test]
    fn checked_inheritance_refuses_a_parameter_receiver_with_a_hidden_enclosing_type() {
        let source = "class B{public int x;}class O{class H{public class T:global::B{}}}class C{void F(global::O.H.T p){p.x=p.x;}}";
        let inaccessible = Unbound(Unresolved::Inaccessible);

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(98, 99), &Bound(Parameter(0, 0))),
                (Span::new(100, 101), &inaccessible),
                (Span::new(102, 103), &Bound(Parameter(0, 0))),
                (Span::new(104, 105), &inaccessible),
            ]
        );
    }

    #[test]
    fn checked_inheritance_retains_an_inaccessible_base_target_as_an_incomplete_edge() {
        use bases::{BaseLookupError::IncompleteBase, IncompleteReason::InaccessibleTarget};
        for (source, from, target, start, end, write, read) in [
            (
                "class B{public int x;}class O{class H:global::B{}}class D:global::O.H{void F(){base.x=base.x;}}",
                3, 2, 58, 69, 84, 91,
            ),
            (
                "class B{public int x;}class O{class H{public class T:global::B{}}}class D:global::O.H.T{void F(){base.x=base.x;}}",
                4, 3, 74, 87, 102, 109,
            ),
        ] {
            let incomplete = Unbound(Unresolved::Base(IncompleteBase {
                span: Span::new(start, end),
                from,
                reason: InaccessibleTarget {
                    target,
                    blocked_by: 2,
                },
            }));

            let index = checked(source).unwrap();

            assert_eq!(
                checked_outcomes(&index),
                [
                    (Span::new(write, write + 1), &incomplete),
                    (Span::new(read, read + 1), &incomplete),
                ],
                "{source}",
            );
        }
    }

    #[test]
    fn checked_inheritance_allows_a_nested_receiver_with_public_lineage_or_exact_caller() {
        for (source, write, read) in [
            (
                "class B{public static int x;}class O{public class H{public class T:global::B{}}}class C{void F(){global::O.H.T.x=global::O.H.T.x;}}",
                111, 127,
            ),
            (
                "class B{public static int x;}class O{class H:global::B{void F(){global::O.H.x=global::O.H.x;}}}",
                76, 90,
            ),
        ] {
            let index = checked(source).unwrap();

            assert_eq!(
                checked_outcomes(&index),
                [
                    (Span::new(write, write + 1), &Bound(Field(0, 0))),
                    (Span::new(read, read + 1), &Bound(Field(0, 0))),
                ],
                "{source}",
            );
        }
    }

    #[test]
    fn checked_inheritance_accepts_a_nested_base_target_with_public_lineage() {
        let source = "class B{public int x;}class O{public class H:global::B{}}class D:global::O.H{void F(){base.x=base.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(91, 92), &Bound(Field(0, 0))),
                (Span::new(98, 99), &Bound(Field(0, 0))),
            ]
        );
    }

    #[test]
    fn checked_rooted_receiver_does_not_inherit_through_a_competing_type_qualifier() {
        let source = "class B{public static int x;}class A{int N;public class N:global::B{}}class C{void F(){global::A.N.x=global::A.N.x;}}";
        let qualifier = |segment| {
            Unbound(Unresolved::UnprovenType(
                QualifiedTypeFailure::AmbiguousTypeMember {
                    segment,
                    prefixes: vec![QualifiedScope::Type(1)],
                    candidates: vec![
                        QualifiedTypeCandidate::Member {
                            id: Field(1, 0),
                            span: Span::new(41, 42),
                        },
                        QualifiedTypeCandidate::Type {
                            ordinal: 2,
                            span: Span::new(56, 57),
                        },
                    ],
                },
            ))
        };

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(99, 100), &qualifier(Span::new(97, 98))),
                (Span::new(113, 114), &qualifier(Span::new(111, 112)))
            ]
        );
    }

    #[test]
    fn checked_inheritance_refuses_malformed_or_out_of_subset_base_clauses() {
        for (source, start, end) in [
            ("class C: {}", 9, 10),
            ("class C:B D{}", 10, 11),
            ("class C:,B{}", 8, 9),
            ("class C:B,,D{}", 9, 10),
            ("class C:B,{}", 9, 10),
            ("class C:B,D{}", 9, 10),
            ("class C:global::B<A,B>{}", 17, 18),
            ("class C:global::B<A,B{}", 17, 18),
            ("class C:global::B(0){}", 17, 18),
            ("class C:global::B[]{}", 17, 18),
            ("class C:global::N.{}", 17, 18),
            ("class C:global::B where T:B{}", 18, 23),
            ("class C:global::B;", 17, 18),
            ("class C:global::B{", 18, 18),
            ("class C<T>:B{}", 11, 12),
            ("class A{class C:B{}}", 16, 17),
            ("namespace N{class C:B{}}", 20, 21),
            ("class C:@global::B{}", 15, 17),
            ("interface C:global::B{}", 11, 12),
            ("struct C{void F(){this.x=base.x;}}", 25, 29),
        ] {
            assert_checked_refusal(source, Span::new(start, end), vec![]);
        }
    }

    #[test]
    fn checked_bad_base_delimiter_keeps_ordered_legacy_diagnostics() {
        let source = "class C:B D{} } ]";
        let recovery = vec![Span::new(14, 14), Span::new(14, 15), Span::new(16, 17)];

        assert_checked_refusal(source, Span::new(10, 11), recovery.clone());

        assert_eq!(cs_parse(source, 0).errors, recovery);
    }

    #[test]
    fn checked_binds_parameter_and_this_to_the_same_instance_field() {
        let source = "class C{int x;void G(global::C p){p.x=this.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            [
                checked_declaration(&index, DeclarationRole::Parameter, 31, 32).id,
                checked_declaration(&index, DeclarationRole::Field, 12, 13).id,
            ],
            [Parameter(0, 0), Field(0, 0)]
        );
        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(34, 35), &Bound(Parameter(0, 0))),
                (Span::new(36, 37), &Bound(Field(0, 0))),
                (Span::new(43, 44), &Bound(Field(0, 0))),
            ]
        );
    }

    const EXACT_STATIC: &str = "namespace A{} namespace @A{class @T{public static int @x;}} namespace B{class T{public static int x;}} class C{void G(){global::@A.T.x=global::B.@T.@x;}}";

    #[test]
    fn checked_static_access_uses_the_exact_namespace_scope_and_raw_spelling() {
        let index = checked(EXACT_STATIC).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(133, 134), &Bound(Field(0, 0))),
                (Span::new(148, 150), &Bound(Field(1, 0))),
            ]
        );
        assert_eq!(
            [
                checked_declaration(&index, DeclarationRole::Field, 54, 56).id,
                checked_declaration(&index, DeclarationRole::Field, 98, 99).id,
            ],
            [Field(0, 0), Field(1, 0)]
        );
        assert_eq!(
            index
                .references
                .iter()
                .map(|reference| reference.qualified_type.as_ref())
                .collect::<Vec<_>>(),
            [
                Some(&QualifiedTypeBinding {
                    prefixes: vec![QualifiedScope::Namespace(0)],
                    terminal: 0
                }),
                Some(&QualifiedTypeBinding {
                    prefixes: vec![QualifiedScope::Namespace(1)],
                    terminal: 1
                }),
            ]
        );
    }

    #[test]
    fn checked_rooted_access_covers_exact_receiver_roles_and_modifier_tokens() {
        use EvidenceRole::{
            MemberModifier, QualifiedReceiverPrefix, QualifiedReceiverRoot,
            QualifiedReceiverTerminal,
        };
        let index = checked(EXACT_STATIC).unwrap();

        for (role, span, expected) in [
            (
                QualifiedReceiverRoot,
                Span::new(120, 126),
                Some(name("global", 120, 126)),
            ),
            (
                QualifiedReceiverPrefix,
                Span::new(128, 130),
                Some(name("@A", 128, 130)),
            ),
            (
                QualifiedReceiverTerminal,
                Span::new(131, 132),
                Some(name("T", 131, 132)),
            ),
            (QualifiedReceiverTerminal, Span::new(128, 130), None),
            (QualifiedReceiverPrefix, Span::new(131, 132), None),
            (QualifiedReceiverRoot, Span::new(120, 125), None),
            (
                MemberModifier,
                Span::new(36, 42),
                Some(name("public", 36, 42)),
            ),
            (
                MemberModifier,
                Span::new(43, 49),
                Some(name("static", 43, 49)),
            ),
        ] {
            assert_eq!(
                index.evidence_at(span, role),
                expected.as_ref().map(CheckedEvidence::Name),
                "{role:?}: {span:?}"
            );
        }
    }

    #[test]
    fn checked_namespace_type_collisions_keep_all_occurrence_evidence_even_with_one_leaf() {
        use QualifiedTypeCandidate::{Namespace, Type};
        let source = "namespace N{} class N{} namespace @N{class T{}} class C{void G(){global::N.T.x=global::N.x;}}";

        let index = checked(source).unwrap();

        let collision = |start, end| {
            Unbound(Unresolved::UnprovenType(QualifiedTypeFailure::Ambiguous {
                segment: Span::new(start, end),
                candidates: vec![
                    Namespace {
                        occurrence: 0,
                        span: Span::new(10, 11),
                    },
                    Type {
                        ordinal: 0,
                        span: Span::new(20, 21),
                    },
                    Namespace {
                        occurrence: 1,
                        span: Span::new(34, 36),
                    },
                ],
            }))
        };
        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(77, 78), &collision(73, 74)),
                (Span::new(89, 90), &collision(87, 88)),
            ]
        );
    }

    #[test]
    fn checked_duplicate_type_owners_refuse_to_select_the_only_leaf() {
        let source = "class A{public class T{public static int x;}} class @A{} class C{void G(){global::A.T.x=global::A.x;}}";

        let index = checked(source).unwrap();

        let duplicate = |start, end| {
            Unbound(Unresolved::UnprovenType(QualifiedTypeFailure::Ambiguous {
                segment: Span::new(start, end),
                candidates: vec![
                    QualifiedTypeCandidate::Type {
                        ordinal: 0,
                        span: Span::new(6, 7),
                    },
                    QualifiedTypeCandidate::Type {
                        ordinal: 2,
                        span: Span::new(52, 54),
                    },
                ],
            }))
        };
        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(86, 87), &duplicate(82, 83)),
                (Span::new(98, 99), &duplicate(96, 97)),
            ]
        );
    }

    #[test]
    fn checked_rooted_receiver_refuses_a_field_competing_with_its_terminal_type() {
        use EvidenceRole::{
            QualifiedReceiverPrefix, QualifiedReceiverRoot, QualifiedReceiverTerminal,
        };
        let source = "class A{public static int T;public class T{public static int x;}} class C{void G(){global::A.T.x=global::A.T.x;}}";

        let index = checked(source).unwrap();

        let conflict = |segment| {
            Unbound(Unresolved::UnprovenType(
                QualifiedTypeFailure::AmbiguousTypeMember {
                    segment,
                    prefixes: vec![QualifiedScope::Type(0)],
                    candidates: vec![
                        QualifiedTypeCandidate::Member {
                            id: Field(0, 0),
                            span: Span::new(26, 27),
                        },
                        QualifiedTypeCandidate::Type {
                            ordinal: 1,
                            span: Span::new(41, 42),
                        },
                    ],
                },
            ))
        };
        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(95, 96), &conflict(Span::new(93, 94))),
                (Span::new(109, 110), &conflict(Span::new(107, 108))),
            ]
        );
        assert_eq!(
            index
                .references
                .iter()
                .map(|reference| (reference.site.role, reference.qualified_type.as_ref()))
                .collect::<Vec<_>>(),
            [(ReferenceRole::Write, None), (ReferenceRole::Read, None)]
        );
        assert_eq!(
            [
                index.evidence_at(Span::new(83, 89), QualifiedReceiverRoot),
                index.evidence_at(Span::new(91, 92), QualifiedReceiverPrefix),
                index.evidence_at(Span::new(93, 94), QualifiedReceiverTerminal),
            ],
            [
                Some(CheckedEvidence::Name(&name("global", 83, 89))),
                Some(CheckedEvidence::Name(&name("A", 91, 92))),
                Some(CheckedEvidence::Name(&name("T", 93, 94))),
            ]
        );
    }

    #[test]
    fn checked_rooted_receiver_refuses_a_hidden_callable_competing_at_an_intermediate_type() {
        let source = "namespace N{class A{public class @T{public class U{public static int x;}} void T(){}}} class C{void G(){global::N.A.T.U.x=global::N.A.T.U.x;}}";

        let index = checked(source).unwrap();

        let conflict = |segment| {
            Unbound(Unresolved::UnprovenType(
                QualifiedTypeFailure::AmbiguousTypeMember {
                    segment,
                    prefixes: vec![QualifiedScope::Namespace(0), QualifiedScope::Type(0)],
                    candidates: vec![
                        QualifiedTypeCandidate::Type {
                            ordinal: 1,
                            span: Span::new(33, 35),
                        },
                        QualifiedTypeCandidate::Member {
                            id: DeclarationId::Callable(0),
                            span: Span::new(79, 80),
                        },
                    ],
                },
            ))
        };
        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(120, 121), &conflict(Span::new(116, 117))),
                (Span::new(138, 139), &conflict(Span::new(134, 135))),
            ]
        );
        assert_eq!(
            index.evidence_at(Span::new(118, 119), EvidenceRole::QualifiedReceiverTerminal),
            Some(CheckedEvidence::Name(&name("U", 118, 119)))
        );
    }

    #[test]
    fn checked_parameter_lookup_does_not_treat_a_field_as_a_type_competitor() {
        let source = "class A{int T;public class T{}} class C{void G(global::A.T p){p=p;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            index.evidence_at(Span::new(47, 58), EvidenceRole::ParameterType),
            Some(CheckedEvidence::ParameterType(&qualified_parameter(
                name("global", 47, 53),
                vec![name("A", 55, 56), name("T", 57, 58)],
                Span::new(47, 58),
                1,
                name("p", 59, 60),
            )))
        );
    }

    #[test]
    fn checked_rooted_lookup_is_case_sensitive_without_short_name_fallback() {
        let source = "namespace A{class T{public static int x;}} class C{void G(){global::a.T.x=global::T.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (
                    Span::new(72, 73),
                    &Unbound(Unresolved::UnprovenType(MissingType {
                        segment: Span::new(68, 69)
                    }))
                ),
                (
                    Span::new(84, 85),
                    &Unbound(Unresolved::UnprovenType(MissingType {
                        segment: Span::new(82, 83)
                    }))
                ),
            ]
        );
    }

    #[test]
    fn checked_rooted_lookup_refuses_generic_only_and_mixed_arity() {
        let source = "namespace N{class T<P>{} class U{} class U<P>{}} class C{void G(){global::N.T.x=global::N.U.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (
                    Span::new(78, 79),
                    &Unbound(Unresolved::UnprovenType(UnsupportedArity {
                        segment: Span::new(76, 77)
                    }))
                ),
                (
                    Span::new(92, 93),
                    &Unbound(Unresolved::UnprovenType(UnsupportedArity {
                        segment: Span::new(90, 91)
                    }))
                ),
            ]
        );
    }

    #[test]
    fn checked_rooted_lookup_does_not_descend_through_a_generic_owner() {
        let source =
            "class A<T>{public class N{public static int x;}} class C{void G(){global::A.N.x=x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (
                    Span::new(78, 79),
                    &Unbound(Unresolved::UnprovenType(UnsupportedArity {
                        segment: Span::new(74, 75),
                    }))
                ),
                (Span::new(80, 81), &Unbound(Unresolved::NotFound)),
            ]
        );
    }

    #[test]
    fn checked_parameter_lookup_preserves_the_full_use_and_the_first_competing_segment() {
        let source = "class C{void G(global::N.T p){}} namespace N{class T{}} class N{}";

        let index = checked(source);

        assert_eq!(
            index,
            Err(GenericCallableIndexError::CheckedParameterType {
                use_span: Span::new(15, 26),
                failure: QualifiedTypeFailure::Ambiguous {
                    segment: Span::new(23, 24),
                    candidates: vec![
                        QualifiedTypeCandidate::Namespace {
                            occurrence: 0,
                            span: Span::new(43, 44)
                        },
                        QualifiedTypeCandidate::Type {
                            ordinal: 2,
                            span: Span::new(62, 63)
                        },
                    ],
                },
            })
        );
    }

    #[test]
    fn checked_static_context_distinguishes_explicit_instance_and_static_places() {
        let source = "class C{int x;public static int s;public static void G(global::C p){this.x=x;p.s=global::C.x;s=p.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(73, 74), &Unbound(Unresolved::ThisInStaticContext)),
                (
                    Span::new(75, 76),
                    &Unbound(Unresolved::StaticMemberRequired)
                ),
                (Span::new(77, 78), &Bound(Parameter(0, 0))),
                (
                    Span::new(79, 80),
                    &Unbound(Unresolved::InstanceMemberRequired)
                ),
                (
                    Span::new(91, 92),
                    &Unbound(Unresolved::StaticMemberRequired)
                ),
                (Span::new(93, 94), &Bound(Field(0, 1))),
                (Span::new(95, 96), &Bound(Parameter(0, 0))),
                (Span::new(97, 98), &Bound(Field(0, 0))),
            ]
        );
        assert_eq!(
            index.lexical.headers.callables[0].modifiers,
            MemberModifiers {
                public: Some(name("public", 34, 40)),
                static_token: Some(name("static", 41, 47)),
            }
        );
    }

    #[test]
    fn checked_static_generic_method_keeps_its_modifiers_and_existing_method_slot_evidence() {
        let source = "class C{static void G<T>(T p){}}";

        let index = checked(source).unwrap();

        assert_eq!(
            (
                index.evidence_at(Span::new(8, 14), EvidenceRole::MemberModifier),
                index.evidence_at(Span::new(25, 26), EvidenceRole::ParameterType),
            ),
            (
                Some(CheckedEvidence::Name(&name("static", 8, 14))),
                Some(CheckedEvidence::ParameterType(&parameter(
                    name("T", 25, 26),
                    CallableParameterType::MethodSlot(0),
                    name("p", 27, 28),
                ))),
            )
        );
    }

    #[test]
    fn checked_member_does_not_fall_back_from_a_before_local_or_conflicting_receiver() {
        let source = "class C{int x;int p;void G(global::C q){p.x=q.x;int p;{int q;q.x=p.x;}}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (
                    Span::new(40, 41),
                    &Unbound(Unresolved::BeforeLocalDeclaration)
                ),
                (Span::new(42, 43), &Unbound(Unresolved::UnknownReceiver)),
                (Span::new(44, 45), &Bound(Parameter(0, 0))),
                (Span::new(46, 47), &Bound(Field(0, 0))),
                (
                    Span::new(61, 62),
                    &Ambiguous(
                        Ambiguity::DeclarationSpaceConflict,
                        vec![Parameter(0, 0), Local(0, 1, 0)]
                    )
                ),
                (Span::new(63, 64), &Unbound(Unresolved::UnknownReceiver)),
                (Span::new(65, 66), &Bound(Local(0, 0, 0))),
                (Span::new(67, 68), &Unbound(Unresolved::UnknownReceiver)),
            ]
        );
    }

    #[test]
    fn checked_fields_primitives_and_bare_type_names_cannot_prove_receivers() {
        let source =
            "class C{int x;int p;void G(int n){p.x=n.x;T.x=x;}} class T{public static int x;}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(34, 35), &Bound(Field(0, 1))),
                (Span::new(36, 37), &Unbound(Unresolved::UnknownReceiver)),
                (Span::new(38, 39), &Bound(Parameter(0, 0))),
                (Span::new(40, 41), &Unbound(Unresolved::UnknownReceiver)),
                (Span::new(42, 43), &Unbound(Unresolved::NotFound)),
                (Span::new(44, 45), &Unbound(Unresolved::UnknownReceiver)),
                (Span::new(46, 47), &Bound(Field(0, 0))),
            ]
        );
    }

    #[test]
    fn checked_escaped_this_is_an_ordinary_parameter_name() {
        let source = "class C{int x;void G(global::C @this){@this.x=this.x;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(38, 43), &Bound(Parameter(0, 0))),
                (Span::new(44, 45), &Bound(Field(0, 0))),
                (Span::new(51, 52), &Bound(Field(0, 0))),
            ]
        );
    }

    #[test]
    fn checked_cross_owner_access_requires_public_members_and_nested_types() {
        let source = "class A{int x;public static void m(){} class N{public static int y;} public class P{public static int y;}} class C{void G(global::A p){p.x=global::A.m;global::A.N.y=global::A.P.y;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(135, 136), &Bound(Parameter(1, 0))),
                (Span::new(137, 138), &Unbound(Unresolved::Inaccessible)),
                (Span::new(149, 150), &Unbound(Unresolved::WrongKind)),
                (Span::new(163, 164), &Unbound(Unresolved::Inaccessible)),
                (Span::new(177, 178), &Bound(Field(2, 0))),
            ]
        );
        assert_eq!(
            index.references[4].qualified_type,
            Some(QualifiedTypeBinding {
                prefixes: vec![QualifiedScope::Type(0)],
                terminal: 2,
            })
        );
    }

    #[test]
    fn checked_cross_owner_instance_access_binds_but_inaccessible_static_access_has_no_fallback() {
        let source = "class A{public int x;static int y;} class C{int y;void G(global::A p){p.x=global::A.y;}}";

        let index = checked(source).unwrap();

        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(70, 71), &Bound(Parameter(0, 0))),
                (Span::new(72, 73), &Bound(Field(0, 0))),
                (Span::new(84, 85), &Unbound(Unresolved::Inaccessible)),
            ]
        );
    }

    #[test]
    fn checked_same_owner_conflicts_do_not_filter_out_wrong_kind_or_wrong_receiver_kind() {
        let source = "class C{static int x;void x(){} void G(){global::C.x=this.x;}}";

        let index = checked(source).unwrap();

        let conflict = Ambiguous(
            Ambiguity::CompetingMember,
            vec![Field(0, 0), DeclarationId::Callable(0)],
        );
        assert_eq!(
            checked_outcomes(&index),
            [
                (Span::new(51, 52), &conflict),
                (Span::new(58, 59), &conflict),
            ]
        );
    }

    #[test]
    fn checked_still_refuses_unmodeled_modifiers_and_assignment_syntax() {
        for (source, start, end) in [
            ("class C{public C(){}}", 8, 14),
            ("class C{static public int x;}", 15, 21),
            ("class C{public static static void G(){}}", 22, 28),
            ("class C{void G(){global::T<int>.x=x;}}", 26, 27),
            ("class C{void G(){@global::T.x=x;}}", 24, 26),
            ("class C{void G(){global::T.x=x+z;}}", 30, 31),
        ] {
            assert_checked_refusal(source, Span::new(start, end), vec![]);
        }
    }

    #[test]
    fn checked_bodies_refuse_call_shapes_outside_standalone_zero_or_one_name() {
        for (source, span) in [
            ("class C{void G(){F(G());}}", Span::new(20, 21)),
            ("class C{void G(){F().G();}}", Span::new(20, 21)),
            ("class C{void G(){global::T.x()();}}", Span::new(30, 31)),
            ("class C{void G(){p.q.F();}}", Span::new(20, 21)),
            ("class C{void G(){F(1);}}", Span::new(19, 20)),
            ("class C{void G(){F(true);}}", Span::new(19, 23)),
            ("class C{void G(){F(x.y);}}", Span::new(20, 21)),
            ("class C{void G(){F(x,y);}}", Span::new(20, 21)),
            ("class C{void G(){F<int>();}}", Span::new(18, 19)),
            ("class C{void G(){x=F();}}", Span::new(20, 21)),
            ("class C{void G(){F(x)}}", Span::new(21, 22)),
        ] {
            assert_checked_refusal(source, span, vec![]);
        }
    }

    fn assert_checked_refusal(source: &str, span: Span, recovery: Vec<Span>) {
        let index = checked(source);

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(span),
                recovery,
            }),
            "{source:?}"
        );
    }

    #[test]
    fn checked_index_refuses_invalid_regions_before_lexing() {
        let source = "π|class C{}";
        for region in [
            Span::new(1, 12),
            Span::new(0, 1),
            Span::new(3, 13),
            Span::new(usize::MAX, usize::MAX),
            Span { start: 3, end: 0 },
        ] {
            let index = index_checked(source, region);

            assert_eq!(index, Err(GenericCallableIndexError::InvalidRegion(region)));
        }
    }

    #[test]
    fn checked_index_refuses_a_bom_at_the_region_start_with_absolute_span() {
        let source = "π|\u{feff}class C{}|";

        let index = index_checked(source, Span::new(3, 15));

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(Span::new(3, 6)),
                recovery: vec![],
            })
        );
    }

    #[test]
    fn checked_index_accepts_closed_ordinary_comments_and_whitespace() {
        let source = "π| \t// π\r\n class C{/**/ int x; /* é */}\n|";

        let index = index_checked(source, Span::new(3, 43)).unwrap();

        assert_eq!(
            index.lexical.declarations.fields,
            [FieldOccurrence {
                modifiers: MemberModifiers::default(),
                containing_type: 0,
                type_use: name("int", 26, 29),
                name: name("x", 30, 31),
            }]
        );
    }

    #[test]
    fn checked_index_refuses_discarded_trivia_that_is_not_proven_inert() {
        for (source, end) in [
            ("class C{int x;}/*", 17),
            ("class C{int x;}/// docs\n", 23),
            ("class C{int x;}/** docs */", 26),
            ("class C{int x;}// tail", 22),
            ("class C{int x;}// tail\rclass D{}\n", 32),
            ("class C{int x;}// tail\u{0085}class D{}\n", 33),
            ("class C{int x;}// tail\u{2028}class D{}\n", 34),
            ("class C{int x;}// tail\u{2029}class D{}\n", 34),
        ] {
            assert_checked_refusal(source, Span::new(15, end), vec![]);
        }
    }

    #[test]
    fn checked_index_refuses_a_legacy_identifier_token_containing_a_line_separator() {
        assert_checked_refusal("class C{int \u{2028}x;}", Span::new(12, 16), vec![]);
    }

    #[test]
    fn checked_index_preserves_an_earlier_unsupported_token_and_later_recovery() {
        assert_checked_refusal(
            "class C{int \u{2028}x;} }",
            Span::new(12, 16),
            vec![Span::new(19, 19), Span::new(19, 20)],
        );
    }

    #[test]
    fn checked_index_preserves_an_earlier_parser_refusal_before_an_unsupported_token() {
        assert_checked_refusal(
            "class C{int x=1;int \u{2028}y;} }",
            Span::new(13, 14),
            vec![Span::new(27, 27), Span::new(27, 28)],
        );
    }

    #[test]
    fn legacy_parse_keeps_a_name_with_a_line_separator_and_recovers_at_the_stray_closer() {
        let source = "class C{int \u{2028}x;} }";

        let unit = cs_parse(source, 0);

        let mut declarations = Vec::new();
        unit.root.walk(&mut |node| {
            if let CsNodeKind::NameDecl(name) = &node.kind {
                declarations.push(name.clone());
            }
        });
        assert_eq!(
            (declarations, unit.errors),
            (
                vec![name("\u{2028}x", 12, 16)],
                vec![Span::new(19, 19), Span::new(19, 20)],
            )
        );
    }

    #[test]
    fn checked_index_keeps_recovery_order_alongside_a_trivia_refusal() {
        // A trivia refusal must not disable the private generic-header bridge.
        assert_checked_refusal(
            "class C{int x;}/// docs\nclass D{void F<T>(T p){}} }",
            Span::new(15, 23),
            vec![Span::new(50, 50), Span::new(50, 51)],
        );
    }

    #[test]
    fn checked_index_refuses_stray_closers_with_original_ordered_recovery() {
        assert_checked_refusal(
            "class C{void F(){x=y+z;}} }",
            Span::new(20, 21),
            vec![Span::new(26, 26), Span::new(26, 27)],
        );
    }

    #[test]
    fn checked_index_requires_real_delimiters_even_when_the_parser_emits_no_errors() {
        for (source, start, end) in [
            ("namespace N{class C{}", 21, 21),
            ("class C{void F(){x=y;}", 22, 22),
            ("class C{void F(){x=y;", 21, 21),
            ("class C{void F(){x=y}}", 20, 21),
        ] {
            assert_checked_refusal(source, Span::new(start, end), vec![]);
        }
    }

    #[test]
    fn checked_index_refuses_opaque_syntax_after_an_otherwise_covered_declaration() {
        for (source, start, end) in [
            ("class C{int x;} [A] class D{}", 16, 17),
            ("class C{int x;} partial class D{}", 16, 23),
            ("class C{int x;} class D{extern void G();}", 24, 30),
            ("class C{int x;} using N;", 16, 21),
            ("class C{int x;} class D{int P{get;}}", 29, 30),
            ("class C{int x;} class D{int a=1;}", 29, 30),
            ("class C{int x;} class D{void G(){x=\"y\";}}", 35, 38),
            ("class C{int x;} class D{void G(){x=$\"{y}\";}}", 35, 41),
            ("class C{int x;} class D{int G()=>x;}", 31, 33),
            ("class C{int x;} class D{void G(){if(x){}}}", 33, 35),
            ("class C{int x;} int y;", 16, 19),
        ] {
            assert_checked_refusal(source, Span::new(start, end), vec![]);
        }
    }

    #[test]
    fn checked_index_refuses_an_unmodeled_directive_and_keeps_recovery() {
        assert_checked_refusal(
            "class C{int x;}\n#if X\n#endif\n",
            Span::new(16, 17),
            vec![Span::new(16, 28)],
        );
    }

    #[test]
    fn checked_index_covers_qualified_parameters_and_both_existing_dotted_places() {
        let source = "π|namespace N.S; public class C<T>{/**/ int x; void F(global::N.S.D p, bool b){int y; {this.x = p.x;}}} class D{}|";
        let index = index_checked(source, Span::new(3, 114)).unwrap();

        let root = index.evidence_at(Span::new(55, 61), EvidenceRole::QualifiedParameterRoot);
        let segment = index.evidence_at(Span::new(67, 68), EvidenceRole::QualifiedParameterSegment);

        assert_eq!(
            (root, segment),
            (
                Some(CheckedEvidence::Name(&name("global", 55, 61))),
                Some(CheckedEvidence::Name(&name("D", 67, 68))),
            )
        );
        assert_eq!(
            index
                .references
                .iter()
                .map(|reference| &reference.site)
                .map(|site| (site.role, &site.name, &site.receiver))
                .collect::<Vec<_>>(),
            [
                (
                    ReferenceRole::Write,
                    &name("x", 93, 94),
                    &binding::Receiver::This(name("this", 88, 92)),
                ),
                (
                    ReferenceRole::ReceiverName,
                    &name("p", 97, 98),
                    &binding::Receiver::Bare,
                ),
                (
                    ReferenceRole::Read,
                    &name("x", 99, 100),
                    &binding::Receiver::Name(name("p", 97, 98)),
                ),
            ]
        );
    }

    fn checked_declaration(
        index: &CheckedIndex,
        role: DeclarationRole,
        start: usize,
        end: usize,
    ) -> CheckedDeclaration<'_> {
        index.declaration_at(Span::new(start, end), role).unwrap()
    }

    #[test]
    fn checked_type_keys_compare_normalized_namespaces_without_merging_occurrences() {
        let source = "π|namespace @N.Tools{class @C<T>{}} namespace N.@Tools{class C<U>{}}|";
        let region = Span::new(3, 69);

        let index = index_checked(source, region).unwrap();

        let first = checked_declaration(&index, DeclarationRole::Type, 28, 30);
        let second = checked_declaration(&index, DeclarationRole::Type, 62, 63);
        assert_eq!(
            [
                (first.id, first.namespace_occurrence, first.name),
                (second.id, second.namespace_occurrence, second.name)
            ],
            [
                (DeclarationId::Type(0), Some(0), &name("@C", 28, 30)),
                (DeclarationId::Type(1), Some(1), &name("C", 62, 63)),
            ]
        );
        assert_eq!(first.key, second.key);
        assert_eq!(
            first.key,
            CheckedDeclarationKey {
                namespace_path: vec!["N", "Tools"],
                containing_type: None,
                type_lineage: vec![],
                name: "C",
                shape: DeclarationShape::Type(CsTypeKind::Class, 1),
            }
        );
    }

    #[test]
    fn checked_type_keys_distinguish_exact_nested_owners_with_equal_lineages() {
        let source = "namespace N{class C{class D{struct X{}}} class @C{class @D{struct @X{}}}}";

        let index = index_checked(source, Span::new(0, source.len())).unwrap();

        let first = checked_declaration(&index, DeclarationRole::Type, 35, 36);
        let second = checked_declaration(&index, DeclarationRole::Type, 66, 68);
        assert_eq!(
            first.key,
            CheckedDeclarationKey {
                namespace_path: vec!["N"],
                containing_type: Some(1),
                type_lineage: vec![("C", CsTypeKind::Class, 0), ("D", CsTypeKind::Class, 0)],
                name: "X",
                shape: DeclarationShape::Type(CsTypeKind::Struct, 0),
            }
        );
        assert_eq!(
            second.key,
            CheckedDeclarationKey {
                containing_type: Some(4),
                ..first.key
            }
        );
    }

    #[test]
    fn checked_type_keys_distinguish_arity_kind_and_namespace_case() {
        let source =
            "namespace N{class C<T>{} class C<T,U>{} struct C<T>{}} namespace n{class C<T>{}}";

        let index = index_checked(source, Span::new(0, source.len())).unwrap();

        let first = checked_declaration(&index, DeclarationRole::Type, 18, 19);
        assert_ne!(
            first.key,
            checked_declaration(&index, DeclarationRole::Type, 31, 32).key
        );
        assert_ne!(
            first.key,
            checked_declaration(&index, DeclarationRole::Type, 47, 48).key
        );
        assert_ne!(
            first.key,
            checked_declaration(&index, DeclarationRole::Type, 73, 74).key
        );
    }

    #[test]
    fn checked_method_keys_ignore_spelling_escape_return_and_parameter_names() {
        let source = "class C{void F(int x){} int @F(int y){}}";

        let index = index_checked(source, Span::new(0, source.len())).unwrap();

        let first = checked_declaration(&index, METHOD, 13, 14);
        let second = checked_declaration(&index, METHOD, 28, 30);
        assert_eq!(first.key, second.key);
        assert_eq!(
            [(first.id, first.name), (second.id, second.name)],
            [
                (DeclarationId::Callable(0), &name("F", 13, 14)),
                (DeclarationId::Callable(1), &name("@F", 28, 30)),
            ]
        );
    }

    #[test]
    fn checked_method_keys_distinguish_primitive_overloads() {
        let source = "class C{void F(int x){} void F(bool y){}}";

        let index = index_checked(source, Span::new(0, source.len())).unwrap();

        let int_method = checked_declaration(&index, METHOD, 13, 14);
        let bool_method = checked_declaration(&index, METHOD, 29, 30);
        assert_eq!(
            [int_method.key.shape, bool_method.key.shape],
            [
                DeclarationShape::Callable {
                    kind: CallableKind::Method,
                    arity: 0,
                    parameters: vec![CallableParameterType::Int],
                },
                DeclarationShape::Callable {
                    kind: CallableKind::Method,
                    arity: 0,
                    parameters: vec![CallableParameterType::Bool],
                },
            ]
        );
    }

    #[test]
    fn checked_method_keys_preserve_parameter_type_order() {
        let source = "class C{void F(int x,bool y){} void F(bool x,int y){}}";

        let index = index_checked(source, Span::new(0, 54)).unwrap();

        let first = checked_declaration(&index, METHOD, 13, 14);
        let second = checked_declaration(&index, METHOD, 36, 37);
        assert_eq!(
            [first.key.shape, second.key.shape],
            [
                DeclarationShape::Callable {
                    kind: CallableKind::Method,
                    arity: 0,
                    parameters: vec![CallableParameterType::Int, CallableParameterType::Bool],
                },
                DeclarationShape::Callable {
                    kind: CallableKind::Method,
                    arity: 0,
                    parameters: vec![CallableParameterType::Bool, CallableParameterType::Int],
                },
            ]
        );
    }

    #[test]
    fn checked_callable_keys_distinguish_constructor_kind_and_method_case() {
        let source = "class C{C(){} void C(){} void c(){}}";

        let index = index_checked(source, Span::new(0, 36)).unwrap();

        let ctor = checked_declaration(
            &index,
            DeclarationRole::Callable(CallableKind::Constructor),
            8,
            9,
        );
        let method = checked_declaration(&index, METHOD, 19, 20);
        assert_ne!(ctor.key, method.key);
        assert_ne!(method.key, checked_declaration(&index, METHOD, 30, 31).key);
    }

    #[test]
    fn checked_generic_method_keys_use_slots_and_arity_but_keep_distinct_occurrences() {
        let source = "class C{void F<T>(T x){} void F<U>(U y){} void F<T,V>(T z){}}";

        let index = index_checked(source, Span::new(0, 61)).unwrap();

        let first = checked_declaration(&index, METHOD, 13, 14);
        let second = checked_declaration(&index, METHOD, 30, 31);
        let third = checked_declaration(&index, METHOD, 47, 48);
        assert_eq!(
            [first.id, second.id, third.id],
            [
                DeclarationId::Callable(0),
                DeclarationId::Callable(1),
                DeclarationId::Callable(2),
            ]
        );
        assert_eq!(first.key, second.key);
        assert_eq!(
            [first.key.shape, third.key.shape],
            [
                DeclarationShape::Callable {
                    kind: CallableKind::Method,
                    arity: 1,
                    parameters: vec![CallableParameterType::MethodSlot(0)],
                },
                DeclarationShape::Callable {
                    kind: CallableKind::Method,
                    arity: 2,
                    parameters: vec![CallableParameterType::MethodSlot(0)],
                },
            ]
        );
    }

    #[test]
    fn checked_index_refuses_containing_type_binders_as_parameter_types() {
        let source = "class C<T>{void F(T x){}}";

        let index = index_checked(source, Span::new(0, 25));

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(Span::new(18, 19)),
                recovery: vec![],
            })
        );
    }

    #[test]
    fn checked_index_keeps_the_generic_method_owner_lineage_limitation() {
        let source = "class C<T>{class D{void F<T>(T x){}}}";

        let index = index_checked(source, Span::new(0, 37));

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(Span::new(25, 26)),
                recovery: vec![
                    Span::new(25, 26),
                    Span::new(26, 27),
                    Span::new(27, 29),
                    Span::new(32, 34),
                    Span::new(36, 36),
                    Span::new(36, 37),
                ],
            })
        );
    }

    #[test]
    fn checked_lexical_declaration_lookup_returns_exact_owner_local_ids() {
        let source = "class C{int a; class D{int a;} int @b; void F(int @p){int x;{bool x;x = @p;} bool y;}}";
        let index = index_checked(source, Span::new(0, 86)).unwrap();

        for (role, raw, start, end, expected) in [
            (
                DeclarationRole::Field,
                "a",
                27,
                28,
                DeclarationId::Field(1, 0),
            ),
            (
                DeclarationRole::Field,
                "@b",
                35,
                37,
                DeclarationId::Field(0, 1),
            ),
            (
                DeclarationRole::Parameter,
                "@p",
                50,
                52,
                DeclarationId::Parameter(0, 0),
            ),
            (
                DeclarationRole::Local,
                "x",
                66,
                67,
                DeclarationId::Local(0, 1, 0),
            ),
            (
                DeclarationRole::Local,
                "y",
                82,
                83,
                DeclarationId::Local(0, 0, 1),
            ),
        ] {
            let declaration = checked_declaration(&index, role, start, end);

            assert_eq!(
                (declaration.id, declaration.name),
                (expected, &name(raw, start, end))
            );
        }
    }

    #[test]
    fn checked_declaration_lookup_rejects_overlapping_spans_and_other_roles() {
        let source = "class C{void @F(){}}";
        let index = index_checked(source, Span::new(0, 20)).unwrap();

        assert_eq!(index.declaration_at(Span::new(14, 15), METHOD), None);
        assert_eq!(
            index.declaration_at(Span::new(13, 15), DeclarationRole::Type),
            None
        );
        assert_eq!(
            index.declaration_at(
                Span::new(13, 15),
                DeclarationRole::Callable(CallableKind::Constructor)
            ),
            None
        );
    }

    #[test]
    fn checked_evidence_lookup_preserves_raw_names_from_the_recorded_roles() {
        let source =
            "namespace @N{class C<T>{int x; void F(int y){bool z;}} class D{void G<U>(U @u){}}}";
        let index = index_checked(source, Span::new(0, source.len())).unwrap();

        for (role, raw, start, end) in [
            (EvidenceRole::NamespaceSegment, "@N", 10, 12),
            (EvidenceRole::TypeBinder, "T", 21, 22),
            (EvidenceRole::FieldType, "int", 24, 27),
            (EvidenceRole::ReturnType, "void", 31, 35),
            (EvidenceRole::LocalType, "bool", 45, 49),
            (EvidenceRole::MethodBinder, "U", 70, 71),
        ] {
            let evidence = index.evidence_at(Span::new(start, end), role);

            assert_eq!(
                evidence,
                Some(CheckedEvidence::Name(&name(raw, start, end)))
            );
        }
    }

    #[test]
    fn checked_parameter_evidence_preserves_the_raw_method_slot_use() {
        let source = "class C{void F<U>(U @u){}}";
        let index = index_checked(source, Span::new(0, source.len())).unwrap();

        let evidence = index.evidence_at(Span::new(18, 19), EvidenceRole::ParameterType);

        assert_eq!(
            evidence,
            Some(CheckedEvidence::ParameterType(&parameter(
                name("U", 18, 19),
                CallableParameterType::MethodSlot(0),
                name("@u", 20, 22),
            )))
        );
    }

    #[test]
    fn checked_evidence_lookup_distinguishes_occurrence_roles_in_an_assignment() {
        let source = "class C{int x; void F(int @p){x = @p;}}";
        let index = index_checked(source, Span::new(0, source.len())).unwrap();
        let span = Span::new(34, 36);

        assert_eq!(
            index.evidence_at(span, EvidenceRole::Reference(ReferenceRole::Read)),
            Some(CheckedEvidence::Reference(&ReferenceSite {
                ordinal: 1,
                owner: SiteOwner {
                    callable: 0,
                    block: 0
                },
                role: ReferenceRole::Read,
                name: name("@p", 34, 36),
                receiver: binding::Receiver::Bare,
                call_argument: None,
            }))
        );
        assert_eq!(
            index.evidence_at(span, EvidenceRole::Reference(ReferenceRole::Write)),
            None
        );
        assert_eq!(index.declaration_at(span, DeclarationRole::Parameter), None);
    }

    fn global_row(
        ordinal: usize,
        kind: CsTypeKind,
        name: &str,
        name_span: Span,
        header_span: Span,
        declaration_span: Span,
    ) -> TypeHeaderRow {
        TypeHeaderRow {
            ordinal,
            public_modifier: None,
            namespace: None,
            containing_type: None,
            kind,
            name: CsName {
                text: name.to_owned(),
                span: name_span,
            },
            type_params: Vec::new(),
            header_span,
            declaration_span,
        }
    }

    fn assert_qualified_type(
        lookup: &QualifiedTypeLookup<'_>,
        path: &[&str],
        expected: Option<usize>,
    ) {
        let ordinal = lookup.resolve(path);

        assert_eq!(ordinal, expected, "path {path:?}");
    }

    fn type_index_for_lookup(source: &str) -> TypeHeaderIndex {
        index_type_headers(source, Span::new(0, source.len()))
            .expect("fixture must produce a complete type index")
    }

    #[test]
    fn qualified_lookup_returns_ordinals_for_global_and_nested_types() {
        let index = type_index_for_lookup(
            "class Outer { struct Inner { interface Contract {} } } \
             enum Empty {} class Other { struct Inner {} }",
        );
        let lookup = QualifiedTypeLookup::new(&index);

        for (path, ordinal) in [
            (&["Outer"][..], 0),
            (&["Outer", "Inner"][..], 1),
            (&["Outer", "Inner", "Contract"][..], 2),
            (&["Empty"][..], 3),
            (&["Other", "Inner"][..], 5),
        ] {
            assert_qualified_type(&lookup, path, Some(ordinal));
        }
    }

    #[test]
    fn qualified_lookup_returns_ordinals_from_each_reopening_of_a_dotted_namespace() {
        let index = type_index_for_lookup(
            "namespace @N /* gap */ . Sub { class First {} } namespace N.Sub { class Second {} }",
        );
        let lookup = QualifiedTypeLookup::new(&index);

        for (path, ordinal) in [(["N", "@Sub", "First"], 0), (["@N", "Sub", "Second"], 1)] {
            assert_qualified_type(&lookup, &path, Some(ordinal));
        }
    }

    #[test]
    fn qualified_lookup_returns_the_terminal_ordinal_in_a_file_scoped_namespace() {
        let index = type_index_for_lookup("namespace N.Sub; class Owner { class @Leaf {} }");
        let lookup = QualifiedTypeLookup::new(&index);

        assert_qualified_type(&lookup, &["N", "Sub", "Owner", "Leaf"], Some(1));
    }

    #[test]
    fn qualified_lookup_returns_none_when_the_requested_path_is_missing_or_namespace_only() {
        let index = type_index_for_lookup("namespace N { class Found {} } class Other {}");
        let lookup = QualifiedTypeLookup::new(&index);

        for path in [
            &[][..],
            &["N"][..],
            &["N", "Other"][..],
            &["Missing", "Found"][..],
        ] {
            assert_qualified_type(&lookup, path, None);
        }
    }

    #[test]
    fn qualified_lookup_returns_none_for_duplicate_terminals_across_namespace_reopenings() {
        let index =
            type_index_for_lookup("namespace N { class Item {} } namespace N { struct @Item {} }");
        let lookup = QualifiedTypeLookup::new(&index);

        assert_qualified_type(&lookup, &["N", "Item"], None);
    }

    #[test]
    fn qualified_lookup_returns_none_for_duplicate_owners_even_with_a_single_leaf() {
        let index = type_index_for_lookup("class A { class Other {} } class A { struct Leaf {} }");
        let lookup = QualifiedTypeLookup::new(&index);

        for path in [["A", "Other"], ["A", "Leaf"]] {
            assert_qualified_type(&lookup, &path, None);
        }
    }

    #[test]
    fn qualified_lookup_returns_none_for_a_namespace_type_prefix_collision() {
        let index = type_index_for_lookup(
            "namespace N { class A { class TypeLeaf {} } } namespace N.A { class NamespaceLeaf {} }",
        );
        let lookup = QualifiedTypeLookup::new(&index);

        for path in [["N", "A", "TypeLeaf"], ["N", "A", "NamespaceLeaf"]] {
            assert_qualified_type(&lookup, &path, None);
        }
    }

    #[test]
    fn qualified_lookup_returns_none_for_a_namespace_type_terminal_collision() {
        let index = type_index_for_lookup("namespace N { class A {} } namespace N.A.More {}");
        let lookup = QualifiedTypeLookup::new(&index);

        assert_qualified_type(&lookup, &["N", "A"], None);
    }

    #[test]
    fn qualified_lookup_returns_none_for_a_generic_terminal() {
        let index = type_index_for_lookup("class Owner { class Leaf<T> {} }");
        let lookup = QualifiedTypeLookup::new(&index);

        assert_qualified_type(&lookup, &["Owner", "Leaf"], None);
    }

    #[test]
    fn qualified_lookup_returns_none_for_a_leaf_under_a_generic_ancestor() {
        let index =
            type_index_for_lookup("namespace N { class A<T> { class B { class Leaf {} } } }");
        let lookup = QualifiedTypeLookup::new(&index);

        assert_qualified_type(&lookup, &["N", "A", "B", "Leaf"], None);
    }

    #[test]
    fn qualified_lookup_counts_a_generic_competitor_at_the_prefix_or_terminal() {
        let index = type_index_for_lookup("class A { class Leaf {} } class A<T> {}");
        let lookup = QualifiedTypeLookup::new(&index);

        for path in [&["A"][..], &["A", "Leaf"][..]] {
            assert_qualified_type(&lookup, path, None);
        }
    }

    #[test]
    fn qualified_lookup_compares_exact_identifier_bytes_after_one_escape() {
        let index = type_index_for_lookup("namespace N { class @Café {} }");
        let lookup = QualifiedTypeLookup::new(&index);

        for (path, expected) in [
            (["N", "Café"], Some(0)),
            (["N", "café"], None),
            (["N", "Cafe\u{301}"], None),
            (["N", "@@Café"], None),
        ] {
            assert_qualified_type(&lookup, &path, expected);
        }
    }

    #[test]
    fn index_returns_one_nongeneric_global_class_row_for_simple_class() {
        let source = "class C {}";
        let region = Span::new(0, 10);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: Vec::new(),
                rows: vec![global_row(
                    0,
                    CsTypeKind::Class,
                    "C",
                    Span::new(6, 7),
                    Span::new(0, 7),
                    Span::new(0, 10),
                )],
            })
        );
    }

    #[test]
    fn index_returns_source_ordered_absolute_rows_for_a_region() {
        let source = "é\n public class A /* gap */ { }\nstruct B{} interface I {} enum E {} ...";
        let region = Span::new(3, 69);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: Vec::new(),
                rows: vec![
                    global_row(
                        0,
                        CsTypeKind::Class,
                        "A",
                        Span::new(17, 18),
                        Span::new(11, 18),
                        Span::new(4, 32),
                    ),
                    global_row(
                        1,
                        CsTypeKind::Struct,
                        "B",
                        Span::new(40, 41),
                        Span::new(33, 41),
                        Span::new(33, 43),
                    ),
                    global_row(
                        2,
                        CsTypeKind::Interface,
                        "I",
                        Span::new(54, 55),
                        Span::new(44, 55),
                        Span::new(44, 58),
                    ),
                    global_row(
                        3,
                        CsTypeKind::Enum,
                        "E",
                        Span::new(64, 65),
                        Span::new(59, 65),
                        Span::new(59, 68),
                    ),
                ],
            })
        );
    }

    #[test]
    fn index_records_generic_and_nested_rows_in_namespace_source_order() {
        let source = "namespace Q.R { class C<T> { class D {} } } namespace Z { class C {} }";
        let region = Span::new(0, 70);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: vec![
                    NamespaceOccurrence {
                        ordinal: 0,
                        segments: vec![name("Q", 10, 11), name("R", 12, 13)],
                    },
                    NamespaceOccurrence {
                        ordinal: 1,
                        segments: vec![name("Z", 54, 55)],
                    },
                ],
                rows: vec![
                    TypeHeaderRow {
                        namespace: Some(0),
                        type_params: vec![name("T", 24, 25)],
                        ..global_row(
                            0,
                            CsTypeKind::Class,
                            "C",
                            Span::new(22, 23),
                            Span::new(16, 26),
                            Span::new(16, 41),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        containing_type: Some(0),
                        ..global_row(
                            1,
                            CsTypeKind::Class,
                            "D",
                            Span::new(35, 36),
                            Span::new(29, 36),
                            Span::new(29, 39),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(1),
                        ..global_row(
                            2,
                            CsTypeKind::Class,
                            "C",
                            Span::new(64, 65),
                            Span::new(58, 65),
                            Span::new(58, 68),
                        )
                    },
                ],
            })
        );
    }

    #[test]
    fn index_retains_generic_struct_spans_in_an_original_utf8_file_namespace_region() {
        let source = "\u{feff}π\nnamespace Ω; struct Δ<T,U> {}!";
        let region = Span::new(6, 37);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: vec![NamespaceOccurrence {
                    ordinal: 0,
                    segments: vec![name("Ω", 16, 18)],
                }],
                rows: vec![TypeHeaderRow {
                    namespace: Some(0),
                    type_params: vec![name("T", 30, 31), name("U", 32, 33)],
                    ..global_row(
                        0,
                        CsTypeKind::Struct,
                        "Δ",
                        Span::new(27, 29),
                        Span::new(20, 34),
                        Span::new(20, 37),
                    )
                }],
            })
        );
    }

    #[test]
    fn index_preserves_raw_interface_parameter_names_across_trivia() {
        let source = "public partial interface I<T, /* gap */ @T, t> {}";
        let region = Span::new(0, 49);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: Vec::new(),
                rows: vec![TypeHeaderRow {
                    type_params: vec![name("T", 27, 28), name("@T", 40, 42), name("t", 44, 45)],
                    ..global_row(
                        0,
                        CsTypeKind::Interface,
                        "I",
                        Span::new(25, 26),
                        Span::new(15, 46),
                        Span::new(0, 49),
                    )
                }],
            })
        );
    }

    #[test]
    fn index_limits_same_spelled_block_namespace_occurrences_to_their_own_bodies() {
        let source = "class C {} namespace @Q /* gap */ . R { class C {} } namespace @Q.R { class C {} } class C {}";
        let region = Span::new(0, 93);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: vec![
                    NamespaceOccurrence {
                        ordinal: 0,
                        segments: vec![name("@Q", 21, 23), name("R", 36, 37)],
                    },
                    NamespaceOccurrence {
                        ordinal: 1,
                        segments: vec![name("@Q", 63, 65), name("R", 66, 67)],
                    },
                ],
                rows: vec![
                    global_row(
                        0,
                        CsTypeKind::Class,
                        "C",
                        Span::new(6, 7),
                        Span::new(0, 7),
                        Span::new(0, 10),
                    ),
                    TypeHeaderRow {
                        namespace: Some(0),
                        ..global_row(
                            1,
                            CsTypeKind::Class,
                            "C",
                            Span::new(46, 47),
                            Span::new(40, 47),
                            Span::new(40, 50),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(1),
                        ..global_row(
                            2,
                            CsTypeKind::Class,
                            "C",
                            Span::new(76, 77),
                            Span::new(70, 77),
                            Span::new(70, 80),
                        )
                    },
                    global_row(
                        3,
                        CsTypeKind::Class,
                        "C",
                        Span::new(89, 90),
                        Span::new(83, 90),
                        Span::new(83, 93),
                    ),
                ],
            })
        );
    }

    #[test]
    fn index_assigns_nested_rows_to_their_exact_owners_and_restores_the_owner_for_siblings() {
        let source = "namespace N { class Outer { class Item { struct Leaf {} } class Leaf {} } class Outer { struct Item {} } }";
        let region = Span::new(0, 106);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: vec![NamespaceOccurrence {
                    ordinal: 0,
                    segments: vec![name("N", 10, 11)],
                }],
                rows: vec![
                    TypeHeaderRow {
                        namespace: Some(0),
                        ..global_row(
                            0,
                            CsTypeKind::Class,
                            "Outer",
                            Span::new(20, 25),
                            Span::new(14, 25),
                            Span::new(14, 73),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        containing_type: Some(0),
                        ..global_row(
                            1,
                            CsTypeKind::Class,
                            "Item",
                            Span::new(34, 38),
                            Span::new(28, 38),
                            Span::new(28, 57),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        containing_type: Some(1),
                        ..global_row(
                            2,
                            CsTypeKind::Struct,
                            "Leaf",
                            Span::new(48, 52),
                            Span::new(41, 52),
                            Span::new(41, 55),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        containing_type: Some(0),
                        ..global_row(
                            3,
                            CsTypeKind::Class,
                            "Leaf",
                            Span::new(64, 68),
                            Span::new(58, 68),
                            Span::new(58, 71),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        ..global_row(
                            4,
                            CsTypeKind::Class,
                            "Outer",
                            Span::new(80, 85),
                            Span::new(74, 85),
                            Span::new(74, 104),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        containing_type: Some(4),
                        ..global_row(
                            5,
                            CsTypeKind::Struct,
                            "Item",
                            Span::new(95, 99),
                            Span::new(88, 99),
                            Span::new(88, 102),
                        )
                    },
                ],
            })
        );
    }

    #[test]
    fn index_scopes_a_file_namespace_through_region_eof_with_absolute_offsets() {
        let source = "é\nnamespace @N.Sub; class A { class B {} } class C {} | class Outside {}";
        let region = Span::new(3, 55);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: vec![NamespaceOccurrence {
                    ordinal: 0,
                    segments: vec![name("@N", 13, 15), name("Sub", 16, 19)],
                }],
                rows: vec![
                    TypeHeaderRow {
                        namespace: Some(0),
                        ..global_row(
                            0,
                            CsTypeKind::Class,
                            "A",
                            Span::new(27, 28),
                            Span::new(21, 28),
                            Span::new(21, 43),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        containing_type: Some(0),
                        ..global_row(
                            1,
                            CsTypeKind::Class,
                            "B",
                            Span::new(37, 38),
                            Span::new(31, 38),
                            Span::new(31, 41),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        ..global_row(
                            2,
                            CsTypeKind::Class,
                            "C",
                            Span::new(50, 51),
                            Span::new(44, 51),
                            Span::new(44, 54),
                        )
                    },
                ],
            })
        );
    }

    fn assert_empty_index(source: &str, region: Span) {
        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: Vec::new(),
                rows: Vec::new(),
            }),
            "region {region} in {source:?}"
        );
    }

    #[test]
    fn index_returns_zero_rows_for_empty_regions_including_eof() {
        for (source, region) in [
            ("", Span::new(0, 0)),
            ("éx", Span::new(2, 2)),
            ("éx", Span::new(3, 3)),
        ] {
            assert_empty_index(source, region);
        }
    }

    #[test]
    fn index_records_empty_block_namespaces_without_type_rows() {
        let source = "namespace N { /* empty */ }";
        let region = Span::new(0, 27);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Ok(TypeHeaderIndex {
                region,
                namespaces: vec![NamespaceOccurrence {
                    ordinal: 0,
                    segments: vec![name("N", 10, 11)],
                }],
                rows: Vec::new(),
            })
        );
    }

    fn assert_invalid_region(source: &str, region: Span) {
        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Err(TypeIndexError::InvalidRegion(region)),
            "region {region} in {source:?}"
        );
    }

    #[test]
    fn index_refuses_reversed_out_of_bounds_and_split_utf8_regions() {
        let source = "éx";

        for region in [
            Span { start: 3, end: 2 },
            Span::new(0, usize::MAX),
            Span::new(4, 4),
            Span::new(1, 3),
            Span::new(0, 1),
            Span::new(1, 1),
        ] {
            assert_invalid_region(source, region);
        }
    }

    fn assert_unsupported_syntax(source: &str, span: Span) {
        let index = index_type_headers(source, Span::new(0, source.len()));

        assert_eq!(
            index,
            Err(TypeIndexError::UnsupportedSyntax(span)),
            "source {source:?}"
        );
    }

    #[test]
    fn index_refuses_file_scope_after_a_global_type_without_returning_earlier_rows() {
        let source = "class Done {} namespace N;";

        assert_unsupported_syntax(source, Span::new(14, 23));
    }

    #[test]
    fn index_refuses_namespaces_with_missing_segments_or_delimiters() {
        for (source, span) in [
            ("namespace {}", Span::new(10, 11)),
            ("namespace Q. { class C {} }", Span::new(11, 12)),
            ("namespace Q. ; class C {}", Span::new(11, 12)),
            ("namespace Q", Span::new(11, 11)),
            ("namespace Q class C {}", Span::new(12, 17)),
            ("namespace Q { class C {}  ", Span::new(26, 26)),
        ] {
            assert_unsupported_syntax(source, span);
        }
    }

    #[test]
    fn index_refuses_namespaces_in_member_dispatch_and_file_scope_after_a_block() {
        for (source, span) in [
            ("namespace A { namespace B {} }", Span::new(14, 23)),
            ("namespace A { namespace B; }", Span::new(14, 23)),
            ("class A { namespace B; }", Span::new(10, 19)),
            ("namespace A; namespace B; class C {}", Span::new(13, 22)),
            ("namespace A; namespace B {} class C {}", Span::new(13, 22)),
            ("namespace A {} namespace B; class C {}", Span::new(15, 24)),
            ("public namespace N {}", Span::new(0, 6)),
        ] {
            assert_unsupported_syntax(source, span);
        }
    }

    #[test]
    fn index_refuses_incomplete_headers_and_bodies() {
        for (source, span) in [
            ("class {}", Span::new(6, 7)),
            ("class { class D {} }", Span::new(6, 7)),
            ("class C", Span::new(7, 7)),
            ("class C {", Span::new(9, 9)),
        ] {
            assert_unsupported_syntax(source, span);
        }
    }

    #[test]
    fn index_refuses_forms_outside_the_simple_type_subset() {
        for (source, span) in [
            ("using N;", Span::new(0, 5)),
            ("int F;", Span::new(0, 3)),
            ("public void M() {}", Span::new(7, 11)),
            ("record R {}", Span::new(0, 6)),
            ("delegate void D();", Span::new(0, 8)),
            ("[Attr] class C {}", Span::new(0, 1)),
            ("class C : B {}", Span::new(8, 9)),
            ("class C<T> : B<T> {}", Span::new(11, 12)),
            ("class C where T : B {}", Span::new(8, 13)),
            ("class C<T> where T : B {}", Span::new(11, 16)),
            ("class C;", Span::new(7, 8)),
            ("class C<T>;", Span::new(10, 11)),
            ("class C { [Attr] class D {} }", Span::new(10, 11)),
            ("class C { int F; }", Span::new(10, 13)),
            ("class C { class D {} int F; }", Span::new(21, 24)),
            ("enum E { A }", Span::new(9, 10)),
        ] {
            assert_unsupported_syntax(source, span);
        }
    }

    #[test]
    fn index_refuses_invalid_generic_parameter_lists_at_the_offending_token_or_boundary() {
        for (source, span) in [
            ("class C<> {}", Span::new(8, 9)),
            ("class C<,T> {}", Span::new(8, 9)),
            ("class C<T,,U> {}", Span::new(10, 11)),
            ("class C<T,> { class D {} }", Span::new(10, 11)),
            ("class A {} class C<T,> {}", Span::new(21, 22)),
            ("class C<T U> {}", Span::new(10, 11)),
            ("class C<T,T> {}", Span::new(10, 11)),
            ("class C<T,@T,@T> {}", Span::new(13, 15)),
            ("class C<T<U>> {}", Span::new(9, 10)),
            ("class C<N.T> {}", Span::new(9, 10)),
            ("class C<123> {}", Span::new(8, 11)),
            ("interface I<in T> {}", Span::new(12, 14)),
            ("interface I<out T> {}", Span::new(12, 15)),
            ("struct S<[A] T> {}", Span::new(9, 10)),
            ("class C<T { class D {} }", Span::new(10, 11)),
            ("class C<T", Span::new(9, 9)),
            ("enum E<T> {}", Span::new(6, 7)),
        ] {
            assert_unsupported_syntax(source, span);
        }
    }

    #[test]
    fn index_refuses_a_bom_at_the_start_of_the_selected_region() {
        let source = "π|\u{feff}class C {}";
        let region = Span::new(3, 16);

        let index = index_type_headers(source, region);

        assert_eq!(
            index,
            Err(TypeIndexError::UnsupportedSyntax(Span::new(3, 6)))
        );
    }

    fn assert_recovered(source: &str, region: Span, diagnostics: Vec<Span>) {
        let index = index_type_headers(source, region);

        assert_eq!(index, Err(TypeIndexError::Recovered(diagnostics)));
    }

    #[test]
    fn index_returns_ordered_absolute_legacy_diagnostics_before_subset_refusals() {
        for (source, region, diagnostics) in [
            (
                "class C {} }",
                Span::new(0, 12),
                vec![Span::new(11, 11), Span::new(11, 12)],
            ),
            (
                "π|using N; class C {} } ]",
                Span::new(3, 26),
                vec![Span::new(23, 23), Span::new(23, 24), Span::new(25, 26)],
            ),
            (
                "class C<T,> {} }",
                Span::new(0, 16),
                vec![Span::new(15, 15), Span::new(15, 16)],
            ),
        ] {
            assert_recovered(source, region, diagnostics);
        }
    }

    #[test]
    fn index_returns_recovered_at_the_legacy_type_depth_boundary() {
        let source = format!(
            "{}class Leaf<T> {{}}{} class Sibling {{}}",
            "class N { ".repeat(199),
            "}".repeat(199),
        );
        let region = Span::new(0, source.len());

        assert_recovered(&source, region, vec![Span::new(1990, 2006)]);
    }

    const CALLABLE_REGION_SOURCE: &str =
        "π|class C{C(){} void F(int x){} int F(int y){} void F(bool x){}}";

    fn parameter(
        type_name: CsName,
        comparison_type: CallableParameterType,
        name: CsName,
    ) -> CallableParameter {
        CallableParameter {
            type_use: CallableParameterTypeUse::Simple(type_name),
            comparison_type,
            name,
        }
    }

    #[test]
    fn callable_index_preserves_source_order_and_absolute_header_evidence() {
        let region = Span::new(3, 65);

        let index = index_callable_headers(CALLABLE_REGION_SOURCE, region);

        assert_eq!(
            index,
            Ok(CallableHeaderIndex {
                types: TypeHeaderIndex {
                    region,
                    namespaces: Vec::new(),
                    rows: vec![global_row(
                        0,
                        CsTypeKind::Class,
                        "C",
                        Span::new(9, 10),
                        Span::new(3, 10),
                        Span::new(3, 65),
                    )],
                },
                callables: vec![
                    CallableHeaderRow {
                        ordinal: 0,
                        containing_type: 0,
                        kind: CallableKind::Constructor,
                        name: name("C", 11, 12),
                        return_type: None,
                        modifiers: MemberModifiers::default(),
                        type_params: Vec::new(),
                        parameters: Vec::new(),
                        header_span: Span::new(11, 14),
                        declaration_span: Span::new(11, 16),
                    },
                    CallableHeaderRow {
                        ordinal: 1,
                        containing_type: 0,
                        kind: CallableKind::Method,
                        name: name("F", 22, 23),
                        return_type: Some(name("void", 17, 21)),
                        modifiers: MemberModifiers::default(),
                        type_params: Vec::new(),
                        parameters: vec![parameter(
                            name("int", 24, 27),
                            CallableParameterType::Int,
                            name("x", 28, 29),
                        )],
                        header_span: Span::new(17, 30),
                        declaration_span: Span::new(17, 32),
                    },
                    CallableHeaderRow {
                        ordinal: 2,
                        containing_type: 0,
                        kind: CallableKind::Method,
                        name: name("F", 37, 38),
                        return_type: Some(name("int", 33, 36)),
                        modifiers: MemberModifiers::default(),
                        type_params: Vec::new(),
                        parameters: vec![parameter(
                            name("int", 39, 42),
                            CallableParameterType::Int,
                            name("y", 43, 44),
                        )],
                        header_span: Span::new(33, 45),
                        declaration_span: Span::new(33, 47),
                    },
                    CallableHeaderRow {
                        ordinal: 3,
                        containing_type: 0,
                        kind: CallableKind::Method,
                        name: name("F", 53, 54),
                        return_type: Some(name("void", 48, 52)),
                        modifiers: MemberModifiers::default(),
                        type_params: Vec::new(),
                        parameters: vec![parameter(
                            name("bool", 55, 59),
                            CallableParameterType::Bool,
                            name("x", 60, 61),
                        )],
                        header_span: Span::new(48, 62),
                        declaration_span: Span::new(48, 64),
                    },
                ],
            })
        );
    }

    #[test]
    fn callable_keys_match_across_return_and_parameter_name_changes_but_not_type_changes() {
        let region = Span::new(3, 65);
        let method_key = |parameter_types| CallableKey {
            containing_type: 0,
            kind: CallableKind::Method,
            name: "F",
            arity: 0,
            parameter_types,
        };

        let index = index_callable_headers(CALLABLE_REGION_SOURCE, region).unwrap();

        assert_eq!(
            [
                index.callables[0].key(),
                index.callables[1].key(),
                index.callables[2].key(),
                index.callables[3].key(),
            ],
            [
                CallableKey {
                    containing_type: 0,
                    kind: CallableKind::Constructor,
                    name: "C",
                    arity: 0,
                    parameter_types: Vec::new(),
                },
                method_key(vec![CallableParameterType::Int]),
                method_key(vec![CallableParameterType::Int]),
                method_key(vec![CallableParameterType::Bool]),
            ]
        );
    }

    #[test]
    fn callable_keys_distinguish_parameter_order() {
        let source = "class C{bool F(int x,bool y){} bool F(bool x,int y){}}";

        let index = index_callable_headers(source, Span::new(0, 54)).unwrap();

        assert_ne!(index.callables[0].key(), index.callables[1].key());
    }

    #[test]
    fn callable_index_uses_the_nested_struct_owner_then_restores_the_class_owner() {
        let source = "namespace N; class @C{void F(){} struct S{S(int x,bool y){/* } */} void F(){}} @C(){} void F(){}}";

        let index = index_callable_headers(source, Span::new(0, 97)).unwrap();

        assert_eq!(
            index.types,
            TypeHeaderIndex {
                region: Span::new(0, 97),
                namespaces: vec![NamespaceOccurrence {
                    ordinal: 0,
                    segments: vec![name("N", 10, 11)],
                }],
                rows: vec![
                    TypeHeaderRow {
                        namespace: Some(0),
                        ..global_row(
                            0,
                            CsTypeKind::Class,
                            "@C",
                            Span::new(19, 21),
                            Span::new(13, 21),
                            Span::new(13, 97),
                        )
                    },
                    TypeHeaderRow {
                        namespace: Some(0),
                        containing_type: Some(0),
                        ..global_row(
                            1,
                            CsTypeKind::Struct,
                            "S",
                            Span::new(40, 41),
                            Span::new(33, 41),
                            Span::new(33, 78),
                        )
                    }
                ],
            }
        );
        assert_eq!(
            index
                .callables
                .iter()
                .map(|row| (row.ordinal, row.containing_type, row.name.text.as_str()))
                .collect::<Vec<_>>(),
            [
                (0, 0, "F"),
                (1, 1, "S"),
                (2, 1, "F"),
                (3, 0, "@C"),
                (4, 0, "F"),
            ]
        );
        assert_ne!(index.callables[0].key(), index.callables[2].key());
    }

    #[test]
    fn type_only_index_still_refuses_the_constructor_start() {
        let region = Span::new(3, 65);

        let index = index_type_headers(CALLABLE_REGION_SOURCE, region);

        assert_eq!(
            index,
            Err(TypeIndexError::UnsupportedSyntax(Span::new(11, 12)))
        );
    }

    fn assert_callable_refusal(source: &str, expected: TypeIndexError) {
        let index = index_callable_headers(source, Span::new(0, source.len()));

        assert_eq!(index, Err(expected), "source {source:?}");
    }

    #[test]
    fn callable_index_refuses_a_field_after_an_earlier_valid_callable() {
        assert_callable_refusal(
            "class C{void F(){} int x;}",
            TypeIndexError::UnsupportedSyntax(Span::new(24, 25)),
        );
    }

    #[test]
    fn callable_index_refuses_members_outside_the_header_subset() {
        for (source, span) in [
            ("void F(){}", Span::new(0, 4)),
            ("namespace N; void F(){}", Span::new(13, 17)),
            ("interface I{void F(){}}", Span::new(12, 16)),
            ("class @C{C(){}}", Span::new(9, 10)),
            ("class C{int P{get;}}", Span::new(13, 14)),
            ("class C{public void F(){}}", Span::new(8, 14)),
            ("class C{[A] void F(){}}", Span::new(8, 9)),
            ("class C{string F(){}}", Span::new(8, 14)),
            ("class C{int[] F(){}}", Span::new(11, 12)),
        ] {
            assert_callable_refusal(source, TypeIndexError::UnsupportedSyntax(span));
        }
    }

    #[test]
    fn callable_index_refuses_parameters_outside_the_by_value_subset() {
        for (source, span) in [
            ("class C{void F(int){}}", Span::new(18, 19)),
            ("class C{void F(void x){}}", Span::new(15, 19)),
            ("class C{void F(ref int x){}}", Span::new(15, 18)),
            ("class C{void F([A]int x){}}", Span::new(15, 16)),
            ("class C{void F(int x=0){}}", Span::new(20, 21)),
            ("class C{void F(int x,){}}", Span::new(21, 22)),
            ("class C{void F(int x bool y){}}", Span::new(21, 25)),
        ] {
            assert_callable_refusal(source, TypeIndexError::UnsupportedSyntax(span));
        }
    }

    #[test]
    fn callable_index_requires_actual_delimiters_and_an_empty_block() {
        for (source, span) in [
            ("class C{void F();}", Span::new(16, 17)),
            ("class C{int F()=>0;}", Span::new(15, 17)),
            ("class C{void F(){;}}", Span::new(17, 18)),
            ("class C{void F(int x", Span::new(20, 20)),
            ("class C{void F(){", Span::new(17, 17)),
        ] {
            assert_callable_refusal(source, TypeIndexError::UnsupportedSyntax(span));
        }
    }

    #[test]
    fn callable_index_returns_ordered_legacy_recovery_before_subset_errors() {
        for (source, diagnostics) in [
            (
                "class C{void F(int){}} }",
                vec![Span::new(23, 23), Span::new(23, 24)],
            ),
            ("class C{C():base(){}}", vec![Span::new(11, 12)]),
            (
                "class C{void F<T>(){}}",
                vec![
                    Span::new(14, 15),
                    Span::new(15, 16),
                    Span::new(16, 20),
                    Span::new(21, 21),
                    Span::new(21, 22),
                ],
            ),
        ] {
            assert_callable_refusal(source, TypeIndexError::Recovered(diagnostics));
        }
    }

    #[test]
    fn callable_index_rejects_a_region_that_splits_a_utf8_character() {
        let source = "π|class C{C(){}}";
        let region = Span::new(1, 17);

        let index = index_callable_headers(source, region);

        assert_eq!(index, Err(TypeIndexError::InvalidRegion(region)));
    }

    #[test]
    fn callable_index_refuses_a_bom_at_the_selected_region_start() {
        let source = "π|\u{feff}class C {}";
        let region = Span::new(3, 16);

        let index = index_callable_headers(source, region);

        assert_eq!(
            index,
            Err(TypeIndexError::UnsupportedSyntax(Span::new(3, 6)))
        );
    }

    const GENERIC_METHOD_SOURCE: &str = "class C{void F<T>(T x){} void F(int y){}}";

    #[test]
    fn generic_callable_index_keeps_both_mixed_overloads_in_the_same_owner() {
        let region = Span::new(0, 41);

        let index = index_generic_callable_headers(GENERIC_METHOD_SOURCE, region);

        assert_eq!(
            index,
            Ok(CallableHeaderIndex {
                types: TypeHeaderIndex {
                    region,
                    namespaces: vec![],
                    rows: vec![global_row(
                        0,
                        CsTypeKind::Class,
                        "C",
                        Span::new(6, 7),
                        Span::new(0, 7),
                        Span::new(0, 41),
                    )],
                },
                callables: vec![
                    CallableHeaderRow {
                        ordinal: 0,
                        containing_type: 0,
                        kind: CallableKind::Method,
                        name: name("F", 13, 14),
                        return_type: Some(name("void", 8, 12)),
                        modifiers: MemberModifiers::default(),
                        type_params: vec![name("T", 15, 16)],
                        parameters: vec![parameter(
                            name("T", 18, 19),
                            CallableParameterType::MethodSlot(0),
                            name("x", 20, 21),
                        )],
                        header_span: Span::new(8, 22),
                        declaration_span: Span::new(8, 24),
                    },
                    CallableHeaderRow {
                        ordinal: 1,
                        containing_type: 0,
                        kind: CallableKind::Method,
                        name: name("F", 30, 31),
                        return_type: Some(name("void", 25, 29)),
                        modifiers: MemberModifiers::default(),
                        type_params: vec![],
                        parameters: vec![parameter(
                            name("int", 32, 35),
                            CallableParameterType::Int,
                            name("y", 36, 37),
                        )],
                        header_span: Span::new(25, 38),
                        declaration_span: Span::new(25, 40),
                    },
                ],
            }),
        );
    }

    #[test]
    fn generic_callable_index_keeps_equal_shapes_as_distinct_rows() {
        let source = "class C{void F<T>(T x){} void F<U>(U y){}}";

        let index = index_generic_callable_headers(source, Span::new(0, 42)).unwrap();

        let [first, second] = index.callables.as_slice() else {
            panic!("expected two callable rows, got {:?}", index.callables);
        };
        assert_eq!(first.key(), second.key());
        assert_eq!((first.ordinal, second.ordinal), (0, 1));
        assert_eq!(
            (&second.type_params[..], &second.parameters[..]),
            (
                &[name("U", 32, 33)][..],
                &[parameter(
                    name("U", 35, 36),
                    CallableParameterType::MethodSlot(0),
                    name("y", 37, 38),
                )][..],
            ),
        );
    }

    #[test]
    fn generic_callable_index_keeps_binder_order_and_parameter_slot_order_separate() {
        let source = "class C{void F<T,U>(U x,T y){}}";

        let index = index_generic_callable_headers(source, Span::new(0, 31)).unwrap();

        assert_eq!(
            index.callables,
            [CallableHeaderRow {
                ordinal: 0,
                containing_type: 0,
                kind: CallableKind::Method,
                name: name("F", 13, 14),
                return_type: Some(name("void", 8, 12)),
                modifiers: MemberModifiers::default(),
                type_params: vec![name("T", 15, 16), name("U", 17, 18)],
                parameters: vec![
                    parameter(
                        name("U", 20, 21),
                        CallableParameterType::MethodSlot(1),
                        name("x", 22, 23),
                    ),
                    parameter(
                        name("T", 24, 25),
                        CallableParameterType::MethodSlot(0),
                        name("y", 26, 27),
                    ),
                ],
                header_span: Span::new(8, 28),
                declaration_span: Span::new(8, 30),
            }],
        );
    }

    #[test]
    fn generic_callable_index_keeps_absolute_byte_spans_in_a_utf8_region() {
        let source = "π|class C{void F<T>(T x){}}|";
        let region = Span::new(3, 28);

        let index = index_generic_callable_headers(source, region).unwrap();

        assert_eq!(
            (index.types.region, index.callables),
            (
                region,
                vec![CallableHeaderRow {
                    ordinal: 0,
                    containing_type: 0,
                    kind: CallableKind::Method,
                    name: name("F", 16, 17),
                    return_type: Some(name("void", 11, 15)),
                    modifiers: MemberModifiers::default(),
                    type_params: vec![name("T", 18, 19)],
                    parameters: vec![parameter(
                        name("T", 21, 22),
                        CallableParameterType::MethodSlot(0),
                        name("x", 23, 24),
                    )],
                    header_span: Span::new(11, 25),
                    declaration_span: Span::new(11, 27),
                }],
            ),
        );
    }

    fn assert_generic_callable_refusal(
        source: &str,
        first_unsupported: Option<Span>,
        recovery: Vec<Span>,
    ) {
        let index = index_generic_callable_headers(source, Span::new(0, source.len()));

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported,
                recovery,
            }),
            "source {source:?}",
        );
    }

    #[test]
    fn generic_callable_index_refuses_at_open_angle_in_a_generic_type_lineage() {
        for (source, first, recovery) in [
            (
                "class C<T>{void F<T>(T x){}}",
                Span::new(17, 18),
                vec![
                    Span::new(17, 18),
                    Span::new(18, 19),
                    Span::new(19, 21),
                    Span::new(24, 26),
                    Span::new(27, 27),
                    Span::new(27, 28),
                ],
            ),
            (
                "class C<T>{class D{void F<T>(T x){}}}",
                Span::new(25, 26),
                vec![
                    Span::new(25, 26),
                    Span::new(26, 27),
                    Span::new(27, 29),
                    Span::new(32, 34),
                    Span::new(36, 36),
                    Span::new(36, 37),
                ],
            ),
        ] {
            assert_generic_callable_refusal(source, Some(first), recovery);
        }
    }

    #[test]
    fn generic_callable_index_admits_simple_methods_and_restores_nongeneric_lineage() {
        let source = "class C<T>{class D{void F(int x){}}} class E{void F<T>(T x){}}";

        let index = index_generic_callable_headers(source, Span::new(0, 62)).unwrap();

        assert_eq!(
            index.callables,
            [
                CallableHeaderRow {
                    ordinal: 0,
                    containing_type: 1,
                    kind: CallableKind::Method,
                    name: name("F", 24, 25),
                    return_type: Some(name("void", 19, 23)),
                    modifiers: MemberModifiers::default(),
                    type_params: vec![],
                    parameters: vec![parameter(
                        name("int", 26, 29),
                        CallableParameterType::Int,
                        name("x", 30, 31),
                    )],
                    header_span: Span::new(19, 32),
                    declaration_span: Span::new(19, 34),
                },
                CallableHeaderRow {
                    ordinal: 1,
                    containing_type: 2,
                    kind: CallableKind::Method,
                    name: name("F", 50, 51),
                    return_type: Some(name("void", 45, 49)),
                    modifiers: MemberModifiers::default(),
                    type_params: vec![name("T", 52, 53)],
                    parameters: vec![parameter(
                        name("T", 55, 56),
                        CallableParameterType::MethodSlot(0),
                        name("x", 57, 58),
                    )],
                    header_span: Span::new(45, 59),
                    declaration_span: Span::new(45, 61),
                },
            ],
        );
    }

    #[test]
    fn legacy_callable_index_still_admits_simple_methods_in_generic_owners() {
        let source = "class C<T>{void F(int x){}}";

        let index = index_callable_headers(source, Span::new(0, 27)).unwrap();

        assert_eq!(
            index.callables,
            [CallableHeaderRow {
                ordinal: 0,
                containing_type: 0,
                kind: CallableKind::Method,
                name: name("F", 16, 17),
                return_type: Some(name("void", 11, 15)),
                modifiers: MemberModifiers::default(),
                type_params: vec![],
                parameters: vec![parameter(
                    name("int", 18, 21),
                    CallableParameterType::Int,
                    name("x", 22, 23),
                )],
                header_span: Span::new(11, 24),
                declaration_span: Span::new(11, 26),
            }],
        );
    }

    #[test]
    fn generic_callable_index_refuses_a_later_field_without_recovery() {
        assert_generic_callable_refusal(
            "class C{void F<T>(T x){} int z;}",
            Some(Span::new(30, 31)),
            vec![],
        );
    }

    #[test]
    fn generic_callable_index_reports_own_refusal_and_recovery_at_a_stray_brace() {
        assert_generic_callable_refusal(
            "class C{void F<T>(T x){}} }",
            Some(Span::new(26, 27)),
            vec![Span::new(26, 26), Span::new(26, 27)],
        );
    }

    #[test]
    fn generic_callable_index_reports_malformed_binders_and_all_fallback_diagnostics() {
        for (source, first, recovery) in [
            (
                "class C{void F<>(){}}",
                Span::new(15, 16),
                vec![Span::new(14, 19), Span::new(20, 20), Span::new(20, 21)],
            ),
            (
                "class C{void F<T,>(){}}",
                Span::new(17, 18),
                vec![
                    Span::new(14, 15),
                    Span::new(15, 16),
                    Span::new(16, 21),
                    Span::new(22, 22),
                    Span::new(22, 23),
                ],
            ),
            (
                "class C{void F<T,T>(){}}",
                Span::new(17, 18),
                vec![
                    Span::new(14, 15),
                    Span::new(15, 16),
                    Span::new(16, 17),
                    Span::new(17, 18),
                    Span::new(18, 22),
                    Span::new(23, 23),
                    Span::new(23, 24),
                ],
            ),
            (
                "class C{void F<T",
                Span::new(16, 16),
                vec![Span::new(14, 15), Span::new(15, 16)],
            ),
        ] {
            assert_generic_callable_refusal(source, Some(first), recovery);
        }
    }

    #[test]
    fn generic_callable_index_falls_back_when_the_body_is_not_empty() {
        assert_generic_callable_refusal(
            "class C{void F<T>(){;}}",
            Some(Span::new(20, 21)),
            vec![
                Span::new(14, 15),
                Span::new(15, 16),
                Span::new(16, 21),
                Span::new(22, 22),
                Span::new(22, 23),
            ],
        );
    }

    #[test]
    fn generic_callable_index_does_not_admit_a_binder_use_from_a_previous_method() {
        assert_generic_callable_refusal(
            "class C{void F<T>(T x){} void G<U>(T y){}}",
            Some(Span::new(35, 36)),
            vec![
                Span::new(31, 32),
                Span::new(32, 33),
                Span::new(33, 35),
                Span::new(38, 40),
                Span::new(41, 41),
                Span::new(41, 42),
            ],
        );
    }

    #[test]
    fn generic_callable_index_rejects_a_region_that_splits_a_utf8_character() {
        let source = "π|class C{void F<T>(T x){}}|";
        let region = Span::new(1, 28);

        let index = index_generic_callable_headers(source, region);

        assert_eq!(index, Err(GenericCallableIndexError::InvalidRegion(region)));
    }

    #[test]
    fn generic_callable_index_refuses_a_bom_at_the_selected_region_start() {
        let source = "π|\u{feff}class C {}";
        let region = Span::new(3, 16);

        let index = index_generic_callable_headers(source, region);

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(Span::new(3, 6)),
                recovery: vec![],
            }),
        );
    }

    #[test]
    fn generic_callable_index_returns_only_recovery_at_the_shared_member_depth_limit() {
        let source = format!(
            "class C{{void F<T>(T x){{}}}} {}void G<T>(T x){{}}{}",
            "class N { ".repeat(199),
            "}".repeat(199),
        );

        assert_generic_callable_refusal(&source, None, vec![Span::new(2016, 2032)]);
    }

    fn generic_method_recovery_spans() -> Vec<Span> {
        vec![
            Span::new(14, 15),
            Span::new(15, 16),
            Span::new(16, 18),
            Span::new(21, 23),
            Span::new(40, 40),
            Span::new(40, 41),
        ]
    }

    #[test]
    fn legacy_parse_projects_generic_method_prefix_as_decl_and_later_method_at_root() {
        let source = GENERIC_METHOD_SOURCE;

        let unit = cs_parse(source, 0);

        let class = unit
            .root
            .children
            .iter()
            .find(|node| {
                matches!(&node.kind, CsNodeKind::Type { kind: CsTypeKind::Class, name, .. } if name.text == "C")
            })
            .expect("class C should be a root child");
        let decl = class
            .children
            .iter()
            .find(|node| node.span.start == 8)
            .expect("the void F prefix should be a child of C");
        assert_eq!(
            (&decl.kind, decl.span),
            (&CsNodeKind::Decl, Span::new(8, 14))
        );
        assert_eq!(
            decl.children
                .iter()
                .find(|node| matches!(node.kind, CsNodeKind::NameDecl(_))),
            Some(&CsNode::leaf(
                CsNodeKind::NameDecl(name("F", 13, 14)),
                Span::new(13, 14),
            )),
        );

        let methods = unit
            .root
            .children
            .iter()
            .map(|node| ("unit", node))
            .chain(class.children.iter().map(|node| ("C", node)))
            .filter(|(_, node)| matches!(node.kind, CsNodeKind::Method(_)))
            .map(|(parent, node)| (parent, &node.kind, node.span))
            .collect::<Vec<_>>();
        assert_eq!(
            methods,
            [(
                "unit",
                &CsNodeKind::Method(name("F", 30, 31)),
                Span::new(25, 40),
            )],
        );
    }

    #[test]
    fn legacy_parse_reports_ordered_diagnostics_for_generic_method_before_simple_method() {
        let expected = generic_method_recovery_spans();

        let unit = cs_parse(GENERIC_METHOD_SOURCE, 0);

        assert_eq!(unit.errors, expected);
    }

    #[test]
    fn type_only_index_returns_ordered_recovery_for_generic_method_before_simple_method() {
        let region = Span::new(0, 41);
        let expected = TypeIndexError::Recovered(generic_method_recovery_spans());

        let index = index_type_headers(GENERIC_METHOD_SOURCE, region);

        assert_eq!(index, Err(expected));
    }

    #[test]
    fn callable_index_returns_ordered_recovery_for_generic_method_before_simple_method() {
        let region = Span::new(0, 41);
        let expected = TypeIndexError::Recovered(generic_method_recovery_spans());

        let index = index_callable_headers(GENERIC_METHOD_SOURCE, region);

        assert_eq!(index, Err(expected));
    }

    const QUALIFIED_PARAMETER_SOURCE: &str = "class C{void F(global::A.X p){}}";

    fn qualified_parameter(
        root: CsName,
        segments: Vec<CsName>,
        span: Span,
        ordinal: usize,
        name: CsName,
    ) -> CallableParameter {
        CallableParameter {
            type_use: CallableParameterTypeUse::Qualified(QualifiedTypeUse {
                root,
                segments,
                span,
            }),
            comparison_type: CallableParameterType::LocalType(ordinal),
            name,
        }
    }

    const QUALIFIED_COLLISION_SOURCE: &str = "π|class C{void F(global::A.X p){} void F(global::B.X q){}} namespace A{class X{}} namespace B{struct X{}}|";
    const QUALIFIED_COLLISION_REGION: Span = Span { start: 3, end: 106 };

    #[test]
    fn qualified_index_distinguishes_same_named_targets_in_different_namespaces() {
        let source = QUALIFIED_COLLISION_SOURCE;
        let region = QUALIFIED_COLLISION_REGION;
        let first_parameter = [qualified_parameter(
            name("global", 18, 24),
            vec![name("A", 26, 27), name("X", 28, 29)],
            Span::new(18, 29),
            1,
            name("p", 30, 31),
        )];
        let second_parameter = [qualified_parameter(
            name("global", 42, 48),
            vec![name("B", 50, 51), name("X", 52, 53)],
            Span::new(42, 53),
            2,
            name("q", 54, 55),
        )];

        let index = index_qualified_callable_headers(source, region).unwrap();

        let [first, second] = index.callables.as_slice() else {
            panic!("expected two callable rows, got {:?}", index.callables);
        };
        assert_eq!(
            [
                (
                    first.ordinal,
                    first.containing_type,
                    first.parameters.as_slice()
                ),
                (
                    second.ordinal,
                    second.containing_type,
                    second.parameters.as_slice()
                ),
            ],
            [
                (0, 0, first_parameter.as_slice()),
                (1, 0, second_parameter.as_slice()),
            ]
        );
        assert_eq!(
            (
                index.types.region,
                &index.types.rows[1].name,
                &index.types.rows[2].name,
            ),
            (region, &name("X", 78, 79), &name("X", 102, 103))
        );
        assert_ne!(first.key(), second.key());
    }

    #[test]
    fn checked_qualified_parameter_evidence_keeps_exact_target_rows_and_spans() {
        let index = index_checked(QUALIFIED_COLLISION_SOURCE, QUALIFIED_COLLISION_REGION).unwrap();

        for (use_span, expected_parameter, target_span, target_id) in [
            (
                Span::new(18, 29),
                qualified_parameter(
                    name("global", 18, 24),
                    vec![name("A", 26, 27), name("X", 28, 29)],
                    Span::new(18, 29),
                    1,
                    name("p", 30, 31),
                ),
                Span::new(78, 79),
                DeclarationId::Type(1),
            ),
            (
                Span::new(42, 53),
                qualified_parameter(
                    name("global", 42, 48),
                    vec![name("B", 50, 51), name("X", 52, 53)],
                    Span::new(42, 53),
                    2,
                    name("q", 54, 55),
                ),
                Span::new(102, 103),
                DeclarationId::Type(2),
            ),
        ] {
            let evidence = index.evidence_at(use_span, EvidenceRole::ParameterType);

            assert_eq!(
                evidence,
                Some(CheckedEvidence::ParameterType(&expected_parameter))
            );
            assert_eq!(
                index
                    .declaration_at(target_span, DeclarationRole::Type)
                    .unwrap()
                    .id,
                target_id
            );
        }
    }

    #[test]
    fn checked_qualified_parameter_evidence_requires_the_exact_span_for_each_role() {
        let index = index_checked(QUALIFIED_COLLISION_SOURCE, QUALIFIED_COLLISION_REGION).unwrap();

        for (role, span, expected) in [
            (
                EvidenceRole::QualifiedParameterRoot,
                Span::new(18, 24),
                Some(name("global", 18, 24)),
            ),
            (
                EvidenceRole::QualifiedParameterSegment,
                Span::new(28, 29),
                Some(name("X", 28, 29)),
            ),
            (
                EvidenceRole::QualifiedParameterRoot,
                Span::new(18, 23),
                None,
            ),
            (
                EvidenceRole::QualifiedParameterSegment,
                Span::new(18, 24),
                None,
            ),
            (EvidenceRole::ParameterType, Span::new(28, 29), None),
        ] {
            let evidence = index.evidence_at(span, role);

            assert_eq!(evidence, expected.as_ref().map(CheckedEvidence::Name));
        }
    }

    #[test]
    fn qualified_index_keeps_equal_keys_for_distinct_callable_rows_with_the_same_target() {
        let source =
            "class C{void F(global::A.X p){} void F(global::A.@X q){}} namespace A{class X{}}";
        let region = Span::new(0, source.len());

        let index = index_qualified_callable_headers(source, region).unwrap();

        let [first, second] = index.callables.as_slice() else {
            panic!("expected two callable rows, got {:?}", index.callables);
        };
        assert_eq!((first.ordinal, second.ordinal), (0, 1));
        assert_eq!(first.key(), second.key());
        assert_eq!(
            second.key().parameter_types,
            [CallableParameterType::LocalType(1)]
        );
    }

    #[test]
    fn qualified_index_resolves_a_nested_interface_and_preserves_raw_spans() {
        let source = "é class C{int F(global /* root */ :: @N /* gap */ . É . @Leaf @p,bool b){/* empty */}} namespace N{class É{interface Leaf{}}}";
        let region = Span::new(3, 128);

        let index = index_qualified_callable_headers(source, region).unwrap();

        assert_eq!(index.types.region, region);
        assert_eq!(
            index.callables,
            [CallableHeaderRow {
                ordinal: 0,
                containing_type: 0,
                kind: CallableKind::Method,
                name: name("F", 15, 16),
                return_type: Some(name("int", 11, 14)),
                modifiers: MemberModifiers::default(),
                type_params: vec![],
                parameters: vec![
                    qualified_parameter(
                        name("global", 17, 23),
                        vec![name("@N", 38, 40), name("É", 53, 55), name("@Leaf", 58, 63)],
                        Span::new(17, 63),
                        2,
                        name("@p", 64, 66),
                    ),
                    parameter(
                        name("bool", 67, 71),
                        CallableParameterType::Bool,
                        name("b", 72, 73),
                    ),
                ],
                header_span: Span::new(11, 74),
                declaration_span: Span::new(11, 87),
            }]
        );
    }

    #[test]
    fn qualified_index_resolves_a_struct_constructor_parameter_to_an_empty_enum() {
        let source = "struct S{S(global::T p){}} enum T{}";

        let index = index_qualified_callable_headers(source, Span::new(0, 35)).unwrap();

        assert_eq!(
            index.callables,
            [CallableHeaderRow {
                ordinal: 0,
                containing_type: 0,
                kind: CallableKind::Constructor,
                name: name("S", 9, 10),
                return_type: None,
                modifiers: MemberModifiers::default(),
                type_params: vec![],
                parameters: vec![qualified_parameter(
                    name("global", 11, 17),
                    vec![name("T", 19, 20)],
                    Span::new(11, 20),
                    1,
                    name("p", 21, 22),
                )],
                header_span: Span::new(9, 23),
                declaration_span: Span::new(9, 25),
            }]
        );
    }

    #[test]
    fn qualified_index_still_collects_primitive_only_callables() {
        let source = "class C{C(int p){}bool F(){}}";

        let index = index_qualified_callable_headers(source, Span::new(0, source.len())).unwrap();

        assert_eq!(
            index.callables,
            [
                CallableHeaderRow {
                    ordinal: 0,
                    containing_type: 0,
                    kind: CallableKind::Constructor,
                    name: name("C", 8, 9),
                    return_type: None,
                    modifiers: MemberModifiers::default(),
                    type_params: vec![],
                    parameters: vec![parameter(
                        name("int", 10, 13),
                        CallableParameterType::Int,
                        name("p", 14, 15),
                    )],
                    header_span: Span::new(8, 16),
                    declaration_span: Span::new(8, 18),
                },
                CallableHeaderRow {
                    ordinal: 1,
                    containing_type: 0,
                    kind: CallableKind::Method,
                    name: name("F", 23, 24),
                    return_type: Some(name("bool", 18, 22)),
                    modifiers: MemberModifiers::default(),
                    type_params: vec![],
                    parameters: vec![],
                    header_span: Span::new(18, 26),
                    declaration_span: Span::new(18, 28),
                },
            ]
        );
    }

    #[test]
    fn qualified_index_refuses_a_region_starting_inside_a_utf8_character() {
        let source = "é class C {}";
        let region = Span::new(1, source.len());

        let index = index_qualified_callable_headers(source, region);

        assert_eq!(index, Err(GenericCallableIndexError::InvalidRegion(region)));
    }

    fn assert_qualified_refusal(
        source: &str,
        region: Span,
        first_unsupported: Option<Span>,
        recovery: Vec<Span>,
    ) {
        let index = index_qualified_callable_headers(source, region);

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported,
                recovery,
            }),
            "{source:?}"
        );
    }

    #[test]
    fn qualified_index_refuses_the_whole_use_without_one_exact_target() {
        for suffix in [
            "",
            " namespace B{class X{}}",
            " namespace A{class X{}} namespace A{struct X{}}",
            " namespace A{class @x{}}",
        ] {
            let source = format!("{QUALIFIED_PARAMETER_SOURCE}{suffix}");
            assert_qualified_refusal(
                &source,
                Span::new(0, source.len()),
                Some(Span::new(15, 26)),
                vec![],
            );
        }
    }

    #[test]
    fn qualified_index_refuses_the_whole_use_when_the_exact_target_is_outside_the_region() {
        let source = "class C{void F(global::A.X p){}} namespace A{class X{}}";
        let region = Span::new(0, 32);

        assert_qualified_refusal(source, region, Some(Span::new(15, 26)), vec![]);
    }

    #[test]
    fn qualified_index_refuses_the_first_unresolved_use_in_callable_and_parameter_source_order() {
        let source =
            "class C{void P(){} void F(global::C ok,global::Z.X p,global::A.Y q){} void G(global::B.X r){}}";
        let region = Span::new(0, source.len());

        assert_qualified_refusal(source, region, Some(Span::new(39, 50)), vec![]);
    }

    #[test]
    fn qualified_index_reports_later_subset_syntax_before_an_earlier_unresolved_path() {
        let source = "class C{void F(global::A.X p){} int z;}";
        let region = Span::new(0, 39);

        assert_qualified_refusal(source, region, Some(Span::new(37, 38)), vec![]);
    }

    #[test]
    fn qualified_index_reports_later_subset_syntax_and_ordered_recovery_before_lookup() {
        let source = "class C{void F(global::A.X p){}} }";
        let region = Span::new(0, 34);

        assert_qualified_refusal(
            source,
            region,
            Some(Span::new(33, 34)),
            vec![Span::new(33, 33), Span::new(33, 34)],
        );
    }

    #[test]
    fn qualified_index_reports_only_recovery_for_a_later_member_at_the_depth_limit() {
        let source = format!(
            "{QUALIFIED_PARAMETER_SOURCE} {}void G(){{}}{}",
            "class N { ".repeat(199),
            "}".repeat(199),
        );
        let region = Span::new(0, source.len());

        assert_qualified_refusal(&source, region, None, vec![Span::new(2023, 2033)]);
    }

    #[test]
    fn qualified_index_refuses_a_bom_at_the_selected_region_start_before_lookup() {
        let source = "π|\u{feff}class C{void F(global::A.X p){}}";
        let region = Span::new(3, 38);

        assert_qualified_refusal(source, region, Some(Span::new(3, 6)), vec![]);
    }

    #[test]
    fn qualified_index_refuses_partial_types_at_the_consumed_modifier_in_each_scope() {
        for (source, span) in [
            ("public /* partial */ partial class C {}", Span::new(21, 28)),
            (
                "namespace N { class C { public /* partial */ partial struct S{} } }",
                Span::new(45, 52),
            ),
            (
                "namespace N; private partial interface I {}",
                Span::new(21, 28),
            ),
        ] {
            assert_qualified_refusal(source, Span::new(0, source.len()), Some(span), vec![]);
        }
    }

    #[test]
    fn qualified_index_does_not_treat_a_type_named_partial_as_having_the_modifier() {
        let source = "class partial {}";

        let index = index_qualified_callable_headers(source, Span::new(0, 16)).unwrap();

        assert_eq!(index.types.rows[0].name, name("partial", 6, 13));
    }

    #[test]
    fn existing_indexes_still_admit_a_partial_type() {
        let source = "public partial class C {}";
        let region = Span::new(0, 25);
        let expected = [global_row(
            0,
            CsTypeKind::Class,
            "C",
            Span::new(21, 22),
            Span::new(15, 22),
            region,
        )];

        let types = index_type_headers(source, region).unwrap();
        let simple = index_callable_headers(source, region).unwrap();
        let generic = index_generic_callable_headers(source, region).unwrap();

        assert_eq!(types.rows, expected);
        assert_eq!(simple.types.rows, expected);
        assert_eq!(generic.types.rows, expected);
    }

    #[test]
    fn qualified_index_refuses_generic_methods_without_changing_legacy_recovery_order() {
        let region = Span::new(0, 41);

        let index = index_qualified_callable_headers(GENERIC_METHOD_SOURCE, region);

        assert_eq!(
            index,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(Span::new(14, 15)),
                recovery: generic_method_recovery_spans(),
            })
        );
    }

    #[test]
    fn legacy_parse_keeps_the_qualified_parameter_of_a_method_under_its_class() {
        let source = QUALIFIED_PARAMETER_SOURCE;

        let unit = cs_parse(source, 0);

        let [class] = unit.root.children.as_slice() else {
            panic!(
                "expected only class C at the root, got {:?}",
                unit.root.children
            );
        };
        let [method] = class.children.as_slice() else {
            panic!("expected only method F under C, got {:?}", class.children);
        };
        assert_eq!(
            (&class.kind, &method.kind),
            (
                &CsNodeKind::Type {
                    kind: CsTypeKind::Class,
                    name: name("C", 6, 7),
                    bases: vec![],
                },
                &CsNodeKind::Method(name("F", 13, 14)),
            )
        );
        // Pin the parameter projection without snapshotting the return type or body.
        let [_, parameter_nodes @ .., _] = method.children.as_slice() else {
            panic!(
                "expected parameter nodes between the return type and body, got {:?}",
                method.children
            );
        };
        assert_eq!(
            parameter_nodes,
            [
                CsNode::leaf(
                    CsNodeKind::NameRef {
                        name: name("global", 15, 21),
                        after_dot: false,
                        receiver: None,
                    },
                    Span::new(15, 21),
                ),
                CsNode::leaf(
                    CsNodeKind::NameRef {
                        name: name("A", 23, 24),
                        after_dot: true,
                        receiver: Some("global".to_owned()),
                    },
                    Span::new(23, 24),
                ),
                CsNode::leaf(
                    CsNodeKind::NameRef {
                        name: name("X", 25, 26),
                        after_dot: true,
                        receiver: Some("A".to_owned()),
                    },
                    Span::new(25, 26),
                ),
                CsNode::leaf(CsNodeKind::Param(name("p", 27, 28)), Span::new(27, 28)),
            ]
        );
        assert_eq!(unit.errors, []);
    }

    #[test]
    fn type_only_index_refuses_a_method_with_a_qualified_parameter_at_void() {
        let region = Span::new(0, 32);

        let index = index_type_headers(QUALIFIED_PARAMETER_SOURCE, region);

        assert_eq!(
            index,
            Err(TypeIndexError::UnsupportedSyntax(Span::new(8, 12)))
        );
    }

    #[test]
    fn callable_index_refuses_a_qualified_parameter_at_global() {
        assert_callable_refusal(
            QUALIFIED_PARAMETER_SOURCE,
            TypeIndexError::UnsupportedSyntax(Span::new(15, 21)),
        );
    }

    #[test]
    fn generic_callable_index_refuses_a_qualified_parameter_at_global_without_recovery() {
        assert_generic_callable_refusal(
            QUALIFIED_PARAMETER_SOURCE,
            Some(Span::new(15, 21)),
            vec![],
        );
    }

    fn block(
        callable: usize,
        ordinal: usize,
        parent: Option<usize>,
        span: Span,
    ) -> BlockOccurrence {
        BlockOccurrence {
            callable,
            ordinal,
            parent,
            span,
        }
    }

    fn field(containing_type: usize, type_use: CsName, name: CsName) -> FieldOccurrence {
        FieldOccurrence {
            containing_type,
            type_use,
            name,
            modifiers: MemberModifiers::default(),
        }
    }

    fn local(callable: usize, block: usize, type_use: CsName, name: CsName) -> LocalOccurrence {
        LocalOccurrence {
            callable,
            block,
            type_use,
            name,
        }
    }

    #[test]
    fn lexical_index_keeps_a_field_parameter_and_same_named_sibling_locals_separate() {
        let source = "class C{int x;void F(int p){{int x;} {int x;}}}";
        let expected = LexicalDeclarations {
            fields: vec![field(0, name("int", 8, 11), name("x", 12, 13))],
            blocks: vec![
                block(0, 0, None, Span::new(27, 46)),
                block(0, 1, Some(0), Span::new(28, 36)),
                block(0, 2, Some(0), Span::new(37, 45)),
            ],
            locals: vec![
                local(0, 1, name("int", 29, 32), name("x", 33, 34)),
                local(0, 2, name("int", 38, 41), name("x", 42, 43)),
            ],
        };

        let index = index_lexical_declarations(source, Span::new(0, 47)).unwrap();

        assert_eq!(index.declarations, expected);
        assert_eq!(
            index.headers.parameter_occurrences().collect::<Vec<_>>(),
            [(
                0,
                0,
                &parameter(
                    name("int", 21, 24),
                    CallableParameterType::Int,
                    name("p", 25, 26),
                ),
            )],
        );
    }

    #[test]
    fn lexical_index_restores_the_immediate_type_owner_after_a_nested_struct() {
        let source = "class C{int x;struct S{bool x;}int y;}";

        let index = index_lexical_declarations(source, Span::new(0, 38)).unwrap();

        assert_eq!(
            index.declarations.fields,
            [
                field(0, name("int", 8, 11), name("x", 12, 13)),
                field(1, name("bool", 23, 27), name("x", 28, 29)),
                field(0, name("int", 31, 34), name("y", 35, 36)),
            ]
        );
    }

    #[test]
    fn lexical_index_restores_exact_parents_and_resets_block_ordinals_for_equal_keys() {
        let source = "class C{void F(int p){{{int a;}bool b;}int c;}int F(int q){bool d;}}";
        let expected = LexicalDeclarations {
            fields: vec![],
            blocks: vec![
                block(0, 0, None, Span::new(21, 46)),
                block(0, 1, Some(0), Span::new(22, 39)),
                block(0, 2, Some(1), Span::new(23, 31)),
                block(1, 0, None, Span::new(58, 67)),
            ],
            locals: vec![
                local(0, 2, name("int", 24, 27), name("a", 28, 29)),
                local(0, 1, name("bool", 31, 35), name("b", 36, 37)),
                local(0, 0, name("int", 39, 42), name("c", 43, 44)),
                local(1, 0, name("bool", 59, 63), name("d", 64, 65)),
            ],
        };

        let index = index_lexical_declarations(source, Span::new(0, 68)).unwrap();

        assert_eq!(
            index.headers.callables[0].key(),
            index.headers.callables[1].key()
        );
        assert_eq!(index.declarations, expected);
        assert_eq!(
            index
                .headers
                .parameter_occurrences()
                .map(|(callable, position, parameter)| (callable, position, parameter.name.span))
                .collect::<Vec<_>>(),
            [(0, 0, Span::new(19, 20)), (1, 0, Span::new(56, 57))]
        );
    }

    #[test]
    fn lexical_index_preserves_absolute_utf8_constructor_and_qualified_parameter_evidence() {
        let source = "π|struct É{int λ;É(global::É @p,int q){bool α;}}|tail";
        let region = Span::new(3, 54);
        let expected = LexicalDeclarations {
            fields: vec![field(0, name("int", 13, 16), name("λ", 17, 19))],
            blocks: vec![block(0, 0, None, Span::new(43, 53))],
            locals: vec![local(0, 0, name("bool", 44, 48), name("α", 49, 51))],
        };

        let index = index_lexical_declarations(source, region).unwrap();

        assert_eq!(
            index,
            LexicalDeclarationIndex {
                headers: CallableHeaderIndex {
                    types: TypeHeaderIndex {
                        region,
                        namespaces: vec![],
                        rows: vec![global_row(
                            0,
                            CsTypeKind::Struct,
                            "É",
                            Span::new(10, 12),
                            Span::new(3, 12),
                            Span::new(3, 54),
                        )],
                    },
                    callables: vec![CallableHeaderRow {
                        ordinal: 0,
                        containing_type: 0,
                        kind: CallableKind::Constructor,
                        name: name("É", 20, 22),
                        return_type: None,
                        modifiers: MemberModifiers::default(),
                        type_params: vec![],
                        parameters: vec![
                            qualified_parameter(
                                name("global", 23, 29),
                                vec![name("É", 31, 33)],
                                Span::new(23, 33),
                                0,
                                name("@p", 34, 36),
                            ),
                            parameter(
                                name("int", 37, 40),
                                CallableParameterType::Int,
                                name("q", 41, 42),
                            ),
                        ],
                        header_span: Span::new(20, 43),
                        declaration_span: Span::new(20, 53),
                    }],
                },
                declarations: expected,
            }
        );
    }

    #[test]
    fn parameter_occurrences_borrow_generic_and_primitive_parameters_in_source_order() {
        let source = "class C{void F<T>(T @x,bool y){} void G(int z){}}";

        let index = index_generic_callable_headers(source, Span::new(0, 49)).unwrap();

        assert_eq!(
            index.parameter_occurrences().collect::<Vec<_>>(),
            [
                (
                    0,
                    0,
                    &parameter(
                        name("T", 18, 19),
                        CallableParameterType::MethodSlot(0),
                        name("@x", 20, 22),
                    )
                ),
                (
                    0,
                    1,
                    &parameter(
                        name("bool", 23, 27),
                        CallableParameterType::Bool,
                        name("y", 28, 29),
                    )
                ),
                (
                    1,
                    0,
                    &parameter(
                        name("int", 40, 43),
                        CallableParameterType::Int,
                        name("z", 44, 45),
                    )
                ),
            ],
        );
    }

    #[test]
    fn lexical_index_rejects_an_invalid_region_before_inspecting_syntax() {
        let source = "π|partial class C{}";
        let region = Span::new(1, 20);

        let result = index_lexical_declarations(source, region);

        assert_eq!(
            result,
            Err(GenericCallableIndexError::InvalidRegion(region))
        );
    }

    fn assert_lexical_refusal(source: &str, span: Span, recovery: Vec<Span>) {
        let result = index_lexical_declarations(source, Span::new(0, source.len()));

        assert_eq!(
            result,
            Err(GenericCallableIndexError::Rejected {
                first_unsupported: Some(span),
                recovery,
            }),
            "{source:?}"
        );
    }

    #[test]
    fn lexical_index_refuses_members_outside_the_field_and_callable_subset() {
        for (source, span) in [
            ("interface I{int x;}", Span::new(12, 15)),
            ("class C{[A]int x;}", Span::new(8, 9)),
            ("class C{static int x;}", Span::new(8, 14)),
            ("class C{void x;}", Span::new(14, 15)),
            ("class C{int x=1;}", Span::new(13, 14)),
            ("class C{int x,y;}", Span::new(13, 14)),
            ("class C{int x}", Span::new(13, 14)),
            ("class C{D(){}}", Span::new(8, 9)),
            ("partial class C{int x;}", Span::new(0, 7)),
        ] {
            assert_lexical_refusal(source, span, vec![]);
        }
    }

    #[test]
    fn lexical_index_refuses_body_syntax_other_than_primitive_locals_and_brace_blocks() {
        for (source, span) in [
            ("class C{void F(){if(p){int x;}}}", Span::new(17, 19)),
            ("class C{void F(){;}}", Span::new(17, 18)),
            ("class C{void F(){Use(x);}}", Span::new(17, 20)),
            ("class C{void F(){x=>{int y;};}}", Span::new(17, 18)),
            ("class C{void F(){int x=1;}}", Span::new(22, 23)),
            ("class C{void F(){int x,y;}}", Span::new(22, 23)),
            ("class C{void F(){int x}}", Span::new(22, 23)),
            ("class C{void F(){int x);}}", Span::new(22, 23)),
            ("class C{void F(){int[] x;}}", Span::new(20, 21)),
            ("class C{void F(){const int x;}}", Span::new(17, 22)),
            ("class C{void F(){[A]int x;}}", Span::new(17, 18)),
        ] {
            assert_lexical_refusal(source, span, vec![]);
        }
    }

    #[test]
    fn lexical_index_requires_actual_callable_delimiters() {
        for (source, span) in [
            ("class C{void F(int p{int x;}}", Span::new(20, 21)),
            ("class C{void F() int x;}", Span::new(17, 20)),
            ("class C{void F(){int x;", Span::new(23, 23)),
        ] {
            assert_lexical_refusal(source, span, vec![]);
        }
    }

    #[test]
    fn lexical_index_refuses_generic_methods_and_keeps_the_legacy_recovery_order() {
        assert_lexical_refusal(
            GENERIC_METHOD_SOURCE,
            Span::new(14, 15),
            generic_method_recovery_spans(),
        );
    }

    #[test]
    fn lexical_index_reports_later_body_refusal_and_ordered_recovery_before_lookup() {
        assert_lexical_refusal(
            "class C{void F(global::A.X p){} void G(){;}} }",
            Span::new(41, 42),
            vec![Span::new(45, 45), Span::new(45, 46)],
        );
    }

    #[test]
    fn lexical_index_refuses_the_full_unresolved_type_use_after_a_clean_parse() {
        assert_lexical_refusal(
            "class C{void F(global:: /* gap */ N.X p){int x;}}",
            Span::new(15, 37),
            vec![],
        );
    }

    #[test]
    fn qualified_header_index_still_refuses_a_nonempty_callable_body() {
        assert_qualified_refusal(
            "class C{void F(global::C p){int x;}}",
            Span::new(0, 36),
            Some(Span::new(28, 31)),
            vec![],
        );
    }

    #[test]
    fn legacy_parse_still_keeps_a_method_with_a_missing_parameter_name() {
        let source = "class C{void F(int){}}";

        let unit = cs_parse(source, 0);

        assert_eq!(
            unit,
            CsUnit {
                root: CsNode::new(
                    CsNodeKind::Unit,
                    Span::new(0, 22),
                    vec![CsNode::new(
                        CsNodeKind::Type {
                            kind: CsTypeKind::Class,
                            name: name("C", 6, 7),
                            bases: Vec::new(),
                        },
                        Span::new(0, 22),
                        vec![CsNode::new(
                            CsNodeKind::Method(name("F", 13, 14)),
                            Span::new(8, 21),
                            vec![
                                CsNode::leaf(
                                    CsNodeKind::NameRef {
                                        name: name("void", 8, 12),
                                        after_dot: false,
                                        receiver: None,
                                    },
                                    Span::new(8, 12),
                                ),
                                CsNode::leaf(
                                    CsNodeKind::NameRef {
                                        name: name("int", 15, 18),
                                        after_dot: false,
                                        receiver: None,
                                    },
                                    Span::new(15, 18),
                                ),
                                CsNode::leaf(CsNodeKind::Block, Span::new(19, 21)),
                            ],
                        )],
                    )],
                ),
                errors: Vec::new(),
            }
        );
    }

    #[test]
    fn legacy_parse_retains_namespace_and_type_when_qualified_name_has_a_dangling_dot() {
        let source = "namespace Q. { class C {} }";

        let unit = cs_parse(source, 0);

        assert_eq!(
            unit,
            CsUnit {
                root: CsNode::new(
                    CsNodeKind::Unit,
                    Span::new(0, 27),
                    vec![CsNode::new(
                        CsNodeKind::Namespace(name("Q", 10, 11)),
                        Span::new(0, 27),
                        vec![CsNode::leaf(
                            CsNodeKind::Type {
                                kind: CsTypeKind::Class,
                                name: name("C", 21, 22),
                                bases: Vec::new(),
                            },
                            Span::new(15, 25),
                        )],
                    )],
                ),
                errors: Vec::new(),
            }
        );
    }

    #[test]
    fn legacy_parse_retains_bases_and_accepts_a_missing_close_brace() {
        let source = "class C<T> : N.B {";

        let unit = cs_parse(source, 0);

        assert_eq!(
            unit,
            CsUnit {
                root: CsNode::new(
                    CsNodeKind::Unit,
                    Span::new(0, 18),
                    vec![CsNode::leaf(
                        CsNodeKind::Type {
                            kind: CsTypeKind::Class,
                            name: CsName {
                                text: "C".to_owned(),
                                span: Span::new(6, 7),
                            },
                            bases: vec![CsName {
                                text: "B".to_owned(),
                                span: Span::new(15, 16),
                            }],
                        },
                        Span::new(0, 18),
                    )],
                ),
                errors: Vec::new(),
            }
        );
    }
}
