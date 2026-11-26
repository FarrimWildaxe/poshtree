//! Binding within one successfully indexed region, without type inference.
//!
//! The older observer binds only bare and immediate `this` places. Checked
//! mode can additionally prove a receiver from a bound parameter's exact
//! local type or an explicit `global::` type path, and use checked base edges.

use super::bases::{BaseGraph, BaseLookupError};
use super::{
    comparison_identifier, BlockOccurrence, CallableHeaderRow, CallableKind, CallableParameterType,
    FieldOccurrence, LexicalDeclarations, LocalOccurrence, MemberModifiers, QualifiedTypeBinding,
    QualifiedTypeCandidate, QualifiedTypeFailure, QualifiedTypeLookup, QualifiedTypeUse,
    TypeHeaderIndex,
};
#[cfg(test)]
use super::{GenericCallableIndexError, LexicalDeclarationIndex};
use crate::v2::csharp::ast::CsName;
#[cfg(test)]
use crate::v2::csharp::parser::parse_binding_headers;
use crate::v2::span::Span;
use std::collections::HashMap;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BindingSubset {
    Lexical,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(super) struct BindingReport {
    pub(super) lexical: LexicalDeclarationIndex,
    pub(super) references: Vec<ReferenceBinding>,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ReferenceBinding {
    pub(super) site: ReferenceSite,
    pub(super) outcome: BindingOutcome,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::v2::csharp) struct CheckedReferenceBinding {
    pub(in crate::v2::csharp) site: ReferenceSite,
    pub(super) outcome: BindingOutcome,
    /// Present for a proven rooted receiver even if its member does not bind.
    pub(super) qualified_type: Option<QualifiedTypeBinding>,
    pub(super) argument: Option<CallArgumentEvidence>,
}

/// The independent read and its recorded exact type, if proved.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CallArgumentEvidence {
    pub(super) span: Span,
    pub(super) declaration: Option<DeclarationId>,
    pub(super) exact_type: Option<CallableParameterType>,
}

impl CheckedReferenceBinding {
    pub(in crate::v2::csharp) fn rename_binding(&self) -> RenameBinding<'_> {
        match &self.outcome {
            BindingOutcome::Bound(id) => RenameBinding::Bound(*id),
            BindingOutcome::Ambiguous(reason, ids) => RenameBinding::Ambiguous(*reason, ids),
            BindingOutcome::Unresolved(_) => RenameBinding::Unresolved,
        }
    }

    pub(in crate::v2::csharp) fn unproven_qualifier(&self) -> Option<&QualifiedTypeFailure> {
        match &self.outcome {
            BindingOutcome::Unresolved(Unresolved::UnprovenType(failure)) => Some(failure),
            _ => None,
        }
    }

    /// Uses the same applicability proof as binding, not overload selection.
    pub(in crate::v2::csharp) fn has_exact_applicability<'a>(
        &self,
        mut methods: impl Iterator<Item = &'a CallableHeaderRow>,
    ) -> bool {
        self.site.role == ReferenceRole::Call
            && exact_argument_type(self.argument.as_ref())
                .is_ok_and(|ty| methods.any(|method| matches_exact_argument(method, ty)))
    }
}

pub(in crate::v2::csharp) enum RenameBinding<'a> {
    Bound(DeclarationId),
    Ambiguous(Ambiguity, &'a [DeclarationId]),
    Unresolved,
}

fn exact_argument_type(
    argument: Option<&CallArgumentEvidence>,
) -> Result<Option<CallableParameterType>, Unresolved> {
    argument
        .map(|argument| {
            let span = argument.span;
            if argument.declaration.is_none() {
                return Err(Unresolved::UnboundArgument { span });
            }
            argument
                .exact_type
                .ok_or(Unresolved::UnknownArgumentType { span })
        })
        .transpose()
}

fn matches_exact_argument(method: &CallableHeaderRow, ty: Option<CallableParameterType>) -> bool {
    method.parameter_types().eq(ty)
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::v2::csharp) struct ReferenceSite {
    /// Source-order ordinal in this report, independent of declaration IDs.
    pub(in crate::v2::csharp) ordinal: usize,
    pub(in crate::v2::csharp) name: CsName,
    pub(in crate::v2::csharp) owner: SiteOwner,
    pub(in crate::v2::csharp) role: ReferenceRole,
    pub(in crate::v2::csharp) receiver: Receiver,
    /// For a one-argument call, the exact span of the immediately following read.
    pub(in crate::v2::csharp) call_argument: Option<Span>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::v2::csharp) struct SiteOwner {
    pub(in crate::v2::csharp) callable: usize,
    pub(in crate::v2::csharp) block: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::v2::csharp) enum ReferenceRole {
    Write,
    Read,
    ReceiverName,
    Call,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::v2::csharp) enum Receiver {
    Bare,
    This(CsName),
    Base(CsName),
    Name(CsName),
    Rooted(QualifiedTypeUse),
}

/// Parse-local occurrence IDs, not spans or name/signature keys. Positions
/// count declarations of that kind within their exact owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::v2::csharp) enum DeclarationId {
    Field(usize, usize),
    Parameter(usize, usize),
    Local(usize, usize, usize),
    // Wrong-kind candidates still need to identify the exact competing row.
    Callable(usize),
    Type(usize),
}

impl LexicalDeclarations {
    /// Field IDs are counted in source order within the exact containing type.
    pub(in crate::v2::csharp) fn field_occurrences(
        &self,
    ) -> impl Iterator<Item = (DeclarationId, &FieldOccurrence)> {
        let mut positions = HashMap::new();
        self.fields.iter().map(move |row| {
            let owner = row.containing_type;
            let position = positions.entry(owner).or_insert(0);
            let id = DeclarationId::Field(owner, *position);
            *position += 1;
            (id, row)
        })
    }

    /// Local IDs are counted in source order within the exact callable block.
    pub(super) fn local_occurrences(
        &self,
    ) -> impl Iterator<Item = (DeclarationId, &LocalOccurrence)> {
        let mut positions = HashMap::new();
        self.locals.iter().map(move |row| {
            let position = positions.entry((row.callable, row.block)).or_insert(0);
            let id = DeclarationId::Local(row.callable, row.block, *position);
            *position += 1;
            (id, row)
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum BindingOutcome {
    Bound(DeclarationId),
    /// Distinct candidate occurrences in declaration source order.
    Ambiguous(Ambiguity, Vec<DeclarationId>),
    Unresolved(Unresolved),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::v2::csharp) enum Ambiguity {
    DeclarationSpaceConflict,
    DuplicateMember,
    CompetingMember,
    Overload,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Unresolved {
    BeforeLocalDeclaration,
    TypeParameterInValuePosition {
        declaration: Span,
    },
    UnknownReceiver,
    UnprovenType(QualifiedTypeFailure),
    ThisInStaticContext,
    BaseInStaticContext,
    Base(BaseLookupError),
    InstanceMemberRequired,
    StaticMemberRequired,
    Inaccessible,
    WrongKind,
    #[cfg(test)]
    GenericMethod,
    WrongArgumentCount,
    UnboundArgument {
        span: Span,
    },
    UnknownArgumentType {
        span: Span,
    },
    NoExactParameterMatch {
        span: Span,
    },
    NotFound,
}

/// Returns no evidence unless parsing and qualified header lookup both succeed.
#[cfg(test)]
pub(super) fn bind_references(
    source: &str,
    region: Span,
    subset: BindingSubset,
) -> Result<BindingReport, GenericCallableIndexError> {
    let code = source
        .get(region.start..region.end)
        .ok_or(GenericCallableIndexError::InvalidRegion(region))?;
    let mut declarations = Default::default();
    let mut sites = Vec::new();
    let pending = match subset {
        BindingSubset::Lexical => {
            parse_binding_headers(code, region, &mut declarations, &mut sites)
        }
    }?;
    let lexical = LexicalDeclarationIndex {
        headers: pending.resolve()?,
        declarations,
    };
    let lookup = BindingLookup::new(
        &lexical.headers.types,
        &lexical.headers.callables,
        &lexical.declarations,
    );
    let references = sites
        .into_iter()
        .map(|site| {
            let outcome = lookup.bind(&site);
            ReferenceBinding { site, outcome }
        })
        .collect();
    Ok(BindingReport {
        lexical,
        references,
    })
}

pub(super) fn bind_checked(
    types: &TypeHeaderIndex,
    headers: &[CallableHeaderRow],
    declarations: &LexicalDeclarations,
    qualified: &QualifiedTypeLookup<'_>,
    bases: &BaseGraph,
    sites: Vec<ReferenceSite>,
) -> Vec<CheckedReferenceBinding> {
    let lookup = BindingLookup::new(types, headers, declarations);
    let value_types = sites
        .iter()
        .any(|site| site.call_argument.is_some())
        .then(|| recorded_value_types(headers, declarations));
    let mut references: Vec<CheckedReferenceBinding> = Vec::with_capacity(sites.len());
    // A call's argument read follows it in source order. Bind that read once
    // before checking applicability and return the original source order.
    for site in sites.into_iter().rev() {
        let argument = site.call_argument.map(|span| {
            let declaration = references
                .last()
                .filter(|read| {
                    read.site.name.span == span
                        && read.site.owner == site.owner
                        && read.site.role == ReferenceRole::Read
                        && read.site.receiver == Receiver::Bare
                })
                .and_then(|read| match read.outcome {
                    BindingOutcome::Bound(id) => Some(id),
                    _ => None,
                });
            CallArgumentEvidence {
                span,
                declaration,
                exact_type: declaration.and_then(|id| value_types.as_ref()?.get(&id).copied()),
            }
        });
        let (outcome, qualified_type) =
            lookup.bind_checked(&site, qualified, bases, argument.as_ref());
        references.push(CheckedReferenceBinding {
            site,
            outcome,
            qualified_type,
            argument,
        });
    }
    references.reverse();
    references
}

fn recorded_value_types(
    headers: &[CallableHeaderRow],
    declarations: &LexicalDeclarations,
) -> HashMap<DeclarationId, CallableParameterType> {
    let fields = declarations
        .field_occurrences()
        .map(|(id, field)| (id, &field.type_use));
    let locals = declarations
        .local_occurrences()
        .map(|(id, local)| (id, &local.type_use));
    let primitives = fields
        .chain(locals)
        .filter_map(|(id, name)| CallableParameterType::primitive(&name.text).map(|ty| (id, ty)));
    let parameters = headers.iter().flat_map(|row| {
        row.parameters
            .iter()
            .enumerate()
            .map(|(position, parameter)| {
                (
                    DeclarationId::Parameter(row.ordinal, position),
                    parameter.comparison_type,
                )
            })
    });
    #[cfg(test)]
    let parameters =
        parameters.filter(|(_, ty)| !matches!(ty, CallableParameterType::MethodSlot(_)));
    primitives.chain(parameters).collect()
}

struct Candidate<'a> {
    id: DeclarationId,
    name: &'a CsName,
}

struct Member<'a> {
    owner: usize,
    declaration: Candidate<'a>,
    field_modifiers: Option<&'a MemberModifiers>,
}

impl Candidate<'_> {
    fn matches(&self, spelling: &str) -> bool {
        comparison_identifier(&self.name.text) == spelling
    }
}

struct Scope<'a> {
    occurrence: &'a BlockOccurrence,
    locals: Vec<Candidate<'a>>,
}

struct Callable<'a> {
    header: &'a CallableHeaderRow,
    parameters: Vec<Candidate<'a>>,
    blocks: Vec<Scope<'a>>,
}

struct BindingLookup<'a> {
    types: &'a TypeHeaderIndex,
    members: Vec<HashMap<&'a str, Vec<Member<'a>>>>,
    callables: Vec<Callable<'a>>,
}

#[derive(Clone, Copy)]
enum MemberAccess {
    #[cfg(test)]
    Legacy,
    Checked {
        from: usize,
        kind: AccessKind,
    },
}

#[derive(Clone, Copy)]
enum AccessKind {
    ImplicitInstance,
    Instance,
    Static,
}

impl<'a> BindingLookup<'a> {
    fn new(
        types: &'a TypeHeaderIndex,
        headers: &'a [CallableHeaderRow],
        declarations: &'a LexicalDeclarations,
    ) -> Self {
        let mut members = Vec::new();
        members.resize_with(types.rows.len(), HashMap::<_, Vec<_>>::new);
        let mut add_member = |owner: usize, name: &'a CsName, id, field_modifiers| {
            members[owner]
                .entry(comparison_identifier(&name.text))
                .or_default()
                .push(Member {
                    owner,
                    declaration: Candidate { id, name },
                    field_modifiers,
                });
        };
        for (id, field) in declarations.field_occurrences() {
            add_member(
                field.containing_type,
                &field.name,
                id,
                Some(&field.modifiers),
            );
        }
        for ty in &types.rows {
            if let Some(owner) = ty.containing_type {
                add_member(owner, &ty.name, DeclarationId::Type(ty.ordinal), None);
            }
        }
        let mut callables: Vec<_> = headers
            .iter()
            .map(|row| {
                add_member(
                    row.containing_type,
                    &row.name,
                    DeclarationId::Callable(row.ordinal),
                    None,
                );
                Callable {
                    header: row,
                    parameters: row
                        .parameters
                        .iter()
                        .enumerate()
                        .map(|(position, parameter)| Candidate {
                            id: DeclarationId::Parameter(row.ordinal, position),
                            name: &parameter.name,
                        })
                        .collect(),
                    blocks: Vec::new(),
                }
            })
            .collect();
        for occurrence in &declarations.blocks {
            callables[occurrence.callable].blocks.push(Scope {
                occurrence,
                locals: Vec::new(),
            });
        }
        for (id, local) in declarations.local_occurrences() {
            let locals = &mut callables[local.callable].blocks[local.block].locals;
            locals.push(Candidate {
                id,
                name: &local.name,
            });
        }
        Self {
            types,
            members,
            callables,
        }
    }

    #[cfg(test)]
    fn bind(&self, site: &ReferenceSite) -> BindingOutcome {
        let spelling = comparison_identifier(&site.name.text);
        let callable = &self.callables[site.owner.callable];
        let owner = callable.header.containing_type;
        match site.receiver {
            Receiver::Bare => Self::bind_lexical(callable, site.owner.block, &site.name)
                .unwrap_or_else(|| self.bind_direct_member(owner, spelling, MemberAccess::Legacy)),
            Receiver::This(_) => self.bind_direct_member(owner, spelling, MemberAccess::Legacy),
            Receiver::Base(_) | Receiver::Name(_) | Receiver::Rooted(_) => {
                BindingOutcome::Unresolved(Unresolved::UnknownReceiver)
            }
        }
    }

    fn bind_checked(
        &self,
        site: &ReferenceSite,
        qualified: &QualifiedTypeLookup<'_>,
        bases: &BaseGraph,
        argument: Option<&CallArgumentEvidence>,
    ) -> (BindingOutcome, Option<QualifiedTypeBinding>) {
        let spelling = comparison_identifier(&site.name.text);
        let callable = &self.callables[site.owner.callable];
        let from = callable.header.containing_type;
        let in_static = callable.header.modifiers.static_token.is_some();
        let access = |kind| MemberAccess::Checked { from, kind };
        let is_call = site.role == ReferenceRole::Call;
        let bind_members = |candidates: &[&Member<'_>], access| {
            if is_call {
                self.bind_call_candidates(candidates, access, argument)
                    .unwrap_or_else(BindingOutcome::Unresolved)
            } else {
                self.bind_member_candidates(candidates, access)
            }
        };
        let inherited = |owner, kind| {
            let access = access(kind);
            self.bind_inherited_member(bases, owner, spelling, access, |candidates| {
                bind_members(candidates, access)
            })
        };
        let member = |owner, kind| {
            let direct = self.direct_members(owner, spelling);
            if direct.is_empty() {
                inherited(owner, kind)
            } else {
                bind_members(&direct.iter().collect::<Vec<_>>(), access(kind))
            }
        };
        let outcome = match &site.receiver {
            Receiver::Bare => match Self::bind_lexical(callable, site.owner.block, &site.name) {
                Some(BindingOutcome::Bound(_)) if is_call => {
                    BindingOutcome::Unresolved(Unresolved::WrongKind)
                }
                Some(outcome) => outcome,
                None => {
                    // A type parameter is not a value, but its name still
                    // prevents an unqualified fallback to a member.
                    let type_parameter = self
                        .types
                        .type_lineage(Some(from))
                        .flat_map(|row| &row.type_params)
                        .find(|name| comparison_identifier(&name.text) == spelling);
                    if let Some(name) = type_parameter {
                        BindingOutcome::Unresolved(Unresolved::TypeParameterInValuePosition {
                            declaration: name.span,
                        })
                    } else {
                        member(
                            from,
                            if in_static {
                                AccessKind::Static
                            } else {
                                AccessKind::ImplicitInstance
                            },
                        )
                    }
                }
            },
            Receiver::This(_) if in_static => {
                BindingOutcome::Unresolved(Unresolved::ThisInStaticContext)
            }
            Receiver::This(_) => member(from, AccessKind::Instance),
            Receiver::Base(_) if in_static => {
                BindingOutcome::Unresolved(Unresolved::BaseInStaticContext)
            }
            Receiver::Base(_) => inherited(from, AccessKind::Instance),
            Receiver::Name(name) => match Self::bind_lexical(callable, site.owner.block, name) {
                Some(BindingOutcome::Bound(DeclarationId::Parameter(_, position))) => {
                    match callable.header.parameters[position].comparison_type {
                        CallableParameterType::LocalType(owner) => {
                            member(owner, AccessKind::Instance)
                        }
                        _ => BindingOutcome::Unresolved(Unresolved::UnknownReceiver),
                    }
                }
                // Fields, later locals, and declaration-space conflicts do not
                // establish a receiver type or permit any member fallback.
                _ => BindingOutcome::Unresolved(Unresolved::UnknownReceiver),
            },
            Receiver::Rooted(path) => match qualified
                .resolve_checked_path(path, |owner, spelling| {
                    self.competing_qualifier_members(owner, spelling)
                }) {
                Ok(bound) => return (member(bound.terminal, AccessKind::Static), Some(bound)),
                Err(failure) => BindingOutcome::Unresolved(Unresolved::UnprovenType(failure)),
            },
        };
        (outcome, None)
    }

    fn competing_qualifier_members(
        &self,
        owner: usize,
        spelling: &str,
    ) -> Vec<QualifiedTypeCandidate> {
        self.direct_members(owner, spelling)
            .iter()
            .map(|member| &member.declaration)
            .filter(|candidate| !matches!(candidate.id, DeclarationId::Type(_)))
            .map(|candidate| QualifiedTypeCandidate::Member {
                id: candidate.id,
                span: candidate.name.span,
            })
            .collect()
    }

    fn bind_lexical(
        callable: &Callable<'_>,
        block: usize,
        name: &CsName,
    ) -> Option<BindingOutcome> {
        let spelling = comparison_identifier(&name.text);
        let path = std::iter::successors(Some(&callable.blocks[block]), |block| {
            block
                .occurrence
                .parent
                .map(|ordinal| &callable.blocks[ordinal])
        });
        // Even a later declaration participates in conflicts. Only a unique
        // match can be checked for declaration order.
        let candidates: Vec<_> = path
            .flat_map(|block| &block.locals)
            .chain(&callable.parameters)
            .filter(|candidate| candidate.matches(spelling))
            .collect();
        match candidates.as_slice() {
            [] => None,
            [candidate] => Some(
                if matches!(candidate.id, DeclarationId::Local(..))
                    && candidate.name.span.start > name.span.start
                {
                    BindingOutcome::Unresolved(Unresolved::BeforeLocalDeclaration)
                } else {
                    BindingOutcome::Bound(candidate.id)
                },
            ),
            _ => Some(ambiguous(Ambiguity::DeclarationSpaceConflict, candidates)),
        }
    }

    fn direct_members(&self, owner: usize, spelling: &str) -> &[Member<'a>] {
        self.members[owner].get(spelling).map_or(&[], Vec::as_slice)
    }

    #[cfg(test)]
    fn bind_direct_member(
        &self,
        owner: usize,
        spelling: &str,
        access: MemberAccess,
    ) -> BindingOutcome {
        let candidates: Vec<_> = self.direct_members(owner, spelling).iter().collect();
        self.bind_member_candidates(&candidates, access)
    }

    fn bind_inherited_member(
        &self,
        bases: &BaseGraph,
        owner: usize,
        spelling: &str,
        access: MemberAccess,
        bind: impl FnOnce(&[&Member<'_>]) -> BindingOutcome,
    ) -> BindingOutcome {
        match bases.nearest_ancestors(owner, |owner| self.direct_members(owner, spelling)) {
            Ok(candidates) => {
                let outcome = bind(&candidates);
                // The selected member's declaring type may be visible even
                // when the receiver's type is not. Require proof of both.
                if matches!(
                    outcome,
                    BindingOutcome::Bound(_) | BindingOutcome::Ambiguous(Ambiguity::Overload, _)
                ) && matches!(
                    access,
                    MemberAccess::Checked { from, .. }
                        if self.types.inaccessible_type_link(from, owner).is_some()
                ) {
                    return BindingOutcome::Unresolved(Unresolved::Inaccessible);
                }
                outcome
            }
            Err(failure) => BindingOutcome::Unresolved(Unresolved::Base(failure)),
        }
    }

    fn bind_member_candidates(
        &self,
        candidates: &[&Member<'_>],
        access: MemberAccess,
    ) -> BindingOutcome {
        match candidates {
            [] => BindingOutcome::Unresolved(Unresolved::NotFound),
            [candidate] => {
                let Some(modifiers) = candidate.field_modifiers else {
                    return BindingOutcome::Unresolved(Unresolved::WrongKind);
                };
                if let Err(reason) = self.check_member_access(candidate.owner, modifiers, access) {
                    return BindingOutcome::Unresolved(reason);
                }
                BindingOutcome::Bound(candidate.declaration.id)
            }
            _ => {
                let reason = if candidates
                    .iter()
                    .all(|candidate| candidate.field_modifiers.is_some())
                {
                    Ambiguity::DuplicateMember
                } else {
                    Ambiguity::CompetingMember
                };
                ambiguous(
                    reason,
                    candidates
                        .iter()
                        .map(|candidate| &candidate.declaration)
                        .collect(),
                )
            }
        }
    }

    fn check_member_access(
        &self,
        owner: usize,
        modifiers: &MemberModifiers,
        access: MemberAccess,
    ) -> Result<(), Unresolved> {
        let (from, kind) = match access {
            #[cfg(test)]
            MemberAccess::Legacy => return Ok(()),
            MemberAccess::Checked { from, kind } => (from, kind),
        };
        if from != owner
            && (modifiers.public.is_none()
                || self.types.inaccessible_type_link(from, owner).is_some())
        {
            return Err(Unresolved::Inaccessible);
        }
        match (kind, modifiers.static_token.is_some()) {
            (AccessKind::Instance, true) => Err(Unresolved::InstanceMemberRequired),
            (AccessKind::Static, false) => Err(Unresolved::StaticMemberRequired),
            _ => Ok(()),
        }
    }

    fn bind_call_candidates(
        &self,
        candidates: &[&Member<'_>],
        access: MemberAccess,
        argument: Option<&CallArgumentEvidence>,
    ) -> Result<BindingOutcome, Unresolved> {
        if candidates.is_empty() {
            return Err(Unresolved::NotFound);
        }
        let count = usize::from(argument.is_some());
        let mut eligible = Vec::new();
        for candidate in candidates {
            let DeclarationId::Callable(ordinal) = candidate.declaration.id else {
                return Err(Unresolved::WrongKind);
            };
            let method = self.callables[ordinal].header;
            if method.kind != CallableKind::Method {
                return Err(Unresolved::WrongKind);
            }
            if method.parameters.len() != count {
                continue;
            }
            // A same-count competitor we cannot classify is not proof that
            // the remaining methods form a complete eligible group.
            #[cfg(test)]
            if !method.type_params.is_empty() {
                return Err(Unresolved::GenericMethod);
            }
            self.check_member_access(candidate.owner, &method.modifiers, access)?;
            eligible.push((&candidate.declaration, method));
        }
        if eligible.is_empty() {
            return Err(Unresolved::WrongArgumentCount);
        }
        let exact_type = exact_argument_type(argument)?;
        if let Some(argument) = argument {
            if !eligible
                .iter()
                .any(|(_, method)| matches_exact_argument(method, exact_type))
            {
                return Err(Unresolved::NoExactParameterMatch {
                    span: argument.span,
                });
            }
        }
        // An exact witness proves applicability only, never a best overload.
        if let [(candidate, _)] = eligible.as_slice() {
            Ok(BindingOutcome::Bound(candidate.id))
        } else {
            Ok(ambiguous(
                Ambiguity::Overload,
                eligible
                    .into_iter()
                    .map(|(candidate, _)| candidate)
                    .collect(),
            ))
        }
    }
}

fn ambiguous(reason: Ambiguity, mut candidates: Vec<&Candidate<'_>>) -> BindingOutcome {
    candidates.sort_unstable_by_key(|candidate| candidate.name.span.start);
    BindingOutcome::Ambiguous(
        reason,
        candidates
            .into_iter()
            .map(|candidate| candidate.id)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::super::{index_checked, index_lexical_declarations, CallableParameterType};
    use super::*;
    use crate::v2::csharp::parser::cs_parse;
    use crate::v2::csharp::{csharp_references, rename_csharp_local};
    use crate::v2::edit::TextEdit;
    use Ambiguity::{CompetingMember, DeclarationSpaceConflict, DuplicateMember};
    use BindingOutcome::{Ambiguous, Bound, Unresolved as Unbound};
    use DeclarationId::{Callable as Method, Field, Local, Parameter, Type};
    use ReferenceRole::{Read, ReceiverName, Write};
    use Unresolved::{BeforeLocalDeclaration, NotFound, UnknownReceiver, WrongKind};

    const LATER_LOCAL: &str = "class C{int x;void F(){x=x;int x;x=x;}}";
    const TRAILING_CLOSER: &str = "class C{void F(){x=y+z;}} }";

    fn depth_fixture() -> String {
        format!(
            "class C{{void F(){{{}x=y;{{x=y;}}{}}}}}",
            "{".repeat(196),
            "}".repeat(196)
        )
    }

    fn bind(source: &str) -> Result<BindingReport, GenericCallableIndexError> {
        bind_references(source, Span::new(0, source.len()), BindingSubset::Lexical)
    }

    fn name((text, start, end): (&str, usize, usize)) -> CsName {
        CsName {
            text: text.to_owned(),
            span: Span::new(start, end),
        }
    }

    fn binding(
        ordinal: usize,
        block: usize,
        token: (&str, usize, usize),
        role: ReferenceRole,
        receiver: Receiver,
        outcome: BindingOutcome,
    ) -> ReferenceBinding {
        ReferenceBinding {
            site: ReferenceSite {
                ordinal,
                name: name(token),
                owner: SiteOwner { callable: 0, block },
                role,
                receiver,
                call_argument: None,
            },
            outcome,
        }
    }

    fn at_starts(report: BindingReport) -> Vec<(usize, BindingOutcome)> {
        report
            .references
            .into_iter()
            .map(|reference| (reference.site.name.span.start, reference.outcome))
            .collect()
    }

    fn rejected(span: Span, recovery: Vec<Span>) -> GenericCallableIndexError {
        GenericCallableIndexError::Rejected {
            first_unsupported: Some(span),
            recovery,
        }
    }

    #[test]
    fn binding_merges_real_block_ownership_with_ordered_assignment_references() {
        use Receiver::{Bare, Name, This};
        let source = "class C{int x;void F(int p){{int x;x=p;this.x=x;}{int x;x=p;}x=p;u.x=p;}}";

        let report = bind(source).unwrap();

        assert_eq!(
            report.references,
            [
                binding(0, 1, ("x", 35, 36), Write, Bare, Bound(Local(0, 1, 0))),
                binding(1, 1, ("p", 37, 38), Read, Bare, Bound(Parameter(0, 0))),
                binding(
                    2,
                    1,
                    ("x", 44, 45),
                    Write,
                    This(name(("this", 39, 43))),
                    Bound(Field(0, 0)),
                ),
                binding(3, 1, ("x", 46, 47), Read, Bare, Bound(Local(0, 1, 0))),
                binding(4, 2, ("x", 56, 57), Write, Bare, Bound(Local(0, 2, 0))),
                binding(5, 2, ("p", 58, 59), Read, Bare, Bound(Parameter(0, 0))),
                binding(6, 0, ("x", 61, 62), Write, Bare, Bound(Field(0, 0))),
                binding(7, 0, ("p", 63, 64), Read, Bare, Bound(Parameter(0, 0))),
                binding(8, 0, ("u", 65, 66), ReceiverName, Bare, Unbound(NotFound)),
                binding(
                    9,
                    0,
                    ("x", 67, 68),
                    Write,
                    Name(name(("u", 65, 66))),
                    Unbound(UnknownReceiver),
                ),
                binding(10, 0, ("p", 69, 70), Read, Bare, Bound(Parameter(0, 0))),
            ]
        );
        let declarations = &report.lexical.declarations;
        assert_eq!(
            (
                (
                    declarations.fields[0].containing_type,
                    &declarations.fields[0].name
                ),
                &report.lexical.headers.callables[0].parameters[0].name,
                declarations
                    .locals
                    .iter()
                    .map(|row| (row.callable, row.block, &row.name))
                    .collect::<Vec<_>>(),
            ),
            (
                (0, &name(("x", 12, 13))),
                &name(("p", 25, 26)),
                vec![(0, 1, &name(("x", 33, 34))), (0, 2, &name(("x", 54, 55))),]
            )
        );
        assert_eq!(
            declarations
                .blocks
                .iter()
                .map(|row| (row.callable, row.ordinal, row.parent, row.span))
                .collect::<Vec<_>>(),
            [
                (0, 0, None, Span::new(27, 72)),
                (0, 1, Some(0), Span::new(28, 49)),
                (0, 2, Some(0), Span::new(49, 61)),
            ]
        );
    }

    #[test]
    fn binding_with_a_later_local_blocks_field_fallback_before_its_declaration() {
        let report = bind(LATER_LOCAL).unwrap();

        assert_eq!(
            report.lexical.declarations.locals[0].name,
            name(("x", 31, 32))
        );
        assert_eq!(
            at_starts(report),
            [
                (23, Unbound(BeforeLocalDeclaration)),
                (25, Unbound(BeforeLocalDeclaration)),
                (33, Bound(Local(0, 0, 0))),
                (35, Bound(Local(0, 0, 0))),
            ]
        );
    }

    #[test]
    fn public_references_still_include_earlier_sites_and_the_local_declaration() {
        let unit = cs_parse(LATER_LOCAL, 0);

        let references = csharp_references(&unit, Span::new(31, 32));

        assert_eq!(
            references.iter().map(|span| span.start).collect::<Vec<_>>(),
            [23, 25, 31, 33, 35]
        );
    }

    #[test]
    fn public_local_rename_still_edits_earlier_sites_and_the_declaration() {
        let unit = cs_parse(LATER_LOCAL, 0);

        let edits = rename_csharp_local(&unit, Span::new(31, 32), "changed");

        assert_eq!(
            edits,
            [
                TextEdit::replace(Span::new(23, 24), "changed"),
                TextEdit::replace(Span::new(25, 26), "changed"),
                TextEdit::replace(Span::new(31, 32), "changed"),
                TextEdit::replace(Span::new(33, 34), "changed"),
                TextEdit::replace(Span::new(35, 36), "changed"),
            ]
        );
    }

    #[test]
    fn old_lexical_index_still_refuses_an_assignment_at_the_first_name() {
        let source = "class C{void F(){x=y;}}";

        let index = index_lexical_declarations(source, Span::new(0, 23));

        assert_eq!(index, Err(rejected(Span::new(17, 18), vec![])));
    }

    #[test]
    fn older_binding_mode_still_refuses_a_rooted_place() {
        let result = bind("class C{void G(){global::C.x=x;}}");

        assert_eq!(result, Err(rejected(Span::new(17, 23), vec![])));
    }

    #[test]
    fn older_binding_mode_still_refuses_a_simple_call_statement() {
        let result = bind("class C{void F(){}void G(){F();}}");

        assert_eq!(result, Err(rejected(Span::new(28, 29), vec![])));
    }

    #[test]
    fn checked_binder_does_not_use_a_method_slot_as_an_exact_argument_type() {
        let source = "class C{void F(int a){}void G(int v){F(v);}}";
        let index = index_checked(source, Span::new(0, source.len())).unwrap();
        let LexicalDeclarationIndex {
            headers,
            declarations,
        } = index.lexical;
        let types = &headers.types;
        let mut callables = headers.callables;
        // The parser currently admits generic declarations only with empty
        // bodies; exercise the binder's unknown-type guard without widening it.
        callables[1].parameters[0].comparison_type = CallableParameterType::MethodSlot(0);
        let sites = index
            .references
            .into_iter()
            .map(|reference| reference.site)
            .collect();

        let references = bind_checked(
            types,
            &callables,
            &declarations,
            &QualifiedTypeLookup::new(types),
            &index.bases.graph,
            sites,
        );

        let span = Span::new(39, 40);
        assert_eq!(
            references
                .iter()
                .map(|row| (row.site.name.span, &row.outcome))
                .collect::<Vec<_>>(),
            [
                (
                    Span::new(37, 38),
                    &Unbound(Unresolved::UnknownArgumentType { span })
                ),
                (span, &Bound(Parameter(1, 0))),
            ]
        );
    }

    #[test]
    fn public_legacy_references_still_attach_a_bare_overloaded_call_to_the_first_method() {
        let source = "class C{void F(bool a){}void F(int b){}void G(int v){F(v);}}";
        let unit = cs_parse(source, 0);

        let spans = csharp_references(&unit, Span::new(13, 14));

        assert_eq!(spans, [Span::new(13, 14), Span::new(53, 54)]);
    }

    #[test]
    fn binding_with_a_local_and_parameter_collision_returns_both_in_source_order() {
        let source = "class C{void F(int x){int x;x=x;}}";
        let conflict = || {
            Ambiguous(
                DeclarationSpaceConflict,
                vec![Parameter(0, 0), Local(0, 0, 0)],
            )
        };

        let report = bind(source).unwrap();

        assert_eq!(
            (
                &report.lexical.headers.callables[0].parameters[0].name,
                &report.lexical.declarations.locals[0].name,
            ),
            (&name(("x", 19, 20)), &name(("x", 26, 27)))
        );
        assert_eq!(at_starts(report), [(28, conflict()), (30, conflict())]);
    }

    #[test]
    fn binding_with_active_later_locals_reports_conflict_before_checking_order() {
        let source = "class C{int x;void F(){{x=x;int x;}int @x;}}";
        let conflict = || {
            Ambiguous(
                DeclarationSpaceConflict,
                vec![Local(0, 1, 0), Local(0, 0, 0)],
            )
        };

        let report = bind(source).unwrap();

        assert_eq!(at_starts(report), [(24, conflict()), (26, conflict())]);
    }

    #[test]
    fn binding_with_duplicate_fields_returns_both_for_bare_and_this_accesses() {
        let source = "class C{int x;int x;void F(){this.x=x;}}";
        let conflict = || Ambiguous(DuplicateMember, vec![Field(0, 0), Field(0, 1)]);

        let report = bind(source).unwrap();

        assert_eq!(
            report
                .lexical
                .declarations
                .fields
                .iter()
                .map(|field| &field.name)
                .collect::<Vec<_>>(),
            [&name(("x", 12, 13)), &name(("x", 18, 19))]
        );
        assert_eq!(at_starts(report), [(34, conflict()), (36, conflict())]);
    }

    #[test]
    fn binding_in_a_utf8_region_compares_unescaped_names_but_preserves_raw_tokens() {
        let source = "π|class C{int @x;void F(){x=this.@x;}}|";
        let region = Span::new(3, 39);

        let report = bind_references(source, region, BindingSubset::Lexical).unwrap();

        assert_eq!(report.lexical.headers.types.region, region);
        assert_eq!(
            report.lexical.declarations.fields[0].name,
            name(("@x", 15, 17))
        );
        assert_eq!(
            report.references,
            [
                binding(
                    0,
                    0,
                    ("x", 27, 28),
                    Write,
                    Receiver::Bare,
                    Bound(Field(0, 0))
                ),
                binding(
                    1,
                    0,
                    ("@x", 34, 36),
                    Read,
                    Receiver::This(name(("this", 29, 33))),
                    Bound(Field(0, 0))
                ),
            ]
        );
    }

    #[test]
    fn binding_prefers_a_parameter_to_a_field_only_for_unqualified_access() {
        let source = "class C{int x;void F(int x){x=this.x;}}";

        let report = bind(source).unwrap();

        assert_eq!(
            at_starts(report),
            [(28, Bound(Parameter(0, 0))), (35, Bound(Field(0, 0)))]
        );
    }

    #[test]
    fn binding_in_a_constructor_sees_a_later_field_of_a_public_class() {
        let source = "public class C{C(){x=x;}int x;}";

        let report = bind(source).unwrap();

        assert_eq!(
            at_starts(report),
            [(19, Bound(Field(0, 0))), (21, Bound(Field(0, 0)))]
        );
    }

    #[test]
    fn binding_keeps_equal_callable_keys_and_nondeclaration_site_ids_separate() {
        let source = "class C{void F(int x){x=x;}void F(int x){x=x;}}";

        let report = bind(source).unwrap();

        let sites: Vec<_> = report
            .references
            .into_iter()
            .map(|reference| {
                let site = reference.site;
                (
                    site.ordinal,
                    site.owner.callable,
                    site.owner.block,
                    site.name.span.start,
                    reference.outcome,
                )
            })
            .collect();
        assert_eq!(
            sites,
            [
                (0, 0, 0, 22, Bound(Parameter(0, 0))),
                (1, 0, 0, 24, Bound(Parameter(0, 0))),
                (2, 1, 0, 41, Bound(Parameter(1, 0))),
                (3, 1, 0, 43, Bound(Parameter(1, 0))),
            ]
        );
    }

    #[test]
    fn binding_uses_name_independent_positions_within_each_declaration_owner() {
        let source = "class C{int a;public class N{int b;}bool x;void F(int p,bool q){int a;bool y;{}{this.x=y;y=q;}}}";

        let report = bind(source).unwrap();

        assert_eq!(
            at_starts(report),
            [
                (85, Bound(Field(0, 1))),
                (87, Bound(Local(0, 0, 1))),
                (89, Bound(Local(0, 0, 1))),
                (91, Bound(Parameter(0, 1))),
            ]
        );
    }

    #[test]
    fn binding_does_not_match_a_field_with_different_case() {
        let source = "class C{int x;void F(){X=x;}}";

        let report = bind(source).unwrap();

        assert_eq!(
            at_starts(report),
            [(23, Unbound(NotFound)), (25, Bound(Field(0, 0)))]
        );
    }

    #[test]
    fn binding_does_not_search_fields_of_an_enclosing_type() {
        let source = "class C{int x;class N{void F(){this.x=x;}}}";

        let report = bind(source).unwrap();

        assert_eq!(
            at_starts(report),
            [(36, Unbound(NotFound)), (38, Unbound(NotFound))]
        );
    }

    #[test]
    fn binding_reports_wrong_kind_for_a_single_direct_method() {
        let source = "class C{void x(){}void F(){x=this.x;}}";

        let report = bind(source).unwrap();

        assert_eq!(
            at_starts(report),
            [(27, Unbound(WrongKind)), (34, Unbound(WrongKind))]
        );
    }

    #[test]
    fn binding_does_not_ignore_a_wrong_kind_member_competing_with_a_field() {
        for (source, starts, wrong_kind) in [
            (
                "class C{void x(){}int x;void F(){this.x=x;}}",
                [38, 40],
                Method(0),
            ),
            (
                "class C{class x{}int x;void F(){this.x=x;}}",
                [37, 39],
                Type(1),
            ),
        ] {
            let conflict = || Ambiguous(CompetingMember, vec![wrong_kind, Field(0, 0)]);

            let report = bind(source).unwrap();

            assert_eq!(
                at_starts(report),
                [(starts[0], conflict()), (starts[1], conflict())],
                "{source}"
            );
        }
    }

    #[test]
    fn binding_resolves_qualified_header_types_but_never_infers_expression_receivers() {
        let source = "class C{void F(global::C p){p.x=p.@return;}}";

        let report = bind(source).unwrap();

        assert_eq!(
            report.lexical.headers.callables[0].parameters[0].comparison_type,
            CallableParameterType::LocalType(0)
        );
        assert_eq!(
            report.references,
            [
                binding(
                    0,
                    0,
                    ("p", 28, 29),
                    ReceiverName,
                    Receiver::Bare,
                    Bound(Parameter(0, 0))
                ),
                binding(
                    1,
                    0,
                    ("x", 30, 31),
                    Write,
                    Receiver::Name(name(("p", 28, 29))),
                    Unbound(UnknownReceiver)
                ),
                binding(
                    2,
                    0,
                    ("p", 32, 33),
                    ReceiverName,
                    Receiver::Bare,
                    Bound(Parameter(0, 0))
                ),
                binding(
                    3,
                    0,
                    ("@return", 34, 41),
                    Read,
                    Receiver::Name(name(("p", 32, 33))),
                    Unbound(UnknownReceiver)
                ),
            ]
        );
    }

    #[test]
    fn binding_treats_escaped_this_as_an_ordinary_receiver_name() {
        let source = "class C{int @this;void F(){@this.x=@this;}}";

        let report = bind(source).unwrap();

        assert_eq!(
            report.references,
            [
                binding(
                    0,
                    0,
                    ("@this", 27, 32),
                    ReceiverName,
                    Receiver::Bare,
                    Bound(Field(0, 0))
                ),
                binding(
                    1,
                    0,
                    ("x", 33, 34),
                    Write,
                    Receiver::Name(name(("@this", 27, 32))),
                    Unbound(UnknownReceiver)
                ),
                binding(
                    2,
                    0,
                    ("@this", 35, 40),
                    Read,
                    Receiver::Bare,
                    Bound(Field(0, 0))
                ),
            ]
        );
    }

    #[test]
    fn binding_refuses_a_non_utf8_boundary_before_examining_syntax() {
        let source = "π|class C{}";
        let region = Span::new(1, 12);

        let result = bind_references(source, region, BindingSubset::Lexical);

        assert_eq!(
            result,
            Err(GenericCallableIndexError::InvalidRegion(region))
        );
    }

    #[test]
    fn binding_refuses_invalid_statements_without_partial_evidence() {
        for (source, start, end) in [
            ("class C{void F(){x=p;int y;{bool z;}x=q();}}", 39, 40),
            ("class C{void F(){x=y.z.q;}}", 22, 23),
            ("class C{void F(){x=y=z;}}", 20, 21),
            ("class C{void F(){x=this;}}", 19, 23),
            ("class C{void F(){x=this.return;}}", 24, 30),
            ("class C{void F(){x=file;}}", 19, 23),
            ("class C{void F(){x=scoped;}}", 19, 25),
            ("class C{void F(){x=base.y;}}", 19, 23),
            ("class C{void F(){int x=1;}}", 22, 23),
            ("class C{void F(){x=y}}", 20, 21),
        ] {
            let result = bind(source);

            assert_eq!(
                result,
                Err(rejected(Span::new(start, end), vec![])),
                "{source}"
            );
        }
    }

    #[test]
    fn binding_refuses_declarations_outside_the_instance_subset() {
        for (source, start, end) in [
            ("public static class C{}", 7, 13),
            ("class C{abstract class D{}}", 8, 16),
            ("class C{static int x;}", 8, 14),
            ("class C:B{}", 7, 8),
            ("class C{int x{get;}}", 13, 14),
            ("class C{int x=1;}", 13, 14),
        ] {
            let result = bind(source);

            assert_eq!(
                result,
                Err(rejected(Span::new(start, end), vec![])),
                "{source}"
            );
        }
    }

    #[test]
    fn binding_refuses_generic_methods_and_keeps_legacy_diagnostics() {
        let source = "class C{void F<T>(){}}";

        let result = bind(source);

        assert_eq!(
            result,
            Err(rejected(
                Span::new(14, 15),
                vec![
                    Span::new(14, 15),
                    Span::new(15, 16),
                    Span::new(16, 20),
                    Span::new(21, 21),
                    Span::new(21, 22),
                ]
            ))
        );
    }

    #[test]
    fn binding_refusal_for_an_unsupported_operator_keeps_ordered_recovery() {
        let result = bind(TRAILING_CLOSER);

        assert_eq!(
            result,
            Err(rejected(
                Span::new(20, 21),
                vec![Span::new(26, 26), Span::new(26, 27)]
            ))
        );
    }

    #[test]
    fn public_parser_without_binding_still_recovers_only_at_the_stray_closer() {
        let unit = cs_parse(TRAILING_CLOSER, 0);

        assert_eq!(unit.errors, [Span::new(26, 26), Span::new(26, 27)]);
    }

    #[test]
    fn binding_requires_an_actual_closing_type_brace_after_an_assignment() {
        let result = bind("class C{void F(){x=y;}");

        assert_eq!(result, Err(rejected(Span::new(22, 22), vec![])));
    }

    #[test]
    fn binding_refuses_a_region_start_bom() {
        let result = bind("\u{feff}class C{}");

        assert_eq!(result, Err(rejected(Span::new(0, 3), vec![])));
    }

    #[test]
    fn binding_refuses_a_failed_qualified_lookup_after_a_clean_parse() {
        let result = bind("class C{void F(global::Missing p){p=p;}}");

        assert_eq!(result, Err(rejected(Span::new(15, 30), vec![])));
    }

    #[test]
    fn binding_refuses_the_opening_brace_of_a_depth_skipped_block() {
        let source = depth_fixture();

        let result = bind(&source);

        assert_eq!(result, Err(rejected(Span::new(217, 218), vec![])));
    }

    #[test]
    fn checked_index_refuses_the_opening_brace_of_a_depth_skipped_block() {
        let source = depth_fixture();

        let result = index_checked(&source, Span::new(0, source.len()));

        assert_eq!(result, Err(rejected(Span::new(217, 218), vec![])));
    }

    #[test]
    fn public_parser_without_binding_still_skips_a_depth_limited_block_silently() {
        use crate::v2::csharp::ast::CsNodeKind;
        let source = depth_fixture();

        let unit = cs_parse(&source, 0);

        let mut skipped = Vec::new();
        unit.root.walk(&mut |node| {
            if node.kind == CsNodeKind::Block && node.children.is_empty() {
                skipped.push(node.span);
            }
        });
        assert_eq!((unit.errors, skipped), (vec![], vec![Span::new(217, 223)]));
    }
}
