//! Complete-or-refuse planning within one checked C# region.

use super::ast::CsName;
#[cfg(test)]
use super::lexer::lex_checked;
use super::local::binding::{
    Ambiguity, CheckedReferenceBinding, DeclarationId, Receiver, RenameBinding,
};
#[cfg(test)]
use super::local::index_checked;
use super::local::{
    comparison_identifier, index_checked_mode, CallableHeaderRow, CallableKind, CheckedIndex,
    DeclarationRole, GenericCallableIndexError, QualifiedTypeCandidate, QualifiedTypeFailure,
    RenameBaseFailure,
};
use super::parser::CheckedHeaderMode;
use super::resolve::DeclKind;
#[cfg(test)]
use super::tokens::CsTokenKind;
use crate::v2::edit::{apply_edits, EditError, TextEdit};
use crate::v2::span::Span;
use std::collections::HashSet;
use std::{error::Error, fmt};

/// A checked snapshot of one caller-selected raw C# region of a UTF-8 document.
///
/// All spans are half-open byte ranges in the **entire original document**.
/// This is an opt-in bounded model, not a C# compiler or a resolver across
/// units. It never edits PowerShell or consults the legacy rename collectors.
/// Parse again and reselect the declaration after any document change.
///
/// ```
/// use poshtree::v2::{Span, TextEdit};
/// use poshtree::v2::csharp::{CheckedCSharpRegion, CSharpRenameMode};
///
/// let source = "π|class C{int @x;void F(){x=this.@x;}}|";
/// let region = CheckedCSharpRegion::parse(source, Span::new(3, 39))?;
/// let member = region.member_at(Span::new(15, 17))?;
/// let plan = member.plan_rename(CSharpRenameMode::Single, "count")?;
/// assert_eq!(plan.edits(), &[
///     TextEdit::replace(Span::new(15, 17), "count"),
///     TextEdit::replace(Span::new(27, 28), "count"),
///     TextEdit::replace(Span::new(34, 36), "count"),
/// ]);
/// assert_eq!(plan.apply_to(source)?, "π|class C{int count;void F(){count=this.count;}}|");
/// # Ok::<(), poshtree::v2::csharp::CSharpRenameError>(())
/// ```
pub struct CheckedCSharpRegion<'src> {
    source: &'src str,
    region: Span,
    index: CheckedIndex,
}

impl fmt::Debug for CheckedCSharpRegion<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckedCSharpRegion")
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

impl<'src> CheckedCSharpRegion<'src> {
    /// Observes only the selected raw C# slice, retaining the document snapshot.
    ///
    /// # Errors
    ///
    /// Refuses invalid UTF-8 ranges, unsupported constructs (including generic
    /// methods anywhere in the region), recovery, depth limits, and unproved
    /// explicit parameter types. Some checks depend on the selected member
    /// and run only when planning the rename.
    pub fn parse(source: &'src str, region: Span) -> Result<Self, CSharpRenameError> {
        let index = index_checked_mode(source, region, CheckedHeaderMode::Rename)
            .map_err(CSharpRenameError::from_index)?;
        Ok(Self {
            source,
            region,
            index,
        })
    }

    /// Selects a field or method by its exact raw declaration-name token span.
    ///
    /// # Errors
    ///
    /// A range outside this region, reversed, or splitting a UTF-8 character
    /// is [`CSharpRenameError::InvalidSelection`]. A valid range that is not
    /// exactly the declaration name is [`CSharpRenameError::NotMemberDeclaration`].
    /// This includes cursors, partial token ranges, uses, and other declarations.
    pub fn member_at<'r>(
        &'r self,
        name: Span,
    ) -> Result<CSharpMember<'r, 'src>, CSharpRenameError> {
        if name.start < self.region.start
            || name.end > self.region.end
            || self.source.get(name.start..name.end).is_none()
        {
            return Err(CSharpRenameError::InvalidSelection(name));
        }
        let target = if self
            .index
            .declaration_at(name, DeclarationRole::Field)
            .is_some()
        {
            RenameTarget::Field(name)
        } else if self
            .index
            .declaration_at(name, DeclarationRole::Callable(CallableKind::Method))
            .is_some()
        {
            RenameTarget::Method(name)
        } else {
            return Err(CSharpRenameError::NotMemberDeclaration(name));
        };
        Ok(CSharpMember {
            snapshot: self,
            target,
        })
    }
}

/// An exact field or method declaration selected in one checked snapshot.
///
/// The member borrows its [`CheckedCSharpRegion`]; it cannot be transferred to
/// another parse. No persistent symbol identity is implied by its name span.
#[derive(Debug)]
pub struct CSharpMember<'r, 'src> {
    snapshot: &'r CheckedCSharpRegion<'src>,
    target: RenameTarget,
}

impl<'src> CSharpMember<'_, 'src> {
    /// Returns the declaration kind, either [`DeclKind::Field`] or [`DeclKind::Method`].
    pub fn kind(&self) -> DeclKind {
        match self.target {
            RenameTarget::Field(_) => DeclKind::Field,
            RenameTarget::Method(_) | RenameTarget::OverloadGroup(_) => DeclKind::Method,
        }
    }

    /// Returns the exact original name span, including any `@` prefix.
    pub fn name_span(&self) -> Span {
        self.target.name_span()
    }

    /// Plans every required edit within the checked model or refuses the operation.
    ///
    /// The plan borrows the original document, not the region handle or `to`.
    ///
    /// # Errors
    ///
    /// Refuses unsupported owners or binding, incomplete bases, ambiguous or
    /// unresolved matching uses, conflicting declarations, and invalid,
    /// unchanged, or non-fresh replacements. A field cannot anchor a group.
    ///
    /// ```
    /// use poshtree::v2::{Span, TextEdit};
    /// use poshtree::v2::csharp::{CheckedCSharpRegion, CSharpRenameError, CSharpRenameMode};
    ///
    /// let source = "class C{void F(){}void F(int a){}void G(int v){F();this.F(v);}}";
    /// let region = CheckedCSharpRegion::parse(source, Span::new(0, 63))?;
    /// let plan = region.member_at(Span::new(13, 14))?.plan_rename(CSharpRenameMode::Single, "Renamed")?;
    /// assert_eq!(plan.edits(), &[
    ///     TextEdit::replace(Span::new(13, 14), "Renamed"),
    ///     TextEdit::replace(Span::new(47, 48), "Renamed"),
    /// ]);
    /// assert_eq!(plan.apply_to(source)?,
    ///     "class C{void Renamed(){}void F(int a){}void G(int v){Renamed();this.F(v);}}");
    /// # Ok::<(), CSharpRenameError>(())
    /// ```
    ///
    /// With same-count overloads, an exact argument type is evidence of
    /// applicability, not a way to choose one declaration.
    ///
    /// ```
    /// use poshtree::v2::{Span, TextEdit};
    /// use poshtree::v2::csharp::{CheckedCSharpRegion, CSharpRenameError, CSharpRenameMode};
    ///
    /// let source = "class C{void F(int a){}void F(bool b){}void G(int v){F(v);}}";
    /// let region = CheckedCSharpRegion::parse(source, Span::new(0, 60))?;
    /// let method = region.member_at(Span::new(28, 29))?;
    /// assert_eq!(method.plan_rename(CSharpRenameMode::Single, "Renamed").unwrap_err(),
    ///     CSharpRenameError::Ambiguous {
    ///         at: Span::new(53, 54),
    ///         candidates: vec![Span::new(13, 14), Span::new(28, 29)],
    ///     });
    /// let plan = method.plan_rename(CSharpRenameMode::OverloadGroup, "Renamed")?;
    /// assert_eq!(plan.edits(), &[
    ///     TextEdit::replace(Span::new(13, 14), "Renamed"),
    ///     TextEdit::replace(Span::new(28, 29), "Renamed"),
    ///     TextEdit::replace(Span::new(53, 54), "Renamed"),
    /// ]);
    /// assert_eq!(plan.apply_to(source)?,
    ///     "class C{void Renamed(int a){}void Renamed(bool b){}void G(int v){Renamed(v);}}");
    /// # Ok::<(), CSharpRenameError>(())
    /// ```
    pub fn plan_rename(
        &self,
        mode: CSharpRenameMode,
        to: &str,
    ) -> Result<CSharpRenamePlan<'src>, CSharpRenameError> {
        let target = match (self.target, mode) {
            (RenameTarget::Field(span), CSharpRenameMode::OverloadGroup) => {
                return Err(CSharpRenameError::GroupRequiresMethod(span));
            }
            (RenameTarget::Method(span), CSharpRenameMode::OverloadGroup) => {
                RenameTarget::OverloadGroup(span)
            }
            (target, _) => target,
        };
        let snapshot = self.snapshot;
        let edits = plan_indexed_rename(
            &snapshot.index,
            snapshot.source,
            snapshot.region,
            target,
            to,
        )
        .map_err(CSharpRenameError::from_plan)?;
        Ok(CSharpRenamePlan {
            source: snapshot.source,
            edits,
        })
    }
}

/// Specifies which declarations a checked C# rename should include.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CSharpRenameMode {
    /// Rename just the selected field or method, refusing ambiguous uses.
    Single,
    /// Rename all same-name methods of the selected method's exact owner.
    /// A group call still needs proof that at least one member is applicable.
    OverloadGroup,
}

/// A complete batch of checked edits tied to the original document bytes.
///
/// The plan owns every replacement. It may outlive the region and member
/// handles, but not the original document it borrows.
pub struct CSharpRenamePlan<'src> {
    source: &'src str,
    edits: Vec<TextEdit>,
}

impl fmt::Debug for CSharpRenamePlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CSharpRenamePlan")
            .field("edits", &self.edits)
            .finish_non_exhaustive()
    }
}

impl CSharpRenamePlan<'_> {
    /// Returns the nonoverlapping edits in original document byte order.
    pub fn edits(&self) -> &[TextEdit] {
        &self.edits
    }

    /// Applies the whole plan to `current_source`, preserving all other bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CSharpRenameError::StaleSource`] unless the entire document
    /// matches byte-for-byte (an identical copy is valid). Edit validation
    /// failures are returned as [`CSharpRenameError::Edit`].
    pub fn apply_to(&self, current_source: &str) -> Result<String, CSharpRenameError> {
        if self.source.as_bytes() != current_source.as_bytes() {
            return Err(CSharpRenameError::StaleSource);
        }
        apply_edits(current_source, &self.edits).map_err(CSharpRenameError::Edit)
    }
}

/// A typed checked-rename refusal. No partially checked edits are returned.
///
/// Spans always refer to the original document. Recovery spans retain parser
/// order; candidate and duplicate-declaration spans are in source order.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CSharpRenameError {
    /// The region was reversed, out of bounds, or split a UTF-8 character.
    InvalidRegion(Span),
    /// The selection was reversed, outside the region, or split a UTF-8 character.
    InvalidSelection(Span),
    /// A valid range did not select an exact field or method declaration name.
    NotMemberDeclaration(Span),
    /// The selected declaration cannot anchor an overload group.
    GroupRequiresMethod(Span),
    /// The replacement is not a supported unescaped ASCII identifier.
    InvalidReplacement,
    /// The replacement matches the existing name after removing a leading `@`.
    UnchangedName,
    /// The selected region or owner is outside the checked subset.
    UnsupportedRegion {
        /// The first subset refusal, if known (possibly at end of region).
        first: Option<Span>,
        /// Parser recovery ranges in their original order.
        recovery: Vec<Span>,
    },
    /// The parser hit its shared nesting limit.
    DepthLimit {
        /// The token at entry to the skipped or recovered construct.
        at: Span,
        /// Parser recovery ranges in their original order.
        recovery: Vec<Span>,
    },
    /// An explicit base path could not be proved within the region.
    IncompleteBase(Span),
    /// An explicit base path closes a cycle within the region.
    BaseCycle(Span),
    /// Binding or exact applicability could not be proved at this site.
    Unresolved {
        /// The use or path segment that could not be proved.
        at: Span,
    },
    /// More than one declaration competes at this site.
    Ambiguous {
        /// The use or path segment that could not be bound uniquely.
        at: Span,
        /// The competing original declaration-name spans.
        candidates: Vec<Span>,
    },
    /// The same supported declaration identity occurs more than once.
    DuplicateDeclaration {
        /// The conflicting original declaration-name spans.
        declarations: Vec<Span>,
    },
    /// The replacement is already used, or an existing name conflict prevents the rename.
    NameConflict(Span),
    /// The checked evidence did not agree with the selected occurrence or source bytes.
    InconsistentEvidence(Span),
    /// The current document does not match the original snapshot byte-for-byte.
    StaleSource,
    /// The existing text-edit engine refused the batch.
    Edit(EditError),
}

impl fmt::Display for CSharpRenameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRegion(span) => write!(f, "invalid C# region at {span}"),
            Self::InvalidSelection(span) => write!(f, "invalid C# member selection at {span}"),
            Self::NotMemberDeclaration(span) => {
                write!(f, "no exact field or method declaration at {span}")
            }
            Self::GroupRequiresMethod(span) => {
                write!(f, "overload group requires a method at {span}")
            }
            Self::InvalidReplacement => f.write_str("invalid C# replacement identifier"),
            Self::UnchangedName => f.write_str("C# replacement leaves the name unchanged"),
            Self::UnsupportedRegion {
                first: Some(span), ..
            } => write!(f, "unsupported C# region at {span}"),
            Self::UnsupportedRegion { first: None, .. } => {
                f.write_str("C# region required parser recovery")
            }
            Self::DepthLimit { at, .. } => write!(f, "C# nesting limit reached at {at}"),
            Self::IncompleteBase(span) => write!(f, "unproved C# base at {span}"),
            Self::BaseCycle(span) => write!(f, "C# inheritance cycle at {span}"),
            Self::Unresolved { at } => write!(f, "unresolved C# use at {at}"),
            Self::Ambiguous { at, .. } => write!(f, "ambiguous C# binding at {at}"),
            Self::DuplicateDeclaration { .. } => f.write_str("duplicate C# declarations"),
            Self::NameConflict(span) => write!(f, "C# name conflict at {span}"),
            Self::InconsistentEvidence(span) => write!(f, "inconsistent C# evidence at {span}"),
            Self::StaleSource => {
                f.write_str("C# rename plan belongs to a different document snapshot")
            }
            Self::Edit(error) => write!(f, "could not apply C# rename edits: {error}"),
        }
    }
}

impl Error for CSharpRenameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Edit(error) => Some(error),
            _ => None,
        }
    }
}

/// The span must be the exact raw name token of the declaration, not a use,
/// an offset into the name, or a name/signature key from a different parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenameTarget {
    Field(Span),
    Method(Span),
    /// Anchor at one method; include all same-name methods of its exact owner.
    OverloadGroup(Span),
}

impl RenameTarget {
    fn name_span(self) -> Span {
        match self {
            Self::Field(span) | Self::Method(span) | Self::OverloadGroup(span) => span,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum CheckedRenameError {
    #[cfg(test)]
    Index(GenericCallableIndexError),
    TargetNotFound(RenameTarget),
    #[cfg(test)]
    GenericMethod(Span),
    GenericOwner(Span),
    UnprovenOwner(QualifiedTypeFailure),
    Base(RenameBaseFailure),
    UnsupportedIdentifier(Span),
    InvalidReplacement,
    UnchangedName,
    ReplacementInUse(Span),
    ConflictingMember(Span),
    DuplicateDeclarations(Vec<Span>),
    InheritedNameConflict(Span),
    UnresolvedUse(Span),
    AmbiguousUse {
        span: Span,
        reason: Ambiguity,
        candidates: Vec<Span>,
    },
    IncompleteOverloadUse(Span),
    InvalidEdit(Span),
    WouldMergeTokens(Span),
    OverlappingEdits {
        first: Span,
        second: Span,
    },
}

impl CSharpRenameError {
    fn unsupported(span: Span) -> Self {
        Self::UnsupportedRegion {
            first: Some(span),
            recovery: Vec::new(),
        }
    }

    fn from_index(error: GenericCallableIndexError) -> Self {
        match error {
            GenericCallableIndexError::InvalidRegion(span) => Self::InvalidRegion(span),
            GenericCallableIndexError::DepthLimit { at, recovery } => {
                Self::DepthLimit { at, recovery }
            }
            GenericCallableIndexError::Rejected {
                first_unsupported,
                recovery,
            } => Self::UnsupportedRegion {
                first: first_unsupported,
                recovery,
            },
            GenericCallableIndexError::CheckedParameterType { failure, .. } => {
                Self::from_qualified(failure)
            }
        }
    }

    fn from_qualified(failure: QualifiedTypeFailure) -> Self {
        match failure {
            QualifiedTypeFailure::Ambiguous {
                segment,
                candidates,
            }
            | QualifiedTypeFailure::AmbiguousTypeMember {
                segment,
                candidates,
                ..
            } => Self::Ambiguous {
                at: segment,
                candidates: candidates
                    .iter()
                    .map(QualifiedTypeCandidate::span)
                    .collect(),
            },
            QualifiedTypeFailure::MissingType { segment } => Self::Unresolved { at: segment },
            QualifiedTypeFailure::UnsupportedArity { segment } => Self::unsupported(segment),
        }
    }

    fn from_plan(error: CheckedRenameError) -> Self {
        match error {
            #[cfg(test)]
            CheckedRenameError::Index(error) => Self::from_index(error),
            #[cfg(test)]
            CheckedRenameError::GenericMethod(span) => Self::unsupported(span),
            CheckedRenameError::GenericOwner(span)
            | CheckedRenameError::UnsupportedIdentifier(span)
            | CheckedRenameError::WouldMergeTokens(span) => Self::unsupported(span),
            CheckedRenameError::TargetNotFound(target) => {
                Self::InconsistentEvidence(target.name_span())
            }
            CheckedRenameError::UnprovenOwner(QualifiedTypeFailure::Ambiguous {
                candidates,
                ..
            }) if candidates
                .iter()
                .all(|candidate| matches!(candidate, QualifiedTypeCandidate::Type { .. })) =>
            {
                Self::DuplicateDeclaration {
                    declarations: candidates
                        .iter()
                        .map(QualifiedTypeCandidate::span)
                        .collect(),
                }
            }
            CheckedRenameError::UnprovenOwner(failure) => Self::from_qualified(failure),
            CheckedRenameError::Base(RenameBaseFailure::Cycle { span, .. }) => {
                Self::BaseCycle(span)
            }
            CheckedRenameError::Base(RenameBaseFailure::Incomplete { span, .. }) => {
                Self::IncompleteBase(span)
            }
            CheckedRenameError::InvalidReplacement => Self::InvalidReplacement,
            CheckedRenameError::UnchangedName => Self::UnchangedName,
            CheckedRenameError::ReplacementInUse(span)
            | CheckedRenameError::ConflictingMember(span)
            | CheckedRenameError::InheritedNameConflict(span) => Self::NameConflict(span),
            CheckedRenameError::DuplicateDeclarations(declarations) => {
                Self::DuplicateDeclaration { declarations }
            }
            CheckedRenameError::UnresolvedUse(at)
            | CheckedRenameError::IncompleteOverloadUse(at) => Self::Unresolved { at },
            CheckedRenameError::AmbiguousUse {
                span, candidates, ..
            } => Self::Ambiguous {
                at: span,
                candidates,
            },
            CheckedRenameError::InvalidEdit(span)
            | CheckedRenameError::OverlappingEdits { second: span, .. } => {
                Self::InconsistentEvidence(span)
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Occurrence {
    Declaration(DeclarationId),
    Reference(usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Observation<'a> {
    occurrence: Occurrence,
    name: &'a CsName,
}

struct Selection<'a> {
    declarations: Vec<Observation<'a>>,
    ids: HashSet<DeclarationId>,
    group: bool,
}

struct NamedMember<'a> {
    owner: usize,
    id: DeclarationId,
    name: &'a CsName,
    method: Option<&'a CallableHeaderRow>,
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
struct CheckedRenamePlan {
    region: Span,
    edits: Vec<TextEdit>,
}

#[cfg(test)]
fn plan_member_rename(
    source: &str,
    region: Span,
    target: RenameTarget,
    replacement: &str,
) -> Result<CheckedRenamePlan, CheckedRenameError> {
    let index = index_checked(source, region).map_err(CheckedRenameError::Index)?;
    if let Some(method) = index
        .lexical
        .headers
        .callables
        .iter()
        .find(|method| !method.type_params.is_empty())
    {
        return Err(generic_method_refusal(source, method));
    }
    let edits = plan_indexed_rename(&index, source, region, target, replacement)?;
    Ok(CheckedRenamePlan { region, edits })
}

fn plan_indexed_rename(
    index: &CheckedIndex,
    source: &str,
    region: Span,
    target: RenameTarget,
    replacement: &str,
) -> Result<Vec<TextEdit>, CheckedRenameError> {
    let headers = &index.lexical.headers;
    let (span, role) = match target {
        RenameTarget::Field(span) => (span, DeclarationRole::Field),
        RenameTarget::Method(span) | RenameTarget::OverloadGroup(span) => {
            (span, DeclarationRole::Callable(CallableKind::Method))
        }
    };
    let declaration = index
        .declaration_at(span, role)
        .ok_or(CheckedRenameError::TargetNotFound(target))?;
    let owner = match declaration.id {
        DeclarationId::Field(owner, _) => owner,
        DeclarationId::Callable(ordinal) => headers.callables[ordinal].containing_type,
        _ => return Err(CheckedRenameError::TargetNotFound(target)),
    };
    if let Some(generic) = headers
        .types
        .type_lineage(Some(owner))
        .find(|row| !row.type_params.is_empty())
    {
        return Err(CheckedRenameError::GenericOwner(generic.name.span));
    }
    index
        .check_rename_owner(owner)
        .map_err(CheckedRenameError::UnprovenOwner)?;
    let relatives = index
        .rename_base_relatives(owner)
        .map_err(CheckedRenameError::Base)?;

    let old_name = comparison_identifier(&declaration.name.text);
    check_replacement(index, source, old_name, replacement)?;
    let selection = select_members(index, owner, &relatives, declaration.id, old_name, target)?;
    let mut observations = selection.declarations;
    for reference in &index.references {
        check_qualifier(reference, old_name, &selection.ids)?;
        let name = &reference.site.name;
        if comparison_identifier(&name.text) != old_name {
            continue;
        }
        if selects_use(reference, &selection.ids, selection.group, index)? {
            observations.push(Observation {
                occurrence: Occurrence::Reference(reference.site.ordinal),
                name,
            });
        }
    }

    checked_edits(source, region, observations, replacement)
}

#[cfg(test)]
fn generic_method_refusal(source: &str, method: &CallableHeaderRow) -> CheckedRenameError {
    // The coverage report kept the binder and method name, not the delimiter.
    // Recover only this already certified token interval, so comments before
    // `<` cannot shift the refusal onto a trivia byte.
    let gap = Span::new(method.name.span.end, method.type_params[0].span.start);
    let Some(code) = source.get(gap.start..gap.end) else {
        return CheckedRenameError::InvalidEdit(gap);
    };
    let (tokens, refusal) = lex_checked(code, gap.start);
    match tokens.first() {
        Some(token) if refusal.is_none() && token.kind == CsTokenKind::Lt => {
            CheckedRenameError::GenericMethod(token.span)
        }
        _ => CheckedRenameError::InvalidEdit(gap),
    }
}

fn check_replacement(
    index: &CheckedIndex,
    source: &str,
    old_name: &str,
    replacement: &str,
) -> Result<(), CheckedRenameError> {
    if !plain_identifier(replacement)
        || reserved(replacement)
        || contextual(replacement)
        || replacement == "_"
    {
        return Err(CheckedRenameError::InvalidReplacement);
    }
    if old_name == replacement {
        return Err(CheckedRenameError::UnchangedName);
    }
    let mut spans: Vec<_> = index.identifier_spans().collect();
    spans.sort_unstable_by_key(|span| (span.start, span.end));
    for span in spans {
        let raw = source
            .get(span.start..span.end)
            .ok_or(CheckedRenameError::UnsupportedIdentifier(span))?;
        let name = comparison_identifier(raw);
        if !plain_identifier(name) || (raw == name && reserved(name)) {
            return Err(CheckedRenameError::UnsupportedIdentifier(span));
        }
        if name == replacement {
            return Err(CheckedRenameError::ReplacementInUse(span));
        }
    }
    Ok(())
}

fn plain_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    // Consecutive underscores are reserved for implementations, whether or
    // not the spelling is one of today's compiler intrinsics.
    matches!(bytes.next(), Some(b) if b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && !name.contains("__")
}

fn reserved(name: &str) -> bool {
    const KEYWORDS: &str = "abstract as base bool break byte case catch char checked class const \
        continue decimal default delegate do double else enum event explicit extern false finally \
        fixed float for foreach goto if implicit in int interface internal is lock long namespace \
        new null object operator out override params private protected public readonly ref return \
        sbyte sealed short sizeof stackalloc static string struct switch this throw true try \
        typeof uint ulong unchecked unsafe ushort using virtual void volatile while";
    KEYWORDS
        .split_ascii_whitespace()
        .any(|keyword| keyword == name)
}

fn contextual(name: &str) -> bool {
    const KEYWORDS: &str = "add alias allows and ascending async await by descending dynamic \
        equals extension field file from get global group init into join let managed nameof nint \
        not notnull nuint on or orderby partial record remove required scoped select set unmanaged \
        value var when where with yield";
    KEYWORDS
        .split_ascii_whitespace()
        .any(|keyword| keyword == name)
}

fn named_members(index: &CheckedIndex) -> impl Iterator<Item = NamedMember<'_>> {
    let headers = &index.lexical.headers;
    let fields = index
        .lexical
        .declarations
        .field_occurrences()
        .map(|(id, row)| NamedMember {
            owner: row.containing_type,
            id,
            name: &row.name,
            method: None,
        });
    let callables = headers.callables.iter().map(|row| NamedMember {
        owner: row.containing_type,
        id: DeclarationId::Callable(row.ordinal),
        name: &row.name,
        method: (row.kind == CallableKind::Method).then_some(row),
    });
    let types = headers.types.rows.iter().filter_map(|row| {
        Some(NamedMember {
            owner: row.containing_type?,
            id: DeclarationId::Type(row.ordinal),
            name: &row.name,
            method: None,
        })
    });
    fields.chain(callables).chain(types)
}

fn select_members<'a>(
    index: &'a CheckedIndex,
    owner: usize,
    relatives: &HashSet<usize>,
    anchor: DeclarationId,
    old_name: &str,
    target: RenameTarget,
) -> Result<Selection<'a>, CheckedRenameError> {
    let owner_name = &index.lexical.headers.types.rows[owner].name;
    if comparison_identifier(&owner_name.text) == old_name {
        return Err(CheckedRenameError::ConflictingMember(owner_name.span));
    }
    let group = matches!(target, RenameTarget::OverloadGroup(_));
    let method_target = !matches!(target, RenameTarget::Field(_));
    let mut members: Vec<_> = named_members(index)
        .filter(|member| {
            comparison_identifier(&member.name.text) == old_name
                && (member.owner == owner || relatives.contains(&member.owner))
        })
        .collect();
    members.sort_unstable_by_key(|member| member.name.span.start);

    let mut signatures = HashSet::new();
    let mut declarations = Vec::new();
    let mut ids = HashSet::new();
    for member in &members {
        let span = member.name.span;
        if member.owner != owner {
            return Err(CheckedRenameError::InheritedNameConflict(span));
        }
        if method_target {
            let method = member
                .method
                .ok_or(CheckedRenameError::ConflictingMember(span))?;
            if !signatures.insert(method.parameter_types().collect::<Vec<_>>()) {
                let duplicates = members.iter().filter_map(|candidate| {
                    let other = candidate.method?;
                    (candidate.owner == owner
                        && other.parameter_types().eq(method.parameter_types()))
                    .then_some(candidate.name.span)
                });
                return Err(CheckedRenameError::DuplicateDeclarations(
                    duplicates.collect(),
                ));
            }
        } else if member.id != anchor {
            if matches!(member.id, DeclarationId::Field(..)) {
                let duplicates = members.iter().filter_map(|candidate| {
                    (candidate.owner == owner && matches!(candidate.id, DeclarationId::Field(..)))
                        .then_some(candidate.name.span)
                });
                return Err(CheckedRenameError::DuplicateDeclarations(
                    duplicates.collect(),
                ));
            }
            return Err(CheckedRenameError::ConflictingMember(span));
        }
        if group || member.id == anchor {
            declarations.push(Observation {
                occurrence: Occurrence::Declaration(member.id),
                name: member.name,
            });
            ids.insert(member.id);
        }
    }
    if !ids.contains(&anchor) {
        return Err(CheckedRenameError::TargetNotFound(target));
    }
    Ok(Selection {
        declarations,
        ids,
        group,
    })
}

fn selects_use(
    reference: &CheckedReferenceBinding,
    selected: &HashSet<DeclarationId>,
    group: bool,
    index: &CheckedIndex,
) -> Result<bool, CheckedRenameError> {
    let span = reference.site.name.span;
    match reference.rename_binding() {
        RenameBinding::Bound(id) => Ok(selected.contains(&id)),
        RenameBinding::Unresolved => Err(CheckedRenameError::UnresolvedUse(span)),
        // A lexical conflict stops lookup before member candidates are
        // collected, so a disjoint set does not prove the use is unrelated.
        RenameBinding::Ambiguous(reason @ Ambiguity::DeclarationSpaceConflict, candidates) => {
            Err(CheckedRenameError::AmbiguousUse {
                span,
                reason,
                candidates: index.candidate_spans(candidates),
            })
        }
        RenameBinding::Ambiguous(reason, candidates) => {
            if candidates.is_empty() {
                return Err(CheckedRenameError::IncompleteOverloadUse(span));
            }
            if !candidates.iter().any(|id| selected.contains(id)) {
                return Ok(false);
            }
            if !group || reason != Ambiguity::Overload {
                return Err(CheckedRenameError::AmbiguousUse {
                    span,
                    reason,
                    candidates: index.candidate_spans(candidates),
                });
            }
            let mut seen = HashSet::new();
            let mut methods = Vec::new();
            for &id in candidates {
                let DeclarationId::Callable(ordinal) = id else {
                    return Err(CheckedRenameError::IncompleteOverloadUse(span));
                };
                if !selected.contains(&id) || !seen.insert(id) {
                    return Err(CheckedRenameError::IncompleteOverloadUse(span));
                }
                methods.push(&index.lexical.headers.callables[ordinal]);
            }
            if !reference.has_exact_applicability(methods.into_iter()) {
                return Err(CheckedRenameError::IncompleteOverloadUse(span));
            }
            Ok(true)
        }
    }
}

fn check_qualifier(
    reference: &CheckedReferenceBinding,
    old_name: &str,
    selected: &HashSet<DeclarationId>,
) -> Result<(), CheckedRenameError> {
    let (Receiver::Rooted(path), Some(failure)) =
        (&reference.site.receiver, reference.unproven_qualifier())
    else {
        return Ok(());
    };
    // Successful prefixes are known type/namespace roles. At the failed
    // segment, some refusals still rule out the selected member; no later
    // segment in that path has been proved unrelated.
    let (failed, unknown) = match failure {
        QualifiedTypeFailure::AmbiguousTypeMember {
            segment,
            candidates,
            ..
        } => {
            if candidates.iter().any(|candidate| {
                matches!(candidate, QualifiedTypeCandidate::Member { id, .. } if selected.contains(id))
            }) {
                return Err(CheckedRenameError::AmbiguousUse {
                    span: *segment,
                    reason: Ambiguity::CompetingMember,
                    candidates: candidates.iter().map(QualifiedTypeCandidate::span).collect(),
                });
            }
            (*segment, false)
        }
        QualifiedTypeFailure::Ambiguous { segment, .. }
        | QualifiedTypeFailure::UnsupportedArity { segment } => (*segment, false),
        QualifiedTypeFailure::MissingType { segment } => (*segment, true),
    };
    let unresolved = path.segments.iter().skip(1).find(|name| {
        (name.span.start > failed.start || (unknown && name.span == failed))
            && comparison_identifier(&name.text) == old_name
    });
    unresolved.map_or(Ok(()), |name| {
        Err(CheckedRenameError::UnresolvedUse(name.span))
    })
}

fn checked_edits(
    source: &str,
    region: Span,
    mut observations: Vec<Observation<'_>>,
    replacement: &str,
) -> Result<Vec<TextEdit>, CheckedRenameError> {
    observations.sort_unstable_by_key(|item| (item.name.span.start, item.name.span.end));
    let mut previous: Option<Observation<'_>> = None;
    let mut edits = Vec::with_capacity(observations.len());
    for item in observations {
        let span = item.name.span;
        if span.start >= span.end
            || span.start < region.start
            || span.end > region.end
            || source.get(span.start..span.end) != Some(item.name.text.as_str())
        {
            return Err(CheckedRenameError::InvalidEdit(span));
        }
        // `int@x` has a token boundary at `@`; replacing only that token with
        // `count` would produce `intcount`. The plan may not rewrite trivia.
        if item.name.text.starts_with('@')
            && span.start > region.start
            && (source.as_bytes()[span.start - 1].is_ascii_alphanumeric()
                || source.as_bytes()[span.start - 1] == b'_')
        {
            return Err(CheckedRenameError::WouldMergeTokens(span));
        }
        if let Some(prior) = previous {
            if prior == item {
                continue;
            }
            if span.start < prior.name.span.end {
                return Err(CheckedRenameError::OverlappingEdits {
                    first: prior.name.span,
                    second: span,
                });
            }
        }
        edits.push(TextEdit::replace(span, replacement));
        previous = Some(item);
    }
    Ok(edits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::edit::apply_edits;

    fn field(start: usize, end: usize) -> RenameTarget {
        RenameTarget::Field(Span::new(start, end))
    }

    fn method(start: usize, end: usize) -> RenameTarget {
        RenameTarget::Method(Span::new(start, end))
    }

    fn group(start: usize, end: usize) -> RenameTarget {
        RenameTarget::OverloadGroup(Span::new(start, end))
    }

    fn plan(
        source: &str,
        target: RenameTarget,
        replacement: &str,
    ) -> Result<CheckedRenamePlan, CheckedRenameError> {
        plan_member_rename(source, Span::new(0, source.len()), target, replacement)
    }

    fn edits<const N: usize>(spans: [(usize, usize); N], replacement: &str) -> [TextEdit; N] {
        spans.map(|(start, end)| TextEdit::replace(Span::new(start, end), replacement))
    }

    #[test]
    fn field_plan_replaces_raw_escaped_tokens_at_absolute_offsets() {
        let source = "π|class C{int @x;void F(){x=this.@x;}}|";
        let region = Span::new(3, 39);

        let plan = plan_member_rename(source, region, field(15, 17), "count").unwrap();

        assert_eq!(plan.region, region);
        assert_eq!(plan.edits, edits([(15, 17), (27, 28), (34, 36)], "count"));
        assert_eq!(
            apply_edits(source, &plan.edits),
            Ok("π|class C{int count;void F(){count=this.count;}}|".to_owned())
        );
    }

    #[test]
    fn single_method_plan_edits_only_the_selected_count() {
        let source = "class C{void F(){}void F(int a){}void G(int v){F();this.F(v);}}";

        let plan = plan(source, method(13, 14), "Renamed").unwrap();

        assert_eq!(plan.edits, edits([(13, 14), (47, 48)], "Renamed"));
    }

    const SAME_COUNT: &str = "class C{void F(int a){}void F(bool b){}void G(int v){F(v);}}";

    #[test]
    fn explicit_group_plan_edits_both_same_count_methods_and_the_witnessed_call() {
        let result = plan(SAME_COUNT, group(28, 29), "Renamed").unwrap();

        assert_eq!(
            result.edits,
            edits([(13, 14), (28, 29), (53, 54)], "Renamed")
        );
    }

    #[test]
    fn single_method_plan_refuses_either_member_of_a_same_count_call() {
        for target in [method(13, 14), method(28, 29)] {
            let result = plan(SAME_COUNT, target, "Renamed");

            assert_eq!(
                result,
                Err(CheckedRenameError::AmbiguousUse {
                    span: Span::new(53, 54),
                    reason: Ambiguity::Overload,
                    candidates: vec![Span::new(13, 14), Span::new(28, 29)],
                }),
                "{target:?}"
            );
        }
    }

    #[test]
    fn explicit_group_plan_refuses_calls_without_an_exact_applicability_witness() {
        for (source, span) in [
            (
                "class C{void F(int a){}void F(bool b){}void G(global::C p){F(p);}}",
                Span::new(59, 60),
            ),
            (
                "class C{void F(int a){}void F(bool b){}void G(int v){F(u);}}",
                Span::new(53, 54),
            ),
        ] {
            let result = plan(source, group(13, 14), "Renamed");

            assert_eq!(
                result,
                Err(CheckedRenameError::UnresolvedUse(span)),
                "{source}"
            );
        }
    }

    #[test]
    fn field_plan_leaves_known_other_declarations_and_roles_unchanged() {
        let source = "namespace x{class A{public static int x;}}class C{int x;void F(int x){x=this.x;}void G(global::x.A a){x=global::x.A.x;}}class G<x>{}";

        let plan = plan(source, field(54, 55), "count").unwrap();

        assert_eq!(plan.edits, edits([(54, 55), (77, 78), (102, 103)], "count"));
    }

    #[test]
    fn field_plan_leaves_bound_local_uses_unchanged() {
        let source = "class C{int x;void F(){int x;x=x;this.x=this.x;}}";

        let plan = plan(source, field(12, 13), "count").unwrap();

        assert_eq!(plan.edits, edits([(12, 13), (38, 39), (45, 46)], "count"));
    }

    #[test]
    fn field_plan_refuses_matching_uses_with_a_lexical_declaration_space_conflict() {
        let source = "class C{int x;void F(int x){int x;x=x;this.x=x;}}";

        let result = plan(source, field(12, 13), "count");

        assert_eq!(
            result,
            Err(CheckedRenameError::AmbiguousUse {
                span: Span::new(34, 35),
                reason: Ambiguity::DeclarationSpaceConflict,
                candidates: vec![Span::new(25, 26), Span::new(32, 33)],
            })
        );
    }

    #[test]
    fn field_plan_uses_a_bound_argument_as_an_independent_read() {
        let source = "class C{int x;void F(int a){F(x);}}";

        let plan = plan(source, field(12, 13), "count").unwrap();

        assert_eq!(plan.edits, edits([(12, 13), (30, 31)], "count"));
    }

    #[test]
    fn explicit_group_plan_does_not_include_unrelated_owners() {
        let source = "class A{void F(){}void F(int a){F();}}class C{void F(){F();}}";

        let plan = plan(source, group(13, 14), "Renamed").unwrap();

        assert_eq!(plan.edits, edits([(13, 14), (23, 24), (32, 33)], "Renamed"));
    }

    #[test]
    fn single_method_plan_leaves_an_ambiguity_with_only_unrelated_candidates_unchanged() {
        let source =
            "class A{void F(){}void G(){F();}}class C{void F(int a){}void F(bool b){}void G(int p){F(p);}}";

        let plan = plan(source, method(13, 14), "Renamed").unwrap();

        assert_eq!(plan.edits, edits([(13, 14), (27, 28)], "Renamed"));
    }

    #[test]
    fn plan_refuses_replacement_in_every_identifier_role_even_without_target_uses() {
        for (source, target, replacement, collision) in [
            (
                "class C{int x;void F(int y){x=y;}}",
                field(12, 13),
                "y",
                (25, 26),
            ),
            (
                "class C{int x;void M(){x=x;int count;}}",
                field(12, 13),
                "count",
                (31, 36),
            ),
            (
                "class C{int x;}class G<count>{}",
                field(12, 13),
                "count",
                (23, 28),
            ),
            (
                "namespace count{class N{}}class C{int x;}",
                field(38, 39),
                "count",
                (10, 15),
            ),
            (
                "class C{int x;}class count{}",
                field(12, 13),
                "count",
                (21, 26),
            ),
            (
                "class C{int x;}class P{void G(){G(count);}}",
                field(12, 13),
                "count",
                (34, 39),
            ),
            (
                "class C{int x;void F(){x=x;}void G(){@y=@y;}}",
                field(12, 13),
                "y",
                (37, 39),
            ),
            (
                "class C{int x;}class B{int count;}class D:B{}",
                field(12, 13),
                "count",
                (27, 32),
            ),
            (
                "class B{public int count;}class D:B{int x;}",
                field(40, 41),
                "count",
                (19, 24),
            ),
        ] {
            let result = plan(source, target, replacement);

            assert_eq!(
                result,
                Err(CheckedRenameError::ReplacementInUse(Span::new(
                    collision.0,
                    collision.1
                ))),
                "{source}"
            );
        }
    }

    #[test]
    fn field_plan_refuses_a_matching_unknown_receiver_member_at_its_own_span() {
        let source = "class C{int x;void M(int z){u.x=z;}}";

        let result = plan(source, field(12, 13), "count");

        assert_eq!(
            result,
            Err(CheckedRenameError::UnresolvedUse(Span::new(30, 31)))
        );
    }

    #[test]
    fn field_plan_refuses_an_unresolved_matching_use_in_another_type() {
        let source = "class C{int x;}class G{void F(){x=x;}}";

        let result = plan(source, field(12, 13), "count");

        assert_eq!(
            result,
            Err(CheckedRenameError::UnresolvedUse(Span::new(32, 33)))
        );
    }

    #[test]
    fn field_plan_allows_a_generic_type_elsewhere_but_not_in_the_selected_lineage() {
        for (source, target, expected) in [
            (
                "class C{int x;}class G<T>{}",
                field(12, 13),
                Ok(edits([(12, 13)], "count").into()),
            ),
            (
                "class G<T>{int x;}",
                field(15, 16),
                Err(CheckedRenameError::GenericOwner(Span::new(6, 7))),
            ),
            (
                "class G<T>{class C{int x;}}",
                field(23, 24),
                Err(CheckedRenameError::GenericOwner(Span::new(6, 7))),
            ),
        ] {
            let result = plan(source, target, "count").map(|plan| plan.edits);

            assert_eq!(result, expected, "{source}");
        }
    }

    #[test]
    fn field_plan_refuses_a_generic_method_elsewhere_at_the_actual_opening_angle() {
        let source = "class C{int x;}class G{void F /*<*/ <T>(){}}";

        let result = plan(source, field(12, 13), "count");

        assert_eq!(
            result,
            Err(CheckedRenameError::GenericMethod(Span::new(36, 37)))
        );
    }

    #[test]
    fn field_plan_refuses_incomplete_or_cyclic_bases_elsewhere_without_uses() {
        for (source, error) in [
            (
                "class C{int x;}class A:Missing{}",
                RenameBaseFailure::Incomplete {
                    span: Span::new(23, 30),
                    from: 1,
                },
            ),
            (
                "class C{int x;}class A:B{}class B:A{}",
                RenameBaseFailure::Cycle {
                    span: Span::new(34, 35),
                    from: 2,
                    to: 1,
                },
            ),
        ] {
            let result = plan(source, field(12, 13), "count");

            assert_eq!(result, Err(CheckedRenameError::Base(error)), "{source}");
        }
    }

    #[test]
    fn member_plan_refuses_same_name_members_in_either_inheritance_direction() {
        let fields = "class B{public int x;}class D:B{public int x;}";
        let methods = "class B{public void F(){}}class D:B{void F(int a){}}";
        for (source, target, collision) in [
            (fields, field(19, 20), Span::new(43, 44)),
            (fields, field(43, 44), Span::new(19, 20)),
            (methods, group(20, 21), Span::new(41, 42)),
            (methods, method(41, 42), Span::new(20, 21)),
        ] {
            let result = plan(source, target, "Renamed");

            assert_eq!(
                result,
                Err(CheckedRenameError::InheritedNameConflict(collision))
            );
        }
    }

    #[test]
    fn field_plan_includes_proved_inherited_uses() {
        let source = "class B{public int x;}class D:B{void F(){x=base.x;}}";

        let plan = plan(source, field(19, 20), "count").unwrap();

        assert_eq!(plan.edits, edits([(19, 20), (41, 42), (48, 49)], "count"));
    }

    #[test]
    fn explicit_group_plan_includes_a_proved_inherited_call() {
        let source = "class B{public void F(int a){}public void F(bool b){}}class D:B{void G(int a){base.F(a);}}";

        let plan = plan(source, group(20, 21), "Renamed").unwrap();

        assert_eq!(plan.edits, edits([(20, 21), (42, 43), (83, 84)], "Renamed"));
    }

    #[test]
    fn member_plan_refuses_duplicates_and_incomplete_groups_even_without_uses() {
        let equal_signatures = "class C{void F(int a){}void @F(int b){}}";
        for (source, target, error) in [
            (
                "class C{int x;int @x;}",
                field(12, 13),
                CheckedRenameError::DuplicateDeclarations(vec![
                    Span::new(12, 13),
                    Span::new(18, 20),
                ]),
            ),
            (
                equal_signatures,
                method(13, 14),
                CheckedRenameError::DuplicateDeclarations(vec![
                    Span::new(13, 14),
                    Span::new(28, 30),
                ]),
            ),
            (
                equal_signatures,
                group(28, 30),
                CheckedRenameError::DuplicateDeclarations(vec![
                    Span::new(13, 14),
                    Span::new(28, 30),
                ]),
            ),
            (
                "class C{void F(){}int F;}",
                group(13, 14),
                CheckedRenameError::ConflictingMember(Span::new(22, 23)),
            ),
            (
                "class C{int x;void F(){x=x;}}class C{}",
                field(12, 13),
                CheckedRenameError::UnprovenOwner(QualifiedTypeFailure::Ambiguous {
                    segment: Span::new(6, 7),
                    candidates: vec![
                        QualifiedTypeCandidate::Type {
                            ordinal: 0,
                            span: Span::new(6, 7),
                        },
                        QualifiedTypeCandidate::Type {
                            ordinal: 1,
                            span: Span::new(35, 36),
                        },
                    ],
                }),
            ),
        ] {
            let result = plan(source, target, "Renamed");

            assert_eq!(result, Err(error), "{source}");
        }
    }

    #[test]
    fn field_plan_refuses_a_rooted_qualifier_if_its_member_role_cannot_be_ruled_out() {
        let source = "class A{public static int x;}class C{void F(){global::A.x.z=global::A.x.z;}}";

        let result = plan(source, field(26, 27), "count");

        assert_eq!(
            result,
            Err(CheckedRenameError::UnresolvedUse(Span::new(56, 57)))
        );
    }

    #[test]
    fn field_plan_refuses_wrong_role_or_nonexact_declaration_starts() {
        let source = "class C{int @x;void F(){@x=@x;}}";
        for target in [method(12, 14), field(13, 14), field(24, 26)] {
            let result = plan(source, target, "count");

            assert_eq!(result, Err(CheckedRenameError::TargetNotFound(target)));
        }
    }

    #[test]
    fn plan_refuses_invalid_replacement_and_unchanged_normalized_name() {
        let source = "class C{int @x;}";
        for replacement in [
            "",
            "@count",
            "é",
            "has space",
            "0name",
            "name.name",
            "class",
            "dynamic",
            "extension",
            "__arglist",
            "user__name",
            "_",
            r"\u0061",
        ] {
            let result = plan(source, field(12, 14), replacement);

            assert_eq!(
                result,
                Err(CheckedRenameError::InvalidReplacement),
                "{replacement:?}"
            );
        }

        let result = plan(source, field(12, 14), "x");

        assert_eq!(result, Err(CheckedRenameError::UnchangedName));
    }

    #[test]
    fn field_plan_supports_an_escaped_reserved_source_identifier() {
        let source = "class C{int @class;void F(){@class=@class;}}";

        let plan = plan(source, field(12, 18), "count").unwrap();

        assert_eq!(plan.edits, edits([(12, 18), (28, 34), (35, 41)], "count"));
    }

    #[test]
    fn plan_refuses_if_removing_an_escape_would_merge_token_boundaries() {
        let source = "class C{int@x;}";

        let result = plan(source, field(11, 13), "count");

        assert_eq!(
            result,
            Err(CheckedRenameError::WouldMergeTokens(Span::new(11, 13)))
        );
    }

    #[test]
    fn field_plan_refuses_source_spellings_outside_its_comparison_model() {
        for (source, span) in [
            ("class C{int café;}", Span::new(12, 17)),
            ("class C{int class;}", Span::new(12, 17)),
            ("class C{int @__arglist;}", Span::new(12, 22)),
        ] {
            let result = plan(source, RenameTarget::Field(span), "count");

            assert_eq!(
                result,
                Err(CheckedRenameError::UnsupportedIdentifier(span)),
                "{source}"
            );
        }
    }

    #[test]
    fn plan_does_not_ignore_region_or_coverage_refusals() {
        let source = "π|class C{int x;void F(){y++;}}|";
        for (region, error) in [
            (
                Span::new(1, 31),
                GenericCallableIndexError::InvalidRegion(Span::new(1, 31)),
            ),
            (
                Span::new(3, 31),
                GenericCallableIndexError::Rejected {
                    first_unsupported: Some(Span::new(27, 29)),
                    recovery: vec![],
                },
            ),
        ] {
            let result = plan_member_rename(source, region, field(15, 16), "count");

            assert_eq!(result, Err(CheckedRenameError::Index(error)));
        }
    }

    // Checked parser coverage rules these out; exercise the independent final
    // validation directly so a later report change cannot mask conflicting edits.
    fn observation(name: &CsName, site: usize) -> Observation<'_> {
        Observation {
            occurrence: Occurrence::Reference(site),
            name,
        }
    }

    fn name(text: &str, start: usize, end: usize) -> CsName {
        CsName {
            text: text.to_owned(),
            span: Span::new(start, end),
        }
    }

    #[test]
    fn materializing_edits_deduplicates_only_an_identical_observation() {
        let raw = name("@x", 3, 5);
        let site = observation(&raw, 0);

        let result = checked_edits("π @x", Span::new(3, 5), vec![site, site], "count");

        assert_eq!(result, Ok(edits([(3, 5)], "count").into()));
    }

    #[test]
    fn materializing_edits_rejects_distinct_overlapping_observations() {
        let raw = name("@x", 3, 5);
        let suffix = name("x", 4, 5);
        for second in [&raw, &suffix] {
            let result = checked_edits(
                "π @x",
                Span::new(3, 5),
                vec![observation(second, 1), observation(&raw, 0)],
                "count",
            );

            assert_eq!(
                result,
                Err(CheckedRenameError::OverlappingEdits {
                    first: Span::new(3, 5),
                    second: second.span,
                })
            );
        }
    }

    #[test]
    fn materializing_edits_rejects_invalid_spans_or_raw_text() {
        for (region, raw) in [
            (Span::new(4, 5), name("@x", 3, 5)),
            (Span::new(0, 5), name("x", 1, 2)),
            (Span::new(3, 5), name("@z", 3, 5)),
        ] {
            let result = checked_edits("π @x", region, vec![observation(&raw, 0)], "count");

            assert_eq!(result, Err(CheckedRenameError::InvalidEdit(raw.span)));
        }
    }
}
