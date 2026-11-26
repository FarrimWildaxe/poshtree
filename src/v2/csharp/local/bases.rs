//! Original base-clause evidence and lookup over parse-local type occurrences.

use super::{CsName, CsTypeKind, QualifiedTypeFailure, QualifiedTypeLookup, QualifiedTypeUse};
use super::{TypeHeaderIndex, TypeHeaderRow};
use crate::v2::span::Span;
use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq, Eq)]
pub(in crate::v2::csharp) struct BaseOccurrence {
    /// The exact declaring type row, not its comparison name.
    pub(in crate::v2::csharp) owner: usize,
    pub(in crate::v2::csharp) clause: BaseClause,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::v2::csharp) struct BaseClause {
    pub(in crate::v2::csharp) colon: Span,
    pub(in crate::v2::csharp) path: BasePath,
    pub(in crate::v2::csharp) body_open: Span,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::v2::csharp) enum BasePath {
    Bare(CsName),
    Rooted {
        path: QualifiedTypeUse,
        separator: Span,
        dots: Vec<Span>,
    },
}

impl BasePath {
    fn span(&self) -> Span {
        match self {
            Self::Bare(name) => name.span,
            Self::Rooted { path, .. } => path.span,
        }
    }

    fn resolve(
        &self,
        types: &TypeHeaderIndex,
        lookup: &QualifiedTypeLookup<'_>,
        owner: &TypeHeaderRow,
    ) -> Result<usize, IncompleteReason> {
        let target = match self {
            Self::Bare(name) => {
                let proven_owner = lookup
                    .resolve_global_type(&owner.name)
                    .map_err(IncompleteReason::BareOwner)?;
                debug_assert_eq!(proven_owner, owner.ordinal);
                lookup.resolve_global_type(name)
            }
            Self::Rooted { path, .. } => lookup.resolve_checked(path).map(|bound| bound.terminal),
        }
        .map_err(IncompleteReason::Target)?;
        let row = &types.rows[target];
        if row.kind != CsTypeKind::Class {
            return Err(IncompleteReason::WrongKind {
                target,
                kind: row.kind,
            });
        }
        if let Some(blocked_by) = types.inaccessible_type_link(owner.ordinal, target) {
            return Err(IncompleteReason::InaccessibleTarget { target, blocked_by });
        }
        Ok(target)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum IncompleteReason {
    BareOwner(QualifiedTypeFailure),
    Target(QualifiedTypeFailure),
    WrongKind { target: usize, kind: CsTypeKind },
    InaccessibleTarget { target: usize, blocked_by: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum BaseLookupError {
    BaseCycle {
        span: Span,
        from: usize,
        to: usize,
    },
    IncompleteBase {
        span: Span,
        from: usize,
        reason: IncompleteReason,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct BaseIndex {
    occurrences: Vec<BaseOccurrence>,
    pub(super) graph: BaseGraph,
}

impl BaseIndex {
    pub(super) fn new(
        types: &TypeHeaderIndex,
        lookup: &QualifiedTypeLookup<'_>,
        occurrences: Vec<BaseOccurrence>,
    ) -> Self {
        let mut outgoing = Vec::new();
        outgoing.resize_with(types.rows.len(), Vec::new);
        for occurrence in &occurrences {
            let owner = occurrence.owner;
            let path = &occurrence.clause.path;
            outgoing[owner].push(Edge {
                span: path.span(),
                target: path.resolve(types, lookup, &types.rows[owner]),
            });
        }
        Self {
            occurrences,
            graph: BaseGraph { outgoing },
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct BaseGraph {
    // The graph policy handles shared ancestry and multiple outgoing edges;
    // checked source syntax supplies at most one class edge per owner.
    outgoing: Vec<Vec<Edge>>,
}

#[derive(Debug, PartialEq, Eq)]
struct Edge {
    span: Span,
    target: Result<usize, IncompleteReason>,
}

#[derive(Clone, Copy)]
enum Visit {
    Active,
    Finished,
}

impl BaseGraph {
    fn validate_reachable(&self, start: usize) -> Result<(), BaseLookupError> {
        let mut states = HashMap::from([(start, Visit::Active)]);
        let mut stack = vec![(start, self.outgoing[start].iter())];
        while let Some((from, edges)) = stack.last_mut() {
            let Some(edge) = edges.next() else {
                states.insert(*from, Visit::Finished);
                stack.pop();
                continue;
            };
            let &to = edge
                .target
                .as_ref()
                .map_err(|reason| BaseLookupError::IncompleteBase {
                    span: edge.span,
                    from: *from,
                    reason: reason.clone(),
                })?;
            match states.get(&to) {
                Some(Visit::Active) => {
                    return Err(BaseLookupError::BaseCycle {
                        span: edge.span,
                        from: *from,
                        to,
                    });
                }
                Some(Visit::Finished) => {}
                None => {
                    states.insert(to, Visit::Active);
                    stack.push((to, self.outgoing[to].iter()));
                }
            }
        }
        Ok(())
    }

    /// An inherited result is only useful after every reachable branch has
    /// proved complete. With that proof, first discovery is the shortest
    /// distance, and multiple paths to one row visit that row just once.
    pub(super) fn nearest_ancestors<T, I: IntoIterator<Item = T>>(
        &self,
        start: usize,
        mut candidates: impl FnMut(usize) -> I,
    ) -> Result<Vec<T>, BaseLookupError> {
        self.validate_reachable(start)?;
        let mut seen = HashSet::from([start]);
        let mut level = vec![start];
        loop {
            let next: Vec<_> = level
                .into_iter()
                .flat_map(|owner| &self.outgoing[owner])
                .map(|edge| *edge.target.as_ref().expect("validated reachable edge"))
                .filter(|target| seen.insert(*target))
                .collect();
            if next.is_empty() {
                return Ok(Vec::new());
            }
            let matches: Vec<_> = next.iter().copied().flat_map(&mut candidates).collect();
            if !matches.is_empty() {
                return Ok(matches);
            }
            level = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::binding::{
        bind_checked, Ambiguity, BindingOutcome, DeclarationId, Unresolved,
    };
    use super::super::{index_checked, QualifiedTypeCandidate, QualifiedTypeFailure::MissingType};
    use super::*;

    fn name(text: &str, start: usize, end: usize) -> CsName {
        CsName {
            text: text.to_owned(),
            span: Span::new(start, end),
        }
    }

    fn edge(target: usize, start: usize, end: usize) -> Edge {
        Edge {
            span: Span::new(start, end),
            target: Ok(target),
        }
    }

    #[test]
    fn checked_base_evidence_preserves_original_owner_path_and_boundaries() {
        let source = "π|namespace N{class @B{public int x;}class D:global /*root*/ :: N . @B{void F(){this.x=base.x;}}}|";

        let index = index_checked(source, Span::new(3, 98)).unwrap();

        assert_eq!(
            index.bases,
            BaseIndex {
                occurrences: vec![BaseOccurrence {
                    owner: 1,
                    clause: BaseClause {
                        colon: Span::new(45, 46),
                        path: BasePath::Rooted {
                            path: QualifiedTypeUse {
                                root: name("global", 46, 52),
                                segments: vec![name("N", 65, 66), name("@B", 69, 71)],
                                span: Span::new(46, 71),
                            },
                            separator: Span::new(62, 64),
                            dots: vec![Span::new(67, 68)],
                        },
                        body_open: Span::new(71, 72),
                    },
                }],
                graph: BaseGraph {
                    outgoing: vec![vec![], vec![edge(0, 46, 71)]],
                },
            }
        );
    }

    #[test]
    fn checked_base_evidence_keeps_ambiguous_and_wrong_kind_targets_incomplete() {
        use IncompleteReason::{BareOwner, Target, WrongKind};
        use QualifiedTypeCandidate::Type;
        use QualifiedTypeFailure::Ambiguous;

        for (source, from, span, reason) in [
            (
                "class B{public int x;}class D:global::N.B{void F(){this.x=base.x;}}",
                1,
                Span::new(30, 41),
                Target(MissingType {
                    segment: Span::new(38, 39),
                }),
            ),
            (
                "class B{public int x;}class @B{}class D:B{void F(){this.x=base.x;}}",
                2,
                Span::new(40, 41),
                Target(Ambiguous {
                    segment: Span::new(40, 41),
                    candidates: vec![
                        Type {
                            ordinal: 0,
                            span: Span::new(6, 7),
                        },
                        Type {
                            ordinal: 1,
                            span: Span::new(28, 30),
                        },
                    ],
                }),
            ),
            (
                "interface B{}class D:global::B{void F(){this.x=base.x;}}",
                1,
                Span::new(21, 30),
                WrongKind {
                    target: 0,
                    kind: CsTypeKind::Interface,
                },
            ),
            (
                "class D:B{void F(){this.x=base.x;}}class @D{}",
                0,
                Span::new(8, 9),
                BareOwner(Ambiguous {
                    segment: Span::new(6, 7),
                    candidates: vec![
                        Type {
                            ordinal: 0,
                            span: Span::new(6, 7),
                        },
                        Type {
                            ordinal: 1,
                            span: Span::new(41, 43),
                        },
                    ],
                }),
            ),
        ] {
            let index = index_checked(source, Span::new(0, source.len())).unwrap();

            let outcome =
                BindingOutcome::Unresolved(Unresolved::Base(BaseLookupError::IncompleteBase {
                    span,
                    from,
                    reason,
                }));
            assert_eq!(
                index
                    .references
                    .iter()
                    .map(|site| &site.outcome)
                    .collect::<Vec<_>>(),
                [&outcome, &outcome],
                "{source}",
            );
        }
    }

    #[test]
    fn checked_rooted_base_edges_attach_to_exact_same_named_owner_occurrences() {
        let source = "class A{}class B{}class D:global::A{}class @D:global::B{}";

        let index = index_checked(source, Span::new(0, source.len())).unwrap();

        assert_eq!(
            index.bases.graph,
            BaseGraph {
                outgoing: vec![vec![], vec![], vec![edge(0, 26, 35)], vec![edge(1, 46, 55)]],
            }
        );
    }

    fn diamond() -> BaseGraph {
        BaseGraph {
            outgoing: vec![
                vec![edge(1, 1, 2), edge(2, 3, 4)],
                vec![edge(3, 5, 6)],
                vec![edge(3, 7, 8)],
                vec![],
            ],
        }
    }

    #[test]
    fn base_graph_visits_a_shared_diamond_ancestor_once_without_reporting_a_cycle() {
        let graph = diamond();

        let candidates = graph.nearest_ancestors(0, |owner| (owner == 3).then_some(owner));

        assert_eq!(candidates, Ok(vec![3]));
    }

    #[test]
    fn base_graph_rejects_a_diamond_with_a_reachable_back_edge() {
        let mut graph = diamond();
        graph.outgoing[3].push(edge(0, 9, 10));

        let candidates = graph.nearest_ancestors(0, Some);

        assert_eq!(
            candidates,
            Err(BaseLookupError::BaseCycle {
                span: Span::new(9, 10),
                from: 3,
                to: 0,
            })
        );
    }

    #[test]
    fn base_graph_rejects_an_incomplete_branch_even_when_another_has_a_candidate() {
        let mut graph = diamond();
        let missing = IncompleteReason::Target(MissingType {
            segment: Span::new(15, 19),
        });
        graph.outgoing[2].push(Edge {
            span: Span::new(10, 19),
            target: Err(missing.clone()),
        });

        let candidates = graph.nearest_ancestors(0, |owner| (owner == 1).then_some(owner));

        assert_eq!(
            candidates,
            Err(BaseLookupError::IncompleteBase {
                span: Span::new(10, 19),
                from: 2,
                reason: missing,
            })
        );
    }

    #[test]
    fn base_graph_lookup_with_reversed_edges_returns_distinct_nearest_members_in_source_order() {
        // Multiple parents are a graph fixture only; the classes themselves
        // have no base clauses. Use real declaration rows to exercise ordering.
        let source = "class R{void F(){base.x=base.x;}}class A{public int x;}class B{public int x;}class C{public int x;}";
        let index = index_checked(source, Span::new(0, source.len())).unwrap();
        let types = &index.lexical.headers.types;
        let sites = index.references.into_iter().map(|site| site.site).collect();
        let graph = BaseGraph {
            outgoing: vec![
                vec![edge(2, 1, 2), edge(1, 3, 4)],
                vec![edge(3, 5, 6)],
                vec![edge(3, 7, 8)],
                vec![],
            ],
        };

        let references = bind_checked(
            types,
            &index.lexical.headers.callables,
            &index.lexical.declarations,
            &QualifiedTypeLookup::new(types),
            &graph,
            sites,
        );

        let duplicate = BindingOutcome::Ambiguous(
            Ambiguity::DuplicateMember,
            vec![DeclarationId::Field(1, 0), DeclarationId::Field(2, 0)],
        );
        assert_eq!(
            references
                .iter()
                .map(|site| (site.site.name.span, &site.outcome))
                .collect::<Vec<_>>(),
            [
                (Span::new(22, 23), &duplicate),
                (Span::new(29, 30), &duplicate)
            ]
        );
    }

    #[test]
    fn base_graph_lookup_reaches_the_end_of_a_long_chain_without_truncation() {
        let graph = BaseGraph {
            outgoing: (1..=4096)
                .map(|target| {
                    if target < 4096 {
                        vec![edge(target, 0, 1)]
                    } else {
                        vec![]
                    }
                })
                .collect(),
        };

        let candidates = graph.nearest_ancestors(0, |owner| (owner == 4095).then_some(owner));

        assert_eq!(candidates, Ok(vec![4095]));
    }
}
