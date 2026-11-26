//! Test-only Add-Type analysis prototype.
//!
//! Strict argument selection and raw literal provenance, not execution.
//! No result here validates PowerShell or C#, proves runtime absence, or
//! authorizes an edit. Straight-line transfers use a deliberately bounded
//! environment. Function findings describe isolated definition contexts, not
//! proof that a function is called or its body executes.
//! Conditional joins model all syntactic outcomes, without choosing a branch;
//! ordinary findings inside those regions remain discovery-only. Opt-in bounded
//! hypotheses describe known text at conditional sites, never execution.
//! Loops invalidate summarized writes without executing any iteration or header.
//! Conditional expressions summarize potential effects, never result values.
//! Try regions union body effects without exception routing or finally restores.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::ast::{Node, NodeKind, StringKind};
use super::lexer::lex;
use super::parser::{body_span, is_constant_string, parse_tokens, string_inner, ParseError};
use super::span::Span;
use super::tokens::{LexError, Token, TokenKind};
use super::variables::{AUTOMATIC_VARIABLES, PREFERENCE_VARIABLES};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct CoreReport {
    confirmed: Vec<Confirmed>,
    unresolved: Vec<Unresolved>,
    coverage: Vec<CoverageGap>,
    hypotheses: Vec<HypothesisConditionalSite>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnalysisOptions {
    include_hypotheses: bool,
    hypothesis_limits: HypothesisLimits,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            include_hypotheses: false,
            hypothesis_limits: HypothesisLimits {
                candidates: 128,
                owned_text_bytes: 1 << 20,
                provenance_context_records: 4096,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HypothesisLimits {
    candidates: usize,
    owned_text_bytes: usize,
    provenance_context_records: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HypothesisLimit {
    CandidateCount,
    OwnedTextBytes,
    ProvenanceContextRecords,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HypothesisReason {
    ConditionalSite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HypothesisConditionalSite {
    site: AddTypeSite,
    reason: HypothesisReason,
    raw_body: String,
    literal_origins: Vec<LiteralOrigin>,
    assignment_origins: Vec<AssignmentOrigin>,
    reference_origins: Vec<ReferenceOrigin>,
    context: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Confirmed {
    site: AddTypeSite,
    raw_body: String,
    literal_origins: Vec<LiteralOrigin>,
    assignment_origins: Vec<AssignmentOrigin>,
    reference_origins: Vec<ReferenceOrigin>,
    context: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiteralOrigin {
    literal_span: Span,
    body_span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AssignmentOrigin {
    assignment_span: Span,
    target_span: Span,
    value_span: Span,
    raw_target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReferenceOrigin {
    span: Span,
    raw_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AddTypeSite {
    command_span: Span,
    name_span: Span,
    // None means strict selection failed, not that the parser lost an argument.
    argument_span: Option<Span>,
    parameter_span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Unresolved {
    site: AddTypeSite,
    reason: DiscoveryReason,
    context: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CoverageGap {
    span: Span,
    reason: DiscoveryReason,
    context: Context,
}

/// All spans refer to the original input, including its BOM and UTF-8 bytes.
/// A path records lexical nesting, not an invocation or a scope binding.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Context {
    path: Vec<ContextFrame>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContextFrame {
    kind: ContextKind,
    span: Span,
    status: ContextStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ContextKind {
    Script,
    Function(String),
    Workflow(String),
    ScriptBlock,
    KeywordBlock(String),
    Class(String),
    ClassMember(String),
    Enum(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextStatus {
    DiscoveryOnly,
    // Includes separately analyzed function definitions, never an invocation.
    Deferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscoveryReason {
    PendingTransferAnalysis,
    InvalidArguments(ArgumentReason),
    UnknownValue(ValueReason),
    PendingFunctionAnalysis,
    UnsupportedRegion(RegionKind),
    DeferredContext,
    FragmentDiagnosticsUnavailable,
    HypothesesTruncated(HypothesisLimit),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgumentReason {
    MissingArgument,
    ExtraElements,
    UnsupportedParameter,
    ColonParameter,
    MissingParameterArgument,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueReason {
    NonConstantString,
    VariableNotKnown,
    SimpleNonString,
    UnsupportedExpression,
    UnsupportedVariable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiteralValue {
    raw_body: String,
    literal_origins: Vec<LiteralOrigin>,
    assignment_origins: Vec<AssignmentOrigin>,
    reference_origins: Vec<ReferenceOrigin>,
}

// Absence is an unknown value, never a fact recovered from the parser's cache.
type Environment = BTreeMap<String, LiteralValue>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum WriteEffects {
    Keys(BTreeSet<String>),
    WholeEnvironment,
}

impl WriteEffects {
    fn invalidate(&self, environment: &mut Environment) {
        match self {
            Self::Keys(keys) => environment.retain(|key, _| !keys.contains(key)),
            Self::WholeEnvironment => environment.clear(),
        }
    }
}

fn is_conditional_expression(node: &Node) -> bool {
    match &unparen(node).kind {
        NodeKind::Ternary { .. } => true,
        NodeKind::Binary { op, .. } => {
            op.eq_ignore_ascii_case("-and") || op.eq_ignore_ascii_case("-or") || op == "??"
        }
        _ => false,
    }
}

fn summarize_conditional_effects(node: &Node, tokens: &[Token]) -> WriteEffects {
    use NodeKind::*;

    let mut keys = BTreeSet::new();
    let mut pending = vec![node];
    // Each frame describes potential effects, never an execution or a value.
    // All operands/arms contribute, even with literal truth/null conditions.
    while let Some(node) = pending.pop() {
        match &node.kind {
            Paren(inner) => pending.push(inner),
            Ternary {
                condition,
                if_true,
                if_false,
            } => pending.extend([if_false.as_ref(), if_true.as_ref(), condition.as_ref()]),
            Binary { left, right, .. } if is_conditional_expression(node) => {
                pending.extend([right.as_ref(), left.as_ref()]);
            }
            Assignment { target, value, .. } => {
                let Some(key) = target_key(target) else {
                    return WriteEffects::WholeEnvironment;
                };
                keys.insert(key);
                pending.push(value);
            }
            Unary { op, operand } | PostfixUnary { op, operand } if op == "++" || op == "--" => {
                let Some(key) = target_key(operand) else {
                    return WriteEffects::WholeEnvironment;
                };
                keys.insert(key);
            }
            Function { .. } => {}
            ScriptBlockExpression(_)
                if first_token(node, tokens).is_some_and(|t| t.kind == TokenKind::LBrace) => {}
            _ if is_effect_free_value(node) => {}
            // Unsupported syntax dominates known keys. In particular, do not
            // infer effects by unwrapping Script/SubExpression or target leaves.
            _ => return WriteEffects::WholeEnvironment,
        }
    }
    WriteEffects::Keys(keys)
}

struct CatchToken<'a> {
    kind: TokenKind,
    span: Span,
    value: &'a str,
}

trait CatchTokenView {
    fn token_count(&self) -> usize;
    fn token_at(&self, index: usize) -> Option<CatchToken<'_>>;
}

impl CatchTokenView for [Token] {
    fn token_count(&self) -> usize {
        self.len()
    }

    fn token_at(&self, index: usize) -> Option<CatchToken<'_>> {
        let token = self.get(index)?;
        Some(CatchToken {
            kind: token.kind,
            span: token.span,
            // The legacy predicate uses the stored value, not the source span.
            value: token.value.as_str(),
        })
    }
}

fn is_braced_script(node: &Node, tokens: &(impl CatchTokenView + ?Sized)) -> bool {
    let Some(last) = node.range.end.checked_sub(1) else {
        return false;
    };
    matches!(node.kind, NodeKind::Script(_))
        && node.range.first < last
        && node.range.end <= tokens.token_count()
        && tokens.token_at(node.range.first).is_some_and(|token| {
            token.kind == TokenKind::LBrace && token.span.start == node.span.start
        })
        && tokens
            .token_at(last)
            .is_some_and(|token| token.kind == TokenKind::RBrace && token.span.end == node.span.end)
}

fn is_unfiltered_catch(node: &Node, tokens: &(impl CatchTokenView + ?Sized)) -> bool {
    let NodeKind::Catch { body } = &node.kind else {
        return false;
    };
    if node.range.first >= node.range.end || node.range.end > tokens.token_count() {
        return false;
    }
    // Filters are absent from this AST. Only adjacency in the original token
    // indices certifies an empty header; trivia is already attached to tokens.
    tokens.token_at(node.range.first).is_some_and(|token| {
        token.kind == TokenKind::Keyword
            && token.value.eq_ignore_ascii_case("catch")
            && token.span.start == node.span.start
    }) && body.range.first == node.range.first + 1
        && body.range.end == node.range.end
        && body.span.end == node.span.end
        && is_braced_script(body, tokens)
}

fn summarize_control_effects(node: &Node, tokens: &[Token]) -> WriteEffects {
    use NodeKind::*;

    let mut keys = BTreeSet::new();
    let mut pending = vec![node];
    // These are effect roles, not discovery edges or an execution order.
    // Nested controls share this stack; no iterations or paths are enumerated.
    while let Some(node) = pending.pop() {
        match &node.kind {
            Script(statements) => pending.extend(statements.iter().rev()),
            While { condition, body }
            | DoWhile {
                condition, body, ..
            } => pending.extend([body.as_ref(), condition.as_ref()]),
            For {
                init,
                condition,
                update,
                body,
            } => {
                pending.push(body);
                pending.extend(update.as_deref());
                pending.extend(condition.as_deref());
                pending.extend(init.as_deref());
            }
            ForEach {
                variable,
                iterable,
                body,
            } => {
                // The iteration target is a write, never an ordinary read leaf.
                let Some(key) = target_key(variable) else {
                    return WriteEffects::WholeEnvironment;
                };
                keys.insert(key);
                pending.extend([body.as_ref(), iterable.as_ref()]);
            }
            If {
                conditions,
                blocks,
                else_block,
            } => {
                pending.extend(else_block.as_deref());
                pending.extend(blocks.iter().rev());
                pending.extend(conditions.iter().rev());
            }
            Try {
                body,
                catches,
                finally_block,
            } => {
                if (catches.is_empty() && finally_block.is_none())
                    || !is_braced_script(body, tokens)
                    || finally_block
                        .as_deref()
                        .is_some_and(|body| !is_braced_script(body, tokens))
                {
                    return WriteEffects::WholeEnvironment;
                }
                // All bodies contribute potential writes. This does not select
                // a handler or install the last assignment from finally.
                pending.extend(finally_block.as_deref());
                pending.extend(catches.iter().rev());
                pending.push(body);
            }
            Catch { body } => {
                if !is_unfiltered_catch(node, tokens) {
                    return WriteEffects::WholeEnvironment;
                }
                pending.push(body);
            }
            _ if is_conditional_expression(node)
                || matches!(&node.kind, Assignment { op, .. } if op == "??=") =>
            {
                match summarize_conditional_effects(node, tokens) {
                    WriteEffects::Keys(conditional_keys) => keys.extend(conditional_keys),
                    WriteEffects::WholeEnvironment => return WriteEffects::WholeEnvironment,
                }
            }
            Assignment { target, value, .. } => {
                let Some(key) = target_key(target) else {
                    return WriteEffects::WholeEnvironment;
                };
                keys.insert(key);
                // A supported target does not make its RHS effect-free.
                pending.push(value);
            }
            Unary { op, operand } | PostfixUnary { op, operand } if op == "++" || op == "--" => {
                let Some(key) = target_key(operand) else {
                    return WriteEffects::WholeEnvironment;
                };
                keys.insert(key);
            }
            // Definitions do not execute in this environment, even when nested
            // in a control. Discovery still visits their sites and records context.
            Function { .. } => {}
            ScriptBlockExpression(_)
                if first_token(node, tokens).is_some_and(|t| t.kind == TokenKind::LBrace) => {}
            _ if is_effect_free_value(node) => {}
            // Commands (including Add-Type), provider reads, declarations,
            // interpolation and unmodeled expressions/control dominate all keys.
            // Never guess writes by walking leaves of unsupported syntax.
            _ => return WriteEffects::WholeEnvironment,
        }
    }
    WriteEffects::Keys(keys)
}

fn join_outcome(joined: &mut Option<Environment>, outcome: Environment) {
    let Some(known) = joined else {
        *joined = Some(outcome);
        return;
    };
    known.retain(|key, value| {
        let Some(other) = outcome
            .get(key)
            .filter(|other| other.raw_body == value.raw_body)
        else {
            return false;
        };
        for origin in &other.literal_origins {
            append_unique(&mut value.literal_origins, origin.clone());
        }
        for origin in &other.assignment_origins {
            append_unique(&mut value.assignment_origins, origin.clone());
        }
        for origin in &other.reference_origins {
            append_unique(&mut value.reference_origins, origin.clone());
        }
        true
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionKind {
    ControlFlow,
    Expression,
    OtherCommand,
    Invocation,
    Redirection,
    ParameterDeclaration,
    KeywordBlock,
    Workflow,
    Declaration,
    Import,
    FragmentError,
    VariableRead,
    UnsupportedWrite,
}

const MAX_SOURCE_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalysisResource {
    SourceBytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AnalysisError {
    ResourceLimit {
        resource: AnalysisResource,
        limit: usize,
    },
    Syntax {
        // Presentation order is lex, then parse; neither collection is sorted
        // or converted to the other diagnostic type.
        lex_errors: Vec<LexError>,
        parse_errors: Vec<ParseError>,
    },
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceLimit {
                resource: AnalysisResource::SourceBytes,
                limit,
            } => write!(f, "source byte limit exceeded ({limit} bytes)"),
            Self::Syntax { .. } => f.write_str("source has syntax errors"),
        }
    }
}

impl std::error::Error for AnalysisError {}

fn analyze_add_type(source: &str, options: AnalysisOptions) -> Result<CoreReport, AnalysisError> {
    analyze_add_type_with_source_limit(source, options, MAX_SOURCE_BYTES)
}

fn analyze_add_type_with_source_limit(
    source: &str,
    options: AnalysisOptions,
    max_source_bytes: usize,
) -> Result<CoreReport, AnalysisError> {
    // Count the original UTF-8 bytes, including any BOM, before starting analysis.
    if source.len() > max_source_bytes {
        return Err(AnalysisError::ResourceLimit {
            resource: AnalysisResource::SourceBytes,
            limit: max_source_bytes,
        });
    }

    let lexed = lex(source);
    let parsed = parse_tokens(source, lexed.tokens);
    if !lexed.errors.is_empty() || !parsed.errors.is_empty() {
        return Err(AnalysisError::Syntax {
            lex_errors: lexed.errors,
            parse_errors: parsed.errors,
        });
    }

    let context = Context {
        path: vec![ContextFrame {
            kind: ContextKind::Script,
            span: Span::new(0, source.len()),
            status: ContextStatus::DiscoveryOnly,
        }],
    };
    // OFF does not construct a collector or inspect candidate values.
    let mut collector = options
        .include_hypotheses
        .then(|| HypothesisCollector::new(options.hypothesis_limits));
    let mut report = transfer(&parsed.script, &parsed.tokens, context, &mut collector);
    if let Some(collector) = collector {
        report.hypotheses = collector.hypotheses;
    }
    Ok(report)
}

struct HypothesisCollector {
    limits: HypothesisLimits,
    owned_text_bytes: usize,
    provenance_context_records: usize,
    stopped: bool,
    hypotheses: Vec<HypothesisConditionalSite>,
}

impl HypothesisCollector {
    fn new(limits: HypothesisLimits) -> Self {
        Self {
            limits,
            owned_text_bytes: 0,
            provenance_context_records: 0,
            stopped: false,
            hypotheses: Vec::new(),
        }
    }

    fn observe(
        &mut self,
        node: &Node,
        tokens: &[Token],
        context: &Context,
        environment: &Environment,
    ) -> Option<CoverageGap> {
        if self.stopped {
            return None;
        }
        let NodeKind::Command { name, elements, .. } = &node.kind else {
            unreachable!("only command completion collects hypotheses");
        };
        if !is_add_type(name) {
            return None;
        }
        let mut site = AddTypeSite {
            command_span: node.span,
            name_span: name.span,
            argument_span: None,
            parameter_span: None,
        };
        let argument = select_argument(elements, tokens, &mut site).ok()?;
        // Deduplicate against admitted, bounded records, not an unbounded set of
        // discovered sites. Context is part of the site's identity.
        if self
            .hypotheses
            .iter()
            .any(|finding| finding.site == site && &finding.context == context)
        {
            return None;
        }
        let cost = hypothesis_cost(argument, environment, context)?;
        let admission = self.admit(cost);
        let (owned_text_bytes, provenance_context_records) = match admission {
            Ok(totals) => totals,
            Err(limit) => {
                self.stopped = true;
                return Some(CoverageGap {
                    span: node.span,
                    reason: DiscoveryReason::HypothesesTruncated(limit),
                    context: context.clone(),
                });
            }
        };
        // All candidate-owned copies, including the evaluator's environment
        // clone, occur only after full borrowed admission.
        let value = evaluate_value(argument, environment).ok()?;
        self.hypotheses.push(HypothesisConditionalSite {
            site,
            reason: HypothesisReason::ConditionalSite,
            raw_body: value.raw_body,
            literal_origins: value.literal_origins,
            assignment_origins: value.assignment_origins,
            reference_origins: value.reference_origins,
            context: context.clone(),
        });
        self.owned_text_bytes = owned_text_bytes;
        self.provenance_context_records = provenance_context_records;
        None
    }

    fn admit(&self, cost: HypothesisCost) -> Result<(usize, usize), HypothesisLimit> {
        // If several limits would be exceeded, report the first in this order:
        // candidate count, then owned text bytes, then provenance/context records.
        budget_total(self.hypotheses.len(), Some(1), self.limits.candidates)
            .ok_or(HypothesisLimit::CandidateCount)?;
        let text = budget_total(
            self.owned_text_bytes,
            cost.owned_text_bytes,
            self.limits.owned_text_bytes,
        )
        .ok_or(HypothesisLimit::OwnedTextBytes)?;
        let records = budget_total(
            self.provenance_context_records,
            cost.provenance_context_records,
            self.limits.provenance_context_records,
        )
        .ok_or(HypothesisLimit::ProvenanceContextRecords)?;
        Ok((text, records))
    }
}

struct HypothesisCost {
    // None means arithmetic overflow, not ineligibility or an unlimited budget.
    owned_text_bytes: Option<usize>,
    provenance_context_records: Option<usize>,
}

fn budget_total(used: usize, additional: Option<usize>, limit: usize) -> Option<usize> {
    used.checked_add(additional?)
        .filter(|total| *total <= limit)
}

fn hypothesis_cost(
    node: &Node,
    environment: &Environment,
    context: &Context,
) -> Option<HypothesisCost> {
    let node = unparen(node);
    // Only recognize the existing evaluator's literal/known-variable cases.
    // This measures borrowed data, without folding or constructing a value.
    let (mut text, mut records) = match &node.kind {
        NodeKind::StringLiteral { value, kind, .. } if is_constant_string(value, *kind) => {
            (Some(string_inner(value, *kind).len()), Some(1usize))
        }
        NodeKind::Variable(raw) => {
            // Environment keys already passed variable_key. Compare its ASCII
            // normalization without allocating a possibly large lookup key.
            let name = raw.strip_prefix('$')?;
            let value = environment
                .iter()
                .find_map(|(key, value)| key.eq_ignore_ascii_case(name).then_some(value))?;
            let mut text = Some(value.raw_body.len());
            for origin in &value.assignment_origins {
                text = text.and_then(|total| total.checked_add(origin.raw_target.len()));
            }
            for origin in &value.reference_origins {
                text = text.and_then(|total| total.checked_add(origin.raw_name.len()));
            }
            let mut records = value
                .literal_origins
                .len()
                .checked_add(value.assignment_origins.len())
                .and_then(|total| total.checked_add(value.reference_origins.len()));
            // Match append_unique's full-record equality, including the span.
            if !value
                .reference_origins
                .iter()
                .any(|origin| origin.span == node.span && origin.raw_name == *raw)
            {
                text = text.and_then(|total| total.checked_add(raw.len()));
                records = records.and_then(|total| total.checked_add(1));
            }
            (text, records)
        }
        _ => return None,
    };
    records = records.and_then(|total| total.checked_add(context.path.len()));
    for frame in &context.path {
        let name = match &frame.kind {
            ContextKind::Script | ContextKind::ScriptBlock => "",
            ContextKind::Function(name)
            | ContextKind::Workflow(name)
            | ContextKind::KeywordBlock(name)
            | ContextKind::Class(name)
            | ContextKind::ClassMember(name)
            | ContextKind::Enum(name) => name,
        };
        text = text.and_then(|total| total.checked_add(name.len()));
    }
    Some(HypothesisCost {
        owned_text_bytes: text,
        provenance_context_records: records,
    })
}

fn discover(
    script: &Node,
    tokens: &[Token],
    context: Context,
    inherited_reason: DiscoveryReason,
) -> CoreReport {
    use DiscoveryReason::*;
    use NodeKind::*;

    let mut report = CoreReport::default();
    // Discovery has no recursive calls, environment, or transfer effects.
    // Parser guards bound the tree; this stack also handles long flat chains.
    let mut pending = vec![(script, context, inherited_reason)];
    while let Some((node, context, inherited_reason)) = pending.pop() {
        let mut child_context = context.clone();
        let mut reason = inherited_reason;
        let mut gap = None;
        match &node.kind {
            Command {
                name,
                invocation,
                redirections,
                elements,
                ..
            } => {
                let region = if !redirections.is_empty() {
                    Some(RegionKind::Redirection)
                } else if *invocation {
                    Some(RegionKind::Invocation)
                } else if !is_add_type(name) {
                    Some(RegionKind::OtherCommand)
                } else {
                    None
                };
                if let Some(region) = region {
                    gap = Some(UnsupportedRegion(region));
                }
                reason = inherit_reason(reason, gap);
                if !invocation && is_add_type(name) {
                    let mut site = AddTypeSite {
                        command_span: node.span,
                        name_span: name.span,
                        argument_span: None,
                        parameter_span: None,
                    };
                    // Selection retains source spans, but discovery never
                    // evaluates a value or performs an environment transfer.
                    let _ = select_argument(elements, tokens, &mut site);
                    report.unresolved.push(Unresolved {
                        site,
                        reason,
                        context: context.clone(),
                    });
                }
            }
            Function { name, .. } => {
                let workflow = first_token(node, tokens).is_some_and(|t| t.value_eq_ci("workflow"));
                let kind = if workflow {
                    ContextKind::Workflow(name.clone())
                } else {
                    ContextKind::Function(name.clone())
                };
                push_context(&mut child_context, kind, node.span);
                gap = Some(if workflow {
                    UnsupportedRegion(RegionKind::Workflow)
                } else {
                    PendingFunctionAnalysis
                });
            }
            ScriptBlockExpression(_) => {
                let keyword = first_token(node, tokens).filter(|t| t.kind == TokenKind::Keyword);
                let kind = if let Some(token) = keyword {
                    gap = Some(UnsupportedRegion(RegionKind::KeywordBlock));
                    ContextKind::KeywordBlock(token.value.clone())
                } else {
                    gap = Some(DeferredContext);
                    ContextKind::ScriptBlock
                };
                push_context(&mut child_context, kind, node.span);
            }
            ClassDefinition { name, .. } => {
                push_context(
                    &mut child_context,
                    ContextKind::Class(name.clone()),
                    node.span,
                );
                gap = Some(UnsupportedRegion(RegionKind::Declaration));
            }
            EnumDefinition { name, .. } => {
                push_context(
                    &mut child_context,
                    ContextKind::Enum(name.clone()),
                    node.span,
                );
                gap = Some(UnsupportedRegion(RegionKind::Declaration));
            }
            ClassMember { name, .. } => {
                push_context(
                    &mut child_context,
                    ContextKind::ClassMember(name.clone()),
                    node.span,
                );
                gap = Some(DeferredContext);
            }
            StringLiteral {
                kind: StringKind::Double | StringKind::HereDouble,
                ..
            } => {
                // Even empty `parts` cannot certify complete fragment syntax
                // or absence of effects. Do not reparse or rebase fragments.
                gap = Some(FragmentDiagnosticsUnavailable);
            }
            ParamBlock(_) | Parameter { .. } => {
                gap = Some(UnsupportedRegion(RegionKind::ParameterDeclaration));
            }
            Pipeline(_)
            | PipelineChain { .. }
            | If { .. }
            | While { .. }
            | DoWhile { .. }
            | For { .. }
            | ForEach { .. }
            | Switch { .. }
            | Try { .. }
            | Catch { .. }
            | Flow { .. }
            | Labeled { .. }
            | Ternary { .. } => {
                gap = Some(UnsupportedRegion(RegionKind::ControlFlow));
            }
            Unary { op, operand } | PostfixUnary { op, operand }
                if (op == "++" || op == "--") && target_key(operand).is_none() =>
            {
                gap = Some(UnsupportedRegion(RegionKind::UnsupportedWrite));
            }
            Binary { .. }
            | Unary { .. }
            | PostfixUnary { .. }
            | Cast { .. }
            | MemberAccess { .. }
            | InvokeMember { .. }
            | Index { .. }
            | SubExpression(_)
            | Array(_)
            | ArrayLiteral(_)
            | Hashtable(_) => {
                gap = Some(UnsupportedRegion(RegionKind::Expression));
            }
            Redirection { .. } => gap = Some(UnsupportedRegion(RegionKind::Redirection)),
            Variable(raw) if variable_key(raw).is_none() => {
                gap = Some(UnsupportedRegion(RegionKind::VariableRead));
            }
            Using { .. } => gap = Some(UnsupportedRegion(RegionKind::Import)),
            // Reported diagnostics already returned Syntax at the entry.
            // Fragment Error nodes need a coverage gap, not invented errors.
            Error(_) => gap = Some(UnsupportedRegion(RegionKind::FragmentError)),
            Paren(_)
                if !matches!(
                    unparen(node).kind,
                    StringLiteral { .. }
                        | Variable(_)
                        | Number(_)
                        | TypeExpression(_)
                        | BareWord(_)
                ) =>
            {
                // Parentheses around executable syntax are not transparent
                // value wrappers; descendants remain discovery-only.
                gap = Some(UnsupportedRegion(RegionKind::Expression));
            }
            Assignment { target, value, op }
                if target_key(target).is_none()
                    || (op == "??=" && !is_effect_free_value(value)) =>
            {
                gap = Some(UnsupportedRegion(if target_key(target).is_none() {
                    RegionKind::UnsupportedWrite
                } else {
                    RegionKind::ControlFlow
                }));
            }
            Script(_)
            | Assignment { .. }
            | CommandParameter { .. }
            | Paren(_)
            | StringLiteral { .. }
            | Variable(_)
            | Number(_)
            | TypeExpression(_)
            | BareWord(_)
            | CSharpMemberDef(_) => {}
        }
        reason = inherit_reason(reason, gap);
        if let Some(gap_reason) = gap {
            report.coverage.push(CoverageGap {
                span: node.span,
                reason: gap_reason,
                context: child_context.clone(),
            });
        }
        for child in source_children(node).into_iter().rev() {
            pending.push((child, child_context.clone(), reason));
        }
    }
    report
}

fn variable_key(raw: &str) -> Option<String> {
    let name = raw.strip_prefix('$')?;
    if name.is_empty()
        || !name.chars().all(|c| c.is_alphanumeric() || c == '_')
        || name.bytes().all(|b| b.is_ascii_digit())
        || AUTOMATIC_VARIABLES
            .iter()
            .chain(PREFERENCE_VARIABLES)
            .any(|builtin| builtin.eq_ignore_ascii_case(name))
    {
        return None;
    }
    Some(name.to_ascii_lowercase())
}

fn target_key(node: &Node) -> Option<String> {
    match &node.kind {
        NodeKind::Variable(raw) => variable_key(raw),
        _ => None,
    }
}

fn is_provider_read(raw: &str) -> bool {
    // This only classifies unsupported access. It never creates a binding by
    // stripping braces or qualifiers.
    let spelling = raw.strip_prefix("${").and_then(|s| s.strip_suffix('}'));
    let spelling = spelling.unwrap_or(raw);
    spelling.split_once(':').is_some_and(|(qualifier, name)| {
        let qualifier = qualifier.trim_start_matches(['$', '@']);
        name.contains(':')
            || !["script", "global", "local", "private", "using"]
                .iter()
                .any(|scope| scope.eq_ignore_ascii_case(qualifier))
    })
}

fn append_unique<T: PartialEq>(origins: &mut Vec<T>, origin: T) {
    // Encounter order is deterministic; repeated self-copies cannot build a
    // recursive provenance graph or duplicate the same source step.
    if !origins.contains(&origin) {
        origins.push(origin);
    }
}

fn append_report(report: &mut CoreReport, other: CoreReport) {
    report.confirmed.extend(other.confirmed);
    report.unresolved.extend(other.unresolved);
    report.coverage.extend(other.coverage);
}

fn discovery_region(
    node: &Node,
    tokens: &[Token],
    context: &Context,
    region: RegionKind,
    environment: &mut Environment,
    report: Option<&mut CoreReport>,
) {
    environment.clear();
    if let Some(report) = report {
        append_report(
            report,
            discover(
                node,
                tokens,
                context.clone(),
                DiscoveryReason::UnsupportedRegion(region),
            ),
        );
    }
    // Discovery is deliberately unable to restore any values.
    environment.clear();
}

fn transfer_discovery(
    node: &Node,
    tokens: &[Token],
    context: Context,
    reason: DiscoveryReason,
    conditional: bool,
) -> CoreReport {
    // An outer conditional already discovered the complete source subtree.
    // Its silent transfers must neither duplicate sites nor confirm definitions.
    if conditional {
        CoreReport::default()
    } else {
        discover(node, tokens, context, reason)
    }
}

fn record_command(
    node: &Node,
    tokens: &[Token],
    context: &Context,
    environment: &Environment,
    report: &mut CoreReport,
) {
    let NodeKind::Command { name, elements, .. } = &node.kind else {
        unreachable!("only command completion records a command");
    };
    if !is_add_type(name) {
        report.coverage.push(CoverageGap {
            span: node.span,
            reason: DiscoveryReason::UnsupportedRegion(RegionKind::OtherCommand),
            context: context.clone(),
        });
        return;
    }
    let mut site = AddTypeSite {
        command_span: node.span,
        name_span: name.span,
        argument_span: None,
        parameter_span: None,
    };
    let value = select_argument(elements, tokens, &mut site)
        .map_err(DiscoveryReason::InvalidArguments)
        .and_then(|argument| {
            evaluate_value(argument, environment).map_err(DiscoveryReason::UnknownValue)
        });
    match value {
        Ok(value) => report.confirmed.push(Confirmed {
            site,
            raw_body: value.raw_body,
            literal_origins: value.literal_origins,
            assignment_origins: value.assignment_origins,
            reference_origins: value.reference_origins,
            context: context.clone(),
        }),
        Err(reason) => report.unresolved.push(Unresolved {
            site,
            reason,
            context: context.clone(),
        }),
    }
}

fn is_ordinary_function(node: &Node, tokens: &[Token]) -> bool {
    // `Function.filter == false` also represents workflow in this parser.
    first_token(node, tokens).is_some_and(|token| {
        token.kind == TokenKind::Keyword
            && (token.value_eq_ci("function") || token.value_eq_ci("filter"))
    })
}

fn analyze_function(
    node: &Node,
    tokens: &[Token],
    mut context: Context,
    collector: &mut Option<HypothesisCollector>,
) -> CoreReport {
    let NodeKind::Function {
        name,
        parameters,
        body,
        ..
    } = &node.kind
    else {
        unreachable!("only ordinary function definitions enter isolated analysis");
    };
    push_context(&mut context, ContextKind::Function(name.clone()), node.span);
    let mut report = CoreReport::default();
    for parameter in parameters {
        append_report(
            &mut report,
            discover(
                parameter,
                tokens,
                context.clone(),
                DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
            ),
        );
    }
    // No parent environment is passed in or returned. Parameters (including
    // literal defaults) and captures start unknown; defaults are discovery-only.
    append_report(&mut report, transfer(body, tokens, context, collector));
    report
}

// An explicit work stack encodes evaluation order. Source discovery edges are
// never used for transfers (notably LHS, deferred bodies and derived csharp).
fn transfer(
    script: &Node,
    tokens: &[Token],
    context: Context,
    collector: &mut Option<HypothesisCollector>,
) -> CoreReport {
    use NodeKind::*;

    struct Conditional<'a> {
        conditions: &'a [Node],
        blocks: &'a [Node],
        else_block: Option<&'a Node>,
        next: usize,
        remaining_false: Environment,
        joined: Option<Environment>,
        any_flow: bool,
        parent_conditional: bool,
    }

    enum Work<'a> {
        Statement(&'a Node),
        Write(&'a Node, String),
        Command(&'a Node),
        Condition(Conditional<'a>),
        Split(Conditional<'a>),
        Arm(Conditional<'a>),
        FinishConditional(Conditional<'a>),
    }

    let mut environment = Environment::new();
    let mut report = CoreReport::default();
    let mut pending = vec![Work::Statement(script)];
    let mut after_flow = false;
    let mut conditional = false;
    while let Some(work) = pending.pop() {
        match work {
            Work::Condition(frame) => {
                if let Some(condition) = frame.conditions.get(frame.next) {
                    pending.push(Work::Split(frame));
                    pending.push(Work::Statement(condition));
                } else {
                    let else_block = frame.else_block;
                    pending.push(Work::FinishConditional(frame));
                    if let Some(body) = else_block {
                        pending.push(Work::Statement(body));
                    }
                }
            }
            Work::Split(mut frame) => {
                // Conditions run only on the remaining-false prefix. Save it
                // after effects, before a body can change its own snapshot.
                frame.remaining_false = environment.clone();
                frame.any_flow |= after_flow;
                after_flow = false;
                let body = &frame.blocks[frame.next];
                pending.push(Work::Arm(frame));
                pending.push(Work::Statement(body));
            }
            Work::Arm(mut frame) => {
                join_outcome(&mut frame.joined, std::mem::take(&mut environment));
                frame.any_flow |= after_flow;
                environment = std::mem::take(&mut frame.remaining_false);
                after_flow = false;
                frame.next += 1;
                pending.push(Work::Condition(frame));
            }
            Work::FinishConditional(mut frame) => {
                // The final prefix is an outcome even without an explicit else.
                join_outcome(&mut frame.joined, std::mem::take(&mut environment));
                environment = frame.joined.expect("the final outcome was joined above");
                after_flow |= frame.any_flow;
                conditional = frame.parent_conditional;
            }
            Work::Write(node, key) => {
                let Assignment { target, value, op } = &node.kind else {
                    unreachable!("only assignments schedule writes");
                };
                let value = (op == "=")
                    .then(|| evaluate_value(value, &environment).ok())
                    .flatten()
                    .map(|mut constant| {
                        let Variable(raw) = &target.kind else {
                            unreachable!("supported targets are variables");
                        };
                        append_unique(
                            &mut constant.assignment_origins,
                            AssignmentOrigin {
                                assignment_span: node.span,
                                target_span: target.span,
                                value_span: value.span,
                                raw_target: raw.clone(),
                            },
                        );
                        constant
                    });
                environment.remove(&key);
                if let Some(value) = value {
                    environment.insert(key, value);
                }
            }
            Work::Command(node) => {
                if !conditional {
                    record_command(node, tokens, &context, &environment, &mut report);
                } else if let Some(collector) = collector {
                    if let Some(gap) = collector.observe(node, tokens, &context, &environment) {
                        report.coverage.push(gap);
                    }
                }
                environment.clear();
            }
            Work::Statement(node) => {
                if after_flow {
                    append_report(
                        &mut report,
                        transfer_discovery(
                            node,
                            tokens,
                            context.clone(),
                            DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                            conditional,
                        ),
                    );
                    continue;
                }
                match &node.kind {
                    Script(statements) => {
                        pending.extend(statements.iter().rev().map(Work::Statement));
                    }
                    _ if is_conditional_expression(node) => {
                        let effects = summarize_conditional_effects(node, tokens);
                        effects.invalidate(&mut environment);
                        let region = if matches!(node.kind, Ternary { .. }) {
                            RegionKind::ControlFlow
                        } else {
                            RegionKind::Expression
                        };
                        append_report(
                            &mut report,
                            transfer_discovery(
                                node,
                                tokens,
                                context.clone(),
                                DiscoveryReason::UnsupportedRegion(region),
                                conditional,
                            ),
                        );
                        // No inner assignment, command or definition transfers.
                        effects.invalidate(&mut environment);
                    }
                    While { .. } | DoWhile { .. } | For { .. } | ForEach { .. } => {
                        let effects = summarize_control_effects(node, tokens);
                        effects.invalidate(&mut environment);
                        append_report(
                            &mut report,
                            transfer_discovery(
                                node,
                                tokens,
                                context.clone(),
                                DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                                conditional,
                            ),
                        );
                        // Discovery cannot install values from the header/body.
                        effects.invalidate(&mut environment);
                    }
                    Try { .. } => {
                        let effects = summarize_control_effects(node, tokens);
                        effects.invalidate(&mut environment);
                        append_report(
                            &mut report,
                            transfer_discovery(
                                node,
                                tokens,
                                context.clone(),
                                DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                                conditional,
                            ),
                        );
                        // No body transfers, exception routing or finally restore.
                        effects.invalidate(&mut environment);
                    }
                    If {
                        conditions,
                        blocks,
                        else_block,
                    } => {
                        append_report(
                            &mut report,
                            transfer_discovery(
                                node,
                                tokens,
                                context.clone(),
                                DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                                conditional,
                            ),
                        );
                        // Only active frames retain a prefix and an incremental
                        // join. No recursive branch transfer or path enumeration.
                        pending.push(Work::Condition(Conditional {
                            conditions,
                            blocks,
                            else_block: else_block.as_deref(),
                            next: 0,
                            remaining_false: Environment::new(),
                            joined: None,
                            any_flow: false,
                            parent_conditional: conditional,
                        }));
                        conditional = true;
                    }
                    Assignment { target, value, op } => {
                        if let Some(key) = target_key(target) {
                            if op == "??=" {
                                // Nullness is not modeled: always kill the target
                                // and every potential RHS write, never run the RHS.
                                let effects = summarize_conditional_effects(node, tokens);
                                effects.invalidate(&mut environment);
                                let reason = if is_effect_free_value(value) {
                                    DiscoveryReason::PendingTransferAnalysis
                                } else {
                                    DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow)
                                };
                                append_report(
                                    &mut report,
                                    transfer_discovery(
                                        node,
                                        tokens,
                                        context.clone(),
                                        reason,
                                        conditional,
                                    ),
                                );
                                effects.invalidate(&mut environment);
                            } else {
                                pending.push(Work::Write(node, key));
                                pending.push(Work::Statement(value));
                            }
                        } else {
                            discovery_region(
                                node,
                                tokens,
                                &context,
                                RegionKind::UnsupportedWrite,
                                &mut environment,
                                (!conditional).then_some(&mut report),
                            );
                        }
                    }
                    Unary { op, operand } | PostfixUnary { op, operand }
                        if op == "++" || op == "--" =>
                    {
                        if let Some(key) = target_key(operand) {
                            environment.remove(&key);
                        } else {
                            discovery_region(
                                node,
                                tokens,
                                &context,
                                RegionKind::UnsupportedWrite,
                                &mut environment,
                                (!conditional).then_some(&mut report),
                            );
                        }
                    }
                    Command {
                        name,
                        elements,
                        invocation,
                        redirections,
                        ..
                    } => {
                        let region = if !redirections.is_empty() {
                            Some(RegionKind::Redirection)
                        } else if *invocation || !matches!(name.kind, BareWord(_)) {
                            Some(RegionKind::Invocation)
                        } else {
                            None
                        };
                        if let Some(region) = region {
                            discovery_region(
                                node,
                                tokens,
                                &context,
                                region,
                                &mut environment,
                                (!conditional).then_some(&mut report),
                            );
                        } else {
                            pending.push(Work::Command(node));
                            for element in elements.iter().rev() {
                                let argument = match &element.kind {
                                    CommandParameter { argument, .. } => argument.as_deref(),
                                    _ => Some(element),
                                };
                                if let Some(argument) = argument {
                                    pending.push(Work::Statement(argument));
                                }
                            }
                        }
                    }
                    Function { .. } if is_ordinary_function(node, tokens) => {
                        if !conditional {
                            append_report(
                                &mut report,
                                analyze_function(node, tokens, context.clone(), collector),
                            );
                        }
                    }
                    Function { .. } => {
                        // Workflow is deferred/unsupported, not an ordinary
                        // function and not an executed barrier in its parent.
                        append_report(
                            &mut report,
                            transfer_discovery(
                                node,
                                tokens,
                                context.clone(),
                                DiscoveryReason::PendingTransferAnalysis,
                                conditional,
                            ),
                        );
                    }
                    ParamBlock(_)
                        if context.path.last().is_some_and(|frame| {
                            matches!(frame.kind, ContextKind::Function(_))
                        }) =>
                    {
                        // A body param block declares unknown inputs; neither
                        // its attributes nor defaults run in this environment.
                        // The parser places it first in the fresh function body.
                        append_report(
                            &mut report,
                            transfer_discovery(
                                node,
                                tokens,
                                context.clone(),
                                DiscoveryReason::UnsupportedRegion(
                                    RegionKind::ParameterDeclaration,
                                ),
                                conditional,
                            ),
                        );
                    }
                    ScriptBlockExpression(_)
                        if first_token(node, tokens)
                            .is_some_and(|t| t.kind == TokenKind::LBrace) =>
                    {
                        append_report(
                            &mut report,
                            transfer_discovery(
                                node,
                                tokens,
                                context.clone(),
                                DiscoveryReason::PendingTransferAnalysis,
                                conditional,
                            ),
                        );
                    }
                    StringLiteral { value, kind, .. } => {
                        // Eligibility precedes the fragment policy: raw
                        // constants remain constants without a new decoder.
                        if is_constant_string(value, *kind) {
                            append_report(
                                &mut report,
                                transfer_discovery(
                                    node,
                                    tokens,
                                    context.clone(),
                                    DiscoveryReason::PendingTransferAnalysis,
                                    conditional,
                                ),
                            );
                        } else {
                            environment.clear();
                            append_report(
                                &mut report,
                                transfer_discovery(
                                    node,
                                    tokens,
                                    context.clone(),
                                    DiscoveryReason::FragmentDiagnosticsUnavailable,
                                    conditional,
                                ),
                            );
                            environment.clear();
                        }
                    }
                    Variable(raw) if variable_key(raw).is_none() => {
                        if is_provider_read(raw) {
                            discovery_region(
                                node,
                                tokens,
                                &context,
                                RegionKind::VariableRead,
                                &mut environment,
                                (!conditional).then_some(&mut report),
                            );
                        } else {
                            append_report(
                                &mut report,
                                transfer_discovery(
                                    node,
                                    tokens,
                                    context.clone(),
                                    DiscoveryReason::PendingTransferAnalysis,
                                    conditional,
                                ),
                            );
                        }
                    }
                    Paren(_) if is_simple_value(node) => {
                        pending.push(Work::Statement(unparen(node)));
                    }
                    Variable(_) | Number(_) | TypeExpression(_) | BareWord(_)
                    | CSharpMemberDef(_) => {}
                    _ => {
                        let region = match &node.kind {
                            Flow { .. } => {
                                after_flow = true;
                                RegionKind::ControlFlow
                            }
                            Pipeline(_)
                            | PipelineChain { .. }
                            | Switch { .. }
                            | Catch { .. }
                            | Labeled { .. } => RegionKind::ControlFlow,
                            ClassDefinition { .. } | ClassMember { .. } | EnumDefinition { .. } => {
                                RegionKind::Declaration
                            }
                            ScriptBlockExpression(_) => RegionKind::KeywordBlock,
                            ParamBlock(_) | Parameter { .. } => RegionKind::ParameterDeclaration,
                            Using { .. } => RegionKind::Import,
                            Redirection { .. } => RegionKind::Redirection,
                            Error(_) => RegionKind::FragmentError,
                            _ => RegionKind::Expression,
                        };
                        discovery_region(
                            node,
                            tokens,
                            &context,
                            region,
                            &mut environment,
                            (!conditional).then_some(&mut report),
                        );
                    }
                }
            }
        }
    }
    // Presentation is lexical, not an assertion that nested arguments execute
    // after their command. Transfers above have already honored RHS/argument
    // timing; provenance retains its own deterministic encounter order.
    report
        .confirmed
        .sort_by_key(|finding| finding.site.command_span.start);
    report
        .unresolved
        .sort_by_key(|finding| finding.site.command_span.start);
    report
        .coverage
        .sort_by_key(|gap| (gap.span.start, std::cmp::Reverse(gap.span.end)));
    report
}

fn is_simple_value(node: &Node) -> bool {
    matches!(
        unparen(node).kind,
        NodeKind::StringLiteral { .. }
            | NodeKind::Variable(_)
            | NodeKind::Number(_)
            | NodeKind::TypeExpression(_)
            | NodeKind::BareWord(_)
    )
}

fn is_effect_free_value(node: &Node) -> bool {
    match &unparen(node).kind {
        NodeKind::StringLiteral { value, kind, .. } => is_constant_string(value, *kind),
        NodeKind::Variable(raw) => !is_provider_read(raw),
        NodeKind::Number(_) | NodeKind::TypeExpression(_) | NodeKind::BareWord(_) => true,
        _ => false,
    }
}

fn is_add_type(name: &Node) -> bool {
    matches!(&name.kind, NodeKind::BareWord(s) if s.eq_ignore_ascii_case("Add-Type"))
}

fn select_argument<'a>(
    elements: &'a [Node],
    tokens: &[Token],
    site: &mut AddTypeSite,
) -> Result<&'a Node, ArgumentReason> {
    let element = match elements {
        [] => return Err(ArgumentReason::MissingArgument),
        [element] => element,
        _ => return Err(ArgumentReason::ExtraElements),
    };
    let argument = if let NodeKind::CommandParameter { argument, .. } = &element.kind {
        site.parameter_span = Some(element.span);
        // The AST name has already lost its colon and leading dashes.
        // Accept only the exact raw parameter token, never its abbreviation.
        let token = first_token(element, tokens)
            .filter(|token| token.kind == TokenKind::Parameter)
            .ok_or(ArgumentReason::UnsupportedParameter)?;
        if token.value.ends_with(':') {
            return Err(ArgumentReason::ColonParameter);
        }
        if !token.value_eq_ci("-TypeDefinition") && !token.value_eq_ci("-MemberDefinition") {
            return Err(ArgumentReason::UnsupportedParameter);
        }
        argument
            .as_deref()
            .ok_or(ArgumentReason::MissingParameterArgument)?
    } else {
        element
    };
    site.argument_span = Some(argument.span);
    Ok(argument)
}

fn unparen(mut node: &Node) -> &Node {
    while let NodeKind::Paren(inner) = &node.kind {
        node = inner;
    }
    node
}

fn evaluate_value(node: &Node, environment: &Environment) -> Result<LiteralValue, ValueReason> {
    // Parentheses alone preserve the selected expression's value. In
    // particular, do not unwrap Script, Pipeline, or SubExpression nodes.
    let node = unparen(node);
    match &node.kind {
        NodeKind::StringLiteral { value, kind, .. } => {
            if !is_constant_string(value, *kind) {
                return Err(ValueReason::NonConstantString);
            }
            Ok(LiteralValue {
                raw_body: string_inner(value, *kind).to_string(),
                literal_origins: vec![LiteralOrigin {
                    literal_span: node.span,
                    body_span: body_span(value, *kind, node.span),
                }],
                assignment_origins: vec![],
                reference_origins: vec![],
            })
        }
        NodeKind::Variable(raw) => {
            let key = variable_key(raw).ok_or(ValueReason::UnsupportedVariable)?;
            let mut value = environment
                .get(&key)
                .cloned()
                .ok_or(ValueReason::VariableNotKnown)?;
            append_unique(
                &mut value.reference_origins,
                ReferenceOrigin {
                    span: node.span,
                    raw_name: raw.clone(),
                },
            );
            Ok(value)
        }
        NodeKind::Number(_) | NodeKind::TypeExpression(_) | NodeKind::BareWord(_) => {
            Err(ValueReason::SimpleNonString)
        }
        _ => Err(ValueReason::UnsupportedExpression),
    }
}

fn first_token<'a>(node: &Node, tokens: &'a [Token]) -> Option<&'a Token> {
    // Interpolation nodes have empty token ranges and already-absolute spans.
    (!node.range.is_empty())
        .then(|| tokens.get(node.range.first))
        .flatten()
}

fn push_context(context: &mut Context, kind: ContextKind, span: Span) {
    context.path.push(ContextFrame {
        kind,
        span,
        status: ContextStatus::Deferred,
    });
}

fn inherit_reason(inherited: DiscoveryReason, local: Option<DiscoveryReason>) -> DiscoveryReason {
    if inherited == DiscoveryReason::PendingTransferAnalysis {
        local.unwrap_or(inherited)
    } else {
        inherited
    }
}

/// Source-only discovery edges, never an evaluation order. Do not replace
/// this with Node::walk: Command.csharp is derived metadata, not another site.
fn source_children(node: &Node) -> Vec<&Node> {
    use NodeKind::*;

    let mut children = Vec::new();
    match &node.kind {
        Script(v) | ParamBlock(v) | Pipeline(v) | Array(v) | ArrayLiteral(v) => {
            children.extend(v);
        }
        Command {
            name,
            elements,
            redirections,
            ..
        } => {
            children.push(name.as_ref());
            children.extend(elements);
            children.extend(redirections);
        }
        CommandParameter { argument, .. } => children.extend(argument.as_deref()),
        Redirection { target, .. } => children.extend(target.as_deref()),
        Assignment { target, value, .. } => {
            children.extend([target.as_ref(), value.as_ref()]);
        }
        PipelineChain { left, right, .. } | Binary { left, right, .. } => {
            children.extend([left.as_ref(), right.as_ref()]);
        }
        If {
            conditions,
            blocks,
            else_block,
        } => {
            children.extend(conditions);
            children.extend(blocks);
            children.extend(else_block.as_deref());
        }
        While { condition, body }
        | DoWhile {
            condition, body, ..
        } => {
            children.extend([condition.as_ref(), body.as_ref()]);
        }
        For {
            init,
            condition,
            update,
            body,
        } => {
            children.extend(init.as_deref());
            children.extend(condition.as_deref());
            children.extend(update.as_deref());
            children.push(body);
        }
        ForEach {
            variable,
            iterable,
            body,
        } => children.extend([variable.as_ref(), iterable.as_ref(), body.as_ref()]),
        Switch {
            flags,
            input,
            cases,
        } => {
            children.extend(flags);
            children.push(input);
            children.extend(cases);
        }
        Function {
            parameters, body, ..
        } => {
            children.extend(parameters);
            children.push(body);
        }
        Try {
            body,
            catches,
            finally_block,
        } => {
            children.push(body.as_ref());
            children.extend(catches);
            children.extend(finally_block.as_deref());
        }
        Catch { body } => children.push(body),
        ClassDefinition { members, .. } | EnumDefinition { members, .. } => {
            children.extend(members);
        }
        ClassMember {
            parameters,
            default,
            body,
            ..
        } => {
            children.extend(parameters);
            children.extend(default.as_deref());
            children.extend(body.as_deref());
        }
        Flow { value, .. } => children.extend(value.as_deref()),
        Parameter {
            attributes,
            default,
            ..
        } => {
            children.extend(attributes);
            children.extend(default.as_deref());
        }
        Labeled { statement, .. } => children.push(statement),
        Ternary {
            condition,
            if_true,
            if_false,
        } => children.extend([condition.as_ref(), if_true.as_ref(), if_false.as_ref()]),
        Unary { operand, .. } | PostfixUnary { operand, .. } | Cast { operand, .. } => {
            children.push(operand);
        }
        MemberAccess { target, .. } => children.push(target),
        InvokeMember { target, args, .. } => {
            children.push(target);
            children.extend(args);
        }
        Index { target, index } => children.extend([target.as_ref(), index.as_ref()]),
        Paren(n) | SubExpression(n) | ScriptBlockExpression(n) => children.push(n),
        Hashtable(pairs) => {
            for (key, value) in pairs {
                children.extend([key, value]);
            }
        }
        StringLiteral { parts, .. } => children.extend(parts),
        CSharpMemberDef(_)
        | Using { .. }
        | Variable(_)
        | Number(_)
        | TypeExpression(_)
        | BareWord(_)
        | Error(_) => {}
    }
    children
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyze_add_type(source: &str) -> Result<CoreReport, AnalysisError> {
        super::analyze_add_type(source, AnalysisOptions::default())
    }

    fn hypothesis_options() -> AnalysisOptions {
        AnalysisOptions {
            include_hypotheses: true,
            ..AnalysisOptions::default()
        }
    }

    #[test]
    fn source_admission_ascii_below_limit_admits() {
        let source = "#x";
        let options = AnalysisOptions::default();

        let actual = analyze_add_type_with_source_limit(source, options, 3);

        assert_eq!(actual, Ok(CoreReport::default()));
    }

    #[test]
    fn source_admission_ascii_at_limit_admits() {
        let source = "#xx";
        let options = AnalysisOptions::default();

        let actual = analyze_add_type_with_source_limit(source, options, 3);

        assert_eq!(actual, Ok(CoreReport::default()));
    }

    #[test]
    fn source_admission_ascii_above_limit_returns_source_byte_error() {
        let source = "#xxx";
        let options = AnalysisOptions::default();
        let expected = AnalysisError::ResourceLimit {
            resource: AnalysisResource::SourceBytes,
            limit: 3,
        };

        let actual = analyze_add_type_with_source_limit(source, options, 3);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn source_admission_empty_at_zero_limit_admits() {
        let source = "";
        let options = AnalysisOptions::default();

        let actual = analyze_add_type_with_source_limit(source, options, 0);

        assert_eq!(actual, Ok(CoreReport::default()));
    }

    const SOURCE_ADMISSION_BOM_CONDITIONAL: &str = "\u{feff}if ($p) { Add-Type 'é' }";

    #[test]
    fn source_admission_bom_conditional_at_limit_preserves_report() {
        let source = SOURCE_ADMISSION_BOM_CONDITIONAL;
        let options = hypothesis_options();
        let site = positional((13, 26), (13, 21), (22, 26));
        let expected = CoreReport {
            confirmed: vec![],
            unresolved: vec![conditional_site(28, site.clone())],
            coverage: vec![gap(28, (3, 28), RegionKind::ControlFlow)],
            hypotheses: vec![literal_hypothesis(
                site,
                "é",
                literal((22, 26), (23, 25)),
                script_context(28),
            )],
        };

        let actual = analyze_add_type_with_source_limit(source, options, 28);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn source_admission_bom_conditional_above_limit_returns_source_byte_error() {
        let source = SOURCE_ADMISSION_BOM_CONDITIONAL;
        let options = hypothesis_options();
        let expected = AnalysisError::ResourceLimit {
            resource: AnalysisResource::SourceBytes,
            limit: 27,
        };

        let actual = analyze_add_type_with_source_limit(source, options, 27);

        assert_eq!(actual, Err(expected));
    }

    const SOURCE_ADMISSION_BOM_MALFORMED: &str = "\u{feff}param( <# Ż";

    fn source_admission_bom_malformed_diagnostics() -> AnalysisError {
        AnalysisError::Syntax {
            lex_errors: vec![LexError {
                span: Span::new(10, 15),
                message: "unterminated block comment".into(),
            }],
            parse_errors: vec![ParseError {
                span: Span::new(15, 15),
                message: "expected ')'".into(),
            }],
        }
    }

    #[test]
    fn source_admission_bom_malformed_at_limit_with_hypotheses_off_preserves_diagnostics() {
        let source = SOURCE_ADMISSION_BOM_MALFORMED;
        let options = AnalysisOptions::default();
        let expected = source_admission_bom_malformed_diagnostics();

        let actual = analyze_add_type_with_source_limit(source, options, 15);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn source_admission_bom_malformed_at_limit_with_hypotheses_on_preserves_diagnostics() {
        let source = SOURCE_ADMISSION_BOM_MALFORMED;
        let options = hypothesis_options();
        let expected = source_admission_bom_malformed_diagnostics();

        let actual = analyze_add_type_with_source_limit(source, options, 15);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn source_admission_bom_malformed_above_limit_with_hypotheses_off_returns_source_byte_error() {
        let source = SOURCE_ADMISSION_BOM_MALFORMED;
        let options = AnalysisOptions::default();
        let expected = AnalysisError::ResourceLimit {
            resource: AnalysisResource::SourceBytes,
            limit: 14,
        };

        let actual = analyze_add_type_with_source_limit(source, options, 14);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn source_admission_bom_malformed_above_limit_with_hypotheses_on_returns_source_byte_error() {
        let source = SOURCE_ADMISSION_BOM_MALFORMED;
        let options = hypothesis_options();
        let expected = AnalysisError::ResourceLimit {
            resource: AnalysisResource::SourceBytes,
            limit: 14,
        };

        let actual = analyze_add_type_with_source_limit(source, options, 14);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn source_admission_default_entry_admits_small_literal() {
        let source = "Add-Type 'X'";
        let expected = CoreReport {
            confirmed: vec![known(
                12,
                positional((0, 12), (0, 8), (9, 12)),
                "X",
                literal((9, 12), (10, 11)),
                vec![],
                vec![],
            )],
            ..CoreReport::default()
        };

        let actual = super::analyze_add_type(source, AnalysisOptions::default());

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn source_admission_resource_error_display_reports_only_the_byte_limit() {
        let error: &dyn std::error::Error = &AnalysisError::ResourceLimit {
            resource: AnalysisResource::SourceBytes,
            limit: 3,
        };

        let actual = error.to_string();

        assert_eq!(actual, "source byte limit exceeded (3 bytes)");
    }

    macro_rules! hypothesis_option_case {
        ($name:ident, $options:expr) => {
            #[test]
            fn $name() {
                let source = "if ($p) { Add-Type 'X' }";
                let options = $options;
                let expected = CoreReport {
                    unresolved: vec![conditional_site(
                        24,
                        positional((10, 22), (10, 18), (19, 22)),
                    )],
                    coverage: vec![gap(24, (0, 24), RegionKind::ControlFlow)],
                    ..CoreReport::default()
                };

                let actual = super::analyze_add_type(source, options);

                assert_eq!(actual, Ok(expected));
            }
        };
    }

    hypothesis_option_case!(
        analyze_default_options_leave_conditional_site_unresolved_without_hypotheses,
        AnalysisOptions::default()
    );
    hypothesis_option_case!(
        analyze_explicit_off_with_zero_limits_never_emits_hypotheses_or_truncation,
        AnalysisOptions {
            include_hypotheses: false,
            hypothesis_limits: HypothesisLimits {
                candidates: 0,
                owned_text_bytes: 0,
                provenance_context_records: 0,
            },
        }
    );

    #[test]
    fn analyze_enabled_conditional_literal_owns_bom_utf8_site_and_provenance_after_source_drop() {
        let expected = CoreReport {
            hypotheses: vec![HypothesisConditionalSite {
                site: positional((13, 26), (13, 21), (22, 26)),
                reason: HypothesisReason::ConditionalSite,
                raw_body: "é".into(),
                literal_origins: vec![literal((22, 26), (23, 25))],
                assignment_origins: vec![],
                reference_origins: vec![],
                context: script_context(28),
            }],
            unresolved: vec![conditional_site(
                28,
                positional((13, 26), (13, 21), (22, 26)),
            )],
            coverage: vec![gap(28, (3, 28), RegionKind::ControlFlow)],
            ..CoreReport::default()
        };

        let actual = {
            let source = String::from("\u{feff}if ($p) { Add-Type 'é' }");
            super::analyze_add_type(&source, hypothesis_options())
        };

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_enabled_current_variable_keeps_origins_before_command_clear_without_restoring_join()
    {
        let source = "$a='X'; if ($p) { Add-Type ($a) }; Add-Type $a";
        let expected = CoreReport {
            hypotheses: vec![HypothesisConditionalSite {
                site: positional((18, 31), (18, 26), (27, 31)),
                reason: HypothesisReason::ConditionalSite,
                raw_body: "X".into(),
                literal_origins: vec![literal((3, 6), (4, 5))],
                assignment_origins: vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                reference_origins: vec![reference((28, 30), "$a")],
                context: script_context(46),
            }],
            unresolved: vec![
                conditional_site(46, positional((18, 31), (18, 26), (27, 31))),
                unknown(
                    46,
                    positional((35, 46), (35, 43), (44, 46)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![gap(46, (8, 33), RegionKind::ControlFlow)],
            ..CoreReport::default()
        };

        let actual = super::analyze_add_type(source, hypothesis_options());

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_enabled_ordinary_script_and_function_confirmations_are_not_hypotheses_even_at_zero()
    {
        let source = "Add-Type 'S'; function f { Add-Type 'F' }";
        let options = AnalysisOptions {
            hypothesis_limits: HypothesisLimits {
                candidates: 0,
                owned_text_bytes: 0,
                provenance_context_records: 0,
            },
            ..hypothesis_options()
        };
        let expected = CoreReport {
            confirmed: vec![
                known(
                    41,
                    positional((0, 12), (0, 8), (9, 12)),
                    "S",
                    literal((9, 12), (10, 11)),
                    vec![],
                    vec![],
                ),
                Confirmed {
                    context: nested_context(
                        script_context(41),
                        ContextKind::Function("f".into()),
                        (14, 41),
                    ),
                    ..known(
                        41,
                        positional((27, 39), (27, 35), (36, 39)),
                        "F",
                        literal((36, 39), (37, 38)),
                        vec![],
                        vec![],
                    )
                },
            ],
            ..CoreReport::default()
        };

        let actual = super::analyze_add_type(source, options);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_count_overflow_stops_globally_at_function_site_and_keeps_ordinary_continuation() {
        let source =
            "if ($p) { Add-Type 'A' }; function f { if ($q) { Add-Type 'B' } }; if ($r) { Add-Type 'C' }; Add-Type 'S'";
        let options = AnalysisOptions {
            hypothesis_limits: HypothesisLimits {
                candidates: 1,
                ..hypothesis_options().hypothesis_limits
            },
            ..hypothesis_options()
        };
        let function_context = nested_context(
            script_context(105),
            ContextKind::Function("f".into()),
            (26, 65),
        );
        let expected = CoreReport {
            hypotheses: vec![HypothesisConditionalSite {
                site: positional((10, 22), (10, 18), (19, 22)),
                reason: HypothesisReason::ConditionalSite,
                raw_body: "A".into(),
                literal_origins: vec![literal((19, 22), (20, 21))],
                assignment_origins: vec![],
                reference_origins: vec![],
                context: script_context(105),
            }],
            confirmed: vec![known(
                105,
                positional((93, 105), (93, 101), (102, 105)),
                "S",
                literal((102, 105), (103, 104)),
                vec![],
                vec![],
            )],
            unresolved: vec![
                conditional_site(105, positional((10, 22), (10, 18), (19, 22))),
                Unresolved {
                    site: positional((49, 61), (49, 57), (58, 61)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: function_context.clone(),
                },
                conditional_site(105, positional((77, 89), (77, 85), (86, 89))),
            ],
            coverage: vec![
                gap(105, (0, 24), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(39, 63),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: function_context.clone(),
                },
                CoverageGap {
                    span: Span::new(49, 61),
                    reason: DiscoveryReason::HypothesesTruncated(HypothesisLimit::CandidateCount),
                    context: function_context,
                },
                gap(105, (67, 91), RegionKind::ControlFlow),
            ],
        };

        let actual = super::analyze_add_type(source, options);

        assert_eq!(actual, Ok(expected));
    }

    macro_rules! hypothesis_budget_case {
        ($name:ident, $limits:expr, $hypotheses:expr, $extra_coverage:expr) => {
            #[test]
            fn $name() {
                let source = "function foo { $a='X'; if ($p) { Add-Type $a } }";
                let options = AnalysisOptions {
                    hypothesis_limits: $limits,
                    ..hypothesis_options()
                };
                let context = nested_context(
                    script_context(48),
                    ContextKind::Function("foo".into()),
                    (0, 48),
                );
                let mut coverage = vec![CoverageGap {
                    span: Span::new(23, 46),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: context.clone(),
                }];
                coverage.extend($extra_coverage);
                let expected = CoreReport {
                    hypotheses: $hypotheses,
                    unresolved: vec![Unresolved {
                        site: positional((33, 44), (33, 41), (42, 44)),
                        reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                        context,
                    }],
                    coverage,
                    ..CoreReport::default()
                };

                let actual = super::analyze_add_type(source, options);

                assert_eq!(actual, Ok(expected));
            }
        };
    }

    fn budget_function_context() -> Context {
        nested_context(
            script_context(48),
            ContextKind::Function("foo".into()),
            (0, 48),
        )
    }

    fn budget_function_truncation(limit: HypothesisLimit) -> Vec<CoverageGap> {
        vec![CoverageGap {
            span: Span::new(33, 44),
            reason: DiscoveryReason::HypothesesTruncated(limit),
            context: budget_function_context(),
        }]
    }

    hypothesis_budget_case!(
        analyze_exact_text_and_record_limits_admit_whole_variable_candidate_with_context_name,
        HypothesisLimits {
            candidates: 1,
            owned_text_bytes: 8,
            provenance_context_records: 5,
        },
        vec![HypothesisConditionalSite {
            site: positional((33, 44), (33, 41), (42, 44)),
            reason: HypothesisReason::ConditionalSite,
            raw_body: "X".into(),
            literal_origins: vec![literal((18, 21), (19, 20))],
            assignment_origins: vec![assignment((15, 21), (15, 17), (18, 21), "$a")],
            reference_origins: vec![reference((42, 44), "$a")],
            context: budget_function_context(),
        }],
        Vec::<CoverageGap>::new()
    );
    hypothesis_budget_case!(
        analyze_one_byte_short_rejects_whole_candidate_not_only_body,
        HypothesisLimits {
            owned_text_bytes: 7,
            ..hypothesis_options().hypothesis_limits
        },
        vec![],
        budget_function_truncation(HypothesisLimit::OwnedTextBytes)
    );
    hypothesis_budget_case!(
        analyze_one_record_short_rejects_whole_candidate_including_context_frames,
        HypothesisLimits {
            provenance_context_records: 4,
            ..hypothesis_options().hypothesis_limits
        },
        vec![],
        budget_function_truncation(HypothesisLimit::ProvenanceContextRecords)
    );
    hypothesis_budget_case!(
        analyze_zero_text_limit_truncates_only_at_eligible_site,
        HypothesisLimits {
            owned_text_bytes: 0,
            ..hypothesis_options().hypothesis_limits
        },
        vec![],
        budget_function_truncation(HypothesisLimit::OwnedTextBytes)
    );
    hypothesis_budget_case!(
        analyze_zero_records_limit_truncates_only_at_eligible_site,
        HypothesisLimits {
            provenance_context_records: 0,
            ..hypothesis_options().hypothesis_limits
        },
        vec![],
        budget_function_truncation(HypothesisLimit::ProvenanceContextRecords)
    );
    hypothesis_budget_case!(
        analyze_zero_limits_choose_count_before_text_and_records,
        HypothesisLimits {
            candidates: 0,
            owned_text_bytes: 0,
            provenance_context_records: 0,
        },
        vec![],
        budget_function_truncation(HypothesisLimit::CandidateCount)
    );

    #[test]
    fn analyze_enabled_empty_input_with_zero_limits_has_no_truncation() {
        let options = AnalysisOptions {
            hypothesis_limits: HypothesisLimits {
                candidates: 0,
                owned_text_bytes: 0,
                provenance_context_records: 0,
            },
            ..hypothesis_options()
        };

        let actual = super::analyze_add_type("", options);

        assert_eq!(actual, Ok(CoreReport::default()));
    }

    #[test]
    fn analyze_enabled_syntax_error_returns_original_diagnostics_before_collection() {
        let source = "\u{feff}'Ż";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![LexError {
                span: Span::new(3, 6),
                message: "unterminated string".into(),
            }],
            parse_errors: vec![],
        };

        let actual = super::analyze_add_type(source, hypothesis_options());

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn hypothesis_budget_addition_overflow_fails_closed_without_large_allocation() {
        let cases = [
            (usize::MAX, Some(0), usize::MAX, Some(usize::MAX)),
            (usize::MAX, Some(1), usize::MAX, None),
            (0, None, usize::MAX, None),
        ];

        for (used, additional, limit, expected) in cases {
            assert_eq!(budget_total(used, additional, limit), expected);
        }
    }

    #[test]
    fn analysis_options_default_to_off_with_approved_limits() {
        let expected = AnalysisOptions {
            include_hypotheses: false,
            hypothesis_limits: HypothesisLimits {
                candidates: 128,
                owned_text_bytes: 1_048_576,
                provenance_context_records: 4_096,
            },
        };

        let actual = AnalysisOptions::default();

        assert_eq!(actual, expected);
    }

    macro_rules! analyze_hypothesis_case {
        ($name:ident, $source:expr, $limits:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let options = AnalysisOptions {
                    include_hypotheses: true,
                    hypothesis_limits: $limits,
                };
                let expected = $expected;

                let actual = super::analyze_add_type(source, options);

                assert_eq!(actual, Ok(expected));
            }
        };
    }

    fn hypothesis_limits(text: usize, records: usize) -> HypothesisLimits {
        HypothesisLimits {
            candidates: 8,
            owned_text_bytes: text,
            provenance_context_records: records,
        }
    }

    fn literal_hypothesis(
        site: AddTypeSite,
        raw: &str,
        origin: LiteralOrigin,
        context: Context,
    ) -> HypothesisConditionalSite {
        HypothesisConditionalSite {
            site,
            reason: HypothesisReason::ConditionalSite,
            raw_body: raw.into(),
            literal_origins: vec![origin],
            assignment_origins: vec![],
            reference_origins: vec![],
            context,
        }
    }

    fn hypothesis_truncation(
        span: (usize, usize),
        limit: HypothesisLimit,
        context: Context,
    ) -> CoverageGap {
        CoverageGap {
            span: Span::new(span.0, span.1),
            reason: DiscoveryReason::HypothesesTruncated(limit),
            context,
        }
    }

    const CUMULATIVE_BUDGET_SOURCE: &str =
        "if ($p) { Add-Type 'AB' }; function f { if ($q) { Add-Type 'CDE' } }";
    const CUMULATIVE_BUDGET_WITH_LATER_SMALLER_SITE: &str =
        "if ($p) { Add-Type 'AB' }; function f { if ($q) { Add-Type 'CDE' } }; if ($r) { Add-Type 'Z' }";

    fn cumulative_budget_function_context(end: usize) -> Context {
        nested_context(
            script_context(end),
            ContextKind::Function("f".into()),
            (27, 68),
        )
    }

    fn cumulative_budget_report(end: usize) -> CoreReport {
        let first_site = positional((10, 23), (10, 18), (19, 23));
        CoreReport {
            hypotheses: vec![literal_hypothesis(
                first_site.clone(),
                "AB",
                literal((19, 23), (20, 22)),
                script_context(end),
            )],
            unresolved: vec![
                conditional_site(end, first_site),
                Unresolved {
                    site: positional((50, 64), (50, 58), (59, 64)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: cumulative_budget_function_context(end),
                },
            ],
            coverage: vec![
                gap(end, (0, 25), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(40, 66),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: cumulative_budget_function_context(end),
                },
            ],
            ..CoreReport::default()
        }
    }

    fn cumulative_budget_exact_report() -> CoreReport {
        let mut report = cumulative_budget_report(68);
        report.hypotheses.push(literal_hypothesis(
            positional((50, 64), (50, 58), (59, 64)),
            "CDE",
            literal((59, 64), (60, 63)),
            cumulative_budget_function_context(68),
        ));
        report
    }

    fn cumulative_budget_short_report(limit: HypothesisLimit) -> CoreReport {
        let mut report = cumulative_budget_report(94);
        report.unresolved.push(conditional_site(
            94,
            positional((80, 92), (80, 88), (89, 92)),
        ));
        report.coverage.push(hypothesis_truncation(
            (50, 64),
            limit,
            cumulative_budget_function_context(94),
        ));
        report
            .coverage
            .push(gap(94, (70, 94), RegionKind::ControlFlow));
        report
    }

    // Script and function cost 2 + (3 body + 1 function-name) bytes and
    // (1 origin + 1 frame) + (1 origin + 2 frames) records. At either
    // one-short limit the later script site (1 byte, 2 records) would fit.
    analyze_hypothesis_case!(
        analyze_cumulative_exact_text_limit_admits_script_and_function_sites,
        CUMULATIVE_BUDGET_SOURCE,
        hypothesis_limits(6, 16),
        cumulative_budget_exact_report()
    );
    analyze_hypothesis_case!(
        analyze_cumulative_one_byte_short_stops_at_function_before_smaller_script_site,
        CUMULATIVE_BUDGET_WITH_LATER_SMALLER_SITE,
        hypothesis_limits(5, 16),
        cumulative_budget_short_report(HypothesisLimit::OwnedTextBytes)
    );
    analyze_hypothesis_case!(
        analyze_cumulative_exact_record_limit_admits_script_and_function_sites,
        CUMULATIVE_BUDGET_SOURCE,
        hypothesis_limits(32, 5),
        cumulative_budget_exact_report()
    );
    analyze_hypothesis_case!(
        analyze_cumulative_one_record_short_stops_at_function_before_smaller_script_site,
        CUMULATIVE_BUDGET_WITH_LATER_SMALLER_SITE,
        hypothesis_limits(32, 4),
        cumulative_budget_short_report(HypothesisLimit::ProvenanceContextRecords)
    );

    const DIRECT_LITERAL_BUDGET_SOURCE: &str = "if ($p) { Add-Type 'é' }";

    fn direct_literal_budget_report() -> CoreReport {
        CoreReport {
            unresolved: vec![conditional_site(
                25,
                positional((10, 23), (10, 18), (19, 23)),
            )],
            coverage: vec![gap(25, (0, 25), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    }

    fn direct_literal_budget_short_report(limit: HypothesisLimit) -> CoreReport {
        let mut report = direct_literal_budget_report();
        report
            .coverage
            .push(hypothesis_truncation((10, 23), limit, script_context(25)));
        report
    }

    analyze_hypothesis_case!(
        analyze_direct_literal_at_exact_text_and_record_limits_keeps_body_and_origin,
        DIRECT_LITERAL_BUDGET_SOURCE,
        hypothesis_limits(2, 2),
        CoreReport {
            hypotheses: vec![literal_hypothesis(
                positional((10, 23), (10, 18), (19, 23)),
                "é",
                literal((19, 23), (20, 22)),
                script_context(25),
            )],
            ..direct_literal_budget_report()
        }
    );
    analyze_hypothesis_case!(
        analyze_direct_literal_one_byte_short_truncates_at_literal_site,
        DIRECT_LITERAL_BUDGET_SOURCE,
        hypothesis_limits(1, 2),
        direct_literal_budget_short_report(HypothesisLimit::OwnedTextBytes)
    );
    analyze_hypothesis_case!(
        analyze_direct_literal_one_record_short_truncates_at_literal_site,
        DIRECT_LITERAL_BUDGET_SOURCE,
        hypothesis_limits(2, 1),
        direct_literal_budget_short_report(HypothesisLimit::ProvenanceContextRecords)
    );

    const INELIGIBLE_CONDITIONAL_SITES: &str =
        "if ($p) { Write-Output 'N'; Add-Type -Type 'X'; Add-Type 'X' 'Y'; Add-Type \"$p\"; Add-Type $missing }; \
         if ($p) { $a='X' } else { $a='Y' }; if ($q) { Add-Type $a }";

    fn ineligible_conditional_sites_report() -> CoreReport {
        CoreReport {
            unresolved: vec![
                conditional_site(
                    161,
                    AddTypeSite {
                        command_span: Span::new(28, 46),
                        name_span: Span::new(28, 36),
                        argument_span: None,
                        parameter_span: Some(Span::new(37, 46)),
                    },
                ),
                conditional_site(
                    161,
                    AddTypeSite {
                        command_span: Span::new(48, 64),
                        name_span: Span::new(48, 56),
                        argument_span: None,
                        parameter_span: None,
                    },
                ),
                conditional_site(161, positional((66, 79), (66, 74), (75, 79))),
                conditional_site(161, positional((81, 98), (81, 89), (90, 98))),
                conditional_site(161, positional((148, 159), (148, 156), (157, 159))),
            ],
            coverage: vec![
                gap(161, (0, 100), RegionKind::ControlFlow),
                gap(161, (10, 26), RegionKind::OtherCommand),
                CoverageGap {
                    span: Span::new(75, 79),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(161),
                },
                gap(161, (102, 136), RegionKind::ControlFlow),
                gap(161, (138, 161), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    }

    analyze_hypothesis_case!(
        analyze_enabled_ineligible_conditional_sites_do_not_trigger_admission_at_zero_limits,
        INELIGIBLE_CONDITIONAL_SITES,
        HypothesisLimits {
            candidates: 0,
            owned_text_bytes: 0,
            provenance_context_records: 0,
        },
        ineligible_conditional_sites_report()
    );
    analyze_hypothesis_case!(
        analyze_enabled_ineligible_conditional_sites_produce_no_candidates_with_budget_available,
        INELIGIBLE_CONDITIONAL_SITES,
        hypothesis_limits(32, 16),
        ineligible_conditional_sites_report()
    );

    fn positional(
        command: (usize, usize),
        name: (usize, usize),
        argument: (usize, usize),
    ) -> AddTypeSite {
        AddTypeSite {
            command_span: Span::new(command.0, command.1),
            name_span: Span::new(name.0, name.1),
            argument_span: Some(Span::new(argument.0, argument.1)),
            parameter_span: None,
        }
    }

    fn literal(literal: (usize, usize), body: (usize, usize)) -> LiteralOrigin {
        LiteralOrigin {
            literal_span: Span::new(literal.0, literal.1),
            body_span: Span::new(body.0, body.1),
        }
    }

    fn assignment(
        whole: (usize, usize),
        target: (usize, usize),
        value: (usize, usize),
        raw: &str,
    ) -> AssignmentOrigin {
        AssignmentOrigin {
            assignment_span: Span::new(whole.0, whole.1),
            target_span: Span::new(target.0, target.1),
            value_span: Span::new(value.0, value.1),
            raw_target: raw.into(),
        }
    }

    fn reference(span: (usize, usize), raw: &str) -> ReferenceOrigin {
        ReferenceOrigin {
            span: Span::new(span.0, span.1),
            raw_name: raw.into(),
        }
    }

    fn known(
        end: usize,
        site: AddTypeSite,
        raw: &str,
        origin: LiteralOrigin,
        assignments: Vec<AssignmentOrigin>,
        references: Vec<ReferenceOrigin>,
    ) -> Confirmed {
        Confirmed {
            site,
            raw_body: raw.into(),
            literal_origins: vec![origin],
            assignment_origins: assignments,
            reference_origins: references,
            context: script_context(end),
        }
    }

    fn unknown(end: usize, site: AddTypeSite, reason: ValueReason) -> Unresolved {
        Unresolved {
            site,
            reason: DiscoveryReason::UnknownValue(reason),
            context: script_context(end),
        }
    }

    fn conditional_site(end: usize, site: AddTypeSite) -> Unresolved {
        Unresolved {
            site,
            reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
            context: script_context(end),
        }
    }

    fn gap(end: usize, span: (usize, usize), region: RegionKind) -> CoverageGap {
        CoverageGap {
            span: Span::new(span.0, span.1),
            reason: DiscoveryReason::UnsupportedRegion(region),
            context: script_context(end),
        }
    }

    macro_rules! transfer_case {
        ($name:ident, $source:literal, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let expected = $expected;

                let actual = analyze_add_type(source);

                assert_eq!(actual, Ok(expected));
            }
        };
    }

    transfer_case!(
        analyze_while_unrelated_body_write_preserves_inherited_constant,
        "$a='X'; while ($p) { $b='Y' }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                42,
                positional((31, 42), (31, 39), (40, 42)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((40, 42), "$a")],
            )],
            coverage: vec![gap(42, (8, 29), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    macro_rules! loop_preserves_case {
        ($name:ident, $source:literal, $end:literal, $start:literal, $coverage:expr) => {
            transfer_case!(
                $name,
                $source,
                CoreReport {
                    confirmed: vec![known(
                        $end,
                        positional(($start, $end), ($start, $start + 8), ($end - 2, $end)),
                        "X",
                        literal((3, 6), (4, 5)),
                        vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                        vec![reference(($end - 2, $end), "$a")],
                    )],
                    coverage: $coverage,
                    ..CoreReport::default()
                }
            );
        };
    }

    macro_rules! loop_unknown_case {
        ($name:ident, $source:literal, $end:literal, $start:literal, $coverage:expr) => {
            transfer_case!(
                $name,
                $source,
                CoreReport {
                    unresolved: vec![unknown(
                        $end,
                        positional(($start, $end), ($start, $start + 8), ($end - 2, $end)),
                        ValueReason::VariableNotKnown,
                    )],
                    coverage: $coverage,
                    ..CoreReport::default()
                }
            );
        };
    }

    loop_preserves_case!(
        analyze_coalescing_assignment_unrelated_rhs_write_preserves_inherited_constant,
        "$a='X'; $b??=($c='Y'); Add-Type $a",
        34,
        23,
        vec![
            gap(34, (8, 21), RegionKind::ControlFlow),
            gap(34, (13, 21), RegionKind::Expression),
        ]
    );

    loop_unknown_case!(
        analyze_coalescing_assignment_known_target_is_invalidated_despite_unrelated_rhs,
        "$a='X'; $a??=($b='Y'); Add-Type $a",
        34,
        23,
        vec![
            gap(34, (8, 21), RegionKind::ControlFlow),
            gap(34, (13, 21), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_known_target_does_not_prune_rhs_equal_write,
        "$a='X'; $b='Q'; $b??=($a='X'); Add-Type $a",
        42,
        31,
        vec![
            gap(42, (16, 29), RegionKind::ControlFlow),
            gap(42, (21, 29), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_coalescing_assignment_known_target_preserves_key_outside_both_write_sets,
        "$a='X'; $b='Q'; $b??=($c='Y'); Add-Type $a",
        42,
        31,
        vec![
            gap(42, (16, 29), RegionKind::ControlFlow),
            gap(42, (21, 29), RegionKind::Expression),
        ]
    );
    transfer_case!(
        analyze_coalescing_assignment_unknown_target_does_not_install_literal_rhs,
        "$b??='X'; Add-Type $b",
        CoreReport {
            unresolved: vec![unknown(
                21,
                positional((10, 21), (10, 18), (19, 21)),
                ValueReason::VariableNotKnown,
            )],
            ..CoreReport::default()
        }
    );
    loop_preserves_case!(
        analyze_coalescing_assignment_unparenthesized_rhs_write_is_not_transferred,
        "$a='X'; $b??=$c='Y'; Add-Type $a",
        32,
        21,
        vec![gap(32, (8, 19), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_nested_rhs_assignment_contributes_all_writes,
        "$a='X'; $b??=$c=$a='X'; Add-Type $a",
        35,
        24,
        vec![gap(35, (8, 22), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_rhs_prefix_increment_invalidates_key,
        "$a='X'; $b??=(++$a); Add-Type $a",
        32,
        21,
        vec![
            gap(32, (8, 19), RegionKind::ControlFlow),
            gap(32, (13, 19), RegionKind::Expression),
            gap(32, (14, 18), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_rhs_postfix_decrement_invalidates_key,
        "$a='X'; $b??=($a--); Add-Type $a",
        32,
        21,
        vec![
            gap(32, (8, 19), RegionKind::ControlFlow),
            gap(32, (13, 19), RegionKind::Expression),
            gap(32, (14, 18), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_coalescing_assignment_rhs_unrelated_mutation_preserves_key,
        "$a='X'; $b??=($c++); Add-Type $a",
        32,
        21,
        vec![
            gap(32, (8, 19), RegionKind::ControlFlow),
            gap(32, (13, 19), RegionKind::Expression),
            gap(32, (14, 18), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_nested_coalescing_assignment_and_ternary_preserve_unwritten_key,
        "$a='X'; $b??=($c??=($p ? ($d='Y') : 'Z')); Add-Type $a",
        54,
        43,
        vec![
            gap(54, (8, 41), RegionKind::ControlFlow),
            gap(54, (13, 41), RegionKind::Expression),
            gap(54, (14, 40), RegionKind::ControlFlow),
            gap(54, (19, 40), RegionKind::Expression),
            gap(54, (20, 39), RegionKind::ControlFlow),
            gap(54, (25, 33), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_nested_coalescing_assignment_ternary_false_arm_write_invalidates_key,
        "$a='X'; $b??=($c??=($p ? 'Y' : ($a='X'))); Add-Type $a",
        54,
        43,
        vec![
            gap(54, (8, 41), RegionKind::ControlFlow),
            gap(54, (13, 41), RegionKind::Expression),
            gap(54, (14, 40), RegionKind::ControlFlow),
            gap(54, (19, 40), RegionKind::Expression),
            gap(54, (20, 39), RegionKind::ControlFlow),
            gap(54, (31, 39), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_ternary_nested_coalescing_assignment_reuses_precise_rhs_summary,
        "$a='X'; $p ? ($b??=($c='Y')) : 'Z'; Add-Type $a",
        47,
        36,
        vec![
            gap(47, (8, 34), RegionKind::ControlFlow),
            gap(47, (13, 28), RegionKind::Expression),
            gap(47, (14, 27), RegionKind::ControlFlow),
            gap(47, (19, 27), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_nested_coalescing_assignment_rhs_equal_write_invalidates_key,
        "$a='X'; $p ? ($b??=($a='X')) : 'Z'; Add-Type $a",
        47,
        36,
        vec![
            gap(47, (8, 34), RegionKind::ControlFlow),
            gap(47, (13, 28), RegionKind::Expression),
            gap(47, (14, 27), RegionKind::ControlFlow),
            gap(47, (19, 27), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_coalescing_assignment_effect_free_double_rhs_keeps_existing_fragment_gap,
        "$a='X'; $b??=\"Y\"; Add-Type $a",
        29,
        18,
        vec![CoverageGap {
            span: Span::new(13, 16),
            reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
            context: script_context(29),
        }]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_member_target_dominates_precise_rhs_writes,
        "$a='X'; $b.X ??=($c='Y'); Add-Type $a",
        37,
        26,
        vec![
            gap(37, (8, 24), RegionKind::UnsupportedWrite),
            gap(37, (8, 12), RegionKind::Expression),
            gap(37, (16, 24), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_provider_rhs_dominates_target_write,
        "$a='X'; $b??=$env:P; Add-Type $a",
        32,
        21,
        vec![
            gap(32, (8, 19), RegionKind::ControlFlow),
            gap(32, (13, 19), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_arithmetic_rhs_remains_whole_barrier,
        "$a='X'; $b??=('Y'+'Z'); Add-Type $a",
        35,
        24,
        vec![
            gap(35, (8, 22), RegionKind::ControlFlow),
            gap(35, (13, 22), RegionKind::Expression),
            gap(35, (14, 21), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_subexpression_is_not_a_transparent_paren,
        "$a='X'; $b??=$($c='Y'); Add-Type $a",
        35,
        24,
        vec![
            gap(35, (8, 22), RegionKind::ControlFlow),
            gap(35, (13, 22), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_loop_and_header_unrelated_assignment_preserves_key,
        "$a='X'; while ($p -and ($b='Y')) { }; Add-Type $a",
        49,
        38,
        vec![
            gap(49, (8, 36), RegionKind::ControlFlow),
            gap(49, (15, 31), RegionKind::Expression),
            gap(49, (23, 31), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_or_header_equal_assignment_invalidates_key_without_execution,
        "$a='X'; while ($p -or ($a='X')) { }; Add-Type $a",
        48,
        37,
        vec![
            gap(48, (8, 35), RegionKind::ControlFlow),
            gap(48, (15, 30), RegionKind::Expression),
            gap(48, (22, 30), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_loop_ternary_body_unrelated_arm_writes_preserve_key,
        "$a='X'; while ($p) { $b=$q ? ($c='Y') : ($d='Z') }; Add-Type $a",
        63,
        52,
        vec![
            gap(63, (8, 50), RegionKind::ControlFlow),
            gap(63, (24, 48), RegionKind::ControlFlow),
            gap(63, (29, 37), RegionKind::Expression),
            gap(63, (40, 48), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_ternary_body_false_arm_equal_write_invalidates_key,
        "$a='X'; while ($p) { $b=$q ? ($c='Y') : ($a='X') }; Add-Type $a",
        63,
        52,
        vec![
            gap(63, (8, 50), RegionKind::ControlFlow),
            gap(63, (24, 48), RegionKind::ControlFlow),
            gap(63, (29, 37), RegionKind::Expression),
            gap(63, (40, 48), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_loop_coalescing_assignment_unrelated_rhs_write_preserves_key,
        "$a='X'; while ($p) { $b??=($c='Y') }; Add-Type $a",
        49,
        38,
        vec![
            gap(49, (8, 36), RegionKind::ControlFlow),
            gap(49, (21, 34), RegionKind::ControlFlow),
            gap(49, (26, 34), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_coalescing_assignment_target_is_killed_with_precise_rhs,
        "$a='X'; while ($p) { $a??=($c='Y') }; Add-Type $a",
        49,
        38,
        vec![
            gap(49, (8, 36), RegionKind::ControlFlow),
            gap(49, (21, 34), RegionKind::ControlFlow),
            gap(49, (26, 34), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_coalescing_assignment_rhs_equal_write_is_killed_not_joined,
        "$a='X'; while ($p) { $b??=($a='X') }; Add-Type $a",
        49,
        38,
        vec![
            gap(49, (8, 36), RegionKind::ControlFlow),
            gap(49, (21, 34), RegionKind::ControlFlow),
            gap(49, (26, 34), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_nested_loops_coalescing_header_and_rhs_preserve_unwritten_key,
        "$a='X'; while ($p) { while ($q ?? ($b='Y')) { $c??=($d='Z') } }; Add-Type $a",
        76,
        65,
        vec![
            gap(76, (8, 63), RegionKind::ControlFlow),
            gap(76, (21, 61), RegionKind::ControlFlow),
            gap(76, (28, 42), RegionKind::Expression),
            gap(76, (34, 42), RegionKind::Expression),
            gap(76, (46, 59), RegionKind::ControlFlow),
            gap(76, (51, 59), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_nested_loops_coalescing_rhs_contributes_deep_equal_write,
        "$a='X'; while ($p) { while ($q ?? ($b='Y')) { $c??=($a='X') } }; Add-Type $a",
        76,
        65,
        vec![
            gap(76, (8, 63), RegionKind::ControlFlow),
            gap(76, (21, 61), RegionKind::ControlFlow),
            gap(76, (28, 42), RegionKind::Expression),
            gap(76, (34, 42), RegionKind::Expression),
            gap(76, (46, 59), RegionKind::ControlFlow),
            gap(76, (51, 59), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_if_coalescing_assignment_preserves_key_through_silent_join,
        "$a='X'; if ($p) { $b??=($c='Y') } else { }; Add-Type $a",
        55,
        44,
        vec![
            gap(55, (8, 42), RegionKind::ControlFlow),
            gap(55, (18, 31), RegionKind::ControlFlow),
            gap(55, (23, 31), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_if_loop_conditional_header_and_coalescing_rhs_preserve_joined_key,
        "$a='X'; if ($p) { while ($q -and ($b='Y')) { $c??=($d='Z') } } else { }; Add-Type $a",
        84,
        73,
        vec![
            gap(84, (8, 71), RegionKind::ControlFlow),
            gap(84, (18, 60), RegionKind::ControlFlow),
            gap(84, (25, 41), RegionKind::Expression),
            gap(84, (33, 41), RegionKind::Expression),
            gap(84, (45, 58), RegionKind::ControlFlow),
            gap(84, (50, 58), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_if_loop_coalescing_rhs_write_prevents_inherited_key_from_joining,
        "$a='X'; if ($p) { while ($q -and ($b='Y')) { $c??=($a='X') } } else { }; Add-Type $a",
        84,
        73,
        vec![
            gap(84, (8, 71), RegionKind::ControlFlow),
            gap(84, (18, 60), RegionKind::ControlFlow),
            gap(84, (25, 41), RegionKind::Expression),
            gap(84, (33, 41), RegionKind::Expression),
            gap(84, (45, 58), RegionKind::ControlFlow),
            gap(84, (50, 58), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_do_until_coalescing_condition_and_assignment_preserve_unwritten_key,
        "$a='X'; do { $b??=($c='Y') } until ($q ?? 'Z'); Add-Type $a",
        59,
        48,
        vec![
            gap(59, (8, 46), RegionKind::ControlFlow),
            gap(59, (13, 26), RegionKind::ControlFlow),
            gap(59, (18, 26), RegionKind::Expression),
            gap(59, (36, 45), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_for_conditional_headers_and_coalescing_update_preserve_unwritten_key,
        "$a='X'; for ($b=$p ? 'Y' : 'Z'; $q -and $r; $c??=($d++)) { }; Add-Type $a",
        73,
        62,
        vec![
            gap(73, (8, 60), RegionKind::ControlFlow),
            gap(73, (16, 30), RegionKind::ControlFlow),
            gap(73, (32, 42), RegionKind::Expression),
            gap(73, (44, 55), RegionKind::ControlFlow),
            gap(73, (49, 55), RegionKind::Expression),
            gap(73, (50, 54), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_foreach_parenthesized_coalescing_iterable_preserves_unwritten_key,
        "$a='X'; foreach ($b in ($p ?? ($c='Y'))) { }; Add-Type $a",
        57,
        46,
        vec![
            gap(57, (8, 44), RegionKind::ControlFlow),
            gap(57, (23, 39), RegionKind::Expression),
            gap(57, (24, 38), RegionKind::Expression),
            gap(57, (30, 38), RegionKind::Expression),
        ]
    );
    transfer_case!(
        analyze_coalescing_assignment_whole_barrier_allows_later_literal_restore_only,
        "$a='X'; $b??=(Mystery); $a='Z'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                43,
                positional((32, 43), (32, 40), (41, 43)),
                "Z",
                literal((27, 30), (28, 29)),
                vec![assignment((24, 30), (24, 26), (27, 30), "$a")],
                vec![reference((41, 43), "$a")],
            )],
            coverage: vec![
                gap(43, (8, 22), RegionKind::ControlFlow),
                gap(43, (13, 22), RegionKind::Expression),
                gap(43, (14, 21), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_coalescing_assignment_selective_kill_does_not_leak_rhs_provenance_after_restore,
        "$a='X'; $b??=($a='X'); $a='Z'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                42,
                positional((31, 42), (31, 39), (40, 42)),
                "Z",
                literal((26, 29), (27, 28)),
                vec![assignment((23, 29), (23, 25), (26, 29), "$a")],
                vec![reference((40, 42), "$a")],
            )],
            coverage: vec![
                gap(42, (8, 21), RegionKind::ControlFlow),
                gap(42, (13, 21), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_coalescing_assignment_nested_conditional_sites_are_unresolved_once_in_source_order,
        "$a='X'; $b??=($p ? (Add-Type 'T') : ($q -or (Add-Type 'F'))); Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(73, positional((20, 32), (20, 28), (29, 32))),
                conditional_site(73, positional((45, 57), (45, 53), (54, 57))),
                unknown(
                    73,
                    positional((62, 73), (62, 70), (71, 73)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(73, (8, 60), RegionKind::ControlFlow),
                gap(73, (13, 60), RegionKind::Expression),
                gap(73, (14, 59), RegionKind::ControlFlow),
                gap(73, (19, 33), RegionKind::Expression),
                gap(73, (36, 59), RegionKind::Expression),
                gap(73, (37, 58), RegionKind::Expression),
                gap(73, (44, 58), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_if_loop_conditional_header_and_coalescing_rhs_sites_are_not_rediscovered,
        "$a='X'; if ($p) { while ($q -and (Add-Type 'H')) { $b??=(Add-Type 'B') } } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(96, positional((34, 46), (34, 42), (43, 46))),
                conditional_site(96, positional((57, 69), (57, 65), (66, 69))),
                unknown(
                    96,
                    positional((85, 96), (85, 93), (94, 96)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(96, (8, 83), RegionKind::ControlFlow),
                gap(96, (18, 72), RegionKind::ControlFlow),
                gap(96, (25, 47), RegionKind::Expression),
                gap(96, (33, 47), RegionKind::Expression),
                gap(96, (51, 70), RegionKind::ControlFlow),
                gap(96, (56, 70), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_coalescing_assignment_deferred_nested_definition_is_not_executed,
        "$a='X'; $b??={ function f { $a='Y'; Add-Type 'D'; return } }; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                73,
                positional((62, 73), (62, 70), (71, 73)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((71, 73), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: nested_context(
                    nested_context(script_context(73), ContextKind::ScriptBlock, (13, 60)),
                    ContextKind::Function("f".into()),
                    (15, 58),
                ),
                ..conditional_site(73, positional((36, 48), (36, 44), (45, 48)))
            }],
            coverage: vec![
                gap(73, (8, 60), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(13, 60),
                    reason: DiscoveryReason::DeferredContext,
                    context: nested_context(script_context(73), ContextKind::ScriptBlock, (13, 60)),
                },
                CoverageGap {
                    span: Span::new(15, 58),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: nested_context(
                        nested_context(script_context(73), ContextKind::ScriptBlock, (13, 60)),
                        ContextKind::Function("f".into()),
                        (15, 58),
                    ),
                },
                CoverageGap {
                    context: nested_context(
                        nested_context(script_context(73), ContextKind::ScriptBlock, (13, 60)),
                        ContextKind::Function("f".into()),
                        (15, 58),
                    ),
                    ..gap(73, (50, 56), RegionKind::ControlFlow)
                },
            ],
        }
    );
    transfer_case!(
        analyze_coalescing_assignment_bom_unicode_preserves_inherited_original_byte_provenance,
        "\u{feff}$ŻCode='λ'; $b??=($p ? ($c='Y') : 'Z'); Add-Type $Żcode",
        CoreReport {
            confirmed: vec![known(
                61,
                positional((45, 61), (45, 53), (54, 61)),
                "λ",
                literal((11, 15), (12, 14)),
                vec![assignment((3, 15), (3, 10), (11, 15), "$ŻCode")],
                vec![reference((54, 61), "$Żcode")],
            )],
            coverage: vec![
                gap(61, (17, 43), RegionKind::ControlFlow),
                gap(61, (22, 43), RegionKind::Expression),
                gap(61, (23, 42), RegionKind::ControlFlow),
                gap(61, (28, 36), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_if_coalescing_assignment_does_not_weaken_flow_tail_barrier,
        "$a='X'; if ($p) { $b??=(++$c); return } else { }; $a='Z'; Add-Type $a",
        CoreReport {
            unresolved: vec![conditional_site(
                69,
                positional((58, 69), (58, 66), (67, 69)),
            )],
            coverage: vec![
                gap(69, (8, 48), RegionKind::ControlFlow),
                gap(69, (18, 29), RegionKind::ControlFlow),
                gap(69, (23, 29), RegionKind::Expression),
                gap(69, (24, 28), RegionKind::Expression),
                gap(69, (31, 37), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    #[test]
    fn analyze_coalescing_assignment_malformed_conditional_rhs_returns_original_syntax_error() {
        let source = "\u{feff}$b??=(<# Ż";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![LexError {
                span: Span::new(9, 14),
                message: "unterminated block comment".into(),
            }],
            parse_errors: vec![ParseError {
                span: Span::new(14, 14),
                message: "expected ')'".into(),
            }],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    loop_unknown_case!(
        analyze_coalescing_assignment_rhs_literal_write_does_not_install_inner_target,
        "$a='X'; $b??=($c='Y'); Add-Type $c",
        34,
        23,
        vec![
            gap(34, (8, 21), RegionKind::ControlFlow),
            gap(34, (13, 21), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_equal_conditional_literals_do_not_install_assignment_result,
        "$a='X'; while ($p) { $b=$q ? 'X' : 'X' }; Add-Type $b",
        53,
        42,
        vec![
            gap(53, (8, 40), RegionKind::ControlFlow),
            gap(53, (24, 38), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_index_target_remains_whole_barrier,
        "$a='X'; $b[0]??=($c='Y'); Add-Type $a",
        37,
        26,
        vec![
            gap(37, (8, 24), RegionKind::UnsupportedWrite),
            gap(37, (8, 13), RegionKind::Expression),
            gap(37, (16, 24), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_qualified_target_is_not_a_plain_key,
        "$a='X'; $script:b??=($c='Y'); Add-Type $a",
        41,
        30,
        vec![
            gap(41, (8, 28), RegionKind::UnsupportedWrite),
            gap(41, (8, 17), RegionKind::VariableRead),
            gap(41, (20, 28), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_braced_target_is_not_a_plain_key,
        "$a='X'; ${b}??=($c='Y'); Add-Type $a",
        36,
        25,
        vec![
            gap(36, (8, 23), RegionKind::UnsupportedWrite),
            gap(36, (8, 12), RegionKind::VariableRead),
            gap(36, (15, 23), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_automatic_target_remains_whole_barrier,
        "$a='X'; $_??=($c='Y'); Add-Type $a",
        34,
        23,
        vec![
            gap(34, (8, 21), RegionKind::UnsupportedWrite),
            gap(34, (8, 10), RegionKind::VariableRead),
            gap(34, (13, 21), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_coalescing_assignment_qualified_nonprovider_read_preserves_unrelated_key,
        "$a='X'; $b??=$script:a; Add-Type $a",
        35,
        24,
        vec![gap(35, (13, 22), RegionKind::VariableRead)]
    );
    loop_unknown_case!(
        analyze_loop_xor_header_still_dominates_precise_coalescing_rhs_writes,
        "$a='X'; while ($p -xor $q) { $b??=($c='Y') }; Add-Type $a",
        57,
        46,
        vec![
            gap(57, (8, 44), RegionKind::ControlFlow),
            gap(57, (15, 25), RegionKind::Expression),
            gap(57, (29, 42), RegionKind::ControlFlow),
            gap(57, (34, 42), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_coalescing_assignment_unsupported_rhs_still_dominates_known_target,
        "$a='X'; while ($p) { $b??=('Y'+'Z') }; Add-Type $a",
        50,
        39,
        vec![
            gap(50, (8, 37), RegionKind::ControlFlow),
            gap(50, (21, 35), RegionKind::ControlFlow),
            gap(50, (26, 35), RegionKind::Expression),
            gap(50, (27, 34), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_for_coalescing_update_mutation_is_killed_even_without_iterations,
        "$a='X'; for ($b=$p ? 'Y' : 'Z'; $q -and $r; $c??=($a++)) { }; Add-Type $a",
        73,
        62,
        vec![
            gap(73, (8, 60), RegionKind::ControlFlow),
            gap(73, (16, 30), RegionKind::ControlFlow),
            gap(73, (32, 42), RegionKind::Expression),
            gap(73, (44, 55), RegionKind::ControlFlow),
            gap(73, (49, 55), RegionKind::Expression),
            gap(73, (50, 54), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_assignment_and_rhs_includes_potential_equal_write,
        "$a='X'; $b??=($p -and ($a='X')); Add-Type $a",
        44,
        33,
        vec![
            gap(44, (8, 31), RegionKind::ControlFlow),
            gap(44, (13, 31), RegionKind::Expression),
            gap(44, (14, 30), RegionKind::Expression),
            gap(44, (22, 30), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_coalescing_assignment_coalescing_rhs_preserves_unwritten_key,
        "$a='X'; $b??=($p ?? ($c='Y')); Add-Type $a",
        42,
        31,
        vec![
            gap(42, (8, 29), RegionKind::ControlFlow),
            gap(42, (13, 29), RegionKind::Expression),
            gap(42, (14, 28), RegionKind::Expression),
            gap(42, (20, 28), RegionKind::Expression),
        ]
    );

    loop_preserves_case!(
        analyze_ternary_unrelated_result_assignment_preserves_inherited_constant,
        "$a='X'; $b=$p ? 'Y' : 'Z'; Add-Type $a",
        38,
        27,
        vec![gap(38, (11, 25), RegionKind::ControlFlow)]
    );
    loop_preserves_case!(
        analyze_ternary_read_operands_do_not_invalidate_inherited_key,
        "$a='X'; $p ? $a : 'Y'; Add-Type $a",
        34,
        23,
        vec![gap(34, (8, 21), RegionKind::ControlFlow)]
    );
    loop_preserves_case!(
        analyze_mixed_case_and_read_operands_preserve_inherited_key,
        "$a='X'; $p -AnD $a; Add-Type $a",
        31,
        20,
        vec![gap(31, (8, 18), RegionKind::Expression)]
    );
    loop_preserves_case!(
        analyze_mixed_case_or_read_operands_preserve_inherited_key,
        "$a='X'; $p -oR $a; Add-Type $a",
        30,
        19,
        vec![gap(30, (8, 17), RegionKind::Expression)]
    );
    loop_preserves_case!(
        analyze_coalescing_read_operands_preserve_inherited_key,
        "$a='X'; $a ?? 'Y'; Add-Type $a",
        30,
        19,
        vec![gap(30, (8, 17), RegionKind::Expression)]
    );
    loop_preserves_case!(
        analyze_ternary_unrelated_arm_writes_preserve_inherited_key,
        "$a='X'; $p ? ($b='Y') : ($c+='Z'); Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (13, 21), RegionKind::Expression),
            gap(46, (24, 33), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_parenthesized_and_rhs_unrelated_write_preserves_inherited_key,
        "$a='X'; $b=($p -and ($c='Y')); Add-Type $a",
        42,
        31,
        vec![
            gap(42, (11, 29), RegionKind::Expression),
            gap(42, (12, 28), RegionKind::Expression),
            gap(42, (20, 28), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_parenthesized_or_rhs_unrelated_write_preserves_inherited_key,
        "$a='X'; $b=($p -or ($c='Y')); Add-Type $a",
        41,
        30,
        vec![
            gap(41, (11, 28), RegionKind::Expression),
            gap(41, (12, 27), RegionKind::Expression),
            gap(41, (19, 27), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_parenthesized_coalescing_rhs_unrelated_write_preserves_inherited_key,
        "$a='X'; $b=($p ?? ($c='Y')); Add-Type $a",
        40,
        29,
        vec![
            gap(40, (11, 27), RegionKind::Expression),
            gap(40, (12, 26), RegionKind::Expression),
            gap(40, (18, 26), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_nested_parens_around_ternary_preserve_unwritten_key,
        "$a='X'; $b=(($p ? ($c='Y') : ($c='Z'))); Add-Type $a",
        52,
        41,
        vec![
            gap(52, (11, 39), RegionKind::Expression),
            gap(52, (12, 38), RegionKind::Expression),
            gap(52, (13, 37), RegionKind::ControlFlow),
            gap(52, (18, 26), RegionKind::Expression),
            gap(52, (29, 37), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_condition_equal_assignment_invalidates_key,
        "$a='X'; ($a='X') ? 'Y' : 'Z'; Add-Type $a",
        41,
        30,
        vec![
            gap(41, (8, 28), RegionKind::ControlFlow),
            gap(41, (8, 16), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_true_arm_equal_assignment_invalidates_key,
        "$a='X'; $p ? ($a='X') : 'Z'; Add-Type $a",
        40,
        29,
        vec![
            gap(40, (8, 27), RegionKind::ControlFlow),
            gap(40, (13, 21), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_false_arm_ascii_folded_assignment_invalidates_key,
        "$a='X'; $p ? 'Y' : ($A='X'); Add-Type $a",
        40,
        29,
        vec![
            gap(40, (8, 27), RegionKind::ControlFlow),
            gap(40, (19, 27), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_and_rhs_compound_write_invalidates_key,
        "$a='X'; $p -and ($a+='X'); Add-Type $a",
        38,
        27,
        vec![
            gap(38, (8, 25), RegionKind::Expression),
            gap(38, (16, 25), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_or_left_prefix_increment_invalidates_key,
        "$a='X'; (++$a) -or $p; Add-Type $a",
        34,
        23,
        vec![
            gap(34, (8, 21), RegionKind::Expression),
            gap(34, (8, 14), RegionKind::Expression),
            gap(34, (9, 13), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_coalescing_rhs_postfix_increment_invalidates_key,
        "$a='X'; $p ?? ($a++); Add-Type $a",
        33,
        22,
        vec![
            gap(33, (8, 20), RegionKind::Expression),
            gap(33, (14, 20), RegionKind::Expression),
            gap(33, (15, 19), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_and_left_prefix_decrement_invalidates_key,
        "$a='X'; (--$a) -and $p; Add-Type $a",
        35,
        24,
        vec![
            gap(35, (8, 22), RegionKind::Expression),
            gap(35, (8, 14), RegionKind::Expression),
            gap(35, (9, 13), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_or_rhs_postfix_decrement_invalidates_key,
        "$a='X'; $p -or ($a--); Add-Type $a",
        34,
        23,
        vec![
            gap(34, (8, 21), RegionKind::Expression),
            gap(34, (15, 21), RegionKind::Expression),
            gap(34, (16, 20), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_literal_false_and_does_not_prune_rhs_write,
        "$a='X'; $false -and ($a='X'); Add-Type $a",
        41,
        30,
        vec![
            gap(41, (8, 28), RegionKind::Expression),
            gap(41, (8, 14), RegionKind::VariableRead),
            gap(41, (20, 28), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_literal_true_or_does_not_prune_rhs_write,
        "$a='X'; $true -or ($a='X'); Add-Type $a",
        39,
        28,
        vec![
            gap(39, (8, 26), RegionKind::Expression),
            gap(39, (8, 13), RegionKind::VariableRead),
            gap(39, (18, 26), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_known_coalescing_left_does_not_prune_rhs_write,
        "$a='X'; $a ?? ($a='X'); Add-Type $a",
        35,
        24,
        vec![
            gap(35, (8, 22), RegionKind::Expression),
            gap(35, (14, 22), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_literal_false_ternary_still_summarizes_false_arm_write,
        "$a='X'; $false ? 'Y' : ($a='X'); Add-Type $a",
        44,
        33,
        vec![
            gap(44, (8, 31), RegionKind::ControlFlow),
            gap(44, (8, 14), RegionKind::VariableRead),
            gap(44, (23, 31), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_assignment_rhs_provider_dominates_known_keys,
        "$a='X'; $p ? ($b=$env:P) : 'Z'; Add-Type $a",
        43,
        32,
        vec![
            gap(43, (8, 30), RegionKind::ControlFlow),
            gap(43, (13, 24), RegionKind::Expression),
            gap(43, (17, 23), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_unknown_command_dominates_unrelated_arm_write,
        "$a='X'; $p ? (Mystery) : ($b='X'); Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (13, 22), RegionKind::Expression),
            gap(46, (14, 21), RegionKind::OtherCommand),
            gap(46, (25, 33), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_member_target_is_whole_barrier_not_leaf_write,
        "$a='X'; $p ? ($b.X='Y') : 'Z'; Add-Type $a",
        42,
        31,
        vec![
            gap(42, (8, 29), RegionKind::ControlFlow),
            gap(42, (13, 23), RegionKind::Expression),
            gap(42, (14, 22), RegionKind::UnsupportedWrite),
            gap(42, (14, 18), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_index_mutation_is_whole_barrier_not_leaf_write,
        "$a='X'; $p ? (++$b[0]) : 'Z'; Add-Type $a",
        41,
        30,
        vec![
            gap(41, (8, 28), RegionKind::ControlFlow),
            gap(41, (13, 22), RegionKind::Expression),
            gap(41, (14, 21), RegionKind::UnsupportedWrite),
            gap(41, (16, 21), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_unsupported_binary_operand_kills_unrelated_key,
        "$a='X'; $p ? ('Y'+'Z') : 'Z'; Add-Type $a",
        41,
        30,
        vec![
            gap(41, (8, 28), RegionKind::ControlFlow),
            gap(41, (13, 22), RegionKind::Expression),
            gap(41, (14, 21), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_interpolation_kills_unrelated_key,
        "$a='X'; $p ? \"$q\" : 'Z'; Add-Type $a",
        36,
        25,
        vec![
            gap(36, (8, 23), RegionKind::ControlFlow),
            CoverageGap {
                span: Span::new(13, 17),
                reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                context: script_context(36),
            },
        ]
    );
    loop_unknown_case!(
        analyze_ternary_subexpression_kills_unrelated_key_without_executing_rhs,
        "$a='X'; $p ? $($b='Y') : 'Z'; Add-Type $a",
        41,
        30,
        vec![
            gap(41, (8, 28), RegionKind::ControlFlow),
            gap(41, (13, 22), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_xor_is_not_recognized_as_supported_conditional_expression,
        "$a='X'; $p -xor $q; Add-Type $a",
        31,
        20,
        vec![gap(31, (8, 18), RegionKind::Expression)]
    );
    loop_unknown_case!(
        analyze_ordinary_parenthesized_assignment_retains_legacy_barrier,
        "$a='X'; ($b='Y'); Add-Type $a",
        29,
        18,
        vec![gap(29, (8, 16), RegionKind::Expression)]
    );
    loop_unknown_case!(
        analyze_equal_ternary_literals_do_not_install_result_value,
        "$b=$p ? 'X' : 'X'; Add-Type $b",
        30,
        19,
        vec![gap(30, (3, 17), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_and_does_not_install_result_value,
        "$b=$p -and 'X'; Add-Type $b",
        27,
        16,
        vec![gap(27, (3, 14), RegionKind::Expression)]
    );
    loop_unknown_case!(
        analyze_or_does_not_install_result_value,
        "$b=$p -or 'X'; Add-Type $b",
        26,
        15,
        vec![gap(26, (3, 13), RegionKind::Expression)]
    );
    loop_unknown_case!(
        analyze_equal_coalescing_literals_do_not_install_result_value,
        "$b='X' ?? 'X'; Add-Type $b",
        26,
        15,
        vec![gap(26, (3, 13), RegionKind::Expression)]
    );
    transfer_case!(
        analyze_equal_ternary_argument_still_returns_unsupported_value,
        "Add-Type ($p ? 'X' : 'X')",
        CoreReport {
            unresolved: vec![unknown(
                25,
                positional((0, 25), (0, 8), (9, 25)),
                ValueReason::UnsupportedExpression
            )],
            coverage: vec![
                gap(25, (9, 25), RegionKind::Expression),
                gap(25, (10, 24), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    loop_preserves_case!(
        analyze_loop_conditional_header_preserves_unwritten_key_without_execution,
        "$a='X'; while ($p ? 'Y' : 'Z') { }; Add-Type $a",
        47,
        36,
        vec![
            gap(47, (8, 34), RegionKind::ControlFlow),
            gap(47, (15, 29), RegionKind::ControlFlow),
        ]
    );
    loop_preserves_case!(
        analyze_coalescing_assignment_conditional_rhs_preserves_unwritten_key,
        "$a='X'; $b??=($p ? 'Y' : 'Z'); Add-Type $a",
        42,
        31,
        vec![
            gap(42, (8, 29), RegionKind::ControlFlow),
            gap(42, (13, 29), RegionKind::Expression),
            gap(42, (14, 28), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_nested_or_in_ternary_contributes_rhs_write,
        "$a='X'; $p ? ($q -or ($a='X')) : 'Z'; Add-Type $a",
        49,
        38,
        vec![
            gap(49, (8, 36), RegionKind::ControlFlow),
            gap(49, (13, 30), RegionKind::Expression),
            gap(49, (14, 29), RegionKind::Expression),
            gap(49, (21, 29), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_nested_coalescing_in_and_preserves_unwritten_key,
        "$a='X'; $p -and ($q ?? ($b='Y')); Add-Type $a",
        45,
        34,
        vec![
            gap(45, (8, 32), RegionKind::Expression),
            gap(45, (16, 32), RegionKind::Expression),
            gap(45, (17, 31), RegionKind::Expression),
            gap(45, (23, 31), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_literal_false_ternary_does_not_prune_true_arm_write,
        "$a='X'; $false ? ($a='X') : 'Z'; Add-Type $a",
        44,
        33,
        vec![
            gap(44, (8, 31), RegionKind::ControlFlow),
            gap(44, (8, 14), RegionKind::VariableRead),
            gap(44, (17, 25), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_nested_assignment_rhs_contributes_its_own_write,
        "$a='X'; $p ? ($b=$a='X') : 'Z'; Add-Type $a",
        43,
        32,
        vec![
            gap(43, (8, 30), RegionKind::ControlFlow),
            gap(43, (13, 24), RegionKind::Expression),
        ]
    );
    loop_preserves_case!(
        analyze_ternary_unsupported_nonprovider_reads_are_not_guessed_writes,
        "$a='X'; $p ? $script:a : $true; Add-Type $a",
        43,
        32,
        vec![
            gap(43, (8, 30), RegionKind::ControlFlow),
            gap(43, (13, 22), RegionKind::VariableRead),
            gap(43, (25, 30), RegionKind::VariableRead),
        ]
    );
    transfer_case!(
        analyze_ternary_deferred_literal_block_writes_and_flow_do_not_affect_parent,
        "$a='X'; $p ? { $a='Y'; Add-Type 'D'; return } : 'Z'; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                64,
                positional((53, 64), (53, 61), (62, 64)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((62, 64), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: nested_context(script_context(64), ContextKind::ScriptBlock, (13, 45)),
                ..conditional_site(64, positional((23, 35), (23, 31), (32, 35)))
            }],
            coverage: vec![
                gap(64, (8, 51), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(13, 45),
                    reason: DiscoveryReason::DeferredContext,
                    context: nested_context(script_context(64), ContextKind::ScriptBlock, (13, 45)),
                },
                CoverageGap {
                    context: nested_context(script_context(64), ContextKind::ScriptBlock, (13, 45)),
                    ..gap(64, (37, 43), RegionKind::ControlFlow)
                },
            ],
        }
    );
    transfer_case!(
        analyze_ternary_nested_definition_site_remains_unresolved_in_both_deferred_contexts,
        "$a='X'; $p ? { function f { Add-Type 'D' } } : 'Z'; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                63,
                positional((52, 63), (52, 60), (61, 63)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((61, 63), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: nested_context(
                    nested_context(script_context(63), ContextKind::ScriptBlock, (13, 44)),
                    ContextKind::Function("f".into()),
                    (15, 42),
                ),
                ..conditional_site(63, positional((28, 40), (28, 36), (37, 40)))
            }],
            coverage: vec![
                gap(63, (8, 50), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(13, 44),
                    reason: DiscoveryReason::DeferredContext,
                    context: nested_context(script_context(63), ContextKind::ScriptBlock, (13, 44)),
                },
                CoverageGap {
                    span: Span::new(15, 42),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: nested_context(
                        nested_context(script_context(63), ContextKind::ScriptBlock, (13, 44)),
                        ContextKind::Function("f".into()),
                        (15, 42),
                    ),
                },
            ],
        }
    );
    transfer_case!(
        analyze_nested_ternary_and_commands_are_discovered_once_without_confirmations,
        "$a='X'; $p ? (Add-Type 'T') : ($q -and (Add-Type 'F')); Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(67, positional((14, 26), (14, 22), (23, 26))),
                conditional_site(67, positional((40, 52), (40, 48), (49, 52))),
                unknown(
                    67,
                    positional((56, 67), (56, 64), (65, 67)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(67, (8, 54), RegionKind::ControlFlow),
                gap(67, (13, 27), RegionKind::Expression),
                gap(67, (30, 54), RegionKind::Expression),
                gap(67, (31, 53), RegionKind::Expression),
                gap(67, (39, 53), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_and_command_keeps_expression_reason_and_invalidates_unrelated_key,
        "$a='X'; $p -and (Add-Type 'D'); Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((17, 29), (17, 25), (26, 29)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context: script_context(43),
                },
                unknown(
                    43,
                    positional((32, 43), (32, 40), (41, 43)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(43, (8, 30), RegionKind::Expression),
                gap(43, (16, 30), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_ternary_in_if_silent_transfer_does_not_duplicate_discovered_site,
        "$a='X'; if ($p) { $q ? (Add-Type 'D') : 'Z' } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(67, positional((24, 36), (24, 32), (33, 36))),
                unknown(
                    67,
                    positional((56, 67), (56, 64), (65, 67)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(67, (8, 54), RegionKind::ControlFlow),
                gap(67, (18, 43), RegionKind::ControlFlow),
                gap(67, (23, 37), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    loop_preserves_case!(
        analyze_if_conditional_condition_preserves_key_outside_effect_set,
        "$a='X'; if ($p -and ($b='Y')) { } else { }; Add-Type $a",
        55,
        44,
        vec![
            gap(55, (8, 42), RegionKind::ControlFlow),
            gap(55, (12, 28), RegionKind::Expression),
            gap(55, (20, 28), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_if_conditional_condition_kills_key_before_both_snapshots,
        "$a='X'; if ($p -and ($a='X')) { } else { }; Add-Type $a",
        55,
        44,
        vec![
            gap(55, (8, 42), RegionKind::ControlFlow),
            gap(55, (12, 28), RegionKind::Expression),
            gap(55, (20, 28), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_elseif_conditional_condition_kills_final_remaining_false_prefix,
        "$a='X'; if ($p) { } elseif ($q -or ($a='X')) { } else { }; Add-Type $a",
        70,
        59,
        vec![
            gap(70, (8, 57), RegionKind::ControlFlow),
            gap(70, (28, 43), RegionKind::Expression),
            gap(70, (35, 43), RegionKind::Expression),
        ]
    );
    transfer_case!(
        analyze_elseif_conditional_effects_preserve_completed_arm_and_exclude_inner_write_provenance,
        "$a='X'; if ($p) { } elseif ($q -or ($a='X')) { $a='X' } else { $a='X' }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((73, 84), (73, 81), (82, 84)),
                raw_body: "X".into(),
                literal_origins: vec![
                    literal((3, 6), (4, 5)),
                    literal((50, 53), (51, 52)),
                    literal((66, 69), (67, 68)),
                ],
                assignment_origins: vec![
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                    assignment((47, 53), (47, 49), (50, 53), "$a"),
                    assignment((63, 69), (63, 65), (66, 69), "$a"),
                ],
                reference_origins: vec![reference((82, 84), "$a")],
                context: script_context(84),
            }],
            coverage: vec![
                gap(84, (8, 71), RegionKind::ControlFlow),
                gap(84, (28, 43), RegionKind::Expression),
                gap(84, (35, 43), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    loop_unknown_case!(
        analyze_ternary_subexpression_declaration_is_not_a_deferred_literal,
        "$a='X'; $p ? $(class C {}) : 'Z'; Add-Type $a",
        45,
        34,
        vec![
            gap(45, (8, 32), RegionKind::ControlFlow),
            gap(45, (13, 26), RegionKind::Expression),
            CoverageGap {
                context: nested_context(
                    script_context(45),
                    ContextKind::Class("C".into()),
                    (15, 25)
                ),
                ..gap(45, (15, 25), RegionKind::Declaration)
            },
        ]
    );
    loop_unknown_case!(
        analyze_ternary_unmodeled_flow_does_not_allow_effect_precision,
        "$a='X'; $p ? $(break) : 'Z'; Add-Type $a",
        40,
        29,
        vec![
            gap(40, (8, 27), RegionKind::ControlFlow),
            gap(40, (13, 21), RegionKind::Expression),
            gap(40, (15, 20), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_ternary_keyword_block_is_not_skipped_as_literal_block,
        "$a='X'; $p ? $(begin { $b='Y' }) : 'Z'; Add-Type $a",
        51,
        40,
        vec![
            gap(51, (8, 38), RegionKind::ControlFlow),
            gap(51, (13, 32), RegionKind::Expression),
            CoverageGap {
                context: nested_context(
                    script_context(51),
                    ContextKind::KeywordBlock("begin".into()),
                    (15, 31),
                ),
                ..gap(51, (15, 31), RegionKind::KeywordBlock)
            },
        ]
    );
    transfer_case!(
        analyze_conditional_effects_preserve_bom_unicode_inherited_provenance,
        "\u{feff}$ŻCode='λ'; $b=$p ? 'Y' : 'Z'; Add-Type $Żcode",
        CoreReport {
            confirmed: vec![known(
                52,
                positional((36, 52), (36, 44), (45, 52)),
                "λ",
                literal((11, 15), (12, 14)),
                vec![assignment((3, 15), (3, 10), (11, 15), "$ŻCode")],
                vec![reference((45, 52), "$Żcode")],
            )],
            coverage: vec![gap(52, (20, 34), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    loop_preserves_case!(
        analyze_while_read_leaves_are_not_guessed_writes,
        "$a='X'; while ($a) { $b=$a }; Add-Type $a",
        41,
        30,
        vec![gap(41, (8, 28), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_while_zero_iterations_do_not_establish_body_constant,
        "while ($false) { $a='X' }; Add-Type $a",
        38,
        27,
        vec![
            gap(38, (0, 25), RegionKind::ControlFlow),
            gap(38, (7, 13), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_while_equal_body_write_still_invalidates_inherited_key,
        "$a='X'; while ($false) { $a='X' }; Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (15, 21), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_while_condition_write_invalidates_target,
        "$a='X'; while ($a='X') { }; Add-Type $a",
        39,
        28,
        vec![gap(39, (8, 26), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_do_while_does_not_execute_body_once_or_keep_equal_write,
        "$a='X'; do { $a='X' } while ($p); Add-Type $a",
        45,
        34,
        vec![gap(45, (8, 32), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_do_until_does_not_install_body_literal,
        "do { $a='X' } until ($true); Add-Type $a",
        40,
        29,
        vec![
            gap(40, (0, 27), RegionKind::ControlFlow),
            gap(40, (21, 26), RegionKind::VariableRead),
        ]
    );
    loop_preserves_case!(
        analyze_do_until_unrelated_write_preserves_inherited_constant,
        "$a='X'; do { $b='Y' } until ($p); Add-Type $a",
        45,
        34,
        vec![gap(45, (8, 32), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_do_while_condition_write_invalidates_target,
        "$a='X'; do { } while ($a='X'); Add-Type $a",
        42,
        31,
        vec![gap(42, (8, 29), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_for_initializer_does_not_install_even_equal_literal,
        "$a='X'; for ($a='X'; $false; ) { }; Add-Type $a",
        47,
        36,
        vec![
            gap(47, (8, 34), RegionKind::ControlFlow),
            gap(47, (21, 27), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_for_condition_write_invalidates_target,
        "$a='X'; for (; $a='X'; ) { }; Add-Type $a",
        41,
        30,
        vec![gap(41, (8, 28), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_for_update_invalidates_even_on_zero_iteration_path,
        "$a='X'; for (; $false; $a++) { }; Add-Type $a",
        45,
        34,
        vec![
            gap(45, (8, 32), RegionKind::ControlFlow),
            gap(45, (15, 21), RegionKind::VariableRead),
            gap(45, (23, 27), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_for_omitted_headers_do_not_execute_body,
        "$a='X'; for (;;) { $a='X' }; Add-Type $a",
        40,
        29,
        vec![gap(40, (8, 27), RegionKind::ControlFlow)]
    );
    loop_preserves_case!(
        analyze_for_supported_init_update_and_compound_writes_preserve_other_key,
        "$a='X'; for ($b=0; $p; ++$b) { $b+=1 }; Add-Type $a",
        51,
        40,
        vec![
            gap(51, (8, 38), RegionKind::ControlFlow),
            gap(51, (23, 27), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_foreach_iteration_target_is_a_write_even_with_empty_body,
        "$a='X'; foreach ($a in $items) { }; Add-Type $a",
        47,
        36,
        vec![gap(47, (8, 34), RegionKind::ControlFlow)]
    );
    loop_preserves_case!(
        analyze_foreach_iterable_read_is_not_a_write,
        "$a='X'; foreach ($b in $a) { $b='Y' }; Add-Type $a",
        50,
        39,
        vec![gap(50, (8, 37), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_foreach_iterable_assignment_invalidates_its_target,
        "$a='X'; foreach ($b in $a='Y') { }; Add-Type $a",
        47,
        36,
        vec![gap(47, (8, 34), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_foreach_unsupported_iteration_target_kills_whole_environment,
        "$a='X'; foreach ($script:b in $items) { }; Add-Type $a",
        54,
        43,
        vec![
            gap(54, (8, 41), RegionKind::ControlFlow),
            gap(54, (17, 26), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_nested_for_body_write_is_in_outer_kill_set,
        "$a='X'; while ($p) { for (;;) { $a='X' } }; Add-Type $a",
        55,
        44,
        vec![
            gap(55, (8, 42), RegionKind::ControlFlow),
            gap(55, (21, 40), RegionKind::ControlFlow),
        ]
    );
    loop_preserves_case!(
        analyze_nested_foreach_summarizes_roles_without_killing_read_only_key,
        "$a='X'; while ($p) { foreach ($b in $a) { $b++ } }; Add-Type $a",
        63,
        52,
        vec![
            gap(63, (8, 50), RegionKind::ControlFlow),
            gap(63, (21, 48), RegionKind::ControlFlow),
            gap(63, (42, 46), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_assignment_rhs_unknown_command_dominates_known_writes,
        "$a='X'; while ($p) { $b=Mystery; $a='X' }; Add-Type $a",
        54,
        43,
        vec![
            gap(54, (8, 41), RegionKind::ControlFlow),
            gap(54, (24, 31), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_while_provider_condition_kills_unrelated_binding,
        "$a='X'; while ($env:P) { $b='Y' }; Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (15, 21), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_loop_rhs_provider_read_is_not_just_a_target_write,
        "$a='X'; while ($p) { $b=$env:P; $a='X' }; Add-Type $a",
        53,
        42,
        vec![
            gap(53, (8, 40), RegionKind::ControlFlow),
            gap(53, (24, 30), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_foreach_unknown_iterable_kills_whole_environment,
        "$a='X'; foreach ($b in Mystery) { $b='X' }; Add-Type $a",
        55,
        44,
        vec![
            gap(55, (8, 42), RegionKind::ControlFlow),
            gap(55, (23, 30), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_for_unknown_initializer_kills_unrelated_key,
        "$a='X'; for (Mystery; $p; ) { $b='X' }; Add-Type $a",
        51,
        40,
        vec![
            gap(51, (8, 38), RegionKind::ControlFlow),
            gap(51, (13, 20), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_for_unknown_update_kills_unrelated_key,
        "$a='X'; for (; $p; Mystery) { $b='X' }; Add-Type $a",
        51,
        40,
        vec![
            gap(51, (8, 38), RegionKind::ControlFlow),
            gap(51, (19, 26), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_loop_member_write_does_not_guess_a_leaf_kill_set,
        "$a='X'; while ($p) { $b.X='Y' }; Add-Type $a",
        44,
        33,
        vec![
            gap(44, (8, 31), RegionKind::ControlFlow),
            gap(44, (21, 29), RegionKind::UnsupportedWrite),
            gap(44, (21, 25), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_indirect_mutation_kills_whole_environment,
        "$a='X'; while ($p) { ++$b[0] }; Add-Type $a",
        43,
        32,
        vec![
            gap(43, (8, 30), RegionKind::ControlFlow),
            gap(43, (21, 28), RegionKind::UnsupportedWrite),
            gap(43, (23, 28), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_unsupported_rhs_expression_kills_unrelated_key,
        "$a='X'; while ($p) { $b='Y'+'Z' }; Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (24, 31), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_interpolation_kills_unrelated_key,
        "$a='X'; while ($p) { $b=\"$q\" }; Add-Type $a",
        43,
        32,
        vec![
            gap(43, (8, 30), RegionKind::ControlFlow),
            CoverageGap {
                span: Span::new(24, 28),
                reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                context: script_context(43),
            },
        ]
    );
    loop_unknown_case!(
        analyze_loop_class_declaration_is_not_a_deferred_function,
        "$a='X'; while ($p) { class C {} }; Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            CoverageGap {
                context: nested_context(
                    script_context(46),
                    ContextKind::Class("C".into()),
                    (21, 31)
                ),
                ..gap(46, (21, 31), RegionKind::Declaration)
            },
        ]
    );
    loop_unknown_case!(
        analyze_loop_enum_declaration_is_not_a_deferred_function,
        "$a='X'; while ($p) { enum E { V } }; Add-Type $a",
        48,
        37,
        vec![
            gap(48, (8, 35), RegionKind::ControlFlow),
            CoverageGap {
                context: nested_context(
                    script_context(48),
                    ContextKind::Enum("E".into()),
                    (21, 33)
                ),
                ..gap(48, (21, 33), RegionKind::Declaration)
            },
        ]
    );
    loop_unknown_case!(
        analyze_loop_unmodeled_flow_keeps_fallback_despite_later_write,
        "$a='X'; while ($p) { break; $b='X' }; Add-Type $a",
        49,
        38,
        vec![
            gap(49, (8, 36), RegionKind::ControlFlow),
            gap(49, (21, 26), RegionKind::ControlFlow),
        ]
    );
    loop_preserves_case!(
        analyze_loop_conditional_read_and_unrelated_writes_preserve_key,
        "$a='X'; while ($p) { if ($q) { $b=$a } else { $b='Y' } }; Add-Type $a",
        69,
        58,
        vec![
            gap(69, (8, 56), RegionKind::ControlFlow),
            gap(69, (21, 54), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_loop_conditional_write_is_not_a_post_loop_join,
        "$a='X'; while ($p) { if ($q) { $a='X' } }; Add-Type $a",
        54,
        43,
        vec![
            gap(54, (8, 41), RegionKind::ControlFlow),
            gap(54, (21, 39), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_loop_conditional_rhs_command_forces_fallback,
        "$a='X'; while ($p) { if ($b=Mystery) { } }; Add-Type $a",
        55,
        44,
        vec![
            gap(55, (8, 42), RegionKind::ControlFlow),
            gap(55, (21, 40), RegionKind::ControlFlow),
            gap(55, (28, 35), RegionKind::OtherCommand),
        ]
    );
    loop_preserves_case!(
        analyze_loop_effect_free_coalescing_write_only_kills_target,
        "$a='X'; while ($p) { $b??='Y' }; Add-Type $a",
        44,
        33,
        vec![gap(44, (8, 31), RegionKind::ControlFlow)]
    );

    loop_unknown_case!(
        analyze_loop_unknown_rhs_command_kills_key_outside_ordinary_write_set,
        "$a='X'; while ($p) { $b=Mystery }; Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (24, 31), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_loop_rhs_provider_read_kills_key_outside_ordinary_write_set,
        "$a='X'; while ($p) { $b=$env:P }; Add-Type $a",
        45,
        34,
        vec![
            gap(45, (8, 32), RegionKind::ControlFlow),
            gap(45, (24, 30), RegionKind::VariableRead),
        ]
    );
    loop_unknown_case!(
        analyze_loop_ascii_folded_write_key_invalidates_same_variable,
        "$a='X'; while ($p) { $A='X' }; Add-Type $a",
        42,
        31,
        vec![gap(42, (8, 29), RegionKind::ControlFlow)]
    );

    loop_preserves_case!(
        analyze_loop_empty_write_set_preserves_constant,
        "$a='X'; while ($p) { }; Add-Type $a",
        35,
        24,
        vec![gap(35, (8, 22), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_loop_prefix_decrement_invalidates_target,
        "$a='X'; while ($p) { --$a }; Add-Type $a",
        40,
        29,
        vec![
            gap(40, (8, 27), RegionKind::ControlFlow),
            gap(40, (21, 25), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_postfix_decrement_invalidates_target,
        "$a='X'; while ($p) { $a-- }; Add-Type $a",
        40,
        29,
        vec![
            gap(40, (8, 27), RegionKind::ControlFlow),
            gap(40, (21, 25), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_loop_compound_assignment_invalidates_target,
        "$a='X'; while ($p) { $a+='X' }; Add-Type $a",
        43,
        32,
        vec![gap(43, (8, 30), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_loop_coalescing_assignment_invalidates_target_without_folding,
        "$a='X'; while ($p) { $a??='X' }; Add-Type $a",
        44,
        33,
        vec![gap(44, (8, 31), RegionKind::ControlFlow)]
    );

    macro_rules! loop_definition_case {
        ($name:ident, $source:literal, $end:literal, $loop_end:literal,
         $context:expr, $definition:expr, $definition_reason:expr,
         $site:literal, $flow:expr) => {
            transfer_case!(
                $name,
                $source,
                CoreReport {
                    hypotheses: vec![],
                    confirmed: vec![known(
                        $end,
                        positional(($end - 11, $end), ($end - 11, $end - 3), ($end - 2, $end)),
                        "X",
                        literal((3, 6), (4, 5)),
                        vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                        vec![reference(($end - 2, $end), "$a")],
                    )],
                    unresolved: vec![Unresolved {
                        context: $context,
                        ..conditional_site(
                            $end,
                            positional(
                                ($site, $site + 12),
                                ($site, $site + 8),
                                ($site + 9, $site + 12)
                            )
                        )
                    }],
                    coverage: vec![
                        gap($end, (8, $loop_end), RegionKind::ControlFlow),
                        CoverageGap {
                            span: Span::new($definition.0, $definition.1),
                            reason: $definition_reason,
                            context: $context,
                        },
                        CoverageGap {
                            context: $context,
                            ..gap($end, $flow, RegionKind::ControlFlow)
                        },
                    ],
                }
            );
        };
    }

    loop_definition_case!(
        analyze_loop_function_writes_commands_and_flow_are_deferred_but_site_is_discovered,
        "$a='X'; while ($p) { function f { $a='Y'; Add-Type 'D'; return } }; Add-Type $a",
        79,
        66,
        nested_context(
            script_context(79),
            ContextKind::Function("f".into()),
            (21, 64)
        ),
        (21, 64),
        DiscoveryReason::PendingFunctionAnalysis,
        42,
        (56, 62)
    );
    loop_definition_case!(
        analyze_loop_filter_writes_commands_and_flow_are_deferred_but_site_is_discovered,
        "$a='X'; while ($p) { filter f { $a='Y'; Add-Type 'D'; return } }; Add-Type $a",
        77,
        64,
        nested_context(
            script_context(77),
            ContextKind::Function("f".into()),
            (21, 62)
        ),
        (21, 62),
        DiscoveryReason::PendingFunctionAnalysis,
        40,
        (54, 60)
    );
    loop_definition_case!(
        analyze_loop_workflow_writes_commands_and_flow_are_deferred_but_site_is_discovered,
        "$a='X'; while ($p) { workflow w { $a='Y'; Add-Type 'D'; return } }; Add-Type $a",
        79,
        66,
        nested_context(
            script_context(79),
            ContextKind::Workflow("w".into()),
            (21, 64)
        ),
        (21, 64),
        DiscoveryReason::UnsupportedRegion(RegionKind::Workflow),
        42,
        (56, 62)
    );
    loop_definition_case!(
        analyze_loop_literal_block_writes_commands_and_flow_are_deferred_but_site_is_discovered,
        "$a='X'; while ($p) { $b={ $a='Y'; Add-Type 'D'; return } }; Add-Type $a",
        71,
        58,
        nested_context(script_context(71), ContextKind::ScriptBlock, (24, 56)),
        (24, 56),
        DiscoveryReason::DeferredContext,
        34,
        (48, 54)
    );
    loop_unknown_case!(
        analyze_loop_keyword_block_is_not_skipped_as_literal_block,
        "$a='X'; while ($p) { begin { $a='X' } }; Add-Type $a",
        52,
        41,
        vec![
            gap(52, (8, 39), RegionKind::ControlFlow),
            CoverageGap {
                context: nested_context(
                    script_context(52),
                    ContextKind::KeywordBlock("begin".into()),
                    (21, 37)
                ),
                ..gap(52, (21, 37), RegionKind::KeywordBlock)
            },
        ]
    );
    transfer_case!(
        analyze_loop_effectful_coalescing_rhs_is_discovered_not_executed,
        "$a='X'; while ($p) { $b??=(Add-Type 'D'); $a='X' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(63, positional((27, 39), (27, 35), (36, 39))),
                unknown(
                    63,
                    positional((52, 63), (52, 60), (61, 63)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(63, (8, 50), RegionKind::ControlFlow),
                gap(63, (21, 40), RegionKind::ControlFlow),
                gap(63, (26, 40), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_nested_loop_commands_are_unresolved_once_and_kill_unrelated_constant,
        "$a='X'; while ($p) { while ($q) { Add-Type 'I' }; Add-Type 'O' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(77, positional((34, 46), (34, 42), (43, 46))),
                conditional_site(77, positional((50, 62), (50, 58), (59, 62))),
                unknown(
                    77,
                    positional((66, 77), (66, 74), (75, 77)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(77, (8, 64), RegionKind::ControlFlow),
                gap(77, (21, 48), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_for_all_header_and_body_sites_are_discovered_once_without_transfers,
        "for ($a=Add-Type 'I'; Add-Type 'C'; Add-Type 'U') { Add-Type 'B' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(79, positional((8, 20), (8, 16), (17, 20))),
                conditional_site(79, positional((22, 34), (22, 30), (31, 34))),
                conditional_site(79, positional((36, 48), (36, 44), (45, 48))),
                conditional_site(79, positional((52, 64), (52, 60), (61, 64))),
                unknown(
                    79,
                    positional((68, 79), (68, 76), (77, 79)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![gap(79, (0, 66), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_after_loop_literal_restore_has_only_new_provenance,
        "$a='X'; while ($p) { $a='Y' }; $a='Z'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                50,
                positional((39, 50), (39, 47), (48, 50)),
                "Z",
                literal((34, 37), (35, 36)),
                vec![assignment((31, 37), (31, 33), (34, 37), "$a")],
                vec![reference((48, 50), "$a")],
            )],
            coverage: vec![gap(50, (8, 29), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_after_whole_loop_fallback_literal_restore_has_only_new_provenance,
        "$a='X'; while (Mystery) { $a='Y' }; $a='Z'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                55,
                positional((44, 55), (44, 52), (53, 55)),
                "Z",
                literal((39, 42), (40, 41)),
                vec![assignment((36, 42), (36, 38), (39, 42), "$a")],
                vec![reference((53, 55), "$a")],
            )],
            coverage: vec![
                gap(55, (8, 34), RegionKind::ControlFlow),
                gap(55, (15, 22), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    loop_preserves_case!(
        analyze_conditional_loop_kill_set_preserves_key_in_join,
        "$a='X'; if ($p) { while ($q) { $b='Y' } } else { }; Add-Type $a",
        63,
        52,
        vec![
            gap(63, (8, 50), RegionKind::ControlFlow),
            gap(63, (18, 39), RegionKind::ControlFlow),
        ]
    );
    transfer_case!(
        analyze_conditional_loop_command_is_not_rediscovered_during_silent_transfer,
        "$a='X'; if ($p) { while ($q) { Add-Type 'I' } } else { $a='X' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(76, positional((31, 43), (31, 39), (40, 43))),
                unknown(
                    76,
                    positional((65, 76), (65, 73), (74, 76)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(76, (8, 63), RegionKind::ControlFlow),
                gap(76, (18, 45), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );

    // Composition cases distinguish potential effects from body execution.
    transfer_case!(
        analyze_try_empty_finally_preserves_inherited_literal_with_original_provenance,
        "$a='X'; try { } finally { }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                40,
                positional((29, 40), (29, 37), (38, 40)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((38, 40), "$a")],
            )],
            coverage: vec![gap(40, (8, 27), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_empty_catch_finally_does_not_prevent_later_direct_literal_confirmation,
        "try { } catch { } finally { }; Add-Type 'X'",
        CoreReport {
            confirmed: vec![known(
                43,
                positional((31, 43), (31, 39), (40, 43)),
                "X",
                literal((40, 43), (41, 42)),
                vec![],
                vec![],
            )],
            coverage: vec![
                gap(43, (0, 29), RegionKind::ControlFlow),
                gap(43, (8, 17), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_second_catch_write_is_not_pruned_or_restored_by_unrelated_finally,
        "$a='X'; try { } catch { $b='Y' } catch { $a='X' } finally { $b='Z' }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                81,
                positional((70, 81), (70, 78), (79, 81)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(81, (8, 68), RegionKind::ControlFlow),
                gap(81, (16, 32), RegionKind::ControlFlow),
                gap(81, (33, 49), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_nested_loop_if_try_unrelated_writes_preserve_unwritten_key,
        "$a='X'; try { while ($p) { if ($q) { try { $b='Y' } finally { } } } } catch { }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                92,
                positional((81, 92), (81, 89), (90, 92)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((90, 92), "$a")],
            )],
            coverage: vec![
                gap(92, (8, 79), RegionKind::ControlFlow),
                gap(92, (14, 67), RegionKind::ControlFlow),
                gap(92, (27, 65), RegionKind::ControlFlow),
                gap(92, (37, 63), RegionKind::ControlFlow),
                gap(92, (70, 79), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_nested_loop_if_try_equal_write_invalidates_inherited_key,
        "$a='X'; try { while ($p) { if ($q) { try { $a='X' } finally { } } } } catch { }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                92,
                positional((81, 92), (81, 89), (90, 92)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(92, (8, 79), RegionKind::ControlFlow),
                gap(92, (14, 67), RegionKind::ControlFlow),
                gap(92, (27, 65), RegionKind::ControlFlow),
                gap(92, (37, 63), RegionKind::ControlFlow),
                gap(92, (70, 79), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_ternary_arm_writes_preserve_key_outside_both_arms,
        "$a='X'; try { $p ? ($b='Y') : ($c='Z') } catch { }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                63,
                positional((52, 63), (52, 60), (61, 63)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((61, 63), "$a")],
            )],
            coverage: vec![
                gap(63, (8, 50), RegionKind::ControlFlow),
                gap(63, (14, 38), RegionKind::ControlFlow),
                gap(63, (19, 27), RegionKind::Expression),
                gap(63, (30, 38), RegionKind::Expression),
                gap(63, (41, 50), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_ternary_false_arm_equal_write_invalidates_inherited_key,
        "$a='X'; try { $p ? ($b='Y') : ($a='X') } catch { }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                63,
                positional((52, 63), (52, 60), (61, 63)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(63, (8, 50), RegionKind::ControlFlow),
                gap(63, (14, 38), RegionKind::ControlFlow),
                gap(63, (19, 27), RegionKind::Expression),
                gap(63, (30, 38), RegionKind::Expression),
                gap(63, (41, 50), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_catch_coalescing_rhs_write_preserves_unwritten_key,
        "$a='X'; try { } catch { $b??=($c='Y') }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                52,
                positional((41, 52), (41, 49), (50, 52)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((50, 52), "$a")],
            )],
            coverage: vec![
                gap(52, (8, 39), RegionKind::ControlFlow),
                gap(52, (16, 39), RegionKind::ControlFlow),
                gap(52, (24, 37), RegionKind::ControlFlow),
                gap(52, (29, 37), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_catch_coalescing_rhs_equal_write_invalidates_inherited_key,
        "$a='X'; try { } catch { $b??=($a='X') }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                52,
                positional((41, 52), (41, 49), (50, 52)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(52, (8, 39), RegionKind::ControlFlow),
                gap(52, (16, 39), RegionKind::ControlFlow),
                gap(52, (24, 37), RegionKind::ControlFlow),
                gap(52, (29, 37), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_if_try_silent_transfer_preserves_key_outside_try_write_set,
        "$a='X'; if ($p) { try { $b='Y' } finally { } } else { }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                68,
                positional((57, 68), (57, 65), (66, 68)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((66, 68), "$a")],
            )],
            coverage: vec![
                gap(68, (8, 55), RegionKind::ControlFlow),
                gap(68, (18, 44), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_if_try_silent_transfer_invalidates_written_key_before_join,
        "$a='X'; if ($p) { try { $a='X' } finally { } } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                68,
                positional((57, 68), (57, 65), (66, 68)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(68, (8, 55), RegionKind::ControlFlow),
                gap(68, (18, 44), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_if_try_silent_transfer_does_not_rediscover_body_or_catch_sites,
        "$a='X'; if ($p) { try { Add-Type 'T' } catch { Add-Type 'C' } } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(85, positional((24, 36), (24, 32), (33, 36))),
                conditional_site(85, positional((47, 59), (47, 55), (56, 59))),
                unknown(
                    85,
                    positional((74, 85), (74, 82), (83, 85)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(85, (8, 72), RegionKind::ControlFlow),
                gap(85, (18, 61), RegionKind::ControlFlow),
                gap(85, (39, 61), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_loop_try_summary_preserves_key_outside_nested_write_set,
        "$a='X'; while ($p) { try { $b='Y' } catch { } }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                60,
                positional((49, 60), (49, 57), (58, 60)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((58, 60), "$a")],
            )],
            coverage: vec![
                gap(60, (8, 47), RegionKind::ControlFlow),
                gap(60, (21, 45), RegionKind::ControlFlow),
                gap(60, (36, 45), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_loop_try_summary_invalidates_equal_nested_write_without_execution,
        "$a='X'; while ($p) { try { $a='X' } catch { } }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                60,
                positional((49, 60), (49, 57), (58, 60)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(60, (8, 47), RegionKind::ControlFlow),
                gap(60, (21, 45), RegionKind::ControlFlow),
                gap(60, (36, 45), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_loop_try_summary_discovers_try_and_finally_sites_once_without_confirmation,
        "$a='X'; while ($p) { try { Add-Type 'T' } finally { Add-Type 'F' } }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(81, positional((27, 39), (27, 35), (36, 39))),
                conditional_site(81, positional((52, 64), (52, 60), (61, 64))),
                unknown(
                    81,
                    positional((70, 81), (70, 78), (79, 81)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(81, (8, 68), RegionKind::ControlFlow),
                gap(81, (21, 66), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_multiple_type_filters_dominate_unrelated_writes_and_unfiltered_catch,
        "$a='X'; try { } catch [Exception],[Other] { } catch { } finally { $b='Y' }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                87,
                positional((76, 87), (76, 84), (85, 87)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(87, (8, 74), RegionKind::ControlFlow),
                gap(87, (16, 45), RegionKind::ControlFlow),
                gap(87, (46, 55), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_skipped_command_header_does_not_invent_a_discovered_header_site,
        "$a='X'; try { } catch (Add-Type 'H') { Add-Type 'C' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(66, positional((39, 51), (39, 47), (48, 51))),
                unknown(
                    66,
                    positional((55, 66), (55, 63), (64, 66)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(66, (8, 53), RegionKind::ControlFlow),
                gap(66, (16, 53), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_if_try_catch_command_whole_effect_survives_silent_transfer_join,
        "$a='X'; if ($p) { try { } catch { $b=Mystery } } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                70,
                positional((59, 70), (59, 67), (68, 70)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(70, (8, 57), RegionKind::ControlFlow),
                gap(70, (18, 46), RegionKind::ControlFlow),
                gap(70, (26, 46), RegionKind::ControlFlow),
                gap(70, (37, 44), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_catch_provider_rhs_whole_effect_is_not_narrowed_to_assignment_target,
        "$a='X'; try { } catch { $b=$env:P } finally { $b='Y' }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                67,
                positional((56, 67), (56, 64), (65, 67)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(67, (8, 54), RegionKind::ControlFlow),
                gap(67, (16, 35), RegionKind::ControlFlow),
                gap(67, (27, 33), RegionKind::VariableRead),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_finally_qualified_target_whole_effect_does_not_guess_leaf_writes,
        "$a='X'; try { } finally { $script:b='Y' }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                54,
                positional((43, 54), (43, 51), (52, 54)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(54, (8, 41), RegionKind::ControlFlow),
                gap(54, (26, 39), RegionKind::UnsupportedWrite),
                gap(54, (26, 35), RegionKind::VariableRead),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_catch_flow_whole_effect_invalidates_unwritten_key,
        "$a='X'; try { } catch { break } finally { $b='Y' }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                63,
                positional((52, 63), (52, 60), (61, 63)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(63, (8, 50), RegionKind::ControlFlow),
                gap(63, (16, 31), RegionKind::ControlFlow),
                gap(63, (24, 29), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_function_writes_and_flow_are_deferred_with_unresolved_definition_site,
        "$a='X'; try { function f { $a='Y'; Add-Type 'D'; return } } catch { }; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                82,
                positional((71, 82), (71, 79), (80, 82)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((80, 82), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: nested_context(
                    script_context(82),
                    ContextKind::Function("f".into()),
                    (14, 57),
                ),
                ..conditional_site(82, positional((35, 47), (35, 43), (44, 47)))
            }],
            coverage: vec![
                gap(82, (8, 69), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(14, 57),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: nested_context(
                        script_context(82),
                        ContextKind::Function("f".into()),
                        (14, 57),
                    ),
                },
                CoverageGap {
                    context: nested_context(
                        script_context(82),
                        ContextKind::Function("f".into()),
                        (14, 57),
                    ),
                    ..gap(82, (49, 55), RegionKind::ControlFlow)
                },
                gap(82, (60, 69), RegionKind::ControlFlow),
            ],
        }
    );
    transfer_case!(
        analyze_try_filter_writes_and_flow_are_deferred_with_unresolved_definition_site,
        "$a='X'; try { filter f { $a='Y'; Add-Type 'D'; return } } catch { }; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                80,
                positional((69, 80), (69, 77), (78, 80)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((78, 80), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: nested_context(
                    script_context(80),
                    ContextKind::Function("f".into()),
                    (14, 55),
                ),
                ..conditional_site(80, positional((33, 45), (33, 41), (42, 45)))
            }],
            coverage: vec![
                gap(80, (8, 67), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(14, 55),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: nested_context(
                        script_context(80),
                        ContextKind::Function("f".into()),
                        (14, 55),
                    ),
                },
                CoverageGap {
                    context: nested_context(
                        script_context(80),
                        ContextKind::Function("f".into()),
                        (14, 55),
                    ),
                    ..gap(80, (47, 53), RegionKind::ControlFlow)
                },
                gap(80, (58, 67), RegionKind::ControlFlow),
            ],
        }
    );
    transfer_case!(
        analyze_try_workflow_writes_and_flow_are_deferred_with_unresolved_workflow_site,
        "$a='X'; try { workflow w { $a='Y'; Add-Type 'D'; return } } catch { }; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                82,
                positional((71, 82), (71, 79), (80, 82)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((80, 82), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: nested_context(
                    script_context(82),
                    ContextKind::Workflow("w".into()),
                    (14, 57),
                ),
                ..conditional_site(82, positional((35, 47), (35, 43), (44, 47)))
            }],
            coverage: vec![
                gap(82, (8, 69), RegionKind::ControlFlow),
                CoverageGap {
                    context: nested_context(
                        script_context(82),
                        ContextKind::Workflow("w".into()),
                        (14, 57),
                    ),
                    ..gap(82, (14, 57), RegionKind::Workflow)
                },
                CoverageGap {
                    context: nested_context(
                        script_context(82),
                        ContextKind::Workflow("w".into()),
                        (14, 57),
                    ),
                    ..gap(82, (49, 55), RegionKind::ControlFlow)
                },
                gap(82, (60, 69), RegionKind::ControlFlow),
            ],
        }
    );
    transfer_case!(
        analyze_try_literal_block_writes_and_flow_are_deferred_with_unresolved_block_site,
        "$a='X'; try { $b={ $a='Y'; Add-Type 'D'; return } } catch { }; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                74,
                positional((63, 74), (63, 71), (72, 74)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((72, 74), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: nested_context(script_context(74), ContextKind::ScriptBlock, (17, 49)),
                ..conditional_site(74, positional((27, 39), (27, 35), (36, 39)))
            }],
            coverage: vec![
                gap(74, (8, 61), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(17, 49),
                    reason: DiscoveryReason::DeferredContext,
                    context: nested_context(script_context(74), ContextKind::ScriptBlock, (17, 49)),
                },
                CoverageGap {
                    context: nested_context(script_context(74), ContextKind::ScriptBlock, (17, 49)),
                    ..gap(74, (41, 47), RegionKind::ControlFlow)
                },
                gap(74, (52, 61), RegionKind::ControlFlow),
            ],
        }
    );
    transfer_case!(
        analyze_after_try_key_barrier_fresh_assignment_has_only_fresh_provenance,
        "$a='X'; try { $a='Y' } finally { $a='X' }; $a='Z'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                62,
                positional((51, 62), (51, 59), (60, 62)),
                "Z",
                literal((46, 49), (47, 48)),
                vec![assignment((43, 49), (43, 45), (46, 49), "$a")],
                vec![reference((60, 62), "$a")],
            )],
            coverage: vec![gap(62, (8, 41), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_after_try_whole_barrier_fresh_assignment_has_only_fresh_provenance,
        "$a='X'; try { Mystery } finally { $a='X' }; $a='Z'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                63,
                positional((52, 63), (52, 60), (61, 63)),
                "Z",
                literal((47, 50), (48, 49)),
                vec![assignment((44, 50), (44, 46), (47, 50), "$a")],
                vec![reference((61, 63), "$a")],
            )],
            coverage: vec![
                gap(63, (8, 42), RegionKind::ControlFlow),
                gap(63, (14, 21), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_result_assignment_does_not_infer_body_or_finally_output_value,
        "$a='X'; $a=try { 'Y' } finally { 'Z' }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                51,
                positional((40, 51), (40, 48), (49, 51)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![gap(51, (11, 38), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_bom_utf8_literal_and_reference_spans_remain_original_bytes,
        "\u{feff}$ŻCode='λ'; try { $b='Y' } catch { } finally { }; Add-Type $Żcode",
        CoreReport {
            confirmed: vec![known(
                71,
                positional((55, 71), (55, 63), (64, 71)),
                "λ",
                literal((11, 15), (12, 14)),
                vec![assignment((3, 15), (3, 10), (11, 15), "$ŻCode")],
                vec![reference((64, 71), "$Żcode")],
            )],
            coverage: vec![
                gap(71, (17, 53), RegionKind::ControlFlow),
                gap(71, (32, 41), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );

    #[test]
    fn analyze_try_bom_utf8_missing_finally_body_preserves_exact_syntax_errors() {
        let source = "\u{feff}try { 'λ' } finally";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![],
            parse_errors: vec![
                ParseError {
                    span: Span::new(23, 23),
                    message: "expected '{'".into(),
                },
                ParseError {
                    span: Span::new(23, 23),
                    message: "expected '}'".into(),
                },
            ],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_try_bom_utf8_unterminated_string_keeps_lex_and_parse_diagnostics_separate() {
        let source = "\u{feff}try { 'λ";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![LexError {
                span: Span::new(9, 12),
                message: "unterminated string".into(),
            }],
            parse_errors: vec![ParseError {
                span: Span::new(12, 12),
                message: "expected '}'".into(),
            }],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    transfer_case!(
        analyze_try_nested_switch_remains_whole_barrier_despite_unrelated_case_write,
        "$a='X'; try { switch ($p) { 'L' { $b='Y' } } } catch { }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                69,
                positional((58, 69), (58, 66), (67, 69)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(69, (8, 56), RegionKind::ControlFlow),
                gap(69, (14, 44), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(32, 42),
                    reason: DiscoveryReason::DeferredContext,
                    context: nested_context(script_context(69), ContextKind::ScriptBlock, (32, 42)),
                },
                gap(69, (47, 56), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_switch_file_nested_try_does_not_narrow_whole_barrier_or_duplicate_input,
        "$a='X'; switch -File (Add-Type 'I') { 'L' { try { $b='Y' } finally { } } }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(87, positional((22, 34), (22, 30), (31, 34))),
                unknown(
                    87,
                    positional((76, 87), (76, 84), (85, 87)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(87, (8, 74), RegionKind::ControlFlow),
                gap(87, (21, 35), RegionKind::Expression),
                CoverageGap {
                    span: Span::new(42, 72),
                    reason: DiscoveryReason::DeferredContext,
                    context: nested_context(script_context(87), ContextKind::ScriptBlock, (42, 72)),
                },
                CoverageGap {
                    context: nested_context(script_context(87), ContextKind::ScriptBlock, (42, 72)),
                    ..gap(87, (44, 70), RegionKind::ControlFlow)
                },
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_labeled_loop_nested_try_retains_existing_whole_barrier,
        "$a='X'; :outer while ($p) { try { $b='Y' } catch { } }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                67,
                positional((56, 67), (56, 64), (65, 67)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(67, (8, 54), RegionKind::ControlFlow),
                gap(67, (15, 54), RegionKind::ControlFlow),
                gap(67, (28, 52), RegionKind::ControlFlow),
                gap(67, (43, 52), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_pipeline_chain_retains_whole_barrier_and_finally_cannot_restore,
        "$a='X'; try { Mystery || Add-Type 'P' } finally { $a='X' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(71, positional((25, 37), (25, 33), (34, 37))),
                unknown(
                    71,
                    positional((60, 71), (60, 68), (69, 71)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(71, (8, 58), RegionKind::ControlFlow),
                gap(71, (14, 37), RegionKind::ControlFlow),
                gap(71, (14, 21), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_pipeline_retains_whole_barrier_and_discovers_literal_site_once,
        "$a='X'; try { Mystery | Add-Type 'P' } catch { }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(61, positional((24, 36), (24, 32), (33, 36))),
                unknown(
                    61,
                    positional((50, 61), (50, 58), (59, 61)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(61, (8, 48), RegionKind::ControlFlow),
                gap(61, (14, 36), RegionKind::ControlFlow),
                gap(61, (14, 21), RegionKind::OtherCommand),
                gap(61, (39, 48), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_try_unrelated_body_writes_preserve_inherited_constant_and_origins,
        "$a='X'; try { $b='Y' } catch { $c='Z' }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                52,
                positional((41, 52), (41, 49), (50, 52)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((50, 52), "$a")],
            )],
            coverage: vec![
                gap(52, (8, 39), RegionKind::ControlFlow),
                gap(52, (23, 39), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    loop_preserves_case!(
        analyze_try_finally_unrelated_writes_preserve_inherited_constant,
        "$a='X'; try { $b='Y' } finally { $c='Z' }; Add-Type $a",
        54,
        43,
        vec![gap(54, (8, 41), RegionKind::ControlFlow)]
    );
    loop_preserves_case!(
        analyze_try_empty_bodies_preserve_inherited_constant,
        "$a='X'; try { } catch { } finally { }; Add-Type $a",
        50,
        39,
        vec![
            gap(50, (8, 37), RegionKind::ControlFlow),
            gap(50, (16, 25), RegionKind::ControlFlow),
        ]
    );
    loop_preserves_case!(
        analyze_try_multiple_unfiltered_catches_preserve_unwritten_key,
        "$a='X'; try { $b='Y' } catch { $c='Z' } catch { $d='W' }; Add-Type $a",
        69,
        58,
        vec![
            gap(69, (8, 56), RegionKind::ControlFlow),
            gap(69, (23, 39), RegionKind::ControlFlow),
            gap(69, (40, 56), RegionKind::ControlFlow),
        ]
    );
    loop_preserves_case!(
        analyze_try_catch_casing_and_unicode_comment_newline_are_not_a_filter,
        "$a='X'; TrY { $b='Y' } CaTcH <# λ #>\n{ $c='Z' }; Add-Type $a",
        61,
        50,
        vec![
            gap(61, (8, 48), RegionKind::ControlFlow),
            gap(61, (23, 48), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_body_same_literal_write_kills_inherited_key,
        "$a='X'; try { $a='X' } catch { $b='Y' } finally { $c='Z' }; Add-Type $a",
        71,
        60,
        vec![
            gap(71, (8, 58), RegionKind::ControlFlow),
            gap(71, (23, 39), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_catch_same_literal_write_kills_inherited_key,
        "$a='X'; try { $b='Y' } catch { $a='X' } finally { $c='Z' }; Add-Type $a",
        71,
        60,
        vec![
            gap(71, (8, 58), RegionKind::ControlFlow),
            gap(71, (23, 39), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_finally_same_literal_write_kills_inherited_key_without_restore,
        "$a='X'; try { $b='Y' } catch { $c='Z' } finally { $a='X' }; Add-Type $a",
        71,
        60,
        vec![
            gap(71, (8, 58), RegionKind::ControlFlow),
            gap(71, (23, 39), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_equal_literals_in_all_bodies_do_not_join_or_restore_value,
        "$a='X'; try { $a='X' } catch { $a='X' } finally { $a='X' }; Add-Type $a",
        71,
        60,
        vec![
            gap(71, (8, 58), RegionKind::ControlFlow),
            gap(71, (23, 39), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_finally_literal_does_not_install_new_key,
        "try { } finally { $a='X' }; Add-Type $a",
        39,
        28,
        vec![gap(39, (0, 26), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_try_filtered_catch_forces_whole_fallback_despite_finally_literal,
        "$a='X'; try { } catch [Exception] { } finally { $a='X' }; Add-Type $a",
        69,
        58,
        vec![
            gap(69, (8, 56), RegionKind::ControlFlow),
            gap(69, (16, 37), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_nonempty_unmodeled_catch_header_forces_whole_fallback,
        "$a='X'; try { } catch nonsense { }; Add-Type $a",
        47,
        36,
        vec![
            gap(47, (8, 34), RegionKind::ControlFlow),
            gap(47, (16, 34), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_without_handler_or_finally_forces_whole_fallback,
        "$a='X'; try { $b='Y' }; Add-Type $a",
        35,
        24,
        vec![gap(35, (8, 22), RegionKind::ControlFlow)]
    );
    loop_unknown_case!(
        analyze_try_unknown_command_dominates_writes_and_finally_cannot_restore,
        "$a='X'; try { Mystery; $a='X' } finally { $a='X' }; Add-Type $a",
        63,
        52,
        vec![
            gap(63, (8, 50), RegionKind::ControlFlow),
            gap(63, (14, 21), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_try_catch_unknown_command_invalidates_unrelated_key,
        "$a='X'; try { } catch { Mystery }; Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (16, 33), RegionKind::ControlFlow),
            gap(46, (24, 31), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_try_finally_unknown_command_invalidates_unrelated_key,
        "$a='X'; try { } finally { Mystery }; Add-Type $a",
        48,
        37,
        vec![
            gap(48, (8, 35), RegionKind::ControlFlow),
            gap(48, (26, 33), RegionKind::OtherCommand),
        ]
    );
    loop_unknown_case!(
        analyze_try_provider_read_invalidates_unrelated_key,
        "$a='X'; try { $env:PATH } catch { }; Add-Type $a",
        48,
        37,
        vec![
            gap(48, (8, 35), RegionKind::ControlFlow),
            gap(48, (14, 23), RegionKind::VariableRead),
            gap(48, (26, 35), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_unsupported_write_invalidates_unrelated_key_without_leaf_guessing,
        "$a='X'; try { $o.P='X' } catch { }; Add-Type $a",
        47,
        36,
        vec![
            gap(47, (8, 34), RegionKind::ControlFlow),
            gap(47, (14, 22), RegionKind::UnsupportedWrite),
            gap(47, (14, 18), RegionKind::Expression),
            gap(47, (25, 34), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_unsupported_rhs_invalidates_unrelated_key,
        "$a='X'; try { $b=1+2 } catch { }; Add-Type $a",
        45,
        34,
        vec![
            gap(45, (8, 32), RegionKind::ControlFlow),
            gap(45, (17, 20), RegionKind::Expression),
            gap(45, (23, 32), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_flow_forces_whole_fallback_and_finally_cannot_restore,
        "$a='X'; try { return } finally { $a='X' }; Add-Type $a",
        54,
        43,
        vec![
            gap(54, (8, 41), RegionKind::ControlFlow),
            gap(54, (14, 20), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_postfix_mutation_kills_target,
        "$a='X'; try { $a++ } catch { }; Add-Type $a",
        43,
        32,
        vec![
            gap(43, (8, 30), RegionKind::ControlFlow),
            gap(43, (14, 18), RegionKind::Expression),
            gap(43, (21, 30), RegionKind::ControlFlow),
        ]
    );
    loop_unknown_case!(
        analyze_try_catch_coalescing_assignment_kills_target_without_restore,
        "$a='X'; try { } catch { $a??='X' }; Add-Type $a",
        47,
        36,
        vec![
            gap(47, (8, 34), RegionKind::ControlFlow),
            gap(47, (16, 34), RegionKind::ControlFlow),
        ]
    );

    #[test]
    fn analyze_try_missing_catch_body_preserves_exact_parser_errors() {
        let source = "try {} catch";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![],
            parse_errors: vec![
                ParseError {
                    span: Span::new(12, 12),
                    message: "expected '{'".into(),
                },
                ParseError {
                    span: Span::new(12, 12),
                    message: "expected '}'".into(),
                },
            ],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_try_missing_finally_body_preserves_exact_parser_errors() {
        let source = "try {} finally";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![],
            parse_errors: vec![
                ParseError {
                    span: Span::new(14, 14),
                    message: "expected '{'".into(),
                },
                ParseError {
                    span: Span::new(14, 14),
                    message: "expected '}'".into(),
                },
            ],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_try_unclosed_catch_body_preserves_exact_parser_error() {
        let source = "try {} catch {";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![],
            parse_errors: vec![ParseError {
                span: Span::new(14, 14),
                message: "expected '}'".into(),
            }],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    fn empty_catch_node(body_first: usize, body_start: usize) -> Node {
        use super::super::span::TokenRange;

        let end = body_first + 2;
        let body_span = Span::new(body_start, body_start + 2);
        Node {
            kind: NodeKind::Catch {
                body: Box::new(Node {
                    kind: NodeKind::Script(vec![]),
                    span: body_span,
                    range: TokenRange {
                        first: body_first,
                        end,
                    },
                }),
            },
            span: Span::new(0, body_span.end),
            range: TokenRange { first: 0, end },
        }
    }

    const UNFILTERED_CATCH_SOURCE: &str = "catch {}";
    const FILTERED_CATCH_SOURCE: &str = "catch [E] {}";
    const CATCH_PAGE_WIDTH: usize = 2;

    struct CatchPages {
        source: &'static str,
        pages: &'static [[(TokenKind, Span); CATCH_PAGE_WIDTH]],
    }

    impl CatchTokenView for CatchPages {
        fn token_count(&self) -> usize {
            self.pages.len() * CATCH_PAGE_WIDTH
        }

        fn token_at(&self, index: usize) -> Option<CatchToken<'_>> {
            let page = self.pages.get(index / CATCH_PAGE_WIDTH)?;
            let (kind, span) = page[index % CATCH_PAGE_WIDTH];
            let value = self.source.get(span.start..span.end)?;
            Some(CatchToken { kind, span, value })
        }
    }

    const fn catch_record(kind: TokenKind, start: usize, end: usize) -> (TokenKind, Span) {
        (kind, Span { start, end })
    }

    // The unfiltered body's braces straddle the fixed page boundary.
    const UNFILTERED_CATCH_PAGES: CatchPages = CatchPages {
        source: UNFILTERED_CATCH_SOURCE,
        pages: &[
            [
                catch_record(TokenKind::Keyword, 0, 5),
                catch_record(TokenKind::LBrace, 6, 7),
            ],
            [
                catch_record(TokenKind::RBrace, 7, 8),
                catch_record(TokenKind::Eof, 8, 8),
            ],
        ],
    };
    const FILTERED_CATCH_PAGES: CatchPages = CatchPages {
        source: FILTERED_CATCH_SOURCE,
        pages: &[
            [
                catch_record(TokenKind::Keyword, 0, 5),
                catch_record(TokenKind::LBracket, 6, 7),
            ],
            [
                catch_record(TokenKind::Generic, 7, 8),
                catch_record(TokenKind::RBracket, 8, 9),
            ],
            [
                catch_record(TokenKind::LBrace, 10, 11),
                catch_record(TokenKind::RBrace, 11, 12),
            ],
        ],
    };

    // Successful entry parsing cannot produce inconsistent node ranges. These
    // fixtures exercise the defensive header boundary without reparsing it.
    fn catch_header_fixture() -> (Node, Vec<Token>) {
        (empty_catch_node(1, 6), lex(UNFILTERED_CATCH_SOURCE).tokens)
    }

    #[test]
    fn token_view_legacy_accepts_unfiltered_catch() {
        let (node, tokens) = catch_header_fixture();

        let actual = is_unfiltered_catch(&node, tokens.as_slice());

        assert!(actual, "expected true, got {actual}");
    }

    #[test]
    fn token_view_paged_accepts_unfiltered_catch_across_pages() {
        let node = empty_catch_node(1, 6);
        let tokens = UNFILTERED_CATCH_PAGES;

        let actual = is_unfiltered_catch(&node, &tokens);

        assert!(actual, "expected true, got {actual}");
    }

    #[test]
    fn token_view_legacy_rejects_filtered_catch() {
        let node = empty_catch_node(4, 10);
        let tokens = lex(FILTERED_CATCH_SOURCE).tokens;

        let actual = is_unfiltered_catch(&node, tokens.as_slice());

        assert!(!actual, "expected false, got {actual}");
    }

    #[test]
    fn token_view_paged_rejects_filtered_catch() {
        let node = empty_catch_node(4, 10);
        let tokens = FILTERED_CATCH_PAGES;

        let actual = is_unfiltered_catch(&node, &tokens);

        assert!(!actual, "expected false, got {actual}");
    }

    #[test]
    fn token_view_legacy_rejects_out_of_bounds_catch_range() {
        let (mut node, tokens) = catch_header_fixture();
        node.range.end = 5;

        let actual = is_unfiltered_catch(&node, tokens.as_slice());

        assert!(!actual, "expected false, got {actual}");
    }

    #[test]
    fn token_view_paged_rejects_out_of_bounds_catch_range() {
        let mut node = empty_catch_node(1, 6);
        let tokens = UNFILTERED_CATCH_PAGES;
        node.range.end = 5;

        let actual = is_unfiltered_catch(&node, &tokens);

        assert!(!actual, "expected false, got {actual}");
    }

    #[test]
    fn token_view_legacy_rejects_reversed_catch_range() {
        let (mut node, tokens) = catch_header_fixture();
        node.range.first = 2;
        node.range.end = 1;

        let actual = is_unfiltered_catch(&node, tokens.as_slice());

        assert!(!actual, "expected false, got {actual}");
    }

    #[test]
    fn token_view_paged_rejects_reversed_catch_range() {
        let mut node = empty_catch_node(1, 6);
        let tokens = UNFILTERED_CATCH_PAGES;
        node.range.first = 2;
        node.range.end = 1;

        let actual = is_unfiltered_catch(&node, &tokens);

        assert!(!actual, "expected false, got {actual}");
    }

    #[test]
    fn token_view_legacy_rejects_catch_with_stored_finally_value() {
        let (node, mut tokens) = catch_header_fixture();
        tokens[0].value = "finally".into();

        let actual = is_unfiltered_catch(&node, tokens.as_slice());

        assert!(!actual, "expected false, got {actual}");
    }

    macro_rules! catch_header_rejected_case {
        ($name:ident, $change:expr) => {
            #[test]
            fn $name() {
                let (mut node, mut tokens) = catch_header_fixture();
                $change(&mut node, &mut tokens);

                let actual = summarize_control_effects(&node, &tokens);

                assert_eq!(actual, WriteEffects::WholeEnvironment);
            }
        };
    }

    catch_header_rejected_case!(
        summarize_catch_empty_range_forces_whole_fallback,
        |node: &mut Node, _: &mut Vec<Token>| node.range.end = node.range.first
    );
    catch_header_rejected_case!(
        summarize_catch_reversed_range_forces_whole_fallback,
        |node: &mut Node, _: &mut Vec<Token>| node.range.first = 4
    );
    catch_header_rejected_case!(
        summarize_catch_out_of_bounds_range_forces_whole_fallback,
        |node: &mut Node, _: &mut Vec<Token>| node.range.end = 5
    );
    catch_header_rejected_case!(
        summarize_catch_nonkeyword_spelling_forces_whole_fallback,
        |_: &mut Node, tokens: &mut Vec<Token>| tokens[0].kind = TokenKind::Generic
    );
    catch_header_rejected_case!(
        summarize_catch_other_keyword_forces_whole_fallback,
        |_: &mut Node, tokens: &mut Vec<Token>| tokens[0].value = "finally".into()
    );
    catch_header_rejected_case!(
        summarize_catch_mismatched_body_start_index_forces_whole_fallback,
        |node: &mut Node, _: &mut Vec<Token>| {
            let NodeKind::Catch { body } = &mut node.kind else {
                unreachable!("fixture is a catch");
            };
            body.range.first = 2;
        }
    );
    catch_header_rejected_case!(
        summarize_catch_empty_body_range_forces_whole_fallback,
        |node: &mut Node, _: &mut Vec<Token>| {
            let NodeKind::Catch { body } = &mut node.kind else {
                unreachable!("fixture is a catch");
            };
            body.range.end = body.range.first;
        }
    );
    catch_header_rejected_case!(
        summarize_catch_non_script_body_forces_whole_fallback,
        |node: &mut Node, _: &mut Vec<Token>| {
            let NodeKind::Catch { body } = &mut node.kind else {
                unreachable!("fixture is a catch");
            };
            body.kind = NodeKind::BareWord("{}".into());
        }
    );

    transfer_case!(
        analyze_switch_flags_and_input_remain_a_whole_barrier_not_sequential_restore,
        "$a='X'; switch -Regex (Add-Type 'I') { $a='X' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(60, positional((23, 35), (23, 31), (32, 35))),
                unknown(
                    60,
                    positional((49, 60), (49, 57), (58, 60)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![gap(60, (8, 47), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_switch_file_flag_input_is_discovered_once_not_executed,
        "$a='X'; switch -File (Add-Type 'F') { $a='X' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(59, positional((22, 34), (22, 30), (31, 34))),
                unknown(
                    59,
                    positional((48, 59), (48, 56), (57, 59)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(59, (8, 46), RegionKind::ControlFlow),
                gap(59, (21, 35), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_try_catch_finally_sites_stay_unresolved_without_sequential_restores,
        "$a='X'; try { Add-Type 'T'; $a='X' } catch { Add-Type 'C'; $a='X' } finally { Add-Type 'F'; $a='X' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(113, positional((14, 26), (14, 22), (23, 26))),
                conditional_site(113, positional((45, 57), (45, 53), (54, 57))),
                conditional_site(113, positional((78, 90), (78, 86), (87, 90))),
                unknown(
                    113, positional((102, 113), (102, 110), (111, 113)), ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(113, (8, 100), RegionKind::ControlFlow),
                gap(113, (37, 67), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    loop_unknown_case!(
        analyze_ternary_equal_writes_invalidate_key_without_a_value_join,
        "$a='X'; $b=$p ? ($a='X') : ($a='X'); Add-Type $a",
        48,
        37,
        vec![
            gap(48, (11, 35), RegionKind::ControlFlow),
            gap(48, (16, 24), RegionKind::Expression),
            gap(48, (27, 35), RegionKind::Expression),
        ]
    );
    loop_unknown_case!(
        analyze_short_circuit_write_invalidates_key_without_sequential_transfer,
        "$a='X'; $b=$p -and ($a='X'); Add-Type $a",
        40,
        29,
        vec![
            gap(40, (11, 27), RegionKind::Expression),
            gap(40, (19, 27), RegionKind::Expression),
        ]
    );
    transfer_case!(
        analyze_pipeline_chain_keeps_conditional_literal_unresolved_and_kills_environment,
        "$a='X'; Mystery && Add-Type 'B'; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(44, positional((19, 31), (19, 27), (28, 31))),
                unknown(
                    44,
                    positional((33, 44), (33, 41), (42, 44)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(44, (8, 31), RegionKind::ControlFlow),
                gap(44, (8, 15), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    loop_unknown_case!(
        analyze_labeled_loop_keeps_whole_barrier_instead_of_precise_loop_transfer,
        "$a='X'; :outer while ($p) { $a='X' }; Add-Type $a",
        49,
        38,
        vec![
            gap(49, (8, 36), RegionKind::ControlFlow),
            gap(49, (15, 36), RegionKind::ControlFlow),
        ]
    );

    transfer_case!(
        analyze_equal_conditional_arms_join_all_literal_and_assignment_origins,
        "if ($x) { $a='X' } else { $a='X' }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((36, 47), (36, 44), (45, 47)),
                raw_body: "X".into(),
                literal_origins: vec![literal((13, 16), (14, 15)), literal((29, 32), (30, 31))],
                assignment_origins: vec![
                    assignment((10, 16), (10, 12), (13, 16), "$a"),
                    assignment((26, 32), (26, 28), (29, 32), "$a"),
                ],
                reference_origins: vec![reference((45, 47), "$a")],
                context: script_context(47),
            }],
            coverage: vec![gap(47, (0, 34), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    macro_rules! conditional_unknown_case {
        ($name:ident, $source:literal, $end:literal, $start:literal, $coverage:expr) => {
            transfer_case!(
                $name,
                $source,
                CoreReport {
                    unresolved: vec![unknown(
                        $end,
                        positional(($start, $end), ($start, $start + 8), ($end - 2, $end)),
                        ValueReason::VariableNotKnown,
                    )],
                    coverage: $coverage,
                    ..CoreReport::default()
                }
            );
        };
    }

    conditional_unknown_case!(
        analyze_different_conditional_raw_bodies_remove_joined_key,
        "if ($x) { $a='X' } else { $a='Y' }; Add-Type $a",
        47,
        36,
        vec![gap(47, (0, 34), RegionKind::ControlFlow)]
    );
    conditional_unknown_case!(
        analyze_unknown_conditional_copy_removes_joined_key,
        "if ($x) { $a='X' } else { $a=$b }; Add-Type $a",
        46,
        35,
        vec![gap(46, (0, 33), RegionKind::ControlFlow)]
    );
    conditional_unknown_case!(
        analyze_missing_conditional_key_removes_joined_key,
        "if ($x) { $a='X' } else { }; Add-Type $a",
        40,
        29,
        vec![gap(40, (0, 27), RegionKind::ControlFlow)]
    );
    conditional_unknown_case!(
        analyze_absent_else_includes_unknown_false_prefix,
        "if ($x) { $a='X' }; Add-Type $a",
        31,
        20,
        vec![gap(31, (0, 18), RegionKind::ControlFlow)]
    );
    conditional_unknown_case!(
        analyze_nonstring_conditional_value_removes_joined_key,
        "if ($x) { $a='X' } else { $a=1 }; Add-Type $a",
        45,
        34,
        vec![gap(45, (0, 32), RegionKind::ControlFlow)]
    );
    conditional_unknown_case!(
        analyze_literal_true_condition_does_not_prune_false_outcome,
        "if ($true) { $a='X' } else { }; Add-Type $a",
        43,
        32,
        vec![
            gap(43, (0, 30), RegionKind::ControlFlow),
            gap(43, (4, 9), RegionKind::VariableRead),
        ]
    );
    conditional_unknown_case!(
        analyze_conditional_raw_spelling_is_not_decoded_to_make_bodies_equal,
        "if ($x) { $a='a''b' } else { $a=\"a'b\" }; Add-Type $a",
        52,
        41,
        vec![
            gap(52, (0, 39), RegionKind::ControlFlow),
            CoverageGap {
                span: Span::new(32, 37),
                reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                context: script_context(52),
            },
        ]
    );
    conditional_unknown_case!(
        analyze_first_condition_command_kills_both_empty_outcomes,
        "$a='X'; if (Mystery) { } else { }; Add-Type $a",
        46,
        35,
        vec![
            gap(46, (8, 33), RegionKind::ControlFlow),
            gap(46, (12, 19), RegionKind::OtherCommand),
        ]
    );
    conditional_unknown_case!(
        analyze_no_else_uses_post_condition_prefix_not_original_input,
        "$a='X'; if (Mystery) { $a='X' }; Add-Type $a",
        44,
        33,
        vec![
            gap(44, (8, 31), RegionKind::ControlFlow),
            gap(44, (12, 19), RegionKind::OtherCommand),
        ]
    );
    conditional_unknown_case!(
        analyze_elseif_command_kills_remaining_false_outcomes,
        "$a='X'; if ($x) { } elseif (Mystery) { } else { }; Add-Type $a",
        62,
        51,
        vec![
            gap(62, (8, 49), RegionKind::ControlFlow),
            gap(62, (28, 35), RegionKind::OtherCommand),
        ]
    );
    conditional_unknown_case!(
        analyze_sibling_overwrite_does_not_replace_else_inherited_value,
        "$a='X'; if ($x) { $a='Y' } else { $b=$a }; Add-Type $a",
        54,
        43,
        vec![gap(54, (8, 41), RegionKind::ControlFlow)]
    );
    conditional_unknown_case!(
        analyze_nested_join_missing_outcome_is_not_repaired_by_outer_else,
        "if ($x) { if ($y) { $a='X' } } else { $a='X' }; Add-Type $a",
        59,
        48,
        vec![
            gap(59, (0, 46), RegionKind::ControlFlow),
            gap(59, (10, 28), RegionKind::ControlFlow),
        ]
    );
    conditional_unknown_case!(
        analyze_final_matching_arm_does_not_resurrect_key_lost_at_earlier_join,
        "if ($x) { $a='X' } elseif ($y) { $a='Y' } else { $a='X' }; Add-Type $a",
        70,
        59,
        vec![gap(70, (0, 57), RegionKind::ControlFlow)]
    );
    conditional_unknown_case!(
        analyze_no_else_after_elseif_uses_final_effectful_false_prefix,
        "$a='X'; if ($x) { } elseif (Mystery) { $a='X' }; Add-Type $a",
        60,
        49,
        vec![
            gap(60, (8, 47), RegionKind::ControlFlow),
            gap(60, (28, 35), RegionKind::OtherCommand),
        ]
    );
    transfer_case!(
        analyze_join_removes_differing_key_without_killing_unaffected_key,
        "$a='X'; if ($x) { $b='Y' } else { $b='Z' }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                55,
                positional((44, 55), (44, 52), (53, 55)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((53, 55), "$a")],
            )],
            coverage: vec![gap(55, (8, 42), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_empty_conditional_arms_preserve_inherited_value_once,
        "$a='X'; if ($x) { } else { }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                41,
                positional((30, 41), (30, 38), (39, 41)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((39, 41), "$a")],
            )],
            coverage: vec![gap(41, (8, 28), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_empty_arm_without_else_preserves_inherited_value_once,
        "$a='X'; if ($x) { }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                32,
                positional((21, 32), (21, 29), (30, 32)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((30, 32), "$a")],
            )],
            coverage: vec![gap(32, (8, 19), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_empty_raw_bodies_join_with_real_zero_width_origins,
        "if ($x) { $a='' } else { $a='' }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((34, 45), (34, 42), (43, 45)),
                raw_body: "".into(),
                literal_origins: vec![literal((13, 15), (14, 14)), literal((28, 30), (29, 29))],
                assignment_origins: vec![
                    assignment((10, 15), (10, 12), (13, 15), "$a"),
                    assignment((25, 30), (25, 27), (28, 30), "$a"),
                ],
                reference_origins: vec![reference((43, 45), "$a")],
                context: script_context(45),
            }],
            coverage: vec![gap(45, (0, 32), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_condition_assignment_transfers_before_both_snapshots,
        "if ($a='X') { } else { }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                37,
                positional((26, 37), (26, 34), (35, 37)),
                "X",
                literal((7, 10), (8, 9)),
                vec![assignment((4, 10), (4, 6), (7, 10), "$a")],
                vec![reference((35, 37), "$a")],
            )],
            coverage: vec![gap(37, (0, 24), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_first_condition_command_all_arms_restore_known_value,
        "if (Mystery) { $a='X' } else { $a='X' }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((41, 52), (41, 49), (50, 52)),
                raw_body: "X".into(),
                literal_origins: vec![literal((18, 21), (19, 20)), literal((34, 37), (35, 36))],
                assignment_origins: vec![
                    assignment((15, 21), (15, 17), (18, 21), "$a"),
                    assignment((31, 37), (31, 33), (34, 37), "$a"),
                ],
                reference_origins: vec![reference((50, 52), "$a")],
                context: script_context(52),
            }],
            coverage: vec![
                gap(52, (0, 39), RegionKind::ControlFlow),
                gap(52, (4, 11), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_elseif_command_does_not_retroactively_kill_completed_arm,
        "$a='X'; if ($x) { } elseif (Mystery) { $a='X' } else { $a='X' }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((65, 76), (65, 73), (74, 76)),
                raw_body: "X".into(),
                literal_origins: vec![
                    literal((3, 6), (4, 5)),
                    literal((42, 45), (43, 44)),
                    literal((58, 61), (59, 60)),
                ],
                assignment_origins: vec![
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                    assignment((39, 45), (39, 41), (42, 45), "$a"),
                    assignment((55, 61), (55, 57), (58, 61), "$a"),
                ],
                reference_origins: vec![reference((74, 76), "$a")],
                context: script_context(76),
            }],
            coverage: vec![
                gap(76, (8, 63), RegionKind::ControlFlow),
                gap(76, (28, 35), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_copy_after_join_keeps_shared_and_distinct_origins_in_outcome_order,
        "$a='X'; if ($x) { $b=$a } else { $b=$a }; $c=$b; Add-Type $c",
        CoreReport {
            confirmed: vec![known(
                60,
                positional((49, 60), (49, 57), (58, 60)),
                "X",
                literal((3, 6), (4, 5)),
                vec![
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                    assignment((18, 23), (18, 20), (21, 23), "$b"),
                    assignment((33, 38), (33, 35), (36, 38), "$b"),
                    assignment((42, 47), (42, 44), (45, 47), "$c"),
                ],
                vec![
                    reference((21, 23), "$a"),
                    reference((36, 38), "$a"),
                    reference((45, 47), "$b"),
                    reference((58, 60), "$c"),
                ],
            )],
            coverage: vec![gap(60, (8, 40), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_nested_equal_joins_preserve_all_three_real_origins,
        "if ($x) { if ($y) { $a='X' } else { $a='X' } } else { $a='X' }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((64, 75), (64, 72), (73, 75)),
                raw_body: "X".into(),
                literal_origins: vec![
                    literal((23, 26), (24, 25)),
                    literal((39, 42), (40, 41)),
                    literal((57, 60), (58, 59)),
                ],
                assignment_origins: vec![
                    assignment((20, 26), (20, 22), (23, 26), "$a"),
                    assignment((36, 42), (36, 38), (39, 42), "$a"),
                    assignment((54, 60), (54, 56), (57, 60), "$a"),
                ],
                reference_origins: vec![reference((73, 75), "$a")],
                context: script_context(75),
            }],
            coverage: vec![
                gap(75, (0, 62), RegionKind::ControlFlow),
                gap(75, (10, 44), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_bom_unicode_casefolded_join_and_copy_keep_original_byte_spans,
        "\u{feff}if ($x) { $ŻCode='λ' } else { $ŻcOdE='λ' }; $b=$Żcode; Add-Type $b",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((63, 74), (63, 71), (72, 74)),
                raw_body: "λ".into(),
                literal_origins: vec![literal((21, 25), (22, 24)), literal((43, 47), (44, 46))],
                assignment_origins: vec![
                    assignment((13, 25), (13, 20), (21, 25), "$ŻCode"),
                    assignment((35, 47), (35, 42), (43, 47), "$ŻcOdE"),
                    assignment((51, 61), (51, 53), (54, 61), "$b"),
                ],
                reference_origins: vec![reference((54, 61), "$Żcode"), reference((72, 74), "$b")],
                context: script_context(74),
            }],
            coverage: vec![gap(74, (3, 49), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_equal_raw_bodies_join_across_quote_kinds_without_decoding,
        "if ($x) { $a='X' } else { $a=\"X\" }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((36, 47), (36, 44), (45, 47)),
                raw_body: "X".into(),
                literal_origins: vec![literal((13, 16), (14, 15)), literal((29, 32), (30, 31))],
                assignment_origins: vec![
                    assignment((10, 16), (10, 12), (13, 16), "$a"),
                    assignment((26, 32), (26, 28), (29, 32), "$a"),
                ],
                reference_origins: vec![reference((45, 47), "$a")],
                context: script_context(47),
            }],
            coverage: vec![
                gap(47, (0, 34), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(29, 32),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(47),
                },
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_no_else_equal_write_orders_body_origin_before_inherited_origin,
        "$a='X'; if ($x) { $a='X' }; Add-Type $a",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((28, 39), (28, 36), (37, 39)),
                raw_body: "X".into(),
                literal_origins: vec![literal((21, 24), (22, 23)), literal((3, 6), (4, 5))],
                assignment_origins: vec![
                    assignment((18, 24), (18, 20), (21, 24), "$a"),
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                ],
                reference_origins: vec![reference((37, 39), "$a")],
                context: script_context(39),
            }],
            coverage: vec![gap(39, (8, 26), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_join_deduplicates_shared_reference_and_assignment_records,
        "$a='X'; $b=$a; if ($x) { $b=$b } else { }; Add-Type $b",
        CoreReport {
            confirmed: vec![known(
                54,
                positional((43, 54), (43, 51), (52, 54)),
                "X",
                literal((3, 6), (4, 5)),
                vec![
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                    assignment((8, 13), (8, 10), (11, 13), "$b"),
                    assignment((25, 30), (25, 27), (28, 30), "$b"),
                ],
                vec![
                    reference((11, 13), "$a"),
                    reference((28, 30), "$b"),
                    reference((52, 54), "$b"),
                ],
            )],
            coverage: vec![gap(54, (15, 41), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_else_copy_uses_own_snapshot_not_prior_arm_overwrite,
        "$a='X'; if ($x) { $a='Y'; $b='X' } else { $b=$a }; Add-Type $b",
        CoreReport {
            confirmed: vec![Confirmed {
                site: positional((51, 62), (51, 59), (60, 62)),
                raw_body: "X".into(),
                literal_origins: vec![literal((29, 32), (30, 31)), literal((3, 6), (4, 5))],
                assignment_origins: vec![
                    assignment((26, 32), (26, 28), (29, 32), "$b"),
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                    assignment((42, 47), (42, 44), (45, 47), "$b"),
                ],
                reference_origins: vec![reference((45, 47), "$a"), reference((60, 62), "$b")],
                context: script_context(62),
            }],
            coverage: vec![gap(62, (8, 49), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_conditional_commands_stay_unresolved_but_restoring_writes_transfer,
        "if (Add-Type 'C') { Add-Type 'T'; $a='X' } else { $a='X' }; Add-Type $a",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![Confirmed {
                site: positional((60, 71), (60, 68), (69, 71)),
                raw_body: "X".into(),
                literal_origins: vec![literal((37, 40), (38, 39)), literal((53, 56), (54, 55))],
                assignment_origins: vec![
                    assignment((34, 40), (34, 36), (37, 40), "$a"),
                    assignment((50, 56), (50, 52), (53, 56), "$a"),
                ],
                reference_origins: vec![reference((69, 71), "$a")],
                context: script_context(71),
            }],
            unresolved: vec![
                conditional_site(71, positional((4, 16), (4, 12), (13, 16))),
                conditional_site(71, positional((20, 32), (20, 28), (29, 32))),
            ],
            coverage: vec![gap(71, (0, 58), RegionKind::ControlFlow)],
        }
    );
    transfer_case!(
        analyze_conditional_add_type_is_not_pure_even_when_literal_is_unresolved,
        "$a='X'; if ($x) { Add-Type 'C' } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(54, positional((18, 30), (18, 26), (27, 30))),
                unknown(
                    54,
                    positional((43, 54), (43, 51), (52, 54)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![gap(54, (8, 41), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_nested_conditional_sites_are_discovered_once_and_never_confirmed,
        "if ($x) { if ($y) { Add-Type 'I' }; Add-Type 'O' }; Add-Type 'S'",
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                64,
                positional((52, 64), (52, 60), (61, 64)),
                "S",
                literal((61, 64), (62, 63)),
                vec![],
                vec![],
            )],
            unresolved: vec![
                conditional_site(64, positional((20, 32), (20, 28), (29, 32))),
                conditional_site(64, positional((36, 48), (36, 44), (45, 48))),
            ],
            coverage: vec![
                gap(64, (0, 50), RegionKind::ControlFlow),
                gap(64, (10, 34), RegionKind::ControlFlow),
            ],
        }
    );

    macro_rules! conditional_flow_case {
        ($name:ident, $source:literal, $end:literal, $if_end:literal,
         $flow_end:literal, $tail:literal, $sibling:literal, $last:literal) => {
            transfer_case!(
                $name,
                $source,
                CoreReport {
                    unresolved: vec![
                        conditional_site(
                            $end,
                            positional(
                                ($tail, $tail + 12),
                                ($tail, $tail + 8),
                                ($tail + 9, $tail + 12)
                            )
                        ),
                        conditional_site(
                            $end,
                            positional(
                                ($sibling, $sibling + 12),
                                ($sibling, $sibling + 8),
                                ($sibling + 9, $sibling + 12)
                            )
                        ),
                        conditional_site(
                            $end,
                            positional(($last, $end), ($last, $last + 8), ($end - 2, $end))
                        ),
                    ],
                    coverage: vec![
                        gap($end, (0, $if_end), RegionKind::ControlFlow),
                        gap($end, (10, $flow_end), RegionKind::ControlFlow),
                    ],
                    ..CoreReport::default()
                }
            );
        };
    }
    conditional_flow_case!(
        analyze_conditional_return_preserves_sibling_discovery_and_taints_continuation,
        "if ($x) { return; Add-Type 'T' } else { Add-Type 'S' }; $a='X'; Add-Type $a",
        75,
        54,
        16,
        18,
        40,
        64
    );
    conditional_flow_case!(
        analyze_conditional_throw_prevents_restore_in_continuation,
        "if ($x) { throw; Add-Type 'T' } else { Add-Type 'S' }; $a='X'; Add-Type $a",
        74,
        53,
        15,
        17,
        39,
        63
    );
    conditional_flow_case!(
        analyze_conditional_break_prevents_restore_in_continuation,
        "if ($x) { break; Add-Type 'T' } else { Add-Type 'S' }; $a='X'; Add-Type $a",
        74,
        53,
        15,
        17,
        39,
        63
    );
    conditional_flow_case!(
        analyze_conditional_continue_prevents_restore_in_continuation,
        "if ($x) { continue; Add-Type 'T' } else { Add-Type 'S' }; $a='X'; Add-Type $a",
        77,
        56,
        18,
        20,
        42,
        66
    );
    conditional_flow_case!(
        analyze_conditional_exit_prevents_restore_in_continuation,
        "if ($x) { exit; Add-Type 'T' } else { Add-Type 'S' }; $a='X'; Add-Type $a",
        73,
        52,
        14,
        16,
        38,
        62
    );
    transfer_case!(
        analyze_elseif_flow_makes_literal_continuation_discovery_only,
        "if ($x) { } elseif ($y) { return } else { }; Add-Type 'Z'",
        CoreReport {
            unresolved: vec![conditional_site(
                57,
                positional((45, 57), (45, 53), (54, 57))
            )],
            coverage: vec![
                gap(57, (0, 43), RegionKind::ControlFlow),
                gap(57, (26, 32), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_else_flow_makes_literal_continuation_discovery_only,
        "if ($x) { } else { return }; Add-Type 'Z'",
        CoreReport {
            unresolved: vec![conditional_site(
                41,
                positional((29, 41), (29, 37), (38, 41))
            )],
            coverage: vec![
                gap(41, (0, 27), RegionKind::ControlFlow),
                gap(41, (19, 25), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_nested_arm_flow_propagates_through_outer_join_and_ignores_restores,
        "if ($x) { if ($y) { throw } else { }; $a='X' } else { $a='X' }; $a='X'; Add-Type 'Z'",
        CoreReport {
            unresolved: vec![conditional_site(
                84,
                positional((72, 84), (72, 80), (81, 84))
            )],
            coverage: vec![
                gap(84, (0, 62), RegionKind::ControlFlow),
                gap(84, (10, 36), RegionKind::ControlFlow),
                gap(84, (20, 25), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_named_assignment_keeps_literal_body_assignment_and_read_spans,
        "$a='X'; Add-Type -TypeDefinition $a",
        CoreReport {
            confirmed: vec![known(
                35,
                AddTypeSite {
                    parameter_span: Some(Span::new(17, 35)),
                    ..positional((8, 35), (8, 16), (33, 35))
                },
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((33, 35), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_copy_retains_literal_and_appends_real_reference_and_assignment_origins,
        "$a='X'; $b=$a; Add-Type $b",
        CoreReport {
            confirmed: vec![known(
                26,
                positional((15, 26), (15, 23), (24, 26)),
                "X",
                literal((3, 6), (4, 5)),
                vec![
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                    assignment((8, 13), (8, 10), (11, 13), "$b"),
                ],
                vec![reference((11, 13), "$a"), reference((24, 26), "$b")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_self_copy_reads_old_value_and_retains_finite_ordered_provenance,
        "$a='X'; $a=$a; $a=$a; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                33,
                positional((22, 33), (22, 30), (31, 33)),
                "X",
                literal((3, 6), (4, 5)),
                vec![
                    assignment((0, 6), (0, 2), (3, 6), "$a"),
                    assignment((8, 13), (8, 10), (11, 13), "$a"),
                    assignment((15, 20), (15, 17), (18, 20), "$a"),
                ],
                vec![
                    reference((11, 13), "$a"),
                    reference((18, 20), "$a"),
                    reference((31, 33), "$a"),
                ],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unknown_self_copy_does_not_invent_literal,
        "$a=$a; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                18,
                positional((7, 18), (7, 15), (16, 18)),
                ValueReason::VariableNotKnown,
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_parenthesized_copy_preserves_inner_read_and_outer_rhs_spans,
        "$a=(('X')); $b=(($a)); Add-Type (($b))",
        CoreReport {
            confirmed: vec![known(
                38,
                positional((23, 38), (23, 31), (32, 38)),
                "X",
                literal((5, 8), (6, 7)),
                vec![
                    assignment((0, 10), (0, 2), (3, 10), "$a"),
                    assignment((12, 21), (12, 14), (15, 21), "$b"),
                ],
                vec![reference((17, 19), "$a"), reference((34, 36), "$b")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_ascii_case_lookup_preserves_both_raw_spellings,
        "$Code='X'; Add-Type $cOdE",
        CoreReport {
            confirmed: vec![known(
                25,
                positional((11, 25), (11, 19), (20, 25)),
                "X",
                literal((6, 9), (7, 8)),
                vec![assignment((0, 9), (0, 5), (6, 9), "$Code")],
                vec![reference((20, 25), "$cOdE")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unicode_variable_folds_only_its_ascii_letters,
        "$ŻCode='X'; Add-Type $ŻcOdE",
        CoreReport {
            confirmed: vec![known(
                29,
                positional((13, 29), (13, 21), (22, 29)),
                "X",
                literal((8, 11), (9, 10)),
                vec![assignment((0, 11), (0, 7), (8, 11), "$ŻCode")],
                vec![reference((22, 29), "$ŻcOdE")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_non_ascii_case_variant_is_a_distinct_unknown_variable,
        "$Ż='X'; Add-Type $ż",
        CoreReport {
            unresolved: vec![unknown(
                21,
                positional((9, 21), (9, 17), (18, 21)),
                ValueReason::VariableNotKnown,
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unrelated_simple_write_does_not_kill_known_value,
        "$a='X'; $other=1; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                29,
                positional((18, 29), (18, 26), (27, 29)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((27, 29), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unknown_copy_invalidates_only_its_target,
        "$a='X'; $a=$b; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                26,
                positional((15, 26), (15, 23), (24, 26)),
                ValueReason::VariableNotKnown,
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_literal_restore_replaces_stale_provenance,
        "$a='X'; $a=1; $a='Y'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                33,
                positional((22, 33), (22, 30), (31, 33)),
                "Y",
                literal((17, 20), (18, 19)),
                vec![assignment((14, 20), (14, 16), (17, 20), "$a")],
                vec![reference((31, 33), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_rhs_command_reads_old_lhs_before_command_and_assignment_kill,
        "$a='X'; $a=Add-Type -TypeDefinition $a; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                51,
                AddTypeSite {
                    parameter_span: Some(Span::new(20, 38)),
                    ..positional((11, 38), (11, 19), (36, 38))
                },
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((36, 38), "$a")],
            )],
            unresolved: vec![unknown(
                51,
                positional((40, 51), (40, 48), (49, 51)),
                ValueReason::VariableNotKnown,
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_add_type_invalidates_even_unrelated_cached_variable_after_finding,
        "$a='X'; Add-Type 'Y'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                33,
                positional((8, 20), (8, 16), (17, 20)),
                "Y",
                literal((17, 20), (18, 19)),
                vec![],
                vec![],
            )],
            unresolved: vec![unknown(
                33,
                positional((22, 33), (22, 30), (31, 33)),
                ValueReason::VariableNotKnown,
            )],
            ..CoreReport::default()
        }
    );

    macro_rules! target_invalidation_cases {
        ($(($name:ident, $source:literal, $end:literal, $command:expr, $name_span:expr,
            $argument:expr)),+ $(,)?) => {
            $(transfer_case!(
                $name, $source,
                CoreReport {
                    unresolved: vec![unknown(
                        $end, positional($command, $name_span, $argument),
                        ValueReason::VariableNotKnown,
                    )],
                    ..CoreReport::default()
                }
            );)+
        };
    }

    target_invalidation_cases!(
        (
            analyze_plus_assign_invalidates_target,
            "$a='X'; $a+='Y'; Add-Type $a",
            28,
            (17, 28),
            (17, 25),
            (26, 28)
        ),
        (
            analyze_minus_assign_invalidates_target,
            "$a='X'; $a-='Y'; Add-Type $a",
            28,
            (17, 28),
            (17, 25),
            (26, 28)
        ),
        (
            analyze_multiply_assign_invalidates_target,
            "$a='X'; $a*='Y'; Add-Type $a",
            28,
            (17, 28),
            (17, 25),
            (26, 28)
        ),
        (
            analyze_divide_assign_invalidates_target,
            "$a='X'; $a/='Y'; Add-Type $a",
            28,
            (17, 28),
            (17, 25),
            (26, 28)
        ),
        (
            analyze_remainder_assign_invalidates_target,
            "$a='X'; $a%='Y'; Add-Type $a",
            28,
            (17, 28),
            (17, 25),
            (26, 28)
        ),
        (
            analyze_coalescing_assign_invalidates_target_without_folding,
            "$a='X'; $a??='Y'; Add-Type $a",
            29,
            (18, 29),
            (18, 26),
            (27, 29)
        ),
        (
            analyze_prefix_increment_invalidates_target,
            "$a='X'; ++$a; Add-Type $a",
            25,
            (14, 25),
            (14, 22),
            (23, 25)
        ),
        (
            analyze_prefix_decrement_invalidates_target,
            "$a='X'; --$a; Add-Type $a",
            25,
            (14, 25),
            (14, 22),
            (23, 25)
        ),
        (
            analyze_postfix_increment_invalidates_target,
            "$a='X'; $a++; Add-Type $a",
            25,
            (14, 25),
            (14, 22),
            (23, 25)
        ),
        (
            analyze_postfix_decrement_invalidates_target,
            "$a='X'; $a--; Add-Type $a",
            25,
            (14, 25),
            (14, 22),
            (23, 25)
        ),
        (
            analyze_nonstring_assignment_invalidates_target,
            "$a='X'; $a=1; Add-Type $a",
            25,
            (14, 25),
            (14, 22),
            (23, 25)
        ),
    );

    transfer_case!(
        analyze_compound_write_preserves_unrelated_constant,
        "$a='X'; $b+=1; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                26,
                positional((15, 26), (15, 23), (24, 26)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((24, 26), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_mutation_preserves_unrelated_constant,
        "$a='X'; ++$b; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                25,
                positional((14, 25), (14, 22), (23, 25)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((23, 25), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    macro_rules! killed_environment_cases {
        ($(($name:ident, $source:literal, $end:literal,
            $command:expr, $name_span:expr, $argument:expr,
            [$($gap:expr => $region:ident),* $(,)?])),+ $(,)?) => {
            $(transfer_case!(
                $name, $source,
                CoreReport {
                    unresolved: vec![unknown(
                        $end, positional($command, $name_span, $argument),
                        ValueReason::VariableNotKnown,
                    )],
                    coverage: vec![$(gap($end, $gap, RegionKind::$region)),*],
                    ..CoreReport::default()
                }
            );)+
        };
    }

    killed_environment_cases!(
        (analyze_unknown_command_kills_environment_after_arguments,
            "$a='X'; Mystery $a; Add-Type $a",
            31, (20, 31), (20, 28), (29, 31), [(8, 18) => OtherCommand]),
        (analyze_unrelated_rhs_command_still_kills_whole_environment,
            "$a='X'; $other=Get-Thing; Add-Type $a",
            37, (26, 37), (26, 34), (35, 37), [(15, 24) => OtherCommand]),
        (analyze_unrelated_provider_read_kills_whole_environment,
            "$a='X'; $other=$env:PATH; Add-Type $a",
            37, (26, 37), (26, 34), (35, 37), [(15, 24) => VariableRead]),
        (analyze_scope_write_does_not_alias_plain_name_and_kills_environment,
            "$a='X'; $script:a='Y'; Add-Type $a",
            34, (23, 34), (23, 31), (32, 34),
            [(8, 21) => UnsupportedWrite, (8, 17) => VariableRead]),
        (analyze_braced_write_kills_environment,
            "$a='X'; ${a}='Y'; Add-Type $a",
            29, (18, 29), (18, 26), (27, 29),
            [(8, 16) => UnsupportedWrite, (8, 12) => VariableRead]),
        (analyze_splatted_write_kills_environment,
            "$a='X'; @a='Y'; Add-Type $a",
            27, (16, 27), (16, 24), (25, 27),
            [(8, 14) => UnsupportedWrite, (8, 10) => VariableRead]),
        (analyze_ascii_digit_only_write_kills_environment,
            "$a='X'; $12='Y'; Add-Type $a",
            28, (17, 28), (17, 25), (26, 28),
            [(8, 15) => UnsupportedWrite, (8, 11) => VariableRead]),
        (analyze_automatic_write_kills_environment,
            "$a='X'; $_='Y'; Add-Type $a",
            27, (16, 27), (16, 24), (25, 27),
            [(8, 14) => UnsupportedWrite, (8, 10) => VariableRead]),
        (analyze_member_write_kills_environment_without_leaf_target_inference,
            "$a='X'; $a.X='Y'; Add-Type $a",
            29, (18, 29), (18, 26), (27, 29),
            [(8, 16) => UnsupportedWrite, (8, 12) => Expression]),
        (analyze_index_write_kills_environment,
            "$a='X'; $a[0]='Y'; Add-Type $a",
            30, (19, 30), (19, 27), (28, 30),
            [(8, 17) => UnsupportedWrite, (8, 13) => Expression]),
        (analyze_typed_write_kills_environment_without_unwrapping_cast,
            "$a='X'; [string]$b='Y'; Add-Type $a",
            35, (24, 35), (24, 32), (33, 35),
            [(8, 22) => UnsupportedWrite, (8, 18) => Expression]),
        (analyze_destructuring_write_kills_environment,
            "$a='X'; $a,$b='Y'; Add-Type $a",
            30, (19, 30), (19, 27), (28, 30),
            [(8, 17) => UnsupportedWrite, (8, 13) => Expression]),
        (analyze_dynamic_invocation_kills_environment_without_static_site,
            "$a='X'; & Add-Type 'Y'; Add-Type $a",
            35, (24, 35), (24, 32), (33, 35), [(8, 22) => Invocation]),
        (analyze_dot_sourcing_kills_environment,
            "$a='X'; . script.ps1; Add-Type $a",
            33, (22, 33), (22, 30), (31, 33), [(8, 20) => Invocation]),
    );

    transfer_case!(
        analyze_conditional_effectful_rhs_is_discovered_not_executed_or_joined,
        "$a='X'; $a??=Add-Type 'Y'; Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((13, 25), (13, 21), (22, 25)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: script_context(38),
                },
                unknown(
                    38,
                    positional((27, 38), (27, 35), (36, 38)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![gap(38, (8, 25), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unknown_conditional_target_does_not_confirm_literal_rhs_call,
        "$a='X'; $b??=Add-Type 'Y'; Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((13, 25), (13, 21), (22, 25)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: script_context(38),
                },
                unknown(
                    38,
                    positional((27, 38), (27, 35), (36, 38)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![gap(38, (8, 25), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unknown_command_argument_nested_site_stays_unresolved_and_kills_afterward,
        "$a='X'; Mystery (Add-Type 'Y'); Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((17, 29), (17, 25), (26, 29)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context: script_context(43),
                },
                unknown(
                    43,
                    positional((32, 43), (32, 40), (41, 43)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(43, (8, 30), RegionKind::OtherCommand),
                gap(43, (16, 30), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );

    macro_rules! unsupported_read_cases {
        ($(($name:ident, $source:literal, $end:literal)),+ $(,)?) => {
            $(transfer_case!(
                $name, $source,
                CoreReport {
                    unresolved: vec![unknown(
                        $end, positional((0, $end), (0, 8), (9, $end)),
                        ValueReason::UnsupportedVariable,
                    )],
                    coverage: vec![gap($end, (9, $end), RegionKind::VariableRead)],
                    ..CoreReport::default()
                }
            );)+
        };
    }

    unsupported_read_cases!(
        (
            analyze_braced_read_is_explicitly_unsupported,
            "Add-Type ${a}",
            13
        ),
        (
            analyze_empty_braced_name_is_explicitly_unsupported,
            "Add-Type ${}",
            12
        ),
        (
            analyze_splatted_read_is_explicitly_unsupported,
            "Add-Type @a",
            11
        ),
        (
            analyze_doubled_sigil_read_is_explicitly_unsupported,
            "Add-Type $$",
            11
        ),
        (
            analyze_question_special_read_is_explicitly_unsupported,
            "Add-Type $?",
            11
        ),
        (
            analyze_digit_only_read_is_explicitly_unsupported,
            "Add-Type $123",
            13
        ),
        (
            analyze_automatic_read_uses_existing_ascii_case_insensitive_list,
            "Add-Type $TrUe",
            14
        ),
        (
            analyze_underscore_automatic_is_not_a_plain_binding,
            "Add-Type $_",
            11
        ),
        (
            analyze_preference_read_uses_existing_ascii_case_insensitive_list,
            "Add-Type $eRrOrAcTiOnPrEfErEnCe",
            31
        ),
        (
            analyze_script_qualified_read_is_explicitly_unsupported,
            "Add-Type $script:a",
            18
        ),
        (
            analyze_global_qualified_read_is_explicitly_unsupported,
            "Add-Type $global:a",
            18
        ),
        (
            analyze_local_qualified_read_is_explicitly_unsupported,
            "Add-Type $local:a",
            17
        ),
        (
            analyze_private_qualified_read_is_explicitly_unsupported,
            "Add-Type $private:a",
            19
        ),
        (
            analyze_provider_read_is_explicitly_unsupported,
            "Add-Type $env:PATH",
            18
        ),
        (
            analyze_braced_provider_read_is_explicitly_unsupported,
            "Add-Type ${env:PATH}",
            20
        ),
    );

    transfer_case!(
        analyze_braced_read_never_uses_plain_environment_key,
        "$a='X'; Add-Type ${a}",
        CoreReport {
            unresolved: vec![unknown(
                21,
                positional((8, 21), (8, 16), (17, 21)),
                ValueReason::UnsupportedVariable,
            )],
            coverage: vec![gap(21, (17, 21), RegionKind::VariableRead)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_scope_read_is_unknown_without_killing_unrelated_constant,
        "$a='X'; $other=$script:a; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                37,
                positional((26, 37), (26, 34), (35, 37)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((35, 37), "$a")],
            )],
            coverage: vec![gap(37, (15, 24), RegionKind::VariableRead)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_provider_argument_is_unknown_and_invalidates_prior_binding,
        "$a='X'; Add-Type $env:PATH; Add-Type $a",
        CoreReport {
            unresolved: vec![
                unknown(
                    39,
                    positional((8, 26), (8, 16), (17, 26)),
                    ValueReason::UnsupportedVariable
                ),
                unknown(
                    39,
                    positional((28, 39), (28, 36), (37, 39)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![gap(39, (17, 26), RegionKind::VariableRead)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_plain_underscore_suffix_is_not_automatic_underscore,
        "$_a='X'; Add-Type $_a",
        CoreReport {
            confirmed: vec![known(
                21,
                positional((9, 21), (9, 17), (18, 21)),
                "X",
                literal((4, 7), (5, 6)),
                vec![assignment((0, 7), (0, 3), (4, 7), "$_a")],
                vec![reference((18, 21), "$_a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_digit_prefixed_alphanumeric_name_is_plain,
        "$1a='X'; Add-Type $1a",
        CoreReport {
            confirmed: vec![known(
                21,
                positional((9, 21), (9, 17), (18, 21)),
                "X",
                literal((4, 7), (5, 6)),
                vec![assignment((0, 7), (0, 3), (4, 7), "$1a")],
                vec![reference((18, 21), "$1a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_non_ascii_digit_name_is_not_excluded_by_ascii_digit_rule,
        "$١='X'; Add-Type $١",
        CoreReport {
            confirmed: vec![known(
                21,
                positional((9, 21), (9, 17), (18, 21)),
                "X",
                literal((4, 7), (5, 6)),
                vec![assignment((0, 7), (0, 3), (4, 7), "$١")],
                vec![reference((18, 21), "$١")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unknown_copy_to_other_target_preserves_independent_binding,
        "$a='X'; $b=$z; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                26,
                positional((15, 26), (15, 23), (24, 26)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((24, 26), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_compound_rhs_reports_old_value_before_whole_environment_kill,
        "$a='X'; $b+=Add-Type $a; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                36,
                positional((12, 23), (12, 20), (21, 23)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((21, 23), "$a")],
            )],
            unresolved: vec![unknown(
                36,
                positional((25, 36), (25, 33), (34, 36)),
                ValueReason::VariableNotKnown,
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_conditional_command_kills_arm_value_and_keeps_nested_literal_unresolved,
        "$a='X'; if ($x) { $a='Y'; Add-Type 'Z' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((26, 38), (26, 34), (35, 38)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: script_context(53),
                },
                unknown(
                    53,
                    positional((42, 53), (42, 50), (51, 53)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![gap(53, (8, 40), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_method_barrier_keeps_nested_literal_unresolved_and_kills_parent,
        "$a='X'; [T]::M((Add-Type 'Y')); Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((16, 28), (16, 24), (25, 28)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context: script_context(43),
                },
                unknown(
                    43,
                    positional((32, 43), (32, 40), (41, 43)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(43, (8, 30), RegionKind::Expression),
                gap(43, (15, 29), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_redirection_blocks_literal_confirmation_and_kills_parent,
        "$a='X'; Add-Type 'Y' > out; Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((8, 26), (8, 16), (17, 20)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Redirection),
                    context: script_context(39),
                },
                unknown(
                    39,
                    positional((28, 39), (28, 36), (37, 39)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(39, (8, 26), RegionKind::Redirection),
                gap(39, (21, 26), RegionKind::Redirection),
            ],
            ..CoreReport::default()
        }
    );

    macro_rules! flow_cases {
        ($(($name:ident, $source:literal, $end:literal,
            $flow:expr, $command:expr, $name_span:expr, $argument:expr)),+ $(,)?) => {
            $(transfer_case!(
                $name, $source,
                CoreReport {
                    unresolved: vec![Unresolved {
                        site: positional($command, $name_span, $argument),
                        reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                        context: script_context($end),
                    }],
                    coverage: vec![gap($end, $flow, RegionKind::ControlFlow)],
                    ..CoreReport::default()
                }
            );)+
        };
    }

    flow_cases!(
        (
            analyze_return_tail_assignment_cannot_restore_confirmation,
            "$a='X'; return; $a='Y'; Add-Type $a",
            35,
            (8, 14),
            (24, 35),
            (24, 32),
            (33, 35)
        ),
        (
            analyze_throw_tail_assignment_cannot_restore_confirmation,
            "$a='X'; throw; $a='Y'; Add-Type $a",
            34,
            (8, 13),
            (23, 34),
            (23, 31),
            (32, 34)
        ),
        (
            analyze_break_tail_assignment_cannot_restore_confirmation,
            "$a='X'; break; $a='Y'; Add-Type $a",
            34,
            (8, 13),
            (23, 34),
            (23, 31),
            (32, 34)
        ),
        (
            analyze_continue_tail_assignment_cannot_restore_confirmation,
            "$a='X'; continue; $a='Y'; Add-Type $a",
            37,
            (8, 16),
            (26, 37),
            (26, 34),
            (35, 37)
        ),
        (
            analyze_exit_tail_assignment_cannot_restore_confirmation,
            "$a='X'; exit; $a='Y'; Add-Type $a",
            33,
            (8, 12),
            (22, 33),
            (22, 30),
            (31, 33)
        ),
        (
            analyze_flow_tail_literal_is_also_discovery_only,
            "return; Add-Type 'X'",
            20,
            (0, 6),
            (8, 20),
            (8, 16),
            (17, 20)
        ),
    );

    killed_environment_cases!(
        (analyze_provider_write_kills_all_bindings,
            "$a='X'; $env:P='Y'; Add-Type $a",
            31, (20, 31), (20, 28), (29, 31),
            [(8, 18) => UnsupportedWrite, (8, 14) => VariableRead]),
        (analyze_indirect_prefix_mutation_is_a_whole_environment_barrier,
            "$a='X'; ++$a.X; Add-Type $a",
            27, (16, 27), (16, 24), (25, 27),
            [(8, 14) => UnsupportedWrite, (10, 14) => Expression]),
        (analyze_indirect_postfix_mutation_is_a_whole_environment_barrier,
            "$a='X'; $a[0]++; Add-Type $a",
            28, (17, 28), (17, 25), (26, 28),
            [(8, 15) => UnsupportedWrite, (8, 13) => Expression]),
        (analyze_unsupported_even_literal_concatenation_kills_unrelated_binding,
            "$a='X'; $b=('Y'+'Z'); Add-Type $a",
            33, (22, 33), (22, 30), (31, 33),
            [(11, 20) => Expression, (12, 19) => Expression]),
    );

    transfer_case!(
        analyze_later_literal_restores_after_compound_invalidation,
        "$a='X'; $a+=1; $a='Y'; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                34,
                positional((23, 34), (23, 31), (32, 34)),
                "Y",
                literal((18, 21), (19, 20)),
                vec![assignment((15, 21), (15, 17), (18, 21), "$a")],
                vec![reference((32, 34), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_eligible_double_assignment_is_classified_before_fragment_effects,
        "$a=\"`$x\"; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                21,
                positional((10, 21), (10, 18), (19, 21)),
                "`$x",
                literal((3, 8), (4, 7)),
                vec![assignment((0, 8), (0, 2), (3, 8), "$a")],
                vec![reference((19, 21), "$a")],
            )],
            coverage: vec![CoverageGap {
                span: Span::new(3, 8),
                reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                context: script_context(21),
            }],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_eligible_double_statement_does_not_kill_unrelated_binding,
        "$a='X'; \"no effects\"; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                33,
                positional((22, 33), (22, 30), (31, 33)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((31, 33), "$a")],
            )],
            coverage: vec![CoverageGap {
                span: Span::new(8, 20),
                reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                context: script_context(33),
            }],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_unrelated_interpolation_kills_environment_and_retains_nested_unresolved_site,
        "$a='X'; $other=\"$(Add-Type 'Y')\"; Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((18, 30), (18, 26), (27, 30)),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(45),
                },
                unknown(
                    45,
                    positional((34, 45), (34, 42), (43, 45)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                CoverageGap {
                    span: Span::new(15, 32),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(45),
                },
                gap(45, (16, 31), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_conditional_interpolation_is_a_barrier_not_an_unconditional_rhs,
        "$a='X'; $b??=\"$(Add-Type 'Y')\"; Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((16, 28), (16, 24), (25, 28)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: script_context(43),
                },
                unknown(
                    43,
                    positional((32, 43), (32, 40), (41, 43)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(43, (8, 30), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(13, 30),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(43),
                },
                gap(43, (14, 29), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_function_body_uses_local_environment_without_overwriting_script,
        "$a='X'; function f { $a='Y'; Add-Type $a }; Add-Type $a",
        CoreReport {
            confirmed: vec![
                Confirmed {
                    context: nested_context(
                        script_context(55),
                        ContextKind::Function("f".into()),
                        (8, 42),
                    ),
                    ..known(
                        55,
                        positional((29, 40), (29, 37), (38, 40)),
                        "Y",
                        literal((24, 27), (25, 26)),
                        vec![assignment((21, 27), (21, 23), (24, 27), "$a")],
                        vec![reference((38, 40), "$a")],
                    )
                },
                known(
                    55,
                    positional((44, 55), (44, 52), (53, 55)),
                    "X",
                    literal((3, 6), (4, 5)),
                    vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                    vec![reference((53, 55), "$a")],
                ),
            ],
            ..CoreReport::default()
        }
    );

    fn nested_context(mut parent: Context, kind: ContextKind, span: (usize, usize)) -> Context {
        parent.path.push(ContextFrame {
            kind,
            span: Span::new(span.0, span.1),
            status: ContextStatus::Deferred,
        });
        parent
    }

    macro_rules! context_case {
        ($name:ident, $source:literal, $context:ident = $path:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let source = $source;
                let $context = $path;
                let expected = $expected;

                let actual = analyze_add_type(source);

                assert_eq!(actual, Ok(expected));
            }
        };
    }

    context_case!(
        analyze_function_local_equal_join_confirms_only_in_isolated_definition_context,
        "function f { if ($x) { $a='X' } else { $a='X' }; Add-Type $a }; Add-Type $a",
        context = nested_context(
            script_context(75),
            ContextKind::Function("f".into()),
            (0, 62)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![Confirmed {
                site: positional((49, 60), (49, 57), (58, 60)),
                raw_body: "X".into(),
                literal_origins: vec![literal((26, 29), (27, 28)), literal((42, 45), (43, 44))],
                assignment_origins: vec![
                    assignment((23, 29), (23, 25), (26, 29), "$a"),
                    assignment((39, 45), (39, 41), (42, 45), "$a"),
                ],
                reference_origins: vec![reference((58, 60), "$a")],
                context: context.clone(),
            }],
            unresolved: vec![unknown(
                75,
                positional((64, 75), (64, 72), (73, 75)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![CoverageGap {
                span: Span::new(13, 47),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                context,
            }],
        }
    );
    context_case!(
        analyze_joined_script_value_is_not_a_capture_in_later_function_analysis,
        "if ($x) { $a='X' } else { $a='X' }; function f { Add-Type $a }; Add-Type $a",
        context = nested_context(
            script_context(75),
            ContextKind::Function("f".into()),
            (36, 62)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![Confirmed {
                site: positional((64, 75), (64, 72), (73, 75)),
                raw_body: "X".into(),
                literal_origins: vec![literal((13, 16), (14, 15)), literal((29, 32), (30, 31))],
                assignment_origins: vec![
                    assignment((10, 16), (10, 12), (13, 16), "$a"),
                    assignment((26, 32), (26, 28), (29, 32), "$a"),
                ],
                reference_origins: vec![reference((73, 75), "$a")],
                context: script_context(75),
            }],
            unresolved: vec![Unresolved {
                context,
                ..unknown(
                    75,
                    positional((49, 60), (49, 57), (58, 60)),
                    ValueReason::VariableNotKnown
                )
            }],
            coverage: vec![gap(75, (0, 34), RegionKind::ControlFlow)],
        }
    );
    context_case!(
        analyze_function_conditional_flow_taints_only_its_own_continuation,
        "$a='S'; function f { if ($x) { return } else { }; $a='L'; Add-Type $a }; Add-Type $a",
        context = nested_context(
            script_context(84),
            ContextKind::Function("f".into()),
            (8, 71)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                84,
                positional((73, 84), (73, 81), (82, 84)),
                "S",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((82, 84), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: context.clone(),
                ..conditional_site(84, positional((58, 69), (58, 66), (67, 69)))
            }],
            coverage: vec![
                CoverageGap {
                    span: Span::new(21, 48),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(31, 37),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context,
                },
            ],
        }
    );
    context_case!(
        analyze_conditional_definition_writes_and_flow_are_deferred_not_parent_effects,
        "$a='X'; if ($x) { function f { return; $a='Y'; Add-Type 'D' } } else { }; Add-Type $a",
        context = nested_context(
            script_context(85),
            ContextKind::Function("f".into()),
            (18, 61)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                85,
                positional((74, 85), (74, 82), (83, 85)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((83, 85), "$a")],
            )],
            unresolved: vec![Unresolved {
                context: context.clone(),
                ..conditional_site(85, positional((47, 59), (47, 55), (56, 59)))
            }],
            coverage: vec![
                gap(85, (8, 72), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(18, 61),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(31, 37),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context,
                },
            ],
        }
    );

    context_case!(
        analyze_function_capture_starts_unknown_even_after_script_literal_assignment,
        "$a='X'; function f { Add-Type $a }; Add-Type $a",
        context = nested_context(
            script_context(47),
            ContextKind::Function("f".into()),
            (8, 34)
        ),
        CoreReport {
            confirmed: vec![known(
                47,
                positional((36, 47), (36, 44), (45, 47)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((45, 47), "$a")],
            )],
            unresolved: vec![Unresolved {
                context,
                ..unknown(
                    47,
                    positional((21, 32), (21, 29), (30, 32)),
                    ValueReason::VariableNotKnown,
                )
            }],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_header_parameter_default_stays_unknown_until_local_literal_assignment,
        "function f($a='X') { Add-Type $a; $a='Y'; Add-Type $a }",
        context = nested_context(
            script_context(55),
            ContextKind::Function("f".into()),
            (0, 55)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![Confirmed {
                context: context.clone(),
                ..known(
                    55,
                    positional((42, 53), (42, 50), (51, 53)),
                    "Y",
                    literal((37, 40), (38, 39)),
                    vec![assignment((34, 40), (34, 36), (37, 40), "$a")],
                    vec![reference((51, 53), "$a")],
                )
            }],
            unresolved: vec![Unresolved {
                context: context.clone(),
                ..unknown(
                    55,
                    positional((21, 32), (21, 29), (30, 32)),
                    ValueReason::VariableNotKnown,
                )
            }],
            coverage: vec![CoverageGap {
                span: Span::new(11, 17),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                context,
            }],
        }
    );

    context_case!(
        analyze_body_parameter_default_stays_unknown_until_local_literal_assignment,
        "function f { param($a='X'); Add-Type $a; $a='Y'; Add-Type $a }",
        context = nested_context(
            script_context(62),
            ContextKind::Function("f".into()),
            (0, 62)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![Confirmed {
                context: context.clone(),
                ..known(
                    62,
                    positional((49, 60), (49, 57), (58, 60)),
                    "Y",
                    literal((44, 47), (45, 46)),
                    vec![assignment((41, 47), (41, 43), (44, 47), "$a")],
                    vec![reference((58, 60), "$a")],
                )
            }],
            unresolved: vec![Unresolved {
                context: context.clone(),
                ..unknown(
                    62,
                    positional((28, 39), (28, 36), (37, 39)),
                    ValueReason::VariableNotKnown,
                )
            }],
            coverage: vec![
                CoverageGap {
                    span: Span::new(13, 26),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(19, 25),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context,
                },
            ],
        }
    );

    context_case!(
        analyze_header_default_effects_are_discovered_without_parent_or_function_transfers,
        "$a='X'; function f($p=$(Add-Type 'D')) { $a='Y'; Add-Type $a }; Add-Type $a",
        context = nested_context(
            script_context(75),
            ContextKind::Function("f".into()),
            (8, 62)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![
                Confirmed {
                    context: context.clone(),
                    ..known(
                        75,
                        positional((49, 60), (49, 57), (58, 60)),
                        "Y",
                        literal((44, 47), (45, 46)),
                        vec![assignment((41, 47), (41, 43), (44, 47), "$a")],
                        vec![reference((58, 60), "$a")],
                    )
                },
                known(
                    75,
                    positional((64, 75), (64, 72), (73, 75)),
                    "X",
                    literal((3, 6), (4, 5)),
                    vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                    vec![reference((73, 75), "$a")],
                ),
            ],
            unresolved: vec![Unresolved {
                site: positional((24, 36), (24, 32), (33, 36)),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                context: context.clone(),
            }],
            coverage: vec![
                CoverageGap {
                    span: Span::new(19, 37),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(22, 37),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context,
                },
            ],
        }
    );

    context_case!(
        analyze_body_default_effects_are_discovered_and_local_assignment_can_establish_value,
        "function f { param($p=$(Add-Type 'D')); $a='L'; Add-Type $a }",
        context = nested_context(
            script_context(61),
            ContextKind::Function("f".into()),
            (0, 61)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![Confirmed {
                context: context.clone(),
                ..known(
                    61,
                    positional((48, 59), (48, 56), (57, 59)),
                    "L",
                    literal((43, 46), (44, 45)),
                    vec![assignment((40, 46), (40, 42), (43, 46), "$a")],
                    vec![reference((57, 59), "$a")],
                )
            }],
            unresolved: vec![Unresolved {
                site: positional((24, 36), (24, 32), (33, 36)),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                context: context.clone(),
            }],
            coverage: vec![
                CoverageGap {
                    span: Span::new(13, 38),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(19, 37),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(22, 37),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context,
                },
            ],
        }
    );

    #[test]
    fn analyze_nested_function_has_unknown_capture_and_writes_neither_parent_environment() {
        let source = "$a='S'; function outer { $a='O'; function inner { Add-Type $a; $a='I'; Add-Type $a }; Add-Type $a }; Add-Type $a";
        let outer = nested_context(
            script_context(112),
            ContextKind::Function("outer".into()),
            (8, 99),
        );
        let inner = nested_context(
            outer.clone(),
            ContextKind::Function("inner".into()),
            (33, 84),
        );
        let expected = CoreReport {
            confirmed: vec![
                Confirmed {
                    context: inner.clone(),
                    ..known(
                        112,
                        positional((71, 82), (71, 79), (80, 82)),
                        "I",
                        literal((66, 69), (67, 68)),
                        vec![assignment((63, 69), (63, 65), (66, 69), "$a")],
                        vec![reference((80, 82), "$a")],
                    )
                },
                Confirmed {
                    context: outer,
                    ..known(
                        112,
                        positional((86, 97), (86, 94), (95, 97)),
                        "O",
                        literal((28, 31), (29, 30)),
                        vec![assignment((25, 31), (25, 27), (28, 31), "$a")],
                        vec![reference((95, 97), "$a")],
                    )
                },
                known(
                    112,
                    positional((101, 112), (101, 109), (110, 112)),
                    "S",
                    literal((3, 6), (4, 5)),
                    vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                    vec![reference((110, 112), "$a")],
                ),
            ],
            unresolved: vec![Unresolved {
                context: inner,
                ..unknown(
                    112,
                    positional((50, 61), (50, 58), (59, 61)),
                    ValueReason::VariableNotKnown,
                )
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    context_case!(
        analyze_function_call_is_unknown_command_not_parameter_specialization,
        "function f($p) { Add-Type $p }; $a='S'; f 'X'; Add-Type $a",
        context = nested_context(
            script_context(58),
            ContextKind::Function("f".into()),
            (0, 30)
        ),
        CoreReport {
            unresolved: vec![
                Unresolved {
                    context: context.clone(),
                    ..unknown(
                        58,
                        positional((17, 28), (17, 25), (26, 28)),
                        ValueReason::VariableNotKnown,
                    )
                },
                unknown(
                    58,
                    positional((47, 58), (47, 55), (56, 58)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                CoverageGap {
                    span: Span::new(11, 13),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context,
                },
                gap(58, (40, 45), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_filter_uses_isolated_local_environment_not_pipeline_execution,
        "filter f { $a='L'; Add-Type $a }",
        context = nested_context(
            script_context(32),
            ContextKind::Function("f".into()),
            (0, 32)
        ),
        CoreReport {
            confirmed: vec![Confirmed {
                context,
                ..known(
                    32,
                    positional((19, 30), (19, 27), (28, 30)),
                    "L",
                    literal((14, 17), (15, 16)),
                    vec![assignment((11, 17), (11, 13), (14, 17), "$a")],
                    vec![reference((28, 30), "$a")],
                )
            }],
            ..CoreReport::default()
        }
    );

    #[test]
    fn analyze_function_context_and_provenance_own_original_bom_utf8_byte_spans() {
        let source = String::from("\u{feff}function Ż { $Code='界'; Add-Type $cOdE }");
        let expected = CoreReport {
            confirmed: vec![Confirmed {
                context: nested_context(
                    script_context(46),
                    ContextKind::Function("Ż".into()),
                    (3, 46),
                ),
                ..known(
                    46,
                    positional((30, 44), (30, 38), (39, 44)),
                    "界",
                    literal((23, 28), (24, 27)),
                    vec![assignment((17, 28), (17, 22), (23, 28), "$Code")],
                    vec![reference((39, 44), "$cOdE")],
                )
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(&source);
        drop(source);

        assert_eq!(actual, Ok(expected));
    }

    context_case!(
        analyze_function_qualified_write_is_unknown_barrier_not_plain_local_alias,
        "function f { $a='L'; $script:a='Q'; Add-Type $a }",
        context = nested_context(
            script_context(49),
            ContextKind::Function("f".into()),
            (0, 49)
        ),
        CoreReport {
            unresolved: vec![Unresolved {
                context: context.clone(),
                ..unknown(
                    49,
                    positional((36, 47), (36, 44), (45, 47)),
                    ValueReason::VariableNotKnown,
                )
            }],
            coverage: vec![
                CoverageGap {
                    span: Span::new(21, 34),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::UnsupportedWrite),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(21, 30),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::VariableRead),
                    context,
                },
            ],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_function_qualified_read_never_uses_plain_local_value,
        "function f { $a='L'; Add-Type $script:a; Add-Type $a }",
        context = nested_context(
            script_context(54),
            ContextKind::Function("f".into()),
            (0, 54)
        ),
        CoreReport {
            unresolved: vec![
                Unresolved {
                    context: context.clone(),
                    ..unknown(
                        54,
                        positional((21, 39), (21, 29), (30, 39)),
                        ValueReason::UnsupportedVariable,
                    )
                },
                Unresolved {
                    context: context.clone(),
                    ..unknown(
                        54,
                        positional((41, 52), (41, 49), (50, 52)),
                        ValueReason::VariableNotKnown,
                    )
                },
            ],
            coverage: vec![CoverageGap {
                span: Span::new(30, 39),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::VariableRead),
                context,
            }],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_function_flow_tail_stays_discovery_only_without_poisoning_script,
        "function f { return; $a='L'; Add-Type $a }; Add-Type 'S'",
        context = nested_context(
            script_context(56),
            ContextKind::Function("f".into()),
            (0, 42)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                56,
                positional((44, 56), (44, 52), (53, 56)),
                "S",
                literal((53, 56), (54, 55)),
                vec![],
                vec![],
            )],
            unresolved: vec![Unresolved {
                site: positional((29, 40), (29, 37), (38, 40)),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                context: context.clone(),
            }],
            coverage: vec![CoverageGap {
                span: Span::new(13, 19),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                context,
            }],
        }
    );

    context_case!(
        analyze_literal_parameter_default_keeps_nested_block_deferred_and_capture_unknown,
        "$a='S'; function f($p={ $a='D'; Add-Type $a }) { Add-Type $a }; Add-Type $a",
        context = nested_context(
            script_context(75),
            ContextKind::Function("f".into()),
            (8, 62)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                75,
                positional((64, 75), (64, 72), (73, 75)),
                "S",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((73, 75), "$a")],
            )],
            unresolved: vec![
                Unresolved {
                    site: positional((32, 43), (32, 40), (41, 43)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context: nested_context(context.clone(), ContextKind::ScriptBlock, (22, 45)),
                },
                Unresolved {
                    context: context.clone(),
                    ..unknown(
                        75,
                        positional((49, 60), (49, 57), (58, 60)),
                        ValueReason::VariableNotKnown,
                    )
                },
            ],
            coverage: vec![
                CoverageGap {
                    span: Span::new(19, 45),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(22, 45),
                    reason: DiscoveryReason::DeferredContext,
                    context: nested_context(context, ContextKind::ScriptBlock, (22, 45)),
                },
            ],
        }
    );

    context_case!(
        analyze_mixed_case_function_token_confirms_single_source_site_not_derived_csharp,
        "FuNcTiOn f { Add-Type 'class C {}' }",
        context = nested_context(
            script_context(36),
            ContextKind::Function("f".into()),
            (0, 36)
        ),
        CoreReport {
            confirmed: vec![Confirmed {
                context,
                ..known(
                    36,
                    positional((13, 34), (13, 21), (22, 34)),
                    "class C {}",
                    literal((22, 34), (23, 33)),
                    vec![],
                    vec![],
                )
            }],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_mixed_case_filter_token_is_an_isolated_definition_not_workflow,
        "FiLtEr f { Add-Type 'L' }",
        context = nested_context(
            script_context(25),
            ContextKind::Function("f".into()),
            (0, 25)
        ),
        CoreReport {
            confirmed: vec![Confirmed {
                context,
                ..known(
                    25,
                    positional((11, 23), (11, 19), (20, 23)),
                    "L",
                    literal((20, 23), (21, 22)),
                    vec![],
                    vec![],
                )
            }],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_function_literal_block_is_deferred_without_writing_local_environment,
        "function f { $a='L'; { $a='D'; Add-Type $a }; Add-Type $a }",
        context = nested_context(
            script_context(59),
            ContextKind::Function("f".into()),
            (0, 59)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![Confirmed {
                context: context.clone(),
                ..known(
                    59,
                    positional((46, 57), (46, 54), (55, 57)),
                    "L",
                    literal((16, 19), (17, 18)),
                    vec![assignment((13, 19), (13, 15), (16, 19), "$a")],
                    vec![reference((55, 57), "$a")],
                )
            }],
            unresolved: vec![Unresolved {
                site: positional((31, 42), (31, 39), (40, 42)),
                reason: DiscoveryReason::DeferredContext,
                context: nested_context(context.clone(), ContextKind::ScriptBlock, (21, 44)),
            }],
            coverage: vec![CoverageGap {
                span: Span::new(21, 44),
                reason: DiscoveryReason::DeferredContext,
                context: nested_context(context, ContextKind::ScriptBlock, (21, 44)),
            }],
        }
    );

    context_case!(
        analyze_function_conditional_command_prevents_known_join_and_keeps_site_unresolved,
        "function f { $a='L'; if ($x) { $a='B'; Add-Type 'D' }; Add-Type $a }",
        context = nested_context(
            script_context(68),
            ContextKind::Function("f".into()),
            (0, 68)
        ),
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((39, 51), (39, 47), (48, 51)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                    context: context.clone(),
                },
                Unresolved {
                    context: context.clone(),
                    ..unknown(
                        68,
                        positional((55, 66), (55, 63), (64, 66)),
                        ValueReason::VariableNotKnown,
                    )
                },
            ],
            coverage: vec![CoverageGap {
                span: Span::new(21, 53),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                context,
            }],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_unreachable_function_definition_keeps_sites_unresolved_in_flow_tail,
        "return; function f { Add-Type 'L' }",
        context = nested_context(
            script_context(35),
            ContextKind::Function("f".into()),
            (8, 35)
        ),
        CoreReport {
            unresolved: vec![Unresolved {
                site: positional((21, 33), (21, 29), (30, 33)),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                context: context.clone(),
            }],
            coverage: vec![
                gap(35, (0, 6), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(8, 35),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context,
                },
            ],
            ..CoreReport::default()
        }
    );

    context_case!(
        analyze_conditional_function_definition_keeps_literal_site_discovery_only,
        "if ($x) { function f { Add-Type 'L' } }; Add-Type 'S'",
        context = nested_context(
            script_context(53),
            ContextKind::Function("f".into()),
            (10, 37)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                53,
                positional((41, 53), (41, 49), (50, 53)),
                "S",
                literal((50, 53), (51, 52)),
                vec![],
                vec![],
            )],
            unresolved: vec![Unresolved {
                site: positional((23, 35), (23, 31), (32, 35)),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                context: context.clone(),
            }],
            coverage: vec![
                gap(53, (0, 39), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(10, 37),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context,
                },
            ],
        }
    );

    context_case!(
        analyze_function_inside_literal_block_is_discovered_not_isolated_as_reachable,
        "{ function f { Add-Type 'L' } }; Add-Type 'S'",
        block_context = nested_context(script_context(45), ContextKind::ScriptBlock, (0, 31)),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                45,
                positional((33, 45), (33, 41), (42, 45)),
                "S",
                literal((42, 45), (43, 44)),
                vec![],
                vec![],
            )],
            unresolved: vec![Unresolved {
                site: positional((15, 27), (15, 23), (24, 27)),
                reason: DiscoveryReason::DeferredContext,
                context: nested_context(
                    block_context.clone(),
                    ContextKind::Function("f".into()),
                    (2, 29),
                ),
            }],
            coverage: vec![
                CoverageGap {
                    span: Span::new(0, 31),
                    reason: DiscoveryReason::DeferredContext,
                    context: block_context.clone(),
                },
                CoverageGap {
                    span: Span::new(2, 29),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: nested_context(
                        block_context,
                        ContextKind::Function("f".into()),
                        (2, 29),
                    ),
                },
            ],
        }
    );

    context_case!(
        analyze_workflow_body_is_unsupported_deferred_without_writing_or_killing_script,
        "$a='X'; workflow w { $a='Y'; Add-Type $a }; Add-Type $a",
        context = nested_context(
            script_context(55),
            ContextKind::Workflow("w".into()),
            (8, 42)
        ),
        CoreReport {
            hypotheses: vec![],
            confirmed: vec![known(
                55,
                positional((44, 55), (44, 52), (53, 55)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((53, 55), "$a")],
            )],
            unresolved: vec![Unresolved {
                site: positional((29, 40), (29, 37), (38, 40)),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::Workflow),
                context: context.clone(),
            }],
            coverage: vec![CoverageGap {
                span: Span::new(8, 42),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::Workflow),
                context,
            }],
        }
    );

    context_case!(
        analyze_function_inside_workflow_remains_unsupported_discovery_only,
        "workflow w { function f { Add-Type 'D' } }",
        context = nested_context(
            script_context(42),
            ContextKind::Workflow("w".into()),
            (0, 42)
        ),
        CoreReport {
            unresolved: vec![Unresolved {
                site: positional((26, 38), (26, 34), (35, 38)),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::Workflow),
                context: nested_context(
                    context.clone(),
                    ContextKind::Function("f".into()),
                    (13, 40),
                ),
            }],
            coverage: vec![
                CoverageGap {
                    span: Span::new(0, 42),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Workflow),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(13, 40),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: nested_context(context, ContextKind::Function("f".into()), (13, 40),),
                },
            ],
            ..CoreReport::default()
        }
    );

    macro_rules! keyword_barrier_cases {
        ($(($name:ident, $source:literal, $keyword:literal, $end:literal, $block_end:literal,
            $nested:expr, $nested_name:expr, $nested_argument:expr,
            $tail:expr, $tail_name:expr, $tail_argument:expr)),+ $(,)?) => {
            $(context_case!(
                $name, $source,
                context = nested_context(
                    script_context($end),
                    ContextKind::KeywordBlock($keyword.into()),
                    (8, $block_end),
                ),
                CoreReport {
                    unresolved: vec![
                        Unresolved {
                            site: positional($nested, $nested_name, $nested_argument),
                            reason: DiscoveryReason::UnsupportedRegion(RegionKind::KeywordBlock),
                            context: context.clone(),
                        },
                        unknown(
                            $end, positional($tail, $tail_name, $tail_argument),
                            ValueReason::VariableNotKnown,
                        ),
                    ],
                    coverage: vec![CoverageGap {
                        span: Span::new(8, $block_end),
                        reason: DiscoveryReason::UnsupportedRegion(RegionKind::KeywordBlock),
                        context,
                    }],
                    ..CoreReport::default()
                }
            );)+
        };
    }

    keyword_barrier_cases!(
        (
            analyze_trap_is_keyword_barrier_not_innocent_literal_block,
            "$a='X'; trap { $a='Y'; Add-Type 'D' }; Add-Type $a",
            "trap",
            50,
            37,
            (23, 35),
            (23, 31),
            (32, 35),
            (39, 50),
            (39, 47),
            (48, 50)
        ),
        (
            analyze_data_header_is_keyword_barrier_not_executed_body,
            "$a='X'; data d { $a='Y'; Add-Type 'D' }; Add-Type $a",
            "data",
            52,
            39,
            (25, 37),
            (25, 33),
            (34, 37),
            (41, 52),
            (41, 49),
            (50, 52)
        ),
        (
            analyze_dynamicparam_is_keyword_barrier_not_executed_body,
            "$a='X'; dynamicparam { $a='Y'; Add-Type 'D' }; Add-Type $a",
            "dynamicparam",
            58,
            45,
            (31, 43),
            (31, 39),
            (40, 43),
            (47, 58),
            (47, 55),
            (56, 58)
        ),
        (
            analyze_begin_is_keyword_barrier_not_executed_body,
            "$a='X'; begin { $a='Y'; Add-Type 'D' }; Add-Type $a",
            "begin",
            51,
            38,
            (24, 36),
            (24, 32),
            (33, 36),
            (40, 51),
            (40, 48),
            (49, 51)
        ),
        (
            analyze_process_is_keyword_barrier_not_executed_body,
            "$a='X'; process { $a='Y'; Add-Type 'D' }; Add-Type $a",
            "process",
            53,
            40,
            (26, 38),
            (26, 34),
            (35, 38),
            (42, 53),
            (42, 50),
            (51, 53)
        ),
        (
            analyze_end_is_keyword_barrier_not_executed_body,
            "$a='X'; end { $a='Y'; Add-Type 'D' }; Add-Type $a",
            "end",
            49,
            36,
            (22, 34),
            (22, 30),
            (31, 34),
            (38, 49),
            (38, 46),
            (47, 49)
        ),
    );

    transfer_case!(
        analyze_function_definition_does_not_write_or_kill_parent_environment,
        "$a='X'; function f { $a='Y' }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                42,
                positional((31, 42), (31, 39), (40, 42)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((40, 42), "$a")],
            )],
            ..CoreReport::default()
        }
    );

    transfer_case!(
        analyze_literal_block_does_not_write_or_kill_parent_environment,
        "$a='X'; { $a='Y' }; Add-Type $a",
        CoreReport {
            confirmed: vec![known(
                31,
                positional((20, 31), (20, 28), (29, 31)),
                "X",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((29, 31), "$a")],
            )],
            coverage: vec![CoverageGap {
                span: Span::new(8, 18),
                reason: DiscoveryReason::DeferredContext,
                context: Context {
                    path: vec![
                        ContextFrame {
                            kind: ContextKind::Script,
                            span: Span::new(0, 31),
                            status: ContextStatus::DiscoveryOnly,
                        },
                        ContextFrame {
                            kind: ContextKind::ScriptBlock,
                            span: Span::new(8, 18),
                            status: ContextStatus::Deferred,
                        },
                    ],
                },
            }],
            ..CoreReport::default()
        }
    );

    #[test]
    fn analyze_class_declaration_kills_parent_but_method_remains_deferred_unresolved() {
        let source = "$a='X'; class C { [void] M() { Add-Type 'Y' } }; Add-Type $a";
        let class_context = Context {
            path: vec![
                ContextFrame {
                    kind: ContextKind::Script,
                    span: Span::new(0, 60),
                    status: ContextStatus::DiscoveryOnly,
                },
                ContextFrame {
                    kind: ContextKind::Class("C".into()),
                    span: Span::new(8, 47),
                    status: ContextStatus::Deferred,
                },
            ],
        };
        let member_context = Context {
            path: vec![
                ContextFrame {
                    kind: ContextKind::Script,
                    span: Span::new(0, 60),
                    status: ContextStatus::DiscoveryOnly,
                },
                ContextFrame {
                    kind: ContextKind::Class("C".into()),
                    span: Span::new(8, 47),
                    status: ContextStatus::Deferred,
                },
                ContextFrame {
                    kind: ContextKind::ClassMember("M".into()),
                    span: Span::new(18, 45),
                    status: ContextStatus::Deferred,
                },
            ],
        };
        let expected = CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((31, 43), (31, 39), (40, 43)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Declaration),
                    context: member_context.clone(),
                },
                unknown(
                    60,
                    positional((49, 60), (49, 57), (58, 60)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                CoverageGap {
                    span: Span::new(8, 47),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Declaration),
                    context: class_context,
                },
                CoverageGap {
                    span: Span::new(18, 45),
                    reason: DiscoveryReason::DeferredContext,
                    context: member_context,
                },
            ],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_enum_declaration_kills_parent_without_transferring_member_assignment() {
        let source = "$a='X'; enum E { A=1 }; Add-Type $a";
        let context = Context {
            path: vec![
                ContextFrame {
                    kind: ContextKind::Script,
                    span: Span::new(0, 35),
                    status: ContextStatus::DiscoveryOnly,
                },
                ContextFrame {
                    kind: ContextKind::Enum("E".into()),
                    span: Span::new(8, 22),
                    status: ContextStatus::Deferred,
                },
            ],
        };
        let expected = CoreReport {
            unresolved: vec![unknown(
                35,
                positional((24, 35), (24, 32), (33, 35)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                CoverageGap {
                    span: Span::new(8, 22),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Declaration),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(17, 20),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::UnsupportedWrite),
                    context,
                },
            ],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    transfer_case!(
        analyze_unsupported_write_discovers_rhs_without_confirming_old_environment,
        "$a='X'; ${b}=Add-Type $a; Add-Type $a",
        CoreReport {
            unresolved: vec![
                Unresolved {
                    site: positional((13, 24), (13, 21), (22, 24)),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::UnsupportedWrite),
                    context: script_context(37),
                },
                unknown(
                    37,
                    positional((26, 37), (26, 34), (35, 37)),
                    ValueReason::VariableNotKnown
                ),
            ],
            coverage: vec![
                gap(37, (8, 24), RegionKind::UnsupportedWrite),
                gap(37, (8, 12), RegionKind::VariableRead),
            ],
            ..CoreReport::default()
        }
    );

    fn script_context(end: usize) -> Context {
        Context {
            path: vec![ContextFrame {
                kind: ContextKind::Script,
                span: Span::new(0, end),
                status: ContextStatus::DiscoveryOnly,
            }],
        }
    }

    fn unresolved(
        command_span: Span,
        name_span: Span,
        argument_span: Span,
        reason: DiscoveryReason,
        context: Context,
    ) -> Unresolved {
        Unresolved {
            site: AddTypeSite {
                command_span,
                name_span,
                argument_span: Some(argument_span),
                parameter_span: None,
            },
            reason,
            context,
        }
    }

    #[test]
    fn analyze_empty_source_returns_empty_owned_report() {
        let source = "";

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(CoreReport::default()));
    }

    #[test]
    fn analyze_literal_add_type_confirms_distinct_spans_without_metadata_duplicate() {
        let source = "Add-Type -TypeDefinition 'X'";
        let expected = CoreReport {
            confirmed: vec![Confirmed {
                site: AddTypeSite {
                    command_span: Span::new(0, 28),
                    name_span: Span::new(0, 8),
                    argument_span: Some(Span::new(25, 28)),
                    parameter_span: Some(Span::new(9, 28)),
                },
                raw_body: "X".into(),
                literal_origins: vec![LiteralOrigin {
                    literal_span: Span::new(25, 28),
                    body_span: Span::new(26, 27),
                }],
                assignment_origins: vec![],
                reference_origins: vec![],
                context: script_context(28),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_bom_unicode_site_preserves_original_byte_spans_and_owns_report() {
        let source = String::from("\u{feff}aDd-TyPe -TypeDefinition 'Ż'");
        let expected = CoreReport {
            confirmed: vec![Confirmed {
                site: AddTypeSite {
                    command_span: Span::new(3, 32),
                    name_span: Span::new(3, 11),
                    argument_span: Some(Span::new(28, 32)),
                    parameter_span: Some(Span::new(12, 32)),
                },
                raw_body: "Ż".into(),
                literal_origins: vec![LiteralOrigin {
                    literal_span: Span::new(28, 32),
                    body_span: Span::new(29, 31),
                }],
                assignment_origins: vec![],
                reference_origins: vec![],
                context: script_context(32),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(&source);
        drop(source);

        assert_eq!(actual, Ok(expected));
    }

    macro_rules! literal_cases {
        ($(($name:ident, $source:literal, $raw:literal, $end:literal,
            $expression:expr, $literal:expr, $body:expr, $spelling:literal,
            $fragment:expr)),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    let source = $source;
                    let literal_span: Span = $literal;
                    let body_span: Span = $body;
                    let fragment: Option<Span> = $fragment;
                    let context = script_context($end);
                    let expected = CoreReport {
                        confirmed: vec![Confirmed {
                            site: AddTypeSite {
                                command_span: Span::new(0, $end),
                                name_span: Span::new(0, 8),
                                argument_span: Some($expression),
                                parameter_span: None,
                            },
                            raw_body: $raw.into(),
                            literal_origins: vec![LiteralOrigin {
                                literal_span,
                                body_span,
                            }],
                            assignment_origins: vec![],
                            reference_origins: vec![],
                            context: context.clone(),
                        }],
                        coverage: fragment.into_iter().map(|span| CoverageGap {
                            span,
                            reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                            context: context.clone(),
                        }).collect(),
                        ..CoreReport::default()
                    };

                    let actual = analyze_add_type(source);

                    assert_eq!(actual, Ok(expected));
                    assert_eq!(&source[literal_span.start..literal_span.end], $spelling);
                    assert_eq!(&source[body_span.start..body_span.end], $raw);
                }
            )+
        };
    }

    literal_cases!(
        (
            analyze_single_literal_preserves_doubled_apostrophe,
            "Add-Type 'a''b'",
            "a''b",
            15,
            Span::new(9, 15),
            Span::new(9, 15),
            Span::new(10, 14),
            "'a''b'",
            None
        ),
        (
            analyze_single_literal_preserves_dollars_and_backticks,
            "Add-Type '$x`'",
            "$x`",
            14,
            Span::new(9, 14),
            Span::new(9, 14),
            Span::new(10, 13),
            "'$x`'",
            None
        ),
        (
            analyze_empty_single_literal_has_real_zero_width_body,
            "Add-Type ''",
            "",
            11,
            Span::new(9, 11),
            Span::new(9, 11),
            Span::new(10, 10),
            "''",
            None
        ),
        (
            analyze_empty_double_literal_keeps_fragment_coverage,
            "Add-Type \"\"",
            "",
            11,
            Span::new(9, 11),
            Span::new(9, 11),
            Span::new(10, 10),
            "\"\"",
            Some(Span::new(9, 11))
        ),
        (
            analyze_double_literal_preserves_unicode_and_raw_escape,
            "Add-Type \"Ż`n\"",
            "Ż`n",
            15,
            Span::new(9, 15),
            Span::new(9, 15),
            Span::new(10, 14),
            "\"Ż`n\"",
            Some(Span::new(9, 15))
        ),
        (
            analyze_double_literal_preserves_escaped_dollar,
            "Add-Type \"`$x\"",
            "`$x",
            14,
            Span::new(9, 14),
            Span::new(9, 14),
            Span::new(10, 13),
            "\"`$x\"",
            Some(Span::new(9, 14))
        ),
        (
            analyze_double_literal_with_odd_backticks_is_raw_constant,
            "Add-Type \"```$x\"",
            "```$x",
            16,
            Span::new(9, 16),
            Span::new(9, 16),
            Span::new(10, 15),
            "\"```$x\"",
            Some(Span::new(9, 16))
        ),
        (
            analyze_single_here_literal_preserves_unicode_crlf_and_dollars,
            "Add-Type @'\r\nŻ\r\n$x`\r\n'@",
            "Ż\r\n$x`",
            24,
            Span::new(9, 24),
            Span::new(9, 24),
            Span::new(13, 20),
            "@'\r\nŻ\r\n$x`\r\n'@",
            None
        ),
        (
            analyze_double_here_literal_trims_only_boundary_newlines,
            "Add-Type @\"\r\nŻ\r\n`$x\r\n\"@",
            "Ż\r\n`$x",
            24,
            Span::new(9, 24),
            Span::new(9, 24),
            Span::new(13, 20),
            "@\"\r\nŻ\r\n`$x\r\n\"@",
            Some(Span::new(9, 24))
        ),
        (
            analyze_empty_single_here_literal_has_real_zero_width_body,
            "Add-Type @'\n'@",
            "",
            14,
            Span::new(9, 14),
            Span::new(9, 14),
            Span::new(12, 12),
            "@'\n'@",
            None
        ),
        (
            analyze_empty_double_here_literal_has_real_zero_width_body,
            "Add-Type @\"\r\n\"@",
            "",
            15,
            Span::new(9, 15),
            Span::new(9, 15),
            Span::new(13, 13),
            "@\"\r\n\"@",
            Some(Span::new(9, 15))
        ),
        (
            analyze_nested_paren_literal_keeps_expression_separate_from_origin,
            "Add-Type (('X'))",
            "X",
            16,
            Span::new(9, 16),
            Span::new(11, 14),
            Span::new(12, 13),
            "'X'",
            None
        ),
        (
            analyze_single_here_literal_trims_only_one_newline_at_each_boundary,
            "Add-Type @'\n\nX\n\n'@",
            "\nX\n",
            18,
            Span::new(9, 18),
            Span::new(9, 18),
            Span::new(12, 15),
            "@'\n\nX\n\n'@",
            None
        ),
    );

    #[test]
    fn analyze_full_member_parameter_is_ascii_case_insensitive() {
        let source = "aDd-TyPe -mEmBeRdEfInItIoN (('X'))";
        let expected = CoreReport {
            confirmed: vec![Confirmed {
                site: AddTypeSite {
                    command_span: Span::new(0, 34),
                    name_span: Span::new(0, 8),
                    argument_span: Some(Span::new(27, 34)),
                    parameter_span: Some(Span::new(9, 34)),
                },
                raw_body: "X".into(),
                literal_origins: vec![LiteralOrigin {
                    literal_span: Span::new(29, 32),
                    body_span: Span::new(30, 31),
                }],
                assignment_origins: vec![],
                reference_origins: vec![],
                context: script_context(34),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_named_backtick_continuation_uses_parser_bound_argument() {
        let source = "Add-Type -TypeDefinition `\r\n'X'";
        let expected = CoreReport {
            confirmed: vec![Confirmed {
                site: AddTypeSite {
                    command_span: Span::new(0, 31),
                    name_span: Span::new(0, 8),
                    argument_span: Some(Span::new(28, 31)),
                    parameter_span: Some(Span::new(9, 31)),
                },
                raw_body: "X".into(),
                literal_origins: vec![LiteralOrigin {
                    literal_span: Span::new(28, 31),
                    body_span: Span::new(29, 30),
                }],
                assignment_origins: vec![],
                reference_origins: vec![],
                context: script_context(31),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    macro_rules! argument_failure_cases {
        ($(($name:ident, $source:literal, $end:literal, $command_end:literal,
            $parameter:expr, $reason:ident)),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    let source = $source;
                    let expected = CoreReport {
                        unresolved: vec![Unresolved {
                            site: AddTypeSite {
                                command_span: Span::new(0, $command_end),
                                name_span: Span::new(0, 8),
                                argument_span: None,
                                parameter_span: $parameter,
                            },
                            reason: DiscoveryReason::InvalidArguments(ArgumentReason::$reason),
                            context: script_context($end),
                        }],
                        ..CoreReport::default()
                    };

                    let actual = analyze_add_type(source);

                    assert_eq!(actual, Ok(expected));
                }
            )+
        };
    }

    argument_failure_cases!(
        (
            analyze_missing_positional_is_unresolved,
            "Add-Type",
            8,
            8,
            None,
            MissingArgument
        ),
        (
            analyze_multiple_positional_never_selects_first_literal,
            "Add-Type 'X' 'Y'",
            16,
            16,
            None,
            ExtraElements
        ),
        (
            analyze_named_and_positional_competition_is_unresolved,
            "Add-Type -TypeDefinition 'X' 'Y'",
            32,
            32,
            None,
            ExtraElements
        ),
        (
            analyze_extra_switch_is_unresolved,
            "Add-Type 'X' -PassThru",
            22,
            22,
            None,
            ExtraElements
        ),
        (
            analyze_duplicate_parameters_are_unresolved,
            "Add-Type -TypeDefinition 'X' -TypeDefinition 'Y'",
            48,
            48,
            None,
            ExtraElements
        ),
        (
            analyze_competing_named_parameters_are_unresolved,
            "Add-Type -TypeDefinition 'X' -MemberDefinition 'Y'",
            50,
            50,
            None,
            ExtraElements
        ),
        (
            analyze_abbreviated_type_parameter_is_unresolved,
            "Add-Type -Type 'X'",
            18,
            18,
            Some(Span::new(9, 18)),
            UnsupportedParameter
        ),
        (
            analyze_abbreviated_member_parameter_is_unresolved,
            "Add-Type -Member 'X'",
            20,
            20,
            Some(Span::new(9, 20)),
            UnsupportedParameter
        ),
        (
            analyze_unrelated_parameter_never_falls_back_to_its_literal,
            "Add-Type -Name 'X'",
            18,
            18,
            Some(Span::new(9, 18)),
            UnsupportedParameter
        ),
        (
            analyze_colon_type_parameter_checks_raw_token_not_normalized_name,
            "Add-Type -TypeDefinition:'X'",
            28,
            28,
            Some(Span::new(9, 28)),
            ColonParameter
        ),
        (
            analyze_colon_member_parameter_is_unresolved_even_with_space,
            "Add-Type -MemberDefinition: 'X'",
            31,
            31,
            Some(Span::new(9, 31)),
            ColonParameter
        ),
        (
            analyze_missing_named_argument_is_unresolved,
            "Add-Type -TypeDefinition",
            24,
            24,
            Some(Span::new(9, 24)),
            MissingParameterArgument
        ),
        (
            analyze_missing_member_argument_is_unresolved,
            "Add-Type -MemberDefinition",
            26,
            26,
            Some(Span::new(9, 26)),
            MissingParameterArgument
        ),
        (
            analyze_newline_does_not_bind_next_statement_as_argument,
            "Add-Type -TypeDefinition\n'X'",
            28,
            24,
            Some(Span::new(9, 24)),
            MissingParameterArgument
        ),
        (
            analyze_crlf_does_not_bind_next_statement_as_argument,
            "Add-Type -TypeDefinition\r\n'X'",
            29,
            24,
            Some(Span::new(9, 24)),
            MissingParameterArgument
        ),
        (
            analyze_missing_argument_before_competing_parameter_has_no_fallback,
            "Add-Type -TypeDefinition -MemberDefinition 'X'",
            46,
            46,
            None,
            ExtraElements
        ),
    );

    macro_rules! unknown_value_cases {
        ($(($name:ident, $source:literal, $end:literal, $reason:ident,
            [$($gap_span:expr => $gap_reason:expr),* $(,)?])),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    let source = $source;
                    let context = script_context($end);
                    let expected = CoreReport {
                        unresolved: vec![unresolved(
                            Span::new(0, $end),
                            Span::new(0, 8),
                            Span::new(9, $end),
                            DiscoveryReason::UnknownValue(ValueReason::$reason),
                            context.clone(),
                        )],
                        coverage: vec![$(CoverageGap {
                            span: $gap_span,
                            reason: $gap_reason,
                            context: context.clone(),
                        }),*],
                        ..CoreReport::default()
                    };

                    let actual = analyze_add_type(source);

                    assert_eq!(actual, Ok(expected));
                }
            )+
        };
    }

    unknown_value_cases!(
        (
            analyze_active_dollar_literal_is_unknown,
            "Add-Type \"x$y\"", 14, NonConstantString,
            [Span::new(9, 14) => DiscoveryReason::FragmentDiagnosticsUnavailable]
        ),
        (
            analyze_bare_dollar_literal_is_unknown_despite_empty_parts,
            "Add-Type \"$\"", 12, NonConstantString,
            [Span::new(9, 12) => DiscoveryReason::FragmentDiagnosticsUnavailable]
        ),
        (
            analyze_even_backticks_leave_dollar_active,
            "Add-Type \"``$x\"", 15, NonConstantString,
            [Span::new(9, 15) => DiscoveryReason::FragmentDiagnosticsUnavailable]
        ),
        (
            analyze_escaped_dollar_does_not_hide_later_active_dollar,
            "Add-Type \"`$x $y\"", 17, NonConstantString,
            [Span::new(9, 17) => DiscoveryReason::FragmentDiagnosticsUnavailable]
        ),
        (
            analyze_active_here_dollar_is_unknown,
            "Add-Type @\"\n$x\n\"@", 17, NonConstantString,
            [Span::new(9, 17) => DiscoveryReason::FragmentDiagnosticsUnavailable]
        ),
        (
            analyze_unicode_interpolation_is_unknown,
            "Add-Type \"$变量\"", 18, NonConstantString,
            [Span::new(9, 18) => DiscoveryReason::FragmentDiagnosticsUnavailable]
        ),
        (
            analyze_interpolated_subexpression_never_infers_literal,
            "Add-Type \"$('X')\"", 17, NonConstantString,
            [
                Span::new(9, 17) => DiscoveryReason::FragmentDiagnosticsUnavailable,
                Span::new(10, 16) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
            ]
        ),
        (
            analyze_numeric_argument_is_unknown_without_effects,
            "Add-Type 42", 11, SimpleNonString, []
        ),
        (
            analyze_number_expression_is_unknown_without_effects,
            "Add-Type (42)", 13, SimpleNonString, []
        ),
        (
            analyze_type_expression_is_unknown_without_effects,
            "Add-Type [string]", 17, SimpleNonString, []
        ),
        (
            analyze_bareword_is_unknown_without_effects,
            "Add-Type word", 13, SimpleNonString, []
        ),
        (
            analyze_nested_paren_number_is_unknown_without_effects,
            "Add-Type ((42))", 15, SimpleNonString, []
        ),
        (
            analyze_nested_paren_variable_remains_unknown,
            "Add-Type (($code))", 18, VariableNotKnown, []
        ),
        (
            analyze_concatenation_is_unsupported_not_folded,
            "Add-Type ('X' + 'Y')", 20, UnsupportedExpression,
            [
                Span::new(9, 20) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                Span::new(10, 19) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
            ]
        ),
        (
            analyze_cast_is_unsupported_not_converted,
            "Add-Type ([string]'X')", 22, UnsupportedExpression,
            [
                Span::new(9, 22) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                Span::new(10, 21) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
            ]
        ),
        (
            analyze_array_is_unsupported_not_converted,
            "Add-Type @('X')", 15, UnsupportedExpression,
            [Span::new(9, 15) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression)]
        ),
        (
            analyze_array_literal_is_unsupported_not_converted,
            "Add-Type ('X','Y')", 18, UnsupportedExpression,
            [
                Span::new(9, 18) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                Span::new(10, 17) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
            ]
        ),
        (
            analyze_subexpression_is_unsupported_not_unwrapped,
            "Add-Type $('X')", 15, UnsupportedExpression,
            [Span::new(9, 15) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression)]
        ),
        (
            analyze_pipeline_is_unsupported_not_inferred_from_output,
            "Add-Type ('X' | Out-String)", 27, UnsupportedExpression,
            [
                Span::new(9, 27) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                Span::new(10, 26) => DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow),
                Span::new(16, 26) => DiscoveryReason::UnsupportedRegion(RegionKind::OtherCommand),
            ]
        ),
        (
            analyze_method_call_is_unsupported_not_evaluated,
            "Add-Type ('X'.ToString())", 25, UnsupportedExpression,
            [
                Span::new(9, 25) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                Span::new(10, 24) => DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
            ]
        ),
    );

    #[test]
    fn analyze_assignment_confirms_from_environment_with_source_provenance() {
        let source = "$code = 'X'\nAdd-Type $code";
        let expected = CoreReport {
            confirmed: vec![Confirmed {
                site: AddTypeSite {
                    command_span: Span::new(12, 26),
                    name_span: Span::new(12, 20),
                    argument_span: Some(Span::new(21, 26)),
                    parameter_span: None,
                },
                raw_body: "X".into(),
                literal_origins: vec![LiteralOrigin {
                    literal_span: Span::new(8, 11),
                    body_span: Span::new(9, 10),
                }],
                assignment_origins: vec![AssignmentOrigin {
                    assignment_span: Span::new(0, 11),
                    target_span: Span::new(0, 5),
                    value_span: Span::new(8, 11),
                    raw_target: "$code".into(),
                }],
                reference_origins: vec![ReferenceOrigin {
                    span: Span::new(21, 26),
                    raw_name: "$code".into(),
                }],
                context: script_context(26),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_full_named_unknown_value_preserves_parameter_and_expression_spans() {
        let source = "Add-Type -TypeDefinition $code";
        let expected = CoreReport {
            unresolved: vec![Unresolved {
                site: AddTypeSite {
                    command_span: Span::new(0, 30),
                    name_span: Span::new(0, 8),
                    argument_span: Some(Span::new(25, 30)),
                    parameter_span: Some(Span::new(9, 30)),
                },
                reason: DiscoveryReason::UnknownValue(ValueReason::VariableNotKnown),
                context: script_context(30),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_parenthesized_command_keeps_nested_literal_discovery_only() {
        let source = "Add-Type (Add-Type 'X')";
        let context = script_context(23);
        let reason = DiscoveryReason::UnsupportedRegion(RegionKind::Expression);
        let expected = CoreReport {
            unresolved: vec![
                unresolved(
                    Span::new(0, 23),
                    Span::new(0, 8),
                    Span::new(9, 23),
                    DiscoveryReason::UnknownValue(ValueReason::UnsupportedExpression),
                    context.clone(),
                ),
                unresolved(
                    Span::new(10, 22),
                    Span::new(10, 18),
                    Span::new(19, 22),
                    reason,
                    context.clone(),
                ),
            ],
            coverage: vec![CoverageGap {
                span: Span::new(9, 23),
                reason,
                context,
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_parenthesized_assignment_keeps_nested_literal_discovery_only() {
        let source = "Add-Type ($x = Add-Type 'X')";
        let context = script_context(28);
        let reason = DiscoveryReason::UnsupportedRegion(RegionKind::Expression);
        let expected = CoreReport {
            unresolved: vec![
                unresolved(
                    Span::new(0, 28),
                    Span::new(0, 8),
                    Span::new(9, 28),
                    DiscoveryReason::UnknownValue(ValueReason::UnsupportedExpression),
                    context.clone(),
                ),
                unresolved(
                    Span::new(15, 27),
                    Span::new(15, 23),
                    Span::new(24, 27),
                    reason,
                    context.clone(),
                ),
            ],
            coverage: vec![CoverageGap {
                span: Span::new(9, 28),
                reason,
                context,
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_unrecognized_command_spellings_never_create_add_type_findings() {
        let cases = [
            (
                "Microsoft.PowerShell.Utility\\Add-Type 'X'",
                41,
                RegionKind::OtherCommand,
            ),
            (". Add-Type 'X'", 14, RegionKind::Invocation),
            ("& 'Add-Type' 'X'", 16, RegionKind::Invocation),
            ("addtype 'X'", 11, RegionKind::OtherCommand),
            ("Ädd-Type 'X'", 13, RegionKind::OtherCommand),
        ];
        for (source, end, region) in cases {
            let expected = CoreReport {
                coverage: vec![CoverageGap {
                    span: Span::new(0, end),
                    reason: DiscoveryReason::UnsupportedRegion(region),
                    context: script_context(end),
                }],
                ..CoreReport::default()
            };

            let actual = analyze_add_type(source);

            assert_eq!(actual, Ok(expected), "{source:?}");
        }
    }

    #[test]
    fn analyze_unknown_variable_add_type_is_unknown_not_syntax_or_confirmed() {
        let source = "Add-Type $unknown";
        let expected = CoreReport {
            unresolved: vec![unresolved(
                Span::new(0, 17),
                Span::new(0, 8),
                Span::new(9, 17),
                DiscoveryReason::UnknownValue(ValueReason::VariableNotKnown),
                script_context(17),
            )],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_unterminated_unicode_string_preserves_owned_lex_error() {
        let source = "\u{feff}'Ż";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![LexError {
                span: Span::new(3, 6),
                message: "unterminated string".into(),
            }],
            parse_errors: vec![],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_multiple_lex_errors_preserves_original_order_and_types() {
        let source = "\u{feff}0x; <# Ż";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![
                LexError {
                    span: Span::new(3, 5),
                    message: "radix prefix with no digits".into(),
                },
                LexError {
                    span: Span::new(7, 12),
                    message: "unterminated block comment".into(),
                },
            ],
            parse_errors: vec![],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_parser_errors_preserves_original_order_without_lex_errors() {
        let source = "param(";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![],
            parse_errors: vec![ParseError {
                span: Span::new(6, 6),
                message: "expected ')'".into(),
            }],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_lex_and_parser_errors_keeps_separate_original_vectors() {
        let source = "\u{feff}param( <# Ż";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![LexError {
                span: Span::new(10, 15),
                message: "unterminated block comment".into(),
            }],
            parse_errors: vec![ParseError {
                span: Span::new(15, 15),
                message: "expected ')'".into(),
            }],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_nested_lex_errors_keeps_unwind_order_not_span_order() {
        let source = "\u{feff}\"$('Ż";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![
                LexError {
                    span: Span::new(6, 9),
                    message: "unterminated string".into(),
                },
                LexError {
                    span: Span::new(4, 9),
                    message: "unterminated subexpression in string".into(),
                },
                LexError {
                    span: Span::new(3, 9),
                    message: "unterminated string".into(),
                },
            ],
            parse_errors: vec![],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_multiple_parser_errors_keeps_original_diagnostic_order() {
        let source = "([T]::";
        let expected = AnalysisError::Syntax {
            lex_errors: vec![],
            parse_errors: vec![
                ParseError {
                    span: Span::new(6, 6),
                    message: "expected member name".into(),
                },
                ParseError {
                    span: Span::new(6, 6),
                    message: "expected ')'".into(),
                },
            ],
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Err(expected));
    }

    #[test]
    fn analyze_nested_subexpression_discovers_each_site_once_without_transfer() {
        let source = "Add-Type $(Add-Type 'X')";
        let context = script_context(24);
        let reason = DiscoveryReason::UnsupportedRegion(RegionKind::Expression);
        let expected = CoreReport {
            unresolved: vec![
                unresolved(
                    Span::new(0, 24),
                    Span::new(0, 8),
                    Span::new(9, 24),
                    DiscoveryReason::UnknownValue(ValueReason::UnsupportedExpression),
                    context.clone(),
                ),
                unresolved(
                    Span::new(11, 23),
                    Span::new(11, 19),
                    Span::new(20, 23),
                    reason,
                    context.clone(),
                ),
            ],
            coverage: vec![CoverageGap {
                span: Span::new(9, 24),
                reason,
                context,
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_branch_literal_is_unresolved_inside_unsupported_region() {
        let source = "if ($x) { Add-Type 'X' }";
        let context = script_context(24);
        let reason = DiscoveryReason::UnsupportedRegion(RegionKind::ControlFlow);
        let expected = CoreReport {
            unresolved: vec![unresolved(
                Span::new(10, 22),
                Span::new(10, 18),
                Span::new(19, 22),
                reason,
                context.clone(),
            )],
            coverage: vec![CoverageGap {
                span: Span::new(0, 24),
                reason,
                context,
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_interpolation_discovers_absolute_sites_with_diagnostic_gap() {
        let source = "\u{feff}\"Ż $(Add-Type 'X')\"";
        let context = script_context(23);
        let reason = DiscoveryReason::FragmentDiagnosticsUnavailable;
        let expected = CoreReport {
            unresolved: vec![unresolved(
                Span::new(9, 21),
                Span::new(9, 17),
                Span::new(18, 21),
                reason,
                context.clone(),
            )],
            coverage: vec![
                CoverageGap {
                    span: Span::new(3, 23),
                    reason,
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(7, 22),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context,
                },
            ],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_expandable_without_parts_still_records_fragment_coverage_gap() {
        let source = "\"Ż\"";
        let expected = CoreReport {
            coverage: vec![CoverageGap {
                span: Span::new(0, 4),
                reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                context: script_context(4),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_fragment_error_returns_coverage_not_fabricated_syntax_error() {
        // A prefix operator forces expression parsing; a standalone `]`
        // instead follows the parser's command-name recovery path.
        let source = "\"$(+])\"";
        let context = script_context(7);
        let expected = CoreReport {
            coverage: vec![
                CoverageGap {
                    span: Span::new(0, 7),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(1, 6),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(3, 5),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(4, 5),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::FragmentError),
                    context,
                },
            ],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_dynamic_invocation_has_coverage_without_invented_add_type_site() {
        let source = "& Add-Type 'X'";
        let expected = CoreReport {
            coverage: vec![CoverageGap {
                span: Span::new(0, 14),
                reason: DiscoveryReason::UnsupportedRegion(RegionKind::Invocation),
                context: script_context(14),
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_redirection_discovers_target_site_under_whole_command_barrier() {
        let source = "Add-Type 'X' > $(Add-Type 'Y')";
        let context = script_context(30);
        let reason = DiscoveryReason::UnsupportedRegion(RegionKind::Redirection);
        let expected = CoreReport {
            unresolved: vec![
                unresolved(
                    Span::new(0, 30),
                    Span::new(0, 8),
                    Span::new(9, 12),
                    reason,
                    context.clone(),
                ),
                unresolved(
                    Span::new(17, 29),
                    Span::new(17, 25),
                    Span::new(26, 29),
                    reason,
                    context.clone(),
                ),
            ],
            coverage: vec![
                CoverageGap {
                    span: Span::new(0, 30),
                    reason,
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(13, 30),
                    reason,
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(15, 30),
                    reason: DiscoveryReason::UnsupportedRegion(RegionKind::Expression),
                    context,
                },
            ],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_function_literal_is_confirmed_only_in_deferred_definition_context() {
        let source = "function f { Add-Type 'X' }";
        let context = Context {
            path: vec![
                ContextFrame {
                    kind: ContextKind::Script,
                    span: Span::new(0, 27),
                    status: ContextStatus::DiscoveryOnly,
                },
                ContextFrame {
                    kind: ContextKind::Function("f".into()),
                    span: Span::new(0, 27),
                    status: ContextStatus::Deferred,
                },
            ],
        };
        let expected = CoreReport {
            confirmed: vec![Confirmed {
                context,
                ..known(
                    27,
                    positional((13, 25), (13, 21), (22, 25)),
                    "X",
                    literal((22, 25), (23, 24)),
                    vec![],
                    vec![],
                )
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_keyword_contexts_uses_tokens_not_shared_ast_variant_as_execution_proof() {
        let cases = [
            (
                "workflow w { Add-Type 'X' }",
                27,
                ContextKind::Workflow("w".into()),
                RegionKind::Workflow,
                Span::new(13, 25),
                Span::new(13, 21),
                Span::new(22, 25),
            ),
            (
                "begin { Add-Type 'X' }",
                22,
                ContextKind::KeywordBlock("begin".into()),
                RegionKind::KeywordBlock,
                Span::new(8, 20),
                Span::new(8, 16),
                Span::new(17, 20),
            ),
            (
                "trap { Add-Type 'X' }",
                21,
                ContextKind::KeywordBlock("trap".into()),
                RegionKind::KeywordBlock,
                Span::new(7, 19),
                Span::new(7, 15),
                Span::new(16, 19),
            ),
        ];
        for (source, end, kind, region, command_span, name_span, argument_span) in cases {
            let context = Context {
                path: vec![
                    ContextFrame {
                        kind: ContextKind::Script,
                        span: Span::new(0, end),
                        status: ContextStatus::DiscoveryOnly,
                    },
                    ContextFrame {
                        kind,
                        span: Span::new(0, end),
                        status: ContextStatus::Deferred,
                    },
                ],
            };
            let reason = DiscoveryReason::UnsupportedRegion(region);
            let expected = CoreReport {
                unresolved: vec![unresolved(
                    command_span,
                    name_span,
                    argument_span,
                    reason,
                    context.clone(),
                )],
                coverage: vec![CoverageGap {
                    span: Span::new(0, end),
                    reason,
                    context,
                }],
                ..CoreReport::default()
            };

            let actual = analyze_add_type(source);

            assert_eq!(actual, Ok(expected), "{source:?}");
        }
    }

    #[test]
    fn analyze_pathological_nesting_returns_guarded_syntax_error_without_discovery() {
        let source = format!("{}1{}", "(".repeat(512), ")".repeat(512));
        // Match the parser's depth-guard tests: its bounded recursive descent
        // requires an explicit default-sized 8 MiB stack, not the harness stack.
        let thread = std::thread::Builder::new().stack_size(8 << 20);

        let actual = thread
            .spawn(move || analyze_add_type(&source))
            .expect("test worker can be spawned")
            .join()
            .expect("parser guard must return without overflowing");

        assert!(
            matches!(
                &actual,
                Err(AnalysisError::Syntax { lex_errors, parse_errors })
                    if lex_errors.is_empty()
                        && parse_errors.iter().any(|error| error.message == "expression nested too deeply")
            ),
            "expected parser-guard Syntax failure, got {actual:?}"
        );
    }

    #[test]
    fn analyze_literal_scriptblock_preserves_deferred_context() {
        let source = "{ Add-Type 'X' }";
        let context = Context {
            path: vec![
                ContextFrame {
                    kind: ContextKind::Script,
                    span: Span::new(0, 16),
                    status: ContextStatus::DiscoveryOnly,
                },
                ContextFrame {
                    kind: ContextKind::ScriptBlock,
                    span: Span::new(0, 16),
                    status: ContextStatus::Deferred,
                },
            ],
        };
        let expected = CoreReport {
            unresolved: vec![unresolved(
                Span::new(2, 14),
                Span::new(2, 10),
                Span::new(11, 14),
                DiscoveryReason::DeferredContext,
                context.clone(),
            )],
            coverage: vec![CoverageGap {
                span: Span::new(0, 16),
                reason: DiscoveryReason::DeferredContext,
                context,
            }],
            ..CoreReport::default()
        };

        let actual = analyze_add_type(source);

        assert_eq!(actual, Ok(expected));
    }

    // Regression tests for fallback when supported controls contain unsupported ones.
    transfer_case!(
        analyze_closure_if_switch_file_keeps_whole_fallback_and_discovers_input_once,
        "$a='X'; if ($p) { switch -File (Add-Type 'I') { $b='Y' } } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(80, positional((32, 44), (32, 40), (41, 44))),
                unknown(
                    80,
                    positional((69, 80), (69, 77), (78, 80)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(80, (8, 67), RegionKind::ControlFlow),
                gap(80, (18, 56), RegionKind::ControlFlow),
                gap(80, (31, 45), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_closure_loop_filtered_try_keeps_whole_fallback_without_invented_header_site,
        "$a='X'; while ($p) { try { $b='Y' } catch (Add-Type 'H') { Add-Type 'C' } }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(88, positional((59, 71), (59, 67), (68, 71))),
                unknown(
                    88,
                    positional((77, 88), (77, 85), (86, 88)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(88, (8, 75), RegionKind::ControlFlow),
                gap(88, (21, 73), RegionKind::ControlFlow),
                gap(88, (36, 73), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_closure_try_labeled_loop_keeps_whole_fallback_and_discovers_body_once,
        "$a='X'; try { :outer while ($p) { Add-Type 'L' } } finally { $b='Y' }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(82, positional((34, 46), (34, 42), (43, 46))),
                unknown(
                    82,
                    positional((71, 82), (71, 79), (80, 82)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(82, (8, 69), RegionKind::ControlFlow),
                gap(82, (14, 48), RegionKind::ControlFlow),
                gap(82, (21, 48), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_closure_if_pipeline_chain_keeps_whole_fallback_without_duplicate_site,
        "$a='X'; if ($p) { Mystery && Add-Type 'P' } else { }; Add-Type $a",
        CoreReport {
            unresolved: vec![
                conditional_site(65, positional((29, 41), (29, 37), (38, 41))),
                unknown(
                    65,
                    positional((54, 65), (54, 62), (63, 65)),
                    ValueReason::VariableNotKnown,
                ),
            ],
            coverage: vec![
                gap(65, (8, 52), RegionKind::ControlFlow),
                gap(65, (18, 41), RegionKind::ControlFlow),
                gap(65, (18, 25), RegionKind::OtherCommand),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_closure_try_second_filtered_catch_dominates_all_unrelated_write_keys,
        "$a='X'; try { $b='Y' } catch { } catch [Exception] { } finally { $c='Z' }; Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                86,
                positional((75, 86), (75, 83), (84, 86)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(86, (8, 73), RegionKind::ControlFlow),
                gap(86, (23, 32), RegionKind::ControlFlow),
                gap(86, (33, 54), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        }
    );
    transfer_case!(
        analyze_closure_coalescing_left_equal_write_invalidates_key_without_value_transfer,
        "$a='X'; ($a='X') ?? ($b='Y'); Add-Type $a",
        CoreReport {
            unresolved: vec![unknown(
                41,
                positional((30, 41), (30, 38), (39, 41)),
                ValueReason::VariableNotKnown,
            )],
            coverage: vec![
                gap(41, (8, 28), RegionKind::Expression),
                gap(41, (8, 16), RegionKind::Expression),
                gap(41, (20, 28), RegionKind::Expression),
            ],
            ..CoreReport::default()
        }
    );

    // Both modes check the full report against the same independently specified
    // baseline, with expected hypotheses added only in the enabled case.
    macro_rules! hypothesis_matrix_case {
        ($off:ident, $on:ident, $source:expr, $ordinary:expr, $hypotheses:expr) => {
            #[test]
            fn $off() {
                let source = $source;
                let expected = $ordinary;

                let actual = analyze_add_type(source);

                assert_eq!(actual, Ok(expected));
            }

            #[test]
            fn $on() {
                let source = $source;
                let expected = CoreReport {
                    hypotheses: $hypotheses,
                    ..$ordinary
                };

                let actual = super::analyze_add_type(source, hypothesis_options());

                assert_eq!(actual, Ok(expected));
            }
        };
    }

    hypothesis_matrix_case!(
        analyze_off_conditional_loop_keeps_header_and_body_discovery_only,
        analyze_on_conditional_loop_keeps_header_and_body_discovery_only,
        "if ($p) { while (Add-Type 'H') { Add-Type 'B' } }",
        CoreReport {
            unresolved: vec![
                conditional_site(49, positional((17, 29), (17, 25), (26, 29))),
                conditional_site(49, positional((33, 45), (33, 41), (42, 45))),
            ],
            coverage: vec![
                gap(49, (0, 49), RegionKind::ControlFlow),
                gap(49, (10, 47), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_conditional_try_discovers_each_clause_once,
        analyze_on_conditional_try_never_admits_clause_literals,
        "if ($p) { try { Add-Type 'T' } catch { Add-Type 'C' } finally { Add-Type 'F' } }",
        CoreReport {
            unresolved: vec![
                conditional_site(80, positional((16, 28), (16, 24), (25, 28))),
                conditional_site(80, positional((39, 51), (39, 47), (48, 51))),
                conditional_site(80, positional((64, 76), (64, 72), (73, 76))),
            ],
            coverage: vec![
                gap(80, (0, 80), RegionKind::ControlFlow),
                gap(80, (10, 78), RegionKind::ControlFlow),
                gap(80, (31, 53), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    fn conditional_deferred_bodies_report() -> CoreReport {
        let block = nested_context(script_context(81), ContextKind::ScriptBlock, (10, 38));
        let function = nested_context(
            script_context(81),
            ContextKind::Function("f".into()),
            (40, 79),
        );
        CoreReport {
            unresolved: vec![
                Unresolved {
                    context: block.clone(),
                    ..conditional_site(81, positional((22, 34), (22, 30), (31, 34)))
                },
                Unresolved {
                    context: function.clone(),
                    ..conditional_site(81, positional((63, 75), (63, 71), (72, 75)))
                },
            ],
            coverage: vec![
                gap(81, (0, 81), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(10, 38),
                    reason: DiscoveryReason::DeferredContext,
                    context: block.clone(),
                },
                CoverageGap {
                    context: block,
                    ..gap(81, (12, 36), RegionKind::ControlFlow)
                },
                CoverageGap {
                    span: Span::new(40, 79),
                    reason: DiscoveryReason::PendingFunctionAnalysis,
                    context: function.clone(),
                },
                CoverageGap {
                    context: function,
                    ..gap(81, (53, 77), RegionKind::ControlFlow)
                },
            ],
            ..CoreReport::default()
        }
    }

    hypothesis_matrix_case!(
        analyze_off_conditional_deferred_bodies_retain_lexical_contexts,
        analyze_on_conditional_deferred_bodies_never_enter_nested_if_transfers,
        "if ($p) { { if ($q) { Add-Type 'B' } }; function f { if ($q) { Add-Type 'F' } } }",
        conditional_deferred_bodies_report(),
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_flow_keeps_arm_and_following_if_tail_unresolved,
        analyze_on_flow_never_admits_arm_or_following_if_tail,
        "if ($p) { return; Add-Type 'T' }; if ($q) { Add-Type 'S' }",
        CoreReport {
            unresolved: vec![
                conditional_site(58, positional((18, 30), (18, 26), (27, 30))),
                conditional_site(58, positional((44, 56), (44, 52), (53, 56))),
            ],
            coverage: vec![
                gap(58, (0, 32), RegionKind::ControlFlow),
                gap(58, (10, 16), RegionKind::ControlFlow),
                gap(58, (34, 58), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_conditional_pipeline_and_coalescing_rhs_keep_discovery,
        analyze_on_conditional_pipeline_and_coalescing_rhs_neither_admit_nor_restore_values,
        "$a='X'; if ($p) { Add-Type 'P' | Out-Null; Add-Type $a; $b??=Add-Type 'N' }",
        CoreReport {
            unresolved: vec![
                conditional_site(75, positional((18, 30), (18, 26), (27, 30))),
                conditional_site(75, positional((43, 54), (43, 51), (52, 54))),
                conditional_site(75, positional((61, 73), (61, 69), (70, 73))),
            ],
            coverage: vec![
                gap(75, (8, 75), RegionKind::ControlFlow),
                gap(75, (18, 41), RegionKind::ControlFlow),
                gap(75, (33, 41), RegionKind::OtherCommand),
                gap(75, (56, 73), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_conditional_command_argument_subexpression_keeps_discovery,
        analyze_on_conditional_command_argument_subexpression_leaves_no_stale_candidate,
        "$a='X'; if ($p) { Write-Output $(Add-Type 'I'); Add-Type $a }",
        CoreReport {
            unresolved: vec![
                conditional_site(61, positional((33, 45), (33, 41), (42, 45))),
                conditional_site(61, positional((48, 59), (48, 56), (57, 59))),
            ],
            coverage: vec![
                gap(61, (8, 61), RegionKind::ControlFlow),
                gap(61, (18, 46), RegionKind::OtherCommand),
                gap(61, (31, 46), RegionKind::Expression),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_equal_join_leaves_later_conditional_site_unresolved,
        analyze_on_equal_join_keeps_all_ordered_origins_in_one_candidate,
        "if ($p) { $a='X' } else { $A='X' }; if ($q) { Add-Type $a }",
        CoreReport {
            unresolved: vec![conditional_site(
                59,
                positional((46, 57), (46, 54), (55, 57))
            )],
            coverage: vec![
                gap(59, (0, 34), RegionKind::ControlFlow),
                gap(59, (36, 59), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![HypothesisConditionalSite {
            site: positional((46, 57), (46, 54), (55, 57)),
            reason: HypothesisReason::ConditionalSite,
            raw_body: "X".into(),
            literal_origins: vec![literal((13, 16), (14, 15)), literal((29, 32), (30, 31))],
            assignment_origins: vec![
                assignment((10, 16), (10, 12), (13, 16), "$a"),
                assignment((26, 32), (26, 28), (29, 32), "$A"),
            ],
            reference_origins: vec![reference((55, 57), "$a")],
            context: script_context(59),
        }]
    );

    hypothesis_matrix_case!(
        analyze_off_if_condition_and_nested_arm_discover_all_sites_once,
        analyze_on_if_condition_has_one_candidate_and_clears_before_nested_arm,
        "$a='X'; if (Add-Type $a) { if ($p) { Add-Type $a; Add-Type 'B' } }",
        CoreReport {
            unresolved: vec![
                conditional_site(66, positional((12, 23), (12, 20), (21, 23))),
                conditional_site(66, positional((37, 48), (37, 45), (46, 48))),
                conditional_site(66, positional((50, 62), (50, 58), (59, 62))),
            ],
            coverage: vec![
                gap(66, (8, 66), RegionKind::ControlFlow),
                gap(66, (27, 64), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![
            HypothesisConditionalSite {
                site: positional((12, 23), (12, 20), (21, 23)),
                reason: HypothesisReason::ConditionalSite,
                raw_body: "X".into(),
                literal_origins: vec![literal((3, 6), (4, 5))],
                assignment_origins: vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                reference_origins: vec![reference((21, 23), "$a")],
                context: script_context(66),
            },
            literal_hypothesis(
                positional((50, 62), (50, 58), (59, 62)),
                "B",
                literal((59, 62), (60, 61)),
                script_context(66),
            ),
        ]
    );

    const ADD_TYPE_IF_CONDITION: &str =
        "if ($p) { Add-Type 'A' } elseif (Add-Type 'B') { Add-Type 'C' }";

    fn add_type_if_condition_report() -> CoreReport {
        CoreReport {
            unresolved: vec![
                conditional_site(63, positional((10, 22), (10, 18), (19, 22))),
                conditional_site(63, positional((33, 45), (33, 41), (42, 45))),
                conditional_site(63, positional((49, 61), (49, 57), (58, 61))),
            ],
            coverage: vec![gap(63, (0, 63), RegionKind::ControlFlow)],
            ..CoreReport::default()
        }
    }

    hypothesis_matrix_case!(
        analyze_off_elseif_command_and_arms_are_ordinary_unresolved_once,
        analyze_on_elseif_command_is_admitted_between_arm_sites,
        ADD_TYPE_IF_CONDITION,
        add_type_if_condition_report(),
        vec![
            literal_hypothesis(
                positional((10, 22), (10, 18), (19, 22)),
                "A",
                literal((19, 22), (20, 21)),
                script_context(63),
            ),
            literal_hypothesis(
                positional((33, 45), (33, 41), (42, 45)),
                "B",
                literal((42, 45), (43, 44)),
                script_context(63),
            ),
            literal_hypothesis(
                positional((49, 61), (49, 57), (58, 61)),
                "C",
                literal((58, 61), (59, 60)),
                script_context(63),
            ),
        ]
    );

    analyze_hypothesis_case!(
        analyze_count_one_stops_at_elseif_condition_after_first_arm,
        ADD_TYPE_IF_CONDITION,
        HypothesisLimits {
            candidates: 1,
            ..AnalysisOptions::default().hypothesis_limits
        },
        CoreReport {
            hypotheses: vec![literal_hypothesis(
                positional((10, 22), (10, 18), (19, 22)),
                "A",
                literal((19, 22), (20, 21)),
                script_context(63),
            )],
            coverage: vec![
                gap(63, (0, 63), RegionKind::ControlFlow),
                hypothesis_truncation(
                    (33, 45),
                    HypothesisLimit::CandidateCount,
                    script_context(63)
                ),
            ],
            ..add_type_if_condition_report()
        }
    );

    hypothesis_matrix_case!(
        analyze_off_conditional_rhs_calls_stay_unresolved_and_ordinary_literal_confirms,
        analyze_on_conditional_rhs_uses_old_copy_once_and_command_clear_kills_later_reads,
        "$a='X'; if ($p) { $b=$A; $b=Add-Type -TypeDefinition ($b); Add-Type $a; Add-Type $b }; Add-Type 'S'",
        CoreReport {
            confirmed: vec![known(
                99,
                positional((87, 99), (87, 95), (96, 99)),
                "S",
                literal((96, 99), (97, 98)),
                vec![],
                vec![],
            )],
            unresolved: vec![
                conditional_site(
                    99,
                    AddTypeSite {
                        parameter_span: Some(Span::new(37, 57)),
                        ..positional((28, 57), (28, 36), (53, 57))
                    },
                ),
                conditional_site(99, positional((59, 70), (59, 67), (68, 70))),
                conditional_site(99, positional((72, 83), (72, 80), (81, 83))),
            ],
            coverage: vec![gap(99, (8, 85), RegionKind::ControlFlow)],
            ..CoreReport::default()
        },
        vec![HypothesisConditionalSite {
            site: AddTypeSite {
                parameter_span: Some(Span::new(37, 57)),
                ..positional((28, 57), (28, 36), (53, 57))
            },
            reason: HypothesisReason::ConditionalSite,
            raw_body: "X".into(),
            literal_origins: vec![literal((3, 6), (4, 5))],
            assignment_origins: vec![
                assignment((0, 6), (0, 2), (3, 6), "$a"),
                assignment((18, 23), (18, 20), (21, 23), "$b"),
            ],
            reference_origins: vec![reference((21, 23), "$A"), reference((54, 56), "$b")],
            context: script_context(99),
        }]
    );

    const ISOLATED_FILTER_SOURCE: &str =
        "$a='S'; filter f($p='D', $q=$(Add-Type 'D')) { if ($x) { Add-Type $a; Add-Type $p; $a='L'; Add-Type $A } }; Add-Type $a";

    fn isolated_filter_context() -> Context {
        nested_context(
            script_context(119),
            ContextKind::Function("f".into()),
            (8, 106),
        )
    }

    fn isolated_filter_ordinary_report() -> CoreReport {
        let context = isolated_filter_context();
        let parameter_reason = DiscoveryReason::UnsupportedRegion(RegionKind::ParameterDeclaration);
        let sites = [
            ((57, 68), (57, 65), (66, 68)),
            ((70, 81), (70, 78), (79, 81)),
            ((91, 102), (91, 99), (100, 102)),
        ];
        let mut unresolved = vec![Unresolved {
            site: positional((30, 42), (30, 38), (39, 42)),
            reason: parameter_reason,
            context: context.clone(),
        }];
        unresolved.extend(sites.map(|(command, name, arg)| Unresolved {
            context: context.clone(),
            ..conditional_site(119, positional(command, name, arg))
        }));
        CoreReport {
            confirmed: vec![known(
                119,
                positional((108, 119), (108, 116), (117, 119)),
                "S",
                literal((3, 6), (4, 5)),
                vec![assignment((0, 6), (0, 2), (3, 6), "$a")],
                vec![reference((117, 119), "$a")],
            )],
            unresolved,
            coverage: vec![
                CoverageGap {
                    span: Span::new(17, 23),
                    reason: parameter_reason,
                    context: context.clone(),
                },
                CoverageGap {
                    span: Span::new(25, 43),
                    reason: parameter_reason,
                    context: context.clone(),
                },
                CoverageGap {
                    context: context.clone(),
                    ..gap(119, (28, 43), RegionKind::Expression)
                },
                CoverageGap {
                    context,
                    ..gap(119, (47, 104), RegionKind::ControlFlow)
                },
            ],
            ..CoreReport::default()
        }
    }

    #[test]
    fn analyze_off_isolated_filter_keeps_defaults_discovery_only_and_parent_value_known() {
        let expected = isolated_filter_ordinary_report();

        let actual = analyze_add_type(ISOLATED_FILTER_SOURCE);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_on_isolated_filter_owns_local_provenance_after_source_drop_without_captures() {
        let expected = CoreReport {
            hypotheses: vec![HypothesisConditionalSite {
                site: positional((91, 102), (91, 99), (100, 102)),
                reason: HypothesisReason::ConditionalSite,
                raw_body: "L".into(),
                literal_origins: vec![literal((86, 89), (87, 88))],
                assignment_origins: vec![assignment((83, 89), (83, 85), (86, 89), "$a")],
                reference_origins: vec![reference((100, 102), "$A")],
                context: isolated_filter_context(),
            }],
            ..isolated_filter_ordinary_report()
        };

        let actual = {
            let source = String::from(ISOLATED_FILTER_SOURCE);
            super::analyze_add_type(&source, hypothesis_options())
        };

        assert_eq!(actual, Ok(expected));
    }

    fn nested_function_hypothesis_report() -> CoreReport {
        let outer = nested_context(
            script_context(100),
            ContextKind::Function("f".into()),
            (0, 100),
        );
        let inner = nested_context(outer.clone(), ContextKind::Function("g".into()), (21, 73));
        CoreReport {
            unresolved: vec![
                Unresolved {
                    context: inner.clone(),
                    ..conditional_site(100, positional((44, 55), (44, 52), (53, 55)))
                },
                Unresolved {
                    context: inner.clone(),
                    ..conditional_site(100, positional((57, 69), (57, 65), (66, 69)))
                },
                Unresolved {
                    context: outer.clone(),
                    ..conditional_site(100, positional((85, 96), (85, 93), (94, 96)))
                },
            ],
            coverage: vec![
                CoverageGap {
                    context: inner,
                    ..gap(100, (34, 71), RegionKind::ControlFlow)
                },
                CoverageGap {
                    context: outer,
                    ..gap(100, (75, 98), RegionKind::ControlFlow)
                },
            ],
            ..CoreReport::default()
        }
    }

    hypothesis_matrix_case!(
        analyze_off_nested_function_keeps_both_isolated_contexts_unresolved,
        analyze_on_nested_function_admits_local_literal_and_preserves_outer_environment,
        "function f { $a='L'; function g { if ($p) { Add-Type $a; Add-Type 'G' } }; if ($q) { Add-Type $a } }",
        nested_function_hypothesis_report(),
        vec![
            literal_hypothesis(
                positional((57, 69), (57, 65), (66, 69)),
                "G",
                literal((66, 69), (67, 68)),
                nested_context(
                    nested_context(script_context(100), ContextKind::Function("f".into()), (0, 100)),
                    ContextKind::Function("g".into()),
                    (21, 73)
                ),
            ),
            HypothesisConditionalSite {
                site: positional((85, 96), (85, 93), (94, 96)),
                reason: HypothesisReason::ConditionalSite,
                raw_body: "L".into(),
                literal_origins: vec![literal((16, 19), (17, 18))],
                assignment_origins: vec![assignment((13, 19), (13, 15), (16, 19), "$a")],
                reference_origins: vec![reference((94, 96), "$a")],
                context: nested_context(
                    script_context(100),
                    ContextKind::Function("f".into()),
                    (0, 100)
                ),
            },
        ]
    );

    hypothesis_matrix_case!(
        analyze_off_missing_if_outcome_leaves_later_site_unresolved,
        analyze_on_missing_if_outcome_does_not_resurrect_known_assignment,
        "if ($p) { $a='X' }; if ($q) { Add-Type $a }",
        CoreReport {
            unresolved: vec![conditional_site(
                43,
                positional((30, 41), (30, 38), (39, 41))
            )],
            coverage: vec![
                gap(43, (0, 18), RegionKind::ControlFlow),
                gap(43, (20, 43), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_unknown_if_outcome_leaves_later_site_unresolved,
        analyze_on_unknown_if_outcome_does_not_keep_other_arms_known_value,
        "if ($p) { $a='X' } else { $a=$missing }; if ($q) { Add-Type $a }",
        CoreReport {
            unresolved: vec![conditional_site(
                64,
                positional((51, 62), (51, 59), (60, 62))
            )],
            coverage: vec![
                gap(64, (0, 39), RegionKind::ControlFlow),
                gap(64, (41, 64), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_interpolation_and_ternary_keep_fragment_and_expression_gaps,
        analyze_on_interpolation_and_ternary_cannot_yield_visible_literals_as_hypotheses,
        "if ($p) { Add-Type \"$(Add-Type 'I')\"; Add-Type ($q ? 'L' : 'R') }",
        CoreReport {
            unresolved: vec![
                conditional_site(65, positional((10, 36), (10, 18), (19, 36))),
                conditional_site(65, positional((22, 34), (22, 30), (31, 34))),
                conditional_site(65, positional((38, 63), (38, 46), (47, 63))),
            ],
            coverage: vec![
                gap(65, (0, 65), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(19, 36),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(65),
                },
                gap(65, (20, 35), RegionKind::Expression),
                gap(65, (47, 63), RegionKind::Expression),
                gap(65, (48, 62), RegionKind::ControlFlow),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    hypothesis_matrix_case!(
        analyze_off_malformed_fragment_keeps_fragment_gaps_and_ineligible_scope_read,
        analyze_on_malformed_fragment_keeps_gaps_without_candidate_or_syntax_error,
        "\u{feff}if ($p) { Add-Type \"$(+])\"; Add-Type $script:a }",
        CoreReport {
            unresolved: vec![
                conditional_site(51, positional((13, 29), (13, 21), (22, 29))),
                conditional_site(51, positional((31, 49), (31, 39), (40, 49))),
            ],
            coverage: vec![
                gap(51, (3, 51), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(22, 29),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(51),
                },
                gap(51, (23, 28), RegionKind::Expression),
                gap(51, (25, 27), RegionKind::Expression),
                gap(51, (26, 27), RegionKind::FragmentError),
                gap(51, (40, 49), RegionKind::VariableRead),
            ],
            ..CoreReport::default()
        },
        vec![]
    );

    fn strict_conditional_sites_report() -> CoreReport {
        let invalid_sites = [
            ((87, 95), (87, 95), None),
            ((97, 123), (97, 105), Some(Span::new(106, 123))),
            ((125, 153), (125, 133), Some(Span::new(134, 153))),
        ];
        let mut unresolved = vec![conditional_site(
            167,
            positional((67, 85), (67, 75), (76, 79)),
        )];
        unresolved.extend(invalid_sites.map(|(command, name, parameter_span)| {
            conditional_site(
                167,
                AddTypeSite {
                    command_span: Span::new(command.0, command.1),
                    name_span: Span::new(name.0, name.1),
                    argument_span: None,
                    parameter_span,
                },
            )
        }));
        unresolved.push(conditional_site(
            167,
            positional((155, 165), (155, 163), (164, 165)),
        ));
        CoreReport {
            unresolved,
            coverage: vec![
                gap(167, (0, 167), RegionKind::ControlFlow),
                gap(167, (10, 24), RegionKind::Invocation),
                gap(167, (26, 40), RegionKind::Invocation),
                gap(167, (54, 65), RegionKind::OtherCommand),
                gap(167, (67, 85), RegionKind::Redirection),
                gap(167, (80, 85), RegionKind::Redirection),
            ],
            ..CoreReport::default()
        }
    }

    hypothesis_matrix_case!(
        analyze_off_strict_conditional_names_and_arguments_keep_literal_discovery_spans,
        analyze_on_strict_conditional_names_and_arguments_reject_invocation_and_malformed_sites,
        "if ($p) { & Add-Type 'X'; . Add-Type 'X'; 'Add-Type'; Add-Typ 'X'; Add-Type 'R' > out; Add-Type; Add-Type -MemberDefinition; Add-Type -TypeDefinition:'X'; Add-Type 7 }",
        strict_conditional_sites_report(),
        vec![]
    );

    fn raw_quote_sites_report() -> CoreReport {
        let named = AddTypeSite {
            parameter_span: Some(Span::new(22, 55)),
            ..positional((13, 55), (13, 21), (40, 55))
        };
        CoreReport {
            unresolved: vec![
                conditional_site(132, named),
                conditional_site(132, positional((56, 80), (56, 64), (65, 80))),
                conditional_site(132, positional((81, 98), (81, 89), (90, 98))),
                conditional_site(132, positional((99, 114), (99, 107), (108, 114))),
                conditional_site(132, positional((116, 130), (116, 124), (125, 130))),
            ],
            coverage: vec![
                gap(132, (3, 132), RegionKind::ControlFlow),
                CoverageGap {
                    span: Span::new(65, 80),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(132),
                },
                CoverageGap {
                    span: Span::new(90, 98),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(132),
                },
                CoverageGap {
                    span: Span::new(125, 130),
                    reason: DiscoveryReason::FragmentDiagnosticsUnavailable,
                    context: script_context(132),
                },
            ],
            ..CoreReport::default()
        }
    }

    hypothesis_matrix_case!(
        analyze_off_raw_quote_sites_keep_bom_utf8_and_fragment_gaps,
        analyze_on_raw_quote_sites_preserve_escapes_doubled_quotes_and_here_body_bytes,
        "\u{feff}if ($p) { aDd-TyPe -mEmBeRdEfInItIoN @'\r\nŻ\r\n$x`\r\n'@\n\
         Add-Type @\"\r\nλ\r\n`$x\r\n\"@\nAdd-Type @\"\n$x\n\"@\nAdd-Type 'a''b'; Add-Type \"`$x\" }",
        raw_quote_sites_report(),
        vec![
            literal_hypothesis(
                AddTypeSite {
                    parameter_span: Some(Span::new(22, 55)),
                    ..positional((13, 55), (13, 21), (40, 55))
                },
                "Ż\r\n$x`",
                literal((40, 55), (44, 51)),
                script_context(132),
            ),
            literal_hypothesis(
                positional((56, 80), (56, 64), (65, 80)),
                "λ\r\n`$x",
                literal((65, 80), (69, 76)),
                script_context(132),
            ),
            literal_hypothesis(
                positional((99, 114), (99, 107), (108, 114)),
                "a''b",
                literal((108, 114), (109, 113)),
                script_context(132),
            ),
            literal_hypothesis(
                positional((116, 130), (116, 124), (125, 130)),
                "`$x",
                literal((125, 130), (126, 129)),
                script_context(132),
            ),
        ]
    );

    macro_rules! hypothesis_syntax_case {
        ($off:ident, $on:ident, $source:literal, $expected:expr) => {
            #[test]
            fn $off() {
                let expected = $expected;

                let actual = analyze_add_type($source);

                assert_eq!(actual, Err(expected));
            }

            #[test]
            fn $on() {
                let expected = $expected;

                let actual = super::analyze_add_type($source, hypothesis_options());

                assert_eq!(actual, Err(expected));
            }
        };
    }

    hypothesis_syntax_case!(
        analyze_off_malformed_tail_preserves_exact_lex_and_parse_error_vectors,
        analyze_on_malformed_tail_preserves_exact_lex_and_parse_error_vectors,
        "\u{feff}if ($p) { Add-Type 'é' }; try { 'λ",
        AnalysisError::Syntax {
            lex_errors: vec![LexError {
                span: Span::new(36, 39),
                message: "unterminated string".into(),
            }],
            parse_errors: vec![ParseError {
                span: Span::new(39, 39),
                message: "expected '}'".into(),
            }],
        }
    );

    hypothesis_syntax_case!(
        analyze_off_incomplete_finally_keeps_same_span_parser_errors_in_order,
        analyze_on_incomplete_finally_keeps_same_span_parser_errors_in_order,
        "\u{feff}if ($p) { Add-Type 'é' }; try { 'λ' } finally",
        AnalysisError::Syntax {
            lex_errors: vec![],
            parse_errors: vec![
                ParseError {
                    span: Span::new(50, 50),
                    message: "expected '{'".into(),
                },
                ParseError {
                    span: Span::new(50, 50),
                    message: "expected '}'".into(),
                },
            ],
        }
    );

    fn default_count_boundary_fixture() -> (String, CoreReport) {
        let source = "if ($p) { Add-Type 'X' }; ".repeat(128)
            + "if ($p) { Add-Type 'Y' }; if ($q) { Add-Type 'Z' }; Add-Type 'S'";
        // The first 128 statements have a 26-byte stride. The overflow
        // and later conditional sites are both still ordinary unresolved.
        let unresolved = (0..130)
            .map(|i| {
                let start = 26 * i;
                conditional_site(
                    3392,
                    positional(
                        (start + 10, start + 22),
                        (start + 10, start + 18),
                        (start + 19, start + 22),
                    ),
                )
            })
            .collect();
        let coverage = (0..130)
            .map(|i| gap(3392, (26 * i, 26 * i + 24), RegionKind::ControlFlow))
            .collect();
        let ordinary = CoreReport {
            confirmed: vec![known(
                3392,
                positional((3380, 3392), (3380, 3388), (3389, 3392)),
                "S",
                literal((3389, 3392), (3390, 3391)),
                vec![],
                vec![],
            )],
            unresolved,
            coverage,
            ..CoreReport::default()
        };
        (source, ordinary)
    }

    #[test]
    fn analyze_default_off_reports_all_sites_without_hypotheses_or_gap_past_default_count() {
        let (source, expected) = default_count_boundary_fixture();

        let actual = analyze_add_type(&source);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_on_default_count_admits_128_and_stops_once_at_129_without_stopping_analysis() {
        let (source, mut expected) = default_count_boundary_fixture();
        expected.hypotheses = (0..128)
            .map(|i| {
                let start = 26 * i;
                literal_hypothesis(
                    positional(
                        (start + 10, start + 22),
                        (start + 10, start + 18),
                        (start + 19, start + 22),
                    ),
                    "X",
                    literal((start + 19, start + 22), (start + 20, start + 21)),
                    script_context(3392),
                )
            })
            .collect();
        expected.coverage.insert(
            129,
            hypothesis_truncation(
                (3338, 3350),
                HypothesisLimit::CandidateCount,
                script_context(3392),
            ),
        );

        let actual = super::analyze_add_type(&source, hypothesis_options());

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_on_zero_text_admits_empty_body_then_stops_before_later_empty_body() {
        let source = "if ($p) { Add-Type ''; Add-Type 'X'; Add-Type '' }";
        let options = AnalysisOptions {
            hypothesis_limits: hypothesis_limits(0, 16),
            ..hypothesis_options()
        };
        let expected = CoreReport {
            unresolved: vec![
                conditional_site(50, positional((10, 21), (10, 18), (19, 21))),
                conditional_site(50, positional((23, 35), (23, 31), (32, 35))),
                conditional_site(50, positional((37, 48), (37, 45), (46, 48))),
            ],
            coverage: vec![
                gap(50, (0, 50), RegionKind::ControlFlow),
                hypothesis_truncation(
                    (23, 35),
                    HypothesisLimit::OwnedTextBytes,
                    script_context(50),
                ),
            ],
            hypotheses: vec![literal_hypothesis(
                positional((10, 21), (10, 18), (19, 21)),
                "",
                literal((19, 21), (20, 20)),
                script_context(50),
            )],
            ..CoreReport::default()
        };

        let actual = super::analyze_add_type(source, options);

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_on_default_text_admits_one_mib_then_reports_the_first_extra_byte() {
        // Eight copies of (131,068 body bytes + "$a" target + "$a" reference)
        // fill 1 MiB exactly, without a 1 MiB source fixture.
        let body = "a".repeat(131_068);
        let source = format!("$a='{body}'; if ($p) {{ Add-Type ($a) }}")
            + &" elseif ($p) { Add-Type ($a) }".repeat(7)
            + " else { Add-Type 'X' }; Add-Type 'S'";
        let arm_site = |i| {
            let start = 131_085 + 30 * i;
            positional(
                (start, start + 13),
                (start, start + 8),
                (start + 9, start + 13),
            )
        };
        let mut unresolved: Vec<_> = (0..8)
            .map(|i| conditional_site(131_346, arm_site(i)))
            .collect();
        unresolved.push(conditional_site(
            131_346,
            positional((131_318, 131_330), (131_318, 131_326), (131_327, 131_330)),
        ));
        let expected = CoreReport {
            unresolved,
            coverage: vec![
                gap(131_346, (131_075, 131_332), RegionKind::ControlFlow),
                hypothesis_truncation(
                    (131_318, 131_330),
                    HypothesisLimit::OwnedTextBytes,
                    script_context(131_346),
                ),
            ],
            hypotheses: (0..8)
                .map(|i| HypothesisConditionalSite {
                    site: arm_site(i),
                    reason: HypothesisReason::ConditionalSite,
                    raw_body: body.clone(),
                    literal_origins: vec![literal((3, 131_073), (4, 131_072))],
                    assignment_origins: vec![assignment((0, 131_073), (0, 2), (3, 131_073), "$a")],
                    reference_origins: vec![reference((131_095 + 30 * i, 131_097 + 30 * i), "$a")],
                    context: script_context(131_346),
                })
                .collect(),
            confirmed: vec![known(
                131_346,
                positional((131_334, 131_346), (131_334, 131_342), (131_343, 131_346)),
                "S",
                literal((131_343, 131_346), (131_344, 131_345)),
                vec![],
                vec![],
            )],
        };

        let actual = super::analyze_add_type(&source, hypothesis_options());

        assert_eq!(actual, Ok(expected));
    }

    #[test]
    fn analyze_on_default_records_charge_each_copy_and_stop_after_exactly_4096() {
        let source = "$a='X'; ".to_owned()
            + &"$a=$a; ".repeat(30)
            + "if ($p) { Add-Type $a }"
            + &" elseif ($p) { Add-Type $a }".repeat(63)
            + " else { Add-Type 'Y' }";
        let mut assignments = vec![assignment((0, 6), (0, 2), (3, 6), "$a")];
        assignments.extend((0..30).map(|i| {
            let start = 8 + 7 * i;
            assignment(
                (start, start + 5),
                (start, start + 2),
                (start + 3, start + 5),
                "$a",
            )
        }));
        let inherited_references: Vec<_> = (0..30)
            .map(|i| reference((11 + 7 * i, 13 + 7 * i), "$a"))
            .collect();
        let arm_site = |i| {
            let start = 228 + 28 * i;
            positional(
                (start, start + 11),
                (start, start + 8),
                (start + 9, start + 11),
            )
        };
        let mut unresolved: Vec<_> = (0..64)
            .map(|i| conditional_site(2027, arm_site(i)))
            .collect();
        unresolved.push(conditional_site(
            2027,
            positional((2013, 2025), (2013, 2021), (2022, 2025)),
        ));
        let expected = CoreReport {
            unresolved,
            coverage: vec![
                gap(2027, (218, 2027), RegionKind::ControlFlow),
                hypothesis_truncation(
                    (2013, 2025),
                    HypothesisLimit::ProvenanceContextRecords,
                    script_context(2027),
                ),
            ],
            // Each of 64 arms owns 1 literal, 31 assignments, 31 references
            // (30 inherited plus its own), and 1 script frame = 64 records.
            hypotheses: (0..64)
                .map(|i| {
                    let mut references = inherited_references.clone();
                    references.push(reference((237 + 28 * i, 239 + 28 * i), "$a"));
                    HypothesisConditionalSite {
                        site: arm_site(i),
                        reason: HypothesisReason::ConditionalSite,
                        raw_body: "X".into(),
                        literal_origins: vec![literal((3, 6), (4, 5))],
                        assignment_origins: assignments.clone(),
                        reference_origins: references,
                        context: script_context(2027),
                    }
                })
                .collect(),
            ..CoreReport::default()
        };

        let actual = super::analyze_add_type(&source, hypothesis_options());

        assert_eq!(actual, Ok(expected));
    }
}
