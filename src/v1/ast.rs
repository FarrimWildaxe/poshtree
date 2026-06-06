//! The PowerShell abstract syntax tree.
//!
//! [`AstNode`] has one variant for each construct the parser produces, covering
//! pipelines, command invocations with their parameters, the expression
//! grammar, literals, sub-expressions, and the statement forms. Every node
//! carries a [`Location`] so callers can map it back to a line and column in
//! the source.

use std::fmt;

/// Source location carried by every AST node.
#[derive(Debug, Clone, Default)]
pub struct Location {
    pub line: u32,
    pub col: u32,
    pub pos: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}:{}", self.line, self.col)
    }
}

// Leaf / helper types

/// A single parameter of a P/Invoke signature.
#[derive(Debug, Clone, Default)]
pub struct CSharpParam {
    pub type_name: String,
    pub name: String,
}

/// A single `[DllImport]` signature extracted from C# code.
#[derive(Debug, Clone, Default)]
pub struct CSharpImport {
    pub dll: String,
    pub function: String,
    pub returns: String,
    pub params: Vec<CSharpParam>,
}

/// The C# body passed to `Add-Type -MemberDefinition` / `-TypeDefinition`.
#[derive(Debug, Clone, Default)]
pub struct CSharpMemberDef {
    pub loc: Location,
    pub code: String,
    pub imports: Vec<CSharpImport>,
    /// Flat set of imported function names for quick matching.
    pub apis: Vec<String>,
    /// Which Add-Type parameter carried it (`memberdefinition` / `typedefinition`).
    pub parameter: String,
}

// Top-level

#[derive(Debug, Clone, Default)]
pub struct ScriptBlock {
    pub loc: Location,
    pub statements: Vec<AstNode>,
    pub param_block: Option<Box<AstNode>>, // ParamBlock
}

#[derive(Debug, Clone, Default)]
pub struct ParamBlock {
    pub loc: Location,
    pub parameters: Vec<AstNode>, // Variable nodes
}

// Statements

#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    pub loc: Location,
    pub elements: Vec<AstNode>,
}

/// A command redirection such as `> out.txt`, `>> log`, or `2>&1`. Stream-merge
/// forms like `2>&1` encode their destination in `operator` and carry no
/// `target`; a file redirection parses its destination into `target`.
#[derive(Debug, Clone, Default)]
pub struct Redirection {
    pub loc: Location,
    /// Full operator text, e.g. `">"`, `">>"`, `"2>"`, `"2>&1"`, `"<"`.
    pub operator: String,
    /// Destination expression (file path, variable, ...), or `None` for a
    /// stream-merge redirection whose destination is encoded in `operator`.
    pub target: Option<Box<AstNode>>,
}

#[derive(Debug, Clone, Default)]
pub struct Command {
    pub loc: Location,
    pub name: String,
    pub name_expr: Option<Box<AstNode>>,
    pub invocation_operator: Option<String>,
    pub elements: Vec<AstNode>, // CommandParameter | expressions
    pub redirections: Vec<Redirection>,
    /// `Add-Type` C# member definition, held as an [`AstNode::CSharpMemberDef`].
    /// This is derived metadata (extracted from the command's string argument),
    /// but it lives in the tree as a real child node, so traversals and AST
    /// transforms reach it like any other node.
    pub csharp: Option<Box<AstNode>>,
}

#[derive(Debug, Clone, Default)]
pub struct CommandParameter {
    pub loc: Location,
    pub name: String,
    pub argument: Option<Box<AstNode>>,
}

#[derive(Debug, Clone, Default)]
pub struct AssignmentStatement {
    pub loc: Location,
    pub target: Option<Box<AstNode>>,
    pub operator: String,
    pub value: Option<Box<AstNode>>,
}

#[derive(Debug, Clone, Default)]
pub struct IfStatement {
    pub loc: Location,
    /// `(condition, body)` pairs: the `if` clause + any `elseif` clauses.
    pub clauses: Vec<(Box<AstNode>, ScriptBlock)>,
    pub else_body: Option<ScriptBlock>,
}

#[derive(Debug, Clone, Default)]
pub struct WhileStatement {
    pub loc: Location,
    pub condition: Option<Box<AstNode>>,
    pub body: Option<ScriptBlock>,
    pub do_while: bool,
    pub until: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ForStatement {
    pub loc: Location,
    pub initializer: Option<Box<AstNode>>,
    pub condition: Option<Box<AstNode>>,
    pub iterator: Option<Box<AstNode>>,
    pub body: Option<ScriptBlock>,
}

#[derive(Debug, Clone, Default)]
pub struct ForEachStatement {
    pub loc: Location,
    pub variable: Option<Box<AstNode>>,
    pub enumerable: Option<Box<AstNode>>,
    pub body: Option<ScriptBlock>,
}

#[derive(Debug, Clone, Default)]
pub struct SwitchStatement {
    pub loc: Location,
    pub condition: Option<Box<AstNode>>,
    pub body: Option<ScriptBlock>,
}

#[derive(Debug, Clone, Default)]
pub struct TryStatement {
    pub loc: Location,
    pub body: Option<ScriptBlock>,
    pub catches: Vec<ScriptBlock>,
    pub finally_body: Option<ScriptBlock>,
}

#[derive(Debug, Clone, Default)]
pub struct FunctionDefinition {
    pub loc: Location,
    pub name: String,
    pub kind: String, // "function" | "filter" | "workflow"
    pub body: Option<ScriptBlock>,
}

#[derive(Debug, Clone, Default)]
pub struct ReturnStatement {
    pub loc: Location,
    pub value: Option<Box<AstNode>>,
}

#[derive(Debug, Clone, Default)]
pub struct ThrowStatement {
    pub loc: Location,
    pub value: Option<Box<AstNode>>,
}

#[derive(Debug, Clone, Default)]
pub struct FlowStatement {
    pub loc: Location,
    pub keyword: String, // "break" | "continue" | "exit"
}

// Expressions

#[derive(Debug, Clone, Default)]
pub struct BinaryExpression {
    pub loc: Location,
    pub left: Box<AstNode>,
    pub operator: String,
    pub right: Box<AstNode>,
}

#[derive(Debug, Clone, Default)]
pub struct UnaryExpression {
    pub loc: Location,
    pub operator: String,
    pub operand: Box<AstNode>,
    pub postfix: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CastExpression {
    pub loc: Location,
    pub type_name: String,
    pub expression: Box<AstNode>,
}

#[derive(Debug, Clone, Default)]
pub struct MemberAccess {
    pub loc: Location,
    pub target: Box<AstNode>,
    pub member: String,
    pub member_expr: Option<Box<AstNode>>,
    pub is_static: bool,
    /// `true` for the PowerShell-7 null-conditional access operator `?.`.
    pub null_conditional: bool,
}

#[derive(Debug, Clone, Default)]
pub struct InvokeMember {
    pub loc: Location,
    pub target: Box<AstNode>,
    pub member: String,
    pub member_expr: Option<Box<AstNode>>,
    pub is_static: bool,
    /// `true` for the PowerShell-7 null-conditional access operator `?.`.
    pub null_conditional: bool,
    pub arguments: Vec<AstNode>,
}

#[derive(Debug, Clone, Default)]
pub struct IndexExpression {
    pub loc: Location,
    pub target: Box<AstNode>,
    pub index: Option<Box<AstNode>>,
    /// `true` for the PowerShell-7 null-conditional index operator `?[`.
    pub null_conditional: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ParenExpression {
    pub loc: Location,
    pub expression: Box<AstNode>,
}

#[derive(Debug, Clone, Default)]
pub struct SubExpression {
    pub loc: Location,
    pub body: ScriptBlock,
}

#[derive(Debug, Clone, Default)]
pub struct ArrayExpression {
    pub loc: Location,
    pub elements: Vec<AstNode>,
}

#[derive(Debug, Clone, Default)]
pub struct ArrayLiteral {
    pub loc: Location,
    pub elements: Vec<AstNode>,
}

#[derive(Debug, Clone, Default)]
pub struct HashtableExpression {
    pub loc: Location,
    pub entries: Vec<(Box<AstNode>, Box<AstNode>)>,
}

#[derive(Debug, Clone, Default)]
pub struct ScriptBlockExpression {
    pub loc: Location,
    pub body: ScriptBlock,
}

// Leaves

#[derive(Debug, Clone, Default)]
pub struct Variable {
    pub loc: Location,
    pub name: String,
    pub scope: Option<String>,
    pub splat: bool,
    pub raw: String,
}

#[derive(Debug, Clone, Default)]
pub struct StringLiteral {
    pub loc: Location,
    pub value: String,
    /// `"single"` | `"double"` | `"here_single"` | `"here_double"`
    pub kind: String,
    pub raw: String,
    pub expandable: bool,
    /// Interpolation nodes ($var, ${…}, $(...)) inside expandable strings.
    pub parts: Vec<AstNode>,
}

#[derive(Debug, Clone, Default)]
pub struct NumberLiteral {
    pub loc: Location,
    pub raw: String,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct TypeExpression {
    pub loc: Location,
    pub name: String,
}

#[derive(Debug, Clone, Default)]
pub struct BareWord {
    pub loc: Location,
    pub value: String,
}

#[derive(Debug, Clone, Default)]
pub struct ErrorNode {
    pub loc: Location,
    pub message: String,
    pub raw: String,
}

// The main enum

/// Ternary conditional `<cond> ? <if-true> : <if-false>` (PowerShell 7).
#[derive(Debug, Clone, Default)]
pub struct TernaryExpression {
    pub loc: Location,
    pub condition: Box<AstNode>,
    pub if_true: Box<AstNode>,
    pub if_false: Box<AstNode>,
}

/// Pipeline-chain `<pipeline> && <pipeline>` / `|| ` (PowerShell 7). These
/// connect *pipelines* (run-on-success / run-on-failure), not boolean
/// expressions, so they live above the pipeline rather than in the expression
/// precedence chain.
#[derive(Debug, Clone, Default)]
pub struct PipelineChain {
    pub loc: Location,
    pub left: Box<AstNode>,
    pub operator: String,
    pub right: Box<AstNode>,
}

/// `using namespace|module|assembly <name>` (PowerShell 5+).
#[derive(Debug, Clone, Default)]
pub struct UsingStatement {
    pub loc: Location,
    pub kind: String, // "namespace" | "module" | "assembly" | "type" | "command" | ""
    pub name: String,
}

/// A `class` declaration with optional base types/interfaces and a list of
/// parsed members (`ClassMember`); unparseable members become `ErrorNode`.
#[derive(Debug, Clone, Default)]
pub struct ClassDefinition {
    pub loc: Location,
    pub name: String,
    pub bases: Vec<String>,
    pub members: Vec<AstNode>,
}

/// A single class member: a property (`[attrs] [type] $name = default`), a
/// method (`[modifiers] [returntype] Name(params) { body }`), or a constructor
/// (`Name(params) { body }`). `member_kind` is `"property"`, `"method"`, or
/// `"constructor"`.
/// A parsed attribute on a class member, such as `[ValidateSet('A','B')]` or
/// `[Parameter(Mandatory = $true)]`. Argument values are kept as their source
/// text rather than parsed into expressions.
#[derive(Debug, Clone, Default)]
pub struct Attribute {
    pub loc: Location,
    /// Attribute name, e.g. `ValidateSet`.
    pub name: String,
    /// Whether the attribute had a parenthesised argument list (so an empty
    /// `[Name()]` round-trips distinctly from a bare `[Name]`).
    pub paren: bool,
    /// Positional argument source text, e.g. `["'A'", "'B'"]`.
    pub positional: Vec<String>,
    /// Named argument `(key, value-source-text)` pairs, e.g. `("Mandatory", "$true")`.
    pub named: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct ClassMember {
    pub loc: Location,
    pub member_kind: String,
    pub name: String,
    /// Property type, or method/constructor return type (`""` if none).
    pub type_name: String,
    /// Parsed attributes, e.g. `[ValidateSet('A','B')]`.
    pub attributes: Vec<Attribute>,
    /// Modifiers, e.g. `["static", "hidden"]`.
    pub modifiers: Vec<String>,
    /// Method/constructor parameters (`Variable`, `CastExpression`, or
    /// `AssignmentStatement` for a defaulted parameter).
    pub parameters: Vec<AstNode>,
    /// Property initializer.
    pub default: Option<Box<AstNode>>,
    /// Method/constructor body.
    pub body: Option<ScriptBlock>,
}

/// An `enum` declaration; `members` are `AssignmentStatement` (`Name = value`)
/// or `BareWord` (`Name`).
#[derive(Debug, Clone, Default)]
pub struct EnumDefinition {
    pub loc: Location,
    pub name: String,
    pub base: String,
    pub members: Vec<AstNode>,
}

/// Unified AST node enum.  Every variant wraps its concrete struct.
#[derive(Debug, Clone)]
pub enum AstNode {
    // top-level / blocks
    ScriptBlock(ScriptBlock),
    ParamBlock(ParamBlock),
    // statements
    Pipeline(Pipeline),
    Command(Command),
    CommandParameter(CommandParameter),
    AssignmentStatement(AssignmentStatement),
    IfStatement(IfStatement),
    WhileStatement(WhileStatement),
    ForStatement(ForStatement),
    ForEachStatement(ForEachStatement),
    SwitchStatement(SwitchStatement),
    TryStatement(TryStatement),
    FunctionDefinition(FunctionDefinition),
    ClassDefinition(ClassDefinition),
    ClassMember(ClassMember),
    EnumDefinition(EnumDefinition),
    UsingStatement(UsingStatement),
    ReturnStatement(ReturnStatement),
    ThrowStatement(ThrowStatement),
    FlowStatement(FlowStatement),
    // expressions
    BinaryExpression(BinaryExpression),
    TernaryExpression(TernaryExpression),
    PipelineChain(PipelineChain),
    UnaryExpression(UnaryExpression),
    CastExpression(CastExpression),
    MemberAccess(MemberAccess),
    InvokeMember(InvokeMember),
    IndexExpression(IndexExpression),
    ParenExpression(ParenExpression),
    SubExpression(SubExpression),
    ArrayExpression(ArrayExpression),
    ArrayLiteral(ArrayLiteral),
    HashtableExpression(HashtableExpression),
    ScriptBlockExpression(ScriptBlockExpression),
    // leaves
    Variable(Variable),
    StringLiteral(StringLiteral),
    NumberLiteral(NumberLiteral),
    TypeExpression(TypeExpression),
    BareWord(BareWord),
    ErrorNode(ErrorNode),
    CSharpMemberDef(CSharpMemberDef),
}

impl Default for AstNode {
    fn default() -> Self {
        AstNode::ErrorNode(ErrorNode::default())
    }
}

impl AstNode {
    /// Source location of this node.
    pub fn loc(&self) -> &Location {
        match self {
            AstNode::ScriptBlock(n) => &n.loc,
            AstNode::ParamBlock(n) => &n.loc,
            AstNode::Pipeline(n) => &n.loc,
            AstNode::Command(n) => &n.loc,
            AstNode::CommandParameter(n) => &n.loc,
            AstNode::AssignmentStatement(n) => &n.loc,
            AstNode::IfStatement(n) => &n.loc,
            AstNode::WhileStatement(n) => &n.loc,
            AstNode::ForStatement(n) => &n.loc,
            AstNode::ForEachStatement(n) => &n.loc,
            AstNode::SwitchStatement(n) => &n.loc,
            AstNode::TryStatement(n) => &n.loc,
            AstNode::FunctionDefinition(n) => &n.loc,
            AstNode::ClassDefinition(n) => &n.loc,
            AstNode::ClassMember(n) => &n.loc,
            AstNode::EnumDefinition(n) => &n.loc,
            AstNode::UsingStatement(n) => &n.loc,
            AstNode::ReturnStatement(n) => &n.loc,
            AstNode::ThrowStatement(n) => &n.loc,
            AstNode::FlowStatement(n) => &n.loc,
            AstNode::BinaryExpression(n) => &n.loc,
            AstNode::TernaryExpression(n) => &n.loc,
            AstNode::PipelineChain(n) => &n.loc,
            AstNode::UnaryExpression(n) => &n.loc,
            AstNode::CastExpression(n) => &n.loc,
            AstNode::MemberAccess(n) => &n.loc,
            AstNode::InvokeMember(n) => &n.loc,
            AstNode::IndexExpression(n) => &n.loc,
            AstNode::ParenExpression(n) => &n.loc,
            AstNode::SubExpression(n) => &n.loc,
            AstNode::ArrayExpression(n) => &n.loc,
            AstNode::ArrayLiteral(n) => &n.loc,
            AstNode::HashtableExpression(n) => &n.loc,
            AstNode::ScriptBlockExpression(n) => &n.loc,
            AstNode::Variable(n) => &n.loc,
            AstNode::StringLiteral(n) => &n.loc,
            AstNode::NumberLiteral(n) => &n.loc,
            AstNode::TypeExpression(n) => &n.loc,
            AstNode::BareWord(n) => &n.loc,
            AstNode::ErrorNode(n) => &n.loc,
            AstNode::CSharpMemberDef(n) => &n.loc,
        }
    }

    /// Visit every node in the subtree rooted at `self`, depth-first.
    pub fn walk<V: FnMut(&AstNode)>(&self, visitor: &mut V) {
        visitor(self);
        for child in self.safe_children() {
            child.walk(visitor);
        }
    }

    /// Children that are safe to iterate (no null placeholders).
    pub fn safe_children(&self) -> Vec<&AstNode> {
        let mut out = Vec::new();
        self.collect_safe_children(&mut out);
        out
    }

    fn collect_safe_children<'a>(&'a self, out: &mut Vec<&'a AstNode>) {
        fn push_opt<'a>(opt: &'a Option<Box<AstNode>>, out: &mut Vec<&'a AstNode>) {
            if let Some(n) = opt {
                out.push(n);
            }
        }
        fn push_sb<'a>(sb: &'a ScriptBlock, out: &mut Vec<&'a AstNode>) {
            for s in &sb.statements {
                out.push(s);
            }
            if let Some(pb) = &sb.param_block {
                out.push(pb);
            }
        }

        match self {
            AstNode::ScriptBlock(n) => {
                for s in &n.statements {
                    out.push(s);
                }
                push_opt(&n.param_block, out);
            }
            AstNode::ParamBlock(n) => {
                for p in &n.parameters {
                    out.push(p);
                }
            }
            AstNode::Pipeline(n) => {
                for e in &n.elements {
                    out.push(e);
                }
            }
            AstNode::Command(n) => {
                push_opt(&n.name_expr, out);
                for e in &n.elements {
                    out.push(e);
                }
                push_opt(&n.csharp, out);
                for r in &n.redirections {
                    push_opt(&r.target, out);
                }
            }
            AstNode::CSharpMemberDef(_) => {}
            AstNode::CommandParameter(n) => {
                push_opt(&n.argument, out);
            }
            AstNode::AssignmentStatement(n) => {
                push_opt(&n.target, out);
                push_opt(&n.value, out);
            }
            AstNode::IfStatement(n) => {
                for (cond, body) in &n.clauses {
                    out.push(cond);
                    push_sb(body, out);
                }
                if let Some(eb) = &n.else_body {
                    push_sb(eb, out);
                }
            }
            AstNode::WhileStatement(n) => {
                push_opt(&n.condition, out);
                if let Some(b) = &n.body {
                    push_sb(b, out);
                }
            }
            AstNode::ForStatement(n) => {
                push_opt(&n.initializer, out);
                push_opt(&n.condition, out);
                push_opt(&n.iterator, out);
                if let Some(b) = &n.body {
                    push_sb(b, out);
                }
            }
            AstNode::ForEachStatement(n) => {
                push_opt(&n.variable, out);
                push_opt(&n.enumerable, out);
                if let Some(b) = &n.body {
                    push_sb(b, out);
                }
            }
            AstNode::SwitchStatement(n) => {
                push_opt(&n.condition, out);
                if let Some(b) = &n.body {
                    push_sb(b, out);
                }
            }
            AstNode::TryStatement(n) => {
                if let Some(b) = &n.body {
                    push_sb(b, out);
                }
                for c in &n.catches {
                    push_sb(c, out);
                }
                if let Some(f) = &n.finally_body {
                    push_sb(f, out);
                }
            }
            AstNode::FunctionDefinition(n) => {
                if let Some(b) = &n.body {
                    push_sb(b, out);
                }
            }
            AstNode::ClassDefinition(n) => {
                for m in &n.members {
                    out.push(m);
                }
            }
            AstNode::ClassMember(n) => {
                for p in &n.parameters {
                    out.push(p);
                }
                if let Some(d) = &n.default {
                    out.push(d);
                }
                if let Some(b) = &n.body {
                    push_sb(b, out);
                }
            }
            AstNode::EnumDefinition(n) => {
                for m in &n.members {
                    out.push(m);
                }
            }
            AstNode::UsingStatement(_) => {}
            AstNode::ReturnStatement(n) => push_opt(&n.value, out),
            AstNode::ThrowStatement(n) => push_opt(&n.value, out),
            AstNode::FlowStatement(_) => {}
            AstNode::BinaryExpression(n) => {
                out.push(&n.left);
                out.push(&n.right);
            }
            AstNode::TernaryExpression(n) => {
                out.push(&n.condition);
                out.push(&n.if_true);
                out.push(&n.if_false);
            }
            AstNode::PipelineChain(n) => {
                out.push(&n.left);
                out.push(&n.right);
            }
            AstNode::UnaryExpression(n) => out.push(&n.operand),
            AstNode::CastExpression(n) => out.push(&n.expression),
            AstNode::MemberAccess(n) => {
                out.push(&n.target);
                push_opt(&n.member_expr, out);
            }
            AstNode::InvokeMember(n) => {
                out.push(&n.target);
                push_opt(&n.member_expr, out);
                for a in &n.arguments {
                    out.push(a);
                }
            }
            AstNode::IndexExpression(n) => {
                out.push(&n.target);
                push_opt(&n.index, out);
            }
            AstNode::ParenExpression(n) => out.push(&n.expression),
            AstNode::SubExpression(n) => push_sb(&n.body, out),
            AstNode::ArrayExpression(n) => {
                for e in &n.elements {
                    out.push(e);
                }
            }
            AstNode::ArrayLiteral(n) => {
                for e in &n.elements {
                    out.push(e);
                }
            }
            AstNode::HashtableExpression(n) => {
                for (k, v) in &n.entries {
                    out.push(k);
                    out.push(v);
                }
            }
            AstNode::ScriptBlockExpression(n) => push_sb(&n.body, out),
            AstNode::StringLiteral(n) => {
                for p in &n.parts {
                    out.push(p);
                }
            }
            AstNode::Variable(_)
            | AstNode::NumberLiteral(_)
            | AstNode::TypeExpression(_)
            | AstNode::BareWord(_)
            | AstNode::ErrorNode(_) => {}
        }
    }

    /// Mutable counterpart of [`safe_children`](Self::safe_children): yields a
    /// mutable borrow of every direct child node, in the **same order** and
    /// covering the **same set** of nodes. Used by AST-rewriting transforms to
    /// splice replacements in place.
    ///
    /// This must stay in lock-step with `collect_safe_children`; the rewrite
    /// pass zips the two together, so any divergence in order or coverage would
    /// misalign replacements.
    pub fn children_mut(&mut self) -> Vec<&mut AstNode> {
        fn push_opt<'a>(opt: &'a mut Option<Box<AstNode>>, out: &mut Vec<&'a mut AstNode>) {
            if let Some(n) = opt.as_deref_mut() {
                out.push(n);
            }
        }
        fn push_sb<'a>(sb: &'a mut ScriptBlock, out: &mut Vec<&'a mut AstNode>) {
            for s in &mut sb.statements {
                out.push(s);
            }
            if let Some(pb) = sb.param_block.as_deref_mut() {
                out.push(pb);
            }
        }

        let mut out: Vec<&mut AstNode> = Vec::new();
        match self {
            AstNode::ScriptBlock(n) => {
                for s in &mut n.statements {
                    out.push(s);
                }
                push_opt(&mut n.param_block, &mut out);
            }
            AstNode::ParamBlock(n) => {
                for p in &mut n.parameters {
                    out.push(p);
                }
            }
            AstNode::Pipeline(n) => {
                for e in &mut n.elements {
                    out.push(e);
                }
            }
            AstNode::Command(n) => {
                push_opt(&mut n.name_expr, &mut out);
                for e in &mut n.elements {
                    out.push(e);
                }
                push_opt(&mut n.csharp, &mut out);
                for r in &mut n.redirections {
                    push_opt(&mut r.target, &mut out);
                }
            }
            AstNode::CSharpMemberDef(_) => {}
            AstNode::CommandParameter(n) => push_opt(&mut n.argument, &mut out),
            AstNode::AssignmentStatement(n) => {
                push_opt(&mut n.target, &mut out);
                push_opt(&mut n.value, &mut out);
            }
            AstNode::IfStatement(n) => {
                for (cond, body) in &mut n.clauses {
                    out.push(cond.as_mut());
                    push_sb(body, &mut out);
                }
                if let Some(eb) = &mut n.else_body {
                    push_sb(eb, &mut out);
                }
            }
            AstNode::WhileStatement(n) => {
                push_opt(&mut n.condition, &mut out);
                if let Some(b) = &mut n.body {
                    push_sb(b, &mut out);
                }
            }
            AstNode::ForStatement(n) => {
                push_opt(&mut n.initializer, &mut out);
                push_opt(&mut n.condition, &mut out);
                push_opt(&mut n.iterator, &mut out);
                if let Some(b) = &mut n.body {
                    push_sb(b, &mut out);
                }
            }
            AstNode::ForEachStatement(n) => {
                push_opt(&mut n.variable, &mut out);
                push_opt(&mut n.enumerable, &mut out);
                if let Some(b) = &mut n.body {
                    push_sb(b, &mut out);
                }
            }
            AstNode::SwitchStatement(n) => {
                push_opt(&mut n.condition, &mut out);
                if let Some(b) = &mut n.body {
                    push_sb(b, &mut out);
                }
            }
            AstNode::TryStatement(n) => {
                if let Some(b) = &mut n.body {
                    push_sb(b, &mut out);
                }
                for c in &mut n.catches {
                    push_sb(c, &mut out);
                }
                if let Some(f) = &mut n.finally_body {
                    push_sb(f, &mut out);
                }
            }
            AstNode::FunctionDefinition(n) => {
                if let Some(b) = &mut n.body {
                    push_sb(b, &mut out);
                }
            }
            AstNode::ClassDefinition(n) => {
                for m in &mut n.members {
                    out.push(m);
                }
            }
            AstNode::ClassMember(n) => {
                for p in &mut n.parameters {
                    out.push(p);
                }
                if let Some(d) = &mut n.default {
                    out.push(d.as_mut());
                }
                if let Some(b) = &mut n.body {
                    push_sb(b, &mut out);
                }
            }
            AstNode::EnumDefinition(n) => {
                for m in &mut n.members {
                    out.push(m);
                }
            }
            AstNode::UsingStatement(_) => {}
            AstNode::ReturnStatement(n) => push_opt(&mut n.value, &mut out),
            AstNode::ThrowStatement(n) => push_opt(&mut n.value, &mut out),
            AstNode::FlowStatement(_) => {}
            AstNode::BinaryExpression(n) => {
                out.push(n.left.as_mut());
                out.push(n.right.as_mut());
            }
            AstNode::TernaryExpression(n) => {
                out.push(n.condition.as_mut());
                out.push(n.if_true.as_mut());
                out.push(n.if_false.as_mut());
            }
            AstNode::PipelineChain(n) => {
                out.push(n.left.as_mut());
                out.push(n.right.as_mut());
            }
            AstNode::UnaryExpression(n) => out.push(n.operand.as_mut()),
            AstNode::CastExpression(n) => out.push(n.expression.as_mut()),
            AstNode::MemberAccess(n) => {
                out.push(n.target.as_mut());
                push_opt(&mut n.member_expr, &mut out);
            }
            AstNode::InvokeMember(n) => {
                out.push(n.target.as_mut());
                push_opt(&mut n.member_expr, &mut out);
                for a in &mut n.arguments {
                    out.push(a);
                }
            }
            AstNode::IndexExpression(n) => {
                out.push(n.target.as_mut());
                push_opt(&mut n.index, &mut out);
            }
            AstNode::ParenExpression(n) => out.push(n.expression.as_mut()),
            AstNode::SubExpression(n) => push_sb(&mut n.body, &mut out),
            AstNode::ArrayExpression(n) => {
                for e in &mut n.elements {
                    out.push(e);
                }
            }
            AstNode::ArrayLiteral(n) => {
                for e in &mut n.elements {
                    out.push(e);
                }
            }
            AstNode::HashtableExpression(n) => {
                for (k, v) in &mut n.entries {
                    out.push(k);
                    out.push(v);
                }
            }
            AstNode::ScriptBlockExpression(n) => push_sb(&mut n.body, &mut out),
            AstNode::StringLiteral(n) => {
                for p in &mut n.parts {
                    out.push(p);
                }
            }
            AstNode::Variable(_)
            | AstNode::NumberLiteral(_)
            | AstNode::TypeExpression(_)
            | AstNode::BareWord(_)
            | AstNode::ErrorNode(_) => {}
        }
        out
    }
}

// Debug introspection (NodeInfo)

/// Debug/introspection metadata for AST nodes, used by the `--dump-ast`
/// renderer ([`dump_ast`](crate::engine::dump_ast)).
///
/// Every node type reports its own [`label`](NodeInfo::label) (a short, stable
/// name) and, for leaf/scalar nodes, a [`scalars`](NodeInfo::scalars) string
/// of its salient fields. Compound nodes keep the default empty `scalars`.
/// [`AstNode`] forwards both methods to whichever variant it holds.
pub trait NodeInfo {
    /// Short, stable node name, e.g. `"Command"` or `"StringLiteral"`.
    fn label(&self) -> &'static str;

    /// Salient scalar fields rendered for debugging (e.g. a command's name, a
    /// string literal's value). Empty for compound nodes.
    fn scalars(&self) -> String {
        String::new()
    }
}

impl NodeInfo for AstNode {
    fn label(&self) -> &'static str {
        match self {
            AstNode::ScriptBlock(_) => "ScriptBlock",
            AstNode::ParamBlock(_) => "ParamBlock",
            AstNode::Pipeline(_) => "Pipeline",
            AstNode::Command(_) => "Command",
            AstNode::CommandParameter(_) => "CommandParameter",
            AstNode::AssignmentStatement(_) => "AssignmentStatement",
            AstNode::IfStatement(_) => "IfStatement",
            AstNode::WhileStatement(_) => "WhileStatement",
            AstNode::ForStatement(_) => "ForStatement",
            AstNode::ForEachStatement(_) => "ForEachStatement",
            AstNode::SwitchStatement(_) => "SwitchStatement",
            AstNode::TryStatement(_) => "TryStatement",
            AstNode::FunctionDefinition(_) => "FunctionDefinition",
            AstNode::ClassDefinition(_) => "ClassDefinition",
            AstNode::ClassMember(_) => "ClassMember",
            AstNode::EnumDefinition(_) => "EnumDefinition",
            AstNode::UsingStatement(_) => "UsingStatement",
            AstNode::ReturnStatement(_) => "ReturnStatement",
            AstNode::ThrowStatement(_) => "ThrowStatement",
            AstNode::FlowStatement(_) => "FlowStatement",
            AstNode::BinaryExpression(_) => "BinaryExpression",
            AstNode::TernaryExpression(_) => "TernaryExpression",
            AstNode::PipelineChain(_) => "PipelineChain",
            AstNode::UnaryExpression(_) => "UnaryExpression",
            AstNode::CastExpression(_) => "CastExpression",
            AstNode::MemberAccess(_) => "MemberAccess",
            AstNode::InvokeMember(_) => "InvokeMember",
            AstNode::IndexExpression(_) => "IndexExpression",
            AstNode::ParenExpression(_) => "ParenExpression",
            AstNode::SubExpression(_) => "SubExpression",
            AstNode::ArrayExpression(_) => "ArrayExpression",
            AstNode::ArrayLiteral(_) => "ArrayLiteral",
            AstNode::HashtableExpression(_) => "HashtableExpression",
            AstNode::ScriptBlockExpression(_) => "ScriptBlockExpression",
            AstNode::Variable(_) => "Variable",
            AstNode::StringLiteral(_) => "StringLiteral",
            AstNode::NumberLiteral(_) => "NumberLiteral",
            AstNode::TypeExpression(_) => "TypeExpression",
            AstNode::BareWord(_) => "BareWord",
            AstNode::ErrorNode(_) => "ErrorNode",
            AstNode::CSharpMemberDef(_) => "CSharpMemberDef",
        }
    }

    fn scalars(&self) -> String {
        match self {
            AstNode::Command(n) => format!("name={:?}", n.name),
            AstNode::CommandParameter(n) => format!("name={:?}", n.name),
            AstNode::Variable(n) => format!("name={:?} raw={:?}", n.name, n.raw),
            AstNode::StringLiteral(n) => {
                let v = crate::textutil::truncate_on_char_boundary(&n.value, 60);
                let v = if v.len() < n.value.len() {
                    format!("{v}...")
                } else {
                    v.to_owned()
                };
                format!("kind={:?} value={:?}", n.kind, v)
            }
            AstNode::BareWord(n) => format!("value={:?}", n.value),
            AstNode::BinaryExpression(n) => format!("operator={:?}", n.operator),
            AstNode::TypeExpression(n) => format!("name={:?}", n.name),
            AstNode::CastExpression(n) => format!("type_name={:?}", n.type_name),
            AstNode::FunctionDefinition(n) => format!("name={:?} kind={:?}", n.name, n.kind),
            AstNode::ClassDefinition(n) => {
                if n.bases.is_empty() {
                    format!("name={:?}", n.name)
                } else {
                    format!("name={:?} bases={:?}", n.name, n.bases)
                }
            }
            AstNode::EnumDefinition(n) => format!("name={:?}", n.name),
            AstNode::ClassMember(n) => {
                let mut s = format!("kind={:?} name={:?}", n.member_kind, n.name);
                if !n.type_name.is_empty() {
                    s.push_str(&format!(" type={:?}", n.type_name));
                }
                if !n.modifiers.is_empty() {
                    s.push_str(&format!(" modifiers={:?}", n.modifiers));
                }
                if !n.attributes.is_empty() {
                    let names: Vec<&str> = n.attributes.iter().map(|a| a.name.as_str()).collect();
                    s.push_str(&format!(" attributes={names:?}"));
                }
                s
            }
            AstNode::UsingStatement(n) => format!("kind={:?} name={:?}", n.kind, n.name),
            AstNode::FlowStatement(n) => format!("keyword={:?}", n.keyword),
            AstNode::ErrorNode(n) => format!("message={:?}", n.message),
            AstNode::PipelineChain(n) => format!("operator={:?}", n.operator),
            AstNode::CSharpMemberDef(n) => {
                format!("imports={}, code={} chars", n.imports.len(), n.code.len())
            }
            _ => String::new(),
        }
    }
}

#[cfg(test)]
mod node_info_tests {
    use super::*;
    use crate::v1::parser::parse;

    fn first_where(root: &AstNode, pred: impl Fn(&AstNode) -> bool) -> Option<String> {
        let mut found = None;
        root.walk(&mut |n| {
            if found.is_none() && pred(n) {
                found = Some(n.scalars());
            }
        });
        found
    }

    #[test]
    fn label_delegates_to_the_variant() {
        let (tree, _) = parse("$x = 'hi'");
        assert_eq!(AstNode::ScriptBlock(tree).label(), "ScriptBlock");
    }

    #[test]
    fn string_literal_reports_kind_and_value() {
        let (tree, _) = parse("'hello world'");
        let s = first_where(&AstNode::ScriptBlock(tree), |n| {
            matches!(n, AstNode::StringLiteral(_))
        })
        .expect("a StringLiteral node");
        assert!(s.contains("value="), "scalars: {s}");
        assert!(s.contains("hello world"), "scalars: {s}");
    }

    #[test]
    fn compound_node_has_empty_scalars() {
        let (tree, _) = parse("if ($true) { 1 }");
        let root = AstNode::ScriptBlock(tree);
        assert_eq!(root.scalars(), "");
        let if_scalars = first_where(&root, |n| matches!(n, AstNode::IfStatement(_)));
        assert_eq!(if_scalars.as_deref(), Some(""));
    }

    #[test]
    fn long_string_value_is_truncated() {
        let long = "A".repeat(200);
        let (tree, _) = parse(&format!("'{long}'"));
        let s = first_where(&AstNode::ScriptBlock(tree), |n| {
            matches!(n, AstNode::StringLiteral(_))
        })
        .expect("a StringLiteral node");
        assert!(s.contains("..."), "expected truncation marker: {s}");
    }

    #[test]
    fn children_mut_mirrors_safe_children() {
        // A script exercising a broad spread of node variants: control flow,
        // try/catch/finally, functions, flow statements, operators, casts,
        // member access / invocation / indexing, sub-expressions, arrays,
        // hashtables, script blocks, expandable strings, the `using` / `class`
        // (with property / constructor / method members) / `enum` statement
        // forms, an `Add-Type` C# member-def child, and the PowerShell-7 ternary
        // and pipeline-chain operators.
        let src = r#"
            using namespace System.Net
            class Downloader : Base {
                [ValidateNotNull()] [string]$Url = "http://x"
                hidden [int]$N = 0
                Downloader([int]$n) { $this.N = $n }
                static [void] Go([string]$u = "d") { Write-Output $u }
            }
            enum E { A; B = 2; C }
            $t = $cond ? 1 : 2
            Get-A && Get-B || Get-C
            Add-Type -TypeDefinition "public class X { }" -Name N -Namespace W
            function Invoke-Thing {
                param($a, $b)
                $h = @{ k = 1; v = ($a + $b) }
                $arr = @(1, 2, 3)
                for ($i = 0; $i -lt $arr.Count; $i++) {
                    if (-not $i) { continue } else { break }
                }
                foreach ($x in $arr) { Write-Output "$x done" }
                while ($true) { throw "stop" }
                switch ($a) { 1 { return [int]$b } }
                try { $obj.Method($arr[0]) } catch { $_.ToString() } finally { $(Get-Date) }
                & { $a -bor $b }
                Write-Output $a > out.txt
                Get-Thing 2>&1
            }
        "#;
        let (tree, _) = parse(src);
        let root = AstNode::ScriptBlock(tree);

        // For every node, `children_mut` must yield the same children in the
        // same order as `safe_children`: `rewrite_tree` maps over
        // `safe_children` and splices the results positionally into the
        // `children_mut` slots, so a divergence in count *or order* would
        // misplace a replacement. Comparing the label sequence catches both.
        fn assert_parity(node: &AstNode) {
            let immutable: Vec<&str> = node.safe_children().iter().map(|c| c.label()).collect();
            let mut owned = node.clone();
            let mutable: Vec<&str> = owned.children_mut().iter().map(|c| c.label()).collect();
            assert_eq!(
                immutable,
                mutable,
                "children_mut/safe_children child mismatch for {}",
                node.label()
            );
            for child in node.safe_children() {
                assert_parity(child);
            }
        }
        assert_parity(&root);
    }
}
