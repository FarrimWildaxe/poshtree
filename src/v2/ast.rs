//! The syntax tree produced by the native [`parse`](super::parser::parse).
//!
//! Unlike the `tree` module, which recovers token ranges from the v1
//! parser, this tree is produced directly from v2 tokens and depends on no
//! `v1` code. Every [`Node`] carries both a byte [`Span`] and a
//! [`TokenRange`], so a node's exact source is `node.range` sliced from the
//! token vector (or `node.span` sliced from the source string).
//!
//! Node labels match v1's `AstNode` names where a construct exists in both,
//! which lets the correlation tree act as a differential-testing oracle
//! during the migration.

use super::span::{Span, TokenRange};

/// Which flavor of string literal a [`NodeKind::StringLiteral`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringKind {
    /// `'literal'`
    Single,
    /// `"expandable $x"`
    Double,
    /// `@' ... '@`
    HereSingle,
    /// `@" ... "@`
    HereDouble,
}

/// A C# parameter in a P/Invoke signature: `int dwFlags` -> type `int`, name
/// `dwFlags`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CSharpParam {
    /// The parameter type.
    pub type_name: String,
    /// The parameter name (empty when none was given).
    pub name: String,
}

/// A `[DllImport]` P/Invoke declaration found in `Add-Type` C# source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CSharpImport {
    /// The imported library name.
    pub dll: String,
    /// The function name.
    pub function: String,
    /// The declared return type.
    pub returns: String,
    /// The function parameters.
    pub params: Vec<CSharpParam>,
}

/// The C# member/type definition handed to `Add-Type`, extracted from the
/// command's string argument. Mirrors v1's `CSharpMemberDef`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CSharpMemberDef {
    /// The raw C# source.
    pub code: String,
    /// Source span of the raw C# body (between the string delimiters), so the
    /// C# can be located in the original file. Drives the C# front-end and
    /// any C#-aware refactoring.
    pub code_span: Span,
    /// `[DllImport]` declarations found in the source.
    pub imports: Vec<CSharpImport>,
    /// Flat list of imported function names, for quick matching.
    pub apis: Vec<String>,
    /// Which `Add-Type` parameter carried the code (`typedefinition`,
    /// `memberdefinition`, or `positional`).
    pub parameter: String,
}

/// A node of the v2 syntax tree: a [`NodeKind`] plus its source location.
#[derive(Debug, Clone)]
pub struct Node {
    /// The node's variant and children.
    pub kind: NodeKind,
    /// Byte range in the source string.
    pub span: Span,
    /// Token range `[first, end)` in the parser's token vector.
    pub range: TokenRange,
}

/// The shape of a [`Node`]. Statement and expression variants share one enum,
/// matching how PowerShell blends the two.
///
/// Marked `#[non_exhaustive]`: future versions may add variants, so downstream
/// matches need a wildcard arm. Within this crate, matches stay exhaustive.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NodeKind {
    // Containers
    /// A script block body: a sequence of statements. The tree root.
    Script(Vec<Node>),
    /// A `param(...)` block: a list of parameter declarations (each an
    /// expression, usually a [`Variable`](NodeKind::Variable)).
    ParamBlock(Vec<Node>),

    // Statements
    /// One or more commands/expressions joined by `|`.
    Pipeline(Vec<Node>),
    /// Two pipelines joined by `&&` or `||`.
    PipelineChain {
        /// Left side.
        left: Box<Node>,
        /// `&&` or `||`.
        op: String,
        /// Right side.
        right: Box<Node>,
    },
    /// A command: a name and its arguments, parameters, and redirections.
    Command {
        /// The command name (a [`BareWord`](NodeKind::BareWord), or an
        /// expression when invoked with `&`/`.`).
        name: Box<Node>,
        /// True when invoked with the call operator `&` or `.`; in that case
        /// the name is an expression and counts as a child.
        invocation: bool,
        /// Arguments and parameters in source order.
        elements: Vec<Node>,
        /// Redirections, kept separate from `elements` as in v1.
        redirections: Vec<Node>,
        /// For `Add-Type`, the extracted C# member definition (a
        /// [`CSharpMemberDef`](NodeKind::CSharpMemberDef) node); `None`
        /// otherwise.
        csharp: Option<Box<Node>>,
    },
    /// Derived `Add-Type` C# metadata, held as a real child of its command.
    CSharpMemberDef(CSharpMemberDef),
    /// A `-Name` style command parameter, with an optional bound argument
    /// (`-Path value`).
    CommandParameter {
        /// The parameter name, without the leading `-`.
        name: String,
        /// A value bound to the parameter, e.g. the `value` in `-Path value`.
        argument: Option<Box<Node>>,
    },
    /// A redirection such as `2>&1` or `> out.txt`.
    Redirection {
        /// The redirection operator text.
        op: String,
        /// The redirection target, if any.
        target: Option<Box<Node>>,
    },
    /// `target op value`, e.g. `$x += 1`.
    Assignment {
        /// The assignment target (lvalue).
        target: Box<Node>,
        /// The assignment operator: `=`, `+=`, `-=`, ...
        op: String,
        /// The assigned pipeline/expression.
        value: Box<Node>,
    },
    /// `if`/`elseif`/`else`. `conditions[i]` guards `blocks[i]`; `else_block`
    /// is the optional trailing `else` body.
    If {
        /// The `if`/`elseif` condition expressions.
        conditions: Vec<Node>,
        /// The bodies guarded by each condition.
        blocks: Vec<Node>,
        /// The optional `else` body.
        else_block: Option<Box<Node>>,
    },
    /// `while (cond) { body }`.
    While {
        /// Loop condition.
        condition: Box<Node>,
        /// Loop body.
        body: Box<Node>,
    },
    /// `do { body } while (cond)` or `do { body } until (cond)`.
    DoWhile {
        /// Loop body.
        body: Box<Node>,
        /// Loop condition.
        condition: Box<Node>,
        /// True for `until`, false for `while`.
        until: bool,
    },
    /// `for (init; cond; update) { body }`.
    For {
        /// Initializer.
        init: Option<Box<Node>>,
        /// Condition.
        condition: Option<Box<Node>>,
        /// Update.
        update: Option<Box<Node>>,
        /// Loop body.
        body: Box<Node>,
    },
    /// `foreach ($v in iterable) { body }`.
    ForEach {
        /// The iteration variable (a [`Variable`](NodeKind::Variable) node).
        variable: Box<Node>,
        /// The collection expression.
        iterable: Box<Node>,
        /// Loop body.
        body: Box<Node>,
    },
    /// `switch (input) { ... }`. Cases are kept as raw child nodes.
    Switch {
        /// Switch options before the input: `-Regex`, `-Wildcard`,
        /// `-CaseSensitive`, `-Exact`, `-File <path>`, in source order. `-File`
        /// keeps its path argument; the others are flags.
        flags: Vec<Node>,
        /// The switch input expression (the `(...)` value, or the `-File`
        /// path when the input comes from a file).
        input: Box<Node>,
        /// Case label expressions and their bodies, in source order.
        cases: Vec<Node>,
    },
    /// `function Name { body }` or `filter Name { body }`.
    Function {
        /// The function name.
        name: String,
        /// Span of the name token alone, not the whole definition. Empty when
        /// the name was missing.
        name_span: Span,
        /// True for `filter`, false for `function`.
        filter: bool,
        /// Parameters from a `function f(...)` list, empty when the function
        /// uses a `param(...)` block inside the body or takes none. Each is a
        /// [`Parameter`](NodeKind::Parameter) node.
        parameters: Vec<Node>,
        /// The function body.
        body: Box<Node>,
    },
    /// `try { body } catch { ... } finally { ... }`.
    Try {
        /// The `try` body.
        body: Box<Node>,
        /// Zero or more `catch` clauses.
        catches: Vec<Node>,
        /// The optional `finally` body.
        finally_block: Option<Box<Node>>,
    },
    /// A `catch [Type] { body }` clause.
    Catch {
        /// The catch body.
        body: Box<Node>,
    },
    /// `using namespace|module|assembly <name>`.
    Using {
        /// `namespace`, `module`, `assembly`, `type`, `command`, or empty.
        kind: String,
        /// The remainder of the line: a dotted name, string, or path.
        name: String,
    },
    /// A `class` declaration with optional base types and a list of members.
    ClassDefinition {
        /// The class name.
        name: String,
        /// Span of the name identifier, for rename tooling.
        name_span: Span,
        /// Base types and interfaces.
        bases: Vec<String>,
        /// Parsed members (properties, methods, constructors).
        members: Vec<Node>,
    },
    /// A single class member: a property, method, or constructor.
    ClassMember {
        /// `"property"`, `"method"`, or `"constructor"`.
        member_kind: String,
        /// The member name.
        name: String,
        /// Method/constructor parameters.
        parameters: Vec<Node>,
        /// Property initializer.
        default: Option<Box<Node>>,
        /// Method/constructor body.
        body: Option<Box<Node>>,
    },
    /// An `enum` declaration; members are `BareWord` or `Assignment`.
    EnumDefinition {
        /// The enum name.
        name: String,
        /// Span of the name identifier, for rename tooling.
        name_span: Span,
        /// The optional backing type.
        base: String,
        /// Enum members.
        members: Vec<Node>,
    },
    /// A control-flow statement: `return`, `throw`, `break`, `continue`,
    /// `exit`, each with an optional value.
    Flow {
        /// The keyword, lowercased.
        keyword: String,
        /// The optional value expression.
        value: Option<Box<Node>>,
    },

    /// A single parameter in a `param(...)` block or function parameter list:
    /// `[Parameter(Mandatory)][int]$Name = $default`. Attribute and type
    /// brackets are kept as `TypeExpression` nodes in `attributes` (the
    /// conventional type is the last one without `(` in its text); `default`
    /// holds the initializer expression when present.
    Parameter {
        /// `[...]` brackets before the name, in source order.
        attributes: Vec<Node>,
        /// The parameter variable name, including the leading `$`.
        name: String,
        /// Span of the name token.
        name_span: Span,
        /// The default value expression, if any.
        default: Option<Box<Node>>,
    },

    /// A labeled statement: `:outer foreach (...) { ... }`. The label is the
    /// target of a `break`/`continue` naming it.
    Labeled {
        /// The label name (without the leading `:`).
        label: String,
        /// Span of the label name, for tooling.
        label_span: Span,
        /// The labeled statement (a loop or switch).
        statement: Box<Node>,
    },

    // Expressions
    /// `condition ? if_true : if_false`.
    Ternary {
        /// The condition.
        condition: Box<Node>,
        /// Value when truthy.
        if_true: Box<Node>,
        /// Value when falsy.
        if_false: Box<Node>,
    },
    /// A binary expression: `left op right`.
    Binary {
        /// The operator text (`+`, `-eq`, `-and`, ...).
        op: String,
        /// Left operand.
        left: Box<Node>,
        /// Right operand.
        right: Box<Node>,
    },
    /// A prefix unary expression: `-x`, `!x`, `-not x`, `++x`.
    Unary {
        /// The operator text.
        op: String,
        /// The operand.
        operand: Box<Node>,
    },
    /// A postfix unary expression: `$x++`, `$x--`.
    PostfixUnary {
        /// The operator text.
        op: String,
        /// The operand.
        operand: Box<Node>,
    },
    /// A cast: `[int]$x`.
    Cast {
        /// The type name inside the brackets.
        type_name: String,
        /// The cast operand.
        operand: Box<Node>,
    },
    /// Member access: `$x.Name` or `[Type]::Member`.
    MemberAccess {
        /// The receiver expression.
        target: Box<Node>,
        /// The member name.
        member: String,
        /// True for `::`, false for `.`.
        is_static: bool,
    },
    /// A method call: `$x.Method(args)` or `[Type]::Method(args)`.
    InvokeMember {
        /// The receiver expression.
        target: Box<Node>,
        /// The method name.
        member: String,
        /// True for `::`, false for `.`.
        is_static: bool,
        /// The call arguments.
        args: Vec<Node>,
    },
    /// Indexing: `$a[expr]`.
    Index {
        /// The indexed expression.
        target: Box<Node>,
        /// The index expression.
        index: Box<Node>,
    },
    /// `( expression )`.
    Paren(Box<Node>),
    /// `$( statements )`.
    SubExpression(Box<Node>),
    /// `{ statements }` used as a value.
    ScriptBlockExpression(Box<Node>),
    /// `@( ... )`.
    Array(Vec<Node>),
    /// A comma list: `a, b, c`.
    ArrayLiteral(Vec<Node>),
    /// `@{ key = value; ... }`.
    Hashtable(Vec<(Node, Node)>),
    /// `$x`, `${a b}`, `$env:PATH`, splatted `@args`. The stored text is the
    /// raw token value, including the leading `$` or `@`.
    Variable(String),
    /// A numeric literal, raw token text.
    Number(String),
    /// A string literal, raw token text (quotes and all).
    StringLiteral {
        /// Which string flavor.
        kind: StringKind,
        /// The raw token value.
        value: String,
        /// Interpolation nodes (`$var`, `${name}`, `$(...)`) inside an
        /// expandable string; empty for single-quoted and here-single strings.
        parts: Vec<Node>,
    },
    /// `[Type]` used as a value rather than a cast.
    TypeExpression(String),
    /// A bareword: command name or argument text.
    BareWord(String),
    /// A node the parser could not build; carries a message.
    Error(String),
}

impl Node {
    /// The v1-compatible label for this node, for debugging and differential
    /// comparison.
    pub fn label(&self) -> &'static str {
        use NodeKind::*;
        match &self.kind {
            Script(_) => "ScriptBlock",
            ParamBlock(_) => "ParamBlock",
            Pipeline(_) => "Pipeline",
            PipelineChain { .. } => "PipelineChain",
            Command { .. } => "Command",
            CSharpMemberDef(_) => "CSharpMemberDef",
            CommandParameter { .. } => "CommandParameter",
            Redirection { .. } => "Redirection",
            Assignment { .. } => "AssignmentStatement",
            If { .. } => "IfStatement",
            While { .. } => "WhileStatement",
            DoWhile { .. } => "WhileStatement",
            For { .. } => "ForStatement",
            ForEach { .. } => "ForEachStatement",
            Switch { .. } => "SwitchStatement",
            Function { .. } => "FunctionDefinition",
            Try { .. } => "TryStatement",
            Catch { .. } => "CatchClause",
            Using { .. } => "UsingStatement",
            ClassDefinition { .. } => "ClassDefinition",
            ClassMember { .. } => "ClassMember",
            EnumDefinition { .. } => "EnumDefinition",
            Flow { keyword, .. } => match keyword.as_str() {
                "return" => "ReturnStatement",
                "throw" => "ThrowStatement",
                _ => "FlowStatement",
            },

            Labeled { .. } => "LabeledStatement",
            Parameter { .. } => "Parameter",
            Ternary { .. } => "TernaryExpression",
            Binary { .. } => "BinaryExpression",
            Unary { .. } => "UnaryExpression",
            PostfixUnary { .. } => "UnaryExpression",
            Cast { .. } => "CastExpression",
            MemberAccess { .. } => "MemberAccess",
            InvokeMember { .. } => "InvokeMember",
            Index { .. } => "IndexExpression",
            Paren(_) => "ParenExpression",
            SubExpression(_) => "SubExpression",
            ScriptBlockExpression(_) => "ScriptBlockExpression",
            Array(_) => "ArrayExpression",
            ArrayLiteral(_) => "ArrayLiteral",
            Hashtable(_) => "HashtableExpression",
            Variable(_) => "Variable",
            Number(_) => "NumberLiteral",
            StringLiteral { .. } => "StringLiteral",
            TypeExpression(_) => "TypeExpression",
            BareWord(_) => "BareWord",
            Error(_) => "ErrorNode",
        }
    }

    /// Applies `f` to each direct child in source order, without allocating.
    /// This is the traversal primitive; [`Node::children`] and the `walk`
    /// methods are built on it.
    pub fn for_each_child<'a>(&'a self, f: &mut impl FnMut(&'a Node)) {
        use NodeKind::*;
        match &self.kind {
            Script(v) | ParamBlock(v) | Pipeline(v) | Array(v) | ArrayLiteral(v) => {
                for n in v {
                    f(n);
                }
            }
            ClassDefinition { members, .. } | EnumDefinition { members, .. } => {
                for n in members {
                    f(n);
                }
            }
            PipelineChain { left, right, .. } => {
                f(left);
                f(right);
            }
            Command {
                name,
                elements,
                redirections,
                csharp,
                ..
            } => {
                f(name);
                for n in elements {
                    f(n);
                }
                if let Some(cs) = csharp {
                    f(cs);
                }
                for n in redirections {
                    f(n);
                }
            }
            CSharpMemberDef(_) => {}
            CommandParameter { argument, .. } => {
                if let Some(a) = argument {
                    f(a);
                }
            }
            Redirection { target, .. } => {
                if let Some(t) = target {
                    f(t);
                }
            }
            Assignment { target, value, .. } => {
                f(target);
                f(value);
            }
            If {
                conditions,
                blocks,
                else_block,
            } => {
                for n in conditions {
                    f(n);
                }
                for n in blocks {
                    f(n);
                }
                if let Some(e) = else_block {
                    f(e);
                }
            }
            While { condition, body } => {
                f(condition);
                f(body);
            }
            DoWhile {
                body, condition, ..
            } => {
                f(body);
                f(condition);
            }
            For {
                init,
                condition,
                update,
                body,
            } => {
                if let Some(n) = init {
                    f(n);
                }
                if let Some(n) = condition {
                    f(n);
                }
                if let Some(n) = update {
                    f(n);
                }
                f(body);
            }
            ForEach {
                variable,
                iterable,
                body,
            } => {
                f(variable);
                f(iterable);
                f(body);
            }
            Switch {
                flags,
                input,
                cases,
            } => {
                for n in flags {
                    f(n);
                }
                f(input);
                for n in cases {
                    f(n);
                }
            }
            Function {
                parameters, body, ..
            } => {
                for prm in parameters {
                    f(prm);
                }
                f(body);
            }
            Try {
                body,
                catches,
                finally_block,
            } => {
                f(body);
                for n in catches {
                    f(n);
                }
                if let Some(fin) = finally_block {
                    f(fin);
                }
            }
            Catch { body } => f(body),
            ClassMember {
                parameters,
                default,
                body,
                ..
            } => {
                for n in parameters {
                    f(n);
                }
                if let Some(d) = default {
                    f(d);
                }
                if let Some(b) = body {
                    f(b);
                }
            }
            Flow { value, .. } => {
                if let Some(v) = value {
                    f(v);
                }
            }
            Labeled { statement, .. } => f(statement),
            Parameter {
                attributes,
                default,
                ..
            } => {
                for a in attributes {
                    f(a);
                }
                if let Some(d) = default {
                    f(d);
                }
            }
            Ternary {
                condition,
                if_true,
                if_false,
            } => {
                f(condition);
                f(if_true);
                f(if_false);
            }
            Binary { left, right, .. } => {
                f(left);
                f(right);
            }
            Unary { operand, .. } | PostfixUnary { operand, .. } | Cast { operand, .. } => {
                f(operand)
            }
            MemberAccess { target, .. } => f(target),
            InvokeMember { target, args, .. } => {
                f(target);
                for n in args {
                    f(n);
                }
            }
            Index { target, index } => {
                f(target);
                f(index);
            }
            Paren(n) | SubExpression(n) | ScriptBlockExpression(n) => f(n),
            Hashtable(pairs) => {
                for (k, v) in pairs {
                    f(k);
                    f(v);
                }
            }
            StringLiteral { parts, .. } => {
                for n in parts {
                    f(n);
                }
            }
            Variable(_) | Number(_) | TypeExpression(_) | BareWord(_) | Using { .. } | Error(_) => {
            }
        }
    }

    /// Visits each direct child node by mutable reference, in source order.
    /// The mutable mirror of [`Node::for_each_child`]; the two must stay in
    /// step arm for arm.
    pub fn for_each_child_mut(&mut self, f: &mut impl FnMut(&mut Node)) {
        use NodeKind::*;
        match &mut self.kind {
            Script(v) | ParamBlock(v) | Pipeline(v) | Array(v) | ArrayLiteral(v) => {
                for n in v {
                    f(n);
                }
            }
            ClassDefinition { members, .. } | EnumDefinition { members, .. } => {
                for n in members {
                    f(n);
                }
            }
            PipelineChain { left, right, .. } => {
                f(left);
                f(right);
            }
            Command {
                name,
                elements,
                redirections,
                csharp,
                ..
            } => {
                f(name);
                for n in elements {
                    f(n);
                }
                if let Some(cs) = csharp {
                    f(cs);
                }
                for n in redirections {
                    f(n);
                }
            }
            CSharpMemberDef(_) => {}
            CommandParameter { argument, .. } => {
                if let Some(a) = argument {
                    f(a);
                }
            }
            Redirection { target, .. } => {
                if let Some(t) = target {
                    f(t);
                }
            }
            Assignment { target, value, .. } => {
                f(target);
                f(value);
            }
            If {
                conditions,
                blocks,
                else_block,
            } => {
                for n in conditions {
                    f(n);
                }
                for n in blocks {
                    f(n);
                }
                if let Some(e) = else_block {
                    f(e);
                }
            }
            While { condition, body } => {
                f(condition);
                f(body);
            }
            DoWhile {
                body, condition, ..
            } => {
                f(body);
                f(condition);
            }
            For {
                init,
                condition,
                update,
                body,
            } => {
                if let Some(n) = init {
                    f(n);
                }
                if let Some(n) = condition {
                    f(n);
                }
                if let Some(n) = update {
                    f(n);
                }
                f(body);
            }
            ForEach {
                variable,
                iterable,
                body,
            } => {
                f(variable);
                f(iterable);
                f(body);
            }
            Switch {
                flags,
                input,
                cases,
            } => {
                for n in flags {
                    f(n);
                }
                f(input);
                for n in cases {
                    f(n);
                }
            }
            Function {
                parameters, body, ..
            } => {
                for prm in parameters {
                    f(prm);
                }
                f(body);
            }
            Try {
                body,
                catches,
                finally_block,
            } => {
                f(body);
                for n in catches {
                    f(n);
                }
                if let Some(fin) = finally_block {
                    f(fin);
                }
            }
            Catch { body } => f(body),
            ClassMember {
                parameters,
                default,
                body,
                ..
            } => {
                for n in parameters {
                    f(n);
                }
                if let Some(d) = default {
                    f(d);
                }
                if let Some(b) = body {
                    f(b);
                }
            }
            Flow { value, .. } => {
                if let Some(v) = value {
                    f(v);
                }
            }
            Labeled { statement, .. } => f(statement),
            Parameter {
                attributes,
                default,
                ..
            } => {
                for a in attributes {
                    f(a);
                }
                if let Some(d) = default {
                    f(d);
                }
            }
            Ternary {
                condition,
                if_true,
                if_false,
            } => {
                f(condition);
                f(if_true);
                f(if_false);
            }
            Binary { left, right, .. } => {
                f(left);
                f(right);
            }
            Unary { operand, .. } | PostfixUnary { operand, .. } | Cast { operand, .. } => {
                f(operand)
            }
            MemberAccess { target, .. } => f(target),
            InvokeMember { target, args, .. } => {
                f(target);
                for n in args {
                    f(n);
                }
            }
            Index { target, index } => {
                f(target);
                f(index);
            }
            Paren(n) | SubExpression(n) | ScriptBlockExpression(n) => f(n),
            Hashtable(pairs) => {
                for (k, v) in pairs {
                    f(k);
                    f(v);
                }
            }
            StringLiteral { parts, .. } => {
                for n in parts {
                    f(n);
                }
            }
            Variable(_) | Number(_) | TypeExpression(_) | BareWord(_) | Using { .. } | Error(_) => {
            }
        }
    }

    /// The node's direct children, in source order. Allocates a `Vec`; prefer
    /// [`Node::for_each_child`] on hot traversal paths.
    pub fn children(&self) -> Vec<&Node> {
        let mut out = Vec::new();
        self.for_each_child(&mut |c| out.push(c));
        out
    }

    /// Visits this node and every descendant, depth-first, parents first.
    pub fn walk(&self, visitor: &mut impl FnMut(&Node)) {
        visitor(self);
        self.for_each_child(&mut |c| c.walk(visitor));
    }

    /// Like [`Node::walk`], but the visitor also receives the ancestor path,
    /// ordered root first (empty for the node the walk started on).
    pub fn walk_with_ancestors(&self, visitor: &mut impl FnMut(&Node, &[&Node])) {
        fn go<'a>(
            node: &'a Node,
            stack: &mut Vec<&'a Node>,
            visitor: &mut impl FnMut(&Node, &[&Node]),
        ) {
            visitor(node, stack);
            stack.push(node);
            node.for_each_child(&mut |c| go(c, stack, visitor));
            stack.pop();
        }
        go(self, &mut Vec::new(), visitor);
    }
}
