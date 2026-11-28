use std::fmt;
use std::io::{self, Write};

use super::ast::{Node, NodeKind, StringKind};

const MAX_FUNCTION_DEPTH: usize = 2;
const MAX_BLOCK_DEPTH: usize = 4;

/// Why an AST was refused by [`write_diagnostic_script`].
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticRenderError {
    /// This kind of node is not supported in the given position.
    NodeNotAllowed {
        /// The position in the diagnostic grammar.
        role: &'static str,
        /// The native AST node label.
        label: &'static str,
    },
    /// A supported node kind contains an unsupported field value.
    FieldNotSupported {
        /// The native AST node label.
        label: &'static str,
        /// The field that caused the refusal.
        field: &'static str,
    },
}

impl fmt::Display for DiagnosticRenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotAllowed { role, label } => write!(f, "{label} is not allowed as {role}"),
            Self::FieldNotSupported { label, field } => {
                write!(f, "{label} has an unsupported {field} field")
            }
        }
    }
}

impl std::error::Error for DiagnosticRenderError {}

/// Writes a bounded subset of the native v2 AST as diagnostic text.
///
/// The root must be a `Script`. Its first node may be a `ParamBlock`, rendered
/// as `param(...)` on one line. The block may be empty; otherwise it contains
/// only `Parameter` declarations separated by `, `. A declaration is a raw
/// variable name (as for atoms below), optionally preceded by one
/// `TypeExpression` whose raw interior is `int` or `string` (ASCII-case-insensitive).
/// The type is written in brackets with its original spelling and no space
/// before the name. An optional default is written as ` = ` and an expression.
///
/// The parameter block and each statement are followed by one LF; an empty
/// script emits nothing. Statements are expressions, `=` assignments
/// with a variable target and expression value, plain commands, pipelines
/// of at least two plain commands, eligible function definitions, conditionals,
/// unlabeled loops, or `Flow` statements as described below. A `Flow` with the
/// exact stored keyword `return` is written as `return`, followed by one ASCII
/// space and an expression if a value is present.
/// An expression is an atom or a single `+`, `-`, or `*` between two atoms.
/// An atom is one of:
///
/// - A raw variable matching `\$[A-Za-z_][A-Za-z0-9_]*`.
/// - A number matching `[0-9]+`.
/// - A single-quoted string with no parts, ASCII `'` delimiters, doubled
///   interior ASCII quotes, and no ASCII control bytes (including DEL).
///
/// Plain commands have no invocation operator, redirections, or C# metadata.
/// A command name is a bareword matching `[A-Za-z_][A-Za-z0-9_-]*`. An argument
/// is an atom or a bareword matching that same pattern or `[0-9]+`. A command
/// parameter is `-name` (the same pattern as a command name), optionally
/// followed by one space and an argument. Command elements stay in AST order,
/// separated by spaces. Assignment and binary operators have one space on
/// each side; pipeline commands are joined by ` | `.
///
/// A `Function` is allowed only as a statement, with at most two function
/// levels (the root is level zero). It must have `filter == false`, a name
/// matching `[A-Za-z_][A-Za-z0-9_-]*`, and a body of kind `Script`.
/// An eligible non-filter node always renders with the canonical `function`
/// keyword; the source keyword is not retained. The name is followed
/// immediately by a parenthesized list of declarations only if the header
/// parameters are nonempty. The AST does not distinguish an absent header
/// list from an empty one, so both omit parentheses.
///
/// The header ends with ` {\n`. The body accepts the same statements as the
/// root, subject to the depth limits. A `ParamBlock` is allowed only as the
/// first child of the root or a function body with no header parameters. It
/// stays in place, including an empty `param()`.
///
/// An `If` is allowed only as a statement. It must have at least one guard
/// and one `Script` body per guard; the optional `else` body must also be a
/// `Script`. A guard is an expression as above or one `-gt` or `-eq` between two
/// atoms. These comparisons are accepted only in guards, ASCII-case-insensitively,
/// and retain their stored spelling. The first arm starts with `if (` followed
/// by its guard and `) {\n`. Later arms join the preceding aligned `}` with
/// ` elseif (` or ` else {\n`. Keywords are lowercase regardless of source
/// spelling. Empty bodies are kept without an extra blank line. Conditional
/// bodies accept statements but not a leading `ParamBlock`.
///
/// An unlabeled `While` or `DoWhile` is allowed only as a statement. It uses
/// the same guard grammar as `If` and requires a `Script` body containing
/// statements but no leading `ParamBlock`. A `while` starts with `while (`,
/// its guard, and `) {\n`. A `do` starts with `do {\n`. On the closing `}` line,
/// the `until` flag selects ` while (` or ` until (`, followed by the guard
/// and `)`. Keywords are lowercase and empty bodies have no extra blank line.
///
/// A `Flow` with the exact stored keyword `break` or `continue` must have no
/// value and be a statement lexically within a supported loop body.
/// Conditional arms inherit this permission, but it resets at function entry
/// and does not leak to siblings outside the loop body. Targeted forms are
/// refused; no runtime loop target is inferred.
///
/// Each enclosing function, conditional, or loop adds four ASCII spaces of
/// indentation and counts as one block. The root is at depth zero; four
/// enclosing blocks allow ordinary statements but no further block statement.
/// The two-function limit still applies across conditionals and loops. Closing
/// braces align with the opening statement, and each complete statement ends
/// with one LF. Labeled statements, `for`, `foreach`, other control-flow
/// forms, nested scripts used as statements, and script-block expressions
/// (including named blocks) are not supported.
///
/// This reads only AST kinds and their payloads, not source, coordinates, or
/// tokens. When starting from source, parse and handle diagnostics separately
/// before deciding whether to render. This is not lossless reconstruction or a
/// formatter, and makes no guarantees about reparsing, runtime behavior, or
/// executability. Rendering has bounded call depth and does not buffer the
/// output; this does not guarantee bounded stack use when constructing or
/// dropping arbitrary trees. This writer does not reproduce the former v1
/// source dump's parse-error tolerance or emitted bytes.
///
/// # Errors
///
/// An unsupported AST returns [`io::ErrorKind::InvalidInput`] containing a
/// [`DiagnosticRenderError`] before the writer is touched. Otherwise writes
/// use [`Write::write_all`] semantics. Errors from `write_all` are returned
/// unchanged, even if some output has already been written. This function
/// does not flush or roll back writes.
///
/// # Example
///
/// ```
/// use poshtree::v2::{write_diagnostic_script, Node, NodeKind, TokenRange};
///
/// let node = |kind| Node {
///     kind,
///     span: Default::default(),
///     range: TokenRange { first: 0, end: 0 },
/// };
/// let script = node(NodeKind::Script(vec![
///     node(NodeKind::ParamBlock(vec![node(NodeKind::Parameter {
///         attributes: vec![node(NodeKind::TypeExpression("INT".into()))],
///         name: "$n".into(),
///         name_span: Default::default(),
///         default: Some(Box::new(node(NodeKind::Number("1".into())))),
///     })])),
///     node(NodeKind::Assignment {
///         target: Box::new(node(NodeKind::Variable("$x".into()))),
///         op: "=".into(),
///         value: Box::new(node(NodeKind::Variable("$n".into()))),
///     }),
/// ]));
/// let mut text = Vec::new();
/// write_diagnostic_script(&script, &mut text)?;
/// assert_eq!(text, b"param([INT]$n = 1)\n$x = $n\n");
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn write_diagnostic_script<W: std::io::Write + ?Sized>(
    script: &Node,
    writer: &mut W,
) -> std::io::Result<()> {
    // Validate the entire tree against the same grammar used for emission,
    // without buffering any output or touching the caller's writer.
    DiagnosticWriter(&mut io::sink()).script(script)?;
    DiagnosticWriter(writer).script(script)
}

#[derive(Clone, Copy)]
struct StatementContext {
    block_depth: usize,
    function_depth: usize,
    allow_leading_param_block: bool,
    in_loop_body: bool,
}

impl StatementContext {
    fn enter_block(self, node: &Node) -> io::Result<Self> {
        if self.block_depth == MAX_BLOCK_DEPTH {
            return Err(node_not_allowed(
                node,
                "statement inside four enclosing blocks",
            ));
        }
        Ok(Self {
            block_depth: self.block_depth + 1,
            allow_leading_param_block: false,
            ..self
        })
    }
}

enum ExpressionRole {
    Ordinary(&'static str),
    Guard(&'static str),
}

// Only function, conditional, and loop bodies re-enter a statement list, with a
// checked increase in block depth.
struct DiagnosticWriter<'a, W: ?Sized>(&'a mut W);

impl<W: Write + ?Sized> DiagnosticWriter<'_, W> {
    fn script(&mut self, node: &Node) -> io::Result<()> {
        self.statement_list(
            script_statements(node, "script root")?,
            StatementContext {
                block_depth: 0,
                function_depth: 0,
                allow_leading_param_block: true,
                in_loop_body: false,
            },
        )
    }

    fn statement_list(&mut self, statements: &[Node], context: StatementContext) -> io::Result<()> {
        for (i, statement) in statements.iter().enumerate() {
            self.indent(context.block_depth)?;
            match (i, &statement.kind) {
                (0, NodeKind::ParamBlock(parameters)) if context.allow_leading_param_block => {
                    self.0.write_all(b"param(")?;
                    self.parameter_list(parameters)?;
                    self.0.write_all(b")")?;
                }
                _ => self.statement(statement, context)?,
            }
            self.0.write_all(b"\n")?;
        }
        Ok(())
    }

    fn indent(&mut self, depth: usize) -> io::Result<()> {
        for _ in 0..depth {
            self.0.write_all(b"    ")?;
        }
        Ok(())
    }

    fn function(&mut self, node: &Node, context: StatementContext) -> io::Result<()> {
        if context.function_depth == MAX_FUNCTION_DEPTH {
            return Err(node_not_allowed(
                node,
                "statement inside two enclosing functions",
            ));
        }
        let body_context = context.enter_block(node)?;
        let NodeKind::Function {
            name,
            filter,
            parameters,
            body,
            ..
        } = &node.kind
        else {
            return Err(node_not_allowed(node, "statement"));
        };
        if *filter {
            return Err(field_not_supported(node, "filter"));
        }
        self.0.write_all(b"function ")?;
        self.text(node, "name", name, is_word(name))?;
        let statements = script_statements(body, "function body")?;
        if !parameters.is_empty() {
            if let Some(
                leading_block @ Node {
                    kind: NodeKind::ParamBlock(_),
                    ..
                },
            ) = statements.first()
            {
                return Err(node_not_allowed(
                    leading_block,
                    "function body statement with header parameters",
                ));
            }
            self.0.write_all(b"(")?;
            self.parameter_list(parameters)?;
            self.0.write_all(b")")?;
        }
        self.braced_body(
            statements,
            StatementContext {
                function_depth: context.function_depth + 1,
                allow_leading_param_block: parameters.is_empty(),
                in_loop_body: false,
                ..body_context
            },
        )
    }

    fn conditional(&mut self, node: &Node, context: StatementContext) -> io::Result<()> {
        let body_context = context.enter_block(node)?;
        let NodeKind::If {
            conditions,
            blocks,
            else_block,
        } = &node.kind
        else {
            return Err(node_not_allowed(node, "statement"));
        };
        if conditions.is_empty() {
            return Err(field_not_supported(node, "conditions"));
        }
        if conditions.len() != blocks.len() {
            return Err(field_not_supported(node, "blocks"));
        }
        for (i, (condition, block)) in conditions.iter().zip(blocks).enumerate() {
            self.0
                .write_all(if i == 0 { b"if (" } else { b" elseif (" })?;
            self.expression(condition, ExpressionRole::Guard("if condition"))?;
            self.0.write_all(b")")?;
            self.braced_body(script_statements(block, "if branch body")?, body_context)?;
        }
        if let Some(body) = else_block {
            self.0.write_all(b" else")?;
            self.braced_body(script_statements(body, "else body")?, body_context)?;
        }
        Ok(())
    }

    fn loop_statement(&mut self, node: &Node, context: StatementContext) -> io::Result<()> {
        let body_context = StatementContext {
            in_loop_body: true,
            ..context.enter_block(node)?
        };
        match &node.kind {
            NodeKind::While { condition, body } => {
                self.0.write_all(b"while (")?;
                self.expression(condition, ExpressionRole::Guard("while condition"))?;
                self.0.write_all(b")")?;
                self.braced_body(script_statements(body, "while body")?, body_context)
            }
            NodeKind::DoWhile {
                body,
                condition,
                until,
            } => {
                self.0.write_all(b"do")?;
                self.braced_body(script_statements(body, "do body")?, body_context)?;
                self.0
                    .write_all(if *until { b" until (" } else { b" while (" })?;
                self.expression(condition, ExpressionRole::Guard("do condition"))?;
                self.0.write_all(b")")
            }
            _ => Err(node_not_allowed(node, "statement")),
        }
    }

    fn braced_body(
        &mut self,
        statements: &[Node],
        body_context: StatementContext,
    ) -> io::Result<()> {
        self.0.write_all(b" {\n")?;
        self.statement_list(statements, body_context)?;
        self.indent(body_context.block_depth - 1)?;
        self.0.write_all(b"}")
    }

    fn parameter_list(&mut self, parameters: &[Node]) -> io::Result<()> {
        for (i, parameter) in parameters.iter().enumerate() {
            if i != 0 {
                self.0.write_all(b", ")?;
            }
            self.parameter_declaration(parameter)?;
        }
        Ok(())
    }

    fn parameter_declaration(&mut self, node: &Node) -> io::Result<()> {
        let NodeKind::Parameter {
            attributes,
            name,
            default,
            ..
        } = &node.kind
        else {
            return Err(node_not_allowed(node, "parameter declaration"));
        };
        match attributes.as_slice() {
            [] => {}
            [Node {
                kind: NodeKind::TypeExpression(value),
                ..
            }] if value.eq_ignore_ascii_case("int") || value.eq_ignore_ascii_case("string") => {
                self.0.write_all(b"[")?;
                self.0.write_all(value.as_bytes())?;
                self.0.write_all(b"]")?;
            }
            _ => return Err(field_not_supported(node, "attributes")),
        }
        self.text(node, "name", name, is_variable_name(name))?;
        if let Some(default) = default {
            self.0.write_all(b" = ")?;
            self.expression(default, ExpressionRole::Ordinary("parameter default"))?;
        }
        Ok(())
    }

    fn statement(&mut self, node: &Node, context: StatementContext) -> io::Result<()> {
        match &node.kind {
            NodeKind::Function { .. } => self.function(node, context),
            NodeKind::If { .. } => self.conditional(node, context),
            NodeKind::While { .. } | NodeKind::DoWhile { .. } => self.loop_statement(node, context),
            NodeKind::Assignment { target, op, value } => {
                if op != "=" {
                    return Err(field_not_supported(node, "op"));
                }
                self.variable(target, "assignment target")?;
                self.0.write_all(b" = ")?;
                self.expression(value, ExpressionRole::Ordinary("assignment value"))
            }
            NodeKind::Command { .. } => self.command(node, "statement"),
            NodeKind::Pipeline(commands) => {
                if commands.len() < 2 {
                    return Err(field_not_supported(node, "elements"));
                }
                for (i, command) in commands.iter().enumerate() {
                    if i != 0 {
                        self.0.write_all(b" | ")?;
                    }
                    self.command(command, "pipeline element")?;
                }
                Ok(())
            }
            NodeKind::Flow { keyword, value } if keyword == "return" => {
                self.0.write_all(b"return")?;
                if let Some(value) = value {
                    self.0.write_all(b" ")?;
                    self.expression(value, ExpressionRole::Ordinary("return value"))?;
                }
                Ok(())
            }
            NodeKind::Flow { keyword, value }
                if context.in_loop_body && matches!(keyword.as_str(), "break" | "continue") =>
            {
                if value.is_some() {
                    return Err(field_not_supported(node, "value"));
                }
                self.0.write_all(keyword.as_bytes())
            }
            _ => self.expression(node, ExpressionRole::Ordinary("statement")),
        }
    }

    fn expression(&mut self, node: &Node, role: ExpressionRole) -> io::Result<()> {
        let NodeKind::Binary { op, left, right } = &node.kind else {
            return self.atom(
                node,
                match role {
                    ExpressionRole::Ordinary(role) | ExpressionRole::Guard(role) => role,
                },
            );
        };
        match op.as_str() {
            "+" | "-" | "*" => {}
            _ if matches!(role, ExpressionRole::Guard(_))
                && (op.eq_ignore_ascii_case("-gt") || op.eq_ignore_ascii_case("-eq")) => {}
            _ => return Err(field_not_supported(node, "op")),
        }
        self.atom(left, "binary operand")?;
        self.0.write_all(b" ")?;
        self.0.write_all(op.as_bytes())?;
        self.0.write_all(b" ")?;
        self.atom(right, "binary operand")
    }

    fn atom(&mut self, node: &Node, role: &'static str) -> io::Result<()> {
        match &node.kind {
            NodeKind::Variable(_) => self.variable(node, role),
            NodeKind::Number(value) => self.text(node, "value", value, is_decimal(value)),
            NodeKind::StringLiteral { kind, value, parts } => {
                if *kind != StringKind::Single {
                    return Err(field_not_supported(node, "kind"));
                }
                if !parts.is_empty() {
                    return Err(field_not_supported(node, "parts"));
                }
                self.text(node, "value", value, is_single_quoted(value))
            }
            _ => Err(node_not_allowed(node, role)),
        }
    }

    fn variable(&mut self, node: &Node, role: &'static str) -> io::Result<()> {
        let NodeKind::Variable(value) = &node.kind else {
            return Err(node_not_allowed(node, role));
        };
        self.text(node, "value", value, is_variable_name(value))
    }

    fn command(&mut self, node: &Node, role: &'static str) -> io::Result<()> {
        let NodeKind::Command {
            name,
            invocation,
            elements,
            redirections,
            csharp,
        } = &node.kind
        else {
            return Err(node_not_allowed(node, role));
        };
        if *invocation {
            return Err(field_not_supported(node, "invocation"));
        }
        if !redirections.is_empty() {
            return Err(field_not_supported(node, "redirections"));
        }
        if csharp.is_some() {
            return Err(field_not_supported(node, "csharp"));
        }
        let NodeKind::BareWord(name_text) = &name.kind else {
            return Err(node_not_allowed(name, "command name"));
        };
        self.text(name, "value", name_text, is_word(name_text))?;
        for element in elements {
            self.0.write_all(b" ")?;
            if let NodeKind::CommandParameter { name, argument } = &element.kind {
                self.0.write_all(b"-")?;
                self.text(element, "name", name, is_word(name))?;
                if let Some(argument) = argument {
                    self.0.write_all(b" ")?;
                    self.argument(argument)?;
                }
            } else {
                self.argument(element)?;
            }
        }
        Ok(())
    }

    fn argument(&mut self, node: &Node) -> io::Result<()> {
        match &node.kind {
            NodeKind::BareWord(value) => {
                self.text(node, "value", value, is_word(value) || is_decimal(value))
            }
            _ => self.atom(node, "command argument"),
        }
    }

    fn text(
        &mut self,
        node: &Node,
        field: &'static str,
        value: &str,
        valid: bool,
    ) -> io::Result<()> {
        if !valid {
            return Err(field_not_supported(node, field));
        }
        self.0.write_all(value.as_bytes())
    }
}

fn script_statements<'a>(node: &'a Node, role: &'static str) -> io::Result<&'a [Node]> {
    let NodeKind::Script(statements) = &node.kind else {
        return Err(node_not_allowed(node, role));
    };
    Ok(statements)
}

fn node_not_allowed(node: &Node, role: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        DiagnosticRenderError::NodeNotAllowed {
            role,
            label: node.label(),
        },
    )
}

fn field_not_supported(node: &Node, field: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        DiagnosticRenderError::FieldNotSupported {
            label: node.label(),
            field,
        },
    )
}

fn is_word(value: &str) -> bool {
    is_identifier(value, true)
}

fn is_variable_name(value: &str) -> bool {
    value
        .strip_prefix('$')
        .is_some_and(|name| is_identifier(name, false))
}

fn is_identifier(value: &str, allow_hyphen: bool) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_' || (allow_hyphen && b == b'-'))
}

fn is_decimal(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
}

fn is_single_quoted(value: &str) -> bool {
    let [b'\'', interior @ .., b'\''] = value.as_bytes() else {
        return false;
    };
    let mut bytes = interior.iter().copied();
    while let Some(b) = bytes.next() {
        if b.is_ascii_control() || (b == b'\'' && bytes.next() != Some(b'\'')) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{CSharpMemberDef, Span, TokenRange};
    use DiagnosticRenderError::{FieldNotSupported, NodeNotAllowed};

    fn node(kind: NodeKind) -> Node {
        Node {
            kind,
            span: Span::default(),
            range: TokenRange { first: 0, end: 0 },
        }
    }

    fn script(statements: Vec<Node>) -> Node {
        node(NodeKind::Script(statements))
    }

    fn variable(value: &str) -> Node {
        node(NodeKind::Variable(value.into()))
    }

    fn number(value: &str) -> Node {
        node(NodeKind::Number(value.into()))
    }

    fn string(value: &str) -> Node {
        string_literal(StringKind::Single, value, Vec::new())
    }

    fn string_literal(kind: StringKind, value: &str, parts: Vec<Node>) -> Node {
        node(NodeKind::StringLiteral {
            kind,
            value: value.into(),
            parts,
        })
    }

    fn bareword(value: &str) -> Node {
        node(NodeKind::BareWord(value.into()))
    }

    fn assignment(target: Node, op: &str, value: Node) -> Node {
        node(NodeKind::Assignment {
            target: Box::new(target),
            op: op.into(),
            value: Box::new(value),
        })
    }

    fn flow(keyword: &str, value: Option<Node>) -> Node {
        node(NodeKind::Flow {
            keyword: keyword.into(),
            value: value.map(Box::new),
        })
    }

    fn binary(left: Node, op: &str, right: Node) -> Node {
        node(NodeKind::Binary {
            left: Box::new(left),
            op: op.into(),
            right: Box::new(right),
        })
    }

    fn command(name: Node, elements: Vec<Node>) -> Node {
        node(NodeKind::Command {
            name: Box::new(name),
            invocation: false,
            elements,
            redirections: Vec::new(),
            csharp: None,
        })
    }

    fn parameter(name: &str, argument: Option<Node>) -> Node {
        node(NodeKind::CommandParameter {
            name: name.into(),
            argument: argument.map(Box::new),
        })
    }

    fn param_block(parameters: Vec<Node>) -> Node {
        node(NodeKind::ParamBlock(parameters))
    }

    fn parameter_declaration(name: &str, attributes: Vec<Node>, default: Option<Node>) -> Node {
        node(NodeKind::Parameter {
            name: name.into(),
            name_span: Span::default(),
            attributes,
            default: default.map(Box::new),
        })
    }

    fn type_expression(value: &str) -> Node {
        node(NodeKind::TypeExpression(value.into()))
    }

    fn function(name: &str, statements: Vec<Node>) -> Node {
        function_definition(name, false, vec![], script(statements))
    }

    fn function_definition(name: &str, filter: bool, parameters: Vec<Node>, body: Node) -> Node {
        node(NodeKind::Function {
            name: name.into(),
            name_span: Span::default(),
            filter,
            parameters,
            body: Box::new(body),
        })
    }

    fn conditional(conditions: Vec<Node>, blocks: Vec<Node>, else_block: Option<Node>) -> Node {
        node(NodeKind::If {
            conditions,
            blocks,
            else_block: else_block.map(Box::new),
        })
    }

    fn if_body(statements: Vec<Node>) -> Node {
        conditional(vec![number("1")], vec![script(statements)], None)
    }

    fn while_loop(condition: Node, body: Node) -> Node {
        node(NodeKind::While {
            condition: Box::new(condition),
            body: Box::new(body),
        })
    }

    fn do_loop(body: Node, condition: Node, until: bool) -> Node {
        node(NodeKind::DoWhile {
            body: Box::new(body),
            condition: Box::new(condition),
            until,
        })
    }

    fn simple_assignment() -> Node {
        script(vec![assignment(variable("$x"), "=", number("1"))])
    }

    #[derive(Debug, PartialEq, Eq)]
    struct BrokenPipeSentinel(u32);

    impl fmt::Display for BrokenPipeSentinel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test writer refused output ({})", self.0)
        }
    }

    impl std::error::Error for BrokenPipeSentinel {}

    struct WriterSpy {
        bytes: Vec<u8>,
        chunk_limit: usize,
        fail_after: usize,
        writes: usize,
        flushes: usize,
    }

    impl WriterSpy {
        fn new(chunk_limit: usize, fail_after: usize) -> Self {
            Self {
                bytes: Vec::new(),
                chunk_limit,
                fail_after,
                writes: 0,
                flushes: 0,
            }
        }
    }

    impl Write for WriterSpy {
        fn write(&mut self, input: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            let remaining = self.fail_after - self.bytes.len();
            if remaining == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    BrokenPipeSentinel(24),
                ));
            }
            let accepted = input.len().min(self.chunk_limit).min(remaining);
            self.bytes.extend_from_slice(&input[..accepted]);
            Ok(accepted)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn empty_script_never_touches_writer() {
        let script = script(Vec::new());
        let mut writer = WriterSpy::new(1, 0);

        write_diagnostic_script(&script, &mut writer).unwrap();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
    }

    #[test]
    fn empty_param_block_emits_one_line() {
        let script = script(vec![param_block(vec![])]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"param()\n");
    }

    #[test]
    fn leading_param_block_preserves_declaration_order_and_raw_text() {
        let script = script(vec![
            param_block(vec![
                node(NodeKind::Parameter {
                    name: "$_Flag9".into(),
                    name_span: Span {
                        start: usize::MAX,
                        end: 0,
                    },
                    attributes: vec![],
                    default: None,
                }),
                parameter_declaration(
                    "$n",
                    vec![type_expression("INT")],
                    Some(binary(number("2"), "*", number("03"))),
                ),
                parameter_declaration(
                    "$Text",
                    vec![type_expression("sTrInG")],
                    Some(string("'café''字'")),
                ),
            ]),
            variable("$Text"),
        ]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            "param($_Flag9, [INT]$n = 2 * 03, [sTrInG]$Text = 'café''字')\n$Text\n".as_bytes()
        );
    }

    #[test]
    fn param_block_refuses_non_declaration_entries() {
        for (entry, label) in [
            (variable("$n"), "Variable"),
            (parameter("n", None), "CommandParameter"),
        ] {
            assert_script_refused(
                &script(vec![param_block(vec![entry])]),
                NodeNotAllowed {
                    role: "parameter declaration",
                    label,
                },
            );
        }
    }

    #[test]
    fn parameter_declaration_refuses_unsupported_attributes() {
        for attributes in [
            vec![type_expression("bool")],
            vec![type_expression("Parameter(Mandatory)")],
            vec![type_expression("")],
            vec![type_expression(" int ")],
            vec![type_expression("[int]")],
            vec![type_expression("strİng")],
            vec![bareword("int")],
            vec![type_expression("int"), type_expression("string")],
        ] {
            let declaration = parameter_declaration("$n", attributes, None);
            assert_script_refused(
                &script(vec![param_block(vec![declaration])]),
                FieldNotSupported {
                    label: "Parameter",
                    field: "attributes",
                },
            );
        }
    }

    #[test]
    fn parameter_declaration_refuses_invalid_raw_names() {
        for name in ["", "$", "n", "$1n", "$a-b", "$env:N", "${n}", "$café", "@n"] {
            let declaration = parameter_declaration(name, vec![], None);
            assert_script_refused(
                &script(vec![param_block(vec![declaration])]),
                FieldNotSupported {
                    label: "Parameter",
                    field: "name",
                },
            );
        }
    }

    #[test]
    fn parameter_declaration_preserves_expression_diagnostics_for_defaults() {
        for (default, expected) in [
            (
                node(NodeKind::ArrayLiteral(vec![number("1")])),
                NodeNotAllowed {
                    role: "parameter default",
                    label: "ArrayLiteral",
                },
            ),
            (
                binary(number("1"), "+", binary(number("2"), "*", number("3"))),
                NodeNotAllowed {
                    role: "binary operand",
                    label: "BinaryExpression",
                },
            ),
            (
                binary(number("1"), "/", number("2")),
                FieldNotSupported {
                    label: "BinaryExpression",
                    field: "op",
                },
            ),
            (
                binary(number("1"), "+", variable("$env:N")),
                FieldNotSupported {
                    label: "Variable",
                    field: "value",
                },
            ),
        ] {
            let declaration = parameter_declaration("$n", vec![], Some(default));
            assert_script_refused(&script(vec![param_block(vec![declaration])]), expected);
        }
    }

    #[test]
    fn script_refuses_param_blocks_outside_the_first_root_child() {
        for (root, role) in [
            (param_block(vec![]), "script root"),
            (script(vec![number("1"), param_block(vec![])]), "statement"),
            (
                script(vec![param_block(vec![]), param_block(vec![])]),
                "statement",
            ),
        ] {
            assert_script_refused(
                &root,
                NodeNotAllowed {
                    role,
                    label: "ParamBlock",
                },
            );
        }
    }

    #[test]
    fn invalid_final_parameter_default_leaves_writer_untouched() {
        let script = script(vec![param_block(vec![
            parameter_declaration("$ok", vec![], None),
            parameter_declaration("$n", vec![type_expression("int")], Some(number("-1"))),
        ])]);
        let mut writer = WriterSpy::new(usize::MAX, 0);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
        assert_refusal(
            &error,
            &FieldNotSupported {
                label: "NumberLiteral",
                field: "value",
            },
        );
    }

    #[test]
    fn empty_function_emits_only_header_and_closing_brace() {
        let script = script(vec![function("F", vec![])]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"function F {\n}\n");
    }

    #[test]
    fn function_with_empty_body_param_block_keeps_it_on_its_own_line() {
        let script = script(vec![function("F", vec![param_block(vec![])])]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"function F {\n    param()\n}\n");
    }

    #[test]
    fn script_emits_two_function_levels_with_local_parameters_and_sibling_indentation() {
        let script = script(vec![
            function_definition(
                "Outer-1",
                false,
                vec![
                    parameter_declaration("$n", vec![type_expression("INT")], Some(number("01"))),
                    parameter_declaration("$text", vec![], None),
                ],
                script(vec![
                    function(
                        "_Inner",
                        vec![
                            param_block(vec![parameter_declaration(
                                "$value",
                                vec![type_expression("sTrInG")],
                                Some(string("'a''b'")),
                            )]),
                            command(bareword("Write-Output"), vec![variable("$value")]),
                        ],
                    ),
                    function("InnerSibling", vec![]),
                    variable("$text"),
                ]),
            ),
            function("RootSibling", vec![]),
            number("2"),
        ]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            concat!(
                "function Outer-1([INT]$n = 01, $text) {\n",
                "    function _Inner {\n",
                "        param([sTrInG]$value = 'a''b')\n",
                "        Write-Output $value\n",
                "    }\n",
                "    function InnerSibling {\n",
                "    }\n",
                "    $text\n",
                "}\n",
                "function RootSibling {\n",
                "}\n",
                "2\n",
            )
            .as_bytes()
        );
    }

    #[test]
    fn function_refuses_filter_before_validating_name_or_body() {
        let filter = function_definition("", true, vec![], number("1"));

        assert_script_refused(
            &script(vec![filter]),
            FieldNotSupported {
                label: "FunctionDefinition",
                field: "filter",
            },
        );
    }

    #[test]
    fn function_refuses_names_outside_the_raw_word_pattern() {
        for name in ["", "1F", "-F", "global:F", "café", "F G"] {
            assert_script_refused(
                &script(vec![function(name, vec![])]),
                FieldNotSupported {
                    label: "FunctionDefinition",
                    field: "name",
                },
            );
        }
    }

    #[test]
    fn function_refuses_non_script_bodies() {
        for (body, label) in [
            (param_block(vec![]), "ParamBlock"),
            (
                node(NodeKind::ScriptBlockExpression(Box::new(script(vec![])))),
                "ScriptBlockExpression",
            ),
        ] {
            assert_script_refused(
                &script(vec![function_definition("F", false, vec![], body)]),
                NodeNotAllowed {
                    role: "function body",
                    label,
                },
            );
        }
    }

    #[test]
    fn function_header_refuses_non_declaration_entries() {
        let definition = function_definition("F", false, vec![variable("$x")], script(vec![]));

        assert_script_refused(
            &script(vec![definition]),
            NodeNotAllowed {
                role: "parameter declaration",
                label: "Variable",
            },
        );
    }

    #[test]
    fn function_with_header_parameters_refuses_leading_body_param_block() {
        let definition = function_definition(
            "F",
            false,
            vec![parameter_declaration("$x", vec![], None)],
            script(vec![param_block(vec![])]),
        );

        assert_script_refused(
            &script(vec![definition]),
            NodeNotAllowed {
                role: "function body statement with header parameters",
                label: "ParamBlock",
            },
        );
    }

    #[test]
    fn function_refuses_disallowed_body_statement_kinds_and_placements() {
        for (statements, label) in [
            (vec![number("1"), param_block(vec![])], "ParamBlock"),
            (vec![param_block(vec![]), param_block(vec![])], "ParamBlock"),
            (vec![script(vec![number("1")])], "ScriptBlock"),
            (
                vec![node(NodeKind::ScriptBlockExpression(Box::new(script(
                    vec![],
                ))))],
                "ScriptBlockExpression",
            ),
            (
                vec![node(NodeKind::For {
                    init: None,
                    condition: None,
                    update: None,
                    body: Box::new(script(vec![])),
                })],
                "ForStatement",
            ),
        ] {
            assert_script_refused(
                &script(vec![function("F", statements)]),
                NodeNotAllowed {
                    role: "statement",
                    label,
                },
            );
        }
    }

    #[test]
    fn script_refuses_third_function_level_before_inspecting_fields() {
        let third = function_definition("", true, vec![number("1")], number("2"));
        let script = script(vec![function(
            "Outer",
            vec![function("Inner", vec![third])],
        )]);

        assert_script_refused(
            &script,
            NodeNotAllowed {
                role: "statement inside two enclosing functions",
                label: "FunctionDefinition",
            },
        );
    }

    #[test]
    fn unsupported_final_inner_statement_leaves_writer_untouched() {
        let script = script(vec![
            number("1"),
            function(
                "Outer",
                vec![function(
                    "Inner",
                    vec![
                        number("2"),
                        node(NodeKind::Flow {
                            keyword: "throw".into(),
                            value: Some(Box::new(number("3"))),
                        }),
                    ],
                )],
            ),
        ]);
        let mut writer = WriterSpy::new(usize::MAX, 0);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
        assert_refusal(
            &error,
            &NodeNotAllowed {
                role: "statement",
                label: "ThrowStatement",
            },
        );
    }

    #[test]
    fn function_write_failure_after_eighteen_bytes_preserves_prefix_and_error_without_flushing() {
        let script = script(vec![function(
            "F",
            vec![command(bareword("Write-Output"), vec![])],
        )]);
        let mut writer = WriterSpy::new(usize::MAX, 18);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.flushes),
            (&b"function F {\n    W"[..], 0)
        );
        assert_eq!(
            (
                error.kind(),
                error
                    .get_ref()
                    .and_then(|e| e.downcast_ref::<BrokenPipeSentinel>())
            ),
            (io::ErrorKind::BrokenPipe, Some(&BrokenPipeSentinel(24)))
        );
    }

    #[test]
    fn root_return_statements_emit_bare_and_valued_lines() {
        let script = script(vec![
            flow("return", None),
            flow("return", Some(number("03"))),
        ]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"return\nreturn 03\n");
    }

    #[test]
    fn function_return_statements_emit_indented_bare_and_valued_lines() {
        let script = script(vec![function(
            "F",
            vec![
                flow("return", Some(binary(variable("$n"), "+", number("2")))),
                flow("return", None),
            ],
        )]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"function F {\n    return $n + 2\n    return\n}\n");
    }

    #[test]
    fn return_with_command_value_leaves_writer_untouched() {
        let script = script(vec![
            number("1"),
            flow("return", Some(command(bareword("Get"), vec![]))),
        ]);
        let mut writer = WriterSpy::new(usize::MAX, 0);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
        assert_refusal(
            &error,
            &NodeNotAllowed {
                role: "return value",
                label: "Command",
            },
        );
    }

    #[test]
    fn script_refuses_flow_keywords_other_than_exact_lowercase_return() {
        for keyword in ["break", "continue", "exit", "RETURN", "return ", ""] {
            assert_script_refused(
                &script(vec![flow(keyword, None)]),
                NodeNotAllowed {
                    role: "statement",
                    label: "FlowStatement",
                },
            );
        }
    }

    #[test]
    fn script_refuses_bareword_return_as_a_statement() {
        assert_script_refused(
            &script(vec![bareword("return")]),
            NodeNotAllowed {
                role: "statement",
                label: "BareWord",
            },
        );
    }

    #[test]
    fn script_refuses_return_in_expression_positions() {
        for (statement, role) in [
            (
                assignment(variable("$x"), "=", flow("return", None)),
                "assignment value",
            ),
            (flow("return", Some(flow("return", None))), "return value"),
        ] {
            assert_script_refused(
                &script(vec![statement]),
                NodeNotAllowed {
                    role,
                    label: "ReturnStatement",
                },
            );
        }
    }

    #[test]
    fn if_emits_each_arm_in_order_and_keeps_raw_guard_spelling() {
        let script = script(vec![conditional(
            vec![
                binary(variable("$n"), "-gT", number("01")),
                binary(variable("$n"), "-Eq", number("0")),
                binary(variable("$n"), "*", number("2")),
            ],
            vec![
                script(vec![flow("return", Some(number("1")))]),
                script(vec![]),
                script(vec![flow("return", None)]),
            ],
            Some(script(vec![])),
        )]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            b"if ($n -gT 01) {\n    return 1\n} elseif ($n -Eq 0) {\n} \
              elseif ($n * 2) {\n    return\n} else {\n}\n"
        );
    }

    #[test]
    fn if_refuses_missing_guards_or_unequal_branch_counts_before_visiting_children() {
        for (conditions, blocks, field) in [
            (vec![], vec![], "conditions"),
            (vec![], vec![number("1")], "conditions"),
            (vec![bareword("bad")], vec![], "blocks"),
            (
                vec![bareword("bad")],
                vec![number("1"), script(vec![])],
                "blocks",
            ),
        ] {
            let statement = conditional(conditions, blocks, Some(number("2")));

            assert_script_refused(
                &script(vec![statement]),
                FieldNotSupported {
                    label: "IfStatement",
                    field,
                },
            );
        }
    }

    #[test]
    fn if_validates_each_guard_then_its_body_before_the_next_arm() {
        for (conditions, blocks, role, label) in [
            (
                vec![bareword("bad")],
                vec![number("1")],
                "if condition",
                "BareWord",
            ),
            (
                vec![number("1"), bareword("bad")],
                vec![number("2"), script(vec![])],
                "if branch body",
                "NumberLiteral",
            ),
            (
                vec![number("1"), bareword("bad")],
                vec![script(vec![]), number("2")],
                "if condition",
                "BareWord",
            ),
        ] {
            let statement = conditional(conditions, blocks, Some(variable("$badElse")));

            assert_script_refused(&script(vec![statement]), NodeNotAllowed { role, label });
        }
    }

    #[test]
    fn if_refuses_non_script_else_after_valid_arms_without_touching_writer() {
        let statement = conditional(
            vec![number("1"), number("2")],
            vec![script(vec![number("3")]), script(vec![])],
            Some(if_body(vec![])),
        );
        let script = script(vec![number("0"), statement]);
        let mut writer = WriterSpy::new(usize::MAX, 0);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
        assert_refusal(
            &error,
            &NodeNotAllowed {
                role: "else body",
                label: "IfStatement",
            },
        );
    }

    #[test]
    fn if_with_malformed_return_at_end_of_else_leaves_writer_untouched() {
        let statement = conditional(
            vec![variable("$x")],
            vec![script(vec![flow("return", None)])],
            Some(script(vec![
                flow("return", Some(number("2"))),
                flow("return", Some(number("3\n"))),
            ])),
        );
        let script = script(vec![number("0"), statement]);
        let mut writer = WriterSpy::new(usize::MAX, 0);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
        assert_refusal(
            &error,
            &FieldNotSupported {
                label: "NumberLiteral",
                field: "value",
            },
        );
    }

    #[test]
    fn if_write_failure_after_sixteen_bytes_preserves_prefix_and_error_without_flushing() {
        let statement = conditional(
            vec![variable("$x")],
            vec![script(vec![flow("return", None)])],
            None,
        );
        let script = script(vec![statement]);
        let mut writer = WriterSpy::new(usize::MAX, 16);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.flushes),
            (&b"if ($x) {\n    re"[..], 0)
        );
        assert_eq!(
            (
                error.kind(),
                error
                    .get_ref()
                    .and_then(|e| e.downcast_ref::<BrokenPipeSentinel>())
            ),
            (io::ErrorKind::BrokenPipe, Some(&BrokenPipeSentinel(24)))
        );
    }

    #[test]
    fn if_refuses_non_expression_guards_with_the_condition_role() {
        for (condition, label) in [
            (command(bareword("Get"), vec![]), "Command"),
            (
                node(NodeKind::Pipeline(vec![
                    command(bareword("Get"), vec![]),
                    command(bareword("Show"), vec![]),
                ])),
                "Pipeline",
            ),
            (
                assignment(variable("$x"), "=", number("1")),
                "AssignmentStatement",
            ),
            (if_body(vec![]), "IfStatement"),
        ] {
            let statement = conditional(vec![condition], vec![script(vec![])], None);

            assert_script_refused(
                &script(vec![statement]),
                NodeNotAllowed {
                    role: "if condition",
                    label,
                },
            );
        }
    }

    #[test]
    fn if_guard_refuses_unsupported_raw_operators_before_visiting_operands() {
        for op in ["-ne", "-cgt", "-eq ", " -gt", "-ｅq", "/"] {
            let condition = binary(bareword("bad"), op, number("1"));
            let statement = conditional(vec![condition], vec![script(vec![])], None);

            assert_script_refused(
                &script(vec![statement]),
                FieldNotSupported {
                    label: "BinaryExpression",
                    field: "op",
                },
            );
        }
    }

    #[test]
    fn if_guard_refuses_nested_binary_operands() {
        for condition in [
            binary(
                binary(variable("$x"), "-gt", number("1")),
                "-eq",
                number("0"),
            ),
            binary(variable("$x"), "-eq", binary(number("1"), "+", number("2"))),
        ] {
            let statement = conditional(vec![condition], vec![script(vec![])], None);

            assert_script_refused(
                &script(vec![statement]),
                NodeNotAllowed {
                    role: "binary operand",
                    label: "BinaryExpression",
                },
            );
        }
    }

    #[test]
    fn if_guard_comparison_still_validates_atom_fields() {
        let condition = binary(number("1"), "-EQ", variable("$env:X"));
        let statement = conditional(vec![condition], vec![script(vec![])], None);

        assert_script_refused(
            &script(vec![statement]),
            FieldNotSupported {
                label: "Variable",
                field: "value",
            },
        );
    }

    #[test]
    fn script_refuses_comparison_operators_outside_guards() {
        let comparison = || binary(variable("$x"), "-Gt", number("1"));
        for statement in [
            comparison(),
            assignment(variable("$y"), "=", comparison()),
            flow("return", Some(comparison())),
            param_block(vec![parameter_declaration(
                "$n",
                vec![],
                Some(comparison()),
            )]),
            if_body(vec![comparison()]),
            while_loop(number("1"), script(vec![comparison()])),
            do_loop(script(vec![comparison()]), number("1"), true),
        ] {
            assert_script_refused(
                &script(vec![statement]),
                FieldNotSupported {
                    label: "BinaryExpression",
                    field: "op",
                },
            );
        }
    }

    #[test]
    fn if_is_refused_outside_statement_position() {
        for (root, role) in [
            (if_body(vec![]), "script root"),
            (
                script(vec![assignment(variable("$x"), "=", if_body(vec![]))]),
                "assignment value",
            ),
        ] {
            assert_script_refused(
                &root,
                NodeNotAllowed {
                    role,
                    label: "IfStatement",
                },
            );
        }
    }

    #[test]
    fn if_refuses_leading_param_blocks_in_conditional_arms() {
        for statement in [
            if_body(vec![param_block(vec![])]),
            conditional(
                vec![number("1")],
                vec![script(vec![])],
                Some(script(vec![param_block(vec![])])),
            ),
        ] {
            assert_script_refused(
                &script(vec![statement]),
                NodeNotAllowed {
                    role: "statement",
                    label: "ParamBlock",
                },
            );
        }
    }

    #[test]
    fn function_with_empty_while_body_aligns_braces() {
        let script = script(vec![function(
            "F",
            vec![while_loop(number("1"), script(vec![]))],
        )]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"function F {\n    while (1) {\n    }\n}\n");
    }

    #[test]
    fn while_checks_its_guard_before_requiring_a_script_body() {
        let body = || node(NodeKind::ScriptBlockExpression(Box::new(script(vec![]))));
        for (condition, role, label) in [
            (
                command(bareword("Get"), vec![]),
                "while condition",
                "Command",
            ),
            (number("1"), "while body", "ScriptBlockExpression"),
        ] {
            assert_script_refused(
                &script(vec![while_loop(condition, body())]),
                NodeNotAllowed { role, label },
            );
        }
    }

    #[test]
    fn do_checks_its_script_body_before_its_guard() {
        for (body, role, label) in [
            (
                node(NodeKind::ScriptBlockExpression(Box::new(script(vec![])))),
                "do body",
                "ScriptBlockExpression",
            ),
            (script(vec![param_block(vec![])]), "statement", "ParamBlock"),
            (script(vec![]), "do condition", "Command"),
        ] {
            let condition = command(bareword("Get"), vec![]);

            assert_script_refused(
                &script(vec![do_loop(body, condition, false)]),
                NodeNotAllowed { role, label },
            );
        }
    }

    #[test]
    fn do_with_unsupported_trailing_guard_leaves_writer_untouched() {
        let body = script(vec![assignment(variable("$n"), "=", number("1"))]);
        let condition = command(bareword("Get-Value"), vec![]);
        let script = script(vec![do_loop(body, condition, true)]);
        let mut writer = WriterSpy::new(usize::MAX, usize::MAX);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
        assert_refusal(
            &error,
            &NodeNotAllowed {
                role: "do condition",
                label: "Command",
            },
        );
    }

    #[test]
    fn do_until_write_failure_after_nine_bytes_preserves_prefix_and_error_without_flushing() {
        let statement = do_loop(script(vec![]), variable("$n"), true);
        let script = script(vec![statement]);
        let mut writer = WriterSpy::new(usize::MAX, 9);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.flushes),
            (&b"do {\n} un"[..], 0)
        );
        assert_eq!(
            (
                error.kind(),
                error
                    .get_ref()
                    .and_then(|e| e.downcast_ref::<BrokenPipeSentinel>())
            ),
            (io::ErrorKind::BrokenPipe, Some(&BrokenPipeSentinel(24)))
        );
    }

    #[test]
    fn loop_guards_refuse_unsupported_operators_and_nested_binary_operands() {
        for (statement, expected) in [
            (
                while_loop(binary(bareword("bad"), "-ne", number("1")), script(vec![])),
                FieldNotSupported {
                    label: "BinaryExpression",
                    field: "op",
                },
            ),
            (
                do_loop(
                    script(vec![]),
                    binary(number("1"), "-ｅq", number("2")),
                    true,
                ),
                FieldNotSupported {
                    label: "BinaryExpression",
                    field: "op",
                },
            ),
            (
                while_loop(
                    binary(number("1"), "-eq", binary(number("2"), "+", number("3"))),
                    script(vec![]),
                ),
                NodeNotAllowed {
                    role: "binary operand",
                    label: "BinaryExpression",
                },
            ),
        ] {
            assert_script_refused(&script(vec![statement]), expected);
        }
    }

    #[test]
    fn while_refuses_a_leading_local_param_block() {
        let statement = while_loop(number("1"), script(vec![param_block(vec![])]));

        assert_script_refused(
            &script(vec![statement]),
            NodeNotAllowed {
                role: "statement",
                label: "ParamBlock",
            },
        );
    }

    #[test]
    fn script_refuses_loops_outside_statement_position() {
        for (root, role) in [
            (while_loop(number("1"), script(vec![])), "script root"),
            (
                script(vec![assignment(
                    variable("$x"),
                    "=",
                    do_loop(script(vec![]), number("1"), true),
                )]),
                "assignment value",
            ),
        ] {
            assert_script_refused(
                &root,
                NodeNotAllowed {
                    role,
                    label: "WhileStatement",
                },
            );
        }
    }

    #[test]
    fn loop_bodies_emit_bare_break_and_continue_lines() {
        let script = script(vec![
            while_loop(number("1"), script(vec![flow("break", None)])),
            do_loop(script(vec![flow("continue", None)]), number("0"), true),
        ]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            b"while (1) {\n    break\n}\ndo {\n    continue\n} until (0)\n"
        );
    }

    #[test]
    fn loop_flow_is_allowed_in_all_conditional_arms_and_nested_loops() {
        let arms = conditional(
            vec![number("1"), number("2")],
            vec![
                script(vec![flow("break", None)]),
                script(vec![flow("continue", None)]),
            ],
            Some(script(vec![
                do_loop(script(vec![flow("break", None)]), number("0"), false),
                flow("continue", None),
            ])),
        );
        let script = script(vec![while_loop(number("1"), script(vec![arms]))]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            concat!(
                "while (1) {\n",
                "    if (1) {\n",
                "        break\n",
                "    } elseif (2) {\n",
                "        continue\n",
                "    } else {\n",
                "        do {\n",
                "            break\n",
                "        } while (0)\n",
                "        continue\n",
                "    }\n",
                "}\n",
            )
            .as_bytes()
        );
    }

    #[test]
    fn function_inside_loop_allows_its_own_loop_flow_without_affecting_outer_siblings() {
        let inner = function(
            "Inner",
            vec![while_loop(
                number("2"),
                script(vec![flow("continue", None)]),
            )],
        );
        let script = script(vec![while_loop(
            number("1"),
            script(vec![inner, flow("break", None)]),
        )]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            concat!(
                "while (1) {\n",
                "    function Inner {\n",
                "        while (2) {\n",
                "            continue\n",
                "        }\n",
                "    }\n",
                "    break\n",
                "}\n",
            )
            .as_bytes()
        );
    }

    #[test]
    fn function_inside_loop_refuses_flow_before_and_after_its_own_loop() {
        let own_loop = || while_loop(number("2"), script(vec![flow("break", None)]));
        for body in [
            vec![flow("break", None), own_loop()],
            vec![own_loop(), flow("continue", None)],
        ] {
            let statement = while_loop(number("1"), script(vec![function("Inner", body)]));

            assert_script_refused(
                &script(vec![statement]),
                NodeNotAllowed {
                    role: "statement",
                    label: "FlowStatement",
                },
            );
        }
    }

    #[test]
    fn loop_permission_does_not_leak_to_a_later_conditional_sibling() {
        let script = script(vec![
            while_loop(number("1"), script(vec![flow("break", None)])),
            if_body(vec![flow("continue", None)]),
        ]);

        assert_script_refused(
            &script,
            NodeNotAllowed {
                role: "statement",
                label: "FlowStatement",
            },
        );
    }

    #[test]
    fn loop_flow_with_any_value_is_refused_before_inspecting_the_value() {
        for statement in [
            flow("break", Some(bareword("Outer"))),
            flow("continue", Some(number("bad"))),
        ] {
            let script = script(vec![while_loop(number("1"), script(vec![statement]))]);

            assert_script_refused(
                &script,
                FieldNotSupported {
                    label: "FlowStatement",
                    field: "value",
                },
            );
        }
    }

    #[test]
    fn loop_body_refuses_malformed_keywords_and_other_unsupported_flow() {
        for (keyword, label) in [
            ("BREAK", "FlowStatement"),
            ("continue ", "FlowStatement"),
            ("exit", "FlowStatement"),
            ("throw", "ThrowStatement"),
        ] {
            let script = script(vec![while_loop(
                number("1"),
                script(vec![flow(keyword, None)]),
            )]);

            assert_script_refused(
                &script,
                NodeNotAllowed {
                    role: "statement",
                    label,
                },
            );
        }
    }

    #[test]
    fn loop_body_does_not_allow_flow_in_nested_guards_or_values() {
        for (statement, role) in [
            (
                conditional(vec![flow("continue", None)], vec![script(vec![])], None),
                "if condition",
            ),
            (
                while_loop(flow("break", None), script(vec![])),
                "while condition",
            ),
            (
                do_loop(script(vec![]), flow("continue", None), false),
                "do condition",
            ),
            (
                assignment(variable("$x"), "=", flow("break", None)),
                "assignment value",
            ),
            (flow("return", Some(flow("continue", None))), "return value"),
        ] {
            let script = script(vec![while_loop(number("1"), script(vec![statement]))]);

            assert_script_refused(
                &script,
                NodeNotAllowed {
                    role,
                    label: "FlowStatement",
                },
            );
        }
    }

    #[test]
    fn four_nested_loops_allow_an_indented_ordinary_statement() {
        let while_body = |statement| while_loop(number("1"), script(vec![statement]));
        let inner = do_loop(script(vec![number("02")]), number("0"), true);
        let script = script(vec![while_body(while_body(while_body(inner)))]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            concat!(
                "while (1) {\n",
                "    while (1) {\n",
                "        while (1) {\n",
                "            do {\n",
                "                02\n",
                "            } until (0)\n",
                "        }\n",
                "    }\n",
                "}\n",
            )
            .as_bytes()
        );
    }

    #[test]
    fn mixed_blocks_at_depth_four_keep_nested_else_and_allow_ordinary_statements() {
        let inner = function("Inner", vec![param_block(vec![]), flow("return", None)]);
        let outer_if = conditional(
            vec![variable("$x")],
            vec![script(vec![number("2")])],
            Some(script(vec![if_body(vec![inner])])),
        );
        let script = script(vec![function("Outer", vec![outer_if, number("3")])]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            concat!(
                "function Outer {\n",
                "    if ($x) {\n",
                "        2\n",
                "    } else {\n",
                "        if (1) {\n",
                "            function Inner {\n",
                "                param()\n",
                "                return\n",
                "            }\n",
                "        }\n",
                "    }\n",
                "    3\n",
                "}\n",
            )
            .as_bytes()
        );
    }

    #[test]
    fn four_mixed_blocks_with_loops_allow_an_indented_ordinary_statement() {
        let inner = do_loop(script(vec![number("02")]), number("0"), true);
        let outer = while_loop(number("1"), script(vec![if_body(vec![inner])]));
        let script = script(vec![function("F", vec![outer])]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            concat!(
                "function F {\n",
                "    while (1) {\n",
                "        if (1) {\n",
                "            do {\n",
                "                02\n",
                "            } until (0)\n",
                "        }\n",
                "    }\n",
                "}\n",
            )
            .as_bytes()
        );
    }

    #[test]
    fn script_refuses_a_fifth_block_before_inspecting_its_fields() {
        for (statement, label) in [
            (conditional(vec![], vec![], None), "IfStatement"),
            (
                function_definition("", true, vec![], number("1")),
                "FunctionDefinition",
            ),
            (while_loop(bareword("bad"), number("1")), "WhileStatement"),
            (
                do_loop(number("1"), bareword("bad"), true),
                "WhileStatement",
            ),
        ] {
            let inner = do_loop(script(vec![statement]), number("0"), true);
            let outer = while_loop(number("1"), script(vec![if_body(vec![inner])]));
            let root = script(vec![function("Outer", vec![outer])]);

            assert_script_refused(
                &root,
                NodeNotAllowed {
                    role: "statement inside four enclosing blocks",
                    label,
                },
            );
        }
    }

    #[test]
    fn script_refuses_third_function_across_conditionals_even_at_the_block_limit() {
        let third = || function_definition("", true, vec![], number("1"));
        for statement in [third(), if_body(vec![third()])] {
            let outer = function(
                "Outer",
                vec![if_body(vec![function("Inner", vec![statement])])],
            );

            assert_script_refused(
                &script(vec![outer]),
                NodeNotAllowed {
                    role: "statement inside two enclosing functions",
                    label: "FunctionDefinition",
                },
            );
        }
    }

    #[test]
    fn script_refuses_third_function_across_loops_even_at_the_block_limit() {
        let third = || function_definition("", true, vec![], number("1"));
        for statement in [third(), do_loop(script(vec![third()]), number("0"), true)] {
            let inner = function("Inner", vec![statement]);
            let loop_statement = while_loop(number("1"), script(vec![inner]));
            let root = script(vec![function("Outer", vec![loop_statement])]);

            assert_script_refused(
                &root,
                NodeNotAllowed {
                    role: "statement inside two enclosing functions",
                    label: "FunctionDefinition",
                },
            );
        }
    }

    #[test]
    fn supported_script_emits_literal_bytes_in_ast_order() {
        let script = script(vec![
            variable("$_a9"),
            number("000"),
            string("''"),
            string("''''"),
            binary(number("42"), "-", number("7")),
            binary(variable("$Z"), "*", number("0")),
            assignment(
                variable("$x"),
                "=",
                binary(string("'café''s'"), "+", string("'字'")),
            ),
            command(bareword("_Read-0"), vec![]),
            node(NodeKind::Pipeline(vec![
                command(
                    bareword("Read-Item"),
                    vec![
                        parameter("Flag", None),
                        bareword("007"),
                        parameter("_Input-1", Some(string("'café'"))),
                        bareword("Tail-"),
                        variable("$x"),
                        number("0"),
                    ],
                ),
                command(
                    bareword("Sort-Item"),
                    vec![parameter("Key", Some(bareword("9")))],
                ),
                command(bareword("Show"), vec![]),
            ])),
        ]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(
            bytes,
            "$_a9\n000\n''\n''''\n42 - 7\n$Z * 0\n$x = 'café''s' + '字'\n_Read-0\n\
             Read-Item -Flag 007 -_Input-1 'café' Tail- $x 0 | Sort-Item -Key 9 | Show\n"
                .as_bytes()
        );
    }

    #[test]
    fn malformed_coordinates_do_not_affect_output() {
        let mut statement = number("0009");
        statement.span = Span {
            start: usize::MAX,
            end: 0,
        };
        statement.range = TokenRange {
            first: usize::MAX,
            end: 0,
        };
        let mut script = script(vec![statement]);
        script.span = Span {
            start: 7,
            end: usize::MAX,
        };
        script.range = TokenRange {
            first: 7,
            end: usize::MAX,
        };
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"0009\n");
    }

    #[test]
    fn two_command_pipeline_emits_one_separator() {
        let script = script(vec![node(NodeKind::Pipeline(vec![
            command(bareword("Read"), vec![]),
            command(bareword("Show"), vec![]),
        ]))]);
        let mut bytes = Vec::new();

        write_diagnostic_script(&script, &mut bytes).unwrap();

        assert_eq!(bytes, b"Read | Show\n");
    }

    #[test]
    fn short_writes_to_trait_object_complete_without_flushing() {
        let script = simple_assignment();
        let mut writer = WriterSpy::new(1, usize::MAX);
        let destination: &mut dyn Write = &mut writer;

        write_diagnostic_script(&script, destination).unwrap();

        assert_eq!(
            (writer.bytes.as_slice(), writer.flushes),
            (&b"$x = 1\n"[..], 0)
        );
    }

    #[test]
    fn first_write_failure_preserves_error_without_output_or_flush() {
        let script = simple_assignment();
        let mut writer = WriterSpy::new(usize::MAX, 0);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 1, 0)
        );
        assert_eq!(
            (
                error.kind(),
                error
                    .get_ref()
                    .and_then(|e| e.downcast_ref::<BrokenPipeSentinel>())
            ),
            (io::ErrorKind::BrokenPipe, Some(&BrokenPipeSentinel(24)))
        );
    }

    #[test]
    fn failure_after_three_bytes_preserves_prefix_and_error_without_flushing() {
        let script = simple_assignment();
        let mut writer = WriterSpy::new(usize::MAX, 3);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!((writer.bytes.as_slice(), writer.flushes), (&b"$x "[..], 0));
        assert_eq!(
            (
                error.kind(),
                error
                    .get_ref()
                    .and_then(|e| e.downcast_ref::<BrokenPipeSentinel>())
            ),
            (io::ErrorKind::BrokenPipe, Some(&BrokenPipeSentinel(24)))
        );
    }

    #[test]
    fn unsupported_final_statement_returns_refusal_without_touching_writer() {
        let script = script(vec![
            number("1"),
            node(NodeKind::Error("incomplete".into())),
        ]);
        let mut writer = WriterSpy::new(usize::MAX, 0);

        let error = write_diagnostic_script(&script, &mut writer).unwrap_err();

        assert_eq!(
            (writer.bytes.as_slice(), writer.writes, writer.flushes),
            (&b""[..], 0, 0)
        );
        assert_refusal(
            &error,
            &NodeNotAllowed {
                role: "statement",
                label: "ErrorNode",
            },
        );
    }

    #[track_caller]
    fn assert_refusal(error: &io::Error, expected: &DiagnosticRenderError) {
        assert_eq!(
            (
                error.kind(),
                error
                    .get_ref()
                    .and_then(|err| err.downcast_ref::<DiagnosticRenderError>())
            ),
            (io::ErrorKind::InvalidInput, Some(expected))
        );
    }

    #[track_caller]
    fn assert_script_refused(script: &Node, expected: DiagnosticRenderError) {
        let error = write_diagnostic_script(script, &mut Vec::new())
            .expect_err("AST is outside the diagnostic grammar");
        assert_refusal(&error, &expected);
    }

    #[test]
    fn script_refuses_non_script_root() {
        assert_script_refused(
            &number("1"),
            NodeNotAllowed {
                role: "script root",
                label: "NumberLiteral",
            },
        );
    }

    #[test]
    fn function_is_refused_outside_statement_position() {
        for (root, role) in [
            (function("F", vec![]), "script root"),
            (
                script(vec![assignment(variable("$x"), "=", function("F", vec![]))]),
                "assignment value",
            ),
            (
                script(vec![command(bareword("Get"), vec![function("F", vec![])])]),
                "command argument",
            ),
            (
                script(vec![node(NodeKind::Pipeline(vec![
                    command(bareword("Get"), vec![]),
                    function("F", vec![]),
                ]))]),
                "pipeline element",
            ),
        ] {
            assert_script_refused(
                &root,
                NodeNotAllowed {
                    role,
                    label: "FunctionDefinition",
                },
            );
        }
    }

    #[test]
    fn script_refuses_nodes_in_disallowed_roles() {
        for (statement, role, label) in [
            (script(vec![number("0")]), "statement", "ScriptBlock"),
            (bareword("Name"), "statement", "BareWord"),
            (
                assignment(number("1"), "=", number("2")),
                "assignment target",
                "NumberLiteral",
            ),
            (
                assignment(variable("$x"), "=", command(bareword("Get"), vec![])),
                "assignment value",
                "Command",
            ),
            (
                binary(number("1"), "+", binary(number("2"), "*", number("3"))),
                "binary operand",
                "BinaryExpression",
            ),
            (
                command(variable("$name"), vec![]),
                "command name",
                "Variable",
            ),
            (
                command(bareword("Get"), vec![binary(number("1"), "+", number("2"))]),
                "command argument",
                "BinaryExpression",
            ),
            (
                command(
                    bareword("Get"),
                    vec![parameter("Outer", Some(parameter("Inner", None)))],
                ),
                "command argument",
                "CommandParameter",
            ),
            (
                node(NodeKind::Pipeline(vec![
                    command(bareword("Get"), vec![]),
                    number("2"),
                ])),
                "pipeline element",
                "NumberLiteral",
            ),
        ] {
            assert_script_refused(&script(vec![statement]), NodeNotAllowed { role, label });
        }
    }

    #[test]
    fn script_refuses_unsupported_operators_and_short_pipelines() {
        for (statement, label, field) in [
            (
                assignment(variable("$x"), "+=", number("1")),
                "AssignmentStatement",
                "op",
            ),
            (
                binary(number("1"), "/", number("2")),
                "BinaryExpression",
                "op",
            ),
            (node(NodeKind::Pipeline(vec![])), "Pipeline", "elements"),
            (
                node(NodeKind::Pipeline(vec![number("1")])),
                "Pipeline",
                "elements",
            ),
        ] {
            assert_script_refused(&script(vec![statement]), FieldNotSupported { label, field });
        }
    }

    #[test]
    fn script_refuses_command_fields_before_visiting_children() {
        for (invocation, redirections, csharp, field) in [
            (true, vec![], None, "invocation"),
            (
                false,
                vec![node(NodeKind::Redirection {
                    op: ">".into(),
                    target: None,
                })],
                None,
                "redirections",
            ),
            (
                false,
                vec![],
                Some(Box::new(node(NodeKind::CSharpMemberDef(
                    CSharpMemberDef::default(),
                )))),
                "csharp",
            ),
        ] {
            let statement = node(NodeKind::Command {
                name: Box::new(node(NodeKind::Error("name".into()))),
                invocation,
                elements: Vec::new(),
                redirections,
                csharp,
            });
            assert_script_refused(
                &script(vec![statement]),
                FieldNotSupported {
                    label: "Command",
                    field,
                },
            );
        }
    }

    #[test]
    fn script_refuses_invalid_unquoted_leaf_text() {
        for (statement, label, field) in [
            (variable(""), "Variable", "value"),
            (variable("$"), "Variable", "value"),
            (variable("$1x"), "Variable", "value"),
            (variable("$user-name"), "Variable", "value"),
            (variable("$env:PATH"), "Variable", "value"),
            (variable("${x}"), "Variable", "value"),
            (variable("$café"), "Variable", "value"),
            (variable("@args"), "Variable", "value"),
            (number(""), "NumberLiteral", "value"),
            (number("-1"), "NumberLiteral", "value"),
            (number("1.0"), "NumberLiteral", "value"),
            (number("１"), "NumberLiteral", "value"),
            (command(bareword("01"), vec![]), "BareWord", "value"),
            (command(bareword("-Get"), vec![]), "BareWord", "value"),
            (
                command(bareword("Get"), vec![bareword("1a")]),
                "BareWord",
                "value",
            ),
            (
                command(bareword("Get"), vec![bareword("a/b")]),
                "BareWord",
                "value",
            ),
            (
                command(bareword("Get"), vec![parameter("Flag:", None)]),
                "CommandParameter",
                "name",
            ),
            (
                command(bareword("Get"), vec![parameter("9", None)]),
                "CommandParameter",
                "name",
            ),
        ] {
            assert_script_refused(&script(vec![statement]), FieldNotSupported { label, field });
        }
    }

    #[test]
    fn script_refuses_strings_with_malformed_quotes_or_ascii_controls() {
        for value in [
            "", "'", "'a", "a'", "‘a’", "'''", "'a'b'", "'a''b'c'", "'\0'", "'\t'", "'\r\n'",
            "'\x7f'",
        ] {
            assert_script_refused(
                &script(vec![string(value)]),
                FieldNotSupported {
                    label: "StringLiteral",
                    field: "value",
                },
            );
        }
    }

    #[test]
    fn script_refuses_non_single_string_kinds_and_nonempty_parts() {
        for (literal, field) in [
            (string_literal(StringKind::Double, "\"x\"", vec![]), "kind"),
            (
                string_literal(StringKind::HereSingle, "'x'", vec![]),
                "kind",
            ),
            (
                string_literal(StringKind::HereDouble, "'x'", vec![]),
                "kind",
            ),
            (
                string_literal(
                    StringKind::Single,
                    "'x'",
                    vec![node(NodeKind::Error("part".into()))],
                ),
                "parts",
            ),
        ] {
            assert_script_refused(
                &script(vec![literal]),
                FieldNotSupported {
                    label: "StringLiteral",
                    field,
                },
            );
        }
    }
}
