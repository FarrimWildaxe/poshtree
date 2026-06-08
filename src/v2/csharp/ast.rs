//! C# abstract syntax tree for the Add-Type dialect.
//!
//! One uniform node type, [`CsNode`], carries a kind, a span, and children.
//! The structure is deliberately shallow: it models the declarations, scopes,
//! and identifier references that renaming needs, and treats expression detail
//! generically. Anything outside the dialect parses into an
//! [`Error`](CsNodeKind::Error) node rather than failing.
//!
//! Every renameable identifier (a declaration name or a reference) is a
//! [`CsName`] with its own precise span. The PowerShell AST has no such span
//! for member and type names, which is what makes correct C# renaming possible.

use crate::v2::span::Span;

/// An identifier occurrence: its text and exact span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsName {
    /// The identifier text.
    pub text: String,
    /// Its span in the original file.
    pub span: Span,
}

/// What kind of C# type a declaration introduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsTypeKind {
    /// `class`
    Class,
    /// `struct`
    Struct,
    /// `interface`
    Interface,
    /// `enum`
    Enum,
}

/// The role of a [`CsNode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsNodeKind {
    /// The compilation unit (the whole C# region). Root scope.
    Unit,
    /// `using X.Y;`, recorded but not otherwise modeled.
    Using,
    /// `namespace N { ... }`. A scope.
    Namespace(CsName),
    /// A type declaration. A scope; its members are children.
    Type {
        /// class / struct / interface / enum.
        kind: CsTypeKind,
        /// The declared type name.
        name: CsName,
    },
    /// A method declaration. A scope; parameters and body are children.
    Method(CsName),
    /// A constructor declaration. A scope.
    Ctor(CsName),
    /// A property declaration. A scope (accessor bodies are children).
    Property(CsName),
    /// An `enum` member.
    EnumMember(CsName),
    /// A parameter declaration (the parameter name).
    Param(CsName),
    /// A field or local declaration group; each declared name is a
    /// [`NameDecl`](CsNodeKind::NameDecl) child.
    Decl,
    /// A single declared name within a field or local declaration.
    NameDecl(CsName),
    /// A `{ ... }` block. A scope.
    Block,
    /// An identifier reference. `after_dot` marks names following `.` or `::`
    /// (member access), which bind differently from root identifiers.
    NameRef {
        /// The referenced identifier.
        name: CsName,
        /// Whether it directly follows `.` or `::`.
        after_dot: bool,
        /// For a member access (`after_dot`), the receiver it is accessed
        /// through, when that receiver is a bare name: `this`, `base`, an
        /// identifier, or a type segment. `None` for a root reference, and also
        /// for a member access whose receiver is not a bare name (for example
        /// `foo().Bar`). Lets resolution tell `this.Field` and `Type.Member`
        /// apart from `unrelated.Field`.
        receiver: Option<String>,
    },
    /// An attribute application, e.g. `[DllImport(...)]`.
    Attribute(CsName),
    /// A region the parser could not match; spans the tokens it recovered over.
    Error,
}

/// A C# syntax node: a kind, its span, and ordered children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsNode {
    /// The node's role.
    pub kind: CsNodeKind,
    /// The node's span in the original file.
    pub span: Span,
    /// Child nodes, in source order.
    pub children: Vec<CsNode>,
}

impl CsNode {
    /// Creates a node with the given kind, span, and children.
    pub fn new(kind: CsNodeKind, span: Span, children: Vec<CsNode>) -> Self {
        CsNode {
            kind,
            span,
            children,
        }
    }

    /// A leaf node (no children).
    pub fn leaf(kind: CsNodeKind, span: Span) -> Self {
        CsNode {
            kind,
            span,
            children: Vec::new(),
        }
    }

    /// The declared name this node introduces, if it is a declaration.
    pub fn declared_name(&self) -> Option<&CsName> {
        match &self.kind {
            CsNodeKind::Namespace(n)
            | CsNodeKind::Type { name: n, .. }
            | CsNodeKind::Method(n)
            | CsNodeKind::Ctor(n)
            | CsNodeKind::Property(n)
            | CsNodeKind::EnumMember(n)
            | CsNodeKind::Param(n)
            | CsNodeKind::NameDecl(n) => Some(n),
            _ => None,
        }
    }

    /// Whether this node opens a new lexical scope.
    pub fn is_scope(&self) -> bool {
        matches!(
            self.kind,
            CsNodeKind::Unit
                | CsNodeKind::Namespace(_)
                | CsNodeKind::Type { .. }
                | CsNodeKind::Method(_)
                | CsNodeKind::Ctor(_)
                | CsNodeKind::Property(_)
                | CsNodeKind::Block
        )
    }

    /// Visits this node and every descendant, pre-order.
    pub fn walk(&self, f: &mut impl FnMut(&CsNode)) {
        f(self);
        for c in &self.children {
            c.walk(f);
        }
    }
}

/// The parsed C# region: the unit root plus any recovery diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsUnit {
    /// The root [`Unit`](CsNodeKind::Unit) node.
    pub root: CsNode,
    /// Spans the parser had to recover over (one per [`Error`](CsNodeKind::Error)).
    pub errors: Vec<Span>,
}
