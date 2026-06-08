//! C#-aware support for `Add-Type` blocks, behind the `csharp` feature.
//!
//! `Add-Type` embeds C# in a PowerShell string. PowerShell tools can see the
//! string but not its structure; this module adds a self-contained C# front
//! end (lexer, parser, name resolver) over the exact source region the C#
//! occupies, so a C# type or member can be found, its references resolved, and
//! the matching PowerShell call sites kept in sync.
//!
//! Scope is a single file: the C# in the PowerShell script being processed.
//! There is no assembly or reference resolution, and BCL types are treated as
//! unresolved externals. Constructs outside the supported dialect degrade to
//! error nodes rather than failing the parse, the same way the PowerShell
//! parser recovers.
//!
//! # Step 1: locating the code
//!
//! [`csharp_code_span`] returns the precise source span of the C# body, which
//! is the region the lexer will run over. The mapping is exact for the normal
//! single-quoted here-string form (`@' ... '@`), since that text is verbatim.

use super::ast::{Node, NodeKind};
use super::span::Span;

pub mod ast;
pub mod imports;
pub mod lexer;
pub mod parser;
pub mod refactor;
pub mod resolve;
pub mod tokens;
pub mod xlang;

pub use imports::{csharp_imports, csharp_imports_and_apis};
pub use refactor::{
    add_csharp_comment, csharp_references, csharp_symbols, csharp_unit, rename_csharp_field,
    rename_csharp_local, rename_csharp_method, rename_csharp_parameter, rename_csharp_symbol,
};
pub use xlang::{rename_member, rename_type};

/// The source span of the C# body carried by an `Add-Type` block.
///
/// Accepts either the `Add-Type` [`Command`](NodeKind::Command) node or the
/// [`CSharpMemberDef`](NodeKind::CSharpMemberDef) node directly, and returns
/// `None` when the node carries no extracted C#. The span covers the raw bytes
/// between the string delimiters, so `span.slice(src)` is the C# source and any
/// offset found within it maps straight back to the original file.
pub fn csharp_code_span(node: &Node) -> Option<Span> {
    match &node.kind {
        NodeKind::CSharpMemberDef(def) => Some(def.code_span),
        NodeKind::Command {
            csharp: Some(cs), ..
        } => match &cs.kind {
            NodeKind::CSharpMemberDef(def) => Some(def.code_span),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::v2::csharp::csharp_code_span;
    use crate::v2::{parse, NodeKind};

    /// Find the first CSharpMemberDef node in a parse.
    fn add_type_node(src: &str) -> crate::v2::Node {
        let out = parse(src);
        let mut found = None;
        out.script.walk(&mut |n| {
            if matches!(n.kind, NodeKind::CSharpMemberDef(_)) && found.is_none() {
                found = Some(n.clone());
            }
        });
        found.expect("expected a CSharpMemberDef")
    }

    #[test]
    fn code_span_points_at_inline_here_string_body() {
        let src = "Add-Type -TypeDefinition @'\npublic class A { }\n'@\n";
        let node = add_type_node(src);
        let span = csharp_code_span(&node).expect("code span");
        assert_eq!(span.slice(src), "public class A { }");
    }

    #[test]
    fn code_span_points_at_single_quoted_body() {
        let src = "Add-Type -MemberDefinition 'public static int N() { return 1; }' -Name X\n";
        let node = add_type_node(src);
        let span = csharp_code_span(&node).expect("code span");
        assert_eq!(span.slice(src), "public static int N() { return 1; }");
    }

    #[test]
    fn code_span_follows_a_variable_assignment() {
        let src = "$code = @'\npublic class B { }\n'@\nAdd-Type -TypeDefinition $code\n";
        let node = add_type_node(src);
        let span = csharp_code_span(&node).expect("code span");
        assert_eq!(span.slice(src), "public class B { }");
    }
}
