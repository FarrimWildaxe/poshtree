//! `v2`: lossless PowerShell tokens, a native parser, and source edits.
//!
//! # Why retain trivia?
//!
//! Formatting and targeted edits both need the original layout:
//!
//! * a **formatter** must read the author's comments and line structure to
//!   place them in the output, and must check the structure of its result;
//! * a **codemod** (e.g. `Get-WmiObject` to `Get-CimInstance`) must rewrite
//!   one command and leave every other byte alone, or the diff is unusable.
//!
//! Both need the same foundation: tokens that account for every byte of the input.
//!
//! # The invariant
//!
//! For any input `src`, including malformed input:
//!
//! ```
//! use poshtree::v2::{lex, reconstruct};
//!
//! let src = "ls -la # list\r\n@'\n raw '@ body\n'@\n";
//! assert_eq!(reconstruct(&lex(src).tokens), src);
//! ```
//!
//! Every byte lands in exactly one token's `leading` trivia, `value`, or
//! `trailing` trivia, in source order. [`reconstruct`] replays those bytes
//! exactly. This reconstruction of retained tokens is distinct from
//! formatter acceptance, which checks a rewritten result rather than
//! reproducing the original bytes.
//!
//! # From tokens to a tree
//!
//! Tokens alone support targeted edits (find a token, patch its span).
//! [`parse`] and [`parse_tokens`] provide the native v2 syntax tree:
//! each [`Node`] carries a byte [`Span`] and a [`TokenRange`] into the
//! returned token vector. This parser works with `v2` alone.
//!
//! Use [`Node::walk`] or [`Node::walk_with_ancestors`] to traverse the tree.
//! The legacy token-correlation bridge was removed in 0.6.0. Native traversal
//! uses native node types and child order; it does not reproduce the bridge's
//! zipped typed traversal, scalar summaries, or trivia-inclusive footprints.
//!
//! For diagnostic text from a supported native AST, including an edited tree,
//! use [`write_diagnostic_script`]. It refuses unsupported trees before writing;
//! an I/O error can leave partial output. Use [`reconstruct`] for exact token
//! replay, [`format_source`] for checked source formatting, or [`apply_edits`]
//! for targeted changes against the original bytes.
//!
//! # Formatting
//!
//! [`formatter::format_source`] is a width-aware formatter built on these
//! tokens and verified with the native parser, so it works under `v2`
//! alone. It normalizes indentation, spacing, blank lines, and over-long
//! lines, then re-lexes and re-parses its output with v2. Acceptance requires
//! clean output, identical significant token kinds and text, and recursively
//! matching node labels and child shape. These checks are not a proof of
//! semantic equivalence or full PowerShell correctness. A failed check
//! returns [`FormatError::Unsafe`] without the candidate output; malformed
//! input returns [`FormatError::Syntax`]. See the `formatter` module docs.
//!
//! # A native parser
//!
//! The recursive-descent parser consumes v2 tokens and builds an [`ast::Node`]
//! tree directly. Because the lexer keeps newlines as trivia, statement
//! boundaries come from [`Token::starts_line`] and `;`.
//!
//! The supported grammar includes pipelines and `&&`/`||` chains, command-versus-
//! expression dispatch with parameter-argument binding and redirections,
//! control-flow constructs, `function`/`filter`/`workflow`, `class`, `enum`,
//! `using`, `trap`/`data`/`dynamicparam`, `param` blocks, and expressions,
//! including double-quoted string interpolation (`$var`, `${name}`, `$(...)`).
//! `Add-Type` C# extraction can follow supported string assignments to variables.
//! Without `csharp`, import extraction uses a fallback scanner. Enabling
//! `csharp` uses bounded parsed extraction with richer type spellings.
//!
//! ```
//! use poshtree::v2::{parse, NodeKind};
//!
//! let out = parse("Get-ChildItem -Recurse | Sort-Object Length\n");
//! assert!(out.errors.is_empty());
//! let mut commands = Vec::new();
//! out.script.walk(&mut |n| {
//!     if let NodeKind::Command { name, .. } = &n.kind {
//!         if let NodeKind::BareWord(s) = &name.kind {
//!             commands.push(s.clone());
//!         }
//!     }
//! });
//! assert_eq!(commands, ["Get-ChildItem", "Sort-Object"]);
//! ```
//!
//! # Token kinds and coordinates
//!
//! * Line breaks, comments, and whitespace ride as [`Trivia`] on
//!   neighboring tokens, separate from the significant stream.
//! * Tokens carry full byte [`Span`]s. [`LineIndex`] derives line/column on
//!   demand. Adjacent spans (`a.span.end == b.span.start`) show when tokens
//!   were glued in the source.
//! * A token's `value` is a raw source slice; decoding is the parser's job.
//!
//! The [`KEYWORDS`](tokens::KEYWORDS) and
//! [`NAMED_OPERATORS`](tokens::NAMED_OPERATORS) tables support classification
//! as `Keyword` vs `Generic` and `Operator` vs `Parameter`.
//!
//! # Lexing details
//!
//! * **Cohesive barewords.** `C:\tmp`, `*.txt`, `/usr/bin/env`, `user@host`
//!   are one `Generic` each. Where the lexer still splits (`.\run.ps1`,
//!   `a=b`, `192.168.1.1`), adjacent spans carry the glue signal.
//! * **`-Path:` keeps its colon** as one `Parameter` token.
//! * **`--%` is handled in the lexer**, which emits the operator and a raw
//!   [`TokenKind::VerbatimArgs`].
//! * **Literals and operators** include binary `0b1010`, PowerShell 7 numeric
//!   suffixes (`u`, `ul`, `n`, ...), and operators such as `-=`, `!`, `?.`,
//!   `?[`, `??`, and `??=`. The lexer uses preceding tokens to distinguish
//!   a decimal such as `.5` from member access.
//! * **Unicode identifiers.** `$żółć` is one `Variable`.
//! * **A UTF-8 BOM becomes whitespace trivia**, so reconstruction stays exact
//!   ([`lex`] never calls [`strip_bom`](crate::encoding::strip_bom)).

#[cfg(test)]
mod analysis;
pub mod ast;
#[cfg(feature = "csharp")]
pub mod csharp;
pub mod edit;
pub mod formatter;
pub mod lexer;
pub mod parser;
mod render;
#[cfg(test)]
mod resource;
pub mod span;
pub mod tokens;
pub mod trivia;
pub mod variables;

pub use ast::{CSharpImport, CSharpMemberDef, CSharpParam, Node, NodeKind, StringKind};
pub use edit::{apply_edits, EditError, TextEdit};
pub use formatter::{format_source, format_source_with, FormatError, FormatOptions};
pub use lexer::{lex, LexOutput};
pub use parser::{parse, parse_tokens, ParseError, ParseOutput};
pub use render::{write_diagnostic_script, DiagnosticRenderError};
pub use span::{LineCol, LineIndex, Span, TokenRange};
pub use tokens::{LexError, Token, TokenKind};
pub use trivia::{Trivia, TriviaKind};
pub use variables::{
    is_automatic_variable, is_preference_variable, variable_name, variable_scope,
    AUTOMATIC_VARIABLES, PREFERENCE_VARIABLES,
};

/// Concatenates `leading + value + trailing` over all tokens. Applied to the
/// output of [`lex`], this reproduces the original source byte-for-byte.
pub fn reconstruct(tokens: &[Token]) -> String {
    let mut out = String::new();
    for token in tokens {
        token.write_full(&mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_is_exact() {
        let src = "  if ($x) { ls }  # done\n";
        assert_eq!(reconstruct(&lex(src).tokens), src);
    }
}
