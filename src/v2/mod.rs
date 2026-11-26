//! `v2`: a lossless, trivia-bearing token layer.
//!
//! # Why a v2 exists
//!
//! The `v1` pipeline is built for analysis: its lexer emits
//! `Newline` and `Comment` tokens but drops plain whitespace, and
//! its `unparse` regenerates spacing from scratch. That is the
//! right trade for linting and metrics, but formatting and targeted edits
//! need the original layout:
//!
//! * a **formatter** must read the author's comments and line structure to
//!   place them in the output, and must check the structure of its result;
//! * a **codemod** (e.g. `Get-WmiObject` to `Get-CimInstance`) must rewrite
//!   one command and leave every other byte alone, or the diff is unusable.
//!
//! Both need the same foundation: tokens that own every byte of the input.
//! This different token shape ships as a sibling `v2` module so the `v1`
//! design remains available. Sibling namespaces do not rule out justified
//! crate-level breaking releases: during `0.x` development, these use a
//! minor-version bump with migration guidance.
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
//! `trailing` trivia, in source order. [`reconstruct`] is the lossless
//! unparse. This exact reconstruction of retained tokens is distinct from
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
//! **The optional v1 bridge is deprecated in 0.5 documentation only.**
//! Both `poshtree::v2::tree::parse_with_tokens` and its `poshtree::v2`
//! re-export, along with `Tree` and `TreeNode`, remain repair-only throughout
//! 0.5 and require `v1` in addition to `v2`. The v1 bridge and its `v2`
//! re-exports will not be removed before 0.6.0.
//!
//! The bridge pairs the v1 AST with v2 tokens by recovering ranges from
//! node offsets. Its node footprints include attached trivia;
//! interpolation children inside a string token have empty ranges because
//! the enclosing token retains their bytes. The bridge keeps v1's
//! BOM-stripped node positions distinct from original-input token spans.
//! It offers untyped walks (`TreeNode::walk`, `walk_with_ancestors`) and a
//! zipped walk (`Tree::walk_zipped`) pairing typed v1 nodes with their
//! range-bearing mirrors and ancestor paths. For native traversal use
//! [`parse`] and [`Node::walk_with_ancestors`]; these do not reproduce the
//! zipped v1 nodes, scalar summaries, or trivia-inclusive bridge footprints.
//!
//! For diagnostic text from a supported native AST, including an edited tree,
//! use [`write_diagnostic_script`]. It is not the legacy error-tolerant
//! source dump. Use [`reconstruct`] for exact token replay, [`format_source`]
//! for checked source formatting, or [`apply_edits`] for targeted changes
//! against the original bytes.
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
//! `tree` reuses the v1 parser and recovers ranges from node offsets, so it
//! needs the `v1` feature. `parser::parse` is the standalone path: a native
//! recursive-descent parser that consumes v2 tokens and builds an `ast::Node`
//! tree directly, with each node carrying both a byte `Span` and a
//! `TokenRange`. This native parser already exists and depends on no `v1`
//! code, so it builds under `v2` alone.
//! Because v2 keeps newlines as trivia, statement boundaries come from
//! `Token::starts_line` and `;`.
//!
//! The grammar tracks v1's: pipelines and `&&`/`||` chains, command-versus-
//! expression dispatch with parameter-argument binding and redirections, every
//! control-flow statement, `function`/`filter`/`workflow`, `class`, `enum`,
//! `using`, `trap`/`data`/`dynamicparam`, `param` blocks, and the full
//! expression layer. A differential test reduces both trees to a label
//! skeleton and asserts the native parser matches the v1 tree shape across a
//! broad corpus, including double-quoted string interpolation parts (`$var`,
//! `${name}`, `$(...)`) and `Add-Type` C# extraction (`[DllImport]` P/Invoke
//! parsing, with constant propagation of a string assigned to a variable). A
//! companion test compares selected metadata fields, not complete extraction
//! parity. Without `csharp`, native `Add-Type` import extraction uses a fallback
//! scanner. Enabling `csharp` uses bounded parsed extraction with richer type
//! spellings; neither is a drop-in replacement for the v1 extractor.
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
//! # What changed against v1, concretely
//!
//! * v1 emits `Newline` and `Comment` as tokens; v2 has neither. A line
//!   break or comment rides as [`Trivia`] on a neighboring token, so the
//!   significant stream is free of layout noise.
//! * v2 keeps the plain whitespace v1 drops as [`TriviaKind::Whitespace`];
//!   that is what makes reconstruction lossless.
//! * Instead of v1's start offset plus line/column per token, v2 stores
//!   full byte [`Span`]s and derives line/column on demand through
//!   [`LineIndex`].
//! * v1 decodes `text`, `scope`, and `splat` at lex time; v2 keeps only the
//!   raw `value`, so a token is a faithful slice of the source and decoding
//!   is the parser's job.
//! * After `--%`, v1's parser re-slices the raw source; v2's lexer emits
//!   the operator and one raw [`TokenKind::VerbatimArgs`] token.
//! * A v1 rewrite reprints the whole tree; v2 patches byte spans with
//!   [`TextEdit`] and [`apply_edits`], so a change shows up as a minimal
//!   diff.
//!
//! Classification agrees with v1 by test rather than by import: v2 carries
//! its own copies of the [`KEYWORDS`](tokens::KEYWORDS) and
//! [`NAMED_OPERATORS`](tokens::NAMED_OPERATORS) tables, and an integration
//! test keeps them equal to v1's, so the two lexers decide `Keyword` vs
//! `Generic` and `Operator` vs `Parameter` identically. The
//! `?.`/`?[`/`??`/`??=` operators and the `.5`-vs-member-access rule also
//! match v1. The lexer, spans, trivia, and edits therefore compile without
//! v1 (build with `default-features = false, features = ["v2"]`); the one
//! v2 component that does use v1 is `tree`, which by design runs the v1
//! parser and is only compiled when both features are on.
//!
//! # Where v2 lexing deliberately differs from v1
//!
//! Each of these favors the formatter/codemod use case; all are kind/shape
//! differences only, never byte loss:
//!
//! * **Cohesive barewords.** `C:\tmp`, `*.txt`, `/usr/bin/env`, `user@host`
//!   are one `Generic` each, where v1 fragments them (a Windows path lexes
//!   as four v1 tokens including an `Unknown` for `\`). The v1 parser
//!   already glues fragments back by byte adjacency; v2 just does less
//!   fragmenting up front. Where v2 still splits (`.\run.ps1`, `a=b`,
//!   `192.168.1.1`), adjacent spans carry the same glue signal.
//! * **`-Path:` keeps its colon** as one `Parameter` token; v1 emits
//!   `-Path` plus a glued `:` operator the parser re-joins by position.
//! * **`--%` is handled in the lexer.** v1 lexes the rest of the line
//!   normally (comments and all) and the parser re-reads the raw source;
//!   v2 emits the `--%` operator and a raw [`TokenKind::VerbatimArgs`].
//! * **Richer literals and operators.** Binary `0b1010`, the PowerShell 7
//!   numeric suffixes (`u`, `ul`, `n`, ...), `-=` as one token, and `!` as
//!   an `Operator` (v1: `Unknown`).
//! * **Unicode identifiers.** `$żółć` is one `Variable`; v1's name scan is
//!   ASCII-only.
//! * **A UTF-8 BOM becomes whitespace trivia** instead of being stripped,
//!   so reconstruction stays exact ([`lex`] never calls
//!   [`strip_bom`](crate::encoding::strip_bom)).
//!
//! # Porting parser code from v1
//!
//! `tok.ty` becomes `tok.kind`, `tok.value` stays `tok.value`. Checks
//! against `TokenType::Newline` become [`Token::starts_line`]. Code that
//! compared `tok.pos` arithmetic to detect glued tokens compares
//! `a.span.end == b.span.start` instead. Decoded payloads (`text`,
//! `scope`, `splat`) are derived from `value` on demand.
//! Comment trivia can locate native comment bytes for caller-supplied masking;
//! it does not reproduce v1's comment recognition or byte-preserving mask.

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
#[cfg(feature = "v1")]
pub mod tree;
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
#[cfg(feature = "v1")]
pub use tree::{parse_with_tokens, Tree, TreeNode};
pub use trivia::{Trivia, TriviaKind};
pub use variables::{
    is_automatic_variable, is_preference_variable, variable_name, variable_scope,
    AUTOMATIC_VARIABLES, PREFERENCE_VARIABLES,
};

/// Concatenates `leading + value + trailing` over all tokens: the lossless
/// unparse. Applied to the output of [`lex`], this reproduces the original
/// source byte-for-byte.
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
