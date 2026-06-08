//! `v2`: a lossless, trivia-bearing token layer.
//!
//! # Why a v2 exists
//!
//! The `v1` pipeline is built for analysis: its lexer emits
//! `Newline` and `Comment` tokens but drops plain whitespace, and
//! its `unparse` regenerates spacing from scratch. That is the
//! right trade for linting and metrics and the wrong one for the two tools
//! planned next:
//!
//! * a **formatter** must read the author's comments and line structure to
//!   place them in the output, and must be able to prove "I only changed
//!   whitespace";
//! * a **codemod** (e.g. `Get-WmiObject` to `Get-CimInstance`) must rewrite
//!   one command and leave every other byte alone, or the diff is unusable.
//!
//! Both need the same foundation: tokens that own every byte of the input.
//! Per the crate's versioning policy this is a breaking change to the token
//! shape, so it ships as a sibling `v2` module; `v1` stays untouched.
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
//! unparse; the formatter and codemod tools never need the v1 unparser.
//!
//! # From tokens to a tree
//!
//! Tokens alone carry the codemod use case (find a token, patch its span).
//! For anything that reasons about structure, `tree::parse_with_tokens`
//! (available when the `v1` feature is also enabled) pairs the v1 AST with
//! these tokens: every node gets a `TokenRange` into the token vector, so
//! `node.unparse_lossless()` returns that node's exact source, trivia and
//! all. It reuses the v1 parser rather than forking the grammar; see the
//! `tree` module for how ranges are recovered from node offsets. The walks
//! come in untyped (`TreeNode::walk`, `walk_with_ancestors`) and typed
//! (`Tree::walk_zipped`) forms; the zipped walk hands a visitor each typed
//! v1 node, its range-bearing mirror, and the ancestor path, which is what
//! a refactoring tool wants.
//!
//! # Formatting
//!
//! [`formatter::format_source`] is a width-aware formatter built on these
//! tokens and verified with the native parser, so it works under `v2`
//! alone. It normalizes indentation, spacing, blank lines, and over-long
//! lines while preserving every token byte-for-byte, and it checks its own
//! output by re-lexing and re-parsing: the result either round-trips to
//! the identical program or the call returns an error instead of damaged
//! source. See the `formatter` module docs for the exact rules.
//!
//! # A native parser
//!
//! `tree` reuses the v1 parser and recovers ranges from node offsets, so it
//! needs the `v1` feature. `parser::parse` is the standalone path: a native
//! recursive-descent parser that consumes v2 tokens and builds an `ast::Node`
//! tree directly, with each node carrying both a byte `Span` and a
//! `TokenRange`. It depends on no `v1` code, so it builds under `v2` alone.
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
//! companion test checks the extracted C# metadata against v1's field by
//! field. The remaining differences are at the lexer, not the parser: the v1
//! and v2 lexers tokenize a few things differently (for example a dotted run
//! such as `a.b.c`).
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
//! parser and is only compiled when both features are on (see its docs for
//! the removal path).
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

pub mod ast;
pub mod edit;
pub mod formatter;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod tokens;
#[cfg(feature = "v1")]
pub mod tree;
pub mod trivia;

pub use ast::{CSharpImport, CSharpMemberDef, CSharpParam, Node, NodeKind, StringKind};
pub use edit::{apply_edits, EditError, TextEdit};
pub use formatter::{format_source, format_source_with, FormatError, FormatOptions};
pub use lexer::{lex, LexOutput};
pub use parser::{parse, parse_tokens, ParseError, ParseOutput};
pub use span::{LineCol, LineIndex, Span, TokenRange};
pub use tokens::{LexError, Token, TokenKind};
#[cfg(feature = "v1")]
pub use tree::{parse_with_tokens, Tree, TreeNode};
pub use trivia::{Trivia, TriviaKind};

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
