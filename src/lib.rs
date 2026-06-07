//! `poshtree` turns PowerShell source into a syntax tree and back again,
//! without losing a byte.
//!
//! The default front-end is `v2`. Its lexer keeps whitespace, newlines, and
//! comments as trivia attached to the tokens, so reconstructing the stream
//! reproduces the source exactly, even for malformed input. A native
//! recursive-descent parser builds a tree where every node carries a byte
//! `Span` and a `TokenRange`; on top of that sit byte-level text edits
//! (`v2::TextEdit`) for minimal-diff rewriting and a width-aware formatter
//! (`v2::format_source`). It has no dependencies.
//!
//! A legacy `v1` front-end (a lexer, a recursive-descent parser building an
//! abstract syntax tree, and an unparser) stays available behind an opt-in
//! feature for code that still uses it. It is off by default and owns the
//! crate's only dependency, `regex`.
//!
//! Versioning works by addition: a breaking change ships as a sibling version
//! module instead of altering the published one, so pinned code keeps
//! compiling. Turn features off to compile just one front-end
//! (`default-features = false, features = ["v1"]`).
//!
//! # Quick start
//!
#![cfg_attr(feature = "v2", doc = " ```")]
#![cfg_attr(not(feature = "v2"), doc = " ```ignore")]
//! use poshtree::v2::parse;
//!
//! let out = parse("$x = 1 + 2 | Write-Output\n");
//! assert!(out.errors.is_empty());
//!
//! // A malformed construct becomes an error node, so there is always a tree.
//! let mut nodes = 0;
//! out.script.walk(&mut |_| nodes += 1);
//! assert!(nodes > 0);
//! ```

pub mod encoding;
pub mod textutil;
#[cfg(feature = "v1")]
pub mod unparse;
#[cfg(feature = "v1")]
pub mod v1;
#[cfg(feature = "v2")]
pub mod v2;

// Flat re-exports: the most common items. These stay v1-only on purpose;
// `v2::Token` would clash with `v1::tokens::Token`, so v2 items are always
// path-qualified.

pub use encoding::{decode_bytes, strip_bom};
#[cfg(feature = "v1")]
pub use unparse::{dump_ast_to_ps1, unparse, unparse_source};
#[cfg(feature = "v1")]
pub use v1::ast::{AstNode, NodeInfo, ScriptBlock};
#[cfg(feature = "v1")]
pub use v1::lexer::tokenize;
#[cfg(feature = "v1")]
pub use v1::parser::{parse, parse_tokens};
#[cfg(feature = "v1")]
pub use v1::tokens::Token;
