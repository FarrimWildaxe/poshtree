//! `poshtree` turns PowerShell source into a syntax tree and back again.
//!
//! It has three pieces: a lexer that produces tokens, a recursive-descent
//! parser that builds an abstract syntax tree ([`v1::ast::AstNode`]), and an
//! [`unparse`]r that regenerates source from a tree. `regex` is the only
//! dependency.
//!
//! The tree types live under a versioned [`v1`] module. A breaking change to
//! the tree ships as a new `v2` module instead of altering `v1`, so code
//! pinned to `v1` keeps compiling.
//!
//! # Quick start
//! ```
//! use poshtree::v1::{ast::AstNode, parser::parse};
//!
//! let (tree, errors) = parse("$x = 1 + 2");
//! assert!(errors.is_empty());
//!
//! // Walk every node in the tree.
//! let mut nodes = 0;
//! AstNode::ScriptBlock(tree).walk(&mut |_| nodes += 1);
//! assert!(nodes > 0);
//!
//! // Or round-trip source through the parser and back to text.
//! let out = poshtree::unparse_source("$x = 1 + 2");
//! assert!(out.contains("$x"));
//! ```

pub mod encoding;
pub mod textutil;
pub mod unparse;
pub mod v1;

// Flat re-exports: the most common items

pub use encoding::{decode_bytes, strip_bom};
pub use unparse::{dump_ast_to_ps1, unparse, unparse_source};
pub use v1::ast::{AstNode, NodeInfo, ScriptBlock};
pub use v1::lexer::tokenize;
pub use v1::parser::{parse, parse_tokens};
pub use v1::tokens::Token;
