//! Version 1 of the PowerShell front-end: tokenizer, AST, and parser.
//!
//! **Deprecated in 0.5 documentation only.** This module and its crate-root
//! re-exports remain available behind the opt-in `v1` feature and are
//! repair-only throughout 0.5. The v1 front end and its crate-root re-exports
//! will not be removed before 0.6.0.
//!
//! For new code, use `poshtree::v2::lex` and `poshtree::v2::parse` with the
//! `v2` feature. Native nodes and visitors are not interchangeable with
//! [`ast::AstNode`] or `NodeInfo::scalars`. Native comment trivia is not a
//! drop-in replacement for v1 comment masking, and native `Add-Type`
//! extraction does not reproduce the legacy extractor. The legacy diagnostic
//! unparser remains available in [`mod@crate::unparse`]; see the crate-level
//! migration notes for its 0.5.0 write-error return change.
//!
//! # Example
//! ```
//! use poshtree::v1::{ast::AstNode, parser::parse};
//!
//! let (tree, errors) = parse("$x = 1 + 2");
//! assert!(errors.is_empty());
//! let root = AstNode::ScriptBlock(tree);
//! let mut count = 0;
//! root.walk(&mut |_| count += 1);
//! assert!(count > 0);
//! ```

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod tokens;
