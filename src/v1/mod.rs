//! Version 1 of the PowerShell front-end: tokenizer, AST, and parser.
//!
//! These types sit under a version namespace so the tree can change shape
//! later without breaking callers: a breaking revision would ship as a sibling
//! `v2` module while `v1` stays put. Anything that walks the tree works against
//! [`ast::AstNode`].
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
