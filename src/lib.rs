//! PowerShell syntax tooling: lossless tokens, a native parser, span-based
//! edits, and a formatter.
//!
//! The default features are `v2` and `csharp`. The `v2` lexer keeps whitespace,
//! newlines, and comments as trivia attached to the tokens, so reconstructing
//! the stream reproduces the source exactly, even for malformed input. A native
//! recursive-descent parser builds nodes with a byte `Span` and a `TokenRange`;
//! on top of that sit byte-level text edits (`v2::TextEdit`) for minimal-diff
//! rewriting and a width-aware formatter
//! (`v2::format_source`). Reconstruction is a token-stream property, not an
//! AST-to-source guarantee. Parsing reports diagnostics and uses error nodes
//! for recovery, without promising complete PowerShell grammar coverage,
//! universal malformed-input recovery, or runtime validation.
//!
//! The `csharp` feature is default-enabled but can be disabled. It enables `v2`
//! and adds bounded C# parsing, extraction, and refactoring for `Add-Type` blocks.
//! Supported references are bound per embedded unit; there is no assembly or
//! external type resolution, full C# semantic analysis, or lossless C# AST
//! unparser. Both features have no external dependencies. For PowerShell-only
//! tooling, use `default-features = false, features = ["v2"]`. With all features
//! disabled, `encoding`, `textutil`, `decode_bytes`, and `strip_bom` remain
//! available.
//!
//! # Changes in 0.6.0
//!
//! The `v1` feature, the legacy front end and unparser, their crate-root
//! re-exports, and the `v2::tree` bridge and its re-exports have been removed
//! without compatibility aliases. Use `v2::lex`, `v2::parse`, and
//! `v2::Node::walk_with_ancestors` for native tokens, parsing, and traversal;
//! tokenization, source coordinates, and child order can differ from 0.5's
//! legacy APIs.
//!
//! Choose `v2::write_diagnostic_script` for bounded diagnostic text from a
//! supported, possibly edited native AST, `v2::reconstruct` for exact
//! retained-token replay, `v2::format_source` for checked formatting, or
//! `v2::apply_edits` for explicit byte-span edits to the original source.
//! These are distinct tasks, not interchangeable replacements for the
//! removed APIs.
//!
//! # Native API changes in 0.5.0
//!
//! Version 0.5.0 adds `v2::EditError::InvalidRange(Span)`.
//! Exhaustive matches on `EditError` must handle this variant. `apply_edits`
//! rejects reversed spans (`start > end`) before checking bounds or UTF-8
//! boundaries for that edit, without guaranteeing error priority across
//! different invalid edits. Valid edit behavior is unchanged.
//!
//! Version 0.5.0 also adds `member_span: Option<Span>` to the public
//! `v2::NodeKind::MemberAccess` and `InvokeMember` variants. Constructors
//! must supply it, and field-complete patterns must bind it or add `..`.
//! `Some` is the original-input UTF-8 byte span of a directly consumed
//! `Generic`/`Keyword` name token, excluding the operator and trivia, including
//! in nested double/here-double `$()` fragments.
//! Broad generic tokens retain spans even if rename-ineligible. Variables,
//! missing names, and unrecognized direct literals (including quoted/computed
//! forms) have `None`, meaning no editable literal name. A consumed name retains
//! its span during recovery, including unclosed calls; this is not whole-expression
//! validity or a grammar change. Token reconstruction and fragment token ranges
//! are unchanged. The C# rename collector separately requires a nonempty name
//! containing only Unicode alphanumeric characters or `_`, a static type-literal
//! receiver, and ASCII-case-insensitive type/member matches. Stored spans enable
//! recognized Unicode/trivia-separated sites; `None` never triggers rescanning.
//! Manually constructed trees must supply a correct original-token span or
//! `None` (no automatic member edit). Apply edits to that original source and
//! reparse after changing it; stored spans are not updated automatically.
//!
#![cfg_attr(feature = "v2", doc = " ```")]
#![cfg_attr(not(feature = "v2"), doc = " ```ignore")]
//! use poshtree::v2::{Node, NodeKind, Span, TokenRange};
//!
//! let target = Node {
//!     kind: NodeKind::TypeExpression("T".into()),
//!     span: Span::new(0, 3),
//!     range: TokenRange { first: 0, end: 3 },
//! };
//! // Coordinates for the literal name in "[T]::M".
//! let access = NodeKind::MemberAccess {
//!     target: Box::new(target.clone()),
//!     member: "M".into(),
//!     member_span: Some(Span::new(5, 6)),
//!     is_static: true,
//! };
//! // A variable method name has no directly editable literal.
//! let call = NodeKind::InvokeMember {
//!     target: Box::new(target),
//!     member: "$name".into(),
//!     member_span: None,
//!     is_static: true,
//!     args: vec![],
//! };
//! fn literal_member_span(kind: &NodeKind) -> Option<Span> {
//!     match kind {
//!         NodeKind::MemberAccess { member_span, .. }
//!         | NodeKind::InvokeMember { member_span, .. } => *member_span,
//!         _ => None,
//!     }
//! }
//! assert_eq!(literal_member_span(&access), Some(Span::new(5, 6)));
//! assert_eq!(literal_member_span(&call), None);
//! ```
//!
//! # Operator precedence
//!
//! The `v2` parser follows PowerShell's documented operator precedence
//! (`about_Operator_Precedence`), including the comma operator, which binds
//! tighter than `-f`, arithmetic, comparison, and logical operators and looser
//! only than casts, index, member access, subexpressions, and the unary
//! operators. So `1,2,1+2` parses as `(1,2,1)+2` (the array `1,2,1` with `2`
//! appended), as in PowerShell, and a comma-separated argument to a format
//! string (`"{0} {1}" -f $a, $b`) is grouped into one array before `-f` is
//! applied. An array used as a parameter default must be parenthesized
//! (`param($x = (1, 2))`) or use `@(...)`, since a bare top-level comma in a
//! parameter list separates parameters.
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
//! // Walk the parsed syntax tree.
//! let mut nodes = 0;
//! out.script.walk(&mut |_| nodes += 1);
//! assert!(nodes > 0);
//! ```
//!
//! # Formatting
//!
//! The formatter checks its output's significant token kinds/text and parse-tree
//! labels/shape against the input using the native front-end. It rejects
//! lexer/parser diagnostics or a check mismatch; these syntactic checks do not
//! prove PowerShell semantic equivalence.
//!
#![cfg_attr(feature = "v2", doc = " ```")]
#![cfg_attr(not(feature = "v2"), doc = " ```ignore")]
//! let source = "ls   -la\n";
//!
//! let pretty = poshtree::v2::format_source(source).unwrap();
//!
//! assert_eq!(pretty, "ls -la\n");
//! ```

pub mod encoding;
#[cfg(feature = "v2")]
mod ops;
pub mod textutil;
#[cfg(feature = "v2")]
pub mod v2;

pub use encoding::{decode_bytes, strip_bom};
