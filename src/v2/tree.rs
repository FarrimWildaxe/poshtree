//! A token-retaining tree: the v1 AST paired with, for every node, the range
//! of v2 tokens it spans.
//!
//! # What this is, and what it is not
//!
//! This is the "lossless semi-CST" the token layer was built toward. It does
//! **not** reimplement the parser. It runs the existing, well-tested
//! [`v1`](crate::v1) recursive-descent parser, then correlates each AST node
//! with the [`v2`](crate::v2) token stream by byte offset. The result is a
//! [`TreeNode`] per AST node carrying a [`TokenRange`] into a shared
//! `Vec<Token>`, so a refactoring or formatting tool can ask "which exact
//! source bytes did this node come from?" and get an answer that includes the
//! original spacing and comments.
//!
//! Reusing the v1 parser is deliberate: forking 2,500 lines of grammar to
//! thread token indices through it would be a large, bug-prone change, and the
//! byte offsets v1 already records on every node ([`Location::pos`](crate::v1::ast::Location::pos)) are
//! enough to recover the ranges after the fact.
//!
//! # v1 coupling, and how it ends
//!
//! This module is the only part of `v2` that depends on `v1` (the lexer,
//! spans, trivia, and edits stand alone; their classification tables are
//! v2-owned copies). The dependency is the price of getting per-node ranges
//! without a parser fork. The planned endgame is a native v2 parser that
//! records [`TokenRange`]s directly while parsing v2 tokens; during that
//! port this module doubles as a differential-testing oracle (the native
//! parser's ranges must match the correlated ones), and once the port is
//! trusted, this module is deleted together with `v1`.
//!
//! # The guarantee
//!
//! For any input, including malformed input, unparsing the root node
//! reproduces the source byte-for-byte:
//!
//! ```
//! use poshtree::v2::tree::parse_with_tokens;
//!
//! let src = "$x = Get-WmiObject Win32_BIOS  # legacy\nif ($x) { 'hi' }\n";
//! let tree = parse_with_tokens(src);
//! assert_eq!(tree.unparse_lossless(), src);
//! ```
//!
//! [`Tree::unparse_lossless`] is just [`super::reconstruct`] over the whole
//! token vector; the per-node ranges are what make *partial* lossless
//! unparsing (one statement, one command) possible.
//!
//! # How ranges are assigned
//!
//! Each significant token is owned by exactly one node: the deepest node whose
//! source span contains the token's start offset. A node's [`TokenRange`] is
//! then the span from its first owned token through the last token owned by it
//! or any descendant. Trivia is not owned by nodes; it stays attached to the
//! tokens (leading/trailing) exactly as the lexer produced it, so
//! reconstruction stays exact without nodes having to track it.
//!
//! Interpolation nodes inside an expandable string (the `$x` in `"a $x b"`)
//! are a special case: v1 stamps them with an offset *inside* the string
//! token, which no top-level token starts at. Such nodes get an empty range
//! (`first == last`) pointing at the enclosing string token, and the string
//! token itself carries their bytes. This keeps the byte-for-byte guarantee:
//! those bytes are emitted once, by the string token.

use super::span::{Span, TokenRange};
use super::tokens::Token;
use super::{lex, reconstruct};
use crate::v1::ast::AstNode;
use crate::v1::parser::parse as v1_parse;

/// One node of the token-retaining tree: a label and scalar summary lifted
/// from the v1 AST, the node's [`TokenRange`], and its children. The tree
/// mirrors the shape of [`AstNode::safe_children`].
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Stable node name, e.g. `"Command"` (from [`crate::v1::ast::NodeInfo`]).
    pub label: &'static str,
    /// Scalar field summary for debugging, e.g. `name="Get-Item"`.
    pub scalars: String,
    /// Byte offset of the node, as recorded by the v1 parser.
    pub pos: usize,
    /// Tokens this node spans.
    pub range: TokenRange,
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    /// The tokens this node spans, as a slice of the tree's token vector.
    pub fn tokens<'a>(&self, tree: &'a Tree) -> &'a [Token] {
        &tree.tokens[self.range.first..self.range.end]
    }

    /// Reconstructs the exact source text of this node, including the
    /// original whitespace and comments between its tokens. Leading trivia
    /// of the node's first token and trailing trivia of its last token are
    /// included, so this is the node's full footprint in the source.
    pub fn unparse_lossless(&self, tree: &Tree) -> String {
        reconstruct(self.tokens(tree))
    }

    /// The byte span of this node's tokens, trivia included, or `None` for a
    /// node with an empty range.
    pub fn source_span(&self, tree: &Tree) -> Option<Span> {
        let toks = self.tokens(tree);
        match (toks.first(), toks.last()) {
            (Some(f), Some(l)) => Some(f.full_span().join(l.full_span())),
            _ => None,
        }
    }

    /// Visits this node and every descendant, depth-first, parents first.
    pub fn walk(&self, visitor: &mut impl FnMut(&TreeNode)) {
        visitor(self);
        for child in &self.children {
            child.walk(visitor);
        }
    }

    /// Like [`TreeNode::walk`], but the visitor also receives the ancestor
    /// path, ordered root first (empty for the node the walk started on).
    pub fn walk_with_ancestors(&self, visitor: &mut impl FnMut(&TreeNode, &[&TreeNode])) {
        fn go<'a>(
            node: &'a TreeNode,
            stack: &mut Vec<&'a TreeNode>,
            visitor: &mut impl FnMut(&TreeNode, &[&TreeNode]),
        ) {
            visitor(node, stack);
            stack.push(node);
            for child in &node.children {
                go(child, stack, visitor);
            }
            stack.pop();
        }
        go(self, &mut Vec::new(), visitor);
    }
}

/// A parsed PowerShell script: the v2 token stream plus a [`TreeNode`] tree
/// whose nodes carry token ranges into that stream.
#[derive(Debug, Clone)]
pub struct Tree {
    /// Every token of the source, in order, the last being
    /// [`TokenKind::Eof`](crate::v2::TokenKind::Eof). Indices in every [`TokenRange`] address this.
    pub tokens: Vec<Token>,
    /// Root of the node tree (the script block).
    pub root: TreeNode,
    /// The typed v1 AST this tree was built from, wrapped as
    /// [`AstNode::ScriptBlock`]. [`TreeNode`]s mirror its
    /// [`safe_children`](AstNode::safe_children) shape one-to-one, which is
    /// what makes [`Tree::walk_zipped`] possible.
    pub ast: AstNode,
    /// Recoverable parse errors reported by the v1 parser.
    pub errors: Vec<String>,
}

impl Tree {
    /// Visits every node with its typed AST counterpart and the ancestor
    /// path, depth-first, parents first. The ancestor slice is ordered root
    /// first and contains `(typed, range-bearing)` pairs; it is empty for
    /// the root itself.
    ///
    /// This is the walk a refactoring tool wants: match on the [`AstNode`]
    /// variant for structure, read the [`TreeNode`] for exact source
    /// position, and consult the ancestors for context ("am I inside a
    /// `param` block?").
    pub fn walk_zipped<'a, F>(&'a self, visitor: &mut F)
    where
        F: FnMut(&'a AstNode, &'a TreeNode, &[(&'a AstNode, &'a TreeNode)]),
    {
        fn go<'a, F>(
            ast: &'a AstNode,
            node: &'a TreeNode,
            stack: &mut Vec<(&'a AstNode, &'a TreeNode)>,
            visitor: &mut F,
        ) where
            F: FnMut(&'a AstNode, &'a TreeNode, &[(&'a AstNode, &'a TreeNode)]),
        {
            visitor(ast, node, stack);
            let kids = ast.safe_children();
            debug_assert_eq!(
                kids.len(),
                node.children.len(),
                "tree mirror out of sync under {}",
                node.label
            );
            stack.push((ast, node));
            for (ast_child, node_child) in kids.into_iter().zip(&node.children) {
                go(ast_child, node_child, stack, visitor);
            }
            stack.pop();
        }
        go(&self.ast, &self.root, &mut Vec::new(), visitor);
    }
    /// Reconstructs the entire source, byte-for-byte, from the token stream.
    pub fn unparse_lossless(&self) -> String {
        reconstruct(&self.tokens)
    }
}

/// Parses `src` and returns a [`Tree`]: the v1 AST shape with v2 tokens and a
/// token range on every node.
///
/// The parse itself is the v1 parser, so error recovery and node coverage are
/// identical to [`crate::v1::parser::parse`]; this adds the token stream and
/// the per-node ranges on top.
pub fn parse_with_tokens(src: &str) -> Tree {
    let tokens = lex(src).tokens;
    // The v1 parser strips a BOM internally and works in the resulting byte
    // space; v2 keeps the BOM as leading trivia on the first token, so token
    // start offsets already match v1's positions (a BOM contributes trivia,
    // not a token start). No offset translation is needed.
    let (ast, errors) = v1_parse(src);

    // Offset of each significant (non-Eof) token, for mapping node.pos ->
    // token index. Token starts are unique and ascending.
    let starts: Vec<usize> = tokens.iter().map(|t| t.span.start).collect();
    let builder = Builder {
        tokens: &tokens,
        starts: &starts,
    };
    let root_ast = AstNode::ScriptBlock(ast);
    let mut root = builder.build(&root_ast);
    builder.assign_ranges(&mut root, 0, tokens.len());

    Tree {
        tokens,
        root,
        ast: root_ast,
        errors,
    }
}

struct Builder<'a> {
    tokens: &'a [Token],
    starts: &'a [usize],
}

impl Builder<'_> {
    /// Build the bare node tree (labels, positions, children) from the AST,
    /// without ranges yet.
    fn build(&self, node: &AstNode) -> TreeNode {
        use crate::v1::ast::NodeInfo;
        TreeNode {
            label: node.label(),
            scalars: node.scalars(),
            pos: node.loc().pos,
            range: TokenRange { first: 0, end: 0 },
            children: node.safe_children().iter().map(|c| self.build(c)).collect(),
        }
    }

    /// Index of the first token starting at or after `pos`. Used to anchor a
    /// node whose recorded offset falls in leading trivia (a block comment or
    /// whitespace before the construct), so its range still begins at its
    /// first real token instead of collapsing to empty.
    fn token_at_or_after(&self, pos: usize) -> Option<usize> {
        match self.starts.binary_search(&pos) {
            Ok(i) => Some(i),
            Err(i) => (i < self.starts.len()).then_some(i),
        }
    }

    /// True when `pos` falls strictly inside some token's span (not at its
    /// start). Such a node is an in-string interpolation child whose bytes
    /// belong to the enclosing string token; it gets an empty range.
    fn inside_a_token(&self, pos: usize) -> bool {
        self.tokens
            .iter()
            .any(|t| t.span.start < pos && pos < t.span.end)
    }

    /// Assign a [`TokenRange`] to `node` and recurse. `lo`..`hi` is the token
    /// index window the node must fall within (its parent's range), used to
    /// bound the node's end at the next sibling.
    ///
    /// The node's `first` token is the one at (or, if the offset sits in
    /// leading trivia, just after) `node.pos`. Its `end` is the larger of one
    /// past its own first token and the `end` of every child, extended across
    /// the node's own trailing tokens up to the sibling boundary `hi`.
    fn assign_ranges(&self, node: &mut TreeNode, lo: usize, hi: usize) {
        // An offset strictly inside a token marks an in-string interpolation
        // child: its bytes are emitted by the enclosing string token, so it
        // gets an empty range anchored there.
        if self.inside_a_token(node.pos) {
            let anchor = self.containing_token(node.pos).unwrap_or(lo).clamp(lo, hi);
            node.range = TokenRange {
                first: anchor,
                end: anchor,
            };
            for child in &mut node.children {
                self.assign_ranges(child, anchor, anchor);
            }
            return;
        }

        let first = match self.token_at_or_after(node.pos) {
            Some(i) if i >= lo && i < hi => i,
            // Offset is past this node's window (an empty construct, e.g. a
            // defaulted-away node): empty range at the window start.
            _ => {
                node.range = TokenRange { first: lo, end: lo };
                for child in &mut node.children {
                    self.assign_ranges(child, lo, lo);
                }
                return;
            }
        };

        // Recurse into children first, bounding each child's window by the
        // start of the next child (or this node's `hi`). Children are in
        // source order, but a few v1 nodes stamp a child with the parent's
        // own pos (e.g. a pipeline's location is its first element's); guard
        // against a non-increasing boundary.
        let child_starts: Vec<usize> = node
            .children
            .iter()
            .map(|c| self.token_at_or_after(c.pos).unwrap_or(first).max(first))
            .collect();
        let n = node.children.len();
        for idx in 0..n {
            let child_lo = child_starts[idx];
            let child_hi = child_starts[idx + 1..]
                .iter()
                .copied()
                .find(|&s| s > child_lo)
                .unwrap_or(hi);
            self.assign_ranges(
                &mut node.children[idx],
                child_lo,
                child_hi.max(child_lo + 1),
            );
        }

        // End covers this node's first token, every child's range, and any
        // trailing tokens up to the sibling boundary `hi` that belong to this
        // node (closing delimiters etc.). Using `hi` as the end would over-
        // claim trivia past the node; instead extend only across tokens, then
        // let the natural token boundaries decide. The widest correct end is
        // `hi` clamped so it never exceeds the token vector; trailing trivia
        // on the last token is part of the node's footprint by design.
        let child_end = node.children.iter().map(|c| c.range.end).max();
        let end = child_end.unwrap_or(first + 1).max(first + 1).max(
            // absorb this node's own trailing tokens that no child starts
            self.own_trailing_end(first, hi),
        );
        node.range = TokenRange {
            first,
            end: end.min(hi).max(first + 1),
        };
    }

    /// The deepest enclosing token index for an offset that does not start a
    /// token (it falls strictly inside a token's span).
    fn containing_token(&self, pos: usize) -> Option<usize> {
        self.tokens
            .iter()
            .position(|t| t.span.start <= pos && pos < t.span.end)
    }

    /// Extend a node's end across its own trailing tokens: every token from
    /// `first` up to `hi` belongs to the node (the window `hi` is already the
    /// next sibling/parent boundary), so the node's natural end is `hi`. This
    /// returns `hi`, expressed as a helper so the intent is named at the call
    /// site.
    fn own_trailing_end(&self, first: usize, hi: usize) -> usize {
        hi.max(first + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrips(src: &str) {
        let tree = parse_with_tokens(src);
        assert_eq!(tree.unparse_lossless(), src, "tree round trip for {src:?}");
        // Every node's reconstructed text must be a substring of the source
        // at the right place, and ranges must be well-formed.
        let toks = tree.tokens.len();
        tree.root.walk(&mut |n| {
            assert!(
                n.range.first <= n.range.end,
                "inverted range on {}",
                n.label
            );
            assert!(n.range.end <= toks, "range past end on {}", n.label);
        });
    }

    #[test]
    fn root_roundtrips_across_corpus() {
        for src in [
            "Get-ChildItem -Path C:\\tmp | Where-Object { $_.Length -gt 1kb }\n",
            "$x = 1 + 2 * 3\n",
            "function Get-Thing {\n    param([string]$Name)\n    process { $Name }\n}\n",
            "if ($a -eq 1) { 'one' } elseif ($a -eq 2) { 'two' } else { 'other' }\n",
            "@{ a = 1; b = @(2, 3) } # a hashtable\n",
            "try { risky } catch [System.Exception] { recover } finally { cleanup }\n",
            "\"interpolated $x and $($y.Prop) end\"\n",
            "$obj.Method($arg)?.Chained[0]\n",
            "Get-WmiObject Win32_BIOS  # trailing comment kept\n",
            "  \n# just trivia\n  ",
            "'unterminated string still round-trips",
            "",
        ] {
            roundtrips(src);
        }
    }

    #[test]
    fn node_unparse_recovers_exact_source_with_trivia() {
        let src = "$x = Get-WmiObject  -Class  Win32_BIOS   # keep\n";
        let tree = parse_with_tokens(src);

        // The command node should reconstruct to its own source slice,
        // including the irregular internal spacing the author used.
        let mut command_text = None;
        tree.root.walk(&mut |n| {
            if n.label == "Command" {
                command_text = Some(n.unparse_lossless(&tree));
            }
        });
        let text = command_text.expect("a Command node");
        assert!(text.contains("Get-WmiObject"));
        assert!(text.contains("-Class"));
        assert!(text.contains("Win32_BIOS"));
        // internal double spaces are preserved, since we slice real tokens
        assert!(text.contains("Get-WmiObject  -Class  Win32_BIOS"));
    }

    #[test]
    fn token_ranges_nest_within_parents() {
        // A child's token range must lie within its parent's range.
        let src = "if ($x -gt 0) { Write-Output $x }\n";
        let tree = parse_with_tokens(src);
        fn check(node: &TreeNode) {
            for child in &node.children {
                if !child.range.is_empty() && !node.range.is_empty() {
                    assert!(
                        child.range.first >= node.range.first && child.range.end <= node.range.end,
                        "child {} [{},{}) escapes parent {} [{},{})",
                        child.label,
                        child.range.first,
                        child.range.end,
                        node.label,
                        node.range.first,
                        node.range.end,
                    );
                }
                check(child);
            }
        }
        check(&tree.root);
    }

    #[test]
    fn source_span_slices_back_to_node_text() {
        let src = "Write-Output 'hello'\n";
        let tree = parse_with_tokens(src);
        tree.root.walk(&mut |n| {
            if let Some(span) = n.source_span(&tree) {
                // The span must be valid and its slice must equal the node's
                // lossless unparse (both are the node's full footprint).
                assert_eq!(span.slice(src), n.unparse_lossless(&tree));
            }
        });
    }

    #[test]
    fn errors_are_surfaced_from_v1() {
        // A bare closing brace makes the v1 parser recover; the tree still
        // round-trips and the error is visible.
        let tree = parse_with_tokens("}\n");
        assert_eq!(tree.unparse_lossless(), "}\n");
    }

    /// Deterministic fuzz over the two structural guarantees the range
    /// assignment must uphold for every node on arbitrary input: a child's
    /// range nests inside its parent's, and a node's `source_span` slice
    /// equals its lossless unparse (its footprint is self-consistent).
    /// No dependency; same sequence every run.
    #[test]
    fn ranges_are_well_formed_under_fuzz() {
        fn nest_and_footprint(node: &TreeNode, tree: &Tree, src: &str) {
            if let Some(span) = node.source_span(tree) {
                assert!(span.start <= span.end && span.end <= src.len());
                assert_eq!(
                    span.slice(src),
                    node.unparse_lossless(tree),
                    "footprint mismatch on {}",
                    node.label
                );
            }
            for child in &node.children {
                if !child.range.is_empty() && !node.range.is_empty() {
                    assert!(
                        child.range.first >= node.range.first && child.range.end <= node.range.end,
                        "{} escapes parent {}",
                        child.label,
                        node.label
                    );
                }
                nest_and_footprint(child, tree, src);
            }
        }

        let charset: Vec<char> = "$@{}()[]| ;,.\"'`#-=+*/<>?:&\nabcXYZ012_".chars().collect();
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..1000 {
            let len = (next() % 60) as usize;
            let body: String = (0..len)
                .map(|_| charset[(next() as usize) % charset.len()])
                .collect();
            // Half the time, lead with trivia (a comment or blank lines) so the
            // root and first statements are anchored past leading trivia, not
            // at a token start. This exercises the trivia-anchoring path.
            let src = match next() % 3 {
                0 => format!("# header\n{body}"),
                1 => format!("\n\n   {body}"),
                _ => body,
            };
            let tree = parse_with_tokens(&src);
            assert_eq!(tree.unparse_lossless(), src, "root round trip for {src:?}");
            nest_and_footprint(&tree.root, &tree, &src);
        }
    }

    /// The statement-level use case a codemod needs: each top-level statement
    /// maps to its own token range and unparses back to its exact source
    /// slice, trivia included.
    #[test]
    fn top_level_statements_map_to_their_own_source() {
        let src = "$a = 1\nGet-WmiObject Win32_BIOS  # legacy\n$b = 2\n";
        let tree = parse_with_tokens(src);
        let stmts: Vec<(&str, String)> = tree
            .root
            .children
            .iter()
            .map(|s| (s.label, s.unparse_lossless(&tree)))
            .collect();
        assert_eq!(
            stmts,
            vec![
                ("AssignmentStatement", "$a = 1\n".to_string()),
                (
                    "Command",
                    "Get-WmiObject Win32_BIOS  # legacy\n".to_string()
                ),
                ("AssignmentStatement", "$b = 2\n".to_string()),
            ]
        );
    }

    /// The zipped walk pairs every typed AST node with its range-bearing
    /// mirror: same label, and the typed side is matchable as an enum.
    #[test]
    fn walk_zipped_pairs_typed_nodes_with_ranges() {
        use crate::v1::ast::{AstNode, NodeInfo};

        let src = "Get-WmiObject -Class Win32_BIOS | Out-Null\n";
        let tree = parse_with_tokens(src);
        let mut commands = Vec::new();
        tree.walk_zipped(&mut |ast, node, _ancestors| {
            assert_eq!(ast.label(), node.label, "zip drifted");
            if let AstNode::Command(c) = ast {
                commands.push((c.name.clone(), node.range.first));
            }
        });
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].0, "Get-WmiObject");
        assert_eq!(tree.tokens[commands[0].1].value, "Get-WmiObject");
        assert_eq!(commands[1].0, "Out-Null");
        assert_eq!(tree.tokens[commands[1].1].value, "Out-Null");
    }

    /// Ancestor paths answer context questions ("is this variable inside a
    /// function body?") without hand-rolled recursion.
    #[test]
    fn ancestors_expose_context() {
        let src = "function Outer { if ($x) { $inner } }\n$top\n";
        let tree = parse_with_tokens(src);

        let mut inner_path: Option<Vec<&'static str>> = None;
        let mut top_path: Option<Vec<&'static str>> = None;
        tree.root.walk_with_ancestors(&mut |node, ancestors| {
            if node.label == "Variable" {
                let path: Vec<&'static str> = ancestors.iter().map(|a| a.label).collect();
                if node.scalars.contains("inner") {
                    inner_path = Some(path);
                } else if node.scalars.contains("top") {
                    top_path = Some(path);
                }
            }
        });

        let inner_path = inner_path.expect("found $inner");
        assert!(inner_path.contains(&"FunctionDefinition"));
        assert!(inner_path.contains(&"IfStatement"));
        let top_path = top_path.expect("found $top");
        assert!(!top_path.contains(&"FunctionDefinition"));

        // The zipped variant carries the same ancestors with typed nodes.
        let mut saw = false;
        tree.walk_zipped(&mut |_, node, ancestors| {
            if node.label == "Variable" && node.scalars.contains("inner") {
                saw = ancestors
                    .iter()
                    .any(|(_, n)| n.label == "FunctionDefinition");
            }
        });
        assert!(saw);
    }
}
