# poshtree

Parse PowerShell into a syntax tree, walk it, and turn it back into source.

`poshtree` is a small PowerShell front-end for Rust. Under the hood it is a
lexer feeding a recursive-descent parser, with an unparser for going the other
way. It handles messy and deliberately obfuscated input without panicking,
which makes it a useful base for formatters, linters, refactoring tools, or an
editor integration that needs to know the shape of a script. The only
dependency is [`regex`](https://crates.io/crates/regex).

## Install

```toml
[dependencies]
poshtree = "0.1"
```

Or point at a local checkout:

```toml
[dependencies]
poshtree = { path = "../poshtree" }
```

## Parse and walk

`parse` returns the parsed script block together with any recoverable errors.
A malformed construct becomes an error node instead of aborting the parse, so
you always get a tree back.

```rust
use poshtree::v1::{ast::AstNode, parser::parse};

let (tree, errors) = parse("$x = 1 + 2 | Write-Output");
assert!(errors.is_empty());

let root = AstNode::ScriptBlock(tree);
let mut count = 0;
root.walk(&mut |_| count += 1);
println!("{count} nodes");
```

## Unparse (tree back to source)

```rust
let source = poshtree::unparse_source("$x = 1 + 2");
assert!(source.contains("$x"));
```

The result is meant to re-parse cleanly rather than match the original byte for
byte. Whitespace and comments are not preserved.

## Tokenize only

If you just want the token stream, skip the parser:

```rust
use poshtree::v1::lexer::tokenize;

let tokens = tokenize("Get-ChildItem -Path . -Recurse");
println!("{} tokens", tokens.len());
```

## What is in the box

The common items are re-exported at the crate root: `parse`, `parse_tokens`,
`tokenize`, `AstNode`, `NodeInfo`, `ScriptBlock`, `Token`, `unparse`,
`unparse_source`, and `dump_ast_to_ps1`. Everything else lives under the `v1`
module (`v1::ast`, `v1::lexer`, `v1::parser`, `v1::tokens`).

## Versioning

The tree types live under a `v1` module. If the tree ever needs an
incompatible redesign, it will arrive as a `v2` module and `v1` will keep
working, so pinning to `v1` is safe.

## License

MIT. See [LICENSE](LICENSE).
