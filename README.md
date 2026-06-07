# poshtree

Lossless PowerShell parsing for Rust. Tokenize or parse a script, walk or
rewrite the result, and get the exact source back.

`poshtree` keeps every byte of the input. The lexer attaches whitespace,
newlines, and comments to the tokens as trivia, so reconstructing the token
stream returns the original source byte-for-byte, malformed input included.
A native recursive-descent parser sits on top and builds a tree whose every
node carries a byte span and a token range. Broken input becomes error nodes
instead of a failed parse, so there is always a tree to work with. That
combination makes it a practical base for formatters, linters, codemods, and
editor tooling. It has no dependencies.

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

Items live under the `v2` module and are used path-qualified; nothing is
re-exported at the crate root.

## Lossless tokens

Whitespace, newlines, and comments ride along as trivia on the tokens, and
`reconstruct` glues them back into the original source.

```rust
use poshtree::v2::{lex, reconstruct, apply_edits, TextEdit, TokenKind};

let src = "get-wmiobject Win32_BIOS   # keep this comment\n";
let out = lex(src);
assert_eq!(reconstruct(&out.tokens), src); // byte-for-byte

// Minimal-diff rewriting: patch one token, leave the rest alone.
let edits: Vec<TextEdit> = out.tokens.iter()
    .filter(|t| t.kind == TokenKind::Generic && t.value_eq_ci("Get-WmiObject"))
    .map(|t| TextEdit::replace(t.span, "Get-CimInstance"))
    .collect();
let fixed = apply_edits(src, &edits).unwrap();
assert_eq!(fixed, "Get-CimInstance Win32_BIOS   # keep this comment\n");
```

Every token and trivia carries a byte `Span`, and a `LineIndex` maps an offset
to line and column. `--%` is handled in the lexer: the rest of the line
becomes one raw `VerbatimArgs` token. A few constructs lex more cohesively
than you might expect, with a path like `C:\tmp` or a dotted run like `a.b.c`
staying a single token; the module docs spell those out.

## Parse and walk

`parse` returns the script tree plus any recoverable errors. Each node carries
a byte `Span` and a `TokenRange`, so a node can be sliced straight back to its
source.

```rust
use poshtree::v2::{parse, NodeKind};

let out = parse("get-process | sort-object CPU\n");
assert!(out.errors.is_empty());

out.script.walk(&mut |n| {
    if let NodeKind::Command { name, .. } = &n.kind {
        if let NodeKind::BareWord(s) = &name.kind {
            println!("command: {s}");
        }
    }
});
```

The grammar covers pipelines and `&&`/`||` chains, commands with
parameter-argument binding and redirections, every control-flow statement,
`function`/`filter`/`workflow`, `class`, `enum`, `using`,
`trap`/`data`/`dynamicparam`, the full expression layer, double-quoted string
interpolation parts, and `Add-Type` C# extraction (it pulls `[DllImport]`
signatures out of the inline C#, following a string through a variable
assignment when it has to). It runs against a broad corpus and is fuzzed, so
adversarial input recovers into error nodes rather than panicking.

## Formatting

`format_source` is a width-aware formatter built on the lossless tokens.

```rust
let pretty = poshtree::v2::format_source("if($x){\nls\n}\n")?;
// "if ($x) {\n    ls\n}\n"
```

It normalizes indentation, spacing, blank lines, backtick continuations, and
over-long lines, breaking them at pipes, chain operators, commas, and
brackets. Comments, here-strings, `--%` arguments, and token adjacency stay
byte-for-byte. It refuses input that has syntax errors, and before returning
it re-lexes and re-parses its own output to confirm the program is unchanged.
If that check fails you get an error instead of altered code.

## Example: pascalize

The `pascalize` example is a small codemod. It parses with `parse`, finds
command names, and rewrites each to PascalCase through `apply_edits`, touching
only the name tokens and leaving comments, strings, arguments, and layout
intact.

```console
$ cargo run --example pascalize           # built-in demo
$ cargo run --example pascalize -- file.ps1
$ cat file.ps1 | cargo run --example pascalize -- -
```

## Versioning

Breaking changes to the token or tree types ship as a new sibling version
module rather than mutating what is already published, so pinned code keeps
compiling. The current module is `v2`.

## License

MIT. See [LICENSE](LICENSE).
