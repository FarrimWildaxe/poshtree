# poshtree

PowerShell syntax tooling for Rust: lossless tokens, a native parser, span-based
edits, and a formatter.

`poshtree` keeps every byte of the input. The lexer attaches whitespace,
newlines, and comments to the tokens as trivia, so reconstructing the token
stream returns the original source byte-for-byte, malformed input included.
A native recursive-descent parser sits on top and builds nodes with byte spans
and token ranges. It reports diagnostics and uses error nodes for recovery;
this is not a guarantee of complete PowerShell grammar coverage or recovery
for every malformed construct. Exact source reconstruction comes from the
tokens, not from regenerating code from the AST. The crate has no
external dependencies.

## Install

```toml
[dependencies]
poshtree = "0.6.0"
```

Or point at a local checkout:

```toml
[dependencies]
poshtree = { path = "../poshtree" }
```

The default features are `v2` and `csharp`. The front-end's items live under the
`v2` module and are used path-qualified.

| Feature | Default | Role |
| --- | --- | --- |
| `v2` | Yes | Lossless PowerShell tokens, native parser, text edits, and formatter; no external dependencies. |
| `csharp` | Yes | Bounded C# parsing, extraction, and refactoring for `Add-Type`; enables `v2`, adds no external dependencies. |

`csharp` is default-enabled but can be disabled. To use only `v2`:

```toml
[dependencies]
poshtree = { version = "0.6.0", default-features = false, features = ["v2"] }
```

With all features disabled, `encoding`, `textutil`, `decode_bytes`, and
`strip_bom` remain available. Upgrading from 0.5? See
[v1 and bridge removal in 0.6.0](#v1-and-bridge-removal-in-060) for the full
feature matrix, migration guidance, and an exact rollback pin.

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

The v2 lexer scans nested double-quoted strings and their `$()` subexpressions
with private, input-bounded heap frames rather than recursive calls. Token text,
trivia, byte spans, and malformed-input diagnostic order are unchanged. Block
comments still stop at the first `#>`, and here-string bodies are scanned
verbatim. The lexer adds no fixed lexical nesting limit or input/memory quota; frame
storage grows with nesting, and unclosed contexts can each produce a diagnostic.
This is lexical boundary scanning only, not a stack-safety guarantee for parsing,
walking, or dropping arbitrary trees.

## Parse and walk

`parse` returns the script tree plus any recoverable errors. Each node carries
a byte `Span` and a `TokenRange`, so a node can be sliced straight back to its
source. Error spans use UTF-8 byte offsets into the original input, including
any leading BOM bytes.

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

Supported syntax includes pipelines and `&&`/`||` chains, command parameters
and redirections, control-flow constructs, `function`/`filter`/`workflow`,
`class`, `enum`, `using`, `trap`/`data`/`dynamicparam`, expressions, and
double-quoted string interpolation parts. `Add-Type` extraction can find inline
C# or follow supported string assignments to variables. Parsing is syntactic:
it does not perform general PowerShell name binding or validate runtime behavior.

`NodeKind::MemberAccess` and `InvokeMember` retain `member` and expose
`member_span: Option<Span>`. `Some` covers only the directly consumed `Generic`
or `Keyword` name token in the original UTF-8 input, excluding `.`/`::` and
trivia. This includes Unicode and broad generic tokens such as `Foo-Bar` or
combining-mark spellings, even if a rename tool would reject them. Variables,
missing names, and unrecognized direct literals (including quoted/computed
forms) have `None`: no editable literal name is supplied. Recognition and
diagnostics are unchanged. A consumed name keeps its span during recovery,
including an unclosed argument list; `Some` does not validate the whole expression.
Inside double-quoted and here-double `$()` fragments, member spans are absolute
original-input byte offsets too, including nested fragments and leading BOM
bytes; the existing empty token ranges and token reconstruction are unchanged.
The C# cross-language member rename collector uses these stored spans for the
bounded static sites described below, not every name token the parser accepts.

### v2 raw interpolation

In double-quoted strings and double-quoted here-strings, variable parts use the
same dollar-reference rules as the v2 lexer: Unicode alphanumeric characters and
underscores (including numeric initials), repeated colon qualifiers, `$_`, `$?`,
`$^`, `$$`, and closed braced references. Backticks inside braces escape the next
character, including `}`; an unclosed braced reference is not a variable part.
This reuses the existing lexer grammar, not a broader Unicode identifier grammar.
Unsupported dollars and malformed references do not gain interpolation diagnostics.

Recognition uses the raw body, with each backtick skipping the next whole UTF-8
character: odd backtick runs escape a following dollar, even runs leave it active.
Parts remain in source order, retain raw variable spelling, and carry original
UTF-8 byte spans (including a leading BOM) and empty token ranges. The stored
string value and here-string newline trimming are unchanged; this is not runtime
string decoding or a decoded-text source map. Single-quoted forms remain literal,
and the conservative Add-Type raw-source eligibility check is unchanged.

The parser locates `$()` ends with the lexer's quote/comment/escape/here-string
boundary scan, including v2's doubled-quote rules. Only closing the root `$()`
counts, not closing a nested expression. A closed fragment is parsed without its
delimiters; an unclosed fragment keeps the exact available body remainder.
Wrapper and descendant spans are rebased to the original source, with empty
token ranges at the enclosing string token, including for empty fragments.
Boundary-probe and fragment diagnostics remain local, not new outer errors.
`parse` still includes outer lexer errors; `parse_tokens` still does not.

This shares the lexer's existing grammar: block comments end at the first
`#>`, and here-string bodies are scanned verbatim.
Fragment parsers now inherit the enclosing statement/expression depth, including
across nested detached fragments, rather than starting a fresh budget. Public
`parse` and `parse_tokens` start at depth zero. The existing statement/primary
expression guard allows depth 200 and rejects the next guarded entry, retaining
`too deep` error nodes. Its sticky overflow latch and variable cache remain
fragment-local; fragment diagnostics are still not merged into outer errors.

The separate existing interpolation-extraction cutoff is unchanged: strings at
depth 64 or above silently have no parts, including variable parts, even if their
raw bodies contain interpolation. Below 64, extracted fragments can still reach
the statement/expression limit. These are syntactic nesting guards, not complete
grammar or semantic validation, an input/memory quota, or an arbitrary-tree
parse/walk/drop stack-safety guarantee.

## C# in Add-Type

`Add-Type` embeds C# inside a PowerShell string, and a type or method defined
there is used back in PowerShell as ordinary syntax: `[Win32]`,
`[Win32]::Beep(...)`, `New-Object Win32`. The default-enabled `csharp` feature
parses a supported C# subset into a span-bearing tree and binds supported
references within each embedded unit. Refactoring helpers produce edits for
recognized C# declarations/references and PowerShell use sites. This is not a
lossless C# AST unparser, a full C# compiler, or a whole-program semantic rename.

### PowerShell source eligibility and cache

The PowerShell parser extracts from eligible literals directly or from a
parser-local cache for plain, unscoped variables. Recognized assignments
(`=`, `+=`, `-=`, `*=`, `/=`, `%=`, `??=`) invalidate the target's cache entry
after parsing the RHS; an `Add-Type` on the RHS can still use the old entry.
Prefix/postfix `++` and `--` also invalidate the affected variable. Only `=`
with an eligible string-literal RHS installs or restores an entry. Writes to
other variables do not clear it.

Single-quoted strings and single-quoted here-strings are eligible.
Double-quoted strings and here-strings are conservatively rejected if their
raw spelling contains an unescaped `$`: the check skips the character after
each backtick, independently of `expandable` and the incomplete
interpolation-parts list.
Unknown variables and rejected strings do not supply confirmed C# metadata.
Missing metadata is not a safety verdict or proof that `Add-Type` is absent
or cannot execute.

This is a bounded syntactic cache in linear parser traversal order, not scope,
branch, function, or dataflow analysis; it does not track indirect writes or
alternate runtime values. The parser extracts the delimiter-stripped raw body
and its span, not a fully decoded PowerShell runtime string or a general
decoded-text source map.

### C# parsing and refactoring bounds

The C# parser bounds nesting with one shared depth counter for type declarations,
members, real blocks, and synthetic control-flow scopes (including brace-free
chains). Each guarded entry increments the counter before checking the private
threshold of 200: reaching 200 triggers recovery, not acceptance of 200 of each
construct. Types, members, and controls produce error nodes and error spans;
real blocks at the threshold are skipped into empty blocks without a diagnostic.
Recovery scans tokens iteratively to a synchronization point or EOF, leaving an
enclosing `}` for the caller when appropriate; enclosing parse loops force
progress if a rule consumes nothing. Following declarations can survive, but
recovery does not promise to preserve every declaration in malformed input.
The bound applies to parser-produced tree nesting for recursive walking and
ordinary drop, not to input size or arbitrarily deep, manually constructed
`CsNode` trees. This does not add full C# syntax or semantic validation.

```rust
use poshtree::v2::{parse, apply_edits};
use poshtree::v2::csharp::rename_type;

let src = "Add-Type -TypeDefinition @'\npublic class Win32 { }\n'@\n[Win32]::Beep(800, 200)\n$h = New-Object Win32\n";
let out = parse(src);

// Renames the C# declaration and the recognized PowerShell uses here.
let edits = rename_type(&out.script, src, "Win32", "NativeMethods");
let fixed = apply_edits(src, &edits).unwrap();
// class NativeMethods ... [NativeMethods]::Beep(800, 200) ... New-Object NativeMethods
```

`rename_member` handles supported member references and `[Type]::Member` sites,
and `rename_csharp_field`, `_method`, `_local`, and `_parameter` rename within
the C# alone. C# resolution is per embedded unit and case-sensitive; PowerShell
name matching ignores ASCII case only, without Unicode case folding or
normalization. On the C# side, a member access is renamed only when its
receiver can mean that member: `this.Length`, or a static `Type.Length`, but
not an unrelated `other.Length` whose type is unknown. There is no assembly,
external type, or general overload resolution. With the feature on,
the `[DllImport]` extraction above also reads from this parse rather than the
fallback scanner. It adds no dependencies.

On the PowerShell side, `rename_member` requires a static access or call whose
receiver is a `TypeExpression`. Its bracket interior must match the requested
type name as a whole or by its final dotted segment, ignoring ASCII case.
It edits only `member_span: Some(...)` when the complete member name matches
and is nonempty with every character satisfying Rust `char::is_alphanumeric`
or being `_`. This includes parser-recognized Unicode, keyword, and
digit-initial generic names, but not Number tokens, dynamic, quoted/computed,
missing, escaped, punctuation-bearing, or non-alphanumeric combining-mark names.
Stored spans support recognized comments and backtick continuations around
`::`, including names inside nested double-quoted/here-double `$()` fragments;
surrounding trivia, comments, and unrelated string text remain unchanged.
There is no fallback source scan. Instance sites are omitted. These are
syntactic receiver checks, not PowerShell/C# type binding: no matching C#
declaration is required, simple-name ambiguity can remain, and replacement
validity and name collisions are not checked.

With the `csharp` front-end, extracted return and parameter types retain the
complete supported type syntax: qualified names, balanced generic arguments,
array suffixes (`[]`, `[,]`, and other ranks), nullable `?`, and pointer `*`
suffixes. Source spelling, including internal whitespace and comments, is
preserved; attributes, declaration/parameter modifiers, and parameter defaults
are excluded. This guarantee applies only to the `csharp` front-end, not to
the fallback scanner used without `csharp`.

This is bounded syntactic extraction, not semantic type resolution or validation
of executable P/Invoke signatures or marshalling types. Only methods and
parameters recognized by the C# parser are reported; it does not validate the
full C# grammar. Unsupported type syntax before a recognized name (for example,
an array rank containing a size expression) produces an empty type string.
A declaration without a recognized method name yields no import.

Type renames include the stored final identifiers in C# base-class and interface
lists when they bind to a type declared in the same embedded unit. C# comments
and string literal contents are left untouched. This uses simple-name binding,
not namespace-qualified type identity: generic arguments in those lists are not
tracked, and this is not a guarantee of full C# semantic resolution.

### Opt in to checked C# region rename

With the `csharp` feature, use `poshtree::v2::csharp::CheckedCSharpRegion`
when you have one raw UTF-8 C# slice and want either a complete edit plan
**within the checked model** or a typed refusal. Pass the entire document and
the raw region's byte span; pass the exact original declaration-name span,
including any `@`, not a cursor, reference, or text-search key. The caller
chooses the region: the checked API does not discover `Add-Type` bodies,
decode PowerShell strings, or rename any PowerShell sites.

```rust
use poshtree::v2::{Span, TextEdit};
use poshtree::v2::csharp::{CheckedCSharpRegion, CSharpRenameError, CSharpRenameMode};

fn main() -> Result<(), CSharpRenameError> {
    // One field: absolute spans include the two UTF-8 bytes of π.
    let source = "π|class C{int @x;void F(){x=this.@x;}}|";
    let region = CheckedCSharpRegion::parse(source, Span::new(3, 39))?;
    let plan = region.member_at(Span::new(15, 17))?
        .plan_rename(CSharpRenameMode::Single, "count")?;
    assert_eq!(plan.edits(), &[
        TextEdit::replace(Span::new(15, 17), "count"),
        TextEdit::replace(Span::new(27, 28), "count"),
        TextEdit::replace(Span::new(34, 36), "count"),
    ]);
    assert_eq!(plan.apply_to(source)?, "π|class C{int count;void F(){count=this.count;}}|");

    // One method: only the zero-argument overload and its call are edited.
    let source = "class C{void F(){}void F(int a){}void G(int v){F();this.F(v);}}";
    let region = CheckedCSharpRegion::parse(source, Span::new(0, 63))?;
    let plan = region.member_at(Span::new(13, 14))?
        .plan_rename(CSharpRenameMode::Single, "Renamed")?;
    assert_eq!(plan.edits(), &[
        TextEdit::replace(Span::new(13, 14), "Renamed"),
        TextEdit::replace(Span::new(47, 48), "Renamed"),
    ]);
    assert_eq!(plan.apply_to(source)?,
        "class C{void Renamed(){}void F(int a){}void G(int v){Renamed();this.F(v);}}");

    // Two same-count overloads: exact type evidence proves applicability,
    // not which overload wins. Single mode refuses; group mode is explicit.
    let source = "class C{void F(int a){}void F(bool b){}void G(int v){F(v);}}";
    let region = CheckedCSharpRegion::parse(source, Span::new(0, 60))?;
    let method = region.member_at(Span::new(28, 29))?;
    assert_eq!(method.plan_rename(CSharpRenameMode::Single, "Renamed").unwrap_err(),
        CSharpRenameError::Ambiguous {
            at: Span::new(53, 54),
            candidates: vec![Span::new(13, 14), Span::new(28, 29)],
        });
    let plan = method.plan_rename(CSharpRenameMode::OverloadGroup, "Renamed")?;
    assert_eq!(plan.edits(), &[
        TextEdit::replace(Span::new(13, 14), "Renamed"),
        TextEdit::replace(Span::new(28, 29), "Renamed"),
        TextEdit::replace(Span::new(53, 54), "Renamed"),
    ]);
    assert_eq!(plan.apply_to(source)?,
        "class C{void Renamed(int a){}void Renamed(bool b){}void G(int v){Renamed(v);}}");
    Ok(())
}
```

The subset includes uninitialized `int`/`bool` fields and local declarations,
ordinary class/struct methods with bounded bodies (brace blocks, simple
assignments and zero- or single-name-argument calls), and exact receiver and
local base evidence in the same region. Replacements must be fresh unescaped
ASCII identifiers, excluding keywords, contextual keywords, `_` and
spellings containing `__`. The model does not perform general overload
resolution, conversions, virtual dispatch, generic binding, external type
resolution, or analysis across units. It rejects generic methods anywhere in
the region and refuses a selected member in a generic type or ancestor; an
unrelated empty generic type can coexist. This is not a full C# syntax or
semantic validity check.

**Migrating from the collectors:** keep the old `rename_csharp_*` and
`rename_member`/`rename_type` calls if you need their existing behavior.
They return a `Vec<TextEdit>` from limited legacy lookup; they do not provide
the checked path's complete-or-refuse contract. To migrate a single-region
field or method workflow, make the caller explicitly opt in to the checked
path, surface `CSharpRenameError` (including unsupported syntax and ambiguous
or unresolved uses), and use `plan.apply_to(current_source)` to apply the batch.
`edits()` is available for preview. The plan owns its replacements but borrows
the **entire original document**: `apply_to` accepts a byte-identical copy and
refuses even a same-length change outside the region. After any edit, parse
the new document and reselect using the new exact span. To roll back, select
the legacy workflow explicitly with current input. Never silently turn a
checked refusal into a legacy edit batch or an empty success.

## Formatting

`format_source` is a width-aware formatter built on the lossless tokens.

```rust
let pretty = poshtree::v2::format_source("ls   -la\n")?;
assert_eq!(pretty, "ls -la\n");
```

It normalizes inter-token layout with width-aware line breaking, retaining
token text and source adjacency. It rejects input with lexer/parser diagnostics.
Before returning, it re-lexes and re-parses the output, comparing significant
token kinds/text and tree labels/shape; a mismatch returns an error. These are
checks against poshtree's own syntax model, not a proof of semantic equivalence
in PowerShell. It does not promise to insert spaces between glued tokens such
as `if(`.

## Diagnostic rendering from a v2 AST

With `v2` enabled (`csharp` is not required), use
`poshtree::v2::write_diagnostic_script` when you have only a native v2 script
tree, including one you built or modified. It reads the node kinds and their
current values and order, not source text, tokens, spans, or token ranges.
If starting from source, parse first and handle parse diagnostics separately.

```rust
use poshtree::v2::{parse, write_diagnostic_script};

let parsed = parse(
    "\u{feff}# demo\r\nparam($Top)\r\n\
     function Get-Msg([String]$s='ż''ółć', $n=1+2) { \
     function Show-One { param($v); Write-Output $v | Sort-Object }; $x=$n; Show-One $s }\r\n",
);
assert!(parsed.errors.is_empty());
let mut bytes = Vec::new();
write_diagnostic_script(&parsed.script, &mut bytes).unwrap();
assert_eq!(
    bytes,
    concat!(
        "param($Top)\n",
        "function Get-Msg([String]$s = 'ż''ółć', $n = 1 + 2) {\n",
        "    function Show-One {\n",
        "        param($v)\n",
        "        Write-Output $v | Sort-Object\n",
        "    }\n",
        "    $x = $n\n",
        "    Show-One $s\n",
        "}\n",
    )
    .as_bytes()
);
```

The writer accepts an optional leading `ParamBlock` in the root `Script` and
in a function's `Script` body if that function has no header parameters.
Header lists and `ParamBlock`s accept simple ASCII `$name` declarations,
with at most one `[int]` or `[string]` annotation (matched ignoring ASCII
case, emitted with the stored spelling) and an optional default from the
bounded expression grammar. The writer refuses filters, a body parameter block
mixed with nonempty header parameters, and other forms outside the subset
rather than moving declarations or guessing how to render unsupported nodes.

Eligible non-filter nodes always render as `function`: the AST does not
distinguish a source `workflow` from a `function` in this case. An absent
header list and an explicit empty `()` are the same in the AST, so both omit
header parentheses. A body `param()` remains visible where it was declared.

An `If` statement has at least one guard and exactly one `Script` body per
guard, plus an optional `Script` for `else`. Branch bodies cannot have their
own `ParamBlock`. The writer emits every arm in AST order using lowercase
`if`, `elseif`, and `else`, including empty bodies.

Unlabeled `while`, `do`–`while`, and `do`–`until` statements require a `Script`
body with no leading `ParamBlock`. Loop guards use the same grammar as `If`
guards. Loop keywords are lowercase, and `while` or `until` after a `do` body
follows the closing brace on the same line. Empty bodies add no blank line.

A `Flow` with the exact stored keyword `return` accepts an optional expression.
Bare `break` and `continue` require those exact lowercase keywords in the AST
and no value. They are allowed only as statements lexically within a supported
loop body, including in `If` arms there. This permission resets at function
entry and does not leak to siblings outside the loop body. The writer does not
infer runtime loop targets. Targeted `break` and `continue`, labeled statements,
`for`, `foreach`, and other flow forms are unsupported.

Ordinary expressions are a simple variable, an ASCII digit run, a supported
single-quoted string, or one `+`, `-`, or `*` between two such atoms. A guard
can also use one `-gt` or `-eq` between two atoms. The writer recognizes these
comparisons ignoring ASCII case, keeps their stored spelling (such as `-gT`),
and refuses them outside guards, including in return values and parameter
defaults.

Each enclosing function, conditional, or loop adds four spaces of indentation,
up to four enclosing blocks total. Ordinary statements can appear at that depth,
but a fifth block is refused. The separate limit of two function levels also
applies across conditionals and loops. Lines end with LF; closing braces align
with their opening statements, and `elseif` and `else` follow the preceding
arm's closing brace on the same line.

Other supported statements include bounded expressions, `=` assignments to
simple variables, plain commands, and pipelines of at least two plain commands.
Plain commands cannot have invocation operators, redirections, or C# metadata.
Check the [`poshtree::v2::write_diagnostic_script` rustdoc](src/v2/render.rs)
for the exact node and field restrictions.

The writer checks the entire tree before writing: unsupported nodes or fields
return an `io::Error` of kind `InvalidInput` containing
`poshtree::v2::DiagnosticRenderError`, leaving the destination untouched. Write
errors are returned unchanged and can leave partial output; the function does
not flush the writer or roll back writes.

This is not lossless rendering or a drop-in replacement for the former v1
unparser. It does not guarantee that the output reparses, is valid or executable
PowerShell, or preserves semantics. Use `reconstruct` for byte-exact token
replay, `format_source` for checked formatting, or `apply_edits` for targeted
edits against original text.

There is no legacy fallback on a render error. See
[the 0.6.0 removal notice](#v1-and-bridge-removal-in-060) for the compatibility
boundary.

## Examples

The `examples/` directory has four runnable programs.

`pascalize` requires `v2`. It capitalizes the first letter of each
hyphen-separated segment in plain-identifier command invocations
(`get-childitem` becomes `Get-Childitem`, not `Get-ChildItem`). It uses `parse`
and `apply_edits`, leaving function-definition names, call-operator targets,
dot-sourced paths, comments, strings, arguments, and layout alone.

```console
$ cargo run --example pascalize           # built-in demo
$ cargo run --example pascalize -- file.ps1
$ cat file.ps1 | cargo run --example pascalize -- -
```

`pinvoke-report` requires `csharp` (enabled by default). It reads recognized C#
in `Add-Type` blocks and prints extracted `[DllImport]` signatures and recognized
declarations, including types, methods, and fields. It does not rewrite source
or validate P/Invoke marshalling.

```console
$ cargo run --features csharp --example pinvoke-report           # built-in demo
$ cargo run --features csharp --example pinvoke-report -- file.ps1
$ cat file.ps1 | cargo run --features csharp --example pinvoke-report -- -
```

`rename-native` requires `csharp`. With no arguments it runs a built-in type
rename demo; with `type FROM TO` or `member TYPE FROM TO` it reads stdin and
writes edited source to stdout. It uses the bounded cross-language rename
helpers described above, not a general semantic binder.

```console
$ cargo run --features csharp --example rename-native            # built-in demo
$ cat file.ps1 | cargo run --features csharp --example rename-native -- type Win32 NativeApi
$ cat file.ps1 | cargo run --features csharp --example rename-native -- member Win32 MessageBox ShowMessage
```

Both `pascalize` and `rename-native` report edit-application failures to stderr
and exit unsuccessfully, without printing source or a success count. This
applies to built-in demos and supplied input alike. Successful output is
unchanged.

`scope-rename` requires only `v2` and runs a built-in demo, printing the name
mapping and before/after source. It numbers user variables by script/function
scope and renames user-defined functions. It leaves automatic/preference
variables and named arguments such as `-Path` unchanged. This is a limited
script/function-scope model, not full PowerShell binding or a general-purpose
file/stdin rename CLI.

```console
$ cargo run --example scope-rename
```

## Versioning

During `0.x` development, breaking changes use a minor-version bump and
include migration guidance.

### v1 and bridge removal in 0.6.0

Version 0.6.0 removes the `v1` feature, `poshtree::v1`, `poshtree::unparse`,
their flat crate-root exports (including `tokenize`, `parse`, `parse_tokens`,
and `dump_ast_to_ps1`), and the `poshtree::v2::tree` bridge and its `v2`
re-exports (`parse_with_tokens`, `Tree`, and `TreeNode`). There are no
compatibility aliases for the feature or the removed APIs. Remove `v1` from
your 0.6 dependency's feature list. The native `v2` APIs and default feature
selection are unchanged.

| Cargo selection | Support in 0.6 |
| --- | --- |
| Defaults or `--all-features` | Native `v2` and `v2::csharp`. |
| `--no-default-features` | Shared utilities only. |
| `--no-default-features --features v2` | Native `v2` with fallback `Add-Type` import scanning; no C# front end. |
| `--no-default-features --features csharp` | Same as defaults: `csharp` enables `v2`. |

For native workflows, choose the API that matches the input and goal:

| Goal | Native API |
| --- | --- |
| Render a supported, possibly edited AST for diagnostics | [`poshtree::v2::write_diagnostic_script`](#diagnostic-rendering-from-a-v2-ast) emits bounded diagnostic text. Unsupported trees return an `InvalidInput` error containing `DiagnosticRenderError` before any write; an I/O failure can leave partial output. |
| Replay the original bytes, including trivia | [`poshtree::v2::reconstruct`](#lossless-tokens) replays retained tokens, not AST edits. |
| Reformat source with syntactic checks | [`poshtree::v2::format_source`](#formatting) formats source text, not an AST. |
| Make targeted edits to original source | [`poshtree::v2::apply_edits`](#lossless-tokens) applies explicit byte-span replacements. |

When moving off `v1`, switch to qualified native imports such as
`poshtree::v2::lex` and `poshtree::v2::parse`. Native tokens carry trivia and
use UTF-8 byte spans into the original input, including any leading BOM;
`LineIndex` derives line and byte column on demand. Do not assume the same
token stream, positions, or traversal order as in v1 or the bridge. Some
workflows have no drop-in native replacement:

- **Error-tolerant source dump:** `write_diagnostic_script` takes a tree, not
  source. Handle parse diagnostics separately; the writer does not reproduce
  `dump_ast_to_ps1`'s parse-error tolerance or emitted bytes and does not
  guarantee faithful source or executable PowerShell.
- **Byte-preserving comment masking:** If v2's comment recognition fits your
  needs, use comment trivia spans from `poshtree::v2::lex` to replace comment
  bytes with spaces while preserving CR and LF. There is no exact replacement
  for v1's `strip_comments` and `strip_comments_tokens` behavior.
- **Bridge traversal:** Use `poshtree::v2::parse` and
  `poshtree::v2::Node::walk_with_ancestors` for a depth-first walk of native
  nodes with root-first ancestor paths. Adapt to native node types and child
  order. There are no zipped typed v1 visitors, scalar summaries, or the
  bridge's trivia-inclusive footprints; if you need trivia coverage, inspect
  the retained tokens and define your own range policy.
- **One-call C# extraction:** There is no drop-in for
  `poshtree::v1::parser::parse_csharp_member_def` on raw C#. When starting
  from PowerShell, inspect `poshtree::v2::NodeKind::CSharpMemberDef` for
  metadata from [recognized `Add-Type` forms](#powershell-source-eligibility-and-cache).
  `v2` alone scans for `[DllImport]` signatures with a fallback scanner.
  With `csharp`, the import metadata comes from a bounded C# parse with richer
  supported type spellings. Neither promises the legacy extractor's results.

#### Roll back to 0.5.0

If you still need the removed behavior, replace the 0.6 dependency with a Git
dependency pinned to the exact commit tagged `v0.5.0`:

```toml
[dependencies]
poshtree = { git = "https://github.com/FarrimWildaxe/poshtree.git", rev = "34131a6e5022dfbc42e82a04c59f1d2da984eff3", default-features = false, features = ["v1"] }
```

Use `features = ["v1", "v2"]` if you need the bridge. If you also used the
default C# front end, include `csharp`. Update your application's lockfile
to use the pinned dependency. On the 0.5 line, v1 and the bridge are
repair-only: fixes, not new functionality. The exact Git pin does not track
later 0.5 changes.

### Native API changes in 0.5.0

`poshtree::v2::EditError` now includes `InvalidRange(Span)`. Exhaustive matches
on `EditError` must handle this new variant. `apply_edits` now rejects
reversed spans (`start > end`) before bounds or UTF-8 boundary checks for
that same edit; this does not guarantee error priority across different
invalid edits. Valid edit behavior is unchanged.

The public `poshtree::v2::NodeKind::MemberAccess` and `InvokeMember` variants
now have a `member_span: Option<Span>` field. This is source-breaking for
constructors and field-complete patterns. Supply the new field when
constructing either variant, and bind it or add `..` in patterns. `None`
is allowed and means no editable literal member name, not a zero-length
span. Manually constructed trees must supply a correct original-token span
or `None` (no automatic member edit).
Apply edits to that original source and reparse after changing it; stored spans
are not updated automatically. For example:

```rust
use poshtree::v2::{Node, NodeKind, Span, TokenRange};

let target = Node {
    kind: NodeKind::TypeExpression("T".into()),
    span: Span::new(0, 3),
    range: TokenRange { first: 0, end: 3 },
};
// Coordinates for the literal name in "[T]::M".
let access = NodeKind::MemberAccess {
    target: Box::new(target.clone()),
    member: "M".into(),
    member_span: Some(Span::new(5, 6)),
    is_static: true,
};
// A variable method name has no directly editable literal.
let call = NodeKind::InvokeMember {
    target: Box::new(target),
    member: "$name".into(),
    member_span: None,
    is_static: true,
    args: vec![],
};
fn literal_member_span(kind: &NodeKind) -> Option<Span> {
    match kind {
        NodeKind::MemberAccess { member_span, .. }
        | NodeKind::InvokeMember { member_span, .. } => *member_span,
        _ => None,
    }
}
assert_eq!(literal_member_span(&access), Some(Span::new(5, 6)));
assert_eq!(literal_member_span(&call), None);
```

## License

MIT. See [LICENSE](LICENSE).
