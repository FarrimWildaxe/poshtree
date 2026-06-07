//! Integration tests for the v2 invariant: reconstructing the token stream
//! reproduces the input byte-for-byte, on well-formed and broken input
//! alike. Plus a small codemod demo built on `lex` + `TextEdit`, since that
//! pairing is the whole point of the module, and a check that v1 and v2
//! really do share their classification tables.

#![cfg(feature = "v2")]

use poshtree::v2::{apply_edits, lex, reconstruct, TextEdit, Token, TokenKind};

fn assert_roundtrip(src: &str) {
    let out = lex(src);
    assert_eq!(
        reconstruct(&out.tokens),
        src,
        "round trip failed for {src:?}"
    );
    assert_eq!(
        out.tokens.last().map(|t| t.kind),
        Some(TokenKind::Eof),
        "stream must end with Eof for {src:?}"
    );
}

#[test]
fn corpus_roundtrips_exactly() {
    let corpus: &[&str] = &[
        // everyday scripts
        "Get-ChildItem -Path C:\\tmp -Recurse | Where-Object { $_.Length -gt 1kb }\n",
        "function Get-Thing {\n    param([string]$Name)\n    process { $Name }\n}\n",
        "foreach ($f in Get-ChildItem) {\n    Write-Host $f.Name  # each one\n}\n",
        "$h = @{ a = 1; b = 'two' }\n$a = @(1, 2, 3)\n",
        "try { 1/0 } catch [System.Exception] { throw } finally { 'done' }\n",
        // comments, spacing, continuations
        "<#\n .SYNOPSIS\n   Block comment header\n#>\nparam()\n",
        "ls `\n  -la `\n  # trailing after continuation\n  /tmp\n",
        "   \t  \n\n# only trivia\n   ",
        // strings and here-strings
        "'it''s' + \"say \"\"hi\"\" to $name\"\n",
        "\"outer $(inner 'a)b' + \")(\" # )\n still string ) tail\"\n",
        "@'\nliteral $x '@ not the end\n'@\n",
        "@\"\nexpand $($obj.Prop)\n\"@ | Out-Null\n",
        // verbatim args
        "icacls.exe --% C:\\Program Files\\* /grant Users:(OI)(CI)F # verbatim\n",
        "ping --%\n",
        // redirection, chains, operators
        "cmd.exe 2>&1 *> all.log >> out.txt < in.txt\n",
        "Test-Path $p && ls || Write-Error 'no'\n",
        "$x = $a ?? $b; $i++; $i--; 1..5; [int]::MaxValue\n",
        "$name?.Length; $arr?[0] ??= 7\n",
        ":outer foreach ($x in 1..3) { break outer }\n",
        // CRLF, BOM, and unicode
        "ls\r\n\r\n  pwd\r\n",
        "\u{feff}Write-Output 'zażółć gęślą jaźń 🦀' # ok 🚀\r\n",
        // broken input still round-trips
        "'unterminated",
        "\"unterminated $(1 + ",
        "@'\nnever closed",
        "<# never closed",
        "${never closed",
        "x `",
        "$ @\n",
        "",
    ];
    for src in corpus {
        assert_roundtrip(src);
    }
}

#[test]
fn broken_corpus_reports_errors() {
    for src in ["'open", "\"open $(", "@'\nx", "<# x", "${x", "y `"] {
        assert!(
            !lex(src).errors.is_empty(),
            "expected at least one error for {src:?}"
        );
    }
}

/// v2 carries its own copies of the classification tables so it compiles
/// without v1. This test is the sync lock: while both versions exist, the
/// copies must stay equal (as sets; order is presentation). When v1 is
/// removed, this test goes with it.
#[cfg(feature = "v1")]
#[test]
fn classification_tables_match_v1() {
    fn sorted<'a>(s: &[&'a str]) -> Vec<&'a str> {
        let mut v = s.to_vec();
        v.sort_unstable();
        v
    }
    assert_eq!(
        sorted(poshtree::v1::tokens::KEYWORDS),
        sorted(poshtree::v2::tokens::KEYWORDS),
        "v2 KEYWORDS drifted from v1"
    );
    assert_eq!(
        sorted(poshtree::v1::tokens::NAMED_OPERATORS),
        sorted(poshtree::v2::tokens::NAMED_OPERATORS),
        "v2 NAMED_OPERATORS drifted from v1"
    );
}

/// Behavioral agreement on top of the table equality above: both lexers
/// classify every dash-word the same way, including the ones the shared
/// table contents happen to leave out (`-iin` is a `Parameter` in both).
#[cfg(feature = "v1")]
#[test]
fn dash_classification_agrees_with_v1() {
    use poshtree::v1::tokens::TokenType;

    for word in [
        "-eq",
        "-Eq",
        "-iMatch",
        "-creplace",
        "-and",
        "-f",
        "-Path",
        "-Force",
        "-iin",
        "-cnotin",
        "-xyzzy",
    ] {
        let v1_is_op = poshtree::tokenize(word)
            .first()
            .map(|t| t.ty == TokenType::Operator)
            .unwrap();
        let v2_is_op = lex(word)
            .tokens
            .first()
            .map(|t| t.kind == TokenKind::Operator)
            .unwrap();
        assert_eq!(v1_is_op, v2_is_op, "v1 and v2 disagree on {word:?}");
    }
}

/// Tiny deterministic xorshift generator; no dependencies, same sequence on
/// every run. Good enough to throw a few hundred random strings at the
/// lexer and check the two hard guarantees: no panic, exact round trip.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[test]
fn pseudo_fuzz_roundtrips_and_never_panics() {
    // Charset is biased toward the characters that drive lexer state:
    // quotes, backticks, sigils, newlines, comment markers.
    let charset: Vec<char> = "'\"`$@#(){}[]|&;,.:<>-+*/%?!= \t\r\nabz019_\\~^eEkKbBxXuUlL ż🦀"
        .chars()
        .collect();
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for _ in 0..500 {
        let len = (rng.next() % 60) as usize;
        let src: String = (0..len)
            .map(|_| charset[(rng.next() as usize) % charset.len()])
            .collect();
        let out = lex(&src); // must not panic
        assert_eq!(
            reconstruct(&out.tokens),
            src,
            "round trip failed for fuzz input {src:?}"
        );
    }
}

/// The codemod primitive, end to end: find deprecated `Get-WmiObject`
/// commands by token, patch only their spans, leave everything else
/// (casing, spacing, comments, the here-string) untouched.
#[test]
fn mini_codemod_preserves_surroundings() {
    let src = "\
# inventory script
$bios = get-wmiobject Win32_BIOS   # keep this comment
$os   = Get-WmiObject -Class Win32_OperatingSystem
@'
Get-WmiObject inside a here-string stays put
'@
";
    let out = lex(src);
    assert!(out.errors.is_empty());

    let edits: Vec<TextEdit> = out
        .tokens
        .iter()
        .filter(|t: &&Token| t.kind == TokenKind::Generic && t.value_eq_ci("Get-WmiObject"))
        .map(|t| TextEdit::replace(t.span, "Get-CimInstance"))
        .collect();
    assert_eq!(edits.len(), 2, "here-string body must not match");

    let rewritten = apply_edits(src, &edits).unwrap();
    assert_eq!(
        rewritten,
        "\
# inventory script
$bios = Get-CimInstance Win32_BIOS   # keep this comment
$os   = Get-CimInstance -Class Win32_OperatingSystem
@'
Get-WmiObject inside a here-string stays put
'@
"
    );

    // and the rewrite is itself lexable and lossless
    assert_roundtrip(&rewritten);
}

/// Spans point at the original bytes: slicing the source with a token's
/// span gives the token text, including through multi-byte characters.
#[test]
fn spans_match_source_slices() {
    let src = "Write-Output 'zażółć 🦀' -NoEnumerate # fin\n";
    let out = lex(src);
    for token in &out.tokens {
        assert_eq!(token.span.slice(src), token.value);
        for trivia in token.leading.iter().chain(&token.trailing) {
            assert_eq!(trivia.span.slice(src), trivia.text);
        }
    }
}

/// The token-retaining tree, end to end: walk typed AST nodes zipped with
/// their token ranges, match `AstNode::Command` by variant (no string
/// sniffing), use the ancestor path for context, and rewrite just the
/// command-name token's span. Structure-aware, and still a minimal diff.
#[cfg(feature = "v1")]
#[test]
fn tree_driven_codemod_rewrites_by_node() {
    use poshtree::v1::ast::AstNode;
    use poshtree::v2::tree::parse_with_tokens;

    let src = "\
# inventory
$bios = Get-WmiObject Win32_BIOS   # keep
Get-Process | Where-Object { $_.Name -eq 'x' }
$os = get-wmiobject Win32_OperatingSystem
";
    let tree = parse_with_tokens(src);

    let mut edits: Vec<TextEdit> = Vec::new();
    let mut saw_pipeline_context = false;
    tree.walk_zipped(&mut |ast, node, ancestors| {
        if let AstNode::Command(c) = ast {
            if c.name.eq_ignore_ascii_case("Get-WmiObject") {
                let name_tok = &tree.tokens[node.range.first];
                assert!(name_tok.value_eq_ci("Get-WmiObject"));
                edits.push(TextEdit::replace(name_tok.span, "Get-CimInstance"));
            }
            if c.name == "Where-Object" {
                saw_pipeline_context = ancestors.iter().any(|(_, n)| n.label == "Pipeline");
            }
        }
    });
    assert_eq!(
        edits.len(),
        2,
        "two Get-WmiObject commands, found by variant"
    );
    assert!(saw_pipeline_context, "ancestors reported the pipeline");

    let rewritten = apply_edits(src, &edits).unwrap();
    assert_eq!(
        rewritten,
        "\
# inventory
$bios = Get-CimInstance Win32_BIOS   # keep
Get-Process | Where-Object { $_.Name -eq 'x' }
$os = Get-CimInstance Win32_OperatingSystem
"
    );
    // the rewrite re-parses and is itself lossless
    assert_eq!(parse_with_tokens(&rewritten).unparse_lossless(), rewritten);
}

/// The native v2 parser, used standalone (no v1): it parses, finds commands,
/// and every node's span slices back to its own source.
#[test]
fn native_parser_extracts_commands_and_spans() {
    use poshtree::v2::parser::parse;
    use poshtree::v2::NodeKind;

    let src = "Get-ChildItem -Path . | Sort-Object Length | Select-Object -First 3\n";
    let out = parse(src);
    assert!(out.errors.is_empty(), "{:?}", out.errors);

    let mut commands = Vec::new();
    out.script.walk(&mut |n| {
        if let NodeKind::Command { name, .. } = &n.kind {
            if let NodeKind::BareWord(s) = &name.kind {
                // the name node's span slices to the command name
                assert_eq!(name.span.slice(src), s);
                commands.push(s.clone());
            }
        }
    });
    assert_eq!(commands, ["Get-ChildItem", "Sort-Object", "Select-Object"]);
}

/// Differential oracle: the native v2 parser and the v1-correlation tree must
/// find the same commands. While v1 exists, it keeps the native parser honest.
#[cfg(feature = "v1")]
#[test]
fn native_parser_agrees_with_correlation_tree_on_commands() {
    use poshtree::v1::ast::AstNode;
    use poshtree::v2::parser::parse;
    use poshtree::v2::tree::parse_with_tokens;
    use poshtree::v2::NodeKind;

    let corpus = [
        "Get-Process | Where-Object { $_.CPU -gt 10 } | Sort-Object CPU\n",
        "$files = Get-ChildItem -Path C:\\temp -Recurse\n",
        "if (Test-Path $p) { Remove-Item $p } else { New-Item $p }\n",
        "function Save { param($x) Set-Content -Path out.txt -Value $x }\n",
        "foreach ($f in Get-ChildItem) { Write-Output $f.Name }\n",
        "$d = $(Get-Date); Write-Host $d\n",
        "Get-Service | ForEach-Object { Start-Service $_ }\n",
        "try { Invoke-Thing } catch { Write-Error 'fail' }\n",
        "while (Test-Connection $h) { Start-Sleep 1 }\n",
    ];
    for src in corpus {
        let out = parse(src);
        assert!(
            out.errors.is_empty(),
            "native errors for {src:?}: {:?}",
            out.errors
        );
        let mut native: Vec<String> = Vec::new();
        out.script.walk(&mut |n| {
            if let NodeKind::Command { name, .. } = &n.kind {
                if let NodeKind::BareWord(s) = &name.kind {
                    if !s.is_empty() {
                        native.push(s.to_ascii_lowercase());
                    }
                }
            }
        });

        let tree = parse_with_tokens(src);
        let mut oracle: Vec<String> = Vec::new();
        tree.walk_zipped(&mut |ast, _, _| {
            if let AstNode::Command(c) = ast {
                oracle.push(c.name.to_ascii_lowercase());
            }
        });

        native.sort();
        oracle.sort();
        assert_eq!(native, oracle, "command sets differ for {src:?}");
    }
}

/// Default Windows PowerShell saves scripts as UTF-16 LE with a BOM. This
/// drives the whole real-world pipeline for v2: decode the bytes, parse the
/// decoded text with the native parser, and confirm the lossless token stream
/// still reproduces the decoded source. UTF-16 BE and UTF-8-with-BOM are
/// checked too, since `Set-Content -Encoding` can produce either.
#[test]
fn windows_utf16_pipeline_feeds_the_native_parser() {
    use poshtree::decode_bytes;
    use poshtree::v2::{parse, NodeKind};

    // Includes a non-ASCII character so the test would fail if a code unit were
    // mishandled rather than merely if the ASCII survived.
    let script = "$name = 'wörld'\n\
                  Get-ChildItem -Path C:\\tmp | Where-Object { $_.Length -gt 1kb }\n\
                  function Greet { param([string]$who) \"hi $who\" }\n";

    let utf16le = |s: &str| -> Vec<u8> {
        let mut v = vec![0xFF, 0xFE];
        v.extend(s.encode_utf16().flat_map(u16::to_le_bytes));
        v
    };
    let utf16be = |s: &str| -> Vec<u8> {
        let mut v = vec![0xFE, 0xFF];
        v.extend(s.encode_utf16().flat_map(u16::to_be_bytes));
        v
    };
    let utf8bom = |s: &str| -> Vec<u8> {
        let mut v = vec![0xEF, 0xBB, 0xBF];
        v.extend_from_slice(s.as_bytes());
        v
    };

    for (label, bytes) in [
        ("utf-16le+bom", utf16le(script)),
        ("utf-16be+bom", utf16be(script)),
        ("utf-8+bom", utf8bom(script)),
    ] {
        // 1) Decoding reproduces the original text, BOM removed.
        let decoded = decode_bytes(&bytes);
        assert_eq!(decoded, script, "decode mismatch for {label}");
        assert!(
            !decoded.starts_with('\u{feff}'),
            "BOM leaked through for {label}"
        );

        // 2) The native parser accepts the decoded text without errors and
        //    recovers the expected structure.
        let out = parse(&decoded);
        assert!(
            out.errors.is_empty(),
            "parse errors for {label}: {:?}",
            out.errors
        );

        let mut saw_name = false;
        let mut commands = Vec::new();
        out.script.walk(&mut |n| match &n.kind {
            NodeKind::Variable(raw) if raw == "$name" => saw_name = true,
            NodeKind::Command {
                name, invocation, ..
            } if !invocation => {
                if let NodeKind::BareWord(cmd) = &name.kind {
                    commands.push(cmd.clone());
                }
            }
            _ => {}
        });
        assert!(saw_name, "expected $name assignment for {label}");
        assert!(
            commands
                .iter()
                .any(|c| c.eq_ignore_ascii_case("get-childitem")),
            "expected Get-ChildItem command for {label}, got {commands:?}"
        );

        // 3) The lossless layer round-trips the decoded text byte-for-byte.
        let relexed = lex(&decoded);
        assert!(relexed.errors.is_empty(), "lex errors for {label}");
        assert_eq!(
            reconstruct(&relexed.tokens),
            decoded,
            "round trip for {label}"
        );
    }
}
