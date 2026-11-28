//! Integration tests for the v2 invariant: reconstructing the token stream
//! reproduces the input byte-for-byte, on well-formed and broken input
//! alike. Plus a small codemod demo built on `lex` + `TextEdit`, since that
//! pairing is the whole point of the module, and checks for native
//! classification and AST traversal.

#![cfg(feature = "v2")]

use poshtree::v2::{
    apply_edits, lex, reconstruct, Span, TextEdit, Token, TokenKind, Trivia, TriviaKind,
};

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
    use poshtree::v2::{parse, ParseError};

    let cases = [
        (
            "'open",
            vec![ParseError {
                message: "unterminated string at 0..5".into(),
                span: Span::new(0, 5),
            }],
        ),
        (
            "\"open $(",
            vec![
                ParseError {
                    message: "unterminated subexpression in string at 6..8".into(),
                    span: Span::new(6, 8),
                },
                ParseError {
                    message: "unterminated string at 0..8".into(),
                    span: Span::new(0, 8),
                },
            ],
        ),
        (
            "@'\nx",
            vec![ParseError {
                message: "unterminated here-string at 0..4".into(),
                span: Span::new(0, 4),
            }],
        ),
        (
            "<# x",
            vec![ParseError {
                message: "unterminated block comment at 0..4".into(),
                span: Span::new(0, 4),
            }],
        ),
        (
            "${x",
            vec![ParseError {
                message: "unterminated braced variable at 0..3".into(),
                span: Span::new(0, 3),
            }],
        ),
        (
            "y `",
            vec![ParseError {
                message: "dangling escape character at 2..3".into(),
                span: Span::new(2, 3),
            }],
        ),
        // BOM (three bytes), `(`, quote, and ż (two bytes): lexer errors
        // precede parser recovery errors, both in original-input coordinates.
        (
            "\u{feff}('ż",
            vec![
                ParseError {
                    message: "unterminated string at 4..7".into(),
                    span: Span::new(4, 7),
                },
                ParseError {
                    message: "expected ')'".into(),
                    span: Span::new(7, 7),
                },
            ],
        ),
    ];
    for (src, expected) in cases {
        assert_parse_errors(src, expected);
    }

    fn assert_parse_errors(src: &str, expected: Vec<ParseError>) {
        let out = parse(src);

        assert_eq!(out.errors, expected, "diagnostics for {src:?}");
    }
}

const DASH_CASES: &[(&str, TokenKind)] = &[
    ("-eq", TokenKind::Operator),
    ("-Eq", TokenKind::Operator),
    ("-iMatch", TokenKind::Operator),
    ("-creplace", TokenKind::Operator),
    ("-and", TokenKind::Operator),
    ("-f", TokenKind::Operator),
    ("-Path", TokenKind::Parameter),
    ("-Force", TokenKind::Parameter),
    ("-iin", TokenKind::Operator),
    ("-cnotin", TokenKind::Operator),
    ("-xyzzy", TokenKind::Parameter),
];

fn assert_native_word_kind(word: &str, kind: TokenKind) {
    let out = lex(word);
    let tokens: Vec<_> = out
        .tokens
        .iter()
        .map(|t| (t.kind, t.value.as_str()))
        .collect();

    assert!(
        out.errors.is_empty(),
        "errors for {word:?}: {:?}",
        out.errors
    );
    assert_eq!(tokens, [(kind, word), (TokenKind::Eof, "")], "{word:?}");
}

#[test]
fn native_keyword_classification_is_case_insensitive_and_word_exact() {
    for (word, kind) in [
        ("if", TokenKind::Keyword),
        ("ElSeIf", TokenKind::Keyword),
        ("RETURN", TokenKind::Keyword),
        ("if-only", TokenKind::Generic),
        ("returning", TokenKind::Generic),
    ] {
        assert_native_word_kind(word, kind);
    }
}

#[test]
fn native_dash_classification_matches_literal_token_kinds() {
    for &(word, kind) in DASH_CASES {
        assert_native_word_kind(word, kind);
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
    let cases = [
        (
            "Write-Output 'zażółć 🦀' -NoEnumerate # fin\n",
            vec![
                Token {
                    kind: TokenKind::Generic,
                    value: "Write-Output".into(),
                    span: Span::new(0, 12),
                    leading: vec![],
                    trailing: vec![Trivia {
                        kind: TriviaKind::Whitespace,
                        text: " ".into(),
                        span: Span::new(12, 13),
                    }],
                },
                Token {
                    kind: TokenKind::StringSq,
                    value: "'zażółć 🦀'".into(),
                    span: Span::new(13, 30),
                    leading: vec![],
                    trailing: vec![Trivia {
                        kind: TriviaKind::Whitespace,
                        text: " ".into(),
                        span: Span::new(30, 31),
                    }],
                },
                Token {
                    kind: TokenKind::Parameter,
                    value: "-NoEnumerate".into(),
                    span: Span::new(31, 43),
                    leading: vec![],
                    trailing: vec![
                        Trivia {
                            kind: TriviaKind::Whitespace,
                            text: " ".into(),
                            span: Span::new(43, 44),
                        },
                        Trivia {
                            kind: TriviaKind::LineComment,
                            text: "# fin".into(),
                            span: Span::new(44, 49),
                        },
                        Trivia {
                            kind: TriviaKind::Newline,
                            text: "\n".into(),
                            span: Span::new(49, 50),
                        },
                    ],
                },
                Token {
                    kind: TokenKind::Eof,
                    value: "".into(),
                    span: Span::new(50, 50),
                    leading: vec![],
                    trailing: vec![],
                },
            ],
        ),
        (
            "\u{feff}# ż\r\n  ls\n ",
            vec![
                Token {
                    kind: TokenKind::Generic,
                    value: "ls".into(),
                    span: Span::new(11, 13),
                    leading: vec![
                        Trivia {
                            kind: TriviaKind::Whitespace,
                            text: "\u{feff}".into(),
                            span: Span::new(0, 3),
                        },
                        Trivia {
                            kind: TriviaKind::LineComment,
                            text: "# ż".into(),
                            span: Span::new(3, 7),
                        },
                        Trivia {
                            kind: TriviaKind::Newline,
                            text: "\r\n".into(),
                            span: Span::new(7, 9),
                        },
                        Trivia {
                            kind: TriviaKind::Whitespace,
                            text: "  ".into(),
                            span: Span::new(9, 11),
                        },
                    ],
                    trailing: vec![Trivia {
                        kind: TriviaKind::Newline,
                        text: "\n".into(),
                        span: Span::new(13, 14),
                    }],
                },
                Token {
                    kind: TokenKind::Eof,
                    value: "".into(),
                    span: Span::new(15, 15),
                    leading: vec![Trivia {
                        kind: TriviaKind::Whitespace,
                        text: " ".into(),
                        span: Span::new(14, 15),
                    }],
                    trailing: vec![],
                },
            ],
        ),
    ];
    for (src, expected) in cases {
        assert_token_spans(src, expected);
    }

    fn assert_token_spans(src: &str, expected: Vec<Token>) {
        let out = lex(src);

        assert_eq!(out.tokens, expected, "tokens and trivia for {src:?}");
        for token in &out.tokens {
            assert_eq!(token.span.slice(src), token.value);
            for trivia in token.leading.iter().chain(&token.trailing) {
                assert_eq!(trivia.span.slice(src), trivia.text);
            }
        }
    }
}

#[test]
fn native_codemod_rewrites_only_pipeline_command_names_by_span() {
    use poshtree::v2::{parse, NodeKind};

    let src = "\u{feff}# ż inventory\r\n\
               Get-WmiObject Outside   # leave the standalone call\r\n\
               get-wmiobject Win32_BIOS | Write-Output Get-WmiObject  # keep\r\n";

    let out = parse(src);
    let mut edits = Vec::new();
    out.script.walk_with_ancestors(&mut |node, ancestors| {
        if let NodeKind::Command { name, .. } = &node.kind {
            if matches!(ancestors, [root, pipeline]
                if matches!(root.kind, NodeKind::Script(_))
                    && matches!(pipeline.kind, NodeKind::Pipeline(_)))
                && matches!(&name.kind, NodeKind::BareWord(word)
                    if word.eq_ignore_ascii_case("Get-WmiObject"))
            {
                edits.push(TextEdit::replace(name.span, "Get-CimInstance"));
            }
        }
    });
    let rewritten = apply_edits(src, &edits).unwrap();

    assert!(out.errors.is_empty(), "native errors: {:?}", out.errors);
    assert_eq!(
        rewritten,
        "\u{feff}# ż inventory\r\n\
         Get-WmiObject Outside   # leave the standalone call\r\n\
         Get-CimInstance Win32_BIOS | Write-Output Get-WmiObject  # keep\r\n"
    );
}

/// The native parser finds commands whose name spans slice back to the source.
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
                commands.push((s.clone(), name.span));
            }
        }
    });
    assert_eq!(
        commands,
        [
            ("Get-ChildItem".to_string(), Span::new(0, 13)),
            ("Sort-Object".to_string(), Span::new(24, 35)),
            ("Select-Object".to_string(), Span::new(45, 58)),
        ]
    );
}

/// Literal command sets catch omissions and arguments mistaken for calls.
fn command_cases() -> [(&'static str, &'static [&'static str]); 9] {
    [
        (
            "Get-Process | Where-Object { $_.CPU -gt 10 } | Sort-Object CPU\n",
            &["get-process", "sort-object", "where-object"],
        ),
        (
            "$files = Get-ChildItem -Path C:\\temp -Recurse\n",
            &["get-childitem"],
        ),
        (
            "if (Test-Path $p) { Remove-Item $p } else { New-Item $p }\n",
            &["new-item", "remove-item", "test-path"],
        ),
        (
            "function Save { param($x) Set-Content -Path out.txt -Value $x }\n",
            &["set-content"],
        ),
        (
            "foreach ($f in Get-ChildItem) { Write-Output $f.Name }\n",
            &["get-childitem", "write-output"],
        ),
        (
            "$d = $(Get-Date); Write-Host $d\n",
            &["get-date", "write-host"],
        ),
        (
            "Get-Service | ForEach-Object { Start-Service $_ }\n",
            &["foreach-object", "get-service", "start-service"],
        ),
        (
            "try { Invoke-Thing } catch { Write-Error 'fail' }\n",
            &["invoke-thing", "write-error"],
        ),
        (
            "while (Test-Connection $h) { Start-Sleep 1 }\n",
            &["start-sleep", "test-connection"],
        ),
    ]
}

#[test]
fn native_parser_matches_literal_commands() {
    use poshtree::v2::{parse, NodeKind};

    for (src, expected) in command_cases() {
        assert_native_commands(src, expected);
    }

    fn assert_native_commands(src: &str, expected: &[&str]) {
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

        native.sort();
        assert_eq!(native, expected, "native commands for {src:?}");
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
