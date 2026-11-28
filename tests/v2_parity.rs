//! Native v2 parser regressions.
//!
//! The corpus checks for clean parsing across a bounded sample of the grammar.
//! Independent literals anchor native child order and selected AST fields,
//! including string interpolation, as well as complete `Add-Type` metadata.
#![cfg(feature = "v2")]

use poshtree::v2::{parse, CSharpImport, CSharpMemberDef, CSharpParam, Node, NodeKind, Span};

fn parse_without_diagnostics(src: &str) {
    let out = parse(src);

    assert_eq!(out.errors, [], "native parse diagnostics for {src:?}");
}

const CORPUS: &[&str] = &[
    // pipelines, commands, parameters with bound arguments, redirections
    "Get-ChildItem -Path 'C:\\tmp' -Recurse | Sort-Object Length\n",
    "Get-Process | Where-Object { $_.CPU -gt 10 } | Select-Object -First 5\n",
    "Write-Output hello world\n",
    "Get-Thing -Name foo -Count 5 -Force\n",
    "Get-Content in.txt > out.txt\n",
    "Run-It 2>&1\n",
    "& $command -Arg value\n",
    "& Invoke-Thing -Flag\n",
    ". script\n",
    // assignments and expressions
    "$x = 1 + 2 * 3\n",
    "$y = $a -gt 5 -and $b -lt 10\n",
    "$z += $a * ($b - $c)\n",
    "$t = $cond ? 'yes' : 'no'\n",
    "$arr = @(1, 2, 3)\n",
    "$h = @{ Name = 'x'; Count = 3 }\n",
    "$n = -$value\n",
    "$m = -not $flag\n",
    "$i++\n",
    "$d = $(Get-Date)\n",
    "$sb = { param($p) $p * 2 }\n",
    // member access, indexing, invocation, casts, types
    "$x.Length\n",
    "$a[0]\n",
    "[System.Math]::Max(1, 2)\n",
    "$obj.Method($arg)\n",
    "$obj.Method($a, $b).Property\n",
    "[int]$value\n",
    "[System.Int32]\n",
    "$x?.Property\n",
    // control flow
    "if ($x -gt 1) { 'big' } elseif ($x -eq 1) { 'one' } else { 'small' }\n",
    "while ($i -lt 10) { $i++ }\n",
    "do { $i++ } while ($i -lt 3)\n",
    "do { $i-- } until ($i -le 0)\n",
    "for ($i = 0; $i -lt 10; $i++) { Write-Output $i }\n",
    "foreach ($f in $files) { $f.Name }\n",
    "switch ($x) { 1 { 'one' } default { 'other' } }\n",
    "try { Invoke-Thing } catch { Write-Error 'fail' } finally { Clean-Up }\n",
    "trap { 'trapped' }\n",
    // flow keywords
    "return $result\n",
    "return\n",
    "throw 'bad'\n",
    "break\n",
    "continue\n",
    // pipeline chains
    "Test-Path $p && Write-Output ok || Write-Error no\n",
    // definitions
    "function Get-Thing { param([string]$Name, [int]$Count = 3) $Name }\n",
    "filter Square { $_ * $_ }\n",
    "workflow Flow1 { Get-Service }\n",
    "using namespace System.Collections.Generic\n",
    "using module MyModule\n",
    "enum Color { Red; Green; Blue }\n",
    "enum Size : int { Small = 1; Large = 100 }\n",
    "class Point { [int]$X; [int]$Y; Point([int]$x, [int]$y) { $this.X = $x; $this.Y = $y } [int] Sum() { return $this.X + $this.Y } }\n",
    "class Derived : Base { static [int]$Count; [string] Describe() { return 'd' } }\n",
    // param block placement
    "param($a, $b)\nWrite-Output $a\n",
    // nesting
    "if ($a) { foreach ($i in 1, 2) { if ($i) { Write-Output $i } } }\n",
    "$result = if ($x) { 1 } else { 2 }\n",
    // multi-statement subexpression
    "$v = $( $a = 1; $a + 1 )\n",
    // operators: range, bitwise, format, comparison chains, concatenation
    "$r = 1..10\n",
    "$b = $x -band $y -bor $z\n",
    "$s = 'name={0}' -f $value\n",
    "$c = $a -eq $b -and $c -ne $d\n",
    "$cat = 'a' + 'b' + 'c'\n",
    "$p = $a * $b + $c / $d - $e % $f\n",
    // nested member/index/invoke chains (the v2 lexer groups a pure dotted
    // run like `a.b.c` into one token, so chains here are broken by brackets
    // or calls)
    "$idx = $a[$i + 1]\n",
    "$chain = $obj.Items[0].Name\n",
    "$call = [System.Math]::Floor($x).ToString()\n",
    "[int][string]$mixed\n",
    "$neg = -$a + -$b\n",
    // collections
    "$empty = @()\n",
    "$eh = @{}\n",
    "$nested = @{ outer = @{ inner = 1 } }\n",
    "$strs = @('a', 'b', 'c')\n",
    "$single = ,$x\n",
    // multiple catches and typed catch
    "try { Risky } catch [System.IO.IOException] { 'io' } catch { 'other' }\n",
    "try { A } finally { B }\n",
    // switch with several cases
    "switch ($n) { 1 { 'one' } 2 { 'two' } default { 'many' } }\n",
    // here-strings
    "$h1 = @'\nliteral\n'@\n",
    "$h2 = @\"\nexpandable\n\"@\n",
    // double-quoted string interpolation parts
    "$s1 = \"hello $name\"\n",
    "$s2 = \"value is $($x.Prop)\"\n",
    "$s3 = \"a $first then $second end\"\n",
    "$s4 = \"count: $(Get-Count -All)\"\n",
    "$s5 = \"scoped $env:PATH here\"\n",
    "$s6 = \"braced ${weird name} ok\"\n",
    "$s7 = \"plain text, no interpolation\"\n",
    "Write-Output \"user $user logged in\"\n",
    // nested control flow and pipelines
    "1..3 | ForEach-Object { $_ * 2 }\n",
    "if ($a) { if ($b) { Write-Output deep } }\n",
    "while ($true) { if ($done) { break } else { continue } }\n",
    "function Outer { function Inner { 1 } Inner }\n",
    "foreach ($x in $items) { if ($x -gt 0) { $x } else { 0 } }\n",
    // statements separated by semicolons
    "$a = 1; $b = 2; $a + $b\n",
    // assignment of an array literal
    "$pair = $first, $second\n",
    // class with hidden/static members and a property default
    "class Config { hidden [string]$Secret; static [int]$Version = 2; [void] Reset() { $this.Secret = '' } }\n",
    // backtick line continuations: the continued lines stay one statement
    "Write-Output `\n  hello `\n  world\n",
    "Get-Thing -Name foo `\n  -Value bar\n",
    "$total = 1 + `\n  2 + `\n  3\n",
    // nested hashtable and array as command arguments
    "Invoke-Stuff -Options @{ Retry = 3 } -Items @(1, 2)\n",
    // Add-Type: the extracted CSharpMemberDef appears as a command child
    "Add-Type -TypeDefinition 'public class A { }' -Language CSharp\n",
    "$code = 'public class B { }'\nAdd-Type $code\n",
    "Add-Type -MemberDefinition '[DllImport(\"user32.dll\")] public static extern int MessageBox(IntPtr h, string t, string c, int o);' -Name Win -Namespace N\n",
];

#[test]
fn native_parser_reports_no_diagnostics_for_corpus() {
    for src in CORPUS {
        parse_without_diagnostics(src);
    }
}

// Hand-written native traversal expectations with selected payloads.
const SKELETON_CASES: &[(&str, &str)] = &[
    (
        "$x = 1 + 2 * 3\n",
        "ScriptBlock[AssignmentStatement(=)[Variable($x),BinaryExpression(+)[NumberLiteral(1),BinaryExpression(*)[NumberLiteral(2),NumberLiteral(3)]]]]",
    ),
    (
        "Get-Content in.txt > out.txt\n",
        "ScriptBlock[Command(direct)[BareWord(Get-Content),BareWord(in.txt),Redirection(>)[BareWord(out.txt)]]]",
    ),
    (
        "& $command -Arg value\n",
        "ScriptBlock[Command(invoked)[Variable($command),CommandParameter(Arg)[BareWord(value)]]]",
    ),
    (
        "if ($x -gt 1) { 'big' } elseif ($x -eq 1) { 'one' } else { 'small' }\n",
        "ScriptBlock[IfStatement(2 guards,2 blocks,else=true)[BinaryExpression(-gt)[Variable($x),NumberLiteral(1)],BinaryExpression(-eq)[Variable($x),NumberLiteral(1)],ScriptBlock[StringLiteral(Single:'big')],ScriptBlock[StringLiteral(Single:'one')],ScriptBlock[StringLiteral(Single:'small')]]]",
    ),
    (
        "function Get-Thing { param([string]$Name, [int]$Count = 3) $Name }\n",
        "ScriptBlock[FunctionDefinition(Get-Thing)[ScriptBlock[ParamBlock[Parameter($Name)[TypeExpression(string)],Parameter($Count)[TypeExpression(int),NumberLiteral(3)]],Variable($Name)]]]",
    ),
    (
        "$s2 = \"value is $($x.Prop)\"\n",
        "ScriptBlock[AssignmentStatement(=)[Variable($s2),StringLiteral(Double:\"value is $($x.Prop)\")[SubExpression[ScriptBlock[MemberAccess(Prop)[Variable($x)]]]]]]",
    ),
];

fn native_skeleton(n: &Node) -> String {
    use NodeKind::*;
    let label = n.label();
    let head = match &n.kind {
        Assignment { op, .. } | Binary { op, .. } | Redirection { op, .. } => {
            format!("{label}({op})")
        }
        Command { invocation, .. } => {
            let mode = if *invocation { "invoked" } else { "direct" };
            format!("{label}({mode})")
        }
        If {
            conditions,
            blocks,
            else_block,
        } => format!(
            "{label}({} guards,{} blocks,else={})",
            conditions.len(),
            blocks.len(),
            else_block.is_some()
        ),
        BareWord(text)
        | Variable(text)
        | Number(text)
        | TypeExpression(text)
        | CommandParameter { name: text, .. }
        | Parameter { name: text, .. }
        | Function { name: text, .. }
        | MemberAccess { member: text, .. } => format!("{label}({text})"),
        StringLiteral { kind, value, .. } => format!("{label}({kind:?}:{value})"),
        _ => label.to_string(),
    };
    let children: Vec<_> = n.children().into_iter().map(native_skeleton).collect();
    if children.is_empty() {
        head
    } else {
        format!("{head}[{}]", children.join(","))
    }
}

fn assert_v2_skeleton_literal(src: &str, expected: &str) {
    let out = parse(src);

    assert!(
        out.errors.is_empty(),
        "native parse errors for {src:?}: {:?}",
        out.errors
    );
    assert_eq!(
        native_skeleton(&out.script),
        expected,
        "native skeleton for {src:?}"
    );
}

#[test]
fn native_parser_matches_literal_skeletons() {
    for &(src, expected) in SKELETON_CASES {
        assert_v2_skeleton_literal(src, expected);
    }
}

struct CSharpCase {
    src: &'static str,
    code: &'static str,
    apis: &'static [&'static str],
    parameter: &'static str,
    code_span: Span,
    imports: &'static [ImportCase],
}

struct ImportCase {
    dll: &'static str,
    function: &'static str,
    returns: &'static str,
    params: &'static [(&'static str, &'static str)],
}

// These constant sources remain eligible under the conservative Add-Type
// extraction contract; no interpolation or runtime evaluation is required.
// The fallback scanner and parsed C# route both cover these simple types
// and omit parameter attributes and defaults from the signature.
const CSHARP_CASES: &[CSharpCase] = &[
    CSharpCase {
        src: "Add-Type -TypeDefinition 'public class Foo { public int N() { return 1; } }'\n",
        code: "public class Foo { public int N() { return 1; } }",
        apis: &[],
        parameter: "typedefinition",
        code_span: Span { start: 26, end: 75 },
        imports: &[],
    },
    CSharpCase {
        src: "$c = @\"\n[DllImport(\"kernel32.dll\", SetLastError=true)] public static extern bool Beep(uint freq, uint dur);\n\"@\nAdd-Type -MemberDefinition $c -Name Native\n",
        code: "[DllImport(\"kernel32.dll\", SetLastError=true)] public static extern bool Beep(uint freq, uint dur);",
        apis: &["Beep"],
        parameter: "memberdefinition",
        code_span: Span { start: 8, end: 107 },
        imports: &[ImportCase {
            dll: "kernel32.dll",
            function: "Beep",
            returns: "bool",
            params: &[("uint", "freq"), ("uint", "dur")],
        }],
    },
    CSharpCase {
        src: "Add-Type -MemberDefinition '[DllImport(\"user32\")] public static extern int Show([In] string msg, int flags = 0);' -Name U\n",
        code: "[DllImport(\"user32\")] public static extern int Show([In] string msg, int flags = 0);",
        apis: &["Show"],
        parameter: "memberdefinition",
        code_span: Span {
            start: 28,
            end: 112,
        },
        imports: &[ImportCase {
            dll: "user32",
            function: "Show",
            returns: "int",
            params: &[("string", "msg"), ("int", "flags")],
        }],
    },
];

fn assert_v2_csharp_literal(case: &CSharpCase) {
    let src = case.src;
    let expected = CSharpMemberDef {
        code: case.code.into(),
        code_span: case.code_span,
        imports: case
            .imports
            .iter()
            .map(|import| CSharpImport {
                dll: import.dll.into(),
                function: import.function.into(),
                returns: import.returns.into(),
                params: import
                    .params
                    .iter()
                    .map(|&(type_name, name)| CSharpParam {
                        type_name: type_name.into(),
                        name: name.into(),
                    })
                    .collect(),
            })
            .collect(),
        apis: case.apis.iter().map(|api| api.to_string()).collect(),
        parameter: case.parameter.into(),
    };

    let out = parse(src);
    let mut metadata = Vec::new();
    out.script.walk(&mut |n| {
        if let NodeKind::CSharpMemberDef(def) = &n.kind {
            metadata.push(def.clone());
        }
    });

    assert!(
        out.errors.is_empty(),
        "native parse errors for {src:?}: {:?}",
        out.errors
    );
    assert_eq!(metadata, [expected], "v2 Add-Type extraction for {src:?}");
}

#[test]
fn native_parser_matches_literal_csharp_extraction() {
    for case in CSHARP_CASES {
        assert_v2_csharp_literal(case);
    }
}
