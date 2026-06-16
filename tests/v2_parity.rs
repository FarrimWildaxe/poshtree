//! Structural parity between the native v2 parser and the v1 parser.
//!
//! Both trees are reduced to a canonical label skeleton and compared. The v1
//! side is read straight from the correlation [`TreeNode`], whose children
//! mirror `AstNode::safe_children`. The v2 side is reduced with the same rules
//! v1's `safe_children` applies: control-flow and block bodies are inlined
//! (no wrapper node), a `param` block sorts last among its siblings, a command
//! name counts as a child only under `&`/`.` invocation, and a redirection
//! contributes its target rather than a node of its own.
//!
//! The corpus exercises the full grammar, including the two pieces of derived
//! metadata: a double-quoted string's interpolation `parts` (children of the
//! `StringLiteral`) and `Add-Type`'s extracted `CSharpMemberDef` child. A
//! companion test, [`native_parser_matches_v1_csharp_extraction`], checks the
//! extracted C# fields against v1's directly.
//!
//! What stays out of scope is lexical, not grammatical: where the two lexers
//! tokenize differently, the parse trees differ in turn. A pure dotted run
//! like `a.b.c` is one v2 token, and a leading-slash bareword like `/tmp` is
//! one v2 `Generic` where v1 sees a `/` operator followed by a word (so v1
//! does not bind it as a parameter argument). The corpus avoids those.
#![cfg(all(feature = "v1", feature = "v2"))]

use poshtree::v2::tree::{parse_with_tokens, TreeNode};
use poshtree::v2::{parse, Node, NodeKind};

// V1 skeleton, straight from the correlation tree

fn skel_v1(n: &TreeNode) -> String {
    if n.children.is_empty() {
        n.label.to_string()
    } else {
        let kids: Vec<String> = n.children.iter().map(skel_v1).collect();
        format!("{}[{}]", n.label, kids.join(","))
    }
}

// V2 skeleton, applying v1's safe_children rules

fn stmts_of(n: &Node) -> &[Node] {
    if let NodeKind::Script(v) = &n.kind {
        v
    } else {
        &[]
    }
}

/// Statement skeletons with a `param` block moved to the end, matching v1's
/// ScriptBlock, which stores the param block separately and lists it last.
fn block(stmts: &[Node]) -> Vec<String> {
    let mut normal = Vec::new();
    let mut params = Vec::new();
    for s in stmts {
        if matches!(s.kind, NodeKind::ParamBlock(_)) {
            params.push(skel_v2(s));
        } else {
            normal.push(skel_v2(s));
        }
    }
    normal.extend(params);
    normal
}

fn kids_v2(n: &Node) -> Vec<String> {
    use NodeKind::*;
    match &n.kind {
        Script(v) => block(v),
        ParamBlock(v) | Pipeline(v) | Array(v) | ArrayLiteral(v) => v.iter().map(skel_v2).collect(),
        ClassDefinition { members, .. } | EnumDefinition { members, .. } => {
            members.iter().map(skel_v2).collect()
        }
        PipelineChain { left, right, .. } => vec![skel_v2(left), skel_v2(right)],
        Command {
            name,
            invocation,
            elements,
            redirections,
            csharp,
        } => {
            let mut k = Vec::new();
            if *invocation {
                k.push(skel_v2(name));
            }
            for e in elements {
                k.push(skel_v2(e));
            }
            if let Some(cs) = csharp {
                k.push(skel_v2(cs));
            }
            for r in redirections {
                if let Redirection {
                    target: Some(t), ..
                } = &r.kind
                {
                    k.push(skel_v2(t));
                }
            }
            k
        }
        CSharpMemberDef(_) => Vec::new(),
        CommandParameter { argument, .. } => argument.iter().map(|a| skel_v2(a)).collect(),
        Redirection { target, .. } => target.iter().map(|t| skel_v2(t)).collect(),
        Assignment { target, value, .. } => vec![skel_v2(target), skel_v2(value)],
        If {
            conditions,
            blocks,
            else_block,
        } => {
            let mut k = Vec::new();
            for (c, b) in conditions.iter().zip(blocks.iter()) {
                k.push(skel_v2(c));
                k.extend(block(stmts_of(b)));
            }
            if let Some(e) = else_block {
                k.extend(block(stmts_of(e)));
            }
            k
        }
        While { condition, body } => {
            let mut k = vec![skel_v2(condition)];
            k.extend(block(stmts_of(body)));
            k
        }
        DoWhile {
            body, condition, ..
        } => {
            let mut k = vec![skel_v2(condition)];
            k.extend(block(stmts_of(body)));
            k
        }
        For {
            init,
            condition,
            update,
            body,
        } => {
            let mut k = Vec::new();
            if let Some(x) = init {
                k.push(skel_v2(x));
            }
            if let Some(x) = condition {
                k.push(skel_v2(x));
            }
            if let Some(x) = update {
                k.push(skel_v2(x));
            }
            k.extend(block(stmts_of(body)));
            k
        }
        ForEach {
            variable,
            iterable,
            body,
        } => {
            let mut k = vec![skel_v2(variable), skel_v2(iterable)];
            k.extend(block(stmts_of(body)));
            k
        }
        Switch { input, cases, .. } => {
            let mut k = vec![skel_v2(input)];
            for c in cases {
                k.extend(block(stmts_of(c)));
            }
            k
        }
        Function { body, .. } => block(stmts_of(body)),
        Try {
            body,
            catches,
            finally_block,
        } => {
            let mut k = block(stmts_of(body));
            for c in catches {
                if let Catch { body } = &c.kind {
                    k.extend(block(stmts_of(body)));
                }
            }
            if let Some(f) = finally_block {
                k.extend(block(stmts_of(f)));
            }
            k
        }
        Catch { body } => block(stmts_of(body)),
        ClassMember {
            parameters,
            default,
            body,
            ..
        } => {
            let mut k: Vec<String> = parameters.iter().map(skel_v2).collect();
            if let Some(d) = default {
                k.push(skel_v2(d));
            }
            if let Some(b) = body {
                k.extend(block(stmts_of(b)));
            }
            k
        }
        Flow { keyword, value } => {
            if keyword == "return" || keyword == "throw" {
                value.iter().map(|v| skel_v2(v)).collect()
            } else {
                Vec::new()
            }
        }
        Ternary {
            condition,
            if_true,
            if_false,
        } => vec![skel_v2(condition), skel_v2(if_true), skel_v2(if_false)],
        Binary { left, right, .. } => vec![skel_v2(left), skel_v2(right)],
        Unary { operand, .. } | PostfixUnary { operand, .. } | Cast { operand, .. } => {
            vec![skel_v2(operand)]
        }
        MemberAccess { target, .. } => vec![skel_v2(target)],
        InvokeMember { target, args, .. } => {
            let mut k = vec![skel_v2(target)];
            k.extend(args.iter().map(skel_v2));
            k
        }
        Index { target, index } => vec![skel_v2(target), skel_v2(index)],
        Paren(inner) => vec![skel_v2(inner)],
        SubExpression(s) | ScriptBlockExpression(s) => block(stmts_of(s)),
        Hashtable(pairs) => {
            let mut k = Vec::new();
            for (a, b) in pairs {
                k.push(skel_v2(a));
                k.push(skel_v2(b));
            }
            k
        }
        StringLiteral { parts, .. } => parts.iter().map(skel_v2).collect(),
        Using { .. } | Variable(_) | Number(_) | TypeExpression(_) | BareWord(_) | Error(_) => {
            Vec::new()
        }
        // NodeKind is non-exhaustive; any variant without its own skeleton
        // contributes no children here.
        _ => Vec::new(),
    }
}

fn skel_v2(n: &Node) -> String {
    use NodeKind::*;
    // The v1 parser does not model parameter metadata; it emits a bare
    // variable for each parameter slot. Render a v2 Parameter the same way so
    // the shared structure still compares equal (the attributes and default
    // are a v2-only enrichment, checked by their own tests).
    if matches!(n.kind, Parameter { .. }) {
        return "Variable".to_string();
    }
    let k = kids_v2(n);
    if k.is_empty() {
        n.label().to_string()
    } else {
        format!("{}[{}]", n.label(), k.join(","))
    }
}

fn assert_parity(src: &str) {
    let out = parse(src);
    assert!(
        out.errors.is_empty(),
        "native parse errors for {src:?}: {:?}",
        out.errors
    );
    let tree = parse_with_tokens(src);
    let v1 = skel_v1(&tree.root);
    let v2 = skel_v2(&out.script);
    assert_eq!(
        v1, v2,
        "skeleton mismatch for {src:?}\n v1: {v1}\n v2: {v2}"
    );
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
    // param block placement (param sorts last in the skeleton)
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
    // or calls, where v1 and v2 tokenize alike)
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
fn native_parser_matches_v1_tree_shape() {
    for src in CORPUS {
        assert_parity(src);
    }
}

/// The skeleton test only proves a `CSharpMemberDef` node exists; this one
/// compares the extracted code, apis, and parameter against v1's values.
#[test]
fn native_parser_matches_v1_csharp_extraction() {
    use poshtree::v1::ast::AstNode;

    let scripts = [
        "Add-Type -TypeDefinition 'public class Foo { public int N() { return 1; } }'\n",
        "$c = @\"\n[DllImport(\"kernel32.dll\", SetLastError=true)] public static extern bool Beep(uint freq, uint dur);\n\"@\nAdd-Type -MemberDefinition $c -Name Native\n",
        "Add-Type -MemberDefinition '[DllImport(\"user32\")] public static extern int Show([In] string msg, int flags = 0);' -Name U\n",
    ];

    for src in scripts {
        // v1 metadata
        let tree = parse_with_tokens(src);
        let mut v1_meta = None;
        fn walk_v1(n: &AstNode, out: &mut Option<(String, Vec<String>, String)>) {
            if let AstNode::CSharpMemberDef(c) = n {
                *out = Some((c.code.clone(), c.apis.clone(), c.parameter.clone()));
            }
            for ch in n.safe_children() {
                walk_v1(ch, out);
            }
        }
        walk_v1(&tree.ast, &mut v1_meta);

        // v2 metadata
        let out = parse(src);
        let mut v2_meta = None;
        out.script.walk(&mut |n| {
            if let NodeKind::CSharpMemberDef(c) = &n.kind {
                v2_meta = Some((c.code.clone(), c.apis.clone(), c.parameter.clone()));
            }
        });

        assert_eq!(v1_meta, v2_meta, "Add-Type extraction differs for {src:?}");
        assert!(v2_meta.is_some(), "expected a CSharpMemberDef for {src:?}");
    }
}
