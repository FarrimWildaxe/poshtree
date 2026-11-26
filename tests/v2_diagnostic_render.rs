//! Public-API tests for diagnostic rendering from native v2 trees.

#![cfg(feature = "v2")]

use std::io::ErrorKind;

use poshtree::v2::{
    parse, write_diagnostic_script, DiagnosticRenderError, Node, NodeKind, Span, TokenRange,
};

#[track_caller]
fn parse_script(source: &str) -> Node {
    let parsed = parse(source);
    assert_eq!(parsed.errors, [], "diagnostics for {source:?}");
    parsed.script
}

#[test]
fn diagnostic_render_of_native_script_omits_source_trivia() {
    let script = parse_script(
        "\u{feff}# example\r\n  $x   = 1 +  2 # sum\r\n\
         Write-Output   -Name 'ż''ółć' $x |  Sort-Object\r\n",
    );
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(
        bytes,
        "$x = 1 + 2\nWrite-Output -Name 'ż''ółć' $x | Sort-Object\n".as_bytes()
    );
}

#[test]
fn diagnostic_render_of_mutated_script_uses_current_values_and_order() {
    let mut script = parse_script("$x = 1; Write-Output $x");
    let NodeKind::Script(statements) = &mut script.kind else {
        panic!("expected a script root, got {:?}", script.kind);
    };
    let NodeKind::Assignment { value, .. } = &mut statements[0].kind else {
        panic!("expected an assignment first, got {:?}", statements[0].kind);
    };
    // Preserve the original span and token range while changing both text and order.
    value.kind = NodeKind::Number("42".into());
    statements.swap(0, 1);
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"Write-Output $x\n$x = 42\n");
}

#[test]
fn diagnostic_render_of_manual_tree_ignores_reversed_coordinates() {
    let node = |kind| Node {
        kind,
        span: Span {
            start: usize::MAX,
            end: 0,
        },
        range: TokenRange {
            first: usize::MAX,
            end: 0,
        },
    };
    let script = node(NodeKind::Script(vec![node(NodeKind::Binary {
        op: "*".into(),
        left: Box::new(node(NodeKind::Number("7".into()))),
        right: Box::new(node(NodeKind::Number("3".into()))),
    })]));
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"7 * 3\n");
}

#[test]
fn diagnostic_render_with_unsupported_statement_refuses_without_appending() {
    let script = parse_script("$x = 1\n:outer while ($x) { Write-Output $x }");
    let mut bytes = b"existing diagnostic\n".to_vec();

    let error = write_diagnostic_script(&script, &mut bytes).unwrap_err();

    assert_eq!(
        (
            error.kind(),
            error
                .get_ref()
                .and_then(|source| source.downcast_ref::<DiagnosticRenderError>())
        ),
        (
            ErrorKind::InvalidInput,
            Some(&DiagnosticRenderError::NodeNotAllowed {
                role: "statement",
                label: "LabeledStatement",
            })
        )
    );
    assert_eq!(bytes, b"existing diagnostic\n");
}

#[test]
fn diagnostic_render_of_root_while_preserves_statement_order() {
    let script = parse_script("$x = 1\nwhile ($x) { Write-Output $x }");
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"$x = 1\nwhile ($x) {\n    Write-Output $x\n}\n");
}

#[test]
fn diagnostic_render_of_do_while_canonicalizes_keywords_and_keeps_guard_spelling() {
    let script = parse_script("Do { Write-Output $x } WhIlE ($x -gT 0)");
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"do {\n    Write-Output $x\n} while ($x -gT 0)\n");
}

#[test]
fn diagnostic_render_of_empty_do_until_omits_blank_line() {
    let script = parse_script("DO {} UnTiL ($x -eQ 0)");
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"do {\n} until ($x -eQ 0)\n");
}

#[test]
fn diagnostic_render_of_parsed_loops_canonicalizes_layout_and_preserves_guard_spelling() {
    let script = parse_script(concat!(
        "\u{feff}# loop example\r\n",
        "FuNcTiOn Count($n){\r\n",
        "  WhIlE ($n -gT 0){ # outer loop\r\n",
        "    dO { cOnTiNuE } UnTiL ($n -eQ 1)\r\n",
        "   BrEaK\r\n",
        " }\r\n",
        "   DO {} wHiLe ($n)\r\n",
        "  Write-Output   $n\r\n",
        "}\r\n",
    ));
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(
        bytes,
        concat!(
            "function Count($n) {\n",
            "    while ($n -gT 0) {\n",
            "        do {\n",
            "            continue\n",
            "        } until ($n -eQ 1)\n",
            "        break\n",
            "    }\n",
            "    do {\n",
            "    } while ($n)\n",
            "    Write-Output $n\n",
            "}\n",
        )
        .as_bytes()
    );
}

#[test]
fn diagnostic_render_of_mutated_do_loop_uses_current_keyword_guard_and_body_order() {
    let mut script = parse_script("do { Write-Output $n; $n = 1 } while ($n -gT 0)");
    let NodeKind::Script(statements) = &mut script.kind else {
        panic!("expected a script root, got {:?}", script.kind);
    };
    let NodeKind::DoWhile {
        body,
        condition,
        until,
    } = &mut statements[0].kind
    else {
        panic!("expected a do loop first, got {:?}", statements[0].kind);
    };
    let NodeKind::Binary { right, .. } = &mut condition.kind else {
        panic!("expected a comparison guard, got {:?}", condition.kind);
    };
    let NodeKind::Script(body_statements) = &mut body.kind else {
        panic!("expected a script body, got {:?}", body.kind);
    };
    // Leave every node's span and token range intact so only the edited AST dictates the output.
    *until = !*until;
    right.kind = NodeKind::Number("100".into());
    body_statements.swap(0, 1);
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(
        bytes,
        b"do {\n    $n = 1\n    Write-Output $n\n} until ($n -gT 100)\n"
    );
}

#[test]
fn diagnostic_render_of_parsed_break_with_target_refuses_without_appending() {
    let script = parse_script("$n = 1\nwhile ($n) { break outer }");
    let mut bytes = b"existing diagnostic\n".to_vec();

    let error = write_diagnostic_script(&script, &mut bytes).unwrap_err();

    assert_eq!(
        (
            error.kind(),
            error
                .get_ref()
                .and_then(|source| source.downcast_ref::<DiagnosticRenderError>()),
            bytes.as_slice(),
        ),
        (
            ErrorKind::InvalidInput,
            Some(&DiagnosticRenderError::FieldNotSupported {
                label: "FlowStatement",
                field: "value",
            }),
            &b"existing diagnostic\n"[..],
        )
    );
}

#[test]
fn diagnostic_render_of_root_if_preserves_statement_order() {
    let script = parse_script("$x = 1\nif ($x) { Write-Output $x }");
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"$x = 1\nif ($x) {\n    Write-Output $x\n}\n");
}

#[test]
fn diagnostic_render_of_function_with_empty_if_branch_aligns_braces() {
    let script = parse_script("function F { if (1) {} }");
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"function F {\n    if (1) {\n    }\n}\n");
}

#[test]
fn diagnostic_render_of_parsed_conditional_canonicalizes_keywords_and_layout() {
    let script = parse_script(concat!(
        "\u{feff}# conditional example\r\n",
        "FuNcTiOn Scale($n) {\r\n",
        "  IF ($n -gT 0){ # positive\r\n",
        "    ReTuRn $n * 2\r\n",
        "} eLsEiF ($n -eq 0) { ReTuRn } eLsE { # empty\r\n",
        "}\r\n",
        "}\r\n",
    ));
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(
        bytes,
        concat!(
            "function Scale($n) {\n",
            "    if ($n -gT 0) {\n",
            "        return $n * 2\n",
            "    } elseif ($n -eq 0) {\n",
            "        return\n",
            "    } else {\n",
            "    }\n",
            "}\n",
        )
        .as_bytes()
    );
}

#[test]
fn diagnostic_render_of_mutated_conditional_uses_current_values_and_paired_arm_order() {
    let mut script = parse_script("if ($n -gt 0) { return 1 } elseif ($n -eq 0) { return 2 }");
    let NodeKind::Script(statements) = &mut script.kind else {
        panic!("expected a script root, got {:?}", script.kind);
    };
    let NodeKind::If {
        conditions, blocks, ..
    } = &mut statements[0].kind
    else {
        panic!("expected a conditional first, got {:?}", statements[0].kind);
    };
    let NodeKind::Binary { right, .. } = &mut conditions[0].kind else {
        panic!("expected a comparison guard, got {:?}", conditions[0].kind);
    };
    let NodeKind::Script(first_branch) = &mut blocks[0].kind else {
        panic!("expected a script body, got {:?}", blocks[0].kind);
    };
    let NodeKind::Flow {
        value: Some(value), ..
    } = &mut first_branch[0].kind
    else {
        panic!("expected a valued return, got {:?}", first_branch[0].kind);
    };
    // Keep the old coordinates and move each edited arm as a guard/body pair.
    right.kind = NodeKind::Number("100".into());
    value.kind = NodeKind::Number("400".into());
    conditions.swap(0, 1);
    blocks.swap(0, 1);
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(
        bytes,
        b"if ($n -eq 0) {\n    return 2\n} elseif ($n -gt 100) {\n    return 400\n}\n"
    );
}

#[test]
fn diagnostic_render_of_nested_functions_canonicalizes_parameters_and_indentation() {
    let script = parse_script(
        "\u{feff}# demo\r\nparam($Top)\r\n\
         function Get-Msg([String]$s='ż''ółć', $n=1+2) { \
         function Show-One { param($v); Write-Output $v | Sort-Object }; $x=$n; Show-One $s }\r\n",
    );
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

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
}

#[test]
fn diagnostic_render_of_mutated_function_uses_edited_names_and_order() {
    let mut script = parse_script("function Old($a, $b=2) { $a; $b }");
    let NodeKind::Script(statements) = &mut script.kind else {
        panic!("expected a script root, got {:?}", script.kind);
    };
    let NodeKind::Function {
        name,
        parameters,
        body,
        ..
    } = &mut statements[0].kind
    else {
        panic!("expected a function first, got {:?}", statements[0].kind);
    };
    let NodeKind::Parameter {
        name: first_parameter_name,
        ..
    } = &mut parameters[0].kind
    else {
        panic!("expected a parameter first, got {:?}", parameters[0].kind);
    };
    let NodeKind::Script(body_statements) = &mut body.kind else {
        panic!("expected a script body, got {:?}", body.kind);
    };
    // Leave coordinates and references untouched so the renderer must use the edited tree.
    *name = "New".into();
    *first_parameter_name = "$first".into();
    parameters.swap(0, 1);
    body_statements.swap(0, 1);
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(
        bytes,
        b"function New($b = 2, $first) {\n    $b\n    $a\n}\n"
    );
}

#[test]
fn diagnostic_render_of_workflow_with_empty_header_uses_function_without_parentheses() {
    let script = parse_script("workflow F() {}");
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"function F {\n}\n");
}

#[test]
fn diagnostic_render_of_empty_function_without_header_omits_parentheses() {
    let script = parse_script("function F {}");
    let mut bytes = Vec::new();

    write_diagnostic_script(&script, &mut bytes).unwrap();

    assert_eq!(bytes, b"function F {\n}\n");
}

#[test]
fn diagnostic_render_of_filter_refuses_with_typed_field_error() {
    let script = parse_script("filter F {}");
    let mut bytes = Vec::new();

    let error = write_diagnostic_script(&script, &mut bytes).unwrap_err();

    assert_eq!(
        (
            error.kind(),
            error
                .get_ref()
                .and_then(|source| source.downcast_ref::<DiagnosticRenderError>())
        ),
        (
            ErrorKind::InvalidInput,
            Some(&DiagnosticRenderError::FieldNotSupported {
                label: "FunctionDefinition",
                field: "filter",
            })
        )
    );
}
