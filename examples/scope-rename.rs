//! Scope-aware alpha-rename of a PowerShell script, driven entirely off the AST.
//!
//! Every user variable becomes `$name_var{N}` and every user-defined function
//! becomes `name_func{N}`, numbered in source order. A variable is identified by
//! its scope (the script, or a function body), so the script's `$LogDir` and the
//! function's `$Path` stay distinct even though `$LogDir` is the value passed to
//! `-Path`.
//!
//! Two pieces that earlier versions hand-rolled are gone:
//!
//! * The function-definition rename used to scan the token stream for the
//!   identifier after the `function` keyword. It now reads `Function.name_span`
//!   straight from the AST node.
//! * The "leave builtins alone" check used to carry a hand-maintained reserved
//!   list, and stripped the sigil and scope off a token by hand. It now asks
//!   `is_automatic_variable` / `is_preference_variable`, and `variable_name`
//!   returns the bare name.
//!
//! Run it with `cargo run --example scope-rename --features v2`.

use poshtree::v2::{
    apply_edits, is_automatic_variable, is_preference_variable, parse, variable_name,
    variable_scope, Node, NodeKind, Span, TextEdit,
};
use std::collections::HashMap;

const SCRIPT: &str = r#"param(
    [string]$LogDir = 'C:\Logs',
    [int]$KeepDays = 30
)

function Remove-OldLogs {
    param([string]$Path, [int]$Days)

    if (-not (Test-Path -Path $Path)) {
        Write-Host "Log directory $Path does not exist; creating it."
        New-Item -Path $Path -ItemType Directory | Out-Null
        return
    }

    $cutoff = (Get-Date).AddDays(-$Days)
    $old = Get-ChildItem -Path $Path -Filter '*.log' |
        Where-Object { $_.LastWriteTime -lt $cutoff }

    foreach ($file in $old) {
        Write-Host "Removing $($file.Name)"
        Remove-Item -Path $file.FullName -Force
    }

    Write-Host "Removed $($old.Count) file(s)."
}

Remove-OldLogs -Path $LogDir -Days $KeepDays
"#;

/// A function body as a named scope region (the whole function node span).
struct Scope {
    name: String,
    region: Span,
}

/// `None` is the script scope; `Some(i)` is the i-th function in `scopes`.
type ScopeId = Option<usize>;

/// Everything a caller needs to report or apply the rename.
struct Rename {
    renamed: String,
    edit_count: usize,
    scopes: Vec<Scope>,
    var_order: Vec<(ScopeId, String)>,
    var_names: HashMap<(ScopeId, String), String>,
    fn_order: Vec<String>,
    fn_names: HashMap<String, String>,
}

fn rename(src: &str) -> Rename {
    let out = parse(src);
    let scopes = collect_scopes(&out.script);

    let mut var_names = HashMap::new();
    let mut var_order = Vec::new();
    register_variables(&out.script, &scopes, &mut var_names, &mut var_order);

    let mut fn_names = HashMap::new();
    let mut fn_order = Vec::new();
    register_functions(&out.script, &mut fn_names, &mut fn_order);

    let mut edits = Vec::new();
    edits.extend(variable_edits(&out.script, &scopes, &var_names));
    edits.extend(function_edits(&out.script, &fn_names));

    let renamed = apply_edits(src, &edits).expect("rename edits do not overlap");
    Rename {
        edit_count: edits.len(),
        renamed,
        scopes,
        var_order,
        var_names,
        fn_order,
        fn_names,
    }
}

/// Each function's span, as a named scope region.
fn collect_scopes(script: &Node) -> Vec<Scope> {
    let mut scopes = Vec::new();
    script.walk(&mut |node| {
        if let NodeKind::Function { name, .. } = &node.kind {
            scopes.push(Scope {
                name: name.clone(),
                region: node.span,
            });
        }
    });
    scopes
}

/// The innermost scope region that contains `span`, or the script scope.
fn scope_of(span: Span, scopes: &[Scope]) -> ScopeId {
    let mut best: ScopeId = None;
    for (i, scope) in scopes.iter().enumerate() {
        let contains = scope.region.start <= span.start && span.end <= scope.region.end;
        let inner = match best {
            Some(b) => scope.region.start > scopes[b].region.start,
            None => true,
        };
        if contains && inner {
            best = Some(i);
        }
    }
    best
}

/// Whether a variable token names a user variable a rename should touch. The
/// classifier replaces the old hand-maintained reserved list.
fn is_user_variable(raw: &str) -> bool {
    !is_automatic_variable(raw) && !is_preference_variable(raw) && !variable_name(raw).is_empty()
}

/// Assigns a new name to each distinct `(scope, variable)`, in source order.
fn register_variables(
    script: &Node,
    scopes: &[Scope],
    names: &mut HashMap<(ScopeId, String), String>,
    order: &mut Vec<(ScopeId, String)>,
) {
    script.walk(&mut |node| {
        let NodeKind::Variable(raw) = &node.kind else {
            return;
        };
        if !is_user_variable(raw) {
            return;
        }
        let bare = variable_name(raw);
        let scope = scope_of(node.span, scopes);
        names
            .entry((scope, bare.to_ascii_lowercase()))
            .or_insert_with(|| {
                let new = format!("name_var{}", order.len());
                order.push((scope, bare.to_string()));
                new
            });
    });
}

/// One edit per `Variable` node. The 0.2.3 interpolation-span fix means every
/// reference span is trustworthy, including those inside `"$( ... )"`, so this
/// edits spans directly with no token rewriting or span guard.
fn variable_edits(
    script: &Node,
    scopes: &[Scope],
    names: &HashMap<(ScopeId, String), String>,
) -> Vec<TextEdit> {
    let mut edits = Vec::new();
    script.walk(&mut |node| {
        let NodeKind::Variable(raw) = &node.kind else {
            return;
        };
        if !is_user_variable(raw) {
            return;
        }
        let bare = variable_name(raw);
        let scope = scope_of(node.span, scopes);
        if let Some(new) = names.get(&(scope, bare.to_ascii_lowercase())) {
            edits.push(rewrite_variable(raw, node.span, new));
        }
    });
    edits
}

/// Rewrites a variable token's bare name while keeping its sigil and any scope:
/// `$cutoff` -> `$new`, `$script:Config` -> `$script:new`, `@splat` -> `@new`.
fn rewrite_variable(raw: &str, span: Span, new_bare: &str) -> TextEdit {
    let sigil = if raw.starts_with('@') { '@' } else { '$' };
    let replacement = match variable_scope(raw) {
        Some(scope) => format!("{sigil}{scope}:{new_bare}"),
        None => format!("{sigil}{new_bare}"),
    };
    TextEdit::replace(span, replacement)
}

/// Numbers each user-defined function, in source order.
fn register_functions(script: &Node, names: &mut HashMap<String, String>, order: &mut Vec<String>) {
    script.walk(&mut |node| {
        let NodeKind::Function {
            name, name_span, ..
        } = &node.kind
        else {
            return;
        };
        if name.is_empty() || name_span.start == name_span.end {
            return;
        }
        names.entry(name.to_ascii_lowercase()).or_insert_with(|| {
            let new = format!("name_func{}", order.len());
            order.push(name.clone());
            new
        });
    });
}

/// Renames function definitions (via `name_span`) and call sites (the command
/// name, a `BareWord` node that carries its own span).
fn function_edits(script: &Node, names: &HashMap<String, String>) -> Vec<TextEdit> {
    let mut edits = Vec::new();
    script.walk(&mut |node| match &node.kind {
        NodeKind::Function {
            name, name_span, ..
        } => {
            if name_span.start == name_span.end {
                return;
            }
            if let Some(new) = names.get(&name.to_ascii_lowercase()) {
                edits.push(TextEdit::replace(*name_span, new.clone()));
            }
        }
        NodeKind::Command { name, .. } => {
            let NodeKind::BareWord(word) = &name.kind else {
                return;
            };
            if let Some(new) = names.get(&word.to_ascii_lowercase()) {
                edits.push(TextEdit::replace(name.span, new.clone()));
            }
        }
        _ => {}
    });
    edits
}

/// A human-readable view of the name mapping.
fn mapping_report(r: &Rename) -> String {
    let mut out = String::from("variables (by scope):\n");
    for scope in std::iter::once(None).chain((0..r.scopes.len()).map(Some)) {
        let in_scope: Vec<_> = r.var_order.iter().filter(|(s, _)| *s == scope).collect();
        if in_scope.is_empty() {
            continue;
        }
        let label = match scope {
            None => "script".to_string(),
            Some(i) => format!("function {}", r.scopes[i].name),
        };
        out.push_str(&format!("  [{label}]\n"));
        for (s, display) in in_scope {
            let new = &r.var_names[&(*s, display.to_ascii_lowercase())];
            out.push_str(&format!("    ${display} -> ${new}\n"));
        }
    }
    out.push_str("functions:\n");
    for display in &r.fn_order {
        let new = &r.fn_names[&display.to_ascii_lowercase()];
        out.push_str(&format!("    {display} -> {new}\n"));
    }
    out
}

#[cfg(not(test))]
fn main() {
    let r = rename(SCRIPT);
    print!("{}", mapping_report(&r));
    println!("\n=== before ===\n{SCRIPT}");
    println!("=== after ({} edits) ===\n{}", r.edit_count, r.renamed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_is_scope_aware_and_skips_builtins() {
        let r = rename(SCRIPT);
        let report = mapping_report(&r);

        // Same edit count as the hand-rolled version this replaces.
        assert_eq!(r.edit_count, 21);

        // Script-scope variables are numbered before the function's.
        assert!(report.contains("$LogDir -> $name_var0"));
        assert!(report.contains("$KeepDays -> $name_var1"));

        // The function's own scope: distinct numbers, in source order.
        assert!(report.contains("$Path -> $name_var2"));
        assert!(report.contains("$Days -> $name_var3"));
        assert!(report.contains("$cutoff -> $name_var4"));
        assert!(report.contains("$old -> $name_var5"));
        assert!(report.contains("$file -> $name_var6"));

        // The function definition and its call site share the new name.
        assert!(report.contains("Remove-OldLogs -> name_func0"));

        // The automatic $_ is untouched; every user identifier is gone.
        assert!(r.renamed.contains("$_"));
        for original in [
            "$LogDir",
            "$KeepDays",
            "$Path",
            "$Days",
            "$cutoff",
            "$old",
            "$file",
            "Remove-OldLogs",
        ] {
            assert!(!r.renamed.contains(original), "{original} should be gone");
        }
    }

    #[test]
    fn function_definition_and_call_both_rename() {
        // Two occurrences of the name (definition + one call), both rewritten.
        let count = SCRIPT.matches("Remove-OldLogs").count();
        assert_eq!(count, 2);
        let r = rename(SCRIPT);
        assert_eq!(r.renamed.matches("name_func0").count(), 2);
    }
}
