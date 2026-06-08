//! Report the C# declared in a script's `Add-Type` blocks, using the `csharp`
//! front-end.
//!
//! ```text
//! cargo run --features csharp --example pinvoke-report -- script.ps1
//! cat script.ps1 | cargo run --features csharp --example pinvoke-report -- -
//! cargo run --features csharp --example pinvoke-report
//! ```
//!
//! The program parses the PowerShell with [`poshtree::v2::parse`], finds each
//! `Add-Type` block, and parses the embedded C# with
//! [`poshtree::v2::csharp::csharp_unit`]. For each block it prints the
//! `[DllImport]` signatures from [`poshtree::v2::csharp::csharp_imports`] (the
//! library, return type, name, and typed parameters) and the types, methods,
//! and fields the C# declares, from [`poshtree::v2::csharp::csharp_symbols`].
//! It only reads the script; nothing is rewritten.

use std::process::ExitCode;

use poshtree::v2::csharp::resolve::DeclKind;
use poshtree::v2::csharp::{csharp_imports, csharp_symbols, csharp_unit};
use poshtree::v2::{parse, CSharpImport, NodeKind};

const DEMO: &str = "\
Add-Type -TypeDefinition @'
public class Native {
    [DllImport(\"user32.dll\")]
    public static extern int MessageBox(IntPtr hWnd, string text, string caption, uint type);

    [DllImport(\"kernel32.dll\")]
    public static extern uint GetLastError();
}
'@

[Native]::MessageBox(0, \"hi\", \"title\", 0)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source = match load_source(&args) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("pinvoke-report: {err}");
            return ExitCode::FAILURE;
        }
    };

    let report = report(&source);
    if report.is_empty() {
        eprintln!("pinvoke-report: no Add-Type blocks found");
    } else {
        print!("{report}");
    }
    ExitCode::SUCCESS
}

fn load_source(args: &[String]) -> std::io::Result<String> {
    match args.first().map(String::as_str) {
        None => Ok(DEMO.to_string()),
        Some("-") => {
            use std::io::Read;
            let mut buffer = String::new();
            std::io::stdin().read_to_string(&mut buffer)?;
            Ok(buffer)
        }
        Some(path) => std::fs::read_to_string(path),
    }
}

/// Builds the report text for every `Add-Type` block in `src`. Returns an empty
/// string when there are none.
fn report(src: &str) -> String {
    let parsed = parse(src);
    let mut out = String::new();
    let mut block = 0usize;

    parsed.script.walk(&mut |node| {
        // The C# lives on the CSharpMemberDef node; match it directly so a
        // block is not counted twice (its parent command also carries the C#).
        if !matches!(node.kind, NodeKind::CSharpMemberDef(_)) {
            return;
        }
        let Some(unit) = csharp_unit(node, src) else {
            return;
        };
        block += 1;
        out.push_str(&format!("Add-Type block {block}\n"));

        out.push_str("  P/Invoke signatures:\n");
        let imports = csharp_imports(&unit, src);
        if imports.is_empty() {
            out.push_str("    (none)\n");
        } else {
            for imp in &imports {
                out.push_str(&format!("    {}  {}\n", imp.dll, signature(imp)));
            }
        }

        out.push_str("  declared:\n");
        let mut printed = false;
        for sym in csharp_symbols(&unit) {
            if let Some(label) = api_label(sym.kind) {
                out.push_str(&format!("    {label:<9}{}\n", sym.name));
                printed = true;
            }
        }
        if !printed {
            out.push_str("    (none)\n");
        }
    });
    out
}

/// Renders one import as `returns name(type name, ...)`.
fn signature(imp: &CSharpImport) -> String {
    let params = imp
        .params
        .iter()
        .map(|p| format!("{} {}", p.type_name, p.name))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} {}({params})", imp.returns, imp.function)
}

/// The label for an API-level declaration, or `None` for parameters and locals,
/// which a signature report has no reason to list.
fn api_label(kind: DeclKind) -> Option<&'static str> {
    match kind {
        DeclKind::Namespace => Some("namespace"),
        DeclKind::Type => Some("type"),
        DeclKind::Method => Some("method"),
        DeclKind::Ctor => Some("ctor"),
        DeclKind::Property => Some("property"),
        DeclKind::Field => Some("field"),
        DeclKind::EnumMember => Some("enum"),
        DeclKind::Param | DeclKind::Local => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_both_pinvoke_signatures() {
        let r = report(DEMO);
        assert!(r.contains(
            "user32.dll  int MessageBox(IntPtr hWnd, string text, string caption, uint type)"
        ));
        assert!(r.contains("kernel32.dll  uint GetLastError()"));
    }

    #[test]
    fn lists_the_type_and_its_methods() {
        let r = report(DEMO);
        assert!(r.contains("type     Native"));
        assert!(r.contains("method   MessageBox"));
        assert!(r.contains("method   GetLastError"));
    }

    #[test]
    fn a_script_without_add_type_reports_nothing() {
        assert_eq!(report("Get-Process | Sort-Object CPU\n"), "");
    }

    #[test]
    fn a_block_with_no_pinvoke_still_lists_declarations() {
        let src = "Add-Type -TypeDefinition @'\npublic class Bag { public int Count; }\n'@\n";
        let r = report(src);
        assert!(r.contains("P/Invoke signatures:\n    (none)"));
        assert!(r.contains("type     Bag"));
        assert!(r.contains("field    Count"));
    }
}
