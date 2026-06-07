//! Capitalize PowerShell command names to PascalCase, using only the v2 layer.
//!
//! ```text
//! cargo run --example pascalize -- script.ps1   # rewrite a file to stdout
//! cat script.ps1 | cargo run --example pascalize -- -   # read stdin
//! cargo run --example pascalize                 # run the built-in demo
//! ```
//!
//! The program lexes and parses with the native v2 parser
//! ([`poshtree::v2::parse`]), walks the tree for command names, and rewrites
//! each identifier-like name so every `-`-separated segment starts with an
//! uppercase letter (`get-childitem` -> `Get-Childitem`). Edits are applied
//! through [`poshtree::v2::apply_edits`], which works on byte spans, so only
//! the command-name tokens change: comments, strings, arguments, parameters,
//! and the original layout are preserved exactly.
//!
//! Only command *invocations* are touched. Function-definition names, call
//! operator targets (`& $cmd`), dot-sourced paths (`. .\x.ps1`), and anything
//! that is not a plain identifier are left alone.

use std::process::ExitCode;

use poshtree::v2::{apply_edits, parse, NodeKind, TextEdit};

const DEMO: &str = "\
get-childitem -Path . -recurse |
    where-object { $_.Length -gt 1kb } |
    sort-object Length |
    select-object -First 5

function get-greeting {
    param([string]$name)
    write-output \"hello, $name\"   # the string is left untouched
}

& $external --raw                   # call-operator target: skipped
. .\\profile.ps1                    # dot-sourced path: skipped
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let source = match load_source(&args) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("pascalize: {err}");
            return ExitCode::FAILURE;
        }
    };

    let (output, changed) = pascalize(&source);
    print!("{output}");
    eprintln!("pascalize: capitalized {changed} command name(s)");
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

/// Rewrites command names in `src` to PascalCase, returning the new source and
/// the number of names that changed.
fn pascalize(src: &str) -> (String, usize) {
    let parsed = parse(src);

    let mut edits: Vec<TextEdit> = Vec::new();
    parsed.script.walk(&mut |node| {
        if let NodeKind::Command { name, .. } = &node.kind {
            if let NodeKind::BareWord(text) = &name.kind {
                if let Some(cased) = pascal_case(text) {
                    if cased != *text {
                        edits.push(TextEdit::replace(name.span, cased));
                    }
                }
            }
        }
    });

    let changed = edits.len();
    // Each command name is a distinct token, so the spans never overlap and
    // apply_edits cannot fail here; keep the original text if that ever changes.
    let output = apply_edits(src, &edits).unwrap_or_else(|_| src.to_string());
    (output, changed)
}

/// PascalCases a plain-identifier command name: every `-`-separated segment
/// gets an uppercase first letter and the rest is left as written. Returns
/// `None` for names that are not plain identifiers (paths, globs, or anything
/// containing `.`, `/`, `\`, or `:`), so those are left alone.
fn pascal_case(name: &str) -> Option<String> {
    let starts_with_letter = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
    let identifier_like = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !starts_with_letter || !identifier_like {
        return None;
    }
    Some(
        name.split('-')
            .map(capitalize_first)
            .collect::<Vec<_>>()
            .join("-"),
    )
}

fn capitalize_first(segment: &str) -> String {
    let mut chars = segment.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_each_segment() {
        assert_eq!(
            pascal_case("get-childitem").as_deref(),
            Some("Get-Childitem")
        );
        assert_eq!(pascal_case("write-host").as_deref(), Some("Write-Host"));
        assert_eq!(pascal_case("echo").as_deref(), Some("Echo"));
    }

    #[test]
    fn leaves_already_cased_names_unchanged() {
        assert_eq!(
            pascal_case("Get-ChildItem").as_deref(),
            Some("Get-ChildItem")
        );
    }

    #[test]
    fn skips_non_identifiers() {
        assert_eq!(pascal_case(".\\profile.ps1"), None);
        assert_eq!(pascal_case("C:\\tmp\\x"), None);
        assert_eq!(pascal_case("*.txt"), None);
        assert_eq!(pascal_case(""), None);
    }

    #[test]
    fn rewrites_commands_but_not_strings_or_params() {
        let src = "get-process -Name \"get-thing\" | where-object { $_.Id }\n";
        let (out, changed) = pascalize(src);
        assert_eq!(changed, 2); // get-process, where-object
        assert!(out.contains("Get-Process"));
        assert!(out.contains("Where-Object"));
        assert!(out.contains("\"get-thing\"")); // the string argument is untouched
        assert!(out.contains("-Name")); // the parameter is untouched
    }

    #[test]
    fn output_is_idempotent() {
        let src = "get-childitem | sort-object\n";
        let (once, _) = pascalize(src);
        let (twice, second_pass) = pascalize(&once);
        assert_eq!(once, twice);
        assert_eq!(second_pass, 0);
    }
}
