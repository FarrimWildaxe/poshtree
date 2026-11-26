//! Rename a C# type or member across an `Add-Type` block and its PowerShell
//! uses at once, using the `csharp` front-end.
//!
//! ```text
//! cargo run --features csharp --example rename-native -- type Win32 NativeApi < script.ps1
//! cat script.ps1 | cargo run --features csharp --example rename-native -- member Win32 MessageBox ShowMessage
//! cargo run --features csharp --example rename-native
//! ```
//!
//! With no arguments the program runs a built-in demo. Otherwise it reads a
//! PowerShell script from stdin and rewrites it to stdout: `type FROM TO`
//! renames a C# type, and `member TYPE FROM TO` renames a member of that type.
//! If applying edits fails, either mode reports the error to stderr and exits
//! unsuccessfully without printing source or a success count.
//!
//! A C# type or method declared in `Add-Type` is used from PowerShell as
//! ordinary syntax (`[Win32]`, `[Win32]::M`, `New-Object Win32`, `[Win32]$x`).
//! [`poshtree::v2::csharp::rename_type`] and
//! [`poshtree::v2::csharp::rename_member`] collect resolver-selected C# references
//! and recognized PowerShell name sites in one set of edits, applied to the
//! original source through [`poshtree::v2::apply_edits`].
//!
//! Matching is bounded and syntactic, not a semantic binder: C# names and owners
//! are case-sensitive, while PowerShell comparisons ignore ASCII case only
//! (no Unicode case folding or normalization). PowerShell matches do not require
//! a C# declaration. Static member access/calls use stored literal name spans;
//! eligible names are nonempty and contain only Unicode alphanumeric characters
//! or `_`, including digit-initial Generic tokens and Keyword tokens, but not
//! Number tokens. Parser-recognized comments and backtick continuations around
//! `::` are preserved, not newly accepted grammar. Instance (`$obj.M`), dynamic,
//! quoted, and computed member names are left alone.
//!
//! C# resolution is per embedded unit with simple-name ownership and bounded
//! receiver/overload support. Neither helper checks replacement-name validity or
//! collisions; edit-range/overlap validation does not prove semantic safety.

use std::process::ExitCode;

use poshtree::v2::csharp::{rename_member, rename_type};
use poshtree::v2::{apply_edits, parse, EditError, TextEdit};

const DEMO_SCRIPT: &str = "\
Add-Type -TypeDefinition @'
public class Win32 {
    [DllImport(\"user32.dll\")]
    public static extern int MessageBox(IntPtr hWnd, string text, string caption, uint type);
}
'@

[Win32]::MessageBox(0, \"Hello\", \"Greeting\", 0)
$handle = New-Object Win32
[Win32]$typed = $handle
";

const USAGE: &str = "\
usage:
  rename-native type <From> <To>
  rename-native member <Type> <From> <To>
reads a PowerShell script from stdin; with no arguments it runs a demo";

/// What to rename.
enum Rename {
    Type {
        from: String,
        to: String,
    },
    Member {
        type_name: String,
        from: String,
        to: String,
    },
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        let demo = Rename::Type {
            from: "Win32".to_string(),
            to: "NativeApi".to_string(),
        };
        let (output, changed) = match apply_rename(DEMO_SCRIPT, &demo) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("rename-native: {err}");
                return ExitCode::FAILURE;
            }
        };
        print!("{output}");
        eprintln!("rename-native: rewrote {changed} occurrence(s) of Win32 -> NativeApi");
        return ExitCode::SUCCESS;
    }

    let rename = match parse_args(&args) {
        Ok(rename) => rename,
        Err(err) => {
            eprintln!("rename-native: {err}\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let source = match read_stdin() {
        Ok(text) => text,
        Err(err) => {
            eprintln!("rename-native: {err}");
            return ExitCode::FAILURE;
        }
    };

    let (output, changed) = match apply_rename(&source, &rename) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("rename-native: {err}");
            return ExitCode::FAILURE;
        }
    };
    print!("{output}");
    eprintln!("rename-native: rewrote {changed} occurrence(s)");
    ExitCode::SUCCESS
}

fn parse_args(args: &[String]) -> Result<Rename, String> {
    match args[0].as_str() {
        "type" => match args {
            [_, from, to] => Ok(Rename::Type {
                from: from.clone(),
                to: to.clone(),
            }),
            _ => Err("`type` takes <From> <To>".to_string()),
        },
        "member" => match args {
            [_, type_name, from, to] => Ok(Rename::Member {
                type_name: type_name.clone(),
                from: from.clone(),
                to: to.clone(),
            }),
            _ => Err("`member` takes <Type> <From> <To>".to_string()),
        },
        other => Err(format!(
            "unknown command {other:?}; expected `type` or `member`"
        )),
    }
}

fn read_stdin() -> std::io::Result<String> {
    use std::io::Read;
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer)
}

/// Applies one rename to `src`, returning the new source and how many spans
/// changed.
///
/// Returns an error if the collected edits cannot be applied.
fn apply_rename(src: &str, rename: &Rename) -> Result<(String, usize), EditError> {
    apply_rename_with_apply(src, rename, apply_edits)
}

fn apply_rename_with_apply(
    src: &str,
    rename: &Rename,
    apply: impl FnOnce(&str, &[TextEdit]) -> Result<String, EditError>,
) -> Result<(String, usize), EditError> {
    let parsed = parse(src);
    let edits = match rename {
        Rename::Type { from, to } => rename_type(&parsed.script, src, from, to),
        Rename::Member {
            type_name,
            from,
            to,
        } => rename_member(&parsed.script, src, type_name, from, to),
    };
    let changed = edits.len();
    let output = apply(src, &edits)?;
    Ok((output, changed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renames_a_type_in_csharp_and_every_powershell_site() {
        let rename = Rename::Type {
            from: "Win32".to_string(),
            to: "NativeApi".to_string(),
        };

        let result = apply_rename(DEMO_SCRIPT, &rename);

        assert_eq!(
            result,
            Ok((
                "\
Add-Type -TypeDefinition @'
public class NativeApi {
    [DllImport(\"user32.dll\")]
    public static extern int MessageBox(IntPtr hWnd, string text, string caption, uint type);
}
'@

[NativeApi]::MessageBox(0, \"Hello\", \"Greeting\", 0)
$handle = New-Object NativeApi
[NativeApi]$typed = $handle
"
                .to_string(),
                4,
            ))
        );
    }

    #[test]
    fn renames_a_member_and_its_static_call_site_only() {
        let rename = Rename::Member {
            type_name: "Win32".to_string(),
            from: "MessageBox".to_string(),
            to: "ShowMessage".to_string(),
        };

        let result = apply_rename(DEMO_SCRIPT, &rename);

        assert_eq!(
            result,
            Ok((
                "\
Add-Type -TypeDefinition @'
public class Win32 {
    [DllImport(\"user32.dll\")]
    public static extern int ShowMessage(IntPtr hWnd, string text, string caption, uint type);
}
'@

[Win32]::ShowMessage(0, \"Hello\", \"Greeting\", 0)
$handle = New-Object Win32
[Win32]$typed = $handle
"
                .to_string(),
                2,
            ))
        );
    }

    #[test]
    fn apply_rename_without_matches_returns_unchanged_source_and_zero_changes() {
        let rename = Rename::Type {
            from: "Missing".to_string(),
            to: "NativeApi".to_string(),
        };

        let result = apply_rename(DEMO_SCRIPT, &rename);

        assert_eq!(result, Ok((DEMO_SCRIPT.to_string(), 0)));
    }

    #[test]
    fn apply_rename_preserves_unicode_and_untouched_text_when_renaming_a_type() {
        let src = "# café 🦀 Win32\nAdd-Type -TypeDefinition @'\n// naïve Win32\npublic class Win32 { public static string M() { return \"héllo, 世界 Win32\"; } }\n'@\n\n[Win32]::M() # 世界\nWrite-Output \"Win32 café\" -NoNewline\n";
        let rename = Rename::Type {
            from: "Win32".to_string(),
            to: "NativeApi".to_string(),
        };

        let result = apply_rename(src, &rename);

        assert_eq!(
            result,
            Ok((
                "# café 🦀 Win32\nAdd-Type -TypeDefinition @'\n// naïve Win32\npublic class NativeApi { public static string M() { return \"héllo, 世界 Win32\"; } }\n'@\n\n[NativeApi]::M() # 世界\nWrite-Output \"Win32 café\" -NoNewline\n".to_string(),
                2,
            ))
        );
    }

    #[test]
    fn apply_rename_without_csharp_renames_trivia_separated_unicode_static_sites_only() {
        let src = "\u{feff}# café 🦀\r\n[Win32]<# Méthod_2 #>::méTHOD_2\n[win32]::`\r\nMéthod_2()\n$obj.Méthod_2\n[Win32]::$Méthod_2()\n[Other]::Méthod_2()\n[Win32]::MÉthod_2()\n'Méthod_2'\n";
        let rename = Rename::Member {
            type_name: "Win32".to_string(),
            from: "Méthod_2".to_string(),
            to: "ShowMessage".to_string(),
        };

        let result = apply_rename(src, &rename);

        assert_eq!(
            result,
            Ok((
                "\u{feff}# café 🦀\r\n[Win32]<# Méthod_2 #>::ShowMessage\n[win32]::`\r\nShowMessage()\n$obj.Méthod_2\n[Win32]::$Méthod_2()\n[Other]::Méthod_2()\n[Win32]::MÉthod_2()\n'Méthod_2'\n".to_string(),
                2,
            ))
        );
    }

    #[test]
    fn apply_rename_with_digit_initial_unicode_generic_names_renames_access_and_call() {
        let src = "[Win32]::<# 猫 #>9猫_2\n[win32]::9猫_2()\n$obj.9猫_2\n";
        let rename = Rename::Member {
            type_name: "Win32".to_string(),
            from: "9猫_2".to_string(),
            to: "Renamed".to_string(),
        };

        let result = apply_rename(src, &rename);

        assert_eq!(
            result,
            Ok((
                "[Win32]::<# 猫 #>Renamed\n[win32]::Renamed()\n$obj.9猫_2\n".to_string(),
                2,
            ))
        );
    }

    #[test]
    fn apply_rename_with_embedded_csharp_renames_unicode_declaration_and_bound_sites() {
        let src = "\
# café 🦀 Méthod_2
Add-Type -TypeDefinition @'
public class Win32 {
    public static int Méthod_2() { return 1; } // Méthod_2
    public static int Call() { return Méthod_2(); }
    public static string Text = \"Méthod_2 世界\";
}
public class Other { public static int Méthod_2() { return 2; } }
'@
[Win32]::<# Méthod_2 #>méTHOD_2
[win32]::`
Méthod_2()
$obj.Méthod_2()
[Win32]::$Méthod_2()
[Other]::Méthod_2()
";
        let rename = Rename::Member {
            type_name: "Win32".to_string(),
            from: "Méthod_2".to_string(),
            to: "ShowMessage".to_string(),
        };

        let result = apply_rename(src, &rename);

        assert_eq!(
            result,
            Ok((
                "\
# café 🦀 Méthod_2
Add-Type -TypeDefinition @'
public class Win32 {
    public static int ShowMessage() { return 1; } // Méthod_2
    public static int Call() { return ShowMessage(); }
    public static string Text = \"Méthod_2 世界\";
}
public class Other { public static int Méthod_2() { return 2; } }
'@
[Win32]::<# Méthod_2 #>ShowMessage
[win32]::`
ShowMessage()
$obj.Méthod_2()
[Win32]::$Méthod_2()
[Other]::Méthod_2()
"
                .to_string(),
                4,
            ))
        );
    }

    #[test]
    fn apply_rename_with_apply_propagates_type_edit_error_without_output_or_count() {
        let rename = Rename::Type {
            from: "Win32".to_string(),
            to: "NativeApi".to_string(),
        };

        let result = apply_rename_with_apply(DEMO_SCRIPT, &rename, |_, _| {
            Err(EditError::Overlap {
                first: poshtree::v2::Span { start: 0, end: 4 },
                second: poshtree::v2::Span { start: 2, end: 6 },
            })
        });

        assert_eq!(
            result,
            Err(EditError::Overlap {
                first: poshtree::v2::Span { start: 0, end: 4 },
                second: poshtree::v2::Span { start: 2, end: 6 },
            })
        );
    }

    #[test]
    fn apply_rename_with_apply_propagates_member_edit_error_without_output_or_count() {
        let rename = Rename::Member {
            type_name: "Win32".to_string(),
            from: "MessageBox".to_string(),
            to: "ShowMessage".to_string(),
        };

        let result = apply_rename_with_apply(DEMO_SCRIPT, &rename, |_, _| {
            Err(EditError::InvalidRange(poshtree::v2::Span {
                start: 7,
                end: 2,
            }))
        });

        assert_eq!(
            result,
            Err(EditError::InvalidRange(poshtree::v2::Span {
                start: 7,
                end: 2,
            }))
        );
    }

    #[test]
    fn unknown_command_is_rejected() {
        let args = ["rename".to_string(), "a".to_string(), "b".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn member_needs_a_type_and_two_names() {
        let args = ["member".to_string(), "T".to_string(), "a".to_string()];
        assert!(parse_args(&args).is_err());
    }
}
