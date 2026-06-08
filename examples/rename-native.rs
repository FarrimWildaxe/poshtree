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
//!
//! A C# type or method declared in `Add-Type` is used from PowerShell as
//! ordinary syntax (`[Win32]`, `[Win32]::M`, `New-Object Win32`, `[Win32]$x`).
//! [`poshtree::v2::csharp::rename_type`] and
//! [`poshtree::v2::csharp::rename_member`] rewrite the C# declaration, its uses
//! inside the C#, and every PowerShell call site in one set of edits, applied
//! through [`poshtree::v2::apply_edits`]. C# is case-sensitive and PowerShell
//! is not, so PowerShell sites match without regard to case and are rewritten
//! to the new name. A member reached through an unknown receiver (`$obj.M`) is
//! left alone, since its type cannot be known from one file.

use std::process::ExitCode;

use poshtree::v2::csharp::{rename_member, rename_type};
use poshtree::v2::{apply_edits, parse};

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
        let (output, changed) = apply_rename(DEMO_SCRIPT, &demo);
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

    let (output, changed) = apply_rename(&source, &rename);
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
fn apply_rename(src: &str, rename: &Rename) -> (String, usize) {
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
    // The renamer returns one non-overlapping edit per occurrence, so
    // apply_edits cannot fail here; keep the input if that ever changes.
    let output = apply_edits(src, &edits).unwrap_or_else(|_| src.to_string());
    (output, changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renames_a_type_in_csharp_and_every_powershell_site() {
        let (out, changed) = apply_rename(
            DEMO_SCRIPT,
            &Rename::Type {
                from: "Win32".to_string(),
                to: "NativeApi".to_string(),
            },
        );
        // C# declaration plus three PowerShell sites.
        assert_eq!(changed, 4);
        assert!(out.contains("public class NativeApi"));
        assert!(out.contains("[NativeApi]::MessageBox"));
        assert!(out.contains("New-Object NativeApi"));
        assert!(out.contains("[NativeApi]$typed"));
    }

    #[test]
    fn renames_a_member_and_its_static_call_site_only() {
        let (out, _) = apply_rename(
            DEMO_SCRIPT,
            &Rename::Member {
                type_name: "Win32".to_string(),
                from: "MessageBox".to_string(),
                to: "ShowMessage".to_string(),
            },
        );
        assert!(out.contains("extern int ShowMessage("));
        assert!(out.contains("[Win32]::ShowMessage("));
        assert!(out.contains("public class Win32")); // the type name is untouched
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
