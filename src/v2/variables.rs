//! Classifying PowerShell variables, and splitting a variable token into its
//! scope and bare name.
//!
//! A refactor or linter that walks [`Variable`](crate::v2::NodeKind::Variable)
//! nodes usually needs to leave language builtins alone. The automatic
//! variables come from `about_Automatic_Variables`; the preference variables
//! from `about_Preference_Variables`. A rename normally skips both.
//!
//! ```
//! use poshtree::v2::{is_automatic_variable, variable_name, variable_scope};
//!
//! assert!(is_automatic_variable("$_"));
//! assert!(!is_automatic_variable("$LogDir"));
//! assert_eq!(variable_name("$script:Config"), "Config");
//! assert_eq!(variable_scope("$script:Config"), Some("script"));
//! ```

/// The automatic variables from `about_Automatic_Variables`, each without its
/// leading `$`. The punctuation forms `$_`, `$?`, `$$`, `$^` appear here as
/// `_`, `?`, `$`, `^`. The set grows across PowerShell releases; this tracks
/// PowerShell 7.x, including the constants `true`, `false`, and `null`.
pub const AUTOMATIC_VARIABLES: &[&str] = &[
    "$",
    "?",
    "^",
    "_",
    "args",
    "ConsoleFileName",
    "EnabledExperimentalFeatures",
    "Error",
    "Event",
    "EventArgs",
    "EventSubscriber",
    "ExecutionContext",
    "false",
    "foreach",
    "HOME",
    "Host",
    "input",
    "IsCoreCLR",
    "IsLinux",
    "IsMacOS",
    "IsWindows",
    "LASTEXITCODE",
    "Matches",
    "MyInvocation",
    "NestedPromptLevel",
    "null",
    "PID",
    "PROFILE",
    "PSBoundParameters",
    "PSCmdlet",
    "PSCommandPath",
    "PSCulture",
    "PSDebugContext",
    "PSHOME",
    "PSItem",
    "PSScriptRoot",
    "PSSenderInfo",
    "PSStyle",
    "PSUICulture",
    "PSVersionTable",
    "PWD",
    "Sender",
    "ShellId",
    "StackTrace",
    "switch",
    "this",
    "true",
];

/// The preference variables from `about_Preference_Variables`, each without its
/// leading `$`. A rename usually skips these alongside the automatics, since
/// renaming one changes behavior.
pub const PREFERENCE_VARIABLES: &[&str] = &[
    "ConfirmPreference",
    "DebugPreference",
    "ErrorActionPreference",
    "ErrorView",
    "FormatEnumerationLimit",
    "InformationPreference",
    "LogCommandHealthEvent",
    "LogCommandLifecycleEvent",
    "LogEngineHealthEvent",
    "LogEngineLifecycleEvent",
    "LogProviderHealthEvent",
    "LogProviderLifecycleEvent",
    "MaximumHistoryCount",
    "OFS",
    "OutputEncoding",
    "ProgressPreference",
    "PSDefaultParameterValues",
    "PSEmailServer",
    "PSModuleAutoLoadingPreference",
    "PSNativeCommandArgumentPassing",
    "PSNativeCommandUseErrorActionPreference",
    "PSSessionApplicationName",
    "PSSessionConfigurationName",
    "PSSessionOption",
    "Transcript",
    "VerbosePreference",
    "WarningPreference",
    "WhatIfPreference",
];

/// Strips a single splat `@`, a single `$`, and a surrounding `{ }` from a
/// variable token, leaving the scope-qualified name. The order matches how
/// PowerShell writes a token: an optional splat, the sigil, then the name.
fn strip_sigil(raw: &str) -> &str {
    let s = raw.strip_prefix('@').unwrap_or(raw);
    let s = s.strip_prefix('$').unwrap_or(s);
    s.strip_prefix('{')
        .and_then(|inner| inner.strip_suffix('}'))
        .unwrap_or(s)
}

/// The bare name of a variable token: the sigil (`$` or splat `@`) and any
/// scope qualifier (`script:`, `env:`, `global:`, ...) removed.
///
/// `"$script:Config"` becomes `"Config"`, `"@args"` becomes `"args"`, `"$_"`
/// becomes `"_"`, and `"${x}"` becomes `"x"`. A token that is only a sigil
/// yields `""`.
pub fn variable_name(raw: &str) -> &str {
    match strip_sigil(raw).rsplit_once(':') {
        Some((_, name)) => name,
        None => strip_sigil(raw),
    }
}

/// The scope qualifier of a variable token, if any.
///
/// `"$script:Config"` yields `Some("script")`; `"$env:PATH"` yields
/// `Some("env")`; `"$x"` yields `None`.
pub fn variable_scope(raw: &str) -> Option<&str> {
    strip_sigil(raw).rsplit_once(':').map(|(scope, _)| scope)
}

/// True when `name` is a PowerShell automatic variable.
///
/// Accepts the name with or without a leading sigil and is case-insensitive,
/// since PowerShell variable names are. A scope qualifier is stripped first, so
/// `"$script:_"` is judged on `"_"`.
pub fn is_automatic_variable(name: &str) -> bool {
    let bare = variable_name(name);
    AUTOMATIC_VARIABLES
        .iter()
        .any(|a| a.eq_ignore_ascii_case(bare))
}

/// True when `name` is a PowerShell preference variable. A rename usually skips
/// these alongside the automatics.
pub fn is_preference_variable(name: &str) -> bool {
    let bare = variable_name(name);
    PREFERENCE_VARIABLES
        .iter()
        .any(|p| p.eq_ignore_ascii_case(bare))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_word_variables() {
        for raw in [
            "$args",
            "$PSItem",
            "$PSScriptRoot",
            "$MyInvocation",
            "$IsWindows",
            "$null",
            "$true",
            "$false",
        ] {
            assert!(is_automatic_variable(raw), "{raw} should be automatic");
        }
    }

    #[test]
    fn automatic_punctuation_variables() {
        assert!(is_automatic_variable("$_"));
        assert!(is_automatic_variable("$?"));
        assert!(is_automatic_variable("$$"));
        assert!(is_automatic_variable("$^"));
    }

    #[test]
    fn accepts_names_without_a_sigil() {
        assert!(is_automatic_variable("PSItem"));
        assert!(is_automatic_variable("_"));
    }

    #[test]
    fn classification_is_case_insensitive() {
        assert!(is_automatic_variable("$True"));
        assert!(is_automatic_variable("$psitem"));
        assert!(is_automatic_variable("$PSSCRIPTROOT"));
    }

    #[test]
    fn user_variables_are_not_automatic() {
        for raw in ["$x", "$LogDir", "$myArgs", "$arguments", "$path"] {
            assert!(!is_automatic_variable(raw), "{raw} should not be automatic");
        }
    }

    #[test]
    fn scope_qualifier_is_stripped_before_classifying() {
        assert!(is_automatic_variable("$script:_"));
        assert!(!is_automatic_variable("$script:Config"));
    }

    #[test]
    fn preference_variables() {
        assert!(is_preference_variable("$ErrorActionPreference"));
        assert!(is_preference_variable("$VerbosePreference"));
        assert!(!is_preference_variable("$x"));
        // A preference variable is not an automatic one.
        assert!(!is_automatic_variable("$ErrorActionPreference"));
    }

    #[test]
    fn variable_name_strips_sigil_and_scope() {
        assert_eq!(variable_name("$x"), "x");
        assert_eq!(variable_name("$script:Config"), "Config");
        assert_eq!(variable_name("$env:PATH"), "PATH");
        assert_eq!(variable_name("@args"), "args");
        assert_eq!(variable_name("@$args"), "args");
        assert_eq!(variable_name("${x}"), "x");
        assert_eq!(variable_name("$_"), "_");
        assert_eq!(variable_name("$$"), "$");
    }

    #[test]
    fn variable_scope_reads_the_qualifier() {
        assert_eq!(variable_scope("$script:Config"), Some("script"));
        assert_eq!(variable_scope("$env:PATH"), Some("env"));
        assert_eq!(variable_scope("$x"), None);
        assert_eq!(variable_scope("$_"), None);
    }
}
