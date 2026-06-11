//! Operator-name classification shared by the v1 and v2 lexers (and consulted
//! by the v2 parser), so the dash-word spelling rules live in one place
//! instead of drifting across hand-kept tables.

/// The comparison operators that accept the `c`/`i` case-sensitivity prefix
/// (`-ceq`, `-iLike`, ...), per about_Comparison_Operators and about_Split.
/// `-and`, `-f`, `-join`, `-is`, and `-isnot` take no prefix.
pub(crate) const CASE_PREFIXABLE: &[&str] = &[
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "like",
    "notlike",
    "match",
    "notmatch",
    "contains",
    "notcontains",
    "in",
    "notin",
    "replace",
    "split",
];

/// Whether a dash-word core (the text after the leading `-`) names an
/// operator, given a lexer's base table. Case-insensitive, no allocation.
///
/// The full name is checked first, so operators that begin with `c` or `i`
/// (`contains`, `in`) are never mis-stripped; only when the full name does not
/// match is a single `c`/`i` prefix stripped and the remainder checked against
/// the prefixable set.
pub(crate) fn is_named_operator_word(core: &str, base: &[&str]) -> bool {
    if base.iter().any(|n| n.eq_ignore_ascii_case(core)) {
        return true;
    }
    match core.as_bytes().first() {
        Some(b'c' | b'C' | b'i' | b'I') => CASE_PREFIXABLE
            .iter()
            .any(|n| n.eq_ignore_ascii_case(&core[1..])),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &[&str] = &["eq", "in", "contains", "and", "f", "is"];

    #[test]
    fn full_name_wins_before_prefix_stripping() {
        // `contains` and `in` begin with the prefix letters; the full-name
        // check must catch them before any stripping happens.
        assert!(is_named_operator_word("contains", BASE));
        assert!(is_named_operator_word("in", BASE));
        assert!(is_named_operator_word("Is", BASE));
    }

    #[test]
    fn case_prefix_applies_only_to_the_prefixable_set() {
        assert!(is_named_operator_word("cin", BASE));
        assert!(is_named_operator_word("ieq", BASE));
        assert!(is_named_operator_word("CnotIn", BASE));
        // `-cand`, `-cf`, `-cis` are not operators: the prefix does not apply.
        assert!(!is_named_operator_word("cand", BASE));
        assert!(!is_named_operator_word("cf", BASE));
        assert!(!is_named_operator_word("cis", BASE));
    }
}
