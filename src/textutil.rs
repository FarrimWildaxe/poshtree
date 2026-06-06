//! Small string helpers used when rendering tree nodes for display.

/// Return the longest prefix of `s` no longer than `max_bytes` that ends on a
/// UTF-8 character boundary.
///
/// Slicing a `str` at an arbitrary byte index panics if it lands inside a
/// multi-byte character; this helper makes fixed-length truncation safe for
/// non-ASCII input. Cheap (no allocation) and a no-op when `s` already fits.
pub fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_short_strings_intact() {
        assert_eq!(truncate_on_char_boundary("abc", 10), "abc");
    }

    #[test]
    fn truncate_never_splits_a_multibyte_char() {
        // "é" is two bytes; cutting at byte 1 must back off to byte 0.
        let s = "é".repeat(10); // 20 bytes
        let out = truncate_on_char_boundary(&s, 5);
        assert!(s.is_char_boundary(out.len()));
        assert_eq!(out, "éé"); // 4 bytes, the largest boundary <= 5
    }
}
