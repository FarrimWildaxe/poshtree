//! Span-based text edits: the primitive a codemod tool builds on.
//!
//! A refactoring pass walks the tree, decides "replace bytes 14..27 with
//! `Get-CimInstance`", and collects [`TextEdit`]s. [`apply_edits`] then
//! rewrites the original source in one pass. Because edits address the
//! original bytes and everything else is copied through untouched, the
//! output is a minimal diff: comments, blank lines, and the author's
//! formatting all survive, with no unparser involved.

use super::span::Span;
use std::fmt;

/// Replace the bytes covered by `span` with `replacement`. An empty span is
/// an insertion point; an empty replacement is a deletion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEdit {
    pub span: Span,
    pub replacement: String,
}

impl TextEdit {
    pub fn replace(span: Span, replacement: impl Into<String>) -> Self {
        Self {
            span,
            replacement: replacement.into(),
        }
    }

    pub fn insert(offset: usize, text: impl Into<String>) -> Self {
        Self::replace(Span::new(offset, offset), text)
    }

    pub fn delete(span: Span) -> Self {
        Self::replace(span, "")
    }
}

/// Why a batch of edits was rejected. No partial output is ever produced:
/// either every edit applies or none does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    /// Two edits cover overlapping byte ranges.
    Overlap { first: Span, second: Span },
    /// An edit reaches past the end of the source.
    OutOfBounds(Span),
    /// An edit boundary would split a multi-byte UTF-8 character.
    NotCharBoundary(usize),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::Overlap { first, second } => {
                write!(f, "overlapping edits at {first} and {second}")
            }
            EditError::OutOfBounds(span) => {
                write!(f, "edit at {span} is out of bounds")
            }
            EditError::NotCharBoundary(offset) => {
                write!(f, "edit boundary at byte {offset} splits a character")
            }
        }
    }
}

impl std::error::Error for EditError {}

/// Applies `edits` to `src` and returns the rewritten text.
///
/// Edits may be given in any order; they are sorted by start offset before
/// application (the sort is stable, so two insertions at the same offset
/// keep their given order). The batch is validated up front: out-of-bounds
/// spans, boundaries inside a UTF-8 character, and overlapping ranges are
/// all rejected. Touching spans (`a.end == b.start`) are fine.
pub fn apply_edits(src: &str, edits: &[TextEdit]) -> Result<String, EditError> {
    let mut order: Vec<usize> = (0..edits.len()).collect();
    // Sort by start, then by end. The secondary key matters when a zero-width
    // insertion shares a start offset with a replacement: the insertion (end ==
    // start) must sort first so the adjacency check below sees it as touching
    // the replacement's left edge rather than as overlapping it.
    order.sort_by_key(|&i| (edits[i].span.start, edits[i].span.end));

    for &i in &order {
        let span = edits[i].span;
        if span.end > src.len() {
            return Err(EditError::OutOfBounds(span));
        }
        for offset in [span.start, span.end] {
            if !src.is_char_boundary(offset) {
                return Err(EditError::NotCharBoundary(offset));
            }
        }
    }
    for pair in order.windows(2) {
        let (a, b) = (edits[pair[0]].span, edits[pair[1]].span);
        if a.end > b.start {
            return Err(EditError::Overlap {
                first: a,
                second: b,
            });
        }
    }

    let extra: usize = edits.iter().map(|e| e.replacement.len()).sum();
    let mut out = String::with_capacity(src.len() + extra);
    let mut cursor = 0;
    for &i in &order {
        let edit = &edits[i];
        out.push_str(&src[cursor..edit.span.start]);
        out.push_str(&edit.replacement);
        cursor = edit.span.end;
    }
    out.push_str(&src[cursor..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_insert_delete() {
        let src = "Get-WmiObject Win32_BIOS # legacy";
        let edits = [
            TextEdit::replace(Span::new(0, 13), "Get-CimInstance"),
            TextEdit::insert(src.len(), "\n"),
        ];
        assert_eq!(
            apply_edits(src, &edits).unwrap(),
            "Get-CimInstance Win32_BIOS # legacy\n"
        );

        let edits = [TextEdit::delete(Span::new(3, 7))];
        assert_eq!(apply_edits("abcdefgh", &edits).unwrap(), "abch");
    }

    #[test]
    fn edits_apply_regardless_of_given_order() {
        let src = "a b c";
        let edits = [
            TextEdit::replace(Span::new(4, 5), "C"),
            TextEdit::replace(Span::new(0, 1), "A"),
        ];
        assert_eq!(apply_edits(src, &edits).unwrap(), "A b C");
    }

    #[test]
    fn touching_spans_are_fine_overlap_is_not() {
        let src = "abcdef";
        let touching = [
            TextEdit::replace(Span::new(0, 3), "X"),
            TextEdit::replace(Span::new(3, 6), "Y"),
        ];
        assert_eq!(apply_edits(src, &touching).unwrap(), "XY");

        let overlapping = [
            TextEdit::replace(Span::new(0, 4), "X"),
            TextEdit::replace(Span::new(3, 6), "Y"),
        ];
        assert!(matches!(
            apply_edits(src, &overlapping),
            Err(EditError::Overlap { .. })
        ));
    }

    #[test]
    fn bounds_and_char_boundaries_are_checked() {
        assert!(matches!(
            apply_edits("ab", &[TextEdit::delete(Span::new(1, 5))]),
            Err(EditError::OutOfBounds(_))
        ));
        // 'ż' is two bytes; offset 1 lands inside it
        assert!(matches!(
            apply_edits("ż", &[TextEdit::insert(1, "x")]),
            Err(EditError::NotCharBoundary(1))
        ));
    }

    #[test]
    fn insertion_at_a_replacement_boundary_is_order_independent() {
        // A zero-width insertion at the start (or end) of a replacement is a
        // boundary insertion, not an overlap, and the result must not depend on
        // the order the edits were supplied in.
        let src = "0123456789";
        let want = "PREX";
        let insert_first = [
            TextEdit::insert(0, "PRE"),
            TextEdit::replace(Span::new(0, 10), "X"),
        ];
        let replace_first = [
            TextEdit::replace(Span::new(0, 10), "X"),
            TextEdit::insert(0, "PRE"),
        ];
        assert_eq!(apply_edits(src, &insert_first).unwrap(), want);
        assert_eq!(apply_edits(src, &replace_first).unwrap(), want);

        // An insertion strictly inside a replacement is still an overlap.
        let inside = [
            TextEdit::replace(Span::new(0, 10), "X"),
            TextEdit::insert(5, "Y"),
        ];
        assert!(matches!(
            apply_edits(src, &inside),
            Err(EditError::Overlap { .. })
        ));
    }

    #[test]
    fn empty_edit_list_is_identity() {
        assert_eq!(apply_edits("unchanged", &[]).unwrap(), "unchanged");
    }
}
