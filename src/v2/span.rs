//! Byte ranges into the original source, and an offset -> line/column map.
//!
//! Everything in `v2` is anchored to byte offsets of the *original* input.
//! `Span` is the unit of that anchoring; [`LineIndex`] converts an offset into
//! a line/column pair for diagnostics and editor protocols. (v1 tokens carry a
//! start offset and a precomputed line/col; v2 keeps full ranges instead and
//! derives line/col on demand.)

use std::fmt;

/// A half-open byte range `[start, end)` into the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// Byte offset of the first byte covered by the span.
    pub start: usize,
    /// Byte offset one past the last byte covered by the span.
    pub end: usize,
}

impl Span {
    /// Creates a span. `start` must not exceed `end`.
    pub fn new(start: usize, end: usize) -> Self {
        debug_assert!(start <= end, "span start {start} > end {end}");
        Self { start, end }
    }

    /// Length of the span in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// True when the span covers zero bytes (an insertion point).
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// True when `offset` falls inside the span.
    pub fn contains(&self, offset: usize) -> bool {
        self.start <= offset && offset < self.end
    }

    /// Smallest span covering both `self` and `other`.
    pub fn join(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }

    /// The slice of `src` this span covers.
    ///
    /// # Panics
    /// Panics if the span is out of bounds for `src` or cuts a UTF-8
    /// character in half; spans produced by the v2 lexer never do either.
    pub fn slice<'a>(&self, src: &'a str) -> &'a str {
        &src[self.start..self.end]
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A zero-based line/column position. The column is a byte offset from the
/// start of the line, not a character count; callers that need character or
/// grapheme columns can compute them from the source line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineCol {
    pub line: u32,
    /// Column as a byte offset within the line, not a character count. On a
    /// line containing multi-byte characters this differs from the column an
    /// editor displays (chars) and from the UTF-16 unit count an LSP client
    /// expects; convert via the line's text when either of those is needed.
    pub col: u32,
}

/// Precomputed table of line start offsets for one source text.
///
/// ```
/// use poshtree::v2::LineIndex;
///
/// let idx = LineIndex::new("ab\ncd");
/// assert_eq!(idx.line_col(4).line, 1);
/// assert_eq!(idx.line_col(4).col, 1);
/// ```
#[derive(Debug, Clone)]
pub struct LineIndex {
    /// Byte offset at which each line starts; `line_starts[0] == 0`.
    line_starts: Vec<usize>,
    len: usize,
}

impl LineIndex {
    /// Builds the index for `src`. Lines are split on `\n`; a `\r\n` pair
    /// therefore ends a line at the `\n` like everywhere else in `v2`.
    pub fn new(src: &str) -> Self {
        let mut line_starts = vec![0];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            line_starts,
            len: src.len(),
        }
    }

    /// Number of lines, counting a trailing line after a final newline.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Byte offset at which `line` (zero-based) starts.
    pub fn line_start(&self, line: u32) -> Option<usize> {
        self.line_starts.get(line as usize).copied()
    }

    /// Maps a byte offset to its line/column. Offsets past the end of the
    /// text clamp to the end.
    pub fn line_col(&self, offset: usize) -> LineCol {
        let offset = offset.min(self.len);
        let line = self.line_starts.partition_point(|&s| s <= offset) - 1;
        LineCol {
            line: line as u32,
            col: (offset - self.line_starts[line]) as u32,
        }
    }
}

/// A half-open range of token indices `[first, end)` into a token vector. An empty range (`first == end`) marks a node whose bytes are
/// owned by an enclosing token (an in-string interpolation node).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenRange {
    /// Index of the first token of the node.
    pub first: usize,
    /// One past the index of the last token of the node.
    pub end: usize,
}

impl TokenRange {
    /// Number of tokens in the range.
    pub fn len(&self) -> usize {
        self.end - self.first
    }

    /// True when the range covers no tokens of its own.
    pub fn is_empty(&self) -> bool {
        self.first == self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_basics() {
        let s = Span::new(2, 5);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert!(s.contains(2) && s.contains(4) && !s.contains(5));
        assert_eq!(s.join(Span::new(7, 9)), Span::new(2, 9));
        assert_eq!(s.slice("abcdefgh"), "cde");
        assert!(Span::new(3, 3).is_empty());
    }

    #[test]
    fn line_index_lf_and_crlf() {
        let idx = LineIndex::new("ab\r\ncd\ne");
        assert_eq!(idx.line_count(), 3);
        // 'a'
        assert_eq!(idx.line_col(0), LineCol { line: 0, col: 0 });
        // the '\r' still belongs to line 0
        assert_eq!(idx.line_col(2), LineCol { line: 0, col: 2 });
        // 'c'
        assert_eq!(idx.line_col(4), LineCol { line: 1, col: 0 });
        // 'e'
        assert_eq!(idx.line_col(7), LineCol { line: 2, col: 0 });
        // past the end clamps
        assert_eq!(idx.line_col(99), LineCol { line: 2, col: 1 });
        assert_eq!(idx.line_start(1), Some(4));
        assert_eq!(idx.line_start(9), None);
    }

    #[test]
    fn line_index_empty_and_trailing_newline() {
        let idx = LineIndex::new("");
        assert_eq!(idx.line_count(), 1);
        assert_eq!(idx.line_col(0), LineCol { line: 0, col: 0 });

        let idx = LineIndex::new("x\n");
        assert_eq!(idx.line_count(), 2);
        assert_eq!(idx.line_col(2), LineCol { line: 1, col: 0 });
    }
}
