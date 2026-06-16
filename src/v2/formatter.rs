//! A width-aware PowerShell formatter built on the lossless token layer.
//!
//! # Approach
//!
//! The formatter is token-driven and structure-checked. It does not build
//! output from the AST; it rewrites the whitespace *between* tokens, using
//! the bracket structure for indentation and a Wadler-style document IR for
//! width-aware line breaking. Token text is never altered, which the safety
//! harness then verifies.
//!
//! What it normalizes: indentation (from nesting depth), runs of blanks to a
//! single space, blank lines to at most one, line endings to `\n`, backtick
//! continuations to real soft wraps, and over-long lines, which break at the
//! safe continuation points (`|`, `&&`, `||`, after `,`, and around
//! brackets). What it preserves: every token byte-for-byte, the author's
//! statement-per-line structure, comments (leading comments keep their
//! lines, trailing comments stay trailing), here-strings and multi-line
//! strings verbatim, everything after `--%` verbatim, and source adjacency:
//! tokens the author glued together stay glued, because adjacency carries
//! meaning in PowerShell.
//!
//! # The safety harness
//!
//! [`format_source`] refuses input that does not lex and parse cleanly
//! ([`FormatError::Syntax`]), and before returning it re-lexes and re-parses
//! its own output and demands two equivalences: the significant token
//! sequence (kind and text) is unchanged, and the parse-tree fingerprint
//! (node labels and shape) is unchanged. The tree check earns its place
//! because reformatting moves newlines, and the parser reads newlines to
//! find statement boundaries. On any mismatch the caller gets
//! [`FormatError::Unsafe`] instead of damaged source. The test suite
//! also pins idempotence: formatting twice equals formatting once.

use super::ast::Node;
use super::tokens::{Token, TokenKind};

/// A token after which a statement continues onto the next line: the pipeline
/// and chain operators. Shared by `line()` (which keeps such a continuation in
/// one logical line) and `separator()` (which hangs it one indent level), so
/// the two halves of the wrap behaviour cannot drift and reopen the
/// continuation idempotency edge. Note `;` is deliberately not here: it
/// separates independent statements and is handled on its own in each site.
fn is_continuation_op(tok: &Token) -> bool {
    matches!(tok.kind, TokenKind::Pipe | TokenKind::Comma)
        || (tok.kind == TokenKind::Operator && matches!(tok.value.as_str(), "&&" | "||"))
}
use super::trivia::TriviaKind;
use std::fmt;

/// Formatting knobs. The defaults are 100 columns and 4-space indents.
#[derive(Debug, Clone)]
pub struct FormatOptions {
    /// Lines longer than this break at safe continuation points.
    pub max_width: usize,
    /// Spaces per indentation level.
    pub indent_width: usize,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            max_width: 100,
            indent_width: 4,
        }
    }
}

/// Why the formatter declined to produce output. It never returns
/// best-effort text: the result is either verified or absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// The input has lexer or parser errors; formatting malformed source
    /// risks changing what the author meant, so it is refused.
    Syntax { errors: Vec<String> },
    /// The formatter could not prove its output equivalent to the input,
    /// so the output was discarded. This is either a formatter bug or an
    /// input where v1 and v2 disagree about lexical structure (adversarial
    /// constructs such as quotes hidden inside what the other version
    /// considers a comment). Real-world scripts do not trip this; in either
    /// case no damaged output is ever returned.
    Unsafe { reason: String },
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::Syntax { errors } => {
                write!(f, "input has syntax errors: {}", errors.join("; "))
            }
            FormatError::Unsafe { reason } => {
                write!(f, "formatter safety check failed: {reason}")
            }
        }
    }
}

impl std::error::Error for FormatError {}

/// Formats PowerShell source with [`FormatOptions::default`].
pub fn format_source(src: &str) -> Result<String, FormatError> {
    format_source_with(src, &FormatOptions::default())
}

/// Formats PowerShell source with explicit options. See the module docs for
/// what is normalized, what is preserved, and the safety guarantees.
pub fn format_source_with(src: &str, options: &FormatOptions) -> Result<String, FormatError> {
    let lexed = super::lex(src);
    if !lexed.errors.is_empty() {
        let errors = lexed.errors.iter().map(|e| e.to_string()).collect();
        return Err(FormatError::Syntax { errors });
    }
    // Parse from the tokens just produced rather than lexing a second time;
    // `parse_tokens` hands them back on `ParseOutput::tokens` for the emitter.
    let parsed = super::parse_tokens(src, lexed.tokens);
    if !parsed.errors.is_empty() {
        let errors = parsed.errors.iter().map(|e| e.message.clone()).collect();
        return Err(FormatError::Syntax { errors });
    }

    let doc = Emitter::new(&parsed.tokens).file();
    let mut out = print(&doc, options);
    // Prefer ending with a newline, but only keep it when the result still
    // verifies: the convenience byte must never change the program.
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
        if verify_equivalent(&parsed.tokens, &parsed.script, &out).is_ok() {
            return Ok(out);
        }
        out.pop();
    }

    verify_equivalent(&parsed.tokens, &parsed.script, &out)?;
    Ok(out)
}

/// The harness: the output must lex and parse to the same program. The token
/// stream is compared exactly, and the parse-tree shape is compared because
/// reformatting moves newlines, which is what the native parser reads to find
/// statement boundaries.
fn verify_equivalent(
    input_tokens: &[Token],
    input_root: &Node,
    output: &str,
) -> Result<(), FormatError> {
    let out_lexed = super::lex(output);
    if !out_lexed.errors.is_empty() {
        return Err(FormatError::Unsafe {
            reason: "formatted output no longer parses cleanly".into(),
        });
    }
    let significant = |tokens: &[Token]| -> Vec<(TokenKind, String)> {
        tokens
            .iter()
            .filter(|t| t.kind != TokenKind::Eof)
            .map(|t| (t.kind, t.value.clone()))
            .collect()
    };
    if significant(input_tokens) != significant(&out_lexed.tokens) {
        return Err(FormatError::Unsafe {
            reason: "token sequence changed".into(),
        });
    }
    // Reuse the tokens just compared for the parse, again avoiding a re-lex.
    let out_parsed = super::parse_tokens(output, out_lexed.tokens);
    if !out_parsed.errors.is_empty() {
        return Err(FormatError::Unsafe {
            reason: "formatted output no longer parses cleanly".into(),
        });
    }
    if !same_shape(input_root, &out_parsed.script) {
        return Err(FormatError::Unsafe {
            reason: "tree structure changed".into(),
        });
    }
    Ok(())
}

/// Structural fingerprint comparison: node labels and shape, ignoring spans
/// and token ranges (those legitimately move). Token values are already
/// pinned by the exact token-sequence check in [`verify_equivalent`].
fn same_shape(a: &Node, b: &Node) -> bool {
    a.label() == b.label() && {
        let (ca, cb) = (a.children(), b.children());
        ca.len() == cb.len() && ca.iter().zip(&cb).all(|(x, y)| same_shape(x, y))
    }
}

// Document IR and printer

/// The document language. `Line`-family nodes render differently depending
/// on whether their enclosing [`Doc::Group`] fits the line flat.
#[derive(Debug, Clone)]
enum Doc {
    /// Single-line text, printed verbatim.
    Text(String),
    /// Multi-line text printed verbatim with no reindentation: here-strings,
    /// multi-line literals, block comments. Forces enclosing groups to break.
    Raw(String),
    /// A space when flat, a line break when broken.
    Line,
    /// Nothing when flat, a line break when broken.
    SoftLine,
    /// Always a line break.
    HardLine,
    /// Always a line break preceded by one empty line.
    BlankLine,
    /// Try to render the contents on one line; break the `Line`s inside if
    /// they do not fit the width.
    Group(Vec<Doc>),
    /// Contents print one indentation level deeper after each break.
    Indent(Vec<Doc>),
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Flat,
    Break,
}

fn print(doc: &Doc, options: &FormatOptions) -> String {
    let mut out = String::new();
    let mut col = 0usize;
    let mut stack: Vec<(usize, Mode, &Doc)> = vec![(0, Mode::Break, doc)];
    let mut fits_stack: Vec<&Doc> = Vec::new();

    let newline = |out: &mut String, col: &mut usize, indent: usize| {
        while out.ends_with(' ') {
            out.pop();
        }
        out.push('\n');
        let pad = indent * options.indent_width;
        out.extend(std::iter::repeat_n(' ', pad));
        *col = pad;
    };

    while let Some((indent, mode, d)) = stack.pop() {
        match d {
            Doc::Text(s) => {
                out.push_str(s);
                col += s.chars().count();
            }
            Doc::Raw(s) => {
                out.push_str(s);
                col = match s.rfind('\n') {
                    Some(i) => s[i + 1..].chars().count(),
                    None => col + s.chars().count(),
                };
            }
            Doc::Line => match mode {
                Mode::Flat => {
                    out.push(' ');
                    col += 1;
                }
                Mode::Break => newline(&mut out, &mut col, indent),
            },
            Doc::SoftLine => {
                if mode == Mode::Break {
                    newline(&mut out, &mut col, indent);
                }
            }
            Doc::HardLine => newline(&mut out, &mut col, indent),
            Doc::BlankLine => {
                while out.ends_with(' ') {
                    out.pop();
                }
                out.push('\n');
                newline(&mut out, &mut col, indent);
            }
            Doc::Group(items) => {
                let chosen = if mode == Mode::Flat
                    || fits(
                        options.max_width.saturating_sub(col),
                        items,
                        &mut fits_stack,
                    ) {
                    Mode::Flat
                } else {
                    Mode::Break
                };
                for item in items.iter().rev() {
                    stack.push((indent, chosen, item));
                }
            }
            Doc::Indent(items) => {
                for item in items.iter().rev() {
                    stack.push((indent + 1, mode, item));
                }
            }
        }
    }
    out
}

/// Would `docs` rendered flat fit in `remaining` columns? Hard breaks and
/// multi-line raws never fit. `stack` is a scratch buffer reused across calls
/// to avoid an allocation per group; its prior contents are discarded.
fn fits<'a>(remaining: usize, docs: &'a [Doc], stack: &mut Vec<&'a Doc>) -> bool {
    let mut budget = remaining as isize;
    stack.clear();
    stack.extend(docs.iter().rev());
    while let Some(d) = stack.pop() {
        // A string never fits in fewer columns than its character count, but
        // counting characters walks the whole string. Byte length is an upper
        // bound on that count, so once the bytes alone exceed the budget the
        // exact count cannot fit either and `fits` stops without the walk.
        match d {
            Doc::Text(s) => {
                if (s.len() as isize) > budget {
                    return false;
                }
                budget -= s.chars().count() as isize;
            }
            Doc::Raw(s) => {
                if s.contains('\n') || (s.len() as isize) > budget {
                    return false;
                }
                budget -= s.chars().count() as isize;
            }
            Doc::Line => budget -= 1,
            Doc::SoftLine => {}
            Doc::HardLine | Doc::BlankLine => return false,
            Doc::Group(items) | Doc::Indent(items) => {
                for item in items.iter().rev() {
                    stack.push(item);
                }
            }
        }
        if budget < 0 {
            return false;
        }
    }
    true
}

// Emitter: tokens -> Doc

/// One piece of the gap between two significant tokens, in source order.
/// `glued_left`/`glued_right` mean byte-adjacency to the neighboring
/// non-whitespace content: v1's raw argument scanner runs barewords through
/// glued `#` and glued backtick continuations, so those must be re-emitted
/// exactly as written or the v1 parse changes.
#[derive(Debug, Clone)]
enum GapItem {
    /// One line break.
    Newline,
    /// A comment: text, is-block, glued_left, glued_right.
    Comment(String, bool, bool, bool),
    /// A backtick line continuation glued to non-space on both sides (the
    /// inside of a v1 bareword). A spaced continuation is dropped, joining
    /// the lines, which is safe; a glued one carries its exact text.
    GluedContinuation(String),
}

/// The fully interpreted gap before a token.
struct Gap {
    items: Vec<GapItem>,
    /// The previous and current token touch byte-to-byte (no trivia at all).
    adjacent: bool,
}

/// A gap, digested: each comment with the number of line breaks before it
/// (0 = inline on the previous line, 1 = own line, 2+ = preceded by a blank
/// line) and its left/right glue flags, plus the line breaks remaining
/// before the token. Glued continuations ride in `comments` as raw entries
/// glued on both sides, so every byte-sensitive gap item flows through one
/// emission path.
struct GapInfo {
    comments: Vec<(usize, String, bool, bool, bool)>,
    newlines_before_token: usize,
}

impl Gap {
    fn newline_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| matches!(i, GapItem::Newline))
            .count()
    }

    /// Removes comments and continuations, keeping line breaks. A gap at a
    /// level's stop position (Eof or an unmatched closer) is observed by
    /// every enclosing level; clearing after the first emission keeps its
    /// comments from being printed once per nesting depth.
    fn clear_comments(&mut self) {
        self.items.retain(|i| matches!(i, GapItem::Newline));
    }

    fn info(&self) -> GapInfo {
        let mut comments = Vec::new();
        let mut newlines = 0usize;
        for item in &self.items {
            match item {
                GapItem::Newline => newlines += 1,
                GapItem::Comment(text, block, gl, gr) => {
                    comments.push((newlines, text.clone(), *block, *gl, *gr));
                    newlines = 0;
                }
                // A glued continuation is "inline raw text glued on both
                // sides": comment_doc emits it verbatim (it contains the
                // newline), and the glue flags suppress added spacing.
                GapItem::GluedContinuation(text) => {
                    comments.push((newlines, text.clone(), true, true, true));
                    newlines = 0;
                }
            }
        }
        GapInfo {
            comments,
            newlines_before_token: newlines,
        }
    }
}

struct Emitter<'a> {
    tokens: &'a [Token],
    /// Gap before each token (same indexing as `tokens`).
    gaps: Vec<Gap>,
    pos: usize,
    /// Whether bracket tokens balance. When they do not (PowerShell lets
    /// brackets live inside bareword arguments and comments, so a v1-clean
    /// file can have unbalanced v2 bracket tokens), the emitter falls back
    /// to flat output: no bracket recursion, no reindentation, spacing and
    /// line handling only.
    structured: bool,
}

impl<'a> Emitter<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        let mut gaps = Vec::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            // Collect the gap's trivia in order, with spans, so gluing can
            // be computed from byte adjacency.
            let mut raw: Vec<(&TriviaKind, &str, super::span::Span)> = Vec::new();
            if i > 0 {
                for t in &tokens[i - 1].trailing {
                    raw.push((&t.kind, &t.text, t.span));
                }
            }
            for t in &tok.leading {
                raw.push((&t.kind, &t.text, t.span));
            }

            // End offset of the nearest non-whitespace content to the left
            // of each item; None at file start or across whitespace.
            let mut items = Vec::new();
            let mut left_end: Option<usize> = (i > 0).then(|| tokens[i - 1].span.end);
            for (idx, (kind, text, span)) in raw.iter().enumerate() {
                let right_start = raw
                    .get(idx + 1)
                    .map(|(_, _, s)| s.start)
                    .unwrap_or(tok.span.start);
                let glued_left = left_end == Some(span.start);
                let glued_right = span.end == right_start;
                match kind {
                    TriviaKind::Newline => items.push(GapItem::Newline),
                    TriviaKind::LineComment => {
                        items.push(GapItem::Comment(
                            (*text).to_string(),
                            false,
                            glued_left,
                            glued_right,
                        ));
                    }
                    TriviaKind::BlockComment => {
                        items.push(GapItem::Comment(
                            (*text).to_string(),
                            true,
                            glued_left,
                            glued_right,
                        ));
                    }
                    TriviaKind::LineContinuation => {
                        if glued_left && glued_right {
                            items.push(GapItem::GluedContinuation((*text).to_string()));
                        }
                        // spaced continuations are joined: emit nothing
                    }
                    TriviaKind::Whitespace => {}
                }
                left_end = match kind {
                    TriviaKind::Whitespace | TriviaKind::Newline => None,
                    _ => Some(span.end),
                };
            }
            let adjacent = i > 0
                && tokens[i - 1].trailing.is_empty()
                && tok.leading.is_empty()
                && tokens[i - 1].span.end == tok.span.start;
            gaps.push(Gap { items, adjacent });
        }
        Self {
            structured: brackets_balanced(tokens),
            tokens,
            gaps,
            pos: 0,
        }
    }

    fn file(mut self) -> Doc {
        let body = self.level(None);
        Doc::Group(body)
    }

    fn cur(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn at_closer_of(&self, closer: Option<TokenKind>) -> bool {
        match closer {
            Some(kind) => self.cur().kind == kind,
            None => false,
        }
    }

    /// Separation before a comment or a line, by how many line breaks the
    /// author used: none keeps it inline (or glued, when the author wrote it
    /// with no whitespace), one is a line break, more collapse to a single
    /// blank line. Nothing is emitted before a level's first content, which
    /// also trims leading blank lines inside blocks.
    fn push_break(out: &mut Vec<Doc>, newlines: usize, glued: bool, emitted: bool) {
        if !emitted {
            return;
        }
        out.push(match newlines {
            0 if glued => Doc::Text(String::new()),
            0 => Doc::Text(" ".into()),
            1 => Doc::HardLine,
            _ => Doc::BlankLine,
        });
    }

    /// Emits one nesting level: lines separated by hard breaks, with
    /// comments interleaved where the author put them, stopping before
    /// `closer` (or Eof). Comments sitting in the closer's own gap (the
    /// final comments of a block or of the file) are emitted too.
    fn level(&mut self, closer: Option<TokenKind>) -> Vec<Doc> {
        let mut out: Vec<Doc> = Vec::new();
        let mut emitted = false;
        loop {
            let info = self.gaps[self.pos].info();
            if !info.comments.is_empty() {
                self.gaps[self.pos].clear_comments();
            }
            let mut token_glued = false;
            for (newlines, text, block, glued_left, glued_right) in &info.comments {
                Self::push_break(&mut out, *newlines, *glued_left, emitted);
                out.push(comment_doc(text, *block));
                emitted = true;
                token_glued = *glued_right;
            }
            if self.cur().kind == TokenKind::Eof || self.at_closer_of(closer) {
                break;
            }
            Self::push_break(&mut out, info.newlines_before_token, token_glued, emitted);
            let line = self.line(closer);
            out.push(Doc::Group(line));
            emitted = true;
        }
        out
    }

    /// Emits one logical line: tokens until the next newline gap, the
    /// level's closer, or Eof. Bracketed sub-structures recurse.
    fn line(&mut self, closer: Option<TokenKind>) -> Vec<Doc> {
        let mut out: Vec<Doc> = Vec::new();
        let mut first = true;
        loop {
            let tok_kind = self.cur().kind;
            if tok_kind == TokenKind::Eof || self.at_closer_of(closer) {
                break;
            }
            if !first && self.gaps[self.pos].newline_count() > 0 {
                // A newline normally ends the logical line. But when the
                // previous token is a line-continuation operator (`|`, `,`,
                // `&&`, `||`), the statement continues onto the next physical
                // line, so keep it in this logical line and let `separator()`
                // decide the wrap. Otherwise the indentation would differ
                // between a freshly wrapped pipeline and one already wrapped in
                // the source (the soft wrap hangs a level; an existing newline
                // would break to the base indent), which is not idempotent.
                if !is_continuation_op(&self.tokens[self.pos - 1]) {
                    break; // a new line of this level starts here
                }
            }
            if !first {
                out.push(self.separator());
            }
            first = false;

            match tok_kind {
                TokenKind::LParen
                | TokenKind::LBracket
                | TokenKind::LBrace
                | TokenKind::DollarParen
                | TokenKind::AtParen
                | TokenKind::AtBrace
                    if self.structured =>
                {
                    let doc = self.bracketed();
                    out.push(doc);
                }
                _ => {
                    out.push(token_doc(self.cur()));
                    self.pos += 1;
                }
            }
        }
        out
    }

    /// Emits an open token, its inner level, and the matching close token as
    /// one group: flat with the brace style (`{ x }`, `($x)`) when it fits,
    /// a fully indented block otherwise or when the author used newlines.
    fn bracketed(&mut self) -> Doc {
        let open_value = self.cur().value.clone();
        let close_kind = match self.cur().kind {
            TokenKind::LParen | TokenKind::DollarParen | TokenKind::AtParen => TokenKind::RParen,
            TokenKind::LBracket => TokenKind::RBracket,
            _ => TokenKind::RBrace,
        };
        let spaced = matches!(self.cur().kind, TokenKind::LBrace | TokenKind::AtBrace);
        self.pos += 1; // open

        // The author's newline right after the opener becomes the group's
        // own break; `level` handles the structure inside.
        let opener_break = self.gaps[self.pos].newline_count() > 0;
        let inner = self.level(Some(close_kind));
        let has_close = self.cur().kind == close_kind;
        if has_close {
            self.pos += 1; // close
        }

        let edge = || -> Doc {
            if opener_break {
                Doc::HardLine
            } else if spaced {
                Doc::Line
            } else {
                Doc::SoftLine
            }
        };

        let mut docs = vec![Doc::Text(open_value)];
        if !inner.is_empty() {
            let mut indented = vec![edge()];
            indented.extend(inner);
            docs.push(Doc::Indent(indented));
            docs.push(edge());
        }
        if has_close {
            docs.push(Doc::Text(close_token_value(close_kind)));
        }
        Doc::Group(docs)
    }

    /// The spacing between the previous significant token and the current
    /// one, both on the same source line.
    fn separator(&mut self) -> Doc {
        let gap = &self.gaps[self.pos];
        let info = gap.info();
        if !info.comments.is_empty() {
            // Inline block comments (`ls <# why #> -la`) and glued
            // continuations, spaced only where the author had whitespace.
            let mut pieces: Vec<Doc> = Vec::new();
            let mut last_glued_right = false;
            for (_, text, block, glued_left, glued_right) in &info.comments {
                if !*glued_left {
                    pieces.push(Doc::Text(" ".into()));
                }
                pieces.push(comment_doc(text, *block));
                last_glued_right = *glued_right;
            }
            if !last_glued_right {
                pieces.push(Doc::Text(" ".into()));
            }
            return Doc::Group(pieces);
        }
        if gap.adjacent {
            return Doc::Text(String::new());
        }
        let prev = &self.tokens[self.pos - 1];
        let cur = self.cur();
        // Tight to the left of list punctuation.
        if matches!(cur.kind, TokenKind::Comma | TokenKind::Semicolon) {
            return Doc::Text(String::new());
        }
        // Breakable after pipeline and chain operators and commas: these are
        // continuations, so the Indent makes them hang one level. A semicolon
        // separates independent statements at the same level, so it breaks to
        // the current indent without an extra level; indenting it made the
        // result depend on whether the break happened, which broke idempotency.
        if is_continuation_op(prev) {
            return Doc::Indent(vec![Doc::Line]);
        }
        if prev.kind == TokenKind::Semicolon {
            return Doc::Line;
        }
        Doc::Text(" ".into())
    }
}

/// Do the bracket tokens pair up? Mismatches mean the brackets are bareword
/// or comment text from v1's point of view, and structural emission would
/// misread the file.
fn brackets_balanced(tokens: &[Token]) -> bool {
    let mut stack: Vec<TokenKind> = Vec::new();
    for tok in tokens {
        match tok.kind {
            TokenKind::LParen | TokenKind::DollarParen | TokenKind::AtParen => {
                stack.push(TokenKind::RParen)
            }
            TokenKind::LBracket => stack.push(TokenKind::RBracket),
            TokenKind::LBrace | TokenKind::AtBrace => stack.push(TokenKind::RBrace),
            TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace
                if stack.pop() != Some(tok.kind) =>
            {
                return false;
            }
            _ => {}
        }
    }
    stack.is_empty()
}

fn token_doc(tok: &Token) -> Doc {
    if tok.value.contains('\n') {
        Doc::Raw(tok.value.clone())
    } else {
        Doc::Text(tok.value.clone())
    }
}

fn comment_doc(text: &str, block: bool) -> Doc {
    if block && text.contains('\n') {
        Doc::Raw(text.to_string())
    } else {
        Doc::Text(text.to_string())
    }
}

fn close_token_value(kind: TokenKind) -> String {
    match kind {
        TokenKind::RParen => ")",
        TokenKind::RBracket => "]",
        _ => "}",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(src: &str) -> String {
        format_source(src).unwrap_or_else(|e| panic!("format failed for {src:?}: {e}"))
    }

    #[test]
    fn normalizes_spacing_and_indentation() {
        assert_eq!(fmt("ls   -la\n"), "ls -la\n");
        assert_eq!(
            fmt("if ($x) {\nWrite-Output $x\n}\n"),
            "if ($x) {\n    Write-Output $x\n}\n"
        );
        // tabs and deep over-indentation are rebuilt from structure
        assert_eq!(
            fmt("if ($x) {\n\t\t\tls\n      pwd\n}\n"),
            "if ($x) {\n    ls\n    pwd\n}\n"
        );
    }

    #[test]
    fn collapses_blank_lines_to_one() {
        assert_eq!(fmt("ls\n\n\n\npwd\n"), "ls\n\npwd\n");
    }

    #[test]
    fn keeps_adjacent_tokens_glued() {
        // Member access, indexing, and static access lex as adjacent tokens;
        // the parser glues by adjacency, so the formatter must never insert
        // a space into them.
        assert_eq!(fmt("$x.Length\n"), "$x.Length\n");
        assert_eq!(fmt("$a[0]\n"), "$a[0]\n");
        assert_eq!(fmt("[int]::MaxValue\n"), "[int]::MaxValue\n");
    }

    #[test]
    fn comments_survive_in_place() {
        let src = "# header\nls   # trailing\n\n# standalone\npwd\n";
        assert_eq!(fmt(src), "# header\nls # trailing\n\n# standalone\npwd\n");
        let block = "<#\n multi\n line\n#>\nls\n";
        assert_eq!(fmt(block), block);
    }

    #[test]
    fn here_strings_and_verbatim_args_untouched() {
        let here = "$x = @'\n  raw   spacing\n'@\n";
        assert_eq!(fmt(here), here);
        let verbatim = "icacls --% C:\\Program Files\\*   /grant x # not a comment\n";
        assert_eq!(fmt(verbatim), verbatim);
    }

    #[test]
    fn long_pipelines_break_at_pipes() {
        let src = "Get-ChildItem -Path C:\\some\\deep\\path -Recurse | Where-Object { $_.Length -gt 1000000 } | Sort-Object Length | Select-Object -First 10\n";
        let out = format_source_with(
            src,
            &FormatOptions {
                max_width: 60,
                indent_width: 4,
            },
        )
        .unwrap();
        assert!(
            out.lines().count() > 1,
            "expected a wrapped pipeline:\n{out}"
        );
        for line in out.lines() {
            assert!(line.chars().count() <= 60 || line.contains('\''), "{line}");
        }
        // broken at pipes: continuation lines start with the next command
        assert!(out.contains("|\n"));
    }

    #[test]
    fn short_blocks_stay_flat_long_ones_break() {
        assert_eq!(
            fmt("gci | Where-Object { $_.x }\n"),
            "gci | Where-Object { $_.x }\n"
        );
        let out = format_source_with(
            "gci | Where-Object { $_.VeryLongPropertyName -gt $someThreshold }\n",
            &FormatOptions {
                max_width: 30,
                indent_width: 4,
            },
        )
        .unwrap();
        assert!(out.contains("{\n"), "expected block to break:\n{out}");
    }

    #[test]
    fn continuations_are_joined() {
        assert_eq!(fmt("ls `\n  -la `\n  /tmp\n"), "ls -la /tmp\n");
    }

    #[test]
    fn refuses_syntax_errors() {
        match format_source("'unterminated\n") {
            Err(FormatError::Syntax { .. }) => {}
            other => panic!("expected Syntax error, got {other:?}"),
        }
    }

    #[test]
    fn wrapped_continuations_with_here_strings_are_idempotent() {
        // A `|`/`&&`/`||` continuation whose line contains a multi-line
        // here-string forces a wrap. The wrap used to hang one level on the
        // first pass but break to the base indent on the second (an existing
        // newline after the operator was treated differently from a fresh
        // wrap). The continuation now stays one logical line either way.
        for src in [
            "cmd @\"\nx\n\"@ | bar\n",
            "$x | Get-Item @\"\ny\n\"@\n",
            "foo @\"\nh\n\"@ | bar | baz\n",
            "$a && cmd @\"\nx\n\"@\n",
        ] {
            let once = format_source(src).unwrap();
            let twice = format_source(&once).unwrap();
            assert_eq!(once, twice, "not idempotent: {src:?}");
        }
        // A comma list with a here-string element reshapes the tree, so the
        // safety guard declines it. Declining is stable: the caller keeps the
        // original both times.
        let comma = "a, @\"\nx\n\"@, b\n";
        assert!(
            format_source(comma).is_err(),
            "comma+here-string should decline"
        );
    }

    #[test]
    fn semicolon_separated_statements_are_idempotent() {
        // A `;` separates independent statements at the same indent. It used to
        // hang one level when it broke, so a here-string argument (which forces
        // a break) produced 4 vs 0 leading spaces across passes.
        for src in [
            "0; -d @\"\nx\n\"@\n",
            "$x = 1; Get-Item @\"\ndata\n\"@\n",
            "$a = 1; $b = 2\n",
            "ls; cd; pwd\n",
            "foo | bar; baz\n",
        ] {
            let once = format_source(src).unwrap();
            let twice = format_source(&once).unwrap();
            assert_eq!(once, twice, "not idempotent: {src:?}");
        }
    }

    #[test]
    fn idempotent_over_corpus() {
        let corpus = [
            "function Get-Thing {\n  param([string]$Name)\n  process { $Name }\n}\n",
            "$h = @{ a = 1; b = @(2, 3) }\n",
            "try { risky } catch [System.Exception] { recover } finally { done }\n",
            "foreach ($f in gci) {\n  Write-Host $f.Name  # each\n}\n",
            "switch ($x) {\n 1 { 'one' }\n default { 'other' }\n}\n",
            "\"interpolated $x and $($y.Prop) end\"\n",
            "Test-Path $p && ls || Write-Error 'no'\n",
            "$x = $a ?? $b; $i++\n",
        ];
        for src in corpus {
            let once = fmt(src);
            let twice = fmt(&once);
            assert_eq!(once, twice, "not idempotent for {src:?}");
        }
    }

    #[test]
    fn empty_and_comment_only_inputs() {
        assert_eq!(fmt(""), "");
        assert_eq!(fmt("# only a comment"), "# only a comment\n");
        assert_eq!(fmt("   \n\n  \n"), "");
    }

    /// Inputs distilled from fuzzing: glued comments, glued backtick
    /// continuations, and stray brackets. The formatter leaves all of that
    /// glue byte-for-byte alone; the only edit is the trailing newline it
    /// prefers a file to end with (the native parser treats that newline as
    /// trivia, so it verifies cleanly).
    #[test]
    fn pathological_glue_is_preserved() {
        let cases = [
            "Y?c<_#2]_-//_[`\na0ab/`+c;X<$b`(/&,2#*#'&",
            "b_2]#\"'}`,b&/{|/a|,0+? ;.0]&c)--?[`*(=/-c",
            "&Z``?>_+-]`'a|1*c1 .?}*:#'1]@'+Y",
        ];
        for src in cases {
            assert_eq!(
                fmt(src),
                format!("{src}\n"),
                "pathological glue was altered"
            );
        }
    }

    /// Deterministic fuzz: the formatter never panics, and every input it
    /// accepts reformats to the identical text. Refusals (syntax errors or
    /// verified declines) are fine; damage is not.
    #[test]
    fn fuzz_no_panics_accepted_inputs_idempotent() {
        fn next(s: &mut u64) -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        }
        let charset: Vec<char> = "$@{}()[]| ;,.\"'`#-=+*/<>?:&\nabcXYZ012_".chars().collect();
        let mut state: u64 = 0x5EED;
        let mut accepted = 0;
        for _ in 0..600 {
            let len = (next(&mut state) % 40) as usize;
            let src: String = (0..len)
                .map(|_| charset[(next(&mut state) as usize) % charset.len()])
                .collect();
            if let Ok(once) = format_source(&src) {
                accepted += 1;
                let twice = format_source(&once).unwrap_or_else(|e| {
                    panic!("accepted output failed to reformat ({e}): {src:?}")
                });
                assert_eq!(once, twice, "not idempotent for {src:?}");
            }
        }
        assert!(accepted > 10, "fuzz accepted too few inputs: {accepted}");
    }
}
