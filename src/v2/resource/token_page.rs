//! Exact-layout storage trial for supplied records or records offered by the lexer.
//!
//! Live bytes cover the requested allocation layouts, including unused slots
//! and each directory entry's padding. Source and input records are borrowed;
//! storage construction uses constant stack space and no heap scratch. The
//! charge is not a bound on allocator overhead or RSS. Only a returned null is
//! recoverable here, not allocator abort, overcommit, or OS termination.
//! Verbatim tails, line comments, block-comment candidates, and here-string
//! suffixes (including nested ones) charge work to the same invocation ledger.
//! A here-string suffix charges one unit on entry and one per examined byte.
//! Other scanner work, scratch, and diagnostics are not charged here.

use std::alloc::{alloc, dealloc, Layout};
use std::num::NonZeroUsize;
use std::ptr::{self, NonNull};

use super::{Ledger, Phase, Refusal, Reservation};
use crate::v2::{
    lexer::lex_records_into,
    span::Span,
    tokens::{LexError, TokenKind},
};

type Record = (TokenKind, Span);

#[derive(Debug, PartialEq, Eq)]
enum StorageError {
    Ledger(Refusal),
    InvalidLayout,
    PageLimit,
    /// Number of successfully guarded acquisitions before the null.
    AllocationNull {
        acquired: usize,
    },
}

struct Plan {
    page_slots: NonZeroUsize,
    pages: usize,
    page_layout: Layout,
    directory_layout: Layout,
    work: usize,
    live_bytes: usize,
}

impl Plan {
    fn for_records(count: usize, page_slots: NonZeroUsize) -> Result<Option<Self>, StorageError> {
        if count == 0 {
            return Ok(None);
        }

        // Nonzero divisor; ceiling division avoids count + page_slots - 1.
        let pages = count.div_ceil(page_slots.get());
        let page_layout =
            Layout::array::<Record>(page_slots.get()).map_err(|_| StorageError::InvalidLayout)?;
        let directory_layout =
            Layout::array::<Page>(pages).map_err(|_| StorageError::InvalidLayout)?;
        if page_layout.size() == 0 || directory_layout.size() == 0 {
            return Err(StorageError::InvalidLayout);
        }

        // Every page has the same full slot capacity, even the partial last page.
        let live_bytes = pages
            .checked_mul(page_layout.size())
            .and_then(|bytes| bytes.checked_add(directory_layout.size()))
            .ok_or(StorageError::InvalidLayout)?;
        // One unit each for initializing and destroying a record or directory
        // entry, and one per block's acquisition/retirement pair. This bounds
        // trial operations, not allocator execution time. Old cleanup was
        // charged with the old build; indexed lookup is constant work.
        let work = count
            .checked_mul(2)
            .and_then(|records| pages.checked_mul(3)?.checked_add(records))
            .and_then(|work| work.checked_add(1))
            .ok_or(StorageError::InvalidLayout)?;

        Ok(Some(Self {
            page_slots,
            pages,
            page_layout,
            directory_layout,
            work,
            live_bytes,
        }))
    }
}

struct Acquisitions {
    // One-based, reset for each build. Directory first, then pages in order.
    fail_at: Option<NonZeroUsize>,
    acquired: usize,
}

// The unique allocation guard, including the initialized prefix length.
// Page and Directory supply the element type and run its drop glue before
// this field releases the raw bytes, including during unwinding.
struct Block {
    ptr: NonNull<u8>,
    layout: Layout,
    initialized: usize,
}

impl Block {
    fn new(layout: Layout, acquisitions: &mut Acquisitions) -> Result<Self, StorageError> {
        if layout.size() == 0 {
            return Err(StorageError::InvalidLayout);
        }
        let next = acquisitions
            .acquired
            .checked_add(1)
            .ok_or(StorageError::InvalidLayout)?;
        let raw = if acquisitions.fail_at == NonZeroUsize::new(next) {
            ptr::null_mut()
        } else {
            // SAFETY: Layout is valid and the size was checked to be nonzero.
            // A non-null result is immediately owned below with this same layout.
            unsafe { alloc(layout) }
        };
        let block = Self {
            ptr: NonNull::new(raw).ok_or(StorageError::AllocationNull {
                acquired: acquisitions.acquired,
            })?,
            layout,
            initialized: 0,
        };
        acquisitions.acquired = next;
        Ok(block)
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        // SAFETY: This non-Copy guard owns the non-null result of alloc with
        // this exact layout. Its typed owner drops the initialized prefix first;
        // no reference to the storage can survive that owner's drop.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct Page(Block);

impl Page {
    fn new(
        plan: &Plan,
        records: &[Record],
        acquisitions: &mut Acquisitions,
    ) -> Result<Self, StorageError> {
        assert!(!records.is_empty() && records.len() <= plan.page_slots.get());
        let mut page = Self(Block::new(plan.page_layout, acquisitions)?);
        // SAFETY: The fresh allocation is aligned and sized for the planned
        // Record array. The input slice is valid and cannot overlap this new
        // block. Record is Copy; after the non-unwinding copy, the prefix is
        // marked initialized before any other operation can observe or drop it.
        unsafe {
            ptr::copy_nonoverlapping(
                records.as_ptr(),
                page.0.ptr.cast::<Record>().as_ptr(),
                records.len(),
            );
        }
        page.0.initialized = records.len();
        Ok(page)
    }

    fn push_record(&mut self, record: Record) {
        let index = self.0.initialized;
        let slots = self.0.layout.size() / std::mem::size_of::<Record>();
        let next = index
            .checked_add(1)
            .filter(|&next| next <= slots)
            .expect("a pending page has an unused record slot");
        // SAFETY: Page owns an aligned Record array and &mut self gives
        // exclusive access. The layout-derived bound puts index at the first
        // unused slot inside that allocation. Write cannot unwind, and the
        // initialized prefix is published immediately afterwards.
        unsafe {
            self.0
                .ptr
                .cast::<Record>()
                .as_ptr()
                .add(index)
                .write(record);
        }
        self.0.initialized = next;
    }

    fn get(&self, index: usize) -> Option<&Record> {
        if index >= self.0.initialized {
            return None;
        }
        // SAFETY: The index is in the initialized prefix of the aligned Record
        // array. The shared borrow prevents mutation or drop while it is used.
        Some(unsafe { &*self.0.ptr.cast::<Record>().as_ptr().add(index) })
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        if self.0.initialized != 0 {
            let prefix = ptr::slice_from_raw_parts_mut(
                self.0.ptr.cast::<Record>().as_ptr(),
                self.0.initialized,
            );
            // SAFETY: The live block contains exactly this initialized prefix.
            // Drop has exclusive access; unused slots stay untouched, and the
            // Block field frees the allocation after the prefix is destroyed.
            unsafe { ptr::drop_in_place(prefix) };
        }
    }
}

struct Directory {
    block: Block,
    slots: usize,
}

impl Directory {
    fn new(plan: &Plan, acquisitions: &mut Acquisitions) -> Result<Self, StorageError> {
        Ok(Self {
            block: Block::new(plan.directory_layout, acquisitions)?,
            slots: plan.pages,
        })
    }

    fn push(&mut self, page: Page) {
        let next = self.block.initialized;
        assert!(next < self.slots);
        // SAFETY: The planned Page array has room for this uninitialized slot
        // after the prefix. The write moves the unique owner into the directory;
        // write and the bounded increment cannot unwind between the two steps.
        unsafe {
            self.block.ptr.cast::<Page>().as_ptr().add(next).write(page);
        }
        self.block.initialized = next + 1;
    }

    fn get(&self, index: usize) -> Option<&Page> {
        if index >= self.block.initialized {
            return None;
        }
        // SAFETY: Only installed Page owners are in the prefix. This borrow of
        // the directory keeps the entry live and prevents mutation.
        Some(unsafe { &*self.block.ptr.cast::<Page>().as_ptr().add(index) })
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        if self.block.initialized != 0 {
            let prefix = ptr::slice_from_raw_parts_mut(
                self.block.ptr.cast::<Page>().as_ptr(),
                self.block.initialized,
            );
            // SAFETY: Drop has exclusive access to the initialized Page owners
            // in this live block. Slice drop glue drops the whole prefix, not
            // unused slots, before the Block field deallocates the directory.
            unsafe { ptr::drop_in_place(prefix) };
        }
    }
}

const WRITER_PAGES: usize = 4;

struct PageWriter<'ledger> {
    // Field order retires pending and installed pages and the directory
    // before releasing the live charge, including on refusal or unwind.
    pending: Option<Page>,
    directory: Directory,
    reservation: Reservation<'ledger>,
    plan: Plan,
    acquisitions: Acquisitions,
    capacity: usize,
    count: usize,
}

impl<'ledger> PageWriter<'ledger> {
    fn new(
        ledger: &'ledger Ledger,
        page_slots: NonZeroUsize,
        fail_at: Option<NonZeroUsize>,
    ) -> Result<Self, StorageError> {
        let capacity = WRITER_PAGES
            .checked_mul(page_slots.get())
            .ok_or(StorageError::InvalidLayout)?;
        // Validate the full geometry without admitting the maximum cost.
        let plan = Plan::for_records(capacity, page_slots)?.ok_or(StorageError::InvalidLayout)?;
        let reservation = ledger
            .admit(Phase::Lex, 1, plan.directory_layout.size())
            .map_err(StorageError::Ledger)?;
        let mut acquisitions = Acquisitions {
            fail_at,
            acquired: 0,
        };
        let directory = Directory::new(&plan, &mut acquisitions)?;
        Ok(Self {
            pending: None,
            directory,
            reservation,
            plan,
            acquisitions,
            capacity,
            count: 0,
        })
    }

    fn offer(&mut self, record: Record) -> Result<(), StorageError> {
        if self.count >= self.capacity {
            return Err(StorageError::PageLimit);
        }
        if let Some(page) = &mut self.pending {
            self.reservation
                .extend(Phase::Lex, 2, 0)
                .map_err(StorageError::Ledger)?;
            page.push_record(record);
        } else {
            // Opening prepays three page units and two record units. Charge
            // the full page layout before making the guarded acquisition.
            self.reservation
                .extend(Phase::Lex, 5, self.plan.page_layout.size())
                .map_err(StorageError::Ledger)?;
            self.pending = Some(Page::new(
                &self.plan,
                std::slice::from_ref(&record),
                &mut self.acquisitions,
            )?);
        }
        // The cap check leaves room for one more record. A multiple of the
        // slots per page finishes the pending page, also when S is one.
        self.count += 1;
        if self.count.is_multiple_of(self.plan.page_slots.get()) {
            self.install_pending();
        }
        Ok(())
    }

    fn install_pending(&mut self) {
        if let Some(page) = self.pending.take() {
            self.directory.push(page);
        }
    }
}

struct TokenPages<'source, 'ledger> {
    source: &'source str,
    count: usize,
    page_slots: NonZeroUsize,
    // Drop order matters: both page blocks and the directory go before the charge.
    directory: Option<Directory>,
    reservation: Reservation<'ledger>,
}

impl<'source, 'ledger> TokenPages<'source, 'ledger> {
    fn from_lexer(
        ledger: &'ledger Ledger,
        source: &'source str,
        page_slots: NonZeroUsize,
        fail_at: Option<NonZeroUsize>,
    ) -> Result<(Self, Vec<LexError>), StorageError> {
        let mut writer = PageWriter::new(ledger, page_slots, fail_at)?;
        // Scanner success includes an accepted EOF. Neither the view nor
        // diagnostics can escape on a scan-work or record refusal.
        let errors = lex_records_into(
            source,
            || {
                ledger
                    .admit(Phase::Lex, 1, 0)
                    .map(drop)
                    .map_err(StorageError::Ledger)
            },
            |kind, span| writer.offer((kind, span)),
        )?;
        writer.install_pending();
        Ok((
            Self {
                source,
                count: writer.count,
                page_slots: writer.plan.page_slots,
                directory: Some(writer.directory),
                reservation: writer.reservation,
            },
            errors,
        ))
    }

    fn new(
        ledger: &'ledger Ledger,
        source: &'source str,
        records: &[Record],
        page_slots: NonZeroUsize,
        fail_at: Option<NonZeroUsize>,
    ) -> Result<Self, StorageError> {
        let plan = Plan::for_records(records.len(), page_slots)?;
        let (work, live_bytes) = plan
            .as_ref()
            .map_or((0, 0), |plan| (plan.work, plan.live_bytes));
        let reservation = ledger
            .admit(Phase::Lex, work, live_bytes)
            .map_err(StorageError::Ledger)?;
        let directory = if let Some(plan) = plan {
            let mut acquisitions = Acquisitions {
                fail_at,
                acquired: 0,
            };
            // A partial directory is a later local than the reservation:
            // failure or unwind drops it and all its pages before the charge.
            let mut directory = Directory::new(&plan, &mut acquisitions)?;
            for chunk in records.chunks(page_slots.get()) {
                directory.push(Page::new(&plan, chunk, &mut acquisitions)?);
            }
            Some(directory)
        } else {
            None
        };

        Ok(Self {
            source,
            count: records.len(),
            page_slots,
            directory,
            reservation,
        })
    }

    fn replace(
        &mut self,
        source: &'source str,
        records: &[Record],
        fail_at: Option<NonZeroUsize>,
    ) -> Result<(), StorageError> {
        // The same ledger already counts self. Do not change it or charge it
        // again while admitting and constructing the separately owned candidate.
        let candidate = Self::new(
            self.reservation.ledger,
            source,
            records,
            self.page_slots,
            fail_at,
        )?;
        drop(std::mem::replace(self, candidate));
        Ok(())
    }

    fn as_view(&self) -> TokenView<'_> {
        TokenView {
            source: self.source,
            count: self.count,
            page_slots: self.page_slots,
            directory: self.directory.as_ref(),
        }
    }
}

struct TokenView<'owner> {
    source: &'owner str,
    count: usize,
    page_slots: NonZeroUsize,
    directory: Option<&'owner Directory>,
}

impl<'owner> TokenView<'owner> {
    fn get(&self, index: usize) -> Option<(&'owner Record, &'owner str)> {
        if index >= self.count {
            return None;
        }
        let slots = self.page_slots.get();
        let page = self.directory?.get(index / slots)?;
        let record = page.get(index % slots)?;
        let span = record.1;
        let spelling = self.source.get(span.start..span.end)?;
        Some((record, spelling))
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::num::NonZeroUsize;
    use std::ptr;

    use super::{Page, Record, StorageError, TokenPages};
    use crate::v2::resource::{Ledger, Limits, Phase, Refusal};
    use crate::v2::{
        span::Span,
        tokens::{LexError, TokenKind},
    };

    const SLOTS: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    const SOURCE: &str = "\u{feff}é猫";
    const BOM: Record = (TokenKind::Unknown, Span { start: 0, end: 3 });
    const E_ACUTE: Record = (TokenKind::Generic, Span { start: 3, end: 5 });
    const CAT: Record = (TokenKind::Generic, Span { start: 5, end: 8 });
    const RECORDS: [Record; 3] = [BOM, E_ACUTE, CAT];
    const OLD: Record = (TokenKind::Keyword, Span { start: 0, end: 3 });

    // One directory entry and a two-record array per page, including unused slots.
    const OLD_BYTES: usize = size_of::<Page>() + size_of::<[Record; 2]>();
    const CANDIDATE_BYTES: usize = size_of::<[Page; 2]>() + size_of::<[[Record; 2]; 2]>();
    // Old: one record and one entry initialized/destroyed, two blocks.
    // Candidate: three records and two entries initialized/destroyed, three blocks.
    const OLD_WORK: usize = 6;
    const CANDIDATE_WORK: usize = 13;

    fn ledger(work: usize, live_bytes: usize) -> Ledger {
        Ledger::new(Limits {
            work: [work, 0, 0],
            live_bytes,
        })
    }

    fn old_store(ledger: &Ledger) -> TokenPages<'static, '_> {
        TokenPages::new(ledger, "old", &[OLD], SLOTS, None).expect("the old store fits")
    }

    fn assert_remaining(ledger: &Ledger, work: usize, live_bytes: usize) {
        let _held = ledger
            .admit(Phase::Lex, work, live_bytes)
            .expect("the expected remainder fits");
        assert_eq!(
            (
                ledger.admit(Phase::Lex, 1, 0).map(drop),
                ledger.admit(Phase::Lex, 0, 1).map(drop),
            ),
            (Err(Refusal::Work), Err(Refusal::LiveBytes)),
        );
    }

    fn assert_old_store(store: &TokenPages<'_, '_>) {
        let view = store.as_view();
        assert_eq!((view.get(0), view.get(1)), (Some((&OLD, "old")), None));
    }

    const DIRECTORY_BYTES: usize = size_of::<[Page; 4]>();
    const PAGE_BYTES: usize = size_of::<[Record; 2]>();
    const FOUR_PAGE_BYTES: usize = DIRECTORY_BYTES + size_of::<[[Record; 2]; 4]>();

    #[test]
    fn lexer_writer_accepts_eof_as_eighth_record_at_exact_limits() {
        let ledger = ledger(29, FOUR_PAGE_BYTES);

        let (store, errors) = TokenPages::from_lexer(&ledger, ";;;;;;;", SLOTS, None).unwrap();

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Semicolon, Span::new(0, 1)), ";")),
                Some(((TokenKind::Semicolon, Span::new(1, 2)), ";")),
                Some(((TokenKind::Semicolon, Span::new(2, 3)), ";")),
                Some(((TokenKind::Semicolon, Span::new(3, 4)), ";")),
                Some(((TokenKind::Semicolon, Span::new(4, 5)), ";")),
                Some(((TokenKind::Semicolon, Span::new(5, 6)), ";")),
                Some(((TokenKind::Semicolon, Span::new(6, 7)), ";")),
                Some(((TokenKind::Eof, Span::new(7, 7)), "")),
                None,
            ],
        );
        assert_eq!(errors.as_slice(), &[] as &[LexError]);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_refuses_ninth_eof_before_other_limits() {
        let ledger = ledger(29, FOUR_PAGE_BYTES);

        let result = TokenPages::from_lexer(&ledger, ";;;;;;;;", SLOTS, NonZeroUsize::new(6));

        assert_eq!(result.map(drop), Err(StorageError::PageLimit));
        assert_remaining(&ledger, 0, FOUR_PAGE_BYTES);
    }

    fn assert_lexer_writer_only_eof(source: &str, end: usize, work: usize) {
        let ledger = ledger(work, DIRECTORY_BYTES + PAGE_BYTES);

        let (store, errors) = TokenPages::from_lexer(&ledger, source, SLOTS, None).unwrap();

        let view = store.as_view();
        assert_eq!(
            (view.get(0), view.get(1), errors.as_slice()),
            (
                Some((&(TokenKind::Eof, Span::new(end, end)), "")),
                None,
                &[][..]
            ),
        );
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_empty_source_stores_only_eof() {
        assert_lexer_writer_only_eof("", 0, 6);
    }

    #[test]
    fn lexer_writer_trivia_only_source_stores_only_eof() {
        // Six writer units, plus entry and four bytes (#, space, x, CR).
        assert_lexer_writer_only_eof("\u{feff} # x\r\n", 9, 11);
    }

    #[test]
    fn lexer_writer_leading_and_trailing_line_comments_fit_at_exact_work() {
        // 13 writer units, 3 for entry/#/LF, and 5 for entry/#/both bytes of é/CR.
        let source = "\u{feff}#\n$x #é\r\n.5";
        let ledger = ledger(21, DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>());

        let (store, errors) = TokenPages::from_lexer(&ledger, source, SLOTS, None).unwrap();

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Variable, Span::new(5, 7)), "$x")),
                Some(((TokenKind::Number, Span::new(13, 15)), ".5")),
                Some(((TokenKind::Eof, Span::new(15, 15)), "")),
                None,
            ],
        );
        assert_eq!(errors.as_slice(), &[] as &[LexError]);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_nested_line_comment_at_eof_preserves_diagnostic_order() {
        // Eight writer units and three for entry, #, and ).
        let ledger = ledger(11, DIRECTORY_BYTES + PAGE_BYTES);

        let (store, errors) = TokenPages::from_lexer(&ledger, "\"$(#)", SLOTS, None).unwrap();

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::StringDq, Span::new(0, 5)), "\"$(#)")),
                Some(((TokenKind::Eof, Span::new(5, 5)), "")),
                None,
            ],
        );
        assert_eq!(
            errors,
            [
                LexError {
                    span: Span::new(1, 5),
                    message: "unterminated subexpression in string".into(),
                },
                LexError {
                    span: Span::new(0, 5),
                    message: "unterminated string".into(),
                },
            ],
        );
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_nested_line_comment_refuses_closing_paren_on_work() {
        // One directory unit and two for entry and #; ) is not admitted.
        let live = DIRECTORY_BYTES + PAGE_BYTES;
        let ledger = ledger(3, live);

        let result = TokenPages::from_lexer(&ledger, "\"$(#)", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_remaining(&ledger, 0, live);
    }

    #[test]
    fn lexer_writer_full_before_line_comment_refuses_cr_on_work() {
        // 29 writer units, then entry, #, and both bytes of é; CR is not admitted.
        let ledger = ledger(33, FOUR_PAGE_BYTES);

        let result = TokenPages::from_lexer(&ledger, ";;;;;;;;\n#é\r\n", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_remaining(&ledger, 0, FOUR_PAGE_BYTES);
    }

    #[test]
    fn lexer_writer_full_after_line_comment_refuses_eof_on_page_limit() {
        // 29 writer units and five for entry, #, both bytes of é, and CR.
        let ledger = ledger(34, FOUR_PAGE_BYTES);

        let result = TokenPages::from_lexer(&ledger, ";;;;;;;;\n#é\r\n", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::PageLimit));
        assert_remaining(&ledger, 0, FOUR_PAGE_BYTES);
    }

    #[test]
    fn lexer_writer_trailing_line_comment_refusal_releases_pending_and_installed_pages() {
        // Three semicolons cost 13 with the directory and leave one installed
        // and one pending page. Entry, #, and the first byte of é use the rest.
        let live = DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>();
        let ledger = ledger(16, live);

        let result = TokenPages::from_lexer(&ledger, ";;;$#é\n", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_remaining(&ledger, 0, live);
    }

    #[test]
    fn lexer_writer_block_comment_after_verbatim_fits_at_exact_work() {
        // 15 writer units, 4 for the verbatim scan (including LF), 6 for
        // the block comment (entry, opener #, both bytes of é, and #>).
        let ledger = ledger(25, DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>());

        let (store, errors) = TokenPages::from_lexer(&ledger, "--%<#\n<#é#>.5", SLOTS, None)
            .expect("verbatim tail and block comment fit at 25 units");

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3, 4].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Operator, Span::new(0, 3)), "--%")),
                Some(((TokenKind::VerbatimArgs, Span::new(3, 5)), "<#")),
                Some(((TokenKind::Number, Span::new(12, 14)), ".5")),
                Some(((TokenKind::Eof, Span::new(14, 14)), "")),
                None,
            ],
        );
        assert_eq!(errors, []);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_nested_block_comment_at_eof_preserves_diagnostic_order() {
        // Eight writer units and six for the comment, with no EOF scan unit.
        let ledger = ledger(14, DIRECTORY_BYTES + PAGE_BYTES);

        let (store, errors) = TokenPages::from_lexer(&ledger, "\"$(<#>é#", SLOTS, None)
            .expect("the unterminated contexts fit at 14 units");

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::StringDq, Span::new(0, 9)), "\"$(<#>é#")),
                Some(((TokenKind::Eof, Span::new(9, 9)), "")),
                None,
            ],
        );
        assert_eq!(
            errors,
            [
                LexError {
                    span: Span::new(3, 9),
                    message: "unterminated block comment".into(),
                },
                LexError {
                    span: Span::new(1, 9),
                    message: "unterminated subexpression in string".into(),
                },
                LexError {
                    span: Span::new(0, 9),
                    message: "unterminated string".into(),
                },
            ],
        );
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_lone_less_than_charges_entry_and_stays_redirect() {
        // Eight writer units plus entry even though there is no second byte.
        let ledger = ledger(9, DIRECTORY_BYTES + PAGE_BYTES);

        let (store, errors) = TokenPages::from_lexer(&ledger, "<", SLOTS, None)
            .expect("redirect and EOF fit at nine units");

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Redirect, Span::new(0, 1)), "<")),
                Some(((TokenKind::Eof, Span::new(1, 1)), "")),
                None,
            ],
        );
        assert_eq!(errors, []);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_nonmatching_block_openers_charge_both_bytes_and_preserve_tokens() {
        // 13 writer units, plus entry and a nonmatching second byte for
        // each candidate, leaving `)` to close the nested subexpression.
        let ledger = ledger(17, DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>());

        let (store, errors) = TokenPages::from_lexer(&ledger, "<\"$(<)\"", SLOTS, None)
            .expect("both rejected candidates and three records fit at 17 units");

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Redirect, Span::new(0, 1)), "<")),
                Some(((TokenKind::StringDq, Span::new(1, 7)), "\"$(<)\"")),
                Some(((TokenKind::Eof, Span::new(7, 7)), "")),
                None,
            ],
        );
        assert_eq!(errors, []);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_full_before_block_comment_refuses_on_work_before_page_limit() {
        // Eight semicolons take 29 units; EOF can only be offered after
        // all seven block-comment units have been admitted.
        for (work, expected, next) in [
            (29, StorageError::Ledger(Refusal::Work), "entry"),
            (30, StorageError::Ledger(Refusal::Work), "opener #"),
            (32, StorageError::Ledger(Refusal::Work), "second byte of é"),
            (35, StorageError::Ledger(Refusal::Work), "closing >"),
            (36, StorageError::PageLimit, "EOF offer"),
        ] {
            let ledger = ledger(work, FOUR_PAGE_BYTES);

            let result = TokenPages::from_lexer(&ledger, ";;;;;;;;<#é##>", SLOTS, None);

            assert_eq!(result.map(drop), Err(expected), "at {next}, budget {work}");
            assert_remaining(&ledger, 0, FOUR_PAGE_BYTES);
        }
    }

    #[test]
    fn lexer_writer_nested_block_comment_refusal_releases_candidate_and_preserves_old_store() {
        // The directory, two semicolons, and lone `$` (with its diagnostic)
        // take 13 units and leave one installed and one pending page.
        // Entry, opener #, and the first byte of é use the remaining three.
        let live = DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>();
        let ledger = ledger(OLD_WORK + 16, OLD_BYTES + live);
        let old = old_store(&ledger);

        let result = TokenPages::from_lexer(&ledger, ";;$\"$(<#é", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_old_store(&old);
        assert_remaining(&ledger, 0, live);
    }

    #[test]
    fn lexer_writer_bom_and_unicode_keep_original_byte_spans_and_borrowed_text() {
        let source = String::from("\u{feff}@'\né\n'@;");
        // 13 writer units and seven for the suffix (entry and six bytes).
        let ledger = ledger(20, DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>());

        let (store, errors) = TokenPages::from_lexer(&ledger, &source, SLOTS, None).unwrap();

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::HereStringSq, Span::new(3, 11)), "@'\né\n'@")),
                Some(((TokenKind::Semicolon, Span::new(11, 12)), ";")),
                Some(((TokenKind::Eof, Span::new(12, 12)), "")),
                None,
            ],
        );
        let spelling = view.get(0).unwrap().1;
        assert!(
            ptr::eq(spelling, &source[3..11]),
            "expected source pointer {:p}, actual spelling pointer {spelling:p}",
            &source[3..11],
        );
        assert_eq!(errors.as_slice(), &[] as &[LexError]);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_full_before_here_string_refuses_on_work_before_page_limit() {
        // Eight semicolons take 29 units. The seventh suffix unit admits
        // the closing `@`, after which the writer can refuse the token offer.
        for (work, expected) in [
            (35, StorageError::Ledger(Refusal::Work)),
            (36, StorageError::PageLimit),
        ] {
            let ledger = ledger(work, FOUR_PAGE_BYTES);

            let result = TokenPages::from_lexer(&ledger, ";;;;;;;;@\"\né\n\"@", SLOTS, None);

            assert_eq!(result.map(drop), Err(expected), "at budget {work}");
            assert_remaining(&ledger, 0, FOUR_PAGE_BYTES);
        }
    }

    #[test]
    fn lexer_writer_nested_here_string_at_eof_preserves_diagnostic_order() {
        // 20 writer units and nine for the unterminated suffix (entry and eight bytes).
        let ledger = ledger(29, DIRECTORY_BYTES + size_of::<[[Record; 2]; 3]>());

        let (_store, errors) = TokenPages::from_lexer(&ledger, ";;$\"$(@'\n猫éé", SLOTS, None)
            .expect("the unterminated contexts fit at 29 units");

        assert_eq!(
            errors,
            [
                LexError {
                    span: Span::new(2, 3),
                    message: "lone `$`".into(),
                },
                LexError {
                    span: Span::new(6, 16),
                    message: "unterminated here-string".into(),
                },
                LexError {
                    span: Span::new(4, 16),
                    message: "unterminated subexpression in string".into(),
                },
                LexError {
                    span: Span::new(3, 16),
                    message: "unterminated string".into(),
                },
            ],
        );
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_nested_here_string_refusal_releases_candidate_and_preserves_old_store() {
        // After 13 writer units the candidate has an installed page, a
        // pending page, and a lone `$` diagnostic. The eight remaining
        // units allow suffix entry and all but the last byte of the final é.
        let live = DIRECTORY_BYTES + size_of::<[[Record; 2]; 3]>();
        let ledger = ledger(OLD_WORK + 21, OLD_BYTES + live);
        let old = old_store(&ledger);

        let result = TokenPages::from_lexer(&ledger, ";;$\"$(@'\n猫éé", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_old_store(&old);
        assert_remaining(&ledger, 0, live);
    }

    #[test]
    fn lexer_writer_stop_parsing_keeps_verbatim_tail_and_crlf_record_order() {
        let ledger = ledger(20, DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>());

        let (store, errors) = TokenPages::from_lexer(&ledger, "--% #x\r\n;", SLOTS, None).unwrap();

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3, 4].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Operator, Span::new(0, 3)), "--%")),
                Some(((TokenKind::VerbatimArgs, Span::new(3, 6)), " #x")),
                Some(((TokenKind::Semicolon, Span::new(8, 9)), ";")),
                Some(((TokenKind::Eof, Span::new(9, 9)), "")),
                None,
            ],
        );
        assert_eq!(errors.as_slice(), &[] as &[LexError]);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_stop_parsing_at_eof_charges_entry_and_stores_no_tail() {
        let ledger = ledger(9, DIRECTORY_BYTES + PAGE_BYTES);

        let (store, errors) = TokenPages::from_lexer(&ledger, "--%", SLOTS, None)
            .expect("expected marker and EOF at 9 work units");

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Operator, Span::new(0, 3)), "--%")),
                Some(((TokenKind::Eof, Span::new(3, 3)), "")),
                None,
            ],
        );
        assert_eq!(errors.as_slice(), &[] as &[LexError]);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_stop_parsing_with_unicode_charges_both_tail_bytes() {
        let ledger = ledger(16, DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>());

        let (store, errors) = TokenPages::from_lexer(&ledger, "--%é", SLOTS, None)
            .expect("expected marker, verbatim tail, and EOF at 16 work units");

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Operator, Span::new(0, 3)), "--%")),
                Some(((TokenKind::VerbatimArgs, Span::new(3, 5)), "é")),
                Some(((TokenKind::Eof, Span::new(5, 5)), "")),
                None,
            ],
        );
        assert_eq!(errors.as_slice(), &[] as &[LexError]);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_full_at_marker_refuses_second_unicode_byte_on_work() {
        // 29 for the directory and eight records, then entry and one byte.
        let ledger = ledger(31, FOUR_PAGE_BYTES);

        let result = TokenPages::from_lexer(&ledger, ";;;;;;;--%é", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_remaining(&ledger, 0, FOUR_PAGE_BYTES);
    }

    #[test]
    fn lexer_writer_full_at_marker_refuses_tail_offer_after_admitting_both_bytes() {
        // Both bytes fit after the directory, eight records, and tail entry.
        let ledger = ledger(32, FOUR_PAGE_BYTES);

        let result = TokenPages::from_lexer(&ledger, ";;;;;;;--%é", SLOTS, None);

        assert_eq!(result.map(drop), Err(StorageError::PageLimit));
        assert_remaining(&ledger, 0, FOUR_PAGE_BYTES);
    }

    #[test]
    fn lexer_writer_malformed_source_preserves_record_and_diagnostic_order() {
        let ledger = ledger(13, DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>());

        let (store, errors) = TokenPages::from_lexer(&ledger, "$ @\n", SLOTS, None).unwrap();

        let view = store.as_view();
        assert_eq!(
            [0, 1, 2, 3].map(|index| view.get(index).map(|(&r, text)| (r, text))),
            [
                Some(((TokenKind::Unknown, Span::new(0, 1)), "$")),
                Some(((TokenKind::Unknown, Span::new(2, 3)), "@")),
                Some(((TokenKind::Eof, Span::new(4, 4)), "")),
                None,
            ],
        );
        assert_eq!(
            errors,
            [
                LexError {
                    span: Span::new(0, 1),
                    message: "lone `$`".into(),
                },
                LexError {
                    span: Span::new(2, 3),
                    message: "lone `@`".into(),
                },
            ],
        );
    }

    fn assert_lexer_writer_rejects_geometry(slots: NonZeroUsize) {
        let ledger = ledger(1, DIRECTORY_BYTES);

        let result = TokenPages::from_lexer(&ledger, "", slots, NonZeroUsize::new(1));

        assert_eq!(result.map(drop), Err(StorageError::InvalidLayout));
        assert_remaining(&ledger, 1, DIRECTORY_BYTES);
    }

    #[test]
    fn lexer_writer_rejects_capacity_overflow_before_admission() {
        assert_lexer_writer_rejects_geometry(NonZeroUsize::MAX);
    }

    #[test]
    fn lexer_writer_rejects_unrepresentable_page_layout_before_admission() {
        assert_lexer_writer_rejects_geometry(NonZeroUsize::new(usize::MAX / 4).unwrap());
    }

    #[test]
    fn lexer_writer_directory_one_byte_short_refuses_before_acquisition() {
        let ledger = ledger(1, DIRECTORY_BYTES - 1);

        let result = TokenPages::from_lexer(&ledger, "", SLOTS, NonZeroUsize::new(1));

        assert_eq!(
            result.map(drop),
            Err(StorageError::Ledger(Refusal::LiveBytes))
        );
        assert_remaining(&ledger, 1, DIRECTORY_BYTES - 1);
    }

    #[test]
    fn lexer_writer_directory_with_both_limits_short_refuses_work_first() {
        let ledger = ledger(0, DIRECTORY_BYTES - 1);

        let result = TokenPages::from_lexer(&ledger, "", SLOTS, NonZeroUsize::new(1));

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_remaining(&ledger, 0, DIRECTORY_BYTES - 1);
    }

    #[test]
    fn lexer_writer_page_one_byte_short_refuses_without_spending_page_work() {
        let ledger = ledger(6, DIRECTORY_BYTES + PAGE_BYTES - 1);

        let result = TokenPages::from_lexer(&ledger, "", SLOTS, NonZeroUsize::new(2));

        assert_eq!(
            result.map(drop),
            Err(StorageError::Ledger(Refusal::LiveBytes))
        );
        assert_remaining(&ledger, 5, DIRECTORY_BYTES + PAGE_BYTES - 1);
    }

    #[test]
    fn lexer_writer_page_one_work_short_and_one_byte_short_refuses_work_first() {
        let ledger = ledger(5, DIRECTORY_BYTES + PAGE_BYTES - 1);

        let result = TokenPages::from_lexer(&ledger, "", SLOTS, NonZeroUsize::new(2));

        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_remaining(&ledger, 4, DIRECTORY_BYTES + PAGE_BYTES - 1);
    }

    #[test]
    fn lexer_writer_append_refusal_releases_pending_and_installed_pages() {
        let live = DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>();
        let ledger = ledger(14, live);

        let result = TokenPages::from_lexer(&ledger, "$ @;", SLOTS, None);

        // Three records left one installed page and one pending page.
        assert_eq!(result.map(drop), Err(StorageError::Ledger(Refusal::Work)));
        assert_remaining(&ledger, 1, live);
    }

    #[test]
    fn lexer_writer_moved_store_holds_live_charge_alongside_old_store() {
        let ledger = ledger(OLD_WORK + 6, OLD_BYTES + DIRECTORY_BYTES + PAGE_BYTES);
        let old = old_store(&ledger);
        let (store, _) = TokenPages::from_lexer(&ledger, "", SLOTS, None).unwrap();

        let moved = store;

        assert_eq!(
            moved.as_view().get(0),
            Some((&(TokenKind::Eof, Span::new(0, 0)), ""))
        );
        assert_old_store(&old);
        assert_remaining(&ledger, 0, 0);
    }

    #[test]
    fn lexer_writer_dropping_moved_store_releases_only_its_live_charge() {
        let ledger = ledger(OLD_WORK + 6, OLD_BYTES + DIRECTORY_BYTES + PAGE_BYTES);
        let old = old_store(&ledger);
        let (store, _) = TokenPages::from_lexer(&ledger, "", SLOTS, None).unwrap();
        let moved = store;

        drop(moved);

        assert_old_store(&old);
        assert_remaining(&ledger, 0, DIRECTORY_BYTES + PAGE_BYTES);
    }

    fn assert_lexer_writer_null(fail_at: usize, acquired: usize, remaining_work: usize) {
        let live = DIRECTORY_BYTES + size_of::<[[Record; 2]; 2]>();
        let ledger = ledger(OLD_WORK + 14, OLD_BYTES + live);
        let old = old_store(&ledger);

        let result = TokenPages::from_lexer(&ledger, "$ @", SLOTS, NonZeroUsize::new(fail_at));

        assert_eq!(
            result.map(drop),
            Err(StorageError::AllocationNull { acquired })
        );
        assert_old_store(&old);
        assert_remaining(&ledger, remaining_work, live);
    }

    #[test]
    fn lexer_writer_null_directory_spends_one_unit_and_keeps_old_store() {
        assert_lexer_writer_null(1, 0, 13);
    }

    #[test]
    fn lexer_writer_null_first_page_spends_six_units_and_keeps_old_store() {
        assert_lexer_writer_null(2, 1, 8);
    }

    #[test]
    fn lexer_writer_null_second_page_spends_thirteen_units_and_keeps_old_store() {
        assert_lexer_writer_null(3, 2, 1);
    }

    #[test]
    fn token_pages_empty_at_zero_limits_never_acquires() {
        let ledger = ledger(0, 0);

        let result = TokenPages::new(&ledger, "", &[], NonZeroUsize::MAX, NonZeroUsize::new(1));

        let store = result.expect("empty storage needs no allocation");
        assert_eq!(store.as_view().get(0), None);
    }

    #[test]
    fn token_pages_overflowing_page_layout_is_refused_before_acquisition() {
        let ledger = ledger(1, 1);

        let result = TokenPages::new(
            &ledger,
            SOURCE,
            &[BOM],
            NonZeroUsize::MAX,
            NonZeroUsize::new(1),
        );

        assert_eq!(result.map(drop), Err(StorageError::InvalidLayout));
        assert_remaining(&ledger, 1, 1);
    }

    #[test]
    fn token_pages_replacement_at_exact_peak_publishes_and_retires_old() {
        let ledger = ledger(OLD_WORK + CANDIDATE_WORK, OLD_BYTES + CANDIDATE_BYTES);
        let mut store = old_store(&ledger);

        let result = store.replace(SOURCE, &RECORDS, None);

        assert_eq!(result, Ok(()));
        assert_eq!(store.as_view().get(2), Some((&CAT, "猫")));
        assert_remaining(&ledger, 0, OLD_BYTES);
    }

    #[test]
    fn token_pages_replacement_one_byte_short_refuses_without_charging_candidate() {
        let ledger = ledger(OLD_WORK + CANDIDATE_WORK, OLD_BYTES + CANDIDATE_BYTES - 1);
        let mut store = old_store(&ledger);

        let result = store.replace(SOURCE, &RECORDS, NonZeroUsize::new(1));

        assert_eq!(result, Err(StorageError::Ledger(Refusal::LiveBytes)));
        assert_old_store(&store);
        assert_remaining(&ledger, CANDIDATE_WORK, CANDIDATE_BYTES - 1);
    }

    #[test]
    fn token_pages_replacement_with_both_limits_exceeded_refuses_work_first() {
        let ledger = ledger(
            OLD_WORK + CANDIDATE_WORK - 1,
            OLD_BYTES + CANDIDATE_BYTES - 1,
        );
        let mut store = old_store(&ledger);

        let result = store.replace(SOURCE, &RECORDS, NonZeroUsize::new(1));

        assert_eq!(result, Err(StorageError::Ledger(Refusal::Work)));
        assert_old_store(&store);
        assert_remaining(&ledger, CANDIDATE_WORK - 1, CANDIDATE_BYTES - 1);
    }

    #[test]
    fn token_pages_replacement_null_on_first_acquisition_keeps_old_and_spends_work() {
        let ledger = ledger(OLD_WORK + CANDIDATE_WORK, OLD_BYTES + CANDIDATE_BYTES);
        let mut store = old_store(&ledger);

        let result = store.replace(SOURCE, &RECORDS, NonZeroUsize::new(1));

        assert_eq!(result, Err(StorageError::AllocationNull { acquired: 0 }));
        assert_old_store(&store);
        assert_remaining(&ledger, 0, CANDIDATE_BYTES);
    }

    #[test]
    fn token_pages_replacement_null_on_second_acquisition_keeps_old_and_spends_work() {
        let ledger = ledger(OLD_WORK + CANDIDATE_WORK, OLD_BYTES + CANDIDATE_BYTES);
        let mut store = old_store(&ledger);

        let result = store.replace(SOURCE, &RECORDS, NonZeroUsize::new(2));

        assert_eq!(result, Err(StorageError::AllocationNull { acquired: 1 }));
        assert_old_store(&store);
        assert_remaining(&ledger, 0, CANDIDATE_BYTES);
    }

    #[test]
    fn token_pages_replacement_null_after_installing_a_page_keeps_old_and_spends_work() {
        let ledger = ledger(OLD_WORK + CANDIDATE_WORK, OLD_BYTES + CANDIDATE_BYTES);
        let mut store = old_store(&ledger);

        let result = store.replace(SOURCE, &RECORDS, NonZeroUsize::new(3));

        assert_eq!(result, Err(StorageError::AllocationNull { acquired: 2 }));
        assert_old_store(&store);
        assert_remaining(&ledger, 0, CANDIDATE_BYTES);
    }

    #[test]
    fn token_view_across_pages_borrows_original_bom_and_multibyte_spelling() {
        let ledger = ledger(CANDIDATE_WORK, CANDIDATE_BYTES);
        let bytes = *b"\xef\xbb\xbf\xc3\xa9\xe7\x8c\xab";
        let source = std::str::from_utf8(&bytes).expect("fixture is BOM, é, then 猫");
        let store =
            TokenPages::new(&ledger, source, &RECORDS, SLOTS, None).expect("two pages fit exactly");
        let view = store.as_view();

        let entries = [view.get(0), view.get(1), view.get(2)];

        assert_eq!(
            entries,
            [
                Some((&BOM, "\u{feff}")),
                Some((&E_ACUTE, "é")),
                Some((&CAT, "猫")),
            ],
        );
        let (_, spelling) = entries[2].expect("the second page contains 猫");
        assert!(
            ptr::eq(spelling, &source[5..8]),
            "expected source pointer {:p}, actual spelling pointer {spelling:p}",
            &source[5..8],
        );
    }

    #[test]
    fn token_view_rejects_first_unused_slot_of_partial_final_page() {
        let ledger = ledger(CANDIDATE_WORK, CANDIDATE_BYTES);
        let store = TokenPages::new(&ledger, SOURCE, &RECORDS, SLOTS, None)
            .expect("three records fit across two pages");

        let entry = store.as_view().get(3);

        assert_eq!(entry, None);
    }

    #[test]
    fn token_view_rejects_a_span_inside_an_original_utf8_character() {
        let ledger = ledger(OLD_WORK, OLD_BYTES);
        let record = (TokenKind::Unknown, Span { start: 4, end: 5 });
        let store = TokenPages::new(&ledger, SOURCE, &[record], SLOTS, None)
            .expect("the trial stores spans without slicing");

        let entry = store.as_view().get(0);

        assert_eq!(entry, None);
    }

    #[test]
    fn token_pages_moved_owner_keeps_live_charge() {
        let ledger = ledger(OLD_WORK, OLD_BYTES);
        let store = old_store(&ledger);

        let moved = store;

        assert_old_store(&moved);
        assert_eq!(
            ledger.admit(Phase::Lex, 0, 1).map(drop),
            Err(Refusal::LiveBytes),
        );
    }

    #[test]
    fn token_pages_dropping_moved_owner_releases_live_but_not_work() {
        let ledger = ledger(OLD_WORK, OLD_BYTES);
        let store = old_store(&ledger);
        let moved = store;

        drop(moved);

        assert_remaining(&ledger, 0, OLD_BYTES);
    }
}
