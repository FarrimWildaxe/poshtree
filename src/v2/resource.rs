//! Allocation-free, invocation-local accounting for ordinary analysis resources.

use std::cell::Cell;

#[cfg(test)]
use crate::v2::span::Span;

mod token_page;

enum Phase {
    Lex,
    Parse,
    Analysis,
}

struct Limits {
    work: [usize; 3], // Indexed by Phase.
    live_bytes: usize,
}

#[derive(Clone, Copy)]
struct Usage {
    work: [usize; 3],
    live_bytes: usize,
}

struct Ledger {
    limits: Limits,
    usage: Cell<Usage>,
}

#[derive(Debug, PartialEq, Eq)]
enum Refusal {
    Work,
    LiveBytes,
}

struct Reservation<'ledger> {
    ledger: &'ledger Ledger,
    live_bytes: usize,
}

impl Ledger {
    fn new(limits: Limits) -> Self {
        Self {
            limits,
            usage: Cell::new(Usage {
                work: [0; 3],
                live_bytes: 0,
            }),
        }
    }

    fn admit(
        &self,
        phase: Phase,
        work: usize,
        live_bytes: usize,
    ) -> Result<Reservation<'_>, Refusal> {
        // Edit a snapshot so neither charge takes effect if either check fails.
        let mut usage = self.usage.get();
        let phase = phase as usize;
        usage.work[phase] = usage.work[phase]
            .checked_add(work)
            .filter(|&total| total <= self.limits.work[phase])
            .ok_or(Refusal::Work)?;
        usage.live_bytes = usage
            .live_bytes
            .checked_add(live_bytes)
            .filter(|&total| total <= self.limits.live_bytes)
            .ok_or(Refusal::LiveBytes)?;

        self.usage.set(usage);
        Ok(Reservation {
            ledger: self,
            live_bytes,
        })
    }
}

impl Reservation<'_> {
    fn extend(&mut self, phase: Phase, work: usize, live_bytes: usize) -> Result<(), Refusal> {
        let mut added = self.ledger.admit(phase, work, live_bytes)?;
        // The checked ledger total includes both reservations, so this sum
        // fits. Transfer the charge without releasing it from the ledger.
        let combined = self.live_bytes + added.live_bytes;
        added.live_bytes = 0;
        self.live_bytes = combined;
        Ok(())
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        let mut usage = self.ledger.usage.get();
        // This guard owns one live charge. Work stays spent for the invocation.
        usage.live_bytes -= self.live_bytes;
        self.ledger.usage.set(usage);
    }
}

/// A storage proof that charges the boxed payload, not allocator overhead.
/// `Refusal` reports policy limits, not allocator failure.
/// `Box::new` is not recoverably fallible; OOM may terminate the process.
#[cfg(test)]
struct AdmittedSpanPair<'ledger> {
    // Fields drop in declaration order: deallocate the box before releasing its charge.
    spans: Box<[Span; 2]>,
    _reservation: Reservation<'ledger>,
}

#[cfg(test)]
impl<'ledger> AdmittedSpanPair<'ledger> {
    fn new(
        ledger: &'ledger Ledger,
        work: usize,
        first: Span,
        second: Span,
    ) -> Result<Self, Refusal> {
        // The spans are caller inputs; the backing array is built only after admission.
        let reservation = ledger.admit(Phase::Lex, work, std::mem::size_of::<[Span; 2]>())?;
        Ok(Self {
            spans: Box::new([first, second]),
            _reservation: reservation,
        })
    }

    fn as_spans(&self) -> &[Span; 2] {
        &self.spans
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::{AdmittedSpanPair, Ledger, Limits, Phase, Refusal};
    use crate::v2::lexer::here_terminator_end;
    use crate::v2::span::Span;

    fn ledger(work: [usize; 3], live_bytes: usize) -> Ledger {
        Ledger::new(Limits { work, live_bytes })
    }

    fn metered_here_terminator_end(
        ledger: &Ledger,
        suffix: &str,
        quote: u8,
    ) -> Result<Option<usize>, Refusal> {
        let charge = || ledger.admit(Phase::Lex, 1, 0).map(drop);
        here_terminator_end(suffix, quote, charge)
    }

    // Hold exactly the expected remainder, then check neither limit has spare room.
    fn assert_remaining_capacity(ledger: &Ledger, work: usize, live_bytes: usize) {
        let _remaining = ledger
            .admit(Phase::Lex, work, live_bytes)
            .expect("the unchanged remainder should fit");
        assert_eq!(
            (
                ledger.admit(Phase::Lex, 1, 0).map(drop),
                ledger.admit(Phase::Lex, 0, 1).map(drop),
            ),
            (Err(Refusal::Work), Err(Refusal::LiveBytes)),
        );
    }

    #[test]
    fn reservation_extend_merges_live_charge_without_refunding_work() {
        let ledger = ledger([7, 0, 0], 10);
        let _other = ledger.admit(Phase::Parse, 0, 4).unwrap();
        let mut candidate = ledger.admit(Phase::Lex, 1, 2).unwrap();
        candidate.extend(Phase::Lex, 2, 1).unwrap();

        let result = candidate.extend(Phase::Lex, 3, 3);

        assert_eq!(result, Ok(()));
        assert_eq!(
            ledger.admit(Phase::Lex, 0, 1).map(drop),
            Err(Refusal::LiveBytes),
        );
        drop(candidate);
        assert_remaining_capacity(&ledger, 1, 6);
    }

    #[test]
    fn reservation_extend_refuses_work_first_without_changing_either_charge() {
        let ledger = ledger([2, 0, 0], 4);
        let mut candidate = ledger.admit(Phase::Lex, 1, 2).unwrap();

        let result = candidate.extend(Phase::Lex, 2, 3);

        assert_eq!(result, Err(Refusal::Work));
        assert_remaining_capacity(&ledger, 1, 2);
    }

    #[test]
    fn reservation_extend_refuses_live_without_spending_work() {
        let ledger = ledger([2, 0, 0], 4);
        let mut candidate = ledger.admit(Phase::Lex, 1, 2).unwrap();

        let result = candidate.extend(Phase::Lex, 1, 3);

        assert_eq!(result, Err(Refusal::LiveBytes));
        assert_remaining_capacity(&ledger, 1, 2);
    }

    #[test]
    fn ledger_admits_zero_work_and_live_at_zero_limits() {
        let ledger = ledger([0, 0, 0], 0);

        let result = ledger.admit(Phase::Lex, 0, 0);

        assert_eq!(result.map(drop), Ok(()));
    }

    #[test]
    fn ledger_admits_exact_cumulative_work_independent_of_other_phases() {
        let ledger = ledger([2, 4, 6], 0);
        let _lex = ledger.admit(Phase::Lex, 2, 0).expect("lex limit fits");
        let _parse = ledger.admit(Phase::Parse, 1, 0).expect("parse work fits");
        let _analysis = ledger.admit(Phase::Analysis, 6, 0).expect("analysis fits");

        let result = ledger.admit(Phase::Parse, 3, 0);

        assert_eq!(result.map(drop), Ok(()));
    }

    #[test]
    fn ledger_refuses_lex_work_one_over_its_limit() {
        let ledger = ledger([2, 4, 6], 0);
        let _prior = ledger.admit(Phase::Lex, 2, 0).expect("lex limit fits");

        let result = ledger.admit(Phase::Lex, 1, 0);

        assert_eq!(result.map(drop), Err(Refusal::Work));
    }

    #[test]
    fn ledger_refuses_parse_work_one_over_its_limit() {
        let ledger = ledger([2, 4, 6], 0);
        let _prior = ledger.admit(Phase::Parse, 4, 0).expect("parse limit fits");

        let result = ledger.admit(Phase::Parse, 1, 0);

        assert_eq!(result.map(drop), Err(Refusal::Work));
    }

    #[test]
    fn ledger_refuses_analysis_work_one_over_its_limit() {
        let ledger = ledger([2, 4, 6], 0);
        let _prior = ledger.admit(Phase::Analysis, 6, 0).expect("analysis fits");

        let result = ledger.admit(Phase::Analysis, 1, 0);

        assert_eq!(result.map(drop), Err(Refusal::Work));
    }

    #[test]
    fn ledger_does_not_share_capacity_between_invocations() {
        let first = ledger([2, 0, 0], 3);
        let second = ledger([2, 0, 0], 3);
        let _held = first
            .admit(Phase::Lex, 2, 3)
            .expect("first invocation fits");

        let result = second.admit(Phase::Lex, 2, 3);

        assert_eq!(result.map(drop), Ok(()));
    }

    #[test]
    fn ledger_counts_overlapping_parent_and_moved_child() {
        let ledger = ledger([0, 0, 0], 5);
        let _parent = ledger.admit(Phase::Lex, 0, 2).expect("parent fits");
        let child = ledger.admit(Phase::Parse, 0, 3).expect("overlap fits");

        let _moved_child = child;

        assert_eq!(
            ledger.admit(Phase::Analysis, 0, 1).map(drop),
            Err(Refusal::LiveBytes),
        );
    }

    #[test]
    fn ledger_releases_only_the_dropped_guard_live_charge() {
        let ledger = ledger([0, 3, 0], 5);
        let _parent = ledger.admit(Phase::Lex, 0, 2).expect("parent fits");
        let child = ledger.admit(Phase::Parse, 3, 3).expect("child fits");

        drop(child);

        let _reused = ledger
            .admit(Phase::Analysis, 0, 3)
            .expect("the child's live bytes should be reusable");
        assert_eq!(
            (
                ledger.admit(Phase::Parse, 1, 0).map(drop),
                ledger.admit(Phase::Analysis, 0, 1).map(drop),
            ),
            (Err(Refusal::Work), Err(Refusal::LiveBytes)),
        );
    }

    #[test]
    fn ledger_admits_replacement_at_the_combined_live_limit() {
        let ledger = ledger([0, 0, 0], 8);
        let _old = ledger
            .admit(Phase::Analysis, 0, 3)
            .expect("old charge fits");

        let result = ledger.admit(Phase::Analysis, 0, 5);

        assert_eq!(result.map(drop), Ok(()));
    }

    #[test]
    fn ledger_refuses_replacement_one_over_the_combined_live_limit() {
        let ledger = ledger([0, 0, 0], 7);
        let _old = ledger
            .admit(Phase::Analysis, 0, 3)
            .expect("old charge fits");

        let result = ledger.admit(Phase::Analysis, 0, 5);

        assert_eq!(result.map(drop), Err(Refusal::LiveBytes));
    }

    #[test]
    fn ledger_refuses_work_first_when_both_limits_are_exceeded() {
        let ledger = ledger([4, 0, 0], 5);
        let _held = ledger.admit(Phase::Lex, 2, 3).expect("initial charge fits");

        let result = ledger.admit(Phase::Lex, 3, 3);

        assert_eq!(result.map(drop), Err(Refusal::Work));
    }

    #[test]
    fn ledger_preserves_both_capacities_after_work_refusal() {
        let ledger = ledger([5, 0, 0], 8);
        let held = ledger.admit(Phase::Lex, 2, 3).expect("initial charge fits");

        let result = ledger.admit(Phase::Lex, 4, 4);

        assert_eq!(result.map(drop), Err(Refusal::Work));
        assert_remaining_capacity(&ledger, 3, 5);
        drop(held);
        assert_eq!(ledger.admit(Phase::Lex, 0, 8).map(drop), Ok(()));
    }

    #[test]
    fn ledger_preserves_both_capacities_after_live_refusal() {
        let ledger = ledger([5, 0, 0], 8);
        let held = ledger.admit(Phase::Lex, 2, 3).expect("initial charge fits");

        let result = ledger.admit(Phase::Lex, 2, 6);

        assert_eq!(result.map(drop), Err(Refusal::LiveBytes));
        assert_remaining_capacity(&ledger, 3, 5);
        drop(held);
        assert_eq!(ledger.admit(Phase::Lex, 0, 8).map(drop), Ok(()));
    }

    #[test]
    fn ledger_admits_exact_usize_max_cumulative_work_and_live() {
        let ledger = ledger([usize::MAX, 0, 0], usize::MAX);
        let _held = ledger.admit(Phase::Lex, 2, 3).expect("initial charge fits");

        let result = ledger.admit(Phase::Lex, usize::MAX - 2, usize::MAX - 3);

        assert_eq!(result.map(drop), Ok(()));
    }

    #[test]
    fn ledger_refuses_work_overflow_without_charging_either_capacity() {
        let ledger = ledger([usize::MAX, 0, 0], usize::MAX);
        let _held = ledger.admit(Phase::Lex, 2, 3).expect("initial charge fits");

        let result = ledger.admit(Phase::Lex, usize::MAX - 1, 4);

        assert_eq!(result.map(drop), Err(Refusal::Work));
        assert_remaining_capacity(&ledger, usize::MAX - 2, usize::MAX - 3);
    }

    #[test]
    fn ledger_refuses_live_overflow_without_charging_either_capacity() {
        let ledger = ledger([usize::MAX, 0, 0], usize::MAX);
        let _held = ledger.admit(Phase::Lex, 2, 3).expect("initial charge fits");

        let result = ledger.admit(Phase::Lex, 4, usize::MAX - 2);

        assert_eq!(result.map(drop), Err(Refusal::LiveBytes));
        assert_remaining_capacity(&ledger, usize::MAX - 2, usize::MAX - 3);
    }

    #[test]
    fn ledger_refuses_work_first_when_both_sums_overflow_without_charging() {
        let ledger = ledger([usize::MAX, 0, 0], usize::MAX);
        let _held = ledger.admit(Phase::Lex, 2, 3).expect("initial charge fits");

        let result = ledger.admit(Phase::Lex, usize::MAX - 1, usize::MAX - 2);

        assert_eq!(result.map(drop), Err(Refusal::Work));
        assert_remaining_capacity(&ledger, usize::MAX - 2, usize::MAX - 3);
    }

    #[test]
    fn here_terminator_contract_empty_suffix_returns_none_with_entry_work() {
        let ledger = ledger([1, 0, 0], 0);
        let suffix = "";

        let result = metered_here_terminator_end(&ledger, suffix, b'\'');

        assert_eq!(result, Ok(None));
    }

    #[test]
    fn here_terminator_contract_empty_suffix_refuses_without_entry_work() {
        let ledger = ledger([0, 0, 0], 0);
        let suffix = "";

        let result = metered_here_terminator_end(&ledger, suffix, b'\'');

        assert_eq!(result, Err(Refusal::Work));
    }

    #[test]
    fn here_terminator_contract_lone_cr_counts_utf8_bytes_and_leaves_tail() {
        let ledger = ledger([6, 0, 0], 0);
        let suffix = "é\r'@tail";

        let result = metered_here_terminator_end(&ledger, suffix, b'\'');

        assert_eq!(result, Ok(Some(5)));
    }

    #[test]
    fn here_terminator_contract_refuses_when_at_byte_exceeds_work_limit() {
        let ledger = ledger([5, 0, 0], 0);
        let suffix = "é\r'@tail";

        let result = metered_here_terminator_end(&ledger, suffix, b'\'');

        assert_eq!(result, Err(Refusal::Work));
    }

    #[test]
    fn here_terminator_contract_double_quote_after_crlf_returns_exclusive_end() {
        let ledger = ledger([5, 0, 0], 0);
        let suffix = "\r\n\"@tail";

        let result = metered_here_terminator_end(&ledger, suffix, b'"');

        assert_eq!(result, Ok(Some(4)));
    }

    #[test]
    fn here_terminator_contract_restarts_after_partial_match_at_line_break() {
        let ledger = ledger([6, 0, 0], 0);
        let suffix = "\n'\n'@";

        let result = metered_here_terminator_end(&ledger, suffix, b'\'');

        assert_eq!(result, Ok(Some(5)));
    }

    #[test]
    fn here_terminator_contract_ignores_indented_and_wrong_quote_markers() {
        let ledger = ledger([8, 0, 0], 0);
        let suffix = "\n '@\n\"@";

        let result = metered_here_terminator_end(&ledger, suffix, b'\'');

        assert_eq!(result, Ok(None));
    }

    const SPAN_PAIR_SOURCE: &str = "\u{feff}é";
    const SPAN_PAIR_BYTES: usize = size_of::<[Span; 2]>();
    const SPAN_PAIR_WORK: usize = 2;

    fn admit_span_pair(ledger: &Ledger) -> Result<AdmittedSpanPair<'_>, Refusal> {
        let bom_end = '\u{feff}'.len_utf8();
        AdmittedSpanPair::new(
            ledger,
            SPAN_PAIR_WORK,
            Span::new(0, bom_end),
            Span::new(bom_end, SPAN_PAIR_SOURCE.len()),
        )
    }

    struct SpanPairConsumer<'ledger>(AdmittedSpanPair<'ledger>);

    impl SpanPairConsumer<'_> {
        fn spans(&self) -> &[Span; 2] {
            self.0.as_spans()
        }
    }

    #[test]
    fn span_pair_storage_admits_exact_fit_and_preserves_original_source_spans() {
        let ledger = ledger([SPAN_PAIR_WORK, 0, 0], SPAN_PAIR_BYTES);

        let result = admit_span_pair(&ledger).map(SpanPairConsumer);

        let consumer = result.expect("the span pair should fit exactly");
        let spans = consumer.spans();
        assert_eq!(spans, &[Span::new(0, 3), Span::new(3, 5)]);
        assert_eq!(
            [
                spans[0].slice(SPAN_PAIR_SOURCE),
                spans[1].slice(SPAN_PAIR_SOURCE),
            ],
            ["\u{feff}", "é"],
        );
    }

    #[test]
    fn span_pair_storage_refuses_one_byte_short_without_charging() {
        let ledger = ledger([SPAN_PAIR_WORK, 0, 0], SPAN_PAIR_BYTES - 1);

        let result = admit_span_pair(&ledger);

        assert_eq!(result.map(drop), Err(Refusal::LiveBytes));
        assert_remaining_capacity(&ledger, SPAN_PAIR_WORK, SPAN_PAIR_BYTES - 1);
    }

    #[test]
    fn span_pair_storage_refuses_work_first_without_charging_when_both_limits_fail() {
        let ledger = ledger([SPAN_PAIR_WORK - 1, 0, 0], SPAN_PAIR_BYTES - 1);

        let result = admit_span_pair(&ledger);

        assert_eq!(result.map(drop), Err(Refusal::Work));
        assert_remaining_capacity(&ledger, SPAN_PAIR_WORK - 1, SPAN_PAIR_BYTES - 1);
    }

    #[test]
    fn span_pair_storage_refuses_second_pair_while_consumer_owns_first() {
        let ledger = ledger([2 * SPAN_PAIR_WORK, 0, 0], SPAN_PAIR_BYTES);
        let _consumer = SpanPairConsumer(admit_span_pair(&ledger).expect("first span pair fits"));

        let result = admit_span_pair(&ledger);

        assert_eq!(result.map(drop), Err(Refusal::LiveBytes));
    }

    #[test]
    fn span_pair_storage_admits_second_pair_after_consumer_drops_first() {
        let ledger = ledger([2 * SPAN_PAIR_WORK, 0, 0], SPAN_PAIR_BYTES);
        let consumer = SpanPairConsumer(admit_span_pair(&ledger).expect("first span pair fits"));
        drop(consumer);

        let result = admit_span_pair(&ledger);

        assert_eq!(result.map(drop), Ok(()));
    }

    #[test]
    fn span_pair_storage_does_not_refund_work_when_consumer_drops_pair() {
        let ledger = ledger([SPAN_PAIR_WORK, 0, 0], SPAN_PAIR_BYTES);
        let consumer = SpanPairConsumer(admit_span_pair(&ledger).expect("the span pair fits"));
        drop(consumer);

        let result = admit_span_pair(&ledger);

        assert_eq!(result.map(drop), Err(Refusal::Work));
    }
}
