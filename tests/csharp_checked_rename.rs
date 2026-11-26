#![cfg(feature = "csharp")]

use poshtree::v2::csharp::resolve::DeclKind;
use poshtree::v2::csharp::{
    CSharpRenameError as Error, CSharpRenameMode as Mode, CSharpRenamePlan, CheckedCSharpRegion,
};
use poshtree::v2::{EditError, Span, TextEdit};

fn whole(source: &str) -> CheckedCSharpRegion<'_> {
    CheckedCSharpRegion::parse(source, Span::new(0, source.len())).unwrap()
}

fn edits<const N: usize>(ranges: [(usize, usize); N], to: &str) -> [TextEdit; N] {
    ranges.map(|(start, end)| TextEdit::replace(Span::new(start, end), to))
}

const FIELD_DOCUMENT: &str = "π|class C{int @x;void F(){x=this.@x;}}|";
const FIELD_OUTPUT: &str = "π|class C{int count;void F(){count=this.count;}}|";
const OVERLOADS: &str = "class C{void F(int a){}void F(bool b){}void G(int v){F(v);}}";

fn field_plan(source: &str) -> CSharpRenamePlan<'_> {
    let region = CheckedCSharpRegion::parse(source, Span::new(3, 39)).unwrap();
    let member = region.member_at(Span::new(15, 17)).unwrap();
    let replacement = String::from("count");
    member.plan_rename(Mode::Single, &replacement).unwrap()
}

#[test]
fn member_at_exact_field_token_returns_kind_and_absolute_span() {
    let region = CheckedCSharpRegion::parse(FIELD_DOCUMENT, Span::new(3, 39)).unwrap();

    let member = region.member_at(Span::new(15, 17)).unwrap();

    assert_eq!(
        (member.kind(), member.name_span()),
        (DeclKind::Field, Span::new(15, 17))
    );
}

#[test]
fn member_at_exact_method_token_returns_method_kind() {
    let region = whole(OVERLOADS);

    let member = region.member_at(Span::new(28, 29)).unwrap();

    assert_eq!(
        (member.kind(), member.name_span()),
        (DeclKind::Method, Span::new(28, 29))
    );
}

#[test]
fn field_plan_owns_replacements_for_the_exact_raw_tokens() {
    let source = FIELD_DOCUMENT.to_owned();

    let plan = field_plan(&source);

    assert_eq!(
        plan.edits(),
        &edits([(15, 17), (27, 28), (34, 36)], "count")
    );
}

#[test]
fn apply_to_an_identical_document_copy_preserves_all_unedited_bytes() {
    let source = FIELD_DOCUMENT.to_owned();
    let copy = source.clone();
    let plan = field_plan(&source);

    let result = plan.apply_to(&copy);

    assert_eq!(result, Ok(FIELD_OUTPUT.to_owned()));
}

#[test]
fn apply_to_same_length_changed_document_refuses_even_outside_the_region() {
    let plan = field_plan(FIELD_DOCUMENT);
    for edited in [
        "λ|class C{int @x;void F(){x=this.@x;}}|",
        "π|class C{int @x;void F(){x=this.@x;}}!",
        "π|class D{int @x;void F(){x=this.@x;}}|",
    ] {
        let result = plan.apply_to(edited);

        assert_eq!(result, Err(Error::StaleSource), "{edited}");
    }
}

#[test]
fn apply_to_previously_renamed_document_refuses_the_old_plan() {
    let plan = field_plan(FIELD_DOCUMENT);

    let result = plan.apply_to(FIELD_OUTPUT);

    assert_eq!(result, Err(Error::StaleSource));
}

#[test]
fn single_method_plan_leaves_a_different_count_overload_untouched() {
    let source = "class C{void F(){}void F(int a){}void G(int v){F();this.F(v);}}";
    let region = whole(source);
    let member = region.member_at(Span::new(13, 14)).unwrap();

    let plan = member.plan_rename(Mode::Single, "Renamed").unwrap();

    assert_eq!(plan.edits(), &edits([(13, 14), (47, 48)], "Renamed"));
    assert_eq!(
        plan.apply_to(source),
        Ok(
            "class C{void Renamed(){}void F(int a){}void G(int v){Renamed();this.F(v);}}"
                .to_owned()
        )
    );
}

#[test]
fn group_plan_renames_same_count_methods_and_their_witnessed_call() {
    let region = whole(OVERLOADS);
    let member = region.member_at(Span::new(28, 29)).unwrap();

    let plan = member.plan_rename(Mode::OverloadGroup, "Renamed").unwrap();

    assert_eq!(
        plan.edits(),
        &edits([(13, 14), (28, 29), (53, 54)], "Renamed")
    );
    assert_eq!(
        plan.apply_to(OVERLOADS),
        Ok(
            "class C{void Renamed(int a){}void Renamed(bool b){}void G(int v){Renamed(v);}}"
                .to_owned()
        )
    );
}

#[test]
fn single_method_plan_refuses_a_same_count_call_with_ordered_candidates() {
    let region = whole(OVERLOADS);
    let member = region.member_at(Span::new(13, 14)).unwrap();

    let error = member.plan_rename(Mode::Single, "Renamed").unwrap_err();

    assert_eq!(
        error,
        Error::Ambiguous {
            at: Span::new(53, 54),
            candidates: vec![Span::new(13, 14), Span::new(28, 29)],
        }
    );
}

#[test]
fn group_plan_without_an_exact_type_witness_refuses_the_call() {
    let source = "class C{void F(int a){}void F(bool b){}void G(global::C p){F(p);}}";
    let region = whole(source);
    let member = region.member_at(Span::new(13, 14)).unwrap();

    let error = member
        .plan_rename(Mode::OverloadGroup, "Renamed")
        .unwrap_err();

    assert_eq!(
        error,
        Error::Unresolved {
            at: Span::new(59, 60)
        }
    );
}

#[test]
fn member_at_valid_nonmember_ranges_refuses_without_treating_them_as_invalid_ranges() {
    let region = CheckedCSharpRegion::parse(FIELD_DOCUMENT, Span::new(3, 39)).unwrap();
    for name in [
        Span::new(15, 15), // cursor
        Span::new(16, 17), // part of the escaped name
        Span::new(34, 36), // use
        Span::new(9, 10),  // type declaration
    ] {
        let error = region.member_at(name).unwrap_err();

        assert_eq!(error, Error::NotMemberDeclaration(name));
    }
}

#[test]
fn member_at_a_constructor_name_refuses_the_other_declaration_kind() {
    let region = whole("class C{int x;C(){x=x;}}");

    let error = region.member_at(Span::new(14, 15)).unwrap_err();

    assert_eq!(error, Error::NotMemberDeclaration(Span::new(14, 15)));
}

#[test]
fn member_at_invalid_byte_ranges_returns_invalid_selection() {
    let source = "π|class C{int café;}|";
    let region = CheckedCSharpRegion::parse(source, Span::new(3, 22)).unwrap();
    for name in [
        Span::new(0, 2),
        Span::new(19, 20),
        Span::new(22, 23),
        Span { start: 17, end: 16 },
    ] {
        let error = region.member_at(name).unwrap_err();

        assert_eq!(error, Error::InvalidSelection(name));
    }
}

#[test]
fn group_plan_anchored_at_a_field_returns_group_requires_method() {
    let region = whole("class C{int @x;}");
    let member = region.member_at(Span::new(12, 14)).unwrap();

    let error = member
        .plan_rename(Mode::OverloadGroup, "count")
        .unwrap_err();

    assert_eq!(error, Error::GroupRequiresMethod(Span::new(12, 14)));
}

#[test]
fn plan_rename_to_the_same_comparison_name_returns_unchanged_name() {
    let region = whole("class C{int @x;}");
    let member = region.member_at(Span::new(12, 14)).unwrap();

    let error = member.plan_rename(Mode::Single, "x").unwrap_err();

    assert_eq!(error, Error::UnchangedName);
}

#[test]
fn plan_rename_to_an_unsupported_identifier_returns_invalid_replacement() {
    let region = whole("class C{int x;}");
    let member = region.member_at(Span::new(12, 13)).unwrap();
    for replacement in ["class", "dynamic", "@count", "café", ""] {
        let error = member.plan_rename(Mode::Single, replacement).unwrap_err();

        assert_eq!(error, Error::InvalidReplacement, "{replacement:?}");
    }
}

#[test]
fn plan_rename_to_a_used_identifier_refuses_even_an_unrelated_local() {
    let region = whole("class C{int x;void M(){x=x;int count;}}");
    let member = region.member_at(Span::new(12, 13)).unwrap();

    let error = member.plan_rename(Mode::Single, "count").unwrap_err();

    assert_eq!(error, Error::NameConflict(Span::new(31, 36)));
}

#[test]
fn plan_rename_with_duplicate_methods_reports_the_declaration_spans() {
    let region = whole("class C{void F(int a){}void @F(int b){}}");
    let member = region.member_at(Span::new(13, 14)).unwrap();

    let error = member
        .plan_rename(Mode::OverloadGroup, "Renamed")
        .unwrap_err();

    assert_eq!(
        error,
        Error::DuplicateDeclaration {
            declarations: vec![Span::new(13, 14), Span::new(28, 30)],
        }
    );
}

#[test]
fn parse_rejects_invalid_region_ranges_before_observation() {
    for region in [
        Span::new(1, 39),
        Span::new(3, 41),
        Span { start: 5, end: 3 },
    ] {
        let error = CheckedCSharpRegion::parse(FIELD_DOCUMENT, region).unwrap_err();

        assert_eq!(error, Error::InvalidRegion(region));
    }
}

#[test]
fn parse_refuses_discarded_trivia_that_is_not_proven_inert() {
    let source = "class C{int x;} /* unfinished";

    let error = CheckedCSharpRegion::parse(source, Span::new(0, 29)).unwrap_err();

    assert_eq!(
        error,
        Error::UnsupportedRegion {
            first: Some(Span::new(16, 29)),
            recovery: vec![],
        }
    );
}

#[test]
fn parse_refuses_an_unproven_explicit_parameter_type_at_its_unbound_segment() {
    let source = "class C{void M(global::Missing x){}}";

    let error = CheckedCSharpRegion::parse(source, Span::new(0, 36)).unwrap_err();

    assert_eq!(
        error,
        Error::Unresolved {
            at: Span::new(23, 30)
        }
    );
}

#[test]
fn parse_refuses_an_ambiguous_explicit_parameter_type_with_ordered_candidates() {
    let source = "class A{}class A{}class C{void M(global::A x){}}";

    let error = CheckedCSharpRegion::parse(source, Span::new(0, 48)).unwrap_err();

    assert_eq!(
        error,
        Error::Ambiguous {
            at: Span::new(41, 42),
            candidates: vec![Span::new(6, 7), Span::new(15, 16)],
        }
    );
}

#[test]
fn parse_refuses_a_generic_method_elsewhere_at_the_token_not_the_comment() {
    let source = "class C{int x;}class G{void F /*<*/ <T>(){}}";

    let error = CheckedCSharpRegion::parse(source, Span::new(0, 44)).unwrap_err();

    assert_eq!(
        error,
        Error::UnsupportedRegion {
            first: Some(Span::new(36, 37)),
            recovery: vec![
                Span::new(36, 37),
                Span::new(37, 38),
                Span::new(38, 42),
                Span::new(43, 43),
                Span::new(43, 44),
            ],
        }
    );
}

#[test]
fn plan_rename_allows_an_unrelated_empty_generic_type() {
    let region = whole("class C{int x;}class G<T>{}");
    let member = region.member_at(Span::new(12, 13)).unwrap();

    let plan = member.plan_rename(Mode::Single, "count").unwrap();

    assert_eq!(plan.edits(), &edits([(12, 13)], "count"));
}

#[test]
fn plan_rename_refuses_a_generic_owner_or_ancestor() {
    for (source, name) in [
        ("class G<T>{int x;}", Span::new(15, 16)),
        ("class G<T>{class C{int x;}}", Span::new(23, 24)),
    ] {
        let region = whole(source);
        let member = region.member_at(name).unwrap();

        let error = member.plan_rename(Mode::Single, "count").unwrap_err();

        assert_eq!(
            error,
            Error::UnsupportedRegion {
                first: Some(Span::new(6, 7)),
                recovery: vec![],
            }
        );
    }
}

#[test]
fn parse_reports_depth_at_the_skipped_block_even_without_legacy_recovery() {
    let source = format!(
        "{}class C{{void F(){{}}}}{}",
        "class N{".repeat(197),
        "}".repeat(197)
    );

    let error = CheckedCSharpRegion::parse(&source, Span::new(0, source.len())).unwrap_err();

    assert_eq!(
        error,
        Error::DepthLimit {
            at: Span::new(1592, 1593),
            recovery: vec![]
        }
    );
}

#[test]
fn parse_reports_depth_at_the_recovered_member_with_its_diagnostic() {
    let source = format!(
        "{}class C{{void F(){{}}}}{}",
        "class N{".repeat(198),
        "}".repeat(198)
    );

    let error = CheckedCSharpRegion::parse(&source, Span::new(0, source.len())).unwrap_err();

    assert_eq!(
        error,
        Error::DepthLimit {
            at: Span::new(1592, 1596),
            recovery: vec![Span::new(1592, 1602)],
        }
    );
}

#[test]
fn plan_rename_with_an_incomplete_base_refuses_at_the_base_use() {
    let region = whole("class C{int x;}class A:Missing{}");
    let member = region.member_at(Span::new(12, 13)).unwrap();

    let error = member.plan_rename(Mode::Single, "count").unwrap_err();

    assert_eq!(error, Error::IncompleteBase(Span::new(23, 30)));
}

#[test]
fn plan_rename_with_a_base_cycle_refuses_at_the_closing_edge() {
    let region = whole("class C{int x;}class A:B{}class B:A{}");
    let member = region.member_at(Span::new(12, 13)).unwrap();

    let error = member.plan_rename(Mode::Single, "count").unwrap_err();

    assert_eq!(error, Error::BaseCycle(Span::new(34, 35)));
}

#[test]
fn edit_error_preserves_the_original_error_as_its_source() {
    let edit_error = EditError::NotCharBoundary(1);
    let error = Error::Edit(edit_error.clone());

    let source = std::error::Error::source(&error).unwrap();

    assert_eq!(source.downcast_ref::<EditError>(), Some(&edit_error));
}
