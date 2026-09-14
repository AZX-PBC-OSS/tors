"""Contract gate for ``tors.scrub_log_text``: named-rule, GIL-free log and
exception-text scrubbing, pinned to the TaskQ exception-text chain it ports
(``src/taskq/obs/_redact_exc.py``; the four regexes are quoted in
``tests/reference.py`` and re-synced against the live module by
``tests/test_scrub_log_text_parity.py``).

What this gate pins, oracle-derived literal by literal:

- pg_detail_lines: real-newline DETAIL lines are deleted line-wise (the
  newline itself stays: a blank line is left behind, matching the source
  chain; CRLF's ``\\r`` is consumed with the line), and repr()-flattened
  ``\\nDETAIL:`` runs are consumed up to (not including) the escaped
  separator or the closing quote, preserving the repr's trailing ``')"``.
  The two non-matches the source chain treats as unreachable-but-real are
  pinned as SPECIFIED behavior, not quietly fixed: an escaped DETAIL with
  no closing quote and no trailing escaped newline is left alone, and one
  terminated by a real newline with no quote before it is left alone.
- uri_userinfo: ``scheme://user:password@host`` -> ``scheme://user:***@host``
  (empty username handled, password ends at the first ``@``, ``\\b`` word
  boundary before the scheme — including the Unicode cases where CPython's
  ``\\w`` and a naive Rust alphanumeric check disagree).
- uri_query_creds: ``[?&](password|passphrase|passwd|pwd)=value`` ->
  ``[?&]name=***`` (exact lowercase names, value runs to whitespace, ``&``
  or ``@``).
- canonical order: the rules apply pg_detail_lines -> uri_userinfo ->
  uri_query_creds, each rule a whole pass before the next; the order is a
  contract (a DETAIL deletion can eat the ``@`` a userinfo mask would have
  needed — pinned), duplicates dedupe, caller order is irrelevant.
- identity return: ``scrub_log_text(s, rules) is s`` exactly when no rule
  fires — including the ``***`` fixed points, where a rule fires but the
  spliced output equals the input: those return a fresh object.
- convergence, not strict idempotence: scrubbing the scrubbed output is a
  fixed point BY the second pass. The one non-idempotence class is
  cross-rule: a password-param replacement deletes a ``/`` that was
  capping the userinfo user run, so the second pass finds one more
  redaction (``x://u?pwd=a/b&:pw@h`` — pinned literally, with oracle
  parity at both passes); the source chain behaves identically, and the
  third pass is always the fixed point (pinned over hypothesis).
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import scrub_log_text
from reference import reference_scrub_log_text

PG = ["pg_detail_lines"]
URI_USER = ["uri_userinfo"]
URI_QUERY = ["uri_query_creds"]


class TestPgDetailLines:
    def test_detail_line_is_deleted_and_a_blank_line_is_left(self) -> None:
        assert scrub_log_text("duplicate key\nDETAIL:  Key (id)=(9) exists.") == "duplicate key\n"

    def test_crlf_detail_line_consumes_the_carriage_return(self) -> None:
        assert (
            scrub_log_text("duplicate key\r\nDETAIL:  Key (id)=(9) exists.\r\nHINT: x")
            == "duplicate key\r\n\nHINT: x"
        )

    def test_leading_spaces_and_tabs_are_part_of_the_line(self) -> None:
        assert scrub_log_text("\t DETAIL: v\nnext") == "\nnext"

    def test_every_detail_line_goes_hint_and_context_stay(self) -> None:
        assert (
            scrub_log_text("boom\nDETAIL: one\nHINT: keep\nCONTEXT: keep2\nDETAIL: two\nend")
            == "boom\n\nHINT: keep\nCONTEXT: keep2\n\nend"
        )

    def test_near_miss_names_are_not_detail_lines(self) -> None:
        text = "  DETAILX: v\ndetail: v\nDETAIL v\nnext"
        assert scrub_log_text(text) is text

    def test_detail_line_at_end_of_text_without_a_newline(self) -> None:
        assert scrub_log_text("x\nDETAIL: tail") == "x\n"

    def test_the_whole_line_to_EOL_goes_not_just_the_word(self) -> None:
        assert scrub_log_text("pre\nDETAIL: v and everything to EOL\nHINT: h") == "pre\n\nHINT: h"

    def test_detail_line_at_start_of_text(self) -> None:
        assert scrub_log_text("DETAIL: x\ny") == "\ny"


class TestPgDetailEscaped:
    def test_repr_run_consumes_to_the_closing_quote_and_keeps_it(self) -> None:
        assert (
            scrub_log_text("PostgresError('msg\\nDETAIL:  Key (id)=(9) exists.')")
            == "PostgresError('msg')"
        )

    def test_escaped_crlf_boundary_is_consumed_too(self) -> None:
        assert scrub_log_text("PostgresError('msg\\r\\nDETAIL: secret')") == "PostgresError('msg')"

    def test_consecutive_escaped_runs_each_terminate_the_previous(self) -> None:
        assert scrub_log_text("E('a\\nDETAIL: one\\nDETAIL: two')") == "E('a')"

    def test_interior_quotes_are_skipped_not_stopped_at(self) -> None:
        # The lazy run only stops at a quote the closing alternative can
        # accept (quote + optional `)` + whitespace + end of line); the
        # quotes inside the DETAIL payload do not satisfy it.
        assert scrub_log_text("E('a\\nDETAIL: Key (x)=('val') exists.')") == "E('a')"

    def test_a_real_tab_prefix_is_consumed_but_a_literal_backslash_t_is_not(self) -> None:
        # `[ \t]*` matches real tab bytes; the two-character `\t` sequence a
        # repr writes is a backslash, which is not in the class.
        assert scrub_log_text("E('a\\n\tDETAIL: v')") == "E('a')"
        text = "E('a\\n\\tDETAIL: v')"
        assert scrub_log_text(text) is text

    def test_no_terminator_means_no_scrub_the_pinned_non_match(self) -> None:
        # Unreachable from repr() output, pinned so a future change that
        # silently alters redaction behavior is a test failure, not a
        # behavior change: no closing quote and no trailing escaped newline.
        text = "E('a\\nDETAIL: leaks"
        assert scrub_log_text(text) is text

    def test_real_newline_termination_without_a_quote_is_the_other_non_match(self) -> None:
        # Also unreachable from repr() output (the run cannot cross a real
        # newline, and no quote sits before it): left alone.
        text = "E('a\\nDETAIL: leaks\nnext line"
        assert scrub_log_text(text) is text

    def test_a_quote_not_at_end_of_line_is_not_a_terminator(self) -> None:
        text = "E('a\\nDETAIL: v')  tail"
        assert scrub_log_text(text) is text

    def test_trailing_whitespace_after_the_quote_still_terminates(self) -> None:
        assert scrub_log_text("E('a\\nDETAIL: v')  \nnext") == "E('a')  \nnext"

    def test_quote_paren_then_real_newline_terminates(self) -> None:
        # `$` is MULTILINE: the closing alternative also fires right before
        # a real newline, which is what a repr line embedded in a rendered
        # traceback looks like.
        assert scrub_log_text("E('a\\nDETAIL: v')\nnext") == "E('a')\nnext"

    def test_repr_line_embedded_in_a_rendered_traceback(self) -> None:
        assert (
            scrub_log_text(
                "Traceback (most recent call last):\nJobError('dup\\nDETAIL: K=(v)')\nafter"
            )
            == "Traceback (most recent call last):\nJobError('dup')\nafter"
        )


class TestUriUserinfo:
    def test_basic_mask(self) -> None:
        assert scrub_log_text("postgresql://worker:hunter2@db/prod") == (
            "postgresql://worker:***@db/prod"
        )

    def test_empty_username_is_a_real_shape(self) -> None:
        assert scrub_log_text("postgresql://:SECRET@host/db") == "postgresql://:***@host/db"

    def test_empty_password_is_not_a_mask(self) -> None:
        text = "postgresql://user:@host/db"
        assert scrub_log_text(text) is text

    def test_password_ends_at_the_first_at(self) -> None:
        assert scrub_log_text("a://u:p@ss@h") == "a://u:***@ss@h"

    def test_colons_inside_the_password_stay_inside_the_mask(self) -> None:
        assert scrub_log_text("a://u:p:q@h") == "a://u:***@h"

    def test_scheme_grammar_letters_digits_plus_dot_dash(self) -> None:
        assert scrub_log_text("DB+sql-x.9://u:pw@h") == "DB+sql-x.9://u:***@h"

    def test_uppercase_ip_port_shape_masks_too(self) -> None:
        assert scrub_log_text("http://u:p@192.168.1.1:8080/x") == "http://u:***@192.168.1.1:8080/x"

    def test_a_run_starting_with_a_digit_has_no_letter_start(self) -> None:
        text = "1st://u:pw@h"
        assert scrub_log_text(text) is text

    def test_a_run_starting_with_a_dash_masks_from_the_first_letter(self) -> None:
        # The `\b` sits between `-` and `s`, so the match (and the mask)
        # starts at the letter; the dash is not part of the scheme.
        assert scrub_log_text("-st://u:pw@h") == "-st://u:***@h"

    def test_a_word_char_before_the_scheme_kills_the_boundary(self) -> None:
        for text in ("0https://u:pw@h", "\u00e9https://u:pw@h", "_https://u:pw@h"):
            assert scrub_log_text(text) is text, text

    def test_a_mark_before_the_scheme_is_not_a_word_char_in_python(self) -> None:
        # U+093E (a Devanagari vowel sign) and U+24B6 (circled capital A)
        # are Other_Alphabetic: CPython's `\w` (L* / N* / _) excludes them,
        # so the boundary exists and the mask fires — the class of input
        # where a naive Rust `is_alphanumeric` boundary check would
        # silently under-redact (see src/scrub_impl.rs's demote table).
        for mark in ("\u093e", "\u24b6"):
            assert scrub_log_text(f"{mark}https://u:pw@h") == f"{mark}https://u:***@h"

    def test_username_cannot_contain_slash_at_or_whitespace(self) -> None:
        # `/` ends the username before the `:` the mask needs; U+00A0 and
        # U+001C are `\s` for the chain (Python's `\s` includes the four
        # file-separator controls U+001C..U+001F, which Rust's
        # `is_whitespace` does not — both pinned here through the class).
        for text in ("a://u/v:pw@h", "a://u\u00a0v:pw@h", "a://u\x1cv:pw@h"):
            assert scrub_log_text(text) is text, text

    def test_password_cannot_contain_whitespace(self) -> None:
        for text in ("a://u:p\x1cq@h", "a://u:p\u00a0q@h"):
            assert scrub_log_text(text) is text, text

    def test_a_scheme_inside_the_password_is_masked_with_it(self) -> None:
        assert scrub_log_text("a://u:p://q@h") == "a://u:***@h"

    def test_no_at_after_the_password_means_no_mask(self) -> None:
        text = "a://u:pw host"
        assert scrub_log_text(text) is text


class TestUriQueryCreds:
    def test_all_four_names_are_masked_name_preserved(self) -> None:
        assert scrub_log_text("?password=x&passphrase=y&passwd=z&pwd=w") == (
            "?password=***&passphrase=***&passwd=***&pwd=***"
        )

    def test_names_are_exact_lowercase(self) -> None:
        text = "?Password=x&PASSWORD=y&passwords=z&passw=w"
        assert scrub_log_text(text) is text

    def test_value_runs_to_whitespace_ampersand_or_at(self) -> None:
        assert scrub_log_text("?password=a b?pwd=c@d&passwd=e") == (
            "?password=*** b?pwd=***@d&passwd=***"
        )

    def test_value_may_contain_equals_and_question_marks(self) -> None:
        assert scrub_log_text("?password=a=b?c") == "?password=***"

    def test_empty_value_is_not_a_mask(self) -> None:
        for text in ("?password=&x=1", "?pwd="):
            assert scrub_log_text(text) is text, text

    def test_no_scheme_is_demanded_bare_host_query(self) -> None:
        assert scrub_log_text("host/db?password=x") == "host/db?password=***"

    def test_value_stops_at_python_whitespace_not_rust_whitespace(self) -> None:
        # U+001C is `\s` for the chain (the file-separator divergence
        # pinned above); U+00A0 is whitespace under both.
        assert scrub_log_text("?password=a\x1cb") == "?password=***\x1cb"
        assert scrub_log_text("?password=a\u00a0b") == "?password=***\u00a0b"

    def test_a_delimiter_inside_a_value_is_swallowed_with_it(self) -> None:
        # Greedy value: the second `?pwd=` is inside the first value, so it
        # is masked as part of it, never rescanned (the no-cascade rule).
        assert scrub_log_text("a?password=1?pwd=2") == "a?password=***"


class TestCanonicalOrder:
    def test_a_dsn_with_both_credential_shapes_masks_both(self) -> None:
        assert scrub_log_text("postgresql://worker:S3cr3t@db/prod?password=fallback") == (
            "postgresql://worker:***@db/prod?password=***"
        )

    def test_userinfo_claims_a_password_that_embeds_a_query_param(self) -> None:
        # The userinfo password class stops only at whitespace/`@`, so the
        # embedded `?password=` is inside the mask; the param rule then has
        # nothing left to find.
        assert scrub_log_text("scheme://user:pa?password=zz@host") == "scheme://user:***@host"

    def test_param_only_leaves_the_embedded_param_visible_masked(self) -> None:
        assert (
            scrub_log_text("scheme://user:pa?password=zz@host", URI_QUERY)
            == "scheme://user:pa?password=***@host"
        )

    def test_detail_deletion_can_eat_the_at_a_userinfo_mask_needs(self) -> None:
        # THE order pin: the escaped DETAIL run terminates at the quote
        # after `@h`, deleting through the `@`; with the full chain the
        # userinfo rule then has no `@` to anchor on and the password `p`
        # survives (the chain's own shape, pinned); with uri_userinfo alone
        # the whole password (escaped DETAIL text included) is masked.
        text = "pg://u:p\\nDETAIL:x@h')"
        assert scrub_log_text(text) == "pg://u:p')"
        assert scrub_log_text(text, URI_USER) == "pg://u:***@h')"
        assert scrub_log_text(text, PG) == "pg://u:p')"

    def test_escaped_detail_without_a_terminator_leaves_userinfo_to_mask_it(self) -> None:
        # No quote/escaped-newline terminator: the DETAIL pass cannot fire,
        # so the userinfo password (escaped DETAIL text included) is masked.
        assert scrub_log_text("a://u:p\\nDETAIL:x@h") == "a://u:***@h"


class TestRulesParameter:
    def test_none_is_the_full_chain_the_default(self) -> None:
        text = "boom\nDETAIL: row value\nsee postgres://u:pw@h/db?password=x"
        assert scrub_log_text(text) == scrub_log_text(text, None)
        assert scrub_log_text(text, None) == "boom\n\nsee postgres://u:***@h/db?password=***"

    def test_empty_rules_is_the_identity_object(self) -> None:
        text = "DETAIL: row value\npostgres://u:pw@h?password=x"
        assert scrub_log_text(text, []) is text

    def test_a_tuple_is_as_good_as_a_list(self) -> None:
        assert scrub_log_text("postgres://u:pw@h", ("uri_userinfo",)) == (
            scrub_log_text("postgres://u:pw@h", ["uri_userinfo"])
        )

    def test_duplicates_dedupe(self) -> None:
        text = "postgres://u:pw@h"
        once = scrub_log_text(text, ["uri_userinfo"])
        assert scrub_log_text(text, ["uri_userinfo", "uri_userinfo"]) == once

    def test_caller_order_is_irrelevant_canonical_order_applies(self) -> None:
        text = "pg://u:p\\nDETAIL:x@h')"
        canonical = scrub_log_text(text, ["pg_detail_lines", "uri_userinfo"])
        assert scrub_log_text(text, ["uri_userinfo", "pg_detail_lines"]) == canonical

    def test_unknown_name_names_the_accepted_set(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            scrub_log_text("x", ["pg_detail_lines", "uri_creds"])
        assert str(excinfo.value) == (
            "rules must be one of ('pg_detail_lines', 'uri_userinfo', "
            "'uri_query_creds'), not \"uri_creds\""
        )

    @pytest.mark.parametrize(
        "rules",
        [
            "uri_userinfo",  # a bare str is not a sequence of rule names
            123,
            {"uri_userinfo"},  # sets are not Sequences
            {"uri_userinfo": 1},  # neither are dicts
            ["uri_userinfo", 5],  # members must be str
            (x for x in ("uri_userinfo",)),  # generators are not Sequences
        ],
        ids=["str", "int", "set", "dict", "non-str-member", "generator"],
    )
    def test_non_sequence_rules_raise_type_error(self, rules: object) -> None:
        with pytest.raises(TypeError):
            scrub_log_text("postgres://u:pw@h", rules)  # type: ignore[arg-type]

    def test_rules_accepts_keyword_form(self) -> None:
        assert scrub_log_text("postgres://u:pw@h", rules=["uri_userinfo"]) == ("postgres://u:***@h")


class TestIdentityReturn:
    def test_clean_text_comes_back_as_the_same_object(self) -> None:
        text = "ordinary error text, no credentials, no DETAIL"
        assert scrub_log_text(text) is text

    def test_each_rule_alone_returns_identity_when_it_does_not_fire(self) -> None:
        for rules in (PG, URI_USER, URI_QUERY):
            text = "plain text\nmore of it"
            assert scrub_log_text(text, rules) is text, rules

    def test_a_rule_that_fires_on_an_already_masked_value_returns_a_new_object(self) -> None:
        # The `***` fixed points: the rule FIRES (the password/value class
        # accepts `***`) and the spliced output equals the input, but the
        # contract is identity exactly when NO rule fires — these return a
        # fresh, equal object.
        for text in ("a://u:***@h", "?password=***"):
            out = scrub_log_text(text)
            assert out == text
            assert out is not text, text


class TestIdempotence:
    @pytest.mark.parametrize(
        "text",
        [
            "duplicate key\nDETAIL:  Key (id)=(9) exists.",
            "PostgresError('msg\\nDETAIL:  Key (id)=(9) exists.')",
            "postgresql://worker:S3cr3t@db/prod?password=fallback",
            "pg://u:p\\nDETAIL:x@h')",
            "a://u:***@h",
            "?password=***",
            "plain text",
        ],
        ids=[
            "detail-real",
            "detail-escaped",
            "dsn-both",
            "order-interaction",
            "fixed-point-userinfo",
            "fixed-point-param",
            "clean",
        ],
    )
    def test_scrubbing_the_scrubbed_output_is_a_value_no_op(self, text: str) -> None:
        once = scrub_log_text(text)
        assert scrub_log_text(once) == once


class TestArgumentBoundary:
    def test_lone_surrogate_is_refused(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            scrub_log_text("\ud800abc")

    @pytest.mark.parametrize("not_str", [b"bytes", 5, None], ids=["bytes", "int", "none"])
    def test_non_str_text_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            scrub_log_text(not_str)  # type: ignore[arg-type]


_ANY_TEXT = st.text(
    alphabet=st.characters(max_codepoint=0x10FFFF, exclude_categories=("Cs",)),
    max_size=64,
)


def _legal_credential_payload(s: str) -> str:
    """A char run the password/value classes accept: no Python whitespace
    (``str.isspace`` is exactly ``re.``'s ``\\s`` set — pinned by
    ``TestCredentialPayloadEquivalence`` below and exhaustively by
    ``test_space_table_exhaustive`` in test_scrub_log_text_parity.py),
    no ``&``, no ``@``."""
    return "".join(ch for ch in s if not ch.isspace() and ch not in "&@")


class TestCredentialPayloadEquivalence:
    def test_isspace_is_exactly_re_s_over_the_edge_alphabet(self) -> None:
        """Pin the helper's core claim without re-paying the 1.1M-char
        exhaustive loop (that lives in the parity file): over ASCII
        whitespace, the U+001C..U+001F seam, NBSP/U+2028/ZWSP, and the
        credential delimiters, ``ch.isspace()`` agrees with ``re \\s`` and
        the helper keeps a char exactly when ``re``'s password/value
        classes would accept it."""
        pat = re.compile(r"\s")
        edge = [
            chr(cp)
            for cp in list(range(0x00, 0x30))
            + [0x7F, 0xA0, 0x1C, 0x1D, 0x1E, 0x1F, 0x2028, 0x2029, 0x200B, 0x3000, 0x093E, 0x24B6]
        ]
        edge += ["&", "@", "a", ":", "/", "?", "=", "*", "p"]
        for ch in edge:
            assert ch.isspace() == (pat.match(ch) is not None), repr(ch)
            assert (ch in _legal_credential_payload(ch)) == (not ch.isspace() and ch not in "&@"), (
                repr(ch)
            )

    @given(_ANY_TEXT)
    @settings(max_examples=200)
    def test_helper_matches_re_classes_char_by_char(self, text: str) -> None:
        pat = re.compile(r"\s")
        for ch in text:
            assert (ch in _legal_credential_payload(ch)) == (
                pat.match(ch) is None and ch not in "&@"
            ), repr(ch)


class TestHypothesisInvariants:
    @given(_ANY_TEXT)
    @settings(max_examples=300)
    def test_never_raises_and_converges_by_the_second_pass(self, text: str) -> None:
        # Convergence, not strict idempotence: a param replacement can
        # delete a `/` that was blocking a userinfo match, so the second
        # pass may find one more redaction (the corner pinned literally
        # below); the third pass is the fixed point. The source chain
        # behaves identically — the parity harness pins both passes.
        once = scrub_log_text(text)
        twice = scrub_log_text(once)
        assert scrub_log_text(twice) == twice

    def test_the_param_mask_can_unblock_a_userinfo_match_on_pass_two(self) -> None:
        """The one documented non-idempotence class, pinned literally
        with oracle parity at BOTH passes: pass 1's param value eats the
        `/` that was capping the userinfo user run (`u?pwd=a` stops at
        the `/`), so pass 2's user class spans the `***` and the `&`
        (`u?pwd=***&`) and the userinfo rule fires — the same shape CI's
        fuzz-smoke found (crash-a2d92f3d). Pass 3 re-matches the
        already-`***` password to itself: the fixed point."""
        text = "x://u?pwd=a/b&:pw@h"
        once = scrub_log_text(text)
        twice = scrub_log_text(once)
        assert once == "x://u?pwd=***&:pw@h"
        assert twice == "x://u?pwd=***&:***@h"
        assert scrub_log_text(twice) == twice
        # The source chain, byte-identical at every pass.
        assert once == reference_scrub_log_text(text)
        assert twice == reference_scrub_log_text(once)

    @given(_ANY_TEXT)
    @settings(max_examples=300)
    def test_the_userinfo_mask_always_claims_the_whole_password(self, text: str) -> None:
        # The exact masked shape, not a substring check: the template's
        # outer match always fires (fixed scheme/username/separator, the
        # `@` right after the payload), and no earlier pass can break it —
        # an escaped-DETAIL deletion inside the payload only shrinks what
        # gets masked — so the output is exactly the masked template
        # whatever the payload (a short payload like `a` or `*` would
        # trivially "survive" a substring check inside the scaffold or the
        # `***` itself).
        payload = _legal_credential_payload(text)
        if payload:
            assert scrub_log_text(f"a://u:{payload}@h") == "a://u:***@h"
        else:
            assert scrub_log_text("a://u:@h") == "a://u:@h"  # empty password: no mask

    @given(_ANY_TEXT)
    @settings(max_examples=300)
    def test_the_param_mask_always_claims_the_whole_value(self, text: str) -> None:
        # Same reasoning: the template carries no `@` at all, so the
        # userinfo pass can never fire inside it, and the param value —
        # whatever an escaped-DETAIL deletion leaves of it — is always
        # masked whole.
        payload = _legal_credential_payload(text)
        if payload:
            assert scrub_log_text(f"?password={payload}&x=1") == "?password=***&x=1"
        else:
            assert scrub_log_text("?password=&x=1") == "?password=&x=1"  # empty: no mask

    @given(_ANY_TEXT)
    @settings(max_examples=300)
    def test_a_detail_line_never_survives(self, text: str) -> None:
        payload = text.replace("\n", "")
        if not payload:
            return
        # The whole DETAIL line (prefix spaces through end of line) goes,
        # leaving exactly the surrounding lines and the blank line behind.
        assert scrub_log_text(f"x\nDETAIL:{payload}\ny") == "x\n\ny"
