"""Contract gate for ``tors.scrub_log_text``: named-rule, GIL-free log and
exception-text scrubbing, pinned to the grammar definition it ships with
(``tests/reference.py``; the four regexes are quoted there and differentially
enforced by ``tests/test_scrub_log_text_parity.py``).

What this gate pins, oracle-derived literal by literal:

- pg_detail_lines: real-newline DETAIL lines are deleted line-wise (the
  newline itself stays: a blank line is left behind, matching the source
  chain; CRLF's ``\\r`` is consumed with the line), ExceptionGroup gutters
  absorbed (``traceback.format_exception``'s ``| ``/``+ `` indentation, one
  layer per nesting level), and repr()-flattened ``\\nDETAIL:`` runs are
  consumed up to (not including) the escaped separator or the repr tail —
  a quote followed by the run of ``)``/``]`` closers ``repr()`` ends with
  (``')`` plain, ``')])`` inside an ExceptionGroup's list), preserving it.
  The lookahead's final bare ``$`` leg is FAIL-CLOSED (#107's security-policy
  change, inverting 0.7.0's pinned "unterminated run is left alone"): an
  escaped DETAIL with no delimiter after it — no closing quote, or a quote
  with more text behind it — scrubs THROUGH END OF LINE. The consumer
  chain's own stated policy: a delimiter miss must delete more text, never
  less of the secret. The two shapes 0.7.0 pinned as left-alone non-matches
  are subsumed by that leg and are pinned here scrubbing.
- uri_userinfo: ``scheme://user:password@host`` -> ``scheme://user:***@host``
  (empty username handled, password ends at the first ``@``, ``\\b`` word
  boundary before the scheme — including the Unicode cases where CPython's
  ``\\w`` and a naive Rust alphanumeric check disagree).
- uri_query_creds / libpq_conninfo_creds: the password-family connection
  parameters, ONE pass under two names (the two anchor grammars of the live
  chain's single combined regex): ``[?&]name=value`` and the libpq keyword
  form ``name=value`` (a non-``[A-Za-z0-9_]`` char — or text start — before
  the name). Names are the five credential parameters
  (``password``/``passphrase``/``passwd``/``pwd``/``sslpassword``) matched
  CASE-INSENSITIVELY (0.7.0 matched exact lowercase and shipped every other
  casing's value verbatim); the value is a libpq single-quoted string
  (spaces, ``\\'``/``\\\\`` escapes) or an unquoted token to whitespace/``&``
  — deliberately NOT stopping at ``@`` (0.7.0 did, leaving the tail of a
  password that legally carries one riding after the ``***``). Selecting
  both names (``rules=None`` included) runs the combined pass once, never
  two sequential substitutions.
- canonical order: the rules apply pg_detail_lines -> uri_userinfo ->
  the conninfo pass, each rule a whole pass before the next; the order is a
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

import collections.abc
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import reference_scrub_log_text
from tors import scrub_log_text

# The extended credential-key set's names (the uri_query_creds_extended
# rule), one spelling for the per-name vectors and the completeness lanes
# below (src/scrub_impl.rs carries the sources and the judicious cuts).
EXTENDED_RULE_NAMES = (
    "sig",
    "api_key",
    "apikey",
    "key",
    "access_key",
    "sas_token",
    "token",
    "secret",
    "passkey",
    "auth",
)

PG = ["pg_detail_lines"]
URI_USER = ["uri_userinfo"]
URI_QUERY = ["uri_query_creds"]
LIBPQ = ["libpq_conninfo_creds"]


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

    def test_the_exception_group_gutters_are_absorbed(self) -> None:
        # #107: one `| `/`+ ` layer per ExceptionGroup nesting level in
        # traceback.format_exception's rendering; a non-DETAIL header line
        # through the same gutters stays.
        assert scrub_log_text("    +   | DETAIL: row-848") == ""
        assert scrub_log_text("  | DETAIL: v\nnext") == "\nnext"
        assert scrub_log_text("||DETAIL: v") == ""
        assert scrub_log_text("  | ExceptionGroup: x\n") == "  | ExceptionGroup: x\n"

    def test_no_terminator_is_fail_closed_through_end_of_line(self) -> None:
        # #107's fail-closed leg, inverting 0.7.0's pinned non-match: a
        # delimiter miss (no closing quote, no trailing escaped newline)
        # scrubs THROUGH END OF LINE — a lookahead miss must delete more
        # text, never less of the secret.
        assert scrub_log_text("E('a\\nDETAIL: leaks") == "E('a"

    def test_real_newline_termination_is_fail_closed_too(self) -> None:
        # The other 0.7.0-pinned non-match, subsumed by the same leg: the
        # run scrubs to the line's end, the newline itself stays.
        assert scrub_log_text("E('a\\nDETAIL: leaks\nnext line") == "E('a\nnext line"

    def test_a_quote_not_at_end_of_line_is_not_a_terminator(self) -> None:
        # The run continues past a mid-line quote; with no closer-run at
        # EOL the fail-closed leg takes the line end.
        assert scrub_log_text("E('a\\nDETAIL: v')  tail") == "E('a"

    def test_the_nested_repr_closer_run_is_preserved(self) -> None:
        # `')])`: a quote plus the run of `)`/`]` closers an
        # ExceptionGroup's list rendering ends with — the run is kept, one
        # more `])` per nesting level.
        assert scrub_log_text("E('m\\nDETAIL: v')])") == "E('m')])"
        assert scrub_log_text("E('m\\nDETAIL: v')])')") == "E('m')"
        # A `]` before the quote is payload, consumed.
        assert scrub_log_text("E('m\\nDETAIL: v]')") == "E('m')"

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

    def test_names_are_case_insensitive(self) -> None:
        # #107: libpq parameter names are case-insensitive and operators'
        # DSNs echo back whatever casing was written; 0.7.0 matched exact
        # lowercase and shipped every other casing's value verbatim.
        assert scrub_log_text("?Password=x&PASSWORD=y&passwords=z&passw=w") == (
            "?Password=***&PASSWORD=***&passwords=z&passw=w"
        )

    def test_sslpassword_is_a_credential_name(self) -> None:
        # #107: the client-TLS key's passphrase joined the name set.
        assert scrub_log_text("postgresql://h/db?sslpassword=p") == (
            "postgresql://h/db?sslpassword=***"
        )

    def test_value_runs_to_whitespace_or_ampersand_at_rides_along(self) -> None:
        # #107: the value class no longer stops at `@` — a password may
        # legally carry one, and 0.7.0 left the tail riding after the mask.
        assert scrub_log_text("?password=a b?pwd=c@d&passwd=e") == (
            "?password=*** b?pwd=***&passwd=***"
        )
        assert scrub_log_text("?password=a@b") == "?password=***"
        assert scrub_log_text("?password=a@b@c&x=1") == "?password=***&x=1"

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


class TestLibpqConninfoCreds:
    """The fourth named rule (#107): the libpq keyword/value conninfo
    anchor grammar of the shared conninfo pass."""

    def test_the_keyword_form_needs_neither_scheme_nor_query_delimiter(self) -> None:
        assert scrub_log_text("host=h password=p") == "host=h password=***"
        assert scrub_log_text("password=p") == "password=***"

    def test_a_longer_word_tail_is_not_an_anchor(self) -> None:
        # The lookbehind: `cpwd=` is not `pwd=`, `apassword=` not
        # `password=`, `_password=` not `password=`.
        for text in ("cpwd=x pwd=y", "apassword=x", "_password=x", "1password=x"):
            if text == "cpwd=x pwd=y":
                assert scrub_log_text(text) == "cpwd=x pwd=***"
            else:
                assert scrub_log_text(text) is text, text

    def test_the_lookbehind_class_is_ascii_so_a_unicode_letter_anchors(self) -> None:
        # The live chain spells the class `[A-Za-z0-9_]` explicitly — a
        # Unicode letter is not in it, so `épassword=x` masks (where the
        # userinfo rule's Unicode `\w` boundary check would disagree).
        assert scrub_log_text("épassword=x") == "épassword=***"

    def test_single_quoted_values_carry_spaces_and_escapes(self) -> None:
        assert scrub_log_text("host=db PASSWORD='hun ter2'") == "host=db PASSWORD=***"
        assert scrub_log_text("host='db host' password='p w' user=u") == (
            "host='db host' password=*** user=u"
        )
        assert scrub_log_text("?password='a\\'b'&x=1") == "?password=***&x=1"
        assert scrub_log_text("?password='a\\\\'&x=1") == "?password=***&x=1"
        assert scrub_log_text("password='a\\'\\'' x") == "password=*** x"

    def test_an_unterminated_quote_falls_back_to_the_token_leg(self) -> None:
        # The quoted leg fails (no closing quote); the unquoted token —
        # which keeps the leading quote — takes the same start.
        assert scrub_log_text("?password='unterminated") == "?password=***"
        assert scrub_log_text("?password='unterminated &password=x") == (
            "?password=*** &password=***"
        )

    def test_a_real_newline_inside_the_quotes_rides_along(self) -> None:
        # The chain's quoted class `[^'\\]` accepts a real newline; only a
        # backslash's follower may not be one.
        assert scrub_log_text("?password='multi\nline real-nl'&x=1") == "?password=***&x=1"

    def test_a_question_mark_before_a_name_is_also_a_keyword_anchor(self) -> None:
        # `?` is a non-`[A-Za-z0-9_]` char: the libpq-only grammar masks
        # it too, which is why the two names must share ONE pass — the
        # combined leftmost scan, never two sequential substitutions.
        assert scrub_log_text("host=h password=p ?password=q", LIBPQ) == (
            "host=h password=*** ?password=***"
        )
        assert scrub_log_text("host=h password=p ?password=q", URI_QUERY) == (
            "host=h password=p ?password=***"
        )

    def test_the_two_names_run_one_combined_pass(self) -> None:
        text = "host=h password=p ?password=q"
        both = scrub_log_text(text, ["uri_query_creds", "libpq_conninfo_creds"])
        assert both == scrub_log_text(text) == "host=h password=*** ?password=***"


class TestUriQueryCredsExtended:
    """The fifth named rule: the SAME ``[?&]`` URI-query anchor over the
    EXTENDED credential-key set — the shared five plus the ops-standard
    query-parameter names the adoption verdict flagged (``sig=``,
    ``api_key=``, ``sas_token=``). The set's sources are the published
    scanner lists, transcribed and closed (ESLint
    ``no-sensitive-data-in-query``'s default sensitive terms,
    detect-secrets' AWS secret-keyword list, Azure's own SAS query
    grammar ``?sv=...&sig=...``; the problem class is CWE-598 — query
    strings land in access logs, proxy logs, browser history, and the
    ``Referer`` header). A NEW NAME, not an ``extra_keys=`` parameter:
    the design charter closes the choice (new scrubs arrive as new named
    rules with their own pinned contracts, never as parameters — a
    pinned contract a caller can widen is not pinned).

    Byte-identity discipline: the shared five's vectors are unchanged
    under every lane, including the default chain (the superset lane
    only ADDS names)."""

    EXTENDED_NAMES = EXTENDED_RULE_NAMES

    def test_every_extended_name_is_masked_name_preserved(self) -> None:
        for name in self.EXTENDED_NAMES:
            assert scrub_log_text(f"?{name}=v&x=1", ["uri_query_creds_extended"]) == (
                f"?{name}=***&x=1"
            ), name

    def test_the_shared_five_ride_along_superset_lane(self) -> None:
        for name in ("password", "passphrase", "passwd", "pwd", "sslpassword"):
            assert scrub_log_text(f"?{name}=v", ["uri_query_creds_extended"]) == (
                f"?{name}=***"
            ), name

    def test_names_are_case_insensitive(self) -> None:
        assert scrub_log_text("?SIG=abc&Api_Key=x&TOKEN=y") == "?SIG=***&Api_Key=***&TOKEN=***"

    def test_the_default_chain_includes_the_extension(self) -> None:
        # The full chain runs the widest key set: the adoption verdict's
        # whole point (a scrubber that leaves ?sig= verbatim without an
        # opt-in is the under-redaction direction). The shared five's
        # default-chain vectors do not move (the parity corpus pins them
        # lane by lane).
        assert scrub_log_text("?sig=abc&api_key=x") == "?sig=***&api_key=***"
        assert scrub_log_text("?password=x&token=y") == "?password=***&token=***"

    def test_the_base_rule_still_answers_only_the_shared_five(self) -> None:
        # uri_query_creds's own contract does not widen: sig/api_key/
        # sas_token/key are invisible to it (an explicit-rules caller
        # keeps the exact five-name behavior).
        for name in self.EXTENDED_NAMES:
            assert scrub_log_text(f"?{name}=v", ["uri_query_creds"]) == f"?{name}=v", name

    def test_delimiter_boundaries(self) -> None:
        # The anchor is the char IMMEDIATELY before the name: `?`/`&`
        # fire, a word char never does (the longest-first resolution
        # puts the check on the right char: `xapi_key=`'s "api_key"
        # suffix match anchors on `x`, a word char — no mask).
        assert scrub_log_text("page?key=v", ["uri_query_creds_extended"]) == "page?key=***"
        assert scrub_log_text("&&key=v", ["uri_query_creds_extended"]) == "&&key=***"
        assert scrub_log_text("??key=v", ["uri_query_creds_extended"]) == "??key=***"
        assert scrub_log_text("?xkey=v", ["uri_query_creds_extended"]) == "?xkey=v"
        assert scrub_log_text("xkey=v", ["uri_query_creds_extended"]) == "xkey=v"
        assert scrub_log_text("?oauth_token=v", ["uri_query_creds_extended"]) == "?oauth_token=v"
        # The `#`-fragment and host splits are the caller's URL parsing:
        # the grammar sees only the query-string text it is given.
        assert scrub_log_text("https://h/p?sig=abc", ["uri_query_creds_extended"]) == (
            "https://h/p?sig=***"
        )

    def test_longest_first_suffix_resolution(self) -> None:
        # `key` is a trailing substring of `api_key`/`access_key`, and
        # `token` of `sas_token`: the LONGEST name wins, whose anchor
        # sits before the whole name. The shorter suffix's own anchor
        # position would be a name char — blocked either way.
        assert scrub_log_text("?api_key=v", ["uri_query_creds_extended"]) == "?api_key=***"
        assert scrub_log_text("?access_key=v", ["uri_query_creds_extended"]) == (
            "?access_key=***"
        )
        assert scrub_log_text("?sas_token=v", ["uri_query_creds_extended"]) == (
            "?sas_token=***"
        )

    def test_value_spanning_to_end_of_string(self) -> None:
        # The token leg runs to whitespace/`&` — end of text included —
        # and a value may carry `=`/`@` (the #107 class).
        assert scrub_log_text("?sig=to_end_of_string", ["uri_query_creds_extended"]) == (
            "?sig=***"
        )
        assert scrub_log_text("?key=a=b@c", ["uri_query_creds_extended"]) == "?key=***"
        assert scrub_log_text("?key=a b?key=c", ["uri_query_creds_extended"]) == (
            "?key=*** b?key=***"
        )

    def test_quoted_values_and_the_empty_value_boundary(self) -> None:
        # The libpq value legs are the shared grammar's.
        assert scrub_log_text("?key='a b'&x=1", ["uri_query_creds_extended"]) == (
            "?key=***&x=1"
        )
        assert scrub_log_text("?key='unterminated &sig=x", ["uri_query_creds_extended"]) == (
            "?key=*** &sig=***"
        )
        assert scrub_log_text("?sig=", ["uri_query_creds_extended"]) == "?sig="
        assert scrub_log_text("?sig", ["uri_query_creds_extended"]) == "?sig"

    def test_the_lookbehind_anchor_stays_the_shared_five(self) -> None:
        # The extension is URI-query-shaped: the libpq keyword grammar
        # keeps the shared five (`key=k` at text start is a libpq
        # keyword shape no source names).
        assert scrub_log_text("key=k password=p", ["libpq_conninfo_creds"]) == "key=k password=***"
        assert scrub_log_text(
            "key=k password=p", ["libpq_conninfo_creds", "uri_query_creds_extended"]
        ) == "key=k password=***"

    def test_the_three_names_still_run_one_combined_pass(self) -> None:
        text = "host=h password=p ?sig=q&token=r"
        all_three = scrub_log_text(
            text, ["uri_query_creds", "uri_query_creds_extended", "libpq_conninfo_creds"]
        )
        assert all_three == scrub_log_text(text) == "host=h password=*** ?sig=***&token=***"

    def test_an_extended_key_value_never_survives_the_default_chain(self) -> None:
        # Redaction completeness, per name: an alnum payload planted
        # behind each extended key shape never survives the full chain.
        for name in (*EXTENDED_RULE_NAMES, "password"):
            assert scrub_log_text(f"?{name}=s3cretVal99&x=1") == f"?{name}=***&x=1", name


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
        # #107: the value runs through the `@` now (a password may legally
        # carry one), so the mask claims `zz@host` whole.
        assert (
            scrub_log_text("scheme://user:pa?password=zz@host", URI_QUERY)
            == "scheme://user:pa?password=***"
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

    def test_escaped_detail_fail_closed_eats_the_password_before_userinfo_sees_it(self) -> None:
        # Order interaction, post-#107: the fail-closed DETAIL deletion
        # scrubs the unterminated run through end of line — the tail (the
        # userinfo password included) is already gone when the userinfo
        # rule runs, so nothing is left for it to mask. (0.7.0 left the run
        # alone here and the userinfo mask claimed the password instead;
        # the current chain deletes more, the fail-closed direction.)
        text = "a://u:p\\nDETAIL:x@h"
        assert scrub_log_text(text) == "a://u:p"
        assert scrub_log_text(text, URI_USER) == "a://u:***@h"
        assert scrub_log_text(text, PG) == "a://u:p"


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
            "'uri_query_creds', 'uri_query_creds_extended', "
            "'libpq_conninfo_creds', 'secret_tokens'), "
            "not \"uri_creds\""
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
        assert scrub_log_text("postgres://u:pw@h", rules=["uri_userinfo"]) == (
            "postgres://u:***@h"
        )


class LyingHugeLen(collections.abc.Sequence):
    """The #112 bomb: a Sequence whose ``__len__`` reports 2**62. pyo3's
    ``Option<Vec<String>>`` extraction sized the Vec from that and died in
    ``Vec::with_capacity`` as an UNCAATCHABLE
    ``pyo3_runtime.PanicException: capacity overflow`` (PanicException
    derives from BaseException, so ``except Exception`` never sees it).
    The bounded walk (``src/py/_borrow.rs``'s ``bounded_str_list``) never
    reads ``__len__``."""

    def __len__(self) -> int:
        return 2**62

    def __getitem__(self, i: int) -> str:
        if i >= 2:
            raise IndexError
        return ("uri_userinfo", "pg_detail_lines")[i]


class EndlessSequence(collections.abc.Sequence):
    """A Sequence that never runs out: the walk must abort at the cap, not
    loop forever (and never trust the lying-huge ``__len__``)."""

    def __len__(self) -> int:
        return 2**62

    def __getitem__(self, i: int) -> str:
        return "uri_userinfo"


class LyingLowLen(collections.abc.Sequence):
    """A Sequence whose ``__len__`` lies LOW (1) but yields two items: the
    walk iterates, it never reserves, so every item is taken."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> str:
        if i >= 2:
            raise IndexError
        return ("uri_userinfo", "pg_detail_lines")[i]


class MidIterationBoom(collections.abc.Sequence):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, i: int) -> str:
        if i == 1:
            raise RuntimeError("boom mid-iteration")
        return "uri_userinfo"


class TestRulesExtractionBoundedWalk:
    """The ``rules=`` extraction's DoS cap and its byte-preservation pins
    (issue #112's residue). The parameter extracts through a bounded
    manual walk (``src/py/_borrow.rs``'s ``bounded_str_list``, the twin of
    ``scrub_pii``'s in ``src/py/pii.rs``) instead of pyo3's
    ``Option<Vec<String>>``: same accepted surface, same refusal bytes,
    and the lying-``__len__`` bomb dies as a catchable ``ValueError``
    instead of the uncatchable PanicException."""

    def test_the_len_bomb_walks_and_never_panics(self) -> None:
        # The #112 repro argument: pyo3's extraction panicked (uncatchable
        # PanicException) on this exact object; the walk ignores __len__,
        # takes the two honest items, and the scrub succeeds. A PanicException
        # derives from BaseException, so this returning normally IS the
        # never-panics pin; the catchable-ValueError refusal path is pinned
        # by the endless and 100_001 rows below.
        assert (
            scrub_log_text("postgres://u:pw@h", rules=LyingHugeLen())
            == scrub_log_text("postgres://u:pw@h", ["uri_userinfo", "pg_detail_lines"])
        )

    def test_an_endless_sequence_refuses_at_the_cap(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            scrub_log_text("postgres://u:pw@h", rules=EndlessSequence())
        assert "yielded too many items" in str(excinfo.value)

    def test_an_honest_100k_list_succeeds(self) -> None:
        text = "postgres://u:pw@h"
        expected = scrub_log_text(text, ["uri_userinfo"])
        assert scrub_log_text(text, ["uri_userinfo"] * 100_000) == expected

    def test_a_100001_item_list_refuses(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            scrub_log_text("postgres://u:pw@h", rules=["uri_userinfo"] * 100_001)
        assert str(excinfo.value) == (
            "scrub_log_text() rules sequence yielded too many items: "
            "refusing an unbounded batch"
        )

    def test_a_len_that_lies_low_takes_every_item(self) -> None:
        # Both names apply (the scrubbed value proves pg_detail_lines ran
        # even though __len__ said 1): the walk never consulted it.
        assert (
            scrub_log_text("pg://u:p\\nDETAIL:x@h", rules=LyingLowLen())
            == scrub_log_text("pg://u:p\\nDETAIL:x@h", ["uri_userinfo", "pg_detail_lines"])
        )

    def test_a_bare_str_keeps_pyo3s_own_refusal_bytes(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            scrub_log_text("postgres://u:pw@h", rules="uri_userinfo")  # type: ignore[arg-type]
        assert str(excinfo.value) == "Can't extract `str` to `Vec`"

    def test_a_non_sequence_keeps_pyo3s_refusal_bytes(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            scrub_log_text("postgres://u:pw@h", rules={"uri_userinfo"})  # type: ignore[arg-type]
        assert str(excinfo.value) == "'set' object is not an instance of 'Sequence'"

    def test_bytes_are_refused_on_the_first_int_item(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            scrub_log_text("postgres://u:pw@h", rules=b"uri_userinfo")  # type: ignore[arg-type]
        assert str(excinfo.value) == "'int' object is not an instance of 'str'"

    def test_a_mid_iteration_error_propagates_unchanged(self) -> None:
        with pytest.raises(RuntimeError, match="boom mid-iteration"):
            scrub_log_text("postgres://u:pw@h", rules=MidIterationBoom())  # type: ignore[arg-type]


class TestIdentityReturn:
    def test_clean_text_comes_back_as_the_same_object(self) -> None:
        text = "ordinary error text, no credentials, no DETAIL"
        assert scrub_log_text(text) is text

    def test_each_rule_alone_returns_identity_when_it_does_not_fire(self) -> None:
        for rules in (PG, URI_USER, URI_QUERY, LIBPQ):
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
    """A char run the USERINFO password class accepts: no Python whitespace
    (``str.isspace`` is exactly ``re.``'s ``\\s`` set — pinned by
    ``TestCredentialPayloadEquivalence`` below and exhaustively by
    ``test_space_table_exhaustive`` in test_scrub_log_text_parity.py),
    no ``@`` (the class stops there), and no backslash — a payload
    carrying ``\\n``-shaped text belongs to the escaped-DETAIL interaction
    (the fail-closed deletion eats it, differentially pinned), excluded
    here so the template properties below pin the VALUE classes in
    isolation."""
    return "".join(ch for ch in s if not ch.isspace() and ch not in "@\\")


def _legal_conninfo_value(s: str) -> str:
    """A char run the conninfo value's UNQUOTED token class accepts: no
    Python whitespace, no ``&`` (the class stops there — ``@`` rides along
    since #107), no backslash (the escaped-DETAIL interaction, as above),
    and no quote (a leading ``'`` would route the value through the
    libpq quoted leg, whose whole-run consumption is pinned literally in
    ``TestLibpqConninfoCreds`` and differentially everywhere — the token
    property below pins the token leg in isolation)."""
    return "".join(ch for ch in s if not ch.isspace() and ch not in "&\\'")


class TestCredentialPayloadEquivalence:
    def test_isspace_is_exactly_re_s_over_the_edge_alphabet(self) -> None:
        """Pin the helper's core claim without re-paying the 1.1M-char
        exhaustive loop (that lives in the parity file): over ASCII
        whitespace, the U+001C..U+001F seam, NBSP/U+2028/ZWSP, and the
        credential delimiters, ``ch.isspace()`` agrees with ``re \\s`` and
        the helpers keep a char exactly when the class they model would
        accept it (userinfo: ``[^\\s@]``; conninfo token: ``[^\\s&]``)."""
        pat = re.compile(r"\s")
        edge = [
            chr(cp)
            for cp in list(range(0x00, 0x30))
            + [0x7F, 0xA0, 0x1C, 0x1D, 0x1E, 0x1F, 0x2028, 0x2029, 0x200B, 0x3000, 0x093E, 0x24B6]
        ]
        edge += ["&", "@", "a", ":", "/", "?", "=", "*", "p", "\\", "'"]
        for ch in edge:
            assert ch.isspace() == (pat.match(ch) is not None), repr(ch)
            assert (ch in _legal_credential_payload(ch)) == (
                not ch.isspace() and ch not in "@\\"
            ), repr(ch)
            assert (ch in _legal_conninfo_value(ch)) == (
                not ch.isspace() and ch not in "&\\'"
            ), repr(ch)

    @given(_ANY_TEXT)
    @settings(max_examples=200)
    def test_helpers_match_re_classes_char_by_char(self, text: str) -> None:
        pat = re.compile(r"\s")
        for ch in text:
            assert (ch in _legal_credential_payload(ch)) == (
                pat.match(ch) is None and ch not in "@\\"
            ), repr(ch)
            assert (ch in _legal_conninfo_value(ch)) == (
                pat.match(ch) is None and ch not in "&\\'"
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

    @given(_ANY_TEXT)
    @settings(max_examples=300)
    def test_the_extended_rule_is_idempotent_and_byte_preserving(self, text: str) -> None:
        # The extended rule's own lane: the identity return is the
        # ORIGINAL object (byte-preservation — a pass that never fired
        # must not spend a copy), and the rule converges by the second
        # pass (its `***` splice re-matches its own value class).
        once = scrub_log_text(text, ["uri_query_creds_extended"])
        twice = scrub_log_text(once, ["uri_query_creds_extended"])
        assert scrub_log_text(twice, ["uri_query_creds_extended"]) == twice
        if once == text:
            assert once is text

    @given(_ANY_TEXT)
    @settings(max_examples=300)
    def test_the_extended_rule_matches_the_reference(self, text: str) -> None:
        # Oracle parity on the single extended lane, over the same raw
        # alphabet the whole-chain hypothesis lanes run.
        assert scrub_log_text(text, ["uri_query_creds_extended"]) == (
            reference_scrub_log_text(text, ["uri_query_creds_extended"])
        )

    @given(_ANY_TEXT)
    @settings(max_examples=200)
    def test_an_extended_key_value_never_survives_the_default_chain(self, text: str) -> None:
        # Redaction completeness: an alnum payload planted behind each
        # extended key shape never survives the full chain (the value
        # classes accept it whole, so a survivor is unambiguously a miss).
        payload = _legal_conninfo_value(text)
        if not payload:
            return
        for name in EXTENDED_RULE_NAMES:
            assert scrub_log_text(f"?{name}={payload}&x=1") == f"?{name}=***&x=1", name

    def test_the_param_mask_can_unblock_a_userinfo_match_on_pass_two(self) -> None:
        """The one documented non-idempotence class, pinned literally
        with oracle parity at BOTH passes: pass 1's param value eats the
        `/` that was capping the userinfo user run (`u?pwd=a` stops at
        the `/`), so pass 2's user class spans the `***` and the `&`
        (`u?pwd=***&`) and the userinfo rule fires — the same shape CI's
        fuzz-smoke found (crash-9268942a). Pass 3 re-matches the
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
        # userinfo pass can never fire inside it, and the conninfo value's
        # token leg — whatever an escaped-DETAIL deletion leaves of it,
        # though the payload class excludes backslashes so none fires
        # here — is always masked whole (the quoted leg's shapes are
        # pinned literally in TestLibpqConninfoCreds and differentially
        # everywhere).
        payload = _legal_conninfo_value(text)
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
