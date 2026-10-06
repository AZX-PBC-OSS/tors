"""The streaming repairer's suite: ``tors.JsonRepairer``, the stateful
incremental repairer for LLM token streams.

The discipline mirrors the sibling json_repair files: the whole-text engine
(``tors.repair_json``) is the differential reference for every class the
stream can decide incrementally (the chunk-boundary sweep feeds every split
position of every corpus text and demands ``end() == repair_json``), the
pinned ``json-repair`` package is the oracle for the repaired-complete
case, and the scaling idiom (``tests/test_scaling_pins.py``'s
``_assert_linear_per_doubling`` + ``loop_harness``) pins the linearity
contract: total streaming time grows ~4x when the input and the chunk
count quadruple, where the per-chunk re-parse anti-pattern measures ~16x.

The documented contract (docs/api.md ``## tors.JsonRepairer``):
``push`` returns the newly emitted repaired text (the delta), and
``snapshot``/``end`` render the stream closed into valid JSON. The
engine-equality classes and the documented divergences are listed in
``src/json_repair/streaming.rs``'s module docs; the sweep corpus below is
drawn from the equality classes, and the divergence shapes get their own
pins (valid output, documented difference).
"""

from __future__ import annotations

import json
import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from loop_harness import first_clean, min_wall_ms

json_repair_lib = pytest.importorskip("json_repair")  # pin: json-repair==0.63.5

# The growth gate per doubling: tests/test_scaling_pins.py's own idiom
# (a linear path measures ~2.0-2.3x per doubling; 3.0x sits ~1.3x above
# the band; the re-parse anti-pattern's ~4x per doubling blows through).
LINEAR_GATE_PER_DOUBLING = 3.0

# --- the corpus ---------------------------------------------------------------
#
# Three classes, each text ALSO fed through every chunk-boundary split:
#
# - VALID: strict-loadable documents; end() == repair_json(text) == the
#   oracle's repair_json(text) (the repaired-complete differential).
# - TRUNCATION: prefixes of valid documents (the LLM ran out of tokens);
#   end() == repair_json(text) (the truncation-heal differential; the
#   engine heals the same cut by the same rules).
# - MALFORMED: the repair classes the stream decides incrementally
#   (missing separators, single quotes, bare words, Python literals,
#   comments, prose prefixes, tuples, mismatched closers, number
#   rollback); end() == repair_json(text) for every text here too.
#
# Deliberately ABSENT (the documented divergences, pinned separately):
# string re-synchronization ('{"a": "hello}'), doubled quotes, in-container
# comments, multi-top-level documents, and the mid-pair first-member shape
# ('{"a"'), where the engine's whole-text machinery re-decides earlier
# text in ways a linear stream cannot replay.

VALID = [
    '{"a": 1}',
    '{ "a" : 1 }',
    '[1, 2, 3]',
    '  [1, 2]  ',
    '{"a": {"b": [1, 2]}, "c": null}',
    "[]",
    "{}",
    '""',
    '"hi"',
    '"\\u00e9"',
    '"\u00e9"',
    '"\U0001f600"',
    '12',
    '-12.5',
    '1.5e3',
    'NaN',
    'Infinity',
    '{"a": NaN}',
    '{"a": Infinity}',
    '{"a": -Infinity}',
    '[NaN, 1]',
    '{"a": 1.5, "b": -0.0}',
    '{"a": 123456789012345678901234567890}',
    '{"": 1}',
    r'{"a": "x\ny\tz"}',
    r'{"a": "x\/y"}',
    r'{"a": "x\ud83d\ude00y"}',
    '{"key": "value with, commas; and: colons"}',
]

TRUNCATION = [
    '{"a": "hel',
    '{"a": [1, 2',
    '{"a": [1,',
    '{"a": [',
    '{"a": {',
    '{"a":',
    '{"a": "x", "b"',
    '{"a": "x", "b": 2, ',
    '{"a": 1, "b": [1, {"c": "d',
    '[1, 2,',
    '[',
    '{',
    '"hi',
    '{"a": "x\\ud83d\\ude00',
    '{"a": "x\\u00',
    '{"a": "x\\',
    '{"a": 1e',
    '{"a": 1e+',
    '{"a": -',
    '{"a": 1,000',
    '{"a": tru',
    '{"a": None',
    '[tru',
    '{"a": [1 2',
    '{"a": "hel" ',
    '{"a": 12',
    '{"a": [1, 2} ',
]

MALFORMED = [
    '{"a": 1 "b": 2}',
    '[1 2]',
    '{"a": 1,}',
    '[1, 2,]',
    "{'a': 1}",
    "['a', 'b']",
    '{a: 1}',
    '{a: 1, b: 2}',
    '{12: 1}',
    '{"a": None}',
    '{"a": True, "b": False}',
    '[True, False, None]',
    '{"a": 1e}',
    '{"a": 1e+}',
    '{"a": -}',
    '{"a": 1,000}',
    '{"a": 12abc}',
    '{"a": 1 2}',
    # note: '{"a": 12_abc}' is deliberately absent: the digit-group
    # underscore followed by a bare word is a shape where the whole-text
    # engine's rewind arithmetic itself diverges from the oracle
    # (tors: "2_abc", oracle: "12_abc"), so no streaming behavior can be
    # differentially pinned there; see the streaming module's divergence
    # list.
    '{"a": [1, 2}',
    '{"a": 1]',
    '[{"a": 1]',
    '{"a": [1, }',
    '{"a": [}',
    '[1,, 2]',
    '(1)',
    '(1,)',
    '(1, 2)',
    '{"a": (1, 2)}',
    '// lead\n{"a": 1}',
    '/*x*/{"a": 1}',
    '# c\n{"a": 1}',
    'Answer: {"a": 1}',
    'Here: [1, 2]',
    '{"a": 1} junk',
    '{a: 1, b}',
    '{"a": "x", "b"}',
]

CORPUS = VALID + TRUNCATION + MALFORMED


def stream(text: str, splits: list[int], *, ensure_ascii: bool = True) -> tuple[list[str], str]:
    """Push ``text`` through a fresh repairer cut at ``splits`` (char
    offsets); return (the per-push deltas, the end() output)."""
    r = tors.JsonRepairer(ensure_ascii=ensure_ascii)
    deltas = []
    prev = 0
    for p in sorted(splits):
        deltas.append(r.push(text[prev:p]))
        prev = p
    deltas.append(r.push(text[prev:]))
    return deltas, r.end()


def _assert_loads_or_empty(out: str) -> None:
    """The valid-output invariant: the machine's closed renders are
    loadable JSON, or the nothing-recoverable empty string."""
    if out == "":
        return
    json.loads(out)


# --- chunk-boundary sweeps ------------------------------------------------------


class TestChunkBoundarySweep:
    @pytest.mark.parametrize("text", CORPUS)
    def test_every_split_matches_the_engine(self, text: str) -> None:
        """Every split position of every corpus text: two pushes, and
        end() is byte-identical to the whole-text engine on the same
        total text (the zero-re-parse contract's observable)."""
        want = tors.repair_json(text)
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    @pytest.mark.parametrize("text", CORPUS)
    def test_every_prefix_snapshot_is_valid(self, text: str) -> None:
        """Every prefix of every corpus text: snapshot() is loadable JSON
        (or the empty sentinel) and equals end() on that prefix: the
        'valid JSON now' contract holds at every point of the stream."""
        for k in range(1, len(text) + 1):
            prefix = text[:k]
            r = tors.JsonRepairer()
            r.push(prefix)
            snap = r.snapshot()
            _assert_loads_or_empty(snap)
            assert snap == r.end(), f"snapshot != end at prefix {k} of {text!r}"

    @pytest.mark.parametrize("text", CORPUS)
    def test_full_text_matches_the_engine(self, text: str) -> None:
        """Without any split (one push), snapshot() and end() both match
        the whole-text engine's answer on the corpus text itself."""
        want = tors.repair_json(text)
        r = tors.JsonRepairer()
        r.push(text)
        assert r.snapshot() == want, f"{text!r}: snapshot != engine"
        assert r.end() == want, f"{text!r}: end != engine"

    def test_many_random_splits_match_the_one_push_answer(self) -> None:
        """Random multi-chunk splits (seeded): end() is chunking-
        invariant, equal to the one-push answer on every corpus text."""
        rng = random.Random(20261005)
        for text in CORPUS:
            one = tors.JsonRepairer()
            one.push(text)
            want = one.end()
            points = sorted(rng.sample(range(1, len(text)), min(4, len(text) - 1)))
            _, got = stream(text, points)
            assert got == want, f"splits {points} of {text!r}: {got!r} != {want!r}"

    @pytest.mark.parametrize("text", VALID)
    def test_repaired_complete_case_matches_the_oracle(self, text: str) -> None:
        """The repaired-complete differential: a valid document streamed
        through chunks repairs to the oracle's own answer."""
        want = json_repair_lib.repair_json(text)
        rng = random.Random(42)
        points = sorted(rng.sample(range(1, len(text)), min(3, len(text) - 1)))
        _, got = stream(text, points)
        assert got == want, f"{text!r}: {got!r} != oracle {want!r}"

    @pytest.mark.parametrize("text", VALID)
    def test_ensure_ascii_false_streams_verbatim(self, text: str) -> None:
        """ensure_ascii=False streams non-ASCII verbatim and still agrees
        with the engine's own ensure_ascii=False answer."""
        want = tors.repair_json(text, ensure_ascii=False)
        rng = random.Random(43)
        points = sorted(rng.sample(range(1, len(text)), min(3, len(text) - 1)))
        _, got = stream(text, points, ensure_ascii=False)
        assert got == want, f"{text!r}: {got!r} != {want!r}"


# --- the documented decisions ---------------------------------------------------


class TestDocumentedDecisions:
    def test_truncated_string_heals(self) -> None:
        """The deliverable's own example: a cut-off string value heals to
        the closed quote and container."""
        r = tors.JsonRepairer()
        r.push('{"a": "hel')
        assert r.snapshot() == '{"a": "hel"}'
        assert r.end() == '{"a": "hel"}'

    def test_truncated_containers_heal(self) -> None:
        """The second deliverable example: unclosed containers close with
        their own brackets."""
        r = tors.JsonRepairer()
        r.push('{"a": [1, 2')
        assert r.snapshot() == '{"a": [1, 2]}'
        assert r.end() == '{"a": [1, 2]}'

    def test_partial_literal_tru_is_the_engines_own_partial_semantics(self) -> None:
        """THE DECISION (documented in docs/api.md and the module docs):
        there is NO 'tru' -> 'true' healing. The engine's own partial
        semantics decide: repair_json('tru') is '' (a top-level word is
        prose unless the whole input is one strict JSON value) and
        repair_json('{"a": tru') is '{"a": "tru"}' (a bare word in a
        container is a string). The stream reproduces exactly that, so
        the end()-equals-engine differential holds unbroken. Mid-stream
        the word is held back as a pending token (a push returns the
        text emitted so far, '' here), not passed through raw."""
        assert tors.repair_json("tru") == ""
        r = tors.JsonRepairer()
        assert r.push("tru") == ""
        assert r.end() == tors.repair_json("tru") == ""
        r2 = tors.JsonRepairer()
        assert r2.push('{"a": tru') == '{"a": '
        assert r2.end() == tors.repair_json('{"a": tru') == '{"a": "tru"}'
        # A complete strict scalar commits; a non-JSON word does not.
        assert tors.JsonRepairer().push("12") == ""
        r3 = tors.JsonRepairer()
        r3.push("12")
        assert r3.end() == "12"
        r4 = tors.JsonRepairer()
        r4.push("true")
        assert r4.end() == "true"

    def test_dangling_key_divergence_is_pinned(self) -> None:
        """The documented divergence: a dangling key drops (the engine's
        continuation shape), including the first-member shape where the
        whole-text engine instead falls back to an array."""
        r = tors.JsonRepairer()
        r.push('{"a": "x", "b"')
        assert r.end() == tors.repair_json('{"a": "x", "b"') == '{"a": "x"}'
        r2 = tors.JsonRepairer()
        r2.push('{"a"')
        # The engine's whole-text fallback heals '{"a"' to '["a"]'; the
        # stream drops the dangling member instead ({}), valid JSON with
        # the same information content (none).
        assert r2.end() == "{}"
        assert tors.repair_json('{"a"') == '["a"]'

    def test_trailing_top_level_tokens_drop(self) -> None:
        """The engine keeps trailing junk off the root value; the stream
        drops later top-level VALUES too (the engine may array-wrap
        them; documented divergence, compose with repair_json there)."""
        r = tors.JsonRepairer()
        r.push('{"a": 1} junk')
        assert r.end() == '{"a": 1}'
        r2 = tors.JsonRepairer()
        r2.push('{"a":1}{"b":2}')
        assert r2.end() == '{"a": 1}'
        assert tors.repair_json('{"a":1}{"b":2}') == '[{"a": 1}, {"b": 2}]'

    def test_string_resynchronization_divergence_is_valid(self) -> None:
        """The engine terminates a damaged string at a structural closer;
        the stream closes strings only at the delimiter or end (the
        output stays valid JSON either way)."""
        r = tors.JsonRepairer()
        r.push('{"a": "hello}')
        got = r.end()
        json.loads(got)
        assert got == '{"a": "hello}"}'
        assert tors.repair_json('{"a": "hello}') == '{"a": "hello"}'

    def test_in_container_comment_divergence_is_valid(self) -> None:
        """The engine's in-container comment consumes the value after it;
        the stream skips the comment and parses the value."""
        r = tors.JsonRepairer()
        r.push('{"a": /*x*/ 1}')
        assert r.end() == '{"a": 1}'
        assert tors.repair_json('{"a": /*x*/ 1}') == '{"a": ""}'

    def test_delta_concatenation_and_its_documented_retraction(self) -> None:
        """push returns the newly emitted text; the concatenation is the
        emitted stream, and snapshot/end are the authority when a late
        repair retracts (the dangling-member drop)."""
        r = tors.JsonRepairer()
        deltas = [r.push(part) for part in ('{"a": 1, "b": ', '2 ')]
        # The concatenation is the EMITTED stream (open form): the
        # pending number's chars stream out when the run terminates, and
        # the document's final `}` only exists at close time.
        assert "".join(deltas) == '{"a": 1, "b": 2'
        assert r.end() == '{"a": 1, "b": 2}'
        # The retraction: the dropped member's separator streamed out and
        # was taken back; the caller's concatenation keeps the stale text.
        r2 = tors.JsonRepairer()
        d1 = r2.push('{"a": 1, b')
        d2 = r2.push("}")
        assert "".join([d1, d2]) == '{"a": 1, }'
        assert r2.end() == '{"a": 1}'


# --- escape and unicode state across boundaries ---------------------------------


class TestEscapeAndUnicodeBoundaries:
    def test_backslash_split_across_pushes(self) -> None:
        """The escape state survives a chunk boundary: the backslash in
        one push, its escape char in the next (both the valid and the
        invalid-escape shapes)."""
        for text in [r'{"a": "x\ny"}', r'{"a": "x\qy"}', r'{"a": "x\\"}']:
            want = tors.repair_json(text)
            for k in range(1, len(text)):
                _, got = stream(text, [k])
                assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    def test_unicode_escape_split_across_pushes(self) -> None:
        """A \\u escape's hex digits straddle pushes; the escape still
        decodes (and an incomplete one stays literal, engine parity)."""
        for text in [r'{"a": "x\u00e9y"}', r'{"a": "x\u41"}', r'{"a": "x\u00']:
            want = tors.repair_json(text)
            for k in range(1, len(text)):
                _, got = stream(text, [k])
                assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    def test_surrogate_pair_split_across_pushes(self) -> None:
        """A surrogate pair's two \\u escapes straddle pushes: the pair
        still combines (the held high surrogate is machine state); a lone
        one heals to U+FFFD exactly like the engine."""
        text = '{"a": "x\\ud83d\\ude00y"}'
        want = tors.repair_json(text)
        lo = text.index("\\ud83d")
        hi = text.index("\\ude00")
        for splits in ([lo + 3], [lo + 6], [lo + 3, hi + 3], [hi + 6]):
            _, got = stream(text, splits)
            assert got == want, f"splits {splits}: {got!r} != {want!r}"
        lone = '{"a": "x\\ud83dy"}'
        r = tors.JsonRepairer()
        r.push(lone)
        # A lone high escape decodes to U+FFFD (the engine's documented
        # divergence from the oracle), exactly on the x-prefixed shape.
        assert r.end() == tors.repair_json(lone)
        assert r.end() == '{"a": "x\\ufffdy"}'

    def test_raw_astral_char_split_across_pushes(self) -> None:
        """A raw astral character split across two pushes survives (the
        chunks are str's: the split is at a codepoint boundary by
        construction)."""
        text = '{"a": "\U0001f600x"}'
        want = tors.repair_json(text)
        k = text.index("\U0001f600") + 1
        _, got = stream(text, [k])
        assert got == want

    def test_lone_surrogate_str_refused_at_the_boundary(self) -> None:
        """A str holding a lone surrogate cannot enter: the standard
        str-in convention (UnicodeEncodeError at the argument boundary)."""
        r = tors.JsonRepairer()
        with pytest.raises(UnicodeEncodeError):
            r.push('{"a": "\ud83d"}')


# --- value coercion agreement ---------------------------------------------------


class TestValueCoercionAgreement:
    @pytest.mark.parametrize(
        "text",
        [
            '{"i": 42, "f": 1.5, "g": 1.5e3, "b": true, "n": null, "s": "hi"}',
            '{"neg": -17, "exp": 2.5e-4, "big": 123456789012345678901234567890}',
            '{"currency": "1,000", "frac": "1/2", "range": "10-20"}',
            '{"py": [True, False, None], "trunc": "tru"}',
            '{"nested": {"deep": [1, {"x": 2.5}]}}',
        ],
    )
    def test_decoded_values_agree_with_the_whole_text_engine(self, text: str) -> None:
        """json.loads(end()) is the whole-text engine's repaired value,
        dict/int/float/bool/None shapes included."""
        r = tors.JsonRepairer()
        r.push(text)
        got = json.loads(r.end())
        want = tors.repair_json_loads(text)
        assert got == want, f"{text!r}: {got!r} != {want!r}"

    def test_oracle_value_agreement_on_the_coercion_corpus(self) -> None:
        """The oracle's decoded values agree on the same corpus (the
        repaired-complete differential, loads spelling)."""
        for text in [
            '{"i": 42, "f": 1.5, "b": true, "n": null, "s": "hi"}',
            '{"big": 123456789012345678901234567890}',
            '{"py": [True, False, None]}',
        ]:
            r = tors.JsonRepairer()
            r.push(text)
            want = json_repair_lib.repair_json(text)
            assert json.loads(r.end()) == json.loads(want)


# --- statefulness ---------------------------------------------------------------


class TestStatefulness:
    def test_reset_reuses_the_repairer(self) -> None:
        """reset() drops all state: a new document streams clean, the
        old one cannot leak."""
        r = tors.JsonRepairer()
        r.push('{"a": [1, 2')
        r.reset()
        assert r.snapshot() == ""
        r.push('[1, ')
        r.push('2]')
        assert r.end() == "[1, 2]"
        assert tors.repair_json("[1, 2]") == "[1, 2]"

    def test_end_is_idempotent_and_push_refuses_after(self) -> None:
        """end() twice returns the same text; push after end raises
        ValueError naming the reset()."""
        r = tors.JsonRepairer()
        r.push('{"a": "hel')
        first = r.end()
        assert r.end() == first
        with pytest.raises(ValueError, match="reset"):
            r.push("x")

    def test_depth_cap_raises_value_error(self) -> None:
        """Past the 200-container cap, push raises the engine's
        normalized ValueError (the engine's own depth guard)."""
        r = tors.JsonRepairer()
        unit = '{"a": '
        with pytest.raises(ValueError, match="recursion depth"):
            r.push(unit * 200 + "[")
        # reset recovers.
        r.reset()
        r.push(unit * 199 + "1}")
        json.loads(r.end())

    def test_gil_released_during_push(self) -> None:
        """A push of a large chunk leaves the event loop schedulable (the
        machine pass runs detached; the delta marshalling is the only
        held residue)."""
        import asyncio

        from loop_harness import assert_heartbeat_clean

        # Built once, outside the measured op (the op is the repairer's
        # own work).
        payload = '{"a": "%s"}' % ("x" * 4_000_000)

        def op() -> None:
            r = tors.JsonRepairer()
            r.push(payload)
            r.end()

        # The op runs on a worker thread (the test_gil_release.py shape):
        # the machine pass is detached, so the loop ticks at heartbeat
        # granularity while it runs.
        assert_heartbeat_clean(lambda: asyncio.to_thread(op), samples=2)


# --- the linearity pin ------------------------------------------------------------


class TestStreamingLinearity:
    """The anti-pattern this surface exists to kill: re-running a
    whole-text pass per chunk. With the chunk SIZE fixed and the total
    quadrupled, the re-parse shape costs O(total^2 / chunk) (~16x wall
    per quadrupling; the suture/repair-json-stream publish measures ~15x)
    while a linear stream holds ~4x. Two gates, tests/test_scaling_pins.py's
    idiom: the doubling gate on the streaming wall itself, and the
    streaming-vs-one-shot ceiling (a fixed 64-chunk stream must not cost
    an order of magnitude more than one repair of the same text)."""

    CHUNK = 4_096

    def _document(self, total: int) -> str:
        # Nested containers with strings, numbers and literals: the
        # machine's real shape, generated deterministically.
        parts = []
        size = 0
        i = 0
        while size < total:
            parts.append(f'{{"k{i}": "v{i}", "n{i}": [{i}, {i}.5, true, None]}}')
            size += len(parts[-1]) + 2
            i += 1
        return "[" + ", ".join(parts) + "]"

    def _stream(self, text: str) -> None:
        r = tors.JsonRepairer()
        for i in range(0, len(text), self.CHUNK):
            r.push(text[i : i + self.CHUNK])
        r.end()

    @pytest.mark.timing
    def test_total_time_stays_linear_at_a_fixed_chunk_size(self) -> None:
        """128KiB -> 512KiB at 4KiB chunks (32 -> 128 pushes): measured
        2.55ms -> 10.87ms, ratio 4.26 (~2.1x per doubling), gate 3.0x per
        doubling. The per-chunk re-parse anti-pattern measures ~16x here
        (quadratic in the total at a fixed chunk size)."""
        small_doc, large_doc = self._document(128 * 1024), self._document(512 * 1024)
        assert len(large_doc) // len(small_doc) >= 3

        def measure() -> tuple[float, float]:
            return min_wall_ms(lambda: self._stream(small_doc)), min_wall_ms(
                lambda: self._stream(large_doc)
            )

        def check(walls: tuple[float, float]) -> None:
            small, large = walls
            factor = len(large_doc) / len(small_doc)
            doublings = math.log2(factor)
            allowed = LINEAR_GATE_PER_DOUBLING**doublings
            assert large < allowed * small, (
                f"streaming grew {small:.2f}ms -> {large:.2f}ms for a "
                f"{factor:.1f}x document ({large / small:.2f}x, allowed "
                f"{allowed:.1f}x at {LINEAR_GATE_PER_DOUBLING}x per doubling): "
                "the machine regressed superlinear in the total bytes"
            )

        first_clean(measure, check, samples=3, label="the streaming linearity pin")

    @pytest.mark.timing
    def test_streaming_stays_within_an_order_of_the_one_shot(self) -> None:
        """The 512KiB stream (128 pushes) against ONE whole-text repair
        of the same text: measured 0.54x (the stream is the same single
        pass, minus the engine's re-serialization, plus the per-push
        delta returns), ceiling 10x: the re-parse anti-pattern lands
        ~30x+ here and blows through."""
        doc = self._document(512 * 1024)

        def measure() -> tuple[float, float]:
            return min_wall_ms(lambda: self._stream(doc)), min_wall_ms(
                lambda: tors.repair_json(doc)
            )

        def check(walls: tuple[float, float]) -> None:
            streaming_ms, one_shot_ms = walls
            assert streaming_ms < 10 * one_shot_ms, (
                f"streaming ({streaming_ms:.2f}ms) cost more than 10x the "
                f"one-shot repair ({one_shot_ms:.2f}ms): per-chunk work is "
                "no longer linear in the chunk"
            )

        first_clean(measure, check, samples=3, label="the streaming-vs-one-shot pin")


# --- hypothesis: valid documents through random chunkings --------------------------


class TestHypothesisValidDocuments:
    @given(st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False),
                        lambda leaf: st.lists(leaf, max_size=4)
                        | st.dictionaries(st.text(min_size=1, max_size=6), leaf, max_size=4),
                        max_leaves=20))
    @settings(max_examples=50, deadline=None)
    def test_valid_document_survives_every_chunking(self, doc) -> None:
        """Any JSON document, streamed through random chunk boundaries,
        ends at the engine's own canonical answer for it."""
        text = json.dumps(doc)
        if len(text) < 2:
            return  # a one-char document has no split points
        rng = random.Random(len(text))
        points = sorted(rng.sample(range(1, len(text)), min(5, len(text) - 1)))
        _, got = stream(text, points)
        assert got == tors.repair_json(text)
