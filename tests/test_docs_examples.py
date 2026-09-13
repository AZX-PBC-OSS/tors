"""The docs' worked examples, pinned: every output literal README.md,
docs/api.md, and docs/recipe-transcripts.md show for the features their
sections document is re-derived here against the built extension, the
``test_diff_opcodes_lines.py::TestReconstruction::test_readme_worked_example``
discipline. The README says "the output above is what they actually
return"; this file is what keeps that sentence true: a behavior change
that would turn a documented example into a lie fails here first, before
the docs drift. Literals are copied byte-exact from the docs, including
the trailing spaces the docs show (the recipe's Priya sentence slice
ends in one: UAX #29 SB10/SB11 attach a terminator's trailing space to
the preceding sentence). If one of these ever fails after an
intentional change, the docs and this pin move together, in the same
commit.
"""

from __future__ import annotations

import time
import uuid as stdlib_uuid

import tors


class TestReadmeExamples:
    def test_chat_thread_spliced_hierarchy(self) -> None:
        thread = (
            "Ana: kickoff at nine.\n"
            "Ben: We briefed the U.S. team on the numbers. They asked for a follow-up.\n"
            "Ana: done."
        )
        chunks = tors.chunk_hierarchical(thread, 40, ["\n", None])
        assert chunks == [(0, 21), (22, 59), (59, 95), (96, 106)]
        assert [thread[s:e] for s, e in chunks] == [
            "Ana: kickoff at nine.",
            "Ben: We briefed the U.S. team on the ",
            "numbers. They asked for a follow-up.",
            "Ana: done.",
        ]


class TestApiReferenceExamples:
    def test_scrub_log_text_detail_dsn_and_repr_examples(self) -> None:
        # docs/api.md's scrub_log_text section, pinned the same way: the
        # literals the doc shows, every rules= spelling included.
        detail = (
            "JobError: duplicate key\n"
            "DETAIL:  Key (idempotency_key)=(customer-4417-a3f2) already exists.\n"
            "HINT: unchanged"
        )
        assert tors.scrub_log_text(detail) == "JobError: duplicate key\n\nHINT: unchanged"
        dsn = (
            "connect dsn=postgresql://worker:S3cr3t-x9@db.internal:5432/prod?password=fallback"
        )
        assert tors.scrub_log_text(dsn) == (
            "connect dsn=postgresql://worker:***@db.internal:5432/prod?password=***"
        )
        assert (
            tors.scrub_log_text(
                "JobError('duplicate key\\nDETAIL:  Key (idempotency_key)=(customer-4417) exists.')"
            )
            == "JobError('duplicate key')"
        )
        assert tors.scrub_log_text(
            "connect dsn=postgresql://worker:S3cr3t-x9@db.internal:5432/prod",
            ["uri_userinfo"],
        ) == "connect dsn=postgresql://worker:***@db.internal:5432/prod"

    def test_chunk_by_lines_log_windows(self) -> None:
        log = (
            "INFO boot\nINFO ready\n\nWARN disk at 90%\n"
            "ERROR io failure\nINFO retry ok\n\nINFO shutdown"
        )
        assert tors.chunk_by_lines(log, 2) == [(0, 20), (22, 55), (56, 84)]
        # the api.md chunk_by_lines_iter section's example, pinned directly
        # (not only transitively through the streaming-parity tests): the
        # iterator must yield the doc's literal list on the same `log`
        assert list(tors.chunk_by_lines_iter(log, 2)) == [(0, 20), (22, 55), (56, 84)]
        assert tors.chunk_by_lines(log, 2, overlap=1) == [
            (0, 20),
            (10, 38),
            (22, 55),
            (39, 69),
            (56, 84),
        ]
        # the overlap slice the doc calls out by content: the repeated
        # whole line plus the blank line riding along after it
        assert log[10:38] == "INFO ready\n\nWARN disk at 90%"

    def test_chunk_by_paragraphs_iter_minutes_windows(self) -> None:
        # docs/api.md's chunk_by_paragraphs_iter section, pinned the same
        # way: the literals the doc shows, list and iter spelling alike.
        minutes = (
            "Attendees: Ada, Grace, Edsger.\n\n"
            "Grace: parser rewrite halves latency.\n\n"
            "Edsger: spec drift question, unresolved.\n\n"
            "Next sync moves to Thursday."
        )
        assert tors.chunk_by_paragraphs(minutes, 2) == [(0, 69), (71, 141)]
        assert list(tors.chunk_by_paragraphs_iter(minutes, 2)) == [(0, 69), (71, 141)]
        assert list(tors.chunk_by_paragraphs_iter(minutes, 2, overlap=1)) == [
            (0, 69),
            (32, 111),
            (71, 141),
        ]
        # the overlap slice the doc calls out by content: the repeated
        # whole paragraph, the blank-line gap riding along after it, and
        # the next chunk's first paragraph
        assert minutes[32:111] == (
            "Grace: parser rewrite halves latency.\n\nEdsger: spec drift question, unresolved."
        )

    def test_chunk_hierarchical_thread(self) -> None:
        thread = (
            "Nathan: kicking off the sync.\n"
            "Priya: We briefed the U.S. team on the numbers. "
            "They asked for a follow-up meeting. The budget holds.\n"
            "Nathan: done."
        )
        assert tors.chunk_hierarchical(thread, 60, ["\n", None]) == [
            (0, 29),
            (30, 78),
            (78, 131),
            (132, 145),
        ]

    def test_minhash_near_dup_gate_example(self) -> None:
        # docs/api.md's minhash_signature section, the near-dup-gate
        # example: the agreement-fraction expression the doc shows, its
        # three pinned outputs (the head elements, the near pair's
        # estimate, the unrelated pair's zero), and the empty-set sentinel
        # row, re-derived against the built extension.
        original = (
            "The quarterly oil sample interval for field outages was adjusted after the "
            "bushing torque specifications changed. Maintenance windows now close within "
            "fourteen days. "
        )
        edited = (
            "The monthly oil sample interval for field outages was adjusted after the "
            "insulator torque specifications changed. Maintenance windows now close within "
            "fourteen days. "
        )
        unrelated = "Pack my box with five dozen liquor jugs."
        a = tors.minhash_signature(original)
        b = tors.minhash_signature(edited)
        u = tors.minhash_signature(unrelated)
        assert a[:3] == [151086443443351341, 59387643775660493, 132191052063639682]
        assert sum(x == y for x, y in zip(a, b, strict=True)) / len(a) == 0.6015625
        assert sum(x == y for x, y in zip(a, u, strict=True)) / len(a) == 0.0
        assert tors.minhash_signature("")[:3] == [
            18446744073709551615,
            18446744073709551615,
            18446744073709551615,
        ]

    def test_unescaped_scan_json_renderings(self) -> None:
        # docs/api.md's contains_unescaped/find_unescaped section, pinned
        # directly (the literals the doc shows, both spellings): the real
        # NUL escape and the literal text differ only in the backslash run
        # before the shared six bytes — offset 2 vs -1, True vs False.
        assert tors.find_unescaped(b'"a\\u0000b"', b"\\u0000") == 2
        assert tors.find_unescaped(b'"a\\\\u0000b"', b"\\u0000") == -1
        assert tors.contains_unescaped(b'"a\\\\u0000b"', b"\\u0000") is False
        assert tors.contains_unescaped(b'"a\\u0000b"', b"\\u0000") is True

    def test_utf8_byte_len_examples(self) -> None:
        # docs/api.md's utf8_byte_len section, pinned directly (the
        # literals the doc shows): the mixed-content example rows and the
        # TaskQ byte-cap gate the section is motivated by, spelled at its
        # 64 KiB MAX_RESULT_BYTES boundary.
        assert tors.utf8_byte_len("caf\u00e9") == 5
        assert tors.utf8_byte_len("\U0001f600") == 4
        serialized_result = "k" * (64 * 1024 + 1)
        assert tors.utf8_byte_len(serialized_result) > 64 * 1024  # the gate trips
        serialized_result = "k" * 64 * 1024
        assert tors.utf8_byte_len(serialized_result) <= 64 * 1024  # the gate passes

    def test_utf16_byte_len_examples(self) -> None:
        # docs/api.md's utf16_byte_len section, pinned directly: the
        # example rows (BMP codepoints, the surrogate pair, the CJK row
        # whose UTF-8 count diverges) and the NVARCHAR(140) column-cap
        # gate the section is motivated by, spelled at its exact 280-byte
        # boundary (140 UTF-16 units).
        assert tors.utf16_byte_len("caf\u00e9") == 8
        assert tors.utf16_byte_len("\U0001f600") == 4
        assert tors.utf16_byte_len("\u6771\u4eac") == 4
        assert tors.utf8_byte_len("\u6771\u4eac") == 6  # the contrast the doc draws
        assert tors.utf16_byte_len("\u6771\u4eac") < tors.utf8_byte_len("\u6771\u4eac")
        s = "k" * 140  # exactly 140 units: the gate passes
        assert tors.utf16_byte_len(s) <= 280
        s = "k" * 140 + "\U0001f600"  # 141 units (the astral codepoint is a
        # PAIR): the gate trips
        assert tors.utf16_byte_len(s) > 280


class TestHashingExamples:
    """docs/api.md's hashing-section examples, pinned the same way: the
    webhook-signature and ETag-check literals the doc shows are re-derived
    here against the built extension."""

    def test_webhook_signature_example(self) -> None:
        import hmac as hmac_module

        secret = "whsec_3f9d2a8c"
        payload = '{"event":"invoice.paid","id":"evt_88213","amount":4200}'
        expected = tors.hmac_sha256_hex(secret, payload)
        assert expected == "43ec3b86ee42fdcf9640ded30e94104b4a11b9fd43f1624b30471f7b13e76a12"
        assert hmac_module.compare_digest(expected, tors.hmac_sha256_hex(secret, payload))

    def test_etag_check_example(self) -> None:
        body = tors.normalize("line one  \n\n\n\nline two\r\n")
        assert body == "line one\n\nline two"
        assert tors.md5_hex(body) == "487f5cc2c45cc57e638d9fce8c33d95c"
        # the declared-ETag comparison the doc shows
        assert tors.md5_hex(body) == "487f5cc2c45cc57e638d9fce8c33d95c"

    def test_webhook_b64_signature_example(self) -> None:
        # the api.md raw-digest subsection's base64-signature webhook
        # example, pinned the same way: the digest spelling,
        # urlsafe-b64-encoded, and the compare_digest verification the doc
        # shows (never ==, in any signature-verification example)
        import base64
        import hmac as hmac_module

        secret = base64.b64decode("whsec_3f9d2a8c".partition("_")[2])
        # the doc spells this as a str + .encode("utf-8"); the literal here
        # is the same bytes
        signed_content = (
            b"msg_5fXn0.1731634200."
            b'{"event":"invoice.paid","id":"evt_88213","amount":4200}'
        )
        signature = base64.urlsafe_b64encode(tors.hmac_sha256_digest(secret, signed_content))
        assert signature == b"q8IyxOXFC_mgdZj-GSqtTl2vj3eTIgMM0HQQ-FfRQVw="
        assert hmac_module.compare_digest(
            signature,
            base64.urlsafe_b64encode(tors.hmac_sha256_digest(secret, signed_content)),
        )

    def test_advisory_lock_int_example(self) -> None:
        # the api.md digest-sliced advisory-lock int, pinned to the doc's
        # literal
        lock_id = int.from_bytes(tors.sha256_digest("tenant:42:resource:7")[:8], "big")
        assert lock_id == 15284293306093710542

    def test_labelled_derivation_chain_example(self) -> None:
        # the api.md labelled derivation chain, pinned to the doc's hex
        # literal
        root = b"root-key-material"
        subkey = tors.hmac_sha256_digest(
            tors.hmac_sha256_digest(root, "tors/db-session-key"), "user:42"
        )
        assert subkey.hex() == (
            "cc3ccddeaa0718afe52e67b9011b49dfe29ae01571495a75e17eea1f59f6e7b6"
        )


class TestContentHashExamples:
    """docs/api.md's ``tors.content_hash`` section: the dedup-key example
    (equal content, any key order, one hash) and the cache-key example,
    their literal hex digests pinned the same way every other docs example
    here is (the README's "the output above is what they actually return"
    discipline)."""

    def test_dedup_key_example(self) -> None:
        a = {"title": "Q3 outage report", "severity": "high", "tags": ["grid", "north"]}
        b = {"tags": ["grid", "north"], "severity": "high", "title": "Q3 outage report"}
        assert tors.content_hash(a) == tors.content_hash(b)
        expected = "558be2127fa557b06ffd3dd0735697e14022546c969c6c37b01cf6278a178cf2"
        assert tors.content_hash(a) == expected

    def test_cache_key_example(self) -> None:
        request = {
            "model": "guss-9",
            "messages": [{"role": "user", "text": "Summarize the Q3 report."}],
        }
        expected = "b0df18f8e089f15fb56fa51c241151213e67b82e7b656de29b1bafedb3785464"
        assert tors.content_hash(request) == expected


class TestScrubPiiExamples:
    """docs/api.md's scrub_pii section, pinned the same way: the two
    worked examples' literal outputs (default rules, default salt — the
    digests are deterministic, so a behavior change that would turn the
    doc into a lie fails here first), plus the two re-fire-corner
    inputs' convergence the section's idempotence paragraph describes."""

    def test_rejection_excerpt_example(self) -> None:
        assert tors.scrub_pii(
            "unknown candidate fungai.chetima@example.com called from +14155552671 twice"
        ) == ("unknown candidate @example.com~3aa8d1d0bb1a called from +14~115f5ee5ea90 twice")

    def test_domestic_number_with_spare_digit_run_example(self) -> None:
        # The domestic spelling's token prefix is "+1 " (the third code
        # point is the space), and the bare digit run survives the "+"
        # anchoring.
        assert tors.scrub_pii("ring +1 (415) 555-2671 about ticket 4096") == (
            "ring +1 ~dc750721a848 about ticket 4096"
        )

    def test_the_refire_corner_inputs_converge_as_documented(self) -> None:
        for text in ("a@b.co9@x.yz", "x@b.co@w.vu"):
            once = tors.scrub_pii(text)
            twice = tors.scrub_pii(once)
            assert tors.scrub_pii(twice) is twice


class TestTranscriptRecipeExamples:
    def test_section_1_thread_spliced_hierarchy(self) -> None:
        thread = (
            "Nathan: kicking off the sync.\n"
            "Priya: We briefed the U.S. team on the numbers. "
            "They asked for a follow-up meeting. The budget holds.\n"
            "Nathan: done."
        )
        chunks = tors.chunk_hierarchical(thread, 60, ["\n", None])
        assert chunks == [(0, 29), (30, 78), (78, 131), (132, 145)]
        assert [thread[s:e] for s, e in chunks] == [
            "Nathan: kicking off the sync.",
            # trailing space, exactly as the recipe shows it: the sentence
            # segmenter attaches it to the preceding sentence
            "Priya: We briefed the U.S. team on the numbers. ",
            "They asked for a follow-up meeting. The budget holds.",
            "Nathan: done.",
        ]
        # the doc's abbreviation contrast: the naive ". " literal list
        # severs "U.S" at (30, 55); the splice does not
        assert tors.chunk_hierarchical(thread, 40, ["\n", ". ", " "])[1] == (30, 55)
        assert thread[30:55] == "Priya: We briefed the U.S"
        assert tors.chunk_hierarchical(thread, 40, ["\n", None])[1] == (30, 69)
        assert thread[30:69] == "Priya: We briefed the U.S. team on the "
        # the count-based twin: two content lines per window
        assert tors.chunk_by_lines(thread, 2) == [(0, 131), (132, 145)]

    def test_section_2_webvtt_cues(self) -> None:
        # the doc's triple-quoted block, spelled as implicit concatenation
        # (the suite's own long-literal idiom) so no line exceeds the lint
        # gate; the string value is byte-identical to the doc's
        vtt = (
            "WEBVTT\n"
            "\n"
            "00:00:01.000 --> 00:00:04.000\n"
            "<v Nathan>Welcome to the weekly sync.\n"
            "\n"
            "00:00:04.000 --> 00:00:09.500\n"
            "<v Priya>Thanks. We briefed the U.S. team on the numbers yesterday, and "
            "they asked for a follow-up meeting. The budget holds.\n"
            "\n"
            "00:00:09.500 --> 00:00:12.000\n"
            "<v Nathan>Great. Let's aim for Thursday then."
        )
        assert tors.chunk_hierarchical(vtt, 120) == [(0, 75), (77, 124), (124, 232), (234, 309)]

    def test_section_3_srt_cues(self) -> None:
        srt = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "Welcome to the weekly sync.\n"
            "\n"
            "2\n"
            "00:00:04,000 --> 00:00:09,500\n"
            "Thanks. We briefed the U.S. team on the numbers yesterday, and they "
            "asked for a follow-up meeting. The budget holds."
        )
        assert tors.chunk_hierarchical(srt, 120) == [(0, 59), (61, 101), (101, 209)]

    def test_section_4_sentence_bounds_on_one_cue(self) -> None:
        cue = (
            "Thanks. We briefed the U.S. team on the numbers yesterday, and "
            "they asked for a follow-up meeting. The budget holds."
        )
        assert tors.sentence_bounds(cue) == [(0, 8), (8, 99), (99, 116)]


class TestRandomGenerationExamples:
    """docs/api.md's random-generation section, pinned: the seeded examples
    are deterministic, so their literals are exact pins; the unseeded
    example (a fresh OS draw every call) and the uuid7 timestamp
    round-trip are pinned by shape, exactly the split the doc's own
    comments state. If one of these fails after an intentional change, the
    doc and this pin move together, in the same commit."""

    def test_family_intro_examples(self) -> None:
        # "len(tors.random_hex(32)) / # 32": the unseeded spelling's shape —
        # 32 asks for 32 characters now, the length-first contract.
        assert len(tors.random_hex(32)) == 32
        # The deterministic spelling's literal (the 32-char hex-key example).
        assert tors.random_hex(32, seed=42) == "861225d7151bf9b14a3617ab9b534d19"

    def test_random_string_example(self) -> None:
        assert tors.random_string(12, "abcdef", seed=42) == "dcabacecacae"

    def test_random_b62_example(self) -> None:
        assert tors.random_b62(22, seed=0) == "1yrBtE6FUlG59Zjj3K2vVn"

    def test_random_b64url_example(self) -> None:
        # The 43-char token (the JWT-sig shape): one length parameter, no
        # padded= spelling — the example moved with the contract.
        assert tors.random_b64url(43, seed=7) == "Bi8SLaf0s_a4pi-vqthbTaOstZjDweDcEC5hW7S_CNp"

    def test_uuid4_example(self) -> None:
        assert tors.uuid4(seed=42) == "7848b5d7-11bc-4883-9963-17a3f9c90269"

    def test_uuid7_timestamp_round_trip_example(self) -> None:
        # The doc's caller-visible contract: the canonical string's first
        # two dash-free groups decode to the call's Unix epoch
        # milliseconds. Shape-pinned (the draw and the clock are both
        # live), the same ±5s window tests/test_random.py asserts.
        before = time.time() * 1000
        value = tors.uuid7()
        after = time.time() * 1000
        assert len(value) == 36
        assert value[14] == "7"
        ts = int(value[:8] + value[9:13], 16)
        assert before - 5_000 <= ts <= after + 5_000

    def test_uuid4_bytes_bytes_uuid4_bytes_example(self) -> None:
        # The bytes spellings' contract line, pinned: uuid4_bytes(seed=s)
        # is exactly the 16 bytes whose canonical hyphenated form is
        # uuid4(seed=s) — the doc's "one construction, two return
        # spellings" claim, at the doc's own example seed.
        raw = tors.uuid4_bytes(seed=42)
        assert len(raw) == 16
        assert stdlib_uuid.UUID(bytes=raw) == stdlib_uuid.UUID(tors.uuid4(seed=42))

    def test_uuid7_bytes_consumer_shape_examples(self) -> None:
        # The doc's two consumer shapes, pinned: the stdlib constructor
        # over the raw bytes carries version and variant through, and the
        # .hex()[:12] slice is the timestamp prefix (the u[:8] + u[9:13]
        # field's own bytes), both within the ±5s clock window.
        u = stdlib_uuid.UUID(bytes=tors.uuid7_bytes())
        assert (u.version, u.variant) == (7, stdlib_uuid.RFC_4122)
        before = time.time() * 1000
        prefix = tors.uuid7_bytes().hex()[:12]
        after = time.time() * 1000
        assert len(prefix) == 12
        assert before - 5_000 <= int(prefix, 16) <= after + 5_000
