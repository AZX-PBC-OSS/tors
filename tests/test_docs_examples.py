"""The docs' worked examples for the #28 features, pinned: every output
literal README.md, docs/api.md, and docs/recipe-transcripts.md show for
``chunk_by_lines`` and the ``None``-entry hierarchy splice is re-derived
here against the built extension, the
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
