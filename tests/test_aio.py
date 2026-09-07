"""The gate for ``tors.aio``: the awaitable spellings of tors's curated
large-input functions (see ``tors/aio.py``'s module docstring for the
exact list and why the rest of ``tors`` intentionally has no async
twin).

What this gate pins:

- COVERAGE: the facade exports exactly ``tors.aio._WRAPPED``, no more, no
  less. A function added to the curated set fails this gate until the
  facade and the generated stub pick it up.
- SIGNATURE PARITY: each async wrapper accepts exactly what its sync
  spelling accepts, parameter names in order, kinds, and defaults
  (``inspect`` against the live function, the ``test_pyi_drift.py``
  method applied across the facade).
- STUB FRESHNESS: ``aio.pyi`` is the exact output of
  ``tools/gen_aio_stub.py`` over the current ``__init__.pyi`` (a stale
  stub fails loudly; regenerate with the tool as part of the change).
- NO HIDDEN BRANCH: every wrapper is a coroutine function that
  unconditionally awaits ``asyncio.to_thread`` — there is no code path
  where a wrapper runs its work inline on the caller's own turn, which
  would make it lie about being async.
- AWAIT CORRECTNESS: results equal the sync spellings, keyword-only
  parameters pass through, and the event loop stays responsive during a
  whole-corpus call: a heartbeat coroutine keeps ticking with worst gaps
  well under the call's own wall while a large ``diff_opcodes`` await
  runs in the worker thread. That is the facade's whole reason to exist,
  asserted directly — the exact property whose absence (an async wrapper
  that silently blocks the loop) is a real, previously-shipped bug in at
  least one other GIL-releasing native-extension library's own async
  wrapper.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path

import pytest

import tors
import tors.aio
from reference import prose

_PYI = Path(__file__).resolve().parent.parent / "python" / "tors" / "aio.pyi"


class TestCoverage:
    def test_the_facade_covers_exactly_the_curated_wrapped_set(self) -> None:
        expected = set(tors.aio._WRAPPED)  # noqa: SLF001
        assert set(tors.aio.__all__) == expected
        assert set(vars(tors.aio)) >= expected
        # Nothing outside the curated set gets an async twin, including the
        # iterator constructors and every small/cheap function.
        for name in tors.__all__:
            if name not in expected:
                assert not hasattr(tors.aio, name), name

    def test_every_wrapper_is_a_coroutine_function(self) -> None:
        for name in tors.aio.__all__:
            assert asyncio.iscoroutinefunction(getattr(tors.aio, name)), name

    def test_no_wrapper_ever_runs_inline_regardless_of_input_size(self) -> None:
        """The design requirement, asserted structurally: the wrapper's
        body is exactly one unconditional ``await asyncio.to_thread(fn,
        ...)``, no branch on argument size or shape anywhere in it — the
        facade never silently decides to run inline."""
        source = inspect.getsource(tors.aio._make_async)  # noqa: SLF001
        wrapper_body = source.split("async def wrapper")[1].split("wrapper.__qualname__")[0]
        assert "asyncio.to_thread" in wrapper_body
        assert "if " not in wrapper_body


class TestSignatureParity:
    def test_each_wrapper_accepts_exactly_what_the_sync_spelling_accepts(self) -> None:
        drift = []
        for name in tors.aio.__all__:
            async_params = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(getattr(tors.aio, name)).parameters.values()
                if p.name not in ("args", "kwargs")
            ]
            sync_params = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(getattr(tors, name)).parameters.values()
            ]
            if async_params != sync_params:
                drift.append((name, sync_params, async_params))
        assert not drift, f"signature drift: {drift[:3]}"


class TestStubFreshness:
    def test_the_committed_stub_is_the_generator_output(self) -> None:
        import ast

        import gen_aio_stub  # type: ignore[import-not-found]  # tools/ on sys.path below

        source = (
            Path(__file__).resolve().parent.parent / "python" / "tors" / "__init__.pyi"
        ).read_text(encoding="utf-8")
        expected, _count = gen_aio_stub._translate(source, frozenset(tors.aio._WRAPPED))  # noqa: SLF001
        assert _PYI.read_text(encoding="utf-8") == expected, (
            "python/tors/aio.pyi is stale: run "
            "`uv run --no-sync python tools/gen_aio_stub.py` and commit it with "
            "the __init__.pyi change"
        )
        # And the stub covers exactly the facade.
        stubbed = {
            node.name
            for node in ast.parse(_PYI.read_text(encoding="utf-8")).body
            if isinstance(node, ast.AsyncFunctionDef)
        }
        assert stubbed == set(tors.aio.__all__)

    def test_no_stub_signature_uses_the_wrong_awaitable_wrapper(self) -> None:
        """An ``async def`` function's return annotation is what awaiting
        it resolves to, never ``Awaitable[T]`` — that describes the
        coroutine object itself. A regression here would silently mistype
        every caller of the facade. Checked against the actual parsed
        signatures, not a substring scan of the file (the module
        docstring legitimately discusses the term in prose)."""
        import ast

        tree = ast.parse(_PYI.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.AsyncFunctionDef) or node.returns is None:
                continue
            returns_source = ast.unparse(node.returns)
            assert "Awaitable" not in returns_source, (
                f"{node.name}'s return annotation is {returns_source!r}"
            )


class TestAwaitCorrectness:
    @pytest.mark.parametrize(
        ("name", "args", "kwargs"),
        [
            ("diff_opcodes", ("qabxcd", "abYcd"), {}),
            ("diff_opcodes_lines", ("a\nb\nc\n", "a\nB\nc\n"), {}),
            ("chunk_text", ("cats are cute and cats are fun", 12), {}),
            ("chunk_by_words", ("one two three four five six", 2), {}),
            ("chunk_by_sentences", ("One. Two. Three.", 1), {}),
            ("chunk_by_paragraphs", ("a\n\nb\n\nc", 1), {}),
            ("chunk_cdc", (b"x" * 20_000,), {}),
            ("chunk_hierarchical", ("One. Two. Three. Four.", 8), {}),
            ("tf_idf", (["the cat sat", "the dog ran"],), {}),
            ("bm25_rank", ("cat", ["the cat sat", "the dog ran"]), {}),
            ("apply_pipeline", (["Café", "MUSEUM"],), {"lowercase": True}),
        ],
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_awaiting_equals_the_sync_spelling(
        self, name: str, args: tuple, kwargs: dict
    ) -> None:
        async def run() -> object:
            return await getattr(tors.aio, name)(*args, **kwargs)

        expected = getattr(tors, name)(*args, **kwargs)
        assert asyncio.run(run()) == expected

    def test_keyword_only_parameters_pass_through(self) -> None:
        async def run() -> object:
            return await tors.aio.diff_opcodes(
                "abcabcabcXdefdefdef",
                "abcabcabcYdefdefdef",
                deadline_ms=60_000.0,
            )

        assert asyncio.run(run()) == tors.diff_opcodes(
            "abcabcabcXdefdefdef", "abcabcabcYdefdefdef", deadline_ms=60_000.0
        )

    def test_the_event_loop_stays_responsive_during_a_whole_corpus_call(self) -> None:
        """The facade's reason to exist, asserted directly: while a large
        ``diff_opcodes`` call runs in the worker thread (a call whose sync
        spelling takes tens of milliseconds or more), a 5 ms heartbeat
        keeps ticking with worst gaps well under the call's own wall. A
        wrapper that silently ran the native pass inline (the "lying"
        failure mode this facade's design deliberately avoids) would
        starve the heartbeat for the whole wall instead."""

        async def run() -> tuple[float, float, int]:
            # 16 MiB, not 2 MiB: the 2 MiB corpus measured ~4.6ms wall on a
            # loaded CI runner, right at the razor's edge of the "must cost
            # something" floor below and prone to landing under it on a
            # faster or differently-loaded box — exactly what happened here.
            # 16 MiB gives real margin toward the docstring's own "tens of
            # milliseconds or more" claim rather than a threshold this test
            # can trip on jitter alone.
            a = prose(16 * 1024 * 1024)
            b = a[: len(a) // 2] + "X" + a[len(a) // 2 :]
            ticks: list[float] = []

            async def heartbeat() -> None:
                last = time.perf_counter()
                while True:
                    await asyncio.sleep(0.005)
                    now = time.perf_counter()
                    ticks.append(now - last)
                    last = now

            hb = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.02)  # the heartbeat establishes cadence
            started = time.perf_counter()
            opcodes = await tors.aio.diff_opcodes(a, b)
            wall = time.perf_counter() - started
            await asyncio.sleep(0.02)
            hb.cancel()
            assert len(opcodes) >= 1
            return wall, max(ticks), len(ticks)

        wall, worst_gap, tick_count = asyncio.run(run())
        assert wall > 0.010, "the corpus must actually cost something"
        assert tick_count >= 4, "the heartbeat must have ticked during the call"
        assert worst_gap < 0.015, (
            f"worst heartbeat gap {worst_gap * 1000:.1f} ms during a "
            f"{wall * 1000:.0f} ms call: the native pass is blocking the loop"
        )
