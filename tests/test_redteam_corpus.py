"""The red-team regression canary: one test loads
tests/redteam_corpus/manifest.toml and asserts every entry's invariant
over its minimal repro payload — one canary per reopened/verified defect
(#91 #92 #99 #100 #101 #102 #103 #108 #110 #111 #112 #113 #114 #115)
plus the harness's own new findings.

Status semantics (see the manifest's header):
* ``fixed-on-main`` — the invariant must keep holding; a failure here is
  a REGRESSION and fails hard.
* ``live`` — the invariant is violated on main (the defect is open or
  its fix is on another branch); a violation is turned into a documented
  xfail NAMING the issue. When the fix lands the entry starts passing
  (an XPASS is visible in the summary); flip its status to
  ``fixed-on-main`` in the same change so the canary hardens again.

The live entries are the harness's contract pins for the in-flight
bounded-DOS branches: each one's invariant is written against the
DOCUMENTED contract (a catchable ValueError past a ceiling, a deadline
respected, a peak tied to input bytes), so the day the fix lands, the
canary goes green without any edit — and hardens to regression duty when
its status flips.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

import tors

CORPUS = Path(__file__).parent / "redteam_corpus"


def _load_manifest() -> dict | None:
    # tomllib is 3.11+; on 3.10 fall back to tomli when present (the
    # test_random.py Cargo-gate pattern) — with neither, the canary skips
    # rather than silently passing.
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # Python 3.10: no stdlib tomllib.
        try:
            import tomli as _toml  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    return _toml.loads((CORPUS / "manifest.toml").read_text(encoding="utf-8"))


_manifest = _load_manifest()
pytestmark = pytest.mark.skipif(
    _manifest is None, reason="no TOML parser available (Python 3.10 without tomli)"
)


def _wall_s(fn) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def _canary(case: dict, check) -> None:
    """Run one canary's invariant check, folding a live-status violation
    into a documented xfail and a fixed-status violation into a hard
    regression failure."""
    try:
        check()
    except AssertionError as exc:
        if case["status"] == "live":
            pytest.xfail(f"{case['id']} is live (tracked #{case['issue']}) on this tree: {exc}")
        raise# --- per-surface invariant checks ---------------------------------------------------


def _check_scrub_pii(case: dict) -> None:
    text = (CORPUS / case["file"]).read_text(encoding="utf-8")
    params = case["params"]
    out = tors.scrub_pii(text)
    limit = params.get("wall_limit_s")
    if limit is not None:
        wall = _wall_s(lambda: tors.scrub_pii(text))
        assert wall < limit, f"the scrub took {wall:.2f}s (limit {limit}s)"
    for needle in params.get("needles", []):
        assert needle not in out, f"the credential survived the scrub: {needle[:30]!r}"


def _check_scrub_pii_multi(case: dict) -> None:
    params = case["params"]
    for name, needle in zip(params["files"], params["needles"], strict=True):
        text = (CORPUS / name).read_text(encoding="utf-8")
        out = tors.scrub_pii(text)
        assert needle not in out, (
            f"the credential survived the scrub: {needle[:40]!r} ({name})"
        )


def _check_minhash(case: dict) -> None:
    text = (CORPUS / case["file"]).read_text(encoding="utf-8")
    params = case["params"]
    with pytest.raises(ValueError) as ei:
        tors.minhash_signature(
            text, num_perm=params["num_perm"], shingle_size=params["shingle_size"]
        )
    assert "shingle" in str(ei.value).lower(), f"the refusal did not name the knob: {ei.value}"
    wall = _wall_s(
        lambda: tors.minhash_signature(
            text, num_perm=params["num_perm"], shingle_size=params["under_ceiling_shingle_size"]
        )
    )
    assert wall < 5.0, f"the under-ceiling call took {wall:.2f}s"


def _check_masked(case: dict) -> None:
    text = (CORPUS / case["file"]).read_text(encoding="utf-8")
    params = case["params"]
    value = params["value_char"] * params["value_repeat"]
    out = tors.replace_many_masked(text, {params["key"]: value}, params["mask"])
    assert len(out) == len(text), (
        f"the masked output is not length-preserving: {len(out)} vs {len(text)}"
    )
    wall = _wall_s(
        lambda: tors.replace_many_masked(text, {params["key"]: value}, params["mask"])
    )
    assert wall < params["wall_limit_s"], f"took {wall:.2f}s (limit {params['wall_limit_s']}s)"


def _check_chunk_text_overlap(case: dict) -> None:
    text = (CORPUS / case["file"]).read_text(encoding="utf-8")
    limit = case["params"]["wall_limit_s"]
    n = len(text)
    m = n // 2
    wall = _wall_s(lambda: tors.chunk_text(text, m, overlap=m - 1))
    assert wall < limit, f"took {wall:.2f}s (limit {limit}s)"
    chunks = tors.chunk_text(text, m, overlap=m - 1)
    assert len(chunks) == n - m + 1


def _check_chunk_hierarchical(case: dict) -> None:
    text = (CORPUS / case["file"]).read_text(encoding="utf-8")
    params = case["params"]
    chunks = tors.chunk_hierarchical(
        text, params["max_chars"], separators=params["separators"], overlap=params["overlap"]
    )
    for start, end in chunks:
        for sep in params["separators"]:
            assert text[start:end] != sep, (
                f"the chunk ({start}, {end}) is the separator {sep!r} itself"
            )


def _check_documents_to_text(case: dict) -> None:
    import tors.documents as documents

    data = (CORPUS / case["file"]).read_bytes()
    fmt = case["params"]["format"]
    limit = case["params"]["wall_limit_s"]
    wall = _wall_s(lambda: documents.to_text(data=data, format=fmt))
    assert wall < limit, f"to_text took {wall:.2f}s (limit {limit}s)"


class _LyingSequence(Sequence):
    """A Sequence whose __len__ lies by an attacker-chosen amount: pyo3
    sizes its Vec from the lie before iterating (#112)."""

    def __init__(self, claimed_len: int) -> None:
        self._claimed = claimed_len

    def __len__(self) -> int:
        return self._claimed

    def __getitem__(self, i: int):
        raise IndexError


def _check_lying_len(case: dict) -> None:
    claimed = case["params"]["claimed_len"]
    liars = {
        "scrub_pii_rules": lambda: tors.scrub_pii("x", rules=_LyingSequence(claimed)),
        "chunk_hierarchical_separators": lambda: tors.chunk_hierarchical(
            "a b", 2, separators=_LyingSequence(claimed)
        ),
    }
    for name, call in liars.items():
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 — the panic class IS the finding
            assert isinstance(exc, Exception), (
                f"{name}: the refusal is uncatchable ({type(exc).__module__}."
                f"{type(exc).__name__}); a lying __len__ must produce a catchable "
                "ValueError, not a PanicException"
            )


# The child-probe discipline (shared with tests/test_memory_spike_guards.py):
# a shape that can hang, balloon, or signal-die never runs in the pytest
# process.
from test_memory_spike_guards import _MIB, _run_child  # noqa: E402


def _check_shared_ref_schema(case: dict) -> None:
    doc = (CORPUS / case["file"]).read_text(encoding="utf-8")
    doublings = case["params"]["doublings"]
    setup = f"x = None\nfor _ in range({doublings}):\n    x = [x, x]\n"
    kind, message, peak_kib, done = _run_child(
        f"output = tors.repair_json_loads({doc!r}, schema={{'enum': x}})",
        setup=setup,
        belt=2 * 1024 * _MIB,
        timeout=case["params"]["timeout_s"],
    )
    assert kind not in ("TIMEOUT", "DEAD"), (
        f"the shared-ref schema walk did not terminate within the belt (child outcome "
        f"{kind}): the walk has no total-node cap"
    )
    assert kind == "ValueError", f"expected a catchable walk-ceiling refusal, got {kind}: {message}"
    assert peak_kib < 512 * 1024, f"the refusal arrived after expanding the graph ({peak_kib} kB)"


def _check_replace_many_amplify(case: dict) -> None:
    text = (CORPUS / case["file"]).read_text(encoding="utf-8")
    params = case["params"]
    value = params["value_char"] * params["value_repeat"]
    input_bytes = len(text) + len(value)
    ceiling_kib = (params["peak_ceiling_x_input"] * input_bytes) // 1024 + 64 * 1024
    kind, message, peak_kib, done = _run_child(
        f"output = tors.replace_many({text!r}, {{'a': {value!r}}})",
        timeout=30.0,
    )
    assert kind not in ("TIMEOUT", "DEAD"), f"the amplification child hung or died: {kind}"
    assert kind == "ValueError", (
        f"expected the documented output-ceiling refusal, got {kind}: {message}"
    )
    assert peak_kib < ceiling_kib, (
        f"peak {peak_kib / 1024:.0f} MiB exceeds the input-tied ceiling "
        f"{ceiling_kib / 1024:.0f} MiB: the output scaled with the amplification"
    )


def _check_deadline_enum(case: dict) -> None:
    params = case["params"]
    doc = (CORPUS / case["file"]).read_text(encoding="utf-8")
    members = [("a" * params["member_len"]) + str(i) for i in range(params["member_count"])]
    started = time.perf_counter()
    with pytest.raises((TimeoutError, ValueError)):
        tors.repair_json_loads(doc, schema={"enum": members}, deadline_ms=params["deadline_ms"])
    elapsed = time.perf_counter() - started
    budget_s = params["deadline_ms"] / 1000.0
    assert elapsed < params["overshoot_limit_x"] * budget_s, (
        f"the deadline escape: {elapsed * 1000:.0f}ms elapsed against a "
        f"{params['deadline_ms']}ms budget ({elapsed / budget_s:.0f}x overshoot)"
    )


def _check_wide_schema(case: dict) -> None:
    doc = (CORPUS / case["file"]).read_text(encoding="utf-8")
    params = case["params"]
    schema = {
        "type": "object",
        "properties": {f"k{i}": {"type": "string"} for i in range(params["properties"])},
    }
    wall = _wall_s(lambda: tors.repair_json_loads(doc, schema=schema))
    assert wall < params["wall_limit_s"], (
        f"the schema-aware repair took {wall * 1000:.0f}ms (limit "
        f"{params['wall_limit_s'] * 1000:.0f}ms) at {params['properties']} properties: "
        "the walk is superlinear in the property count"
    )


_CHECKS = {
    "scrub_pii": _check_scrub_pii,
    "scrub_pii_multi": _check_scrub_pii_multi,
    "minhash_signature": _check_minhash,
    "replace_many_masked": _check_masked,
    "chunk_text_overlap": _check_chunk_text_overlap,
    "chunk_hierarchical": _check_chunk_hierarchical,
    "documents_to_text": _check_documents_to_text,
    "lying_len": _check_lying_len,
    "shared_ref_schema": _check_shared_ref_schema,
    "replace_many_amplify": _check_replace_many_amplify,
    "deadline_enum": _check_deadline_enum,
    "wide_schema": _check_wide_schema,
}


_CANARY_CASES = [] if _manifest is None else [
    case for case in _manifest["cases"] if case["surface"] != "utf8_byte_len_starve"
]


@pytest.mark.parametrize(
    "case",
    _CANARY_CASES,
    ids=[case["id"] for case in _CANARY_CASES],
)
def test_redteam_canary(case: dict) -> None:
    _canary(case, lambda: _CHECKS[case["surface"]](case))


# --- the loop-responsiveness lane (#108), its own timing cell -----------------------


@pytest.mark.timing
def test_canary_issue_108_fresh_string_starve() -> None:
    """#108's canary: a worker thread looping ``utf8_byte_len`` over
    FRESH (uncacheable) large non-ASCII slices must leave a co-resident
    heartbeat at heartbeat granularity. The live starvation measures a
    ~2000ms max gap (the heartbeat never re-ticks); the healthy band is
    tens of ms. Runs only when the corpus carries the entry; xfail-folds
    on a live tree via the shared _canary discipline."""
    cases = {c["id"]: c for c in (_manifest or {}).get("cases", [])}
    case = cases.get("issue-108-fresh-string-gil-starve")
    if case is None:
        pytest.skip("the #108 canary entry is not in the manifest")

    text = (CORPUS / case["file"]).read_text(encoding="utf-8")
    params = case["params"]
    deadline = time.monotonic() + params["seconds"]
    gaps: list[float] = []
    stop = threading.Event()

    def heartbeat() -> None:
        last = time.monotonic()
        while not stop.is_set():
            time.sleep(params["heartbeat_ms"] / 1000.0)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    def worker() -> None:
        while time.monotonic() < deadline:
            tors.utf8_byte_len(text[1:])  # a fresh slice: the cache never hits

    beat = threading.Thread(target=heartbeat)
    work = threading.Thread(target=worker)
    beat.start()
    work.start()
    work.join()
    stop.set()
    beat.join()
    worst_gap_ms = max(gaps) * 1000.0

    def check() -> None:
        assert worst_gap_ms < params["max_gap_ms"], (
            f"the heartbeat's worst gap was {worst_gap_ms:.0f}ms (limit "
            f"{params['max_gap_ms']}ms): the fresh-string lane starved the "
            "co-resident loop"
        )

    _canary(case, check)
