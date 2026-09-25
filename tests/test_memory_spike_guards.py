"""Peak-memory guards for the amplification shapes: a small adversarial
input whose OUTPUT (or walk) is a large multiple of the input's size.
These pin the memory class of #114 (a 40GB string built from a 1KB
input), #113 (a shared-ref schema walk exploding to 2^n nodes), and the
PEM-flood lane's peak — the harness that catches the NEXT unbounded
amplification, wherever its ceiling lands.

DESIGN — why /proc VmHWM and not tracemalloc: the amplification happens
in Rust (a 40GB `String` allocation, a 2^n node walk over Rust values);
`tracemalloc` sees only the Python heap and would report a quiet floor
while the process balloons. The codebase's Rust-side memory oracle is
the disposable-child discipline (tests/test_rt_sameclass.py, the #86
lane): run the amplification in a subprocess, read VmHWM (peak RSS) from
the child's own /proc/self/status, assert the peak in the parent. The
children are disposable by design: the shapes under test must never
balloon the pytest process. Where the shape can hang or signal-die, the
child additionally runs under an RLIMIT_AS belt, the same belt
test_rt_sameclass.py's probes use.

CONTRACT PINNING — these guards assert the SAFE behavior: a catchable
ValueError past the documented ceiling (the #91/#114 fix shape:
"ValueError past a documented ceiling, allocated nothing") AND a peak
tied to input bytes, not to the amplification factor. The fixes land on
other branches, so each guard checks the live behavior first and
`pytest.skip`s — naming the live defect and the peak it measured — when
it observes the unsafe signature, rather than failing the suite on
main. When the bounded-DOS branch lands, the skip arm goes dead and the
asserts below become the contract pins. (The empirical gate beats a
version check: the guards flip to asserting the moment the behavior
exists, on whatever branch runs them.)

Where the ValueError contract tests slot in once fixed: the ceilings'
own surfaces — tests/test_replace_many.py (output-size ceiling),
tests/test_json_repair_schema_hostility.py (the schema walk's node cap),
tests/test_scrub_pii.py — own those pins; the corpus canary
(tests/redteam_corpus/ + tests/test_redteam_corpus.py) carries the same
shapes as minimal repros.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_MIB = 1024 * 1024
_KILL_SIGNALS = (-9, -6, 137, 134)

# The child template: run one amplification call, report the outcome and
# the child's own peak RSS as one parseable line. VmHWM is read AFTER the
# call: peak-RSS-since-start, which is exactly the quantity the ceiling
# bounds. The RLIMIT_AS belt (set when the env var says so) keeps a live
# amplification from pressuring the host instead of dying cleanly.
_CHILD_TEMPLATE = """\
import resource, sys
belt = int(sys.argv[1])
if belt:
    resource.setrlimit(resource.RLIMIT_AS, (belt, belt))
import tors
{setup}output = None
try:
    {call}
    kind = "OK"
    message = ""
except BaseException as exc:
    kind = type(exc).__name__
    message = str(exc)[:200]
hwm = 0
with open("/proc/self/status") as status:
    for line in status:
        if line.startswith("VmHWM:"):
            hwm = int(line.split()[1])  # kB
print(f"RESULT|{kind}|{message}|{hwm}")
"""


def _run_child(call: str, setup: str = "", *, belt: int = 0, timeout: float = 60.0):
    """Run one amplification child; return (kind, message, peak_kib, done).
    A child that dies (belt abort, signal) reports kind="DEAD" with the
    return code; a child that hangs reports kind="TIMEOUT"."""
    code = _CHILD_TEMPLATE.replace("{setup}", setup).replace("{call}", call)
    try:
        done = subprocess.run(
            [sys.executable, "-c", code, str(belt)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", 0, None
    found = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT|")]
    if not found:
        return "DEAD", done.stderr[-300:], 0, done
    parts = found[-1].split("|", 3)
    return parts[1], parts[2], int(parts[3]), done


def _assert_alive(kind: str, message: str, done) -> None:
    assert kind not in ("DEAD",), (
        f"the amplification child died by signal/abort rc="
        f"{getattr(done, 'returncode', '?')}: the shape is not catchable: {message}"
    )


# --- #114-class: replace_many's unbounded output -----------------------------------
#
# The documented contract (once the bounded-output branch lands): an
# output size past the ceiling raises ValueError BEFORE allocating (with
# try_reserve so a sub-ceiling allocator refusal is also catchable). The
# guard's amplification: 10,000 matches x a 100KB value = ~1GB of output
# from ~110KB of input — a ~9,000x factor. The ceiling is tied to the
# INPUT (100x input bytes + the interpreter's own baseline), not to the
# amplification factor, so an unbounded build (1GB) trips it by ~90x
# while any ceiling-respecting behavior passes.
#
# NOTE: this call SUCCEEDS on main today (measured: the 40GB repro builds
# in ~14s); the guard self-skips there — see the module docstring.


class TestReplaceManyOutputPeakBounded:
    def test_short_key_huge_value_peak_is_tied_to_input_not_amplification(self) -> None:
        text = "a" * 10_000
        value = "y" * 100_000
        input_bytes = len(text) + len(value) + 8
        ceiling_kib = (100 * input_bytes) // 1024 + 64 * 1024  # + 64 MiB baseline slack
        kind, message, peak_kib, done = _run_child(
            f"output = tors.replace_many({text!r}, {{'a': {value!r}}})",
            timeout=60.0,
        )
        _assert_alive(kind, message, done)
        if kind == "OK":
            pytest.skip(
                "#114 is live on this tree: replace_many built the unbounded output "
                f"(peak {peak_kib / 1024:.0f} MiB, ~{peak_kib * 1024 // input_bytes}x the "
                "input bytes). These asserts pin the bounded-output contract once the "
                "fix lands (ValueError past the documented ceiling, peak tied to input)."
            )
        assert kind == "ValueError", (
            f"expected the documented ceiling refusal, got {kind}: {message}"
        )
        assert peak_kib < ceiling_kib, (
            f"peak {peak_kib / 1024:.0f} MiB exceeds the input-tied ceiling "
            f"{ceiling_kib / 1024:.0f} MiB: the refusal arrived after the allocation"
        )
        # The amplification bound: ~9000x the input would be the
        # unbounded output's size. Units: peak_kib is KiB, input_bytes is
        # bytes — compare in bytes (the former spelling divided by _MIB
        # and compared against KiB, off by 1024).
        assert peak_kib * 1024 < 9000 * input_bytes, (
            "the peak scaled with the amplification factor, not the input"
        )

    def test_compiled_patterns_twin_shares_the_ceiling(self) -> None:
        """The CompiledPatterns.replace_many twin (#114 names it too): the
        same input shape, the same contract. Skips alongside the surface
        twin while #114 is live."""
        text = "a" * 10_000
        value = "y" * 100_000
        kind, message, peak_kib, done = _run_child(
            f"output = tors.CompiledPatterns(['a']).replace_many({text!r}, {{'a': {value!r}}})",
            timeout=60.0,
        )
        _assert_alive(kind, message, done)
        if kind == "OK":
            pytest.skip(
                "#114 is live on this tree (CompiledPatterns.replace_many built the "
                f"unbounded output, peak {peak_kib / 1024:.0f} MiB): pins the contract "
                "once the fix lands."
            )
        assert kind == "ValueError", message


# --- #113-class: repair_json's shared-ref schema walk -------------------------------
#
# The documented contract (once the node-cap branch lands): the schema
# walk raises ValueError past a total-node ceiling (the content_hash
# MAX_WALK_NODES precedent) BEFORE expanding the graph. The structure is
# tiny (26 doubling levels of shared refs); the WALK is the bomb —
# 2^26 node visits — so on the fixed branch the refusal is immediate and
# the peak stays at interpreter baseline, while on main the child either
# walks for minutes or dies on the 2 GiB belt.


class TestRepairJsonSharedRefSchemaPeakBounded:
    def test_shared_ref_schema_terminates_with_a_catchable_error(self) -> None:
        setup = (
            "x = None\n"
            "for _ in range(26):\n"
            "    x = [x, x]\n"
        )
        kind, message, peak_kib, done = _run_child(
            'output = tors.repair_json_loads("{}", schema={"enum": x})',
            setup=setup,
            belt=2 * 1024 * _MIB,
            timeout=20.0,
        )
        if kind in ("TIMEOUT", "DEAD"):
            pytest.skip(
                "#113 is live on this tree: the shared-ref schema walk did not terminate "
                f"within the guard's 20s/2GiB belt (child outcome {kind}). These asserts "
                "pin the node-cap contract once the fix lands (ValueError past the "
                "documented walk ceiling, catchable, peak at baseline)."
            )
        assert kind == "ValueError", (
            f"expected the documented walk-ceiling refusal, got {kind}: {message}"
        )
        assert peak_kib < 512 * 1024, (
            f"peak {peak_kib / 1024:.0f} MiB: the refusal arrived after expanding the graph"
        )

    def test_cycle_schema_terminates_at_the_depth_cap(self) -> None:
        """The CYCLE shape (a self-referential list): the depth cap
        (MAX_SCHEMA_WALK_DEPTH, 200) already fires on main — this pin
        holds on every branch and anchors the catchable-error half of
        the contract the shared-ref guard awaits."""
        setup = "x = []\nx.append(x)\n"
        kind, message, _peak_kib, done = _run_child(
            'output = tors.repair_json_loads("{}", schema={"enum": x})',
            setup=setup,
            timeout=20.0,
        )
        _assert_alive(kind, message, done)
        assert kind == "ValueError", (
            f"a cyclic schema must refuse at the depth cap with a catchable error, "
            f"got {kind}: {message}"
        )


# --- the PEM-flood lane's peak (the #92 shape's memory half) ------------------------


class TestScrubPiiPemFloodPeakBounded:
    def test_pem_flood_peak_is_tied_to_input(self) -> None:
        """The #92 flood (all-different header words, no verifying END) is
        fixed for TIME; this guard pins the MEMORY half on every branch:
        the scrub's peak must stay tied to the input (~600KB), not to the
        match structure. Unbounded per-BEGIN buffering trips the 30x
        ceiling; the fix's measured peak sits at a few multiples of the
        input."""
        n = 20_000
        input_bytes = n * 63  # one BEGIN + one END line per header unit
        ceiling_kib = (30 * input_bytes) // 1024
        setup = (
            f"n = {n}\n"
            "text = ''.join(f'-----BEGIN K{{i}} PRIVATE KEY-----\\n' for i in range(n)) \\\n"
            "       + ''.join(f'-----END L{{i}} PRIVATE KEY-----\\n' for i in range(n))\n"
        )
        kind, message, peak_kib, done = _run_child(
            "output = tors.scrub_pii(text)", setup=setup, timeout=60.0
        )
        _assert_alive(kind, message, done)
        assert kind == "OK", f"the flood must scrub cleanly, got {kind}: {message}"
        assert peak_kib < ceiling_kib, (
            f"peak {peak_kib / 1024:.0f} MiB exceeds the input-tied ceiling "
            f"{ceiling_kib / 1024:.0f} MiB ({input_bytes / _MIB:.1f} MiB input): the "
            "keys pass buffers per-BEGIN state that scales with the flood"
        )


# --- the python-side lane (tracemalloc), for completeness ---------------------------
#
# The amplification classes above are all Rust-side (invisible to
# tracemalloc); the one Python-side amplifier in the surface —
# chunk_hierarchical's list-of-tuples marshalling for a chunk-count-heavy
# output — is bounded by the text length itself (chunks <= total), so
# there is no Python-side unbounded-amplification shape to guard. This
# cell pins that reasoning: tracemalloc over the marshalling of a
# chunk-count-heavy call grows with the OUTPUT list only, linearly.


class TestPythonSideMarshallingIsLinear:
    def test_chunk_output_marshalling_is_bounded_by_the_text(self) -> None:
        import tracemalloc

        import tors

        text = "ab" * 50_000
        tracemalloc.start()
        chunks = tors.chunk_hierarchical(text, 2)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(chunks) <= len(text) + 2, "the chunk count itself is unbounded"
        assert peak < 200 * len(text), (
            f"the Python-side marshalling peak ({peak / len(text):.0f}x the text) is not "
            "linear in the output list"
        )


# --- the grounding batch: the grounding surfaces' memory classes --------------------
#
# Both new surfaces are bounded by construction (src/grounding_impl.rs,
# src/grounded_impl.rs): ground_sentences holds one SentenceScore per
# sentence (O(text)) and marshals one dict per sentence (O(sentences),
# Python-side); grounding_coverage's DP is two reusable rows over the
# SHORTER token stream, O(min(|S|, |T|)), never a materialized n*m
# matrix. The guards pin both at their amplification shapes: a child runs
# the call and reports its own peak RSS (VmHWM, the disposable-child
# discipline), the ceiling tied to input bytes with baseline slack.

class TestGroundingBatchMemoryGuards:
    def test_ground_sentences_sentence_soup_peak_is_tied_to_the_text(self) -> None:
        # ~349k sentences from ~2.1 MB of input: the O(sentences) result
        # shape at its densest legal form (one two-word sentence per 6
        # bytes). A result structure NOT tied to the text size would blow
        # the input-tied ceiling; the dicts' marshalling peak is Python-
        # side, so the child measures the whole pipeline's peak.
        input_bytes = len("Ab cd. ") * 50_000
        kind, message, peak_kib, done = _run_child(
            "output = tors.ground_sentences(text, 'ab cd')",
            setup="text = 'Ab cd. ' * 50_000\n",
            timeout=60.0,
        )
        _assert_alive(kind, message, done)
        assert kind == "OK", f"the soup must batch cleanly, got {kind}: {message}"
        ceiling_kib = (100 * input_bytes) // 1024 + 64 * 1024
        assert peak_kib < ceiling_kib, (
            f"peak {peak_kib / 1024:.0f} MiB exceeds the input-tied ceiling "
            f"{ceiling_kib / 1024:.0f} MiB ({input_bytes / _MIB:.1f} MiB input): the "
            "batch's per-sentence state is not tied to the text"
        )

    def test_grounding_coverage_peak_is_two_rows_never_the_product_matrix(self) -> None:
        # THE amplification guard for the recall twin: two operands at the
        # DP's 16384-token cap (~100 KB each, ~200 KB of input). The
        # documented memory class is O(min(|S|, |T|)) (two rows); a
        # materialized n*m f64 matrix at this size would be ~2.1 GiB,
        # ~10,000x the input, and trip the input-tied ceiling by far.
        input_bytes = len("word ") * 16_384 * 2
        kind, message, peak_kib, done = _run_child(
            "output = tors.grounding_coverage(source, text)",
            setup=(
                "source = 'word ' * 16_384\n"
                "text = 'word ' * 16_384\n"
            ),
            timeout=120.0,
        )
        _assert_alive(kind, message, done)
        assert kind == "OK", f"the capped DP must complete cleanly, got {kind}: {message}"
        ceiling_kib = (100 * input_bytes) // 1024 + 64 * 1024
        assert peak_kib < ceiling_kib, (
            f"peak {peak_kib / 1024:.0f} MiB exceeds the input-tied ceiling "
            f"{ceiling_kib / 1024:.0f} MiB ({input_bytes / _MIB:.1f} MiB input): the "
            "DP materialized the n*m matrix (or worse) instead of two rows"
        )


# --- scrub_secrets: the token-dense output lane -------------------------------------
#
# The secret-token grammars replace each span with a ~17-21-char token.
# The SHORTEST matches (a minimum Slack shape: "xoxb-1-2-a" is 10 chars
# for a 17-char token) make the OUTPUT a constant ~1.7x the input, the
# family's worst factor and a constant: never input-dependent, never
# matching-structure-dependent. The guard runs the dense
# minimum-shape lane in a disposable child and pins the peak to the
# input bytes (a few multiples of it), the same input-tied ceiling the
# PEM-flood guard uses: any per-match buffering that scales with the
# match COUNT beyond the output's own size trips it.


class TestScrubSecretsPeakIsTiedToInput:
    def test_dense_minimum_shape_peak_is_tied_to_input(self) -> None:
        """Dense minimum Slack shapes (8 MiB of "xoxb-1-2-abc" units,
        ~840k matches): the scrub's peak must stay tied to the input
        (~8 MiB and its ~1.7x token output), never to the match
        structure. A 30x-input ceiling holds the interpreter's own
        baseline; per-match buffering beyond the output trips it."""
        unit = "xoxb-1-2-abc "
        units = 8 * _MIB // len(unit)
        input_bytes = units * len(unit)
        ceiling_kib = (30 * input_bytes) // 1024
        setup = f"units = {units}\ntext = {unit!r} * units\n"
        kind, message, peak_kib, done = _run_child(
            "output = tors.scrub_secrets(text)", setup=setup, timeout=60.0
        )
        _assert_alive(kind, message, done)
        assert kind == "OK", f"the dense lane must scrub cleanly, got {kind}: {message}"
        assert peak_kib < ceiling_kib, (
            f"peak {peak_kib / 1024:.0f} MiB exceeds the input-tied ceiling "
            f"{ceiling_kib / 1024:.0f} MiB ({input_bytes / _MIB:.1f} MiB input): the "
            "secret scan buffers per-match state that scales with the match count"
        )
