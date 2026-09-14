"""PR #89's docs, turned into executable contracts (docs-contract testing:
every doc PROMISE in the changed files becomes a test and is RUN — a promise
that fails is a docs lie or a code divergence, either way a finding).

The promises, quoted verbatim from the diff (file:line is this tree's):

- P1  docs/api.md:3013-3015 — "An explicit `max_bytes` binds every engine
  lane, pdf and HTML included, and is enforced before any work runs: the
  file's size at open (a `path=` call never reads an over-budget byte)";
  docs/api.md:2899 — "an explicit budget binds every lane, the PDF-only
  family included, checked before a byte is read or copied".
- P2  docs/api.md:2899 — "under the default the metered anydoc and
  office_oxide lanes refuse during the read at the 32 MiB ceiling";
  docs/api.md:3017-3021 — "refuse *during* the read at the 32 MiB default
  ceiling".
- P3  docs/api.md:2899 — "the pdf lane reads under the 512 MiB
  backstop" (HTML now meters, the F-2 fix); docs/documents.md:129-135 —
  "`None` reads under
  `MAX_INPUT_READ`, a finite 512 MiB read ceiling enforced during the read
  itself".
- P4  docs/documents.md:127-129 — "An explicit value binds every engine
  lane before a byte is read or copied, overriding the default in either
  direction".
- P5  docs/api.md:3024-3025 — "Over the ceiling, the call raises
  `ValueError` naming the ceiling and the `max_bytes=` override." (the
  replaced text promised "naming both sizes"; the new shape names the
  ceiling — the during-the-read refusal cannot know the size past the cap).
- P6  tors-documents/python/tors_documents/__init__.py:160-165,
  228-230, 249-252, 270-272, 288-290 — the public docstrings' None-doctrine
  text; stale phrases ("unmetered", "keeps the pdf lane unmetered",
  "post-read" as the sole doctrine) must be gone (the precedent is this
  suite's TestDocstringHonesty).
- P7  docs/documents.md:129-135 — the 512 MiB backstop is a user-visible
  change; this repo's changelog mechanism (release-please over conventional
  commits) must be positioned to surface it (static, report-only).

Memory- and time-shaped probes run in disposable subprocesses (the _probe
pattern from tests/test_documents_hardening.py) so an OOM/hang shape dies
in the child; RSS claims read the child's VmHWM. Fixture sizes are the
docs' own operating points (40 MiB against the 32 MiB default, well under
the 512 MiB backstop).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from test_documents_hardening import _hwm_kb, _run_or_fail
from tors_documents import (
    pdf_classify,
    pdf_extract,
    pdf_link_uris,
    pdf_page_count,
    to_markdown,
    to_text,
)

REPO = Path(__file__).resolve().parents[1]

_CONVENTIONAL = re.compile(
    r"^(?:feat|fix|docs|perf|refactor|test|build|ci|chore|style|revert)"
    r"(\([\w.-]+\))?!?: "
)
LIB_RS = REPO / "tors-documents" / "src" / "lib.rs"
IMPL_RS = REPO / "src" / "documents_impl.rs"

_CSV_ROWS = 8 * 1024 * 1024  # b"a,ok\n" * rows + header ~= 40 MiB, near the 32 MiB default
_HTML_ROW = b"<p>lorem ipsum dolor sit amet</p>\n"


def _write_csv(path: Path, rows: int = _CSV_ROWS) -> int:
    """The hardening suite's 40 MiB csv fixture: two short columns, the
    anydoc lane's amplifying shape."""
    with open(path, "wb") as f:
        f.write(b"unit,status\n")
        f.write(b"a,ok\n" * rows)
    return path.stat().st_size


def _write_html(path: Path, target_bytes: int) -> int:
    with open(path, "wb") as f:
        f.write(b"<!doctype html><html><head><title>t</title></head><body>\n")
        f.write(_HTML_ROW * (target_bytes // len(_HTML_ROW)))
    return path.stat().st_size


def _line_of(haystack: str, needle: str) -> int:
    """1-based line of the first occurrence, for file:line evidence."""
    found = haystack.find(needle)
    assert found >= 0, f"static anchor vanished from the source: {needle!r}"
    return haystack.count("\n", 0, found) + 1


# --- P1: an explicit max_bytes is pre-read on every lane --------------------


class TestP1ExplicitBudgetIsCheckedBeforeAnyWork:
    """P1's pins (docs/api.md:3013-3015, docs/api.md:2899). Green,
    measured on this tree: the refusal is the stat-gate message naming
    max_bytes, 0.000s in, and a 32 MiB HTML under a 1 KiB budget refuses
    with the fixture never materialized in RSS."""

    def test_small_html_over_budget_refuses_before_any_parse_work(self, tmp_path: Path) -> None:
        """A 2 KiB HTML over a 1024-byte budget: ValueError (rc clean —
        _run_or_fail folds a crash/timeout into the failure), the message
        names max_bytes. Pre-read: nothing parse-shaped can have run."""
        path = tmp_path / "small.html"
        size = _write_html(path, 2048)
        assert size > 1024, "fixture must sit over the budget for the gate to bite"
        code = r"""
import sys
from tors_documents import to_text
try:
    out = to_text(path=sys.argv[1], max_bytes=1024)
    print(f"CONVERTED {out[0]}")
except BaseException as exc:
    print(f"{type(exc).__name__}: {exc}")
"""
        report = _run_or_fail(code, str(path), timeout=60)
        assert report.startswith("ValueError"), f"not a typed refusal: {report!r}"
        assert "max_bytes" in report, (
            "docs/api.md:3024-3025 promises the refusal names the "
            f"max_bytes= override:\n{report}"
        )

    @pytest.mark.skipif(
        not Path("/proc/self/status").exists(), reason="the VmHWM half of the probe is Linux-shaped"
    )
    def test_a_32mib_html_refuses_preread_fast_and_light(self, tmp_path: Path) -> None:
        """The timing/RSS proof of 'checked before any work runs': a 32 MiB
        HTML (expensive to parse) under max_bytes=1024 refuses in <5s with
        no large RSS — the budget gate read the stat, not the file."""
        path = tmp_path / "big.html"
        _write_html(path, 32 * 1024 * 1024)
        code = r"""
import sys, time
from tors_documents import to_text
t0 = time.perf_counter()
try:
    out = to_text(path=sys.argv[1], max_bytes=1024)
    print(f"CONVERTED {out[0]}")
except BaseException as exc:
    hwm = next(l for l in open("/proc/self/status") if l.startswith("VmHWM"))
    print(f"{type(exc).__name__}: {exc}")
    print(f"elapsed={time.perf_counter() - t0:.3f}s {hwm.strip()}")
"""
        report = _run_or_fail(code, str(path), timeout=60)
        found = re.search(r"elapsed=([0-9.]+)s", report)
        assert found, f"the probe did not report its timing:\n{report}"
        assert float(found.group(1)) < 5.0, (
            "docs/api.md:3013-3015 promises the budget is checked before "
            f"any work runs; the refusal took too long:\n{report}"
        )
        peak = _hwm_kb(report)
        assert peak < 100_000, (
            "docs/api.md:3014 promises a path= call never reads an "
            f"over-budget byte; the child materialized the file (VmHWM "
            f"{peak} kB):\n{report}"
        )


# --- P2: under None, metered lanes refuse during the read at 32 MiB --------


class TestP2MeteredLanesRefuseDuringTheReadAt32Mib:
    """P2's pin (docs/api.md:2899, docs/api.md:3017-3021). Green, measured:
    the 40 MiB csv refuses at the 32 MiB ceiling with the whole file never
    resident (VmHWM 55.7 MB, 0.02s)."""

    @pytest.mark.skipif(
        not Path("/proc/self/status").exists(), reason="the VmHWM half of the probe is Linux-shaped"
    )
    def test_a_40mib_csv_under_none_refuses_at_the_ceiling_during_the_read(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "big.csv"
        _write_csv(path)
        code = r"""
import sys, time
from tors_documents import to_text
t0 = time.perf_counter()
try:
    out = to_text(path=sys.argv[1])
    print(f"CONVERTED {out[0]}")
except BaseException as exc:
    hwm = next(l for l in open("/proc/self/status") if l.startswith("VmHWM"))
    print(f"{type(exc).__name__}: {exc}")
    print(f"elapsed={time.perf_counter() - t0:.3f}s {hwm.strip()}")
"""
        report = _run_or_fail(code, str(path), timeout=60)
        assert report.startswith("ValueError"), f"not a typed refusal:\n{report}"
        assert "32" in report, (
            "docs/api.md:2899 promises the metered lane refuses at the "
            f"32 MiB ceiling; the message does not name it:\n{report}"
        )
        peak = _hwm_kb(report)
        assert peak < 100_000, (
            "docs/api.md:3020 promises the refusal lands *during* the read "
            f"(the 40 MiB input must never be resident); VmHWM {peak} kB "
            f"says it was:\n{report}"
        )


# --- P3: the pdf lane reads under the 512 MiB backstop; HTML meters ---------


class TestP3ThePdfLaneReadsUnderThe512MibBackstop:
    """P3 (docs/api.md:2899, docs/documents.md:129-135, updated by the F-2
    fix): static evidence that MAX_INPUT_READ (512 MiB) is the provisional
    ceiling's fallback — the pdf lane's alone now — plus the empirical
    half flipped by F-2: a 40 MiB HTML under None REFUSES at the 32 MiB
    metered ceiling during the read (the HTML lane amplifies ~23x, so it
    meters like anydoc/office_oxide)."""

    def test_static_max_input_read_is_the_provisional_fallback(self) -> None:
        """docs/documents.md:130 names `MAX_INPUT_READ` as the None ceiling.
        Static wiring: the const is 512 MiB and is the fallback argument of
        the provisional_read_ceiling call in the binding's two-phase read."""
        source = LIB_RS.read_text(encoding="utf-8")
        const_line = _line_of(source, "pub(crate) const MAX_INPUT_READ: usize =")
        assert re.search(
            r"pub\(crate\) const MAX_INPUT_READ: usize = 512 \* 1024 \* 1024;", source
        ), f"tors-documents/src/lib.rs:{const_line}: MAX_INPUT_READ is not the documented 512 MiB"
        call = re.search(r"provisional_read_ceiling\((.*?)\);", source, re.S)
        assert call, "tors-documents/src/lib.rs: the provisional_read_ceiling call vanished"
        args = [a.strip() for a in call.group(1).split(",") if a.strip()]
        assert args[-1] == "MAX_INPUT_READ", (
            "tors-documents/src/lib.rs:{}: the fallback argument of "
            "provisional_read_ceiling is {!r}, not MAX_INPUT_READ — the "
            "docs' 512 MiB backstop is no longer wired".format(
                _line_of(source, "provisional_read_ceiling("), args[-1]
            )
        )

    def test_a_40mib_html_under_none_meters_during_the_read(self, tmp_path: Path) -> None:
        """F-2 fix pin (a 512 MiB pdf fixture is out of budget for a test):
        40 MiB — over the 32 MiB metered ceiling — refuses DURING the read
        on the HTML lane under None with the read-phase message; the lane
        no longer converts unbounded at the measured ~23x."""
        path = tmp_path / "big.html"
        _write_html(path, 40 * 1024 * 1024)
        code = r"""
import sys
from tors_documents import to_text
try:
    fmt, text = to_text(path=sys.argv[1])
    print(f"CONVERTED {fmt} out_len={len(text)}")
except BaseException as exc:
    print(f"{type(exc).__name__}: {exc}")
"""
        report = _run_or_fail(code, str(path), timeout=60)
        assert "ValueError" in report and "default read ceiling" in report, (
            "docs/api.md:2899 promises the HTML lane refuses during the read "
            f"at the 32 MiB ceiling; a 40 MiB HTML did not:\n{report}"
        )


# --- P4: an explicit budget overrides the default in either direction ------


class TestP4ExplicitBudgetOverridesTheDefaultInEitherDirection:
    """P4 (docs/documents.md:127-129) — the highest-risk promise: an
    explicit budget above the 32 MiB default must let the metered lane
    through. Red would mean the core's post-read 32 MiB check ignores the
    override and the docs lie. Static half: the core reads the budget
    first (unwrap_or over DEFAULT_ANYDOC_INPUT_LIMIT)."""

    def test_static_the_cores_post_read_check_honors_an_explicit_budget(self) -> None:
        source = IMPL_RS.read_text(encoding="utf-8")
        line = _line_of(source, "options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT)")
        assert "options.max_bytes.unwrap_or(DEFAULT_ANYDOC_INPUT_LIMIT)" in source, (
            f"src/documents_impl.rs:{line}: the core's post-read check no "
            "longer takes the explicit budget over the 32 MiB default — "
            "docs/documents.md:128-129's 'overriding the default' half breaks"
        )

    def test_a_40mib_csv_with_a_64mib_budget_converts(self, tmp_path: Path) -> None:
        """40 MiB csv under an explicit 64 MiB budget: SUCCESS (rc=0, real
        output). Downward is pinned by TestP1 (1024 bytes refuses a 32 MiB
        HTML)."""
        path = tmp_path / "big.csv"
        size = _write_csv(path)
        assert size > 32 * 1024 * 1024, "fixture must sit over the 32 MiB default"
        code = r"""
import sys
from tors_documents import to_text
fmt, text = to_text(path=sys.argv[1], max_bytes=64 * 1024 * 1024)
print(f"CONVERTED {fmt} out_len={len(text)} sample={text[:32]!r}")
"""
        report = _run_or_fail(code, str(path), timeout=60)
        assert "CONVERTED" in report, (
            "docs/documents.md:128-129 promises an explicit budget "
            "'overrid[es] the default in either direction'; a 40 MiB csv "
            f"under max_bytes=64 MiB refused instead:\n{report}"
        )
        assert "a | ok" in report, (
            f"the conversion succeeded but its output is not the csv's text:\n{report}"
        )


# --- P5: the refusal names the ceiling and the override ---------------------


class TestP5RefusalNamesTheCeilingAndTheOverride:
    """P5 (docs/api.md:3024-3025): "Over the ceiling, the call raises
    `ValueError` naming the ceiling and the `max_bytes=` override." Both
    refusal shapes are pinned. Size-naming note: the during-the-read
    refusal renders the ceiling only (past the cap the size is unknowable —
    the docs' old 'naming both sizes' promise is gone); the pre-read stat
    gate additionally names the stat size, which docs/api.md:3013-3014
    itself documents ('the file's size at open'), so no contract breach
    there."""

    def test_metered_none_refusal_names_the_ceiling_the_override_not_the_size(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "big.csv"
        _write_csv(path)
        code = r"""
import sys
from tors_documents import to_text
try:
    to_text(path=sys.argv[1])
except ValueError as exc:
    print(f"ValueError: {exc}")
"""
        report = _run_or_fail(code, str(path), timeout=60)
        message = report.strip()
        assert "exceeds the 32.0 MiB default read ceiling" in message, (
            "docs/api.md:3024-3025 promises the refusal names A ceiling in "
            f"human units; got:\n{message}"
        )
        assert "max_bytes" in message, (
            "docs/api.md:3024-3025 promises the refusal names the "
            f"max_bytes= override; got:\n{message}"
        )
        assert not re.search(r"the document is \d", message), (
            "the old message shape ('the document is N ... and the input "
            "ceiling is M') named a size the during-the-read refusal cannot "
            f"know; docs/api.md:3024-3025's new shape names the ceiling "
            f"instead. Got:\n{message}"
        )

    def test_explicit_budget_refusals_name_a_ceiling_and_the_override(
        self, tmp_path: Path
    ) -> None:
        """Both explicit-budget refusal shapes: the pre-read stat gate (a
        2 KiB HTML over a 1 KiB budget) and the during-the-read gate (a
        lying-stat file whose read overruns the budget)."""
        path = tmp_path / "small.html"
        _write_html(path, 2048)
        with pytest.raises(ValueError) as raised:
            to_text(str(path), max_bytes=1024)
        message = str(raised.value)
        assert "the input ceiling is 1024 bytes" in message, (
            "docs/api.md:3024-3025 promises a ceiling in human units; got:\n"
            f"{message}"
        )
        assert "max_bytes" in message, message

        if Path("/proc/self/status").exists():  # the lying-stat shape needs procfs
            with pytest.raises(ValueError) as raised:
                to_text("/proc/self/maps", format="csv", max_bytes=64)
            message = str(raised.value)
            assert "exceeds the max_bytes ceiling of 64 bytes" in message, (
                "the during-the-read explicit-budget refusal must name the "
                f"ceiling and the override; got:\n{message}"
            )
            assert not re.search(r"the document is \d", message), message


# --- P6: docstring honesty sweep -------------------------------------------


PUBLIC_CONVERSION_SURFACE: tuple[tuple[Callable[..., object], bool], ...] = (
    # (function, whether its own docstring states the None doctrine)
    (to_markdown, True),
    (to_text, False),  # delegates: "the same ... max_bytes= semantics as to_markdown"
    (pdf_classify, True),
    (pdf_extract, True),
    (pdf_link_uris, True),
    (pdf_page_count, True),
)


class TestP6DocstringHonestySweep:
    """P6: every public docstring's None-doctrine text is the new one.
    Absence of the stale phrases (the sweep the task pins), plus —
    TestDocstringHonesty's style — presence of the true claim where the
    docstring makes one."""

    @pytest.mark.parametrize(("call", "states_doctrine"), PUBLIC_CONVERSION_SURFACE)
    def test_no_stale_phrases_and_the_new_doctrine_present(
        self, call: Callable[..., object], states_doctrine: bool
    ) -> None:
        doc = call.__doc__ or ""
        lower = doc.lower()
        assert "unmetered" not in lower, (
            f"{call.__name__}'s docstring still calls a lane 'unmetered' "
            "(the pre-#89 doctrine the diff rewrote): tors-documents/python/"
            f"tors_documents/__init__.py"
        )
        assert "keeps the pdf lane unmetered" not in lower, (
            f"{call.__name__}'s docstring carries the stale 'keeps the pdf "
            "lane unmetered' promise"
        )
        if "post-read" in lower:
            assert "during the read" in lower or "512 mib" in lower, (
                f"{call.__name__}'s docstring presents 'post-read' as the "
                "sole doctrine for the read (the two-phase text is gone)"
            )
        if states_doctrine:
            assert "512 mib" in lower, (
                f"{call.__name__}'s docstring states None-doctrine text but "
                "no longer names the 512 MiB backstop it promises"
            )


# --- P7: changelog mechanism (static, report-only) --------------------------


class TestP7ChangelogMechanism:
    """P7 (static): the 512 MiB backstop is a user-visible change
    (docs/documents.md:129-135). This repo's changelog is release-please
    generated (CHANGELOG.md's only touchers are 'chore(main): release'
    commits; past behavior changes surface as entries at release time, e.g.
    the 0.6.1 fixes), so the entry rides the PR's conventional commits.
    The pin: every commit on this branch is conventional (feat/fix/docs
    surface), and at least one carries the behavior/docs change."""

    def test_the_backstop_change_rides_the_conventional_commit_mechanism(self) -> None:
        changelog = REPO / "CHANGELOG.md"
        assert changelog.exists(), "no CHANGELOG.md: past behavior changes got entries"
        log = subprocess.run(
            ["git", "-C", str(REPO), "log", "--format=%s", "main..HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert log.returncode == 0, log.stderr
        subjects = [line for line in log.stdout.splitlines() if line.strip()]
        assert subjects, "no commits between main and HEAD"
        non_conventional = [
            subject
            for subject in subjects
            if not _CONVENTIONAL.match(subject)
        ]
        assert not non_conventional, (
            "commits without a conventional type will be invisible to "
            f"release-please, so the 512 MiB backstop change would never "
            f"get its changelog entry: {non_conventional}"
        )
        assert any(
            re.match(r"^(feat|fix|docs)\(documents\)", subject) for subject in subjects
        ), f"no user-visible (feat/fix/docs) documents commit on this branch: {subjects}"
        touched = subprocess.run(
            ["git", "-C", str(REPO), "log", "--format=%s", "main..HEAD", "--", "CHANGELOG.md"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Report-only evidence: PRs here never hand-edit CHANGELOG.md (the
        # release commit does); the convention, not an in-PR entry, is the pin.
        assert not touched.stdout.strip(), (
            "this branch edits CHANGELOG.md directly, against the "
            f"release-please convention: {touched.stdout.splitlines()}"
        )
