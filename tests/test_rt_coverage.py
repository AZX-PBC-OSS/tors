"""Empirical red-team coverage for PR #89 (issue #80): the bounded
two-phase read. Each test pins one guarantee the PR's own suite leaves
unpinned; the HYPOTHESIS (the guarantee) and the surface (path/data,
lane, budget) are stated per test. Probes run in disposable
subprocesses, the ``test_documents_hardening.py`` pattern: hangs and
aborts die in the child, and the parent asserts on the captured outcome,
folding the child's output into every failure message."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MiB = 1024 * 1024
SNIFF_PREFIX = 64 * 1024  # the binding's SNIFF_PREFIX (tors-documents/src/lib.rs)
SIGNAL_RCS = (-9, -6, 137, 134)
# The metered-lane witness rows: the delimiter evidence both the csv sniff
# and the provisional lane guess key on (the hardening suite's shape).
_CSV_UNIT = b"unit,status\na,ok\n"


_THIN_SPAWNER = (
    "import subprocess, sys\n"
    "raise SystemExit(subprocess.run([sys.executable, '-c', *sys.argv[1:]]).returncode)\n"
)


def _probe(code: str, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """One disposable child (this venv's own interpreter); outcome captured
    verbatim. ``args`` ride as argv, never interpolated into the source.

    Spawns through a thin intermediate: a child forked directly from a fat
    pytest process inherits the parent's RSS high-water mark (fork records
    the parent's resident pages before exec resets the address space;
    ru_maxrss never resets), so RSS-measuring children would report the
    PARENT'S footprint. fork(pytest) -> exec(thin) -> fork(thin) ->
    exec(probe) keeps the measured child's ru_maxrss its own."""
    return subprocess.run(
        [sys.executable, "-c", _THIN_SPAWNER, textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _csv_sparse(path: Path, total_bytes: int) -> Path:
    """A csv of ``total_bytes``: >=128 real witness rows up front (the csv
    heuristic samples the first 64 lines, which stay real), then a sparse
    hole out to size."""
    with open(path, "wb") as f:
        f.write(_CSV_UNIT * 128)
        f.truncate(total_bytes)
    return path


def _html_exact(total: int, head_token: str, tail_token: str) -> bytes:
    """An HTML document of exactly ``total`` bytes: ``head_token`` within the
    first 50 bytes, ``tail_token`` within the last 50, both in text nodes so
    the extracted text carries them."""
    head = f"<!doctype html><html><body><p>{head_token}</p>\n".encode()
    tail = f"<p>{tail_token}</p></body></html>".encode()
    assert head.index(head_token.encode()) + len(head_token) <= 50
    filler = b"<p>lorem ipsum dolor sit amet</p>\n"
    body_room = total - len(head) - len(tail) - 1  # 1 for the pre-tail newline
    assert body_room > 0
    body = filler * (body_room // len(filler))
    out = head + body + b" " * (body_room - len(body)) + b"\n" + tail
    assert len(out) == total
    assert tail_token.encode() in out[-50:]
    return out


def _maxrss_kb(stdout: str) -> int:
    found = next(
        int(line.split()[1])
        for line in stdout.splitlines()
        if line.startswith("MAXRSS_KB")
    )
    return found


class TestH1SpliceAtEveryPrefixSeam:
    @pytest.mark.parametrize(
        "total",
        [SNIFF_PREFIX - 1, SNIFF_PREFIX, SNIFF_PREFIX + 1, SNIFF_PREFIX + 2],
        ids=["64KiB-1", "64KiB", "64KiB+1", "64KiB+2"],
    )
    def test_html_at_the_seam_keeps_head_and_tail(self, tmp_path: Path, total: int) -> None:
        """HYPOTHESIS (surface: path=, HTML/unmetered lane, max_bytes=None):
        the two-phase read is byte-correct at every 64 KiB seam case — a file
        one byte under the prefix, exactly at it, one over (fully consumed by
        the prefix read's one-past-ceiling grab), and two over (the minimal
        case where phase 2 actually supplies a byte). The head sentinel
        (first 50 bytes) and the tail sentinel (last 50) must both survive
        into to_text, rc=0, and a second conversion of the same file must be
        byte-identical (splice determinism)."""
        p = tmp_path / "seam.html"
        p.write_bytes(_html_exact(total, "HEAD01X", "TAIL02X"))
        code = (
            "from tors_documents import to_text\n"
            f"_p = {str(p)!r}\n"
            "_fmt, text = to_text(path=_p)\n"
            "_ok = 'HEAD01X' in text and 'TAIL02X' in text\n"
            "_fmt2, text2 = to_text(path=_p)\n"
            "print('SENTINELS_OK', _ok, 'DETERMINISTIC', text == text2, len(text))\n"
            "raise SystemExit(0 if _ok and text == text2 else 3)\n"
        )
        done = _probe(code)
        assert done.returncode not in SIGNAL_RCS, f"signal death:\n{done.stderr}"
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
        assert "SENTINELS_OK True DETERMINISTIC True" in done.stdout


class TestH2LaneGuessIsContentNotExtension:
    @pytest.mark.parametrize("name", ["x.pdf", "x", "x.csv"])
    def test_metered_ceiling_identical_under_any_name(self, tmp_path: Path, name: str) -> None:
        """HYPOTHESIS (surface: path=, ~33 MiB csv, max_bytes=None): the
        provisional lane guess is content-driven, so the SAME csv body
        refuses with the SAME metered 32 MiB read-time ceiling message no
        matter the name — .pdf, extensionless, or .csv. A refusal at a
        different ceiling, a POST-read refusal ("engine lane's input
        ceiling" is the core's message, not the guess's), or a conversion
        would be a lane-guess inconsistency finding.

        Fixture note: a ceiling is only observable from a file OVER it, so
        the fixture is 33 MiB (sparse: real witness rows up front, hole to
        size); a sub-ceiling csv converts on every lane and could not pin
        the guess at all."""
        p = _csv_sparse(tmp_path / name, 33 * MiB)
        code = (
            "from tors_documents import to_text\n"
            f"to_text(path={str(p)!r})\n"
        )
        done = _probe(code)
        assert done.returncode not in SIGNAL_RCS, f"signal death:\n{done.stderr}"
        assert "ValueError" in done.stderr, done.stderr
        assert "default read ceiling" in done.stderr, (
            f"refusal did not come from the phase-2 metered read ceiling:\n{done.stderr}"
        )
        assert "32.0 MiB" in done.stderr, done.stderr


class TestH3ExplicitBudgetRaisesTheDefault:
    def test_40mib_csv_converts_under_64mib_budget(self, tmp_path: Path) -> None:
        """HYPOTHESIS (surface: path=, csv/anydoc lane, max_bytes=64 MiB):
        an explicit budget overrides the 32 MiB lane default UPWARD — a
        40 MiB csv converts successfully (rc=0) and its last-row sentinel
        reaches the output. The docs promise max_bytes 'overrides the
        default in either direction'; a refusal here is a High
        docs-contract break."""
        p = tmp_path / "big.csv"
        with open(p, "wb") as f:
            f.write(b"unit,status\n")
            f.write(b"a,ok\n" * (8 * 1024 * 1024))  # 40 MiB of rows
            f.write(b"zz-last-row-9q,ok\n")
        code = (
            "from tors_documents import to_text\n"
            f"_fmt, text = to_text(path={str(p)!r}, max_bytes={64 * MiB})\n"
            "_ok = 'zz-last-row-9q' in text\n"
            "print('CONVERTED', _ok, len(text))\n"
            "raise SystemExit(0 if _ok else 3)\n"
        )
        done = _probe(code, timeout=60)
        assert done.returncode not in SIGNAL_RCS, f"signal death:\n{done.stderr}"
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
        assert "CONVERTED True" in done.stdout


class TestH4PrefixBlindMeteredFile:
    def test_whitespace_prefix_csv_refuses_during_read_not_after(self, tmp_path: Path) -> None:
        """HYPOTHESIS (surface: path=, csv/anydoc lane, max_bytes=None): a
        metered file whose first 64 KiB gives the sniffer NOTHING (pure
        whitespace) is still metered — through the name hint — so the
        refusal comes clean (ValueError, rc not a signal) DURING the read at
        the 32 MiB ceiling, and the child's ru_maxrss stays far under the
        whole-file size. If the whole file is buffered before the refusal
        (RSS at the fixture size), the read is still pre-refusal-unbounded
        for prefix-blind files: a Medium finding (the sniffer can be
        starved).

        Fixture: 200 MiB sparse csv whose first 64 KiB is b' \\n' — 200 MiB
        so the starved case (full read before any refusal) clears the
        200 MiB bar while the metered case (~32 MiB buffered) cannot."""
        p = tmp_path / "blind.csv"
        with open(p, "wb") as f:
            f.write(b" \n" * (SNIFF_PREFIX // 2))  # 64 KiB of pure whitespace
            f.write(_CSV_UNIT * 128)  # the csv witness, past the blind prefix
            f.truncate(200 * MiB)
        code = (
            "import resource\n"
            "from tors_documents import to_text\n"
            f"_p = {str(p)!r}\n"
            "try:\n"
            "    to_text(path=_p)\n"
            "    print('ERRTYPE UNEXPECTED-CONVERT')\n"
            "except Exception as e:\n"
            "    print('ERRTYPE', type(e).__name__)\n"
            "    print('ERRMSG', e)\n"
            "print('MAXRSS_KB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
        )
        done = _probe(code, timeout=60)
        assert done.returncode not in SIGNAL_RCS, f"signal death:\n{done.stderr}"
        assert done.returncode == 0, f"probe child failed:\n{done.stdout}\n{done.stderr}"
        assert "ERRTYPE ValueError" in done.stdout, done.stdout
        maxrss_kb = _maxrss_kb(done.stdout)
        assert maxrss_kb < 200 * 1024, (
            f"child ru_maxrss {maxrss_kb} kB: the read buffered past the 32 MiB "
            "metered ceiling before refusing (the blind prefix starved the "
            f"sniffer):\n{done.stdout}"
        )
        assert "default read ceiling" in done.stdout, (
            "refusal did not come from the phase-2 metered read:\n" + done.stdout
        )
        assert "32.0 MiB" in done.stdout


class TestH5DataPathParity:
    def test_same_1mib_csv_refuses_identically_via_path_and_data(self, tmp_path: Path) -> None:
        """HYPOTHESIS (surface: path= AND data=, ~1 MiB csv, max_bytes=1024):
        the same content refuses on BOTH lanes with a ValueError whose
        message names max_bytes and never discloses the tmp path (the
        data-lane TypeError-repr lesson must hold for value refusals too:
        no filesystem context in any refusal message)."""
        p = tmp_path / "one.csv"
        with open(p, "wb") as f:
            f.write(_CSV_UNIT * 60000)  # ~1.02 MiB
        code = (
            "from tors_documents import to_text\n"
            f"_p = {str(p)!r}\n"
            "_data = open(_p, 'rb').read()\n"
            "for lane, kw in (('path', dict(path=_p)), ('data', dict(data=_data))):\n"
            "    try:\n"
            "        to_text(max_bytes=1024, **kw)\n"
            "        print(lane.upper(), 'NO-ERROR')\n"
            "    except Exception as e:\n"
            "        print(lane.upper(), 'TYPE', type(e).__name__)\n"
            "        print(lane.upper(), 'MSG', e)\n"
        )
        done = _probe(code)
        assert done.returncode not in SIGNAL_RCS, f"signal death:\n{done.stderr}"
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
        assert "NO-ERROR" not in done.stdout, done.stdout
        for lane in ("PATH", "DATA"):
            assert f"{lane} TYPE ValueError" in done.stdout, done.stdout
            msg_line = next(
                ln for ln in done.stdout.splitlines() if ln.startswith(f"{lane} MSG ")
            )
            assert "max_bytes" in msg_line, msg_line
            assert str(tmp_path) not in msg_line, f"tmp path disclosed: {msg_line}"


class TestH6OomMapsToValueError:
    @pytest.mark.skipif(
        not Path("/proc/self/status").exists(),
        reason="leg (b) forces a real allocation failure with Linux RLIMIT_AS "
        "semantics (issue #86): the belt does not constrain malloc the same "
        "way off Linux",
    )
    def test_tiny_budget_and_real_alloc_failure_both_valueerror(self, tmp_path: Path) -> None:
        """HYPOTHESIS (surface: path=, the budget family): the OOM mapping is
        not a lie —
        (a) a tiny explicit budget on an over-budget file is a ValueError
            naming max_bytes, never MemoryError/OSError;
        (b) a GENUINE allocation failure inside the bounded read surfaces as
            ValueError too: an RLIMIT_AS=384 MiB child reading a 3 GiB
            sparse pdf under None can never fit the 512 MiB backstop
            (384 < any base + 512), so read_bounded_into's try_reserve_exact
            fails for real, and map_read_err must turn that into ValueError
            ('too large to buffer'), not MemoryError and not an abort. The
            fixture is a PDF (not html) because the F-2 fix meters the HTML
            lane: a sparse html now refuses at the 32 MiB lane ceiling long
            before any backstop-sized allocation is attempted; the pdf lane
            is the one that still reads toward the 512 MiB backstop;
        (c) STATIC, not executed: an explicit huge budget (10**12) on a
            small file cannot abort either — read_bounded_into asks for at
            most one 64 KiB chunk per read regardless of ceiling, so no
            budget-sized allocation exists, and a real env-OOM at any
            ceiling maps ErrorKind::OutOfMemory -> InputError::Refused ->
            PyValueError (map_read_err / input_document_error /
            document_error, tors-documents/src/lib.rs) — the same class as
            (a). We never allocate a terabyte to prove it;
        (d) the (c) no-preallocation half, empirically: max_bytes=10**12 on
            a small file converts, rc=0, promptly."""
        # (a) tiny explicit budget: stat pre-gate refusal.
        big = _csv_sparse(tmp_path / "two.csv", 2 * MiB)
        code_a = (
            "from tors_documents import to_text\n"
            f"try:\n    to_text(path={str(big)!r}, max_bytes=1024)\n"
            "    print('A NO-ERROR')\n"
            "except Exception as e:\n"
            "    print('A TYPE', type(e).__name__)\n"
            "    print('A MSG', e)\n"
        )
        done_a = _probe(code_a)
        assert done_a.returncode not in SIGNAL_RCS, f"signal death:\n{done_a.stderr}"
        assert done_a.returncode == 0, f"{done_a.stdout}\n{done_a.stderr}"
        assert "A TYPE ValueError" in done_a.stdout, done_a.stdout
        a_msg = next(ln for ln in done_a.stdout.splitlines() if ln.startswith("A MSG "))
        assert "max_bytes" in a_msg, a_msg

        # (b) genuine allocation failure inside the bounded read.
        huge = tmp_path / "huge.pdf"
        with open(huge, "wb") as f:
            f.write(b"%PDF-1.4\n")
            f.truncate(3 * 1024**3)  # sparse: 3 GiB of address space, ~40 bytes of disk
        code_b = (
            "import resource\n"
            "limit = 384 * 1024 * 1024\n"
            "resource.setrlimit(resource.RLIMIT_AS, (limit, limit))\n"
            "from tors_documents import to_text\n"
            f"try:\n    to_text(path={str(huge)!r})\n"
            "    print('B UNEXPECTED-CONVERT')\n"
            "except Exception as e:\n"
            "    print('B TYPE', type(e).__name__)\n"
            "    print('B MSG', e)\n"
        )
        done_b = _probe(code_b, timeout=60)
        assert done_b.returncode not in SIGNAL_RCS, (
            f"the alloc failure aborted the child instead of mapping to a "
            f"ValueError:\n{done_b.stderr}"
        )
        assert done_b.returncode == 0, f"{done_b.stdout}\n{done_b.stderr}"
        assert "B TYPE ValueError" in done_b.stdout, done_b.stdout
        b_msg = next(ln for ln in done_b.stdout.splitlines() if ln.startswith("B MSG "))
        assert "too large to buffer" in b_msg, (
            f"the OOM did not come from read_bounded_into's try_reserve_exact "
            f"path: {b_msg}"
        )

        # (d) a huge explicit budget is a ceiling, never a pre-allocation.
        small = tmp_path / "small.html"
        small.write_bytes(_html_exact(200_000, "HEAD01X", "TAIL02X"))
        code_d = (
            "from tors_documents import to_text\n"
            f"_fmt, text = to_text(path={str(small)!r}, max_bytes={10**12})\n"
            "_ok = 'HEAD01X' in text and 'TAIL02X' in text\n"
            "print('D CONVERTED', _ok)\n"
            "raise SystemExit(0 if _ok else 3)\n"
        )
        done_d = _probe(code_d)
        assert done_d.returncode not in SIGNAL_RCS, f"signal death:\n{done_d.stderr}"
        assert done_d.returncode == 0, f"{done_d.stdout}\n{done_d.stderr}"
        assert "D CONVERTED True" in done_d.stdout


class TestH7ThreadedConvertSmoke:
    def test_four_threads_four_files_no_corruption(self, tmp_path: Path) -> None:
        """HYPOTHESIS (surface: path= x4 across 4 threads in ONE child, HTML
        lane, max_bytes=None): the bounded two-phase read introduced no
        cross-thread corruption: 4 threads converting 4 distinct 1 MiB HTML
        files all return correct text (each file's own head/tail sentinels,
        none of another's), and the child stays under 300 MiB (catches
        splice races and unbounded-read regressions)."""
        paths: list[Path] = []
        for i in range(4):
            p = tmp_path / f"t{i}.html"
            p.write_bytes(_html_exact(MiB, f"HEAD{i}X", f"TAIL{i}X"))
            paths.append(p)
        code = (
            "import resource, sys, threading\n"
            "from tors_documents import to_text\n"
            "_paths = sys.argv[1:]\n"
            "_results = [None] * len(_paths)\n"
            "def _work(i, p):\n"
            "    try:\n"
            "        _fmt, text = to_text(path=p)\n"
            "        _results[i] = f'HEAD{i}X' in text and f'TAIL{i}X' in text\n"
            "    except Exception as e:\n"
            "        _results[i] = f'{type(e).__name__}: {e}'\n"
            "_ts = [threading.Thread(target=_work, args=(i, p)) for i, p in enumerate(_paths)]\n"
            "[t.start() for t in _ts]\n"
            "[t.join() for t in _ts]\n"
            "for i, r in enumerate(_results):\n"
            "    print('RESULT', i, r)\n"
            "print('MAXRSS_KB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
        )
        done = _probe(code, *(str(p) for p in paths), timeout=60)
        assert done.returncode not in SIGNAL_RCS, f"signal death:\n{done.stderr}"
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
        for i in range(4):
            assert f"RESULT {i} True" in done.stdout, done.stdout
        assert _maxrss_kb(done.stdout) < 300 * 1024, (
            f"child RSS blew past 300 MiB converting 4x1 MiB files:\n{done.stdout}"
        )
