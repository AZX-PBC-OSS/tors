"""PR #89 red-team engagement: empirical attacks on the two-phase bounded
read (`read_bounded_into` + `provisional_read_ceiling` + `into_input`).
Method matches tests/test_documents_hardening.py: every attack runs in a
disposable subprocess child (`_probe`, 60 s deadline), the child catches
its own outcome and measures `resource.getrusage(...).ru_maxrss`, and the
parent parses and asserts. Attack BLOCKED = a pinned defense (green).
Attack SUCCEEDS or degrades unacceptably = a FINDING whose measured
evidence (RSS MiB, refusal text, timing) rides in the failing assertion.

Attack inventory (one test each; hypothesis + pinned safe behavior live in
each test's ATTACK comment):

- A1  63 KiB whitespace leader before csv rows: the csv witness still lands
      inside the 64 KiB sniff prefix, so the metered lane refuses during the
      read at 32 MiB.
- A1b The leader grown past the sniff prefix (80 KiB), file named WITHOUT a
      known extension: pre-F-1 this content-blind prefix plus a silent name
      picked the 512 MiB backstop lane and buffered the FULL attacker file
      before the core's post-read check refused (measured red: F-1). The
      fix errs toward metering, so now the read-phase 32 MiB refusal must
      fire — this test pins that fix.
- A1c A1b's exact bytes named ``.csv``: the extension fallback meters a
      blinded prefix. The control that shows the file name is no longer
      load-bearing at all.
- A2  A ``%PDF-1.4`` header on csv bytes, named ``.csv``: content must beat
      the name (pdf lane), never route to the amplifying csv lane.
- A3  Truncated-ZIP garbage (``PK\\x03\\x04`` + zeros, never a central
      directory), no useful extension: unresolvable prefix. Updated by the
      F-1 fix: the read refuses at the 32 MiB metered ceiling DURING the
      read, not at the backstop after a full buffered read.
- A4  Four threads in one child, four distinct A1b files: concurrency
      multiplication of the evaded read must stay bounded and clean.
- A5a ``path=/proc/self/mem`` under None: the lying-stat special file must
      end as a typed fast refusal (OSError/ValueError), never a hang.
- A5b A no-writer FIFO under None: refused at the stat, BEFORE open(2)
      blocks forever.
- A6  ``data=`` parity: 48 MiB whitespace-led csv bytes under None add no
      new unbounded copy beyond the bytes themselves; clean refusal.
- A6b ``data=`` with ``max_bytes=1024``: pre-copy refusal, fast, input-sized
      RSS only.
- A7  Message hygiene: the over-ceiling refusals under None name neither an
      absolute filesystem path nor the exact byte size of the attacker file
      (both full messages are dumped to the log for the report).
- A8  A 0-bytes-on-disk sparse file (513 MiB logical, unresolvable content,
      no extension): updated by the F-1 fix — an unresolvable prefix now
      errs toward metering, so the read refuses at the 32 MiB provisional
      ceiling DURING the read, and silent content can no longer select the
      backstop lane.
- A9  The HTML lane METERS now (F-2 fix): a 48 MiB doctype-led HTML under
      None refuses during the read at the 32 MiB lane ceiling (pre-fix it
      converted at a measured ~23x input / 1118 MiB peak) — the uncapped
      conversion vector on the lane is closed.

Every probe carries a 60 s timeout (a hang is a finding, never a fixture
bug) and the memory-heavy children carry an RLIMIT_AS belt so a regression
dies loudly in the disposable child instead of on the box.

Harness note — why _probe double-spawns: a child forked directly from a
pytest process inherits the parent's RSS high-water mark (fork records the
parent's resident pages in the child before exec replaces the address
space; ru_maxrss never resets). After a suite whose fixtures fatten the
pytest process (32-200 MiB csv/html files), every directly-spawned child
reports the PARENT'S footprint, not its own (~272 MiB bare-child readings
vs ~37 MiB from a thin parent — verified). So probes spawn through a thin
intermediate: fork(pytest) -> exec(thin python) -> fork(thin) -> exec(the
real probe child), and only the real child's ru_maxrss is trusted."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_TARGET_BYTES = 48 * 1024 * 1024
_ROW = b"a,ok\n"
_HEADER = b"unit,status\n"
_SNIFF_PREFIX = 64 * 1024  # SNIFF_PREFIX, tors-documents/src/lib.rs

# The thin intermediate _probe execs into first: it re-forks the real probe
# child from its own slim image, so the measured child's ru_maxrss reflects
# the probe alone, never the (possibly fat) pytest process above us.
_THIN_SPAWNER = textwrap.dedent(
    """
    import subprocess, sys
    raise SystemExit(subprocess.run([sys.executable, "-c", *sys.argv[1:]]).returncode)
    """
).strip()

# The generic single-call child: run one to_text attack, report type, full
# message, wall time, and the child's own peak RSS (ru_maxrss, kB on Linux).
# The {CALL} slot is filled with .replace — the rest of the braces are the
# child's own real f-strings and must stay single.
_CHILD_TEMPLATE = r"""
import resource, sys, time
from tors_documents import to_text
t0 = time.perf_counter()
try:
    fmt, text = {CALL}
    print("TYPE=CONVERTED")
    print(f"MSG=resolved={fmt} out_len={len(text)}")
except BaseException as exc:
    print(f"TYPE={type(exc).__name__}")
    print(f"MSG={exc}")
print(f"ELAPSED={time.perf_counter() - t0:.3f}")
print(f"RSS_KB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
"""


def _child(call: str) -> str:
    """One to_text attack call, wrapped in the standard report block."""
    return _CHILD_TEMPLATE.replace("{CALL}", call)


def _probe(code: str, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run one probe child (this venv's own interpreter) and capture its
    outcome verbatim; the child is disposable by design. ``args`` ride as
    argv, never interpolated into the source."""
    return subprocess.run(
        [sys.executable, "-c", _THIN_SPAWNER, textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _probe_report(code: str, *args: str, timeout: float = 60.0) -> str:
    """_probe plus the timeout translation (a child that blocks past the
    deadline IS the hang under test) and the rc gate (the child catches its
    own exceptions, so a nonzero rc is a crash, which is itself a finding
    shape: report it loudly)."""
    try:
        done = _probe(code, *args, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"probe child blocked past the {timeout:g}s deadline — the hang "
            "shape under test; child output lost with the kill"
        )
    assert done.returncode == 0, f"probe child crashed:\n{done.stdout}\n{done.stderr}"
    return done.stdout


def _parse(report: str) -> tuple[str, str, float, float | None]:
    """(TYPE, MSG, peak RSS MiB, elapsed s) from a standard report block."""
    kind = re.search(r"^TYPE=(.*)$", report, re.M)
    msg = re.search(r"^MSG=(.*)$", report, re.M)
    rss = re.search(r"^RSS_KB=(\d+)$", report, re.M)
    elapsed = re.search(r"^ELAPSED=(\d+\.\d+)$", report, re.M)
    assert kind and msg and rss, f"probe did not report type/msg/rss:\n{report}"
    return (
        kind.group(1),
        msg.group(1),
        int(rss.group(1)) / 1024.0,
        float(elapsed.group(1)) if elapsed else None,
    )


def _write_head_rows(path: Path, head: bytes) -> int:
    """An attack file: `head` bytes of attacker-chosen prefix, then the csv
    witness rows, chunked writes so the fixture build never holds the whole
    file. Returns the exact byte size (for the message-hygiene pins)."""
    with open(path, "wb") as f:
        f.write(head)
        f.write(_HEADER)
        remaining = _TARGET_BYTES - len(head) - len(_HEADER)
        chunk = _ROW * 1_000_000  # 5 MB, never the whole file
        full, rem = divmod(remaining, len(chunk))
        for _ in range(full):
            f.write(chunk)
        f.write(_ROW * (rem // len(_ROW)))
    return path.stat().st_size


def _write_sparse(path: Path, magic: bytes, size: int) -> int:
    """A sparse attack file: `magic` then a hole to `size` (reads back zeros;
    ~0 bytes on disk)."""
    with open(path, "wb") as f:
        f.write(magic)
        f.truncate(size)
    return path.stat().st_size


def _write_html(path: Path) -> int:
    """A doctype-led HTML attack file (the backstop lane's happy path)."""
    head = b"<!doctype html><html><body>\n"
    row = b"<p>lorem ipsum dolor sit amet consectetur</p>\n"
    with open(path, "wb") as f:
        f.write(head)
        remaining = _TARGET_BYTES - len(head)
        chunk = row * 100_000
        full, rem = divmod(remaining, len(chunk))
        for _ in range(full):
            f.write(chunk)
        f.write(row * (rem // len(row)))
    return path.stat().st_size


# --- A1: the sniff prefix catches a csv witness inside its 64 KiB window ---


def test_a1_63k_leader_csv_is_metered_during_the_read(tmp_path: Path) -> None:
    """ATTACK: a 48 MiB file of 63 KiB blank lines followed by csv rows,
    named .csv, under max_bytes=None — the whitespace might blind the 64 KiB
    sniff prefix into picking the 512 MiB backstop lane. PINNED SAFE
    BEHAVIOR: the csv witness sits inside the prefix, so the metered lane
    refuses DURING the read at the 32 MiB default ceiling (message names
    'default read ceiling') with RSS well under the file size's next
    doubling."""
    path = tmp_path / "attack.csv"
    exact = _write_head_rows(path, b"\n" * (63 * 1024))
    assert exact > _SNIFF_PREFIX
    kind, msg, rss_mib, _ = _parse(
        _probe_report(_child("to_text(path=sys.argv[1], max_bytes=None)"), str(path))
    )
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert "default read ceiling" in msg, (
        f"the refusal did not come from the read-phase metered ceiling "
        f"(the 32 MiB lane) — the sniffer was evaded:\n{msg}"
    )
    assert "32.0 MiB" in msg, msg
    assert rss_mib < 100.0, f"peak RSS {rss_mib:.1f} MiB — the file was buffered:\n{msg}"


# --- A1b: the evasion — a leader past the sniff prefix + a silent name -----


def test_a1b_leader_past_the_sniff_prefix_evades_read_metering(tmp_path: Path) -> None:
    """ATTACK: the same 48 MiB csv behind an 80 KiB whitespace leader (the
    csv witness entirely outside the 64 KiB sniff prefix), named with NO
    extension (the doctrine's own 'temp files with no suffix' case). The
    prefix is content-blind and the name is silent, so pre-F-1-fix
    provisional_read_ceiling could not tell the lane and picked the 512 MiB
    backstop; the fix errs toward metering on unresolvable prefixes. PINNED
    SAFE BEHAVIOR (per the PR's contract that metered-lane content 'refuses
    during the read at the 32 MiB ceiling'): the read-phase
    refusal fires with the 'default read ceiling' message and RSS stays near
    the 32 MiB stop, not the file size. RED = the sniffer was evaded: the
    full attacker file is buffered under the backstop lane and the refusal
    only comes from the core's post-read check."""
    path = tmp_path / "attackpayload"  # no extension: the name stays silent
    exact = _write_head_rows(path, b"\n" * (80 * 1024))
    kind, msg, rss_mib, _ = _parse(
        _probe_report(_child("to_text(path=sys.argv[1], max_bytes=None)"), str(path))
    )
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert "default read ceiling" in msg, (
        f"FINDING F-1 (red): the read-phase 32 MiB metering was evaded — the "
        f"prefix is content-blind past 64 KiB and the extensionless name "
        f"gives resolve() nothing, so provisional_read_ceiling picked the "
        f"512 MiB backstop lane and buffered the FULL {exact} byte file "
        f"(peak RSS {rss_mib:.1f} MiB) before the core's post-read check "
        f"refused. Refusal actually seen:\n{msg}"
    )
    assert rss_mib < 150.0, (
        f"peak RSS {rss_mib:.1f} MiB for a 48 MiB file — evidence reported, "
        f"the message above is the discriminator"
    )


# --- A1c: the control — the extension fallback is the only remaining guard -


def test_a1c_a_csv_name_meters_a_blinded_prefix(tmp_path: Path) -> None:
    """ATTACK (control for A1b): A1b's exact bytes — 80 KiB whitespace
    leader, csv witness outside the sniff prefix — but named .csv, under
    None. The prefix resolves nothing, so the ONLY thing standing between
    this file and the 512 MiB backstop lane is the extension fallback in
    resolve(). PINNED SAFE BEHAVIOR: the name meters it — read-phase
    refusal at the 32 MiB default ceiling, RSS near the lane stop."""
    path = tmp_path / "attack.csv"
    _write_head_rows(path, b"\n" * (80 * 1024))
    kind, msg, rss_mib, _ = _parse(
        _probe_report(_child("to_text(path=sys.argv[1], max_bytes=None)"), str(path))
    )
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert "default read ceiling" in msg, (
        f"the .csv name failed to meter a content-blinded prefix:\n{msg}"
    )
    assert rss_mib < 100.0, f"peak RSS {rss_mib:.1f} MiB — the file was buffered:\n{msg}"


# --- A2: a %PDF content marker must beat the .csv name ----------------------


def test_a2_pdf_content_marker_wins_over_the_csv_name(tmp_path: Path) -> None:
    """ATTACK: a 48 MiB file whose bytes lead with '%PDF-1.4' and whose body
    is csv rows, named .csv, under None: if the NAME won, the bytes would
    ride the amplifying anydoc csv lane (~146x measured) and only the
    post-read 32 MiB check would catch it. PINNED SAFE BEHAVIOR: content
    beats name at every resolution point (prefix sniff, full resolve), so
    the file routes to the pdf_oxide lane and dies as a pdf parse refusal,
    cleanly, with RSS ~ one input buffer, never the csv lane's message."""
    path = tmp_path / "tricky.csv"
    _write_head_rows(path, b"%PDF-1.4\n")
    kind, msg, rss_mib, _ = _parse(
        _probe_report(_child("to_text(path=sys.argv[1], max_bytes=None)"), str(path))
    )
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert "anydoc engine lane" not in msg, (
        f"the .csv name routed %PDF content to the amplifying csv lane:\n{msg}"
    )
    assert "default read ceiling" not in msg, (
        f"the pdf lane was metered by the read-phase guess (content lost):\n{msg}"
    )
    assert rss_mib < 300.0, f"pdf_oxide buffered {rss_mib:.1f} MiB for a 48 MiB input"
    print(f"[a2] pdf lane refusal: {msg}")  # full message for the report log


# --- A3: truncated-ZIP garbage: full read under the backstop, clean refusal


def test_a3_truncated_zip_refuses_during_the_read_at_the_metered_ceiling(tmp_path: Path) -> None:
    """ATTACK: a 48 MiB file starting with a ZIP local-file header but never
    containing a central directory (zeros), named .bin, under None: the
    prefix cannot resolve the lane at all. UPDATED after the F-1 fix (an
    unresolvable prefix now errs toward metering): the read must refuse at
    the 32 MiB provisional ceiling DURING the read — not buffer the whole
    file under the 512 MiB backstop and defer the refusal to resolve()'s
    post-read UnknownFormat. SECURITY QUESTION: bounded RSS, clean typed
    exception, and the read-phase message shape."""
    path = tmp_path / "payload.bin"
    _write_sparse(path, b"PK\x03\x04\x14\x00\x00\x00", _TARGET_BYTES)
    kind, msg, rss_mib, _ = _parse(
        _probe_report(_child("to_text(path=sys.argv[1], max_bytes=None)"), str(path))
    )
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert "default read ceiling" in msg, f"not the read-phase refusal:\n{msg}"
    assert rss_mib < 100.0, (
        f"peak RSS {rss_mib:.1f} MiB for a read capped at the 32 MiB metered "
        f"ceiling — the backstop buffered the file:\n{msg}"
    )
    print(f"[a3] refusal during the read (path echo is caller-supplied): {msg}")


# --- A4: four threads multiplying the evaded read --------------------------


def test_a4_four_threads_of_the_evading_read_stay_bounded(tmp_path: Path) -> None:
    """ATTACK: four THREADS in one child each call to_text under None on a
    distinct A1b file (48 MiB evading csv, extensionless name). The read
    runs GIL-free, so the four reads genuinely overlap; if the evaded lane's
    buffer multiplied into anything uncontrolled (or the threads' buffers
    plus engine work blew past ~4x the input), this is where it shows.
    PINNED SAFE BEHAVIOR: every thread ends in the clean ValueError refusal,
    and the child's peak RSS stays bounded — under a GiB against the by-design
    4 x 512 MiB worst case — with an RLIMIT_AS belt so a regression dies in
    the child, not on the box."""
    paths = [tmp_path / f"t{i}" for i in range(4)]
    for p in paths:
        _write_head_rows(p, b"\n" * (80 * 1024))
    child = textwrap.dedent(
        r"""
        import resource, sys, threading
        resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024**2, 1536 * 1024**2))
        from tors_documents import to_text
        results = []
        def attack(p):
            try:
                to_text(path=p, max_bytes=None)
                results.append((p, "CONVERTED", ""))
            except BaseException as exc:
                results.append((p, type(exc).__name__, str(exc)))
        threads = [threading.Thread(target=attack, args=(p,)) for p in sys.argv[1:5]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for p, kind, msg in results:
            print(f"OUT {kind} {msg[:90]}")
        print(f"RSS_KB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
        """
    )
    report = _probe_report(child, *[str(p) for p in paths])
    outcomes = re.findall(r"^OUT (\S+) (.*)$", report, re.M)
    rss = re.search(r"^RSS_KB=(\d+)$", report, re.M)
    assert rss, report
    rss_mib = int(rss.group(1)) / 1024.0
    assert len(outcomes) == 4, f"thread outcomes lost:\n{report}"
    for kind, _ in outcomes:
        assert kind == "ValueError", f"a thread escaped the refusal:\n{report}"
    assert rss_mib < 1024.0, (
        f"4 concurrent evading reads peaked at {rss_mib:.1f} MiB — more than "
        f"4 input buffers plus sane overhead:\n{report}"
    )
    print(f"[a4] measured peak RSS {rss_mib:.1f} MiB for 4 x 48 MiB evading reads")


# --- A5a: /proc/self/mem — the lying-stat special file ----------------------


def test_a5a_proc_self_mem_is_a_typed_fast_refusal_not_a_hang() -> None:
    """ATTACK: to_text(path='/proc/self/mem', max_bytes=None) — procfs
    reports S_ISREG with st_size=0, so every stat-shaped gate passes and the
    bounded read meets a file whose reads fail or mislead. PINNED SAFE
    BEHAVIOR: a typed OSError/ValueError surfaces quickly (the read's own
    EIO, or the clean refusal after an EOF-shaped read); a hang past the 60 s
    deadline is the finding. Skipped only if the child cannot even open its
    own /proc/self/mem."""
    child = textwrap.dedent(
        r"""
        import sys, time
        from tors_documents import to_text
        try:
            open("/proc/self/mem", "rb").close()
        except OSError as exc:
            print("TYPE=SKIP")
            print(f"MSG=preflight open failed: {exc}")
            sys.exit(0)
        t0 = time.perf_counter()
        try:
            to_text(path="/proc/self/mem", max_bytes=None)
            print("TYPE=CONVERTED")
            print("MSG=resolved")
        except BaseException as exc:
            print(f"TYPE={type(exc).__name__}")
            print(f"MSG={exc}")
        print(f"ELAPSED={time.perf_counter() - t0:.3f}")
        """
    )
    report = _probe_report(child)
    kind = re.search(r"^TYPE=(.*)$", report, re.M)
    assert kind, report
    if kind.group(1) == "SKIP":
        pytest.skip(f"/proc/self/mem unopenable in the child: {report}")
    assert kind.group(1) in {"OSError", "ValueError"}, (
        f"unexpected outcome class: {report}"
    )
    elapsed = re.search(r"^ELAPSED=(\d+\.\d+)$", report, re.M)
    assert elapsed and float(elapsed.group(1)) < 10.0, (
        f"the /proc/self/mem attack did not end quickly: {report}"
    )
    print(f"[a5a] {report.strip()}")


# --- A5b: a no-writer FIFO must be refused before open(2) blocks ------------


def test_a5b_fifo_is_refused_before_open_blocks(tmp_path: Path) -> None:
    """ATTACK: os.mkfifo then to_text(path=<fifo>, max_bytes=None): a FIFO
    with no writer blocks inside open(2) with the GIL released, so any path
    that let the kind check slip past the stat would hang the caller's
    thread forever. PINNED SAFE BEHAVIOR: the pre-open stat refuses the kind
    with a ValueError naming 'a FIFO', within milliseconds; the 60 s
    deadline is the hang detector and a hit there is a finding."""
    fifo = tmp_path / "feed"
    os.mkfifo(fifo)
    child = textwrap.dedent(
        r"""
        import sys, time
        from tors_documents import to_text
        t0 = time.perf_counter()
        try:
            to_text(path=sys.argv[1], max_bytes=None)
            print("TYPE=CONVERTED")
            print("MSG=resolved")
        except BaseException as exc:
            print(f"TYPE={type(exc).__name__}")
            print(f"MSG={exc}")
        print(f"ELAPSED={time.perf_counter() - t0:.3f}")
        """
    )
    report = _probe_report(child, str(fifo))
    kind = re.search(r"^TYPE=(.*)$", report, re.M)
    msg = re.search(r"^MSG=(.*)$", report, re.M)
    elapsed = re.search(r"^ELAPSED=(\d+\.\d+)$", report, re.M)
    assert kind and msg and elapsed, report
    assert kind.group(1) == "ValueError", f"the FIFO was not a typed refusal: {report}"
    assert "regular file" in msg.group(1) and "FIFO" in msg.group(1), (
        f"refusal lost the kind: {msg.group(1)}"
    )
    assert float(elapsed.group(1)) < 10.0, f"refusal was not fast: {report}"


# --- A6: the data= lane adds no copy beyond the bytes themselves ------------


def test_a6_data_lane_under_none_adds_no_unbounded_copy(tmp_path: Path) -> None:
    """ATTACK: to_text(data=<48 MiB whitespace-led csv bytes>, max_bytes=None)
    — the data lane has no two-phase read and no backstop of its own ('no
    new backstop: resident bytes cannot grow past themselves'), so the
    question is whether the lane adds any copy beyond the caller's bytes
    before the core's post-read check refuses. PINNED SAFE BEHAVIOR: clean
    ValueError, peak RSS < 2x input plus a fixed interpreter slack (the one
    to_vec copy), never an engine run."""
    fixture = tmp_path / "fixture.csv"
    _write_head_rows(fixture, b"\n" * (80 * 1024))
    child = textwrap.dedent(
        r"""
        import resource, sys, time
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
        from tors_documents import to_text
        data = open(sys.argv[1], "rb").read()  # one buffer, sized by fstat
        t0 = time.perf_counter()
        try:
            to_text(data=data, max_bytes=None)
            print("TYPE=CONVERTED")
            print("MSG=resolved")
        except BaseException as exc:
            print(f"TYPE={type(exc).__name__}")
            print(f"MSG={exc}")
        print(f"ELAPSED={time.perf_counter() - t0:.3f}")
        print(f"RSS_KB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
        """
    )
    report = _probe_report(child, str(fixture))
    kind, msg, rss_mib, _ = _parse(report)
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert rss_mib < 2 * 48 + 40, (
        f"peak RSS {rss_mib:.1f} MiB — more than the bytes plus one copy:\n{msg}"
    )
    print(f"[a6] measured peak RSS {rss_mib:.1f} MiB (input + one copy): {msg}")


def test_a6b_data_lane_explicit_budget_refuses_precopy(tmp_path: Path) -> None:
    """ATTACK: the same 48 MiB bytes with max_bytes=1024 — if the budget ran
    after the copy, the lane would memcpy the whole input before refusing.
    PINNED SAFE BEHAVIOR: the refusal is instant and its message is the
    pre-gate's own ('checked before any work runs'); the measured peak
    (65-94 MiB across runs on this box) rides pyo3's argument-extraction
    transient of the already-resident input, never a library-side copy of
    it, and never an engine run."""
    fixture = tmp_path / "fixture.csv"
    _write_head_rows(fixture, b"\n" * (80 * 1024))
    child = textwrap.dedent(
        r"""
        import resource, sys, time
        from tors_documents import to_text
        data = open(sys.argv[1], "rb").read()
        t0 = time.perf_counter()
        try:
            to_text(data=data, max_bytes=1024)
            print("TYPE=CONVERTED")
            print("MSG=resolved")
        except BaseException as exc:
            print(f"TYPE={type(exc).__name__}")
            print(f"MSG={exc}")
        print(f"ELAPSED={time.perf_counter() - t0:.3f}")
        print(f"RSS_KB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
        """
    )
    kind, msg, rss_mib, elapsed = _parse(_probe_report(child, str(fixture)))
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert "input ceiling is 1024 bytes" in msg, (
        f"the refusal is not the pre-gate's own message (a copy or engine run "
        f"may have preceded it):\n{msg}"
    )
    assert elapsed is not None and elapsed < 5.0, "the pre-copy refusal was not fast"
    assert rss_mib < 2 * 48 + 40, (
        f"peak RSS {rss_mib:.1f} MiB — past the input plus an extraction transient:\n{msg}"
    )
    print(f"[a6b] {msg}")


# --- A7: message hygiene of the over-ceiling refusals -----------------------


def test_a7_over_ceiling_messages_hide_paths_and_exact_sizes(tmp_path: Path) -> None:
    """ATTACK: harvest BOTH over-ceiling refusals under None (the read-phase
    32 MiB lane stop and the post-read lane check) from 48 MiB attack files
    and scan them for operational intelligence an attacker would want: the
    absolute filesystem path (server layout disclosure if messages are
    logged and replayed cross-tenant) or the exact byte size (tuning
    fidelity for a size-gated filter). PINNED SAFE BEHAVIOR: the messages
    name ceilings in humanized MiB only — no path, no byte-exact figure.
    Both full messages are dumped to the log for the report."""
    read_phase = tmp_path / "attack.csv"
    post_read = tmp_path / "attackpayload"
    exact_read = _write_head_rows(read_phase, b"\n" * (63 * 1024))
    exact_post = _write_head_rows(post_read, b"\n" * (80 * 1024))
    child = textwrap.dedent(
        r"""
        import sys
        from tors_documents import to_text
        for label, p in (("READPHASE", sys.argv[1]), ("POSTREAD", sys.argv[2])):
            try:
                to_text(path=p, max_bytes=None)
                print(f"{label} TYPE=CONVERTED")
            except BaseException as exc:
                print(f"{label} TYPE={type(exc).__name__}")
                print(f"{label} MSG={exc}")
        """
    )
    report = _probe_report(child, str(read_phase), str(post_read))
    read_msg = re.search(r"^READPHASE MSG=(.*)$", report, re.M)
    post_msg = re.search(r"^POSTREAD MSG=(.*)$", report, re.M)
    assert read_msg and post_msg, report
    for label, msg, exact in (
        ("read-phase", read_msg.group(1), exact_read),
        ("post-read", post_msg.group(1), exact_post),
    ):
        print(f"[a7] full {label} message: {msg}")
        assert str(tmp_path) not in msg, f"{label} message leaked the attacker path: {msg}"
        assert "/tmp" not in msg and "/home" not in msg, (
            f"{label} message leaked a filesystem path: {msg}"
        )
        assert str(exact) not in msg and f"{exact:,}" not in msg, (
            f"{label} message leaked the exact byte size ({exact}): {msg}"
        )


# --- A8: unresolvable content can no longer buy the backstop lane ----------


def test_a8_sparse_unresolvable_file_meters_during_the_read(tmp_path: Path) -> None:
    """ATTACK: a 513 MiB-logical sparse file (0 bytes on disk) of pure zeros,
    extensionless name, under None: the content is unresolvable and the name
    is silent, so pre-F-1 the attacker alone selected the 512 MiB backstop
    lane and buffered 530 MiB of RSS from a 0-byte file. UPDATED after the
    F-1 fix (an unresolvable prefix errs toward metering): the read must
    refuse at the 32 MiB provisional ceiling DURING the read — bounded,
    typed, no kill, no backstop-sized buffer — with an RLIMIT_AS belt so an
    unbounded regression dies as MemoryError in the child. GREEN = the
    attacker can no longer select the big-buffer lane with silent content."""
    path = tmp_path / "sparse.bin"
    exact = _write_sparse(path, b"", 513 * 1024 * 1024)
    assert exact == 513 * 1024 * 1024
    blocks = path.stat().st_blocks * 512
    child = textwrap.dedent(
        r"""
        import resource, sys, time
        resource.setrlimit(resource.RLIMIT_AS, (1280 * 1024**2, 1280 * 1024**2))
        from tors_documents import to_text
        t0 = time.perf_counter()
        try:
            to_text(path=sys.argv[1], max_bytes=None)
            print("TYPE=CONVERTED")
            print("MSG=resolved")
        except BaseException as exc:
            print(f"TYPE={type(exc).__name__}")
            print(f"MSG={exc}")
        print(f"ELAPSED={time.perf_counter() - t0:.3f}")
        print(f"RSS_KB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
        """
    )
    kind, msg, rss_mib, _ = _parse(_probe_report(child, str(path)))
    assert kind == "ValueError", f"not a clean refusal: {kind}: {msg}"
    assert "default read ceiling" in msg and "32.0 MiB" in msg, (
        f"the unresolvable sparse file was not refused at the 32 MiB metered "
        f"ceiling during the read (message: {msg!r}) — the backstop lane is "
        f"reachable again by silent content (F-1 regression)"
    )
    assert rss_mib < 200.0, (
        f"peak RSS {rss_mib:.1f} MiB for a read capped at 32 MiB — the "
        f"backstop buffered attacker content again:\n{msg}"
    )
    print(
        f"[a8] sparse file: {blocks} bytes on disk -> refusal during the "
        f"read at the 32 MiB metered ceiling, child peak RSS {rss_mib:.1f} MiB"
    )


# --- A9: the backstop lane CONVERTS: the measured cost of the attacker's lane


def test_a9_html_lane_meters_during_the_read_under_none(tmp_path: Path) -> None:
    """ATTACK: a 48 MiB doctype-led HTML file under None: pre-F-2-fix the
    content marker resolved the unmetered Html2Md lane and the conversion
    ran at a measured ~23x input (1118.1 MiB peak RSS for 48 MiB, no
    refusal at all). UPDATED after the F-2 fix (the HTML lane now meters):
    the read must refuse at the 32 MiB lane ceiling DURING the read — the
    attacker can no longer buy an uncapped conversion on the HTML lane.
    RED = the ~23x amplification vector is back."""
    path = tmp_path / "page.html"
    _write_html(path)
    kind, msg, rss_mib, _ = _parse(
        _probe_report(_child("to_text(path=sys.argv[1], max_bytes=None)"), str(path))
    )
    assert kind == "ValueError", f"unexpected outcome: {kind}: {msg}"
    assert "default read ceiling" in msg, f"not the read-phase refusal:\n{msg}"
    assert "32.0 MiB" in msg, msg
    assert rss_mib < 200.0, (
        f"the HTML lane buffered {rss_mib:.1f} MiB for a read capped at the "
        f"32 MiB ceiling — the amplification vector is back:\n{msg}"
    )
    print(
        f"[a9] refusal during the read at the 32 MiB HTML-lane ceiling; "
        f"peak RSS {rss_mib:.1f} MiB"
    )
