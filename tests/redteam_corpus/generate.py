"""One-time deterministic generator for the red-team corpus payloads in
this directory. Run from the repo root: ``uv run --no-sync python
tests/redteam_corpus/generate.py``. The OUTPUT IS COMMITTED and the
canary (tests/test_redteam_corpus.py) reads the committed files — the
tests never regenerate anything; this script is provenance only, so a
corpus payload's shape is always explainable. It is idempotent: re-run
reproduces byte-identical files.

Each payload is the minimal repro shape for one manifest entry
(manifest.toml maps payload -> surface -> invariant -> status); sizes
are chosen so a RELAPSE of the fixed defects crosses the manifest's
wall limits at corpus scale (e.g. issue-92's flood is 1000 headers:
the pre-fix cubic measured 2.6s, over the 2s canary limit, while the
fix measures 0.5ms) — see the manifest for each entry's numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

out = Path(__file__).parent

# issue-91: the shingle-flood stream (20k tokens; the budget repro pairs
# it with shingle_size=10000, the under-ceiling canary with 1000).
(out / "issue-91-shingle-flood.txt").write_text(
    " ".join(f"w{i}" for i in range(20_000)), encoding="utf-8"
)

# issue-92: the PEM flood, all-different header words, no verifying END.
n = 1000
(out / "issue-92-pem-flood.txt").write_text(
    "".join(f"-----BEGIN K{i} PRIVATE KEY-----\n" for i in range(n))
    + "".join(f"-----END L{i} PRIVATE KEY-----\n" for i in range(n)),
    encoding="utf-8",
)

# issue-99: the leaked credential shapes (all redact on main now).
(out / "issue-99-credentials.txt").write_text(
    "auth failed for gho_16C7y42VZ6TyZwTEZwZ0Czfs6k5F9wJZ0RZ8x\n"
    "auth failed for ghu_16C7y42VZ6TyZwTEZwZ0Czfs6k5F9wJZ0RZ8x\n"
    "auth failed for ghs_16C7e42F292c6912E7710c838347Ae178B4a1\n"
    "auth failed for ghr_16C7y42VZ6TyZwTEZwZ0Czfs6k5F9wJZ0RZ8x\n"
    "auth failed for glpat-abcdefghij0123456789\n"
    "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQdGBGX...\n-----END PGP PRIVATE KEY BLOCK-----\n",
    encoding="utf-8",
)

# issue-100: escape-glued keys (the three spellings the boundary rule
# recognizes post-fix).
(out / "issue-100-escaped-newline.txt").write_text(
    "err:\\nsk-proj-abcdefghijklmnopqrstuvwx", encoding="utf-8"
)
(out / "issue-100-uXXXX.txt").write_text(
    "invalid key \\u0027sk-proj-abcdefghijklmnopqrstuvwx", encoding="utf-8"
)
(out / "issue-100-pctXX.txt").write_text("url?key%3DAIza" + "a" * 35, encoding="utf-8")

# issue-101: the masked O(matches x value_len) shape (400k matches; the
# value is a runner param: "x" * 400_000 — pre-fix ~5s, over the limit).
(out / "issue-101-masked-dense.txt").write_text("a" * 400_000, encoding="utf-8")

# issue-102: the overlap whitespace run (300k: pre-fix ~11s over the 5s
# limit; post-fix ~39ms).
(out / "issue-102-whitespace-run.txt").write_text("\n" * 300_000, encoding="utf-8")

# issue-103: the separator-as-its-own-chunk repro (realized-cut path).
(out / "issue-103-separator-chunk.txt").write_text("a\n\n\n\nb", encoding="utf-8")

# issue-108: the fresh (uncached) non-ASCII string shape, 64 KiB.
(out / "issue-108-fresh-nonascii.txt").write_text(
    ("éü中īñ" * 13_107)[:65_536], encoding="utf-8"
)

# issue-110: a PEM key glued to a preceding block's END marker.
(out / "issue-110-glued-pem.txt").write_text(
    "-----END CERTIFICATE----------BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
    encoding="utf-8",
)

# issue-111: backtick runs of increasing length (the code-span lifter).
runs = 800
(out / "issue-111-backticks.html").write_text(
    "<html><body><p>" + "".join(("`" * k + "x") for k in range(1, runs)) + "</p></body></html>",
    encoding="utf-8",
)

# issue-112: the payload is trivial; the lying Sequence is built by the
# runner from the manifest params.
(out / "issue-112-lying-len.txt").write_text("x", encoding="utf-8")

# issue-113: the shared-ref schema's DOCUMENT (the doubling builder is a
# runner param: the structure is depth-tiny; the walk is the bomb).
(out / "issue-113-shared-ref-schema.txt").write_text("{}", encoding="utf-8")

# issue-114: short key, huge value (runner params: value_char "y" x
# value_repeat 125_000 -> a 1MB value against 1000 matches).
(out / "issue-114-amplification.txt").write_text("a" * 1000, encoding="utf-8")

# issue-115: the enum-miss document (members are runner params).
(out / "issue-115-deadline-escape.txt").write_text("b" * 1500, encoding="utf-8")

# finding-1: the escape spellings the boundary rule does NOT recognize
# (\xHH hex escapes, octal \NNN) glued to valid keys — the #100 residual.
(out / "finding-1-xhh-octal-glued.txt").write_text(
    "prefix\\x1bsk-abcdefghijklmnopqrstuvwx\n"
    "prefix\\163k-abcdefghijklmnopqrstuvwx\n",
    encoding="utf-8",
)

# finding-2: the separator-material chunk via the remainder/hard-cut path
# (the #103 residual).
(out / "finding-2-separator-remainder-chunk.txt").write_text("aaaa", encoding="utf-8")

# finding-3: the wide-schema quadratic (document with matching keys; the
# properties width is a runner param).
(out / "finding-3-wide-schema.txt").write_text(
    json.dumps({f"k{i}": f"v{i}" for i in range(2500)}), encoding="utf-8"
)

for p in sorted(out.iterdir()):
    if p.is_file() and p.suffix != ".py" and p.name != "manifest.toml":
        print(f"{p.name}: {p.stat().st_size} bytes")
