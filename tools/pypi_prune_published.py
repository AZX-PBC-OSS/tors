"""Prune distribution files from a dist/ directory that PyPI already hosts.

The publish workflow's idempotency gate. The motivating failure (the 2026-09-26
double-dispatch of publish.yml against v0.15.1, already live on PyPI since that
morning's run): every build leg goes green, ~20 minutes of matrix burns, and
then all three publish jobs die red — ``uv publish`` uploads optimistically and
PyPI answers ``400 File already exists``, because PyPI never allows filename
reuse (https://pypi.org/help/#file-name-reuse): not for identical content, not
ever. A re-dispatch against an already-published release can therefore never go
green, and pins a red X on main for a release that actually shipped intact.

Why filename matching, not `uv publish --check-url`: that flag compares file
*hashes* against the index, which recovers a partial upload of the SAME build —
but maturin wheels are not byte-reproducible, so a re-dispatch rebuilds
artifacts whose hashes differ from the released ones. Filenames are the only
stable identity here: if the exact filename is on PyPI, that release already
shipped it, full stop (PyPI's no-reuse rule means the file cannot be superseded
in place either — a same-named, hash-differing local file is exactly the case
that fails upload). Pruning by filename turns a re-dispatch into a no-op, and a
partial upload into "publish just the missing files" — the same recovery
--check-url documents, one level up, covering the rebuilt-artifact case it
cannot see.

The script is deliberately stdlib-only (the publish jobs download artifacts but
do not set up a project environment; python3 on the runner is all they have)
and fetches one JSON document per (project, version) in the dist directory:
a 404 means nothing of that version is on PyPI yet, so nothing is pruned.

Usage: ``python3 tools/pypi_prune_published.py DIST`` — prints what it pruned
and what remains; exits nonzero on anything other than a clean prune (the
workflow must not publish blind if the existence check itself failed).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PYPI_JSON_API = "https://pypi.org/pypi"


def parse_dist_filename(filename: str) -> tuple[str, str] | None:
    """Split a distribution filename into (project, version).

    Per the wheel / sdist filename specs, both spellings escape ``-`` in the
    project name and version as ``_``, so the first two dash-separated fields
    are exactly (project, version) — for wheels
    ``{distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl`` and
    for sdists ``{name}-{version}.tar.gz``. Returns None for anything that
    does not parse (it is not our artifact; the caller keeps it).
    """
    head = filename.removesuffix(".whl").removesuffix(".tar.gz")
    parts = head.split("-", 2)
    if len(parts) < 2:
        return None
    project, version = parts[0], parts[1]
    if not project or not version:
        return None
    return project, version


def normalize_project(name: str) -> str:
    """PEP 503 normalization: PyPI's JSON API routes on this canonical form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch_release_filenames(project: str, version: str) -> set[str] | None:
    """The set of filenames PyPI hosts for (project, version).

    None means "version unknown to PyPI" (404): nothing is published under
    that version, so nothing can be pruned. Any other HTTP problem raises —
    the caller must not prune (and so must not publish) on a check that
    silently failed.
    """
    url = f"{PYPI_JSON_API}/{urllib.parse.quote(project)}/{urllib.parse.quote(version)}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "tors-publish-prune"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    # The version-scoped endpoint lists the files under "urls"; the
    # package-scoped one nests them per release under "releases". Accepting
    # both keeps this correct if the lookup ever switches scope.
    entries = payload.get("urls") or payload.get("files") or []
    return {entry["filename"] for entry in entries}


def prune_published(dist_dir: Path, fetch=fetch_release_filenames) -> tuple[list[str], list[str]]:
    """Delete dist files PyPI already hosts; return (removed, kept).

    ``fetch`` is injectable so the pytest suite can exercise the pruning logic
    against an in-memory registry instead of the live PyPI API.
    """
    filenames = sorted(entry.name for entry in dist_dir.iterdir() if entry.is_file())
    # One fetch per unique (project, version), not per file: the 8-wheel +
    # 1-sdist base release is one version, one JSON document.
    releases = {parsed: None for f in filenames if (parsed := parse_dist_filename(f)) is not None}
    existing: dict[tuple[str, str], set[str] | None] = {}
    for project, version in releases:
        existing[(project, version)] = fetch(project, version)

    removed: list[str] = []
    kept: list[str] = []
    for filename in filenames:
        parsed = parse_dist_filename(filename)
        hosted = existing.get(parsed) if parsed is not None else None
        if hosted is not None and filename in hosted:
            (dist_dir / filename).unlink()
            removed.append(filename)
        else:
            kept.append(filename)
    return removed, kept


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} DIST", file=sys.stderr)
        return 2
    dist_dir = Path(argv[1])
    if not dist_dir.is_dir():
        print(f"error: {dist_dir} is not a directory", file=sys.stderr)
        return 2

    removed, kept = prune_published(dist_dir)
    for filename in removed:
        print(f"already on PyPI, pruned: {filename}")
    for filename in kept:
        print(f"not on PyPI, will publish: {filename}")
    if not removed:
        print("nothing pruned: this is not a re-dispatch of published files")
    if not kept:
        # Not an error: the whole dist was already shipped. The workflow's
        # Publish step reads this state (dist/ is empty) and skips
        # `uv publish`, which cannot express "no files" as success.
        print("everything in dist/ is already on PyPI: nothing to publish (idempotent re-run)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
