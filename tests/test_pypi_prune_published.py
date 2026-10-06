"""Contract gate for ``tools/pypi_prune_published.py``: the publish workflow's
idempotency prune never publishes a file PyPI already hosts, and never prunes
a file it does not.

Why this is a gate and not a script detail: the motivating incident (the
2026-09-26 double-dispatch of publish.yml against the already-released 0.15.1)
failed all three publish jobs with "File already exists" after ~20 minutes of
green build legs — a re-dispatch of a published version could NEVER go green.
The prune is the fix's load-bearing step; these tests pin its two failure
directions (over-pruning hides a genuinely-new file forever; under-pruning
re-introduces the 400), the filename grammar both wheel and sdist specs
define, and the 404 path (nothing published yet → keep everything).

All tests run against an in-memory registry injected as the ``fetch``
dependency — the suite must stay offline (ci.yml has no PyPI egress needs)
and deterministic.
"""

from __future__ import annotations

from pathlib import Path

from pypi_prune_published import (
    normalize_project,
    parse_dist_filename,
    prune_published,
)


def _write_dist(dist: Path, *filenames: str) -> None:
    for filename in filenames:
        (dist / filename).write_bytes(b"not a real artifact; the prune never reads contents")


def _registry_fetch(known: dict[tuple[str, str], set[str]]):
    """A ``fetch`` stub over an in-memory (project, version) -> files map."""

    def fetch(project: str, version: str) -> set[str] | None:
        return known.get((project, version))

    return fetch


class TestParseDistFilename:
    def test_wheel_project_and_version(self) -> None:
        # The payload project's name carries underscores in the filename
        # (PEP 491 escapes '-'); both fields still split on the first dash.
        assert parse_dist_filename("tors_documents-0.15.1-cp310-abi3-win_arm64.whl") == (
            "tors_documents",
            "0.15.1",
        )
        assert parse_dist_filename("tors-0.15.1-cp310-abi3-macosx_10_12_x86_64.whl") == (
            "tors",
            "0.15.1",
        )

    def test_wheel_with_build_tag(self) -> None:
        # The optional build tag sits after the version; version stays field 2.
        assert parse_dist_filename("tors-0.15.1-1-cp310-abi3-manylinux_2_17_x86_64.whl") == (
            "tors",
            "0.15.1",
        )

    def test_sdist(self) -> None:
        assert parse_dist_filename("tors-0.15.1.tar.gz") == ("tors", "0.15.1")

    def test_unparseable_is_none(self) -> None:
        # Stray files (e.g. a metadata fragment) are kept, never fed to the registry.
        assert parse_dist_filename("README") is None


class TestNormalizeProject:
    def test_pep503_forms_route_to_the_same_registry_name(self) -> None:
        assert normalize_project("tors_documents") == "tors-documents"
        assert normalize_project("Tors") == "tors"


class TestPrunePublished:
    def test_prunes_only_files_pypi_already_hosts(self, tmp_path: Path) -> None:
        _write_dist(
            tmp_path,
            "tors-0.15.1-cp310-abi3-win_amd64.whl",
            "tors-0.15.1.tar.gz",
            "tors-0.16.0-cp310-abi3-win_amd64.whl",  # genuinely new: must survive
        )
        fetch = _registry_fetch(
            {
                ("tors", "0.15.1"): {
                    "tors-0.15.1-cp310-abi3-win_amd64.whl",
                    "tors-0.15.1.tar.gz",
                }
            }
        )
        removed, kept = prune_published(tmp_path, fetch=fetch)
        assert removed == ["tors-0.15.1-cp310-abi3-win_amd64.whl", "tors-0.15.1.tar.gz"]
        assert kept == ["tors-0.16.0-cp310-abi3-win_amd64.whl"]
        assert [p.name for p in tmp_path.iterdir()] == kept

    def test_404_version_prunes_nothing(self, tmp_path: Path) -> None:
        # The first-dispatch case: the version is unknown to PyPI, so the
        # prune must be a no-op and the release publishes in full.
        _write_dist(tmp_path, "tors-0.16.0.tar.gz", "tors-0.16.0-cp310-abi3-win_arm64.whl")
        fetch = _registry_fetch({})  # every lookup is a 404
        removed, kept = prune_published(tmp_path, fetch=fetch)
        assert removed == []
        assert len(kept) == 2

    def test_hash_differing_rebuild_is_still_pruned(self, tmp_path: Path) -> None:
        # The re-dispatch case in its sharpest form: the local wheel was
        # REBUILT and hashes differently from the released file, so any
        # hash-based check would re-upload and die on the 400. Filenames are
        # the identity PyPI enforces, so this must prune.
        _write_dist(tmp_path, "tors_documents-0.15.1-cp310-abi3-win_arm64.whl")
        fetch = _registry_fetch(
            {("tors_documents", "0.15.1"): {"tors_documents-0.15.1-cp310-abi3-win_arm64.whl"}}
        )
        removed, _kept = prune_published(tmp_path, fetch=fetch)
        assert removed == ["tors_documents-0.15.1-cp310-abi3-win_arm64.whl"]

    def test_mixed_projects_fetch_once_each(self, tmp_path: Path) -> None:
        # One JSON document per (project, version), not per file: the base
        # release's 8 wheels + sdist are ONE registry lookup.
        _write_dist(tmp_path, "tors-0.16.0.tar.gz", "tors_documents-0.16.0-cp310-abi3-win_x64.whl")
        calls: list[tuple[str, str]] = []

        def fetch(project: str, version: str) -> set[str] | None:
            calls.append((project, version))
            return None

        prune_published(tmp_path, fetch=fetch)
        assert sorted(calls) == [("tors", "0.16.0"), ("tors_documents", "0.16.0")]

    def test_all_pruned_leaves_empty_dist(self, tmp_path: Path) -> None:
        # The full re-dispatch case: everything is already shipped, dist/
        # ends empty, and the workflow's Publish step reads exactly this
        # state to skip `uv publish` (which cannot express "no files").
        _write_dist(tmp_path, "tors-0.15.1.tar.gz")
        fetch = _registry_fetch({("tors", "0.15.1"): {"tors-0.15.1.tar.gz"}})
        removed, kept = prune_published(tmp_path, fetch=fetch)
        assert removed == ["tors-0.15.1.tar.gz"]
        assert kept == []
        assert list(tmp_path.iterdir()) == []
