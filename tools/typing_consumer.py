"""The typed-surface consumer scratch: the calls a real integrator spells,
fully annotated. Pyright checks this file in strict mode (ci.yml's lint
job) so a stub regression — a wrong annotation, a lost overload, a broken
type import — fails CI instead of the consumer's build. The example calls
are the docs' own examples, with their real outputs asserted; the runtime
behavior itself is pinned by the test suite (tests/test_docs_examples.py
for the doc literals, tests/test_pyi_drift.py for stub-vs-runtime
signatures, which is also where default-value drift is caught — a type
checker cannot see a default's value)."""

from __future__ import annotations

import tors

normalized: str = tors.normalize("line one  \n\n\n\nline two\r\n")
assert normalized == "line one\n\nline two"

spans: list[tuple[int, int]] = tors.chunk_text("cats are cute and cats are fun", 12)
assert spans == [(0, 8), (8, 17), (17, 26), (26, 30)]

hierarchy: list[tuple[int, int]] = tors.chunk_hierarchical(
    "# Title\nIntro paragraph here with some words.\n\n"
    "## Section One\nContent for section one goes here and continues a bit further.\n\n"
    "## Section Two\nMore content for section two, also fairly short.",
    80,
    ["\n## ", "\n\n", ". ", " "],
)
assert hierarchy == [(0, 46), (50, 125), (129, 189)]

scrubbed: str = tors.scrub_log_text("postgres://u:pw@h/db")
assert "***" in scrubbed

valid: bool = tors.json_is_valid('{"a": 1}')
assert valid is True
invalid: bool = tors.json_is_valid("{")
assert invalid is False

decoded: bytes = tors.b64_decode("aGVsbG8=", validate=True)
assert decoded == b"hello"

report = tors.scrub_pii_report("email a@b.com")
redacted: dict[str, int] = report["redacted"]
span_type: str = report["spans"][0]["type"]
assert redacted == {"contact_email": 1}
assert span_type == "contact_email"
assert report["skipped"] == {}

value, actions = tors.repair_json_diagnostics(
    '{"count": "12"}',
    schema={"type": "object", "properties": {"count": {"type": "integer"}}},
)
assert value == {"count": 12}
assert actions[0]["action"] == "coerce"
assert actions[0]["from"] == "12"
assert actions[0]["to"] == 12
