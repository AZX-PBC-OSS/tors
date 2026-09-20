"""The typed-surface consumer scratch: the calls a real integrator spells,
fully annotated, checked by the pyright gate (ci.yml's lint job) so a stub
regression (a wrong annotation, a lost overload, a drifted default) fails
CI and not the consumer's build. Every assert is runtime-true too —
`pytest tests/test_docs_examples.py` pins the behaviors these call shapes
come from — but THIS file's contract is the type check itself: run
`uvx pyright@1.1.414` (the CI pin) with zero errors, strictly."""

from __future__ import annotations

import tors

normalized: str = tors.normalize("line one  \n\n\n\nline two\r\n")
assert normalized == "line one\n\nline two"

spans: list[tuple[int, int]] = tors.chunk_text("cats are cute and cats are fun", 12)
assert spans == [(0, 8), (8, 17), (17, 26), (26, 30)]

hierarchy: list[tuple[int, int]] = tors.chunk_hierarchical(
    "# Title\nIntro paragraph here with some words.\n\n## Section One\nmore",
    80,
    ["\n## ", "\n\n", ". ", " "],
)
assert hierarchy[0] == (0, 46)

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

value, actions = tors.repair_json_diagnostics('{"count": "12"}', schema={"type": "object"})
action_name: str = actions[0]["action"] if actions else ""
