"""The streaming repairer's suite: ``tors.JsonRepairer``, the stateful
incremental repairer for LLM token streams.

The discipline mirrors the sibling json_repair files: the whole-text engine
(``tors.repair_json``) is the differential reference for every class the
stream can decide incrementally (the chunk-boundary sweep feeds every split
position of every corpus text and demands ``end() == repair_json``), the
pinned ``json-repair`` package is the oracle for the repaired-complete
case, and the scaling idiom (``tests/test_scaling_pins.py``'s
``_assert_linear_per_doubling`` + ``loop_harness``) pins the linearity
contract: total streaming time grows ~4x when the input and the chunk
count quadruple, where the per-chunk re-parse anti-pattern measures ~16x.

The documented contract (docs/api.md ``## tors.JsonRepairer``):
``push`` returns the newly emitted repaired text (the delta), and
``snapshot``/``end`` render the stream closed into valid JSON. The
engine-equality classes and the documented divergences are listed in
``src/json_repair/streaming.rs``'s module docs; the sweep corpus below is
drawn from the equality classes, and the divergence shapes get their own
pins (valid output, documented difference).
"""

from __future__ import annotations

import json
import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from loop_harness import first_clean, min_wall_ms

json_repair_lib = pytest.importorskip("json_repair")  # pin: json-repair==0.63.5

# The growth gate per doubling: tests/test_scaling_pins.py's own idiom
# (a linear path measures ~2.0-2.3x per doubling; 3.0x sits ~1.3x above
# the band; the re-parse anti-pattern's ~4x per doubling blows through).
LINEAR_GATE_PER_DOUBLING = 3.0

# --- the corpus ---------------------------------------------------------------
#
# Three classes, each text ALSO fed through every chunk-boundary split:
#
# - VALID: strict-loadable documents; end() == repair_json(text) == the
#   oracle's repair_json(text) (the repaired-complete differential).
# - TRUNCATION: prefixes of valid documents (the LLM ran out of tokens);
#   end() == repair_json(text) (the truncation-heal differential; the
#   engine heals the same cut by the same rules).
# - MALFORMED: the repair classes the stream decides incrementally
#   (missing separators, single quotes, bare words, Python literals,
#   comments, prose prefixes, tuples, mismatched closers, number
#   rollback); end() == repair_json(text) for every text here too.
#
# Deliberately ABSENT (the documented divergences, pinned separately in
# DIVERGENCES below): string re-synchronization ('{"a": "hello}'),
# doubled quotes/escapes, in-container comments, multi-top-level
# documents, the missing-colon shape ('{"a" 1}'), the first-member
# dangling key ('{"a"'), the compound missing-key shape, the
# word-swallow past a mismatched closer, the mid-document closed-string
# newline-run, and the mid-pair first-member shape ('{"a"'), where the
# engine's whole-text machinery re-decides earlier text in ways a linear
# stream cannot replay.

VALID = [
    '{"a": 1}',
    '{ "a" : 1 }',
    '[1, 2, 3]',
    '  [1, 2]  ',
    '{"a": {"b": [1, 2]}, "c": null}',
    "[]",
    "{}",
    '""',
    '"hi"',
    '"\\u00e9"',
    '"\u00e9"',
    '"\U0001f600"',
    '12',
    '-12.5',
    '1.5e3',
    'NaN',
    'Infinity',
    '{"a": NaN}',
    '{"a": Infinity}',
    '{"a": -Infinity}',
    '[NaN, 1]',
    '{"a": 1.5, "b": -0.0}',
    '{"a": 123456789012345678901234567890}',
    '{"": 1}',
    r'{"a": "x\ny\tz"}',
    r'{"a": "x\/y"}',
    r'{"a": "x\ud83d\ude00y"}',
    '{"key": "value with, commas; and: colons"}',
    # the valid-keeps neighbors of the strictly-empty element rule: the
    # element's own array closed properly, the fast paths kept it
    '[[], 1]',
    '[[] , 1]',
    '[true, [] , 1]',
    '[{}]',
    '[[] ]',
    '[""]',
    '{"a": ["" , 1]}',
    '{"a": [[] , 1]}',
    # the trailing-garbage keeps: the element's array closed properly
    # and the strict-suffix probe re-decided the valid root
    '[[] , 1] x',
    '{"a": ["" , 1]} x',
    '[true, [] , 1] x',
    'x {"a": ["" , 1]}',
    # the deferred-escape keeps: the strict lanes decided (the strings'
    # `\f`/`\/` decode through the canonical table, the whole-input
    # loads and the suffix probe both keep the rendering verbatim)
    '["\\f"]',
    '["\\f%"]',
    '{"\\f\\/%": 1}',
    '["\\f", 1]',
    '["\\f", "\\/"]',
    '{"a": ["\\f"], "b": 1}',
    '[["\\f"], ["\\/"]]',
    '["\\\\f"]',
    # the smart-quote keeps: a `„` in a string the strict lanes decide
    # (the whole-input loads here) is plain content on both sides -- the
    # fast path takes it, no span machinery engages
    '["e„", 1]',
    '["e„\\\"]',
]

TRUNCATION = [
    '{"a": "hel',
    '{"a": [1, 2',
    '{"a": [1,',
    '{"a": [',
    '{"a": {',
    '{"a":',
    '{"a": "x", "b"',
    '{"a": "x", "b": 2, ',
    '{"a": 1, "b": [1, {"c": "d',
    '[1, 2,',
    '[',
    '{',
    '"hi',
    '{"a": "x\\ud83d\\ude00',
    '{"a": "x\\u00',
    '{"a": "x\\',
    '{"a": 1e',
    '{"a": 1e+',
    '{"a": -',
    '{"a": 1,000',
    '{"a": tru',
    '{"a": None',
    '[tru',
    '{"a": [1 2',
    '{"a": "hel" ',
    '{"a": 12',
    '{"a": [1, 2} ',
    # the nested-empty-container prefixes: the heal drops a still-open
    # empty container at an item position and keeps it at a member-value
    # position (the engine's own heal, byte-exact)
    '[[',
    '[[[',
    '[[{',
    '[1, [',
    '[1, [{',
    '[{}, [',
    '[[], [',
    '[[1], [',
    '["x", [',
    '{"a": [1, {',
    '{"a": [1, [',
    '{"a": {"b": [',
    '{"a": [',
    '{"a": {',
    '{"a": [{"b": {',
    '[{',
    '[1, {',
    '{"a": 1, "b": [',
    '[1, []',
    '{"a": []',
    '{"a": [[]',
    '[1, [2, [',
    '[1, [2, [3, [',
    '[[1], [2, [',
    '{"a": [1], "b": {',
    '{"a": [1], "b": [',
    '{"x": [[',
    '{"x": [[1], [',
    '[1, tru, {',
    '{"a": (',
    '{"a": [(',
    '[(',
    '[1, (',
    '[[], []',
    '[[], \r',
    '[[], [',
    '[[]',
    '[[1]',
    '{"a": [[], []',
    '{"a": [[]',
    '{"a": {b: {',
    '{"a": [{',
    '[1, [], 2',
    '[[], 1',
    '[[], []]',
    '{"a": 1, "b": [[], []',
    '[[], [], [',
    '{"a": [[1], [[2]]',
    '[[[]',
    '[[[], [',
    '{: [',
    '{"a": [[[]',
    '[1, [[[]',
    '([',
    '(1, [',
    '([1',
    '[[], [{"a": [',
    '{: [1, 2]}',
    '[1, []] ,',
    '[1, [[',
    # the empty-element retractions (the falsy-nudge class)
    '[1, "',
    '[1, ""',
    '[1, -',
    '[{}, "',
    '{"a": [1, "',
    '[1, "\\',
    # the array lane's item drops at the cut: the stray '...' (an open
    # string's content or a bare number-run parsing to the exact string
    # "..." with the parse ending on a '.'; the closed ["..."] element's
    # parse ends on its quote and stays) and the strictly-empty item
    # whose next char is not a separator (the engine's own skip, the
    # decision deferred to close time: the whole-input loads fast path
    # keeps every element whenever the text ends up valid)
    ' [\r\r"...',
    '["...',
    '[[...',
    '[[...',
    '[1, "...',
    '[1, 2, "...',
    '{"a": ["...',
    '{"a": [1, "...',
    '("...',
    '[1, ...',
    '[..., 1]',
    '[...-',
    '[...-]',
    '[1, ...-]',
    '(...-',
    '["....',
    '["..',
    '[" ...',
    '["...',
    '[[] ,',
    '[[]  ,',
    '[[] , 1',
    '[[] , 2',
    '[[]\r, ',
    '[[], [] ,',
    '[[], [] , 1',
    '[[] ,[]',
    '[[] , []',
    '[1, []\r, ',
    '[1, []\r, 2',
    '[1, [] , 2 ',
    '[[], [[] , 2], 3',
    '[[] , "k',
    '[{}\r, ',
    '[""\r, ',
    '["" , 1',
    '[[] , 1 , 2 ',
    # the escape-tail heal (rstrip on the open string's content, the
    # newline-run on a closed string at the tail)
    '{"k": "a\\n',
    '{"k": "a\\n"',
    '{"k": "a\\n\\t',
    '{"k": "a\\n\\t"',
    '{"k": "a\\n ',
    '{"k": "a \\n"',
    '{"k": "a\\n\\n',
    '{"k": "a\\r\\n"',
    '{"k": "a\\r"',
    '{"k": "a "',
    '{"k": "a b ',
    '{"k": "  ',
    '{"k": "a\\u0041',
    '{"k": "a\\u0041"',
    '["a\\n',
    '{"a": "x\\n", "b": 1}',
    # the deferred-escape family: a CLOSED string holding `\f` or `\/`
    # (the two escapes the repair lane does not accept) or an odd
    # backslash run of three or more does not commit its rendering at
    # its own close -- the mark resolves at end() through the repair
    # lane's escape normalization when the repair lane decided (a cut
    # root here), verbatim when the strict lanes did (the complete
    # documents above). The distinction survives every separator:
    # `["\f", 1` escapes while `["\f", 1]` keeps.
    '["\\f%"',
    '["\\f", 1',
    '["\\f"',
    '{"\\f\\/%": 1',
    '[[ "\\f"]',
    '["\\/%"',
    '["\\f\\/%", 1',
    '["a\\fb"',
    '[{"k": "\\f"}, 1',
    '{"\\f": 1',
    '{"k": "a\\/b"',
    '["a\\/b"',
    '["\\t\\f"',
    '["\\f\\t"',
    '["\\/\\f"',
    '["\\\\\\f"',
    '["\\\\\\f", 1',
    '["\\\\\\t"',
    '["\\\\\\u0046"',
    '["\\x41"',
    '["\\x41", 1',
    '["\\x%"',
    '[[] , "\\f", 1',
    '[1, [] , "\\f"',
    # the in-flight special-float word at the cut heals to the string
    # (the strict class keeps the float in the complete document)
    '[NaN',
    '[Infinity',
    '{"a": NaN',
    '{"a": Infinity',
    '{"a": -Infinity',
    '[1, -NaN',
    # the open-string cut's escape drain: the cut's repair lane always
    # decides (a truncated document never strict-loads), so the
    # IN-FLIGHT content's divergent escapes re-render through the same
    # escape walk as the closed-string marks, then the escape-tail
    # rstrip applies to the DECODED value (`["\f` -> `["\\f"]`,
    # `["\/` -> `["\\/"]`, `{"k": "x\n\f` -> `{"k": "x\n\\f"}` -- the
    # rstrip is the tail's, the mid-string newline-escape stands)
    '["\\f',
    '["\\/',
    '["\\x',
    '["\\',
    '["\\\\\\',
    '["\\\\\\\\\\',
    '["a\\f',
    '["a\\fb',
    '["\\ftext',
    '["\\f more',
    '["\\/x',
    '["\\f ',
    '[" \\f',
    '["\\f\\n',
    '["\\f\\t',
    '["\\f%',
    '["\\t\\f',
    '["\\fx',
    '["x\\f',
    '["\\f\\/',
    '["\\f", "\\/',
    '{"k": "\\f',
    '{"k": "\\/',
    '{"k": "\\f\\n',
    '{"k": "\\f\\t',
    '{"k": "\\f ',
    '{"k": " \\f',
    '{"k": "x\\n\\f',
    '{"k": "a\\t\\f',
    '{"k": "\\f", "\\/',
    '[[ "\\f',
    '[1, "\\f',
    '{"a": ["\\f',
    '{"a": "x\\f',
    '[[], "\\f',
    '{"a": "\\f", "b',
    # the engine's unterminated-string fixup on a trailing escape run:
    # NOW PARITY (the open-string drain's escape walk collapses the odd
    # run with the run's own delimiter taken INTO the value, the
    # engine's own observable; the doubled-escape even-run shape stays
    # the pinned divergence)
    '{"k": "a\\n\\t\\\\\\"b',
    '{"k": "a\\n\\t\\\\\\"b c',
    '{"k": "a\\n\\t\\\\',
    '{"k": "a\\n\\t\\\\\\',
    '{"k": "a\\n\\t\\\\\\"',
    # the smart-quote re-decision in the open-string drain's odd-run
    # collapse: the value's ordinary `„` pushes the lib's low
    # smart-quote span, so the run's own delimiter JOINS the collapsed
    # backslash as content (the engine's smart-quote machinery keeps
    # the halved run; the plain content drops it). The run lengths
    # 3/5/7 collapse identically; the `'` delimiter re-decides the
    # same (the push is the `„` char's); the other STRING_DELIMITERS
    # members as content never re-decide; `”` pops the span (the
    # plain drop stands)
    '["e„\\\\\\"',
    '["„\\\\\\"',
    '["e„\\\\\\"b',
    '["e„\\\\\\\\\\"',
    '["e„\\\\\\\\\\\\\\"',
    "['e„\\\\\\'",
    '["e“\\\\\\"',
    '["e”\\\\\\"',
    '["e„”\\\\\\"',
    '["e„\\\\\\"x\\\\\\"',
    '["\\\\„\\\\\\"',
    '{"k": "e„\\\\\\"',
    # the held partial escape at the cut: a lone high surrogate escape
    # plus a bare `\u`/`\x` tail -- the drain's predicate fires on the
    # incomplete escape and the walk renders the engine's own answer
    # (the flushed U+FFFD, the literal `\u` content)
    '["\\ud83d\\u',
    '["\\ud83d\\x',
    '{"k": "\\ud83d\\u',
    # the closed smart-quote keeps: the loads-valid `„`-holding string
    # (the run's own delimiter escaped, one close quote) keeps through
    # the strict lanes; the doubled close quote after the run is the
    # loads-invalid shape (the engine's span machinery absorbs it --
    # the documented divergence) and still matches here (the engine's
    # own repair-lane answer, one string)
    '["e„\\\\\\"]',
    '["e„\\\\\\""]',
]

MALFORMED = [
    '{"a": 1 "b": 2}',
    '[1 2]',
    '{"a": 1,}',
    '[1, 2,]',
    "{'a': 1}",
    "['a', 'b']",
    '{a: 1}',
    '{a: 1, b: 2}',
    '{12: 1}',
    '{: 1}',
    '{ : 1}',
    '{:}',
    '{"a": None}',
    '{"a": True, "b": False}',
    '[True, False, None]',
    # the keyword family matches case-insensitively (the engine's own
    # observable); the special floats stay exact-spelling
    '{"a": TRUE}',
    '{"a": FALSE}',
    '{"a": NULL}',
    '{"a": NONE}',
    '{"a": tRuE}',
    '[TRUE, FALSE, NULL, NONE, tRuE]',
    '{"a": TRUE, "b": FALSE, "c": NULL, "d": NONE}',
    '{"a": nan}',
    '{"a": NAN}',
    '{"a": nAn}',
    '{"a": infinity}',
    '{"a": INFINITY}',
    '{"a": -infinity}',
    '{"a": -TRUE}',
    '{"a": -tru}',
    '{"a": -abc}',
    '{"a": -x}',
    '{"a": -nan}',
    '{"a": -NaN}',
    '[-NaN]',
    '[1, -ab]',
    '{"a": -12abc}',
    '{"a": 1e}',
    '{"a": 1e+}',
    '{"a": -}',
    '{"a": 1,000}',
    '{"a": 12abc}',
    '{"a": 1 2}',
    # note: '{"a": 12_abc}' is deliberately absent: the digit-group
    # underscore followed by a bare word is a shape where the whole-text
    # engine's rewind arithmetic itself diverges from the oracle
    # (tors: "2_abc", oracle: "12_abc"), so no streaming behavior can be
    # differentially pinned there; see the streaming module's divergence
    # list.
    '{"a": [1, 2}',
    '{"a": 1]',
    '[{"a": 1]',
    '{"a": [1, }',
    '{"a": [}',
    '[1,, 2]',
    '(1)',
    '(1,)',
    '(1, 2)',
    '{"a": (1, 2)}',
    '(1, -)',
    # the mismatched-closer paren family: the group's own array loop
    # swallows a `]` as a garbage item (the group stays open and keeps
    # parsing) and a `}` terminates and is consumed by it; the empty
    # group's item skip reads the char after its own `)` and an
    # unterminated group retracts at close time like any empty container
    '[(]',
    '[(],',
    '[(] ]',
    '[(])',
    '[(,]',
    '[( ]',
    '[( ],',
    '((]',
    '[[(]',
    '([( ]',
    '[1,(]',
    '[(),(]',
    '[(]x',
    '[(], 1, 2]',
    '[( ], 1',
    '[(}]',
    '[() ]',
    '[() ,]',
    '[()), ]',
    '[()x]',
    '[()x',
    '[()]x',
    '[()]',
    '[()],',
    '[(), ',
    '[1, ()]',
    '[1, () ',
    '{"a": (]',
    '{"a": [()]',
    # the empty-object fallback: the body recovers nothing, the object
    # drains to the empty array (the engine's own classifier)
    '{]}',
    '{,}',
    '{"a": {]}}',
    '// lead\n{"a": 1}',
    '/*x*/{"a": 1}',
    '# c\n{"a": 1}',
    'Answer: {"a": 1}',
    'Here: [1, 2]',
    '{"a": 1} junk',
    '{a: 1, b}',
    '{"a": "x", "b"}',
    # the rollback-cycle shapes: the empty element retracts (the
    # falsy-nudge class), the popped sign never becomes one
    '[1e-}',
    '[1E-}',
    '[1e+-}',
    '[1,-]',
    '[1-]',
    '[1-}',
    '[1, 2e-}',
    '[1e-]',
    '[1e- ',
    '[1e- 2]',
    # the rewind lane's string-entry skip: a number run terminated by an
    # alphabetic char re-enters the string lane, whose leading skip
    # consumes the run's non-alphanumeric garbage (a dot run, sign and
    # dot, an underscore) before the literal table
    '[.t]',
    '[..t]',
    '[...t]',
    '[....t]',
    '[...t x]',
    '[.x.y]',
    '[.5x]',
    '[.5t]',
    '[.5x, 1]',
    '[-.t]',
    '[-.5x]',
    '[-.Infinity]',
    '[._t]',
    '[.true]',
    '[.true, 1]',
    '[.false]',
    '[.none]',
    '[.null]',
    '[.NaN]',
    '[--t]',
    '[1, .t]',
    '{"a": .t}',
    '{"a": .false}',
    '[...]',
    '[....]',
    '[.e]',
    '[..e]',
    '[.1e]',
    '[-e]',
    '{: 1, : 2}',
    # the object-key lane's string-lane semantics: the entry skip's
    # strip and the whitespace-touched invalidation (the key attempt
    # dies at whitespace unless the next char is its colon)
    '{-ab: 1}',
    '{.ab: 1}',
    '{-.ab: 1}',
    '{-12: 1}',
    '{.5: 1}',
    '{-true: 1}',
    '{--ab: 1}',
    '{-eOe: 6}',
    '{12ab: 1}',
    '{a b: 1}',
    '{a  b: 1}',
    '{a b: 1, c: 2}',
    '{a b c: 1}',
    '{-ab x: 1}',
    '{-ab : 1}',
    '{: 1, : 2}',
    # the attempt that strips to nothing drops the pair
    '{-: 1',
    '{-.: 1}',
    '{-:}',
    '{-: , "a": 1}',
    # the member-boundary heals: the ','/closer where the colon was due
    # acts AS the colon; the key attempt's string absorbs its own
    # ','/']'
    '{a,b: 1}',
    '{a]: 1}',
    '{a ,: 1}',
    '{a b,c: 1}',
    '{-one,f: false}',
    '{"a" , 1}',
    # the open string's tail rstrip: the escaped Python-whitespace code
    # points join the escape-tail heal (the raw `\u2000` en quad)
    '["\\u2000',
    '["\\u2000\\u2000',
    '{"k": "\\u2000',
    # the strictly-empty element's LOCAL rule: the trailing-separator
    # drops (the array's own closer ended on a `,` where a value was
    # required) and the ws-then-value-start drops (the stall guard ate
    # the whitespace, the char re-parsed as the next item)
    '[[] ,]',
    '[1, [] ,]',
    '[[], [] ,]',
    '[[] , [] ,]',
    '{"a": [[] ,]}',
    '{"a": [1, [] ,]}',
    '[[] ,] x',
    '[[] x]',
    '[[] x, 1]',
    '[1, [] x, 1]',
    '[[]x]',
    # the deferred-escape class on repair-lane-decided COMPLETE
    # documents: the whole-input loads and the suffix probe both died
    # on the input (a bare word, a tuple, a non-strict number, a
    # missing comma, a trailing separator), so the engine's repair lane
    # re-normalized the held strings even though the machine's render
    # is itself strict-loadable
    '("\\f")',
    '("\\f", 1)',
    '[(  "\\f" , 1)]',
    '[tru, "\\f"]',
    '[tru, "\\f"] x',
    '[True, "\\f"]',
    '[None, "\\f"]',
    '[01, "\\f"]',
    '["\\x41"]',
    '["\\x41", 1]',
    '[1 2, "\\f"]',
    '[1, "\\f",]',
    '{"a": 1, "b": "\\f",}',
    '{"a": 1 2, "\\f"]}',
    '["\\f"] x',
]

CORPUS = VALID + TRUNCATION + MALFORMED

# The documented divergences (whole-text-only machinery the linear
# stream cannot replay): deliberately ABSENT from the parity corpus
# above, pinned here to their documented shapes — the stream's end()
# stays valid JSON (or the empty sentinel) and the divergence class
# stays put at every split position.
DIVERGENCES = [
    # multi top-level values: the engine array-wraps, the stream keeps
    # the first
    ('{"a": 1}\n{"b": 2}\n', '[{"a": 1}, {"b": 2}]', '{"a": 1}'),
    # in-container comment: the engine consumes the value, the stream
    # parses it
    ('{"a": /*c*/ 1}', '{"a": ""}', '{"a": 1}'),
    # string re-synchronization: the engine terminates the damaged string
    # at the structural closer
    ('{"a": "hello}', '{"a": "hello"}', '{"a": "hello}"}'),
    ('{"a": "x\\", "b": 1}', '{"a": "x\\"", "b": 1}', '{"a": "x\\", ", "b\\"": 1}'),
    # doubled-escape repair: whole-text-only
    ('{"a": "\\\\\\\\"', '{"a": "\\""}', '{"a": "\\\\\\\\"}'),
    # the engine's unterminated-string fixup re-deciding a trailing
    # backslash-escape run: NOW PARITY (the open-string drain's escape
    # walk collapses the odd run with the run's own delimiter taken
    # INTO the value, the engine's own observable) — pinned in
    # TRUNCATION
    # missing colon: the stream keeps the value, the engine heals ""
    ('{"a" 1}', '{"a": ""}', '{"a": 1}'),
    # the first-member dangling key: the engine falls back to an array
    ('{"a"', '["a"]', '{}'),
    # the compound missing-key shape: NOW PARITY (the key lane's string
    # semantics commit the compound attempt the engine's own way,
    # `{"1,": 2}`) — pinned in MALFORMED
    # the word-swallow: a bare word born at an array position runs past
    # a mismatched closer in the engine's reparse
    ('[1e}', '[1, "e}"]', '[1, "e"]'),
    ('[1, tru}', '[1, "tru}"]', '[1, "tru"]'),
    ('[1, e-}', '[1, "e-}"]', '[1, "e-"]'),
    ('[1, e }', '[1, "e }"]', '[1, "e"]'),
    ('[e-}]', '["e-}"]', '["e-"]'),
    # the mismatched-closer cascade: the `)` closes THROUGH the emptied
    # inner array and the engine's whole-text close-up re-decides the
    # emptied shapes away (the stream's cascade keeps them; both
    # outputs valid)
    ('[([)$', '[]', '[[[]]]'),
    # the mid-document closed-string newline-run: the engine's repair
    # lane strips it wherever the string appears (the whole input not
    # strict-valid); the linear stream keeps the strict spelling — only
    # the tail-of-stream case (the string last before the cut) matches
    ('{"a": "x\\n", "b": 1', '{"a": "x", "b": 1}', '{"a": "x\\n", "b": 1}'),
    ('[1, "x\\n", 2', '[1, "x", 2]', '[1, "x\\n", 2]'),
    # the mid-document special-float word: the engine's repair lane
    # quotes it wherever it appears (the whole input not strict-valid);
    # the stream keeps the float mid-stream (the strict spelling) and
    # matches only the in-flight-at-the-cut case
    ('[NaN, 1', '["NaN", 1]', '[NaN, 1]'),
    ('{"a": NaN, "b": 1', '{"a": "NaN", "b": 1}', '{"a": NaN, "b": 1}'),
    # the paren-with-colon conversion: the engine turns a colon inside
    # the parenthesized container into an object
    ('("a": ', '{"a": ""}', '"a"'),
    # the emptied-frame cascade: the whole-text lane re-decides the
    # emptied items' shapes through the parent's own close
    ('[[[] ,][] ,]', '[]', '[[]]'),
    # the empty-key attempt at a member boundary: the engine's key
    # loop discards it and the object drains through the fallback
    ('{"" ,}', '[]', '{"": ""}'),
    # the missing-colon shape with a CONTAINER where the colon was due:
    # the engine's object loop consumes the container-open as the
    # colon-substitute and the array's repair-lane item skip empties it;
    # the stream inserts the colon and keeps the value it saw
    ('{"a" [[] , 1]}', '{"a": []}', '{"a": [[], 1]}'),
    # the completed-paren taint on a LATER deferred element: the paren
    # killed both fast paths (the whole-input loads and the suffix
    # probe both fail on a tuple), so the engine's repair lane decided
    # the array and its skip fired; the stream's local rule (the
    # array's own closing shape) keeps it
    ('[(2), [] , 1]', '[2, 1]', '[2, [], 1]'),
    # the COMMA-first tail after a valid root: the engine's suffix
    # probe is gated on non-comma trailing content, so the repair lane
    # decided and the element dropped; the stream's local rule kept it
    ('[[] , 1] ,', '[1]', '[[], 1]'),
    # the object-lane comma classification: a comma after a bare VALUE
    # word belongs to the string when the next member's key is
    # garbage-led — the engine's whole-text classifier absorbs the tail
    ('{"a": xyz, -ab: 1}', '{"a": "xyz, -ab: 1"}', '{"a": "xyz", "ab": 1}'),
    # the unquoted-value run's absorption: the engine's unquoted-value
    # lane eats container chars and internal whitespace into the string
    # and re-decides literal prefixes
    ('[{"b": undefined},]', '[{"b": "undefined},"}]', '[{"b": "undefined"}]'),
    ('[null x]', '[null, "x"]', '["null x"]'),
    # the unquoted-value `}`-absorption without the ws: a non-strict run
    # VALUE at an object's close absorbs the object's own `}` when the
    # tail past it holds two chars before the next `}`
    ('{"a": b}66', '{"a": "b}66"}', '{"a": "b"}'),
    # a lone high surrogate escape followed by ANOTHER escape: NOW
    # PARITY (the deferred-escape re-render reproduces the repair lane's
    # escape-run re-decision), pinned in TestDeferredEscapeStrings
    # the smart-quote span machinery absorbing a close quote: the
    # engine's slow-path string scan pushes a low smart-quote span at
    # an ordinary `„` and the span-active close quote is CONTENT (the
    # string runs on, `["e„\\\"""` -> the value `e„\"""`); the linear
    # stream closes at the quote and the tail re-synchronizes
    ('["e„\\\\\\""', '["e\\u201e\\\\\\"\\""]', '["e\\u201e\\\\\\""]'),
    ('["„\\\\\\""', '["\\u201e\\\\\\"\\""]', '["\\u201e\\\\\\""]'),
    # the same span machinery swallowing the post-close tail: the
    # engine's string runs to the structural break, the stream re-pairs
    # at the close quote (the `„`-free analog `["e"x"]` diverges the
    # same way, the quoted-section re-pair above)
    ('["e„"x"]', '["e\\u201e\\"x\\"]"]', '["e\\u201e", "x\\""]'),
    # the empty-object fallback's body reparse: the stream reproduces it
    # only when the body recovers nothing ({]} -> [], {"a": {(] -> ...
    # match); a body that would re-parse to members keeps {}
    ('{ x}', '["x}"]', '{}'),
    ('{"a": {(]', '{"a": []}', '{"a": []}'),
    # duplicate object keys: the dict update-in-place is whole-text-only
    ('{"a": 1, "a": 2}', '{"a": 2}', '{"a": 1, "a": 2}'),
    # the mismatched-closer garbage: the engine's whole-text close-up
    # re-decides earlier structure
    ("{{\r]0\x01\x0b/\u2001/\u2001\u2000\u2000('", "[]", '{}'),]


def stream(text: str, splits: list[int], *, ensure_ascii: bool = True) -> tuple[list[str], str]:
    """Push ``text`` through a fresh repairer cut at ``splits`` (char
    offsets); return (the per-push deltas, the end() output)."""
    r = tors.JsonRepairer(ensure_ascii=ensure_ascii)
    deltas = []
    prev = 0
    for p in sorted(splits):
        deltas.append(r.push(text[prev:p]))
        prev = p
    deltas.append(r.push(text[prev:]))
    return deltas, r.end()


def _assert_loads_or_empty(out: str) -> None:
    """The valid-output invariant: the machine's closed renders are
    loadable JSON, or the nothing-recoverable empty string."""
    if out == "":
        return
    json.loads(out)


# --- chunk-boundary sweeps ------------------------------------------------------


class TestChunkBoundarySweep:
    @pytest.mark.parametrize("text", CORPUS)
    def test_every_split_matches_the_engine(self, text: str) -> None:
        """Every split position of every corpus text: two pushes, and
        end() is byte-identical to the whole-text engine on the same
        total text (the zero-re-parse contract's observable)."""
        want = tors.repair_json(text)
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    @pytest.mark.parametrize("text", CORPUS)
    def test_every_prefix_snapshot_is_valid(self, text: str) -> None:
        """Every prefix of every corpus text: snapshot() is loadable JSON
        (or the empty sentinel) and equals end() on that prefix: the
        'valid JSON now' contract holds at every point of the stream."""
        for k in range(1, len(text) + 1):
            prefix = text[:k]
            r = tors.JsonRepairer()
            r.push(prefix)
            snap = r.snapshot()
            _assert_loads_or_empty(snap)
            assert snap == r.end(), f"snapshot != end at prefix {k} of {text!r}"

    @pytest.mark.parametrize("text", CORPUS)
    def test_full_text_matches_the_engine(self, text: str) -> None:
        """Without any split (one push), snapshot() and end() both match
        the whole-text engine's answer on the corpus text itself."""
        want = tors.repair_json(text)
        r = tors.JsonRepairer()
        r.push(text)
        assert r.snapshot() == want, f"{text!r}: snapshot != engine"
        assert r.end() == want, f"{text!r}: end != engine"

    def test_many_random_splits_match_the_one_push_answer(self) -> None:
        """Random multi-chunk splits (seeded): end() is chunking-
        invariant, equal to the one-push answer on every corpus text."""
        rng = random.Random(20261005)
        for text in CORPUS:
            one = tors.JsonRepairer()
            one.push(text)
            want = one.end()
            points = sorted(rng.sample(range(1, len(text)), min(4, len(text) - 1)))
            _, got = stream(text, points)
            assert got == want, f"splits {points} of {text!r}: {got!r} != {want!r}"

    @pytest.mark.parametrize("text", VALID)
    def test_repaired_complete_case_matches_the_oracle(self, text: str) -> None:
        """The repaired-complete differential: a valid document streamed
        through chunks repairs to the oracle's own answer."""
        want = json_repair_lib.repair_json(text)
        rng = random.Random(42)
        points = sorted(rng.sample(range(1, len(text)), min(3, len(text) - 1)))
        _, got = stream(text, points)
        assert got == want, f"{text!r}: {got!r} != oracle {want!r}"

    @pytest.mark.parametrize("text", VALID)
    def test_ensure_ascii_false_streams_verbatim(self, text: str) -> None:
        """ensure_ascii=False streams non-ASCII verbatim and still agrees
        with the engine's own ensure_ascii=False answer."""
        want = tors.repair_json(text, ensure_ascii=False)
        rng = random.Random(43)
        points = sorted(rng.sample(range(1, len(text)), min(3, len(text) - 1)))
        _, got = stream(text, points, ensure_ascii=False)
        assert got == want, f"{text!r}: {got!r} != {want!r}"


# --- the deferred-escape family -------------------------------------------------


class TestDeferredEscapeStrings:
    """The weekly-fuzz crash class: a closed string whose raw source
    holds a repair-lane-divergent escape (`\\f`/`\\/`, the two escapes
    the repair lane does not accept, or an odd backslash run of three
    or more) resolves at end()/snapshot() close time, never at a
    separator. The strict lanes keep the verbatim rendering (a complete
    document, trailing garbage and comments behind a valid root
    included); the repair lane (a cut root, a paren anywhere, a
    repair-lane-decided complete document) re-renders the raw source
    through the lib's escape normalization (`\\f` to `\\\\f`). Byte-exact
    against the oracle at every chunking."""

    CUT_FAMILY = [
        '["\\f%"',
        '["\\f", 1',
        '["\\f"',
        '{"\\f\\/%": 1',
        '[[ "\\f"]',
        '["\\/%"',
        '["\\f\\/%", 1',
        '["a\\fb"',
        '[{"k": "\\f"}, 1',
        '{"\\f": 1',
        '{"k": "a\\/b"',
        '["a\\/b"',
        '["\\t\\f"',
        '["\\f\\t"',
        '["\\/\\f"',
        '["\\\\\\f"',
        '["\\\\\\f", 1',
        '["\\\\\\t"',
        '["\\\\\\u0046"',
        '["\\x41"',
        '["\\x41", 1',
        '["\\x%"',
        '[[] , "\\f", 1',
        '[1, [] , "\\f"',
        # the crash artifact itself (fuzz/corpus seed 8eca3c68...)
        '["\\f%"',
    ]

    KEEP_FAMILY = [
        '["\\f"]',
        '["\\f%"]',
        '{"\\f\\/%": 1}',
        '["\\f"]x',
        '["\\f"] // x',
        '["\\f", 1]',
        '["\\f", "\\/"]',
        '{"a": ["\\f"], "b": 1}',
        '[["\\f"], ["\\/"]]',
        '["\\\\f"]',
        '["\\\\f"',
    ]

    REPAIR_COMPLETE_FAMILY = [
        '("\\f")',
        '("\\f", 1)',
        '[(  "\\f" , 1)]',
        '[tru, "\\f"]',
        '[tru, "\\f"] x',
        '[True, "\\f"]',
        '[None, "\\f"]',
        '[01, "\\f"]',
        '["\\x41"]',
        '["\\x41", 1]',
        '[1 2, "\\f"]',
        '[1, "\\f",]',
        '{"a": 1, "b": "\\f",}',
        '["\\f"] x',
    ]

    # The suffix-probe gate's tail class (the lib's
    # `_initial_container_has_non_comma_trailing_content`, mirrored at
    # the top-level sites): the KEPT half -- leading consumption of any
    # kind puts the root's char at `index != 0` and the probe fires
    # whatever the tail holds; a trailing token (or the comment opener
    # read as the decision char) fires it from a clean index too.
    TAIL_KEEP_FAMILY = [
        '\x0b["\\f"]',
        '\x0c["\\f"]',
        'x ["\\f"]',
        '\x0bx ["\\f"]',
        'x ["\\f"] ,',
        '\x0b["\\f"] x',
        '\x0b["\\f"],',
        '["\\f"]\x0b x',
        '["\\f"]\x0b // c',
        '["\\f"]\x0b # c',
        '["\\f"] x\x0b',
        '["\\f"] x \x0b y',
    ]
    # The tail class's TAINT half: from a clean index (the root's char
    # IS the input's first byte) the trailing scan fires the probe only
    # on a NON-COMMA decision char -- the input ending inside a trailing
    # non-strict whitespace run (no content behind it: no probe) and the
    # COMMA-FIRST tail (the scan reads a `,`, returns False: no probe)
    # are repair-lane inputs -- the held strings re-render.
    TAIL_TAINT_FAMILY = [
        '["\\f"]\x0b',
        '["\\f"]\x0c',
        '["\\f"]\x0b\x0c',
        '["\\f"]\t\x0b',
        '["\\f"] ,',
        '["\\f"],',
        '["\\f"], x',
        '["\\f"]\x0b,',
        '["\\f"]\x0b ,',
        '["\\/", 1] ,',
        '{"k": "\\f"} ,',
    ]

    @pytest.mark.parametrize(
        "text", CUT_FAMILY + REPAIR_COMPLETE_FAMILY, ids=lambda t: repr(t)
    )
    def test_matches_the_engine_and_the_oracle_at_every_split(self, text: str) -> None:
        want = tors.repair_json(text)
        assert want == json_repair_lib.repair_json(text), f"{text!r}: engine != oracle"
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"
        _, whole = stream(text, [])
        assert whole == want, f"{text!r}: one-push != {want!r}"

    def test_the_lone_surrogate_plus_escape_run_renders_the_repair_lane(self) -> None:
        """The previously-pinned divergence is gone: a lone high
        surrogate escape followed by another escape (`\\ud83d\\falsee`,
        a cut root) re-renders through the repair lane's escape
        normalization -- the held surrogate flushes U+FFFD (the machine's
        documented lone-surrogate divergence from the ORACLE, which keeps
        the `\\ud83d` escape) and the `\\f` stays a literal backslash +
        f, byte-exact against the engine at every split."""
        text = '["\\ud83d\\falsee"'
        want = tors.repair_json(text)
        assert want == '["\\ufffd\\\\falsee"]'
        assert want != json_repair_lib.repair_json(text)  # the oracle keeps \\ud83d
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    @pytest.mark.parametrize("text", KEEP_FAMILY, ids=lambda t: repr(t))
    def test_the_strict_lanes_keep_verbatim_at_every_split(self, text: str) -> None:
        want = tors.repair_json(text)
        assert want == json_repair_lib.repair_json(text), f"{text!r}: engine != oracle"
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    @pytest.mark.parametrize(
        "text", TAIL_KEEP_FAMILY + TAIL_TAINT_FAMILY, ids=lambda t: repr(t)
    )
    def test_the_tail_gate_matches_the_oracle_at_every_split(self, text: str) -> None:
        """The tail class of the suffix-probe gate, byte-exact against
        the oracle at every chunking: leading consumption (whitespace,
        prose) puts the root's char at `index != 0` and the probe fires
        whatever the tail holds -- the held strings keep when the root
        rendering is strict-loadable; a trailing token fires the probe
        from a clean index too; the input ENDING inside a trailing
        non-strict whitespace run and the COMMA-FIRST tail are repair-
        lane inputs (no probe) -- the held strings re-render."""
        want = tors.repair_json(text)
        assert want == json_repair_lib.repair_json(text), f"{text!r}: engine != oracle"
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"
        _, whole = stream(text, [])
        assert whole == want, f"{text!r}: one-push != {want!r}"

    @pytest.mark.parametrize(
        "text", TAIL_KEEP_FAMILY + TAIL_TAINT_FAMILY, ids=lambda t: repr(t)
    )
    def test_the_tail_gate_holds_over_random_chunkings(self, text: str) -> None:
        rng = random.Random(len(text))
        points = sorted(rng.sample(range(1, len(text)), min(4, len(text) - 1)))
        want = tors.repair_json(text)
        _, got = stream(text, points)
        assert got == want, f"splits {points} of {text!r}: {got!r} != {want!r}"

    @pytest.mark.parametrize(
        "text",
        CUT_FAMILY + KEEP_FAMILY + REPAIR_COMPLETE_FAMILY + TAIL_KEEP_FAMILY + TAIL_TAINT_FAMILY,
    )
    def test_snapshot_equals_end_at_every_prefix(self, text: str) -> None:
        for k in range(1, len(text) + 1):
            r = tors.JsonRepairer()
            r.push(text[:k])
            snap = r.snapshot()
            _assert_loads_or_empty(snap)
            assert snap == r.end(), f"snapshot != end at prefix {k} of {text!r}"

    def test_the_distinction_survives_a_snapshot_boundary(self) -> None:
        """A snapshot taken while a deferred string is held resolves the
        same way one-push does, and the stream continues as if it had
        not run (the snapshot contract)."""
        r = tors.JsonRepairer()
        r.push('["\\f", ')
        held = r.snapshot()
        _assert_loads_or_empty(held)
        r2 = tors.JsonRepairer()
        r2.push('["\\f", ')
        r2.push("1")
        assert r2.end() == '["\\\\f", 1]'
        r.push("1")
        assert r.end() == r2.end()

    def test_multi_string_documents_hold_independently(self) -> None:
        r = tors.JsonRepairer()
        r.push('["\\f", 1, "\\/"')
        assert r.end() == '["\\\\f", 1, "\\\\/"]'
        # the complete document keeps both through the strict lane (the
        # canonical dumps does not escape the solidus)
        r2 = tors.JsonRepairer()
        r2.push('["\\f", 1, "\\/"]')
        assert r2.end() == '["\\f", 1, "/"]'
        assert r2.end() == json_repair_lib.repair_json('["\\f", 1, "\\/"]')


# --- the open-string cut's escape drain ------------------------------------------


class TestOpenStringEscapeDrain:
    """A string still OPEN at the cut is the deferred-escape class the
    mark resolves: the cut's repair lane always decides (a truncated
    document never strict-loads), so the in-flight content's raw source
    re-renders through the SAME escape walk the closed-string marks use
    (`render_open_string_repair_lane`, `render_repair_lane_string`'s
    twin), then the engine's unterminated-string finalize rstrips the
    DECODED value (the escape-tail heal's own operand). Byte-exact
    against the engine -- and through it the oracle -- at every
    chunking."""

    DRAIN_FAMILY = [
        # the divergent escape at the tail: the backslash stays a
        # literal character in the value, the re-dump re-escapes it
        ('["\\f', '["\\\\f"]'),
        ('["\\/', '["\\\\/"]'),
        ('["\\x', '["\\\\x"]'),
        ('["a\\f', '["a\\\\f"]'),
        ('["a\\fb', '["a\\\\fb"]'),
        ('["\\fx', '["\\\\fx"]'),
        ('["x\\f', '["x\\\\f"]'),
        ('["\\t\\f', '["\\t\\\\f"]'),
        ('["\\f\\/', '["\\\\f\\\\/"]'),
        ('["\\f%", 1', '["\\\\f%", 1]'),
        ('{"k": "\\f', '{"k": "\\\\f"}'),
        ('{"k": "\\/', '{"k": "\\\\/"}'),
        ('[[ "\\f', '[["\\\\f"]]'),
        ('[1, "\\f', '[1, "\\\\f"]'),
        ('{"a": ["\\f', '{"a": ["\\\\f"]}'),
        ('{"a": "x\\f', '{"a": "x\\\\f"}'),
        # the rstrip is the DECODED value's tail: the escape stands,
        # the raw whitespace and the newline-escape runs around it heal
        ('["\\f ', '["\\\\f"]'),
        ('[" \\f', '[" \\\\f"]'),
        ('["\\f\\n', '["\\\\f"]'),
        ('["\\f\\t', '["\\\\f"]'),
        ('{"k": "\\f\\n', '{"k": "\\\\f"}'),
        ('{"k": "x\\n\\f', '{"k": "x\\n\\\\f"}'),
        ('{"k": "a\\t\\f', '{"k": "a\\t\\\\f"}'),
        # the backslash runs at the cut: the dangling one a literal
        # (parity already, both lanes agree), the odd 3+-run the repair
        # lane's own collapse (the streaming decode's run pairing gave
        # two literal backslashes)
        ('["\\', '["\\\\"]'),
        ('["\\\\', '["\\\\"]'),
        ('["\\\\\\', '["\\\\"]'),
        ('["\\\\\\\\\\', '["\\\\"]'),
        # the smart-quote re-decision in the odd-run collapse: the
        # value's ordinary `„` pushes the lib's low smart-quote span
        # (`_push_low_smart_quote_span`, the active delimiter becoming
        # the `\0` sentinel), so the trailing-run while loop can no
        # longer take the run's own delimiter INTO the value -- the
        # collapsed backslash STANDS and the delimiter joins it as
        # content (`["e„\\\"` -> the value `e„\"`, where the plain
        # content drops it, `["e\\\"` -> `e"`). `„` is the ONLY pusher
        # (no other STRING_DELIMITERS member re-decides: `“`/`”` as
        # content are ordinary chars); `”` pops the span (plain drop
        # again), and spans NEST (`„„` holds through one `”`); the
        # `\`+`"` escape pops it (normalize's low-span arm, tested
        # against the LITERAL `"` whatever the own delimiter is); a `„`
        # hard on a backslash-ending value (the halved even run's own
        # `\`, a decoded `\u005c`) does NOT push.
        ('["e„\\\\\\"', '["e\\u201e\\\\\\""]'),
        ('["„\\\\\\"', '["\\u201e\\\\\\""]'),
        ('["e„\\\\\\"b', '["e\\u201e\\\\\\"b"]'),
        # the run lengths 3/5/7 collapse identically (one backslash
        # kept, the delimiter content)
        ('["e„\\\\\\\\\\"', '["e\\u201e\\\\\\""]'),
        ('["e„\\\\\\\\\\\\\\"', '["e\\u201e\\\\\\""]'),
        # the push is the `„` char's, not the `"` delimiter's: a
        # `'`-delimited string re-decides the same way
        ("['e„\\\\\\'", '["e\\u201e\\\\\'"]'),
        # the other STRING_DELIMITERS members as content never re-decide
        ('["e“\\\\\\"', '["e\\u201c\\""]'),
        ('["e”\\\\\\"', '["e\\u201d\\""]'),
        # the pops: an ordinary `”` (plain drop again), two `”` past one
        # `„` (the span is gone), the `\`+`"` escape mid-value
        ('["e„”\\\\\\"', '["e\\u201e\\u201d\\""]'),
        ('["„””\\\\\\"', '["\\u201e\\u201d\\u201d\\""]'),
        ('["e„\\"x\\\\\\"', '["e\\u201e\\"x\\""]'),
        # the nesting: `„„` holds through one `”`
        ('["„„\\\\\\"', '["\\u201e\\u201e\\\\\\""]'),
        ('["„„”\\\\\\"', '["\\u201e\\u201e\\u201d\\\\\\""]'),
        # the no-push controls: a `„` hard on a backslash-ending value
        # (the halved even run's own `\`, a decoded `\u005c`) stays
        # plain content -- the drop stands
        ('["\\\\„\\\\\\"', '["\\\\\\u201e\\""]'),
        ('["\\\\„e\\\\\\"', '["\\\\\\u201ee\\""]'),
        ('["\\u005c„\\\\\\"', '["\\\\\\u201e\\""]'),
        # the span persists past its own run+delim (content follows), a
        # push after an accepted escape (tab, a decoded `\u0041`), the
        # object context
        ('["e„\\\\\\"x', '["e\\u201e\\\\\\"x"]'),
        ('["e„\\t\\\\\\"', '["e\\u201e\\t\\\\\\""]'),
        ('["„\\u0041\\\\\\"', '["\\u201eA\\\\\\""]'),
        ('{"k": "e„\\\\\\"', '{"k": "e\\u201e\\\\\\""}'),
    ]

    @pytest.mark.parametrize("text,want", DRAIN_FAMILY, ids=lambda p: repr(p[0]))
    def test_the_cut_renders_the_repair_lane_byte_exact(self, text: str, want: str) -> None:
        assert tors.repair_json(text) == want, f"{text!r}: engine moved"
        assert json_repair_lib.repair_json(text) == want, f"{text!r}: oracle moved"
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"
        _, whole = stream(text, [])
        assert whole == want, f"{text!r}: one-push != {want!r}"

    @pytest.mark.parametrize("text,_", DRAIN_FAMILY, ids=lambda p: repr(p[0]))
    def test_snapshot_is_end_and_valid_at_every_prefix(self, text: str, _: str) -> None:
        for k in range(1, len(text) + 1):
            r = tors.JsonRepairer()
            r.push(text[:k])
            snap = r.snapshot()
            _assert_loads_or_empty(snap)
            assert snap == r.end(), f"snapshot != end at prefix {k} of {text!r}"

    # the held partial escape at the cut: a lone high surrogate escape
    # whose tail holds a bare `\u`/`\x` (complete or not, `u00`/`ux`/
    # `u` alike -- anything the decode does not accept). The streaming
    # decode's tail DROPPED the held partial (the stream rendered
    # `["\ufffd"]` where the engine keeps the literal `\u` after the
    # flushed U+FFFD); the drain's predicate now fires on the
    # incomplete escape and the walk renders the engine's own answer.
    # The ORACLE keeps the `\ud83d` escape (the documented lone-
    # surrogate divergence, the same one the closed-string re-render
    # pins below) -- byte-exact against the ENGINE here.
    HELD_PARTIAL_FAMILY = [
        ('["\\ud83d\\u', '["\\ufffd\\\\u"]'),
        ('["\\ud83d\\x', '["\\ufffd\\\\x"]'),
        ('["a\\ud83d\\u', '["a\\ufffd\\\\u"]'),
        ('{"k": "\\ud83d\\u', '{"k": "\\ufffd\\\\u"}'),
        ('[1, "\\ud83d\\u', '[1, "\\ufffd\\\\u"]'),
        ('["\\ud83d\\u00', '["\\ufffd\\\\u00"]'),
        ('["\\ud83d\\ux', '["\\ufffd\\\\ux"]'),
        ('["\\ud83d\\uZ1', '["\\ufffd\\\\uZ1"]'),
        ('["\\ud83d\\x4', '["\\ufffd\\\\x4"]'),
        ('["\\ud83d\\xzz', '["\\ufffd\\\\xzz"]'),
    ]

    @pytest.mark.parametrize("text,want", HELD_PARTIAL_FAMILY, ids=lambda p: repr(p[0]))
    def test_the_held_partial_escape_renders_the_repair_lane(
        self, text: str, want: str
    ) -> None:
        assert tors.repair_json(text) == want, f"{text!r}: engine moved"
        assert want != json_repair_lib.repair_json(text), f"{text!r}: oracle moved"
        for k in range(1, len(text)):
            _, got = stream(text, [k])
            assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"
        _, whole = stream(text, [])
        assert whole == want, f"{text!r}: one-push != {want!r}"

    def test_the_streamed_bytes_are_provisional(self) -> None:
        """The streaming decode's own bytes are provisional exactly as a
        mark's rendering: the push delta carried the canonical decode
        (`["\f`), the close render replaces the span (`["\\f"]`)."""
        r = tors.JsonRepairer()
        delta = r.push('["\\f')
        assert delta == '["\\f'
        assert r.end() == '["\\\\f"]'
        # the clean tail still streams through the escape-tail path
        r2 = tors.JsonRepairer()
        assert r2.push('{"k": "a b ') == '{"k": "a b '
        assert r2.end() == '{"k": "a b"}'


# --- the documented decisions ---------------------------------------------------


class TestDocumentedDecisions:
    def test_truncated_string_heals(self) -> None:
        """The deliverable's own example: a cut-off string value heals to
        the closed quote and container."""
        r = tors.JsonRepairer()
        r.push('{"a": "hel')
        assert r.snapshot() == '{"a": "hel"}'
        assert r.end() == '{"a": "hel"}'

    def test_truncated_containers_heal(self) -> None:
        """The second deliverable example: unclosed containers close with
        their own brackets."""
        r = tors.JsonRepairer()
        r.push('{"a": [1, 2')
        assert r.snapshot() == '{"a": [1, 2]}'
        assert r.end() == '{"a": [1, 2]}'

    def test_partial_literal_tru_is_the_engines_own_partial_semantics(self) -> None:
        """THE DECISION (documented in docs/api.md and the module docs):
        there is NO 'tru' -> 'true' healing. The engine's own partial
        semantics decide: repair_json('tru') is '' (a top-level word is
        prose unless the whole input is one strict JSON value) and
        repair_json('{"a": tru') is '{"a": "tru"}' (a bare word in a
        container is a string). The stream reproduces exactly that, so
        the end()-equals-engine differential holds unbroken. Mid-stream
        the word is held back as a pending token (a push returns the
        text emitted so far, '' here), not passed through raw."""
        assert tors.repair_json("tru") == ""
        r = tors.JsonRepairer()
        assert r.push("tru") == ""
        assert r.end() == tors.repair_json("tru") == ""
        r2 = tors.JsonRepairer()
        assert r2.push('{"a": tru') == '{"a": '
        assert r2.end() == tors.repair_json('{"a": tru') == '{"a": "tru"}'
        # A complete strict scalar commits; a non-JSON word does not.
        assert tors.JsonRepairer().push("12") == ""
        r3 = tors.JsonRepairer()
        r3.push("12")
        assert r3.end() == "12"
        r4 = tors.JsonRepairer()
        r4.push("true")
        assert r4.end() == "true"

    def test_dangling_key_divergence_is_pinned(self) -> None:
        """The documented divergence: a dangling key drops (the engine's
        continuation shape), including the first-member shape where the
        whole-text engine instead falls back to an array."""
        r = tors.JsonRepairer()
        r.push('{"a": "x", "b"')
        assert r.end() == tors.repair_json('{"a": "x", "b"') == '{"a": "x"}'
        r2 = tors.JsonRepairer()
        r2.push('{"a"')
        # The engine's whole-text fallback heals '{"a"' to '["a"]'; the
        # stream drops the dangling member instead ({}), valid JSON with
        # the same information content (none).
        assert r2.end() == "{}"
        assert tors.repair_json('{"a"') == '["a"]'

    def test_trailing_top_level_tokens_drop(self) -> None:
        """The engine keeps trailing junk off the root value; the stream
        drops later top-level VALUES too (the engine may array-wrap
        them; documented divergence, compose with repair_json there)."""
        r = tors.JsonRepairer()
        r.push('{"a": 1} junk')
        assert r.end() == '{"a": 1}'
        r2 = tors.JsonRepairer()
        r2.push('{"a":1}{"b":2}')
        assert r2.end() == '{"a": 1}'
        assert tors.repair_json('{"a":1}{"b":2}') == '[{"a": 1}, {"b": 2}]'

    def test_string_resynchronization_divergence_is_valid(self) -> None:
        """The engine terminates a damaged string at a structural closer;
        the stream closes strings only at the delimiter or end (the
        output stays valid JSON either way)."""
        r = tors.JsonRepairer()
        r.push('{"a": "hello}')
        got = r.end()
        json.loads(got)
        assert got == '{"a": "hello}"}'
        assert tors.repair_json('{"a": "hello}') == '{"a": "hello"}'

    def test_in_container_comment_divergence_is_valid(self) -> None:
        """The engine's in-container comment consumes the value after it;
        the stream skips the comment and parses the value."""
        r = tors.JsonRepairer()
        r.push('{"a": /*x*/ 1}')
        assert r.end() == '{"a": 1}'
        assert tors.repair_json('{"a": /*x*/ 1}') == '{"a": ""}'

    def test_delta_concatenation_and_its_documented_retraction(self) -> None:
        """push returns the newly emitted text; the concatenation is the
        emitted stream, and snapshot/end are the authority when a late
        repair retracts (the dangling-member drop)."""
        r = tors.JsonRepairer()
        deltas = [r.push(part) for part in ('{"a": 1, "b": ', '2 ')]
        # The concatenation is the EMITTED stream (open form): the
        # pending number's chars stream out when the run terminates, and
        # the document's final `}` only exists at close time.
        assert "".join(deltas) == '{"a": 1, "b": 2'
        assert r.end() == '{"a": 1, "b": 2}'
        # The retraction: the dropped member's separator streamed out and
        # was taken back; the caller's concatenation keeps the stale text.
        r2 = tors.JsonRepairer()
        d1 = r2.push('{"a": 1, b')
        d2 = r2.push("}")
        assert "".join([d1, d2]) == '{"a": 1, }'
        assert r2.end() == '{"a": 1}'

    def test_the_strictly_empty_drops_are_local_to_their_array(self) -> None:
        """THE ROUND-5 CONTRACT (docs/api.md and the module docs): the
        strictly-empty element's deferred drop is LOCAL to its own
        array's closing shape -- it drops iff that array closed on a
        trailing separator (a `,` where a value was required), the
        stream cut with the element pending, or the paren taint holds;
        it keeps when the array closed properly -- a valid document or
        a valid root with trailing garbage alike. The review round's
        four breaks, pinned at every chunking (and the valid-keeps
        neighbors with them): the streamed shape at every split set,
        and the engine's own answer with it (except the one documented
        divergence below, pinned to its engine shape)."""
        for text, want, engine in [
            # break 1 (+2): a valid OBJECT keeps its strictly-empty
            # element -- the whole document is strict-valid, the
            # whole-input fast path kept it
            ('{"a": ["" , 1]}', '{"a": ["", 1]}', '{"a": ["", 1]}'),
            # break 3: trailing garbage after a valid root keeps the
            # element (the engine's strict-suffix probe re-decided the
            # valid initial container; a whole-document loads failure
            # does NOT imply the drop)
            ('[[] , 1] x', '[[], 1]', '[[], 1]'),
            ('[true, [] , 1] x', '[true, [], 1]', '[true, [], 1]'),
            # break 4's class (the missing-colon shape with a
            # container): the DOCUMENTED divergence -- the stream keeps
            # the value it saw, the engine's object loop consumed the
            # container-open as the colon-substitute and its repair
            # lane emptied the array
            ('{"a" [[] , 1]}', '{"a": [[], 1]}', '{"a": []}'),
            # the trailing-separator drops: the element's OWN array
            # closed on a `,` where a value was required
            ('[[] ,]', '[]', '[]'),
            ('[1, [] ,]', '[1]', '[1]'),
            ('[[], [] ,]', '[[]]', '[[]]'),
            ('{"a": [[] ,]}', '{"a": []}', '{"a": []}'),
            ('{"a": [1, [] ,]}', '{"a": [1]}', '{"a": [1]}'),
            ('[[] ,] x', '[]', '[]'),
            # the cut drops: the stream ended with the element pending
            # (or the root never closed)
            ('[[] ,', '[]', '[]'),
            ('[[] , 1', '[1]', '[1]'),
            ('[[], [[] , 2], 3', '[[], [2], 3]', '[[], [2], 3]'),
            # the ws-then-value-start drops: the engine's one-char
            # stall guard ate the whitespace, the char re-parsed as
            # the next item
            ('[[] x]', '["x"]', '["x"]'),
            ('[[] x, 1]', '["x", 1]', '["x", 1]'),
            ('[1, [] x, 1]', '[1, "x", 1]', '[1, "x", 1]'),
            ('[[]x]', '[]', '[]'),
            # the paren-taint drops: a paren-derived or paren-enclosed
            # element never defers (the tuple grammar is never
            # strict-valid, the repair lane always decides)
            ('[() ]', '[]', '[]'),
            ('[()]', '[[]]', '[[]]'),
            ('[([] , 1)]', '[[1]]', '[[1]]'),
            # the valid-keeps neighbors
            ('[[], 1]', '[[], 1]', '[[], 1]'),
            ('[[] , 1]', '[[], 1]', '[[], 1]'),
            ('[true, [] , 1]', '[true, [], 1]', '[true, [], 1]'),
            ('[{}]', '[{}]', '[{}]'),
            ('[[] ]', '[[]]', '[[]]'),
            ('[""]', '[""]', '[""]'),
            ('{"a": [[] , 1]}', '{"a": [[], 1]}', '{"a": [[], 1]}'),
        ]:
            assert tors.repair_json(text) == engine, f"engine drifted on {text!r}"
            for k in range(1, len(text) + 1):
                _, got = stream(text, list(range(k)))
                assert got == want, f"splits {list(range(k))} of {text!r}: {got!r} != {want!r}"

    def test_empty_trailing_container_heal_matches_the_engine(self) -> None:
        """The heal class the review round caught: a still-open empty
        container drops whole at an item position and closes at a
        member-value position or the root, and an array's trailing
        strictly-empty member drops — all byte-identical to the
        whole-text engine (and through it the oracle)."""
        for text, want in [
            ("[[", "[]"),
            ("[[{", "[]"),
            ("[1, [", "[1]"),
            ('[{}, [', "[{}]"),
            ("[[1], [", "[[1]]"),
            ('{"a": {"b": [', '{"a": {"b": []}}'),
            ('{"a": [', '{"a": []}'),
            ("[[], []", "[[]]"),
            ("[[[]", "[]"),
            ("[1, []", "[1]"),
            ('[1, "', "[1]"),
            ('[1, -', "[1]"),
            ("[1,-]", "[1]"),
            ("[1e-}", '["1e"]'),
            ("[1e+-}", '["1e+"]'),
            ("(\x0b-\x00", "[]"),
            ("(1, -)", "[1]"),
        ]:
            r = tors.JsonRepairer()
            r.push(text)
            assert r.end() == want, f"{text!r}"
            assert r.end() == tors.repair_json(text), f"{text!r} (engine)"

    def test_missing_key_colon_drops_the_pair(self) -> None:
        """`{: 1}` heals to `{}` (the engine's repair lane: the whole
        pair drops), while the valid shape `{"": 1}` passes through
        untouched."""
        r = tors.JsonRepairer()
        r.push("{: 1}")
        assert r.end() == tors.repair_json("{: 1}") == "{}"
        r2 = tors.JsonRepairer()
        r2.push('{: 1, "b": 2}')
        assert r2.end() == tors.repair_json('{: 1, "b": 2}') == '{"b": 2}'
        r3 = tors.JsonRepairer()
        r3.push('{"": 1}')
        assert r3.end() == '{"": 1}' == tors.repair_json('{"": 1}')

    def test_keyword_family_is_case_insensitive(self) -> None:
        """The literal family matches case-insensitively (the engine's
        observable); the special floats stay exact-spelling; a word born
        from a number run drops its stray leading sign; keys keep their
        casing."""
        for text, want in [
            ('{"a": TRUE}', '{"a": true}'),
            ('{"a": tRuE}', '{"a": true}'),
            ('{"a": FALSE}', '{"a": false}'),
            ('{"a": NULL}', '{"a": null}'),
            ('{"a": NONE}', '{"a": null}'),
            ("[TRUE, FALSE, NULL, NONE, tRuE]", "[true, false, null, null, true]"),
            ('{"a": nan}', '{"a": "nan"}'),
            ('{"a": NAN}', '{"a": "NAN"}'),
            ('{"a": NaN}', '{"a": NaN}'),
            ('{"a": Infinity}', '{"a": Infinity}'),
            ('{"a": INFINITY}', '{"a": "INFINITY"}'),
            ('{"a": -Infinity}', '{"a": -Infinity}'),
            ('{"a": -NaN}', '{"a": "NaN"}'),
            ('{"a": -abc}', '{"a": "abc"}'),
            ('{"a": -TRUE}', '{"a": true}'),
            ('{"a": -12abc}', '{"a": "12abc"}'),
            ('{"TRUE": 1}', '{"TRUE": 1}'),
        ]:
            r = tors.JsonRepairer()
            r.push(text)
            assert r.end() == want, f"{text!r}"
            assert r.end() == tors.repair_json(text), f"{text!r} (engine)"

    def test_top_level_quoted_string_end_matches_the_engine(self) -> None:
        """end() equals the engine on the same total text: a completed
        strict double-quoted top-level string commits (`"hi"` ->
        `"hi"`); a single-quoted one was prose to the whole-text engine
        (`'hi'` -> `''`). Mid-stream a top-level string is a provisional
        scalar (the deltas are empty, the snapshot the end's own answer
        — the pinned snapshot==end invariant); the distinction from the
        whole-text engine is only ever visible in WHAT end() commits."""
        r = tors.JsonRepairer()
        assert r.push("'hi'") == ""
        assert r.snapshot() == ""
        assert r.end() == tors.repair_json("'hi'") == ""
        r2 = tors.JsonRepairer()
        assert r2.push('"hi"') == ""
        assert r2.snapshot() == '"hi"'
        assert r2.end() == '"hi"' == tors.repair_json('"hi"')

    def test_in_flight_word_heals_through_the_literal_table(self) -> None:
        """The engine's repair-lane observable for a word still in flight
        at the cut: the special floats' spellings heal to STRINGS (the
        strict fast path keeps them mid-stream: ``[NaN]`` streams to
        ``[NaN]``, the cut ``[NaN`` heals to ``["NaN"]``), and a word
        born from a number run drops its stray leading sign."""
        for text, want in [
            ("[NaN", '["NaN"]'),
            ("[Infinity", '["Infinity"]'),
            ('{"a": NaN', '{"a": "NaN"}'),
            ('{"a": Infinity', '{"a": "Infinity"}'),
            ('{"a": -Infinity', '{"a": "Infinity"}'),
            ("[1, -NaN", '[1, "NaN"]'),
            # the complete documents keep the floats (the strict class)
            ("[NaN]", "[NaN]"),
            ("[Infinity]", "[Infinity]"),
            ('{"a": NaN}', '{"a": NaN}'),
            ('{"a": -Infinity}', '{"a": -Infinity}'),
        ]:
            r = tors.JsonRepairer()
            r.push(text)
            assert r.end() == want, f"{text!r}"
            assert r.end() == tors.repair_json(text), f"{text!r} (engine)"

    def test_escape_tail_heal_matches_the_engine(self) -> None:
        """The engine's own escape-tail heal: an unterminated string's
        content rstrips (Python's whitespace set), and a closed string
        that ENDS the stream on a newline-run loses it; a tab, a
        backspace, or a mid-string newline passes."""
        for text, want in [
            ('{"k": "a\\n', '{"k": "a"}'),
            ('{"k": "a\\n\\t', '{"k": "a"}'),
            ('{"k": "a\\n ', '{"k": "a"}'),
            ('{"k": "a \\n"', '{"k": "a"}'),
            ('{"k": "a\\n"', '{"k": "a"}'),
            ('{"k": "\\n"', '{"k": ""}'),
            ('{"k": "a\\n\\t"', '{"k": "a\\n\\t"}'),
            ('{"k": "a\\r"', '{"k": "a\\r"}'),
            ('{"k": "a\\t"', '{"k": "a\\t"}'),
            ('{"k": "a "', '{"k": "a "}'),
            ('{"k": "a ', '{"k": "a"}'),
            ('{"k": "a b ', '{"k": "a b"}'),
            ('{"k": "  ', '{"k": ""}'),
            ('{"k": "a\x08', '{"k": "a\\b"}'),
            ('["a\\n', '["a"]'),
            ('{"a": "x\\n", "b": 1}', '{"a": "x\\n", "b": 1}'),
        ]:
            r = tors.JsonRepairer()
            r.push(text)
            assert r.end() == want, f"{text!r}"
            assert r.end() == tors.repair_json(text), f"{text!r} (engine)"

    def test_documented_divergences_stay_valid_and_put(self) -> None:
        """The documented divergence classes: the stream's end() keeps
        the documented shape at every split position, valid JSON or the
        empty sentinel throughout."""
        for text, engine_shape, stream_shape in DIVERGENCES:
            assert tors.repair_json(text) == engine_shape, (
                f"engine drifted on {text!r}"
            )
            for k in range(1, len(text)):
                _, got = stream(text, [k])
                _assert_loads_or_empty(got)
            _, one = stream(text, [])
            assert one == stream_shape, f"{text!r}: {one!r} != {stream_shape!r}"


# --- escape and unicode state across boundaries ---------------------------------


class TestEscapeAndUnicodeBoundaries:
    def test_backslash_split_across_pushes(self) -> None:
        """The escape state survives a chunk boundary: the backslash in
        one push, its escape char in the next (both the valid and the
        invalid-escape shapes)."""
        for text in [r'{"a": "x\ny"}', r'{"a": "x\qy"}', r'{"a": "x\\"}']:
            want = tors.repair_json(text)
            for k in range(1, len(text)):
                _, got = stream(text, [k])
                assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    def test_unicode_escape_split_across_pushes(self) -> None:
        """A \\u escape's hex digits straddle pushes; the escape still
        decodes (and an incomplete one stays literal, engine parity)."""
        for text in [r'{"a": "x\u00e9y"}', r'{"a": "x\u41"}', r'{"a": "x\u00']:
            want = tors.repair_json(text)
            for k in range(1, len(text)):
                _, got = stream(text, [k])
                assert got == want, f"split {k} of {text!r}: {got!r} != {want!r}"

    def test_surrogate_pair_split_across_pushes(self) -> None:
        """A surrogate pair's two \\u escapes straddle pushes: the pair
        still combines (the held high surrogate is machine state); a lone
        one heals to U+FFFD exactly like the engine."""
        text = '{"a": "x\\ud83d\\ude00y"}'
        want = tors.repair_json(text)
        lo = text.index("\\ud83d")
        hi = text.index("\\ude00")
        for splits in ([lo + 3], [lo + 6], [lo + 3, hi + 3], [hi + 6]):
            _, got = stream(text, splits)
            assert got == want, f"splits {splits}: {got!r} != {want!r}"
        lone = '{"a": "x\\ud83dy"}'
        r = tors.JsonRepairer()
        r.push(lone)
        # A lone high escape decodes to U+FFFD (the engine's documented
        # divergence from the oracle), exactly on the x-prefixed shape.
        assert r.end() == tors.repair_json(lone)
        assert r.end() == '{"a": "x\\ufffdy"}'

    def test_raw_astral_char_split_across_pushes(self) -> None:
        """A raw astral character split across two pushes survives (the
        chunks are str's: the split is at a codepoint boundary by
        construction)."""
        text = '{"a": "\U0001f600x"}'
        want = tors.repair_json(text)
        k = text.index("\U0001f600") + 1
        _, got = stream(text, [k])
        assert got == want

    def test_lone_surrogate_str_refused_at_the_boundary(self) -> None:
        """A str holding a lone surrogate cannot enter: the standard
        str-in convention (UnicodeEncodeError at the argument boundary)."""
        r = tors.JsonRepairer()
        with pytest.raises(UnicodeEncodeError):
            r.push('{"a": "\ud83d"}')


# --- value coercion agreement ---------------------------------------------------


class TestValueCoercionAgreement:
    @pytest.mark.parametrize(
        "text",
        [
            '{"i": 42, "f": 1.5, "g": 1.5e3, "b": true, "n": null, "s": "hi"}',
            '{"neg": -17, "exp": 2.5e-4, "big": 123456789012345678901234567890}',
            '{"currency": "1,000", "frac": "1/2", "range": "10-20"}',
            '{"py": [True, False, None], "trunc": "tru"}',
            '{"nested": {"deep": [1, {"x": 2.5}]}}',
        ],
    )
    def test_decoded_values_agree_with_the_whole_text_engine(self, text: str) -> None:
        """json.loads(end()) is the whole-text engine's repaired value,
        dict/int/float/bool/None shapes included."""
        r = tors.JsonRepairer()
        r.push(text)
        got = json.loads(r.end())
        want = tors.repair_json_loads(text)
        assert got == want, f"{text!r}: {got!r} != {want!r}"

    def test_oracle_value_agreement_on_the_coercion_corpus(self) -> None:
        """The oracle's decoded values agree on the same corpus (the
        repaired-complete differential, loads spelling)."""
        for text in [
            '{"i": 42, "f": 1.5, "b": true, "n": null, "s": "hi"}',
            '{"big": 123456789012345678901234567890}',
            '{"py": [True, False, None]}',
        ]:
            r = tors.JsonRepairer()
            r.push(text)
            want = json_repair_lib.repair_json(text)
            assert json.loads(r.end()) == json.loads(want)


# --- statefulness ---------------------------------------------------------------


class TestStatefulness:
    def test_reset_reuses_the_repairer(self) -> None:
        """reset() drops all state: a new document streams clean, the
        old one cannot leak."""
        r = tors.JsonRepairer()
        r.push('{"a": [1, 2')
        r.reset()
        assert r.snapshot() == ""
        r.push('[1, ')
        r.push('2]')
        assert r.end() == "[1, 2]"
        assert tors.repair_json("[1, 2]") == "[1, 2]"

    def test_end_is_idempotent_and_push_refuses_after(self) -> None:
        """end() twice returns the same text; push after end raises
        ValueError naming the reset()."""
        r = tors.JsonRepairer()
        r.push('{"a": "hel')
        first = r.end()
        assert r.end() == first
        with pytest.raises(ValueError, match="reset"):
            r.push("x")

    def test_depth_cap_raises_value_error(self) -> None:
        """Past the 200-container cap, push raises the engine's
        normalized ValueError (the engine's own depth guard)."""
        r = tors.JsonRepairer()
        unit = '{"a": '
        with pytest.raises(ValueError, match="recursion depth"):
            r.push(unit * 200 + "[")
        # reset recovers.
        r.reset()
        r.push(unit * 199 + "1}")
        json.loads(r.end())

    def test_gil_released_during_push(self) -> None:
        """A push of a large chunk leaves the event loop schedulable (the
        machine pass runs detached; the delta marshalling is the only
        held residue)."""
        import asyncio

        from loop_harness import assert_heartbeat_clean

        # Built once, outside the measured op (the op is the repairer's
        # own work).
        payload = '{"a": "%s"}' % ("x" * 4_000_000)

        def op() -> None:
            r = tors.JsonRepairer()
            r.push(payload)
            r.end()

        # The op runs on a worker thread (the test_gil_release.py shape):
        # the machine pass is detached, so the loop ticks at heartbeat
        # granularity while it runs.
        assert_heartbeat_clean(lambda: asyncio.to_thread(op), samples=2)


# --- the linearity pin ------------------------------------------------------------


class TestStreamingLinearity:
    """The anti-pattern this surface exists to kill: re-running a
    whole-text pass per chunk. With the chunk SIZE fixed and the total
    quadrupled, the re-parse shape costs O(total^2 / chunk) (~16x wall
    per quadrupling; the suture/repair-json-stream publish measures ~15x)
    while a linear stream holds ~4x. Two gates, tests/test_scaling_pins.py's
    idiom: the doubling gate on the streaming wall itself, and the
    streaming-vs-one-shot ceiling (a fixed 64-chunk stream must not cost
    an order of magnitude more than one repair of the same text)."""

    CHUNK = 4_096

    def _document(self, total: int) -> str:
        # Nested containers with strings, numbers and literals: the
        # machine's real shape, generated deterministically.
        parts = []
        size = 0
        i = 0
        while size < total:
            parts.append(f'{{"k{i}": "v{i}", "n{i}": [{i}, {i}.5, true, None]}}')
            size += len(parts[-1]) + 2
            i += 1
        return "[" + ", ".join(parts) + "]"

    def _stream(self, text: str) -> None:
        r = tors.JsonRepairer()
        for i in range(0, len(text), self.CHUNK):
            r.push(text[i : i + self.CHUNK])
        r.end()

    @pytest.mark.timing
    def test_total_time_stays_linear_at_a_fixed_chunk_size(self) -> None:
        """128KiB -> 512KiB at 4KiB chunks (32 -> 128 pushes): measured
        2.55ms -> 10.87ms, ratio 4.26 (~2.1x per doubling), gate 3.0x per
        doubling. The per-chunk re-parse anti-pattern measures ~16x here
        (quadratic in the total at a fixed chunk size)."""
        small_doc, large_doc = self._document(128 * 1024), self._document(512 * 1024)
        assert len(large_doc) // len(small_doc) >= 3

        def measure() -> tuple[float, float]:
            return min_wall_ms(lambda: self._stream(small_doc)), min_wall_ms(
                lambda: self._stream(large_doc)
            )

        def check(walls: tuple[float, float]) -> None:
            small, large = walls
            factor = len(large_doc) / len(small_doc)
            doublings = math.log2(factor)
            allowed = LINEAR_GATE_PER_DOUBLING**doublings
            assert large < allowed * small, (
                f"streaming grew {small:.2f}ms -> {large:.2f}ms for a "
                f"{factor:.1f}x document ({large / small:.2f}x, allowed "
                f"{allowed:.1f}x at {LINEAR_GATE_PER_DOUBLING}x per doubling): "
                "the machine regressed superlinear in the total bytes"
            )

        first_clean(measure, check, samples=3, label="the streaming linearity pin")

    @pytest.mark.timing
    def test_streaming_stays_within_an_order_of_the_one_shot(self) -> None:
        """The 512KiB stream (128 pushes) against ONE whole-text repair
        of the same text: measured 0.54x (the stream is the same single
        pass, minus the engine's re-serialization, plus the per-push
        delta returns), ceiling 10x: the re-parse anti-pattern lands
        ~30x+ here and blows through."""
        doc = self._document(512 * 1024)

        def measure() -> tuple[float, float]:
            return min_wall_ms(lambda: self._stream(doc)), min_wall_ms(
                lambda: tors.repair_json(doc)
            )

        def check(walls: tuple[float, float]) -> None:
            streaming_ms, one_shot_ms = walls
            assert streaming_ms < 10 * one_shot_ms, (
                f"streaming ({streaming_ms:.2f}ms) cost more than 10x the "
                f"one-shot repair ({one_shot_ms:.2f}ms): per-chunk work is "
                "no longer linear in the chunk"
            )

        first_clean(measure, check, samples=3, label="the streaming-vs-one-shot pin")


class TestCloseTimeLinearity:
    """The close render's own growth gate: the deferred-escape marks
    resolve at `end()`/`snapshot()` close time, and a per-mark in-place
    rewrite of the emitted text (`replace_range` per mark, descending)
    is quadratic in the mark count -- the weekly-fuzz cut stream of
    `\\/`-strings measured 0.29s -> 1.20s -> 4.90s -> 20.43s at
    45k/90k/180k/360k marks (4.2x per doubling). The rebuild is one
    ascending pass (the segments between marks copied once, each
    resolved rendering spliced at its span), ~2.1x per doubling here.
    tests/test_scaling_pins.py's idiom: min-of-samples walls, retried
    on the first dirty sample."""

    # The 4x-step growth gate: a linear close measures ~4x per
    # quadrupling (~2.1x per doubling at these sizes); the quadratic
    # relapse measures ~16x per quadrupling and blows through 6x.
    GATE_PER_QUAD = 6.0

    @staticmethod
    def _document(marks: int) -> str:
        # The cut divergent-escape class: an array of `\\/`-holding
        # strings, the root never closed -- every string arms a
        # deferred-escape mark and the close render resolves every one
        # through the repair lane's escape normalization.
        return "[" + ", ".join(f'"\\/{i}"' for i in range(marks))

    @staticmethod
    def _prepared(marks: int) -> tors.JsonRepairer:
        r = tors.JsonRepairer()
        r.push(TestCloseTimeLinearity._document(marks))
        return r

    @pytest.mark.timing
    def test_the_close_resolves_the_marks_linearly(self) -> None:
        """The close render at 45k/90k/180k/360k marks (4x per step):
        measured 6.4ms -> 13.2ms -> 24.4ms -> 50.4ms (~2.1x per
        doubling), gate 6x per quadrupling. `snapshot` is pure and
        renders closed exactly what `end` renders, so the wall is the
        close's own (the push is paid once, outside the measurement)."""

        def measure() -> list[float]:
            return [
                min_wall_ms(lambda r=r: r.snapshot())
                for r in (self._prepared(m) for m in (45_000, 90_000, 180_000, 360_000))
            ]

        def check(walls: list[float]) -> None:
            # Content-match first: the close actually resolved the
            # marks (the re-rendered spelling is out, not the verbatim
            # `\/` decode -- and every element's mark is in the count).
            resolved = self._prepared(100)
            out = resolved.snapshot()
            assert out.startswith('["\\\\/0'), out[:24]
            assert out.count('", "') == 99, "a mark stopped resolving"
            for i in range(len(walls) - 1):
                small, large = walls[i], walls[i + 1]
                assert large < self.GATE_PER_QUAD * small, (
                    f"the close render grew {small:.2f}ms -> {large:.2f}ms for a "
                    f"2x mark count ({large / small:.2f}x, allowed "
                    f"{self.GATE_PER_QUAD:.1f}x per 4x of marks): the close-time "
                    "resolution regressed superlinear in the mark count"
                )

        first_clean(
            measure, check, samples=3, label="the close-time linearity pin"
        )


# --- hypothesis: valid documents through random chunkings --------------------------


class TestHypothesisValidDocuments:
    @given(st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False),
                        lambda leaf: st.lists(leaf, max_size=4)
                        | st.dictionaries(st.text(min_size=1, max_size=6), leaf, max_size=4),
                        max_leaves=20))
    @settings(max_examples=50, deadline=None)
    def test_valid_document_survives_every_chunking(self, doc) -> None:
        """Any JSON document, streamed through random chunk boundaries,
        ends at the engine's own canonical answer for it."""
        text = json.dumps(doc)
        if len(text) < 2:
            return  # a one-char document has no split points
        rng = random.Random(len(text))
        points = sorted(rng.sample(range(1, len(text)), min(5, len(text) - 1)))
        _, got = stream(text, points)
        assert got == tors.repair_json(text)
