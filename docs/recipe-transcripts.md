# Recipe: chunking threads and transcripts

The realistic shape of splitting line-oriented conversational text (a chat
thread one message per line, a WebVTT or SRT subtitle track) for embedding
or context-window packing without cutting mid-message or mid-cue. `tors`
cuts on lines and blank-line-separated blocks, does no format parsing (see
the end), and shows the real return value at each step below.

## 1. A chat thread, one message per line

For a budget-aware split that never severs a message, put the line level
first and splice the default accurate hierarchy in below it with a `None`
entry:

```python
import tors

thread = (
    "Nathan: kicking off the sync.\n"
    "Priya: We briefed the U.S. team on the numbers. "
    "They asked for a follow-up meeting. The budget holds.\n"
    "Nathan: done."
)

chunks = tors.chunk_hierarchical(thread, 60, ["\n", None])
# [(0, 29), (30, 78), (78, 131), (132, 145)]

[thread[s:e] for s, e in chunks]
# ['Nathan: kicking off the sync.',
#  'Priya: We briefed the U.S. team on the numbers. ',
#  'They asked for a follow-up meeting. The budget holds.',
#  'Nathan: done.']
```

Messages that fit stay whole; the oversized Priya message falls back to the
UAX #29 sentence segmenter (the `None` entry's doing), not to literal
guesses, which matters the moment a message contains an abbreviation:

```python
tors.chunk_hierarchical(thread, 40, ["\n", ". ", " "])[1]
# (30, 55)  -> "Priya: We briefed the U.S": the naive ". " list severs the name
tors.chunk_hierarchical(thread, 40, ["\n", None])[1]
# (30, 69)  -> "Priya: We briefed the U.S. team on the ": the splice does not
```

For fixed windows of N messages regardless of budget, `chunk_by_lines` is
the count-based twin: blank lines ride along but never count, so
`chunk_by_lines(thread, 2)` is two messages per window. At multi-MiB scale
reach for `chunk_by_lines_iter` or `await tors.aio.chunk_by_lines(...)`.

```python
tors.chunk_by_lines(thread, 2)
# [(0, 131), (132, 145)]
```

## 2. A WebVTT transcript

Cue blocks are blank-line separated, and a run of 2+ newlines is exactly
what the default hierarchy's paragraph level cuts on, so the default
(`chunk_hierarchical(vtt, max)`, equivalently `[None]`) already makes
cue-aligned cuts, falling back to sentences only inside an oversized cue:

```python
vtt = """WEBVTT

00:00:01.000 --> 00:00:04.000
<v Nathan>Welcome to the weekly sync.

00:00:04.000 --> 00:00:09.500
<v Priya>Thanks. We briefed the U.S. team on the numbers yesterday, and they asked for a follow-up meeting. The budget holds.

00:00:09.500 --> 00:00:12.000
<v Nathan>Great. Let's aim for Thursday then."""

chunks = tors.chunk_hierarchical(vtt, 120)
# [(0, 75), (77, 124), (124, 232), (234, 309)]

[vtt[s:e] for s, e in chunks]
# ['WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Nathan>Welcome to the weekly sync.',
#  '00:00:04.000 --> 00:00:09.500\n<v Priya>Thanks. ',
#  'We briefed the U.S. team on the numbers yesterday, and they asked for a follow-up meeting. The budget holds.',
#  "00:00:09.500 --> 00:00:12.000\n<v Nathan>Great. Let's aim for Thursday then."]
```

Every cut except the one inside the oversized Priya cue lands on a cue gap;
that cue falls to sentence boundaries, its timestamp line riding with the
first piece. The `WEBVTT` header is ordinary text; strip it yourself first
if you don't want it riding with the first chunk.

## 3. An SRT file

The same blank-line cue structure, so the same default hierarchy gives
cue-aligned cuts. The comma timestamps (`00:00:01,000`) are ordinary text:
nothing parses them; the alignment comes entirely from the cue gaps:

```python
srt = """1
00:00:01,000 --> 00:00:04,000
Welcome to the weekly sync.

2
00:00:04,000 --> 00:00:09,500
Thanks. We briefed the U.S. team on the numbers yesterday, and they asked for a follow-up meeting. The budget holds."""

chunks = tors.chunk_hierarchical(srt, 120)
# [(0, 59), (61, 101), (101, 209)]

[srt[s:e] for s, e in chunks]
# ['1\n00:00:01,000 --> 00:00:04,000\nWelcome to the weekly sync.',
#  '2\n00:00:04,000 --> 00:00:09,500\nThanks. ',
#  'We briefed the U.S. team on the numbers yesterday, and they asked for a follow-up meeting. The budget holds.']
```

## 4. One oversized cue, split at sentence boundaries

When the application has already parsed the file and holds a single cue's
text, `tors.sentence_bounds` is the whole job: UAX #29 boundaries as
offsets, trailing spaces attached to the preceding sentence per the standard:

```python
cue = "Thanks. We briefed the U.S. team on the numbers yesterday, and they asked for a follow-up meeting. The budget holds."

tors.sentence_bounds(cue)
# [(0, 8), (8, 99), (99, 116)]
```

Slice the cue with the pairs, then re-attach whatever per-cue metadata
(timestamp, speaker) only the application knows about.

## What tors does not do

`tors` does not parse or validate cue formats. Timestamps, `-->` arrows,
sequence numbers, and `<v Speaker>` voice spans are ordinary text: a
malformed timestamp sails through as content. Voice spans and token counts
are likewise out (grouping a speaker's turns is application-level work over
parsed cues; every budget here is a codepoint budget). Format semantics
belong to the application: parse with a real subtitle library and hand
`tors` the payload for split points, the split a transcript-ingestion
pipeline makes (`webvtt-py` for parsing, `tors.sentence_bounds` for the
offsets).
