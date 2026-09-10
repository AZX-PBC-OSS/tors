# Recipe: ingesting a real document

The realistic shape of turning arbitrary bytes (an uploaded file, a scraped
page, an OCR'd PDF's raw text layer) into clean, chunked text ready for
downstream use (embedding, indexing, feeding to a model). Every step below
composes functions documented individually in the [API reference](api.md);
this page shows the pipeline shape and the real return value at each stage.

## 1. Establish the encoding

Bytes from an upload or a scrape carry no reliable encoding metadata. Check
UTF-8 validity first (the common case, and the cheap check) and only reach
for a heuristic guesser on the bytes that fail it:

```python
import tors

raw = "Café société — déjà vu".encode("windows-1252")

if tors.utf8_is_valid(raw):
    text = tors.decode_utf8(raw)
else:
    codec = tors.detect_encoding(raw)
    text = raw.decode(codec)

text
# "Café société — déjà vu"
```

`tors.utf8_is_valid` costs nothing but the SIMD scan: no exception flow, no
throwaway `str`. `tors.detect_encoding` is a heuristic guesser (`chardetng`,
the detector Firefox ships), not a validator: it always returns some codec
name, so it only belongs after `utf8_is_valid` has already ruled out the
common case. The returned name is a real `bytes.decode()` codec name
(`"windows-1252"` above), so the final decode is a plain stdlib call, not a
`tors` function: `tors` doesn't ship a decoder for encodings it can only
guess at, since a wrong guess should surface as a normal `UnicodeDecodeError`
from the stdlib, not a silently-successful call into `tors`.

If the source is already known-UTF-16 (a Windows-authored `.txt`, some SDK
exports), `tors.decode_utf16(raw, byteorder=...)` is the equivalent entry
point and takes the same `errors="strict"`/`"replace"` choice as
`decode_utf8`. There's no `detect_encoding` step for UTF-16: the BOM (or an
explicit `byteorder=`) is the only signal, and `tors.utf16_is_valid` answers
the validity question the same way `utf8_is_valid` does for UTF-8.

For a source that's reliably UTF-8 (an API response, a database column),
skip the branch entirely and call `tors.decode_utf8` directly. The
`errors="strict"` default raises the same `UnicodeDecodeError` the stdlib
would on malformed bytes, so nothing is silently swallowed.

## 2. Normalize

Whatever the source, run the decoded text through `tors.normalize` before
anything else touches it. It folds `\r\n`/`\r` to `\n`, drops trailing
whitespace before a newline, collapses 3+ blank lines to exactly 2, and NFC-
normalizes, the shape PDF extraction and OCR output reliably need:

```python
messy = "Line one   \n\n\n\nLine two\r\nLine three  "
tors.normalize(messy)
# "Line one\n\nLine two\nLine three"
```

If the input was already clean, `normalize` returns the original string
object unchanged (`tors.normalize(s) is s`) rather than a fresh allocation,
so running it unconditionally on every document costs nothing when there
was nothing to fix.

## 3. Strip markdown wrapping, if the source might carry it

Scraped pages and model output frequently arrive as prose with embedded code
fences, or as a single fenced block wrapping the whole response. Handle both
shapes explicitly rather than guessing which one you have:

```python
md = "Some notes.\n\n```python\nprint('hi')\n```\n\nMore prose after."

tors.extract_code_blocks(md)
# [('python', "print('hi')\n", 13, 39)]

# a whole response wrapped in exactly one fence unwraps automatically;
# anything else (prose around a fence, no fence, multiple fences) is
# returned completely unchanged, so it's safe to call unconditionally
only_fence = "```python\nprint('hi')\n```"
tors.strip_code_fences(only_fence)
# "print('hi')\n"

tors.strip_code_fences(md) == md
# True: md has prose around its fence, so nothing is touched
```

`extract_code_blocks` is the tool when you want the code and the prose
separately (e.g. code goes to one index, prose to another). `strip_code_fences`
is the tool when you just want the fence gone from an otherwise-code
response and don't want to write the "is this the single-fence case" check
yourself. If neither applies to your source (plain text, no markdown)
skip this step.

JSON model output is the next lane over, same machinery:
`tors.repair_json` / `tors.repair_json_loads` repair malformed JSON (missing
commas and quotes, truncated containers, stray prose) in one call,
and unwrap the single-fence case themselves via this same fence grammar, so
a response wrapped in exactly one json-tagged fence needs no extraction
step at all. For multiple blocks, compose the two:
`tors.repair_json_loads(code)` over each `code` from
`tors.extract_code_blocks(md, lang="json")`.

## 4. Chunk

Pick the chunker that matches your document's structure. For structured
markdown (headers, paragraphs), `chunk_hierarchical` tries coarser
separators first and falls back automatically:

```python
doc = (
    "# Title\nIntro paragraph here with some words.\n\n"
    "## Section One\nContent for section one goes here and continues a bit further.\n\n"
    "## Section Two\nMore content for section two, also fairly short."
)

chunks = tors.chunk_hierarchical(doc, 80, ["\n## ", "\n\n", ". ", " "])
# [(0, 46), (50, 125), (129, 189)]

[doc[s:e] for s, e in chunks]
# ['# Title\nIntro paragraph here with some words.\n',
#  'Section One\nContent for section one goes here and continues a bit further.\n',
#  'Section Two\nMore content for section two, also fairly short.']
```

`chunk_hierarchical` returns `(start, end)` codepoint offsets into the
*original* string, not copies of the text; slice `doc` yourself as shown
above. The separator itself is dropped between chunks (unlike
`tors.chunk_text`'s lossless-partition contract), which is what you want
when splitting on a header marker: the header text stays with the section
it introduces, and the marker itself isn't duplicated into both chunks. A
`separators` entry may also be `None`, splicing the default accurate
hierarchy in at that position; the one-message-per-line shape
`["\n", None]` is the subject of the
[transcripts recipe](recipe-transcripts.md).

For plain prose with no structure to key off, `chunk_by_sentences` (or
`chunk_by_words` for a token-count budget) is the simpler choice:

```python
prose = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
chunks = tors.chunk_by_sentences(prose, 2)
# [(0, 44), (44, 90)]
[prose[s:e] for s, e in chunks]
# ['This is sentence one. This is sentence two. ',
#  'This is sentence three. This is sentence four.']
```

Every chunker in this family returns offsets, not strings: the pipeline's
last step is always "slice the normalized text with these pairs." None of
these functions makes a retrieval-quality claim for any particular
downstream task; they guarantee the mechanical contract (correct boundaries,
genuine overlap when requested) and nothing about how well a given chunk
size will perform for your model.

## The whole pipeline, together

```python
import tors


def ingest(raw: bytes, *, markdown: bool = False) -> list[str]:
    if tors.utf8_is_valid(raw):
        text = tors.decode_utf8(raw)
    else:
        text = raw.decode(tors.detect_encoding(raw))

    text = tors.normalize(text)

    if markdown:
        text = tors.strip_code_fences(text)

    chunks = tors.chunk_hierarchical(text, 500, ["\n## ", "\n\n", ". ", " "])
    return [text[s:e] for s, e in chunks]
```

Swap `chunk_hierarchical` for `chunk_by_sentences`/`chunk_by_words` if the
source isn't markdown-structured; both slot into the same last line.
