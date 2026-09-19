# API reference

Every function releases the GIL for its whole native pass (`py.detach`); the
GIL-held residue of a call is only pyo3's argument borrow and the return
marshalling. The one deliberate exception is `utf8_byte_len`'s no-detach
spelling — its whole body is the argument borrow plus an O(1) read, so a
detach would bracket no work and starve a co-resident event loop's
heartbeat instead of feeding it (the inverted static pin in
`tests/test_utf8_byte_len.py` holds this both ways: utf8 must NOT detach,
utf16's twin must). Signatures below are the typed surface of
`python/tors/__init__.pyi`, pinned to the live functions by
`tests/test_pyi_drift.py`.

## `tors.normalize`

```python
def normalize(text: str) -> str: ...
```

Applies, in order:

1. Unicode NFC normalization.
2. `\r\n` and `\r` folded to `\n`.
3. Trailing whitespace (spaces/tabs) before a newline is dropped.
4. Runs of 3 or more consecutive newlines collapse to exactly 2.
5. A final strip, using Python's exact whitespace set (including the `0x1c`-`0x1f`
   separator control characters, which `str.strip()` treats as whitespace but Rust's
   `char::is_whitespace()` does not).

This is a common preprocessing pipeline for text extracted from PDFs, OCR, and other messy
sources.

### GIL behavior

The entire transform runs inside `py.detach` (PyO3's GIL-release call): the GIL is free
for the rest of your program for the whole call, not just part of it. This is the reason
`tors` exists: Python's `re` module and `str` methods never release the GIL regardless of
input size, so the equivalent pure-Python pipeline holds the GIL for its whole duration no
matter what thread it runs on.

### Identity return

When the complete pipeline is a no-op on the input, i.e. already NFC (Unicode
quick-check Yes), no CR, no spaces/tabs before a newline, no 3+ newline runs, and no
whitespace at either end, `normalize` returns the original input object:

```python
tors.normalize(s) is s  # True whenever the pipeline changes nothing
```

Zero allocation, zero copy, zero marshalling (the same idiom CPython's
`unicodedata.normalize` fast path uses). An output-equals-input comparison after any
full pass extends the guarantee beyond provably-clean input: `tors.normalize(s) is s`
holds exactly whenever `tors.normalize(s) == s`. The same contract applies to
`tors.nfc` / `nfd` / `nfkc` / `nfkd`, `tors.html_unescape`, and the string element of
`tors.finalize` / `tors.finalize_utf8`.

### Example

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"
```

**Async**: `await tors.aio.normalize(...)` runs this under `asyncio.to_thread` (see [Async use](async.md); the facade is a submodule — `import tors.aio` once, not an attribute of `import tors`).

## `tors.finalize`

```python
def finalize(text: str) -> tuple[str, str]: ...
```

`(normalize(text), sha256)` in the same single GIL-released pass, where `sha256` is the
lowercase-hex SHA-256 of the normalized text's UTF-8 bytes: byte-identical to
`hashlib.sha256(normalized.encode("utf-8")).hexdigest()`. It exists for pipelines that
normalize then content-hash (dedupe gates), collapsing two whole-text passes into one.
On the identity path the digest is computed straight from the borrowed input buffer and
the string element is the original object (`tors.finalize(s)[0] is s` when the pipeline
changes nothing). A `str` holding lone surrogates is refused at the argument boundary
with `UnicodeEncodeError` ("surrogates not allowed"): pyo3's `&str` extraction
behavior, the same boundary every str-in function here documents.

```python
text, digest = tors.finalize("line one  \n\n\n\nline two\r\n")
# ("line one\n\nline two", "e986ba08...f7b942a")
```

**Async**: `await tors.aio.finalize(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.strip_controls`

```python
def strip_controls(text: str) -> str: ...
```

Replace every maximal run of C0 controls (`U+0000`–`U+001F`, tabs and newlines
included) and DEL (`U+007F`) with a single ASCII space, one GIL-released native
pass: byte-identical to `re.compile(r"[\x00-\x1f\x7f]+").sub(" ", text)`. The
scrub model-authored display text needs before it is stored or served (a chatty
or injected model cannot plant terminal control sequences in a row the UI
renders).

Two scope cuts. C1 controls (`U+0080`–`U+009F`) pass through
untouched: the adopted call-site regexes do not cover them either, so covering
them here would silently change adopted behavior; C1 scrubbing is a follow-up
with its own contract, not a silent extension of this one. And `\t`, `\n`,
`\r` are scrubbed (they are C0): do not reach for this on multi-line prose you
want to keep line-shaped. No edge strip either: a control run at either end
becomes an edge space for the caller to `.strip()`.

`tors.strip_controls(s) is s` exactly when `s` holds no C0/DEL character.

```python
tors.strip_controls("score: 4\x00\x01great\x7f")
# "score: 4 great "
```

**Async**: `await tors.aio.strip_controls(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.scrub_pii`

```python
def scrub_pii(
    text: str,
    rules: Sequence[Literal["contact_email", "contact_phone", "api_keys"]] | None = None,
    *,
    salt: str | None = None,
    families: Sequence[str] | None = None,
) -> str: ...
```

Replace contact material — email addresses and phone numbers —
and credential material — provider/platform API keys — inside
free text with correlation tokens, in one GIL-released pass. The call-site
driver: telemetry is the one store a data purge cannot reach, so an error
excerpt, rejection message, or response-body excerpt that echoes a person's
address or number — or the credential the request authenticated with —
must be scrubbed before it reaches the log: provider and platform error
text can quote the credential back (five private consumers evidenced; the
strongest, a platform whose own code comments that a vendor auth failure
"can quote the key" and keeps the full text in an admin-served ledger).
The contact rules are a port of a private consumer's telemetry-safety
module, pinned byte-identical to it at `salt=""`; the api_keys rule is an
extension past that contract.

What the tokens are: `@domain~<digest>` for an email (the domain is the
non-identifying half an operator actually reasons about — "the ambiguity is on
the corporate domain"), `prefix~<digest>` for a phone number, where `prefix`
is the dialling prefix of a `+`-led match: its first three code points (a
canonical E.164's country code: `"+47"` compact, `"+1 "` for the
international spelling of a NANP number, where the third code point is the
space), while any other spelling gets the digest alone: a domestic match's
head digits are the area code, the identifying half of the number, exactly
the context a visible prefix would surface, and `<family prefix>~<digest>`
for an API key, where the family
prefix is kept VERBATIM (`sk-`, `sk-ant-`, `github_pat_`, `AIza`, `Bearer`) —
the non-secret half that tells the operator WHICH credential to rotate. In
every rule `<digest>` is the first 12 hex chars of `sha256(salt + match)`.
A token is a correlation handle, not a secret: it lets an operator tie two log
lines to the same address without the record holding the address.

```python
tors.scrub_pii("unknown candidate fungai.chetima@example.com called from +14155552671 twice")
# "unknown candidate @example.com~3aa8d1d0bb1a called from +14~115f5ee5ea90 twice"

tors.scrub_pii("ring +1 (415) 555-2671 about ticket 4096")
# "ring +1 ~dc750721a848 about ticket 4096"
```

The three rules, a closed set (anything else is a `ValueError` naming it):

- `contact_email` — the deliberately permissive local part
  `[A-Za-z0-9._%+\-]+`, a domain of ASCII letters/digits/dots/hyphens, and a
  two-plus-letter ASCII tail after the largest dot the greedy backtracking
  can reach (`a@b.co9` scrubs `a@b.co` and leaves the `9`; `a@b.c.d` never
  matches; `a@.b.co` matches with the domain kept verbatim). URLs, `mailto:`
  links, and code spans get no special treatment: whatever email sits inside
  them matches — over-matching costs a token where a literal string would
  have read fine, while under-matching leaks. Plus-addressed spellings match
  whole, the tag included: `ada+tag@azx.io` is one match, multiple tags and
  a trailing `+` included (pinned in the battery).
- `contact_phone`, two matchers —
  - *international*: anchored on a literal `+` because a bare digit run is an
    order number, byte count, or timestamp, and redacting that would destroy
    the diagnostic the scrubber exists to preserve. The grammar is the
    ported `\+\d[\d\-. ()]{6,}\d`: a digit directly after the `+`, six or
    more middle class characters, a final digit — the middle counts
    separators, so a long spelling matches on seven digits
    (`+1 415 555`) while `+1234567` never does. The separators are the ones
    humans and upstream APIs use (`( ) - . ` and space); the digit class is
    Unicode Nd (every decimal-digit script); and a `~` or a second `+`
    breaks a run. One run is ONE match: two numbers split by one space
    (`+4712345678 1234567890`) are a single match with a single digest
    over the whole run (space-bridged over-merge, documented not fixed).
    A `+` before a run marks international intent for the whole run: the
    grammar matches, or the run is the ported non-match, and the domestic
    matcher never fires behind a `+` — `+ (415) 555-2671` stays untouched,
    exactly as the source leaves it. `ticket 4096` above survives
    untouched.
  - *domestic* (the extension past the ported source): un-plussed NANP
    shapes — a full run of exactly ten digits, or eleven with an ASCII
    leading `1`, in any `( ) - . ` spelling, `1 (415) 555-2671` and
    `1-415-555-2671` included. THREE discipline rules, all load-bearing:
    (1) the match must carry at least one SEPARATOR, so a bare unseparated
    digit run is never scrubbed even at exactly ten digits — it is an
    order number or id (the same reasoning that anchors the
    international matcher on `+`), and the requirement is also what
    keeps a token's own digest hex unmatchable, so phone-only stays
    strictly idempotent and scrub-twice convergence cannot be chained
    adversarially through chosen digests; (2) the match must start at a
    CLEAN boundary — its first char glued to neither `~` nor a lowercase
    `a`-`f` (the token-interior alphabet: hex `a`-`f` plus `~`, exactly —
    conjunction, not disjunction: either predecessor alone is dirty; not
    "letters" — `g`/`z`/`A`-`F` are clean and still match), so a run
    starting inside a token digest can never flow out through a separator
    into following text (`…~e292cb255128 4096` stays untouched) — and a
    token span itself (`~` + 12 digest hex) is a breaker: runs never
    start inside a digest (the scan resumes after the token instead of
    spending a composed digest-plus-number run whole, which would
    silently swallow the real number after it), and the byte after a
    token is a clean boundary even when the digest ends hex-dirty, so
    an adjacent number still scrubs exactly; and
    (3) no partial match inside a longer run — twelve-plus digits is an
    id, not a phone. Leading spaces are skipped (word separation); a
    leading structural separator (`-`, `.`, `(`, `)`) is ABSORBED into
    the match, not stripped (`x -415-555-2671` scrubs `-415-555-2671`
    whole); trailing separators survive past the last digit, same as
    international; and eleven digits not led by ASCII `1` are excluded
    (the NANP trunk-prefix shape) — ten-digit runs carry no
    leading-digit check at all, so a `0`-led ten-digit shape
    (`020-794-6095`) scrubs like any other and only its eleven-digit
    `0`-led spelling stays out (the `+` form is the international
    spelling of those). Its token is the digest alone, no digits of the
    number in it; the `+`-led spelling of the same digits is what keeps
    the dialling prefix.
    One reachable interaction is documented rather than fixed: an email
    token whose DOMAIN spells a domestic number (`user@555.1234567.co`)
    has its digit half re-tokenized by the phone pass — over-redaction
    in the safe direction, converging on the second scrub like every
    other shape.
- `api_keys` (the credential extension past the ported source) — the
  evidence-backed closed set of provider/platform key families, each a
  literal prefix plus a minimal tail (the shared `[A-Za-z0-9_-]`
  alphabet except where the table names its own) consumed MAXIMALLY
  (a key glued to further charset material is one long key):

  | family | shape (`tail` is `[A-Za-z0-9_-]` except where the row names its own alphabet) | token prefix |
  |---|---|---|
  | OpenAI | `sk-` / `sk-proj-` / `sk-svcacct-` + tail{20,} | the matched prefix verbatim |
  | Anthropic | `sk-ant-` + tail{20,} | `sk-ant-` |
  | Google | `AIza` + tail{35,} | `AIza` |
  | Fireworks | `fw-` / `fw_` + tail{20,} | verbatim |
  | Modal | `ak-` / `wk-` + tail{20,} | verbatim |
  | GitHub | `ghp_` / `gho_` / `ghu_` / `ghs_` / `ghr_` + tail{36,}; `github_pat_` + tail{22,} | verbatim |
  | GitLab | `glpat-` / `glagent-` / `glsoat-` / `glrtr-` / `glcbt-` / `glptt-` / `glimt-` / `gloas-` / `glft-` / `gldt-` / `glrt-` / `glwt-` / `glffct-` + tail{20,} and `_gitlab_session=` + `[A-Za-z0-9+/=]{40,}` (the token-prefix set of GitLab's documented token overview, the session-cookie marker included) | verbatim |
  | Minted | `azxdev_` + tail{20,}; `wd-` / `w-` + tail{43,}; `cn-` + tail{20,} | verbatim |
  | JWT | `Bearer eyJ` + three base64url segments, single-dot separated | `Bearer` |
  | AWS | `AKIA` / `ASIA` + `[0-9A-Z]{16,}`; `A3T` + `[0-9A-Z]{17,}`; `AGPA` / `AIDA` / `AIPA` / `ANPA` / `ANVA` / `AROA` + `[0-9A-Z]{16,}` (the access-key-ID regex's full prefix set — the legacy and resource-ID siblings the same regex carries, every row 20 chars total) | the matched head verbatim |
  | XAI | `xai-` + tail{20,} | `xai-` |
  | GCP OAuth | `ya29.` + tail{20,} | `ya29.` |
  | PEM | `-----BEGIN <words> PRIVATE KEY-----` … `-----END <same words> PRIVATE KEY-----` (the PGP label's ` PRIVATE KEY BLOCK-----` close accepted the same way, both markers required); an unterminated BEGIN is a non-match and the whole block is the match (the PKCS#8 bare `BEGIN PRIVATE KEY` header carries no algorithm words and is excluded) | `PEM` |
  | Azure | `AccountKey=` + `[A-Za-z0-9+/=]{40,}` | `AccountKey=` |

  AWS `AKIA`/`ASIA` access-key IDs are now INCLUDED, evidence-backed
  (the table's `aws` row). Deliberately EXCLUDED, on zero evidence:
  Slack `xox…` and Stripe `sk_live_`/`pk_live_`. Three further
  exclusions are shape decisions, not evidence gaps, each with its why:
  the AWS secret key (the 40-char secret carries no public prefix —
  undetectable without false-positive shape matching), Azure client
  secrets (no distinctive public prefix), and Mistral keys (no
  distinctive public prefix). One exclusion is evidence WITH a shape
  cut: Google's OAuth refresh-token spelling `1//…` — the prefix is
  documented, but its verbatim token head would be the one family head
  ending in an Nd digit, and a phone number and a refresh token in one
  space-bridged run (`+14155552671 1//…`) compose a longer international
  phone match through the token's own head (the phone pass runs after
  keys and cannot tell a token head from run material), breaking the
  report's overlap contract. The row waits for a head that ends outside
  the digit class. The set is closed on evidence —
  the families five private consumers' leaked-credential shapes
  backed — and growing it is a new-evidence decision, never a
  drive-by; an unlisted provider's key shape passes through whole.
  The JWT family is MARKER-SCOPED (`Bearer eyJ…` only): a bare `eyJ…`
  triple is never touched, because one consumer's API legitimately
  carries eyJ-shaped non-secret cursors in its payloads, and redacting
  those would destroy the diagnostic this scrubber exists to preserve.
  Three discipline rules. (1) The prefixes are tried LONGEST-FIRST with
  fall-through: `sk-ant-` outranks bare `sk-` when its own grammar
  holds, and a too-short `sk-ant-` tail falls through to the `sk-`
  family, whose tail swallows the `ant-` spelling — still scrubbed,
  with the generic prefix. (2) A prefix glued to a preceding
  key-charset char is MID-TOKEN and never fires: `xak-…` survives
  whole, the same reasoning as the phone rule's clean-boundary cut (in
  real text a key glued to a word is that word's fragment), and it is
  what keeps a second key glued to a token's digest hex from firing,
  UNLESS that char ends a complete escape sequence (`%XX`, `\uXXXX`,
  `\UHHHHHHHH`, `\xHH`, `\NNN` octal, `\X`, or an ANSI CSI sequence
  `ESC [ params final`): logs carry keys inside JSON strings
  (`\n`), .NET spellings (`\u0027`), URL encodings (`%3D`), C byte
  reprs (`\x1f`), octal spellings (git's quoted-path `\346…`
  output), Python's `ascii()`/`backslashreplace` non-BMP spelling
  (`\U0001F600`), and ANSI-colored terminal output (`\x1b[31m…`),
  and the escape's tail letter/digit is formatting material,
  not the word a key head would be glued to. The sequence must be
  COMPLETE and DIRECTLY before the head: a doubled backslash escapes
  itself (the neighbor stays a literal, the head stays mid-token —
  `\xHH` and `\NNN` recount their backslash run; `\uXXXX`/`\UHHHHHHHH`
  do not,
  the one released over-trigger), a percent sign without two hex
  digits is prose, a one-digit `\x4` is prose, octal munch is
  maximal (`\1234` is escape `\123` + a literal `4`), a `[` without
  the ESC byte is prose (`[31msk-…` stays mid-token), and the
  control-char spelling `\cX` (shell/Perl) stays a documented
  non-match — the escape grammar is closed on evidence like the
  family set. A `-----BEGIN `
  head directly after a DASH RUN or a SHARED CLOSE is likewise a clean
  boundary — a preceding block's `-----END …-----` close is armor, not
  a word: either a dash run directly before the head (the close's own
  run, `END CERTIFICATE----------BEGIN …`), or the shared close, the
  head's five dashes doubling as the preceding close's five
  (`…CERTIFICATE-----BEGIN …`, a word directly before the head). The
  carve only opens the boundary; the PEM match downstream still
  requires both markers with the same words, so only a complete,
  self-validating private-key block redacts, its glue word verbatim.
  (3) The tail run is
  MAXIMAL, dots included nowhere: `sk-….x.co`
  scrubs the key and leaves `.x.co` (only the JWT grammar carries dots,
  inside its own marker-scoped shape; the `ya29.` dot is prefix, not
  tail — see the table). The digest is of the FULL match
  (prefix + tail).

**Key-family selection (`families=`).** The family set is closed:

```python
tors.KEY_FAMILIES: tuple[str, ...]
# ("openai", "anthropic", "google", "fireworks", "modal", "github",
#  "minted", "jwt", "aws", "xai", "gcp_oauth", "pem", "azure", "gitlab")
```

the canonical tuple, in the KeyFamily discriminant order, and the base for all-but-X
selections (`[f for f in tors.KEY_FAMILIES if f != "jwt"]`). `families=None` (the
default) is every key family this version knows; the set grows when new
families land — semver-visible — so callers needing stability list names
explicitly. A list selects exactly those families (order irrelevant,
duplicates deduped; tuples accepted); a bare string is a `TypeError`,
never an iterated character list. A valid selection is harmless —
ignored — when `api_keys` is not in the active rules, but validation is
still at the boundary: an unknown name is a `ValueError` naming the accepted set — `families must be one of ('openai', 'anthropic', 'google', 'fireworks', 'modal', 'github', 'minted', 'jwt', 'aws', 'xai', 'gcp_oauth', 'pem', 'azure', 'gitlab'), not "ssn"` — and `[]` is a `ValueError`, never a silent no-op — `families selects no key families; use None for all or list names`.

**Threat model: diagnostic-preserving, NOT adversarial-robust.** The
grammars above are parity-correct against the ported source at
`salt=""`, and parity is exactly why they are narrow: widening them
silently would break the byte-identical contract. The residual
bypassables below are DOCUMENTED, not fixed — attacker-controlled
formatting bypasses this scrubber, and that is the expected trade-off
for a scrubber that must never eat order numbers, byte counts, and
timestamps:

- `/`, `:`, `,`, `;` (and every other non-class char) split runs:
  `a/b.co`, `a:b.co`, `a,b.co` never match, exactly as the source
  leaves them.
- Fullwidth `＋` (U+FF0B) is not the literal `+`, and fullwidth/odd
  spaces are not the ASCII space class: `＋12345678` stays untouched.
- RFC local characters outside `[A-Za-z0-9._%+-]` fragment-leak: `!`,
  `#`, `$`, `&`, `'`, `*`, `/`, `=`, `?`, `^`, `` ` ``, `{`, `|`, `}`,
  `~` split the local part, so `a!b@x.co` scrubs `b@x.co` and the `a!`
  survives (pinned in the battery).
- RFC quoted-string locals leak WHOLE, not as fragments: `"user@name"@example.com`
  and `"a@b"@x.co` never match at all — `"` is outside the local class,
  and the quote immediately before the real `@` blocks the match, so the
  whole address survives. Canonicalize (strip RFC quotes / split
  display-names) before scrubbing if quoted locals are in threat; no
  grammar widening — widening would break the `salt=""` byte-identical
  contract.
- IP-literal and dotted-quad domains leak WHOLE: `user@[192.168.1.1]`
  never matches (`[` breaks the domain run) and `user@192.168.1.1` never
  matches (the trailing quad has no two-letter ASCII tail), so the whole
  address survives. Canonicalize (normalize IP literals / dotted quads to
  a scrubbed spelling) before scrubbing if these spellings are in threat;
  no grammar widening (parity).
- IDN / non-ASCII domains leak whole: the email classes are ASCII-only,
  so `a@exämple.com` never matches (the documented non-match). For
  internationalized domains, idna-to-punycode BEFORE scrubbing
  (`user@exämple.com` → `user@xn--exmple-cua.com`) so the ASCII grammar
  can see it — and treat the pre-image as sensitive until that step runs.
- Bare digit runs never match even at ten/eleven digits (`4155552671`,
  `14155552671` are order numbers), and short `+`-led runs never match
  (`+1234567`, seven digits, is the ported non-match).
- Extensions survive past the last digit: `415-555-2671 x1234` scrubs
  `415-555-2671` and leaves ` x1234` untouched, same as international
  (trailing separators survive; the extension is not part of the match).
- NPA/NXX are unvalidated: any ten-digit NANP-width run with a separator
  matches — area/exchange codes are never checked against the NANP plan.
- Email-then-phone composition over-redacts in the safe direction: the
  email local removal can trim a too-long digit run into exactly ten
  (or eleven-with-`1`) digits that then scrub (`1415 555 2671
  12345a@b.co` scrubs the trimmed `1415 555 2671` though the input run
  was sixteen digits). No digits leak; the id is partially tokenized.
  The parity lanes route these inputs to the extension lane.
- A natural (non-email) `~` + 12-lowercase-hex span is treated as a
  token breaker: the number after it scrubs even when glued
  hex-dirty. Over-redaction in the safe direction — and the reason a
  digest tail can never suppress the number after it.
- Punctuation inside a key splits the tail: `sk-abc…/…rest` never
  matches whole, and an inserted `/`, `:`, `;`, or `.` leaks the key's
  fragments — the same attacker-formatting trade-off as the contact
  grammars (the shared tail charset is deliberately narrow — the Azure
  row names its own wider alphabet — and widening it would eat
  identifiers that merely look key-shaped). Invisible characters
  inside a key or a PEM marker behave the same way: a zero-width
  space (U+200B), RTL override (U+202E), or combining mark inserted
  into the tail run or the `-----BEGIN ` literal breaks the grammar
  and leaks. Directly BEFORE a key head the same characters are the
  safe direction — they are not key-charset bytes, so the head fires
  clean (pinned). Canonicalize (strip Cf, map combining marks away)
  before scrubbing if invisible-character insertion is in threat.
- An unlisted provider's key shape leaks WHOLE: the family table is the
  evidence-backed closed set above — Slack `xox…` and Stripe are
  excluded on zero evidence (AWS `AKIA`/`ASIA` moved to the table on
  evidence), the prefix-less shapes pass through as documented above —
  and a new family is a new-evidence decision with its own pins, never
  a silent widening.
- Contact-glue under-redaction: a key glued with ZERO separator to
  preceding contact material is mid-token under the boundary rule and
  survives whole, and no later pass recovers it — a phone number's last
  digit (`+14155552671sk-…`), an email domain's last letter
  (`a@b.cosk-…`), a prior token's digest hex (`+14~abc123def456sk-…`):
  field concatenation without separators, the machine-plausible form.
  The composed call still scrubs the contact half; the key half leaks.
  Separate fields before scrubbing if concatenations are in threat.
- Preserved-span cloaking under family selection: a span an unselected
  family spends whole stays whole — including any selected family's key
  inside it (a key in a deselected PEM block's body, a key shape inside
  a deselected JWT's segments). Deselecting families reintroduces leak
  risk for material inside the preserved spans; `families=None` is the
  only full-coverage selection, and `report["skipped"]` names exactly
  what each narrowed call preserved.
- >3-segment JWT/JWE survival: the JWT grammar is exactly three
  segments, so a 5-part JWE behind `Bearer ` scrubs through the third
  segment and keeps parts 4-5 verbatim. Split or reject multi-dot
  `Bearer` material before scrubbing if JWEs are in threat.
- A bare `eyJ…` triple (no `Bearer ` marker) is never touched, even
  three well-shaped base64url segments: one consumer's API legitimately
  carries eyJ-shaped non-secret cursors, so the JWT family fires only
  behind the marker — the deliberate under-redaction side of that
  scoping.
- The kept family prefix is a coarse provider label (`sk-` vs
  `sk-ant-`), not a credential: it exists so the token tells the
  operator which credential to rotate. It narrows the search space for
  whoever already holds the log — the same known-salt reasoning as
  below, priced in.

If your threat includes adversarial formatting (an attacker choosing
the spelling to dodge the scrubber), canonicalize BEFORE scrubbing —
and treat the scrub as one layer, not the whole control. `tors.nfkc`
alone is insufficient: NFKC folds fullwidth alphanumerics and some
compatibility spaces, but it does not fold every separator the phone
grammar splits on. Map explicitly first:

```python
import re, unicodedata

_SEP = re.compile(r"[\t\n\r\f\v\u0020\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")

def canonicalize_for_scrub(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Zs/Zl/Zp + \t\n\r\f\v -> U+0020: every Unicode space separator
    # (Zs, Zl, Zp) plus ASCII whitespace the grammar does not class.
    text = _SEP.sub(" ", text)
    # Then canonicalize separators/domains to the grammar above:
    # `/ : , ;` -> " " (or ""), fullwidth ＋ (U+FF0B) -> "+",
    # IDN -> idna punycode, RFC quotes -> stripped.
    return text
```

Then scrub the canonicalized text.

`rules=None` applies every rule in the canonical order — the keys
substitution over the whole string first, then the email substitution over
its result, then the phone substitution over that, each exactly once —
because the credential must be eaten whole before the contact passes scan
(a key's tail can spell a dash-separated domestic phone run, and a whole
key can spell an email local part: `sk-…@x.co` would otherwise be one
email match), and an email's local part may itself
carry a `+`-led digit run (`user+14155552671@example.com` is one email, and
the phone rule must see its token, never the address). One reachable
composition is documented rather than fixed: the keys pass eats a
key-shaped local part and leaves `<family>~<digest>@domain`, whose digest
hex is itself local-part material, so the email pass tokens
`hex@domain` — over-redaction in the safe direction, converging on the
second scrub. `[]` is the identity
(the original object); duplicates dedupe and listing order is irrelevant;
each name restricts the scrub to that rule.

**The report twin: `tors.scrub_pii_report`.** The same scrub under the
same single GIL-released pass, plus the accounting:

```python
def scrub_pii_report(
    text: str,
    rules: Sequence[Literal["contact_email", "contact_phone", "api_keys"]] | None = None,
    *,
    salt: str | None = None,
    families: Sequence[str] | None = None,
) -> dict: ...
```

`report["text"] == scrub_pii(text, rules, salt=salt, families=families)`
for the same arguments, byte-exact — the shape below is the accounting
for that string:

```python
{"text": <scrubbed str>, "redacted": {"contact_email": 1, "api_keys": 3, "openai": 2, "jwt": 1}, "skipped": {"jwt": 1}, "spans": [{"type": "api_keys:openai", "start": 12, "end": 64}]}
```

`redacted` counts what scrubbed: per-rule totals (`contact_email`,
`contact_phone`, `api_keys`) plus per-family totals under their lowercase
names (`openai`, `jwt`, …) — absent types omitted. `skipped` counts what
detection saw but redaction preserved: families NOT in the active
selection that would have matched anyway (detection runs, redaction does
not) — the "preserved a JWT, log it separately" signal: select
all-but-jwt, scrub, then route `report["skipped"]` to the separate
credential-rotation queue. Always `{}` when `families=None`; `{}` when
`api_keys` is inactive. `spans` mark every redaction, ordered by start —
codepoint indices into the INPUT, each typed `<rule>`
(`contact_email`, `contact_phone`) or `api_keys:<family>`
(`api_keys:openai`, `api_keys:jwt`, …). `text[start:end]` is the match
it replaced, with two documented exceptions where passes compose: a
match that began inside a prior token's digest is recorded from that
token's input end (only the input-side suffix is addressable), and a
match that ran into a prior token's verbatim head overlaps the
producing span (the input bytes fed two tokens; both spans are
recorded, keys before email before phone at shared starts; a span END
at the head boundary maps affinely to the head end, so the span text
is exactly what the match ate). Empty input
is the empty report: `{"text": "", "redacted": {}, "skipped": {},
"spans": []}`. Salt, idempotence, and the surrogate boundary behave
exactly as documented above.

**The salt trade-off, stated plainly.** `salt=None` resolves PER RULE:
the contact rules use tors's documented default `"tors/scrub_pii/v1"`
and the keys rule its own `"tors/scrub_keys/v1"` — a fixed, non-secret,
versioned domain-separation tag each (frozen: changing either would
silently change every deployment's token values), and deliberately
different so a key digest can never alias a contact digest at the
default settings. An explicit salt string salts every rule alike (the
`""` spelling is unsalted for every rule — the migration lane:
byte-identical tokens with an unsalted upstream). A KNOWN salt, the public defaults
included, does
not make the digest secret: the E.164 space is small enough to enumerate, so
an attacker with a candidate list can still confirm whether a specific
number appeared — and the same holds for EMAIL addresses (common
local-parts on common domains are enumerable too), for API keys
(provider key alphabets are smaller than their length suggests: the
secret tail of an `sk-` key is ~62^20, far beyond enumeration, but the
PREFIX half is kept in the clear by design — see above), and for `salt=""`
(the unsalted migration lane re-publishes the source chain's documented
weakness by design: byte-identical tokens with an unsalted upstream).
The digest is `sha256(salt + match)` with plain concatenation, so
`salt`/`match` boundaries can alias (`salt="a"` + `match="bc"` digests
like `salt="ab"` + `match="c"`) — a second reason the salt is
domain-separation, not a key. The digest is 12 hex chars (48 bits):
at ~119k distinct matches the birthday merge rate is ~2.5% — two
different contacts sharing one token — frozen-for-stability (widening
the digest would break the `salt=""` byte-identical contract).
The tokens are redaction, not pseudonymization crypto;
deployments that care pass their own secret salt, and a consumer migrating
from an unsalted upstream scrubber passes `salt=""` to keep its token values
byte-identical — any other salt changes them.

**Idempotence, as it true is.** Tokens are individually fixed points, and the
output is a fixed point unless an email token is immediately followed by
`@`-shaped text — reachable when two email matches were adjacent
(`a@b.co9@x.yz`) or when a match is followed by an unmatched `@`-run whose
own local part the match consumed (`x@b.co@w.vu`). A token's digest hex is
local-part material, so a second pass fires once more on that boundary and
then holds: **scrubbing twice always converges**. If your excerpts can chain
emails like that and you need a guaranteed fixed point, scrub twice
(`scrub_pii(scrub_pii(x))` — the documented helper shape; the fuzz target
asserts convergence structurally — no input match survives verbatim into
the converged output and the third pass is the identity — not token-exact
values, which the parity gates pin; the hypothesis ports mirror that
boundary); everything else is one pass. Phone-only is strictly idempotent,
and keys-only is too, by construction: a key token's family prefix ends
`-`, `_`, `=`, or `.`, or is a bare head (`AIza`, `Bearer`, `PEM`, the
4-char `AKIA`/`ASIA` head), the byte after it is `~` — never tail
charset — and no family prefix can be spelled inside 12 lowercase digest
hex, so the second scrub never re-fires (a key token's `~` + 12 hex is a
breaker for the phone pass as well, so a number after it keeps its own
clean run).

`tors.scrub_pii(s, ...) is s` exactly when no active rule matches. A `str`
holding lone surrogates is refused at the argument boundary with
`UnicodeEncodeError` ("surrogates not allowed"), the same boundary every
str-in function here documents — the ported chain diverges there by design:
its classes never match a surrogate, so it returns such text with the
matches AROUND the surrogate still scrubbed (scrub-plus-surround), while
tors refuses the whole call at the boundary. Pin the call-site shape
accordingly (sanitize-or-skip surrogates before scrubbing if your pipeline
ingests lone-surrogate text), and see the parity gates which pin the
divergence in both directions. Live re-sync cadence/owner: the quoted pin
in `tests/reference.py` is the CI oracle; the live source module is
re-checked manually whenever its grammar changes and at least once per
Unicode/dependency bump (owner: the scrub_pii maintainer).

**Async**: `await tors.aio.scrub_pii(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)). The thread-hop costs tens of microseconds: noise next to a millisecond pass, real overhead next to a microsecond-scale excerpt — prefer the sync spelling below ~100KB of error-excerpt text, `tors.aio` above it. `await tors.aio.scrub_pii_report(...)` is the report twin under the same hop.

## `tors.scrub_log_text`

```python
def scrub_log_text(
    text: str,
    rules: Sequence[
        Literal["pg_detail_lines", "uri_userinfo", "uri_query_creds", "libpq_conninfo_creds"]
    ]
    | None = None,
) -> str: ...
```

Named-rule log and exception-text scrubbing, four linear scans + splice
under one `py.detach`: the scrub a worker applies to
`str(exc)`/`repr(exc)`/rendered tracebacks before any of it reaches a log
line, a span, or an exported attribute, byte-identical to the four
compiled regexes that define its grammar — pinned by a differential
harness that races tors against that reference (see
[Design and scope](design.md) for why this is a *named-rule* surface
rather than a pattern parameter).

Four rules, one closed set:

- `pg_detail_lines` — PostgreSQL `DETAIL:` lines quote caller-supplied row
  values, so the whole line is dropped. Both separator spellings: real
  newlines (line content deleted, the newline kept — a blank line is left
  behind; a CRLF line's `\r` is consumed with the content; the
  `traceback.format_exception` gutter run of an `ExceptionGroup`/`except*`
  sub-exception — repeated `| `/`+ ` markers, one layer per nesting level —
  absorbed before the anchor), and the `repr()`-flattened `\nDETAIL:` runs
  a traceback's final line carries (consumed up to the next escaped
  separator or the repr tail — a quote followed by the run of `)`/`]`
  closers `repr()` ends with, `')` plain and `')])` inside an
  ExceptionGroup's list — which is preserved:
  `PostgresError('msg\nDETAIL: … exists.')` comes back as
  `PostgresError('msg')`).

  > [!WARNING]
  > SECURITY POLICY, changed in this release (issue #107), inverting the
  > 0.7.0 behavior: the repr-flattened run's lookahead is FAIL-CLOSED. A
  > run whose tail matches neither safe delimiter — an unterminated repr
  > (no closing quote), or one with more text behind the quote — scrubs
  > THROUGH END OF LINE. 0.7.0 left such runs alone (a pinned non-match);
  > the consumer chain's stated policy is that a delimiter miss must
  > delete MORE text, never less of the secret, and tors follows it. The
  > deletion can now also eat text a userinfo mask would have needed
  > (`scrub("a://u:p\\nDETAIL:x@h")` is `"a://u:p"` — the DETAIL deletion
  > takes the whole tail); run `uri_userinfo` separately when credential
  > removal must outrank DETAIL parity.
- `uri_userinfo` — `scheme://user:password@host` becomes
  `scheme://user:***@host`: scheme and username preserved verbatim, empty
  username handled, password ending at the first `@`.
- `uri_query_creds` / `libpq_conninfo_creds` — the password-family
  connection parameters, ONE pass under two names (the two anchor grammars
  of the live chain's single combined regex): `[?&]name=value` (URI query)
  and the libpq keyword form `name=value` (a non-`[A-Za-z0-9_]` char — or
  text start — before the name, so `host=h password=p` masks and `cpwd=`
  does not). Names are the five credential parameters
  (`password`, `passphrase`, `passwd`, `pwd`, `sslpassword`) matched
  case-insensitively; the value is a libpq single-quoted string (spaces
  allowed, `\'`/`\\` escapes honored) or an unquoted token running to
  whitespace or `&` — deliberately NOT stopping at `@`: a password may
  legally carry an unencoded `@`, and a mask that stops there leaves the
  tail riding after the `***` (0.7.0 did exactly that). Name and delimiter
  are preserved: `?password=a@b` → `?password=***`.

`rules=None` (the default) runs the full chain in canonical order:
`pg_detail_lines` → `uri_userinfo` → the conninfo credential pass, each
rule a whole pass over the current text before the next begins (a DETAIL
deletion can eat the `@` a userinfo mask anchors on — rule interaction is
why the order is a contract, not a caller choice). Selecting both conninfo
names runs the combined pass once, never two sequential substitutions.

> [!WARNING]
> The default chain can leave a credential fragment by design:
> `scrub("pg://u:p\\nDETAIL:x@h')")` is `"pg://u:p')"` (the DETAIL deletion
> eats the `@`, the userinfo mask then has nothing to anchor on, the
> password `p` survives). The order is the documented contract (the
> scrub a worker applies to `str(exc)`/`repr(exc)`/rendered tracebacks
> before any of it reaches a log line, a span, or an exported attribute —
> up to four passes per text, ~24 per failed job across its
> message/traceback/span texts); do NOT reorder to "fix" the fragment.
> Safe pattern when credential removal outranks DETAIL handling: run
> `uri_userinfo` separately (e.g.
> `scrub_log_text(text, ["uri_userinfo"])`, which gives
> `"pg://u:***@h')"` here) — trading the DETAIL deletion for the mask,
> deliberately and visibly at the call site.
>
> A second documented seam: a digit-leading scheme defeats the
> userinfo anchor everywhere in the family — `scrub_log_text`,
> `scrub_log_text(text, ["uri_userinfo"])`, and `scrub_pii` all leave
> `1postgres://user:pass@host` whole (the shared scheme anchor is
> `\b[a-zA-Z]`, so `1postgres` is not a scheme to it; verified against
> all three spellings). This is the grammar's pinned shape, not a defect;
> there is currently no tors surface that redacts a digit-prefixed scheme
> URI — a caller seeing that shape needs its own pre-pass.

`rules=[]` is the identity; duplicates
dedupe and caller order is irrelevant; an unknown name raises `ValueError`
naming the accepted set. A pass never rescans its own output.

`tors.scrub_log_text(s, rules) is s` exactly when no rule fires — including
the `***` fixed points, where a rule fires and splices to an equal value:
those return a fresh, equal string. Scrubbing the scrubbed output is a
fixed point by the second pass, with one documented cross-rule exception
to strict idempotence: a password-param replacement deletes a `/` that was
capping a userinfo user run, so the second pass can find one more
redaction (`x://u?pwd=a/b&:pw@h` scrubs to `x://u?pwd=***&:pw@h`, and a
second scrub claims the password too) — the ported chain behaves
identically, and a third scrub is always the fixed point.

```python
tors.scrub_log_text(
    "JobError: duplicate key\n"
    "DETAIL:  Key (idempotency_key)=(customer-4417-a3f2) already exists.\n"
    "HINT: unchanged"
)
# "JobError: duplicate key\n\nHINT: unchanged"  (blank line left behind)
tors.scrub_log_text(
    "connect dsn=postgresql://worker:S3cr3t-x9@db.internal:5432/prod?password=fallback"
)
# "connect dsn=postgresql://worker:***@db.internal:5432/prod?password=***"
tors.scrub_log_text("JobError('duplicate key\\nDETAIL:  Key (idempotency_key)=(customer-4417) exists.')")
# "JobError('duplicate key')"  (closing quote preserved)
tors.scrub_log_text(
    "connect dsn=postgresql://worker:S3cr3t-x9@db.internal:5432/prod",
    ["uri_userinfo"],
)
# "connect dsn=postgresql://worker:***@db.internal:5432/prod"  (one rule alone)
```

**Async**: `await tors.aio.scrub_log_text(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.nfc` / `tors.nfd` / `tors.nfkc` / `tors.nfkd`

```python
def nfc(text: str) -> str: ...
def nfd(text: str) -> str: ...
def nfkc(text: str) -> str: ...
def nfkd(text: str) -> str: ...
```

`unicodedata.normalize(form, text)` for each form, in one GIL-released pass each: the
standalone normalization forms without normalize's folding/collapsing/strip
stages. Compatibility decompositions fire under the K-forms and only under them
(`"\ufb01"` stays `"\ufb01"` for nfc/nfd, becomes `"fi"` for nfkc/nfkd). Same
identity-return contract as `normalize` (`tors.nfc(s) is s` whenever `tors.nfc(s) ==
s`). tors ships its own Unicode tables (Unicode 16.0.0, the same UCD CPython 3.14
ships), so the same input produces the same output on every supported Python: on older
interpreters the only divergences from that interpreter's own `unicodedata` are on
codepoints it leaves unassigned; no assigned codepoint diverges in any form (pinned
exhaustively per interpreter by `tests/test_parity.py`).

## `tors.html_unescape`

```python
def html_unescape(text: str) -> str: ...
```

`html.unescape(text)` over the full HTML5 named-entity table (all 2231 entries of
`html.entities.html5`, with- and without-semicolon spellings) plus CPython's exact
numeric-reference classification (the Windows-1252 remap, the 126
invalid-codepoints-to-empty quirk, out-of-range to U+FFFD, the greedy-decimal
`&#10FFFF;` → `"\nFFFF;"` behavior, the longest-matching-prefix fallback with verbatim
remainder). On text the unescape leaves unchanged it returns the input object itself
(CPython's own no-`&` fast-path idiom).

**Raises `ValueError` on CPython 3.11+** when a decimal numeric reference's digit run
exceeds `sys.get_int_max_str_digits()` (default 4300): the stdlib's own integer string
conversion limit, with its exact message (`"Exceeds the limit (4300 digits) for integer
string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase
the limit"`). Leading zeros count toward the run; the raise precedes every
classification; the limit is read from the running interpreter once per call, so
`sys.set_int_max_str_digits()` changes are honored on the next call. Hex references are
exempt (base 16 is a power of two: the limit applies only to non-power-of-two bases).
CPython 3.10 has no limit at all, and tors matches it there: nothing raises.

```python
tors.html_unescape("Tom &amp; Jerry &#233;")
# "Tom & Jerry é"
```

## `tors.grapheme_count`

```python
def grapheme_count(text: str) -> int: ...
```

The number of UAX #29 extended grapheme clusters (ZWJ emoji sequences are one cluster,
combining-mark chains join their base, CRLF is one cluster, regional-indicator pairs
join). The stdlib has no segmenter at all: the gap this exists to fill. The return is
a single `int`: no list marshalling class at all, the cleanest GIL cell in the suite.
Pinned by a hand-derived UAX #29 table plus structural properties
(`tests/test_segmentation.py`).

```python
tors.grapheme_count("a\r\n\U0001f469\u200d\U0001f52c")  # 3
```

## `tors.word_bounds`

```python
def word_bounds(text: str) -> list[tuple[int, int]]: ...
```

The UAX #29 word-boundary segments as `(start, end)` pairs in Python `str` index
(codepoint) units: `text[start:end]` is the segment; bounds cover `[0, len(text))` and
joining the slices reproduces the input. Offsets, never string lists (marshalling
thousands of small `PyString`s under the GIL would eat the win). One measured caveat:
the return marshalling constructs one 2-tuple of ints per segment under the GIL,
O(number-of-segments), a measured 328–344 ms hold at 12 MiB of prose
(3.67M segments; worst-gap band, box-pace-dependent: the dev box the test ledger
records measured 428–497 ms, and the load-stable constant is the ratio, ~0.72 of the
call's wall). Fine at document scale; for whole-file sizes use the iterator
below.

```python
tors.word_bounds("Hello, world!")
# [(0, 5), (5, 6), (6, 7), (7, 12), (12, 13)]
```

**Async**: `await tors.aio.word_bounds(...)` runs this under `asyncio.to_thread`
(see [Async use](async.md)) — the marshalling hold above is exactly what the hop
removes from the event loop.

## `tors.word_bounds_iter`

```python
def word_bounds_iter(text: str) -> Iterator[tuple[int, int]]: ...
```

The streaming spelling of `word_bounds`: the same `(start, end)` sequence (pinned to
sequence-parity with the list API over every tricky row and hypothesis text), yielded
lazily. The segmentation runs under one GIL-released pass when the iterator is
constructed, and each `__next__` holds the GIL only to construct one tuple (µs-scale):
worst heartbeat gap 15.4 ms at 12 MiB, against the list shape's structurally
unattainable 328–344 ms band; and the full drain is also ~2.1x faster in wall time
than the list API (measured at 12 MiB). `__length_hint__` reports the
remaining bound count and tracks partial consumption. The list API stays the right
shape for small inputs and one-shot batch work.

## `tors.word_count`

```python
def word_count(text: str) -> int: ...
```

The `grapheme_count` precedent for the word segmenter: a single `int` return (no
marshalling class at all), the whole scan GIL-released, `O(1)` memory where
`len(word_bounds(text))` materializes the full tuple list. `word_count(t) ==
len(word_bounds(t))`, pinned: the answer for a caller that only needs the count, not
the bounds themselves.

```python
tors.word_count("Hello, world!")
# 5
```

## `tors.decode_utf8`

```python
def decode_utf8(raw: bytes, *, errors: Literal["strict", "replace"] = "strict") -> str: ...
```

`raw.decode("utf-8", errors=...)` byte-exact over arbitrary bytes, including every
ill-formed class. Strict raises the very `UnicodeDecodeError` CPython's own decoder
raises (same type, `start`/`end`/`reason`, `object`, and message), and replace emits
CPython's exact U+FFFD placements (maximal-subpart substitution). An `errors` value
outside `{"strict", "replace"}` raises `ValueError` (the `unicodedata.normalize`
closed-set convention). The argument must be exactly `bytes` (`bytearray`/`memoryview`
raise `TypeError`): the pass reads a zero-copy borrow of the immutable buffer with the
GIL released, and a writable buffer would be a data race, not a semantic difference.

**Async**: `await tors.aio.decode_utf8(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.finalize_utf8`

```python
def finalize_utf8(
    raw: bytes, *, errors: Literal["strict", "replace"] = "strict"
) -> tuple[str, str]: ...
```

Decode + normalize + hash in one GIL-released call:
`(normalize(raw.decode("utf-8", errors=...)), sha256-of-the-result)`: the shape an
extraction pipeline wants for its text reads. Strict (default) raises the stdlib's own
`UnicodeDecodeError` on invalid bytes; `errors="replace"` flows the U+FFFD
substitutions through the pipeline. Same bytes-in GIL model as `decode_utf8`: no
argument-materialization class at all.

**Async**: `await tors.aio.finalize_utf8(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.b64_encode_bytes`

```python
def b64_encode_bytes(raw: bytes) -> str: ...
```

`base64.b64encode(raw).decode("ascii")` (RFC 4648 standard alphabet, padded), for
content-addressing paths that today hold the GIL for the whole encode. The argument
contract matches the bytes-in surface (exactly `bytes`).

**Async**: `await tors.aio.b64_encode_bytes(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.b64_decode`

```python
def b64_decode(s: str, *, validate: bool = True) -> bytes: ...
```

`base64.b64decode(s, validate=...)` over ASCII strings: same decoded bytes, same
raised exception (the real `binascii.Error`, so `except binascii.Error` and
`except ValueError` both keep working), same messages, including all five strict-mode
messages and the lenient mode's discard-non-alphabet rules (`'Zm9v=Zg=='` → `b'foof'`).
The core is a line-for-line port of the post-gh-145264 `binascii_a2b_base64_impl`
(CPython's 3.13/3.14 maintenance branches): in lenient mode, excess padding is ignored
and data after a completed pad sequence is decoded, not silently dropped
(`'Zg==Zg=='` → `b'f\x06`'`). CPython before that fix truncated the trailing data
there, a parser differential it fixed as a security issue, and tors ships the fixed
machine on every interpreter it supports rather than matching each stdlib it happens
to run under (pre-fix stdlibs, and 3.10's distinct regex validator, are documented
divergences). `validate=True` is the default (decode-side callers want invalid input
to fail loudly); a non-ASCII `str` (including one holding lone surrogates) raises the
stdlib's own plain `ValueError`; non-`str` input raises `TypeError`.

**Async**: `await tors.aio.b64_decode(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.utf8_is_valid`

```python
def utf8_is_valid(raw: bytes) -> bool: ...
```

Answers "is this well-formed UTF-8?" without materializing a `str` or paying exception
flow: SIMD validation, GIL-released, `bool` out. `True` exactly when
`raw.decode("utf-8")` would succeed. The stdlib has no boolean primitive for the
question (its only spelling is decode-and-catch); the argument contract matches the
bytes-in surface.

## `tors.decode_utf16`

```python
def decode_utf16(
    raw: bytes,
    *,
    errors: Literal["strict", "replace"] = "strict",
    byteorder: Literal["native", "little", "big"] = "native",
) -> str: ...
```

`raw.decode("utf-16", errors=...)`, or the `"utf-16-le"`/`"utf-16-be"` spellings via
`byteorder=`: byte-exact over arbitrary bytes, one GIL-released pass. `byteorder="native"`
(the default) sniffs a leading BOM (`FF FE` little, `FE FF` big) and strips it from the
output, falling back to the host's own endianness when no BOM is present, matching the
plain `"utf-16"` codec name exactly. `byteorder="little"`/`"big"` never sniff or strip a
BOM: a leading BOM-like byte pair decodes as the literal U+FEFF character, matching
`"utf-16-le"`/`"utf-16-be"`.

Strict mode raises CPython's own `UnicodeDecodeError`, including `.encoding` (always the
resolved label, `"utf-16-le"` or `"utf-16-be"`, never the bare `"utf-16"` name, even under
`byteorder="native"`). UTF-16 has four distinct reasons, all verified against a running
interpreter:

- `"truncated data"`: a lone trailing byte, one short of a full code unit, with no
  pending high surrogate. Span: that one byte.
- `"unexpected end of data"`: a high surrogate with fewer than two bytes following it.
  Span: the surrogate through the end of the input (the partial tail has nothing useful
  left to do with it, so it folds into the one error).
- `"illegal UTF-16 surrogate"`: a high surrogate followed by a full code unit that is
  not a valid low surrogate. Span: the high surrogate's two bytes alone; the following
  unit is reprocessed from scratch.
- `"illegal encoding"`: a low surrogate reached other than as the second half of a
  valid pair. Span: its own two bytes alone.

`errors="replace"` emits exactly one U+FFFD per error unit and continues decoding:
CPython's own granularity, not one U+FFFD per byte.

No `encode_utf16` or `finalize_utf16`: encoding a trusted `str` carries none of the
untrusted-bytes-parsing risk that motivates the decode side, and `str.encode("utf-16-le")`
already covers it; a fused decode+normalize+hash twin has no demonstrated caller need,
unlike `finalize_utf8`'s.

```python
tors.decode_utf16(b"\xff\xfeh\x00i\x00")  # "hi"
tors.decode_utf16(b"h\x00i", errors="replace")  # "h�"
```

**Async**: `await tors.aio.decode_utf16(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.utf16_is_valid`

```python
def utf16_is_valid(
    raw: bytes, *, byteorder: Literal["native", "little", "big"] = "native"
) -> bool: ...
```

`tors.utf8_is_valid`'s shape for the UTF-16 side: `True` exactly when
`decode_utf16(raw, byteorder=byteorder)` (strict) would succeed, with no `str`
materialized on either path.

```python
tors.utf16_is_valid(b"h\x00i\x00")  # True
tors.utf16_is_valid(b"h\x00i")  # False
```

## `tors.json_is_valid`

```python
def json_is_valid(data: bytes | str) -> bool: ...
```

The RFC 8259 validity gate: `True` exactly when `orjson.loads(data)` would
succeed — one linear scan over the raw bytes, no object tree, GIL-released.
Built for the validate-and-discard gate, bytes that are parsed once and
thrown away: on a 64 KiB list-of-small-dicts document, ~95% of a full
`orjson.loads` is constructing objects nobody reads, and the scan alone is
a 4-5x cheaper pass at these sizes (64 KiB ~40 µs, 1 MiB ~0.7 ms; issue
#61's measured prototype table). The scanner is hand-rolled and iterative —
no recursion, no heap, an O(1) fixed stack — so pathological inputs cost
the same linear pass.

**The acceptance set is orjson 3.x's, not the stdlib's** (`json.loads`'s):
where the two disagree, orjson's reading wins, because the gate stands in
front of a consumer whose next step IS `orjson.loads`. The documented
seams, each pinned in tests/test_json_is_valid.py:

- **Float-overflow literals reject**: `1e400`, `-1e400`, `1e309`, `2e308`,
  `1.7976931348623159e308` raise `JSONDecodeError` in orjson ("number is
  infinity when parsed as double") where the stdlib hands back `inf`. The
  scanner computes the literal's f64 value and rejects a non-finite result;
  underflow (`1e-400` → `0.0`) is finite and accepts. Long-integer literals
  (20+ digits) ride the same fallback — orjson parses them as doubles — so
  309 `9`s (9.99e308) reject where 308 accept; ≤ 19 digits always accept.
  *Caveat*: the decision trusts correctly-rounded parsing (Rust's `f64`
  parser), so a literal within one rounding step of ±1.8e308 could in
  principle disagree with orjson's own float parser; a 40,000-case
  knife-edge sweep plus the 2,666-input differential corpus found zero such
  disagreements.
- **NaN / Infinity / -Infinity reject** (no such grammar in RFC 8259; the
  stdlib accepts them as floats).
- **A leading UTF-8 BOM rejects** (the stdlib strips it).
- **Lone `\ud800`-class surrogate escapes reject** — a high surrogate
  escape must be immediately followed by `\u` + a low surrogate — and so
  does a UTF-8-*encoded* surrogate (`"\xed\xa0\x80"`); the stdlib builds
  lone surrogates from both. This is the class that makes a looser
  validator the unsafe direction for a gate (serde's `IgnoredAny`, #61's
  rejected alternative, accepted both).
- **Depth cap 1024** (orjson's): the 1025th open container rejects, objects
  and arrays counting against one shared cap. It is an answer (`False`),
  not an exception — a validity gate is a boolean question.
- Invalid UTF-8 anywhere rejects; raw control characters in strings reject
  (`\u0000` the escape accepts); trailing garbage, trailing commas, leading
  zeros, and unterminated strings reject; duplicate keys accept (orjson
  last-wins).
- **A `str` holding a raw lone surrogate answers `False`, not a raise**:
  such a str has no UTF-8 view to scan, and every consumer's parser
  (orjson's `loads` included) raises on one, so the gate's
  "would-loads-raise" answer is `False` — the raise-free contract holds
  on this lane too (the stdlib would accept some of these; orjson's
  reading wins, as everywhere in this surface). The escape TEXT
  (`"\ud800"` spelled out) is ordinary content and rejects through the
  normal grammar.

Booleans only: no invalid input raises — nothing in the scanner has an
error path. A wrong-TYPE argument (not `bytes`, not `str`) raises
`TypeError` like the bytes-in surface; `bytearray`/`memoryview` are
refused with it (a writable buffer mutated by another thread mid-scan
under the released GIL is a data race, not a semantic difference).

**GIL behavior**: `utf8_is_valid`'s class. A `bytes` argument is a zero-copy
immutable borrow, the whole scan runs under one `py.detach`, and the `bool`
return has no marshalling class at all — the borrow alone is the call's
GIL-held residue, valid and invalid input alike. A `str` argument pays the
standard str-in borrow first, under the GIL: a zero-copy alias when the
string is pure ASCII or its UTF-8 view is already cached (repeat calls on
the same object: O(1) borrow, then the detached scan), a one-time O(input)
materialization+cache-fill on the first non-ASCII call (encode-parity;
CPython caches the view on the object, and `encode` reads it but never
fills it). No aio twin, matching `utf8_is_valid`/`utf16_is_valid`: a
sub-millisecond scan needs no thread hop (see [Async use](async.md) for the
size guidance).

```python
tors.json_is_valid(b'{"a": [1, 2.5, true, null]}')  # True
tors.json_is_valid(b"1e400")  # False — orjson raises; the stdlib says inf
tors.json_is_valid(b'"\\ud800"')  # False — lone surrogate escape
tors.json_is_valid(b"\xef\xbb\xbf{}")  # False — BOM
tors.json_is_valid('{"a": 1}\ud800')  # False — raw surrogate in the str, no raise
```

A `str` argument holding a lone surrogate (a raw one — not the escape text)
never reaches the scan at all: the str-in borrow cannot materialize its
UTF-8 view and raises `UnicodeEncodeError`, the standard str-argument
contract every tors str-in function shares.

## `tors.detect_encoding`

```python
def detect_encoding(raw: bytes, *, tld: str | None = None) -> str: ...
```

Best-guess character encoding for non-UTF8 legacy/OCR byte content, via `chardetng`
(the detector Firefox ships), GIL-released. Unlike `utf8_is_valid`/`decode_utf8`
there is no ground truth here: this is a heuristic guesser, not a validator, and it
always returns some codec name rather than raising for well-formedness reasons. The
intended pipeline: call `utf8_is_valid` first, and only reach for `detect_encoding`
on the bytes that already failed that check, then decode with the returned name.

The returned name is a Python `bytes.decode()`-usable codec name: mostly the WHATWG
Encoding Standard label `chardetng` reports (`"windows-1252"`, `"Shift_JIS"`,
`"UTF-8"`), translated to Python's own spelling for the two labels in this
detector's candidate set that Python's codec registry doesn't recognize under
their WHATWG names (`"windows-874"` → `"cp874"`; the logical-ordered Hebrew label
`"ISO-8859-8-I"` → `"ISO-8859-8"`, since the byte mapping is identical and a
decode doesn't consult the bidi-ordering hint the `-I` suffix carries). `tld` is
an optional top-level-domain hint that
disambiguates language-family-ambiguous input: accepted in whatever natural
spelling a caller has (`".jp"`, `"JP"`, or `"jp"`; a full domain or non-ASCII input
degrades to "no hint" rather than raising, since `chardetng`'s own `tld` parameter
panics on anything but a bare lowercase ASCII label with no period: a caller
shouldn't crash over a hint spelled the normal way).

```python
tors.detect_encoding("Café".encode("windows-1252"))
# "windows-1252"
```

## `tors.diff_opcodes`

```python
def diff_opcodes(
    a: str, b: str, *, deadline_ms: float | None = None
) -> list[tuple[str, int, int, int, int]]: ...
```

**Async**: `await tors.aio.diff_opcodes(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

`difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()`'s shape at native speed:
`(tag, i1, i2, j1, j2)` tuples with `tag` in `{"equal", "replace", "delete",
"insert"}`, ranges monotone/contiguous/covering both sides, adjacent delete+insert
merged into `replace` exactly as difflib presents it, indices in Python `str`
(codepoint) units. Character-level, like difflib on `str` operands: that is what
makes difflib the parity oracle; `tors.diff_opcodes_lines` (below) is the
line-level spelling. Exact agreement with
`difflib.SequenceMatcher(None, a, b, autojunk=False)` is pinned on the classes whose
canonical opcode list is forced: verified by difflib's own answer carrying the
canonical single-op shape (pure insert/delete, single-run replace, all-equal,
empty operands); structural validity (the opcodes reconstruct both sides) is
pinned by hypothesis over arbitrary pairs; and the boundary cases where the two
algorithms pick different valid alignments (repeated-flank contexts, where
difflib's longest-match recursion emits a non-minimal insert+delete split and
Myers + run-maximization emits the minimal contiguous change) are documented
and tested, never silent. The four tag strings are interned once per call, so
`op[0] is "equal"` holds exactly as it does for difflib's own tuples.

A third divergence class needs no repeated flanks at all: difflib's default
`autojunk=True` junk-handles every element appearing more than `len(b)//100 + 1`
times once `len(b) >= 200`, dropping it from matching, so past 200 elements the
default-constructor spelling is not the oracle — `SequenceMatcher(None, a, b,
autojunk=False)` is, and tors deliberately keeps the un-heuristic'd answer
(`tests/test_similarity.py` pins the threshold).

`deadline_ms` bounds the superlinear worst case: on hard inputs (few anchorable unique
records: a character-level permutation is the measured shape) the Myers search's work
grows roughly ~n² with size (measured: 50k chars 0.32 s, 1M chars 183.6 s). On expiry the incomplete result is discarded and `TimeoutError` is raised
naming the elapsed cost and the deadline; `None` (the default) is the unbounded diff,
unchanged; a non-positive or non-finite value raises `ValueError` before any work runs
(an enormous-but-finite value is legal: it saturates to "unbounded" rather than
erroring).

```python
tors.diff_opcodes("qabxcd", "abYcd")
# [("delete", 0, 1, 0, 0), ("equal", 1, 3, 0, 2),
#  ("replace", 3, 4, 2, 3), ("equal", 4, 6, 3, 5)]
```

## `tors.diff_opcodes_lines`

```python
def diff_opcodes_lines(
    a: str, b: str, *, deadline_ms: float | None = None
) -> list[tuple[str, int, int, int, int]]: ...
```

**Async**: `await tors.aio.diff_opcodes_lines(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

The line-level spelling of `diff_opcodes`: the same opcode shape, engine, validity
contract, boundary-class divergences, and `deadline_ms` machinery; but the operands
are tokenized as lines (each line keeps its `\n`, the last may lack one) and the
indices address lines, so `a_lines[i1:i2]` slicing reconstructs. Measured on the
12 MiB near-identical pair: 5 line-opcodes and 1.9–2.6 ms walls where the char-level
spelling emits 235 opcodes over 76–88 ms: one call instead of a Python split
round-trip plus the diff.

**The tokenization is `'\n'`-only, not full `str.splitlines()`.** `a_lines`/`b_lines`
must be built as `a.split("\n")`-with-terminators-reattached (equivalently,
`re.split(r"(?<=\n)", a)` with a trailing empty string dropped if `a` ends
in `'\n'`) for the reconstruction contract above to hold: not
Python's `str.splitlines(keepends=True)`, which additionally breaks on `\r`,
`\r\n` collapsed to one line, `\v`, `\f`, `\x1c`–`\x1e`, `\x85`, U+2028 LINE SEPARATOR, and U+2029 PARAGRAPH
SEPARATOR. Text using any of those as its only line terminator (classic Mac
`\r`-only line endings are the realistic case) tokenizes as one line under
`diff_opcodes_lines` where `str.splitlines()` would see several, a real,
measured divergence. `'\n'` is the terminator every
practical document/version-diff pipeline actually uses; the narrower contract is
documented.

```python
tors.diff_opcodes_lines("l1\nl2\nl3\n", "l1\nX\nl3\nl4\n")
# [("equal", 0, 1, 0, 1), ("replace", 1, 2, 1, 2), ("equal", 2, 3, 2, 3),
#  ("insert", 3, 3, 3, 4)]
```

## `tors.replace_many`

```python
def replace_many(text: str, replacements: dict[str, str]) -> str: ...
```

**Async**: `await tors.aio.replace_many(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Simultaneous multi-pattern replace in one GIL-released native pass, a primitive
CPython does not have: every occurrence of every key is replaced by its value with
`find_patterns`'s exact search semantics.

- **leftmost-longest**: among the keys matching at a position, the longest wins
  regardless of dict order (not `re.sub` alternation's leftmost-first priority);
- **non-overlapping**: the scan resumes at each match's end;
- **never re-scanned**: a value that itself contains a key does not cascade:
  `replace_many("a", {"a": "ba"})` is `"ba"`, not `"baba"`.

The alternatives are all worse: chained `str.replace` calls are N whole-text
GIL-held passes, and `re.sub` alternation is leftmost-first and rescans its own
output. The canonical consumers are redaction and normalization maps. Identity
contract: `replace_many(s, m) is s` exactly when `replace_many(s, m) == s` (no
match, net-identity, or the empty dict all return the original object). Argument
contract: `replacements` must be exactly a `dict[str, str]` (a list/tuple of pairs
raises `TypeError`; dict order cannot matter: pinned); an empty key raises
`ValueError("empty pattern")`; lone surrogates raise `UnicodeEncodeError` at the
standard str-in boundary. No list-shape marshalling class: the return is one
string; the GIL-held residue is the argument walk plus the O(output) marshalling,
measured at the ping floor even on a 1.28M-replacement dense map.

**Output ceiling.** The spliced output's byte size is computed before anything is
allocated (saturating arithmetic over the match spans) and refused past **32 MiB**
with a catchable `ValueError` naming the refused size and the ceiling — the
documents layer's own byte contract (`DEFAULT_ANYDOC_INPUT_LIMIT`), one
convention for how many bytes is one tors object. The unit is UTF-8 bytes, and no
replacement value is ever truncated mid-character: values are spliced whole or
the call is refused. A sub-ceiling allocation refusal surfaces as the same
catchable `ValueError` (the reservation is attempted with `try_reserve`, never an
uncatchable allocator abort). The ceiling applies before the identity contract's
borrowed return, so a replace whose output would exceed it raises even when the
net effect would be the identity — a >32 MiB "identity" replace is the
amplification shape wearing a disguise. The masked spelling below is
length-preserving and has no ceiling (a short key cannot amplify into a long
output there).

```python
tors.replace_many("the cat sat in the catalogue", {"cat": "dog", "catalogue": "library"})
# "the dog sat in the library"
```

## `tors.replace_many_masked`

```python
def replace_many_masked(text: str, replacements: dict[str, str], mask: str = "*") -> str: ...
```

**Async**: `await tors.aio.replace_many_masked(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

The length-preserving redaction spelling of `replace_many`: same one-automaton
leftmost-longest, non-overlapping, never-rescanned scan, but each matched span's
character length is never allowed to change. For a matched span of L characters
and its key's replacement value of C characters, exactly one branch fires:

- **C >= L (value at least as long) means truncation**: the value's first L characters
  are emitted; `mask` is never consulted.
- **C < L (value shorter) means padding**: the whole value is emitted, followed by
  `L - C` copies of `mask`.

The replacement value is never ignored: it is truncated or padded to the
matched pattern's own character length, not masked outright. To redact a span
completely, use `""` as the value: the empty value always takes the padding
branch and becomes pure `mask * L`.

```python
tors.replace_many_masked("the cat sat", {"cat": "[REDACTED]"}, "*")
# "the [RE sat"   -- "[REDACTED]" (10 chars) truncates to "[RE" (3 chars); mask unused
tors.replace_many_masked("the cat sat", {"cat": "X"}, "*")
# "the X** sat"   -- "X" (1 char) pads to "X**" (3 chars)
tors.replace_many_masked("the cat sat", {"cat": ""}, "*")
# "the *** sat"   -- empty value pads to pure mask
```

Because every matched span is replaced by exactly as many characters as it
matched, and every non-matching span is copied through untouched, the output's
character count and every non-matching span's offsets equal the input's:
offsets computed before redaction (`find_patterns` spans, word/sentence bounds,
diff opcodes) stay valid against the redacted text. Byte length may still change
(a multibyte value or mask replaces the matched key's bytes); only the
character-unit guarantee is made, which is the unit every offset this crate
reports uses.

`mask` must be exactly one character (`str` of length 1, i.e. one Python
codepoint): a multi-character or empty `mask` raises `ValueError`, including a
visually-single grapheme built from more than one codepoint (e.g. NFD `"e" +
COMBINING ACUTE ACCENT"`, two codepoints, is rejected). The rest of the argument
contract is `replace_many`'s exactly: `replacements` must be a `dict[str, str]`,
an empty key raises `ValueError("empty pattern")`, lone surrogates raise
`UnicodeEncodeError`. Identity contract: `replace_many_masked(s, m, c) is s`
exactly when `replace_many_masked(s, m, c) == s` (the empty dict, no match, or a
net-identity mask/truncation/padding all return the original object).

## `tors.sentence_bounds` / `tors.sentence_bounds_iter`

```python
def sentence_bounds(text: str) -> list[tuple[int, int]]: ...
def sentence_bounds_iter(text: str) -> Iterator[tuple[int, int]]: ...
```

UAX #29 sentence segmentation (rules SB1–SB999) via the same `unicode-segmentation`
tables as the word segmenter: the other segmenter the stdlib lacks. Offsets are
Python `str` indices (`text[start:end]` IS the sentence); per the standard
(SB10/SB11), trailing spaces after a terminator belong to the preceding sentence.
Pinned by a hand-derived rule-cited table (`"3.4"` SB6, `"U.S.A"` SB7, `"etc. and
so on"` SB8, the ideographic `。`, the degenerates) plus structural properties and
list/iter sequence parity. Measured at 12 MiB of prose: 170,037 sentences (~1/22nd
the word count), so the list API's marshalling meets the shared GIL budgets. The
limitations note shared with `word_bounds`: rule-based UAX #29 only, no dictionary
segmentation for spaceless scripts (Thai/Khmer/Burmese/Japanese); ICU4X is the
heavyweight alternative if you need those.

```python
tors.sentence_bounds("One. Two. U.S. stocks fell.")
# [(0, 5), (5, 10), (10, 27)]
```

**Async**: `await tors.aio.sentence_bounds(...)` runs this under
`asyncio.to_thread` (see [Async use](async.md)).

## `tors.sentence_count`

```python
def sentence_count(text: str) -> int: ...
```

`word_count`'s shape over sentences: a single `int`, `O(1)` memory, the whole scan
GIL-released. `sentence_count(t) == len(sentence_bounds(t))`, pinned.

```python
tors.sentence_count("One. Two. U.S. stocks fell.")
# 3
```

## `tors.find_patterns`

```python
def find_patterns(patterns: list[str], text: str) -> list[tuple[int, int, int]]: ...
```

**Async**: `await tors.aio.find_patterns(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Leftmost-longest, non-overlapping multi-pattern substring search in one GIL-released
native pass: every occurrence of every pattern, reported as `(start, end,
pattern_index)` with `end` exclusive, offsets in Python `str` (codepoint) units:
`text[start:end] == patterns[pattern_index]` for every reported match.

- **leftmost**: a match is reported at the earliest position any pattern matches;
- **longest**: among the patterns matching at that position, the longest wins
  regardless of list order (not regex-alternation leftmost-first priority);
- **non-overlapping**: the scan resumes at each match's end; results come back in
  strictly increasing start order;
- **duplicates report the first index** they occupy in the list.

No stdlib primitive has these semantics (`re` alternation is leftmost-first), so the
contract is proven against a brute-force pure-Python leftmost-longest reference plus
structural validity over hypothesis-generated multi-byte alphabets. Argument contract:
exactly `list[str]` (a tuple raises `TypeError`); an empty pattern string raises
`ValueError("empty pattern")`; an empty list returns `[]` immediately; lone surrogates
raise `UnicodeEncodeError` at the argument boundary.

```python
tors.find_patterns(["cat", "catalogue"], "the cat sat in the catalogue")
# [(4, 7, 0), (19, 28, 1)]
```

## `tors.find_patterns_iter`

```python
def find_patterns_iter(patterns: list[str], text: str) -> Iterator[tuple[int, int, int]]: ...
```

The streaming spelling of `find_patterns`: the whole search (pattern-list walk,
automaton build, scan, byte→char offset conversion) fills an internal buffer under
one GIL-released pass at construction, and each `__next__` holds the GIL only for a
single 3-tuple of ints: the streaming answer to the list shape's O(matches)
tuple-marshalling caveat (~13 ms held per 100k matches in the list shape). Same
match sequence as the list API, pinned to sequence parity.

```python
list(tors.find_patterns_iter(["cat", "catalogue"], "the cat sat in the catalogue"))
# [(4, 7, 0), (19, 28, 1)]
```

## `tors.count_matches`

```python
def count_matches(patterns: list[str], text: str) -> int: ...
```

**Async**: `await tors.aio.count_matches(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

The count spelling of `find_patterns`: the same leftmost-longest, non-overlapping
search answering just the number, with no match vector materialized (the list
spelling would materialize ~250 MiB of matches on a 100 MiB dense corpus just to
answer "how many") and no byte→char offset pass (counting is offset-free).
`count_matches(p, t) == len(find_patterns(p, t))`, pinned. A single `int` return: no
marshalling class at all.

```python
tors.count_matches(["cat", "catalogue"], "the cat sat in the catalogue")
# 2
```

## `tors.contains_unescaped` / `tors.find_unescaped`

```python
def contains_unescaped(haystack: bytes, needle: bytes) -> bool: ...
def find_unescaped(haystack: bytes, needle: bytes) -> int: ...
```

Escape-parity-aware byte search in one GIL-released native pass: an occurrence of
`needle` at byte offset `i` counts only when the maximal run of `\` immediately
before `i` has **even** length (an empty run is even, so an occurrence at offset
0 counts). An odd run means the run's backslash pairs escape each other and the
leftover one escapes the occurrence's first byte — the occurrence is literal
text, not the sequence. `find_unescaped` returns the offset of the first
unescaped occurrence, `-1` when none exists (`bytes.find`'s own sentinel, kept
over `Optional[int]` deliberately: it is the spelling stdlib callers already
branch on); `contains_unescaped` is the boolean spelling of the same question.

The motivating case is JSON, and it is byte-ambiguous by construction: a
serializer (orjson is the measured consumer) renders a real NUL codepoint
(U+0000) as the six-byte escape text `\u0000` and the literal six-character
text of the same spelling as seven bytes (the backslash itself escaped), and
the second contains the first at offset +1 — so a plain substring test answers
"yes" for both. PostgreSQL settles it downstream: a real NUL is fatal in a
`jsonb` column (SQLSTATE 22P05), the literal text is fine, so a pipeline that
binds serialized JSON must know which one it holds before the INSERT — reject
the wrong one and you either refuse legal text or ship a value that fails the
bind. The stdlib's only spelling is find-then-re-parse-the-whole-value-and-
recursively-walk-it; the parity rule answers from the raw bytes directly:

```python
tors.find_unescaped(b'"a\\u0000b"', b"\\u0000")  # 2: the real NUL escape
tors.find_unescaped(b'"a\\\\u0000b"', b"\\u0000")  # -1: the literal text
tors.contains_unescaped(b'"a\\\\u0000b"', b"\\u0000")  # False
```

Contract details, each pinned in tests/test_unescaped_scan.py:

- **The offsets are byte offsets, not `str` indices**: the return indexes the
  `bytes` it was handed — `haystack[i:i + len(needle)] == needle` for every
  answer that is not `-1`. Over multibyte UTF-8 content the byte offset and the
  decoded text's character offset are different numbers (`find_patterns` needed
  a byte→char mapping for exactly this confusion; this API has none, because
  the input is bytes and the contract is byte-space end to end).
- **Rejected hits advance the scan one byte past the hit, not past the whole
  match**, so self-overlapping needles stay correct: the two-byte needle
  `b"00"` in a haystack of one backslash then `000` rejects the hit at 1 and
  finds the overlapping live hit at 2, which a resume-at-match-end scan would
  skip entirely.
- An empty needle raises `ValueError("empty needle")` (it would match at every
  position and has no parity meaning, the `find_patterns` empty-pattern
  rationale); both arguments take exactly `bytes` (`bytearray` / `memoryview` /
  `str` raise `TypeError`, the bytes-in surface's zero-copy immutable-borrow
  contract).
- **No JSON knowledge lives in the functions**: parity is the mechanism;
  "the needle is an escape sequence, so even means live and odd means literal"
  is the caller's reading of it. Any backslash-escaped grammar (printf format
  strings, shell quotes, regex sources) can drive the same scan.

GIL behavior: `utf8_is_valid`'s class exactly — two zero-copy `PyBytes` borrows,
the whole memmem scan plus the per-hit parity walk under one `py.detach`, and
`bool`/`int` returns, so there is no marshalling class at all and no error path
past the empty-needle `ValueError` (raised under the GIL, before the detach).
The 12 MiB pass sits well under the 10 ms heartbeat floor (memchr-class;
≈0.26 ms no-match and ≈0.70 ms over a 72,520-rejected-hit false-positive
corpus — box- and load-dependent figures from the wall cells in
tests/test_unescaped_scan.py, same order as the ≈0.25/≈0.79 ms inline
band in tests/test_gil_release.py), so the heartbeat cells are
ceiling-only. No aio twin: a sub-millisecond call needs no thread hop.

## `tors.utf8_byte_len`

```python
def utf8_byte_len(s: str) -> int: ...
```

The UTF-8 byte length of a `str`: `len(s.encode("utf-8"))` with the copy taken
out. That expression allocates a full `bytes` object, measures it, and throws
it away — pure waste whenever only the count is wanted, which is the shape of
every size cap in front of a store. The motivating sites are a write
path's: argument validation checks idempotency-key and scope byte caps on
every enqueue, and a terminal handler re-encodes a serialized result of
up to 64 KiB (`MAX_RESULT_BYTES`) on every success just to take its
length — a
genuine double pass, the byte count having existed inside the serializer's
output and been discarded by the `.decode()` that produced the `str`.

Fresh vs repeat, stated up front (the cold-lane economics the lane table
pins: cold-encode 4.9 ms | first-call 4.7 ms | warm-encode 196 µs |
cached-tors 0.13 µs at 12 MiB): a terminal fresh `str` per success never
repeats on the same object, so the one-shot lane is encode-parity by
construction — the win there is only the absence of a Python-visible
`bytes` object, not wall time. The wall-time win is repeat counts on the
same object (O(1) cached borrow vs a fresh alloc+memcpy per `encode`)
and every ASCII count (zero-copy alias, flat ~0.1 µs). Memory on every
lane is zero-alloc: one field read off the borrowed `&str` (the chunked
count is the utf16 twin's shape; this twin allocates nothing).

A companion, not a standalone motivation: this ships in the scan family's
binding module as the pinned companion of `contains_unescaped`/
`find_unescaped`, same module and same harness patterns — and honest sizing
says it would not stand alone (a short-string encode is a few hundred
nanoseconds; the win is large inputs and hot paths, where the copy is the
cost — the 64 KiB terminal case is ~0.9 µs of pure alloc+memcpy per success,
the lane table's ASCII 64 KiB expression cell).

The implementation is the standard str borrow, not arithmetic over CPython's
internal UCS storage: pyo3's `to_str` hands the core a Rust `&str`, whose
`len()` IS its UTF-8 byte length — one field read, the mechanism dried into
the language instead of hand-rolled (an unsafe `PyUnicode_KIND`/data walk with
surrogate-pair arithmetic would exist only to avoid one cached materialization).
The cost model that buys, measured — all wall numbers below are
CPython-3.12 calibration-box figures (arm64, release build; re-measure
per target — semantic pins are the contract, timing is not; the full
lane table is in `tests/test_performance.py`):

- **ASCII** (serialized JSON with `ensure_ascii=True` is pure ASCII): compact
  ASCII data is its own UTF-8, so the borrow is a zero-copy alias and the call
  is O(1) with no allocation at all — measured flat ~0.1 µs from 1 KiB to
  12 MiB, against the expression's alloc+memcpy every call (~0.9 µs at 64 KiB,
  ~14 µs at 1 MiB, ~180 µs at 12 MiB).
- **Non-ASCII, first call on the object (a cold UTF-8 cache)**: CPython
  materializes and caches the UTF-8 view on the `str` object (an internal
  cache, not a Python-visible `bytes`), so the first call is O(n) —
  encode-parity in cost class (measured within ~10-20% of a cold encode: the
  same encoder pass plus a malloc plus a second memcpy into the permanent
  cache), with no Python-visible object to allocate and collect. What "cold"
  means, exactly: the cache is filled by the str-in borrow itself — this
  call, or any earlier str-in tors call on the same object — and nothing
  else fills it. `encode` shares the cache but only reads it, so the sharing
  is one-directional: after one `utf8_byte_len`, a subsequent
  `len(s.encode())` on the same 12 MiB object dropped from ~4.9 ms to
  ~184 µs, measured — while a prior `len(s.encode())` does not warm this
  lane at all: the first `utf8_byte_len` after an encode still pays the full
  materialization (measured ~3.8-4.8 ms at 12 MiB on fresh objects, the cold
  class exactly; observed on CPython 3.12 here, expected from the sources
  on 3.10–3.14 — see `docs/cache-proof.md` for the per-version
  `Objects/unicodeobject.c` links — where the encode path reads the cache
  and only the `PyUnicode_AsUTF8AndSize` borrow writes it). Semantic pins
  (same object, same answer) are the contract; timing is not — the cache
  is CPython-internal and the wall cells record it, never assert it.
- **Non-ASCII, repeat calls on the same object**: O(1) — strictly better than
  the expression, which re-copies on every call (measured ~0.1 µs against the
  warm expression's 1.8 µs at 64 KiB and 196 µs at 12 MiB).

```python
tors.utf8_byte_len("café")  # 5: three ASCII bytes + one two-byte é
tors.utf8_byte_len("\U0001f600")  # 4: one astral codepoint, four bytes
# the byte-cap gate a terminal handler spells on every success:
if tors.utf8_byte_len(serialized_result) > 64 * 1024:
    reject()  # over MAX_RESULT_BYTES — no bytes object built to find out
```

Error parity is exact: a `str` holding lone surrogates cannot be UTF-8-encoded,
and the borrow raises CPython's own `UnicodeEncodeError` (pyo3 propagates it
before any tors code runs) — the same exception `encode` raises, attributes
included (`.encoding`, `.reason`, `.start`/`.end`, `.object`, pinned
attribute-for-attribute in tests/test_utf8_byte_len.py); there is no
tors-side error path. The argument takes exactly `str` (`bytes` /
`bytearray` / `memoryview` / `int` raise `TypeError`).

GIL behavior: a single `int` return, no marshalling class — and one honest
difference from the scan twins: the call's only O(n) work is the borrow
itself, which is GIL-held (a non-ASCII object's first call materializes the
UTF-8 view under the GIL; there is no way to fill an object's cache without
it). At 12 MiB that materialization measures ~5 ms, under the 10 ms heartbeat
interval, so the heartbeat cells are ceiling-only; the linear envelope
(~0.4-0.5 ms of GIL hold per MiB) puts a ~200 MiB non-ASCII string at the
100 ms ceiling — the recorded scale guidance for this one heavy lane. No aio
twin: an O(1)-to-borrow call needs no thread hop.

## `tors.utf16_byte_len`

```python
def utf16_byte_len(s: str) -> int: ...
```

The UTF-16 byte length of a `str`: `len(s.encode("utf-16-le"))` with the
2n copy taken out — 2 bytes per BMP codepoint, 4 per astral codepoint (a
surrogate pair). This is the other half of the `len()` → bytes pair
`utf8_byte_len` opens (the maintainer's ask: a util to convert a UTF8/UTF16
python `len()` into bytes for the API, because backend devs need byte caps
for storage and interop). UTF-16 is the code-unit world of JavaScript,
Java, Windows, and .NET — `String.prototype.length` counts UTF-16 units,
and an astral emoji is length 2 there — so column caps (`NVARCHAR`), wire
caps, and interop size checks in that world are UTF-16 bytes, and the
Python spelling of the count allocates and copies the entire 2n `bytes`
object just to throw it away.

The implementation is the utf8 twin's borrow plus derived arithmetic, no
FFI: the standard str borrow hands the core a Rust `&str`, and the core
derives the answer from its UTF-8 bytes — UTF-16 bytes =
`2 * (#codepoints + #astral codepoints)`, where over valid UTF-8
`#codepoints` is the lead-byte count and `#astral` is the count of 4-byte
lead bytes (`matches!(b, 0xF0..=0xF4)` — `0xF5..=0xFF` never occur in a
`&str`, so the closed range fails safe) — one pass of byte-class
arithmetic, no allocation, no per-codepoint decoding (the derivation, its
proof against a `chars()`-based naive count, and the exhaustive boundary
sweep are in `src/scan_impl.rs`). The corners the identity buys: no astral
codepoints means exactly `2 * len(s)` for ALL BMP text — CJK and combining
marks included, where the UTF-8 byte count diverges — pure ASCII means `2 *`
the UTF-8 byte count, and every answer is even. The final doubling is
checked arithmetic that refuses instead of wrapping — and it is
unreachable for real inputs on every width: a `&str` is at most
`isize::MAX` bytes and `#codepoints + #astral` never exceeds one per
byte, so the doubled answer is at most `2 * isize::MAX`, which fits
`usize` on 32-bit and 64-bit alike. The `OverflowError` contract is the
loud refusal if that invariant ever breaks, pinned at synthetic
boundary counts crate-side (the injectable combine unit).

The cost model, measured — all wall numbers below are
CPython-3.12 calibration-box figures (arm64, release build; re-measure
per target — semantic pins are the contract, timing is not; the full
lane table is in `tests/test_performance.py`):

- **Warm lanes, both corpus kinds alike** — the scan is
  representation-independent and the utf-16 expression pays its 2n
  alloc + encode pass on every kind: measured ~2.2 µs at 64 KiB against
  the expression's 8.8-10.4 µs, ~34 µs at 1 MiB against 142-147 µs, and
  ~400 µs at 12 MiB against 1.7-1.8 ms — ratios 0.21-0.26 on every warm
  lane. Repeat counts on the same object stay O(1)-borrow plus the
  detached scan; memory is zero-alloc on every lane (field read plus the
  chunked count, no `bytes` object on any lane).
- **Non-ASCII, first call on the object (a cold UTF-8 cache)**: the
  borrow materializes and caches the UTF-8 view first (the utf8 twin's
  cold class — encode-parity cost, GIL-held, and a prior `encode` does
  not warm it; observed on CPython 3.12 here, expected from the sources
  on 3.10–3.14 — see `docs/cache-proof.md`), then the scan runs
  detached. Measured 376 µs at 1 MiB against the utf-16 expression's own
  cold 146 µs — the one lane the expression wins, because it never
  touches UTF-8 at all; `utf16_byte_len` is NOT recommended for one-shot
  counts of fresh non-ASCII strings. The trade buys every subsequent
  call at 4-5x and the absence of a 2n `bytes` object per call. The
  fresh-object end-to-end bench is the `measured_not_asserted` cell in
  `tests/test_performance.py`.

```python
tors.utf16_byte_len("café")  # 8: four BMP codepoints, 2 bytes each
tors.utf16_byte_len("\U0001f600")  # 4: one astral codepoint, a surrogate pair
tors.utf16_byte_len("東京")  # 4: two BMP codepoints — the UTF-8 count is 6
# a JS/Windows column cap, counted without the 2n copy:
if tors.utf16_byte_len(s) > 280:  # over 140 UTF-16 units: NVARCHAR(140)'s budget
    reject()
```

The surrogate lane is REFUSAL PARITY with the replaced expression, measured
and pinned (a first-draft claim that the stdlib's utf-16-le encode accepts
lone surrogates was wrong — the test battery caught it): the strict
`encode("utf-16-le")` refuses lone surrogates exactly like the utf-8 codec
does ("surrogates not allowed"), so the function refuses the same strings
the expression itself refuses.

> **⚠ Breaking differences from the replaced expression.** The refusal
> decision is parity; the error SHAPE is not:
> (1) `.encoding` is `"utf-8"` (the str-in borrow materializing the UTF-8
> view is the step that fails) where the expression's error says
> `"utf-16-le"` — an encoding-label switch a caller matching on
> `.encoding` will observe;
> (2) on a multi-surrogate run the borrow's `(start, end)` names the
> whole run where the utf-16-le error reports only the first unit —
> a whole-run span vs a first-unit span;
> (3) `errors="surrogatepass"` (one 2-byte unit per lone surrogate, the
> 3-byte WTF-8 spelling in utf-8) is unsupported — no str-argument tors
> function offers a surrogate mode, since every one needs the UTF-8 view
> first.
> Pinned attribute-for-attribute in tests/test_utf8_byte_len.py, the
> byte-len family file (including the `encode("utf-16")` BOM form, which
> answers the le length plus the 2-byte BOM). The argument takes exactly `str` (`bytes` /
`bytearray` / `memoryview` / `int` raise `TypeError`) — it counts a `str`,
not decodes bytes; the decode side of UTF-16 is `decode_utf16`.

GIL behavior: the borrow is the call's GIL-held residue (a non-ASCII
object's first call materializes the UTF-8 view under the GIL — the utf8
twin's class, ~5 ms at 12 MiB, under the 10 ms heartbeat interval), and the
detach carries the real O(n) scan (~0.4 ms at 12 MiB, memchr-class) where
the utf8 twin's detach is nominal around one field read. The heartbeat
cells are ceiling-only; the linear envelope (~0.4-0.5 ms of GIL hold per
MiB) puts a ~200 MiB non-ASCII string at the 100 ms ceiling, the twin's
recorded scale guidance. No aio twin: the GIL-held residue is the borrow
alone, and the scan detaches.

## `tors.CompiledPatterns`

```python
class CompiledPatterns:
    def __init__(self, patterns: list[str]) -> None: ...
    def __len__(self) -> int: ...
    def find(self, text: str) -> list[tuple[int, int, int]]: ...
    def find_iter(self, text: str) -> Iterator[tuple[int, int, int]]: ...
    def count(self, text: str) -> int: ...
    def replace_many(self, text: str, replacements: dict[str, str]) -> str: ...
    def replace_many_masked(
        self, text: str, replacements: dict[str, str], mask: str = "*"
    ) -> str: ...
```

The `re.compile()` answer to `find_patterns`/`replace_many`'s per-call automaton build
(the same `tors.CompiledLemmaDict` pattern, applied to pattern search instead of
lemmatization; see `tors.apply_pipeline`'s docs below): building the Aho-Corasick
automaton is the expensive part of every one of these calls, and a fixed vocabulary
scanned over many documents (a redaction pipeline, a tagger) otherwise rebuilds it on
every single call for no reason. `CompiledPatterns(patterns)` builds it once under one
GIL-released pass; every method after that is the free function's exact scan minus the
automaton build, sharing the compiled automaton by one `Arc` clone per call, sound to
reuse across many calls and threads with no synchronization beyond that refcount.
The `*_iter` objects are the opposite: an iterator returned by e.g.
`word_bounds_iter` holds a mutable native cursor, belongs to one thread, and must
not be drained concurrently — a concurrent `__next__` raises `RuntimeError: Already
borrowed` (a clean exception, no corruption); the `Compiled*` classes are the
shareable ones.

Each method mirrors its free-function twin exactly: `cp.find(text) ==
tors.find_patterns(patterns, text)`, `cp.count(text) == tors.count_matches(patterns,
text)`, and so on, for every method above. The two `replace_many*` methods validate
their `replacements` dict at call time (values can change per call; only the pattern
set is fixed at construction): it must key exactly the compiled pattern set, every
compiled pattern paired with a value and no extra keys, or `ValueError` names the
unknown and missing keys. `len(cp)` is the number of compiled patterns. Construction
takes the same argument contract as `find_patterns`' pattern list (`list[str]`,
non-empty
entries); the empty pattern list compiles successfully (every scan finds nothing) and
then accepts only the empty replacements dict. Immutable once built: there is no way to
add or remove a pattern from an existing `CompiledPatterns`.

```python
cp = tors.CompiledPatterns(["cat", "catalogue"])  # pay the automaton build once
for doc in corpus:
    cp.find(doc)  # O(1) automaton reuse per call
cp.replace_many("the cat sat", {"cat": "dog", "catalogue": "library"})
# 'the dog sat'
```

## `tors.extract_code_blocks`

```python
def extract_code_blocks(
    text: str, lang: str | None = None
) -> list[tuple[str | None, str, int, int]]: ...
```

**Async**: `await tors.aio.extract_code_blocks(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Extracts every fenced code block in `text` per CommonMark §4.5's fenced-code-block
grammar, one GIL-released native pass: `(language, code, start, end)` per block, `end`
exclusive, offsets in Python `str` (codepoint) units. The grammar is hand-rolled
directly in Rust rather than delegated to a general Markdown parser: LLM chat output
is rarely deeply-nested Markdown, so the extra correctness a full CommonMark engine
buys is mostly wasted weight for this. Rules: 0–3 leading spaces, then 3+ backticks or
tildes of the same character open a fence; an optional info string's first word is the
reported `language`; content lines are dedented by the fence's own indentation; a
closing fence needs the same character, at least as long as the opener, with nothing
but whitespace after it; absent one, the block simply runs to end of input (an
unterminated fence still yields a block). `lang=` filters to an exact, case-sensitive
language match. The scope is narrower than full CommonMark: no indented-code-block
recognition, no tab-expansion of indentation; both are documented non-goals.

```python
tors.extract_code_blocks("hi\n```py\nprint(1)\n```\n")
# [("py", "print(1)\n", 3, 22)]
```

## `tors.strip_code_fences`

```python
def strip_code_fences(text: str) -> str: ...
```

**Async**: `await tors.aio.strip_code_fences(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Unwraps the single most common case on its own: a whole response wrapped in one
fence. If `text`, trimmed of leading/trailing whitespace, is *exactly* one fenced
code block, its dedented content comes back; anything else (prose around a fence,
multiple blocks, no fence at all) comes back completely unchanged (not even
whitespace-trimmed), so it is safe to call unconditionally on arbitrary model output.
`tors.strip_code_fences(s) is s` exactly when `s` is not that single-block case (the
same identity-return idiom as `normalize`/`replace_many`).

```python
tors.strip_code_fences("```py\nprint(1)\n```")
# "print(1)\n"
```

## `tors.dedent`

```python
def dedent(text: str) -> str: ...
```

`textwrap.dedent(text)`, byte-exact against CPython 3.14's rewritten algorithm
(`Lib/textwrap.py`, gh-131792): the longest common leading-whitespace-run *string*
shared by every non-whitespace-only line is stripped from each line, and
whitespace-only lines normalize to empty. Tabs and spaces are distinct characters for
the common-prefix computation, exactly as the stdlib documents (`"  x"` and `"\tx"`
share no margin); never tab-expanded. `tors.dedent(s) is s` exactly when
`tors.dedent(s) == s`. The natural companion to
`extract_code_blocks`/`strip_code_fences` for re-indenting pulled-out code, and a
GIL-released primitive in its own right for any pipeline calling `textwrap.dedent` on
large text today.

**Version note**: `textwrap.dedent` was rewritten in CPython 3.14, and the rewrite
changed observable behavior, not just performance: a line consisting solely of some
*other* Unicode whitespace character (`\v`, `\f`, a non-breaking space, ...) is now
recognized as blank and normalized, where the pre-3.14 implementation only recognized
`[ \t]` as blank-line whitespace. `tors.dedent` ships the 3.14 behavior unconditionally
on every Python version it supports (3.10+), the same cross-version parity convention
`b64_decode` uses (see its docs above): on an older interpreter, `tors.dedent` and that
interpreter's own `textwrap.dedent` can diverge on inputs containing a non-space/tab
whitespace-only line. Ordinary space/tab indentation, the overwhelming common case, is
unaffected either way.

```python
tors.dedent("  a\n  b\n")
# "a\nb\n"
```

## `tors.repair_json`

```python
def repair_json(
    s: str,
    *,
    skip_json_loads: bool = False,
    ensure_ascii: bool = True,
    strict: bool = False,
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
    deadline_ms: float | None = None,
) -> str: ...
```

Repairs malformed JSON from LLMs, APIs, logs, and user input, returning the
repaired document as a JSON string: a Rust port of the Python library
json_repair (Stefano Baccianella, MIT,
[github.com/mangiucugna/json_repair](https://github.com/mangiucugna/json_repair)),
with parity pinned to json-repair==0.63.4: the upstream behavior corpus
is ported into `tests/test_json_repair_{core,schema,parity}.py`, and a
differential suite runs tors and json_repair over the same inputs, so the
pin is proven, not asserted. The repair parser handles the failure modes model output actually
exhibits: missing and trailing commas, unquoted keys and values,
single-quoted and curly-quoted strings, truncated containers, comments,
stray prose around the payload, Python-isms (`None`/`True`/`False`, tuple
literals), doubled quotes, and broken escapes. In the default mode it
repairs instead of raising, whatever the input looks like.

**The fast path.** Unless `skip_json_loads`, a strict `json.loads`-parity
parse of the (fence-unwrapped, below) input runs first, and valid JSON
short-circuits the repair parser. The result is re-serialized through a
`json.dumps`-parity serializer either way, so valid-but-noncanonical input
normalizes (`{ "a" : 1 }` comes back as `{"a": 1}`), exactly like
upstream, which also re-dumps. Under a `schema` the probe does not
short-circuit: valid JSON is validated and repaired if noncompliant before
any fallback to the schema-guided parser, so a schema is enforced on clean
input too.

**The empty-string sentinel.** When nothing recoverable is found, the
repaired value is the empty string and `repair_json` returns the empty
string, not `"\"\""`: upstream's convention for never returning a bare
pair of quotes. The price is an ambiguity shared with upstream: a
legitimately repaired top-level empty-string value renders identically.
Check `result == ""` for "nothing recoverable". **Under `schema=` the sentinel
does not escape as a return**: the empty-string value is itself validated against
the schema, so a non-string-typed schema answers the same `ValueError` every other
nonconformant value does (an object schema: `"" is not of type "object"`):
"nothing recoverable" becomes a raise, not a sentinel. A string-typed
schema accepts it, an empty string being a valid string. Pinned by
`tests/test_json_repair_native.py`.

**The fence pre-pass.** Before anything else, the whole input is tested
against CommonMark's fenced-code-block grammar, the same single-block
unwrapping `tors.strip_code_fences` documents. If the trimmed input is
exactly one fenced block, its content is unwrapped and repaired: tilde
fences, closers longer than their openers, and indented fences are handled
by the grammar rather than by the repair parser's garbage-skip reaching the
same answer by accident, and fenced-but-valid JSON takes the fast path
instead of the repair parser. The pre-pass also recovers fenced top-level
scalars (a fence wrapping just `"hi"` yields `"hi"`) where upstream returns
`""`: its one behavior change beyond parity, listed with the divergences
below. A response with prose around a fence, or multiple blocks, is not the
single-block case; compose with `tors.extract_code_blocks` for those:

```python
tors.repair_json("{ 'name': 'Ada', 'role': 'admin', }")
# '{"name": "Ada", "role": "admin"}'

tors.repair_json('```json\n{"ok": true}\n```')
# '{"ok": true}'

# a multi-block response: pull the json blocks, repair each
blocks = tors.extract_code_blocks(response, lang="json")
repaired = [tors.repair_json(code) for _, code, _, _ in blocks]
```

**`skip_json_loads`** skips the upfront strict parse only, forcing the
repair parser even on valid JSON: upstream's knob for callers who already
know the input is broken. It does not skip the parser-internal suffix
probe: after a prose prefix, once a top-level container starts, the parser
still tries a targeted strict decode of the value from that point; that
probe is part of the repair parser proper and stays on, exactly as
upstream behaves.

**`ensure_ascii`** is the one `json.dumps` serialization knob carried over:
`True` (the default) escapes every non-ASCII codepoint as `\uXXXX` (astral
characters as surrogate pairs), `False` emits them verbatim. Upstream's
other pass-through kwargs (`indent`, `sort_keys`, ...) are not ported.

**`strict`** flips the documented leniencies into `ValueError`s at the
first structural ambiguity instead of repairing past them: duplicate keys,
empty keys, a missing `:` after a key, an empty parsed value, an object
that parses empty but still carries characters, multiple top-level
elements, doubled quotes. The mode for input that should be valid and whose
first real defect you want named rather than patched. `strict=True`
together with `schema` raises
`ValueError("schema and strict cannot be used together.")`.

**`schema`** switches on schema-guided repair: the parsed value is aligned
to the schema (scalar coercions (`"4"` → 4, `"yes"` → `true`), fills for
missing values, defaults inserted for absent properties, extra properties
dropped where `additionalProperties` forbids them, union branches tried
until one validates), then validated in full, with failures raising
`ValueError` at the offending path (`"Expected string at $.name."`).
Accepts a JSON Schema dict, a boolean schema, or a pydantic v2 model (class
or instance): the model's `model_json_schema()` output is used directly,
with field `default`s and `default_factory`s injected into it (factory
first) and Enum-member defaults carried as their `.value`, so the
model → schema → LLM → repair → `Model.model_validate` agent loop needs no
manual schema step. Mutually exclusive with `strict` (above). See
`tors.repair_json_diagnostics` for the action-by-action log of everything
schema mode does.

**`salvage`** is the best-effort spelling of schema mode (upstream's
`schema_repair_mode="salvage"`, spelled as a bool): top-level fragments
that fail the schema are skipped until one validates, invalid array items
and extra properties are dropped rather than raised, and missing `required`
properties are filled from their subschema's `default`/`const`/`enum[0]`.
It requires a schema: `salvage=True` without one raises
`ValueError("salvage=True requires schema.")`.

**`deadline_ms`** (default `None` = unbounded) bounds the whole repair the
way `diff_opcodes`' `deadline_ms` does: a positive-finite-or-`None` budget
validated up front, `TimeoutError` on expiry. The clock starts at the top of
the call: the fence pre-pass and the `json.loads` fast-path attempt burn
the budget too, and a fast path that *completes* past the budget still
returns its answer (the deadline stops further work; it does not nullify
done work). It is a DoS backstop for the pathological O(n²) parser shapes
`tors` shares with upstream `json_repair` (duplicate-key-in-array splices,
empty-object splices, and a backslash-run string scan): a bounded *abort*,
not a speed-up, since a completing parse is
byte-identical whether or not a deadline is set, and a benign large document
does not trip a generous budget (the deadline discriminates pathological
*shape*, not *size*). When a schema is passed, the budget bounds the schema
alignment layer too: the key-remap ladder, the union and type-union branch
retries, scalar coercion, missing-key fill, enum suggestion scoring, and
validation all sample the same clock, with the same soft bound (at most one
property or one union branch past expiry). The enum suggestion loop
reads the clock before every member's comparison and hands the clock's
remaining budget to each jaro-winkler score, so both the many-members axis
and the one-very-long-member axis are bounded: a wide enum of long members
raises `TimeoutError` within a small multiple of the budget instead of
running the loop to completion, and an expired clock raises — a `None`
suggestion on an expired clock would mask the timeout as a plain data
miss. The key-remap ladder's fuzzy tier samples the same way — a forced
clock read per property (short window-disjoint key comparisons never
consult a clock internally) plus the remaining budget handed to each
comparison — so the one-unknown-key-over-a-wide-schema axis is bounded
like the enum's, by the same design. It applies to all three spellings and is checked with
the GIL released, so `TimeoutError` is raised after reacquiring it, the
same shape as `diff_opcodes`, including the message:
`"<spelling> deadline exceeded: elapsed 101.2ms > deadline_ms 100.0ms"`.

Two limits. The bound is *soft*: the tight loops sample the clock
1-in-256, but every O(n) unit (a buffer splice, a long scan, a wide span
build, one validation pass — the validator crate's error-carrying
`validate` is the most expensive opaque unit, so the validity gate rides
its boolean API and `validate()` always reads the clock before it) forces
the very next check to read it, so at most one such unit
runs past an expired budget (measured worst overshoot ~8% at n=1M). And
when a schema is passed, the schema's own PRE-alignment phase — the
repairer's construction: the resolve walk, the root validator's
`validator_for` compile, the address-set walk, all eager and linear in
the schema's size — runs BEFORE the clock is armed (the budget attaches
to the constructed repairer, `mod.rs` arms it right after `new`), so a
wide schema's fixed phase is not deadlineable: ~0.5s at 100,000
properties, whatever the budget. The alignment layer that follows samples
the clock per property, so the phase is the overshoot's whole body for a
wide-schema call (a 5ms budget over a 100k-property schema raises at the
first ladder consult, ~0.5s in); making that phase interruptible is a
design change (the compile is the validator crate's one opaque call), not
a sampling one. The
schema alignment layer shares the same clock throughout, including the
salvage unwrap's nested repair: a `salvage=True` call inherits the
caller's budget inside the unwrap instead of restarting unbounded. And
it bounds CPU *time*, not native stack growth: a runaway continuation
recursion can still overflow the stack before the budget expires; that
class is depth-guarded separately (`MAX_NESTING`), not time-bounded.
Cost when unset: nothing on the valid-JSON fast path, and one predicted
branch per dispatch turn in the repair parser (measured ~+6% worst-case
on a multi-MB `skip_json_loads=True` parse). Cost when set: ≤2% on
top of that (the checks are sampled).

Argument contract: a non-`str` `s` raises `TypeError` (pyo3 extraction); a
`schema` that is not a dict, bool, model, or `None` raises
`ValueError("schema must be a JSON Schema dict, boolean schema, or pydantic
v2 model.")`; a bad `locale` (below) raises a `ValueError` naming it; a
lone surrogate in `s` raises `UnicodeEncodeError` at the argument boundary,
the same str-in convention every function here documents.

**GIL.** One detached native pass covers the fence pre-pass, the repair
parse, the schema alignment, and the validator compile. The GIL-held
residue is the caller-supplied `schema` dict walk (converted into the
internal value tree before the pass starts) and the return marshalling
after it: one string for this spelling; for `tors.repair_json_loads` and
`tors.repair_json_diagnostics`, the construction of the Python object tree
and the diagnostics list, O(result): the same disclosed marshalling class
the list-returning search functions document (see `tors.find_patterns_iter`'s
O(matches) note and [Performance](performance.md)).

**Divergences from upstream json_repair**, the complete list; the
differential suite pins everything else to json-repair==0.63.4:

- **Not ported**: `stream_stable`, the text repair log (superseded by
  `tors.repair_json_diagnostics`), the file and CLI flavors
  (`json_fd`/`load`/`from_file` and the CLI module), `json.dumps`
  pass-through kwargs beyond `ensure_ascii`, and the `schema_repair_mode`
  string (the `salvage=` bool instead).
- **Lone surrogates**: a `\uXXXX` escape that decodes to an unpaired
  surrogate becomes U+FFFD (a Rust string cannot hold a lone surrogate,
  and pyo3 could not return one anyway). Input text containing literal lone
  surrogates never reaches the parser: it raises `UnicodeEncodeError` at
  the argument boundary, the same str-in convention every function here
  documents.
- **Non-ASCII digits**: Unicode Nd digits beyond 0-9 (Arabic-Indic and
  friends) do not enter the number path: upstream's `str.isdigit` is
  Unicode-wide, tors's check is ASCII-only. Pure runs (like `"١٢٣"`) fail
  to repair on both sides; a non-ASCII digit leading ASCII digits
  (`"²5"`) makes upstream abandon the value where tors skips the mark and
  parses the digits, the one mixed-run shape where the classes differ.
- **Fenced top-level scalars** are recovered where upstream returns `""`
  (the fence pre-pass above).
- **The validation boundary**: schema validation runs on the Rust
  `jsonschema` crate, so failure-message wording is that crate's, not
  Python `jsonschema`'s; a constrained schema position — an `enum` member
  value, `const`, `minimum`, `maximum`, `exclusiveMinimum`,
  `exclusiveMaximum`, or `multipleOf` — carrying an integer outside the
  exact-integer range (`i64::MIN`..=`u64::MAX`) refuses the schema
  outright with a `ValueError` naming the position's JSON pointer
  (e.g. `/properties/count/minimum`): compiled as `f64`, such a
  constraint cannot distinguish neighboring integers, so it would
  silently accept values the schema rejects, and tors refuses to guess
  rather than validate lossily (in-range integers — the whole
  `i64`/`u64` span — are exact and never refuse; schema floats keep the
  historical lossy-float reading, and the document's own huge integers
  are untouched); non-finite numbers under a schema raise `ValueError`
  where Python tolerates `NaN`; union branches are validated wrapped so
  `#/...` refs keep root scope (a pathological subschema-local `$defs`
  shadowing root `$defs` diverges). `format` is unasserted on both sides:
  upstream passes no `format_checker`, and tors matches it.
- **Deep nesting**: `ValueError("Input nesting exceeds the supported parser
  recursion depth.")` at 200 nested containers, where upstream raises an
  uncaught `RecursionError` at roughly its own recursion limit: the same
  failure normalized into the error catalog at a lower, pinned threshold.
- **Shared-reference schemas**: the schema walk visits every path, so a
  schema built from shared references (48 nested shared lists, depth 48)
  would expand exponentially; the walk caps container visits at the canon
  walk's 2,000,000-node ceiling and refuses past it with a catchable
  `ValueError` ("Input schema visits too many objects"). A legitimately
  large FLAT schema of the same node count is unaffected — the cap counts
  container visits, and a flat schema's cost is linear in the caller's own
  input.
- **On by default, tors-native**: key-typo remap, enum "Did you mean ..."
  suffixes,   date/uuid normalization, numeric extraction tiers, and the
  diagnostics output are extensions upstream does not have; see
  `tors.repair_json_diagnostics`. One consequence: on the strict fast path
  tors normalizes already-valid values (date formats, fold-matching key
  spellings, directly-declared `properties` and `allOf` members) where
  upstream's valid-JSON shortcut returns them untouched, matching what
  upstream's own repair lane does with `skip_json_loads=True`, so tors is
  self-consistent across its two lanes for everything except
  `oneOf`/`anyOf`-wrapped guidance, where the fast path does not guess
  which branch applies (upstream's fast path has the same
  reach).

## `tors.repair_json_loads`

```python
def repair_json_loads(
    s: str,
    *,
    skip_json_loads: bool = False,
    strict: bool = False,
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
    deadline_ms: float | None = None,
) -> dict[str, Any] | list[Any] | str | int | float | bool | None: ...
```

The `json.loads` drop-in spelling of `tors.repair_json`: the same repair
pipeline and the same knobs (`skip_json_loads`, `strict`, `schema`,
`salvage`; see `tors.repair_json`'s docs above), with the decoded object
returned directly instead of a re-serialized string, so a repaired document
goes straight into use with no `json.loads` round trip. There is no
`ensure_ascii` here because nothing is serialized. The empty-string
sentinel carries over as a real `""` return ("nothing recoverable",
ambiguous with a legitimately repaired top-level empty-string value, the
same collapse upstream's `loads` has). Under `schema=` the sentinel is
schema-validated instead: a non-string-typed schema turns "nothing
recoverable" into the same `ValueError` every other nonconformant value
raises; see `tors.repair_json`'s sentinel paragraph above.

```python
tors.repair_json_loads("{'users': [{'name': 'Ada',}]}")
# {'users': [{'name': 'Ada'}]}
```

## `tors.repair_json_diagnostics`

```python
def repair_json_diagnostics(
    s: str,
    *,
    skip_json_loads: bool = False,
    strict: bool = False,
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
    deadline_ms: float | None = None,
) -> tuple[dict[str, Any] | list[Any] | str | int | float | bool | None, list[dict[str, Any]]]: ...
```

The `(value, diagnostics)` spelling: `tors.repair_json_loads`'s exact
result paired with one record per repair action taken. Upstream narrates
these actions to its `logging` facility; tors does not port that text log:
this function is the replacement, the same narration as data. Each record
carries `action` (from the closed vocabulary below), `path` (json_repair's
path spelling: `"$"`, `"$.key"`, `"$.items[3]"`), `detail` (one human
sentence), and, where the action has them, `from`/`to` (the value before
and after) and `suggestion`. The vocabulary:

| action | when it fires |
|---|---|
| `coerce` | a scalar is converted to the schema's type: `"4"` → 4, `"4.0"` → 4, `4` → `"4"`, `"1.5"` → 1.5, `"yes"`/`"on"`/`"1"` → `true` and `"no"`/`"off"`/`"0"` → `false`, a number to its truthiness |
| `fill` | a missing value (an object value slot that runs straight into `,` or `}`, e.g. `{"key":}`) is filled from the schema: `const`, else `enum[0]`, else `default`, else the type's empty value (`""`, `0`, `false`, `[]`, `{}`, `null`) |
| `insert_default` | a property absent from the object whose subschema declares `default` (and is not `required`) gets that default |
| `remap_key` | tors-native: a key matching no property is remapped to the closest property name; thresholds below |
| `suggest` | tors-native, report-only: a near-miss that was not auto-corrected: a key, an enum member, an ambiguous date, or a disclosed numeric-format assumption, per the four features below |
| `drop_property` | an extra property not covered by the schema is dropped (`additionalProperties` does not allow it) |
| `drop_item` | an array item is dropped: invalid under its item schema while salvaging, or beyond tuple-form `items` not covered by `additionalItems` |
| `unwrap_string` | a string value holding a JSON document is parsed and unwrapped to the object/array the schema expects (under salvage, repaired first if merely malformed) |
| `wrap_array` | a non-array value is wrapped in a single-element array to match an array schema |
| `fill_required` | salvage: a `required` property missing from the object is filled from its subschema's `default`/`const`/`enum[0]` |
| `format_date` | tors-native: a date/date-time string is normalized to the RFC 3339 form; below |
| `skip_fragment` | salvage: a top-level fragment that does not match the schema is skipped while hunting for one that does |
| `map_array_to_object` | salvage: a list with exactly the schema's property count is mapped onto those property names, in order |
| `unwrap_root_array` | salvage: a single-item root array `[{...}]` is unwrapped to `{...}` |

**Scope of the log (v1).** Parser-level repair narration (the syntax-layer
fixes `repair_json` performs without a schema) is not recorded yet, so a
schema-free call returns an empty list; the log covers schema-layer actions
and the tors-native suggestions below.

**Key-normalization ladder (deterministic tier).** A key that differs
from a property only by case or separator style (`first-name`,
`First Name`, `FIRST_NAME` → `first_name`) remaps with confidence 1.0
(the match is exact after folding, not a guess), so it fires even on
permissive schemas and on the valid-JSON fast path (the un-remapped shape
strands real data on a dead key while the property takes its default).
One guard keeps it safe: the rename is kept only when the value can live
under the target property (a speculative repair through it succeeds), so
an incompatible value keeps its original, already-valid key instead of
turning valid input into a coercion failure. This tier also reaches
`allOf` members (pydantic's inheritance shape); `oneOf`/`anyOf` stay
unreached on the fast path.

**Key-typo remap.** When an object key matches no `properties` entry and no
`patternProperties` pattern, tors scores it against every property name
with `tors.jaro_winkler` and remaps it (`from` the old key, `to` the new)
only when the evidence is strong AND the alternative is loss or failure:
the best score is >= 0.75, it is unique (the second-best more than 0.05
lower), the target property is absent from the object, and either
`additionalProperties` is false (the key would otherwise be dropped) or the
target is in `required` (validation would otherwise fail). An exact
case-insensitive match scores a perfect 1.0. Below the remap bar, a best
score >= 0.60 still emits a report-only `suggest` diagnostic and falls
through to upstream semantics: no remap, no drop, the key keeps its
spelling.

**Enum suggestions.** When a value fails an `enum` check (never `const`),
the `ValueError` gains " Did you mean '...'?" naming the closest string
enum member under the same jaro-winkler >= 0.60 threshold, plus a
`suggest` diagnostic. Enum values are never auto-remapped: a near-miss is
reported, not guessed.

**Date, time, and uuid normalization.** For string values whose subschema
declares `format: "date"`, `"date-time"`, `"time"`, or `"uuid"` (directly
or through an `allOf` member), accepted forms normalize: ISO dates
(`YYYY-MM-DD`, two-digit padded) and slash dates (`YYYY/MM/DD`,
padding-tolerant); date-times `<date>[T ]HH:MM[:SS[.frac]]` on either date
spelling; month-name forms (`March 15, 2024`, `15 March 2024`,
`15 Mar, 2024`, case-insensitive). Dates come out `YYYY-MM-DD`;
date-times come out `YYYY-MM-DDTHH:MM:SS[.frac]` (seconds always
emitted, the fractional part trimmed to its shortest exact form), and an
input that carried an offset (Z or ±HH:MM/±HHMM) normalizes to its UTC
instant with a `Z` rendering (`2024-03-15T14:30:00+0530` →
`2024-03-15T09:00:00Z`); a missing offset stays missing. `format: "time"`
gains seconds (`14:30` → `14:30:00`); `format: "uuid"` canonicalizes to
lowercase when the shape is a UUID (non-UUID strings pass through for
validation to judge). Numeric `X/Y/YYYY` (or `YYYY/X/Y`) forms coerce only
when a component over 12 disambiguates month from day; when both
candidates are 12 or under (`03/04/2024`) the date is ambiguous and gets a
`suggest` diagnostic instead of a guess. Invalid calendar dates
(month lengths, leap years) are left for validation. `format` is not
otherwise enforced: upstream passes no `format_checker`, and tors matches
it; this normalization is the one place `format` is consulted at all.

**Numeric coercion ladder and `locale`.** String values under an
`integer`/`number` property climb a four-tier ladder: (1) the whole
trimmed string parses; (2) the string minus unambiguous noise (underscores,
fullwidth and Arabic-Indic script digits, and, with a known locale, that
locale's own separators) parses; (3) exactly one number token in the prose
extracts (`"USD 50"` → 50, `"$1,234.56"` → 1234.56, `-"50"` → -50), with
percent suffixes read by the declared type (`"50%"` → 0.5 on `number`
fields, the fraction; → 50 on `integer` fields, the percent count); (4)
Auto mode's separator-ambiguity resolution, below. A dropped decimal
marker never extracts (`.5` on an integer field refuses, never 5), and
Python's unbounded integer semantics hold at any magnitude:
`"12345678901234567890123"` coerces exactly, never a saturating cast.

`locale=` tells tors which separator convention the model uses: a BCP 47
tag string (`"de-DE"`, case-insensitive, `-` or `_`; region variants like
`de-CH` carry their CLDR separators; Lakh-style grouping locales such as
`en-IN` are refused) or a dict `{"decimal": ..., "grouping": ...}` of
one-character separators for conventions the table does not carry. With a
known locale every form is deterministic: `"1,234"` reads 1.234 in German
and 1234 in English, by data rather than by guess. The default (`locale=None`,
Auto) assumes en-US for the separator-ambiguous shapes, but never
silently: both readings are extracted and filtered by the declared type
and the property schema, a single surviving reading is the deterministic
answer (silent), and when both survive the en-US reading wins with a
`suggest` diagnostic naming the discarded reading's `locale=` override
(`"1,234"` on a number field → 1234 plus `locale='de-DE'`). A single
separated number whose readings all fail the declared type refuses with
the retry-able hint (`pass locale='en-US' or 'de-DE' ...`); prose without
a single number gets the plain upstream refusal.

```python
value, diags = tors.repair_json_diagnostics(
    '{"count": "4"}',
    schema={"type": "object", "properties": {"count": {"type": "integer"}}},
)
# ({'count': 4}, [{'action': 'coerce', 'path': '$.count', ...}])
```

## `tors.truncate_to_bounds`

```python
def truncate_to_bounds(
    text: str, max_chars: int, boundary: Literal["word", "sentence"] = "word"
) -> str: ...
```

Truncates `text` to at most `max_chars` codepoints, one GIL-released native pass,
cutting at the last word (or, `boundary="sentence"`, sentence) boundary at or before
`max_chars` instead of mid-word/mid-sentence: composing `word_bounds`/
`sentence_bounds`, the crate's own UAX #29 segmentation already shipped, rather than a
new algorithm. The context-window/token-budget-fitting primitive: `text[:max_chars]`
risks cutting a word or a grapheme in half, which this avoids.

Every cut point is also grapheme-cluster-safe: a word/sentence boundary that would
split a cluster (Thai SARA AM, combining accents, ZWJ emoji sequences) is never used,
so a combining mark is never separated from its base character. If no boundary fits at
or before `max_chars` (a single word/sentence longer than the budget, or `max_chars ==
0`), the fallback is a hard cut at the largest grapheme boundary `<= max_chars`: a
documented fallback, never a silent surprise, and still cluster-safe, so it
can land short of `max_chars` when the budget would otherwise split a cluster. The one
invariant that never breaks either way: the result never exceeds `max_chars`
codepoints. The cut point is then trimmed of trailing whitespace: whole whitespace
clusters only, so the trim never ends the result mid-cluster (a Prepend plus a
no-break space is one cluster; its non-whitespace half keeps it whole):
cutting right after a word/sentence boundary can otherwise leave a
dangling separator space, since `word_bounds` segments the inter-word space on its own
and `sentence_bounds` carries a sentence-terminal's trailing space on the preceding
sentence (UAX #29 SB9-SB11).

`tors.truncate_to_bounds(s, n) is s` exactly when `s` already has `<= n` codepoints
(the `Cow` identity convention: zero allocation, zero marshalling). `max_chars < 0`
and an unrecognized `boundary` both raise `ValueError` before any work runs.

```python
tors.truncate_to_bounds("cats are cute", 9)
# "cats are"
tors.truncate_to_bounds("One. Two. Three.", 10, boundary="sentence")
# "One. Two."
```

## `tors.truncate_ellipsis`

```python
def truncate_ellipsis(text: str, max_chars: int) -> str: ...
```

The DB-column truncation shape: hard cut to at most `max_chars` codepoints plus
a U+2026 `…` marker, one GIL-released native pass, never mid-grapheme-cluster.
Unlike `truncate_to_bounds` there is no word/sentence awareness: a storage
bound is positional, not semantic, and the marker tells the reader the value
continues.

At most `max_chars - 1` codepoints are kept plus the one-codepoint marker, so
the result never exceeds `max_chars` (it falls short when cluster backoff
requires it: a cut landing inside a combining sequence, ZWJ emoji chain, or
regional-indicator flag pair snaps back past the whole cluster rather than
splitting it). On plain text the stored length is exactly the bound. No
trailing-whitespace trim: the cut is positional.

`tors.truncate_ellipsis(s, n) is s` exactly when `s` already has `<= n`
codepoints. `max_chars == 0` yields `""` (no room for even the marker; the
naive `value[:0] + "…"` spelling answers `"…"` here, exceeding a zero bound);
`max_chars < 0` raises `ValueError` before any work runs.

```python
tors.truncate_ellipsis("hello world", 6)
# "hello…"
```

**Async**: `await tors.aio.truncate_ellipsis(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

## `tors.is_grounded`

```python
def is_grounded(
    claim: str,
    source: str,
    *,
    fuzzy: bool = False,
    threshold: float = 0.85,
    deadline_ms: float | None = None,
) -> bool: ...
```

Checks whether `claim` is grounded in `source`: a lexical check, not a
semantic/NLI one, and not a hallucination-detection model.

`fuzzy=False` (the default) is exact substring containment: the `memchr` crate's
SIMD-skipped two-way search (`memmem`, already a dependency), a byte-level find that is
UTF-8-boundary-safe by construction. `fuzzy=True` compares `claim` against
overlapping windows of `source` (stride `claim`'s length / 2) using the
only diffing engine already in the crate (the `similar` Myers engine backing
`diff_opcodes`), and reports whether the best window's region score —
`2 * matched_chars / (len(claim) + len(claim))`, the difflib ratio over
equal-length operands — reaches `threshold`. Every window is scored against
that claim-length denominator: a window truncated by the source's end (its
last, shorter window) is scored as the claim-length region it truncates —
the missing characters are mismatches, never a discounted
`len(claim) + len(window)` denominator, which would inflate a source that
just ends partway through the evidence above the identical evidence sitting
mid-source and make the verdict depend on where the evidence sits. The one
exception is a `source` shorter than the claim: there is nothing to window
over, so the whole source is the evidence and the score is one direct
difflib `2 * matched / (len(claim) + len(source))` ratio (the unwindowed
convention `tests/test_grounded.py` pins to exact difflib parity); the two
formulas agree at `len(source) == len(claim)`, so the boundary is continuous.
`fuzzy=True` is a
superset of `fuzzy=False`: an exact-containment floor runs first, so a claim present
verbatim in `source` is grounded before any windowing (independent of window alignment,
and before `deadline_ms` applies: a verbatim substring never times out). The windowed
score is consulted only when there is no exact match.

The floor guarantees the verbatim case unconditionally; near matches get a bounded
guarantee band instead of raw window luck: a same-length source region whose aligned
ratio is `r` is detected at any offset whenever `r >= max(0.75, threshold + 1/32)`:
a bounded refinement pass re-scans the best coarse windows at a fine stride, a
constant budget on top of the linear scan. One substitution in a 9+ character claim
clears the `0.85` default wherever it sits. Below `r = 0.75` detection is
best-effort (the recall floor of the DoS windowing), and a genuine region can be
evicted from the 64 refinement candidates by adversarial decoy text scoring higher,
the regime `deadline_ms` exists for (both limits are pinned in `tests/test_grounded.py`).

Windowing, rather than one whole-string diff of `claim` against all of `source`, is
DoS discipline: the realistic RAG-grounding shape is a short claim against a
long retrieved passage, so bounding each diff's operands to roughly `claim`'s length
keeps the total work close to linear in `source`'s length instead of the O(source ×
claim) a single unwindowed diff would cost, and windows slide through one reusable
O(claim)-sized buffer, so a 12 MiB passage costs kilobytes rather than a
whole-source char vector, and an early exit stops consuming input mid-source.
`deadline_ms` (only accepted, and only
meaningful, when `fuzzy=True`) bounds the whole scan on top of that, the same
discretionary escape hatch `diff_opcodes`'s `deadline_ms` already has, checked after every window
diff, coarse and refinement alike: `TimeoutError`
on expiry naming the elapsed cost and the deadline, a positive-finite-or-`None`
precondition validated before any work runs. Even a single very large window's own
Myers search is itself deadline-bounded (`similar`'s `capture_diff_slices_deadline`),
so one pathological window cannot blow through the budget uninterrupted between checks.

An empty `claim` is vacuously grounded in anything on both paths. `threshold` must be
in `[0.0, 1.0]`.

```python
tors.is_grounded("cat", "the cat sat")
# True
tors.is_grounded(
    "the cat sat", "Lorem ipsum. The cats sit on mats today.", fuzzy=True, threshold=0.6
)
# True
```

## `tors.similarity_ratio` / `tors.get_close_matches`

```python
def similarity_ratio(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...


def get_close_matches(
    word: str,
    possibilities: list[str],
    n: int = 3,
    cutoff: float = 0.6,
    *,
    deadline_ms: float | None = None,
) -> list[str]: ...
```

`difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()` and `difflib.get_close_matches()`'s
shapes at native speed, over the same Myers engine `diff_opcodes` uses.
`similarity_ratio` is `2.0 * M / T` (`T = len(a) + len(b)`, both in character units)
with `M` the matched-character total over the Myers equal-ops, difflib's own formula
but over a different alignment: difflib's `M` comes from its longest-match recursion,
which anchors a match and splits the surrounding change around it, so on
repeated-pattern inputs its `M` can be smaller than the maximal one. The pinned
divergence rows (`tests/test_similarity.py`): `"ppp"` vs `"pwpp"`, difflib `4/7`
(its anchored `"pp"` splits the insert, `M = 2`) vs tors `6/7` (`M = 3 = LCS`);
and `"qpqpq"` vs `"qpwqpq"`, difflib `6/11` (the anchored rotated equal block
`"qpq"`, a non-minimal insert+delete split, `M = 3`) vs tors `10/11` (`M = 5 =
LCS`). tors's `M` is always valid — the equal ops spell a common subsequence,
so `M <= LCS(a, b)` (structural, pinned as a size ladder in
`tests/test_similarity.py`) — and it is maximal on the forced-alignment
classes (identical, empty, pure insert/delete with differing flanks) and
measured exactly maximal through the ladder's 512-char octave, but it is
BOUNDED-maximal, not guaranteed maximal: from the 1024-char octave upward the
bounded middle-snake search can accept a good non-minimal split and score
slightly UNDER the true LCS ratio (measured: 0.7043 vs the true 0.7107 on
3000-char random strings, a ~1% undercount; in the pinned ladder the
undercounts start at 1024, worst observed ratio 0.972 at n=1024 over a
26-symbol alphabet, held above an empirical 0.9 drift-guard floor that is
NOT a contract). There is no lower bound on `M` past the search's limits;
see `src/diff_impl.rs`'s module docs for
exactly what the engine does and does not guarantee, and the size-ladder
property test in `tests/test_similarity.py` for the pinned bounded
property. So the two
ratios agree exactly wherever the alignment is forced (identical operands, empty
pairs, disjoint alphabets, pure insert/delete with differing flanks) and are both
valid but may diverge on repeated-flank contexts. difflib's anchored `M` is also
direction-dependent (`similarity_ratio` is symmetric; difflib's `ratio()` is not, in
general). `("", "")` is `1.0`, the convention both engines share.

A third divergence class sits past difflib's `autojunk` threshold: with the default
`autojunk=True` — the spelling `get_close_matches` uses internally and cannot turn
off — any element appearing more than `len(b)//100 + 1` times in a `b` of 200+
elements is junked before matching, so plain difflib's ratio collapses on such
inputs (`"y" + "x"*300` vs `"x"*300`: difflib `0.0`) while tors keeps the
un-heuristic'd answer (`0.9983…`), exact agreement holding with `autojunk=False` —
the oracle's spelling above (`tests/test_similarity.py` pins the threshold).

`get_close_matches` keeps every candidate scoring `similarity_ratio(candidate, word)
>= cutoff` and returns the top `n` sorted by score descending, then by the candidate
string descending: `heapq.nlargest`'s tuple order, the stdlib's own tie-break (`ac`
sorts after `ca`, so `get_close_matches("ab", ["ac", "ca"], 2, 0.5)` returns `["ca",
"ac"]`). Returned elements are the original candidate objects, not copies. `n <= 0`
and a `cutoff` outside `[0.0, 1.0]` raise `ValueError` with difflib's exact
message, the offending value interpolated (`"n must be > 0: 0"`, `"cutoff must
be in [0.0, 1.0]: -0.5"`, measured against the running stdlib on 3.10–3.15;
`n` is taken signed at the pyo3 boundary precisely so every `n <= 0` case lands
in `except ValueError` identically to difflib's own). An empty `possibilities`
list returns `[]`.

`deadline_ms` bounds the whole call: every candidate's diff for `get_close_matches`,
one pair's diff for `similarity_ratio`: under one shared clock; `TimeoutError` on
expiry names the elapsed cost and the deadline, `None` (the default) is unbounded, and
an enormous-but-finite budget saturates to unbounded rather than erroring.

```python
tors.similarity_ratio("kitten", "sitting")
# 0.6153846153846154
tors.get_close_matches("appel", ["ape", "apple", "peach", "puppy"])
# ['apple', 'ape']
```

## `tors.levenshtein` / `tors.jaro` / `tors.jaro_winkler`

```python
def levenshtein(a: str, b: str, *, deadline_ms: float | None = None) -> int: ...
def jaro(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...
def jaro_winkler(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...
```

The classic edit-distance/similarity metrics CPython has no stdlib spelling of
(`difflib`'s ratio is not a metric: see above; every real spelling is third-party).
All three operate on character sequences (Rust `char`s, i.e. Unicode scalar values:
the same unit a Python `str` index addresses), so an emoji or a combining-mark
sequence costs what its codepoint count says, not its UTF-8 byte count.

`levenshtein(a, b)` is the unit-cost edit distance (insert/delete/substitute each cost
1) as an `int`: `levenshtein("kitten", "sitting") == 3`. Symmetric; `("", "")` is
`0`; one empty operand is the other's character count. Implemented
as a two-row DP (O(len(b)) space, not a full O(n·m) matrix), so memory stays
linear in operand size even on adversarial multi-megabyte inputs; identical
operands short-circuit to `0` via a single equality scan, without entering the DP at
all.

`jaro(a, b)` is the Jaro similarity in `[0.0, 1.0]` (higher is more similar): a
flag-based matching window of `max(|a|, |b|)//2 - 1`, transpositions counted over the
matched subsequences, `(m/|a| + m/|b| + (m - t)/m) / 3`. `("", "")` is `1.0`; one
empty operand is `0.0`. `jaro_winkler(a, b)` is `jaro(a, b)` plus a common-prefix
boost: `jaro + l * 0.1 * (1 - jaro)`: applied only when the Jaro score is `> 0.7`,
with `l` the common-prefix length capped at 4 characters; the `0.1` scale, the `4`-char
cap, and the `> 0.7` threshold are the standard convention from the original papers
(also `strsim`'s exact spelling, used as the differential oracle for all three
functions). Known literature vectors: `jaro("MARTHA", "MARHTA") == 17/18 ≈ 0.944`,
`jaro_winkler("MARTHA", "MARHTA") ≈ 0.961`.

All three are O(n·m) worst case; `deadline_ms` bounds the DP/matching pass with a
check once per row (Levenshtein) or per phase/1024 outer steps (Jaro), the same
DoS-discipline shape as `diff_opcodes`'s: `TimeoutError` on expiry naming the elapsed
cost and the deadline, `None` (the default) unbounded, an enormous-but-finite budget
saturating to unbounded.

```python
tors.levenshtein("kitten", "sitting")
# 3
tors.jaro_winkler("MARTHA", "MARHTA")
# 0.9611111111111111
```

## `tors.quote` / `tors.quote_plus` / `tors.unquote` / `tors.unquote_plus`

```python
def quote(text: str, safe: str = "/") -> str: ...
def quote_plus(text: str, safe: str = "") -> str: ...
def unquote(text: str) -> str: ...
def unquote_plus(text: str) -> str: ...
```

`urllib.parse.quote` / `quote_plus` / `unquote` / `unquote_plus`, byte-exact, for `str`
input (the stdlib's `bytes`-in/`encoding=`/`errors=` overloads are out of scope: the
encode side is always strict UTF-8, the decode side always `errors="replace"`). The
stdlib's own spellings are pure Python: a GIL-held regex/loop over the whole string;
these are one `py.detach`ed native pass each.

`quote(text, safe="/")` never percent-encodes ASCII letters, digits, or `_.-~` (the RFC
3986 unreserved set, the stdlib's `_ALWAYS_SAFE`), plus the ASCII members of `safe`;
every other byte of `text`'s UTF-8 encoding becomes `%XX` with uppercase hex. `safe` is
byte-level and ASCII-only, exactly like the stdlib's own `safe.encode("ascii", "ignore")`
normalization: a non-ASCII `safe` member is silently dropped (`tors.quote("é", "é") ==
"%C3%A9"`, not `"é"`), and `%` in `safe` is honored like any other byte (stays literal).
`quote_plus(text, safe="")` is not "quote, then replace `%20` with `+`": the stdlib
quotes with `" "` appended to `safe` (so a space never encodes at all) and then replaces
every `" "` with `"+"`; the observable difference is a literal `+` in `text`, which
escapes to `%2B` unless the caller puts `+` in `safe` (`tors.quote_plus("a+b") ==
"a%2Bb"`).

`unquote(text)` decodes `%XX` (either hex case) as UTF-8 with `errors="replace"`
(an invalid sequence becomes one or more U+FFFD, CPython's maximal-subpart rule); a `%`
not followed by two hex digits (`%zz`, a trailing `%`, `%e` at end of input) stays
verbatim. It also reproduces the stdlib's `_asciire` fragmentation exactly: each maximal
ASCII run is unquoted and UTF-8-decoded independently, with non-ASCII segments passed
through verbatim: so a multi-byte escape interrupted by a non-ASCII character is not an
escape at all (`tors.unquote("%Cé3") == "%Cé3"`), and an escape split across an
ASCII/non-ASCII boundary decodes as two independently-replaced fragments
(`tors.unquote("%C3é%A9") == "�é�"`, not `"é"`). `unquote_plus(text)` replaces
every `+` with a space before unquoting (order is semantics, not an
implementation detail), so an escaped `%2B` survives as a literal `+` while a raw `+` becomes a space
(`tors.unquote_plus("%2B") == "+"`, `tors.unquote_plus("+") == " "`).

On text the call leaves byte-for-byte unchanged, all four return the
original input object: `quote`/`quote_plus` when no byte needs encoding, `unquote` when `text` has no
`%`, `unquote_plus` when `text` has neither `+` nor `%`: CPython's own fast-path idiom,
zero allocation, zero copy, zero marshalling. `unquote`'s borrow is narrower than a plain
equality check would suggest: an input whose every `%` is invalid hex (`"%zz"`) decodes
to an equal-but-new string in the stdlib, not the original object, and `tors.unquote`
matches that residue exactly rather than widening the identity lane past parity.

**Known deviation: lone (unpaired) surrogates.** pyo3's `str` argument extraction
(`PyUnicode_AsUTF8AndSize`) requires the whole input to be valid UTF-8 up front, which a
lone surrogate codepoint never is. `quote`/`quote_plus` are unaffected in practice: the
stdlib's own encode-based implementation raises the identical `UnicodeEncodeError` for
such input, so the two still agree, but `unquote`/`unquote_plus` diverge: CPython's
`unquote` only UTF-8-encodes the ASCII runs it is about to percent-decode and passes
every non-ASCII character (surrogates included) through untouched, so it never raises
for a lone surrogate anywhere in `text`. `tors.unquote`/`tors.unquote_plus` raise
`UnicodeEncodeError` for any `text` containing a lone surrogate, even one nowhere near a
`%` escape, because the whole-string extraction fails before the Rust core ever runs.
This is a real, narrow parity gap (native Rust `&str`/`String` cannot represent an
unpaired surrogate at all, so silently downgrading to `errors="replace"`-style
substitution at the boundary would corrupt the character rather than reproduce it):
callers who round-trip `surrogateescape`-decoded text (e.g. from `os.fsdecode`) through
`unquote` should be aware of it.

```python
tors.quote("café/data", "/")  # "caf%C3%A9/data"
tors.quote_plus("a b+c")  # "a+b%2Bc"
tors.unquote("caf%C3%A9%20data")  # "café data"
tors.unquote_plus("a+b%2Bc")  # "a b+c"
```

## `tors.chunk_text`

```python
def chunk_text(
    text: str,
    max_chars: int,
    *,
    overlap: int = 0,
    boundary: Literal["word", "sentence"] = "word",
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_text(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

The context-window/RAG packing primitive: boundary-aware chunking of `text` into
`(start, end)` pairs in Python `str` index (codepoint) units, each chunk at most
`max_chars` codepoints, cut at word or sentence boundaries wherever the budget
allows: `truncate_to_bounds`'s own cut rule (the largest boundary end within the
budget, a grapheme-safe hard cut when a single word/sentence exceeds it) applied
repeatedly, with the whole-text bounds computed once, GIL-released. Every cut is also
grapheme-cluster-safe, so `max_chars` can be exceeded only in the pathological case of
a single grapheme cluster (e.g. an oversized ZWJ emoji chain) wider than the remaining
budget: a covering chunker cannot drop content, so that one chunk goes out past
`max_chars` rather than split the cluster; ordinary text never hits this.

**`overlap=0`** (the default): the original lossless-partition contract, unchanged;
chunks are non-empty, contiguous, strictly increasing, cover `[0, len(text))`, and
joining the slices reproduces the input exactly.

**`overlap > 0`**: each chunk after the first starts `overlap` codepoints before the
previous chunk's end, SNAPPED to the nearest `boundary`, never mid-word or
mid-sentence, so a fact split across a cut is still whole in at least one chunk (the
RAG-retrieval shape). This trades the lossless-join guarantee for genuine shared
content between consecutive chunks; every chunk's own `<= max_chars` and
boundary-safety invariants still hold regardless. `overlap` must be `< max_chars`
(`ValueError` otherwise: an overlap at least as large as the budget means no forward
progress is possible). A chunk shorter than the requested `overlap` silently
degrades to zero overlap for just that one transition rather than stall or violate
the budget, a documented degradation under the one invariant that never breaks:
forward progress (the chunk count can never exceed the codepoint count). The
overlap is declined the same way — zero overlap for that one transition — when
taking it would not buy new context: if the re-cut from the snapped start would
land a span strictly inside the previous chunk (the same text embedded twice,
the failure mode #83 fixed), the next chunk starts at the previous chunk's end
instead, so a chunk is never contained in its predecessor.

`max_chars < 1` or `overlap < 0` raise `ValueError`; an unrecognized `boundary` raises
`ValueError` (the `truncate_to_bounds` spelling). `chunk_cdc`'s byte-level sibling:
`chunk_text` is the semantic/embedding-pipeline chunker (word/sentence-aware, sized
for a context window), `chunk_cdc` is the storage/sync chunker (content-defined byte
boundaries, sized for dedup). Both compose naturally with `merkle_root`/`merkle_diff`
for integrity-checked chunks on top of either strategy.

The return marshalling is `O(chunks)` 2-tuples of ints: the `word_bounds` list-shape
class, at chunk-count scale rather than segment-count scale.

```python
tors.chunk_text("cats are cute and cats are fun", 12)
# [(0, 8), (8, 17), (17, 26), (26, 30)]
tors.chunk_text("cats are cute and cats are fun", 12, overlap=3)
# [(0, 8), (5, 17), (14, 26), (23, 30)]
```

## `tors.chunk_text_iter`

```python
def chunk_text_iter(
    text: str, max_chars: int, *, overlap: int = 0, boundary: Literal["word", "sentence"] = "word"
) -> Iterator[tuple[int, int]]: ...
```

`chunk_text`'s streaming twin, the `word_bounds`/`word_bounds_iter` shape: the whole
scan runs once under `py.detach` at iterator construction, and each `__next__` holds
the GIL only to build one 2-tuple, rather than marshalling the whole result into a
`list` under one GIL hold. Same sequence, same argument contract as `chunk_text`;
prefer it over the list API once a document chunks into the hundreds of
thousands of pieces, where the marshalling cost dominates.

The argument contract includes error precedence, stated here once for the whole
chunking family (every list/`_iter` pair: `chunk_text`, `chunk_by_words`,
`chunk_by_sentences`, `chunk_by_paragraphs`, `chunk_by_lines`, and each one's `_iter`
twin): a `text` that fails UTF-8 conversion (a lone-surrogate `str`, buildable in
CPython, impossible in UTF-8) raises `UnicodeEncodeError` before any
count/overlap `ValueError`, identically in both spellings. The list functions take
`text` as an already-converted argument, so the conversion error always fires first
there; the `_iter` twins borrow the text before validating counts, so they raise the
same error on the same call: the two spellings of a function never disagree on
which error type a bad call raises (the one message-level divergence, a lone
surrogate in both `text` and `boundary`, is frozen by test: both spellings raise
`UnicodeEncodeError`, the list reporting the text's surrogate and the iter the
boundary's, pyo3 extracting `boundary` ahead of the iter body but after the list's
`text` extraction). For `chunk_text`/`chunk_text_iter` the
unrecognized-`boundary` `ValueError` comes last of all, after the count/overlap
checks, in both spellings.

```python
list(tors.chunk_text_iter("cats are cute and cats are fun", 12))
# [(0, 8), (8, 17), (17, 26), (26, 30)]
```

## `tors.chunk_by_words`

```python
def chunk_by_words(
    text: str, words_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_by_words(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

The unit-count twin of `chunk_text`: instead of a character budget, each chunk spans
exactly `words_per_chunk` consecutive word tokens, not `word_bounds`'
raw segment count. `word_bounds` follows UAX #29 exactly, which gives an inter-word
space run its own segment (`"one two"` is three segments: `"one"`, `" "`, `"two"`, the
same convention `word_count` already carries): grouping *raw* segments here would
silently mean "`words_per_chunk` roughly halved" for ordinary space-separated prose,
the opposite of what a caller reaching for `words_per_chunk=100` (a "~100 word chunk"
for an embedding budget) wants. So this filters to segments carrying at least one
non-whitespace codepoint first, and only then windows over what remains: a "word" is
a real token, and whitespace between two tokens *inside* one chunk still rides along
naturally (each chunk's span is a contiguous slice of the original text between two
real absolute offsets). `(start, end)` codepoint offsets span the first included word
token's start through the last included token's end, not through any trailing
whitespace after it, so unlike `chunk_text`'s covering-partition contract,
non-overlapping chunks here are not necessarily contiguous. The final chunk may hold
fewer than `words_per_chunk` tokens when the total doesn't divide evenly. `overlap`
words repeat at the start of the next chunk. Empty text, or text with no word tokens
at all, returns `[]`.

`words_per_chunk < 1` or `overlap < 0` raise `ValueError`; `overlap >= words_per_chunk`
raises `ValueError`: the chunk stride is `words_per_chunk - overlap` tokens, and
unlike `chunk_text`'s character-granularity overlap this stride is always `>= 1` by
construction once validated, so forward progress needs no runtime fallback.

Cost at document scale: one `word_bounds` walk, one grapheme boundary index (a
one-bit-per-codepoint bitmap; on pure-ASCII text it is two SIMD byte scans, no
segmentation walk, shared with `chunk_hierarchical` and `chunk_by_sentences`), a
zero-copy merge fast path when no boundary needs it, and one streaming decode pass
for the token filter. Measured on 12 MiB of prose (min-of-3, `tools/bench_chunking.py`):
~160 ms, ~90 MiB transient (the word-bounds list itself). Before this change it built a `HashSet` of every grapheme boundary
plus a whole-text `Vec<char>` unconditionally: ~1.9 s and ~500 MiB on the same input.

```python
tors.chunk_by_words("one two three four five six seven", 3)
# [(0, 13), (14, 27), (28, 33)]
tors.chunk_by_words("one two three four five six seven", 3, overlap=1)
# [(0, 13), (8, 23), (19, 33)]
```

## `tors.chunk_by_words_iter`

```python
def chunk_by_words_iter(
    text: str, words_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...
```

`chunk_by_words`' streaming twin, the same `chunk_text_iter` shape: one detached
whole-text pass at construction, one 2-tuple per `__next__`, identical sequence to
the list API.

```python
list(tors.chunk_by_words_iter("one two three four five six seven", 3))
# [(0, 13), (14, 27), (28, 33)]
```

## `tors.chunk_by_sentences`

```python
def chunk_by_sentences(
    text: str, sentences_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_by_sentences(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

`chunk_by_words`' sentence-count twin (`sentence_bounds`'s UAX #29 segmenter): each
chunk spans `sentences_per_chunk` consecutive sentence segments, `overlap` sentences
repeated. Same argument contract, same empty-input answer, same
forward-progress-by-construction guarantee as `chunk_by_words`.

```python
tors.chunk_by_sentences("One. Two. Three. Four. Five.", 2)
# [(0, 10), (10, 23), (23, 28)]
```

## `tors.chunk_by_sentences_iter`

```python
def chunk_by_sentences_iter(
    text: str, sentences_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...
```

`chunk_by_sentences`' streaming twin, the same `chunk_text_iter` shape.

```python
list(tors.chunk_by_sentences_iter("One. Two. Three. Four. Five.", 2))
# [(0, 10), (10, 23), (23, 28)]
```

## `tors.chunk_by_paragraphs`

```python
def chunk_by_paragraphs(
    text: str, paragraphs_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_by_paragraphs(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

`chunk_by_words`/`chunk_by_sentences`'s paragraph-count twin: each chunk spans
`paragraphs_per_chunk` consecutive paragraphs, `overlap` paragraphs repeated. A
paragraph boundary here is a run of 2+ consecutive newlines (`\r\n` counts as one
unit, matching `tors.normalize`'s own CR/CRLF folding): the same "2+ newlines
survive as the paragraph gap" convention `normalize`'s own pipeline already uses (it
collapses 3+ down to exactly 2, never below). **This is a heuristic, not a Unicode
Standard segmentation** (there is no UAX for paragraphs, unlike UAX #29 for
words/sentences): a single `\n` is ordinary content, not a break. A
leading or trailing blank-line run is trimmed rather than emitted as an empty
paragraph; text with no qualifying run at all is one paragraph. Unlike the
word/line twins, paragraphs have no content filter here: a whitespace-only
paragraph is emitted as a chunk (only fully-empty spans are dropped), so an
overlapping pair of chunks can share blank content. Same argument contract,
same empty-input answer, same forward-progress-by-construction guarantee
as `chunk_by_words`/`chunk_by_sentences`.
No retrieval or LLM-quality claim is made for any chunking strategy in this family:
tors guarantees the mechanical contract (correct boundaries, genuine overlap, the
right knobs).

```python
tors.chunk_by_paragraphs(
    "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.", 2
)
# [(0, 45), (47, 68)]
```

## `tors.chunk_by_paragraphs_iter`

```python
def chunk_by_paragraphs_iter(
    text: str, paragraphs_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...
```

`chunk_by_paragraphs`' streaming twin, the same `chunk_text_iter` shape: one
detached whole-text pass at construction, one 2-tuple per `__next__`, identical
sequence to the list API, same argument contract (the family-wide error-precedence
paragraph in `chunk_text_iter`'s section included). Like every `_iter` spelling it
has no async twin (an iterator is not an awaitable shape; see [Async use](async.md)). It exists for the same reason as the other `_iter` twins: the
list shape's GIL-held marshalling cost is measured for segment-count-heavy outputs
(`word_bounds` on 12 MiB of prose, 3.67M segments, holds the GIL for 328–344 ms
just marshalling the list; see [Performance](performance.md)), and a paragraph-heavy
corpus (a multi-MiB article dump or report batch, one blank line per record) is in
that piece-count class, chunking into hundreds of thousands of pieces.

```python
minutes = (
    "Attendees: Ada, Grace, Edsger.\n\n"
    "Grace: parser rewrite halves latency.\n\n"
    "Edsger: spec drift question, unresolved.\n\n"
    "Next sync moves to Thursday."
)

list(tors.chunk_by_paragraphs_iter(minutes, 2))
# [(0, 69), (71, 141)]: 4 paragraphs, 2 chunks of exactly 2; the blank-line
# gap between chunks belongs to neither (chunk 2 starts at "Edsger")
list(tors.chunk_by_paragraphs_iter(minutes, 2, overlap=1))
# [(0, 69), (32, 111), (71, 141)]: overlap repeats whole paragraphs;
# (32, 111) is "Grace: parser rewrite halves latency.\n\nEdsger: spec drift
# question, unresolved."
```

## `tors.chunk_by_lines`

```python
def chunk_by_lines(
    text: str, lines_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_by_lines(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

`chunk_by_words`/`chunk_by_sentences`/`chunk_by_paragraphs`'s line-count twin: each
chunk spans `lines_per_chunk` consecutive lines, `overlap` lines repeated at the start
of the next chunk. A line break is a `\n`, a lone `\r`, or a `\r\n` pair counted as
one unit (the same CR/CRLF folding convention `chunk_by_paragraphs` and `normalize`'s
own pipeline use; `str.splitlines`' exotic separators (`\v`, `\f`, NEL, LS, PS)
are not breaks here). A line counts as a line only when it carries at least one
non-whitespace codepoint, the same real-token discipline `chunk_by_words` applies to
word segments: blank lines neither count toward `lines_per_chunk` nor split a chunk's
interior; they ride along inside a chunk's span exactly as inter-word whitespace
rides along in `chunk_by_words`, so `lines_per_chunk=200` means 200 content lines.
"Non-whitespace" is definitional here: the Unicode `White_Space` property
(`char::is_whitespace`), under which an NBSP-only line is blank and
U+001C–U+001F (FS/GS/RS/US) count as line content, diverging from
Python's `str.isspace()` (which treats those four as whitespace) and from
`str.splitlines` (which even breaks on them; tors does not).
`(start, end)` codepoint offsets span the first included line's start through the
last included line's end (not through the trailing break after it, so unlike
`chunk_text`'s covering-partition contract, non-overlapping chunks here are not
necessarily contiguous); a trailing break at end of text yields no trailing empty
line. The final chunk may hold fewer lines when the total doesn't divide evenly.
Empty text, or text with no content lines at all, returns `[]`. Same argument
contract, same empty-input answer, same forward-progress-by-construction guarantee
as its siblings.

The line-oriented-text shape this exists for: one message per line (a chat thread),
one record per line (a log), one cue per block. Cost at document scale: one
byte-level walk: `memchr2` hops between break bytes behind an 8-byte inline
density window before each hop (so dense break runs never pay a hop), the
real-line whitespace filter folded into the same pass, a per-codepoint fallback
for non-ASCII segments, and ASCII certification batched as one 4 KiB stride per
~50 segments (a sliding certificate over the whole scan, not a per-segment
`is_ascii` check). The structure is a trade against the simpler whole-text
`is_ascii` gate a sparse scan could use (interleaved best-of-N, 12 MiB):
+0.1-1.0 ms on pure-ASCII densities (`chunk_by_lines` ~0.45 ms on prose and
~2.2 ms on a one-line-per-~80-bytes log against the gate's ~0.4 and ~1.2-1.3 ms;
`chunk_by_paragraphs` ~1.5 vs ~1.3 ms on that log) and ~1-3 ms on a CJK-dense log
(~21 vs ~20 ms by-lines, every segment taking the fallback; the by-paragraphs
cell pays the most, ~12 vs ~9 ms, and wobbles ~10-12.5 ms with binary code layout
across rebuilds). What it buys: ~4x on break soup (~10 vs ~42 ms; the gate has no
density guard), ~16x on mixed text (~0.46 vs ~7.6 ms; one non-ASCII byte no longer
forfeits the document to a per-codepoint decoder), and the same wins on
`chunk_by_paragraphs`' soup cell (~22 vs ~54 ms). Outputs are differential-pinned
identical across all of these shapes; O(text) time, O(1) memory beyond the output,
no segmentation walk and no grapheme boundary
index at all, unlike `chunk_by_words`/`chunk_by_sentences`: every split lands
strictly between a break character and adjacent content, so the split point is
structurally grapheme-safe with nothing to check.

```python
log = "INFO boot\nINFO ready\n\nWARN disk at 90%\nERROR io failure\nINFO retry ok\n\nINFO shutdown"

tors.chunk_by_lines(log, 2)
# [(0, 20), (22, 55), (56, 84)]: 6 content lines, 3 chunks of exactly 2; the
# blank between chunks 1 and 2 falls in the gap (chunk 2 starts at "WARN"),
# while the blank inside the final chunk rides along, never counted
tors.chunk_by_lines(log, 2, overlap=1)
# [(0, 20), (10, 38), (22, 55), (39, 69), (56, 84)]: overlap repeats whole
# lines; (10, 38) is "INFO ready\n\nWARN disk at 90%", blank riding along
```

## `tors.chunk_by_lines_iter`

```python
def chunk_by_lines_iter(
    text: str, lines_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...
```

`chunk_by_lines`' streaming twin, the same `chunk_text_iter` shape: one detached
whole-text pass at construction, one 2-tuple per `__next__`, identical sequence to
the list API. Like every `_iter` spelling it has no async twin (an iterator is not
an awaitable shape; see [Async use](async.md)). It exists for the
same reason as the other `_iter` twins: the list shape's GIL-held marshalling
cost is measured for segment-count-heavy outputs (`word_bounds` on 12 MiB of
prose, 3.67M segments, holds the GIL for 328–344 ms just marshalling the list;
see [Performance](performance.md)), and a line-oriented corpus (a multi-MiB log or
transcript) is in that piece-count class, chunking into hundreds of thousands
of pieces.

```python
list(tors.chunk_by_lines_iter(log, 2))
# [(0, 20), (22, 55), (56, 84)]
```

## `tors.chunk_hierarchical`

```python
def chunk_hierarchical(
    text: str,
    max_chars: int,
    separators: Sequence[str | None] | None = None,
    *,
    overlap: int = 0,
    overlap_boundary: Literal["grapheme", "word"] = "grapheme",
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_hierarchical(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Priority-ordered fallback chunking: the `chunk_text`/`chunk_by_*` family's
fourth shape, and the pattern LangChain's `RecursiveCharacterTextSplitter`
popularized: a list of separator levels, coarsest first, tried in order for
each chunk, falling back to the next level only when the coarser one has no
in-budget cut over the current window.

`separators=None` (the default) uses tors's own accurate hierarchy:
heading → paragraph → sentence → word → a grapheme-safe raw cut, always
the final, unconditional fallback (this never fails to produce a chunk);
it reuses the same UAX #29 segmenters `chunk_by_sentences`/
`chunk_by_words` do, rather than LangChain's own naive literal guesses
(`"\n\n"`, `". "`, `" "`).

The heading level (#63) is what makes this structure-aware for markdown:
its cuts sit at ATX heading lines, so **a section's content never merges
across a heading of higher rank**. Each cut lands *before* the heading —
the heading itself rides with the section that follows it (the chunk that
starts at a heading includes the heading text), and the newline run
between a section's content and the next heading is dropped between
chunks, the same convention the paragraph level applies to blank-line
runs (and when a heading follows a blank-line run, its cut *is* that
paragraph gap's cut: the two levels agree exactly where both can cut;
the heading level's addition is the single-newline case the paragraph
level cannot see). A heading-bearing document under a whole-document
budget comes back as its sections, not one giant chunk: the
final-chunk-runs-untrimmed exception is a size concession, and the
heading level's structural contract outranks it. The level's scope is
deliberately ATX-only for v1, verified against the markdown the
`tors.documents` engines actually emit — pdf_oxide's
`StructType::markdown_prefix` writes `"# "`…`"###### "`, anydoc's
markdown renderer writes `"#".repeat(level) + " "`, and
html-to-markdown-rs's `HeadingStyle::default()` is `Atx` — so setext
underlines (`===`/`---` runs under a paragraph line) are excluded, as are
every line that is not shaped `1-3 spaces, 1-6 '#', space/tab/EOL`:
`#no-space`, a 7-hash run, an escaped `\#`, a blockquote's `> # x`, a
list item's `- # x`, 4-space indented code, and any mid-line hash are
ordinary content, and a heading-shaped line inside a fenced code block
(``` fences, CommonMark §4.5's own state machine, shared with
`extract_code_blocks`) is code. The level is gated: text with no `#` byte
anywhere never realizes it (a one-pass `memchr` probe is the only cost
heading-free input adds), so heading-free text chunks exactly as the
pre-#63 hierarchy did. Precedence is budget > heading > paragraph >
sentence > word: the heading level bounds sections, the budget still
bounds oversized sections (split at the finer levels), and the
grapheme-safe raw cut stays the unconditional last fallback.

`separators=[...]` is a caller-supplied sequence (a list or a tuple) of
literal strings, not regex
(a documented scope line: literals are LangChain's own default
too, cover the motivating markdown-header case completely, and avoid
reopening the regex-semantics question `re` support was already declined
over), coarsest first, e.g. `["\n## ", "\n\n", ". ", " "]` for
markdown-header-aware chunking. Any `Sequence` of literals and `None` entries
is accepted: `("\n", None)` behaves identically to `["\n", None]`, while
`str`, `dict`, `set`, and other non-`Sequence` inputs (generators included)
raise `TypeError` at argument extraction. A custom sequence replaces the default
hierarchy for the levels it specifies, but the grapheme-safe raw cut is
still always appended as the final fallback regardless; unlike LangChain,
no trailing `""` sentinel is required (one is accepted and ignored if
supplied).

An entry in that sequence may also be `None`: it splices the default
hierarchy's accurate levels in at that position, the mix an
all-literal list could not express before. `["\n", None]` is
line → heading → paragraph → sentence → word → raw cut, the
line-oriented-text shape (a chat thread, one message per line, never split
mid-line) whose oversized-line fallback is the real UAX #29 sentence/word
segmenter rather than the `". "`/`" "` literal guesses an all-literal
`["\n", ". ", " "]` pins it to: a `". "` match after `"U.S."` is not a
sentence boundary, and the naive list severs `"U.S. team"` where the
spliced hierarchy does not. `[None]` is identical to `separators=None`.
The reserved literal `"heading"` is the heading level itself — the
explicit opt-in for custom hierarchies: `["heading", None]` is heading →
paragraph → sentence → word → raw cut, the markdown-RAG shape (sections
bounded by headings, oversized sections split at accurate sentence/word
boundaries); the word is reserved, so a literal split on `"heading"` is
not expressible (spell it in different case if you truly need the word).
Cost: every level (each of the default walks, each distinct custom
literal) pays its one whole-text walk at most once per call, and only when a
window consults it: levels are built at their first consultation (the window
loop walks the list strictly through `find_map`, in priority order), so a
budget that answers every window at the paragraph level never runs the
sentence or word walks at all, and a `["\n", None]` thread whose every line
fits the budget builds none of the spliced levels. Duplicate entries, `None`
or a repeated literal, are recognized at slot construction and skipped, inert
(identical levels can never change the answer; the first occurrence of a level
always dominates its duplicate), so `[None] * 100` costs what `[None]` does
(~0.5 ms at a 2000-codepoint budget over 6 MiB of prose, the paragraph walk
alone) and `[" "] * 100` what `[" "]` does (~9 ms). Two former spellings paid
more: before the dedup, every duplicate entry re-paid the walks plus ~45 MiB
of cut vectors per duplicate on a 6 MiB document, a caller-controlled
unbounded cost (`[None] * 100` measured 17.2 s and +3,120 MiB of peak RSS,
`[" "] * 100` 790 ms and +1,560 MiB, OOM shapes); and between the dedup and
the lazy levels the walks were paid once per call even when no window ever
consulted them, a single-chunk budget over 6 MiB spending ~176 ms building
levels that supplied zero cuts. Both closed, by the construction-time dedup
and the first-consultation deferral.

**Rust API note**: 0.6.0 changes the public `tors-core` crate's
`chunk_hierarchical(text, max_chars, separators, overlap)` signature:
`separators` moved from `Option<&[&str]>` to `Option<&[Option<&str>]>`,
the type the `None`-entry splice requires, which is breaking for direct
Rust consumers of the separately-published crate at 0.x. Python callers
are unaffected: an all-literal list behaves identically under either
spelling. The merge that introduced it (#28) landed as a plain `feat:`
with no `BREAKING CHANGE:` footer, which release-please would not have
surfaced on its own, so the footer is restated on the follow-up fix
commit (#31), which release-please carried into the 0.6.0 changelog
(released 2026-09-10); this note is the docs-side record.

**Unlike `chunk_text`, this is not a lossless covering partition**: at
every level except the raw cut, the separator itself is dropped between
chunks: the chunk ends where the separator starts, the next chunk begins
where it ends, the same convention `chunk_by_paragraphs` already applies
to blank-line runs. A caller splitting on a marker wants it gone, not
duplicated. Since #103 that holds at the end of the document too: the
final-chunk shortcut (the whole remainder fits the budget, so the chunk
runs to the end untrimmed) answers the separator-skip question first, so
a window that OPENS on a separator match is never emitted — not even as
the untrimmed final chunk. A trailing separator run therefore survives
only as a suffix of a content-bearing chunk, and an all-separator
document chunks to zero chunks.

`overlap` snaps the next chunk's start backward to the nearest GRAPHEME
boundary at or before the target, not necessarily a semantic
paragraph/sentence/word boundary the way `chunk_text_overlapping`'s
single-hierarchy overlap snap is (a documented simplification of the
general multi-level case). A target at or before the chunk's own start
silently degrades to zero overlap for just that one transition, the same
snap-collapse `chunk_text` already applies — and so does an overlap whose
re-cut would land the next chunk strictly inside its predecessor (the same
text twice, no new context): the transition falls back to the zero-overlap
cut instead, so ends always strictly advance.

`overlap_boundary="word"` opts the snap into word-aware tails (#47) for
exactly the embedding-pipeline shape a mid-word tail start is a rough
edge for. The composition order is: grapheme snap, then word snap, then
the decline-the-snap lookahead. The grapheme candidate lands first (never
mid-cluster), then the snap moves further back to the nearest UAX #29
word boundary at or before it — the same word-bounds level the default
hierarchy already builds: realized lazily at the first snap that consults
it, shared with any window that descends to it (never a second walk), and
for a hierarchy with no word level at all (an all-literal custom list)
the same one-off word-bounds level is built at the first snap instead. A
mid-word tail therefore starts at its word's first codepoint; when the
word level has no boundary in the snap-back range (dense CJK runs, Thai
without a dictionary, one long token) the plain grapheme candidate is
kept, and at `overlap=0` the mode is accepted and is a no-op (no snap
site ever runs). The word-snapped candidate then goes through the
decline-the-snap lookahead unchanged — it may reach back to or past the
previous chunk's start, in which case the transition degrades to zero
overlap exactly as a grapheme candidate reaching that far would: no chunk
is ever contained in (or duplicated across) its predecessor. Unknown
values raise `ValueError` naming the closed set `('grapheme', 'word')`,
unconditionally at the argument boundary (an irrelevant knob never errors
late).

`max_chars < 1` or `overlap < 0` raise `ValueError`; `overlap >= max_chars`
raises `ValueError`. Empty `text` returns `[]`. An empty `separators` sequence
is legal and skips straight to the raw-cut fallback for every chunk. Every
level's cut candidates are additionally grapheme-cluster-safe (the same
Thai SARA AM / combining-mark fix applied crate-wide), including custom
literal separators.

Cost at document scale: one scan per consulted level (each level's walk or
literal search runs at most once per call, at the first window that consults
it: the default hierarchy's paragraph/sentence/word walks, or one literal
search per distinct custom separator), so levels no window descends to are
never scanned at all: a custom hierarchy that never matches under a
whole-document budget is one codepoint count and nothing else, so not even
the literal's own scan runs. Since #103 the final-chunk exit answers one
extra question — could a separator match open at the exit's start — and a
cheap necessary-condition pre-test over the unrealized level specs keeps
that answer free whenever no separator could possibly open there (a text
that does not begin with one of the literal separators, under the default
hierarchy always: a paragraph-gap cut cannot begin at codepoint 0), so the
whole-document cells below keep their zero-build exit; a text that DOES
open with a separator match pays the hierarchy's first search there (the
skip question needs the level), once, memoized. Since #63 the default
hierarchy adds one more lazily-paid pass: the heading level's `#`-byte
gate probe (~0.12 ms at 12 MiB, `memchr`, once per call at the first
heading-site consultation — a level realization, the scan itself, or a
demotion check), which heading-free text answers "no heading can exist"
without ever building the level. The one other whole-text
structure is the grapheme
boundary index, a one-bit-per-codepoint bitmap built lazily, only when a
realized level has cuts to filter, a window needs the raw-cut fallback, or
`overlap` snaps; on pure-ASCII text the index is two SIMD byte scans instead
of a segmentation walk, and a call that realizes no level with cuts and never
falls back or snaps builds none of it. Measured on 12 MiB prose (min-of-3,
`tests/test_performance.py` and `tools/bench_chunking.py`): a whole-document
budget (`max_chars` at or above the text's length) consults no level at all
and costs ~0.1 ms whatever the hierarchy, default, custom, or `None`-spliced
(formerly ~340 ms for the default hierarchy); a 2000-codepoint budget whose
windows are all served by the paragraph level costs ~1.2 ms (formerly ~350 ms:
the eager build paid the sentence and word walks no window consulted); budgets
that genuinely descend pay the walks they use, ~190 ms at a 600-codepoint
budget (paragraph plus sentence), ~400 ms at 100 (all three levels plus a 20x
denser chunk loop: the word walk ~130 ms plus sentence walk ~190 ms, the
accurate UAX #29 segmentation the function exists to provide). The laziness
prices consultation, it does not skip walks the answer needs. Before #22 an
unconditional `Vec<char>` collect plus a `HashSet` of every grapheme boundary
in the document ran before anything else (~1.0-1.2 s for the never-matching
case regardless of budget, ~3.0 s for the default hierarchy, superlinear in
input size), and until the lazy levels every level was built before the first
window, consulted or not.

No retrieval or LLM-quality claim is made for any chunking strategy in
this family: tors guarantees the mechanical contract (correct boundaries,
genuine overlap, the right knobs). This
is additive: `chunk_text`/`chunk_by_words`/`chunk_by_sentences`/
`chunk_by_paragraphs`/`chunk_by_lines` remain the right choice for the
common case; `chunk_hierarchical` is for custom, format-aware, or
multi-granularity needs those simpler functions can't express.

```python
md = "# Title\nSome intro text here.\n## Section\nMore content in this section."
tors.chunk_hierarchical(md, 60, ["\n## ", "\n\n", ". ", " "])
# [(0, 29), (33, 70)]  ->  "# Title\nSome intro text here." / "Section\nMore content in this section."

thread = (
    "Nathan: kicking off the sync.\n"
    "Priya: We briefed the U.S. team on the numbers. "
    "They asked for a follow-up meeting. The budget holds.\n"
    "Nathan: done."
)
tors.chunk_hierarchical(thread, 60, ["\n", None])
# [(0, 29), (30, 78), (78, 131), (132, 145)]
#  -> whole lines where they fit; the oversized Priya line falls to real
#     sentence boundaries ("on the numbers. " / "meeting. The budget holds.")

tors.chunk_hierarchical(thread, 40, ["\n", ". ", " "])[1]
# (30, 55)  -> "Priya: We briefed the U.S"; the naive ". " list severs the name
tors.chunk_hierarchical(thread, 40, ["\n", None])[1]
# (30, 69)  -> "Priya: We briefed the U.S. team on the "; the splice does not

tors.chunk_hierarchical(thread, 24, [None]) == tors.chunk_hierarchical(thread, 24)
# True: [None] is separators=None

md = (
    "# Guide\n\nIntro paragraph for the guide.\n\n"
    "## Setup\n\nBody of the setup section, long enough to matter.\n\n"
    "## Usage\n\nTail section."
)
tors.chunk_hierarchical(md, 10_000)
# [(0, 39), (41, 100), (102, 125)]
#  -> the sections come back separate even under a whole-document budget:
#     each chunk starts at its own heading ("# Guide...", "## Setup...",
#     "## Usage..."); a section wider than max_chars would still split at
#     paragraph/sentence/word levels (budget > heading > paragraph > ...).
tors.chunk_hierarchical(md, 10_000, ["heading", None]) == tors.chunk_hierarchical(md, 10_000)
# True: the "heading" sentinel + splice is the default hierarchy spelled out.
```

## `tors.chunk_cdc`

```python
def chunk_cdc(
    data: bytes, *, min_size: int = 4096, avg_size: int = 16384, max_size: int = 65534
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_cdc(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

FastCDC 2020 content-defined chunking, one GIL-released native pass: `(start, end)`
**byte** spans (not codepoints: unlike every other segmentation function here, this
operates on raw bytes, not text) partitioning `data` exactly: `end` exclusive, the last
span's `end == len(data)`, no gaps or overlaps.

Content-defined chunking's whole point over fixed-size splitting: cut points are chosen
by local content (a rolling hash over a sliding window), not a fixed stride, so a small
edit near the start of `data` only perturbs the 1-2 chunks nearest the edit; every
chunk further away reappears unchanged, just shifted by the edit's byte delta. Fixed-size
chunking has no such property: an insertion reshuffles every boundary after it. This is
the natural upstream of `tors.merkle_root`/`tors.merkle_diff`'s `list[bytes]` argument
for a byte-level dedup/incremental-sync pipeline (chunk, then hash each chunk into the
tree): same framing `tors.finalize`'s normalize-then-hash dedupe gate already
established at the whole-document level, extended to sub-document granularity.

Empty input returns `[]`. Input shorter than `min_size` returns exactly one chunk
covering the whole input (the underlying algorithm's own documented special case).
Deterministic: the same bytes at the same parameters always cut at the same offsets.

`min_size`/`avg_size`/`max_size` must satisfy the wrapped `fastcdc` crate's own
documented bounds: each even; `min_size` in `[64, 1_048_576]`, `avg_size` in `[256,
4_194_304]`, `max_size` in `[1024, 16_777_216]`, and `min_size <= avg_size <=
max_size`, checked and raised as `ValueError` before any chunking runs. The crate itself
only `debug_assert!`s these bounds, a no-op in a release build, so an out-of-range call
would otherwise silently misbehave rather than error: tors validates them itself at the
argument boundary instead, the same discipline `is_grounded`'s `threshold` and
`truncate_to_bounds`' `max_chars` already apply. Defaults are the crate's own documented
example values, not independently chosen.

```python
tors.chunk_cdc(b"hello world " * 10_000)
# [(0, 65534), (65534, 120000)]
```

## `tors.content_hash`

```python
def content_hash(obj: str | int | float | bool | None | list | tuple | dict) -> str: ...
```

The object content hash: the lowercase-hex SHA-256 of the object's
**canonical form**, where the canonical form is EXACTLY

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"))
```

with `json.dumps`'s defaults `ensure_ascii=True` and `allow_nan` — so the
oracle is the stdlib itself, total and always available:

```python
hashlib.sha256(
    json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
```

`tors.content_hash(obj)` returns that string, byte-identical — for
surrogate-free input where `json.dumps` succeeds; where `json.dumps`
rejects (mixed unsortable keys, over-limit ints, circular references) both
sides raise the same exception type, and lone-surrogate `str` input is the
documented exclusion below — pinned
differentially against the oracle over every contract below
(`tests/test_content_hash.py`) and literal-pinned at the byte level
crate-side (`src/canon_impl.rs`). Deterministic by construction: any dict
key order yields the same hash; the one exception is a dict with distinct
NaN keys: NaN never compares equal to NaN, so their output order is
timsort's own rather than a mathematical property (the delegated-sort
paragraph below), and each ordering still hashes exactly as the
`json.dumps` oracle spells it, the parity that matters. An equal-value
`list` and `tuple` hash identically (tuples serialize as lists,
recursively).

**The type contract.** Leaves: `str` (lone-surrogate strings excluded —
they raise `UnicodeEncodeError` where `json.dumps` succeeds, the
documented divergence below), `int` (arbitrary precision), `float`,
`bool`, `None`. Containers: `list`, `tuple`, `dict`. Anything else raises
`TypeError` naming the type (`set`, `frozenset`, `bytes`, `bytearray`,
custom classes, plain `Enum`, views, iterators alike). Circular references
raise `ValueError` on both sides (`json.dumps`'s own marker semantics: a
shared sibling is fine, only a true cycle raises). One scoped exception-type
divergence: *doubly* invalid input — a circular dict that also carries a
non-coercible key — raises `ValueError` from `json.dumps` (its per-pair
check interleaves coercion and cycle detection) and `TypeError` from
`content_hash` (all keys of a dict are coerced before its values are
walked); single-defect inputs raise the same type on both sides.

**Dict keys: coercion, then json's own sort order.** Non-str keys are
coerced exactly as `json.dumps` coerces them: `1` -> `"1"`, `True` ->
`"true"`, `None` -> `"null"`, `1.0` -> `"1.0"`, `nan`/`inf` -> `"NaN"`/
`"Infinity"`. Keys are sorted BEFORE stringification — all-int keys come
out in numeric order (`2` before `10`), where the same digits as str keys
sort lexicographically (`"10"` before `"2"`). The exotic key shapes — any
float key, big-int keys, mixed int/float, NaN keys, and any
str/int-SUBCLASS key (whose overridden rich comparison `json.dumps`'s
sort honors) — are sorted by delegating to CPython's own `list.sort` over
the dict's own `(key, value)` items: the very tuples `json.dumps` sorts,
the same timsort, so the output order matches the running interpreter
byte-for-byte, including the corners no reimplementation would dare: two
distinct NaN objects legally coexist as dict keys, and their output order
is timsort's behavior, not a mathematical property. Mixed unsortable key
types (`{1: ..., "a": ...}`) raise the sort's own `TypeError`,
byte-identical with `json.dumps`'s; non-coercible key types (`tuple`,
`bytes`, ...) raise `TypeError` naming the key type.

**Strings: the `ensure_ascii` escape table.** Inside a string the raw
bytes are exactly printable ASCII (U+0020-U+007E) minus `"` and `\` — the
forward slash is never escaped. Everything else is escaped: `\"`, `\\`,
the five short escapes `\b` `\t` `\n` `\f` `\r` (U+0008/9/A/C/D), `\u00XX`
lowercase for the other controls and for DEL (U+007F is outside printable
ASCII), `\uXXXX` lowercase for all non-ASCII, and astral codepoints as
surrogate-pair escapes: `chr(0x1F600)` -> `"\ud83d\ude00"`. Every
codepoint U+0000-U+007F and the BMP/astral boundary codepoints are pinned
as values and as dict keys.

**Floats: Python's own repr, never reimplemented.** Finite floats are
materialized via Python's float `repr` during the walk (tors does not
reimplement float formatting — exact by construction, `0.1` is `"0.1"`,
`1e16` is `"1e+16"`, `5e-324` is `"5e-324"`, `sys.float_info.max` is its
full 17-digit spelling). Non-finite values use json's `allow_nan`
literals: `NaN`, `Infinity`, `-Infinity`.

**Ints: arbitrary precision.** The i64 fast path covers
`-(2**63)`..`2**63-1` (a storage read plus fixed-buffer decimal digits;
measured 4.8ns per int against 29.3ns for the repr call, ~6x), and
everything beyond falls back to Python's own `int`->`str`, so the
interpreter's `sys.set_int_max_str_digits` limit raises identically on
both sides (`ValueError` on 3.11+; on 3.10, no limit, both sides hash).

**Why the emitter is hand-written.** No maintained crate emits this byte
format: `serde_json` and `orjson` both output raw UTF-8 for non-ASCII
strings with their own escape rules and no `ensure_ascii` mode at all.
The contract here is the stdlib's exact wire format, so the emission is
tors's own ~150 lines (`src/canon_impl.rs`) rather than a dependency that
approximates it. Nothing else is hand-rolled where a crate exists: the
hash is `sha2`, the hex is `const_hex`, and every float/int spelling is
Python's own.

**The surrogate divergence (documented, pinned).** A `str` holding lone
surrogates — value or key, exact or subclass — raises `UnicodeEncodeError`
("surrogates not allowed") from the standard str borrow, the crate-wide
boundary every str-in surface here documents, where `json.dumps` ACCEPTS
lone surrogates (it emits `\udXXX` escapes for them). Real astral text
(valid surrogate pairs) hashes with full parity. The other divergence
lanes: the walk is iterative, so tors accepts nesting deeper than
`json.dumps`, which `RecursionError`s at an interpreter-version-dependent
depth — bounded by the untrusted-input ceiling (`MAX_TOTAL_DEPTH`
150k total frames exact+protocol, `MAX_WALK_NODES` 2M visited objects;
100k exact levels hash, 200k raises `RecursionError`); and mixed
exact+protocol interleavings diverge by construction (json counts every
container against one C budget, tors counts only protocol frames against
the interpreter budget), pinned as a documented divergence, not parity.

**Subclass hooks run to completion under the GIL.** A dict subclass's
`.items()` and a list/tuple subclass's `__iter__` are pulled to completion
before the child walk begins; a hook yielding an unbounded stream aborts
with a generic `ValueError` (the per-container bound is a DoS backstop, its
value deliberately not leaked), bounded never infinite. Treat
`content_hash` as trusted-input-only for subclass instances with
attacker-controlled hooks — the same posture `json.dumps` itself has
(it materializes unboundedly and would spin forever). The same
trusted-input-only posture covers depth/breadth beyond a modest envelope
(depth on the order of 10-20k frames, visited objects on the order of
200-500k nodes): the untrusted-input ceiling (`MAX_TOTAL_DEPTH` 150k total
frames exact+protocol, `MAX_WALK_NODES` 2M visited objects; 100k exact
levels hash, 200k raises `RecursionError`) is test-fitted to the pinned
superset lane, not to an adversarial wall/RSS budget — a 149k-deep exact
tree still holds tens of MiB GIL-held before the ceiling fires.

**Protocol nesting past ~1000 is unsupported.** Subclass containers are
counted against `sys.getrecursionlimit()` and raise `RecursionError` at the
cap. On 3.10/3.11 that matches `json.dumps`'s own boundary; on 3.12+ the
stdlib C-stack budget is looser (a legit 2k-deep subclass chain hashes under
`json.dumps` at ~100k frames on 3.14 while tors raises at ~1000) — a
deliberate conservative divergence, same error class, tighter boundary,
never a silent hash. Exact nesting is the deep lane (100k levels hash);
protocol nesting is the parity lane up to the interpreter limit.

**Delegated sorts are bounded.** Exotic-key dicts (any float/big-int/
mixed/NaN/subclass key) sort by delegating to CPython's own `list.sort`
over live `(key, value)` tuples — O(n log n) Python comparisons plus one
hash entry and one tuple per key, all GIL-held (≈100 MiB for 1M exotic
keys). Past `MAX_DELEGATED_SORT_KEYS` (100k) keys in one dict, or
`MAX_TOTAL_DELEGATED_PAIRS` (500k) delegated pairs walked in one call, the
walk refuses with a generic `ValueError`. Dict *subclasses* whose keys are
all exact `str` (or exact `int`/`bool`, with no two keys comparing equal —
a `True`/`1` tie sorts by value in the interpreter's timsort) sort natively
in Rust on the same fast paths the exact lane uses, so a 100k-str-key
`Counter`/`OrderedDict`/`defaultdict` hashes without touching the delegated
bounds; subclass keys of any other type keep the delegated lane and its
bounds. The str and int/bool fast paths
take no interpreter sort and cost nothing against these bounds.

**GIL model.** The object walk and the leaf spellings run under the GIL
(the standard arg-walk class scaled to an object, O(tree): one borrow
plus copy per str, one i64 read per int, one `repr` call per float); the
canonical-form emission, the SHA-256, AND the owned tree's teardown run
under one `py.detach` (the tree moves into the detached closure, so no
deep-tree `Drop` tail holds the GIL after the digest), streaming into the
hasher. At 12 MiB of the records corpus the walk's
worst heartbeat gap measured 23-32ms of 42-60ms walls (the stdlib
spelling holds ~the whole wall: ratio 1.00 inline vs tors's 0.45-0.60;
`tests/test_gil_release.py`), and the wall race with the full stdlib
expression is a measured dead heat at 64 KiB-12 MiB
(`tests/test_performance.py`): the value is the GIL release and the
parity guarantees, not raw speed over the C encoder. At 1 MiB the wall
stays within 1.5x the stdlib expression (`tests/test_content_hash.py`'s
wall-ceiling pin); the exotic-key delegated sort (`list.sort` over live
objects) is the known slow lane versus the str/int fast paths.

```python
a = {"title": "Q3 outage report", "severity": "high", "tags": ["grid", "north"]}
b = {"tags": ["grid", "north"], "severity": "high", "title": "Q3 outage report"}
tors.content_hash(a) == tors.content_hash(b)
# True: equal content, any key order, one dedup key
tors.content_hash(a)
# '558be2127fa557b06ffd3dd0735697e14022546c969c6c37b01cf6278a178cf2'

request = {"model": "guss-9", "messages": [{"role": "user", "text": "Summarize the Q3 report."}]}
tors.content_hash(request)
# 'b0df18f8e089f15fb56fa51c241151213e67b82e7b656de29b1bafedb3785464'
```

**Async**: no `tors.aio.content_hash` twin: a fast one-shot call over an
in-memory object, not a large-input pass worth a thread dispatch (see
[Async use](async.md)).

## `tors.merkle_root`

```python
def merkle_root(chunks: list[bytes]) -> str: ...
```

A domain-separated SHA-256 Merkle root over `chunks`, as lowercase hex, one
GIL-released native pass wrapping the `rs_merkle` crate. Leaves hash as
`SHA-256(0x00 || chunk)`; internal (two-child) nodes hash as `SHA-256(0x01 ||
left || right)`: the RFC 6962 / Certificate Transparency convention.

**Why this is tors's own hasher, not the wrapped crate's built-in one.**
`rs_merkle`'s default `Sha256Algorithm` has NO domain separation: its
`hash()` is plain undifferentiated `SHA-256(data)`, and its default
`concat_and_hash` feeds `SHA-256(left || right)` through that same
function; verified directly against the crate's vendored source
(`hasher.rs`'s default `concat_and_hash`, `algorithms/sha256.rs`'s `hash`,
rs_merkle 1.5.0), not just its docs. With no domain byte anywhere, a leaf
hash and an internal-node hash live in the same output space: exactly the
ambiguity behind **CVE-2012-2459**, the Bitcoin Merkle-tree bug class where a
forged proof can present an internal node's hash as though it were some
leaf's digest. `merkle_root`/`merkle_diff` don't expose proof generation
yet, but the hash scheme is part of the root's output contract from v1
regardless: roots are meant to be computed once and compared/stored across
calls, and changing the scheme later would silently change every
previously-computed root. It needs to be correct now, not patched in when
proof generation is added.

An unpaired left node at any layer is **promoted** unchanged to the next
layer (rs_merkle's own default `concat_and_hash`, its `None => *left` arm),
never duplicated against itself. Duplication is Bitcoin's original
convention and the actual mechanism CVE-2012-2459 exploited: two
differently-shaped chunk lists (one a duplicate-padded version of a shorter
one) could otherwise produce the same root; promotion is the standard
mitigation and matches RFC 6962. A single chunk's root is just its own leaf
hash: no internal node is built for it.

Deterministic and order-sensitive: the same chunks in the same order always
produce the same root; reordering changes it (this is a Merkle tree, not an
order-insensitive digest/set hash). An empty `chunks` list raises
`ValueError("root of no chunks")`: no non-arbitrary root value exists for
it, and returning some fixed sentinel hash risks being mistaken for a real
chunk's digest by a caller comparing roots. A non-`list` argument or a
non-`bytes` list entry raises `TypeError`.

```python
tors.merkle_root([b"a", b"b", b"c"])
# "36642e73...e6c021ec1"  (hex-encoded SHA-256, RFC 6962-style domain-separated)
```

## `tors.merkle_diff`

```python
def merkle_diff(chunks_a: list[bytes], chunks_b: list[bytes]) -> list[int]: ...
```

Indices where `chunks_a[i] != chunks_b[i]`, one GIL-released native pass.
Comparison is over chunk digests, the same `0x00`-prefixed leaf hash
`merkle_root` uses at a fixed 32-byte cost per index, rather than raw chunk
contents. Every index at or beyond the shorter list's length is reported:
there is no counterpart chunk to compare against there, so a length
mismatch is, in full, "differs at every trailing index," not a partial
answer. Two empty lists diff to `[]`.

This does **not** walk a tree. With both chunk lists held
locally as random-access arrays, hashing each chunk once already answers
"does chunk `i` differ" in O(1) per index afterward: a flat scan over the
digest arrays does exactly the work a tree walk would, without needing one.
A Merkle tree's "skip identical subtrees without transferring them" payoff
matters when the comparison itself is expensive to perform per index (e.g.
over a network); not here, where the O(n) hashing pass already is the
entire cost. `merkle_root` still builds a real tree: that's its actual
job; `merkle_diff` just doesn't need one to answer this particular
question. Argument contract matches `merkle_root`'s: a non-`list` argument
or a non-`bytes` list entry raises `TypeError`.

```python
tors.merkle_diff([b"a", b"b", b"c"], [b"a", b"X", b"c"])
# [1]
tors.merkle_diff([b"a", b"b"], [b"a", b"b", b"c", b"d"])
# [2, 3]: every trailing index beyond the shorter list's length
```

## `tors.md5_hex` / `tors.sha1_hex` / `tors.sha256_hex` / `tors.sha512_hex` / `tors.hmac_sha256_hex` and their `_digest` twins

```python
def md5_hex(data: str | bytes) -> str: ...        # 32 lowercase hex chars
def sha1_hex(data: str | bytes) -> str: ...       # 40
def sha256_hex(data: str | bytes) -> str: ...     # 64
def sha512_hex(data: str | bytes) -> str: ...     # 128
def hmac_sha256_hex(key: str | bytes, data: str | bytes) -> str: ...  # 64

def md5_digest(data: str | bytes) -> bytes: ...   # 16 raw digest bytes
def sha1_digest(data: str | bytes) -> bytes: ...  # 20
def sha256_digest(data: str | bytes) -> bytes: ...  # 32
def sha512_digest(data: str | bytes) -> bytes: ...  # 64
def hmac_sha256_digest(key: str | bytes, data: str | bytes) -> bytes: ...  # 32
```

The one-shot hashing primitives a text pipeline keeps reaching for, each
algorithm in two output spellings over ONE digest computation: the
lowercase-hex `_hex` names and the raw-digest-bytes `_digest` names. The
hex spellings are the cache-key/ETag/request-ID shapes: webhook signature
verification and API auth (`hmac_sha256_hex`, the GitHub/Stripe/Slack
HMAC-SHA-256 convention), ETag and Content-MD5 checks against object
stores (`md5_hex`), quick content compares and legacy-interop digests
(`sha1_hex`), and dedup-cache keys / content addressing
(`sha256_hex`/`sha512_hex` — the same engine `finalize`'s hash tail and
`merkle_root`'s leaves use, without the normalize stage). The digest
spellings serve the call sites that want the bytes themselves: signature
schemes that base64-encode the digest, key-derivation chains that feed a
digest back in as a key, certificate/content thumbprints, advisory-lock
ints sliced off the front — see
[Raw digest bytes](#raw-digest-bytes-the-_digest-spellings). Every
algorithm is the maintained RustCrypto implementation (`md-5`, `sha1`,
`sha2`, `hmac`); nothing is hand-rolled, the same dependency policy as the
rest of the crate. Byte-identical to the stdlib spellings in both output
shapes —
`tors.sha256_hex(data) == hashlib.sha256(data).hexdigest()`,
`tors.sha256_digest(data) == hashlib.sha256(data).digest()`,
`tors.hmac_sha256_hex(key, data) == hmac.new(key, data,
hashlib.sha256).hexdigest()` — and internally one computation:
`tors.sha256_hex(data) == tors.sha256_digest(data).hex()`. All pinned by
exact differentials over hypothesis corpora plus the primary-source
known-answer vectors (RFC 1321, FIPS 180-4, RFC 4231's every
HMAC-SHA-256 case) in `tests/test_hash.py`.

**`md5_hex`/`md5_digest` and `sha1_hex`/`sha1_digest` are
checksum/legacy-interop primitives, never security primitives — either
spelling.** Both are broken and have been since the 2000s: practical md5
collisions date to 2004, sha1's first public collision to 2017 (Google's
SHAttered). Use them for Content-MD5, S3 ETags, cache-busting,
rsync-style quick compares — never for signatures, certificates, or
password handling. The security side of this surface is
`sha256`/`sha512`/`hmac_sha256`, either spelling. The HMAC key is held
in memory for the call and is not zeroized on return — the same posture
as the stdlib `hmac`/`hashlib` spelling, which likewise keeps key
material in ordinary memory.

**str input is its UTF-8 bytes, on purpose.** `hashlib` raises TypeError on
str and makes every caller spell `s.encode("utf-8")` first; tors takes the
str directly — `tors.sha256_hex(s) == hashlib.sha256(s.encode("utf-8"))
.hexdigest()` — because the str-in convention is crate-wide (`normalize`,
`finalize`, the segmentation family). A str holding lone surrogates
therefore raises `UnicodeEncodeError` at the argument boundary, the same
crate-wide contract. bytes input is exactly `bytes`: `bytearray` and
`memoryview` raise TypeError rather than being copied, the bytes-in
family's immutable-buffer doctrine (`b64_encode_bytes`,
tests/test_b64.py) — callers holding one wrap it first,
`tors.sha256_hex(bytes(buf))`, then hash. Any input length is legal, empty included (the
empty-input digests are pinned known-answer vectors); there are no
ValueError paths on this surface.

**Stateless one-shot only — streaming/incremental hashing is out of
scope by charter.** No hash object, no streaming update surface: every
spelling here hashes its whole input in one call, and tors is stateless
by charter ([Design and scope](design.md)), so a `hashlib`-style
constructor object is exactly the persistent-handle shape that charter
cuts (the two measured exceptions, `CompiledPatterns` and
`CompiledLemmaDict`, exist for per-call re-materialization costs a digest
object doesn't have: `hashlib.sha256()` construction is O(1)). A caller
hashing a stream hashes chunk digests and combines them (the
`merkle_root` shape, or a running HMAC chain over chunks); for incremental
feeding, `hashlib`'s object API is the right tool and is not duplicated.

**GIL model.** The argument borrow (zero-copy for ASCII/cached str, for
bytes always; on the two-argument HMAC spellings the one-time O(input)
UTF-8 materialization applies independently per str argument, so two
non-ASCII str inputs pay two materializations — the measured HMAC wall
cells use bytes key+data, equivalently the ASCII zero-copy lane) runs
under the GIL; the whole digest computation — update
and finalize, plus the O(digest-size) hex formatting on the `_hex`
spellings — runs under one `py.detach`. The GIL-held residue is the
marshalling of one short hex string (the `_hex` names) or one fixed-size
`PyBytes` of 16/20/32/64 bytes (the `_digest` names — the
`b64_decode` bytes-return class). The honest
comparison with `hashlib`, measured (Apple Silicon, min-of-3):
CPython's `hashlib` releases the GIL for digest updates of 2048+ bytes
(the `_hashopenssl` threshold), so at multi-MiB sizes the stdlib is
loop-friendly too and tors's GIL release is uniformity, not a latency
win; below the threshold `hashlib` holds the GIL but a sub-2048-byte
digest is microseconds, immaterial either way. On raw throughput the
OpenSSL engines (hardware SHA extensions) win or tie at engine-dominated
  sizes — sha256 12 MiB ~4-5ms (tors) vs ~4ms (hashlib), ratio ~1.1-1.2x
  (absolute figures move with box and load; the band is the statement),
  sha1 ~1.06-1.11, md5 and sha512 dead heats — recorded, not hidden. The wall
  wins tors can assert are the sizes this surface exists for, where
  per-call overhead dominates the engine: hashing a short ASCII str at 0.40-0.54
  of `hashlib.sha256(s.encode("utf-8")).hexdigest()` (the encode is a real
  cost the stdlib makes you pay; non-ASCII str pays a one-time O(input)
  UTF-8 materialization, so the win narrows there), and HMAC at request-signature sizes at
  ~0.31 of even the stdlib's fastest one-shot spelling
  (`hmac.digest(key, data, "sha256").hex()`). Full tables:
  [Performance](performance.md).

  **No `deadline_ms`.** Every deadline-bearing primitive in this crate
  protects against an adversarial-input superlinear blowup. This surface
  has no such shape: hashing cost is linear in the input length, with no
  superlinear structure for an adversary to feed — a caller already
  controls the one lever that bounds the cost (how many bytes they pass).

### Raw digest bytes: the `_digest` spellings

The five `_digest` names return the digest as raw `bytes` — the same
engines, the same single detach, the same contracts as their `_hex`
twins (str input is its UTF-8 bytes; exactly-`bytes` in, so
`bytearray`/`memoryview` raise TypeError — wrap first, `bytes(buf)`;
a lone surrogate raises
UnicodeEncodeError at the borrow; any key length legal, empty included;
the key borrowed and validated before the data; empty input legal) —
without the hex tail. One digest computation per call, two output
spellings to choose from; `tors.md5_digest(x)` is exactly
`bytes.fromhex(tors.md5_hex(x))`.

The raw bytes are what a second consumer layer wants, the shapes a
hex string forces to decode first:

- **Base64 webhook signatures.** The webhook schemes that sign with
  HMAC and *base64*-encode the digest (not hex) need
  `hmac_sha256_digest`: `urlsafe_b64encode` over the raw 32 bytes is
  the signature, verified with `hmac.compare_digest` (below).
- **Key derivation chains.** A labelled subkey is a digest fed back in
  as an HMAC key — `hmac_sha256_digest(hmac_sha256_digest(root,
  label), data)` — the HKDF-style extract/expand shape; a hex string
  would have to be decoded before every link.
- **Content/certificate thumbprints and digest-sliced ints.** A
  stable identifier for an arbitrary name — an advisory-lock int
  (`int.from_bytes(sha256_digest(name)[:8], "big")`) or a thumbprint
  key — is a slice or re-encoding of the raw bytes.

`md5_digest` and `sha1_digest` carry their twins' warning verbatim:
checksum/legacy-interop only, never security. And the streaming
boundary above is the whole family's: the `_digest` spellings are
one-shot too — there is no incremental feeding surface on either
spelling, by charter.

A base64-signature webhook verification, the `hmac_sha256_digest`
shape (the secret is the bytes after the `whsec_` prefix; the signed
content is `"{msg_id}.{timestamp}.{payload}"`; the verify compare is
`hmac.compare_digest`, never `==`):

```python
import base64
import hmac as hmac_module
import tors

secret = base64.b64decode("whsec_3f9d2a8c".partition("_")[2])
signed_content = (
    "msg_5fXn0.1731634200."
    '{"event":"invoice.paid","id":"evt_88213","amount":4200}'
).encode("utf-8")

signature = base64.urlsafe_b64encode(tors.hmac_sha256_digest(secret, signed_content))
# b'q8IyxOXFC_mgdZj-GSqtTl2vj3eTIgMM0HQQ-FfRQVw='

hmac_module.compare_digest(
    signature,
    base64.urlsafe_b64encode(tors.hmac_sha256_digest(secret, signed_content)),
)
# True
```

A digest-sliced advisory-lock int (the stable-identifier shape: a
64-bit int for an arbitrary resource name):

```python
import tors

lock_id = int.from_bytes(tors.sha256_digest("tenant:42:resource:7")[:8], "big")
# 15284293306093710542
```

A labelled derivation chain (the extract-then-expand shape: the label
derives an intermediate key, the data expands it — both links consume
the raw digest bytes):

```python
import tors

root = b"root-key-material"
subkey = tors.hmac_sha256_digest(
    tors.hmac_sha256_digest(root, "tors/db-session-key"), "user:42"
)
subkey.hex()  # the 32 raw bytes, hex for inspection
# "cc3ccddeaa0718afe52e67b9011b49dfe29ae01571495a75e17eea1f59f6e7b6"
```

A webhook-verification shape with the hex spelling, for the schemes that
compare hex signatures (the `str` key and payload spell exactly how
they arrive off the wire; `hmac.compare_digest` stays the right compare):

```python
import hmac as hmac_module
import tors

secret = "whsec_3f9d2a8c"
payload = '{"event":"invoice.paid","id":"evt_88213","amount":4200}'

expected = tors.hmac_sha256_hex(secret, payload)
# "43ec3b86ee42fdcf9640ded30e94104b4a11b9fd43f1624b30471f7b13e76a12"

hmac_module.compare_digest(expected, tors.hmac_sha256_hex(secret, payload))
# True
```

An S3/HTTP ETag check (the md5 checksum use; the value is the normalized
body's digest, matching what the store computed):

```python
import tors

body = tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"

tors.md5_hex(body)
# "487f5cc2c45cc57e638d9fce8c33d95c"

tors.md5_hex(body) == "487f5cc2c45cc57e638d9fce8c33d95c"  # the declared ETag
# True
```

`==` is the correct compare here: an ETag is a non-secret checksum, so
a timing side channel has nothing to leak. Secret comparisons (HMAC
signatures, tokens) must use `hmac.compare_digest` as in the webhook
cells above — never `==`.

## `tors.uuid7_timestamp_ms` / `tors.uuid_version` / `tors.uuid_parse`

```python
def uuid7_timestamp_ms(value: bytes | str) -> int: ...
def uuid_version(value: bytes | str) -> int: ...
def uuid_parse(value: str) -> bytes: ...
```

The UUIDv7 field operations: the hex-grammar work (the structural parse
and the canonical encode) delegated to the Rust `uuid` crate — the uuid-rs
org's, Apache-2.0 OR MIT, no default features, zero transitive
dependencies — with tors's own thin strictness layer on top (generating
IDs is not this surface's job — every producer from `uuid.uuid7()` to
`uuid_utils` already does that — reading the standard time-ordered ID's
fields back out is). A store keyed by UUIDv7 IDs ends up reimplementing
these three operations at every site that paginates by recency or buckets
by time: the 48-bit unix-millisecond timestamp (keyset cursors and
time-bucketed queries against an ID column), the version nibble that says
whether that timestamp means anything, and a strict text-to-bytes parse at
the ID-validation boundary. 16 bytes in, integer out — trivial, which is
exactly why it keeps getting reimplemented slightly wrong.

`uuid7_timestamp_ms` returns the leading six bytes as a big-endian
unix-millisecond timestamp (RFC 9562 section 5.7's `unix_ts_ms` field):
`datetime.datetime.fromtimestamp(ms / 1000, UTC)` is the ID's creation
instant. It raises `ValueError` naming the version actually found when the
version nibble is not 7 — the field is only defined for v7, and the nil
UUID's all-zero field must not silently answer `0`.

`uuid_version` returns the version nibble (byte 6's high half), `0`-`15`, for
any UUID of any variant: the field itself, not an RFC-4122-ness check. The
variant is a different field (byte 8's top two bits) and out of scope; the
stdlib `uuid.UUID.version` property refuses to answer off the RFC 4122
variant, where this answers the nibble question unconditionally.

`uuid_parse` is canonical text to the 16 raw bytes, strict, the
validation-primitive direction. **The stdlib `uuid.UUID` accepts strictly
more than this, deliberately not matched here**: it parses braced text
(`{...}`), the URN prefix (`urn:uuid:...`), hyphen-less hex, and uppercase,
while `uuid_parse` accepts exactly one grammar — 36 ASCII characters
(36 bytes), hyphens at positions 8/13/18/23 (the 8-4-4-4-12 groups),
lowercase hexadecimal elsewhere — raising `ValueError` naming the problem,
the position
(0-based), and the accepted form otherwise. A caller using this as a gate
wants one grammar, the canonical one every producer emits, not the
stdlib's permissive union; that closed-set strictness is the same contract
`b64_decode`'s `validate=True` default and the `errors=`/`boundary=`
parameters already establish. The divergences are pinned, not accidental:
`tests/test_uuid.py` proves the stdlib accepts each loose form in the same
test that pins tors rejecting it. The parse mechanics behind the contract:
`Uuid::parse_str` (the crate's) owns structure and the text-to-bytes
transcode, and tors's strictness layer is one comparison — accepted text
must be byte-equal to its parsed value's re-encoded canonical form, which
is what rejects the crate's (and the stdlib's) loose spellings at exactly
the canonical grammar. The strict contract is not the crate's default,
which is precisely why the layer is tors's own.

The int-out pair takes exactly `bytes` (16 bytes, `ValueError` naming the
count otherwise; `bytearray`/`memoryview` are `TypeError`, the bytes-in
surface's borrows-the-argument-copies-the-16-bytes contract: the bytes
spelling borrows the argument, copies the 16 bytes out) or canonical `str`
(same strict grammar, same errors); `uuid_parse` takes exactly `str`. A
`str` holding lone surrogates fails the borrow itself
(`UnicodeEncodeError`) before any grammar check runs.

```python
import datetime
import uuid

# any v7 works; this one is fixed so the outputs below are literals
u = uuid.UUID("01977420-dc00-7abc-9def-98765432100f")
tors.uuid_version(u.bytes)  # 7
tors.uuid7_timestamp_ms(u.bytes)  # 1750000000000
tors.uuid7_timestamp_ms(str(u))  # 1750000000000: canonical text accepted too
datetime.datetime.fromtimestamp(1750000000000 / 1000, datetime.timezone.utc)
# datetime.datetime(2025, 6, 15, 15, 6, 40, tzinfo=datetime.timezone.utc)
tors.uuid_parse(str(u)) == u.bytes  # True
tors.uuid_parse(str(u).upper())
# ValueError: UUID text must be lowercase hex: found uppercase 'D' at
# position 9 (the stdlib uuid module accepts uppercase; tors's canonical
# form deliberately does not)
```

GIL model: the bytes spelling borrows the argument, copies the 16 bytes,
and the bit extraction (field read) runs detached under `py.detach`; the
str spelling validates and
transcodes under the GIL (the whole input is 36 bytes, smaller than the
call's own marshalling residue — a detached parse would be overhead for
its own sake) with the extraction detached after it, keeping the crate's
GIL-free-core contract uniform across the trio. `uuid_parse` is that
contract's honest exception: its whole work *is* the 36-byte parse (there
is no int-out tail to detach) and it runs GIL-held by design, ~70ns a call
(re-measured after the `uuid`-crate adoption; the crate's const-fn parser
is faster than the hand-rolled scan it replaced).
Every per-call GIL-held residue on this surface is sub-microsecond; the
heartbeat cell in `tests/test_gil_release.py` pins the batch-loop band.

## `tors.simhash64`

```python
def simhash64(text: str) -> int: ...
```

A 64-bit SimHash fingerprint of `text`, one GIL-released native pass: the
fuzzy near-duplicate gate that sits alongside `tors.finalize`'s exact
SHA-256 gate. Charikar's weighted-bit-voting construction (the near-web-scale
near-duplicate-detection shape Manku, Jain, and Das Sarma built at Google,
WWW 2007): `text` is tokenized into words via the same UAX #29 word
segmentation `tors.word_bounds` drives (`split_word_bounds`, its sibling
spelling over the same tables), skipping any segment that is entirely
whitespace; each token's UTF-8 bytes are hashed with a deterministic
FNV-1a-64; then, for each of the 64 bit positions, every token votes +1 if
its hash has that bit set and −1 if it does not, and the output bit is 1
wherever the vote sums positive (ties, including the zero-token case,
resolve to 0).

**Why FNV-1a and not `std`'s `DefaultHasher`.** A dedupe fingerprint must be
stable across processes and machines: `DefaultHasher` is seeded per process
(`RandomState`), so a fingerprint it produced would silently change between
runs, breaking any cross-run/cross-machine dedupe built on it. FNV-1a is deterministic forever and adequate for a
voting hash: it only needs to
spread tokens reasonably uniformly across the 64 bit positions, not resist
adversarial collisions; a collision between two distinct tokens merely
blurs one vote among 64 counters.

**What this answers that `finalize` cannot.** `finalize`'s SHA-256 tail and
`merkle_root` answer "is this text byte-identical": a single changed comma
already fails that gate. `simhash64` answers "is this text nearly the
same": Hamming distance `(a ^ b).bit_count()` (a one-liner at the call
site, which is why this returns the raw `int` rather than shipping a
redundant distance function) grows slowly with edit distance, so
near-duplicates cluster within a handful of differing bits while unrelated
texts sit far apart. The realistic pipeline runs both gates off one store:
exact dupes at Hamming distance 0 via `finalize`'s hash, near-dupes at
small Hamming distance via `simhash64`, everything else far apart.

**Bag of words: order does not matter.** The vote is over the multiset of
tokens, not their sequence: `tors.simhash64("the quick brown fox")` and
`tors.simhash64("fox brown quick the")` fingerprint identically. A repeated
token votes once per occurrence, so frequency is still part of the bag:
appending one more `"the"` to a sentence that already has several is a
different multiset and (usually) a different fingerprint.

**Tokenization is UAX #29, not a whitespace split**: the same segmentation
`word_bounds` uses, not an ad-hoc `str.split()`. This matters for
scriptio-continua text (CJK, Thai, and similar scripts with no spaces
between words): the segmenter still finds word boundaries inside a
space-free run rather than treating it as one giant token. A whitespace-only
`text` has no word tokens (the WSegSpace rule joins a whitespace run into
one segment, which the tokenizer then skips as not-a-word) and fingerprints
to `0`, same as empty text.

**Limitations: read before deploying a threshold.** SimHash is
**not cryptographic and not collision-resistant**: FNV-1a is a fast,
uniformly-spreading voting hash, not a security primitive, and two
unrelated documents can coincidentally land close together, especially on
short text. There is **no universal near-duplicate cutoff**: the distance
a "same document, small edit" pair sits at scales with document length
(short texts have thin per-bit vote margins, on the order of the square
root of the token count, so the same edit flips more bits), and the
unrelated floor depends on vocabulary overlap. The measured anchors
(pinned in `src/simhash_impl.rs`'s test suite and mirrored in
`tests/test_simhash.py`): over a 95-word document, every single-word
swap/drop/insert moves the fingerprint at most 4 bits; over 8-12-word
sentences, the identical class of edit moves up to 14 bits; unrelated
sentence pairs measured as close as 23 bits apart in the same battery.
Calibrate any "probable near-duplicate" threshold per deployment against
known near-dup and known-far pairs: do not import a textbook rule of
thumb unchanged.

Deterministic across processes, versions, and machines (FNV-1a has no
per-process seed). `O(n)` in the length of `text` with `O(1)` extra memory
beyond the 64 vote counters: no intermediate token list is materialized.

```python
tors.simhash64("the quick brown fox jumps over the lazy dog")
# 14607312263354641902
tors.simhash64("the quick brown fox jumps over the lazy dog!")  # one edit
# 14601682762745638348: Hamming distance 6, not ~32 (unrelated-text scale)
tors.simhash64("")
# 0
```

## `tors.simhash128`

```python
def simhash128(text: str) -> int: ...
```

The 128-bit spelling of `tors.simhash64`: identical tokenization, identical
vote (each token's FNV-1a hash votes ±1 per bit, positive sum sets the bit),
run at 128 bits instead of 64: twice the bit positions, not a
64-bit fingerprint zero-extended into a wider int (the FNV-1a offset basis
and prime are the real 128-bit constants, and every one of the 128 bits gets
its own independent vote). Use it for corpora where a 64-bit fingerprint's
near-duplicate band and unrelated floor sit too close together to separate
reliably.

Measured by the same battery as `simhash64` and pinned in
`src/simhash_impl.rs`'s test suite: the unrelated floor widens from 23 bits
at 64 bits to 40 at 128, while the near-duplicate bands grow only
sublinearly (95-word document scale: 3 bits worst case versus the 64-bit
4; 8-12-word sentence scale: 20 versus 14). The wider width buys more
separation between the near-duplicate band and the unrelated floor, which is
what a corpus whose 64-bit bands overlap needs. The same calibration caveat
applies: there is no universal cutoff, and thresholds must be measured
against known near-dup and known-far pairs for the corpus at hand.

Same contract otherwise: deterministic across processes and machines
(FNV-1a has no per-process seed), order-independent (a bag-of-words vote),
empty or whitespace-only text fingerprints to `0`, and `(a ^ b).bit_count()`
at the call site gives the Hamming distance.

```python
tors.simhash128("the quick brown fox jumps over the lazy dog")
# 317101931942163558849153286541522090150
tors.simhash128("")
# 0
```

## `tors.minhash_signature`

```python
def minhash_signature(
    text: str,
    *,
    num_perm: int = 128,
    shingle_size: int = 3,
    seed: int = 0,
) -> list[int]: ...
```

The MinHash signature of `text`, one GIL-released native pass: the
recall-side near-duplicate complement to the SimHash family. `simhash64`/
`simhash128` are the precision side — one compact fingerprint, cheap to
store and compare, but a weak recall instrument at corpus scale (two
paraphrased documents sit far apart in Hamming space however related
their vocabulary). MinHash is the recall side: `num_perm` min-hashes per
document, and the fraction of positions on which two signatures agree
estimates the Jaccard similarity of the two documents' shingle sets — the
quantity a corpus-scale ingestion pipeline (100k+ documents) bands into an
LSH table and recalls candidate pairs on (Broder's MinHash, the primitive
behind near-duplicate detection at web scale).

**The estimator contract.** For shingle sets `A` and `B`, each position
`i` of the two signatures agrees exactly when the shingle achieving the
minimum of `h_i` over `A ∪ B` lies in `A ∩ B`, which happens with
probability `J(A, B) = |A ∩ B| / |A ∪ B|`. So
`E[agreement] = J` and the standard error over `k = num_perm` positions is
`sqrt(J(1 - J) / k)` — `O(1/sqrt(k))`, about 0.044 at the default
`num_perm = 128` at the worst case `J = 0.5`, halving with every 4x in
`num_perm`. The comparison is one expression at the call site:

```python
sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)
```

**Tokenization and shingling.** Tokens are the same tokenizer
`tf_idf`/`bm25_rank` ride: UAX #29 word segments (`word_bounds`' own
segmentation), segments made entirely of whitespace skipped, each
lowercased with Unicode-correct case folding — so `"Hello, WORLD!"` and
`"hello, world!"` signature identically, and whitespace shape (tabs,
newlines, runs) is invisible. A shingle is `shingle_size` consecutive
tokens hashed under an injective length-prefixed framing: the window's
token count as one little-endian u64, then per token its UTF-8 byte
length as one little-endian u64 followed by the bytes themselves, the
whole frame fed to XXH64 (seed 0). Length-prefix codes are uniquely
decodable, so the framing is injective by construction — deliberately,
because "no UAX #29 token can contain U+001F" is false: U+001F is not
whitespace, so the segmenter keeps it (`tors.word_bounds("a\x1fb")` is
the three tokens `["a", "\x1f", "b"]`), and UAX #29 WB4 (ignore
Extend/Format/ZWJ) then glues a following combining mark, ZWJ, or SOFT
HYPHEN onto it (`tors.word_bounds("\x1f\u0301")` is the single token
`["\x1f\u0301"]`). A `token + U+001F + token` join would rest on the
narrower (and segmentation-table-sensitive) claim that U+001F never
mixes into a longer segment; the framing rests on nothing the segmenter
can take away. Both directions are pinned in `tests/test_minhash.py`:
the `\x1f`-adjacent mark corpus (every combining mark in U+0300–U+036F,
plus ZWJ and SOFT HYPHEN, attaches) and the framing's injectivity over a
separator-bearing window domain. Word shingles,
not character shingles: natural-text near-duplicates preserve word
sequence far more often than exact character spans — a reflowed paragraph
or a swapped word shifts character k-grams wholesale while word k-grams
survive, which is what recall at corpus scale needs.

**The pinned arithmetic (the determinism contract).** Every element is
fixed, documented arithmetic, platform-independent (integer ops only), so
the same text at the same parameters produces the identical signature
across processes, machines, and platforms within one tors version — the
same stability requirement the simhash family states for its FNV-1a. The
boundary that claim stops at: the signature is a function of the crate's
UAX #29 segmentation tables (`unicode-segmentation`, pinned in
`Cargo.lock`) as well as of the frozen arithmetic, so a tors release that
bumps those tables can change signatures — re-fingerprinting every
affected document. This surface's advertised use is persisted signatures
and LSH tables at corpus scale, where a re-fingerprinting upgrade
silently invalidates the table: a caller persisting either across tors
versions must re-baseline on upgrade (within a version nothing varies —
the arithmetic and XXH64 are frozen):

- each shingle is hashed with XXH64, seed 0: the frozen-spec algorithm
  (final since xxHash 0.7.0 — the digest for a given seed and byte stream
  is part of the spec, not an implementation detail), via `twox-hash`;
- `signature[i] = min` over shingles of `(a_i * x + b_i) mod p`, with
  `p = 2^61 - 1` (the Mersenne prime, the standard MinHash field) and the
  `(a_i, b_i)` pairs derived from `seed` by a SplitMix64 stream — `a_i`
  in `[1, p-1]` (a zero multiplier would collapse the permutation to a
  constant), `b_i` in `[0, p-1)`, two draws per permutation in
  permutation order (so a smaller `num_perm` yields a prefix of a larger
  signature at the same seed). `rand` is deliberately not involved: it is
  not in tors's dependency tree and its stream internals are not a
  semver-stable contract, while the ~8-line SplitMix64 derivation is
  golden-pinned on both the Rust and Python sides
  (`tests/test_minhash.py`).

This is fixture-grade determinism, not cryptography: nothing here resists
an adversary crafting collisions, and nothing needs to.

**The empty-shingle-set convention.** Empty text, whitespace-only text,
or fewer tokens than `shingle_size` yields `num_perm` copies of the u64
MAX sentinel (`2**64 - 1`): deterministic, seed-invariant, and unable to
collide with any real minimum (the affine outputs live in
`[0, 2**61 - 1)`, a disjoint range). Two empty documents agree at every
position, an empty-vs-nonempty pair at none, and an all-sentinel
signature is a stable digest an LSH table can bucket empty documents
under.

**The LSH table is caller state.** tors stays stateless
([design.md](design.md)): this function computes one document's signature
and keeps nothing; the banding table, the candidate store, and the
threshold calibration are the caller's. A banding helper (cut a signature
into `r`-element bands and hash them for table keys) is a future
companion question, not something hidden inside this core.

**Bounds.** `num_perm` must be in `[1, 1024]` and `shingle_size` at least
1; an in-range value outside those bounds raises `ValueError` naming the
bounds before any work runs, while an int outside the i64 range the
binding extracts (`num_perm=10**30`) raises pyo3's own `OverflowError`
at extraction instead — the `truncate_to_bounds`-identical pattern for
every i64-typed size argument.
`seed` is any int, reduced mod `2**64` with two's-complement semantics
for negatives (`seed=-1` is `seed=2**64 - 1`). All three int parameters
are accepted through the `__index__` protocol (numpy integers and other
int-likes work; the slot is dispatched, never the instance's own
`__and__`, so masking cannot alter the value) — and `__index__` itself
is caller code: it runs exactly once, with its own side effects, and
its own failure propagates unchanged rather than masking as a parameter
error. `bool` is rejected explicitly in every position (`num_perm=True`
is a `TypeError`, not 1), including as an `__index__` result; anything
without `__index__` (str, float, None, bytes) and an `__index__`
returning a non-int are `TypeError`. A non-str
`text` raises `TypeError`; text bearing lone surrogates raises
`UnicodeEncodeError` (the crate-wide str-borrow contract). Cost is
`O(tokens × shingle_size)` hashing (every step re-hashes the whole live
window under the length-prefixed framing — widths 64/256 on 100 KiB cost
~9/~32 ms against ~2 ms at the default width on the dev box,
macOS/arm64, release) plus `O(distinct × num_perm)` in the min-sweep, the
dedup-first shape: each distinct shingle hash updates the minima once,
so a repeated-token document rides its handful of distinct shingles
rather than its thousands of occurrences. Resident memory is the live
`shingle_size`-deep token window — `O(min(tokens, shingle_size))` —
plus the distinct-hash set plus the `num_perm` coefficients: the token
list is streamed, never retained
(the pre-fix shape held the whole `Vec<String>` across the sweep, a
measured ~77 MB transient at 12 MB of prose: 133 MB peak vs 56 MB after,
same interpreter and corpus baseline). Past 1024 tokens of width the
short stream never materializes at all: a retention-free token count
decides "fewer tokens than the width" first, so a huge `shingle_size`
over a large text answers the sentinel at the tokenizer transient
(~13.5 MB at width 10⁹ peaks ~30 MB, not the ~145 MB the retaining
shape held, same dev box). There is no
`deadline_ms` on this call (unlike `diff_opcodes`): the sweep has no
superlinear shape, only the linear ones above, so the lever is the
caller's own input size — bound it before calling (truncate, chunk, or
`max_bytes`-gate the read) rather than after. One guard does exist: a
wide fillable shingle window (`shingle_size > 1024` with at least
`shingle_size` tokens) costs `(tokens − shingle_size + 1) × shingle_size`
token-hashes — work peaking near `shingle_size ≈ tokens/2`, not at the
huge widths the sentinel short-circuit already absorbs — and past the
sweep budget of 2^26 token-hashes (~0.4 s) the call raises `ValueError`
naming the bound, the same two-sided pattern as `num_perm`'s cap
(pinned in `tests/test_minhash.py`). The gate's own token count is
capped at `⌊budget/shingle_size⌋ + shingle_size` tokens (≤ ~66.5k for
every admissible width), so a huge stream is rejected in constant time
without a GIL-held walk of the whole input, and the error reports the
*minimum* provable spend past the cap ("at least N token-hashes"). The gate's own token count is
capped at `⌊budget/shingle_size⌋ + shingle_size` tokens (≤ ~66.5k for
every admissible width), so a huge stream is rejected in constant time
without a GIL-held walk of the whole input, and the error reports the
*minimum* provable spend past the cap ("at least N token-hashes"). Measured (dev box,
macOS/arm64, release): ~13.5 MB of repetitive prose at the default 128
permutations completes in ~0.25 s, while 1 MiB of distinct-rich text
(every token unique, the `minhash_signature_distinct` bench row and the
Python worst-case cell) costs ~104 ms at k=128 and ~226 ms at k=1024
(criterion medians, 10 samples) — the ~100M-affine-op worst case the
caller bound is calibrated on; the suite's 2.5 s regression ceiling is
~10x the repetitive nominal
(see `tests/test_minhash.py`), a tripwire, not a target. `await
tors.aio.minhash_signature(...)` runs the call under `asyncio.to_thread`
(see [Async use](async.md)).

```python
original = (
    "The quarterly oil sample interval for field outages was adjusted after the bushing "
    "torque specifications changed. Maintenance windows now close within fourteen days. "
)
edited = (
    "The monthly oil sample interval for field outages was adjusted after the insulator "
    "torque specifications changed. Maintenance windows now close within fourteen days. "
)
unrelated = "Pack my box with five dozen liquor jugs."

a = tors.minhash_signature(original)
b = tors.minhash_signature(edited)
u = tors.minhash_signature(unrelated)
a[:3]
# [151086443443351341, 59387643775660493, 132191052063639682]
sum(x == y for x, y in zip(a, b)) / len(a)
# 0.6015625: near-duplicates — the two-word swap leaves most 3-word
# shingles intact, so the estimated shingle-set Jaccard stays high
# (exact J here: 0.6429)
sum(x == y for x, y in zip(a, u)) / len(a)
# 0.0: an unrelated text agrees on no positions
tors.minhash_signature("")[:3]
# [18446744073709551615, 18446744073709551615, 18446744073709551615]:
# the empty-shingle-set sentinel
```

## `tors.CompiledLemmaDict`

```python
class CompiledLemmaDict:
    def __init__(self, mapping: dict[str, str]) -> None: ...
    def __len__(self) -> int: ...
```

A pre-built `lemma_dict` mapping: the `re.compile()` answer to `tf_idf`/
`bm25_rank`/`apply_pipeline`'s per-call dict-materialization cost (the
measured cost and the loading recipes are in `tors.tf_idf`'s docs below).
`CompiledLemmaDict(mapping)` extracts the mapping into a Rust `HashMap`
once, under the GIL (the same linear-in-size walk `tf_idf`/`bm25_rank`/
`apply_pipeline` pay per call for a raw `dict`: the whole point is paying
it here, a single time, instead); every later call is an `Arc::clone`. The
argument must be exactly a `dict[str, str]` (a non-`dict` argument, or one
with a non-`str` key or value, raises `TypeError`); `len(cl)` is the
number of entries. Immutable once built, and not a caching mechanism:
nothing inside tors remembers a raw `dict` between calls, so a caller who
mutates their `dict` and re-passes it is always honored: the handle is
the caller's explicit opt-in to fixness, the same narrow shape
`re.compile()` has in the stdlib.

## `tors.tf_idf`

```python
def tf_idf(
    corpus: list[str],
    *,
    strip_accents: bool = False,
    stemmer: str | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[list[tuple[str, float]]]: ...
```

**Async**: `await tors.aio.tf_idf(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Stateless TF-IDF over `corpus`, one GIL-released native pass: no
vocabulary/vectorizer object persists between calls; every call scores
fresh over exactly the documents given. Fills a real gap: Python's stdlib
has no TF-IDF at all, and `scikit-learn`'s `TfidfVectorizer` pulls in
numpy/scipy for a lightweight pipeline that just wants keyword weighting or
document similarity. Every TF-IDF crate on crates.io is stale or effectively
abandoned, and the math is a few dozen lines with no ML machinery: hand-rolled
directly, the same surgical tradition as `html_unescape`/`extract_code_blocks`,
not a dependency pull.

**Tokenization**: UAX #29 word segments (`tors.word_bounds`'s own tables),
restricted to segments carrying at least one non-whitespace codepoint: a
"term" is a real token, not a raw `word_bounds` segment (which gives
inter-word whitespace its own segment). Terms are lowercased with Rust's
Unicode-correct `str::to_lowercase` (not an ASCII-only fold) before
counting: `"Cat"` and `"cat"` are the same term.

**TF** (per document `d`, term `t`): the raw count of `t` in `d`, not
length-normalized. A caller wanting `tf / len(d)` divides the returned raw
count themselves; the raw count is the more broadly reusable number (a
length-normalized score would silently discard the total-count information
some callers want directly).

**IDF** (term `t`, corpus size `N`, document frequency `df(t)` = number of
documents containing `t` at least once): the smoothed formula
`ln((1 + N) / (1 + df(t))) + 1`: scikit-learn's own `smooth_idf=True`
default (as if one extra document existed containing every term exactly
once), not the textbook `ln(N / df(t))`. The textbook formula
gives a term appearing in every document an IDF of exactly `ln(1) = 0`, so
its score is `0` regardless of how often it occurs: an unhelpfully sharp
cliff for exactly the "practically universal term" case a caller most wants
a small-but-nonzero weight for. The smoothed form stays strictly positive
there (`ln((1+N)/(1+N)) + 1 = 1` exactly, for the universal-term case
`df(t) = N`) while still monotonically favoring rarer terms.

**score**(t, d) = `tf(t, d) * idf(t)`.

**Output is sparse**: one `(term, score)` list per input document,
alphabetically sorted, holding only that document's own terms, never a
vocabulary-size-by-corpus-size dense structure (wasteful for anything but a
tiny shared vocabulary). An empty corpus returns `[]`. An empty-string
document returns `[]` at its position: the output always has exactly
`len(corpus)` entries, position-matched to the input. A non-`list` argument
or a non-`str` entry raises `TypeError`; a lone surrogate raises
`UnicodeEncodeError` at the argument boundary, the same str-in convention
every other function here documents.

**`strip_accents=True`** NFD-decomposes each token (`tors.nfd`'s own
algorithm) and drops every combining-mark codepoint before scoring:
`"café"` and `"cafe"` become the same term. This is unconditional: it does
NOT replicate a real bug in scikit-learn's own `strip_accents_unicode`
(gh-15087), which short-circuits and silently skips stripping when a token
arrives already NFD-decomposed (e.g. `"e"` + a combining accent rather than
precomposed `"é"`), tors always strips regardless of whether decomposition
itself was a no-op. NFD, not NFKD: NFKD's extra compatibility decomposition
would also touch ligatures/width variants (`"ﬁ"` → `"fi"`), which
accent-folding shouldn't. Default `False`: accents are preserved unless
asked to fold them.

A token made entirely of combining marks (a bare accent with no base
letter, e.g. from already-decomposed input) strips down to the empty
string; such tokens are dropped, never counted as a `""` term. This is
the same structural exclusion `scikit-learn`'s default `token_pattern`
(`r"(?u)\b\w\w+\b"`, which can never match zero characters) achieves for
`TfidfVectorizer`, applied here after folding rather than via a regex over
the raw text.

**`stemmer`** names a Snowball algorithm (`"english"`, `"french"`,
`"german"`, ... 18 languages via the `rust-stemmers` crate: an
unrecognized name raises `ValueError` naming every valid choice), applied
after lowercasing/accent-folding: `"running"`/`"runs"`/`"runner"` all stem
toward `"run"`. Default `None`: no stemming. Full lemmatization is
explicitly out of scope: it needs a per-language dictionary or a
POS-tagging model, not an algorithm, which breaks tors's no-external-model
posture (the same boundary that kept schema-aware JSON/YAML coercion out of
this crate). Stemming is cruder: it can't distinguish "better" the
comparative from "better" the verb, but it's correct, deterministic, and
dependency-light.

**`lemma_dict`** is a caller-supplied `word -> lemma` map, applied last
(after any stemming): the fully-folded token is looked up, and its mapped
value replaces it if present, else the folded token is kept as-is. tors
does not bundle a lemma dictionary: full lemmatization needs a
per-language dataset or a POS model, the same "no external model"
boundary that kept stemming's own decision above. `lemma_dict` is the
mechanism, not the data: the same shape `replace_many` already takes a
caller-supplied replacement map instead of a bundled one. Combining
`stemmer` and `lemma_dict` together is unusual but well-defined, not an
error: the dict is consulted on the already-stemmed form. A non-`dict`
argument, or one with a non-`str` key/value, raises `TypeError`. Default
`None`: no substitution.

**Loading a lemma dictionary.** `lemma_dict` takes a plain `dict[str, str]`,
so any source you can turn into one works. Three common ones:

- **spaCy's lemma lookup tables** ship as JSON, already `word -> lemma`:
  `lemma_dict = json.load(open("spacy-lookups-data/spacy_lookups_data/data/en_lemma_lookup.json"))`.
- **The Lemmatization Lists project** (`michmech/lemmatization-lists`) ships
  TSV in the opposite direction, `lemma<TAB>word`: invert each row:
  `lemma_dict = {word: lemma for lemma, word in (line.split("\t") for line in open("lemmatization-en.txt"))}`.
- **NLTK's WordNet exception lists** (`nltk_data/corpora/wordnet/*.exc`, one
  file per part of speech: `adj.exc`, `adv.exc`, `noun.exc`, `verb.exc`)
  are already `word lemma` pairs, one per line, space-separated:
  `lemma_dict = {k: v for path in exc_paths for line in open(path) for k, v in [line.split()[:2]]}`.
  Each `.exc` file carries no part-of-speech tag of its own, so merging all
  four into one `lemma_dict` means a word ambiguous across parts of speech
  (a verb and a noun spelled the same, lemmatized differently by each)
  collapses to whichever file's entry was merged in last; if that matters
  for your corpus, keep one `lemma_dict` per part of speech instead and pass
  each into its own `tf_idf`/`bm25_rank`/`apply_pipeline` call.

**`lemma_dict`'s real cost, measured, and its fix.** A raw
`dict[str, str]` marshals the whole Python `dict` into a Rust
`HashMap<String, String>` fresh on every call. For a small map (a handful
of entries) that cost is negligible. For a realistically-sized lemma table
(spaCy's own English lookup data is tens of thousands of entries), it is
not. A 20,000-entry `lemma_dict` costs
roughly 1.4ms of marshalling per call, independent of how much text that
call processes. Called once over a large batch, that cost amortizes away.
Called repeatedly (once per short text, the shape a naive loop reaches
for), that fixed cost dominates and can make `apply_pipeline`/
`tf_idf`/`bm25_rank` measurably slower than an equivalent idiomatic Python
loop (`tools/bench_lemma_dict.py` measured roughly 10-460x slower at small
per-call batch sizes with a 20,000-entry map).

`tors.CompiledLemmaDict` is the fix: build the `HashMap` once, reuse it
across every call. `lemma_dict` accepts either a raw `dict[str, str]` (the
per-call cost above) or a `CompiledLemmaDict` (an `Arc::clone` per call
after the one-time build: measured at roughly 600x faster than the raw-
`dict` path for a 20,000-entry map called repeatedly). Building a
`CompiledLemmaDict` still costs the same materialization time; it pays
that cost once instead of on every call:

```python
compiled = tors.CompiledLemmaDict(lemma_dict)  # pay the cost once
for batch in many_batches:
    tors.apply_pipeline(batch, lowercase=True, lemma_dict=compiled)  # O(1) per call after
```

`CompiledLemmaDict` is immutable once built and not a general
caching mechanism (no identity-keyed cache lives inside tors, silently
reusing a stale mapping if a caller mutated their `dict` between calls:
that footgun is exactly what an explicit, caller-controlled handle avoids).
It does not reopen the case for a stateful pipeline object: it is scoped
to this one parameter, on this one measured cost, the same narrow shape
`re.compile()` has in the stdlib.

**What this is for, and what it is not.** A lightweight keyword-weighting
and document-similarity primitive for pipelines that don't want an ML
dependency: surfacing a document's most distinctive terms, or comparing
documents by their score vectors. It is not a full NLP pipeline stage: no
lemmatization, no stop-word removal, no n-grams; stemming and accent-folding
are opt-in, everything else stays case-folding only. And, matching this
crate's posture: this is a correctness/capability
primitive, not a retrieval- or model-quality promise; whether TF-IDF
weighting helps a particular downstream task is the caller's question to
answer.

```python
tors.tf_idf(["the cat sat on the mat", "the dog sat on the log", "birds fly in the sky"])
# [[('cat', 1.6931471805599454), ('mat', 1.6931471805599454), ('on', 1.2876820724517808),
#   ('sat', 1.2876820724517808), ('the', 2.0)],
#  [('dog', 1.6931471805599454), ('log', 1.6931471805599454), ('on', 1.2876820724517808),
#   ('sat', 1.2876820724517808), ('the', 2.0)],
#  [('birds', 1.6931471805599454), ('fly', 1.6931471805599454),
#   ('in', 1.6931471805599454), ('sky', 1.6931471805599454),
#   ('the', 1.0)]]
```

## `tors.bm25_rank`

```python
def bm25_rank(
    query: str,
    corpus: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    strip_accents: bool = False,
    stemmer: str | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[tuple[int, float]]: ...
```

**Async**: `await tors.aio.bm25_rank(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

Okapi BM25 score for every document in `corpus` against `query`, one
GIL-released native pass: `(index, score)` pairs for every document: no
top-k cutoff baked in, slice/sort the result yourself; sorted by score
descending, ties broken by ascending original index.

**A reranking primitive, not a search index.** `bm25_rank` recomputes
corpus statistics from scratch on every call. That is the right shape for
the common RAG pattern of reranking a small, already-retrieved candidate
set (tens to a few hundred documents: a vector-search step's top-k, say)
against one query: no state to manage, composes with the rest of tors's
flat, stateless primitives, cheap enough at that scale to recompute per
call. It is NOT a search engine: a corpus with thousands of
documents queried repeatedly wants a real inverted index built once and
queried many times: recomputing corpus statistics from scratch on every
single query wastes that work every time. For that, reach for a real
search engine (`tantivy` is the mature, dominant choice in Rust); tors
does not build persistent index objects, the same scope line that kept a
Merkle inclusion-proof API out of this crate.

**The formula**: for query `Q` (tokenized to a set of DISTINCT terms: a
repeated query word contributes its IDF once, the standard Robertson/
Spärck-Jones convention) and document `D`:

```text
score(D, Q) = sum over t in Q of IDF(t) * f(t,D) * (k1 + 1)
                                  -----------------------------------
                                  f(t,D) + k1 * (1 - b + b * |D| / avgdl)

IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
```

`N` = corpus size, `n(t)` = number of documents containing `t`, `f(t,D)` =
`t`'s occurrence count in `D`, `|D|` = `D`'s token count, `avgdl` = the
corpus's mean document length. `IDF` is the always-non-negative "+1"
(Lucene-since-2011) variant, not the classic
`ln((N-n(t)+0.5)/(n(t)+0.5))` form, which goes negative for a term
appearing in more than half the corpus: a surprising, unwanted answer for
a reranking primitive with no stopword list to filter such terms out
first.

`k1` (>= 0, default 1.5) tunes term-frequency saturation: how much repeat
occurrences of a term keep adding to the score; `b` (in `[0, 1]`, default
0.75) tunes length normalization: how much a longer-than-average document
is penalized. Both are Lucene/Elasticsearch's own defaults. Out-of-range
values raise `ValueError`.

**Tokenization**: the same "real word token, lowercased" convention
`tf_idf` uses (UAX #29 word segments, non-whitespace only). `strip_accents`/
`stemmer`/`lemma_dict` are `tf_idf`'s exact same opt-in knobs (see its docs
for the accent-folding/stemming/lemma-substitution details), applied
identically to `query` and every `corpus` document, since scoring a query
normalized differently from its corpus produces meaningless scores, not
just imprecise ones. All default off, reproducing the original
lowercase-only tokenization exactly.

An empty `corpus` returns `[]`. An empty (or all-whitespace, or
no-real-tokens) `query` scores every document `0.0`: there are no query
terms to accumulate a score over, the correct, unsurprising answer, not an
error. A non-`list` `corpus` or a non-`str` entry raises `TypeError`; a
lone surrogate raises `UnicodeEncodeError` at the argument boundary. A
non-`dict` `lemma_dict`, or one with a non-`str` key/value, raises
`TypeError`.

**No `deadline_ms`**: every deadline-bearing primitive in this crate
protects against adversarial-input superlinear blowup (`levenshtein`/
`jaro`'s O(n·m) DP tables, `similarity_ratio`'s windowed Myers scans).
`bm25_rank` has no such shape: cost is linear in total corpus token count
plus `corpus_size * distinct_query_terms`, both driven directly and
proportionally by the sizes of the caller's own arguments, not by
adversarial structure within them. A caller already controls the one lever
that bounds the cost (how large a `corpus` they pass).

`tors` makes no claim about retrieval or relevance quality for any
particular corpus or query: BM25 is a well-specified ranking formula,
correctly implemented here, not a model-quality promise.

```python
tors.bm25_rank(
    "quick fox",
    [
        "the quick brown fox jumps over the lazy dog",
        "a lazy cat sleeps all day",
        "the fox and the dog are friends",
    ],
)
# [(0, 1.3162195220480066), (2, 0.4798180901812613), (1, 0.0)]

tors.bm25_rank("cafe", ["café société", "totally unrelated text"], strip_accents=True)
# [(0, 0.7617001984175222), (1, 0.0)]
```

## `tors.apply_pipeline`

```python
def apply_pipeline(
    texts: list[str],
    *,
    nfd: bool = False,
    lowercase: bool = False,
    strip_accents: bool = False,
    stemmer: str | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
    collapse_whitespace: bool = False,
) -> list[str]: ...
```

**Async**: `await tors.aio.apply_pipeline(...)` runs this under `asyncio.to_thread` (see [Async use](async.md)).

A stateless, general-purpose batch text preprocessor: every requested step
fused into one GIL-released native pass over the whole `texts` list.

**No pipeline object for the pipeline itself: pure function composition.**
A `re.compile()`-style compiled-pipeline handle for the whole pipeline
(build once, `.apply()` many times) was considered and ruled out: tors
does not build persistent Rust-side state, the same scope line that kept
a Merkle inclusion-proof API and a real search index out of this crate.
Every call re-describes and re-applies its steps fresh. `lemma_dict` is
the one narrow, measured exception: materializing a large caller-supplied
dict into a Rust `HashMap` is linear and, for a realistic multi-thousand-
entry lemma table, costs enough per call to matter at small batch sizes
(see `tors.CompiledLemmaDict` above): that one measured cost is why
`lemma_dict` alone accepts a pre-built handle. It does not reopen the case
for a general pipeline object: nothing else in `apply_pipeline` gets one.

**Order of operations**: `nfd` → `lowercase` → `strip_accents` →
(`stemmer` / `lemma_dict`) → `collapse_whitespace`, each step skipped
entirely when its flag is off/`None`. `nfd`/`lowercase`/`strip_accents`
are codepoint-level transforms: they don't care about word boundaries,
so they run over the whole text directly (reusing `tors.nfd`'s and
`tf_idf`'s own accent-folding algorithm verbatim). `stemmer`/`lemma_dict`
are word-level: the already-transformed text is walked segment-by-segment
(the same UAX #29 `split_word_bounds` walk `tf_idf`/`bm25_rank` tokenize
with), transforming only real-word segments and preserving every other
segment (punctuation, whitespace) verbatim, so the output stays
readable prose, not a token list. `collapse_whitespace` runs last,
reducing every run of Python-whitespace-equivalent codepoints to exactly
one ASCII space, not `tors.normalize`'s full pipeline (no
CRLF folding, no blank-line-run collapsing, no leading/trailing strip),
just whitespace-run collapsing.

**Identity contract**: all six steps default off, and
`apply_pipeline(texts)` with nothing else is a true identity: the
original `texts` list object comes back unchanged, not just
content-equal output, matching the zero-allocation contract
`normalize`/`quote`/`replace_many` already give for their own no-op case.
Argument validation (every element a `str`) still runs even on
this fast path: a non-`str` element always raises `TypeError`, never
silently passes through untouched. Empty `texts` → `[]`.

**`stemmer`/`lemma_dict`** are `tf_idf`'s exact same opt-in knobs (see its
docs for the accent-folding/stemming/lemma-substitution details, the
Snowball language list, and why lemmatization stays out of scope as a
bundled dataset). One note specific to this function: `rust-stemmers`'
`Stemmer::stem` expects already-lowercased input; passing `stemmer`
without `lowercase=True` is not an error, but the stem quality degrades:
this is not silently corrected, matching tors's "the caller composes"
posture throughout.

**Relationship to `tf_idf`/`bm25_rank`**: those two already fuse the same
`strip_accents`/`stemmer`/`lemma_dict` knobs directly into their own
tokenization. Calling `apply_pipeline` first and then `tf_idf`/`bm25_rank`
on the result tokenizes twice for no benefit: reach for their own knobs
when they're the only consumer; reach for `apply_pipeline` to preprocess
text feeding anything else (`chunk_text`, `find_patterns`, your own
logic).

A non-`list` argument or non-`str` element raises `TypeError`; an
unrecognized `stemmer` name raises `ValueError` naming every valid
choice; a non-`dict` `lemma_dict`, or one with a non-`str` key/value,
raises `TypeError`.

```python
tors.apply_pipeline(
    ["  Café  RUNNERS   are   RUNNING!  "],
    nfd=True,
    lowercase=True,
    strip_accents=True,
    stemmer="english",
    collapse_whitespace=True,
)
# [' cafe runner are run! ']

tors.apply_pipeline(["This is better, right?"], lowercase=True, lemma_dict={"better": "good"})
# ['this is good, right?']
```

## `tors.soundex` / `tors.metaphone`

```python
def soundex(text: str) -> str: ...
def metaphone(text: str) -> str: ...
```

Two classic phonetic-code algorithms, via `rphonetic` (an Apache Commons
Codec port): `soundex` (a 1918-patent-era letter-plus-three-digits code,
e.g. `"Robert"`/`"Rupert"` both encode to `"R163"`) and `metaphone` (the
Double Metaphone primary code, Lawrence Philips' 2000 successor to
classic Metaphone, e.g. `"jumped"` → `"JMPT"`). Both are
English/Latin-script-oriented heuristics, not general Unicode phonetics:
they group words that sound alike, typically alongside
`levenshtein`/`jaro_winkler` distance scoring rather than instead of it,
for name-matching and dedup pipelines.

**Input is pre-filtered to ASCII letters before encoding.** `rphonetic` 4.0.0's
`Soundex::encode` and `DoubleMetaphone::encode` both panic on ordinary
accented input, confirmed directly against the raw crate: `Soundex`'s own
"clean" step filters by the full-Unicode `char::is_alphabetic` (too
broad: Cyrillic, CJK, Greek, and accented Latin like `'é'` all pass it),
then unconditionally indexes a 26-element ASCII mapping table with
`ch as usize - 65`, out of bounds for anything outside plain `A`-`Z`;
`DoubleMetaphone` separately panics on multi-byte characters via a
byte-index slice that assumes one byte per character. Both crash on
exactly the realistic input a name-matching consumer would pass:
`Soundex::default().encode("José")` and
`DoubleMetaphone::default().encode("Björk")` both panic on the raw crate.
tors never lets a Rust panic reach Python, so `text` is filtered to ASCII
letters (`char::is_ascii_alphabetic`) before either algorithm sees it:
accents and non-Latin characters are dropped, not encoded, a documented
degradation consistent with these algorithms' documented English-only
scope even where the upstream crate doesn't crash. Empty input, or input
with no ASCII letters at all, → `""`.

`py.detach` around each call; a single `str` return (no marshalling
class).

```python
tors.soundex("Robert"), tors.soundex("Rupert")
# ('R163', 'R163')
tors.metaphone("jumped")
# 'JMPT'
tors.soundex("José"), tors.metaphone("café")
# ('J200', 'KF'): accents dropped, not crashed on
```

## `tors.double_metaphone`

```python
def double_metaphone(text: str) -> tuple[str, str]: ...
```

The full dual-key form of `tors.metaphone`: `(primary, alternate)`. The
alternate code is the algorithm's whole point: for names readable two
ways (Germanic/Slavic vs. Anglicized) it carries the second
pronunciation, so a name-matching pipeline scores a match when either
key of two names agrees; for words with one plausible pronunciation the
two elements are equal. Same English/Latin-script scope, ASCII-letters
pre-filter, and upstream-panic-avoidance note as `soundex`/`metaphone`
above. Empty input (or input with no ASCII letters) → `("", "")`.

```python
tors.double_metaphone("jumped")
# ('JMPT', 'AMPT')
```

## `tors.nysiis`

```python
def nysiis(text: str) -> str: ...
```

The NYSIIS code (New York State Identification and Intelligence System,
1970), via `rphonetic`'s strict commons-codec variant (codes capped at 6
characters). A Soundex successor with better first-letter and vowel
handling. Same scope/pre-filter/panic-avoidance note as `soundex`; the
filter additionally keeps NYSIIS keys pure ASCII, since the crate's own
clean step would otherwise let accented letters through into the code
itself. Empty input (or input with no ASCII letters) → `""`.

```python
tors.nysiis("Washington")
# 'WASANG'
```

## `tors.daitch_mokotoff`

```python
def daitch_mokotoff(text: str) -> list[str]: ...
```

The Daitch-Mokotoff Soundex codes (1985), via `rphonetic`'s port of
Apache Commons Codec with branching enabled: 6-digit codes designed for
Central/Eastern European surnames, the standard of Jewish-genealogy
surname matching, distinguishing sounds (guttural vs. sibilant) classic
Soundex conflates. Returns a list, not a single string: the rule table
branches on ambiguous transliterations, so one name can legitimately
encode to several codes; two names match if any of their code lists
intersect. Each code is padded to 6 digits, so input with no encodable
letters yields `["000000"]`, not `""`. Same
scope/pre-filter/upstream-panic-avoidance note as `soundex`.

```python
tors.daitch_mokotoff("Peters")
# ['734000', '739400']
```

## `tors.refined_soundex`

```python
def refined_soundex(text: str) -> str: ...
```

A Soundex variant with a finer-grained letter-to-digit mapping table
than classic Soundex (more consonant classes distinguished, at the
cost of a longer, uncapped code rather than Soundex's fixed
letter-plus-three-digit shape). A distinct mapping, not a
formatting variant of `tors.soundex` (confirmed: `"Robert"` encodes
differently under each). Shares Soundex's exact upstream panic bug
(confirmed directly against the raw crate: `RefinedSoundex::default()
.encode("José")` panics with an out-of-bounds table index), so it
carries the same ASCII-letters pre-filter. Empty input (or input with
no ASCII letters) → `""`.

```python
tors.refined_soundex("Robert"), tors.refined_soundex("Rupert")
# ('R901096', 'R901096')
```

## Random generation (`random_string` / `random_hex` / `random_b62` / `random_b64url` / `uuid4` / `uuid7`)

The universally hand-rolled helpers — token hex, urlsafe tokens, base62 ids,
UUIDv4/v7 — as fast GIL-free tors primitives. The Python incumbents (`secrets`,
`uuid`, `uuid_utils`) hold the GIL through their formatting passes; tors runs
the whole draw-plus-sample/format pass under one `py.detach`.

**Length-first, uniformly.** Every token spelling takes the output length
directly — "I want a base62 id X characters long" is the whole call, which is
how backend callers actually think about ids, keys, and tokens (the
maintainer's framing, and the reason `random_hex` and `random_b64url` were
moved off their byte-count spellings). `random_hex(32)` is a 32-character hex
key; `random_b62(22)` is a 22-character id; `random_b64url(43)` is a
43-character urlsafe token; `random_string(n, alphabet)` is n characters of
whatever alphabet you carry. All four are ONE char-sampling engine:
`random_hex`/`random_b62`/`random_b64url` are exactly `random_string` over
their fixed alphabets (pinned as literal equality in `tests/test_random.py`).
The uuids are the family's other engine — bit-structured byte fills handed to
the uuid crate's builders — because a UUID is a field layout, not a string
length.

**The security contract, stated first because it is the one way to misuse this
family.** Two spellings, never interchangeable:

- **Unseeded (the default)**: fresh bytes from the operating system's CSPRNG
  on every call — `getrandom`/`getentropy` through `rand`'s `OsRng`, drawn
  per call. There is no process or thread RNG state anywhere in tors, so a
  `fork()` child cannot inherit and replay a parent's stream — fork-safe,
  matching stdlib `secrets`' own per-call semantics exactly. This is the
  spelling safe for keys, tokens, and secrets.
- **Seeded (`seed=`)**: a deterministic ChaCha20 stream keyed by the seed.
  The output is a pure function of (seed, arguments), **fully predictable
  from the seed** — a reproducible-test/fixture tool, **NEVER safe for
  secrets, keys, or tokens**: any adversary who learns the seed can
  reproduce the stream (and 64-bit seeds are brute-forceable). A seeded
  stream replays exactly across processes, threads, and forks by design —
  same seed, same output — which is why it is never safe for secrets.
  rand_core's
  own docs say the same about the derivation ("not suitable for
  cryptography ... the input size is only 64 bits"). Use the unseeded
  spelling for anything an adversary must not guess.

The seeded mode is deterministic cross-platform (ChaCha20 plus Lemire
sampling plus the crate formatters are platform-independent arithmetic), so
seeded outputs are safe to pin as test fixtures; `tests/test_random.py`
pins them two ways — against committed literals, and against an
independent Python re-implementation of the documented construction
(rand_core's PCG32 seed derivation, the RFC 8439 ChaCha block, Lemire's
nearly-divisionless draw).

Sampling has **no modulo bias**: all four token spellings (the
`random_string` engine and its hex/b62/b64url delegations) draw alphabet
indices with Lemire's nearly-divisionless method (reject exactly the draws
whose product low-half falls below `2^64 mod n`, leaving every output the
same number of preimages — uniform by construction; a plain `x % n` does not
have this property). The uuids sample raw bytes, uniform by construction.
The committed goldens pin determinism (same seed, same output), never
unbiasedness on their own — a modulo transcription reproduces short
goldens without ever hitting its < 2^-58-per-draw rejection region; the
unbiasedness claim is owned by the 300k-draw chi-square/exact-count pins
in `tests/test_random.py` (whose digests a `%` transcription fails in
full) and the crate-side reject-path unit.

Errors follow the repo taxonomy, uniformly across the four token spellings:
a negative `length` raises `ValueError` naming the parameter and the accepted
form; an empty alphabet raises `ValueError`; non-`str` alphabet and non-`int`
length raise `TypeError`, as does a `seed` that is neither `None` nor int-like
(an `int` instance — bools and `IntEnum`s ride along — or any `__index__`
object: the same int-like convention `length` and every size param in the
crate accept; an `__index__` that itself raises surfaces its own error); an
alphabet holding
lone surrogates raises `UnicodeEncodeError` (the repo-wide str contract).
`0` is legal everywhere and returns `""` (`secrets.token_hex(0)`'s own
shape). There is no size cap inside the argument: `length` is `Py_ssize_t`
(2^63 - 1 on 64-bit — anything wider raises `OverflowError` at extraction),
and memory is the only bound inside that — a catchable `MemoryError`: an
impossible length is refused before any allocation is attempted
(`try_reserve`), the exact shape `'x' * n` and `secrets.token_hex(n)` give
for the same request — never a process abort.

No `tors.aio` twins: these are fast CPU/syscall calls, not the
detached-transform input class the async surface exists for
([Async use](async.md)).

```python
len(tors.random_hex(32))          # the unseeded spelling: fresh OS draw
# 32                                # every call — never a pinned literal

tors.random_hex(32, seed=42)      # the deterministic spelling: pin it in tests
# '861225d7151bf9b14a3617ab9b534d19'
```

### `tors.random_string`

```python
def random_string(length: int, alphabet: str, *, seed: int | SupportsIndex | None = None) -> str: ...
```

`length` characters sampled uniformly (Lemire, no modulo bias at any
alphabet size) from any non-empty `alphabet` str, in one GIL-released pass.
Multibyte characters are sampled as characters: the output is always exactly
`length` characters over the alphabet's own characters.
Duplicate characters are weighted, not deduplicated — each position is an
independent draw over the alphabet's character positions, so `"aaab"`
yields `a` with probability 3/4: dedupe the alphabet first for
uniform-over-distinct-characters. Sampling is over Unicode scalar values,
not grapheme clusters: a combining mark in the alphabet samples
independently of its base character. The alphabet is materialized fresh on
every call (no cross-call cache, by the fork-safe no-state discipline);
cost is O(alphabet) to materialize plus O(length) draws. Bulk callers
reusing one huge alphabet across many calls should prefer the
stdlib. The only size ceiling is the argument itself (`Py_ssize_t`) and
memory beyond it (`u64`-based sampling handles any alphabet a `str` can
hold).

```python
tors.random_string(12, "abcdef", seed=42)
# 'dcabacecacae'
```

### `tors.random_hex`

```python
def random_hex(length: int, *, seed: int | SupportsIndex | None = None) -> str: ...
```

`length` lowercase hex characters — exactly
`random_string(length, "0123456789abcdef")`, one engine — the hex-key
spelling, by output length: a 32-char key is `random_hex(32)`.

Odd lengths are legal and first-class: a 31-char hex id is a real shape, and
`random_hex` produces it directly (the old byte-count spelling could not ask
for it at all). Even lengths are what digest-shaped keys want — every 2
characters are exactly one byte, so `random_hex(2 * n)` is byte-exact
material. `secrets.token_hex(n)` and `random_hex(2 * n)` are the same uniform
distribution over 2n-char hex strings — different draws, never different
contracts; the `2 *` is the whole mapping (the stdlib thinks in bytes, this
spelling in characters).

```python
tors.random_hex(32, seed=42)
# '861225d7151bf9b14a3617ab9b534d19'
```

### `tors.random_b62`

```python
def random_b62(length: int, *, seed: int | SupportsIndex | None = None) -> str: ...
```

Exactly `random_string(length, BASE62_CHARS)` — one engine, delegated — over
`[0-9A-Za-z]`: the URL-safe, case-sensitive, human-transcribable id
spelling.

```python
tors.random_b62(22, seed=0)
# '1yrBtE6FUlG59Zjj3K2vVn'
```

### `tors.random_b64url`

```python
def random_b64url(length: int, *, seed: int | SupportsIndex | None = None) -> str: ...
```

`length` characters uniform over the 64-character RFC 4648 §5 urlsafe
alphabet (`A-Za-z0-9-_`; `+` and `/` never), every position unconstrained —
exactly `random_string(length, B64URL_CHARS)`, one engine. The
opaque-token spelling: a 43-char urlsafe token (the JWT-signature shape) is
`random_b64url(43)`, and `secrets.token_urlsafe(32)` formats into the same
43-character length class.

**The boundary, stated honestly: this is the token contract, NOT "a valid
base64 encoding of N random bytes."** An encoding's final character is
constrained (an encoding of 32 bytes can only ever show 16 distinct final
characters at length 43 — the final char carries the final byte's low 4
bits, shifted into place; the 31-byte encoding is the 4-distinct case, its
final char carrying just the low 2 bits; this spelling shows all 64), and
lengths that are not valid base64 output lengths (41, 45, ...) are legal
here — a uniform token has no alignment constraint. There is no `padded=`
parameter: padding is an encoding concept, not a token concept, and `=`
never appears. Callers who want encodable random material should take
`random_hex` of even length — byte-exact via hex.

```python
tors.random_b64url(43, seed=7)
# 'Bi8SLaf0s_a4pi-vqthbTaOstZjDweDcEC5hW7S_CNp'
```

### `tors.uuid4`

```python
def uuid4(*, seed: int | SupportsIndex | None = None) -> str: ...
```

An RFC 4122 version-4 UUID string (36 chars, lowercase, hyphens at
8/13/18/23) from one 16-byte entropy fill — the stdlib `uuid.uuid4()`
spelling, GIL-released, with the family's optional deterministic mode.
Uniqueness is probabilistic (122 random bits), the same guarantee
`uuid.uuid4()` carries.

```python
tors.uuid4(seed=42)
# '7848b5d7-11bc-4883-9963-17a3f9c90269'
```

### `tors.uuid7`

```python
def uuid7() -> str: ...
```

An RFC 9562 version-7 UUID string: 48-bit Unix-epoch milliseconds + version
7 + variant + 74 random bits (12-bit `rand_a` + 62-bit `rand_b`) from one
OS draw — the sortable-id spelling. **No `seed=` parameter, by design**: the
timestamp is external state, so a seeded uuid7 would still vary with the
clock — the deterministic tool is `uuid4(seed=...)`.

**The uniqueness boundary, stated honestly**: probabilistically unique
(birthday bound over 74 random bits within a millisecond, distinct
timestamps across milliseconds), **NOT counter-monotonic**. Two calls within
one millisecond are ordered by their random bits, not by call order; a
backwards clock step flows straight into the timestamp. `uuid_utils`'
strict-monotonic counter is a different product promise — its engine keeps
process-local counter state (and pays no syscall per call, which is why it
benchmarks faster; see [Performance](performance.md)), where tors draws
fresh OS entropy per call for the same fork-safety the rest of this family
carries. The caller-visible contract is the timestamp itself:

```python
u = tors.uuid7()
u
# '01a09927-e261-7153-8d74-283f6d488162'
int(u[:8] + u[9:13], 16)  # the call's Unix epoch milliseconds
# 1789275923041
```

### `tors.uuid4_bytes` / `tors.uuid7_bytes`

```python
def uuid4_bytes(*, seed: int | SupportsIndex | None = None) -> bytes: ...
def uuid7_bytes() -> bytes: ...
```

The uuids' 16 raw bytes — the same buffers `uuid4`/`uuid7` format, version
and variant fields set, **no canonical formatting** — because the surveyed
consumers re-wrap the canonical str back into bytes anyway: id-assignment
call sites construct `UUID(bytes=...)` from the str's decoded bytes, and
timestamp slicing takes `.hex()[:12]`. The str spellings force those sites
into a format-then-reparse roundtrip (36-char format, hyphen strip, hex
re-decode); the bytes spellings are one native draw and the field layout,
no roundtrip:

```python
u = uuid.UUID(bytes=tors.uuid7_bytes())  # the re-wrap shape, native
u.version, u.variant
# (7, RFC_4122)

tors.uuid7_bytes().hex()[:12]  # the timestamp-slice shape
# '01a09927e261'               # the call's Unix epoch milliseconds, hex
```

Contracts: `uuid4_bytes(seed=s)` yields exactly the 16 bytes whose
canonical hyphenated form is `uuid4(seed=s)` — one construction, two return
spellings, pinned as literal equality in `tests/test_random.py` (the seeded
bytes are pure functions of the seed, pinned against the same independent
oracle). `uuid7_bytes()[:6]` big-endian is the call's Unix-epoch
milliseconds (`int.from_bytes(b[:6], "big")` — the `u[:8] + u[9:13]`
field's own bytes), the same 5-second clock tolerance `uuid7`'s timestamp
contract carries. The seed contract and security warning are `uuid4`'s own
(any int-like, reduced mod 2^64; a seeded stream is fully predictable —
never for secrets), and `uuid7_bytes` takes no `seed=` for `uuid7`'s own
reason. Fixed 16 bytes always: there is no length argument, so the token
spellings' memory bound does not exist here. No `tors.aio` twins (the
family's own rule: fast CPU/syscall calls).

## `tors.first_invalid_charset`

```python
def first_invalid_charset(
    items: Sequence[str], *, first: str | None = None, rest: str
) -> int: ...
```

Batch codepoint-set validation for identifier-style rules, one GIL-released
pass per batch: the index into `items` of the first item not built entirely
from the caller's two sets — `first` the set of codepoints allowed at
position 0, `rest` the set allowed at every position after it (and at
position 0 too when `first` is `None`, the uniform spelling: one set
everywhere) — or `-1` when every item passes. An empty batch answers `-1`
even under spellings where every item would offend (`first=""`,
`rest=""`): vacuously valid, no items to offend. An empty item is an offender,
wherever it sits. `first=""` allows nothing at position 0 (every item
offends); `rest=""` allows nothing after position 0 (under the uniform
spelling every item offends; with a non-empty `first`, only single-codepoint
items drawn from `first` can pass). The scan short-circuits at the first
offender — no promise about work done past it, though the argument walk does
validate the whole sequence up front (a bad entry anywhere raises at the
boundary, past a first offender or not). The answer is always an item index,
never a position within an item. The argument walk is bounded, the same DoS
backstop `content_hash`'s protocol walk carries: a sequence that yields past
the walk's ceiling (1M items; legitimate batches sit orders of magnitude
below it) aborts with a `ValueError` naming nothing, instead of holding the
GIL while an unbounded `__iter__` (never trusted: the walk iterates, it does
not consult `len`) grows the batch until the process dies.

The sets are **data, not patterns**: plain strings of permitted codepoints,
membership per codepoint (never per byte; duplicate codepoints in a spelling
are harmless and their order is irrelevant). No ranges, no escapes, no
classes — an `^[a-z_][a-z0-9_]*$`-style rule is spelled by listing the
codepoints — and no Unicode-category classes (`\w`), which would need
property tables: the charter boundary already drawn for lexical data
([design and scope](design.md)). A rule that needs `\w` is not expressible;
a rule that needs specific codepoints is, whatever their script.

Membership is per scalar value, with no normalization: precomposed `é`
(U+00E9, one codepoint) and decomposed `e` + U+0301 (two codepoints) are
different inputs and get different verdicts — `first_invalid_charset(["é"],
rest="é")` is `-1` while `first_invalid_charset(["é"], rest="é")`
is `0`, correctly per the per-codepoint engine. When NFC and NFD forms
must agree, normalize (`tors.normalize` / `tors.nfc`) before validating.
Normalization does not fold confusables (visually similar but distinct
codepoints, e.g. Latin A vs Cyrillic А, stay distinct under NFC); allow-list
exactly the codepoints you mean.

Membership is per codepoint, not per grapheme cluster: a ZWJ family emoji
(5 codepoints), a flag pair (2 regional indicators), or a base letter plus
a combining mark (2 codepoints) each pass only when every constituent
codepoint is in the set — one grapheme can still offend.

**Batch-only by design, and honestly so.** The motivating consumer (an
enqueue path) validates identifier-shaped strings with anchored regexes at
84–950 ns per call, each under a single `py.detach` round trip — so a
per-item tors call (~0.25 µs measured, against ~0.08 µs for one compiled
regex match) is *slower* than the regex it would replace. The win exists
only when one call covers the batch: one detach, one pass over all items.
Measured on the dev box (min-of-7): the crossover is at single-digit item
counts (~2x at a 10-item batch), ~5x at 100 items (2.0 µs vs the per-item
regex loop's 9.7 µs), ~7x at 1000 (14.0 µs vs 96.7 µs). The Rust core alone
measures ~64 ns fixed (the two set builds) plus ~5.4 ns per item
(post-`#[inline]` band, `first_invalid_charset` group, `benches/search.rs`).
Do not call this once
per string; call it on the batch.

```python
import string

ident_first = string.ascii_letters + "_"
ident_rest = ident_first + string.digits
tors.first_invalid_charset(["job_42", "queue_eu", "9bad"], first=ident_first, rest=ident_rest)
# 2   ("9bad": a digit is not allowed at position 0)
tors.first_invalid_charset(["job_42", "queue_eu"], first=ident_first, rest=ident_rest)
# -1  (every item passes)
tors.first_invalid_charset(["worker:01", "tag name"], rest=ident_rest + "-:.")
# 1   (first=None: one set everywhere; "tag name" carries a space)
```

Argument contract: `items` is a sequence of `str` — a `list`, a `tuple`, or
any `Sequence` (a bare `str` raises `TypeError`, as do non-sequences: dict,
set, generators, scalars; a non-`str` entry raises `TypeError`); `first` and
`rest` must be exactly `str`, both keyword-only, `rest` required, `first`
defaulting to `None`; lone surrogates raise `UnicodeEncodeError` at the
argument boundary (the standard str-in boundary, paid by every item and both
set arguments). Error precedence on both-bad calls is pyo3 left-to-right
extraction order: `first` beats `rest`, and set-argument conversion errors
beat the items-walk entry errors. The contract is proven against a pure-Python membership-loop
oracle plus the equivalent anchored regexes rebuilt from the same set halves
over hypothesis-generated inputs (tests/test_first_invalid_charset.py).

GIL model: one GIL-held walk of the items sequence (the standard str-in
borrow class, O(items) handles; the one-time O(input) UTF-8 materialization
applies per non-ASCII item object on first extraction, so a batch of
never-before-touched non-ASCII items holds the GIL for that
materialization — unmeasured for this function: no dedicated non-ASCII cell,
same str-in materialization class as every other function), then the set builds and
the whole batch scan under one `py.detach`, then a single int return — no
marshalling class at all. The set builds are O(set) over the ASCII spelling
plus O(set log set) over the non-ASCII tail (sort + dedup, inside the
detach): negligible for the few-dozen-codepoint ASCII rules this validator
exists for (the ~64 ns fixed band above, which includes both per-call set
builds — hoist huge set spellings to module constants: define once, reuse
the same string), a real sort for a 10k-codepoint
non-ASCII spelling whose sort cost is deferred (correctness pinned,
unmeasured beyond the ASCII band). At realistic batch sizes the whole call sits far under
the 10 ms ping floor, so the GIL cell is ceiling-only (the `utf8_is_valid`
class), pinned in tests/test_gil_release.py. The core is a trivial one-pass
membership walk (one-bit-per-codepoint ASCII mask plus a sorted non-ASCII
tail; no index arithmetic, no `unsafe`), which is why it ships no
cargo-fuzz target: the hypothesis differentials over arbitrary Unicode
cover its input space. The Rust core band is the one above — ~64 ns fixed
(the two set builds) plus ~5.4 ns per item, post-`#[inline]`
(`first_invalid_charset` group, `benches/search.rs`).

### Building rejection messages: `tors.first_invalid_offender`

```python
def first_invalid_offender(
    items: Sequence[str], *, first: str | None = None, rest: str
) -> tuple[int, int, str] | None: ...
```

The integration survey's finding made the detail a function: the
consumer's rejection UX names the losing CHARACTER and POSITION (its
per-character messages), and an item index alone cannot. The offender
spelling is the same scan answering that question: it returns
`(item_index, char_position, offending_char)` for the first offending
item's FIRST offending position, `None` when every item passes.

- `char_position` is a **codepoint index** within the item — the family's
  data model is codepoints (membership is per codepoint), so positions
  count codepoints, never UTF-8 byte offsets: in `"🦀x"` the `x` sits at
  position 1 (its byte offset would be 4). `offending_char` is that
  codepoint as a 1-char `str`, a 4-byte astral codepoint included.
  `char_position` may land inside a grapheme cluster — the flag-partial
  `(0, 1, "🇷")` under `rest="🇫"` is one grapheme but offends at codepoint
  1 — so do not slice the item at that position for display (it would split
  the grapheme); build messages from `(item, char)`, which the tuple already
  carries.
- **The empty item reports `(i, 0, "")`**: an empty item is an offender
  with no codepoint at position 0 to name, so the char field is empty
  exactly when the item is — a message builder branches on it.
- The consistency invariant with the int spelling is structural — the
  Rust core's ONE scan returns the stop detail and each spelling projects
  it — so `first_invalid_offender(...)` is `None` exactly when
  `first_invalid_charset(...)` answers `-1`, and when not `None` the
  tuple's first field IS the int spelling's answer. The differential
  battery pins it against the reference oracle extended to return the
  detail (tests/test_first_invalid_charset.py).
- A `== -1` / `!= -1` test ported from the int spelling does not
  transfer: a tuple never equals `-1`, so an `!= -1` invalidity guard
  fires on every batch and an `== -1` validity guard never does, both
  silently. The clean spelling is `is None` / `is not None`, pinned on
  both branches (clean AND offending) in tests/test_first_invalid_charset.py.

```python
import string

queue_first = string.ascii_letters + string.digits + "_"
queue_rest = queue_first + ".-"
names = ["jobs_eu", "foo:eu", "queue_us"]

offender = tors.first_invalid_offender(names, first=queue_first, rest=queue_rest)
# (1, 3, ":")   ("foo:eu": ":" is not allowed at position 3)

def rejection(names, offender):
    if offender is None:
        return None
    item, position, char = offender
    if not char:  # the empty item: no offending character to name
        return f"queue name {names[item]!r} is invalid: it is empty"
    return f"queue name {names[item]!r} is invalid: {char!r} at position {position} is not allowed"

rejection(names, offender)
# "queue name 'foo:eu' is invalid: ':' at position 3 is not allowed"
```

The argument contract is the int spelling's exactly — the same shared
walk, so every boundary refusal (a bare `str`, non-sequences, non-`str`
entries, non-`str` `first`/`rest`, lone surrogates; the
walk-validates-the-whole-list-first precedence) raises the same exception
with the byte-identical message, pinned by raising both spellings on the
same bad input and comparing them. Error precedence on both-bad calls is
pyo3 left-to-right extraction order: `first` beats `rest`, and set-argument
conversion errors beat the items-walk entry errors. No new error classes.

One engine, one scan, and the int path pays nothing for the detail: the
walk stops at the first offending codepoint, and at that stop point it
already holds the position and the codepoint — the detail is free by
construction, the only addition to the shared walk being the position
counter (one integer add per codepoint, a dead field in the int
projection). The scan is `#[inline]`, so each projection compiles away
the fields it drops: without the inline the multi-word stop state
crosses a call boundary and the int spelling paid ~10% on the criterion
group's small cells for detail it discarded (measured); with it the band
is back — the cells above are the post-extension numbers. The int
spelling's performance cells (the wall race, the criterion group, the
GIL ceiling cell) carry the family's contract unchanged, and this
spelling adds no cells of its own: same engine, same walk, same
one-detach batch pass, the only delta being the return — one small
tuple, built only when an offender is found.

### Common alphabets

The scope question, answered up front: *is it worthwhile adding any other
charset validators — b62, b64, hex, UUID?* No new validators. Named wrapper
functions (`is_valid_b62`, `is_valid_b64url`, ...) would each delegate to
the same core — N wrappers of zero performance gain, pure API surface and
maintenance cost — so the generic `first_invalid_charset` stays the single
engine. What *is* worth shipping is the data: the alphabets long enough
that re-spelling them at every call site invites silent transcription
errors (a wrong 62-character set still validates *something*). Those ship
as pinned module constants — data, not code:

| constant | alphabet | length | for |
|---|---|---|---|
| `tors.CHARSET_B62` | `0-9 A-Z a-z` | 62 | base62 ids |
| `tors.CHARSET_B64URL` | `A-Z a-z 0-9 - _` — RFC 4648 §5, **unpadded** | 64 | JWT segments, url-safe tokens |
| `tors.CHARSET_HEX_LOWER` | `0-9 a-f` | 16 | lowercase hex digests |
| `tors.CHARSET_HEX_UPPER` | `0-9 A-F` | 16 | uppercase hex digests |
| `tors.CHARSET_HEX_MIXED` | the 22-codepoint union of the two hex alphabets | 22 | case-insensitive hex digests |

The `first=None` uniform spelling (one set at every position) makes each
constant a single argument:

```python
ids = ["7xK9mQ2pZv", "0Zz8", "bad!"]
tors.first_invalid_charset(ids, rest=tors.CHARSET_B62)
# 2   ("bad!": "!" is outside the base62 alphabet)
segments = ["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "eyJzdWIiOiIxMjM0NTY3ODkwIn0="]
tors.first_invalid_charset(segments, rest=tors.CHARSET_B64URL)
# 1   (the second segment is padded — the OUT case below)
```

Three shapes are deliberately OUT, each for a structural reason:

- **Standard/padded base64.** The `=` padding is positionally structured —
  terminal only. A flat charset cannot express "positions `0..len-2` from
  the alphabet, position `len-1` optionally `=`", and a charset that
  admitted `=` would wrongly accept mid-string padding. `tors.b64_decode`
  IS the strict base64 validator — decode-as-validation, which raises on
  misplaced padding, wrong lengths, and non-alphabet codepoints alike, is
  the right tool. That is why `CHARSET_B64URL` is the unpadded alphabet
  and the padded segment above is an offender, not a pass.
- **UUID.** The hyphens at fixed positions 8/13/18/23 are structure, not
  charset; `first_invalid_charset` cannot express them. There is no
  strict UUID validator in this release — hyphens are structure,
  tracked on the uuid7-helpers branch — not a flat alphabet.
- **Digits.** `"0123456789"` is trivially spelled with zero typo risk; not
  worth a name.

The constants' contents are contract, pinned byte-exact with the
length/uniqueness/subset algebra that makes the family coherent
(tests/test_first_invalid_charset.py); the stub carries their type (`str`)
and is held to `tors.__all__` by the same drift guard as every function
(tests/test_pyi_drift.py).

## `tors.documents`

Document-format extraction: PDF, the office and text formats (doc/docx, xls/xlsx,
ppt/pptx, rtf, odt/ods/odp, epub, csv/tsv), and HTML, converted to GitHub-Flavored
Markdown or plain text, one GIL-released native pass per call, the same discipline as
every function above.

This surface ships in a second wheel: `pip install tors[documents]`. The base `tors`
wheel re-exports it as `tors.documents` (an `ImportError` with the install hint fires
when the payload is absent), and the payload (`tors-documents`) is version-locked to
`tors`: same number, released together. The split is weight discipline: the engines
live in `tors-core` behind the cargo feature `documents`, default OFF, so the base
build compiles none of them; only the payload wheel does.

The split is also the lazy-import design, and the laziness is measured (fresh
processes, `/proc/self/status` `VmHWM`): a bare `python` peaks at
11.9 MB; `import tors` at 14.7 MB, with `tors.documents` absent from `sys.modules`
(the base wheel carries no engine code at all, so the import has nothing to touch;
a subprocess gate in the suite pins the absence); `import tors.documents` at
18.7 MB despite all four engines being compiled into the one payload `.so`, by
demand paging: the engine code pages only materialize as conversions first run. You
pay nothing for documents unless you install and import them.

The engines are chosen per format family by head-to-head measurement (the
comparison and its fixtures are documented in the `documents_impl` crate docs, and the
suite that pins them is `tests/test_documents_engines.py`):

| format family | engine | why (measured) |
|---|---|---|
| PDF | pdf_oxide 0.3.78 | two-column layouts come back as separate reading-order blocks (not interleaved), `/Link` annotations render as `[text](uri)`, heading detection on |
| HTML | html-to-markdown-rs 3.12 | drops `<script>`/`<style>` by construction (the disqualifying failure of the alternatives, which leak CSS/JS text into the body); padded GFM tables, indented nested lists, clean code fences |
| office + text (doc/docx, xls/xlsx, ppt/pptx, rtf, odt/ods/odp, epub, csv/tsv) | anydoc 0.2.4 | renders style-based docx headings and list markers that office_oxide drops entirely; covers rtf/odt/epub/csv, which office_oxide cannot read at all |
| `backend="oxide"` (caller-selectable) | office_oxide 0.1.10 | the alternative reader for the OOXML + legacy office formats: exact entity text (no `&`-escaping), against the heading/list losses above; a documented lane for diffing the two engines on your own corpus, never the default (decompression posture: 512 MiB per-part caps, no total-across-parts or output cap; the measured cases are in the `max_bytes=` paragraph below) |

**The input is `path` OR `data`**: every path-taking function also accepts the
document as `data=` bytes, the in-memory caller's entry, so an upload already held
as bytes converts with no temp-file roundtrip. Exactly one of the two (both →
`ValueError`, neither → `TypeError`, a non-bytes `data=` → `TypeError`, all raised
under the GIL before any work runs). A `data=` call has no file name, so format
resolution rests on `format=` and the content markers alone.

**GIL model, every function in this section**: argument marshalling
(validation) happens under the GIL; the one O(n) bytes copy a `data=` call
pays (a borrow cannot cross `py.detach`) rides inside the detach with the
rest of the pass (a 400 MB `data=` call's max
heartbeat gap is ~1.1 ms under a 1 ms ping, where a GIL-side copy starved
the same ping for ~78 ms), and the whole native pass (file read when
`path=`, format sniff, engine conversion, and for `to_text` the markdown
strip) runs inside one `py.detach`, and exceptions are constructed after
the GIL is reacquired; nothing raises from inside the detached region. The
hazard this removes is concrete: the official pdf_oxide pyo3 wheel
measures as GIL-held per call (worst heartbeat gap 23.6ms on a 9-page
document under a 10ms ping, growing with document size); this
payload calls the crate's Rust API directly under `py.detach` instead. The
band is pinned by the suite
(`tests/test_documents_engines.py` and `tests/test_pdf.py` hold the
heartbeat-granularity and 8-thread byte-identical concurrency gates).

**Error taxonomy** (shared by every path-taking function here):

| exception | raised when |
|---|---|
| `OSError` | the file is missing or unreadable (IO): a missing path is `FileNotFoundError`, a directory `IsADirectoryError` on Linux (the matched subclass, errno text in the message) |
| `TypeError` | neither `path` nor `data=` was passed; a non-str `path=` (`to_markdown(123)` names `path`, never os.fspath's bare error); a non-bytes `data=` (the refusal names the type only, never the value's content or a heap address, the `password=` doctrine); a wrong-typed `pages=` entry (a bool, float, or str where a 0-based int belongs); a non-str `backend=` (a bool, int, float, or bytes: `b"anydoc"` is a str-shaped value of the wrong type, not a lane name); a non-int `max_bytes=` (a bool, str, or float; a bool would launder through an int extraction as 1, so the type is refused first); a non-str `password=`; all argument-contract failures, raised under the GIL before any work runs |
| `ValueError` | an unknown `format=` name; content and extension both fail to name a format; a `backend=`+format pair the forced engine cannot read; on the PDF-only family, `backend="anydoc"`: a capability refusal (not a format one: PDF+anydoc converts) naming the per-page surface the call needs and the `to_markdown`/`to_text` pair that is anydoc's whole PDF surface, raised before any work runs; an invalid `pages=` selection; a malformed document; an encrypted PDF without its `password=` (every entry fails closed: the door check raises at open); an input over `max_bytes=` (an explicit budget binds every lane, the PDF-only family included, checked before a byte is read or copied; under the default the metered anydoc, office_oxide, and HTML lanes refuse during the read at the 32 MiB ceiling, and the pdf lane reads under the 512 MiB backstop); a non-regular `path=`: a FIFO, device, or socket is a typed refusal naming `path` and the kind, before open(2) can block (directories keep their `OSError` above); a NUL byte inside `path=` (CPython's own `open("a\0b")` convention, naming `path`) |
| `NeedsOcrError` (a `ValueError` subclass) | the anydoc backend hit a PDF with scanned/image-only pages: route the document to an OCR stage |

**Format resolution order**: a mislabeled or extensionless file (a temp-file download,
say) still converts, because content, not the name, picks the extractor:

1. an explicit `format=` (extension spelling, no dot, case-insensitive);
2. the content markers: the binary signatures first (PDF header, RTF open group, OLE
   stream names, ZIP package mimetype), then the HTML document marker (after a BOM and
   whitespace, the first markup is `<!DOCTYPE html` or `<html`, case-insensitive), then
   the CSV heuristic, the last resort of content resolution: text, not markup, where
   the first up-to-64 non-empty lines each carry the same count (≥1) of one delimiter
   candidate (`,` / `;` / TAB, tried in that order), two lines minimum;
3. the input name's extension, the `path=` when there was one (`data=` has no
   name; a one-line `.tsv` or an `.xhtml` fragment resolve here, through the same
   name vocabulary `format=` uses);
4. else `ValueError`: nothing names a format.

**The typed surface**: `Backend`/`Format`/`PageKind` are `str` enums: each member is
its accepted string, so `format="docx"` and `format=Format.DOCX` are the same call, and
plain strings the Rust validator accepts (container variants like `"docm"`/`"xlsm"`)
keep working without enum churn. `to_markdown`/`to_text` return the resolved format as
a `Format` member; `sniff` returns `Format | None`. The rules live native; the typed
view adds typing only, never a second copy of a rule.

**One page convention, everywhere**: every page number this surface names is a 0-based
index: `pages=`, `PdfClassification.pages_needing_ocr`, `pdf_extract`'s list positions,
and `NeedsOcrError.pages`. anydoc internally reports 1-based page numbers; the core
re-bases its list once, at the seam where the engine's answer crosses into this API, so
a caller routing pages to OCR never has to remember which list carries which convention
(a mixed-convention API is a silent off-by-one aimed at exactly that caller).

## `tors.documents.to_markdown` / `tors.documents.to_text`

```python
def to_markdown(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = None,
    backend: Backend | str = Backend.AUTO,
    pages: int | list[int] | tuple[int, int] | None = None,
    password: str | None = None,
    max_bytes: int | None = None,
) -> tuple[Format, str]: ...


def to_text(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    format: Format | str | None = None,
    backend: Backend | str = Backend.AUTO,
    pages: int | list[int] | tuple[int, int] | None = None,
    password: str | None = None,
    max_bytes: int | None = None,
) -> tuple[Format, str]: ...
```

Convert any working-format document to GitHub-Flavored Markdown (`to_markdown`) or
plain text (`to_text`), returning `(format, output)` where `format` is the format the
conversion actually used (a `Format` member). The document is `path` (a file, the
only positional, a regular file: FIFOs/devices/sockets are refused before the read)
or `data=` (its bytes: the same conversion, byte-identical output;
no name to consult, so resolution rests on `format=` and the content markers).
`format=` names the format explicitly (`"pdf"`, `"html"`/`"htm"`/`"xhtml"`,
`"docx"`, `"xlsx"`, `"pptx"`, `"doc"`, `"xls"`, `"ppt"`, `"rtf"`, `"odt"`, `"ods"`,
`"odp"`, `"epub"`, `"csv"`, `"tsv"`, plus the container variants
`"docm"`/`"xlsm"`/`"ppsx"` mapping onto these: the same OOXML packages with the
content-type override naming the macro/show variant, resolved and sniffed as their
base kinds; `"xlsb"` routes the Excel kind as vocabulary sugar, but genuine xlsb
content is BIFF12 `.bin` sheets, not worksheet XML, and is refused: the engines do
not read it); `None` (the default) resolves
it by the order above, content markers first, the input name's extension last.

`backend=` picks the engine where they overlap: `"auto"` (the default) routes by the
measured table; `"oxide"` forces pdf_oxide for PDF and office_oxide for the
OOXML/legacy office formats; `"anydoc"` forces anydoc. A forced backend raises
`ValueError` on a format that engine cannot read, never a silent fallback.

`pages=` selects a PDF page subset, and is valid on the pdf_oxide lane only (any other
format, or `backend="anydoc"` on a PDF, raises `ValueError`):

- a single `int`: one 0-based page;
- a `list` of ints: the explicit set;
- a 2-tuple `(start, stop)`: a half-open range of 0-based page indices (`(0, 2)` on a
  two-page document is both pages; `(1, 3)` selects the pages at indices 1 and 2).

The refusals split by Python's own convention, and the suite's red-team lane pins
the split: a wrong type is `TypeError`, a bool, float, or str where a page index
belongs (`pages=[True]`, `[1.5]`, `["a"]`, `(True, 2)`; `range(1.0)` and `seq[1.5]`
raise `TypeError` in the stdlib too, and a bool would otherwise launder through
pyo3's i64 extraction as page 0 or 1); a wrong value or shape is `ValueError`, a
negative index, an empty list, an empty or backwards range, a tuple that is not the
`(start, stop)` pair, an int too large for the i64 the binding extracts
(`pages=[2**70]`). Both classes raise under the GIL, before any native work runs,
each repr'ing the offending value; bounds are validated against the
real page count inside the detached pass. The selection is deduped into document
order (caller-supplied order and repeats are normalized away), the per-page
conversions are joined with pdf_oxide's own inter-page separator, and a full range is
byte-identical to the whole-document conversion.

`password=` unlocks an encrypted PDF, the PDF kinds only (a password on any other
format is `ValueError`, "password= applies to PDF documents only": a
silently-ignored password would leave the caller believing a document is protected
on a lane that cannot know; a non-str `password=` is the same `TypeError`
convention as every argument here). Without it, an encrypted PDF fails closed on
every entry: the door check raises at open, `ValueError`, "PDF is encrypted and
requires a password", never empty output masquerading as "no content" (the
pre-fix shape, measured on an RC4-128 fixture: `to_markdown`/`pdf_extract` returned
`""` on a locked document while `pdf_classify` raised, the two entries disagreeing
about the same bytes; the check now raises at open for all). A wrong password is its
own clean `ValueError`, "the password did not unlock this PDF". The unlock rides
the pdf_oxide lane, the default `auto` and forced `oxide` both; `backend="anydoc"`
on a PDF has no unlock and refuses encrypted documents outright.

`max_bytes=` is the input ceiling, and its contract has two halves. An explicit
`max_bytes` binds every engine lane, pdf and HTML included, and is enforced
before any work runs: the file's size at open (a `path=` call never reads an
over-budget byte), the buffer's length on entry (a `data=` call never copies
one). `max_bytes=None` (the default) is the two-phase doctrine: the read is
bounded from the start, a prefix sniff picks the lane, and then the metered
lanes — the ones that amplify input into resident
memory: anydoc, office_oxide, and HTML (~23x input; the HTML converter holds
the whole input and output at once, a 48 MiB doctype HTML peaked at 1118 MiB)
— refuse *during* the read at the 32 MiB default ceiling, while the
pdf lane reads under the finite 512 MiB backstop (a memory-safety
floor, not a lane policy). The lane is unknowable before the container sniff,
which is why only the explicit budget can be checked fully pre-read. Over the
ceiling, the call raises `ValueError` naming the ceiling and the `max_bytes=`
override.

The anydoc amplification, restated at the measured worst case: a many-short-cells
csv amplifies ~146x (a 24 MiB one peaked at 3.4 GiB
RSS, stable across input sizes; the earlier "~36x" figure was a benign
long-cell shape, and short cells are the common upload and the expensive one), so
the 32 MiB default budgets ~4.6 GiB of worst-case headroom on the converting
worker; tighter is often right, and `max_bytes=` is the knob. The motivating
integrator shape, a service capping uploads at 100 MB, passes
`max_bytes=100 * 1024 * 1024` and must budget for the worst case at that
ceiling: ~146x of 100 MiB is ~14 GiB of RSS headroom on the converting worker
(benign csv shapes measure far lower, ~36x; budget for the worst case, not the
benign one).

The opt-in `backend="oxide"` lane's decompression posture, measured on
office_oxide 0.1.10 (locked): per-part caps of 512 MiB, declared and actual,
refused pre-decompression (a 600 MiB declared part is refused with
"decompression limit exceeded: part 'word/document.xml'
expands to more than 536870912 bytes"), plus an XML nesting cap of 256 on a
16 MiB parse stack. It has no total-across-parts cap and no output cap: a
399 KiB zip carrying a 400 MiB `word/document.xml` (under the per-part cap)
converts at ~1.6 GiB peak RSS, emitting ~400 MiB of markdown, so
multi-part and output blowups remain the caller's risk on that lane, which is
one reason it is never the default. (The older "333 KiB zip-bomb docx →
1.7 GiB" figure was office_oxide 0.1.9, before the per-part caps, and is
history, not current posture. anydoc, by contrast, caps decompression engine-side:
128 MiB per entry, 512 MiB total; a zip-bomb fixture lane in the suite pins
both engines' caps firing.)

`to_text` is the same conversion, routing, and `pages=`/`password=`/`max_bytes=`
semantics, with the markdown normalized to plain text, one text shape for every
format and engine: headings keep their text (markers dropped), list items keep
indentation and numbering, table rows join their cells with `" | "`, code blocks
keep their content without fences, links become `label (url)`. The normalization
runs inside the same detached pass, over the markdown, because each
engine's own plain-text surface differs (pdf_oxide's, measured, merges two-column
layouts line-by-line; the strip preserves the markdown converter's reading-order
blocks). The strip's inline machinery (link labels, image labels, emphasis)
recurses per nesting level and is depth-bounded at 256: past the bound the
remaining `[…](…)` machinery degrades to literal text instead of recursing toward
a stack overflow: a 30,000-deep `[[[…x…]]()…]()` nest converts (exit 0,
non-empty output) where the unbounded strip crashed the process (the fix's
subprocess pin is in `tests/test_documents_engines.py`; the exact degradation
shape is unit-pinned in `src/gfm_strip_impl.rs`).

A multi-sheet workbook renders whole: anydoc emits each sheet as its own
`## <sheet name>` section (a measured two-sheet workbook, an `Alpha` table over a
`Beta` one, comes back as `## Alpha\n\n|...|\n\n## Beta\n\n|...|`), so per-sheet
output is the caller's split on the `## ` headings. A single-sheet workbook
renders with no `##` heading at all (the table is the whole output; the
committed engines_samples.xlsx does exactly that), so that split must tolerate
its absence. There is no
sheet-selection argument: whole-document output is the shape downstream callers
consume.

```python
import tors.documents

fmt, markdown = tors.documents.to_markdown("tests/engines_corpus/engines_page.html")
# (Format.HTML, "# Annual Engineering Report\n\n## Transformer Program\n\nSee the
#  [field handbook](https://handbook.example.com/torque) for torque tables.\n\n...")

fmt, text = tors.documents.to_text("tests/engines_corpus/engines_link.pdf")
# (Format.PDF, "Visit the field handbook (https://handbook.example.com/guide)\n")
#  the /Link annotation survived as `label (url)`: the plain-text link shape

fmt, text = tors.documents.to_text("tests/engines_corpus/engines_units.csv")
# (Format.CSV, "unit | status\nT-101 | healthy\nT-102 | needs review\n")
#  table rows join their cells with " | "

# a page subset: page index 1 only (0-based), byte-identical rule included
fmt, md = tors.documents.to_markdown("tests/engines_corpus/engines_two_page.pdf", pages=1)
# (Format.PDF, "second page line\n")
tors.documents.to_markdown("tests/engines_corpus/engines_two_page.pdf", pages=(0, 2))[
    1
] == tors.documents.to_markdown("tests/engines_corpus/engines_two_page.pdf")[1]
# True: a full range is the whole document, byte-identical
```

A file with no usable extension still converts; the content markers decide:

```python
# engines_report.docx's bytes, saved with no extension (a temp-file download):
tors.documents.to_markdown("upload.bin")
# (Format.DOCX, "Quarterly Review Q3 2026\n\n...")

# ...or never write the temp file at all: the bytes in, the same answer out
data = open("engines_report.docx", "rb").read()
tors.documents.to_markdown(data=data) == tors.documents.to_markdown("engines_report.docx")
# True: byte-identical, the pinned contract of the in-memory entry
```

**Async**: `await tors.documents.aio.to_markdown(...)` / `to_text(...)` run under
`asyncio.to_thread` (see [`tors.documents.aio`](#torsdocumentsaio) below).

## `tors.documents.sniff`

```python
def sniff(data: bytes) -> Format | None: ...
```

The standalone content-marker format detector: what `to_markdown`/`to_text` would
resolve these bytes to from content alone, with no path and no extension: the PDF
header, the RTF open group, OLE stream names, the ZIP package mimetype, the HTML
document marker. `sniff` opens and parses the container: anydoc's detection reads
the ZIP/OLE package's metadata (and the main part when the markers need it), so the
call's cost is a package parse, not a marker scan (a 120 KiB zip measured 267 MiB
peak RSS to answer docx). Budget accordingly when sniffing
untrusted leading bytes: the container is parsed before the format is named. (It
still has no async twin: the call is a single short native pass, and the thread hop
plus the parse would price the awaitable spelling above its value; the sync call
is the surface.)

`None` is not an error: it is the answer "the content names no format", a
signature-less text format such as CSV (name it via `format=` or let the extension),
or not a document at all. The routing caller's mislabeled-download answer: the bytes'
verdict overrides any label the download carried.

```python
import tors.documents

tors.documents.sniff(b"%PDF-1.7 ...")
# Format.PDF
tors.documents.sniff(b"{\\rtf1\\ansi ...")
# Format.RTF
tors.documents.sniff(b"<!DOCTYPE html>\n<html><body>hi</body></html>")
# Format.HTML
tors.documents.sniff(b"unit,status\nT-101,healthy\nT-102,failing\n")
# Format.CSV (the content heuristic: two non-empty lines, same comma count)
tors.documents.sniff(b"just some words\nover two lines\n")
# None: prose, the content names no format
tors.documents.sniff(b'{"unit": "T-101", "ok": true}\n{"unit": "T-102", "ok": false}\n')
# None: JSON-lines. The comma counts agree, but the record lines open with `{`,
#  declined by the CSV heuristic; not a documents format, `format=` the escape hatch
```

Three doctrine notes, all probed and pinned. XHTML: an `<?xml version="1.0"?>`
prologue before the doctype is skipped by the HTML marker (XHTML is HTML's XML
serialization), so `sniff(b'<?xml version="1.0"?><!DOCTYPE html...')` is `Format.HTML`,
while every other XML vocabulary (`<svg`, DocBook) sniff-answers `None`. JSON-lines:
record lines opening with `{` (or `[`) are declined by the CSV heuristic: their
comma counts agree across lines (every record serializes the same keys), so the
delimiter witness alone would claim them, and anydoc's csv parser would then mangle
records that are not cells; json-lines is deliberately not a documents format,
`None` is the answer, and `format=` is the escape hatch, the same hatch a
csv whose first field opens with a brace takes (the guard reads the first
non-empty line). And the answer is the container-true name a conversion would
report: an OLE workbook sniffs `Format.XLS`, a ZIP-based one `Format.XLSX`, so
`sniff` and `to_markdown` cannot disagree about what the bytes are.

## `tors.documents.pdf_extract`

```python
def pdf_extract(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> tuple[list[str], str]: ...
```

Read a PDF (`path`, or `data=` bytes) and return `(per_page_plain_text, markdown)`, one native pass over one
open document, the parse paid once for both outputs. `per_page_plain_text` is a
`list[str]`, one entry per page in page order: the text-layer-probe view. An
image-only/scanned page is an empty string, not an error, and a zero-page or textless
document yields empty output; routing decisions ("this PDF needs OCR") are the
caller's, made on these values (or on `pdf_classify`'s verdicts), never silently made
here. `markdown` is pdf_oxide's whole-document conversion: heading detection on,
images off, Tagged-PDF structure-tree reading order falling back to XY-Cut on
untagged documents, `/Link` annotations rendered as `[text](uri)`.

**The PDF family's engine lanes and input budget**: `backend=` and `max_bytes=`
on all four PDF-only functions, the same vocabulary the conversion pair takes.
`backend="auto"` (the default) and `backend="oxide"` both run pdf_oxide, the same
mapping the routing table makes for PDF ("oxide" is the oxide-family engine for
this format), byte-identical output either way. `backend="anydoc"` is refused
before any work runs with a named `ValueError`: a capability refusal, not the
format-level one (PDF+anydoc converts on `to_markdown`/`to_text`), because
anydoc's entire PDF surface is whole-document markdown (`to_markdown(bytes)` is
the one function its PDF module exposes, ~anydoc-0.2.4/src/formats/pdf.rs), and
its only per-page knowledge is the `NeedsOcr` refusal, while these four calls
are the probe-rich ones: `pdf_extract`'s per-page plain text is the OCR-routing
signal (an image-only page comes back as an empty string, the caller's
route-to-OCR witness), `pdf_page_count` walks the page tree (a count its
reader never returns on success), `pdf_classify` classifies per page,
and `pdf_link_uris` walks `/Annots` (anydoc has no annotation surface at all).
Whole-document markdown from anydoc is one `to_markdown(path, backend="anydoc")`
call away. `max_bytes=` is the input budget: an explicit value binds pre-read
exactly as on the conversion pair, the `path=`'s size at open, the `data=`
length on entry, never an over-budget byte read or copied; `None` (the default)
reads the pdf lane under the finite 512 MiB backstop (a memory-safety floor,
raised with an explicit `max_bytes=`); the 32 MiB default ceiling is the
amplified lanes' (anydoc, office_oxide, HTML) during-the-read check, lanes
these PDF-only calls never run.
`password=` unlocks an encrypted PDF; without it the entry fails closed,
`ValueError` at open, never empty output masquerading as "no content" (the empty
strings above are for unlocked documents; a contract failure precedes the work,
so an encrypted document under `backend="anydoc"` surfaces the capability
refusal, never the door-check error).

```python
import tors.documents

pages, markdown = tors.documents.pdf_extract("tests/engines_corpus/engines_two_page.pdf")
# (["first page line", "second page line"], "first page line\n\n---\n\nsecond page line\n")
#  per-page plain text + the joined markdown, one open, one pass

# the lane vocabulary: "oxide" is the engine "auto" already routes PDF to,
# "anydoc" a capability refusal pointing at the conversion pair
tors.documents.pdf_extract("tests/engines_corpus/engines_two_page.pdf", backend="oxide") == (
    pages,
    markdown,
)
# True: the same pdf_oxide lane either way, byte-identical answers
try:
    tors.documents.pdf_extract("tests/engines_corpus/engines_two_page.pdf", backend="anydoc")
except ValueError as exc:
    exc
    # ValueError('backend "anydoc" cannot serve the per-page text probe (the
    #  OCR-routing signal): anydoc\'s PDF surface is whole-document conversion
    #  only — to_markdown/to_text with backend="anydoc" (NeedsOcrError is that
    #  lane\'s scanned-page signal); use backend=\'auto\' or \'oxide\' here')
    #  the capability refusal, its message the pointer at the pair
```

**Async**: `await tors.documents.aio.pdf_extract(...)` runs this under
`asyncio.to_thread` so the event loop stays responsive across the call.

## `tors.documents.pdf_page_count`

```python
def pdf_page_count(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> int: ...
```

The page tree and nothing else, no content extraction. For gating expensive
downstream work (an OCR or conversion pass that scales with page count) without
paying for any of it. `password=` unlocks an encrypted PDF; without it the entry
fails closed (`ValueError` at open). `backend=`/`max_bytes=` follow the family's
shared lane note in the [`pdf_extract`](#torsdocumentspdf_extract) section
above: auto/oxide run pdf_oxide byte-identically, `backend="anydoc"` the
capability refusal (a count that engine's reader never returns on
success), an explicit budget binding pre-read.

```python
import tors.documents

tors.documents.pdf_page_count("tests/engines_corpus/engines_two_page.pdf")
# 2
```

**Async**: `await tors.documents.aio.pdf_page_count(...)` runs this under
`asyncio.to_thread` so the event loop stays responsive across the call.

## `tors.documents.pdf_link_uris`

```python
def pdf_link_uris(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> list[list[str]]: ...
```

The `/Annots` link walk: for every page, the URIs of its link annotations whose action
is a URI, in annotation order, one `list[str]` per page, page order, empty lists for
pages without link annotations. This is the raw navigation surface, deliberately
beside the markdown's inline `[text](uri)` links because the two answer different
questions: the markdown carries links whose visible text belongs in prose; this walk
carries every URI, including ones behind link rectangles whose text is not itself a
link (a "click here" button, an image, a bare rectangle) which no text rendering
surfaces at all. The caller that motivated it measured the difference on real
resumes: 60 documents, 16 links from the text layer, 34 from the annotations, a
quarter of candidates gained a LinkedIn/GitHub URL no text shape would show.

Verbatim and narrow, both on purpose: the lists are never deduped or canonicalized
(callers canonicalize differently: per-page review panels vs whole-document
projections), and only URI actions surface (`GoTo` is in-document navigation,
`GoToR` a remote file; neither is a web URI, and neither is fabricated into one).
Malformed annotation dictionaries are skipped by the engine's parser, not propagated
as page failures. `backend=`/`max_bytes=` follow the family's shared lane note in
the [`pdf_extract`](#torsdocumentspdf_extract) section above; the annotation walk
is pdf_oxide's reader (auto/oxide byte-identically), and `backend="anydoc"` the
capability refusal: anydoc has no annotation surface at all.
`password=` unlocks an encrypted PDF; without it the entry fails closed
(`ValueError` at open, never empty lists masquerading as "no links").

```python
import tors.documents

tors.documents.pdf_link_uris("tests/engines_corpus/engines_link.pdf")
# [["https://handbook.example.com/guide"]]: page 0's one link annotation
tors.documents.pdf_link_uris("tests/engines_corpus/engines_two_page.pdf")
# [[], []]: no link annotations anywhere, empty lists, never fabricated
```

**Async**: `await tors.documents.aio.pdf_link_uris(...)` runs this under
`asyncio.to_thread` so the event loop stays responsive across the call.

## `tors.documents.pdf_classify`

```python
def pdf_classify(
    path: str | os.PathLike[str] | None = None,
    data: bytes | None = None,
    password: str | None = None,
    backend: Backend | str = Backend.AUTO,
    max_bytes: int | None = None,
) -> PdfClassification: ...
```

The cheap text-vs-image preflight over a PDF: no content conversion, no OCR,
no rasterization. The answer to "does this PDF have a text layer, or is it an image
we can do nothing with locally", as a `PdfClassification` (below): every page's
`PageKind` verdict, the pages needing OCR, and the two derived routing booleans.
Encrypted documents fail closed on every entry (`ValueError` at open, pdf_oxide's
security rule: a security state is never masked as "all pages empty");
`password=` unlocks one. `backend=`/`max_bytes=` follow the family's shared lane
note in the [`pdf_extract`](#torsdocumentspdf_extract) section above: auto/oxide
run pdf_oxide byte-identically, `backend="anydoc"` the capability refusal
(per-page classification; that engine's only per-page knowledge is the binary
needs-OCR refusal), an explicit budget binding pre-read.

`PageKind` is the per-page vocabulary: `"text"` (a native text layer), `"scanned"`
(image-dominated: OCR the page), `"image_text"` (hybrid), `"mixed"`, or `"empty"`.
**`"empty"` is distinct from `"scanned"`**, and the distinction is the point: a blank
page is neither extractable nor an image to recover (not an error, not OCR work),
so `pages_needing_ocr` deliberately excludes it. `has_text` is true when at least one
page is `text`/`image_text`/`mixed` (extraction will yield something); `image_only` is
true when every page is `scanned` and there is at least one page (route the whole
document to an OCR stage).

The indices here are 0-based, like every page number this surface names; see the
convention note in the [`tors.documents`](#torsdocuments) section above.

```python
import tors.documents

cls = tors.documents.pdf_classify("tests/engines_corpus/engines_mixed.pdf")
# PdfClassification(page_count=2, page_kinds=[<PageKind.TEXT: 'text'>,
#                   <PageKind.SCANNED: 'scanned'>], pages_needing_ocr=[1])
cls.has_text, cls.image_only, cls.pages_needing_ocr
# (True, False, [1]): page index 1 (0-based: the second page) is the scan

cls = tors.documents.pdf_classify("tests/engines_corpus/engines_blank.pdf")
# PdfClassification(page_count=1, page_kinds=[<PageKind.EMPTY: 'empty'>], pages_needing_ocr=[])
#  blank ≠ scanned: nothing to extract, nothing to recover, no OCR routing
```

**Async**: `await tors.documents.aio.pdf_classify(...)` runs this under
`asyncio.to_thread` so the event loop stays responsive across the call.

## `tors.documents.PdfClassification`

The `pdf_classify` result: the preflight's answer with the routing rules derived
exactly once (the `has_text`/`image_only` rules live in the native getters; the typed
view adds typing only). Attributes: `page_count: int`, `page_kinds: list[PageKind]`
(every page's verdict, page order), `pages_needing_ocr: list[int]` (the 0-based
indices of the image-only pages: empty for a born-digital document, every page for a
scan, the difference for a mixed one), `has_text: bool`, `image_only: bool`. The
`repr` is the construction shape shown above.

## `tors.documents.NeedsOcrError`

```python
class NeedsOcrError(ValueError):
    pages: list[int]  # the 0-based page indices needing OCR
    page_count: int
```

Raised by `to_markdown`/`to_text` when the anydoc backend hits a PDF with
scanned/image-only pages: the "route this document to an OCR stage" signal, as an
exception because the conversion cannot proceed on those pages. A
`ValueError` subclass, so a broad `except ValueError` still catches it.
`.pages` holds 0-based page indices, the same convention as `pages=` and
`PdfClassification.pages_needing_ocr` (anydoc's 1-based numbers are re-based once, at
the core's seam); `.page_count` the document's page count.

```python
import tors.documents

try:
    tors.documents.to_markdown("tests/engines_corpus/engines_scanned.pdf", backend="anydoc")
except tors.documents.NeedsOcrError as exc:
    exc.pages, exc.page_count
    # ([0], 1): page index 0 of 1 needs OCR
```

The default `"auto"` routing sends PDF to pdf_oxide, whose lane yields empty text for
scanned pages instead (`pdf_extract`/`pdf_classify` are the preflight calls); the
exception is the anydoc lane's answer.

## `tors.documents.PageKind`

```python
class PageKind(str, Enum):
    TEXT = "text"
    SCANNED = "scanned"
    IMAGE_TEXT = "image_text"
    MIXED = "mixed"
    EMPTY = "empty"
```

One page's `pdf_classify` verdict, a `str` enum whose members are their
accepted strings. `"empty"` is deliberately distinct from `"scanned"` (see
[`pdf_classify`](#torsdocumentspdf_classify) above): a blank page is
neither extractable nor an image to recover, and `pages_needing_ocr`
excludes it.

## `tors.documents.Backend` / `tors.documents.Format`

```python
class Backend(str, Enum):
    AUTO = "auto"  # route by the measured table (the default)
    OXIDE = "oxide"  # force pdf_oxide / office_oxide
    ANYDOC = "anydoc"  # force anydoc


class Format(str, Enum):
    PDF = "pdf"
    HTML = "html"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    PPT = "ppt"
    PPTX = "pptx"
    RTF = "rtf"
    ODT = "odt"
    ODS = "ods"
    ODP = "odp"
    EPUB = "epub"
    CSV = "csv"
    TSV = "tsv"  # name-only vocabulary: accepted as format= input, never a resolved or sniffed answer (tsv bytes resolve and sniff as csv)
```

`str` enums: each member is its accepted string (`Backend.AUTO == "auto"` is `True`),
so every plain-string call keeps working and the enums cost nothing at the boundary;
the Rust validator remains the authority, and vocabulary the enums don't enumerate
yet (container variants like `"docm"`/`"xlsm"`) stays accepted as plain strings. A
resolved format outside the vocabulary is a bug and surfaces as `ValueError`.

`tors.documents.__version__` (and `tors_documents.__version__`, its source) is the
payload wheel's version, baked from the crate's `Cargo.toml` at build time: the
same number release-please bumps in lockstep across both wheels, so the two can
never disagree.

## `tors.documents.aio`

The awaitable spellings of the six path functions (`to_markdown`, `to_text`,
`pdf_classify`, `pdf_extract`, `pdf_page_count`, `pdf_link_uris`), each an
unconditional `asyncio.to_thread` dispatch, signatures identical to the sync
spellings, `path`/`data=` flowing through unchanged (pinned by the suite). The same doctrine as `tors.aio`: every one of these calls is a single
native pass whose cost scales with the document (a small one is milliseconds, a large
one hundreds), the exact class a thread hop pays for. `sniff` stays sync-only: its
cost is a container parse (see its section above), but it remains a single short
native pass a sync caller runs directly; no awaitable spelling ships. There is
no size-based branching inside any wrapper, and the choice between the sync spelling
and `tors.documents.aio` is the caller's, made once at the call site.

**Cancellation semantics**: `asyncio.to_thread` cannot cancel the native pass, and a
caller can be hurt by that. `wait_for`/`timeout()` on one of
these awaitables cancels the future (the `asyncio` wrapper returns control at the
deadline) while the underlying thread runs the conversion to completion, holding
its memory (the amplification lanes' worth: potentially gigabytes), and repeated
timeouts pile up blocked threads on the shared default executor. Treat these
awaitables as uncancellable work: size the input before the call (`max_bytes=` is
the pre-read guard), and only call with a timeout you are also willing to abandon
the thread to.

```python
import tors.documents.aio

fmt, text = await tors.documents.aio.to_text("report.docx")
# same conversion, same (Format, text) return, off the event loop's turn
```
