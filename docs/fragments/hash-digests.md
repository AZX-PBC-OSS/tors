<!--
Merge target: docs/api.md, the section "## `tors.md5_hex` / ... and their
`_digest` twins" (line ~4188). The prose below is additive; drop the
blocks marked MERGE NOTE into the merge decisions.

Verified against src/py/hash.rs, src/hash_impl.rs, and
tests/test_hash.py (862 lines): every output literal already printed in
that section (the webhook b64 signature b'q8IyxOXFC_mgdZj-GSqtTl2vj3eTIgMM0HQQ-FfRQVw=',
the lock int 15284293306093710542, the derivation chain
cc3ccddeaa0718afe52e67b9011b49dfe29ae01571495a75e17eea1f59f6e7b6, the hex
signature 43ec3b86..., the ETag 487f5cc2...) reproduces byte-exact, and
every contract sentence (str-is-UTF-8-bytes, exactly-bytes TypeError,
surrogate UnicodeEncodeError, key-validated-before-data, empty-input
legal, no ValueError paths, one computation two spellings) matches the
tests. No inaccurate sentences found; see the one redundancy nit at the
bottom.
-->

Add to the section's contract prose (after the "Stateless one-shot" and
GIL-model paragraphs, before "### Raw digest bytes"):

**No `tors.aio` twin.** The hashing family is absent from
`tors.aio._WRAPPED` on the module's own size discipline: a one-shot
digest over the inputs this surface exists for (cache keys, request
signatures, webhook bodies) is a microsecond-scale call, and the
`asyncio.to_thread` hop costs tens of microseconds, more than the call
itself (see [Async use](async.md)). A coroutine that does hash a
multi-MiB payload wraps manually, `await asyncio.to_thread(tors.sha256_digest, blob)`;
the GIL is released for the digest either way, so the wrap buys the loop
its wall clock back and nothing else.

### The `_digest` twins: the bytes-side pins

The five raw-digest names return a fresh, immutable `bytes` object of
exactly one length each: `md5_digest` 16 bytes, `sha1_digest` 20,
`sha256_digest` 32, `sha512_digest` 64, `hmac_sha256_digest` 32. The
object is built per call (`PyBytes::new`), never cached or interned, so
handing the digest to a second consumer layer cannot alias tors's
internals. Three parity identities hold for every input and are pinned
differentially in `tests/test_hash.py` (`TestDifferentialParity`,
hypothesis corpora plus RFC 1321 / FIPS 180-4 / RFC 4231 known-answer
vectors through both spellings):

- `tors.md5_digest(x) == hashlib.md5(x).digest()` and likewise for
  sha1/sha256/sha512 (cross-implementation agreement: RustCrypto vs
  OpenSSL);
- `tors.hmac_sha256_digest(k, x) == hmac.new(k, x,
  hashlib.sha256).digest()`, any key length, empty included (an empty
  key IS the zero-padded 64-byte key, RFC 2104's padding rule);
- `tors.<alg>_hex(x) == tors.<alg>_digest(x).hex()`: one digest
  computation per call, the hex path consumes the digest path.

The argument contract is the `_hex` twins' byte for byte, and the error
messages name the argument that failed:

```python
import hashlib
import tors

digest = tors.sha512_digest("payload")            # str in: its UTF-8 bytes
len(digest)
# 64
digest == hashlib.sha512(b"payload").digest()     # cross-implementation parity
# True
tors.sha512_hex("payload") == digest.hex()        # one computation, two spellings
# True
```

```python
import tors

tors.sha256_digest(bytearray(b"payload"))
# TypeError: data must be str or bytes, not <class 'bytearray'>
```

A `bytes` SUBCLASS is accepted (pyo3's `PyBytes` borrow covers
subclasses); `bytearray`, `memoryview`, and everything else that is not
exactly `str` or `bytes` raise `TypeError` with that message shape,
because the GIL-released digest reads an immutable buffer. On the HMAC
spellings the key is borrowed and validated before the data, so when
both arguments are bad the KEY's error is the one that fires:

```python
import hashlib
import hmac as hmac_module
import tors

tors.hmac_sha256_digest(None, bytearray(b"data"))
# TypeError: key must be str or bytes, not <class 'NoneType'>

tors.hmac_sha256_digest(b"", b"payload") == hmac_module.new(b"", b"payload", hashlib.sha256).digest()
# True
tors.hmac_sha256_digest(b"", b"payload") == tors.hmac_sha256_digest(b"\x00" * 64, b"payload")
# True
```

The empty input is a pinned known answer in every algorithm, the digest
of the empty string (FIPS 180-4 / RFC 1321's vectors themselves):

```python
import tors

tors.sha256_digest(b"").hex()
# 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
tors.md5_digest(b"").hex()
# 'd41d8cd98f00b204e9800998ecf8427e'
```

<!--
MERGE NOTES for the existing section, in lieu of inaccuracies:

1. NO inaccurate sentences found. Every output literal and every
   contract claim in the section as it stands (line ~4188 through the
   ETag example) verifies byte-exact against the built extension.
2. One redundancy, a deletion candidate if the merge touches it: the
   "Raw digest bytes" subsection's contract recap (line ~4310-4314)
   ends with "any key length legal, empty included; the key borrowed
   and validated before the data; empty input legal": the "empty
   input legal" clause repeats the parent paragraph's "Any input length
   is legal, empty included" (line ~4254). Suggested trim: drop "; empty
   input legal" from the recap sentence.
3. The section has no Async statement; the "No `tors.aio` twin"
   paragraph above supplies it (place it after the "No deadline_ms"
   paragraph, matching the placement pattern of the other sections'
   Async lines).
-->
