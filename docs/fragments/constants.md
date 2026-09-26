## Module data: `tors.CHARSET_B62` / `tors.CHARSET_B64URL` / `tors.CHARSET_HEX_LOWER` / `tors.CHARSET_HEX_MIXED` / `tors.CHARSET_HEX_UPPER` / `tors.KEY_FAMILIES`

```python
tors.CHARSET_B62: str
tors.CHARSET_B64URL: str
tors.CHARSET_HEX_LOWER: str
tors.CHARSET_HEX_MIXED: str
tors.CHARSET_HEX_UPPER: str
tors.KEY_FAMILIES: tuple[str, ...]
```

Six published module data constants: the five pinned alphabets the
batch validators consume and the canonical key-family tuple the
credential scrubber consumes. They are data, not functions, on purpose
(the [Common alphabets](#common-alphabets) subsection under
`first_invalid_charset` records the scope decision): named per-alphabet
validator wrappers would each delegate to the same core for zero
performance gain, so the constants kill the only real friction, the
transcription risk of re-spelling a 62-character alphabet (a wrong
62-character set still validates *something*) or re-enumerating a
closed family set by hand, at every call site. Being data, they carry
no argument contract, no GIL story (nothing runs), and no `tors.aio`
twin (`aio` wraps functions; a constant read is not a call to hide).
All six sit in `tors.__all__`, and the stub is held to it by the same
drift guard as every function (`tests/test_pyi_drift.py`).

The five alphabets are plain `str` literals defined in the package's
`__init__.py`, and their CONTENT is contract, pinned byte-exact in
`tests/test_first_invalid_charset.py`; the stub carries the type only,
so the live module stays the single spelling of each alphabet:

```python
import tors

tors.CHARSET_B62
# '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
tors.CHARSET_B64URL
# 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
tors.CHARSET_HEX_LOWER
# '0123456789abcdef'
tors.CHARSET_HEX_UPPER
# '0123456789ABCDEF'
tors.CHARSET_HEX_MIXED
# '0123456789abcdefABCDEF'

(len(tors.CHARSET_B62), len(tors.CHARSET_B64URL),
 len(tors.CHARSET_HEX_LOWER), len(tors.CHARSET_HEX_MIXED), len(tors.CHARSET_HEX_UPPER))
# (62, 64, 16, 22, 16)
```

`CHARSET_B64URL` is RFC 4648 §5's url-safe alphabet, UNPADDED: `=` is
positionally structured, not a codepoint a flat charset can admit
(b64_decode is the strict base64 validator; the Common alphabets
subsection has the full reasoning). `CHARSET_HEX_MIXED` is the
22-codepoint union of the two hex spellings, for validators that must
accept either case. Membership is per scalar codepoint with no
normalization, order and duplicates in a spelling irrelevant. Each
constant feeds `first_invalid_charset`/`first_invalid_offender` as the
`rest=` set (or `first=` for position 0); the uniform `first=None`
spelling makes it a single argument:

```python
import tors

digests = ["487f5cc2c45cc57e638d9fce8c33d95c", "487F5CC2C45CC57E638D9FCE8C33D95C"]
tors.first_invalid_charset(digests, rest=tors.CHARSET_HEX_LOWER)
# 1    (the uppercase digest offends the lowercase alphabet)
tors.first_invalid_charset(digests, rest=tors.CHARSET_HEX_MIXED)
# -1   (both pass the case-insensitive union)
```

`KEY_FAMILIES` is different in one respect: it is exported from the
Rust core, built from the same single source (`pii_impl`'s
`KEY_FAMILY_NAMES`) the `api_keys` scanner's grammar table and the
`families=` unknown-name error are spelled from, so the tuple, the
accepted names, and the error message cannot drift apart. It is the
fourteen key-family names in the `KeyFamily` discriminant order:

```python
import tors

tors.KEY_FAMILIES
# ('openai', 'anthropic', 'google', 'fireworks', 'modal', 'github',
#  'minted', 'jwt', 'aws', 'xai', 'gcp_oauth', 'pem', 'azure', 'gitlab')
tuple(f for f in tors.KEY_FAMILIES if f != "jwt")[:3]
# ('openai', 'anthropic', 'google')
```

Its consumer is `scrub_pii`/`scrub_pii_report`'s `families=` parameter
(see that section's Key-family selection): `families=None` is every
family this version knows, the set grows when new families land
(semver-visible, so stability-seeking callers list names explicitly),
and an unknown name is refused with an error that names the accepted
set, which is the tuple itself:

```python
import tors

tors.scrub_pii("ssn 123-45-6789", families=["ssn"])
# ValueError: families must be one of ('openai', 'anthropic', 'google', 'fireworks', 'modal', 'github', 'minted', 'jwt', 'aws', 'xai', 'gcp_oauth', 'pem', 'azure', 'gitlab'), not "ssn"
```

## `tors.__version__`

The installed distribution's version, read once at import from its
metadata (`importlib.metadata.version("tors")`, the same source
`pip`/`uv` report, so they cannot disagree; pinned by
`tests/test_package_version.py`). It is never a second literal in the
source that release tooling cannot bump. Where the distribution
metadata is absent (a source tree imported off-path) it answers
`"unknown"` rather than lying. Declared on the typed surface as plain
`str` (the value is import-computed, not a constant the stub could
lie about).

```python
import tors

tors.__version__
# '0.14.0'
```
