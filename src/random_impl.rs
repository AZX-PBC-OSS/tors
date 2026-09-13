//! The random-generation family's pure-Rust core: `random_string`,
//! `random_hex`, `random_b62`, `random_b64url`, `uuid4`, `uuid7`, and the
//! uuids' bytes spellings `uuid4_bytes`/`uuid7_bytes` (the pyo3 bindings
//! live in `py/random.rs`; the criterion bench in `benches/random.rs`
//! drives this module directly, and the fuzz target in
//! `fuzz/fuzz_targets/random.rs`).
//!
//! # The entropy contract (the family's whole security model)
//!
//! Two spellings, one per caller intent, never interchangeable:
//!
//! * **Unseeded (the default)**: fresh bytes from the operating system's
//!   CSPRNG on every call — `OsRng` (rand_core's zero-sized handle over the
//!   getrandom crate's syscall), drawn through `try_fill_bytes` per call.
//!   There is no process or thread RNG state anywhere in this module, so a
//!   `fork()` child cannot inherit and replay a parent's stream — the
//!   fork-safety hazard a cached userspace RNG carries — matching stdlib
//!   `secrets`' semantics exactly (secrets itself calls the OS per call).
//!   This is the spelling safe for keys, tokens, and secrets.
//! * **Seeded (`seed=`)**: a deterministic ChaCha20 stream
//!   (`rand_chacha::ChaCha20Rng`) keyed by `seed_from_u64(seed)`. The output
//!   is a pure function of (seed, arguments): fully predictable from the
//!   seed — a reproducible-test/fixture tool, NEVER safe for secrets, keys,
//!   or tokens (rand_core's own docs say the same about `seed_from_u64`:
//!   "not suitable for cryptography ... the input size is only 64 bits").
//!   A seeded stream replays exactly across processes, threads, and forks
//!   by design — same seed, same output — which is why it is never safe
//!   for secrets. The derivation is rand_core's documented-stable default (its docs call
//!   changing it a value-breaking change): a PCG32 (XSH-RR) generator keyed
//!   by the seed emits the 32-byte ChaCha key as 8 little-endian u32s.
//!   Note the spec-vs-source finding this records: the design brief called
//!   this a SplitMix64 derivation; rand_core 0.9's resolved source is PCG32,
//!   and this module (and the Python test oracle) follow the source.
//!
//! # The engine spec (what the seeded pins freeze)
//!
//! One byte-stream view over both sources, so every generator's seeded
//! output is a documentable function of the stream:
//!
//! * The stream is ChaCha20 blocks (Bernstein layout: constants, 8 key
//!   words, 64-bit block counter words 12-13, 64-bit stream-id words 14-15,
//!   both zero under `seed_from_u64`), serialized little-endian, block 0
//!   first — verified against rand_chacha 0.9's source, and re-implemented
//!   independently by the Python oracle in `tests/test_random.py`, which the
//!   seeded pins must match (the pins do not merely freeze this module's
//!   output).
//! * The byte-fill consumers are the uuids alone (`uuid4`, one 16-byte
//!   `fill_bytes`; `uuid7`, one 10-byte counter/random fill): they consume
//!   exactly the first n stream bytes of one call and hand them to the
//!   uuid crate's builders, whose bit-structured fields are exactly what a
//!   byte fill is for. `uuid4_bytes`/`uuid7_bytes` return those builders'
//!   16-byte buffers pre-formatting (the bytes-out spellings for consumers
//!   who re-wrap the canonical str back into bytes anyway), and the string
//!   spellings are the canonical formatting of the same buffers — one
//!   construction per uuid, two return spellings. (Before the
//!   length-first refactor `random_hex` and `random_b64url` byte-filled
//!   and encoded here too; the maintainer ergonomics directive — backend
//!   devs think "I want a base62 id X characters long", so every token
//!   spelling takes the output length directly — moved both onto the
//!   char-sampling engine, and the byte path shrank to the uuids.)
//! * The char-sampling engine is `random_string`, with `random_b62`,
//!   `random_hex`, and `random_b64url` all delegating to it over their
//!   fixed alphabets (one engine, never duplicated logic — the Python
//!   suite pins the delegation as literal equality). It consumes the
//!   stream as u64 words — 8 consecutive stream bytes, little-endian, in
//!   order (the same word order rand_core's own `BlockRng::next_u64`
//!   produces, "least significant first") — buffered one 1024-byte block
//!   per `fill_bytes` so the unseeded spelling costs one OS syscall per
//!   128 words instead of one per word (`OsRng`'s `try_next_u64` is a
//!   syscall per call). Buffering changes no output: the u64 sequence is
//!   the stream's u64 sequence either way.
//! * Consequently `random_hex(n, seed=s)` IS
//!   `random_string(n, "0123456789abcdef", seed=s)` — same length, same
//!   stream consumption, same output, pinned literally (the
//!   pre-refactor "agree in distribution but not in value" subtlety was
//!   an artifact of hex byte-filling n bytes against 2n sampled
//!   characters; it died with the byte path).
//!
//! # Unbiased alphabet sampling (the no-modulo-bias argument)
//!
//! All four token spellings (`random_string` itself, plus the hex/b62/
//! b64url delegations) draw alphabet indices with Lemire's
//! "nearly-divisionless" method (Fast Random Integer Generation in an
//! Interval, ACM TOPLAS 2019): for a fresh u64 `x` and alphabet size `n`,
//! compute the 128-bit product `m = x * n`, return `m >> 64`, and reject the
//! draw when the low 64 bits `l = m mod 2^64` fall below `t = 2^64 mod n`.
//! Why that is exactly unbiased: the map `x -> m = x*n` is a bijection from
//! `[0, 2^64)` onto the multiples of `n` below `n * 2^64`, and each output
//! `h = m >> 64` is hit by the multiples inside `[h * 2^64, (h+1) * 2^64)` —
//! either `q = floor(2^64 / n)` or `q + 1` of them. Conditioning on
//! `l >= t` keeps exactly `q` multiples per window, because a half-open
//! window of length `2^64 - t = q * n` (exact: `2^64 = q*n + t`) contains
//! exactly `q` multiples of `n` no matter where it starts. Every output then
//! has `q` preimages: uniform. The rejection probability is `t / 2^64 < n /
//! 2^64` (< 2^-58 for any alphabet a Python str can hold), so a redraw is
//! astronomically rare in practice — but the property is by construction,
//! not by luck, and a plain `x % n` does NOT have it (modulo concentrates
//! the first `2^64 mod n` outputs' preimages by one).
//!
//! # uuid7's boundary, stated honestly
//!
//! `uuid7` is the timestamped member: 48-bit Unix-epoch milliseconds (read
//! from `SystemTime::now()` inside the call), version 7, RFC 4122 variant,
//! and 74 random bits (12-bit rand_a + 62-bit rand_b) from the same one-shot
//! OS draw — the uuid crate's zero-feature
//! `Builder::from_unix_timestamp_millis` lays the fields out (verified in
//! its resolved source: the builder masks rand_a's top nibble and rand_b's
//! top 2 bits into the version/variant nibbles, which is where the 74 of
//! the 80 drawn bits go). Its uniqueness promise is probabilistic
//! (birthday bound over 74 random bits within a millisecond, distinct
//! timestamps across milliseconds), NOT counter-monotonic: two calls in the
//! same millisecond are ordered by their random bits, and a backwards clock
//! step flows straight into the timestamp. uuid_utils' strict-monotonic
//! counter is a different product promise. There is no `seed=` parameter by
//! design: the timestamp is external state, so a seeded uuid7 would still
//! vary with time — the deterministic tool is `uuid4(seed=...)`. The
//! caller-visible contract is the timestamp: the canonical string's first
//! two dash-free groups (`u[:8] + u[9:13]`) decode to the call's Unix
//! epoch milliseconds.
//!
//! # Dependencies
//!
//! Per the crate's dependency policy the hard parts are all maintained
//! crates: rand (OsRng), rand_chacha (the stream), uuid (field layout and
//! formatting) — and after the length-first refactor that is the whole
//! list: const-hex and base64 served the old byte-fill+encode hex/b64url
//! spellings and dropped out of this module (both crates stay in
//! Cargo.toml for their other users — finalize/merkle's digest hex and
//! `b64_impl`'s RFC 4648 core). What this module owns is the
//! tors-specific glue: the block-buffered word sampler, the Lemire index
//! draw (the algorithm is transcribed with its proof sketch above), and
//! the argument contracts.

use std::time::{SystemTime, UNIX_EPOCH};

use rand::rngs::OsRng;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng, TryRngCore};
use uuid::{Builder as UuidBuilder, Uuid};

/// The hex alphabet `random_hex` samples over: `[0-9a-f]`, lowercase —
/// `random_hex` is exactly `random_string(length, HEX_CHARS)` — one
/// engine, no copied logic — and this constant is what that delegation
/// means.
pub const HEX_CHARS: &str = "0123456789abcdef";

/// The base62 alphabet `random_b62` samples over: `[0-9A-Za-z]`, the
/// URL-safe, case-sensitive, human-transcribable set. `random_b62` is
/// exactly `random_string(length, BASE62_CHARS)` — one engine, no copied
/// logic — and this constant is what that delegation means.
pub const BASE62_CHARS: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// The urlsafe-token alphabet `random_b64url` samples over: RFC 4648 §5's
/// 64-character url-safe set (`A-Za-z0-9-_`; `+` and `/` never, and `=`
/// is not a member — padding is an encoding concept, not a token
/// concept). `random_b64url` is exactly `random_string(length,
/// B64URL_CHARS)` — one engine, no copied logic — and this constant is
/// what that delegation means.
pub const B64URL_CHARS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// The error shapes of the family, mapped to Python exceptions by the pyo3
/// layer (the cores themselves never see a Python type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RandomError {
    /// `random_string`'s alphabet argument was empty: no alphabet, no
    /// output. ValueError naming the parameter in the binding.
    EmptyAlphabet,
    /// The OS CSPRNG draw failed (`getrandom`/`getentropy` error). getrandom
    /// documents this as effectively impossible after the first successful
    /// boot-time initialization; possible shapes are early-boot blocking
    /// (waited out by the syscall itself) and system misconfiguration. The
    /// string carries the OS error's own message.
    Os(String),
    /// `uuid7`'s timestamp source failed: the system clock reads before the
    /// Unix epoch (clock skew backwards past 1970), or at/above 2^48
    /// milliseconds (year ~10,892 — the uuid 1.26 builder silently
    /// TRUNCATES the timestamp above the 48 bits the field holds, verified
    /// in its source: `timestamp.rs` masks `millis_high` to the field
    /// width; this core refuses explicitly with `Clock` instead, so the
    /// truncation is unreachable by code, not just by calendar — no real
    /// system clock reads year 10,892, and only a mocked clock past the
    /// horizon can hit it).
    Clock(String),
    /// The output string's reservation failed: the requested length's
    /// worst-case byte size does not fit in memory. The usize is that
    /// worst-case (length × the alphabet's widest UTF-8 width, saturating).
    /// The binding maps this to Python's `MemoryError` — catchable, the
    /// `'x' * n` / `secrets.token_hex(n)` convention — where the reserve it
    /// replaces (`String::with_capacity`) ABORTED the process on the same
    /// request (Rust's default allocation-failure handler: SIGABRT,
    /// uncatchable, takes the interpreter with it). `try_reserve` refuses
    /// oversized requests as this error BEFORE any allocation is attempted
    /// (its own capacity check rejects anything past `isize::MAX` without
    /// calling the allocator; larger-but-in-range requests fail inside the
    /// allocator, which returns null rather than aborting under this
    /// spelling), so the pinned in-suite behavior is deterministic on every
    /// 64-bit wheel.
    Memory(usize),
}

impl RandomError {
    /// The message the binding attaches to the mapped Python exception.
    pub fn message(&self) -> String {
        match self {
            RandomError::EmptyAlphabet => "alphabet must be a non-empty str".to_string(),
            RandomError::Os(err) => format!("operating system entropy source failed: {err}"),
            RandomError::Clock(err) => format!("uuid7 timestamp source failed: {err}"),
            RandomError::Memory(requested) => {
                format!("cannot allocate {requested} bytes for the output string")
            }
        }
    }
}

/// The two entropy spellings as one fill interface: the unseeded default
/// (fresh OS bytes per call — see the module docs for the fork-safety
/// argument) and the seeded deterministic ChaCha20 stream.
enum Source {
    Os(OsRng),
    // Boxed per clippy's large_enum_variant: ChaCha20Rng carries a
    // quarter-KiB results buffer, and the boxed spelling keeps `Source`
    // (and the `Words` struct embedding it) off the fat-enum path. The one
    // allocation per seeded call is noise next to the ChaCha key setup.
    Seeded(Box<ChaCha20Rng>),
}

impl Source {
    /// `None` is the unseeded spelling (OS entropy); `Some(seed)` is the
    /// deterministic stream keyed by the caller's (already mod-2^64-reduced)
    /// seed.
    fn new(seed: Option<u64>) -> Self {
        match seed {
            None => Source::Os(OsRng),
            Some(seed) => Source::Seeded(Box::new(ChaCha20Rng::seed_from_u64(seed))),
        }
    }

    /// Fill `buf` from the stream. The OS spelling's failure (see
    /// [`RandomError::Os`]) is returned, never panicked on: the binding
    /// maps it to `RuntimeError` after the GIL is reacquired. The seeded
    /// spelling cannot fail (ChaCha20 in userspace, no IO).
    fn fill(&mut self, buf: &mut [u8]) -> Result<(), RandomError> {
        match self {
            Source::Os(rng) => rng
                .try_fill_bytes(buf)
                .map_err(|err| RandomError::Os(err.to_string())),
            Source::Seeded(rng) => {
                rng.fill_bytes(buf);
                Ok(())
            }
        }
    }
}

/// The u64 word source the alphabet sampler draws from: the stream as
/// little-endian u64 words in order, block-buffered (one 1024-byte fill per
/// 128 words, un-tuned — no block-size sweep has calibrated it) so the
/// unseeded spelling pays one OS syscall per block rather
/// than one per word. See the module docs' engine spec: the buffer changes
/// consumption cost, never the word sequence, so seeded output is identical
/// with or without it.
struct Words {
    source: Source,
    buf: [u8; 1024],
    /// The read cursor into `buf`, always a multiple of 8 (`1024` means
    /// exhausted: the next draw refills).
    pos: usize,
}

impl Words {
    fn new(source: Source) -> Self {
        Words {
            source,
            buf: [0; 1024],
            pos: 1024,
        }
    }

    /// Test-only constructor: the exact u64 sequence `words`, in draw
    /// order, with no entropy source behind it — so a test can place a
    /// rejecting word first and assert the redraw consumes the next one.
    #[cfg(test)]
    fn for_test(words: &[u64]) -> Self {
        let mut buf = [0u8; 1024];
        assert!(
            words.len() * 8 <= buf.len(),
            "test words exceed the block buffer"
        );
        for (i, w) in words.iter().enumerate() {
            buf[i * 8..(i + 1) * 8].copy_from_slice(&w.to_le_bytes());
        }
        Words {
            // Never touched: `pos + 8 <= len` holds until the words run
            // out, and the tests never draw past their own sequence.
            source: Source::new(Some(0)),
            buf,
            pos: 0,
        }
    }

    fn next_u64(&mut self) -> Result<u64, RandomError> {
        if self.pos + 8 > self.buf.len() {
            self.source.fill(&mut self.buf)?;
            self.pos = 0;
        }
        let word = u64::from_le_bytes(
            self.buf[self.pos..self.pos + 8]
                .try_into()
                .expect("cursor + 8 is in bounds by the refill above"),
        );
        self.pos += 8;
        Ok(word)
    }
}

/// Lemire's nearly-divisionless unbiased draw from `[0, n)`: see the module
/// docs for the full no-modulo-bias argument (the short version: rejecting
/// exactly the draws whose product low-half falls below `2^64 mod n` leaves
/// every output the same number of preimages, `floor(2^64 / n)`).
///
/// `n` must be >= 1 (the caller checks the non-empty alphabet before
/// sampling; an empty alphabet has no index to draw).
///
/// The single-draw step as a pure function, factored out so the rejection
/// path is unit-testable without a stream: `None` is "rejected, consume
/// the next word and redraw", `Some(h)` is the accepted index. `x = 0`
/// always rejects for `n >= 2` (its product is 0, below every threshold),
/// which is what the reject-path test feeds.
fn lemire_accept(x: u64, n: u64) -> Option<u64> {
    let product = (x as u128) * (n as u128);
    let low = product as u64;
    // Fast path (probability 1 - n/2^64): the low half alone certifies
    // the draw, no division needed — the "nearly divisionless" property.
    if low >= n {
        return Some((product >> 64) as u64);
    }
    // Rare: compare against the rejection threshold t = 2^64 mod n.
    // (-n) mod n == (2^64 - n) mod n == 2^64 mod n, and for powers of
    // two it is 0 (exact division, no rejection region at all).
    let threshold = n.wrapping_neg() % n;
    if low >= threshold {
        return Some((product >> 64) as u64);
    }
    // Rejected (probability < n/2^64): the caller redraws, consuming
    // the next word.
    None
}

fn lemire_below(words: &mut Words, n: u64) -> Result<u64, RandomError> {
    loop {
        let x = words.next_u64()?;
        if let Some(h) = lemire_accept(x, n) {
            return Ok(h);
        }
        // Rejected (probability < n/2^64): redraw. Consumes the next word.
    }
}

/// `random_string`'s core: `length` characters sampled uniformly (Lemire)
/// from `alphabet`'s characters. Any non-empty str is a legal alphabet:
/// multibyte characters are pushed by character, not by byte, so the output
/// is always `length` characters over the alphabet's own characters. The
/// caller (binding) has already rejected the empty alphabet with
/// `RandomError::EmptyAlphabet` checked here — the cores validate their own
/// contracts too, so the fuzz target exercises the same refusal.
///
/// Duplicate characters are WEIGHTED, not deduplicated: each of the
/// `length` positions is an independent uniform draw over the alphabet's
/// character POSITIONS, so `"aaab"` yields `a` with probability 3/4 and
/// `b` with 1/4. Callers wanting uniform-over-distinct-characters must
/// dedupe the alphabet first.
///
/// Sampling is over Unicode scalar values (`char`), not grapheme
/// clusters: an alphabet holding a base character plus combining marks
/// samples each codepoint independently, so a combining mark can land
/// without its base. Callers needing cluster-atomic output should pass
/// precomposed characters.
///
/// The alphabet is materialized into a `Vec<char>` fresh on every call
/// (one O(alphabet) pass under the binding's detach): there is no cross-
/// call cache, by the same no-state discipline that keeps the unseeded
/// spelling fork-safe. Cost is O(alphabet) to materialize plus O(length)
/// draws. Bulk callers reusing one huge alphabet across many calls should
/// prefer the stdlib (`random.choices`) or hold the
/// materialization themselves; this spelling optimizes for the
/// token/id case (short alphabets, one call per token).
pub fn random_string(
    length: usize,
    alphabet: &str,
    seed: Option<u64>,
) -> Result<String, RandomError> {
    let chars: Vec<char> = alphabet.chars().collect();
    if chars.is_empty() {
        return Err(RandomError::EmptyAlphabet);
    }
    let n = chars.len() as u64;
    // Reserve for the worst case (every output char the alphabet's widest);
    // saturating so an absurd length fails at the reservation, not
    // arithmetic. The reserve is `try_reserve`, NOT `with_capacity`: an
    // impossible length must come back as `RandomError::Memory` (the
    // binding's catchable MemoryError, Python's own `'x' * n` convention),
    // where `with_capacity`'s allocation failure aborts the process —
    // uncatchable SIGABRT, the whole interpreter down. `try_reserve`
    // refuses before any allocation is attempted, so the refusal is a
    // plain error value, raised after the detach in the binding.
    let widest = chars.iter().map(|c| c.len_utf8()).max().unwrap_or(1);
    let mut out = String::new();
    out.try_reserve(length.saturating_mul(widest))
        .map_err(|_| RandomError::Memory(length.saturating_mul(widest)))?;
    let mut words = Words::new(Source::new(seed));
    for _ in 0..length {
        let index = lemire_below(&mut words, n)? as usize;
        out.push(chars[index]);
    }
    Ok(out)
}

/// `random_hex`'s core: `random_string` over [`HEX_CHARS`] — the one
/// engine, delegated, never duplicated. `length` lowercase hex characters,
/// any length: odd is legal (a 31-char hex id is a real shape), and even
/// lengths are what digest-shaped keys want (every 2 characters are
/// exactly one byte, so `random_hex(2 * n)` is the byte-exact spelling
/// for callers who need encodable random material). `secrets.token_hex(n)`
/// and `random_hex(2 * n)` are the same uniform distribution over 2n-char
/// hex strings — different draws, never different contracts.
pub fn random_hex(length: usize, seed: Option<u64>) -> Result<String, RandomError> {
    random_string(length, HEX_CHARS, seed)
}

/// `random_b62`'s core: `random_string` over [`BASE62_CHARS`] — the one
/// engine, delegated, never duplicated.
pub fn random_b62(length: usize, seed: Option<u64>) -> Result<String, RandomError> {
    random_string(length, BASE62_CHARS, seed)
}

/// `random_b64url`'s core: `random_string` over [`B64URL_CHARS`] — the one
/// engine, delegated, never duplicated. `length` characters uniform over
/// the 64-character RFC 4648 §5 urlsafe alphabet (`A-Za-z0-9-_`), every
/// position unconstrained: the opaque-token contract, NOT "a valid base64
/// encoding of N random bytes" (an encoding's final character is
/// constrained — at 43 characters, an encoding of 32 bytes can only ever
/// show 16 distinct final characters, the final char carrying the final
/// byte's low 4 bits shifted into place; the 31-byte encoding is the
/// 4-distinct case, its final char carrying just the low 2 bits; this
/// spelling shows all 64 — and lengths that are not valid base64 output
/// lengths, like 41, are legal here). There is no `padded` spelling at
/// all: padding is an encoding concept, not a token concept, and `=` can
/// never appear.
pub fn random_b64url(length: usize, seed: Option<u64>) -> Result<String, RandomError> {
    random_string(length, B64URL_CHARS, seed)
}

/// `uuid4_bytes`' core — and `uuid4`'s, shared: one 16-byte fill handed to
/// the uuid crate's zero-feature `Builder::from_random_bytes` (it sets the
/// version-4 and RFC 4122 variant nibbles; verified in the resolved
/// source: byte 6 `(b & 0x0f) | 0x40`, byte 8 `(b & 0x3f) | 0x80`),
/// returned as the raw 16 bytes — no canonical formatting. This is the
/// bytes-out spelling for consumers who re-wrap the canonical str back
/// into bytes anyway (`UUID(bytes=...)` construction, `.hex()` slicing):
/// one draw and the field layout, no format-then-reparse roundtrip.
/// Seeded calls are pure functions of the seed (the pins and the Python
/// oracle freeze exactly this construction); `uuid4` formats this same
/// buffer, so the two spellings are one construction by code, not by
/// coincidence.
pub fn uuid4_bytes(seed: Option<u64>) -> Result<[u8; 16], RandomError> {
    let mut bytes = [0u8; 16];
    Source::new(seed).fill(&mut bytes)?;
    Ok(*UuidBuilder::from_random_bytes(bytes).into_uuid().as_bytes())
}

/// `uuid4`'s core: the canonical string of [`uuid4_bytes`]' buffer — 36
/// chars, lowercase hex, hyphens at 8/13/18/23. The pre-format bytes ARE
/// the construction; this spelling is their canonical formatting.
pub fn uuid4(seed: Option<u64>) -> Result<String, RandomError> {
    Ok(Uuid::from_bytes(uuid4_bytes(seed)?).to_string())
}

/// `uuid7_bytes`' core — and `uuid7`'s, shared: see the module docs'
/// "uuid7's boundary" section for the honest statement (probabilistically
/// unique, not counter-monotonic, no seed parameter because the timestamp
/// is external state). The 10-byte counter/random draw is one OS fill; the
/// builder consumes 74 of its 80 bits (rand_a's top nibble and rand_b's
/// top 2 bits are the version and variant nibbles' territory). Returned
/// as the raw 16 bytes, no canonical formatting — the bytes-out spelling
/// for the same re-wrap consumers [`uuid4_bytes`] serves; `uuid7`
/// formats this same buffer.
///
/// Uniqueness is probabilistic over the 74 random bits (122-bit uuid4
/// needs ~2^61 draws for a 50% collision; 10k uuid4 draws collide with
/// probability ~1e-29 — the same birthday arithmetic the Python suite
/// pins; uuid7's per-millisecond bound is ~2^37 same-millisecond draws).
pub fn uuid7_bytes() -> Result<[u8; 16], RandomError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| {
            RandomError::Clock(format!("system clock is before the Unix epoch: {err}"))
        })?;
    let millis = u64::try_from(duration.as_millis())
        .map_err(|_| RandomError::Clock("unix milliseconds do not fit in u64".to_string()))?;
    uuid7_bytes_at(millis)
}

/// The timestamp-parameterized construction behind [`uuid7_bytes`]:
/// the same builder over a caller-supplied millisecond timestamp.
/// Production passes `SystemTime::now()`; the factorization exists so
/// the 48-bit horizon refusal is unit-testable — no real system clock
/// reads year 10,892, so only this spelling can exercise the
/// `>= 1u64 << 48` arm (see the `uuid7_horizon_is_refused` test).
fn uuid7_bytes_at(millis: u64) -> Result<[u8; 16], RandomError> {
    // The field is 48 bits: refuse at/above 2^48 explicitly (Clock, mapped
    // to RuntimeError) rather than let the builder truncate silently.
    if millis >= 1u64 << 48 {
        return Err(RandomError::Clock(
            "unix milliseconds do not fit in the 48-bit uuid7 timestamp field".to_string(),
        ));
    }
    let mut counter_random = [0u8; 10];
    Source::new(None).fill(&mut counter_random)?;
    Ok(
        *UuidBuilder::from_unix_timestamp_millis(millis, &counter_random)
            .into_uuid()
            .as_bytes(),
    )
}

/// `uuid7`'s core: the canonical string of [`uuid7_bytes`]' buffer. The
/// pre-format bytes ARE the construction (timestamp read + one OS draw +
/// the builder's field layout); this spelling is their canonical
/// formatting.
pub fn uuid7() -> Result<String, RandomError> {
    Ok(Uuid::from_bytes(uuid7_bytes()?).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The seeded literals below are the same pins the Python suite freezes
    // (tests/test_random.py computes them against its independent PCG32 +
    // ChaCha20 + Lemire oracle; these crate-side copies pin the core without
    // the pyo3 layer, the b64 family's RFC-vector discipline). They were
    // computed from that oracle BEFORE the length-first refactor's
    // implementation changed, and then matched by it. One cross-reference
    // the old byte-fill engines carried is deliberately gone: "uuid4(seed=0)'s
    // first hex digits are hex(8, seed=0)'s" was an artifact of both
    // consumers reading the same stream bytes — hex char-samples now, so the
    // relationship is false, and the structural pin is the delegation
    // equality below instead.

    #[test]
    fn lemire_threshold_matches_2_pow_64_mod_n() {
        // The rejection threshold the sampler computes as (-n) mod n must
        // equal 2^64 mod n for every n, the identity the no-bias argument
        // rests on; 0 exactly when n is a power of two (no rejection then).
        for n in 1u64..=257 {
            assert_eq!(
                n.wrapping_neg() % n,
                ((1u128 << 64) % n as u128) as u64,
                "n = {n}"
            );
        }
        for n in [
            62u64,
            u32::MAX as u64,
            u64::MAX,
            u64::MAX - 1,
            1 << 32,
            1 << 63,
        ] {
            assert_eq!(
                n.wrapping_neg() % n,
                ((1u128 << 64) % n as u128) as u64,
                "n = {n}"
            );
        }
        assert_eq!(64u64.wrapping_neg() % 64, 0);
    }

    #[test]
    fn lemire_over_a_single_element_alphabet_always_returns_zero() {
        let mut words = Words::new(Source::new(Some(0)));
        for _ in 0..256 {
            assert_eq!(lemire_below(&mut words, 1).unwrap(), 0);
        }
    }

    #[test]
    fn lemire_rejection_consumes_the_next_word() {
        // n = 3: the threshold t = 2^64 mod 3 is 1, and x = 0 gives the
        // product 0 whose low half 0 falls below it: rejected. (For
        // powers of two t is 0 and even x = 0 accepts — no rejection
        // region at all.)
        assert_eq!(lemire_accept(0, 3), None);
        assert_eq!(lemire_accept(0, 64), Some(0));
        // A rejecting head word followed by a known word: the accepted
        // index is the SECOND word's mapping, proving the redraw
        // consumed the next word rather than returning or stalling.
        // (u64::MAX * 3) >> 64 is 2, accepted on the fast path
        // (low half 2^64 - 3 >= 3).
        let mut words = Words::for_test(&[0, u64::MAX]);
        assert_eq!(lemire_below(&mut words, 3).unwrap(), 2);
        assert_eq!(words.pos, 16, "both words must be consumed");
    }

    #[test]
    fn chacha20_seed_zero_first_block_is_pinned() {
        // The derivation + block 0 through rand_chacha itself (not the
        // Python oracle): seed 0's PCG32 key streams this block first.
        // Cross-check: these are the 16 pre-mask bytes behind the
        // uuid4(seed=0) golden (byte 6 pre-mask 0x3c, masked to 0x4c by
        // the version-4 field layout; byte 8 already variant-conformant).
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let mut block = [0u8; 64];
        rng.fill_bytes(&mut block);
        assert_eq!(
            &block[..16],
            &[
                0xb2, 0xf7, 0xf5, 0x81, 0xd6, 0xde, 0x3c, 0x06, 0xa8, 0x22, 0xfd, 0x6e, 0x7e, 0x82,
                0x65, 0xfb
            ]
        );
    }

    #[test]
    fn lemire_stays_uniform_at_scale_and_is_not_modulo() {
        // HIGH-2's two halves in one test. 300k seeded draws over the
        // 62-symbol alphabet (seed 0): the chi-square over 61 degrees of
        // freedom must stay far under the 99.9% critical value (~109 —
        // measured ~52), and every bucket within ±500 of the 4838.7
        // mean (measured extremes 4706/4940). The bounds prove
        // uniformity; they alone cannot tell Lemire from `%` (a modulo
        // map is uniform too) — so the test ALSO asserts the engine is
        // not `%` on the same stream: the first 64 draws' Lemire
        // high-bits indices must differ from the naive low-bits
        // `x % 62` indices (a `%` transcription fails here on draw 0).
        // The exact-value identity at scale lives in the Python suite
        // (a 300k-draw digest pin, whose modulo-mapped digest differs
        // in full); the committed goldens above pin determinism, never
        // unbiasedness — this test owns the unbiasedness claim.
        let out = random_string(300_000, BASE62_CHARS, Some(0)).unwrap();
        let mut counts = [0u64; 62];
        for b in out.bytes() {
            let idx = BASE62_CHARS.bytes().position(|b2| b2 == b).unwrap();
            counts[idx] += 1;
        }
        let mean = 300_000f64 / 62.0;
        let chi2: f64 = counts
            .iter()
            .map(|&c| (c as f64 - mean).powi(2) / mean)
            .sum();
        assert!(chi2 < 110.0, "chi-square {chi2} over 61 df is not uniform");
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                (c as f64 - mean).abs() < 500.0,
                "bucket {i} count {c} too far from {mean}"
            );
        }
        let mut words = Words::new(Source::new(Some(0)));
        let lemire_seq: Vec<u64> = (0..64)
            .map(|_| lemire_below(&mut words, 62).unwrap())
            .collect();
        let mut words = Words::new(Source::new(Some(0)));
        let modulo_seq: Vec<u64> = (0..64).map(|_| words.next_u64().unwrap() % 62).collect();
        assert_ne!(
            lemire_seq, modulo_seq,
            "Lemire must not degenerate to x % n on the same stream"
        );
    }

    #[test]
    fn the_char_engine_is_prefix_continuous_across_lengths() {
        // The sampler consumes u64 words in order, so a longer seeded draw
        // extends a shorter one character-for-character — the word-order
        // property that makes every golden pin length-independent on the
        // char path. (The old byte-fill prefix test died with the byte path:
        // the uuids are its only consumers now, and uuid4's fixed 16-byte
        // draw is pinned by its own goldens.)
        assert_eq!(
            &random_hex(64, Some(5)).unwrap()[..32],
            &random_hex(32, Some(5)).unwrap()
        );
        assert_eq!(
            &random_b64url(43, Some(9)).unwrap()[..22],
            &random_b64url(22, Some(9)).unwrap()
        );
    }

    #[test]
    fn seeded_hex_goldens() {
        // The same literals the Python suite pins (computed there against
        // its independent PCG32 + ChaCha20 + Lemire oracle first): these
        // crate-side copies pin the core without the pyo3 layer. Odd lengths
        // are pinned members of the ladder (31: a legal hex id shape).
        assert_eq!(random_hex(0, Some(0)).unwrap(), "");
        assert_eq!(random_hex(8, Some(0)).unwrap(), "0fd2e314");
        assert_eq!(random_hex(8, Some(1)).unwrap(), "9286e6a4");
        assert_eq!(
            random_hex(31, Some(5)).unwrap(),
            "9fddbe3cca89ec73270d1f133677747"
        );
        assert_eq!(
            random_hex(64, Some(7)).unwrap(),
            "08f4267dbf6ea8fbab86463bb680c70710e85e4f03affac31420c55574847728"
        );
    }

    #[test]
    fn seeded_b62_goldens() {
        assert_eq!(random_b62(0, Some(0)).unwrap(), "");
        assert_eq!(random_b62(22, Some(0)).unwrap(), "1yrBtE6FUlG59Zjj3K2vVn");
        assert_eq!(random_b62(22, Some(1)).unwrap(), "c9XQvNdIRcHcYnEMMFqaNP");
    }

    #[test]
    fn seeded_b64url_goldens() {
        assert_eq!(random_b64url(0, Some(0)).unwrap(), "");
        assert_eq!(random_b64url(4, Some(0)).unwrap(), "B-3L");
        assert_eq!(
            random_b64url(22, Some(42)).unwrap(),
            "gaGKKW0cEXGu8nuERoMZFe"
        );
        assert_eq!(
            random_b64url(43, Some(7)).unwrap(),
            "Bi8SLaf0s_a4pi-vqthbTaOstZjDweDcEC5hW7S_CNp"
        );
    }

    #[test]
    fn seeded_uuid4_goldens() {
        assert_eq!(
            uuid4(Some(0)).unwrap(),
            "b2f7f581-d6de-4c06-a822-fd6e7e8265fb"
        );
        assert_eq!(
            uuid4(Some(1)).unwrap(),
            "9a374450-4560-439e-8670-b7a17d492b27"
        );
        assert_eq!(
            uuid4(Some(42)).unwrap(),
            "7848b5d7-11bc-4883-9963-17a3f9c90269"
        );
    }

    #[test]
    fn seeded_uuid4_bytes_goldens() {
        // The uuid4 goldens' own buffers, unhyphenated — the same literals
        // the string pins commit, as bytes — plus the DRY pin: the string
        // spelling is the canonical formatting of the bytes spelling, one
        // construction behind both.
        assert_eq!(
            uuid4_bytes(Some(0)).unwrap(),
            [
                0xb2, 0xf7, 0xf5, 0x81, 0xd6, 0xde, 0x4c, 0x06, 0xa8, 0x22, 0xfd, 0x6e, 0x7e, 0x82,
                0x65, 0xfb
            ]
        );
        assert_eq!(
            uuid4_bytes(Some(42)).unwrap(),
            [
                0x78, 0x48, 0xb5, 0xd7, 0x11, 0xbc, 0x48, 0x83, 0x99, 0x63, 0x17, 0xa3, 0xf9, 0xc9,
                0x02, 0x69
            ]
        );
        for seed in [0u64, 1, 42] {
            assert_eq!(
                uuid4(Some(seed)).unwrap(),
                Uuid::from_bytes(uuid4_bytes(Some(seed)).unwrap()).to_string()
            );
        }
    }

    #[test]
    fn seeded_multibyte_alphabet_golden() {
        // length 9, alphabet "éüß漢", seed 2 — the same literal the Python
        // suite pins: multibyte characters sampled as characters.
        assert_eq!(random_string(9, "éüß漢", Some(2)).unwrap(), "éüßßé漢漢éé");
    }

    #[test]
    fn the_named_spellings_delegate_to_the_string_engine_exactly() {
        for seed in [0u64, 1, 42, u64::MAX] {
            for length in [0usize, 1, 16, 128] {
                assert_eq!(
                    random_b62(length, Some(seed)).unwrap(),
                    random_string(length, BASE62_CHARS, Some(seed)).unwrap()
                );
                assert_eq!(
                    random_hex(length, Some(seed)).unwrap(),
                    random_string(length, HEX_CHARS, Some(seed)).unwrap()
                );
                assert_eq!(
                    random_b64url(length, Some(seed)).unwrap(),
                    random_string(length, B64URL_CHARS, Some(seed)).unwrap()
                );
            }
        }
    }

    #[test]
    fn the_empty_alphabet_is_refused_before_any_entropy_is_drawn() {
        assert_eq!(random_string(8, "", None), Err(RandomError::EmptyAlphabet));
        assert_eq!(
            random_string(0, "", Some(1)),
            Err(RandomError::EmptyAlphabet)
        );
    }

    #[test]
    fn error_messages_carry_the_reason() {
        assert_eq!(
            RandomError::EmptyAlphabet.message(),
            "alphabet must be a non-empty str"
        );
        assert_eq!(
            RandomError::Os("getrandom: not supported".into()).message(),
            "operating system entropy source failed: getrandom: not supported"
        );
        assert_eq!(
            RandomError::Memory(4611686018427387904).message(),
            "cannot allocate 4611686018427387904 bytes for the output string"
        );
    }

    #[test]
    fn an_impossible_length_is_a_memory_error_not_an_abort() {
        // Three refusal routes, all the plain `Memory` error value — where
        // the `with_capacity` spelling this replaces aborted the process
        // (SIGABRT through Rust's default allocation-failure handler).
        // 2^62 one-byte chars: inside the reserve's own capacity limit, so
        // the ALLOCATOR refuses it (past the userspace address space on any
        // 64-bit target, deterministically).
        assert_eq!(
            random_string(1 << 62, "ab", None),
            Err(RandomError::Memory(1 << 62))
        );
        // 2^61 four-byte chars: the worst case 2^63 is past `isize::MAX`,
        // so the reserve's capacity check refuses WITHOUT consulting the
        // allocator.
        assert_eq!(
            random_string(1 << 61, "\u{1F600}", None),
            Err(RandomError::Memory(1 << 63))
        );
        // 2^62 four-byte chars: the worst-case product saturates to
        // `usize::MAX` — the arithmetic itself never panics on the way to
        // the refusal.
        assert_eq!(
            random_string(1 << 62, "\u{1F600}", None),
            Err(RandomError::Memory(usize::MAX))
        );
        // And the refusal precedes any entropy draw: a seeded stream is
        // not consumed by a call that cannot build its output.
        assert_eq!(
            random_hex(1 << 62, Some(0)),
            Err(RandomError::Memory(1 << 62))
        );
    }

    #[test]
    fn uuid7_horizon_is_refused() {
        // The 48-bit horizon arm, reachable only through the
        // timestamp-parameterized spelling (no real clock reads year
        // 10,892): below the horizon builds, at/above refuses with Clock.
        assert!(uuid7_bytes_at((1u64 << 48) - 1).is_ok());
        assert_eq!(
            uuid7_bytes_at(1u64 << 48),
            Err(RandomError::Clock(
                "unix milliseconds do not fit in the 48-bit uuid7 timestamp field".to_string()
            ))
        );
        assert_eq!(
            uuid7_bytes_at(u64::MAX),
            Err(RandomError::Clock(
                "unix milliseconds do not fit in the 48-bit uuid7 timestamp field".to_string()
            ))
        );
    }

    #[test]
    fn huge_alphabet_single_draw_stays_linear() {
        // The O(alphabet) + O(length) cost pin at small scale: a 100k-char
        // alphabet with length 1 builds exactly one output char — the
        // materialization pass runs once, one draw follows, and the seeded
        // spelling is deterministic.
        let alphabet: String = (0..100_000u32)
            .map(|i| char::from_u32(0xE000 + i).expect("private-use range is valid"))
            .collect();
        assert_eq!(alphabet.chars().count(), 100_000);
        let first = random_string(1, &alphabet, Some(0)).unwrap();
        assert_eq!(first.chars().count(), 1);
        assert!(alphabet.contains(&first[..]));
        assert_eq!(first, random_string(1, &alphabet, Some(0)).unwrap());
        let unseeded = random_string(1, &alphabet, None).unwrap();
        assert_eq!(unseeded.chars().count(), 1);
        assert!(alphabet.contains(&unseeded[..]));
    }

    #[test]
    fn unseeded_outputs_have_the_documented_shapes() {
        let hex = random_hex(32, None).unwrap();
        assert_eq!(hex.len(), 32);
        assert!(
            hex.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        );
        // Odd lengths are first-class on the unseeded path too.
        assert_eq!(random_hex(31, None).unwrap().len(), 31);
        let b62 = random_b62(22, None).unwrap();
        assert_eq!(b62.len(), 22);
        assert!(b62.bytes().all(|b| b.is_ascii_alphanumeric()));
        // Any length, including the non-multiple-of-4 token shapes; '=' can
        // never appear (no padded spelling exists).
        let b64 = random_b64url(3001, None).unwrap();
        assert_eq!(b64.len(), 3001);
        assert!(
            b64.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        );
        assert!(!b64.contains('='));
    }

    #[test]
    fn unseeded_uuid_shapes_and_distinctness() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2048 {
            let v4 = uuid4(None).unwrap();
            let v7 = uuid7().unwrap();
            for (value, version) in [(v4.as_str(), b'4'), (v7.as_str(), b'7')] {
                assert_eq!(value.len(), 36);
                assert_eq!(&value[8..9], "-");
                assert_eq!(&value[13..14], "-");
                assert_eq!(&value[18..19], "-");
                assert_eq!(&value[23..24], "-");
                assert_eq!(value.as_bytes()[14], version);
                assert!(matches!(value.as_bytes()[19], b'8' | b'9' | b'a' | b'b'));
                assert!(value.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
            }
            assert!(seen.insert(v4));
        }
    }

    #[test]
    fn unseeded_uuid_bytes_shapes_and_distinctness() {
        // The bytes spellings carry the same field layout on the raw
        // buffer: version nibble at byte 6's high half (4 / 7), RFC 4122
        // variant at byte 8's high nibble (8..=0xb) — and the same
        // distinctness arithmetic as the string spellings.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2048 {
            let v4 = uuid4_bytes(None).unwrap();
            let v7 = uuid7_bytes().unwrap();
            for (bytes, version) in [(&v4, 4u8), (&v7, 7u8)] {
                assert_eq!(bytes.len(), 16);
                assert_eq!(bytes[6] >> 4, version);
                assert!((8..=0xb).contains(&(bytes[8] >> 4)));
            }
            assert!(seen.insert(v4));
            assert!(seen.insert(v7));
        }
    }
}
