//! The random-generation family's pure-Rust core: `random_string`,
//! `random_hex`, `random_b62`, `random_b64url`, `uuid4`, `uuid7` (the pyo3
//! bindings live in `py/random.rs`; the criterion bench in `benches/random.rs`
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
//!   The derivation is rand_core's documented-stable default (its docs call
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
//!   byte fill is for. (Before the length-first refactor `random_hex` and
//!   `random_b64url` byte-filled and encoded here too; the maintainer
//!   ergonomics directive — backend devs think "I want a base62 id X
//!   characters long", so every token spelling takes the output length
//!   directly — moved both onto the char-sampling engine, and the byte
//!   path shrank to the uuids.)
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
use uuid::Builder as UuidBuilder;

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
    /// Unix epoch (clock skew backwards past 1970), or (theoretically) past
    /// the u64 millisecond horizon (year ~584 million).
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
/// 128 words) so the unseeded spelling pays one OS syscall per block rather
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
fn lemire_below(words: &mut Words, n: u64) -> Result<u64, RandomError> {
    loop {
        let x = words.next_u64()?;
        let product = (x as u128) * (n as u128);
        let low = product as u64;
        // Fast path (probability 1 - n/2^64): the low half alone certifies
        // the draw, no division needed — the "nearly divisionless" property.
        if low >= n {
            return Ok((product >> 64) as u64);
        }
        // Rare: compare against the rejection threshold t = 2^64 mod n.
        // (-n) mod n == (2^64 - n) mod n == 2^64 mod n, and for powers of
        // two it is 0 (exact division, no rejection region at all).
        let threshold = n.wrapping_neg() % n;
        if low >= threshold {
            return Ok((product >> 64) as u64);
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
/// show 4 distinct final characters; this spelling shows all 64 — and
/// lengths that are not valid base64 output lengths, like 41, are legal
/// here). There is no `padded` spelling at all: padding is an encoding
/// concept, not a token concept, and `=` can never appear.
pub fn random_b64url(length: usize, seed: Option<u64>) -> Result<String, RandomError> {
    random_string(length, B64URL_CHARS, seed)
}

/// `uuid4`'s core: one 16-byte fill handed to the uuid crate's zero-feature
/// `Builder::from_random_bytes` (it sets the version-4 and RFC 4122 variant
/// nibbles; verified in the resolved source: byte 6 `(b & 0x0f) | 0x40`,
/// byte 8 `(b & 0x3f) | 0x80`), formatted canonical: 36 chars, lowercase
/// hex, hyphens at 8/13/18/23. Seeded calls are pure functions of the seed
/// (the pins and the Python oracle freeze exactly this construction).
pub fn uuid4(seed: Option<u64>) -> Result<String, RandomError> {
    let mut bytes = [0u8; 16];
    Source::new(seed).fill(&mut bytes)?;
    Ok(UuidBuilder::from_random_bytes(bytes)
        .into_uuid()
        .to_string())
}

/// `uuid7`'s core: see the module docs' "uuid7's boundary" section for the
/// honest statement (probabilistically unique, not counter-monotonic, no
/// seed parameter because the timestamp is external state). The 10-byte
/// counter/random draw is one OS fill; the builder consumes 74 of its 80
/// bits (rand_a's top nibble and rand_b's top 2 bits are the version and
/// variant nibbles' territory).
pub fn uuid7() -> Result<String, RandomError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| {
            RandomError::Clock(format!("system clock is before the Unix epoch: {err}"))
        })?;
    let millis = u64::try_from(duration.as_millis())
        .map_err(|_| RandomError::Clock("unix milliseconds do not fit in u64".to_string()))?;
    let mut counter_random = [0u8; 10];
    Source::new(None).fill(&mut counter_random)?;
    Ok(
        UuidBuilder::from_unix_timestamp_millis(millis, &counter_random)
            .into_uuid()
            .to_string(),
    )
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
}
