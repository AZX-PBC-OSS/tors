//! Merkle tree primitives: the pure-Rust cores of `tors.merkle_root` and
//! `tors.merkle_diff`.
//!
//! # Domain separation (why this doesn't just use `rs_merkle::algorithms::Sha256`)
//!
//! Leaves hash as `SHA-256(0x00 || chunk)`; internal nodes hash as
//! `SHA-256(0x01 || left || right)`: the RFC 6962 / Certificate
//! Transparency convention. `rs_merkle`'s *built-in* `Sha256Algorithm`
//! does NOT do this: its `hash()` is undifferentiated `SHA-256(data)`, and
//! its default `Hasher::concat_and_hash` feeds `SHA-256(left || right)`
//! through that same function; verified directly against the crate's
//! vendored source (`hasher.rs`'s default `concat_and_hash`,
//! `algorithms/sha256.rs`'s `hash`, rs_merkle 1.5.0), not just its docs.
//! With no domain byte, a leaf hash and an internal-node hash live in the
//! same space: the exact ambiguity RFC 6962 domain separation exists to
//! close (a forged proof could present an internal node's hash as if it
//! were some leaf's digest). `merkle_root`/`merkle_diff` don't expose
//! proof generation yet, but the hash SCHEME is part of the root's output
//! contract from v1 regardless: changing it later would silently change
//! every previously-computed root, so it needs to be correct now rather
//! than patched in when proofs are added. `DomainSeparatedSha256` below is
//! tors's own `Hasher` impl, not the crate's default.
//!
//! # Odd-sized layers
//!
//! `rs_merkle`'s default `concat_and_hash` PROMOTES an unpaired left node
//! to the next layer unchanged (its `None => *left` arm) rather than
//! duplicating it against itself. Duplication is Bitcoin's original
//! convention and the actual mechanism CVE-2012-2459 exploited: two
//! differently-shaped leaf lists (one a duplicate-padded version of a
//! shorter list) could produce the SAME root. tors keeps rs_merkle's
//! promotion behavior (only the two-child case gets the domain-separating
//! prefix), which is the standard mitigation and matches RFC 6962.

use rs_merkle::{Hasher, MerkleTree};
use sha2::{Digest, Sha256};

const LEAF_PREFIX: u8 = 0x00;
const NODE_PREFIX: u8 = 0x01;

#[derive(Clone)]
struct DomainSeparatedSha256;

impl Hasher for DomainSeparatedSha256 {
    type Hash = [u8; 32];

    // Required by the trait, but NOT part of this tree's construction path:
    // `concat_and_hash` below is overridden instead of relying on the
    // default (the only caller of `hash` in rs_merkle's own tree-building
    // code). Kept as a plain, undifferentiated SHA-256 for trait
    // completeness only; nothing in this module calls it.
    fn hash(data: &[u8]) -> [u8; 32] {
        Sha256::digest(data).into()
    }

    fn concat_and_hash(left: &[u8; 32], right: Option<&[u8; 32]>) -> [u8; 32] {
        match right {
            Some(right) => {
                let mut hasher = Sha256::new();
                hasher.update([NODE_PREFIX]);
                hasher.update(left);
                hasher.update(right);
                hasher.finalize().into()
            }
            None => *left,
        }
    }
}

fn leaf_hash(chunk: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([LEAF_PREFIX]);
    hasher.update(chunk);
    hasher.finalize().into()
}

/// The Merkle root over `chunks`, as lowercase hex. `None` for an empty
/// input: the pyo3 wrapper turns that into `ValueError`: "root of no
/// chunks" has no non-arbitrary value, and returning some fixed sentinel
/// hash risks being mistaken for a real chunk's digest by a caller
/// comparing roots. A single chunk's root is just its own leaf hash (no
/// internal node needed).
pub fn merkle_root(chunks: &[&[u8]]) -> Option<String> {
    if chunks.is_empty() {
        return None;
    }
    let leaves: Vec<[u8; 32]> = chunks.iter().map(|c| leaf_hash(c)).collect();
    let tree = MerkleTree::<DomainSeparatedSha256>::from_leaves(&leaves);
    let root = tree
        .root()
        .expect("from_leaves on a non-empty leaf slice always has a root");
    Some(const_hex::encode(root))
}

/// Indices where `chunks_a[i] != chunks_b[i]`, comparing chunk DIGESTS
/// (a fixed 32-byte cost per comparison) rather than raw chunk contents.
/// Every index at or beyond the shorter list's length is reported: there
/// is no counterpart chunk to compare against, so a length mismatch is, in
/// full, "differs at every trailing index," not a partial answer.
///
/// This does not walk a tree. With both chunk lists held locally as
/// random-access arrays, hashing each chunk once already answers "does
/// chunk i differ" in O(1) per index afterward: a flat scan over the
/// (unavoidable: every chunk must be hashed once regardless) digest
/// arrays does exactly the same work a tree walk would, without the tree.
/// A Merkle tree's "skip identical subtrees" advantage matters when the
/// comparison itself is expensive to perform per-index (e.g. over a
/// network, without transferring full subtrees); not here, where the
/// O(n) hashing pass IS the entire cost and a tree buys nothing further.
/// `merkle_root` still builds a real tree: that's its actual job;
/// `merkle_diff` simply doesn't need one.
pub fn merkle_diff(chunks_a: &[&[u8]], chunks_b: &[&[u8]]) -> Vec<usize> {
    let len = chunks_a.len().max(chunks_b.len());
    let mut out = Vec::new();
    for i in 0..len {
        let differs = match (chunks_a.get(i), chunks_b.get(i)) {
            (Some(a), Some(b)) => leaf_hash(a) != leaf_hash(b),
            _ => true,
        };
        if differs {
            out.push(i);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chunks_has_no_root() {
        assert_eq!(merkle_root(&[]), None);
    }

    #[test]
    fn single_chunk_root_is_its_own_leaf_hash() {
        let root = merkle_root(&[b"abc"]).unwrap();
        let expected = const_hex::encode(leaf_hash(b"abc"));
        assert_eq!(root, expected);
    }

    #[test]
    fn root_is_deterministic() {
        let chunks: &[&[u8]] = &[b"a", b"b", b"c", b"d"];
        assert_eq!(merkle_root(chunks), merkle_root(chunks));
    }

    #[test]
    fn root_is_order_sensitive() {
        let a: &[&[u8]] = &[b"a", b"b", b"c"];
        let b: &[&[u8]] = &[b"b", b"a", b"c"];
        assert_ne!(merkle_root(a), merkle_root(b));
    }

    #[test]
    fn root_differs_from_naive_undifferentiated_hash_of_the_same_leaves() {
        // Proves domain separation actually changes the output versus the
        // undifferentiated scheme rs_merkle's own built-in Sha256Algorithm
        // would produce: the leaf hash includes the 0x00 prefix, so it must
        // not equal a plain SHA-256 of the raw chunk.
        let root = merkle_root(&[b"a", b"b"]).unwrap();
        let plain_leaf_a = const_hex::encode(Sha256::digest(b"a"));
        assert_ne!(root, plain_leaf_a);
    }

    #[test]
    fn odd_leaf_count_promotes_unpaired_node_rather_than_duplicating() {
        // Three leaves: the tree has one paired node (hash(a,b)) and one
        // unpaired leaf (c) promoted as-is to the next layer, then paired
        // together at the root. If duplication were used instead (c paired
        // with itself), the root would differ from this hand-computed value.
        let leaf = |b: &[u8]| leaf_hash(b);
        let node = |l: [u8; 32], r: [u8; 32]| {
            let mut hasher = Sha256::new();
            hasher.update([NODE_PREFIX]);
            hasher.update(l);
            hasher.update(r);
            let out: [u8; 32] = hasher.finalize().into();
            out
        };
        let (a, b, c) = (leaf(b"a"), leaf(b"b"), leaf(b"c"));
        let ab = node(a, b);
        // c is promoted unchanged to the next layer (rs_merkle's `None =>
        // *left`), where it pairs with ab.
        let expected_root = const_hex::encode(node(ab, c));
        assert_eq!(merkle_root(&[b"a", b"b", b"c"]), Some(expected_root));
    }

    #[test]
    fn diff_finds_exactly_the_differing_indices() {
        let a: &[&[u8]] = &[b"a", b"b", b"c", b"d"];
        let b: &[&[u8]] = &[b"a", b"X", b"c", b"Y"];
        assert_eq!(merkle_diff(a, b), vec![1, 3]);
    }

    #[test]
    fn diff_of_identical_lists_is_empty() {
        let a: &[&[u8]] = &[b"a", b"b", b"c"];
        assert_eq!(merkle_diff(a, a), Vec::<usize>::new());
    }

    #[test]
    fn diff_reports_every_trailing_index_on_length_mismatch() {
        let a: &[&[u8]] = &[b"a", b"b"];
        let b: &[&[u8]] = &[b"a", b"b", b"c", b"d"];
        assert_eq!(merkle_diff(a, b), vec![2, 3]);
        assert_eq!(merkle_diff(b, a), vec![2, 3]);
    }

    #[test]
    fn diff_of_two_empty_lists_is_empty() {
        assert_eq!(merkle_diff(&[], &[]), Vec::<usize>::new());
    }

    #[test]
    fn diff_against_brute_force_reference() {
        // Random-ish mutation battery cross-checked against a naive
        // elementwise Vec<u8> comparison (not via hashes at all).
        let base: Vec<Vec<u8>> = (0u8..20).map(|i| vec![i; (i as usize) + 1]).collect();
        let mut mutated = base.clone();
        for idx in [2usize, 5, 5, 17] {
            if let Some(chunk) = mutated.get_mut(idx) {
                chunk.push(0xff);
            }
        }
        mutated.truncate(15);

        let a: Vec<&[u8]> = base.iter().map(|v| v.as_slice()).collect();
        let b: Vec<&[u8]> = mutated.iter().map(|v| v.as_slice()).collect();

        let expected: Vec<usize> = (0..a.len().max(b.len()))
            .filter(|&i| a.get(i).map(|s| s.to_vec()) != b.get(i).map(|s| s.to_vec()))
            .collect();
        assert_eq!(merkle_diff(&a, &b), expected);
    }
}
