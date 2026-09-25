//! `rank_fusion_impl` never panics on arbitrary fused/metric inputs: the
//! fusion emits exactly the distinct ids that received votes (vote
//! existence, not score positivity — a denormal weight can underflow a
//! doc's every vote to 0.0 and the doc still appears), in
//! score-descending order with ties broken by first appearance, all
//! scores finite and non-negative; the metrics answer floats in [0, 1] for
//! any flag vector and any k >= 1.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
struct Input {
    /// The ranked lists, as RAW label values the target first remaps to
    /// the dense dedup-index space the py layer materializes (one index
    /// per distinct id, assigned on first sight), the core's contract
    /// is a dense id table, so the hostile shapes a caller can actually
    /// reach are duplicates within a list, overlaps across lists, and
    /// empty lists, all of which the remap preserves.
    lists: Vec<Vec<u32>>,
    /// A padding added to the document count (bounded: the core's
    /// never-voted-index filter must drop the padding, never emit it).
    doc_count_bias: u8,
    /// k, steered across the small-k regime the damping constant lives
    /// in (validated to >= 1 here, the pyo3 layer's contract).
    k: u8,
    /// Per-list weights, drawn across the None/short/exact spellings the
    /// binding can produce: bit 0 selects weighted (the all-weights
    /// spelling), bit 1 pads the slice short by one (the core defaults
    /// the missing tail to 1.0; the binding rejects a mismatch, the
    /// padding shape is the core's own contract), and each weight is a
    /// positive finite f64 assembled from bit patterns that avoid the
    /// zero/negative/NaN/inf domain the binding rejects.
    weight_bits: u32,
    /// The metric rankings' relevance flags and k.
    flags: Vec<bool>,
    metric_k: u8,
}

fuzz_target!(|input: Input| {
    if input.lists.is_empty() {
        return; // the zero-list ValueError is the binding's, pinned py-side
    }
    // The dense remap (the binding's dedup table, spelled in Rust):
    // first-appearance indices over the raw labels.
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let lists: Vec<Vec<u32>> = input
        .lists
        .iter()
        .map(|list| {
            list.iter()
                .map(|label| {
                    let next = remap.len() as u32;
                    *remap.entry(*label).or_insert(next)
                })
                .collect()
        })
        .collect();
    let k = (input.k % 64) as u64 + 1;
    // The padded count: up to 255 never-voted indices past the real
    // table (the fuzz-hostile version of a caller's loose dedup table;
    // past this the padding is unbounded input amplification, which the
    // binding cannot produce; its ids.len() IS the distinct count).
    let n_docs = remap.len() + input.doc_count_bias as usize;

    // The per-list weights: positive finite f64s assembled from raw
    // bits (an exponent in the normal range, mantissa bits from the
    // fuzzer), then the None/short/exact spellings steered by the low
    // bits. The value-domain validation (positive, finite) is the
    // binding's; the core consumes whatever slice it is handed.
    let weights: Option<Vec<f64>> = if input.weight_bits & 1 == 1 {
        let mut w: Vec<f64> = input
            .lists
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let bits = (input.weight_bits as u64) << 32 | (i as u64) << 16;
                // Mantissa bits only under a fixed 0x3FF exponent (the
                // [1.0, 2.0) normal range, always finite, never signed
                // or zero), the fuzzer steering the low bits.
                f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000)
            })
            .collect();
        if input.weight_bits & 2 == 2 && w.len() > 1 {
            w.pop(); // the short-slice spelling: the core defaults the tail
        }
        Some(w)
    } else {
        None
    };

    let fused = tors::rank_fusion_impl::rank_fuse(&lists, k, n_docs, weights.as_deref());

    // Every emitted score is finite and non-negative (0.0 is the
    // documented underflow shape: a voted doc whose denormal-weight
    // votes round away — this target's own weights stay in the normal
    // range, so positive here in practice); the emitted indices
    // are in range and pairwise distinct (one entry per voted doc).
    let mut seen = std::collections::HashSet::new();
    for (idx, score) in &fused {
        assert!(score.is_finite() && *score >= 0.0, "bad score {score}");
        assert!(*idx < n_docs as u32, "index out of range: {idx}");
        assert!(seen.insert(*idx), "duplicate fused index: {idx}");
    }
    // Exactly the voted documents are emitted.
    let voted: std::collections::HashSet<u32> =
        lists.iter().flat_map(|l| l.iter().copied()).collect();
    assert_eq!(seen, voted, "fused set != voted set");
    // Score-descending, ties by first appearance: scores are stored per
    // index, so re-derive them and check the order pairwise (the same
    // weighted formula, the slice as-spelled, short tails at 1.0).
    let mut scores = vec![0.0f64; n_docs];
    for (list_idx, list) in lists.iter().enumerate() {
        let weight = weights
            .as_deref()
            .and_then(|w| w.get(list_idx).copied())
            .unwrap_or(1.0);
        for (position, &doc) in list.iter().enumerate() {
            scores[doc as usize] += weight / (k as f64 + position as f64 + 1.0);
        }
    }
    for w in fused.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (sa, sb) = (scores[a.0 as usize], scores[b.0 as usize]);
        assert!(
            sa > sb || (sa == sb && a.0 < b.0),
            "order violated: {a:?} then {b:?} (scores {sa}, {sb})"
        );
    }

    // The metrics: any flag vector, any k >= 1, a float in [0, 1]
    // every time, well-defined (never NaN) on empty inputs.
    let metric_k = (input.metric_k % 128) as usize + 1;
    let flags = input.flags;
    let ndcg = tors::rank_fusion_impl::ndcg_at_k(
        &flags
            .iter()
            .map(|h| if *h { 1.0 } else { 0.0 })
            .collect::<Vec<f64>>(),
        vec![1.0; flags.len().max(1)],
        metric_k.min(flags.len().max(1)),
    );
    assert!((0.0..=1.0).contains(&ndcg), "ndcg out of range: {ndcg}");
    let m = tors::rank_fusion_impl::mrr(&flags);
    assert!(
        (0.0..=1.0).contains(&m) && m.is_finite(),
        "mrr out of range: {m}"
    );
    let recall =
        tors::rank_fusion_impl::recall_at_k(&flags, flags.iter().filter(|h| **h).count(), metric_k);
    assert!(
        (0.0..=1.0).contains(&recall),
        "recall out of range: {recall}"
    );
    let precision = tors::rank_fusion_impl::precision_at_k(&flags, metric_k);
    assert!(
        (0.0..=1.0).contains(&precision),
        "precision out of range: {precision}"
    );
});
