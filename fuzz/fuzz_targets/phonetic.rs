//! Every phonetic code never panics on arbitrary text. The ASCII
//! pre-filter each spelling applies is the documented guard against the
//! upstream `rphonetic` panic class (accented and non-Latin input indexing
//! its ASCII mapping tables out of bounds), so this target's whole job is
//! raw adversarial strings against that guard.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|s: &str| {
    let _ = tors::phonetic_impl::soundex(s);
    let _ = tors::phonetic_impl::metaphone(s);
    let _ = tors::phonetic_impl::double_metaphone(s);
    let _ = tors::phonetic_impl::nysiis(s);
    let _ = tors::phonetic_impl::daitch_mokotoff(s);
    let _ = tors::phonetic_impl::refined_soundex(s);
});
