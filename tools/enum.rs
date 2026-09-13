// Enumerate every codepoint Rust's std classifies as alphanumeric.
//
// Regen recipe step 1 of 2 (step 2 is tools/gen_word_demote_table.py):
//   rustc -O tools/enum.rs -o /tmp/enum && /tmp/enum > /tmp/rust_alnum.txt
//
// Prints one decimal codepoint per line for every cp in 0..=0x10FFFF where
// `char::from_u32(cp).is_some_and(|c| c.is_alphanumeric())`. The output is
// the Rust side of the WORD_DEMOTE_RANGES intersection; the Python side is
// the running interpreter's `re.compile(r"\w")`. Pinned toolchain for the
// committed table: see tools/gen_word_demote_table.py (PINNED_RUSTC).
fn main() {
    for cp in 0..=0x10FFFFu32 {
        if char::from_u32(cp).is_some_and(|c| c.is_alphanumeric()) {
            println!("{cp}");
        }
    }
}
