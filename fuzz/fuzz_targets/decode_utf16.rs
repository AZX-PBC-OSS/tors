//! Never panics on arbitrary bytes under any byte order, and `utf16_is_valid`
//! never disagrees with `decode_strict`'s own success/failure.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tors::utf16_impl::ByteOrder;

#[derive(Arbitrary, Debug)]
enum FuzzByteOrder {
    Native,
    Little,
    Big,
}

impl From<FuzzByteOrder> for ByteOrder {
    fn from(b: FuzzByteOrder) -> Self {
        match b {
            FuzzByteOrder::Native => ByteOrder::Native,
            FuzzByteOrder::Little => ByteOrder::Little,
            FuzzByteOrder::Big => ByteOrder::Big,
        }
    }
}

#[derive(Arbitrary, Debug)]
struct Input {
    data: Vec<u8>,
    byteorder: FuzzByteOrder,
}

fuzz_target!(|input: Input| {
    let byteorder: ByteOrder = input.byteorder.into();
    let strict = tors::utf16_impl::decode_strict(&input.data, byteorder);
    let valid = tors::utf16_impl::is_valid(&input.data, byteorder);
    assert_eq!(
        strict.is_ok(),
        valid,
        "utf16_is_valid disagreed with decode_strict on {:?} ({byteorder:?})",
        input.data
    );
    if let Ok((s, _encoding)) = &strict {
        assert!(std::str::from_utf8(s.as_bytes()).is_ok());
    }

    let (replaced, _encoding) = tors::utf16_impl::decode_replace(&input.data, byteorder);
    assert!(std::str::from_utf8(replaced.as_bytes()).is_ok());
});
