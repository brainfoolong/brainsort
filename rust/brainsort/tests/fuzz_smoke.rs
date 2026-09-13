//! The fuzz body on a few thousand seeded random inputs, so every `cargo
//! test` covers it.
mod fuzz_body;

#[test]
fn fuzz_smoke() {
    // Miri interprets: a few dozen inputs there, a few hundred on the quick knob.
    let iterations = if cfg!(miri) {
        40
    } else if std::env::var_os("BRAINSORT_TEST_QUICK").is_some() {
        300
    } else {
        3000
    };
    let mut s = 20260912u64;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for it in 0..iterations {
        let size = (next()
            % if cfg!(miri) {
                300
            } else if it % 10 == 0 {
                20000
            } else {
                600
            }) as usize;
        let modulus = if it % 3 == 0 { 4 } else { 256 };
        let bytes: Vec<u8> = (0..size).map(|_| (next() % modulus) as u8).collect();
        fuzz_body::fuzz_one(&bytes);
    }
}
