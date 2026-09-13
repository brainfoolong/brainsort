//! libFuzzer target: the shared fuzz body.
#![no_main]
#[path = "../../tests/fuzz_body/mod.rs"]
mod fuzz_body;
libfuzzer_sys::fuzz_target!(|data: &[u8]| fuzz_body::fuzz_one(data));
