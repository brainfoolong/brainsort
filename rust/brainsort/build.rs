// Declares the `brainsort_no_simd` cfg so `--cfg brainsort_no_simd` (the twin
// of the C++ BRAINSORT_NO_SIMD) does not trip the unexpected-cfg lint.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(brainsort_no_simd)");
    println!("cargo::rerun-if-changed=build.rs");
}
