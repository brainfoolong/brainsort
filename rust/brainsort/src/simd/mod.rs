//! The vector kernels. Only x86-64 has any; every other target runs the
//! scalar code, which is also what the vector paths fall back to for their
//! tails and what `--cfg brainsort_no_simd` selects on x86-64.
#[cfg(all(target_arch = "x86_64", not(brainsort_no_simd)))]
pub mod x86;
