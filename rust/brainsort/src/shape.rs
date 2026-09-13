//! The compile-time shape of a key: its leaves, in order.
//!
//! Every key type reports its shape as a constant: a sequence of at most 32
//! leaves, each a fixed part of some bit width or a byte string, each
//! possibly reversed. The record type the sort uses is chosen from the
//! shape at compile time (see `record.rs`), so a key of two `i32` packs into
//! one 64-bit radix value while a key with a string part takes a composite
//! record. This is the port of `PartList` / `kParts` in the C++ `keys.hpp`.

/// One leaf of a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Part {
    /// A byte string (compared lexicographically as unsigned bytes) rather
    /// than a fixed part.
    pub bytes: bool,
    /// The width of a fixed part in bits, 1..=64; 0 for a byte string.
    pub bits: u8,
    /// The leaf sorts in descending order.
    pub desc: bool,
}

/// The most leaves a key may have.
pub const MAX_PARTS: usize = 32;

/// The leaves of a key type, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The leaves; only the first `n` are meaningful.
    pub parts: [Part; MAX_PARTS],
    /// The number of leaves. More than `MAX_PARTS` is recorded as
    /// `MAX_PARTS + 1` and rejected when the key is sorted.
    pub n: usize,
}

impl Shape {
    const EMPTY: Shape = Shape { parts: [Part { bytes: false, bits: 0, desc: false }; MAX_PARTS], n: 0 };

    /// A single fixed leaf of `bits` bits.
    pub const fn fixed(bits: u8) -> Shape {
        let mut s = Shape::EMPTY;
        s.parts[0] = Part { bytes: false, bits, desc: false };
        s.n = 1;
        s
    }
    /// A single byte-string leaf.
    pub const fn bytes() -> Shape {
        let mut s = Shape::EMPTY;
        s.parts[0] = Part { bytes: true, bits: 0, desc: false };
        s.n = 1;
        s
    }
    /// The leaves of `a` followed by the leaves of `b`.
    pub const fn concat(a: &Shape, b: &Shape) -> Shape {
        let mut s = *a;
        let mut i = 0;
        while i < b.n && i < MAX_PARTS {
            if a.n + i < MAX_PARTS {
                s.parts[a.n + i] = b.parts[i];
            }
            i += 1;
        }
        let n = a.n + b.n;
        s.n = if n > MAX_PARTS { MAX_PARTS + 1 } else { n };
        s
    }
    /// `s` repeated `count` times (the shape of an array of keys).
    pub const fn repeat(s: &Shape, count: usize) -> Shape {
        let mut out = Shape::EMPTY;
        let mut i = 0;
        while i < count {
            out = Shape::concat(&out, s);
            i += 1;
        }
        out
    }
    /// Every leaf reversed.
    pub const fn reversed(&self) -> Shape {
        let mut s = *self;
        let mut i = 0;
        while i < s.n && i < MAX_PARTS {
            s.parts[i].desc = !s.parts[i].desc;
            i += 1;
        }
        s
    }
    /// Between one and `MAX_PARTS` leaves.
    pub const fn is_valid(&self) -> bool {
        self.n >= 1 && self.n <= MAX_PARTS
    }
    /// No byte-string leaf.
    pub const fn all_fixed(&self) -> bool {
        let mut i = 0;
        while i < self.n && i < MAX_PARTS {
            if self.parts[i].bytes {
                return false;
            }
            i += 1;
        }
        true
    }
    /// Some byte-string leaf.
    pub const fn has_bytes(&self) -> bool {
        !self.all_fixed()
    }
    /// The sum of the fixed leaves' widths.
    pub const fn total_bits(&self) -> u32 {
        let mut b = 0u32;
        let mut i = 0;
        while i < self.n && i < MAX_PARTS {
            b += self.parts[i].bits as u32;
            i += 1;
        }
        b
    }
    /// The `i`-th leaf.
    pub const fn part(&self, i: usize) -> Part {
        self.parts[i]
    }
    /// The kind code of leaf `i`: 0 fixed, 1 bytes, 2 bytes reversed (the
    /// `Kinds` of the C++ `CompRec`).
    pub const fn kind(&self, i: usize) -> u8 {
        let p = self.parts[i];
        if !p.bytes {
            0
        } else if p.desc {
            2
        } else {
            1
        }
    }
}
