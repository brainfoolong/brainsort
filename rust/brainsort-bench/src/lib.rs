//! The counted harness of the Rust port: the port of the C++ `sortbench`
//! pieces the golden test needs. The four element types with their
//! original-index `id`, the datasets (generated exactly as the C++
//! `datasets.hpp` does, from a `std::mt19937_64` port, so the inputs are
//! byte-identical), the trace hooks that count every read, write, compare,
//! table and key access and feed a fixed cache model, and the counted run
//! that produces the deterministic columns of `results/counts.csv`.
//!
//! Every number here is a pure function of (algorithm, input): the golden
//! test compares them with the C++ file, which proves the port runs the
//! same algorithm step for step.
#![allow(clippy::needless_range_loop)]
use brainsort::internals::*;
use std::cell::UnsafeCell;

pub mod counts;
pub mod datasets;
pub mod timing;

// ---- the counting allocator ----------------------------------------------------
// The scratch memory of any sort, seen through the global allocator: armed
// around one call, it records the peak of the bytes live and the number of
// allocations. brainsort's counted view arms it too, so its row in
// rust-counts.csv is measured like every other sort's and the check holds
// it to the trace's own accounting. Armed per thread: the sorts run on the
// calling thread, and a thread pool a crate keeps (rdst starts one on its
// first call) must not be able to add its own start-up allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

pub struct CountingAlloc;
/// The sizes of the first armed allocations, for the mismatch report.
const SIZES_KEPT: usize = 16;
// Everything per thread, const-initialised and without a destructor, so the
// allocator may read it at any time. Test threads count independently.
thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static ALLOC_CURRENT: Cell<usize> = const { Cell::new(0) };
    static ALLOC_PEAK: Cell<usize> = const { Cell::new(0) };
    static ALLOC_COUNT: Cell<u64> = const { Cell::new(0) };
    static ALLOC_SIZES: Cell<[usize; SIZES_KEPT]> = const { Cell::new([0; SIZES_KEPT]) };
}
#[inline]
fn armed() -> bool {
    // A thread being torn down reads as not armed.
    ARMED.try_with(|a| a.get()).unwrap_or(false)
}
fn alloc_grow(bytes: usize) {
    let cur = ALLOC_CURRENT.get() + bytes;
    ALLOC_CURRENT.set(cur);
    ALLOC_PEAK.set(ALLOC_PEAK.get().max(cur));
    let i = ALLOC_COUNT.get() as usize;
    ALLOC_COUNT.set(ALLOC_COUNT.get() + 1);
    if i < SIZES_KEPT {
        let mut sizes = ALLOC_SIZES.get();
        sizes[i] = bytes;
        ALLOC_SIZES.set(sizes);
    }
}
fn alloc_shrink(bytes: usize) {
    ALLOC_CURRENT.set(ALLOC_CURRENT.get().saturating_sub(bytes));
}
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() && armed() {
            alloc_grow(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() && armed() {
            alloc_grow(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if armed() {
            alloc_shrink(l.size());
        }
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new_size) };
        if !q.is_null() && armed() {
            alloc_shrink(l.size());
            alloc_grow(new_size);
        }
        q
    }
}
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// What the allocator saw on this thread while armed.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocStats {
    pub peak: usize,
    pub allocs: u64,
    /// Bytes still live at disarm: 0 unless the sort leaked.
    pub leak: usize,
    /// The sizes of the first allocations, in order.
    pub sizes: [usize; SIZES_KEPT],
}
pub fn alloc_arm() {
    ALLOC_CURRENT.set(0);
    ALLOC_PEAK.set(0);
    ALLOC_COUNT.set(0);
    ALLOC_SIZES.set([0; SIZES_KEPT]);
    ARMED.set(true);
}
pub fn alloc_disarm() -> AllocStats {
    ARMED.set(false);
    AllocStats { peak: ALLOC_PEAK.get(), allocs: ALLOC_COUNT.get(), leak: ALLOC_CURRENT.get(), sizes: ALLOC_SIZES.get() }
}

// ---- element types ---------------------------------------------------------

/// 8 bytes: int32 key.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Item {
    pub key: i32,
    pub id: u32,
}
/// 16 bytes: double key.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DblItem {
    pub key: f64,
    pub id: u32,
    pub pad: u32,
}
/// 16 bytes: int64 key.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct I64Item {
    pub key: i64,
    pub id: u32,
    pub pad: u32,
}
/// 16 bytes: string key (pointer + length).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct StrItem {
    pub ptr: *const u8,
    pub len: u32,
    pub id: u32,
}
impl Default for StrItem {
    fn default() -> Self {
        StrItem { ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0, id: 0 }
    }
}
impl PartialEq for StrItem {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id && self.bytes() == o.bytes()
    }
}
impl StrItem {
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the item points into a pool that outlives it.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len as usize) }
    }
}
unsafe impl Send for StrItem {}
unsafe impl Sync for StrItem {}

/// What the harness needs from an element type beyond `Elem`.
pub trait Item2: Elem + PartialEq + Send + Sync + 'static {
    const NAME: &'static str;
    fn id(&self) -> u32;
    fn set_id(&mut self, v: u32);
}
impl Item2 for Item {
    const NAME: &'static str = "int32";
    fn id(&self) -> u32 {
        self.id
    }
    fn set_id(&mut self, v: u32) {
        self.id = v;
    }
}
impl Item2 for DblItem {
    const NAME: &'static str = "double";
    fn id(&self) -> u32 {
        self.id
    }
    fn set_id(&mut self, v: u32) {
        self.id = v;
    }
}
impl Item2 for I64Item {
    const NAME: &'static str = "int64";
    fn id(&self) -> u32 {
        self.id
    }
    fn set_id(&mut self, v: u32) {
        self.id = v;
    }
}
impl Item2 for StrItem {
    const NAME: &'static str = "string";
    fn id(&self) -> u32 {
        self.id
    }
    fn set_id(&mut self, v: u32) {
        self.id = v;
    }
}

impl Elem for Item {
    type Key = u32;
    const CHUNKED: bool = false;
    const SIMD: SimdKind = SimdKind::I32;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        a.key < b.key
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        (a.key > b.key) as i32 - (a.key < b.key) as i32
    }
    #[inline(always)]
    fn compare_from(a: Self, b: Self, _c: i32) -> i32 {
        Self::compare(a, b)
    }
    #[inline(always)]
    fn radix_key(a: Self, _c: i32) -> u32 {
        (a.key as u32) ^ 0x8000_0000
    }
    #[inline(always)]
    fn chunk_ends(_a: Self, _c: i32) -> bool {
        true
    }
}
impl Elem for DblItem {
    type Key = u64;
    const CHUNKED: bool = false;
    const SIMD: SimdKind = SimdKind::F64;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        a.key < b.key
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        (a.key > b.key) as i32 - (a.key < b.key) as i32
    }
    #[inline(always)]
    fn compare_from(a: Self, b: Self, _c: i32) -> i32 {
        Self::compare(a, b)
    }
    #[inline(always)]
    fn radix_key(a: Self, _c: i32) -> u64 {
        let bits = a.key.to_bits();
        let sign = 0x8000_0000_0000_0000u64;
        if bits & sign != 0 { sign - (bits & !sign) } else { bits | sign }
    }
    #[inline(always)]
    fn chunk_ends(_a: Self, _c: i32) -> bool {
        true
    }
}
impl Elem for I64Item {
    type Key = u64;
    const CHUNKED: bool = false;
    const SIMD: SimdKind = SimdKind::I64;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        a.key < b.key
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        (a.key > b.key) as i32 - (a.key < b.key) as i32
    }
    #[inline(always)]
    fn compare_from(a: Self, b: Self, _c: i32) -> i32 {
        Self::compare(a, b)
    }
    #[inline(always)]
    fn radix_key(a: Self, _c: i32) -> u64 {
        (a.key as u64) ^ 0x8000_0000_0000_0000
    }
    #[inline(always)]
    fn chunk_ends(_a: Self, _c: i32) -> bool {
        true
    }
}
impl Elem for StrItem {
    type Key = u64;
    const CHUNKED: bool = true;
    const SIMD: SimdKind = SimdKind::None;
    #[inline(always)]
    fn less(a: Self, b: Self) -> bool {
        Self::compare_from(a, b, 0) < 0
    }
    #[inline(always)]
    fn compare(a: Self, b: Self) -> i32 {
        Self::compare_from(a, b, 0)
    }
    #[inline(always)]
    fn compare_from(a: Self, b: Self, chunk: i32) -> i32 {
        unsafe { str_compare_from(a.ptr, a.len, b.ptr, b.len, chunk as u32 * STR_CHUNK_BYTES) }
    }
    #[inline(always)]
    fn radix_key(a: Self, chunk: i32) -> u64 {
        unsafe { str_chunk_key(a.ptr, a.len, chunk) }
    }
    #[inline(always)]
    fn chunk_ends(a: Self, chunk: i32) -> bool {
        str_chunk_ends(a.len, chunk)
    }
    #[inline(always)]
    fn report_key_chunk(a: &Self, chunk: i32, sink: &mut impl FnMut(*const u8, usize)) {
        let off = chunk as u32 * STR_CHUNK_BYTES;
        sink(a.ptr.wrapping_add(off as usize), str_chunk_bytes(a.len, chunk) as usize);
    }
    #[inline(always)]
    fn report_key_compare(a: &Self, b: &Self, chunk: i32, sink: &mut impl FnMut(*const u8, usize)) {
        let off = chunk as u32 * STR_CHUNK_BYTES;
        let ex = unsafe { str_examined(a.ptr, a.len, b.ptr, b.len, off) } as usize;
        sink(a.ptr.wrapping_add(off as usize), ex);
        sink(b.ptr.wrapping_add(off as usize), ex);
    }
}

// ---- the trace --------------------------------------------------------------
// A fixed cache model: 64-byte lines, true LRU, an L1 of 32 KiB 8-way and an
// L2 of 1 MiB 16-way, probed only on an L1 miss. Addresses are not real:
// every buffer (the array, each scratch allocation, the string pool) gets its
// own base at a 4 GiB boundary in the order it was registered.

struct Level {
    sets: u32,
    ways: u32,
    tag: Vec<u64>,
    stamp: Vec<u64>,
}
impl Level {
    fn new(sets: u32, ways: u32) -> Self {
        Level { sets, ways, tag: vec![u64::MAX; (sets * ways) as usize], stamp: vec![0; (sets * ways) as usize] }
    }
    fn clear(&mut self) {
        self.tag.iter_mut().for_each(|t| *t = u64::MAX);
        self.stamp.iter_mut().for_each(|s| *s = 0);
    }
    fn access(&mut self, line: u64, now: u64) -> bool {
        let s = ((line & (self.sets as u64 - 1)) * self.ways as u64) as usize;
        let (mut victim, mut oldest) = (0usize, u64::MAX);
        for w in 0..self.ways as usize {
            if self.tag[s + w] == line {
                self.stamp[s + w] = now;
                return true;
            }
            if self.stamp[s + w] < oldest {
                oldest = self.stamp[s + w];
                victim = w;
            }
        }
        self.tag[s + victim] = line;
        self.stamp[s + victim] = now;
        false
    }
}
pub struct CacheModel {
    l1: Level,
    l2: Level,
    now: u64,
    pub l1_misses: u64,
    pub l2_misses: u64,
}
impl CacheModel {
    fn new() -> Self {
        CacheModel { l1: Level::new(64, 8), l2: Level::new(1024, 16), now: 0, l1_misses: 0, l2_misses: 0 }
    }
    fn reset(&mut self) {
        self.l1.clear();
        self.l2.clear();
        self.now = 0;
        self.l1_misses = 0;
        self.l2_misses = 0;
    }
    fn touch(&mut self, va: u64, bytes: usize) {
        let last = (va + bytes as u64 - 1) >> 6;
        let mut line = va >> 6;
        while line <= last {
            self.now += 1;
            if !self.l1.access(line, self.now) {
                self.l1_misses += 1;
                if !self.l2.access(line, self.now) {
                    self.l2_misses += 1;
                }
            }
            line += 1;
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Region {
    lo: usize,
    hi: usize,
    vbase: u64,
    bytes: usize,
    used: bool,
}
const MAX_REGIONS: usize = 256;
pub const FNV_OFFSET: u64 = 1469598103934665603;
pub const FNV_PRIME: u64 = 1099511628211;

/// Deterministic facts about how a run went.
#[derive(Clone, Copy, Default, Debug)]
pub struct AlgoStats {
    pub depth: u32,
    pub max_depth: u32,
    pub first_route: u8,
    pub route: [u64; 6],
    pub giveups: u64,
    pub splits: u64,
    pub split_retries: u64,
    pub part_sorts: u64,
    pub dict_tries: u64,
    pub dict_hits: u64,
    pub plan_retries: u64,
    pub radix_passes: u64,
    pub passes_skipped: u64,
}

pub struct Trace {
    pub active: bool,
    pub model: bool,
    pub reads: u64,
    pub writes: u64,
    pub compares: u64,
    pub table_reads: u64,
    pub table_writes: u64,
    pub table_bytes: u64,
    pub key_bytes: u64,
    pub cmp_flips: u64,
    pub hash: u64,
    pub unmapped: u64,
    have_cmp: bool,
    last_cmp: bool,
    regions: [Region; MAX_REGIONS],
    n_regions: usize,
    next_seq: u64,
    last: usize,
    pub cache: CacheModel,
    pub aux_current: usize,
    pub aux_peak: usize,
    pub aux_allocs: u64,
    pub stats: AlgoStats,
}
impl Trace {
    fn new() -> Self {
        Trace {
            active: false,
            model: false,
            reads: 0,
            writes: 0,
            compares: 0,
            table_reads: 0,
            table_writes: 0,
            table_bytes: 0,
            key_bytes: 0,
            cmp_flips: 0,
            hash: FNV_OFFSET,
            unmapped: 0,
            have_cmp: false,
            last_cmp: false,
            regions: [Region::default(); MAX_REGIONS],
            n_regions: 0,
            next_seq: 0,
            last: 0,
            cache: CacheModel::new(),
            aux_current: 0,
            aux_peak: 0,
            aux_allocs: 0,
            stats: AlgoStats::default(),
        }
    }
    /// Starts recording. `array` is the element array being sorted; every
    /// scratch allocation made from now on becomes a region as well.
    pub fn begin(&mut self, array: *const u8, bytes: usize, with_model: bool) {
        self.reads = 0;
        self.writes = 0;
        self.compares = 0;
        self.table_reads = 0;
        self.table_writes = 0;
        self.table_bytes = 0;
        self.key_bytes = 0;
        self.cmp_flips = 0;
        self.unmapped = 0;
        self.hash = FNV_OFFSET;
        self.have_cmp = false;
        self.last_cmp = false;
        self.regions = [Region::default(); MAX_REGIONS];
        self.n_regions = 0;
        self.next_seq = 0;
        self.last = 0;
        self.model = with_model;
        if with_model {
            self.cache.reset();
        }
        self.aux_current = 0;
        self.aux_peak = 0;
        self.aux_allocs = 0;
        self.stats = AlgoStats::default();
        self.active = true;
        self.add_region(array, bytes);
    }
    pub fn end(&mut self) {
        self.active = false;
        self.model = false;
    }
    pub fn add_region(&mut self, p: *const u8, bytes: usize) {
        let mut i = 0;
        while i < self.n_regions && self.regions[i].used {
            i += 1;
        }
        if i == MAX_REGIONS {
            return;
        }
        if i == self.n_regions {
            self.n_regions += 1;
        }
        self.next_seq += 1;
        let lo = p as usize;
        self.regions[i] = Region { lo, hi: lo + bytes.max(1), vbase: self.next_seq << 32, bytes, used: true };
    }
    pub fn remove_region(&mut self, p: *const u8) -> usize {
        let lo = p as usize;
        for i in 0..self.n_regions {
            if self.regions[i].used && self.regions[i].lo == lo {
                self.regions[i].used = false;
                return self.regions[i].bytes;
            }
        }
        0
    }
    fn locate(&mut self, p: *const u8) -> Option<u64> {
        let a = p as usize;
        let r = self.regions[self.last];
        if r.used && a >= r.lo && a < r.hi {
            return Some(r.vbase + (a - r.lo) as u64);
        }
        for i in 0..self.n_regions {
            let q = self.regions[i];
            if q.used && a >= q.lo && a < q.hi {
                self.last = i;
                return Some(q.vbase + (a - q.lo) as u64);
            }
        }
        None
    }
    #[inline]
    fn mix(&mut self, w: u64) {
        self.hash = (self.hash ^ w).wrapping_mul(FNV_PRIME);
    }
    fn event(&mut self, p: *const u8, bytes: usize, op: u64) {
        match self.locate(p) {
            None => self.unmapped += 1,
            Some(va) => {
                self.mix((va << 3) | op);
                self.cache.touch(va, bytes);
            }
        }
    }
    fn on_read(&mut self, p: *const u8, bytes: usize) {
        self.reads += 1;
        if self.model {
            self.event(p, bytes, 1);
        }
    }
    fn on_write(&mut self, p: *const u8, bytes: usize) {
        self.writes += 1;
        if self.model {
            self.event(p, bytes, 2);
        }
    }
    fn on_table_read(&mut self, p: *const u8, bytes: usize) {
        self.table_reads += 1;
        self.table_bytes += bytes as u64;
        if self.model {
            self.event(p, bytes, 3);
        }
    }
    fn on_table_write(&mut self, p: *const u8, bytes: usize) {
        self.table_writes += 1;
        self.table_bytes += bytes as u64;
        if self.model {
            self.event(p, bytes, 4);
        }
    }
    fn on_table_sweep(&mut self, p: *const u8, entries: usize, entry_bytes: usize, read: bool, write: bool) {
        if read {
            self.table_reads += entries as u64;
            self.table_bytes += (entries * entry_bytes) as u64;
        }
        if write {
            self.table_writes += entries as u64;
            self.table_bytes += (entries * entry_bytes) as u64;
        }
        if self.model && entries > 0 {
            self.event(
                p,
                entries * entry_bytes,
                if read && write {
                    6
                } else if read {
                    3
                } else {
                    4
                },
            );
        }
    }
    fn on_key(&mut self, p: *const u8, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.key_bytes += bytes as u64;
        if self.model {
            self.event(p, bytes, 5);
        }
    }
    fn on_compare(&mut self, r: bool) {
        self.compares += 1;
        if self.have_cmp && r != self.last_cmp {
            self.cmp_flips += 1;
        }
        self.have_cmp = true;
        self.last_cmp = r;
        if self.model {
            self.mix(0x100 | r as u64);
        }
    }
    fn on_alloc(&mut self, p: *const u8, bytes: usize) {
        self.aux_current += bytes;
        if self.aux_current > self.aux_peak {
            self.aux_peak = self.aux_current;
        }
        self.aux_allocs += 1;
        if self.active {
            self.add_region(p, bytes);
        }
    }
    fn on_free(&mut self, p: *const u8, bytes: usize) {
        self.aux_current -= bytes;
        if self.active {
            self.remove_region(p);
        }
    }
}

thread_local! {
    static TRACE: UnsafeCell<Trace> = UnsafeCell::new(Trace::new());
}
/// The trace of the current thread. The hooks are called sequentially from
/// the sort and never re-enter, so the exclusive access is not aliased.
#[allow(clippy::mut_from_ref)]
pub fn with_trace<R>(f: impl FnOnce(&mut Trace) -> R) -> R {
    TRACE.with(|t| f(unsafe { &mut *t.get() }))
}

/// The hooks type the counted view hands to the library.
pub struct TraceHooks;
impl Hooks for TraceHooks {
    const COUNTED: bool = true;
    fn on_read(p: *const u8, b: usize) {
        with_trace(|t| t.on_read(p, b))
    }
    fn on_write(p: *const u8, b: usize) {
        with_trace(|t| t.on_write(p, b))
    }
    fn on_table_read(p: *const u8, b: usize) {
        with_trace(|t| t.on_table_read(p, b))
    }
    fn on_table_write(p: *const u8, b: usize) {
        with_trace(|t| t.on_table_write(p, b))
    }
    fn on_table_sweep(p: *const u8, entries: usize, entry_bytes: usize, read: bool, write: bool) {
        with_trace(|t| t.on_table_sweep(p, entries, entry_bytes, read, write))
    }
    fn on_key(p: *const u8, b: usize) {
        with_trace(|t| t.on_key(p, b))
    }
    fn on_compare(less: bool) {
        with_trace(|t| t.on_compare(less))
    }
    fn stat(s: Stat, n: u64) {
        with_trace(|t| {
            let st = &mut t.stats;
            match s {
                Stat::Giveups => st.giveups += n,
                Stat::Splits => st.splits += n,
                Stat::SplitRetries => st.split_retries += n,
                Stat::PartSorts => st.part_sorts += n,
                Stat::DictTries => st.dict_tries += n,
                Stat::DictHits => st.dict_hits += n,
                Stat::PlanRetries => st.plan_retries += n,
                Stat::RadixPasses => st.radix_passes += n,
                Stat::PassesSkipped => st.passes_skipped += n,
            }
        })
    }
    fn note_route(r: u8) {
        with_trace(|t| {
            t.stats.route[r as usize] += 1;
            if t.stats.first_route == 0 {
                t.stats.first_route = r;
            }
        })
    }
    fn enter_depth() {
        with_trace(|t| {
            t.stats.depth += 1;
            if t.stats.depth > t.stats.max_depth {
                t.stats.max_depth = t.stats.depth;
            }
        })
    }
    fn leave_depth() {
        with_trace(|t| t.stats.depth -= 1)
    }
}

/// The tracked allocator: every scratch block is a region of the trace
/// and counts towards the peak.
pub struct TraceAlloc;
impl Alloc for TraceAlloc {
    fn allocate(bytes: usize) -> Result<std::ptr::NonNull<u8>, AllocError> {
        let layout = std::alloc::Layout::from_size_align(bytes.max(1), 16).map_err(|_| AllocError)?;
        let p = std::ptr::NonNull::new(unsafe { std::alloc::alloc(layout) }).ok_or(AllocError)?;
        with_trace(|t| t.on_alloc(p.as_ptr(), bytes));
        Ok(p)
    }
    unsafe fn deallocate(p: std::ptr::NonNull<u8>, bytes: usize) {
        with_trace(|t| t.on_free(p.as_ptr(), bytes));
        unsafe { std::alloc::dealloc(p.as_ptr(), std::alloc::Layout::from_size_align_unchecked(bytes.max(1), 16)) }
    }
}

// ---- the counted run ------------------------------------------------------------

/// The deterministic columns of one run, in the order of `results/counts.csv`.
#[derive(Clone, Debug, Default)]
pub struct DetRecord {
    pub ok: bool,
    pub error: String,
    pub values: Vec<(&'static str, String)>,
    /// The scratch memory as the global allocator saw it: the same numbers
    /// as `aux_peak_bytes` and `aux_allocs`, taken the way every other Rust
    /// sort's are in `counts.rs`.
    pub alloc_peak: usize,
    pub alloc_allocs: u64,
    pub alloc_sizes: [usize; SIZES_KEPT],
}
pub const DET_COLUMNS: [&str; 34] = [
    "reads",
    "writes",
    "compares",
    "table_reads",
    "table_writes",
    "table_bytes",
    "key_bytes",
    "cmp_flips",
    "traffic_bytes",
    "l1_misses",
    "l2_misses",
    "unmapped",
    "aux_peak_bytes",
    "aux_allocs",
    "max_depth",
    "fallbacks",
    "bad_parts",
    "route",
    "r_sorted",
    "r_reverse",
    "r_runs",
    "r_displaced",
    "r_radix",
    "giveups",
    "splits",
    "split_retries",
    "part_sorts",
    "dict_tries",
    "dict_hits",
    "plan_retries",
    "radix_passes",
    "passes_skipped",
    "order_hash",
    "trace_hash",
];
pub fn route_name(r: u8) -> &'static str {
    ["-", "sorted", "reverse", "runs", "displaced", "radix"][r.min(5) as usize]
}

/// A stable sort by key of the input: the reference output.
pub fn make_reference<T: Item2>(input: &[T]) -> Vec<T> {
    let mut r = input.to_vec();
    r.sort_by(|a, b| {
        if T::less(*a, *b) {
            std::cmp::Ordering::Less
        } else if T::less(*b, *a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    r
}
pub fn verify_sorted<T: Item2>(out: &[T], reference: &[T]) -> Result<(), String> {
    if out.len() != reference.len() {
        return Err("size changed".into());
    }
    for i in 0..out.len() {
        if out[i] != reference[i] {
            return Err(format!("mismatch at index {i} (id {} vs expected id {}) - not sorted or not stable", out[i].id(), reference[i].id()));
        }
    }
    Ok(())
}

/// Runs the counted variant once on a copy of the input with the cache
/// model on, and reports every deterministic column.
pub fn counted_run<T: Item2>(items: &[T], pool: Option<&[u8]>, reference: &[T]) -> (DetRecord, Vec<T>) {
    let n = items.len();
    let mut work = items.to_vec(); // allocated before the trace starts, so it is not scratch
    with_trace(|t| {
        t.begin(work.as_ptr() as *const u8, std::mem::size_of_val(items), true);
        if let Some(p) = pool {
            t.add_region(p.as_ptr(), p.len());
        }
    });
    let view = View::<T, TraceHooks, TraceAlloc>::new(work.as_mut_ptr(), n);
    let mut scratch = Scratch::new();
    alloc_arm();
    let result = brainsort_impl(view, &mut scratch, false, 0);
    drop(scratch);
    let galloc = alloc_disarm();
    with_trace(|t| t.end());
    let mut r = DetRecord { alloc_peak: galloc.peak, alloc_allocs: galloc.allocs, alloc_sizes: galloc.sizes, ..DetRecord::default() };
    if result.is_err() {
        r.error = "allocation failed".into();
        return (r, work);
    }
    let mut h = FNV_OFFSET;
    for w in &work {
        h = (h ^ w.id() as u64).wrapping_mul(FNV_PRIME);
    }
    let (values, leak) = with_trace(|t| {
        let esz = std::mem::size_of::<T>() as u64;
        let traffic = (t.reads + t.writes) * esz + t.table_bytes + t.key_bytes;
        let st = t.stats;
        let v: Vec<(&'static str, String)> = vec![
            ("reads", t.reads.to_string()),
            ("writes", t.writes.to_string()),
            ("compares", t.compares.to_string()),
            ("table_reads", t.table_reads.to_string()),
            ("table_writes", t.table_writes.to_string()),
            ("table_bytes", t.table_bytes.to_string()),
            ("key_bytes", t.key_bytes.to_string()),
            ("cmp_flips", t.cmp_flips.to_string()),
            ("traffic_bytes", traffic.to_string()),
            ("l1_misses", t.cache.l1_misses.to_string()),
            ("l2_misses", t.cache.l2_misses.to_string()),
            ("unmapped", t.unmapped.to_string()),
            ("aux_peak_bytes", t.aux_peak.to_string()),
            ("aux_allocs", t.aux_allocs.to_string()),
            ("max_depth", st.max_depth.to_string()),
            ("fallbacks", "0".into()),
            ("bad_parts", "0".into()),
            ("route", route_name(st.first_route).into()),
            ("r_sorted", st.route[1].to_string()),
            ("r_reverse", st.route[2].to_string()),
            ("r_runs", st.route[3].to_string()),
            ("r_displaced", st.route[4].to_string()),
            ("r_radix", st.route[5].to_string()),
            ("giveups", st.giveups.to_string()),
            ("splits", st.splits.to_string()),
            ("split_retries", st.split_retries.to_string()),
            ("part_sorts", st.part_sorts.to_string()),
            ("dict_tries", st.dict_tries.to_string()),
            ("dict_hits", st.dict_hits.to_string()),
            ("plan_retries", st.plan_retries.to_string()),
            ("radix_passes", st.radix_passes.to_string()),
            ("passes_skipped", st.passes_skipped.to_string()),
            ("order_hash", format!("{h:016x}")),
            ("trace_hash", format!("{:016x}", t.hash)),
        ];
        (v, t.aux_current)
    });
    r.values = values;
    if leak != 0 {
        r.error = "aux_memory_leak".into();
        return (r, work);
    }
    if let Err(e) = verify_sorted(&work, reference) {
        r.error = format!("counted_pass_failed:{e}");
        return (r, work);
    }
    r.ok = true;
    (r, work)
}

/// One golden row of `results/counts.csv`.
#[derive(Clone, Debug)]
pub struct GoldenRow {
    pub ty: String,
    pub dataset: String,
    pub n: usize,
    pub seed: u64,
    pub values: Vec<(String, String)>,
}
/// The brainsort rows of the golden file.
pub fn load_golden(path: &std::path::Path) -> Result<Vec<GoldenRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("empty golden file")?.split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name);
    let (ct, cd, ca, cn, cs, cok) = (col("type").unwrap(), col("dataset").unwrap(), col("algorithm").unwrap(), col("n").unwrap(), col("seed").unwrap(), col("ok").unwrap());
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f[ca] != "brainsort" || f[cok] != "1" {
            continue;
        }
        let values = DET_COLUMNS.iter().filter_map(|c| col(c).map(|i| (c.to_string(), f[i].to_string()))).collect();
        rows.push(GoldenRow { ty: f[ct].into(), dataset: f[cd].into(), n: f[cn].parse().map_err(|_| "bad n")?, seed: f[cs].parse().map_err(|_| "bad seed")?, values });
    }
    Ok(rows)
}

/// Recomputes one golden row; returns the differing columns (empty = match).
pub fn check_row(row: &GoldenRow) -> Result<Vec<String>, String> {
    fn run<T: Item2 + datasets::KeyGen>(row: &GoldenRow) -> Result<Vec<String>, String>
    where
        datasets::Dataset<T>: datasets::Generate<T>,
    {
        let data = datasets::generate::<T>(&row.dataset, row.n, row.seed);
        let reference = make_reference(&data.items);
        let (det, _) = counted_run(&data.items, data.pool.as_deref(), &reference);
        if !det.ok {
            return Err(det.error);
        }
        let mut diff = Vec::new();
        for (name, want) in &row.values {
            if let Some((_, got)) = det.values.iter().find(|(c, _)| c == name)
                && got != want
            {
                diff.push(format!("{name} {want} -> {got}"));
            }
        }
        Ok(diff)
    }
    match row.ty.as_str() {
        "int32" => run::<Item>(row),
        "double" => run::<DblItem>(row),
        "int64" => run::<I64Item>(row),
        "string" => run::<StrItem>(row),
        t => Err(format!("unknown type {t}")),
    }
}
