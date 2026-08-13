//! The mmap-backed slab: "the vector space that is also a database"
//! (§ FIELD.md). No serde — a fixed-size binary header plus fixed-size
//! append-only records, cast to/from bytes with `bytemuck` (zero-copy).
//!
//! On-disk layout:
//! ```text
//! [ header: HEADER_BYTES ] [ record 0 ] [ record 1 ] ... [ record N-1 ]
//! ```
//! Header (little-endian, `HEADER_BYTES` = 64, rest reserved/zeroed):
//! - `[0..8)`  magic: `b"FIELDv0\0"`
//! - `[8..12)` dim: u32
//! - `[12..20)` capacity: u64 (record slots currently allocated in the file)
//! - `[20..28)` len: u64 (records actually stored)
//!
//! Record (`record_bytes(dim)` = `NAME_BYTES + dim * 4`):
//! - `[0..NAME_BYTES)` name, UTF-8, zero-padded/truncated
//! - `[NAME_BYTES..)` the `dim` f32 lanes, raw LE bytes
//!
//! Growth: capacity doubles when full (SSD-as-RAM doctrine — the file *is*
//! the store; grow it, don't spill to a second in-memory index).

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use memmap2::MmapMut;

use crate::hv::{as_bytes, similarity};

const MAGIC: [u8; 8] = *b"FIELDv0\0";
const HEADER_BYTES: usize = 64;
const NAME_BYTES: usize = 64;
const INITIAL_CAPACITY: u64 = 64;

fn record_bytes(dim: usize) -> usize {
    NAME_BYTES + dim * 4
}

/// The field substrate: an mmap-backed, append-only vector store with
/// nearest-vector ("cleanup memory") probing.
pub struct Field {
    file: File,
    mmap: MmapMut,
    dim: usize,
    capacity: u64,
    len: u64,
}

/// One ranked match returned by [`Field::probe`].
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub name: String,
    pub similarity: f32,
}

impl Field {
    /// Open (or create) the field at `path`.
    ///
    /// A fresh file is initialized with [`crate::hv::DIM`]-wide records and
    /// `INITIAL_CAPACITY` slots. An existing file is validated against the
    /// magic and re-mapped as-is (its stored `dim` wins, not `hv::DIM`, so a
    /// field created at one dimensionality never silently reinterprets its
    /// bytes at another).
    pub fn open(path: impl AsRef<Path>) -> io::Result<Field> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let file_len = file.metadata()?.len();

        if file_len == 0 {
            let dim = crate::hv::DIM;
            let total = HEADER_BYTES as u64 + INITIAL_CAPACITY * record_bytes(dim) as u64;
            file.set_len(total)?;
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            mmap[0..8].copy_from_slice(&MAGIC);
            mmap[8..12].copy_from_slice(&(dim as u32).to_le_bytes());
            mmap[12..20].copy_from_slice(&INITIAL_CAPACITY.to_le_bytes());
            mmap[20..28].copy_from_slice(&0u64.to_le_bytes());
            mmap.flush()?;
            Ok(Field {
                file,
                mmap,
                dim,
                capacity: INITIAL_CAPACITY,
                len: 0,
            })
        } else {
            let mmap = unsafe { MmapMut::map_mut(&file)? };
            if mmap[0..8] != MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "field: bad magic — not a FIELDv0 slab",
                ));
            }
            let dim = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
            let capacity = u64::from_le_bytes(mmap[12..20].try_into().unwrap());
            let len = u64::from_le_bytes(mmap[20..28].try_into().unwrap());
            Ok(Field {
                file,
                mmap,
                dim,
                capacity,
                len,
            })
        }
    }

    /// This field's hypervector width (fixed at creation).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of records stored so far.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Store a named, already-composed vector (the caller has already
    /// `bind`/`bundle`d it into whatever it represents). Returns the
    /// record's index (a stable id: the append order).
    ///
    /// # Errors
    /// If `composed.len() != self.dim()`.
    pub fn store(&mut self, name: &str, composed: Vec<f32>) -> io::Result<u64> {
        if composed.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "field: vector has {} lanes, field dim is {}",
                    composed.len(),
                    self.dim
                ),
            ));
        }
        if self.len == self.capacity {
            self.grow()?;
        }
        let idx = self.len;
        let rb = record_bytes(self.dim);
        let offset = HEADER_BYTES + (idx as usize) * rb;

        let name_bytes = name.as_bytes();
        let n = name_bytes.len().min(NAME_BYTES);
        self.mmap[offset..offset + NAME_BYTES].fill(0);
        self.mmap[offset..offset + n].copy_from_slice(&name_bytes[..n]);

        let vec_off = offset + NAME_BYTES;
        self.mmap[vec_off..vec_off + self.dim * 4].copy_from_slice(as_bytes(&composed));

        self.len += 1;
        self.mmap[20..28].copy_from_slice(&self.len.to_le_bytes());
        Ok(idx)
    }

    fn grow(&mut self) -> io::Result<()> {
        let new_capacity = (self.capacity * 2).max(INITIAL_CAPACITY);
        let rb = record_bytes(self.dim);
        let new_total = HEADER_BYTES as u64 + new_capacity * rb as u64;

        // mmap can't be resized in place: flush, drop the map, grow the
        // file, remap.
        self.mmap.flush()?;
        self.file.set_len(new_total)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        self.capacity = new_capacity;
        self.mmap[12..20].copy_from_slice(&self.capacity.to_le_bytes());
        Ok(())
    }

    fn record_name_and_vec(&self, idx: u64) -> (String, &[f32]) {
        let (name_bytes, vec) = self.record_raw(idx);
        let end = name_bytes.iter().position(|&b| b == 0).unwrap_or(NAME_BYTES);
        let name = String::from_utf8_lossy(&name_bytes[..end]).into_owned();
        (name, vec)
    }

    /// Raw record access, no `String` allocation — the hot path for
    /// `probe`/`cleanup`'s O(len) scan, which must not pay a heap alloc per
    /// candidate just to compare vectors (only the *winning* name needs to
    /// be materialized).
    fn record_raw(&self, idx: u64) -> (&[u8], &[f32]) {
        let rb = record_bytes(self.dim);
        let offset = HEADER_BYTES + (idx as usize) * rb;
        let name_bytes = &self.mmap[offset..offset + NAME_BYTES];
        let vec_off = offset + NAME_BYTES;
        let bytes = &self.mmap[vec_off..vec_off + self.dim * 4];
        let vec: &[f32] = bytemuck::cast_slice(bytes);
        (name_bytes, vec)
    }

    fn name_at(&self, idx: u64) -> String {
        let (name_bytes, _) = self.record_raw(idx);
        let end = name_bytes.iter().position(|&b| b == 0).unwrap_or(NAME_BYTES);
        String::from_utf8_lossy(&name_bytes[..end]).into_owned()
    }

    /// Fetch the stored `(name, vector)` for record `id` by index (the id
    /// [`Field::store`] returned), or `None` if out of range.
    pub fn get(&self, id: u64) -> Option<(String, Vec<f32>)> {
        if id >= self.len {
            return None;
        }
        let (name, vec) = self.record_name_and_vec(id);
        Some((name, vec.to_vec()))
    }

    /// Rank every stored record against `query` by cosine similarity,
    /// descending. O(len * dim) — a linear scan of the slab (V0; an index is
    /// future work, not required for the substrate to be correct).
    ///
    /// Names are only materialized for the final ranked list (`len` allocs,
    /// one per record, since every record's name is returned) — the
    /// similarity scan itself touches only raw vector bytes.
    ///
    /// # Panics
    /// If `query.len() != self.dim()`.
    pub fn probe(&self, query: &[f32]) -> Vec<Match> {
        assert_eq!(query.len(), self.dim, "probe: dimension mismatch");
        let mut out: Vec<Match> = (0..self.len)
            .map(|i| {
                let (_, vec) = self.record_raw(i);
                Match {
                    name: self.name_at(i),
                    similarity: similarity(query, vec),
                }
            })
            .collect();
        out.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        out
    }

    /// Cleanup memory: the single nearest stored vector to `query`, if any
    /// records are stored. Only the winning record's name is ever
    /// allocated — the O(len) scan itself is name-alloc-free.
    pub fn cleanup(&self, query: &[f32]) -> Option<Match> {
        assert_eq!(query.len(), self.dim, "cleanup: dimension mismatch");
        let best = (0..self.len)
            .map(|i| {
                let (_, vec) = self.record_raw(i);
                (i, similarity(query, vec))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
        Some(Match {
            name: self.name_at(best.0),
            similarity: best.1,
        })
    }
}
