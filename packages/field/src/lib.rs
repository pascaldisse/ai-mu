//! # field — the FIELD SUBSTRATE, V0
//!
//! § FIELD.md: "everything is one field; this atom = the vector space that
//! is also a database." V0 is that atom: a vector-symbolic (VSA/HRR)
//! hyperdimensional store — deterministic hypervectors (§ ENTROPY.md, no
//! randomness), a bind/bundle/similarity algebra over them, and an
//! mmap-backed slab so the store IS the field (no separate load step, no
//! serde round-trip — § "SSD-as-RAM doctrine").
//!
//! - [`DIM`], [`random_vec`] — deterministic `d`-wide bipolar hypervectors,
//!   `hash(seed, index)` per lane.
//! - [`bind`] / [`unbind`] — element-wise bipolar product (MAP-B), exact
//!   self-inverse; see the doc comment on [`bind`] for why this was chosen
//!   over circular convolution.
//! - [`bundle`] — sum + L2-renormalize (superposition).
//! - [`similarity`] — cosine similarity.
//! - [`permute`] — cyclic shift, a role/position compose op.
//! - [`Field`] — `Field::open(path)`, `.store(name, composed)`,
//!   `.probe(query)` (ranked matches), `.cleanup(query)` (nearest stored
//!   vector).

mod hv;
mod store;

pub use hv::{bind, bundle, permute, random_vec, similarity, unbind, DIM};
pub use store::{Field, Match};
