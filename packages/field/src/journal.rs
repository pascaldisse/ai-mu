//! SSD COLD JOURNAL. OWNED BY W3. SIGNATURES FROZEN.
//! Adversary law: SSD = cold backing ONLY. Append-only op log; replay =
//! bit-exact state reconstruction (ENTROPY.md: seed + journal = the world).
//! NO write/flush inside the frame hot path: append() buffers in RAM,
//! flush() is explicit, called off-cadence.
//! Binary format v1 (LE): magic "FLDJ" u32, version u32=1, w u32, h u32,
//! params (c,dt,damping,dx as f32-bits u32 · seed u64 · range f32-bits u32),
//! n_slots u32, then ops: tag u8 + payload. All f32 AS BIT PATTERNS (u32)
//! -> bit-exact roundtrip. Truncated trailing op on read = io error.

use crate::plane::WaveParams;
use crate::FieldConfig;
use std::io;
use std::io::{Cursor, Read, Write};
use std::path::Path;

/// The op alphabet. Everything that mutates the world goes through this.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// store[slot] = atoms::seeded_atom(seed)          tag 1
    SeedAtom { slot: u32, seed: u64 },
    /// plane::excite(slot0, slot1, x, y, amp)          tag 2
    Excite { x: u32, y: u32, amp_bits: u32 },
    /// count wave ticks                                 tag 3
    Step { count: u32 },
    /// store[dst] = bind(store[a], store[b])           tag 4
    Bind { dst: u32, a: u32, b: u32 },
    /// store[dst] = bundle(store[srcs...])             tag 5
    Bundle { dst: u32, srcs: Vec<u32> },
    /// store[slot] = raw f32-bits row                  tag 6
    WriteRaw { slot: u32, data_bits: Vec<u32> },
}

/// Magic "FLDJ" as LE u32 (bytes F L D J).
const MAGIC: u32 = 0x4A44_4C46;
const VERSION: u32 = 1;
/// Header = magic(4) + version(4) + w(4) + h(4) + params(28) + n_slots(4).
const HEADER_LEN: usize = 48;

fn header_bytes(cfg: FieldConfig, params: &WaveParams, n_slots: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(HEADER_LEN);
    b.extend_from_slice(&MAGIC.to_le_bytes());
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&(cfg.width as u32).to_le_bytes());
    b.extend_from_slice(&(cfg.height as u32).to_le_bytes());
    b.extend_from_slice(&params.c.to_bits().to_le_bytes());
    b.extend_from_slice(&params.dt.to_bits().to_le_bytes());
    b.extend_from_slice(&params.damping.to_bits().to_le_bytes());
    b.extend_from_slice(&params.dx.to_bits().to_le_bytes());
    b.extend_from_slice(&params.seed.to_le_bytes());
    b.extend_from_slice(&params.range.to_bits().to_le_bytes());
    b.extend_from_slice(&(n_slots as u32).to_le_bytes());
    b
}

/// tag u8 + payload LE. Vec payloads prefixed by u32 len.
fn op_bytes(op: &Op) -> Vec<u8> {
    let mut buf = Vec::new();
    match op {
        Op::SeedAtom { slot, seed } => {
            buf.push(1);
            buf.extend_from_slice(&slot.to_le_bytes());
            buf.extend_from_slice(&seed.to_le_bytes());
        }
        Op::Excite { x, y, amp_bits } => {
            buf.push(2);
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&amp_bits.to_le_bytes());
        }
        Op::Step { count } => {
            buf.push(3);
            buf.extend_from_slice(&count.to_le_bytes());
        }
        Op::Bind { dst, a, b } => {
            buf.push(4);
            buf.extend_from_slice(&dst.to_le_bytes());
            buf.extend_from_slice(&a.to_le_bytes());
            buf.extend_from_slice(&b.to_le_bytes());
        }
        Op::Bundle { dst, srcs } => {
            buf.push(5);
            buf.extend_from_slice(&dst.to_le_bytes());
            buf.extend_from_slice(&(srcs.len() as u32).to_le_bytes());
            for s in srcs {
                buf.extend_from_slice(&s.to_le_bytes());
            }
        }
        Op::WriteRaw { slot, data_bits } => {
            buf.push(6);
            buf.extend_from_slice(&slot.to_le_bytes());
            buf.extend_from_slice(&(data_bits.len() as u32).to_le_bytes());
            for v in data_bits {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    buf
}

fn read_exact(c: &mut Cursor<&[u8]>, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    c.read_exact(&mut buf)?; // Cursor read_exact -> UnexpectedEof on truncation
    Ok(buf)
}

fn rd_u32(c: &mut Cursor<&[u8]>) -> io::Result<u32> {
    let b = read_exact(c, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn rd_u64(c: &mut Cursor<&[u8]>) -> io::Result<u64> {
    let b = read_exact(c, 8)?;
    let mut a = [0u8; 8];
    a.copy_from_slice(&b);
    Ok(u64::from_le_bytes(a))
}

fn rd_u8(c: &mut Cursor<&[u8]>) -> io::Result<u8> {
    let b = read_exact(c, 1)?;
    Ok(b[0])
}

/// Reject vec-lengths whose payload cannot fit in the remaining bytes.
fn check_len(remaining: usize, len: usize) -> io::Result<()> {
    if len.checked_mul(4).map_or(true, |n| n > remaining) {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "journal: truncated vec payload",
        ));
    }
    Ok(())
}

pub struct Journal {
    buf: Vec<u8>,
    path: std::path::PathBuf,
}

impl Journal {
    /// Create file, write header immediately, buffer ops thereafter.
    pub fn create(
        path: &Path,
        cfg: FieldConfig,
        params: &WaveParams,
        n_slots: usize,
    ) -> io::Result<Self> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&header_bytes(cfg, params, n_slots))?;
        Ok(Journal { buf: Vec::new(), path: path.to_path_buf() })
    }

    /// Buffer one op in RAM. NEVER touches disk (hot-path safe).
    pub fn append(&mut self, op: &Op) {
        self.buf.extend_from_slice(&op_bytes(op));
    }

    /// Flush buffered ops to disk. Off-cadence only.
    pub fn flush(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let mut file = std::fs::OpenOptions::new().append(true).open(&self.path)?;
        file.write_all(&self.buf)?;
        self.buf.clear();
        Ok(())
    }

    /// Read a journal back: (cfg, params, n_slots, ops). Bit-exact.
    pub fn read_all(path: &Path) -> io::Result<(FieldConfig, WaveParams, usize, Vec<Op>)> {
        let bytes = std::fs::read(path)?;
        let mut c = Cursor::new(&bytes[..]);

        let magic = rd_u32(&mut c)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "journal: bad magic"));
        }
        let version = rd_u32(&mut c)?;
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "journal: bad version"));
        }
        let w = rd_u32(&mut c)? as usize;
        let h = rd_u32(&mut c)? as usize;
        let params = WaveParams {
            c: f32::from_bits(rd_u32(&mut c)?),
            dt: f32::from_bits(rd_u32(&mut c)?),
            damping: f32::from_bits(rd_u32(&mut c)?),
            dx: f32::from_bits(rd_u32(&mut c)?),
            seed: rd_u64(&mut c)?,
            range: f32::from_bits(rd_u32(&mut c)?),
        };
        let n_slots = rd_u32(&mut c)? as usize;
        // native fieldrun と同一の wire 契約: n_slots < 2 = malformed(slot0/1 は予約)。
        // panic に非ず回復可能 Err で返す(入力不備 = bug に非ず)。
        if n_slots < crate::store::RESERVED_SLOTS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("journal: n_slots {} < {}", n_slots, crate::store::RESERVED_SLOTS),
            ));
        }
        let cfg = FieldConfig { width: w, height: h };
        // d = w*h と n_slots*d の checked 積(usize 溢 = 拒絶、割当前)。
        let d = w.checked_mul(h).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "journal: w*h overflow")
        })?;
        n_slots.checked_mul(d).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "journal: n_slots*d overflow")
        })?;

        let mut ops = Vec::new();
        while c.position() < bytes.len() as u64 {
            let tag = rd_u8(&mut c)?;
            let remaining = bytes.len() - c.position() as usize;
            let op = match tag {
                1 => Op::SeedAtom { slot: rd_u32(&mut c)?, seed: rd_u64(&mut c)? },
                2 => Op::Excite {
                    x: rd_u32(&mut c)?,
                    y: rd_u32(&mut c)?,
                    amp_bits: rd_u32(&mut c)?,
                },
                3 => Op::Step { count: rd_u32(&mut c)? },
                4 => Op::Bind { dst: rd_u32(&mut c)?, a: rd_u32(&mut c)?, b: rd_u32(&mut c)? },
                5 => {
                    let dst = rd_u32(&mut c)?;
                    let len = rd_u32(&mut c)? as usize;
                    check_len(remaining - 8, len)?;
                    let mut srcs = Vec::with_capacity(len);
                    for _ in 0..len {
                        srcs.push(rd_u32(&mut c)?);
                    }
                    Op::Bundle { dst, srcs }
                }
                6 => {
                    let slot = rd_u32(&mut c)?;
                    let len = rd_u32(&mut c)? as usize;
                    check_len(remaining - 8, len)?;
                    let mut data_bits = Vec::with_capacity(len);
                    for _ in 0..len {
                        data_bits.push(rd_u32(&mut c)?);
                    }
                    Op::WriteRaw { slot, data_bits }
                }
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("journal: bad op tag {}", other),
                    ))
                }
            };
            ops.push(op);
        }
        Ok((cfg, params, n_slots, ops))
    }
}
