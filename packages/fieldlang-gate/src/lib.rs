//! ============================================================
//! GATE TOOL — NOT THE COMPILER.
//! Purity decree (主令改 2026-08-01): the fieldlang compiler is ARM64 asm
//! (field/lang-asm: lexer.s/emit.s/driver.s per CONTRACT v1). This crate is
//! the sanctioned ACCEPTANCE ORACLE: a reference .fldj generator whose byte
//! format comes from the FROZEN Rust journal writer (field::journal, W3,
//! tested by packages/field/tests/store_journal.rs), plus replay/digest
//! helpers. It exists to byte-audit the asm compiler's output. It is never
//! shipped as a compiler, never linked into one.
//! ============================================================
//!
//! Reference parser for CONTRACT v1 grammar (docs/design/2026-08-01-field-lang.md):
//! line-based, 7 glyphs (界種撃歩縛束寫), decimal u64 args only, `#` comments.
//! Errors carry line/col (1-based CHARACTER columns).

use field::journal::Op;
use field::plane::WaveParams;
use field::FieldConfig;

/// CONTRACT §Defaults — driver-owned law. DISTINCT from field crate defaults.
pub const DEFAULT_W: u32 = 64;
pub const DEFAULT_H: u32 = 64;
pub const DEFAULT_N_SLOTS: u32 = 16;
pub const DEFAULT_SEED: u64 = 42;
pub const DEFAULT_C_BITS: u32 = 0x3F80_0000; // 1.0
pub const DEFAULT_DT_BITS: u32 = 0x3DCC_CCCD; // 0.1
pub const DEFAULT_DAMPING_BITS: u32 = 0x3F7F_BE77; // 0.999
pub const DEFAULT_DX_BITS: u32 = 0x3F80_0000; // 1.0
pub const DEFAULT_RANGE_BITS: u32 = 0x3F80_0000; // 1.0

/// A parsed .fld program: journal header fields + op sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct RefProgram {
    pub cfg: FieldConfig,
    pub params: WaveParams,
    pub n_slots: usize,
    pub ops: Vec<Op>,
}

/// Parse error with 1-based line/col (char columns).
#[derive(Clone, Debug, PartialEq)]
pub struct RefError {
    pub line: usize,
    pub col: usize,
    pub msg: String,
}

impl std::fmt::Display for RefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {} col {}: {}", self.line, self.col, self.msg)
    }
}
impl std::error::Error for RefError {}

fn err<T>(line: usize, col: usize, msg: impl Into<String>) -> Result<T, RefError> {
    Err(RefError { line, col, msg: msg.into() })
}

fn default_params(seed: u64) -> WaveParams {
    WaveParams {
        c: f32::from_bits(DEFAULT_C_BITS),
        dt: f32::from_bits(DEFAULT_DT_BITS),
        damping: f32::from_bits(DEFAULT_DAMPING_BITS),
        dx: f32::from_bits(DEFAULT_DX_BITS),
        seed,
        range: f32::from_bits(DEFAULT_RANGE_BITS),
    }
}

/// One whitespace-separated token with its 1-based char column.
struct Tok<'a> {
    text: &'a str,
    col: usize,
}

fn tokenize(line: &str) -> Vec<Tok<'_>> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None; // byte index
    for (i, ch) in line.char_indices() {
        if ch == ' ' || ch == '\t' {
            if let Some(s) = start.take() {
                out.push(Tok { text: &line[s..i], col: line[..s].chars().count() + 1 });
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push(Tok { text: &line[s..], col: line[..s].chars().count() + 1 });
    }
    out
}

fn parse_u64(t: &Tok<'_>, lno: usize) -> Result<u64, RefError> {
    if t.text.is_empty() || !t.text.bytes().all(|b| b.is_ascii_digit()) {
        return err(lno, t.col, format!("expected decimal u64, got `{}`", t.text));
    }
    t.text
        .parse::<u64>()
        .map_err(|_| RefError { line: lno, col: t.col, msg: format!("u64 overflow: `{}`", t.text) })
}

fn parse_u32(t: &Tok<'_>, lno: usize) -> Result<u32, RefError> {
    let v = parse_u64(t, lno)?;
    u32::try_from(v)
        .map_err(|_| RefError { line: lno, col: t.col, msg: format!("arg > u32 max: `{}`", t.text) })
}

fn arity(toks: &[Tok<'_>], want: usize, lno: usize, glyph: &str) -> Result<(), RefError> {
    if toks.len() != want {
        let col = toks.last().map_or(1, |t| t.col + t.text.chars().count());
        return err(lno, col, format!("{} wants {} args, got {}", glyph, want - 1, toks.len() - 1));
    }
    Ok(())
}

/// Parse CONTRACT v1 source text → RefProgram. Total: every line classified,
/// first error wins, no panics on user input.
pub fn parse(src: &str) -> Result<RefProgram, RefError> {
    let mut w: u32 = DEFAULT_W;
    let mut h: u32 = DEFAULT_H;
    let mut n_slots: u32 = DEFAULT_N_SLOTS;
    let mut seed: u64 = DEFAULT_SEED;
    let mut seen_header = false;
    let mut ops: Vec<Op> = Vec::new();

    for (lno, raw) in src.lines().enumerate() {
        let lno = lno + 1;
        let line = raw.split('#').next().unwrap_or(""); // `#` comment → EOL
        let toks = tokenize(line);
        if toks.is_empty() {
            continue;
        }
        let glyph = &toks[0];
        let g = glyph.text;
        let args = &toks[1..];

        match g {
            "界" => {
                if seen_header {
                    return err(lno, glyph.col, "界 twice (header is once, first)");
                }
                if !ops.is_empty() {
                    return err(lno, glyph.col, "界 must precede all ops");
                }
                arity(&toks, 5, lno, "界")?;
                w = parse_u32(&args[0], lno)?;
                h = parse_u32(&args[1], lno)?;
                n_slots = parse_u32(&args[2], lno)?;
                seed = parse_u64(&args[3], lno)?;
                seen_header = true;
            }
            "種" => {
                arity(&toks, 3, lno, "種")?;
                ops.push(Op::SeedAtom { slot: parse_u32(&args[0], lno)?, seed: parse_u64(&args[1], lno)? });
            }
            "撃" => {
                arity(&toks, 4, lno, "撃")?;
                ops.push(Op::Excite {
                    x: parse_u32(&args[0], lno)?,
                    y: parse_u32(&args[1], lno)?,
                    amp_bits: parse_u32(&args[2], lno)?,
                });
            }
            "歩" => {
                arity(&toks, 2, lno, "歩")?;
                ops.push(Op::Step { count: parse_u32(&args[0], lno)? });
            }
            "縛" => {
                arity(&toks, 4, lno, "縛")?;
                ops.push(Op::Bind {
                    dst: parse_u32(&args[0], lno)?,
                    a: parse_u32(&args[1], lno)?,
                    b: parse_u32(&args[2], lno)?,
                });
            }
            "束" | "寫" => {
                if toks.len() < 3 {
                    return err(lno, glyph.col, format!("{} wants dst/slot, n, then n values", g));
                }
                let first = parse_u32(&args[0], lno)?;
                let n = parse_u32(&args[1], lno)? as usize;
                if args.len() != 2 + n {
                    let col = toks.last().map_or(1, |t| t.col + t.text.chars().count());
                    return err(lno, col, format!("{} declares n={} but has {} values", g, n, args.len() - 2));
                }
                let mut vals = Vec::with_capacity(n);
                for t in &args[2..] {
                    vals.push(parse_u32(t, lno)?);
                }
                if g == "束" {
                    ops.push(Op::Bundle { dst: first, srcs: vals });
                } else {
                    ops.push(Op::WriteRaw { slot: first, data_bits: vals });
                }
            }
            other => {
                return err(lno, glyph.col, format!("bad char / unknown glyph `{}`", other));
            }
        }
    }

    // Gate-tool hygiene (NOT compiler law): keep replay panic-free.
    let cfg = FieldConfig { width: w as usize, height: h as usize };
    if !cfg.valid() {
        return err(1, 1, format!("界 dims {}x{} must be powers of two >= 2", w, h));
    }
    if n_slots < 2 {
        return err(1, 1, format!("界 n_slots {} < 2 (RESERVED_SLOTS)", n_slots));
    }

    Ok(RefProgram {
        cfg,
        params: default_params(seed),
        n_slots: n_slots as usize,
        ops,
    })
}
