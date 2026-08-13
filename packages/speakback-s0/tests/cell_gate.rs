//! s1 gates — cell-energy speak-back on REAL spatial form (Gaussian blobs
//! mirroring speak-in), not dense atoms.

use speakback_s0::cell_decode::*;

const W: usize = 128;
const H: usize = 128;
const LEX: &[&str] = &["fire", "water", "wind", "storm", "stone", "life", "light", "void"];

fn plane() -> Vec<f32> {
    vec![0.0; W * H]
}

#[test]
fn empty_plane_decodes_nothing() {
    let p = plane();
    assert_eq!(cell_decode_top1(&p, W, H, LEX, 0.1), None);
}

#[test]
fn single_word_blob_decodes_correctly() {
    for word in LEX {
        let mut p = plane();
        synth_speak(&mut p, W, H, word, 1.0);
        assert_eq!(
            cell_decode_top1(&p, W, H, LEX, 0.1).as_deref(),
            Some(*word),
            "failed for {word}"
        );
    }
}

#[test]
fn stronger_word_wins() {
    let mut p = plane();
    synth_speak(&mut p, W, H, "fire", 1.0);
    synth_speak(&mut p, W, H, "water", 0.4);
    assert_eq!(cell_decode_top1(&p, W, H, LEX, 0.1).as_deref(), Some("fire"));
}

#[test]
fn two_words_both_in_top2() {
    let mut p = plane();
    synth_speak(&mut p, W, H, "fire", 1.0);
    synth_speak(&mut p, W, H, "water", 0.9);
    let top: Vec<String> = cell_speak_back(&p, W, H, LEX, 0.0, 2)
        .into_iter()
        .map(|(w, _)| w)
        .collect();
    assert!(top.contains(&"fire".to_string()), "top2={top:?}");
    assert!(top.contains(&"water".to_string()), "top2={top:?}");
}

#[test]
fn deterministic_rebuild() {
    let build = || {
        let mut p = plane();
        synth_speak(&mut p, W, H, "storm", 1.0);
        synth_speak(&mut p, W, H, "life", 0.7);
        cell_speak_back(&p, W, H, LEX, 0.0, LEX.len())
    };
    let a = build();
    let b = build();
    assert_eq!(a, b);
    // scores bit-identical
    for ((_, sa), (_, sb)) in a.iter().zip(&b) {
        assert_eq!(sa.to_bits(), sb.to_bits());
    }
}

#[test]
fn damped_blob_still_decodes() {
    // damping scales amplitude uniformly → cosine invariant → decode survives.
    let mut p = plane();
    synth_speak(&mut p, W, H, "wind", 1.0);
    for v in &mut p {
        *v *= 0.05; // heavy uniform decay
    }
    assert_eq!(cell_decode_top1(&p, W, H, LEX, 0.1).as_deref(), Some("wind"));
}

#[test]
fn signature_mirror_matches_channels() {
    // signature nonzero exactly on word_channels
    for word in LEX {
        let sig = word_signature(word);
        let chs = word_channels(word);
        for (i, s) in sig.iter().enumerate() {
            assert_eq!(*s > 0.0, chs.contains(&i), "word={word} ch={i}");
        }
    }
}

/// PARITY PIN (kimi audit F1, 2026-08-01): ground truth captured from the
/// REAL field::speak (cosmo-speak@c5ef614e, via field crate example run) and
/// independently re-derived in Python (different-path verification, all
/// values bit-exact). If this test fails, this crate's mirror drifted from
/// the field crate again.
#[test]
fn prime_parity_vs_field_truth() {
    // (word, seed, channels, amp f32 bit patterns, centers @32x32)
    let truth: &[(&str, u64, [usize; 3], [u32; 3], [(usize, usize); 3])] = &[
        ("fire", 0xaa77f578efdfc4b9, [26, 9, 20],
         [0x3ef89eeb, 0x3f797dfe, 0x3eebf2b3], [(10, 14), (6, 6), (18, 10)]),
        ("water", 0xd3cacd4c82e5be70, [27, 5, 21],
         [0x3effc7ce, 0x3efbceee, 0x3f3bc5be], [(14, 14), (22, 2), (22, 10)]),
        ("wind", 0xa6200ef65560e507, [8, 16, 41],
         [0x3f46d7a8, 0x3f0046e5, 0x3f004f1a], [(2, 6), (2, 10), (6, 22)]),
        ("愛", 0x33a7291b74bac894, [45, 34, 12],
         [0x3ed35b28, 0x3f55627e, 0x3f75937a], [(22, 22), (10, 18), (18, 6)]),
        ("mu", 0x08a94f07b5503577, [16, 50, 10],
         [0x3f4c8900, 0x3ef111b9, 0x3f233564], [(2, 10), (10, 26), (10, 6)]),
    ];
    for (word, seed, chs, amp_bits, ctrs) in truth {
        let s = word_seed(word);
        assert_eq!(s, *seed, "seed drift for {word}");
        assert_eq!(word_channels(word), chs.to_vec(), "channel drift for {word}");
        for (i, &c) in chs.iter().enumerate() {
            assert_eq!(
                channel_amp(s, c).to_bits(),
                amp_bits[i],
                "amp drift for {word} ch {c}"
            );
            assert_eq!(channel_center(32, 32, c), ctrs[i], "center drift for {word}");
        }
    }
}
