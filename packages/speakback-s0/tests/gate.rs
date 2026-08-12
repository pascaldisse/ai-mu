//! speak-back s0 gates.

use speakback_s0::{decode_top1, speak, speak_back, Lexicon, Slice};

const D: usize = 4096;
const SEED: u64 = 0xC0DE_CAFE;
const THRESH: f32 = 0.25;

fn lexicon() -> Lexicon {
    Lexicon::new(D, &["fire", "water", "storm", "love", "seed"], SEED, THRESH)
}

#[test]
fn single_channel_decodes_correctly() {
    let lex = lexicon();
    let mut state = Slice::zeros(D);
    speak(&mut state, &lex, "fire", 1.0);
    assert_eq!(decode_top1(&state, &lex).as_deref(), Some("fire"));
}

#[test]
fn distinct_words_decode_distinctly() {
    let lex = lexicon();
    for word in ["fire", "water", "storm", "love", "seed"] {
        let mut state = Slice::zeros(D);
        speak(&mut state, &lex, word, 1.0);
        let top = decode_top1(&state, &lex).expect("threshold crossed");
        assert_eq!(top, word, "{} decoded as {}", word, top);
    }
}

#[test]
fn stronger_channel_wins() {
    let lex = lexicon();
    let mut state = Slice::zeros(D);
    speak(&mut state, &lex, "fire", 0.3);
    speak(&mut state, &lex, "water", 1.0);
    let top = decode_top1(&state, &lex).expect("threshold crossed");
    assert_eq!(top, "water");
}

#[test]
fn two_channels_both_in_top2() {
    let lex = lexicon();
    let mut state = Slice::zeros(D);
    speak(&mut state, &lex, "fire", 1.0);
    speak(&mut state, &lex, "water", 0.8);
    let top2 = speak_back(&state, &lex, 2);
    let words: Vec<&str> = top2.iter().map(|(w, _)| w.as_str()).collect();
    assert!(words.contains(&"fire"), "top2 missing fire: {:?}", words);
    assert!(words.contains(&"water"), "top2 missing water: {:?}", words);
}

#[test]
fn empty_field_decodes_nothing() {
    let lex = lexicon();
    let state = Slice::zeros(D);
    assert_eq!(decode_top1(&state, &lex), None);
}

#[test]
fn deterministic_rebuild() {
    let lex1 = lexicon();
    let lex2 = lexicon();
    assert_eq!(lex1.anchors, lex2.anchors);
    let mut state = Slice::zeros(D);
    speak(&mut state, &lex1, "seed", 1.0);
    let top1_a = decode_top1(&state, &lex1);
    let top1_b = decode_top1(&state, &lex2);
    assert_eq!(top1_a, top1_b);
}
