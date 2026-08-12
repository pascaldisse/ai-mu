//! speak-back s0 demo: synthetic field state → word.

use speakback_s0::{decode_top1, speak, speak_back, Lexicon, Slice};

fn main() {
    let d = 4096;
    let lex = Lexicon::new(
        d,
        &["fire", "water", "storm", "love", "seed"],
        0xC0DE_CAFE,
        0.25,
    );

    let mut state = Slice::zeros(d);
    speak(&mut state, &lex, "fire", 1.0);

    println!("spoke 'fire' into field");
    println!("top-1 decode: {:?}", decode_top1(&state, &lex));
    println!("top-3: {:?}", speak_back(&state, &lex, 3));

    speak(&mut state, &lex, "water", 0.8);
    println!("\nadded 'water' @ 0.8");
    println!("top-3: {:?}", speak_back(&state, &lex, 3));
}
