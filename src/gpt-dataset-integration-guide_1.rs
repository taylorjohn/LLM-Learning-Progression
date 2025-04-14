use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let file = File::open("data/wikitext-2-raw/wiki.train.raw").unwrap();
    let reader = BufReader::new(file);
    let mut corpus = String::new();
    
    for line in reader.lines().take(100) {  // Take first 100 lines for simplicity
        corpus.push_str(&line.unwrap());
        corpus.push('\n');
    }

    let mut model = UnigramModel::new();
    model.train(&corpus);

    // Generate text
    let generated = model.generate(10);  // Generate 10 words
    println!("Generated: {}", generated);
}