fn main() {
    let file = File::open("data/wikitext-2-raw/wiki.train.raw").unwrap();
    let reader = BufReader::new(file);
    let mut corpus = String::new();
    
    for line in reader.lines().take(1000) {  // Take first 1000 lines
        corpus.push_str(&line.unwrap());
        corpus.push('\n');
    }

    let mut model = NGramModel::new(3);  // For trigram model
    model.train(&corpus);

    // Generate text
    let generated = model.generate("The quick brown", 20);  // Generate 20 words
    println!("Generated: {}", generated);
}