use std::collections::HashMap;
use rand::Rng;

struct UnigramModel {
    word_counts: HashMap<String, usize>,
    total_words: usize,
}

impl UnigramModel {
    fn new() -> Self {
        UnigramModel {
            word_counts: HashMap::new(),
            total_words: 0,
        }
    }

    fn train(&mut self, text: &str) {
        for word in text.split_whitespace() {
            *self.word_counts.entry(word.to_string()).or_insert(0) += 1;
            self.total_words += 1;
        }
    }

    fn generate(&self, num_words: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut output = Vec::new();

        for _ in 0..num_words {
            let rand_num = rng.gen_range(0..self.total_words);
            let mut cumulative = 0;
            for (word, &count) in &self.word_counts {
                cumulative += count;
                if cumulative > rand_num {
                    output.push(word.clone());
                    break;
                }
            }
        }

        output.join(" ")
    }
}

fn main() {
    let mut model = UnigramModel::new();
    
    // Training data: famous quotes
    let corpus = "To be or not to be that is the question \
                  I think therefore I am \
                  Ask not what your country can do for you ask what you can do for your country";
    
    model.train(corpus);
    
    // Generate text
    let generated_text = model.generate(10);
    println!("Generated text: {}", generated_text);
}