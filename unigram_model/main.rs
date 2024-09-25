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
    fn perplexity(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len();
        let mut log_likelihood = 0.0;

        for word in words {
            let count = self.word_counts.get(word).unwrap_or(&0);
            let probability = *count as f64 / self.total_words as f64;
            log_likelihood += (probability + 1e-10).ln();
        }

        (-log_likelihood / n as f64).exp()
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

    // Calculate perplexity on a test sentence
    let test_sentence = "to be or not to be";
    let perplexity = model.perplexity(test_sentence);
    println!("Perplexity on '{}': {}", test_sentence, perplexity);
}