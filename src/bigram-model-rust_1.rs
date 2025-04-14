use std::collections::HashMap;
use rand::Rng;

struct BigramModel {
    bigram_counts: HashMap<(String, String), usize>,
    word_counts: HashMap<String, usize>,
    vocab: Vec<String>,
}

impl BigramModel {
    fn new() -> Self {
        BigramModel {
            bigram_counts: HashMap::new(),
            word_counts: HashMap::new(),
            vocab: Vec::new(),
        }
    }

    fn train(&mut self, text: &str) {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        self.vocab = words.clone();

        for word in &words {
            *self.word_counts.entry(word.clone()).or_insert(0) += 1;
        }

        for pair in words.windows(2) {
            let bigram = (pair[0].clone(), pair[1].clone());
            *self.bigram_counts.entry(bigram).or_insert(0) += 1;
        }
    }

    fn generate(&self, num_words: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut output = Vec::new();

        // Start with a random word
        let mut current_word = self.vocab[rng.gen_range(0..self.vocab.len())].clone();
        output.push(current_word.clone());

        for _ in 1..num_words {
            let possible_next_words: Vec<_> = self.bigram_counts
                .iter()
                .filter(|((w1, _), _)| w1 == &current_word)
                .collect();

            if possible_next_words.is_empty() {
                // If no bigrams starting with the current word, choose a random word
                current_word = self.vocab[rng.gen_range(0..self.vocab.len())].clone();
            } else {
                let total_count: usize = possible_next_words.iter().map(|(_, &count)| count).sum();
                let mut rand_num = rng.gen_range(0..total_count);

                for ((_, next_word), &count) in possible_next_words {
                    if rand_num < count {
                        current_word = next_word.clone();
                        break;
                    }
                    rand_num -= count;
                }
            }

            output.push(current_word.clone());
        }

        output.join(" ")
    }

    fn perplexity(&self, text: &str) -> f64 {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let n = words.len() - 1;  // number of bigrams
        let mut log_likelihood = 0.0;

        for pair in words.windows(2) {
            let bigram = (pair[0].clone(), pair[1].clone());
            let bigram_count = *self.bigram_counts.get(&bigram).unwrap_or(&0) as f64;
            let word_count = *self.word_counts.get(&pair[0]).unwrap_or(&0) as f64;
            
            let probability = if word_count > 0.0 { bigram_count / word_count } else { 0.0 };
            log_likelihood += (probability + 1e-10).ln();
        }

        (-log_likelihood / n as f64).exp()
    }
}

fn main() {
    let mut model = BigramModel::new();
    
    // Training data: famous quotes
    let corpus = "To be or not to be that is the question \
                  I think therefore I am \
                  Ask not what your country can do for you ask what you can do for your country";
    
    model.train(corpus);
    
    // Generate text
    let generated_text = model.generate(15);
    println!("Generated text: {}", generated_text);

    // Calculate perplexity
    let test_text = "To be or not to be";
    let perplexity = model.perplexity(test_text);
    println!("Perplexity on '{}': {:.2}", test_text, perplexity);
}