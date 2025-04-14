use std::collections::HashMap;
use rand::Rng;

struct NGramModel {
    n: usize,
    ngram_counts: HashMap<Vec<String>, HashMap<String, usize>>,
    vocab: Vec<String>,
}

impl NGramModel {
    fn new(n: usize) -> Self {
        NGramModel {
            n,
            ngram_counts: HashMap::new(),
            vocab: Vec::new(),
        }
    }

    fn train(&mut self, text: &str) {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        self.vocab = words.clone();

        for window in words.windows(self.n) {
            let context = window[..self.n-1].to_vec();
            let word = window[self.n-1].clone();
            self.ngram_counts
                .entry(context)
                .or_insert_with(HashMap::new)
                .entry(word)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    fn generate(&self, num_words: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut output: Vec<String> = self.vocab
            .choose_multiple(&mut rng, self.n - 1)
            .cloned()
            .collect();

        for _ in 0..num_words {
            let context = output[output.len() - (self.n - 1)..].to_vec();
            let next_word = if let Some(word_counts) = self.ngram_counts.get(&context) {
                let total_count: usize = word_counts.values().sum();
                let mut rand_num = rng.gen_range(0..total_count);
                word_counts
                    .iter()
                    .find(|(_, &count)| {
                        if rand_num < count {
                            true
                        } else {
                            rand_num -= count;
                            false
                        }
                    })
                    .map(|(word, _)| word.clone())
                    .unwrap_or_else(|| self.vocab.choose(&mut rng).unwrap().clone())
            } else {
                self.vocab.choose(&mut rng).unwrap().clone()
            };
            output.push(next_word);
        }

        output.join(" ")
    }

    fn perplexity(&self, text: &str) -> f64 {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let n = words.len() - (self.n - 1);
        let mut log_likelihood = 0.0;

        for window in words.windows(self.n) {
            let context = window[..self.n-1].to_vec();
            let word = &window[self.n-1];
            
            let context_count = self.ngram_counts.get(&context).map(|counts| counts.values().sum::<usize>()).unwrap_or(0) as f64;
            let word_count = self.ngram_counts.get(&context).and_then(|counts| counts.get(word)).unwrap_or(&0) as f64;
            
            let probability = if context_count > 0.0 { word_count / context_count } else { 0.0 };
            log_likelihood += (probability + 1e-10).ln();
        }

        (-log_likelihood / n as f64).exp()
    }
}

fn main() {
    let mut model = NGramModel::new(3);  // Trigram model
    
    // Training data: famous quotes
    let corpus = "To be or not to be that is the question \
                  I think therefore I am \
                  Ask not what your country can do for you ask what you can do for your country";
    
    model.train(corpus);
    
    // Generate text
    let generated_text = model.generate(20);
    println!("Generated text: {}", generated_text);

    // Calculate perplexity
    let test_text = "To be or not to be that is";
    let perplexity = model.perplexity(test_text);
    println!("Perplexity on '{}': {:.2}", test_text, perplexity);
}