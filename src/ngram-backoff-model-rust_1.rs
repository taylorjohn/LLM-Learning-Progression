use std::collections::HashMap;
use rand::Rng;

struct NGramModelWithBackoff {
    max_n: usize,
    models: Vec<HashMap<Vec<String>, HashMap<String, usize>>>,
    vocab: Vec<String>,
}

impl NGramModelWithBackoff {
    fn new(max_n: usize) -> Self {
        NGramModelWithBackoff {
            max_n,
            models: vec![HashMap::new(); max_n],
            vocab: Vec::new(),
        }
    }

    fn train(&mut self, text: &str) {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        self.vocab = words.clone();

        for n in 1..=self.max_n {
            for window in words.windows(n) {
                let context = window[..n-1].to_vec();
                let word = window[n-1].clone();
                self.models[n-1]
                    .entry(context)
                    .or_insert_with(HashMap::new)
                    .entry(word)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
    }

    fn predict_next_word(&self, context: &[String]) -> String {
        let mut rng = rand::thread_rng();

        for n in (1..=self.max_n).rev() {
            if context.len() >= n - 1 {
                let current_context = &context[context.len() - (n - 1)..];
                if let Some(word_counts) = self.models[n-1].get(current_context) {
                    let total_count: usize = word_counts.values().sum();
                    let mut rand_num = rng.gen_range(0..total_count);
                    if let Some((word, _)) = word_counts
                        .iter()
                        .find(|(_, &count)| {
                            if rand_num < count {
                                true
                            } else {
                                rand_num -= count;
                                false
                            }
                        })
                    {
                        return word.clone();
                    }
                }
            }
        }

        // If we've backed off all the way and still haven't found a match, choose a random word
        self.vocab.choose(&mut rng).unwrap().clone()
    }

    fn generate(&self, num_words: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut output: Vec<String> = self.vocab
            .choose_multiple(&mut rng, self.max_n - 1)
            .cloned()
            .collect();

        for _ in 0..num_words {
            let next_word = self.predict_next_word(&output);
            output.push(next_word);
        }

        output.join(" ")
    }

    fn perplexity(&self, text: &str) -> f64 {
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let n = words.len();
        let mut log_likelihood = 0.0;

        for i in 0..n {
            let context = &words[if i < self.max_n - 1 { 0 } else { i - (self.max_n - 1) }..i];
            let word = &words[i];
            
            let mut probability = 0.0;
            for m in (1..=self.max_n).rev() {
                if context.len() >= m - 1 {
                    let current_context = &context[context.len() - (m - 1)..];
                    if let Some(word_counts) = self.models[m-1].get(current_context) {
                        let total_count = word_counts.values().sum::<usize>() as f64;
                        let word_count = *word_counts.get(word).unwrap_or(&0) as f64;
                        probability = word_count / total_count;
                        if probability > 0.0 {
                            break;
                        }
                    }
                }
            }
            
            log_likelihood += (probability + 1e-10).ln();
        }

        (-log_likelihood / n as f64).exp()
    }
}

fn main() {
    let mut model = NGramModelWithBackoff::new(3);  // Max trigram model
    
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