use std::collections::HashMap;
use rand::Rng;

// Make struct public
pub struct UnigramModel {
    word_counts: HashMap<String, usize>,
    total_words: usize,
}

// Make impl block public
impl UnigramModel {
    // Make methods public
    pub fn new() -> Self {
        UnigramModel {
            word_counts: HashMap::new(),
            total_words: 0,
        }
    }

    pub fn train(&mut self, text: &str) {
        for word in text.split_whitespace() {
            *self.word_counts.entry(word.to_string()).or_insert(0) += 1;
            self.total_words += 1;
        }
    }

    pub fn generate(&self, num_words: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut output = Vec::new();

        // Handle edge case where model is not trained
        if self.total_words == 0 {
            return String::new();
        }

        for _ in 0..num_words {
            let rand_num = rng.gen_range(0..self.total_words);
            let mut cumulative = 0;
            let mut selected_word = ""; // Placeholder
            for (word, &count) in &self.word_counts {
                cumulative += count;
                if cumulative > rand_num {
                    selected_word = word;
                    break;
                }
            }
             // Ensure a word is selected even if rand_num equals total_words (rare edge case)
            if selected_word.is_empty() {
                 // Use map_or to handle Option<&String> and provide &str default
                 selected_word = self.word_counts.keys().next().map_or("", |s| s);
            }
            output.push(selected_word.to_string());
        }

        output.join(" ")
    }
    pub fn perplexity(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len();
        if n == 0 || self.total_words == 0 {
            return f64::INFINITY; // Or some other indicator of undefined perplexity
        }
        let mut log_likelihood = 0.0;

        for word in words {
            // Handle unknown words - assign a very small probability (add-epsilon smoothing)
            let count = self.word_counts.get(word).unwrap_or(&0);
            let probability = (*count as f64 + 1e-10) / (self.total_words as f64 + 1e-10 * self.word_counts.len() as f64);
            log_likelihood += probability.ln();
        }

        (-log_likelihood / n as f64).exp()
    }

}

/* // Keep main function commented out as an example if desired
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
*/

// Add test module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_model() {
        let model = UnigramModel::new();
        assert_eq!(model.total_words, 0);
        assert!(model.word_counts.is_empty());
    }

    #[test]
    fn test_train_model() {
        let mut model = UnigramModel::new();
        let corpus = "hello world hello";
        model.train(corpus);

        assert_eq!(model.total_words, 3);
        assert_eq!(model.word_counts.len(), 2);
        assert_eq!(model.word_counts.get("hello"), Some(&2));
        assert_eq!(model.word_counts.get("world"), Some(&1));
    }

    #[test]
    fn test_generate_empty() {
        let model = UnigramModel::new();
        let generated = model.generate(10);
        assert!(generated.is_empty());
    }

    #[test]
    fn test_generate_simple() {
        let mut model = UnigramModel::new();
        model.train("a a a"); // Only one word
        let generated = model.generate(5);
        assert_eq!(generated, "a a a a a");
    }

    #[test]
    fn test_generate_produces_correct_length() {
        let mut model = UnigramModel::new();
        model.train("some random text here");
        let num_words = 7;
        let generated = model.generate(num_words);
        // Check number of words generated matches requested number
        assert_eq!(generated.split_whitespace().count(), num_words);
    }

     #[test]
    fn test_perplexity_empty_text() {
        let mut model = UnigramModel::new();
        model.train("a b c");
        let perp = model.perplexity("");
        assert!(perp.is_infinite());
    }

     #[test]
    fn test_perplexity_empty_model() {
        let model = UnigramModel::new();
        let perp = model.perplexity("a b c");
        assert!(perp.is_infinite());
    }

    #[test]
    fn test_perplexity_known_words() {
        let mut model = UnigramModel::new();
        // Simple corpus: each word appears once
        let corpus = "a b c";
        model.train(corpus);
        // P(a) = 1/3, P(b) = 1/3, P(c) = 1/3
        // Perplexity = ( (1/3) * (1/3) * (1/3) ) ^ (-1/3)
        // Perplexity = (1/27) ^ (-1/3) = 27 ^ (1/3) = 3
        let perplexity = model.perplexity("a b c");
        // Use approx assertion due to floating point and smoothing
        assert!((perplexity - 3.0).abs() < 1e-9);
    }

     #[test]
    fn test_perplexity_repeated_words() {
        let mut model = UnigramModel::new();
        let corpus = "a a b";
        model.train(corpus);
        // P(a) = 2/3, P(b) = 1/3
        // Perplexity("a b") = ( P(a) * P(b) ) ^ (-1/2)
        // Perplexity = ( (2/3) * (1/3) ) ^ (-1/2)
        // Perplexity = ( 2/9 ) ^ (-1/2) = (9/2) ^ (1/2) = 3 / sqrt(2)
        let expected_perplexity = 3.0 / (2.0 as f64).sqrt();
        let perplexity = model.perplexity("a b");
        assert!((perplexity - expected_perplexity).abs() < 1e-9);
    }

    #[test]
    fn test_perplexity_unknown_word() {
        let mut model = UnigramModel::new();
        model.train("a b");
        // Test with 'c', which is unknown. Perplexity should be high due to smoothing.
        let perplexity = model.perplexity("a c");
        assert!(perplexity > 0.0);
        assert!(perplexity.is_finite());
        // We expect perplexity to be higher than if 'c' was known with some probability
        let perplexity_known = model.perplexity("a b");
        assert!(perplexity > perplexity_known);
    }
}
