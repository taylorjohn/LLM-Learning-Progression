# N-gram Language Model with Backoff in Rust

## Introduction

The N-gram Model with Backoff is an improvement over the basic N-gram model. It addresses the problem of unseen n-grams by "backing off" to shorter n-grams when a particular sequence hasn't been observed in the training data. This approach allows the model to make more robust predictions, even for sequences it hasn't explicitly seen before.

## How It Works

1. **Training**: Similar to the basic N-gram model, but we train models for all orders from 1 to N.
2. **Probability Calculation**: 
   - First, try to use the highest order N-gram.
   - If that N-gram hasn't been seen, "back off" to the (N-1)-gram, and so on.
   - Each backoff step includes a penalty factor to prefer higher-order matches when available.
3. **Text Generation**: Words are selected based on the highest-order N-gram available, backing off as needed.

## Implementation in Rust

Here's a basic implementation of an N-gram Model with Backoff in Rust:

```rust
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
```

## Explanation

1. The `NGramModelWithBackoff` struct contains multiple n-gram models, from unigram up to the specified maximum N.
2. The `train` method builds all n-gram models simultaneously.
3. The `predict_next_word` method implements the backoff strategy, trying higher-order n-grams first and backing off to lower orders as needed.
4. The `generate` and `perplexity` methods use this backoff strategy for text generation and evaluation.

## Advantages over Basic N-gram Model

- More robust handling of unseen n-grams
- Can make predictions even for completely novel contexts
- Often performs better on test data, especially for larger N

## Limitations

- Still fundamentally based on counting statistics, limiting its ability to generalize
- Can be computationally expensive, especially for large N
- Doesn't capture semantic relationships between words

## Evaluation

The perplexity calculation now takes into account the backoff strategy. This model should generally achieve lower (better) perplexity scores on test data compared to the basic N-gram model, especially for larger N.

## Next Steps

While the N-gram Model with Backoff addresses some limitations of the basic N-gram model, it's still fundamentally a count-based method. The next significant step in our progression would be to move towards neural network-based models, which can potentially capture more complex patterns and semantic relationships in language. A good next step might be a simple feedforward neural network language model.