# N-gram Language Model in Rust

## Introduction

The N-gram Model is a generalization of the Unigram and Bigram models. It considers sequences of N words, where N can be any positive integer. This allows the model to capture more context than its predecessors, potentially leading to more coherent and contextually appropriate text generation.

## How It Works

1. **Training**: The model counts the frequency of word sequences of length N in the training corpus.
2. **Probability Calculation**: The probability of a word is calculated based on the previous N-1 words: P(wordN | word1, ..., wordN-1) = count(word1, ..., wordN) / count(word1, ..., wordN-1).
3. **Text Generation**: Words are selected based on their probability given the previous N-1 words.

## Implementation in Rust

Here's a basic implementation of an N-gram Model in Rust:

```rust
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
```

## Explanation

1. The `NGramModel` struct is parametrized by `n`, allowing for different n-gram sizes.
2. The `train` method counts occurrences of n-grams in the input text.
3. The `generate` method produces text by selecting words based on the previous n-1 words.
4. The `perplexity` method calculates the model's perplexity on given text, adapting to the n-gram size.

## Advantages over Previous Models

- Flexible context size: Can capture more or less context as needed
- Potentially more coherent text generation for larger N
- Can learn longer phrases and word sequences

## Limitations

- As N increases, data sparsity becomes a problem (many possible n-grams may never appear in training data)
- Large N can lead to overfitting on training data
- Still limited in capturing very long-range dependencies
- Computationally more expensive for large N

## Evaluation

We continue to use perplexity for evaluation. Generally, as N increases, perplexity on the training set decreases (improves), but perplexity on unseen test data might start to increase after a certain point due to overfitting.

## Next Steps

While the N-gram Model offers flexibility in context size, it still has limitations, particularly in handling unseen sequences and capturing long-range dependencies. The next step in our progression could be to introduce smoothing techniques to handle unseen n-grams better, or to move towards neural network-based models that can potentially capture more complex patterns in language.