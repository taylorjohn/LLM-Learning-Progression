# Bigram Language Model in Rust

## Introduction

The Bigram Model is a step up from the Unigram Model in statistical language modeling. It considers pairs of adjacent words (bigrams) rather than individual words. This allows the model to capture some basic context and local word dependencies, leading to more coherent text generation.

## How It Works

1. **Training**: The model counts the frequency of word pairs (bigrams) in the training corpus.
2. **Probability Calculation**: The probability of a word is calculated based on the previous word: P(word2 | word1) = count(word1, word2) / count(word1).
3. **Text Generation**: Words are selected based on their probability given the previous word.

## Implementation in Rust

Here's a basic implementation of a Bigram Model in Rust:

```rust
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
```

## Explanation

1. The `BigramModel` struct stores bigram counts, individual word counts, and a vocabulary list.
2. The `train` method tokenizes the input text and counts bigram and word occurrences.
3. The `generate` method produces new text by selecting words based on their probability given the previous word.
4. The `perplexity` method calculates the model's perplexity on a given text, which is used for evaluation.

## Advantages over Unigram Model

- Captures some local context and word order
- Generally produces more coherent text
- Can learn common phrases and word pairs

## Limitations

- Still limited in capturing long-range dependencies
- May struggle with rare or unseen bigrams
- Can't capture context beyond the immediately preceding word

## Evaluation

We use perplexity to evaluate the Bigram Model, just as we did with the Unigram Model. However, the Bigram Model typically achieves lower (better) perplexity scores on the same text compared to the Unigram Model, as it captures more information about word sequences.

## Next Steps

The Bigram Model improves upon the Unigram Model by considering pairs of words, but it's still limited in the amount of context it can capture. The next step in our progression will be the N-gram Model, which generalizes to considering any number of previous words, allowing for even more context to be captured.