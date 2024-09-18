# Unigram Language Model in Rust

## Introduction

The Unigram Model is the simplest form of statistical language model. It treats each word as an independent event, ignoring any context or relationship between words. Despite its simplicity, it serves as a fundamental building block for understanding more complex language models.

## How It Works

1. **Training**: The model counts the frequency of each word in the training corpus.
2. **Probability Calculation**: The probability of a word is calculated as its count divided by the total number of words in the corpus.
3. **Text Generation**: Words are randomly selected based on their probabilities.

## Implementation in Rust

Here's a basic implementation of a Unigram Model in Rust:

```rust
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
```

## Explanation

1. The `UnigramModel` struct stores word counts and the total number of words.
2. The `train` method tokenizes the input text and counts word occurrences.
3. The `generate` method produces new text by randomly selecting words based on their frequencies.

## Advantages

- Simple to implement and understand
- Fast training and generation
- Requires minimal memory

## Limitations

- Ignores word order and context
- Cannot capture phrases or multi-word expressions
- Generated text often lacks coherence and grammatical structure

## Evaluation

To evaluate the Unigram Model, we can use perplexity, which measures how well the model predicts a sample of text. Lower perplexity indicates better performance.

```rust
impl UnigramModel {
    // ... (previous methods)

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
```

## Next Steps

While the Unigram Model serves as a good starting point, it's limited in its ability to generate coherent text. The next step in our progression will be the Bigram Model, which considers pairs of words, allowing it to capture some basic context and improve text generation.