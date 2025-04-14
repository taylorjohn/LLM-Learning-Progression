# Unigram Language Model in Rust

## Introduction

The Unigram Model is the simplest form of statistical language model. It treats each word as an independent event, ignoring any context or relationship between words. Despite its simplicity, it serves as a fundamental building block for understanding more complex language models.

## How It Works

1. **Training**: The model counts the frequency of each word in the training corpus.
2. **Probability Calculation**: The probability of a word is calculated as its count divided by the total number of words in the corpus.
3. **Text Generation**: Words are randomly selected based on their probabilities.

## Implementation in Rust

Here's a basic implementation of a Unigram Model in Rust:

[Link to `unigram-model-rust_1.rs`](../src/unigram-model-rust_1.rs)

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

[Link to `unigram-model-rust_2.rs`](../src/unigram-model-rust_2.rs)

## Next Steps

While the Unigram Model serves as a good starting point, it's limited in its ability to generate coherent text. The next step in our progression will be the Bigram Model, which considers pairs of words, allowing it to capture some basic context and improve text generation.