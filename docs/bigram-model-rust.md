# Bigram Language Model in Rust

## Introduction

The Bigram Model is a step up from the Unigram Model in statistical language modeling. It considers pairs of adjacent words (bigrams) rather than individual words. This allows the model to capture some basic context and local word dependencies, leading to more coherent text generation.

## How It Works

1. **Training**: The model counts the frequency of word pairs (bigrams) in the training corpus.
2. **Probability Calculation**: The probability of a word is calculated based on the previous word: P(word2 | word1) = count(word1, word2) / count(word1).
3. **Text Generation**: Words are selected based on their probability given the previous word.

## Implementation in Rust

Here's a basic implementation of a Bigram Model in Rust:

[Link to `bigram-model-rust_1.rs`](../src/bigram-model-rust_1.rs)

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