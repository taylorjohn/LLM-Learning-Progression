# N-gram Language Model in Rust

## Introduction

The N-gram Model is a generalization of the Unigram and Bigram models. It considers sequences of N words, where N can be any positive integer. This allows the model to capture more context than its predecessors, potentially leading to more coherent and contextually appropriate text generation.

## How It Works

1. **Training**: The model counts the frequency of word sequences of length N in the training corpus.
2. **Probability Calculation**: The probability of a word is calculated based on the previous N-1 words: P(wordN | word1, ..., wordN-1) = count(word1, ..., wordN) / count(word1, ..., wordN-1).
3. **Text Generation**: Words are selected based on their probability given the previous N-1 words.

## Implementation in Rust

Here's a basic implementation of an N-gram Model in Rust:

[Link to `n-gram-model-rust_1.rs`](../src/n-gram-model-rust_1.rs)

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