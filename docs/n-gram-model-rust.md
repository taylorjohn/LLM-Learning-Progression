# N-gram Language Model in Rust

## Introduction

The N-gram Model generalizes the Unigram (N=1) and Bigram (N=2) models. It considers a context of the previous **N-1** words to predict the probability of the next word, \( w_i \). This allows the model to capture longer dependencies and potentially generate more coherent text than simpler models, with the trade-off of increased complexity and data requirements. Common choices for N include 3 (trigram), 4 (4-gram), or sometimes 5 (5-gram).

## How It Works

1.  **Training**: The model processes a corpus, counting the frequencies of all unique sequences of N words (n-grams) and N-1 words ((n-1)-grams). Padding with start/end tokens is common.
2.  **Probability Calculation**: The probability of a word \( w_i \) given the preceding N-1 words is estimated using the Maximum Likelihood Estimation (MLE):
    \[ P(w_i | w_{i-N+1}, \dots, w_{i-1}) = \frac{\text{count}(w_{i-N+1}, \dots, w_{i-1}, w_i)}{\text{count}(w_{i-N+1}, \dots, w_{i-1})} \]
3.  **Text Generation**: Generation starts with an initial context (e.g., N-1 start tokens). The next word is sampled based on the conditional probability given the preceding N-1 words generated so far. The context window then slides forward.

## Example (N=3, Trigram)

Consider calculating P("sat" | "the", "cat").
*   We need the count of the trigram ("the", "cat", "sat").
*   We need the count of the bigram ("the", "cat").
*   The probability is count("the", "cat", "sat") / count("the", "cat").

## Data Sparsity and Smoothing

A major challenge with N-gram models, especially for N > 2, is **data sparsity**. The number of possible word sequences grows exponentially with N. For a vocabulary V, there are V<sup>N</sup> possible N-grams. Even with large corpora, many plausible N-grams will never appear in the training data.

This makes the **zero-frequency problem** (encountered in bigrams) much worse. Relying solely on MLE probabilities means the model assigns zero probability to any unseen N-gram, drastically limiting its generative capabilities and causing issues with evaluation metrics like perplexity (potentially resulting in infinite perplexity if a test sequence contains an unseen N-gram).

Therefore, **smoothing techniques** are absolutely essential for practical N-gram models. These techniques redistribute probability mass from seen N-grams to unseen ones:
*   **Add-k Smoothing:** Simple but often performs poorly for N > 2.
*   **Interpolation:** Linearly combines probabilities from different order models (e.g., trigram, bigram, unigram). \( P_{int}(w_i|w_{i-2}w_{i-1}) = \lambda_3 P(w_i|w_{i-2}w_{i-1}) + \lambda_2 P(w_i|w_{i-1}) + \lambda_1 P(w_i) \), where \( \sum \lambda_j = 1 \).
*   **Backoff:** Uses the highest order (N-gram) probability if the count is non-zero; otherwise, "backs off" to the (N-1)-gram probability, potentially recursively. (Detailed in the next section). Advanced backoff methods include Katz backoff and Kneser-Ney smoothing (often considered state-of-the-art for N-gram models).

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