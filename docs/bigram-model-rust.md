# Bigram Language Model in Rust

## Introduction

The Bigram Model is a step up from the Unigram Model in statistical language modeling. It considers the context of the *immediately preceding word* when calculating the probability of the current word. It's based on the Markov assumption that the probability of a word depends only on the previous word. This allows the model to capture some basic local word dependencies, leading to more coherent text generation than the Unigram model.

## How It Works

1.  **Training**: The model processes a training corpus, counting the frequency of each individual word (unigrams) and each sequence of two adjacent words (bigrams). Special start-of-sentence (`<s>`) and end-of-sentence (`</s>`) tokens are often added to handle beginnings and ends of sequences.
2.  **Probability Calculation**: The probability of a word \( w_i \) given the preceding word \( w_{i-1} \) is calculated using the counts obtained during training:
    \[ P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})} \]
    This is the Maximum Likelihood Estimation (MLE) probability.
3.  **Text Generation**: Generation typically starts with a special start token (`<s>`) or a randomly chosen word. Then, the next word is sampled based on the conditional probability given the *last* word generated. This process repeats.

## Example

Consider the corpus: "<s> the cat sat </s> <s> the cat ran </s>"

*   Unigram Counts: {"<s>": 2, "the": 2, "cat": 2, "sat": 1, "</s>": 2, "ran": 1}
*   Bigram Counts: {("<s>", "the"): 2, ("the", "cat"): 2, ("cat", "sat"): 1, ("sat", "</s>"): 1, ("cat", "ran"): 1, ("ran", "</s>"): 1}
*   Example Probabilities:
    *   P("the" | "<s>") = count("<s>", "the") / count("<s>") = 2 / 2 = 1.0
    *   P("cat" | "the") = count("the", "cat") / count("the") = 2 / 2 = 1.0
    *   P("sat" | "cat") = count("cat", "sat") / count("cat") = 1 / 2 = 0.5
    *   P("ran" | "cat") = count("cat", "ran") / count("cat") = 1 / 2 = 0.5
    *   P("</s>" | "sat") = count("sat", "</s>") / count("sat") = 1 / 1 = 1.0
    *   P("dog" | "the") = count("the", "dog") / count("the") = 0 / 2 = 0.0 (Problem!)

## Smoothing

The example highlights a key problem: what happens if a bigram (like "the dog") never appeared in the training data? The MLE probability is 0. This means the model can never generate that sequence, even if it's plausible. This is the **zero-frequency problem**.

**Smoothing techniques** are used to address this by assigning a small non-zero probability to unseen events. Common methods include:
*   **Add-k Smoothing (e.g., Add-One/Laplace Smoothing):** Add a small constant \( k \) to all counts.
*   **Interpolation:** Combine unigram and bigram probabilities.
*   **Backoff:** Use unigram probabilities if the bigram count is zero. (Covered in the next stage).

## Implementation in Rust

Here's a basic implementation of a Bigram Model in Rust:

[Link to `bigram-model-rust_1.rs`](../src/bigram-model-rust_1.rs) | [Python Equivalent (`bigram_model.py`)](../src/bigram_model.py)

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