# Unigram Language Model in Rust

## Introduction

The Unigram Model is the simplest form of statistical language model. It treats each word as an independent event, ignoring any context or relationship between words. This is based on the strong (and often unrealistic) assumption that the probability of a word occurring depends only on the word itself. Despite its simplicity, it serves as a fundamental building block for understanding more complex language models.

## How It Works

1. **Training**: The model processes a training corpus (a collection of text) and counts the frequency of each unique word.
2. **Probability Calculation**: The probability of a specific word \( w \) appearing is calculated as its frequency (count) divided by the total number of words (\( N \)) in the corpus:
    \[ P(w) = \frac{\text{count}(w)}{N} \]
3. **Text Generation**: To generate text, words are randomly sampled from the vocabulary according to their calculated probabilities. Imagine a weighted roulette wheel where each word gets a slice proportional to its probability.

## Example

Consider the corpus: "the cat sat on the mat"

*   Total words (N) = 6
*   Counts: {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}
*   Probabilities:
    *   P("the") = 2/6 = 1/3
    *   P("cat") = 1/6
    *   P("sat") = 1/6
    *   P("on") = 1/6
    *   P("mat") = 1/6

Generating text would involve randomly picking words based on these probabilities.

## Implementation in Rust

Here's a basic implementation of a Unigram Model in Rust:

[Link to `unigram-model-rust_1.rs`](../src/unigram-model-rust_1.rs) | [Python Equivalent (`unigram_model.py`)](../src/unigram_model.py)

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