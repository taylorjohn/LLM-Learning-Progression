# Word Embeddings: Representing Words as Vectors

## Introduction

Before diving into neural network-based language models (like FFNNs, RNNs, LSTMs, and Transformers), it's crucial to understand how we represent words numerically. Simple methods like one-hot encoding (representing each word as a huge vector with a single '1' and the rest '0's) are inefficient and don't capture semantic relationships (e.g., "king" and "queen" are related, but their one-hot vectors are orthogonal).

**Word embeddings** solve this by representing words as dense, low-dimensional vectors in a continuous vector space. The key idea is that words with similar meanings or that appear in similar contexts should have vectors that are close to each other in this space.

## Why Use Word Embeddings?

-   **Dimensionality Reduction:** Instead of sparse vectors with tens of thousands of dimensions (vocabulary size), embeddings are typically much smaller (e.g., 50-300 dimensions).
-   **Capturing Semantics:** The spatial relationships between vectors capture semantic similarities and analogies (e.g., vector("king") - vector("man") + vector("woman") ≈ vector("queen")).
-   **Improved Model Performance:** Neural networks learn more effectively from these dense representations compared to sparse ones.

## Common Techniques

While modern models like Transformers often learn embeddings as part of their end-to-end training, several foundational techniques were developed specifically for pre-computing word embeddings from large text corpora:

1.  **Word2Vec (Mikolov et al., 2013):**
    *   **CBOW (Continuous Bag-of-Words):** Predicts the current word based on its surrounding context words.
    *   **Skip-gram:** Predicts the surrounding context words given the current word. Skip-gram generally performs better for infrequent words and captures finer-grained relationships.

2.  **GloVe (Global Vectors for Word Representation - Pennington et al., 2014):**
    *   Combines aspects of global matrix factorization (like Latent Semantic Analysis) and local context window methods (like Word2Vec).
    *   Learns embeddings by factorizing a matrix of word-word co-occurrence counts from the corpus.

3.  **FastText (Bojanowski et al., 2016):**
    *   An extension of Word2Vec that represents words as bags of character n-grams.
    *   This allows it to generate embeddings for out-of-vocabulary words and often works well for morphologically rich languages.

## Usage in Language Models

These pre-trained embeddings can be used as the initial input layer for downstream NLP tasks and models (like the FFNN, RNN, and LSTM language models discussed later). Alternatively, embeddings can be learned from scratch specifically for the task at hand, which is common in large transformer models.

Understanding the concept of representing words as dense vectors is fundamental to grasping how modern neural language models process and "understand" text.

---

**(Next Section: [Feed-Forward Neural Network Language Model](ffnn-language-model-rust.md))**
