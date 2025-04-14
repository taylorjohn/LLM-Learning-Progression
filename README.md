# GPT Model Progression Repository

## Introduction

This repository provides a step-by-step progression of **Language Models** (LMs), from **unigram models** to the advanced **GPT-3** architecture. Each stage includes explanations, exercises, and solutions to help learners understand the concepts behind each model and how to implement them in **Rust**.

This repository is structured to facilitate learning through:
- **Model descriptions**: Each model has a dedicated README file.
- **Exercises**: Practical exercises to solidify understanding.
- **Solutions**: Example solutions to exercises, helping learners check their work.

---

## Repository Structure

Here is an overview of the repository structure, with links to each stage of the progression:

### 1. [Unigram Model](docs/unigram-model-rust.md)

### 2. [Bigram Model](docs/bigram-model-rust.md)

### 3. [N-gram Model](docs/n-gram-model-rust.md)

### 4. [N-gram Model with Backoff](docs/ngram-backoff-model-rust.md)

### 5. [Word Embeddings (Word2Vec/GloVe)](docs/word-embeddings.md) *(New - Conceptual Introduction)*

### 6. [Feed-Forward Neural Network Language Model](docs/ffnn-language-model-rust.md)

### 7. [Recurrent Neural Network (RNN) Language Model](docs/rnn-language-model-rust.md)

### 8. [Long Short-Term Memory (LSTM) Language Model](docs/lstm-language-model-rust.md) *(Could mention GRUs here)*

### 9. [Attention Mechanism](docs/attention-mechanism.md) *(New - Conceptual Introduction)*

### 10. [Transformer Language Model](docs/transformer-language-model-rust.md)

### 11. [Simplified GPT Language Model](docs/gpt-language-model-rust.md)

### 12. [Advanced GPT Implementations](docs/advanced-gpt-rust.md) 
(Covers variations like M2 Optimized and State-of-the-Art versions - see also [m2-optimized-gpt-rust.md](docs/m2-optimized-gpt-rust.md), [state-of-the-art-gpt-rust.md](docs/state-of-the-art-gpt-rust.md), etc.)

### 13. [Fine-Tuning GPT Models](docs/fine-tuning-gpt.md) *(New - Practical Guide)*

---

## Supporting Documentation

Additional documents that provide further explanation of various concepts, comparisons, and model progression.

- [Progression Summary](docs/language-model-progression-summary.md)
- [GPT Progression Readme](docs/gpt-progression-readme.md)
- [Terminology & Definitions (BPE)](docs/llm-terminology-BPE.md)
- [GPT Repo Structure Guide](docs/gpt-repo-structure.md)
- [GPT Training Data Guide](docs/gpt-training-data-guide.md)
- [GPT Dataset Integration Guide](docs/gpt-dataset-integration-guide.md)
- [Model Comparison](docs/language-model-comparison.md)
- [Future Directions (Beyond GPT)](docs/beyond-gpt-next-steps.md)
- [Evaluation Metrics (Perplexity, BLEU, etc.)](docs/evaluation-metrics.md) *(New)*

---

## How to Use the Repository

1.  **Follow the Progression:** Start with the first stage listed in the **Repository Structure** section above ([Unigram Model](docs/unigram-model-rust.md)). Read the corresponding documentation file in the `docs/` directory.
2.  **Understand the Concepts:** Each documentation file explains the core ideas, advantages, and limitations of that particular model or concept.
3.  **Examine the Code:** Where applicable, the documentation files link to relevant code examples (Rust `.rs` or Python `.py` files) located in the `src/` directory. Review this code to see a simplified implementation of the concepts discussed.
4.  **Run Tests (Optional):**
    *   For the Rust code (primarily the Unigram model initially), you can navigate to the `src/` directory and run `cargo test` (ensure Rust/Cargo is installed).
    *   For Python code, you can run `python3 -m pytest` from the root directory (ensure Python3, pip3, pytest, and dependencies from `requirements.txt` are installed). Note that the current Python tests primarily check if code can be imported.
5.  **Advance Sequentially:** Move through the stages listed in the **Repository Structure** section in order. Each stage builds upon the previous ones. Conceptual stages (like Word Embeddings, Attention Mechanism) provide background for subsequent models.

---

## Contributing

If you would like to contribute, feel free to submit a pull request with improvements, additional exercises, or more advanced techniques. Please make sure to follow the current structure and style of the repository.

---

## License

This repository is licensed under the MIT License.

---

## Next Steps

Once you've completed this progression, you can explore additional models and tasks such as:
- **GPT-Neo** or **GPT-J** for larger transformer models.
- **Fine-tuning** on specific datasets for more advanced tasks like text summarization, translation, or domain-specific generation.

---

This updated `README.md` provides **easy navigation** to each of the models, exercises, and solution files, ensuring that learners can quickly find and access the material they need.

Would you like to refine any specific part of the content, or is this good to be implemented in your repository?
