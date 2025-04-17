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
- [Guide: Running External Assignments on Colab](docs/running_external_assignments_colab.md) *(New)*
- [Guide: BPE Tokenizer Implementation](docs/bpe_tokenizer_implementation.md) *(New)*

---

## Overview: Language Model Creation Lifecycle

While this repository progresses through historical and foundational models, creating a modern, large-scale language model like GPT involves a complex lifecycle. Here's a high-level overview:

### 1. Data Collection & Curation (Pre-training)

*   **Goal:** Gather a massive and diverse text dataset representing the breadth of language use and world knowledge the model should learn.
*   **Sources:** Typically involves scraping vast amounts of text from the public web (e.g., using crawls like Common Crawl), books (e.g., Project Gutenberg, library scans), articles (e.g., Wikipedia, news archives), code repositories (e.g., GitHub), and potentially specialized corpora.
*   **Scale:** Modern pre-training datasets often contain hundreds of billions to trillions of tokens.
*   **Considerations:** Licensing, PII (Personally Identifiable Information), bias representation, domain coverage.

### 2. Data Cleansing & Preprocessing

*   **Goal:** Prepare the raw collected data for model training, improving quality and consistency.
*   **Steps:**
    *   **Deduplication:** Removing duplicate or near-duplicate documents/passages.
    *   **Quality Filtering:** Removing low-quality text (e.g., boilerplate, short/nonsensical content, excessive code/markup). Filtering based on heuristics or classifier models is common.
    *   **PII Removal/Anonymization:** Attempts to remove or mask sensitive personal information.
    *   **Tokenization:** Applying a subword tokenizer (like BPE or SentencePiece) trained on a representative sample of the data to convert text into sequences of integers (token IDs).
    *   **Formatting:** Structuring the data into sequences suitable for model input (e.g., packing multiple documents into fixed-length sequences).

### 3. Transformer Model Construction

*   **Goal:** Define the specific architecture of the language model.
*   **Components (based on Transformer Decoder):**
    *   **Embedding Layer:** Maps token IDs to dense vectors.
    *   **Positional Encoding:** Adds position information.
    *   **Stack of Decoder Layers:** Typically dozens of layers (e.g., GPT-3 has 96). Each layer contains:
        *   Masked Multi-Head Self-Attention
        *   Layer Normalization & Residual Connections
        *   Position-wise Feed-Forward Network
    *   **Final Layer:** Maps final representations to vocabulary logits.
*   **Hyperparameters:** Selecting the number of layers, hidden dimension size, number of attention heads, vocabulary size, context window length, activation functions, etc.

### 4. Model Training (Pre-training)

*   **Goal:** Optimize the model parameters (weights and biases) to predict the next token in a sequence accurately based on the pre-processed data.
*   **Process:**
    *   **Objective:** Typically Autoregressive Language Modeling using Cross-Entropy Loss.
    *   **Optimization:** Using optimizers like Adam or AdamW with learning rate schedules (e.g., warmup followed by decay).
    *   **Large-Scale Distributed Training:** Training requires massive computational resources (hundreds or thousands of GPUs/TPUs) and sophisticated distributed training techniques (e.g., data parallelism, tensor parallelism, pipeline parallelism) using frameworks like PyTorch FSDP, DeepSpeed, Megatron-LM.
    *   **Duration:** Can take weeks or months on large compute clusters.
    *   **Checkpointing:** Saving model state frequently is crucial.

### 5. Evaluation (Pre-deployment)

*   **Goal:** Assess the pre-trained model's capabilities, limitations, and potential risks before fine-tuning or deployment.
*   **Methods:**
    *   **Language Modeling Performance:** Measuring perplexity on held-out validation datasets.
    *   **Benchmark Evaluations:** Testing performance on a wide range of downstream NLP tasks (e.g., GLUE, SuperGLUE benchmarks) often in zero-shot or few-shot settings.
    *   **Bias and Safety Testing:** Probing the model for social biases, toxicity generation, and potential harms using specialized datasets and red-teaming techniques.
    *   **Qualitative Analysis:** Human evaluation of generated text for coherence, factuality, and adherence to instructions.

This lifecycle highlights the significant engineering, resource, and ethical considerations involved in building state-of-the-art language models beyond the simplified examples in this repository.

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

## Project Implementation Learnings

This section captures specific insights gained during the implementation phases of this project.

### Rust (`cs336_basics_rs`)

*   **Project Structure:** Utilized `cargo new --lib` for library setup.
*   **Testing Strategy:** Employed both unit tests (`src/lib.rs` with `#[cfg(test)]`) and integration tests (`tests/` directory) to ensure correctness and validate the public API.
*   **Core Workflow:** Relied on `cargo build` for compilation and `cargo test` for running all test suites.
*   **Iterators & Closures:** Leveraged Rust's iterator methods (`split_whitespace`, `map`, `filter`) and closures for concise data processing, particularly in the `tokenize` function.
*   **String Handling:** Worked with `&str` and `String`, using methods like `trim_matches`, `to_lowercase`, `is_empty`, `chars().any()`, and `collect()` for tokenization logic.
*   **`trim_matches` Nuance:** Discovered the importance of the closure's return value (`true` => trim). Corrected the initial logic for trimming leading/trailing punctuation by simplifying the condition to `!c.is_alphanumeric()`.
*   **`HashMap` Usage:** Implemented word counting using `HashMap::entry` and `or_insert` for efficient frequency tracking.
*   **Test-Driven Debugging:** Used failing tests (unit and integration) to identify and fix subtle logical errors in both the implementation (`tokenize`) and the tests themselves.

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
