# Language Model Comparison: Inputs, Outputs, and Significance

This document compares the key characteristics and significance of each language model stage in our progression.

## Model Comparison Table

| Stage # | Model                                | Typical Input (Training / Generation)                | Typical Output (Generation)                          | Key Difference / Significance                                                                 |
| :------ | :----------------------------------- | :--------------------------------------------------- | :--------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| 1       | Unigram Model                        | Corpus / Previous word (ignored)                     | Single word (based on global frequency)              | Baseline; No context; Probabilistic selection.                                                |
| 2       | Bigram Model                         | Corpus / Previous word                               | Single word (based on previous word)                 | Considers previous word; Basic context; Smoothing needed.                                   |
| 3       | N-gram Model                         | Corpus / Previous N-1 words                          | Single word (based on previous N-1 words)            | Longer fixed context; Increased sparsity.                                                     |
| 4       | N-gram Model w/ Backoff              | Corpus / Previous N-1 words                          | Single word (using highest available n-gram)         | Handles unseen n-grams via smoothing; More robust.                                            |
| 5       | *Word Embeddings (Concept)*          | *(Large Corpus)*                                     | *(Dense word vectors)*                               | *Concept: Represent words as dense vectors capturing semantics; Enables neural models.*       |
| 6       | Feed-Forward NN LM                   | Corpus (Windowed) / Previous N-1 words (indices)     | Probability distribution over vocab                  | Neural approach; Uses embeddings; Generalizes better but fixed context.                       |
| 7       | Recurrent NN (RNN) LM                | Corpus (Sequences) / Sequence of words (indices)     | Probability distribution over vocab (per step)       | Handles variable length via hidden state; Vanishing gradient problem.                         |
| 8       | Long Short-Term Memory (LSTM) LM     | Corpus (Sequences) / Sequence of words (indices)     | Probability distribution over vocab (per step)       | Gated RNN; Mitigates vanishing gradients; Better long-range memory.                         |
| 9       | *Attention Mechanism (Concept)*      | *(Sequence pairs or single sequence)*                | *(Contextualized representations / Alignment weights)* | *Concept: Allows focus on relevant parts regardless of distance; Enables Transformer.*        |
| 10      | Transformer LM                       | Corpus (Sequences) / Sequence of words (indices)     | Probability distribution over vocab (per step)       | Relies solely on self-attention; Parallelizable; Excellent long-range dependencies.           |
| 11      | Simplified GPT LM (Architecture)     | Corpus (Sequences) / Sequence of words (indices)     | Probability distribution over vocab (per step)       | Decoder-only Transformer; Autoregressive; Foundation for LLMs.                                |
| 12      | Advanced GPT Impls. (Architecture) | Corpus (Sequences) / Sequence of words (indices)     | Probability distribution over vocab (per step)       | Adds BPE, RAG concepts, etc.; Focus on scaling and efficiency.                              |
| 13      | *Fine-Tuning GPT Models (Process)*   | *(Pre-trained Model + Task-specific labeled data)*   | *(Task-specific output, e.g., classification, summary)* | *Process: Adapting large pre-trained models for specific tasks; Specialization.*              |

## Key Progressions Summary

*   **Context Handling:** Increasing ability to model longer and more complex dependencies (No context -> Fixed N -> Variable RNN state -> Global Attention).
*   **Word Representation:** Moving from discrete counts to learned continuous vector spaces (Embeddings).
*   **Sequence Modeling:** From processing fixed windows independently to handling variable-length sequences with state or global attention.
*   **Handling Sparsity/Generalization:** Development of smoothing techniques (Backoff) and architectures that generalize better (Neural Networks).
*   **Computational Paradigm:** Shift from sequential processing (RNNs) to highly parallelizable architectures (Transformers).
*   **Training Paradigm:** Emergence of large-scale pre-training followed by fine-tuning or prompting (GPT).

Each step built upon previous ideas, addressing limitations and enabling models to capture more intricate aspects of language, ultimately leading to the powerful foundation models we see today.