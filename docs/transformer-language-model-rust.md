# Transformer Language Model

## Introduction

The Transformer model, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), revolutionized sequence modeling and forms the basis of most modern Large Language Models (LLMs). It discards the recurrent connections of RNNs/LSTMs entirely and relies solely on **attention mechanisms** (specifically **self-attention**, explained in detail in `docs/attention-mechanism.md`) to model dependencies between words in a sequence. This allows for significantly more parallelization during training and has proven exceptionally effective at capturing long-range dependencies.

While the original Transformer was designed for sequence-to-sequence tasks (like machine translation) and had both an **Encoder** (to process the input sequence) and a **Decoder** (to generate the output sequence), many successful language models (like GPT) utilize only the **Decoder** part of the architecture in a stack.

## How It Works (Decoder-Only Perspective)

A Transformer language model typically consists of a stack of identical **Decoder blocks**. Each block processes the input sequence and passes its output to the next block. The key components within each block are:

1.  **Input Embedding + Positional Encoding**:
    *   Input words are converted to dense vectors using an **Embedding** layer.
    *   Since there's no recurrence, **Positional Encodings** (often fixed sinusoidal functions or learned embeddings) are *added* to the word embeddings to give the model information about the position of each word in the sequence.
2.  **(Masked) Multi-Head Self-Attention**:
    *   This is the core of the Transformer. Each word attends to other words in the sequence to compute a context-aware representation.
    *   **Self-Attention** typically uses the **Scaled Dot-Product Attention** mechanism: \( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \), where Q (Query), K (Key), and V (Value) are learned linear projections of the input embeddings (plus positional encodings). The scaling factor \( \sqrt{d_k} \) (where \( d_k \) is the dimension of keys/queries) stabilizes gradients.
    *   **Multi-Head** attention applies the attention mechanism multiple times in parallel with different learned projections (Q, K, V weights), capturing different types of relationships. The results are concatenated and projected back.
    *   For language modeling (predicting the next word), **Masking** is applied to the attention scores. This prevents a position from attending to *future* positions during training, ensuring the model remains causal (autoregressive).
3.  **Add & Norm (Residual Connection + Layer Normalization)**:
    *   **Residual Connection:** The input to the attention layer is added to its output (\( x + \text{Attention}(x) \)). This helps gradients flow directly through the network during backpropagation, enabling much deeper models.
    *   **Layer Normalization:** Normalizes the activations across the feature dimension for each sequence element independently. This stabilizes training dynamics.
4.  **Position-wise Feed-Forward Network (FFN)**:
    *   A standard two-layer feed-forward network (e.g., Linear -> ReLU -> Linear) is applied independently to *each position* in the sequence after the attention layer. This provides additional non-linearity and processing capacity.
5.  **Add & Norm**: Another residual connection and layer normalization step are applied after the FFN.
6.  **Final Output Layer**: After the last Decoder block, a final linear layer followed by a softmax converts the processed representations into probabilities over the vocabulary for predicting the next word.

## Implementation in Rust

For this implementation, we'll create a simplified Transformer model focusing on the key concept of self-attention. We'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation.

[Link to `transformer-language-model-rust_1.rs`](../src/transformer-language-model-rust_1.rs)

## Explanation

1. The `TransformerLanguageModel` struct contains the model parameters: embeddings, positional encodings, attention weights, feed-forward weights, and output weights.
2. The `get_positional_encoding` method creates sinusoidal positional encodings.
3. The `attention` method implements the core self-attention mechanism.
4. The `forward` method applies the full Transformer layer stack to the input sequence.
5. The `train` method updates the model parameters (note that the actual backpropagation is omitted for brevity).
6. The `generate` method produces new text by repeatedly applying the model to the growing sequence.

## Advantages over RNN/LSTM Models

-   **Superior Long-Range Dependency Handling:** Attention allows direct connections between any two positions in the sequence, overcoming the sequential bottleneck of RNNs. Path length is O(1).
-   **Parallelization:** Computations within each layer (attention, FFN) can be performed in parallel across the sequence dimension, leading to much faster training on parallel hardware (GPUs/TPUs).
-   **State-of-the-Art Performance:** Transformers have become the dominant architecture for a wide range of NLP tasks, achieving top results.

## Limitations

-   **Quadratic Complexity (Self-Attention):** The computational and memory cost of standard self-attention scales quadratically with sequence length (O(N<sup>2</sup>)), making it very expensive for extremely long sequences. (Many efficient variants exist now).
-   **Positional Encoding:** Relies on explicit positional encodings, which might be less natural than the inherent sequential processing of RNNs for some tasks.
-   **Less Interpretability (compared to simpler models):** While attention weights can offer some insight, understanding the complex interactions in deep Transformers can be challenging.

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, Transformer language models often achieve state-of-the-art perplexity on various language modeling benchmarks.

## Next Steps

The Transformer architecture, particularly the decoder-only variant, serves as the direct foundation for the **Generative Pre-trained Transformer (GPT)** models, which are the next stage in this progression. GPT models leverage the power of Transformers, pre-train them on massive text datasets, and demonstrate remarkable few-shot and zero-shot learning capabilities.