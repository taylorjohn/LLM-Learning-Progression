# Positional Encoding Improvements for Long Context

Transformers, lacking inherent recurrence or convolution, don't automatically know the order of tokens in a sequence. Positional Encodings (PE) are vectors added to token embeddings to inject this crucial information.

For long contexts, the choice of PE becomes critical for performance and the model's ability to understand relationships between distant tokens.

## Types of Positional Encodings

### 1. Absolute Positional Encodings

These assign a unique embedding based solely on the token's absolute position (index) in the sequence.

*   **Sinusoidal (Fixed):**
    *   Introduced in the original "Attention Is All You Need" paper.
    *   Uses sine and cosine functions of different frequencies:
        *   `PE(pos, 2i) = sin(pos / 10000^(2i / d_model))`
        *   `PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))`
    *   **Pros:** No learned parameters, can potentially extrapolate beyond training length (though performance may degrade).
    *   **Cons:** Fixed pattern, may not be optimal for all tasks, performance can degrade significantly beyond trained length.
    *   **Example Implementation:** A PyTorch implementation of sinusoidal embeddings can be found in `spring2024-assignment1-basics/cs336_basics/transformer.py` within this repository.

*   **Learned:**
    *   Treats positions like tokens, creating a learnable embedding vector for each position up to a maximum length.
    *   **Pros:** Can potentially learn more optimal position representations for the specific task/data.
    *   **Cons:** Requires learning parameters, strictly cannot extrapolate beyond the maximum learned length.

### 2. Relative Positional Encodings

These methods encode the *relative* distance or relationship between pairs of tokens, rather than just their absolute position. They are often favored for better generalization and handling long sequences.

*   **Rotary Positional Embedding (RoPE):**
    *   Popularized by models like Llama and PaLM.
    *   Applies rotations to the query and key vectors in the attention mechanism based on their absolute positions.
    *   The rotation implicitly encodes relative positional information within the dot product calculation of attention scores.
    *   **Pros:** Computationally efficient, good performance, naturally handles relative positions, some extrapolation capability.
    *   **Cons:** Can be slightly more complex to implement than absolute PEs.

*   **Attention with Linear Biases (ALiBi):**
    *   Introduced by Press et al.
    *   Does *not* add positional embeddings to the input tokens.
    *   Instead, it adds a static, non-learned bias to the attention scores based on the distance between the query and key tokens.
    *   The bias penalizes attention scores for distant tokens.
    *   **Pros:** Very simple, no parameters, excellent extrapolation capabilities demonstrated.
    *   **Cons:** The fixed bias might not be optimal for all tasks requiring complex long-range dependencies.

## Choosing for Long Context

*   **Absolute methods (Sinusoidal, Learned)** often struggle with significant extrapolation beyond their training length.
*   **Relative methods (RoPE, ALiBi)** generally show better performance and extrapolation for long contexts, making them preferred choices in modern LLMs designed for long sequences. 