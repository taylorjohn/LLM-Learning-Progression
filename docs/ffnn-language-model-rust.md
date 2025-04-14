# Feed-Forward Neural Network Language Model (FFNNLM)

## Introduction

The Feed-Forward Neural Network Language Model (FFNNLM) marks a departure from count-based n-gram models. It leverages **neural networks** and **word embeddings** (as discussed in `docs/word-embeddings.md`) to learn relationships between words and predict the next word in a sequence. Unlike n-grams which memorize sequences, FFNNLMs aim to *generalize* based on learned features of the context words.

## How It Works

The core idea is to predict the probability of the next word \( w_t \) given a fixed window of preceding words \( (w_{t-n+1}, \dots, w_{t-1}) \).

1.  **Input Representation**: Each word in the context window (size \( n-1 \)) is looked up in an **Embedding Matrix (E)** to get its corresponding dense vector representation.
2.  **Concatenation**: The embedding vectors for the \( n-1 \) context words are concatenated together to form a single large input vector \( x \).
3.  **Network Structure**: This input vector \( x \) is fed through a standard feed-forward neural network:
    *   **Input Layer:** Represents the concatenated context embeddings \( x \).
    *   **Hidden Layer:** Computes \( h = \tanh(W_h x + b_h) \), where \( W_h, b_h \) are weights and biases, and \( \tanh \) (or ReLU, sigmoid) is a non-linear activation function. There might be multiple hidden layers.
    *   **Output Layer:** Computes scores for each word in the vocabulary: \( \text{scores} = W_o h + b_o \).
    *   **Softmax Layer:** Converts the output scores into a probability distribution over the entire vocabulary: \( P(w_t | w_{t-n+1}, \dots, w_{t-1}) = \text{softmax}(\text{scores}) \). The word with the highest probability is the model's prediction.
4.  **Training**:
    *   The **Embedding Matrix (E)** and the network weights (\( W_h, b_h, W_o, b_o \)) are the model's parameters, typically initialized randomly.
    *   The model is trained using **backpropagation** and optimization algorithms (like Stochastic Gradient Descent or Adam) to minimize a **loss function**. The standard loss for language modeling is **Cross-Entropy Loss** between the predicted probability distribution and the actual next word (represented as a one-hot vector).
5.  **Generation**: Similar to n-grams, generation involves feeding the last \( n-1 \) words as context, getting the probability distribution for the next word via a forward pass, sampling a word from this distribution, and sliding the context window forward.

## Implementation in Rust

For this implementation, we'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation. Note that this is a simplified version without some optimizations you'd use in a production setting.

[Link to `ffnn-language-model-rust_1.rs`](../src/ffnn-language-model-rust_1.rs)

## Explanation

1. The `FFNNLM` struct contains the model parameters: word embeddings, hidden layer weights, and output layer weights.
2. The `forward` method computes the probability distribution for the next word given a context.
3. The `train` method updates the model parameters using (a simplified version of) backpropagation.
4. The `generate` method produces new text by repeatedly sampling from the model's predictions.

## Advantages over N-gram Models

-   **Generalization:** By using embeddings, the model can handle unseen n-grams if the individual words have been seen (or if embeddings capture similarity). If the model knows "cat drinks milk" is likely, it might infer "dog drinks water" is also somewhat likely if "cat"/"dog" and "milk"/"water" have similar embeddings. N-grams treat these as completely unrelated if the specific sequence wasn't seen.
-   **Shared Representations:** Word embeddings learned by the model capture semantic/syntactic similarities.
-   **No Explicit Smoothing Needed:** The distributed representation and neural network structure provide inherent smoothing.

## Limitations

-   **Fixed Context Size:** The biggest limitation. The model can only look back \( n-1 \) words, regardless of how long the true dependency might be. Processing time increases with context size.
-   **Independent Computations:** Each context window is processed independently; no state is carried over, unlike RNNs.
-   **Training Complexity:** Neural network training is computationally more intensive than counting n-grams.

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, FFNNLMs often achieve lower perplexity than n-gram models, especially on larger datasets.

## Next Steps

The fixed context window is the primary drawback of FFNNLMs. To handle sequences of arbitrary length and capture longer dependencies more effectively, we need models that maintain some form of "memory" or state as they process the sequence. This leads us to **Recurrent Neural Networks (RNNs)**.