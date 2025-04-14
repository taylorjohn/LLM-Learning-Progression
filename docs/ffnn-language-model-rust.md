# Feed-Forward Neural Network Language Model in Rust

## Introduction

The Feed-Forward Neural Network Language Model (FFNNLM) is a significant step forward from n-gram models. Instead of explicitly counting sequences, it learns to predict the next word based on a fixed-size context window, using dense vector representations of words (embeddings) and non-linear transformations.

## How It Works

1. **Word Embeddings**: Each word is represented as a dense vector.
2. **Context Window**: A fixed number of previous words are used as context.
3. **Network Structure**: 
   - Input layer: concatenated embeddings of context words
   - Hidden layer(s): non-linear transformations of the input
   - Output layer: probability distribution over the vocabulary
4. **Training**: The network is trained to predict the next word given the context, using backpropagation and gradient descent.
5. **Generation**: Words are sampled from the predicted probability distribution.

## Implementation in Rust

For this implementation, we'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation. Note that this is a simplified version without some optimizations you'd use in a production setting.

[Link to `ffnn-language-model-rust_1.rs`](../src/ffnn-language-model-rust_1.rs)

## Explanation

1. The `FFNNLM` struct contains the model parameters: word embeddings, hidden layer weights, and output layer weights.
2. The `forward` method computes the probability distribution for the next word given a context.
3. The `train` method updates the model parameters using (a simplified version of) backpropagation.
4. The `generate` method produces new text by repeatedly sampling from the model's predictions.

## Advantages over N-gram Models

- Can generalize to unseen sequences more effectively
- Learns dense word representations that can capture semantic relationships
- Can potentially capture longer-range dependencies (though still limited by the fixed context size)

## Limitations

- Still uses a fixed context size
- Training can be slow and require a lot of data
- Prone to overfitting without proper regularization (not implemented in this simple version)

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, FFNNLMs often achieve lower perplexity than n-gram models, especially on larger datasets.

## Next Steps

The FFNNLM is a big step forward, but it still has limitations, particularly in handling variable-length sequences. The next major advancement would be to introduce recurrent connections, leading us to Recurrent Neural Networks (RNNs) and their variants like Long Short-Term Memory (LSTM) networks.