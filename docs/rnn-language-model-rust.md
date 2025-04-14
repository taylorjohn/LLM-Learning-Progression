# Recurrent Neural Network Language Model in Rust

## Introduction

The Recurrent Neural Network (RNN) Language Model is a significant advancement over the Feed-Forward Neural Network model. RNNs introduce the concept of memory, allowing the model to process sequences of variable length and potentially capture long-range dependencies in the text.

## How It Works

1. **Word Embeddings**: Each word is represented as a dense vector, similar to the FFNN model.
2. **Recurrent Structure**: Unlike FFNN, RNNs process input sequences one element at a time, maintaining a hidden state that's updated at each step.
3. **Network Structure**: 
   - Input layer: word embedding of the current word
   - Hidden layer: combines the current input with the previous hidden state
   - Output layer: probability distribution over the vocabulary
4. **Training**: The network is trained using Backpropagation Through Time (BPTT), a variant of backpropagation for sequence data.
5. **Generation**: Words are sampled from the predicted probability distribution, and the hidden state is updated at each step.

## Implementation in Rust

For this implementation, we'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation. Note that this is a simplified version of an RNN, without some optimizations you'd use in a production setting.

[Link to `rnn-language-model-rust_1.rs`](../src/rnn-language-model-rust_1.rs)

## Explanation

1. The `RNNLanguageModel` struct contains the model parameters: word embeddings, recurrent weights, and biases.
2. The `forward` method computes the next hidden state and output probabilities given an input word and the previous hidden state.
3. The `train` method updates the model parameters using a simplified version of Backpropagation Through Time (BPTT).
4. The `generate` method produces new text by repeatedly sampling from the model's predictions and updating the hidden state.

## Advantages over Feed-Forward Neural Network

- Can handle variable-length sequences naturally
- Potentially captures longer-range dependencies in the text
- Shared weights across time steps, leading to more efficient parameter usage

## Limitations

- Still struggles with very long-range dependencies due to vanishing/exploding gradients
- Training can be unstable and sensitive to hyperparameters
- This simple implementation doesn't include more advanced techniques like gradient clipping or proper BPTT

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, RNN language models often achieve lower perplexity than FFNN models, especially on tasks requiring longer-range context.

## Next Steps

While RNNs are a significant improvement, they still struggle with long-range dependencies. The next major advancement would be to introduce more sophisticated recurrent architectures, such as Long Short-Term Memory (LSTM) networks or Gated Recurrent Units (GRUs), which are designed to better handle long-range dependencies.