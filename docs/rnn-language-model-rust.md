# Recurrent Neural Network (RNN) Language Model

## Introduction

Recurrent Neural Networks (RNNs) overcome the primary limitation of Feed-Forward Neural Network Language Models (FFNNLMs) – the fixed-size context window. RNNs are specifically designed to process **sequences** of data by introducing **recurrent connections**, which allow information from previous timesteps to persist and influence processing at the current timestep. This internal "memory" or **hidden state** enables RNNs to handle sequences of variable length and theoretically capture dependencies over longer distances in the text.

## How It Works

RNNs process input sequences one element (word) at a time. At each timestep \( t \):

1.  **Input**: The model takes the word embedding \( x_t \) for the current word \( w_t \).
2.  **Hidden State Update**: The core of the RNN lies in updating its hidden state \( h_t \). The new hidden state is calculated based on the *current input* \( x_t \) and the *previous hidden state* \( h_{t-1} \). A common formulation (Simple RNN) uses:
    \[ h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h) \]
    *   \( W_{xh} \) are the weights connecting input to the hidden layer.
    *   \( W_{hh} \) are the **recurrent weights** connecting the previous hidden state to the current hidden state (this forms the "loop" or recurrence).
    *   \( b_h \) is the hidden layer bias.
    *   \( \tanh \) (or other activation like ReLU) introduces non-linearity.
    The crucial aspect is that the *same* weight matrices (\( W_{xh}, W_{hh} \)) and bias (\( b_h \)) are used at **every timestep**, allowing the model to apply the same transformation logic across the sequence.
3.  **Output Prediction**: An output \( y_t \) (often the probability distribution for the *next* word \( w_{t+1} \)) is typically calculated based on the current hidden state:
    \[ y_t = \text{softmax}(W_{hy} h_t + b_y) \]
    *   \( W_{hy} \) are the weights connecting the hidden state to the output layer.
    *   \( b_y \) is the output layer bias.
4.  **Training (BPTT)**: RNNs are trained using **Backpropagation Through Time (BPTT)**. Conceptually, this involves:
    *   "Unrolling" the RNN over the input sequence length, creating a deep feed-forward network where each layer corresponds to a timestep.
    *   Calculating the loss (e.g., Cross-Entropy) based on the outputs at each timestep (or just the final output, depending on the task).
    *   Calculating gradients by backpropagating the error through the unrolled network.
    *   Crucially, the gradients for the shared weights (\( W_{xh}, W_{hh}, W_{hy} \)) are summed or averaged across all timesteps.
5.  **Generation**: Start with an initial hidden state \( h_0 \) (often zeros) and a starting word/token. Feed the word's embedding \( x_1 \) to get \( h_1 \) and predict the next word \( w_2 \) from \( y_1 \). Feed \( w_2 \)'s embedding \( x_2 \) and \( h_1 \) to get \( h_2 \) and predict \( w_3 \), and so on.

## Implementation in Rust

For this implementation, we'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation. Note that this is a simplified version of an RNN, without some optimizations you'd use in a production setting.

[Link to `rnn-language-model-rust_1.rs`](../src/rnn-language-model-rust_1.rs)

## Explanation

1. The `RNNLanguageModel` struct contains the model parameters: word embeddings, recurrent weights, and biases.
2. The `forward` method computes the next hidden state and output probabilities given an input word and the previous hidden state.
3. The `train` method updates the model parameters using a simplified version of Backpropagation Through Time (BPTT).
4. The `generate` method produces new text by repeatedly sampling from the model's predictions and updating the hidden state.

## Advantages over FFNNLM

-   **Variable Length Input:** Can process sequences of any length without a fixed window.
-   **Theoretical Long-Range Dependencies:** The hidden state *can* potentially carry information across many timesteps.
-   **Parameter Sharing:** Weights are shared across timesteps, making the model more parameter-efficient than an FFNN with a very large window.

## Limitations

-   **Vanishing/Exploding Gradients:** The primary practical limitation. During BPTT, gradients are propagated backward through time. If the recurrent weight matrix (\( W_{hh} \)) components are consistently small (<1), gradients can shrink exponentially (**vanish**), preventing the model from learning long-range dependencies (error signals from the future don't reach the distant past). If they are consistently large (>1), gradients can grow exponentially (**explode**), destabilizing training. Gradient clipping can mitigate explosion, but vanishing gradients are harder to solve in simple RNNs.
-   **Difficulty with Very Long Dependencies:** Even without severe vanishing gradients, simple RNNs struggle to effectively retain information over very long sequences.
-   **Sequential Computation:** Processing is inherently sequential, making parallelization across the time dimension difficult (unlike Transformers).

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, RNN language models often achieve lower perplexity than FFNN models, especially on tasks requiring longer-range context.

## Next Steps

The vanishing gradient problem severely limits the practical effectiveness of simple RNNs for capturing long dependencies. To address this, more sophisticated recurrent units with **gating mechanisms** were developed, namely **Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRUs)**. These will be explored next.