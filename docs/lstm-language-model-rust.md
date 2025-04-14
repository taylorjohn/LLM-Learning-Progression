# Long Short-Term Memory (LSTM) Language Model in Rust

## Introduction

While simple Recurrent Neural Networks (RNNs) can theoretically handle long sequences, they suffer from the **vanishing gradient problem**, making it difficult for them to learn dependencies between elements that are far apart in a sequence. **Long Short-Term Memory (LSTM)** networks were specifically designed by Hochreiter & Schmidhuber (1997) to overcome this limitation. They introduce a more complex internal structure involving **gates** that regulate the flow of information, allowing the network to selectively remember relevant information over long periods and forget irrelevant details.

## How It Works

The core idea of an LSTM unit is the **cell state** (\( c_t \)), often visualized as a "conveyor belt" running through the entire sequence. Information can be added to or removed from this cell state, regulated by specialized neural network layers called **gates**. LSTMs also maintain a **hidden state** (\( h_t \)), similar to a simple RNN, which is a filtered version of the cell state used for making predictions.

At each timestep \( t \), given the current input \( x_t \) and the previous hidden state \( h_{t-1} \) and cell state \( c_{t-1} \):

1.  **Forget Gate (\( f_t \)):** Decides what information to *throw away* from the previous cell state \( c_{t-1} \). It looks at \( h_{t-1} \) and \( x_t \) and outputs a number between 0 and 1 for each number in \( c_{t-1} \) (using a sigmoid function, \( \sigma \)). 1 means "completely keep this," while 0 means "completely get rid of this."
    \[ f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \]
2.  **Input Gate (\( i_t \)) & Candidate Values (\( \tilde{c}_t \)):** Decides what *new information* to store in the cell state. This has two parts:
    *   The **input gate layer** (sigmoid) decides which values we'll update: \( i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \).
    *   A **tanh layer** creates a vector of new candidate values, \( \tilde{c}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \), that *could* be added to the state.
3.  **Cell State Update (\( c_t \)):** Updates the old cell state \( c_{t-1} \) to the new cell state \( c_t \).
    *   First, multiply the old state by the forget gate values: \( c_{t-1} * f_t \) (pointwise multiplication).
    *   Then, add the new candidate values, scaled by the input gate values: \( i_t * \tilde{c}_t \).
    \[ c_t = f_t * c_{t-1} + i_t * \tilde{c}_t \]
4.  **Output Gate (\( o_t \)) & Hidden State (\( h_t \)):** Decides what part of the cell state to output as the hidden state \( h_t \).
    *   The **output gate layer** (sigmoid) decides which parts of the cell state we'll output: \( o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \).
    *   The cell state is put through \( \tanh \) (to push values between -1 and 1) and multiplied by the output gate's output:
    \[ h_t = o_t * \tanh(c_t) \]
    This \( h_t \) is then used to predict the next word (e.g., via a softmax layer) and is also passed to the next timestep.

*(Note: \( [h_{t-1}, x_t] \) denotes concatenation of the two vectors. W and b represent weight matrices and bias vectors for each gate/layer.)*

**Gated Recurrent Units (GRUs):** A simpler variant of LSTMs, introduced by Cho et al. (2014), combines the forget and input gates into a single "update gate" and merges the cell state and hidden state. GRUs often perform comparably to LSTMs on many tasks but have fewer parameters.

## Implementation in Rust

For this implementation, we'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation. Note that this is a simplified version of an LSTM, without some optimizations you'd use in a production setting.

[Link to `lstm-language-model-rust_1.rs`](../src/lstm-language-model-rust_1.rs)

## Explanation

1. The `LSTMCell` struct implements the core LSTM logic with forget, input, and output gates.
2. The `LSTMLanguageModel` struct wraps the LSTM cell with word embeddings and an output layer.
3. The `forward` method computes the next hidden state, cell state, and output probabilities given an input word and the previous states.
4. The `train` method updates the model parameters using a simplified version of Backpropagation Through Time (BPTT).
5. The `generate` method produces new text by repeatedly sampling from the model's predictions and updating the LSTM states.

## Advantages over Standard RNN

-   **Mitigation of Vanishing Gradients:** The gating mechanism allows gradients to flow more easily through time, making it possible to learn much longer-term dependencies. The cell state acts like an additive component, helping gradients propagate without repeated matrix multiplications causing them to vanish or explode as quickly.
-   **Selective Memory:** Explicitly designed to add or remove information, enabling better control over the hidden state.

## Limitations

-   **Complexity:** More complex than simple RNNs, with more parameters and computations per step.
-   **Still Sequential:** Processing remains sequential, limiting parallelization over time.
-   **Not Perfect Memory:** While much better than simple RNNs, LSTMs/GRUs can still eventually "forget" very distant information or struggle with extremely long dependencies compared to attention-based models.

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, LSTM language models often achieve lower perplexity than standard RNN models, especially on tasks requiring longer-range context.

## Next Steps

LSTMs and GRUs were the state-of-the-art for many sequence modeling tasks for several years. However, their sequential nature remained a bottleneck. The development of the **Attention Mechanism** (initially used *with* RNNs/LSTMs) and later the **Transformer architecture** (which relies solely on attention) provided ways to capture dependencies regardless of distance and allowed for much greater parallelization, leading to the next major leap in language modeling.