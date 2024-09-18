# Long Short-Term Memory (LSTM) Language Model in Rust

## Introduction

The Long Short-Term Memory (LSTM) Network is an advanced recurrent neural network architecture designed to better capture long-term dependencies in sequence data. LSTMs introduce a more complex structure of gates within each recurrent unit, allowing the network to selectively remember or forget information over long sequences.

## How It Works

1. **Word Embeddings**: Each word is represented as a dense vector, similar to previous models.
2. **LSTM Cell**: The core of the LSTM is its cell state and three gates:
   - Forget gate: decides what information to discard from the cell state
   - Input gate: decides what new information to store in the cell state
   - Output gate: decides what to output based on the cell state
3. **Network Structure**: 
   - Input layer: word embedding of the current word
   - LSTM layer: processes the input and updates its cell state and hidden state
   - Output layer: probability distribution over the vocabulary
4. **Training**: The network is trained using Backpropagation Through Time (BPTT), similar to standard RNNs but with more complex gradient flow through the LSTM cell.
5. **Generation**: Words are sampled from the predicted probability distribution, and the LSTM state is updated at each step.

## Implementation in Rust

For this implementation, we'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation. Note that this is a simplified version of an LSTM, without some optimizations you'd use in a production setting.

```rust
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::collections::HashMap;

struct LSTMCell {
    input_dim: usize,
    hidden_dim: usize,
    W_fi: Array2<f32>,  // Forget gate weights (input)
    W_fh: Array2<f32>,  // Forget gate weights (hidden)
    b_f: Array1<f32>,   // Forget gate bias
    W_ii: Array2<f32>,  // Input gate weights (input)
    W_ih: Array2<f32>,  // Input gate weights (hidden)
    b_i: Array1<f32>,   // Input gate bias
    W_oi: Array2<f32>,  // Output gate weights (input)
    W_oh: Array2<f32>,  // Output gate weights (hidden)
    b_o: Array1<f32>,   // Output gate bias
    W_ci: Array2<f32>,  // Cell state weights (input)
    W_ch: Array2<f32>,  // Cell state weights (hidden)
    b_c: Array1<f32>,   // Cell state bias
}

impl LSTMCell {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(-0.1, 0.1);
        
        LSTMCell {
            input_dim,
            hidden_dim,
            W_fi: Array2::random((hidden_dim, input_dim), uniform),
            W_fh: Array2::random((hidden_dim, hidden_dim), uniform),
            b_f: Array1::zeros(hidden_dim),
            W_ii: Array2::random((hidden_dim, input_dim), uniform),
            W_ih: Array2::random((hidden_dim, hidden_dim), uniform),
            b_i: Array1::zeros(hidden_dim),
            W_oi: Array2::random((hidden_dim, input_dim), uniform),
            W_oh: Array2::random((hidden_dim, hidden_dim), uniform),
            b_o: Array1::zeros(hidden_dim),
            W_ci: Array2::random((hidden_dim, input_dim), uniform),
            W_ch: Array2::random((hidden_dim, hidden_dim), uniform),
            b_c: Array1::zeros(hidden_dim),
        }
    }
    
    fn forward(&self, input: &Array1<f32>, hidden: &Array1<f32>, cell: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let forget_gate = (self.W_fi.dot(input) + self.W_fh.dot(hidden) + &self.b_f).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let input_gate = (self.W_ii.dot(input) + self.W_ih.dot(hidden) + &self.b_i).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let output_gate = (self.W_oi.dot(input) + self.W_oh.dot(hidden) + &self.b_o).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let cell_candidate = (self.W_ci.dot(input) + self.W_ch.dot(hidden) + &self.b_c).mapv(|x| x.tanh());
        
        let cell_new = &forget_gate * cell + &input_gate * &cell_candidate;
        let hidden_new = &output_gate * cell_new.mapv(|x| x.tanh());
        
        (hidden_new, cell_new)
    }
}

struct LSTMLanguageModel {
    vocab: Vec<String>,
    word_to_index: HashMap<String, usize>,
    embedding_dim: usize,
    hidden_dim: usize,
    embeddings: Array2<f32>,
    lstm: LSTMCell,
    W_hy: Array2<f32>,  // Hidden to output weights
    b_y: Array1<f32>,   // Output bias
}

impl LSTMLanguageModel {
    fn new(vocab: Vec<String>, embedding_dim: usize, hidden_dim: usize) -> Self {
        let vocab_size = vocab.len();
        let word_to_index: HashMap<_, _> = vocab.iter().cloned().enumerate().map(|(i, w)| (w, i)).collect();
        
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(-0.1, 0.1);
        
        LSTMLanguageModel {
            vocab,
            word_to_index,
            embedding_dim,
            hidden_dim,
            embeddings: Array2::random((vocab_size, embedding_dim), uniform),
            lstm: LSTMCell::new(embedding_dim, hidden_dim),
            W_hy: Array2::random((vocab_size, hidden_dim), uniform),
            b_y: Array1::zeros(vocab_size),
        }
    }
    
    fn forward(&self, input: usize, hidden: &Array1<f32>, cell: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        let x = self.embeddings.row(input).to_owned();
        let (hidden_new, cell_new) = self.lstm.forward(&x, hidden, cell);
        let output = self.W_hy.dot(&hidden_new) + &self.b_y;
        
        let max = output.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = output.mapv(|x| (x - max).exp());
        let sum = exp.sum();
        let output = exp / sum;
        
        (output, hidden_new, cell_new)
    }
    
    fn train(&mut self, text: &str, epochs: usize, learning_rate: f32) {
        let words: Vec<usize> = text.split_whitespace()
            .map(|w| *self.word_to_index.get(w).unwrap_or(&0))
            .collect();
        
        for epoch in 0..epochs {
            let mut hidden = Array1::zeros(self.hidden_dim);
            let mut cell = Array1::zeros(self.hidden_dim);
            let mut total_loss = 0.0;
            
            for i in 0..words.len() - 1 {
                let (output, new_hidden, new_cell) = self.forward(words[i], &hidden, &cell);
                let target = words[i + 1];
                
                // Compute cross-entropy loss
                let loss = -output[target].ln();
                total_loss += loss;
                
                // Backpropagation (simplified)
                let mut d_output = output;
                d_output[target] -= 1.0;
                
                // Update weights (simplified, without proper BPTT)
                self.W_hy -= &(learning_rate * d_output.outer(&hidden));
                self.b_y -= &(learning_rate * d_output);
                
                // Update other weights similarly (omitted for brevity)
                
                hidden = new_hidden;
                cell = new_cell;
            }
            
            println!("Epoch {}: loss = {}", epoch, total_loss / words.len() as f32);
        }
    }
    
    fn generate(&self, start_word: &str, num_words: usize) -> String {
        let mut words = vec![start_word.to_string()];
        let mut hidden = Array1::zeros(self.hidden_dim);
        let mut cell = Array1::zeros(self.hidden_dim);
        let mut current_word = *self.word_to_index.get(start_word).unwrap_or(&0);
        
        for _ in 0..num_words {
            let (output, new_hidden, new_cell) = self.forward(current_word, &hidden, &cell);
            let next_word_idx = (0..output.len())
                .max_by(|&i, &j| output[i].partial_cmp(&output[j]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            words.push(self.vocab[next_word_idx].clone());
            current_word = next_word_idx;
            hidden = new_hidden;
            cell = new_cell;
        }
        
        words.join(" ")
    }
}

fn main() {
    let vocab = "to be or not that is the question".split_whitespace().map(String::from).collect();
    let mut model = LSTMLanguageModel::new(vocab, 10, 20);
    
    let corpus = "to be or not to be that is the question";
    model.train(corpus, 1000, 0.1);
    
    let generated = model.generate("to", 10);
    println!("Generated: {}", generated);
}
```

## Explanation

1. The `LSTMCell` struct implements the core LSTM logic with forget, input, and output gates.
2. The `LSTMLanguageModel` struct wraps the LSTM cell with word embeddings and an output layer.
3. The `forward` method computes the next hidden state, cell state, and output probabilities given an input word and the previous states.
4. The `train` method updates the model parameters using a simplified version of Backpropagation Through Time (BPTT).
5. The `generate` method produces new text by repeatedly sampling from the model's predictions and updating the LSTM states.

## Advantages over Standard RNN

- Better at capturing long-term dependencies in the text
- More stable gradient flow during training, mitigating vanishing/exploding gradient problems
- Ability to selectively remember or forget information, leading to more flexible learning

## Limitations

- More complex architecture with more parameters, potentially requiring more data and computational resources
- This simple implementation doesn't include more advanced techniques like gradient clipping, proper BPTT, or regularization
- Still may struggle with very long sequences or capturing global document structure

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, LSTM language models often achieve lower perplexity than standard RNN models, especially on tasks requiring longer-range context.

## Next Steps

LSTMs represent a significant advancement in sequence modeling, but there are still further improvements to be made. Some potential next steps could include:

1. Implementing bidirectional LSTMs to capture both past and future context
2. Exploring attention mechanisms to allow the model to focus on different parts of the input sequence
3. Moving towards transformer-based architectures, which have largely supplanted RNNs in many NLP tasks