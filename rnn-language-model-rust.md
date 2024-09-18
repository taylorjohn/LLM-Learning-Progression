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

```rust
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::collections::HashMap;

struct RNNLanguageModel {
    vocab: Vec<String>,
    word_to_index: HashMap<String, usize>,
    embedding_dim: usize,
    hidden_dim: usize,
    embeddings: Array2<f32>,
    W_xh: Array2<f32>,  // Input to hidden weights
    W_hh: Array2<f32>,  // Hidden to hidden weights
    W_hy: Array2<f32>,  // Hidden to output weights
    b_h: Array1<f32>,   // Hidden bias
    b_y: Array1<f32>,   // Output bias
}

impl RNNLanguageModel {
    fn new(vocab: Vec<String>, embedding_dim: usize, hidden_dim: usize) -> Self {
        let vocab_size = vocab.len();
        let word_to_index: HashMap<_, _> = vocab.iter().cloned().enumerate().map(|(i, w)| (w, i)).collect();
        
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(-0.1, 0.1);
        
        RNNLanguageModel {
            vocab,
            word_to_index,
            embedding_dim,
            hidden_dim,
            embeddings: Array2::random((vocab_size, embedding_dim), uniform),
            W_xh: Array2::random((hidden_dim, embedding_dim), uniform),
            W_hh: Array2::random((hidden_dim, hidden_dim), uniform),
            W_hy: Array2::random((vocab_size, hidden_dim), uniform),
            b_h: Array1::zeros(hidden_dim),
            b_y: Array1::zeros(vocab_size),
        }
    }
    
    fn forward(&self, input: usize, hidden: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let x = self.embeddings.row(input).to_owned();
        let hidden = self.W_xh.dot(&x) + self.W_hh.dot(hidden) + &self.b_h;
        let hidden = hidden.mapv(|x| x.tanh());
        let output = self.W_hy.dot(&hidden) + &self.b_y;
        
        let max = output.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = output.mapv(|x| (x - max).exp());
        let sum = exp.sum();
        let output = exp / sum;
        
        (output, hidden)
    }
    
    fn train(&mut self, text: &str, epochs: usize, learning_rate: f32) {
        let words: Vec<usize> = text.split_whitespace()
            .map(|w| *self.word_to_index.get(w).unwrap_or(&0))
            .collect();
        
        for _ in 0..epochs {
            let mut hidden = Array1::zeros(self.hidden_dim);
            let mut total_loss = 0.0;
            
            for i in 0..words.len() - 1 {
                let (output, new_hidden) = self.forward(words[i], &hidden);
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
            }
            
            println!("Epoch loss: {}", total_loss / words.len() as f32);
        }
    }
    
    fn generate(&self, start_word: &str, num_words: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut words = vec![start_word.to_string()];
        let mut hidden = Array1::zeros(self.hidden_dim);
        let mut current_word = *self.word_to_index.get(start_word).unwrap_or(&0);
        
        for _ in 0..num_words {
            let (output, new_hidden) = self.forward(current_word, &hidden);
            let next_word_idx = (0..output.len())
                .max_by(|&i, &j| output[i].partial_cmp(&output[j]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            words.push(self.vocab[next_word_idx].clone());
            current_word = next_word_idx;
            hidden = new_hidden;
        }
        
        words.join(" ")
    }
}

fn main() {
    let vocab = "to be or not that is the question".split_whitespace().map(String::from).collect();
    let mut model = RNNLanguageModel::new(vocab, 10, 20);
    
    let corpus = "to be or not to be that is the question";
    model.train(corpus, 1000, 0.1);
    
    let generated = model.generate("to", 10);
    println!("Generated: {}", generated);
}
```

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