# Transformer Language Model in Rust

## Introduction

The Transformer model, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), represents a significant shift in approach to sequence modeling. Unlike RNNs and LSTMs, Transformers rely entirely on attention mechanisms, allowing them to capture long-range dependencies more effectively and to be trained more efficiently in parallel.

## How It Works

1. **Positional Encoding**: Since Transformers don't use recurrence, positional information is added to the input embeddings.
2. **Self-Attention**: The core mechanism that allows the model to weigh the importance of different words in the input when producing each output word.
3. **Multi-Head Attention**: Multiple attention mechanisms run in parallel, allowing the model to focus on different aspects of the input.
4. **Feed-Forward Networks**: Applied to each position separately and identically.
5. **Layer Normalization and Residual Connections**: Used to stabilize the network and allow for deeper architectures.

## Implementation in Rust

For this implementation, we'll create a simplified Transformer model focusing on the key concept of self-attention. We'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation.

```rust
use ndarray::{Array, Array1, Array2, Array3, Axis};
use rand::Rng;
use std::collections::HashMap;
use std::f32::consts::PI;

struct TransformerLanguageModel {
    vocab: Vec<String>,
    word_to_index: HashMap<String, usize>,
    embedding_dim: usize,
    num_heads: usize,
    num_layers: usize,
    max_seq_length: usize,
    embeddings: Array2<f32>,
    positional_encoding: Array2<f32>,
    attention_weights: Vec<Array3<f32>>,
    ff_weights1: Vec<Array2<f32>>,
    ff_weights2: Vec<Array2<f32>>,
    output_weights: Array2<f32>,
}

impl TransformerLanguageModel {
    fn new(vocab: Vec<String>, embedding_dim: usize, num_heads: usize, num_layers: usize, max_seq_length: usize) -> Self {
        let vocab_size = vocab.len();
        let word_to_index: HashMap<_, _> = vocab.iter().cloned().enumerate().map(|(i, w)| (w, i)).collect();
        
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(-0.1, 0.1);
        
        let embeddings = Array2::random((vocab_size, embedding_dim), uniform);
        let positional_encoding = Self::get_positional_encoding(max_seq_length, embedding_dim);
        
        let attention_weights = (0..num_layers)
            .map(|_| Array3::random((num_heads, embedding_dim, embedding_dim), uniform))
            .collect();
        
        let ff_weights1 = (0..num_layers)
            .map(|_| Array2::random((embedding_dim * 4, embedding_dim), uniform))
            .collect();
        
        let ff_weights2 = (0..num_layers)
            .map(|_| Array2::random((embedding_dim, embedding_dim * 4), uniform))
            .collect();
        
        let output_weights = Array2::random((vocab_size, embedding_dim), uniform);
        
        TransformerLanguageModel {
            vocab,
            word_to_index,
            embedding_dim,
            num_heads,
            num_layers,
            max_seq_length,
            embeddings,
            positional_encoding,
            attention_weights,
            ff_weights1,
            ff_weights2,
            output_weights,
        }
    }
    
    fn get_positional_encoding(max_len: usize, d_model: usize) -> Array2<f32> {
        let mut pos_encoding = Array2::zeros((max_len, d_model));
        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / f32::powf(10000.0, (2 * i) as f32 / d_model as f32);
                pos_encoding[[pos, i]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        pos_encoding
    }
    
    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let d_k = k.shape()[1] as f32;
        let attention_scores = q.dot(&k.t()) / d_k.sqrt();
        let attention_weights = attention_scores.mapv(|x| x.exp()) / attention_scores.mapv(|x| x.exp()).sum_axis(Axis(1)).insert_axis(Axis(1));
        attention_weights.dot(v)
    }
    
    fn forward(&self, input_ids: &[usize]) -> Array2<f32> {
        let seq_len = input_ids.len();
        let mut x = Array2::zeros((seq_len, self.embedding_dim));
        for (i, &id) in input_ids.iter().enumerate() {
            x.row_mut(i).assign(&self.embeddings.row(id));
        }
        x += &self.positional_encoding.slice(s![..seq_len, ..]);
        
        for layer in 0..self.num_layers {
            let mut attention_output = Array2::zeros((seq_len, self.embedding_dim));
            for head in 0..self.num_heads {
                let q = x.dot(&self.attention_weights[layer].slice(s![head, .., ..]));
                let k = x.dot(&self.attention_weights[layer].slice(s![head, .., ..]));
                let v = x.dot(&self.attention_weights[layer].slice(s![head, .., ..]));
                attention_output += &self.attention(&q, &k, &v);
            }
            x = &x + &attention_output;
            
            let ff_output = x.dot(&self.ff_weights1[layer]).mapv(|x| x.max(0.0)).dot(&self.ff_weights2[layer]);
            x = &x + &ff_output;
        }
        
        x
    }
    
    fn train(&mut self, text: &str, epochs: usize, learning_rate: f32) {
        let words: Vec<usize> = text.split_whitespace()
            .map(|w| *self.word_to_index.get(w).unwrap_or(&0))
            .collect();
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for i in 0..words.len() - 1 {
                let input_sequence = &words[..=i];
                let target = words[i + 1];
                
                let output = self.forward(input_sequence);
                let logits = output.dot(&self.output_weights.t());
                
                let mut loss = 0.0;
                for (j, &logit) in logits.iter().enumerate() {
                    if j == target {
                        loss -= logit;
                    }
                    loss += logit.exp();
                }
                loss = loss.ln();
                
                total_loss += loss;
                
                // Backpropagation and weight updates would go here
                // This is a complex process and is omitted for brevity
            }
            
            println!("Epoch {}: loss = {}", epoch, total_loss / words.len() as f32);
        }
    }
    
    fn generate(&self, start_word: &str, num_words: usize) -> String {
        let mut words = vec![start_word.to_string()];
        let mut current_word = *self.word_to_index.get(start_word).unwrap_or(&0);
        
        for _ in 0..num_words {
            let input_sequence: Vec<usize> = words.iter()
                .map(|w| *self.word_to_index.get(w).unwrap_or(&0))
                .collect();
            
            let output = self.forward(&input_sequence);
            let logits = output.dot(&self.output_weights.t());
            
            let next_word_idx = logits.iter().enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();
            
            words.push(self.vocab[next_word_idx].clone());
            current_word = next_word_idx;
        }
        
        words.join(" ")
    }
}

fn main() {
    let vocab = "to be or not that is the question".split_whitespace().map(String::from).collect();
    let mut model = TransformerLanguageModel::new(vocab, 16, 2, 2, 50);
    
    let corpus = "to be or not to be that is the question";
    model.train(corpus, 100, 0.01);
    
    let generated = model.generate("to", 10);
    println!("Generated: {}", generated);
}
```

## Explanation

1. The `TransformerLanguageModel` struct contains the model parameters: embeddings, positional encodings, attention weights, feed-forward weights, and output weights.
2. The `get_positional_encoding` method creates sinusoidal positional encodings.
3. The `attention` method implements the core self-attention mechanism.
4. The `forward` method applies the full Transformer layer stack to the input sequence.
5. The `train` method updates the model parameters (note that the actual backpropagation is omitted for brevity).
6. The `generate` method produces new text by repeatedly applying the model to the growing sequence.

## Advantages over RNN/LSTM Models

- Can capture long-range dependencies more effectively
- Parallelizable, allowing for more efficient training on modern hardware
- No vanishing gradient problem due to direct connections between any two positions
- Can generate output in a non-autoregressive manner (though not implemented in this simple version)

## Limitations

- This implementation is greatly simplified and lacks many optimizations and techniques used in state-of-the-art Transformers
- Requires more memory for long sequences due to attention over all positions
- May struggle with very long documents or tasks requiring hierarchical understanding

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, Transformer language models often achieve state-of-the-art perplexity on various language modeling benchmarks.

## Next Steps

This Transformer model represents the foundation of many state-of-the-art NLP models. Some potential next steps could include:

1. Implementing more advanced Transformer variants like GPT (for unidirectional language modeling) or BERT (for bidirectional encoding)
2. Exploring techniques for handling longer sequences, such as Transformer-XL or Longformer
3. Investigating methods for more efficient training and inference, such as mixed-precision training or quantization