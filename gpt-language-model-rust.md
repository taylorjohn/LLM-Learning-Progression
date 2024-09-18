# Simplified GPT (Generative Pre-trained Transformer) Language Model in Rust

## Introduction

GPT (Generative Pre-trained Transformer) is a family of large language models that have achieved state-of-the-art results on many NLP tasks. GPT uses a decoder-only Transformer architecture and is trained on a vast amount of text data in an unsupervised manner. Our implementation will be a greatly simplified version, focusing on the key architectural elements that make GPT unique.

## How It Works

1. **Tokenization**: Text is broken down into tokens (in our simple version, we'll use words as tokens).
2. **Embeddings**: Each token is converted to a dense vector representation.
3. **Positional Encoding**: Position information is added to the embeddings.
4. **Multi-Layer Transformer Decoder**: A stack of Transformer decoder layers processes the input.
5. **Language Modeling Head**: The final layer predicts the next token in the sequence.

## Implementation in Rust

This implementation is a simplified version of GPT, focusing on the core architectural elements. We'll use the `ndarray` crate for matrix operations and the `rand` crate for random number generation.

```rust
use ndarray::{Array, Array1, Array2, Array3, Axis};
use rand::Rng;
use std::collections::HashMap;

struct GPTLanguageModel {
    vocab: Vec<String>,
    word_to_index: HashMap<String, usize>,
    embedding_dim: usize,
    num_heads: usize,
    num_layers: usize,
    max_seq_length: usize,
    embeddings: Array2<f32>,
    positional_encoding: Array2<f32>,
    layers: Vec<TransformerLayer>,
    lm_head: Array2<f32>,
}

struct TransformerLayer {
    attention: MultiHeadAttention,
    ff_network: FeedForwardNetwork,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
}

struct FeedForwardNetwork {
    w1: Array2<f32>,
    w2: Array2<f32>,
}

struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    epsilon: f32,
}

impl GPTLanguageModel {
    fn new(vocab: Vec<String>, embedding_dim: usize, num_heads: usize, num_layers: usize, max_seq_length: usize) -> Self {
        let vocab_size = vocab.len();
        let word_to_index: HashMap<_, _> = vocab.iter().cloned().enumerate().map(|(i, w)| (w, i)).collect();
        
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(-0.1, 0.1);
        
        let embeddings = Array2::random((vocab_size, embedding_dim), uniform);
        let positional_encoding = Self::get_positional_encoding(max_seq_length, embedding_dim);
        
        let layers = (0..num_layers)
            .map(|_| TransformerLayer::new(embedding_dim, num_heads))
            .collect();
        
        let lm_head = Array2::random((vocab_size, embedding_dim), uniform);
        
        GPTLanguageModel {
            vocab,
            word_to_index,
            embedding_dim,
            num_heads,
            num_layers,
            max_seq_length,
            embeddings,
            positional_encoding,
            layers,
            lm_head,
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
    
    fn forward(&self, input_ids: &[usize]) -> Array2<f32> {
        let seq_len = input_ids.len();
        let mut x = Array2::zeros((seq_len, self.embedding_dim));
        for (i, &id) in input_ids.iter().enumerate() {
            x.row_mut(i).assign(&self.embeddings.row(id));
        }
        x += &self.positional_encoding.slice(s![..seq_len, ..]);
        
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        
        x.dot(&self.lm_head.t())
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
                
                let logits = self.forward(input_sequence);
                let output_logits = logits.row(logits.nrows() - 1);
                
                let mut loss = 0.0;
                for (j, &logit) in output_logits.iter().enumerate() {
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
    
    fn generate(&self, start_text: &str, num_words: usize) -> String {
        let mut words: Vec<String> = start_text.split_whitespace().map(String::from).collect();
        
        for _ in 0..num_words {
            let input_sequence: Vec<usize> = words.iter()
                .map(|w| *self.word_to_index.get(w).unwrap_or(&0))
                .collect();
            
            let logits = self.forward(&input_sequence);
            let output_logits = logits.row(logits.nrows() - 1);
            
            let next_word_idx = output_logits.iter().enumerate()
                .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();
            
            words.push(self.vocab[next_word_idx].clone());
        }
        
        words.join(" ")
    }
}

impl TransformerLayer {
    fn new(embedding_dim: usize, num_heads: usize) -> Self {
        TransformerLayer {
            attention: MultiHeadAttention::new(embedding_dim, num_heads),
            ff_network: FeedForwardNetwork::new(embedding_dim),
            layer_norm1: LayerNorm::new(embedding_dim),
            layer_norm2: LayerNorm::new(embedding_dim),
        }
    }
    
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let attention_output = self.attention.forward(x);
        let x = self.layer_norm1.forward(&(x + &attention_output));
        let ff_output = self.ff_network.forward(&x);
        self.layer_norm2.forward(&(x + &ff_output))
    }
}

impl MultiHeadAttention {
    fn new(embedding_dim: usize, num_heads: usize) -> Self {
        let head_dim = embedding_dim / num_heads;
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(-0.1, 0.1);
        
        MultiHeadAttention {
            num_heads,
            head_dim,
            w_q: Array2::random((embedding_dim, embedding_dim), uniform),
            w_k: Array2::random((embedding_dim, embedding_dim), uniform),
            w_v: Array2::random((embedding_dim, embedding_dim), uniform),
            w_o: Array2::random((embedding_dim, embedding_dim), uniform),
        }
    }
    
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);
        
        let seq_len = x.nrows();
        let q = q.into_shape((seq_len, self.num_heads, self.head_dim)).unwrap();
        let k = k.into_shape((seq_len, self.num_heads, self.head_dim)).unwrap();
        let v = v.into_shape((seq_len, self.num_heads, self.head_dim)).unwrap();
        
        let attention_scores = q.dot(&k.permuted_axes([0, 2, 1])) / (self.head_dim as f32).sqrt();
        let attention_probs = attention_scores.mapv(|x| x.exp()) / attention_scores.mapv(|x| x.exp()).sum_axis(Axis(2)).insert_axis(Axis(2));
        
        let attention_output = attention_probs.dot(&v)
            .into_shape((seq_len, self.num_heads * self.head_dim)).unwrap();
        
        attention_output.dot(&self.w_o)
    }
}

impl FeedForwardNetwork {
    fn new(embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(-0.1, 0.1);
        
        FeedForwardNetwork {
            w1: Array2::random((embedding_dim, embedding_dim * 4), uniform),
            w2: Array2::random((embedding_dim * 4, embedding_dim), uniform),
        }
    }
    
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.w1).mapv(|x| x.max(0.0)).dot(&self.w2)
    }
}

impl LayerNorm {
    fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            gamma: Array1::ones(embedding_dim),
            beta: Array1::zeros(embedding_dim),
            epsilon: 1e-5,
        }
    }
    
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);
        (x - &mean.insert_axis(Axis(1))) / (var + self.epsilon).sqrt().insert_axis(Axis(1)) * &self.gamma + &self.beta
    }
}

fn main() {
    let vocab = "to be or not that is the question".split_whitespace().map(String::from).collect();
    let mut model = GPTLanguageModel::new(vocab, 32, 2, 2, 50);
    
    let corpus = "to be or not to be that is the question";
    model.train(corpus, 100, 0.01);
    
    let generated = model.generate("to be", 10);
    println!("Generated: {}", generated);
}
```

## Explanation

1. The `GPTLanguageModel` struct contains the overall model structure, including embeddings, positional encodings, and multiple Transformer layers.
2. `TransformerLayer` represents a single layer of the GPT model, including multi-head attention, feed-forward network, and layer normalization.
3. `MultiHeadAttention` implements the core attention mechanism used in Transformers.
4. `FeedForwardNetwork` represents the position-wise feed-forward network in each Transformer layer.
5. `LayerNorm` implements layer normalization, which helps stabilize the network.
6. The `forward` method in `GPTLanguageModel` processes input through all layers to produce output logits.
7. The `train` method implements a simple training loop (without actual backpropagation for brevity).
8. The `generate` method uses the trained model to generate new text.

## Advantages of GPT

- Powerful language modeling capabilities due to its large-scale pre-training on diverse text data.
- Can be fine-tuned for various downstream tasks with minimal task-specific architecture modifications.
- Exhibits strong few-shot and zero-shot learning abilities on many tasks.
- Generates more coherent and contextually appropriate text compared to previous models.

## Limitations

- This implementation is greatly simplified and lacks many optimizations and techniques used in full-scale GPT models.
- Requires significant computational resources for training and inference, especially for larger versions.
- May produce biased or inconsistent outputs, reflecting biases in its training data.
- Lacks explicit reasoning capabilities and can sometimes generate plausible-sounding but incorrect information.

## Evaluation

While we haven't implemented perplexity calculation for this model, it could be done similarly to previous models. In practice, GPT models achieve state-of-the-art perplexity on various language modeling benchmarks and exhibit strong performance on a wide range of NLP tasks.

## Next Steps

This simplified GPT model represents the foundation of many current state-of-the-art language models. Some potential next steps could include:

1. Implementing more advanced training techniques like adaptive learning rates and proper tokenization.
2. Exploring methods for efficient fine-tuning on specific tasks.
3. Investigating techniques for improving model interpretability and controlling generation.
4. Exploring ways to combine the strengths of GPT with other model architectures or external knowledge sources.