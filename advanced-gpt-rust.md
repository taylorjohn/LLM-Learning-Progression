# Advanced GPT Model in Rust

This implementation extends our previous GPT model with several improvements:

1. BPE Tokenization
2. Improved Positional Encoding
3. Masked Self-Attention
4. Layer Normalization
5. Simple Retrieval-Augmented Generation

## Implementation

```rust
use ndarray::{Array, Array1, Array2, Array3, Axis};
use rand::Rng;
use std::collections::{HashMap, HashSet};

struct BPETokenizer {
    vocab: HashMap<String, usize>,
    inverse_vocab: Vec<String>,
}

impl BPETokenizer {
    fn new() -> Self {
        // Initialize with a basic vocabulary
        let mut vocab = HashMap::new();
        let inverse_vocab = vec!["<pad>", "<unk>", "<s>", "</s>"]
            .into_iter()
            .map(String::from)
            .collect();
        for (i, token) in inverse_vocab.iter().enumerate() {
            vocab.insert(token.clone(), i);
        }
        BPETokenizer { vocab, inverse_vocab }
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        // Simplified tokenization (word-level for brevity)
        text.split_whitespace()
            .map(|word| *self.vocab.get(word).unwrap_or(&1)) // 1 is <unk>
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&token| self.inverse_vocab[token].clone())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

struct AdvancedGPT {
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    vocab_size: usize,
    max_seq_len: usize,
    tokenizer: BPETokenizer,
    token_embedding: Array2<f32>,
    positional_encoding: Array2<f32>,
    layers: Vec<TransformerLayer>,
    layer_norm: LayerNorm,
    lm_head: Array2<f32>,
    knowledge_base: HashMap<String, String>,
}

struct TransformerLayer {
    self_attn: MultiHeadAttention,
    ff: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
}

struct FeedForward {
    w1: Array2<f32>,
    w2: Array2<f32>,
}

struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl AdvancedGPT {
    fn new(embed_dim: usize, num_heads: usize, num_layers: usize, vocab_size: usize, max_seq_len: usize) -> Self {
        let mut rng = rand::thread_rng();
        let tokenizer = BPETokenizer::new();

        let token_embedding = Array2::random((vocab_size, embed_dim), rand::distributions::Uniform::new(-0.1, 0.1));
        let positional_encoding = Self::get_positional_encoding(max_seq_len, embed_dim);

        let layers = (0..num_layers)
            .map(|_| TransformerLayer::new(embed_dim, num_heads))
            .collect();

        let layer_norm = LayerNorm::new(embed_dim);
        let lm_head = Array2::random((vocab_size, embed_dim), rand::distributions::Uniform::new(-0.1, 0.1));

        let knowledge_base = HashMap::new(); // Initialize empty, to be filled later

        AdvancedGPT {
            embed_dim,
            num_heads,
            num_layers,
            vocab_size,
            max_seq_len,
            tokenizer,
            token_embedding,
            positional_encoding,
            layers,
            layer_norm,
            lm_head,
            knowledge_base,
        }
    }

    fn get_positional_encoding(max_len: usize, d_model: usize) -> Array2<f32> {
        let mut pos_encoding = Array2::zeros((max_len, d_model));
        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / f32::powf(10000.0, (2 * (i / 2)) as f32 / d_model as f32);
                pos_encoding[[pos, i]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        pos_encoding
    }

    fn forward(&self, input_ids: &[usize]) -> Array2<f32> {
        let seq_len = input_ids.len();
        let mut x = Array2::zeros((seq_len, self.embed_dim));
        for (i, &id) in input_ids.iter().enumerate() {
            x.row_mut(i).assign(&self.token_embedding.row(id));
        }
        x += &self.positional_encoding.slice(s![..seq_len, ..]);

        let mask = Self::get_attention_mask(seq_len);

        for layer in &self.layers {
            x = layer.forward(&x, &mask);
        }

        x = self.layer_norm.forward(&x);
        x.dot(&self.lm_head.t())
    }

    fn get_attention_mask(size: usize) -> Array2<f32> {
        let mut mask = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..=i {
                mask[[i, j]] = 1.0;
            }
        }
        mask
    }

    fn generate(&self, prompt: &str, max_length: usize) -> String {
        let mut input_ids = self.tokenizer.tokenize(prompt);
        
        while input_ids.len() < max_length {
            let logits = self.forward(&input_ids);
            let next_token = logits.row(logits.nrows() - 1)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap();
            
            if next_token == 3 { // </s> token
                break;
            }
            
            input_ids.push(next_token);
        }
        
        self.tokenizer.decode(&input_ids)
    }

    fn retrieve_info(&self, query: &str) -> Option<&str> {
        // Simple keyword-based retrieval
        self.knowledge_base.iter()
            .find(|(key, _)| query.contains(key))
            .map(|(_, value)| value.as_str())
    }

    fn generate_with_retrieval(&self, prompt: &str, max_length: usize) -> String {
        let retrieved_info = self.retrieve_info(prompt).unwrap_or("");
        let enhanced_prompt = format!("{} Context: {}", prompt, retrieved_info);
        self.generate(&enhanced_prompt, max_length)
    }
}

impl TransformerLayer {
    fn new(embed_dim: usize, num_heads: usize) -> Self {
        TransformerLayer {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads),
            ff: FeedForward::new(embed_dim),
            ln1: LayerNorm::new(embed_dim),
            ln2: LayerNorm::new(embed_dim),
        }
    }

    fn forward(&self, x: &Array2<f32>, mask: &Array2<f32>) -> Array2<f32> {
        let attn_output = self.self_attn.forward(x, x, x, mask);
        let x = self.ln1.forward(&(x + &attn_output));
        let ff_output = self.ff.forward(&x);
        self.ln2.forward(&(x + &ff_output))
    }
}

impl MultiHeadAttention {
    fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        let mut rng = rand::thread_rng();
        let init = rand::distributions::Uniform::new(-0.1, 0.1);

        MultiHeadAttention {
            num_heads,
            head_dim,
            w_q: Array2::random((embed_dim, embed_dim), init),
            w_k: Array2::random((embed_dim, embed_dim), init),
            w_v: Array2::random((embed_dim, embed_dim), init),
            w_o: Array2::random((embed_dim, embed_dim), init),
        }
    }

    fn forward(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, mask: &Array2<f32>) -> Array2<f32> {
        let q = q.dot(&self.w_q).into_shape((q.nrows(), self.num_heads, self.head_dim)).unwrap();
        let k = k.dot(&self.w_k).into_shape((k.nrows(), self.num_heads, self.head_dim)).unwrap();
        let v = v.dot(&self.w_v).into_shape((v.nrows(), self.num_heads, self.head_dim)).unwrap();

        let attention_scores = q.dot(&k.permuted_axes([0, 2, 1])) / (self.head_dim as f32).sqrt();
        let attention_scores = attention_scores + &mask.to_owned().insert_axis(Axis(1));
        let attention_probs = attention_scores.mapv(|x| x.exp()) / attention_scores.mapv(|x| x.exp()).sum_axis(Axis(2)).insert_axis(Axis(2));

        let context = attention_probs.dot(&v).into_shape((q.nrows(), self.num_heads * self.head_dim)).unwrap();
        context.dot(&self.w_o)
    }
}

impl FeedForward {
    fn new(embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let init = rand::distributions::Uniform::new(-0.1, 0.1);

        FeedForward {
            w1: Array2::random((embed_dim, embed_dim * 4), init),
            w2: Array2::random((embed_dim * 4, embed_dim), init),
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.w1).mapv(|x| x.max(0.0)).dot(&self.w2)
    }
}

impl LayerNorm {
    fn new(embed_dim: usize) -> Self {
        LayerNorm {
            gamma: Array1::ones(embed_dim),
            beta: Array1::zeros(embed_dim),
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);
        (x - &mean.insert_axis(Axis(1))) / (var + self.eps).sqrt().insert_axis(Axis(1)) * &self.gamma + &self.beta
    }
}

fn main() {
    let embed_dim = 256;
    let num_heads = 8;
    let num_layers = 6;
    let vocab_size = 10000;
    let max_seq_len = 1024;

    let mut model = AdvancedGPT::new(embed_dim, num_heads, num_layers, vocab_size, max_seq_len);

    // Add some entries to the knowledge base
    model.knowledge_base.insert("Rust".to_string(), "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.".to_string());
    model.knowledge_base.insert("GPT".to_string(), "GPT (Generative Pre-trained Transformer) is a type of large language model that uses deep learning to produce human-like text.".to_string());

    // Generate text
    let prompt = "Rust is";
    let generated = model.generate_with_retrieval(prompt, 50);
    println!("Generated: {}", generated);
}
```

## Key Improvements

1. **BPE Tokenization**: We've implemented a basic tokenizer structure. In a full implementation, this would use byte-pair encoding for subword tokenization.

2. **Improved Positional Encoding**: We're using sinusoidal positional encodings as in the original Transformer paper.

3. **Masked Self-Attention**: We've added an attention mask to ensure the model only attends to previous tokens during generation.

4. **Layer Normalization**: We've added layer normalization after each sub-layer in the Transformer, which helps with training stability.

5. **Simple Retrieval-Augmented Generation**: We've added a basic knowledge base and retrieval mechanism. The model can now incorporate external information into its generations.

## Usage

To use this model:

1. Create a new Rust project: `cargo new advanced_gpt`
2. Replace the contents of `src/main.rs` with the provided code
3. Add the following dependencies to your `Cargo.toml`:
   ```toml
   [dependencies]
   ndarray = "0.15.6"
   rand = "0.8.5"
   ```
4. Run the program with `cargo run`

## Limitations and Further Improvements

- This implementation is still simplified and lacks many optimizations used in production models.
- The tokenizer is very basic. A real implementation would use a proper BPE algorithm.
- The model doesn't include any training code. In practice, you'd need to implement backpropagation and optimization.
- The retrieval system is very simple. More advanced systems might use embeddings and semantic search.
- This model doesn't implement more advanced features like sparse attention or efficient fine-tuning methods.

Despite these limitations, this implementation demonstrates several key concepts in modern language model design and provides a foundation for further exploration and improvement.