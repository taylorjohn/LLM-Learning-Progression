# M2-Optimized GPT Model in Rust and PyTorch

This implementation optimizes our GPT model for training on MacBook Air M2 GPUs. It includes an improved BPE tokenizer, an efficient training loop, and uses the Adam optimizer with learning rate scheduling. We'll use a combination of Rust for preprocessing and PyTorch for the model and training, leveraging PyTorch's MPS backend for Apple Silicon GPUs.

## Rust Implementation (src/main.rs)

First, let's implement an improved BPE tokenizer in Rust:

```rust
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use regex::Regex;

struct BPETokenizer {
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    regex: Regex,
}

impl BPETokenizer {
    fn new() -> Self {
        BPETokenizer {
            encoder: HashMap::new(),
            decoder: HashMap::new(),
            bpe_ranks: HashMap::new(),
            regex: Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap(),
        }
    }

    fn train(&mut self, texts: &[String], vocab_size: usize, min_frequency: usize) {
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        
        // Count word frequencies
        for text in texts {
            for word in self.regex.find_iter(text) {
                *word_freq.entry(word.as_str().to_string()).or_insert(0) += 1;
            }
        }

        // Initialize with characters
        let mut vocab: HashSet<String> = word_freq.keys()
            .flat_map(|word| word.chars().map(|c| c.to_string()))
            .collect();

        while vocab.len() < vocab_size {
            let mut pairs = HashMap::new();
            for (word, freq) in &word_freq {
                let symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                for i in 0..symbols.len() - 1 {
                    let pair = (symbols[i].clone(), symbols[i + 1].clone());
                    *pairs.entry(pair).or_insert(0) += freq;
                }
            }

            let best_pair = pairs.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, _)| pair);

            if let Some((first, second)) = best_pair {
                let new_token = format!("{}{}", first, second);
                vocab.insert(new_token.clone());
                self.bpe_ranks.insert((first.clone(), second.clone()), self.bpe_ranks.len());

                // Update word frequencies
                let mut new_word_freq = HashMap::new();
                for (word, freq) in word_freq {
                    let new_word = word.replace(&format!("{}{}", first, second), &new_token);
                    *new_word_freq.entry(new_word).or_insert(0) += freq;
                }
                word_freq = new_word_freq;
            } else {
                break;
            }
        }

        // Create encoder and decoder
        for (i, token) in vocab.into_iter().enumerate() {
            self.encoder.insert(token.clone(), i);
            self.decoder.insert(i, token);
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        for word in self.regex.find_iter(text) {
            let mut word = word.as_str().to_string();
            let mut sub_tokens = vec![word.clone()];
            
            loop {
                let mut min_pair = None;
                let mut min_rank = std::usize::MAX;

                for i in 0..sub_tokens.len() - 1 {
                    let pair = (sub_tokens[i].clone(), sub_tokens[i + 1].clone());
                    if let Some(&rank) = self.bpe_ranks.get(&pair) {
                        if rank < min_rank {
                            min_pair = Some(i);
                            min_rank = rank;
                        }
                    }
                }

                if let Some(i) = min_pair {
                    let new_token = format!("{}{}", sub_tokens[i], sub_tokens[i + 1]);
                    sub_tokens[i] = new_token;
                    sub_tokens.remove(i + 1);
                } else {
                    break;
                }
            }

            tokens.extend(sub_tokens.into_iter().filter_map(|t| self.encoder.get(&t)).cloned());
        }
        tokens
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&token| self.decoder.get(&token))
            .collect::<Vec<_>>()
            .join("")
            .replace("</w>", " ")
            .trim()
            .to_string()
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        for (token, id) in &self.encoder {
            writeln!(file, "{}\t{}", token, id)?;
        }
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut tokenizer = BPETokenizer::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() == 2 {
                let token = parts[0].to_string();
                let id = parts[1].parse().unwrap();
                tokenizer.encoder.insert(token.clone(), id);
                tokenizer.decoder.insert(id, token);
            }
        }

        Ok(tokenizer)
    }
}

fn main() {
    // Example usage
    let texts = vec![
        "Hello, world!".to_string(),
        "This is a test.".to_string(),
        "BPE tokenization is fun.".to_string(),
    ];

    let mut tokenizer = BPETokenizer::new();
    tokenizer.train(&texts, 100, 2);

    // Save the tokenizer
    tokenizer.save("tokenizer.txt").unwrap();

    // Load the tokenizer
    let loaded_tokenizer = BPETokenizer::load("tokenizer.txt").unwrap();

    // Test encoding and decoding
    let text = "Hello, this is a test of BPE tokenization!";
    let encoded = loaded_tokenizer.encode(text);
    let decoded = loaded_tokenizer.decode(&encoded);

    println!("Original: {}", text);
    println!("Encoded: {:?}", encoded);
    println!("Decoded: {}", decoded);
}
```

Now, let's implement the GPT model and training loop in PyTorch, optimized for M2 GPUs:

## Python Implementation (train_gpt.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd // config.n_head, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd // config.n_head, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd // config.n_head, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        return att @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class TextDataset(Dataset):
    def __init__(self, text, block_size, tokenizer):
        self.tokenizer = tokenizer
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i:i+self.block_size+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def train(model, train_dataset, val_dataset, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataset), eta_min=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=config