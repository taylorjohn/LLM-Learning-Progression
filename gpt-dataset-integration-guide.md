# Dataset Integration and Usage Guide for GPT Model Progression

## Recommended Dataset: WikiText-2

For our GPT model progression, we'll use the WikiText-2 dataset. It's a good choice because:

1. It's derived from Wikipedia articles, providing a mix of topics and styles.
2. It's large enough for advanced models but can be subsampled for simpler ones.
3. It's freely available and widely used in NLP research.

## Downloading the Dataset

1. Create a `data` directory in your project root:
   ```
   mkdir data
   cd data
   ```

2. Download the WikiText-2 dataset:
   ```
   wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
   unzip wikitext-2-raw-v1.zip
   ```

3. You'll now have `wiki.train.raw`, `wiki.valid.raw`, and `wiki.test.raw` files.

## Integrating the Dataset

Here's how to integrate the WikiText-2 dataset into each model and run them:

### 1. Unigram Model

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let file = File::open("data/wikitext-2-raw/wiki.train.raw").unwrap();
    let reader = BufReader::new(file);
    let mut corpus = String::new();
    
    for line in reader.lines().take(100) {  // Take first 100 lines for simplicity
        corpus.push_str(&line.unwrap());
        corpus.push('\n');
    }

    let mut model = UnigramModel::new();
    model.train(&corpus);

    // Generate text
    let generated = model.generate(10);  // Generate 10 words
    println!("Generated: {}", generated);
}
```

Run with: `cargo run --bin unigram_model`

### 2-4. N-gram Models (including Backoff)

```rust
fn main() {
    let file = File::open("data/wikitext-2-raw/wiki.train.raw").unwrap();
    let reader = BufReader::new(file);
    let mut corpus = String::new();
    
    for line in reader.lines().take(1000) {  // Take first 1000 lines
        corpus.push_str(&line.unwrap());
        corpus.push('\n');
    }

    let mut model = NGramModel::new(3);  // For trigram model
    model.train(&corpus);

    // Generate text
    let generated = model.generate("The quick brown", 20);  // Generate 20 words
    println!("Generated: {}", generated);
}
```

Run with: `cargo run --bin ngram_model`

### 5. Feed-Forward Neural Network Language Model

```python
import torch
from torch.utils.data import Dataset, DataLoader

class WikiTextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        with open(file_path, 'r') as f:
            self.data = f.read()
        self.words = self.data.split()
        self.word_to_ix = {word: i for i, word in enumerate(set(self.words))}
        self.ix_to_word = {i: word for word, i in self.word_to_ix.items()}
        self.seq_length = seq_length

    def __len__(self):
        return len(self.words) - self.seq_length

    def __getitem__(self, idx):
        inputs = [self.word_to_ix[word] for word in self.words[idx:idx+self.seq_length]]
        targets = [self.word_to_ix[word] for word in self.words[idx+1:idx+self.seq_length+1]]
        return torch.tensor(inputs), torch.tensor(targets)

# In your main training loop:
dataset = WikiTextDataset('data/wikitext-2-raw/wiki.train.raw', seq_length=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FFNNLanguageModel(vocab_size=len(dataset.word_to_ix), embed_dim=128, hidden_dim=256)
# ... (rest of your training code)
```

Run with: `python train_ffnn.py`

### 6-8. RNN, LSTM, and Transformer Models

These models can use a similar data loading approach as the FFNN, but may process longer sequences:

```python
dataset = WikiTextDataset('data/wikitext-2-raw/wiki.train.raw', seq_length=50)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = TransformerLanguageModel(vocab_size=len(dataset.word_to_ix), d_model=512, nhead=8, num_layers=6)
# ... (rest of your training code)
```

Run with: `python train_transformer.py`

### 9-11. GPT Models (Simplified, Advanced, and M2-Optimized)

For these models, you'll want to use the entire dataset and potentially implement more sophisticated data loading:

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
    
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator

train_dataset, test_dataset, data_collator = load_dataset(
    'data/wikitext-2-raw/wiki.train.raw',
    'data/wikitext-2-raw/wiki.test.raw',
    tokenizer
)

training_args = TrainingArguments(
    output_dir="./gpt2-wikitext2",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
```

Run with: `python train_gpt.py`

## General Tips

1. For simpler models, use a subset of the data to keep training times reasonable.
2. For more advanced models, use the full dataset and consider using GPU acceleration if available.
3. Always split your data into train, validation, and test sets. WikiText-2 provides these splits.
4. For the M2-optimized version, make sure to use the MPS backend as described in the earlier implementation.

By following these guidelines, you can effectively integrate the WikiText-2 dataset into each model in our progression, providing a consistent basis for comparison and demonstrating the increasing capabilities of more advanced models.