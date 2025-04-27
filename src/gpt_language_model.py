import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math
import numpy as np

# --- Re-usable components --- #

class SequenceDataset(Dataset):
    """Dataset for handling sequences for training."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return sequence and target (sequence shifted by 1)
        return self.sequences[idx, :-1], self.sequences[idx, 1:]

class PositionalEncoding(nn.Module):
    """Injects positional information into the token embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Changed shape for easier broadcasting with batch_first=True
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # self.pe shape [1, max_len, d_model]
        # x shape [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- GPT Model --- #

class GPTLanguageModel(nn.Module):
    """
    GPT-style (decoder-only) Transformer Language Model using PyTorch's nn.TransformerDecoder.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, max_seq_length: int = 100):
        """
        Args:
            vocab_size: Size of the vocabulary.
            d_model: Embedding dimension (must be divisible by nhead).
            nhead: Number of attention heads.
            d_hid: Dimension of the feedforward network model in nn.TransformerDecoderLayer.
            nlayers: Number of nn.TransformerDecoderLayer layers.
            dropout: Dropout value.
            max_seq_length: Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.model_type = 'GPT'
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        # Use TransformerDecoderLayer for masked self-attention
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Optional: Tie weights between embedding and final layer
        # self.fc_out.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the decoder-only Transformer.

        Args:
            tgt: Target sequence tensor, shape [batch_size, seq_len]
            tgt_mask: Target sequence mask (causal mask), shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, vocab_size]
        """
        # Embedding and positional encoding
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_pos_encoded = self.pos_encoder(tgt_embedded)

        # Transformer decoder
        # For decoder-only model, input sequence acts as both target and memory
        # We only need the target mask (tgt_mask) for causal self-attention.
        output = self.transformer_decoder(tgt=tgt_pos_encoded,
                                          memory=tgt_pos_encoded, # Use target as memory
                                          tgt_mask=tgt_mask,
                                          memory_mask=tgt_mask) # Also mask memory self-attention
        # Output shape: [batch_size, seq_len, d_model]

        # Final linear layer
        output = self.fc_out(output)
        # output shape: [batch_size, seq_len, vocab_size]
        return output

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # --- build_vocab, _resize_layers, create_sequences are identical to Transformer --- #
    def build_vocab(self, text: str):
        print("Building vocabulary...")
        words = text.split()
        word_counts = Counter(words)
        self.vocab = sorted(word_counts.keys())
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        if hasattr(self, 'vocab_size') and self.vocab_size != len(self.vocab):
             print(f"Warning: Provided vocab_size {self.vocab_size} does not match actual vocab size {len(self.vocab)}. Using actual size.")
             self.vocab_size = len(self.vocab)
             self._resize_layers()
        elif not hasattr(self, 'vocab_size'):
             self.vocab_size = len(self.vocab)

    def _resize_layers(self):
         print("Resizing layers due to vocabulary change...")
         self.embedding = nn.Embedding(self.vocab_size, self.d_model)
         self.fc_out = nn.Linear(self.d_model, self.vocab_size)
         self._init_weights()

    def create_sequences(self, text: str, seq_len: int) -> torch.Tensor:
        print(f"Creating sequences of length {seq_len}...")
        words = text.split()
        word_indices = [self.word_to_idx.get(w, -1) for w in words]
        word_indices = [idx for idx in word_indices if idx != -1]

        if len(word_indices) <= seq_len:
            print(f"Warning: Not enough words ({len(word_indices)}) to create sequences of length {seq_len}.")
            return torch.empty((0, seq_len + 1), dtype=torch.long)

        sequences = []
        for i in range(len(word_indices) - seq_len):
            sequences.append(word_indices[i : i + seq_len + 1])

        print(f"Created {len(sequences)} sequences.")
        return torch.tensor(sequences, dtype=torch.long)
    # --- End of re-used methods --- #

    def train_model(self, text: str, epochs: int = 10, learning_rate: float = 0.0005, # Lower LR for GPT
                    batch_size: int = 16, seq_len: int = 30, clip: float = 0.5):
        """
        Trains the GPT language model.
        """
        if not hasattr(self, 'word_to_idx'):
            self.build_vocab(text)

        sequences_tensor = self.create_sequences(text, seq_len)
        if sequences_tensor.numel() == 0:
            print("Training aborted: No valid sequences created.")
            return

        dataset = SequenceDataset(sequences_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        device = next(self.parameters()).device
        print(f"\nStarting training on {device} for {epochs} epochs...")
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for inputs, targets in dataloader:
                current_seq_len = inputs.size(1)
                # Skip if batch seq len doesn't match expected (can happen with last batch if not drop_last)
                if current_seq_len != seq_len:
                     continue

                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                # Generate causal mask for the current sequence length
                tgt_mask = self._generate_square_subsequent_mask(current_seq_len, device)

                logits = self(inputs, tgt_mask)
                loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            scheduler.step()

        print("Training finished.")

    def generate(self, start_text: str, num_words: int, temperature: float = 1.0) -> str:
        """
        Generates text autoregressively using the GPT model.
        """
        if not hasattr(self, 'word_to_idx'):
            return "[Model vocabulary not built. Train the model first.]"

        self.eval()
        device = next(self.parameters()).device

        words = start_text.split()
        generated_indices = [self.word_to_idx.get(w, 0) for w in words] # Use index 0 for OOV in start

        print(f"\nGenerating {num_words} words starting with: '{start_text}'...")

        with torch.no_grad():
            for _ in range(num_words):
                input_tensor = torch.tensor([generated_indices], dtype=torch.long).to(device)
                current_seq_len = input_tensor.size(1)
                mask = self._generate_square_subsequent_mask(current_seq_len, device)

                logits = self(input_tensor, mask)
                last_logits = logits[:, -1, :] / temperature
                probabilities = torch.softmax(last_logits, dim=-1)
                next_word_idx = torch.multinomial(probabilities, num_samples=1).item()

                generated_indices.append(next_word_idx)

        generated_words = [self.idx_to_word.get(idx, "<UNK>") for idx in generated_indices] # Handle potential unknown idx
        return " ".join(generated_words)

def main():
    """Main function to demonstrate the GPTLanguageModel."""
    # Parameters
    d_model = 32  # Embedding dimension
    nhead = 2     # Attention heads
    d_hid = 64    # Feedforward hidden dim
    nlayers = 2   # Number of decoder layers
    dropout = 0.1
    seq_len = 15  # Sequence length for training
    max_len_pos_enc = 60

    # Training data
    corpus = "to be or not to be that is the question I think therefore I am ask not what your country can do for you ask what you can do for your country explore discover dream"
    words_in_corpus = corpus.split()
    vocab_size = len(set(words_in_corpus))

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        dropout=dropout,
        max_seq_length=max_len_pos_enc
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Train the model
    model.train_model(corpus, epochs=60, learning_rate=0.0005, batch_size=8, seq_len=seq_len, clip=0.5)

    # Generate text
    start_text = "country can"
    generated_text = model.generate(start_text, num_words=25, temperature=0.7)
    print(f"\nGenerated text (T=0.7):\n{generated_text}")

if __name__ == "__main__":
    main() 