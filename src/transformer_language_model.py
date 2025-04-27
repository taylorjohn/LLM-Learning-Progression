import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math
import numpy as np

# Re-use SequenceDataset from RNN/LSTM implementation
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
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so they can be summed.
    Here, we use sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer so it's not a model parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
               (Note: PyTorch Transformer layers expect seq_len first if batch_first=False)
               Or shape [batch_size, seq_len, embedding_dim] if batch_first=True for the nn.TransformerEncoderLayer
        """
        # Assumes x shape is [batch_size, seq_len, embedding_dim]
        # self.pe[:x.size(1)] shape is [seq_len, 1, embedding_dim]
        # Need to transpose pe to match batch dimension if needed, or adjust indexing
        # Let's adjust pe access assuming x is [batch, seq, embed]
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0) # Add PE to batch
        return self.dropout(x)

class TransformerLanguageModel(nn.Module):
    """
    Transformer Language Model using PyTorch's nn.TransformerEncoder.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, max_seq_length: int = 100):
        """
        Args:
            vocab_size: Size of the vocabulary.
            d_model: Embedding dimension (must be divisible by nhead).
            nhead: Number of attention heads.
            d_hid: Dimension of the feedforward network model in nn.TransformerEncoderLayer.
            nlayers: Number of nn.TransformerEncoderLayer layers.
            dropout: Dropout value.
            max_seq_length: Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]
                A square attention mask prepared for nn.TransformerEncoder.

        Returns:
            output Tensor of shape [batch_size, seq_len, vocab_size]
        """
        # Embedding and positional encoding
        # src shape: [batch, seq]
        embedded = self.embedding(src) * math.sqrt(self.d_model) # Scale embeddings
        # embedded shape: [batch, seq, d_model]
        pos_encoded = self.pos_encoder(embedded)
        # pos_encoded shape: [batch, seq, d_model]

        # Transformer encoder
        # Needs src_mask [seq, seq]
        output = self.transformer_encoder(pos_encoded, src_mask)
        # output shape: [batch, seq, d_model]

        # Final linear layer
        output = self.fc_out(output)
        # output shape: [batch, seq, vocab_size]
        return output

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generates a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0)."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def build_vocab(self, text: str):
        """Builds vocabulary from the training text."""
        print("Building vocabulary...")
        words = text.split()
        word_counts = Counter(words)
        self.vocab = sorted(word_counts.keys())
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        if self.vocab_size != len(self.vocab):
             print(f"Warning: Provided vocab_size {self.vocab_size} does not match actual vocab size {len(self.vocab)}. Using actual size.")
             self.vocab_size = len(self.vocab)
             self._resize_layers()

    def _resize_layers(self):
         print("Resizing layers due to vocabulary change...")
         self.embedding = nn.Embedding(self.vocab_size, self.d_model)
         self.fc_out = nn.Linear(self.d_model, self.vocab_size)
         self._init_weights()

    def create_sequences(self, text: str, seq_len: int) -> torch.Tensor:
        """Creates overlapping sequences from text."""
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

    def train_model(self, text: str, epochs: int = 10, learning_rate: float = 0.001, # LR often lower for transformers
                    batch_size: int = 16, seq_len: int = 20, clip: float = 0.5):
        """
        Trains the Transformer language model.

        Args:
            text: The training text corpus.
            epochs: The number of training epochs.
            learning_rate: The learning rate for the optimizer.
            batch_size: The batch size for training.
            seq_len: The length of the sequences for training.
            clip: Gradient clipping value.
        """
        if not hasattr(self, 'word_to_idx'):
            self.build_vocab(text)

        sequences_tensor = self.create_sequences(text, seq_len)
        if sequences_tensor.numel() == 0:
            print("Training aborted: No valid sequences created.")
            return

        dataset = SequenceDataset(sequences_tensor)
        # Note: Batch size might need adjustment based on memory
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) # Example scheduler

        print(f"\nStarting training for {epochs} epochs...")
        self.train() # Set model to training mode

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            # Generate mask once per epoch if seq_len is fixed
            src_mask = self._generate_square_subsequent_mask(seq_len).to(next(self.parameters()).device)

            for inputs, targets in dataloader:
                optimizer.zero_grad()

                # inputs/targets shape: [batch_size, seq_len]
                if inputs.size(1) != seq_len:
                    # Handle potential last batch if not dropping
                    # This requires regenerating the mask - simpler to use drop_last=True
                    # Or pad batches, but SequenceDataset doesn't handle that.
                    # For simplicity, let's assume drop_last=True or handle fixed seq_len
                    continue # Skip incomplete batches if drop_last=False

                # Ensure mask matches device
                if src_mask.device != inputs.device:
                    src_mask = src_mask.to(inputs.device)

                logits = self(inputs, src_mask)
                # logits shape: [batch_size, seq_len, vocab_size]

                # Reshape for CrossEntropyLoss: [batch_size * seq_len, vocab_size]
                # Targets shape: [batch_size * seq_len]
                loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            scheduler.step() # Adjust learning rate

        print("Training finished.")


    def generate(self, start_word: str, num_words: int, temperature: float = 1.0) -> str:
        """
        Generates text autoregressively using the Transformer model.

        Args:
            start_word: The word to start generation.
            num_words: The number of words to generate.
            temperature: Sampling temperature.

        Returns:
            A string containing the generated text.
        """
        if not hasattr(self, 'word_to_idx'):
            return "[Model vocabulary not built. Train the model first.]"
        if start_word not in self.word_to_idx:
            return f"[Error: Start word '{start_word}' not in vocabulary.]"
        if temperature <= 0:
            print("Warning: Temperature must be positive. Using default T=1.0")
            temperature = 1.0

        self.eval() # Set model to evaluation mode
        device = next(self.parameters()).device

        # Start with the initial word index
        generated_indices = [self.word_to_idx[start_word]]

        print(f"\nGenerating {num_words} words starting with: '{start_word}'...")

        with torch.no_grad():
            for _ in range(num_words):
                # Prepare input tensor from current sequence
                input_tensor = torch.tensor([generated_indices], dtype=torch.long).to(device)
                seq_len = input_tensor.size(1)

                # Generate mask for the current sequence length
                mask = self._generate_square_subsequent_mask(seq_len).to(device)

                # Forward pass
                logits = self(input_tensor, mask)
                # logits shape: (1, seq_len, vocab_size)

                # Get logits for the *last* token only
                last_logits = logits[:, -1, :] / temperature

                # Calculate probabilities and sample
                probabilities = torch.softmax(last_logits, dim=-1)
                next_word_idx = torch.multinomial(probabilities, num_samples=1).item()

                # Append the chosen word index
                generated_indices.append(next_word_idx)

        # Convert indices back to words
        generated_words = [self.idx_to_word[idx] for idx in generated_indices]
        return " ".join(generated_words)

def main():
    """Main function to demonstrate the TransformerLanguageModel."""
    # Parameters (loosely based on Rust example, adjusted for PyTorch defaults)
    d_model = 16 # Embedding dimension
    nhead = 2    # Number of attention heads (d_model must be divisible by nhead)
    d_hid = 32   # Dimension of feedforward network
    nlayers = 2  # Number of Transformer encoder layers
    dropout = 0.1
    seq_len = 10 # Max sequence length for training/generation context window
    max_len_pos_enc = 50 # Max length for positional encoding buffer

    # Training data
    corpus = "to be or not to be that is the question I think therefore I am ask not what your country can do for you ask what you can do for your country"
    words_in_corpus = corpus.split()
    vocab_size = len(set(words_in_corpus))

    # Initialize model
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        dropout=dropout,
        max_seq_length=max_len_pos_enc
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Train the model
    # Transformers usually require more data and careful tuning
    model.train_model(corpus, epochs=50, learning_rate=0.001, batch_size=8, seq_len=seq_len, clip=0.5)

    # Generate text
    start_word = "is"
    # Ensure start word is in vocab
    if start_word not in model.word_to_idx:
        start_word = list(model.word_to_idx.keys())[0]
        print(f"Start word not in vocab, using '{start_word}' instead.")

    generated_text = model.generate(start_word, num_words=15, temperature=0.8)
    print(f"\nGenerated text (T=0.8):\n{generated_text}")

if __name__ == "__main__":
    main() 