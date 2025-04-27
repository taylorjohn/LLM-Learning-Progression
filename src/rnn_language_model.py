import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

class SequenceDataset(Dataset):
    """Dataset for handling sequences for RNN training."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return sequence and target (sequence shifted by 1)
        return self.sequences[idx, :-1], self.sequences[idx, 1:]

class RNNLanguageModel(nn.Module):
    """
    Recurrent Neural Network (RNN) Language Model using PyTorch.
    Predicts the next word based on the sequence processed so far.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1):
        """
        Initializes the RNN Language Model.

        Args:
            vocab_size: The total number of unique words in the vocabulary.
            embedding_dim: The dimension of the word embeddings.
            hidden_dim: The dimension of the RNN hidden state.
            num_layers: Number of stacked RNN layers (default: 1).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Use batch_first=True so input tensors have shape (batch, seq, feature)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # Linear layer to map RNN output to vocabulary space
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Basic weight initialization
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Initialization for RNN weights can be more complex, but default is often okay

    def forward(self, input_seq: torch.Tensor, hidden_state: torch.Tensor):
        """
        Performs the forward pass through the RNN.

        Args:
            input_seq: Tensor of word indices. Shape: (batch_size, seq_len).
            hidden_state: Tensor of the initial hidden state. Shape: (num_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits: Raw output scores for each word in the sequence.
                          Shape: (batch_size, seq_len, vocab_size).
                - hidden_state: The final hidden state. Shape: (num_layers, batch_size, hidden_dim).
        """
        # Get embeddings
        # Input: (batch, seq_len)
        # Output: (batch, seq_len, embedding_dim)
        embeds = self.embedding(input_seq)

        # Pass embeddings and hidden state through RNN
        # Output: (batch, seq_len, hidden_dim)
        # Hidden: (num_layers, batch, hidden_dim)
        rnn_out, hidden_state = self.rnn(embeds, hidden_state)

        # Pass RNN output through the fully connected layer
        # Input: (batch, seq_len, hidden_dim)
        # Output: (batch, seq_len, vocab_size)
        logits = self.fc(rnn_out)

        return logits, hidden_state

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initializes the hidden state to zeros."""
        # The dimensions are (num_layers, batch_size, hidden_dim)
        weight = next(self.parameters()).data
        # Match device and type of model parameters
        hidden = weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()
        return hidden

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
             # Need to resize layers if vocab size changed after init
             self._resize_layers()

    def _resize_layers(self):
         print("Resizing layers due to vocabulary change...")
         self.embedding = nn.Embedding(self.vocab_size, self.embedding.embedding_dim)
         self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
         self._init_weights() # Re-initialize resized layers

    def create_sequences(self, text: str, seq_len: int) -> torch.Tensor:
        """Creates overlapping sequences from text."""
        print(f"Creating sequences of length {seq_len}...")
        words = text.split()
        word_indices = [self.word_to_idx.get(w, -1) for w in words]
        word_indices = [idx for idx in word_indices if idx != -1]

        if len(word_indices) <= seq_len:
            print(f"Warning: Not enough words ({len(word_indices)}) to create sequences of length {seq_len}.")
            return torch.empty((0, seq_len + 1), dtype=torch.long)

        # Create overlapping sequences of length seq_len + 1
        sequences = []
        for i in range(len(word_indices) - seq_len):
            sequences.append(word_indices[i : i + seq_len + 1])

        print(f"Created {len(sequences)} sequences.")
        return torch.tensor(sequences, dtype=torch.long)

    def train_model(self, text: str, epochs: int = 10, learning_rate: float = 0.01,
                    batch_size: int = 32, seq_len: int = 10, clip: float = 1.0):
        """
        Trains the RNN language model.

        Args:
            text: The training text corpus.
            epochs: The number of training epochs.
            learning_rate: The learning rate for the optimizer.
            batch_size: The batch size for training.
            seq_len: The length of the sequences for training.
            clip: Gradient clipping value to prevent exploding gradients.
        """
        if not hasattr(self, 'word_to_idx'):
            self.build_vocab(text)

        sequences_tensor = self.create_sequences(text, seq_len)
        if sequences_tensor.numel() == 0:
            print("Training aborted: No valid sequences created.")
            return

        dataset = SequenceDataset(sequences_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # drop_last=True ensures all batches have the same size, simplifying hidden state handling

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        print(f"\nStarting training for {epochs} epochs...")
        self.train() # Set model to training mode

        for epoch in range(epochs):
            # Initialize hidden state at the start of each epoch
            hidden = self.init_hidden(batch_size)
            total_loss = 0
            num_batches = 0

            for inputs, targets in dataloader:
                # Detach hidden state from previous batch history (Truncated BPTT)
                hidden = hidden.detach()

                optimizer.zero_grad()

                # Forward pass
                # inputs shape: (batch, seq_len)
                # targets shape: (batch, seq_len)
                # hidden shape: (num_layers, batch, hidden_dim)
                logits, hidden = self(inputs, hidden)
                # logits shape: (batch, seq_len, vocab_size)

                # Calculate loss
                # CrossEntropyLoss expects logits as (N, C) and targets as (N)
                # Reshape logits: (batch * seq_len, vocab_size)
                # Reshape targets: (batch * seq_len)
                loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))

                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(self.parameters(), clip)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        print("Training finished.")

    def generate(self, start_word: str, num_words: int, temperature: float = 1.0) -> str:
        """
        Generates text starting from a given word.

        Args:
            start_word: The word to start generation.
            num_words: The number of words to generate.
            temperature: Sampling temperature. Higher values increase randomness,
                         lower values make it more deterministic (closer to greedy).
                         Value must be positive.

        Returns:
            A string containing the generated text (including the start word).
        """
        if not hasattr(self, 'word_to_idx'):
            return "[Model vocabulary not built. Train the model first.]"
        if start_word not in self.word_to_idx:
            return f"[Error: Start word '{start_word}' not in vocabulary.]"
        if temperature <= 0:
             print("Warning: Temperature must be positive. Using default T=1.0")
             temperature = 1.0

        self.eval() # Set model to evaluation mode
        generated_words = [start_word]
        hidden = self.init_hidden(1) # Batch size is 1 for generation
        current_word_idx = self.word_to_idx[start_word]
        # Prepare input tensor (batch_size=1, seq_len=1)
        input_tensor = torch.tensor([[current_word_idx]], dtype=torch.long)

        print(f"\nGenerating {num_words} words starting with: '{start_word}'...")

        with torch.no_grad():
            for _ in range(num_words):
                logits, hidden = self(input_tensor, hidden)
                # logits shape: (1, 1, vocab_size)

                # Apply temperature scaling to logits before softmax
                # Get logits for the last time step, squeeze unnecessary dims
                output_logits = logits.squeeze() / temperature

                # Calculate probabilities using softmax
                probabilities = torch.softmax(output_logits, dim=-1)

                # Sample the next word index from the probability distribution
                next_word_idx = torch.multinomial(probabilities, num_samples=1).item()

                # Get the word string
                next_word = self.idx_to_word[next_word_idx]
                generated_words.append(next_word)

                # Update input for the next iteration
                input_tensor = torch.tensor([[next_word_idx]], dtype=torch.long)

        return " ".join(generated_words)

def main():
    """Main function to demonstrate the RNNLanguageModel."""
    # Parameters
    embedding_dim = 10
    hidden_dim = 20 # Same as Rust example
    num_layers = 1
    seq_len = 5 # Length of sequences for training

    # Training data
    corpus = "to be or not to be that is the question think therefore am ask country can do for you"
    # Slightly longer corpus for potentially better RNN learning
    words_in_corpus = corpus.split()
    vocab_size = len(set(words_in_corpus))

    # Initialize model
    model = RNNLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    # Train the model
    # Rust example used 1000 epochs with simplified BPTT. Needs fewer with proper BPTT.
    model.train_model(corpus, epochs=150, learning_rate=0.01, batch_size=8, seq_len=seq_len, clip=1.0)

    # Generate text
    # Note: RNN generation is stateful, result depends heavily on training
    start_word = "to"
    generated_text = model.generate(start_word, num_words=15, temperature=0.8)
    print(f"\nGenerated text (T=0.8):\n{generated_text}")

    start_word_2 = "is"
    generated_text_2 = model.generate(start_word_2, num_words=10, temperature=1.0)
    print(f"\nGenerated text (T=1.0):\n{generated_text_2}")

if __name__ == "__main__":
    main() 