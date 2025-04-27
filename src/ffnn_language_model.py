import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter, deque
import random
import math

class TextDataset(Dataset):
    """Simple Dataset for FFNN language model training."""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class FFNNLanguageModel(nn.Module):
    """
    Feedforward Neural Network Language Model using PyTorch.

    Predicts the next word based on a fixed-size context window.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, context_size: int):
        """
        Initializes the FFNN Language Model.

        Args:
            vocab_size: The total number of unique words in the vocabulary.
            embedding_dim: The dimension of the word embeddings.
            hidden_dim: The dimension of the hidden layer.
            context_size: The number of preceding words to use as context.
        """
        super().__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Linear layers
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights (optional but often helpful)
        self._init_weights()

    def _init_weights(self):
        # Simple initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            context_indices: A tensor of word indices for the context window.
                             Shape: (batch_size, context_size)

        Returns:
            A tensor of raw scores (logits) for each word in the vocabulary.
            Shape: (batch_size, vocab_size)
        """
        # Get embeddings for the context words
        # Input shape: (batch_size, context_size)
        # Output shape: (batch_size, context_size, embedding_dim)
        embeds = self.embeddings(context_indices)

        # Concatenate the embeddings
        # Reshape to (batch_size, context_size * embedding_dim)
        inputs = embeds.view(embeds.size(0), -1)

        # Pass through the hidden layer and activation function
        # Shape: (batch_size, hidden_dim)
        hidden = self.activation(self.linear1(inputs))

        # Pass through the output layer
        # Shape: (batch_size, vocab_size)
        logits = self.linear2(hidden)

        return logits

    def build_vocab(self, text: str):
        """Builds vocabulary from the training text."""
        print("Building vocabulary...")
        words = text.split()
        word_counts = Counter(words)
        # Simple vocab construction - could add <UNK> handling later
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
         self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
         self.linear2 = nn.Linear(self.linear1.out_features, self.vocab_size)
         self._init_weights() # Re-initialize resized layers


    def create_sequences(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Creates context sequences and target words from text."""
        print("Creating sequences...")
        words = text.split()
        word_indices = [self.word_to_idx.get(w, -1) for w in words] # Handle potential OOV if vocab built separately
        word_indices = [idx for idx in word_indices if idx != -1] # Filter out unknown words for now

        if len(word_indices) <= self.context_size:
            print("Warning: Not enough words in text to create any sequences.")
            return torch.empty((0, self.context_size), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        sequences = []
        targets = []
        window = deque(maxlen=self.context_size)

        for i in range(len(word_indices)):
            if len(window) == self.context_size:
                sequences.append(list(window))
                targets.append(word_indices[i])
            window.append(word_indices[i])

        print(f"Created {len(targets)} sequences.")
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def train_model(self, text: str, epochs: int = 10, learning_rate: float = 0.01, batch_size: int = 32):
        """
        Trains the language model.

        Args:
            text: The training text corpus.
            epochs: The number of training epochs.
            learning_rate: The learning rate for the optimizer.
            batch_size: The batch size for training.
        """
        if not hasattr(self, 'word_to_idx'):
             self.build_vocab(text) # Build vocab if not done externally

        sequences, targets = self.create_sequences(text)
        if sequences.numel() == 0:
             print("Training aborted: No valid sequences created.")
             return

        dataset = TextDataset(sequences, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        print(f"\nStarting training for {epochs} epochs...")
        self.train() # Set model to training mode

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            for batch_sequences, batch_targets in dataloader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                logits = self(batch_sequences)

                # Calculate loss
                loss = criterion(logits, batch_targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        print("Training finished.")


    def generate(self, start_context: list[str], num_words: int) -> str:
        """
        Generates text starting from a given context.

        Args:
            start_context: A list of words to start the generation. Its length
                           must be equal to the model's context_size.
            num_words: The number of words to generate after the start_context.

        Returns:
            A string containing the generated text.
        """
        if not hasattr(self, 'word_to_idx'):
            return "[Model vocabulary not built. Train the model first.]"
        if len(start_context) != self.context_size:
            return f"[Error: Start context length ({len(start_context)}) must equal model context size ({self.context_size})]"
        if not all(word in self.word_to_idx for word in start_context):
             unknown = [w for w in start_context if w not in self.word_to_idx]
             return f"[Error: Start context contains unknown words: {unknown}]"

        self.eval() # Set model to evaluation mode
        generated_words = list(start_context)
        context_window = deque(self.word_to_idx[word] for word in start_context)

        print(f"\nGenerating {num_words} words starting with: '{' '.join(start_context)}'...")

        with torch.no_grad(): # No need to track gradients during generation
            for _ in range(num_words):
                # Prepare context tensor
                context_tensor = torch.tensor([list(context_window)], dtype=torch.long)

                # Forward pass
                logits = self(context_tensor)

                # Get probabilities (optional, could use logits directly for argmax)
                # probabilities = torch.softmax(logits, dim=1)

                # Sample the next word (greedy decoding like Rust example)
                next_word_idx = torch.argmax(logits, dim=1).item()
                # Alternative: Sampling using probabilities
                # next_word_idx = torch.multinomial(probabilities, num_samples=1).item()

                # Convert index to word
                next_word = self.idx_to_word[next_word_idx]
                generated_words.append(next_word)

                # Update the context window
                context_window.popleft()
                context_window.append(next_word_idx)

        return " ".join(generated_words)

def main():
    """Main function to demonstrate the FFNNLanguageModel."""
    # Parameters (match Rust example where possible)
    embedding_dim = 10
    hidden_dim = 20
    context_size = 2 # Predicts the 3rd word given the first 2

    # Training data
    corpus = "to be or not to be that is the question"
    words_in_corpus = corpus.split()
    vocab_size = len(set(words_in_corpus)) # Determine vocab size from corpus

    # Initialize model
    model = FFNNLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        context_size=context_size
    )

    # Train the model
    # Note: Rust example uses 1000 epochs which might be excessive for this small dataset/model
    # and a simplified update rule. Using fewer epochs here.
    model.train_model(corpus, epochs=200, learning_rate=0.01, batch_size=4)

    # Generate text
    start_context = ["to", "be"]
    generated_text = model.generate(start_context, num_words=10)
    print(f"\nGenerated text:\n{generated_text}")

    start_context_2 = ["that", "is"]
    generated_text_2 = model.generate(start_context_2, num_words=5)
    print(f"\nGenerated text:\n{generated_text_2}")

if __name__ == "__main__":
    main() 