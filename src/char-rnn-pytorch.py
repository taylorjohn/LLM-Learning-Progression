import torch
import torch.nn as nn
import torch.optim as optim
import string
import random

# Define the RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

# Preprocessing
all_characters = string.printable
n_characters = len(all_characters)
char_to_index = {char: i for i, char in enumerate(all_characters)}
index_to_char = {i: char for i, char in enumerate(all_characters)}

# Training data
data = """
The quick brown fox jumps over the lazy dog.
A stitch in time saves nine.
An apple a day keeps the doctor away.
Birds of a feather flock together.
Every cloud has a silver lining.
"""

# Generate text
def generate(model, start_char, length):
    model.eval()
    hidden = model.init_hidden(1)
    input = torch.tensor([[char_to_index[start_char]]], dtype=torch.long)
    output_string = start_char

    with torch.no_grad(): # Ensure no gradients are calculated during generation
        for _ in range(length):
            output, hidden = model(input, hidden)
            # Apply softmax to the output of the correct shape
            probabilities = nn.functional.softmax(output.view(-1), dim=0) # Use view(-1) 
            # Handle potential issues with multinomial on CPU with non-finite values
            if not torch.all(probabilities.isfinite()):
                print("Warning: Non-finite probabilities detected, using uniform distribution.")
                probabilities = torch.ones_like(probabilities) / probabilities.numel()
                
            predicted_index = torch.multinomial(probabilities, 1).item()
            predicted_char = index_to_char[predicted_index]
            output_string += predicted_char
            input = torch.tensor([[predicted_index]], dtype=torch.long)

    return output_string

# --- Main execution block ---
if __name__ == "__main__":
    # Model parameters
    hidden_size = 128
    n_layers = 1
    lr = 0.002
    n_epochs = 1000 # Reduced for faster testing during import
    print_every = 100

    # Create the model
    model = CharRNN(n_characters, hidden_size, n_characters, n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training (from import)...") # Add print statement
    # Training loop
    for epoch in range(n_epochs):
        hidden = model.init_hidden(1)
        model.zero_grad()
        loss = 0

        # Use enumerate to avoid potential issues with data.index()
        for i, char in enumerate(data[:-1]):
            target_char = data[i+1]
            input_tensor = torch.tensor([[char_to_index[char]]], dtype=torch.long)
            target_tensor = torch.tensor([[char_to_index[target_char]]], dtype=torch.long)
            
            # Ensure hidden state is detached if it comes from previous iteration
            hidden = hidden.detach()
            
            output, hidden = model(input_tensor, hidden)
            # Ensure output and target shapes match criterion expectation (Batch x Classes, Batch)
            loss += criterion(output.view(-1, n_characters), target_tensor.view(-1))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')

    print("\nGenerating text (from import)...") # Add print statement
    # Generate some text
    generated_output = generate(model, 'T', 100)
    print(generated_output)