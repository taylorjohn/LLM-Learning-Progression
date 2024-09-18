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

# Model parameters
hidden_size = 128
n_layers = 1
lr = 0.002
n_epochs = 1000
print_every = 100

# Create the model
model = CharRNN(n_characters, hidden_size, n_characters, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(n_epochs):
    hidden = model.init_hidden(1)
    model.zero_grad()
    loss = 0

    for char in data[:-1]:
        input = torch.tensor([[char_to_index[char]]], dtype=torch.long)
        target = torch.tensor([[char_to_index[data[data.index(char)+1]]]], dtype=torch.long)
        
        output, hidden = model(input, hidden)
        loss += criterion(output.squeeze(0), target.squeeze(0))

    loss.backward()
    optimizer.step()

    if (epoch + 1) % print_every == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')

# Generate text
def generate(model, start_char, length):
    model.eval()
    hidden = model.init_hidden(1)
    input = torch.tensor([[char_to_index[start_char]]], dtype=torch.long)
    output_string = start_char

    for _ in range(length):
        output, hidden = model(input, hidden)
        probabilities = nn.functional.softmax(output.squeeze(), dim=0)
        predicted_index = torch.multinomial(probabilities, 1).item()
        predicted_char = index_to_char[predicted_index]
        output_string += predicted_char
        input = torch.tensor([[predicted_index]], dtype=torch.long)

    return output_string

# Generate some text
print(generate(model, 'T', 100))