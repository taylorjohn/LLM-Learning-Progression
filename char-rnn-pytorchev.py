import torch
import torch.nn as nn
import torch.optim as optim
import string
import random
import math

# ... [Previous code for CharRNN class, preprocessing, and training data remains the same]

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
# ... [Training loop code remains the same]

# Evaluation function
def evaluate(model, data, criterion):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        for char in data[:-1]:
            input = torch.tensor([[char_to_index[char]]], dtype=torch.long)
            target = torch.tensor([[char_to_index[data[data.index(char)+1]]]], dtype=torch.long)
            
            output, hidden = model(input, hidden)
            loss = criterion(output.squeeze(0), target.squeeze(0))
            total_loss += loss.item()
    
    avg_loss = total_loss / (len(data) - 1)
    perplexity = math.exp(avg_loss)
    return perplexity

# Text generation function
# ... [generate function remains the same]

# Train the model
for epoch in range(n_epochs):
    # ... [Training code remains the same]

    if (epoch + 1) % print_every == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')
        
        # Evaluate on training data
        train_perplexity = evaluate(model, data, criterion)
        print(f'Training Perplexity: {train_perplexity:.2f}')

# Prepare test data (a held-out portion not used in training)
test_data = """
The early bird catches the worm.
Actions speak louder than words.
Where there's smoke, there's fire.
"""

# Evaluate on test data
test_perplexity = evaluate(model, test_data, criterion)
print(f'\nFinal Test Perplexity: {test_perplexity:.2f}')

# Generate some text
print("\nGenerated Text:")
print(generate(model, 'T', 100))