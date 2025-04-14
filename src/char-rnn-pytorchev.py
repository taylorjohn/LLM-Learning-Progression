import torch
import torch.nn as nn
import torch.optim as optim
import string
import random
import math

# Assume CharRNN class, all_characters, n_characters, char_to_index, index_to_char, 
# and data are defined as in char-rnn-pytorch.py or imported if refactored.
# For standalone execution, these would need to be present here.

# --- Reusable Functions ---

# Evaluation function
def evaluate(model, eval_data, criterion):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    hidden = model.init_hidden(1)
    
    with torch.no_grad(): # Disable gradient calculation
        # Use enumerate for safer iteration
        for i, char in enumerate(eval_data[:-1]):
            target_char = eval_data[i+1]
            input_tensor = torch.tensor([[char_to_index[char]]], dtype=torch.long)
            target_tensor = torch.tensor([[char_to_index[target_char]]], dtype=torch.long)
            
            # Ensure hidden state is detached if it comes from previous iteration
            hidden = hidden.detach()
            
            output, hidden = model(input_tensor, hidden)
            # Ensure shapes match criterion expectation
            loss = criterion(output.view(-1, n_characters), target_tensor.view(-1))
            total_loss += loss.item()
    
    # Avoid division by zero if eval_data has length 1 or 0
    num_chars = len(eval_data) - 1
    if num_chars <= 0:
        return float('inf') 
        
    avg_loss = total_loss / num_chars
    perplexity = math.exp(avg_loss)
    return perplexity

# Text generation function (assume identical to the one in char-rnn-pytorch.py or imported)
def generate(model, start_char, length):
    # (Implementation as in char-rnn-pytorch.py, including with torch.no_grad() etc.)
    model.eval()
    hidden = model.init_hidden(1)
    input = torch.tensor([[char_to_index[start_char]]], dtype=torch.long)
    output_string = start_char

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input, hidden)
            probabilities = nn.functional.softmax(output.view(-1), dim=0)
            if not torch.all(probabilities.isfinite()):
                probabilities = torch.ones_like(probabilities) / probabilities.numel()
            predicted_index = torch.multinomial(probabilities, 1).item()
            predicted_char = index_to_char[predicted_index]
            output_string += predicted_char
            input = torch.tensor([[predicted_index]], dtype=torch.long)
    return output_string

# --- Main execution block ---
if __name__ == "__main__":
    # Assume CharRNN class and preprocessing vars (all_characters, etc.) are defined above here
    # Or imported if the code was refactored into a separate module.
    class CharRNN(nn.Module): # Placeholder if not imported
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
            
    all_characters = string.printable
    n_characters = len(all_characters)
    char_to_index = {char: i for i, char in enumerate(all_characters)}
    index_to_char = {i: char for i, char in enumerate(all_characters)}
    data = "The quick brown fox jumps over the lazy dog.\nA stitch in time saves nine." # Simplified data

    # Model parameters
    hidden_size = 128
    n_layers = 1
    lr = 0.002
    n_epochs = 500 # Reduced for faster testing
    print_every = 100

    # Create the model
    model = CharRNN(n_characters, hidden_size, n_characters, n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training (from import)...")
    # Train the model
    for epoch in range(n_epochs):
        hidden = model.init_hidden(1)
        model.zero_grad()
        loss = 0
        for i, char in enumerate(data[:-1]):
            target_char = data[i+1]
            input_tensor = torch.tensor([[char_to_index[char]]], dtype=torch.long)
            target_tensor = torch.tensor([[char_to_index[target_char]]], dtype=torch.long)
            hidden = hidden.detach()
            output, hidden = model(input_tensor, hidden)
            loss += criterion(output.view(-1, n_characters), target_tensor.view(-1))
        
        loss.backward()
        optimizer.step()
        current_loss = loss.item()

        if (epoch + 1) % print_every == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {current_loss:.4f}')
            
            # Evaluate on training data
            train_perplexity = evaluate(model, data, criterion)
            print(f'Training Perplexity: {train_perplexity:.2f}')

    # Prepare test data (a held-out portion not used in training)
    test_data = """
    An apple a day keeps the doctor away.
    Birds of a feather flock together.
    Every cloud has a silver lining.
    """

    print("\nEvaluating on test data...")
    # Evaluate on test data
    test_perplexity = evaluate(model, test_data, criterion)
    print(f'\nFinal Test Perplexity: {test_perplexity:.2f}')

    print("\nGenerating text (from import)...")
    # Generate some text
    generated_output = generate(model, 'T', 100)
    print(generated_output)