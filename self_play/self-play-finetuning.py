"""
Self-Play Fine-Tuning for Simple LLM
Implements a self-training loop where the model learns from its own generated data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Tuple
import numpy as np
from collections import deque

class SelfPlayDataset(Dataset):
    """Dataset for storing self-play generated examples"""
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
        
    def add_example(self, input_ids, target_ids, reward):
        """Add a self-play example with its reward"""
        self.buffer.append({
            'input_ids': input_ids,
            'target_ids': target_ids,
            'reward': reward
        })
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]

class SimpleLLM(nn.Module):
    """Simple transformer-based language model"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        return self.lm_head(x)

class SelfPlayTrainer:
    """Manages self-play fine-tuning process"""
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = SelfPlayDataset()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def generate_response(self, prompt, max_length=50, temperature=0.7):
        """Generate a response for self-play"""
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_tensor[0].tolist())
    
    def evaluate_response(self, prompt, response, task_type='qa'):
        """Evaluate the quality of generated response"""
        if task_type == 'qa':
            # Simple heuristic evaluation for QA tasks
            reward = 0.0
            
            # Check if response addresses the question
            if '?' in prompt and '.' in response:
                reward += 0.3
            
            # Check response length
            words = response.split()
            if 5 <= len(words) <= 50:
                reward += 0.3
            
            # Check for specific content based on prompt
            if "who" in prompt.lower() and any(word.istitle() for word in words):
                reward += 0.4
            elif "when" in prompt.lower() and any(char.isdigit() for char in response):
                reward += 0.4
            elif "where" in prompt.lower() and any(word.istitle() for word in words):
                reward += 0.4
            
            return reward
        
        elif task_type == 'math':
            # Evaluation for math problems
            try:
                # Extract numbers from response
                result = [float(s) for s in response.split() if s.replace('.', '').isdigit()]
                if result:
                    # Check if result is reasonable (simple heuristic)
                    if 0 < result[0] < 1000:
                        return 1.0
                return 0.0
            except:
                return 0.0
        
        elif task_type == 'reasoning':
            # Evaluation for reasoning tasks
            steps = response.lower().count('therefore') + response.lower().count('because')
            coherence = 1.0 if len(response.split()) > 10 else 0.5
            return min(1.0, steps * 0.3 + coherence * 0.7)
        
        return 0.5  # Default reward
    
    def self_play_loop(self, num_iterations=100, batch_size=32):
        """Main self-play training loop"""
        prompts = [
            ("What is the capital of France?", "qa"),
            ("Who wrote Romeo and Juliet?", "qa"),
            ("When did World War II end?", "qa"),
            ("Calculate 15 + 27", "math"),
            ("What is 144 divided by 12?", "math"),
            ("If John has 5 apples and gives 2 to Mary, how many does he have left?", "reasoning"),
            ("Why do leaves change color in autumn?", "reasoning")
        ]
        
        for iteration in range(num_iterations):
            print(f"Self-play iteration {iteration + 1}/{num_iterations}")
            
            # Generate responses and evaluate them
            for prompt, task_type in random.sample(prompts, min(5, len(prompts))):
                # Generate multiple responses for the same prompt
                for _ in range(3):
                    response = self.generate_response(prompt)
                    reward = self.evaluate_response(prompt, response, task_type)
                    
                    # Add to dataset if reward is above threshold
                    if reward > 0.5:
                        input_ids = self.tokenizer.encode(prompt)
                        target_ids = self.tokenizer.encode(response)
                        self.dataset.add_example(input_ids, target_ids, reward)
            
            # Fine-tune on self-generated data
            if len(self.dataset) >= batch_size:
                self.fine_tune(num_epochs=1, batch_size=batch_size)
            
            # Periodic evaluation
            if (iteration + 1) % 10 == 0:
                self.evaluate_model()
    
    def fine_tune(self, num_epochs=1, batch_size=32):
        """Fine-tune model on self-generated data"""
        if len(self.dataset) < batch_size:
            return
        
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(ex['input_ids']) for ex in batch],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                ).to(self.device)
                
                target_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(ex['target_ids']) for ex in batch],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                ).to(self.device)
                
                rewards = torch.tensor([ex['reward'] for ex in batch]).to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1)
                )
                
                # Weight loss by rewards
                weighted_loss = (loss.view(target_ids.size()) * rewards.unsqueeze(1)).mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
                
                total_loss += weighted_loss.item()
            
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        test_prompts = [
            "What is the largest planet in our solar system?",
            "Calculate 123 + 456",
            "Why do we need to sleep?"
        ]
        
        self.model.eval()
        print("\nModel Evaluation:")
        for prompt in test_prompts:
            response = self.generate_response(prompt)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}\n")

# Example usage
class MockTokenizer:
    """Simple mock tokenizer for demonstration"""
    def __init__(self):
        self.vocab = {
            'What': 0, 'is': 1, 'the': 2, 'capital': 3, 'of': 4, 'France': 5,
            'Paris': 6, '.': 7, '?': 8, 'Who': 9, 'wrote': 10, 'Romeo': 11,
            'and': 12, 'Juliet': 13, 'Shakespeare': 14, '<pad>': 15, '<eos>': 16
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 15
        self.eos_token_id = 16
    
    def encode(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]
    
    def decode(self, ids):
        return ' '.join([self.inverse_vocab.get(id, '<unk>') for id in ids])

def main():
    # Initialize model and tokenizer
    vocab_size = 10000
    model = SimpleLLM(vocab_size=vocab_size, embed_dim=256, num_heads=8, num_layers=4)
    tokenizer = MockTokenizer()
    
    # Create trainer and run self-play
    trainer = SelfPlayTrainer(model, tokenizer, device='cpu')  # Use 'cuda' if available
    trainer.self_play_loop(num_iterations=10, batch_size=8)

if __name__ == "__main__":
    main()
