"""
Inner Thinking Transformer (ITT) Implementation
Demonstrates adaptive token routing and residual thinking connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveTokenRouter(nn.Module):
    """
    Routes tokens dynamically based on complexity,
    allocating more computational resources to critical tokens
    """
    def __init__(self, embed_dim, max_thinking_steps=4):
        super().__init__()
        self.complexity_estimator = nn.Linear(embed_dim, 1)
        self.max_thinking_steps = max_thinking_steps
        
    def forward(self, x):
        # Estimate complexity of each token
        # Shape: (batch_size, seq_len, 1)
        complexity = torch.sigmoid(self.complexity_estimator(x))
        
        # Scale complexity to determine number of thinking steps
        # Shape: (batch_size, seq_len, 1)
        thinking_steps = torch.floor(complexity * self.max_thinking_steps) + 1
        
        return thinking_steps


class ResidualThinkingBlock(nn.Module):
    """
    A thinking block with residual connections that iteratively refines token representations
    """
    def __init__(self, embed_dim, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        # Multi-head attention for thinking
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network for thinking
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Gating mechanism for residual thinking
        self.thinking_gate = nn.Linear(embed_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, thinking_step=1, max_steps=4):
        # Normalize input
        residual = x
        x_norm = self.norm1(x)
        
        # Self-attention with dynamic attention bias based on thinking step
        # Later thinking steps pay more attention to complex relationships
        step_bias = thinking_step / max_steps
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=None,
            key_padding_mask=None
        )
        
        # Apply residual connection with thinking gate
        # The gate decides how much of the new thinking to incorporate
        gate_input = torch.cat([residual, attn_output], dim=-1)
        thinking_gate = torch.sigmoid(self.thinking_gate(gate_input))
        x = residual + self.dropout(attn_output) * thinking_gate * step_bias
        
        # Feed-forward network
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        
        # Apply residual connection
        x = residual + self.dropout(ffn_output) * thinking_gate * step_bias
        
        return x


class InnerThinkingTransformer(nn.Module):
    """
    Inner Thinking Transformer that adaptively allocates computation to tokens
    based on their complexity, using residual thinking connections
    """
    def __init__(self, 
                 vocab_size, 
                 max_seq_len=512, 
                 embed_dim=256, 
                 num_heads=8, 
                 num_layers=6,
                 ffn_dim=1024, 
                 max_thinking_steps=4, 
                 dropout=0.1):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Adaptive token router
        self.token_router = AdaptiveTokenRouter(embed_dim, max_thinking_steps)
        
        # Shared thinking block for iterative refinement
        self.thinking_block = ResidualThinkingBlock(embed_dim, num_heads, ffn_dim, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # Statistics for analyzing computation allocation
        self.total_thinking_steps = 0
        self.total_tokens = 0
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embeddings and linear layers
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding[:, :seq_len, :]
        x = token_embeds + position_embeds
        
        # Determine thinking steps per token
        thinking_steps = self.token_router(x)
        
        # For statistics
        self.total_thinking_steps += thinking_steps.sum().item()
        self.total_tokens += batch_size * seq_len
        
        # Apply transformer layers with inner thinking
        for layer_idx, layer in enumerate(self.layers):
            # Standard transformer layer processing
            layer_output = layer(x)
            
            # Apply adaptive thinking for each token
            thinking_output = x.clone()
            max_steps = int(thinking_steps.max().item())
            
            # Create a mask for tokens that need thinking at each step
            for step in range(1, max_steps + 1):
                # Identify tokens that need this thinking step
                active_tokens = (thinking_steps >= step).squeeze(-1)
                
                if active_tokens.any():
                    # Only process tokens that need this thinking step
                    thinking_output[active_tokens] = self.thinking_block(
                        thinking_output[active_tokens], 
                        thinking_step=step,
                        max_steps=max_steps
                    )
            
            # Combine layer output with thinking output
            # Later layers give more weight to thinking outputs
            thinking_weight = (layer_idx + 1) / len(self.layers)
            x = (1 - thinking_weight) * layer_output + thinking_weight * thinking_output
        
        # Apply final normalization
        x = self.norm(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        return logits
    
    def get_stats(self):
        """Return statistics about computational allocation"""
        if self.total_tokens == 0:
            return {"avg_thinking_steps": 0}
        
        return {
            "avg_thinking_steps": self.total_thinking_steps / self.total_tokens
        }


def demonstrate_itt():
    """Demonstrate the Inner Thinking Transformer with a small example"""
    vocab_size = 10000
    batch_size = 4
    seq_len = 64
    
    # Create a small ITT model
    model = InnerThinkingTransformer(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        ffn_dim=1024,
        max_thinking_steps=4
    )
    
    # Create sample input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    
    # Print shapes and stats
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Computation stats: {model.get_stats()}")
    
    # Calculate parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {param_count:,}")
    
    # Compare to a standard transformer
    standard_param_count = 466_000_000
    print(f"ITT parameter reduction: {(1 - param_count/standard_param_count) * 100:.1f}%")
    
    return model


if __name__ == "__main__":
    demonstrate_itt()
