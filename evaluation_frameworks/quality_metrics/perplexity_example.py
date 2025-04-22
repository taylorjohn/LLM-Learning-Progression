import torch
import torch.nn.functional as F
import math

def calculate_perplexity(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Calculates perplexity from model logits and target token IDs.

    Args:
        logits: Model output logits. Shape: (batch_size, sequence_length, vocab_size)
        targets: Target token IDs. Shape: (batch_size, sequence_length)
        ignore_index: Token ID to ignore in loss calculation (e.g., padding token).

    Returns:
        Perplexity score (float).
    """
    # Ensure logits and targets are on the same device
    logits = logits.to(targets.device)

    # Flatten the logits and targets for cross_entropy
    # Logits shape: (batch_size * sequence_length, vocab_size)
    # Targets shape: (batch_size * sequence_length)
    batch_size, sequence_length, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Calculate cross-entropy loss (average negative log-likelihood)
    # reduction='mean' averages the loss over the non-ignored tokens
    try:
        avg_neg_log_likelihood = F.cross_entropy(
            logits_flat, targets_flat, ignore_index=ignore_index, reduction='mean'
        )
    except IndexError as e:
        print(f"Error calculating cross_entropy: {e}")
        print(f"Logits shape: {logits_flat.shape}")
        print(f"Targets shape: {targets_flat.shape}")
        print(f"Min target: {targets_flat.min()}, Max target: {targets_flat.max()}")
        print(f"Vocab size from logits: {vocab_size}")
        # Handle potential index out of bounds if target IDs exceed vocab size
        # This often indicates a mismatch between model vocab and target data
        # Returning NaN or raising an error might be appropriate
        return float('nan')

    # Perplexity is the exponential of the average negative log-likelihood
    perplexity = torch.exp(avg_neg_log_likelihood)

    return perplexity.item()

# --- Example Usage ---

if __name__ == "__main__":
    # --- Setup ---
    vocab_size = 1000  # Size of the vocabulary
    batch_size = 2
    sequence_length = 10
    padding_token_id = 0 # Example padding token ID

    # --- Simulate Model Output ---
    # Normally comes from a language model
    # Shape: (batch_size, sequence_length, vocab_size)
    mock_logits = torch.randn(batch_size, sequence_length, vocab_size)

    # --- Simulate Target Data ---
    # Shape: (batch_size, sequence_length)
    # Ensure target IDs are within [0, vocab_size-1]
    mock_targets = torch.randint(1, vocab_size, (batch_size, sequence_length))

    # Add some padding tokens to simulate realistic data
    if sequence_length > 4:
        mock_targets[0, -2:] = padding_token_id # Pad last two tokens of first sequence
        mock_targets[1, -1] = padding_token_id  # Pad last token of second sequence

    print("--- Example Data ---")
    print(f"Logits shape: {mock_logits.shape}")
    print(f"Targets shape: {mock_targets.shape}")
    print(f"Example Targets (with padding={padding_token_id}):\n{mock_targets}")

    # --- Calculate Perplexity ---
    print("\n--- Calculating Perplexity ---")
    ppl = calculate_perplexity(mock_logits, mock_targets, ignore_index=padding_token_id)

    if not math.isnan(ppl):
        print(f"\nCalculated Perplexity: {ppl:.4f}")
        print("(Lower is better. Random logits usually result in high perplexity)")
    else:
        print("\nPerplexity calculation failed (likely due to target index out of bounds).")

    # --- Example with Perfect Prediction (for illustration) ---
    # Create logits where the probability of the target token is very high
    perfect_logits = torch.full_like(mock_logits, -10.0) # Low probability for non-targets
    for b in range(batch_size):
        for s in range(sequence_length):
            target_id = mock_targets[b, s].item()
            if target_id != padding_token_id:
                perfect_logits[b, s, target_id] = 10.0 # High probability for target

    print("\n--- Calculating Perplexity (Near-Perfect Prediction) ---")
    ppl_perfect = calculate_perplexity(perfect_logits, mock_targets, ignore_index=padding_token_id)

    if not math.isnan(ppl_perfect):
        print(f"\nCalculated Perplexity (Near-Perfect): {ppl_perfect:.4f}")
        print("(Perplexity approaches 1.0 as predictions become perfect)")
    else:
        print("\nPerfect perplexity calculation failed.") 