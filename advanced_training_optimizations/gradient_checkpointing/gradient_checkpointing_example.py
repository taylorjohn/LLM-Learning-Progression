import torch
import torch.nn as nn
import torch.nn.functional as F # Import F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import time
import random

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
has_cuda = torch.cuda.is_available()

print(f"Using device: {device}")

# --- Simple Model with Multiple Layers ---

class DeepNetwork(nn.Module):
    """A mock deep network with several linear layers."""
    def __init__(self, input_dim=512, hidden_dim=1024, num_layers=20):
        """
        Initializes the DeepNetwork.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of linear layers in the network.
        """
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.sequential_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 1) # Final output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through all layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.sequential_layers(x)
        return self.output_layer(x)

    def forward_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using gradient checkpointing on the sequential layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Apply checkpointing to the bulk of the layers
        # checkpoint_sequential splits the sequence into segments (default 2)
        # and recomputes intermediate activations during backward pass.
        x = checkpoint_sequential(self.sequential_layers, 2, x)
        return self.output_layer(x)

# --- Training Simulation Function ---

def simulate_training_step(model: DeepNetwork,
                           inputs: torch.Tensor,
                           targets: torch.Tensor,
                           optimizer: torch.optim.Optimizer,
                           use_checkpointing: bool = False):
    """
    Simulates a single training step (forward, loss, backward, optimizer step).

    Args:
        model (DeepNetwork): The model to train.
        inputs (torch.Tensor): Batch of input data.
        targets (torch.Tensor): Batch of target data.
        optimizer (torch.optim.Optimizer): The optimizer.
        use_checkpointing (bool): Whether to use the checkpointed forward pass.

    Returns:
        float: The loss value for the step.
    """
    model.train()
    optimizer.zero_grad()

    if use_checkpointing:
        outputs = model.forward_checkpointed(inputs)
    else:
        outputs = model(inputs)

    loss = F.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

# --- Main Demonstration ---

if __name__ == "__main__":
    input_dim = 512
    hidden_dim = 2048 # Larger hidden dim to emphasize memory usage
    num_layers = 50    # Deeper network to emphasize computation
    batch_size = 32
    num_batches = 10

    # Create dummy data
    dummy_inputs = [torch.randn(batch_size, input_dim).to(device) for _ in range(num_batches)]
    dummy_targets = [torch.randn(batch_size, 1).to(device) for _ in range(num_batches)]

    # --- Training Without Checkpointing ---
    print("\n--- Training without Gradient Checkpointing ---")
    model_no_cp = DeepNetwork(input_dim, hidden_dim, num_layers).to(device)
    optimizer_no_cp = optim.Adam(model_no_cp.parameters(), lr=1e-4)

    start_time_no_cp = time.time()
    start_mem_no_cp = 0.0
    if has_cuda:
        start_mem_no_cp = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    for i in range(num_batches):
        loss = simulate_training_step(model_no_cp, dummy_inputs[i], dummy_targets[i], optimizer_no_cp, use_checkpointing=False)
        print(f"  Batch {i+1}/{num_batches}, Loss: {loss:.4f}")

    end_time_no_cp = time.time()
    time_no_cp = end_time_no_cp - start_time_no_cp

    end_mem_no_cp = 0.0
    if has_cuda:
        end_mem_no_cp = torch.cuda.max_memory_allocated() # Peak memory
        print(f"Peak Memory (No CP): {end_mem_no_cp / 1024**2:.2f} MB")

    print(f"Time (No CP): {time_no_cp:.3f} seconds")

    # --- Training With Checkpointing ---
    # Need separate model instance
    print("\n--- Training with Gradient Checkpointing ---")
    model_cp = DeepNetwork(input_dim, hidden_dim, num_layers).to(device)
    optimizer_cp = optim.Adam(model_cp.parameters(), lr=1e-4)

    # Allow memory to be freed before next run if on GPU
    del model_no_cp
    del optimizer_no_cp
    if has_cuda:
        torch.cuda.empty_cache()
    time.sleep(1) # Allow time for cleanup

    start_time_cp = time.time()
    start_mem_cp = 0.0
    if has_cuda:
        start_mem_cp = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    for i in range(num_batches):
        loss = simulate_training_step(model_cp, dummy_inputs[i], dummy_targets[i], optimizer_cp, use_checkpointing=True)
        print(f"  Batch {i+1}/{num_batches}, Loss: {loss:.4f}")

    end_time_cp = time.time()
    time_cp = end_time_cp - start_time_cp

    end_mem_cp = 0.0
    if has_cuda:
        end_mem_cp = torch.cuda.max_memory_allocated()
        print(f"Peak Memory (With CP): {end_mem_cp / 1024**2:.2f} MB")
        mem_saving = (end_mem_no_cp - end_mem_cp) / end_mem_no_cp * 100 if end_mem_no_cp > 0 else 0
        print(f"Approx Memory Saving: {mem_saving:.1f}%")
    else:
        print("CUDA not available - Memory comparison not possible.")

    print(f"Time (With CP): {time_cp:.3f} seconds")

    time_overhead = (time_cp / time_no_cp - 1) * 100 if time_no_cp > 0 else 0
    print(f"Approx Time Overhead: {time_overhead:.1f}%")
    print("\nNote: Actual memory savings and time overhead depend heavily on model architecture, hardware, and checkpointing strategy.") 