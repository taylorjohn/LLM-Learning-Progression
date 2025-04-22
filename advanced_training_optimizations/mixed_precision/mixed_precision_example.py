import torch
import torch.nn as nn
import torch.optim as optim
import time

# Check if CUDA is available for AMP demonstration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
has_cuda = torch.cuda.is_available()

print(f"Using device: {device}")
if not has_cuda:
    print("\nWARNING: CUDA not available. Mixed precision (torch.cuda.amp) requires a CUDA-enabled GPU.")
    print("         The example will run on CPU without actual precision changes or speedups.\n")

# --- Simple Model Definition ---

class SimpleModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=10):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        # Multiple layers to simulate more computation
        for _ in range(4):
            x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# --- Training Setup ---

input_dim = 128
hidden_dim = 1024 # Larger hidden dim to simulate memory usage
output_dim = 10
batch_size = 64
num_batches = 50

# Create dummy data
dummy_data = [torch.randn(batch_size, input_dim).to(device) for _ in range(num_batches)]
dummy_targets = [torch.randint(0, output_dim, (batch_size,)).to(device) for _ in range(num_batches)]

criterion = nn.CrossEntropyLoss()

# --- Standard Precision Training (FP32) ---

print("\n--- Standard Precision Training (FP32) ---")
model_fp32 = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=1e-3)

start_time_fp32 = time.time()

for i in range(num_batches):
    inputs = dummy_data[i]
    targets = dummy_targets[i]

    optimizer_fp32.zero_grad()
    outputs = model_fp32(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_fp32.step()

    if (i + 1) % 10 == 0:
        print(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")

end_time_fp32 = time.time()
time_fp32 = end_time_fp32 - start_time_fp32
print(f"FP32 Training Time: {time_fp32:.3f} seconds")

# --- Mixed Precision Training (AMP) ---

print("\n--- Automatic Mixed Precision Training (AMP) ---")
model_amp = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
optimizer_amp = optim.Adam(model_amp.parameters(), lr=1e-3)

# Initialize GradScaler for loss scaling (essential for FP16 stability)
# enabled=has_cuda ensures it only runs if CUDA is available
scaler = torch.cuda.amp.GradScaler(enabled=has_cuda)

start_time_amp = time.time()

for i in range(num_batches):
    inputs = dummy_data[i]
    targets = dummy_targets[i]

    optimizer_amp.zero_grad()

    # Use autocast for the forward pass
    # Operations inside this context run in lower precision (FP16/BF16) where supported
    with torch.cuda.amp.autocast(enabled=has_cuda):
        outputs = model_amp(inputs)
        loss = criterion(outputs, targets)

    # Scale the loss
    # scaler.scale multiplies the loss by the scaler's current scale factor
    scaler.scale(loss).backward()

    # scaler.step runs the optimizer step (on unscaled gradients)
    # It automatically unscales gradients before the optimizer uses them
    scaler.step(optimizer_amp)

    # Update the scale factor for the next iteration
    scaler.update()

    if (i + 1) % 10 == 0:
        print(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")

end_time_amp = time.time()
time_amp = end_time_amp - start_time_amp
print(f"AMP Training Time: {time_amp:.3f} seconds")

print("\n--- Comparison ---")
if has_cuda:
    speedup = (time_fp32 / time_amp - 1) * 100
    print(f"AMP resulted in approximately {speedup:.1f}% speedup.")
    print("Note: Actual speedup and memory savings depend heavily on the specific model, hardware, and task.")
else:
    print("Cannot calculate speedup without CUDA. Timings reflect CPU execution.")
print("Memory savings are also a key benefit of AMP but not directly measured here.") 