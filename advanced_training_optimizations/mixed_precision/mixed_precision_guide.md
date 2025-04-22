# Mixed Precision Training

Mixed precision training is a technique to accelerate deep learning training and reduce memory footprint by performing computations using lower-precision floating-point formats (like FP16 or BF16) where possible, while strategically using standard single-precision (FP32) where necessary to maintain model accuracy.

## Floating-Point Formats

*   **FP32 (Single Precision):** Standard 32-bit format. Offers a wide range and good precision. Default for most deep learning frameworks.
*   **FP16 (Half Precision):** 16-bit format. Significantly smaller memory footprint (half of FP32) and computations can be much faster on supported hardware (like NVIDIA Tensor Cores). However, it has a much smaller representable range and lower precision, making it susceptible to numerical overflow or underflow.
*   **BF16 (BFloat16):** Another 16-bit format. It maintains the same dynamic range as FP32 but with reduced precision. This makes it less prone to overflow/underflow than FP16 but potentially less precise for some operations.

## Why Use Mixed Precision?

1.  **Faster Computation:** Operations on 16-bit numbers (FP16/BF16) can be significantly faster on modern GPUs and TPUs equipped with specialized hardware units (e.g., Tensor Cores).
2.  **Reduced Memory Usage:** Using 16-bit formats halves the memory required for storing activations and gradients compared to FP32, allowing for larger models or batch sizes.
3.  **Lower Memory Bandwidth:** Moving less data between memory and compute units can further improve performance.

## Challenges and Solutions

The main challenge with using lower-precision formats, especially FP16, is their limited range and precision.

*   **Problem 1: Gradient Underflow:** Small gradient values computed during backpropagation might become zero when converted to FP16 because they fall below the smallest representable number. This stops weights from being updated.
*   **Problem 2: Activation Overflow/Underflow:** Intermediate activation values might exceed the maximum representable value (overflow) or become zero (underflow) in FP16.

To overcome these, mixed precision training typically employs:

1.  **Master Weights in FP32:** A primary copy of the model weights is kept in FP32. Updates are applied to these master weights to maintain precision over the course of training.
2.  **FP16/BF16 Computations:** The forward and backward passes are performed using FP16 or BF16 for speed and memory savings. Weights are cast to the lower precision format just before use.
3.  **Loss Scaling (Primarily for FP16):** To prevent gradient underflow, the loss value is scaled up by a chosen factor before the backward pass. This scales up the gradients proportionally, pushing them into the representable range of FP16. Before the optimizer updates the FP32 master weights, these gradients are scaled back down.

## Implementation (PyTorch Example)

PyTorch provides convenient tools for automatic mixed precision (AMP) through `torch.cuda.amp`:

*   **`autocast` context manager:** Automatically selects the appropriate precision (FP16/BF16 or FP32) for different operations within its scope during the forward pass.
*   **`GradScaler`:** Handles the loss scaling and unscaling process to prevent gradient underflow when using FP16.

This automates the process of casting tensors and managing loss scaling. 