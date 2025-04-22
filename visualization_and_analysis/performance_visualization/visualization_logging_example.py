import time
import random
import math

# Note: This script simulates logging metrics for visualization tools like
# TensorBoard or Weights & Biases (W&B). It does NOT actually use these libraries,
# as that would require installation and setup.

# --- Mock Logger ---

class MockLogger:
    """Simulates logging metrics to a visualization backend."""
    def __init__(self):
        self.logs = {}
        print("Initialized MockLogger.")
        print("In a real scenario, initialize TensorBoard SummaryWriter or wandb here.")

    def log_metric(self, metric_name: str, value: float, step: int):
        """Simulates logging a single metric value at a given step."""
        if metric_name not in self.logs:
            self.logs[metric_name] = []
        self.logs[metric_name].append((step, value))
        # In TensorBoard: writer.add_scalar(metric_name, value, step)
        # In W&B: wandb.log({metric_name: value}, step=step)
        # Print log message for demonstration
        print(f"  LOG: Step {step} - {metric_name} = {value:.4f}")

    def log_metrics(self, metrics_dict: dict, step: int):
        """Simulates logging multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
        # In W&B: wandb.log(metrics_dict, step=step)

    def close(self):
        """Simulates closing the logger."""
        print("\nClosing MockLogger.")
        # In TensorBoard: writer.close()
        # In W&B: wandb.finish()
        # You could potentially plot the logged data here using matplotlib
        # if needed for a simple local visualization.

# --- Mock Training Loop ---

def mock_training_step(step: int):
    """Simulates a single training step and returns mock metrics."""
    # Simulate loss calculation (e.g., decreasing loss)
    base_loss = 10.0 / math.log(step + 10)
    train_loss = base_loss + random.uniform(-0.1, 0.1)

    # Simulate accuracy calculation (e.g., increasing accuracy)
    base_acc = 1.0 - (15.0 / (step + 15))
    train_accuracy = base_acc + random.uniform(-0.02, 0.02)

    # Simulate validation (less frequent)
    val_loss = None
    val_accuracy = None
    if step % 10 == 0:
        val_loss = base_loss * 1.1 + random.uniform(-0.05, 0.05) # Slightly higher val loss
        val_accuracy = base_acc * 0.98 + random.uniform(-0.01, 0.01) # Slightly lower val acc

    # Simulate resource usage
    gpu_mem = 4500 + math.sin(step / 5) * 500 + random.uniform(-50, 50) # MB

    metrics = {
        "train/loss": train_loss,
        "train/accuracy": train_accuracy,
        "system/gpu_memory_mb": gpu_mem
    }
    if val_loss is not None:
        metrics["validation/loss"] = val_loss
        metrics["validation/accuracy"] = val_accuracy

    time.sleep(0.05) # Simulate computation time
    return metrics

# --- Main Simulation ---

if __name__ == "__main__":
    num_training_steps = 100

    # Initialize the logger
    logger = MockLogger()

    print(f"\nStarting mock training loop for {num_training_steps} steps...")

    for step in range(1, num_training_steps + 1):
        # Simulate one step of training
        metrics = mock_training_step(step)

        # Log the metrics
        logger.log_metrics(metrics, step=step)

    # Close the logger at the end
    logger.close()

    print("\nMock training finished.")
    print("In a real setup, you would now view these logged metrics")
    print("in your TensorBoard or Weights & Biases dashboard to see plots.") 