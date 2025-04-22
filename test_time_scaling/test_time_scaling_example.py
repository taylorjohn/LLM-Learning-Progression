import time
import random

# --- Mock LLM Generation Functions ---

def generate_fast(prompt: str, max_tokens: int = 50) -> tuple[str, float]:
    """Simulates a faster, potentially lower-quality generation."""
    start_time = time.time()
    # Simulate work
    time.sleep(0.1 + random.uniform(0, 0.1))
    simulated_output = f"[Fast Response] {prompt[:30]}... {random.randint(1,100)}"
    truncated_output = simulated_output[:max_tokens]
    latency = time.time() - start_time
    print(f"    Used FAST generator (Latency: {latency:.3f}s)")
    return truncated_output, latency

def generate_slow(prompt: str, max_tokens: int = 50) -> tuple[str, float]:
    """Simulates a slower, potentially higher-quality generation."""
    start_time = time.time()
    # Simulate more work
    time.sleep(0.5 + random.uniform(0, 0.2))
    simulated_output = f"[Slow & Detailed Response] Input was: '{prompt}'. Analysis complete. Result: {random.uniform(100, 1000):.2f}"
    truncated_output = simulated_output[:max_tokens]
    latency = time.time() - start_time
    print(f"    Used SLOW generator (Latency: {latency:.3f}s)")
    return truncated_output, latency

# --- Adaptive Logic ---

def estimate_complexity(prompt: str) -> float:
    """Placeholder function to estimate prompt complexity (e.g., based on length)."""
    # Simple heuristic: longer prompts are more complex
    length_score = len(prompt) / 100.0 # Normalize roughly
    # Could add keyword checks, etc.
    complexity_score = min(1.0, length_score + random.uniform(-0.1, 0.1)) # Add noise
    print(f"    Estimated complexity score: {complexity_score:.2f}")
    return complexity_score

def adaptive_generate(prompt: str, max_tokens: int = 50, complexity_threshold: float = 0.4) -> tuple[str, float]:
    """
    Dynamically chooses generation method based on estimated complexity.
    """
    print(f"Processing prompt: '{prompt}'")
    complexity = estimate_complexity(prompt)

    if complexity < complexity_threshold:
        # Use the faster model for simple prompts
        print("    Complexity below threshold. Choosing FAST generator.")
        result, latency = generate_fast(prompt, max_tokens)
    else:
        # Use the slower, more detailed model for complex prompts
        print("    Complexity meets or exceeds threshold. Choosing SLOW generator.")
        result, latency = generate_slow(prompt, max_tokens)

    return result, latency

# --- Demonstration ---

if __name__ == "__main__":
    prompts = [
        "What is 2+2?",
        "Summarize the main points of the theory of relativity in detail.",
        "Define 'photosynthesis'.",
        "Write a short story about a robot discovering music.",
        "Capital of France?"
    ]

    total_latency = 0
    for p in prompts:
        print("-" * 40)
        output, lat = adaptive_generate(p, max_tokens=100)
        print(f"  Output: {output}")
        total_latency += lat
        print("-" * 40)
        time.sleep(0.5) # Pause for readability

    print(f"\nTotal processing time for all prompts: {total_latency:.3f}s")

    # Example of running only fast or slow for comparison
    print("\n--- Running all with FAST --- ")
    fast_latency = sum(generate_fast(p, 100)[1] for p in prompts)
    print(f"Total FAST latency: {fast_latency:.3f}s")

    print("\n--- Running all with SLOW --- ")
    slow_latency = sum(generate_slow(p, 100)[1] for p in prompts)
    print(f"Total SLOW latency: {slow_latency:.3f}s")

    print(f"\nAdaptive approach saved approx {slow_latency - total_latency:.3f}s vs always SLOW.")
    print(f"Adaptive approach cost approx {total_latency - fast_latency:.3f}s vs always FAST.") 