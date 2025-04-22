import random
from typing import List, Dict, Any

# --- Mock Model ---

class MockLLM:
    """A mock LLM that gives random answers."""
    def __init__(self, possible_answers: List[str]):
        self.possible_answers = possible_answers

    def predict(self, prompt: str) -> str:
        # Simulate generating an answer
        # In reality, this would involve tokenization, model inference, decoding
        return random.choice(self.possible_answers)

# --- Mock Benchmark Dataset ---

# Example format similar to a simple multiple-choice task like MMLU subset
def load_mock_benchmark_data() -> List[Dict[str, Any]]:
    """Loads a small, hardcoded benchmark dataset."""
    return [
        {
            "id": 1,
            "question": "What is the capital of France?",
            "choices": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer": "Paris"
        },
        {
            "id": 2,
            "question": "Which planet is known as the Red Planet?",
            "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
            "answer": "Mars"
        },
        {
            "id": 3,
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": "4"
        },
        {
            "id": 4,
            "question": "Who wrote Hamlet?",
            "choices": ["Charles Dickens", "Leo Tolstoy", "William Shakespeare", "Mark Twain"],
            "answer": "William Shakespeare"
        },
         {
            "id": 5,
            "question": "What is the chemical symbol for water?",
            "choices": ["O2", "H2O", "CO2", "NaCl"],
            "answer": "H2O"
        }
    ]

# --- Evaluation Logic ---

def format_prompt(task_data: Dict[str, Any]) -> str:
    """Formats the task data into a prompt for the LLM."""
    prompt = f"Question: {task_data['question']}\nChoices:\n"
    for i, choice in enumerate(task_data['choices']):
        prompt += f"{chr(ord('A') + i)}. {choice}\n"
    prompt += "Answer:"
    # Note: Actual benchmarking frameworks use more sophisticated few-shot prompting
    return prompt

def run_benchmark(model: MockLLM, benchmark_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Runs the model on the benchmark data and calculates metrics."""
    correct_predictions = 0
    total_predictions = 0

    for task in benchmark_data:
        prompt = format_prompt(task)
        # In a real scenario, you might need to parse the predicted choice
        # Here, we assume the mock model directly predicts one of the choices
        prediction = model.predict(prompt)

        print(f"Task {task['id']}: Question: {task['question'][:30]}... | Ground Truth: {task['answer']} | Prediction: {prediction}")

        # Simple accuracy check (assuming prediction matches a choice format)
        if prediction == task["answer"]:
            correct_predictions += 1
        total_predictions += 1

    accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0.0

    return {"accuracy": accuracy, "total_tasks": total_predictions}

# --- Main Execution ---

if __name__ == "__main__":
    print("Loading benchmark data...")
    benchmark_data = load_mock_benchmark_data()
    print(f"Loaded {len(benchmark_data)} benchmark tasks.")

    # Assume the model can predict any of the choices seen in the data
    all_possible_choices = list(set(choice for task in benchmark_data for choice in task['choices']))

    print("\nInitializing mock LLM...")
    mock_model = MockLLM(possible_answers=all_possible_choices)

    print("\nRunning benchmark...")
    results = run_benchmark(mock_model, benchmark_data)

    print("\n--- Benchmark Results ---")
    print(f"Total Tasks: {results['total_tasks']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print("\nNote: This is a simplified simulation. Real benchmarks use standardized datasets,")
    print("      complex prompting, robust metrics, and frameworks like lm-evaluation-harness.") 