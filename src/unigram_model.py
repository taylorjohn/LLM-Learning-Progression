import random
from collections import defaultdict

class UnigramModel:
    """
    A simple Unigram language model implemented in Python.
    Learns word frequencies from text and generates text based on those frequencies.
    """
    def __init__(self):
        """Initializes an empty Unigram model."""
        # Using defaultdict for easier counting
        self.word_counts = defaultdict(int)
        self.total_words = 0

    def train(self, text: str):
        """
        Trains the model on the provided text corpus.

        Args:
            text: A string containing the training text.
        """
        print("Training model...")
        # Simple whitespace splitting
        words = text.split() 
        for word in words:
            self.word_counts[word] += 1
            self.total_words += 1
        print(f"Training complete. Vocab size: {len(self.word_counts)}, Total words: {self.total_words}")

    def generate(self, num_words: int) -> str:
        """
        Generates text based on the learned word frequencies.

        Args:
            num_words: The number of words to generate.

        Returns:
            A string containing the generated text.
        """
        if self.total_words == 0:
            return "[Model not trained]"

        output_words = []
        words = list(self.word_counts.keys())
        counts = list(self.word_counts.values())
        
        print(f"Generating {num_words} words...")
        # Using random.choices for weighted random sampling (more Pythonic)
        # Note: This is equivalent to the cumulative probability method in the Rust code
        if words: # Check if words list is not empty
            generated_sequence = random.choices(words, weights=counts, k=num_words)
            output_words.extend(generated_sequence)
        else:
             print("Warning: No words in vocabulary after training.")

        return " ".join(output_words)

def main():
    """Main function to demonstrate the UnigramModel."""
    model = UnigramModel()
    
    # Training data: famous quotes (same as Rust example)
    corpus = """To be or not to be that is the question
                  I think therefore I am
                  Ask not what your country can do for you ask what you can do for your country"""
    
    # Clean up corpus slightly (replace newlines with spaces for simple splitting)
    corpus = corpus.replace('\n', ' ').strip()
    
    model.train(corpus)
    
    # Generate text
    generated_text = model.generate(20) # Generate 20 words
    print(f"\nGenerated text:\n{generated_text}")

if __name__ == "__main__":
    main() 