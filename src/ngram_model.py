import random
from collections import defaultdict, deque
from itertools import islice

class NGramModel:
    """
    A simple N-Gram language model implemented in Python.
    Learns frequencies of word sequences (N-grams) and generates text based on
    conditional probabilities P(word_n | word_1, ..., word_{n-1}).
    """
    def __init__(self, n: int):
        """Initializes an empty N-Gram model.

        Args:
            n: The order of the N-gram (e.g., 2 for bigram, 3 for trigram).
        """
        if n < 2:
            raise ValueError("N must be at least 2 for N-gram models.")
        self.n = n
        # Stores counts: counts[ (word1, ..., word_{n-1}) ][ word_n ] = count
        # Using tuple for the context key as dict keys must be hashable
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        # Stores counts of contexts: context_counts[(word1, ..., word_{n-1})] = total_count
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def train(self, text: str):
        """
        Trains the model on the provided text corpus.

        Args:
            text: A string containing the training text.
        """
        print(f"Training {self.n}-gram model...")
        words = text.split()
        if len(words) < self.n:
            print(f"Warning: Training text has fewer than {self.n} words. Cannot train N-grams.")
            return

        self.vocab.update(words)

        # Use deque for efficient sliding window
        window = deque(maxlen=self.n)
        for word in words:
            window.append(word)
            if len(window) == self.n:
                context = tuple(islice(window, 0, self.n - 1))
                target_word = window[-1]
                self.ngram_counts[context][target_word] += 1
                self.context_counts[context] += 1

        print(f"Training complete. Vocab size: {len(self.vocab)}, Contexts observed: {len(self.context_counts)}")

    def generate(self, num_words: int, start_context: tuple[str, ...] | None = None) -> str:
        """
        Generates text based on the learned N-gram frequencies.

        Args:
            num_words: The number of words to generate (including the initial context).
            start_context: Optional starting context (tuple of n-1 words).
                           If None, chooses a random sequence from the vocabulary.

        Returns:
            A string containing the generated text.
        """
        if not self.vocab:
            return "[Model not trained or empty vocabulary]"
        
        if len(self.vocab) < self.n - 1:
             return f"[Vocabulary size ({len(self.vocab)}) too small for context size {self.n-1}]"

        # Initialize the starting sequence
        if start_context is None or len(start_context) != self.n - 1 or not all(w in self.vocab for w in start_context):
            print(f"No valid start context provided, choosing random {self.n-1} words...")
            current_context_list = random.sample(list(self.vocab), self.n - 1)
        else:
            current_context_list = list(start_context)

        output_words = list(current_context_list)
        print(f"Generating {num_words} words starting with context: {tuple(current_context_list)}...")

        for _ in range(num_words - (self.n - 1)): # Generate remaining words
            current_context_tuple = tuple(current_context_list)
            
            next_word_options = list(self.ngram_counts.get(current_context_tuple, {}).keys())
            
            if next_word_options:
                # Sample from the observed next words for this context
                next_word_weights = list(self.ngram_counts[current_context_tuple].values())
                next_word = random.choices(next_word_options, weights=next_word_weights, k=1)[0]
            else:
                # Fallback: Context not seen during training, choose a random word
                # print(f"Warning: Context {current_context_tuple} not seen. Choosing random word.")
                next_word = random.choice(list(self.vocab))
            
            output_words.append(next_word)
            # Slide the context window
            current_context_list.pop(0)
            current_context_list.append(next_word)

        return " ".join(output_words)

def main():
    """Main function to demonstrate the NGramModel."""
    n_value = 3 # Example: Trigram
    model = NGramModel(n=n_value)
    
    # Training data: famous quotes (same as Rust example)
    corpus = """To be or not to be that is the question
                  I think therefore I am
                  Ask not what your country can do for you ask what you can do for your country"""
    
    # Clean up corpus slightly
    corpus = corpus.replace('\n', ' ').strip()
    
    model.train(corpus)
    
    # Generate text (e.g., starting with a specific context if possible)
    start = ("ask", "what") # Example start for n=3
    generated_text = model.generate(25, start_context=start)
    print(f"\nGenerated text (starting with {start}):\n{generated_text}")
    
    # Generate text (random start)
    generated_text_random = model.generate(25)
    print(f"\nGenerated text (random start):\n{generated_text_random}")

if __name__ == "__main__":
    main() 