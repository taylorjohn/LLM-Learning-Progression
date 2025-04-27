import random
from collections import defaultdict, Counter

class BigramModel:
    """
    A simple Bigram language model implemented in Python.
    Learns word pair frequencies and generates text based on conditional probabilities.
    """
    def __init__(self):
        """Initializes an empty Bigram model."""
        # Stores counts of word pairs: counts[word1][word2] = count(word1, word2)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        # Stores counts of individual words (for context probability): context_counts[word1] = count(word1, ...)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def train(self, text: str):
        """
        Trains the model on the provided text corpus.

        Args:
            text: A string containing the training text.
        """
        print("Training model...")
        # Simple whitespace splitting
        words = text.split() 
        if not words:
            print("Warning: Training text is empty.")
            return
            
        self.vocab.update(words)
        
        # Count contexts and bigrams
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i+1]
            self.bigram_counts[word1][word2] += 1
            self.context_counts[word1] += 1
            
        # Add count for the last word's context (though it won't predict anything)
        self.context_counts[words[-1]] += 1 

        print(f"Training complete. Vocab size: {len(self.vocab)}, Contexts: {len(self.context_counts)}")

    def _get_next_word_distribution(self, context_word: str) -> tuple[list[str], list[float]]:
        """ Calculates the probability distribution of the next word given the context. """
        next_word_counts = self.bigram_counts.get(context_word, None)
        if not next_word_counts:
            # Fallback: If no bigrams start with this word, return uniform distribution over vocab?
            # Or choose a random word? Let's choose randomly for simplicity matching Rust fallback.
            return list(self.vocab), [1.0 / len(self.vocab)] * len(self.vocab) # Effectively uniform
            
        possible_next_words = list(next_word_counts.keys())
        counts = list(next_word_counts.values())
        total_context_count = self.context_counts[context_word] # Should be sum(counts)
        
        probabilities = [count / total_context_count for count in counts]
        
        return possible_next_words, probabilities

    def generate(self, num_words: int, start_word: str | None = None) -> str:
        """
        Generates text based on the learned bigram frequencies.

        Args:
            num_words: The number of words to generate.
            start_word: Optional starting word. If None, chooses a random word from vocab.

        Returns:
            A string containing the generated text.
        """
        if not self.vocab:
            return "[Model not trained or empty vocabulary]"

        if start_word is None or start_word not in self.vocab:
            current_word = random.choice(list(self.vocab))
            print(f"No valid start word provided, starting with random word: '{current_word}'")
        else:
            current_word = start_word
            
        output_words = [current_word]
        print(f"Generating {num_words} words starting with '{current_word}'...")

        for _ in range(1, num_words):
            next_words, probabilities = self._get_next_word_distribution(current_word)
            
            if not next_words: # Should only happen if vocab was empty initially
                print(f"Warning: Could not find next word distribution for '{current_word}'. Stopping.")
                break
                
            # Sample the next word based on the calculated distribution
            # Need to handle the case where probabilities might not sum perfectly to 1 due to float issues
            # random.choices handles weights, don't need to normalize if using counts directly
            next_word_options = list(self.bigram_counts.get(current_word, {}).keys())
            if next_word_options:
                 next_word_weights = list(self.bigram_counts[current_word].values())
                 next_word = random.choices(next_word_options, weights=next_word_weights, k=1)[0]
            else:
                 # Fallback if no bigram starts with current_word
                 next_word = random.choice(list(self.vocab))
            
            output_words.append(next_word)
            current_word = next_word

        return " ".join(output_words)

def main():
    """Main function to demonstrate the BigramModel."""
    model = BigramModel()
    
    # Training data: famous quotes (same as Rust example)
    corpus = """To be or not to be that is the question
                  I think therefore I am
                  Ask not what your country can do for you ask what you can do for your country"""
    
    # Clean up corpus slightly
    corpus = corpus.replace('\n', ' ').strip()
    
    model.train(corpus)
    
    # Generate text (e.g., starting with "I")
    generated_text = model.generate(20, start_word="I") 
    print(f"\nGenerated text (starting with 'I'):\n{generated_text}")
    
    # Generate text (random start)
    generated_text_random = model.generate(20)
    print(f"\nGenerated text (random start):\n{generated_text_random}")

if __name__ == "__main__":
    main() 