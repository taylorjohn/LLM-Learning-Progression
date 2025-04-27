import random
from collections import defaultdict, deque
from itertools import islice
import math

class NGramModelWithBackoff:
    """
    An N-Gram language model with Katz-style backoff implemented in Python.

    If a higher-order N-gram (e.g., trigram) hasn't been seen, it backs off
    to a lower-order N-gram (e.g., bigram, then unigram) to predict the next word.
    """
    def __init__(self, max_n: int):
        """Initializes the N-Gram model with backoff.

        Args:
            max_n: The maximum order of N-gram to consider (e.g., 3 for trigrams).
                   Models for 1-gram (unigram) up to max_n-gram will be trained.
        """
        if max_n < 1:
            raise ValueError("max_n must be at least 1.")
        self.max_n = max_n
        # models[i] stores the (i+1)-gram counts.
        # Format: List[defaultdict[tuple, defaultdict[str, int]]]
        # Example: models[2] (trigrams) -> { ('word1', 'word2'): {'word3': count} }
        # Example: models[0] (unigrams) -> { (): {'word1': count} }
        self.models = [defaultdict(lambda: defaultdict(int)) for _ in range(max_n)]
        # We also need context counts for probability calculation, especially for perplexity
        # context_counts[i] stores counts for (i+1)-gram contexts
        # Format: List[defaultdict[tuple, int]]
        self.context_counts = [defaultdict(int) for _ in range(max_n)]
        self.vocab = set()
        self._vocab_list = [] # Keep a list version for random sampling

    def train(self, text: str):
        """
        Trains all N-gram models from 1 to max_n on the provided text corpus.

        Args:
            text: A string containing the training text.
        """
        print(f"Training N-gram models up to N={self.max_n} with backoff...")
        words = text.split()
        if not words:
            print("Warning: Training text is empty.")
            return

        self.vocab.update(words)
        self._vocab_list = list(self.vocab)

        for n in range(1, self.max_n + 1):
            print(f"  Training {n}-grams...")
            window = deque(maxlen=n)
            num_ngrams = 0
            for word in words:
                window.append(word)
                if len(window) == n:
                    context = tuple(islice(window, 0, n - 1))
                    target_word = window[-1]
                    self.models[n-1][context][target_word] += 1
                    self.context_counts[n-1][context] += 1
                    num_ngrams += 1
            print(f"    Processed {num_ngrams} {n}-grams.")

        print(f"Training complete. Vocab size: {len(self.vocab)}")
        # print("Context counts summary:")
        # for i, counts in enumerate(self.context_counts):
        #      print(f"  {(i+1)}-grams: {len(counts)} distinct contexts")


    def _get_next_word_distribution(self, context: tuple[str, ...]) -> tuple[list[str], list[float] | list[int], bool]:
        """
        Calculates the probability distribution or counts of the next word using backoff.

        Args:
            context: A tuple of words representing the preceding context.

        Returns:
            A tuple containing:
            - list[str]: Possible next words.
            - list[float] or list[int]: Corresponding probabilities or raw counts.
            - bool: True if probabilities are returned, False if counts.
        """
        for n in range(self.max_n, 0, -1): # Check from longest N-gram down to unigram
            # Determine the relevant context suffix for this n-gram order
            if n == 1: # Unigram
                current_context_tuple = ()
            elif len(context) >= n - 1:
                current_context_tuple = context[-(n - 1):]
            else:
                continue # Context is too short for this n-gram order

            next_word_counts = self.models[n-1].get(current_context_tuple)

            if next_word_counts:
                # Found a match at this N-gram level
                possible_next_words = list(next_word_counts.keys())
                counts = list(next_word_counts.values())
                # Return counts directly for generation sampling
                # print(f"  [Debug] Matched {n}-gram context: {current_context_tuple} -> {len(possible_next_words)} options")
                return possible_next_words, counts, False # Return counts for random.choices

        # Final fallback: If no context matched (even unigram?), return uniform distribution over vocab
        # This shouldn't happen if train() was called unless vocab is empty
        # print("  [Debug] Backed off completely. Using uniform distribution.")
        if not self._vocab_list:
             return [], [], False # Should not happen if trained
        probabilities = [1.0 / len(self._vocab_list)] * len(self._vocab_list)
        return self._vocab_list, probabilities, True


    def generate(self, num_words: int, start_context: tuple[str, ...] | None = None) -> str:
        """
        Generates text using the backoff model.

        Args:
            num_words: The number of words to generate.
            start_context: Optional starting context (tuple of words). Length should ideally be
                           max_n-1, but the model will use the available suffix. If None,
                           chooses a random sequence.

        Returns:
            A string containing the generated text.
        """
        if not self.vocab:
            return "[Model not trained or empty vocabulary]"

        # Initialize the starting sequence
        current_context_list = []
        if start_context is None:
            # Need max_n-1 words to start predicting with highest order model
            context_len = min(self.max_n -1 , len(self._vocab_list))
            if context_len > 0:
                 current_context_list = random.sample(self._vocab_list, context_len)
            print(f"No start context provided, starting with random sequence: {tuple(current_context_list)}")
        else:
            current_context_list = list(start_context)
            print(f"Starting generation with context: {tuple(current_context_list)}")

        output_words = list(current_context_list) # Copy initial context

        print(f"Generating {num_words} words...")

        for _ in range(num_words):
            current_context_tuple = tuple(current_context_list) # Use the full available history

            next_words, weights, are_probabilities = self._get_next_word_distribution(current_context_tuple)

            if not next_words:
                print("Warning: No next words found (empty vocab?). Stopping generation.")
                break

            # Sample the next word
            # random.choices works with counts (weights) or probabilities
            next_word = random.choices(next_words, weights=weights, k=1)[0]
            output_words.append(next_word)

            # Update context for the next prediction (maintain a sliding window of max size max_n-1)
            current_context_list.append(next_word)
            if len(current_context_list) >= self.max_n:
                 current_context_list.pop(0) # Keep context size manageable

        return " ".join(output_words)

    def _get_word_probability(self, word: str, context: tuple[str, ...]) -> float:
        """Calculates the probability P(word | context) using backoff."""
        for n in range(self.max_n, 0, -1):
            if n == 1:
                current_context_tuple = ()
            elif len(context) >= n - 1:
                current_context_tuple = context[-(n - 1):]
            else:
                continue

            next_word_counts = self.models[n-1].get(current_context_tuple)
            if next_word_counts:
                total_context_count = self.context_counts[n-1].get(current_context_tuple, 0)
                if total_context_count > 0:
                    word_count = next_word_counts.get(word, 0)
                    probability = word_count / total_context_count
                    if probability > 0.0: # Found a non-zero probability
                        return probability
                # If probability is 0 at this level, continue backing off

        # Backed off completely, estimate probability using uniform distribution over vocab
        # (Could use unigram frequency if available, but Rust version uses uniform)
        # Avoid division by zero if vocab is empty
        return 1.0 / len(self.vocab) if self.vocab else 0.0


    def perplexity(self, text: str) -> float:
        """
        Calculates the perplexity of the model on a given text.
        Perplexity = exp(-1/N * sum(log P(word_i | context_i)))

        Args:
            text: The test text string.

        Returns:
            The perplexity score (lower is better). Returns float('inf') if calculation fails.
        """
        print(f"\nCalculating perplexity...")
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return float('inf')

        log_likelihood = 0.0
        # Use a deque for the context window
        context_window = deque(maxlen=self.max_n - 1)

        for i, word in enumerate(words):
            current_context = tuple(context_window)
            probability = self._get_word_probability(word, current_context)

            # Add small epsilon for numerical stability (avoid log(0))
            log_likelihood += math.log(probability + 1e-10)

            # Update context window for the next word
            context_window.append(word)

        # Perplexity calculation
        cross_entropy = -log_likelihood / n_words
        perplexity = math.exp(cross_entropy)
        return perplexity

def main():
    """Main function to demonstrate the NGramModelWithBackoff."""
    max_n_value = 3 # Example: Trigram model with backoff to bigram/unigram
    model = NGramModelWithBackoff(max_n=max_n_value)

    # Training data: famous quotes (same as Rust example)
    corpus = """To be or not to be that is the question
                  I think therefore I am
                  Ask not what your country can do for you ask what you can do for your country"""

    # Clean up corpus slightly
    corpus = corpus.replace('\\n', ' ').strip()

    model.train(corpus)

    # Generate text (random start)
    generated_text_random = model.generate(30) # Generate 30 words
    print(f"\nGenerated text (random start):\n{generated_text_random}")

    # Generate text (specific start)
    start = ("what", "your") # Example start for n=3
    generated_text_start = model.generate(30, start_context=start)
    print(f"\nGenerated text (starting with {start}):\n{generated_text_start}")

    # Calculate perplexity on a test sentence
    test_text = "to be or not to be" # Use lowercase to match training potentially
    # Preprocess test text same way as training (simple split)
    # Note: The Rust code doesn't lowercase, so we might get OOV if case differs.
    # For simplicity, let's test with words known to be in the vocab.
    test_text_in_vocab = " ".join(w for w in test_text.split() if w in model.vocab)
    if not test_text_in_vocab:
         print(f"\nWarning: Test text '{test_text}' contains no words from the training vocabulary.")
         perplexity = float('inf')
    else:
         print(f"\nCalculating perplexity on test text (words in vocab): '{test_text_in_vocab}'")
         perplexity = model.perplexity(test_text_in_vocab)

    if perplexity == float('inf'):
        print(f"Perplexity calculation failed or resulted in infinity.")
    else:
        print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main() 