import re
from typing import List, Dict
from collections import Counter

def tokenize(text: str) -> List[str]:
    """
    Tokenizes the input text into a list of lowercase words, removing leading/trailing punctuation.

    This function splits the text by whitespace, converts each word to lowercase,
    and removes any non-alphanumeric characters from the beginning and end of each word.
    Empty strings resulting from this process are filtered out.

    Args:
        text: The input string to tokenize.

    Returns:
        A list of processed word tokens.
    """
    # Split by whitespace
    words = text.split()
    
    processed_tokens = []
    for word in words:
        # Convert to lowercase
        lower_word = word.lower()
        
        # Remove leading/trailing non-alphanumeric characters
        # Regex: ^[^a-z0-9]* matches any non-alphanumeric chars at the start
        # Regex: [^a-z0-9]*$ matches any non-alphanumeric chars at the end
        # We replace these matches with an empty string.
        # Note: This handles internal hyphens/apostrophes correctly as they are not at the start/end.
        stripped_word = re.sub(r'^[^a-z0-9]*|[^a-z0-9]*$', '', lower_word)
        
        # Add to list if not empty
        if stripped_word:
            processed_tokens.append(stripped_word)
            
    return processed_tokens

def count_words(text: str) -> Dict[str, int]:
    """
    Counts the occurrences of each word in the input text.

    Uses the `tokenize` function to process the text and then counts
    the frequency of each token.

    Args:
        text: The input string.

    Returns:
        A dictionary where keys are unique words (lowercase, punctuation-stripped)
        and values are their counts.
    """
    tokens = tokenize(text)
    return Counter(tokens) 