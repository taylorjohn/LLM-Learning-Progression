"""
Implements the Tokenizer class for Byte-Pair Encoding (BPE).
"""

import re
import json # Added for from_files
import ast
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

# Helper for vocab loading if needed (handling byte tokens like <0x80>)
# For now, we assume standard UTF-8 encoding covers the string keys in vocab.json
# def _parse_vocab_key(key_str: str) -> bytes:
#     # Implement logic here if needed based on vocab format
#     return key_str.encode('utf-8')

# Helper function to decode string representations from vocab/merges
def _parse_token_string(token_str: str) -> bytes:
    try:
        # Decode standard Python string escapes (like \n, \u0120)
        decoded_escapes = token_str.encode('utf-8').decode('unicode_escape')
        # Re-encode to UTF-8 bytes
        return decoded_escapes.encode('utf-8')
    except UnicodeDecodeError:
        # This might happen if the string is already raw bytes or has invalid escapes
        # Fallback to direct UTF-8 encoding
        try:
            return token_str.encode('utf-8')
        except UnicodeEncodeError:
            raise ValueError(f"Could not encode token string '{token_str}' to UTF-8.")
    except Exception as e:
        print(f"Warning: Error parsing token string '{token_str}': {e}. Falling back to direct UTF-8 encoding.")
        try:
            return token_str.encode('utf-8')
        except UnicodeEncodeError:
            raise ValueError(f"Could not encode token string '{token_str}' after fallback.")

class Tokenizer:
    """
    Byte-Pair Encoding Tokenizer.

    Handles vocabulary loading, merges, special tokens, encoding, and decoding.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Constructs a Tokenizer.

        Args:
            vocab: A dictionary mapping token IDs (int) to their byte sequences (bytes).
                   Must include mappings for individual bytes (0-255).
            merges: A list of byte-pair merges, ordered by priority.
            special_tokens: An optional list of special tokens (strings) to handle.
        """
        self.vocab: Dict[int, bytes] = vocab
        # Convert merges list to a tuple for immutability/hashing if needed later
        self.merges: Tuple[Tuple[bytes, bytes], ...] = tuple(merges)
        # Ensure special_tokens is a list, even if None is passed
        _special_tokens_list = special_tokens if special_tokens is not None else [] 

        # --- Precompute data structures ---

        # Inverse vocab: map byte sequences back to token IDs
        self.inverse_vocab: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Validate vocab: Ensure all single bytes (0-255) are present
        for i in range(256):
            if i not in self.vocab:
                raise ValueError(f"Vocabulary missing required byte token: {i}")
            if len(self.vocab[i]) != 1:
                raise ValueError(
                    f"Token ID {i} maps to {self.vocab[i]!r}, expected single byte."
                )
            if self.vocab[i] != bytes([i]):
                raise ValueError(
                    f"Token ID {i} maps to {self.vocab[i]!r}, expected {bytes([i])!r}."
                )

        # Precompute merge ranks for O(1) lookup during encoding
        # Lower rank means higher priority (applied earlier)
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        # --- Handle Special Tokens ---
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}
        next_special_id = len(self.vocab)
        for token_str in _special_tokens_list:
            token_bytes = token_str.encode("utf-8")
            if token_bytes in self.inverse_vocab:
                token_id = self.inverse_vocab[token_bytes]
                print(f"Warning: Special token '{token_str}' already exists in vocab with ID {token_id}.")
            else:
                # Assign a new ID if it doesn't exist
                while next_special_id in self.vocab:
                    next_special_id += 1 # Ensure we don't overwrite existing IDs
                token_id = next_special_id
                self.vocab[token_id] = token_bytes
                self.inverse_vocab[token_bytes] = token_id
                next_special_id += 1
            
            if token_str in self.special_tokens:
                 print(f"Warning: Duplicate special token '{token_str}' provided.")
            self.special_tokens[token_str] = token_id
            self.inverse_special_tokens[token_id] = token_str

        # Create regex pattern for finding special tokens OR sequences of non-special bytes
        if self.special_tokens:
            # Escape special characters in tokens for regex safety
            escaped_tokens = [re.escape(t) for t in self.special_tokens.keys()]
            # Sort by length descending to match longest tokens first
            escaped_tokens.sort(key=len, reverse=True)
            # Pattern matches special tokens OR any sequence of bytes not part of a special token
            special_pattern = "(" + "|".join(escaped_tokens) + ")"
            self.split_pattern = re.compile(special_pattern)
        else:
            # If no special tokens, the pattern isn't needed for splitting (handled by byte encoding)
            self.split_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        Class method to construct a Tokenizer from vocabulary and merge files.

        Args:
            vocab_filepath: Path to the vocabulary file (e.g., vocab.json).
                            Expected format: JSON dictionary mapping token strings to IDs.
            merges_filepath: Path to the merges file (e.g., merges.txt).
                             Expected format: Each line contains two space-separated tokens
                             representing a merge rule (e.g., "G" "l").
            special_tokens: An optional list of special tokens (strings).

        Returns:
            A Tokenizer instance.
        """
        # 1. Read vocab file (JSON: string -> ID)
        try:
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                str_token_to_id: Dict[str, int] = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_filepath}")
        except json.JSONDecodeError:
            raise ValueError(f"Could not decode JSON from vocab file: {vocab_filepath}")

        # 2. Convert to the required format (ID -> bytes)
        vocab: Dict[int, bytes] = {}
        for token_str, token_id in str_token_to_id.items():
            token_bytes = _parse_token_string(token_str)
            if token_id in vocab:
                 print(f"Warning: Duplicate token ID {token_id} found in vocab file.")
            vocab[token_id] = token_bytes

        # 3. Read merges file (text: "token1_str token2_str" per line)
        merges: List[Tuple[bytes, bytes]] = []
        try:
            with open(merges_filepath, 'r', encoding='utf-8') as f:
                # Skip potential header line (like version info in some formats)
                next(f, None) 
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"): # Skip empty lines/comments
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        print(f"Warning: Invalid merge rule format on line {i+2} in {merges_filepath}: {line}")
                        continue
                    
                    # Convert string parts to bytes using the same helper
                    try:
                        part1_bytes = _parse_token_string(parts[0])
                        part2_bytes = _parse_token_string(parts[1])
                    except ValueError as e:
                        print(f"Warning: {e} on line {i+2}: {line}")
                        continue

                    merges.append((part1_bytes, part2_bytes))
        except FileNotFoundError:
            raise FileNotFoundError(f"Merges file not found: {merges_filepath}")

        # 4. Call the constructor with loaded data
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a sequence of token IDs.

        Args:
            text: The input string.

        Returns:
            A list of integer token IDs.
        """
        final_ids: List[int] = []

        if self.split_pattern:
            # Split the text by special tokens, keeping the delimiters
            chunks = self.split_pattern.split(text)
            # Filter out empty strings that can result from splitting
            chunks = [chunk for chunk in chunks if chunk]
        else:
            # No special tokens, treat the whole text as one chunk
            chunks = [text]

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                # It's a special token, get its ID directly
                final_ids.append(self.special_tokens[chunk])
            else:
                # It's a regular text chunk, needs BPE processing
                # 1. Convert chunk to bytes
                chunk_bytes = chunk.encode("utf-8")

                # 2. Initial sequence of single-byte token IDs
                ids = [self.inverse_vocab[bytes([b])] for b in chunk_bytes]

                # 3. Iteratively apply BPE merges
                while len(ids) >= 2:
                    # Find the best merge pair (lowest rank) in the current sequence
                    best_pair_info = None # (rank, merge_pair_bytes, index_in_ids)
                    for i in range(len(ids) - 1):
                        pair_ids = (ids[i], ids[i+1])
                        # Get the byte representation of the pair
                        byte1 = self.vocab.get(pair_ids[0])
                        byte2 = self.vocab.get(pair_ids[1])
                        
                        if byte1 is None or byte2 is None: 
                            # Should not happen if vocab is valid, but safety check
                            continue 
                        
                        current_pair_bytes = (byte1, byte2)
                        rank = self.merge_ranks.get(current_pair_bytes)

                        if rank is not None:
                            # Found a potential merge
                            if best_pair_info is None or rank < best_pair_info[0]:
                                best_pair_info = (rank, current_pair_bytes, i)
                    
                    if best_pair_info is None:
                        # No more possible merges in this sequence
                        break
                    
                    # Apply the best merge found
                    _rank, merge_pair, index = best_pair_info
                    merged_bytes = merge_pair[0] + merge_pair[1]
                    merged_id = self.inverse_vocab.get(merged_bytes)

                    if merged_id is None:
                        # This indicates an inconsistency between merges and vocab
                        print(f"Warning: Merged bytes {merged_bytes!r} not found in vocab.")
                        # Skip this merge and hope for the best?
                        # Or raise an error? For now, let's break to avoid infinite loops.
                        # A robust implementation might need better error handling here.
                        break 

                    # Create new list of IDs with the merge applied
                    ids = ids[:index] + [merged_id] + ids[index+2:]
                
                # Add the final IDs for this chunk to the result
                final_ids.extend(ids)
                
        return final_ids

    def encode_iterable(self, text_iterable: Iterable[str]) -> Iterator[int]:
        """
        Encodes an iterable of strings into a lazy iterator of token IDs.

        Useful for large datasets that cannot fit into memory.

        Args:
            text_iterable: An iterable (e.g., file handle, list) yielding strings.

        Returns:
            An iterator yielding integer token IDs.
        """
        # Iterate through the input strings
        for text_chunk in text_iterable:
            # Encode each chunk using the main encode method
            encoded_ids = self.encode(text_chunk)
            # Yield each token ID from the encoded list
            for token_id in encoded_ids:
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back into a string.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.
        """
        # Map token IDs to byte sequences using vocab
        token_bytes_list = []
        for token_id in ids:
            # Look up in main vocab first
            token_bytes = self.vocab.get(token_id)
            if token_bytes is None:
                # Check if it's a special token ID that might have been added
                # (Technically covered by self.vocab, but good practice)
                special_token_str = self.inverse_special_tokens.get(token_id)
                if special_token_str is not None:
                    token_bytes = special_token_str.encode("utf-8")
                else:
                     # Or handle unknown tokens if necessary (e.g., raise error or use replacement)
                    print(f"Warning: Unknown token ID encountered during decoding: {token_id}")
                    # Option: Replace with a known placeholder or skip
                    # For now, let's use the Unicode replacement character bytes
                    token_bytes = b'\xef\xbf\xbd' 

            token_bytes_list.append(token_bytes)

        # Concatenate byte sequences
        full_byte_sequence = b"".join(token_bytes_list)

        # Decode the resulting bytes to a UTF-8 string, replacing errors
        try:
            text = full_byte_sequence.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            # This shouldn't happen with errors='replace', but as a fallback:
            print("Warning: Final decoding failed despite errors='replace'.")
            text = "" # Or some other fallback representation
        return text

# Optional: Add helper functions or constants if needed outside the class 