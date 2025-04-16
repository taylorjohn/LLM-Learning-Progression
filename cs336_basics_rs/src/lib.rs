// Placeholder for library functions corresponding to cs336_basics Python modules

use std::collections::HashMap;

/// Adds one to the given number.
///
/// # Arguments
/// * `x` - The number to add one to.
///
/// # Returns
/// The number plus one.
pub fn add_one(x: i32) -> i32 {
    x + 1
}

/// Tokenizes the input text into a vector of lowercase words.
///
/// This function splits the input text by whitespace, trims leading and trailing
/// non-alphanumeric characters from each word, converts the remaining characters
/// to lowercase, and filters out any empty strings that result from the trimming process.
///
/// # Arguments
/// * `text` - The input string slice to tokenize.
///
/// # Returns
/// A vector of strings, where each string is a lowercase word with punctuation removed
/// from the ends.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            // Trim characters that are NOT alphanumeric from both ends
            let trimmed_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            // Convert to lowercase
            trimmed_word.to_lowercase()
        })
        // Filter out empty strings that might result from trimming
        .filter(|s| !s.is_empty())
         // Retain only tokens that are alphanumeric or contain internal hyphens/apostrophes
         // This step might be redundant given the trimming logic, but can act as a safeguard
         // Let's refine this: keep if it contains at least one alphanumeric char
         .filter(|s| s.chars().any(|c| c.is_alphanumeric()))
        .collect()
}

/// Counts the occurrences of each word in the input text.
///
/// This function uses the `tokenize` function to split the text into words,
/// handling punctuation and case, and then counts the frequency of each unique word.
///
/// # Arguments
/// * `text` - The input string slice.
///
/// # Returns
/// A HashMap where keys are unique words (lowercase) and values are their counts.
pub fn count_words(text: &str) -> HashMap<String, u32> {
    let mut counts = HashMap::new();
    let tokens = tokenize(text);
    for token in tokens {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

// Unit tests go within the same file or submodules using #[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module (the library itself)

    #[test]
    fn test_add_one() {
        assert_eq!(add_one(1), 2);
        assert_eq!(add_one(0), 1);
        assert_eq!(add_one(-1), 0);
    }

    // --- Tokenize Tests ---

    #[test]
    fn test_tokenize_simple() {
        let text = "hello world";
        let expected = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(tokenize(text), expected);
    }

    #[test]
    fn test_tokenize_case() {
        let text = "Hello World";
        let expected = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(tokenize(text), expected, "Should convert to lowercase");
    }

    #[test]
    fn test_tokenize_punctuation() {
        let text = "Hello, world! Go.";
        let expected = vec!["hello".to_string(), "world".to_string(), "go".to_string()];
        assert_eq!(tokenize(text), expected, "Should trim punctuation");
    }

    #[test]
    fn test_tokenize_mixed() {
        // Test case from the original problem description that failed previously
        let text = "alpha beta-- gamma delta! dash-dash";
        let expected = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string(), "delta".to_string(), "dash-dash".to_string()];
        assert_eq!(tokenize(text), expected);
    }

    #[test]
    fn test_tokenize_empty() {
        let text = "";
        let expected: Vec<String> = Vec::new();
        assert_eq!(tokenize(text), expected);
    }

    #[test]
    fn test_tokenize_only_punctuation() {
        let text = ", . ! -- ??";
        let expected: Vec<String> = Vec::new();
        assert_eq!(tokenize(text), expected, "Should return empty for only punctuation");
    }

    #[test]
    fn test_tokenize_numbers_and_symbols() {
        let text = "word1 123 !!another-word?? 456";
        let expected = vec!["word1".to_string(), "123".to_string(), "another-word".to_string(), "456".to_string()];
        assert_eq!(tokenize(text), expected, "Should handle numbers and internal hyphens/symbols");
    }

    // --- Count Words Tests ---

    #[test]
    fn test_count_words_simple() {
        let text = "hello world hello";
        let counts = count_words(text);
        let mut expected = HashMap::new();
        expected.insert("hello".to_string(), 2);
        expected.insert("world".to_string(), 1);
        assert_eq!(counts, expected);
    }

    #[test]
    fn test_count_words_case_insensitive() {
        let text = "Hello hello HELLO";
        let counts = count_words(text);
        let mut expected = HashMap::new();
        expected.insert("hello".to_string(), 3);
        assert_eq!(counts, expected, "Should be case-insensitive");
    }

    #[test]
    fn test_count_words_with_punctuation() {
        let text = "Go, team, go! Go team?";
        let counts = count_words(text);
        let mut expected = HashMap::new();
        expected.insert("go".to_string(), 3);
        expected.insert("team".to_string(), 2);
        assert_eq!(counts, expected, "Should handle punctuation"); // Modified expected count based on new tests
    }

    #[test]
    fn test_count_words_empty_after_tokenize() {
        let text = "! , . --"; // These should all be removed by tokenize
        let counts = count_words(text);
        let expected: HashMap<String, u32> = HashMap::new();
        assert_eq!(counts, expected, "Should be empty if all words are punctuation");
    }
} 