// Integration tests live in the tests/ directory
// They compile as separate crates and link against the library crate

// Import the library crate (defined in Cargo.toml and src/lib.rs)
// We can import specific items if needed
use cs336_basics_rs::{add_one, count_words, tokenize};
use std::collections::HashMap;

#[test]
fn test_add_one_integration() {
    // Call function from the library crate
    assert_eq!(add_one(5), 6);
}

#[test]
fn test_tokenize_integration() {
    let text = "Test, Tokenize! Me.";
    let expected_tokens = vec!["test".to_string(), "tokenize".to_string(), "me".to_string()];
    let actual_tokens = tokenize(text);
    assert_eq!(actual_tokens, expected_tokens);
}

#[test]
fn test_count_words_integration() {
    // Use text where tokenization matters
    let text = "Integration Test integration TEST - ignore me!";
    let expected_counts: HashMap<String, u32> = [
        ("integration".to_string(), 2),
        ("test".to_string(), 2),
        ("ignore".to_string(), 1),
        ("me".to_string(), 1),
    ].iter().cloned().collect();
    
    let actual_counts = count_words(text);
    assert_eq!(actual_counts, expected_counts);
}

// You would add more integration tests here, possibly in separate files
// within the tests/ directory, testing the public API of your library. 