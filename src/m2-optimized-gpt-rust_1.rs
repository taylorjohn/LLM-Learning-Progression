use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use regex::Regex;

struct BPETokenizer {
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    regex: Regex,
}

impl BPETokenizer {
    fn new() -> Self {
        BPETokenizer {
            encoder: HashMap::new(),
            decoder: HashMap::new(),
            bpe_ranks: HashMap::new(),
            regex: Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap(),
        }
    }

    fn train(&mut self, texts: &[String], vocab_size: usize, min_frequency: usize) {
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        
        // Count word frequencies
        for text in texts {
            for word in self.regex.find_iter(text) {
                *word_freq.entry(word.as_str().to_string()).or_insert(0) += 1;
            }
        }

        // Initialize with characters
        let mut vocab: HashSet<String> = word_freq.keys()
            .flat_map(|word| word.chars().map(|c| c.to_string()))
            .collect();

        while vocab.len() < vocab_size {
            let mut pairs = HashMap::new();
            for (word, freq) in &word_freq {
                let symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                for i in 0..symbols.len() - 1 {
                    let pair = (symbols[i].clone(), symbols[i + 1].clone());
                    *pairs.entry(pair).or_insert(0) += freq;
                }
            }

            let best_pair = pairs.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, _)| pair);

            if let Some((first, second)) = best_pair {
                let new_token = format!("{}{}", first, second);
                vocab.insert(new_token.clone());
                self.bpe_ranks.insert((first.clone(), second.clone()), self.bpe_ranks.len());

                // Update word frequencies
                let mut new_word_freq = HashMap::new();
                for (word, freq) in word_freq {
                    let new_word = word.replace(&format!("{}{}", first, second), &new_token);
                    *new_word_freq.entry(new_word).or_insert(0) += freq;
                }
                word_freq = new_word_freq;
            } else {
                break;
            }
        }

        // Create encoder and decoder
        for (i, token) in vocab.into_iter().enumerate() {
            self.encoder.insert(token.clone(), i);
            self.decoder.insert(i, token);
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        for word in self.regex.find_iter(text) {
            let mut word = word.as_str().to_string();
            let mut sub_tokens = vec![word.clone()];
            
            loop {
                let mut min_pair = None;
                let mut min_rank = std::usize::MAX;

                for i in 0..sub_tokens.len() - 1 {
                    let pair = (sub_tokens[i].clone(), sub_tokens[i + 1].clone());
                    if let Some(&rank) = self.bpe_ranks.get(&pair) {
                        if rank < min_rank {
                            min_pair = Some(i);
                            min_rank = rank;
                        }
                    }
                }

                if let Some(i) = min_pair {
                    let new_token = format!("{}{}", sub_tokens[i], sub_tokens[i + 1]);
                    sub_tokens[i] = new_token;
                    sub_tokens.remove(i + 1);
                } else {
                    break;
                }
            }

            tokens.extend(sub_tokens.into_iter().filter_map(|t| self.encoder.get(&t)).cloned());
        }
        tokens
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&token| self.decoder.get(&token))
            .collect::<Vec<_>>()
            .join("")
            .replace("</w>", " ")
            .trim()
            .to_string()
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        for (token, id) in &self.encoder {
            writeln!(file, "{}\t{}", token, id)?;
        }
        Ok(())
    }

    fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut tokenizer = BPETokenizer::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() == 2 {
                let token = parts[0].to_string();
                let id = parts[1].parse().unwrap();
                tokenizer.encoder.insert(token.clone(), id);
                tokenizer.decoder.insert(id, token);
            }
        }

        Ok(tokenizer)
    }
}

fn main() {
    // Example usage
    let texts = vec![
        "Hello, world!".to_string(),
        "This is a test.".to_string(),
        "BPE tokenization is fun.".to_string(),
    ];

    let mut tokenizer = BPETokenizer::new();
    tokenizer.train(&texts, 100, 2);

    // Save the tokenizer
    tokenizer.save("tokenizer.txt").unwrap();

    // Load the tokenizer
    let loaded_tokenizer = BPETokenizer::load("tokenizer.txt").unwrap();

    // Test encoding and decoding
    let text = "Hello, this is a test of BPE tokenization!";
    let encoded = loaded_tokenizer.encode(text);
    let decoded = loaded_tokenizer.decode(&encoded);

    println!("Original: {}", text);
    println!("Encoded: {:?}", encoded);
    println!("Decoded: {}", decoded);
}