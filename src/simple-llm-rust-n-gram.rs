use std::collections::HashMap;
use rand::Rng;

struct NGramModel {
    n: usize,
    model: HashMap<Vec<String>, HashMap<String, u32>>,
}

impl NGramModel {
    fn new(n: usize) -> Self {
        NGramModel {
            n,
            model: HashMap::new(),
        }
    }

    fn train(&mut self, text: &str) {
        let words: Vec<String> = text.split_whitespace().map(String::from).collect();
        for i in 0..=words.len() - self.n {
            let context: Vec<String> = words[i..i + self.n - 1].to_vec();
            let next_word = words[i + self.n - 1].clone();
            self.model
                .entry(context)
                .or_insert_with(HashMap::new)
                .entry(next_word)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    fn generate(&self, start: &[String], length: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut current_context: Vec<String> = start.to_vec();
        let mut output = current_context.join(" ");

        for _ in 0..length {
            if let Some(next_words) = self.model.get(&current_context) {
                let total: u32 = next_words.values().sum();
                let mut rand_val = rng.gen_range(0..total);
                let next_word = next_words
                    .iter()
                    .find(|(_, &count)| {
                        if rand_val < count {
                            true
                        } else {
                            rand_val -= count;
                            false
                        }
                    })
                    .map(|(word, _)| word)
                    .unwrap();

                output.push_str(&format!(" {}", next_word));
                current_context.push(next_word.clone());
                current_context = current_context[1..].to_vec();
            } else {
                break;
            }
        }

        output
    }
}

fn main() {
    // Create a new 3-gram model
    let mut model = NGramModel::new(3);
    
    // Training data
    let training_texts = [
        "the quick brown fox jumps over the lazy dog",
        "a stitch in time saves nine",
        "an apple a day keeps the doctor away",
        "birds of a feather flock together",
        "every cloud has a silver lining",
    ];

    // Train the model
    println!("Training the model...");
    for text in training_texts.iter() {
        model.train(text);
        println!("Trained on: {}", text);
    }

    // Generate text
    println!("\nGenerating text:");
    let start_words = vec!["the".to_string(), "quick".to_string()];
    for _ in 0..5 {
        let generated_text = model.generate(&start_words, 10);
        println!("{}", generated_text);
    }

    // Try different starting words
    println!("\nGenerating with different start:");
    let new_start = vec!["an".to_string(), "apple".to_string()];
    for _ in 0..3 {
        let generated_text = model.generate(&new_start, 8);
        println!("{}", generated_text);
    }
}