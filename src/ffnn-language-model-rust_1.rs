use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::collections::HashMap;

struct FFNNLM {
    vocab: Vec<String>,
    word_to_index: HashMap<String, usize>,
    embedding_dim: usize,
    hidden_dim: usize,
    context_size: usize,
    embeddings: Array2<f32>,
    hidden_weights: Array2<f32>,
    output_weights: Array2<f32>,
}

impl FFNNLM {
    fn new(vocab: Vec<String>, embedding_dim: usize, hidden_dim: usize, context_size: usize) -> Self {
        let vocab_size = vocab.len();
        let word_to_index: HashMap<_, _> = vocab.iter().cloned().enumerate().map(|(i, w)| (w, i)).collect();
        
        let mut rng = rand::thread_rng();
        
        FFNNLM {
            vocab,
            word_to_index,
            embedding_dim,
            hidden_dim,
            context_size,
            embeddings: Array2::random((vocab_size, embedding_dim), rand::distributions::Uniform::new(-1.0, 1.0)),
            hidden_weights: Array2::random((context_size * embedding_dim, hidden_dim), rand::distributions::Uniform::new(-1.0, 1.0)),
            output_weights: Array2::random((hidden_dim, vocab_size), rand::distributions::Uniform::new(-1.0, 1.0)),
        }
    }
    
    fn forward(&self, context: &[usize]) -> Array1<f32> {
        let mut input = Array1::zeros(self.context_size * self.embedding_dim);
        for (i, &word_idx) in context.iter().enumerate() {
            let start = i * self.embedding_dim;
            let end = (i + 1) * self.embedding_dim;
            input.slice_mut(s![start..end]).assign(&self.embeddings.row(word_idx));
        }
        
        let hidden = input.dot(&self.hidden_weights).mapv(|x| x.max(0.0)); // ReLU activation
        let output = hidden.dot(&self.output_weights);
        
        let max = output.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = output.mapv(|x| (x - max).exp());
        let sum = exp.sum();
        exp / sum
    }
    
    fn train(&mut self, text: &str, epochs: usize, learning_rate: f32) {
        let words: Vec<usize> = text.split_whitespace()
            .map(|w| *self.word_to_index.get(w).unwrap_or(&0))
            .collect();
        
        for _ in 0..epochs {
            for i in 0..words.len() - self.context_size {
                let context = &words[i..i + self.context_size];
                let target = words[i + self.context_size];
                
                let output = self.forward(context);
                let mut gradient = output.clone();
                gradient[target] -= 1.0;
                
                // Backpropagation (simplified)
                let d_output_weights = Array2::from_shape_fn((self.hidden_dim, self.vocab.len()),
                    |(_i, j)| gradient[j]);
                self.output_weights -= &(d_output_weights * learning_rate);
                
                // Update other weights similarly (omitted for brevity)
            }
        }
    }
    
    fn generate(&self, start_context: &[String], num_words: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut context: Vec<usize> = start_context.iter()
            .map(|w| *self.word_to_index.get(w).unwrap_or(&0))
            .collect();
        let mut output = start_context.to_vec();
        
        for _ in 0..num_words {
            let probs = self.forward(&context);
            let next_word_idx = (0..probs.len())
                .max_by(|&i, &j| probs[i].partial_cmp(&probs[j]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            output.push(self.vocab[next_word_idx].clone());
            context.push(next_word_idx);
            context = context[1..].to_vec();
        }
        
        output.join(" ")
    }
}

fn main() {
    let vocab = "to be or not that is the question".split_whitespace().map(String::from).collect();
    let mut model = FFNNLM::new(vocab, 10, 20, 2);
    
    let corpus = "to be or not to be that is the question";
    model.train(corpus, 1000, 0.01);
    
    let generated = model.generate(&["to", "be"], 10);
    println!("Generated: {}", generated);
}