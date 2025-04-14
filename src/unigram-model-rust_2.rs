impl UnigramModel {
    // ... (previous methods)

    fn perplexity(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len();
        let mut log_likelihood = 0.0;

        for word in words {
            let count = self.word_counts.get(word).unwrap_or(&0);
            let probability = *count as f64 / self.total_words as f64;
            log_likelihood += (probability + 1e-10).ln();
        }

        (-log_likelihood / n as f64).exp()
    }
}