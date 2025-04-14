impl TransformerLayer {
    // ... (previous implementation)

    fn forward(&self, x: &Array2<f32>, cache: Option<&(Array2<f32>, Array2<f32>)>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let (attn_output, new_k, new_v) = self.self_attn.forward(x, cache);
        let x = self.ln1.forward(&(x + &attn_output));
        let ff_output = self.ff.forward(&x);
        let x = self.ln2.forward(&(x + &ff_output));
        (x, new_k, new_v)
    }
}

impl LayerNorm {
    fn new(embed_dim: usize) -> Self {
        LayerNorm {
            gamma: Array1::ones(embed_dim),
            beta: Array1::zeros(embed_dim),
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);
        (x - &mean.insert_axis(Axis(1))) / (var + self.eps).sqrt().insert_axis(Axis(1)) * &self.gamma + &self.beta
    }
}

fn main() {
    let embed_dim = 512;
    let num_heads = 8;
    let num_layers = 12;
    let vocab_size = 50000;
    let max_seq_len = 2048;

    let mut model = StateOfTheArtGPT::new(embed_dim, num_heads, num_layers, vocab_size, max_seq_len);

    // Add some entries to the knowledge base
    model.knowledge_base.insert(
        "Rust".to_string(),
        model.embed_text("Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.")
    );
    model.knowledge_base.insert(
        "GPT".to_string(),
        model.embed_text("GPT (Generative Pre-trained Transformer) is a type of large language model that uses deep learning to produce human-like text.")
    );
    model.knowledge_base.insert(
        "Quantum Computing".to_string(),
        model.embed_text("Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.")
    );

    // Generate text with retrieval
    let prompt = "Explain the relationship between Rust and GPT in the context of quantum computing";
    let generated = model.generate_with_retrieval(prompt, 100);
    println!("Generated text:\n{}", generated);

    // Demonstrate sliding window attention
    let long_prompt = "This is a very long input sequence that exceeds the normal context window of the model. ".repeat(100);
    let long_generated = model.generate(&long_prompt, 50);
    println!("\nLong sequence generation:\n{}", long_generated);
}