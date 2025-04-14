# GPT Model Progression

This repository demonstrates the evolution of GPT (Generative Pre-trained Transformer) model implementations, from simple n-gram models to a state-of-the-art, M2-optimized version.

## Project Structure

```
GPT-Progression/
│
├── 01_unigram_model/
│   └── unigram_model.rs
├── 02_bigram_model/
│   └── bigram_model.rs
├── 03_ngram_model/
│   └── ngram_model.rs
├── 04_ngram_backoff_model/
│   └── ngram_backoff_model.rs
├── 05_ffnn_language_model/
│   └── ffnn_language_model.rs
├── 06_rnn_language_model/
│   └── rnn_language_model.rs
├── 07_lstm_language_model/
│   └── lstm_language_model.rs
├── 08_transformer_language_model/
│   └── transformer_language_model.rs
├── 09_simplified_gpt/
│   └── simplified_gpt.rs
├── 10_advanced_gpt/
│   └── advanced_gpt.rs
└── 11_m2_optimized_gpt/
    ├── src/
    │   └── main.rs
    ├── train_gpt.py
    └── Cargo.toml
```

## Model Progression

```
         Unigram
            │
            ▼
         Bigram
            │
            ▼
         N-gram
            │
            ▼
    N-gram w/ Backoff
            │
            ▼
   Feed-Forward Neural Network
            │
            ▼
   Recurrent Neural Network (RNN)
            │
            ▼
 Long Short-Term Memory (LSTM)
            │
            ▼
        Transformer
            │
            ▼
      Simplified GPT
            │
            ▼
      Advanced GPT
            │
            ▼
    M2-Optimized GPT
```

## Key Features

1. **Progressive Complexity**: From simple statistical models to advanced neural architectures.
2. **Rust Implementation**: Core components implemented in Rust for efficiency.
3. **PyTorch Integration**: Advanced models use PyTorch for GPU acceleration.
4. **M2 Optimization**: Final model optimized for MacBook Air M2 GPUs.
5. **BPE Tokenization**: Efficient subword tokenization implemented in Rust.
6. **Training Pipeline**: Complete training loop with optimization and scheduling.

## Simplified Aspects

While this project aims to demonstrate the progression of language models, some aspects are simplified for clarity and ease of implementation:

1. **Tokenization**: The BPE implementation is basic and doesn't include all optimizations used in production tokenizers.
2. **Model Sizes**: The models use smaller dimensions than typical large language models for faster training on consumer hardware.
3. **Training Data**: The examples use small datasets. Production models would use much larger and more diverse datasets.
4. **Fine-tuning**: The implementation doesn't include techniques for efficient fine-tuning or transfer learning.
5. **Optimizations**: Advanced techniques like mixed-precision training or model parallelism are not implemented.

## Usage

Each subdirectory contains a standalone implementation of its respective model. For the M2-optimized GPT:

1. Navigate to the `11_m2_optimized_gpt` directory.
2. Compile and run the Rust tokenizer:
   ```
   cargo run --release
   ```
3. Install required Python packages:
   ```
   pip install torch torchvision torchaudio
   ```
4. Prepare your text data in a file named "your_text_data.txt".
5. Run the Python training script:
   ```
   python train_gpt.py
   ```

## Future Directions

- Implement more advanced attention mechanisms (e.g., sparse attention).
- Add support for larger model sizes and distributed training.
- Incorporate advanced prompting techniques and few-shot learning.
- Implement RLHF (Reinforcement Learning from Human Feedback) for alignment.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.