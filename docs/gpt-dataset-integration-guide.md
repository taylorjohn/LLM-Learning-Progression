# Dataset Integration and Usage Guide for GPT Model Progression

## Recommended Dataset: WikiText-2

For our GPT model progression, we'll use the WikiText-2 dataset. It's a good choice because:

1. It's derived from Wikipedia articles, providing a mix of topics and styles.
2. It's large enough for advanced models but can be subsampled for simpler ones.
3. It's freely available and widely used in NLP research.

## Downloading the Dataset

1. Create a `data` directory in your project root:
   ```
   mkdir data
   cd data
   ```

2. Download the WikiText-2 dataset:
   ```
   wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
   unzip wikitext-2-raw-v1.zip
   ```

3. You'll now have `wiki.train.raw`, `wiki.valid.raw`, and `wiki.test.raw` files.

## Integrating the Dataset

Here's how to integrate the WikiText-2 dataset into each model and run them:

### 1. Unigram Model

[Link to `gpt-dataset-integration-guide_1.rs`](../src/gpt-dataset-integration-guide_1.rs)

Run with: `cargo run --bin unigram_model`

### 2-4. N-gram Models (including Backoff)

[Link to `gpt-dataset-integration-guide_2.rs`](../src/gpt-dataset-integration-guide_2.rs)

Run with: `cargo run --bin ngram_model`

### 5. Feed-Forward Neural Network Language Model

[Link to `gpt-dataset-integration-guide_3.py`](../src/gpt-dataset-integration-guide_3.py)

Run with: `python train_ffnn.py`

### 6-8. RNN, LSTM, and Transformer Models

These models can use a similar data loading approach as the FFNN, but may process longer sequences:

[Link to `gpt-dataset-integration-guide_4.py`](../src/gpt-dataset-integration-guide_4.py)

Run with: `python train_transformer.py`

### 9-11. GPT Models (Simplified, Advanced, and M2-Optimized)

For these models, you'll want to use the entire dataset and potentially implement more sophisticated data loading:

[Link to `gpt-dataset-integration-guide_5.py`](../src/gpt-dataset-integration-guide_5.py)

Run with: `python train_gpt.py`

## General Tips

1. For simpler models, use a subset of the data to keep training times reasonable.
2. For more advanced models, use the full dataset and consider using GPU acceleration if available.
3. Always split your data into train, validation, and test sets. WikiText-2 provides these splits.
4. For the M2-optimized version, make sure to use the MPS backend as described in the earlier implementation.

By following these guidelines, you can effectively integrate the WikiText-2 dataset into each model in our progression, providing a consistent basis for comparison and demonstrating the increasing capabilities of more advanced models.