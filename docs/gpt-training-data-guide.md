# Training Data Guide for Model Progression

This guide provides general recommendations for the **type and scale** of training data suitable for illustrating the capabilities and differences between each model stage in this repository's progression. *Note: The simplified implementations provided here are not trained on these scales; this guide is conceptual.* For actual large model pre-training, vastly larger datasets (billions or trillions of tokens) are used.

## Model Stages and Data Recommendations

1.  **Unigram Model**
    *   **Concept:** Models word frequency only.
    *   **Data Type:** Small corpus of simple sentences.
    *   **Example:** 50-100 basic sentences (`The cat sat. Dogs bark.`).
    *   **Purpose:** Demonstrate basic probability calculation.

2.  **Bigram Model**
    *   **Concept:** Models P(word | previous_word).
    *   **Data Type:** Short paragraphs showing word pairs.
    *   **Example:** 200-300 sentences, maybe a children's story excerpt.
    *   **Purpose:** Show basic context dependence and smoothing needs.

3.  **N-gram Model (N=3+)**
    *   **Concept:** Models P(word | previous_N-1_words).
    *   **Data Type:** Longer paragraphs or short articles.
    *   **Example:** 1000-2000 sentences (simple Wikipedia article).
    *   **Purpose:** Illustrate longer context vs. increased sparsity.

4.  **N-gram Model with Backoff**
    *   **Concept:** Smoothing for unseen n-grams.
    *   **Data Type:** Mix of common and less common constructions/phrases.
    *   **Example:** 2000-5000 sentences (news articles + specific domain text).
    *   **Purpose:** Show improved handling of unseen sequences.

5.  **Word Embeddings (Concept)**
    *   **Concept:** Learning dense vector representations.
    *   **Data Type:** N/A (Conceptual stage). Real embedding training (Word2Vec, GloVe) uses large, diverse corpora (millions/billions of words).

6.  **Feed-Forward Neural Network LM**
    *   **Concept:** Neural prediction with fixed context window using embeddings.
    *   **Data Type:** Diverse corpus (news, wiki, literature). Larger than n-grams.
    *   **Example:** 10k-50k sentences.
    *   **Purpose:** Demonstrate generalization via embeddings, fixed context limitation.

7.  **Recurrent Neural Network (RNN) LM**
    *   **Concept:** Processing sequences with a hidden state.
    *   **Data Type:** Coherent texts with sequential dependencies.
    *   **Example:** 50k-100k sentences (full articles, short stories).
    *   **Purpose:** Show handling of variable length, illustrate vanishing gradients.

8.  **Long Short-Term Memory (LSTM) LM**
    *   **Concept:** RNN with gates to manage memory and gradients.
    *   **Data Type:** Longer coherent texts than basic RNN.
    *   **Example:** 100k-500k sentences (book chapters, longer articles).
    *   **Purpose:** Demonstrate improved handling of long-range dependencies.

9.  **Attention Mechanism (Concept)**
    *   **Concept:** Allowing focus on relevant parts of input regardless of distance.
    *   **Data Type:** N/A (Conceptual stage).

10. **Transformer Language Model**
    *   **Concept:** Architecture based solely on self-attention.
    *   **Data Type:** Large, diverse corpus.
    *   **Example:** 1M-5M sentences (books, articles, websites).
    *   **Purpose:** Show power of attention, parallelization benefits.

11. **Simplified GPT Language Model (Architecture Demo)**
    *   **Concept:** Decoder-only Transformer for generation.
    *   **Data Type:** (For demo/testing the *architecture*) - Similar scale to Transformer example.
    *   **Example:** 5M-10M sentences.
    *   **Purpose:** Illustrate the specific GPT structure. *(Real GPT pre-training uses vastly more data).*

12. **Advanced GPT Implementations (Architecture Demo)**
    *   **Concept:** Incorporating BPE, RAG concepts, etc.
    *   **Data Type:** (For demo/testing the *architecture*) - Large, diverse corpus.
    *   **Example:** 10M-100M sentences.
    *   **Purpose:** Illustrate advanced components. *(Real GPT pre-training uses vastly more data).*

13. **Fine-Tuning GPT Models (Process)**
    *   **Concept:** Adapting a pre-trained model to a specific task.
    *   **Data Type:** Task-specific, labeled dataset (e.g., question-answer pairs, sentiment-labeled reviews, summarization pairs). Size varies greatly depending on task (hundreds to tens of thousands).
    *   **Purpose:** Demonstrate specialization of pre-trained knowledge.

## General Guidelines for Selecting Training Data:

1. **Diversity**: Ensure a mix of topics, styles, and sources appropriate to the model's complexity.
2. **Quality**: For simpler models, use clean, well-structured text. For advanced models, include more varied and realistic language use.
3. **Size**: Increase the corpus size as models become more complex and capable.
4. **Domain Relevance**: If training for a specific application, include domain-specific text in addition to general language data.
5. **Ethical Considerations**: Be mindful of biases in the training data, especially for more advanced models.
6. **Copyright**: Ensure you have the right to use the selected texts for training.

When using this guide, adjust the recommendations based on your specific needs and computational resources. For the more advanced models, you may need to use publicly available datasets or create a custom web scraping pipeline to gather sufficient data.