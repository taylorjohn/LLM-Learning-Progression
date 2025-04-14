# Subword Tokenization: Byte Pair Encoding (BPE)

## What is Byte Pair Encoding (BPE)?

**Byte Pair Encoding (BPE)** is a widely used **subword tokenization** algorithm. Instead of treating words or characters as the basic units (tokens), BPE learns to break down words into smaller, frequently occurring multi-character units. This allows language models like GPT to handle large vocabularies, including rare or unseen words, much more effectively than word-level or character-level tokenization alone.

---

### Why Use Subword Tokenization (like BPE)?

1.  **Vocabulary Size Management**: A purely word-based vocabulary can become enormous (millions of entries), especially with inflections, typos, and rare words. A purely character-based vocabulary is small but loses word-level semantics and creates very long sequences. Subword tokenization finds a balance, creating a vocabulary typically in the range of 30,000-100,000 tokens that are more meaningful than characters but more flexible than whole words.
2.  **Handling Out-of-Vocabulary (OOV) / Rare Words**: Word-level tokenizers assign a special `<UNK>` (unknown) token to words not seen during training. BPE can represent rare or new words by composing them from known subword units (e.g., "subword" -> "sub", "word"), eliminating the `<UNK>` problem for any sequence of known characters.
3.  **Improved Generalization & Efficiency**: Models learn representations for frequent subwords (like prefixes "un-", suffixes "-ing", "-ly") once and can reuse this knowledge when encountering new words containing these units.

---

### How Does BPE Work?

BPE starts with a base vocabulary of individual characters and iteratively merges the most frequent **adjacent** pair of existing tokens (characters or merged subwords) in the training corpus until a desired vocabulary size (a hyperparameter) is reached.

#### Step-by-Step Breakdown:

1.  **Initialization**: Define the base vocabulary as all individual characters present in the training corpus. Split every word in the corpus into a sequence of these characters (often adding a special end-of-word symbol like `</w>`).
    ```
    Corpus: {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    Initial Splits (with frequency & end-of-word):
    l o w </w>          : 5
    l o w e r </w>      : 2
    n e w e s t </w>    : 6
    w i d e s t </w>    : 3
    ```
2.  **Iteration**: Repeat the following steps until the desired vocabulary size is reached:
    a.  **Find Most Frequent Pair**: Count the occurrences of all adjacent pairs of tokens in the current corpus representation. Identify the pair that occurs most frequently (e.g., perhaps 'e' + 's' is most frequent initially).
    b.  **Merge Pair**: Create a *new* token representing this merged pair (e.g., "es"). Add this new token to the vocabulary.
    c.  **Update Corpus**: Replace all occurrences of the most frequent pair (e.g., 'e', 's') in the corpus representation with the newly merged token (e.g., "es").

    *Example First Merge (assuming 'e' + 's' is most frequent):*
    ```
    New token: "es"
    Vocabulary: {l,o,w,</w>,n,e,s,t,i,d, es}
    Updated Splits:
    l o w </w>          : 5
    l o w e r </w>      : 2
    n e w es t </w>     : 6  <-- updated
    w i d es t </w>     : 3  <-- updated
    ```
    *Example Second Merge (assume 'es' + 't' is most frequent now):*
    ```
    New token: "est"
    Vocabulary: {l,o,w,</w>,n,e,s,t,i,d, es, est}
    Updated Splits:
    l o w </w>          : 5
    l o w e r </w>      : 2
    n e w est </w>    : 6  <-- updated
    w i d est </w>    : 3  <-- updated
    ```
3.  **Final Vocabulary & Merge Rules**: The final vocabulary consists of the initial characters plus all the merged subword tokens. The learned sequence of merges defines the rules for tokenizing new text.

---

### Example of BPE in Action

Let's say we have a corpus with the words:

```text
lower, lowest, slower, slowest
```

1. **Initial Split**: Break each word into individual characters:

   ```text
   lower → l o w e r
   lowest → l o w e s t
   slower → s l o w e r
   slowest → s l o w e s t
   ```

2. **Find the Most Frequent Pair**: The most frequent character pair is `l` + `o`.

   ```text
   Frequent pair: l o
   ```

3. **Merge the Pair**:

   ```text
   lower → lo w e r
   lowest → lo w e s t
   slower → s lo w e r
   slowest → s lo w e s t
   ```

4. **Repeat**: The next frequent pair is `lo` + `w`.

   ```text
   Frequent pair: lo w
   ```

5. **Merge Again**:

   ```text
   lower → low e r
   lowest → low e s t
   slower → s low e r
   slowest → s low e s t
   ```

6. **Continue**: This process continues until the vocabulary reaches the desired size. Eventually, subword units like `low` and `est` are learned, allowing the model to efficiently tokenize and represent words.

---

### ASCII Visualization of BPE Process

Here's a simplified ASCII art visualization of the BPE process:

```
Initial Words:
lower → [l] [o] [w] [e] [r]
lowest → [l] [o] [w] [e] [s] [t]

Step 1: Find and Merge Frequent Pairs
Frequent Pair: "l" + "o"

lower → [lo] [w] [e] [r]
lowest → [lo] [w] [e] [s] [t]

Step 2: Merge Next Frequent Pair
Frequent Pair: "w" + "e"

lower → [lo] [we] [r]
lowest → [lo] [we] [s] [t]

Step 3: Continue Until Vocabulary Limit
(lower, lowest, slower, slowest)
```

---

### Key Benefits of BPE

1. **Efficiency**: By merging frequent subword patterns, BPE allows models to represent words as a combination of known subword units, reducing the overall vocabulary size.
2. **Flexibility**: BPE can handle both common and rare words by breaking them into meaningful subword units.
3. **Generalization**: Since BPE captures subword patterns, the model can generalize to new words by reusing known subword units.

---

### How is BPE Used in GPT Models?

BPE is a key part of tokenization in GPT models:

- **Training Stage**: During training, the model learns to associate subword units (like "low" and "er") with meaningful patterns in the data.
- **Inference Stage**: When generating text, the model can recombine these subword units to form coherent words and sentences, even if it hasn't seen a specific word before.

---

### Advanced Example: Handling Rare Words

Let's say the word "astronaut" is rare in the corpus. Instead of learning it as a whole, BPE might split it into subword units like this:

```
"astronaut" → ["astro", "naut"]
```

When the model encounters "astronaut," it can generate it by combining the subword units "astro" (which it has seen in words like "astronomy") and "naut" (which it has seen in "cosmonaut"). This approach reduces the number of rare tokens and allows the model to generate more varied text.

---

### Other Subword Methods

While BPE is common (used by GPT-2, GPT-3), other methods exist:
*   **WordPiece:** Used by BERT. Similar to BPE but merges pairs based on maximizing likelihood rather than frequency.
*   **SentencePiece:** Treats the input text as a raw stream, including whitespace. Can learn tokens that cross word boundaries. Often uses BPE or Unigram LM as the underlying algorithm.

---

### Conclusion

**Byte Pair Encoding (BPE)** and other subword tokenization methods are fundamental to modern NLP. They strike a balance between word and character-level approaches, enabling models to handle large vocabularies, rare/unseen words, and morphology effectively, which is crucial for the performance of models like GPT.

