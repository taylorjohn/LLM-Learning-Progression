dataset = WikiTextDataset('data/wikitext-2-raw/wiki.train.raw', seq_length=50)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = TransformerLanguageModel(vocab_size=len(dataset.word_to_ix), d_model=512, nhead=8, num_layers=6)
# ... (rest of your training code)