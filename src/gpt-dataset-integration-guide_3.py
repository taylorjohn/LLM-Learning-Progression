import torch
from torch.utils.data import Dataset, DataLoader

class WikiTextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        with open(file_path, 'r') as f:
            self.data = f.read()
        self.words = self.data.split()
        self.word_to_ix = {word: i for i, word in enumerate(set(self.words))}
        self.ix_to_word = {i: word for word, i in self.word_to_ix.items()}
        self.seq_length = seq_length

    def __len__(self):
        return len(self.words) - self.seq_length

    def __getitem__(self, idx):
        inputs = [self.word_to_ix[word] for word in self.words[idx:idx+self.seq_length]]
        targets = [self.word_to_ix[word] for word in self.words[idx+1:idx+self.seq_length+1]]
        return torch.tensor(inputs), torch.tensor(targets)

# In your main training loop:
dataset = WikiTextDataset('data/wikitext-2-raw/wiki.train.raw', seq_length=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FFNNLanguageModel(vocab_size=len(dataset.word_to_ix), embed_dim=128, hidden_dim=256)
# ... (rest of your training code)