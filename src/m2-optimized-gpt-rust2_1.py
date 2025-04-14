def train(model, train_dataset, val_dataset, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataset), eta_min=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Generate sample text
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
        decoded = train_dataset.tokenizer.decode(generated[0].tolist())
        print(f"Generated sample: {decoded}")

def main():
    # Load the tokenizer
    tokenizer = BPETokenizer.load("tokenizer.txt")

    # Load your text data
    with open("your_text_data.txt", "r") as f:
        text = f.read()

    # Create config
    config = GPTConfig(
        vocab_size=len(tokenizer.encoder),
        block_size=128,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1
    )

    # Create datasets
    full_dataset = TextDataset(text, config.block_size, tokenizer)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create model
    model = GPT(config).to(device)

    # Training configuration
    train_config = type('TrainConfig', (), {
        'batch_size': 64,
        'num_epochs': 10,
    })()

    # Train the model
    train(model, train_dataset, val_dataset, train_config)

    # Save the model
    torch.save(model.state_dict(), "gpt_model.pth")

if __name__ == "__main__":
    main()