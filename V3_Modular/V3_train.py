import torch
from tqdm import tqdm
from V3_model import BigramLanguageModel, get_vocab

# Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel
block_size = 256   # maximum context length for the model
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
progress_bar = tqdm(range(max_iters))
# --------------------------#

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# --------------------------#

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size, encode, decode = get_vocab(text)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ...define get_batch, estimate_loss as before...
# Data loading
def get_batch(split):
    # Get a small batch of data - x is the input, y is the target
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        progress_bar.update(eval_interval)

    if iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    
    # Sample a batch of data
    xb, yb = get_batch('train')
    # Evaluate the model
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'V3_Modular/Model_weights/v3_model.pt')