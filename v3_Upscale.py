import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel
block_size = 256   # maximum context length for the model
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384  # size of the embedding vector for each token
n_layer = 6  # number of transformer blocks
n_head = 6  # number of attention heads
dropout = 0.2  # dropout probability
# --------------------------#

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# --------------------------#

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#-- print("length of dataset in characters: ", len(text))
#-- print(text[:1000])

# Number of unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
#-- print(''.join(chars))
#-- print(vocab_size)
# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } 
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encode a string to a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decode a list of integers to a string
#-- print(encode("hii there"))

# Train/test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% for training, rest for validation
train_data = data[:n]
val_data = data[n:]
# print(data.shape, data.dtype)
# print(data[:1000])
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")

# Data loading
def get_batch(split):
    # Get a small batch of data - x is the input, y is the target
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
# xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)--------------input to transformer
# print('targets:')
# print(yb.shape)
# print(yb)--------------target to transformer
# print('----')
# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")

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

# One Head of Attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        # calculate attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module in parallel."""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """A simple feed-forward neural network with a ReLU activation."""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """A single transformer block consisting of multi-head self-attention and feed-forward network."""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # 4 heads, each of size 8
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # normalization layer for self-attention
        self.ln2 = nn.LayerNorm(n_embd)  # normalization layer for feed-forward network

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # apply self-attention
        x = x + self.ffwd(self.ln2(x))  # apply feed-forward network
        return x

# Create a simple model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final normalization layer
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        #idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C) where C is the embedding size
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C) after passing through transformer blocks
        x = self.ln_f(x)  # (B, T, C) after final layer normalization
        logits = self.lm_head(x)  # (B, T, C) where C is vocab size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# Initialize the model
model = BigramLanguageModel()
m = model.to(device)
# logits, loss = model(xb, yb)
# print(logits.shape)
# print(loss)
# print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# Pytorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('output.txt', 'w').write(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))