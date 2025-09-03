import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel
block_size = 8   # maximum context length for the model
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'  # Uncomment this line to force CPU usage
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

# Create a simple model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        #idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C) where C is the vocab size

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
            #get predictions
            logits, loss = self(idx)
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
model = BigramLanguageModel(vocab_size)
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
