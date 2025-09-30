import torch
from bygram_model import BigramLanguageModel, get_vocab

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size, encode, decode = get_vocab(text)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BigramLanguageModel(vocab_size).to(device)
model.load_state_dict(torch.load('Modular_Bygram/Model_weights/bigram_model.pt', map_location=device))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
open('Modular_Bygram/generated_text/output.txt', 'w').write(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))