import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import time
from gpt_model import GPTLanguageModel, block_size

# modelini yükle
device = 'cuda' if torch.cuda.is_available() else 'mps'
input_file = 'nutuk.txt'

# karakter mapping tekrar lazım
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
decode = lambda l: ''.join([itos[i] for i in l])

# modeli oluştur ve load et
model = GPTLanguageModel()
model.load_state_dict(torch.load(f'model_{input_file.split(".")[0]}.pth', map_location=device))
model.eval()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# başlangıç context (boş string = 0 token)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# sonsuz text üretme
try:
    while True:
        # sadece son block_size kadarını modelle
        idx_cond = context[:, -block_size:]

        # model forward
        logits, _ = m(idx_cond)
        logits = logits[:, -1, :]  # son token
        probs = F.softmax(logits, dim=-1)

        # örnekleme
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)

        # decode ve terminale bas
        char = decode([idx_next.item()])
        sys.stdout.write(char)
        sys.stdout.flush()

        # istersen hızını yavaşlatabilirsin (ör. daktilo efekti için)
        # time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
