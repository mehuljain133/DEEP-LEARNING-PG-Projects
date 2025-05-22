# Unit-IV Sequence Modeling: Recurrent Nets: Unfolding computational graphs, recurrent neural networks (RNNs), bidirectional RNNs, encoder-decoder sequence to sequence architectures, deep recurrent networks.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# Sample data for sequence modeling
text = "hello world. this is a simple RNN sequence modeling demo. enjoy learning!"
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

# Prepare sequences
def make_sequences(text, seq_len=10):
    X, y = [], []
    for i in range(len(text) - seq_len):
        seq = text[i:i + seq_len]
        target = text[i + 1:i + seq_len + 1]
        X.append([char2idx[ch] for ch in seq])
        y.append([char2idx[ch] for ch in target])
    return torch.tensor(X), torch.tensor(y)

X, y = make_sequences(text)
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Recurrent Network Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, num_layers=1, bidirectional=False, encoder_decoder=False):
        super(RNNModel, self).__init__()
        self.encoder_decoder = encoder_decoder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)

        if encoder_decoder:
            self.decoder_rnn = nn.RNN(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)  # [B, T, H]
        if self.encoder_decoder:
            _, h = self.rnn(embedded)
            # repeat last hidden state for each timestep
            decoder_input = embedded  # simple way to mimic decoding step
            output, _ = self.decoder_rnn(decoder_input, h)
        else:
            output, _ = self.rnn(embedded)
        out = self.fc(output)
        return out

# Training function
def train_model(model, dataloader, epochs=20):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out.view(-1, len(chars)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Sampling function
def sample(model, start_seq="hello ", length=50):
    model.eval()
    input_seq = torch.tensor([[char2idx[c] for c in start_seq]], device=device)
    with torch.no_grad():
        for _ in range(length):
            out = model(input_seq)
            probs = F.softmax(out[:, -1], dim=-1)
            next_char = torch.multinomial(probs, num_samples=1).item()
            input_seq = torch.cat([input_seq, torch.tensor([[next_char]], device=device)], dim=1)
        result = ''.join([idx2char[i] for i in input_seq[0].tolist()])
        print(f"Generated Text: {result}")

# Choose model type:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Simple RNN
print("\n--- Simple RNN ---")
model1 = RNNModel(vocab_size=len(chars)).to(device)
train_model(model1, dataloader)
sample(model1)

# 2. Bidirectional RNN
print("\n--- Bidirectional RNN ---")
model2 = RNNModel(vocab_size=len(chars), bidirectional=True).to(device)
train_model(model2, dataloader)
sample(model2)

# 3. Deep RNN
print("\n--- Deep RNN (2 Layers) ---")
model3 = RNNModel(vocab_size=len(chars), num_layers=2).to(device)
train_model(model3, dataloader)
sample(model3)

# 4. Encoder-Decoder (Seq2Seq style)
print("\n--- Encoder-Decoder RNN ---")
model4 = RNNModel(vocab_size=len(chars), encoder_decoder=True).to(device)
train_model(model4, dataloader)
sample(model4)
