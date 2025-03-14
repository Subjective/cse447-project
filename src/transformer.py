#!/usr/bin/env python
import os
import string
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Global vocabulary: we restrict predictions to a fixed set of characters.
VOCAB = string.printable
char2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2char = {i: ch for i, ch in enumerate(VOCAB)}

# Positional Encoding module (from standard transformer examples)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            # Compute positional encoding on the fly for longer sequences.
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(1)
        else:
            pe = self.pe[:seq_len]
        x = x + pe
        return self.dropout(x)

# Transformer language model for character-level prediction.
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1, max_len=1024):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, 
                                                    dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        # src shape: (seq_len, batch_size)
        embedded = self.embedding(src) * math.sqrt(self.embed_size)
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer_encoder(embedded)
        # transformer_out shape: (seq_len, batch_size, embed_size)
        out = self.fc_out(transformer_out)  # shape: (seq_len, batch_size, vocab_size)
        return out

class MyModel:
    """
    Transformer-based model for next character prediction.
    """

    def __init__(self):
        # Setup device and vocabulary.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = VOCAB
        self.char2idx = char2idx
        self.idx2char = idx2char
        # Initialize a small transformer language model.
        self.model = TransformerLM(len(self.vocab), embed_size=128, num_heads=4, hidden_dim=256, num_layers=2, max_len=1024).to(self.device)

    @classmethod
    def load_training_data(cls):
        # Load training data from training_data.txt located in the same directory as this file.
        training_file = os.path.join(os.path.dirname(__file__), 'training_data.txt')
        if not os.path.exists(training_file):
            print(f"[WARN] {training_file} not found. No training data loaded.")
            return []
        data = []
        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
        return data

    @classmethod
    def load_test_data(cls, fname):
        # Load test data.
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # remove the newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # If no training data is provided, skip training.
        if len(data) == 0:
            print("No training data provided. Skipping training.")
            return

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        epochs = 10  # Number of training epochs; adjust as needed.
        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            count = 0
            for line in data:
                # Skip lines that are too short.
                if len(line) < 2:
                    continue
                # Prepare input (all but last character) and target (all but first character)
                input_indices = [self.char2idx.get(c, 0) for c in line[:-1]]
                target_indices = [self.char2idx.get(c, 0) for c in line[1:]]
                # Convert lists to tensor (shape: (seq_len, 1))
                input_tensor = torch.tensor(input_indices, dtype=torch.long, device=self.device).unsqueeze(1)
                target_tensor = torch.tensor(target_indices, dtype=torch.long, device=self.device).unsqueeze(1)
                optimizer.zero_grad()
                output = self.model(input_tensor)  # output shape: (seq_len, 1, vocab_size)
                loss = F.cross_entropy(output.view(-1, len(self.vocab)), target_tensor.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            avg_loss = total_loss / count if count > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def run_pred(self, data):
        # Predict the next character by choosing the top three candidates.
        self.model.eval()
        preds = []
        with torch.no_grad():
            for inp in data:
                if len(inp) == 0:
                    # If the input is empty, choose three random characters.
                    top_chars = random.sample(self.vocab, 3)
                    preds.append(''.join(top_chars))
                    continue
                # Convert the input string into a tensor of indices.
                input_indices = [self.char2idx.get(c, 0) for c in inp]
                input_tensor = torch.tensor(input_indices, dtype=torch.long, device=self.device).unsqueeze(1)
                output = self.model(input_tensor)  # shape: (seq_len, 1, vocab_size)
                # Use the logits from the last token to predict the next character.
                last_logits = output[-1, 0, :]  # shape: (vocab_size)
                probs = F.softmax(last_logits, dim=0)
                topk = torch.topk(probs, 3)
                top_indices = topk.indices.tolist()
                top_chars = [self.idx2char[i] for i in top_indices]
                preds.append(''.join(top_chars))
        return preds

    def save(self, work_dir):
        # Save the model state dict (and vocabulary info, if needed).
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab
        }
        torch.save(checkpoint, os.path.join(work_dir, 'model.checkpoint'))

    @classmethod
    def load(cls, work_dir):
        # Instantiate a new model and load state from checkpoint.
        instance = cls()
        checkpoint = torch.load(os.path.join(work_dir, 'model.checkpoint'), map_location=instance.device)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        return instance


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))

