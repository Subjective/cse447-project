#!/usr/bin/env python
import os
import string
import random
import torch
import torch.nn as nn
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class CharLSTM(nn.Module):
    """
    A simple character-level LSTM for next-character prediction:
      - Embedding layer to map characters to dense vectors
      - LSTM layer to process the sequence
      - Linear output layer for prediction
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super(CharLSTM, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Embedding layer: maps each character ID -> embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

        # Output layer: projects LSTM hidden state to character logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len) of character indices
        hidden: (h, c) LSTM hidden states (optional)
        Returns:
          logits over characters, plus updated (h, c)
        """
        # (batch_size, seq_len, embed_dim)
        emb = self.embedding(x)
        # output: (batch_size, seq_len, hidden_dim)
        # hidden_out: (h, c) each of shape (1, batch_size, hidden_dim)
        output, hidden_out = self.lstm(emb, hidden)
        # project each timestep to vocabulary logits
        # (batch_size, seq_len, vocab_size)
        logits = self.fc(output)
        return logits, hidden_out


class MyModel:
    """
    Character LSTM model for next-character prediction.
    """

    def __init__(self):
        self.model = None
        self.vocab = None  # {char: idx}
        self.idx2char = None  # reverse mapping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def load_training_data(cls):
        """
        Load training data from data file.
        """
        data_file = 'src/training_data.txt'
        if not os.path.exists(data_file):
            print(f"[WARN] {data_file} not found. No training data loaded.")
            return []

        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
        return data

    @classmethod
    def load_test_data(cls, fname):
        """
        Reads test lines from the file.
        """
        data = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                inp = line.rstrip('\n')
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """
        Writes predictions (one line per input).
        """
        with open(fname, 'w', encoding='utf-8') as f:
            for p in preds:
                f.write(f"{p}\n")

    def build_vocab(self, data):
        """
        Builds a character-level vocabulary from the training data.
        """
        chars = set()
        for line in data:
            for ch in line:
                chars.add(ch)
        extra_chars = string.ascii_letters + string.digits + string.punctuation + " "
        for ch in extra_chars:
            chars.add(ch)

        # Sort to have a consistent ordering
        sorted_chars = sorted(list(chars))
        self.vocab = {ch: idx for idx, ch in enumerate(sorted_chars)}
        self.idx2char = {idx: ch for ch, idx in self.vocab.items()}

    def sequence_to_tensor(self, seq):
        """
        Convert a string sequence to a tensor of indices.
        """
        indices = [self.vocab.get(ch, 0) for ch in seq]  # 0 if char not in vocab
        return torch.tensor(indices, dtype=torch.long)

    def tensor_to_sequence(self, tensor):
        """
        Convert a tensor of indices back to a string sequence.
        """
        return ''.join([self.idx2char[int(idx)] for idx in tensor])

    def run_train(self, data, work_dir):
        """
        1. Build a vocabulary from training data
        2. Initialize CharLSTM model
        3. Train on next-char prediction
        4. Save model
        """
        if not data:
            print("[WARN] No training data available. Training aborted.")
            return

        # 1) Build vocab
        self.build_vocab(data)
        vocab_size = len(self.vocab)

        # 2) Initialize model
        self.model = CharLSTM(vocab_size=vocab_size, embed_dim=128, hidden_dim=128).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 3) Prepare examples for next-char prediction
        # Each line: we create pairs (input_seq, target_seq) for training
        all_examples = []
        for line in data:
            if len(line) < 2:
                continue
            # Example: "hello"
            # input: "hell", target: "ello"
            inp_seq = line[:-1]
            tgt_seq = line[1:]
            all_examples.append((inp_seq, tgt_seq))

        # If no valid pairs, abort
        if not all_examples:
            print("[WARN] No valid training pairs. Aborting.")
            return

        # TODO: Hyperparameter tuning to determine optimal epochs, batch_size, etc.
        self.model.train()
        num_epochs = 10
        for epoch in range(num_epochs):
            random.shuffle(all_examples)
            total_loss = 0.0
            for inp_seq, tgt_seq in all_examples:
                x_tensor = self.sequence_to_tensor(inp_seq).unsqueeze(0).to(self.device)
                y_tensor = self.sequence_to_tensor(tgt_seq).to(self.device)
                # x_tensor: shape (1, seq_len)
                # y_tensor: shape (seq_len,)

                self.model.zero_grad()
                logits, _ = self.model(x_tensor)  # (1, seq_len, vocab_size)

                # We want to compare logits against y_tensor for each timestep
                # reshape logits to (seq_len, vocab_size)
                logits = logits.squeeze(0)  # shape (seq_len, vocab_size)

                loss = criterion(logits, y_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(all_examples)
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # 4) Save model
        self.save(work_dir)

    def run_pred(self, data):
        """
        Predict next character for each line in `data`.
        We only predict ONE next character, then pick top-3 likely chars.
        """
        if not self.model or not self.vocab:
            # fallback if no model
            print("[WARN] No model loaded, returning random predictions.")
            return [''.join(random.choices(string.ascii_letters, k=3)) for _ in data]

        preds = []
        self.model.eval()
        with torch.no_grad():
            for line in data:
                # if line is empty, fallback
                if not line:
                    # just pick top 3 chars from vocab
                    top3 = list(self.vocab.keys())[:3]
                    preds.append(''.join(top3))
                    continue

                # Convert line to tensor
                x_tensor = self.sequence_to_tensor(line).unsqueeze(0).to(self.device)
                # Pass entire line to the LSTM
                logits, _ = self.model(x_tensor)
                # The last timestep is the one that predicts next char
                # shape: (1, seq_len, vocab_size)
                # We want the final step's logits: (vocab_size,)
                final_step_logits = logits[0, -1, :]  # shape (vocab_size,)

                # Get probabilities
                probs = torch.softmax(final_step_logits, dim=0)
                # Top 3
                top3_indices = torch.topk(probs, 3).indices
                top3_chars = [self.idx2char[int(idx)] for idx in top3_indices]
                preds.append(''.join(top3_chars))

        return preds

    def save(self, work_dir):
        """
        Save model and vocab.
        """
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')

        # We'll store the entire model state plus vocab in a dict
        save_dict = {
            'model_state': self.model.state_dict() if self.model else None,
            'vocab': self.vocab,
            'idx2char': self.idx2char
        }
        torch.save(save_dict, checkpoint_path)
        print(f"[INFO] Model saved to {checkpoint_path}")

    @classmethod
    def load(cls, work_dir):
        """
        Load model and vocab from checkpoint.
        """
        model_instance = cls()
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        if not os.path.exists(checkpoint_path):
            print(f"[WARN] No checkpoint found at {checkpoint_path}. Returning blank model.")
            return model_instance

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_instance.vocab = checkpoint.get('vocab', None)
        model_instance.idx2char = checkpoint.get('idx2char', None)

        if model_instance.vocab:
            # Recreate LSTM model
            vocab_size = len(model_instance.vocab)
            model_instance.model = CharLSTM(vocab_size=vocab_size)
            model_instance.model.load_state_dict(checkpoint['model_state'])
            # Move to GPU if available
            model_instance.model.to(model_instance.device)
        else:
            print("[WARN] Missing vocab in checkpoint. Model won't predict properly.")

        return model_instance


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
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
        assert len(pred) == len(test_data), \
            f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
