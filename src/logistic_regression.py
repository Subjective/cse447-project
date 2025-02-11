#!/usr/bin/env python
import os
import random
import pickle
import fasttext
import numpy as np
import string
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import urllib.request
import gzip
import shutil
from tqdm import tqdm


class MyModel:
    """
    Multiclass Logistic Regression with FastText character embeddings,
    automatically downloading the FastText file if it doesn't exist.
    """

    def __init__(self):
        # Holds the trained logistic regression model
        self.model = None
        # Label encoder for mapping characters <-> integer labels
        self.label_encoder = None
        # Pre-trained FastText model for multilingual embeddings
        self.ft_model = None
        # List of possible characters the model knows about
        self.known_chars = None

        # Default path and URL for multilingual FastText
        self.fasttext_model_path = "cc.all.300.bin"
        self.fasttext_model_url = (
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
        )

    def ensure_fasttext_model(self):
        """
        Checks if the FastText model file exists; if not, downloads and extracts it.
        """
        if not os.path.exists(self.fasttext_model_path):
            print(f"[INFO] FastText model not found at {self.fasttext_model_path}.")
            print(f"[INFO] Downloading from {self.fasttext_model_url} ... (this is LARGE!)")

            gz_path = self.fasttext_model_path + ".gz"
            # Download
            urllib.request.urlretrieve(self.fasttext_model_url, gz_path)
            print("[INFO] Download complete. Extracting...")

            # Extract
            with gzip.open(gz_path, 'rb') as f_in:
                with open(self.fasttext_model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)
            print("[INFO] Extraction complete.")
        else:
            print(f"[INFO] FastText model found at {self.fasttext_model_path}.")

    @classmethod
    def load_training_data(cls):
        """
        Loads training data from 'training_data.txt' if it exists.
        """
        data_file = 'src/training_data.txt'
        data = []
        if not os.path.exists(data_file):
            print(f"Warning: {data_file} not found. No training data loaded.")
            return data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(line.strip())
        return data

    @classmethod
    def load_test_data(cls, fname):
        """
        Reads test data (one input string per line).
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
        Writes predictions to a file (one line per input).
        """
        with open(fname, 'w', encoding='utf-8') as f:
            for p in preds:
                f.write(f"{p}\n")

    def _prepare_training_samples(self, data):
        """
        Create training pairs (X, y) from input lines.
        X: Embedding of current character
        y: Next character
        """
        X = []
        y = []

        for line in data:
            # For each pair of consecutive characters
            for i in range(len(line) - 1):
                current_char = line[i]
                next_char = line[i + 1]

                # Get FastText embedding for the current character
                current_char_vec = self.ft_model.get_word_vector(current_char)

                X.append(current_char_vec)
                y.append(next_char)

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        return X, y

    def run_train(self, data, work_dir):
        """
        Train a logistic regression model on the provided data.
        1. Ensures FastText model is available
        2. Loads FastText model
        3. Prepares training samples
        4. Trains logistic regression
        5. Saves model artifacts
        """
        # -----------------------------
        # 1) Ensure FastText model is available or download it
        # -----------------------------
        self.ensure_fasttext_model()

        # -----------------------------
        # 2) Load the FastText model
        # -----------------------------
        self.ft_model = fasttext.load_model(self.fasttext_model_path)

        # -----------------------------
        # 3) Prepare training samples
        # -----------------------------
        if not data:
            print("No training data provided. Training aborted.")
            return
        X, y = self._prepare_training_samples(data)
        if len(X) == 0:
            print("No valid training pairs (X, y) could be constructed. Training aborted.")
            return

        # -----------------------------
        # 4) Train logistic regression
        # -----------------------------
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        self.model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs'
        )
        self.model.fit(X, y_enc)
        print("Training complete.")

        # Keep track of known characters
        self.known_chars = self.label_encoder.classes_

        # -----------------------------
        # 5) Save model artifacts
        # -----------------------------
        self.save(work_dir)

    def run_pred(self, data):
        """
        Predict the next character for each input line and return top-3 guesses.
        """
        preds = []
        if not self.model or not self.ft_model or not self.label_encoder:
            # Fallback if model wasn't trained / loaded
            print("[WARN] Model not loaded or trained. Returning random guesses.")
            all_chars = string.ascii_letters
            for _ in data:
                top_guesses = [random.choice(all_chars) for _ in range(3)]
                preds.append(''.join(top_guesses))
            return preds

        for inp in data:
            # If input is empty, no context:
            if not inp:
                # Return top-3 frequent characters from known set, or random fallback
                if self.known_chars is not None:
                    preds.append(''.join(self.known_chars[:3]))
                else:
                    preds.append(''.join(random.choices(string.ascii_letters, k=3)))
                continue

            # 1) Get last character as context
            last_char = inp[-1]

            # 2) Get its embedding
            last_char_vec = self.ft_model.get_word_vector(last_char)

            # 3) Predict probabilities over all known classes
            probs = self.model.predict_proba([last_char_vec])[0]

            # 4) Get indices of top 3
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3_chars = self.label_encoder.inverse_transform(top3_indices)

            # Combine as a string
            preds.append(''.join(top3_chars))

        return preds

    def save(self, work_dir):
        """
        Saves model artifacts to a checkpoint file using pickle.
        """
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'known_chars': self.known_chars,
                'fasttext_model_path': self.fasttext_model_path,
                'fasttext_model_url': self.fasttext_model_url
            }, f)
        print(f"[INFO] Model saved to {checkpoint_path}")

    @classmethod
    def load(cls, work_dir):
        """
        Loads the model artifacts from model.checkpoint,
        and ensures the FastText file is available/loaded.
        """
        model_instance = cls()
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                model_instance.model = data.get('model')
                model_instance.label_encoder = data.get('label_encoder')
                model_instance.known_chars = data.get('known_chars')
                model_instance.fasttext_model_path = data.get('fasttext_model_path')
                model_instance.fasttext_model_url = data.get('fasttext_model_url')

            # Ensure the FastText model is downloaded
            model_instance.ensure_fasttext_model()

            # Load the FastText model
            if os.path.exists(model_instance.fasttext_model_path):
                model_instance.ft_model = fasttext.load_model(model_instance.fasttext_model_path)
            else:
                print(f"[WARN] Could not find {model_instance.fasttext_model_path}. Predictions may fail.")

        else:
            print(f"[WARN] No checkpoint found at {checkpoint_path}. Returning uninitialized model.")
        return model_instance


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
            print(f'[INFO] Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('[INFO] Instantiating model')
        model = MyModel()
        print('[INFO] Loading training data')
        train_data = MyModel.load_training_data()
        print(f'[INFO] Training on {len(train_data)} lines of data')
        model.run_train(train_data, args.work_dir)
        print('[INFO] Saving model')
        model.save(args.work_dir)

    elif args.mode == 'test':
        print('[INFO] Loading model')
        model = MyModel.load(args.work_dir)
        print(f'[INFO] Loading test data from {args.test_data}')
        test_data = MyModel.load_test_data(args.test_data)
        print('[INFO] Making predictions')
        pred = model.run_pred(test_data)
        print(f'[INFO] Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), \
            f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')
