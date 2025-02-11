#!/usr/bin/env python
import os
import random
import pickle
import numpy as np
import string
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from laserembeddings import Laser  # Import LASER

class MyModel:
    """
    Multiclass Logistic Regression with LASER character embeddings.
    This version uses LASER to generate embeddings (even though LASER is 
    optimized for sentences) so that multiple languages can be handled.
    """

    def __init__(self):
        # Holds the trained logistic regression model
        self.model = None
        # Label encoder for mapping characters <-> integer labels
        self.label_encoder = None
        # Pre-trained LASER model for multilingual embeddings
        self.laser = None
        # List of possible characters the model knows about
        self.known_chars = None

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
        X: LASER embedding of the current character
        y: Next character
        """
        X = []
        y = []
        for line in data:
            # For each pair of consecutive characters
            for i in range(len(line) - 1):
                current_char = line[i]
                next_char = line[i + 1]

                # Get LASER embedding for the current character.
                # Since LASER is optimized for sentences, we pass the single character
                # as a one-element list. We use a default language ('en')—adjust as needed.
                current_char_vec = self.laser.embed_sentences([current_char], lang='en')[0]

                X.append(current_char_vec)
                y.append(next_char)

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        return X, y

    def run_train(self, data, work_dir):
        """
        Train a logistic regression model on the provided data.
        1. Initializes the LASER model (downloads files if needed)
        2. Prepares training samples
        3. Trains logistic regression
        4. Saves model artifacts
        """
        # Initialize the LASER model (this will download necessary files if not already present)
        self.laser = Laser()

        if not data:
            print("No training data provided. Training aborted.")
            return
        X, y = self._prepare_training_samples(data)
        if len(X) == 0:
            print("No valid training pairs (X, y) could be constructed. Training aborted.")
            return

        # Train logistic regression
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

        # Save model artifacts
        self.save(work_dir)

    def run_pred(self, data):
        """
        Predict the next character for each input line and return top-3 guesses.
        """
        preds = []
        if not self.model or not self.laser or not self.label_encoder:
            print("[WARN] Model not loaded or trained. Returning random guesses.")
            all_chars = string.ascii_letters
            for _ in data:
                top_guesses = [random.choice(all_chars) for _ in range(3)]
                preds.append(''.join(top_guesses))
            return preds

        for inp in data:
            if not inp:
                if self.known_chars is not None:
                    preds.append(''.join(self.known_chars[:3]))
                else:
                    preds.append(''.join(random.choices(string.ascii_letters, k=3)))
                continue

            # Get last character as context
            last_char = inp[-1]
            # Get LASER embedding for the last character
            last_char_vec = self.laser.embed_sentences([last_char], lang='en')[0]

            # Predict probabilities over all known classes
            probs = self.model.predict_proba([last_char_vec])[0]
            # Get indices of top 3 predictions
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3_chars = self.label_encoder.inverse_transform(top3_indices)
            preds.append(''.join(top3_chars))

        return preds

    def save(self, work_dir):
        """
        Saves model artifacts to a checkpoint file using pickle.
        Note: The LASER model is not pickled, but reinitialized during loading.
        """
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'known_chars': self.known_chars
            }, f)
        print(f"[INFO] Model saved to {checkpoint_path}")

    @classmethod
    def load(cls, work_dir):
        """
        Loads the model artifacts from model.checkpoint and initializes the LASER model.
        """
        model_instance = cls()
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                model_instance.model = data.get('model')
                model_instance.label_encoder = data.get('label_encoder')
                model_instance.known_chars = data.get('known_chars')

            # Initialize the LASER model (downloads files if needed)
            model_instance.laser = Laser()
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
