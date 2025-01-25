#!/usr/bin/env python
import os
import json
import string
from collections import defaultdict, Counter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv

class MyModel:
    """
    A bigram language model that attempts to predict
    the next character given the last one or two characters of input.

      - For each character in training data, we update:
          * total frequency of characters
          * frequency of characters following each character
          * frequency of characters following two characters in a row
      - We use our counts to pick the top 3 most likely next characters.
    """

    def __init__(self):
        # Frequency of chars appearing after one letter
        self.next_char_freq_1 = defaultdict(Counter)

        # Frequency of chars appearing after two letters
        self.next_char_freq_2 = defaultdict(Counter)

        # Total chars
        self.total_char_counts = Counter()

        # Looks for all lowercase + uppercase letters + digits + punctuation.
        self.allowed_chars = (
            string.ascii_letters + string.digits + string.punctuation
        )

    @classmethod
    def load_training_data(cls):
        """
        Training data
        """
        training_lines = []
        with open('training.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        for token, freq in data.items():
            training_lines.extend([token] * freq)

        return training_lines

    @classmethod
    def load_test_data(cls, fname):
        """
        Reads lines from a test input file.
        """
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """
        Writes one line of predictions.
        """
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        """
        Train the model.
        """
        for line in data:
            filtered_line = [c for c in line if c in self.allowed_chars]
            if not filtered_line:
                continue

            for c in filtered_line:
                self.total_char_counts[c] += 1

            for i in range(len(filtered_line) - 1):
                c1 = filtered_line[i]
                c_next = filtered_line[i + 1]
                self.next_char_freq_1[c1][c_next] += 1

            for i in range(len(filtered_line) - 2):
                c1 = filtered_line[i]
                c2 = filtered_line[i + 1]
                c_next = filtered_line[i + 2]
                self.next_char_freq_2[(c1, c2)][c_next] += 1
        for key, value in self.total_char_counts.items():
            print(f"Key: {key}, Value: {value}")
        self.save(work_dir)

    def run_pred(self, data):
        """
        For each line in 'data', we guess by:
          * If we have >=2 chars, then we use next_char_freq_2 for top 3
          * If only 1 char, then use next_char_freq_1
          * Otherwise, we use total
        """

        preds = []
        global_top_3 = [c for c, _ in self.total_char_counts.most_common(3)]
        # If less than 3 total chars trained on, then just give back "ear"
        if len(global_top_3) < 3:
           print("Not enough trained data!!!")
           global_top_3 = ['e','a','r'] 

        for inp in data:
            filtered_inp = [c for c in inp if c in self.allowed_chars]

            # if >=2 chars
            candidates = []
            if len(filtered_inp) >= 2:
                c1, c2 = filtered_inp[-2], filtered_inp[-1]
                if (c1, c2) in self.next_char_freq_2:
                    ctr = self.next_char_freq_2[(c1, c2)]
                    candidates = [c for c, _ in ctr.most_common(3)]

            # else if >= 1 chars
            if len(candidates) < 3 and len(filtered_inp) >= 1:
                c1 = filtered_inp[-1]
                if c1 in self.next_char_freq_1:
                    ctr = self.next_char_freq_1[c1]
                    needed = 3 - len(candidates)
                    new_candidates = [c for c, _ in ctr.most_common(needed) if c not in candidates]
                    candidates.extend(new_candidates)

            # else
            if len(candidates) < 3:
                needed = 3 - len(candidates)
                for c in global_top_3:
                    if c not in candidates:
                        candidates.append(c)
                    if len(candidates) == 3:
                        break

            preds.append(''.join(candidates))

        return preds

    def save(self, work_dir):
        """
        Saves the model to disk as JSON. 
        """
        path = os.path.join(work_dir, 'model.checkpoint')
        with open(path, 'w', encoding='utf-8') as f:
            # We need to convert the Counters to regular dicts to store in JSON.
            data_to_save = {
                "next_char_freq_1": {
                    c1: dict(counter) for c1, counter in self.next_char_freq_1.items()
                },
                "next_char_freq_2": {
                    "{}\t{}".format(c1[0], c1[1]): dict(counter)
                    for c1, counter in self.next_char_freq_2.items()
                },
                "total_char_counts": dict(self.total_char_counts)
            }
            json.dump(data_to_save, f, ensure_ascii=False)

    @classmethod
    def load(cls, work_dir):
        """
        Loads the model from JSON. This will reconstruct the bigram counters.
        """
        model = cls()
        path = os.path.join(work_dir, 'model.checkpoint')
        if not os.path.exists(path):
            # If there's no file, return an empty model 
            return model

        with open(path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        for c1, next_dict in saved_data["next_char_freq_1"].items():
            model.next_char_freq_1[c1] = Counter(next_dict)
        for c1_str, next_dict in saved_data["next_char_freq_2"].items():
            c1_parts = c1_str.split('\t')
            if len(c1_parts) == 2:
                key_tuple = (c1_parts[0], c1_parts[1])
                model.next_char_freq_2[key_tuple] = Counter(next_dict)
        model.total_char_counts = Counter(saved_data["total_char_counts"])

        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir)) 
            os.makedirs(args.work_dir)
        print('Instantiating model...')
        model = MyModel()
        print('Loading training data...')
        train_data = MyModel.load_training_data()
        print('Training model...')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir) 
    elif args.mode == 'test':
        print('Loading model...')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions...')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')

