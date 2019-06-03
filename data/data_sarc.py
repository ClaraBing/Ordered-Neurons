import os
import torch
from collections import Counter

from data.data_utils_sarc import sarc_reader

import pdb


class SARCDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
    
        self.total = 0

        # reserve IDs for special inputs.
        # 0: label 0
        self.add_word('0')
        # 1: label 1
        self.add_word('1')
        # 2: '<eos>'
        self.add_word('<eos>')
        # self.idx2word.append(0)
        # self.word2idx[0] = 0
        # # 1: label 1
        # self.idx2word.append(1)
        # self.word2idx[1] = 1
        # # 2: '<eos>'
        # self.idx2word.append('<eos>')
        # self.word2idx['<eos>'] = 2


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class SARCCorpus(object):
    def __init__(self, path, ftrain, ftest):
        self.dictionary = SARCDictionary()

        # prepare absolute file paths
        fcomment = os.path.join(path, 'comments.json')
        ftrain = os.path.join(path, ftrain)
        ftest = os.path.join(path, ftest)

        # tokenize files
        self.train = self.tokenize(ftrain, fcomment)
        self.test = self.tokenize(ftest, fcomment)

    def tokenize(self, fsplit, fcomment):
        """Tokenizes a text file."""
        assert os.path.exists(fsplit)
        assert os.path.exists(fcomment)
        gen = sarc_reader(fcomment, fsplit, lower=True)
        responses = []
        labels = []
        for each in gen:
            responses += each['response'],
            labels += each['label'],

        # Add words to the dictionary
        tokens = 0
        for (response, label) in zip(responses, labels):
            words = response.split() + ['<eos>', str(int(label))]
            if len(words) < 3:
                print('empty response')
                continue
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        # pdb.set_trace()
        ids = torch.LongTensor(tokens)
        token = 0
        for (response, label) in zip(responses, labels):
            words = response.split() + ['<eos>', str(int(label))]
            if len(words) < 3:
                print('empty response')
                continue
            prev_word = ''
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids
