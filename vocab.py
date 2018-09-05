# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            word, frequency = line.strip('\r\n').split('\t')
            word = word

            frequency = int(frequency)

            if frequency < self._num_word_threshold:
                continue

            idx = len(self._word_to_id)

            if word == '<UNK>':
                self._unk = idx

            self._word_to_id[word] = idx

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) \
                    for cur_word in sentence.split()]
        return word_ids


if __name__ == '__main__':
    pass
