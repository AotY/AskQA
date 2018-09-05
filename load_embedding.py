# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np
from six.moves import xrange

def load_word2vec(vocab, vec_path, vocab_size, embedding_size):
    # initial matrix with uniform
    initW = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_size))

    # load any vectors from the word2vec
    print ("Load word2vec file {}\n".format(vec_path))

    with open(vec_path, "rb") as f:
        header = f.readline()

        word2vec_vocab_size, word2vec_embedding_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * word2vec_embedding_size

        for line in xrange(word2vec_vocab_size):
            word = []
            while True:
                ch = f.read(1)

                if ch == ' ':
                    word = ''.join(word)
                    break

                if ch != '\n':
                    word.append(ch)

            idx = vocab.word_to_id(word)

            if idx != 0 :
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return initW


if __name__ == '__main__':

    # load_word2vec()
    pass