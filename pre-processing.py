# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

# word -> id
#   matrix -> [|V|, embed_size]
#   word A -> id(5)
#   word vocabulary
#   label -> id

import sys
import os
import codecs
import paths

def generate_vocab_file(input_file, output_vocab_file, cur_word_dict=None):
    # in order to append word
    if cur_word_dict is None:
        word_dict = {}
    else:
        word_dict = cur_word_dict

    with codecs.open(input_file, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            content = line.strip('\r\n')
            for word in content.split():
                word_dict.setdefault(word, 0)
                word_dict[word] += 1

    if cur_word_dict is not None:
        # [(word, frequency), ..., ()]
        sorted_word_dict = sorted(
            word_dict.items(), key=lambda d: d[1], reverse=True)

        with codecs.open(output_vocab_file, 'w', encoding='utf-8') as f:
            f.write('<UNK>\t10000000\n')
            for item in sorted_word_dict:
                f.write('%s\t%d\n' % (item[0], item[1]))

    print ('word size: %d' % len(word_dict))
    return word_dict

if __name__ == '__main__':
    print (paths.train_post_file)
    word_dict = generate_vocab_file(paths.train_post_file, paths.vocab_file, None)
    generate_vocab_file(paths.train_response_file, paths.vocab_file, word_dict)











