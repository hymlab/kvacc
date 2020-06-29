import unittest
import logging.config
from random import random

from kvacc.commons import BaseTest, AMINO_ACIDS

# Logger
logger = logging.getLogger('kvacc')

class AAVocab(object):
    PAD =  ('[PAD]', 0)
    UNK =  ('[UNK]', 1)
    MASK = ('[MASK]', 2)
    CLS =  ('[CLS]', 3)
    SEP =  ('[SEP]', 4)
    EOS =  ('[EOS]', 5)

    SPECIAL_ENTRIES = [PAD, UNK, CLS, EOS, SEP, MASK]

    def __init__(self, itow=None, wtoi=None):
        self.itow = itow
        self.wtoi = wtoi
        self.special_words = [entry[0] for entry in AAVocab.SPECIAL_ENTRIES]
        self.special_indices = [entry[1] for entry in AAVocab.SPECIAL_ENTRIES]

    @property
    def size(self):
        return len(self.itow)

    def indices(self, str_or_words=None):
        if isinstance(str_or_words, str):
            return self.split(str_or_words, as_index=True)
        else:
            return [self.wtoi[w] for w in str_or_words]

    def split(self, seqstr=None, as_index=True):
        tokens = []
        seqlen = len(seqstr)
        i = 0
        while i < seqlen:
            word = seqstr[i]
            if word == '[':
                ic = seqstr.find(']', i)
                if ic < 0:
                    raise ValueError('No found closing bracket: ]')
                word = seqstr[i:(ic + 1)]
                i = ic + 1
            else:
                i += 1
            tokens.append(self.wtoi[word] if as_index else word)
        return tokens

    def words(self, indices=None):
        return [self.itow[i] for i in indices]

    def is_special_index(self, word_index):
        return word_index in self.special_indices

    def is_special_word(self, word):
        return word in self.special_words

    def pad(self, indices_or_tokens, n, to=None, by_index=True):
        pad_val = AAVocab.PAD[1] if by_index else AAVocab.PAD[0]
        if to == 'begin':
            return [pad_val] * n + indices_or_tokens, (n, 0)
        elif to == 'end':
            return indices_or_tokens + [pad_val] * n, (0, n)
        else: # random padding
            padded = indices_or_tokens
            n_l = n_r = 0
            for i in range(n):
                if random() < 0.5:
                    padded = [pad_val] + padded
                    n_l += 1
                else:
                    padded = padded + [pad_val]
                    n_r += 1
            return padded, (n_l, n_r)

    @classmethod
    def load_aavocab(cls, fn_vocab='../data/aavocab.txt'):
        itow = []
        wtoi = {}
        with open(fn_vocab, 'r') as f:
            for i, line in enumerate(f):
                logger.debug('%s line: %s' % (i, line))
                s = line.strip()
                itow.append(s)
                wtoi[s] = i
        return cls(itow, wtoi)

class AAVocabTest(BaseTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        logger.setLevel(logging.INFO)

    def setUp(self):
        self.vocab = AAVocab.load_aavocab()
        self.assertIsNotNone(self.vocab.itow)
        self.assertIsNotNone(self.vocab.wtoi)
        self.assertTrue(len(self.vocab.itow) > 0)
        self.assertTrue(len(self.vocab.wtoi) > 0)

    def test_vocab_size(self):
        expected = len(AMINO_ACIDS) + len(AAVocab.SPECIAL_ENTRIES)
        self.assertEqual(expected, self.vocab.size)

    def test_split(self):
        seq = AAVocab.PAD[0] + 'AK' + AAVocab.MASK[0] + 'M' + AAVocab.PAD[0]
        expected = [AAVocab.PAD[1]] + [self.vocab.wtoi['A']] +  [self.vocab.wtoi['K']] + \
                   [AAVocab.MASK[1]] + [self.vocab.wtoi['M']] + [AAVocab.PAD[1]]
        self.assertArrayEqual(expected, self.vocab.split(seq, as_index=True))

        expected = [AAVocab.PAD[0], 'A', 'K', AAVocab.MASK[0], 'M', AAVocab.PAD[0]]
        self.assertArrayEqual(expected, self.vocab.split(seq, as_index=False))

    def test_indices(self):
        seq = 'AKMLMK'
        expected = [self.vocab.wtoi[aa] for aa in list(seq)]
        self.assertArrayEqual(expected, self.vocab.indices(seq))

        seq = AAVocab.PAD[0] + 'AK' + AAVocab.MASK[0] + 'M' + AAVocab.PAD[0]
        expected = [AAVocab.PAD[1]] + [self.vocab.wtoi['A']] +  [self.vocab.wtoi['K']] + \
                   [AAVocab.MASK[1]] + [self.vocab.wtoi['M']] + [AAVocab.PAD[1]]
        self.assertArrayEqual(expected, self.vocab.indices(seq))

        words = [AAVocab.PAD[0], 'A', 'K', AAVocab.MASK[0], 'M', AAVocab.PAD[0]]
        self.assertArrayEqual(expected, self.vocab.indices(words))


    def test_words(self):
        indices = [AAVocab.PAD[1]] + [self.vocab.wtoi['A']] +  [self.vocab.wtoi['K']] + \
                  [AAVocab.MASK[1]] + [self.vocab.wtoi['M']] + [AAVocab.PAD[1]]
        expected = expected = [AAVocab.PAD[0], 'A', 'K', AAVocab.MASK[0], 'M', AAVocab.PAD[0]]

        self.assertArrayEqual(expected, self.vocab.words(indices))

    def test_is_special_entries(self):
        for i, word in enumerate(self.vocab.itow):
            if i in self.vocab.special_indices:
                self.assertTrue(self.vocab.is_special_index(i))
                self.assertTrue(self.vocab.is_special_word(word))
            else:
                self.assertFalse(self.vocab.is_special_index(i))
                self.assertFalse(self.vocab.is_special_word(i))

    def test_pad(self):
        indices = [AAVocab.PAD[1]] + [self.vocab.wtoi['A']] +  [self.vocab.wtoi['K']] + \
                  [AAVocab.MASK[1]] + [self.vocab.wtoi['M']] + [AAVocab.PAD[1]]
        tokens = [AAVocab.PAD[0], 'A', 'K', AAVocab.MASK[0], 'M', AAVocab.PAD[0]]

        expected = ([AAVocab.PAD[1]]*2 + indices, (2, 0))
        actual = self.vocab.pad(indices, 2, to='begin', by_index=True)
        self.assertArrayEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])

        expected = (indices + [AAVocab.PAD[1]]*2, (0, 2))
        actual = self.vocab.pad(indices, 2, to='end', by_index=True)
        self.assertArrayEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])

        expected = ([AAVocab.PAD[0]]*2 + tokens, (2, 0))
        actual = self.vocab.pad(tokens, 2, to='begin', by_index=False)
        self.assertArrayEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])

        expected = (tokens + [AAVocab.PAD[0]]*2, (0, 2))
        actual = self.vocab.pad(tokens, 2, to='end', by_index=False)
        self.assertArrayEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])

        actual = self.vocab.pad(indices, 2, by_index=True)
        n_lr = actual[1]
        expected = ([AAVocab.PAD[1]]*n_lr[0] + indices + [AAVocab.PAD[1]]*n_lr[1], n_lr)
        self.assertArrayEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])



if __name__ == '__main__':
    unittest.main()
