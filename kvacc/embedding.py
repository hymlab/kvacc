import math
import logging.config
import torch
from torch import nn as nn

from kvacc.aavocab import AAVocab

# Logger
logger = logging.getLogger('kvacc')


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=AAVocab.PAD[1])

    def forward(self, x):
        out = self.token_embedding(x)
        logger.debug('TokenEmbedding.forward: in.shape: %s, out.shape: %s' % (str(x.shape), str(out.shape)))
        return out


class SegmentEmbedding(nn.Module):
    def __init__(self, segment_size=3, embed_size=512):
        """
        Construct SegmentEmbedding for two sentence segments
        :param segment_size: the size of segment lables, including 1, 2 for sentence A, B, respectively, and padding
        :param embed_size: the embedding size
        """
        super().__init__()

        self.segment_embedding = nn.Embedding(segment_size, embed_size, padding_idx=AAVocab.PAD[1])

    def forward(self, x):
        """segments: (batch_size, seq_len)"""
        out = self.segment_embedding(x)  # (batch_size, seq_len, embed_size)
        logger.debug('PositionalEmbedding.forward: in.shape: %s, out.shape: %s' % (str(x.shape), str(out.shape)))
        return out


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        # Compute the positional encodings once in log space.
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        out = self.pe[:, :x.size(1)]
        logger.debug('PositionalEmbedding.forward: in.shape: %s, out.shape: %s' % (str(x.shape), str(out.shape)))
        return out


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, segment_size=3, embed_size=512, max_len=512, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param segment_size: the size of segment lables, including 1, 2 for sentence A, B, respectively, and padding
        :param embed_size: embedding size of token embedding
        :param max_len: maximum sequence length
        :param dropout: dropout rate
        """
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position_embedding = PositionalEmbedding(d_model=embed_size, max_len=max_len)
        self.segment_embedding = SegmentEmbedding(segment_size=segment_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment):
        out = self.token_embedding(sequence) + self.position_embedding(sequence) + self.segment_embedding(segment)
        logger.debug('BERTEmbedding.forward: in.sequence.shape: %s, in.segment.shape: %s' % (str(sequence.shape), str(segment.shape)))
        logger.debug('out.shape: %s' % str(out.shape))
        return self.dropout(out)
