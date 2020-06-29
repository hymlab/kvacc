import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging.config
import copy

from kvacc.aavocab import AAVocab
from kvacc.embedding import BERTEmbedding
from kvacc.transformer import TransformerEncoder

# Logger
logger = logging.getLogger('kvacc')

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, n_layers=12, n_heads=12, max_len=512, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len

        # BERT paper noted they used 4*hidden_size for feed-forward network hidden size
        d_ff = hidden_size * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=self.vocab_size, embed_size=self.hidden_size, max_len=self.max_len)

        # Transfomer encoder
        self.encoder = TransformerEncoder(n_layers=n_layers,
                                          d_model=hidden_size,
                                          n_heads=n_heads,
                                          d_ff=d_ff,
                                          dropout=dropout)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
        #                                            nhead=self.n_heads,
        #                                            dim_feedforward=d_ff,
        #                                            dropout=dropout)
        # self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        # self.init_weights()

    def forward(self, sequence, segment):
        # x(=sequence).shape: (batch_size, seq_len), segment_info.shape: (batch_size, seq_len)
        # attention masking for padded token
        # mask: torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        logger.debug('BERT.forward: sequence.shape: %s, segment.shape: %s' % (str(sequence.shape), str(segment.shape)))
        mask = (sequence > 0).unsqueeze(1).repeat(1, sequence.size(1), 1).unsqueeze(1)
        logger.debug('mask.shape: %s' % str(mask.shape))

        # embedding the indexed sequence to sequence of vectors
        # embedding output.shape: (batch_size, seq_len, embed_size(=hidden_size))
        out = self.embedding(sequence, segment)
        logger.debug('Embeded out.shape: %s' % str(out.shape))

        # Running over multiple transformer encoding blocks
        # Encoded out.shape: (batch_size, seq_len, embed_size)
        out = self.encoder(out, mask)

        logger.debug('Encoded out.shape: %s' % str(out.shape)) # (batch_size, seq_len, hidden_size)
        return out

    # def init_weights(self):
    #     self.embedding.token_embedding.weight.data.uniform_(-0.1, 0.1)
    #     self.embedding.segment_embedding.weight.data.uniform_(-0.1, 0.1)


    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

class BERTLM(nn.Module):
    """
    BERT Language Model: Masked Language Model + Next Sentence Prediction
    """
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.token_predictor = nn.Linear(bert.hidden_size, bert.vocab_size)
        self.classifier = nn.Linear(bert.hidden_size, 2)

    def forward(self, input):
        sequence, segment = input
        logger.debug('BERTLM.forward: in.sequence.shape: %s, in.segment.shape: %s' % (str(sequence.shape),
                                                                                  str(segment.shape)))
        bert_out = self.bert(sequence, segment)
        # bert_out.shape: (batch_size, seq_len, hidden_size
        # token_pred_out: (batch_size, seq_len, vocab_size)
        # classifier_out: (batch_size, seq_len, 2)
        token_pred_out = F.log_softmax(self.token_predictor(bert_out), dim=-1)
        # TODO: We use only the first token id([CLS]) in the sequence for classification task.
        classifier_out = F.log_softmax(self.classifier(bert_out[:, 0]), dim=-1)
        logger.debug('BERTLM.forward: bert_out.shape: %s, token_pred_out.shape: %s, classifier_out.shape: %s' %
                     (str(bert_out.shape), str(token_pred_out.shape), str(classifier_out.shape)))
        return token_pred_out, classifier_out

class BERTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.NLLLoss(ignore_index=AAVocab.PAD[1])

    def forward(self, output, target):
        token_pred_out, classifier_out = output
        target_token_indices, target_clf_labels = target
        token_pred_loss = self.criterion(token_pred_out.transpose(1, 2), target_token_indices)
        clf_loss = self.criterion(classifier_out, target_clf_labels)
        return token_pred_loss + clf_loss

### Tests
# class BERTLMTest(BaseTest):
#     def setUp(self):
#         logger.setLevel(logging.INFO)
#         self.aavocab = AAVocab.load_aavocab()
#         self.max_pep_len = 12
#         self.mask_ratio = 0.2

    # def test_bert(self):
    #     bert = BERT(vocab_size=self.aavocab.size, hidden_size=512, n_layers=6, n_heads=8, max_len=100)
    #     sequence, segment, target = self.get_bert_input(pep_seq='YMKLVPNDK', mhc_seq='GHMLDYPDSNKLNK')
    #     out = bert(sequence.unsqueeze(0), segment.unsqueeze(0))
    #
    # def get_bert_input(self, pep_seq, mhc_seq):
    #     pep_len = len(pep_seq)
    #     pepseq_indices = self.aavocab.indices(pep_seq)
    #     masked_pepseq_indices = copy.deepcopy(pepseq_indices)
    #     masked_pepseq_indices[2] = self.aavocab.wtoi['R']
    #     masked_pepseq_indices[6] = self.aavocab.MASK[1]
    #     target_pepseq_indices = [AAVocab.PAD[1]] * pep_len
    #     target_pepseq_indices[2] = self.aavocab.wtoi[pep_seq[2]]
    #     target_pepseq_indices[6] = self.aavocab.wtoi[pep_seq[6]]
    #
    #     mhc_len = len(mhc_seq)
    #     mhcseq_indices = self.aavocab.indices(mhc_seq)
    #     masked_mhcseq_indices = copy.deepcopy(mhcseq_indices)
    #     masked_mhcseq_indices[2] = self.aavocab.MASK[1]
    #     masked_mhcseq_indices[6] = self.aavocab.wtoi['A']
    #     masked_mhcseq_indices[10] = self.aavocab.MASK[1]
    #     target_mhcseq_indices = [AAVocab.PAD[1]] * mhc_len
    #     target_mhcseq_indices[2] = self.aavocab.wtoi[mhc_seq[2]]
    #     target_mhcseq_indices[6] = self.aavocab.wtoi[mhc_seq[6]]
    #     target_mhcseq_indices[10] = self.aavocab.wtoi[mhc_seq[10]]
    #
    #     masked_pepseq_indices = [AAVocab.CLS[1]] + masked_pepseq_indices + [AAVocab.SEP[1]]
    #     masked_mhcseq_indices = masked_mhcseq_indices + [AAVocab.EOS[1]]
    #
    #     target_pepseq_indices = [AAVocab.PAD[1]] + target_pepseq_indices + [AAVocab.PAD[1]]
    #     target_mhcseq_indices = target_mhcseq_indices + [AAVocab.PAD[1]]
    #
    #     segment_indices = [1 for _ in range(len(masked_pepseq_indices))] + \
    #                       [2 for _ in range(len(masked_mhcseq_indices))]
    #
    #     n_pads = self.max_pep_len - pep_len
    #
    #     sequence_indices = [AAVocab.PAD[1]]*n_pads + masked_pepseq_indices + masked_mhcseq_indices
    #     segment_indices  = [AAVocab.PAD[1]]*n_pads + segment_indices
    #     target_indices   = [AAVocab.PAD[1]]*n_pads + target_pepseq_indices + target_mhcseq_indices
    #
    #     return torch.tensor((sequence_indices, segment_indices, target_indices))
    #

if __name__ == '__main__':
    unittest.main()
