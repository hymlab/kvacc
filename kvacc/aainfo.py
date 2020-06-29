import numpy as np
import pandas as pd
import logging.config

from kvacc.commons import BaseTest, AA_INDEX

# Logger
logger = logging.getLogger('kvacc')

class AAPairwiseScoreMatrix(object):
    def __init__(self, df=None):
        if df is not None:
            if not np.array_equal(df.index, AA_INDEX):
                raise ValueError('df.index != %s' % AA_INDEX)
            if not np.array_equal(df.columns, AA_INDEX):
                raise ValueError('df.columns != %s' % AA_INDEX)

        self._df = df

    def scores(self, aa=None):
        return self._df.loc[aa, :].values


class AASubstScoreMatrix(AAPairwiseScoreMatrix):

    def __init__(self, df=None):
        super().__init__(df)

    def subst_aa(self, aa=None):
        probs = self.scores(aa)
        iaa = AA_INDEX.index(aa)

        # Scaling between 0 and 1
        probs = (probs - probs.min(axis=0)) / (probs.max(axis=0) - probs.min(axis=0))
        probs[iaa] = 0.
        default_prob = 1./len(AA_INDEX)
        probs = np.nan_to_num(probs, nan=default_prob)
        probs = probs / probs.sum(axis=0, keepdims=True)
        st_aa = np.random.choice(AA_INDEX, 1, p=probs)[0]
        return st_aa

    @classmethod
    def load_from_blosum(cls, fn_blosum='../data/blosum/blosum62.blast.new'):
        df = pd.read_table(fn_blosum, header=6, index_col=0, sep=' +')
        df = df.loc[AA_INDEX, AA_INDEX]
        df = df.transpose()
        df.index = AA_INDEX
        df.columns = AA_INDEX
        return cls(df=df)


class AASubstScoreMatrixTest(BaseTest):

    def setUp(self):
        self.subst_mat = AASubstScoreMatrix.load_from_blosum()

    def test_scores(self):
        n_aa = len(AA_INDEX)
        for aa in AA_INDEX:
            scores = self.subst_mat.scores(aa)
            self.assertIsNotNone(scores)
            self.assertEquals(n_aa, len(scores))
            logger.debug('%s: %s' % (aa, scores))

    def test_subst_aa(self):
        for i, aa in enumerate(AA_INDEX):
            st_aa = self.subst_mat.subst_aa(aa)
            self.assertIn(st_aa, AA_INDEX)
            self.assertNotEqual(aa, st_aa)
            logger.debug('%s==>%s' % (aa, st_aa))