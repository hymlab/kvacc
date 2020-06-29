import torch
import unittest
from torch.utils.data import Dataset
from enum import auto
from unittest import TestCase
import pandas as pd
import numpy as np
from functools import reduce
import re
import warnings
import logging.config
from random import random, randint

from kvacc.commons import BaseTest, AMINO_ACIDS, BindLevel, StrEnum
from kvacc.cdomain import PanMHCIContactDomain
from kvacc.mhcnc import MHCAlleleName
from kvacc.aavocab import AAVocab
from kvacc.bioseq import is_valid_aaseq

# Logger
logger = logging.getLogger('kvacc')

IC50_THRESHOLD = 500
IEDB_BIND_LEVELS = ['NEGATIVE', 'POSITIVE-LOW', 'POSITIVE-INTERMEDIATE', 'POSITIVE', 'POSITIVE-HIGH']

class PeptideMHCBindDataFrameColumn(StrEnum):
    original_allele = auto()
    allele = auto()
    pep_seq = auto()
    pep_len = auto()
    meas_value = auto()
    meas_method = auto()
    assay_type = auto()
    meas_units = auto()
    meas_inequality = auto()
    bind_level = auto()
    binder = auto()
    antigen = auto()
    organism = auto()
    pubmed_id = auto()
    journal = auto()
    authors = auto()
    pub_date = auto()
    title = auto()

    @classmethod
    def values(cls):
        return [c.value for c in cls]

def load_df(fn_kim2014=None, fn_iedb=None, fn_systemhc=None, fn_sarkizova2019=None, select_patterns=None):
    # def select_one_with_major_label(df):
    #     new_df = pd.DataFrame(columns=df.columns)
    #     if df.shape[0] > 1:
    #         DC = self.Column
    #         n_binder = df[DC.binder].sum()
    #         n_nonbinder = df.shape[0] - n_binder
    #         if n_binder > n_nonbinder:
    #             new_df = new_df.append(df[df[DC.binder]].iloc[0, :])
    #         elif n_binder < n_nonbinder:
    #             new_df = new_df.append(df[~df[DC.binder]].iloc[0, :])
    #     else:
    #         return df
    #         # new_df = new_df.append(df)
    #     return new_df

    dfs = []
    if fn_kim2014 is not None:
        dfs.append(load_df_kim2014(fn=fn_kim2014))

    if fn_iedb is not None:
        dfs.append(load_df_iedb(fn=fn_iedb))

    if fn_systemhc is not None:
        dfs.append(load_df_systemhc(fn=fn_systemhc))

    if fn_sarkizova2019 is not None:
        dfs.append(load_df_sarkizova2019(fn=fn_sarkizova2019))

    df = pd.concat(dfs)
    logger.debug('Concatenated data_tab.shape: %s' % str(df.shape))

    if select_patterns is not None:
        matches = None
        for cname, pattern in select_patterns.items():
            cur_matches = df[cname].map(lambda x: (re.match(pattern, '%s' % x) is not None))
            if matches is None:
                matches = cur_matches
            else:
                matches = np.logical_and(matches, cur_matches)

        logger.debug('Selecting rows by patterns: %s, %s' % (str(df.shape), select_patterns))
        df = df[matches]
        logger.debug('Current data_tab.shape: %s' % str(df.shape))

    # Select only one entry of the major label in duplicated cases with the same {allele, peptide} pair
    # logger.debug('Selecting only one entry in duplicated cases with the same {allele, pep_seq}')
    # grouped = self._df.groupby([self.Column.allele, self.Column.pep_seq])
    # self._df = grouped.apply(select_one_with_major_label)
    # logger.debug('Current df.shape: %s' % str(self._df.shape))

    logger.debug('Removing duplicated records with the same {allele, pep_seq}, current df.shape: %s' % str(df.shape))
    df = df[~df.index.duplicated(keep='first')]
    logger.debug('Final df.shape: %s' % str(df.shape))
    return df

def load_df_kim2014(fn='data/bdata.20130222.mhci.txt'):
    DC = PeptideMHCBindDataFrameColumn

    df = pd.read_table(fn, names=['species', DC.allele, DC.pep_len, DC.pep_seq,
                                  DC.meas_inequality, DC.meas_value],
                       na_values=['None'], header=0)
    logger.info('Loaded kim2014 binding data from %s: %s' % (fn, str(df.shape)))
    logger.debug('Dropping NA')
    df = df.dropna()
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug('Dropping species column')
    df = df.drop(['species'], axis=1)
    logger.debug('Current df.shape: %s' % str(df.shape))

    df[DC.original_allele] = df[DC.allele].copy()
    df[DC.meas_method] = 'BA'
    df[DC.assay_type] = 'ic50'
    df[DC.meas_units] = 'nM'
    df[DC.pubmed_id] = '25017736'
    df[DC.journal] = 'BMC Bioinformatics'
    df[DC.authors] = 'Kim, Y.'
    df[DC.pub_date] = '2014'
    df[DC.title] = ' Dataset size and composition impact the reliability of performance benchmarks for peptide-MHC binding predictions'
    df[DC.bind_level] = df.meas_value.map(lambda x: BindLevel.POSITIVE if x < IC50_THRESHOLD else BindLevel.NEGATIVE)
    df[DC.binder] = df.meas_value.map(lambda x: x < IC50_THRESHOLD)
    df[DC.organism] = None
    df[DC.antigen] = None

    # Convert to the standard allele names{Robinson:2014ca}
    df[DC.allele] = df[DC.allele].map(MHCAlleleName.std_name)

    logger.debug("Dropping alleles with ambiguous names")
    df = df[df[DC.allele].map(MHCAlleleName.is_valid)]
    logger.debug('Current df.shape: %s' % str(df.shape))

    # Select valid peptide sequences
    logger.debug('Selecting valid peptide sequences')
    df = df[
        df[DC.pep_seq].map(lambda x: is_valid_aaseq(x))
    ]

    # Set indices to combined string allele and pep_seq and drop duplicated entries
    df.index = list(map(lambda x, y: '%s_%s' % (x, y), df[DC.allele], df[DC.pep_seq]))

    # logger.debug('Dropping duplicated entries by allele and pep_seq')
    # df = df[~df.index.duplicated(keep='first')]
    # logger.debug('Current df.shape: %s' % str(df.shape))
    #
    # # Selecting peptides of length range
    # logger.debug('Selecting peptides of length %s-%s' % (select_pep_len[0], select_pep_len[1]))
    # df = df[
    #     (df.pep_len >= select_pep_len[0]) & (df.pep_len <= select_pep_len[1])
    #     ]
    #
    # Dropping alleles with fewer than 25 entries
    # logger.debug('Dropping alleles with fewer than 25 entries')
    # grouped = df.groupby([self.TabColumn.allele])
    # df = grouped.filter(lambda x: x.shape[0] > 25)
    # logger.debug('Current df.shape: %s' % str(df.shape))

    df = df.loc[:, DC.values()]

    logger.debug('Measurement count per allele: %s' % df[DC.allele].value_counts())
    logger.debug('Measurement count per pep_len: %s' % df[DC.pep_len].value_counts())
    logger.debug('Measurement count per bind_level: %s' % df[DC.bind_level].value_counts())
    logger.debug('Measurement count per binder: %s' % df[DC.binder].value_counts())
    logger.info(df.head().to_string())
    logger.info('Final kim2014 dataset, df.shape: %s' % str(df.shape))

    return df

def load_df_iedb(fn=None):
    df = pd.read_csv(fn, skiprows=1, low_memory=False)
    logger.info("Loaded iedb data from %s: %s" % (fn, str(df.shape)))
    logger.debug("Selecting only class I")
    df = df[
        df["MHC allele class"].str.strip().str.upper() == "I"
    ]
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug('Converting to standard allele name')
    DC = PeptideMHCBindDataFrameColumn

    df[DC.original_allele] = df["Allele Name"]
    df[DC.allele] = df["Allele Name"].map(MHCAlleleName.std_name)
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug("Dropping mutant alleles")
    df = df[
        (~df[DC.allele].str.contains("mutant")) &
        (~df[DC.allele].str.contains("CD1"))
        ]
    logger.debug('Current df.shape: %s' % str(df.shape))

    # invalid_alleles = np.unique(iedb_df.allele[~iedb_df['allele'].map(MHCAlleleName.is_valid)])
    # print('Invalid allele names:', invalid_alleles)
    logger.debug("Dropping alleles with ambiguous names")
    df = df[df[DC.allele].map(MHCAlleleName.is_valid)]
    logger.debug('Current df.shape: %s' % str(df.shape))

    # Select valid peptide sequences
    logger.debug('Selecting valid peptide sequences')
    df[DC.pep_seq] = df['Description'].str.strip()
    df = df[
        df[DC.pep_seq].map(lambda x: is_valid_aaseq(x))
    ]
    df[DC.pep_len] = df[DC.pep_seq].map(lambda x: len(x))
    logger.debug('Current df.shape: %s' % str(df.shape))

    # Set indices to combined string allele and pep_seq and drop duplicated entries
    df.index = list(map(lambda x, y: '%s_%s' % (x, y), df[DC.allele], df[DC.pep_seq]))

    # logger.debug('Dropping duplicated entries by allele and pep_seq')
    # df = df[~df.index.duplicated(keep='first')]
    # logger.debug('Current df.shape: %s' % str(df.shape))

    # # Selecting peptides of length 8-15
    # logger.debug('Selecting peptides of length 8-15')
    # df = df[
    #     (df.pep_len >= 8) & (df.pep_len <= 15)
    # ]

    # # Dropping alleles with fewer than 25 entries
    # logger.debug('Dropping alleles with fewer than 25 entries')
    # grouped = df.groupby(['allele'])
    # df = grouped.filter(lambda x: x.shape[0] > 25)
    # logger.debug('Current df.shape: %s' % str(df.shape))

    # Measurement labels and values
    df[DC.meas_value] = df['Quantitative measurement']
    df[DC.meas_method] = df['Method/Technique']
    df[DC.assay_type] = df['Assay Group']
    df[DC.meas_units] = df['Units']
    df[DC.meas_inequality] = df['Measurement Inequality']
    df[DC.bind_level] = df['Qualitative Measure'].map(lambda x: BindLevel(IEDB_BIND_LEVELS.index(x.upper())))
    df[DC.binder] = df[DC.bind_level].map(BindLevel.is_binder)

    df[DC.pubmed_id] = df['PubMed ID']
    df[DC.journal] = df['Journal']
    df[DC.authors] = df['Authors']
    df[DC.pub_date] = df['Date']
    df[DC.title] = df['Title']
    df[DC.organism] = df['Organism Name']
    df[DC.antigen] = df['Antigen Name']

    df = df.loc[:, DC.values()]

    logger.debug('Measurement count per allele: %s' % df[DC.allele].value_counts())
    logger.debug('Measurement count per pep_len: %s' % df[DC.pep_len].value_counts())
    logger.debug('Measurement count per bind_level: %s' % df[DC.bind_level].value_counts())
    logger.debug('Measurement count per binder: %s' % df[DC.binder].value_counts())
    logger.info(df.head().to_string())
    logger.info('Final IEDB dataset, df.shape: %s' % str(df.shape))
    return df

def load_df_systemhc(fn=None):
    df = pd.read_csv(fn)
    logger.info("Loaded SysteMHC data from %s: %s" % (fn, str(df.shape)))
    logger.debug("Selecting only class I")
    df = df[
        df["MHCClass"].str.strip().str.upper() == "I"
        ]
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug('Dropping rows with N/A allele or pep_seq')
    df = df.dropna(subset=['top_allele', 'search_hit'])
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug("Selecting high confidence prob >= 0.99")
    df = df[df.prob >= 0.99]
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug('Converting to standard allele name')
    DC = PeptideMHCBindDataFrameColumn
    # TODO: top_allele have multiple alleles such as HLA-A02:01;HLA-B39:24;HLA-C07:01,
    # it could be necessary to insert multiple rows; HLA-A02:01_seq, HLA-B39:24_seq, HLA-C07:01_seq
    df[DC.original_allele] = df['top_allele'].map(lambda x: x.split(';')[0].strip())
    df[DC.allele] = df[DC.original_allele].map(MHCAlleleName.std_name)
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug("Dropping alleles with ambiguous names")
    df = df[df[DC.allele].map(MHCAlleleName.is_valid)]
    logger.debug('Current df.shape: %s' % str(df.shape))

    # Select valid peptide sequences
    logger.debug('Selecting valid peptide sequences')
    df[DC.pep_seq] = df['search_hit'].str.strip()
    df = df[
        df[DC.pep_seq].map(lambda x: is_valid_aaseq(x))
    ]
    df[DC.pep_len] = df[DC.pep_seq].map(lambda x: len(x))
    logger.debug('Current df.shape: %s' % str(df.shape))

    # Set indices to combined string allele and pep_seq and drop duplicated entries
    df.index = list(map(lambda x, y: '%s_%s' % (x, y), df[DC.allele], df[DC.pep_seq]))


    df[DC.meas_method] = 'cellular MHC/mass spectrometry'
    df[DC.assay_type] = 'mass spectrometry'
    df[DC.meas_units] = None
    df[DC.meas_value] = None
    df[DC.meas_units] = None
    df[DC.meas_inequality] = None
    df[DC.bind_level] = BindLevel.POSITIVE
    df[DC.binder] = True

    df[DC.pubmed_id] = '28985418'
    df[DC.journal] = 'Nucleic Acids Research'
    df[DC.authors] = 'Wenguang Shao'
    df[DC.pub_date] = '2017'
    df[DC.title] = 'The SysteMHC Atlas project'
    df[DC.organism] = df['Organism']
    df[DC.antigen] = None

    df = df.loc[:, DC.values()]

    logger.debug('Measurement count per allele: %s' % df[DC.allele].value_counts())
    logger.debug('Measurement count per pep_len: %s' % df[DC.pep_len].value_counts())
    logger.debug('Measurement count per bind_level: %s' % df[DC.bind_level].value_counts())
    logger.debug('Measurement count per binder: %s' % df[DC.binder].value_counts())
    logger.info(df.head().to_string())
    logger.info('Final SysteMHC Atlas dataset, df.shape: %s' % str(df.shape))

    return df

def load_df_sarkizova2019(fn=None):
    df = pd.read_csv(fn)
    logger.info("Loaded Sarkizova2019 data from %s: %s" % (fn, str(df.shape)))

    logger.debug('Dropping rows with N/A allele or pep_seq')
    df = df.dropna(subset=['allele', 'sequence'])
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug('Converting to standard allele name')
    DC = PeptideMHCBindDataFrameColumn
    df[DC.original_allele] = df['allele']
    df[DC.allele] = df[DC.original_allele].map(MHCAlleleName.std_name)
    logger.debug('Current df.shape: %s' % str(df.shape))

    logger.debug("Dropping alleles with ambiguous names")
    df = df[df[DC.allele].map(MHCAlleleName.is_valid)]
    logger.debug('Current df.shape: %s' % str(df.shape))

    # Select valid peptide sequences
    logger.debug('Selecting valid peptide sequences')
    df[DC.pep_seq] = df['sequence'].str.strip().str.upper()
    df = df[
        df[DC.pep_seq].map(lambda x: is_valid_aaseq(x))
    ]
    df[DC.pep_len] = df[DC.pep_seq].map(lambda x: len(x))
    logger.debug('Current df.shape: %s' % str(df.shape))

    # Set indices to combined string allele and pep_seq and drop duplicated entries
    df.index = list(map(lambda x, y: '%s_%s' % (x, y), df[DC.allele], df[DC.pep_seq]))


    df[DC.meas_method] = 'cellular MHC/mass spectrometry'
    df[DC.assay_type] = 'mass spectrometry'
    df[DC.meas_units] = None
    df[DC.meas_value] = None
    df[DC.meas_units] = None
    df[DC.meas_inequality] = None
    df[DC.bind_level] = BindLevel.POSITIVE
    df[DC.binder] = True

    df[DC.pubmed_id] = '31844290'
    df[DC.journal] = 'Nature Biotechnology'
    df[DC.authors] = 'Siranush Sarkizova'
    df[DC.pub_date] = '2019'
    df[DC.title] = 'A large peptidome dataset improves HLA class I epitope prediction across most of the human population'
    df[DC.organism] = 'Human'
    df[DC.antigen] = None

    df = df.loc[:, DC.values()]

    logger.debug('Measurement count per allele: %s' % df[DC.allele].value_counts())
    logger.debug('Measurement count per pep_len: %s' % df[DC.pep_len].value_counts())
    logger.debug('Measurement count per bind_level: %s' % df[DC.bind_level].value_counts())
    logger.debug('Measurement count per binder: %s' % df[DC.binder].value_counts())
    logger.info(df.head().to_string())
    logger.info('Final Sarkizova 2019 dataset, df.shape: %s' % str(df.shape))

    return df


class MaskedPeptideMHCBindSentenceDataset(Dataset):
    def __init__(self, df=None,
                 cdomain=None,
                 aavocab=None,
                 max_pep_len=15,
                 pepseq_mask_ratio=0.2,
                 mhcseq_mask_ratio=0.2,
                 pepseq_mask_probs=(0.5, 0.4),  # (deletion, substitution)
                 mhcseq_mask_probs=(0.4, 0.5)):

        self._df = df
        self._cdomain = cdomain
        self._aavocab = aavocab
        self._max_pep_len = max_pep_len
        self._pepseq_mask_ratio = pepseq_mask_ratio
        self._mhcseq_mask_ratio = mhcseq_mask_ratio
        self._pepseq_mask_probs = pepseq_mask_probs
        self._mhcseq_mask_probs = mhcseq_mask_probs

    def __getitem__(self, index):
        DC = PeptideMHCBindDataFrameColumn
        row = self._df.iloc[index, :]
        allele = row[DC.allele]
        pep_seq = row[DC.pep_seq]
        is_binder = 1 if row[DC.binder] else 0
        sequence_indices, segment_indices, target_indices = self.masked_seq_indices(allele, pep_seq)
        input = (torch.tensor(sequence_indices), torch.tensor(segment_indices))
        target = (torch.tensor(target_indices), torch.tensor(is_binder))
        return input, target

    def masked_seq_indices(self, allele, pep_seq):
        logger.info('>>>masked_seq_indices(allele, pep_seq): %s, %s' % (allele, pep_seq))
        pep_len = len(pep_seq)
        css = self._cdomain.contact_sites(pep_len)
        hsites = sorted(np.unique([cs[1] for cs in css]))
        mhc_seq = self._cdomain.domain_seq(allele)
        mhc_seq = ''.join([mhc_seq[i] for i in hsites])
        logger.debug('>>>mhc_seq, hsites: %s, %s' % (mhc_seq, hsites))

        n_pads = (self._max_pep_len - pep_len)
        if n_pads < 0:
            raise ValueError('Longer peptide length than max_pep_len: %s>%s' % (pep_len, self._max_pep_len))

        # Select masking positions in both pep_seq and mhc_seq
        pepseq_pssm = self._cdomain.pepseq_pssm(allele, pep_len)
        mhcseq_pssm = self._cdomain.domain_pssm(allele, hsites)

        pepseq_mask_positions = self._choice_mask_positions(pep_len, pepseq_pssm, mask_ratio=self._pepseq_mask_ratio)
        mhcseq_mask_positions = self._choice_mask_positions(len(mhc_seq), mhcseq_pssm, mask_ratio=self._mhcseq_mask_ratio)

        # Get masked sequence indices both pep_seq and mhc_seq at masking positions and
        # target_indices which have original token indices at the masked positions and pad index at the other positions
        masked_pepseq_indices, target_pepseq_indices = self._mask_pepseq_indices(pep_seq,
                                                                                 pepseq_mask_positions,
                                                                                 pepseq_pssm)
        masked_mhcseq_indices, target_mhcseq_indices = self._mask_mhcseq_indices(mhc_seq,
                                                                                 mhcseq_mask_positions,
                                                                                 self._cdomain.aasubst_matrix())

        masked_pepseq_indices = [AAVocab.CLS[1]] + masked_pepseq_indices + [AAVocab.SEP[1]]
        masked_mhcseq_indices = masked_mhcseq_indices + [AAVocab.EOS[1]]

        target_pepseq_indices = [AAVocab.PAD[1]] + target_pepseq_indices + [AAVocab.PAD[1]]
        target_mhcseq_indices = target_mhcseq_indices + [AAVocab.PAD[1]]

        segment_indices = [1 for _ in range(len(masked_pepseq_indices))] + \
                          [2 for _ in range(len(masked_mhcseq_indices))]

        sequence_indices = masked_pepseq_indices + masked_mhcseq_indices + [AAVocab.PAD[1]]*n_pads
        segment_indices  = segment_indices + [AAVocab.PAD[1]]*n_pads
        target_indices   = target_pepseq_indices + target_mhcseq_indices + [AAVocab.PAD[1]]*n_pads

        return sequence_indices, segment_indices, target_indices

    def _mhc_seq(self, allele, pep_len):
        return self._cdomain.contact_site_seq(allele, pep_len)

    def __len__(self):
        return self._df.shape[0]

    def _choice_mask_positions(self, seqlen, pssm, mask_ratio):
        '''
        Select masked positions of the sequence. We want our model to better predict more variable positions using pssm
        :param seqlen: the sequence length
        :param pssm: PositionSpecificScoringMatrix(# of AAs x seqlen)
        :return: list of AA positions
        '''
        if seqlen != len(pssm):
            raise ValueError('seqlen should be equal to len(pssm)' % (seqlen, len(pssm)))

        scores = pssm.conservation_scores()
        # Scaling between 0 and 1
        scores = (scores - scores.min(axis=0))/(scores.max(axis=0) - scores.min(axis=0))
        probs = 1 - scores
        probs = probs/probs.sum(axis=0)
        n_masks = round(seqlen * mask_ratio)
        return sorted(np.random.choice(seqlen, n_masks, replace=False, p=probs))

    def _mask_pepseq_indices(self, seq, mask_positions, pssm):
        masked_indices = self._aavocab.indices(seq)
        target_indices = [AAVocab.PAD[1]]*len(seq)
        for pos in mask_positions:
            target_indices[pos] = masked_indices[pos]

            r = random()
            if r < self._pepseq_mask_probs[0]:
                # Deletion
                masked_indices[pos] = AAVocab.MASK[1]
            elif r < (self._pepseq_mask_probs[0] + self._pepseq_mask_probs[1]):
                # Substitution
                new_aa = pssm.subst_aa_at(pos, seq[pos])
                masked_indices[pos] = self._aavocab.wtoi[new_aa]

        return masked_indices, target_indices

    def _mask_mhcseq_indices(self, seq, mask_positions, subst_mat):
        masked_indices = self._aavocab.indices(seq)
        target_indices = [AAVocab.PAD[1]]*len(seq)
        for pos in mask_positions:
            target_indices[pos] = masked_indices[pos]

            r = random()
            if r < self._mhcseq_mask_probs[0]:
                # Deletion
                masked_indices[pos] = AAVocab.MASK[1]
            elif r < (self._mhcseq_mask_probs[0] + self._mhcseq_mask_probs[1]):
                # Substitution
                new_aa = subst_mat.subst_aa(seq[pos])
                masked_indices[pos] = self._aavocab.wtoi[new_aa]

        return masked_indices, target_indices

class DataFrameLoadTest(BaseTest):
    @classmethod
    def setUpClass(cls):
        super(DataFrameLoadTest, cls).setUpClass()
        logger.setLevel(logging.INFO)

    def setUp(self):
        self.expected_columns = PeptideMHCBindDataFrameColumn.values()
        self.n_sample_kim2014 = 10000
        self.fn_kim2014 = '../data/bdata.20130222.mhci.sample.txt'
        self.fn_iedb = '../data/mhc_ligand_full.sample.csv'
        self.fn_systemhc = '../data/systemhcatlas_180409/data.sample.csv'
        self.fn_sarkizova2019 = '../data/Sarkizova_NatBiotech2019/data_HLA-I_95.sample.csv'

    def is_valid_index(self, index):
        tokens = index.split('_')
        allele = tokens[0]
        pep_seq = tokens[1]

        return MHCAlleleName.is_valid(allele) and is_valid_aaseq(pep_seq)

    def assert_df_loaded(self, df):
        self.assertTrue(np.array_equal(df.columns, self.expected_columns))
        self.assertTrue(df.shape[0] > 0)
        self.assertTrue(all(df.index.map(lambda x: self.is_valid_index(x))))
        self.assertTrue(all(df[PeptideMHCBindDataFrameColumn.allele].map(MHCAlleleName.is_valid)))
        self.assertTrue(all(df[PeptideMHCBindDataFrameColumn.pep_seq].map(lambda x: is_valid_aaseq(x))))

    def test_load_df_kim2014(self):
        df = load_df_kim2014(fn=self.fn_kim2014)
        self.assert_df_loaded(df)

    def test_load_df_iedb(self):
        df = load_df_iedb(fn=self.fn_iedb)
        self.assert_df_loaded(df)

    def test_load_df_systemhc(self):
        df = load_df_systemhc(fn=self.fn_systemhc)
        self.assert_df_loaded(df)

    def test_load_df_sarkizova2019(self):
        df = load_df_sarkizova2019(fn=self.fn_sarkizova2019)
        self.assert_df_loaded(df)

    def test_load_df(self):
        DC = PeptideMHCBindDataFrameColumn

        df_kim2014 = load_df_kim2014(fn=self.fn_kim2014)
        df_iedb = load_df_iedb(fn=self.fn_iedb)
        df_systemhc = load_df_systemhc(fn=self.fn_systemhc)
        df_sarkizova2019 = load_df_sarkizova2019(fn=self.fn_sarkizova2019)

        alleles_kim2014 = df_kim2014[DC.allele].unique()
        alleles_iedb = df_iedb[DC.allele].unique()
        alleles_systemhc = df_systemhc[DC.allele].unique()
        alleles_sarkizova2019 = df_sarkizova2019[DC.allele].unique()

        df_all = load_df(fn_kim2014=self.fn_kim2014,
                         fn_iedb=self.fn_iedb,
                         fn_systemhc=self.fn_systemhc,
                         fn_sarkizova2019=self.fn_sarkizova2019)

        self.assert_df_loaded(df_all)

        self.assertTrue(df_all.shape[0] > df_kim2014.shape[0])
        self.assertTrue(df_all.shape[0] > df_iedb.shape[0])
        self.assertTrue(df_all.shape[0] > df_systemhc.shape[0])
        self.assertTrue(df_all.shape[0] > df_sarkizova2019.shape[0])

        alleles_all = df_all[DC.allele].unique()
        alleles_merged = reduce(np.union1d, (alleles_kim2014, alleles_iedb, alleles_systemhc, alleles_sarkizova2019))
        self.assertSetEqual(set(alleles_all), set(alleles_merged))

        # Assert indices duplicated
        self.assertTrue(all(~df_all.index.duplicated(keep=False)))

    def test_load_data_tab_with_select_patterns(self):
        DC = PeptideMHCBindDataFrameColumn
        allele_pattern = '^HLA-[ABC]\*[0-9]{2}:[0-9]{2}$'
        pep_len_pattern = '^([8-9]|1[0-5])$'

        df = load_df(fn_kim2014=self.fn_kim2014, fn_iedb=self.fn_iedb)

        alleles = df[DC.allele].unique()
        pep_lens = df[DC.pep_len].unique()

        allele_matches = list(map(lambda x: re.match(allele_pattern, '%s' % x) is None, alleles))
        pep_len_matches = list(map(lambda x: re.match(pep_len_pattern, '%s' % x) is None, pep_lens))

        self.assertTrue(any(allele_matches))
        self.assertTrue(any(pep_len_matches))

        patterns = {
            DC.allele: allele_pattern,
            DC.pep_len: pep_len_pattern
        }

        df = load_df(fn_kim2014=self.fn_kim2014, fn_iedb=self.fn_iedb, select_patterns=patterns)

        alleles = df[DC.allele].unique()
        pep_lens = df[DC.pep_len].unique()

        allele_matches = list(map(lambda x: re.match(allele_pattern, '%s' % x) is not None, alleles))
        pep_len_matches = list(map(lambda x: re.match(pep_len_pattern, '%s' % x) is not None, pep_lens))

        self.assertTrue(all(allele_matches))
        self.assertTrue(all(pep_len_matches))

### Tests
class MaskedPeptideMHCBindSentenceDatasetTest(BaseTest):
    def setUp(self):
        logger.setLevel(logging.DEBUG)

        self.cdomain = PanMHCIContactDomain()
        self.cdomain.set_contact_sites(9, PanMHCIContactDomain.NETMHCPAN_MHCI_9_CONTACT_SITES)
        self.aavocab = AAVocab.load_aavocab()
        self.max_pep_len = 15
        self.mask_ratio = 0.2

    def test_masked_seq_indices(self):
        ds = MaskedPeptideMHCBindSentenceDataset(df=None, cdomain=self.cdomain,
                                                 aavocab=self.aavocab, max_pep_len=self.max_pep_len,
                                                 pepseq_mask_ratio=self.mask_ratio,
                                                 mhcseq_mask_ratio=self.mask_ratio)
        allele = 'HLA-A*03:01'
        pep_seq = 'VQQQRQEQ'
        pep_len = len(pep_seq)
        pepseq_indices = self.aavocab.indices(pep_seq)

        hla_seq = self.cdomain.contact_site_seq(allele, pep_len)
        hlaseq_indices = self.aavocab.indices(hla_seq)

        n_hsites = len(self.cdomain.all_hla_sites)

        sequence_indices, segment_indices, target_indices = ds.masked_seq_indices(allele, pep_seq)
        print('sequence_indices:', sequence_indices)
        print('segment_indices :', segment_indices)
        print('target_indices  :', target_indices)

        expected_input_len = self.max_pep_len + n_hsites + 3

        self.assertEqual(expected_input_len, len(sequence_indices))
        self.assertEqual(expected_input_len, len(segment_indices))
        self.assertEqual(expected_input_len, len(target_indices))

        # self.assertEqual(expected_n_masks, len(target_token_indices))
        # self.assertEqual(expected_n_masks, len(mask_positions))

    def test_get_item_for_sample(self):
        train_csv = '../output/pretrain_data_HLA-ABC_plen8-15.sample.csv'
        df = pd.read_csv(train_csv, index_col=0)
        ds = MaskedPeptideMHCBindSentenceDataset(df=df, cdomain=self.cdomain,
                                                 aavocab=self.aavocab, max_pep_len=self.max_pep_len,
                                                 pepseq_mask_ratio=self.mask_ratio,
                                                 mhcseq_mask_ratio=self.mask_ratio)
        DC = PeptideMHCBindDataFrameColumn
        for i in range(df.shape[0]):
            row = df.iloc[i, :]
            allele = row[DC.allele]
            pep_seq = row[DC.pep_seq]
            print('>>>%s, %s, %s' % (i, allele, pep_seq))
            print(ds.masked_seq_indices(allele, pep_seq))

if __name__ == '__main__':
    unittest.main()
