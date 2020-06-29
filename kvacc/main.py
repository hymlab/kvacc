import json
from argparse import ArgumentParser
import logging.config
import warnings
import glob
import os
import pandas as pd
import numpy as np
import pickle
import copy
import json
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import unittest

from kvacc.commons import AMINO_ACIDS, basename
from kvacc.mhcnc import MHCAlleleName
from kvacc.cdomain import PanMHCIContactDomain
from kvacc.aavocab import AAVocab, AAVocabTest
from kvacc.datasets import load_df, PeptideMHCBindDataFrameColumn, MaskedPeptideMHCBindSentenceDataset
from kvacc.bioseq import PositionSpecificScoringMatrix, MultipleSequenceAlignment, write_fa
from kvacc.bert import BERT, BERTLM, BERTLMLoss
from kvacc.train import ModelTrainer
from kvacc.optimizer import NoamOptimizer

# Disable warning that can be ignored

warnings.filterwarnings("ignore")

# Logger
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger('kvacc')

def run_test_suite(args):
    suite = unittest.TestSuite()
    suite.addTest(AAVocabTest)

# def generate_mhcdomain_pssm_map(args):
#     logger.info('Start generate_mhcdomain_pssm_map...')
#     logger.debug('args.target_genes: %s' % args.target_genes)
#     logger.debug('args.input_msa_file: %s' % args.input_msa_file)
#     logger.debug('args.output_pkl: %s' % args.output_pkl)
#
#     pssm_map = {}
#     for mhcgene in args.target_genes:
#         logger.debug('Generate PSSM for %s' % mhcgene)
#         fn_msa = args.input_msa_file.format(mhcgene)
#         logger.debug('Loading MSA from %s' % fn_msa)
#         msa = MultipleSequenceAlignment.from_fasta(fn_fasta=fn_msa)
#         pssm = msa.pssm()
#         pssm_map[mhcgene] = pssm
#         logger.debug('Generated PSSM for %s, shape: %s' % (mhcgene, str(pssm.values.shape)))
#
#     logger.debug('Save %s PSSMs to %s' % (len(pssm_map), args.output_pkl))
#     logger.debug('Target keys: %s' % pssm_map.keys())
#     with open(args.output_pkl, 'wb') as f:
#         pickle.dump(pssm_map, f)
#
# def generate_mhcdomain_seqs(args):
#     logger.info('Start generate_mhcdomain_seqs...')
#     logger.debug('args.target_genes: %s' % args.target_genes)
#     logger.debug('args.output_format: %s' % args.output_format)
#     logger.debug('args.output_file: %s' % args.output_file)
#
#     bdomain = PanMHCIContactDomain()
#     for mhcgene in args.target_genes:
#         fn = args.output_file.format(mhcgene)
#         logger.debug('Saving %s to %s...' % (mhcgene, fn))
#         tokens = mhcgene.split('-')
#         alleles = MHCAlleleName.alleles(species=tokens[0], gene=tokens[1])
#         headers = []
#         seqs = []
#         for allele in alleles:
#             headers.append(allele)
#             seq = bdomain.domain_seq(allele)
#             seqs.append(seq)
#             logger.debug('Appended: allele:%s, seq: %s' % (allele, seq))
#         if args.output_format.lower() == 'fasta':
#             write_fa(fn, seqs, headers)
#             logger.debug('Saved: %s domain seqs for %s to %s' % (len(headers), mhcgene, fn))
#         else:
#             raise NotImplementedError('Unsupported format: %s' % args.output_format)

def generate_pepseq_pssm_map(args):
    logger.info('Start generate_pepseq_pssm...')
    logger.debug('args.source_dir: %s' % args.source_dir)
    logger.debug('args.target_peplens: %s' % str(args.target_peplens))
    logger.debug('args.output_pkl: %s' % args.output_pkl)

    pssm_map = {}

    for path in glob.glob('%s/*.txt' % args.source_dir):
        bn = basename(path, ext=False)
        tmp = bn.split('-')
        peplen = tmp[-1]
        allele = MHCAlleleName.std_name('-'.join(tmp[:-1]))
        logger.debug('file.bn: %s, peplen: %s, allele: %s' % (bn, peplen, allele))
        if MHCAlleleName.is_valid(allele):
            logger.debug('Loading PSSM for (%s, %s)' % (allele, peplen))
            df = pd.read_csv(path, sep='\t', skiprows=[0, 21], index_col=0, header=None)
            df = df.loc[list(AMINO_ACIDS), :]
            # If the sum of a column(position) is 0, fill with the same value(1/len(AMINO_ACIDS)
            # select_zero = np.flatnonzero(df.sum() == 0)
            # if len(select_zero) > 0:
            #     val = 1.0 / len(AMINO_ACIDS)
            #     logger.debug('The sum of a column is zero: %s, set to %s' % (select_zero, val))
            #     df.iloc[:, select_zero] = val

            logger.debug('Adding PSSM for (%s, %s)' % (allele, peplen))
            pssm_map[(allele, int(peplen))] = PositionSpecificScoringMatrix(values=df.values)
        else:
            logger.debug('Invalid allele name: %s, %s' % (bn, allele))

    # Extend and shrink PSSMs
    logger.debug('Further PSSMs by extending or shrinking 9-mer PSSMs')
    new_pssm_map = copy.deepcopy(pssm_map)
    for key, pssm in pssm_map.items():
        allele, peplen = key
        if peplen == 9:
            for new_peplen in range(args.target_peplens[0], args.target_peplens[1] + 1):
                if new_peplen != 9 and (allele, new_peplen) not in pssm_map:
                    new_pssm = PositionSpecificScoringMatrix(values=pssm.values.copy())
                    new_pssm.fit_length(new_peplen)
                    logger.debug('Adding PSSM for (%s, %s)' % (allele, new_peplen))
                    new_pssm_map[(allele, new_peplen)] = new_pssm

    logger.debug('Save %s PSSMs to %s' % (len(new_pssm_map), args.output_pkl))
    logger.debug('Target keys: %s' % new_pssm_map.keys())
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(new_pssm_map, f)
    #
    # with open(args.output_pkl, 'rb') as f:
    #     pssm_map = pickle.load(f)
    #     print(pssm_map)

def generate_pretrain_data(args):
    logger.info('Start generate_pretrain_data...')
    logger.debug('args.source_kim2014: %s' % args.source_kim2014)
    logger.debug('args.source_iedb: %s' % args.source_iedb)
    logger.debug('args.source_systemhc: %s' % args.source_systemhc)
    logger.debug('args.source_sarkizova2019: %s' % args.source_sarkizova2019)

    logger.debug('args.select_allele: %s' % args.select_allele)
    logger.debug('args.select_peplen: %s' % args.select_peplen)
    logger.debug('args.output_csv: %s' % args.output_csv)
    logger.debug('args.output_allelelist: %s' % args.output_allelelist)

    logger.info('Loading pretrain data...')
    select_patterns = {
        PeptideMHCBindDataFrameColumn.allele: args.select_allele,
        PeptideMHCBindDataFrameColumn.pep_len: args.select_peplen
    }
    df = load_df(fn_kim2014=args.source_kim2014,
               fn_iedb=args.source_iedb,
               fn_systemhc=args.source_systemhc,
               fn_sarkizova2019=args.source_sarkizova2019,
               select_patterns=select_patterns)

    logger.info('Done to load pretrain data.')
    df.to_csv(args.output_csv)
    logger.info('Saved pretrain data %s to %s.' % (str(df.shape), args.output_csv))

    alleles = sorted(np.unique(df[PeptideMHCBindDataFrameColumn.allele].values))
    with open(args.output_allelelist, 'w') as f:
        f.write(str(alleles))

    logger.info('Saved pretrain allele list %s to %s.' % (alleles, args.output_allelelist))

def generate_finetune_data(args):
    pass
    # logger.info('Start generate_finetune_csv...')
    # logger.debug('args.source_iedb: %s' % args.source_iedb)
    # logger.debug('args.output_csv: %s' % args.output_csv)
    #
    # logger.info('Loading finetune data...')
    # df = load_data_iedb(fn=args.source_iedb, select='MS')
    # logger.info('Done to load finetune data.')
    # df.to_csv(args.output_csv)
    # logger.info('Saved finetune data %s to %s.' % (str(df.shape), args.output_csv))

def pretrain(args):
    logger.info('Start pretrain...')
    logger.info('model_key: %s' % args.model_key)

    model_key = args.model_key
    train_conf = None
    bert_conf = None
    data_conf = None
    optim_conf = None


    with open('../config/pretrain.json', 'r') as f:
        train_conf = json.load(f)

    with open('../config/bert.json', 'r') as f:
        bert_conf = json.load(f)

    with open('../config/data.json', 'r') as f:
        data_conf = json.load(f)

    with open('../config/optim.json', 'r') as f:
        optim_conf = json.load(f)

    data_key = train_conf[model_key]['data_key']
    bert_key = train_conf[model_key]['bert_key']
    optim_key = train_conf[model_key]['optim_key']

    train_conf = train_conf[model_key]
    data_conf = data_conf[data_key]
    bert_conf = bert_conf[bert_key]
    optim_conf = optim_conf[optim_key]

    logger.debug('train_conf: %s' % train_conf)
    logger.debug('data_conf: %s' % data_conf)
    logger.debug('bert_conf: %s' % bert_conf)
    logger.debug('optim_conf: %s' % optim_conf)

    train_csv = data_conf['train_csv']
    df = pd.read_csv(train_csv, index_col=0)
    logger.debug('Loaded df.shape: %s train data from %s' % (str(df.shape), train_csv))
    logger.debug('df.head(): %s' % df.head())

    test_size = train_conf['test_size']

    train_df, test_df = train_test_split(df, test_size=test_size)

    logger.debug('test_size: %s, train_df.shape: %s, test_df.shape: %s' % (test_size,
                                                                           str(train_df.shape),
                                                                           str(test_df.shape)))
    css_list = data_conf['css_list']
    cdomain = PanMHCIContactDomain(css_list=css_list)
    aavocab = AAVocab.load_aavocab()

    max_pep_len = data_conf['max_pep_len']
    mask_ratio = data_conf['mask_ratio']
    pepseq_mask_probs = data_conf['pepseq_mask_probs']
    mhcseq_mask_probs = data_conf['mhcseq_mask_probs']

    n_epochs = train_conf['n_epochs']
    batch_size = train_conf['batch_size']
    lr = optim_conf['lr']
    warmup_steps = optim_conf['warmup_steps']
    chk_file = train_conf['chk_file']

    hidden_size = bert_conf['hidden_size']
    n_layers = bert_conf['n_layers']
    n_heads = bert_conf['n_heads']
    dropout = bert_conf['dropout']

    train_ds = MaskedPeptideMHCBindSentenceDataset(df=train_df,
                                                   cdomain=cdomain,
                                                   aavocab=aavocab,
                                                   max_pep_len=max_pep_len,
                                                   pepseq_mask_ratio=mask_ratio,
                                                   mhcseq_mask_ratio=mask_ratio,
                                                   pepseq_mask_probs=pepseq_mask_probs,
                                                   mhcseq_mask_probs=mhcseq_mask_probs)

    test_ds = MaskedPeptideMHCBindSentenceDataset(df=test_df,
                                                  cdomain=cdomain,
                                                  aavocab=aavocab,
                                                  max_pep_len=max_pep_len,
                                                  pepseq_mask_ratio=mask_ratio,
                                                  mhcseq_mask_ratio=mask_ratio,
                                                  pepseq_mask_probs=pepseq_mask_probs,
                                                  mhcseq_mask_probs=mhcseq_mask_probs)

    train_data_loader = DataLoader(train_ds, batch_size=batch_size)
    test_data_loader = DataLoader(test_ds, batch_size=batch_size)

    max_len = max_pep_len + len(cdomain.all_hla_sites) + 3

    bert = BERT(vocab_size=aavocab.size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=max_len,
                dropout=dropout)

    model = BERTLM(bert)
    trainer = ModelTrainer(model)

    optimizer = NoamOptimizer(model.parameters(),
                              d_model=hidden_size,
                              lr=lr,
                              warmup_steps=warmup_steps)
    criterion = BERTLMLoss()

    trainer.fit(train_data_loader=train_data_loader,
                test_data_loader=test_data_loader,
                optimizer=optimizer,
                criterion=criterion,
                n_epochs=n_epochs,
                use_cuda=torch.cuda.is_available())


def main():
    parser = ArgumentParser('KVACC')
    parser.add_argument('--debug_level', type=str, default='DEBUG')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'all_test'
    sub_parser = subparsers.add_parser('all_test')
    sub_parser.set_defaults(func=run_test_suite)

    # Arguments for sub command 'generate_pretrain_data'
    sub_parser = subparsers.add_parser('generate_pretrain_data')
    sub_parser.set_defaults(func=generate_pretrain_data)
    sub_parser.add_argument('--source_iedb', type=str, default='../data/mhc_ligand_full.csv')
    sub_parser.add_argument('--source_kim2014', type=str, default='../data/bdata.20130222.mhci.txt')
    sub_parser.add_argument('--source_systemhc', type=str, default='../data/systemhcatlas_180409/data.csv')
    sub_parser.add_argument('--source_sarkizova2019', type=str, default='../data/Sarkizova_NatBiotech2019/data_HLA-I_95.sample.csv')
    sub_parser.add_argument('--select_allele', type=str, default='^HLA-[ABC]\*[0-9]{2}:[0-9]{2}$')
    sub_parser.add_argument('--select_peplen', type=str, default='^([8-9]|1[0-5])$')
    sub_parser.add_argument('--output_csv', type=str, default='../output/pretrain_data_HLA-ABC_plen8-15.csv')
    sub_parser.add_argument('--output_allelelist', type=str, default='../output/pretrain_allelelist.txt')

    # Arguments for sub command 'generate_finetune_data'
    sub_parser = subparsers.add_parser('generate_finetune_data')
    sub_parser.set_defaults(func=generate_finetune_data)
    sub_parser.add_argument('--source_iedb', type=str, default='../data/mhc_ligand_full.csv')
    sub_parser.add_argument('--output_csv', type=str, default='../output/finetune_data.csv')
    sub_parser.add_argument('--output_allelelist', type=str, default='output/finetune_allelelist.txt')

    # Arguments for sub command 'generate_pepseq_pssm'
    sub_parser = subparsers.add_parser('generate_pepseq_pssm_map')
    sub_parser.set_defaults(func=generate_pepseq_pssm_map)
    sub_parser.add_argument('--source_dir', type=str, default='../data/pssm/smmpmbec_matrix')
    sub_parser.add_argument('--target_peplens', type=tuple, default=(8, 15))
    sub_parser.add_argument('--output_pkl', type=str, default='../data/pssm/smmpmbec_map.pkl')

    # Arguments for sub command 'generate_mhcdomain_seqs'
    # sub_parser = subparsers.add_parser('generate_mhcdomain_seqs')
    # sub_parser.set_defaults(func=generate_mhcdomain_seqs)
    # sub_parser.add_argument('--target_genes', type=list, default=['HLA-A', 'HLA-B', 'HLA-C'])
    # sub_parser.add_argument('--output_format', type=str, default='fasta')
    # sub_parser.add_argument('--output_file', type=str, default='../data/mhcinfo/{0}.domain.fa')
    #
    # # Arguments for sub command 'generate_mhcdomain_pssm_map'
    # sub_parser = subparsers.add_parser('generate_mhcdomain_pssm_map')
    # sub_parser.set_defaults(func=generate_mhcdomain_pssm_map)
    # sub_parser.add_argument('--target_genes', type=list, default=['HLA-A', 'HLA-B', 'HLA-C'])
    # sub_parser.add_argument('--input_msa_file', type=str, default='../data/mhcinfo/{0}.domain.aln.fa')
    # sub_parser.add_argument('--output_pkl', type=str, default='../data/mhcinfo/mhcpssm_map.pkl')

    # Arguments for sub command 'pretrain'
    sub_parser = subparsers.add_parser('pretrain')
    sub_parser.set_defaults(func=pretrain)
    sub_parser.add_argument('--model_key', type=str, default='pretrain_model.0')
    args = parser.parse_args()

    print('Logging level: %s' % args.debug_level)
    logger.setLevel(args.debug_level)

    args.func(args)

if __name__ == '__main__':
    main()
