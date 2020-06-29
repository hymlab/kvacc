import unittest
import logging.config
import os

from kvacc.mhcnc import MHCAlleleName
from kvacc.commons import BaseTest, RemoteUtils
from kvacc.bioseq import FastaSeqParser

# Logger
logger = logging.getLogger('kvacc')

# MHC sequence registry dervied from IMGT/HLA database{Robinson:2014ca}
class MHCAlleleSeqRegistry(object):
    MHC_PROT_FASTA_PATH = '../data/mhcinfo/prot'

    class FastaSeqLoader(FastaSeqParser.Listener):
        '''
        Load fasta sequences into MHCSeqRegistry object, listening a on_seq_read event from FastaSeqParser
        Arguments:
            :argument seq_registry: the MHCSeqRegistry(outer class) object
        '''
        def __init__(self, seq_registry):
            self._seq_registry = seq_registry

        def on_seq_read(self, header=None, seq=None):
            # logger.debug('on_seq_read: header:%s, seq:%s' % (header, seq))
            allele_name = self._get_allele_name(header)
            self._seq_registry._set_allele_seq(allele_name, seq)

        def _get_allele_name(self, header):
            allele_name = None
            if header.startswith('HLA'):
                tokens = header.split()
                allele_name = 'HLA-' + tokens[1].strip()
            else:
                tokens = header.split(',')
                allele_name = tokens[1].strip()

            allele_name = MHCAlleleName.std_name(allele_name)
            return MHCAlleleName.parse(allele_name).format()

    # Class variable
    _allele_seq_map = {}

    SPECIES_FASTA_BASEURL_MAP ={
        'Patr': 'https://www.ebi.ac.uk/ipd/mhc/group/NHP/download/Patr',
        'Mamu': 'https://www.ebi.ac.uk/ipd/mhc/group/NHP/download/Mamu',
        'Gogo': 'https://www.ebi.ac.uk/ipd/mhc/group/NHP/download/Mamu',
        'Eqca': 'https://www.ebi.ac.uk/ipd/mhc/group/ELA/download/Eqca',
        'SLA': 'https://www.ebi.ac.uk/ipd/mhc/group/SLA/download/Susc',
        'BoLA': 'https://www.ebi.ac.uk/ipd/mhc/group/BoLA/download/BoLA',
        'Rano': 'https://www.ebi.ac.uk/ipd/mhc/group/RT1/download/Rano'
    }

    def protein_seq(self, name):
        sname = MHCAlleleName.std_name(name)
        if sname not in self._allele_seq_map:
            allele = MHCAlleleName.parse(sname)
            self._load_protein_fasta_by_gene(species=allele.species, gene=allele.gene)
        seq = self._allele_seq_map[sname]
        return seq

    def _load_protein_fasta_by_gene(self, species, gene):
        fn = self.MHC_PROT_FASTA_PATH + '/%s/%s.fa' % (species, gene)
        if not os.path.exists(fn):
            url = self._get_fasta_from_url(species, gene)
            if url is None:
                raise ValueError('Undefined URL for %s, %s' % (species, gene))
            logger.debug('Downloading the fasta from %s' % url)

            # Download and save as the file
            RemoteUtils.download_to(url, decode='utf-8', fnout=fn)

        # Parse fasta file
        parser = FastaSeqParser()
        parser.add_parse_listener(self.FastaSeqLoader(self))

        with open(fn, 'r') as fin:
            logger.debug('Parsing fasta sequences from file %s' % fn)
            parser.parse(fin)

    def _set_allele_seq(self, allele, seq):
        # logger.debug('_set_allele_seq: allele:%s, seq:%s' % (allele, seq))
        if allele not in self._allele_seq_map:
            self._allele_seq_map[allele] = seq

    def _get_fasta_from_url(self, species, gene):
        url = None
        if species in self.SPECIES_FASTA_BASEURL_MAP:
            baseurl = self.SPECIES_FASTA_BASEURL_MAP[species]
            url = '%s/%s?type=protein' % (baseurl, gene)
        return url

class MHCSeqRegistryTest(BaseTest):

    def test_protein_seq(self):
        seq_registry = MHCAlleleSeqRegistry()
        for i, allele in enumerate(self.target_classI_alleles):
            # std_name = MHCAlleleName.std_name(old_name)
            seq = seq_registry.protein_seq(allele)
            self.assertIsNotNone(seq)
            self.assertTrue(SeqUtils.is_valid_aaseq(seq, allow_gap=True))

        # alleles = seq_registry._allele_seq_map.keys()
        # tmp = list(filter(lambda s: s.startswith('HLA-A'), alleles))
        # print(tmp)
        # tmp = list(filter(lambda s: s.startswith('HLA-B'), alleles))
        # print(tmp)
        # tmp = list(filter(lambda s: s.startswith('HLA-C'), alleles))
        # print(tmp)

if __name__ == '__main__':
    unittest.main()
