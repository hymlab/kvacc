import unittest
import numpy as np
import re
from enum import Enum, IntEnum, auto
from urllib import request
import ssl
import os
import warnings
import logging.config


# Logger
logger = logging.getLogger('kvacc')

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_PAIRS = [(AMINO_ACIDS[i], AMINO_ACIDS[j]) for i in range(len(AMINO_ACIDS)) for j in range(i, len(AMINO_ACIDS))]
AA_INDEX = list(AMINO_ACIDS)

# Common constants
EPS = 0.000001


def basename(path, ext=True):
    bn = os.path.basename(path)
    if not ext:
        bn = os.path.splitext(bn)[0]
    return bn


# Ref: https://github.com/irgeek/StrEnum
class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    # pylint: disable=no-self-argument
    # The first argument to this function is documented to be the name of the
    # enum member, not `self`:
    # https://docs.python.org/3.6/library/enum.html#using-automatic-values
    def _generate_next_value_(name, *_):
        return name

class BindLevel(IntEnum):
    POSITIVE_HIGH = 4
    POSITIVE = 3
    POSITIVE_INTERMEDIATE = 2
    POSITIVE_LOW = 1
    NEGATIVE = 0

    @classmethod
    def is_binder(cls, level):
        return level > BindLevel.NEGATIVE

    @classmethod
    def bind_levels(cls):
        return list(cls)

class StrUtils(object):
    @staticmethod
    def rm_nonwords(s):
        import re
        return re.sub('\\W', '', s)

    @staticmethod
    def empty(s):
        return (s is None) or (len(s) == 0)

    @staticmethod
    def default_str(s, ds):
        return ds if s is None else s

class RemoteUtils(object):
    _ssl_context = ssl._create_unverified_context()

    @classmethod
    def download_to(cls, url, decode='utf-8', fnout=None):
        with request.urlopen(url, context=cls._ssl_context) as response, open(fnout, 'w') as fout:
            fout.write(response.read().decode(decode))

### Tests
class BaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore')
        logging.config.fileConfig('../config/logging.conf')

        cls.test_synonyms = [
            'BoLA-AW10', 'BoLA-D18.4', 'BoLA-HD6', 'BoLA-JSP.1', 'BoLA-T2C', 'BoLA-T2a',
            'BoLA-T2b', 'ELA-A1', 'Gogo-B*0101', 'H-2-Db', 'H-2-Dd', 'H-2-Kb', 'H-2-Kbm8', 'H-2-Kwm7',
            'H-2-Kd', 'H-2-Kk', 'H-2-Ld', 'H-2-Lq', 'HLA-Cw1', 'HLA-Cw4', 'HLA-E*01:01',
            'HLA-E*01:03', 'Mamu-A*01', 'Mamu-A01', 'Mamu-A*02', 'Mamu-A02', 'Mamu-A*07', 'Mamu-A07',
            'Mamu-A*11', 'Mamu-A11', 'Mamu-A*2201', 'Mamu-A2201',
            'Mamu-A*2601', 'Mamu-A2*0102', 'Mamu-A7*0103', 'Mamu-B*01', 'Mamu-B*03', 'Mamu-B*04',
            'Mamu-B*08', 'Mamu-B*1001', 'Mamu-B*17', 'Mamu-B*3901', 'Mamu-B*52', 'Mamu-B*6601',
            'Mamu-B*8301', 'Mamu-B*8701', 'Patr-A*0101', 'Patr-A*0301', 'Patr-A*0401',
            'Patr-A*0602', 'Patr-A*0701', 'Patr-A*0901', 'Patr-B*0101', 'Patr-B*0901',
            'Patr-B*1301', 'Patr-B*1701', 'Patr-B*2401', 'RT1A', 'SLA-1*0401',
            'SLA-1*0701', 'SLA-2*0401', 'SLA-3*0401'
        ]
        cls.target_classI_alleles = [
            'BoLA-1*023:01', 'BoLA-2*012:01', 'BoLA-3*001:01', 'BoLA-3*002:01',
            'BoLA-6*013:01', 'BoLA-6*041:01', 'BoLA-T2c', 'H2-Db', 'H2-Dd',
            'H2-Kb', 'H2-Kd', 'H2-Kk', 'H2-Ld', 'HLA-A*01:01', 'HLA-A*02:01',
            'HLA-A*02:02', 'HLA-A*02:03', 'HLA-A*02:05', 'HLA-A*02:06',
            'HLA-A*02:07', 'HLA-A*02:11', 'HLA-A*02:12', 'HLA-A*02:16',
            'HLA-A*02:17', 'HLA-A*02:19', 'HLA-A*02:50', 'HLA-A*03:01',
            'HLA-A*03:19', 'HLA-A*11:01', 'HLA-A*23:01', 'HLA-A*24:02',
            'HLA-A*24:03', 'HLA-A*25:01', 'HLA-A*26:01', 'HLA-A*26:02',
            'HLA-A*26:03', 'HLA-A*29:02', 'HLA-A*30:01', 'HLA-A*30:02',
            'HLA-A*31:01', 'HLA-A*32:01', 'HLA-A*32:07', 'HLA-A*32:15',
            'HLA-A*33:01', 'HLA-A*66:01', 'HLA-A*68:01', 'HLA-A*68:02',
            'HLA-A*68:23', 'HLA-A*69:01', 'HLA-A*80:01', 'HLA-B*07:02',
            'HLA-B*08:01', 'HLA-B*08:02', 'HLA-B*08:03', 'HLA-B*14:01',
            'HLA-B*14:02', 'HLA-B*15:01', 'HLA-B*15:02', 'HLA-B*15:03',
            'HLA-B*15:09', 'HLA-B*15:17', 'HLA-B*18:01', 'HLA-B*27:05',
            'HLA-B*27:20', 'HLA-B*35:01', 'HLA-B*35:03', 'HLA-B*37:01',
            'HLA-B*38:01', 'HLA-B*39:01', 'HLA-B*40:01', 'HLA-B*40:02',
            'HLA-B*40:13', 'HLA-B*42:01', 'HLA-B*44:02', 'HLA-B*44:03',
            'HLA-B*45:01', 'HLA-B*46:01', 'HLA-B*48:01', 'HLA-B*51:01',
            'HLA-B*53:01', 'HLA-B*54:01', 'HLA-B*57:01', 'HLA-B*57:03',
            'HLA-B*58:01', 'HLA-B*58:02', 'HLA-B*73:01', 'HLA-B*81:01',
            'HLA-B*83:01', 'HLA-C*03:03', 'HLA-C*04:01', 'HLA-C*05:01',
            'HLA-C*06:02', 'HLA-C*07:01', 'HLA-C*07:02', 'HLA-C*08:02',
            'HLA-C*12:03', 'HLA-C*14:02', 'HLA-C*15:02', 'HLA-E*01:01',
            'HLA-E*01:03', 'Mamu-A1*001:01', 'Mamu-A1*002:01', 'Mamu-A1*007:01',
            'Mamu-A1*011:01', 'Mamu-A1*022:01', 'Mamu-A1*026:01',
            'Mamu-A2*01:02', 'Mamu-A7*01:03', 'Mamu-B*001:01', 'Mamu-B*003:01',
            'Mamu-B*008:01', 'Mamu-B*010:01', 'Mamu-B*017:01', 'Mamu-B*039:01',
            'Mamu-B*052:01', 'Mamu-B*066:01', 'Mamu-B*084:01', 'Mamu-B*087:01',
            'Patr-A*01:01', 'Patr-A*03:01', 'Patr-A*04:01', 'Patr-A*07:01',
            'Patr-A*09:01', 'Patr-B*01:01', 'Patr-B*13:01', 'Patr-B*24:01',
            'Rano-A1*b', 'SLA-1*04:01', 'SLA-1*07:01', 'SLA-2*04:01','SLA-3*04:01']

        with open('../output/pretrain_allelelist.txt', 'r') as f:
            cls.pretrain_alleles = eval(f.read())

        for allele in cls.pretrain_alleles:
            if allele not in cls.target_classI_alleles:
                cls.target_classI_alleles.append(allele)

    def assertArrayEqual(self, first, second):
        self.assertTrue(np.array_equal(first, second))

class BindLevelTest(BaseTest):
    def test_is_binder(self):
        self.assertTrue(BindLevel.is_binder(BindLevel.POSITIVE_HIGH))
        self.assertTrue(BindLevel.is_binder(BindLevel.POSITIVE))
        self.assertTrue(BindLevel.is_binder(BindLevel.POSITIVE_INTERMEDIATE))
        self.assertTrue(BindLevel.is_binder(BindLevel.POSITIVE_LOW))
        self.assertFalse(BindLevel.is_binder(BindLevel.NEGATIVE))

    def test_bind_levels(self):
        expected_levels = [
            BindLevel.POSITIVE_HIGH,
            BindLevel.POSITIVE,
            BindLevel.POSITIVE_INTERMEDIATE,
            BindLevel.POSITIVE_LOW,
            BindLevel.NEGATIVE
        ]
        self.assertListEqual(expected_levels, BindLevel.bind_levels())

    def test_bind_level_is_integer(self):
        self.assertEqual(4, BindLevel.POSITIVE_HIGH)
        self.assertEqual(3, BindLevel.POSITIVE)
        self.assertEqual(2, BindLevel.POSITIVE_INTERMEDIATE)
        self.assertEqual(1, BindLevel.POSITIVE_LOW)
        self.assertEqual(0, BindLevel.NEGATIVE)

class StrUtilsTest(BaseTest):
    def test_empty(self):
        self.assertTrue(StrUtils.empty(None))
        self.assertTrue(StrUtils.empty(''))
        self.assertFalse(StrUtils.empty(' '))

    def test_default_str(self):
        self.assertEqual('Kim', StrUtils.default_str('Kim', 'Anon'))
        self.assertEqual('Anon', StrUtils.default_str(None, 'Anon'))
        self.assertIsNone(None, StrUtils.default_str(None, None))

# class SeqUtilsTest(BaseTest):
#     def setUp(self):
#         np.random.seed(76923)
#         self.pssm_9 = np.random.sample((len(AMINO_ACIDS), 9))
#         self.blosum = np.random.sample((len(AMINO_ACIDS), len(AMINO_ACIDS)))
#         self.blosum = (self.blosum + self.blosum.T)/2 # make symetric matrix
#         np.fill_diagonal(self.blosum, 1)
#
#     def test_format_fa(self):
#         seqs = ['AAA', 'BBB', 'CCC']
#         headers = ['HA', 'HB', 'HC']
#         expected_with_headers = '>HA\nAAA\n>HB\nBBB\n>HC\nCCC'
#         expected_without_headers = '>1\nAAA\n>2\nBBB\n>3\nCCC'
#         self.assertEqual(expected_with_headers, SeqUtils.format_fa(seqs=seqs, headers=headers))
#         self.assertEqual(expected_without_headers, SeqUtils.format_fa(seqs=seqs))
#
#     def test_is_valid_aaseq(self):
#         self.assertTrue(SeqUtils.is_valid_aaseq('ACDEFRY'))
#         self.assertFalse(SeqUtils.is_valid_aaseq('BQWPSMKYY'))
#         self.assertFalse(SeqUtils.is_valid_aaseq('AQWPSMKYY-'))
#         self.assertFalse(SeqUtils.is_valid_aaseq('ACDEFRY' + GAP))
#         self.assertTrue(SeqUtils.is_valid_aaseq('ACDEFRY' + GAP, allow_gap=True))
#
#     def test_rand_seqs(self):
#         seq = SeqUtils.rand_aaseq(seq_len=15)
#         self.assertTrue(len(seq) == 15)
#         self.assertTrue(SeqUtils.is_valid_aaseq(seq))
#
#     def test_rand_subst_by_pssm_same_len(self):
#         seq1 = 'KSDGSFIGY'
#         positions = np.array([1, 8])
#         seq2 = SeqUtils.rand_subst_by_pssm(seq1, target_positions=positions, pssm=self.pssm_9)
#         print('seq1:%s, seq2:%s' % (seq1, seq2))
#         for i in range(len(seq1)):
#             if i not in positions:
#                 self.assertEqual(seq1[i], seq2[i])
#             else:
#                 self.assertNotEqual(seq1[i], seq2[i])
#
#     def test_rand_subst_by_pssm_longer_len(self):
#         seq1 = 'KSDGSFIGYM'
#         positions = np.array([1, 9])
#         seq2 = SeqUtils.rand_subst_by_pssm(seq1, target_positions=positions, pssm=self.pssm_9)
#         print('seq1:%s, seq2:%s' % (seq1, seq2))
#         for i in range(len(seq1)):
#             if i not in positions:
#                 self.assertEqual(seq1[i], seq2[i])
#             else:
#                 self.assertNotEqual(seq1[i], seq2[i])
#
#         seq1 = 'KSDGSFIGYML'
#         positions = np.array([9, 10])
#         seq2 = SeqUtils.rand_subst_by_pssm(seq1, target_positions=positions, pssm=self.pssm_9)
#         print('seq1:%s, seq2:%s' % (seq1, seq2))
#         for i in range(len(seq1)):
#             if i not in positions:
#                 self.assertEqual(seq1[i], seq2[i])
#             else:
#                 self.assertNotEqual(seq1[i], seq2[i])
#
#     def test_rand_subst_by_pssm_shorter_len(self):
#         seq1 = 'KSDGSFI'
#         positions = np.array([1, 4])
#         seq2 = SeqUtils.rand_subst_by_pssm(seq1, target_positions=positions, pssm=self.pssm_9)
#         print('seq1:%s, seq2:%s' % (seq1, seq2))
#         for i in range(len(seq1)):
#             if i not in positions:
#                 self.assertEqual(seq1[i], seq2[i])
#             else:
#                 self.assertNotEqual(seq1[i], seq2[i])
#
#     def test_rand_subst_by_blosum(self):
#         seq1 = 'KSDGSFIGYM'
#         positions = np.array([1, 3, 9])
#         seq2 = SeqUtils.rand_subst_by_blosum(seq1, target_positions=positions, blosum=self.blosum)
#         print('seq1:%s, seq2:%s' % (seq1, seq2))
#         for i in range(len(seq1)):
#             if i not in positions:
#                 self.assertEqual(seq1[i], seq2[i])
#             else:
#                 self.assertNotEqual(seq1[i], seq2[i])

class RemoteUtilsTest(BaseTest):
    def test_download_to(self):
        fn_test = '../tmp/test.fa'
        if os.path.exists(fn_test):
            os.unlink(fn_test)

        url = 'http://www.google.com'

        RemoteUtils.download_to(url, fnout=fn_test)

        self.assertTrue(os.path.exists(fn_test))
        self.assertTrue(os.path.getsize(fn_test) > 0)
        os.unlink(fn_test)

class HttpMethod(StrEnum):
    GET = auto()
    HEAD = auto
    POST = auto()
    PUT = auto()
    DELETE = auto()
    CONNECT = auto()
    OPTIONS = auto()
    TRACE = auto()
    PATCH = auto()

class StrEnumTest(BaseTest):

    def test_isinstance_str(self):
        self.assertTrue(isinstance(HttpMethod.GET, str))

    def test_value_isinstance_str(self):
        self.assertTrue(isinstance(HttpMethod.GET.value, str))

    def test_str_builtin(self):
        self.assertTrue(str(HttpMethod.GET) == "GET")
        self.assertTrue(HttpMethod.GET == "GET")

if __name__ == '__main__':
    unittest.main()
