from argparse import ArgumentParser
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

parser = ArgumentParser()
parser.add_argument(help="dataset name", dest="glove_path")
params = parser.parse_args()


# glove_file = datapath(params.glove_path)
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(params.glove_path, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
model.save_word2vec_format(params.glove_path+"_word2vec.bin", binary=True)