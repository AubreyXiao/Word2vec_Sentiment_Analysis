from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy

modelname = "300features_40minwords_10window"

model = Word2Vec.load(modelname)

vocab = set(model.wv.index2word)

words = list(model.wv.vocab)
