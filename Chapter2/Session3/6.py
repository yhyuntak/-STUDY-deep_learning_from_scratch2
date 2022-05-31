import sys
sys.path.append('../..')
from common.np import *
from common.util import *


text = 'YOu say goodbye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(corpus)
C = create_co_matrix(corpus,vocab_size)

print(C)
print(np.sum(C,axis=0))
print(np.sum(C))
print(C.shape)
print(most_similar('you',word_to_id,id_to_word,C))