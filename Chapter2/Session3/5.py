import sys
sys.path.append('../..')
from common.util import *

text = 'YOu say goodbye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(corpus)
C = create_co_matrix(corpus,vocab_size)

c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]

print(cos_similarity(c0,c1))