import sys
sys.path.append('../..')
from common.np import *
from common.util import preprocess,create_co_matrix

text = 'YOu say goodbye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)

print(corpus)

print(id_to_word)

C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0]
],dtype=np.int32)

print(C)
print(create_co_matrix(corpus,7,1))
