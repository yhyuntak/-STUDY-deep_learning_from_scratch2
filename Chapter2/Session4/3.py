import sys
sys.path.append('../..')
from common.np import *
import matplotlib.pyplot as plt
from common.util import *

text = 'YOu say goodbye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(corpus)
C = create_co_matrix(corpus,vocab_size)
W=ppmi(C)

U,S,V=np.linalg.svd(W)

print(C[0])
print(W[0])
print(U[0])
print(S)

for word,word_id in word_to_id.items():
    plt.annotate(word,(U[word_id,0],U[word_id,1]))

plt.scatter(U[:,0],U[:,1],alpha=0.5)
plt.show()