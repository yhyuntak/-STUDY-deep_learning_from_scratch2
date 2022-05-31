import sys
sys.path.append('../..')
from common import config
config.GPU = True
from common.np import *
import matplotlib.pyplot as plt
from common.util import *
from dataset import ptb

window_size =2
wordvec_size=100

corpus,word_to_id,id_to_word = ptb.load_data('train')
vocab_size = len(corpus)

print('<<동시발생 수 계산>>')
C = create_co_matrix(corpus,vocab_size)

print('<<PPMI 계산>>')
W=ppmi(C,verbose=True)

print('<<SVD 계산>>')
try :
    # truncated SVD
    from sklearn.utils.extmath import randomized_svd
    U,S,V, = randomized_svd(W,n_components=wordvec_size,n_iter=5,random_state=None)

except ImportError:
    # SVD
    U,S,V, = np.linalg.svd(W)

word_vecs = U[:,:wordvec_size]
querys = ['you','year','car','toyota']
for query in querys:
    most_similar(query,word_to_id,id_to_word,word_vecs,top=5)
