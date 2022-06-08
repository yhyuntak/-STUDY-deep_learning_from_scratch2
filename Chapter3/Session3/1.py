import sys
sys.path.append('../..')
from common.util import preprocess,create_contexts_target

text = 'You say good bye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)

contexts,target = create_contexts_target(corpus)
print(contexts)
print(target)

