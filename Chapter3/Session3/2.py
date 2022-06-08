import sys
sys.path.append('../..')
from common.util import preprocess,create_contexts_target,convert_one_hot

text = 'You say good bye and I say hello.'
corpus,word_to_id,id_to_word = preprocess(text)

contexts,target = create_contexts_target(corpus)

vocab_size= len(word_to_id)
target= convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)

print(target)
print(contexts)

