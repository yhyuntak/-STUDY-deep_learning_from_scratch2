import numpy as np

print(np.random.choice(10))

words = ['you','say','goodbye','I','hello','.']
print(np.random.choice(words))

print(np.random.choice(words,size=5))

print(np.random.choice(words,size=5,replace=False))

p = [0.5,0.1,0.05,0.2,0.05,0.1]
print(np.random.choice(words,p=p))
print(np.random.choice(words,p=p,size=3))




