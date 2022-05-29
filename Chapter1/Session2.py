import numpy as np

input = np.random.rand(10,2)
W1 = np.random.rand(2,4)
b1 = np.random.rand(1,4)
W2 = np.random.rand(4,3)
b2 = np.random.rand(1,3)

h = np.dot(input,W1)+b1
a = sigmoid(h)
s = np.dot(a,W2)+b2

print(softmax(s))



