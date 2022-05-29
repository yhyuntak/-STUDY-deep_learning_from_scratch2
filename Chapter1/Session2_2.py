import numpy as np
from common.layers import *

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O = input_size,hidden_size,output_size

        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layer = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]

        self.params = []
        for layer in self.layer:
            self.params += layer.params

    def predict(self,x):
        for layer in self.layer:
            x = layer.forward(x)
        return x

x = np.random.randn(10,2)
model = TwoLayerNet(2,4,3)
s = model.predict(x)

print(s)
