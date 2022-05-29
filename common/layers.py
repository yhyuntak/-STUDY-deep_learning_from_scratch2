import numpy as np

from common.functions import *

class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self,x):
        return 1/(1+np.exp(-x))
class Affine :
    def __init__(self,W,b):
        self.params = [W,b]
    def forward(self,x):
        W,b = self.params
        out = np.dot(x,W)+b
        return out

class SoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads = [],[]
        self.y = None
        self.t = None
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size :
            self.t = self.t.argmax(axis=1)
        loss = cross_entropy_error(self.y,self.t)