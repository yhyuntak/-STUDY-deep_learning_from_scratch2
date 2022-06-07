from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine :
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None

    def forward(self,x):
        W,b = self.params
        out = np.dot(x,W)+b
        self.x = x
        return out
    def backward(self,dout):
        W,b = self.params
        dx = np.matmul(dout,W.T)
        dW = np.matmul(self.x.T,dout)
        db = np.sum(dout,axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

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

        return loss
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size),self.t] -=1
        dx *= dout
        dx = dx/batch_size

        return dx

class MatMul:
    def __init__(self,W):
        self.params=[W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self,x):
        W, = self.params
        self.x = x
        out = np.matmul(x,W)
        return out
    def backward(self,dout):
        W, = self.params
        dx = np.matmul(dout,W.T)
        dW = np.matmul(self.x.T,dout)
        self.grads[0][...]=dW
        return dx

class Embedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self,idx):
        W, = self.params
        self.idx = idx
        out = W[idx.get()]
        return out

    def backward(self,dout):
        dW, = self.grads
        dW[...] = 0

        if GPU:
            import cupyx
            cupyx.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)

        #
        # for i ,word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]
        return None

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        x = np.asarray(x)
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx