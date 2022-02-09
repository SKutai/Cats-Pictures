import numpy as np
from losses import *
from layers import *

class NeuralNet:
    """
    A class for learning with fully connected neural networks

    Parameters
    ----------
    d: int
        Dimensions of the input
    est_lossderiv: function ndarray(N) -> ndarray(N)
        Gradient of the loss function with respect to the inputs
        to the last layer, using the output of the last layer
    """
    def __init__(self, d, est_lossderiv):
        # Parameters for the network
        self.d = d
        self.Ws = []
        self.bs = []
        self.fs = []
        self.est_lossderiv = est_lossderiv

        # Temporary variables to store outputs of each layer
        self.h = []
        self.a = []
    
    def add_layer(self, m, f, fderiv):
        """
        Parameters
        ----------
        m: int
            Number of nodes in the layer
        f: function ndarray(N) -> ndarray(N)
            Activation function, which is applied element-wise
        fderiv: function ndarray(N) -> ndarray(N)
            Derivative of activation function, which is applied element-wise
        """
        n = self.d
        if len(self.Ws) > 0:
            n = self.Ws[-1].shape[0]
        self.Ws.append(np.random.randn(m, n))
        self.bs.append(np.random.randn(m))
        self.fs.append({"f":f, "fderiv":fderiv})

    
    def forward(self, x):
        """
        Do a forward pass on the network
        
        Parameters
        ----------
        x: ndarray(d)
            Input to feed through
        
        Returns
        -------
        ndarray(m)
            Output of the network
        """
        self.h = [x]
        self.a = [None]
        for k in range(len(self.Ws)):
            a = self.bs[k] + self.Ws[k].dot(self.h[k])
            h = self.fs[k]["f"](a)
            self.a.append(a)
            self.h.append(h)
        return self.h[-1].flatten()
    
    def backprop_descent(self, x, y, alpha=0.01):
        """
        Do stochastic gradient backpropagation to compute the 
        gradient of all parameters on a single example, then 
        take a step against that direction by the learning rate
        
        Parameters
        ----------
        x: ndarray(d)
            Input to feed through
        y: float or ndarray(k)
            Goal output.  Dimensionality should match dimensionality
            of the last output layer
        alpha: float
            Learning rate
        """
        self.forward(x) # Always do the forward step first to compute the a's and h's
        ## TODO: Fill this in to complete backpropagation and weight updates

        b_derivs = []
        W_derivs = []

        # number of layers
        L = len(self.fs)

        # result of last layer
        y_est = self.h[-1]

        # initial gradient
        g = self.est_lossderiv(y_est, y)

        for k in range(L-1, -1, -1):
            # step 1
            if k < L-1:
                g *= self.fs[k]['fderiv'](self.a[k+1]) # k+1 aligns self.a since it starts with None
            
            # step 2
            b_derivs = [g] + b_derivs
            g2D = np.reshape(g, (g.shape[0], 1))
            hT = np.reshape(self.h[k], (1, self.h[k].shape[0]))
            W_derivs = [np.matmul(g2D, hT)] + W_derivs

            # step 3
            g = np.matmul(self.Ws[k].T, g)

        # subtract learning factor alpha``
        for l in range(L):
            self.bs[l] -= alpha * b_derivs[l]
            self.Ws[l] -= alpha * W_derivs[l]

