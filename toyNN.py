import numpy as np

__author__ = 'collected'

class synapse:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def training(self,epochs,verbose=False):
        X = self.X
        y =self.y
        nonlin = lambda x : 1/(1+np.exp(-x))
        deriv = lambda  x : x*(1-x)

        np.random.seed(10)
        w0 = 2*np.random.random((3,1))-1

        for iter in xrange(epochs):
            l0 = X
            l1 = nonlin(np.dot(l0,w0))
            #measure error
            l1_error = y-l1
            if(verbose): print l1_error[0][0]
            l1_delta = l1_error * deriv(l1)
            #update weight
            w0 +=np.dot(l0.T,l1_delta)
        self.l1 = l1
        self.w0 = w0

    def predict(self,X):
        nonlin = lambda x : 1/(1+np.exp(-x))
        w0 = self.w0
        return nonlin(np.dot(X,w0))



if __name__=="__main__":
    X = np.array([  [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1] ])
    y = np.array([[1,1,0,0]]).T
    syn = synapse(X,y)
    syn.training(1000,True)
    print syn.predict(X)

