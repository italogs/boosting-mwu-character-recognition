from __future__ import print_function
import random
import numpy as np
import time
import sys
import pandas as pd
import keras
from sklearn.model_selection import train_test_split

number_a = 4
number_b = 6


def predict(x, m, w=None, voting=True, tval=number_a, fval=number_b):
    if w is None:
        w = np.ones_like(m)
    y = x[:,...] * m * w
    if voting:
        y = np.array([ tval if np.sum( y[i,...] ) >= 0 else fval for i in range(y.shape[0]) ])
    else:
        tidx = y>=0
        y[...] = fval
        y[tidx] = tval
        
    return y
      
def accuracy(y_pred, y_true):
    if ( len(y_pred.shape) == 3 ):
        acc = np.zeros((y_pred.shape[1],y_pred.shape[2]))
        for i in range(acc.shape[0]):
            for j in range(acc.shape[1]):
                acc[i,j] = np.sum(y_pred[:,i,j]==y_true).astype(float)
    else:
        acc = np.sum(y_pred==y_true).astype(float)

    acc = acc / y_true.shape[0]
    return acc

def loadDataset():
    # download from an online repository
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    return X_train, y_train, X_test, y_test


class single_clf:
    def __init__(self, x, y, p=None):
        self.idx = None
        self.sign = 0
        self.train(x,y,p)
    
    def predict_unit(self, x, sign, i, j, posval=number_a, negval=number_b):
        fval =  negval if sign >= 0 else  posval
        tval =  posval if sign  > 0 else  negval
        y = np.ones(x.shape[0])*fval
        y[ x[:,i,j] > 0 ] =  tval
        return y
    
    def train(self, x, y, p=None):
        if p is None:
            p = np.ones(y.shape[0])
        if self.idx is not None:
            hitA = np.sum((self.predict_unit(x, 1,self.idx[0],self.idx[1])==y).astype(float) * p)
            hitB = np.sum((self.predict_unit(x,-1,self.idx[0],self.idx[1])==y).astype(float) * p)
            self.sign = 1 if (hitA>=hitB) else -1
        else:
            besthit=0
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    hitA = np.sum((self.predict_unit(x, 1,i,j)==y).astype(float) * p)
                    hitB = np.sum((self.predict_unit(x,-1,i,j)==y).astype(float) * p)
                    if(hitA > besthit):
                        besthit = hitA
                        self.sign = 1
                        self.idx =(i,j)
                    if(hitB > besthit):
                        besthit = hitB
                        self.sign = -1
                        self.idx =(i,j)
            
    def predict(self, x, posval=1, negval=-1):
        fval =  negval if self.sign >= 0 else  posval
        tval =  posval if self.sign  > 0 else  negval
        y = np.ones(x.shape[0])*fval
        y[ x[:,self.idx[0],self.idx[1]] > 0 ] =  tval
        return y

class MWU:
    def __init__(self, gamma):
        self.gamma = gamma
    
    def train(self, train, test, T=100, w=None):

        x_train, y_train = train
        x_test, y_test = test
        x_train = (x_train>0).astype(float)

        self.learners = []
        self.t_hist = []
        self.test_accuracy = []     
        self.train_accuracy = []
        
        eps = np.sqrt( np.log(x_train.size) / T )
        P = np.ones(x_train.shape[0]) / x_train.shape[0]
        self.w = np.zeros( (x_train.shape[1],x_train.shape[2]) )
        for it in range(T):
            ci = single_clf(x_train,y_train,p=P)
            
            y_p = ci.predict(x_train, posval=number_a, negval=number_b)
            acc = np.sum((y_p==y_train).astype(float)*P)
            if acc < 0.5 + self.gamma:
                print ("There is no more {}-weak-learners".format(0.5 + self.gamma))
                break
                
            self.w[ci.idx[0],ci.idx[1]] += 1    
            self.learners.append(ci)
            miss = (y_p!=y_train)
            P[miss] *= np.exp(eps) 
            P = P/np.sum(P)
            
            ############# history log....############
            y_p = self.predict(x_test)
            v_acc = accuracy(y_p,y_test)
            y_p = self.predict(x_train)
            t_acc = accuracy(y_p,y_train)
            
            self.test_accuracy.append(v_acc)
            self.train_accuracy.append(t_acc)
            self.t_hist.append(it)
            ##########################################
            

            print("iteration {}: Validation accuracy: {}".format(it, v_acc))    
                
        print("{} : Final validation accuracy: {}".format(it,v_acc))   
        return P
        
    def predict(self, x, posval=number_a, negval=number_b):
        y = np.zeros(x.shape[0])
        for e in self.learners:
            y += e.predict(x)
        pos = (y>0)
        y[pos] = posval
        y[~pos] = negval
        return y
  
if __name__ == "__main__":
    
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = loadDataset()
    print("Filtering data based on a ={} b = {}".format(number_a,number_b))
    df_train = pd.DataFrame( data={'y' : y_train } )
    df_train = df_train[ (df_train.y==number_a) | (df_train.y==number_b) ]
    X_train = X_train[df_train.index,...]
    y_train = y_train[df_train.index,...]
    X_test = []
    y_test = []
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.2, random_state=1)
    print("Size of data for training and testing. Format (elements,dimension,dimension)")
    print("Training: {}".format(X_train.shape))
    print("Testing: {}".format(X_test.shape)) 
    T = 100
    GAMMA = 0.05
    print("T ={}, GAMMA = {}".format(T,GAMMA))
    mwu = MWU(GAMMA)
    P = mwu.train( train=(X_train, y_train), test=(X_test,y_test), T=T)