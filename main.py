# Algorithms and Uncertainty (2019) - PUC-Rio
#
# MWU Classifier to distinguish two digits based on one pixel (MNIST)
#
# Last updated: 27/04/2019
#
# Authors: Ítalo G. Santana & Rafael Azevedo M. S. Cruz

from __future__ import print_function
import random
import numpy as np
import time
import sys
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Chosen numbers to be distinguished.
number_a = 4
number_b = 6


# Computes the accuracy of a prediction given a true output y_true.
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
    # Dataset X values normalization (values are between 0 and 255 for pixels).
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    return X_train, y_train, X_test, y_test



# Single classifier for one digit which distinguishes two digits a and b based on one pixel.
class single_clf:


    def __init__(self, x, y, p=None):
        # Position indexes for the pixel that best predicts the digits.
        self.idx = None
        # sign == 1 iff predicts number_a else predicts number_b.
        self.sign = 0
        # weights for each digit instance in y (this characterizes the classifier).
        self.p = None
        #print("x size = {}".format(x.shape))
        self.train(x,y,p)


    # Predicts the number based on a pixel (i, j). Comparing the pixel (i, j) of
    # all the 28x28 pixel digits in x, predicts each digit.
    # x.shape = (n, 28, 28) where n is the number of digits in x and pixels are 28x28.
    def predict_unit(self, x, sign, i, j, posval=number_a, negval=number_b):
        # fval is the false value (the number a or b that is not being predicted)
        # If sign > 0, then posval is being predicted (fval == negval, tval == posval)
        # If sign < 0, then negval is being predicted (fval == posval, tval == negval)
        fval =  negval if sign >= 0 else  posval
        tval =  posval if sign  > 0 else  negval
        y = np.ones(x.shape[0])*fval
        # Predicts as a tval digit based on pixel (i, j) for each digit in x.
        y[ x[:,i,j] > 0 ] =  tval
        return y

    def train(self, x, y, p=None):
        if p is None:
            self.p = np.ones(y.shape[0])
        else:
            self.p = p
        if self.idx is not None:
            # hitA predicts number_a based on pixel (self.idx[0],self.idx[1]).
            # hitB predicts number_b based on pixel (self.idx[0],self.idx[1]).
            hitA = np.sum((self.predict_unit(x, 1,self.idx[0],self.idx[1])==y).astype(float) * self.p)
            hitB = np.sum((self.predict_unit(x,-1,self.idx[0],self.idx[1])==y).astype(float) * self.p)
            # If hitA >= hitB then number_a predictions were better than number_b predictions.
            # Otherwise, number_b predictions are more accurate based for this pixel.
            self.sign = 1 if (hitA>=hitB) else -1
        else:
            besthit=0
            # Iterates through every possible 28x28 pixels and predicts a digit a or b based on this single pixel (i, j).
            # Stores the pixel (i, j) for which more predictions were correct (either a or b).
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    # hitA predicts number_a based on pixel (i, j).
                    # hitB predicts number_b based on pixel (i, j).
                    hitA = np.sum((self.predict_unit(x, 1,i,j)==y).astype(float) * self.p)
                    hitB = np.sum((self.predict_unit(x,-1,i,j)==y).astype(float) * self.p)
                    if(hitA > besthit):
                        # The best pixel for predicting A.
                        besthit = hitA
                        self.sign = 1
                        self.idx =(i,j)
                    if(hitB > besthit):
                        # The best pixel for predicting B.
                        besthit = hitB
                        self.sign = -1
                        self.idx =(i,j)


    def predict(self, x, posval=1, negval=-1):
        # Predicts the digit based on the single best pixel determined during training.
        # Best pixel for prediction is (self.idx[0], self.idx[1])
        fval =  negval if self.sign >= 0 else  posval
        tval =  posval if self.sign  > 0 else  negval
        y = np.ones(x.shape[0])*fval
        y[ x[:,self.idx[0],self.idx[1]] > 0 ] =  tval
        return y



# MWU Classifier for distinguishing two digits A and B.
class MWU:


    def __init__(self, gamma):
        self.gamma = gamma


    # MWU algorithm to compute the final weight w_i of each expert i in an horizon T.
    def train(self, train, test, T=100, w=None):

        x_train, y_train = train
        x_test, y_test = test
        x_train = (x_train>0).astype(float)

        self.learners = []
        self.t_hist = []
        self.test_accuracy = []
        self.train_accuracy = []

        eps = np.sqrt( np.log(x_train.size) / T )
        # Initializing the weight for each digit instance in x_train as 1/x_train.shape[0]
        P = np.ones(x_train.shape[0]) / x_train.shape[0]
        # Initializes the 28x28 pixel matrix such that w(i, j) is 1 iff pixel (i, j) is selected as a good classifier pixel; else w(i, j) is 0.
        self.w = np.zeros( (x_train.shape[1],x_train.shape[2]) )

        train_file = open("train_log.txt", "w")

        for it in range(T):
            # Create and train a classifier ci with weights P
            ci = single_clf(x_train, y_train, p=P)

            # Predicts the digits a and b from x_train
            y_p = ci.predict(x_train, posval=number_a, negval=number_b)
            # Computes the weighted (P) sum of predictions y_p that are correct (equal to digit label in y_train).
            acc = np.sum((y_p==y_train).astype(float)*P)
            if acc < 0.5 + self.gamma:
                train_file.write("There is no more {}-weak-learners".format(0.5 + self.gamma))
                print ("\n\tThere is no more {}-weak-learners".format(0.5 + self.gamma))
                break

            # Increments 1 to the pixel position (ci.idx[0], ci.idx[1]) that is a 0.5+gamma-weak-learner.
            # Note: the same pixel position might be selected in different rounds.
            self.w[ci.idx[0],ci.idx[1]] += 1
            # Stores the current weak-learner.
            self.learners.append(ci)
            # Computes all the digits for which many predictions were wrong (misses).
            miss = (y_p!=y_train)
            # Reduces the weight of these digit instances by exp of eps.
            P[miss] *= np.exp(eps)
            # Updates weights P such that their sum is exactly 1.
            P = P/np.sum(P)

            ############# history log....############
            # Predicts and computes the validation accuracy.
            y_p = self.predict(x_test)
            v_acc = accuracy(y_p,y_test)
            # Predicts and computes the test accuracy.
            y_p = self.predict(x_train)
            t_acc = accuracy(y_p,y_train)

            self.test_accuracy.append(v_acc)
            self.train_accuracy.append(t_acc)
            self.t_hist.append(it)
            ##########################################

            train_file.write("\niteration {}: Validation accuracy: {}".format(it, v_acc))
            print("\niteration {}: Validation accuracy: {}".format(it, v_acc))

        print("\n\n{} : Number of learners = {}".format(it,len(self.learners)))
        print("\n\n{} : Learners: ")
        for i in range(len(self.learners)):
            nmbr = "A" if self.learners[i].sign > 0 else "B"
            print("\n\nLearner {} ==> (pixel_i, pixel_j) = {} ; predicted number = {} ; p = {} ".format(i, self.learners[i].idx, nmbr, self.learners[i].p))
            train_file.write("\n\nLearner {} ==> (pixel_i, pixel_j) = {} ; predicted number = {} ; p = {} ".format(i, self.learners[i].idx, nmbr, self.learners[i].p))

        for i in range(self.w.shape[0]):
            print("\n")
            train_file.write("\n")
            for j in range(self.w.shape[1]):
                print("{} ".format(self.w[i, j]), end="")
                train_file.write("{} ".format(self.w[i, j]))

        print("\n")

        train_file.write("\n\n\t{} : Final validation accuracy: {}".format(it,v_acc))
        train_file.write("\n\n\t{} : Final test accuracy: {}\n\n".format(it,t_acc))
        print("\n\n{} : Final validation accuracy: {}".format(it,v_acc))
        print("\n\n{} : Final test accuracy: {}\n\n".format(it,t_acc))

        train_file.close()

        # Plotting classfifier over iterations
        th = np.array(self.train_accuracy)
        x = np.array(self.t_hist)

        plt.plot(x, th, label="Final classifier")
        plt.legend()
        plt.xlabel('Iterações') 
        plt.ylabel('Qualidade') 
        plt.axis([0,it + 5,0.8,1])
        plt.show()

        return P


    # Considers the prediction done by all the learners added (already weighted).
    def predict(self, x, posval=number_a, negval=number_b):
        y = np.zeros(x.shape[0])
        # Predicts the digits value based on the contribution of each learner.
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
    print("Size of data for training and testing. Format (elements, dimension, dimension)")
    print("Training: {}".format(X_train.shape))
    print("Testing: {}".format(X_test.shape))
    T = 150
    GAMMA = 0.05
    print("T = {}, GAMMA = {}".format(T,GAMMA))

    # Creates and trains a mwu classifier.
    mwu = MWU(GAMMA)
    P = mwu.train( train=(X_train, y_train), test=(X_test,y_test), T=T)
