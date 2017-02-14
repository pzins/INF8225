from symbol import yield_arg

import scipy.io as sio
import numpy as np
import random
import math
from scipy.sparse import vstack



def create_train_valid_test_splits(X, Y):
    indices = np.arange(Y.shape[0])
    np.random.shuffle(indices)


    index = np.floor([0.7*len(indices), 0.85*len(indices)]).astype(int)
    XA = X[:, indices[0:index[0]]]
    XV = X[:, indices[index[0]:index[1]]]
    XT = X[:, indices[index[1]:]]

    YA = Y[indices[0:index[0]],:]
    YV = Y[indices[index[0]:index[1]],:]
    YT = Y[indices[index[1]:],:]


    return XA, XV, XT, YA, YV, YT


data = sio.loadmat("20news_w100")
groupnames = data["groupnames"]
newsgroups = data["newsgroups"]
documents = data["documents"]
wordlist = data["wordlist"]

n = 4
m = newsgroups.shape[1]
o = np.ones(m)
i = range(m)
j = newsgroups
Y = j
X = documents
X = vstack((X, np.ones(16242)),"csc")


j = np.array(j)

n = 4
m = len(j)
Y = []


for i in range(j.shape[1]):
    Y.append([0]*n)
    Y[i][j[0,i] - 1] = 1
Y = np.array(Y)

XA, XV, XT, YA, YV, YT = create_train_valid_test_splits(X,Y)

Theta_save = np.random.random((4,101))-0.5
Theta = Theta_save


# BATCH
taux = [0.0001, 0.0005, 0.0008]

for k in range(3):
    k=0
    # initial values
    logV = []
    print(type(logV))
    precisions = [0, 0]
    taux_dapprentissage = taux[k]
    converged = False

    # reset Theta random (the ame for the three learning rate)
    Theta = Theta_save


    yixi = YA.T * XA.T #left part for the gradient. constant

    while not converged:
        oldPrecisions = precisions

        #compute log vraisemblance
        Z = np.sum(np.exp(Theta * XA.todense()),0)
        numerator = np.sum(np.multiply(np.dot(YA, Theta).T, XA.todense()), 0)
        logV.append(np.sum(np.subtract(numerator, np.log(Z))))
        print("Log Vraisemblance = %d" % logV[-1])

        # compute P(Y|X)
        P_Y_sachant_X = np.exp(Theta * XA.todense()) / np.tile(Z, (4,1))
        # compute gradient
        right_part = P_Y_sachant_X * XA.T
        gradient = -np.subtract(yixi, right_part)

        #compute training set precision
        precisionsA = get_precision(XA,YA, Theta)

        print(P_Y_sachant_X)

        converged = True