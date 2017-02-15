from symbol import yield_arg

import scipy.io as sio
import numpy as np
import random
import math
from scipy.sparse import vstack
import matplotlib.pyplot as plt



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

def get_precision(X, Y, Theta):
    # compute P(Y|X)
    Z = np.sum(np.exp(Theta * X.todense()), 0)
    P_Y_sachant_X = np.exp(Theta * X.todense()) / np.tile(Z, (4,1))
    maxi = np.argmax(P_Y_sachant_X, 0)
    maxi_truth = np.argmax(Y.T,0)
    return np.sum(maxi==maxi_truth)/len(Y)

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

'''
# BATCH
taux = [0.0001, 0.0005, 0.0008]

for k in range(3):
    k=0
    # initial values
    logV = []
    print(type(logV))
    precisions = np.matrix([0, 0])
    taux_dapprentissage = taux[k]
    converged = False

    # reset Theta random (the ame for the three learning rate)
    Theta = Theta_save
    yixi = YA.T * XA.T #left part for the gradient. constant

    counter = 0
    while not converged:

        counter += 1
        oldPrecisions = precisions

        # compute log vraisemblance
        Z = np.sum(np.exp(Theta * XA.todense()),0)
        numerator = np.sum(np.multiply(np.dot(YA, Theta).T, XA.todense()), 0)
        logV.append(np.sum(np.subtract(numerator, np.log(Z))))
        print("Log Vraisemblance = %d" % logV[-1])

        # compute P(Y|X)
        P_Y_sachant_X = np.exp(Theta * XA.todense()) / np.tile(Z, (4,1))
        # compute gradient
        right_part = P_Y_sachant_X * XA.T
        gradient = -np.subtract(yixi, right_part)

        # compute training set precision
        precisionsA = get_precision(XA,YA, Theta)
        print('Precision sur l\' ensemble d\'apprentissage : %f' %precisionsA)


        # compute validation set precision
        precisionsV = get_precision(XV,YV, Theta)
        print('Precision sur l\' ensemble de validation: %f' %precisionsV)

        precisions = np.vstack((precisions, [precisionsA, precisionsV]))
        # update theta
        Theta = Theta - taux_dapprentissage * gradient;

        # check convergence
        if counter > 100000 or abs(oldPrecisions[-1, -1] - precisions[-1, -1]) < 0.0001:
            converged = True

    # remove first entry which was only for initialization
    precisions = precisions[1:,:]

    # compute test set precision
    precisionsT = get_precision(XT, YT, Theta)
    print('Precision sur l\' ensemble de test: %f' %precisionsT)



precisions = np.hstack((precisions, np.full((len(precisions),1), precisionsT)))


g1 = plt.figure(1)
plt.plot(np.arange(len(precisions)), precisions)
plt.xlabel('itérations')
plt.ylabel('précision')
plt.title('Précision de la descente de gradient par batch')
plt.legend(['learning set', 'validation set', 'test set'])
g1.show()



g2 = plt.figure(2)
plt.plot(np.arange(len(logV)), logV)
plt.xlabel('itérations')
plt.ylabel('log-vraisemblance')
plt.title('Log vraisemblance pendant la descente de gradient par batch')
g2.show()

plt.show()

'''
# MINI BATCH
def get_mini_batch(X, Y, n):
    size_data = Y.shape[0]
    indices = np.arange(size_data)
    np.random.shuffle(indices)



    Y = Y[indices, :]
    X = X[:, indices]

    limit = size_data - (size_data % (n-1))
    x = X.todense()[:,:limit]
    x = np.hsplit(x, n-1)
    x.append(X[:,limit:].todense())

    y = Y[:limit,:]
    y = np.vsplit(y, n-1)
    y.append(Y[limit:,:])
    return x,y

#initial values
logV = []
mbPrecisions = np.matrix([0, 0])

Theta = Theta_save
converged = False
NB_mini_batch = 20
taux_dapprentissage = 0.0001
t = 1

while not converged:

    # compute mini batch
    X_batchs, Y_batchs = get_mini_batch(XA, YA, NB_mini_batch)
    oldMbPrecisions = mbPrecisions

    taux_dapprentissage = t/2

    for i in range(NB_mini_batch):

        # compute log vraisemblance
        Z = np.sum(np.exp(Theta * X_batchs[i]),0)
        print(type(X_batchs[i]))
        numerator = np.sum(np.multiply(np.dot(Y_batchs[i],Theta).T, X_batchs[i]), 0)
        logV.append(np.sum(np.subtract(numerator, np.log(Z))))
        print("Log Vraisemblance = %d" % logV[-1])

        # compute P(Y|X)
        P_Y_sachant_X = np.exp(Theta * X_batchs[i]) / np.tile(Z, (4, 1))

        # compute gradient
        right_part = P_Y_sachant_X * X_batchs[i].T
        yixi = Y_batchs[i].T * X_batchs[i].T
        gradient = -np.subtract(yixi, right_part) / len(X_batchs[i])

        # update Theta
        Theta = Theta - taux_dapprentissage * gradient

    # compute precision on learning set
    precisionA = get_precision(XA, YA, Theta)
    print("precision ensemble d''apprentissage : %f" % precisionA)
    # compute precision on validation set
    precisionV = get_precision(XV, YV, Theta)
    print("precision ensemble de validation : %f" % precisionV)
    # update precision
    mbPrecisions = np.vstack((mbPrecisions, [precisionA, precisionV]))

    # check convergence
    if abs(oldMbPrecisions[-1, -1] - mbPrecisions[-1, -1]) < 0.0001:
        converged = True

    t += 1

# remove first entry which was only for initialization
mbPrecisions = mbPrecisions[1:,:]

# compute test set precision
precisionsT = get_precision(XT, YT, Theta)
print('Precision sur l\' ensemble de test: %f' %precisionsT)


mbPrecisions = np.hstack((mbPrecisions, np.full((len(mbPrecisions),1), precisionsT)))


g1 = plt.figure(1)
plt.plot(np.arange(len(mbPrecisions)), mbPrecisions)
plt.xlabel('itérations')
plt.ylabel('précision')
plt.title('Précision de la descente de gradient par batch')
plt.legend(['learning set', 'validation set', 'test set'])
g1.show()



g2 = plt.figure(2)
plt.plot(np.arange(len(logV)), logV)
plt.xlabel('itérations')
plt.ylabel('log-vraisemblance')
plt.title('Log vraisemblance pendant la descente de gradient par batch')
g2.show()
plt.show()