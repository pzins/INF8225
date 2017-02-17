import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.sparse import vstack
from scipy import sparse


# function which create traning, validation and test sets from data
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


# compute the precision on a set
def get_precision(X, Y, Theta):
    # compute P(Y|X)
    Z = np.sum(np.exp(Theta * X), 0)
    P_Y_sachant_X = np.exp(Theta * X) / np.tile(Z, (4,1))
    maxi = np.argmax(P_Y_sachant_X, 0)
    maxi_truth = np.argmax(Y.T,0)
    return np.sum(maxi==maxi_truth)/Y.shape[0]

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


# creation of Y
j = np.array(j)
m = len(j)
Y = []
for i in range(j.shape[1]):
    Y.append([0]*n)
    Y[i][j[0,i] - 1] = 1
Y = np.array(Y)
# Y = sparse.csr_matrix(Y)

XA, XV, XT, YA, YV, YT = create_train_valid_test_splits(X,Y)

# Initilisation of Theta
save_Theta = np.asmatrix(np.random.random((4,101))-0.5)
Theta = np.asmatrix(np.copy(save_Theta))


# BATCH
# initial values
logV = np.array([0]) # save log likelihood during iterations of gradient descent
precisions = np.matrix([0, 0]) # save precisions on learning and validation set during gradient descent

converged = False

taux_dapprentissage = 0.0005
v =  0 # momentum initialisation
gamma = 0.5 # momentum factor
lmbda = 0.1 #regularization factor

yixi = YA.T * XA.T # left part for the gradient computation, out of the while loop because it is constant



while not converged:

    oldPrecisions = precisions

    # compute log vraisemblance
    Z = np.sum(np.exp(Theta * XA),0)
    numerator = np.sum(np.multiply(np.dot(YA,Theta).T, XA.todense()), 0)
    logV = np.vstack((logV, np.sum(np.subtract(numerator, np.log(Z)))))
    print("Log Vraisemblance = %d" % logV[-1])

    # compute P(Y|X)
    P_Y_sachant_X = np.exp(Theta * XA) / np.tile(Z, (4,1))
    # compute gradient
    right_part = P_Y_sachant_X * XA.T

    # compute gradient with regularization
    gradient = -np.subtract(yixi, right_part) + np.tile(lmbda * np.sum(Theta, 1), Theta.shape[1])

    # compute training set precision
    precisionsA = get_precision(XA,YA, Theta)
    print('Precision sur l\'ensemble d\'apprentissage : %f' %precisionsA)


    # compute validation set precision
    precisionsV = get_precision(XV,YV, Theta)
    print('Precision sur l\'ensemble de validation: %f' %precisionsV)

    precisions = np.vstack((precisions, [precisionsA, precisionsV]))

    # update Theta with momentum
    v = gamma*v + taux_dapprentissage * gradient
    Theta = Theta - v

    # update Theta without momentum
    # Theta -= taux_dapprentissage*gradient


    # check convergence
    if abs(oldPrecisions[-1, -1] - precisions[-1, -1]) < 0.001:
        converged = True

# remove first entry which was only for initialization
precisions = precisions[1:,:]
logV = logV[1:,:]

# compute test set precision
precisionsT = get_precision(XT, YT, Theta)
print('Precision sur l\' ensemble de test: %f' %precisionsT)



precisions = np.hstack((precisions, np.full((len(precisions),1), precisionsT)))

plt.figure(1)
plt.plot(np.arange(len(precisions)), precisions)
plt.xlabel('itérations')
plt.ylabel('précision')
plt.title('Précision de la descente de gradient par batch')
plt.legend(['learning set', 'validation set', 'test set'], loc=4)


plt.figure(2)
plt.plot(np.arange(len(logV)), logV)
plt.xlabel('itérations')
plt.ylabel('log-vraisemblance')
plt.title('Log vraisemblance pendant la descente de gradient par batch')

plt.show()




# MINI BATCH
Theta = save_Theta


# return n mini-batch from input data
def get_mini_batch(X, Y, n):
    size_data = Y.shape[0] # number of instance

    # shuffle X Y
    indices = np.arange(size_data)
    np.random.shuffle(indices)
    Y = Y[indices, :]
    X = X[:, indices]

    # limit which is divisible by n-1
    limit = size_data - (size_data % (n-1))

    # split into 2 parts (n-1 batchs / the last batch)
    x = X.todense()[:,:limit]
    # split into n-1 batchs
    x = np.hsplit(x, n-1)
    # add the last batch (which could be smaller)
    x.append(X[:,limit:].todense())

    # the same for Y
    y = Y[:limit,:]
    y = np.vsplit(y, n-1)
    y.append(Y[limit:,:])
    x = [sparse.csr_matrix(i) for i in x]

    return x,y


#initial values
logV = np.array([0])
mbPrecisions = np.matrix([0, 0]) # precision epoch iteration
mbPrecisions_mini_batch = np.matrix([0, 0]) # precision iteration on each batch

converged = False
NB_mini_batch = 20
t = 1
v = 0 # momentum initialisation
gamma = 0.5 # momentum factor
lmbda = 0.1 #regularization factor

while not converged:

    # compute mini batch
    X_batchs, Y_batchs = get_mini_batch(XA, YA, NB_mini_batch)
    oldMbPrecisions = mbPrecisions

    taux_dapprentissage = 1/(t+1)

    # compute log vraisemblance
    Z = np.sum(np.exp(Theta * XA), 0)
    numerator = np.sum(np.multiply(np.dot(YA, Theta).T, XA.todense()), 0)
    logV = np.vstack((logV, np.sum(np.subtract(numerator, np.log(Z)))))

    for i in range(NB_mini_batch):

        # compute P(Y|X)
        Z = np.sum(np.exp(Theta * X_batchs[i]),0)
        P_Y_sachant_X = np.exp(Theta * X_batchs[i]) / np.tile(Z, (4, 1))

        # compute gradient
        right_part = P_Y_sachant_X * X_batchs[i].T
        yixi = Y_batchs[i].T * X_batchs[i].T
        gradient = -np.subtract(yixi, right_part) / X_batchs[i].shape[1]
        # + np.tile(lmbda * np.sum(Theta, 1), Theta.shape[1]))

        # compute precision on learning set
        precisionA = get_precision(XA, YA, Theta)

        # compute precision on validation set
        precisionV = get_precision(XV, YV, Theta)
        mbPrecisions_mini_batch = np.vstack((mbPrecisions_mini_batch, [precisionA, precisionV]))

        # update Theta with momentum
        v = gamma * v + taux_dapprentissage * gradient
        Theta = Theta - v

    # compute precision on learning set
    precisionA = get_precision(XA, YA, Theta)
    print("precision ensemble d'apprentissage : %f" % precisionA)
    # compute precision on validation set
    precisionV = get_precision(XV, YV, Theta)
    print("precision ensemble de validation : %f" % precisionV)
    # update precision
    mbPrecisions = np.vstack((mbPrecisions, [precisionA, precisionV]))

    # check convergence
    if abs(oldMbPrecisions[-1, -1] - mbPrecisions[-1, -1]) < 0.001:
        converged = True

    t += 1

# remove first entry which was only for initialization
mbPrecisions = mbPrecisions[1:,:]
mbPrecisions_mini_batch = mbPrecisions_mini_batch[1:,:]
logV = logV[1:,:]

# compute test set precision
precisionsT = get_precision(XT, YT, Theta)
print('Precision sur l\' ensemble de test: %f' %precisionsT)


mbPrecisions = np.hstack((mbPrecisions, np.full((len(mbPrecisions),1), precisionsT)))
mbPrecisions_mini_batch = np.hstack((mbPrecisions_mini_batch, np.full((len(mbPrecisions_mini_batch),1), precisionsT)))


g1 = plt.figure(1)
plt.plot(np.arange(len(mbPrecisions)), mbPrecisions)
plt.xlabel('itérations')
plt.ylabel('précision')
plt.title('Précision de la descente de gradient par mini-batch (chaque epoque')
plt.legend(['learning set', 'validation set', 'test set'],loc=4)
g1.show()

g2 = plt.figure(2)
plt.plot(np.arange(len(mbPrecisions_mini_batch)), mbPrecisions_mini_batch)
plt.xlabel('itérations')
plt.ylabel('précision')
plt.title('Précision de la descente de gradient par mini-batch (chaque iteration pr chaque mini-batch)')
plt.legend(['learning set', 'validation set', 'test set'],loc=4)
g2.show()

g3 = plt.figure(3)
plt.plot(np.arange(len(logV)), logV)
plt.xlabel('itérations')
plt.ylabel('log-vraisemblance')
plt.title('Log vraisemblance pendant la descente de gradient par batch')
g3.show()

plt.show()



# graph : batch and mini-batch precision on the same graph
# g1 = plt.figure(1)
# plt.plot(np.arange(len(precisions)), precisions, '--',)
# plt.plot(np.arange(len(mbPrecisions)), mbPrecisions)
# plt.xlabel('itérations')
# plt.ylabel('précision')
# plt.title('Comparaison précisions, batch / mini-batch')
# plt.legend(['batch : learning set', 'batch : validation set', 'batch : test set', 'mini-batch : learning set', 'mini-batch : validation set', 'mini-batch : test set'],loc=4)
# g1.show()
#
# plt.show()
