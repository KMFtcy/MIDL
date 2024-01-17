import numpy as np
#Entropy
def entropy(Y):
    """
    Shanon Entropy
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    H(Y|X) = H(Y;X) - H(X)
    """
    return jEntropy(Y, X) - entropy(X)

#Mutual Information
def Mutual_Info(Y, X):
    """
    I(Y;X) = H(Y) - H(Y|X)
    """
    return entropy(Y) - cEntropy(Y,X)