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

# Calculating I(X,T) and I(T,Y)
def information_plane(X,Y,activations_list, layers, EPOCHS):

    I_XT = np.zeros((len(layers),EPOCHS))
    I_TY = np.zeros((len(layers),EPOCHS))

    for layer in range(0,len(layers)):
        for epoch in range(0,EPOCHS):
            I_XT[layer,epoch] = Mutual_Info(activations_list[epoch][layer][0],X)
            I_TY[layer,epoch] = Mutual_Info(activations_list[epoch][layer][0],Y)

    return I_XT,I_TY