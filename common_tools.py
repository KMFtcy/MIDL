import numpy as np

# Discretization of continous values from the layers
def discretization(activations_list,bins,layers, EPOCHS):
## -------------------------------------
#     df: DataFrame
#     bins: int
#     layers: list with the number of neurons in each layer
#     EPOCHS: int
## -----------------------------------
    n_bins = bins
    for layer in range(0,len(layers)):
        for epoch in range(0,EPOCHS):
            bins = np.linspace(min(np.min(activations_list[epoch][layer][0],axis=1)),
                               max(np.max(activations_list[epoch][layer][0],axis=1)), n_bins+1)
            activations_list[epoch][layer][0] = np.digitize(activations_list[epoch][layer][0], bins)
    return activations_list

# Usage: Discretize the continous output of the layers
# activations_list = discretization(activations_list,30, layers, EPOCHS)