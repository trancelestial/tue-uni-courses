# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

# plot function
# input: Nx3 matrix of values & title string
def plot(vals,title=''):
    plt.close()
    vals /= np.tile(np.sum(vals,1),(3,1)).transpose()
    f, axarr = plt.subplots(1,10,figsize=(10,2))
    plt.suptitle(title, fontsize=16, fontweight='bold')
    for i in range(vals.shape[0]):
        axarr[i].barh(y=[0,1,2], width=np.ones(3), color='white',edgecolor='black',linewidth=2.0)
        axarr[i].barh([0,1,2],vals[i],color='red')
        axarr[i].axis('off')
    plt.show()

# unary: Nx3 matrix specifying unary likelihood of each state
unary = np.array([[0.7,0.1,0.2],[0.7,0.2,0.1],[0.2,0.1,0.7],[0.7,0.2,0.1],
                  [0.2,0.6,0.2],[0.1,0.8,0.1],[0.4,0.3,0.3],[0.1,0.8,0.1],
                  [0.1,0.1,0.8],[0.1,0.5,0.4]])
# pairwise: 3x3 matrix specifying transition probabilities (rows=t -> columns=t+1)
pairwise = np.array([[0.8,0.2,0.0],[0.2,0.6,0.2],[0.0,0.2,0.8]])

# # Inference
# * **Outputs:**
#   * map: array comprising the estimated MAP state of each variable

# plot unaries
plot(unary,'Unary')

# model parameters (number of variables/states)
[num_vars,num_states] = unary.shape

msg = np.zeros([num_vars, num_states]) # (num_vars-1) x num_states matrix
# TODO # compute messages for the chain structured Markov random field
def init(start_factor, f):
    if(start_factor==True):
        return f
    else:
        return 1

msg[0] = init(start_factor=True, f=unary[0])

def factor_to_node(neighbors):
    result = np.ones(neighbors[0].shape)
    for y in neighbors:
        result = result * y
    return result



#Since we have a chain there is only one incoming node.
def node_to_factor(neighbors, g):
    return g @ neighbors



for idx in range(msg.shape[0] - 1):
    msg[idx + 1] = factor_to_node([node_to_factor(msg[idx], pairwise), unary[idx+1]])

plot(msg,'Messages')

map = np.zeros(num_vars, dtype=int)
map = np.argmax(msg, axis=1) + 1

# print MAP state
print("MAP Estimate:")
print(map)