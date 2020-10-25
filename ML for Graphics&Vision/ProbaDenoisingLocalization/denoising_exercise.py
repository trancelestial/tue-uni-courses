# import modules
import numpy as np
import matplotlib.pyplot as plt
import scipy
import imageio

# # Set Model Parameters & Create Factors
# * **Outputs:**
#   * num_vars: number of variables in MRF
#   * num_states: number of states for each variable (binary -> num_states=2)
#   * factors: list of dictionaries where each entry of the list comprises two variables:
#     (vars = array of variables involved, vals = vector/matrix specifying the vector)

# load and plot input image
img = imageio.imread('image.png') / 255
plt.imshow(img, interpolation='nearest');
plt.gray()
plt.show()

# model parameters
[h, w] = img.shape  # get width & height of image
num_vars = w * h  # number of variables = width * height
num_states = 2  # binary segmentation -> two states

# initialize factors (list of dictionaries), each factor comprises:
#   vars: array of variables involved
#   vals: vector/matrix of factor values

def ising2(a,b):
    if (a == b):
        return 1
    else:
        return 0

def ising(a,b):
    if (a == b):
        return np.array([0, 1])
    else:
        return np.array([1, 0])

factors = []
alpha = 1.8
# add unary factors
for i in range(h):
    for j in range(w):
        #Add unary factor
        factors.append({'vars': [i*w+j], 'vals': np.array([1-img[i][j],img[i][j]])})

        truth_table = np.array([[1,0],[0,1]])
        #Right border
        if((j == (w-1)) and (i < (h-1))):
            factors.append({'vars': [i * w + j, i * w + j + w], 'vals': alpha * np.array(truth_table)})  # Factor to the bottom
            #factors.append({'vars': [i * w + j, i * w + j + w], 'vals': alpha * np.array([img[i][j], img[i + 1][j]])})  # Factor to the bottom
            #factors.append({'vars': [i * w + j, i * w + j + w], 'vals': alpha * ising(img[i][j], img[i + 1][j])})  # Factor to the bottom
        #Bottom border
        elif(i == (h-1) and (j < (w-1))):
            factors.append({'vars': [i * w + j, i * w + j + 1], 'vals': alpha * np.array(truth_table)})  # Factor to the right
            #factors.append({'vars': [i * w + j, i * w + j + 1], 'vals': alpha * np.array([img[i][j], img[i][j+1]])})  # Factor to the right
            #factors.append({'vars': [i * w + j, i * w + j + 1], 'vals': alpha * ising(img[i][j], img[i][j+1])})       # Factor to the right
        #Bottom right corner -> No factors to right or bottom
        elif(i == (h-1) and (j == (w-1))):
            pass
        #All other nodes add a pairwise factor to right and towards the bottom
        else:
            factors.append({'vars': [i * w + j, i * w + j + 1], 'vals': alpha * np.array(truth_table)})  # Factor to the right
            factors.append({'vars': [i * w + j, i * w + j + w], 'vals': alpha * np.array(truth_table)})  # Factor to the bottom

            #factors.append({'vars': [i * w + j, i * w + j + 1], 'vals': alpha * np.array([img[i][j], img[i][j+1]])}) #Factor to the right
            #factors.append({'vars': [i * w + j, i * w + j + w], 'vals': alpha * np.array([img[i][j], img[i+1][j]])})#Factor to the bottom

            #factors.append({'vars': [i * w + j, i * w + j + 1], 'vals': alpha * ising(img[i][j], img[i][j + 1])})  # Factor to the right
            #factors.append({'vars': [i * w + j, i * w + j + w], 'vals': alpha * ising(img[i][j], img[i + 1][j])})  # Factor to the bottom
# add pairwise factors


# # Initialize Messages
# * **Inputs:**
#   * num_vars, num_states, factors
# * **Outputs:**
#   * msg_fv: dictionary of all messages from factors to variables
#   * msg_vf: dictionary of all messages from variables to factors
#   * ne_var: list which comprises the neighboring factors of each variable

# initialize all messages
msg_fv = {}  # f->v messages (dictionary)
msg_vf = {}  # v->f messages (dictionary)
ne_var = [[] for i in range(num_vars)]  # neighboring factors of variables (list of list)

# set messages to zero; determine factors neighboring each variable
for [f_idx, f] in enumerate(factors):
    for v_idx in f['vars']:
        msg_fv[(f_idx, v_idx)] = np.zeros(num_states)  # factor->variable message
        msg_vf[(v_idx, f_idx)] = np.zeros(num_states)  # variable->factor message
        ne_var[v_idx].append(f_idx)  # factors neighboring variable v_idx

# status message
print("Messages initialized!")

# # Inference
# * **Inputs:**
#   * num_vars, num_states, factors, msg_fv, msg_vf, ne_var
# * **Outputs:**
#   * max_marginals: num_vars x num_states array of estimated max-marginals
#   * map_est: array comprising the estimated MAP state of each variable
#
# * **Algorithm Pseudocode:**
#   * For N=10 iterations do:
#     * Update all **unary factor-to-variable** messages: $\lambda_{f\rightarrow x}(x) = f(x)$
#     * Update all **pairwise factor-to-variable** messages: $\lambda_{f\rightarrow x}(x) = \max_y \left[f(x,y)+\lambda_{y\rightarrow f}(y)\right]$
#     * Update all **variable-to-factor** messages: $\lambda_{x\rightarrow f}(x) = \sum_{g\in\{ ne(x)\setminus f\}}\lambda_{g\rightarrow x}(x)$
#   * Calculate **Max-Marginals**: $\gamma_x(x) = \sum_{g\in\{ ne(x)\}}\lambda_{g\rightarrow x}(x)$
#   * Calculate **MAP Solution**: $x^* = \underset{x}{\mathrm{argmax}} ~ \gamma_x(x)$

# run inference
for it in range(30):
    # for all factor-to-variable messages do
    for [key, msg] in msg_fv.items():

        # shortcuts to variables
        f_idx = key[0]  # factor (source)
        v_idx = key[1]  # variable (target)
        f_vars = factors[f_idx]['vars']  # variables connected to factor
        f_vals = factors[f_idx]['vals']  # vector/matrix of factor values

        # unary factor-to-variable message
        if np.size(f_vars) == 1:
            msg_fv[(f_idx, v_idx)] = f_vals

        # pairwise factor-to-variable-message
        else:

            # if target variable is first variable of factor
            if v_idx == f_vars[0]:
                msg_in = np.tile(msg_vf[(f_vars[1], f_idx)], (num_states, 1))
                msg_fv[(f_idx, v_idx)] = (f_vals + msg_in).max(1)  # max over columns

            # if target variable is second variable of factor
            else:
                msg_in = np.tile(msg_vf[(f_vars[0], f_idx)], (num_states, 1))
                msg_fv[(f_idx, v_idx)] = (f_vals + msg_in.transpose()).max(0)  # max over rows

        # normalize
        msg_fv[(f_idx,v_idx)] = msg_fv[(f_idx,v_idx)] - np.mean(msg_fv[(f_idx,v_idx)])

    # for all variable-to-factor messages do
    for [key, msg] in msg_vf.items():

        # shortcuts to variables
        v_idx = key[0]  # variable (source)
        f_idx = key[1]  # factor (target)

        # add messages from all factors send to this variable (except target factor)
        # and send the result to the target factor
        msg_vf[(v_idx, f_idx)] = np.zeros(num_states)
        for f_idx2 in ne_var[v_idx]:
            if f_idx2 != f_idx:
                msg_vf[(v_idx, f_idx)] += msg_fv[(f_idx2, v_idx)]

        # normalize
        msg_vf[(v_idx,f_idx)] = msg_vf[(v_idx,f_idx)] - np.mean(msg_vf[(v_idx,f_idx)])

# calculate max-marginals (num_vars x num_states matrix)
max_marginals = np.zeros([num_vars, num_states])
for v_idx in range(num_vars):

    # add messages from all factors sent to this variable
    max_marginals[v_idx] = np.zeros(num_states)
    for f_idx in ne_var[v_idx]:
        max_marginals[v_idx] += msg_fv[(f_idx, v_idx)]

# get MAP solution
map_est = np.argmax(max_marginals, axis=1)

# # Show Inference Results

# plot MAP estimate
plt.imshow(map_est.reshape(h, w), interpolation='nearest');
plt.gray()
plt.show()