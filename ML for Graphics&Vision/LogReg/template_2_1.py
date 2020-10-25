from __future__ import print_function
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-5
def log(x):
    return np.log(x + eps)

class LogisticRegression():
    def __init__(self):
        self.weights = np.array([])
        self.losses = []  
        self.lr = 1e-1
        self.max_iter = 10 
        
    def init_weights(self, dim):        
        # uniform initialization of weights
        self.weights = np.ones((dim,1)) / dim
    
    def predict_proba(self, features):
        """
        Exercise 1a: Compute the probability of assigning a class to each feature of an image
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
        Returns:
            prob (np.array): probabilities [N] of N examples
        """
    		# TODO: INSERT
        temp = features @ self.weights
        prob = 1 /(1 + np.exp(-temp))
        return prob
    
    def predict(self, features):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
        Returns:
            pred (np.array): predictions [N] of N examples
        """
        prob = self.predict_proba(features)
        # decision boundary at 0.5
        pred = np.array([ 1.0 if x >= 0.5 else 0.0 for x in prob])[:,np.newaxis]
        return pred
    
    def compute_loss(self, features, labels):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
        Returns:
            loss (scalar): loss of the current model
        """
        examples = len(labels)
        
        '''
        Exercise 1b:    Compute the loss for the features of all input images
                        NOTE: Don't forget to remove the first quit() command in the main program!

        HINT: Use the provided log function to avoid nans with large learning rate
        '''
        prob = self.predict(features)
        loss = -(labels * log(prob) + (1 - labels) * log(1 - prob)) # TODO: REPLACE

        return loss.sum() / examples   #Why not loss.mean() instead of doing this, agreed :')
            
    def score(self, pred, labels):
        """
        Args:
            pred (np.array): predictions [N, 1] of N examples
            labels (np.array): labels [N, 1] of N examples
        Returns:
            score (scalar): accuracy of the predicted labels
        """
        diff = pred - labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))
    
    def update_weights(self, features, labels, lr):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
            lr (scalar): learning rate scales the gradients
        """
        examples = len(labels)
        
        '''
        Exercise 1c:    Compute the gradients given the features of all input images
                        NOTE: Don't forget to remove the second quit() command in the main program!
        '''
        gradient = 0 # TODO: REPLACE
        prob = self.predict(features)
        gradient = features.T @ (prob - labels) 
        
        # update weights
        self.weights -= lr * gradient / examples 
        
    def fit(self, features, labels):
        """
        Args:
            features (np.array): feature matrix [N, D] consisting of N examples with D features
            labels (np.array): labels [N, 1] of N examples
        """
        # gradient descent    
        for i in range(self.max_iter):
            # update weights using the gradients
            self.update_weights(features, labels, self.lr)
            
            # compute loss
            loss = self.compute_loss(features, labels)
            self.losses.append(loss)
            
            # print current loss
            print('Iteration {}\t Loss {}'.format(i, loss))
    

def load_mnist(path, kind='train', each=1):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images[::each, :], labels[::each]

# load fashion mnist
train_img, train_label = load_mnist('.', kind='train', each=1)
test_img, test_label = load_mnist('.', kind='t10k', each=1)
train_img = train_img.astype(np.float)/255.
test_img = test_img.astype(np.float)/255.

# label definition of fashion mnist
labels = { 0: 'T-shirt/top',
           1: 'Trouser',
           2: 'Pullover',
           3: 'Dress',
           4: 'Coat',
           5: 'Sandal',
           6: 'Shirt',
           7: 'Sneaker',
           8: 'Bag',
           9: 'Ankle boot'}

# consider only the classes 'Pullover' and 'Coat'
labels_mask = [2, 4]
train_mask = np.zeros(len(train_label), dtype=bool)
test_mask = np.zeros(len(test_label), dtype=bool)
train_mask[(train_label == labels_mask[0]) | (train_label == labels_mask[1])] = 1
test_mask[(test_label == labels_mask[0]) | (test_label == labels_mask[1])] = 1

# classification of Pullover
train_img = train_img[train_mask,:]
test_img = test_img[test_mask,:]
train_label = np.array([ 1.0 if x == labels_mask[0] else 0.0 for x in train_label[train_mask]])[:,np.newaxis]
test_label = np.array([ 1.0 if x == labels_mask[0] else 0.0 for x in test_label[test_mask]])[:,np.newaxis]


# init logistic regression
logreg = LogisticRegression()
logreg.init_weights(train_img.shape[1])
logreg.lr = 1e-2
logreg.max_iter = 10

accs = []

# testing without training
y_pred = logreg.predict(test_img)
score = logreg.score(y_pred, test_label)
accs.append(score)
print('Accuracy of initial logistic regression classifier on test set: {:.2f}'.format(score))

# quit() ### Exercise 1b: Remove exit ### 
            
# compute initialization loss
loss = logreg.compute_loss(train_img, train_label)
print('Initialization loss {}'.format(loss))

# quit() ### Exercise 1c: Remove exit ###

'''
Exercise 1d: Plot the cross entropy loss for t=0 and t=1
'''
#TODO: Insert
# x = np.linspace(0,1,100)
# fig, axes = plt.subplots(1,2)
# axes[0].plot(x, -log(x))
# axes[1].plot(x, -log(1-x))
# axes[0].set_title('Loss t=1')
# axes[1].set_title('Loss t=0')
# plt.show()
        
# compute test error after max_iter

for i in range(0,100):
    # training
    logreg.fit(train_img, train_label)
    
    # testing
    y_pred = logreg.predict(test_img)
    score = logreg.score(y_pred, test_label)
    accs.append(score)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(score))
exit() 
'''
Exercise 1e: Plot the learning curves (losses and accs) using different learning rates (1e-4,1e-3,1e-2,1e-1,1e-0)
'''
# losses = logreg.losses
# plot_losses, plot_accs = [], []
# lrs = [1e-4,1e-3,1e-2,1e-1,1e-0]

# fig, axes = plt.subplots(1,2)
# out_iter = 100
# logreg.max_iter = 10
# for lr in lrs:
#     logreg.init_weights(train_img.shape[1])
#     logreg.lr = lr
#     logreg.losses, plot_accs = [], []

#     for i in range(out_iter):
#         logreg.fit(train_img, train_label)

#         y_pred = logreg.predict(test_img)
#         score = logreg.score(y_pred, test_label)
#         plot_accs.append(score)
    
#     axes[0].plot(range(out_iter*logreg.max_iter), logreg.losses, c='orange')
#     axes[1].plot(range(out_iter), plot_accs, c='blue')
#     axes[0].set_title(f'lr: {lr} training loss')
#     axes[1].set_title(f'lr: {lr} test accuracy')
#     plt.savefig(f'lr{lr}.png')
#     axes[0].clear()
#     axes[1].clear()
'''
Exercise 1f: Plot the optimized weights and weights.*img (.* denotes element-wise multiplication)
'''
# logreg = LogisticRegression()
logreg.init_weights(train_img.shape[1])
logreg.lr = 1e-2
logreg.losses = []
logreg.max_iter = 100
logreg.fit(train_img, train_label)

mask = (test_label==0).squeeze()
n0 = mask.sum()
imgs0 = test_img[mask,:]
imgs1 = test_img[~mask,:]

inds0 = np.random.choice(n0, 5, replace=False)
inds1 = np.random.choice(test_img.shape[0]-n0, 5, replace=False)
imgs0 = imgs0[inds0]
imgs1 = imgs1[inds1]

imgs0 *= logreg.weights.T 
imgs1 *= logreg.weights.T

plt.title('Weights')
plt.imshow(logreg.weights.reshape(28,28))
plt.axis('off')
# plt.show()
plt.savefig('weights.png')

imgs0 = imgs0.reshape(5,28,28)
imgs1 = imgs1.reshape(5,28,28)

fig, axes = plt.subplots(2,5)
fig.suptitle('Activations')
for i in range(len(imgs0)):
    axes[0,i].imshow(imgs0[i])
    axes[1,i].imshow(imgs1[i])

    axes[0,i].axis('off')
    axes[1,i].axis('off')
axes[0,2].set_title('Coats')
axes[1,2].set_title('Pullovers')
# plt.show()
plt.savefig('activations.png')