import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
Complete the code sections marked with TODO

You are not allowed to use any higher-level methods or classes that are not already provided

all variables should use tf.float32 as dtype and use the provided initializer

about Layers:
- build(shape)  is called exactly once just before the first inputs are given to the layer
                since you get input shape info here, it's suitable to initialize variables
- call(inputs)  the forward pass
"""

tf.random.set_seed(0)
initializer = tf.initializers.TruncatedNormal()


class ReLULayer(tf.keras.layers.Layer):
    """ classic ReLU function for non-linearity """

    def call(self, inputs):
        """
        :param inputs: outputs from the layer before
        :return: ReLu(inputs)
        """
        return tf.math.maximum(0., inputs)


class SoftMaxLayer(tf.keras.layers.Layer):
    """ SoftMax (or SoftArgMax) function to transform logits into probabilities """

    def call(self, inputs):
        """
        :param inputs: outputs from the layer before
        :return: SoftMax(inputs)
        """
        max_val = tf.reduce_max(inputs, axis=1, keepdims=True)
        outputs = inputs - max_val
        outputs = tf.exp(outputs)
        sum = tf.reduce_sum(outputs, axis=1, keepdims=True)
        return tf.divide(outputs,sum)

class DenseLayer(tf.keras.layers.Layer):
    """ a fully connected layer """

    def __init__(self, num_neurons: int, use_bias=True):
        """
        :param num_neurons: number of output neurons
        :param use_bias: whether to use a bias term or not
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.bias = use_bias
        self.w = self.b = None

    def build(self, input_shape):
        self.w = tf.Variable(initializer(shape=[input_shape[-1],
            self.num_neurons], dtype=tf.float32), trainable=True, dtype=tf.float32)

        if self.bias:
            self.b = tf.Variable(initializer(shape=[1,self.num_neurons], dtype=tf.float32),
                    trainable=True, dtype=tf.float32)

    def call(self, inputs):
        mul = tf.matmul(inputs, self.w)
        if self.bias:
            mul += self.b
        return mul


class SequentialModel(tf.keras.layers.Layer):
    """ a sequential model containing other layers """

    def __init__(self, num_neurons: [int], use_bias=True):
        """
        :param num_neurons: number of output neurons for each DenseLayer.
        :param use_bias: whether a use_bias terms should be used or not
        """
        super().__init__()
        self.modules = []
        for i in range(len(num_neurons) - 1):
            self.modules.append(DenseLayer(num_neurons=num_neurons[i],
                use_bias=use_bias))
            self.modules.append(ReLULayer())
        
        self.modules.append(DenseLayer(num_neurons=num_neurons[-1],
            use_bias=use_bias))
        self.modules.append(SoftMaxLayer())

    def call(self, inputs):
        x = inputs
        
        for module in self.modules:
            x = module(x)

        return x


def test_model(model, test_set):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for batch_input, batch_target in test_set:
        output = model(batch_input)
        test_accuracy.update_state(batch_target, output)

    return test_accuracy.result()


def train_model(model, train_set, eval_set, loss, learning_rate, epochs) -> (list, list):
    """
    :param model: a sequential model defining the network
    :param train_set: a tf.data.Dataset providing the training data
    :param eval_set: a tf.data.Dataset providing the evaluation data
    :param loss: a tensor defining the kind of lost used
    :param learning_rate: learning rate (step size) for stochastic gradient descent
    :param epochs: num epochs to train
    :return: list of evaluation accuracies and list of train accuracies
    """
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    train_accuracy_per_epoch = []
    eval_accuracy_per_epoch = []


    # some train loop
    for e in range(epochs):
        train_accuracy.reset_states()
        for batch_input, batch_target in train_set:
            with tf.GradientTape() as t:
                batch_output = model(batch_input)
                batch_loss = loss(batch_target, batch_output)
            
            weights = []
            for layer in model.modules:
                if type(layer) is DenseLayer:
                    weights += [layer.w, layer.b]

            grads = t.gradient(batch_loss, weights)
            optimizer.apply_gradients(zip(grads, weights))

            train_accuracy.update_state(batch_target, batch_output) 

        train_accuracy_per_epoch.append(train_accuracy.result())
        eval_accuracy_per_epoch.append(test_model(model, eval_set))
        tf.print("epoch: ", e, "\t train accuracy: ", train_accuracy_per_epoch[-1], "\t eval accuracy: ",
                 eval_accuracy_per_epoch[-1])

    return eval_accuracy_per_epoch, train_accuracy_per_epoch


def single_training(train_set, eval_set, test_set, epochs, loss, learning_rate):
    """
    :param train_set: a tf.data.Dataset providing the training data
    :param eval_set: a tf.data.Dataset providing the evaluation data
    :param epochs: num epochs to train
    :param loss: a tensor defining the kind of lost used
    :param learning_rate: learning rate (step size) for stochastic gradient descent
    :return: list of evaluation accuracies and list of train accuracies
    """
    model = SequentialModel([12, 12, 12, 10], use_bias=True)
    eval_accuracies, train_accuracies = train_model(model, train_set, eval_set, loss, learning_rate, epochs)
    print('Train accuracy per epoch: %s' % ', '.join(['%.3f' % a for a in train_accuracies]))
    print('Evaluation accuracy per epoch: %s' % ', '.join(['%.3f' % a for a in eval_accuracies]))
    print('Test  accuracy: %.3f' % test_model(model, test_set))


def grid_training(train_set, eval_set, test_set, epochs, loss, learning_rate: float, depths: [int], widths: [int]):
    """
    :param train_set: a tf.data.Dataset providing the training data
    :param eval_set: a tf.data.Dataset providing the evaluation data
    :param epochs: num epochs to train
    :param loss: a tensor defining the kind of lost used
    :param learning_rate: learning rate (step size) for stochastic gradient descent
    :param depths: a list of depths to perform the grid search on
    :param widths: a list of widths to perform the grid search on
    :return: list of evaluation accuracies and list of train accuracies
    """
    z = np.zeros([len(depths), len(widths)])

    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            num_neurons = [w] * d + [10]
            model = SequentialModel(num_neurons, use_bias=True)
            train_model(model, train_set, eval_set, loss, learning_rate, epochs)
            z[i, j] = test_model(model, test_set)
            print('finished for d=%d, w=%d, acc=%.3f' % (d, w, z[i, j]))

    plt.title('Grid search')
    ax = plt.gca()
    im = ax.imshow(z, cmap='Wistia')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(widths)))
    ax.set_yticks(np.arange(len(depths)))
    ax.set_xticklabels(['w=%d' % w for w in widths])
    ax.set_yticklabels(['d=%d' % d for d in depths])
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xticks(np.arange(len(widths) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(depths) + 1) - .5, minor=True)
    for i in range(len(depths)):
        for j in range(len(widths)):
            im.axes.text(j, i, '%.3f' % z[i, j], None)
    plt.show()
    #use plt.savefig('grid.png') when working on the cluster

def grid_training2(train_set, eval_set, test_set, epochs, loss, learning_rates:
        [float], depth: int, widths: [int]):
    """
    :param train_set: a tf.data.Dataset providing the training data
    :param eval_set: a tf.data.Dataset providing the evaluation data
    :param epochs: num epochs to train
    :param loss: a tensor defining the kind of lost used
    :param learning_rates: a list of learning rates (step sizes) for SGD to
    perform a grid search on
    :param depth: depth of a network 
    :param widths: a list of widths to perform the grid search on
    :return: list of evaluation accuracies and list of train accuracies
    """
    z = np.zeros([len(learning_rates), len(widths)])

    for i, lr in enumerate(learning_rates):
        for j, w in enumerate(widths):
            num_neurons = [w] * depth + [10]
            model = SequentialModel(num_neurons, use_bias=True)
            train_model(model, train_set, eval_set, loss, lr, epochs)
            z[i, j] = test_model(model, test_set)
            print('finished for lr=%d, w=%d, acc=%.3f' % (lr, w, z[i, j]))

    plt.title('Grid search')
    ax = plt.gca()
    im = ax.imshow(z, cmap='Wistia')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(widths)))
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_xticklabels(['w=%d' % w for w in widths])
    ax.set_yticklabels(['lr=%f' % lr for lr in learning_rates])
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xticks(np.arange(len(widths) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(learning_rates) + 1) - .5, minor=True)
    for i in range(len(learning_rates)):
        for j in range(len(widths)):
            im.axes.text(j, i, '%.3f' % z[i, j], None)
    plt.show()
 


def normalize_dataset(x_train, x_eval, x_test):
    """
    Subtracts mean for each pixel and divides it by the standard deviation.
    Mean and standard deviation for each pixel are estimated over the training data set.
    If the standard deviation is 0 we divide by 1.
    :param x_train: numpy ndarray containing the training data
    :param x_eval:  numpy ndarray containing the eval data
    :param x_test:  numpy ndarray containing the test data
    :return: the normalized datasets
    """

    print(f'x_train: {x_train.shape}')
    x_train_means = np.average(x_train, axis=0)
    x_train_stds = np.std(x_train, axis=0)
    np.place(x_train_stds, x_train_stds == 0, 1)
    norm_x_train = (x_train - x_train_means) / x_train_stds 

    x_eval_means = np.average(x_eval, axis=0)
    x_eval_stds = np.std(x_eval, axis=0)
    np.place(x_eval_stds, x_eval_stds == 0, 1)
    norm_x_eval = (x_eval - x_eval_means) / x_eval_stds 

    x_test_means = np.average(x_test, axis=0)
    x_test_stds = np.std(x_test, axis=0)
    np.place(x_test_stds, x_test_stds == 0, 1)
    norm_x_test = (x_test - x_test_means) / x_test_stds 
    
    # print(f'means: {x_train_means.shape}\nstds: {x_train_stds.shape}')
    return norm_x_train, norm_x_eval, norm_x_test


def main():
    # Loading the MNIST Dataset and creating tf Datasets
    batch_size = 100
    evaluation_set_size = 10000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[evaluation_set_size:].reshape(50000, 784).astype('float32')
    y_train = y_train[evaluation_set_size:]
    x_eval = x_train[:evaluation_set_size].reshape(10000, 784).astype('float32')
    y_eval = y_train[:evaluation_set_size]
    x_test = x_test[:].reshape(10000, 784).astype('float32')


    # Normalize the data sets
    # x_train, x_eval, x_test = normalize_dataset(x_train,x_eval,x_test)

    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_set = train_set.shuffle(1024).batch(batch_size)
    eval_set = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
    eval_set = eval_set.batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_set = test_set.batch(batch_size)

    # Instantiate a logistic loss function that expects probabilities.
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    single_training(train_set, eval_set, test_set, epochs=5, loss=loss,
            learning_rate=0.01)
    # grid_training(train_set,eval_set, test_set, epochs=3, loss=loss, learning_rate=0.1, depths=[0, 1, 2], widths=[12, 24])
    # grid_training2(train_set, eval_set, test_set, epochs=3, loss=loss,
            # learning_rates=[1, 0.1, 0.001], depth=1, widths=[12, 24])
   


if __name__ == '__main__':
    main()
