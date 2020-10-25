import numpy as np
import utils

OUT_PATH = 'out/pca'

def main():
    """The main function of this script."""
    # HINT: you can used numpys advanced indexing feature to
    # select an appropriate subset of x_train / x_test
    x_train, y_train = utils.load_fashion_mnist('train')
    x_test, y_test = utils.load_fashion_mnist('test')

    y_trains = y_train.numpy()
    idx = np.argwhere(y_trains == 7)[:,0]
    x_train = x_train.numpy()
    x_train = x_train[idx,:]
    mean, s, V = compute_pca(x_train)

    #plot pca reconstruction
    for i in range(0, 5):
        pc = mean + s[i] * V[i]
        utils.plot_principal_components(pc, "./" + str(i))


    analyze_variance(s)
    create_reconstructions(mean, V, x_test.numpy())
    create_samples(mean, V, s)

def compute_pca(x):
    """Compute PCA of x and plot mean as well as first 5 principal
    components.

    Args:
        x (np.array):snumpy array containing N data points of dimensionality
            K as inpusPCA.

    Returns:
        mu (np.array)sn of x
        s (np.array):sular values of the covariance matrix of x
        V (np.array):sarray containing all K principal components of x as
            row vectos
    """
    #PCA with SVD after subtracting the mean. Don't need covariance matrix now :)
    mu = np.mean(x, axis=0, keepdims=True)
    utils.plot_samples(mu, "./")
    centered = x - mu
    #cov  = np.cov(centered)
    _, s, V = np.linalg.svd(centered, full_matrices=False)

    return mu, s, V


def analyze_variance(s):
    """Analyze the variance of the singular values of the PCA

    Args:
        s (np.array): singular values
    """
    norm_s = s / np.sum(s)
    cum = np.cumsum(norm_s)
    idx_99 = np.argwhere(cum > 0.99)[0]
    idx_95 = np.argwhere(cum > 0.95)[0]
    idx_90 = np.argwhere(cum > 0.90)[0]
    idx_50 = np.argwhere(cum > 0.50)[0]
    print(idx_99, idx_95, idx_90, idx_50)
    utils.plot_cummulative_distribution(cum, "./")
    pass

def create_reconstructions(mean, V, x, ncomp=5, nplots=5):
    """Apply PCA to test data, print mean squared  error and plot first
    `nplots` reconstructions.

    Args:
        mean (np.array): mean of PCA
        V (np.array): array containing principal components as row vectors
        x (np.array): test data
        ncomp (int, optional): number of principal components to use
        nplots (int, optional): number of principal components to plot
    """
    x_cent = x - mean
    mult = x_cent @ V.T
    res = np.zeros(mult.shape)
    res[:,:ncomp] = mult[:,:ncomp]
    reconst = res @ V + mean

    mse = np.mean((x - reconst)**2)
    print("MSE reconstruction error:", mse)
    utils.plot_samples(reconst[:5], "./reconst")
    print("Compression ration: ", ncomp / x.shape[1])
    return reconst

def create_samples(mean, V, s, nsamples=5):
    """Use PCA to sample synthetic data points and plot them

    Args:
        mean (np.array): mean of PCA
        V (np.array): array containing principal components
        nsamples (int, optional): number of samples to draw
    """
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5.1d)
    # 1. sample a normally distributed random
    #    vector of size s.size
    # 2. multiply  this vector with np.sqrt(s)
    # 3. multiply this vector from the left with V.T
    #    to project it back to image space and add
    #    the mean vector
    # 4. append sample to x_rnd
    #
    # -------------------------------------
    # List of samples
    x_rnd = []
    # Loop over the number of samples to draw
    for i in range(nsamples):
        mul = np.random.multivariate_normal(np.zeros(s.size), np.eye(s.size)) * np.sqrt(s)
        m = V.T @ mul
        mplusmean = m + mean
        x_rnd.append(mplusmean)
    utils.plot_samples(x_rnd, OUT_PATH)


if __name__ == '__main__':
    main()
