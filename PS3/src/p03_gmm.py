import matplotlib.pyplot as plt
import numpy as np
import os
import math

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 1  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)

def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))
    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    m , d = x.shape                                     # Number of examples = m, dimension of data = d
    
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    
    X = split_inputs(x,K)

    mu = np.zeros((K,d))
    sigma = np.zeros((K,d,d))

    for i in range(K):
        mu[i] = np.mean(X[i], axis=0)
        sigma[i] = np.cov(X[i].T) 

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    
    phi = 1/K * np.ones(K)                              # Equal weight on each Gaussian

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)

    w = 1/K * np.ones((m,K))                            # Equal weights on each Gaussian.


    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)
    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    (m, d) , (_ , K) = x.shape , w.shape

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        prev_ll = ll

        # (1) E-step: Update your estimates in w

        w = update_w(x,phi=phi,mu=mu,sigma=sigma)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        
        W = w * (1/np.sum(w,axis=0)).reshape(1,-1)                      # Normalized weight matrix. Columns sum to 1.

        phi = np.mean(w, axis = 0)                                      # Updated phi, shape (K,)
        mu = W.T @ x                                                    # Updated mu, shape (K,d)

        outers = outer_prods(x,mu)                                      # See helper function `outer_prods`
        sigma = (W.reshape(m,K,1,1) * outers).sum(axis=0)               # Updated sigma       

        # (3) Compute the log-likelihood of the data to check for convergence.

        ll = log_like_unsup(x=x,phi=phi,sigma=sigma,mu=mu)

        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
            

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        prev_ll = ll
        # (1) E-step: Update your estimates in w

        w = update_w(x=x,phi=phi,mu=mu,sigma=sigma)
        

        # (2) M-step: Update the model parameters phi, mu, and sigma

        (m , d) , (K, _) = x.shape, mu.shape

        lbl_cts = np.zeros(K)
        for i in range(K):
            lbl_cts[i] = np.count_nonzero(z==i)

        m_tilde = int(lbl_cts.sum())
        
        ## Update phi

        phi = 1 / (m + alpha * m_tilde) * (w.sum(axis=0) + alpha*lbl_cts)

    ## Update mu

        ## (m,K) shape where entry ij is 1 iff z^{(i)} = j. Sharp weight matrix for labeled examples.

        w_tilde = ((z - np.arange(K).reshape(1,-1)) == 0).astype(np.int8)
        mu = 1 / (w.sum(axis = 0) + alpha * lbl_cts).reshape(-1,1) * (w.T @ x + alpha * (w_tilde).T @ x_tilde)

        # Update sigma
        outers = outer_prods(x,mu)
        outers_tilde = outer_prods(x_tilde,mu)
        weighted_outer_sum = (w.reshape((m,K,1,1)) * outers).sum(axis=0) + alpha * (w_tilde.reshape((m_tilde,K,1,1)) * outers_tilde).sum(axis=0)
        sigma = (1 / (w.sum(axis = 0) + alpha * lbl_cts)).reshape(K,1,1) * weighted_outer_sum

        # (3) Compute the log-likelihood of the data to check for convergence.
        z_ind = z.astype(int)
        ll = log_like_unsup(x,phi,mu,sigma) + alpha*log_like_sup(x_tilde,z_ind,phi,mu,sigma)

        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


def split_inputs(x,K):
    """Takes in a matrix of training inputs and partitions x into K equal sized subarrays consisting of rows chosen uniformly at random.

    Args:
        x: (m,d) shape array of training inputs
        K: a divisor of m

    Returns:
        (K, m/K, d) shape array whose ith matrix is a randomly chosen slice of m/K rows of x.
    """
    m , _ = x.shape
    X = x.copy()
    splits = np.split(np.arange(0,m),K) # Create a list of K blocks of indicies from 1 to 1000
    np.random.shuffle(X)                # Randomly shuffle around the rows of x

    return np.array([X[splits[i]] for i in range(K)])


def outer_prods(x,mu):
    """Returns a 4 tensor whose ijkl entry is [(x^{i} - \mu^{j})(x^{i} - \mu^j)^T]_{kl}

    Args:
        x: (m,d) shape with rows x^{i}
        mu: (K,d) shape array with rows mu^{i}

    Returns:
        (m,K,d,d) shape array where the last two dimensions is the outer square of x^i - mu^j.
    """
    (m , d) , (K , _) = x.shape , mu.shape
    diffs = x.reshape((m,1,d)) - mu.reshape((1,K,d))                # (m,K,d) array - ij* entry is the d-vector x^{(i)} - \mu_{j}
    outers = diffs.reshape(m,K,d,1) * diffs.reshape(m,K,1,d)        # 4 tensor - ij** entry is the outer square of x^(i) - \mu_j
    return outers

def log_probs(x,mu,sigma):
    """Calculates the probabilities $\log p(x^{(i)} | z^{(i)} = j; theta)$.

    Args:
        x: (m,d) shape array of training inputs
        phi: (K,) shape vector of probabilities phi_j = p(z^{(i)} = j)
        sigma: (K,d,d) shape vector of covariance matrices of the distinct Gaussians
        mu: (K,d) vector of means of the distinct Gaussians

    Returns:
        (m,K) array whose ij entry is $\log p(x^{(i)} | z^{(i)} = j; theta)$.
    """
    (m , d) , (K , _) = x.shape , mu.shape

    sigma_inv = np.linalg.inv(sigma)                                                      # K vector matrix inverses
    inner_prods = np.sum(sigma_inv.reshape((1,K,d,d)) * outer_prods(x,mu), axis=(2,3))    # Inner products in the argument of the exp

    return -1/2 * (np.log(2*math.pi) + np.log(np.linalg.det(sigma)).reshape(1,-1) + inner_prods)


def update_w(x,phi,mu,sigma):
    """Returns the updated weight matrix for both SS and US GMM.

    Args:
        x: (m,d) shape array of training examples
        phi: (K,) vector of probabilities 
        mu: (K,d) shape array of means
        sigma: (K,d,d) shape array of covariance matrices

    Returns:
        The new weight matrix calculated from x and these parameters.
    """

    P = np.exp(log_probs(x=x,mu=mu,sigma=sigma))                    # (m,K) array - P_{ij} = P(x^(i) | z^(i) = j ; params)
    l_x = P @ phi                                                   # (K,) array - (l_x)_i = P(x^(i) ; params)
    w = P * phi.reshape(1,-1) * (1/l_x).reshape(-1,1)               # Vector implementation of Bayes' rule

    return w
    

def log_like_unsup(x,phi,mu,sigma):

    P = log_probs(x,mu,sigma)
    return np.log( np.exp(P) @ phi).sum()

def log_like_sup(x_tilde,z,phi,mu,sigma):
    m_tilde , _ = x_tilde.shape
    square_phi = np.ones((m_tilde,1)) * phi.reshape(1,-1) 
    phi_tilde = square_phi[range(m_tilde),z] ## Correctly spits out the vector whose k component is phi_{z^{(k)}}

    return np.sum( log_probs(x_tilde,mu,sigma)[range(m_tilde),z] + phi_tilde)


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    path = 'test_{}.pdf'.format(plot_id)
    plt.savefig(fname=save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***

