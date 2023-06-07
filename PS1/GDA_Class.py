import numpy as np

class GDA:
    """Gaussian discriminant analysis for classification."""


    def __init__(self):
        self.phi = None
        self.sigma = None
        self.mu = None
        self.probs = None
    
    def center(X,y,mu):
        """Returns a centered version of X around the empirical means mu.
        
        Args:

        Output:       
        
        """

        m , n = np.shape(X)

        return X - np.tile(mu[0,:] , (m,1)) * np.tile(1-y , (n,1)).T - np.tile(mu[1,:] , (m,1)) * np.tile(y, (n,1)).T

    
    def fit(self,X,y):
        """Sets the optimal GDA parameters for the inputs X and labels y."""

        m , n = np.shape(X)


        def phi(X,y):

            return np.sum(y) / m
        
        def mu(X,y):
            """Returns an array of shape (2,n) whose rows are the conditional means of X given y = 0 and y = 1."""

            labels = np.array([1-y,y]).T
            return np.einsum('ij,ik->jk', labels, X)

        def sigma(X,y):
            """Returns an array of shape (n,n) which is the Covariance matrix of the conditional distributions of x given y."""

            X_centered = self.center(X,y,mu(X,y))

            return (1/m) * np.einsum('ij,ik->jk', X_centered, X_centered)
        

        self.phi = phi(X,y)
        self.mu = mu(X,y)
        self.sigma = sigma(X,y)

    def predict(self,X):
        """Predicts labels for inputs X using GDA."""

        m , n = np.shape(X)

        def Q(X,j):
            """(m,) size array whose ith entry is Q(x^i) where C(\Sigma)\exp ( -1/2 Q(x)) is the cond. distribution p(x|y=j) and x^i is the ith row of X."""

            X_centered = X - np.tile(self.mu[j,:], (m,1))

            return np.einsum('jk,ijik->i', self.sigma, np.tensordot(X_centered,X_centered, axes=0))

        def p(X,j):
            """Returns an (m,) array whose ith entry is value of the probability density of x given y = j evaluated at x^i, row i of X."""

            return (( (2 * np.pi) ** n) * np.linalg.det(self.sigma)) ** (-1/2) * np.exp(-(1/2) * Q(X,j))

        
        self.probs = p(X,1) * self.phi / (p(X,0)*(1-self.phi) + p(X,1)*(self.phi))


        return (self.probs > 0.5).astype(np.int8)
    
