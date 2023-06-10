import matplotlib.pyplot as plt
import numpy as np
from src import util

from src.linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data

class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y



    def predict(self, Z):
        """Make predictions given inputs Z.

        Args:
            Z: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        x = self.x
        y = self.y
        tau = self.tau
        k , n = Z.shape
        
        def W(x,z,tau):
            """The matrix of Gaussian weights of width tau for a single input z with respect to the training set x.

            Args:
                x ((m,n) shape array)): Training inputs
                z ((n,) shape array): A new input
                tau (floating point): Dispersion parameter for the Gaussian weight

            Returns:
                ((m,m) shape array): Diagonal matrix of weights.
            """

            norms =  np.diag( (x - z) @ (x - z).T ) 
            return np.diag( np.exp( (-1/2 * np.power(tau,-2) * norms )))

        def theta(x,y,W):
                return (np.linalg.inv(x.T @ W @ x)) @ (x.T @ W @ y.reshape(-1,1))
        
        def W_row(z):
            return W(x,z,tau)

        wts = np.apply_along_axis(arr=Z, func1d=W_row, axis=1)
        
        params = theta(x,y,wts).reshape(k,n)
        return np.diag(params @ Z.T)
       
### Testing code below

x_train , y_train = util.load_dataset('./data/ds5_train.csv', add_intercept=True)
x_test , y_test = util.load_dataset('./data/ds5_test.csv', add_intercept=True)

model = LocallyWeightedLinearRegression(tau=0.5)
model.fit(x_train,y_train)
y_predicts = model.predict(x_test)