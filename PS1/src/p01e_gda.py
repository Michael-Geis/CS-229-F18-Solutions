import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)



class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m , n = np.shape(x)

        def phi(x,y):
            '''returns Bernoulli parameter phi that models the distribution of the labels y^i.

            :param X: array of shape (m,n). ith row should be ith data point x^i.
            :param y: array of shape (m,). ith entry should be the label y^i of data point x^i.   
            '''
            return np.sum(y) / m

        def mu(x,y):
            '''Returns the conditional mean of X given y in the GDA model.
            Args:
                :param x: array of shape (m,n).
                :param y: array of shape (m,).

            Returns:
                array of shape (2,n) row i is the empirical conditional mean of x given
            '''
            mu = np.zeros((2,n))
            mu[0,:] = np.mean(x[y==0,:], axis=0)
            mu[1,:] = np.mean(x[y==1,:], axis=0)
            return mu
        
        def sigma(x,y):
            '''returns the covariance matrix for the distributions of x given y.
    
            :param X: array of shape (m,n).
            :param y: array of shape (m,).
            '''
            # create an array where row i is the mean mu_j where j matches the label of x^i.
            
            mean = np.tile(mu(x,y)[0,:], (m,1)) * (1-y).reshape(-1,1) + np.tile(mu(x,y)[1,:], (m,1)) * y.reshape(-1,1)

            # create an array where row i is x^i minus its label empirical mean.
            centered = x - mean
            outer_along_rows = np.einsum('ij,ik->jk', centered , centered)
            return 1/m * outer_along_rows

        def calculate_logistic_fit(phi,mu,sigma):
            sig_inv = np.linalg.inv(sigma)
            theta = np.zeros(n+1)

            theta[0] = 1/2 * (np.dot(sig_inv @ mu[0,:] , mu[0,:]) - np.dot(sig_inv @ mu[1,:] , mu[1,:])) + np.log(phi/(1-phi))
            theta[1:] = sig_inv @ (mu[1,:] - mu[0,:])   
            
            return theta
        
        self.theta = calculate_logistic_fit(phi(x,y),mu(x,y),sigma(x,y))

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,) array of predicted labels.
        """
        if self.theta is None:
            raise Exception('Model has not been trained!')
        return (util.add_intercept(x) @ self.theta >= 0).astype(np.int8)
        

