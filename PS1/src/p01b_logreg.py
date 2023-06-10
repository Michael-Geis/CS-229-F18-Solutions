import numpy as np
from src import util

from src.linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    logreg_model = LogisticRegression()
    logreg_model.fit(x_train,y_train)
    
    x_valid , _ = util.load_dataset(eval_path, add_intercept=True)
    y_predicted = logreg_model.predict(x_valid)
    np.savetxt(pred_path, y_predicted , delimiter=',')


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit(self, x, y, display_loop=False):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        x = util.add_intercept(x)
        m , n = x.shape

        def h(x,theta):
            '''Vectorized implementation of the conditional expectation of y = 1 given x and theta.
            Args:
                x: (m,n) shape array containing the training data as rows. Warning: x should already have an intercept added.
                theta: (n,) shape array.
            
            Returns
                (m,) shape array.
            '''
            return  np.reciprocal(1 + np.exp(-x @ theta))
        
        def gradient(x,y,theta):
            '''Vectorized implementation of the gradient of the normalized log likelihood for logistic regression.
            Args:
                x: (m,n) shape array containing the training data as rows
                y: (m,) shape array containing the labels for the training data
                theta: (n,) shape array.
            
            Returns:
                (n,) shape array.
            '''
            
            return 1/m * (h(x,theta) - y).T @ x
        
        def hessian(x,theta):
            '''Vectorized implementation of the hessian of the normalized log likelihood for logistic regression.
            Args:
                x: (m,n) shape array containing the training data as rows
                theta: (n,) shape array.
            
            Returns:
                (n,n) shape array 
            '''
            coeffs = h(x,theta) * (1 - h(x,theta))
            x_outer = np.multiply.outer(x,x)
            
            return (1/m) * np.einsum('i,ijik->jk' , coeffs , x_outer)
        
        def update(theta):
             '''Runs a single iteration of newton's method with the input theta on the gradient of the log likelihood.
             Args:
                theta: (n,) shape array.
            
            Returns:
                array of shape(n,).
             '''
             return theta - np.linalg.inv(hessian(x,theta)) @ gradient(x,y,theta)

        # initalize loop
        theta_old = np.zeros(n)
        theta_new = update(theta_old)      
        j=0  

        while np.linalg.norm(theta_new - theta_old) > self.eps:
            theta_old = theta_new
            theta_new = update(theta_old)
            j+=1
        if display_loop:
            print('looped {} times'.format(j))

        self.theta = theta_new

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        if self.theta is None:
            raise Exception('Model has not been trained!')
        return (util.add_intercept(x) @ self.theta >= 0).astype(np.int8)
        