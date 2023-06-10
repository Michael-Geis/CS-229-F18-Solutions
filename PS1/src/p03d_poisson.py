import numpy as np
from src import util

from src.linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    
    # Train the model
    model = PoissonRegression()
    model.fit(x_train,y_train)
        
    # Load the eval set and make prediction
    x_eval , y_eval = util.load_dataset(eval_path,add_intercept=False)
    y_predict = model.predict(x_eval)

    # Save the predicted labels to the specified path
    np.savetxt(y_predict, fname=pred_path,delimiter=',', header='predicted labels for ds4_valid')

    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self,x,y):
        """Parameter fitting for Poisson Regression.

        Args:
            x ((m,n) shape array)): Training inputs.
            y ((m,) shape array)): Training labels.
            lr (floating point): The learning rate for gradient ascent.
        """
        m , n = x.shape
        lr = self.step_size

        def h(x,theta):
            return np.exp(x @ theta)
        
        def gradient(x,y,theta):
            return ((y - h(x,theta)).reshape(-1,1) * x).mean(axis=0)

        def update(theta,lr):
            return theta + lr * gradient(x,y,theta)
        
        ## Initialize iteration counter and theta parameter.
        j=0
        theta = np.zeros(n)
        lr = self.step_size

        ## Loop gradient ascent.
        while True:
            theta_prev = theta
            theta = update(theta,lr)
            j+=1
            if np.linalg.norm(theta - theta_prev) <= self.eps:
                print(f'Gradient ascent converged after {j} iterations.')
                self.theta = theta
                break
            elif j >= self.max_iter:
                print(f'Gradient ascent has failed to converge after {self.max_iter} iterations. Stopping.')
                self.theta = None
                break

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        if self.theta is None:
            raise Exception('The model has not been trained yet!')
        return np.exp(util.add_intercept(x) @ self.theta )
