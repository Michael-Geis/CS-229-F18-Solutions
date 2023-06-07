import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
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
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def stochastic_fit(self, x, y,lr):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m , n = x.shape

        def h(x,theta):
            """The expected value of y given x and theta for Poisson regression.
            
            Args:
                x: (m,n) shape array. Rows are training inputs.
                theta: (n,) shape array. Vector of parameters to be fit.
            
            Returns:
                (m,) shape array. Entry j is h(x^j,theta)
            """

            return np.exp(x @ theta)
        
        def gradient(x,y,theta,j):
            """Gradient of the log likelihood with respect to the training example j.

            Args:
                x ((m,n) shape array): Array of training examples.
                theta ((n,) shape array): Vector of parameters.
                j (int): integer between 1 and m specifying a row of the array x.

            Returns:
                (n,) shape array representation of the gradient.
            """
            return (y[j] - h(x,theta)[j]) * x[j,:]
        
        def update_params(theta,j):
            """Updates the value of theta according to stochastic gradient ascent using training example j.

            Args:
                theta ((n,) shape array): Parameters to be updated.
                j (int): integer between 1 and m specifying which training example to use in the update.
            """
            return theta + lr * gradient(x,y,theta,j)
        
        theta_old = np.zeros(n)
        theta_new = update_params(theta_old,1)
        
        j=1

        while np.linalg.norm(theta_new - theta_old) > 10 ** (-3):
            if j < m-1:
                j+=1
            else:
                j = 1
            theta_old = theta_new
            theta_new = update_params(theta_old,j)
        
        self.theta = theta_new

    def fit(self,x,y,lr):
        """Parameter fitting for Poisson Regression.

        Args:
            x ((m,n) shape array)): Training inputs.
            y ((m,) shape array)): Training labels.
            lr (floating point): The learning rate for gradient ascent.
        """
        m , n = x.shape

        def h(x,theta):
            return np.exp(x @ theta)
        
        def gradient(x,y,theta):
            return np.mean(np.reshape(y - h(x,theta),(-1,1)) * x , axis=0)

        def update(theta):
            return theta + lr * gradient(x,y,theta)
        
        theta_old = np.zeros(n)
        theta_new = update(theta_old)

        while np.linalg.norm(theta_new - theta_old) > self.eps:
            theta_old = theta_new
            theta_new = update(theta_old)
        
        self.theta = theta_new

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        if self.theta is None:
            raise('The model has not been trained yet!')
        return np.exp(util.add_intercept(x) @ self.theta )
