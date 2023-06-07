import numpy as np

class LogisticRegression:
    
    def __init__(self):
        self.theta = None
    
    def fit(self,X,y):
        
        m , n = np.shape(X)

        def h(X,theta):

            return np.reciprocal(1 + np.exp(- X @ theta))


        def gradient(X,y,theta):

            return 1/m * (h(X,theta) - y).T @ X
        
        def hessian(X,theta):

            return 1/m * np.einsum('i,ijik->jk' , h(X,theta)*(1-h(X,theta)) , np.tensordot(X,X,axes=0) )
    
        def update_theta(theta):
            
            return theta - np.linalg.inv(hessian(X,theta)) @ gradient(theta, X, y)
    
        self.theta = np.zeros(n)
        old_theta = self.theta
        new_theta = update_theta(self.theta)
        error = np.linalg.norm(new_theta - self.theta)

        while error > 10 ** (-5):
            old_theta = new_theta
            new_theta = update_theta(old_theta)
            error = np.linalg.norm(new_theta - old_theta)

        self.theta = new_theta






    def predict(self,X):
        
        return (X @ self.theta > 0).astype(np.int8)

