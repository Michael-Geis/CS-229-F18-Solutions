import numpy as np
from src import util

from src.p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # Part (c): Train and test on true labels
    
    ## Load training set and test set with true labels
    x_train , t_train = util.load_dataset(train_path,label_col='t')
    x_test , t_test = util.load_dataset(test_path,label_col='t')

    ## Create and train a model on t labels
    t_trained_model = LogisticRegression()
    t_trained_model.fit(x_train,t_train)

    ## Make predictions on the test set and save to pred_path_c
    t_model_test_preds = t_trained_model.predict(x_test)
    np.savetxt(pred_path_c,X=t_model_test_preds,delimiter=',')
    
    # Part (d): Train on y-labels and test on true labels
    
    ## Load y-labels 
    _ , y_train = util.load_dataset(train_path,label_col='y')

    ## Create and train a model on y-labels
    y_trained_model = LogisticRegression()
    y_trained_model.fit(x_train,y_train)

    y_model_test_preds = y_trained_model.predict(x_test)
    np.savetxt(pred_path_d,X=y_model_test_preds,delimiter=',')
    # Part (e): Apply correction factor using validation set and test on true labels
    
    ## Load the validation set
    x_valid , y_valid = util.load_dataset(valid_path,label_col='y')

    x_valid_pos = x_valid[y_valid == 1,:]

    def h(x,theta):
        return np.reciprocal(1 + np.exp(-x @ theta))
   
    alpha = h(util.add_intercept(x_valid_pos),y_trained_model.theta).mean()

    y_model_corrected_preds = (h(util.add_intercept(x_test),
                                  y_trained_model.theta) > 0.5 * alpha).astype(np.int8)

    ## Save outputs to pred_path_e
    np.savetxt(pred_path_e,X=y_model_corrected_preds,delimiter=',')

    ## Make plots and save graphics

    ## Plot of test data with fit line determined by t-trained model
    util.plot(x_test,t_test,include_decision=True,theta=t_trained_model.theta,
              save_path='./output/p02_plot_c.png')

    ## Plot of the test data with the fit line determined by y-trained model
    util.plot(x_test,t_test,include_decision=True,theta=y_trained_model.theta,
              save_path='./output/p02_plot_d.png')

    ## Plot of test data with the fit line determined by the corrected y-trained model
    util.plot(x_test,t_test,include_decision=True,theta=y_trained_model.theta,q_2e=True,
              beta = np.log(alpha/(2-alpha)),
              save_path='./output/p02_plot_e.png')


