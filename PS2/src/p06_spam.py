import collections
import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    return message.lower().split(" ")


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.
d
    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    
    i = 0
    output = {}
    full_dict = {}
    for message in messages:
        for word in set(get_words(message)):
            if word in output.keys():
                continue
            try:
                full_dict[word] += 1
            except:
                full_dict[word] = 1
            if full_dict[word] == 5:
                output[word] = i
                i+=1
    return output

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """

    n = len(messages)
    N = len(word_dictionary)
    array = np.zeros((n,N))
    for i in range(n):
        for word in get_words(messages[i]):
            if word in word_dictionary.keys():
                array[i,word_dictionary[word]] += 1
            else:
                continue
    return array
    

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    def add_ones(matrix):
        m , n = matrix.shape
        new = np.ones((m+1,n))
        new[:m,:] = matrix
        return new
    
    num_of_ex , _ = matrix.shape

    ##* The positive (negative) matrix has rows with label 0 (1) set to zero, and rows with label 1 (0) unchanged.
    pos_matrix = labels.reshape(-1,1) * matrix
    neg_matrix = matrix - pos_matrix

    pos_token_probs = add_ones(pos_matrix).sum(axis=0) / add_ones(pos_matrix).sum()
    neg_token_probs = add_ones(neg_matrix).sum(axis=0) / add_ones(neg_matrix).sum()
    pos_prior = labels.sum() / num_of_ex

    ##* In order to predict, we only need to store the vector whose ith entry is the log of p(t_j | y = 1)/p(t_j | y = 0).
    ##* Together with the log of pos_prior / 1 - pos_prior
    state = np.log( pos_token_probs / neg_token_probs) , np.log(pos_prior / (1 - pos_prior))
    return state
    
def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    
    log_probs , log_prior = model

    ##* The decision_vector is the value of log p(t | y = 1)p(y=1) / p(t | y = 0)p(y=0) for each of the rows in the test matrix.
    ##* the linear decision boundary decision_vector > 0 is equivalent to MAP estimation for the posterior distribtution p(y | t).

    decision_vector = (log_probs.reshape(1,-1) * matrix).sum(axis=1) + log_prior
    return (decision_vector > 0).astype(np.int8)

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """

    log_probs , _ = model
    top_indicies = np.argsort(log_probs)[-5:]

    keys = dictionary.keys()
    n = len(keys)
    dict = {i : key for (i, key) in zip(range(n) , keys )}
    top_words = [dict[i] for i in top_indicies]
    top_words.reverse()
    return top_words

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: A list of the radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    scores = []
    for radius in radius_to_consider:
        predicted_labels = svm.train_and_predict_svm(train_matrix=train_matrix, train_labels=train_labels,test_matrix=val_matrix, radius=radius)
        score = np.mean(predicted_labels == val_labels)
        scores.append(score)

    max_index = np.array(scores).argmax()
    return radius_to_consider[max_index]

def main():
    train_messages, train_labels = util.load_spam_dataset('./data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('./data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('./data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()



