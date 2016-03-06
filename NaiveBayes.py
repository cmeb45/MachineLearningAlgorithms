def bayes_parameters(X,Y):
    # Computes parameters of Naive Bayes generative model
    import numpy as np
    K = len(np.unique(Y))
    # Total number of classes
    p = np.shape(X)[1]
    # Total number of features
    n = np.shape(X)[0]
    # Total number of examples
    conditional = []
    # Class conditional distribution parameters (Laplace smoothing)    
    prior = []
    # Class prior distribution parameters (MLE)
    for label in xrange(K):
        indices = np.where(Y==label+1)[0]
        temp_split = X[indices,:]
        # All instances of class (label+1)
        temp_count = np.shape(temp_split)[0]
        # Number of instances of class (label+1)
        prior.append(1.*temp_count/n)
        temp_sum = np.apply_along_axis(sum,0,temp_split.toarray())
        # Number of instances of class (label+1) with a specific word
        conditional.append(1.*(1+1.*temp_sum)/(2+temp_count))
    params = [prior,conditional]
    return params

def bayes_predictions(params,test):
    # Makes predictions on test data based on Naive Bayes model
    import numpy as np
    def logit(x):
        return np.log(1.*x/(1-x))
    # Computes logit function
    def complement(x):
        return 1-x
    # Computes complement of a probability/parameter
    vect_logit = np.vectorize(logit)
    # Vectorized form of logit function
    vect_comp = np.vectorize(complement)
    # Vectorized form of complement function
    n_test = np.shape(test)[0]
    # Total number of test examples
    prior = params[0]
    # Class prior distribution parameters
    conditional = params[1]
    # Class conditional distribution parameters
    K = len(prior)
    # Total number of classes
    p = np.shape(conditional)[1]
    # Total number of features
    test_p = np.shape(test)[1]
    # Total number of features in test set
    if p != test_p:
        print "Error: Number of training and testing features differ"
        return
    preds = []
    # Stores predicted class for each test point
    weight = vect_logit(conditional)
    # Stores weight matrix
    condition_comp = vect_comp(conditional)
    intercept = np.log(prior) + np.sum(np.apply_along_axis(np.log,1,condition_comp),axis=1)
    # Stores intercept vector
    weight = weight.transpose()
    classifier = test.dot(weight) + intercept
    # Represents classifier as linear function of test examples
    preds = np.argmax(classifier,axis=1) + 1
    # Add 1 because the classes have a 1-based index
    preds = preds.reshape(n_test,1)
    return preds
