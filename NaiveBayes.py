#!/usr/bin/python

import numpy as np

def logit(x):
    """ Computes logit function
    
    Parameters
    ----------
    x : {float, int}
    
    Returns
    -------
    out : {float, int}
        Logit value
    """
    if x > 0:
        out = np.log(1.*x/(1-x))
        return out

def complement(x):
    """ Computes complement of a probability/parameter
    
    Parameters
    ----------
    x : {float, int}
    
    Returns
    -------
    out : {float, int}
        Complement
    """
    out = 1-x
    return out

class NaiveBayes:

    def __init__(self,prior=[],conditional=[]):
        self.model_prior = prior
        self.model_conditional = conditional
        # Parameters from fitted model: prior & class conditional

    def fit(self,X,Y):
        """Fits Naive Bayes generative model according to training data.
        
        Parameters
        ----------
        X : {array}, shape (n_samples, n_features)
        
        Y : array-like, shape (n_samples,)
        
        Returns
        -------
        self : object
            Returns self.
        """
        K = len(np.unique(Y))
        p = np.shape(X)[1]
        n = np.shape(X)[0]
        conditional = []
        # Class conditional distribution parameters (Laplace smoothing)
        prior = []
        # Class prior distribution parameters (MLE)
        for label in xrange(K):
            indices = np.where(Y==label+1)[0]
            temp_split = X[indices,:]
            temp_count = np.shape(temp_split)[0]
            prior.append(1.*temp_count/n)
            temp_sum = np.apply_along_axis(sum,0,temp_split.toarray())
            conditional.append(1.*(1+1.*temp_sum)/(2+temp_count))
        self.model_prior = prior
        self.model_conditional = conditional
        return self

    def predict(self,params,test):
        """Makes predictions on test data based on Naive Bayes model
        
        Parameters
        ----------
        test: {array}, shape (n_samples, n_features)
        
        Returns
        -------
        preds : list
            Returns predicted class for each test point.
        """
        # 
        # 
        vect_logit = np.vectorize(logit)
        vect_comp = np.vectorize(complement)
        n_test = np.shape(test)[0]
        prior = self.model_prior
        conditional = self.model_conditional
        K = len(prior)
        p = np.shape(conditional)[1]
        test_p = np.shape(test)[1]
        if p != test_p:
            print "Error: Number of training and testing features differ"
            return
        preds = []
        weight = vect_logit(conditional)
        # Stores weight matrix
        condition_comp = vect_comp(conditional)
        intercept = np.log(prior) + np.sum(np.apply_along_axis(np.log,1,condition_comp),axis=1)
        # Stores intercept vector
        weight = weight.transpose()
        classifier = test.dot(weight) + intercept
        preds = np.argmax(classifier,axis=1) + 1
        # Add 1 because the classes have a 1-based index
        preds = preds.reshape(n_test,1)
        return preds
