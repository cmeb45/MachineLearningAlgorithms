#!/usr/bin/python

import numpy as np

class AveragePerceptron:
    
    def __init__(self,avg_weight=[],avg_bias=0):
        self.model_avg_weight = avg_weight
        self.model_avg_bias = avg_bias
        # Parameters from fitted model: average weight & average bias


    def fit(self,InputData,InputLabels,NoPasses):
        """Fits average perceptron model according to training data.
        
        Parameters
        ----------
        InputData : {array}, shape (n_samples, n_features)
        
        InputLabels : array-like, shape (n_samples,)
        
        NoPasses : number of iterations
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        n = np.shape(InputData)[0]
        d = np.shape(InputData)[1]
        zero = np.array([0])
        weight = np.repeat(zero,d)
        bias = 0
        c_weight = np.repeat(zero,d)
        c_bias = 0
        counter = 1
        for i in xrange(NoPasses):
            ShuffleIndices = np.random.permutation(n)
            InputData = InputData[ShuffleIndices,:]
            InputLabels = InputLabels[ShuffleIndices]
            for t in xrange(n):
                # Online perceptron
                if InputLabels[t]*InputData[t,:].dot(weight)<=0:
                    weight = weight + (InputLabels[t]*InputData[t,:])
                    bias = bias + InputLabels[t]
                    c_weight = c_weight + (InputLabels[t]*counter*InputData[t,:])
                    c_bias = c_bias + (InputLabels[t]*counter)
                counter += 1
        avg_weight = weight - ((1./counter)*c_weight)
        avg_bias = bias - ((1./counter)*c_bias)
        self.model_avg_weight = avg_weight
        self.model_avg_bias = avg_bias
        return self
    
    def predict(self,InputData):
        """Runs fitted average perceptron model on test data.
        
        Parameters
        ----------
        InputData : {array}, shape (n_samples, n_features)
        
        Returns
        -------
        preds : {array}
            Returns predicted class for each test point.
        """
        import numpy as np
        n = np.shape(InputData)[0]
        d = np.shape(InputData)[1]
        avg_weight = self.model_avg_weight
        preds = np.array([])
        for t in xrange(n):
            if InputData[t,:].dot(avg_weight) <= 0: 
                preds = np.concatenate((preds,[-1]),axis=0)
            else:
                preds = np.concatenate((preds,[1]),axis=0)
        return preds
