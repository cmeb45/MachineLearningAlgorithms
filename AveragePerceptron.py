#!/usr/bin/python

class AveragePerceptron:
    
    def __init__(self,avg_weight=[],avg_bias=0):
        self.model_avg_weight = avg_weight
        self.model_avg_bias = avg_bias
        # Parameters from fitted model: average weight & average bias


    def fit(self,InputData,InputLabels,NoPasses):
        """Fits average perceptron model according to training data.
        
        Parameters
        ----------
        InputData: {array}, shape (n_samples, n_features)
        
        InputLabels: array-like, shape (n_samples,)
        
        NoPasses: number of iterations
        """
        import numpy as np
        n = np.shape(InputData)[0]
        # Number of training examples
        d = np.shape(InputData)[1]
        # Number of features
        zero = np.array([0])
        weight = np.repeat(zero,d)
        # Initialize weight vector
        bias = 0
        # Intialize bias
        c_weight = np.repeat(zero,d)
        # Initialize cached weight vector
        c_bias = 0
        # Intialize cached bias
        counter = 1
        # Initialize example counter to one
        for i in xrange(NoPasses):
            ShuffleIndices = np.random.permutation(n)
            InputData = InputData[ShuffleIndices,:]
            InputLabels = InputLabels[ShuffleIndices]
            # Randomly shuffle order of training examples
            for t in xrange(n):
                # Online perceptron: Start
                if InputLabels[t]*InputData[t,:].dot(weight)<=0:
                    weight = weight + (InputLabels[t]*InputData[t,:])
                    # Update weights
                    bias = bias + InputLabels[t]
                    # Update bias
                    c_weight = c_weight + (InputLabels[t]*counter*InputData[t,:])
                    # Update cached weights
                    c_bias = c_bias + (InputLabels[t]*counter)
                    # Update cached bias
                counter += 1
            # Update parameters for next iteration of online perceptron
        avg_weight = weight - ((1./counter)*c_weight)
        # Averaged weight
        avg_bias = bias - ((1./counter)*c_bias)
        # Averaged bias
        self.model_avg_weight = avg_weight
        self.model_avg_bias = avg_bias
        return self
    
    def predict(self,InputData):
        import numpy as np
        n = np.shape(InputData)[0]
        # Number of testing examples
        d = np.shape(InputData)[1]
        # Number of features
        avg_weight = self.model_avg_weight
        # Weight vector from training model
        preds = np.array([])
        for t in xrange(n):
            if InputData[t,:].dot(avg_weight) <= 0: 
                preds = np.concatenate((preds,[-1]),axis=0)
            else:
                preds = np.concatenate((preds,[1]),axis=0)
        return preds
