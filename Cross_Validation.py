#!/usr/bin/python

class CV:

    def __init__(self,K,model,train,labels,inputMap=0):
        """
        Constructs a CrossValidation object with the following attributes:
        n_folds: The number of folds
        classifier: object for the specific model
        input_data: {array}, shape (n_samples, n_features)
        input_labels: array-like, shape (n_samples,)
        folds_data: the examples for each fold
        folds_labels: the labels for each fold
        merge_data: examples of the K-1 folds merged into a single array
        merge_labels: labels of the K-1 folds merged into a single array
        trainedModel: the fitted model
        inputMapFlag: flag variable for whether feature map is used (0==No,1==Yes)
        
        """
        self.n_folds = K
        self.classifier = model
        self.inputMapFlag = inputMap
        self.input_data = train
        self.input_labels = labels
        self.folds_data = []
        self.folds_labels = []
        self.merge_data = []
        self.merge_labels = []
        self.trainedModel = []

    def feature_map(self):
        """
        Creates new feature space,
        based on all interaction features with degree <= 2
        """
        n = np.shape(self.input_data)[0]
        # Number of rows
        d = np.shape(self.input_data)[1]
        # Number of fields
        X2 = np.power(self.input_data,2)
        # Squares all the features
        self.input_data = np.concatenate((self.input_data,X2),axis=1)
        # Merge the arrays column-wise
        for i in xrange(d):
            for j in xrange(i+1,d):
                temp = self.input_data[:,i]*self.input_data[:,j]
                # Multiples of every distinct combination of features
                temp = temp.reshape(n,1)
                self.input_data = np.concatenate((self.input_data,temp),axis=1)
        return self.input_data

    def partition_set(self):
        """Partitions input dataset (inc. labels) into K folds
        """
        import numpy as np
        if self.inputMapFlag == 1:
            self.input_data = self.feature_map()
        # If requested, maps input data to higher dimensional space
        n = np.shape(self.input_data)[0]
        indices = np.random.permutation(n)
        self.input_data = self.input_data[indices,:]
        self.input_labels = self.input_labels[indices]
        self.folds_data = np.array_split(self.input_data,self.n_folds)
        # Partitions input data
        self.folds_labels = np.array_split(self.input_labels,self.n_folds)
        # Partitions input labels
        return self
    
    
    def merge_folds(self,index):
        """Merges K-1 of the folds
        """
        if index <0 or index > self.n_folds:
            raise ValueError(
                "Fold index must be between 0 and total number of folds"
                )
        d = np.shape(self.input_data)[1]
        self.merge_data = np.array([]).reshape(0,d)
        self.merge_labels = np.array([])
        K = self.n_folds
        for i in range(K):
            if i == index:
                continue
            else:
                self.merge_data = np.concatenate((self.merge_data,
                                  self.folds_data[i]),axis=0)
                # Merges K-1 of input data folds
                self.merge_labels = np.concatenate((self.merge_labels,
                                    self.folds_labels[i]),axis=0)
                # Merges K-1 of input label folds
                fold_test.append(i)
        return self
    
    def train(self,**kwargs):
        """Cross-validated training on K-1 folds
        """
        self.trainedModel = self.classifier.fit(self.merge_data,
                            self.merge_labels,**kwargs)
        return self
    
    def test(self,index):
        """Cross-validated testing on single fold
        """
        test_fold = self.folds_data[index]
        test_fold_labels = self.folds_labels[index]
        n_test = np.shape(test_fold)[0]
        test_predict = self.classifier.predict(test_fold)
        model_test_diff = test_fold_labels==test_predict
        test_incorrect = len(np.where(model_test_diff==False)[0])
        test_error = 1.*test_incorrect/n_test
        # Computes test error on the fold
        return test_error
    
    def cross_validation(self,**kwargs):
        _ = self.partition_set()
        K = self.n_folds
        cv_error = []
        for i in range(K):
            _ = self.merge_folds(i)
            _ = self.train(**kwargs)
            cv_error.append(self.test(i))
        return np.average(cv_error)
