""" model.py

Code and objects to build ML models

"""
import os, sys, re
import json
import pickle
from copy import deepcopy
import random
import numpy as np
import scipy
from gmutils.utils import err, argparser
from gmutils.objects import Object
try:
    from sklearn.preprocessing import Binarizer
    from sklearn import metrics
    from sklearn.metrics import mean_absolute_error
    from sklearn.externals.joblib.parallel import parallel_backend
except Exception as e: err([], {'exception':e, 'level':0})
try:
    import pandas as pd
except Exception as e: err([], {'exception':e, 'level':0})
    
################################################################################
# CONFIG

default = {
    'default_dir'        : 'model',
}
  
################################################################################
# OBJECTS

class Model(Object):
    """
    An object to manage the training, storage, and utilizating of a classifier

    Attributes  (depends on subclass)
    ----------
    label : str
        The supervised label

    model : an underlying Keras, TensorFlow, or Scikit-Learn Model

    model_dir : str
        Path to where the model is stored

    model_file : str
        File path where the model is stored

    estimators : array of str
        Names of the underlying estimator(s) to be used

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options
        """
        self.set_options(options, default)
        super().__init__(options)
        

    def generate(self):
        """
        Override in subclass.

        In some cases will create the attribue self.model

        """
        pass


    def summary(self):
        """
        Print and/or save info which describes this model

        """
        self.model.summary()
        if self.isTrue('model'):
            self.to_png()


    def to_png(self):
        """
        Generate a png-format visual description of the model

        """
        save_dir = self.get('model_dir')
        try:
            plot_model(model, to_file=save_dir + '/model.png', show_shapes=True)
        except:
            pass


    def load_weights(self, weights_file):
        """
        Typically for underlying Keras Models.  Load weights saved from a trained version of this identical architecture.

        """
        self.model.load_weights(weights_file)
        
    
    def fit(self, dataset):
        """
        Train a supervised model from a list of inputs X and their corresponding classifications Y.

        Parameters
        ----------
        dataset : TrainingDataset object

        the 'train' attribute of the dataset will be split into:
            X : array of inputs
            Y : array of supervised outputs

        """
        verbose = self.get('verbose')

        X = dataset.x_train
        Y = dataset.y_train.values

        # Y = dataset.y_train.values.ravel()   # to prevent DataConversionWarning
        with parallel_backend('threading'):
            self.model.fit(X, Y)
        sys.stderr.write("Done Training.\n")


    def binarize(self, X):
        """
        Uses a stored threshold to binarize predictions

        """
        if not self.get('thresh'):
            self.set_thresh()
        
        if not self.get('binarizer'):
            self.set('binarizer', Binarizer(threshold=self.get('thresh')))
        
        X = self.get('binarizer').transform(X.values.reshape(1, -1))
        X = pandasize(X)
        
        return X
    
        
    def predict(self, X, binarizer=None):
        """
        Return the prediction for each input line.

        Parameters
        ----------
        X : array of str, Sentence objects, ints, floats, etc

        Returns
        -------
        pandas.Series

        """
        if X is None:
            return None
        if self.get('vocab'):
            X = self.embed(X)

        preds = self.model.predict(X)

        if self.get('binarize_predictions'):
            preds = self.binarize(preds)

        preds = pandasize(preds)
        if not isinstance(preds, pd.Series):  err([], {'ex':"ERROR: preds not a pd.Series."})

        return preds

    
    def predict_one(self, x, binarizer=None):
        """
        Return the prediction for one input line

        Parameters
        ----------
        x : str, or Sentence object, int, float, etc

        Returns
        -------
        float or label (model output for one input)

        """
        #x = x.reshape(1, -1)
        #X = pd.DataFrame(x)
        X = [x]
        preds = self.predict(X, binarizer=binarizer)
        return preds[0]
        
        
    def predict_proba(self, X):
        """
        Return the prediction for each input line.

        Parameters
        ----------
        X : array of str, Sentence objects, ints, floats, etc

        Returns
        -------
        pandas.Series

        """
        if self.get('vocab'):
            X = self.embed(X)
        preds = self.model.predict_proba(X)

        preds = pandasize(preds)
        if not isinstance(preds, pd.Series):  err([], {'ex':"ERROR: preds not a pd.Series."})
        
        return preds


    def next_thresh(self, thresh, F1):
        """
        Attempts to find a threshold which is half of the resulting F1

        will raise or lower the thresh accordingly

        """
        if F1 > 0.0:
            ratio = F1 / (2*thresh)
            thresh = ratio * thresh
        else:
            thresh = 0.5 * thresh
            
        return thresh
        
    
    def set_thresh(self, X, Y):
        """
        Heuristically try a few different thresholds to find a semi-optimal one.  Set it in the default binarizer

        """
        best_thresh = self.get('thresh')
        best_F1 = 0.0

        thresh = 0.5
        n = 2
        while n > 0:
            binarizer = Binarizer(threshold=thresh)
            preds = self.predict(X, binarizer)
            F1 = metrics.f1_score(Y, preds)
            sys.stderr.write("\tthresh: %0.8f  =>  F1: %0.8f ...\n"% (thresh, F1))

            if F1 > best_F1:
                best_F1 = F1
                best_thresh = thresh

            # prepare next loop
            thresh = self.next_thresh(thresh, F1)
            n -= 1
            
        self.set('thresh', best_thresh)
        self.set('binarizer', Binarizer(threshold=best_thresh))
                
    
    def evaluate_regression(self, dataset):
        """
        Evaluate this regression model against the test portion of a TrainingDataset

        Parameters
        ----------
        dataset : TrainingDataset object
            This function only pays attention to the testing data: dataset.test

        """
        X = dataset.x_test
        Y = dataset.y_test

        # sys.stderr.write("\nMaking predictions ...\n")
        preds = self.predict(X)
        # err([Y, preds])
        
        mae = mean_absolute_error(Y, preds)
        print("MAE:", mae)

        # Same for positive cases
        X, Y = dataset.get_positive_testdata()

        # sys.stderr.write("\nMaking predictions ...\n")
        preds = self.predict(X)
        # err([Y, preds])
        
        mae = mean_absolute_error(Y, preds)
        print("MAE:", mae)
        
        
    def evaluate(self, dataset):
        """
        Evaluate this model against the test portion of a TrainingDataset

        Parameters
        ----------
        dataset : TrainingDataset object
            This function only pays attention to the testing data: dataset.test

        """
        X = dataset.x_test
        Y = dataset.y_test

        if self.get('binarize_predictions'):
            Y = self.binarize(Y)
        
        sys.stderr.write("\nMaking predictions ...\n")
        preds = self.predict(X)
        # err([Y, preds])
        report = metrics.classification_report(Y, preds)
        cm = metrics.confusion_matrix(Y, preds)
        print(report)
        print("Confusion matrix:\n")
        print(cm)
        print()
        
        return
        
        # old way:
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        thresh = 0.5
        for i, claim_score in enumerate(preds):
            tp += claim_score >= thresh and bool(Y[i]) == True
            tn += claim_score < thresh and bool(Y[i]) == False
            fp += claim_score >= thresh and bool(Y[i]) == False
            fn += claim_score < thresh and bool(Y[i]) == True

        acc = float(tp + tn) / (tp+tn+fp+fn)
        prec = float(tp) / (tp + fp)
        rec = float(tp) / (tp + fn)
        f1 = float(2*tp) / (2 * tp + fp + fn)

        print ("\nAccuracy: %s   F1: %s   Precision: %s   Recall: %s\n" % (acc, f1, prec, rec))

        
    def embed(self, X):
        """
        Take rows of input X, embed the text inputs, and collate everything into a ANN-readable input

        X : DataFrame of rows having various input types, all to be numericised and inserted into a single matrix.

        """
        lemmas_array = X[:,0]
        Xs = np.zeros((len(lemmas_array), self.get('sentence_length')), dtype='int32')
        for i, lemmas in enumerate(lemmas_array):
            j = 0
            for lemma in lemmas:
                vector_id = self.get('vocab')(key=lemma)
                if vector_id >= 0:
                    Xs[i, j] = vector_id
                else:
                    Xs[i, j] = 0
                j += 1
                if j >= self.get('sentence_length'):
                    break
                
        return Xs

################################################################################
# FUNCTIONS    
    
def pandasize(X):
    """
    Convert some incoming data to a pandas DataFrame or Series, depending on its dimensionality

    Parameters
    ----------
    X : list, array, or numpy Array

    Returns
    -------
    pandas Series or DataFrame
    """
    make_series = False   # Assume DataFrame until otherwise indicated
    
    if isinstance(X, pd.Series):
        return X
    elif isinstance(X, pd.DataFrame):
        return X
        
    elif isinstance(X, list):
        make_series = True
    
    elif isinstance(X, np.ndarray):

        if len(X.shape) == 1:
            make_series = True

        if len(X.shape) == 2:
            if X.shape[0] == 1:
                make_series = True
                X = X[0]

    if make_series:
        X = pd.Series(X)
    elif isinstance(X, scipy.sparse.csr_matrix):
        X = pd.DataFrame(X.toarray())
    else:
        X = pd.DataFrame(X)

    return X


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    try:
        parser = argparser({'desc': "Tools to model patterns in text using scikit-learn, keras, and TF: model.py"})
        #  --  Tool-specific command-line args may be added here
        args = parser.parse_args()   # Get inputs and options

        print(__doc__)

    except Exception as e:
        print(__doc__)
        err([], {'exception':e, 'exit':True})

        
################################################################################
################################################################################
