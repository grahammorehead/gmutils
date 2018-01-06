""" model.py

Code and objects to build ML models

"""
import os, sys, re
import json
import pickle
from copy import deepcopy
import random

from keras.utils.vis_utils import plot_model

from .utils import *
from .objects import *


################################################################################
# CONFIG

default = {
    'batch_size'         : 100,
    'num_epochs'         : 5,
    'optimizer'          : 'adam',
    'loss'               : 'binary_crossentropy',
}

  
################################################################################
# OBJECTS

class Model(Object):
    """
    An object to manage the training, storage, and utilizating of a classifier

    """
    def __init__(self, options=None):
        """
        Initialize the object

        Parameters
        ----------
        options : dict

        """
        self.set_options(options, default)            # For more on 'self.set_options()' see DSObject class
        self.generate()


    def generate(self):
        """
        override in subclass

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
        save_dir = self.get('model')
        try:
            plot_model(model, to_file=save_dir + '/model.png', show_shapes=True)
        except:
            pass
        
    
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
        verbose = self.get('verbose')     # For more on 'self.get(option_name)' see objects.py DSObject class

        X, Y = dataset.get_training_XY(self.get('signal'))
        X = self.embed(X)


        
        sys.stderr.write("Done Training.\n")


    def binarize(self, X):
        """
        Uses a stored threshold to binarize predictions

        """
        if not self.get('binarizer'):
            thresh = self.get('thresh')
            self.set('binarizer', Binarizer(threshold=thresh))
        binarizer = self.get('binarizer')
        
        return binarizer.transform(X)
        
        
    def predict(self, input, binarizer=None):
        """
        Return the prediction for each input line.

        Parameters
        ----------
        input : array of str, or Sentence objects

        Returns : array of float

        """
        X = self.embed(input)
        preds = self.model.predict(X)
        
        if binarizer is None:
            preds = self.binarize(preds)
        else:
            preds = binarizer.transform(preds)    
        
        return preds

    
    def predict_proba(self, input):
        """
        Return the prediction for each input line.

        Parameters
        ----------
        input : array of str, or Sentence

        Returns : array of float
        """
        X = self.embed(input)
        preds = self.model.predict(X)

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
            
        self.set('binarizer', Binarizer(threshold=best_thresh))
                
    
    def evaluate(self, dataset):
        """
        Evaluate this model against the test portion of a TrainingDataset

        Parameters
        ----------
        dataset : TrainingDataset object
            This function only pays attention to the testing data: dataset.test

        """
        X, Y = dataset.get_testing_XY(self.get('signal'))
        
        sys.stderr.write("\nSetting threshold ...\n")
        self.set_thresh(X, Y)
        
        sys.stderr.write("\nMaking predictions ...\n")
        preds = self.predict(X)
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
##   MAIN   ##

if __name__ == '__main__':
    try:
        parser = argparser({'desc': "Tools to model patterns in text using keras and TF: model.py"})
        #  --  Tool-specific command-line args may be added here
        args = parser.parse_args()   # Get inputs and options

        print(__doc__)

    except Exception as e:
        print(__doc__)
        err([], {'exception':e, 'exit':True})

        
################################################################################
################################################################################
