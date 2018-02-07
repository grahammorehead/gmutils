""" sklearn_model.py

    Tools for building ML models with Scikit-Learn

"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import csv
import math

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from .utils import err
from .model import Model

################################################################################
# CONFIG

default = {
    'default_dir'        : 'model',
    'batch_size'         : 100,
    'epochs'             : 5,
    'optimizer'          : 'adam',
    'loss'               : 'binary_crossentropy',
    'voting'             : 'soft',
    'nn_max_iter'        : 1000,
    'nn_hl'              : (36, 18, 18, 9, 3),   # hidden layer sizes
    'binarize_predictions' : False
}

################################################################################
# OBJECTS

class SklearnModel(Model):
    """
    An object to manage the training, storage, and utilizating of a Scikit-Learn classifier

    Attributes  (depends on subclass)
    ----------
    label : str
        The supervised label

    model : a Keras, TensorFlow, or Scikit-Learn Model

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
        self.set_options(options, default)        # For more on 'self.set_options()' see object.Object
        self.generate()


    def generate(self):
        """
        Generate the guts of a capsule network

        Parameters
        ----------
        input_shape : data shape, 3d, [width, height, channels]

        n_class : number of classes

        routings: number of routing iterations

        Generates
        ---------
        Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.

        """
        verbose = True
        estimators = []
        
        """  SVM multi-label classifier
        """
        if 'svm' in self.get('estimators'):
            svm = SGDClassifier(loss='log', alpha=0.0001, n_iter=50, penalty='l2', verbose=verbose)
            estimators.append(('SVM', svm))

        """  Multi-Layer Perceptron (AKA Fully-connected feed-forward Neural Net)
                 more info: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
        """
        if 'nn_adam' in self.get('estimators'):
            nn_adam = MLPClassifier(hidden_layer_sizes=self.get('nn_hl'), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=self.get('nn_max_iter'), shuffle=True, tol=1e-10, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            estimators.append(('NN_adam', nn_adam))

        if 'nn' in self.get('estimators'):
            nn = MLPClassifier(hidden_layer_sizes=self.get('nn_hl'), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=1.0, power_t=0.05, max_iter=self.get('nn_max_iter'), shuffle=True, tol=1e-08, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            estimators.append(('NN', nn))

        """  Ada Boost Ensemble Decision Tree Classifier
                 more info: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        """
        if 'ada' in self.get('estimators'):
            be = DecisionTreeClassifier(max_depth=4)
            ada = AdaBoostClassifier(base_estimator=be, n_estimators=1000, learning_rate=0.4, algorithm='SAMME.R')
            estimators.append(('ADA', ada))


        """  Random Forest Classifier (also uses Decision Trees)
                 more info: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        if 'rf' in self.get('estimators'):                #####################  Removed: max_features=1000,
            rf = RandomForestClassifier(n_estimators=10000, criterion='entropy', max_depth=None, min_samples_split=2, max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=True, n_jobs=-1, verbose=verbose)
            estimators.append(('RF', rf))


        """  Voting Classifier: Ensemble of all the classifiers above
                 more info: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
        """
        if len(estimators) == 1:
            self.model = estimators[0][1]
        else:
            self.model = VotingClassifier(estimators=estimators, voting=self.get('voting'), n_jobs=-1)



################################################################################
# FUNCTIONS


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    try:
        parser = argparser_ml({'desc': "Tools to train and use Scikit-Learn models: sklearn_model.py"})
        args = parser.parse_args()   # Get inputs and options

    except Exception as e:
        print(__doc__)
        err([], {'exception':e, 'exit':True})

        
################################################################################
################################################################################
