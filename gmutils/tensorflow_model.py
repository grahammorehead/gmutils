""" tensorflow_model.py

    Tools for building ML models with TensorFlow

"""
import os, sys, re
import numpy as np

from gmutils.utils import err, argparser
from gmutils.model import Model

################################################################################
# CONFIG

default = {
    'default_dir'        : 'model',
    'batch_size'         : 100,
    'epochs'             : 5,
}

################################################################################
# OBJECTS

class TensorflowModel(Model):
    """
    An object to manage the training, storage, and utilizating of TensorFlow models

    Attributes  (depends on subclass)
    ----------
    label : str
        The supervised label

    model : a TensorFlow Model

    model_dir : str
        Path to where the model is stored

    model_file : str
        File path where the model is stored

    estimators : array of str
        Names of the underlying estimator(s) to be used

    """
    def run(self):
        """
        Run the session
        """
        with tf.Session() as sess:
            while True:
                pass



        
################################################################################
# FUNCTIONS


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
