""" run.py

    Train, evaluate, and / or use a Model

"""
import os, sys
import pandas as pd

from utils import *
from dataset import Dataset
from model import Model

################################################################################
##   MAIN   #

if __name__ == '__main__':

    # Parse command-line args
    parser = argparser_classifier({'desc': "Model Tool: run.py"})
    
    #   --  Tool-specific command-line args may be added here

    # Examples:
    #   parser.add_argument('--samplenoarg', help='Use this option (does not take an argument)', required=False, action='store_true')
    #   parser.add_argument('--sampleyesarg', nargs='?', action='append', help='Use this option (does take an argument)', required=False)
    
    args = parser.parse_args()                               # Get inputs and options

    # Set some options explicitly
    # args.signal          = 'the-column-with-output'
    # args.smote           = True                            # For synthetic minority oversampling
    # args.majunder        = True                            # For majority undersampling
    dataset = None
    args.datasetFile     = '../data/dataset.pkl'
    args.modelFile       = '../models/model.pkl'

    # Load existing training Dataset
    if args.loadDataset:
        dataset = deserialize(args.datasetFile)


    # Load an existing Model
    if args.loadModel:
        model = deserialize(args.modelFile)

    # Generate a Model
    else:
        model = DeepModel(options=args)
        model.train(dataset, options=args)
        sys.stderr.write('Claims DeepModel generated.\n')
        serialize_DeepModel(model)

    # Test
    if args.test  or  args.trainAndTest:
        model.test(dataset)

        
################################################################################
################################################################################
