""" nlp.py

Perform some basic nlp tasks

"""
import sys
import os, re
import numpy as np
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split

from gmutils.objects import Options
from gmutils.normalize import normalize
from gmutils.utils import err, argparser, read_file, read_dir, iter_file, isTrue, monitor_setup, monitor, serialize, deserialize

import spacy
try:
    spacy_nlp = spacy.load('en_core_web_lg')    # download separately: https://spacy.io/models/
except:
    pass

################################################################################
# FUNCTIONS


def lemmatize(text):
    spacy_doc = spacy_nlp(text)
    span = spacy_doc[:]
    return span.lemma_


def lemmatize_file(file):
    lines = []
    for line in read_file(file):
        line = normalize(line)
        lines.append(lemmatize(line))
    return lines
        

def series_to_dict(names, row):
    """
    Take a pandas series and return a dict where the key are the original columns
    """
    out = {}
    for i, name in enumerate(names):
        if not name in cols_to_compare:
            continue
        out[name] = row[i]

    return out


def parse(text):
    """
    Generate a detailed dependency parse of some text.

    """
    spacy_doc = spacy_nlp(text)


    
################################################################################
# MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "Some general NLP tasks: nlp.py"})
    args = parser.parse_args()

    text = ''
    
    if args.file:
        for file in args.file:
            text += read_file(file)
            
    elif args.str:
        text = '  '.join(args.str)
        
    else:
        print(__doc__)
        exit()

    # text = normalize(text)
    parsed_text = parse(text)
    print(parsed_text)
    
        
################################################################################
################################################################################
