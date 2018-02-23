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

################################################################################
# SPACY INTEGRATION

import spacy
try:
    spacy_nlp = spacy.load('en_core_web_lg')    # download separately: https://spacy.io/models/
except:
    spacy_nlp = None


def combine_with_previous(previous, current):
    """
    Correct for some errors made by the spaCy sentence splitter

    Parameters
    ----------
    previous: spaCy Span

    current: spaCy Span

    Returns
    -------
    bool
    """
    verbose = False
    if verbose: err([previous.text, current.text])

    # Current sentence too short
    if current.end - current.start < 3:
        if verbose:
            err([[current.text]])
        return True

    # Previous sentence had no ending punctuation
    if not ( re.search("[\.?!]$", previous.text) \
                 or re.search("[\.?!]\S$", previous.text) \
                 or re.search("[\.?!]\S\S$", previous.text) \
                 or re.search("[\.?!]\s$", previous.text) \
                 or re.search("[\.?!]\s\s$", previous.text) ):
        if verbose:
            err([[previous.text]])
        return True

    return False
    
    
def sentence_segmenter(doc):
    """
    Begin with spaCy sentence splits, then correct for some mistakes.

    Parameters
    ----------
    doc : spaCy Doc

    Returns
    -------
    array of pair (start, end)

    """
    verbose = False
    sen_spans = []
    sen_offsets = []  # Sentence offset pairs
    spacy_sentences = list(doc.sents)
    for i,sen in enumerate(spacy_sentences):

        # Current Sentence
        start = sen.start
        end = sen.end
        current = doc[start:end]

        # Previous Sentence
        previous = p_start = p_end = None
        if i > 0:
            prev = spacy_sentences[i-1]
            p_start = prev.start
            p_end = prev.end
            previous = doc[p_start:p_end]
            if verbose:
                err([[previous.text, current.text]])

        # Correct for mistakes
        if previous is not None  and  combine_with_previous(previous, current):
            sen_spans[-1] = doc[p_start:end]
            sen_offsets[-1] = [p_start, end]

        else:
            sen_spans.append( doc[start:end] )
            sen_offsets.append( [start, end] )
            
    return sen_offsets
    

def set_sent_starts(doc):
    for offsets in sentence_segmenter(doc):
        print("OFFSETS:", offsets)
        start, end = offsets
        sent = doc[start:end]
        print('SENT:', sent)
        sent[0].sent_start = True
        for token in sent[1:]:
            token.sent_start = False
    return doc

spacy_nlp.add_pipe(set_sent_starts, name='sentence_segmenter', before='parser')
spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'), before='sentence_segmenter')



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
