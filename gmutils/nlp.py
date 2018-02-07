""" nlp.py

Perform some basic nlp tasks

"""
import sys
import os, re
import numpy as np
import itertools

import pandas as pd
from sklearn.model_selection import train_test_split

from ds_objects.objects import Options
from normalize import normalize
from ds_objects.document import Document, generate_standalone_sentence, generate_documents
from ds_objects.dataset import Dataset
from ds_narrative_patterns.claims import ClaimsAnalyzer

import spacy
try:
    spacy_nlp = spacy.load('en_core_web_lg')    # download separately: https://spacy.io/models/
except:
    pass

from .utils import err, argparser, read_file, read_dir, iter_file, isTrue, monitor_setup, monitor, serialize, deserialize

################################################################################
##   FUNCTIONS   ##


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
        

def split_further(X):
    out = []
    for x in X:
        if re.search('$', x):
            x = re.sub(r'\$', '\$ ', x)
            x2 = x.split(' ')
            out.extend(x2)
        else:
            out.append(x)
            
    return out


def intersperse(A, B):
    if isinstance(A, str)  or  isinstance(A, unicode):
        A = A.split(' ')
        A = split_further(A)
    if isinstance(B, str)  or  isinstance(B, unicode):
        B = B.split(' ')
        B = split_further(B)

    C = []
    for i in range(len(A)):
        try:
            C.append(A[i] + '/' + B[i])
        except:
            pass
    return '  '.join(C)
        

def make_int(xstr):
    xint = 0
    try:
        xint = int(xstr)
    except:
        pass
    return xint

                
def study_TF_positives(Xs, X, X_algo, line=None):
    """
    Track the True/False positives for some variable X

    The Xs dict has the form:

        Xs = { 'N':0, 'TP':0, 'FP':0, 'TN':0, 'FN':0 }

    X      element of: (0, 1)
    X_algo element of: (0, 1)

    """
    Xs['N'] += X
    if X_algo > 0:
        if X_algo == X:
            Xs['TP'] += 1
            if line is not None:
                print('TP:', line)
        else:
            Xs['FP'] += 1
            if line is not None:
                print('FP:', line)
    else:
        if X_algo == X:
            Xs['TN'] += 1
            if line is not None:
                print('TN:', line)
        else:
            Xs['FN'] += 1
            if line is not None:
                print('FN:', line)


def choose_two(X):
    """
    Creates a list having every pairwise combination in X

    Parameters
    ----------
    X : array

    Returns
    -------
    list of pairs
    """
    return list(itertools.combinations(X, 2))



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




def simple_avg(thisrow, thresh):
    """
    Compute the average, make a binary vote
    """
    verbose = True
    
    total = 0.0
    norm = 0.0
    n = 0
    for name,tag in thisrow.items():
        if tag is None:
            continue
        if name not in cols_to_compare:
            continue
        
        n += 1
        coef = 1.0
        total += coef * float(tag)
        norm  += coef

    if n == 0:
        return None  # if no data was read
        
    avg = 0.0
    if norm > 0.0:
        avg = total / norm
    else:
        return None

    out = 0
    if avg >= thresh:
        out = 1
        
    return out

    

def save_dataframe_as_tsv(file):
    """
    Load serialized dataframe.  Save as TSV

    """
    df = pd.DataFrame.from_csv(file)
    newfile = re.sub(r'\.df$', '.tsv')
    err('not complete'])

    
            
################################################################################
##   MAIN   ##

if __name__ == '__main__':

    parser = argparser({'desc': "Some general NLP tasks: nlp.py"})

    #  --  Tool-specific command-line args may be added here
    parser.add_argument('--classification_var',   nargs='?', action='append', help='Name of the column to compare', required=False)
    parser.add_argument('--output_file',   nargs='?', action='append', help='Path to output file (only used by some functions)', required=False)
    parser.add_argument('--output_dir',   nargs='?', action='append', help='Path to output dir (only used by some functions)', required=False)
    parser.add_argument('--df2tsv',   nargs='?', action='append', help='Loads DataFrame, saves as TSV', required=False)
    
    parser.add_argument('--for_claims_regex', help='Build list of regexes for claims', required=False, action='store_true')
    parser.add_argument('--split_sentences',  help='Splay documents out into individual sentences, one on each line', required=False, action='store_true')
    parser.add_argument('--read_claim_annotations',  help='Read claim annotation file', required=False, action='store_true')
    parser.add_argument('--print_with_pos',  help='Splay documents out into individual sentences, print with POS', required=False, action='store_true')
    parser.add_argument('--pairwise_agreement',  help='Study the pairwise interannotator agreement', required=False, action='store_true')
    parser.add_argument('--pairwise_agreement_within_file',  help='Study the pairwise interannotator agreement between columns within a single file', required=False, action='store_true')
    parser.add_argument('--pairwise_examples',  help='Obtain examples of annotator agreement', required=False, action='store_true')
    parser.add_argument('--binary',  help='Classification task at hand is binary (0,1)', required=False, action='store_true')
    parser.add_argument('--print_combined',  help='Print a combined file having both class columns', required=False, action='store_true')
    parser.add_argument('--sentence_pairs_by_entity',  help='Generate pairs of sentences sharing an entity', required=False, action='store_true')
    parser.add_argument('--generate_dataset_from_files',  help='Generate a dataset for HVC training from annotated files', required=False, action='store_true')
    parser.add_argument('--reload_dataset_from_file',  help='Reload a Dataset object', required=False, action='store_true')

    args = parser.parse_args()   # Get inputs and options

    options = {'normalize':True}

    if args.file  or  args.dir:

        # Split documents into sentences
        if args.split_sentences:
            for file in args.file:
                print_sentences(file)

        # Print sentences with POS
        elif args.print_with_pos:
            for file in args.file:
                print_sentences(file, {'with POS':True})
                        
        # Build List of regexes for claims
        elif args.for_claims_regex:
            for file in args.file:
                printed = {}
                print("claims_regexes = [ \\")
                for line in lemmatize_file(file):
                    line = re.sub(r'[^(\w )]', '', line)
                    if not line in printed:
                        print('    "'+line+'", \\')
                        printed[line] = True
                print("]")

        # Read a claims annotation file
        elif args.read_claim_annotations:
            read_claim_annotations(args.dir[0])

        # Compare interannotator agreement between annotation files in a pairwise manner
        elif args.pairwise_agreement:
            analyze_pairwise_agreement(args.file, args)
                
        # Compare interannotator agreement between annotation files in a pairwise manner
        elif args.pairwise_agreement_within_file:
            analyze_pairwise_agreement_within_file(args.file[0], args)
                
        # Obtain examples of interannotator agreement
        elif args.pairwise_examples:
            obtain_pairwise_examples(args.file, args)
                
        # Compare interannotator agreement between annotation files in a pairwise manner
        elif args.generate_dataset_from_files:
            generate_dataset_from_files(args.file[0], args)
                
        # Reload a Dataset object
        elif args.reload_dataset_from_file:
            file = args.file[0]
            dataset = deserialize(file)
            for document in dataset.documents:
                print('\n\nDOC:\n', document)
                
        # Find sentence pairs sharing an entity
        elif args.sentence_pairs_by_entity:
            for file in args.file:
                print_sentence_pairs_by_entity(file)

        # Print a combined version of multiple matching annotation files
        elif args.print_combined:
            print_combined_annotations(args.file, args)
                
        # Load a DataFrame file, save as a TSV
        elif args.df2tsv:
            save_dataframe_as_tsv(args.df2tsv[0])
                
        # Default: Lemmatize file
        else:
            spacy_nlp = spacy.load('en_core_web_md')
            for file in args.file:
                for line in lemmatize_file(file):
                    print(line)
    else:
        print(__doc__)

        
################################################################################
