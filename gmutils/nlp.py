""" nlp.py

Perform some basic nlp tasks

"""
import sys
import os, re
import numpy as np
import itertools
from copy import deepcopy
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet

from gmutils.objects import Options
from gmutils.normalize import normalize, naked_words
from gmutils.utils import err, argparser, read_file, read_dir, iter_file, isTrue, monitor_setup, monitor, serialize, deserialize

try:
    import pandas as pd
except Exception as e: err([], {'exception':e, 'level':0})
    
################################################################################
# SPACY INTEGRATION

def set_sentence_starts(doc):
    """
    Adjust the elements in a spaCy Doc to record the sentence starts as output by sentence_fixer()

    This function is designed to be a spaCy pipeline element
    """
    verbose = False
    starts = set([])
    
    for offsets in find_sentence_offsets(doc):
        if verbose:  err([offsets])
        start, end = offsets
        starts.add(start)

    for token in doc[:-1]:
        if token.i in starts:
            token.is_sent_start = True
            if verbose:  err(["START", token])
        else:
            token.is_sent_start = False
            if verbose:  err([token])
            
    return doc


spacy_parsing = spacy_ner = None

try:
    spacy_ner     = spacy.load('en_core_web_lg')    # download separately: https://spacy.io/models/
    if not os.environ.get('GM_NERONLY'):
        spacy_parsing = spacy.load('en_core_web_lg')
        
    """ POST-DEFINITIONAL LOADING

        Add elements to the parsing pipeline to correct for the sentence tokenizer problems

    This section attempts to directly alter the spacy pipeline so that it tags as correctly as possible.  The spacy sentencizer makes some
    obvious mistakes.

    After the end of the pipeline, some new errors are introduced and it is not known where, so the custom 'set_sentence_starts' will be applied again
    to the result, but not here.

    """
    if not os.environ.get('GM_NERONLY'):
        spacy_parsing.add_pipe(set_sentence_starts, name='set_sentence_starts', before='tagger')
        spacy_parsing.add_pipe(spacy_parsing.create_pipe('sentencizer'), before='set_sentence_starts')
    tokenizer = spacy.tokenizer.Tokenizer(spacy_ner.vocab)
    
except:
    raise
    err([], {'exception':e})

################################################################################

def get_sentences(doc):
    """
    Even after adding a custom func to the spacy pipeline, the sentence tokenization still gets messed up.  Use this for a final split.
    """
    verbose       = False
    starts        = set([])
    sentences     = []
    this_sentence = set([])
    """
    for offsets in find_sentence_offsets(doc):
        if verbose:  err([offsets])
        start, end = offsets
        starts.add(start)
    """
    for i, token in enumerate(doc[:-1]):
        # if token.i in starts:
        if token.is_sent_start:
            if verbose:  err(["START", token])
            if len(this_sentence) > 0:
                sentences.append( doc[min(this_sentence):max(this_sentence)] )
            this_sentence = set([i])
        else:
            if verbose:  err([token])
            this_sentence.add(i)

    if len(this_sentence) > 0:
        sentences.append( doc[min(this_sentence):max(this_sentence)] )

    return sentences
            

def sanity_check(spacy_doc):
    """
    To use for checking a parse
    """
    verbose = False
    for i, token in enumerate(spacy_doc):
        if token.is_sent_start:
            if verbose:  err(["START", token])
        else:
            if verbose:  err([token])


def generate_spacy_data(text):
    """
    Used in the creation of Document objects

    Parameters
    ----------
    text : str

    Returns
    -------
    spaCy Doc

    """
    # Perform NER corrections
    # err([text])
    spacy_doc = spacy_parsing(text)
    spacy_nerdoc = spacy_ner(text)
    assert( len(spacy_doc) == len(spacy_nerdoc) )
    ner = {}
    for i, token in enumerate(spacy_doc):
        ner[token.i]  = (spacy_nerdoc[i].ent_type_, spacy_nerdoc[i].ent_iob_)

    return spacy_doc, ner, spacy_ner.vocab


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
    if verbose:
        err([previous.text, previous.end - previous.start, current.text, current.end - current.start])

    # This sentence too short and not capitalized or previous is a paren
    if current.end - current.start < 3  and  ( current.text[0].islower()  or  re.search(r"\)", previous.text) ):
        if verbose:
            err([[current.text]])
        return True

    # This sentence moderately short and has a close paren
    if current.end - current.start < 7  and  re.search(r"\)", current.text):
        if verbose:
            err([[current.text]])
        return True

    # Previous sentence too short and is capitalized
    if previous.end - previous.start < 3  and  previous.text[0].isupper():
        if verbose:
            err([[previous.text]])
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
    
    
def find_sentence_offsets(doc):
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
    sen_offsets = []                                   # Sentence offset pairs -- starts empty
    spacy_sentences = list(doc.sents)
    previous = None
    
    for i, current in enumerate(spacy_sentences):

        # Current Sentence
        start = current.start
        end   = current.end
        if verbose:  err([current, start, end])

        # Previous Sentence
        if len(sen_offsets) > 0:
            p_start = previous.start
            p_end   = previous.end
            if verbose:
                err([previous.text, (p_start, p_end), current.text, (start, end)])

        # Correct for mistakes
        if previous is not None  and  combine_with_previous(previous, current)  and  False:      ### NEVER
            sen_offsets[-1] = [p_start, end]           # Fold this sentence's tokens into previous sentence
            if verbose:  err(sen_offsets)
        else:
            sen_offsets.append( [start, end] )         # Add current offsets to list
            if verbose:  err(sen_offsets)

        previous = current
                
    # Shift sentence starting token to previous sentence when necessary
    final_offsets = []
    for offsets in sen_offsets:
        start, end = offsets
        if start > 0:            # Not the first sentence start
            if not re.search(r'\S', doc[start].text, flags=re.I):   # Current sentence starts with whitespace
                prev_start, prev_end = final_offsets[-1]
                final_offsets[-1] = [prev_start, prev_end+1]   # Make previous sentence longer
                final_offsets.append( [start+1, end] )         # Start current sentence one char later
            else:
                final_offsets.append(offsets)
        else:
            final_offsets.append(offsets)
                
    sen_offsets = final_offsets
    
    if verbose:
        err(sen_offsets)
    return sen_offsets
    

################################################################################
# FUNCTIONS

def split_words(text):
    """
    Poor man's tokenization
    """
    verbose = False
    words = text.split()
    
    ready = []
    for word in words:
        if re.search(r'[a-zA-Z]-[a-zA-Z]', word):       # Handle hyphens
            parts = word.split('-')
            ready.append(parts[0])
            for part in parts:
                ready.append('-')
                ready.append(part)
        else:
            ready.append(word)
    if verbose: err([ready])
    words = ready
    
    ready = []
    for word in words:
        if re.search(r"\w'\w+$", word):                # Handle apostrophes
            starting = re.sub(r"'(\w+)$", '', word)
            ending   = re.sub(r"^.*'(\w+)$", r'\1', word)
            ready.extend( [starting, "'" + ending] )
        else:
            ready.append(word)
    if verbose: err([ready])
    words = ready
    
    return words
    

def tokenize_spacy(text):
    """
    Tokenize without handling most of the structure

    Parameters
    ----------
    text : str

    Returns
    -------
    array of str

    """
    final = []
    for token in tokenizer(text):
        final.append(token.text)
    return final

    
def tokenize(text):
    """
    Tokenize and do a couple extra things
    """
    verbose = False

    if verbose:
        err([text])
    final = []
    for word in tokenize_spacy(text):
        if verbose:
            err([word])
        final.extend( split_words(word) )
    if verbose:
        err([final])
        
    return final
    

def lemmatize(text):
    spacy_doc = spacy_parsing(text)
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
    spacy_doc, ner, vocab = generate_spacy_data(text)   # Parse with spacy, get NER
    spacy_sentences = list(spacy_doc.sents)
    trees = []
    for i, sen in enumerate(spacy_sentences):
        err([sen.root])

    return spacy_doc


def generate_onehot_vocab(words):
    """
    For a list of words, generate a one-hot vocab of the appropriate size
    """
    vocab = {}
    vocab['_empty_'] = np.array([0] * len(words))
    for i, word in enumerate(words):
        vocab[word] = deepcopy(vocab['_empty_'])
        vocab[word][i] = 1
        
    return vocab


def words_below_freq(lines, freq=0.01):
    """
    Take a list of lines of text.  Return a set of all words below a certain frequency
    """
    words = {}
    N     = 0    # total occurrences of any word
    for line in lines:
        for w in naked_words(line):
            N += 1
            if w in words:
                words[w] += 1   # increment number of occurrences of w
            else:
                words[w]  = 1

    output = set([])
    N = float(N)
    for word,occ in sorted(words.items(), key=lambda x: x[1]):
        f = occ/N
        print("\t(%0.4f)  %s"% (f, word))
        if f < freq:
            output.add(word)
        else:
            print("HF word:", word)

    return output


def get_synonyms(lemma):
    """
    Return a list of NLTK Sentiword synsets for 'word'

    Parameters
    ----------
    word : str
        For best results 'word' will be lemmatized in a way consistent with its POS

    pos : str
        Part of Speech
        Used to get the lemma, but may also be used for future synset generators
    """
    output = []
    synsets = wordnet.synsets(lemma)
    for synset in synsets:
        for sl in synset.lemmas():
            output.append(sl.name())
    return output

                
################################################################################
# MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "Some general NLP tasks: nlp.py"})
    parser.add_argument('--synset', help='Get the WordNet synonym set for a word', required=False, type=str)
    args = parser.parse_args()

    text = ''
    
    if args.file:
        for file in args.file:
            text += read_file(file)
            
    elif args.str:
        text = '  '.join(args.str)
        
    elif args.synset:
        lemma = lemmatize(args.synset)
        synset = get_synonyms(lemma)
        print(synset)
        exit()
        
    else:
        print(__doc__)
        exit()

    # text = normalize(text)
    parsed_text = parse(text)
    print(parsed_text)
    
        
################################################################################
################################################################################
