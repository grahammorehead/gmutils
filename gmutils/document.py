""" document.py

    Class to manage all internal elements of a document having an underlying Spacy Doc

"""
import os, sys, re, time
from copy import deepcopy
from collections import deque
import numpy as np
import pandas as pd

from gmutils.utils import err
err()
from gmutils.utils import argparser, read_file, read_conceptnet_vectorfile
err()
from gmutils import spacy_nlp
err()
from gmutils.objects import Object
err()
from gmutils.normalize import normalize, clean_spaces, ascii_fold
err()
from gmutils.node import Node, iprint
err()

################################################################################

class Document(Object):
    """
    A document object that builds on top of the spaCy Doc and Spans.
    Provides easy access to paragraphs and sentences.

    Attributes
    ----------
    spacy_docs : array of spacy.Doc
        The underlying spacy Doc object(s).  More than one is generated only when spacy makes mistakes on sentence tokenization.

    trees : array of Node
        Each of these Node objects represents the root of a parse tree

    """
    def __init__(self, text=None, file=None, options={}):
        """
        Instantiate a single Document, either from a long string, or the contents of a file.

        Parameters
        ----------
        text : str

        file : str

        options : dict or namespace

        """
        self.set_options(options)
        
        # If reading the document from a file, <text> should be None
        if text is None:
            options['one str'] = True
            text = read_file(file, options)

        if self.get('normalize'):
            text = normalize(text, {'verbose':False})

        self.spacy_docs = [ spacy_nlp(text) ]   # Parse with spacy
        self.generate_trees()               # Generate Node trees representing sentences, then do some checks

            
    def __repr__(self):
        """
        Print Document in an easily-understood manner
        """
        return self.spacy.text

    
    def __str__(self):
        return self.__repr__()

    
    def get_text(self):
        return self.spacy.text
    

    def get_lemmas(self):
        return self.spacy[:].lemma_
    

    def combine_with_previous(self, previous, current):
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
        
    
    def generate_trees(self):
        """
        Parse doc into sentences, then generate a Node tree for each

        """
        verbose = False
        sentences = []        # array of spacy.Span  (might not need this ...)
        self.trees = []       # array of Node
        
        ##  Iterate over the spaCy sentence boundaries collecting offset pairs of Sentence candidates
        sen_offsets = []  # Sentence offset pairs
        sen_spans = []

        # Iterate over sentences as tokenized by spacy.  Correct and reparse if needed
        checked_docs = []   # spacy_docs after checking sentences
        for doc in self.spacy_docs:
            need_to_reparse = False
            spacy_sentences = list(doc.sents)
            for i,sen in enumerate(spacy_sentences):
                
                # Current Sentence
                start = sen.start
                end = sen.end
                current = doc[start:end]

                # Previous Sentence
                previous = None
                if i > 0:
                    prev = spacy_sentences[i-1]
                    p_start = prev.start
                    p_end = prev.end
                    previous = doc[p_start:p_end]
                    if verbose:
                        err([[previous.text, current.text]])

                # Correct for mistakes
                if previous is not None  and  self.combine_with_previous(previous, current):
                    need_to_reparse = True
                    sen_spans[-1] = doc[sen_offsets[-1][0]:end]
                    sen_offsets[-1][1] = end
                else:
                    sen_offsets.append( [start, end] )
                    sen_spans.append( doc[start:end] )

            if need_to_reparse:                        # Sentence tokenization failed, this is a last ditch effort
                for span in sen_spans:
                    doc2 = spacy_nlp(span.text)        # Reparse text of this sentence as a single sentence (with potentially less other text around it)
                    checked_docs.append(doc2)          # Append to list of docs
                    span2 = doc2[:]
                    sentences.append(span2)     # Take entirety of this new doc as the sentence span
                    self.trees.append(Node(doc2, span2.root))
                    
            else:
                checked_docs.append(doc)
                for span in sen_spans:
                    sentences.append(span)
                    self.trees.append(Node(doc, span.root))

        self.spacy_docs = checked_docs


    def get_verb_nodes(self):
        """
        From each of the constituent trees, return a list of all nodes that are verbs

        """
        verbs = []
        for tree in self.trees:
            verbs.extend( tree.get_verb_nodes() )
        return verbs


    def agglomerate_entities(self):
        """
        For the purpose of dealing sensibly with extracted entities, agglomerate tokens from a single entity into a node

        """
        altered = True
        while altered:
            for tree in self.trees:
                beginners = tree.get_entity_beginners()
                for node in beginners:
                    altered = node.agglomerate_entities()
        

    def agglomerate_verbs_preps(self, vocab=None):
        """
        For the purpose of sense disambiguation, agglomerate verbs with prepositional children

        e.g. If "jump" is used to describe A jumping over B, the real sense of the verb is "jump over"

        When an embedding (vocab) is provided, it will be consulted before agglomeration

        """
        altered = True
        while altered:
            for tree in self.trees:
                altered = tree.agglomerate_verbs_preps(vocab=vocab)
        

    def embed(self, vocab, options={}):
        """
        Use a given embedding (vocab) to vectorize this node and its children

        """
        # Recursively embed each tree in this vector space
        for tree in self.trees:
            tree.embed(vocab)

        
    def pretty_print(self):
        """
        Print parsed elements in an easy-to-read format

        """
        print('\nSENTENCE:')
        for tree in self.trees:
            tree.pretty_print(options={'supporting text':False})

        # Head semantic roles -ish (some semblance thereof)
        # for tree in self.trees:
        #    tree.print_semantic_roles()
        
err()
        
################################################################################
##  FUNCTIONS

def generate_documents(input, options={'normalize':True}):
    documents = []

    it = str(type(input))
    logger.info(" >>> input type: %s\n", it)
    
    if isinstance(input, pd.core.frame.DataFrame):
        for index, row in input.iterrows():
            documents.append( Document(text=row['content'], options=options) )

    elif isinstance(input, list):
        for text in input:
            documents.append( Document(text=text, options=options) )

    else:  # Default: input is a str
        documents.append( Document(text=input, options=options) )
            
    return documents
    

################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser({'desc': "Document object: document.py"})
    parser.add_argument('--embed', help='Embed a Document in a vector space', required=False, action='store_true')
    args = parser.parse_args()   # Get inputs and options

    if args.file:
        docs = []
        for file in args.file:
            docs.append( Document(text) )

    elif args.str:
        for text in args.str:
            doc = Document(text)
            doc.agglomerate_verbs_preps()
            doc.pretty_print()
            if args.embed:
                doc.embed()
                
    else:
        print(__doc__)

            
################################################################################
################################################################################
