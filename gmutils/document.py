""" document.py

    Class to manage all internal elements of a document having an underlying Spacy Doc

"""
import os, sys, re, time
from copy import deepcopy
from collections import deque
import numpy as np
import pandas as pd

from gmutils.utils import err
from gmutils.utils import argparser, read_file, read_conceptnet_vectorfile
from gmutils import generate_spacy_data
from gmutils.objects import Object
from gmutils.normalize import normalize, clean_spaces, ascii_fold
from gmutils.node import Node, iprint

################################################################################

class Document(Object):
    """
    A document object that builds on top of the spaCy Doc and Spans.
    Provides easy access to paragraphs and sentences.

    Attributes
    ----------
    spacy_doc : spaCy.Doc
        The underlying spacy Doc object

    ner : dict
        token.i : (ent_type_, ent_iob_)
        This is needed because the parse pipeline doesn't do NER well.  Must be handled separately.

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
        if self.get('remove_brackets'):
            text = remove_brackets(text)
            
        self.spacy_doc, self.ner = generate_spacy_data(text)   # Parse with spacy, get NER
        self.generate_trees()                                  # Generate Node trees representing sentences

            
    def __repr__(self):
        """
        Print Document in an easily-understood manner
        """
        return self.spacy_doc[:].text

    
    def __str__(self):
        return self.__repr__()

    
    def get_text(self):
        return self.spacy_doc[:].text
    

    def get_lemmas(self):
        return self.spacy_doc[:].lemma_
    

    def get_num_tokens(self):
        return len(self.spacy_doc)
    

    def token_by_index(self, i):
        return self.spacy_doc[i]


    def get_span(self, start, end):
        return self.spacy_doc[start:end]
        
    
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
        self.trees = []            # array of Node
        need_to_reparse = False
        spacy_sentences = list(self.spacy_doc.sents)
        for i,sen in enumerate(spacy_sentences):
            self.trees.append(Node(self, sen.root))


    def print_sentences(self):
        """
        Print the supporting text for each tree
        """
        for tree in self.trees:
            print('\nNEXT SENTENCE:')
            print (tree.get_supporting_text())
        

    def get_verb_nodes(self):
        """
        From each of the constituent trees, return a list of all nodes that are verbs

        """
        verbs = []
        for tree in self.trees:
            verbs.extend( tree.get_verb_nodes() )
        return verbs


    def print_entity_status(self):
        for token in self.spacy_doc:
            ent_type, iob = self.ner[token.i]
            print(token, iob)
            

    def agglomerate_entities(self):
        """
        For the purpose of dealing sensibly with extracted entities, agglomerate tokens from a single entity into a node

        """
        altered = True
        while altered:
            altered = False
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
            altered = False
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
        print('\nPARSED SENTENCE:')
        for tree in self.trees:
            tree.pretty_print(options={'supporting text':False})


    def print_semantic_roles(self):
        """
        Get the verb Nodes and print something like a theta-role breakdown for each

        """
        for node in self.get_verb_nodes():
            node.print_semantic_roles()

            
################################################################################
##  FUNCTIONS

def generate_documents(input, options={'normalize':True, 'remove_brackets':True}):
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
