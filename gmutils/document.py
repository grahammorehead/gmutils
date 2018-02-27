""" document.py

    Class to manage all internal elements of a document having an underlying Spacy Doc

"""
import os, sys, re, time
from copy import deepcopy
from collections import deque
import numpy as np
import pandas as pd

from gmutils.utils import err, argparser, deserialize, read_file, read_conceptnet_vectorfile
from gmutils import generate_spacy_data
from gmutils.objects import Object
from gmutils.normalize import normalize, clean_spaces, ascii_fold
from gmutils.node import Node, iprint

################################################################################

class Document(Object):
    """
    A document object that builds on top of the spaCy Doc and Spans.

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
        str version of Object in an easily-understood manner
        """
        text = self.get_text()
        
    
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

        
    def preprocess(self, vocab):
        """
        Collapse the parse tree to a more simplified format where sensible,  Identify the verb nodes and find their theta roles

        Parameters
        ----------
        vocab : dict { lemma string -> vector }

        """
        self.agglomerate_verbs_preps(vocab)
        self.agglomerate_entities()
        self.embed(vocab)

        
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


    def get_token_at_char_index(self, index):
        """
        Return the token containing character index 'index'

        """
        last_token = None
        for token in self.spacy_doc:
            if token.idx > index:
                return last_token
            last_token = token
            
        return last_token
        

    def tokens_matching(self, text, start_char=0):
        """
        Find a contiguous sequence of tokens matching the input string, starting at a specified CHARACTER index

        """
        verbose = False
        end_char = start_char + len(text)
        tokens = []
        words = text.split()
        start = self.get_token_at_char_index(start_char)
        end   = self.get_token_at_char_index(end_char)
        span = self.get_span(start.i, end.i)

        if text == span.text:                                      # Make sure we have the right tokens
            return span

        elif len(span.text) < len(text):                           # Tokenization left out the following token

            while len(span.text) < len(text):
                end_char += 1
                end   = self.get_token_at_char_index(end_char)
                if end is None:
                    break
                span = self.get_span(start.i, end.i)
                if text == span.text:                              # Try again
                    return span
            
        err([], {'ex':"TEXT [%s] doesn't match SPAN [%s]"% (text, span.text)})


    def old_way(self):
        if verbose:
            print("\nLooking for", words, 'in:', self.spacy_doc)
            
        j0 = token.i    # Token index at starting point
        for i, word in enumerate(words):
            j = j0 + i     # Corresponding token index in spacy_doc (i begins at 0)
            token = self.spacy_doc[j]
            if verbose:
                print("\tword %d: [%s]  \ttoken %d: [%s]"% (i, word, j, token.text))
            if word == token.text:
                tokens.append(token)
            else:
                err([], {'ex':"Token [%s] doesn't match word [%s]"% (token.text, word)})
                
        return tokens

            
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


def load_vocab(file):
    """
    Load word vectors
    """
    mult_vocab = deserialize(file)
    vocab = mult_vocab['en']
    return vocab


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser({'desc': "Document object: document.py"})
    parser.add_argument('--vocab', help='File with word embeddings', required=False, type=str)
    args = parser.parse_args()   # Get inputs and options
    vocab = load_vocab(args.vocab)
    
    if args.file:
        docs = []
        for file in args.file:
            docs.append( Document(text) )

    elif args.str:
        for text in args.str:
            doc = Document(text)
            doc.preprocess(vocab)
            doc.pretty_print()
            print('\nSEMANTIC ROLES:')
            doc.print_semantic_roles()
                
    else:
        print(__doc__)

            
################################################################################
################################################################################
