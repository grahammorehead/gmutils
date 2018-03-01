""" document.py

    Class to manage all internal elements of a document having an underlying Spacy Doc

"""
import os, sys, re, time
from copy import deepcopy
from collections import deque
import numpy as np
import pandas as pd

from gmutils.utils import err, argparser, deserialize, read_file, read_conceptnet_vectorfile, start_with_same_word
from gmutils.normalize import normalize, clean_spaces, ascii_fold, remove_brackets, ends_with_punctuation, close_enough, split_words
from gmutils.nlp import generate_spacy_data, tokenize
from gmutils.objects import Object
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


    def get_span(self, start_index, end_index):
        """
        Use indices to retrieve a slice of a spaCy Span.  Just like other Pythonic slicing, this Span does NOT include the final index!

        Parameters
        ----------
        start_index : int

        end_index : int

        Returns
        -------
        spaCy.Span

        """
        return self.spacy_doc[start_index:end_index]
        

    def next_token(self, token):
        """
        Return the following token in the document
        """
        if token is None:
            return None
        if token.i + 1 < len(self.spacy_doc):
            return self.spacy_doc[token.i + 1]
        return None
        
    
    def previous_token(self, token):
        """
        Return the previous token in the document
        """
        if token is None:
            return None
        if token.i > 0:
            return self.spacy_doc[token.i - 1]
        return None
        
    
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
            if token.idx >= index:
                return last_token
            last_token = token
            
        return last_token
        

    def nearby_matching_token(self, char_index, word):
        """
        Search near <char_index> for a token matching word.  Uses the 'close_enough' comparator.

        Begins with word at char_index in the spacy_doc, then looks ahead and behind, ever further, until the edges of the paragraph are found.

        """
        verbose = True
        token = self.get_token_at_char_index(char_index)

        if close_enough(token.text, word):
            return token
        elif verbose:
            err([token.text, word])

        # Setup for iteration that expands outward until token is found
        keep_going  = True
        previous    = token
        following   = token
        
        while keep_going:
            keep_going = False

            # Previous token
            previous = self.previous_token(previous)
            if previous is not None:
                if close_enough(previous.text, word):
                    return previous
                keep_going = True
                if verbose:
                    err([previous.text, word])

            # Following token
            following = self.next_token(following)
            if following is not None:
                if close_enough(following.text, word):
                    return following
                keep_going = True
                if verbose:
                    err([following.text, word])
                    
        err([], {'ex':"No matching token for [%s] in:\n\n%s\n"% (word, str(list(self.spacy_doc)))})
            
    
    def tokens_matching(self, text, start_char=0):
        """
        Find a contiguous sequence of tokens matching the input string, starting at a specified CHARACTER index
        """
        verbose = False
        tokens = set([])
        words = special_tokenize(text)
        
        char_index = start_char
        for word in words:
            token = self.nearby_matching_token(char_index, word)
            if token is None:
                err([list(self.spacy_doc)], {'ex':"WORD [%s] from TEXT [%s] at char %d resulted in None token."% (word, text, char_index)})
                
            if token not in tokens:
                tokens.add(token)
                for token in tokens:
                    if token is None:
                        err([list(self.spacy_doc)], {'ex':"WORD [%s] from TEXT [%s] at char %d resulted in None token."% (word, text, char_index)})
                    err(["WORD [%s] from TEXT [%s]"% (word, text)])
            char_index += len(word) + 1

        for token in tokens:
            if token is None:
                err([list(self.spacy_doc)], {'ex':"WORD [%s] from TEXT [%s] at char %d resulted in None token."% (word, text, char_index)})
            err(["TOKEN [%s]"% token.text])
        st = sorted(tokens, key=lambda x: x.i)
        start = st[0]
        end   = st[-1]
        span = self.get_span(start.i, end.i+1)
        
        if text == span.text:  # Make sure we have the right tokens
            return span
        
        if re.search(r'^[\.,;:!\?\'\"]*$', span[-1].text):  # If span puctuated at end, remove it
            span = span[:-1]
        
        if text == span.text:  # Try again
            return span

        err([self.spacy_doc], {'ex':"TEXT [%s] doesn't match SPAN [%s]"% (text, str(list(span)))})

            
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


def special_tokenize(text):
    """
    Tokenize and do a couple extra things
    """
    verbose = False

    if verbose:
        err([text])
    final = []
    for word in tokenize(text):
        if verbose:
            err([word])
        final.extend( split_words(word) )
    if verbose:
        err([final])
        
    return final
    

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
