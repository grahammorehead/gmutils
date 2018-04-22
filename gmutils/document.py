""" document.py

    Class to manage all internal elements of a document having an underlying Spacy Doc

"""
import os, sys, re, time
from copy import deepcopy
from collections import deque
import numpy as np
import pandas as pd

from gmutils.utils import err, argparser, deserialize, read_file, read_conceptnet_vectorfile, start_with_same_word, cosine_similarity, deepcopy_list
from gmutils.normalize import normalize, clean_spaces, ascii_fold, ends_with_punctuation, close_enough
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
        verbose = False
        self.set_options(options)
        
        # If reading the document from a file, <text> should be None
        if text is None:
            options['one str'] = True
            text = read_file(file, options=options)

        if self.get('normalize'):
            text = normalize(text, options=options)

        try:
            self.spacy_doc, self.ner = generate_spacy_data(text)   # Parse with spacy, get NER
            self.generate_trees()                                  # Generate Node trees representing sentences
        except:
            raise

        if verbose:
            self.print_sentences()
        
            
    def __repr__(self):
        """
        str version of Object in an easily-understood manner
        """
        return self.get_text()
        
    
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
        

    def get_sentences_by_breaks(self):
        """
        Instead of iterating over spacy.doc.sents attribute (unreliable), look at the .is_sent_start attributed (which we set)
        """
        sents = []
        ind   = []   # Indices observed in a sentence
        for i, token in enumerate(self.spacy_doc):
            if token.is_sent_start:
                if len(ind) > 0:
                    sents.append( self.spacy_doc[ind[0]:ind[-1]] )
                ind = [i]
            else:
                ind.append(i)
                
        if len(ind) > 0:
            sents.append( self.spacy_doc[ind[0]:ind[-1]] )
        
        return sents
    
    
    def generate_trees(self):
        """
        Parse doc into sentences, then generate a Node tree for each

        """
        verbose = False
        self.trees = []            # array of Node
        need_to_reparse = False
        spacy_sentences = self.get_sentences_by_breaks()
        for i, sen in enumerate(spacy_sentences):
            self.trees.append(Node(self, sen.root, options={'ID':'root.T'+str(i)}))
            

    def analyze_trees(self):
        """
        For the purpose of debugging, analyze trees
        """
        for tree in self.trees:
            tree.analyze()
        

    def disown(self, node):
        """
        Remove <node> from self.trees
        """
        self.trees.remove(node)
        

    def adopt(self, node):
        """
        Add <node> to self.trees
        """
        self.trees.append(node)
        

    def get_head_verb_nodes(self):
        """
        Look for a Node with the head verb.  Returns all reasonable options.
        """
        possible = []
        for tree in self.trees:
            if tree.is_verb():
                possible.append(tree)

        if len(possible) == 0:
            possible = self.trees
        
        final = []
        for p in possible:
            if p.get_lemmas_str() == 'be':
                pass
            else:
                final.append(p)

        if len(final) == 0:
            final = self.trees
        
        return final
                
                
    def get_head_verb_node(self):
        """
        Look for a Node with the head verb.  Returns first reasonable option.
        """
        possible = []
        for tree in self.trees:
            if tree.is_verb():
                possible.append(tree)
                
        for p in possible:
            if p.get_lemmas_str() == 'be':
                pass
            else:
                return p

        if len(possible) == 0:
            # Ignore for now, but should deal with later
            # self.pretty_print()
            # err([self.get_text()], {'ex':"Couldn't find head verb node!  Providing root in its stead.", 'level':1})
            return self.trees[0]
        
        else:
            return possible[0]   # Of the head verbs found, return the first one

            
    def get_all_pos(self):
        """
        Get all POS from this Document
        """
        pos = set([])
        for tree in self.trees:
            pos.update(tree.get_all_pos())

        return pos
            

    def get_all_ner(self):
        """
        Get all NER from this Document
        """
        ner = set([])
        for tree in self.trees:
            ner.update(tree.get_all_ner())

        return ner


    def get_all_dep(self):
        """
        Get all DEP from this Document
        """
        dep = set([])
        for tree in self.trees:
            dep.update(tree.get_all_dep())

        return dep
            

    def get_verbs(self):
        """
        From each of the constituent trees, return a list of all nodes that are verbs
        """
        verbs = []
        for tree in self.trees:
            verbs.extend( tree.get_verbs() )
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
                for node in tree.get_entity_beginners():
                    a = node.agglomerate_entities()
                    if a:  altered = a  # only switch if going to True
        

    def agglomerate_idioms(self):
        """
        For the purpose of dealing sensibly with extracted meaning, agglomerate tokens from a single idiom into a node
        """
        altered = True
        while altered:
            altered = False
            for tree in self.trees:
                for node in tree.get_idiom_parents():
                    a = node.agglomerate_idiom()
                    if a:  altered = a  # only switch if going to True
        

    def agglomerate_verbs_preps(self, vocab=None):
        """
        For the purpose of sense disambiguation, agglomerate verbs with prepositional children

        e.g. If "jump" is used to describe A jumping over B, the real sense of the verb is "jump over"

        When an embedding (vocab) is provided, it will be consulted for the best embedding before agglomeration.

        """
        altered = True
        while altered:
            altered = False
            for tree in self.trees:
                for node in tree.get_verbs():
                    a = node.agglomerate_verbs_preps(vocab=vocab)
                    if a:  altered = a  # only switch if going to True
        

    def agglomerate_compound_adj(self, vocab=None):
        """
        For the purpose of sense disambiguation, agglomerate compound adjectives, like "full-time", and "cross-country".

        When an embedding (vocab) is provided, it will be consulted before agglomeration for the best embedding.

        """
        altered = True
        while altered:
            altered = False
            for tree in self.trees:
                for node in tree.get_compound_prefixes():
                    a = node.agglomerate_compound_adj()   #  vocab=vocab)  No vocab for now!
                    if a:  altered = a  # only switch if going to True


    def agglomerate_modifiers(self):
        """
        For the purpose of tree simplification (lower branching factor), absorb childless modifiers into their parents
        """
        altered = True
        while altered:
            altered = False
            for tree in self.trees:
                for node in tree.get_modifiers():
                    a = node.agglomerate_modifier()
                    if a:  altered = a  # only switch if going to True


    def agglomerate_twins(self):
        """
        For the purpose of tree simplification (lower branching factor), absorb childless modifiers into their parents
        """
        altered = True
        while altered:
            altered = False
            for tree in self.trees:
                for node in tree.get_parents_of_twins():
                    a = node.agglomerate_twins()
                    if a:  altered = a  # only switch if going to True


    def agglomerate_verbauxes(self):
        """
        For the purpose of tree simplification (lower branching factor), absorb verb auxilaries
        """
        altered = True
        while altered:
            altered = False
            for tree in self.trees:
                for node in tree.get_verbs():
                    a = node.agglomerate_verbaux()
                    if a:  altered = a  # only switch if going to True


    def delegate_to_conjunctions(self):
        """
        For the purpose of tree simplification (lower branching factor), and logical faithfulness, take conjunction arguments and bring them in under
        the conjunction node.
        """
        verbose = False
        altered = True
        while altered:
            altered = False
            if verbose:  err([altered])
            for tree in self.trees:
                for node in tree.get_conjunctions():
                    if verbose:  err([tree, "calling:", node])
                    a = node.delegate_to_conjunction()
                    if a:  altered = a  # only switch if going to True
                if verbose:  err([tree, altered])


    def delegate_to_negations(self):
        """
        For the purpose of logical faithfulness, delegate the subject of a negation under it.
        """
        altered = True
        while altered:
            altered = False
            for tree in self.trees:
                for node in tree.get_negations():
                    a = node.delegate_to_negation()
                    if a:  altered = a  # only switch if going to True


    def embed(self, vocab):
        """
        Use a given vocab to embed each node in some vector space
        """
        for tree in self.trees:
            tree.embed(vocab)
                        

    def get_embedding(self, options={}):
        """
        After a given embedding (vocab) has already been used to vectorize each node, use this method to compile it together.

        Options
        -------
        ID : str
            Of the format: "root.T1.2.3.x etc." where each dot implies a level down, and the integer indicates sibling number

        """
        ems = []
        tree_num = 0
        for tree in self.trees:
            em = tree.get_tree_embedding(options)
            if em:
                ems.append(em)
            tree_num += 1
            
        return ems
    
        
    def node_by_ID(self, ID):
        """
        Search for and return the Node having a given ID string.  Begin at the root of each tree in this Document.  There is a pattern to
        the ID-naming of each node.  This pattern is followed for efficient retrieval.  The desired node will be in the subtree of a given
        node IFF this node's ID is a prefix of <ID>.

        Parameters
        ----------
        ID : str

        Returns
        -------
        Node

        """
        for tree in self.trees:
            node = tree.node_by_ID(ID)
            if node is not None:
                return node

        
    def preprocess(self, vocab):
        """
        Collapse the parse tree to a more simplified format where sensible,  Identify the verb nodes and find their theta roles

        Parameters
        ----------
        vocab : dict { lemma string -> vector }

        """
        verbose = False
        # self.agglomerate_verbs_preps(vocab)
        if verbose:  err()
        self.agglomerate_compound_adj(vocab)
        if verbose:  err()
        self.agglomerate_entities()
        if verbose:  err()
        self.delegate_to_negations()
        if verbose:  err()
        self.agglomerate_modifiers()
        if verbose:  err()
        self.agglomerate_twins()
        if verbose:  err()
        self.agglomerate_verbauxes()
        if verbose:  err()
        self.delegate_to_conjunctions()
        if verbose:  err()
        self.agglomerate_idioms()
        if verbose:  err()
        self.analyze_trees()                    # For debugging
        if verbose:  err()
        self.embed(vocab)
        if verbose:  err()

        
    def pretty_print(self, options={}):
        """
        Print parsed elements in an easy-to-read format
        """
        for tree in self.trees:
            print("\nSENTENCE:", tree.get_supporting_text(), "\n")
            tree.pretty_print(options=options)


    def print_sentences(self, options={}):
        """
        Print the supporting text for each tree
        """
        for i, tree in enumerate(self.trees):
            print(i, ":", tree.get_supporting_text())
            

    def print_semantic_roles(self):
        """
        Get the verb Nodes and print something like a theta-role breakdown for each
        """
        for node in self.get_verbs():
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
        

    def char_span(self, start, end):
        """
        Get the span from start/end char indices

        Parameters
        ----------
        start : int

        end : int

        Returns
        -------
        spacy.Span

        """
        verbose = False
        span = self.spacy_doc.char_span(start, end)
        if span is None:
            start_token = self.get_token_at_char_index(start)
            end_token   = self.get_token_at_char_index(end)
            if verbose:
                err([start_token.i, end_token.i, end_token.text])
            if start_token.i == end_token.i:
                span = self.spacy_doc[start_token.i:start_token.i+1]
            else:
                span = self.spacy_doc[start_token.i:end_token.i+1]     # Adding 1 here very important

        return span
        
    
    def text_to_char_offsets(self, text, start_char=0):
        """
        Search in the text of this Document for a substring matching text.  If more than one is found, select the one having a first character
        closest to start_char.

        Parameters
        ----------
        text : str

        start_char : index

        Returns
        -------
        pair of int
            start/end char offsets

        """
        verbose = False

        text             = re.escape(text)
        lowest_distance  = None
        best_span        = None
        
        # Iterate over matches.  Select one closest to start_char
        for m in re.finditer(text, self.get_text(), flags=re.I):
            span = m.span()
            distance = abs(span[0] - start_char)
            if lowest_distance is None  or  distance < lowest_distance:
                lowest_distance = distance
                best_span = span

        if best_span is None:
            err([], {'ex':"No best span for [%s] in:\n%s"% (text, self.get_text())})
                
        return best_span  # pair of (int, int), NOT a spacy.Span object


    def get_nodes(self):
        """
        Return all nodes under this Docment, in order of lowest token index
        """
        nodes = set([])
        for tree in self.trees:
            nodes.update(tree.get_nodes())

        final = []
        for node in nodes:
            if len(node.tokens) > 0:
                final.append(node)

        return sorted(final, key=lambda x: x.get_index())
        

    def get_num_nodes(self):
        """
        Return int number of nodes in this Document
        """
        if not self.done():
            nodes = self.get_nodes()
            self.set('num_nodes', len(nodes))

        return self.get('num_nodes')
    
    
    def get_related_nodes(self, head, thresh=0.7):
        """
        For a given node (not from this Document), sort this Document's nodes by embedding similarity.
        """
        sim = {}
        for node in self.get_nodes():
            s = head.cosine_similarity(node)
            if s > thresh:
                # print("(%0.5f)  %s  <=>  %s"% (s, node.get_text(), head.get_text())) 
                sim[node] = s

        rels = sorted(sim.keys(), key=lambda x: sim[x], reverse=True)

        return rels

        
        
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
    if file is None:
        file = os.environ['HOME'] + '/data/ConceptNet/numberbatch_en.pkl'
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
            print("\nTEXT:", doc.get_text())
            doc.pretty_print(options={'supporting_text':True})
            print("LEMMAS:", doc.get_lemmas())
            #doc.print_semantic_roles()
                
    else:
        print(__doc__)

            
################################################################################
################################################################################
