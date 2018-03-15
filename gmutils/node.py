""" node.py

Tools to manage nodes in a parse tree

"""
import os, sys, re, json
from copy import deepcopy
import numpy as np

from gmutils.objects import Object
from gmutils.utils import err, argparser, vector_average, cosine_similarity
from gmutils.nlp import generate_onehot_vocab

################################################################################
# DEFAULTS

pos_indices = ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SYM', 'VERB', 'X']

ner_indices = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

dep_indices = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']
    
default = {
    'pos_embedding' : generate_onehot_vocab(pos_indices),
    'ner_embedding' : generate_onehot_vocab(ner_indices),
    'dep_embedding' : generate_onehot_vocab(dep_indices),
    'compound_adj_prefixes' : set(['all', 'cross', 'full', 'part', 'half', 'high', 'low', 'upper', 'lower', 'middle', 'mid', 'like', 'self'])
    }

################################################################################
# OBJECTS

class Node(Object):
    """
    A node in a dependency parse tree.  Based on an underlying Spacy Doc

    Attributes
    ----------
    is_dead : boolean
        Used to ensure activity by absorbed nodes

    doc : Document object (as defined in document.py)

    tokens : array of spacy.Token

    parent : Node

    children : array of Node

    embedding : array of float
        An embedding in some defined vector space

    """
    def __init__(self, doc, spacy_token, parent=None, options={}):
        """
        Instantiate the object and set options

        doc : Document (as defined in document.py)
        
        spacy_token : spacy.Token

        parent : Node

        """
        self.set_options(options)        # For more on 'self.set_options()' see object.Object
        self.is_dead = False
        self.doc = doc

        if isinstance(spacy_token, list):
            self.tokens = spacy_token
        else:
            self.tokens = [spacy_token]
        
        self.parent = parent
        self.children = []
        
        # Base original tree on spacy Token tree.  Add a Node-child for every token-child
        if len(self.tokens) < 1:
            err([], {'ex':"No tokens in this Node. See parent:\n%s\n\nSee Document:\n%s"% (self.parent, self.doc)})
        for token in self.tokens:
            for child in token.children:
                self.children.append(Node(self.doc, child, parent=self, options=options))

                
    def kill(self):
        """
        Remove a Node and make sure it doesn't get used accidentally
        """
        if self.is_dead:
            err([], {'ex':"ERROR: Node was already dead!"})
            
        self.is_dead = True
        self.tokens = self.children = self.embedding = None
        
                
    def __repr__(self):
        return self.get_text()
        
    ############################################################################
    # ALTERATIONS

    def adopt(self, node):
        """
        Create both sides of parent-child relationships
        """
        verbose = False
        
        if isinstance(node, list):
            for n in node:
                self.adopt(n)
        else:
            if node in self.children:
                return
            if node == self:
                return
            if verbose:  print(self, 'adopting', node)
            node.parent = self
            self.children.append(node)
    

    def disown(self, node):
        """
        Break both sides of parent-child relationship
        """
        self.children.remove(node)
        node.parent = None

        
    def absorb(self, node):
        """
        Merge one Node with another.  Afterwards, nothing should link to 'node', only to 'self'

        """
        verbose = False
        
        if self == node:
            err([self], {'ex':'node = self'})   # Sanity check

        if self.is_descendant(node):
            self.absorb_descendant(node)

        elif self.is_ancestor(node):
            self.absorb_ancestor(node)

        else:
            self.absorb_cousin(node)   # anywhere else on same tree

        if verbose:
            print("\n[%s] done absorbing."% str(self))
            if len(self.children):
                print('\t(children -after):', self.children)
            else:
                print("\t(self has no children)")
        
        # Final sanity check
        for child in self.children:
            if child == self:
                err([self], {'ex':'child = self'})

                
    def absorb_descendant(self, node):
        """
        Merge descendant Node into self
        """
        parent = node.parent               # Might be self
        parent.disown(node)                # Separate child from old parent
        self.tokens.extend(node.tokens)    # Absorb their tokens
        self.adopt(node.children)          # Adopt their children
        node.kill()


    def absorb_parent(self):
        """
        Merge parent Node into self
        """
        if self.is_root():                     # self is root.  There is no parent
            pass
        
        elif self.parent.is_root():            # parent is root
            node = self.parent
            self.doc.disown(node)              # take parent out of the list of doc.trees
            self.doc.adopt(self)               # add self to that list
            self.tokens.extend(node.tokens)    # Absorb node's tokens
            self.adopt(node.children)          # Adopt node's children (ignores self, of course)
            self.parent = None                        # self is the new root
            
        else:
            node = self.parent
            grandparent = node.parent          # grandparent of self is new parent
            grandparent.disown(node)
            self.tokens.extend(node.tokens)    # Absorb node's tokens
            self.adopt(node.children)          # Adopt node's children (ignores self, of course)
            grandparent.adopt(self)            # New parental relationship with grandparent
            node.kill()

            
    def absorb_ancestor(self, node):
        """
        Merge ancestor Node into self
        """
        if node == self.parent:                # Base case
            return self.absorb_parent()
        
        else:                                  # must be grandparent or more senior
            self.absorb_parent()               # Absorb <node>, then continue up toward root
            return self.absorb_ancestor(node)  # Recursion


    def absorb_cousin(self, node):
        """
        Merge a node from another part of the tree into self
        """
        old_parent = node.parent
        old_parent.disown(node)                # Cut old parental ties
        self.tokens.extend(node.tokens)        # Absorb their tokens
        self.adopt(node.children)              # Adopt their children (if any)
        node.kill()

        
    def agglomerate_entities(self, options={}):
        """
        For the purpose of dealing sensibly with extracted entities, agglomerate tokens from a single entity into a node.

        No need to apply this function recursively as it is applied separately to all Nodes that represent the beginning
        of an entity.

        """
        altered = False
        if self.is_dead:
            return altered

        if self.get_entity_position() == 'B':
            next_token = self.get_next_token()            # Get token immediately following last token of this Node
            next_node = self.node_of_token(next_token)    # Get the Node containing that token
            if next_node is not None:
                if next_node.get_entity_position() == 'I':
                    self.absorb(next_node)
                    altered = True

        return altered


    def agglomerate_verbs_preps(self, vocab=None, options={}):
        """
        For the purpose of sense disambiguation, agglomerate verbs with prepositional children

        e.g. If "jump" is used to describe A jumping over B, the real sense of the verb is "jump over"

        """
        altered = False
        if self.is_dead:
            return altered
            
        if self.is_leaf():
            return altered

        # Select which children, if any, to absorb
        to_absorb = []
        if self.is_verb:
            for child in self.children:
                if 'prep' in child.get_dep():  # Only looking to agglomerate nodes with a 'prep' dependency
                    if vocab is not None:       # Only allow agglomerations corresponding to a vocab entry
                        if not combined_lemmas_in_vocab(vocab, self.get_lemmas_str(), child.get_lemmas_str()):
                            continue
                    to_absorb.append(child)     # Add to the list of children to absorb

        # Execute absorptions in order
        for child in to_absorb:
            self.absorb(child)
            altered = True
                    
        return altered


    def agglomerate_compound_adj(self, options={}):
        """
        For the purpose of sense disambiguation, agglomerate compound adjectives

        e.g. If "jump" is used to describe A jumping over B, the real sense of the verb is "jump over"
        """
        verbose = False
        if verbose:  err([self])
        
        altered = False
        if self.is_dead:
            return altered
            
        # Select which children, if any, to absorb
        to_absorb = []
        if self.is_compound_prefix():
            if verbose:  err([self])
            hyphen = None
            suffix = None
            
            for node in self.get_siblings():
                if node.get_text() == '-':
                    hyphen = node

            if self.has_dep( set(['amod', 'compound']) ):
                if self.parent.has_pos( set(['NOUN']) ):
                    suffix = self.parent
            
            if  hyphen  and  suffix:
                to_absorb.append(hyphen)     # Add to the list of children to absorb
                to_absorb.append(suffix)

        # Execute absorptions in order
        for child in to_absorb:
            self.absorb(child)
            altered = True
                    
        return altered


    # end ALTERATIONS
    ############################################################################
    
    def is_root(self):
        if self.parent is None:
            return True
        return False
        
            
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False

    
    def is_child(self, node):
        """
        Is <node> one of my children?
        """
        if node in self.children:
            return True
        else:
            return False


    def is_ancestor(self, node):
        """
        Is <node> an ancestor?
        """
        focus = self
        while focus:
            if focus.parent == node:
                return True
            focus = focus.parent   # One step up the tree toward the root
            
        return False               # After reaching the root, <node> was not found


    def is_descendant(self, node):
        """
        is <node> a descendant of self?
        """
        if self.is_child(node):       # Base case
            return True

        for child in self.children:   # Recursion
            if child.is_descendant(node):
                return True

        return False   # After testing self and each child, node is not a descendant
    

    def get_descendants_at_relative_depth(self, d):
        """
        Get all descendants at a depth of precisely d, relative to self.

        Children all have relative depth=1
        """
        verbose = False
        if verbose:
            print('#'*10, d)
            print("Seeking children of", self.get_text())
                
        if d == 0:
            return [self]
        
        elif d == 1:
            if verbose:  print("\tReturning:", self.children)
            return self.children

        ddts = []
        for child in self.children:
            ddts.extend( child.get_descendants_at_relative_depth(d-1) )
            
        if verbose:
            if len(ddts) == 0:
                print("NO descendants at depth %d under: [%s]"% (d, self.get_text()))
            
        return ddts
    
    
    def get_next_token(self):
        """
        Return the token just after the last token currently held by this Node
        """
        highest = 0
        if len(self.tokens) < 1:
            return None
        
        for token in self.tokens:
            if token.i > highest:
                highest = token.i
        next_index = highest + 1
        if next_index < self.doc.get_num_tokens():
            return self.doc.token_by_index(next_index)
        else:
            return None


    def get_root(self):
        """
        Recursive climb toward the root
        """
        if self.is_root():
            return self
        return self.parent.get_root()


    def node_with_token(self, token):
        """
        Search locally and recursively down to the leaves for the Node containing <token>
        """
        if self.is_dead:
            return None
        
        if token in self.tokens:                    # Base case
            return self
        
        for child in self.children:
            node = child.node_with_token(token)     # Recursion
            if node is not None:
                return node
            
        return None
    

    def node_of_token(self, token):
        """
        Given a token, find the Node to which it currently belongs. There can only be one.
        """
        root = self.get_root()
        node = root.node_with_token(token)
        return node


    def get_text(self):
        """
        Get a string representing, in tree order, the tokens comprising this Node

        """
        texts = []
        try:
            for token in self.tokens:
                texts.append(token.text)
        except:
            pass
        return ' '.join(texts)


    def get_lemmas(self):
        """
        Get a string representing, in tree order, the tokens comprising this Node

        """
        lemmas = []
        try:
            for token in self.tokens:
                lemmas.append(token.lemma_)
        except:
            pass
        
        return lemmas


    def ner_by_token(self, token):
        ent_type, iob = self.doc.ner[token.i]
        return ent_type
    

    def iob_by_token(self, token):
        ent_type, iob = self.doc.ner[token.i]
        return iob
    

    def get_lemmas_str(self):
        return ' '.join(self.get_lemmas())


    def has_lemma(self, lemma_set):
        verbose = False
        self_lemmas = set(self.get_lemmas())
        insec = self_lemmas.intersection(lemma_set)
        if verbose:  err([self_lemmas, lemma_set, insec])
        if len(insec) > 0:
            return True
        return False

    
    def is_compound_prefix(self):
        if len(self.tokens) > 1:
            return False
        if self.has_lemma(default.get('compound_adj_prefixes')):
            return True
        return False
    

    def get_pos(self):
        """
        Get the part of speech (could be multiple)

        """
        pos = []
        try:
            for token in self.tokens:
                pos.append(token.pos_)
        except:
            pass
        return pos


    def get_pos_str(self):
        return ' '.join(self.get_pos())


    def get_all_pos(self):
        """
        Get all POS from this Node and descendants
        """
        pos = set([])
        pos.update(self.get_pos())
        for child in self.children:
            pos.update(child.get_all_pos())

        return pos
            

    def has_pos(self, pos_set):
        self_pos = set(self.get_pos())
        insec = self_pos.intersection(pos_set)

        if len(insec) > 0:
            return True
        return False

    
    def get_ner(self):
        """
        Get the part of speech (could be multiple)

        """
        ner = []
        try:
            for token in self.tokens:
                if token.ent_type > 0:
                    ner.append(token.ent_type_)
        except:
            pass
        return ner


    def get_all_ner(self):
        """
        Get all NER from this Node and descendants
        """
        ner = set([])
        ner.update(self.get_ner())
        for child in self.children:
            ner.update(child.get_all_ner())

        return ner
            

    def get_dep(self):
        """
        Get the part of speech (could be multiple)

        """
        dep = []
        try:
            for token in self.tokens:
                dep.append(token.dep_)
        except:
            pass
        return dep


    def has_dep(self, dep_set):
        self_deps = set(self.get_dep())
        insec = self_deps.intersection(dep_set)
        if len(insec) > 0:
            return True
        return False


    def get_all_dep(self):
        """
        Get all Dependencies from this Node and descendants
        """
        dep = set([])
        dep.update(self.get_dep())
        for child in self.children:
            dep.update(child.get_all_dep())

        return dep
            

    def is_verb(self):
        if 'VERB' in self.get_pos():
            return True
        return False


    def shares_pos(self, node):
        """
        Boolean.  Determine if 'node' shares at least one POS with self
        """
        s      = set(self.get_pos())
        other  = set(node.get_pos())
        insec  = s.intersection(other)
        if len(insec) > 0:
            return True
        return False
    

    def get_entity_type(self):
        """
        Return an array containing the recognized entity types purely for this one entity (No Node should contain more than one).
        Only in the case where spaCy has made an error will this array have more than one element.

        Known types
        -----------
        PERSON
        ORG
        GPE
        CARDINAL
        DATE
        TIME
        MONEY
        LAW

        Returns
        -------
        array of str

        """
        ents = []
        try:
            for token in self.tokens:
                ent_type = self.ner_by_token(token)
                if not ent_type in ents:
                    ents.append(ent_type)
        except:
            pass

        return ents


    def get_entity_position(self):
        """
        If a token is part of an entity, it is either the beginning (B) or the interior (I), which includes the last token.
        "O" refers to being outside any entity.

        Returns
        -------
        str

        """
        position = 'O'
        try:
            for token in self.tokens:
                iob = self.iob_by_token(token)
                if iob == 'B':
                    position = 'B'
                if iob == 'I'  and  position == 'O':
                    position = 'I'
        except:
            pass
        return position


    def get_entity_beginners(self):
        """
        Recursively accrue a list of all Nodes representing the beginning of an entity.

        """
        beginners = []
        if self.get_entity_position() == 'B':                   # Base case
            beginners.append(self)

        for child in self.children:
            beginners.extend( child.get_entity_beginners() )    # Recursion

        return beginners

        
    def get_dep(self):
        """
        Get the dependency relation type (could be multiple)

        """
        dep = []
        try:
            for token in self.tokens:
                dep.append(token.dep_)
        except:
            pass
        return sorted(dep)


    def get_dep_str(self):
        """
        A simple str representation of the dependency type

        """
        deps = self.get_dep()
        return ' ' .join( sorted(deps) )
    
    
    def get_supporting_tokens(self):
        """
        Find all tokens in this subtree, including this Node
        """
        if not self.done():   # perform once
            tokens = set(self.tokens)
            for child in self.children:
                tokens.update( child.get_supporting_tokens() )
            self.set('supporting_tokens', tokens)

        return self.get('supporting_tokens')
    

    def get_supporting_text(self):
        """
        Find the shortest substring in the original text such that all nodes on this subtree are represented
        """
        if not self.done():   # perform once

            left = right = None
            if len(self.tokens) < 1:
                return ''

            for token in self.get_supporting_tokens():
                if left is None or token.left_edge.i < left:
                    left = token.left_edge.i
                if right is None or token.right_edge.i > right:
                    right = token.right_edge.i

            subtree_span = self.doc.get_span(left, right+1)
            self.set('supporting_text', subtree_span.text)

        return self.get('supporting_text')

    
    def get_semantic_roles_str(self, options={}):
        """
        Like a poor man's semantic roles

        Glean what we can about semantic roles, represented in a str.  Applies best theoretically if self is a verb

        Returns
        -------
        dict
          dep_str : support_str
          Maps the dependency type string onto a substring in the text

        """
        nodes_by_type = {}
        for child in self.children:
            dep = child.get_dep_str()   # Could be a string with multiple dep types
            if dep == 'PUNCT'  and  child.is_leaf():
                continue
            if nodes_by_type.get(dep):
                nodes_by_type[dep].append(child)
            else:
                nodes_by_type[dep] = [child]

        role = {}
        for t, nodes in nodes_by_type.items():
            text = get_support_for_nodes(nodes)
            role[t] = text
                
        return role


    def get_role_vector(self, options={}):
        """
        Like a poor man's semantic role vectorization.  For the role-taking argument, not the head verb.

        Glean what we can about the semantic role of a node, represented in a vector.  Applies best theoretically if parent is a verb

        Options
        -------
        as_float : boolean

        Returns
        -------
        numpy array
            Represents the state of this Node with respect to POS, NER, and Dependencies
        """
        vector = None

        dep_str = self.get_dep_str()   # Could be a string with multiple dep types
        if dep_str == 'PUNCT'  and  self.is_leaf():
            pass

        else:
            # Get each part of the embedding
            dep = self.get_dep_embedding()
            pos = self.get_pos_embedding()
            ner = self.get_ner_embedding()
            c = np.concatenate([dep, pos, ner])

            if vector is None:
                vector = c
            else:
                vector = np.maximum.reduce([vector, c])  # binary OR: maintain sparse vector

        if vector is None:  # For cases such as leaf punctuation
            dep    = self.get_empty_dep_embedding()
            pos    = self.get_empty_pos_embedding()
            ner    = self.get_empty_ner_embedding()
            vector = np.concatenate([dep, pos, ner])

        if options.get('as_float'):
            vector = vector.astype(float).tolist()
        
        return vector


    def get_pos_embedding(self, vocab=default.get('pos_embedding'), options={}):
        """
        Return a vectorized representation of the POS of this Node
        """
        pos_vector = None
        for pos in self.get_pos():
            if pos_vector is None:
                pos_vector = vocab[pos]
            else:
                pos_vector = np.maximum.reduce([ pos_vector, vocab[pos] ])  # maintain one-hot vector
                
        if pos_vector is None:
            pos_vector = vocab['_empty_']
            
        return pos_vector
    

    def get_empty_pos_embedding(self, vocab=default.get('pos_embedding'), options={}):
        return vocab['_empty_']
    

    def get_ner_embedding(self, vocab=default.get('ner_embedding'), options={}):
        """
        Return a vectorized representation of the NER of this Node
        """
        ner_vector = None
        for ner in self.get_ner():
            if ner_vector is None:
                ner_vector = vocab[ner]
            else:
                ner_vector = np.maximum.reduce([ ner_vector, vocab[ner] ])  # maintain one-hot vector
                
        if ner_vector is None:
            ner_vector = vocab['_empty_']
            
        return ner_vector
    

    def get_empty_ner_embedding(self, vocab=default.get('ner_embedding'), options={}):
        return vocab['_empty_']
    

    def get_dep_embedding(self, vocab=default.get('dep_embedding'), options={}):
        """
        Return a vectorized representation of the DEP of this Node
        """
        dep_vector = None
        for dep in self.get_dep():
            if dep_vector is None:
                dep_vector = vocab[dep]
            else:
                dep_vector = np.maximum.reduce([ dep_vector, vocab[dep] ])  # maintain one-hot vector
                
        if dep_vector is None:
            dep_vector = vocab['_empty_']
            
        return dep_vector
    

    def get_empty_dep_embedding(self, vocab=default.get('dep_embedding'), options={}):
        return vocab['_empty_']
    

    def get_verb_nodes(self):
        """
        From each of the constituent trees, return a list of all nodes that are verbs

        """
        verbs = []
        if self.is_verb():
            verbs = [self]
        
        for child in self.children:
            verbs.extend( child.get_verb_nodes() )
        return verbs


    def get_compound_prefix_nodes(self):
        """
        From each of the constituent trees, return a list of all nodes that are compound prefixes
        """
        nodes = []
        if self.is_compound_prefix():
            nodes = [self]
        
        for child in self.children:
            nodes.extend( child.get_compound_prefix_nodes() )
        return nodes


    def get_index(self):
        """
        Return lowest token index
        """
        lowest = None
        for token in self.tokens:
            if lowest is None  or  token.i < lowest:
                lowest = token.i
        return lowest
        
    
    def embed(self, vocab):
        """
        Given some vocab (embedding) produce a vector that represents this node

        """
        verbose = False
        found_embedding = False

        # First try full lemma string
        lemmas_str = self.get_lemmas_str()
        lemmas_str = re.sub(r' - ', '-', lemmas_str)
        lemmas_str = re.sub(r' ', '_', lemmas_str)
        lemmas_str = re.sub(r'_+', '_', lemmas_str)
        vec = vocab.get(lemmas_str)
        
        if vec is not None:
            if verbose: print("Found vector for: %s"% lemmas_str)
            self.embedding = vec
            found_embedding = True

        # Remove hyphens
        if not found_embedding:
            if re.search(r'-', lemmas_str):
                lemmas_str = re.sub(r'-', '', lemmas_str)
                lemmas_str = re.sub(r'_+', '_', lemmas_str)
                vec = vocab.get(lemmas_str)
            
            if vec is not None:
                if verbose: print("Found vector for: %s"% lemmas_str)
                self.embedding = vec
                found_embedding = True

        # Remove underscores
        if not found_embedding:
            if re.search(r'_', lemmas_str):
                lemmas_str = re.sub(r'_', '', lemmas_str)
                vec = vocab.get(lemmas_str)
            
            if vec is not None:
                if verbose: print("Found vector for: %s"% lemmas_str)
                self.embedding = vec
                found_embedding = True

        # Use an averaging of as many vectors as available            
        if not found_embedding:
            if len(self.get_lemmas()) > 1:
                if verbose: print("Found nothing for: %s ... Trying an average ..."% lemmas_str)
                vecs = []
                for lemma in self.get_lemmas():
                    vec = vocab.get(lemma)
                    if vec is not None:
                        if verbose: print("Found sub-vector for: %s"% lemma)
                        vecs.append(vec)
                    else:
                        if verbose: print("Found nothing for: %s"% lemma)
                if len(vecs) > 1:
                    vec = vector_average(vecs)
                elif len(vecs) == 1:
                    vec = vecs[0]
                self.embedding = vec

        # No embedding found
        if not found_embedding:
            self.embedding = None
            
        # Recursive application
        if self.children is not None:
            for child in self.children:
                child.embed(vocab)
            

    def get_nodes(self):
        """
        Returns a self-inclusive list of this Node and all descendants
        """
        nodes = set([self])
        for node in self.children:
            nodes.update(node.get_nodes())

        return sorted(nodes, key=lambda x: x.get_index())


    def get_depth(self):
        """
        Number of links to root
        """
        if self.is_root():
            return 0
        else:
            return 1 + self.parent.get_depth()


    def get_siblings(self):
        """
        Get sibling nodes
        """
        siblings = []
        if self.is_root():
            return siblings
        
        for node in self.parent.children:
            if node == self:
                continue
            siblings.append(node)
            
        return siblings


    def get_num_siblings(self):
        """
        Number of parent's children minus one
        """
        if self.is_root():
            return 0
        
        num = len(self.parent.children)
        return num - 1


    def get_idx(self):
        """
        Find the start/end character offset of the aggregate of tokens represented by this node.  Those tokens in many but not all
        cases will be contiguous in the text.
        """
        idxs = []
        for token in self.tokens:
            idxs.append(token.idx)
            end = token.idx + len(token.text)
            idxs.append(end)
        idxs.sort()
        
        return (idxs[0], idxs[-1])
        

    def overlap_idx(self, idx):
        """
        Determine if this nodes's idx (character start/end offsets) overlaps the given idx

        Parameters
        ----------
        idx : (int, int)

        Returns
        -------
        boolean

        """
        sidx = self.get_idx()
        bot  = max( idx[0], sidx[0] )
        top  = min( idx[1], sidx[1] )
        spr  = top - bot
        if spr > 0:
            return True
        return False


    def cosine_similarity(self, node):
        """
        The similarity between the embedding of this and some other node
        """
        return cosine_similarity(self.embedding, node.embedding)
    
        
    ############################################################################
    # Printing

    def print_semantic_roles(self, options={}):
        """
        Print what we know about the semantic roles

        """
        roles = self.get_semantic_roles(options=options)
        if len(roles) == 0:
            return
        
        print("\nSemantic Roles:")
        print(self.get_lemmas_str())
        for dep, nodes in roles.items():
            print('    {%s}'% dep, get_support_for_nodes(nodes))
        print()
    
            
    def pretty_print(self, depth=0, options={}):
        """
        Print out the tree recursively from this point

        """
        indent = depth * '    '
        print(indent + self.get_text() + ' {%s}'% self.get_dep_str() + ' (%s)'% self.get_pos_str())

        # Options
        if options.get('supporting_text'):  # Print the text supporting subtree
            print(indent + '[ST]: ' + self.get_supporting_text())
        
        # Recursion
        for child in self.children:
            child.pretty_print(depth + 1, options=options)

            

################################################################################
# FUNCTIONS

def iprint(title, X):
    first = True
    for x in X:
        if first:
            print(' - ' + title)
            first = False
        print('\t', x)
        

def get_support_for_tokens(doc, tokens):
    """
    Give a doc and some tokens, find the shortest contiguous substring containing all the tokens

    """
    if len(tokens) == 0:
        return ''
    
    left = right = None
    for token in tokens:
        if left is None or token.left_edge.i < left:
            left = token.left_edge.i
        if right is None or token.right_edge.i > right:
            right = token.right_edge.i

    subtree_span = doc.get_span(left, right+1)
    return subtree_span.text
    

def get_support_for_nodes(nodes):
    """
    For some set of nodes (from single Document), find the shortest contiguous substring containing all constituents tokens

    """
    if len(nodes) == 0:
        return ''
    
    left = right = None
    doc = nodes[0].doc
    for node in nodes:
        if len(node.tokens) < 1:
            return ''
        for token in node.tokens:
            if left is None or token.left_edge.i < left:
                left = token.left_edge.i
            if right is None or token.right_edge.i > right:
                right = token.right_edge.i

    subtree_span = doc.get_span(left, right+1)
    return subtree_span.text
    

def combined_lemmas_in_vocab(vocab, lemmas_str_A, lemmas_str_B):
    """
    Look for a combination of these two lemmas strings such that vocab has an entry

    """
    lemmas_str = lemmas_str_A +'_'+ lemmas_str_B
    lemmas_str = re.sub(r' ', '_', lemmas_str)
    if vocab.get(lemmas_str) is not None:
        return True
    
    lemmas_str = lemmas_str_B +'_'+ lemmas_str_A
    lemmas_str = re.sub(r' ', '_', lemmas_str)
    if vocab.get(lemmas_str) is not None:
        return True

    return False


def get_group_ancestor(nodes):
    """
    For a given set of nodes (from same Document), find the lowest ancestor of all nodes.  It may be a member of 'nodes'

    Parameters
    ----------
    nodes : set of Node

    Returns
    -------
    Node
    """
    nodes_copy = []
    nodes_copy.extend(nodes)
    sorted_nodes  = sorted(nodes, key=lambda x: x.get_depth())
    first_node    = None
    
    for i in sorted_nodes:
        if first_node is None:
            first_node = i
        if i.is_root():
            return i
        
        is_the_ancestor = True
        for j in nodes_copy:
            if i != j:
                if not i.is_descendant(j):
                    is_the_ancestor = False

        if is_the_ancestor:
            return i

    # No ancestor was found. Add a parent an recurse
    nodes.add(first_node.parent)
    return get_group_ancestor(nodes)


################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Tools to handle dependency parses: node.py"})
    args = parser.parse_args()   # Get inputs and options

    if args.str:
        pass
    else:
        print(__doc__)

        
################################################################################
################################################################################
    
