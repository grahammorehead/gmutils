""" node.py

Tools to manage nodes in a parse tree

"""
import os, sys, re, json

from gmutils.objects import Object
from gmutils.utils import err, argparser, vector_average

################################################################################
# DEFAULTS

default = {
    'sr_embedding' : {
        '_empty_' : [0, 0, 0, 0, 0]
        }
    }

pos_indices = ['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SYM', 'VERB', 'X']

ner_indices = []

dep_indices = []
    
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
        self.is_dead = True
        self.tokens = self.children = self.embedding = None
        
                
    def __repr__(self):
        return self.get_text()

        
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
        is <node> a descendant?
        """
        if self.is_child(node):       # Base case
            return True

        for child in self.children:   # Recursion
            if child.is_descendant(node):
                return True

        return False   # After testing self and each child, node is not a descendant
    
    
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
        Given a token, find the Node to which it currently belongs
        """
        root = self.get_root()
        node = root.node_with_token(token)
        return node

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
        
        elif self.parent.is_root():                   # parent is root
            self.tokens.extend(self.parent.tokens)    # Absorb node's tokens
            self.adopt(self.parent.children)          # Adopt node's children (ignores self, of course)
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
                if 'prep' in child.get_deps():  # Only looking to agglomerate nodes with a 'prep' dependency
                    if vocab is not None:       # Only allow agglomerations corresponding to a vocab entry
                        if not combined_lemmas_in_vocab(vocab, self.get_lemmas_str(), child.get_lemmas_str()):
                            continue
                    to_absorb.append(child)     # Add to the list of children to absorb

        # Execute absorptions in order, re-confirming vocab presence along the way
        for child in to_absorb:
            if combined_lemmas_in_vocab(vocab, self.get_lemmas_str(), child.get_lemmas_str()):
                self.absorb(child)
                altered = True
                    
        # Apply recursively to children
        for child in self.children:
            child.agglomerate_verbs_preps(vocab=vocab)
            
        return altered

    # end ALTERATIONS
    ############################################################################
    
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


    def get_pos_num(self):
        """
        Get the part of speech (could be multiple)

        """
        pos = []
        try:
            for token in self.tokens:
                pos.append(token.pos)
        except:
            pass
        return pos


    def get_all_pos(self):
        """
        Get all POS from this Node and descendants
        """
        pos = set([])
        pos.update(self.get_pos())
        for child in self.children:
            pos.update(child.get_all_pos())

        return pos
            

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


    def get_ner_num(self):
        """
        Get the part of speech (could be multiple)

        """
        ner = []
        try:
            for token in self.tokens:
                if token.ent_type > 0:
                    ner.append(token.ent_type)
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
            

    def get_deps(self):
        """
        Get the part of speech (could be multiple)

        """
        deps = []
        try:
            for token in self.tokens:
                deps.append(token.dep_)
        except:
            pass
        return deps


    def get_deps_num(self):
        """
        Get the part of speech (could be multiple)

        """
        deps = []
        try:
            for token in self.tokens:
                deps.append(token.dep)
        except:
            pass
        return deps


    def get_all_deps(self):
        """
        Get all Dependencies from this Node and descendants
        """
        deps = set([])
        deps.update(self.get_deps())
        for child in self.children:
            deps.update(child.get_all_deps())

        return deps
            

    def is_verb(self):
        if 'VERB' in self.get_pos():
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

        
    def get_deps(self):
        """
        Get the dependency relation type (could be multiple)

        """
        deps = []
        try:
            for token in self.tokens:
                deps.append(token.dep_)
        except:
            pass
        return sorted(deps)


    def get_deps_str(self):
        """
        A simple str representation of the dependency type

        """
        return ' ' .join(self.get_deps())
    
    
    def get_supporting_text(self):
        """
        Find the shortest substring in the original text such that all nodes on this subtree are represented

        """
        left = right = None
        if len(self.tokens) < 1:
            return ''
        
        for token in self.tokens:
            if left is None or token.left_edge.i < left:
                left = token.left_edge.i
            if right is None or token.right_edge.i > right:
                right = token.right_edge.i
                
        subtree_span = self.doc.get_span(left, right+1)
        return subtree_span.text
    

    def get_semantic_roles(self, options={}):
        """
        Glean what we can about semantic roles.  Especially applicable if self is_verb

        Returns
        -------
        dict
          dep_type : list of spacy.Token
          Maps the dependency type onto an array of Tokens and therefore a substring

        """
        roles = {}
        for child in self.children:
            deps = child.get_deps_str()

            if deps == 'punct'  and  child.is_leaf():
                continue
            
            if deps in roles:
                roles[deps].append(child)
            else:
                roles[deps] = [child]

        return roles


    def get_embedded_semantic_roles(self, vocab=default['sr_embedding'], options={}):
        """
        Get a vectorized representation of this Node's semantic roles
        """
        # Begin with self.  Self's SR along with POS
        self_dep = vocab.get('_empty_')
        for dep in self.get_deps():
            self_vector
            
        roles = self.get_semantic_roles(options=options)
        if len(roles) == 0:
            return vocab.get('_empty_')
        
        for dep, nodes in roles.items():
            print('    {%s}'% dep, get_support_for_nodes(nodes))   # An child having a role under self

        
    
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

        # First try full lemma string
        lemmas_str = self.get_lemmas_str()
        lemmas_str = re.sub(r' ', '_', lemmas_str)
        vec = vocab.get(lemmas_str)
        if vec is not None:
            if verbose: print("Found vector for: %s"% lemmas_str)
            self.embedding = vec
            
        elif len(self.get_lemmas()) > 1:   # Use an averaging of as many vectors as available
            if verbose: print("Found nothing for: %s"% lemmas_str)
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

        # Recursive application
        if self.children is not None:
            for child in self.children:
                child.embed(vocab)
            

    def get_nodes(self):
        """
        Returns a self-inclusive list of this Node and all descendants
        """
        nodes = set([])
        for node in self.children:
            nodes.update(node.get_nodes())

        return sorted(nodes, key=lambda x: x.get_index())
    
                
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
        print(indent + self.get_text() + ' {%s}'% self.get_deps_str())

        # Options
        if options.get('supporting text'):  # Print the text supporting subtree
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
    
