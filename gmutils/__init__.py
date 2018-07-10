import sys, os
verbose = False

if verbose:  sys.stderr.write("\tLoading utils ...\n")
from .utils import err, argparser, argparser_ml, serialize, deserialize, set_missing_attributes, isTrue, read_file, iter_file, read_dir, generate_file_iterator, monitor_setup, monitor, read_conceptnet_vectorfile, cosine_similarity, binary_distance, mkdirs, json_dump_gz, json_load_gz, deepcopy_list, deepcopy_dict, file_exists, dir_exists, file_timestamp, concat_from_list_of_dicts, binary_F1, iter_next, args_to_options, num_pos_elements, num_lines_in_file

if verbose:  sys.stderr.write("\tLoading normalize ...\n")
from .normalize import normalize, ascii_fold, simplify_for_distance, naked_words, clean_spaces

try:
    if verbose:  sys.stderr.write("\tLoading Elasticsearch ...\n")
    from .elastic_utils import list_indices, index_dicts, index_dict, index_text_with_synonyms, match_all, match_search, prefix_search, wildcard_search, synonym_search, constant_score_search, phrase_search
except Exception as e: err([], {'exception':e, 'level':0})

try:
    from .kinesis_utils import KinesisStream
except Exception as e: err([], {'exception':e, 'level':0})

try:
    from .mongo_utils import mongo_iterator, mongo_find_one, mongo_count
except Exception as e: err([], {'exception':e, 'level':0})

try:
    if not os.environ.get('GM_NO_NLP'):
        if verbose:  sys.stderr.write("\tLoading spacy ...")
        from .nlp import generate_spacy_data, spacy_ner, spacy_parsing, words_below_freq, get_synonyms, lemmatize
        
        if verbose:  sys.stderr.write("\tLoading Document ...\n")
        from .document import Document

        if verbose:  sys.stderr.write("\tLoading Node ...\n")
        from .node import get_group_ancestor
except Exception as e: err([], {'exception':e, 'level':0})

if verbose:  sys.stderr.write("\tLoading Dataset ...\n")
from .dataset import Dataset

if verbose:  sys.stderr.write("\tLoading Model ...\n")
from .model import Model
try:
    from .vectorizer import Vectorizer
except Exception as e: err([], {'exception':e, 'level':0})

if verbose:  sys.stderr.write("\tLoading SklearnModel ...\n")
from .sklearn_model import SklearnModel

from .objects import Object, Options

try:
    from .lexical import damerauLevenshtein, phrase_similarity
except Exception as e: err([], {'exception':e, 'level':0})

try:
    if verbose:  sys.stderr.write("\tLoading TensorFlow ...\n")
    from .tensorflow_layer import TensorflowLayer
    from .tensorflow_graph import TensorflowGraph
    from .tensorflow_model import TensorflowModel
except Exception as e: err([], {'exception':e, 'level':0})

try:
    from .mysql_utils import mysql_connect, mysql_query
except Exception as e: err([], {'exception':e, 'level':0})

