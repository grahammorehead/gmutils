import sys, os
verbose = False

if verbose:  sys.stderr.write("\tLoading utils ...\n")
from .utils import err, argparser, argparser_ml, serialize, deserialize, set_missing_attributes, isTrue, read_file, read_dir, monitor_setup, monitor, read_conceptnet_vectorfile, cosine_similarity, binary_distance, mkdirs, json_dump_gz, json_load_gz, deepcopy_list, file_exists

if verbose:  sys.stderr.write("\tLoading normalize ...\n")
from .normalize import normalize, ascii_fold, simplify_for_distance

try:
    if verbose:  sys.stderr.write("\tLoading Elasticsearch ...\n")
    from .elastic_utils import list_indices, index_dicts, index_dict, index_text_with_synonyms, match_search, prefix_search, wildcard_search, synonym_search
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
        from .nlp import generate_spacy_data, spacy_ner, spacy_parsing
        
        if verbose:  sys.stderr.write("\tLoading Document ...\n")
        from .document import Document

        if verbose:  sys.stderr.write("\tLoading Node ...\n")
        from .node import get_group_ancestor
except Exception as e: err([], {'exception':e, 'level':0})

if verbose:  sys.stderr.write("\tLoading Dataset ...\n")
from .dataset import Dataset

if verbose:  sys.stderr.write("\tLoading Model ...\n")
from .model import Model

if verbose:  sys.stderr.write("\tLoading SklearnModel ...\n")
from .sklearn_model import SklearnModel

from .objects import Object, Options

from .lexical import damerauLevenshtein

try:
    if verbose:  sys.stderr.write("\tLoading TensorFlow ...\n")
    from .tensorflow_graph import TensorflowGraph
    from .tensorflow_model import TensorflowModel
except Exception as e:
    err([], {'exception':e, 'level':0})
