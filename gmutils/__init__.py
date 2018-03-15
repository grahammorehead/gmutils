
from .utils import err, argparser, argparser_ml, serialize, deserialize, set_missing_attributes, isTrue, read_file, monitor_setup, monitor, read_conceptnet_vectorfile, cosine_similarity, binary_distance

from .normalize import normalize, ascii_fold, simplify_for_distance

from .elastic_utils import list_indices, index_dicts, index_dict, index_text_with_synonyms, match_search, prefix_search, wildcard_search, synonym_search

try:
    from .kinesis_utils import KinesisStream
except Exception as e: err([], {'exception':e, 'level':0})

try:
    from .mongo_utils import mongo_iterator, mongo_find_one, mongo_count
except Exception as e: err([], {'exception':e, 'level':0})

from .nlp import generate_spacy_data, spacy_ner, spacy_parsing

from .dataset import Dataset

from .document import Document

from .node import get_group_ancestor

from .model import Model

from .sklearn_model import SklearnModel

from .objects import Object, Options

from .lexical import damerauLevenshtein
