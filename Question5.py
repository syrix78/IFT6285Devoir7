import re

import numpy as np
import nltk
import unicodedata
from nltk.tag import CRFTagger, BrillTagger, BrillTaggerTrainer, RegexpTagger, TaggerI
from nltk.corpus import treebank
import spacy

class Question5(TaggerI):
    pass