import re

import numpy as np
import nltk
import unicodedata
from nltk.tag import CRFTagger, BrillTagger, BrillTaggerTrainer, RegexpTagger
from nltk.corpus import treebank

train_sentences = treebank.tagged_sents()[:3000]
test_sentences = treebank.tagged_sents()[3000:]

"""
Question 1: Redefine this function so that it considers the context
"""
def feature_func(tokens, idx):
    """
            Extract basic features about this word including
                 - Current Word
                 - Is Capitalized ?
                 - Has Punctuation ?
                 - Has Number ?
                 - Suffixes up to length 3
            Note that : we might include feature over previous word, next word ect.

            :return : a list which contains the features
            :rtype : list(str)

            """

    pattern = re.compile(r"\d")
    token = tokens[idx]

    feature_list = []

    if not token:
        return feature_list

    # Capitalization
    if token[0].isupper():
        feature_list.append("CAPITALIZATION")

    # Number
    if re.search(pattern, token) is not None:
        feature_list.append("HAS_NUM")

    # Punctuation
    punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
    if all(unicodedata.category(x) in punc_cat for x in token):
        feature_list.append("PUNCTUATION")

    # Suffix up to length 3
    if len(token) > 1:
        feature_list.append("SUF_" + token[-1:])
    if len(token) > 2:
        feature_list.append("SUF_" + token[-2:])
    if len(token) > 3:
        feature_list.append("SUF_" + token[-3:])

    feature_list.append("WORD_" + token)

    return feature_list

def question1():
    tagger = CRFTagger(feature_func=feature_func)

    tagger.train(train_sentences, 'model.crf.tagger')

    print(tagger.evaluate(train_sentences))
    return

"""
Question 2: Train using BrillTagger
"""
def question2():
    #Taken from http://www.nltk.org/book/ch05.html
    patterns = [
        (r'.*ing$', 'VBG'),  # gerunds
        (r'.*ed$', 'VBD'),  # simple past
        (r'.*es$', 'VBZ'),  # 3rd singular present
        (r'.*ould$', 'MD'),  # modals
        (r'.*\'s$', 'NN$'),  # possessive nouns
        (r'.*s$', 'NNS'),  # plural nouns
        (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')  # nouns (default)
        ]

    train_words = treebank.words()
    init_tagger = RegexpTagger(patterns)

    #Not sure if we need to use BrillTagger or BrillTaggerTrainer??
    #tagger = BrillTagger(init_tagger)
    # tagger = BrillTaggerTrainer(init_tagger)
    return

if __name__ == "__main__":
    question1()
