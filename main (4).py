import re

import numpy as np
import nltk
import unicodedata
from nltk.tag import CRFTagger, BrillTagger, BrillTaggerTrainer, RegexpTagger
from nltk.corpus import treebank


train_sentences = treebank.tagged_sents()[:3000]
test_sentences = treebank.tagged_sents()[3000:]


"""
Question 3: Redefine this function so that it considers the context
"""
def feature_func(tokens, idx, window_size=1):
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

    actual_word_idx = idx

    idx -= window_size # Pour se rendre au debut de la fenetre

    feature_list = []

    indicator = ""

    before = True
    after = False
    features = False
    using_word = True

    for i in range(window_size*2 + 1):

        if(idx < actual_word_idx):
            pos = actual_word_idx - idx
            indicator = "PRE_" + str(pos) + "_"
        elif(idx == actual_word_idx):
            indicator = ""
        else:
            pos = idx - actual_word_idx
            indicator = "POST_" + str(pos) + "_"

        if(idx < 0):
            idx += 1

        elif(idx >= len(tokens)):
            break

        elif(idx < actual_word_idx and after):
            idx += 1

        elif(idx > actual_word_idx and before):
            break

        else:

            token = tokens[idx]

            if not token:
                return feature_list

            if (idx == actual_word_idx or features):
                # Capitalization
                if token[0].isupper():
                    feature_list.append(indicator + "CAPITALIZATION")

                # Number
                if re.search(pattern, token) is not None:
                    feature_list.append(indicator + "HAS_NUM")

                # Punctuation
                punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
                if all(unicodedata.category(x) in punc_cat for x in token):
                    feature_list.append(indicator + "PUNCTUATION")

                # Suffix up to length 3
                if len(token) > 1:
                    feature_list.append(indicator + "SUF_" + token[-1:])
                if len(token) > 2:
                    feature_list.append(indicator + "SUF_" + token[-2:])
                if len(token) > 3:
                    feature_list.append(indicator + "SUF_" + token[-3:])

            if (idx == actual_word_idx):
                feature_list.append("WORD_" + token)
            elif (using_word):
                feature_list.append(indicator + "WORD_" + token)
                feature_list.append(indicator + "WORD_" + token + "/" + tokens[actual_word_idx])

            idx += 1

    return feature_list

def question3():

    tagger = CRFTagger(feature_func=feature_func)
    tagger.train(train_sentences, 'model_windows_size_1.crf.tagger')

    #tagger = CRFTagger(feature_func=feature_func)
    #tagger.set_model_file('model_windows_size_1.crf.tagger')

    print(tagger.evaluate(test_sentences))
    return

"""
Question 4: Train using BrillTagger
"""
def question4():
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
    question3()

