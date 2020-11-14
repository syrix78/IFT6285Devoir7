import csv
import re

import numpy as np
import nltk
import unicodedata
from nltk.tag import CRFTagger, BrillTagger, untag, BrillTaggerTrainer, RegexpTagger
from nltk.corpus import treebank
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word


train_sentences = treebank.tagged_sents()[:3000]
test_sentences = treebank.tagged_sents()[3000:]

"""
Question 3: Redefine this function so that it considers the context
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

def question3():
    tagger = CRFTagger(feature_func=feature_func)

    tagger.train(train_sentences, 'model.crf.tagger')

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

    #init_tagger = CRFTagger(feature_func=feature_func)
    #init_tagger.train(train_sentences, 'model.crf.tagger')

    init_tagger = RegexpTagger(patterns)

    nb_iterations = 31
    currentTagger = None
    current_evaluation = 0.0
    evaluations = []

    for i in range(1, nb_iterations):
        # https://www.nltk.org/api/nltk.tag.html#module-nltk.tag.brill_trainer
        Template._cleartemplates()
        templates = [Template(Pos([-1])), Template(Pos([-1]), Word([0]))]

        tt = BrillTaggerTrainer(init_tagger, templates, trace=3)
        currentTagger = tt.train(train_sentences, max_rules=i*50)
        current_evaluation = currentTagger.evaluate(test_sentences)
        evaluations.append(current_evaluation)

    """
    for i in range(nb_iterations):
        #Not sure if we need to use BrillTagger or BrillTaggerTrainer??
        #https://www.nltk.org/api/nltk.tag.html#module-nltk.tag.brill_trainer
        Template._cleartemplates()
        templates = [Template(Pos([-1])), Template(Pos([-1]), Word([0]))]

        if i == 0:
            tt = BrillTaggerTrainer(init_tagger, templates, trace=3)
            currentTagger = tt.train(train_sentences)
            current_evaluation = currentTagger.evaluate(test_sentences)
            evaluations.append(current_evaluation)

        else:
            tt = BrillTaggerTrainer(currentTagger, templates, trace=3)
            tagger = tt.train(train_sentences)
            current_evaluation = tagger.evaluate(test_sentences)
            evaluations.append(current_evaluation)
            currentTagger = tagger
    """
    print(current_evaluation)
    return evaluations

if __name__ == "__main__":
    #question3()
    question4_evals = question4()

    with open('question4_meta_max_rules_one.csv', mode='w+') as file:
        employee_writer = csv.writer(file, delimiter=',')

        employee_writer.writerow(list(range(50, len(question4_evals)*50 + 1, 50)))
        employee_writer.writerow(question4_evals)

